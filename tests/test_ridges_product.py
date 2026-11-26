"""One ridges product per classifiable state, built from saved products only."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis.ridges import Ridges
from dubanevics2026topological.build import CONTEXT_AWARE_COMMANDS, ridges_product
from dubanevics2026topological.results import RunContext, parse_label

REPO_ROOT = Path(__file__).resolve().parents[1]
STATES = {
    "full": ("graph", {"graph": "full", "tree": "nominal"}),
    "published": ("thr", {"thr": "published", "fdr": "none", "tree": "nominal"}),
    "published_by": (
        "by",
        {"thr": "published", "fdr": "by", "q": 0.05, "tree": "nominal"},
    ),
    "published_by_band": (
        "by",
        {
            "thr": "published",
            "fdr": "by",
            "q": 0.05,
            "draws": 10000,
            "seed": 20261019,
            "tree": "band",
            "bandq": 0.95,
        },
    ),
}
SNAPSHOT = "f" * 64
SUMMITS = {"kem": "AAGGAAATG", "tcl": "ACCTGTCTT"}
VERTICES = {
    "AAGGAAATG": 9.0,
    "ACCTGTCTT": 8.0,
    "CTCTGCTTG": 4.0,
    "AAAAAAAAA": 1.0,
}


def _product(kind, label, tmp_path, identity=None, **files):
    directory = tmp_path / kind / label
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, content in files.items():
        path = directory / name
        if isinstance(content, pl.DataFrame):
            content.write_parquet(path)
        else:
            path.write_text(json.dumps(content))
        artifacts[name] = path
    return SimpleNamespace(
        product_type=kind,
        label=label,
        artifacts=artifacts,
        manifest=SimpleNamespace(
            scientific_id_sha256=hashlib.sha256(f"{kind}:{label}".encode()).hexdigest(),
            scientific_identity=identity or {},
        ),
    )


def _tree(tmp_path, label, identity):
    return _product(
        "tree",
        label,
        tmp_path,
        identity,
        **{
            "critical_nodes.parquet": pl.DataFrame(
                {
                    "tree_node_identity": [
                        "AAGGAAATG",
                        "ACCTGTCTT",
                        "CTCTGCTTG",
                        "AAAAAAAAA",
                    ],
                    "scalar": [9.0, 8.0, 4.0, 1.0],
                }
            ),
            "branches.parquet": pl.DataFrame(
                {
                    "source_tree_node_identity": [
                        "AAGGAAATG",
                        "ACCTGTCTT",
                        "CTCTGCTTG",
                    ],
                    "target_tree_node_identity": [
                        "CTCTGCTTG",
                        "CTCTGCTTG",
                        "AAAAAAAAA",
                    ],
                }
            ),
            "branch_members.parquet": pl.DataFrame(
                schema={
                    "source_tree_node_identity": pl.String,
                    "target_tree_node_identity": pl.String,
                    "dna_sequence": pl.String,
                }
            ),
        },
    )


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    context = RunContext(root=tmp_path, output_root=tmp_path / "out", env={})
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "analysis.toml").write_text(
        '[codon_entropy]\nkem_summit_dna_sequence = "AAGGAAATG"\n'
        'tcl_summit_dna_sequence = "ACCTGTCTT"\n'
    )
    trees = {
        state: _tree(tmp_path, f"tree-{state}", identity)
        for state, (_, identity) in STATES.items()
    }
    leaves = {
        state: _product(
            "leaves",
            f"leaves-{state}",
            tmp_path,
            **{
                "leaves.parquet": pl.DataFrame(
                    {
                        "peak_dna_sequence": ["AAGGAAATG", "ACCTGTCTT"],
                        "member_count": [1, 1],
                    }
                )
            },
        )
        for state in STATES
    }
    landscape = _product(
        "landscape",
        "landscape",
        tmp_path,
        **{
            "vertices.parquet": pl.DataFrame(
                {
                    "dna_sequence": list(VERTICES),
                    "raw_fitness": list(VERTICES.values()),
                }
            ),
            "diagnostics.json": {"threshold": 5.0},
        },
    )
    by_leaf = {leaf.label: state for state, leaf in leaves.items()}
    by_tree = {tree.label: state for state, tree in trees.items()}
    monkeypatch.setattr(
        ridges_product, "_verified", lambda loader, kind, label: leaves[label[7:]]
    )
    monkeypatch.setattr(
        ridges_product,
        "_parent",
        lambda loader, child, kind, state: (
            trees[by_leaf[child.label]]
            if kind == "tree" and child.product_type == "leaves"
            else trees["published_by"]
        ),
    )
    monkeypatch.setattr(ridges_product, "_landscape_for", lambda *a: landscape)
    monkeypatch.setattr(
        ridges_product,
        "_classes",
        lambda *a: (
            {
                "AAGGAAATG": "functional",
                "ACCTGTCTT": "uncertain",
                "CTCTGCTTG": "non_functional",
                "AAAAAAAAA": "non_functional",
            },
            SNAPSHOT,
        ),
    )
    del by_tree
    return context, {state: f"leaves-{state}" for state in STATES}


def test_command_is_context_aware():
    assert "ridges-product" in CONTEXT_AWARE_COMMANDS


def test_publishes_ridges_for_classifiable_states_only(stubbed):
    context, labels = stubbed
    paths = ridges_product.build_ridges_products(context, labels=labels)
    assert set(paths) == {"published", "published_by", "published_by_band"}
    for path in paths.values():
        assert parse_label(path.name).product_name == "ridges"
        ridges = Ridges.from_artifacts({p.name: p for p in path.iterdir()})
        assert ridges.summary["ridge_name"].to_list() == ["kem", "tcl"]
        assert (
            ridges.summary["common_saddle_dna_sequence"].to_list() == ["CTCTGCTTG"] * 2
        )
        assert ridges.members["ridge_name"].null_count() == 2
        hills = ridges.sidecar["leaf_hills"]
        assert [h["top_label"] for h in hills] == ["top_1", "top_2"]
        assert [h["ridge_name"] for h in hills] == ["kem", "tcl"]


def test_point_estimate_state_classes_use_the_threshold(stubbed):
    context, labels = stubbed
    paths = ridges_product.build_ridges_products(context, labels=labels)
    members = pl.read_parquet(paths["published"] / "members.parquet")
    classes = dict(
        zip(members["member_dna_sequence"], members["functional_class"], strict=True)
    )
    assert classes["ACCTGTCTT"] == "functional"
    assert "uncertain" not in set(classes.values())
    by = pl.read_parquet(paths["published_by"] / "members.parquet")
    assert "uncertain" in set(by["functional_class"])


def test_justfile_recipe_exists():
    text = (REPO_ROOT / "justfile").read_text(encoding="utf-8")
    assert re.search(r"^ridges-product:", text, re.MULTILINE)
