"""One merge_flow product per state, built from saved products only."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis.merge_flow import MergeFlow
from dubanevics2026topological.build import CONTEXT_AWARE_COMMANDS, merge_flow_product
from dubanevics2026topological.results import RunContext, parse_label

REPO_ROOT = Path(__file__).resolve().parents[1]
STATES = {
    "full": {"graph": "full", "tree": "nominal"},
    "published": {"thr": "published", "fdr": "none", "tree": "nominal"},
    "published_by": {"thr": "published", "fdr": "by", "q": 0.05, "tree": "nominal"},
    "published_by_band": {
        "thr": "published",
        "fdr": "by",
        "q": 0.05,
        "draws": 10000,
        "seed": 20261019,
        "tree": "band",
        "bandq": 0.95,
    },
}
SNAPSHOT = "f" * 64
VERTICES = {"AAA": 9.0, "CCC": 7.0, "GGG": 5.0, "TTT": 1.0}


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
                    "tree_node_identity": list(VERTICES),
                    "scalar": list(VERTICES.values()),
                }
            ),
            "branches.parquet": pl.DataFrame(
                {
                    "source_tree_node_identity": ["AAA", "CCC", "GGG"],
                    "target_tree_node_identity": ["GGG", "GGG", "TTT"],
                }
            ),
            "branch_members.parquet": pl.DataFrame(
                schema={
                    "source_tree_node_identity": pl.String,
                    "target_tree_node_identity": pl.String,
                    "dna_sequence": pl.String,
                }
            ),
            "persistence_pairs.parquet": pl.DataFrame(
                {"birth_sequence": ["CCC", "AAA"], "death_sequence": ["GGG", "TTT"]}
            ),
        },
    )


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    context = RunContext(root=tmp_path, output_root=tmp_path / "out", env={})
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "analysis.toml").write_text(
        '[merge_flow]\nmin_functional_count_K = 1\nlevel_grid = "exact_breakpoints"\n'
    )
    trees = {s: _tree(tmp_path, f"tree-{s}", ident) for s, ident in STATES.items()}
    leaves = {
        s: _product(
            "leaves",
            f"leaves-{s}",
            tmp_path,
            **{
                "leaves.parquet": pl.DataFrame(
                    {
                        "peak_dna_sequence": ["AAA", "CCC"],
                        "peak_aa_sequence": ["K", "L"],
                    }
                )
            },
        )
        for s in STATES
    }
    landscape = _product(
        "landscape",
        "landscape",
        tmp_path,
        **{
            "vertices.parquet": pl.DataFrame(
                {"dna_sequence": list(VERTICES), "raw_fitness": list(VERTICES.values())}
            ),
            "diagnostics.json": {"threshold": 4.0},
        },
    )
    by_leaf = {leaf.label: s for s, leaf in leaves.items()}
    monkeypatch.setattr(
        merge_flow_product, "_verified", lambda loader, kind, label: leaves[label[7:]]
    )
    monkeypatch.setattr(
        merge_flow_product,
        "_parent",
        lambda loader, child, kind, state: (
            trees[by_leaf[child.label]]
            if child.product_type == "leaves"
            else trees["published_by"]
        ),
    )
    monkeypatch.setattr(merge_flow_product, "_landscape_for", lambda *a: landscape)
    monkeypatch.setattr(
        merge_flow_product,
        "_classes",
        lambda *a: (
            {
                "AAA": "functional",
                "CCC": "uncertain",
                "GGG": "non_functional",
                "TTT": "non_functional",
            },
            SNAPSHOT,
        ),
    )
    return context, {s: f"leaves-{s}" for s in STATES}


def test_command_is_context_aware():
    assert "merge-flow-product" in CONTEXT_AWARE_COMMANDS


def test_publishes_one_product_per_state(stubbed):
    context, labels = stubbed
    paths = merge_flow_product.build_merge_flow_products(context, labels=labels)
    assert set(paths) == set(STATES)
    for state, path in paths.items():
        assert parse_label(path.name).product_name == "merge_flow"
        flow = MergeFlow.from_artifacts({p.name: p for p in path.iterdir()})
        assert flow.components.height == 4
        assert set(flow.components["width_basis"]) == {
            "all" if state == "full" else "functional"
        }
    by = MergeFlow.from_artifacts({p.name: p for p in paths["published_by"].iterdir()})
    assert by.sidecar["class_counts"]["uncertain"] == 1
    assert by.components["summit_aa_sequence"].drop_nulls().len() == 4


def test_provenance_parents_digest_and_classification_hash(stubbed):
    context, labels = stubbed
    paths = merge_flow_product.build_merge_flow_products(context, labels=labels)
    for state, path in paths.items():
        text = "".join(p.read_text() for p in path.rglob("*.json"))
        assert "scientific_id_sha256" in text
        assert "code_digest" in text
        assert ("classification_table_sha256" in text) == state.startswith(
            "published_by"
        )


def test_justfile_recipe_exists():
    text = (REPO_ROOT / "justfile").read_text(encoding="utf-8")
    assert re.search(r"^merge-flow-product:", text, re.MULTILINE)
