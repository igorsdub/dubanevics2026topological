"""Ridge entropy sweeps and their per-band-tree product, built from saved products only."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis.codon_entropy import compute_entropy_sweeps
from dubanevics2026topological.analysis.ridge_entropy import (
    RidgeEntropy,
    build_ridge_entropy,
    translate,
)
from dubanevics2026topological.build import CONTEXT_AWARE_COMMANDS
from dubanevics2026topological.build import ridge_entropy_product as product
from dubanevics2026topological.results import (
    STANDARD_PRODUCT_TYPES,
    RunContext,
    parse_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMITS = {"kem": "AAGGAAATG", "tcl": "ACCTGTCTT"}
VERTICES = {
    "AAGGAAATG": 9.0,
    "ACCTGTCTT": 8.0,
    "CTCTGCTTG": 4.0,
    "AAAAAAAAA": 1.0,
}
RIDGE_OF = {"AAGGAAATG": "kem", "ACCTGTCTT": "tcl"}
BAND = {
    "thr": "published",
    "fdr": "by",
    "q": 0.05,
    "draws": 10000,
    "seed": 20261019,
    "tree": "band",
    "bandq": 0.95,
}


def _members(ridge_of=RIDGE_OF) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "member_dna_sequence": list(VERTICES),
            "raw_fitness": list(VERTICES.values()),
            "ridge_name": [ridge_of.get(dna) for dna in VERTICES],
        },
        schema={
            "member_dna_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "ridge_name": pl.String,
        },
    )


def test_translate_gives_amino_acids_and_stops() -> None:
    assert translate("AAGGAAATG") == "KEM"
    assert translate("TAATAGTGA") == "%^&"


def test_populations_are_core_and_the_two_ridges() -> None:
    result = build_ridge_entropy(_members(), sweep_points=5)
    counts = result.sidecar["counts"]
    assert counts == {"core": 4, "kem": 1, "tcl": 1}
    populations = result.populations
    assert set(populations["population"]) == {"core", "kem", "tcl"}
    kem = populations.filter(pl.col("population") == "kem")
    assert kem["dna_sequence"].to_list() == ["AAGGAAATG"]
    assert kem["aa_sequence"].to_list() == ["KEM"]
    assert result.sidecar["ridges_present"] is True


def test_sweeps_equal_the_existing_computation() -> None:
    result = build_ridge_entropy(_members(), sweep_points=5)
    frames = {
        name: result.populations.filter(pl.col("population") == name)
        .with_columns(
            pl.col("raw_fitness").alias("fitness"),
            *(
                pl.col("aa_sequence").str.slice(i, 1).alias(f"aa{i + 1}")
                for i in range(3)
            ),
        )
        .drop("population")
        for name in ("core", "kem", "tcl")
    }
    expected = compute_entropy_sweeps(frames, num_thresholds=5)
    for name, table in expected.items():
        saved = result.sweeps.filter(pl.col("population") == name).drop("population")
        assert saved.equals(table)


def test_a_ridge_without_members_has_a_null_sweep() -> None:
    result = build_ridge_entropy(_members({"AAGGAAATG": "kem"}), sweep_points=5)
    assert result.sidecar["counts"]["tcl"] == 0
    assert result.sidecar["ridges_present"] is False
    tcl = result.sweeps.filter(pl.col("population") == "tcl")
    assert tcl["entropy_bits"].null_count() == tcl.height


def test_repeated_sequence_is_refused() -> None:
    with pytest.raises(ValueError, match="repeats"):
        build_ridge_entropy(pl.concat([_members(), _members()]), sweep_points=5)


def test_artifacts_round_trip(tmp_path: Path) -> None:
    result = build_ridge_entropy(_members(), sweep_points=5)
    result.save(tmp_path)
    again = RidgeEntropy.from_artifacts({p.name: p for p in tmp_path.iterdir()})
    assert again.sweeps.equals(result.sweeps)
    assert again.populations.equals(result.populations)
    assert again.sidecar == result.sidecar


def _tree(members=RIDGE_OF):
    return SimpleNamespace(
        nodes=pl.DataFrame(
            {
                "tree_node_identity": list(VERTICES),
                "scalar": list(VERTICES.values()),
            }
        ),
        edges=pl.DataFrame(
            {
                "source_tree_node_identity": ["AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG"],
                "target_tree_node_identity": ["CTCTGCTTG", "CTCTGCTTG", "AAAAAAAAA"],
            }
        ),
        branch_members=pl.DataFrame(
            schema={
                "source_tree_node_identity": pl.String,
                "target_tree_node_identity": pl.String,
                "dna_sequence": pl.String,
            }
        ),
    )


def test_membership_must_match_the_tree() -> None:
    product.check_membership(_tree(), SUMMITS, _members())
    with pytest.raises(ValueError, match="differs from the tree"):
        product.check_membership(_tree(), SUMMITS, _members({"AAGGAAATG": "kem"}))


def _saved(kind, label, tmp_path, identity=None, **files):
    directory = tmp_path / kind / label
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, content in files.items():
        path = directory / name
        content.write_parquet(path)
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


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    context = RunContext(root=tmp_path, output_root=tmp_path / "out", env={})
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "analysis.toml").write_text(
        '[codon_entropy]\nsweep_points = 5\nkem_summit_dna_sequence = "AAGGAAATG"\n'
        'tcl_summit_dna_sequence = "ACCTGTCTT"\n'
    )
    tree_files = _tree()
    tree = _saved(
        "tree",
        "tree-band",
        tmp_path,
        BAND,
        **{
            "critical_nodes.parquet": tree_files.nodes,
            "branches.parquet": tree_files.edges,
            "branch_members.parquet": tree_files.branch_members,
        },
    )
    landscape = _saved(
        "landscape",
        "landscape",
        tmp_path,
        **{
            "vertices.parquet": pl.DataFrame(
                {
                    "dna_sequence": list(VERTICES),
                    "raw_fitness": list(VERTICES.values()),
                }
            )
        },
    )
    ridges = _saved("ridges", "ridges", tmp_path, **{"members.parquet": _members()})
    leaf = SimpleNamespace(label="leaves-band")
    monkeypatch.setattr(
        product,
        "_verified",
        lambda loader, kind, label: ridges if kind == "ridges" else leaf,
    )
    monkeypatch.setattr(product, "_parent", lambda *a: tree)
    monkeypatch.setattr(product, "_landscape_for", lambda *a: landscape)
    return context, {state: "leaves-band" for state in product.BAND_STATES}


def test_publishes_one_product_per_band_tree(stubbed) -> None:
    context, labels = stubbed
    paths = product.build_ridge_entropy_products(context, labels=labels)
    assert set(paths) == {"published_by_band", "reflected_by_band"}
    for path in paths.values():
        parsed = parse_label(path.name)
        assert parsed.product_name == "ridge_entropy_sweeps"
        assert parsed.params["tree"] == "band"
        saved = RidgeEntropy.from_artifacts({p.name: p for p in path.iterdir()})
        assert saved.sidecar["counts"] == {"core": 4, "kem": 1, "tcl": 1}
        manifest = json.loads((path / "manifest.json").read_text())
        assert [p["product_type"] for p in manifest["parents"]] == [
            "tree",
            "landscape",
            "ridges",
        ]


def test_publishing_twice_gives_the_same_product(stubbed) -> None:
    context, labels = stubbed
    first = product.build_ridge_entropy_products(context, labels=labels)
    assert product.build_ridge_entropy_products(context, labels=labels) == first


def test_command_recipe_and_type_are_registered() -> None:
    assert "ridge-entropy-product" in CONTEXT_AWARE_COMMANDS
    assert "ridge_entropy_sweeps" in STANDARD_PRODUCT_TYPES
    text = (REPO_ROOT / "justfile").read_text(encoding="utf-8")
    assert re.search(r"^ridge-entropy-product:", text, re.MULTILINE)


def test_band_label_needs_the_band_keys() -> None:
    label = (
        "ridge_entropy_sweeps__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019"
        "_tree=band_bandq=0.95_hash=9b3f0432"
    )
    assert parse_label(label).format() == label
    with pytest.raises(ValueError, match="requires"):
        parse_label(
            "ridge_entropy_sweeps__thr=published_fdr=none_tree=band_hash=9b3f0432"
        )


def test_code_digest_covers_the_entropy_code(stubbed, monkeypatch) -> None:
    context, labels = stubbed
    seen = []
    real = product._sha256
    monkeypatch.setattr(
        product, "_sha256", lambda path: seen.append(Path(path).name) or real(path)
    )
    product.build_ridge_entropy_products(context, labels=labels)
    assert {
        "ridge_entropy.py",
        "codon_entropy.py",
        "codon_entropy_populations.py",
        "ridges.py",
    } <= set(seen)


def test_membership_must_be_the_landscape_vertex_set(stubbed, monkeypatch) -> None:
    context, labels = stubbed
    short = SimpleNamespace(
        artifacts={
            "vertices.parquet": _write(
                context, pl.DataFrame({"dna_sequence": list(VERTICES)[:3]})
            )
        }
    )
    monkeypatch.setattr(product, "_landscape_for", lambda *a: short)
    with pytest.raises(ValueError, match="vertex set"):
        product.build_ridge_entropy_products(context, labels=labels)


def _write(context, frame) -> Path:
    path = Path(context.root) / "short_vertices.parquet"
    frame.write_parquet(path)
    return path
