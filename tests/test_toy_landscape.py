"""End-to-end checks for the hand-derived five-sequence toy landscape."""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path

import polars as pl
import pytest
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from dubanevics2026topological.analysis.biological_annotations import (
    _build_biological_annotations as build_biological_annotations,
)
from dubanevics2026topological.analysis.sequence_growth_rates import stop_codon_count
from dubanevics2026topological.landscape import selection as cs
from dubanevics2026topological.landscape.selected_landscape import (
    analyze_prepared_landscape,
    prepare_clamped_landscape,
)
from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
    nominal_hills,
)
from dubanevics2026topological.simplification.selected_leaf_simplification import (
    absorb_terminal_branches,
)
from dubanevics2026topological.threshold import classify_functional_sequences
from dubanevics2026topological.threshold import fdr_selection as fdr

DATA = Path(__file__).parent / "data" / "toy"
WT = "TAAACA"
T = -1.3
SE = 0.03
EXPECTED_CLASS = {
    "TAATAA": "non_functional",
    "CAAAAA": "uncertain",
    "TAAAAA": "functional",
    "TAAAAG": "functional",
    "TAAACA": "functional",
}


def _full_network():
    return Papkou2023Workflow(
        landscape="full",
        fitness_wt=pl.read_parquet(DATA / "fitness_wt.parquet"),
        fitness_diff=pl.read_parquet(DATA / "fitness_diff.parquet"),
    ).build_network()


def _toy_pipeline_module():
    path = DATA.parents[2] / "scripts" / "toy_pipeline.py"
    spec = spec_from_file_location("toy_pipeline_for_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_snapshot(tmp_path: Path, growth: pl.DataFrame) -> dict:
    sequence_ids = growth["sequence_id"].to_list()
    result = classify_functional_sequences(
        growth["s"].to_numpy(),
        growth["se_pooled"].to_numpy(),
        T,
        sequence_ids=sequence_ids,
        library_ids=sequence_ids,
    )
    classified = growth.with_columns(
        pl.Series("gated_p_value", result.gated_p_values),
        pl.Series("adjusted_p_value", result.q_values),
        pl.Series("fdr_class", result.classes),
    )
    table_path = tmp_path / fdr.table_name(T, "median_reflected", "by", 0.05)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    classified.select("dna_sequence", "fdr_class").write_parquet(table_path)
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "cutoffs": [
                    {
                        "name": "median_reflected",
                        "source_precision_threshold": T,
                        "applied_threshold": T,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "rule": "fdr_fixed_cutoff",
        "decimals": 6,
        "cutoff": "median_reflected",
        "method": "by",
        "q": 0.05,
        "input_dir": ".",
    }


def test_toy_growth_table_classes_and_stop_composition():
    growth = pl.read_parquet(DATA / "growth_rates.parquet")
    sequence_ids = growth["sequence_id"].to_list()
    dna_sequences = growth["dna_sequence"].to_list()
    result = classify_functional_sequences(
        growth["s"].to_numpy(),
        growth["se_pooled"].to_numpy(),
        T,
        sequence_ids=sequence_ids,
        library_ids=sequence_ids,
    )
    assert dict(zip(dna_sequences, result.classes, strict=True)) == EXPECTED_CLASS
    assert growth.height == 5
    assert growth.filter(pl.col("dna_sequence") == WT)["s"].to_list() == [0.0]
    assert growth["se_pooled"].unique().to_list() == [SE]
    assert growth.group_by("stop_codon_count").len().sort(
        "stop_codon_count"
    ).to_dicts() == [
        {"stop_codon_count": 0, "len": 1},
        {"stop_codon_count": 1, "len": 3},
        {"stop_codon_count": 2, "len": 1},
    ]
    p_values = dict(zip(dna_sequences, result.gated_p_values, strict=True))
    q_values = dict(zip(dna_sequences, result.q_values, strict=True))
    assert p_values["CAAAAA"] == pytest.approx(0.1586552539)
    assert q_values["CAAAAA"] == pytest.approx(0.4528285373)
    assert p_values["TAAAAA"] == pytest.approx(2.242774e-75, rel=1e-6, abs=0)
    assert p_values["TAAAAG"] == pytest.approx(1.408108e-83, rel=1e-6, abs=0)
    assert q_values["TAAAAA"] == pytest.approx(8.535002e-75, rel=1e-6, abs=0)
    assert q_values["TAAAAG"] == pytest.approx(8.037949e-83, rel=1e-6, abs=0)
    assert p_values["TAATAA"] == q_values["TAATAA"] == 1.0
    assert stop_codon_count("TAATAGTGA", ["TAA", "TAG", "TGA"]) == 3


def test_selection_tree_absorption_and_top_leaves(tmp_path, monkeypatch):
    growth = pl.read_parquet(DATA / "growth_rates.parquet")
    snapshot = tmp_path / "toy_snapshot"
    declaration = _write_snapshot(snapshot, growth)
    monkeypatch.setattr(fdr, "ROOT", tmp_path)
    declaration["input_dir"] = "toy_snapshot"
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "tests/data/toy/fitness_wt.parquet"},
            "clamped": {"candidate_functional_threshold": declaration},
        }
    }
    full = _full_network()
    prepared, selection = cs.prepare_candidate_landscape(full, settings)
    assert selection.threshold.threshold == T
    assert selection.functional_sequences == frozenset(
        dna for dna, label in EXPECTED_CLASS.items() if label == "functional"
    )
    ids = {
        prepared.network.graph[n]["dna_sequence"]
        for n in prepared.network.graph.node_indices()
    }
    assert ids == set(EXPECTED_CLASS)
    assert prepared.network.edge_count == 4
    actual_edges = {
        frozenset(prepared.network.graph[n]["dna_sequence"] for n in edge)
        for edge in prepared.network.graph.edge_list()
    }
    selected_ids = sorted(ids)
    induced_edges = {
        frozenset((left, right))
        for left, right in combinations(selected_ids, 2)
        if sum(a != b for a, b in zip(left, right, strict=True)) == 1
    }
    assert actual_edges == induced_edges
    analysis = analyze_prepared_landscape(prepared, name="toy", wild_type_sequence=WT)
    tree = analysis.trees["split"].tree
    saddles = [n for n in tree.node_indices() if tree[n]["event"] == "split_saddle"]
    assert len(saddles) == 1
    assert tree.in_degree(saddles[0]) == 2
    annotated = build_biological_annotations(analysis, landscape="toy")
    pairs = {row["birth_sequence"]: row for row in annotated.pairs.to_dicts()}
    assert set(pairs) == {"TAAACA", "TAAAAG"}
    assert pairs[WT]["death_sequence"] == "TAATAA"
    assert pairs[WT]["persistence"] == pytest.approx(1.4)
    syn = pairs["TAAAAG"]
    assert syn["adjacent_saddle_sequence"] == "TAAAAA"
    assert syn["persistence"] == pytest.approx(0.03)
    assert syn["synonymous_endpoints"] is True
    hills = nominal_hills(analysis.trees["split"])
    ranked = sorted(hills, key=lambda hill: (-len(hill.members), hill.peak_sequence))
    assert [(hill.peak_sequence, len(hill.members)) for hill in ranked[:2]] == [
        ("TAAAAG", 1),
        ("TAAACA", 1),
    ]
    absorbed = absorb_terminal_branches(analysis.trees["split"], {"TAAAAG"})
    assert absorbed.assignments.height == 5
    assert absorbed.assignments["dna_sequence"].n_unique() == 5
    assert set(absorbed.pairs["birth_label"]) == {WT}


def test_median_reflected_graph_is_root_only():
    full = _full_network()
    values = [full.graph[n]["s"] for n in full.graph.node_indices()]
    threshold = 2 * float(pl.Series(values).median()) - min(values)
    assert threshold == pytest.approx(-0.1)
    reflected = prepare_clamped_landscape(full, threshold)
    ids = {
        reflected.network.graph[n]["dna_sequence"]
        for n in reflected.network.graph.node_indices()
    }
    assert ids == {"TAAACA", "TAAAAA"}
    analysis = analyze_prepared_landscape(
        reflected, name="reflected", wild_type_sequence=WT
    )
    annotations = build_biological_annotations(analysis, landscape="reflected")
    leaves = annotations.pairs.filter(pl.col("adjacent_lower_is_root"))
    assert leaves.height == 1
    row = leaves.row(0, named=True)
    assert row["birth_sequence"] == WT
    assert row["adjacent_lower_sequence"] == "TAAAAA"
    assert row["adjacent_saddle_sequence"] is None
    # A root-only leaf has no adjacent saddle, so its synonymous contrast is ineligible.


def test_seeded_bootstrap_band_absorbs_local_leaf(tmp_path, monkeypatch, capsys):
    toy_pipeline = _toy_pipeline_module()
    output_root = tmp_path / "output"
    monkeypatch.setattr(toy_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(toy_pipeline, "OUTPUT", output_root)
    monkeypatch.setattr(fdr, "ROOT", tmp_path)

    toy_pipeline.main()
    capsys.readouterr()
    result_dir = output_root / "analysis" / "toy"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))

    # Deterministic 50-draw run; allow one part per million for platform float noise.
    assert summary["bootstrap_c"] == pytest.approx(0.07178182986543097, rel=1e-6)
    assert summary["bootstrap_summary"]["peaks_before"] == 2
    assert summary["band_peak_count"] == 1
    assert summary["codon_entropy"] == "not run (toy sequences are 6 nt)"

    assignments = pl.read_parquet(result_dir / "band_assignments.parquet")
    assert assignments["dna_sequence"].n_unique() == 5
    assert set(assignments["dna_sequence"]) == set(EXPECTED_CLASS)
    absorbed = set(assignments.filter(pl.col("absorbed_peak"))["dna_sequence"])
    assert absorbed == {"TAAAAG"}
    band_pairs = pl.read_parquet(result_dir / "band_pairs.parquet")
    assert set(band_pairs["birth_label"]) == {WT}
