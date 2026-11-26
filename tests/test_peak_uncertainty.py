"""Scientific contract checks for threshold-dependent peak uncertainty."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.analysis import peak_uncertainty as peak_analysis
from dubanevics2026topological.analysis.peak_uncertainty import build_peak_uncertainty
from dubanevics2026topological.landscape.selected_landscape import (
    prepare_clamped_landscape,
)


def _prepared_landscape():
    vertices = pl.DataFrame(
        {
            "dna_sequence": ["peak", "valley", "second_peak", "low", "low_two"],
            "fitness": [0.5, 0.2, 0.6, 0.1, 0.08],
            "aa_sequence": ["P", "V", "Q", "L", "LL"],
            "has_stop_codon": [False] * 5,
        }
    )
    edges = pl.DataFrame(
        {
            "source": ["peak", "valley", "peak", "peak"],
            "target": ["valley", "second_peak", "low", "low_two"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "aa_sequence", "has_stop_codon"),
    )
    return prepare_clamped_landscape(ScalarNetwork(graph, scalar_attr="fitness"), -1.0)


def _inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    # The WT table carries per-sequence measurements; its source WT is not a
    # measured row. Pairwise records include both landscape edges and an
    # unrelated full-data comparison that must still affect global BH q-values.
    wt = pl.DataFrame(
        {
            "dna_from": ["wt"] * 6,
            "dna_to": ["wt", "peak", "valley", "second_peak", "low", "low_two"],
            "aa_from": ["WT"] * 6,
            "aa_to": ["WT", "P", "V", "Q", "L", "LL"],
            "s": [0.0, 0.5, 0.2, 0.6, 0.1, 0.08],
            "se": [0.0, 0.05, 0.08, 0.04, 0.06, 0.07],
            "pval": [1.0, 0.001, 0.2, 0.002, 0.3, 0.35],
        }
    )
    diff = pl.DataFrame(
        {
            "dna_from": ["peak", "peak", "valley", "peak", "unselected_from"],
            "dna_to": ["valley", "low", "second_peak", "low_two", "unselected_to"],
            "aa_from": ["P", "P", "V", "P", "X"],
            "aa_to": ["V", "L", "Q", "LL", "Y"],
            # The first estimate sign conflicts with the WT-relative fitness
            # ordering. The pair remains a landscape edge by sequence identity.
            "s": [-0.3, 0.2, 0.4, 0.2, 0.1],
            "se": [0.1, 0.2, 0.2, 0.2, 0.3],
            "pval": [0.01, 0.4, 0.02, 0.4, 0.03],
        }
    )
    return wt, diff


def _build(wt: pl.DataFrame, diff: pl.DataFrame):
    prepared = _prepared_landscape()
    return build_peak_uncertainty(
        wt,
        diff,
        {"study_clamped": prepared, "candidate_clamped": prepared},
    )


def test_peak_contrasts_follow_real_unordered_edges_and_global_bh_family() -> None:
    wt, diff = _inputs()
    result = _build(wt, diff)
    tables = result["tables"]

    contrasts = tables["study_clamped_contrasts"]
    peak_to_valley = contrasts.filter(
        (pl.col("dna_sequence") == "peak") & (pl.col("neighbor_sequence") == "valley")
    ).row(0, named=True)
    assert peak_to_valley["neighbor_fitness"] < peak_to_valley["raw_fitness"]
    assert peak_to_valley["pairwise_delta"] == -0.3
    # Five full-data p-values yield BH q=.05 for p=.01. Adjusting only the
    # four selected-edge comparisons would instead produce q=.04.
    assert peak_to_valley["q_value"] == pytest.approx(0.05)
    assert "unselected_from" not in contrasts["dna_sequence"].to_list()

    peaks = tables["study_clamped_peaks"].sort("dna_sequence")
    assert peaks["dna_sequence"].to_list() == ["peak", "second_peak"]
    peak = peaks.filter(pl.col("dna_sequence") == "peak").row(0, named=True)
    assert peak["neighbor_count"] == 3
    assert peak["max_neighbor_q"] == pytest.approx(0.4)
    assert peak["min_neighbor_score"] == pytest.approx(0.6)
    neighbor_q = contrasts.filter(pl.col("dna_sequence") == "peak")["q_value"].to_list()
    assert neighbor_q.count(0.4) == 2
    assert min(neighbor_q) < peak["max_neighbor_q"]


def test_scatter_excludes_wt_and_retention_curve_is_inclusive_and_monotone() -> None:
    wt, diff = _inputs()
    result = _build(wt, diff)
    tables = result["tables"]

    scatter = tables["study_clamped_scatter"]
    assert "wt" not in scatter["dna_sequence"].to_list()
    assert set(scatter["dna_sequence"]) == {
        "peak",
        "valley",
        "second_peak",
        "low",
        "low_two",
    }
    assert scatter.filter(pl.col("dna_sequence") == "peak")["is_peak"].item()

    decay = tables["study_clamped_decay"].sort("threshold")
    assert decay["threshold"][0] == 0.0
    assert decay["threshold"][-1] == 1.0
    counts = decay["retained_peak_count"].to_list()
    assert counts[0] == 2
    assert all(left >= right for left, right in pairwise(counts))
    score = (
        tables["study_clamped_peaks"]
        .filter(pl.col("dna_sequence") == "peak")["min_neighbor_score"]
        .item()
    )
    assert decay.filter(pl.col("threshold") == score)["retained_peak_count"].item() == 2
    after_score = math.nextafter(score, math.inf)
    assert (
        decay.filter(pl.col("threshold") == after_score)["retained_peak_count"].item()
        == 1
    )
    assert decay.filter(pl.col("threshold") == 1.0)["retained_peak_count"].item() == 0


def test_positive_tiny_qvalues_do_not_round_into_surviving_threshold_one() -> None:
    wt, diff = _inputs()
    diff = diff.with_columns(pl.lit(1e-100).alias("pval"))
    tables = _build(wt, diff)["tables"]
    peaks = tables["study_clamped_peaks"]
    assert (peaks["max_neighbor_q"] > 0.0).all()
    assert peaks["min_neighbor_score"].to_list() == [math.nextafter(1.0, 0.0)] * 2
    decay = tables["study_clamped_decay"]
    assert (
        decay.filter(pl.col("threshold") == math.nextafter(1.0, 0.0))[
            "retained_peak_count"
        ].item()
        == 2
    )
    assert decay.filter(pl.col("threshold") == 1.0)["retained_peak_count"].item() == 0


def test_zero_and_one_qvalues_follow_inclusive_score_endpoints() -> None:
    wt, diff = _inputs()
    diff = diff.with_columns(
        pl.when(pl.col("dna_to") == "second_peak")
        .then(1.0)
        .when(pl.col("dna_from") == "peak")
        .then(0.0)
        .otherwise(pl.col("pval"))
        .alias("pval")
    )
    result = _build(wt, diff)
    tables = result["tables"]

    peaks = tables["study_clamped_peaks"].sort("dna_sequence")
    assert peaks["min_neighbor_score"].to_list() == [1.0, 0.0]
    decay = tables["study_clamped_decay"]
    assert decay.filter(pl.col("threshold") == 0.0)["retained_peak_count"].item() == 2
    assert (
        decay.filter(pl.col("threshold") == math.nextafter(0.0, math.inf))[
            "retained_peak_count"
        ].item()
        == 1
    )
    assert decay.filter(pl.col("threshold") == 1.0)["retained_peak_count"].item() == 1


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("pval", float("nan")), ("se", 0.0)],
)
def test_invalid_pairwise_uncertainty_inputs_fail_clearly(
    column: str, bad_value: float
):
    wt, diff = _inputs()
    diff = diff.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(bad_value))
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(ValueError):
        _build(wt, diff)


def test_missing_selected_edge_comparison_is_rejected() -> None:
    wt, diff = _inputs()
    diff = diff.filter(pl.col("dna_from") != "valley")

    with pytest.raises(ValueError, match="comparison"):
        _build(wt, diff)


def test_peak_cache_loader_rejects_pending_incompatible_and_incomplete_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "peak-cache"
    pending = tmp_path / "pending"
    table_dir = cache / "tables"
    table_dir.mkdir(parents=True)
    monkeypatch.setattr(
        peak_analysis, "cache_paths", lambda context=None: (cache, pending)
    )

    provenance = {
        "schema": peak_analysis.CACHE_SCHEMA,
        "analysis_scalar": "raw_fitness",
        "edge_selection_rule": "at_least_one_endpoint_functional_class",
        "candidate_edge_selection_rule": "at_least_one_endpoint_functional_class",
        "published_edge_selection_rule": (
            "at_least_one_endpoint_raw_fitness_ge_threshold"
        ),
        "bh_family": "all_fitness_diff_rows_including_zero_deltas",
    }
    (cache / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (cache / "diagnostics.json").write_text("{}", encoding="utf-8")
    names = [
        f"{landscape}_{suffix}"
        for landscape in peak_analysis.LANDSCAPE_KEYS
        for suffix in peak_analysis._TABLE_SUFFIXES
    ] + ["pairwise_sign_disagreements"]
    for name in names:
        pl.DataFrame({"marker": [1]}).write_parquet(table_dir / f"{name}.parquet")

    assert set(peak_analysis.load_cached_peak_uncertainty()["tables"]) == set(names)
    (table_dir / f"{names[0]}.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        peak_analysis.load_cached_peak_uncertainty()
    pl.DataFrame({"marker": [1]}).write_parquet(table_dir / f"{names[0]}.parquet")

    provenance["schema"] = -1
    (cache / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        peak_analysis.load_cached_peak_uncertainty()
    provenance["schema"] = peak_analysis.CACHE_SCHEMA
    (cache / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    pending.write_text("pending", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pending"):
        peak_analysis.load_cached_peak_uncertainty()


def test_scientific_fingerprint_tracks_inputs_and_thresholds_not_appearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(peak_analysis, "ROOT", tmp_path)
    monkeypatch.setattr(
        peak_analysis,
        "candidate_selection_fingerprint",
        lambda _, **kwargs: {
            "rounded_threshold": -0.25,
            "source": {"rule": "median_reflected"},
        },
    )
    # No config/analysis.toml here: the fingerprint declares parsed settings only.
    (tmp_path / "pyproject.toml").write_text("project", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    (tmp_path / "wt.parquet").write_bytes(b"wt-v1")
    (tmp_path / "diff.parquet").write_bytes(b"diff-v1")
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "published_functional_threshold": -0.5,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
                "reference_topographer_commit": "abc123",
            },
        },
        "figures": {"figures": {"width": 300}},
    }
    original = peak_analysis._fingerprint(settings)
    settings["figures"]["figures"]["width"] = 600
    assert peak_analysis._fingerprint(settings) == original

    settings["analysis"]["clamped"]["published_functional_threshold"] = -0.4
    assert peak_analysis._fingerprint(settings) != original
    settings["analysis"]["clamped"]["published_functional_threshold"] = -0.5
    (tmp_path / "diff.parquet").write_bytes(b"diff-v2")
    assert peak_analysis._fingerprint(settings) != original
