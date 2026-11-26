"""Direct persistence-band behaviour on the small nominal split tree."""

from types import SimpleNamespace

import polars as pl
import pytest
from test_bootstrap_smoothed_tree import _cooc, _growth, _toy

from dubanevics2026topological.simplification import bootstrap_smoothed_tree as stage
from dubanevics2026topological.simplification.bootstrap_aggregation import (
    BootstrapDistances,
)


def _evidence(two_c):
    _, tree = _toy()
    hills = stage.nominal_hills(tree)
    diagram = stage.nominal_diagram(tree, hills)
    support = {hill.peak_sequence: 0.0 for hill in hills}
    return tree, diagram, stage._hill_table(hills, support, diagram, two_c)


def test_band_keeps_unsupported_peak_above_threshold():
    tree, _, hills = _evidence(0.6)
    assert set(stage.build_band_simplified_view(tree, hills).pairs["birth_label"]) == {
        "b",
        "f",
    }
    assert hills.filter(pl.col("peak_sequence") == "b")["hill_support"][0] == 0


def test_band_drops_peak_at_exact_threshold():
    _, diagram, _ = _evidence(0.0)
    cutoff = diagram.filter(pl.col("peak_sequence") == "d")["persistence"][0]
    tree, _, hills = _evidence(cutoff)
    assert not hills.filter(pl.col("peak_sequence") == "d")["clears_band"][0]
    assert "d" not in set(
        stage.build_band_simplified_view(tree, hills).pairs["birth_label"]
    )


def test_tied_persistences_obey_same_strict_rule():
    _, tree = _toy()
    hills = stage.nominal_hills(tree)
    diagram = stage.nominal_diagram(tree, hills).with_columns(
        pl.lit(0.5).alias("persistence")
    )
    evidence = stage._hill_table(
        hills, {h.peak_sequence: 0.0 for h in hills}, diagram, 0.5
    )
    assert evidence["clears_band"].to_list() == [False] * len(hills)


def test_global_pair_is_closed_at_root():
    _, diagram, _ = _evidence(0.0)
    row = diagram.filter(pl.col("peak_sequence") == "f").row(0, named=True)
    assert row["birth"] == pytest.approx(0.6)
    assert row["death"] == pytest.approx(-0.2)
    assert row["persistence"] == pytest.approx(0.8)


def test_no_surviving_peaks_is_explicitly_rejected():
    tree, _, hills = _evidence(10.0)
    with pytest.raises(
        ValueError, match="Selected leaves would remove every surviving peak"
    ):
        stage.build_band_simplified_view(tree, hills)


def test_producer_uses_core_free_results_and_direct_schema(tmp_path):
    network, tree = _toy()
    old = _cooc([], bottleneck=[0.3] * 20)
    direct = BootstrapDistances(
        old.draw_stats, old.nominal_hill_support, 20, 7, "split"
    )

    class Runner:
        def __init__(self):
            self.store = SimpleNamespace(directory=None)
            self.session_draws = {"observed": 0, "flat": 0}

        def run(self, name, network, statistic, *, label=None):
            return direct

    result = stage.build_bootstrap_smoothed_tree(
        network,
        tree,
        _growth(),
        {"n_replicates": 20, "seed": 7, "band_quantile": 0.95, "n_jobs": 1},
        log=lambda _: None,
        chunks=Runner(),
    )
    assert result.hills.columns == list(stage.HILL_COLUMNS)
    assert set(result.hills.filter(pl.col("clears_band"))["peak_sequence"]) == {
        "b",
        "f",
    }
    assert (
        result.summary["peaks_after"]
        == result.summary["band_simplification"]["peaks_after"]
        == 2
    )
    assert "false_hill_rate" not in result.summary["diagnostics"]["flat_landscape"]
    assert not hasattr(result, "cores")
    fingerprint = {
        "schema": 7,
        "config": {
            "n_replicates": 20,
            "seed": 7,
            "band_quantile": 0.95,
            "noise_model": "pooled",
        },
        "candidate_selection": {"selected_sequence_set_sha256": "a" * 64},
        "candidate_functional_threshold": -1.0,
        "growth_rates_checkpoint": {},
        "files": {"data/raw/fitness.parquet": "b" * 64},
    }
    stage._write_checkpoint(result, tmp_path, fingerprint)
    assert {path.name for path in tmp_path.iterdir()} == {
        "hills.parquet",
        "draw_stats.parquet",
        "summary.json",
        "provenance.json",
    }
    loaded = stage._read_checkpoint(tmp_path)
    assert loaded.hills.equals(result.hills)
    assert (
        loaded.summary["scientific_identities"]["band"]["scientific_identity"][
            "simplification_method"
        ]
        == "direct_band"
    )
