from __future__ import annotations


import polars as pl
import pytest

from dubanevics2026topological.analysis import (
    bootstrap_band_quantile_scan as scan_module,
)
from dubanevics2026topological.analysis.bootstrap_band_quantile_scan import (
    band_quantile_scan,
    quantile_grid,
    steps_around,
    validate_config,
)
from dubanevics2026topological.simplification.bootstrap_smoothed_tree import band_width

# Real distances 0.1, 0.2, 0.3, 0.4, 0.5: c(q) = 0.1 + 0.4 q (numpy linear).
REAL = [0.1, 0.2, 0.3, 0.4, 0.5]
# The flat rows are far larger and must never enter c.
FLAT = [9.0, 9.0, 9.0, 9.0, 9.0]
DRAWS = pl.DataFrame(
    {
        "condition": ["real"] * 5 + ["flat"] * 5,
        "bottleneck": REAL + FLAT,
    }
)
# Persistence 0.2 is exactly 2c at q = 0 (c = 0.1): a strict comparison drops it.
PERSISTENCE = [0.2, 0.5, 0.65, 0.9, 1.5]


def test_grid_includes_both_ends_and_0_95_exactly() -> None:
    grid = quantile_grid(0.001)
    assert len(grid) == 1001
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert 0.95 in grid


def test_scan_counts_at_both_ends_and_in_between() -> None:
    scan = band_quantile_scan(DRAWS, PERSISTENCE, [0.0, 0.5, 1.0])
    assert scan.columns == ["q", "c", "surviving_peaks"]
    assert scan["c"].to_list() == pytest.approx([0.1, 0.3, 0.5])
    # q = 0: 2c = 0.2, the peak at exactly 0.2 does not survive (strict >).
    # q = 0.5: 2c = 0.6 keeps 0.65, 0.9, 1.5.  q = 1: 2c = 1.0 keeps 1.5.
    assert scan["surviving_peaks"].to_list() == [4, 3, 1]


def test_flat_rows_are_ignored() -> None:
    real_only = DRAWS.filter(pl.col("condition") == "real")
    grid = [0.0, 0.25, 0.5, 0.95, 1.0]
    assert band_quantile_scan(DRAWS, PERSISTENCE, grid).equals(
        band_quantile_scan(real_only, PERSISTENCE, grid)
    )


def test_c_comes_from_band_width_and_counts_do_not_increase() -> None:
    grid = quantile_grid(0.01)
    scan = band_quantile_scan(DRAWS, PERSISTENCE, grid)
    for q, c in zip(scan["q"], scan["c"], strict=True):
        assert c == band_width(DRAWS, q)
    counts = scan["surviving_peaks"].to_list()
    assert counts == sorted(counts, reverse=True)


def test_peak_exactly_at_two_c_does_not_survive() -> None:
    # c(1) = 0.5, so a peak of persistence 1.0 sits exactly at 2c.
    scan = band_quantile_scan(DRAWS, [1.0], [1.0])
    assert scan["surviving_peaks"].to_list() == [0]
    below = band_quantile_scan(DRAWS, [1.0 + 1e-9], [1.0])
    assert below["surviving_peaks"].to_list() == [1]


def test_steps_around_finds_the_nearest_changes() -> None:
    scan = band_quantile_scan(DRAWS, PERSISTENCE, quantile_grid(0.01))
    steps = steps_around(scan, 0.5)
    assert steps["at"]["surviving_peaks"] == 3
    assert steps["below"]["surviving_peaks"] == 4
    assert steps["above"]["surviving_peaks"] == 2
    assert steps["below"]["q"] < 0.5 < steps["above"]["q"]


def test_validate_config_requires_a_grid_that_hits_one() -> None:
    assert validate_config({"grid_step": 0.001}) == 0.001
    with pytest.raises(ValueError, match="whole number"):
        validate_config({"grid_step": 0.3})
    with pytest.raises(ValueError, match="Unknown"):
        validate_config({"grid_step": 0.1, "extra": 1})


def test_survival_curve_is_exact_and_strict() -> None:
    values = [0.2, 0.5, 0.5, 0.9, 1.5]
    curve = scan_module.survival_curve(values)
    assert curve.columns == ["tau", "surviving_peaks"]
    # tau = 0 and every distinct persistence; ties count once and drop together.
    assert curve["tau"].to_list() == [0.0, 0.2, 0.5, 0.9, 1.5]
    assert curve["surviving_peaks"].to_list() == [5, 4, 2, 1, 0]


def test_survival_is_full_at_zero_and_empty_beyond_the_maximum() -> None:
    curve = scan_module.survival_curve(PERSISTENCE)
    assert curve["surviving_peaks"][0] == len(PERSISTENCE)
    assert curve["surviving_peaks"][-1] == 0
    counts = curve["surviving_peaks"].to_list()
    assert counts == sorted(counts, reverse=True)


RESULT_SETTINGS = {
    "n_replicates": 3,
    "seed": 1,
    "core_threshold": 0.95,
    "consensus_peak_frequency": 0.5,
    "band_quantile": 0.5,
}


def test_table_round_trip_and_stale_tables_are_refused(tmp_path, monkeypatch) -> None:
    result = type(
        "R",
        (),
        {
            "draw_stats": DRAWS,
            "hills": pl.DataFrame({"nominal_persistence": PERSISTENCE}),
            "summary": {"settings": RESULT_SETTINGS},
            "provenance": {"fingerprint": {"config": RESULT_SETTINGS}},
        },
    )()
    settings = {
        "analysis": {
            "bootstrap_band_quantile_scan": {"grid_step": 0.1},
            "bootstrap_smoothed_tree": {"band_quantile": 0.5},
        }
    }
    path, survival_path = tmp_path / "scan.parquet", tmp_path / "survival.parquet"
    monkeypatch.setattr(
        scan_module, "load_cached_bootstrap_smoothed_tree", lambda given, **_: result
    )
    written = scan_module.compute_bootstrap_band_quantile_scan(
        settings, path=path, survival_path=survival_path
    )

    def load():
        return scan_module.load_bootstrap_band_quantile_scan(
            settings, result, path=path, survival_path=survival_path
        )

    scan, survival = load()
    assert scan.equals(written) and scan.height == 11
    assert survival["surviving_peaks"][0] == len(PERSISTENCE)
    survival.with_columns(pl.col("surviving_peaks") + 1).write_parquet(survival_path)
    with pytest.raises(RuntimeError, match="stale"):
        load()
    survival_curve = scan_module.survival_curve(PERSISTENCE)
    survival_curve.write_parquet(survival_path)
    written.with_columns(pl.col("surviving_peaks") + 1).write_parquet(path)
    with pytest.raises(RuntimeError, match="stale"):
        load()
    path.unlink()
    # Nothing is mirrored: a missing table is reported, naming the recipes.
    with pytest.raises(FileNotFoundError, match="just bootstrap-band-quantile-scan"):
        load()


def test_the_scan_record_is_written_last_and_matches_the_tables(
    tmp_path, monkeypatch
) -> None:
    import json

    from dubanevics2026topological.simplification import checkpoint_identity

    result = type(
        "R",
        (),
        {
            "draw_stats": DRAWS,
            "hills": pl.DataFrame({"nominal_persistence": PERSISTENCE}),
            "summary": {"settings": RESULT_SETTINGS},
            "provenance": {"fingerprint": {"config": RESULT_SETTINGS}},
        },
    )()
    settings = {
        "analysis": {
            "bootstrap_band_quantile_scan": {"grid_step": 0.1},
            "bootstrap_smoothed_tree": {"band_quantile": 0.5},
        }
    }
    monkeypatch.setattr(
        scan_module, "load_cached_bootstrap_smoothed_tree", lambda given, **_: result
    )
    order: list[str] = []
    real = scan_module._write_atomic
    monkeypatch.setattr(
        scan_module,
        "_write_atomic",
        lambda frame, path: (order.append(path.name), real(frame, path))[1],
    )
    real_json = scan_module._write_json_atomic
    monkeypatch.setattr(
        scan_module,
        "_write_json_atomic",
        lambda value, path: (order.append(path.name), real_json(value, path))[1],
    )
    path, survival_path = tmp_path / "scan.parquet", tmp_path / "survival.parquet"
    (tmp_path / "provenance.json").write_text("stale record of an earlier run")
    seen: list[bool] = []
    real_scan = scan_module.write_scan
    monkeypatch.setattr(
        scan_module,
        "write_scan",
        lambda scan, to: (
            seen.append((tmp_path / "provenance.json").exists()),
            real_scan(scan, to),
        )[1],
    )
    scan_module.compute_bootstrap_band_quantile_scan(
        settings, path=path, survival_path=survival_path
    )
    assert seen == [False]  # the stale record is gone before the tables change
    assert order[:3] == ["survival.parquet", "scan.parquet", "provenance.json"]
    record = json.loads((tmp_path / "provenance.json").read_text())
    scientific = checkpoint_identity.scientific_fingerprint({"config": RESULT_SETTINGS})
    assert record["config"] == scan_module.scan_config(
        RESULT_SETTINGS, 0.1, scientific
    )
    assert record["artifacts"] == {
        "scan.parquet": checkpoint_identity.sha256_of(path),
        "survival.parquet": checkpoint_identity.sha256_of(survival_path),
    }
