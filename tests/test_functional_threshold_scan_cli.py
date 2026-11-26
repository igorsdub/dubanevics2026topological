"""The ``functional-threshold-scan`` command: unsliced, ``--slice`` and ``--merge``.

A tiny seeded growth table, ``k_range`` [1, 2] and four resamples; the growth table and
the classification are patched in, so no checkpoint or ``data/raw/`` is read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from test_functional_classification import (
    CONFIG as CLASSIFICATION,
)
from test_functional_classification import (
    DECLARATION,
    PUBLISHED,
    growth_table,
)

from dubanevics2026topological import build
from dubanevics2026topological.analysis import functional_threshold_scan as fts
from dubanevics2026topological.threshold import functional_classification as fc

SCAN = {
    "families": ["gaussian", "skew_normal", "student_t"],
    "k_range": [1, 2],
    "quantiles": [0.95, 0.99, 0.999, 0.9999],
    "confidences": [0.95, 0.99, 0.999],
    "component_rules": ["stop_anchored", "lowest_mean"],
    "n_resamples": 4,
    "random_seed": 20261025,
    "n_init": CLASSIFICATION["n_init"],
    "max_iter": CLASSIFICATION["max_iter"],
    "tol": CLASSIFICATION["tol"],
    "flip_fraction_limit": 0.05,
    "double_stop_below_min": 0.99,
    "n_jobs": 2,
}
COMMAND = "functional-threshold-scan"


@pytest.fixture(scope="module")
def growth() -> pl.DataFrame:
    table = growth_table([(-1.0, 0.15, 1500, 200, 100), (0.0, 0.10, 1500, 0, 0)])
    stops = table["stop_codon_count"].to_list()
    stops[:20] = [3] * 20
    return table.with_columns(stop_codon_count=pl.Series(stops, dtype=pl.Int64))


@pytest.fixture(scope="module")
def canonical(growth: pl.DataFrame) -> fc.FunctionalClassificationResult:
    return fc.classify_growth_rates(growth, CLASSIFICATION, PUBLISHED, DECLARATION)


@pytest.fixture
def cli(monkeypatch, growth, canonical, tmp_path: Path) -> dict[str, Any]:
    """Patch the settings, the inputs and the scan directory; return the settings."""
    settings = {"analysis": {fts.SECTION: dict(SCAN), fc.SECTION: dict(CLASSIFICATION)}}
    source = {"stage": "functional_classification", "fingerprint": "f1"}
    monkeypatch.delenv("FITNESSTREE_N_JOBS", raising=False)
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(fts, "SCAN_DIR", tmp_path / "scan")
    monkeypatch.setattr(
        fts, "_load_inputs", lambda settings, *, lock: (growth, source, "growth-sha")
    )
    monkeypatch.setattr(
        fc,
        "load_cached_functional_classification_with_source",
        lambda settings, *, tables=True, lock=True: (canonical, source),
    )
    return settings


def test_unsliced_computes_every_resample_and_writes_the_scan(
    cli, tmp_path: Path, capsys
) -> None:
    assert build.main([COMMAND]) == 0
    out = capsys.readouterr().out
    assert "Functional threshold scan ready" in out and "acceptance a pass" in out
    scan = fts.load_functional_threshold_scan(cli, scan_dir=tmp_path / "scan")
    assert scan.resamples["resample"].unique().sort().to_list() == [0, 1, 2, 3]
    assert scan.intervals.height == 6
    assert set(scan.acceptance["passed"]) == {"a", "b", "c", "all"}
    assert "n_jobs" not in scan.provenance["scan_config"]


def test_unsliced_skips_a_current_scan_unless_forced(
    cli, monkeypatch, tmp_path: Path, capsys
) -> None:
    assert build.main([COMMAND]) == 0
    capsys.readouterr()

    def refuse(*args, **kwargs):
        raise AssertionError("recomputed a current scan")

    monkeypatch.setattr(fts, "compute_functional_threshold_scan", refuse)
    assert build.main([COMMAND]) == 0
    assert "is current" in capsys.readouterr().out
    with pytest.raises(AssertionError):
        build.main([COMMAND, "--force"])


def test_a_changed_setting_makes_the_scan_stale_and_recomputes(
    cli, tmp_path: Path, capsys
) -> None:
    assert build.main([COMMAND]) == 0
    cli["analysis"][fts.SECTION]["flip_fraction_limit"] = 0.1
    capsys.readouterr()
    assert build.main([COMMAND]) == 0
    assert "ready" in capsys.readouterr().out
    scan = fts.load_functional_threshold_scan(cli, scan_dir=tmp_path / "scan")
    assert scan.provenance["scan_config"]["flip_fraction_limit"] == 0.1


def test_the_worker_count_comes_from_the_environment_first(
    cli, monkeypatch, tmp_path: Path
) -> None:
    seen = []
    real = fts.compute_functional_threshold_scan

    def spy(settings, **kwargs):
        seen.append(kwargs["n_jobs"])
        return real(settings, **kwargs)

    monkeypatch.setattr(fts, "compute_functional_threshold_scan", spy)
    monkeypatch.setenv("FITNESSTREE_N_JOBS", "1")
    assert build.main([COMMAND]) == 0
    assert seen == [1]
    monkeypatch.delenv("FITNESSTREE_N_JOBS")
    assert build.main([COMMAND, "--force"]) == 0
    assert seen == [1, 2]


def test_slices_then_merge_give_the_unsliced_scan(cli, tmp_path: Path, capsys) -> None:
    assert build.main([COMMAND]) == 0
    whole = fts.load_functional_threshold_scan(cli, scan_dir=tmp_path / "scan")
    # An empty scan directory again: the merge writes there.
    for path in (tmp_path / "scan").iterdir():
        path.unlink()
    a, b = tmp_path / "slice-a", tmp_path / "slice-b"
    assert build.main([COMMAND, "--slice", "0", "3", "--out", str(a)]) == 0
    assert build.main([COMMAND, "--slice", "3", "4", "--out", str(b)]) == 0
    assert not (tmp_path / "scan" / fts.PROVENANCE_NAME).exists()
    assert sorted(p.name for p in a.iterdir()) == [fts.RESAMPLES_NAME, fts.SLICE_NAME]
    capsys.readouterr()
    assert build.main([COMMAND, "--merge", str(a), str(b)]) == 0
    assert "Functional threshold scan ready" in capsys.readouterr().out
    merged = fts.load_functional_threshold_scan(cli, scan_dir=tmp_path / "scan")
    assert merged.resamples.equals(whole.resamples)
    assert merged.intervals.equals(whole.intervals)
    assert merged.cells.equals(whole.cells)
    assert merged.acceptance == whole.acceptance


def test_a_merge_with_a_missing_slice_fails_naming_the_resamples(
    cli, tmp_path: Path, capsys
) -> None:
    a = tmp_path / "slice-a"
    assert build.main([COMMAND, "--slice", "0", "2", "--out", str(a)]) == 0
    with pytest.raises(SystemExit) as stopped:
        build.main([COMMAND, "--merge", str(a)])
    assert stopped.value.code == 1
    assert "missing resample indices [2, 3]" in capsys.readouterr().err
    assert not (tmp_path / "scan").exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--slice", "0", "2"], "needs --out"),
        (["--out", "x"], "applies to --slice"),
        (["--slice", "2", "2", "--out", "x"], "not inside 0..4"),
        (["--slice", "0", "5", "--out", "x"], "not inside 0..4"),
        (["--slice", "0", "2", "--out", "x", "--merge", "y"], "exclude each other"),
        (["--slice", "0", "2", "--out", "x", "--force"], "take no --force"),
        (["--merge", "y", "--force"], "take no --force"),
    ],
)
def test_bad_forms_are_refused_before_any_work(
    cli, tmp_path: Path, capsys, argv: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as stopped:
        build.main([COMMAND, *argv])
    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "scan").exists()


def test_the_slice_options_belong_to_this_command_only(capsys) -> None:
    with pytest.raises(SystemExit):
        build.main(["analysis", "--slice", "0", "1", "--out", "x"])
    assert "apply to functional-threshold-scan" in capsys.readouterr().err


def test_the_registry_reads_a_scan_the_command_wrote(
    cli, monkeypatch, tmp_path: Path
) -> None:
    from dubanevics2026topological.analysis import manuscript_quantities as mq

    output = tmp_path / "output"
    scan_dir = output.joinpath(*mq.SCAN_DIR)
    monkeypatch.setattr(fts, "SCAN_DIR", scan_dir)
    assert build.main([COMMAND]) == 0
    scan = fts.load_functional_threshold_scan(cli, scan_dir=scan_dir)
    readers = {mq.SCAN_READER: lambda: scan}
    upstream = mq.load_threshold_scan(output, readers)
    assert upstream is not None and upstream.stage == "functional-threshold-scan"
    rows = mq._Rows(git_commit=None)
    mq._threshold_scan(rows, {mq.SCAN_READER: upstream})
    by_key = {row["key"]: row for row in rows.rows}
    assert all(row["gap_reason"] is None for row in by_key.values()), by_key
    passed = scan.acceptance["passed"]
    for name in ("a", "b", "c", "all"):
        key = f"functional_threshold_scan.acceptance_{name}_passed"
        assert by_key[key]["value"] == ("true" if passed[name] else "false")
    assert by_key["functional_threshold_scan.resample_count"]["value"] == "4"
    canonical = scan.acceptance["canonical"]
    interval = scan.intervals.filter(
        (pl.col("family") == canonical["family"]) & (pl.col("k") == canonical["k"])
    ).row(0, named=True)
    low = by_key["functional_threshold_scan.canonical_threshold_interval_low"]
    assert float(low["value"]) == interval["threshold_low"]
    limit = by_key["functional_threshold_scan.interval_width_limit"]
    assert float(limit["value"]) == scan.acceptance["b"]["limit"]
