"""Drift check against the original-run reference hashes, on toy tables."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.rawdata.files import load_config
from dubanevics2026topological.threshold import reference_drift as drift


def _table(classes: list[str], s: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dna_sequence": [f"AAAAAAAA{c}" for c in "ACGT"][: len(classes)],
            "s": s,
            "fdr_class": classes,
        }
    )


def test_config_keeps_the_three_reference_hashes() -> None:
    hashes = drift.reference_hashes(load_config())
    assert hashes["growth_rates_sha256"].startswith("9556393f")
    assert hashes["classification_median_reflected_sha256"].startswith("3c16ddf2")
    assert len(hashes["classification_papkou_original_sha256"]) == 64


def test_matching_hash_is_silent(tmp_path: Path) -> None:
    table = tmp_path / "t.parquet"
    _table(["functional"] * 4, [0.1] * 4).write_parquet(table)
    log: list[str] = []
    assert (
        drift.check_drift(
            table,
            label="x",
            reference_sha256=drift._sha256(table),
            reference_pattern="t.parquet",
            report_dir=tmp_path / "out",
            log=log.append,
        )
        is None
    )
    assert log == [] and not (tmp_path / "out").exists()


def test_mismatch_warns_and_reports_class_deltas(tmp_path: Path) -> None:
    new, reference = tmp_path / "new" / "t.parquet", tmp_path / "ref" / "t.parquet"
    new.parent.mkdir()
    reference.parent.mkdir()
    _table(["functional", "functional", "uncertain", "non_functional"], [1.0, 2.0, 3.0, 4.0]).write_parquet(new)
    _table(["functional", "uncertain", "uncertain", "non_functional"], [1.0, 2.0, 3.5, 4.0]).write_parquet(reference)
    log: list[str] = []
    report = drift.check_drift(
        new,
        label="classification_median_reflected",
        reference_sha256="0" * 64,
        reference_pattern="t.parquet",
        report_dir=tmp_path / "out",
        class_column="fdr_class",
        reference_dir=tmp_path / "ref",
        log=log.append,
    )
    assert any(line.startswith("WARNING") for line in log)
    assert report["class_counts"]["delta"] == {
        "functional": 1,
        "uncertain": -1,
        "non_functional": 0,
    }
    assert report["class_changed_rows"] == 1
    assert report["max_abs_difference"]["s"] == 0.5
    written = json.loads((tmp_path / "out" / drift.REPORT_NAME).read_text())
    assert written["classification_median_reflected"]["class_counts"]["new"]["functional"] == 2


def test_mismatch_without_a_reference_copy_only_warns(tmp_path: Path) -> None:
    table = tmp_path / "t.parquet"
    _table(["functional"] * 4, [0.1] * 4).write_parquet(table)
    log: list[str] = []
    report = drift.check_drift(
        table,
        label="growth_rates",
        reference_sha256="0" * 64,
        reference_pattern="missing.parquet",
        report_dir=tmp_path / "out",
        reference_dir=tmp_path / "nowhere",
        log=log.append,
    )
    assert report["reference_copy"] is None
    assert "class_counts" not in report
    assert any("No reference copy" in line for line in log)


def test_class_counts_skips_a_missing_column() -> None:
    table = _table(["functional"], [0.1])
    with pytest.warns(UserWarning, match="class column"):
        assert drift.class_counts(table, "no_such") is None
    with pytest.warns(UserWarning):
        report = drift.compare_tables(table, table, class_column="no_such")
    assert "class_counts" not in report


def test_null_versus_value_counts_as_changed() -> None:
    new = _table(["functional", "uncertain", "functional"], [0.1, 0.2, 0.3])
    reference = new.with_columns(
        pl.when(pl.col("fdr_class") == "uncertain").then(None).otherwise(pl.col("fdr_class")).alias("fdr_class")
    )
    report = drift.compare_tables(new, reference, class_column="fdr_class")
    assert report["class_changed_rows"] == 1


def test_ambiguous_reference_copy_is_not_guessed(tmp_path: Path) -> None:
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "t.parquet").write_bytes(b"x")
    with pytest.warns(UserWarning, match="Ambiguous"):
        assert drift.find_reference_copy("t.parquet", tmp_path / "a" / "..") is None
    assert drift.find_reference_copy("a/t.parquet", tmp_path) == tmp_path / "a" / "t.parquet"
    assert drift.find_reference_copy("*.parquet", tmp_path / "a") == tmp_path / "a" / "t.parquet"
