from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.analysis import barseq_inputs
from dubanevics2026topological.analysis.barseq_inputs import (
    BarseqExport,
    load_barseq_export,
)

ENV = "TEST_BARSEQ_EXPORT"
SETTINGS = {"analysis": {"candidate_tree_contrast": {"barseq_export_env": ENV}}}


def _isolate_env_file(monkeypatch, tmp_path) -> Path:
    env_file = tmp_path / ".env"
    monkeypatch.setattr(barseq_inputs, "ENV_FILE", env_file)
    return env_file


def _write_export(directory: Path, *, counts_columns: tuple[str, ...] | None = None):
    columns = counts_columns or ("sequence_id", "sample_id", "count")
    counts = pl.DataFrame({name: [1, 2, 3] for name in columns})
    counts.write_parquet(directory / "counts.parquet")
    pl.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "replicate_id": ["r1", "r2"],
            "condition": ["c", "c"],
            "time": [0, 1],
            "sample_total": [100, 100],
        }
    ).write_parquet(directory / "samples.parquet")
    pl.DataFrame({"sequence_id": ["a"], "sequence": ["ACGT"]}).write_parquet(
        directory / "sequences.parquet"
    )


def test_loads_three_tables(tmp_path, monkeypatch):
    _isolate_env_file(monkeypatch, tmp_path)
    _write_export(tmp_path)
    monkeypatch.setenv(ENV, str(tmp_path))
    export = load_barseq_export(SETTINGS)
    assert isinstance(export, BarseqExport)
    assert export.counts.columns == ["sequence_id", "sample_id", "count"]
    assert export.counts.height == 3
    assert export.samples.height == 2
    assert export.sequences.columns == ["sequence_id", "sequence"]
    assert export.source_dir == tmp_path.resolve()


def test_unset_variable_raises(tmp_path, monkeypatch):
    _isolate_env_file(monkeypatch, tmp_path)
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(ValueError, match=ENV):
        load_barseq_export(SETTINGS)


def test_missing_directory_raises(tmp_path, monkeypatch):
    _isolate_env_file(monkeypatch, tmp_path)
    missing = tmp_path / "absent"
    monkeypatch.setenv(ENV, str(missing))
    with pytest.raises(ValueError, match=re.escape(str(missing))):
        load_barseq_export(SETTINGS)


def test_missing_file_raises(tmp_path, monkeypatch):
    _isolate_env_file(monkeypatch, tmp_path)
    _write_export(tmp_path)
    (tmp_path / "samples.parquet").unlink()
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(ValueError, match=r"samples\.parquet"):
        load_barseq_export(SETTINGS)


def test_missing_column_raises(tmp_path, monkeypatch):
    _isolate_env_file(monkeypatch, tmp_path)
    _write_export(tmp_path, counts_columns=("sequence_id", "sample_id"))
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(ValueError, match=r"counts.*count"):
        load_barseq_export(SETTINGS)


def test_env_file_supplies_variable(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_export(export_dir)
    env_file = _isolate_env_file(monkeypatch, tmp_path)
    env_file.write_text(f"{ENV}={export_dir}\n")
    # Registers teardown so the value dotenv writes to os.environ is removed.
    monkeypatch.delenv(ENV, raising=False)
    export = load_barseq_export(SETTINGS)
    assert export.source_dir == export_dir.resolve()


def test_shell_variable_overrides_env_file(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_export(export_dir)
    env_file = _isolate_env_file(monkeypatch, tmp_path)
    env_file.write_text(f"{ENV}={tmp_path / 'absent'}\n")
    monkeypatch.setenv(ENV, str(export_dir))
    export = load_barseq_export(SETTINGS)
    assert export.source_dir == export_dir.resolve()
