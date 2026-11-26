"""Table-stage publishing: export under the lock; upstream keys ignore timestamps."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological import build
from dubanevics2026topological.analysis import checkpoint
from dubanevics2026topological.analysis.produce_or_load import Stage
from dubanevics2026topological.provenance import upstream_identity


def test_publish_exports_and_writes_record_under_the_lock(tmp_path, monkeypatch):
    from dubanevics2026topological.analysis import produce_or_load as module

    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    code = tmp_path / "code.py"
    code.write_text("v1", encoding="utf-8")
    stage = Stage("demo", settings={"a": 1}, code=(code,), root=tmp_path)
    checkpoint_dir = module.result_directory(tmp_path / "checkpoints" / "demo", stage)
    observed: list[str] = []

    original_export = module.Checkpoint.export_to

    def spy_export(self, destination, names=None):
        try:
            with checkpoint.checkpoint_lock(checkpoint_dir, "demo"):
                observed.append("unlocked")
        except RuntimeError:
            observed.append("locked")
        return original_export(self, destination, names)

    monkeypatch.setattr(module.Checkpoint, "export_to", spy_export)
    record = tmp_path / "tables" / ".demo.json"
    original_write = build._write_json_atomic

    def spy_write(path, value):
        if path == record:
            try:
                with checkpoint.checkpoint_lock(checkpoint_dir, "demo"):
                    observed.append("record-unlocked")
            except RuntimeError:
                observed.append("record-locked")
        return original_write(path, value)

    monkeypatch.setattr(build, "_write_json_atomic", spy_write)

    def produce(directory: Path) -> None:
        (directory / "t.csv").write_text("a\n1\n", encoding="utf-8")

    for _ in range(2):  # produce, then reuse
        written = build._publish_stage_tables(
            stage, produce, checkpoint_name="demo", record_path=record, force=False
        )
    assert observed == ["locked", "record-locked", "locked", "record-locked"]
    assert [p.name for p in written] == ["t.csv"]
    assert json.loads(record.read_text())["stage"] == "demo"


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2]})


def _tree_contrast(provenance):
    result = SimpleNamespace(pairs=_frame(), edges=_frame(), provenance=provenance)
    return (
        build.build_candidate_tree_contrast_tables,
        result,
        ".candidate-tree-contrast-provenance.json",
    )


def _growth(provenance):
    result = SimpleNamespace(growth_rates=_frame(), provenance=provenance)
    return (
        build.build_sequence_growth_rate_tables,
        result,
        ".sequence-growth-rates-provenance.json",
    )


def _bootstrap(provenance):
    hills = pl.DataFrame(
        {
            "peak_sequence": ["a"],
            "member_count": [1],
            "hill_support": [0.8],
            "nominal_birth": [1.0],
            "nominal_death": [0.0],
            "nominal_persistence": [1.0],
            "clears_band": [True],
            "band_outcome": ["kept"],
        }
    )
    result = SimpleNamespace(
        hills=hills,
        draw_stats=_frame(),
        provenance=provenance,
    )
    return (
        build.build_bootstrap_smoothed_tree_tables,
        result,
        ".bootstrap-smoothed-tree-provenance.json",
    )


def test_bootstrap_exports_only_direct_band_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    builder, result, _ = _bootstrap({"fingerprint": {"g": 1}, "artifacts": {}})
    written = builder(result)
    assert sorted(path.name for path in written) == [
        "papkou2023_bootstrap_draw_stats.csv",
        "papkou2023_bootstrap_draw_stats.parquet",
        "papkou2023_bootstrap_hills.csv",
        "papkou2023_bootstrap_hills.parquet",
    ]


def test_bootstrap_refuses_stale_retired_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    tables = tmp_path / "tables"
    tables.mkdir()
    stale = tables / "papkou2023_bootstrap_cores.csv"
    stale.write_text("historical evidence\n")
    builder, result, _ = _bootstrap({"fingerprint": {"g": 1}, "artifacts": {}})
    with pytest.raises(RuntimeError, match="Retired core-era bootstrap table"):
        builder(result)
    assert stale.read_text() == "historical evidence\n"


@pytest.mark.parametrize("factory", [_tree_contrast, _growth, _bootstrap])
def test_timestamp_only_upstream_change_reuses(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    base = {"fingerprint": {"g": 1}, "artifacts": {"x": "1"}}
    builder, result, record_name = factory(
        {**base, "created_at": "2026-01-01T00:00:00+00:00"}
    )
    builder(result)
    record = tmp_path / "tables" / record_name
    first = json.loads(record.read_text())

    result.provenance = {
        **base,
        "created_at": "2027-02-02T00:00:00+00:00",
        "versions": {"python": "9"},
    }
    builder(result)
    assert json.loads(record.read_text()) == first

    result.provenance = {**base, "artifacts": {"x": "2"}}
    builder(result)
    assert json.loads(record.read_text())["fingerprint"] != first["fingerprint"]


def test_upstream_identity_keeps_fingerprint_and_artifact_hashes_only():
    provenance = {
        "fingerprint": {"g": 1},
        "artifacts": {"x": "1"},
        "created_at": "2026-01-01T00:00:00+00:00",
        "versions": {"python": "9"},
    }
    assert upstream_identity(provenance) == {
        "fingerprint": {"g": 1},
        "artifacts": {"x": "1"},
    }


@pytest.mark.parametrize("missing", ["fingerprint", "artifacts"])
def test_upstream_identity_rejects_provenance_missing_a_key(missing):
    provenance = {"fingerprint": {"g": 1}, "artifacts": {"x": "1"}}
    del provenance[missing]
    with pytest.raises(KeyError, match=missing):
        upstream_identity(provenance)


def test_build_and_the_branch_decomposition_stage_share_one_upstream_identity():
    from dubanevics2026topological.analysis import branch_decomposition

    assert build.upstream_identity is upstream_identity
    assert branch_decomposition.upstream_identity is upstream_identity
    assert not hasattr(build, "_upstream_identity")


def test_build_stage_binds_the_current_build_root(tmp_path, monkeypatch):
    """Tests (and the watch loop) repoint ``build.ROOT``; stages must follow it."""
    monkeypatch.setattr(build, "ROOT", tmp_path)
    stage = build._stage("demo", settings={"a": 1})
    assert stage.name == "demo" and stage.root == tmp_path
    assert set(stage.execution_keys) == {"n_jobs", "chunk_size"}
    assert tuple(stage.libraries) == ("polars", "numpy", "pyarrow", "topographer")
