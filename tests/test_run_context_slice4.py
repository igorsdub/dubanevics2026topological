"""Ticket 12b (#137) Slice 4 tests: RunContext for the peak-analysis and uncertainty stage boundaries.

Tests for:
1. no import-time OUTPUT or environment-derived path in the peak stage module
2. checkpoint paths follow the context, not the importer
3. fitness tables and fingerprints follow ``context.raw_data_dir`` (the slice 3 follow-up)
4. the build command hands the parsed context to the stage and the table export
5. the pooled-scatter helpers no longer refuse a non-default output root
6. the leading-saddle-band fingerprint reads the component record through the context
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological import analysis
from dubanevics2026topological.analysis import (
    candidate_edge_evidence,
    candidate_tree_contrast,
    peak_uncertainty,
)
from dubanevics2026topological.landscape import full_population, selection
from dubanevics2026topological.results import RunContext, serialize_run_context
from dubanevics2026topological.results.context import REPO_ROOT

FROZEN_NAMES = ("OUTPUT", "ANALYSIS_DIR", "CACHE_DIR", "PENDING_PATH")
DATASET = {
    "landscape": "full",
    "fitness_wt": "data/raw/wt.parquet",
    "fitness_diff": "data/raw/diff.parquet",
}


def _raw_dir(path: Path, *, wt: str = "A") -> Path:
    path.mkdir(parents=True)
    pl.DataFrame({"dna_from": ["AAA"], "dna_to": [wt]}).write_parquet(
        path / "wt.parquet"
    )
    pl.DataFrame({"dna_from": ["AAA"], "dna_to": ["AAC"]}).write_parquet(
        path / "diff.parquet"
    )
    return path


def _settings() -> dict:
    return {
        "analysis": {
            "dataset": dict(DATASET),
            "clamped": {
                "published_functional_threshold": 0.0,
                "candidate_functional_threshold": {"kind": "fixed", "value": 0.0},
            },
        }
    }


def _context(tmp_path: Path, raw: Path) -> RunContext:
    return RunContext(root=REPO_ROOT, output_root=tmp_path / "out", raw_data_dir=raw)


def test_peak_module_has_no_import_time_output_paths() -> None:
    assert [n for n in FROZEN_NAMES if hasattr(peak_uncertainty, n)] == []


def test_peak_paths_follow_the_context(tmp_path: Path) -> None:
    context = RunContext(root=tmp_path, output_root=tmp_path / "run_a")
    analysis_dir = tmp_path / "run_a" / "analysis"
    assert peak_uncertainty.cache_paths(context) == (
        analysis_dir / "peak_uncertainty",
        analysis_dir / ".peak-uncertainty-pending",
    )


def test_contextless_peak_paths_resolve_the_environment_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FITNESSTREE_OUTPUT_ROOT", str(tmp_path / "late"))
    assert peak_uncertainty.cache_paths()[0] == (
        tmp_path / "late" / "analysis" / "peak_uncertainty"
    )


def test_cached_peak_loader_reads_only_under_its_context(tmp_path: Path) -> None:
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError):
        peak_uncertainty.load_cached_peak_uncertainty(context=context)


def test_pending_marker_is_the_contexts_own(tmp_path: Path) -> None:
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "ctx")
    cache_dir, pending = peak_uncertainty.cache_paths(context)
    cache_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        peak_uncertainty._load_peak_checkpoint(cache_dir, context=context)
    pending.write_text("pending\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pending"):
        peak_uncertainty._load_peak_checkpoint(cache_dir, context=context)
    # Another context's checkpoint is not held up by this marker.
    other = RunContext(root=REPO_ROOT, output_root=tmp_path / "other")
    other_dir, _ = peak_uncertainty.cache_paths(other)
    other_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        peak_uncertainty._load_peak_checkpoint(other_dir, context=other)


def test_dataset_input_path_follows_the_raw_data_dir(tmp_path: Path) -> None:
    context = RunContext(
        root=tmp_path / "repo",
        output_root=tmp_path / "o",
        raw_data_dir=tmp_path / "raw",
    )
    assert analysis.dataset_input_path("data/raw/wt.parquet", context) == (
        tmp_path / "raw" / "wt.parquet"
    )
    assert analysis.dataset_input_path("data/other/x.csv", context) == (
        tmp_path / "repo" / "data" / "other" / "x.csv"
    )
    assert analysis.dataset_input_path("data/raw/wt.parquet", None, root=tmp_path) == (
        tmp_path / "data" / "raw" / "wt.parquet"
    )


def test_peak_fingerprint_follows_the_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the raw directory keeps the fingerprint; changing its bytes does not."""
    monkeypatch.setattr(
        peak_uncertainty,
        "candidate_selection_fingerprint",
        lambda settings, context=None: "selection",
    )
    settings = _settings()

    def fingerprint(raw: Path) -> str:
        return peak_uncertainty._fingerprint(settings, context=_context(tmp_path, raw))

    first = _raw_dir(tmp_path / "raw_a")
    assert fingerprint(first) == fingerprint(_raw_dir(tmp_path / "raw_b"))
    assert fingerprint(first) != fingerprint(_raw_dir(tmp_path / "raw_c", wt="G"))


def test_edge_evidence_fingerprint_follows_the_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        candidate_edge_evidence,
        "candidate_selection_fingerprint",
        lambda settings, context=None: "selection",
    )
    settings = _settings()

    def fingerprint(raw: Path) -> dict:
        return candidate_edge_evidence._fingerprint(
            settings, context=_context(tmp_path, raw)
        )

    first = fingerprint(_raw_dir(tmp_path / "raw_a"))
    assert first == fingerprint(_raw_dir(tmp_path / "raw_b"))
    assert first != fingerprint(_raw_dir(tmp_path / "raw_c", wt="G"))
    assert "dataset:data/raw/wt.parquet" in first["files"]


def test_tree_contrast_fingerprint_follows_the_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        candidate_tree_contrast,
        "candidate_selection_fingerprint",
        lambda settings, context=None: "selection",
    )
    settings = _settings()
    settings["analysis"]["candidate_tree_contrast"] = {}

    def fingerprint(raw: Path) -> dict:
        return candidate_tree_contrast._fingerprint_without_export(
            settings, context=_context(tmp_path, raw)
        )

    first = fingerprint(_raw_dir(tmp_path / "raw_a"))
    assert first == fingerprint(_raw_dir(tmp_path / "raw_b"))
    assert first != fingerprint(_raw_dir(tmp_path / "raw_c", wt="G"))


def test_selection_reads_fitness_from_the_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pl.DataFrame({"dna_from": ["AAA"], "dna_to": ["AAC"], "s": [0.5]}).write_parquet(
        raw / "wt.parquet"
    )
    context = _context(tmp_path, raw)
    settings = _settings()
    assert selection.wild_type_sequence(settings, context) == "AAA"
    seen: list[Path] = []

    class Stop(Exception):
        pass

    def spy(path, *args, **kwargs):
        seen.append(Path(path))
        raise Stop

    # The full-population loader reads the same table through the context.
    monkeypatch.setattr(full_population.pl, "read_parquet", spy)
    with pytest.raises(Stop):
        full_population.load_full_network_fitness(settings, context=context)
    assert seen == [raw / "wt.parquet"]


def test_build_passes_the_context_to_peak_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module

    seen: dict[str, list[RunContext | None]] = {}

    def fake_compute(*, force=False, settings=None, context=None):
        seen.setdefault("peak", []).append(context)
        return {"tables": {}, "diagnostics": {}}

    def fake_tables(result=None, *, force=False, context=None):
        seen.setdefault("tables", []).append(context)
        return []

    monkeypatch.setattr(
        peak_uncertainty, "load_or_compute_peak_uncertainty", fake_compute
    )
    monkeypatch.setattr(build_module, "build_peak_tables", fake_tables)
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "peak")
    assert build_module.main(["peak-analysis", *serialize_run_context(context)]) == 0
    assert [c.output_root for c in seen["peak"]] == [tmp_path / "peak"]
    assert [c.output_root for c in seen["tables"]] == [tmp_path / "peak"]


def test_peak_tables_record_follows_the_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module

    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "ctx")
    provenance = tmp_path / "ctx" / "analysis" / "peak_uncertainty" / "provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"fingerprint": "f"}), encoding="utf-8")
    assert build_module._peak_provenance_path(context) == provenance
    digest = build_module._peak_analysis_fingerprint(context)
    captured: dict = {}

    def fake_publish(stage, produce, **kwargs):
        captured.update(kwargs, stage=stage)
        return []

    monkeypatch.setattr(build_module, "_publish_stage_tables", fake_publish)
    build_module.build_peak_tables({"tables": {}, "diagnostics": {}}, context=context)
    assert captured["context"] is context
    assert captured["record_path"] == (
        tmp_path / "ctx" / "tables" / ".peak-uncertainty-provenance.json"
    )
    assert captured["stage"].settings["peak_checkpoint"] == digest


def test_pooled_scatter_inputs_follow_a_non_default_output_root(
    tmp_path: Path,
) -> None:
    """The default-output-root refusal is gone: the context's own checkpoint is read."""
    from dubanevics2026topological.build.pooled_scatter_product import _peak_inputs

    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "elsewhere")
    peak_dir, _ = peak_uncertainty.cache_paths(context)
    (peak_dir / "tables").mkdir(parents=True)
    for key in ("study_clamped", "candidate_clamped"):
        for suffix in ("scatter", "contrasts", "peaks", "decay"):
            (peak_dir / "tables" / f"{key}_{suffix}.parquet").write_bytes(b"x")
    (peak_dir / "tables" / "pairwise_sign_disagreements.parquet").write_bytes(b"x")
    (peak_dir / "diagnostics.json").write_text("{}", encoding="utf-8")
    hashes = _peak_inputs(context)
    assert "diagnostics.json" in hashes
    assert len(hashes) == 10


def test_leading_saddle_band_stage_reads_the_component_record_of_its_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module
    from dubanevics2026topological.analysis import component_persistence

    context = _context(tmp_path, _raw_dir(tmp_path / "raw"))
    seen: list[RunContext | None] = []
    record = tmp_path / "component_provenance.json"
    record.write_text(
        json.dumps(
            {"fingerprint": "f", "pairs_sha256": "p", "diagnostics_sha256": "d"}
        ),
        encoding="utf-8",
    )

    def fake_paths(ctx=None):
        seen.append(ctx)
        return SimpleNamespace(provenance=record)

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(component_persistence, "component_paths", fake_paths)
    monkeypatch.setattr(build_module, "selection_source", lambda s, context=None: {})
    monkeypatch.setattr(build_module, "_stage", stop)
    settings = {
        "analysis": {
            "clamped": {"published_functional_threshold": 0.0},
            "dataset": dict(DATASET),
        }
    }
    with pytest.raises((Stop, KeyError)):
        build_module._leading_saddle_band_stage(settings, context=context)
    assert seen == [context]


