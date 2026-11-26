"""Ticket 12b (#137) Slice 5 tests: RunContext for the bootstrap smoothed tree and band scan stages.

Tests for:
1. no import-time OUTPUT, CACHE or SCAN path in the bootstrap and scan modules
2. checkpoint, slot, mirror and scan paths follow the context, not the importer
3. the bootstrap fingerprint follows ``context.raw_data_dir``
4. the build commands hand the parsed context to the stages and the table exports
5. the band-scan and codon-entropy products no longer refuse a non-default output root
6. the follow-ups of slice 4: the full-analysis stage, the leading saddle band stage and the
   figure input loaders follow the context, and a missing outside input is a missing-inputs error
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological import analysis
from dubanevics2026topological.analysis import (
    bootstrap_band_quantile_scan as scan_module,
)
from dubanevics2026topological.analysis import (
    clamped_cache,
    peak_uncertainty,
)
from dubanevics2026topological.results import RunContext, serialize_run_context
from dubanevics2026topological.results.context import REPO_ROOT
from dubanevics2026topological.simplification import (
    bootstrap_chunks,
)
from dubanevics2026topological.simplification import (
    bootstrap_smoothed_tree as bst,
)

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


def _context(tmp_path: Path, raw: Path | None = None) -> RunContext:
    return RunContext(
        root=REPO_ROOT,
        output_root=tmp_path / "out",
        raw_data_dir=raw or tmp_path / "raw",
    )


def test_bootstrap_modules_have_no_import_time_output_paths() -> None:
    frozen = ("OUTPUT", "CACHE_DIR", "CHECKPOINT_DIR")
    assert [n for n in frozen if hasattr(bst, n)] == []
    frozen = ("OUTPUT", "SCAN_DIR", "SCAN_PATH", "SURVIVAL_PATH")
    assert [n for n in frozen if hasattr(scan_module, n)] == []


def test_bootstrap_paths_follow_the_context(tmp_path: Path) -> None:
    context = RunContext(root=tmp_path, output_root=tmp_path / "run_a")
    leaf = bst.checkpoint_dir(context)
    assert leaf.parent == (
        tmp_path / "run_a" / "checkpoints" / "bootstrap_smoothed_tree"
    )
    assert bst.cache_dir(context) == (
        tmp_path / "run_a" / "analysis" / "bootstrap_smoothed_tree"
    )
    scan_path, survival_path = scan_module.scan_paths(context)
    scan_dir = tmp_path / "run_a" / "analysis" / "bootstrap_band_quantile_scan"
    assert (scan_path, survival_path) == (
        scan_dir / "scan.parquet",
        scan_dir / "survival.parquet",
    )


def test_contextless_paths_resolve_the_environment_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FITNESSTREE_OUTPUT_ROOT", str(tmp_path / "late"))
    assert bst.cache_dir() == tmp_path / "late" / "analysis" / "bootstrap_smoothed_tree"
    assert scan_module.scan_dir() == (
        tmp_path / "late" / "analysis" / "bootstrap_band_quantile_scan"
    )


def test_processed_and_chunk_roots_follow_the_context(tmp_path: Path) -> None:
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "o",
        processed_root=tmp_path / "p",
        bootstrap_chunk_dir=tmp_path / "chunks",
        env={},
    )
    assert bst._processed_root(context) == tmp_path / "p"
    assert bst._chunk_base(context) == tmp_path / "chunks"
    assert bst._chunk_base(None) is None


def test_chunks_complete_reads_the_given_base(tmp_path: Path) -> None:
    assert not bootstrap_chunks.chunks_complete(
        {"config": {}}, n_draws=4, seed=1, chunk_size=2, base=tmp_path / "none"
    )


def test_cached_bootstrap_loader_reads_only_under_its_context(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FileNotFoundError, match="bootstrap-smoothed tree"):
            bst.load_cached_bootstrap_smoothed_tree(context=context)


def test_n_jobs_follows_the_context(tmp_path: Path) -> None:
    context = RunContext(
        root=tmp_path, output_root=tmp_path / "o", worker_concurrency=3
    )
    config = {
        "n_replicates": 10,
        "n_jobs": 1,
        "seed": 1,
        "band_quantile": 0.9,
    }
    assert bst._validate_config(config, context)["n_jobs"] == 3


def test_bootstrap_fingerprint_follows_the_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the raw directory keeps the fingerprint; changing its bytes does not."""
    monkeypatch.setattr(bst, "candidate_selection_identity", lambda *a, **k: "sel")
    settings = analysis.load_settings()
    settings["analysis"]["dataset"] = dict(DATASET)

    def fingerprint(raw: Path) -> dict:
        return bst._fingerprint(
            settings, {"fingerprint": "growth"}, context=_context(tmp_path, raw)
        )

    first = fingerprint(_raw_dir(tmp_path / "raw_a"))
    assert first == fingerprint(_raw_dir(tmp_path / "raw_b"))
    assert first != fingerprint(_raw_dir(tmp_path / "raw_c", wt="G"))
    assert "dataset:data/raw/wt.parquet" in first["files"]


def test_build_passes_the_context_to_the_bootstrap_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module

    seen: dict[str, list] = {}

    def fake_compute(settings=None, **kwargs):
        seen.setdefault("compute", []).append(kwargs["context"])
        raise SystemExit(0)

    monkeypatch.setattr(bst, "compute_bootstrap_smoothed_tree", fake_compute)
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "boot")
    with pytest.raises(SystemExit):
        build_module.main(["bootstrap-smoothed-tree", *serialize_run_context(context)])
    assert [c.output_root for c in seen["compute"]] == [tmp_path / "boot"]


def test_build_passes_the_context_to_the_band_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module

    seen: dict[str, list] = {}

    def fake_compute(settings, *, mirror=False, context=None):
        seen.setdefault("scan", []).append(context)
        return pl.DataFrame({"q": [0.95], "c": [0.0], "surviving_peaks": [1]})

    monkeypatch.setattr(
        scan_module, "compute_bootstrap_band_quantile_scan", fake_compute
    )
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "scan")
    assert (
        build_module.main(
            ["bootstrap-band-quantile-scan", *serialize_run_context(context)]
        )
        == 0
    )
    assert [c.output_root for c in seen["scan"]] == [tmp_path / "scan"]


def test_bootstrap_commands_are_context_aware() -> None:
    from dubanevics2026topological import build as build_module

    commands = {
        "bootstrap-smoothed-tree",
        "bootstrap-band-quantile-scan",
        "bootstrap-band-quantile-scan-product",
        "bootstrap-codon-entropy-product",
        "leading-saddle-band",
        "codon-entropy-analysis",
    }
    assert commands <= build_module.CONTEXT_AWARE_COMMANDS



def test_band_scan_inputs_follow_a_non_default_output_root(tmp_path: Path) -> None:
    from dubanevics2026topological.build import (
        bootstrap_band_quantile_scan_product as product,
    )

    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "elsewhere")
    scan_dir = scan_module.scan_dir(context)
    scan_dir.mkdir(parents=True)
    (scan_dir / scan_module.SCAN_FILE).write_bytes(b"s")
    (scan_dir / scan_module.SURVIVAL_FILE).write_bytes(b"v")
    (scan_dir / scan_module.RECORD_NAME).write_text("{}", encoding="utf-8")
    hashes = product._band_scan_inputs(context)
    assert set(hashes) == {
        "scan/scan.parquet",
        "scan/survival.parquet",
        "scan/provenance.json",
    }
    with pytest.raises(FileNotFoundError, match="bootstrap"):
        product._bootstrap_inputs(context)
    slot = bst.cache_dir(context)
    slot.mkdir(parents=True)
    (slot / "hills.parquet").write_bytes(b"h")
    assert set(product._bootstrap_inputs(context)) == {"bootstrap/slot/hills.parquet"}


def test_codon_entropy_bootstrap_inputs_follow_a_non_default_output_root(
    tmp_path: Path,
) -> None:
    from dubanevics2026topological.build import (
        bootstrap_codon_entropy_product as product,
    )

    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "elsewhere")
    published = bst.checkpoint_dir(context)
    published.mkdir(parents=True)
    (published / "hills.parquet").write_bytes(b"h")
    assert set(product._bootstrap_inputs(context)) == {"hills.parquet"}


def test_full_analysis_stage_follows_the_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = analysis.load_settings()
    settings["analysis"]["dataset"] = dict(DATASET)
    from dubanevics2026topological.analysis import produce_or_load

    def fingerprint(raw: Path) -> str:
        stage = analysis.analysis_stage(settings, context=_context(tmp_path, raw))
        return produce_or_load.stage_fingerprint(stage)

    first = fingerprint(_raw_dir(tmp_path / "raw_a"))
    assert first == fingerprint(_raw_dir(tmp_path / "raw_b"))
    assert first != fingerprint(_raw_dir(tmp_path / "raw_c", wt="G"))


def test_missing_outside_inputs_are_a_missing_inputs_error(tmp_path: Path) -> None:
    """A raw directory outside the repository that lacks its files is not a bare error."""
    settings = analysis.load_settings()
    settings["analysis"]["dataset"] = dict(DATASET)
    context = _context(tmp_path, tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="are missing"):
        analysis.analysis_stage(settings, context=context)
    with pytest.raises(FileNotFoundError, match="are missing"):
        peak_uncertainty.peak_stage(settings, context=context)
    with pytest.raises(FileNotFoundError, match="are missing"):
        clamped_cache.comparison_stage(settings, context=context)


def test_leading_saddle_band_stage_follows_the_raw_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module
    from dubanevics2026topological.analysis import component_persistence

    record = tmp_path / "component_provenance.json"
    record.write_text(
        json.dumps(
            {"fingerprint": "f", "pairs_sha256": "p", "diagnostics_sha256": "d"}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        component_persistence,
        "component_paths",
        lambda ctx=None: type("P", (), {"provenance": record})(),
    )
    monkeypatch.setattr(build_module, "selection_source", lambda s, context=None: {})
    settings = {
        "analysis": {
            "clamped": {
                "published_functional_threshold": 0.0,
                "candidate_functional_threshold": {
                    "rule": "median_reflected", "decimals": 6
                },
            },
            "dataset": dict(DATASET),
        }
    }

    def stage(raw: Path):
        return build_module._leading_saddle_band_stage(
            settings, context=_context(tmp_path, raw)
        )

    first = stage(_raw_dir(tmp_path / "raw_a"))
    assert "fitness_wt" in first.settings["outside_input_sha256"]
    assert first.settings == stage(_raw_dir(tmp_path / "raw_b")).settings
    assert first.settings != stage(_raw_dir(tmp_path / "raw_c", wt="G")).settings
    with pytest.raises(FileNotFoundError, match="are missing"):
        stage(tmp_path / "empty")


def test_figure_inputs_load_through_the_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module
    from dubanevics2026topological.analysis import clamped_cache as clamped
    from dubanevics2026topological.analysis import peak_uncertainty as peak

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        peak,
        "load_cached_peak_uncertainty",
        lambda *, context=None: seen.setdefault("peak", context),
    )
    monkeypatch.setattr(
        clamped,
        "load_cached_clamped_comparison",
        lambda *, context=None, **kwargs: seen.setdefault("comparison", context),
    )
    context = _context(tmp_path)
    build_module.load_inputs(["peak", "comparison"], {}, context=context)
    assert seen == {"peak": context, "comparison": context}


def test_context_without_a_chunk_dir_keeps_the_xdg_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage hands no base to the store, so XDG_DATA_HOME still applies."""
    monkeypatch.delenv("FITNESSTREE_BOOTSTRAP_CHUNK_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    context = RunContext(root=tmp_path, output_root=tmp_path / "o", env={})
    assert bst._chunk_base(context) is None
    directory = bootstrap_chunks.chunk_directory({"a": 1}, bst._chunk_base(context))
    assert directory.parent == tmp_path / "xdg" / "fitnesstree" / "bootstrap"


def test_a_symlinked_default_raw_dir_keeps_the_in_repository_key(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    store = _raw_dir(tmp_path / "store")
    (repo / "data" / "raw").symlink_to(store)
    context = RunContext(root=repo, output_root=repo / "output")
    path = analysis.dataset_input_path("data/raw/wt.parquet", context, root=repo)
    assert path == repo / "data" / "raw" / "wt.parquet"
    assert analysis.dataset_input_key("data/raw/wt.parquet", path, repo) == (
        "data/raw/wt.parquet"
    )


