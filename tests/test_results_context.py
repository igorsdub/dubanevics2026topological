"""Tests for explicit RunContext (Spec 1, ADR 0015, ADR 0016)."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from dubanevics2026topological.results import RunContext
from dubanevics2026topological.results.context import (
    ENV_ANALYSIS_VARIANT,
    ENV_BARSEQ_EXPORT,
    ENV_BOOTSTRAP_CHUNK_DIR,
    ENV_BOOTSTRAP_NO_DRAW,
    ENV_FONT_DIR,
    ENV_HOST,
    ENV_N_JOBS,
    ENV_OUTPUT_LEGACY,
    ENV_OUTPUT_ROOT,
    ENV_PROCESSED_ROOT,
    ENV_RAW_DIR,
    ENV_RUN_ID,
)


def test_default_construction_empty_env(tmp_path: Path) -> None:
    """RunContext defaults to repo paths and safe defaults when env is empty."""
    ctx = RunContext(env={}, root=tmp_path)
    assert ctx.root == tmp_path.resolve()
    assert ctx.output_root == tmp_path / "output"
    assert ctx.processed_root == tmp_path / "data" / "processed"
    assert ctx.raw_data_dir == tmp_path / "data" / "raw"
    assert ctx.raw_dir == tmp_path / "data" / "raw"
    assert ctx.barseq_export_dir is None
    assert ctx.bootstrap_chunk_dir is None
    assert ctx.chunk_dir is None
    assert ctx.font_dir is None
    assert ctx.host == "local"
    assert ctx.run_id is None
    assert ctx.worker_concurrency is None
    assert ctx.n_jobs is None
    assert ctx.variant_name is None
    assert ctx.no_draw_allowed is False


def test_construction_from_environment_variables(tmp_path: Path) -> None:
    """Every Spec 1 environment variable maps cleanly to its corresponding RunContext field."""
    out = tmp_path / "custom_output"
    processed = tmp_path / "custom_processed"
    raw = tmp_path / "custom_raw"
    barseq = tmp_path / "custom_barseq"
    chunks = tmp_path / "custom_chunks"
    fonts = tmp_path / "custom_fonts"

    env = {
        ENV_OUTPUT_ROOT: str(out),
        ENV_PROCESSED_ROOT: str(processed),
        ENV_RAW_DIR: str(raw),
        ENV_BARSEQ_EXPORT: str(barseq),
        ENV_BOOTSTRAP_CHUNK_DIR: str(chunks),
        ENV_FONT_DIR: str(fonts),
        ENV_HOST: "cluster",
        ENV_RUN_ID: "51756602",
        ENV_N_JOBS: "16",
        ENV_ANALYSIS_VARIANT: "fdr_by_q0.05_median_reflected",
        ENV_BOOTSTRAP_NO_DRAW: "1",
    }

    ctx = RunContext(env=env)
    assert ctx.output_root == out.resolve()
    assert ctx.processed_root == processed.resolve()
    assert ctx.raw_data_dir == raw.resolve()
    assert ctx.raw_dir == raw.resolve()
    assert ctx.barseq_export_dir == barseq.resolve()
    assert ctx.bootstrap_chunk_dir == chunks.resolve()
    assert ctx.chunk_dir == chunks.resolve()
    assert ctx.font_dir == fonts.resolve()
    assert ctx.host == "cluster"
    assert ctx.run_id == "51756602"
    assert ctx.worker_concurrency == 16
    assert ctx.n_jobs == 16
    assert ctx.variant_name == "fdr_by_q0.05_median_reflected"
    assert ctx.no_draw_allowed is True


def test_legacy_output_env_var(tmp_path: Path) -> None:
    """Legacy OUTPUT environment variable is honoured when FITNESSTREE_OUTPUT_ROOT is not set."""
    out = tmp_path / "legacy_output"
    ctx = RunContext(env={ENV_OUTPUT_LEGACY: str(out)})
    assert ctx.output_root == out.resolve()


def test_explicit_arguments_override_environment(tmp_path: Path) -> None:
    """Explicit constructor arguments take precedence over environment variables."""
    env = {
        ENV_OUTPUT_ROOT: str(tmp_path / "env_out"),
        ENV_HOST: "cluster",
        ENV_N_JOBS: "8",
    }
    explicit_out = tmp_path / "explicit_out"
    ctx = RunContext(
        output_root=explicit_out,
        host="slurm",
        n_jobs=4,
        env=env,
    )
    assert ctx.output_root == explicit_out.resolve()
    assert ctx.host == "slurm"
    assert ctx.worker_concurrency == 4
    assert ctx.n_jobs == 4


def test_immutability() -> None:
    """RunContext is frozen and immutable."""
    ctx = RunContext(env={})
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ctx.host = "new_host"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ctx.output_root = Path("/tmp")  # type: ignore[misc]


def test_env_vars_and_activated(tmp_path: Path) -> None:
    """env_vars exports all fields and activated() context manager sets and restores env."""
    out = tmp_path / "my_out"
    processed = tmp_path / "my_processed"
    ctx = RunContext(
        output_root=out,
        processed_root=processed,
        host="cluster",
        run_id="999",
        worker_concurrency=12,
        variant_name="fdr_by_q0.05_papkou_original",
        no_draw_allowed=True,
        env={},
    )

    ev = ctx.env_vars()
    assert ev[ENV_OUTPUT_ROOT] == str(out.resolve())
    assert ev[ENV_OUTPUT_LEGACY] == str(out.resolve())
    assert ev[ENV_PROCESSED_ROOT] == str(processed.resolve())
    assert ev[ENV_HOST] == "cluster"
    assert ev[ENV_RUN_ID] == "999"
    assert ev[ENV_N_JOBS] == "12"
    assert ev[ENV_ANALYSIS_VARIANT] == "fdr_by_q0.05_papkou_original"
    assert ev[ENV_BOOTSTRAP_NO_DRAW] == "1"

    # Verify activated() context manager
    original_out = os.environ.get(ENV_OUTPUT_ROOT)
    with ctx.activated():
        assert os.environ[ENV_OUTPUT_ROOT] == str(out.resolve())
        assert os.environ[ENV_HOST] == "cluster"
        assert os.environ[ENV_ANALYSIS_VARIANT] == "fdr_by_q0.05_papkou_original"
    assert os.environ.get(ENV_OUTPUT_ROOT) == original_out


def test_activated_context_clears_unset_compatibility_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit context with unset options must not inherit stale process env."""
    monkeypatch.setenv(ENV_ANALYSIS_VARIANT, "stale")
    monkeypatch.setenv(ENV_BOOTSTRAP_CHUNK_DIR, "/stale/chunks")
    context = RunContext(env={})

    with context.activated():
        assert os.environ[ENV_ANALYSIS_VARIANT] == ""
        assert os.environ[ENV_BOOTSTRAP_CHUNK_DIR] == ""


@pytest.mark.parametrize("invalid", ["0", "-1", "+2", "٢", "2.0"])
def test_invalid_n_jobs_environment_is_rejected(invalid: str) -> None:
    from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
        n_jobs_override,
    )

    with pytest.raises(ValueError, match="FITNESSTREE_N_JOBS"):
        RunContext(env={ENV_N_JOBS: invalid})
    with pytest.raises(ValueError, match="FITNESSTREE_N_JOBS"):
        n_jobs_override(environ={ENV_N_JOBS: invalid})


@pytest.mark.parametrize("invalid", [0, -1, 2.5, True, "+2"])
def test_invalid_explicit_n_jobs_is_rejected(invalid: object) -> None:
    with pytest.raises(ValueError, match="FITNESSTREE_N_JOBS"):
        RunContext(worker_concurrency=invalid)  # type: ignore[arg-type]


def test_migrated_helpers_ignore_environment_after_context_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dubanevics2026topological.analysis import load_settings
    from dubanevics2026topological.analysis.barseq_inputs import load_barseq_export
    from dubanevics2026topological.build import register_font_directory
    from dubanevics2026topological.simplification.bootstrap_chunks import (
        DEFAULT_CHUNK_BASE,
        ENV_XDG_DATA,
        chunk_base,
        no_draw_requested,
    )
    from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
        n_jobs_override,
    )

    context = RunContext(env={})
    monkeypatch.setenv(ENV_ANALYSIS_VARIANT, "missing-variant")
    monkeypatch.setenv(ENV_BOOTSTRAP_CHUNK_DIR, "/stale/chunks")
    monkeypatch.setenv(ENV_BOOTSTRAP_NO_DRAW, "1")
    monkeypatch.setenv(ENV_N_JOBS, "9")
    monkeypatch.setenv(ENV_FONT_DIR, "/missing/fonts")
    monkeypatch.setenv(ENV_XDG_DATA, "/stale/xdg")
    monkeypatch.setenv(ENV_BARSEQ_EXPORT, "/stale/barseq")

    assert load_settings(context=context)["analysis"]["dataset"]
    assert chunk_base(context=context) == DEFAULT_CHUNK_BASE.expanduser()
    assert no_draw_requested(context=context) is False
    assert n_jobs_override(context=context) is None
    assert register_font_directory(context=context) is None
    with pytest.raises(ValueError, match="RunContext has no barseq_export_dir"):
        load_barseq_export({}, context=context)


def test_build_rejects_output_override_for_unmigrated_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dubanevics2026topological.build import main

    with pytest.raises(SystemExit):
        main(["--output-root", str(tmp_path / "alternate"), "figures"])
    assert "overrides are not yet supported by the 'figures' command" in (
        capsys.readouterr().err
    )


def test_tables_dispatch_reports_paths_under_external_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import dubanevics2026topological.build as build_module

    output_root = tmp_path / "outside-repository"
    expected = output_root / "tables" / "synthetic.csv"
    monkeypatch.setattr(build_module, "ANALYSIS_VARIANT_AT_IMPORT", None)
    monkeypatch.delenv(ENV_ANALYSIS_VARIANT, raising=False)
    monkeypatch.setattr(
        build_module,
        "build_tables",
        lambda *, force, context: [expected],
    )

    assert build_module.main(["--output-root", str(output_root), "tables"]) == 0
    assert "Table: tables/synthetic.csv" in capsys.readouterr().out


def test_legacy_variant_from_import_environment_allows_unmigrated_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dubanevics2026topological.build as build_module

    calls: list[str] = []
    monkeypatch.setenv(ENV_ANALYSIS_VARIANT, "legacy_variant")
    monkeypatch.setattr(build_module, "ANALYSIS_VARIANT_AT_IMPORT", "legacy_variant")
    monkeypatch.setattr(
        build_module, "build_figures", lambda *args, **kwargs: calls.append("figures")
    )

    assert build_module.main(["figures"]) == 0
    assert calls == ["figures"]


def test_different_explicit_variant_is_rejected_for_unmigrated_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import dubanevics2026topological.build as build_module

    monkeypatch.setenv(ENV_ANALYSIS_VARIANT, "legacy_variant")
    monkeypatch.setattr(build_module, "ANALYSIS_VARIANT_AT_IMPORT", "legacy_variant")
    with pytest.raises(SystemExit):
        build_module.main(["--analysis-variant", "different_variant", "figures"])
    assert "overrides are not yet supported by the 'figures' command" in (
        capsys.readouterr().err
    )


def test_directory_helpers(tmp_path: Path) -> None:
    """Standard subdirectory helper methods return predictable paths."""
    out = tmp_path / "out"
    ctx = RunContext(output_root=out, env={})

    assert ctx.analysis_dir == out.resolve() / "analysis"
    assert ctx.tables_dir == out.resolve() / "tables"
    assert ctx.figures_dir == out.resolve() / "figures"
    assert ctx.checkpoint_dir("analysis") == out.resolve() / "checkpoints" / "analysis"
    assert ctx.product_dir("fitness", "my_fit") == out.resolve() / "fitness" / "my_fit"
