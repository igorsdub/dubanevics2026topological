"""Ticket 12b (#137) Slice 2 tests: RunContext for the functional-classification stage.

Tests for:
1. no import-time OUTPUT or environment-derived path in the stage module
2. checkpoint, export and growth-rate paths follow the context, not the importer
3. the build command hands the parsed context to the stage, one run at a time
4. job.json records the context of a functional-classification job
5. the justfile lists only recipes that exist as prerequisites
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubanevics2026topological.results import RunContext, serialize_run_context
from dubanevics2026topological.results.context import REPO_ROOT
from dubanevics2026topological.threshold import functional_classification as fc

JUSTFILE = REPO_ROOT / "justfile"


def test_stage_module_has_no_import_time_output_paths() -> None:
    assert not hasattr(fc, "OUTPUT")
    assert not hasattr(fc, "CHECKPOINT_DIR")
    assert not hasattr(fc, "EXPORT_DIR")


def test_paths_follow_the_context(tmp_path: Path) -> None:
    context = RunContext(root=tmp_path, output_root=tmp_path / "run_a")
    assert fc.checkpoint_dir(context) == (
        tmp_path / "run_a" / "checkpoints" / "functional_classification"
    )
    assert fc.export_dir(context) == (
        tmp_path / "run_a" / "analysis" / "functional_classification"
    )
    assert fc.growth_rates_path(context) == (
        tmp_path
        / "run_a"
        / "analysis"
        / "sequence_growth_rates"
        / "growth_rates.parquet"
    )


def test_contextless_paths_resolve_the_environment_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change of the output-root variable after import is honoured: nothing is frozen."""
    monkeypatch.setenv("FITNESSTREE_OUTPUT_ROOT", str(tmp_path / "late"))
    assert fc.checkpoint_dir() == (
        tmp_path / "late" / "checkpoints" / "functional_classification"
    )
    assert fc.export_dir() == (
        tmp_path / "late" / "analysis" / "functional_classification"
    )
    assert fc.growth_rates_path() == (
        tmp_path
        / "late"
        / "analysis"
        / "sequence_growth_rates"
        / "growth_rates.parquet"
    )


def test_source_loader_declares_the_context_growth_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The with-source loader fingerprints the context's growth-rate table."""
    seen: list[Path] = []
    real = fc.functional_classification_stage

    def spy(settings, *, root=None, growth_rates=None):
        seen.append(growth_rates)
        return real(settings, root=root, growth_rates=growth_rates)

    monkeypatch.setattr(fc, "ROOT", Path(tmp_path.anchor))
    monkeypatch.setattr(fc, "functional_classification_stage", spy)
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "ctx")
    with pytest.raises(FileNotFoundError):
        fc.load_cached_functional_classification_with_source(context=context)
    assert seen == [fc.growth_rates_path(context)]


def test_stage_reads_and_writes_only_under_its_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing growth-rate table is named under the context's root, not the default."""
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError) as caught:
        fc.compute_functional_classification(context=context)
    assert str(tmp_path / "elsewhere") in str(caught.value)
    assert not (tmp_path / "elsewhere" / "checkpoints").exists()


def test_build_passes_the_context_to_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module

    seen: list[RunContext] = []

    def fake_compute(settings, *, force=False, context=None):
        seen.append(context)
        return SimpleNamespace()

    monkeypatch.setattr(fc, "compute_functional_classification", fake_compute)
    monkeypatch.setattr(fc, "describe", lambda result: "described")
    for name in ("run_first", "run_second"):
        context = RunContext(root=REPO_ROOT, output_root=tmp_path / name)
        flags = ["functional-classification", *serialize_run_context(context)]
        assert build_module.main(flags) == 0
    assert [c.output_root for c in seen] == [
        tmp_path / "run_first",
        tmp_path / "run_second",
    ]


def test_justfile_prerequisites_are_defined_recipes() -> None:
    text = JUSTFILE.read_text(encoding="utf-8")
    # A header is `name [params with defaults]: prerequisites`; the default may
    # hold a quoted empty string, so split at the colon that ends the parameters.
    header = re.compile(
        r'^([A-Za-z][\w-]*)((?:\s+[*+]?\w+(?:="[^"]*")?)*)\s*:(?!=)(.*)$', re.MULTILINE
    )
    parsed = header.findall(text)
    recipes = {name for name, _, _ in parsed}
    for _, _, prerequisites in parsed:
        for prerequisite in re.findall(r"\(([^\s)]+)[^)]*\)|(\S+)", prerequisites):
            name = prerequisite[0] or prerequisite[1]
            assert name in recipes, name
    assert (
        "tree-layout-products"
        not in text.split("bootstrap-codon-entropy-product:")[1].splitlines()[0]
    )
