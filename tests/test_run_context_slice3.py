"""Ticket 12b (#137) Slice 3 tests: RunContext for component-analysis, comparison-analysis and the Candidate stages.

Tests for:
1. no import-time OUTPUT or environment-derived path in the stage modules
2. checkpoint paths follow the context, not the importer
3. the Selection helpers hand the context to functional-classification
4. the build commands hand the parsed context to each stage
5. job.json records the context of a job running these stages
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dubanevics2026topological.analysis import (
    candidate_edge_evidence,
    candidate_tree_contrast,
    clamped_cache,
    component_persistence,
)
from dubanevics2026topological.landscape import selection
from dubanevics2026topological.results import RunContext, serialize_run_context
from dubanevics2026topological.results.context import REPO_ROOT
from dubanevics2026topological.threshold import functional_classification as fc

STAGE_MODULES = (
    component_persistence,
    clamped_cache,
    candidate_edge_evidence,
    candidate_tree_contrast,
)
FROZEN_NAMES = (
    "OUTPUT",
    "ANALYSIS_DIR",
    "CACHE_DIR",
    "CACHE_ROOT",
    "PENDING_PATH",
    "PAIRS_PATH",
    "DIAGNOSTICS_PATH",
    "PROVENANCE_PATH",
)
SLICE_COMMANDS = (
    "comparison-analysis",
    "candidate-edge-analysis",
    "candidate-tree-contrast",
)


@pytest.mark.parametrize("module", STAGE_MODULES, ids=lambda m: m.__name__)
def test_stage_modules_have_no_import_time_output_paths(module) -> None:
    assert [name for name in FROZEN_NAMES if hasattr(module, name)] == []


def test_paths_follow_the_context(tmp_path: Path) -> None:
    context = RunContext(root=tmp_path, output_root=tmp_path / "run_a")
    analysis = tmp_path / "run_a" / "analysis"
    paths = component_persistence.component_paths(context)
    assert paths.analysis_dir == analysis
    assert paths.pairs == analysis / "component_pairs.parquet"
    assert paths.diagnostics == analysis / "component_diagnostics.json"
    assert paths.provenance == analysis / "component_provenance.json"
    assert paths.pending == analysis / ".component-analysis-pending"
    assert clamped_cache.cache_dirs(context) == (
        analysis / "clamped_comparison",
        analysis / ".clamped-comparison-pending",
    )
    assert candidate_edge_evidence.cache_dir(context) == (
        analysis / "candidate_edge_evidence"
    )
    assert candidate_tree_contrast.cache_paths(context) == (
        analysis / "candidate_tree_contrast",
        analysis / ".candidate-tree-contrast.pending",
    )


def test_contextless_paths_resolve_the_environment_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change of the output-root variable after import is honoured: nothing is frozen."""
    monkeypatch.setenv("FITNESSTREE_OUTPUT_ROOT", str(tmp_path / "late"))
    analysis = tmp_path / "late" / "analysis"
    assert component_persistence.component_paths().pairs == (
        analysis / "component_pairs.parquet"
    )
    assert clamped_cache.cache_dirs()[0] == analysis / "clamped_comparison"
    assert candidate_edge_evidence.cache_dir() == analysis / "candidate_edge_evidence"
    assert candidate_tree_contrast.cache_paths()[0] == (
        analysis / "candidate_tree_contrast"
    )


def test_cached_loaders_read_only_under_their_context(tmp_path: Path) -> None:
    """A missing checkpoint is named under the context's root, never the default."""
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError):
        component_persistence.load_cached_component_persistence(context=context)
    with pytest.raises(FileNotFoundError) as caught:
        candidate_tree_contrast.load_cached_candidate_tree_contrast(context=context)
    assert str(tmp_path / "elsewhere") in str(caught.value)
    with pytest.raises(FileNotFoundError):
        candidate_edge_evidence.load_cached_candidate_edge_evidence(context=context)
    assert not (tmp_path / "elsewhere").exists()


def test_selection_helpers_hand_the_context_to_the_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection and threshold diagnostics no longer read the default classification."""
    seen: list[RunContext | None] = []
    summary = {"threshold": {"rounded": 0.5}}

    def spy(settings, *, tables=True, lock=True, context=None):
        seen.append(context)
        return SimpleNamespace(summary=summary), {}

    monkeypatch.setattr(fc, "load_cached_functional_classification_with_source", spy)
    monkeypatch.setattr(
        selection,
        "_declared_rule",
        lambda settings: (selection.MIXTURE_QUANTILE, {}),
    )
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "ctx")
    settings = {"analysis": {}}
    assert selection.selection_source(settings, context=context) == {
        "rule": selection.MIXTURE_QUANTILE
    }
    assert selection.selection_fingerprint(settings, context=context)[
        "rounded_threshold"
    ] == pytest.approx(0.5)
    monkeypatch.setattr(selection, "_raw_fitness", lambda *args: [0.0])
    monkeypatch.setattr(
        selection, "compute_candidate_functional_threshold", lambda *a, **k: "T"
    )
    assert selection.threshold_from_summary(settings, context=context) == "T"
    assert seen == [context, context, context]


@pytest.mark.parametrize("command", SLICE_COMMANDS)
def test_slice_commands_accept_a_context_override(command: str) -> None:
    from dubanevics2026topological import build as build_module

    assert command in build_module.CONTEXT_AWARE_COMMANDS
    assert "component-analysis" in build_module.CONTEXT_AWARE_COMMANDS


def test_build_passes_the_context_to_each_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build as build_module

    seen: dict[str, list[RunContext]] = {}

    def recorder(name: str, result):
        def fake(*args, context=None, **kwargs):
            seen.setdefault(name, []).append(context)
            return result

        return fake

    monkeypatch.setattr(
        clamped_cache,
        "load_or_compute_clamped_comparison",
        recorder("comparison", {"diagnostics": {}}),
    )
    monkeypatch.setattr(
        build_module,
        "build_comparison_tables",
        recorder("comparison-tables", []),
    )
    monkeypatch.setattr(
        candidate_edge_evidence,
        "load_or_compute_candidate_edge_evidence",
        recorder("edge", {"edges": SimpleNamespace(height=0)}),
    )
    monkeypatch.setattr(
        build_module,
        "load_or_compute_component_persistence",
        recorder("component", (SimpleNamespace(height=0), {})),
    )
    for command, key in (
        ("comparison-analysis", "comparison"),
        ("candidate-edge-analysis", "edge"),
        ("component-analysis", "component"),
    ):
        context = RunContext(root=REPO_ROOT, output_root=tmp_path / key)
        assert build_module.main([command, *serialize_run_context(context)]) == 0
        assert [c.output_root for c in seen[key]] == [tmp_path / key]
    assert [c.output_root for c in seen["comparison-tables"]] == [
        tmp_path / "comparison"
    ]


def test_context_override_is_refused_by_other_commands(tmp_path: Path) -> None:
    from dubanevics2026topological import build as build_module

    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "x")
    with pytest.raises(SystemExit):
        build_module.main(["branch-decomposition", *serialize_run_context(context)])


def test_tree_contrast_hands_the_context_to_the_barseq_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The barseq_export_dir of the context is used, not only the environment."""
    seen: list[RunContext | None] = []

    class Stop(Exception):
        pass

    def fake_load(settings, section="candidate_tree_contrast", *, context=None):
        seen.append(context)
        raise Stop

    monkeypatch.setattr(candidate_tree_contrast, "load_barseq_export", fake_load)
    context = RunContext(root=REPO_ROOT, output_root=tmp_path / "ctx")
    with pytest.raises(Stop):
        candidate_tree_contrast.compute_candidate_tree_contrast(
            {"analysis": {}}, context=context
        )
    assert seen == [context]
