"""Every stage reuses across a commit change and a dirty flip (ADR 0013, D3, Phase 2).

``tests/test_git_stamp_independence.py`` pins the digests of whole records. These
tests run each stage in a throwaway git repository and change the repository
underneath it: a new commit, then a tracked file edited (a dirty flip), then both.
A stage must report the same fingerprint and reuse its checkpoint every time, and
a reuse must not rewrite or re-stamp anything it published.

The stages that publish through ``produce_or_load`` (the main analysis and the
bootstrap) carry the helper's stamp in their records. The comparison and peak
checkpoints still publish through ``publish_checkpoint`` and carry none, but the
table stages that read their records must not care whether they do, so those
records are given a stamp (and a different one after every change) as well.
"""

from __future__ import annotations

from dubanevics2026topological.rawdata.value_digest import value_digest

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from _selection import fixed_selection
from test_analysis_stage import (
    _analysis_leaf,
    _run,
    _Workflow,
    stage,  # noqa: F401 (the fixture, used by name)
)
from test_codon_entropy_build import _write_cached_outputs
from test_git_stamp import _git, _make_repo

from dubanevics2026topological import analysis, build
from dubanevics2026topological.analysis import clamped_cache as comparison_module
from dubanevics2026topological.analysis import peak_uncertainty as peak_module
from dubanevics2026topological.analysis.git_stamp import GIT_KEY, git_stamp
from dubanevics2026topological.analysis.produce_or_load import Stage, _declared
from dubanevics2026topological.results import RunContext

STAMPS = (
    {"commit": "a" * 40, "dirty": False},
    {"commit": "b" * 40, "dirty": True},
    {"commit": None, "dirty": None},
)


# --- changing the repository under a stage ---------------------------------------


def _new_commit(root: Path) -> None:
    _git(root, "commit", "-q", "--allow-empty", "-m", "another commit")


def _dirty_flip(root: Path) -> None:
    path = root / "tracked.txt"
    path.write_text(path.read_text(encoding="utf-8") + "+", encoding="utf-8")


def _commit_and_flip(root: Path) -> None:
    _new_commit(root)
    _dirty_flip(root)


CHANGES = (
    pytest.param(_dirty_flip, id="dirty-flip"),
    pytest.param(_new_commit, id="new-commit"),
    pytest.param(_commit_and_flip, id="commit-and-dirty"),
)


def _changes_to(root: Path, change: Callable[[Path], None]) -> None:
    """Apply ``change`` and prove the stamp of ``root`` really moved."""
    before = git_stamp(root)
    change(root)
    assert git_stamp(root) != before


# --- the main analysis, published by the helper ----------------------------------


@pytest.fixture
def analysis_repo(request: pytest.FixtureRequest) -> Any:
    """The fake-workflow analysis repository of ``test_analysis_stage``."""
    return request.getfixturevalue("stage")


@pytest.mark.parametrize("change", CHANGES)
def test_the_analysis_reuses_after_a_git_change_and_rewrites_nothing(
    analysis_repo, change
) -> None:
    root = analysis_repo.root
    _git(root, "init", "-q")
    (root / "tracked.txt").write_text("v1", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "first")

    _run()
    stamped = json.loads(analysis.PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert stamped[GIT_KEY] == git_stamp(root)  # the helper stamped the run
    fingerprint = analysis_repo.fingerprint()
    figure_digest = build._analysis_fingerprint()
    record = analysis.PROVENANCE_PATH.read_bytes()
    checkpoint = (_analysis_leaf() / "provenance.json").read_bytes()

    _changes_to(root, change)
    _run()

    assert _Workflow.runs == 1  # reused, not recomputed
    assert analysis_repo.fingerprint() == fingerprint
    assert build._analysis_fingerprint() == figure_digest
    # A reuse keeps the stamp of the run that made the checkpoint: nothing is
    # rewritten, the record on disk included.
    assert analysis.PROVENANCE_PATH.read_bytes() == record
    assert (_analysis_leaf() / "provenance.json").read_bytes() == checkpoint


# --- the comparison and peak checkpoints -----------------------------------------


@pytest.fixture
def science_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repository standing in for ROOT, with the raw inputs the stages hash."""
    root = _make_repo(tmp_path / "root")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    for name in ("wt.parquet", "diff.parquet", "graph.graphml", "peaks.csv"):
        (raw / name).write_text(f"{name}-v1", encoding="utf-8")
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text(name, encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1] / "src/dubanevics2026topological"
    package_root = root / "src/dubanevics2026topological"
    analysis_dir = package_root / "analysis"
    for directory in (
        analysis_dir,
        package_root / "landscape",
        package_root / "threshold",
        package_root / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for source_dir, target_dir, names in (
        (
            source_root / "analysis",
            analysis_dir,
            (
                "clamped_cache.py",
                "describe.py",
                "graph_summary.py",
                "__init__.py",
            ),
        ),
        (
            source_root / "landscape",
            package_root / "landscape",
            ("selected_landscape.py", "selection.py", "full_population.py"),
        ),
        (
            source_root / "threshold",
            package_root / "threshold",
            ("functional_threshold.py",),
        ),
        (source_root / "tables", package_root / "tables", ("__init__.py",)),
    ):
        for name in names:
            shutil.copyfile(source_dir / name, target_dir / name)
    monkeypatch.setattr(
        comparison_module, "__file__", str(analysis_dir / "clamped_cache.py")
    )
    for module in (analysis, comparison_module, peak_module):
        monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(comparison_module, "scientific_code_paths", list)
    for module in (comparison_module, peak_module):
        monkeypatch.setattr(
            module,
            "candidate_selection_fingerprint",
            lambda _, **kwargs: {
                "rounded_threshold": -0.25,
                "source": {"rule": "median_reflected"},
            },
        )
    return root


def _science_settings(root: Path) -> dict[str, Any]:
    return {
        "analysis": {
            "dataset": {
                "fitness_wt": "data/raw/wt.parquet",
                "fitness_diff": "data/raw/diff.parquet",
            },
            "clamped": {
                "graphml": "data/raw/graph.graphml",
                "peaks": "data/raw/peaks.csv",
                "graphml_value_sha256": value_digest(
                    root / "data/raw/graph.graphml"
                ),
                "peaks_value_sha256": value_digest(root / "data/raw/peaks.csv"),
                "published_functional_threshold": -0.5,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
                "reference_topographer_commit": "abc123",
            },
        }
    }


@pytest.mark.parametrize("change", CHANGES)
def test_the_comparison_reuses_after_a_git_change(
    science_repo: Path, monkeypatch: pytest.MonkeyPatch, change
) -> None:
    root = science_repo
    settings = _science_settings(root)
    analysis_record = root / "output" / "analysis" / "provenance.json"
    analysis_record.parent.mkdir(parents=True)
    analysis_record.write_text(
        json.dumps({"fingerprint": "full", GIT_KEY: STAMPS[0]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        comparison_module,
        "analysis_provenance_path",
        lambda context=None: analysis_record,
    )
    fingerprint = comparison_module._fingerprint(settings)

    cache_dir = root / "output" / "analysis" / "clamped_comparison"
    cache_dir.mkdir(parents=True)
    (cache_dir / "provenance.json").write_text(
        json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
    )
    monkeypatch.setattr(
        comparison_module,
        "cache_dirs",
        lambda context=None: (cache_dir, cache_dir.parent / ".pending"),
    )
    # A context-less call resolves a fresh context: root it at the toy tree.
    toy_context = RunContext(root=root, output_root=root / "output")
    monkeypatch.setattr(comparison_module, "get_run_context", lambda: toy_context)
    monkeypatch.setattr(
        comparison_module, "load_or_compute_analysis", lambda **kwargs: "full"
    )
    monkeypatch.setattr(
        comparison_module,
        "_load_clamped_checkpoint",
        lambda path, **kwargs: {
            "provenance": json.loads((path / "provenance.json").read_text("utf-8")),
            "landscapes": {},
        },
    )
    monkeypatch.setattr(
        comparison_module,
        "Papkou2023Workflow",
        lambda **kwargs: pytest.fail("a git change must not recompute the comparison"),
    )
    monkeypatch.setattr(comparison_module.pl, "read_parquet", lambda path: None)

    def reused() -> bool:
        result = comparison_module.load_or_compute_clamped_comparison(settings=settings)
        return result["provenance"]["fingerprint"] == fingerprint

    assert reused()
    _changes_to(root, change)
    # The analysis record is stamped by the helper, and a re-publish elsewhere
    # would stamp it again; the comparison keys on its fingerprint field only.
    analysis_record.write_text(
        json.dumps({"fingerprint": "full", GIT_KEY: STAMPS[1]}), encoding="utf-8"
    )
    assert comparison_module._fingerprint(settings) == fingerprint
    assert reused()


@pytest.mark.parametrize("change", CHANGES)
def test_the_peak_checkpoint_reuses_after_a_git_change(
    science_repo: Path, monkeypatch: pytest.MonkeyPatch, change
) -> None:
    root = science_repo
    settings = _science_settings(root)
    fingerprint = peak_module._fingerprint(settings)
    cache_dir = root / "output" / "analysis" / "peak_uncertainty"
    cache_dir.mkdir(parents=True)
    (cache_dir / "provenance.json").write_text(
        json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
    )
    monkeypatch.setattr(
        peak_module,
        "cache_paths",
        lambda context=None: (cache_dir, cache_dir.parent / ".pending"),
    )
    monkeypatch.setattr(
        peak_module,
        "_load_peak_checkpoint",
        lambda path, **kwargs: {
            "provenance": json.loads((path / "provenance.json").read_text("utf-8")),
            "tables": {},
            "diagnostics": {},
        },
    )
    monkeypatch.setattr(
        peak_module,
        "Papkou2023Workflow",
        lambda **kwargs: pytest.fail("a git change must not recompute the peaks"),
    )
    monkeypatch.setattr(peak_module.pl, "read_parquet", lambda path: None)

    def reused() -> bool:
        result = peak_module.load_or_compute_peak_uncertainty(settings=settings)
        return result["provenance"]["fingerprint"] == fingerprint

    assert reused()
    _changes_to(root, change)
    assert peak_module._fingerprint(settings) == fingerprint
    assert reused()


# --- the table stages that read the comparison and peak records -------------------


class _TableEnv:
    """A git repository standing in for ROOT, with upstream records and a run log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.reused: list[bool] = []
        self.stages: list[Stage] = []
        self.comparison = root / "output/analysis/clamped_comparison/provenance.json"
        self.peak = root / "output/analysis/peak_uncertainty/provenance.json"
        self.analysis = root / "output/analysis/provenance.json"

    def write_upstream(self, stamp: Any = None, *, generation: int = 1) -> None:
        """Write the three upstream records, each with ``stamp`` (absent for None)."""
        extra = {} if stamp is None else {GIT_KEY: stamp}
        for path, name in (
            (self.comparison, "comparison"),
            (self.peak, "peak"),
            (self.analysis, "analysis"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "fingerprint": f"{name}-{generation}",
                        "artefacts": {"a": "0" * 64},
                        **extra,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


@pytest.fixture
def table_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _TableEnv:
    root = _make_repo(tmp_path / "root")
    env = _TableEnv(root)
    for name in (
        "analysis/biological_annotations.py",
        "analysis/describe.py",
        "analysis/codon_entropy_populations.py",
        "analysis/codon_entropy.py",
        "simplification/selected_leaf_simplification.py",
        "tables/__init__.py",
    ):
        path = root / "src/dubanevics2026topological" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n", encoding="utf-8")
    env.write_upstream()
    (root / "output/analysis/component_pairs.parquet").write_bytes(b"pairs")
    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "OUTPUT", root / "output")
    monkeypatch.setattr(build, "PROVENANCE_PATH", env.analysis)
    monkeypatch.setattr(build, "CHECKPOINT_DIR", root / "output/checkpoints/analysis")
    real_produce_or_load = build.produce_or_load

    def spy(declared: Stage, *args: Any, **kwargs: Any) -> Any:
        env.stages.append(declared)
        checkpoint = real_produce_or_load(declared, *args, **kwargs)
        env.reused.append(checkpoint.reused)
        return checkpoint

    monkeypatch.setattr(build, "produce_or_load", spy)
    return env


def _run_peak_tables(env: _TableEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    build.build_peak_tables(
        {
            "tables": {"study_clamped_peaks": pl.DataFrame({"dna_sequence": ["AAA"]})},
            "diagnostics": {"peak_count": 1},
        }
    )


def _run_comparison_tables(env: _TableEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    build.build_comparison_tables(
        {"tables": {"study_peaks": pl.DataFrame({"dna_sequence": ["AAA"]})}}
    )


def _run_biological_tables(env: _TableEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    import dubanevics2026topological.analysis.biological_annotations as biology

    def export(data: Any, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("branches", "pairs", "critical_points"):
            for extension in ("csv", "parquet"):
                (directory / f"biological_annotation_{name}.{extension}").write_text(
                    "evidence", encoding="utf-8"
                )
        (directory / "biological_annotation_counts.json").write_text("{}")

    monkeypatch.setattr(
        biology, "_build_biological_annotations", lambda data, **_: data
    )
    monkeypatch.setattr(biology, "_export_biological_annotations", export)
    build.build_biological_tables(
        {"landscapes": {"study_clamped": object(), "candidate_clamped": object()}}
    )


def _run_codon_entropy_tables(env: _TableEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = {
        "analysis": {
            "codon_entropy": {"sweep_points": 100},
            "candidate_peak_biology": {"q_value_threshold": 0.05},
        }
    }
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.peak_uncertainty.load_or_compute_peak_uncertainty",
        lambda **kwargs: {"tables": {"candidate_clamped_contrasts": pl.DataFrame()}},
    )
    monkeypatch.setattr(
        "dubanevics2026topological.simplification.selected_leaf_simplification.build_candidate_leaf_view",
        lambda *args: "view",
    )
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.codon_entropy_populations.extract_codon_entropy_populations",
        lambda **kwargs: _write_cached_outputs(Path(kwargs["output_dir"])),
    )
    monkeypatch.setattr(
        "dubanevics2026topological.landscape.selection.resolve_selection",
        fixed_selection(0.0),
    )
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.codon_entropy.process_codon_entropy_datasets",
        lambda **kwargs: None,
    )
    build.build_codon_entropy_tables(
        comparison={"landscapes": {"candidate_clamped": object()}}
    )


def _run_analysis_tables(env: _TableEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build, "ensure_analysis_checkpoint", lambda: object())
    monkeypatch.setattr(
        build, "load_or_compute_component_persistence", lambda **kwargs: (None, None)
    )
    monkeypatch.setattr(
        build,
        "export_tables",
        lambda analysis_, directory: (directory / "split-node-table.csv").write_text(
            "nodes"
        ),
    )
    monkeypatch.setattr(
        build,
        "export_component_persistence_table",
        lambda pairs, directory: (directory / "component_pairs.csv").write_text("p"),
    )
    build.build_tables()


STAGES = (
    pytest.param(_run_analysis_tables, id="analysis-tables"),
    pytest.param(_run_peak_tables, id="peak-uncertainty-tables"),
    pytest.param(_run_comparison_tables, id="clamped-comparison-tables"),
    pytest.param(_run_biological_tables, id="biological-annotation-tables"),
    pytest.param(_run_codon_entropy_tables, id="codon-entropy"),
)


@pytest.mark.parametrize("run", STAGES)
@pytest.mark.parametrize("change", CHANGES)
def test_a_table_stage_reuses_after_a_git_change(
    table_env: _TableEnv, monkeypatch: pytest.MonkeyPatch, run, change
) -> None:
    run(table_env, monkeypatch)
    assert table_env.reused == [False]
    sidecars = {
        path: path.read_bytes()
        for path in (table_env.root / "output/tables").glob(".*provenance.json")
    }
    assert sidecars

    _changes_to(table_env.root, change)
    run(table_env, monkeypatch)

    assert table_env.reused == [False, True]
    # The sidecars keep the stamp of the run that made them.
    for path, content in sidecars.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("run", STAGES)
def test_a_table_stage_reuses_whatever_stamp_its_upstream_records_carry(
    table_env: _TableEnv, monkeypatch: pytest.MonkeyPatch, run
) -> None:
    run(table_env, monkeypatch)
    for stamp in (*STAMPS, None):
        table_env.write_upstream(stamp)
        run(table_env, monkeypatch)
    assert table_env.reused == [False] + [True] * (len(STAMPS) + 1)


@pytest.mark.parametrize("run", STAGES)
def test_a_table_stage_still_rebuilds_when_its_upstream_really_changes(
    table_env: _TableEnv, monkeypatch: pytest.MonkeyPatch, run
) -> None:
    """Not vacuous: dropping the stamp is not dropping the whole record."""
    run(table_env, monkeypatch)
    table_env.write_upstream(STAMPS[0], generation=2)
    run(table_env, monkeypatch)
    assert table_env.reused == [False, False]


@pytest.mark.parametrize("run", STAGES)
def test_no_table_stage_hashes_a_provenance_file_as_an_input(
    table_env: _TableEnv, monkeypatch: pytest.MonkeyPatch, run
) -> None:
    """A provenance file's bytes would carry the stamp: keys name fields instead."""
    run(table_env, monkeypatch)
    (declared,) = table_env.stages
    assert [key for key in _declared(declared)["inputs"] if "provenance" in key] == []
