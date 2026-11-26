"""The registry and the annotation tables it reads build in order from an empty output/."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from dubanevics2026topological import build
from dubanevics2026topological.analysis import manuscript_quantities as mq
from dubanevics2026topological.figures import _quantities

COUNTS = {
    "landscape": "study_clamped",
    "stop_codon_peak_count": 3,
    "peak_denominator": 11,
    "synonymous_endpoint_peak_count": 4,
    "adjacent_saddle_peak_denominator": 10,
    "singleton_peak_count": 2,
}


def _absent() -> Any:
    raise FileNotFoundError("no checkpoint")


@pytest.fixture
def empty_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """An empty ``output/`` with every heavy step replaced; the comparison identity is mutable."""
    import dubanevics2026topological.analysis.biological_annotations as biology
    from dubanevics2026topological.analysis import clamped_cache

    output = tmp_path / "output"
    # The stage code files are the real ones, reached through the temporary root.
    (tmp_path / "src").symlink_to(build.ROOT / "src", target_is_directory=True)
    settings = build.load_settings()
    monkeypatch.setattr(build, "ROOT", tmp_path)
    state = {"comparison": "comparison-1", "export_calls": 0}
    monkeypatch.setattr(build, "OUTPUT", output)
    # Model a process started with this output for the legacy CLI guard.
    monkeypatch.setattr(build, "OUTPUT_AT_IMPORT", output)

    def comparison_fingerprint(context=None):
        if context is not None:
            assert context.output_root == output
        return state["comparison"]

    def comparison(*, context=None):
        assert context is not None and context.output_root == output
        return {
            "landscapes": {"study_clamped": object(), "candidate_clamped": object()}
        }

    monkeypatch.setattr(
        build, "_comparison_analysis_fingerprint", comparison_fingerprint
    )
    monkeypatch.setattr(
        clamped_cache,
        "load_cached_clamped_comparison",
        comparison,
    )
    monkeypatch.setattr(
        biology, "_build_biological_annotations", lambda *_args, **_kwargs: object()
    )

    def export(annotations: Any, directory: Path) -> None:
        state["export_calls"] += 1
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("branches", "pairs", "critical_points"):
            (directory / f"biological_annotation_{name}.csv").write_text("x")
        pl.DataFrame(
            {"paired_synonymous_endpoints": [True, False, True]}
        ).write_parquet(directory / "biological_annotation_pairs.parquet")
        for name in ("branches", "critical_points"):
            pl.DataFrame({"x": [1]}).write_parquet(
                directory / f"biological_annotation_{name}.parquet"
            )
        (directory / "biological_annotation_counts.json").write_text(json.dumps(COUNTS))

    monkeypatch.setattr(biology, "_export_biological_annotations", export)
    readers = {
        name: _absent
        for name in ("clamped", "component", "peak", "bootstrap", "branches", "band")
    }
    readers["band_view"] = lambda *args: _absent()
    readers["annotation_fingerprint"] = build.current_biological_annotation_fingerprint
    monkeypatch.setattr(mq, "cached_readers", lambda: dict(readers))
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    state["output"] = output
    return state


def _published_count(output: Path) -> str:
    registry = _quantities.load_manuscript_quantities(output)
    return registry.display("annotations.published.stop_codon_peak_count")


def test_the_registry_builds_from_an_empty_output_after_the_annotation_tables(
    empty_output: dict[str, Any],
) -> None:
    output = empty_output["output"]
    assert build.main(["biological-annotation-tables"]) == 0
    assert (output / "tables" / "biological_annotation_counts.json").is_file()
    assert (output / "tables" / ".biological-annotation-provenance.json").is_file()
    assert build.main(["manuscript-quantities"]) == 0
    assert _published_count(output) == "3"


def test_a_stale_annotation_export_blocks_the_registry_until_its_own_command_reruns(
    empty_output: dict[str, Any],
) -> None:
    output = empty_output["output"]
    build.main(["biological-annotation-tables"])
    build.main(["manuscript-quantities"])
    empty_output["comparison"] = "comparison-2"
    with pytest.raises(ValueError, match="just biological-annotation-tables"):
        build.main(["manuscript-quantities"])
    # The repair is the annotation command alone: it needs no registry.
    build.main(["biological-annotation-tables"])
    assert empty_output["export_calls"] == 4
    build.main(["manuscript-quantities"])
    assert _published_count(output) == "3"


def test_a_figure_build_writes_no_annotation_tables(
    empty_output: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SimpleNamespace(
        figure_id="study_tree_persistence_biology",
        inputs=frozenset({"comparison"}),
        variants=("annotated",),
        config_table="study_tree_persistence_biology",
        notebook=None,
        default=True,
        multi=False,
        render=lambda inputs, settings, variant: pytest.fail("not reached"),
    )
    monkeypatch.setattr(build, "FIGURE_SPECS", (spec,))
    monkeypatch.setattr(
        build, "load_inputs", lambda names, settings=None: {"comparison": {}}
    )
    monkeypatch.setattr(
        build,
        "_figure_dependencies",
        lambda *args: {"fixed": True},
    )
    monkeypatch.setattr(build, "FIGURE_DIR", empty_output["output"] / "figures")
    monkeypatch.setattr(build, "RECORD_DIR", empty_output["output"] / "figures/.rec")
    settings = {"figures": {"variants": ["annotated"], "formats": ["svg"]}}
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(
        build, "_comparison_analysis_fingerprint", lambda context=None: "c"
    )
    monkeypatch.setattr(
        build,
        "build_biological_tables",
        lambda *args, **kwargs: pytest.fail("build_figures must not write the tables"),
    )
    with pytest.raises(pytest.fail.Exception, match="not reached"):
        build.build_figures()


# Analysis modules the registry reaches that carry no registry value, so they are
# neither required in the stage code nor followed. The rule: a module is excluded when it
# only stores, locks or fingerprints (checkpoint, produce_or_load), or when its output is an
# upstream stage that the registry stage's upstream identities already key on and the
# registry reads from disk (sequence_growth_rates). Importing only ``load_cached*`` names is
# the same rule applied per import.
_NOT_REGISTRY_CODE = frozenset(
    {
        "checkpoint.py",
        "produce_or_load.py",
        "sequence_growth_rates.py",
        "selection.py",
        "functional_threshold.py",
        "selected_landscape.py",
        "full_population.py",
    }
)


def _computing_imports(path: Path) -> set[str]:
    """Analysis modules imported for more than a cached loader (direct imports only)."""
    import ast

    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module is None:
                modules.update(
                    f"{alias.name}.py"
                    for alias in node.names
                    if alias.name
                    in {
                        "bootstrap_chunks",
                        "checkpoint_identity",
                    }
                )
            elif node.level == 1 and node.module:
                module_name = node.module
                names = {alias.name for alias in node.names}
                if any(not name.startswith("load_cached") for name in names):
                    modules.add(f"{module_name}.py")
            elif (
                node.level == 2
                and node.module
                and node.module.startswith("simplification.")
            ):
                names = {alias.name for alias in node.names}
                if any(not name.startswith("load_cached") for name in names):
                    modules.add(f"{node.module.split('.')[-1]}.py")
            elif node.module == "dubanevics2026topological.simplification":
                modules.update(
                    f"{alias.name}.py"
                    for alias in node.names
                    if alias.name
                    in {
                        "bootstrap_smoothed_tree",
                        "selected_leaf_simplification",
                    }
                )
    return modules


def _registry_code_closure(analysis: Path, roots: tuple[str, ...]) -> set[str]:
    """The roots and every computing module they reach through package-internal imports."""
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in seen or name in _NOT_REGISTRY_CODE:
            continue
        path = analysis.parent / "simplification" / name
        if not path.is_file():
            path = analysis / name
        if not path.is_file():
            continue
        seen.add(name)
        pending.extend(_computing_imports(path))
    return seen


def test_the_registry_stage_code_covers_every_module_that_computes_a_value() -> None:
    analysis = build.ROOT / "src/dubanevics2026topological/analysis"
    needed = _registry_code_closure(
        analysis, ("manuscript_quantities.py", "band_view_ridges.py")
    )
    # Spot checks that the walk is not vacuous and follows imports transitively:
    # build_candidate_tree (candidate_tree_contrast) pulls in the last four.
    assert {
        "band_view_ridges.py",
        "biological_annotations.py",
        "component_persistence.py",
        "bootstrap_smoothed_tree.py",
        "barseq_inputs.py",
        "selected_leaf_simplification.py",
    } <= needed
    assert not needed & _NOT_REGISTRY_CODE
    stage, _ = build._manuscript_quantities_stage(build.load_settings(), mq.Sources({}))
    listed = {Path(path).name for path in stage.code}
    assert needed <= listed, sorted(needed - listed)
    simplification_dir = build.ROOT / "src/dubanevics2026topological/simplification"
    resolved_stage_code = {Path(path).resolve() for path in stage.code}
    assert {
        (simplification_dir / name).resolve()
        for name in (
            "__init__.py",
            "bootstrap_smoothed_tree.py",
            "bootstrap_chunks.py",
            "checkpoint_identity.py",
            "identities.py",
            "selected_leaf_simplification.py",
        )
    } <= resolved_stage_code
    threshold_dir = build.ROOT / "src/dubanevics2026topological/threshold"
    resolved_code = {Path(path).resolve() for path in stage.code}
    assert {
        (threshold_dir / name).resolve() for name in build.REGISTRY_THRESHOLD_MODULES
    } <= resolved_code
    landscape_dir = build.ROOT / "src/dubanevics2026topological/landscape"
    assert {
        (landscape_dir / name).resolve() for name in build.REGISTRY_LANDSCAPE_MODULES
    } <= resolved_code
