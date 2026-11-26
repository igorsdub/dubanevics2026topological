"""Registry contract: spec validity, input loading and variant resolution."""

from __future__ import annotations

import ast
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from _fake_figures import write_renderer_sources

from dubanevics2026topological import build
from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint
from dubanevics2026topological.figures import FIGURE_IDS, registry

PREVIOUS_FIGURE_IDS = (
    "candidate_edge_evidence",
    "tree_comparison",
    "full_tree_histogram",
    "published_tree_histogram",
    "candidate_tree_histogram",
    "linked_tree_persistence",
    "full_tree_persistence",
    "full_tree_persistence_vertical_histogram",
    "full_tree_fitness_landmarks",
    "full_join_tree",
    "study_tree_persistence_metrics",
    "study_tree_persistence_biology",
    "candidate_tree_bootstrap_biology",
    "bootstrap_tree_histogram",
    "bootstrap_codon_entropy",
    "bootstrap_global_stability",
    "bootstrap_band_quantile_scan",
    "bootstrap_band_quantile_scan_seeds",
    "bootstrap_band_quantile_scan_grid",
    "peak_uncertainty",
    "published_leaf_persistence",
    "candidate_leaf_persistence",
    "published_leaf_swarm",
    "candidate_leaf_swarm",
    "published_leaf_persistence_band",
    "candidate_leaf_persistence_band",
    "published_leaf_swarm_band",
    "candidate_leaf_swarm_band",
    "published_fdr_leaf_swarm",
    "candidate_fdr_leaf_swarm",
    "graph_tree_persistence",
    "peak_rank",
    "peak_rank_annotated",
    "peak_rank_histogram",
    "ridge_entropy",
    "graph_codon_entropy",
    "multiple_testing_comparison",
    "functional_counts_vs_q",
    "functional_counts_vs_q_bh",
    "functional_counts_vs_q_bonferroni",
    "papkou_agreement",
    "fitness_se_correlation",
    "fitness_vs_pooled_se",
    "branch_vertex_counts_2x2",
    "peak_pair_matrices",
    "peak_pair_hamming_fitness_scatter",
    "peak_pair_direct_via_saddle_scatter",
)


# The story views (Decision 10) are multi-output families registered after the earlier figures.
STORY_VIEW_IDS = ("tree_persistence", "bootstrap_band", "tree_comparison_views")


def test_figure_ids_derive_from_specs_and_preserve_existing_ids() -> None:
    assert FIGURE_IDS == tuple(s.figure_id for s in registry.FIGURE_SPECS)
    assert FIGURE_IDS == (
        "reflected_fdr_trees",
        "bootstrap_four_graphs",
        *PREVIOUS_FIGURE_IDS,
        *STORY_VIEW_IDS,
    )


def test_spec_inputs_and_variants_are_valid() -> None:
    for spec in registry.FIGURE_SPECS:
        assert spec.inputs and spec.inputs <= registry.INPUT_NAMES
        assert spec.variants and spec.variants <= registry.VARIANT_NAMES
        expected = "tree_comparison" if spec.figure_id == "reflected_fdr_trees" else spec.figure_id
        assert spec.config_table == expected


def test_spec_notebooks_exist_and_config_tables_are_configured() -> None:
    import tomllib

    root = Path(build.ROOT)
    tables = tomllib.loads((root / "config/figures.toml").read_text("utf-8"))
    for spec in registry.FIGURE_SPECS:
        assert spec.config_table in tables["figures"]
        if spec.notebook is not None:
            assert (root / spec.notebook).is_file()
    assert set(tables["figures"]) == {s.config_table for s in registry.FIGURE_SPECS}


def test_get_spec_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown figure ID"):
        registry.get_spec("nope")


def test_load_inputs_calls_matching_readers_once(monkeypatch) -> None:
    calls: list[str] = []

    def reader(name):
        return lambda *args: calls.append(name) or name

    monkeypatch.setattr(build, "load_cached_analysis", reader("split_analysis"))
    monkeypatch.setattr(
        build, "load_cached_component_persistence", reader("component_persistence")
    )
    monkeypatch.setattr(build, "load_full_network_fitness", reader("full_fitness"))
    monkeypatch.setattr(
        build, "load_saved_biological_annotations", reader("annotations")
    )
    monkeypatch.setattr(
        build, "load_cached_leading_saddle_band", reader("leading_saddle_band")
    )
    monkeypatch.setattr(
        "dubanevics2026topological.figures.join_tree.load_full_network",
        reader("full_network"),
    )
    monkeypatch.setattr(
        build,
        "load_settings",
        lambda: {
            "analysis": {
                "graph_tree_persistence": {"marker": "manifold-settings"},
            },
            "figures": {"rendering": {"tree_horizontal_layout": "leaf_order"}},
        },
    )
    monkeypatch.setattr(
        "dubanevics2026topological.figures._quantities.load_manuscript_quantities",
        lambda output: "manuscript_quantities",
    )
    manifold_calls: list[object] = []
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.manifold_graph_tree.build_manifold_graph_tree",
        lambda config, **kwargs: (
            manifold_calls.append((config, kwargs)) or "graph_tree_persistence"
        ),
    )
    for module, attr, name in (
        ("clamped_cache", "load_cached_clamped_comparison", "comparison"),
        ("peak_uncertainty", "load_cached_peak_uncertainty", "peak"),
        (
            "sequence_growth_rates",
            "load_cached_sequence_growth_rates",
            "growth_rates",
        ),
        (
            "candidate_edge_evidence",
            "load_cached_candidate_edge_evidence",
            "edge_evidence",
        ),
        (
            "bootstrap_smoothed_tree",
            "load_cached_bootstrap_smoothed_tree",
            "bootstrap_smoothed",
        ),
    ):
        package = (
            "simplification" if module == "bootstrap_smoothed_tree" else "analysis"
        )
        monkeypatch.setattr(
            importlib.import_module(f"dubanevics2026topological.{package}.{module}"),
            attr,
            reader(name),
        )
    # fdr_q_scan is a stage result read from disk, not one of the cached readers.
    stage_results = {
        "fdr_q_scan",
        "band_quantile_scan_grid",
        "four_graph_comparisons",
        "peak_pair_distances",
    }
    # published_fitness reads data/raw through load_fitness_wt; it is not a cached reader.
    raw_tables = {"published_fitness"}
    ordinary = (
        registry.INPUT_NAMES
        - registry.SAVED_PRODUCT_INPUTS
        - stage_results
        - raw_tables
    )
    names = sorted(ordinary) + ["comparison"]
    result = build.load_inputs(names)
    assert result == {name: name for name in ordinary}
    assert sorted(calls) == sorted(
        ordinary - {"graph_tree_persistence", "manuscript_quantities"}
    )
    assert manifold_calls == []  # saved product, never built by load_inputs


def test_load_inputs_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown figure input"):
        build.load_inputs(["bogus"])


@pytest.fixture
def variant_build(tmp_path, monkeypatch):
    rendered: list[tuple[str, str]] = []
    peak = registry.get_spec("peak_uncertainty")
    # No real figure but the full-graph one allows both variants and its build
    # needs cached analysis files, so a synthetic two-variant spec stands in.
    two_variants = replace(
        peak,
        figure_id="two_variants",
        config_table="two_variants",
        variants=frozenset({"vanilla", "annotated"}),
    )
    specs = tuple(
        replace(
            replace(spec, inputs=frozenset(), notebook=None),
            render=lambda inputs, settings, variant, spec=spec: (
                rendered.append((spec.figure_id, variant)) or _Chart()
            ),
        )
        for spec in (two_variants, peak)
    )
    settings = {
        "figures": {
            "formats": ["svg"],
            "variants": ["vanilla", "annotated"],
            "figures": {spec.figure_id: {} for spec in specs},
        }
    }
    monkeypatch.setattr(build, "FIGURE_SPECS", specs)
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(build, "load_inputs", lambda names, settings=None: {})
    monkeypatch.setattr(
        build,
        "load_figure_inputs",
        lambda specs, settings=None, context=None: {
            "saved_figure_inputs": {"product_ids": {"pooled_scatter": "dummy"}}
        },
    )
    monkeypatch.setattr(build, "_figure_dependencies", lambda *args: {"fake": args[:2]})
    monkeypatch.setattr(build, "_analysis_fingerprint", lambda: "a")
    monkeypatch.setattr(build, "_peak_analysis_fingerprint", lambda context=None: "p")
    monkeypatch.setattr(build, "_pooled_scatter_dependency", lambda *args: "g")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "figures/.records")
    return rendered


class _Chart:
    def save(self, path, *, format, **kwargs):
        Path(path).write_text(format)


def test_empty_variant_intersection_raises_for_explicit_figure(variant_build) -> None:
    with pytest.raises(ValueError, match="canonical vanilla"):
        build.build_figures(("peak_uncertainty",), variants=("annotated",))
    assert variant_build == []


def test_partial_intersection_renders_only_allowed_variants(variant_build) -> None:
    build.build_figures(variants=("vanilla", "annotated"))
    assert sorted(variant_build) == [
        ("peak_uncertainty", "vanilla"),
        ("two_variants", "annotated"),
        ("two_variants", "vanilla"),
    ]
    variant_build.clear()
    build.build_figures(variants=("annotated",), force=True)
    assert variant_build == [("two_variants", "annotated")]


def test_render_output_can_be_outside_repository(variant_build, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(build, "ROOT", tmp_path / "repo")
    outputs = build.build_figures(("peak_uncertainty",), variants=("vanilla",))
    assert len(outputs) == 1 and outputs[0].is_file()
    assert str(outputs[0]) in capsys.readouterr().out


def test_keep_going_reports_input_failure_and_renders_other_figure(
    variant_build, tmp_path, monkeypatch, capsys
) -> None:
    import json

    from dubanevics2026topological.results import RunContext

    def load_inputs(specs, settings=None, context=None):
        names = [spec.figure_id for spec in specs]
        if "two_variants" in names:
            raise RuntimeError("missing input")
        return {"saved_figure_inputs": {"product_ids": {"pooled_scatter": "dummy"}}}

    monkeypatch.setattr(build, "load_figure_inputs", load_inputs)
    monkeypatch.setattr(build, "register_font_directory", lambda **kwargs: None)
    context = RunContext(root=tmp_path, output_root=tmp_path, env={})
    monkeypatch.setattr(build, "OUTPUT_AT_IMPORT", context.output_root)
    monkeypatch.setattr(build, "ANALYSIS_VARIANT_AT_IMPORT", context.variant_name)
    args = ["figures", "--figure", "two_variants", "--figure", "peak_uncertainty", "--variant", "vanilla"]
    assert build.main([*args, "--keep-going"], context=context) == 1
    assert variant_build == [("peak_uncertainty", "vanilla")]
    assert (tmp_path / "figures/canonical/peak_uncertainty.svg").is_file()
    report = json.loads((tmp_path / "figures/figure-failures.json").read_text())
    shared, failure = report["failures"]
    assert shared["scope"] == "shared_inputs"
    assert shared["exception_type"] == "RuntimeError"
    assert "missing input" in shared["traceback"]
    assert (failure["figure_id"], failure["variant"]) == ("two_variants", "vanilla")
    assert (failure["exception_type"], failure["message"]) == ("RuntimeError", "missing input")
    assert "RuntimeError: missing input" in failure["traceback"]
    output = capsys.readouterr().out
    assert "FAILED two_variants (vanilla): RuntimeError: missing input" in output
    assert "FAILED shared figure inputs" in output
    record = json.loads((tmp_path / "figures/.records/peak_uncertainty.svg.json").read_text())
    assert record["keep_going_provenance"] == "incomplete: shared input loading failed"

    variant_build.clear()
    with pytest.raises(RuntimeError, match="missing input"):
        build.main(args, context=context)
    assert variant_build == []


@pytest.mark.parametrize("failure_stage", ["render", "second_format"])
def test_keep_going_reports_render_and_partial_format_failures(
    variant_build, tmp_path, monkeypatch, failure_stage
) -> None:
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    first, second = build.FIGURE_SPECS
    if failure_stage == "render":
        def broken(*args):
            raise ValueError("render failed")
        first = replace(first, render=broken)
        monkeypatch.setattr(build, "FIGURE_SPECS", (first, second))
    else:
        original_render = build._render_one

        def fail_png(chart, target, format_name, **kwargs):
            if target.name == "two_variants_vanilla.png":
                raise OSError("png failed")
            return original_render(chart, target, format_name, **kwargs)

        monkeypatch.setattr(build, "_render_one", fail_png)
    context = SimpleNamespace(output_root=tmp_path, variant_name=None)
    assert not build.build_figures_keep_going(
        ["two_variants", "peak_uncertainty"], variants=["vanilla"],
        formats=["svg", "png"], context=context,
    )
    failures = json.loads((tmp_path / "figures/figure-failures.json").read_text())["failures"]
    assert len(failures) == 1
    assert failures[0]["figure_id"] == "two_variants"
    assert failures[0]["exception_type"] == ("ValueError" if failure_stage == "render" else "OSError")
    assert "failed" in failures[0]["traceback"]
    assert (tmp_path / "figures/canonical/peak_uncertainty.svg").is_file()
    if failure_stage == "second_format":
        assert (tmp_path / "figures/canonical/two_variants_vanilla.svg").is_file()
        assert not (tmp_path / "figures/.records/two_variants_vanilla.png.json").exists()


def test_keep_going_invalidates_old_record_after_render_failure(
    variant_build, tmp_path, monkeypatch
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    context = SimpleNamespace(output_root=tmp_path, variant_name=None)
    build.build_figures(["two_variants"], variants=["vanilla"], context=context)
    record = tmp_path / "figures/.records/two_variants_vanilla.svg.json"
    assert record.is_file()
    first, second = build.FIGURE_SPECS

    def broken(*args):
        raise ValueError("failed rerender")

    monkeypatch.setattr(build, "FIGURE_SPECS", (replace(first, render=broken), second))
    assert not build.build_figures_keep_going(
        ["two_variants"], variants=["vanilla"], force=True, context=context
    )
    assert not record.exists()


def test_keep_going_preserves_full_plan_product_records(variant_build, tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    def load_inputs(specs, settings=None, context=None):
        names = {spec.figure_id for spec in specs}
        ids = {name: name + "-product" for name in names}
        ids["pooled_scatter"] = "pooled-product"
        return {"saved_figure_inputs": {"product_ids": ids}}

    first, peak = build.FIGURE_SPECS
    monkeypatch.setattr(
        build, "FIGURE_SPECS", (first, replace(peak, inputs=frozenset({"pooled_scatter"})))
    )
    monkeypatch.setattr(build, "load_figure_inputs", load_inputs)
    context = SimpleNamespace(output_root=tmp_path, variant_name=None)
    selected = ["two_variants", "peak_uncertainty"]
    build.build_figures(selected, variants=["vanilla"], force=True, context=context)
    records = tmp_path / "figures/.records"
    baseline = {path.name: json.loads(path.read_text()) for path in records.glob("*.json")}
    assert len(baseline) == 2
    assert baseline["peak_uncertainty.svg.json"]["saved_products"] == {
        "two_variants": "two_variants-product", "peak_uncertainty": "peak_uncertainty-product",
        "pooled_scatter": "pooled-product",
    }
    assert build.build_figures_keep_going(selected, variants=["vanilla"], force=True, context=context)
    assert {path.name: json.loads(path.read_text()) for path in records.glob("*.json")} == baseline


def test_spec_notebooks_match_config_notebook_keys() -> None:
    import tomllib

    tables = tomllib.loads(
        (Path(build.ROOT) / "config/figures.toml").read_text("utf-8")
    )["figures"]
    for spec in registry.FIGURE_SPECS:
        configured = tables[spec.config_table].get("notebook")
        if configured is not None and spec.figure_id != "reflected_fdr_trees":
            assert spec.notebook == Path(configured)


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    figure_dir = tmp_path / "src/dubanevics2026topological/figures"
    figure_dir.mkdir(parents=True)
    write_renderer_sources(figure_dir)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    # Both figures read their labels from the registry, whose record the dependency
    # fingerprint joins; the output tree is not under test here.
    monkeypatch.setattr(
        "dubanevics2026topological.figures._quantities.registry_fingerprint",
        lambda output: "registry-fingerprint",
    )
    settings = {
        "figures": {
            "figures": {
                "linked_tree_persistence": {},
                "peak_uncertainty": {},
            }
        }
    }
    return figure_dir, settings


def _fingerprint(settings, figure_id):
    return build._figure_dependencies(figure_id, "vanilla", settings)["figure_files"]


def _shared_config(settings, figure_id="linked_tree_persistence"):
    return build._figure_dependencies(figure_id, "vanilla", settings)["shared_config"]


def test_exploratory_appearance_is_not_part_of_any_figures_dependencies(
    source_tree,
) -> None:
    """Editing a notebook-only table must not re-render a publication figure."""
    _, settings = source_tree
    settings["figures"].update(
        {
            "fitness_domain": [-1.0, 1.0],
            "rendering": {"font_size": 15},
            "formats": ["png"],
            "variants": ["vanilla"],
            "exploratory": {"branch_decomposition": {"histogram_color": "#111111"}},
            "layout_exploration": {"dpi": 200, "edge_color": "#8c8c8c"},
        }
    )
    before = {
        figure_id: build._figure_dependencies(figure_id, "vanilla", settings)
        for figure_id in (
            "linked_tree_persistence",
            "peak_uncertainty",
        )
    }
    assert set(before["peak_uncertainty"]["shared_config"]) == {
        "fitness_domain",
        "rendering",
    }
    settings["figures"]["exploratory"]["branch_decomposition"]["histogram_color"] = (
        "#222222"
    )
    settings["figures"]["exploratory"]["another_notebook"] = {"panel_width": 9}
    settings["figures"]["layout_exploration"]["dpi"] = 300
    for figure_id, dependencies in before.items():
        assert (
            build._figure_dependencies(figure_id, "vanilla", settings) == dependencies
        )


def test_bootstrap_exclusion_table_is_not_part_of_any_figures_dependencies(
    source_tree,
) -> None:
    _, settings = source_tree
    figure_ids = ("linked_tree_persistence", "peak_uncertainty")
    before = {
        figure_id: build._figure_dependencies(figure_id, "vanilla", settings)
        for figure_id in figure_ids
    }
    settings["figures"]["bootstrap_identity_exclusions"] = {
        "bootstrap_tree_histogram": "one reason"
    }
    settings["figures"]["bootstrap_identity_exclusions"]["bootstrap_tree_histogram"] = (
        "edited reason"
    )
    for figure_id in figure_ids:
        assert (
            build._figure_dependencies(figure_id, "vanilla", settings)
            == before[figure_id]
        )


def test_shared_appearance_still_reaches_every_figures_dependencies(
    source_tree,
) -> None:
    _, settings = source_tree
    settings["figures"]["rendering"] = {"font_size": 15}
    before = _shared_config(settings)
    settings["figures"]["rendering"] = {"font_size": 16}
    assert _shared_config(settings) != before


def test_shared_module_edit_invalidates_dependent_figures(source_tree) -> None:
    figure_dir, settings = source_tree
    for figure_id in (
        "linked_tree_persistence",
        "peak_uncertainty",
    ):
        before = _fingerprint(settings, figure_id)
        (figure_dir / "_branch_count_style.py").write_text("# edited\n")
        assert _fingerprint(settings, figure_id) != before
        (figure_dir / "_branch_count_style.py").write_text("# _branch_count_style.py\n")


def test_dedicated_renderer_edit_leaves_unrelated_figure_current(source_tree) -> None:
    figure_dir, settings = source_tree
    raw = _fingerprint(settings, "linked_tree_persistence")
    peak = _fingerprint(settings, "peak_uncertainty")
    (figure_dir / "peak_uncertainty.py").write_text("# edited\n")
    assert _fingerprint(settings, "linked_tree_persistence") == raw
    assert _fingerprint(settings, "peak_uncertainty") != peak


def test_multi_figure_selection_raises_when_one_has_no_variant(variant_build) -> None:
    with pytest.raises(ValueError, match="canonical vanilla"):
        build.build_figures(
            ("two_variants", "peak_uncertainty"),
            variants=("annotated",),
        )
    assert variant_build == []


def test_real_spec_render_reaches_real_builder() -> None:
    from test_peak_uncertainty_figure import _panels, _settings

    spec = registry.get_spec("peak_uncertainty")
    assert spec.inputs == {"peak_panels"}
    panels = _panels()
    # The figure receives the four saved variant panels (nothing is read or computed at render).
    inputs = {"saved_figure_inputs": {"peak_panels": panels}}
    chart = spec.render(inputs, _settings(), "vanilla")
    rendered = chart.to_dict()
    assert len(rendered["hconcat"]) == 2
    assert all(len(column["vconcat"]) == 2 for column in rendered["hconcat"])
    # Each variant's own table is plotted.
    for variant in panels:
        plotted = rendered["datasets"][f"{variant}_measurements"]
        assert {row["dna_sequence"] for row in plotted} == {"wt", f"peak_{variant}"}
    # The payload a notebook composes is the one the figure renders.
    payload = registry.compose_payload(spec, inputs, _settings())
    assert payload == {"panels": panels}


def test_every_spec_maps_to_an_existing_renderer_file() -> None:
    figure_dir = Path(registry.__file__).parent
    for spec in registry.FIGURE_SPECS:
        assert (figure_dir / registry.renderer_file(spec)).is_file()
    for name in registry.SHARED_RENDERER_FILES:
        assert (figure_dir / name).is_file()


def test_unmapped_spec_has_no_renderer_file() -> None:
    spec = registry.FigureSpec(
        "unmapped", frozenset(), frozenset(), "unmapped", None, lambda *_: None
    )
    with pytest.raises(ValueError, match="No renderer file"):
        registry.renderer_file(spec)


def test_graph_tree_persistence_spec_declares_synthetic_input_only() -> None:
    spec = registry.get_spec("graph_tree_persistence")
    assert spec.inputs == {"graph_tree_persistence"}
    assert spec.variants == {"vanilla"}
    assert spec.config_table == "graph_tree_persistence"
    assert spec.notebook == Path("notebooks/graph_tree_persistence.py")


def test_graph_tree_persistence_spec_renders_from_settings() -> None:
    from dubanevics2026topological.analysis import load_settings
    from dubanevics2026topological.analysis.manifold_graph_tree import (
        build_manifold_graph_tree,
    )

    settings = load_settings()
    settings["figures"]["figures"]["graph_tree_persistence"][
        "presentation_surface_resolution"
    ] = 8
    construction = build_manifold_graph_tree(
        settings["analysis"]["graph_tree_persistence"]
    )
    chart = registry.get_spec("graph_tree_persistence").render(
        {"graph_tree_persistence": construction}, settings, "vanilla"
    )
    # Peak/valley panels + shared axis furniture + one-row persistence_legend.
    assert len(chart.to_dict(validate=True)["vconcat"]) == 5


def test_manifold_figure_records_own_dependencies_not_split_analysis(
    tmp_path, monkeypatch
) -> None:
    import json

    write_renderer_sources(tmp_path / "src/dubanevics2026topological/figures")
    analysis_dir = tmp_path / "src/dubanevics2026topological/analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "manifold_graph_tree.py").write_text("# source\n")
    spec = next(
        replace(spec, inputs=frozenset(), notebook=None, render=lambda *a: _Chart())
        for spec in registry.FIGURE_SPECS
        if spec.figure_id == "graph_tree_persistence"
    )
    settings = {
        "analysis": {"graph_tree_persistence": {"grid_rows": 6}},
        "figures": {
            "formats": ["svg"],
            "variants": ["vanilla"],
            "figures": {"graph_tree_persistence": {}},
        },
    }

    def forbidden():
        raise AssertionError("split analysis fingerprint must not be used")

    monkeypatch.setattr(build, "FIGURE_SPECS", (spec,))
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(build, "load_inputs", lambda names, settings=None: {})
    monkeypatch.setattr(build, "_analysis_fingerprint", forbidden)
    monkeypatch.setattr(build, "_comparison_analysis_fingerprint", forbidden)
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "figures/.records")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    outputs = build.build_figures()
    assert {path.stem for path in outputs} == {"graph_tree_persistence"}
    manifold = json.loads(
        (tmp_path / "figures/.records/graph_tree_persistence.svg.json").read_text()
    )
    assert manifold["graph_tree_persistence"] == {"grid_rows": 6}
    assert "manifold_source" in manifold and "analysis" not in manifold


def test_tree_comparison_record_tracks_its_own_config(tmp_path, monkeypatch) -> None:
    import json
    from dataclasses import replace

    from dubanevics2026topological import build

    spec = next(
        replace(s, inputs=frozenset(), notebook=None, render=lambda *a: _Chart())
        for s in registry.FIGURE_SPECS
        if s.figure_id == "tree_comparison"
    )

    write_renderer_sources(tmp_path / "src/dubanevics2026topological/figures")

    def record_for(fraction: float) -> dict:
        settings = {
            "analysis": {},
            "figures": {
                "formats": ["svg"],
                "variants": ["annotated"],
                "figures": {
                    "tree_comparison": {"shared_colorbar_length": fraction},
                },
            },
        }
        monkeypatch.setattr(build, "FIGURE_SPECS", (spec,))
        monkeypatch.setattr(build, "load_settings", lambda: settings)
        monkeypatch.setattr(build, "load_inputs", lambda names, settings=None: {})
        monkeypatch.setattr(build, "_analysis_fingerprint", lambda: {"a": 1})
        monkeypatch.setattr(
            build, "_comparison_analysis_fingerprint", lambda context=None: {"a": 1}
        )
        monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "figures")
        monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "figures/.records")
        monkeypatch.setattr(build, "ROOT", tmp_path)
        monkeypatch.setattr(build, "OUTPUT", tmp_path)
        build.build_figures()
        (path,) = (tmp_path / "figures/.records").glob(
            "tree_comparison*.json"
        )
        return json.loads(path.read_text())

    first = record_for(0.5)
    # The comparison figure reads only its own table, so that is what keys it.
    assert not any(key.endswith("_tree_histogram_config") for key in first)
    assert record_for(0.6) != first


def test_tree_comparison_record_records_saved_products_with_layout_ids(
    tmp_path, monkeypatch
) -> None:
    import json
    from dataclasses import replace

    from dubanevics2026topological import build

    spec = next(
        replace(
            s,
            inputs=frozenset({"comparison"}),
            notebook=None,
            render=lambda *a: _Chart(),
        )
        for s in registry.FIGURE_SPECS
        if s.figure_id == "tree_comparison"
    )

    write_renderer_sources(tmp_path / "src/dubanevics2026topological/figures")

    def record_for() -> dict:
        settings = {
            "analysis": {},
            "figures": {
                "formats": ["svg"],
                "variants": ["annotated"],
                "figures": {
                    "tree_comparison": {},
                },
            },
        }
        monkeypatch.setattr(build, "FIGURE_SPECS", (spec,))
        monkeypatch.setattr(build, "load_settings", lambda: settings)
        monkeypatch.setattr(
            build, "load_inputs", lambda names, settings=None: {"comparison": object()}
        )
        monkeypatch.setattr(
            build, "selection_source", lambda settings: {"rule": "median_reflected"}
        )
        monkeypatch.setattr(
            "dubanevics2026topological.build.saved_figure_inputs.load_saved_figure_inputs",
            lambda context, selection_identity, include_band, **kwargs: {
                "product_ids": {"layout": "LAYOUTID", "band_layout": "BANDLAYOUTID"},
            },
        )
        monkeypatch.setattr(build, "_analysis_fingerprint", lambda: {"a": 1})
        monkeypatch.setattr(
            build, "_comparison_analysis_fingerprint", lambda context=None: {"a": 1}
        )
        monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "figures")
        monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "figures/.records")
        monkeypatch.setattr(build, "ROOT", tmp_path)
        monkeypatch.setattr(build, "OUTPUT", tmp_path)
        build.build_figures()
        (path,) = (tmp_path / "figures/.records").glob(
            "tree_comparison*.json"
        )
        return json.loads(path.read_text())

    rec = record_for()
    assert rec.get("saved_products") == {
        "layout": "LAYOUTID",
        "band_layout": "BANDLAYOUTID",
    }


BOOTSTRAP_FIGURES = {
    "candidate_tree_bootstrap_biology",
    "bootstrap_tree_histogram",
    "bootstrap_codon_entropy",
    "bootstrap_global_stability",
    "bootstrap_band_quantile_scan",
    "bootstrap_band_quantile_scan_seeds",
}
BOOTSTRAP_INPUTS = {"bootstrap_smoothed"}


class _StopLoading(Exception):
    pass


def _inputs_requested(monkeypatch, selected) -> set[str]:
    requested: list[str] = []

    def load(names, settings=None):
        requested.extend(names)
        raise _StopLoading

    monkeypatch.setattr(
        build,
        "load_settings",
        lambda: {"figures": {"variants": ["annotated", "vanilla"], "formats": ["svg"]}},
    )
    monkeypatch.setattr(build, "load_inputs", load)
    with pytest.raises(_StopLoading):
        build.build_figures(selected)
    return set(requested)


def test_only_the_unnumbered_bootstrap_figures_are_opt_in() -> None:
    # The manuscript bootstrap figures are default; the bootstrap checkpoint they read is cached.
    opt_in = {f for f in BOOTSTRAP_FIGURES if not registry.get_spec(f).default}
    assert opt_in == {"candidate_tree_bootstrap_biology", "bootstrap_band_quantile_scan"}


def test_default_selection_reads_the_cached_bootstrap(monkeypatch) -> None:
    assert _inputs_requested(monkeypatch, None) & BOOTSTRAP_INPUTS


@pytest.mark.parametrize("figure_id", sorted(BOOTSTRAP_FIGURES))
def test_explicit_figure_selects_opt_in_figure(monkeypatch, figure_id) -> None:
    needed = _inputs_requested(monkeypatch, [figure_id])
    assert needed & BOOTSTRAP_INPUTS


def test_bootstrap_biology_fingerprint_tracks_the_shared_assembler(
    source_tree,
) -> None:
    figure_dir, settings = source_tree
    figure_id = "candidate_tree_bootstrap_biology"
    settings["figures"]["figures"][figure_id] = {}
    before = _fingerprint(settings, figure_id)
    (figure_dir / "candidate_tree_biology.py").write_text("# edited\n")
    assert _fingerprint(settings, figure_id) != before


@pytest.mark.parametrize(
    "missing", ["df_core", "entropy", "candidate_leaf_view", "metadata"]
)
def test_codon_renderer_raises_instead_of_reading_output(monkeypatch, missing) -> None:
    from dubanevics2026topological.analysis import peak_uncertainty

    def boom(*args, **kwargs):
        raise AssertionError("renderer touched output/")

    monkeypatch.setattr(peak_uncertainty, "load_cached_peak_uncertainty", boom)
    payload = {
        "df_core": None,
        "df_kem": None,
        "df_tcl": None,
        "metadata": {},
        "entropy": {"core": None, "kem": None, "tcl": None},
        "candidate_leaf_view": None,
    }
    del payload[missing]
    spec = registry.get_spec("bootstrap_codon_entropy")
    with pytest.raises(KeyError, match="compose_payload"):
        registry.render_payload(spec, payload, {"figures": {}}, "annotated")


def test_codon_provenance_changes_with_q_value_threshold(monkeypatch, tmp_path) -> None:
    for name in ("comparison", "peak"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"fingerprint": name}))
    monkeypatch.setattr(
        build,
        "_comparison_provenance_path",
        lambda context=None: tmp_path / "comparison.json",
    )
    monkeypatch.setattr(
        build, "_peak_provenance_path", lambda context=None: tmp_path / "peak.json"
    )
    monkeypatch.setattr(build, "ROOT", tmp_path)
    for relative in (
        "codon_entropy_populations.py",
        "codon_entropy.py",
    ):
        path = tmp_path / "src/dubanevics2026topological/analysis" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    moved_source = (
        tmp_path
        / "src/dubanevics2026topological/simplification/selected_leaf_simplification.py"
    )
    moved_source.parent.mkdir(parents=True, exist_ok=True)
    moved_source.write_text("selected_leaf_simplification.py")

    def provenance(threshold: float) -> str:
        return stage_fingerprint(
            build._codon_entropy_stage(
                {
                    "analysis": {
                        "codon_entropy": {"sweep_points": 10},
                        "candidate_peak_biology": {"q_value_threshold": threshold},
                    }
                }
            )
        )

    assert provenance(0.05) == provenance(0.05)
    assert provenance(0.05) != provenance(0.01)


_TESTS_DIR = Path(__file__).parent
_FIGURE_TEST_FILES = sorted(
    {*_TESTS_DIR.glob("test_*figure*.py"), _TESTS_DIR / "test_codon_entropy_plot.py"}
)


@pytest.mark.parametrize("path", _FIGURE_TEST_FILES, ids=lambda path: path.name)
def test_figure_tests_import_no_private_renderer_names(path):
    """Figure tests cross the spec seam; only `_charts` and `_quantities` may be reached privately."""
    tree = ast.parse(path.read_text())
    private = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("dubanevics2026topological.figures")
        and node.module.rsplit(".", 1)[-1] not in {"_charts", "_quantities"}
        for alias in node.names
        if alias.name.startswith("_") and alias.name not in {"_charts", "_quantities"}
    ]
    assert private == []


def test_config_default_formats_are_png_only() -> None:
    assert build.load_settings()["figures"]["formats"] == ["png"]


def test_formats_override_renders_only_requested_format(
    variant_build, tmp_path
) -> None:
    png = tmp_path / "figures/peak_uncertainty.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(b"keep")
    outputs = build.build_figures(
        ("peak_uncertainty",), variants=("vanilla",), formats=("svg",)
    )
    assert {path.suffix for path in outputs} == {".svg"}
    assert png.read_bytes() == b"keep"
    assert not (
        tmp_path / "figures/.records/peak_uncertainty.png.json"
    ).exists()


def _justfile_figure_recipes() -> dict[str, tuple[str, list[str]]]:
    """Single-figure recipes of the justfile: name -> (figure id, prerequisites)."""
    root = Path(__file__).resolve().parents[1]
    recipes: dict[str, tuple[str, list[str]]] = {}
    lines = (root / "justfile").read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        header = line.split(":", 1)
        if line.startswith((" ", "#", "\t")) or len(header) != 2 or "=" in header[0]:
            continue
        name = header[0].split()[0]
        if name == "figures":
            continue
        body = []
        for follow in lines[number + 1 :]:
            if not follow.startswith(" "):
                break
            body.append(follow)
        for text in body:
            if "build figures --figure " in text:
                figure_id = text.split("--figure ", 1)[1].split()[0]
                recipes[name] = (figure_id, header[1].split())
    return recipes


def test_a_figure_recipe_depends_on_the_registry_exactly_when_its_spec_reads_it() -> (
    None
):
    specs = {spec.figure_id: spec for spec in registry.FIGURE_SPECS}
    recipes = _justfile_figure_recipes()
    assert "candidate-edge-figure" in recipes and "full-histogram-figure" in recipes
    for name, (figure_id, prerequisites) in recipes.items():
        reads = "manuscript_quantities" in specs[figure_id].inputs
        assert ("manuscript-quantities" in prerequisites) == reads, name


def test_story_views_are_multi_output_families() -> None:
    multi = {spec.figure_id for spec in registry.FIGURE_SPECS if spec.multi}
    assert multi == {
        *STORY_VIEW_IDS,
        "ridge_entropy",
        "graph_codon_entropy",
    }
    for figure_id in STORY_VIEW_IDS:
        spec = registry.get_spec(figure_id)
        assert spec.default is True
        assert spec.inputs == frozenset({"tree_views"})
    assert registry.get_spec("tree_comparison_views").variants == {"annotated"}


def test_graph_codon_entropy_is_an_opt_in_family() -> None:
    spec = registry.get_spec("graph_codon_entropy")
    assert spec.default is False
    assert spec.multi is True
    assert spec.variants == {"vanilla", "annotated"}
    assert spec.inputs == frozenset({"graph_codon_entropy"})
    assert "graph_codon_entropy" in build.NON_CANDIDATE_FIGURE_IDS


# File stems of the multi-output story-view families: the stem before ``__`` names the family.
_FAMILY_OF_STEM = {
    "tree_persistence": "tree_persistence",
    "tree_persistence_combined": "tree_persistence",
    "bootstrap_stops": "bootstrap_band",
    "bootstrap_synonymous": "bootstrap_band",
    "bootstrap_combined": "bootstrap_band",
}


def test_every_manuscript_source_resolves_to_a_default_figure() -> None:
    """Every built manuscript figure is rendered by a plain ``build figures``."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    entries = tomllib.loads((root / "config" / "manuscript.toml").read_text("utf-8"))[
        "figure"
    ]
    unresolved, not_default = [], []
    for entry in entries:
        source = entry.get("source")
        if source is None:
            continue
        figure_id = source if source in FIGURE_IDS else _FAMILY_OF_STEM.get(
            source.split("__")[0]
        )
        if figure_id is None:
            unresolved.append(source)
        elif not registry.get_spec(figure_id).default:
            not_default.append(figure_id)
    assert unresolved == []
    assert not_default == []
