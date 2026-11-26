"""Stop-codon row and synonymous row on the saved-tree views (Group 3b).

The Published graph's ``study_tree_persistence_biology`` draws stop codons in one row
and synonymous leaves in a second, never both in one chart. ``tree_persistence``,
``bootstrap_band`` and the tree comparisons do the same through the shared helpers
of ``figures/_charts.py``; Group 1's single chart stays as the separately named
``combined`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest

from dubanevics2026topological.build.d10_tree_inputs import (
    SavedTreeInputs,
    persistence_table_from_pairs,
)
from dubanevics2026topological.figures import _branch_count_style, _charts, palette
from dubanevics2026topological.figures import bootstrap_band_view as band_view
from dubanevics2026topological.figures import study_tree_persistence_biology as study
from dubanevics2026topological.figures import tree_comparison_view as comparison_view
from dubanevics2026topological.figures import story_views
from dubanevics2026topological.figures import tree_persistence_view as view
from dubanevics2026topological.figures.view_naming import (
    D10_VIEWS,
    comparison_figure_name,
    figure_relative_path,
    single_result_figure_name,
)


def _mark() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [0.0], "y": [0.0]}))
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .properties(width=50, height=50)
    )


# --- shared helpers ---------------------------------------------------------------


def test_leaf_row_label_and_title_name_each_row() -> None:
    assert _charts.leaf_row_label("stops", "3") == "3 stop-codon containing peaks"
    assert _charts.leaf_row_label("synonymous", "12") == "12 synonymous leaves"
    assert _charts.leaf_row_label("stops") == "Stop-codon containing peaks"
    assert _charts.leaf_row_label("synonymous") == "Synonymous leaves"
    plain = _charts.leaf_row_title("synonymous").to_dict()
    assert plain["text"] == "Synonymous leaves" and plain["anchor"] == "start"
    assert "subtitle" not in plain
    titled = _charts.leaf_row_title("stops", "4", ["a", "b"]).to_dict()
    assert titled["text"] == "4 stop-codon containing peaks"
    assert titled["subtitle"] == ["a", "b"]


def test_stack_leaf_rows_stacks_independent_rows_with_the_configured_gap() -> None:
    chart = _charts.stack_leaf_rows([_mark(), _mark()], {"comparison_panel_gap": 9})
    assert len(chart.vconcat) == 2
    spec = chart.to_dict()
    assert spec["spacing"] == 9
    assert spec["resolve"]["scale"] == {
        "x": "independent",
        "y": "independent",
        "color": "independent",
    }


def test_the_biology_figure_and_the_saved_tree_views_share_the_row_composition() -> None:
    for module in (study, view, band_view, comparison_view):
        assert module.stack_leaf_rows is _charts.stack_leaf_rows
    assert study.leaf_row_title is _charts.leaf_row_title
    assert view.leaf_row_title is _charts.leaf_row_title
    assert comparison_view.leaf_row_title is _charts.leaf_row_title
    for module in (view, band_view, comparison_view):
        assert module.synonymous_branch_layer is _charts.synonymous_branch_layer
        assert module.synonymous_leaf_segments is _charts.synonymous_leaf_segments


@dataclass
class _PlotData:
    edge_table: pl.DataFrame
    edge_segment_table: pl.DataFrame
    node_table: pl.DataFrame


def _plot_data() -> _PlotData:
    # P1 and its saddle share the amino acids "K" (synonymous leaf); P2 differs.
    nodes = pl.DataFrame(
        {
            "tree_node_index": [0, 1, 2, 3],
            "dna_sequence": ["ROOT", "SADDLE", "P1", "P2"],
            "aa_sequence": ["M", "K", "K", "R"],
            "display_event": ["minimum", "split_saddle", "maximum", "maximum"],
        }
    )
    edges = pl.DataFrame(
        {
            "source_tree_node_index": [0, 1, 1],
            "target_tree_node_index": [1, 2, 3],
            "source_label": ["ROOT", "SADDLE", "SADDLE"],
            "target_label": ["SADDLE", "P1", "P2"],
            "vertex_count": [2, 3, 1],
        }
    )
    segments = pl.DataFrame(
        {
            "source_label": ["SADDLE", "SADDLE"],
            "target_label": ["P1", "P2"],
            "x_start": [0.0, 0.0],
            "x_end": [1.0, 2.0],
            "y_start": [0.0, 0.0],
            "y_end": [0.5, 0.4],
            "logical_edge_index": [1, 2],
            "vertex_count": [3, 1],
        }
    )
    return _PlotData(edges, segments, nodes)


def test_synonymous_leaf_segments_keep_only_the_synonymous_branches() -> None:
    data = _plot_data()
    found = _charts.synonymous_leaf_segments(data)
    assert found["target_label"].to_list() == ["P1"]
    # The edges can be handed in; the label pair is matched whichever way round.
    flipped = pl.DataFrame({"source_label": ["P2"], "target_label": ["SADDLE"]})
    assert _charts.synonymous_leaf_segments(data, flipped)["target_label"].to_list() == [
        "P2"
    ]
    nothing = _charts.synonymous_leaf_segments(data, data.edge_table.clear())
    assert nothing.is_empty() and nothing.columns == data.edge_segment_table.columns


# --- tree_persistence ---------------------------------------------------------------


def _inputs() -> SavedTreeInputs:
    pairs = pl.DataFrame(
        {
            "birth_label": ["P1", "P2"],
            "death_label": ["ROOT", "SADDLE"],
            "birth_scalar": [0.9, 0.5],
            "death_scalar": [-0.2, 0.1],
            "persistence": [1.1, 0.4],
            "pair_kind": ["ordinary", "ordinary"],
        }
    )
    tree = SimpleNamespace(
        wild_type_marker_table=pl.DataFrame({"x": [0.0], "y": [0.0]}),
        all_stop_marker_table=pl.DataFrame(),
    )
    return SavedTreeInputs(
        tree_label="tree__thr=published_fdr=by_q=0.05_tree=nominal_hash=00000000",
        tree_id="a" * 64,
        layout_id="b" * 64,
        landscape_label="landscape__thr=published_fdr=by_q=0.05_hash=00000000",
        landscape_id="c" * 64,
        tree_analysis=tree,
        persistence_table=persistence_table_from_pairs(pairs),
        scalar_domain=(-0.2, 1.0),
        vertex_count=4,
        fitness_values=pl.Series([-0.1, 0.0, 0.5, 1.0]),
    )


def _patched_view(monkeypatch, style: str, layout: str):
    calls: list[dict] = []
    monkeypatch.setattr(view, "plot_data_with_metrics", lambda tree: _plot_data())
    monkeypatch.setattr(
        view,
        "persistence_with_counts",
        lambda pairs, plot, domain: pairs.with_columns(
            pl.Series("vertex_count", [3, 1])
        ),
    )

    def tree_panel(*args, **kwargs):
        calls.append(kwargs)
        return _mark()

    monkeypatch.setattr(view, "tree_chart", tree_panel)
    monkeypatch.setattr(view, "overlay_count_colorbar", lambda tree, *a: tree)
    monkeypatch.setattr(view, "persistence_chart", lambda *a, **k: _mark())
    monkeypatch.setattr(view, "biology_axis_contract", lambda chart, **kw: chart)
    monkeypatch.setattr(view, "publication_scalar_scales", lambda chart, *a: chart)
    settings = {"figures": {"d10": {"tree_persistence": {"histogram_height": 70}}}}
    chart = view.build_tree_persistence_figure(
        _inputs(), settings, style, title="t", subtitle="s", layout=layout
    )
    return chart, calls


def _diagram_texts(diagram) -> list[str]:
    return [
        layer["mark"]["text"]
        for layer in diagram.to_dict()["layer"]
        if layer["mark"]["type"] == "text" and layer["mark"]["text"] not in {"▲", "★"}
    ]


def _green_points(diagram) -> list[dict]:
    return [
        layer
        for layer in diagram.to_dict()["layer"]
        if layer["mark"]["type"] == "point"
        and layer["mark"].get("color") == palette.SYNONYMOUS_GREEN
    ]


def test_tree_persistence_rows_draw_stops_in_one_row_and_synonymous_in_the_other(
    monkeypatch,
) -> None:
    chart, calls = _patched_view(monkeypatch, "annotated", "rows")
    stops_row, synonymous_row = chart.vconcat
    # Row 1: stop crosses on the tree, no green branch; row 2: green branch, no crosses.
    assert [call["annotated"] for call in calls] == [True, False]
    assert not calls[0]["overlays"]
    (overlay,) = calls[1]["overlays"]
    assert overlay.to_dict()["mark"]["color"] == palette.SYNONYMOUS_GREEN
    assert overlay.data["target_label"].to_list() == ["P1"]
    # Diagrams: no green triangle and a plain peak key in row 1; the green triangle and
    # the synonymous key alone in row 2.
    assert not _green_points(stops_row.hconcat[1].vconcat[1])
    assert _diagram_texts(stops_row.hconcat[1].vconcat[1]) == ["peak"]
    assert len(_green_points(synonymous_row.hconcat[1])) == 1
    assert _diagram_texts(synonymous_row.hconcat[1]) == ["synonymous leaves"]


def test_tree_persistence_rows_have_the_marginal_histogram_above_the_first_row_only(
    monkeypatch,
) -> None:
    chart, _ = _patched_view(monkeypatch, "annotated", "rows")
    stops_row, synonymous_row = chart.vconcat
    assert len(stops_row.hconcat[0].vconcat) == 2  # spacer, tree
    assert len(stops_row.hconcat[1].vconcat) == 2  # histogram, diagram
    assert len(synonymous_row.hconcat) == 2
    assert not hasattr(synonymous_row.hconcat[0], "vconcat")


def test_tree_persistence_rows_are_headed_with_their_counts(monkeypatch) -> None:
    chart, _ = _patched_view(monkeypatch, "annotated", "rows")
    stops_row, synonymous_row = chart.vconcat
    assert stops_row.title.to_dict()["text"] == "0 stop-codon containing peaks"
    assert synonymous_row.title.to_dict()["text"] == "1 synonymous leaves"
    assert chart.title.to_dict()["text"] == "t"


def test_tree_persistence_combined_keeps_one_chart_with_both_leaf_kinds(
    monkeypatch,
) -> None:
    chart, calls = _patched_view(monkeypatch, "annotated", "combined")
    assert len(calls) == 1 and calls[0]["annotated"] is True
    assert len(calls[0]["overlays"]) == 1
    assert len(chart.hconcat) == 2  # one tree beside one diagram, not stacked rows
    diagram = chart.hconcat[1].vconcat[1]
    assert len(_green_points(diagram)) == 1
    assert _diagram_texts(diagram) == ["peak", "synonymous leaves"]


def test_tree_persistence_vanilla_is_one_row_whatever_the_layout(monkeypatch) -> None:
    chart, calls = _patched_view(monkeypatch, "vanilla", "rows")
    assert len(calls) == 1 and calls[0]["annotated"] is False
    assert not calls[0]["overlays"]
    assert len(chart.hconcat) == 2


def test_tree_persistence_refuses_an_unknown_layout() -> None:
    with pytest.raises(ValueError, match="layout must be one of"):
        view.build_tree_persistence_figure(
            _inputs(),
            {"figures": {"d10": {"tree_persistence": {}}}},
            "annotated",
            title="t",
            subtitle="s",
            layout="grid",
        )


# --- bootstrap_band -----------------------------------------------------------------


def _band_stubs(monkeypatch):
    calls: dict = {"panels": [], "diagrams": [], "legends": []}

    def fake_panel(item, *args, keyed=True, leaves="stops", **kwargs):
        calls["panels"].append((keyed, leaves))
        return _mark(), SimpleNamespace()

    def fake_diagram(*args, leaves="stops", **kwargs):
        calls["diagrams"].append(leaves)
        return _mark()

    def fake_legend(*args, **kwargs):
        calls["legends"].append(kwargs)
        return [_mark()]

    monkeypatch.setattr(band_view, "_tree_panel", fake_panel)
    monkeypatch.setattr(band_view, "_diagram", fake_diagram)
    monkeypatch.setattr(band_view, "_leaf_count", lambda *a: 2)
    monkeypatch.setattr(band_view, "persistence_legend_layers", fake_legend)
    monkeypatch.setattr(band_view, "persistence_panel_size", lambda *a: (10, 10))
    monkeypatch.setattr(band_view, "wild_type_identities", lambda analysis: frozenset())
    monkeypatch.setattr(band_view, "peak_nodes", lambda table, names: pl.DataFrame())
    monkeypatch.setattr(
        band_view, "publication_scalar_scales", lambda chart, *a, **k: chart
    )
    return calls


def _band_inputs():
    plot_data = SimpleNamespace(node_table=None, edge_table=SimpleNamespace(height=3))
    item = SimpleNamespace(
        scalar_domain=(-1.0, 1.0),
        tree_analysis=SimpleNamespace(plot_data=plot_data),
        persistence_table=pl.DataFrame({"birth_label": ["a"]}),
    )
    survival = SimpleNamespace(
        nominal_peaks=4, surviving=["a"], absorbed=["b"], two_c=0.1
    )
    settings = {"figures": {"d10": {"bootstrap_band": {}}, "rendering": {}}}
    return item, survival, settings


def test_band_rows_layout_stacks_a_stop_block_over_a_synonymous_block(
    monkeypatch,
) -> None:
    calls = _band_stubs(monkeypatch)
    item, survival, settings = _band_inputs()
    chart = band_view.build_bootstrap_band_figure(
        item,
        item,
        survival,
        settings,
        "annotated",
        title="T",
        subtitle="S",
        layout="rows",
    )
    assert len(chart.vconcat) == 4
    # Colour bar and tree key on the very first row only; leaves follow the block.
    assert calls["panels"] == [
        (True, "stops"),
        (False, "stops"),
        (False, "synonymous"),
        (False, "synonymous"),
    ]
    assert calls["diagrams"] == ["stops", "stops", "synonymous", "synonymous"]
    # A diagram key on the first row of each block; the second names only the green.
    assert len(calls["legends"]) == 2
    assert calls["legends"][0]["stop_codon"] is True
    assert calls["legends"][1] == {
        "peak": False,
        "stop_codon": False,
        "synonymous": True,
    }
    subtitles = [row.title.to_dict()["subtitle"][1] for row in chart.vconcat]
    assert subtitles == [
        "2 stop-codon containing peaks",
        "2 stop-codon containing peaks",
        "2 synonymous leaves",
        "2 synonymous leaves",
    ]


def test_band_stops_and_synonymous_layouts_are_two_rows_each(monkeypatch) -> None:
    calls = _band_stubs(monkeypatch)
    item, survival, settings = _band_inputs()
    stops = band_view.build_bootstrap_band_figure(
        item, item, survival, settings, "annotated",
        title="T", subtitle="S", layout="stops",
    )
    assert len(stops.vconcat) == 2
    assert calls["panels"] == [(True, "stops"), (False, "stops")]
    assert [r.title.to_dict()["subtitle"][1] for r in stops.vconcat] == [
        "2 stop-codon containing peaks"
    ] * 2
    calls["panels"].clear()
    synonymous = band_view.build_bootstrap_band_figure(
        item, item, survival, settings, "annotated",
        title="T", subtitle="S", layout="synonymous",
    )
    assert len(synonymous.vconcat) == 2
    # Each file draws its own colour bar and tree key on its first row.
    assert calls["panels"] == [(True, "synonymous"), (False, "synonymous")]
    assert [r.title.to_dict()["subtitle"][1] for r in synonymous.vconcat] == [
        "2 synonymous leaves"
    ] * 2


def test_band_combined_is_the_two_row_chart_of_group_one(monkeypatch) -> None:
    calls = _band_stubs(monkeypatch)
    item, survival, settings = _band_inputs()
    chart = band_view.build_bootstrap_band_figure(
        item, item, survival, settings, "annotated", title="T", subtitle="S"
    )
    assert len(chart.vconcat) == 2
    assert calls["panels"] == [(True, "stops"), (False, "stops")]
    assert len(calls["legends"]) == 1


def test_band_vanilla_has_no_synonymous_rows(monkeypatch) -> None:
    calls = _band_stubs(monkeypatch)
    item, survival, settings = _band_inputs()
    chart = band_view.build_bootstrap_band_figure(
        item,
        item,
        survival,
        settings,
        "vanilla",
        title=None,
        subtitle=None,
        layout="rows",
    )
    assert len(chart.vconcat) == 2
    assert set(calls["diagrams"]) == {"stops"}


def test_band_synonymous_tree_overlays_the_green_branches_and_no_crosses(
    monkeypatch,
) -> None:
    seen: dict = {}
    data = _plot_data()

    def fake_tree(*args, **kwargs):
        seen.update(kwargs)
        return _mark()

    monkeypatch.setattr(band_view, "plot_data_with_metrics", lambda analysis: data)
    monkeypatch.setattr(band_view, "count_domain", lambda table: (1.0, 3.0))
    monkeypatch.setattr(band_view, "tree_chart", fake_tree)
    monkeypatch.setattr(band_view, "overlay_count_colorbar", lambda tree, *a, **k: tree)
    inputs = SimpleNamespace(tree_analysis=SimpleNamespace())
    band_view._tree_panel(
        inputs, {}, {}, (-1.0, 1.0), True, None, keyed=False, leaves="synonymous"
    )
    assert seen["annotated"] is False
    (overlay,) = seen["overlays"]
    assert overlay.data["target_label"].to_list() == ["P1"]
    band_view._tree_panel(inputs, {}, {}, (-1.0, 1.0), True, None, keyed=False)
    assert seen["annotated"] is True and not seen["overlays"]


def test_band_refuses_an_unknown_layout(monkeypatch) -> None:
    _band_stubs(monkeypatch)
    item, survival, settings = _band_inputs()
    with pytest.raises(ValueError, match="layout must be one of"):
        band_view.build_bootstrap_band_figure(
            item,
            item,
            survival,
            settings,
            "annotated",
            title="T",
            subtitle="S",
            layout="grid",
        )


# --- tree comparisons ---------------------------------------------------------------


def _comparison(monkeypatch, layout: str):
    records: list[dict] = []
    colorbars: list[int] = []

    def fake_tree(*args, **kwargs):
        records.append(kwargs)
        return _mark()

    monkeypatch.setattr(comparison_view, "tree_chart", fake_tree)
    monkeypatch.setattr(
        comparison_view,
        "plot_data_with_metrics",
        lambda analysis: SimpleNamespace(
            edge_table=None, edge_segment_table=pl.DataFrame()
        ),
    )
    monkeypatch.setattr(comparison_view, "count_domain", lambda edges: (1, 3))
    monkeypatch.setattr(
        comparison_view,
        "overlay_count_colorbar",
        lambda tree, settings, rendering, domain, **kw: colorbars.append(1) or tree,
    )
    monkeypatch.setattr(
        comparison_view,
        "synonymous_leaf_edges",
        lambda data: pl.DataFrame({"source_label": ["a"], "target_label": ["b"]}),
    )
    monkeypatch.setattr(
        comparison_view,
        "synonymous_leaf_segments",
        lambda data, edges: pl.DataFrame({"x_start": [0.0]}),
    )
    monkeypatch.setattr(
        comparison_view,
        "synonymous_branch_layer",
        lambda segments, rendering, **kw: alt.Chart(pl.DataFrame({"x": [0]})).mark_rule(),
    )

    def inputs():
        return SimpleNamespace(
            scalar_domain=(0.0, 1.0), threshold=None, tree_analysis=SimpleNamespace()
        )

    panels = [
        comparison_view.ComparisonPanel("a", "", inputs()),
        comparison_view.ComparisonPanel("b", "", inputs()),
    ]
    settings = {"figures": {"rendering": {}, "d10": {"tree_comparison": {}}}}
    chart = comparison_view.build_tree_comparison_figure(
        "tree_comparison_band",
        panels,
        settings,
        title="t",
        subtitle="s",
        layout=layout,
    )
    return chart, records, colorbars


def test_comparison_rows_stack_a_stop_block_over_a_synonymous_block(
    monkeypatch,
) -> None:
    chart, records, colorbars = _comparison(monkeypatch, "rows")
    assert [r["annotated"] for r in records] == [True, True, False, False]
    assert [len(r["overlays"]) for r in records] == [0, 0, 1, 1]
    # The cross key sits on the first panel of the first block only.
    assert [r.get("annotation_key") for r in records] == [True, False, None, None]
    stops_block, synonymous_block = chart.vconcat
    assert stops_block.title.to_dict()["text"] == "Stop-codon containing peaks"
    assert synonymous_block.title.to_dict()["text"] == "Synonymous leaves"
    assert colorbars == [1]  # one colour bar per figure


def test_comparison_combined_draws_both_leaf_kinds_in_one_block(monkeypatch) -> None:
    chart, records, colorbars = _comparison(monkeypatch, "combined")
    assert [r["annotated"] for r in records] == [True, True]
    assert [len(r["overlays"]) for r in records] == [1, 1]
    assert len(chart.vconcat) == 1  # the band view's panels stand in one row
    assert colorbars == [1]


def test_comparison_plain_draws_neither_leaf_kind(monkeypatch) -> None:
    chart, records, _ = _comparison(monkeypatch, "plain")
    assert [r["annotated"] for r in records] == [False, False]
    assert [len(r["overlays"]) for r in records] == [0, 0]
    assert len(chart.vconcat) == 1


def test_comparison_refuses_an_unknown_layout() -> None:
    with pytest.raises(ValueError, match="layout must be one of"):
        comparison_view.build_tree_comparison_figure(
            "tree_comparison_band", [], {}, title="t", subtitle="s", layout="grid"
        )


# --- output names and the build -----------------------------------------------------


def test_layout_outputs_name_the_two_row_default_and_the_combined_extra() -> None:
    assert story_views.layout_outputs("tree_persistence", "annotated") == [
        ("tree_persistence", "rows"),
        ("tree_persistence_combined", "encoded"),
    ]
    # The band view is two files of two rows each, one per leaf kind.
    assert story_views.layout_outputs("bootstrap_band", "annotated")[:2] == [
        ("bootstrap_band_stops", "stops"),
        ("bootstrap_band_synonymous", "synonymous"),
    ]
    assert story_views.layout_outputs("bootstrap_band", "vanilla") == [
        ("bootstrap_band", "rows")
    ]
    # Vanilla marks neither leaf kind: one figure, under the plain view name.
    assert story_views.layout_outputs("tree_persistence", "vanilla") == [
        ("tree_persistence", "rows")
    ]
    assert story_views.comparison_layouts("main") == [
        ("rows", "main"),
        ("combined", "main_combined"),
    ]


def test_the_combined_extras_have_their_own_valid_file_names() -> None:
    params = {"graph": "full", "tree": "nominal", "hash": "a1b2c3d4"}
    assert single_result_figure_name(
        "tree_persistence_combined", params, "annotated", "png"
    ) == (
        "tree_persistence_combined__graph=full_tree=nominal_hash=a1b2c3d4"
        "_style=annotated.png"
    )
    for view_name in ("tree_persistence_combined", "bootstrap_band_combined"):
        assert D10_VIEWS[view_name].kind == "single"
    name = comparison_figure_name("tree_comparison_fdr", "main_combined", "12345678", "png")
    assert figure_relative_path("tree_comparison_fdr", name)


def test_tree_persistence_outputs_are_two_rows_then_the_combined_extra(
    monkeypatch,
) -> None:
    layouts: list[tuple[str, str]] = []

    def fake_figure(inputs, settings, style, *, layout, **kwargs):
        layouts.append((style, layout))
        return _mark()

    monkeypatch.setattr(story_views, "build_tree_persistence_figure", fake_figure)
    inputs = SimpleNamespace(
        vertex_count=3, persistence_table=pl.DataFrame({"a": [1, 2]})
    )
    label = "tree__graph=full_tree=nominal_hash=e6b7c141"
    payload = {"nominal": {label: inputs}}
    stems = []
    for style in ("annotated", "vanilla"):
        for output in story_views.tree_persistence_outputs(payload, {}, style):
            stems.append(output.stem)
            output.build()
    assert layouts == [
        ("annotated", "rows"),
        ("annotated", "encoded"),
        ("vanilla", "rows"),
    ]
    assert stems == [
        "tree_persistence__graph=full_tree=nominal_style=annotated",
        "tree_persistence_combined__graph=full_tree=nominal_style=annotated",
        "tree_persistence__graph=full_tree=nominal_style=vanilla",
    ]


def test_bootstrap_band_outputs_are_a_stops_and_a_synonymous_file(
    monkeypatch,
) -> None:
    layouts: list[tuple[str, str]] = []

    def fake_figure(nominal, band, survival, settings, style, *, layout, **kwargs):
        layouts.append((style, layout))
        return _mark()

    monkeypatch.setattr(story_views, "build_bootstrap_band_figure", fake_figure)
    label = (
        "tree__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019"
        "_tree=band_bandq=0.95_hash=1c35ccfe"
    )
    item = SimpleNamespace(
        nominal=SimpleNamespace(vertex_count=3), band=object(), survival=object()
    )
    stems = []
    for style in ("annotated", "vanilla"):
        for output in story_views.bootstrap_band_outputs(
            {"band": {label: item}}, {}, style
        ):
            stems.append(output.stem)
            output.build()
    assert layouts == [
        ("annotated", "stops"),
        ("annotated", "synonymous"),
        ("annotated", "encoded"),
        ("vanilla", "rows"),
    ]
    tail = (
        "thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95"
    )
    assert stems == [
        f"bootstrap_stops__{tail}_style=annotated",
        f"bootstrap_synonymous__{tail}_style=annotated",
        f"bootstrap_combined__{tail}_style=annotated",
        f"bootstrap__{tail}_style=vanilla",
    ]


def test_synonymous_branch_layer_can_leave_the_trees_fitness_axis_alone() -> None:
    segments = _plot_data().edge_segment_table
    plain = _charts.synonymous_branch_layer(segments, {}).to_dict()
    kept = _charts.synonymous_branch_layer(segments, {}, keep_fitness_axis=True).to_dict()
    # An explicit ``axis=None`` would merge with the tree's axis into no axis at all.
    assert plain["encoding"]["y"]["axis"] is None
    assert "axis" not in kept["encoding"]["y"]
    assert kept["encoding"]["x"]["axis"] is None


def test_tree_persistence_encoded_draws_teal_leaves_and_a_teal_diagram_marker(
    monkeypatch,
) -> None:
    chart, calls = _patched_view(monkeypatch, "annotated", "encoded")
    assert len(calls) == 1 and calls[0]["annotated"] is True
    (overlay,) = calls[0]["overlays"]
    spec = overlay.to_dict()
    # Teal by vertex count (a per-branch colour field), never the single green.
    assert spec["encoding"]["color"]["field"] == "teal_color"
    assert overlay.data["target_label"].to_list() == ["P1"]
    assert overlay.data["teal_color"].to_list() == [
        _branch_count_style.branch_count_teal(3, {}, (1.0, 3.0))
    ]
    # The heading is the row's own title (a panel title, so the untitled canonical file
    # keeps it); the histogram's axis reads "Count", not "Sequences".
    assert chart.title is alt.Undefined
    (row,) = chart.vconcat
    heading = row.title.to_dict()
    assert heading["text"] == "t" and heading["subtitle"] == ["s"]
    histogram = row.hconcat[1].vconcat[0].to_dict()
    assert histogram["encoding"]["y"]["axis"]["title"] == "Count"
    diagram = row.hconcat[1].vconcat[1]
    teal = [
        layer
        for layer in diagram.to_dict()["layer"]
        if layer["mark"]["type"] == "point"
        and layer["mark"].get("color") == palette.SYNONYMOUS_TEAL
    ]
    assert len(teal) == 1 and not _green_points(diagram)
    # The key: grey triangle peak, then a teal filled circle for synonymous.
    glyphs = {
        layer["mark"]["text"]: layer["mark"]["color"]
        for layer in diagram.to_dict()["layer"]
        if layer["mark"]["type"] == "text" and layer["mark"]["text"] in {"▲", "●", "×"}
    }
    assert glyphs == {"▲": palette.PEAK_GREY, "●": palette.SYNONYMOUS_TEAL}
    assert _diagram_texts(diagram) == ["peak", "●", "synonymous"]


def test_encoded_diagram_triangles_take_the_trees_cross_size(monkeypatch) -> None:
    seen: list[int] = []
    _patched_view(monkeypatch, "annotated", "encoded")  # installs the stubs
    monkeypatch.setattr(
        view,
        "persistence_chart",
        lambda table, settings, *a, **k: seen.append(settings["persistence"]["point_size"])
        or _mark(),
    )
    settings = {
        "figures": {
            "rendering": {"stop_codon_tree_size": 91},
            "d10": {"tree_persistence": {"histogram_height": 70, "persistence": {"point_size": 5}}},
        }
    }
    view.build_tree_persistence_figure(
        _inputs(), settings, "annotated", title="t", subtitle="s", layout="encoded"
    )
    assert seen and set(seen) == {91}  # one size setting for the cross and the triangles


def test_a_deeper_branch_is_a_darker_teal_on_the_grey_ramps_log_scale() -> None:
    domain = (1.0, 10000.0)
    light = _branch_count_style.branch_count_teal(1, {}, domain)
    middle = _branch_count_style.branch_count_teal(100, {}, domain)
    dark = _branch_count_style.branch_count_teal(10000, {}, domain)
    assert light == palette.SYNONYMOUS_TEAL_LIGHT and dark == palette.SYNONYMOUS_TEAL_DARK

    def total(hex_color: str) -> int:
        return sum(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))

    assert total(light) > total(middle) > total(dark)
    # Half-way on the log scale (count 100 of 1..10000) is half-way between the ends.
    for i in (1, 3, 5):
        ends = (int(light[i : i + 2], 16) + int(dark[i : i + 2], 16)) / 2
        assert abs(int(middle[i : i + 2], 16) - ends) <= 1


def test_a_valley_is_synonymous_when_it_has_the_amino_acids_of_its_saddle() -> None:
    valleys = pl.DataFrame(
        {
            "birth_sequence": ["AAAGGG", "AAAGGG", "AAAGGG", "AAAGG*"],
            "death_sequence": ["AAGGGA", "AAGGGG", "AACGGG", "AAGGGG"],
        }
    )
    # AAA=K, AAG=K, GGG=G; AAC=N. Row 1 (GGA = G, AAG = K) differs in the second codon
    # only through frame shift, so it is not synonymous; row 2 is; rows 3 and 4 are not
    # (a different amino acid; a sequence that does not translate is never synonymous).
    found = _charts.synonymous_valley_sequences(valleys)
    assert found == frozenset({"AAAGGG"})
    assert _charts.synonymous_valley_sequences(valleys.clear()) == frozenset()


def test_band_encoded_is_one_nominal_row_and_one_band_row_with_both_encodings(
    monkeypatch,
) -> None:
    calls = _band_stubs(monkeypatch)
    monkeypatch.setattr(band_view, "_encoded_flags", lambda *a: (True, True))
    item, survival, settings = _band_inputs()
    chart = band_view.build_bootstrap_band_figure(
        item, item, survival, settings, "annotated", title="T", subtitle="S", layout="encoded"
    )
    assert [kwargs.get("encoded") for kwargs in calls["legends"]] == [True]
    assert calls["legends"][0]["synonymous"] and calls["legends"][0]["stop_codon"]
    assert len(chart.vconcat) == 2
    assert calls["panels"] == [(True, "encoded"), (False, "encoded")]
