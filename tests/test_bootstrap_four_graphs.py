"""The four-graph composite keeps each verified payload in its own panel."""

import altair as alt
import pytest

from dubanevics2026topological.figures import bootstrap_four_graphs as composite


def test_layout_and_variant_payloads(monkeypatch):
    seen = []

    def panel(payload, settings):
        seen.append(payload["variant"])
        return alt.Chart().mark_point().encode(x=alt.datum(1))

    monkeypatch.setattr(composite, "build_bootstrap_tree_histogram_figure", panel)
    settings = {"analysis": {"clamped": {"candidate_functional_threshold": {"value": -0.25}}}}
    panels = {key: ({"variant": key}, settings) for key, _ in composite.PANEL_ORDER}
    result = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    assert seen == [key for key, _ in composite.PANEL_ORDER]
    assert [panel["title"]["text"] for panel in result["concat"]] == [
        composite.panel_title(key, -0.25) for key, _ in composite.PANEL_ORDER
    ]


def test_composite_rejects_missing_graphs():
    with pytest.raises(ValueError, match="exactly the four"):
        composite.build_bootstrap_four_graphs_figure({})


def test_composite_accepts_saved_panel_specs():
    spec = alt.Chart(alt.Data(values=[{"x": 1}])).mark_point().encode(x="x:Q").to_dict()
    thresholds = [-0.5, -0.25, -0.6, -0.3]
    panels = {
        key: {**spec, "_composite_threshold": threshold}
        for (key, _), threshold in zip(composite.PANEL_ORDER, thresholds, strict=True)
    }
    result = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    assert result["columns"] == 2 and len(result["concat"]) == 4
    assert [panel["title"]["text"] for panel in result["concat"]] == [
        composite.panel_title(key, threshold)
        for (key, _), threshold in zip(composite.PANEL_ORDER, thresholds, strict=True)
    ]


def test_nested_panel_preserves_named_dataset():
    saved = {"$schema": "schema", "datasets": {"sample": [{"x": 7}]}, "data": {"name": "sample"}, "mark": "point"}
    nested = composite._nested_panel_spec(saved)
    assert nested["data"] == {"values": [{"x": 7}]}
    assert "$schema" not in nested and "datasets" not in nested


def _saved_panel():
    """A saved panel: the tree (count labels, leaders), the colour bar, the tree key; a histogram."""
    return {
        "hconcat": [
            {"layer": [
                {"layer": [
                    {"mark": {"type": "text"}, "encoding": {"text": {"field": "label"}}, "data": {"values": [{"vertex_count": 1}, {"vertex_count": 5}]}},
                    {"mark": {"type": "rule"}, "encoding": {"x": {"field": "leader_x"}}, "data": {"values": [{"vertex_count": 1}, {"vertex_count": 5}]}},
                ]},
                {"layer": [{"mark": {"type": "text", "text": "Branch vertex count"}}]},
                {"layer": [
                    {"mark": {"type": "text", "text": "★"}},
                    {"mark": {"type": "text", "text": "Wild type"}},
                ]},
            ]},
            {"width": 150, "encoding": {"x": {"axis": {"title": "Vertex count"}}}},
        ],
    }


def test_composite_presentation_reduces_repetition_and_label_crowding():
    saved = _saved_panel()
    compact = composite.compact_panel_spec(saved, show_legend=False, show_histogram_title=False)
    tree, histogram = compact["hconcat"]
    # Only the tree itself remains: the colour bar and the tree key are dropped.
    assert len(tree["layer"]) == 1
    assert tree["layer"][0]["layer"][0]["data"]["values"] == [{"vertex_count": 5}]
    assert tree["layer"][0]["layer"][1]["data"]["values"] == [{"vertex_count": 5}]
    assert histogram["width"] < 150
    assert histogram["encoding"]["x"]["axis"]["title"] is None
    assert saved["hconcat"][1]["width"] == 150
    assert len(saved["hconcat"][0]["layer"]) == 3


def test_the_first_panel_keeps_the_colour_bar_and_the_tree_key():
    compact = composite.compact_panel_spec(
        _saved_panel(), show_legend=True, show_histogram_title=True
    )
    tree, _ = compact["hconcat"]
    texts = [
        leaf["mark"].get("text")
        for item in tree["layer"][1:]
        for leaf in (item["layer"])
    ]
    assert texts == ["Branch vertex count", "★", "Wild type"]


def _text_marks(node):
    found = []
    if isinstance(node, dict):
        mark = node.get("mark")
        if isinstance(mark, dict) and mark.get("type") == "text":
            found.append(mark.get("text"))
        for value in node.values():
            found.extend(_text_marks(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_text_marks(value))
    return found


def _real_saved_panel():
    """A valid saved panel built with Altair: tree, colour bar, tree key, histogram."""

    def text(label):
        return alt.Chart(alt.Data(values=[{"v": 1}])).mark_text(x=5, y=5, text=label)

    base = alt.Chart(alt.Data(values=[{"x": 1, "y": 1}])).mark_point().encode(x="x:Q", y="y:Q")
    tree = alt.layer(
        base,
        alt.layer(text("Branch vertex count")),
        alt.layer(text("\u2605"), text("Wild type")),
    )
    return alt.hconcat(tree, base).to_dict()


def test_one_colour_bar_and_one_key_for_the_whole_figure():
    panels = {
        key: {**_real_saved_panel(), "_composite_threshold": -0.25}
        for key, _ in composite.PANEL_ORDER
    }
    result = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    cells = result["concat"]
    counts = [
        (_text_marks(cell).count("Branch vertex count"), _text_marks(cell).count("Wild type"))
        for cell in cells
    ]
    # The top-left panel alone carries the colour bar and the key.
    assert counts == [(1, 1), (0, 0), (0, 0), (0, 0)]


def test_the_chart_configuration_is_applied_once_at_the_top_level():
    spec = alt.Chart(alt.Data(values=[{"x": 1}])).mark_point().encode(x="x:Q").to_dict()
    spec["config"] = {"font": "Nested"}
    panels = {
        key: {**spec, "_composite_threshold": -0.25} for key, _ in composite.PANEL_ORDER
    }
    settings = {
        "figures": {
            "rendering": {"font_family": "Test Sans", "axis_label_font_size": 17},
            "figures": {"bootstrap_four_graphs": {"legend_font_size": 19}},
        }
    }
    result = composite.build_bootstrap_four_graphs_figure(panels, settings).to_dict()
    config = result["config"]
    # No frame around any view, and the fonts of the single-figure tree histograms.
    assert config["view"]["strokeWidth"] == 0
    assert config["font"] == "Test Sans"
    assert config["axis"]["labelFontSize"] == 17
    assert config["legend"]["labelFontSize"] == 19
    # No panel carries a configuration of its own.
    cells = result["concat"]
    assert all("config" not in cell for cell in cells)
    # The default configuration still applies when no settings are given.
    default = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    assert default["config"]["view"]["strokeWidth"] == 0


def test_every_panel_keeps_its_own_scales_and_axes():
    spec = alt.Chart(alt.Data(values=[{"x": 1}])).mark_point().encode(x="x:Q").to_dict()
    panels = {
        key: {**spec, "_composite_threshold": -0.25} for key, _ in composite.PANEL_ORDER
    }
    result = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    assert result["resolve"]["scale"]["y"] == "independent"
    assert result["resolve"]["axis"]["y"] == "independent"


def _panel_with_y_axis(width, tick_format):
    tree = (
        alt.Chart(alt.Data(values=[{"x": 1, "y": 1}]))
        .mark_point()
        .encode(x="x:Q", y=alt.Y("y:Q", axis=alt.Axis(format=tick_format)))
        .properties(width=width, height=300)
    )
    return alt.hconcat(tree, tree).to_dict()


def test_the_four_trees_share_plot_size_and_y_axis_extent():
    widths = [300, 300, 300, 300]
    formats = [".1f", ".3f", ".1f", ".3f"]
    panels = {
        key: {**_panel_with_y_axis(width, fmt), "_composite_threshold": -0.25}
        for (key, _), width, fmt in zip(composite.PANEL_ORDER, widths, formats, strict=True)
    }
    settings = {"figures": {"figures": {composite.FIGURE_ID: {"tree_y_axis_extent": 61}}}}
    result = composite.build_bootstrap_four_graphs_figure(panels, settings).to_dict()
    trees = [cell["hconcat"][0] for cell in result["concat"]]
    assert len({(tree["width"], tree["height"]) for tree in trees}) == 1
    axes = [tree["encoding"]["y"]["axis"] for tree in trees]
    assert all(axis["minExtent"] == axis["maxExtent"] == 61 for axis in axes)


def _tree_rows(n_peaks):
    """Branch rows of a comb tree: a root branch, then ``n_peaks`` tips off node 1."""
    rows = [{"source_tree_node_index": 0, "target_tree_node_index": 1}]
    rows += [
        {"source_tree_node_index": 1, "target_tree_node_index": 2 + i} for i in range(n_peaks)
    ]
    return rows


def test_each_title_carries_the_peak_count_of_its_own_panel_data():
    fixture = [1200, 3, 7, 4]
    panels = {}
    for (key, _), n in zip(composite.PANEL_ORDER, fixture, strict=True):
        spec = (
            alt.Chart(alt.Data(values=_tree_rows(n)))
            .mark_point()
            .encode(x="target_tree_node_index:Q")
            .to_dict()
        )
        panels[key] = {**spec, "_composite_threshold": -0.25}
    result = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    titles = [cell["title"] for cell in result["concat"]]
    assert [t["subtitle"] for t in titles] == [
        f"{kept} peaks after the bootstrap band" for kept in ("1,200", "3", "7", "4")
    ]


def test_each_title_states_the_kept_and_the_nominal_peak_counts():
    kept, nominal = [3, 4, 5, 6], [514, 158, 242, 1350]
    panels = {}
    for (key, _), n, total in zip(composite.PANEL_ORDER, kept, nominal, strict=True):
        spec = (
            alt.Chart(alt.Data(values=_tree_rows(n)))
            .mark_point()
            .encode(x="target_tree_node_index:Q")
            .to_dict()
        )
        panels[key] = {**spec, "_composite_threshold": -0.25, "_composite_nominal_peaks": total}
    result = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    assert [cell["title"]["subtitle"] for cell in result["concat"]] == [
        "3 of 514 peaks kept by the bootstrap band",
        "4 of 158 peaks kept by the bootstrap band",
        "5 of 242 peaks kept by the bootstrap band",
        "6 of 1,350 peaks kept by the bootstrap band",
    ]


def test_the_figure_is_one_two_column_grid_that_aligns_its_columns():
    spec = alt.Chart(alt.Data(values=[{"x": 1}])).mark_point().encode(x="x:Q").to_dict()
    panels = {key: {**spec, "_composite_threshold": -0.25} for key, _ in composite.PANEL_ORDER}
    result = composite.build_bootstrap_four_graphs_figure(panels).to_dict()
    assert "vconcat" not in result and "hconcat" not in result
    assert result["columns"] == 2 and len(result["concat"]) == 4
    assert result["align"] == "all" and result["bounds"] == "full"


def test_a_panel_without_branch_rows_keeps_a_blank_subtitle():
    assert composite.peak_count_from_spec({"data": {"values": [{"x": 1}]}}) is None
    assert composite.panel_peaks_subtitle(None).strip() == ""
