"""Vector rendering contract for the conceptual construction panels."""

from __future__ import annotations

import colorsys
import copy
import math

import polars as pl
import pytest

from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.analysis.manifold_graph_tree import (
    build_manifold_graph_tree,
)
from dubanevics2026topological.figures import graph_tree_persistence as figure_module
from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures.graph_tree_persistence import (
    build_graph_tree_persistence_panel_data,
    build_graph_tree_persistence_panels,
    build_graph_tree_persistence_preview,
)
from dubanevics2026topological.figures.registry import get_spec, render_payload

SETTINGS = load_settings()


@pytest.fixture(scope="module")
def construction():
    return build_manifold_graph_tree()


@pytest.fixture(scope="module")
def composition_settings():
    settings = load_settings()
    settings["figures"]["figures"]["graph_tree_persistence"][
        "presentation_surface_resolution"
    ] = 8
    return settings


@pytest.fixture(scope="module")
def composition_spec(construction, composition_settings):
    return render_payload(
        get_spec("graph_tree_persistence"),
        construction,
        composition_settings,
        "vanilla",
    ).to_dict(validate=True)


@pytest.fixture(scope="module")
def opening_data(construction):
    return build_graph_tree_persistence_panel_data(construction, 1.0, SETTINGS)


@pytest.fixture(scope="module")
def saddle_data(construction):
    return build_graph_tree_persistence_panel_data(
        construction, construction.thresholds[3], SETTINGS
    )


def _sample(data, sample_id):
    return next(row for row in data.samples if row["sample_id"] == sample_id)


def _tree_node(data, sample_id):
    return next(row for row in data.tree_nodes if row["sample_id"] == sample_id)


def test_panels_are_native_altair_and_share_fixed_projected_sample_geometry(
    construction, saddle_data
):
    panels = build_graph_tree_persistence_panels(
        construction, construction.thresholds[3], SETTINGS
    )
    assert set(panels) == {"graph", "tree", "persistence"}
    graph_spec = panels["graph"].to_dict(validate=True)
    assert graph_spec["projection"]["type"] == "identity"
    assert not _contains_mark(graph_spec, "image")
    assert _contains_mark(panels["tree"].to_dict(validate=True), "rule")
    assert len(saddle_data.samples) == 36
    assert len(saddle_data.graph_surface) > 0
    assert saddle_data.projection["scale"] > 0


def test_samples_start_as_circles_and_critical_shapes_activate_at_events(construction):
    ids = construction.critical_ids
    opening = build_graph_tree_persistence_panel_data(construction, 1.0, SETTINGS)
    assert all(sample["shape"] == "circle" for sample in opening.samples)
    assert {sample["size"] for sample in opening.samples} == {32}
    assert {sample["graph_size"] for sample in opening.samples} == {46}
    assert opening.tree_nodes == ()

    for event_name, shape in (
        ("left_peak", "triangle-up"),
        ("saddle", "diamond"),
        # No valley markers: the global minimum keeps the plain circle.
        ("global_minimum", "circle"),
    ):
        sample_id = ids[event_name]
        scalar = construction.sample_table.filter(
            construction.sample_table["sample_id"] == sample_id
        )["scalar"][0]
        before = build_graph_tree_persistence_panel_data(
            construction, scalar + 1e-8, SETTINGS
        )
        at = build_graph_tree_persistence_panel_data(construction, scalar, SETTINGS)
        after = build_graph_tree_persistence_panel_data(
            construction, scalar - 1e-8, SETTINGS
        )
        assert _sample(before, sample_id)["shape"] == "circle"
        assert _sample(before, sample_id)["display_event"] == "regular"
        assert _sample(at, sample_id)["shape"] == shape
        assert (
            _sample(at, sample_id)["display_event"] == _sample(at, sample_id)["event"]
        )
        assert _sample(after, sample_id)["shape"] == shape
        assert "triangle-down" not in {sample["shape"] for sample in after.samples}
        assert "triangle-down" not in {node["shape"] for node in after.tree_nodes}

    # The maximum has scalar one, which is reserved as the unactivated sentinel.
    right_peak = ids["right_peak"]
    assert _sample(opening, right_peak)["shape"] == "circle"
    first_peak_row = build_graph_tree_persistence_panel_data(
        construction, 0.99, SETTINGS
    )
    assert _sample(first_peak_row, right_peak)["shape"] == "triangle-up"
    assert _sample(first_peak_row, right_peak)["size"] == 116
    assert _sample(first_peak_row, right_peak)["graph_size"] == 116


def test_graph_panel_shows_every_sample_and_axes_carry_a_title_only(
    construction, composition_settings
):
    graph_spec = build_graph_tree_persistence_panels(
        construction, construction.opening_threshold, composition_settings
    )["graph"].to_dict(validate=True)
    regular_layer, critical_layer = graph_spec["layer"][4:6]
    regular = _dataset_values(graph_spec, regular_layer["data"])
    critical = _dataset_values(graph_spec, critical_layer["data"])
    assert len(regular) + len(critical) == construction.sample_table.height
    assert len(critical) == len(construction.critical_ids)
    assert not any(row["is_critical"] for row in regular)
    assert all(row["is_critical"] for row in critical)
    # Critical samples are drawn after regular ones; the height axis is its title
    # alone: no tick marks and no tick numbers.
    assert [layer["mark"]["type"] for layer in graph_spec["layer"][6:]] == ["text"]
    first = build_graph_tree_persistence_panels(
        construction, construction.thresholds[1], composition_settings
    )
    later = build_graph_tree_persistence_panels(
        construction,
        construction.thresholds[2],
        composition_settings,
        show_fitness_axis=False,
        show_death_axis=False,
    )

    def axes(spec):
        encoding = next(
            layer["encoding"]
            for layer in spec["layer"]
            if layer["encoding"].get("x", {}).get("field") == "x"
            and layer["encoding"].get("color")
        )
        return encoding["x"]["axis"], encoding["y"]["axis"]

    persistence = first["persistence"].to_dict(validate=True)
    x_axis, y_axis = axes(persistence)
    assert x_axis["title"] == "Fitness at death"
    assert y_axis["title"] == "Fitness at birth"
    for axis in (x_axis, y_axis):
        assert axis["ticks"] is False and axis["labels"] is False
        assert axis["domain"] is False and axis["grid"] is False
        assert "values" not in axis
    later_persistence = later["persistence"].to_dict(validate=True)
    # The composed figure titles the death axis once, so a panel built without it
    # carries no x title.
    assert axes(later_persistence)[0]["title"] is None
    assert axes(later_persistence)[1]["title"] is None
    # The Tree row titles its fitness axis "Fitness" in the first column only.
    tree_axes = [
        layer["encoding"]["y"]["axis"]
        for layer in first["tree"].to_dict(validate=True)["layer"]
        if isinstance(layer["encoding"].get("y", {}).get("axis"), dict)
    ]
    assert tree_axes
    for axis in tree_axes:
        assert axis["title"] == "Fitness"
        assert axis["ticks"] is False and axis["labels"] is False
    assert all(
        layer["encoding"].get("y", {}).get("axis") is None
        for layer in later["tree"].to_dict(validate=True)["layer"]
        if "y" in layer["encoding"]
    )


def test_graph_height_axis_is_a_title_centred_on_the_projected_heights(
    construction, composition_settings
):
    data = build_graph_tree_persistence_panel_data(
        construction, construction.thresholds[1], composition_settings
    )
    axis = data.fitness_axis
    assert set(axis) == {"title"}
    assert [row["label"] for row in axis["title"]] == ["Fitness"]
    projector = figure_module._projector(figure_module._appearance(SETTINGS))
    corner_heights = [
        float(projector(x, y, value)[1])
        for x in (0.0, 1.0)
        for y in (0.0, 1.0)
        for value in (0.0, 1.0)
    ]
    assert min(corner_heights) < axis["title"][0]["y"] < max(corner_heights)


def test_height_ticks_use_data_extent_when_not_normalised(construction):
    assert figure_module._height_ticks(construction) == [0.0, 1.0]
    shifted = copy.copy(construction)
    object.__setattr__(
        shifted,
        "sample_table",
        construction.sample_table.with_columns(
            (pl.col("scalar") * 2 - 0.5).alias("scalar")
        ),
    )
    ticks = figure_module._height_ticks(shifted)
    assert ticks == pytest.approx([-0.5, 1.5])
    axis = figure_module._fitness_axis("Height")
    assert axis["title"] == "Height"
    assert axis["ticks"] is False and axis["labels"] is False
    # The scale domain follows the ticks with the same small padding (-0.03 and
    # +0.05 of the span for [0, 1] data), so both ticks stay on the scale.
    assert figure_module._height_domain([0.0, 1.0]) == [-0.03, 1.05]
    low, high = figure_module._height_domain(ticks)
    assert [low, high] == pytest.approx([-0.5 - 0.03 * 2, 1.5 + 0.05 * 2])
    assert low < ticks[0] and ticks[-1] < high
    # The persistence panel's own scales use that domain, so its ticks are on-scale.
    persistence = figure_module._persistence_chart(
        shifted, 1.0, figure_module._appearance(SETTINGS)
    ).to_dict()
    for layer in persistence["layer"]:
        for channel in ("x", "y"):
            scale = layer["encoding"].get(channel, {}).get("scale")
            if scale:
                assert scale["domain"] == pytest.approx([low, high])


def test_persistence_markers_use_tree_shapes(composition_spec):
    _, rows = _grid_rows(composition_spec)
    for cell in rows[2][1:]:
        _, _, filled = _persistence_layers(cell)
        assert filled["mark"]["shape"] == "triangle-up"


def _dataset_values(spec, data):
    if "values" in data:
        return data["values"]
    return spec["datasets"][data["name"]]


def test_peak_lineage_colors_persist_and_shared_region_enters_below_saddle(
    construction,
):
    lineage = {"#6F7F8F", "#8B6F8E"}
    shared = "#AA9470"
    for threshold in (construction.thresholds[3], 0.0):
        data = build_graph_tree_persistence_panel_data(
            construction, threshold, SETTINGS
        )
        # Colours follow tree layout order (may swap geometric left/right).
        assert (
            _sample(data, construction.critical_ids["right_peak"])["color"] in lineage
        )
        assert (
            _sample(data, construction.critical_ids["left_peak"])["color"] in lineage
        )
        assert {
            _sample(data, construction.critical_ids["right_peak"])["color"],
            _sample(data, construction.critical_ids["left_peak"])["color"],
        } == lineage
    below_saddle = build_graph_tree_persistence_panel_data(
        construction, construction.thresholds[3] - 1e-5, SETTINGS
    )
    assert (
        _sample(below_saddle, construction.critical_ids["saddle"])["color"] == shared
    )


def test_partial_tree_branches_end_at_threshold_and_trunk_stays_vertical(construction):
    data = build_graph_tree_persistence_panel_data(construction, 0.99, SETTINGS)
    partial_edges = [edge for edge in data.tree_edges if edge["partial"]]
    assert partial_edges
    threshold_endpoints = [
        edge
        for edge in partial_edges
        if edge["point_order"] == 1 and edge["y"] == pytest.approx(0.99)
    ]
    assert threshold_endpoints
    final_data = build_graph_tree_persistence_panel_data(construction, 0.0, SETTINGS)
    trunk_nodes = [
        _tree_node(final_data, sample_id) for sample_id in data.trunk_sample_ids
    ]
    assert {node["x"] for node in trunk_nodes} == {0.5}


def test_tree_regular_and_critical_node_sizes_are_independent(construction):
    data = build_graph_tree_persistence_panel_data(construction, 0.70, SETTINGS)
    right_id = construction.critical_ids["right_peak"]
    regular_id = "sample-r5-c6"
    assert _tree_node(data, regular_id)["size"] == 70
    assert _tree_node(data, regular_id)["shape"] == "circle"
    assert _tree_node(data, right_id)["size"] == 130
    assert _tree_node(data, right_id)["shape"] == "triangle-up"

    final = build_graph_tree_persistence_panel_data(construction, 0.0, SETTINGS)
    assert len(final.tree_nodes) == 36


def _persistence_layers(spec):
    """Return (diagonal, threshold rule, filled markers); no hollow layer exists."""
    diagonal, threshold, filled = spec["layer"]
    return diagonal, threshold, filled


def _lightness(color):
    red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(red, green, blue)[1]


def _grid_rows(composition_spec):
    heading, *rows, _death_title = composition_spec["vconcat"]
    return heading["hconcat"], [row["hconcat"] for row in rows]


def _death_title_row(composition_spec):
    return composition_spec["vconcat"][-1]["hconcat"]


def _text_marks(spec):
    return [
        layer["mark"]
        for layer in spec["layer"]
        if isinstance(layer.get("mark"), dict) and layer["mark"].get("type") == "text"
    ]


def test_height_at_death_is_titled_once_under_the_persistence_row(
    composition_spec,
):
    _, rows = _grid_rows(composition_spec)
    for cell in rows[2][1:]:
        for layer in cell["layer"]:
            x = layer["encoding"].get("x", {})
            if isinstance(x.get("axis"), dict):
                assert x["axis"]["title"] is None
    title_row = _death_title_row(composition_spec)
    labels = [cell["data"]["values"][0]["label"] for cell in title_row]
    assert labels == ["", "Fitness at death"]
    assert str(composition_spec).count("Fitness at death") == 1
    # "Fitness at birth" stays on the first persistence panel only.
    first_y = [
        layer["encoding"]["y"]["axis"]["title"]
        for layer in rows[2][1]["layer"]
        if isinstance(layer["encoding"].get("y", {}).get("axis"), dict)
    ]
    assert first_y and set(first_y) == {"Fitness at birth"}
    assert all(
        layer["encoding"].get("y", {}).get("axis", {}).get("title") is None
        for cell in rows[2][2:]
        for layer in cell["layer"]
        if isinstance(layer["encoding"].get("y", {}).get("axis"), dict)
    )


def test_peak_saddle_legend_sits_in_the_scan_complete_graph_panel(
    composition_spec,
):
    _, rows = _grid_rows(composition_spec)
    graph_cells = rows[0][1:]
    keys = ("peak", "valley", "saddle")  # "valley" must never be drawn
    legend_texts = [
        [m["text"] for m in _text_marks(cell) if m.get("text") in keys]
        for cell in graph_cells
    ]
    assert legend_texts[:3] == [[], [], []]
    assert legend_texts[3] == ["peak", "saddle"]
    rightmost_x = max(
        m["x"] for m in _text_marks(graph_cells[3]) if "x" in m and "text" in m
    )
    marks = [m for m in _text_marks(graph_cells[3]) if m.get("text") == "peak"]
    # Top right: right half of the panel, top of the panel.
    assert marks[0]["x"] > graph_cells[3]["width"] / 2
    assert marks[0]["y"] < graph_cells[3]["height"] / 4
    assert rightmost_x <= graph_cells[3]["width"]


def test_global_minimum_is_drawn_after_regular_tree_vertices(
    construction, composition_settings
):
    tree = build_graph_tree_persistence_panels(
        construction, construction.thresholds[4], composition_settings
    )["tree"].to_dict(validate=True)
    node_layer = next(
        layer for layer in tree["layer"] if layer["mark"]["type"] == "point"
    )
    nodes = _dataset_values(tree, node_layer["data"])
    flags = [bool(row["is_critical"]) for row in nodes]
    assert flags == sorted(flags)
    minimum = construction.critical_ids["global_minimum"]
    assert nodes[-len(construction.critical_ids) :][0]["is_critical"]
    assert minimum in {row["sample_id"] for row in nodes if row["is_critical"]}
    minimum_row = next(
        i for i, row in enumerate(nodes) if row["sample_id"] == minimum
    )
    assert all(
        minimum_row > i for i, row in enumerate(nodes) if not row["is_critical"]
    )
    # The global minimum outranks every other vertex, and the layer orders its marks by
    # that rank explicitly, so the renderer cannot draw a regular vertex over it.
    assert minimum_row == len(nodes) - 1
    ranks = {row["sample_id"]: row["draw_rank"] for row in nodes}
    assert ranks[minimum] == max(ranks.values())
    assert sorted(ranks.values()).count(ranks[minimum]) == 1
    order = node_layer["encoding"]["order"]
    assert order["field"] == "draw_rank"
    assert order["sort"] == "ascending"


def test_graph_row_is_rotated_180_degrees_in_the_plane(saddle_data):
    appearance = figure_module._appearance(SETTINGS)
    azimuth = math.radians(float(appearance["camera_azimuth_degrees"]))
    right = (-math.sin(azimuth), math.cos(azimuth))
    projector = figure_module._projector(appearance)
    # The unrotated camera reads the horizontal screen axis as (x - 0.5, y - 0.5)
    # . right; the 180 degree rotation negates it (x -> 1 - x, y -> 1 - y).
    for x, y in ((0.0, 0.0), (0.2, 0.7), (1.0, 0.4)):
        expected = -((x - 0.5) * right[0] + (y - 0.5) * right[1])
        assert float(projector(x, y, 0.3)[0]) == pytest.approx(expected)
    # The mark data uses that projector, and the height axis is untouched.
    for row in saddle_data.samples:
        expected = -((row["x"] - 0.5) * right[0] + (row["y"] - 0.5) * right[1])
        assert row["projected_x"] == pytest.approx(expected)
    assert float(projector(0.3, 0.8, 1.0)[1]) > float(projector(0.3, 0.8, 0.0)[1])


def test_composed_figure_is_a_heading_row_plus_three_four_column_rows(
    composition_spec,
):
    heading, rows = _grid_rows(composition_spec)
    assert len(composition_spec["vconcat"]) == 5
    assert len(heading) == 5 and all(len(row) == 5 for row in rows)
    assert [cell["data"]["values"][0]["label"] for cell in heading[1:]] == [
        "1. First peak appears",
        "2. Second peak appears",
        "3. Peaks merge",
        "4. Scan complete",
    ]
    assert [row[0]["data"]["values"][0]["label"] for row in rows] == [
        "Graph",
        "Tree",
        "Persistence",
    ]
    serialized = str(composition_spec)
    for removed_text in ("Before the sweep", "Manifold", "Regular"):
        assert removed_text not in serialized
    assert not _contains_mark(composition_spec, "image")
    # Graph row is geographic; Tree and Persistence rows are cartesian layers.
    assert all("projection" in cell for cell in rows[0][1:])
    assert not any("projection" in cell for row in rows[1:] for cell in row[1:])


def test_text_sizes_are_16_for_axes_and_20_for_headings_and_row_titles(
    composition_spec,
):
    heading, rows = _grid_rows(composition_spec)
    # Stage headings and row titles are 20 pt, set apart from each other.
    assert {cell["mark"]["fontSize"] for cell in heading} == {20}
    assert {row[0]["mark"]["fontSize"] for row in rows} == {20}
    # Every axis title and tick label is 16 pt: the Tree and Persistence axes read
    # the figure-wide axis config, the Graph row's title is a text mark.
    assert composition_spec["config"]["axis"]["titleFontSize"] == 16
    assert composition_spec["config"]["axis"]["labelFontSize"] == 16
    graph_title = next(
        layer["mark"]
        for layer in rows[0][1]["layer"]
        if isinstance(layer.get("mark"), dict) and layer["mark"].get("type") == "text"
    )
    assert graph_title["fontSize"] == 16
    assert graph_title["angle"] == 270


def test_persistence_panels_follow_the_four_stage_events(
    construction, composition_spec, composition_settings
):
    _, rows = _grid_rows(composition_spec)
    scalars = dict(
        zip(
            construction.sample_table["critical_name"],
            construction.sample_table["scalar"],
            strict=True,
        )
    )
    # Every peak is filled, the one appearing at a stage included. The global peak
    # stays in the last column, where its pair is closed at the global minimum.
    expected = [
        {"right_peak"},
        {"right_peak", "left_peak"},
        {"right_peak", "left_peak"},
        {"right_peak", "left_peak"},
    ]
    appearance = composition_settings["figures"]["figures"]["graph_tree_persistence"]
    # The closed-peak colour is the shared dark grey of the palette module.
    assert appearance["persistence_dead_color"] == palette.DARK_GREY
    for cell, threshold, filled_names in zip(
        rows[2][1:], construction.thresholds[1:], expected, strict=True
    ):
        _, _, filled_layer = _persistence_layers(cell)
        filled = _dataset_values(composition_spec, filled_layer["data"])
        assert {row["peak_name"] for row in filled} == filled_names
        assert len(filled) == len(filled_names)
        assert all(row["x"] == threshold for row in filled if row["alive"])
        assert all(row["y"] == row["birth"] for row in filled)
        for row in filled:
            if not row["alive"]:
                assert row["color"] == appearance["persistence_dead_color"]
        by_name = {row["peak_name"]: row for row in filled}
        if "left_peak" in by_name and not by_name["left_peak"]["alive"]:
            assert by_name["left_peak"]["x"] == pytest.approx(scalars["saddle"])
    final_filled = _dataset_values(
        composition_spec, _persistence_layers(rows[2][4])[2]["data"]
    )
    survivor = max(final_filled, key=lambda row: row["birth"])
    assert survivor["x"] == pytest.approx(scalars["global_minimum"])
    assert survivor["alive"] is False


def test_no_persistence_marker_is_hollow(construction, composition_spec):
    # No hollow markers (GLOSSARY, Figure marker rules): the peak appearing at a stage,
    # a merged peak and the global peak closed at the global minimum are all filled.
    _, rows = _grid_rows(composition_spec)
    for cell in rows[2][1:]:
        assert len(cell["layer"]) == 3
        for layer in cell["layer"]:
            assert layer["mark"].get("filled", True) is True
    merge_filled = _dataset_values(
        composition_spec, _persistence_layers(rows[2][3])[2]["data"]
    )
    assert {row["peak_name"] for row in merge_filled} == {"right_peak", "left_peak"}


def test_persistence_markers_use_count_opacity(construction, composition_spec):
    _, rows = _grid_rows(composition_spec)
    _, _, filled_layer = _persistence_layers(rows[2][2])
    for layer in (filled_layer,):
        assert layer["encoding"]["opacity"]["field"] == "count_opacity"
        assert layer["encoding"]["color"]["value"] == palette.INK_GREY
        assert all(0 < row["count_opacity"] <= 1 for row in _dataset_values(composition_spec, layer["data"]))


def test_filled_markers_are_outlined_in_a_darker_shade_of_their_own_fill(
    construction, composition_spec, composition_settings
):
    appearance = composition_settings["figures"]["figures"]["graph_tree_persistence"]
    _, rows = _grid_rows(composition_spec)
    seen = 0
    # Graph and Tree rows: each marker record carries its own outline colour.
    for row in rows[:2]:
        for cell in row[1:]:
            layers = [
                layer for layer in cell["layer"] if layer["mark"]["type"] == "point"
            ]
            assert layers
            for layer in layers:
                assert layer["encoding"]["stroke"]["field"] == "outline_color"
                assert (
                    layer["mark"]["strokeWidth"] == appearance["marker_outline_width"]
                )
                for record in _dataset_values(composition_spec, layer["data"]):
                    assert _lightness(record["outline_color"]) < _lightness(
                        record["color"]
                    )
                    seen += 1
    # Persistence points use the count opacity ramp independently of tree colours.
    for cell in rows[2][1:]:
        _, _, filled = _persistence_layers(cell)
        assert filled["encoding"]["opacity"]["field"] == "count_opacity"
        for record in _dataset_values(composition_spec, filled["data"]):
            assert 0 < record["count_opacity"] <= 1
            seen += 1
    assert seen


def test_threshold_is_a_dashed_rule_on_tree_and_persistence_rows(
    construction, composition_spec, composition_settings
):
    appearance = composition_settings["figures"]["figures"]["graph_tree_persistence"]
    _, rows = _grid_rows(composition_spec)
    dash = appearance["tree_threshold_dash"]
    assert dash and all(value > 0 for value in dash)
    for tree, persistence in zip(rows[1][1:], rows[2][1:], strict=True):
        tree_rule = next(
            layer for layer in tree["layer"] if layer["mark"]["type"] == "rule"
        )
        persistence_rule = _persistence_layers(persistence)[1]
        for rule in (tree_rule, persistence_rule):
            assert rule["mark"]["strokeDash"] == dash


def test_tree_and_persistence_threshold_rules_align(construction, composition_spec):
    _, rows = _grid_rows(composition_spec)
    for tree, persistence, threshold in zip(
        rows[1][1:], rows[2][1:], construction.thresholds[1:], strict=True
    ):
        tree_rule = next(
            layer for layer in tree["layer"] if layer["mark"]["type"] == "rule"
        )
        persistence_rule = _persistence_layers(persistence)[1]
        assert persistence_rule["mark"]["type"] == "rule"
        tree_value = _dataset_values(composition_spec, tree_rule["data"])
        persistence_value = _dataset_values(composition_spec, persistence_rule["data"])
        assert tree_value == persistence_value == [{"threshold": threshold}]
        assert (
            tree_rule["encoding"]["y"]["scale"]
            == persistence_rule["encoding"]["y"]["scale"]
        )
        assert tree["height"] == persistence["height"]
        assert tree["width"] == persistence["width"]


def test_persistence_axes_share_the_tree_domain_and_diagonal_spans_it(
    construction, composition_spec
):
    _, rows = _grid_rows(composition_spec)
    diagonal, _, filled = _persistence_layers(rows[2][1])
    for layer in (diagonal, filled):
        assert layer["encoding"]["x"]["scale"]["domain"] == [-0.03, 1.05]
        assert layer["encoding"]["y"]["scale"]["domain"] == [-0.03, 1.05]
    assert _dataset_values(composition_spec, diagonal["data"]) == [
        {"x": -0.03, "y": -0.03},
        {"x": 1.05, "y": 1.05},
    ]


def test_graph_surface_uses_configured_dimmed_color_and_opacity(
    construction, composition_settings
):
    modified = copy.deepcopy(composition_settings)
    appearance = modified["figures"]["figures"]["graph_tree_persistence"]
    appearance["graph_surface_color"] = "#123456"
    appearance["graph_surface_opacity"] = 0.37
    data = build_graph_tree_persistence_panel_data(
        construction, construction.thresholds[1], modified
    )
    assert data.graph_surface
    assert all(row["opacity"] == 0.37 for row in data.graph_surface)
    # Colors are lightness-shaded per face, so check the hue ordering survives.
    for row in data.graph_surface:
        red, green, blue = (int(row["color"][i : i + 2], 16) for i in (1, 3, 5))
        assert red < green < blue


def test_single_stage_preview_uses_the_three_named_panels(
    construction, composition_settings
):
    spec = build_graph_tree_persistence_preview(
        construction, 1.0, composition_settings
    ).to_dict(validate=True)
    assert len(spec["hconcat"]) == 3
    assert [chart["title"] for chart in spec["hconcat"]] == [
        "Graph",
        "Tree",
        "Persistence",
    ]


@pytest.mark.parametrize(
    "critical_name", ["left_peak", "right_peak", "saddle", "global_minimum"]
)
def test_critical_marker_center_meets_plane_at_its_scalar_value(
    construction, composition_settings, critical_name
):
    sample_id = construction.critical_ids[critical_name]
    sample = construction.sample_table.filter(
        construction.sample_table["sample_id"] == sample_id
    ).row(0, named=True)
    data = build_graph_tree_persistence_panel_data(
        construction, sample["scalar"], composition_settings
    )
    corners = data.sweep_plane["geometry"]["coordinates"][0][:4]
    x, y = sample["x"], sample["y"]
    weights = ((1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y)
    intersection = tuple(
        sum(weight * corner[axis] for weight, corner in zip(weights, corners))
        for axis in (0, 1)
    )
    marker = _sample(data, sample_id)
    assert (marker["projected_x"], marker["projected_y"]) == pytest.approx(
        intersection, abs=1e-12
    )


def _contains_mark(spec, mark_type):
    mark = spec.get("mark")
    if isinstance(mark, dict) and mark.get("type") == mark_type:
        return True
    return any(
        _contains_mark(layer, mark_type)
        for composition in ("layer", "hconcat", "vconcat", "concat")
        for layer in spec.get(composition, [])
        if isinstance(layer, dict)
    )


def test_peak_separator_follows_saddle_for_unequal_peak_heights(construction):
    from dubanevics2026topological.analysis.manifold_graph_tree import scalar_value

    # Projected panel geometry hides the xy-plane basin boundary, so the
    # separator rule is checked on the module's helpers by attribute.
    _appearance = figure_module._appearance
    _lineage_color = figure_module._lineage_color

    saddle = next(
        row
        for row in construction.sample_table.iter_rows(named=True)
        if row["sample_id"] == construction.critical_ids["saddle"]
    )
    boundary = saddle["x"] + saddle["y"]
    appearance = _appearance(load_settings())
    value = float(scalar_value(0.5, 0.5, settings=construction.scientific_settings))
    assert value > saddle["scalar"]
    assert (
        _lineage_color(
            0.5,
            0.5,
            value,
            (value + saddle["scalar"]) / 2,
            saddle["scalar"],
            appearance,
            boundary,
        )
        == appearance["right_peak_color"]
    )


@pytest.mark.parametrize("threshold_index", range(5))
def test_tree_horizontal_span_fraction_changes_only_horizontal_rendering(
    construction, composition_settings, threshold_index
):
    import copy

    baseline = copy.deepcopy(composition_settings)
    baseline["figures"]["figures"]["graph_tree_persistence"][
        "tree_horizontal_span_fraction"
    ] = 1.0
    narrowed = copy.deepcopy(composition_settings)
    threshold = construction.thresholds[threshold_index]

    def tree_layers(settings):
        return build_graph_tree_persistence_panels(construction, threshold, settings)[
            "tree"
        ].to_dict(validate=True)["layer"]

    baseline_layers = tree_layers(baseline)
    narrowed_layers = tree_layers(narrowed)
    for layer_index in (1, 2):
        base_layer = baseline_layers[layer_index]
        narrow_layer = narrowed_layers[layer_index]
        base_domain = base_layer["encoding"]["x"]["scale"]["domain"]
        narrow_domain = narrow_layer["encoding"]["x"]["scale"]["domain"]
        assert base_domain == pytest.approx([-0.08, 1.08])
        assert narrow_domain == pytest.approx([0.5 - 0.58 / 0.65, 0.5 + 0.58 / 0.65])
        # Data equality preserves node and edge y positions and node areas.
        assert base_layer["data"]["values"] == narrow_layer["data"]["values"]
