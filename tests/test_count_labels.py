"""Branch-count colours and labels of the band-smoothed trees."""

from __future__ import annotations

import math

import polars as pl
import pytest
from test_bootstrap_smoothed_tree import _toy
from topographer.plotting import TreePlotConfig, TreePresentationConfig, plot_tree

from dubanevics2026topological.figures._branch_count_style import (
    branch_count_color_encoding,
    branch_count_label_collision,
    branch_count_label_max,
    branch_count_label_mode,
)
from dubanevics2026topological.figures._charts import (
    color_scale,
    plot_data_with_metrics,
    selected_view_as_tree,
)
from dubanevics2026topological.figures._count_labels import (
    LEADER_GAP,
    LEADER_WIDTH,
    ROTATED_ANGLE,
    ROTATED_THICKNESS,
    SIDE_CLEARANCE,
    WILD_TYPE_CLEARANCE,
    PanelGeometry,
    branch_count_labels,
    count_label_font_size,
    count_label_layer,
    internal_branches,
    label_boxes,
    label_extent,
    leader_layer,
    marker_boxes,
    place_side_labels,
    place_step_labels,
    place_tip_labels,
    ridge_count_labels,
    stop_marker_half,
    terminal_tips,
    tree_geometry,
)
from dubanevics2026topological.figures.candidate_tree_biology import estimate_text_width
from dubanevics2026topological.figures.palette import PEAK_GREY
from dubanevics2026topological.simplification.selected_leaf_simplification import (
    absorb_terminal_branches,
)

# The "all" mode labels every tip whatever its count, so the placement tests below
# need not pick counts under the "tips" mode's maximum.
RENDERING = {"font_size": 15, "branch_count_label_mode": "all"}
ROTATED = {**RENDERING, "branch_count_label_collision": "rotate"}
# 10 data units across 400 px (40 px a unit); fitness 0..1 over 250 px.
GEOMETRY = PanelGeometry((0.0, 10.0), (0.0, 1.0), 400.0, 250.0)


def _tips(*rows):
    return pl.DataFrame(
        {
            "dna_sequence": [row[0] for row in rows],
            "x": [float(row[1]) for row in rows],
            "y": [float(row[2]) for row in rows],
            "vertex_count": [row[3] for row in rows],
        }
    )


def _label_box(row, width_digits=2):
    """Pixel box of a label row, from its centre and an estimated digit width."""
    half_height = count_label_font_size(RENDERING) / 2
    half_width = 0.556 * count_label_font_size(RENDERING) * width_digits / 2
    return (
        row["px"] - half_width,
        row["py"] - half_height,
        row["px"] + half_width,
        row["py"] + half_height,
    )


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _text_box(row):
    """Pixel box of a label row, from its text and the figure's width estimate."""
    font = count_label_font_size(RENDERING)
    half_width = estimate_text_width(row["label"], font) / 2
    return (
        row["px"] - half_width,
        row["py"] - font / 2,
        row["px"] + half_width,
        row["py"] + font / 2,
    )


def _gap(a, b):
    """Pixels of clear space between two boxes (negative when they overlap)."""
    return max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])


# -- colour: the post-absorption count through the shared norm -----------------


def test_a_smoothed_branch_count_folds_the_absorbed_members_in():
    _, tree = _toy()

    def counts(absorbed):
        shim = selected_view_as_tree(absorb_terminal_branches(tree, absorbed))
        data = plot_data_with_metrics(shim)
        return {
            (row["source_label"], row["target_label"]): row["vertex_count"]
            for row in data.edge_table.iter_rows(named=True)
        }

    # Nominal: f holds {f, g}; the lone peak d is one vertex below the saddle e.
    assert counts(set()) == {
        ("e", "f"): 2,
        ("c", "b"): 2,
        ("e", "d"): 1,
        ("c", "e"): 1,
    }
    # d is absorbed into its saddle: its vertex and the saddle e join f's branch,
    # which now counts f, g, e and d.
    assert counts({"d"}) == {("c", "f"): 4, ("c", "b"): 2}


def test_branch_colours_use_the_same_scale_as_the_trees_edges():
    _, tree = _toy()
    data = plot_data_with_metrics(
        selected_view_as_tree(absorb_terminal_branches(tree, {"d"}))
    )
    domain = (1.0, 15261.0)
    encoding = branch_count_color_encoding(RENDERING, domain).to_dict()
    # Topographer draws a tree's edges with this scale config; an excerpt's own
    # segments use the encoding built from the same config, so a count has one
    # colour wherever it is drawn.
    chart = plot_tree(
        data,
        config=TreePlotConfig(edge_color_scale=color_scale(RENDERING, domain)),
        presentation=TreePresentationConfig(mode="metric"),
        show_nodes=False,
        show_structural_events=False,
        interactive=False,
        include_config=False,
    ).to_dict()
    edges = next(
        item["encoding"]["color"]
        for item in _walk(chart)
        if isinstance(item.get("encoding"), dict)
        and item["encoding"].get("color", {}).get("field") == "vertex_count"
    )
    assert encoding == edges
    # Log black ramp from count one (faintest) to the largest count, with no
    # special colour for count one.
    assert encoding["scale"] == {
        "scheme": "greys",
        "type": "log",
        "domain": [1.0, 15261.0],
        "range": ["rgba(0,0,0,0.3)", "rgba(0,0,0,1.0)"],
    }
    assert "condition" not in encoding
    assert encoding["field"] == "vertex_count"


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


# -- labels --------------------------------------------------------------------


def test_the_label_style_is_the_peak_grey_in_the_body_font_less_two():
    assert count_label_font_size({"font_size": 15}) == 13
    assert count_label_font_size({"font_size": 18}) == 16
    labels = pl.DataFrame(
        {
            "dna_sequence": ["A"],
            "vertex_count": [12],
            "label": ["12"],
            "px": [5.0],
            "py": [5.0],
        }
    )
    spec = count_label_layer(labels, RENDERING, opacity=0.25).to_dict()
    assert spec["mark"]["color"] == PEAK_GREY == "#737373"
    assert spec["mark"]["fontSize"] == 13
    assert spec["mark"]["opacity"] == 0.25
    # Placed by pixel, so the layer cannot move the tree's own scales.
    assert spec["encoding"]["x"]["scale"] is None
    assert spec["encoding"]["y"]["scale"] is None
    assert [tip["title"] for tip in spec["encoding"]["tooltip"]] == [
        "Branch tip",
        "Branch vertex count",
    ]


def test_the_label_keys_default_to_tips_of_nine_or_fewer_with_a_vertical_nudge():
    assert branch_count_label_max({}) == 9
    assert branch_count_label_max({"branch_count_label_max": 3}) == 3
    assert branch_count_label_mode({}) == "tips"
    assert branch_count_label_mode({"branch_count_label_mode": "all"}) == "all"
    assert branch_count_label_collision({}) == "nudge"
    assert (
        branch_count_label_collision({"branch_count_label_collision": "rotate"})
        == "rotate"
    )


@pytest.mark.parametrize(
    ("key", "reader"),
    [
        ("branch_count_label_mode", branch_count_label_mode),
        ("branch_count_label_collision", branch_count_label_collision),
    ],
)
def test_an_unknown_label_choice_is_refused(key, reader):
    with pytest.raises(ValueError, match=key):
        reader({key: "sideways"})


def test_the_old_minimum_key_is_gone():
    from dubanevics2026topological.figures import _branch_count_style

    assert not hasattr(_branch_count_style, "branch_count_label_min")
    assert not hasattr(_branch_count_style, "DEFAULT_BRANCH_COUNT_LABEL_MIN")


def test_the_tips_mode_labels_only_tips_of_at_most_the_maximum():
    tips = _tips(("A", 2, 0.5, 8), ("B", 5, 0.5, 9), ("C", 8, 0.5, 10))
    tips_mode = {**RENDERING, "branch_count_label_mode": "tips"}
    labels = place_tip_labels(tips, tips_mode, GEOMETRY)
    assert labels["dna_sequence"].to_list() == ["A", "B"]  # ten is too many
    assert labels["label"].to_list() == ["8", "9"]
    raised = place_tip_labels(
        tips, {**tips_mode, "branch_count_label_max": 10}, GEOMETRY
    )
    assert raised["dna_sequence"].to_list() == ["A", "B", "C"]
    none = place_tip_labels(tips, {**tips_mode, "branch_count_label_max": 7}, GEOMETRY)
    assert none.is_empty()
    # The tips mode is the default: the config key alone carries the rule.
    assert place_tip_labels(tips, {"font_size": 15}, GEOMETRY)[
        "dna_sequence"
    ].to_list() == ["A", "B"]


def test_the_all_mode_labels_every_tip_whatever_its_count():
    tips = _tips(("A", 2, 0.5, 1), ("B", 5, 0.5, 9), ("C", 8, 0.5, 12526))
    labels = place_tip_labels(tips, RENDERING, GEOMETRY)
    assert labels["dna_sequence"].to_list() == ["A", "B", "C"]
    # Large counts read with a thousands separator; no sequence name in the text.
    assert labels["label"].to_list() == ["1", "9", "12,526"]


def test_a_label_sits_above_its_tip_and_clears_a_stop_codon_cross():
    tips = _tips(("A", 5, 0.5, 12))
    (row,) = place_tip_labels(tips, RENDERING, GEOMETRY).iter_rows(named=True)
    tip_x, tip_y = GEOMETRY.to_pixels(5.0, 0.5)
    assert row["px"] == pytest.approx(tip_x)  # centred on the tip
    cross = stop_marker_half(RENDERING)
    box = _label_box(row)
    assert box[3] <= tip_y - cross  # the foot is above the cross's top
    cross_box = (tip_x - cross, tip_y - cross, tip_x + cross, tip_y + cross)
    assert not _overlap(box, cross_box)


def test_labels_keep_clear_of_each_other_markers_and_the_colour_bar():
    # Two tips 8 px apart cannot share a row of two-digit labels.
    tips = _tips(("A", 5.0, 0.5, 12), ("B", 5.2, 0.5, 13))
    labels = place_tip_labels(tips, RENDERING, GEOMETRY)
    first, second = (_label_box(row) for row in labels.iter_rows(named=True))
    assert not _overlap(first, second)
    # A colour bar over the usual spot pushes the label off it.
    tip_x, tip_y = GEOMETRY.to_pixels(5.0, 0.5)
    bar = (tip_x - 30, tip_y - 40, tip_x + 30, tip_y - 5)
    (row,) = place_tip_labels(
        _tips(("A", 5, 0.5, 12)), RENDERING, GEOMETRY, [bar]
    ).iter_rows(named=True)
    assert not _overlap(_label_box(row), bar)
    # So does a branch segment passing through it.
    wall = (tip_x - 1, 0.0, tip_x + 1, tip_y - 5)
    (row,) = place_tip_labels(
        _tips(("A", 5, 0.5, 12)), RENDERING, GEOMETRY, lines=[wall]
    ).iter_rows(named=True)
    assert not _overlap(_label_box(row), wall)


def test_a_tip_at_the_top_of_the_panel_is_labelled_beside_it_inside_the_panel():
    (row,) = place_tip_labels(
        _tips(("A", 5, 1.0, 12526)), RENDERING, GEOMETRY
    ).iter_rows(named=True)
    tip_x, _ = GEOMETRY.to_pixels(5.0, 1.0)
    box = _label_box(row, width_digits=5.3)
    assert box[1] >= 0 and box[0] >= 0 and box[2] <= GEOMETRY.width
    assert box[0] > tip_x or box[2] < tip_x  # beside the tip, not over its branch


def test_ridge_labels_go_to_the_side_with_room():
    branches = pl.DataFrame(
        {
            "dna_sequence": ["KEM"],
            "vertex_count": [12526],
            "x": [5.0],
            "y_top": [1.0],
            "y_bottom": [0.0],
        }
    )
    middle = GEOMETRY.to_pixels(5.0, 0.5)
    left_blocker = (0.0, middle[1] - 20, middle[0] - 2, middle[1] + 20)
    (right,) = place_side_labels(
        branches, RENDERING, GEOMETRY, [left_blocker]
    ).iter_rows(named=True)
    assert right["px"] > middle[0]
    assert right["py"] == pytest.approx(middle[1])  # level with the ridge's middle
    right_blocker = (middle[0] + 2, middle[1] - 20, GEOMETRY.width, middle[1] + 20)
    (left,) = place_side_labels(
        branches, RENDERING, GEOMETRY, [right_blocker]
    ).iter_rows(named=True)
    assert left["px"] < middle[0]
    # With both sides free the right wins.
    (free,) = place_side_labels(branches, RENDERING, GEOMETRY).iter_rows(named=True)
    assert free["px"] > middle[0]


def test_a_ridge_label_moves_along_the_ridge_when_its_middle_is_blocked():
    branches = pl.DataFrame(
        {
            "dna_sequence": ["KEM"],
            "vertex_count": [12526],
            "x": [5.0],
            "y_top": [1.0],
            "y_bottom": [0.0],
        }
    )
    px, middle = GEOMETRY.to_pixels(5.0, 0.5)
    # Both sides of the ridge's middle are taken (a neighbouring ridge and a mark).
    wall = (px - 80, middle - 20, px - 2, middle + 20)
    other = (px + 2, middle - 20, px + 80, middle + 20)
    (row,) = place_side_labels(branches, RENDERING, GEOMETRY, [wall, other]).iter_rows(
        named=True
    )
    box = _text_box(row)
    assert not _overlap(box, wall) and not _overlap(box, other)
    assert row["py"] != pytest.approx(middle)  # moved up or down the ridge
    top, bottom = GEOMETRY.to_pixels(5.0, 1.0)[1], GEOMETRY.to_pixels(5.0, 0.0)[1]
    assert top <= row["py"] <= bottom  # still level with the ridge
    gap = min(abs(box[0] - px), abs(box[2] - px))
    assert gap == pytest.approx(SIDE_CLEARANCE)  # flush against it
    assert row["leader_x"] is None  # beside its ridge, so no leader


def test_a_ridge_label_goes_above_a_short_ridge_when_both_sides_are_taken():
    branches = pl.DataFrame(
        {
            "dna_sequence": ["KEM"],
            "vertex_count": [12526],
            "x": [5.0],
            "y_top": [0.5],
            "y_bottom": [0.45],
        }
    )
    px, top = GEOMETRY.to_pixels(5.0, 0.5)
    sides = [
        (px - 100, top - 5, px - 2, top + 60),
        (px + 2, top - 5, px + 100, top + 60),
    ]
    (row,) = place_side_labels(branches, RENDERING, GEOMETRY, sides).iter_rows(
        named=True
    )
    box = _text_box(row)
    assert box[3] < top and not any(_overlap(box, side) for side in sides)
    assert row["px"] == pytest.approx(px)  # centred over the ridge's top


def test_a_raised_label_carries_a_leader_down_to_its_tip():
    # Eight tips 12 px apart: two-digit labels (14 px wide) cannot all sit above
    # their tips, so some go up or aside.
    tips = _tips(*((f"T{i}", 3.0 + 0.3 * i, 0.4, 12 + i) for i in range(8)))
    labels = place_tip_labels(tips, RENDERING, GEOMETRY)
    boxes = []
    leaders = []
    for row, tip in zip(
        labels.iter_rows(named=True), tips.iter_rows(named=True), strict=True
    ):
        tip_x, tip_y = GEOMETRY.to_pixels(tip["x"], tip["y"])
        box = _text_box(row)
        boxes.append(box)
        if row["leader_x"] is None:
            # No leader: the label is right over its tip or beside it.
            over = abs(row["px"] - tip_x) < 1 and tip_y - box[3] < 10
            beside = tip_y - 15 <= row["py"] <= tip_y + 15 and (
                _gap(box, (tip_x, tip_y, tip_x, tip_y)) <= SIDE_CLEARANCE + 1
            )
            assert over or beside
            continue
        # A leader stands on the tip's column, from under the label to above the tip.
        assert row["leader_x"] == pytest.approx(tip_x)
        assert row["leader_y0"] == pytest.approx(box[3])
        assert box[3] < row["leader_y1"] < tip_y - stop_marker_half(RENDERING)
        leaders.append(
            (
                row["leader_x"] - LEADER_WIDTH / 2,
                row["leader_y0"],
                row["leader_x"] + LEADER_WIDTH / 2,
                row["leader_y1"],
            )
        )
    assert leaders  # the cluster does force some labels up
    # No label overlaps another label, and no leader runs through a label.
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            assert not _overlap(first, second)
    for leader in leaders:
        assert not any(_overlap(leader, box) for box in boxes)


def test_a_label_next_to_its_tip_needs_no_leader():
    labels = place_tip_labels(_tips(("A", 5, 0.5, 12)), RENDERING, GEOMETRY)
    (row,) = labels.iter_rows(named=True)
    assert (row["leader_x"], row["leader_y0"], row["leader_y1"]) == (None, None, None)
    assert leader_layer(labels) is None


def test_the_leader_layer_is_a_thin_peak_grey_rule_placed_by_pixel():
    labels = place_tip_labels(
        _tips(*((f"T{i}", 3.0 + 0.3 * i, 0.4, 12) for i in range(8))),
        RENDERING,
        GEOMETRY,
    )
    spec = leader_layer(labels).to_dict()
    assert spec["mark"]["type"] == "rule"
    assert spec["mark"]["color"] == PEAK_GREY
    assert spec["mark"]["strokeWidth"] == LEADER_WIDTH < 1.2  # thinner than a branch
    for channel in ("x", "y"):
        assert spec["encoding"][channel]["scale"] is None
    assert spec["encoding"]["y2"]["field"] == "leader_y1"
    # Only the raised labels are in the layer.
    rows = spec["datasets"][spec["data"]["name"]]
    assert len(rows) == labels["leader_x"].drop_nulls().len()


def _toy_data(absorbed=()):
    _, tree = _toy()
    return plot_data_with_metrics(
        selected_view_as_tree(absorb_terminal_branches(tree, set(absorbed)))
    )


def _layer_rows(layer):
    spec = layer.to_dict()
    return spec["datasets"][spec["data"]["name"]]


def test_branch_count_labels_draw_the_leaders_under_the_labels():
    data = _toy_data()
    layers = branch_count_labels(
        data,
        RENDERING,
        size=(400, 250),
        y_domain=(-2.0, 2.0),
    )
    kinds = [layer.to_dict()["mark"]["type"] for layer in layers]
    assert kinds[-1] == "text"
    assert kinds.count("text") == 1
    assert all(kind == "rule" for kind in kinds[:-1])


def test_a_ridge_label_keeps_the_wild_type_clearance_from_the_wild_type_text():
    source = type(
        "Source",
        (),
        {
            "wild_type_marker_table": pl.DataFrame(
                {"x": [5.0], "y": [0.5], "marker_label": ["WT"]}
            )
        },
    )()
    obstacles = marker_boxes(RENDERING, GEOMETRY, wild_type_source=source)
    px, py = GEOMETRY.to_pixels(5.0, 0.5)
    right = px - 9  # wild_type_label_dx: the text is right-aligned there
    *_, label_box = obstacles
    assert label_box[2] == pytest.approx(right + WILD_TYPE_CLEARANCE)
    assert label_box[3] - label_box[1] == pytest.approx(15 + 2 * WILD_TYPE_CLEARANCE)
    text = (right - estimate_text_width("WT", 15), py - 7.5, right, py + 7.5)

    # A ridge whose right-hand spot would stop 2 px short of the text: clear of it,
    # but inside the wild-type clearance, so the label goes elsewhere.
    ridge_x = text[0] - 2 - SIDE_CLEARANCE - estimate_text_width("749", 13)
    ridge = pl.DataFrame(
        {
            "dna_sequence": ["TCL"],
            "vertex_count": [749],
            "x": [ridge_x / GEOMETRY.width * 10.0],
            "y_top": [0.9],
            "y_bottom": [0.1],
        }
    )
    (row,) = place_side_labels(ridge, RENDERING, GEOMETRY, obstacles).iter_rows(
        named=True
    )
    assert _gap(_text_box(row), text) >= WILD_TYPE_CLEARANCE


def test_terminal_tips_are_the_edges_nothing_drains_into():
    _, tree = _toy()
    data = plot_data_with_metrics(
        selected_view_as_tree(absorb_terminal_branches(tree, set()))
    )
    tips = terminal_tips(data)
    assert sorted(zip(tips["dna_sequence"], tips["vertex_count"], strict=True)) == [
        ("b", 2),
        ("d", 1),
        ("f", 2),
    ]
    assert tips["x"].to_list() == sorted(tips["x"].to_list())  # left to right
    assert all(math.isfinite(value) for value in tips["y"])
    assert terminal_tips(
        type("Empty", (), {"edge_table": data.edge_table.clear()})()
    ).is_empty()


# -- every branch: internal branches and their steps -----------------------------


def _steps(*rows):
    """Internal branches: ``(sequence, count, x, y_top, y_bottom, x_end)`` rows."""
    return pl.DataFrame(
        {
            "dna_sequence": [row[0] for row in rows],
            "vertex_count": [row[1] for row in rows],
            "x": [float(row[2]) for row in rows],
            "y_top": [float(row[3]) for row in rows],
            "y_bottom": [float(row[4]) for row in rows],
            "x_end": [float(row[5]) for row in rows],
        }
    )


def test_internal_branches_are_the_edges_something_drains_into():
    data = _toy_data()
    branches = internal_branches(data)
    assert branches.columns == [
        "dna_sequence",
        "vertex_count",
        "x",
        "y_top",
        "y_bottom",
        "x_end",
        "source_tree_node_index",
        "target_tree_node_index",
    ]
    assert branches["dna_sequence"].to_list() == ["e"]
    assert branches["vertex_count"].to_list() == [1]
    nodes = {row["dna_sequence"]: row for row in data.node_table.iter_rows(named=True)}
    row = branches.row(0, named=True)
    # Drawn as a vertical segment at the upper end's x, then a run at the lower
    # end's fitness to the lower end's x.
    assert (row["x"], row["y_top"]) == (nodes["e"]["x"], nodes["e"]["y"])
    assert (row["x_end"], row["y_bottom"]) == (nodes["c"]["x"], nodes["c"]["y"])
    # Terminal and internal branches between them are every branch, once.
    assert terminal_tips(data).height + branches.height == data.edge_table.height
    empty = type("Empty", (), {"edge_table": data.edge_table.clear()})()
    assert internal_branches(empty).is_empty()


def test_a_step_label_sits_centred_under_the_middle_of_its_step():
    steps = _steps(("S", 456, 5.0, 0.52, 0.5, 6.0))
    (row,) = place_step_labels(steps, RENDERING, GEOMETRY).iter_rows(named=True)
    start, bottom = GEOMETRY.to_pixels(5.0, 0.5)
    end = GEOMETRY.to_pixels(6.0, 0.5)[0]
    assert row["px"] == pytest.approx((start + end) / 2)
    assert _text_box(row)[1] == pytest.approx(bottom + SIDE_CLEARANCE)  # just under it
    assert row["label"] == "456"
    assert (row["leader_x"], row["leader_y0"], row["leader_y1"]) == (None, None, None)


def test_neighbouring_step_labels_nudge_down_and_carry_a_leader_up_to_the_step():
    # Two steps 9.2 px apart: two-digit labels (14 px) overlap in one row, but the
    # leader of the lower one clears the upper one's box.
    steps = _steps(("A", 12, 5.0, 0.502, 0.5, 5.0), ("B", 13, 5.23, 0.502, 0.5, 5.23))
    first, second = place_step_labels(steps, RENDERING, GEOMETRY).iter_rows(named=True)
    assert first["leader_x"] is None  # the first keeps the row right under its step
    _, bottom = GEOMETRY.to_pixels(5.23, 0.5)
    box = _text_box(second)
    assert not _overlap(_text_box(first), box)
    assert box[1] > _text_box(first)[1]  # lower, never higher
    # The leader stands under the step (a gap below it) down to the label's top.
    assert second["leader_x"] == pytest.approx(second["px"])
    assert second["leader_y0"] == pytest.approx(bottom + LEADER_GAP)
    assert second["leader_y1"] == pytest.approx(box[1])
    assert second["leader_y0"] < second["leader_y1"]
    leader = (
        second["leader_x"] - LEADER_WIDTH / 2,
        second["leader_y0"],
        second["leader_x"] + LEADER_WIDTH / 2,
        second["leader_y1"],
    )
    assert not _overlap(leader, _text_box(first))
    assert (
        leader_layer(place_step_labels(steps, RENDERING, GEOMETRY)).to_dict()["mark"][
            "type"
        ]
        == "rule"
    )


def test_a_long_vertical_internal_branch_is_labelled_beside_it_like_a_ridge():
    steps = _steps(("V", 7017, 5.0, 0.9, 0.1, 5.5))
    (row,) = place_step_labels(steps, RENDERING, GEOMETRY).iter_rows(named=True)
    px, middle = GEOMETRY.to_pixels(5.0, 0.5)
    box = _text_box(row)
    assert min(abs(box[0] - px), abs(box[2] - px)) == pytest.approx(SIDE_CLEARANCE)
    assert row["py"] == pytest.approx(middle)  # level with the segment's middle
    assert row["leader_x"] is None


def test_a_step_near_the_panel_bottom_is_labelled_inside_the_panel():
    # Under this step there is no room for a label: it goes beside or above it, clear
    # of the step itself (the stub up and the run at the branch's foot).
    steps = _steps(("L", 12526, 5.0, 0.03, 0.02, 5.5))
    start, top = GEOMETRY.to_pixels(5.0, 0.03)
    end, bottom = GEOMETRY.to_pixels(5.5, 0.02)
    own = [(start - 0.6, top - 0.6, start + 0.6, bottom + 0.6)]
    own.append((start, bottom - 0.6, end, bottom + 0.6))
    (row,) = place_step_labels(steps, RENDERING, GEOMETRY, lines=own).iter_rows(
        named=True
    )
    box = _text_box(row)
    assert 0 <= box[1] and box[3] <= GEOMETRY.height and box[2] <= GEOMETRY.width
    assert not any(_overlap(box, segment) for segment in own)


def test_step_labels_keep_clear_of_markers_and_the_colour_bar():
    steps = _steps(("S", 456, 5.0, 0.52, 0.5, 6.0))
    start, bottom = GEOMETRY.to_pixels(5.0, 0.5)
    bar = (start - 20, bottom + 2, start + 60, bottom + 20)
    (row,) = place_step_labels(steps, RENDERING, GEOMETRY, [bar]).iter_rows(named=True)
    assert not _overlap(_text_box(row), bar)


# -- every branch: the driver ----------------------------------------------------


def _text_rows(layers):
    (text,) = [layer for layer in layers if layer.to_dict()["mark"]["type"] == "text"]
    return _layer_rows(text)


def test_the_all_mode_labels_every_branch_with_its_post_absorption_count():
    layers = branch_count_labels(
        _toy_data(), RENDERING, size=(400, 250), y_domain=(-2.0, 2.0)
    )
    rows = _text_rows(layers)
    # Three tips (b and f two vertices each, d one) and the internal branch e (one).
    assert {row["dna_sequence"]: row["label"] for row in rows} == {
        "b": "2",
        "d": "1",
        "f": "2",
        "e": "1",
    }
    # Each branch once, and no two labels on each other.
    assert len({row["dna_sequence"] for row in rows}) == len(rows)
    boxes = [_text_box(row) for row in rows]
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            assert not _overlap(first, second)


def test_the_tips_mode_labels_only_the_small_terminal_tips():
    tips_mode = {**RENDERING, "branch_count_label_mode": "tips"}
    only_d = branch_count_labels(
        _toy_data(),
        {**tips_mode, "branch_count_label_max": 1},
        size=(400, 250),
        y_domain=(-2.0, 2.0),
    )
    assert [row["dna_sequence"] for row in _text_rows(only_d)] == ["d"]
    # The internal branch e (one vertex) is not a tip, so it is never labelled here.
    nine = branch_count_labels(
        _toy_data(), tips_mode, size=(400, 250), y_domain=(-2.0, 2.0)
    )
    assert sorted(row["dna_sequence"] for row in _text_rows(nine)) == ["b", "d", "f"]
    assert (
        branch_count_labels(
            _toy_data(),
            {**tips_mode, "branch_count_label_max": 0},
            size=(400, 250),
            y_domain=(-2.0, 2.0),
        )
        == []
    )


def test_internal_labels_keep_clear_of_the_tip_labels():
    data = _toy_data()
    geometry = tree_geometry(data, RENDERING, size=(400, 250), y_domain=(-2.0, 2.0))
    tips = place_tip_labels(terminal_tips(data), RENDERING, geometry)
    steps = place_step_labels(
        internal_branches(data), RENDERING, geometry, label_boxes(tips, RENDERING)
    )
    assert steps.height == 1
    step = _text_box(steps.row(0, named=True))
    assert not any(_overlap(step, _text_box(row)) for row in tips.iter_rows(named=True))


def test_label_boxes_include_the_leader_of_a_moved_label():
    tips = _tips(*((f"T{i}", 3.0 + 0.3 * i, 0.4, 12 + i) for i in range(8)))
    labels = place_tip_labels(tips, RENDERING, GEOMETRY)
    boxes = label_boxes(labels, RENDERING)
    assert len(boxes) == labels.height + labels["leader_x"].drop_nulls().len()
    first = labels.row(0, named=True)
    width, height = label_extent(first["label"], RENDERING)
    assert boxes[0] == pytest.approx(
        (
            first["px"] - width / 2,
            first["py"] - height / 2,
            first["px"] + width / 2,
            first["py"] + height / 2,
        )
    )


# -- the ridge labels keep their own rule ----------------------------------------


def _all_pairs(data):
    return {
        (int(row["source_tree_node_index"]), int(row["target_tree_node_index"]))
        for row in data.edge_table.iter_rows(named=True)
    }


def test_ridge_labels_label_every_ridge_branch_whatever_the_tip_label_keys_say():
    data = _toy_data()
    geometry = PanelGeometry((-1.0, 3.0), (-2.0, 2.0), 400.0, 250.0)
    pairs = _all_pairs(data)
    strict = {
        **RENDERING,
        "branch_count_label_mode": "tips",
        "branch_count_label_max": 0,  # no tip would be labelled
    }
    layers = ridge_count_labels(
        data, pairs, set(), strict, geometry, dimmed_opacity=0.25
    )
    (layer,) = layers
    assert layer.to_dict()["mark"]["opacity"] == 1.0
    rows = _layer_rows(layer)
    # Every branch of the ridge, once: no branch is labelled twice.
    assert sorted(row["dna_sequence"] for row in rows) == ["b", "d", "e", "f"]
    assert sorted(row["label"] for row in rows) == ["1", "1", "2", "2"]


def test_ridge_labels_follow_the_branch_opacity():
    data = _toy_data()
    geometry = PanelGeometry((-1.0, 3.0), (-2.0, 2.0), 400.0, 250.0)
    edges = {
        row["target_label"]: (
            int(row["source_tree_node_index"]),
            int(row["target_tree_node_index"]),
        )
        for row in data.edge_table.iter_rows(named=True)
    }
    dimmed, focused = layers = ridge_count_labels(
        data,
        {edges["b"], edges["e"]},
        {edges["d"]},
        RENDERING,
        geometry,
        dimmed_opacity=0.25,
    )
    assert dimmed.to_dict()["mark"]["opacity"] == 0.25
    assert focused.to_dict()["mark"]["opacity"] == 1.0
    assert [row["dna_sequence"] for row in _layer_rows(dimmed)] == ["d"]
    assert sorted(row["dna_sequence"] for row in _layer_rows(focused)) == ["b", "e"]
    assert len(layers) == 2


# -- the rotated variant ---------------------------------------------------------


def test_a_rotated_label_is_a_digit_high_and_as_long_as_its_text():
    font = count_label_font_size(ROTATED)
    length = estimate_text_width("12,526", font)
    assert label_extent("12,526", RENDERING) == (length, font)
    width, height = label_extent("12,526", ROTATED)
    assert width == pytest.approx(font * ROTATED_THICKNESS)
    assert width < font
    assert height == pytest.approx(length)


def test_the_font_reduction_key_sets_the_label_size():
    assert count_label_font_size({"font_size": 15}) == 13
    assert (
        count_label_font_size({"font_size": 15, "branch_count_label_font_reduction": 4})
        == 11
    )


def test_rotated_labels_fit_between_tips_where_horizontal_labels_collide():
    # Two tips 14 px apart: two-digit labels (14 px wide, a space apart) cannot share a row, so
    # the nudge raises one; rotated labels (9.4 px wide) can.
    tips = _tips(("A", 5.0, 0.5, 12), ("B", 5.35, 0.5, 13))
    nudged = place_tip_labels(tips, RENDERING, GEOMETRY)
    assert nudged["leader_x"].drop_nulls().len() == 1
    assert nudged["py"][0] != nudged["py"][1]
    rotated = place_tip_labels(tips, ROTATED, GEOMETRY)
    assert rotated["leader_x"].null_count() == 2
    assert rotated["py"][0] == rotated["py"][1]
    first, second = label_boxes(rotated, ROTATED)
    assert not _overlap(first, second)
    # The box is as tall as the text is long and a digit wide.
    assert first[3] - first[1] == pytest.approx(estimate_text_width("12", 13))
    assert first[2] - first[0] == pytest.approx(13 * ROTATED_THICKNESS)


def test_a_collision_left_after_rotating_is_still_resolved():
    # Tips 2 px apart collide even turned: the second label is moved off the first
    # (here beside its tip, a leader's column being the first label's), as without
    # the rotation.
    tips = _tips(("A", 5.0, 0.5, 12), ("B", 5.05, 0.5, 13))
    rotated = place_tip_labels(tips, ROTATED, GEOMETRY)
    first, second = label_boxes(rotated, ROTATED)[:2]
    assert not _overlap(first, second)
    assert (rotated["px"][0], rotated["py"][0]) != (rotated["px"][1], rotated["py"][1])


def test_only_a_rotated_layer_turns_its_text():
    labels = pl.DataFrame(
        {
            "dna_sequence": ["A"],
            "vertex_count": [12],
            "label": ["12"],
            "px": [5.0],
            "py": [5.0],
        }
    )
    assert "angle" not in count_label_layer(labels, RENDERING).to_dict()["mark"]
    turned = count_label_layer(labels, ROTATED).to_dict()["mark"]
    assert turned["angle"] == ROTATED_ANGLE == 270  # a quarter turn, reading upward
    # Still centred on its pixel, so the box the placement reserved is the one drawn.
    assert turned["align"] == "center" and turned["baseline"] == "middle"


def test_rotated_step_labels_hang_a_digit_wide_under_the_step():
    steps = _steps(("S", 456, 5.0, 0.52, 0.5, 6.0))
    (row,) = place_step_labels(steps, ROTATED, GEOMETRY).iter_rows(named=True)
    (box,) = label_boxes(place_step_labels(steps, ROTATED, GEOMETRY), ROTATED)
    _, bottom = GEOMETRY.to_pixels(5.0, 0.5)
    assert box[1] == pytest.approx(bottom + SIDE_CLEARANCE)
    assert box[2] - box[0] == pytest.approx(13 * ROTATED_THICKNESS)
    assert box[3] - box[1] == pytest.approx(estimate_text_width("456", 13))
    assert row["leader_x"] is None
