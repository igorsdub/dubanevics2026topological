"""Outcome key placement: the key sits at the corner of its tree panel that holds the least."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import vl_convert as vlc
from _vega import _walk
from test_candidate_tree_contrast import _tree

from dubanevics2026topological.figures._charts import (
    padded_scalar_domain,
    plot_data_with_saved_metrics,
    publication_scalar_scales,
    study_biology_tree_chart,
)
from dubanevics2026topological.figures.candidate_tree_biology import (
    INSET_CORNERS,
    OutcomeKey,
    _tree_axes_x_domain,
    estimate_text_width,
    outcome_corner_counts,
    outcome_key_metrics,
    outcome_legend_corner,
    outcome_legend_layer,
    tree_x_domain,
)

# -- text widths -----------------------------------------------------------------


def test_text_width_follows_arial_metrics():
    # Measured in a rendered figure: 209.5 px for this label at 15 px.
    assert estimate_text_width("absorbed: persistence ≤ τ (127)", 15) == pytest.approx(
        209.5, abs=2.5
    )
    assert estimate_text_width("WT", 15) == pytest.approx(23.3, abs=0.1)
    assert estimate_text_width("", 15) == 0.0
    assert estimate_text_width("kept (9)", 30) == pytest.approx(
        2 * estimate_text_width("kept (9)", 15)
    )
    bold = estimate_text_width("Peak", 15, bold=True)
    assert bold > estimate_text_width("Peak", 15)
    # A character outside the table counts as a digit; two digits are twice one.
    assert estimate_text_width("é", 10) == pytest.approx(estimate_text_width("0", 10))


# -- the tree's pixel model ------------------------------------------------------


def test_x_domain_widens_by_the_pixel_padding_before_rounding_outward():
    segments = pl.DataFrame(
        {
            "x_start": [0.0, 168.0],
            "x_end": [0.0, 168.0],
            "y_start": [0.0, 0.0],
            "y_end": [1.0, 1.0],
        }
    )
    # Vega's padding widens [0, 168] to [-7.875, 175.875]; nice then gives -20..180.
    assert tree_x_domain(segments, 18, 420) == (-20.0, 180.0)
    assert tree_x_domain(segments) == (0.0, 180.0)
    # The padding is capped at a quarter of the width.
    assert tree_x_domain(segments, 400, 420) == tree_x_domain(segments, 105, 420)


def _line_points(node):
    return [
        item
        for part in _walk(node)
        if part.get("marktype") == "line"
        for item in part.get("items", [])
    ]


def test_pixel_model_matches_vegas_scales_on_a_real_tree():
    tree = _tree()
    plot = tree.plot_data
    labels = dict(
        zip(
            plot.node_table["tree_node_index"],
            plot.node_table["dna_sequence"],
            strict=True,
        )
    )
    branch_table = pl.DataFrame(
        {
            "source_tree_node_identity": [
                labels[i] for i in plot.edge_table["source_tree_node_index"]
            ],
            "target_tree_node_identity": [
                labels[i] for i in plot.edge_table["target_tree_node_index"]
            ],
            "branch_vertex_count": 1,
            "aa_hamming": 0,
            "horton_strahler_order": 1,
        }
    )
    data = plot_data_with_saved_metrics(tree, branch_table)
    scalar_domain = (-0.3, 0.7)
    width, height = 420, 200
    chart = study_biology_tree_chart(
        data,
        SimpleNamespace(scalar_domain=scalar_domain),
        {"tree": {"width": width, "height": height}},
        {},
    )
    chart = publication_scalar_scales(chart, scalar_domain, {})
    drawn = _line_points(vlc.vegalite_to_scenegraph(chart.to_dict()))

    segments = data.edge_segment_table
    x_domain = _tree_axes_x_domain(segments, {}, width)
    y_domain = padded_scalar_domain(scalar_domain, {})
    model_x = {
        round((value - x_domain[0]) / (x_domain[1] - x_domain[0]) * width, 4)
        for column in ("x_start", "x_end")
        for value in segments[column].to_list()
    }
    model_y = {
        round((y_domain[1] - value) / (y_domain[1] - y_domain[0]) * height, 4)
        for column in ("y_start", "y_end")
        for value in segments[column].to_list()
    }
    assert {round(item["x"], 4) for item in drawn} == model_x
    assert {round(item["y"], 4) for item in drawn} == model_y


# -- the key's geometry ----------------------------------------------------------

ENTRIES = [("kept (99)", "#111111", True), ("absorbed (61)", "#595959", False)]


def _placed(layer):
    spec = layer.to_dict()
    return [
        item["mark"]
        for item in _walk(spec)
        if isinstance(item.get("mark"), dict) and "x" in item["mark"]
    ]


@pytest.mark.parametrize("corner", INSET_CORNERS)
def test_key_sits_flush_in_its_corner(corner):
    width, height = 420.0, 250.0
    layer = outcome_legend_layer(
        ENTRIES, {}, width=width, height=height, title="Title", corner=corner
    )
    marks = _placed(layer)
    metrics = outcome_key_metrics(OutcomeKey(ENTRIES, "Title"), {})
    inset = metrics.inset
    rows = sorted({mark["y"] for mark in marks})
    assert len(rows) == 3  # title and two entries
    vertical, horizontal = corner.split("-")
    if vertical == "top":
        assert rows[0] == pytest.approx(inset + metrics.line_height / 2)
    else:
        assert rows[-1] == pytest.approx(height - inset - metrics.line_height / 2)
    assert np.diff(rows) == pytest.approx([metrics.line_height] * 2)
    labels = [
        mark for mark in marks if mark["type"] == "text" and "fontWeight" not in mark
    ]
    glyphs = [mark for mark in marks if mark["type"] == "point"]
    longest = max(estimate_text_width(mark["text"], 15) for mark in labels)
    if horizontal == "right":
        # The longest label ends at the inset; markers sit a glyph gap left of the text.
        ends = [mark["x"] + estimate_text_width(mark["text"], 15) for mark in labels]
        assert max(ends) == pytest.approx(width - inset)
        assert glyphs[0]["x"] == pytest.approx(width - inset - longest - 14)
    else:
        assert glyphs[0]["x"] == pytest.approx(inset + metrics.marker_half)
        assert labels[0]["x"] == pytest.approx(glyphs[0]["x"] + 14)


def test_key_corner_must_be_one_of_the_four():
    with pytest.raises(ValueError, match="corner"):
        outcome_legend_layer(ENTRIES, {}, width=420, height=250, corner="bottom")
    assert not _placed(outcome_legend_layer([], {}, width=420, height=250))


# -- the corner choice -----------------------------------------------------------

SIZE = (420.0, 250.0)
SCALAR_DOMAIN = (-1.0, 1.0)
KEY = OutcomeKey([("kept (9)", "#111111", True)], "Peak")


def _x(u: float) -> float:
    """Data x of a fraction ``u`` of the panel width, for a frame spanning 0..100."""
    return -10.0 + 120.0 * u


def _y(v: float) -> float:
    """Data y of a fraction ``v`` of the panel height, from the bottom."""
    return -1.1 + 2.2 * v


def _segment(x0, y0, x1, y1):
    return {"x_start": x0, "x_end": x1, "y_start": y0, "y_end": y1}


def _block(u, v):
    """A short vertical branch centred at panel fractions ``(u, v)``."""
    return _segment(_x(u), _y(v - 0.03), _x(u), _y(v + 0.03))


# A level branch across the middle fixes the x domain at -10..110 and touches no corner.
FRAME = _segment(0.0, 0.0, 100.0, 0.0)
TOP_RIGHT, BOTTOM_RIGHT = _block(0.9, 0.9), _block(0.9, 0.1)
TOP_LEFT, BOTTOM_LEFT = _block(0.1, 0.9), _block(0.1, 0.1)
NO_POINTS = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})


def _points(*positions):
    return pl.DataFrame(
        {"x": [_x(u) for u, _ in positions], "y": [_y(v) for _, v in positions]}
    )


def _corner(*branches, stops=NO_POINTS, wild_type=None, rendering=None, key=KEY):
    return outcome_legend_corner(
        key,
        rendering or {},
        size=SIZE,
        scalar_domain=SCALAR_DOMAIN,
        segments=pl.DataFrame([FRAME, *branches]),
        stop_points=stops,
        wild_type_source=wild_type,
    )


def test_an_empty_tree_ties_every_corner_and_top_right_wins():
    assert _corner() == "top-right"


def test_outcome_legend_corner_respects_forced_rendering():
    assert _corner(rendering={"outcome_legend_corner": "bottom-left"}) == "bottom-left"
    assert (
        _corner(TOP_RIGHT, rendering={"outcome_legend_corner": "top-right"})
        == "top-right"
    )
    with pytest.raises(ValueError, match="Unknown outcome_legend_corner 'invalid'"):
        _corner(rendering={"outcome_legend_corner": "invalid"})


def test_each_occupied_corner_hands_the_key_to_the_next_one():
    assert _corner(TOP_RIGHT) == "bottom-right"
    assert _corner(TOP_RIGHT, BOTTOM_RIGHT) == "top-left"
    assert _corner(TOP_RIGHT, BOTTOM_RIGHT, TOP_LEFT) == "bottom-left"
    assert _corner(BOTTOM_RIGHT, TOP_LEFT, BOTTOM_LEFT) == "top-right"


def test_when_every_corner_is_crossed_the_least_crossed_wins():
    def branches(counts):
        blocks = []
        for (u, v), count in zip(
            [(0.9, 0.9), (0.9, 0.1), (0.1, 0.9), (0.1, 0.1)], counts, strict=True
        ):
            blocks += [_block(u + 0.01 * index, v) for index in range(count)]
        return blocks

    for counts, expected in [
        ((3, 2, 4, 1), "bottom-left"),
        ((3, 2, 4, 5), "bottom-right"),
        ((2, 2, 2, 2), "top-right"),
        ((4, 3, 1, 3), "top-left"),
    ]:
        assert _corner(*branches(counts)) == expected


def test_the_count_is_of_distinct_branches_not_of_their_length():
    long_branch = _segment(_x(0.82), _y(0.9), _x(0.99), _y(0.9))
    short_pair = [_block(0.1, 0.9), _block(0.12, 0.9)]
    assert _corner(long_branch, *short_pair, BOTTOM_RIGHT, BOTTOM_LEFT) == "top-right"


def test_zero_length_branches_draw_nothing():
    assert _corner(_segment(_x(0.9), _y(0.9), _x(0.9), _y(0.9))) == "top-right"


def test_stop_crosses_count_and_so_does_their_extent():
    assert _corner(stops=_points((0.9, 0.9))) == "bottom-right"
    # Every corner holds one branch; the cross centred just outside the bottom-left
    # box (77.1 px wide) still reaches in by half its width, so it ties and top-right wins.
    near = _points((78.5 / 420, 0.1))
    blocked = (TOP_RIGHT, BOTTOM_RIGHT, TOP_LEFT)
    assert _corner(*blocked, stops=near) == "top-right"
    far = _points((90 / 420, 0.1))
    assert _corner(*blocked, stops=far) == "bottom-left"


def _wild_type(u, v):
    table = pl.DataFrame({"x": [_x(u)], "y": [_y(v)], "marker_label": ["WT"]})
    return SimpleNamespace(wild_type_marker_table=table)


def test_the_wild_type_star_counts():
    assert _corner(wild_type=_wild_type(0.9, 0.9)) == "bottom-right"


def test_the_wild_type_label_counts_where_the_star_is_outside_the_box():
    # The star is right of the top-left box and its margin; its label runs left into it.
    star = _wild_type(95 / 420, 0.9)
    blocked = (TOP_RIGHT, BOTTOM_RIGHT)
    assert _corner(*blocked, wild_type=star) == "bottom-left"
    assert _corner(*blocked) == "top-left"
    no_label = SimpleNamespace(
        wild_type_marker_table=star.wild_type_marker_table.drop("marker_label")
    )
    assert _corner(*blocked, wild_type=no_label) == "top-left"


def test_a_corner_that_is_overlapped_loses_to_one_that_is_only_close():
    # A branch just past the bottom-left box (inside the clearance) does not overlap
    # the key, while a branch crosses the top-right box: bottom-left wins, though the
    # two corners hold one branch each once the clearance is counted.
    near = _block(0.19, 0.1)
    assert _corner(TOP_RIGHT, BOTTOM_RIGHT, TOP_LEFT, near) == "bottom-left"
    # Without the clearance margin the branch is not even close.
    tight = {"outcome_legend_clearance": 0}
    assert _corner(near, rendering=tight) == "top-right"


def test_the_clearance_breaks_a_tie_between_corners_nothing_overlaps():
    near = _block(0.81, 0.9)  # just outside the top-right box, inside the margin
    assert _corner(near) == "bottom-right"
    assert _corner(near, rendering={"outcome_legend_clearance": 0}) == "top-right"


def test_the_key_box_is_the_keys_own_size():
    # A branch 30 per cent in from the right edge is clear of a short key's box but
    # crosses a key with long labels.
    branch = _block(0.7, 0.9)
    assert _corner(branch) == "top-right"
    long_key = OutcomeKey(
        [("elder-protected with a very long label (123,456)", "#ff7f0e", True)], "Peak"
    )
    assert _corner(branch, key=long_key) == "bottom-right"


def _random_tree(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(60):
        along = rng.uniform(0, 100, 2)
        across = rng.uniform(-1.0, 1.0, 2)
        if rng.random() < 0.5:
            rows.append(_segment(along[0], across[0], along[0], across[1]))
        else:
            rows.append(_segment(along[0], across[0], along[1], across[0]))
    return pl.DataFrame(rows)


def _exact_scores(segments, key, rendering):
    """Count the segments reaching each corner box, by rectangle overlap in pixels."""
    width, height = SIZE
    x_domain = _tree_axes_x_domain(segments, rendering, width)
    y_domain = padded_scalar_domain(SCALAR_DOMAIN, rendering)
    metrics = outcome_key_metrics(key, rendering)
    clearance = float(rendering.get("outcome_legend_clearance", 6))
    x0, x1 = (
        (segments[column].to_numpy() - x_domain[0])
        / (x_domain[1] - x_domain[0])
        * width
        for column in ("x_start", "x_end")
    )
    y0, y1 = (
        (y_domain[1] - segments[column].to_numpy())
        / (y_domain[1] - y_domain[0])
        * height
        for column in ("y_start", "y_end")
    )
    left, right = np.minimum(x0, x1), np.maximum(x0, x1)
    top, bottom = np.minimum(y0, y1), np.maximum(y0, y1)
    scores = {}
    for corner in INSET_CORNERS:
        vertical, horizontal = corner.split("-")
        counts = []
        for margin in (0.0, clearance):
            box_width = metrics.inset + metrics.width + margin
            box_height = metrics.inset + metrics.height + margin
            in_x = (
                right >= width - box_width
                if horizontal == "right"
                else left <= box_width
            )
            in_y = (
                bottom >= height - box_height
                if vertical == "bottom"
                else top <= box_height
            )
            counts.append(int(np.count_nonzero(in_x & in_y)))
        scores[corner] = tuple(counts)
    return scores


@pytest.mark.parametrize("seed", range(25))
def test_the_chosen_corner_has_the_fewest_overlaps_on_random_trees(seed):
    segments = _random_tree(seed)
    exact = _exact_scores(segments, KEY, {})
    counted = outcome_corner_counts(
        KEY,
        {},
        size=SIZE,
        scalar_domain=SCALAR_DOMAIN,
        segments=segments,
        stop_points=NO_POINTS,
        wild_type_source=None,
    )
    assert counted == exact
    chosen = outcome_legend_corner(
        KEY,
        {},
        size=SIZE,
        scalar_domain=SCALAR_DOMAIN,
        segments=segments,
        stop_points=NO_POINTS,
    )
    assert exact[chosen] == min(exact.values())
    # Ties on both counts go to the first corner in the order top-right first.
    assert chosen == min(INSET_CORNERS, key=lambda corner: exact[corner])
