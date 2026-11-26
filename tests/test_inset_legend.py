"""Shared inset legend: corner choice from data occupancy, no box."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dubanevics2026topological.figures.candidate_tree_biology import (
    INSET_CORNERS,
    choose_legend_orient,
    corner_counts,
    emptiest_corner,
    inset_legend,
    resolve_legend_orient,
    segment_sample_points,
    tree_legend_orient,
    tree_x_domain,
)

UNIT = (0.0, 1.0)


def _cloud(x_range, y_range, n=200):
    rng = np.random.default_rng(0)
    return rng.uniform(*x_range, n), rng.uniform(*y_range, n)


def test_corner_is_the_emptiest_box_in_plotted_coordinates():
    xs, ys = _cloud((0.0, 0.55), (0.0, 1.0))  # right 45 per cent empty
    orient = choose_legend_orient(xs, ys, UNIT, UNIT, {})
    assert orient == "top-right"  # tie between right corners goes to the top


def test_occupied_top_right_moves_the_legend_to_the_next_emptiest_corner():
    left = _cloud((0.0, 0.4), (0.0, 1.0))
    top_right = _cloud((0.7, 1.0), (0.75, 1.0), 50)
    xs = np.concatenate([left[0], top_right[0]])
    ys = np.concatenate([left[1], top_right[1]])
    assert choose_legend_orient(xs, ys, UNIT, UNIT, {}) == "bottom-right"


def test_full_coverage_falls_back_to_the_strip_below_the_tree():
    xs, ys = _cloud((0.0, 1.0), (0.0, 1.0), 2000)
    assert choose_legend_orient(xs, ys, UNIT, UNIT, {}) == "bottom"
    config = {"legend_fallback_orient": "top"}
    assert choose_legend_orient(xs, ys, UNIT, UNIT, config) == "top"


def test_explicit_orient_overrides_the_data_driven_choice():
    xs, ys = _cloud((0.0, 0.4), (0.0, 1.0))
    assert resolve_legend_orient({"legend_orient": "left"}, xs, ys, UNIT, UNIT) == (
        "left"
    )
    assert resolve_legend_orient({"legend_orient": "auto"}, xs, ys, UNIT, UNIT) in (
        INSET_CORNERS
    )


def test_tree_orient_uses_segment_extent_and_the_padded_fitness_axis():
    # High-fitness cluster top-left, long branch along the bottom: the right half
    # near the top is empty.
    segments = pl.DataFrame(
        {
            "x_start": [0.0, 0.0, 0.0],
            "x_end": [0.0, 0.3, 1.0],
            "y_start": [0.0, 1.4, -1.0],
            "y_end": [1.4, 1.4, -1.0],
        }
    )
    orient = tree_legend_orient(segments, (-1.0, 1.4), {})
    assert orient == "top-right"


def test_inset_legend_has_no_box_and_an_offset_only_for_corners():
    corner = inset_legend("Title", {"legend_inset_offset": 8}, orient="top-left")
    spec = corner.to_dict()
    assert spec["orient"] == "top-left" and spec["offset"] == 8
    assert not {"fillColor", "strokeColor", "padding"} & set(spec)
    strip = inset_legend("Title", {}, orient="bottom").to_dict()
    assert strip["orient"] == "bottom" and "offset" not in strip


def test_samples_are_proportional_to_segment_length_and_keep_their_owner():
    segments = pl.DataFrame(
        {
            "x_start": [0.0, 0.0],
            "x_end": [1.0, 0.1],
            "y_start": [0.0, 0.0],
            "y_end": [0.0, 0.0],
        }
    )
    xs, ys, owners = segment_sample_points(segments, UNIT, UNIT, step=0.01)
    assert xs.size == ys.size == owners.size
    long_count, short_count = (np.count_nonzero(owners == i) for i in (0, 1))
    assert long_count == 101 and short_count == 11  # one per 0.01 plus the end
    assert xs[owners == 0].min() == 0.0 and xs[owners == 0].max() == 1.0


def _short(x, y):
    return {"x_start": x, "x_end": x, "y_start": y, "y_end": y + 0.005}


def test_corner_score_counts_distinct_segments_not_samples():
    # One long branch lies along the top-right box (about 40 samples, one
    # segment); three tiny branches sit in the top-left box and four in each
    # bottom box (two samples each).
    rows = [{"x_start": 0.58, "x_end": 1.0, "y_start": 0.9, "y_end": 0.9}]
    rows += [_short(0.05 * i + 0.02, 0.85) for i in range(3)]
    rows += [_short(0.05 * i + 0.02, 0.05) for i in range(4)]
    rows += [_short(0.95 - 0.05 * i, 0.05) for i in range(4)]
    segments = pl.DataFrame(rows)
    xs, ys, owners = segment_sample_points(segments, UNIT, UNIT)
    loose = {"legend_collision_fraction": 1.0}
    assert choose_legend_orient(xs, ys, UNIT, UNIT, loose) == "top-left"
    assert (
        choose_legend_orient(
            xs, ys, UNIT, UNIT, loose, owners=owners, total=segments.height
        )
        == "top-right"
    )


def test_collision_fallback_triggers_on_a_fraction_of_segments():
    # Ten horizontal branches span the whole axes: every corner is crossed by
    # all of them, far above 2 per cent of the segments.
    segments = pl.DataFrame(
        {
            "x_start": [0.0] * 10,
            "x_end": [1.0] * 10,
            "y_start": [i / 9 for i in range(10)],
            "y_end": [i / 9 for i in range(10)],
        }
    )
    xs, ys, owners = segment_sample_points(segments, UNIT, UNIT)
    kwargs = {"owners": owners, "total": segments.height}
    assert choose_legend_orient(xs, ys, UNIT, UNIT, {}, **kwargs) == "bottom"
    # Configurable: tolerating every segment keeps the corner.
    tolerant = {"legend_collision_fraction": 1.0}
    assert choose_legend_orient(xs, ys, UNIT, UNIT, tolerant, **kwargs) in (
        INSET_CORNERS
    )


def test_tree_x_domain_includes_zero_and_rounds_outward_like_vega():
    segments = pl.DataFrame(
        {
            "x_start": [-3.2, -1.0],
            "x_end": [0.0, -1.0],
            "y_start": [0.0, 0.0],
            "y_end": [0.0, 1.0],
        }
    )
    low, high = tree_x_domain(segments)
    assert low <= -3.2 and high == 0.0
    assert low == pytest.approx(-3.5)  # a 0.5 step, as Vega's nice picks here


def test_tree_orient_accounts_for_the_horizontal_pixel_padding():
    # A vertical branch on the right edge of the data (x = 1) is drawn 18 px in
    # from the axes edge: with a 420 px panel it stays out of a box 4 per cent
    # wide, so the top-right corner is free; with no padding it would be crossed.
    segments = pl.DataFrame(
        {
            "x_start": [0.0, 1.0],
            "x_end": [0.0, 1.0],
            "y_start": [-1.0, -1.0],
            "y_end": [-0.9, 1.4],
        }
    )
    rendering = {"legend_box_fraction": [0.04, 0.3], "legend_collision_fraction": 0.0}
    assert tree_legend_orient(segments, (-1.0, 1.4), rendering) == "top-right"
    unpadded = {**rendering, "tree_horizontal_padding": 0}
    assert tree_legend_orient(segments, (-1.0, 1.4), unpadded) == "top-left"


def test_emptiest_corner_is_restricted_to_the_allowed_corners():
    # Dense top-left, bottom-left and top-right; the bottom-right box is empty
    # but not allowed, so the lighter of the two allowed corners wins.
    heavy = _cloud((0.0, 0.4), (0.0, 0.3), 200)
    top_left = _cloud((0.0, 0.4), (0.75, 1.0), 200)
    light = _cloud((0.7, 1.0), (0.75, 1.0), 20)
    xs = np.concatenate([heavy[0], top_left[0], light[0]])
    ys = np.concatenate([heavy[1], top_left[1], light[1]])
    assert emptiest_corner(xs, ys, UNIT, UNIT, {}) == "bottom-right"
    allowed = ("top-right", "bottom-left")
    assert emptiest_corner(xs, ys, UNIT, UNIT, {}, allowed) == "top-right"
    counts = corner_counts(xs, ys, UNIT, UNIT, {}, allowed)
    assert set(counts) == set(allowed) and counts["top-right"] < counts["bottom-left"]


def test_emptiest_corner_ties_go_to_the_first_allowed_corner():
    xs, ys = _cloud((0.45, 0.55), (0.45, 0.55))  # leaves every corner empty
    assert emptiest_corner(xs, ys, UNIT, UNIT, {}, ("bottom-left", "top-right")) == (
        "bottom-left"
    )
    assert emptiest_corner(np.empty(0), np.empty(0), UNIT, UNIT, {}, ("top-left",)) == (
        "top-left"
    )
