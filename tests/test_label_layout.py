"""Hard-rule placement of labels with leader lines (pure geometry)."""

from __future__ import annotations

import numpy as np
import pytest

from dubanevics2026topological.figures import _label_layout as layout
from dubanevics2026topological.figures._label_layout import (
    Item,
    Scene,
    Spot,
    clashing_pairs,
    conflict_matrix,
    leader_to,
    segment_hits_box,
    segments_meet,
    solve,
)

GAP = (5.0, 1.0)
NONE = np.asarray([[layout._FAR] * 4])


def _box(left, top, width=30.0, height=13.0):
    return (left, top, left + width, top + height)


def _hits(segment, box, margin=0.0):
    return bool(
        segment_hits_box(
            np.asarray([segment], dtype=float), np.asarray([box], dtype=float), margin
        )[0]
    )


def _meet(first, second, margin=0.0):
    """Whether the segments cross or run within the margin; also which of the two."""
    crossing, along = segments_meet(
        np.asarray([first], dtype=float), np.asarray([second], dtype=float), margin
    )
    return bool(crossing[0] or along[0])


def test_a_segment_is_tested_against_a_box_grown_by_the_margin():
    box = (10.0, 10.0, 20.0, 20.0)
    assert _hits((0, 15, 30, 15), box)  # straight through
    assert _hits((15, 0, 15, 30), box)  # vertical through
    assert not _hits((15, 0, 15, 5), box)  # stops above it
    assert not _hits((15, 0, 15, 9.5), box)
    assert _hits((15, 0, 15, 9.5), box, margin=1.0)  # within the margin
    assert not _hits((0, 0, 5, 5), box)  # misses the corner
    assert _hits((0, 0, 12, 12), box)  # ends inside


def test_two_segments_conflict_when_they_cross_or_run_a_margin_apart():
    assert _meet((0, 0, 10, 10), (0, 10, 10, 0))  # an X
    assert not _meet((0, 0, 0, 10), (5, 0, 5, 10), margin=1.0)  # parallel, apart
    assert _meet((0, 0, 0, 10), (0.5, 0, 0.5, 10), margin=1.0)  # parallel, a hair
    # A cross is a crossing; hairlines running along each other are not.
    crossing, along = segments_meet(
        np.asarray([(0, 0, 10, 10), (0, 0, 0, 10)], dtype=float),
        np.asarray([(0, 10, 10, 0), (0.5, 0, 0.5, 10)], dtype=float),
        1.0,
    )
    assert crossing.tolist() == [True, False]
    assert along.tolist() == [False, True]
    # Two collinear verticals that only share a column do not touch.
    assert not _meet((0, 0, 0, 10), (0, 20, 0, 30), margin=1.0)
    assert _meet((0, 0, 0, 10), (0, 9, 0, 30), margin=1.0)  # collinear and overlapping


def test_a_missing_leader_touches_nothing_not_even_another_missing_one():
    boxes = np.asarray([_box(0, 0), _box(100, 100)], dtype=float)
    hit = conflict_matrix(
        boxes, np.vstack([NONE, NONE]), boxes, np.vstack([NONE, NONE]), GAP
    )
    assert hit.tolist() == [[True, False], [False, True]]  # a box meets only itself


def test_the_conflicts_are_the_boxes_within_the_gap_and_the_leaders_through_labels():
    boxes = np.asarray([_box(0, 0), _box(34, 0), _box(60, 40)], dtype=float)
    leaders = np.vstack([NONE, NONE, [[15.0, 20.0, 15.0, 60.0]]])
    hit = conflict_matrix(boxes, leaders, boxes, leaders, GAP)
    # 0 and 1 are 4 px apart across, under the 5 px gap; the leader of 2 runs through
    # the column of 0 only below it, clear of it, so it touches nobody.
    assert hit[0, 1] and hit[1, 0]
    assert not hit[2, 0] and not hit[2, 1]
    through = np.vstack([NONE, NONE, [[15.0, -20.0, 15.0, 60.0]]])
    assert conflict_matrix(boxes, through, boxes, through, GAP)[2, 0]  # through box 0
    assert clashing_pairs([Spot(tuple(box)) for box in boxes[:2]], GAP) == [(0, 1)]
    assert clashing_pairs([Spot(_box(0, 0)), Spot(_box(36, 0))], GAP) == []


def test_a_leader_runs_from_the_anchor_to_the_nearest_edge_of_its_box():
    box = (10.0, 10.0, 40.0, 23.0)
    assert leader_to((25.0, 0.0), box) == (25.0, 0.0, 25.0, 10.0)  # down to the top
    assert leader_to((25.0, 40.0), box) == (25.0, 23.0, 25.0, 40.0)  # up to the foot
    assert leader_to((0.0, 16.0), box) == (0.0, 16.0, 10.0, 16.0)  # across to the side
    corner = leader_to((0.0, 0.0), box)
    assert corner[:2] == (0.0, 0.0)  # an end is the anchor, the other on the box
    assert corner[2] == pytest.approx(11.5)
    assert corner[3] == pytest.approx(10.0)
    assert leader_to((25.0, 16.0), box) is None  # the anchor is under the label


def _items(anchors, size=(30.0, 13.0), spots=None, **kwargs):
    return [
        Item(
            anchor,
            size,
            spots(anchor) if spots else [Spot(_box(anchor[0] - 15, anchor[1] - 20))],
            **kwargs,
        )
        for anchor in anchors
    ]


def test_labels_wanting_the_same_spot_are_kept_apart():
    scene = Scene(300.0, 200.0, gap=GAP)
    anchors = [(100.0, 100.0), (104.0, 100.0), (108.0, 100.0)]
    spots, breaches = solve(_items(anchors), scene)
    assert breaches == 0
    assert clashing_pairs(spots, GAP) == []
    assert all(0 <= spot.box[0] and spot.box[2] <= 300 for spot in spots)


def test_a_label_leaves_a_marker_and_keeps_inside_the_panel():
    scene = Scene(300.0, 200.0, fixed=[_box(85, 80, 30, 13)], gap=GAP)
    (spot,), breaches = solve(_items([(100.0, 100.0)]), scene)
    assert breaches == 0
    assert not layout.overlaps(spot.box, _box(85, 80, 30, 13))
    # A label whose preferred spot is off the panel goes where it fits.
    off = Item((10.0, 5.0), (30.0, 13.0), [Spot(_box(-5, -20))])
    (spot,), breaches = solve([off], Scene(300.0, 200.0, gap=GAP))
    assert breaches == 0
    left, top, right, bottom = spot.box
    assert left >= 0 and top >= 0 and right <= 300 and bottom <= 200
    assert spot.leader is not None  # it is off its branch, so it has a leader
    assert leader_to((10.0, 5.0), spot.box) == spot.leader


def test_a_crowded_row_spreads_on_slanted_leaders_without_a_collision():
    # Twenty anchors 7 px apart, each wanting the one spot over it: there is no room
    # for them in a row, so they stack and fan out.
    anchors = [(40.0 + 7.0 * k, 150.0) for k in range(20)]

    def over(anchor):
        return [Spot(_box(anchor[0] - 15, anchor[1] - 20))]

    scene = Scene(300.0, 200.0, gap=GAP)
    spots, breaches = solve(_items(anchors, spots=over), scene)
    assert breaches == 0
    assert clashing_pairs(spots, GAP) == []
    assert any(
        spot.leader is not None and spot.leader[0] != spot.leader[2] for spot in spots
    )  # some leaders are slanted


def test_a_branch_of_another_edge_costs_more_than_one_of_the_labels_own():
    own, other = (1, 2), (3, 4)
    branch = _box(95, 80, 10, 13)  # the branch's box lies under the label's spot
    scene = Scene(
        300.0,
        200.0,
        branches=[branch],
        branch_edges=[own],
        gap=GAP,
    )
    spot = [Spot(_box(85, 80, 30, 13))]
    item = Item((100.0, 100.0), (30.0, 13.0), spot, edge=own, fixed=True)
    (kept,), _ = solve([item], scene)
    assert kept.cost == pytest.approx(layout.OWN_BRANCH)  # its own branch
    item = Item((100.0, 100.0), (30.0, 13.0), spot, edge=other, fixed=True)
    (kept,), _ = solve([item], scene)
    assert kept.cost == pytest.approx(layout.OTHER_BRANCH)  # another's


def test_a_label_is_counted_in_breach_when_nothing_keeps_the_rules():
    # A panel as narrow as the label: two labels cannot both be clear of each other.
    scene = Scene(30.0, 13.0, gap=GAP)
    items = _items([(15.0, 6.0), (15.0, 7.0)], spots=lambda a: [Spot(_box(0, 0))])
    spots, breaches = solve(items, scene)
    assert breaches >= 1
    assert len(spots) == 2


def test_a_label_without_an_allowed_spot_that_also_clashes_is_counted_once():
    # Both labels have only one spot, and it lies on a marker, so neither has an
    # allowed spot; the two spots also overlap each other, so both clash in the
    # search too. Each label is in breach once, not once per way of being in breach.
    marker = _box(0, 0)
    scene = Scene(300.0, 200.0, fixed=[marker], gap=GAP)
    items = [
        Item((15.0, 6.0), (30.0, 13.0), [Spot(_box(0, 0))], fixed=True),
        Item((15.0, 7.0), (30.0, 13.0), [Spot(_box(1, 0))], fixed=True),
    ]
    spots, breaches = solve(items, scene)
    assert len(spots) == 2
    assert breaches == 2


def test_two_leaders_that_only_cross_are_not_a_breach():
    # Hairlines crossing are a soft cost (CROSSING_COST), not a hard-rule pair.
    first = Spot(_box(0, 0), (15.0, 100.0, 15.0, 13.0))
    second = Spot(_box(100, 0), (5.0, 60.0, 130.0, 13.0))
    assert _meet(first.leader, second.leader)
    assert clashing_pairs([first, second], GAP) == []


def test_the_layout_is_reproducible():
    anchors = [(40.0 + 7.0 * k, 150.0) for k in range(12)]
    scene = Scene(300.0, 200.0, gap=GAP)
    first, _ = solve(_items(anchors), scene)
    second, _ = solve(_items(anchors), scene)
    assert first == second


def test_labels_placed_before_the_call_are_kept_clear_of():
    placed = Spot(_box(85, 80), (100.0, 93.0, 100.0, 140.0))
    scene = Scene(300.0, 200.0, placed=[placed], gap=GAP)
    # The anchor's own spot is the placed label's neighbour, inside its gap.
    (spot,), breaches = solve(_items([(130.0, 100.0)]), scene)
    assert breaches == 0
    assert clashing_pairs([placed, spot], GAP) == []
    assert not layout.overlaps(spot.box, placed.box, *GAP)


def test_a_dense_comb_of_branches_leaves_no_label_on_a_branch():
    # Twelve tips ten pixels apart, each with a branch standing 60 px above it. A
    # 22 px label centred over its tip lies across the neighbouring branches, so every
    # preferred spot is on a branch; clear spots exist only past the branches' tops
    # or below the tips, on a leader. The solver must find them rather than accept a
    # label printed over a branch.
    tips = [(20.0 + 10.0 * k, 100.0) for k in range(12)]
    branches = [(x - 0.6, 40.0, x + 0.6, 100.0) for x, _ in tips]
    edges = [(k, k + 100) for k in range(12)]
    items = [
        Item(
            (x, y),
            (22.0, 13.0),
            [Spot((x - 11.0, y - 20.0, x + 11.0, y - 7.0))],
            edge=edges[k],
        )
        for k, (x, y) in enumerate(tips)
    ]
    scene = Scene(
        200.0,
        200.0,
        branches=branches,
        branch_edges=edges,
        gap=GAP,
    )
    spots, breaches = solve(items, scene)
    assert breaches == 0
    assert clashing_pairs(spots, GAP) == []
    for spot in spots:
        assert not any(layout.overlaps(spot.box, branch, 0.0) for branch in branches), (
            spot.box
        )
