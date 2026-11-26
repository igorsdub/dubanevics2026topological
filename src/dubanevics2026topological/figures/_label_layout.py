"""Place text labels clear of one another, with leader lines to what they name.

Pure geometry, in panel pixels (origin top-left, y down); no tree or figure
knowledge. A label has a box and, when it sits away from the branch it names, a
leader line (a straight segment from the branch to the label). ``solve`` places the
labels under two kinds of rule.

The hard rules are never broken while any placement keeps them: a label lies inside
the panel and on no ``fixed`` marker, keeps ``Scene.gap`` from every other label,
and no leader line runs through a label or along another leader. The soft cost
ranks the placements the hard rules leave: how far a label lies from its branch, a
label on another branch (it would read as that branch's count), a leader along a
branch or through a marker, and a leader crossing another leader (a hairline over a
hairline reads fine, so a crossing is a cost, never a breach).

Each label brings its preferred spots (``Item.spots``, each with its own cost) and,
unless it is ``Item.fixed``, may also stand at any lattice position round its branch
on a straight leader. The search is a greedy pass that gives each label, in order,
the cheapest spot clear of those placed before it, followed by min-conflicts repair:
a label still in breach of a hard rule moves to the spot that conflicts least (then
costs least, give or take a little noise), until none is, and last a pass that moves
each label to its cheapest clean spot again. The random choices come from a fixed
seed, so a layout is reproducible. Labels for which no clean placement was found are
counted in the breaches ``solve`` returns, and sit at the least bad spot found.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# (left, top, right, bottom) and (x0, y0, x1, y1), in panel pixels, y down.
Box = tuple[float, float, float, float]
Segment = tuple[float, float, float, float]

# Pixels of air kept between a label and a marker or a leader line.
CLEARANCE = 1.0
# Cost of a label, or a leader along a vertical, on a branch that is not the label's
# own (it reads as that branch's count); a leader through a marker.
OTHER_BRANCH = 30.0
OWN_BRANCH = 25.0
LEADER_OTHER_BRANCH = 3.0
LEADER_CROSSING = 3.0
# The lattice of positions a free label may stand at: every ``LATTICE_STEP`` pixels
# along its text, on the grid of rows (see ``Scene.stack``) across it, up to
# ``LATTICE_REACH`` (horizontal, vertical) from the branch. Each costs
# ``LATTICE_BASE_COST`` for its slanted leader and ``LATTICE_COST_PER_PX`` a pixel of
# the leader's length.
LATTICE_STEP = 4.0
LATTICE_REACH = (520.0, 300.0)
LATTICE_BASE_COST = 1.0
LATTICE_COST_PER_PX = 0.08
# Pixels in from a label's edge where a leader line ends.
LEADER_INSET = 1.5
# Repair: iterations allowed per label, and the random spread of a repair's choice (in
# cost units), which starts at ``REPAIR_NOISE`` and grows ``REPAIR_NOISE_GROWTH`` a
# step up to ``REPAIR_NOISE_MAX`` so that a search stuck in a cycle of the same few
# spots leaves it.
REPAIR_STEPS_PER_LABEL = 30
REPAIR_PATIENCE_PER_LABEL = 4
REPAIR_NOISE = 0.6
REPAIR_NOISE_GROWTH = 0.1
REPAIR_NOISE_MAX = 40.0
# Weight of one hard-rule conflict against any cost, and the cost of one leader
# crossing another.
CONFLICT_WEIGHT = 1000.0
CROSSING_COST = 1.5
# Spots examined at a time, cheapest first: a label usually finds a clean spot in the
# first chunk, and the rest is only searched when it does not (a repair step looks at
# ``REPAIR_CHUNKS`` of them at most).
CHUNK = 192
REPAIR_CHUNKS = 4
# Passes moving every label to its cheapest clean spot once the layout is clean, and
# the attempts (the first in order, the rest in a shuffled one) made while labels stay
# in breach.
IMPROVE_PASSES = 12
ATTEMPTS = 6
# Ejection chains, once the layout is clean: a label that still stands at a costly spot
# (``CHAIN_MIN_COST`` or more: on a branch, say) tries each of its ``CHAIN_SPOTS``
# cheapest spots; one held by at most ``CHAIN_EVICT`` other labels is taken and those
# labels re-placed at their best spots, and the move kept when the layout stays clean
# and the total cost falls. ``CHAIN_ROUNDS`` rounds at most.
CHAIN_MIN_COST = 10.0
CHAIN_SPOTS = 40
CHAIN_EVICT = 3
CHAIN_ROUNDS = 4
# The placeholder leader of a spot with none, far from anything.
_FAR = -1.0e9
_NO_LEADER = (_FAR, _FAR, _FAR, _FAR)


@dataclass(frozen=True, slots=True)
class Spot:
    """A label's box, its leader line (or ``None``) and the cost of being there."""

    box: Box
    leader: Segment | None = None
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class Item:
    """A label to place.

    ``anchor`` is where its leader line starts (the end of the branch it names) and
    ``size`` the label's ``(width, height)``; ``spots`` are the preferred spots, with
    their costs; ``edge`` identifies the label's own branch among the
    ``Scene.branches`` (``None`` when unknown); ``fixed`` keeps the label to its
    preferred spots (no lattice).
    """

    anchor: tuple[float, float]
    size: tuple[float, float]
    spots: Sequence[Spot]
    edge: tuple[int, int] | None = None
    fixed: bool = False


@dataclass(frozen=True, slots=True)
class Scene:
    """What the labels are placed among.

    ``width`` and ``height`` are the panel's; ``fixed`` the boxes labels keep off
    (markers, the colour bar); ``branches`` the boxes of the drawn branch segments
    with their ``branch_edges`` (``(lower end, upper end)`` tree nodes, or ``None``);
    ``placed`` the labels already laid out; ``gap`` the ``(horizontal, vertical)``
    air two label boxes keep between them; ``stack`` the axis along which labels stack
    side by side (``"y"``: rows of horizontal text, ``"x"``: columns of rotated
    text), on which the lattice keeps to a grid fixed to the panel's far edge, so that
    labels pack in tidy rows or columns.
    """

    width: float
    height: float
    fixed: Sequence[Box] = ()
    branches: Sequence[Box] = ()
    branch_edges: Sequence[tuple[int, int] | None] | None = None
    placed: Sequence[Spot] = ()
    gap: tuple[float, float] = (CLEARANCE, CLEARANCE)
    stack: str = "y"


def overlaps(
    a: Box, b: Box, margin: float = CLEARANCE, margin_y: float | None = None
) -> bool:
    """Whether two boxes come closer than ``margin`` (``margin_y`` vertically)."""
    vertical = margin if margin_y is None else margin_y
    return (
        a[0] < b[2] + margin
        and b[0] < a[2] + margin
        and a[1] < b[3] + vertical
        and b[1] < a[3] + vertical
    )


def _array(rows: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def box_matrix(
    a: np.ndarray, b: np.ndarray, margin: float, margin_y: float | None = None
) -> np.ndarray:
    """Return, for each box of ``a`` (rows) and ``b`` (columns), whether they come
    closer than the margins."""
    vertical = margin if margin_y is None else margin_y
    return (
        (a[:, None, 0] < b[None, :, 2] + margin)
        & (b[None, :, 0] < a[:, None, 2] + margin)
        & (a[:, None, 1] < b[None, :, 3] + vertical)
        & (b[None, :, 1] < a[:, None, 3] + vertical)
    )


def _extents(segments: np.ndarray) -> np.ndarray:
    """Return the box round each segment."""
    return np.stack(
        [
            np.minimum(segments[:, 0], segments[:, 2]),
            np.minimum(segments[:, 1], segments[:, 3]),
            np.maximum(segments[:, 0], segments[:, 2]),
            np.maximum(segments[:, 1], segments[:, 3]),
        ],
        axis=1,
    )


def _slab(start, delta, lo, hi, low, high):
    """Clip the parameter range ``[low, high]`` of ``start + t * delta`` to the slab
    ``[lo, hi]`` on one axis; every argument broadcasts elementwise."""
    parallel = np.abs(delta) < 1e-12
    safe = np.where(parallel, 1.0, delta)
    first, second = (lo - start) / safe, (hi - start) / safe
    near = np.where(parallel, -np.inf, np.minimum(first, second))
    far = np.where(parallel, np.inf, np.maximum(first, second))
    missed = parallel & ((start < lo) | (start > hi))
    return np.maximum(low, near), np.where(missed, -np.inf, np.minimum(high, far))


def segment_hits_box(segments: np.ndarray, boxes: np.ndarray, margin: float):
    """Return, elementwise over matching rows of ``segments`` and ``boxes``, whether
    the segment passes within ``margin`` of the box (clipped to the box grown by it)."""
    low = np.zeros(len(segments))
    high = np.ones(len(segments))
    low, high = _slab(
        segments[:, 0],
        segments[:, 2] - segments[:, 0],
        boxes[:, 0] - margin,
        boxes[:, 2] + margin,
        low,
        high,
    )
    low, high = _slab(
        segments[:, 1],
        segments[:, 3] - segments[:, 1],
        boxes[:, 1] - margin,
        boxes[:, 3] + margin,
        low,
        high,
    )
    return low <= high


def _point_distance(px, py, x0, y0, x1, y1):
    """Return the distance of points to segments (elementwise)."""
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    safe = np.where(length < 1e-12, 1.0, length)
    along = np.clip(((px - x0) * dx + (py - y0) * dy) / safe, 0.0, 1.0)
    return np.hypot(px - (x0 + along * dx), py - (y0 + along * dy))


def segments_meet(a: np.ndarray, b: np.ndarray, margin: float):
    """Return, elementwise over matching rows of ``a`` and ``b``, whether the two
    segments properly cross and whether they run within ``margin`` of one another
    without (parallel or collinear hairlines a pixel apart, an end on the other
    line)."""
    ax0, ay0, ax1, ay1 = a.T
    bx0, by0, bx1, by1 = b.T

    def orient(px0, py0, px1, py1, qx, qy):
        return (px1 - px0) * (qy - py0) - (py1 - py0) * (qx - px0)

    crossing = (
        orient(ax0, ay0, ax1, ay1, bx0, by0) * orient(ax0, ay0, ax1, ay1, bx1, by1) < 0
    ) & (
        orient(bx0, by0, bx1, by1, ax0, ay0) * orient(bx0, by0, bx1, by1, ax1, ay1) < 0
    )
    nearest = np.minimum.reduce(
        [
            _point_distance(bx0, by0, ax0, ay0, ax1, ay1),
            _point_distance(bx1, by1, ax0, ay0, ax1, ay1),
            _point_distance(ax0, ay0, bx0, by0, bx1, by1),
            _point_distance(ax1, ay1, bx0, by0, bx1, by1),
        ]
    )
    return crossing, ~crossing & (nearest < margin)


def conflict_matrix(
    boxes_a: np.ndarray,
    leaders_a: np.ndarray,
    boxes_b: np.ndarray,
    leaders_b: np.ndarray,
    gap: tuple[float, float],
    crossings: np.ndarray | None = None,
) -> np.ndarray:
    """Return, for each label of ``a`` (rows) and ``b`` (columns), whether they break a
    hard rule against each other: the boxes come within ``gap``, a leader runs through
    the other label, or two leaders run along each other. A missing leader is the
    placeholder ``_NO_LEADER``, which touches nothing.

    Two leaders that cross are not a breach (a hairline over a hairline reads fine);
    pass a zero ``crossings`` array of the same shape to have them counted in it.

    The exact tests (slab clipping, orientation) run only on the pairs whose extents
    come close, which a few hairlines among many labels keep few.
    """
    result = box_matrix(boxes_a, boxes_b, gap[0], gap[1])
    real_a, real_b = leaders_a[:, 0] > _FAR / 2, leaders_b[:, 0] > _FAR / 2
    extents_a, extents_b = _extents(leaders_a), _extents(leaders_b)
    rows, cols = np.nonzero(box_matrix(extents_a, boxes_b, CLEARANCE))
    keep = real_a[rows]
    rows, cols = rows[keep], cols[keep]
    if rows.size:
        hit = segment_hits_box(leaders_a[rows], boxes_b[cols], CLEARANCE)
        result[rows[hit], cols[hit]] = True
    cols, rows = np.nonzero(box_matrix(extents_b, boxes_a, CLEARANCE))
    keep = real_b[cols]
    rows, cols = rows[keep], cols[keep]
    if rows.size:
        hit = segment_hits_box(leaders_b[cols], boxes_a[rows], CLEARANCE)
        result[rows[hit], cols[hit]] = True
    rows, cols = np.nonzero(box_matrix(extents_a, extents_b, CLEARANCE))
    keep = real_a[rows] & real_b[cols]
    rows, cols = rows[keep], cols[keep]
    if rows.size:
        crossing, along = segments_meet(leaders_a[rows], leaders_b[cols], CLEARANCE)
        result[rows[along], cols[along]] = True
        if crossings is not None:
            crossings[rows[crossing], cols[crossing]] = True
    return result


def leader_ends(
    anchor: tuple[float, float], boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the leader lines from ``anchor`` to ``boxes`` and which boxes allow one.

    Each line runs from the anchor to the nearest edge of its box, ends
    ``LEADER_INSET`` pixels in from the box's corners, and is ordered from its upper
    end to its lower end. A box with the anchor inside it (or against it) allows none.
    """
    ax, ay = anchor
    left, top, right, bottom = boxes.T
    valid = ~(
        (left - CLEARANCE <= ax)
        & (ax <= right + CLEARANCE)
        & (top - CLEARANCE <= ay)
        & (ay <= bottom + CLEARANCE)
    )
    inside_x = (left + LEADER_INSET <= ax) & (ax <= right - LEADER_INSET)
    inside_y = (top + LEADER_INSET <= ay) & (ay <= bottom - LEADER_INSET)
    top_or_bottom = np.where(ay < top, top, bottom)
    left_or_right = np.where(ax < left, left, right)
    # Leave through the edge nearest the anchor: the top or bottom one when the anchor
    # is over or under the box, else the side one.
    end_x = np.where(
        inside_x,
        ax,
        np.where(
            inside_y,
            left_or_right,
            np.clip(ax, left + LEADER_INSET, right - LEADER_INSET),
        ),
    )
    end_y = np.where(inside_y & ~inside_x, ay, top_or_bottom)
    anchor_x, anchor_y = np.full_like(end_x, ax), np.full_like(end_x, ay)
    anchor_first = (ay < end_y) | ((ay == end_y) & (ax <= end_x))
    lines = np.where(
        anchor_first[:, None],
        np.stack([anchor_x, anchor_y, end_x, end_y], axis=1),
        np.stack([end_x, end_y, anchor_x, anchor_y], axis=1),
    )
    return lines, valid


def leader_to(anchor: tuple[float, float], box: Box) -> Segment | None:
    """Return the leader line from ``anchor`` to ``box`` (see ``leader_ends``), or
    ``None`` when the anchor lies inside the box."""
    lines, valid = leader_ends(anchor, _array([box]))
    if not valid[0]:
        return None
    x0, y0, x1, y1 = (float(value) for value in lines[0])
    return x0, y0, x1, y1


def _lattice_boxes(item: Item, scene: Scene) -> np.ndarray:
    """Return the boxes a free label may stand at round its anchor.

    Along the text's reading direction the boxes step ``LATTICE_STEP`` pixels; across
    it they sit on a grid of rows (or columns) as thick as a label plus the gap, fixed
    to the panel's far edge, so that labels packed into them align. Only boxes within
    ``LATTICE_REACH`` of the anchor, and inside the panel, are returned.
    """
    width, height = item.size
    ax, ay = item.anchor
    reach_x, reach_y = LATTICE_REACH
    if scene.stack == "y":
        pitch = height + scene.gap[1]
        tops = scene.height - pitch * np.arange(1, int(scene.height // pitch) + 1)
        tops = tops[np.abs(tops + height / 2 - ay) <= reach_y]
        steps = np.arange(
            -int(reach_x // LATTICE_STEP), int(reach_x // LATTICE_STEP) + 1
        )
        lefts = ax - width / 2 + steps * LATTICE_STEP
        lefts = lefts[(lefts >= 0) & (lefts + width <= scene.width)]
        left, top = np.meshgrid(lefts, tops)
    else:
        pitch = width + scene.gap[0]
        lefts = scene.width - pitch * np.arange(1, int(scene.width // pitch) + 1)
        lefts = lefts[np.abs(lefts + width / 2 - ax) <= reach_x]
        steps = np.arange(
            -int(reach_y // LATTICE_STEP), int(reach_y // LATTICE_STEP) + 1
        )
        tops = ay - height / 2 + steps * LATTICE_STEP
        tops = tops[(tops >= 0) & (tops + height <= scene.height)]
        left, top = np.meshgrid(lefts, tops)
    left, top = left.ravel(), top.ravel()
    return np.stack([left, top, left + width, top + height], axis=1)


def _add_leader_hits(
    cost: np.ndarray,
    leaders: np.ndarray,
    boxes: np.ndarray,
    weights: np.ndarray,
    margin: float,
) -> None:
    """Add ``weights[box]`` to the cost of every leader that passes within ``margin``
    of ``box`` (the placeholder leaders touch nothing)."""
    rows, cols = np.nonzero(box_matrix(_extents(leaders), boxes, margin))
    keep = (leaders[rows, 0] > _FAR / 2) & (weights[cols] > 0)
    rows, cols = rows[keep], cols[keep]
    if rows.size:
        hit = segment_hits_box(leaders[rows], boxes[cols], margin)
        np.add.at(cost, rows[hit], weights[cols[hit]])


class _Pool:
    """The candidate spots of one label that keep the hard rules against the scene,
    cheapest first, as arrays: ``boxes``, ``leaders`` (the placeholder where a spot
    has none) and ``cost``, which includes the branches and markers a spot meets.

    ``clean`` is false when no spot was allowed: the pool then holds the preferred
    spots regardless, and the label is in breach wherever it goes.
    """

    def __init__(self, item: Item, scene: Scene) -> None:
        preferred = len(item.spots)
        boxes = _array([spot.box for spot in item.spots])
        leaders = _array([spot.leader or _NO_LEADER for spot in item.spots])
        cost = np.asarray([spot.cost for spot in item.spots], dtype=float)
        if not item.fixed:
            lattice = _lattice_boxes(item, scene)
            lines, valid = leader_ends(item.anchor, lattice)
            length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
            boxes = np.concatenate([boxes, lattice[valid]])
            leaders = np.concatenate([leaders, lines[valid]])
            cost = np.concatenate(
                [cost, LATTICE_BASE_COST + LATTICE_COST_PER_PX * length[valid]]
            )
        allowed = self._allowed(boxes, leaders, scene)
        self.clean = bool(allowed.any())
        if not self.clean:
            allowed = np.arange(len(boxes)) < preferred
        boxes, leaders, cost = boxes[allowed], leaders[allowed], cost[allowed]
        cost = cost + self._met(item, boxes, leaders, scene)
        if scene.placed:
            others = _array([spot.box for spot in scene.placed])
            theirs = _array([spot.leader or _NO_LEADER for spot in scene.placed])
            crossings = np.zeros((len(boxes), len(others)), dtype=bool)
            conflict_matrix(boxes, leaders, others, theirs, scene.gap, crossings)
            cost = cost + CROSSING_COST * crossings.sum(axis=1)
        order = np.argsort(cost, kind="stable")
        self.boxes, self.leaders, self.cost = boxes[order], leaders[order], cost[order]

    @staticmethod
    def _allowed(boxes: np.ndarray, leaders: np.ndarray, scene: Scene) -> np.ndarray:
        """Return which spots keep the hard rules against the panel's edges, the
        markers and the labels placed before this call."""
        allowed = (
            (boxes[:, 0] >= 0)
            & (boxes[:, 1] >= 0)
            & (boxes[:, 2] <= scene.width)
            & (boxes[:, 3] <= scene.height)
        )
        if scene.fixed:
            allowed &= ~box_matrix(boxes, _array(scene.fixed), CLEARANCE).any(axis=1)
        if scene.placed:
            others = _array([spot.box for spot in scene.placed])
            theirs = _array([spot.leader or _NO_LEADER for spot in scene.placed])
            allowed &= ~conflict_matrix(boxes, leaders, others, theirs, scene.gap).any(
                axis=1
            )
        return allowed

    @staticmethod
    def _met(
        item: Item, boxes: np.ndarray, leaders: np.ndarray, scene: Scene
    ) -> np.ndarray:
        """Return the cost of the branches and markers each spot meets."""
        cost = np.zeros(len(boxes))
        if scene.branches:
            branches = _array(scene.branches)
            # A branch is the label's own only when both edges are known and equal.
            foreign = np.ones(len(branches), dtype=bool)
            if item.edge is not None and scene.branch_edges is not None:
                foreign = np.fromiter(
                    (e is None or e != item.edge for e in scene.branch_edges),
                    dtype=bool,
                    count=len(branches),
                )
            weight = np.where(foreign, OTHER_BRANCH, OWN_BRANCH)
            cost += (box_matrix(boxes, branches, 0.0) * weight).sum(axis=1)
            # A leader along another branch's vertical reads as that branch.
            vertical = (branches[:, 2] - branches[:, 0]) < (
                branches[:, 3] - branches[:, 1]
            )
            along = np.where(vertical, LEADER_OTHER_BRANCH, 1.0) * foreign
            _add_leader_hits(cost, leaders, branches, along, 0.0)
        if scene.fixed:
            markers = _array(scene.fixed)
            _add_leader_hits(
                cost,
                leaders,
                markers,
                np.full(len(markers), LEADER_CROSSING),
                CLEARANCE,
            )
        return cost


def clashing_pairs(
    spots: Sequence[Spot], gap: tuple[float, float]
) -> list[tuple[int, int]]:
    """Return the index pairs of ``spots`` that break a hard rule against each other:
    their boxes come within ``gap``, a leader runs through the other label, or the two
    leaders run along each other. Two leaders that merely cross are a soft cost, not a
    pair."""
    if len(spots) < 2:
        return []
    boxes = _array([spot.box for spot in spots])
    leaders = _array([spot.leader or _NO_LEADER for spot in spots])
    hit = np.triu(conflict_matrix(boxes, leaders, boxes, leaders, gap), 1)
    return [(int(first), int(second)) for first, second in np.argwhere(hit)]


class _Search:
    """One attempt at a layout: the chosen spot of every label, and the moves."""

    def __init__(self, pools: Sequence[_Pool], gap: tuple[float, float], seed: int):
        self.pools = pools
        self.gap = gap
        self.count = len(pools)
        self.boxes = np.full((self.count, 4), _FAR)
        self.leaders = np.full((self.count, 4), _FAR)
        self.pick = np.zeros(self.count, dtype=int)
        self.rng = np.random.default_rng(seed)

    def penalty(self, index: int, start: int, stop: int) -> np.ndarray:
        """Return the penalty of each spot ``start:stop`` of label ``index`` against
        the other labels: ``CONFLICT_WEIGHT`` a hard-rule conflict, ``CROSSING_COST``
        a leader crossing another."""
        pool = self.pools[index]
        spots = pool.boxes[start:stop]
        crossings = np.zeros((len(spots), self.count), dtype=bool)
        hit = conflict_matrix(
            spots,
            pool.leaders[start:stop],
            self.boxes,
            self.leaders,
            self.gap,
            crossings,
        )
        hit[:, index] = False
        crossings[:, index] = False
        return hit.sum(axis=1) * CONFLICT_WEIGHT + crossings.sum(axis=1) * CROSSING_COST

    def best(self, index: int, noise: float = 0.0, chunks: int | None = None) -> int:
        """Return the spot of label ``index`` with the least penalty, then the lowest
        cost (plus up to ``noise``); the first chunk with a clean spot decides, and
        at most ``chunks`` chunks are looked at."""
        pool = self.pools[index]
        winner, winning = 0, np.inf
        for start in range(0, len(pool.cost), CHUNK)[:chunks]:
            stop = start + CHUNK
            score = self.penalty(index, start, stop) + pool.cost[start:stop]
            if noise:
                score = score + self.rng.random(len(score)) * noise
            local = int(np.argmin(score))
            if score[local] < winning:
                winner, winning = start + local, float(score[local])
            if winning < CONFLICT_WEIGHT:
                break
        return winner

    def put(self, index: int, spot: int) -> None:
        self.pick[index] = spot
        self.boxes[index] = self.pools[index].boxes[spot]
        self.leaders[index] = self.pools[index].leaders[spot]

    def clashing(self) -> np.ndarray:
        """Return the labels that break a hard rule against another."""
        hit = conflict_matrix(
            self.boxes, self.leaders, self.boxes, self.leaders, self.gap
        )
        np.fill_diagonal(hit, False)
        return np.flatnonzero(hit.any(axis=1))

    def total(self) -> float:
        """Return the summed cost of the chosen spots."""
        return float(
            sum(self.pools[i].cost[self.pick[i]] for i in range(self.count))
        )

    def chain(self) -> None:
        """Lower the cost of a clean layout by ejection chains (see ``CHAIN_SPOTS``)."""
        for _ in range(CHAIN_ROUNDS):
            improved = False
            costly = [
                i
                for i in range(self.count)
                if self.pools[i].cost[self.pick[i]] >= CHAIN_MIN_COST
            ]
            for index in self.rng.permutation(costly):
                index = int(index)
                pool = self.pools[index]
                held = pool.cost[self.pick[index]]
                if held < CHAIN_MIN_COST:
                    continue
                stop = min(CHAIN_SPOTS, len(pool.cost))
                hit = conflict_matrix(
                    pool.boxes[:stop],
                    pool.leaders[:stop],
                    self.boxes,
                    self.leaders,
                    self.gap,
                )
                hit[:, index] = False
                for spot in np.argsort(hit.sum(axis=1), kind="stable"):
                    spot = int(spot)
                    evicted = np.flatnonzero(hit[spot])
                    if (
                        pool.cost[spot] >= held
                        or len(evicted) > CHAIN_EVICT
                        or spot == self.pick[index]
                    ):
                        continue
                    before = self.total()
                    saved = [
                        (int(i), int(self.pick[i])) for i in [index, *evicted.tolist()]
                    ]
                    self.put(index, spot)
                    for other in evicted:
                        self.put(int(other), self.best(int(other)))
                    if not self.clashing().size and self.total() < before - 1e-9:
                        improved = True
                        break
                    for i, pick in saved:
                        self.put(i, pick)
            if not improved:
                break

    def run(self, order: Sequence[int]) -> set[int]:
        """Place the labels in ``order``, repair, tidy; return the labels in breach."""
        for index in order:
            self.put(index, self.best(index))
        steps = REPAIR_STEPS_PER_LABEL * self.count
        fewest, since = self.count + 1, 0
        for step in range(steps):
            bad = self.clashing()
            if not bad.size:
                break
            # Give up the attempt when the breaches have not fallen for a while.
            fewest, since = (len(bad), 0) if len(bad) < fewest else (fewest, since + 1)
            if since > REPAIR_PATIENCE_PER_LABEL * self.count:
                break
            index = int(bad[self.rng.integers(len(bad))])
            # Late in the search a label may look at every spot it has.
            chunks = REPAIR_CHUNKS if step < steps // 2 else None
            noise = min(REPAIR_NOISE_MAX, REPAIR_NOISE + REPAIR_NOISE_GROWTH * step)
            self.put(index, self.best(index, noise, chunks))
        for _ in range(IMPROVE_PASSES):
            stuck = set(self.clashing().tolist())
            moved = False
            for index in self.rng.permutation(self.count):
                if int(index) in stuck:
                    continue
                spot = self.best(int(index))
                if spot != self.pick[index]:
                    moved = True
                self.put(int(index), spot)
            if not moved:
                break
        if not self.clashing().size:
            self.chain()
        return set(self.clashing().tolist())


def solve(items: Sequence[Item], scene: Scene) -> tuple[list[Spot], int]:
    """Place ``items``; return their spots (in order) and the number in breach.

    A label in breach is one for which no placement keeping every hard rule was found:
    one with no allowed spot at all, or one that still breaks a hard rule against
    another label in the least bad attempt. Each label is counted once, whichever of
    the two it is (or both).
    The first attempt places the labels in order; if it leaves a breach, up to
    ``ATTEMPTS`` more start from a shuffled order, and the least bad is kept.
    """
    pools = [_Pool(item, scene) for item in items]
    unplaced = {index for index, pool in enumerate(pools) if not pool.clean}
    best: _Search | None = None
    fewest: set[int] = set()
    for attempt in range(ATTEMPTS):
        search = _Search(pools, scene.gap, attempt)
        order = list(range(len(pools)))
        if attempt:
            search.rng.shuffle(order)
        breaches = search.run(order)
        if best is None or len(breaches | unplaced) < len(fewest | unplaced):
            best, fewest = search, breaches
        if not breaches:
            break
    assert best is not None or not pools
    spots = (
        []
        if best is None
        else [
            Spot(
                tuple(float(v) for v in best.boxes[index]),  # type: ignore[arg-type]
                None
                if best.leaders[index][0] == _FAR
                else tuple(float(v) for v in best.leaders[index]),  # type: ignore[arg-type]
                float(pools[index].cost[best.pick[index]]),
            )
            for index in range(len(pools))
        ]
    )
    return spots, len(fewest | unplaced)
