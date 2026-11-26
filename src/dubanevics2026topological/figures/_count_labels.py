"""Count labels on a count-coloured tree: at tips, on every branch, beside ridges.

A tree coloured by branch vertex count shows a branch's count only as a colour; the
labels write it out. The ``branch_count_label_mode`` of ``config/figures.toml``
picks the branches: ``"tips"`` labels the terminal branches of at most
``branch_count_label_max`` sequences, above their tips, and ``"all"`` labels every
branch of the tree, the terminal ones above their tips and the others below the
step of the branch (the horizontal run that joins it to its lower neighbour) or,
for a long vertical branch, beside it. The cropped ridge panels of the
codon-entropy figure write the KEM and TCL ridge counts beside the ridges, every
ridge branch whatever its count, and take no tree labels of their own.

The text is the palette's peak grey, two points smaller than the figure's body
font, and carries no amino-acid name. Labels are laid out in panel pixels, by
``_label_layout.solve``, under hard rules that no placement breaks while any spot in
the panel keeps them: a label sits on no stop-codon cross, wild-type star and label
or colour bar, comes within a space (``LABEL_GAP_EM``) of no other label, so that the
digits of neighbouring labels never read as one number, and has no leader line
through it. Among the spots the rules leave, a label prefers the one that covers
the fewest branches, a branch of another edge counting more than a branch of its
own (it would read as that branch's count), and lies nearest its own branch.

A tip label goes centred above its tip; when that spot is taken it is raised one label
height, slid sideways along its leader, or moved beside the tip, and only then raised
further. A step label goes centred below its step, and a taken spot is answered the
same way, by going down a label height at a time. A label moved off its branch carries
a thin peak-grey leader line to the branch, so a reader can tell which branch the
count names (the vertical nudge, ``branch_count_label_collision = "nudge"``). In a
dense tree the preferred spots can all be taken, and the label then stands anywhere
round its branch, on a slanted leader. With ``"rotate"`` every label is turned a
quarter turn (reading upward), which takes the width of a digit's height instead of the
text's length, so labels on tips a digit apart collide less; the search is the same.
A ridge label goes beside its ridge: level with the ridge's middle, else further up
or down the ridge, else above its top. The layers are meant to go under the
stop-codon and wild-type marks, which stay on top.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import altair as alt
import polars as pl

from ._branch_count_style import (
    branch_count_label_collision,
    branch_count_label_max,
    branch_count_label_mode,
)
from ._charts import marker_config
from ._label_layout import (
    Box,
    Item,
    Scene,
    Spot,
    clashing_pairs,
    solve,
)
from .candidate_tree_biology import _tree_axes_x_domain, estimate_text_width
from .palette import PEAK_GREY

LABEL_COLOR = PEAK_GREY
# The label's font is the figure's body font less this many points.
FONT_REDUCTION = 2
# Pixels between a tip's stop-codon cross and the foot of the label above it.
TIP_CLEARANCE = 3.0
# Pixels between a ridge and a label set beside it.
SIDE_CLEARANCE = 4.0
# Pixels of air kept between a label and anything else it must not touch, and between
# two stacked horizontal labels.
LABEL_MARGIN = 1.0
# Air kept between two labels along their reading direction, as a fraction of the
# font size: about the width of a space, so that single digits on neighbouring tips
# never read as one number (3 and 4 as 34). A rotated label's line gap is the second.
LABEL_GAP_EM = 0.4
ROTATED_GAP_EM = 0.25
# Pixels a raised or lowered label's leader must meet its edge inside: a label slides
# sideways along its leader (see ``_shifts``) only while the leader still lands on it.
LEADER_CONTACT = 3.0
# How far along the leader a label slides, as fractions of the sliding it has room
# for, each to either side.
SHIFT_PARTS = (0.5, 1.0)
# Label heights a label may be raised above its tip (each raised label carries a
# leader line down to the tip): the tips of a dense tree are a few pixels apart, so
# neighbours need rows of their own.
MAX_RAISES = 10
# Label heights a step label may be lowered below its step (each lowered label
# carries a leader line up to the step): the steps of a tree's saddle staircase are
# only a label's width apart, so several rows can be needed.
MAX_STEP_ROWS = 10
# Label heights a step label may be raised above its branch's top once the rows under
# the step are all taken.
MAX_STEP_RAISES = 4
# A rotated label is this fraction of the font size across (a digit is about this
# tall: Arial's digits stand 0.716 em), and as long as its text is wide.
ROTATED_THICKNESS = 0.72
# Vega text angle of a rotated label: a quarter turn counter-clockwise, so the
# digits read upward.
ROTATED_ANGLE = 270
# Raises tried before the sides of the tip: a label one height up is still close to
# its tip, one further is better placed beside it when there is room.
NEAR_RAISES = 1
# Fractions of a ridge's length, from its top, where a side label is tried: the
# middle first, then ever nearer the ends of the ridge.
SIDE_FRACTIONS = (0.5, 0.35, 0.65, 0.2, 0.8, 0.05, 0.95)
# Pixels of air kept around the wild-type label on top of ``LABEL_MARGIN``: the
# text overruns its estimated width and a count label read beside it as one word.
WILD_TYPE_CLEARANCE = 4.0
# Stroke width in pixels of a leader line, and the gap left between the tip's
# stop-codon cross and the foot of the leader.
LEADER_WIDTH = 0.8
LEADER_GAP = 1.0
# Pixel width of the box a branch segment occupies as an obstacle. Labels keep no
# extra air from branches (a digit's advance width already carries its side
# bearings), which lets two-digit labels sit between tips a few pixels apart.
SEGMENT_THICKNESS = 1.2
# Pixels added around the colour bar's nominal box (its text runs a little past it).
COLORBAR_PAD = 4.0
# Cost of a spot per label height lifted or lowered away from the branch, of sliding
# along the leader (up to this for the whole reach), and of a spot beside the tip.
ROW_COST = 0.7
SHIFT_COST = 0.6
SIDE_COST = 1.0
# Cost added to a step label's spots above its branch: they sit among the tips'
# stems, so the band under the staircase is kept for the labels it holds clear.
ABOVE_PENALTY = 3.0

# ``leader_x``, ``leader_y0``, ``leader_x1`` and ``leader_y1`` are null unless the label
# is moved off its branch: the leader line runs from ``(leader_x, leader_y0)`` (its upper
# end) to ``(leader_x1, leader_y1)`` (its lower end): from under a raised label to just
# above its tip, or from just below a step down to a lowered label. It is vertical
# (``leader_x == leader_x1``) unless the label stands off its branch's column.
_LABEL_SCHEMA = {
    "dna_sequence": pl.String,
    "vertex_count": pl.Int64,
    "label": pl.String,
    "px": pl.Float64,
    "py": pl.Float64,
    "leader_x": pl.Float64,
    "leader_y0": pl.Float64,
    "leader_x1": pl.Float64,
    "leader_y1": pl.Float64,
}


def count_label_font_size(rendering: dict[str, Any]) -> int:
    """Return the label font size: the figure's body font minus two points.

    The ``branch_count_label_font_reduction`` key of ``config/figures.toml`` sets the
    points taken off (``FONT_REDUCTION`` when absent).
    """
    reduction = int(rendering.get("branch_count_label_font_reduction", FONT_REDUCTION))
    return int(rendering.get("font_size", 15)) - reduction


def count_label_text(count: int) -> str:
    """Return a branch vertex count as label text, thousands separated."""
    return f"{int(count):,}"


def fixed_label_text(value: float, digits: int) -> str:
    """Return a threshold as label text with ``digits`` decimals."""
    return f"{float(value):.{digits}f}"


def general_label_text(value: float) -> str:
    """Return a value as label text with six significant digits."""
    return f"{float(value):.6g}"


def label_rotated(rendering: dict[str, Any]) -> bool:
    """Return whether the labels are turned a quarter turn (the ``rotate`` variant)."""
    return branch_count_label_collision(rendering) == "rotate"


def label_extent(text: str, rendering: dict[str, Any]) -> tuple[float, float]:
    """Return the pixel ``(width, height)`` of a label's box.

    The text's estimated width by the font size; a rotated label swaps them, taking
    ``ROTATED_THICKNESS`` of the font size for its width (its digits' height).
    """
    font = float(count_label_font_size(rendering))
    length = estimate_text_width(text, font)
    if label_rotated(rendering):
        return font * ROTATED_THICKNESS, length
    return length, font


@dataclass(frozen=True, slots=True)
class PanelGeometry:
    """A tree panel's data domains and pixel size, to place data in the panel.

    ``x_domain`` and ``y_domain`` are the axes' data extents; ``width`` and
    ``height`` the panel's pixels. Pixels run from the top-left, y down.
    """

    x_domain: tuple[float, float]
    y_domain: tuple[float, float]
    width: float
    height: float

    def to_pixels(self, x: float, y: float) -> tuple[float, float]:
        """Return the panel pixel of a data point."""
        (x0, x1), (y0, y1) = self.x_domain, self.y_domain
        return (
            (x - x0) / (x1 - x0) * self.width,
            (y1 - y) / (y1 - y0) * self.height,
        )


def tree_geometry(
    plot_data: Any,
    rendering: dict[str, Any],
    *,
    size: tuple[float, float],
    y_domain: Sequence[float],
) -> PanelGeometry:
    """Return the geometry of a tree panel drawn by Topographer's ``plot_tree``.

    The x domain is the tree scale's (``tree_x_domain``, Topographer's horizontal
    padding included); ``y_domain`` is the fitness axis the figure sets.
    """
    return PanelGeometry(
        _tree_axes_x_domain(plot_data.edge_segment_table, rendering, size[0]),
        (float(y_domain[0]), float(y_domain[1])),
        float(size[0]),
        float(size[1]),
    )


def terminal_tips(plot_data: Any) -> pl.DataFrame:
    """Return the tip of every terminal branch with the branch's vertex count.

    A terminal branch is an edge whose upper end (its target) is the lower end of no
    other edge. ``plot_data`` carries ``vertex_count`` on its edge table (as
    ``plot_data_with_metrics`` leaves it). Columns: ``dna_sequence``, ``x``, ``y``,
    ``vertex_count``, ``source_tree_node_index`` and ``target_tree_node_index`` (the
    branch's edge, lower end then upper end), ordered left to right.
    """
    edges = plot_data.edge_table
    schema = {
        "dna_sequence": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "vertex_count": pl.Int64,
        "source_tree_node_index": pl.Int64,
        "target_tree_node_index": pl.Int64,
    }
    if edges.is_empty():
        return pl.DataFrame(schema=schema)
    lower_ends = edges["source_tree_node_index"].unique().to_list()
    tips = edges.filter(~pl.col("target_tree_node_index").is_in(lower_ends)).select(
        pl.col("target_tree_node_index").alias("tree_node_index"),
        "vertex_count",
        "source_tree_node_index",
        "target_tree_node_index",
    )
    nodes = plot_data.node_table.select("tree_node_index", "dna_sequence", "x", "y")
    return (
        tips.join(nodes, on="tree_node_index", how="left", validate="1:1")
        .select(*schema)
        .cast(schema)
        .sort("x", "y")
    )


def internal_branches(plot_data: Any) -> pl.DataFrame:
    """Return every non-terminal branch with the path Topographer draws for it.

    An internal branch is an edge whose upper end (its target) is the lower end of
    another edge. It is drawn as a vertical segment at the upper end's ``x`` from
    the upper end's fitness down to the lower end's, then a horizontal run at the
    lower end's fitness to the lower end's ``x``. In a tree of close saddles the
    vertical is a pixel or two and the branch is its horizontal step.
    Columns: ``dna_sequence`` (the branch's upper end), ``vertex_count``, ``x`` (the
    vertical segment), ``y_top`` and ``y_bottom`` (its ends, in data coordinates) and
    ``x_end`` (where the horizontal run ends), then ``source_tree_node_index`` and
    ``target_tree_node_index`` (the branch's edge, lower end then upper end), ordered
    left to right by the middle of the horizontal run.
    """
    schema = {
        "dna_sequence": pl.String,
        "vertex_count": pl.Int64,
        "x": pl.Float64,
        "y_top": pl.Float64,
        "y_bottom": pl.Float64,
        "x_end": pl.Float64,
        "source_tree_node_index": pl.Int64,
        "target_tree_node_index": pl.Int64,
    }
    edges = plot_data.edge_table
    if edges.is_empty():
        return pl.DataFrame(schema=schema)
    lower_ends = edges["source_tree_node_index"].unique().to_list()
    nodes = plot_data.node_table.select("tree_node_index", "dna_sequence", "x", "y")
    upper = nodes.rename({"tree_node_index": "target_tree_node_index", "y": "y_top"})
    lower = nodes.select(
        pl.col("tree_node_index").alias("source_tree_node_index"),
        pl.col("x").alias("x_end"),
        pl.col("y").alias("y_bottom"),
    )
    return (
        edges.filter(pl.col("target_tree_node_index").is_in(lower_ends))
        .select("source_tree_node_index", "target_tree_node_index", "vertex_count")
        .join(upper, on="target_tree_node_index", how="left", validate="m:1")
        .join(lower, on="source_tree_node_index", how="left", validate="m:1")
        .select(*schema)
        .cast(schema)
        .sort(pl.col("x") + pl.col("x_end"), "y_top")
    )


def stop_marker_half(rendering: dict[str, Any]) -> float:
    """Return half the pixel extent of a tree's stop-codon cross."""
    stop = marker_config(rendering, "stop_codon", default_shape="x")
    return math.sqrt(int(rendering.get("stop_codon_tree_size", stop.size))) / 2


def segment_boxes(segments: pl.DataFrame, geometry: PanelGeometry) -> list[Box]:
    """Return a thin pixel box for every drawn branch segment.

    ``segments`` is a tree's ``edge_segment_table`` (``x_start``, ``x_end``,
    ``y_start``, ``y_end`` in data coordinates). A label that keeps clear of these
    boxes sits on no branch.
    """
    half = SEGMENT_THICKNESS / 2
    boxes: list[Box] = []
    for row in segments.select("x_start", "x_end", "y_start", "y_end").iter_rows(
        named=True
    ):
        x0, y0 = geometry.to_pixels(float(row["x_start"]), float(row["y_start"]))
        x1, y1 = geometry.to_pixels(float(row["x_end"]), float(row["y_end"]))
        boxes.append(
            (
                min(x0, x1) - half,
                min(y0, y1) - half,
                max(x0, x1) + half,
                max(y0, y1) + half,
            )
        )
    return boxes


def segment_edges(segments: pl.DataFrame) -> list[tuple[int, int]]:
    """Return the ``(lower end, upper end)`` tree nodes of the edge of each segment,
    in the order of ``segment_boxes``."""
    return [
        (int(row["source_tree_node_index"]), int(row["target_tree_node_index"]))
        for row in segments.select(
            "source_tree_node_index", "target_tree_node_index"
        ).iter_rows(named=True)
    ]


def marker_boxes(
    rendering: dict[str, Any],
    geometry: PanelGeometry,
    *,
    stop_points: pl.DataFrame | None = None,
    wild_type_source: Any = None,
    colorbar_box: Box | None = None,
) -> list[Box]:
    """Return the pixel boxes of everything a label must keep clear of.

    The stop-codon crosses at ``stop_points`` (columns ``x`` and ``y``), the
    wild-type star and its label (``wild_type_source.wild_type_marker_table``) and
    the count colour bar's box.
    """
    boxes: list[Box] = []
    if colorbar_box is not None:
        left, top, right, bottom = colorbar_box
        boxes.append(
            (
                left - COLORBAR_PAD,
                top - COLORBAR_PAD,
                right + COLORBAR_PAD,
                bottom + COLORBAR_PAD,
            )
        )
    half = stop_marker_half(rendering)
    if stop_points is not None:
        for row in stop_points.select("x", "y").iter_rows(named=True):
            px, py = geometry.to_pixels(float(row["x"]), float(row["y"]))
            boxes.append((px - half, py - half, px + half, py + half))
    table = getattr(wild_type_source, "wild_type_marker_table", None)
    if table is not None and not table.is_empty():
        marker = marker_config(rendering, "wild_type", default_shape="star")
        star = math.sqrt(marker.size) / 2
        font = int(rendering.get("font_size", 15))
        for row in table.iter_rows(named=True):
            px, py = geometry.to_pixels(float(row["x"]), float(row["y"]))
            boxes.append((px - star, py - star, px + star, py + star))
            if "marker_label" in table.columns:
                width = estimate_text_width(str(row["marker_label"]), font)
                right = px + marker.label_dx
                boxes.append(
                    (
                        right - width - WILD_TYPE_CLEARANCE,
                        py - font / 2 - WILD_TYPE_CLEARANCE,
                        right + WILD_TYPE_CLEARANCE,
                        py + font / 2 + WILD_TYPE_CLEARANCE,
                    )
                )
    return boxes


def label_gap(rendering: dict[str, Any]) -> tuple[float, float]:
    """Return the ``(horizontal, vertical)`` pixels of air two label boxes keep.

    Along their reading direction labels keep ``LABEL_GAP_EM`` of the font (about a
    space, so that the digits of neighbouring labels never read as one number: 3 and
    4 as 34); across it a horizontal label keeps ``LABEL_MARGIN`` and a rotated one
    ``ROTATED_GAP_EM`` of the font (the gap between two lines of text).
    """
    font = float(count_label_font_size(rendering))
    if label_rotated(rendering):
        return ROTATED_GAP_EM * font, LABEL_GAP_EM * font
    return LABEL_GAP_EM * font, LABEL_MARGIN


def _label_rows(entries: Iterable[tuple[dict[str, Any], str, Spot]]) -> pl.DataFrame:
    rows = []
    for item, label, spot in entries:
        box = spot.box
        row = {
            "dna_sequence": item["dna_sequence"],
            "vertex_count": int(item["vertex_count"]),
            "label": label,
            "px": (box[0] + box[2]) / 2,
            "py": (box[1] + box[3]) / 2,
            "leader_x": None,
            "leader_y0": None,
            "leader_x1": None,
            "leader_y1": None,
        }
        if spot.leader is not None:
            x0, y0, x1, y1 = spot.leader
            row.update(leader_x=x0, leader_y0=y0, leader_x1=x1, leader_y1=y1)
        rows.append(row)
    return pl.DataFrame(rows, schema=_LABEL_SCHEMA)


def _item_edge(item: dict[str, Any]) -> tuple[int, int] | None:
    """Return the ``(lower end, upper end)`` tree nodes of the edge a row names."""
    source = item.get("source_tree_node_index")
    target = item.get("target_tree_node_index")
    if source is None or target is None:
        return None
    return int(source), int(target)


def _spot_cost(rows: int, shift: float, reach: float) -> float:
    """Return how far a spot lies from its branch: ``ROW_COST`` a label height lifted
    or lowered and up to ``SHIFT_COST`` for sliding along the leader."""
    return rows * ROW_COST + (abs(shift) / reach * SHIFT_COST if reach else 0.0)


def _shifts(width: float) -> list[float]:
    """Return the sideways slides of a raised or lowered label, the unshifted first.

    A label slides at most to the point where its leader (which stays over the
    branch) still meets its edge ``LEADER_CONTACT`` pixels in.
    """
    reach = width / 2 - LEADER_CONTACT
    if reach < 1:
        return [0.0]
    return [0.0, *(sign * reach * part for part in SHIFT_PARTS for sign in (1, -1))]


def _above_tip(
    px: float,
    py: float,
    width: float,
    height: float,
    half: float,
    raised: int,
    shift: float = 0.0,
    gap: float = LABEL_MARGIN,
) -> Spot:
    """Return the spot of a label ``raised`` label heights above the tip at (px, py).

    ``half`` is a stop-codon cross's half-size; ``width`` and ``height`` the label's
    box and ``gap`` the air between two stacked labels. A raised label also gets its
    leader line, which stands on the tip above the cross and reaches up to the
    label's foot; the unraised label sits right above the tip and needs none. A
    ``shift`` slides the box sideways, the leader staying on the tip's column.
    """
    bottom = py - half - TIP_CLEARANCE - raised * (height + gap)
    left = px - width / 2 + shift
    box: Box = (left, bottom - height, left + width, bottom)
    leader = (px, bottom, px, py - half - LEADER_GAP) if raised else None
    return Spot(box, leader)


def _labelled_tips(tips: pl.DataFrame, rendering: dict[str, Any]) -> pl.DataFrame:
    """Return the tips that carry a label: all of them in the ``"all"`` mode, else
    those whose count is at most ``branch_count_label_max``."""
    if branch_count_label_mode(rendering) == "all":
        return tips
    return tips.filter(pl.col("vertex_count") <= branch_count_label_max(rendering))


def _placed_spots(labels: pl.DataFrame | None, rendering: dict[str, Any]) -> list[Spot]:
    """Return the spots (label box and leader line) of placed ``labels``."""
    if labels is None:
        return []
    spots: list[Spot] = []
    for row in labels.iter_rows(named=True):
        width, height = label_extent(row["label"], rendering)
        box: Box = (
            row["px"] - width / 2,
            row["py"] - height / 2,
            row["px"] + width / 2,
            row["py"] + height / 2,
        )
        leader = None
        if row["leader_x"] is not None:
            leader = (
                row["leader_x"],
                row["leader_y0"],
                row["leader_x1"],
                row["leader_y1"],
            )
        spots.append(Spot(box, leader))
    return spots


def _scene(
    rendering: dict[str, Any],
    geometry: PanelGeometry,
    obstacles: Iterable[Box],
    lines: Iterable[Box],
    line_edges: Sequence[tuple[int, int] | None] | None,
    placed: pl.DataFrame | None,
) -> Scene:
    return Scene(
        geometry.width,
        geometry.height,
        fixed=list(obstacles),
        branches=list(lines),
        branch_edges=line_edges,
        placed=_placed_spots(placed, rendering),
        gap=label_gap(rendering),
        stack="x" if label_rotated(rendering) else "y",
    )


def _solve(items: list[Item], scene: Scene) -> list[Spot]:
    spots, _ = solve(items, scene)
    return spots


def _tip_spots(
    px: float, py: float, width: float, height: float, half: float, across: float
) -> list[Spot]:
    """Return a tip's preferred spots, best first.

    Right above the tip, then ``NEAR_RAISES`` label heights up (sliding sideways
    along the leader), beside the tip, and last further up to ``MAX_RAISES``.
    """
    shifts = _shifts(width)
    reach = width / 2 - LEADER_CONTACT
    spots: list[Spot] = []

    def above(raised: int, shift: float) -> None:
        spot = _above_tip(px, py, width, height, half, raised, shift, across)
        spots.append(replace(spot, cost=_spot_cost(raised, shift, reach)))

    above(0, 0.0)
    for raised in range(1, NEAR_RAISES + 1):
        for shift in shifts:
            above(raised, shift)
    for top in (max(0.0, py - height / 2), max(0.0, py)):
        for sign in (1, -1):
            left = px + SIDE_CLEARANCE if sign > 0 else px - SIDE_CLEARANCE - width
            spots.append(Spot((left, top, left + width, top + height), None, SIDE_COST))
    for raised in range(NEAR_RAISES + 1, MAX_RAISES + 1):
        for shift in shifts:
            above(raised, shift)
    return spots


def place_tip_labels(
    tips: pl.DataFrame,
    rendering: dict[str, Any],
    geometry: PanelGeometry,
    obstacles: Iterable[Box] = (),
    lines: Iterable[Box] = (),
    *,
    line_edges: Sequence[tuple[int, int] | None] | None = None,
    placed: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Lay out the count labels of the labelled tips, above them.

    The labelled tips are those of at most ``branch_count_label_max`` sequences, or
    every tip when the ``branch_count_label_mode`` is ``"all"``.

    Each label goes centred above its tip, its foot a cross's half-size and
    ``TIP_CLEARANCE`` above the tip so it clears a stop-codon cross drawn on the tip,
    unless that is taken. Then it goes up a label height (``NEAR_RAISES``), slides
    sideways along the leader that joins it to its tip (``_shifts``), goes beside
    the tip, level with it and just below it (as near the panel top as fits), and
    only then rises further (up to ``MAX_RAISES``). Past those preferred spots, and
    whenever they are all poor, a label may stand anywhere round its tip on a
    slanted leader (``_label_layout.solve``).

    A label never overlaps a marker (``obstacles``: a stop-codon cross, the wild-type
    star, the colour bar), comes within ``label_gap`` of another label, or has a
    leader line through it, and its own leader never crosses another's: while any
    spot in the panel keeps these rules, the label takes one. The spots they leave are
    ranked by how many branches (``lines``, the boxes of ``segment_boxes``;
    ``line_edges`` their edges, to tell a tip's own branch from another's) they
    cover and how far they lie from the tip. ``placed`` is a table of labels laid out
    earlier, which these keep clear of in the same way.

    Columns: ``dna_sequence``, ``vertex_count``, ``label``, the label's centre ``px``
    and ``py`` in panel pixels (origin top-left, y down) and the leader line from
    ``leader_x``, ``leader_y0`` (its upper end, under the label) to ``leader_x1``,
    ``leader_y1`` (its lower end, above the tip), null when the label carries none. A
    rotated label (``branch_count_label_collision = "rotate"``) is a digit's height
    wide and its text's length tall; the search is the same.
    """
    rows, texts, items = _tip_items(tips, rendering, geometry)
    scene = _scene(rendering, geometry, obstacles, lines, line_edges, placed)
    return _label_rows(zip(rows, texts, _solve(items, scene), strict=True))


def _tip_items(
    tips: pl.DataFrame, rendering: dict[str, Any], geometry: PanelGeometry
) -> tuple[list[dict[str, Any]], list[str], list[Item]]:
    """Return the labelled tips' rows, their label texts and the items to place."""
    half = stop_marker_half(rendering)
    across = label_gap(rendering)[1]
    rows = list(_labelled_tips(tips, rendering).iter_rows(named=True))
    texts: list[str] = []
    items: list[Item] = []
    for tip in rows:
        label = count_label_text(tip["vertex_count"])
        texts.append(label)
        width, height = label_extent(label, rendering)
        px, py = geometry.to_pixels(float(tip["x"]), float(tip["y"]))
        items.append(
            Item(
                (px, py - half - LEADER_GAP),
                (width, height),
                _tip_spots(px, py, width, height, half, across),
                _item_edge(tip),
            )
        )
    return rows, texts, items


def _side_candidates(
    px: float, top: float, bottom: float, width: float, height: float
) -> list[Box]:
    """Return the boxes beside a vertical segment, middle first, then up and down.

    The segment stands at pixel column ``px`` from ``top`` to ``bottom``; a box sits
    ``SIDE_CLEARANCE`` from it, the right of each pair before the left
    (``SIDE_FRACTIONS``).
    """
    candidates: list[Box] = []
    for fraction in SIDE_FRACTIONS:
        centre = top + (bottom - top) * fraction
        candidates += [
            (
                px + SIDE_CLEARANCE,
                centre - height / 2,
                px + SIDE_CLEARANCE + width,
                centre + height / 2,
            ),
            (
                px - SIDE_CLEARANCE - width,
                centre - height / 2,
                px - SIDE_CLEARANCE,
                centre + height / 2,
            ),
        ]
    return candidates


def place_side_labels(
    branches: pl.DataFrame,
    rendering: dict[str, Any],
    geometry: PanelGeometry,
    obstacles: Iterable[Box] = (),
    lines: Iterable[Box] = (),
) -> pl.DataFrame:
    """Lay out one count label beside each branch, level with its middle.

    ``branches`` has columns ``dna_sequence`` (the branch's tip), ``vertex_count``,
    ``x`` (the branch's vertical segment) and ``y_top`` and ``y_bottom`` (its ends,
    in data coordinates). A label sits ``SIDE_CLEARANCE`` pixels to the right or
    left of the segment: the spots are the middle of the ridge, then further up and
    down it (``SIDE_FRACTIONS``, the right before the left), then above its top. The
    first spot that keeps the hard rules of ``place_tip_labels`` (no overlap with a
    marker in ``obstacles`` or another label) and the fewest branches (``lines``,
    the boxes of ``segment_boxes``) wins. Every spot touches the ridge it names, so
    none needs a leader line. Columns as ``place_tip_labels``.
    """
    foot = stop_marker_half(rendering) + TIP_CLEARANCE
    rows = list(branches.iter_rows(named=True))
    texts: list[str] = []
    items: list[Item] = []
    for branch in rows:
        label = count_label_text(branch["vertex_count"])
        texts.append(label)
        width, height = label_extent(label, rendering)
        px, top = geometry.to_pixels(float(branch["x"]), float(branch["y_top"]))
        bottom = geometry.to_pixels(float(branch["x"]), float(branch["y_bottom"]))[1]
        boxes = _side_candidates(px, top, bottom, width, height)
        boxes.append((px - width / 2, top - foot - height, px + width / 2, top - foot))
        items.append(
            Item(
                (px, top),
                (width, height),
                [Spot(box) for box in boxes],
                fixed=True,
            )
        )
    scene = _scene(rendering, geometry, obstacles, lines, None, None)
    return _label_rows(zip(rows, texts, _solve(items, scene), strict=True))


def _below_step(
    mid: float,
    bottom: float,
    width: float,
    height: float,
    row: int,
    shift: float = 0.0,
    gap: float = LABEL_MARGIN,
) -> Spot:
    """Return the spot of a label ``row`` label heights below a step, centred on it.

    The step's horizontal run is at pixel row ``bottom`` and centred on column
    ``mid``. Row 0 sits ``SIDE_CLEARANCE`` under the step and needs no leader; each
    further row is a label height lower and gets the leader line that stands under
    the step down to the label. A ``shift`` slides the label sideways, the leader
    staying under the step.
    """
    top = bottom + SIDE_CLEARANCE + row * (height + gap)
    left = mid - width / 2 + shift
    box: Box = (left, top, left + width, top + height)
    leader = (mid, bottom + LEADER_GAP, mid, top) if row else None
    return Spot(box, leader)


def _beside_top(
    px: float, top: float, width: float, height: float, half: float, sign: int
) -> Spot:
    """Return the spot of a label above a branch's top, to the right (``sign`` 1) or
    the left of its column, with no leader."""
    bottom = top - half - TIP_CLEARANCE
    left = px + SIDE_CLEARANCE if sign > 0 else px - SIDE_CLEARANCE - width
    return Spot((left, bottom - height, left + width, bottom), None, ABOVE_PENALTY)


def _step_spots(
    px: float,
    top: float,
    mid: float,
    bottom: float,
    width: float,
    height: float,
    half: float,
    across: float,
) -> list[Spot]:
    """Return an internal branch's preferred spots, best first.

    Beside a long vertical segment, under the step (row 0 centred, each lower row
    with its leader and sliding sideways along it), and last above the branch's top:
    beside its column first (never over it: another branch may stand there, and the
    label would read as its count), then raised on it with a leader.
    """
    shifts = _shifts(width)
    reach = width / 2 - LEADER_CONTACT
    spots: list[Spot] = []
    if bottom - top >= height:
        spots += [
            Spot(box)  # beside the segment: as near its branch as a spot gets
            for box in _side_candidates(px, top, bottom, width, height)
        ]
    for row in range(MAX_STEP_ROWS + 1):
        for shift in shifts if row else [0.0]:
            spot = _below_step(mid, bottom, width, height, row, shift, across)
            spots.append(replace(spot, cost=_spot_cost(row, shift, reach)))
    spots += [_beside_top(px, top, width, height, half, sign) for sign in (1, -1)]
    for raised in range(1, MAX_STEP_RAISES + 1):
        for shift in shifts:
            spot = _above_tip(px, top, width, height, half, raised, shift, across)
            cost = ABOVE_PENALTY + _spot_cost(raised, shift, reach)
            spots.append(replace(spot, cost=cost))
    return spots


def place_step_labels(
    branches: pl.DataFrame,
    rendering: dict[str, Any],
    geometry: PanelGeometry,
    obstacles: Iterable[Box] = (),
    lines: Iterable[Box] = (),
    *,
    line_edges: Sequence[tuple[int, int] | None] | None = None,
    placed: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Lay out one count label for each internal branch, under its step.

    ``branches`` is ``internal_branches``: ``dna_sequence``, ``vertex_count``, ``x``
    (the vertical segment), ``y_top``, ``y_bottom`` and ``x_end`` (where the
    horizontal run at ``y_bottom`` ends). The close saddles of a split tree make
    most internal branches little more than that horizontal step, so a label goes
    centred under the middle of the step, ``SIDE_CLEARANCE`` below it. A branch
    whose vertical segment is a label's height or more long is also tried beside
    that segment first (``SIDE_FRACTIONS``, as a ridge label); the vertical nudge
    follows: when the spot is taken the label goes a label height lower, up to
    ``MAX_STEP_ROWS`` times, each lowered label joined to its step by a leader line
    and free to slide sideways along it (``_shifts``). Last come the spots above the
    segment's top, beside the column of the branch's upper end and then raised on
    it. Past those preferred spots, and whenever they are all poor, a label may
    stand anywhere round its step on a slanted leader (``_label_layout.solve``).

    The hard rules and the ranking are those of ``place_tip_labels``: no overlap with
    a marker (``obstacles``), another label or a leader line, and a branch of another
    edge under the label costing more than a branch of its own, both heavily
    (``lines``).
    ``placed`` is the table of the tip labels laid out before, which these keep clear
    of the same way. Columns as ``place_tip_labels``.
    """
    rows, texts, items = _step_items(branches, rendering, geometry)
    scene = _scene(rendering, geometry, obstacles, lines, line_edges, placed)
    return _label_rows(zip(rows, texts, _solve(items, scene), strict=True))


def _step_items(
    branches: pl.DataFrame, rendering: dict[str, Any], geometry: PanelGeometry
) -> tuple[list[dict[str, Any]], list[str], list[Item]]:
    """Return the internal branches' rows, their label texts and the items to place."""
    half = stop_marker_half(rendering)
    across = label_gap(rendering)[1]
    rows = list(branches.iter_rows(named=True))
    texts: list[str] = []
    items: list[Item] = []
    for branch in rows:
        label = count_label_text(branch["vertex_count"])
        texts.append(label)
        width, height = label_extent(label, rendering)
        px, top = geometry.to_pixels(float(branch["x"]), float(branch["y_top"]))
        end, bottom = geometry.to_pixels(
            float(branch["x_end"]), float(branch["y_bottom"])
        )
        mid = (px + end) / 2
        items.append(
            Item(
                (mid, bottom + LEADER_GAP),
                (width, height),
                _step_spots(px, top, mid, bottom, width, height, half, across),
                _item_edge(branch),
            )
        )
    return rows, texts, items


def label_boxes(labels: pl.DataFrame, rendering: dict[str, Any]) -> list[Box]:
    """Return the pixel boxes of placed ``labels`` and of their leader lines.

    A leader's box is the rectangle round its line, ``LEADER_WIDTH`` wide.
    """
    boxes: list[Box] = []
    for spot in _placed_spots(labels, rendering):
        boxes.append(spot.box)
        if spot.leader is not None:
            x0, y0, x1, y1 = spot.leader
            boxes.append(
                (
                    min(x0, x1) - LEADER_WIDTH / 2,
                    min(y0, y1),
                    max(x0, x1) + LEADER_WIDTH / 2,
                    max(y0, y1),
                )
            )
    return boxes


def label_overlaps(
    labels: pl.DataFrame, rendering: dict[str, Any]
) -> list[tuple[str, str]]:
    """Return the ``(text, text)`` pairs of placed ``labels`` that break a hard rule.

    Two labels come within ``label_gap`` of each other, or a leader line runs through
    another label or runs along another leader (two leaders that only cross are a soft
    cost, not a breach). An empty list means the layout is clean;
    the placement aims for it and ``branch_count_label_table`` warns when it falls
    short.
    """
    texts = labels["label"].to_list()
    return [
        (texts[first], texts[second])
        for first, second in clashing_pairs(
            _placed_spots(labels, rendering), label_gap(rendering)
        )
    ]


def count_label_layer(
    labels: pl.DataFrame, rendering: dict[str, Any], *, opacity: float = 1.0
) -> Any:
    """Draw ``labels`` (centres in panel pixels) in the peak grey, body font - 2.

    The text is placed by pixel (``scale=None``), so the layer adds nothing to the
    tree's x or y scale, which the branches alone define; ``opacity`` follows the
    branch the labels belong to. With ``branch_count_label_collision = "rotate"`` the
    text is turned a quarter turn about its centre (``ROTATED_ANGLE``).
    """
    turn = {"angle": ROTATED_ANGLE} if label_rotated(rendering) else {}
    return (
        alt.Chart(labels)
        .mark_text(
            align="center",
            baseline="middle",
            color=LABEL_COLOR,
            fontSize=count_label_font_size(rendering),
            opacity=opacity,
            **turn,
        )
        .encode(
            x=alt.X("px:Q", scale=None, axis=None),
            y=alt.Y("py:Q", scale=None, axis=None),
            text="label:N",
            tooltip=[
                alt.Tooltip("dna_sequence:N", title="Branch tip"),
                alt.Tooltip("vertex_count:Q", title="Branch vertex count"),
            ],
        )
    )


def leader_layer(labels: pl.DataFrame) -> Any | None:
    """Draw the leader lines of the labels raised off their tips, or ``None``.

    A thin peak-grey rule, placed by pixel like the labels (``scale=None``), runs from
    under each such label down to just above its tip (or from just below a step down
    to the label), straight and slanted when the label stands off its branch's column.
    """
    leaders = labels.filter(pl.col("leader_x").is_not_null())
    if leaders.is_empty():
        return None
    return (
        alt.Chart(leaders)
        .mark_rule(color=LABEL_COLOR, strokeWidth=LEADER_WIDTH)
        .encode(
            x=alt.X("leader_x:Q", scale=None, axis=None),
            x2="leader_x1:Q",
            y=alt.Y("leader_y0:Q", scale=None, axis=None),
            y2="leader_y1:Q",
            tooltip=[
                alt.Tooltip("dna_sequence:N", title="Branch tip"),
                alt.Tooltip("vertex_count:Q", title="Branch vertex count"),
            ],
        )
    )


def branch_count_label_table(
    plot_data: Any,
    rendering: dict[str, Any],
    *,
    size: tuple[float, float],
    y_domain: Sequence[float],
    stop_points: pl.DataFrame | None = None,
    wild_type_source: Any = None,
    colorbar_box: Box | None = None,
) -> pl.DataFrame:
    """Return the placed count labels of a tree (columns as ``place_tip_labels``).

    ``plot_data`` carries ``vertex_count`` on its edge table; ``size`` is the panel's
    pixel ``(width, height)`` and ``y_domain`` its fitness axis. The
    ``branch_count_label_mode`` picks the branches: the terminal tips of at most
    ``branch_count_label_max`` sequences (``"tips"``), or every branch (``"all"``:
    the tips first, then the internal branches, which keep clear of the tip labels
    and their leaders). The labels keep clear of ``stop_points`` (a table with ``x``
    and ``y``), the wild-type star of ``wild_type_source``, the ``colorbar_box`` (see
    ``count_colorbar_box``) and the drawn branches.
    """
    geometry = tree_geometry(plot_data, rendering, size=size, y_domain=y_domain)
    obstacles = marker_boxes(
        rendering,
        geometry,
        stop_points=stop_points,
        wild_type_source=wild_type_source,
        colorbar_box=colorbar_box,
    )
    branches = segment_boxes(plot_data.edge_segment_table, geometry)
    edges = segment_edges(plot_data.edge_segment_table)
    scene = _scene(rendering, geometry, obstacles, branches, edges, None)
    tip_rows, tip_texts, tip_items = _tip_items(
        terminal_tips(plot_data), rendering, geometry
    )
    step_rows: list[dict[str, Any]] = []
    step_texts: list[str] = []
    step_items: list[Item] = []
    if branch_count_label_mode(rendering) == "all":
        step_rows, step_texts, step_items = _step_items(
            internal_branches(plot_data), rendering, geometry
        )
    # The tips and the steps are laid out together, so that a tip label can make room
    # for a step label; the steps, which have the least room, go first.
    spots = _solve([*step_items, *tip_items], scene)
    step_spots, tip_spots = spots[: len(step_items)], spots[len(step_items) :]
    labels = pl.concat(
        [
            _label_rows(zip(tip_rows, tip_texts, tip_spots, strict=True)),
            _label_rows(zip(step_rows, step_texts, step_spots, strict=True)),
        ]
    )
    overlapping = label_overlaps(labels, rendering)
    if overlapping:
        warnings.warn(
            f"{len(overlapping)} branch count label(s) could not be placed clear of "
            "each other; the overprinted pairs are "
            + ", ".join(f"{first}/{second}" for first, second in overlapping[:6]),
            RuntimeWarning,
            stacklevel=2,
        )
    return labels


def branch_count_labels(
    plot_data: Any,
    rendering: dict[str, Any],
    *,
    size: tuple[float, float],
    y_domain: Sequence[float],
    stop_points: pl.DataFrame | None = None,
    wild_type_source: Any = None,
    colorbar_box: Box | None = None,
) -> list[Any]:
    """Return the layers of a tree's branch count labels, or ``[]``.

    The labels are those of ``branch_count_label_table``: leader lines first, so
    they go under their labels.
    """
    labels = branch_count_label_table(
        plot_data,
        rendering,
        size=size,
        y_domain=y_domain,
        stop_points=stop_points,
        wild_type_source=wild_type_source,
        colorbar_box=colorbar_box,
    )
    if labels.is_empty():
        return []
    leaders = leader_layer(labels)
    return [
        *([leaders] if leaders is not None else []),
        count_label_layer(labels, rendering),
    ]


def ridge_branches(
    plot_data: Any, edge_pairs: Iterable[tuple[int, int]]
) -> pl.DataFrame:
    """Return the branches of ``plot_data`` named by ``edge_pairs`` for side labels.

    ``edge_pairs`` are ``(source, target)`` tree-node indices as the plot tables
    orient them (lower end, then upper end). Columns as ``place_side_labels`` takes
    them: ``dna_sequence`` (the branch's upper end), ``vertex_count``, ``x`` (the
    branch's vertical segment, at its upper end) and ``y_top`` and ``y_bottom``.
    """
    nodes = {
        int(row["tree_node_index"]): row
        for row in plot_data.node_table.select(
            "tree_node_index", "dna_sequence", "x", "y"
        ).iter_rows(named=True)
    }
    wanted = {(int(source), int(target)) for source, target in edge_pairs}
    rows = []
    for edge in plot_data.edge_table.iter_rows(named=True):
        pair = (
            int(edge["source_tree_node_index"]),
            int(edge["target_tree_node_index"]),
        )
        if pair not in wanted:
            continue
        upper, lower = nodes[pair[1]], nodes[pair[0]]
        rows.append(
            {
                "dna_sequence": upper["dna_sequence"],
                "vertex_count": int(edge["vertex_count"]),
                "x": float(upper["x"]),
                "y_top": float(upper["y"]),
                "y_bottom": float(lower["y"]),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "dna_sequence": pl.String,
            "vertex_count": pl.Int64,
            "x": pl.Float64,
            "y_top": pl.Float64,
            "y_bottom": pl.Float64,
        },
    )


def ridge_count_labels(
    plot_data: Any,
    focused_pairs: Iterable[tuple[int, int]],
    dimmed_pairs: Iterable[tuple[int, int]],
    rendering: dict[str, Any],
    geometry: PanelGeometry,
    *,
    dimmed_opacity: float,
    stop_points: pl.DataFrame | None = None,
    wild_type_source: Any = None,
) -> list[Any]:
    """Return the count labels beside the ridges of a cropped excerpt, or ``[]``.

    Each ridge branch (``focused_pairs`` and ``dimmed_pairs``, see
    ``ridge_branches``) gets its count beside it, whatever the count: the two main
    ridges always carry their count, and the tree labels of ``branch_count_labels``
    are not drawn in an excerpt, so no branch is labelled twice. The label is level
    with the branch's middle, on the side with more room (``place_side_labels``). A
    focused ridge's label is drawn at full opacity and a dimmed one at
    ``dimmed_opacity``, like the branch it names. Returns one layer per opacity, the
    dimmed one first.
    """
    focused, dimmed = (
        ridge_branches(plot_data, focused_pairs),
        ridge_branches(plot_data, dimmed_pairs),
    )
    branches = pl.concat([focused, dimmed])
    if branches.is_empty():
        return []
    labels = place_side_labels(
        branches,
        rendering,
        geometry,
        marker_boxes(
            rendering,
            geometry,
            stop_points=stop_points,
            wild_type_source=wild_type_source,
        ),
        segment_boxes(plot_data.edge_segment_table, geometry),
    )
    layers = []
    for names, opacity in (
        (dimmed["dna_sequence"].to_list(), dimmed_opacity),
        (focused["dna_sequence"].to_list(), 1.0),
    ):
        chosen = labels.filter(pl.col("dna_sequence").is_in(names))
        if not chosen.is_empty():
            layers.append(count_label_layer(chosen, rendering, opacity=opacity))
    return layers
