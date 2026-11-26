"""Leaves in tree order, one beeswarm column of member sequences each.

``published_leaf_swarm`` and ``candidate_leaf_swarm`` order the
leaves of one clamped landscape's nominal split tree on x by their left-to-right
position on the tree drawing (``tree_order`` of the leaf table, from the tree layout's
own x; birth fitness and then the DNA sequence break ties). Each leaf is a column; y is raw fitness
and every sequence of the leaf's hill, the peak included, is one dot, laid out as a
deterministic beeswarm so overlap is visible. A triangle marks the leaf's birth
(peak) fitness, a triangle marker coloured by annotation (ADR 0012: wild type red,
stop codon crimson, else mid-grey), and a black diamond marks its adjacent saddle's
fitness. There are no lines: dots, triangles and diamonds only. Member dots are drawn at
``member_opacity`` (0.5); the stop-codon and synonymous-to-peak members are drawn in
their annotation colours on top of the plain grey ones.

Layout. Hundreds of columns do not fit one row legibly, so the ranked leaves wrap into
stacked rows of ``leaves_per_row`` columns that share the y scale. A column's swarm is
capped at ``swarm_half_width_fraction`` of the column width on each side of its centre.
Dots are placed at fixed slots half a dot apart; a dot takes the free slot nearest the
centre (the right one first, then the left), so equal fitnesses fan out symmetrically.
When a big hill fills every slot within the cap, the dot takes the slot farthest from
its neighbours instead: it overlaps, the overlap is the visible density, and the
subtitle counts such dots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ._charts import peak_annotation_kinds, peak_annotation_style
from ._count_labels import count_label_text
from ._four_graphs import PANELS
from ._leaf_data import LeafTables, figure_graph, fitness_axis, leaf_tables
from .palette import EMPHASIS_BLACK, PEAK_GREY, SYNONYMOUS_GREEN

SUFFIX = "leaf_swarm"
BAND_SUFFIX = "leaf_swarm_band"
_TIE = 1e-9


@dataclass(frozen=True)
class SwarmLayout:
    """Leaf and member tables with their wrapped position and swarm offset.

    ``leaves`` gains ``row`` and ``column`` (both from 0); ``members`` gains the same,
    ``swarm_offset`` (pixels from the column centre), ``overflow`` (the dot overlaps
    a neighbour because the column is full) and ``x`` (column units, the column's
    centre at ``column + 0.5``).
    """

    leaves: pl.DataFrame
    members: pl.DataFrame


def swarm_offsets(
    heights: Sequence[float], *, diameter: float, max_half_width: float
) -> tuple[list[float], list[bool]]:
    """Place dots of ``diameter`` at pixel ``heights`` in one column, deterministically.

    Returns ``(offsets, overflow)`` aligned with ``heights``. Dots are placed from the
    highest down (ties in input order) on slots ``diameter / 2`` apart, ordered by
    distance from the centre with the positive side first: ``0, +s, -s, +2s, ...``,
    up to ``max_half_width``. A dot takes the first slot whose distance to every dot
    already placed is at least ``diameter``. If none is free, ``overflow`` is true
    and it takes the slot with the greatest distance to its nearest neighbour (the
    one nearest the centre on a tie), so a full column spreads its excess instead of
    stacking it on one slot.
    """
    count = len(heights)
    if diameter <= 0 or max_half_width < 0:
        raise ValueError(
            "Dot diameter must be positive and the half-width not negative."
        )
    step = diameter / 2
    per_side = math.floor(max_half_width / step + _TIE)
    slots = np.array(
        [0.0] + [sign * k * step for k in range(1, per_side + 1) for sign in (1, -1)]
    )
    y = np.asarray(heights, dtype=float)
    order = np.argsort(-y, kind="stable")
    placed_y = np.empty(count)
    placed_x = np.empty(count)
    offsets = [0.0] * count
    overflow = [False] * count
    window = 0
    for done, index in enumerate(order):
        height = y[index]
        while window < done and placed_y[window] - height >= diameter:
            window += 1
        if window == done:
            choice = 0
        else:
            squared = (slots[:, None] - placed_x[None, window:done]) ** 2 + (
                placed_y[window:done] - height
            ) ** 2
            nearest = squared.min(axis=1)
            free = nearest >= diameter**2 - _TIE
            if free.any():
                choice = int(np.argmax(free))
            else:
                choice = int(np.argmax(nearest))
                overflow[index] = True
        placed_y[done] = height
        placed_x[done] = slots[choice]
        offsets[index] = float(slots[choice])
    return offsets, overflow


def swarm_layout(
    tables: LeafTables,
    *,
    leaves_per_row: int,
    column_width: float,
    row_height: float,
    domain: Sequence[float],
    diameter: float,
    half_width_fraction: float,
) -> SwarmLayout:
    """Wrap the ranked leaves into rows and lay out every column's swarm.

    ``domain`` is the y axis' display domain over ``row_height`` pixels, so distances
    between dots are measured as drawn. Within a leaf the dots are ordered by
    fitness descending then DNA sequence, so the layout depends on identities only.
    """
    if leaves_per_row < 1:
        raise ValueError("leaves_per_row must be at least one.")
    low, high = float(domain[0]), float(domain[1])
    pixels_per_unit = row_height / (high - low)
    cap = half_width_fraction * column_width
    leaves = tables.leaves.with_columns(
        ((pl.col("tree_order") - 1) // leaves_per_row).alias("row"),
        ((pl.col("tree_order") - 1) % leaves_per_row).alias("column"),
    )
    position = {
        rank: (row, column)
        for rank, row, column in leaves.select("rank", "row", "column").iter_rows()
    }
    parts = []
    for rank, group in tables.members.group_by("rank", maintain_order=True):
        group = group.sort(["raw_fitness", "dna_sequence"], descending=[True, False])
        offsets, overflow = swarm_offsets(
            [(value - low) * pixels_per_unit for value in group["raw_fitness"]],
            diameter=diameter,
            max_half_width=cap,
        )
        row, column = position[rank[0]]
        parts.append(
            group.with_columns(
                pl.lit(row, dtype=pl.Int64).alias("row"),
                pl.lit(column, dtype=pl.Int64).alias("column"),
                pl.Series("swarm_offset", offsets, dtype=pl.Float64),
                pl.Series("overflow", overflow, dtype=pl.Boolean),
            )
        )
    members = pl.concat(parts) if parts else tables.members.clear()
    members = members.with_columns(
        (pl.col("column") + 0.5 + pl.col("swarm_offset") / column_width).alias("x")
    )
    return SwarmLayout(leaves=leaves, members=members)


def swarm_rows(
    tables: LeafTables,
    appearance: dict[str, Any],
    *,
    prefix: str = SUFFIX,
) -> tuple[list[Any], dict[str, list[dict[str, Any]]], int, int]:
    """Lay out one graph's leaves and draw its stacked rows.

    Returns ``(rows, datasets, overlapping, leaf_rows)``: the row charts (each reads
    the named datasets ``{prefix}_row_{n}_members`` and ``{prefix}_row_{n}_leaves``),
    those datasets, how many dots overlap, and the number of rows. A dot is the
    mid-grey member colour, or the synonymous green when the sequence has the peak's
    amino-acid genotype; the adjacent saddle diamond is green when its genotype is
    the peak's, black otherwise. Green is ``synonymous_color`` (the trees' green).
    """
    for frame, column in (
        (tables.members, "is_synonymous"),
        (tables.leaves, "saddle_is_synonymous"),
    ):
        if column not in frame.columns:
            raise ValueError(
                f"The leaf tables lack the column {column!r}: the synonymous marks "
                "cannot be drawn (leaf_tables writes it)."
            )
    width = float(appearance.get("row_width", 960))
    height = float(appearance.get("row_height", 130))
    leaves = tables.leaves
    # A tree with fewer leaves than one row holds fills the row: the x domain is the
    # leaf count, not the configured row length.
    leaves_per_row = min(int(appearance.get("leaves_per_row", 60)), leaves.height)
    lows = [tables.members["raw_fitness"].min()]
    saddles = leaves["adjacent_saddle_scalar"].drop_nulls()
    if not saddles.is_empty():
        lows.append(saddles.min())
    domain, ticks = fitness_axis(
        min(lows),
        max(leaves["birth_scalar"].max(), tables.members["raw_fitness"].max()),
        appearance,
    )
    layout = swarm_layout(
        tables,
        leaves_per_row=leaves_per_row,
        column_width=width / leaves_per_row,
        row_height=height,
        domain=domain,
        diameter=float(appearance.get("member_dot_diameter", 3)),
        half_width_fraction=float(appearance.get("swarm_half_width_fraction", 0.4)),
    )
    kinds = peak_annotation_kinds(
        layout.leaves,
        wild_type=frozenset(
            layout.leaves.filter(pl.col("is_wild_type"))["peak_sequence"].to_list()
        ),
        label="peak_sequence",
    )
    colors = [
        peak_annotation_style(kind, appearance, color=PEAK_GREY)[0] for kind in kinds
    ]
    green = str(appearance.get("synonymous_color", SYNONYMOUS_GREEN))
    plain_dot = str(appearance.get("member_color", PEAK_GREY))
    plain_saddle = str(appearance.get("saddle_color", EMPHASIS_BLACK))
    members = layout.members
    leaf_frame = layout.leaves
    stop = peak_annotation_style("stop_codon", appearance, color=PEAK_GREY)[0]
    # Stop codon over synonymous over plain grey; the first two are drawn on top.
    members = members.with_columns(
        pl.when(pl.col("is_stop_codon"))
        .then(pl.lit(stop))
        .when(pl.col("is_synonymous"))
        .then(pl.lit(green))
        .otherwise(pl.lit(plain_dot))
        .alias("dot_color"),
        (pl.col("is_stop_codon") | pl.col("is_synonymous")).alias("highlighted"),
    )
    leaf_rows = leaf_frame.with_columns(
        pl.Series("birth_color", colors),
        pl.when(pl.col("saddle_is_synonymous"))
        .then(pl.lit(green))
        .otherwise(pl.lit(plain_saddle))
        .alias("saddle_color"),
    )
    row_count = int(leaf_rows["row"].max()) + 1
    key_flags = {
        "stop_codon": bool(leaves["birth_has_stop_codon"].any()),
        "wild_type": bool(leaves["is_wild_type"].any()),
        "synonymous": bool(
            members["is_synonymous"].any() or leaf_rows["saddle_is_synonymous"].any()
        ),
        "stop_member": bool(members["is_stop_codon"].any()),
    }
    datasets: dict[str, list[dict[str, Any]]] = {}
    panels = []
    for row in range(row_count):
        start = row * leaves_per_row
        row_members = members.filter(pl.col("row") == row).select(
            pl.col("x").round(4),
            pl.col("raw_fitness").round(6),
            "dot_color",
            "highlighted",
        )
        row_leaves = leaf_rows.filter(pl.col("row") == row).sort("column").select(
            "peak_sequence",
            "member_count",
            "column",
            "birth_scalar",
            "adjacent_saddle_scalar",
            "birth_color",
            "saddle_color",
        )
        datasets[f"{prefix}_row_{row}_members"] = row_members.to_dicts()
        datasets[f"{prefix}_row_{row}_leaves"] = row_leaves.to_dicts()
        panels.append(
            _row_panel(
                row,
                start=start,
                columns=row_leaves.height,
                domain=domain,
                ticks=ticks,
                width=width,
                height=height,
                appearance=appearance,
                first=row == 0,
                legend=key_flags if row == row_count - 1 else None,
                x_title=row == row_count - 1,
                prefix=prefix,
                per_row=leaves_per_row,
                single_row=row_count == 1,
            )
        )
    overlapping = int(layout.members["overflow"].sum())
    return panels, datasets, overlapping, leaves_per_row


def _finish_swarm(chart: Any, datasets: dict[str, Any], appearance: dict[str, Any]) -> Any:
    chart.datasets = datasets
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(appearance.get("font_size", 15)),
            titleFontSize=int(appearance.get("font_size", 15)),
            grid=bool(appearance.get("grid", False)),
        )
        .configure_view(strokeWidth=0)
    )


def _swarm_figure(
    tables: LeafTables,
    appearance: dict[str, Any],
    heading: str,
    first_line: str,
) -> Any:
    """The stacked-row swarm of ``tables`` under ``heading`` and its first subtitle line."""
    panels, datasets, overlapping, leaves_per_row = swarm_rows(tables, appearance)
    subtitle = [
        first_line,
        f"Leaves in the order they appear on the tree, left to right, {leaves_per_row} per row",
    ]
    if overlapping:
        subtitle.append(_overlap_note(overlapping))
    chart = alt.vconcat(*panels, spacing=int(appearance.get("row_gap", 14))).properties(
        title=alt.TitleParams(
            text=f"{heading}: sequences of every leaf",
            subtitle=subtitle,
            fontSize=int(appearance.get("heading_font_size", 18)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 14)),
            fontWeight="bold",
            anchor="start",
            offset=int(appearance.get("title_offset", 10)),
        )
    )
    return _finish_swarm(chart, datasets, appearance)


def build_leaf_swarm_figure(
    payload: dict[str, Any], settings: dict[str, Any], figure_id: str
) -> Any:
    """Build one graph's stacked-row leaf swarm from the comparison payload.

    The subtitle's leaf and hill-sequence counts are the registry's display text
    (``payload["manuscript_quantities"]``); the overlap line is a layout diagnostic.
    """
    slug, key, heading = figure_graph(figure_id, SUFFIX)
    quantities = payload["manuscript_quantities"]
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][figure_id],
    }
    tables = leaf_tables(
        payload["landscapes"][key], key=key, pairs=payload["annotations"][key].pairs
    )
    first_line = (
        f"Papkou et al. (2023) · {quantities.display(f'{slug}_graph.peak_count')}"
        f" leaves · {quantities.display(f'{slug}_graph.hill_sequence_count')} "
        "sequences in hills"
    )
    return _swarm_figure(tables, appearance, heading, first_line)


def build_leaf_swarm_band_figure(
    payload: dict[str, Any], settings: dict[str, Any], figure_id: str
) -> Any:
    """Build one graph's stacked-row leaf swarm of its BY band-simplified tree.

    ``payload`` is ``load_band_leaf_inputs``' result; the counts are those of the
    saved band tree's leaf tables.
    """
    slug, _, heading = figure_graph(figure_id, BAND_SUFFIX)
    panel = payload["panels"][slug]
    tables = panel["leaf_tables"]
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][figure_id],
    }
    first_line = (
        f"Papkou et al. (2023) · {count_label_text(tables.leaves.height)} leaves · "
        f"{count_label_text(tables.members.height)} sequences in hills · bootstrap "
        f"band quantile {panel['band_quantile']}, {count_label_text(panel['draws'])} draws"
    )
    return _swarm_figure(
        tables, appearance, f"{heading}, BY band-simplified tree", first_line
    )


# The four graphs of the saved-variant swarm figures (``_four_graphs``). The Published
# and Candidate nominal singles stay on the comparison path (``build_leaf_swarm_figure``);
# the two BY singles read the variants' saved comparison checkpoints.
PANEL_SLUGS = PANELS
FDR_PANEL_IDS = tuple(f"{slug}_leaf_swarm" for slug in ("published_fdr", "candidate_fdr"))


def _panel_tables(payload: dict[str, Any], slug: str) -> LeafTables:
    variant = PANEL_SLUGS[slug][0]
    try:
        return payload["panels"][variant]["leaf_tables"]
    except KeyError as error:
        raise ValueError(f"The four-graph payload lacks the {variant!r} leaves.") from error


def _panel_subtitle(tables: LeafTables, overlapping: int, per_row: int) -> list[str]:
    lines = [
        f"{count_label_text(tables.leaves.height)} leaves · "
        f"{count_label_text(tables.members.height)} sequences in hills",
        f"Leaves in the order they appear on the tree, left to right, {per_row} per row",
    ]
    if overlapping:
        lines.append(_overlap_note(overlapping))
    return lines


def build_leaf_swarm_panel_figure(
    payload: dict[str, Any], settings: dict[str, Any], figure_id: str
) -> Any:
    """One BY panel as its own figure (``{slug}_leaf_swarm`` for a saved variant)."""
    slug = figure_id.removesuffix(f"_{SUFFIX}")
    if slug not in PANEL_SLUGS or not figure_id.endswith(f"_{SUFFIX}"):
        raise ValueError(f"{figure_id!r} is not a leaf swarm panel ID.")
    _, graph, selection = PANEL_SLUGS[slug]
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][figure_id],
    }
    tables = _panel_tables(payload, slug)
    panels, datasets, overlapping, per_row = swarm_rows(tables, appearance)
    chart = alt.vconcat(*panels, spacing=int(appearance.get("row_gap", 14))).properties(
        title=alt.TitleParams(
            text=f"{graph} · {selection}: sequences of every leaf",
            subtitle=_panel_subtitle(tables, overlapping, per_row),
            fontSize=int(appearance.get("heading_font_size", 18)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 14)),
            fontWeight="bold",
            anchor="start",
            offset=int(appearance.get("title_offset", 10)),
        )
    )
    return _finish_swarm(chart, datasets, appearance)


def _key_flags(tables: LeafTables) -> dict[str, bool]:
    """Which optional key entries a graph's leaves and members call for."""
    return {
        "stop_codon": bool(tables.leaves["birth_has_stop_codon"].any()),
        "wild_type": bool(tables.leaves["is_wild_type"].any()),
        "synonymous": bool(
            tables.members["is_synonymous"].any()
            or tables.leaves["saddle_is_synonymous"].any()
        ),
        "stop_member": bool(tables.members["is_stop_codon"].any()),
    }


def _overlap_note(overlapping: int) -> str:
    """The layout diagnostic line: a property of the layout, exempt from the registry."""
    return f"{overlapping:,} dots overlap where a hill is wider than its column"


def _row_panel(
    row: int,
    *,
    start: int,
    columns: int,
    domain: list[float],
    ticks: list[float],
    width: float,
    height: float,
    appearance: dict[str, Any],
    first: bool,
    legend: dict[str, bool] | None,
    x_title: bool,
    prefix: str = SUFFIX,
    per_row: int | None = None,
    single_row: bool = False,
) -> Any:
    """One stacked row: dots, then the saddle diamonds and birth triangles on top."""
    if per_row is None:
        per_row = int(appearance.get("leaves_per_row", 60))
    step = int(appearance.get("rank_tick_step", 10))
    # Ticks at ranks that are multiples of ``step``; a column's centre is column + 0.5
    # and holds rank start + column + 1.
    ranks = [r for r in range(start + 1, start + columns + 1) if r % step == 0]
    x_axis = alt.Axis(
        values=[r - start - 0.5 for r in ranks],
        labelExpr=f"round(datum.value + {start} + 0.5)",
        title=(
            appearance.get("rank_axis_title", "Leaf order on tree")
            if x_title
            else None
        ),
        grid=False,
    )
    x_scale = alt.Scale(domain=[0, per_row], nice=False, zero=False)
    y_axis = alt.Axis(
        values=ticks,
        format=".1f",
        # The title fits one row's height only barely; the first row carries it.
        title=appearance.get("fitness_axis_title", "Fitness (WT = 0)")
        if first
        else None,
        grid=False,
    )
    y_scale = alt.Scale(domain=domain, nice=False, zero=False)
    x = alt.X("x:Q", axis=x_axis, scale=x_scale)
    y = alt.Y("raw_fitness:Q", axis=y_axis, scale=y_scale)
    diameter = float(appearance.get("member_dot_diameter", 3))
    members_data = alt.NamedData(name=f"{prefix}_row_{row}_members")

    def dots_layer(highlighted: bool) -> Any:
        return (
            alt.Chart(members_data)
            .transform_filter("datum.highlighted" if highlighted else "!datum.highlighted")
            .mark_circle(
                size=math.pi * diameter**2 / 4,
                opacity=float(appearance.get("member_opacity", 0.5)),
                aria=False,
            )
            .encode(x=x, y=y, color=alt.Color("dot_color:N", scale=None, legend=None))
        )

    # The plain grey members first, the stop-codon and synonymous ones over them.
    dots = [dots_layer(False), dots_layer(True)]
    leaves = alt.NamedData(name=f"{prefix}_row_{row}_leaves")
    x_centre = alt.X("centre:Q", scale=x_scale)
    birth_size = float(appearance.get("birth_marker_size", 40))
    saddle_size = float(appearance.get("saddle_marker_size", 30))
    saddle = (
        alt.Chart(leaves)
        .transform_filter("isValid(datum.adjacent_saddle_scalar)")
        .transform_calculate(centre="datum.column + 0.5")
        .mark_point(
            shape="diamond",
            filled=True,
            size=saddle_size,
            opacity=1.0,
            strokeWidth=0,
        )
        .encode(
            x=x_centre,
            y=alt.Y("adjacent_saddle_scalar:Q", scale=y_scale),
            color=alt.Color("saddle_color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("peak_sequence:N", title="Peak"),
                alt.Tooltip(
                    "adjacent_saddle_scalar:Q", title="Adjacent saddle", format=".4f"
                ),
            ],
        )
    )
    birth = (
        alt.Chart(leaves)
        .transform_calculate(centre="datum.column + 0.5")
        .mark_point(
            shape="triangle-up",
            filled=True,
            size=birth_size,
            opacity=1.0,
            strokeWidth=0,
        )
        .encode(
            x=x_centre,
            y=alt.Y("birth_scalar:Q", scale=y_scale),
            color=alt.Color("birth_color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("peak_sequence:N", title="Peak"),
                alt.Tooltip("member_count:Q", title="Sequences in hill"),
                alt.Tooltip("birth_scalar:Q", title="Birth", format=".4f"),
            ],
        )
    )
    layers = [*dots, saddle, birth]
    if legend is not None:
        layers.extend(_legend_layers(appearance, width, outside=single_row, **legend))
    return alt.layer(*layers).properties(width=int(width), height=int(height))


def _legend_layers(
    appearance: dict[str, Any],
    width: float,
    *,
    stop_codon: bool,
    wild_type: bool,
    synonymous: bool = False,
    stop_member: bool = False,
    outside: bool = False,
) -> list[Any]:
    """Draw the key in the last row's upper right, where no ranked leaf reaches.

    When the figure is one row (``outside``) every column is filled, so the key sits
    right of the plot instead.

    The last row is the lowest-ranked one and usually part empty, and its births sit
    far below the first row's. One entry per row, a glyph at a pixel offset and its
    label right of it: a dot for a member sequence, a triangle for birth (peak)
    fitness and a diamond for the adjacent saddle's, styled from the same keys as
    the marks. A stop-codon or wild-type birth marker is explained only when the
    figure has one.
    """
    font_size = int(appearance.get("legend_font_size", 15))
    row_gap = float(appearance.get("legend_row_gap", 4))
    padding = float(appearance.get("legend_padding", 8))
    glyph_x = (
        width + float(appearance.get("legend_outside_gap", 14))
        if outside
        else width - float(appearance.get("legend_width", 230))
    )
    member = appearance.get("member_color", PEAK_GREY)
    birth_size = float(appearance.get("birth_marker_size", 40))
    saddle_size = float(appearance.get("saddle_marker_size", 30))
    dot_size = math.pi * float(appearance.get("member_dot_diameter", 3)) ** 2 / 4
    entries: list[tuple[str, str, str, float]] = [
        ("Member sequence", "circle", member, dot_size),
        ("Birth (peak) fitness", "triangle-up", PEAK_GREY, birth_size),
        (
            "Adjacent saddle fitness",
            "diamond",
            appearance.get("saddle_color", EMPHASIS_BLACK),
            saddle_size,
        ),
    ]
    for present, kind, label in (
        (stop_codon, "stop_codon", "Stop-codon peak"),
        (wild_type, "wild_type", "Wild type"),
    ):
        if present:
            color, _ = peak_annotation_style(kind, appearance, color=PEAK_GREY)
            entries.append((label, "triangle-up", color, birth_size))
    if stop_member:
        entries.append(
            (
                "Stop-codon sequence",
                "circle",
                peak_annotation_style("stop_codon", appearance, color=PEAK_GREY)[0],
                dot_size,
            )
        )
    if synonymous:
        # A dot or diamond in the trees' green: the sequence has the peak's amino acids.
        entries.append(
            (
                "Synonymous to its peak",
                "circle",
                str(appearance.get("synonymous_color", SYNONYMOUS_GREEN)),
                dot_size,
            )
        )
    layers: list[Any] = []
    for index, (label, shape, color, size) in enumerate(entries):
        y = padding + font_size / 2 + index * (font_size + row_gap)
        mark: dict[str, Any] = {
            "shape": shape,
            "color": color,
            "size": size,
            "filled": True,
            "strokeWidth": 0,
        }
        if shape == "circle":
            mark["opacity"] = float(appearance.get("member_opacity", 0.5))
        data = alt.Data(values=[{"label": label}])
        layers.append(
            alt.Chart(data)
            .mark_point(**mark)
            .encode(x=alt.value(glyph_x), y=alt.value(y))
        )
        layers.append(
            alt.Chart(data)
            .mark_text(align="left", baseline="middle", fontSize=font_size)
            .encode(x=alt.value(glyph_x + 16), y=alt.value(y), text="label:N")
        )
    return layers
