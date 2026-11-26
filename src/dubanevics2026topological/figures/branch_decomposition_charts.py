"""Charts for the exploratory branch-decomposition notebook.

This is not a registered figure: it has no ``FigureSpec``, no figure identifier
and no variants, so ``build figures`` never reaches it. The notebook
``notebooks/papkou2023_branch_decomposition.py`` reads the cached stage
(``analysis/branch_decomposition.py``) and draws, per Horton–Strahler order, a
fitness histogram of the sequences on the branches (tree edges) of that order and
a vertex-count histogram of those branches, and optionally one small fitness
histogram per branch. Appearance comes from ``[exploratory.branch_decomposition]``
in ``config/figures.toml``, which sits outside ``[figures]`` because no figure
owns it (``build`` leaves it out of every figure's dependency record); fonts and
the fitness and vertex-count axis titles come from ``[rendering]``.

Histograms are binned here and drawn as bars between explicit bin edges, so every
panel of a chart shares its edges and the counts can be checked: fitness bins are
equal in width over the shared fitness domain, vertex-count bins are equal in
log10 over 1 to the largest branch (branch sizes span orders of magnitude, and a
log axis shows the real counts).

Count axes carry explicit integer ticks and a domain that keeps the top tick label
inside the panel, so the headings of panels in one row sit at one height. The
fitness histograms' count axis is a symmetric-log one by default
(``log_count_axis``, ``ln(1 + count / log_count_constant)``): a few functional
sequences stay visible beside thousands of low-fitness ones, an empty bin sits on
the baseline, and a count of 1 is not flattened (a plain log axis would put it on
the baseline). The y-axis title
offset is fixed across the panels of a chart (``minExtent``), so the row headings
of the order chart start at one x, and the facets are one grid, so their columns
line up whatever the width of the tick labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from .palette import LIGHT_GREY

SETTINGS_PATH = ("exploratory", "branch_decomposition")


def appearance_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``[exploratory.branch_decomposition]`` of ``config/figures.toml``."""
    table: Any = settings["figures"]
    for key in SETTINGS_PATH:
        if not isinstance(table, Mapping) or key not in table:
            raise KeyError(
                "config/figures.toml has no [exploratory.branch_decomposition] table."
            )
        table = table[key]
    return dict(table)


def fitness_edges(
    domain: Sequence[float], bins: int, values: Sequence[float] = ()
) -> np.ndarray:
    """Equal-width bin edges over ``domain``, widened to hold every value."""
    low = min([float(domain[0]), *map(float, values)])
    high = max([float(domain[1]), *map(float, values)])
    if bins < 1 or not high > low:
        raise ValueError("Fitness histograms need at least one bin over a real range.")
    return np.linspace(low, high, bins + 1)


def vertex_count_edges(maximum: int, bins: int) -> np.ndarray:
    """Bin edges equal in log10 from 1 to the largest branch (at least 10).

    The last edge is the largest size exactly: ``10 ** log10(n)`` often rounds
    just below ``n``, which would leave the largest branch outside the bins.
    """
    if bins < 1:
        raise ValueError("Vertex-count histograms need at least one bin.")
    top = max(int(maximum), 10)
    edges = np.logspace(0.0, math.log10(top), bins + 1)
    edges[0], edges[-1] = 1.0, float(top)
    return edges


def histogram_rows(values: Sequence[float], edges: np.ndarray) -> list[dict[str, Any]]:
    """Counts of ``values`` between consecutive ``edges`` as ``start``/``end``/``count``.

    The last bin is closed, so the largest value is counted; every value must fall
    inside the edges, and the counts sum to the number of values.
    """
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    if int(counts.sum()) != len(values):
        raise ValueError("Histogram edges do not cover every value.")
    return [
        {"start": float(a), "end": float(b), "count": int(c)}
        for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)
    ]


def _x_axis(title: str | None, rows: list[dict[str, Any]], log_x: bool) -> alt.Axis:
    """No grid; a log axis ticks at the powers of ten it spans."""
    if not log_x:
        return alt.Axis(title=title, grid=False)
    top = math.floor(math.log10(rows[-1]["end"]))
    return alt.Axis(
        title=title,
        grid=False,
        values=[10**power for power in range(top + 1)],
        format=",d",
    )


# Vega's default axis: a tick of 5 px and 2 px between tick and label.
TICK_SIZE = 5
LABEL_PADDING = 2


def _nice_step(raw: float) -> int:
    """The step of 1, 2, 5 times a power of ten nearest ``raw``, as d3 picks ticks."""
    power = math.floor(math.log10(raw))
    error = raw / 10**power
    if error >= math.sqrt(50):
        factor = 10
    elif error >= math.sqrt(10):
        factor = 5
    elif error >= math.sqrt(2):
        factor = 2
    else:
        factor = 1
    return max(1, round(factor * 10**power))


def count_ticks(maximum: int, height: int, *, log: bool) -> list[int]:
    """Integer ticks of a count axis from 0 up to ``maximum``.

    A linear axis gets about one tick per 40 px of ``height`` at a round step; a
    symmetric-log axis gets 0 and the powers of ten.
    """
    if maximum < 1:
        return [0]
    if log:
        powers = range(math.floor(math.log10(maximum)) + 1)
        return [0, *(10**power for power in powers)]
    wanted = max(2, math.ceil(height / 40))
    return list(range(0, maximum + 1, _nice_step(maximum / wanted)))


def count_domain_top(
    maximum: int,
    ticks: Sequence[int],
    height: int,
    font_size: int,
    *,
    log: bool,
    constant: float = 1.0,
) -> float:
    """The top of a count axis: it holds the largest count and the top tick label.

    A tick at the very top would push half its label above the panel and raise the
    panel's heading with it; the domain leaves the room of half a label (and 2 px)
    above the top tick, in the axis's own transform: linear, or symmetric-log
    ``ln(1 + x / constant)``.
    """
    room = (font_size / 2 + 2) / height
    last = ticks[-1]
    if log:
        needed = constant * math.expm1(math.log1p(last / constant) / (1 - room))
    else:
        needed = last / (1 - room)
    return max(float(maximum), needed, 1.0)


def count_axis_extent(
    maxima: Sequence[int], height: int, font_size: int, *, log: bool
) -> int:
    """Pixels the tick labels of a count axis need, for the widest of ``maxima``.

    Digits count 0.6 em and commas 0.3 em, a little over Arial's, so a label is
    never wider than the estimate. Used as ``minExtent``, which fixes where the
    axis title sits, so the panels of a chart share one title offset.
    """
    widest = max(
        max(count_ticks(maximum, height, log=log), default=0) for maximum in maxima
    )
    em = sum(0.3 if char == "," else 0.6 for char in f"{widest:,}")
    return math.ceil(em * font_size) + TICK_SIZE + LABEL_PADDING


def _count_axis(
    title: str | None, ticks: Sequence[int], extent: int | None = None
) -> alt.Axis:
    """Counts are whole numbers: the given integer ticks, thousands separated."""
    options: dict[str, Any] = {"values": list(ticks), "format": ",d"}
    if title is not None and extent is not None:
        options["minExtent"] = extent
    return alt.Axis(title=title, grid=False, **options)


def _bars(
    rows: list[dict[str, Any]],
    *,
    x_title: str | None,
    y_title: str | None,
    color: str,
    seam: float,
    height: int,
    font_size: int,
    log_x: bool = False,
    log_y: bool = False,
    log_constant: float = 1.0,
    y_extent: int | None = None,
) -> alt.Chart:
    """Bars between the bin edges of ``rows``; empty bins draw nothing.

    ``seam`` is the width of a stroke in the fill colour: antialiasing leaves a pale
    line between two bars of one height, and the stroke covers it. A zero-height
    bar would show as a line, so the empty bins are filtered out of the drawing
    (they stay in the data, so every panel keeps its edges).
    """
    domain = [rows[0]["start"], rows[-1]["end"]]
    largest = max(row["count"] for row in rows)
    ticks = count_ticks(largest, height, log=log_y)
    top = count_domain_top(
        largest, ticks, height, font_size, log=log_y, constant=log_constant
    )
    count_scale = (
        alt.Scale(type="symlog", constant=log_constant, domain=[0, top], nice=False)
        if log_y
        else alt.Scale(domain=[0, top], nice=False)
    )
    return (
        alt.Chart(alt.Data(values=rows))
        .transform_filter("datum.count > 0")
        .mark_bar(
            color=color,
            stroke=color,
            strokeWidth=seam,
            binSpacing=0,
            orient="vertical",
        )
        .encode(
            x=alt.X(
                "start:Q",
                axis=_x_axis(x_title, rows, log_x),
                scale=alt.Scale(
                    type="log" if log_x else "linear", domain=domain, nice=False
                ),
            ),
            x2="end:Q",
            y=alt.Y(
                "count:Q",
                axis=_count_axis(y_title, ticks, y_extent),
                scale=count_scale,
            ),
            tooltip=[
                alt.Tooltip("start:Q", title="from", format=",.3~g"),
                alt.Tooltip("end:Q", title="to", format=",.3~g"),
                alt.Tooltip("count:Q", title="count", format=","),
            ],
        )
    )


def _configured(
    chart: Any,
    rendering: Mapping[str, Any],
    *,
    font_size: int | None = None,
    heading_font_size: int | None = None,
) -> Any:
    """Apply the shared fonts; a chart of small panels passes its own sizes."""
    size = font_size or int(rendering.get("font_size", 15))
    heading = heading_font_size or int(rendering.get("heading_font_size", size))
    return (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(labelFontSize=size, titleFontSize=size)
        .configure_title(fontSize=heading, anchor="start")
        .configure_view(strokeWidth=0)
    )


def axis_titles(
    appearance: Mapping[str, Any], rendering: Mapping[str, Any]
) -> dict[str, str]:
    """The four axis titles, from ``config/figures.toml``.

    The fitness and vertex-count titles are ``[rendering]``'s ``tree_axis_title``
    and ``branch_count_title`` (a two-line legend title there, one line on an
    axis); the two count titles are the exploratory table's.
    """
    return {
        "fitness": str(rendering["tree_axis_title"]),
        "vertex_count": " ".join(
            str(rendering.get("branch_count_title", "Branch vertex count")).split()
        ),
        "sequences": str(appearance["sequence_axis_title"]),
        "branches": str(appearance["branch_axis_title"]),
    }


def _selected(branches: pl.DataFrame, orders: Sequence[int]) -> pl.DataFrame:
    chosen = branches.filter(pl.col("horton_strahler_order").is_in(list(orders)))
    if chosen.is_empty():
        raise ValueError("No branch has one of the selected Horton–Strahler orders.")
    return chosen


def order_histograms(
    branches: pl.DataFrame,
    members: pl.DataFrame,
    orders: Sequence[int],
    settings: Mapping[str, Any],
) -> Any:
    """One row per selected order: fitness of its sequences beside its branch sizes.

    The fitness panel counts every sequence on the branches of the order (each
    sequence once); the vertex-count panel counts the branches themselves. Bin
    edges are shared by every row, so the rows compare, and so are the y-axis title
    offsets (the widest tick labels of any row), so the rows' headings start at one
    x. The fitness panel's count axis is symmetric-log when ``log_count_axis`` is
    set.
    """
    appearance = appearance_settings(settings)
    rendering = settings["figures"].get("rendering", {})
    titles = axis_titles(appearance, rendering)
    chosen = _selected(branches, orders)
    fitness = fitness_edges(
        settings["figures"]["fitness_domain"],
        int(appearance["fitness_bins"]),
        members["raw_fitness"].to_list(),
    )
    sizes = vertex_count_edges(
        int(branches["branch_vertex_count"].max()), int(appearance["vertex_count_bins"])
    )
    color = appearance.get("histogram_color", LIGHT_GREY)
    seam = float(appearance["bar_seam_width"])
    log_count = bool(appearance["log_count_axis"])
    constant = float(appearance["log_count_constant"])
    width, height = int(appearance["panel_width"]), int(appearance["panel_height"])
    size = int(rendering.get("font_size", 15))
    drawn = []
    for order in sorted(set(chosen["horton_strahler_order"].to_list())):
        here = chosen.filter(pl.col("horton_strahler_order") == order)
        sequences = members.filter(pl.col("horton_strahler_order") == order)
        drawn.append(
            (
                order,
                here.height,
                sequences.height,
                histogram_rows(sequences["raw_fitness"].to_list(), fitness),
                histogram_rows(here["branch_vertex_count"].to_list(), sizes),
            )
        )
    left_extent = count_axis_extent(
        [max(row["count"] for row in rows) for _, _, _, rows, _ in drawn],
        height,
        size,
        log=log_count,
    )
    right_extent = count_axis_extent(
        [max(row["count"] for row in rows) for _, _, _, _, rows in drawn],
        height,
        size,
        log=False,
    )
    rows = []
    for order, branch_total, sequence_total, fitness_rows, size_rows in drawn:
        title = alt.TitleParams(
            text=(
                f"Horton–Strahler order {order}: {branch_total:,} "
                f"{'branch' if branch_total == 1 else 'branches'}, "
                f"{sequence_total:,} "
                f"{'sequence' if sequence_total == 1 else 'sequences'}"
            ),
            anchor="start",
        )
        left = _bars(
            fitness_rows,
            x_title=titles["fitness"],
            y_title=titles["sequences"],
            color=color,
            seam=seam,
            height=height,
            font_size=size,
            log_y=log_count,
            log_constant=constant,
            y_extent=left_extent,
        ).properties(width=width, height=height)
        right = _bars(
            size_rows,
            x_title=titles["vertex_count"],
            y_title=titles["branches"],
            color=color,
            seam=seam,
            height=height,
            font_size=size,
            log_x=True,
            y_extent=right_extent,
        ).properties(width=width, height=height)
        rows.append(
            alt.hconcat(left, right, spacing=int(rendering.get("panel_gap", 10)))
            .resolve_scale(y="independent")
            .properties(title=title)
        )
    chart = alt.vconcat(*rows, spacing=int(rendering.get("comparison_panel_gap", 16)))
    return _configured(chart, rendering)


def facet_label(row: Mapping[str, Any]) -> str:
    """Panel heading of one branch: its id, its order and its sequence count."""
    count = int(row["branch_vertex_count"])
    return (
        f"{row['branch_id']} · order {row['horton_strahler_order']} · "
        f"{count:,} {'sequence' if count == 1 else 'sequences'}"
    )


def branch_facets(
    branches: pl.DataFrame,
    members: pl.DataFrame,
    orders: Sequence[int],
    settings: Mapping[str, Any],
) -> Any:
    """One small fitness histogram per branch of the selected orders.

    Panels run from the highest order down, and from the fittest upper endpoint
    down within an order, ``facet_columns`` to a row, in one grid, so the columns
    line up across rows whatever the width of the tick labels; the last row holds
    only the panels there are, so no cell is left with empty axes. Bin edges are
    shared; each panel has its own count axis (the trunk holds orders of magnitude
    more sequences than a hill; symmetric-log when ``log_count_axis`` is set) and
    its own fitness tick labels. The count title sits on the first panel of each
    row and the fitness title on every panel with no panel below it. Every font is
    ``facet_font_size``, which suits the panel size.
    """
    appearance = appearance_settings(settings)
    rendering = settings["figures"].get("rendering", {})
    titles = axis_titles(appearance, rendering)
    chosen = _selected(branches, orders).sort(
        ["horton_strahler_order", "birth_fitness", "branch_id"],
        descending=[True, True, False],
    )
    ids = chosen["branch_id"].to_list()
    edges = fitness_edges(
        settings["figures"]["fitness_domain"],
        int(appearance["fitness_bins"]),
        members.filter(pl.col("branch_id").is_in(ids))["raw_fitness"].to_list(),
    )
    size = int(appearance["facet_font_size"])
    columns = int(appearance["facet_columns"])
    height = int(appearance["facet_panel_height"])
    color = appearance.get("histogram_color", LIGHT_GREY)
    seam = float(appearance["bar_seam_width"])
    log_count = bool(appearance["log_count_axis"])
    constant = float(appearance["log_count_constant"])
    histograms = [
        histogram_rows(
            members.filter(pl.col("branch_id") == row["branch_id"])[
                "raw_fitness"
            ].to_list(),
            edges,
        )
        for row in chosen.iter_rows(named=True)
    ]
    extent = count_axis_extent(
        [max(row["count"] for row in rows) for rows in histograms],
        height,
        size,
        log=log_count,
    )
    panels = []
    for position, (row, rows) in enumerate(
        zip(chosen.iter_rows(named=True), histograms, strict=True)
    ):
        panels.append(
            _bars(
                rows,
                x_title=None if position + columns < len(ids) else titles["fitness"],
                y_title=titles["sequences"] if position % columns == 0 else None,
                color=color,
                seam=seam,
                height=height,
                font_size=size,
                log_y=log_count,
                log_constant=constant,
                y_extent=extent,
            ).properties(
                width=int(appearance["facet_panel_width"]),
                height=height,
                title=alt.TitleParams(
                    text=facet_label(row), anchor="start", fontSize=size
                ),
            )
        )
    chart = alt.concat(
        *panels,
        columns=columns,
        spacing=int(rendering.get("comparison_panel_gap", 16)),
    ).resolve_scale(y="independent")
    return _configured(chart, rendering, font_size=size, heading_font_size=size)


def save_pngs(
    charts: Mapping[str, Any], directory: Path, settings: Mapping[str, Any]
) -> list[Path]:
    """Save each named chart as ``<name>.png`` in ``directory`` at the configured dpi."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, chart in charts.items():
        path = directory / f"{name}.png"
        chart.save(path, ppi=int(settings["figures"]["png_dpi"]))
        written.append(path)
    return written
