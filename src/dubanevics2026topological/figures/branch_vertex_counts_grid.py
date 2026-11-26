"""Branch vertex counts of the four graphs, overlaid by Horton-Strahler order, as a 2x2.

Columns are the Published graph (left) and the Candidate graph (right); rows are the
nominal selection (top) and the Benjamini-Yekutieli q = 0.05 selection (bottom), the
layout of the main-text figures (``_four_graphs``). Each panel is an overlaid bar
chart: x is the branch vertex count (one tree edge is one branch), y is the branch
count as a fraction of the panel's own tallest bar (domain 0 to 1.05), and the bars of
Horton-Strahler orders 1 to 4 (``HORTON_STRAHLER_COLORS``) are drawn over one another, not stacked, at ``bar_opacity`` (0.7); the order with the fewest
branches over the whole figure is drawn in front, the most numerous at the back. The
panel's branch total ("1,013 branches") is written inside the plot at the top left.
All four panels share the bin edges (log-spaced when the counts are heavy-tailed, 100
bins by default). One horizontal legend titled
``Horton-Strahler order`` sits inside the top-right panel, drawn on top of the bars.
The branch tables are the saved comparison checkpoints' own; nothing is recomputed.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ._count_labels import count_label_text
from ._four_graphs import GRID_ORDER, PANELS, panel_title
from .branch_decomposition_charts import histogram_rows, vertex_count_edges
from .palette import HORTON_STRAHLER_COLORS, INK_GREY

FIGURE_ID = "branch_vertex_counts_2x2"
ORDERS = (1, 2, 3, 4)
LEGEND_TITLE = "Horton–Strahler order"
# A span of ten or more between the smallest and largest branch is heavy-tailed.
HEAVY_TAIL = 10
Y_TITLE = "Branches (fraction of max)"
# Each panel's y is count / that panel's tallest bar; a little headroom above 1.
Y_MAX = 1.05
_REQUIRED = {"branch_vertex_count", "horton_strahler_order"}


def _edges(counts: list[int], bins: int) -> tuple[np.ndarray, bool]:
    """Bin edges, log-spaced when the counts are heavy-tailed."""
    low = min(counts)
    high = max(counts)
    log_scale = high >= low * HEAVY_TAIL and high > low
    if log_scale:
        return vertex_count_edges(high, bins), True
    if bins < 1 or high < low:
        raise ValueError("Vertex-count histograms need at least one bin.")
    if high == low:
        return np.array([float(low), float(low) + 1.0]), False
    edges = np.linspace(float(low), float(high), bins + 1)
    edges[0], edges[-1] = float(low), float(high)
    return edges, False


def panel_table(payload: dict[str, Any], slug: str) -> pl.DataFrame:
    """The branch table of one panel's graph, checked."""
    variant = PANELS[slug][0]
    try:
        table = payload["panels"][variant]["branch_table"]
    except KeyError as error:
        raise ValueError(f"The four-graph payload lacks the {variant!r} branches.") from error
    missing = _REQUIRED - set(table.columns)
    if missing:
        raise ValueError(f"The {variant!r} branch table is missing {sorted(missing)}.")
    if table.is_empty():
        raise ValueError(f"The {variant!r} branch table has no branches.")
    unknown = set(table["horton_strahler_order"].to_list()) - set(ORDERS)
    if unknown:
        raise ValueError(f"Horton–Strahler orders outside 1–4: {sorted(unknown)}.")
    if int(table["branch_vertex_count"].min()) < 1:
        raise ValueError("Branch vertex counts start at one.")
    return table


def shared_edges(tables: list[pl.DataFrame], bins: int) -> tuple[np.ndarray, bool]:
    """Bin edges over the counts of every panel, log-spaced when heavy-tailed."""
    counts = [
        int(value)
        for table in tables
        for value in table["branch_vertex_count"].to_list()
    ]
    return _edges(counts, bins)


def draw_ranks(tables: list[pl.DataFrame]) -> dict[str, int]:
    """Draw rank of each order: 0 is drawn first (at the back), the highest in front.

    Orders are ranked by their branch count over every panel, the most numerous
    first, so the order with the fewest branches is drawn in front. Ties put the
    higher Horton-Strahler order at the back.
    """
    totals = {
        order: sum(
            int((table["horton_strahler_order"] == order).sum()) for table in tables
        )
        for order in ORDERS
    }
    sequence = sorted(ORDERS, key=lambda order: (-totals[order], -order))
    return {str(order): rank for rank, order in enumerate(sequence)}


def stacked_rows(
    table: pl.DataFrame, edges: np.ndarray, ranks: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """One row per order per bin on ``edges``; an order's counts sum to its branches.

    ``ranks`` adds each row's ``draw_rank`` (see ``draw_ranks``).
    """
    rows: list[dict[str, Any]] = []
    for order in ORDERS:
        values = table.filter(pl.col("horton_strahler_order") == order)[
            "branch_vertex_count"
        ].to_list()
        for bin_row in histogram_rows(values, edges):
            row = {**bin_row, "order": str(order)}
            if ranks is not None:
                row["draw_rank"] = ranks[str(order)]
            rows.append(row)
    return rows


def _tallest_bar(rows: list[dict[str, Any]]) -> int:
    """The largest count of any single bar (the bars overlay, they do not stack)."""
    return max(row["count"] for row in rows)


def fraction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows with ``fraction`` = count / the panel's tallest bar (in [0, 1])."""
    tallest = _tallest_bar(rows)
    return [{**row, "fraction": row["count"] / tallest} for row in rows]


def _panel(
    rows: list[dict[str, Any]],
    *,
    title: str,
    total_label: str,
    domain: list[float],
    log_scale: bool,
    y_max: float,
    width: int,
    height: int,
    seam: float,
    appearance: dict[str, Any],
    show_x_title: bool,
    show_y_title: bool,
    show_legend: bool,
) -> alt.Chart:
    order_labels = [str(order) for order in ORDERS]
    legend = (
        alt.Legend(
            orient="none",
            legendX=width - float(appearance.get("legend_width", 190)),
            legendY=float(appearance.get("legend_inset_y", 6)),
            direction="horizontal",
            columns=len(ORDERS),
            title=LEGEND_TITLE,
            symbolType="square",
            padding=float(appearance.get("legend_padding", 4)),
            fillColor="white",
            # Above the bars (z-index 0 is the marks' layer).
            zindex=1,
            labelFontSize=int(appearance.get("font_size", 13)),
            titleFontSize=int(appearance.get("font_size", 13)),
        )
        if show_legend
        else None
    )
    color = alt.Color(
        "order:N",
        title=LEGEND_TITLE,
        scale=alt.Scale(
            domain=order_labels,
            range=[
                *appearance.get("horton_strahler_colors", HORTON_STRAHLER_COLORS)
            ],
        ),
        sort=order_labels,
        legend=legend,
    )
    bars = (
        alt.Chart(alt.Data(values=rows))
        .transform_filter("datum.count > 0")
        .mark_bar(
            strokeWidth=seam,
            binSpacing=0,
            opacity=float(appearance.get("bar_opacity", 0.7)),
        )
        .encode(
            x=alt.X(
                "start:Q",
                title="Branch vertex count" if show_x_title else None,
                scale=alt.Scale(
                    type="log" if log_scale else "linear", domain=domain, nice=False
                ),
                axis=alt.Axis(grid=False),
            ),
            x2="end:Q",
            y=alt.Y(
                "fraction:Q",
                title=Y_TITLE if show_y_title else None,
                stack=None,
                scale=alt.Scale(domain=[0, y_max], nice=False),
                axis=alt.Axis(grid=False),
            ),
            y2=alt.Y2Datum(0),
            color=color,
            # Drawing order of the overlaid bars: the lowest rank first (at the back).
            order=alt.Order("draw_rank:Q", sort="ascending"),
        )
    )
    # The panel's branch total, inside the plot area at the top left (the legend
    # of the top-right panel is on the right).
    total = (
        alt.Chart(alt.Data(values=[{"label": total_label}]))
        .mark_text(
            align="left",
            baseline="top",
            dx=float(appearance.get("total_inset_x", 6)),
            dy=float(appearance.get("total_inset_y", 4)),
            fontSize=int(appearance.get("font_size", 13)),
            color=INK_GREY,
        )
        .encode(
            x=alt.value(0),
            y=alt.value(0),
            text="label:N",
        )
    )
    return alt.layer(bars, total).properties(
        width=width,
        height=height,
        title=alt.TitleParams(
            text=title,
            anchor="start",
            fontSize=int(appearance.get("heading_font_size", 16)),
        ),
    )


def build_branch_vertex_counts_grid(
    payload: dict[str, Any], settings: dict[str, Any]
) -> alt.Chart:
    """The four overlaid histograms as a 2x2, one legend inside the top-right panel."""
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][FIGURE_ID],
    }
    tables = {slug: panel_table(payload, slug) for slug in PANELS}
    edges, log_scale = shared_edges(list(tables.values()), int(appearance.get("vertex_count_bins", 100)))
    ranks = draw_ranks(list(tables.values()))
    rows = {slug: stacked_rows(table, edges, ranks) for slug, table in tables.items()}
    rows = {slug: fraction_rows(panel_rows) for slug, panel_rows in rows.items()}
    y_max = Y_MAX
    domain = [float(edges[0]), float(edges[-1])]
    last_row = len(GRID_ORDER) - 1
    legend_cell = (0, len(GRID_ORDER[0]) - 1)
    grid_rows = []
    for row_index, slugs in enumerate(GRID_ORDER):
        cells = []
        for column_index, slug in enumerate(slugs):
            cells.append(
                _panel(
                    rows[slug],
                    title=panel_title(slug),
                    total_label=f"{count_label_text(tables[slug].height)} branches",
                    domain=domain,
                    log_scale=log_scale,
                    y_max=y_max,
                    width=int(appearance.get("panel_width", 360)),
                    height=int(appearance.get("panel_height", 230)),
                    seam=float(appearance.get("bar_seam_width", 0.5)),
                    appearance=appearance,
                    show_x_title=row_index == last_row,
                    show_y_title=column_index == 0,
                    show_legend=(row_index, column_index) == legend_cell,
                )
            )
        grid_rows.append(
            alt.hconcat(
                *cells, spacing=int(appearance.get("panel_gap", 30))
            ).resolve_scale(color="independent")
        )
    chart = alt.vconcat(
        *grid_rows, spacing=int(appearance.get("row_gap", 30))
    ).resolve_scale(color="independent")
    heading = appearance.get("title")
    if heading:
        chart = chart.properties(
            title=alt.TitleParams(
                text=str(heading),
                anchor="start",
                fontSize=int(appearance.get("grid_heading_font_size", 18)),
            )
        )
    font_size = int(appearance.get("font_size", 15))
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size, grid=False)
        .configure_view(strokeWidth=0)
    )
