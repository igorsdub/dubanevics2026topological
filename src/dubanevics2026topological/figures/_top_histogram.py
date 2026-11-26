"""Shared top histogram of the tree-persistence figures: count axis with headroom."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from .palette import (
    HISTOGRAM_GREY,
    TOP_HISTOGRAM_BINS,
    TOP_HISTOGRAM_COUNT_TICKS,
    TOP_HISTOGRAM_HEADROOM,
)


def top_histogram_bins(
    fitness: pl.Series,
    scalar_domain: tuple[float, float],
    bins: int = TOP_HISTOGRAM_BINS,
) -> pl.DataFrame:
    """Equal-width counts of every sequence's fitness over the fitness extent."""
    counts, edges = np.histogram(fitness.to_numpy(), bins=bins, range=scalar_domain)
    return pl.DataFrame(
        {"bin_min": edges[:-1], "bin_max": edges[1:], "count": counts.astype(int)}
    )


def top_histogram_settings(rendering: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    """Return ``(bins, count_ticks)`` from ``rendering``, else the palette defaults."""
    return (
        int(rendering.get("top_histogram_bins", TOP_HISTOGRAM_BINS)),
        tuple(
            int(tick)
            for tick in rendering.get("top_histogram_count_ticks", TOP_HISTOGRAM_COUNT_TICKS)
        ),
    )


def count_axis_limit(largest_bins: Iterable[int], rendering: dict[str, Any]) -> float:
    """Upper end of the count axis: headroom times the tallest bin of the panels that
    share the axis (``largest_bins``, one tallest bin per panel), never below the last
    configured tick so both ticks stay on the axis."""
    _, ticks = top_histogram_settings(rendering)
    if not ticks:
        raise ValueError("top_histogram_count_ticks must list at least one tick")
    headroom = float(rendering.get("top_histogram_headroom", TOP_HISTOGRAM_HEADROOM))
    tallest = max((int(value) for value in largest_bins), default=0)
    return max(headroom * tallest, float(ticks[-1]))


def top_histogram_chart(
    fitness: pl.Series,
    scalar_domain: tuple[float, float],
    rendering: dict[str, Any],
    *,
    width: int,
    height: int,
    bins: int | None = None,
    hidden: bool = False,
    title: str = "Sequences",
    count_limit: float | None = None,
) -> alt.Chart:
    """Marginal histogram on the diagram's fitness axis: count axis from 0 to
    ``count_limit`` (default: ``top_histogram_headroom`` times its own tallest bin) with
    ticks at the configured ``top_histogram_count_ticks`` only. With ``hidden`` the chart
    is drawn invisibly (the layout spacer above a tree)."""
    configured_bins, ticks = top_histogram_settings(rendering)
    table = top_histogram_bins(fitness, scalar_domain, bins or configured_bins)
    if int(table["count"].sum()) != fitness.len():
        raise ValueError("Fitness histogram bins do not count every sequence once.")
    if count_limit is None:
        count_limit = count_axis_limit([int(table["count"].max())], rendering)
    return (
        alt.Chart(table.with_columns(zero=pl.lit(0)))
        .mark_rect(
            color=rendering.get("histogram_color", HISTOGRAM_GREY),
            clip=True,
            opacity=0 if hidden else 1,
        )
        .encode(
            x=alt.X("bin_min:Q", axis=None),
            x2="bin_max:Q",
            y2="zero:Q",
            y=alt.Y(
                "count:Q",
                scale=alt.Scale(domain=[0, count_limit], nice=False),
                axis=alt.Axis(
                    title=title,
                    values=list(ticks),
                    labelExpr="datum.value === 0 ? '0' : upper(format(datum.value, '~s'))",
                    grid=False,
                    domain=False,
                    ticks=False,
                    labelOpacity=0 if hidden else 1,
                    titleOpacity=0 if hidden else 1,
                ),
            ),
            tooltip=["bin_min:Q", "bin_max:Q", "count:Q"],
        )
        .properties(width=width, height=height)
    )
