"""Peak-rank figures of the Papkou et al. (2023) published peak set.

``peak_rank``: each peak's fitness against its rank (1 = highest fitness).
``peak_rank_annotated``: the same, with the stop-codon peaks drawn as
crosses (the tree figures' stop marker). ``peak_rank_histogram``: the
annotated figure beside the marginal histogram of every sequence's fitness, on the
same fitness axis. None carries a title or subtitle, and none marks synonymous
variants. The fitness axis has the shared ``figures.fitness_domain`` and the tree
panel's height, so it lines up with the tree figures pixel for pixel.
Everything is read from the payload (``analysis/peak_rank.py`` table plus the Full
population); nothing is rebuilt.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import polars as pl

from ._charts import (
    histogram_chart,
    marker_config,
    publication_scalar_scales,
    tree_histogram_chart_config,
    tree_histogram_settings,
)
from .palette import COUNT_BLACK, tree_panel_size

PEAK_RANK_ID = "peak_rank"
PEAK_RANK_ANNOTATED_ID = "peak_rank_annotated"
PEAK_RANK_HISTOGRAM_ID = "peak_rank_histogram"
PEAK_RANK_IDS = (PEAK_RANK_ID, PEAK_RANK_ANNOTATED_ID, PEAK_RANK_HISTOGRAM_ID)
STOP_LEGEND_LABEL = "Stop-codon peak"


def rank_ticks(count: int, step: int) -> list[int]:
    """Integer rank ticks: rank 1, then every ``step``-th rank up to ``count``."""
    return [1, *range(step, count + 1, step)]


def _peak_frame(table: pl.DataFrame) -> pl.DataFrame:
    """The ranked peaks with the ``scalar`` fitness column the fitness axes share."""
    return table.select(
        "rank",
        "dna_sequence",
        "aa_sequence",
        "has_stop_codon",
        pl.col("fitness").alias("scalar"),
    )


def peak_rank_chart(
    table: pl.DataFrame,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    fitness_domain: tuple[float, float],
    *,
    stop_codons: bool,
) -> Any:
    """The rank-fitness panel; ``stop_codons`` adds the crosses and their legend."""
    width, height = tree_panel_size(rendering, figure_settings)
    frame = _peak_frame(table)
    rank_step = int(rendering.get("rank_tick_step", 100))
    ticks = sorted({round(fitness_domain[0], 1), 0.0, round(fitness_domain[1], 1)})
    x = alt.X(
        "rank:Q",
        title=rendering.get("peak_rank_x_title", "Peak rank (by fitness)"),
        scale=alt.Scale(domain=[0.5, frame.height + 0.5], nice=False, zero=False),
        axis=alt.Axis(
            values=rank_ticks(frame.height, rank_step), format="d", grid=False
        ),
    )
    y = alt.Y(
        "scalar:Q",
        title=rendering.get(
            "shared_tree_axis_title", "Fitness (at birth, WT = 0)"
        ),
        scale=alt.Scale(domain=list(fitness_domain), nice=False, zero=False),
        axis=alt.Axis(values=ticks, format=".1f", grid=False),
    )
    tooltip = ["rank:Q", "dna_sequence:N", "aa_sequence:N", "scalar:Q"]
    peaks = (
        alt.Chart(frame)
        .mark_point(
            filled=True,
            color=str(rendering.get("peak_color", COUNT_BLACK)),
            size=int(rendering.get("peak_size", 24)),
            opacity=1,
            clip=True,
        )
        .encode(x=x, y=y, tooltip=tooltip)
    )
    layers = [peaks]
    if stop_codons:
        marker = marker_config(rendering, "stop_codon", default_shape="x")
        stops = frame.filter(pl.col("has_stop_codon")).with_columns(
            kind=pl.lit(STOP_LEGEND_LABEL)
        )
        layers.append(
            alt.Chart(stops)
            .mark_point(
                shape=marker.shape,
                filled=False,
                size=int(rendering.get("stop_codon_tree_size", marker.size)),
                strokeWidth=float(rendering.get("stop_codon_stroke_width", 1.1)),
                opacity=marker.opacity,
                clip=True,
            )
            .encode(
                x=x,
                y=y,
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(
                        domain=[STOP_LEGEND_LABEL], range=[str(marker.color)]
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                        symbolType=marker.shape,
                        symbolStrokeWidth=float(
                            rendering.get("stop_codon_stroke_width", 1.1)
                        ),
                        symbolFillColor="transparent",
                        labelFontSize=int(rendering.get("legend_font_size", 15)),
                    ),
                ),
                tooltip=tooltip,
            )
        )
    return alt.layer(*layers).properties(width=width, height=height)


def build_peak_rank_figure(
    payload: dict[str, Any], settings: dict[str, Any], figure_id: str
) -> Any:
    """One of the three peak-rank figures; ``figure_id`` is one of ``PEAK_RANK_IDS``."""
    if figure_id not in PEAK_RANK_IDS:
        raise ValueError(f"{figure_id!r} is not a peak-rank figure.")
    figure_settings, rendering, fitness_domain = tree_histogram_settings(
        settings, figure_id
    )
    table = payload["peak_rank"]
    if not table.height:
        raise ValueError("The peak-rank table has no peaks.")
    outside = table.filter(
        (pl.col("fitness") < fitness_domain[0])
        | (pl.col("fitness") > fitness_domain[1])
    )
    if outside.height:
        raise ValueError(
            f"{outside.height} ranked peaks lie outside figures.fitness_domain "
            f"{fitness_domain} and would be clipped; widen it."
        )
    panel = peak_rank_chart(
        table,
        figure_settings,
        rendering,
        fitness_domain,
        stop_codons=figure_id != PEAK_RANK_ID,
    )
    def finish(chart):
        # The shared scales the D10 and slide trees get, so fitness 0 sits on the same row.
        return publication_scalar_scales(
            tree_histogram_chart_config(chart, rendering), fitness_domain, rendering
        )

    if figure_id != PEAK_RANK_HISTOGRAM_ID:
        return finish(panel)
    population = payload["full_fitness"].rename({"raw_fitness": "scalar"})
    histogram = histogram_chart(
        population,
        None,
        fitness_domain,
        None,
        figure_settings,
        rendering,
        count_axis=None,
    )
    return finish(
        alt.hconcat(
            panel, histogram, spacing=int(rendering.get("panel_gap", 10))
        ).resolve_scale(x="independent", y="shared", color="independent")
    )
