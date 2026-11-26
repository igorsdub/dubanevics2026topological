"""D10 view ``ridge_entropy``: positional codon entropy of the KEM and TCL ridges.

Spec 2, Section 6, supplementary view. Three heatmaps of one band-simplified tree
share a vertical raw-fitness axis: the whole selected graph (core) and its KEM and
TCL ridges. Each cell is the Shannon entropy in bits of the amino acid at one codon
position, over the sequences of that population at or above the cell's fitness
threshold. Fitness outside a population's own span is left blank. The ``annotated``
style adds each ridge's functional count and the legend note. Everything is read
from a saved ``ridge_entropy_sweeps`` product (``analysis.ridge_entropy``) and
its ridges summary; nothing is recomputed.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import altair as alt
import polars as pl

from ..analysis.ridge_entropy import POPULATION_NAMES, RidgeEntropy
from ._count_labels import count_label_text
from .codon_entropy import MAX_ENTROPY
from .labels import KEM_BRANCH, TCL_BRANCH
from .tree_persistence_view import STYLES

POSITIONS = ("1", "2", "3")
HEADINGS = {
    "core": "Band-simplified graph",
    "kem": KEM_BRANCH,
    "tcl": TCL_BRANCH,
}


def entropy_cells(sweep: pl.DataFrame, low: float, high: float) -> list[dict[str, Any]]:
    """Cells of one population: each retained threshold tiled to its midpoints.

    Cells are clamped to the population's own fitness span ``low`` to ``high``;
    thresholds outside it and null entropies draw nothing.
    """
    rows = sweep.sort(["aa_position", "threshold_scalar"]).to_dicts()
    thresholds = sorted(
        {
            row["threshold_scalar"]
            for row in rows
            if row["entropy_bits"] is not None
            and low <= row["threshold_scalar"] <= high
        }
    )
    midpoints = [(a + b) / 2 for a, b in pairwise(thresholds)]
    bounds = {
        value: (
            low if index == 0 else midpoints[index - 1],
            high if index == len(thresholds) - 1 else midpoints[index],
        )
        for index, value in enumerate(thresholds)
    }
    return [
        {
            "position": str(row["aa_position"]),
            "lower": bounds[row["threshold_scalar"]][0],
            "upper": bounds[row["threshold_scalar"]][1],
            "threshold": row["threshold_scalar"],
            "entropy_bits": row["entropy_bits"],
            "active": row["active_sequence_count"],
        }
        for row in rows
        if row["entropy_bits"] is not None and row["threshold_scalar"] in bounds
    ]


def _panel(
    sweep: pl.DataFrame,
    span: tuple[float, float] | None,
    domain: tuple[float, float],
    rendering: dict[str, Any],
    *,
    show_axis: bool,
    show_legend: bool,
) -> alt.LayerChart:
    """One population's heatmap; outside its span, and for an empty population, nothing is drawn."""
    width = int(rendering.get("heatmap_width", 220))
    height = int(rendering.get("heatmap_height", 320))
    x = alt.X(
        "position:N",
        sort=list(POSITIONS),
        title=rendering.get("position_axis_title", "Amino-acid position"),
        axis=alt.Axis(labelAngle=0, domain=False),
        scale=alt.Scale(paddingInner=0, paddingOuter=0),
    )
    y_scale = alt.Scale(domain=list(domain), zero=False, nice=False)
    cells = [] if span is None else entropy_cells(sweep, *span)
    y_encoding = alt.Y(
        "lower:Q",
        scale=y_scale,
        title=rendering.get("fitness_axis_title", "Raw fitness") if show_axis else None,
        axis=alt.Axis(
            format=str(rendering.get("fitness_tick_format", ".1f")),
            domain=False,
            ticks=True,
        )
        if show_axis
        else None,
    )
    # An empty rectangle that carries the shared axes; outside the span nothing is drawn.
    layers = [
        alt.Chart(
            alt.Data(values=[{"position": "1", "lower": domain[0], "upper": domain[0]}])
        )
        .mark_rect(opacity=0)
        .encode(x=x, y=y_encoding, y2="upper:Q")
    ]
    if cells:
        layers.append(
            alt.Chart(alt.Data(values=cells))
            .mark_rect()
            .encode(
                x=x,
                y=y_encoding,
                y2="upper:Q",
                color=alt.Color(
                    "entropy_bits:Q",
                    scale=alt.Scale(
                        scheme=str(rendering.get("entropy_scheme", "viridis")),
                        domain=[0.0, MAX_ENTROPY],
                    ),
                    legend=alt.Legend(
                        title=list(
                            rendering.get("entropy_title", ["Entropy", "(bits)"])
                        ),
                        gradientLength=int(rendering.get("legend_length", 110)),
                        orient="right",
                    )
                    if show_legend
                    else None,
                ),
                tooltip=[
                    alt.Tooltip("position:N", title="Position"),
                    alt.Tooltip("threshold:Q", title="Fitness", format=".3f"),
                    alt.Tooltip("entropy_bits:Q", title="Entropy (bits)", format=".3f"),
                    alt.Tooltip("active:Q", title="Active sequences"),
                ],
            )
        )
    return alt.layer(*layers).properties(width=width, height=height)


def build_ridge_entropy_figure(
    entropy: RidgeEntropy,
    functional: dict[str, int],
    settings: dict[str, Any],
    style: str,
    *,
    title: str,
    subtitle: str,
) -> Any:
    """The ``ridge_entropy`` figure in ``annotated`` or ``vanilla`` style.

    ``functional`` maps ``kem`` and ``tcl`` to their functional counts from the
    ridges summary. ``settings`` is the figures configuration; the view's
    appearance table is ``d10.ridge_entropy``.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}.")
    figures = settings["figures"]
    rendering = {**figures.get("rendering", {}), **figures["d10"]["ridge_entropy"]}
    annotated = style == "annotated"
    ranges = entropy.sidecar["fitness_range"]
    counts = entropy.sidecar["counts"]
    low, high = ranges["core"]
    pad = (high - low) * float(rendering.get("fitness_padding_fraction", 0.02))
    domain = (low - pad, high + pad)
    panels = []
    for index, name in enumerate(POPULATION_NAMES):
        heading = [HEADINGS[name], f"{count_label_text(counts[name])} sequences"]
        if annotated and name in functional:
            heading[1] += f" · {count_label_text(functional[name])} functional"
        if not counts[name]:
            heading = [HEADINGS[name], "absent (summit not a maximum)"]
        panel = _panel(
            entropy.sweeps.filter(pl.col("population") == name),
            None if not counts[name] else tuple(ranges[name]),
            domain,
            rendering,
            show_axis=index == 0,
            show_legend=index == len(POPULATION_NAMES) - 1,
        )
        panels.append(
            panel.properties(
                title=alt.TitleParams(
                    heading,
                    anchor="start",
                    fontSize=int(rendering.get("panel_title_font_size", 14)),
                )
            )
        )
    note = "Entropy of the amino acid at each codon position"
    font_size = int(rendering.get("font_size", 15))
    chart = (
        alt.hconcat(*panels, spacing=int(rendering.get("panel_gap", 12)))
        .resolve_scale(color="independent", y="shared")
        .properties(
            title=alt.TitleParams(text=title, subtitle=[subtitle, note], anchor="start")
        )
    )
    return (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(grid=False, labelFontSize=font_size, titleFontSize=font_size)
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
        .configure_title(
            fontSize=int(rendering.get("heading_font_size", font_size)),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


__all__ = ["STYLES", "build_ridge_entropy_figure", "entropy_cells"]
