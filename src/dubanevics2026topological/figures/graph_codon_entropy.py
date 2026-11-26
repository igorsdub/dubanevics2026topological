"""Supplementary S14: positional codon entropy of the Full and Papkou graphs.

Two heatmaps, and no tree. Each cell is the Shannon entropy in bits of the amino
acid at one codon position, over that graph's sequences at or above the cell's
raw-fitness threshold. The values come from ``analysis.graph_codon_entropy``,
which calls the same ``compute_entropy_sweeps`` as the main-text codon-entropy
figure. Fitness outside a graph's own span is left blank.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import polars as pl

from ..analysis.graph_codon_entropy import POPULATION_NAMES, GraphCodonEntropy
from ._count_labels import count_label_text
from .ridge_entropy_view import _panel
from .labels import graph_label
from .tree_persistence_view import STYLES

HEADINGS = {"full": graph_label("full"), "published": graph_label("published")}


def build_graph_codon_entropy_figure(
    entropy: GraphCodonEntropy,
    settings: dict[str, Any],
    style: str,
    *,
    title: str,
    subtitle: str,
) -> Any:
    """The ``graph_codon_entropy`` figure in ``annotated`` or ``vanilla`` style.

    Appearance is ``d10.graph_codon_entropy``, merged over ``[rendering]``.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}.")
    figures = settings["figures"]
    rendering = {
        **figures.get("rendering", {}),
        **figures["d10"]["graph_codon_entropy"],
    }
    annotated = style == "annotated"
    ranges = entropy.sidecar["fitness_range"]
    counts = entropy.sidecar["counts"]
    low, high = ranges["full"]
    pad = (high - low) * float(rendering.get("fitness_padding_fraction", 0.02))
    domain = (low - pad, high + pad)
    panels = []
    for index, name in enumerate(POPULATION_NAMES):
        heading = [HEADINGS[name]]
        if annotated:
            heading.append(f"{count_label_text(counts[name])} sequences")
        panel = _panel(
            entropy.sweeps.filter(pl.col("population") == name),
            tuple(ranges[name]),
            domain,
            rendering,
            show_axis=index == 0,
            # Every panel declares the same legend so the shared colour scale merges them
            # into one bar; ``legend=None`` on any panel would suppress it.
            show_legend=True,
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
        .resolve_scale(color="shared", y="shared")
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


__all__ = ["STYLES", "build_graph_codon_entropy_figure"]
