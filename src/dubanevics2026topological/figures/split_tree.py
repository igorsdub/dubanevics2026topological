"""Linked split tree and persistence figure."""

from __future__ import annotations

from typing import Any

import altair as alt

from ._charts import (
    biology_axis_contract,
    count_domain,
    overlay_count_colorbar,
    persistence_chart,
    persistence_legend_layers,
    persistence_panel_size,
    persistence_with_counts,
    plot_data_with_metrics,
    publication_scalar_scales,
    tree_chart,
)
from ._quantities import ManuscriptQuantities


def build_split_tree_figure(
    analysis: Any,
    figure_id: str,
    settings: dict[str, Any],
    quantities: ManuscriptQuantities,
) -> Any:
    """Build the linked tree and persistence figure (vanilla: count colours only).

    The title's peak and sequence counts are the registry's display text
    (``full_graph.peak_count``, ``full_graph.vertex_count``).
    """
    figure_settings = settings["figures"]["figures"][figure_id]
    rendering = settings["figures"].get("rendering", {})
    tree_analysis = analysis.trees["split"]
    plot_data = plot_data_with_metrics(tree_analysis)
    tree_count_domain = count_domain(plot_data.edge_table)

    persistence_table = persistence_with_counts(
        tree_analysis.persistence_table,
        plot_data,
        tree_count_domain,
    )
    tree = tree_chart(
        plot_data,
        tree_analysis,
        analysis,
        figure_settings,
        rendering,
        tree_count_domain,
        linked=True,
        annotated=False,
    )
    # The colour bar reads its tick settings from the merged table, as the other
    # tree figures do, so a figure's ``count_legend_ticks`` takes effect.
    tree = overlay_count_colorbar(
        tree, figure_settings, {**rendering, **figure_settings}, tree_count_domain
    )
    persistence = persistence_chart(
        persistence_table,
        figure_settings,
        rendering,
        tree_count_domain,
        analysis.scalar_domain,
        include_config=False,
    )
    persistence = biology_axis_contract(
        persistence,
        y_title=None,
        x_title=rendering.get("persistence_x_title", "Fitness at death"),
    )
    # The key sits inside the diagram, bottom right: peak entry only (no valley markers).
    persistence = alt.layer(
        persistence,
        *persistence_legend_layers(
            *persistence_panel_size(figure_settings, rendering),
            rendering,
            stop_codon=False,
        ),
    )
    chart = alt.hconcat(
        tree,
        persistence,
        spacing=int(rendering.get("panel_gap", 10)),
    ).resolve_scale(x="independent", y="shared", color="independent")

    chart = chart.properties(
        title=alt.TitleParams(
            text="Full graph",
            subtitle=[
                "Papkou et al. (2023) · full dataset",
                (
                    f"{quantities.display('full_graph.peak_count')} peaks · "
                    f"{quantities.display('full_graph.vertex_count')} sequences"
                ),
            ],
        )
    )
    panel_settings = figure_settings.get("tree", figure_settings.get("persistence", {}))
    font_size = int(panel_settings.get("font_size", rendering.get("font_size", 15)))
    chart = (
        chart.configure(
            font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"),
        )
        .configure_axis(
            labelFontSize=int(rendering.get("axis_label_font_size", font_size)),
            titleFontSize=int(font_size),
        )
        .configure_legend(
            labelFontSize=int(rendering.get("legend_font_size", 15)),
            titleFontSize=int(rendering.get("legend_font_size", 15)),
        )
        .configure_title(
            fontSize=int(
                rendering.get("heading_font_size", rendering.get("font_size", 15))
            ),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )
    return publication_scalar_scales(chart, analysis.scalar_domain, rendering)
