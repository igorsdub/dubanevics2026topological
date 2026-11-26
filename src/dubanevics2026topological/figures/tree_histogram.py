"""Full, Papkou and Topographer single-population tree-histogram figures."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import altair as alt

from ._charts import (
    apply_tree_histogram_domain,
    count_colorbar_box,
    count_domain,
    histogram_chart,
    overlay_count_colorbar,
    plot_data_with_metrics,
    publication_scalar_scales,
    selected_view_as_tree,
    tree_chart,
    tree_histogram_chart_config,
    tree_histogram_settings,
    tree_population,
)
from ._count_labels import branch_count_labels
from .palette import tree_panel_size
from .slide_tree import (
    slide_stop_overrides,
    slide_tree_analysis,
)
from .tree_comparison import (
    _threshold_layers,
    _threshold_value,
    _with_histogram_threshold,
)

BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID = "bootstrap_tree_histogram"

# Landscape key of the figures that draw their own functional threshold; the Full
# graph and the band-smoothed twin have none.
_THRESHOLD_LANDSCAPE = {
    "published_tree_histogram": "study_clamped",
    "candidate_tree_histogram": "candidate_clamped",
}


def _own_threshold(analysis: Any, item: Any, figure_id: str) -> float | None:
    """The panel's functional threshold, read as ``tree_comparison`` reads it.

    ``None`` when the figure has none (Full graph) or the payload carries none.
    """
    key = _THRESHOLD_LANDSCAPE.get(figure_id)
    if key is None:
        return None
    try:
        return _threshold_value(analysis, item, key)
    except (ValueError, AttributeError):
        return None


def build_tree_histogram_figure(
    analysis: dict[str, Any],
    settings: dict[str, Any],
    figure_id: str,
    *,
    slide: dict[str, Any] | None = None,
):
    """Show one population's split tree beside its all-vertex histogram."""
    population_id = {
        "full_tree_histogram": "full",
        "published_tree_histogram": "article",
        "candidate_tree_histogram": "candidate",
    }[figure_id]
    item = analysis["landscapes"].get(
        {"full": "full", "article": "study_clamped", "candidate": "candidate_clamped"}[
            population_id
        ]
    )
    if getattr(analysis, "legacy_mixture", False):
        if item is None:
            # Context-aware loads skip the full analysis; read the same cached
            # checkpoint the context-free load reads.
            from ..analysis import load_cached_analysis

            item = load_cached_analysis()
        tree_analysis = item.trees["split"]
    else:
        tree_analysis = analysis["saved_layouts"][population_id]
        if item is None:
            # The saved full-graph tree is drawn without the in-memory full analysis.
            population = analysis["saved_populations"][population_id]
            item = SimpleNamespace(
                scalar_domain=(
                    float(population["scalar"].min()),
                    float(population["scalar"].max()),
                )
            )
    return _tree_histogram_figure(
        analysis,
        settings,
        figure_id,
        item,
        tree_analysis,
        population_id,
        slide=slide,
    )


def _tree_histogram_figure(
    analysis: dict[str, Any],
    settings: dict[str, Any],
    figure_id: str,
    item: Any,
    tree_analysis: Any,
    population_id: str,
    *,
    count_labels: bool = False,
    slide: dict[str, Any] | None = None,
):
    """Draw ``tree_analysis`` beside the histogram of its vertices' raw fitness.

    ``tree_analysis`` carries the plot data, branch table, critical-node and
    branch-member tables and the wild-type marker; ``item`` is the landscape
    whose scalar domain the tree panel is drawn on. With ``count_labels`` the
    branches are labelled with their counts as ``branch_count_label_mode`` says
    (``_count_labels.branch_count_labels``, under the wild-type star).
    """
    figure_settings, rendering, fitness_domain = tree_histogram_settings(
        settings, figure_id
    )
    population = tree_population(tree_analysis, analysis, population_id)
    annotated = False
    coloring = "count"
    stops = None
    if slide is not None:
        # Slide render: ``{"style": vanilla|annotated, "coloring": count|none}``.
        rendering = {
            **rendering,
            **slide_stop_overrides(settings, population_id == "full"),
        }
        tree_analysis, annotated = slide_tree_analysis(tree_analysis, slide["style"])
        coloring = slide.get("coloring", "count")
    threshold = _own_threshold(analysis, item, figure_id)
    plot_data = plot_data_with_metrics(tree_analysis)
    branch_domain = count_domain(plot_data.edge_table)
    labels = (
        branch_count_labels(
            plot_data,
            rendering,
            size=tree_panel_size(rendering, figure_settings),
            y_domain=fitness_domain,
            wild_type_source=tree_analysis,
            colorbar_box=count_colorbar_box(figure_settings, rendering, branch_domain),
        )
        if count_labels
        else []
    )
    tree = tree_chart(
        plot_data,
        tree_analysis,
        item,
        figure_settings,
        rendering,
        branch_domain,
        linked=False,
        annotated=annotated,
        count_labels=labels,
        coloring=coloring,
        stop_identities=stops,
        underlays=(
            []
            if threshold is None
            else _threshold_layers(
                threshold, fitness_domain, rendering, labelled=False
            )
        ),
    )
    tree = apply_tree_histogram_domain(tree, fitness_domain)
    if coloring == "count":
        tree = overlay_count_colorbar(tree, figure_settings, rendering, branch_domain)
    histogram = histogram_chart(
        population,
        None,
        fitness_domain,
        None,
        figure_settings,
        rendering,
        count_axis=None,
    )
    if threshold is not None:
        histogram = _with_histogram_threshold(
            histogram, threshold, fitness_domain, rendering
        )
    chart = tree_histogram_chart_config(
        alt.hconcat(
            tree, histogram, spacing=int(rendering.get("panel_gap", 10))
        ).resolve_scale(x="independent", y="shared", color="independent"),
        rendering,
    )
    # Three fitness ticks (the domain's two ends and 0) on every render, so fitness 0
    # lines up across figures. A slide keeps the shared padded scales; the paper
    # figure keeps its configured display domain exactly.
    return publication_scalar_scales(
        chart,
        fitness_domain,
        rendering,
        exact_domain=slide is None,
        extra_ticks=(
            {} if threshold is None else {"y": [threshold], "bin_min": [threshold]}
        ),
    )


def build_bootstrap_tree_histogram_figure(
    analysis: dict[str, Any], settings: dict[str, Any]
):
    """The candidate tree-histogram composition on the band-smoothed tree.

    The same builder as the nominal candidate histogram, given the tree of
    ``analysis["bootstrap_band_view"]`` (step 2 of ADR 0008) instead of the
    nominal one. The branches are coloured by their post-absorption
    vertex count (``selected_view_as_tree``: absorbed members folded in) and the
    large terminal branches are labelled with it. The histogram counts every
    retained vertex, which the smoothing keeps unchanged.
    """
    return _tree_histogram_figure(
        analysis,
        settings,
        BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID,
        analysis["landscapes"]["candidate_clamped"],
        selected_view_as_tree(
            analysis["bootstrap_band_view"],
            legacy_mixture=getattr(analysis, "legacy_mixture", False),
        ),
        "candidate",
        count_labels=True,
    )

