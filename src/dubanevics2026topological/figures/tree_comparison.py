"""Full, Published and Candidate tree-histogram comparison figure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import altair as alt
from topographer.plotting import compute_fitness_histogram_table

from ._charts import (
    apply_tree_histogram_domain,
    functional_threshold_layers,
    functional_threshold_text,
    histogram_chart,
    overlay_count_colorbar,
    plot_data_with_metrics,
    publication_scalar_scales,
    threshold_label_padding,
    tree_chart,
    tree_histogram_chart_config,
    tree_histogram_settings,
    tree_population,
)
from ._top_histogram import count_axis_limit
from .palette import THRESHOLD_GREY
from .slide_tree import (
    slide_stop_overrides,
    slide_tree_analysis,
)


def build_tree_comparison_figure(
    analysis: dict[str, Any],
    settings: dict[str, Any],
    *,
    slide: dict[str, Any] | None = None,
):
    """Build Full, Published, and Candidate tree–histogram rows.

    ``slide`` (``{"style", "coloring"}``) is the frameless slide render: no row titles,
    the style's markers and, with ``coloring="none"``, plain-edge trees with no colour bar.

    Every number in a row's subtitle is the registry's display text
    (``analysis["manuscript_quantities"]``, see ``docs/manuscript-quantity-registry-design.md``).
    """
    # A slide draws no subtitle, so it needs no registry display text.
    quantities = None if slide is not None else analysis["manuscript_quantities"]
    figure_settings, rendering, fitness_domain = tree_histogram_settings(
        settings, "tree_comparison"
    )
    landscapes = analysis["landscapes"]
    panel_specs = (
        ("full", "Full graph", "full"),
        ("study_clamped", "Published graph", "article"),
        ("candidate_clamped", "Candidate graph", "candidate"),
    )
    products = {}
    max_branch_count = 1
    for key, _title, population_id in panel_specs:
        item = landscapes[key]
        if item is None and getattr(analysis, "legacy_mixture", False):
            from ..analysis import load_cached_analysis

            item = load_cached_analysis()
        tree_analysis = (
            (item.trees["split"] if hasattr(item, "trees") else item)
            if getattr(analysis, "legacy_mixture", False)
            else analysis["saved_layouts"][population_id]
        )
        population = tree_population(tree_analysis, analysis, population_id)
        if item is None:
            item = SimpleNamespace(
                scalar_domain=(
                    float(population["scalar"].min()),
                    float(population["scalar"].max()),
                )
            )
        bins = compute_fitness_histogram_table(
            population.rename({"scalar": "raw_fitness"}),
            scalar_column="raw_fitness",
            bin_count=int(figure_settings.get("histogram_bins", 200)),
        )
        plot_data = plot_data_with_metrics(tree_analysis)
        if plot_data.edge_table.height:
            max_branch_count = max(
                max_branch_count, int(plot_data.edge_table["vertex_count"].max())
            )
        products[key] = (item, tree_analysis, population, bins, plot_data)

    # One count axis for every histogram panel: headroom over the tallest bin of any panel.
    count_limit = count_axis_limit(
        [int(products[key][3]["count"].max()) for key in products], rendering
    )
    # One colour scale for every panel: the union of the plotted branch counts.
    branch_domain = (1.0, float(max_branch_count))
    rows = []
    for key, title, population_id in panel_specs:
        item, tree_analysis, population, bins, plot_data = products[key]
        annotated, coloring, stops = False, "count", None
        tree_rendering = rendering
        if slide is not None:
            tree_rendering = {
                **rendering,
                **slide_stop_overrides(settings, key == "full"),
            }
            tree_analysis, annotated = slide_tree_analysis(tree_analysis, slide["style"])
            coloring = slide.get("coloring", "count")
        graph = _REGISTRY_GRAPH[key]
        subtitle = []
        threshold = None
        if slide is None:
            subtitle = [
                (
                    "Papkou et al. (2023) · "
                    f"{quantities.display(f'{graph}_graph.peak_count')} peaks"
                )
            ]
        if key == "full":
            if slide is None:
                subtitle.append(
                    f"{quantities.display('full_graph.vertex_count')} sequences"
                )
        else:
            threshold = _threshold_value(analysis, item, key)
            if slide is None:
                subtitle.append(
                    f"{quantities.display(f'{graph}_graph.functional_count')} "
                    "functional / "
                    f"{quantities.display(f'{graph}_graph.vertex_count')} total "
                    "sequences · "
                    + functional_threshold_text(
                        quantities.display(f"thresholds.{graph}_functional.value")
                    )
                )
        # The rule goes behind the branches and its label beneath the tree's markers
        # (the wild-type star), so the label never covers one: ``tree_chart`` stacks
        # both in its fixed order. The full-graph row has no threshold.
        underlays, labels = [], []
        if key != "full":
            # Only this panel's own threshold, a dashed rule with no text: its value
            # is an extra three-decimal tick on the fitness axis (see below).
            underlays = _threshold_layers(
                threshold, fitness_domain, rendering, labelled=False
            )
        tree = tree_chart(
            plot_data,
            tree_analysis,
            item,
            figure_settings,
            tree_rendering,
            branch_domain,
            linked=False,
            annotated=annotated,
            underlays=underlays,
            count_labels=labels,
            coloring=coloring,
            stop_identities=stops,
        )
        tree = apply_tree_histogram_domain(tree, fitness_domain)
        if key == "full" and coloring == "count":
            # One horizontal bar inside the top tree panel, top right; its range
            # is the union of the branch counts plotted in every panel.
            tree = overlay_count_colorbar(
                tree, figure_settings, rendering, branch_domain
            )
        if slide is None:
            tree = tree.properties(
                title=alt.TitleParams(text=title, subtitle=subtitle)
            )
        # None selects the shared count axis (labelled panel only).
        count_axis = (
            None
            if key == "candidate_clamped"
            else alt.Axis(
                title=None, labels=False, ticks=False, domain=False, grid=False
            )
        )
        histogram = histogram_chart(
            population,
            bins,
            fitness_domain,
            None,
            figure_settings,
            rendering,
            count_axis=count_axis,
            count_limit=count_limit,
        )
        if key != "full":
            # The histogram shares the tree's fitness axis, so the same rule lands
            # at the same value across it; only the tree carries the label.
            histogram = _with_histogram_threshold(
                histogram, threshold, fitness_domain, rendering
            )
        row = alt.hconcat(
            tree, histogram, spacing=int(rendering.get("panel_gap", 10))
        ).resolve_scale(x="independent", y="shared", color="independent")
        rows.append(
            publication_scalar_scales(
                row,
                fitness_domain,
                rendering,
                exact_domain=slide is None,
                extra_ticks=(
                    {} if threshold is None else {"y": [threshold], "bin_min": [threshold]}
                ),
            )
        )

    chart = alt.vconcat(
        *rows, spacing=int(rendering.get("comparison_panel_gap", 16))
    ).resolve_scale(x="independent", y="independent", color="independent")
    return tree_histogram_chart_config(chart, rendering)


# Landscape key to the registry's graph prefix.
_REGISTRY_GRAPH = {
    "full": "full",
    "study_clamped": "published",
    "candidate_clamped": "candidate",
}


def _threshold_value(analysis: dict[str, Any], item: Any, key: str) -> float:
    """The threshold the rule is drawn at (geometry only; the label is the registry's)."""
    threshold = analysis.get("thresholds", {}).get(key)
    if threshold is None:
        threshold = getattr(item, "config", {}).get("threshold")
    if threshold is None:
        raise ValueError(f"Landscape {key!r} has no declared threshold.")
    return float(threshold)


def _threshold_layers(
    threshold: float,
    fitness_domain: tuple[float, float],
    rendering: dict[str, Any],
    *,
    text: str | None = None,
    label_x: float | None = None,
    labelled: bool = True,
) -> list[Any]:
    """Return the rule and label of one panel's functional threshold.

    ``labelled=False`` returns the rule alone (a one-item list): the value is then
    named by an extra axis tick rather than by text.

    ``text`` and ``label_x`` replace the default label and its pixel column (the
    D10 comparison views name the selection and stagger the labels).

    The tree and its histogram both read their rule from here, so the two cannot
    differ in colour, dash, width or value. The fitness scale is the one display
    domain both panels share, and the layers state no axis: the panel they join
    owns it.
    """
    return functional_threshold_layers(
        threshold,
        scale=alt.Scale(domain=list(fitness_domain), nice=False, zero=False),
        axis=alt.Undefined,
        color=rendering.get("threshold_color", THRESHOLD_GREY),
        dash=[float(value) for value in rendering.get("threshold_dash", [5, 4])],
        width=float(rendering.get("threshold_width", 1.2)),
        font_size=int(rendering.get("threshold_label_font_size", 14)),
        label_x=threshold_label_padding(rendering) if label_x is None else label_x,
        label_align="left",
        label_below=True,
        **(
            {"label_text": None}
            if not labelled
            else {}
            if text is None
            else {"label_text": text, "label_backing": "white"}
        ),
    )


def _with_histogram_threshold(
    histogram: Any,
    threshold: float,
    fitness_domain: tuple[float, float],
    rendering: dict[str, Any],
) -> Any:
    """Continue the functional-threshold rule across a tree's histogram, unlabelled.

    The rule is the tree's own (same colour, dash and width) at the same fitness,
    drawn over the bars so it stays visible across them, and cropped at the panel.
    The histogram's fitness axis is the tree's, shared by the row's
    ``resolve_scale(y="shared")``, so the rule lands at the same value in both.
    """
    (rule,) = _threshold_layers(threshold, fitness_domain, rendering, labelled=False)
    rule.mark = alt.MarkDef(**{**rule.mark.to_dict(), "clip": True})
    return alt.layer(histogram, rule)
