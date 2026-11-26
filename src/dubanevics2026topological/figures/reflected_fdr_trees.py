"""Two saved median-reflected nominal trees with raw-fitness histograms."""

from types import SimpleNamespace

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
)
from ._top_histogram import count_axis_limit
from .palette import THRESHOLD_GREY

ROWS = (
    ("reflected", "Candidate threshold point estimate"),
    ("reflected_by", "Candidate threshold"),
)


def build_reflected_fdr_trees_figure(payload, settings):
    """Compose the point-estimate row above the BY row from saved inputs."""
    # Reuse the existing comparison's appearance without changing presentation settings.
    figure_settings, rendering, domain = tree_histogram_settings(settings, "tree_comparison")
    prepared = []
    max_branches = 1
    for key, title in ROWS:
        row = payload["rows"][key]
        if key == "reflected_by":
            title = f"{title} {row['method'].upper()}, nominal q = {row['q']:g}"
        population, layout = row["population"], row["layout"]
        bins = compute_fitness_histogram_table(
            population.rename({"scalar": "raw_fitness"}), scalar_column="raw_fitness",
            bin_count=int(figure_settings.get("histogram_bins", 200)),
        )
        plot = plot_data_with_metrics(layout)
        if plot.edge_table.height:
            max_branches = max(max_branches, int(plot.edge_table["vertex_count"].max()))
        prepared.append((title, row, bins, plot))
    branch_domain = (1.0, float(max_branches))
    count_limit = count_axis_limit([int(p[2]["count"].max()) for p in prepared], rendering)
    charts = []
    for index, (title, row, bins, plot) in enumerate(prepared):
        threshold = row["threshold"]
        # A dashed rule, no text: the value is an extra three-decimal fitness tick.
        (rule,) = functional_threshold_layers(
            threshold, scale=alt.Scale(domain=list(domain), nice=False, zero=False),
            axis=alt.Undefined, color=rendering.get("threshold_color", THRESHOLD_GREY),
            dash=rendering.get("threshold_dash", [5, 4]),
            width=float(rendering.get("threshold_width", 1.2)),
            font_size=int(rendering.get("threshold_label_font_size", 14)),
            label_x=threshold_label_padding(rendering), label_align="left", label_below=True,
            label_text=None,
        )
        population = row["population"]
        item = SimpleNamespace(scalar_domain=(float(population["scalar"].min()), float(population["scalar"].max())))
        tree = tree_chart(plot, row["layout"], item, figure_settings, rendering,
                          branch_domain, linked=False, annotated=False,
                          underlays=[rule], count_labels=[], coloring="count")
        tree = apply_tree_histogram_domain(tree, domain)
        if index == 0:
            tree = overlay_count_colorbar(tree, figure_settings, rendering, branch_domain)
        tree = tree.properties(title=alt.TitleParams(text=title, subtitle=[functional_threshold_text(f"{threshold:.3f}")]))
        axis = None if index == 1 else alt.Axis(title=None, labels=False, ticks=False, domain=False, grid=False)
        histogram = histogram_chart(population, bins, domain, None,
                                    figure_settings, rendering, count_axis=axis,
                                    count_limit=count_limit)
        (histogram_rule,) = functional_threshold_layers(
            threshold, scale=alt.Scale(domain=list(domain), nice=False, zero=False),
            axis=alt.Undefined, color=rendering.get("threshold_color", THRESHOLD_GREY),
            dash=rendering.get("threshold_dash", [5, 4]),
            width=float(rendering.get("threshold_width", 1.2)),
            font_size=int(rendering.get("threshold_label_font_size", 14)),
            label_x=threshold_label_padding(rendering), label_align="left", label_below=True,
            label_text=None,
        )
        histogram_rule.mark = alt.MarkDef(**{**histogram_rule.mark.to_dict(), "clip": True})
        histogram = alt.layer(histogram, histogram_rule)
        row_chart = alt.hconcat(tree, histogram, spacing=int(rendering.get("panel_gap", 10))).resolve_scale(x="independent", y="shared", color="independent")
        charts.append(publication_scalar_scales(
            row_chart, domain, rendering, exact_domain=True,
            extra_ticks={"y": [float(threshold)], "bin_min": [float(threshold)]},
        ))
    chart = alt.vconcat(*charts, spacing=int(rendering.get("comparison_panel_gap", 16))).resolve_scale(x="independent", y="independent", color="independent")
    return tree_histogram_chart_config(chart, rendering)