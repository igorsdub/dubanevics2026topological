"""Published-threshold tree and persistence metric figure (branch count, AA Hamming, H-S order)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import altair as alt
import polars as pl
from topographer.plotting import (
    ColorScaleConfig,
    PersistencePlotConfig,
    TreePlotConfig,
    TreePresentationConfig,
    plot_persistence,
    plot_tree,
)

from ._branch_count_style import branch_count_color_scale
from ._charts import (
    annotation_legend,
    biology_axis_contract,
    compose_tree_layers,
    horizontal_count_legend,
    horizontal_metric_legend,
    landscape_population_subtitle,
    overlay_colorbar,
    plot_data_with_saved_metrics,
    publication_scalar_scales,
    set_count_point_opacity,
    persistence_chart,
    persistence_legend_layers,
    persistence_panel_size,
    wild_type_layers,
)
from .palette import (
    COUNT_BLACK,
    HORTON_STRAHLER_COLORS,
    PERSISTENCE_HEIGHT,
    PERSISTENCE_WIDTH,
    TREE_HEIGHT,
    TREE_WIDTH,
    tree_panel_size,
)


def build_study_tree_persistence_metrics_figure(
    analysis: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Build the three vanilla tree/persistence metric rows for the published functional threshold."""
    figure_id = "study_tree_persistence_metrics"
    figure_settings = settings["figures"]["figures"][figure_id]
    rendering = {
        **settings["figures"].get("rendering", {}),
        **figure_settings,
    }
    figure_font_size = int(
        figure_settings.get("font_size", rendering.get("font_size", 15))
    )
    rendering = {
        **rendering,
        "font_size": figure_font_size,
        "axis_label_font_size": figure_font_size,
        "legend_font_size": figure_font_size,
    }
    study = analysis["landscapes"]["study_clamped"]
    tree_analysis = study.trees["split"]
    support = analysis["tables"]
    branch_table = support["study_clamped_branch_table"]
    persistence_table = support["study_clamped_persistence_table"]
    plot_data = plot_data_with_saved_metrics(tree_analysis, branch_table)

    row_specs = (
        ("branch_vertex_count", "Branch vertex count", "count"),
        ("aa_hamming", "AA Hamming distance", "aa_hamming"),
        ("horton_strahler_order", "Horton–Strahler order", "horton_strahler_order"),
    )
    rows = []
    for metric, title, scale_name in row_specs:
        pairs = study_metric_persistence_table(
            persistence_table, branch_table, metric=metric
        )
        shared_values = pl.concat(
            [plot_data.edge_table[metric], pairs[metric]], how="vertical"
        )
        scale = study_metric_scale(rendering, scale_name, metric, title, shared_values)
        tree = _study_metric_tree_chart(
            plot_data,
            study,
            figure_settings,
            rendering,
            metric=metric,
            scale=scale,
            key=metric == "branch_vertex_count",
        )
        inset_rendering = {
            **rendering,
            "legend_font_size": int(figure_settings.get("inset_legend_font_size", 11)),
        }
        legend_rendering = {
            **inset_rendering,
            "legend_length": int(figure_settings.get("colorbar_length", 150)),
        }
        if metric == "branch_vertex_count":
            legend = horizontal_count_legend(
                {
                    **legend_rendering,
                    "branch_count_title": figure_settings.get(
                        "branch_count_title", rendering.get("branch_count_title")
                    ),
                },
                (2.0, max(2.001, float(shared_values.max() or 2))),
            )
        elif metric == "aa_hamming":
            legend = horizontal_metric_legend(
                legend_rendering,
                title=title,
                values=tuple(scale.categories),
                colors=tuple(scale.colors),
            )
        else:
            legend = horizontal_metric_legend(
                legend_rendering,
                title=title,
                values=(1, 2, 3, 4),
                colors=tuple(scale.colors),
            )
        tree = overlay_colorbar(
            tree,
            legend,
            x_fraction=float(figure_settings["colorbar_x_fraction"]),
            top_inset=float(rendering.get("colorbar_top_inset", 8)),
            width=tree_panel_size(rendering, figure_settings)[0],
        )
        first_row = metric == "branch_vertex_count"
        if first_row:
            count_pairs = pairs.rename({"branch_vertex_count": "vertex_count"})
            maximum = max(1, int(plot_data.edge_table["vertex_count"].max() or 1))
            persistence = biology_axis_contract(
                persistence_chart(count_pairs, figure_settings, rendering,
                                  (1.0, float(maximum)), study.scalar_domain),
                y_title=None,
                x_title=rendering.get("persistence_x_title", "Fitness at death"),
            )
        else:
            # The diagram's markers take the branch colours of their row.
            persistence = _study_metric_persistence_chart(
                pairs,
                study,
                figure_settings,
                rendering,
                metric=metric,
                title=title,
                scale=scale,
            )
        # Only the first diagram is keyed: "peak", inside it, bottom right.
        if first_row:
            persistence = alt.layer(
                persistence,
                *persistence_legend_layers(
                    *persistence_panel_size(figure_settings, rendering),
                    rendering,
                    stop_codon=False,
                    peak_color=COUNT_BLACK,
                ),
            )
        rows.append(
            alt.hconcat(
                tree,
                persistence,
                spacing=int(rendering.get("panel_gap", 10)),
            ).resolve_scale(x="independent", y="shared", color="independent")
        )

    chart = alt.vconcat(
        *rows, spacing=int(rendering.get("comparison_panel_gap", 16))
    ).resolve_scale(x="independent", y="independent", color="independent")
    study_subtitle = landscape_population_subtitle(
        analysis["manuscript_quantities"], "published"
    )
    chart = chart.properties(
        title=alt.TitleParams(
            text="Published graph",
            subtitle=study_subtitle,
        )
    )
    chart = publication_scalar_scales(chart, study.scalar_domain, rendering)
    chart = (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(rendering.get("axis_label_font_size", 15)),
            titleFontSize=int(rendering.get("font_size", 15)),
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
    return chart


def study_metric_scale(
    rendering: dict[str, Any],
    scale_name: str,
    metric: str,
    title: str,
    values: pl.Series,
) -> ColorScaleConfig:
    if scale_name == "count":
        maximum = max(1, int(values.max() or 1))
        return replace(
            branch_count_color_scale(
                rendering, (1.0, float(maximum)), attribute=metric
            ),
            title=title,
            legend_length=int(rendering.get("legend_length", 110)),
        )
    scheme = rendering.get("aa_hamming_color_scheme", "bluepurple")
    if scale_name == "aa_hamming":
        # Width-1 bins from the observed minimum to the observed maximum distance,
        # one scheme colour per integer distance, like the Horton-Strahler bar.
        categories = hamming_bins(values)
        colors = discrete_scheme_colors(scheme, len(categories))
    else:
        categories = (1, 2, 3, 4)
        colors = tuple(rendering.get("horton_strahler_colors", HORTON_STRAHLER_COLORS))
    return ColorScaleConfig(
        attribute=metric,
        title=title,
        scheme=scheme,
        scale_type="ordinal",
        domain=None,
        categories=categories,
        colors=colors,
        show_legend=False,
        legend_length=int(rendering.get("legend_length", 110)),
    )


def hamming_bins(values: pl.Series) -> tuple[int, ...]:
    """Integer Hamming distances from the observed minimum to the observed maximum."""
    if values.is_empty():
        raise ValueError("Hamming distances are needed to build the colour bar.")
    return tuple(range(int(values.min()), int(values.max()) + 1))


# Vega's continuous "bluepurple" scheme is the middle of d3's ``interpolateBuPu``:
# the nine ColorBrewer BuPu colours blended by a uniform cubic B-spline in RGB,
# of which Vega draws the span 0.2 to 0.9 (it leaves out the near-white start and
# the near-black end). Altair knows the scheme by name, but the tree and the
# colour bar take explicit colours, so the figure samples it here the way Vega
# does, to within one RGB level.
_VEGA_SCHEMES: dict[str, tuple[tuple[int, ...], tuple[float, float]]] = {
    # d3-scale-chromatic's nine-class ColorBrewer BuPu, as 24-bit RGB integers.
    "bluepurple": (
        (
            0xF7FCFD,
            0xE0ECF4,
            0xBFD3E6,
            0x9EBCDA,
            0x8C96C6,
            0x8C6BB1,
            0x88419D,
            0x810F7C,
            0x4D004B,
        ),
        (0.2, 0.9),
    ),
}


def discrete_scheme_colors(scheme: str, count: int) -> tuple[str, ...]:
    """Return ``count`` colours, one per discrete bin, from a named colour scheme.

    A Vega scheme in ``_VEGA_SCHEMES`` is sampled as Vega samples a continuous
    scheme for an ordinal scale, at ``i / (count + 1)`` for ``i = 1..count`` of
    its span, so neither end of the scheme is used. Any other name is a
    Matplotlib colormap sampled evenly from its first to its last colour.
    """
    if scheme in _VEGA_SCHEMES:
        ramp, (start, stop) = _VEGA_SCHEMES[scheme]
        return tuple(
            _vega_ramp_color(ramp, start + (stop - start) * (index + 1) / (count + 1))
            for index in range(count)
        )
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    colormap = colormaps[scheme]
    fractions = [0.5] if count == 1 else [i / (count - 1) for i in range(count)]
    return tuple(to_hex(colormap(fraction)) for fraction in fractions)


def _vega_ramp_color(ramp: tuple[int, ...], fraction: float) -> str:
    """Colour at ``fraction`` of a d3 ``interpolateRgbBasis`` ramp, as ``#rrggbb``.

    ``ramp`` holds the colours as 24-bit integers, red in the high byte.
    """
    channels = [[(code >> shift) & 0xFF for code in ramp] for shift in (16, 8, 0)]
    # JavaScript's Math.round (half up), as d3 formats the blended colour.
    return "#" + "".join(
        f"{max(0, min(255, math.floor(_basis_spline(values, fraction) + 0.5))):02x}"
        for values in channels
    )


def _basis_spline(values: list[int], t: float) -> float:
    """Uniform cubic B-spline through ``values`` at ``t`` in [0, 1] (d3 ``basis``)."""
    last = len(values) - 1
    if t <= 0:
        t, i = 0.0, 0
    elif t >= 1:
        t, i = 1.0, last - 1
    else:
        i = int(t * last)
    v1, v2 = values[i], values[i + 1]
    v0 = values[i - 1] if i > 0 else 2 * v1 - v2
    v3 = values[i + 2] if i < last - 1 else 2 * v2 - v1
    t1 = (t - i / last) * last
    t2, t3 = t1 * t1, t1 * t1 * t1
    return (
        (1 - 3 * t1 + 3 * t2 - t3) * v0
        + (4 - 6 * t2 + 3 * t3) * v1
        + (1 + 3 * t1 + 3 * t2 - 3 * t3) * v2
        + t3 * v3
    ) / 6


def _study_metric_tree_chart(
    plot_data: Any,
    analysis: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    *,
    metric: str,
    scale: ColorScaleConfig,
    key: bool = False,
) -> Any:
    """Draw one metric-coloured tree; ``key`` adds the one-line wild-type key."""
    appearance = figure_settings.get("tree", {})
    config = TreePlotConfig(
        width=int(appearance.get("width", rendering.get("tree_width", TREE_WIDTH))),
        height=int(appearance.get("height", rendering.get("tree_height", TREE_HEIGHT))),
        font_size=int(appearance.get("font_size", rendering.get("font_size", 15))),
        scalar_axis_title=rendering.get(
            "shared_tree_axis_title", "Fitness (at birth, WT = 0)"
        ),
        scalar_domain=analysis.scalar_domain,
        edge_width=float(rendering.get("edge_width", 1.1)),
        node_size=int(rendering.get("node_size", 24)),
        edge_color_scale=scale,
    )
    # Ascending by the colour attribute so higher values draw last, on top.
    plot_data = replace(
        plot_data, edge_table=plot_data.edge_table.sort(metric, maintain_order=True)
    )
    chart = plot_tree(
        plot_data,
        config=config,
        presentation=TreePresentationConfig(mode="metric"),
        tooltip_attrs=("dna_sequence", "aa_sequence"),
        edge_tooltip_attrs=(metric, "source_label", "target_label"),
        show_structural_events=False,
        interactive=False,
        include_config=False,
    )
    wild_type = wild_type_layers(analysis.trees["split"], rendering)
    width, height = tree_panel_size(rendering, figure_settings)
    keys = (
        [annotation_legend(width, height, rendering, wild_type=True, stop_codon=False)]
        if key and wild_type
        else []
    )
    return compose_tree_layers(chart, keys=keys, wild_type=wild_type)


def study_metric_persistence_table(
    persistence_table: pl.DataFrame,
    branch_table: pl.DataFrame,
    *,
    metric: str,
) -> pl.DataFrame:
    """Attach pair metrics, deriving branch count from the branch below birth."""
    required = {"birth_label", "death_label", "birth_scalar", "death_scalar"}
    if metric != "branch_vertex_count":
        required.add(metric)
    missing = required - set(persistence_table.columns)
    if missing:
        raise ValueError(
            f"Saved persistence table is missing fields: {sorted(missing)}"
        )
    pairs = persistence_table
    if metric == "branch_vertex_count":
        branches = branch_table.select(
            "source_tree_node_identity", "branch_vertex_count"
        )
        if branches["source_tree_node_identity"].n_unique() != branches.height:
            raise ValueError(
                "Duplicate birth endpoints make pair branch counts ambiguous."
            )
        counts = {
            str(row["source_tree_node_identity"]): int(row["branch_vertex_count"])
            for row in branches.to_dicts()
        }
        values = []
        for row in pairs.select("birth_label").to_dicts():
            identity = str(row["birth_label"])
            if identity not in counts:
                raise ValueError(
                    f"Persistence birth {identity!r} has no branch below it."
                )
            values.append(counts[identity])
        return pairs.with_columns(pl.Series(metric, values, dtype=pl.Int64))
    if pairs[metric].null_count():
        raise ValueError(
            f"Saved persistence metric {metric!r} contains missing values."
        )
    return pairs


def _study_metric_persistence_chart(
    pairs: pl.DataFrame,
    analysis: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    *,
    metric: str,
    title: str,
    scale: ColorScaleConfig,
) -> Any:
    appearance = figure_settings.get("persistence", {})
    config = PersistencePlotConfig(
        width=int(
            appearance.get(
                "width", rendering.get("persistence_width", PERSISTENCE_WIDTH)
            )
        ),
        height=int(
            appearance.get(
                "height", rendering.get("persistence_height", PERSISTENCE_HEIGHT)
            )
        ),
        font_size=int(appearance.get("font_size", rendering.get("font_size", 15))),
        birth_axis_title=rendering.get(
            "persistence_y_title", "Fitness (at birth, WT = 0)"
        ),
        death_axis_title=rendering.get("persistence_x_title", "Fitness at death"),
        domain=analysis.scalar_domain,
        point_size=int(
            appearance.get("point_size", rendering.get("persistence_point_size", 70))
        ),
        show_grid=bool(rendering.get("persistence_grid", False)),
        color_scale=scale,
    )
    chart = plot_persistence(
        pairs,
        config=config,
        color_by=metric,
        draw_order_by=metric,
        shape_by=None,
        separate_event_layers=False,
        tooltip_attrs=("persistence", "birth_label", "death_label", metric),
        include_config=False,
    )
    chart = set_count_point_opacity(
        chart,
        float(rendering.get("count_point_opacity", 1.0)),
        shape=rendering.get("persistence_point_shape", "triangle-up"),
        color_field=metric,
    )
    return biology_axis_contract(
        chart,
        y_title=None,
        x_title=rendering.get("persistence_x_title", "Fitness at death"),
    )
