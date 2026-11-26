"""Fitness against pooled standard error for the four graphs of the peak analysis, 2×2.

Columns are the graph, rows the selection: Papkou graph left of Topographer graph,
nominal (point estimate, no FDR) on top and FDR-corrected (Benjamini-Yekutieli, q from
the saved product, 0.05) below. Each panel is the selected graph of
its own analysis variant (``point_papkou_original``, ``fdr_by_q0.05_papkou_original``,
``point_median_reflected``, ``fdr_by_q0.05_median_reflected``; the variants of
``bootstrap_four_graphs``), read from that variant's saved ``pooled_scatter`` and
``peak_panel_summary`` products. The scatter table gives every measured sequence and the
wild type with their ``is_peak`` flag; the panel's threshold, peak count, SEQUENCE counts
(functional, uncertain where the variant has the class, non-functional) and largest
component size are the product's diagnostics, computed in the product stage and printed
as given (never recomputed here, never read from the manuscript quantity registry).

Every panel: y = fitness relative to WT (ticks: the domain's two ends, 0 and the panel's
own threshold), x = pooled standard error, the threshold as a dashed unlabelled rule.
The wild type is the repository's red star, drawn last on top; peaks are black triangles.
The non-peak cloud is a matplotlib raster placed as a Vega-Lite image on the same scales,
so the inline row budget stays on the vector marks. Y axis titles only on the left column,
x axis titles only on the bottom row; each column is titled with its graph and each panel with its selection. One legend,
in the top-right panel (Topographer graph, no FDR). Scales are shared across all four panels.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ._charts import (
    functional_threshold_layers,
    marker_config,
)
from ._count_labels import count_label_text, fixed_label_text, general_label_text
from .labels import PAPKOU, TOPOGRAPHER, state_label
from .palette import EMPHASIS_BLACK, LIGHT_GREY, MEDIAN_GREY

logger = logging.getLogger(__name__)

FIGURE_ID = "peak_uncertainty"
# The same four variants as ``bootstrap_four_graphs.PANEL_ORDER``; ``build/peak_panels.py``
# declares what each one selects and a test keeps the two lists equal. Display order is
# separate: columns are the graph, rows nominal then FDR-corrected.
PANEL_VARIANTS = (
    "threshold=published_fdr=none",
    "threshold=reflected_fdr=none",
    "threshold=published_fdr=by_q=0.05",
    "threshold=reflected_fdr=by_q=0.05",
)
# Columns are the graph (Papkou | Topographer), rows the selection (no FDR above BY FDR).
PANEL_GRID = (
    ("threshold=published_fdr=none", "threshold=reflected_fdr=none"),
    ("threshold=published_fdr=by_q=0.05", "threshold=reflected_fdr=by_q=0.05"),
)
# The scatter table of the panel's own graph inside the variant's pooled_scatter product.
SCATTER_TABLE = "candidate_clamped_scatter"


def panel_state(variant: str) -> str:
    """The graph state of a panel variant, for its strict display label."""
    base = "published" if "threshold=published" in variant else "reflected"
    return f"{base}_by" if "fdr=by" in variant else base


def threshold_name(variant: str) -> str:
    """``Papkou`` for a Papkou-cutoff variant, ``Topographer`` for the median-reflected one."""
    if variant not in PANEL_VARIANTS:
        raise ValueError(f"Unknown peak uncertainty panel variant {variant!r}.")
    return PAPKOU if "threshold=published" in variant else TOPOGRAPHER


def row_label(diagnostics: dict[str, Any]) -> str:
    """``Nominal`` for a point estimate, else the method and q of the FDR selection."""
    method = diagnostics.get("fdr_method")
    if method is None:
        return "Nominal"
    return f"{str(method).upper()} q = {general_label_text(diagnostics['fdr_q'])}"


def _panel_subtitle(diagnostics: dict[str, Any]) -> list[str]:
    """Subtitle lines of one panel, every number from the product's diagnostics.

    The FDR setting and the peak count, the functional threshold, then the sequence counts
    (SEQUENCES, not peaks; the Uncertain class only where the variant has it) and the
    largest component. The counts are printed through ``count_label_text``.
    """
    lines = [
        f"{count_label_text(diagnostics['peak_count'])} peaks",
        (
            "Papkou et al. (2023) · Functional threshold = "
            f"{fixed_label_text(diagnostics['threshold'], 3)}"
        ),
        f"{count_label_text(diagnostics['functional_sequence_count'])} functional sequences",
    ]
    uncertain = diagnostics.get("uncertain_sequence_count")
    if uncertain is not None:
        lines.append(f"{count_label_text(uncertain)} uncertain sequences")
    lines.append(
        f"{count_label_text(diagnostics['nonfunctional_sequence_count'])} non-functional sequences"
    )
    lines.append(
        "Largest component: "
        f"{count_label_text(diagnostics['largest_component_vertex_count'])} sequences"
    )
    return lines


def build_peak_uncertainty_figure(
    result: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Build the 2×2 figure (graphs as columns, FDR settings as rows) from ``result["panels"]``, one payload per analysis variant.

    ``result["panels"]`` maps each of ``PANEL_VARIANTS`` to a payload with a
    ``scatter`` DataFrame and a ``diagnostics`` dict (the variant's saved products). A missing variant is an
    error: no panel stands in for another. Columns are the Papkou and Topographer graphs
    (``PANEL_GRID``); rows are nominal and FDR-corrected. Shared scales, Y axis only
    on the left column, one legend in the top-right panel, a dashed unlabelled rule per panel.
    """
    figure_settings = settings["figures"]["figures"][FIGURE_ID]
    rendering = settings["figures"].get("rendering", {})
    appearance = {**rendering, **figure_settings}

    panels = result.get("panels") or {}
    missing = [variant for variant in PANEL_VARIANTS if variant not in panels]
    if missing:
        raise ValueError(f"Peak uncertainty payload lacks the variants {missing}.")

    # Shared ranges are figure settings, the same on every panel.
    fitness_domain = _configured_domain(appearance, "fitness_domain")
    se_domain = _configured_domain(appearance, "standard_error_domain")
    # Three fitness ticks on every panel: the domain's two ends and 0.
    fitness_ticks = _end_ticks(
        fitness_domain, float(appearance.get("fitness_tick_step", 0.1))
    )

    gap = int(appearance.get("panel_gap", 30))
    row_gap = int(appearance.get("row_gap", gap))
    datasets: dict[str, Any] = {}
    rows = len(PANEL_GRID)
    legend_row_idx = 0  # top row (no FDR)
    legend_col_idx = 1  # right column (Topographer graph)
    row_labels: list[str] = []
    columns_graph: list[set[str]] = [set() for _ in PANEL_GRID[0]]
    cells: list[list[Any]] = []

    for row_idx, row_variants in enumerate(PANEL_GRID):
        row_diagnostics = [panels[variant]["diagnostics"] for variant in row_variants]
        labels = [row_label(diagnostics) for diagnostics in row_diagnostics]
        titles = [state_label(panel_state(variant)) for variant in row_variants]
        if len(set(labels)) != 1:
            raise ValueError(f"Panels of one row disagree on the selection: {labels}.")
        row_labels.append(labels[0])
        nominal = row_diagnostics[0].get("fdr_method") is None
        if (row_idx == 0) != nominal:
            raise ValueError(
                "The top row must be nominal (no FDR) and the bottom row FDR-corrected, "
                f"found {labels[0]!r} in row {row_idx}."
            )
        subtitles = [_panel_subtitle(diagnostics) for diagnostics in row_diagnostics]
        # The panels of a row get the same number of lines, so their plots stay level.
        longest = max(len(lines) for lines in subtitles)
        for lines in subtitles:
            lines.extend(["\u00a0"] * (longest - len(lines)))
        row_cells = []
        for col_idx, variant in enumerate(row_variants):
            diagnostics = row_diagnostics[col_idx]
            columns_graph[col_idx].add(threshold_name(variant))
            scatter = _frame(panels[variant], "scatter")
            wild_type = diagnostics.get("wild_type_sequence")
            _warn_outside_domain(
                scatter,
                fitness_domain=fitness_domain,
                se_domain=se_domain,
                where=f"{threshold_name(variant)} graph / {labels[col_idx]}",
            )
            data_name = f"{variant}_measurements"
            # Inline rows are the vector marks only (peaks and the wild type).
            # The non-peak cloud is the image layer, not another copy of the table.
            datasets[data_name] = _vector_rows(scatter, wild_type).to_dicts()
            row_cells.append(
                _scatter_panel(
                    scatter,
                    data_name=data_name,
                    threshold=float(diagnostics["threshold"]),
                    wild_type=wild_type,
                    fitness_domain=fitness_domain,
                    fitness_ticks=fitness_ticks,
                    se_domain=se_domain,
                    appearance=appearance,
                    label=titles[col_idx],
                    subtitle=subtitles[col_idx],
                    heading_font_size=int(appearance.get("heading_font_size", 18)),
                    subtitle_font_size=int(appearance.get("subtitle_font_size", 14)),
                    show_column_title=True,
                    show_x_labels=(row_idx == rows - 1),
                    show_y_labels=(col_idx == 0),
                    show_legend=(row_idx == legend_row_idx and col_idx == legend_col_idx),
                )
            )
        cells.append(row_cells)

    column_charts = []
    for col_idx, graphs in enumerate(columns_graph):
        if len(graphs) != 1:
            raise ValueError(f"Panels of one column are not the same graph: {sorted(graphs)}.")
        (_graph,) = graphs  # one graph per column
        column_chart = alt.vconcat(
            *[row_cells[col_idx] for row_cells in cells], spacing=row_gap
        ).resolve_scale(x="shared", y="shared")
        column_charts.append(column_chart)

    # Each panel draws its own y axis (the right column's carries its threshold tick).
    chart = (
        alt.hconcat(*column_charts, spacing=gap)
        .resolve_axis(y="independent")
        .resolve_scale(x="shared", y="shared")
    )
    chart.datasets = datasets
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(appearance.get("font_size", 15)),
            titleFontSize=int(appearance.get("font_size", 15)),
            labelPadding=int(appearance.get("axis_label_padding", 4)),
            titlePadding=int(appearance.get("axis_title_padding", 8)),
            grid=bool(appearance.get("grid", False)),
            domain=False,
            ticks=True,
        )
        .configure_title(fontSize=int(appearance.get("font_size", 15)), anchor="start")
        .configure_view(strokeWidth=0)
    )


def _end_ticks(domain: list[float], step: float) -> list[float]:
    """The domain's two ends, rounded outward to ``step``, and 0."""
    precision = max(0, math.ceil(-math.log10(step)))
    lower = round(math.floor(domain[0] / step + 1e-9) * step, precision)
    upper = round(math.ceil(domain[1] / step - 1e-9) * step, precision)
    return sorted({lower, 0.0, upper})


def _frame(panel: dict[str, Any], name: str) -> pl.DataFrame:
    try:
        frame = panel[name]
    except KeyError as error:
        raise KeyError(f"Peak uncertainty panel is missing {name!r}.") from error
    if not isinstance(frame, pl.DataFrame):
        frame = pl.DataFrame(frame)
    return frame


def _outside_domain_counts(
    frame: pl.DataFrame,
    *,
    fitness_domain: list[float],
    se_domain: list[float],
) -> tuple[int, int]:
    """How many rows fall outside the configured fitness and SE ranges.

    A point on the boundary is inside. The raster clamps anything past the
    range onto the frame, so these counts are the stripes that clamp would draw.
    """
    fitness = frame["raw_fitness"]
    error = frame["standard_error"]
    outside_fitness = int(
        ((fitness < fitness_domain[0]) | (fitness > fitness_domain[1])).sum()
    )
    outside_se = int(((error < se_domain[0]) | (error > se_domain[1])).sum())
    return outside_fitness, outside_se


def _warn_outside_domain(
    frame: pl.DataFrame,
    *,
    fitness_domain: list[float],
    se_domain: list[float],
    where: str,
) -> tuple[int, int]:
    outside_fitness, outside_se = _outside_domain_counts(
        frame, fitness_domain=fitness_domain, se_domain=se_domain
    )
    if outside_fitness or outside_se:
        logger.warning(
            "%s: %d %s outside the configured fitness domain %s, "
            "%d %s outside the configured SE domain %s",
            where,
            outside_fitness,
            "point" if outside_fitness == 1 else "points",
            fitness_domain,
            outside_se,
            "point" if outside_se == 1 else "points",
            se_domain,
        )
    return outside_fitness, outside_se


def _configured_domain(appearance: dict[str, Any], key: str) -> list[float]:
    """A two-number axis range from the figure table."""
    domain = appearance.get(key)
    if not isinstance(domain, (list, tuple)) or len(domain) != 2:
        raise ValueError(f"Figure settings must include {key} as [minimum, maximum].")
    lower, upper = float(domain[0]), float(domain[1])
    if upper <= lower:
        raise ValueError(f"{key} must run from a lower value to a higher one.")
    return [lower, upper]


def _ticks(domain: list[float], step: float) -> list[float]:
    """Tick positions on ``domain`` at multiples of ``step``."""
    if step <= 0:
        raise ValueError("Tick step must be positive.")
    precision = max(0, math.ceil(-math.log10(step)))
    start = math.ceil(domain[0] / step - 1e-9) * step
    stop = math.floor(domain[1] / step + 1e-9) * step
    ticks: list[float] = []
    value = start
    while value <= stop + step * 1e-6:
        ticks.append(round(value, precision))
        value += step
    return ticks


def _scatter_panel(
    frame: pl.DataFrame,
    *,
    data_name: str,
    threshold: float,
    wild_type: str | None = None,
    fitness_domain: list[float],
    fitness_ticks: list[float],
    se_domain: list[float],
    appearance: dict[str, Any],
    label: str,
    heading_font_size: int,
    subtitle_font_size: int,
    show_column_title: bool,
    show_x_labels: bool,
    show_y_labels: bool,
    show_legend: bool = False,
    subtitle: list[str],
) -> Any:
    required = {
        "dna_sequence",
        "aa_sequence",
        "raw_fitness",
        "standard_error",
        "is_peak",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scatter table is missing columns: {sorted(missing)}.")
    data = alt.NamedData(name=data_name)
    x_title = appearance.get("standard_error_title", "Pooled fitness SE")
    y_title = appearance.get("fitness_axis_title", "Fitness (WT = 0)")
    # One switch, ``grid`` in the figure's table (false: ticks only, no grid lines),
    # for the chart-wide default and for every axis.
    grid = bool(appearance.get("grid", False))
    se_step = float(
        appearance.get(
            "standard_error_tick_step", appearance.get("se_tick_step", 0.04)
        )
    )
    se_precision = max(0, math.ceil(-math.log10(se_step)))
    se_min = round(math.floor(se_domain[0] / se_step) * se_step, se_precision)
    se_max = round(math.ceil(se_domain[1] / se_step) * se_step, se_precision)
    se_ticks = [
        float(round(t, se_precision))
        for t in np.arange(se_min, se_max + se_step / 2, se_step)
        if se_domain[0] <= t <= se_domain[1]
    ]
    x_axis = alt.Axis(
        title=x_title if show_x_labels else None,
        values=se_ticks,
        format=f".{se_precision}f",
        labels=show_x_labels,
        ticks=True,
        domain=False,
        grid=grid,
    )
    # Three fitness ticks (the domain's two ends and 0, one decimal) and this panel's
    # threshold as a three-decimal tick. The right column keeps only its threshold
    # tick, so each panel names the one threshold it uses.
    threshold_tick = float(threshold)
    y_values = sorted({*fitness_ticks, threshold_tick}) if show_y_labels else [threshold_tick]
    y_axis = alt.Axis(
        title=y_title if show_y_labels else None,
        values=y_values,
        labels=True,
        labelExpr=(
            f"format(datum.value, abs(datum.value - ({threshold_tick!r})) < 1e-9 "
            "? '.3f' : '.1f')"
        ),
        ticks=True,
        domain=False,
        grid=grid,
    )
    x_scale = alt.Scale(domain=se_domain, nice=False, zero=False)
    y_scale = alt.Scale(domain=fitness_domain, nice=False, zero=False)
    common = {
        "x": alt.X("standard_error:Q", title=x_title, axis=x_axis, scale=x_scale),
        "y": alt.Y("raw_fitness:Q", title=y_title, axis=y_axis, scale=y_scale),
        "tooltip": [
            alt.Tooltip("dna_sequence:N", title="DNA"),
            alt.Tooltip("aa_sequence:N", title="AA"),
            alt.Tooltip("raw_fitness:Q", title=y_title, format=".6f"),
            alt.Tooltip("standard_error:Q", title=x_title, format=".6f"),
            alt.Tooltip("is_peak:N", title="Raw-fitness peak"),
        ],
    }
    # The wild type is drawn once, as the star on top; the cloud and the triangles
    # leave it out. The cloud is rasterized so Vega never receives those rows.
    not_wild_type = (
        "true"
        if wild_type is None
        else f"datum.dna_sequence != {json.dumps(wild_type)}"
    )
    panel_size = int(appearance.get("panel_size", 300))
    cloud_url = _raster_cloud(
        _cloud_rows(frame, wild_type),
        se_domain=se_domain,
        fitness_domain=fitness_domain,
        size=panel_size,
        color=str(appearance.get("sequence_color", LIGHT_GREY)),
        opacity=float(appearance.get("sequence_opacity", 0.65)),
        point_size=float(appearance.get("sequence_point_size", 12)),
    )
    # One row at the domain's top-left. Same fields and scales as the vector
    # marks, so the image covers the panel and the axes stay a single pair.
    non_peaks = (
        alt.Chart(
            alt.Data(
                values=[
                    {
                        "standard_error": se_domain[0],
                        "raw_fitness": fitness_domain[1],
                        "url": cloud_url,
                    }
                ]
            )
        )
        .mark_image(
            width=panel_size,
            height=panel_size,
            align="left",
            baseline="top",
            aspect=False,
        )
        .encode(
            x=alt.X("standard_error:Q", title=x_title, axis=x_axis, scale=x_scale),
            y=alt.Y("raw_fitness:Q", title=y_title, axis=y_axis, scale=y_scale),
            url="url:N",
        )
    )
    peaks = (
        alt.Chart(data)
        .transform_filter(f"datum.is_peak == true && {not_wild_type}")
        .mark_point(
            shape="triangle-up",
            filled=True,
            size=int(appearance.get("peak_point_size", 112)),
            color=appearance.get("peak_color", EMPHASIS_BLACK),
            opacity=float(appearance.get("peak_opacity", 1.0)),
            strokeWidth=0,
        )
        .encode(**common)
    )
    (threshold_rule,) = functional_threshold_layers(
        threshold,
        scale=y_scale,
        axis=y_axis,
        color=appearance.get("threshold_color", MEDIAN_GREY),
        dash=[float(value) for value in appearance.get("threshold_dash", [5, 4])],
        width=float(appearance.get("threshold_width", 1.2)),
        # No label is drawn (``label_text=None``), so these two are unused.
        font_size=0,
        label_x=0,
        label_text=None,
    )
    # Peaks sit on the topmost data layer, above the threshold rule.
    layers = [non_peaks, threshold_rule]
    layers.append(peaks)
    if show_legend:
        layers.extend(_legend_layers(appearance))
    if wild_type is not None and wild_type in set(frame["dna_sequence"].to_list()):
        # The topmost layer of the panel, in the style of the trees' wild-type star.
        star = marker_config(appearance, "wild_type", default_shape="star")
        layers.append(
            alt.Chart(data)
            .transform_filter(f"datum.dna_sequence == {json.dumps(wild_type)}")
            .mark_point(
                shape=star.shape,
                color=star.color,
                size=star.size,
                filled=True,
                opacity=star.opacity,
            )
            .encode(**common)
        )
    panel = alt.layer(*layers).properties(
        width=panel_size,
        height=int(appearance.get("panel_size", 300)),
        title=alt.TitleParams(
            text=label if show_column_title else "",
            subtitle=subtitle,
            fontSize=heading_font_size if show_column_title else 0,
            subtitleFontSize=subtitle_font_size,
            fontWeight="bold",
            anchor="start",
            offset=int(appearance.get("title_offset", 10)),
        ),
    )
    return panel


def _cloud_rows(frame: pl.DataFrame, wild_type: str | None) -> pl.DataFrame:
    """Non-peak sequences, excluding the wild type, in table order."""
    mask = ~pl.col("is_peak")
    if wild_type is not None:
        mask = mask & (pl.col("dna_sequence") != wild_type)
    return frame.filter(mask)


def _vector_rows(frame: pl.DataFrame, wild_type: str | None) -> pl.DataFrame:
    """Rows drawn as vector marks: peaks, plus the wild type when it is present."""
    mask = pl.col("is_peak")
    if wild_type is not None:
        mask = mask | (pl.col("dna_sequence") == wild_type)
    return frame.filter(mask)


def _raster_cloud(
    cloud: pl.DataFrame,
    *,
    se_domain: list[float],
    fitness_domain: list[float],
    size: int,
    color: str,
    opacity: float,
    point_size: float,
) -> str:
    """PNG of the non-peak cloud, one pixel per panel pixel, as a data URL.

    The image covers ``se_domain`` × ``fitness_domain`` exactly (left/bottom to
    right/top). Vega's symbol ``size`` is the bounding-square area, so a circle
    of that size has diameter ``sqrt(size)``. Matplotlib's scatter area is the
    disc, ``π r²``, and at 72 dpi one point is one pixel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 72
    figure = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    axes = figure.add_axes((0, 0, 1, 1))
    if cloud.height:
        radius = math.sqrt(point_size) / 2
        axes.scatter(
            cloud["standard_error"].to_numpy(),
            cloud["raw_fitness"].to_numpy(),
            s=math.pi * radius * radius,
            c=color,
            alpha=opacity,
            linewidths=0,
            edgecolors="none",
        )
    axes.set_xlim(se_domain[0], se_domain[1])
    axes.set_ylim(fitness_domain[0], fitness_domain[1])
    axes.axis("off")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, transparent=True, pad_inches=0)
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _legend_font_size(appearance: dict[str, Any]) -> int:
    """The legend uses the panel title's font size (``heading_font_size``)."""
    return int(appearance.get("heading_font_size", 18))


def _legend_layers(appearance: dict[str, Any]) -> list[Any]:
    """Legend in the top-right panel's upper-right corner: Peak above Not peak.

    Each glyph repeats the plotted mark's own style (colour, opacity, shape, size).
    The block stands ``legend_padding`` pixels from the panel's top and right
    edges. Both glyphs stand in one column, left of the longest label, and both
    labels start in one column. The text is the title's font size, and
    ``legend_char_width`` is the average glyph width as a fraction of it.
    """
    size = int(appearance.get("panel_size", 300))
    font_size = _legend_font_size(appearance)
    char_width = float(appearance.get("legend_char_width", 0.6))
    marker_gap = float(appearance.get("legend_marker_gap", 10))
    row_gap = float(appearance.get("legend_row_gap", 2))
    padding = float(appearance.get("legend_padding", 8))
    marker_size = int(appearance.get("peak_point_size", 112))
    star = marker_config(appearance, "wild_type", default_shape="star")
    entries = (
        (
            "Peak",
            "triangle-up",
            appearance.get("peak_color", EMPHASIS_BLACK),
            float(appearance.get("peak_opacity", 1.0)),
        ),
        (
            "Not peak",
            "circle",
            appearance.get("sequence_color", LIGHT_GREY),
            float(appearance.get("sequence_opacity", 0.65)),
        ),
        # The wild type repeats its plotted star (the trees' red star).
        (
            "Wild type",
            star.shape,
            star.color,
            star.opacity,
        ),
    )
    text_x = (
        size
        - padding
        - max(len(label) for label, *_ in entries) * font_size * char_width
    )
    marker_x = text_x - marker_gap
    layers: list[Any] = []
    for index, (label, shape, color, opacity) in enumerate(entries):
        # Row 0 is the top one; y counts pixels down from the plot's top edge.
        y = padding + font_size / 2 + index * (font_size + row_gap)
        data = alt.Data(values=[{"label": label}])
        layers.append(
            alt.Chart(data)
            .mark_point(
                shape=shape,
                filled=True,
                color=color,
                size=marker_size,
                opacity=opacity,
                strokeWidth=0,
            )
            .encode(x=alt.value(marker_x), y=alt.value(y))
        )
        layers.append(
            alt.Chart(data)
            .mark_text(align="left", baseline="middle", fontSize=font_size, dx=0)
            .encode(x=alt.value(text_x), y=alt.value(y), text="label:N")
        )
    return layers
