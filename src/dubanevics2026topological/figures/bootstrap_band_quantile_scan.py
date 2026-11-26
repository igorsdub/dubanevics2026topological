"""Survival curve of the nominal peak persistences with the bootstrap quantile range.

The bootstrapped persistence threshold is twice the radius, the band quantile of the
real draws' bottleneck distance to the nominal diagram. The step plot is the
survival curve of the nominal peaks: the threshold on x from 0, the number of peaks
with persistence strictly above it on y, read from ``survival.parquet`` (exact
steps, no bootstrap draws). The range of thresholds the bootstrap can produce,
from the threshold at band quantile 0 to the one at band quantile 1 (``scan.parquet``),
is shaded; the ticks along its top edge sit at the threshold of each configured
band quantile (monotone linear interpolation over the scan's strictly increasing
quantile grid, so no tick falls outside the shading) and a dashed rule marks the
stage's band quantile with its threshold and count. The panel size and every font
size are those of ``linked_tree_persistence`` (the shared
``[rendering]`` settings); the subtitle is a list of phrases, each wrapped on its
own to ``subtitle_wrap`` characters.

Every number the marker label and the subtitle print is the manuscript quantity
registry's display text (``analysis["manuscript_quantities"]``); the figure's own
floats only place the marks. The top-axis band-quantile ticks are scale furniture
and stay exempt.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ..analysis.manuscript_quantities import BOOTSTRAP, agree, value_text
from ._count_labels import fixed_label_text, general_label_text
from .candidate_tree_biology import biology_renderings, wrap_phrases
from .palette import BAND_GREY, EMPHASIS_BLACK, TREE_HEIGHT, TREE_WIDTH

FIGURE_ID = "bootstrap_band_quantile_scan"
GRID_FIGURE_ID = "bootstrap_band_quantile_scan_grid"


def top_axis_ticks(scan: pl.DataFrame, quantiles: list[float]) -> pl.DataFrame:
    """Return each requested band quantile and its bootstrapped threshold (twice the radius)."""
    q = scan["q"].to_numpy()
    c = scan["c"].to_numpy()
    if not np.all(np.diff(q) > 0):
        raise ValueError("The scan's q grid must be strictly increasing.")
    if np.any(np.diff(c) < 0):
        raise ValueError("The scan's c must be non-decreasing in q.")
    wanted = np.asarray(quantiles, dtype=float)
    if wanted.size and (wanted.min() < q[0] or wanted.max() > q[-1]):
        raise ValueError("Top-axis quantiles must lie within the scanned range.")
    return pl.DataFrame(
        {
            "q": wanted,
            "tau": 2.0 * np.interp(wanted, q, c),
            "label": [f"{value:g}" for value in wanted],
        }
    )


def scan_run(
    scan: pl.DataFrame,
    survival: pl.DataFrame | None,
    summary_settings: dict[str, Any],
    total: int,
    label: str | None = None,
) -> dict[str, Any]:
    """Collect one bootstrap run's tables and numbers for drawing (never printed).

    ``survival`` is only drawn by the single-run figure; the seed overlay passes
    ``None``.
    """
    quantile = float(summary_settings["band_quantile"])
    row = scan.filter(pl.col("q") == quantile)
    if row.height != 1:
        raise ValueError(f"The scan has no grid point at q = {quantile}.")
    return {
        "scan": scan,
        "survival": survival,
        "label": label,
        "quantile": quantile,
        "tau_star": 2.0 * float(row["c"][0]),
        "count_star": int(row["surviving_peaks"][0]),
        "draws": int(summary_settings["n_replicates"]),
        "seed": int(summary_settings["seed"]),
        "total": int(total),
        "tau_low": 2.0 * float(scan["c"].min()),
        "tau_high": 2.0 * float(scan["c"].max()),
    }


def check_run_against_registry(
    run: dict[str, Any], quantities: Any, *, range_keys: bool = False
) -> None:
    """Raise unless each scan-derived value equals the registry value that is printed.

    The marker position and the shaded range come from ``scan.parquet`` while the
    printed text comes from the registry; nothing else ties them together. Compared
    exactly (the registry's ``agree`` rule) against the registry's exact ``value``
    text, not its rounded display. ``range_keys`` adds the minimum and maximum
    bootstrapped persistence threshold, for a figure that prints them.
    """
    pairs = [
        ("bootstrap.persistence_threshold.value", run["tau_star"]),
        ("bootstrap.candidate.after_band_rule.peak_count", run["count_star"]),
    ]
    if range_keys:
        prefix = "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_"
        pairs += [(prefix + "min", run["tau_low"]), (prefix + "max", run["tau_high"])]
    for key, drawn in pairs:
        agree(
            (f"registry {key}", quantities.value(key)),
            (f"scan.parquet value drawn for {key}", value_text(drawn)),
            BOOTSTRAP,
        )


def scan_axes(
    run: dict[str, Any],
    rendering: dict[str, Any],
    *,
    shared_x_max: float | None = None,
    show_titles: bool = True,
    show_x_labels: bool = True,
    grid: bool = False,
) -> tuple[Any, Any]:
    """Return the x and y axes of the run's survival plot.

    ``shared_x_max`` fixes the x domain to ``[0, shared_x_max]`` (a scale shared by
    several panels); otherwise it reaches the run's own ``tau_high``. ``show_titles``
    False leaves both axis titles to the caller (one shared title for a grid), and
    ``show_x_labels`` False drops the x tick labels. ``grid`` turns gridlines on.
    """
    if shared_x_max is not None:
        tau_max = float(shared_x_max)
    else:
        tau_max = max(
            min(
                float(run["survival"]["tau"].max()),
                float(rendering.get("tau_axis_max", 1.5)),
            ),
            run["tau_high"],
        )
    x_title = (
        str(rendering.get("x_axis_title", "Persistence threshold"))
        if show_titles
        else None
    )
    y_title = (
        str(rendering.get("y_axis_title", "Peaks with persistence > threshold"))
        if show_titles
        else None
    )
    x_axis = alt.X(
        "tau:Q",
        title=x_title,
        scale=alt.Scale(domain=[0, tau_max], nice=False, zero=True),
        axis=alt.Axis(grid=grid, domain=False, labels=show_x_labels, title=x_title),
    )
    y_axis = alt.Y(
        "surviving_peaks:Q",
        title=y_title,
        scale=alt.Scale(domain=[0, run["total"]], nice=False),
        axis=alt.Axis(grid=grid, domain=False, title=y_title),
    )
    return x_axis, y_axis


def scan_run_layers(
    run: dict[str, Any],
    rendering: dict[str, Any],
    x_axis: Any,
    y_axis: Any,
    marker_text: str,
) -> list[Any]:
    """Return the shading, step line, rule and note of the run.

    The rule and the note stand at the point where the quantile meets the curve;
    no point mark is drawn there. ``marker_text`` is the note's text.
    """
    ink = str(rendering.get("rule_color", EMPHASIS_BLACK))
    line = (
        alt.Chart(run["survival"])
        .mark_line(
            clip=True,
            interpolate="step-after",
            strokeWidth=float(rendering.get("line_width", 1.4)),
            color=str(rendering.get("line_color", EMPHASIS_BLACK)),
        )
        .encode(x=x_axis, y=y_axis, tooltip=["tau:Q", "surviving_peaks:Q"])
    )
    shade_table = pl.DataFrame({"tau": [run["tau_low"]], "tau_end": [run["tau_high"]]})
    shade = (
        alt.Chart(shade_table)
        .mark_rect(
            clip=True,
            opacity=float(rendering.get("shade_opacity", 0.25)),
            color=str(rendering.get("shade_color", BAND_GREY)),
        )
        .encode(x=x_axis, x2="tau_end:Q")
    )
    marker = pl.DataFrame(
        {
            "tau": [run["tau_star"]],
            "surviving_peaks": [run["count_star"]],
            "label": [marker_text],
        }
    )
    rule = (
        alt.Chart(marker)
        .mark_rule(
            strokeDash=list(rendering.get("rule_dash", [4, 3])),
            strokeWidth=float(rendering.get("rule_width", 1.0)),
            color=ink,
        )
        .encode(x=x_axis)
    )
    note = (
        alt.Chart(marker)
        .mark_text(
            align="left",
            baseline="bottom",
            dx=6,
            dy=-6,
            fontSize=int(rendering.get("font_size", 15)),
            color=ink,
        )
        .encode(x=x_axis, y=y_axis, text="label:N")
    )
    return [shade, line, rule, note]


def scan_panel_size(rendering: dict[str, Any]) -> tuple[int, int]:
    """Return the plot area's width and height in pixels (the shared tree size)."""
    return (
        int(rendering.get("panel_width", rendering.get("tree_width", TREE_WIDTH))),
        int(rendering.get("panel_height", rendering.get("tree_height", TREE_HEIGHT))),
    )


def scan_subtitle(
    rendering: dict[str, Any], defaults: list[str], fields: dict[str, Any]
) -> list[str]:
    """Return the subtitle lines: each template is a phrase wrapped on its own.

    The templates come from ``subtitle_templates`` (``defaults`` otherwise) and
    are filled from ``fields``; a phrase wraps to ``subtitle_wrap`` characters
    with balanced lines, so the subtitle fits the panel at the subtitle font size.
    """
    templates = rendering.get("subtitle_templates", defaults)
    phrases = [str(template).format(**fields) for template in templates]
    return wrap_phrases(phrases, int(rendering.get("subtitle_wrap", 62)))


def finish_scan_chart(
    layers: list[Any],
    rendering: dict[str, Any],
    subtitle: list[str],
    *,
    titled: bool = True,
) -> Any:
    """Layer, size and title the chart with the shared configuration.

    ``titled=False`` is the publication variant: no figure title or subtitle.
    The default keeps today's titled chart.
    """
    from ._title_flag import figure_heading

    font_size = int(rendering.get("font_size", 15))
    width, height = scan_panel_size(rendering)
    chart = alt.layer(*layers).properties(width=width, height=height)
    heading = figure_heading(
        str(
            rendering.get(
                "title", "Nominal peaks surviving a persistence threshold"
            )
        ),
        subtitle,
        titled=titled,
    )
    if heading is not None:
        heading["offset"] = int(rendering.get("title_offset", 40))
        chart = chart.properties(title=alt.TitleParams(**heading))
    return (
        chart
        .configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size, grid=False)
        .configure_title(
            fontSize=int(rendering.get("heading_font_size", 18)),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


def build_bootstrap_band_quantile_scan_figure(
    analysis: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Draw the survival step plot with the shaded range of thresholds the bootstrap can produce."""
    quantities = analysis["manuscript_quantities"]
    _, rendering, _, _ = biology_renderings(settings, FIGURE_ID)
    cached = analysis["bootstrap_smoothed"]
    scan: pl.DataFrame = analysis["band_quantile_scan"]
    run = scan_run(
        scan,
        analysis["survival"],
        cached.summary["settings"],
        cached.hills.height,
    )
    check_run_against_registry(run, quantities, range_keys=True)
    tau_low, tau_high = run["tau_low"], run["tau_high"]
    shown = {
        "draws": quantities.display("bootstrap.draws"),
        "seed": quantities.display("bootstrap.seed"),
        "threshold": quantities.display("bootstrap.persistence_threshold.value"),
        "band_quantile": quantities.display("bootstrap.band_quantile"),
        "count": quantities.display("bootstrap.candidate.after_band_rule.peak_count"),
        "total": quantities.display("bootstrap.candidate.nominal.peak_count"),
        "threshold_min": quantities.display(
            "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min"
        ),
        "threshold_max": quantities.display(
            "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max"
        ),
    }
    font_size = int(rendering.get("font_size", 15))
    ink = str(rendering.get("rule_color", EMPHASIS_BLACK))
    x_axis, y_axis = scan_axes(run, rendering)
    marker_text = str(
        rendering.get(
            "marker_label_template",
            "{band_quantile} quantile: {threshold}, {count} peaks",
        )
    ).format(**shown)
    shade, line, rule, note = scan_run_layers(
        run, rendering, x_axis, y_axis, marker_text
    )
    ticks = top_axis_ticks(
        scan,
        [
            float(v)
            for v in rendering.get("top_axis_quantiles", [0.0, 0.05, 0.95, 0.99])
        ],
    )
    # Neighbouring labels would overlap; alternate their height. The lower row
    # clears the ticks; the upper row stands one text height above it.
    tick_clearance = float(rendering.get("top_tick_length", 5.0)) + 1.0
    row_height = font_size + 2
    ticks = ticks.with_columns(
        (pl.int_range(pl.len()) % 2 * -row_height - tick_clearance).alias("lift")
    )
    # The ticks stand just above the plot's top edge, outside the plot area, so
    # they are not clipped (clip=True would hide every one of them).
    tick_marks = (
        alt.Chart(ticks)
        .mark_tick(
            clip=False,
            orient="vertical",
            color=ink,
            thickness=float(rendering.get("top_tick_width", 1.0)),
            size=float(rendering.get("top_tick_length", 5.0)),
            yOffset=-float(rendering.get("top_tick_length", 5.0)) / 2.0,
        )
        .encode(x=x_axis, y=alt.value(0))
    )
    tick_labels = [
        alt.Chart(ticks.filter(pl.col("lift") == lift))
        .mark_text(baseline="bottom", dy=lift, color=ink, fontSize=font_size)
        .encode(x=x_axis, y=alt.value(0), text="label:N")
        for lift in sorted(set(ticks["lift"].to_list()))
    ]
    # The axis title stands one text height above the upper row of tick labels.
    title_lift = -(tick_clearance + row_height + font_size + 4)
    axis_title = (
        alt.Chart(
            pl.DataFrame(
                {
                    "tau": [(tau_low + tau_high) / 2.0],
                    "title": [
                        str(
                            rendering.get(
                                "top_axis_title",
                                "Band quantile at its bootstrapped threshold",
                            )
                        )
                    ],
                }
            )
        )
        .mark_text(baseline="bottom", dy=title_lift, color=ink, fontSize=font_size)
        .encode(x=x_axis, y=alt.value(0), text="title:N")
    )
    dataset = str(rendering.get("dataset_label", "Papkou 2023 Candidate landscape"))
    subtitle = scan_subtitle(
        rendering,
        [
            (
                "{dataset}; {draws} bootstrap draws, seed {seed}; bootstrapped "
                "persistence threshold {threshold} at band quantile "
                "{band_quantile}, where {count} of {total} nominal peaks survive"
            ),
            (
                "The threshold is twice the radius, the band quantile of the "
                "bootstrap bottleneck distance; shaded: the smallest threshold "
                "the bootstrap can produce ({threshold_min}, band quantile 0) to "
                "the largest ({threshold_max}, band quantile 1). A peak survives "
                "when its persistence is strictly above the threshold."
            ),
        ],
        {"dataset": dataset, **shown},
    )
    return finish_scan_chart(
        [shade, line, rule, note, tick_marks, *tick_labels, axis_title],
        rendering,
        subtitle,
    )


def _grid_rendering(settings: dict[str, Any]) -> dict[str, Any]:
    """Appearance for one cell: the single-panel table, then the grid table."""
    figures = settings.get("figures", {}).get("figures", {})
    return {**figures.get(FIGURE_ID, {}), **figures.get(GRID_FIGURE_ID, {})}


def _panel_chart(
    panel: dict[str, Any],
    rendering: dict[str, Any],
    *,
    show_x_labels: bool = True,
) -> Any:
    """One cell: the existing survival layers, titled with its graph and FDR row.

    The cell is square (``panel_size``), its x scale is the shared ``[0, tau_axis_max]``
    and it carries no gridlines (``grid`` turns them on) and no axis titles: the grid prints one shared x title and
    one shared y title. Its y scale is its own (the run's peak count).
    """
    spec = panel["panel"]
    cached_settings = panel["summary_settings"]
    run = scan_run(
        panel["scan"],
        panel["survival"],
        cached_settings,
        int(panel.get("total") or panel["survival"]["surviving_peaks"][0]),
    )
    x_axis, y_axis = scan_axes(
        run,
        rendering,
        shared_x_max=float(rendering.get("tau_axis_max", 1.5)),
        show_titles=False,
        show_x_labels=show_x_labels,
        grid=bool(rendering.get("grid", False)),
    )
    shown = {
        "band_quantile": general_label_text(run["quantile"]),
        "threshold": fixed_label_text(run["tau_star"], 3),
        "count": str(run["count_star"]),
    }
    marker_text = str(
        rendering.get(
            "marker_label_template",
            "{band_quantile} quantile: {threshold}, {count} peaks",
        )
    ).format(**shown)
    layers = scan_run_layers(run, rendering, x_axis, y_axis, marker_text)
    size = int(rendering.get("panel_size", 300))
    title = alt.TitleParams(
        f"{spec.row_title} · {spec.column_title}",
        anchor="start",
        fontSize=int(rendering.get("heading_font_size", 16)),
    )
    return alt.layer(*layers).properties(width=size, height=size, title=title)


def build_bootstrap_band_quantile_scan_grid(
    panels: list[dict[str, Any]], settings: dict[str, Any]
) -> Any:
    """Draw the two BY scans in one row: the Published graph left, the Candidate graph right.

    ``panels`` stays in ``PANELS`` order (Published BY, Candidate BY). Every cell is
    square (``panel_size``), shares the x scale ``[0, tau_axis_max]`` (no gridlines) and
    keeps its own y scale. One x title sits under the row and one y title to its left.
    Each cell uses the same layers as the single-panel figure. Marker text is taken from
    that cell's scan table: the manuscript registry holds one graph, and a shared
    registry string would mislabel the other.
    """
    if len(panels) != 2:
        raise ValueError("The band-quantile scan grid needs two panels.")
    rendering = _grid_rendering(settings)
    font_size = int(rendering.get("font_size", 15))
    row = alt.hconcat(
        _panel_chart(panels[0], rendering), _panel_chart(panels[1], rendering)
    )
    # Scales: x shared by every cell, y independent per cell.
    core = (
        row.resolve_scale(x="shared", y="independent")
        .properties(
            title=alt.TitleParams(
                str(rendering.get("x_axis_title", "Persistence threshold")),
                orient="bottom",
                anchor="middle",
                fontSize=font_size,
                fontWeight="normal",
            )
        )
    )
    chart = alt.hconcat(core).properties(
        title=alt.TitleParams(
            str(rendering.get("y_axis_title", "Peaks with persistence > threshold")),
            orient="left",
            anchor="middle",
            fontSize=font_size,
            fontWeight="normal",
        )
    )
    heading = rendering.get("title")
    if heading:
        chart = alt.vconcat(chart).properties(
            title=alt.TitleParams(
                str(heading),
                anchor="start",
                fontSize=int(rendering.get("heading_font_size", 18)),
            )
        )
    return (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)
        .configure_view(strokeWidth=0)
    )
