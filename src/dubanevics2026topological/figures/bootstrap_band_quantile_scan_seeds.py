"""Band-quantile scan of the 10,000-draw bootstrap run: surviving peaks against the band quantile.

Plots the run's surviving peaks (persistence strictly above the bootstrapped
persistence threshold, twice the radius) against the band quantile from its scan
table, as a step line. The run's translucent band is the contiguous band-quantile
interval over which its count at the stage's band quantile holds (from the scan
stage's ``steps_around``); one vertical rule marks that quantile.

The run's legend entry and marker (draws, seed, threshold, surviving peaks) and the
subtitle's band quantile are the manuscript quantity registry's display text
(``analysis["manuscript_quantities"]``). The legend sits inside the plot area, in
whichever of the top-right and bottom-left corners the step line and marker label
leave emptier. No point mark is drawn where the dashed rule meets the step line; the
marker label alone names it. The panel size and every font size are those of
``linked_tree_persistence``.

Only this one run is read or drawn; there is no second (1,000-draw) run to overlay.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ..analysis.bootstrap_band_quantile_scan import steps_around
from .bootstrap_band_quantile_scan import (
    check_run_against_registry,
    finish_scan_chart,
    scan_panel_size,
    scan_run,
    scan_subtitle,
)
from .candidate_tree_biology import (
    biology_renderings,
    emptiest_corner,
    inset_legend,
    segment_sample_points,
)
from .palette import EMPHASIS_BLACK

FIGURE_ID = "bootstrap_band_quantile_scan_seeds"


def holding_range(scan: pl.DataFrame, quantile: float) -> tuple[float, float]:
    """Return the first and last grid ``q`` where the count at ``quantile`` holds."""
    steps = steps_around(scan, quantile)
    q = scan["q"]
    below, above = steps["below"], steps["above"]
    low = q.filter(q > below["q"]).min() if below else q.min()
    high = q.filter(q < above["q"]).max() if above else q.max()
    return float(low), float(high)


def holding_extent(scan: pl.DataFrame, quantile: float) -> tuple[float, float]:
    """Return the ``q`` interval a step line holds the count at ``quantile``.

    The line steps after each grid point, so the count at the last holding grid
    point is drawn until the next grid point: the interval ends at the first grid
    ``q`` where the count changes, or at the scan's last ``q`` when it never does.
    """
    low, _ = holding_range(scan, quantile)
    above = steps_around(scan, quantile)["above"]
    return low, float(above["q"]) if above else float(scan["q"].max())


def window_counts(scan: pl.DataFrame, q_min: float, q_max: float) -> pl.DataFrame:
    """Return the scan rows with ``q`` in the plotted window."""
    return scan.filter((pl.col("q") >= q_min) & (pl.col("q") <= q_max)).select(
        "q", "surviving_peaks"
    )


def step_segments(window: pl.DataFrame) -> pl.DataFrame:
    """Return the segments a step-after line draws through ``window``.

    From each grid point the line runs level to the next ``q`` and then steps to
    the next count; the last point draws nothing further.
    """
    q = window["q"].to_numpy()
    count = window["surviving_peaks"].to_numpy().astype(float)
    return pl.DataFrame(
        {
            "x_start": np.concatenate([q[:-1], q[1:]]),
            "x_end": np.concatenate([q[1:], q[1:]]),
            "y_start": np.concatenate([count[:-1], count[:-1]]),
            "y_end": np.concatenate([count[:-1], count[1:]]),
        }
    )


def label_segment(
    anchor: tuple[float, float],
    text: str,
    placement: Sequence[Any],
    rendering: dict[str, Any],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
) -> dict[str, float]:
    """Return the level run a marker label occupies, in axis coordinates.

    ``placement`` is the label's ``[align, dx, dy, baseline]`` from the config
    (pixel offsets, ``dy`` down). The text is ``label_char_width`` font sizes per
    character wide; its run is taken at the text's vertical centre.
    """
    align, dx, dy, baseline = placement
    font_size = int(rendering.get("font_size", 15))
    width, height = scan_panel_size(rendering)
    text_width = len(text) * font_size * float(rendering.get("label_char_width", 0.55))
    start = float(dx) + {"left": 0.0, "center": -0.5, "right": -1.0}[align] * text_width
    centre = (
        -float(dy) + {"top": -0.5, "middle": 0.0, "bottom": 0.5}[baseline] * font_size
    )
    x_per_pixel = (x_domain[1] - x_domain[0]) / width
    y_per_pixel = (y_domain[1] - y_domain[0]) / height
    y = anchor[1] + centre * y_per_pixel
    x_start = anchor[0] + start * x_per_pixel
    return {
        "x_start": x_start,
        "x_end": x_start + text_width * x_per_pixel,
        "y_start": y,
        "y_end": y,
    }


def legend_orient(
    windows: Sequence[pl.DataFrame],
    labels: Sequence[dict[str, float]],
    rendering: dict[str, Any],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
) -> str:
    """Pick the corner for the run legend: the emptier of ``legend_corners``.

    Ink is sampled along every run's step line and along each marker label, in
    axis coordinates, and the corner box (``legend_box_fraction``) with the least
    of it wins; a tie goes to the first listed corner.
    """
    corners = [
        str(corner)
        for corner in rendering.get("legend_corners", ["top-right", "bottom-left"])
    ]
    segments = pl.concat(
        [*(step_segments(window) for window in windows), pl.DataFrame(labels)]
    )
    xs, ys, _ = segment_sample_points(
        segments,
        x_domain,
        y_domain,
        step=float(rendering.get("legend_sample_step", 0.01)),
    )
    return emptiest_corner(xs, ys, x_domain, y_domain, rendering, corners)


def build_bootstrap_band_quantile_scan_seeds_figure(
    analysis: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Draw the live run's surviving peaks against the band quantile."""
    quantities = analysis["manuscript_quantities"]
    _, rendering, _, _ = biology_renderings(settings, FIGURE_ID)
    template = str(
        rendering.get("run_label_template", "current run, {draws} draws, seed {seed}")
    )
    cached = analysis["bootstrap_smoothed"]
    total = cached.hills.height
    live = scan_run(
        analysis["band_quantile_scan"],
        None,
        cached.summary["settings"],
        total,
        template.format(
            draws=quantities.display("bootstrap.draws"),
            seed=quantities.display("bootstrap.seed"),
        ),
    )
    check_run_against_registry(live, quantities)
    quantile = live["quantile"]
    runs = [live]
    q_min = float(rendering.get("q_axis_min", 0.90))
    q_max = float(rendering.get("q_axis_max", 1.0))
    windows = [window_counts(run["scan"], q_min, q_max) for run in runs]
    counts = pl.concat([w["surviving_peaks"] for w in windows])
    pad = int(rendering.get("y_padding", 2))
    y_low, y_high = max(0, int(counts.min()) - pad), int(counts.max()) + pad
    font_size = int(rendering.get("font_size", 15))
    label_template = str(
        rendering.get("marker_label_template", "{threshold}, {count} peaks")
    )
    # Marker label placement: [align, dx, dy, baseline].
    placements = [rendering.get("label_placement", ["left", 7, -9, "bottom"])]
    marker_texts = [
        label_template.format(
            threshold=quantities.display("bootstrap.persistence_threshold.value"),
            count=quantities.display("bootstrap.candidate.after_band_rule.peak_count"),
        ),
    ]
    x_domain, y_domain = (q_min, q_max), (float(y_low), float(y_high))
    show_legend = bool(rendering.get("show_legend", True))
    corner = legend_orient(
        windows,
        [
            label_segment(
                (quantile, float(run["count_star"])),
                text,
                place,
                rendering,
                x_domain,
                y_domain,
            )
            for run, text, place in zip(runs, marker_texts, placements, strict=True)
        ],
        rendering,
        x_domain,
        y_domain,
    )
    # At the shared font size a run label is wider than the strip between the plot
    # edge and the shaded bands, so it may break into two lines at a separator.
    legend_options: dict[str, Any] = {}
    split = rendering.get("legend_label_split")
    if split:
        legend_options["labelExpr"] = f"split(datum.label, {json.dumps(str(split))})"
        legend_options["rowPadding"] = int(rendering.get("legend_row_padding", 6))
    colors = alt.Color(
        "run:N",
        title=None,
        scale=alt.Scale(
            domain=[run["label"] for run in runs],
            range=[str(rendering.get("live_color", EMPHASIS_BLACK))],
        ),
        # ``show_legend = false`` (a single run needs no key) leaves the legend out.
        legend=inset_legend(
            None,
            rendering,
            orient=corner,
            labelFontSize=font_size,
            labelLimit=int(rendering.get("legend_label_limit", 400)),
            symbolOpacity=1,
            # Every plotted mark is a step line, so the key shows a line segment.
            symbolType="stroke",
            symbolStrokeWidth=float(rendering.get("line_width", 1.4)),
            **legend_options,
        )
        if show_legend
        else None,
    )
    x_axis = alt.X(
        "q:Q",
        title=str(rendering.get("x_axis_title", "Band quantile")),
        scale=alt.Scale(domain=[q_min, q_max], nice=False, zero=False),
    )
    y_scale = alt.Scale(domain=[y_low, y_high], nice=False, zero=False)
    y_axis = alt.Y(
        "surviving_peaks:Q",
        title=str(rendering.get("y_axis_title", "Peaks with persistence > threshold")),
        scale=y_scale,
    )
    ink = str(rendering.get("rule_color", EMPHASIS_BLACK))
    layers: list[Any] = []
    marks: list[Any] = []
    for run, window, place, text in zip(
        runs, windows, placements, marker_texts, strict=True
    ):
        label = run["label"]
        run["holding"] = holding_range(run["scan"], quantile)
        low, high = holding_extent(run["scan"], quantile)
        shade = pl.DataFrame(
            {"q": [low], "q_end": [high], "y_low": [y_low], "y_high": [y_high]}
        ).with_columns(pl.lit(label).alias("run"))
        layers.append(
            alt.Chart(shade)
            .mark_rect(clip=True, opacity=float(rendering.get("shade_opacity", 0.25)))
            .encode(
                x=x_axis,
                x2="q_end:Q",
                y=alt.Y("y_low:Q", scale=y_scale),
                y2="y_high:Q",
                color=colors,
            )
        )
        layers.append(
            alt.Chart(window.with_columns(pl.lit(label).alias("run")))
            .mark_line(
                clip=True,
                interpolate="step-after",
                strokeWidth=float(rendering.get("line_width", 1.4)),
            )
            .encode(
                x=x_axis,
                y=y_axis,
                color=colors,
                tooltip=["q:Q", "surviving_peaks:Q"],
            )
        )
        marker = pl.DataFrame(
            {
                "q": [quantile],
                "surviving_peaks": [run["count_star"]],
                "label": [text],
            }
        ).with_columns(pl.lit(label).alias("run"))
        align, dx, dy, baseline = place
        backing = rendering.get("marker_label_backing", "white")
        if backing:
            # A white box the size of the label, so a step line cannot show
            # through the glyphs. The box is the same estimate the legend uses.
            pad = float(rendering.get("marker_label_backing_pad", 3))
            seg = label_segment(
                (quantile, float(run["count_star"])),
                text,
                place,
                rendering,
                x_domain,
                y_domain,
            )
            panel_w, panel_h = scan_panel_size(rendering)
            x_per = (x_domain[1] - x_domain[0]) / panel_w
            y_per = (y_domain[1] - y_domain[0]) / panel_h
            half = (0.5 * font_size + pad) * y_per
            box = pl.DataFrame(
                {
                    "x": [seg["x_start"] - pad * x_per],
                    "x2": [seg["x_end"] + pad * x_per],
                    "y": [seg["y_start"] - half],
                    "y2": [seg["y_end"] + half],
                }
            )
            marks.append(
                alt.Chart(box)
                .mark_rect(color=str(backing), opacity=1)
                .encode(
                    x=alt.X(
                        "x:Q",
                        scale=alt.Scale(domain=[q_min, q_max], nice=False, zero=False),
                    ),
                    x2="x2:Q",
                    y=alt.Y("y:Q", scale=y_scale),
                    y2="y2:Q",
                )
            )
        marks.append(
            alt.Chart(marker)
            .mark_text(
                align=align,
                baseline=baseline,
                dx=int(dx),
                dy=int(dy),
                fontSize=font_size,
            )
            .encode(x=x_axis, y=y_axis, text="label:N", color=colors)
        )
    rule = (
        alt.Chart(pl.DataFrame({"q": [quantile]}))
        .mark_rule(
            color=ink,
            strokeDash=list(rendering.get("rule_dash", [4, 3])),
            strokeWidth=float(rendering.get("rule_width", 1.0)),
        )
        .encode(x=x_axis)
    )
    dataset = str(rendering.get("dataset_label", "Papkou 2023 Topographer + FDR landscape"))
    fields = {
        "dataset": dataset,
        "band_quantile": quantities.display("bootstrap.band_quantile"),
    }
    subtitle = scan_subtitle(
        rendering,
        [
            (
                "{dataset}; peaks surviving the bootstrapped persistence threshold "
                "against the band quantile (the threshold is twice the radius, "
                "the band quantile of the bootstrap bottleneck distance)"
            ),
            (
                "One step line; shaded: the band-quantile interval over "
                "which the run's count at band quantile {band_quantile} holds."
            ),
            (
                "Marker label: bootstrapped persistence threshold, surviving "
                "peaks. A peak survives when its persistence is strictly above "
                "the threshold."
            ),
        ],
        fields,
    )
    return finish_scan_chart([*layers, rule, *marks], rendering, subtitle)
