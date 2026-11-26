"""Global verdict of the bootstrap: the nominal diagram against a confidence band.

One figure, two panels stacked. Top, 250 x 100: the bottleneck distance to the
nominal persistence diagram of the nominal-landscape draws (dark grey) and of the flat-landscape
draws (light grey, a sanity check, not the verdict) overlaid as translucent
histograms on shared bins, with the legend above and a dashed rule at the radius, the
nominal-landscape draws' band quantile of the bottleneck distance (Fasy et al. 2014). Bottom, 250 x 250: the
nominal persistence diagram, death on x against birth on y (the axes are flipped
against the usual convention, so the diagram reads like the trees' fitness axis),
with the diagonal and the cone of persistence at most the bootstrapped threshold (twice
the radius) as a translucent grey
trapezoid. Peaks are filled triangles (ADR 0012): mid-grey above the threshold (clear of
the band) and black below it (inside the band); none is hollow. The two panels do not share an x axis (distance against
death), no grid or spines are drawn and no outcome colours are used.

The diagram is the ordinary H0 diagram of the nominal tree with the global
maximum's pair closed at the global minimum, so every birth and death is finite;
it is the diagram Topographer measures every draw against, for both conditions.
The verdict is the band alone; the flat histogram and the sanity-check line of
the subtitle only confirm that the nominal-landscape draws sit far below the flat ones.

Every number the subtitle prints is the manuscript quantity registry's display text
(``analysis["manuscript_quantities"]``); the verdict's floats only place the marks and
are checked against the cached tables.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ..threshold.functional_threshold import DECLARATION_KEY, FDR_FIXED_CUTOFF
from ._charts import band_trapezoid_frame, band_trapezoid_layer
from .candidate_tree_biology import (
    biology_renderings,
    estimate_text_width,
    wrap_balanced,
    wrap_phrases,
)
from .palette import (
    DARK_GREY,
    EMPHASIS_BLACK,
    LIGHT_GREY,
    PEAK_GREY,
    PERSISTENCE_HEIGHT,
    PERSISTENCE_WIDTH,
)

FIGURE_ID = "bootstrap_global_stability"

# Subtitle phrases: ``str.format`` templates over the registry's display text, each
# wrapped on its own. No symbol stands for a quantity; the quantities are named.
SUBTITLE_VERDICT = (
    (
        "Radius {radius} (band quantile {band_quantile} of the bottleneck distance "
        "over {draws} nominal-landscape draws); bootstrapped persistence threshold "
        "{threshold}, twice the radius"
    ),
)
SUBTITLE_PEAKS = (
    (
        "{significant} of {total} nominal peaks have persistence above the "
        "threshold (grey); black peaks lie in the shaded cone."
    ),
    "Relative Wasserstein-1 median {relative}",
)
SUBTITLE_SANITY = (
    (
        "Sanity check, not the verdict: "
        "{nominal_landscape_draws_below_flat_fifth_percentile_percent}% of "
        "nominal-landscape draws lie below the flat-landscape fifth percentile "
        "({flat_bottleneck_distance_fifth_percentile});"
    ),
    (
        "medians {nominal_bottleneck_distance_median} (nominal) and "
        "{flat_bottleneck_distance_median} (flat)"
    ),
)
HISTOGRAM_COLUMNS = ("bin_start", "bin_end", "count")
SIGNIFICANT = "significant"
WITHIN_BAND = "within the band"


def condition_bottleneck(draw_stats: pl.DataFrame, condition: str) -> np.ndarray:
    """Return one condition's finite bottleneck distances."""
    column = (
        draw_stats.filter(pl.col("condition") == condition)["bottleneck"]
        .cast(pl.Float64)
        .to_numpy()
    )
    if not column.size:
        raise ValueError(f"Draw statistics have no {condition!r} draws.")
    if not np.isfinite(column).all():
        raise ValueError(f"{condition} bottleneck has non-finite values.")
    return column


def real_bottleneck(draw_stats: pl.DataFrame) -> np.ndarray:
    """Return the nominal-landscape draws' finite bottleneck distances."""
    return condition_bottleneck(draw_stats, "real")


def bottleneck_histogram(
    draw_stats: pl.DataFrame,
    bins: int,
    condition: str = "real",
    value_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Histogram one condition's bottleneck distance on equal-width bins.

    ``condition`` is ``real`` (the default) or ``flat``. The bins span that
    condition's own range unless ``value_range`` fixes them, which the overlay
    uses so both conditions share bin edges. Empty bins are left out.
    """
    values = condition_bottleneck(draw_stats, condition)
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    rows = [
        {
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "count": int(count),
        }
        for i, count in enumerate(counts)
        if count
    ]
    return pl.DataFrame(
        rows,
        schema={"bin_start": pl.Float64, "bin_end": pl.Float64, "count": pl.Int64},
    ).select(*HISTOGRAM_COLUMNS)


def diagram_points(hills: pl.DataFrame) -> pl.DataFrame:
    """Return the nominal diagram's points with their outcome and significance.

    One row per hill (the diagram point of a hill is its peak's pair): ``birth``
    and ``death`` in fitness units, ``persistence``, the hill's ``band_outcome`` and
    ``status`` (``significant`` when ``clears_band``, else ``within the band``).
    """
    unknown = sorted(set(hills["band_outcome"]) - {"kept", "absorbed"})
    if unknown:
        raise ValueError(f"Unknown hill outcomes {unknown}.")
    return hills.select(
        "peak_sequence",
        pl.col("nominal_birth").alias("birth"),
        pl.col("nominal_death").alias("death"),
        pl.col("nominal_persistence").alias("persistence"),
        "band_outcome",
        pl.when(pl.col("clears_band"))
        .then(pl.lit(SIGNIFICANT))
        .otherwise(pl.lit(WITHIN_BAND))
        .alias("status"),
    )


def diagonal_line(points: pl.DataFrame, padding: float) -> pl.DataFrame:
    """Return the two ends of the diagonal.

    The ends span the diagram's fitness range (births and deaths) plus ``padding``
    on both axes' shared domain.
    """
    low = min(points["birth"].min(), points["death"].min()) - padding
    high = max(points["birth"].max(), points["death"].max()) + padding
    return pl.DataFrame(
        {"birth": [low, high], "diagonal": [low, high]},
        schema={"birth": pl.Float64, "diagonal": pl.Float64},
    )


def _check_summary(
    summary: dict[str, Any], hills: pl.DataFrame, draw_stats: pl.DataFrame
) -> dict[str, Any]:
    """Return the verdict after checking it agrees with the cached tables."""
    verdict = summary["diagnostics"]["global_verdict"]
    significant = int(hills["clears_band"].sum())
    c = float(
        np.quantile(
            real_bottleneck(draw_stats),
            float(verdict["band_quantile"]),
            method=str(verdict["quantile_method"]),
        )
    )
    sanity = verdict["sanity_check"]
    flat = condition_bottleneck(draw_stats, "flat")
    flat_p5 = float(np.percentile(flat, 5))
    share = float(np.mean(real_bottleneck(draw_stats) < flat_p5))
    if (
        hills.height != verdict["nominal_point_count"]
        or significant != verdict["significant_point_count"]
        or not math.isclose(c, float(verdict["c"]), rel_tol=1e-9, abs_tol=1e-12)
        or not math.isclose(
            2.0 * float(verdict["c"]), float(verdict["two_c"]), rel_tol=1e-12
        )
        or not math.isclose(
            flat_p5, float(sanity["flat_bottleneck_p5"]), rel_tol=1e-9, abs_tol=1e-12
        )
        or not math.isclose(
            share,
            float(sanity["real_share_below_flat_bottleneck_p5"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "stale bootstrap-smoothed summary: its global verdict disagrees with "
            "the cached hill and draw tables; rerun `just bootstrap-smoothed-tree`."
        )
    return verdict


def _distance_panel(
    draw_stats: pl.DataFrame, verdict: dict[str, Any], rendering: dict[str, Any]
) -> Any:
    """Overlay the real and flat histograms on one x axis, legend above."""
    bins = int(rendering.get("bottleneck_bins", 40))
    both = np.concatenate(
        [condition_bottleneck(draw_stats, name) for name in ("real", "flat")]
    )
    value_range = (float(both.min()), float(both.max()))
    labels = {
        "real": str(rendering.get("real_title", "Nominal landscape")),
        "flat": str(rendering.get("flat_title", "Flat landscape")),
    }
    colors = {
        "real": str(rendering.get("real_color", DARK_GREY)),
        "flat": str(rendering.get("flat_color", LIGHT_GREY)),
    }
    table = pl.concat(
        [
            bottleneck_histogram(draw_stats, bins, name, value_range).with_columns(
                pl.lit(labels[name]).alias("condition")
            )
            for name in ("flat", "real")
        ]
    )
    font_size = int(rendering.get("font_size", 10))
    # The legend stands inside the panel, centred on its width: one symbol, the
    # label offset and the longest label wide (Vega draws a 10 pixel symbol and
    # leaves ``legend_label_offset`` pixels before the text).
    legend_width = (
        float(rendering.get("legend_symbol_size", 10))
        + float(rendering.get("legend_label_offset", 4))
        + max(estimate_text_width(label, font_size) for label in labels.values())
    )
    panel_width = int(rendering.get("histogram_width", PERSISTENCE_WIDTH))
    legend_x = (panel_width - legend_width) / 2.0
    x_axis = alt.X(
        "bin_start:Q",
        title=str(rendering.get("bottleneck_axis_title", "Bottleneck distance")),
        scale=alt.Scale(zero=False, nice=False),
        axis=alt.Axis(grid=False, domain=False),
    )
    color = alt.Color(
        "condition:N",
        title=None,
        scale=alt.Scale(
            domain=[labels["real"], labels["flat"]],
            range=[colors["real"], colors["flat"]],
        ),
        legend=alt.Legend(
            orient="none",
            direction="vertical",
            legendX=legend_x,
            legendY=float(rendering.get("legend_inset", 2)),
            labelFontSize=font_size,
            labelLimit=int(rendering.get("legend_label_limit", 240)),
            labelOffset=float(rendering.get("legend_label_offset", 4)),
            symbolOpacity=float(rendering.get("bar_opacity", 0.7)),
            padding=0,
        ),
    )
    bars = (
        alt.Chart(table)
        .mark_rect(opacity=float(rendering.get("bar_opacity", 0.7)))
        .encode(
            x=x_axis,
            x2="bin_end:Q",
            y=alt.Y(
                "count:Q",
                title=str(rendering.get("count_axis_title", "Draws")),
                axis=alt.Axis(grid=False, domain=False),
            ),
            y2=alt.datum(0),
            color=color,
            tooltip=["condition:N", "bin_start:Q", "bin_end:Q", "count:Q"],
        )
    )
    c = float(verdict["c"])
    quantile = float(verdict["band_quantile"])
    marker = pl.DataFrame({"bin_start": [c]})
    rule = (
        alt.Chart(marker)
        .mark_rule(
            color=str(rendering.get("rule_color", EMPHASIS_BLACK)),
            strokeDash=list(rendering.get("rule_dash", [4, 3])),
            strokeWidth=float(rendering.get("rule_width", 1.0)),
        )
        .encode(
            x=x_axis,
            tooltip=[alt.Tooltip("bin_start:Q", title=f"Radius ({quantile} quantile)")],
        )
    )
    return alt.layer(bars, rule).properties(
        width=panel_width,
        height=int(rendering.get("histogram_height", 100)),
    )


def _diagram_panel(
    points: pl.DataFrame, verdict: dict[str, Any], rendering: dict[str, Any]
) -> Any:
    """Draw the nominal diagram with death on x and birth on y."""
    two_c = float(verdict["two_c"])
    span = max(points["birth"].max(), points["death"].max()) - min(
        points["birth"].min(), points["death"].min()
    )
    padding = float(rendering.get("axis_padding_fraction", 0.04)) * span
    lines = diagonal_line(points, padding)
    low, high = float(lines["birth"][0]), float(lines["birth"][1])
    domain = [low, high]
    axis = alt.Axis(grid=False, domain=False)
    x_axis = alt.X(
        "death:Q",
        title=str(rendering.get("death_axis_title", "Depth, saddle or root")),
        scale=alt.Scale(domain=domain, nice=False, zero=False),
        axis=axis,
    )
    y_axis = alt.Y(
        "birth:Q",
        title=str(rendering.get("birth_axis_title", "Birth (peak fitness)")),
        scale=alt.Scale(domain=domain, nice=False, zero=False),
        axis=axis,
    )
    # Cone between the diagonal and birth = death + 2c, clipped to the shared
    # domain; no edge line.
    band = band_trapezoid_layer(
        band_trapezoid_frame(low, high, two_c, above=True).rename(
            {"along": "death", "edge": "band_edge"}
        ),
        alt.X("death:Q", scale=alt.Scale(domain=domain, nice=False, zero=False)),
        alt.Y("band_edge:Q", scale=alt.Scale(domain=domain, nice=False)),
        rendering,
    )
    diagonal = (
        alt.Chart(lines.rename({"birth": "death"}))
        .mark_line(
            clip=True,
            color=str(rendering.get("diagonal_color", EMPHASIS_BLACK)),
            strokeWidth=float(rendering.get("diagonal_width", 1.0)),
        )
        .encode(
            x=alt.X("death:Q", scale=alt.Scale(domain=domain, nice=False, zero=False)),
            y=alt.Y("diagonal:Q", scale=alt.Scale(domain=domain, nice=False)),
        )
    )
    tooltip = [
        "peak_sequence:N",
        "birth:Q",
        "death:Q",
        "persistence:Q",
        "status:N",
    ]
    layers = [band, diagonal]
    # Peaks last, so they sit on the topmost layer. Every marker is filled: mid-grey
    # above 2c (survives), black inside the band (absorbed), as in the bootstrap
    # biology figure. No hollow markers.
    for status, color in (
        (WITHIN_BAND, str(rendering.get("absorbed_color", EMPHASIS_BLACK))),
        (SIGNIFICANT, str(rendering.get("peak_color", PEAK_GREY))),
    ):
        subset = points.filter(pl.col("status") == status)
        if not subset.height:
            continue
        layers.append(
            alt.Chart(subset)
            .mark_point(
                shape=str(rendering.get("persistence_point_shape", "triangle-up")),
                filled=True,
                color=color,
                size=int(rendering.get("persistence_point_size", 70)),
                strokeWidth=float(rendering.get("point_stroke_width", 1.6)),
                opacity=float(rendering.get("point_opacity", 0.9)),
            )
            .encode(x=x_axis, y=y_axis, tooltip=tooltip)
        )
    return alt.layer(*layers).properties(
        width=int(rendering.get("persistence_width", PERSISTENCE_WIDTH)),
        height=int(rendering.get("persistence_height", PERSISTENCE_HEIGHT)),
    )


def build_bootstrap_global_stability_figure(
    analysis: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Draw the bottleneck histogram stacked over the nominal diagram and band."""
    _, rendering, _, _ = biology_renderings(settings, FIGURE_ID)
    cached = analysis["bootstrap_smoothed"]
    verdict = _check_summary(cached.summary, cached.hills, cached.draw_stats)

    points = diagram_points(cached.hills)
    quantities = analysis["manuscript_quantities"]
    shown = {
        "radius": quantities.display("bootstrap.persistence_threshold.radius"),
        "band_quantile": quantities.display("bootstrap.band_quantile"),
        "draws": quantities.display("bootstrap.draws"),
        "threshold": quantities.display("bootstrap.persistence_threshold.value"),
        "significant": quantities.display(
            "bootstrap.candidate.after_band_rule.peak_count"
        ),
        "total": quantities.display("bootstrap.candidate.nominal.peak_count"),
        "relative": quantities.display(
            "bootstrap.global_stability.relative_wasserstein_1_median"
        ),
        **{
            name: quantities.display(f"bootstrap.global_stability.sanity_check.{name}")
            for name in (
                "nominal_landscape_draws_below_flat_fifth_percentile_percent",
                "flat_bottleneck_distance_fifth_percentile",
                "nominal_bottleneck_distance_median",
                "flat_bottleneck_distance_median",
            )
        },
    }
    wrap = int(rendering.get("subtitle_wrap", 52))
    # Each phrase wraps on its own (balanced), so no line holds a lone leftover word.
    subtitle = [
        *wrap_phrases([line.format(**shown) for line in SUBTITLE_VERDICT], wrap),
        *wrap_phrases([line.format(**shown) for line in SUBTITLE_PEAKS], wrap),
        *wrap_phrases([line.format(**shown) for line in SUBTITLE_SANITY], wrap),
    ]
    font_size = int(rendering.get("font_size", 10))
    chart = alt.vconcat(
        _distance_panel(cached.draw_stats, verdict, rendering),
        _diagram_panel(points, verdict, rendering),
        spacing=int(rendering.get("panel_gap", 40)),
    ).resolve_scale(x="independent", y="independent", color="independent")
    # Figure callers may supply partial presentation settings without a declaration.
    clamped_cfg = settings.get("analysis", {}).get("clamped", {})
    declaration = clamped_cfg.get(DECLARATION_KEY, {})
    is_fdr = (
        isinstance(declaration, Mapping)
        and declaration.get("rule") == FDR_FIXED_CUTOFF
    )
    raw_title = str(
        rendering.get(
            "title",
            "Nominal persistence diagram against the bootstrap "
            "confidence band",
        )
    )
    if is_fdr:
        raw_title = raw_title.replace("confidence band", "band")

    return (
        chart.properties(
            title=alt.TitleParams(
                text=wrap_balanced(
                    raw_title,
                    wrap - 8,
                ),
                subtitle=subtitle,
                anchor="start",
            )
        )
        .configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=font_size,
            titleFontSize=font_size,
            grid=False,
            domain=False,
        )
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
        .configure_title(
            fontSize=int(rendering.get("heading_font_size", font_size + 1)),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 9)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )
