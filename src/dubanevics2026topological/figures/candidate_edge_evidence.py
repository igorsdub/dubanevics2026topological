"""Aggregated overview and exploratory filtering for Topographer edge evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import altair as alt
import polars as pl

from dubanevics2026topological.figures.palette import INTERVAL_GREY, MID_GREY

_REQUIRED_COLUMNS = {
    "edge_id",
    "dna_left",
    "dna_right",
    "fitness_left",
    "fitness_right",
    "fitness_lower",
    "fitness_upper",
    "fitness_delta",
    "absolute_fitness_delta",
    "contrast_id",
    "reported_dna_from",
    "reported_dna_to",
    "reported_delta",
    "pairwise_standard_error",
    "test_statistic",
    "p_value",
    "q_value",
}


def select_edges(
    edges: pl.DataFrame,
    q_lower: float,
    q_upper: float,
    alpha: float,
    edge_ids: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Select the q-value interval and strict exploratory q > alpha subset."""
    _validate_unit_interval(q_lower, "q_lower")
    _validate_unit_interval(q_upper, "q_upper")
    _validate_unit_interval(alpha, "alpha")
    if q_lower > q_upper:
        raise ValueError("q_lower must be less than or equal to q_upper.")
    missing = {"edge_id", "q_value"} - set(edges.columns)
    if missing:
        raise ValueError(f"Edge table is missing columns: {sorted(missing)}.")
    selected = edges.filter(
        pl.col("q_value").is_between(q_lower, q_upper, closed="both")
    ).with_columns((pl.col("q_value") > alpha).alias("qualifies"))
    if edge_ids is not None:
        ids = list(edge_ids)
        selected = selected.filter(pl.col("edge_id").is_in(ids))
    if "reported_raw_sign_disagreement" not in selected.columns and {
        "reported_delta",
        "reported_dna_from",
        "reported_dna_to",
        "dna_left",
        "dna_right",
        "fitness_delta",
    }.issubset(selected.columns):
        oriented_delta = (
            pl.when(
                (pl.col("reported_dna_from") == pl.col("dna_left"))
                & (pl.col("reported_dna_to") == pl.col("dna_right"))
            )
            .then(pl.col("reported_delta"))
            .when(
                (pl.col("reported_dna_from") == pl.col("dna_right"))
                & (pl.col("reported_dna_to") == pl.col("dna_left"))
            )
            .then(-pl.col("reported_delta"))
            .otherwise(None)
        )
        reported_sign = (
            pl.when(oriented_delta > 0)
            .then(1)
            .when(oriented_delta < 0)
            .then(-1)
            .otherwise(0)
        )
        fitness_sign = (
            pl.when(pl.col("fitness_delta") > 0)
            .then(1)
            .when(pl.col("fitness_delta") < 0)
            .then(-1)
            .otherwise(0)
        )
        selected = selected.with_columns(
            (reported_sign != fitness_sign).alias("reported_raw_sign_disagreement")
        )
    return selected.sort(["q_value", "edge_id"])


def build_candidate_edge_evidence_figure(
    result: dict[str, Any], settings: dict[str, Any]
) -> alt.TopLevelMixin:
    """Build four compact, aggregated views of all Topographer mutation edges."""
    try:
        edges = result["edges"]
    except KeyError as error:
        raise KeyError("Topographer evidence result is missing 'edges'.") from error
    if not isinstance(edges, pl.DataFrame):
        edges = pl.DataFrame(edges)
    missing = _REQUIRED_COLUMNS - set(edges.columns)
    if missing:
        raise ValueError(f"Topographer edge table is missing columns: {sorted(missing)}.")
    if edges.is_empty():
        raise ValueError("Topographer edge evidence is empty.")

    appearance = settings["figures"]["figures"]["candidate_edge_evidence"]
    q_bins = _positive_bins(appearance.get("q_bins", 50), "q_bins")
    effect_bins = _positive_bins(appearance.get("effect_bins", 60), "effect_bins")
    fitness_bins = _positive_bins(appearance.get("fitness_bins", 60), "fitness_bins")
    q_hist = _q_histogram(edges, q_bins)
    exceedance = _q_exceedance(edges, q_bins)
    effect_q = _bivariate_counts(
        edges,
        "absolute_fitness_delta",
        "q_value",
        effect_bins,
        q_bins,
        x_domain=_finite_domain(edges, "absolute_fitness_delta", lower=0.0),
        y_domain=(0.0, 1.0),
    )
    fitness_q_coverage = _fitness_q_interval_coverage(
        edges,
        q_bins,
        fitness_bins,
        fitness_domain=(
            min(
                float(edges["fitness_lower"].min()), float(edges["fitness_upper"].min())
            ),
            max(
                float(edges["fitness_lower"].max()), float(edges["fitness_upper"].max())
            ),
        ),
    )

    base = alt.Chart
    hist = (
        base(alt.Data(values=q_hist))
        .mark_bar(orient="vertical", color=MID_GREY, binSpacing=0)
        .encode(
            x=alt.X(
                "q_start:Q",
                title="Adjusted q-value",
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            x2="q_end:Q",
            y=alt.Y("count:Q", title="Topographer edges", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("q_start:Q", title="q from", format=".2f"),
                alt.Tooltip("q_end:Q", title="q to", format=".2f"),
                alt.Tooltip("count:Q", format=","),
            ],
        )
        .properties(
            title="Adjusted q-values",
            width=int(appearance.get("panel_size", 320)),
            height=int(appearance.get("panel_size", 320)),
        )
    )
    curve = (
        base(alt.Data(values=exceedance))
        .mark_line(color="#315a7d", strokeWidth=2, interpolate="linear", point=True)
        .encode(
            x=alt.X(
                "alpha:Q",
                title="Sampled exploratory alpha",
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            y=alt.Y(
                "count:Q", title="Edges with q > alpha", scale=alt.Scale(zero=True)
            ),
            tooltip=[
                alt.Tooltip("alpha:Q", format=".2f"),
                alt.Tooltip("count:Q", format=","),
            ],
        )
        .properties(
            title="Strict q-value exceedance at sampled thresholds",
            width=int(appearance.get("panel_size", 320)),
            height=int(appearance.get("panel_size", 320)),
        )
    )
    effect_chart = _heatmap(
        effect_q,
        "x_start",
        "x_end",
        "y_start",
        "y_end",
        "count",
        "Absolute raw-fitness difference",
        "Adjusted q-value",
        "Edges",
        "Effect size by q-value",
        appearance,
    )
    interval_chart = _heatmap(
        fitness_q_coverage,
        "x_start",
        "x_end",
        "y_start",
        "y_end",
        "coverage",
        "Endpoint raw fitness bin",
        "Adjusted q-value",
        "Edge interval coverage",
        "Fitness-bin coverage by q-value",
        appearance,
    )
    font_size = int(appearance.get("font_size", 13))
    gap = int(appearance.get("panel_gap", 24))
    return (
        alt.vconcat(
            alt.hconcat(hist, curve, spacing=gap),
            alt.hconcat(effect_chart, interval_chart, spacing=gap).resolve_scale(
                color="independent"
            ),
            spacing=int(appearance.get("row_gap", 20)),
        )
        .configure(font="Arial, Liberation Sans, Helvetica, sans-serif")
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size, grid=False)
        .configure_title(fontSize=font_size + 2, anchor="start")
        .configure_view(strokeWidth=0)
    )


def selected_edge_rows(value: Any) -> pl.DataFrame:
    """Normalize Marimo table selections from list or dataframe payloads."""
    if isinstance(value, pl.DataFrame):
        return value
    if isinstance(value, dict):
        return pl.DataFrame(value)
    if isinstance(value, list):
        return pl.DataFrame(value) if value else pl.DataFrame()
    return pl.DataFrame()


def build_selected_edge_detail_figure(edges: pl.DataFrame) -> alt.HConcatChart:
    """Plot exact selected rows as q-versus-effect and q-versus-fitness intervals."""
    if edges.is_empty():
        raise ValueError("Select at least one evidence edge.")
    if edges.height > 100:
        raise ValueError("Select at most 100 evidence edges for the detail views.")
    required = {
        "edge_id",
        "dna_left",
        "dna_right",
        "fitness_left",
        "fitness_right",
        "absolute_fitness_delta",
        "q_value",
    }
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Selected edge table is missing columns: {sorted(missing)}.")
    scatter = (
        alt.Chart(edges)
        .mark_circle(size=50, opacity=0.8, color="#315a7d")
        .encode(
            x=alt.X(
                "q_value:Q",
                title="Adjusted q-value",
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            y=alt.Y(
                "absolute_fitness_delta:Q",
                title="Absolute raw-fitness difference",
            ),
            tooltip=[
                alt.Tooltip("edge_id:N", title="Edge"),
                alt.Tooltip("dna_left:N", title="DNA left"),
                alt.Tooltip("dna_right:N", title="DNA right"),
                alt.Tooltip("fitness_left:Q", title="Fitness left", format=".6f"),
                alt.Tooltip("fitness_right:Q", title="Fitness right", format=".6f"),
                alt.Tooltip("q_value:Q", title="q-value", format=".6g"),
            ],
        )
        .properties(height=320, title="Selected edges: effect size by q-value")
    )
    segments = []
    for row in edges.select(
        "edge_id",
        "dna_left",
        "dna_right",
        "fitness_left",
        "fitness_right",
        "q_value",
    ).to_dicts():
        for fitness in (row["fitness_left"], row["fitness_right"]):
            segments.append(
                {
                    "edge_id": row["edge_id"],
                    "dna_left": row["dna_left"],
                    "dna_right": row["dna_right"],
                    "q_value": row["q_value"],
                    "fitness": fitness,
                }
            )
    intervals = (
        alt.Chart(alt.Data(values=segments))
        .mark_line(opacity=0.7, color=INTERVAL_GREY)
        .encode(
            x=alt.X(
                "q_value:Q",
                title="Adjusted q-value",
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            y=alt.Y("fitness:Q", title="Endpoint raw fitness (WT = 0)"),
            detail="edge_id:N",
            tooltip=[
                alt.Tooltip("edge_id:N", title="Edge"),
                alt.Tooltip("dna_left:N", title="DNA left"),
                alt.Tooltip("dna_right:N", title="DNA right"),
                alt.Tooltip("q_value:Q", title="q-value", format=".6g"),
                alt.Tooltip("fitness:Q", title="Raw fitness", format=".6f"),
            ],
        )
        .properties(height=320, title="Selected edge fitness intervals")
    )
    return alt.hconcat(scatter, intervals, spacing=24)


def _q_histogram(edges: pl.DataFrame, bins: int) -> list[dict[str, float | int]]:
    counts = (
        edges.select(
            (pl.col("q_value").clip(0.0, 1.0) * bins)
            .floor()
            .cast(pl.Int64)
            .clip(0, bins - 1)
            .alias("bin")
        )
        .group_by("bin")
        .len()
        .rename({"len": "count"})
    )
    indexed = {int(row["bin"]): int(row["count"]) for row in counts.to_dicts()}
    return [
        {"q_start": i / bins, "q_end": (i + 1) / bins, "count": indexed.get(i, 0)}
        for i in range(bins)
    ]


def _q_exceedance(edges: pl.DataFrame, bins: int) -> list[dict[str, float | int]]:
    thresholds = [i / bins for i in range(bins + 1)]
    exprs = [
        (pl.col("q_value") > threshold).sum().alias(f"n_{i}")
        for i, threshold in enumerate(thresholds)
    ]
    counts = edges.select(exprs).row(0)
    return [
        {"alpha": alpha, "count": int(count)}
        for alpha, count in zip(thresholds, counts)
    ]


def _bivariate_counts(
    edges: pl.DataFrame,
    x_column: str,
    y_column: str,
    x_bins: int,
    y_bins: int,
    *,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
) -> list[dict[str, float | int]]:
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    x_bin = (
        ((pl.col(x_column) - x_min) / x_span * x_bins)
        .floor()
        .cast(pl.Int64)
        .clip(0, x_bins - 1)
    ).alias("_x")
    y_bin = (
        ((pl.col(y_column) - y_min) / y_span * y_bins)
        .floor()
        .cast(pl.Int64)
        .clip(0, y_bins - 1)
    ).alias("_y")
    rows = edges.select(x_bin, y_bin).group_by("_x", "_y").len().to_dicts()
    output = []
    for row in rows:
        xi, yi = int(row["_x"]), int(row["_y"])
        output.append(
            {
                "x_start": x_min + xi * x_span / x_bins,
                "x_end": x_min + (xi + 1) * x_span / x_bins,
                "y_start": y_min + yi * y_span / y_bins,
                "y_end": y_min + (yi + 1) * y_span / y_bins,
                "count": int(row["len"]),
            }
        )
    return output


def _fitness_q_interval_coverage(
    edges: pl.DataFrame,
    q_bins: int,
    fitness_bins: int,
    *,
    fitness_domain: tuple[float, float],
) -> list[dict[str, float | int]]:
    """Count each edge in each fitness bin touched, grouped by its q-value bin."""
    minimum, maximum = fitness_domain
    span = max(maximum - minimum, 1e-12)
    coverage: dict[tuple[int, int], int] = {}
    rows = edges.select("q_value", "fitness_lower", "fitness_upper").iter_rows()
    for q_value, lower, upper in rows:
        q_bin = min(int(float(q_value) * q_bins), q_bins - 1)
        first = max(
            0,
            min(int((float(lower) - minimum) / span * fitness_bins), fitness_bins - 1),
        )
        last = max(
            0,
            min(int((float(upper) - minimum) / span * fitness_bins), fitness_bins - 1),
        )
        for fitness_bin in range(first, last + 1):
            key = (q_bin, fitness_bin)
            coverage[key] = coverage.get(key, 0) + 1
    return [
        {
            "x_start": minimum + fitness_bin * span / fitness_bins,
            "x_end": minimum + (fitness_bin + 1) * span / fitness_bins,
            "y_start": q_bin / q_bins,
            "y_end": (q_bin + 1) / q_bins,
            "coverage": count,
        }
        for (q_bin, fitness_bin), count in coverage.items()
    ]


def _heatmap(
    rows: list[dict[str, float | int]],
    x: str,
    x2: str,
    y: str,
    y2: str,
    color: str,
    x_title: str,
    y_title: str,
    color_title: str,
    title: str,
    appearance: dict[str, Any],
) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:Q", title=x_title),
            x2=f"{x2}:Q",
            y=alt.Y(f"{y}:Q", title=y_title),
            y2=f"{y2}:Q",
            color=alt.Color(
                f"{color}:Q",
                title=color_title,
                scale=alt.Scale(scheme="viridis", type="log", domainMin=1),
            ),
            tooltip=[
                alt.Tooltip(f"{x}:Q", title=x_title, format=".4f"),
                alt.Tooltip(f"{y}:Q", title=y_title, format=".4f"),
                alt.Tooltip(f"{color}:Q", title=color_title, format=","),
            ],
        )
        .properties(
            title=title,
            width=int(appearance.get("panel_size", 320)),
            height=int(appearance.get("panel_size", 320)),
        )
    )


def _finite_domain(
    edges: pl.DataFrame, column: str, *, lower: float | None = None
) -> tuple[float, float]:
    minimum, maximum = edges.select(
        pl.col(column).min().alias("minimum"),
        pl.col(column).max().alias("maximum"),
    ).row(0)
    minimum, maximum = float(minimum), float(maximum)
    if lower is not None:
        minimum = lower
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError(f"Column {column!r} must contain finite values.")
    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 1e-6)
        minimum -= padding
        maximum += padding
    return minimum, maximum


def _positive_bins(value: Any, name: str) -> int:
    bins = int(value)
    if bins <= 0:
        raise ValueError(f"{name} must be positive.")
    return bins


def _validate_unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1].")
