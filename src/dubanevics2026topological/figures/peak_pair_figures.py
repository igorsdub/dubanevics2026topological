"""Peak-pair distance figures: matrices and two scatters.

All read the tables of the ``peak-pair-distances`` stage (one row per unordered
pair of the 29 peaks of the band-simplified tree; the Mantel table beside it) and
draw them as they are: raw fitness, no transform. See
``docs/peak-pair-distance-design.md``.

``peak_pair_matrices`` is three square heatmaps (Hamming, fitness, difference).
Rows and columns are the peaks in the tree's leaf order, the same on both axes.
The lower triangle holds the direct quantity and the upper triangle the
via-saddle quantity; in the third heatmap the lower triangle is the Hamming
difference and the upper triangle the fitness difference, each with its own
colour bar.

``peak_pair_hamming_fitness_scatter`` is two rows (nucleotide, amino-acid Hamming
distance) of three panels (direct, via saddle, via saddle minus direct) against
fitness, one filled point per pair. The Spearman correlation and the Mantel
p-value of each panel are read from the Mantel table, never recomputed here.

``peak_pair_direct_via_saddle_scatter`` is three square panels (nucleotide
Hamming, amino-acid Hamming, fitness) of the direct distance against the
via-saddle distance, with the dashed diagonal ``y = x``. It is descriptive: the
fraction of pairs on the diagonal and the mean and median excess (via saddle minus
direct), with no correlation or p-value, because via saddle is at least direct
by construction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from dubanevics2026topological.analysis.peak_rank import translate

from ._title_flag import figure_heading
from .palette import (
    AGREEMENT_BIN_DARK,
    AGREEMENT_BIN_LIGHT,
    COUNT_RAMP_DARK,
    DIAGONAL_GREY,
    LIGHT_GREY,
    PEAK_GREY,
)

MATRICES_ID = "peak_pair_matrices"
SCATTER_ID = "peak_pair_hamming_fitness_scatter"
DIRECT_VIA_ID = "peak_pair_direct_via_saddle_scatter"
LABEL_MODES = ("amino_acid", "dna", "both")
# The three comparisons of the Mantel table, in panel order.
COMPARISONS = ("direct", "via_saddle", "via_minus_direct")
_PANEL_NAMES = {
    "direct": "Direct",
    "via_saddle": "Via saddle",
    "via_minus_direct": "Via saddle minus direct",
}
_PAIR_COLUMNS = (
    "peak_1_dna",
    "peak_2_dna",
    "peak_1_leaf_order",
    "peak_2_leaf_order",
    "nucleotide_direct_hamming",
    "nucleotide_via_saddle_hamming",
    "amino_acid_direct_hamming",
    "amino_acid_via_saddle_hamming",
    "direct_fitness_difference",
    "via_saddle_fitness_distance",
)
_MANTEL_COLUMNS = ("comparison", "alphabet", "spearman_rho", "mantel_p")


def _appearance(settings: Mapping[str, Any], figure_id: str) -> dict[str, Any]:
    return {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][figure_id],
    }


def _check_columns(frame: pl.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"The {what} table is missing columns: {missing}.")


def _finish(chart: Any, appearance: Mapping[str, Any], symbol: str = "square") -> Any:
    font_size = int(appearance.get("font_size", 12))
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size, grid=False)
        .configure_legend(
            labelFontSize=int(appearance.get("legend_font_size", font_size)),
            titleFontSize=int(appearance.get("legend_font_size", font_size)),
            symbolType=symbol,
        )
        .configure_title(
            fontSize=int(appearance.get("heading_font_size", 16)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 12)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


def _heading(appearance: Mapping[str, Any], fallback: str) -> Any:
    subtitle = appearance.get("subtitle")
    return figure_heading(
        str(appearance.get("title", fallback)),
        [str(subtitle)] if subtitle else None,
    )


# -- peak labels -------------------------------------------------------------


def peak_labels(dna_by_order: list[str], mode: str) -> list[str]:
    """One tick label per peak, in leaf order.

    ``amino_acid`` is the translation (stop codons as ``%``, ``^``, ``&``),
    ``dna`` the nucleotide sequence, ``both`` the two separated by a space. No
    suffix is added: peaks whose labels collide (synonymous variants) show the
    same text.
    """
    if mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {mode!r}.")
    if mode == "dna":
        return list(dna_by_order)
    if mode == "amino_acid":
        return [translate(dna) for dna in dna_by_order]
    return [f"{dna} {translate(dna)}" for dna in dna_by_order]


def peak_order(pairs: pl.DataFrame) -> list[str]:
    """Peak DNA sequences by leaf order, read from the pair table."""
    _check_columns(pairs, _PAIR_COLUMNS, "peak-pair")
    seen: dict[int, str] = {}
    for index_column, dna_column in (
        ("peak_1_leaf_order", "peak_1_dna"),
        ("peak_2_leaf_order", "peak_2_dna"),
    ):
        for index, dna in zip(pairs[index_column], pairs[dna_column], strict=True):
            if seen.setdefault(int(index), dna) != dna:
                raise ValueError(f"Leaf position {index} names two different peaks.")
    if sorted(seen) != list(range(len(seen))):
        raise ValueError("Leaf positions must be 0..n-1 with no gap.")
    return [seen[index] for index in range(len(seen))]


# -- matrices ----------------------------------------------------------------


def matrix_cells(pairs: pl.DataFrame, lower: str, upper: str) -> pl.DataFrame:
    """Long-form cells of a square matrix: ``lower`` below the diagonal, ``upper`` above.

    Row ``i`` and column ``j`` are leaf positions. The cell for the pair at
    positions ``(a, b)`` with ``a < b`` sits at row ``b``, column ``a`` (lower
    triangle, ``lower`` column) and at row ``a``, column ``b`` (upper triangle,
    ``upper`` column). The diagonal is 0 and tagged ``diagonal``.
    """
    _check_columns(pairs, _PAIR_COLUMNS, "peak-pair")
    n = len(peak_order(pairs))
    first = pairs["peak_1_leaf_order"].to_list()
    second = pairs["peak_2_leaf_order"].to_list()
    rows = [
        {"row": index, "column": index, "triangle": "diagonal", "value": 0}
        for index in range(n)
    ]
    for a, b, low, up in zip(first, second, pairs[lower], pairs[upper], strict=True):
        rows.append({"row": b, "column": a, "triangle": "lower", "value": low})
        rows.append({"row": a, "column": b, "triangle": "upper", "value": up})
    return pl.DataFrame(rows, schema_overrides={"row": pl.Int64, "column": pl.Int64})


def _with_difference(pairs: pl.DataFrame) -> pl.DataFrame:
    return pairs.with_columns(
        (
            pl.col("nucleotide_via_saddle_hamming")
            - pl.col("nucleotide_direct_hamming")
        ).alias("hamming_via_minus_direct"),
        (
            pl.col("via_saddle_fitness_distance") - pl.col("direct_fitness_difference")
        ).alias("fitness_via_minus_direct"),
    )


def _axes(order_labels: list[str], appearance: Mapping[str, Any]) -> tuple[Any, Any]:
    """Index axes whose tick text is the peak label (labels may repeat)."""
    table = json.dumps({str(i): text for i, text in enumerate(order_labels)})
    size = int(appearance.get("tick_label_font_size", 9))
    sort = list(range(len(order_labels)))
    x = alt.X(
        "column:O",
        sort=sort,
        title=None,
        axis=alt.Axis(
            labelExpr=f"{table}[datum.value]",
            labelAngle=-90,
            labelFontSize=size,
            labelLimit=400,
            ticks=True,
            orient="bottom",
        ),
    )
    y = alt.Y(
        "row:O",
        sort=sort,
        title=None,
        axis=alt.Axis(
            labelExpr=f"{table}[datum.value]",
            labelFontSize=size,
            labelLimit=400,
            ticks=True,
        ),
    )
    return x, y


def _discrete(
    values: list[int] | list[float], title: list[str], scheme: str, legend: Any
) -> Any:
    domain = list(range(int(min(values)), int(max(values)) + 1))
    return alt.Color(
        "value:O",
        title=title,
        scale=alt.Scale(domain=domain, scheme=scheme),
        legend=legend,
    )


def legend_ticks(high: float) -> list[int]:
    """Colour-bar ticks at whole fitness units only: ``0 .. floor(high)``.

    A maximum below 1 therefore gets the single tick ``0``.
    """
    return list(range(math.floor(float(high)) + 1))


def _continuous(high: float, title: list[str], scheme: str, legend: Any) -> Any:
    return alt.Color(
        "value:Q",
        title=title,
        scale=alt.Scale(domain=[0.0, float(high)], scheme=scheme, nice=False),
        legend=legend,
    )


def _legend(
    appearance: Mapping[str, Any],
    *,
    discrete: bool,
    direction: str,
    high: float | None = None,
) -> Any:
    options: dict[str, Any] = {
        "orient": "bottom",
        "direction": direction,
        "titleLimit": int(appearance.get("legend_title_limit", 400)),
    }
    if discrete:
        options["symbolType"] = "square"
        options["columns"] = int(appearance.get("legend_columns", 7))
    else:
        options["gradientLength"] = int(appearance.get("colorbar_length", 160))
        if high is not None:
            options["values"] = legend_ticks(high)
            options["format"] = ".0f"
    return alt.Legend(**options)


def build_peak_pair_matrices_figure(
    payload: Mapping[str, Any], settings: Mapping[str, Any]
) -> Any:
    """Three heatmaps: Hamming, fitness and the via-saddle-minus-direct difference."""
    appearance = _appearance(settings, MATRICES_ID)
    pairs = _with_difference(payload["pairs"])
    labels = peak_labels(
        peak_order(pairs), str(appearance.get("label_mode", "amino_acid"))
    )
    cell = int(appearance.get("cell_size", 14))
    size = len(labels) * cell
    neutral = str(appearance.get("diagonal_color", LIGHT_GREY))
    x, y = _axes(labels, appearance)
    direction = "horizontal"

    def layer(cells: pl.DataFrame, color: Any) -> Any:
        return (
            alt.Chart(cells)
            .mark_rect()
            .encode(
                x=x,
                y=y,
                color=color,
                tooltip=[
                    alt.Tooltip("row:Q"),
                    alt.Tooltip("column:Q"),
                    alt.Tooltip("value:Q"),
                ],
            )
        )

    def panel(chart: Any, title: str, subtitle: str) -> Any:
        return chart.properties(
            width=size,
            height=size,
            title=alt.TitleParams(text=title, subtitle=[subtitle], anchor="start"),
        )

    hamming_cells = matrix_cells(
        pairs, "nucleotide_direct_hamming", "nucleotide_via_saddle_hamming"
    )
    hamming = panel(
        layer(
            hamming_cells,
            _discrete(
                hamming_cells["value"].to_list(),
                ["Hamming distance", "(nucleotides)"],
                str(appearance.get("hamming_scheme", "viridis")),
                _legend(appearance, discrete=True, direction=direction),
            ),
        ),
        "Hamming distance",
        "Lower: direct. Upper: via saddle.",
    )
    fitness_cells = matrix_cells(
        pairs, "direct_fitness_difference", "via_saddle_fitness_distance"
    )
    fitness = panel(
        layer(
            fitness_cells,
            _continuous(
                fitness_cells["value"].max(),
                "Fitness distance",
                str(appearance.get("fitness_scheme", "magma")),
                _legend(
                    appearance,
                    discrete=False,
                    direction=direction,
                    high=fitness_cells["value"].max(),
                ),
            ),
        ),
        "Fitness distance",
        "Lower: direct. Upper: via saddle.",
    )
    low = matrix_cells(
        pairs, "hamming_via_minus_direct", "hamming_via_minus_direct"
    ).filter(pl.col("triangle") == "lower")
    up = matrix_cells(
        pairs, "fitness_via_minus_direct", "fitness_via_minus_direct"
    ).filter(pl.col("triangle") == "upper")
    diagonal = matrix_cells(
        pairs, "hamming_via_minus_direct", "hamming_via_minus_direct"
    ).filter(pl.col("triangle") == "diagonal")
    difference = panel(
        alt.layer(
            alt.Chart(diagonal).mark_rect(color=neutral).encode(x=x, y=y),
            layer(
                low,
                _discrete(
                    low["value"].to_list(),
                    ["Via saddle minus direct", "Hamming (nucleotides)"],
                    str(appearance.get("hamming_difference_scheme", "oranges")),
                    _legend(appearance, discrete=True, direction=direction),
                ),
            ),
            layer(
                up,
                _continuous(
                    up["value"].max(),
                    ["Via saddle minus direct", "fitness"],
                    str(appearance.get("fitness_difference_scheme", "teals")),
                    _legend(
                        appearance,
                        discrete=False,
                        direction=direction,
                        high=up["value"].max(),
                    ),
                ),
            ),
        ).resolve_scale(color="independent"),
        "Difference",
        "Lower: Hamming. Upper: fitness. Via saddle minus direct.",
    )
    chart = alt.hconcat(
        hamming, fitness, difference, spacing=int(appearance.get("panel_gap", 40))
    ).resolve_scale(color="independent")
    heading = _heading(appearance, "Peak-pair distances")
    if heading is not None:
        chart = chart.properties(title=alt.TitleParams(**heading))
    return _finish(chart, appearance)


# -- scatter -----------------------------------------------------------------


def mantel_lookup(mantel: pl.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """``(comparison, alphabet)`` -> ``(spearman_rho, mantel_p)`` from the Mantel table."""
    _check_columns(mantel, _MANTEL_COLUMNS, "Mantel")
    return {
        (row["comparison"], row["alphabet"]): (row["spearman_rho"], row["mantel_p"])
        for row in mantel.iter_rows(named=True)
    }


def _statistic_line(rho: float, p_value: float) -> str:
    if not (np.isfinite(rho) and np.isfinite(p_value)):
        return "Spearman \u03c1 and Mantel p undefined"
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    return f"Spearman \u03c1 = {rho:.3f} (Mantel {p_text})"


def jittered(values: list[int], amount: float, seed: int) -> np.ndarray:
    """``values`` plus uniform noise in ``[-amount, amount]``; display only."""
    rng = np.random.default_rng(seed)
    return np.asarray(values, dtype=float) + rng.uniform(
        -amount, amount, size=len(values)
    )


def scatter_points(
    pairs: pl.DataFrame, comparison: str, alphabet: str, amount: float, seed: int
) -> pl.DataFrame:
    """One row per pair for one panel: Hamming (jittered and exact) and fitness."""
    if comparison not in COMPARISONS:
        raise ValueError(f"Unknown comparison {comparison!r}.")
    if alphabet not in ("nucleotide", "amino_acid"):
        raise ValueError(f"Unknown alphabet {alphabet!r}.")
    direct = pairs[f"{alphabet}_direct_hamming"]
    via = pairs[f"{alphabet}_via_saddle_hamming"]
    if comparison == "direct":
        exact, y = direct, pairs["direct_fitness_difference"]
    elif comparison == "via_saddle":
        exact, y = via, pairs["via_saddle_fitness_distance"]
    else:
        exact = via - direct
        y = pairs["via_saddle_fitness_distance"] - pairs["direct_fitness_difference"]
    noise = jittered([0] * pairs.height, amount, seed) if pairs.height else np.array([])
    return pl.DataFrame(
        {
            "hamming": exact.cast(pl.Float64) + pl.Series(noise),
            "hamming_exact": exact,
            "fitness": y,
        }
    )


_SCATTER_Y_TITLES = {
    "direct": "Direct fitness difference",
    "via_saddle": "Via-saddle fitness distance",
    "via_minus_direct": "Via saddle minus direct fitness",
}
_SCATTER_X_PREFIX = {
    "direct": "Direct Hamming distance",
    "via_saddle": "Via-saddle Hamming distance",
    "via_minus_direct": "Via saddle minus direct Hamming distance",
}
_ALPHABET_NOUNS = {"nucleotide": "nucleotides", "amino_acid": "amino acids"}


def padded_range(low: float, high: float, fraction: float = 0.05) -> list[float]:
    """``[low, high]`` widened by ``fraction`` of its width (a unit width if flat or inverted)."""
    width = high - low
    pad = fraction * (width if width > 0 else 1.0)
    if width <= 0:
        return [low - pad, low + pad]
    return [low - pad, high + pad]


def build_peak_pair_hamming_fitness_scatter_figure(
    payload: Mapping[str, Any], settings: Mapping[str, Any]
) -> Any:
    """Two rows (nucleotide, amino acid) of three panels (direct, via saddle, difference)."""
    appearance = _appearance(settings, SCATTER_ID)
    pairs = payload["pairs"]
    _check_columns(pairs, _PAIR_COLUMNS, "peak-pair")
    statistics = mantel_lookup(payload["mantel"])
    amount = float(appearance.get("jitter", 0.15))
    seed = int(appearance.get("jitter_seed", 20261121))
    color = str(appearance.get("point_color", AGREEMENT_BIN_DARK))
    width = int(appearance.get("panel_width", 300))
    height = int(appearance.get("panel_height", 300))
    alphabets = ("nucleotide", "amino_acid")
    points = {
        (c, a): scatter_points(pairs, c, a, amount, seed)
        for c in COMPARISONS
        for a in alphabets
    }
    # Each column shares one y range across both rows.
    y_domain = {}
    for comparison in COMPARISONS:
        values = pl.concat([points[(comparison, a)]["fitness"] for a in alphabets])
        low = float(values.min())
        low = min(low, 0.0) if comparison != "via_minus_direct" else low
        y_domain[comparison] = padded_range(low, float(values.max()))
    rows = []
    for alphabet in alphabets:
        panels = []
        for comparison in COMPARISONS:
            data = points[(comparison, alphabet)]
            exact = data["hamming_exact"]
            low, high = int(exact.min()), int(exact.max())
            x_pad = amount + 0.05 * (high - low if high > low else 1)
            noun = _ALPHABET_NOUNS[alphabet]
            encoded = (
                alt.Chart(data)
                .mark_point(
                    filled=True,
                    shape="circle",
                    color=color,
                    size=float(appearance.get("point_size", 36)),
                    opacity=float(appearance.get("point_opacity", 0.8)),
                )
                .encode(
                    x=alt.X(
                        "hamming:Q",
                        title=f"{_SCATTER_X_PREFIX[comparison]} ({noun})",
                        scale=alt.Scale(domain=[low - x_pad, high + x_pad], nice=False),
                        axis=alt.Axis(values=list(range(low, high + 1)), grid=False),
                    ),
                    y=alt.Y(
                        "fitness:Q",
                        title=_SCATTER_Y_TITLES[comparison],
                        scale=alt.Scale(domain=y_domain[comparison], nice=False),
                        axis=alt.Axis(grid=False),
                    ),
                )
            )
            title = alt.TitleParams(
                text=f"{_PANEL_NAMES[comparison]} ({noun})",
                subtitle=_statistic_line(*statistics[(comparison, alphabet)]),
                anchor="start",
            )
            panels.append(encoded.properties(width=width, height=height, title=title))
        rows.append(alt.hconcat(*panels, spacing=int(appearance.get("panel_gap", 40))))
    chart = alt.vconcat(*rows, spacing=int(appearance.get("row_gap", 40)))
    heading = _heading(appearance, "Hamming distance against fitness distance")
    if heading is not None:
        chart = chart.properties(title=alt.TitleParams(**heading))
    return _finish(chart, appearance)


# -- direct against via-saddle -------------------------------------------------

# A pair is on the diagonal when via saddle equals direct exactly (Hamming distances
# are integers; fitness values are compared as stored, with no tolerance).
_DIRECT_VIA_COLUMNS = {
    "nucleotide": ("nucleotide_direct_hamming", "nucleotide_via_saddle_hamming"),
    "amino_acid": ("amino_acid_direct_hamming", "amino_acid_via_saddle_hamming"),
}


def direct_via_cells(pairs: pl.DataFrame, alphabet: str) -> pl.DataFrame:
    """One row per occupied (direct, via-saddle) Hamming cell with its pair count."""
    _check_columns(pairs, _PAIR_COLUMNS, "peak-pair")
    direct, via = _DIRECT_VIA_COLUMNS[alphabet]
    return (
        pairs.group_by(pl.col(direct).alias("direct"), pl.col(via).alias("via_saddle"))
        .agg(pl.len().alias("pairs"))
        .sort("direct", "via_saddle")
    )


def diagonal_summary(
    direct: list[float], via_saddle: list[float]
) -> tuple[float, float, float]:
    """``(fraction on the diagonal, mean excess, median excess)``, via minus direct.

    On the diagonal means via saddle equals direct exactly. The median of an even
    count is the mean of the two middle values.
    """
    if not direct or len(direct) != len(via_saddle):
        raise ValueError(
            "Direct and via-saddle values must be equal-length and non-empty."
        )
    excess = [v - d for d, v in zip(direct, via_saddle, strict=True)]
    on_diagonal = sum(1 for value in excess if value == 0)
    return on_diagonal / len(excess), float(np.mean(excess)), float(np.median(excess))


def _number(value: float, decimals: int) -> str:
    """Fixed decimals with trailing zeros dropped, so whole numbers print bare."""
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _diagonal_subtitle(
    direct: list[float], via_saddle: list[float], decimals: int
) -> list[str]:
    """Two lines: diagonal count, then mean / median excess (via minus direct).

    Numbers use ``decimals`` places (Hamming 1, fitness 3), trailing zeros dropped.
    """
    fraction, mean, median = diagonal_summary(direct, via_saddle)
    count = round(fraction * len(direct))
    return [
        f"Pairs on the diagonal: {count} of {len(direct)} ({fraction:.0%})",
        f"Mean / median excess: {_number(mean, decimals)} / {_number(median, decimals)}",
    ]


def fitness_ticks(high: float) -> list[float]:
    """Round tick values from 0 to ``high`` (about five intervals), shared by x and y."""
    if high <= 0:
        return [0.0]
    raw = high / 5
    magnitude = 10 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    count = int(np.floor(high / step + 1e-9))
    return [round(float(i * step), 10) for i in range(count + 1)]


def build_peak_pair_direct_via_saddle_figure(
    payload: Mapping[str, Any], settings: Mapping[str, Any]
) -> Any:
    """Three square panels of direct against via-saddle distance, one row."""
    appearance = _appearance(settings, DIRECT_VIA_ID)
    pairs = payload["pairs"]
    _check_columns(pairs, _PAIR_COLUMNS, "peak-pair")
    side = int(appearance.get("panel_size", 300))
    colour = str(appearance.get("point_color", AGREEMENT_BIN_DARK))
    diagonal_colour = str(appearance.get("diagonal_color", DIAGONAL_GREY))
    dash = [int(v) for v in appearance.get("dash", [6, 4])]
    marker_size = float(appearance.get("marker_size", 90))
    grey_low = str(appearance.get("count_grey_low", AGREEMENT_BIN_LIGHT))
    grey_high = str(appearance.get("count_grey_high", COUNT_RAMP_DARK))
    point_size = float(appearance.get("point_size", 40))
    fitness_opacity = float(appearance.get("fitness_opacity", 0.5))
    cells = {a: direct_via_cells(pairs, a) for a in _DIRECT_VIA_COLUMNS}
    most = max(int(frame["pairs"].max()) for frame in cells.values())
    stroke = str(appearance.get("circle_stroke", PEAK_GREY))
    stroke_width = float(appearance.get("circle_stroke_width", 0.5))
    # Legend ticks are exactly 1, 10 and 100, as far as the largest cell count reaches.
    legend_values = [v for v in (1, 10, 100) if v <= most]
    count_scale = alt.Scale(
        type="log", domain=[1, max(most, 2)], range=[grey_low, grey_high], clamp=True
    )
    panels = []

    def panel(
        data: pl.DataFrame, x_title: str, y_title: str, title: Any, sized: bool
    ) -> Any:
        high = float(max(data["direct"].max(), data["via_saddle"].max()))
        low, top = padded_range(0.0, high)
        scale = alt.Scale(domain=[low, top], nice=False)
        ticks = list(range(int(high) + 1)) if sized else fitness_ticks(high)
        axis = alt.Axis(values=ticks, grid=False)
        x = alt.X("direct:Q", title=x_title, scale=scale, axis=axis)
        y = alt.Y("via_saddle:Q", title=y_title, scale=scale, axis=axis)
        line = (
            alt.Chart(pl.DataFrame({"direct": [low, top], "via_saddle": [low, top]}))
            .mark_line(color=diagonal_colour, strokeDash=dash, strokeWidth=1.2)
            .encode(x=x, y=y)
        )
        if sized:
            mark = alt.Chart(data).mark_circle(
                size=marker_size, opacity=1, stroke=stroke, strokeWidth=stroke_width
            )
        else:
            mark = alt.Chart(data).mark_point(
                shape="circle",
                filled=True,
                color=colour,
                opacity=fitness_opacity,
                size=point_size,
                strokeWidth=stroke_width,
            )
        encoding: dict[str, Any] = {"x": x, "y": y}
        if sized:
            encoding["color"] = alt.Color(
                "pairs:Q",
                title="Pairs in the cell",
                scale=count_scale,
                legend=alt.Legend(orient="bottom", values=legend_values, format="d"),
            )
        circles = mark.encode(**encoding)
        return alt.layer(line, circles).properties(width=side, height=side, title=title)

    for alphabet, noun in (
        ("nucleotide", "nucleotides"),
        ("amino_acid", "amino acids"),
    ):
        data = cells[alphabet]
        direct_col, via_col = _DIRECT_VIA_COLUMNS[alphabet]
        title = alt.TitleParams(
            text="Nucleotide Hamming distance"
            if alphabet == "nucleotide"
            else "Amino-acid Hamming distance",
            subtitle=_diagonal_subtitle(
                pairs[direct_col].to_list(), pairs[via_col].to_list(), 1
            ),
            anchor="start",
        )
        panels.append(
            panel(
                data,
                f"Direct Hamming distance ({noun})",
                f"Via-saddle Hamming distance ({noun})",
                title,
                True,
            )
        )
    fitness = pl.DataFrame(
        {
            "direct": pairs["direct_fitness_difference"],
            "via_saddle": pairs["via_saddle_fitness_distance"],
        }
    )
    title = alt.TitleParams(
        text="Fitness",
        subtitle=_diagonal_subtitle(
            fitness["direct"].to_list(),
            fitness["via_saddle"].to_list(),
            3,
        ),
        anchor="start",
    )
    panels.append(
        panel(
            fitness,
            "Direct fitness difference",
            "Via-saddle fitness distance",
            title,
            False,
        )
    )
    chart = alt.hconcat(
        *panels, spacing=int(appearance.get("panel_gap", 40))
    ).resolve_scale(color="shared")
    heading = _heading(appearance, "Direct against via-saddle distance")
    if heading is not None:
        chart = chart.properties(title=alt.TitleParams(**heading))
    return _finish(chart, appearance, symbol="circle")
