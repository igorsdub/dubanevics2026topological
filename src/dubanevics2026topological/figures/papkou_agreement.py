"""Agreement of our growth-rate fits with Papkou et al. 2023.

Supplementary S1. Two panels, one point per genotype matched on
``dna_sequence``. Panel A plots Papkou's published fitness (column ``s`` of
``fitness_wt``, their ``m``) against our fitted ``s``. Panel B plots Papkou's
published standard error (column ``se``, their ``m_se``) against our
``se_pooled``. Both panels draw ``y = x``. A panel with at least
``hexbin_min`` genotypes is a fine hexagonal binned density (axes cropped to the
data); a smaller panel is a scatter.
Nothing here refits a growth rate.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

from dubanevics2026topological.figures._count_labels import (
    count_label_text,
    fixed_label_text,
)
from dubanevics2026topological.figures._hexbin import (
    data_domain,
    hexagon_marks,
    hexbin_counts,
)
from dubanevics2026topological.figures.palette import (
    AGREEMENT_BIN_DARK,
    AGREEMENT_BIN_LIGHT,
    DIAGONAL_GREY,
)

FIGURE_ID = "papkou_agreement"
# fitness_wt stores Papkou's published m and m_se under these names.
PUBLISHED_FITNESS = "s"
PUBLISHED_SE = "se"
_HEXBIN_MIN = 2500
_HEXBIN_BINS = 70


def matched_genotypes(
    growth_rates: pl.DataFrame, published: pl.DataFrame
) -> pl.DataFrame:
    """Inner-join our fits to Papkou's published values on ``dna_sequence``.

    ``published`` is the table ``load_fitness_wt`` returns. Its ``dna_to`` is
    the genotype and its ``s`` and ``se`` are the published fitness and
    standard error. Rows that exist on only one side are left unmatched. A
    duplicate genotype on either side is an error.
    """
    fit_cols = {"dna_sequence", "s", "se_pooled"}
    missing = sorted(fit_cols - set(growth_rates.columns))
    if missing:
        raise ValueError(f"Growth-rate table is missing columns: {missing}.")
    pub_cols = {"dna_to", PUBLISHED_FITNESS, PUBLISHED_SE}
    missing = sorted(pub_cols - set(published.columns))
    if missing:
        raise ValueError(f"Published fitness table is missing columns: {missing}.")
    fits = growth_rates.select(
        pl.col("dna_sequence").cast(pl.String),
        pl.col("s").cast(pl.Float64).alias("fitted_s"),
        pl.col("se_pooled").cast(pl.Float64),
    )
    papkou = published.select(
        pl.col("dna_to").cast(pl.String).alias("dna_sequence"),
        pl.col(PUBLISHED_FITNESS).cast(pl.Float64).alias("published_m"),
        pl.col(PUBLISHED_SE).cast(pl.Float64).alias("published_m_se"),
    )
    for frame, label in ((fits, "growth-rate"), (papkou, "published fitness")):
        duplicated = frame.filter(pl.col("dna_sequence").is_duplicated())
        if duplicated.height:
            names = ", ".join(
                repr(name) for name in duplicated["dna_sequence"].unique().sort()
            )
            raise ValueError(f"Duplicate {label} rows for {names}.")
    return fits.join(papkou, on="dna_sequence", how="inner").sort("dna_sequence")


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Pearson and Spearman correlation of two equal finite samples, plus ``n``."""
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Correlation inputs must be equal-length 1-d samples.")
    if left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Correlation needs at least two finite values.")
    pearson = float(pearsonr(left, right).statistic)
    spearman = float(spearmanr(left, right).statistic)
    if not np.isfinite(pearson) or not np.isfinite(spearman):
        raise ValueError("Correlation is undefined for a constant sample.")
    return {"pearson": pearson, "spearman": spearman, "n": int(left.size)}


def _finite_pairs(frame: pl.DataFrame, x: str, y: str) -> pl.DataFrame:
    return frame.filter(pl.col(x).is_finite() & pl.col(y).is_finite())


def _fmt_corr(value: float) -> str:
    """Three decimals, or six when three would print ±1.000.

    Fitted ``s`` reproduces published fitness within ``identity_tolerance``
    (1e-6), and ``se_pooled`` tracks the published standard error, so both
    correlations can round to 1.000 without either column being correlated
    with itself. The extra digits keep that agreement readable.
    """
    text = fixed_label_text(value, 3)
    if text in {"1.000", "-1.000"}:
        return fixed_label_text(value, 6)
    return text


def _stats_line(stats: dict[str, float | int]) -> str:
    return (
        f"Pearson r = {_fmt_corr(float(stats['pearson']))}, "
        f"Spearman ρ = {_fmt_corr(float(stats['spearman']))}, "
        f"n = {count_label_text(stats['n'])} genotypes"
    )


def density_panel(
    frame: pl.DataFrame,
    *,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    panel_title: str,
    subtitle: str,
    width: int,
    height: int,
    point_size: float,
    hexbin_min: int,
    hexbin_bins: int,
    color: str,
    identity_line: bool = True,
    legend_orient: str = "right",
) -> alt.Chart:
    """One genotype panel: a fine hexbin density past ``hexbin_min`` rows, a scatter below.

    Each axis is cropped to the data extent (a 2 % pad), not to a shared square.
    ``hexbin_bins`` is the number of hexagons across the panel. The density colour
    scale runs from the lightest grey at the least-occupied bin to the darkest at
    the most-occupied. ``identity_line`` draws ``y = x`` clipped to the axes.
    ``legend_orient`` places the colour bar (``"right"`` outside the plot, or an
    inside corner such as ``"top-right"``).
    """
    xs = frame[x].to_numpy()
    ys = frame[y].to_numpy()
    x_low, x_high = data_domain(xs)
    y_low, y_high = data_domain(ys)
    x_scale = alt.Scale(domain=[x_low, x_high], nice=False)
    y_scale = alt.Scale(domain=[y_low, y_high], nice=False)
    layers: list[alt.Chart] = []
    if frame.height >= hexbin_min:
        bins, radius = hexbin_counts(
            xs,
            ys,
            x_domain=(x_low, x_high),
            y_domain=(y_low, y_high),
            width=width,
            height=height,
            bins_across=hexbin_bins,
        )
        observed = bins["n_genotypes"].to_numpy()
        cmin = float(np.min(observed))
        cmax = float(np.max(observed))
        if cmax == cmin:
            cmax = cmin + 1.0
        # ``n_genotypes`` is not named ``count``: that token is a Vega-Lite
        # aggregate, and a quantitative colour encoding of it would be merged
        # with the identity line, which has no such column.
        color_encoding = alt.Color(
            "n_genotypes:Q",
            title="Genotypes",
            scale=alt.Scale(
                domain=[cmin, cmax],
                range=[AGREEMENT_BIN_LIGHT, AGREEMENT_BIN_DARK],
                clamp=True,
            ),
            legend=alt.Legend(orient=legend_orient),
        )
        layers.append(
            hexagon_marks(
                bins,
                radius_px=radius,
                x_field=x,
                y_field=y,
                x_scale=x_scale,
                y_scale=y_scale,
                x_title=x_title,
                y_title=y_title,
                color=color_encoding,
            )
        )
        # Bins at the lowest count are single genotypes (or the thinnest bins).
        # Draw them as points so they stay visible on the light end of the scale.
        sparse = bins.filter(
            pl.col("n_genotypes") == pl.col("n_genotypes").min()
        ).rename({"hx": x, "hy": y})
        layers.append(
            alt.Chart(sparse)
            .mark_circle(size=point_size, color=color, opacity=0.9)
            .encode(
                x=alt.X(f"{x}:Q", scale=x_scale, title=x_title),
                y=alt.Y(f"{y}:Q", scale=y_scale, title=y_title),
            )
        )
    else:
        layers.append(
            alt.Chart(frame)
            .mark_circle(size=point_size, color=color, opacity=0.85)
            .encode(
                x=alt.X(f"{x}:Q", scale=x_scale, title=x_title),
                y=alt.Y(f"{y}:Q", scale=y_scale, title=y_title),
            )
        )
    if identity_line:
        low = min(x_low, y_low)
        high = max(x_high, y_high)
        layers.append(
            alt.Chart(pl.DataFrame({x: [low, high], y: [low, high]}))
            .mark_line(color=DIAGONAL_GREY, strokeWidth=1, strokeDash=[6, 4], clip=True)
            .encode(
                x=alt.X(f"{x}:Q", scale=x_scale, title=x_title),
                y=alt.Y(f"{y}:Q", scale=y_scale, title=y_title),
            )
        )
    # The identity line has no colour field. An independent colour scale keeps
    # the hexagons' domain on the finite bin counts.
    return (
        alt.layer(*layers)
        .resolve_scale(color="independent")
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(
                text=panel_title, subtitle=[subtitle], anchor="start"
            ),
        )
    )


def _panel(frame: pl.DataFrame, *, x: str, y: str, **kwargs: Any) -> alt.Chart:
    stats = correlations(frame[x].to_numpy(), frame[y].to_numpy())
    return density_panel(
        frame, x=x, y=y, subtitle=_stats_line(stats), identity_line=True, **kwargs
    )


def build_papkou_agreement_figure(
    payload: dict[str, Any], settings: dict[str, Any]
) -> alt.Chart:
    """Draw the two agreement panels from the matched genotypes."""
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][FIGURE_ID],
    }
    matched = matched_genotypes(payload["growth_rates"], payload["published"])
    if matched.is_empty():
        raise ValueError(
            "No genotypes matched between the fits and the published table."
        )
    width = int(appearance.get("panel_width", 360))
    height = int(appearance.get("panel_height", 360))
    font_size = int(appearance.get("font_size", 12))
    hexbin_min = int(appearance.get("hexbin_min", _HEXBIN_MIN))
    hexbin_bins = int(appearance.get("hexbin_bins", _HEXBIN_BINS))
    color = str(appearance.get("point_color", AGREEMENT_BIN_DARK))
    point_size = float(appearance.get("point_size", 36))
    fitness = _finite_pairs(matched, "fitted_s", "published_m")
    error = _finite_pairs(matched, "se_pooled", "published_m_se")
    if fitness.is_empty() or error.is_empty():
        raise ValueError("A panel has no finite matched genotypes.")
    panels = [
        _panel(
            fitness,
            x="fitted_s",
            y="published_m",
            x_title="Fitted s",
            y_title="Published fitness m",
            panel_title="A  Fitness",
            width=width,
            height=height,
            point_size=point_size,
            hexbin_min=hexbin_min,
            hexbin_bins=hexbin_bins,
            color=color,
        ),
        _panel(
            error,
            x="se_pooled",
            y="published_m_se",
            x_title="Fitted se_pooled",
            y_title="Published standard error m_se",
            panel_title="B  Standard error",
            width=width,
            height=height,
            point_size=point_size,
            hexbin_min=hexbin_min,
            hexbin_bins=hexbin_bins,
            color=color,
        ),
    ]
    chart = alt.hconcat(*panels, spacing=int(appearance.get("panel_gap", 28)))
    from ._title_flag import figure_heading

    fitness_stats = correlations(
        fitness["fitted_s"].to_numpy(), fitness["published_m"].to_numpy()
    )
    error_stats = correlations(
        error["se_pooled"].to_numpy(), error["published_m_se"].to_numpy()
    )
    heading = figure_heading(
        str(appearance.get("title", "Agreement with Papkou et al. 2023")),
        [
            "A  " + _stats_line(fitness_stats),
            "B  " + _stats_line(error_stats),
        ],
    )
    if heading is not None:
        chart = chart.properties(title=alt.TitleParams(**heading))
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size, grid=False)
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
        .configure_title(
            fontSize=int(appearance.get("heading_font_size", 16)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 12)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )
