"""Agreement of our growth-rate fits with Papkou et al. 2023.

Supplementary S1. Four panels in 2 rows, one point per genotype matched on
``dna_sequence``, styled as ``fitness_se_correlation``. Top row: fitness against
standard error for Papkou et al. (2023) and BarSeqFit. Bottom row: agreement of
fitness plots Papkou's published fitness (column ``s`` of
``fitness_wt``, their ``m``) against our fitted ``s``; agreement of standard error plots Papkou's
published standard error (column ``se``, their ``m_se``) against our
``se_pooled``. Both draw ``y = x``. A panel with at least
``hexbin_min`` genotypes is a fine hexagonal binned density (axes cropped to the
data); a smaller panel is a scatter.
The one genotype-count colour bar (logarithmic, 1 to the largest bin count)
serves all four panels. BarSeqFit's wild type (its growth-rate row) is a red star
in panel B alone, with a "Wild type" legend entry under the bar; no star is drawn
when that row is absent.
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
    LIGHT_GREY,
)

FIGURE_ID = "papkou_agreement"
# The file stem of the one output (the registry id stays ``papkou_agreement``).
STEM = "papkou2023_fitness_and_se_barseqfit_agreement"
# fitness_wt stores Papkou's published m and m_se under these names.
PUBLISHED_FITNESS = "s"
PUBLISHED_SE = "se"
PAPKOU_LABEL = "Papkou et al. (2023)"
BARSEQFIT_LABEL = "BarSeqFit"
WILD_TYPE_LABEL = "Wild type"
# Panel size, fonts and axis styling follow the fitness_se_correlation figure.
STYLE_SOURCE = "peak_uncertainty"
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


def _bins_for(
    frame: pl.DataFrame,
    *,
    x: str,
    y: str,
    width: int,
    height: int,
    hexbin_min: int,
    hexbin_bins: int,
) -> tuple[pl.DataFrame, float] | None:
    """Hexagon bins and radius of a panel, or ``None`` for a scatter panel."""
    if frame.height < hexbin_min:
        return None
    return hexbin_counts(
        frame[x].to_numpy(),
        frame[y].to_numpy(),
        x_domain=data_domain(frame[x].to_numpy()),
        y_domain=data_domain(frame[y].to_numpy()),
        width=width,
        height=height,
        bins_across=hexbin_bins,
    )


def count_range(
    panels: list[tuple[pl.DataFrame, str, str]],
    *,
    width: int,
    height: int,
    hexbin_min: int,
    hexbin_bins: int,
) -> tuple[float, float] | None:
    """Smallest and largest bin count over several ``(frame, x, y)`` panels.

    This is the common colour domain of a figure whose panels share one bar.
    ``None`` when no panel is binned.
    """
    counts: list[np.ndarray] = []
    for frame, x, y in panels:
        found = _bins_for(
            frame,
            x=x,
            y=y,
            width=width,
            height=height,
            hexbin_min=hexbin_min,
            hexbin_bins=hexbin_bins,
        )
        if found is not None:
            counts.append(found[0]["n_genotypes"].to_numpy())
    if not counts:
        return None
    low = float(min(c.min() for c in counts))
    high = float(max(c.max() for c in counts))
    return (low, high if high > low else low + 1.0)


def _log_ticks(high: float) -> list[int]:
    """1, 10, 100, ... up to the largest power of ten not above ``high``."""
    top = int(np.floor(np.log10(max(high, 1.0))))
    return [10**power for power in range(top + 1)]


# "Wild type" entry under the colour bar: pixels from the panel's right and top edges.
WT_LEGEND_X = 118
WT_LEGEND_Y = 88
WT_LEGEND_FONT = 15
# Length in pixels of the logarithmic colour bar.
LOG_BAR_LENGTH = 140
# Tick values of the bar (those not above the busiest hexagon are drawn) and their
# label expression: thousands read as 1K and 10K.
LOG_TICKS = (1, 10, 100, 1000, 10000)
LOG_TICK_LABEL_EXPR = "datum.value >= 1000 ? datum.value / 1000 + 'K' : datum.value"


def density_panel(
    frame: pl.DataFrame,
    *,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    panel_title: str,
    subtitle: str | list[str],
    width: int,
    height: int,
    point_size: float,
    hexbin_min: int,
    hexbin_bins: int,
    color: str,
    identity_line: bool = True,
    legend_orient: str = "right",
    color_domain: tuple[float, float] | None = None,
    show_legend: bool = True,
    wild_type: tuple[float, float] | None = None,
    wild_type_marker: Any = None,
    show_wild_type_legend: bool = False,
    log_color: bool = False,
) -> alt.Chart:
    """One genotype panel: a fine hexbin density past ``hexbin_min`` rows, a scatter below.

    Each axis is cropped to the data extent (a 2 % pad), not to a shared square.
    ``hexbin_bins`` is the number of hexagons across the panel. The density colour
    scale runs from the lightest grey at the least-occupied bin to the darkest at
    the most-occupied; ``color_domain`` overrides that range (a figure whose panels
    share one bar passes the common range) and ``show_legend=False`` hides this
    panel's bar; ``log_color`` makes the colour scale logarithmic from 1. ``identity_line`` draws ``y = x`` clipped to the axes.
    ``wild_type`` is the ``(x, y)`` of the wild-type point: it widens the axis
    extent, and is drawn last (above the density) as the ``wild_type_marker`` star
    (a ``TreeMarkerConfig``); ``show_wild_type_legend`` adds the "Wild type" legend
    entry, which stacks under the colour bar in the same panel. ``legend_orient`` places the colour bar (``"right"`` outside the plot, or an
    inside corner such as ``"top-right"``).
    """
    xs = frame[x].to_numpy()
    ys = frame[y].to_numpy()
    if wild_type is not None:
        xs_domain = np.append(xs, wild_type[0])
        ys_domain = np.append(ys, wild_type[1])
    else:
        xs_domain, ys_domain = xs, ys
    x_low, x_high = data_domain(xs_domain)
    y_low, y_high = data_domain(ys_domain)
    x_scale = alt.Scale(domain=[x_low, x_high], nice=False)
    y_scale = alt.Scale(domain=[y_low, y_high], nice=False)
    layers: list[alt.Chart] = []
    found = _bins_for(
        frame,
        x=x,
        y=y,
        width=width,
        height=height,
        hexbin_min=hexbin_min,
        hexbin_bins=hexbin_bins,
    )
    if found is not None:
        bins, radius = found
        observed = bins["n_genotypes"].to_numpy()
        cmin = float(np.min(observed))
        cmax = float(np.max(observed))
        if color_domain is not None:
            cmin, cmax = color_domain
        if log_color:
            cmin = 1.0
        if cmax <= cmin:
            cmax = cmin + 1.0
        # ``n_genotypes`` is not named ``count``: that token is a Vega-Lite
        # aggregate, and a quantitative colour encoding of it would be merged
        # with the identity line, which has no such column.
        legend_kwargs: dict[str, Any] = {}
        if log_color:
            legend_kwargs = {
                "values": [tick for tick in LOG_TICKS if cmin <= tick <= cmax],
                "labelExpr": LOG_TICK_LABEL_EXPR,
                "gradientLength": LOG_BAR_LENGTH,
                "labelOverlap": False,
            }
        color_encoding = alt.Color(
            "n_genotypes:Q",
            title="Genotypes",
            scale=alt.Scale(
                type="log" if log_color else "linear",
                domain=[cmin, cmax],
                range=[AGREEMENT_BIN_LIGHT, AGREEMENT_BIN_DARK],
                clamp=True,
            ),
            legend=(
                alt.Legend(orient=legend_orient, **legend_kwargs)
                if show_legend
                else None
            ),
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
    if wild_type is not None and wild_type_marker is not None:
        star = wild_type_marker
        layers.append(
            alt.Chart(pl.DataFrame({x: [wild_type[0]], y: [wild_type[1]]}))
            .mark_point(
                shape=star.shape,
                color=star.color,
                size=star.size,
                filled=True,
                opacity=star.opacity,
            )
            .encode(
                x=alt.X(f"{x}:Q", scale=x_scale, title=x_title),
                y=alt.Y(f"{y}:Q", scale=y_scale, title=y_title),
            )
        )
        if show_wild_type_legend:
            # Under the colour bar (top right): a star and its label at fixed pixel
            # offsets from the panel's top-right corner.
            entry = alt.Chart(alt.Data(values=[{}]))
            layers.append(
                entry.mark_point(
                    shape=star.shape,
                    color=star.color,
                    size=star.size * 0.5,
                    filled=True,
                    opacity=1.0,
                ).encode(
                    x=alt.value(width - WT_LEGEND_X), y=alt.value(WT_LEGEND_Y)
                )
            )
            layers.append(
                entry.mark_text(
                    align="left",
                    baseline="middle",
                    dx=10,
                    color=star.label_color,
                    fontSize=WT_LEGEND_FONT,
                ).encode(
                    x=alt.value(width - WT_LEGEND_X),
                    y=alt.value(WT_LEGEND_Y),
                    text=alt.value(WILD_TYPE_LABEL),
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
                text=panel_title,
                subtitle=[subtitle] if isinstance(subtitle, str) else list(subtitle),
                anchor="start",
            ),
        )
    )


def apply_density_style(
    chart: Any, appearance: dict[str, Any], *, y_axis_extent: int | None = None
) -> Any:
    """Fonts, axis and title styling shared by the fitness-against-SE figures."""
    font_size = int(appearance.get("font_size", 15))
    styled = (
        chart.configure(
            font=appearance.get(
                "font_family", "Arial, Liberation Sans, Helvetica, sans-serif"
            )
        )
        .configure_axis(
            labelFontSize=font_size,
            titleFontSize=font_size,
            labelPadding=int(appearance.get("axis_label_padding", 4)),
            titlePadding=int(appearance.get("axis_title_padding", 8)),
            grid=bool(appearance.get("grid", False)),
            domain=False,
            ticks=True,
        )
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
        .configure_title(
            fontSize=int(appearance.get("heading_font_size", 18)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )
    if y_axis_extent is not None:
        # Equal left margins: panels of a grid start their plot areas at the same x.
        styled = styled.configure_axisY(
            minExtent=y_axis_extent, maxExtent=y_axis_extent
        )
    return styled


def wild_type_point(
    growth_rates: pl.DataFrame, published: pl.DataFrame
) -> tuple[float, float] | None:
    """BarSeqFit's wild type as ``(se_pooled, s)``, or ``None`` when it has no finite row.

    The wild type is the one ``dna_from`` of the published ``fitness_wt`` table;
    its values are its row of the growth-rate table. The published table has no
    wild-type measurement, so nothing is looked up there.
    """
    from dubanevics2026topological.figures.fitness_pooled_se_pair import (
        wild_type_sequence,
    )

    wild_type = wild_type_sequence(published)
    row = growth_rates.filter(pl.col("dna_sequence").cast(pl.String) == wild_type)
    if row.is_empty():
        return None
    point = (float(row["se_pooled"][0]), float(row["s"][0]))
    return point if np.isfinite(point).all() else None


def build_papkou_agreement_figure(
    payload: dict[str, Any], settings: dict[str, Any]
) -> alt.Chart:
    """Draw the 2 x 2 agreement figure from the matched genotypes.

    Top row: fitness against standard error as ``fitness_se_correlation`` draws it, for
    Papkou et al. (2023) (left) and BarSeqFit (right). Bottom row: the agreement of the
    fitness and of the standard error, with their Pearson and Spearman correlations.
    Style, panel size and settings are those of ``fitness_se_correlation``; one
    logarithmic colour scale (and one bar, in the top-right panel) covers every panel.
    """
    figure_tables = settings["figures"]["figures"]
    appearance = {
        **settings["figures"].get("rendering", {}),
        **figure_tables.get(STYLE_SOURCE, {}),
        **figure_tables.get("fitness_se_correlation", {}),
        **figure_tables[FIGURE_ID],
    }
    from dubanevics2026topological.figures._charts import marker_config

    wt_ours = wild_type_point(payload["growth_rates"], payload["published"])
    wt_marker = marker_config(appearance, "wild_type", default_shape="star")
    matched = matched_genotypes(payload["growth_rates"], payload["published"])
    if matched.is_empty():
        raise ValueError(
            "No genotypes matched between the fits and the published table."
        )
    size = int(appearance.get("panel_size", 300))
    hexbin_min = int(appearance.get("hexbin_min", _HEXBIN_MIN))
    hexbin_bins = int(appearance.get("hexbin_bins", _HEXBIN_BINS))
    color = str(appearance.get("point_color", LIGHT_GREY))
    point_size = float(appearance.get("point_size", 12))
    fitness = _finite_pairs(matched, "fitted_s", "published_m")
    error = _finite_pairs(matched, "se_pooled", "published_m_se")
    se_papkou = _finite_pairs(matched, "published_m_se", "published_m")
    se_ours = _finite_pairs(matched, "se_pooled", "fitted_s")
    if min(fitness.height, error.height, se_papkou.height, se_ours.height) == 0:
        raise ValueError("A panel has no finite matched genotypes.")
    if min(se_papkou.height, se_ours.height) < 2:
        raise ValueError("Fewer than two genotypes have a finite fitness and SE.")
    specs: list[dict[str, Any]] = [
        {
            "frame": se_papkou,
            "x": "published_m_se",
            "y": "published_m",
            "x_title": f"{PAPKOU_LABEL} standard error m_se",
            "y_title": f"{PAPKOU_LABEL} fitness m",
            "panel_title": PAPKOU_LABEL,
            "identity": False,
        },
        {
            "frame": se_ours,
            "x": "se_pooled",
            "y": "fitted_s",
            "x_title": f"{BARSEQFIT_LABEL} pooled standard error",
            "y_title": f"{BARSEQFIT_LABEL} fitness s",
            "panel_title": BARSEQFIT_LABEL,
            "identity": False,
        },
        {
            "frame": fitness,
            "x": "fitted_s",
            "y": "published_m",
            "x_title": f"{BARSEQFIT_LABEL} fitness s",
            "y_title": f"{PAPKOU_LABEL} fitness m",
            "panel_title": "Fitness agreement",
            "identity": True,
        },
        {
            "frame": error,
            "x": "se_pooled",
            "y": "published_m_se",
            "x_title": f"{BARSEQFIT_LABEL} pooled standard error",
            "y_title": f"{PAPKOU_LABEL} standard error m_se",
            "panel_title": "Standard error agreement",
            "identity": True,
        },
    ]
    # One colour scale for every panel: genotypes per hexagon over all four, on a
    # logarithmic scale from one genotype to the busiest hexagon.
    found = count_range(
        [(item["frame"], item["x"], item["y"]) for item in specs],
        width=size,
        height=size,
        hexbin_min=hexbin_min,
        hexbin_bins=hexbin_bins,
    )
    domain = (1.0, found[1]) if found is not None else None
    wt_in_matched = False
    if wt_ours is not None:
        from dubanevics2026topological.figures.fitness_pooled_se_pair import (
            wild_type_sequence,
        )

        wt_in_matched = bool(
            (se_ours["dna_sequence"] == wild_type_sequence(payload["published"])).any()
        )
    panels = []
    stats_lines = []
    for index, item in enumerate(specs):
        frame = item["frame"]
        if item["identity"]:
            stats = correlations(
                frame[item["x"]].to_numpy(), frame[item["y"]].to_numpy()
            )
            subtitle = [
                f"Pearson r = {_fmt_corr(float(stats['pearson']))}, "
                f"Spearman ρ = {_fmt_corr(float(stats['spearman']))}",
                f"n = {count_label_text(stats['n'])} genotypes",
            ]
            stats_lines.append(f"{item['panel_title']}: {_stats_line(stats)}")
        else:
            # Top row: the count only. The BarSeqFit panel also shows the wild type
            # (a star), so its count is the matched genotypes plus the wild-type row.
            count = frame.height
            if index == 1 and wt_ours is not None and not wt_in_matched:
                count += 1
            subtitle = [f"n = {count_label_text(count)} genotypes"]
        panels.append(
            density_panel(
                frame,
                x=item["x"],
                y=item["y"],
                x_title=item["x_title"],
                y_title=item["y_title"],
                panel_title=item["panel_title"],
                subtitle=subtitle,
                width=size,
                height=size,
                point_size=point_size,
                hexbin_min=hexbin_min,
                hexbin_bins=hexbin_bins,
                color=color,
                identity_line=item["identity"],
                legend_orient=str(appearance.get("colorbar_orient", "top-right")),
                color_domain=domain,
                show_legend=(index == 1),
                wild_type=wt_ours if index == 1 else None,
                wild_type_marker=wt_marker,
                show_wild_type_legend=(index == 1 and wt_ours is not None),
                log_color=True,
            )
        )
    gap = int(appearance.get("panel_gap", 30))
    # One 2-column grid (not rows of hconcats) so Vega aligns every column and row.
    chart = alt.concat(*panels, columns=2, spacing=gap).resolve_scale(
        x="independent", y="independent", color="independent"
    )
    from ._title_flag import figure_heading

    heading = figure_heading(
        str(appearance.get("title", "Agreement with Papkou et al. (2023)")),
        stats_lines,
    )
    if heading is not None:
        chart = chart.properties(title=alt.TitleParams(**heading))
    return apply_density_style(
        chart, appearance, y_axis_extent=int(appearance.get("y_axis_extent", 110))
    )
