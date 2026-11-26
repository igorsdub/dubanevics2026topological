"""Fitness against standard error, Papkou published beside our estimate.

Supplementary. Two panels side by side, drawn in the style of the fitness-against-
pooled-SE 2x2 of ``peak_uncertainty`` (grey raster cloud on shared axes, fitness
ticks at the domain ends and 0). Left: Papkou et al. (2023) published fitness ``s``
against their published standard error ``se`` (their ``m`` and ``m_se``). The
published table has no wild-type row, so no wild type is drawn there. Right: our
fitted ``s = r - r_wt`` against ``se_pooled``, with the wild type (``s = 0``, its own
pooled SE) as the repository's red star on that panel only. Both read the cached
growth-rate table and the published fitness table and refit nothing; sequence
identities are ``dna_sequence`` values.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import polars as pl

from ._charts import marker_config
from ._count_labels import count_label_text
from .palette import LIGHT_GREY
from .peak_uncertainty import _configured_domain, _end_ticks, _raster_cloud

FIGURE_ID = "fitness_vs_pooled_se"
PUBLISHED_TITLE = "Papkou et al. (2023), published"
OURS_TITLE = "Our estimate, corrected wild type"


def published_frame(published: pl.DataFrame) -> pl.DataFrame:
    """Published fitness and SE per measured genotype (no wild-type row), finite only."""
    missing = sorted({"dna_to", "s", "se"} - set(published.columns))
    if missing:
        raise ValueError(f"Published fitness table is missing columns: {missing}.")
    return published.select(
        pl.col("dna_to").cast(pl.String).alias("dna_sequence"),
        pl.col("s").cast(pl.Float64).alias("raw_fitness"),
        pl.col("se").cast(pl.Float64).alias("standard_error"),
    ).filter(pl.col("raw_fitness").is_finite() & pl.col("standard_error").is_finite())


def wild_type_sequence(published: pl.DataFrame) -> str:
    """The one ``dna_from`` of the published table: the wild type."""
    origins = published["dna_from"].unique().to_list()
    if len(origins) != 1:
        raise ValueError(
            f"Published table must measure against one wild type, found {len(origins)}."
        )
    return str(origins[0])


def our_frame(growth_rates: pl.DataFrame) -> pl.DataFrame:
    """Our fitness ``s`` and pooled SE per genotype, the wild type included."""
    missing = sorted({"dna_sequence", "s", "se_pooled"} - set(growth_rates.columns))
    if missing:
        raise ValueError(f"Growth-rate table is missing columns: {missing}.")
    return growth_rates.select(
        pl.col("dna_sequence").cast(pl.String),
        pl.col("s").cast(pl.Float64).alias("raw_fitness"),
        pl.col("se_pooled").cast(pl.Float64).alias("standard_error"),
    ).filter(pl.col("raw_fitness").is_finite() & pl.col("standard_error").is_finite())


def _panel(
    frame: pl.DataFrame,
    *,
    title: str,
    subtitle: str,
    wild_type: str | None,
    appearance: dict[str, Any],
    fitness_domain: list[float],
    se_domain: list[float],
    show_y_title: bool,
) -> alt.Chart:
    size = int(appearance.get("panel_size", 300))
    font_size = int(appearance.get("font_size", 15))
    grid = bool(appearance.get("grid", False))
    x_title = appearance.get("standard_error_title", "Pooled fitness SE")
    y_title = appearance.get("fitness_axis_title", "Fitness (WT = 0)")
    x_scale = alt.Scale(domain=se_domain, nice=False, zero=False)
    y_scale = alt.Scale(domain=fitness_domain, nice=False, zero=False)
    x_axis = alt.Axis(
        title=x_title, tickCount=4, format=".2f", domain=False, ticks=True, grid=grid
    )
    y_axis = alt.Axis(
        title=y_title if show_y_title else None,
        values=_end_ticks(
            fitness_domain, float(appearance.get("fitness_tick_step", 0.1))
        ),
        format=".1f",
        domain=False,
        ticks=True,
        grid=grid,
    )
    cloud = frame
    if wild_type is not None:
        cloud = cloud.filter(pl.col("dna_sequence") != wild_type)
    url = _raster_cloud(
        cloud,
        se_domain=se_domain,
        fitness_domain=fitness_domain,
        size=size,
        color=str(appearance.get("sequence_color", LIGHT_GREY)),
        opacity=float(appearance.get("sequence_opacity", 0.65)),
        point_size=float(appearance.get("sequence_point_size", 12)),
    )
    encode = {
        "x": alt.X("standard_error:Q", title=x_title, axis=x_axis, scale=x_scale),
        "y": alt.Y("raw_fitness:Q", title=y_title, axis=y_axis, scale=y_scale),
    }
    layers: list[alt.Chart] = [
        alt.Chart(
            alt.Data(
                values=[
                    {
                        "standard_error": se_domain[0],
                        "raw_fitness": fitness_domain[1],
                        "url": url,
                    }
                ]
            )
        )
        .mark_image(
            width=size, height=size, align="left", baseline="top", aspect=False
        )
        .encode(url="url:N", **encode)
    ]
    if wild_type is not None:
        wt_rows = frame.filter(pl.col("dna_sequence") == wild_type)
        if wt_rows.is_empty():
            raise ValueError("The growth-rate table has no wild-type row.")
        star = marker_config(appearance, "wild_type", default_shape="star")
        points = alt.Chart(
            alt.Data(
                values=[
                    {
                        "dna_sequence": wild_type,
                        "raw_fitness": float(wt_rows["raw_fitness"][0]),
                        "standard_error": float(wt_rows["standard_error"][0]),
                        "label": "Wild type",
                    }
                ]
            )
        ).encode(**encode)
        layers.append(
            points.mark_point(
                shape=star.shape,
                color=star.color,
                size=star.size,
                filled=True,
                opacity=star.opacity,
            ).encode(tooltip=[alt.Tooltip("dna_sequence:N", title="DNA")])
        )
        layers.append(
            points.mark_text(
                dx=12, align="left", color=star.label_color, fontSize=font_size
            ).encode(text="label:N")
        )
    return alt.layer(*layers).properties(
        width=size,
        height=size,
        title=alt.TitleParams(
            text=title,
            subtitle=[subtitle],
            fontSize=int(appearance.get("heading_font_size", 18)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 14)),
            fontWeight="bold",
            anchor="start",
            offset=int(appearance.get("title_offset", 10)),
        ),
    )


def build_fitness_vs_pooled_se_figure(
    payload: dict[str, Any], settings: dict[str, Any]
) -> alt.Chart:
    """Draw Papkou published (left) beside our estimate with the wild type (right).

    ``payload`` has ``growth_rates`` (our table, wild type included) and
    ``published`` (``load_fitness_wt``). The axis ranges are this figure's own
    ``fitness_domain`` and ``standard_error_domain``; absent, they are the ranges
    of ``peak_uncertainty``, so the two figures are drawn to one scale.
    """
    figures = settings["figures"]["figures"]
    own = figures[FIGURE_ID]
    fallback = {
        key: figures.get("peak_uncertainty", {}).get(key)
        for key in ("fitness_domain", "standard_error_domain")
    }
    appearance = {
        **settings["figures"].get("rendering", {}),
        **{key: value for key, value in fallback.items() if value is not None},
        **own,
    }
    fitness_domain = _configured_domain(appearance, "fitness_domain")
    se_domain = _configured_domain(appearance, "standard_error_domain")
    published = payload["published"]
    wild_type = wild_type_sequence(published)
    theirs = published_frame(published)
    ours = our_frame(payload["growth_rates"])
    if theirs.is_empty() or ours.is_empty():
        raise ValueError("A panel has no finite genotypes.")
    left = _panel(
        theirs,
        title=PUBLISHED_TITLE,
        subtitle=(
            "Published fitness m and standard error m_se, "
            f"{count_label_text(theirs.height)} genotypes"
        ),
        wild_type=None,
        appearance=appearance,
        fitness_domain=fitness_domain,
        se_domain=se_domain,
        show_y_title=True,
    )
    right = _panel(
        ours,
        title=OURS_TITLE,
        subtitle=(
            "Fitted s and pooled SE, "
            f"{count_label_text(ours.height - 1)} genotypes and the wild type"
        ),
        wild_type=wild_type,
        appearance=appearance,
        fitness_domain=fitness_domain,
        se_domain=se_domain,
        show_y_title=False,
    )
    chart = alt.hconcat(left, right, spacing=int(appearance.get("panel_gap", 30)))
    chart = chart.resolve_scale(x="shared", y="shared").resolve_axis(y="independent")
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(appearance.get("font_size", 15)),
            titleFontSize=int(appearance.get("font_size", 15)),
            labelPadding=int(appearance.get("axis_label_padding", 4)),
            titlePadding=int(appearance.get("axis_title_padding", 8)),
            grid=bool(appearance.get("grid", False)),
        )
        .configure_title(fontSize=int(appearance.get("font_size", 15)), anchor="start")
        .configure_view(strokeWidth=0)
    )

