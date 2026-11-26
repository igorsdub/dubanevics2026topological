"""Full-graph raw-fitness histogram with the leading saddle band's landmarks.

``full_tree_fitness_landmarks`` bins the raw fitness of every Full-graph
sequence, shades the leading saddle band (the closed interval between the two
trailing saddles) and draws one vertical rule per landmark with a label. The
landmark values, the band bounds and the counts are read from the
``leading-saddle-band`` stage's tables (``analysis/leading_saddle_band.py``);
nothing about them is recomputed here. Only the histogram bins come from the
population, which the stage's tables do not hold. Every printed number (the landmark
values and the subtitle's counts) is the manuscript quantity registry's display text,
asked for by key (``payload["manuscript_quantities"]``).

Each label sits on its own row, ordered by fitness from the top, at its rule, so
landmarks a few hundredths apart do not overprint. The median is a dotted rule and
the functional thresholds dashed ones, the repository's line-style rule; the other
landmarks are solid. Row, colour and text settings are in ``config/figures.toml``.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ._leaf_data import fitness_axis
from ._quantities import ManuscriptQuantities
from .labels import PAPKOU_THRESHOLD, TOPOGRAPHER_THRESHOLD
from .palette import BAND_GREY, INK_GREY, MID_GREY

FIGURE_ID = "full_tree_fitness_landmarks"
HISTOGRAM_BINS = 200
# Landmarks drawn dashed (the functional thresholds) and dotted (the median).
DASHED = frozenset({"published_functional_threshold", "candidate_functional_threshold"})
DOTTED = frozenset({"median"})
DEFAULT_LABELS = {
    "global_minimum": "Global minimum",
    "split_trailing_saddle": "Split-tree trailing saddle",
    "join_trailing_saddle": "Join-tree trailing saddle",
    "median": "Median",
    "split_leading_saddle": "Split-tree leading saddle",
    "join_leading_saddle": "Join-tree leading saddle",
    "published_functional_threshold": PAPKOU_THRESHOLD,
    "candidate_functional_threshold": TOPOGRAPHER_THRESHOLD,
}


def histogram_table(fitness: pl.Series, bins: int) -> pl.DataFrame:
    """Equal-width counts of ``fitness`` over its own range: ``start``, ``end``, ``count``."""
    if bins < 1:
        raise ValueError("The histogram needs at least one bin.")
    values = fitness.to_numpy()
    counts, edges = np.histogram(values, bins=bins)
    return pl.DataFrame(
        {
            "start": edges[:-1].astype(float),
            "end": edges[1:].astype(float),
            "count": counts.astype(int),
        }
    )


def label_rows(
    landmarks: pl.DataFrame,
    labels: dict[str, str],
    quantities: ManuscriptQuantities,
):
    """One label per landmark, ranked by fitness (ties keep table order): the row index.

    The label's number is the registry's display text for
    ``leading_saddle_band.<landmark>.fitness``, printed as given.
    """
    order = sorted(range(landmarks.height), key=lambda i: (landmarks["fitness"][i], i))
    rows = []
    for row, index in enumerate(order):
        key = landmarks["landmark"][index]
        fitness = landmarks["fitness"][index]
        rows.append(
            {
                "landmark": key,
                "fitness": fitness,
                "label": (
                    f"{labels.get(key, key)} "
                    f"{quantities.display(f'leading_saddle_band.{key}.fitness')}"
                ),
                "row": row,
            }
        )
    return pl.DataFrame(rows)


def build_fitness_landmarks_figure(payload: dict[str, Any], settings: dict[str, Any]):
    """Compose the histogram, the shaded band, the landmark rules and their labels."""
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][FIGURE_ID],
    }
    stage = payload["leading_saddle_band"]
    population = payload["full_fitness"]["raw_fitness"]
    summary = stage.summary
    if population.len() != summary["total_sequence_count"]:
        raise ValueError(
            "The Full-graph population does not match the stage's sequence count."
        )
    width = int(appearance.get("width", 640))
    height = int(appearance.get("height", 320))
    row_height = float(appearance.get("label_row_height", 14))
    font_size = int(appearance.get("label_font_size", 11))
    quantities = payload["manuscript_quantities"]
    bins = int(appearance.get("histogram_bins", HISTOGRAM_BINS))
    labels = {**DEFAULT_LABELS, **appearance.get("labels", {})}

    table = histogram_table(population, bins)
    marks = label_rows(stage.landmarks, labels, quantities).with_columns(
        pl.col("landmark").is_in(sorted(DASHED)).alias("dashed"),
        pl.col("landmark").is_in(sorted(DOTTED)).alias("dotted"),
    )
    label_area = row_height * marks.height
    if label_area >= height:
        raise ValueError("The label rows leave no room for the histogram.")
    top = float(table["count"].max()) * height / (height - label_area)
    low = min(float(population.min()), float(marks["fitness"].min()))
    high = max(float(population.max()), float(marks["fitness"].max()))
    domain, ticks = fitness_axis(low, high, appearance)
    x_scale = alt.Scale(domain=domain, nice=False)
    x_axis = alt.Axis(
        title=appearance.get("x_title", "Fitness (WT = 0)"),
        values=ticks,
        grid=False,
    )

    bars = (
        alt.Chart(table)
        .mark_bar(color=appearance.get("histogram_color", MID_GREY))
        .encode(
            x=alt.X("start:Q", scale=x_scale, axis=x_axis),
            x2="end:Q",
            y=alt.Y(
                "count:Q",
                scale=alt.Scale(domain=[0, top], nice=False),
                axis=alt.Axis(
                    title=appearance.get("y_title", "Count"),
                    format="~s",
                    grid=False,
                ),
            ),
        )
    )
    band = (
        alt.Chart(
            pl.DataFrame(
                {"lower": [summary["band_lower"]], "upper": [summary["band_upper"]]}
            )
        )
        .mark_rect(
            color=appearance.get("band_color", BAND_GREY),
            opacity=float(appearance.get("band_opacity", 0.35)),
        )
        .encode(
            x=alt.X("lower:Q", scale=x_scale),
            x2="upper:Q",
            y=alt.value(0),
            y2=alt.value(height),
        )
    )
    rule_color = appearance.get("rule_color", INK_GREY)
    rule_width = float(appearance.get("rule_width", 1.0))
    rule_layers = [
        alt.Chart(group)
        .mark_rule(color=rule_color, strokeWidth=rule_width, strokeDash=dash)
        .encode(x=alt.X("fitness:Q", scale=x_scale))
        for dash, group in (
            ([1, 0], marks.filter(~pl.col("dashed") & ~pl.col("dotted"))),
            (appearance.get("threshold_dash", [5, 4]), marks.filter(pl.col("dashed"))),
            (appearance.get("median_dash", [1, 3]), marks.filter(pl.col("dotted"))),
        )
        if group.height
    ]
    flip_at = domain[0] + float(appearance.get("label_flip_fraction", 0.8)) * (
        domain[1] - domain[0]
    )
    labelled = marks.with_columns((pl.col("row") * row_height).alias("row_px"))
    # A label that would run past the right edge is written to the left of its rule.
    text_layers = [
        alt.Chart(group)
        .mark_text(
            align=align,
            baseline="top",
            dx=3 if align == "left" else -3,
            fontSize=font_size,
            color=appearance.get("label_color", INK_GREY),
        )
        .encode(
            x=alt.X("fitness:Q", scale=x_scale),
            y=alt.Y("row_px:Q", scale=None),
            text="label:N",
        )
        for align, group in (
            ("left", labelled.filter(pl.col("fitness") < flip_at)),
            ("right", labelled.filter(pl.col("fitness") >= flip_at)),
        )
        if group.height
    ]
    chart = alt.layer(bars, band, *rule_layers, *text_layers).properties(
        width=width,
        height=height,
        title=alt.TitleParams(
            text="Full graph",
            subtitle=[_subtitle(quantities)],
        ),
    )
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(appearance.get("axis_label_font_size", font_size)),
            titleFontSize=int(appearance.get("font_size", 15)),
        )
        .configure_title(
            fontSize=int(appearance.get("heading_font_size", 18)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


def _subtitle(quantities: ManuscriptQuantities) -> str:
    """The band's counts line, every number the registry's display text."""
    inside = quantities.display("leading_saddle_band.median_inside_band")
    if inside not in {"true", "false"}:
        raise ValueError(
            "leading_saddle_band.median_inside_band must display true or false, "
            f"got {inside!r}."
        )
    return (
        f"{quantities.display('leading_saddle_band.total_sequence_count')} sequences"
        " · leading saddle band "
        f"{quantities.display('leading_saddle_band.band.count')} "
        f"({quantities.display('leading_saddle_band.band.percentage')}%)"
        f" · median {'inside' if inside == 'true' else 'outside'} the band"
    )
