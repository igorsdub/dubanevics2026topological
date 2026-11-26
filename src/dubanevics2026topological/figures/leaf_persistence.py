"""Persistence diagram of the nominal split tree with one marker per leaf.

``published_leaf_persistence_band`` and ``candidate_leaf_persistence_band`` draw the
same diagram for the BY band-simplified tree (D10 saved tree, 29 peaks for the
Candidate graph).

``published_leaf_persistence`` and ``candidate_leaf_persistence``
draw the ordinary H0 diagram of the unsimplified split tree of one clamped
landscape: x is the leaf's paired death fitness (``death_scalar`` of the persistence
table, never the adjacent saddle), y its birth fitness (the peak), and the diagonal
``y = x`` is drawn. Marker area is proportional to the hill's sequence count (peak
plus the members of its terminal branch); every marker is filled, singleton hills
included (no hollow markers, GLOSSARY Figure marker rules). The global maximum's
pair, closed at the global minimum by convention, is a filled black triangle with a
label. Colour follows ADR 0012: wild type red, stop-codon peak
crimson, every other peak mid-grey.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import polars as pl

from ._branch_count_style import branch_count_opacity
from ._charts import (
    FIVE_POINT_STAR,
    peak_annotation_kinds,
    peak_annotation_style,
    persistence_legend_layers,
)
from ._count_labels import count_label_text
from ._leaf_data import (
    figure_graph,
    fitness_axis,
    leaf_tables,
    with_marker_area,
)
from .palette import DIAGONAL_GREY, EMPHASIS_BLACK, PEAK_GREY

SUFFIX = "leaf_persistence"
BAND_SUFFIX = "leaf_persistence_band"
_NOTES = {"wild_type": "Wild type", "stop_codon": "Stop-codon peak"}


def build_leaf_persistence_figure(
    payload: dict[str, Any], settings: dict[str, Any], figure_id: str
) -> Any:
    """Build one graph's leaf persistence diagram from the comparison payload.

    The subtitle's counts are the registry's display text
    (``payload["manuscript_quantities"]``).
    """
    slug, key, heading = figure_graph(figure_id, SUFFIX)
    quantities = payload["manuscript_quantities"]
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][figure_id],
    }
    tables = leaf_tables(
        payload["landscapes"][key], key=key, pairs=payload["annotations"][key].pairs
    )
    leaves = with_marker_area(
        tables.leaves,
        max_area=float(appearance.get("max_marker_area", 400)),
        min_area=float(appearance.get("min_marker_area", 12)),
    )
    subtitle = [
        f"Papkou et al. (2023) · {quantities.display(f'{slug}_graph.peak_count')} leaves",
        (
            f"{quantities.display(f'annotations.{slug}.singleton_peak_count')} "
            "singleton hills · largest hill "
            f"{quantities.display(f'{slug}_graph.largest_hill_sequence_count')} "
            "sequences"
        ),
    ]
    return build_from_leaves(leaves, heading, appearance, subtitle)


def band_subtitle(panel: dict[str, Any], leaves: pl.DataFrame) -> list[str]:
    """The two subtitle lines of a band-tree leaf figure, counted from its leaf table."""
    singletons = int(leaves["is_singleton"].sum())
    return [
        (
            f"Papkou et al. (2023) · {count_label_text(leaves.height)} leaves · "
            f"bootstrap band quantile {panel['band_quantile']}, "
            f"{count_label_text(panel['draws'])} draws"
        ),
        (
            f"{count_label_text(singletons)} singleton hills · largest hill "
            f"{count_label_text(int(leaves['member_count'].max()))} sequences"
        ),
    ]


def band_footer(figure_id: str, payload: dict[str, Any]) -> list[str]:
    """The footer lines of a band-tree leaf figure: its graph, BY band tree, band settings."""
    slug, _, heading = figure_graph(figure_id, BAND_SUFFIX)
    panel = payload["panels"][slug]
    return [
        f"{heading}: BY band-simplified tree",
        (
            f"Bootstrap band quantile {panel['band_quantile']} · "
            f"{count_label_text(panel['draws'])} draws"
        ),
    ]


def build_leaf_persistence_band_figure(
    payload: dict[str, Any], settings: dict[str, Any], figure_id: str
) -> Any:
    """Build one graph's leaf persistence diagram of its BY band-simplified tree.

    ``payload`` is ``load_band_leaf_inputs``' result: per graph slug the saved band
    tree's leaf tables. The counts are those of the tables, not the manuscript
    registry (it has no entry for the D10 band tree).
    """
    slug, _, heading = figure_graph(figure_id, BAND_SUFFIX)
    panel = payload["panels"][slug]
    appearance = {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][figure_id],
    }
    max_area = float(appearance.get("max_marker_area", 400))
    min_area = float(appearance.get("min_marker_area", 12))
    # The band tree's global-maximum hill holds most of the graph, so scaling every
    # marker to it would shrink the rest to the floor. The area scale is set by the
    # largest other hill; the global pair's own marker is capped at ``max_area``.
    all_leaves = panel["leaf_tables"].leaves
    others = all_leaves.filter(~pl.col("is_global_pair"))
    reference = others if not others.is_empty() else all_leaves
    top = int(reference["member_count"].max())
    leaves = all_leaves.with_columns(
        pl.Series(
            "marker_area",
            [
                min(max(max_area * count / top, min_area), max_area)
                for count in all_leaves["member_count"].to_list()
            ],
            dtype=pl.Float64,
        )
    )
    return build_from_leaves(
        leaves,
        f"{heading}, BY band-simplified tree",
        appearance,
        band_subtitle(panel, leaves),
    )


def build_from_leaves(
    leaves: pl.DataFrame,
    heading: str,
    appearance: dict[str, Any],
    subtitle: list[str],
) -> Any:
    """Draw the diagram of a leaf table that already has ``marker_area``.

    ``subtitle`` is the title's two lines, already carrying the registry's numbers.
    """
    size = int(appearance.get("panel_size", 420))
    domain, ticks = fitness_axis(
        min(leaves["death_scalar"].min(), leaves["birth_scalar"].min()),
        leaves["birth_scalar"].max(),
        appearance,
    )
    x_title = appearance.get("persistence_x_title", "Fitness at death")
    y_title = appearance.get("persistence_y_title", "Fitness at birth")
    grid = bool(appearance.get("grid", False))
    x = alt.X(
        "death_scalar:Q",
        title=x_title,
        scale=alt.Scale(domain=domain, nice=False, zero=False),
        axis=alt.Axis(values=ticks, format=".1f", grid=grid),
    )
    y = alt.Y(
        "birth_scalar:Q",
        title=y_title,
        scale=alt.Scale(domain=domain, nice=False, zero=False),
        axis=alt.Axis(values=ticks, format=".1f", grid=grid),
    )
    diagonal = (
        alt.Chart(
            pl.DataFrame(
                {
                    "death_scalar": [domain[0]],
                    "birth_scalar": [domain[0]],
                    "death_end": [domain[1]],
                    "birth_end": [domain[1]],
                }
            )
        )
        .mark_rule(
            color=appearance.get("diagonal_color", DIAGONAL_GREY),
            strokeWidth=float(appearance.get("diagonal_width", 1.0)),
        )
        .encode(x=x, y=y, x2="death_end:Q", y2="birth_end:Q")
    )
    layers = [diagonal, *marker_layers(leaves, appearance, x, y)]
    layers.extend(_global_pair_layers(leaves, appearance, x, y))
    # The leaf diagram draws peaks only, so its key has no valley entry.
    layers.extend(
        persistence_legend_layers(
            size,
            size,
            appearance,
            wild_type=bool(leaves["is_wild_type"].any()),
            stop_codon=bool(leaves["birth_has_stop_codon"].any()),
        )
    )
    chart = (
        alt.layer(*layers)
        .properties(
            width=size,
            height=size,
            title=alt.TitleParams(
                text=f"{heading}: persistence of every leaf",
                subtitle=subtitle,
                fontSize=int(appearance.get("heading_font_size", 18)),
                subtitleFontSize=int(appearance.get("subtitle_font_size", 14)),
                fontWeight="bold",
                anchor="start",
                offset=int(appearance.get("title_offset", 10)),
            ),
        )
    )
    return (
        chart
        .configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(appearance.get("font_size", 15)),
            titleFontSize=int(appearance.get("font_size", 15)),
        )
        .configure_view(strokeWidth=0)
    )


def marker_layers(
    leaves: pl.DataFrame, appearance: dict[str, Any], x: Any, y: Any
) -> list[Any]:
    """One filled triangle layer per annotation, plain peaks first.

    A leaf takes the first of wild type, stop codon (ADR 0012). The size channel is
    the precomputed ``marker_area`` read as pixels (``scale=None``), so the area is
    exactly the table's and Altair never rescales it. Every marker is filled.
    """
    kinds = peak_annotation_kinds(
        leaves,
        wild_type=frozenset(
            leaves.filter(pl.col("is_wild_type"))["peak_sequence"].to_list()
        ),
        label="peak_sequence",
    )
    layers = []
    domain = (1.0, float(max(1, int(leaves["member_count"].max() or 1))))
    for kind in (None, "stop_codon", "wild_type"):
        color, opacity = peak_annotation_style(kind, appearance, color=PEAK_GREY)
        in_kind = pl.Series([value == kind for value in kinds], dtype=pl.Boolean)
        rows = leaves.filter(in_kind)
        if rows.is_empty():
            continue
        rows = rows.with_columns(
            pl.col("member_count").map_elements(
                lambda count: branch_count_opacity(count, appearance, domain),
                return_dtype=pl.Float64,
            ).alias("count_opacity"),
        )
        layers.append(
            alt.Chart(
                rows.select(
                    "peak_sequence",
                    "peak_aa_sequence",
                    "birth_scalar",
                    "death_scalar",
                    "member_count",
                    "marker_area",
                    "count_opacity",
                )
            )
            .mark_point(
                shape=FIVE_POINT_STAR if kind == "wild_type" else "triangle-up",
                filled=True,
                color=color,
                strokeWidth=0,
            )
            .encode(
                x=x,
                y=y,
                size=alt.Size("marker_area:Q", scale=None, legend=None),
                opacity=alt.value(1.0) if kind == "wild_type" else alt.Opacity("count_opacity:Q", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("peak_sequence:N", title="Peak"),
                    alt.Tooltip("member_count:Q", title="Sequences in hill"),
                    alt.Tooltip("birth_scalar:Q", title="Birth", format=".4f"),
                    alt.Tooltip("death_scalar:Q", title="Death", format=".4f"),
                ],
            )
        )
    return layers


def _global_pair_layers(
    leaves: pl.DataFrame, appearance: dict[str, Any], x: Any, y: Any
) -> list[Any]:
    """Mark the global pair with a filled black triangle and label it.

    The triangle repeats the leaf's own size, so the area still reads as the hill's
    sequence count; the label sits to its right. Nothing is hollow.
    """
    rows = leaves.filter(pl.col("is_global_pair"))
    if rows.is_empty():
        return []
    ring = (
        alt.Chart(rows.select("birth_scalar", "death_scalar", "marker_area"))
        .mark_point(
            shape="triangle-up",
            filled=True,
            color=appearance.get("global_pair_color", EMPHASIS_BLACK),
            strokeWidth=0,
        )
        .encode(x=x, y=y, size=alt.Size("marker_area:Q", scale=None, legend=None))
    )
    label = (
        alt.Chart(rows.select("birth_scalar", "death_scalar"))
        .mark_text(
            text=appearance.get("global_pair_label", "Global maximum pair"),
            align="left",
            baseline="middle",
            dx=float(appearance.get("global_pair_label_dx", 12)),
            fontSize=int(appearance.get("legend_font_size", 15)),
            color=appearance.get("global_pair_color", EMPHASIS_BLACK),
        )
        .encode(x=x, y=y)
    )
    return [ring, label]


def legend_entries(leaves: pl.DataFrame, appearance: dict[str, Any]) -> list[dict]:
    """Return the legend rows: label, colour and whether the glyph is filled.

    Only the annotations present in ``leaves`` get a row.
    """
    entries = [{"label": "Hill", "color": PEAK_GREY, "filled": True}]
    if bool(leaves["birth_has_stop_codon"].any()):
        color, _ = peak_annotation_style("stop_codon", appearance, color=PEAK_GREY)
        entries.append({"label": _NOTES["stop_codon"], "color": color, "filled": True})
    if bool(leaves["is_wild_type"].any()):
        color, _ = peak_annotation_style("wild_type", appearance, color=PEAK_GREY)
        entries.append({"label": _NOTES["wild_type"], "color": color, "filled": True})
    return entries


def _legend_layers(
    leaves: pl.DataFrame, appearance: dict[str, Any], size: int
) -> list[Any]:
    """Draw the legend in the empty lower-right corner (below the diagonal).

    Births never fall below deaths, so no marker lies under the diagonal; the block
    stands ``legend_padding`` pixels from the panel's bottom and right edges, one
    entry per row with the last at the bottom, and ends with the area rule.
    """
    font_size = int(appearance.get("legend_font_size", 15))
    char_width = float(appearance.get("legend_char_width", 0.6))
    padding = float(appearance.get("legend_padding", 8))
    row_gap = float(appearance.get("legend_row_gap", 4))
    marker_gap = float(appearance.get("legend_marker_gap", 12))
    glyph_size = int(appearance.get("legend_marker_area", 80))
    entries = legend_entries(leaves, appearance)
    rule = appearance.get("area_legend_label", "Area proportional to hill sequences")
    glyph_x = (
        size
        - padding
        - max(len(e["label"]) for e in entries) * font_size * char_width
        - marker_gap
    )
    rows = len(entries) + 1
    layers: list[Any] = []
    for index in range(rows):
        # Row ``rows - 1`` is the bottom one; y counts pixels down from the top edge.
        y = size - padding - font_size / 2 - (rows - 1 - index) * (font_size + row_gap)
        if index < len(entries):
            entry = entries[index]
            data = alt.Data(values=[{"label": entry["label"]}])
            layers.append(
                alt.Chart(data)
                .mark_point(
                    shape="triangle-up",
                    filled=True,
                    color=entry["color"],
                    size=glyph_size,
                    strokeWidth=0,
                )
                .encode(x=alt.value(glyph_x), y=alt.value(y))
            )
            label = entry["label"]
        else:
            label = rule
        layers.append(
            alt.Chart(alt.Data(values=[{"label": label}]))
            .mark_text(align="left", baseline="middle", fontSize=font_size)
            .encode(
                x=alt.value(glyph_x + marker_gap if index < len(entries) else glyph_x),
                y=alt.value(y),
                text="label:N",
            )
        )
    return layers
