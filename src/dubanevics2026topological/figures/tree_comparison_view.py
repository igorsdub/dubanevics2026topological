"""D10 comparison views: split trees of several saved states on one fitness axis.

Spec 2, Section 6, views 3, 5 and 7 (``tree_comparison_thresholds``,
``tree_comparison_fdr``, ``tree_comparison_band``). Every panel is one saved tree
(``SavedTreeInputs``, ``build/d10_tree_inputs.py``); nothing is rebuilt. All panels
share one fitness domain (the union of their landscapes' extents) and one branch
count colour scale. Every panel draws only the dashed rule of the functional
threshold it uses (Published, Candidate threshold), with no text; the value is an
extra three-decimal tick on its fitness axis. A panel with no threshold of its own
(Full) draws none. The band view subtitle
counts the surviving peaks including the global maximum's pair, closed at the
global minimum by convention; a panel with ``peak_labels`` writes each surviving
peak's top-leaf amino-acid and DNA sequence above it.

The arrangement follows the view: ``thresholds`` stacks its panels in one column,
``fdr`` is a two-column grid filled row by row (point estimate | BY, one row per
threshold) and ``band`` places its panels side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import altair as alt
import polars as pl

from ._charts import (
    _scalar_bounds,
    count_domain,
    leaf_row_title,
    overlay_count_colorbar,
    plot_data_with_metrics,
    publication_scalar_scales,
    stack_leaf_rows,
    synonymous_branch_layer,
    synonymous_leaf_edges,
    synonymous_leaf_segments,
    tree_chart,
)
from .candidate_tree_biology import tree_x_domain
from .palette import INK_GREY, tree_panel_size
from .tree_comparison import _threshold_layers

COLUMNS = {
    "tree_comparison_thresholds": 1,
    "tree_comparison_fdr": 2,
    "tree_comparison_band": None,  # every panel in one row
}

# ``plain``: the panels draw neither leaf kind. ``rows``: the Published biology figure's
# layout, a block of all panels marking stop-codon peaks and, below it, the same panels
# marking synonymous leaves, never both in one chart (the published default).
# ``combined``: one block, both leaf kinds on the same trees (the extra output).
LAYOUTS = ("plain", "rows", "combined")
# ``(heading kind, stop crosses drawn, synonymous branches drawn)`` per block.
_BLOCKS = {
    "plain": ((None, False, False),),
    "rows": (("stops", True, False), ("synonymous", False, True)),
    "combined": (("combined", True, True),),
}


@dataclass(frozen=True)
class ComparisonPanel:
    """One panel: the saved tree and the display text of its heading."""

    title: str
    subtitle: str
    inputs: Any
    peak_labels: Any = None  # x, y, aa_sequence, dna_sequence of the labelled peaks


def landmark_thresholds(panels: list[ComparisonPanel]) -> list[tuple[str, float]]:
    """``(name, T)`` of each distinct functional threshold among the panels, in order.

    The name is the panel's ``thr=`` parameter in its tree label (``published``,
    ``reflected``); a threshold shared by several panels is listed once.
    """
    landmarks: list[tuple[str, float]] = []
    for panel in panels:
        threshold = panel.inputs.threshold
        if threshold is None:
            continue
        label = getattr(panel.inputs, "tree_label", "")
        name = next(
            (p.split("=", 1)[1] for p in label.split("_") if p.startswith("thr=")),
            "functional",
        )
        if all(abs(threshold - other) > 1e-12 for _, other in landmarks):
            landmarks.append((name, float(threshold)))
    return landmarks


def union_scalar_domain(panels: list[ComparisonPanel]) -> tuple[float, float]:
    """The fitness domain every panel is drawn on."""
    domains = [panel.inputs.scalar_domain for panel in panels]
    return min(d[0] for d in domains), max(d[1] for d in domains)


def place_identity_labels(
    peaks: pl.DataFrame,
    rendering: dict[str, Any],
    *,
    x_domain: tuple[float, float] | None = None,
    y_domain: tuple[float, float] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> pl.DataFrame:
    """Return one row per peak: identity text and the ``y`` its label is drawn at.

    Labels are rotated, so each occupies a vertical strip about one font high.
    Strips that would meet are raised, in steps, until they clear. Every peak
    keeps its label; ``raised`` is true when the label moved off its tip (the
    caller draws a leader back to the peak). With no axis geometry the labels
    stay on their tips.
    """
    frame = peaks.with_columns(
        pl.format("{} {}", pl.col("aa_sequence"), pl.col("dna_sequence")).alias(
            "identity"
        )
    )
    font = float(rendering.get("peak_label_font_size", 8))
    offset = float(rendering.get("peak_label_offset", 6))
    if (
        x_domain is None
        or y_domain is None
        or not width
        or not height
        or x_domain[1] == x_domain[0]
        or y_domain[1] == y_domain[0]
    ):
        return frame.with_columns(
            pl.col("y").alias("y_text"), pl.lit(False).alias("raised")
        )
    char = float(rendering.get("peak_label_char_width_em", 1.0)) * font
    thick = float(rendering.get("peak_label_line_height_em", 1.35)) * font
    gap = float(rendering.get("peak_label_gap_px", 2))
    x0, x1 = (float(v) for v in x_domain)
    y0, y1 = (float(v) for v in y_domain)
    y_per_px = (y1 - y0) / float(height)
    records = frame.to_dicts()
    # High tips first, so a lower neighbour is the one that moves.
    order = sorted(range(len(records)), key=lambda i: -float(records[i]["y"]))
    boxes: list[tuple[float, float, float, float]] = []
    for index in order:
        row = records[index]
        x_px = (float(row["x"]) - x0) / (x1 - x0) * float(width)
        length = max(len(str(row["identity"])), 1) * char
        tip = (y1 - float(row["y"])) / (y1 - y0) * float(height) - offset
        raise_px = 0.0
        box = (0.0, 0.0, 0.0, 0.0)
        for _ in range(48):
            bottom = tip - raise_px
            top = bottom - length
            box = (x_px - thick / 2, x_px + thick / 2, top, bottom)
            hit = next(
                (
                    other
                    for other in boxes
                    if box[0] < other[1] + gap
                    and other[0] < box[1] + gap
                    and box[2] < other[3] + gap
                    and other[2] < box[3] + gap
                ),
                None,
            )
            if hit is None:
                break
            # Sit this label just above the one it met.
            raise_px += max(bottom - (hit[2] - gap), 1.0)
        boxes.append(box)
        row["y_text"] = float(row["y"]) + raise_px * y_per_px
        row["raised"] = raise_px > 0.5
    return pl.DataFrame(records).select(
        "x", "y", "y_text", "aa_sequence", "dna_sequence", "identity", "raised"
    )


def identity_label_layers(
    peaks: pl.DataFrame,
    rendering: dict[str, Any],
    *,
    x_domain: tuple[float, float] | None = None,
    y_domain: tuple[float, float] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[Any]:
    """Rotated text above each peak: its top leaf's amino-acid and DNA sequence.

    A label that would cross a neighbour is drawn higher, with a thin leader
    down to its peak, so adjacent names stay readable and none is omitted.
    """
    rows = place_identity_labels(
        peaks,
        rendering,
        x_domain=x_domain,
        y_domain=y_domain,
        width=width,
        height=height,
    )
    color = str(rendering.get("peak_label_color", INK_GREY))
    offset = int(rendering.get("peak_label_offset", 6))
    layers: list[Any] = []
    leaders = rows.filter(pl.col("raised"))
    if leaders.height:
        layers.append(
            alt.Chart(leaders)
            .mark_rule(color=color, strokeWidth=0.6)
            .encode(x=alt.X("x:Q", axis=None), y="y:Q", y2="y_text:Q")
        )
    layers.append(
        alt.Chart(rows)
        .mark_text(
            align="left",
            angle=270,
            baseline="middle",
            dy=-offset,
            fontSize=int(rendering.get("peak_label_font_size", 8)),
            color=color,
        )
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y_text:Q"), text="identity:N")
    )
    return layers


def grid_rows(view: str, charts: list[Any]) -> list[list[Any]]:
    """Charts in rows: the view's column count, or one row for ``band``."""
    if view not in COLUMNS:
        raise ValueError(f"{view!r} is not a tree comparison view.")
    columns = COLUMNS[view] or len(charts)
    if len(charts) % columns:
        raise ValueError(f"{view} needs a multiple of {columns} panels.")
    return [charts[i : i + columns] for i in range(0, len(charts), columns)]


def panel_trees(
    plots: list[Any],
    domain: tuple[float, float],
    branch_domain: tuple[float, float],
    rendering: dict[str, Any],
    figure_settings: dict[str, Any],
    *,
    panel_size: tuple[int, int],
    y_axis_extent: int,
    stops: bool,
    synonymous: bool,
    labelled: bool,
    colorbar_index: int | None = None,
) -> list[Any]:
    """One tree chart per panel: stop crosses, green synonymous leaves, neither or both.

    ``labelled`` draws a band panel's peak identity labels (unless the figure's
    ``peak_identity_labels`` is false, as for the manuscript comparisons) and, with ``stops``, the
    one-entry key of the cross on the figure's first panel (a second block repeats
    neither).

    ``colorbar_index`` puts the figure's one branch-count colour bar inside that
    panel, over the free space above its tree.
    """
    panel_width, panel_height = panel_size
    charts = []
    for index, (panel, analysis, data) in enumerate(plots):
        underlays, labels, overlays = [], [], []
        own = panel.inputs.threshold
        if own is not None:
            # Only the threshold this panel uses: a dashed rule with no text. Its
            # value is an extra three-decimal tick on the fitness axis (below).
            underlays = _threshold_layers(own, domain, rendering, labelled=False)
        if (
            labelled
            and bool(rendering.get("peak_identity_labels", True))
            and panel.peak_labels is not None
            and panel.peak_labels.height
        ):
            y_low, y_high, _ = _scalar_bounds(domain, rendering)
            labels.extend(
                identity_label_layers(
                    panel.peak_labels,
                    rendering,
                    x_domain=tree_x_domain(
                        data.edge_segment_table,
                        float(rendering.get("tree_horizontal_padding", 18)),
                        float(panel_width),
                    ),
                    y_domain=(y_low, y_high),
                    width=panel_width,
                    height=panel_height,
                )
            )
        if synonymous:
            edges = synonymous_leaf_edges(data)
            if not edges.is_empty():
                overlays.append(
                    synonymous_branch_layer(
                        synonymous_leaf_segments(data, edges),
                        rendering,
                        keep_fitness_axis=True,
                    )
                )
        extra = {"annotation_key": labelled and index == 0} if stops else {}
        tree = tree_chart(
            data,
            analysis,
            SimpleNamespace(scalar_domain=domain),
            figure_settings,
            rendering,
            branch_domain,
            linked=False,
            annotated=stops,
            underlays=underlays,
            overlays=overlays,
            count_labels=labels,
            **extra,
        )
        if index == colorbar_index:
            tree = overlay_count_colorbar(
                tree, figure_settings, rendering, branch_domain, width=panel_width
            )
        # Three ticks (domain ends and 0) plus the panel's own threshold, three decimals.
        tree = publication_scalar_scales(
            tree,
            domain,
            rendering,
            extra_ticks={} if own is None else {"y": [float(own)]},
            exact_domain=True,
            y_axis_extent=y_axis_extent,
        )
        charts.append(
            tree.properties(
                title=alt.TitleParams(
                    text=panel.title, subtitle=[panel.subtitle], anchor="start"
                )
            )
        )
    return charts


def build_tree_comparison_figure(
    view: str,
    panels: list[ComparisonPanel],
    settings: dict[str, Any],
    *,
    title: str,
    subtitle: str,
    layout: str = "plain",
) -> Any:
    """The comparison figure of ``view`` over ordered ``panels``.

    ``settings`` is the figures configuration (``settings["figures"]``); the
    appearance table is ``d10.tree_comparison``. ``layout`` is one of ``LAYOUTS``:
    ``"rows"`` stacks a block of every panel with stop-codon crosses over a block
    with synonymous leaves green (the Published biology figure's two rows),
    ``"combined"`` draws both on one set of trees, ``"plain"`` neither. Wild-type
    stars stay out of every comparison.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}.")
    figures = settings["figures"]
    figure_settings = figures["d10"]["tree_comparison"]
    rendering = {**figures.get("rendering", {}), **figure_settings}
    domain = union_scalar_domain(panels)
    plots = []
    maximum = 1
    for panel in panels:
        # Wild-type star and stop crosses belong to the annotated single views.
        analysis = SimpleNamespace(
            **{
                k: v
                for k, v in vars(panel.inputs.tree_analysis).items()
                if k != "wild_type_marker_table"
            }
        )
        data = plot_data_with_metrics(analysis)
        maximum = max(maximum, int(count_domain(data.edge_table)[1]))
        plots.append((panel, analysis, data))
    branch_domain = (1.0, float(maximum))
    y_axis_extent = int(rendering.get("tree_y_axis_extent", 60))
    panel_width, panel_height = tree_panel_size(rendering, figure_settings)
    gap = int(rendering.get("panel_gap", 10))
    blocks = []
    for position, (kind, stops, synonymous) in enumerate(_BLOCKS[layout]):
        charts = panel_trees(
            plots,
            domain,
            branch_domain,
            rendering,
            figure_settings,
            panel_size=(panel_width, panel_height),
            y_axis_extent=y_axis_extent,
            stops=stops,
            synonymous=synonymous,
            labelled=position == 0,
            # One colour bar per figure, inside the top-right panel (the last of the
            # first row), in the free space above its tree.
            colorbar_index=(COLUMNS[view] or len(plots)) - 1 if position == 0 else None,
        )
        block = [
            alt.hconcat(*row, spacing=gap).resolve_scale(color="independent")
            for row in grid_rows(view, charts)
        ]
        blocks.append((kind, block))
    spacing = int(rendering.get("comparison_panel_gap", 16))
    if len(blocks) == 1:
        chart = alt.vconcat(*blocks[0][1], spacing=spacing).resolve_scale(
            color="independent"
        )
    else:
        chart = stack_leaf_rows(
            [
                alt.vconcat(*block, spacing=spacing)
                .resolve_scale(color="independent")
                .properties(title=leaf_row_title(kind))
                for kind, block in blocks
            ],
            rendering,
        )
    chart = chart.properties(
        title=alt.TitleParams(text=title, subtitle=[subtitle], anchor="start")
    )
    font_size = int(
        figure_settings.get("tree", {}).get("font_size", rendering.get("font_size", 15))
    )
    chart = (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(rendering.get("axis_label_font_size", font_size)),
            titleFontSize=font_size,
        )
        .configure_legend(
            labelFontSize=int(rendering.get("legend_font_size", 15)),
            titleFontSize=int(rendering.get("legend_font_size", 15)),
        )
        .configure_title(
            fontSize=int(rendering.get("heading_font_size", font_size)),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )
    return chart
