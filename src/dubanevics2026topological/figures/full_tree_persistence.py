"""Full-graph tree and persistence diagram figure.

Two panels share one fitness axis: the raw split tree coloured by branch
vertex count and the persistence diagram (peaks above the diagonal as
split-tree pairs, valleys below it as join-tree pairs, filled triangles up and
down) under its marginal death histogram with the median marked.

Both figures draw a grey noise square (data minimum to the median-reflected
threshold on both axes) at the very back of the diagram, and the diagram key's stop
entry reads "≥1 stop codon".

The ``full_tree_persistence_vertical_histogram`` variant turns the histogram
vertical and puts it leftmost (histogram | tree | diagram). The histogram owns the
one fitness axis (title "Fitness (WT = 0)", ticks at the data minimum, 0 and the data
maximum, two decimals); the tree and the diagram share its scale and draw none. A
median rule runs across the histogram and the tree, a separate one on the diagram,
and the diagram writes the median and the median-reflected threshold values at the
left of their lines. Ordinary H0 persistence,
extended persistence (not computed) and split-tree branch span stay distinct
quantities; the figure labels the diagram simply "Persistence".
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import altair as alt
import polars as pl

from ..analysis.leading_saddle_band import tree_saddles
from ..threshold.functional_threshold import (
    FDR_FIXED_CUTOFF,
    MEDIAN_REFLECTED,
    CandidateThreshold,
    configured_declaration,
)
from ._branch_count_style import branch_count_opacity
from ._charts import (
    FIVE_POINT_STAR,
    biology_axis_contract,
    count_domain,
    dashed_threshold_rule_style,
    fixed_count_axis,
    padded_scalar_domain,
    peak_annotation_kinds,
    persistence_legend_layers,
    plot_data_with_metrics,
    publication_scalar_scales,
    rows_with_annotation,
    stop_cross_layer,
    synonymous_leaf_edges,
    synonymous_leaf_segments,
    synonymous_teal_branch_layer,
    synonymous_valley_sequences,
    tree_figure,
    tree_marker_size,
    wild_type_identities,
)
from ._top_histogram import count_axis_limit, top_histogram_settings
from .component_persistence import descending_component_pairs
from .palette import (
    BAND_GREY,
    COUNT_BLACK,
    DIAGONAL_GREY,
    LIGHT_GREY,
    MEDIAN_GREY,
    PEAK_GREY,
    SYNONYMOUS_TEAL,
    THRESHOLD_GREY,
    WILD_TYPE_RED,
    tree_panel_size,
)

LOGGER = logging.getLogger(__name__)

FIGURE_ID = "full_tree_persistence"
VERTICAL_FIGURE_ID = "full_tree_persistence_vertical_histogram"
# The median rule is dotted, as (dash, gap) pixel lengths. The shipped value lives
# in config/figures.toml; this is the fallback.
MEDIAN_DASH = [1.0, 3.0]
# Marginal histogram bins: the tree-histogram figures' 200, so the bars match.
HISTOGRAM_BINS = 200
# The diagram key's stop entry covers peaks and valleys; the tree key keeps its own label.
DIAGRAM_STOP_LABEL = "≥1 stop codon"


def median_reflected_cutoff(rule: str, declaration: Mapping[str, Any]) -> bool:
    """Whether the Candidate cutoff is the median-reflected one.

    True for the ``median_reflected`` rule and for the ``fdr_fixed_cutoff`` rule
    declared with ``cutoff = "median_reflected"``; the rule's name alone does not
    say, since an FDR selection can sit on either fixed cutoff.
    """
    if rule == MEDIAN_REFLECTED:
        return True
    return rule == FDR_FIXED_CUTOFF and declaration.get("cutoff") == MEDIAN_REFLECTED


def build_full_tree_persistence_figure(
    payload: dict[str, Any],
    settings: dict[str, Any],
    variant: str,
    *,
    slide: dict[str, Any] | None = None,
) -> Any:
    """Compose the tree, persistence diagram and histogram on one fitness axis.

    ``slide`` (``{"coloring": "count"|"none"}``) is the frameless slide render: no title
    block and, with ``coloring="none"``, a plain-edge tree without the count colour bar.
    """
    return _build_figure(payload, settings, variant, slide=slide, vertical=False)


def build_full_tree_persistence_vertical_histogram_figure(
    payload: dict[str, Any], settings: dict[str, Any], variant: str
) -> Any:
    """The alternative layout: vertical histogram | tree | persistence diagram.

    The histogram (bars horizontal, fitness on the shared vertical axis) sits
    leftmost; a median rule runs from it across the tree and a separate one across
    the diagram, with the median's value written between the tree and the diagram.
    The shared fitness axis carries ticks at the median-reflected threshold and the
    minimum fitness, and the diagram has a grey square from the minimum to the
    threshold on both axes behind its points (the noise region).
    """
    return _build_figure(payload, settings, variant, slide=None, vertical=True)


def _figure_settings(settings: dict[str, Any], vertical: bool) -> dict[str, Any]:
    """The figure's table; the variant's keys override the main figure's table."""
    tables = settings["figures"]["figures"]
    if not vertical:
        return tables[FIGURE_ID]
    main, own = tables[FIGURE_ID], tables[VERTICAL_FIGURE_ID]
    merged = {**main, **own}
    for key in ("tree", "persistence"):
        merged[key] = {**main.get(key, {}), **own.get(key, {})}
    return merged


def _valley_pairs(pairs: pl.DataFrame) -> pl.DataFrame:
    """The join-tree (relative) pairs of the descending view: the valleys."""
    return descending_component_pairs(pairs).filter(pl.col("pair_kind") == "relative")


def _valley_tree_layer(
    tree_analysis: Any,
    pairs: pl.DataFrame,
    appearance: dict[str, Any],
    size: int,
    *,
    synonymous: frozenset[str] = frozenset(),
    teal: str = SYNONYMOUS_TEAL,
) -> Any:
    """Filled ▼ at the tree nodes that are join-tree valley saddles: grey, or teal
    where the valley is synonymous to its saddle (``synonymous`` holds those valleys'
    DNA identities).

    A join saddle is a tree node only where its sequence is also a node of the
    split tree; the others have no position on this tree and are not drawn.
    """
    valleys = _valley_pairs(pairs)
    nodes = tree_analysis.plot_data.node_table
    columns = set(nodes.columns)
    sequences = set(valleys["death_sequence"].to_list())
    grey = str(appearance.get("point_color", PEAK_GREY))
    if not {"dna_sequence", "x", "scalar"} <= columns or not sequences:
        points = pl.DataFrame(
            schema={
                "dna_sequence": pl.String,
                "x": pl.Float64,
                "y": pl.Float64,
                "marker_color": pl.String,
            }
        )
    else:
        # A saddle shared by a synonymous and a plain valley is drawn teal once.
        teal_saddles = set(
            valleys.filter(pl.col("birth_sequence").is_in(sorted(synonymous)))[
                "death_sequence"
            ].to_list()
        )
        points = nodes.filter(pl.col("dna_sequence").is_in(sorted(sequences))).select(
            "dna_sequence",
            pl.col("x").cast(pl.Float64),
            pl.col("scalar").cast(pl.Float64).alias("y"),
            pl.when(pl.col("dna_sequence").is_in(sorted(teal_saddles)))
            .then(pl.lit(teal))
            .otherwise(pl.lit(grey))
            .alias("marker_color"),
        )
    return (
        alt.Chart(points)
        .mark_point(shape="triangle-down", filled=True, size=size)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q"),
            color=alt.Color("marker_color:N", scale=None, legend=None),
            tooltip=["dna_sequence:N"],
        )
    )


def _build_figure(
    payload: dict[str, Any],
    settings: dict[str, Any],
    variant: str,
    *,
    slide: dict[str, Any] | None,
    vertical: bool,
) -> Any:
    if variant not in {"vanilla", "annotated"}:
        raise ValueError("variant must be 'vanilla' or 'annotated'.")
    analysis = payload["analysis"]
    quantities = payload["manuscript_quantities"]
    figure_settings = _figure_settings(settings, vertical)
    rendering = settings["figures"].get("rendering", {})
    annotated = variant == "annotated"
    tree_analysis = analysis.trees["split"]
    plot_data = plot_data_with_metrics(tree_analysis)
    branch_domain = count_domain(plot_data.edge_table)
    fitness_domain = tuple(float(x) for x in analysis.scalar_domain)

    appearance = figure_settings.get("persistence", {})
    teal = str(rendering.get("synonymous_teal", SYNONYMOUS_TEAL))
    # Synonymous peaks (annotated only): leaf branches teal by vertex count on the
    # tree, teal markers on the diagram. A leaf is synonymous to its adjacent saddle.
    synonymous_peaks: frozenset[str] = frozenset()
    synonymous_overlay = []
    nodes_have_genotype = "aa_sequence" in plot_data.node_table.columns and bool(
        {"display_event", "event"} & set(plot_data.node_table.columns)
    )
    if (
        annotated
        and nodes_have_genotype
        and {"source_label", "target_label"} <= set(plot_data.edge_table.columns)
    ):
        edges = synonymous_leaf_edges(plot_data)
        if not edges.is_empty():
            synonymous_peaks = frozenset(edges["target_label"].to_list())
            synonymous_overlay.append(
                synonymous_teal_branch_layer(
                    synonymous_leaf_segments(plot_data, edges),
                    {**rendering, **figure_settings},
                    branch_domain,
                    keep_fitness_axis=True,
                )
            )
    synonymous_valleys = (
        synonymous_valley_sequences(_valley_pairs(payload["pairs"]))
        if annotated
        else frozenset()
    )
    tree_overlays = [
        *synonymous_overlay,
        _valley_tree_layer(
            tree_analysis,
            payload["pairs"],
            appearance,
            tree_marker_size(rendering),
            synonymous=synonymous_valleys,
            teal=teal,
        ),
    ]
    median = float(payload["full_fitness"]["raw_fitness"].median())
    diagnostics = payload.get("threshold_diagnostics")
    if vertical:
        if diagnostics is None:
            from ..results import MissingProductError

            raise MissingProductError(
                "threshold_diagnostics product required; run `threshold-diagnostics-product`."
            )
        tree_overlays.append(_median_rule(median, None, appearance, on_tree=True))

    # Stop markers follow [rendering] stop_codon_tree_node_kinds (peaks and valleys
    # by default), applied by the shared stop-marker layer.
    tree = tree_figure(
        plot_data,
        tree_analysis,
        analysis,
        figure_settings,
        {**rendering, **figure_settings},
        branch_domain,
        annotated=annotated,
        coloring=(slide or {}).get("coloring", "count"),
        overlays=tree_overlays,
    )
    if vertical:
        # The histogram owns the one fitness axis; the tree draws none.
        tree = _without_y_axes(tree)
    persistence = _persistence_panel(
        payload["pairs"],
        payload["full_fitness"],
        fitness_domain,
        figure_settings,
        rendering,
        configured_declaration(settings),
        annotated=annotated,
        wild_type=wild_type_identities(tree_analysis),
        threshold_diagnostics=diagnostics,
        rule_style=settings["figures"]["figures"]["graph_tree_persistence"],
        branch_counts={
            str(row["target_label"]): int(row["vertex_count"])
            for row in plot_data.edge_table.select(
                "target_label", "vertex_count"
            ).to_dicts()
        },
        branch_domain=branch_domain,
        vertical=vertical,
        synonymous_peaks=synonymous_peaks,
        synonymous_valleys=synonymous_valleys,
    )
    panel_gap = int(rendering.get("panel_gap", 10))
    if vertical:
        # The panels sit flush: nothing bridges the seams.
        panel_gap = int(appearance.get("seam_gap", 0))
        histogram = _vertical_histogram(
            payload["full_fitness"],
            fitness_domain,
            figure_settings,
            rendering,
            median=median,
        )
        # Vega-Lite spaces panels by their full bounds, labels that overhang a plot area
        # included, so the "WT" label left of the tree's plot area would leave a blank
        # band between the histogram and the tree. A negative spacing on that seam alone
        # lets the label overlap the histogram's empty right edge instead; Vega stops at
        # the histogram's own bounds (its "0" tick label), a few px from the bars.
        overlap = int(appearance.get("tree_left_overlap_px", 0))
        left = alt.hconcat(histogram, tree, spacing=panel_gap - overlap).resolve_scale(
            x="independent", color="independent"
        )
        chart = alt.hconcat(left, persistence, spacing=panel_gap).resolve_scale(
            x="independent", color="independent"
        )
    else:
        # The persistence panel carries a marginal histogram above its diagram, so
        # the tree gets a blank band of the same height to stay aligned.
        marginal_height = int(appearance.get("histogram_height", 65))
        band_gap = int(appearance.get("histogram_gap", 8))
        chart = alt.hconcat(
            _with_top_band(tree, marginal_height, band_gap),
            persistence,
            spacing=panel_gap,
        ).resolve_scale(x="independent", color="independent")
    # Fitness alignment relies on the explicit shared domain that
    # publication_scalar_scales applies; each panel is a vconcat, so a
    # shared y resolve on this outer concat would have no effect.
    if slide is None:
        chart = chart.properties(
            title=alt.TitleParams(
                text="Full graph",
                subtitle=[
                    (
                        "Papkou et al. (2023) · Persistence: "
                        f"{quantities.display('full_graph.persistence.ordinary_h0_pair_count')}"
                        " ordinary H0 pairs · "
                        f"{quantities.display('full_graph.persistence.valley_count')} valleys"
                    ),
                    f"{quantities.display('full_graph.vertex_count')} total sequences",
                ],
            )
        )
    font_size = int(
        figure_settings.get("tree", {}).get("font_size", rendering.get("font_size", 15))
    )
    chart = (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(rendering.get("axis_label_font_size", font_size)),
            titleFontSize=font_size,
            grid=False,
        )
        .configure_legend(
            labelFontSize=int(rendering.get("legend_font_size", 15)),
            titleFontSize=int(rendering.get("legend_font_size", 15)),
        )
        .configure_title(
            fontSize=int(rendering.get("heading_font_size", 18)),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )
    # The median is a persistence-panel death-axis tick that the shared fitness
    # ticks would otherwise replace.
    extra_ticks = {} if vertical else {"death_scalar": [median]}
    exact_ticks = None
    if vertical:
        # The diagram's x axis: the data minimum, 0 and the data maximum, three decimals.
        low = float(payload["full_fitness"]["raw_fitness"].min())
        high = float(payload["full_fitness"]["raw_fitness"].max())
        exact_ticks = {"death_scalar": [low, 0.0, high]}
    return publication_scalar_scales(
        chart,
        fitness_domain,
        rendering,
        extra_ticks=extra_ticks,
        exact_ticks=exact_ticks,
    )


@dataclass(frozen=True)
class TopologicalNoiseBand:
    """Numbers behind the topological noise band of the full graph.

    ``global_minimum`` is ``g``, ``median`` is ``m``, ``median_gap`` is
    ``h = m - g``, ``noise_band_upper`` is ``m + h`` (the band's upper edge, a
    symmetry assumption that holds under any Selection rule) and
    ``functional_threshold`` is the rounded Candidate functional threshold of the
    active rule (``candidate_selection``). The two saddle deaths are ``None`` when
    the checkpoint has no pair of that kind.
    """

    global_minimum: float
    median: float
    median_gap: float
    noise_band_upper: float
    functional_threshold: float
    lowest_split_saddle_death: float | None
    highest_join_saddle_death: float | None


def topological_noise_band(
    pairs: pl.DataFrame,
    full_fitness: pl.DataFrame,
    declaration: Mapping[str, Any],
    candidate: CandidateThreshold,
    reflected: CandidateThreshold,
) -> TopologicalNoiseBand:
    """Return the topological noise band numbers for the full-graph data.

    The numbers come from the precomputed ``threshold_diagnostics`` product
    (real CandidateThreshold objects) so the figure performs no refit.
    """
    saddles = tree_saddles(pairs)
    return TopologicalNoiseBand(
        global_minimum=candidate.global_minimum,
        median=candidate.median,
        median_gap=candidate.median_gap,
        noise_band_upper=reflected.threshold,
        functional_threshold=candidate.threshold,
        lowest_split_saddle_death=(
            None if saddles.split_trailing is None else saddles.split_trailing.fitness
        ),
        highest_join_saddle_death=(
            None if saddles.join_trailing is None else saddles.join_trailing.fitness
        ),
    )


def _threshold_rule_layers(
    noise_band_upper: float,
    appearance: Mapping[str, Any],
    rule_style: dict[str, Any],
    x_scale: Any,
    width: int,
    *,
    y_axis: bool = True,
) -> list[Any]:
    """Dashed rule at birth = the threshold, its single y label and its caption.

    With ``y_axis=False`` the rule owns no axis (the vertical variant's histogram
    owns the fitness axis and the diagram writes the value beside the rule).

    The rule spans the persistence diagram only. It owns the diagram's left axis:
    one label at the threshold (three decimals through the axis ``format``), no
    tick marks, no domain line and no title. The caption sits just above the rule
    at the diagram's right edge.
    """
    upper = float(noise_band_upper)
    label = str(appearance.get("threshold_apex_label", "Median-reflected\nthreshold"))
    # Vega draws a list of strings as lines; a "\n" inside one string stays on one line.
    lines = label.split("\n")
    frame = pl.DataFrame({"functional_threshold": [upper]})
    y = alt.Y(
        "functional_threshold:Q",
        scale=x_scale,
        axis=alt.Axis(
            values=[upper], format=".3f", ticks=False, domain=False, title=None
        )
        if y_axis
        else None,
    )
    rule = alt.Chart(frame).mark_rule(**dashed_threshold_rule_style(rule_style)).encode(y=y)
    text = (
        alt.Chart(frame)
        .mark_text(
            align="right",
            baseline="bottom",
            # Vega's bottom baseline anchors a multi-line block by its first line, so
            # lift the block by the lines after the first.
            dy=int(appearance.get("threshold_apex_label_dy", -4))
            - (len(lines) - 1) * round(1.25 * int(appearance.get("median_label_font_size", 11))),
            fontSize=int(appearance.get("median_label_font_size", 11)),
            color=appearance.get("threshold_color", THRESHOLD_GREY),
            text=lines,
        )
        .encode(
            x=alt.value(width),
            y=alt.Y("functional_threshold:Q", scale=x_scale, axis=None),
        )
    )
    return [rule, text]


def _median_rule(
    median: float,
    scale: Any,
    appearance: Mapping[str, Any],
    *,
    on_tree: bool = False,
    label_values: list[float] | None = None,
) -> Any:
    """Horizontal dotted median rule on a fitness (vertical) axis; it owns no axis.

    On the tree (``on_tree``) the rule uses the tree's own ``y`` field and axis, like
    the wild-type and stop layers: a layer with ``axis=None`` cannot be merged with
    the tree line's visible axis.

    With ``label_values`` the rule owns the diagram's left axis, which carries only
    those values: black tick-label text just outside the plot area, no tick marks, no
    domain line, no title.
    """
    mark = alt.Chart(
        pl.DataFrame({"y" if on_tree else "median_fitness": [median]})
    ).mark_rule(
        color=appearance.get("median_color", MEDIAN_GREY),
        strokeDash=[float(v) for v in appearance.get("median_dash", MEDIAN_DASH)],
        strokeWidth=float(appearance.get("median_width", 1.2)),
    )
    if on_tree:
        return mark.encode(y=alt.Y("y:Q"))
    if label_values is None:
        return mark.encode(y=alt.Y("median_fitness:Q", scale=scale, axis=None))
    axis = alt.Axis(
        values=[float(v) for v in label_values],
        format=".3f",
        ticks=False,
        domain=False,
        title=None,
        labelColor=COUNT_BLACK,
    )
    return mark.encode(y=alt.Y("median_fitness:Q", scale=scale, axis=axis))


def _without_y_axes(chart: Any) -> Any:
    """The chart with every y axis removed; scales and domains are kept."""

    def visit(node: Any) -> Any:
        encoding = getattr(node, "encoding", alt.Undefined)
        if encoding is not alt.Undefined:
            channel = getattr(encoding, "y", alt.Undefined)
            if channel is not alt.Undefined:
                spec = dict(channel) if isinstance(channel, dict) else channel.to_dict()
                if "field" in spec:
                    spec["axis"] = None
                    node = node.encode(y=spec)
        for composition in ("layer", "hconcat", "vconcat", "concat"):
            children = getattr(node, composition, alt.Undefined)
            if children is not alt.Undefined:
                node[composition] = [visit(child) for child in children]
        return node

    return visit(chart)


def _left_value_layer(
    value: float, appearance: Mapping[str, Any], scale: Any, color: str
) -> Any:
    """A value written at the left edge of the diagram, just above its horizontal line.

    The number is computed, not typed: the chart formats it to three decimals.
    """
    return (
        alt.Chart(pl.DataFrame({"line_value": [float(value)]}))
        .mark_text(
            align="left",
            baseline="bottom",
            dx=int(appearance.get("line_value_dx", 4)),
            dy=int(appearance.get("line_value_dy", -3)),
            fontSize=int(appearance.get("median_label_font_size", 11)),
            color=color,
        )
        .encode(
            x=alt.value(0),
            y=alt.Y("line_value:Q", scale=scale, axis=None),
            text=alt.Text("line_value:Q", format=".3f"),
        )
    )


def _histogram_table(
    full_fitness: pl.DataFrame,
    bin_count: int,
    fallback_span: float,
) -> pl.DataFrame:
    """Equal-width bins over the fitness range, with each bin's sequence count."""
    data_low = float(full_fitness["raw_fitness"].min())
    data_high = float(full_fitness["raw_fitness"].max())
    bin_span = (
        (data_high - data_low) / bin_count if data_high > data_low else fallback_span
    )
    counts = [0] * bin_count
    for value in full_fitness["raw_fitness"]:
        index = min(bin_count - 1, int((value - data_low) / bin_span))
        counts[index] += 1
    return pl.DataFrame(
        {
            "bin_start": [data_low + i * bin_span for i in range(bin_count)],
            "bin_end": [data_low + (i + 1) * bin_span for i in range(bin_count)],
            "sequence_count": counts,
        }
    )


def _vertical_histogram(
    full_fitness: pl.DataFrame,
    scalar_domain: tuple[float, float],
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    *,
    median: float,
) -> Any:
    """The fitness histogram turned vertical: horizontal bars, fitness on y.

    Bars grow leftward from the tree side (count axis reversed, at the bottom), and
    the fitness axis is the figure's only one (the tree and the diagram share its
    scale). It draws no median rule; the median is a tick on its axis.
    """
    appearance = figure_settings.get("persistence", {})
    height = tree_panel_size(rendering, figure_settings)[1]
    bin_count = int(figure_settings.get("histogram_bins", HISTOGRAM_BINS))
    if height <= 0 or bin_count <= 0:
        raise ValueError("Figure dimensions and histogram bins must be positive.")
    step = float(rendering.get("fitness_tick_step", 0.1))
    _, ticks = top_histogram_settings(rendering)
    table = _histogram_table(full_fitness, bin_count, step).rename(
        {"bin_start": "bin_min", "bin_end": "bin_max"}
    )
    count_limit = count_axis_limit([int(table["sequence_count"].max())], rendering)
    histogram_width = int(appearance.get("histogram_width", 90))
    domain = padded_scalar_domain(scalar_domain, rendering)
    y_scale = alt.Scale(domain=domain, nice=False, zero=False, padding=0)
    # The figure's one fitness axis: ticks at the data maximum, 0, the median and the data
    # minimum, written with their actual values to three decimals. (The field names are not in
    # ``publication_scalar_scales``' scalar fields, so that pass leaves this axis alone.)
    data_low = float(full_fitness["raw_fitness"].min())
    data_high = float(full_fitness["raw_fitness"].max())
    y_axis = alt.Axis(
        values=[data_high, 0.0, median, data_low],
        format=".3f",
        title=rendering.get("tree_axis_title", "Fitness (WT = 0)"),
        ticks=True,
        domain=False,
        grid=False,
    )
    table = table.rename({"bin_min": "hist_bin_min", "bin_max": "hist_bin_max"})
    bars = (
        alt.Chart(table)
        .mark_bar(
            orient="horizontal",
            color=figure_settings.get("histogram_color", LIGHT_GREY),
            clip=True,
        )
        .encode(
            y=alt.Y("hist_bin_min:Q", scale=y_scale, axis=y_axis),
            y2="hist_bin_max:Q",
            x=alt.X(
                "sequence_count:Q",
                title=appearance.get("histogram_y_title", "Count"),
                scale=alt.Scale(domain=[0, count_limit], nice=False, reverse=True),
                axis=_count_axis(ticks, figure_settings, centred=True),
            ),
            x2=alt.datum(0),
            tooltip=["hist_bin_min:Q", "hist_bin_max:Q", "sequence_count:Q"],
        )
    )
    return bars.properties(width=histogram_width, height=height)


def _count_axis(
    ticks: Any, figure_settings: dict[str, Any], *, centred: bool = False
) -> Any:
    """Count-axis style of the marginal histogram: ticks only, no line or grid.

    ``centred`` writes each tick label centred under its tick (no flush end labels)."""
    title = figure_settings.get("persistence", {}).get("histogram_y_title", "Count")
    base = fixed_count_axis(ticks, title=title).to_dict()
    extra = {"labelAlign": "center", "labelFlush": False} if centred else {}
    return alt.Axis(**{**base, "domain": False, "ticks": True, "grid": False, **extra})


def _with_top_band(panel: Any, height: int, gap: int) -> Any:
    """Stack an invisible band above ``panel`` so its plot area lines up.

    The stack is measured ``flush``, like the persistence column's, so only the
    declared heights and ``gap`` place the plot area (see ``_persistence_panel``).
    """
    band = (
        alt.Chart(pl.DataFrame({"band": [0.0]}))
        .mark_point(opacity=0)
        .encode(x=alt.X("band:Q", axis=None))
        .properties(width=1, height=max(height, 1))
    )
    return alt.vconcat(band, panel, spacing=gap, bounds="flush")


def _persistence_panel(
    pairs: pl.DataFrame,
    full_fitness: pl.DataFrame,
    scalar_domain: tuple[float, float],
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    declaration: Mapping[str, Any],
    *,
    annotated: bool,
    wild_type: frozenset[str] = frozenset(),
    threshold_diagnostics: dict[str, Any] | None = None,
    branch_counts: Mapping[str, int] | None = None,
    branch_domain: tuple[float, float] = (1.0, 1.001),
    rule_style: dict[str, Any] | None = None,
    vertical: bool = False,
    synonymous_peaks: frozenset[str] = frozenset(),
    synonymous_valleys: frozenset[str] = frozenset(),
) -> Any:
    """Persistence diagram under its marginal death histogram.

    Peaks are filled up triangles (split-tree pairs, above the diagonal) and valleys
    filled down triangles (join-tree pairs, below it). An annotated
    export draws stop-codon and wild-type critical points in their annotation
    colour in place of the plain mid-grey (ADR 0012).

    ``vertical`` is the vertical-histogram variant: the diagram comes without the
    marginal histogram, with a horizontal median rule (the median's own, not the
    tree's) and the grey noise square from the minimum to the median-reflected
    threshold on both axes behind the points.
    """
    branch_counts = branch_counts or {}
    required = {"dna_sequence": pl.String, "raw_fitness": pl.Float64}
    if any(full_fitness.schema.get(name) != dtype for name, dtype in required.items()):
        raise ValueError("Full fitness requires string identities and Float64 fitness.")
    if (
        not full_fitness.height
        or full_fitness["dna_sequence"].null_count()
        or full_fitness["dna_sequence"].n_unique() != full_fitness.height
        or full_fitness["raw_fitness"].null_count()
        or not full_fitness["raw_fitness"].is_finite().all()
    ):
        raise ValueError(
            "Full fitness must contain unique identities and finite values."
        )
    appearance = figure_settings.get("persistence", {})
    # The panel height is the tree height so the shared fitness axis aligns.
    height = tree_panel_size(rendering, figure_settings)[1]
    width = height  # square: the diagram is as wide as the tree is tall
    bin_count = int(figure_settings.get("histogram_bins", HISTOGRAM_BINS))
    if width <= 0 or height <= 0 or bin_count <= 0:
        raise ValueError("Figure dimensions and histogram bins must be positive.")
    scientific = descending_component_pairs(pairs)
    global_pair = scientific.filter(pl.col("pair_kind") == "extended_positive")
    if global_pair.height != 1:
        raise ValueError("Component diagram requires exactly one global pair.")
    row = global_pair.row(0, named=True)
    if row["birth_scalar"] != float(full_fitness["raw_fitness"].max()) or row[
        "death_scalar"
    ] != float(full_fitness["raw_fitness"].min()):
        raise ValueError("Global pair must represent the full-population extrema.")
    endpoints = pl.concat(
        [
            scientific.select(
                pl.col("birth_sequence").alias("dna_sequence"),
                pl.col("birth_scalar").alias("raw_fitness"),
            ),
            scientific.select(
                pl.col("death_sequence").alias("dna_sequence"),
                pl.col("death_scalar").alias("raw_fitness"),
            ),
        ]
    ).unique()
    if endpoints.join(
        full_fitness, on=["dna_sequence", "raw_fitness"], how="anti"
    ).height:
        raise ValueError(
            "Persistence endpoints must match the full fitness population."
        )
    # Peaks are the split-tree pairs (and the global pair); valleys are the relative
    # (join-tree) pairs, below the diagonal.
    plot_table = scientific.filter(pl.col("pair_kind") != "relative").with_columns(
        pl.lit("Peaks").alias("critical_category")
    )
    # The global pair is also the global minimum's valley: its mirror (death and birth
    # swapped) sits in the bottom-right corner, matching the printed valley count.
    mirror = {
        **row,
        "birth_sequence": row["death_sequence"],
        "death_sequence": row["birth_sequence"],
        "birth_scalar": row["death_scalar"],
        "death_scalar": row["birth_scalar"],
        "birth_has_stop_codon": row["death_has_stop_codon"],
        "death_has_stop_codon": row["birth_has_stop_codon"],
    }
    valley_table = pl.concat(
        [
            scientific.filter(pl.col("pair_kind") == "relative"),
            pl.DataFrame([mirror], schema=scientific.schema),
        ]
    ).with_columns(pl.lit("Valleys").alias("critical_category"))
    low, high = (float(v) for v in scalar_domain)
    domain = padded_scalar_domain(scalar_domain, rendering)
    step = float(rendering.get("fitness_tick_step", 0.1))
    ticks = sorted(
        {
            round(math.floor(low / step) * step, 10),
            0.0,
            round(math.ceil(high / step) * step, 10),
        }
    )
    median = float(full_fitness["raw_fitness"].median())
    gray = appearance.get("point_color", PEAK_GREY)

    def with_opacity(table: pl.DataFrame) -> pl.DataFrame:
        return table.with_columns(
            pl.col("birth_sequence")
            .map_elements(
                lambda identity: branch_count_opacity(
                    branch_counts.get(identity, branch_domain[1]),
                    rendering,
                    branch_domain,
                ),
                return_dtype=pl.Float64,
            )
            .alias("count_opacity")
        )

    plot_table = with_opacity(plot_table)
    # Valleys fade with the count ramp, a little lighter than the peaks.
    valley_table = with_opacity(valley_table).with_columns(
        pl.col("count_opacity") * float(appearance.get("valley_opacity_factor", 0.55))
    )
    # padding=0 is explicit because the marginal histogram's bar marks give a
    # shared continuous x scale a default 5 px inset on each side, which would
    # leave the diagram 240 px wide against its 250 px height.
    x_scale = alt.Scale(domain=domain, nice=False, zero=False, padding=0)
    fitness_axis = alt.Axis(
        grid=False,
        domain=False,
        ticks=True,
        values=ticks,
        format=".1f",
    )
    # The vertical variant's diagram carries no median tick: the median is written
    # at the diagram's left edge instead.
    data_low = float(full_fitness["raw_fitness"].min())
    data_high = float(full_fitness["raw_fitness"].max())
    death_axis = (
        alt.Axis(
            grid=False,
            domain=False,
            ticks=True,
            values=[data_low, 0.0, data_high],
            format=".3f",
        )
        if vertical
        else alt.Axis(**{**fitness_axis.to_dict(), "values": sorted({*ticks, median})})
    )
    # One marker size for the tree's cross and the diagram's triangles.
    peak_size = tree_marker_size(rendering)

    def points_layer(
        data: pl.DataFrame,
        color: str,
        *,
        faded: bool,
        star_size: int | None = None,
        shape: str = "triangle-up",
    ) -> Any:
        # ``star_size`` draws the wild type as a fixed-size star instead of the
        # peak triangles; ``shape`` is ``triangle-down`` for valleys. Every marker is
        # filled.
        mark = alt.Chart(data).mark_point(
            filled=True,
            color=color,
            strokeWidth=1.0,
            **(
                {"shape": FIVE_POINT_STAR, "size": star_size}
                if star_size is not None
                else {"shape": shape, "size": peak_size}
            ),
        )
        return mark.encode(
            x=alt.X(
                "death_scalar:Q",
                title=appearance.get("x_title", "Fitness at death"),
                scale=x_scale,
                axis=death_axis,
            ),
            y=alt.Y(
                "birth_scalar:Q",
                title=None,
                scale=x_scale,
                axis=fitness_axis,
            ),
            opacity=alt.Opacity("count_opacity:Q", scale=None, legend=None)
            if faded
            else alt.value(1.0),
            tooltip=[
                "pair_kind:N",
                "birth_sequence:N",
                "death_sequence:N",
                "birth_scalar:Q",
                "death_scalar:Q",
                "persistence:Q",
            ],
        )

    kinds = (
        peak_annotation_kinds(plot_table, wild_type=wild_type, label="birth_sequence")
        if annotated
        else [None] * plot_table.height
    )
    teal = str(rendering.get("synonymous_teal", SYNONYMOUS_TEAL))
    # The combined stop-and-synonymous encoding (annotated export): peaks ▲ and valleys
    # ▼ are teal where synonymous, mid-grey otherwise, each with the branch-count fade;
    # a stop codon is the red cross drawn on top of the marker; the wild type is the star.
    wild_rows = rows_with_annotation(plot_table, kinds, "wild_type")
    not_wild = plot_table.filter(
        pl.Series([kind != "wild_type" for kind in kinds], dtype=pl.Boolean)
    )
    peak_teal = not_wild.filter(pl.col("birth_sequence").is_in(sorted(synonymous_peaks)))
    peak_grey = not_wild.filter(~pl.col("birth_sequence").is_in(sorted(synonymous_peaks)))
    stop_rows = (
        not_wild.filter(pl.col("birth_has_stop_codon"))
        if annotated
        else not_wild.clear()
    )
    relative = scientific.filter(pl.col("pair_kind") == "relative")
    valley_syn = valley_table.filter(
        pl.col("birth_sequence").is_in(sorted(synonymous_valleys))
        & (pl.col("pair_kind") == "relative")
    )
    valley_grey = valley_table.filter(
        ~(
            pl.col("birth_sequence").is_in(sorted(synonymous_valleys))
            & (pl.col("pair_kind") == "relative")
        )
    )
    valley_stop_rows = (
        valley_table.filter(pl.col("birth_has_stop_codon"))
        if annotated
        else valley_table.clear()
    )
    both_peaks = stop_rows.filter(
        pl.col("birth_sequence").is_in(sorted(synonymous_peaks))
    ).height
    both_valleys = valley_stop_rows.filter(
        pl.col("birth_sequence").is_in(sorted(synonymous_valleys))
        & (pl.col("pair_kind") == "relative")
    ).height
    if annotated:
        LOGGER.info(
            "Combined encoding: %d synonymous peaks (%d with a stop codon), "
            "%d synonymous valleys of %d (%d with a stop codon); %d stop peaks, "
            "%d stop valleys.",
            peak_teal.height,
            both_peaks,
            valley_syn.height,
            relative.height,
            both_valleys,
            stop_rows.height,
            valley_stop_rows.height,
        )
    points = [points_layer(peak_grey, gray, faded=True)]
    if not peak_teal.is_empty():
        points.append(points_layer(peak_teal, teal, faded=True))
    if not valley_grey.is_empty():
        points.append(points_layer(valley_grey, gray, faded=True, shape="triangle-down"))
    if not valley_syn.is_empty():
        points.append(points_layer(valley_syn, teal, faded=True, shape="triangle-down"))
    annotation_points = []
    cross_x = alt.X("death_scalar:Q", scale=x_scale, axis=death_axis)
    cross_y = alt.Y("birth_scalar:Q", scale=x_scale, axis=fitness_axis)
    for rows in (stop_rows, valley_stop_rows):
        if not rows.is_empty():
            annotation_points.append(
                stop_cross_layer(rows, rendering, x=cross_x, y=cross_y)
            )
    if not wild_rows.is_empty():
        annotation_points.append(
            points_layer(
                wild_rows,
                str(rendering.get("wild_type_color", WILD_TYPE_RED)),
                faded=False,
                star_size=int(appearance.get("wild_type_size", 130)),
            )
        )
    median_dash = [float(value) for value in appearance.get("median_dash", MEDIAN_DASH)]
    median_width = float(appearance.get("median_width", 1.2))
    guide = (
        alt.Chart(pl.DataFrame({"median_fitness": [median]}))
        .mark_rule(
            color=appearance.get("median_color", MEDIAN_GREY),
            strokeDash=median_dash,
            strokeWidth=median_width,
        )
        .encode(x=alt.X("median_fitness:Q", scale=x_scale))
    )
    diagonal = (
        alt.Chart(pl.DataFrame({"death_scalar": domain, "birth_scalar": domain}))
        .mark_line(
            color=appearance.get("diagonal_color", DIAGONAL_GREY),
            strokeWidth=float(appearance.get("diagonal_width", 1)),
        )
        .encode(
            x=alt.X("death_scalar:Q", scale=x_scale),
            y=alt.Y("birth_scalar:Q", scale=x_scale),
        )
    )
    if threshold_diagnostics is None:
        from ..results import MissingProductError

        raise MissingProductError(
            "threshold_diagnostics product required; run `threshold-diagnostics-product`."
        )
    candidate = threshold_diagnostics["candidate"]
    # The topological noise band is drawn whenever the cutoff is the median-reflected
    # one: a square from (g, g) to (m + h, m + h) that straddles the diagonal (a
    # symmetry assumption). The cutoff kind comes from the rule and, for the FDR
    # rule, from its declared ``cutoff``, never from the rule's name alone: an FDR
    # selection under the median-reflected cutoff draws the square too. Under
    # ``mixture_quantile`` and the other cutoffs the band is not drawn (GLOSSARY:
    # Topological noise band is retained for the comparison rule only).
    draw_noise_band = median_reflected_cutoff(candidate.rule, declaration)
    if draw_noise_band:
        # Only the ``median_reflected`` rule stores a separate reflected record; for
        # an FDR selection under that cutoff the applied threshold is the reflected one.
        reflected = threshold_diagnostics.get("reflected", candidate)
        noise_band = topological_noise_band(
            pairs, full_fitness, declaration, candidate=candidate, reflected=reflected
        )
    else:
        noise_band = None

    threshold_marked = annotated and draw_noise_band and noise_band is not None
    # The grey noise square at the very back: from the data minimum to the
    # median-reflected threshold on both axes, behind the rules, the diagonal and the
    # points. It reuses the diagram's own field names so the axis contract below still
    # gives the diagram its x axis.
    reflected_cutoff = float(
        threshold_diagnostics.get("reflected", candidate).threshold
    )
    square = (
        alt.Chart(
            pl.DataFrame(
                {
                    "death_scalar": [float(full_fitness["raw_fitness"].min())],
                    "birth_scalar": [float(full_fitness["raw_fitness"].min())],
                    "square_high": [reflected_cutoff],
                }
            )
        )
        .mark_rect(
            clip=True,
            # The bootstrap band's own fill (D10 band and tree figures).
            color=appearance.get("band_color", BAND_GREY),
            opacity=float(appearance.get("band_opacity", 0.3)),
        )
        .encode(
            x=alt.X("death_scalar:Q", scale=x_scale, axis=death_axis),
            x2="square_high:Q",
            y=alt.Y("birth_scalar:Q", scale=x_scale, axis=fitness_axis),
            y2="square_high:Q",
        )
    )
    # The vertical variant always marks the median-reflected threshold; the top-histogram
    # figure draws the square whenever the cutoff is the median-reflected one (the
    # noise band is not drawn under the mixture and other cutoffs, see GLOSSARY).
    back = [square] if vertical or draw_noise_band else []
    layers = [*back, guide, diagonal]
    if vertical:
        # The diagram's own horizontal median rule in place of the vertical one, the
        # dashed median-reflected threshold rule, and both values written at the left
        # of their lines. Neither rule owns an axis: the histogram does.
        # The dotted rule starts left of the plot area, across the label room, so it runs
        # unbroken from the tree; the vertical median rule is the diagram's own.
        layers = [
            square,
            guide,
            _median_rule(
                median,
                x_scale,
                appearance,
                label_values=[reflected_cutoff],
            ),
            diagonal,
            *_threshold_rule_layers(
                reflected_cutoff,
                appearance,
                rule_style or {},
                x_scale,
                width,
                y_axis=False,
            )[: 2 if annotated else 1],
        ]
    # The plain points, then the wild type and the stop points on top.
    layers.extend([*points, *annotation_points])
    if threshold_marked and not vertical:
        # The annotated export marks the median-reflected threshold only: a dashed
        # rule with its one axis label and caption; the vanilla export has none.
        layers.extend(
            _threshold_rule_layers(
                noise_band.noise_band_upper,
                appearance,
                rule_style or {},
                x_scale,
                width,
            )
        )
    # The key sits inside the diagram, bottom right, one entry per line. In the vertical
    # variant it straddles the dotted median rule on the right of the point cloud: peak
    # and valley just above the rule, the other entries just below it, in the font size
    # of the threshold caption and with no backing box.
    legend_rendering = {**rendering, **appearance}
    around = None
    has_valleys = not valley_table.is_empty()
    if vertical:
        domain_low, domain_high = (float(v) for v in domain)
        median_y = height * (domain_high - median) / (domain_high - domain_low)
        around = (median_y, 1 + int(has_valleys))
        legend_rendering["persistence_legend_font_size"] = int(
            appearance.get("median_label_font_size", 11)
        )
    layers.extend(
        persistence_legend_layers(
            width,
            height,
            legend_rendering,
            valley=has_valleys,
            wild_type=not wild_rows.is_empty(),
            stop_codon=annotated,
            synonymous=not (peak_teal.is_empty() and valley_syn.is_empty()),
            peak_color=gray,
            stop_label=DIAGRAM_STOP_LABEL,
            encoded=annotated,
            around=around,
        )
    )
    scatter = alt.layer(*layers).properties(width=width, height=height)
    if (threshold_marked and not vertical) or vertical:
        # The other layers hide their y axis; with one merged axis Vega-Lite would
        # drop the rule's label too, so the rule keeps an axis of its own.
        scatter = scatter.resolve_axis(y="independent")
    # Duplicate fitness labels are dropped: the tree owns the shared y axis.
    scatter = biology_axis_contract(
        scatter,
        y_title=None,
        x_title=appearance.get("x_title", "Fitness at death"),
    )
    if vertical:
        # The histogram is a panel of its own on the left; this column is the diagram.
        return scatter
    _, ticks = top_histogram_settings(rendering)
    histogram_table = _histogram_table(full_fitness, bin_count, step)
    count_limit = count_axis_limit(
        [int(histogram_table["sequence_count"].max())], rendering
    )
    bars = (
        alt.Chart(histogram_table)
        .mark_bar(
            orient="vertical",
            color=figure_settings.get("histogram_color", LIGHT_GREY),
            clip=True,
        )
        .encode(
            x=alt.X(
                "bin_start:Q",
                scale=x_scale,
                axis=alt.Axis(labels=False, ticks=False, domain=False, title=None),
            ),
            x2="bin_end:Q",
            y=alt.Y(
                "sequence_count:Q",
                title=appearance.get("histogram_y_title", "Count"),
                scale=alt.Scale(domain=[0, count_limit], nice=False),
                axis=_count_axis(ticks, figure_settings),
            ),
            y2=alt.datum(0),
            tooltip=["bin_start:Q", "bin_end:Q", "sequence_count:Q"],
        )
    )
    median_label = (
        alt.Chart(pl.DataFrame({"median_fitness": [median]}))
        .mark_text(
            text="Median fitness",
            align="center",
            baseline="bottom",
            dy=-4,
            fontSize=int(appearance.get("median_label_font_size", 11)),
            color=appearance.get("median_color", MEDIAN_GREY),
        )
        .encode(x=alt.X("median_fitness:Q", scale=x_scale), y=alt.value(0))
    )
    marginal = alt.layer(guide, bars, median_label).properties(
        width=width, height=int(appearance.get("histogram_height", 65))
    )
    # bounds="flush" places the diagram by the declared marginal height and gap
    # alone. With the default "full" bounds Vega also reserves the Count axis's
    # overflow below the histogram (the half-height of its "0" tick label, about
    # 10 px), which pushed the diagram down and broke its alignment with the
    # tree's fitness axis. The Count axis space on the left is still kept: the
    # outer concat measures this column's full extent.
    panel = alt.vconcat(
        marginal,
        scatter,
        spacing=int(appearance.get("histogram_gap", 8)),
        bounds="flush",
    ).resolve_scale(x="shared", y="independent", color="independent")
    return panel
