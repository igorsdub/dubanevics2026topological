"""Internal shared chart vocabulary for the figure renderer modules."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import altair as alt
import polars as pl
from topographer.plotting import (
    PersistencePlotConfig,
    TreeMarkerConfig,
    TreePlotConfig,
    TreePresentationConfig,
    plot_persistence,
    plot_tree,
)

from ._branch_count_style import (
    branch_count_color_scale,
    branch_count_colors,
    branch_count_opacity,
    branch_count_teal,
)
from .labels import graph_label
from .palette import (
    BAND_GREY,
    COUNT_BLACK,
    LIGHT_GREY,
    PEAK_GREY,
    PERSISTENCE_HEIGHT,
    PERSISTENCE_WIDTH,
    STOP_CODON_CRIMSON,
    SYNONYMOUS_GREEN,
    SYNONYMOUS_TEAL,
    TREE_GREY,
    TREE_HEIGHT,
    TREE_WIDTH,
    WILD_TYPE_RED,
    tree_panel_size,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._quantities import ManuscriptQuantities

OPEN_X = "M-1,-1L1,1M-1,1L1,-1"
FIVE_POINT_STAR = (
    "M0,-1L0.2245,-0.309L0.9511,-0.309L0.3633,0.118L0.5878,0.809L0,0.382"
    "L-0.5878,0.809L-0.3633,0.118L-0.9511,-0.309L-0.2245,-0.309Z"
)


FUNCTIONAL_THRESHOLD_LABEL = "Functional threshold"
# Fallbacks for the figure-table keys ``threshold_label_padding`` (pixels from the
# label to the panel edge it is aligned to) and ``threshold_label_offset`` (pixels
# between the rule and its label); ``config/figures.toml`` sets both.
THRESHOLD_LABEL_PADDING = 8
THRESHOLD_LABEL_OFFSET = 3


def threshold_label_padding(appearance: dict[str, Any]) -> int:
    """Return the pixels between a threshold label and the panel edge it hugs."""
    return int(appearance.get("threshold_label_padding", THRESHOLD_LABEL_PADDING))


def functional_threshold_text(threshold: str) -> str:
    """Return ``Functional threshold = <threshold>`` for a subtitle.

    ``threshold`` is the registry's display text for the threshold
    (``thresholds.<graph>_functional.value``), printed as given.
    """
    return f"{FUNCTIONAL_THRESHOLD_LABEL} = {threshold}"


def empty_rule_layer() -> alt.Chart:
    """A rule layer with no rows, for a layer slot that has nothing to draw.

    Vega-Lite 6.4.1's normalizer crashes ("Cannot read properties of undefined
    (reading 'shape')") on a rule unit that has no encoding, so the value
    channels are declared even though no row is drawn.
    """
    return (
        alt.Chart(alt.Data(values=[]))
        .mark_rule()
        .encode(x=alt.value(0), y=alt.value(0))
    )


def dashed_threshold_rule_style(appearance: dict[str, Any]) -> dict[str, Any]:
    """Mark properties of the dashed threshold rule of ``graph_tree_persistence``.

    Read from the ``tree_threshold_*`` keys of ``[figures.graph_tree_persistence]``;
    every figure that draws that rule takes its colour, opacity, width and dash here.
    """
    return {
        "color": appearance["tree_threshold_color"],
        "opacity": float(appearance["tree_threshold_opacity"]),
        "strokeWidth": float(appearance["tree_threshold_width"]),
        "strokeDash": [float(value) for value in appearance["tree_threshold_dash"]],
    }


def functional_threshold_layers(
    threshold: float,
    *,
    scale: Any,
    axis: Any = None,
    color: str,
    dash: list[float],
    width: float,
    font_size: int,
    label_x: float,
    label_align: str = "right",
    label_below: bool = False,
    label_offset: float = THRESHOLD_LABEL_OFFSET,
    label_backing: str | None = None,
    label_text: str | list[str] | None = FUNCTIONAL_THRESHOLD_LABEL,
) -> list[Any]:
    """Draw the dashed functional-threshold rule and its ``Functional threshold`` label.

    ``label_text`` replaces the label (a figure names the Selection rule there);
    ``None`` draws the rule alone and returns just it (the tree figures name the
    threshold as a three-decimal axis tick instead, see ``publication_scalar_scales``).

    Returns the rule layer and the label layer, in that order; with ``label_backing``
    (a colour, normally white) a copy of the label in a thick ``label_backing``
    stroke goes between them, so the label stays legible where it crosses another
    rule or a mark drawn beneath it. ``scale`` and ``axis``
    encode the fitness (y) axis the rule is read against; ``axis`` is ``None`` when
    another layer owns it. The label sits ``label_offset`` pixels above the rule
    (below it with ``label_below``, for a panel whose data crowd the space above) at
    the pixel column ``label_x``, aligned ``label_align`` there, in ``color`` and
    ``font_size``.
    """
    frame = pl.DataFrame({"functional_threshold": [threshold]})
    y = alt.Y("functional_threshold:Q", scale=scale, axis=axis)
    rule = (
        alt.Chart(frame)
        .mark_rule(color=color, strokeDash=dash, strokeWidth=width)
        .encode(y=y)
    )
    if label_text is None:
        return [rule]
    label = (
        alt.Chart(frame)
        .mark_text(
            text=label_text,
            align=label_align,
            baseline="top" if label_below else "bottom",
            dy=label_offset if label_below else -label_offset,
            fontSize=font_size,
            color=color,
        )
        .encode(x=alt.value(label_x), y=y)
    )
    if label_backing is None:
        return [rule, label]
    backing = label.mark_text(
        text=label_text,
        align=label_align,
        baseline="top" if label_below else "bottom",
        dy=label_offset if label_below else -label_offset,
        fontSize=font_size,
        color=label_backing,
        stroke=label_backing,
        strokeWidth=4,
    )
    return [rule, backing, label]


def landscape_population_subtitle(
    quantities: ManuscriptQuantities, graph: str, *, include_graph: bool = False
) -> list[str]:
    """Describe a graph's population with the registry's own numbers.

    ``graph`` is ``"published"`` or ``"candidate"``; every number is the registry's
    display text for ``<graph>_graph.peak_count``, ``.vertex_count``,
    ``.functional_count`` and ``thresholds.<graph>_functional.value``.
    """
    reference = "Papkou et al. (2023)"
    if include_graph:
        state = {"published": "published", "candidate": "reflected_by"}[graph]
        reference = f"{graph_label(state)} · {reference}"
    peaks = quantities.display(f"{graph}_graph.peak_count")
    total = quantities.display(f"{graph}_graph.vertex_count")
    functional = quantities.display(f"{graph}_graph.functional_count")
    threshold = quantities.display(f"thresholds.{graph}_functional.value")
    return [
        f"{reference} · {peaks} peaks",
        (
            f"{functional} functional / {total} total sequences · "
            f"{functional_threshold_text(threshold)}"
        ),
    ]


def biology_axis_contract(
    chart: Any,
    *,
    y_title: str | None,
    x_title: str | None = None,
    y_tick_labels: bool = False,
) -> Any:
    """Apply the biological figure's single-label axis contract to a panel.

    With ``y_tick_labels`` a panel without a y title still shows its y tick
    labels (a diagram that shares the tree's fitness scale).
    """

    def visit(node: Any) -> Any:
        encoding = getattr(node, "encoding", alt.Undefined)
        if encoding is not alt.Undefined:
            encoding = encoding.to_dict()
            updates = {}
            for channel, title in (("y", y_title), ("x", x_title)):
                definition = encoding.get(channel)
                if not isinstance(definition, dict):
                    continue
                if channel == "y" and definition.get("field") not in {
                    "y",
                    "y_start",
                    "birth_scalar",
                }:
                    continue
                if channel == "x" and definition.get("field") != "death_scalar":
                    continue
                if title is None and channel == "y" and y_tick_labels:
                    axis = definition.get("axis")
                    definition["axis"] = {
                        **(axis if isinstance(axis, dict) else {}),
                        "title": None,
                    }
                elif title is None:
                    definition["axis"] = None
                else:
                    axis = definition.get("axis")
                    definition["axis"] = {
                        **(axis if isinstance(axis, dict) else {}),
                        "title": title,
                    }
                updates[channel] = definition
            if updates:
                if hasattr(node, "encode"):
                    node = node.encode(**updates)
                else:
                    for channel, value in updates.items():
                        setattr(node.encoding, channel, value)
        for composition in ("layer", "hconcat", "vconcat", "concat"):
            children = getattr(node, composition, alt.Undefined)
            if children is not alt.Undefined:
                node[composition] = [visit(child) for child in children]
        return node

    return visit(chart)


def plot_data_with_saved_metrics(tree_analysis: Any, branch_table: pl.DataFrame):
    """Join saved branch metrics to public plot edges by endpoint sequence IDs."""
    required = {
        "source_tree_node_identity",
        "target_tree_node_identity",
        "branch_vertex_count",
        "aa_hamming",
        "horton_strahler_order",
    }
    missing = required - set(branch_table.columns)
    if missing:
        raise ValueError(f"Saved branch table is missing metrics: {sorted(missing)}")
    metrics: dict[tuple[str, str], dict[str, int]] = {}
    for row in branch_table.select(*sorted(required)).to_dicts():
        key = tuple(
            sorted(
                (
                    str(row["source_tree_node_identity"]),
                    str(row["target_tree_node_identity"]),
                )
            )
        )
        if key in metrics:
            raise ValueError(f"Duplicate saved metrics for branch endpoints {key!r}.")
        metrics[key] = {
            name: int(row[name])
            for name in ("branch_vertex_count", "aa_hamming", "horton_strahler_order")
        }

    plot_data = tree_analysis.plot_data
    node_labels = {
        int(row["tree_node_index"]): str(row["dna_sequence"])
        for row in plot_data.node_table.to_dicts()
        if row.get("dna_sequence") is not None
    }
    edges = plot_data.edge_table
    attached = {
        name: []
        for name in ("branch_vertex_count", "aa_hamming", "horton_strahler_order")
    }
    for row in edges.to_dicts():
        try:
            endpoints = tuple(
                sorted(
                    (
                        node_labels[int(row["source_tree_node_index"])],
                        node_labels[int(row["target_tree_node_index"])],
                    )
                )
            )
            values = metrics[endpoints]
        except KeyError as error:
            raise ValueError(
                f"Displayed tree edge has no saved metric: {error}."
            ) from error
        for name, values_list in attached.items():
            values_list.append(values[name])
    edge_table = edges.with_columns(
        *(pl.Series(name, values, dtype=pl.Int64) for name, values in attached.items())
    ).with_columns(pl.col("branch_vertex_count").alias("vertex_count"))
    # The tree renderer needs labels for tooltips; these are stable DNA identities.
    edge_table = add_edge_labels(edge_table, node_labels)
    segment_table = plot_data.edge_segment_table
    if segment_table.height:
        segment_values = {name: [] for name in attached}
        for row in segment_table.to_dicts():
            endpoints = tuple(
                sorted(
                    (
                        node_labels[int(row["source_tree_node_index"])],
                        node_labels[int(row["target_tree_node_index"])],
                    )
                )
            )
            try:
                values = metrics[endpoints]
            except KeyError as error:
                raise ValueError(
                    f"Displayed tree segment has no saved metric: {error}."
                ) from error
            for name, values_list in segment_values.items():
                values_list.append(values[name])
        segment_table = segment_table.with_columns(
            *(
                pl.Series(name, values, dtype=pl.Int64)
                for name, values in segment_values.items()
            )
        )
        segment_table = segment_table.with_columns(
            pl.col("branch_vertex_count").alias("vertex_count")
        )
        segment_table = add_edge_labels(segment_table, node_labels)
    return replace(plot_data, edge_table=edge_table, edge_segment_table=segment_table)


def _scalar_bounds(
    scalar_domain: tuple[float, float],
    rendering: dict[str, Any],
    tick_step: float = 0.1,
    padding_fraction: float = 0.05,
) -> tuple[float, float, list[float]]:
    """Outward-rounded bounds and the padded display domain of a scalar axis."""
    tick_step = float(rendering.get("fitness_tick_step", tick_step))
    padding_fraction = float(
        rendering.get("fitness_padding_fraction", padding_fraction)
    )
    if tick_step <= 0 or padding_fraction < 0:
        raise ValueError("Fitness tick step must be positive and padding non-negative.")
    precision = max(0, math.ceil(-math.log10(tick_step)))
    minimum = round(math.floor(scalar_domain[0] / tick_step) * tick_step, precision)
    maximum = round(math.ceil(scalar_domain[1] / tick_step) * tick_step, precision)
    span = maximum - minimum or tick_step
    return (
        minimum,
        maximum,
        [minimum - padding_fraction * span, maximum + padding_fraction * span],
    )


def padded_scalar_domain(
    scalar_domain: tuple[float, float], rendering: dict[str, Any] | None = None
) -> list[float]:
    """Display domain ``publication_scalar_scales`` gives every scalar axis."""
    return _scalar_bounds(scalar_domain, rendering or {})[2]


def publication_scalar_scales(
    chart: Any,
    scalar_domain: tuple[float, float],
    rendering: dict[str, Any] | None = None,
    *,
    tick_step: float = 0.1,
    padding_fraction: float = 0.05,
    extra_ticks: dict[str, list[float]] | None = None,
    exact_domain: bool = False,
    y_axis_extent: int | None = None,
    exact_ticks: dict[str, list[float]] | None = None,
) -> Any:
    """Apply shared outward-rounded bounds, ticks, and padding to scalar axes.

    Every scalar axis gets exactly three ticks (the padded display domain's outward-
    rounded lower end, 0 and its upper end), written with one decimal.

    ``extra_ticks`` maps a field name to additional tick values (for example a
    functional threshold) merged into the standard ticks of that field's axes and
    written with three decimals through ``labelExpr``.

    ``exact_domain`` draws the outward-rounded bounds (one decimal, no padding) as
    the display domain and puts the two end ticks exactly there, so tick position
    and its one-decimal label agree (the tree figures). An extra tick within ~0.05
    of a regular tick would overprint it; the current thresholds (-0.508, -0.253)
    are safe.

    ``y_axis_extent`` fixes the width of every visible y scalar axis to that many
    pixels (equal ``minExtent`` and ``maxExtent``), so panels drawn in separate
    concats start their plot areas at the same x whatever their tick labels are.

    ``exact_ticks`` maps a field name to the only tick values of that field's axes,
    each written with three decimals (in place of the standard three).
    """
    rendering = rendering or {}
    tick_step = float(rendering.get("fitness_tick_step", tick_step))
    minimum, maximum, domain = _scalar_bounds(
        scalar_domain, rendering, tick_step, padding_fraction
    )
    precision = max(0, math.ceil(-math.log10(tick_step)))
    if exact_domain:
        domain = [minimum, maximum]
    ticks = sorted(
        {minimum, 0.0, maximum}
        if exact_domain
        else {round(minimum, precision), 0.0, round(maximum, precision)}
    )
    scalar_fields = {
        "y",
        "scalar",
        "bin_min",
        "birth_scalar",
        "death_scalar",
    }

    def apply(node: Any) -> Any:
        encoding = getattr(node, "encoding", alt.Undefined)
        if encoding is not alt.Undefined:
            for channel in ("x", "y"):
                axis = getattr(encoding, channel, alt.Undefined)
                if axis is alt.Undefined:
                    continue
                axis_spec = dict(axis) if isinstance(axis, dict) else axis.to_dict()
                if axis_spec.get("field") not in scalar_fields:
                    continue
                axis_spec["scale"] = {"domain": domain, "nice": False, "zero": False}
                if axis_spec.get("axis", alt.Undefined) is not None:
                    extras = sorted(
                        {float(v) for v in (extra_ticks or {}).get(axis_spec["field"], ())}
                    )
                    axis_options = {
                        **(axis_spec.get("axis") or {}),
                        "values": sorted({*ticks, *extras}),
                        "format": ".1f",
                    }
                    exact = (exact_ticks or {}).get(axis_spec["field"])
                    if exact is not None:
                        axis_options["values"] = sorted(float(v) for v in exact)
                        axis_options["format"] = ".3f"
                    if y_axis_extent is not None and channel == "y":
                        axis_options["minExtent"] = int(y_axis_extent)
                        axis_options["maxExtent"] = int(y_axis_extent)
                    if extras and exact is None:
                        # The three regular ticks keep one decimal; an extra tick
                        # (a functional threshold) is written with three.
                        is_extra = " || ".join(
                            f"abs(datum.value - ({value!r})) < 1e-9" for value in extras
                        )
                        axis_options["labelExpr"] = (
                            f"format(datum.value, ({is_extra}) ? '.3f' : '.1f')"
                        )
                    axis_spec["axis"] = axis_options
                if hasattr(node, "encode"):
                    node = node.encode(**{channel: axis_spec})
                else:
                    setattr(node.encoding, channel, axis_spec)
        for composition in ("layer", "hconcat", "vconcat", "concat"):
            children = getattr(node, composition, alt.Undefined)
            if children is not alt.Undefined:
                node[composition] = [apply(child) for child in children]
        return node

    return apply(chart)


def selected_view_as_tree(
    view: Any, *, legacy_mixture: bool = False
) -> SimpleNamespace:
    """Expose saved band-branch counts to the tree renderer."""
    from ..results.presentation import SavedTreeLayout
    from ..simplification.selected_leaf_simplification import SelectedLeafView

    if legacy_mixture or isinstance(view, SelectedLeafView):
        from ._legacy_mixture import selected_view_as_tree as legacy_selected_view

        return legacy_selected_view(view)
    if not isinstance(view, SavedTreeLayout):
        raise TypeError(
            "Expected a saved tree layout or an in-memory SelectedLeafView."
        )
    return SimpleNamespace(
        plot_data=view.plot_data,
        branch_table=view.branch_table,
        wild_type_marker_table=view.wild_type_marker_table,
    )


def plot_data_with_metrics(tree_analysis: Any):
    """Attach branch metrics by undirected tree-edge identity."""
    plot_data = tree_analysis.plot_data
    branch_table = tree_analysis.branch_table
    if "branch_vertex_count" not in branch_table.columns:
        raise ValueError(
            "The split-tree branch table has no branch_vertex_count metric."
        )
    metric_names = [
        name
        for name in ("branch_vertex_count", "synonymous_endpoints")
        if name in branch_table.columns
    ]
    metrics = branch_table.select(
        "source_tree_node_index", "target_tree_node_index", *metric_names
    )
    edge_table = add_metrics(plot_data.edge_table, metrics)
    segment_table = add_metrics(plot_data.edge_segment_table, metrics)
    node_labels = {
        int(row["tree_node_index"]): str(row.get("dna_sequence", ""))
        for row in plot_data.node_table.to_dicts()
    }
    edge_table = add_edge_labels(edge_table, node_labels)
    segment_table = add_edge_labels(segment_table, node_labels)
    # Display direction can vary across equal-scalar events; branch metrics are
    # joined by undirected tree-edge identity above.
    if {"target_scalar", "source_scalar"}.issubset(
        edge_table.columns
    ) and edge_table.filter(pl.col("target_scalar") < pl.col("source_scalar")).height:
        raise ValueError("Displayed split-tree edges must target their upper endpoint.")
    return replace(plot_data, edge_table=edge_table, edge_segment_table=segment_table)


def add_metrics(table: pl.DataFrame, metrics: pl.DataFrame) -> pl.DataFrame:
    if not metrics.schema["branch_vertex_count"].is_integer():
        raise ValueError("Branch vertex counts must be nonnegative integers.")
    if metrics.filter(pl.col("branch_vertex_count") < 0).height:
        raise ValueError("Branch vertex counts must be nonnegative integers.")
    # The branch table is authoritative; Topographer's plot data may carry its
    # own copy of a metric, which would otherwise be suffixed by the join.
    edge_keys = table.drop(
        [
            name
            for name in metrics.columns
            if name in table.columns
            and name not in ("source_tree_node_index", "target_tree_node_index")
        ]
    ).with_columns(
        pl.min_horizontal("source_tree_node_index", "target_tree_node_index").alias(
            "_metric_vertex_low"
        ),
        pl.max_horizontal("source_tree_node_index", "target_tree_node_index").alias(
            "_metric_vertex_high"
        ),
    )
    metric_keys = metrics.with_columns(
        pl.min_horizontal("source_tree_node_index", "target_tree_node_index").alias(
            "_metric_vertex_low"
        ),
        pl.max_horizontal("source_tree_node_index", "target_tree_node_index").alias(
            "_metric_vertex_high"
        ),
    ).drop("source_tree_node_index", "target_tree_node_index")
    if (
        metric_keys.select("_metric_vertex_low", "_metric_vertex_high")
        .is_duplicated()
        .any()
    ):
        raise ValueError("Duplicate scientific branch metrics for an undirected edge.")
    result = edge_keys.join(
        metric_keys,
        on=["_metric_vertex_low", "_metric_vertex_high"],
        how="left",
        validate="m:1",
    ).drop("_metric_vertex_low", "_metric_vertex_high")
    if result["branch_vertex_count"].null_count():
        raise ValueError("A displayed edge has no matching scientific branch metric.")
    # ``branch_vertex_count`` already includes the branch's upper critical vertex.
    result = result.with_columns(pl.col("branch_vertex_count").alias("vertex_count"))
    if "synonymous_endpoints" not in result.columns:
        result = result.with_columns(pl.lit(0).alias("synonymous_endpoints"))
    return result


def add_edge_labels(table: pl.DataFrame, node_labels: dict[int, str]) -> pl.DataFrame:
    return table.with_columns(
        pl.Series(
            "source_label",
            [node_labels.get(int(i), "") for i in table["source_tree_node_index"]],
        ),
        pl.Series(
            "target_label",
            [node_labels.get(int(i), "") for i in table["target_tree_node_index"]],
        ),
    )


def count_domain(edge_table: pl.DataFrame) -> tuple[float, float]:
    maximum = int(edge_table["vertex_count"].max() or 1) if edge_table.height else 1
    return 1.0, float(max(1, maximum))


def persistence_with_counts(
    persistence_table: pl.DataFrame,
    plot_data: Any,
    count_domain: tuple[float, float],
) -> pl.DataFrame:
    """Color each pair by the immediate displayed branch below its birth."""
    branch_counts = plot_data.edge_table.select("target_label", "vertex_count")
    if branch_counts["target_label"].n_unique() != branch_counts.height:
        raise ValueError(
            "Duplicate branch endpoint identities make birth counts ambiguous."
        )
    birth_counts = {
        str(row["target_label"]): int(row["vertex_count"])
        for row in branch_counts.to_dicts()
    }
    # A one-vertex tree has no edge; its global extremum still owns one vertex.
    node_labels = set(plot_data.node_table["dna_sequence"].drop_nulls().to_list())
    counts = []
    for row in persistence_table.to_dicts():
        identity = str(row["birth_label"])
        if identity in birth_counts:
            counts.append(birth_counts[identity])
        elif not plot_data.edge_table.height and identity in node_labels:
            counts.append(1)
        else:
            raise ValueError(
                f"Persistence birth {identity!r} has no displayed branch below its maximum."
            )
    if counts and max(counts) > count_domain[1]:
        raise ValueError("Persistence branch count exceeds the full-tree color domain.")
    return persistence_table.with_columns(
        pl.Series("vertex_count", counts, dtype=pl.Int64)
    )


def color_scale(rendering: dict[str, Any], count_domain: tuple[float, float]):
    return branch_count_color_scale(
        rendering,
        count_domain,
        attribute="vertex_count",
        show_legend=False,
    )


def multiline_title(value: str) -> str:
    return value.replace("\\n", "\n")


def tree_plot_config(
    analysis: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    edge_table: pl.DataFrame,
    count_domain: tuple[float, float],
    *,
    coloring: str = "count",
    horizontal_domain: tuple[float, float] | None = None,
) -> TreePlotConfig:
    """The tree panel configuration; ``coloring="none"`` draws one plain edge colour.

    ``horizontal_domain`` pins the tree's x scale to that extent with no pixel padding,
    so the drawn tree reaches the panel's left and right edges.
    """
    appearance = figure_settings.get("tree", {})

    plain = (
        {"edge_color": str(rendering.get("plain_edge_color", COUNT_BLACK))}
        if coloring == "none"
        else {}
    )
    if horizontal_domain is not None:
        plain = {
            **plain,
            "horizontal_domain": horizontal_domain,
            "horizontal_padding": 0,
        }
    return TreePlotConfig(
        **plain,
        width=int(appearance.get("width", rendering.get("tree_width", TREE_WIDTH))),
        height=int(appearance.get("height", rendering.get("tree_height", TREE_HEIGHT))),
        font_size=int(appearance.get("font_size", rendering.get("font_size", 15))),
        scalar_axis_title=rendering.get(
            "tree_axis_title", "Fitness (at birth, WT = 0)"
        ),
        scalar_domain=analysis.scalar_domain,
        edge_width=float(rendering.get("edge_width", 1.1)),
        node_size=int(rendering.get("node_size", 24)),
        edge_color_scale=None
        if coloring == "none"
        else color_scale(rendering, count_domain),
    )


def compose_tree_layers(
    branches: Any,
    *,
    underlays: Sequence[Any] = (),
    overlays: Sequence[Any] = (),
    count_labels: Sequence[Any] = (),
    stop_markers: Sequence[Any] = (),
    keys: Sequence[Any] = (),
    wild_type: Sequence[Any] = (),
) -> Any:
    """Stack the layers of one tree panel in the repository's fixed order.

    Bottom to top: the ``underlays`` (a rule drawn behind the branches), the tree's
    ``branches`` and the ``overlays`` drawn on them
    (highlighted or absorbed branches), the ``count_labels``, the ``stop_markers``
    (stop-codon crosses), the ``keys`` (inset legends) and last the ``wild_type``
    star and its label. A caller says what a layer is, never where it goes: the
    keyword names fix the order whatever order they are written in, so no branch
    overlay or count label can hide a stop-codon cross and nothing hides the star.
    ``tree_chart`` and the biology rows (``assemble_biology_chart``) both stack
    their panel here; the colour bar, which ``overlay_count_colorbar`` adds, sits in
    a free corner over the finished panel.
    """
    return alt.layer(
        *underlays,
        branches,
        *overlays,
        *count_labels,
        *stop_markers,
        *keys,
        *wild_type,
    )


# The one legend label for stop-codon peaks, on the tree and the diagram key.
STOP_CODON_LEGEND_LABEL = "Stop codon containing peaks"
# The diagram key's synonymous entry in the combined encoding (a teal filled circle).
SYNONYMOUS_LEGEND_LABEL = "synonymous"
ENCODED_STOP_LEGEND_LABEL = "≥1 stop codon"

LEAF_ROW_NOUNS = {
    "stops": "stop-codon containing peaks",
    "synonymous": "synonymous leaves",
}


def leaf_row_label(kind: str, count_text: str | None = None) -> str:
    """The words naming one leaf row: ``kind`` is ``"stops"`` or ``"synonymous"``.

    With ``count_text`` (a printed count) it reads ``<count> <noun>``; without it the
    noun alone, capitalised, for a row heading that has no single count.
    """
    noun = LEAF_ROW_NOUNS[kind]
    return f"{count_text} {noun}" if count_text is not None else noun.capitalize()


def leaf_row_title(
    kind: str, count_text: str | None = None, subtitle: Any = None
) -> alt.TitleParams:
    """Heading of one leaf row (the Published biology figure's and the saved-tree views')."""
    options: dict[str, Any] = {} if subtitle is None else {"subtitle": subtitle}
    return alt.TitleParams(
        text=leaf_row_label(kind, count_text), anchor="start", **options
    )


def stack_leaf_rows(rows: Sequence[Any], rendering: dict[str, Any]) -> Any:
    """Stack the stop-codon row above the synonymous row, every scale independent.

    One composition for every two-row figure: the Published graph's
    ``study_tree_persistence_biology`` and the saved-tree views draw stops in one row
    and synonymous leaves in the other, never both in one chart, and stack them here.
    """
    return alt.vconcat(
        *rows, spacing=int(rendering.get("comparison_panel_gap", 16))
    ).resolve_scale(x="independent", y="independent", color="independent")


def _tight_horizontal_domain(
    plot_data: Any, figure_settings: dict[str, Any]
) -> tuple[float, float] | None:
    """The exact x extent of the tree when the figure's ``tree`` table asks for it.

    ``tight_horizontal = true`` drops Vega's zero-inclusion, outward rounding and
    Topographer's pixel padding, so the tree touches its panel's left and right edges
    (the vertical-histogram figure's panels meet). ``None`` keeps the default scale.
    """
    if not figure_settings.get("tree", {}).get("tight_horizontal", False):
        return None
    segments = plot_data.edge_segment_table.filter(
        (pl.col("x_start") != pl.col("x_end")) | (pl.col("y_start") != pl.col("y_end"))
    )
    xs = pl.concat([segments["x_start"], segments["x_end"]])
    return float(xs.min()), float(xs.max())


def _without_nice_x(chart: Any) -> Any:
    """The chart with every pinned x scale told not to round its domain outward.

    Vega-Lite rounds an explicit domain to a ``nice`` one by default, which would leave
    blank room at the right of a tree whose x extent was pinned exactly.
    """

    def visit(node: Any) -> Any:
        encoding = getattr(node, "encoding", alt.Undefined)
        if encoding is not alt.Undefined:
            channel = getattr(encoding, "x", alt.Undefined)
            if channel is not alt.Undefined:
                spec = dict(channel) if isinstance(channel, dict) else channel.to_dict()
                scale = spec.get("scale")
                if isinstance(scale, dict) and "domain" in scale:
                    spec["scale"] = {**scale, "nice": False}
                    node = node.encode(x=spec)
        for composition in ("layer", "hconcat", "vconcat", "concat"):
            children = getattr(node, composition, alt.Undefined)
            if children is not alt.Undefined:
                node[composition] = [visit(child) for child in children]
        return node

    return visit(chart)


def tree_chart(
    plot_data: Any,
    tree_analysis: Any,
    analysis: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    *,
    linked: bool,
    annotated: bool,
    stop_identities: set[str] | None = None,
    annotation_key: bool = True,
    underlays: Sequence[Any] = (),
    overlays: Sequence[Any] = (),
    count_labels: Sequence[Any] = (),
    coloring: str = "count",
) -> Any:
    """Draw one tree panel.

    ``coloring`` is ``"count"`` (branches coloured by vertex count) or ``"none"``
    (every branch in ``plain_edge_color``, no count colour scale; the caller then
    skips ``overlay_count_colorbar``, which sits over the panel and so never moves it).

    ``stop_identities`` restricts the annotated stop markers to those sequence
    identities (for example peaks only); ``annotation_key`` toggles the lower-left
    marker legend. ``underlays`` (a rule behind the branches), ``overlays``
    (highlighted branches) and ``count_labels`` (or any text that must stay beneath
    the markers) are layers the caller hands in rather than layers it draws on the
    result: ``compose_tree_layers`` stacks them, so the underlays sit below the
    branches, branches and overlays below the count labels, which sit below the
    stop-codon crosses, which sit below the wild-type star.
    """
    # Ascending by branch count so higher counts draw last, on top.
    plot_data = replace(
        plot_data,
        edge_table=plot_data.edge_table.sort("vertex_count", maintain_order=True),
    )
    horizontal_domain = _tight_horizontal_domain(plot_data, figure_settings)
    chart = plot_tree(
        plot_data,
        config=tree_plot_config(
            analysis,
            figure_settings,
            {
                **rendering,
                "tree_axis_title": rendering.get(
                    "shared_tree_axis_title", "Fitness (at birth, WT = 0)"
                )
                if linked
                else rendering.get("tree_axis_title", "Fitness (WT = 0)"),
            },
            plot_data.edge_table,
            count_domain,
            coloring=coloring,
            horizontal_domain=horizontal_domain,
        ),
        presentation=TreePresentationConfig(mode="metric"),
        tooltip_attrs=("dna_sequence", "aa_sequence"),
        edge_tooltip_attrs=(
            "vertex_count",
            "synonymous_endpoints",
            "source_label",
            "target_label",
        ),
        show_structural_events=False,
        interactive=False,
        include_config=False,
    )
    if horizontal_domain is not None:
        chart = _without_nice_x(chart)
    width = int(
        figure_settings.get("tree", {}).get(
            "width", rendering.get("tree_width", TREE_WIDTH)
        )
    )
    height = int(
        figure_settings.get("tree", {}).get(
            "height", rendering.get("tree_height", TREE_HEIGHT)
        )
    )
    stop_markers = (
        [stop_marker_layer(tree_analysis, rendering, identities=stop_identities)]
        if annotated
        else []
    )
    wild_type = wild_type_layers(tree_analysis, rendering)
    # The key names the markers the panel draws: the wild-type star where there is
    # one, the stop-codon cross where stop codons are annotated.
    keys = (
        [
            annotation_legend(
                width,
                height,
                rendering,
                wild_type=bool(wild_type),
                stop_codon=annotated,
            )
        ]
        if annotation_key and (annotated or wild_type)
        else []
    )
    return compose_tree_layers(
        chart,
        underlays=underlays,
        overlays=overlays,
        count_labels=count_labels,
        stop_markers=stop_markers,
        keys=keys,
        wild_type=wild_type,
    ).resolve_scale(color="independent")


def tree_figure(
    plot_data,
    tree_analysis,
    analysis,
    figure_settings,
    rendering,
    count_domain,
    *,
    annotated,
    stop_identities=None,
    annotation_key=True,
    coloring="count",
    overlays=(),
):
    tree = tree_chart(
        plot_data,
        tree_analysis,
        analysis,
        figure_settings,
        rendering,
        count_domain,
        linked=False,
        annotated=annotated,
        stop_identities=stop_identities,
        annotation_key=annotation_key,
        coloring=coloring,
        overlays=overlays,
    )
    if coloring == "none":
        return tree
    return overlay_count_colorbar(tree, figure_settings, rendering, count_domain)


def overlay_count_colorbar(
    tree: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    *,
    width: int | None = None,
) -> Any:
    """Overlay the configured branch-count colour bar on one tree panel.

    ``width`` is the panel's pixel width when it is not the figure's ``tree`` table
    or the shared ``tree_width``.
    """
    return overlay_colorbar(
        tree,
        horizontal_count_legend(rendering, count_domain),
        x_fraction=float(figure_settings["colorbar_x_fraction"]),
        top_inset=float(rendering.get("colorbar_top_inset", 8)),
        width=_tree_panel_width(figure_settings, rendering, width),
    )


def count_colorbar_box(
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    *,
    width: int | None = None,
) -> tuple[float, float, float, float]:
    """Return the ``(left, top, right, bottom)`` pixel box of the count colour bar.

    The box is where ``overlay_count_colorbar`` puts the bar inside its tree panel
    (panel pixels, origin top-left, y down), so a mark can keep clear of it.
    """
    return colorbar_box(
        horizontal_count_legend(rendering, count_domain),
        x_fraction=float(figure_settings["colorbar_x_fraction"]),
        top_inset=float(rendering.get("colorbar_top_inset", 8)),
        width=_tree_panel_width(figure_settings, rendering, width),
    )


def _tree_panel_width(
    figure_settings: dict[str, Any], rendering: dict[str, Any], width: int | None
) -> int:
    if width is not None:
        return int(width)
    return int(
        figure_settings.get("tree", {}).get(
            "width", rendering.get("tree_width", TREE_WIDTH)
        )
    )


def marker_config(rendering: dict[str, Any], name: str, *, default_shape: str):
    """Return the tree marker of ``name`` (``wild_type`` or ``stop_codon``)."""
    default_colors = {"wild_type": WILD_TYPE_RED, "stop_codon": STOP_CODON_CRIMSON}
    default_sizes = {"wild_type": 200, "stop_codon": 42}
    color = rendering.get(f"{name}_color", default_colors[name])
    size = int(rendering.get(f"{name}_size", default_sizes[name]))
    shape = rendering.get(f"{name}_shape", default_shape)
    if shape == "x":
        shape = OPEN_X
    elif shape == "star":
        shape = FIVE_POINT_STAR
    label_color = rendering.get(f"{name}_label_color", color)
    label_dx = float(
        rendering.get(f"{name}_label_dx", -9 if name == "wild_type" else 9)
    )
    return TreeMarkerConfig(
        shape=shape,
        color=color,
        size=size,
        # Wild type and stop codon are solid (ADR 0012): no opacity setting applies.
        opacity=1.0,
        filled=name != "stop_codon",
        label_color=label_color,
        label_dx=label_dx,
        label_field="marker_label" if name == "wild_type" else None,
    )


def wild_type_layers(tree_analysis: Any, rendering: dict[str, Any]) -> list[Any]:
    """Draw the wild-type star and label after every biological marker layer."""
    if not hasattr(tree_analysis, "wild_type_marker_table"):
        return []
    marker = marker_config(rendering, "wild_type", default_shape="star")
    positions = alt.Chart(tree_analysis.wild_type_marker_table).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q"),
        tooltip=["dna_sequence:N", "aa_sequence:N"],
    )
    return [
        positions.mark_point(
            shape=marker.shape,
            color=marker.color,
            size=marker.size,
            filled=True,
            opacity=marker.opacity,
        ),
        positions.mark_text(
            dx=marker.label_dx,
            align="right",
            color=marker.label_color,
            fontSize=int(rendering.get("font_size", 15)),
        ).encode(text="marker_label:N"),
    ]


STOP_NODE_KINDS = ("maximum", "minimum", "saddle", "regular")
DEFAULT_STOP_NODE_KINDS = ("maximum", "minimum")


def stop_codon_node_kinds(rendering: dict[str, Any]) -> tuple[str, ...]:
    """The node kinds allowed to carry a stop X: ``[rendering] stop_codon_tree_node_kinds``.

    Allowed values are ``STOP_NODE_KINDS``; anything else raises ``ValueError``.
    """
    kinds = rendering.get("stop_codon_tree_node_kinds", DEFAULT_STOP_NODE_KINDS)
    if isinstance(kinds, str) or not isinstance(kinds, (list, tuple)):
        raise ValueError(
            "stop_codon_tree_node_kinds must be a list of node kinds from "
            f"{list(STOP_NODE_KINDS)}, got {kinds!r}."
        )
    unknown = [kind for kind in kinds if kind not in STOP_NODE_KINDS]
    if unknown:
        raise ValueError(
            f"Unknown stop_codon_tree_node_kinds {unknown}; allowed are "
            f"{list(STOP_NODE_KINDS)}."
        )
    return tuple(kinds)


def stop_codon_nodes(
    nodes: pl.DataFrame, rendering: dict[str, Any], *, stop_only: bool = False
) -> pl.DataFrame:
    """The tree nodes that may carry a stop X.

    A node is kept when its own sequence has a stop codon (``has_stop_codon``) and its
    kind is in ``stop_codon_tree_node_kinds``. ``stop_only=True`` says the table is
    already filtered to stop-codon sequences, so it needs no ``has_stop_codon``
    column (a column that is present is still applied); without it a table lacking the column raises ``ValueError``.
    The kind is ``display_event`` (the global minimum, the root, reads ``minimum``)
    or else ``event``, where ``is_root`` marks the global minimum; ``*_saddle`` events
    are ``saddle`` and anything unrecognised is ``regular``.
    """
    kinds = stop_codon_node_kinds(rendering)
    if nodes.is_empty():
        return nodes
    columns = nodes.columns
    if "has_stop_codon" not in columns and not stop_only:
        raise ValueError(
            "A stop-codon node table needs a has_stop_codon column, or stop_only=True "
            f"when it is already filtered to stop sequences; it has {columns}."
        )
    if "display_event" in columns:
        raw = pl.col("display_event")
    elif "event" in columns:
        raw = pl.col("event")
    else:
        raise ValueError(
            "A stop-codon node table needs a display_event or event column to "
            f"select node kinds; it has {columns}."
        )
    kind = (
        pl.when(raw.is_in(["maximum", "minimum"]))
        .then(raw)
        .when(raw.cast(pl.String).str.ends_with("saddle"))
        .then(pl.lit("saddle"))
        .otherwise(pl.lit("regular"))
    )
    if "display_event" not in columns and "is_root" in columns:
        kind = pl.when(pl.col("is_root")).then(pl.lit("minimum")).otherwise(kind)
    keep = kind.is_in(list(kinds))
    if "has_stop_codon" in columns:
        keep = keep & pl.col("has_stop_codon").fill_null(False)
    return nodes.filter(keep)


def stop_marker_layer(
    tree_analysis: Any,
    rendering: dict[str, Any],
    *,
    identities: set[str] | None = None,
):
    """Draw stop-codon sequences as solid crimson crosses (diagonal X marks).

    Only nodes of the kinds in ``stop_codon_tree_node_kinds`` carry a cross.
    ``identities`` keeps only the listed DNA sequences.
    """
    table = tree_analysis.all_stop_marker_table
    plot_nodes = getattr(getattr(tree_analysis, "plot_data", None), "node_table", None)
    if (
        plot_nodes is not None
        and "display_event" not in table.columns
        and {"tree_node_index", "display_event"} <= set(plot_nodes.columns)
        and "tree_node_index" in table.columns
        and plot_nodes["tree_node_index"].is_unique().all()
    ):
        # Topographer's stop table has no display_event; only the plot-data node table
        # reads the root as ``minimum``. tree_node_index is one row per node there, so
        # the left join cannot duplicate rows; swarm members (null index) keep ``event``.
        kinds = plot_nodes.select("tree_node_index", "display_event")
        table = table.join(kinds, on="tree_node_index", how="left").with_columns(
            pl.coalesce("display_event", "event").alias("display_event")
            if "event" in table.columns
            else pl.col("display_event")
        )
    points = stop_codon_nodes(table, rendering, stop_only=True)
    if identities is not None:
        points = points.filter(pl.col("dna_sequence").is_in(sorted(identities)))
    return stop_tree_points_layer(points, rendering)


def stop_tree_points_layer(points: pl.DataFrame, rendering: dict[str, Any]):
    """Draw the supplied sequence positions with the shared stop marker style."""
    marker = marker_config(rendering, "stop_codon", default_shape="x")
    return (
        alt.Chart(points)
        .mark_point(
            shape=marker.shape,
            color=marker.color,
            size=int(rendering.get("stop_codon_tree_size", marker.size)),
            opacity=float(rendering.get("stop_codon_tree_opacity", 0.7)),
            filled=False,
            strokeWidth=float(rendering.get("stop_codon_stroke_width", 0.5)),
        )
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q"),
            tooltip=["dna_sequence:N", "aa_sequence:N"],
        )
    )


def persistence_chart(
    table,
    figure_settings,
    rendering,
    count_domain,
    scalar_domain,
    *,
    include_config=True,
):
    appearance = figure_settings.get("persistence", {})
    config = PersistencePlotConfig(
        width=int(
            appearance.get(
                "width", rendering.get("persistence_width", PERSISTENCE_WIDTH)
            )
        ),
        height=int(
            appearance.get(
                "height", rendering.get("persistence_height", PERSISTENCE_HEIGHT)
            )
        ),
        font_size=int(appearance.get("font_size", rendering.get("font_size", 15))),
        birth_axis_title=rendering.get(
            "persistence_y_title", "Fitness (at birth, WT = 0)"
        ),
        death_axis_title=rendering.get("persistence_x_title", "Fitness at death"),
        domain=scalar_domain,
        point_size=int(
            appearance.get("point_size", rendering.get("persistence_point_size", 65))
        ),
        show_grid=bool(rendering.get("persistence_grid", False)),
        color_scale=color_scale(rendering, count_domain),
    )
    chart = plot_persistence(
        table,
        config=config,
        color_by="vertex_count",
        draw_order_by="vertex_count",
        shape_by=None,
        separate_event_layers=False,
        tooltip_attrs=(
            "persistence",
            "birth_label",
            "death_label",
            "pair_kind",
            "vertex_count",
        ),
        include_config=False,
    )
    return set_count_point_opacity(
        chart,
        float(rendering.get("count_point_opacity", 1.0)),
        shape=rendering.get("persistence_point_shape", "triangle-up"),
    )


def set_plain_point_color(chart: Any, color: str, *, color_field: str = "vertex_count"):
    """Draw count-coloured persistence points in one colour (``coloring="none"``)."""
    for layer in getattr(chart, "layer", ()):
        spec = getattr(getattr(layer, "encoding", None), "color", None)
        if spec is None or spec is alt.Undefined:
            continue
        spec = spec if isinstance(spec, dict) else spec.to_dict()
        if spec.get("field") == color_field:
            layer.encoding.color = alt.value(color)
    return chart


def set_count_point_opacity(
    chart: Any,
    opacity: float,
    *,
    shape: str,
    color_field: str = "vertex_count",
):
    """Make persistence hues match the fully opaque tree and legend scale."""
    for layer in getattr(chart, "layer", ()):
        mark = getattr(layer, "mark", None)
        color = getattr(getattr(layer, "encoding", None), "color", None)
        if (
            mark is None
            or mark is alt.Undefined
            or color is None
            or color is alt.Undefined
        ):
            continue
        mark_spec = mark if isinstance(mark, dict) else mark.to_dict()
        color_spec = color if isinstance(color, dict) else color.to_dict()
        if mark_spec.get("type") == "point" and color_spec.get("field") == color_field:
            layer.mark = alt.MarkDef(
                **{**mark_spec, "opacity": opacity, "shape": shape}
            )
    return chart


# What the annotation adds to a peak's own tooltip name.
PEAK_ANNOTATION_NOTES = {
    "wild_type": "wild type",
    "stop_codon": "stop-codon containing",
}


def wild_type_identities(source: Any) -> frozenset[str]:
    """Return the wild-type DNA identities of a tree or a derived view.

    ``source`` carries a ``wild_type_marker_table``. The wild type is a peak of a
    persistence panel only if its identity is also a persistence birth.
    """
    table = getattr(source, "wild_type_marker_table", None)
    if table is None or table.is_empty() or "dna_sequence" not in table.columns:
        return frozenset()
    return frozenset(str(value) for value in table["dna_sequence"].to_list())


def peak_annotation_style(
    kind: str | None, rendering: dict[str, Any], *, color: str
) -> tuple[str, float]:
    """Return the colour and opacity of a persistence-panel peak triangle.

    Colour means annotation (ADR 0012): wild type is red and a stop-codon peak
    crimson, each solid (opacity 1). A peak with no annotation keeps ``color``,
    its own.
    """
    if kind == "wild_type":
        return str(rendering.get("wild_type_color", WILD_TYPE_RED)), 1.0
    if kind == "stop_codon":
        return str(rendering.get("stop_codon_color", STOP_CODON_CRIMSON)), 1.0
    return color, 1.0


def peak_annotation_kinds(
    table: pl.DataFrame,
    *,
    wild_type: frozenset[str] = frozenset(),
    label: str = "birth_label",
) -> list[str | None]:
    """Return the annotation each peak of a persistence table carries, one per row.

    A peak carries at most one, taken in the order wild type, stop codon; ``None``
    is a plain peak. ``label`` names the column holding the birth's identity. Stop
    codon is the table's ``birth_has_stop_codon``: a stop in the paired endpoint
    alone does not qualify.
    """
    labels = table[label].to_list()
    if "birth_has_stop_codon" in table.columns:
        stops = [bool(value) for value in table["birth_has_stop_codon"].to_list()]
    else:
        stops = [False] * table.height
    return [
        "wild_type" if identity in wild_type else "stop_codon" if stop else None
        for identity, stop in zip(labels, stops, strict=True)
    ]


def rows_with_annotation(
    table: pl.DataFrame, kinds: list[str | None], kind: str | None
) -> pl.DataFrame:
    """Return the rows of ``table`` whose annotation (see ``peak_annotation_kinds``) is ``kind``."""
    return table.filter(pl.Series([value == kind for value in kinds], dtype=pl.Boolean))


def unannotated_peaks(table: pl.DataFrame, **annotations: Any) -> pl.DataFrame:
    """Return the plain peaks of a persistence table: those carrying no annotation.

    ``annotations`` are the keywords of ``peak_annotation_kinds``. An annotated
    diagram draws its annotated peaks in their annotation colour, in place of the
    plain peak colour, so the base layer keeps only these rows.
    """
    return rows_with_annotation(
        table, peak_annotation_kinds(table, **annotations), None
    )


def annotated_peaks(table: pl.DataFrame, **annotations: Any) -> pl.DataFrame:
    """Return the annotated peaks of a persistence table, the complement of ``unannotated_peaks``."""
    kinds = peak_annotation_kinds(table, **annotations)
    return table.filter(
        pl.Series([kind is not None for kind in kinds], dtype=pl.Boolean)
    )


def synonymous_leaf_edges(plot_data: Any) -> pl.DataFrame:
    """The displayed terminal peak branches whose adjacent saddle has the peak's amino acids.

    A branch is a synonymous leaf when it is a terminal peak branch (its upper end is a
    ``maximum``) and its lower end is a saddle with the same ``aa_sequence``. Returns the
    edge rows of ``plot_data.edge_table`` that qualify (``source_label`` the saddle,
    ``target_label`` the peak, as ``plot_data_with_metrics`` labels them), empty when
    there are no such edges. It is the definition of the Published graph's synonymous
    row, applied to any saved tree. Raises ``ValueError`` naming the missing column when
    the node table lacks ``tree_node_index``, ``aa_sequence`` or an event column
    (``display_event`` or ``event``): no synonymous leaf can be named without them, and
    an empty result would silently draw a figure with none.
    """
    nodes = plot_data.node_table
    edges = plot_data.edge_table
    event = "display_event" if "display_event" in nodes.columns else "event"
    missing = [
        name
        for name in ("tree_node_index", "aa_sequence", event)
        if name not in nodes.columns
    ]
    if missing:
        raise ValueError(
            "The synonymous leaves cannot be found: the node table lacks the column(s) "
            + ", ".join(repr(name) for name in missing)
            + " (it needs tree_node_index, aa_sequence and display_event or event)."
        )
    info = nodes.select(
        "tree_node_index",
        pl.col("aa_sequence"),
        pl.col(event).cast(pl.String).alias("_kind"),
    )
    upper = info.rename(
        {"tree_node_index": "target_tree_node_index", "aa_sequence": "_aa_peak", "_kind": "_kind_peak"}
    )
    lower = info.rename(
        {"tree_node_index": "source_tree_node_index", "aa_sequence": "_aa_saddle", "_kind": "_kind_saddle"}
    )
    return (
        edges.join(upper, on="target_tree_node_index", how="left")
        .join(lower, on="source_tree_node_index", how="left")
        .filter(
            (pl.col("_kind_peak") == "maximum")
            & pl.col("_kind_saddle").str.ends_with("saddle")
            & (pl.col("_aa_peak") == pl.col("_aa_saddle"))
        )
        .drop("_aa_peak", "_aa_saddle", "_kind_peak", "_kind_saddle")
    )


def synonymous_leaf_segments(
    plot_data: Any, edges: pl.DataFrame | None = None
) -> pl.DataFrame:
    """The orthogonal segments of the synonymous leaves' branches, for the green layer.

    ``edges`` defaults to ``synonymous_leaf_edges(plot_data)``. The segments are the rows
    of ``plot_data.edge_segment_table`` whose (source, target) label pair names one of
    those edges (order-free); empty when there are no such edges or labels.
    """
    segments = plot_data.edge_segment_table
    edges = synonymous_leaf_edges(plot_data) if edges is None else edges
    labels = ("source_label", "target_label")
    if edges.is_empty() or not set(labels) <= set(edges.columns):
        return segments.clear()
    keys = {
        tuple(sorted((row["source_label"], row["target_label"])))
        for row in edges.select(*labels).to_dicts()
    }
    flags = [
        tuple(sorted((row["source_label"], row["target_label"]))) in keys
        for row in segments.select(*labels).to_dicts()
    ]
    return segments.filter(pl.Series(flags, dtype=pl.Boolean))


def synonymous_branch_layer(
    segments: pl.DataFrame,
    rendering: dict[str, Any],
    *,
    keep_fitness_axis: bool = False,
):
    """Stroke every orthogonal segment of the selected peak branches in the synonymous green.

    The layer states ``axis=None`` on its y channel, which Vega-Lite merges with the
    tree's own axis into no axis at all; ``keep_fitness_axis`` leaves the axis
    undefined so the tree keeps its fitness axis (the D10 band view's synonymous rows).
    """
    # Coincident endpoints have no visible segment and cannot be stroked by
    # the PNG renderer. Their branch's nonzero segment remains highlighted.
    segments = segments.filter(
        (pl.col("x_start") != pl.col("x_end")) | (pl.col("y_start") != pl.col("y_end"))
    )
    if segments.is_empty():
        return empty_rule_layer()
    return (
        alt.Chart(segments)
        .mark_rule(
            color=rendering.get("synonymous_color", SYNONYMOUS_GREEN),
            strokeWidth=float(
                rendering.get("synonymous_edge_width", rendering.get("edge_width", 1.1))
            ),
            opacity=float(rendering.get("synonymous_opacity", 1.0)),
        )
        .encode(
            x=alt.X("x_start:Q", axis=None),
            x2="x_end:Q",
            y=alt.Y("y_start:Q") if keep_fitness_axis else alt.Y("y_start:Q", axis=None),
            y2="y_end:Q",
            detail="logical_edge_index:N",
            tooltip=["source_label:N", "target_label:N"],
        )
    )


def synonymous_valley_sequences(valleys: pl.DataFrame) -> frozenset[str]:
    """The valleys that are synonymous to the saddle they merge at.

    The mirror of the synonymous-leaf rule (``synonymous_leaf_edges``: a peak whose
    adjacent saddle has its amino acids): a valley is synonymous when its own
    amino-acid sequence equals that of the join saddle it dies at. ``valleys`` is a
    table of valley pairs, ``birth_sequence`` the valley and ``death_sequence`` the
    saddle (DNA identities, as in ``descending_component_pairs``' relative pairs);
    returns the valleys' DNA identities. Translation is the repository's codon table.
    """
    from ..analysis.peak_rank import translate

    if valleys.is_empty():
        return frozenset()
    found = set()
    untranslatable = 0
    for valley, saddle in zip(
        valleys["birth_sequence"].to_list(), valleys["death_sequence"].to_list(), strict=True
    ):
        if valley is None or saddle is None:
            continue
        try:
            same = translate(valley) == translate(saddle)
        except ValueError:
            # Not whole codons: no amino-acid sequence, so not synonymous.
            untranslatable += 1
            continue
        if same:
            found.add(str(valley))
    if untranslatable:
        LOGGER.warning(
            "%d valley pair(s) have a sequence that does not translate; not marked synonymous.",
            untranslatable,
        )
    return frozenset(found)


def synonymous_teal_branch_layer(
    segments: pl.DataFrame,
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    *,
    keep_fitness_axis: bool = False,
):
    """Stroke the synonymous leaf branches in teal, darker the more vertices they hold.

    The combined encoding's version of ``synonymous_branch_layer``: each branch takes
    the teal of its ``vertex_count`` on the same log scale as the grey count ramp
    (``branch_count_teal``), instead of the single green. ``segments`` carry
    ``vertex_count`` (``plot_data_with_metrics``' segment table does).
    """
    segments = segments.filter(
        (pl.col("x_start") != pl.col("x_end")) | (pl.col("y_start") != pl.col("y_end"))
    )
    if segments.is_empty():
        return empty_rule_layer()
    segments = segments.with_columns(
        pl.col("vertex_count")
        .map_elements(
            lambda count: branch_count_teal(count, rendering, count_domain),
            return_dtype=pl.String,
        )
        .alias("teal_color")
    )
    return (
        alt.Chart(segments)
        .mark_rule(
            strokeWidth=float(
                rendering.get("synonymous_edge_width", rendering.get("edge_width", 1.1))
            ),
            opacity=float(rendering.get("synonymous_opacity", 1.0)),
        )
        .encode(
            x=alt.X("x_start:Q", axis=None),
            x2="x_end:Q",
            y=alt.Y("y_start:Q") if keep_fitness_axis else alt.Y("y_start:Q", axis=None),
            y2="y_end:Q",
            color=alt.Color("teal_color:N", scale=None, legend=None),
            detail="logical_edge_index:N",
            tooltip=["source_label:N", "target_label:N", "vertex_count:Q"],
        )
    )


def tree_marker_size(rendering: dict[str, Any]) -> int:
    """The one marker size (mark area, px squared) of a combined-encoding figure.

    The tree's stop cross and the diagram's peak and valley triangles all read
    ``stop_codon_tree_size``; there is no second size setting.
    """
    marker = marker_config(rendering, "stop_codon", default_shape="x")
    return int(rendering.get("stop_codon_tree_size", marker.size))


def stop_cross_layer(
    rows: pl.DataFrame,
    rendering: dict[str, Any],
    *,
    x: Any = None,
    y: Any = None,
):
    """The stop-codon cross drawn on top of a persistence-diagram marker.

    The very same cross as the trees' stop marker (``stop_tree_points_layer``): shape,
    colour, size (``stop_codon_tree_size``), stroke width (``stop_codon_stroke_width``)
    and opacity (``stop_codon_tree_opacity``) all come from the tree's settings, so the
    × on a diagram and on its tree are indistinguishable. It sits at each row's (death,
    birth) point, so a point that is also synonymous reads as a teal marker with a red
    cross. ``x`` and ``y`` replace the default field encodings with the diagram's own
    scaled ones.
    """
    marker = marker_config(rendering, "stop_codon", default_shape="x")
    fields = [
        name
        for name in ("death_scalar", "birth_scalar", "birth_label", "death_label", "birth_sequence", "death_sequence")
        if name in rows.columns
    ]
    return (
        alt.Chart(rows.select(fields))
        .mark_point(
            shape=marker.shape,
            color=marker.color,
            size=tree_marker_size(rendering),
            opacity=float(rendering.get("stop_codon_tree_opacity", 0.7)),
            # The stop_codon cross is a stroke with no interior.
            filled=False,
            strokeWidth=float(rendering.get("stop_codon_stroke_width", 0.5)),
        )
        .encode(
            x=x if x is not None else alt.X("death_scalar:Q"),
            y=y if y is not None else alt.Y("birth_scalar:Q"),
            tooltip=[f"{name}:N" if "label" in name or "sequence" in name else f"{name}:Q" for name in fields],
        )
    )


def peak_marker_layers(
    table: pl.DataFrame,
    rendering: dict[str, Any],
    *,
    color: str,
    annotation: str,
    wild_type: frozenset[str] = frozenset(),
) -> list[Any]:
    """Return the triangle-up marker layers of ``table``'s peaks, one per colour.

    ``color`` is the peaks' own, usually their outcome colour (ADR 0012: mid-grey
    kept, black absorbed); every marker is filled. A stop-codon peak
    (``birth_has_stop_codon``) or a wild-type peak takes its annotation's colour
    instead. The layers run
    plain peaks first, annotated peaks on top.
    """
    kinds = peak_annotation_kinds(table, wild_type=wild_type)
    layers = []
    for kind in (None, "stop_codon", "wild_type"):
        rows = rows_with_annotation(table, kinds, kind)
        if rows.is_empty():
            continue
        if kind == "stop_codon":
            # The stop points themselves are crimson, drawn above the plain peaks
            # with the branch-count opacity of the diagram; no × overlay.
            layers.append(
                biology_persistence_marker_layer(
                    rows,
                    rendering,
                    color=str(rendering.get("stop_codon_color", STOP_CODON_CRIMSON)),
                    annotation=f"{annotation} ({PEAK_ANNOTATION_NOTES[kind]})",
                )
            )
            continue
        kind_color, opacity = peak_annotation_style(kind, rendering, color=color)
        layers.append(
            biology_persistence_marker_layer(
                rows,
                rendering,
                color=kind_color,
                annotation=annotation
                if kind is None
                else f"{annotation} ({PEAK_ANNOTATION_NOTES[kind]})",
                opacity=opacity,
                shape=FIVE_POINT_STAR if kind == "wild_type" else "triangle-up",
            )
        )
    return layers


def horizontal_count_legend(
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    *,
    zero_color: str | None = None,
):
    """Build a horizontal log count bar with its title above, in local pixels.

    The bar runs left (count 1) to right (maximum), black from translucent to
    opaque. An optional zero swatch sits to its left, labelled below. The returned layer's ``width`` and ``height`` are the legend's own
    pixel extent, with its origin at the top-left; ``overlay_colorbar`` moves it
    into a tree panel.
    """
    font_size = int(rendering.get("legend_font_size", 15))
    title = " ".join(
        multiline_title(
            rendering.get("branch_count_title", "Branch\\nvertex\\ncount")
        ).split()
    )
    title_width = math.ceil(0.55 * font_size * len(title))
    bar_height, swatch_width, swatch_gap, bar_gap = 10, 12, 8, 12
    maximum = max(1, int(count_domain[1]))
    length = int(rendering.get("legend_length", 110))
    swatches = [("0", zero_color)] if zero_color is not None else []
    swatch_block = max(
        0, len(swatches) * swatch_width + (len(swatches) - 1) * swatch_gap
    )
    content = swatch_block + (bar_gap if swatches and length else 0) + length
    configured_ticks = rendering.get("count_legend_ticks")
    ticks = (
        [int(value) for value in configured_ticks if int(value) <= maximum]
        if configured_ticks is not None
        else [value for value in (1, 10, 100, 1_000, 10_000) if value <= maximum]
    )
    if maximum == 1:
        ticks = [1]
    # The last tick label is centred on the bar's right end and overhangs it.
    overhang = (
        math.ceil(0.55 * font_size * len(f"{ticks[-1]:,}") / 2)
        if length and ticks
        else 0
    )
    legend_width = max(content + overhang, title_width)
    left = (legend_width - overhang - content) / 2
    bar_top = font_size + 6
    label_top = bar_top + bar_height + 4
    layers = [
        alt.Chart(alt.Data(values=[{}])).mark_text(
            x=legend_width / 2,
            y=0,
            text=title,
            align="center",
            baseline="top",
            fontSize=font_size,
        )
    ]
    for index, (label, color) in enumerate(swatches):
        centre = left + swatch_width / 2 + index * (swatch_width + swatch_gap)
        layers.append(
            alt.Chart(alt.Data(values=[{}])).mark_rect(
                x=centre,
                y=bar_top + bar_height / 2,
                width=swatch_width,
                height=bar_height,
                color=color,
            )
        )
        layers.append(
            alt.Chart(alt.Data(values=[{}])).mark_text(
                x=centre,
                y=label_top,
                text=label,
                align="center",
                baseline="top",
                fontSize=font_size,
            )
        )
    if length:
        bar_left = left + swatch_block + (bar_gap if swatches else 0)
        log_min, log_max = 0.0, math.log(maximum)
        colors = branch_count_colors(rendering)
        layers.append(
            alt.Chart(alt.Data(values=[{}])).mark_rect(
                x=bar_left + length / 2,
                y=bar_top + bar_height / 2,
                width=length,
                height=bar_height,
                color=(
                    colors[0]
                    if maximum == 1
                    else alt.Gradient(
                        gradient="linear",
                        stops=[
                            {"offset": 0, "color": colors[0]},
                            {"offset": 1, "color": colors[1]},
                        ],
                        x1=0,
                        y1=0,
                        x2=1,
                        y2=0,
                    )
                ),
            )
        )
        for value in ticks:
            fraction = (
                (math.log(value) - log_min) / (log_max - log_min)
                if maximum > 1
                else 0.5
            )
            layers.append(
                alt.Chart(alt.Data(values=[{}])).mark_text(
                    x=bar_left + length * fraction,
                    y=label_top,
                    text=f"{value:,}",
                    align="center",
                    baseline="top",
                    fontSize=font_size,
                )
            )
    return alt.layer(*layers).properties(
        width=int(legend_width), height=label_top + font_size
    )


def horizontal_metric_legend(
    rendering: dict[str, Any],
    *,
    title: str,
    values: tuple[int, ...],
    scheme: str | None = None,
    domain: tuple[float, float] | None = None,
    colors: tuple[str, ...] | None = None,
    length: int | None = None,
    lead: int = 24,
    trail_label: str = "10,000",
):
    """Build a horizontal integer-ticked colour bar in the count bar's style.

    ``lead`` and ``trail_label`` reserve the singleton swatch's room to the left
    and the last count label's overhang to the right, so this bar has the same
    extent, and therefore the same position and bar origin, as the count bar.

    Same geometry as ``horizontal_count_legend`` (title above, 10 px bar, labels
    below, local pixels with the origin at the top-left). With ``colors`` the bar
    is one swatch per value and each label is centred under its swatch; with
    ``scheme`` and ``domain`` it is a continuous linear ramp and each label sits
    at its value's position along the ramp.
    """
    if (colors is None) == (scheme is None or domain is None):
        raise ValueError("Give either discrete colors or a scheme with a domain.")
    font_size = int(rendering.get("legend_font_size", 15))
    bar_height = 10
    length = int(length if length is not None else rendering.get("legend_length", 110))
    title_width = math.ceil(0.55 * font_size * len(title))
    overhang = math.ceil(0.55 * font_size * len(trail_label) / 2)
    legend_width = max(lead + length + overhang, title_width)
    left = (legend_width - overhang - lead - length) / 2 + lead
    bar_top = font_size + 6
    label_top = bar_top + bar_height + 4
    layers = [
        alt.Chart(alt.Data(values=[{}])).mark_text(
            x=legend_width / 2,
            y=0,
            text=title,
            align="center",
            baseline="top",
            fontSize=font_size,
        )
    ]
    if colors is not None:
        step = length / len(values)
        for index, (value, color) in enumerate(zip(values, colors, strict=True)):
            layers.append(
                alt.Chart(alt.Data(values=[{}])).mark_rect(
                    x=left + step * (index + 0.5),
                    y=bar_top + bar_height / 2,
                    width=step + 0.5,
                    height=bar_height,
                    color=color,
                )
            )
        positions = [left + step * (index + 0.5) for index in range(len(values))]
    else:
        samples = 60
        for index in range(samples):
            fraction = (index + 0.5) / samples
            layers.append(
                alt.Chart(
                    alt.Data(
                        values=[
                            {"value": domain[0] + (domain[1] - domain[0]) * fraction}
                        ]
                    )
                )
                .mark_rect(
                    x=left + length * fraction,
                    y=bar_top + bar_height / 2,
                    width=length / samples + 0.5,
                    height=bar_height,
                )
                .encode(
                    color=alt.Color(
                        "value:Q",
                        scale=alt.Scale(domain=list(domain), scheme=scheme),
                        legend=None,
                    )
                )
            )
        positions = [
            left + length * (value - domain[0]) / (domain[1] - domain[0])
            for value in values
        ]
    for value, position in zip(values, positions, strict=True):
        layers.append(
            alt.Chart(alt.Data(values=[{}])).mark_text(
                x=position,
                y=label_top,
                text=str(value),
                align="center",
                baseline="top",
                fontSize=font_size,
            )
        )
    return alt.layer(*layers).properties(
        width=int(legend_width), height=label_top + font_size
    )


def shift_pixel_marks(chart: Any, dx: float, dy: float) -> None:
    """Shift, in place, every numeric mark-level x/y pixel position in a chart."""
    mark = getattr(chart, "mark", alt.Undefined)
    if mark is not alt.Undefined:
        spec = (
            {"type": mark}
            if isinstance(mark, str)
            else mark
            if isinstance(mark, dict)
            else mark.to_dict()
        )
        for coordinate, offset in (("x", dx), ("y", dy)):
            if isinstance(spec.get(coordinate), (int, float)):
                spec[coordinate] += offset
        chart.mark = alt.MarkDef(**spec)
    for composition in ("layer", "hconcat", "vconcat", "concat"):
        children = getattr(chart, composition, alt.Undefined)
        if children is not alt.Undefined:
            for child in children:
                shift_pixel_marks(child, dx, dy)


def overlay_colorbar(
    tree_chart: Any,
    legend: Any,
    *,
    x_fraction: float,
    top_inset: float,
    width: int,
) -> Any:
    """Layer a locally positioned colour bar over a tree chart in pixels.

    ``legend`` is a layer of pixel-positioned marks whose ``width`` and
    ``height`` give its own extent, as built by ``horizontal_count_legend``.
    Its horizontal centre is placed at ``x_fraction * width`` (``width`` is the
    tree panel's pixel width), nudged inward only as far as needed to keep the
    whole legend inside the panel, and its top edge ``top_inset`` px below the
    panel top.
    """
    left, top, _, _ = colorbar_box(
        legend, x_fraction=x_fraction, top_inset=top_inset, width=width
    )
    legend = copy.deepcopy(legend)
    shift_pixel_marks(legend, left, top)
    legend.width = alt.Undefined
    legend.height = alt.Undefined
    return alt.layer(tree_chart, legend).resolve_scale(color="independent")


def colorbar_box(
    legend: Any, *, x_fraction: float, top_inset: float, width: int
) -> tuple[float, float, float, float]:
    """Return the ``(left, top, right, bottom)`` pixel box ``overlay_colorbar`` gives
    ``legend`` inside a tree panel ``width`` pixels wide.

    Its horizontal centre is at ``x_fraction * width``, nudged inward only as far as
    needed to keep the whole legend inside the panel; its top edge is ``top_inset``
    below the panel top.
    """
    legend_width = float(legend.width)
    left = min(
        max(0.0, x_fraction * width - legend_width / 2),
        max(0.0, width - legend_width),
    )
    return (
        left,
        float(top_inset),
        left + legend_width,
        float(top_inset) + float(legend.height),
    )


def annotation_legend(
    width: int,
    height: int,
    rendering: dict[str, Any],
    wild_type: bool = True,
    stop_codon: bool = True,
):
    """Key of the tree markers: wild type, then stop codon, on one horizontal line.

    The key sits along the panel's bottom edge (``annotation_legend_y``), starting at
    ``annotation_legend_x``. ``wild_type`` and ``stop_codon`` leave an entry out
    where the panel does not draw that marker. Entries are ``annotation_legend_spacing``
    pixels apart, all on the same row.
    """
    entries = []
    if wild_type:
        entries.append(
            ("★", "Wild type", rendering.get("wild_type_color", WILD_TYPE_RED))
        )
    if stop_codon:
        entries.append(
            (
                "×",
                STOP_CODON_LEGEND_LABEL,
                rendering.get("stop_codon_color", STOP_CODON_CRIMSON),
            )
        )
    font_size = int(
        rendering.get("annotation_legend_font_size", rendering.get("legend_font_size", 15))
    )
    x = int(rendering.get("annotation_legend_x", 12))
    y = int(rendering.get("annotation_legend_y", height - 12))
    spacing = int(rendering.get("annotation_legend_spacing", 135))
    layers = []
    for marker, label, color in entries:
        layers.extend(_legend_entry_layers(marker, label, color, x, y, font_size))
        x += spacing
    return alt.layer(*layers).properties(width=width, height=height)


def _legend_entry_layers(
    marker: str, label: str, color: str, x: float, y: float, font_size: int
) -> list[Any]:
    """One legend entry: a bold coloured glyph at ``x`` and its label beside it."""
    return [
        alt.Chart(alt.Data(values=[{"v": marker}])).mark_text(
            x=x,
            y=y,
            text=marker,
            color=color,
            fontSize=font_size,
            fontWeight="bold",
        ),
        alt.Chart(alt.Data(values=[{"v": label}])).mark_text(
            x=x + 14, y=y, text=label, align="left", fontSize=font_size
        ),
    ]


def persistence_legend_layers(
    width: int,
    height: int,
    rendering: dict[str, Any],
    *,
    peak: bool = True,
    valley: bool = False,
    wild_type: bool = False,
    stop_codon: bool = True,
    synonymous: bool = False,
    peak_color: str = PEAK_GREY,
    stop_label: str = STOP_CODON_LEGEND_LABEL,
    encoded: bool = False,
    around: tuple[float, int] | None = None,
) -> list[Any]:
    """Key drawn inside a persistence diagram, bottom right, one entry per line.

    The layer is ``width`` by ``height`` pixels (the diagram's own plot area), so a
    caller layers it over the diagram. Entries run top to bottom: ▲ peak, ▼ valley
    (filled; ``valley=True`` only on a diagram that draws valleys, the full-graph
    figures), ▲ stop codon (crimson, ``stop_codon=False`` leaves it out for a diagram with no
    stops), ▲ synonymous leaves (green), ★ wild type (only where the panel shows
    it). ``peak=False`` leaves the peak entry out. Entries are
    right-aligned as a block: the block's right edge sits ``persistence_legend_inset_x``
    (default: ``persistence_legend_inset``) pixels from the right edge and its last line ``persistence_legend_inset_y`` (default:
    the same inset) pixels above the bottom edge; lines are ``font_size + persistence_legend_line_gap``
    pixels apart. Same glyph-and-label style as ``annotation_legend``.

    ``encoded`` is the combined stop-and-synonymous encoding's key: ▲ peak and ▼ valley
    in grey, then ``● synonymous`` (the teal filled circle) and ``× <stop_label>`` (the
    red cross), then the wild type.

    ``around=(line_y, above)`` straddles a horizontal rule at pixel height ``line_y``:
    the first ``above`` entries sit just above it and the rest just below it
    (``persistence_legend_rule_gap`` px clear of it), one column, no backing box, all
    right-aligned at the same inset as the bottom-right key.
    """
    entries: list[tuple[str, str, str]] = []
    if peak:
        entries.append(("▲", "peak", peak_color))
    if valley:
        entries.append(("▼", "valley", peak_color))
    if encoded:
        if synonymous:
            entries.append(
                (
                    "●",
                    SYNONYMOUS_LEGEND_LABEL,
                    rendering.get("synonymous_teal", SYNONYMOUS_TEAL),
                )
            )
        if stop_codon:
            entries.append(
                (
                    "×",
                    # The encoded key reads "≥1 stop codon" (it covers peaks and valleys).
                    ENCODED_STOP_LEGEND_LABEL
                    if stop_label == STOP_CODON_LEGEND_LABEL
                    else stop_label,
                    rendering.get("stop_codon_color", STOP_CODON_CRIMSON),
                )
            )
    else:
        if stop_codon:
            entries.append(
                (
                    "▲",
                    stop_label,
                    rendering.get("stop_codon_color", STOP_CODON_CRIMSON),
                )
            )
        if synonymous:
            entries.append(
                (
                    "▲",
                    "synonymous leaves",
                    rendering.get("synonymous_color", SYNONYMOUS_GREEN),
                )
            )
    if wild_type:
        entries.append(
            ("★", "wild type", rendering.get("wild_type_color", WILD_TYPE_RED))
        )
    # Smaller than the tree key: "Stop codon containing peaks" must fit a 250 px diagram.
    font_size = int(
        rendering.get(
            "persistence_legend_font_size", rendering.get("legend_font_size", 15)
        )
    )
    inset = int(rendering.get("persistence_legend_inset", 10))
    inset_x = int(rendering.get("persistence_legend_inset_x", inset))
    inset_y = int(rendering.get("persistence_legend_inset_y", inset))
    line = font_size + int(rendering.get("persistence_legend_line_gap", 4))
    glyph = 14
    char_width = 0.55 * font_size
    # The block is as wide as its longest label and right-aligned.
    x0 = width - inset_x - glyph - max(
        (math.ceil(char_width * len(label)) for _, label, _ in entries), default=0
    )
    layers: list[Any] = []
    for index, (marker, label, color) in enumerate(entries):
        if around is not None:
            line_y, above = around
            gap = float(rendering.get("persistence_legend_rule_gap", 4))
            if index < above:
                y = line_y - gap - (above - 1 - index) * line
            else:
                y = line_y + gap + font_size + (index - above) * line
        else:
            y = height - inset_y - (len(entries) - 1 - index) * line
        layers.extend(_legend_entry_layers(marker, label, color, x0, y, font_size))
    return layers


def persistence_legend(
    width: int, height: int, rendering: dict[str, Any], **entries: Any
):
    """``persistence_legend_layers`` as one layer chart of the diagram's size."""
    layers = persistence_legend_layers(width, height, rendering, **entries)
    return alt.layer(
        *(layers or [alt.Chart(alt.Data(values=[])).mark_point()])
    ).properties(width=width, height=height)


def persistence_panel_size(
    figure_settings: dict[str, Any], rendering: dict[str, Any]
) -> tuple[int, int]:
    """Return the pixel ``(width, height)`` ``persistence_chart`` gives its diagram."""
    appearance = figure_settings.get("persistence", {})
    return (
        int(
            appearance.get(
                "width", rendering.get("persistence_width", PERSISTENCE_WIDTH)
            )
        ),
        int(
            appearance.get(
                "height", rendering.get("persistence_height", PERSISTENCE_HEIGHT)
            )
        ),
    )


def with_persistence_legend(
    diagram: Any,
    size: tuple[int, int],
    rendering: dict[str, Any],
    **entries: Any,
) -> Any:
    """Layer the in-diagram ``persistence_legend`` over ``diagram``.

    ``size`` is the diagram's ``(width, height)`` in pixels; ``entries`` are the
    keywords of ``persistence_legend``.
    """
    return alt.layer(
        diagram, persistence_legend(size[0], size[1], rendering, **entries)
    ).properties(width=size[0], height=size[1])


def tree_histogram_settings(
    settings: dict[str, Any], figure_id: str
) -> tuple[dict[str, Any], dict[str, Any], tuple[float, float]]:
    figures = settings["figures"]
    figure_settings = figures["figures"][figure_id]
    rendering = {**figures.get("rendering", {}), **figure_settings}
    fitness_domain = tuple(float(x) for x in figures.get("fitness_domain", (-1.2, 1.4)))
    if (
        len(fitness_domain) != 2
        or not all(math.isfinite(x) for x in fitness_domain)
        or fitness_domain[0] >= fitness_domain[1]
    ):
        raise ValueError(
            "figures.fitness_domain must contain finite increasing bounds."
        )
    return figure_settings, rendering, fitness_domain


def tree_population(
    tree_analysis: Any, analysis: dict[str, Any], population_id: str
) -> pl.DataFrame:
    if getattr(analysis, "legacy_mixture", False):
        from ._legacy_mixture import tree_population as legacy_tree_population

        population = legacy_tree_population(tree_analysis, population_id)
    else:
        try:
            population = analysis["saved_populations"][population_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing saved {population_id} landscape population; run `tree-layout-products` first."
            ) from exc
    diagnostic_key = {"article": "study_clamped", "candidate": "candidate_clamped"}.get(
        population_id, "full"
    )
    diagnostics = analysis.get("diagnostics", {}).get(diagnostic_key, {})
    expected = diagnostics.get("selected_component_vertex_count")
    if expected is None:
        expected = diagnostics.get(
            "selected_vertex_count", diagnostics.get("vertex_count")
        )
    if expected is not None and population.height != int(expected):
        raise ValueError(
            f"{population_id.title()} tree tables do not represent every selected graph vertex: "
            f"{population.height} unique identities for {int(expected)} vertices."
        )
    if not population.height or population["dna_sequence"].null_count():
        raise ValueError(
            f"{population_id.title()} tree population must have stable identities."
        )
    return population


def histogram_chart(
    population: pl.DataFrame,
    bins: pl.DataFrame | None,
    fitness_domain: tuple[float, float],
    count_domain: tuple[float, float] | None,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    *,
    count_axis: Any | None,
    count_limit: float | None = None,
) -> Any:
    """Tree-side fitness histogram; the count axis runs 0 to ``count_limit``.

    A figure with several histogram panels passes one ``count_limit`` (see
    ``count_axis_limit``) so they share the axis; alone, a panel uses headroom times its
    own tallest bin. A caller-supplied ``count_domain`` is ignored."""
    from topographer.plotting import (
        compute_fitness_histogram_table,
        plot_fitness_histogram,
    )

    from ._top_histogram import count_axis_limit, top_histogram_settings

    raw = population.rename({"scalar": "raw_fitness"})
    if bins is None:
        bins = compute_fitness_histogram_table(
            raw,
            scalar_column="raw_fitness",
            bin_count=int(figure_settings.get("histogram_bins", 200)),
        )
    _, ticks = top_histogram_settings(rendering)
    if count_limit is None:
        count_limit = count_axis_limit([int(bins["count"].max() or 1)], rendering)
    count_axis = count_axis or fixed_count_axis(ticks, title="Vertex count")
    chart = plot_fitness_histogram(
        raw,
        scalar_column="raw_fitness",
        bin_count=int(figure_settings.get("histogram_bins", 200)),
        scalar_domain=fitness_domain,
        count_domain=(0.0, float(count_limit)),
        scalar_axis=alt.Axis(
            title=None, labels=False, ticks=False, domain=False, grid=False
        ),
        count_axis=count_axis,
        width=int(figure_settings.get("histogram_width", 140)),
        height=tree_panel_size(rendering, figure_settings)[1],
        color=figure_settings.get("histogram_color", LIGHT_GREY),
        include_config=False,
    )
    chart.mark = alt.MarkDef(**{**chart.mark.to_dict(), "clip": True})
    return chart


def fixed_count_axis(ticks: Iterable[int], *, title: str) -> alt.Axis:
    """Count axis with ticks at the given values only (config ``top_histogram_count_ticks``)."""
    return alt.Axis(
        title=title,
        values=list(ticks),
        grid=False,
        labelExpr="datum.value === 0 ? '0' : upper(format(datum.value, '~s'))",
    )


def histogram_count_axis(domain: tuple[float, float], *, title: str) -> alt.Axis:
    maximum = float(domain[1])
    interval = max(1.0, nice_histogram_tick_step(maximum / 2))
    values = [0.0, interval, 2 * interval]
    values = [value for value in values if value <= maximum]
    return alt.Axis(
        title=title,
        values=values,
        format="~s",
        grid=False,
        labelExpr="datum.value === 0 ? '0' : format(datum.value, '~s')",
    )


def nice_histogram_tick_step(value):
    exponent = math.floor(math.log10(value)) if value > 0 else 0
    magnitude = 10**exponent
    fraction = value / magnitude
    nice = (
        1 if fraction <= 1.5 else 2 if fraction <= 3.5 else 5 if fraction <= 7.5 else 10
    )
    return nice * magnitude


def apply_tree_histogram_domain(chart, domain):
    """Apply exact display bounds and crop all tree marks at the view frame."""
    fields = {"y", "scalar", "bin_min", "birth_scalar", "death_scalar"}

    def visit(node):
        encoding = getattr(node, "encoding", alt.Undefined)
        if encoding is not alt.Undefined:
            for channel in ("x", "y"):
                channel_spec = getattr(encoding, channel, alt.Undefined)
                if channel_spec is alt.Undefined:
                    continue
                if isinstance(channel_spec, str):
                    field = channel_spec.split(":", 1)[0]
                    if field not in fields:
                        continue
                    channel_spec = {"field": field, "type": "quantitative"}
                else:
                    channel_spec = (
                        dict(channel_spec)
                        if isinstance(channel_spec, dict)
                        else channel_spec.to_dict()
                    )
                if channel_spec.get("field") not in fields:
                    continue
                scale = dict(channel_spec.get("scale") or {})
                scale.update(domain=list(domain), nice=False, zero=False)
                channel_spec["scale"] = scale
                node = node.encode(**{channel: channel_spec})
        mark = getattr(node, "mark", alt.Undefined)
        if mark is not alt.Undefined:
            mark_spec = (
                {"type": mark}
                if isinstance(mark, str)
                else mark
                if isinstance(mark, dict)
                else mark.to_dict()
            )
            if mark_spec.get("type") in {"line", "point", "rule", "text", "trail"}:
                node.mark = alt.MarkDef(**{**mark_spec, "clip": True})
        for composition in ("layer", "hconcat", "vconcat", "concat"):
            children = getattr(node, composition, alt.Undefined)
            if children is not alt.Undefined:
                node[composition] = [visit(child) for child in children]
        return node

    return visit(chart)


def tree_histogram_chart_config(chart: Any, rendering: dict[str, Any]) -> Any:
    return (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(rendering.get("axis_label_font_size", 15)),
            titleFontSize=int(rendering.get("font_size", 15)),
        )
        .configure_legend(
            labelFontSize=int(rendering.get("legend_font_size", 15)),
            titleFontSize=int(rendering.get("legend_font_size", 15)),
        )
        .configure_title(
            fontSize=int(
                rendering.get("heading_font_size", rendering.get("font_size", 15))
            ),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


def study_biology_tree_chart(
    plot_data: Any,
    analysis: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    count_domain: tuple[float, float] | None = None,
) -> Any:
    """Draw the study tree as quiet gray structure with its public layout.

    With ``count_domain`` the branches are instead coloured by their
    ``vertex_count`` on the shared log grey count scale over that domain
    (``plot_data`` then carries the count, as ``plot_data_with_metrics`` leaves it),
    higher counts drawn last, on top. The count colour bar is the caller's.
    """
    appearance = figure_settings.get("tree", {})
    colouring: dict[str, Any] = (
        {"edge_color": rendering.get("biology_tree_color", TREE_GREY)}
        if count_domain is None
        else {"edge_color_scale": color_scale(rendering, count_domain)}
    )
    config = TreePlotConfig(
        width=int(appearance.get("width", rendering.get("tree_width", TREE_WIDTH))),
        height=int(appearance.get("height", rendering.get("tree_height", TREE_HEIGHT))),
        font_size=int(appearance.get("font_size", rendering.get("font_size", 15))),
        scalar_axis_title=rendering.get(
            "tree_axis_title", "Fitness (at birth, WT = 0)"
        ),
        scalar_domain=analysis.scalar_domain,
        edge_width=float(rendering.get("edge_width", 1.1)),
        node_size=int(rendering.get("node_size", 24)),
        **colouring,
    )
    edge_tooltip_attrs: tuple[str, ...] = ()
    if count_domain is not None:
        # Ascending by branch count so higher counts draw last, on top.
        plot_data = replace(
            plot_data,
            edge_table=plot_data.edge_table.sort("vertex_count", maintain_order=True),
        )
        edge_tooltip_attrs = ("vertex_count", "source_label", "target_label")
    return plot_tree(
        plot_data,
        config=config,
        presentation=TreePresentationConfig(mode="metric"),
        tooltip_attrs=("dna_sequence", "aa_sequence"),
        edge_tooltip_attrs=edge_tooltip_attrs,
        show_structural_events=False,
        interactive=False,
        include_config=False,
    )


def study_biology_persistence_chart(
    table: pl.DataFrame,
    analysis: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
) -> Any:
    """Draw unencoded mid-grey filled triangles for every persistence pair."""
    if rendering.get("biology_count_domain") is not None:
        chart = persistence_chart(
            table, figure_settings, rendering, rendering["biology_count_domain"],
            analysis.scalar_domain,
        )
        if float(rendering.get("biology_persistence_opacity", 1.0)) == 0:
            chart = set_count_point_opacity(chart, 0.0, shape="triangle-up")
        return chart
    appearance = figure_settings.get("persistence", {})
    config = PersistencePlotConfig(
        width=int(
            appearance.get(
                "width", rendering.get("persistence_width", PERSISTENCE_WIDTH)
            )
        ),
        height=int(
            appearance.get(
                "height", rendering.get("persistence_height", PERSISTENCE_HEIGHT)
            )
        ),
        font_size=int(appearance.get("font_size", rendering.get("font_size", 15))),
        birth_axis_title=rendering.get(
            "persistence_y_title", "Fitness (at birth, WT = 0)"
        ),
        death_axis_title=rendering.get("persistence_x_title", "Fitness at death"),
        domain=analysis.scalar_domain,
        point_size=int(
            appearance.get("point_size", rendering.get("persistence_point_size", 70))
        ),
        show_grid=bool(rendering.get("persistence_grid", False)),
        color_scale=None,
    )
    chart = plot_persistence(
        table,
        config=config,
        color_by=None,
        shape_by=None,
        separate_event_layers=False,
        tooltip_attrs=("persistence", "birth_label", "death_label"),
        include_config=False,
    )
    for layer in getattr(chart, "layer", ()):
        mark = getattr(layer, "mark", None)
        if mark is None or mark is alt.Undefined:
            continue
        mark_spec = mark if isinstance(mark, dict) else mark.to_dict()
        if mark_spec.get("type") == "point":
            layer.mark = alt.MarkDef(
                **{
                    **mark_spec,
                    "shape": "triangle-up",
                    "filled": True,
                    "color": rendering.get("biology_persistence_color", PEAK_GREY),
                    "opacity": float(rendering.get("biology_persistence_opacity", 1.0)),
                }
            )
    return chart


def band_trapezoid_frame(
    low: float, high: float, width: float, *, above: bool = True
) -> pl.DataFrame:
    """Return the trapezoid between the diagonal and a parallel line ``width`` away.

    The diagonal runs from ``low`` to ``high`` on both axes. The trapezoid is
    clipped to that extent: ``edge`` is the parallel line (``diagonal + width``
    when ``above``, else ``diagonal - width``) cut at the extent, so the shape has
    four corners and the three rows are its breakpoints. Columns: ``along`` (the
    abscissa), ``edge`` and ``diagonal`` (the two ordinates).
    """
    width = max(float(width), 0.0)
    if above:
        along = [low, min(max(low, high - width), high), high]
        edge = [min(value + width, high) for value in along]
    else:
        along = [low, max(min(high, low + width), low), high]
        edge = [max(value - width, low) for value in along]
    return pl.DataFrame(
        {"along": along, "edge": edge, "diagonal": along},
        schema={"along": pl.Float64, "edge": pl.Float64, "diagonal": pl.Float64},
    )


def band_trapezoid_layer(
    frame: pl.DataFrame,
    x: Any,
    y: Any,
    rendering: dict[str, Any],
):
    """Draw a translucent grey band trapezoid with no edge line.

    ``x`` is the encoding of ``along``; ``y`` that of ``edge``; the area closes on
    ``diagonal``. Colour and opacity come from ``band_color`` and ``band_opacity``.
    """
    return (
        alt.Chart(frame)
        .mark_area(clip=True, **_band_fill(rendering))
        .encode(x=x, y=y, y2="diagonal:Q")
    )


def band_square_layer(
    low: float, high: float, x: Any, y: Any, rendering: dict[str, Any]
):
    """Draw a translucent grey square with corners ``(low, low)`` and ``(high, high)``.

    ``x`` and ``y`` are the encodings of the ``low`` column on the two axes; the
    square closes on ``high`` for both. It has the same fill as the trapezoid
    (``band_color``, ``band_opacity``) and no edge line.
    """
    return (
        alt.Chart(pl.DataFrame({"low": [float(low)], "high": [float(high)]}))
        .mark_rect(clip=True, **_band_fill(rendering))
        .encode(x=x, x2="high:Q", y=y, y2="high:Q")
    )


def _band_fill(rendering: dict[str, Any]) -> dict[str, Any]:
    """Return the translucent grey fill shared by every band shape."""
    return {
        "color": str(rendering.get("band_color", BAND_GREY)),
        "opacity": float(rendering.get("band_opacity", 0.3)),
    }


def biology_persistence_marker_layer(
    table: pl.DataFrame,
    rendering: dict[str, Any],
    *,
    color: str,
    annotation: str,
    opacity: float = 1.0,
    size: int | None = None,
    shape: str = "triangle-up",
):
    """Put a filled colored peak marker (triangle-up) above the persistence points.

    The marker is always filled (no hollow markers). ``opacity`` is 1 for every annotation; ``size`` (an area in
    px squared) defaults to ``biology_persistence_point_size``. Count-driven opacity
    never falls below ``biology_marker_opacity_floor`` (default 0: no floor).
    """
    if table.is_empty():
        return alt.Chart(alt.Data(values=[])).mark_point()
    fields = ["death_scalar", "birth_scalar", "birth_label", "death_label"]
    count_domain = rendering.get("biology_count_domain") if shape != FIVE_POINT_STAR else None
    data = table.select(*fields, *(["vertex_count"] if count_domain is not None else []))
    if count_domain is not None:
        data = data.with_columns(
            pl.col("vertex_count").map_elements(
                lambda count: max(
                    branch_count_opacity(count, rendering, count_domain),
                    float(rendering.get("biology_marker_opacity_floor", 0.0)),
                ),
                return_dtype=pl.Float64,
            ).alias("count_opacity")
        )
    data = data.with_columns(pl.lit(annotation).alias("annotation"))
    chart = (
        alt.Chart(data)
        .mark_point(
            shape=shape,
            filled=True,
            color=color,
            size=int(
                size
                if size is not None
                else rendering.get("biology_persistence_point_size", 70)
            ),
            strokeWidth=float(rendering.get("open_marker_stroke_width", 1.6)),
            **({} if count_domain is not None else {"opacity": opacity}),
        )
        .encode(
            x="death_scalar:Q",
            y="birth_scalar:Q",
            tooltip=["annotation:N", "birth_label:N", "death_label:N"],
        )
    )
    if count_domain is not None:
        chart = chart.encode(opacity=alt.Opacity("count_opacity:Q", scale=None, legend=None))
    return chart
