"""Render the band-smoothed codon-entropy figure from prepared scientific products."""

from __future__ import annotations

import logging
import math
from itertools import pairwise
from typing import Any

import altair as alt
import matplotlib
import polars as pl
from topographer.biology import AMINO_ACIDS_23

from dubanevics2026topological.figures._branch_count_style import (
    branch_count_color_encoding,
)
from dubanevics2026topological.figures._charts import (
    annotation_legend,
    compose_tree_layers,
    count_colorbar_box,
    count_domain,
    overlay_count_colorbar,
    plot_data_with_metrics,
    selected_view_as_tree,
    stop_codon_nodes,
    stop_tree_points_layer,
    wild_type_layers,
)
from dubanevics2026topological.figures._count_labels import (
    branch_count_labels,
    tree_geometry,
)
from dubanevics2026topological.figures.palette import (
    EMPHASIS_BLACK,
    INK_GREY,
    KEM_BRANCH_COLOR,
    OUTSIDE_GREY,
    TCL_BRANCH_COLOR,
)

MAX_ENTROPY = math.log2(len(AMINO_ACIDS_23))

LOGGER = logging.getLogger(__name__)


FIGURE_ID = "bootstrap_codon_entropy"
# Opacity of every branch but the focused ridge in a ridge excerpt; the figure's
# ``dimmed_branch_opacity`` key overrides it.
DEFAULT_DIMMED_BRANCH_OPACITY = 0.25


def _figure_settings(
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return codon-entropy and shared rendering settings from parsed TOML."""
    config = settings.get("figures", settings)
    rendering = config.get("rendering", settings.get("rendering", {}))
    figures = config.get("figures", {})
    if isinstance(figures.get("figures"), dict):
        figures = figures["figures"]
    figure = figures.get(FIGURE_ID, {})
    # A narrow settings mapping is useful to callers and unit tests.
    if not figure and ("tree_width" in settings or "width" in settings):
        figure = settings
    return figure, rendering


def _require_keys(comparison: Any, keys: tuple[str, ...], what: str) -> None:
    if not isinstance(comparison, dict):
        raise TypeError("The codon entropy payload must be a mapping.")
    missing = [key for key in keys if key not in comparison]
    if missing:
        raise KeyError(
            f"Codon entropy payload lacks {what} {missing}; compose it with "
            "figures.compose_payload from build.load_inputs. Renderers never "
            "read output/ or recompute analysis."
        )


def _population_frames(
    comparison: Any,
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    """Return the prepared population frames from the payload only."""
    _require_keys(comparison, ("df_core", "df_kem", "df_tcl", "metadata"), "keys")
    populations = {
        "core": comparison["df_core"],
        "kem": comparison["df_kem"],
        "tcl": comparison["df_tcl"],
    }
    return comparison["metadata"], populations


def _entropy_frames(comparison: Any) -> dict[str, pl.DataFrame]:
    """Return the prepared entropy sweeps from the payload only."""
    _require_keys(comparison, ("entropy",), "keys")
    entropy = comparison["entropy"]
    missing = [name for name in ("core", "kem", "tcl") if name not in entropy]
    if missing:
        raise KeyError(f"Codon entropy payload lacks entropy sweeps {missing}.")
    return entropy


def _candidate_leaf_view(comparison: Any) -> Any:
    """Return the prepared Candidate simplification from the payload only.

    When precomputed render tables from the bootstrap_codon_entropy product
    are present, the live view is not required (presentation purity).
    """
    render_tables = comparison.get("codon_render_tables")
    if render_tables and "candidate_leaf_view" not in comparison:
        return None
    _require_keys(comparison, ("candidate_leaf_view",), "keys")
    view = comparison["candidate_leaf_view"]
    if view is None:
        raise ValueError(
            f"{FIGURE_ID} requires its selected-leaf view as candidate_leaf_view."
        )
    return view


def _rounded_fitness_domain(
    values: list[float], tick_step: float
) -> tuple[tuple[float, float], tuple[float, float, float]]:
    if tick_step <= 0:
        raise ValueError("fitness tick step must be positive")
    lower = math.floor(min(values) / tick_step) * tick_step
    upper = math.ceil(max(values) / tick_step) * tick_step
    if lower == upper:
        lower -= tick_step
        upper += tick_step
    ticks = tuple(sorted({lower, 0.0, upper}))
    return (lower, upper), ticks


def _full_tree_chart(
    plot_data: Any,
    focused_pairs: set[tuple[int, int]],
    *,
    width: int,
    height: int,
    domain: tuple[float, float],
    ticks: tuple[float, ...],
    padding_fraction: float,
    edge_width: float,
    x_domain: tuple[float, float],
    show_fitness_axis: bool,
    fitness_axis_title: str,
    fitness_tick_format: str,
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    dimmed_opacity: float,
) -> alt.LayerChart:
    """Draw the full tree, one ridge emphasised and the rest muted.

    Every row uses this chart at the same width, height and x domain (the whole
    layout), so the three tree panels are the same size. ``focused_pairs`` (the
    plot tables' ``(source, target)`` tree-node pairs) is drawn at full opacity, on
    top; every other branch at ``dimmed_opacity``. The Candidate row passes every
    segment pair, so nothing is muted. Colours stay the shared count scale
    (``branch_count_color_encoding`` over ``count_domain``). Higher counts are drawn
    last within each layer.

    The x and y scales are explicit and not niced, the same contract on every row:
    overlays that carry no scale of their own must not round the domain
    (``_pin_explicit_scales``).
    """
    all_segments = plot_data.edge_segment_table.to_dicts()

    def on_ridge(row: dict[str, Any]) -> bool:
        return (
            int(row["source_tree_node_index"]),
            int(row["target_tree_node_index"]),
        ) in focused_pairs

    focused = [row for row in all_segments if on_ridge(row)]
    if not focused:
        raise ValueError("The selected ridge has no drawable simplified-tree edges.")
    dimmed = [row for row in all_segments if not on_ridge(row)]
    colour = branch_count_color_encoding(rendering, count_domain)

    def layer(rows: list[dict[str, Any]], opacity: float) -> alt.Chart:
        rows = sorted(rows, key=lambda row: int(row["vertex_count"]))
        return (
            alt.Chart(alt.Data(values=rows))
            .mark_rule(strokeWidth=edge_width, opacity=opacity)
            .encode(
                x=alt.X(
                    "x_start:Q",
                    axis=None,
                    scale=alt.Scale(domain=list(x_domain), zero=False, nice=False),
                ),
                x2="x_end:Q",
                y=alt.Y(
                    "y_start:Q",
                    title=fitness_axis_title if show_fitness_axis else None,
                    scale=alt.Scale(
                        domain=list(domain),
                        zero=False,
                        padding=padding_fraction,
                        nice=False,
                    ),
                    axis=alt.Axis(
                        values=list(ticks),
                        format=fitness_tick_format,
                        domain=False,
                        ticks=True,
                    )
                    if show_fitness_axis
                    else None,
                ),
                y2="y_end:Q",
                color=colour,
                tooltip=["vertex_count:Q"],
            )
            .properties(width=width, height=height)
        )

    layers = [layer(focused, 1.0)]
    if dimmed:
        layers.insert(0, layer(dimmed, dimmed_opacity))
    return alt.layer(*layers).properties(width=width, height=height)


def _pin_explicit_scales(
    node: Any,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
) -> Any:
    """Give every x and y encoding the row's explicit, un-niced domains.

    Wild-type and stop layers carry no scale of their own. Left unset, Vega-Lite's
    implicit ``nice`` would merge into the tree scale and the three rows would no
    longer share one domain.
    """
    encoding = getattr(node, "encoding", alt.Undefined)
    if encoding is not alt.Undefined and hasattr(node, "encode"):
        spec = encoding.to_dict()
        updates = {}
        for channel, domain in (("x", x_domain), ("y", y_domain)):
            definition = spec.get(channel)
            if isinstance(definition, dict) and definition.get("field"):
                scale = {
                    **(definition.get("scale") or {}),
                    "domain": list(domain),
                    "zero": False,
                    "nice": False,
                }
                updates[channel] = {**definition, "scale": scale}
        if updates:
            node = node.encode(**updates)
    for composition in ("layer", "hconcat", "vconcat", "concat"):
        children = getattr(node, composition, alt.Undefined)
        if children is not alt.Undefined:
            node[composition] = [
                _pin_explicit_scales(child, x_domain, y_domain) for child in children
            ]
    return node


def _scaled_stop_layer(
    points: pl.DataFrame,
    rendering: dict[str, Any],
    *,
    x_domain: tuple[float, float],
    domain: tuple[float, float],
) -> alt.Chart:
    """Draw stop crosses on the full tree's explicit, un-niced scales."""
    return stop_tree_points_layer(points, rendering).encode(
        x=alt.X(
            "x:Q",
            axis=None,
            scale=alt.Scale(domain=list(x_domain), zero=False, nice=False),
        ),
        y=alt.Y(
            "y:Q",
            scale=alt.Scale(domain=list(domain), zero=False, nice=False),
        ),
    )


BRANCH_NAMES = ("kem", "tcl")


def _branch_colors(figure: dict[str, Any]) -> dict[str, str]:
    """The KEM and TCL branch colours: used by no other element of the palette."""
    return {
        "kem": str(figure.get("kem_color", KEM_BRANCH_COLOR)),
        "tcl": str(figure.get("tcl_color", TCL_BRANCH_COLOR)),
    }


def _peak_identity(metadata: dict[str, Any], name: str) -> str:
    record = metadata.get("summits", {}).get(name)
    if record is None:
        record = metadata[f"{name}_peak"]
    return str(record["dna_sequence"])


def _node_by_dna(plot_data: Any, dna: str) -> dict[str, Any]:
    """The tree node whose sequence is ``dna`` (identities, never indices, name it)."""
    rows = plot_data.node_table.filter(pl.col("dna_sequence") == dna)
    if rows.is_empty():
        raise ValueError(f"Sequence {dna} is not a node of the simplified tree.")
    return rows.row(0, named=True)


def _amino_acids(node: dict[str, Any]) -> str:
    """The node's amino-acid genotype: its saved ``aa_sequence``, else the translation."""
    saved = node.get("aa_sequence")
    if saved:
        return str(saved)
    from dubanevics2026topological.analysis.ridge_entropy import translate

    return translate(str(node["dna_sequence"]))


def _branch_segments(
    segments: list[dict[str, Any]], peak_index: int, saddle_index: int, name: str
) -> list[dict[str, Any]]:
    """Every segment on the way down from a peak to the common saddle.

    The tree table runs each edge from a lower ``source`` node to a higher
    ``target`` node. Starting at the peak, take the edge into the current node and
    step to its source until the saddle is reached, so a peak whose own edge ends at
    an intermediate saddle (the KEM peak) is still drawn all the way down to the
    saddle it shares with the other peak. The saddle itself is the last node.
    """
    by_target: dict[int, list[dict[str, Any]]] = {}
    for row in segments:
        by_target.setdefault(int(row["target_tree_node_index"]), []).append(row)
    path: list[dict[str, Any]] = []
    current = peak_index
    seen = {current}
    while current != saddle_index:
        own = by_target.get(current)
        if not own:
            raise ValueError(
                f"The {name.upper()} branch does not reach the common saddle."
            )
        path.extend(own)
        current = int(own[0]["source_tree_node_index"])
        if current in seen:
            raise ValueError(f"The {name.upper()} branch loops before the saddle.")
        seen.add(current)
    return path


def _branch_geometry(
    plot_data: Any, metadata: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The KEM and TCL peaks (with their branch's segments) and the saddle.

    Each branch runs from the peak down to the common saddle the two peaks share
    (the TCL peak's own edge ends there already; the KEM peak's passes through an
    intermediate saddle first). Positions come from the plot tables; the genotypes
    are amino-acid sequences.
    """
    segments = plot_data.edge_segment_table.to_dicts()
    saddle_node = _node_by_dna(plot_data, str(metadata["common_saddle"]["dna_sequence"]))
    peaks: dict[str, dict[str, Any]] = {}
    for name in BRANCH_NAMES:
        node = _node_by_dna(plot_data, _peak_identity(metadata, name))
        own = _branch_segments(
            segments,
            int(node["tree_node_index"]),
            int(saddle_node["tree_node_index"]),
            name,
        )
        peaks[name] = {
            "dna_sequence": str(node["dna_sequence"]),
            "aa_sequence": _amino_acids(node),
            "x": float(node["x"]),
            "y": float(node["y"]),
            "segments": own,
        }
    saddle = {
        "dna_sequence": str(saddle_node["dna_sequence"]),
        "aa_sequence": _amino_acids(saddle_node),
        "x": float(saddle_node["x"]),
        "y": float(saddle_node["y"]),
    }
    return peaks, saddle


def _scaled_xy(
    x_domain: tuple[float, float], domain: tuple[float, float]
) -> dict[str, Any]:
    """Explicit, un-niced x and y encodings on the tree's own scales."""
    return {
        "x": alt.X(
            "x:Q",
            axis=None,
            scale=alt.Scale(domain=list(x_domain), zero=False, nice=False),
        ),
        "y": alt.Y(
            "y:Q", scale=alt.Scale(domain=list(domain), zero=False, nice=False)
        ),
    }


def _highlight_layer(
    segments: list[dict[str, Any]],
    color: str,
    *,
    edge_width: float,
    x_domain: tuple[float, float],
    domain: tuple[float, float],
) -> alt.Chart:
    """One branch in its own solid colour, over the count-coloured tree."""
    rows = [
        row
        for row in segments
        if row["x_start"] != row["x_end"] or row["y_start"] != row["y_end"]
    ]
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_rule(color=color, strokeWidth=edge_width, opacity=1.0)
        .encode(
            x=alt.X(
                "x_start:Q",
                axis=None,
                scale=alt.Scale(domain=list(x_domain), zero=False, nice=False),
            ),
            x2="x_end:Q",
            y=alt.Y(
                "y_start:Q",
                scale=alt.Scale(domain=list(domain), zero=False, nice=False),
            ),
            y2="y_end:Q",
            detail="logical_edge_index:N",
            tooltip=["source_label:N", "target_label:N"],
        )
    )


def _marker_layers(
    peaks: dict[str, dict[str, Any]],
    saddle: dict[str, Any],
    colors: dict[str, str],
    figure: dict[str, Any],
    rendering: dict[str, Any],
    *,
    x_domain: tuple[float, float],
    domain: tuple[float, float],
) -> list[alt.Chart]:
    """Peak genotype labels and the filled saddle diamond with its genotype.

    A peak is named by its genotype above the tip, in the branch's colour, with no
    marker; the saddle's genotype stands to the left of and a little below its
    diamond. The diamond is filled and nothing is a valley.
    """
    encodings = _scaled_xy(x_domain, domain)
    font = int(figure.get("label_font_size", rendering.get("font_size", 15)))
    saddle_size = int(figure.get("saddle_marker_size", 90))
    layers: list[alt.Chart] = []
    for name in BRANCH_NAMES:
        peak = peaks[name]
        frame = pl.DataFrame(
            {
                "x": [peak["x"]],
                "y": [peak["y"]],
                "label": [peak["aa_sequence"]],
                "dna_sequence": [peak["dna_sequence"]],
            }
        )
        base = alt.Chart(frame).encode(**encodings)
        layers.append(
            base.mark_text(
                align="center",
                baseline="bottom",
                dy=-int(figure.get("peak_label_offset", 9)),
                fontSize=font,
                fontWeight="bold",
                color=colors[name],
            ).encode(text="label:N")
        )
    frame = pl.DataFrame(
        {
            "x": [saddle["x"]],
            "y": [saddle["y"]],
            "label": [saddle["aa_sequence"]],
            "dna_sequence": [saddle["dna_sequence"]],
        }
    )
    base = alt.Chart(frame).encode(**encodings)
    layers.append(
        base.mark_point(
            shape="diamond",
            filled=True,
            size=saddle_size,
            color=EMPHASIS_BLACK,
            opacity=1.0,
        ).encode(tooltip=["dna_sequence:N", "label:N"])
    )
    layers.append(
        base.mark_text(
            align="right",
            baseline="middle",
            dx=-int(figure.get("saddle_label_offset", 10)),
            dy=int(figure.get("saddle_label_dy", 8)),
            fontSize=font,
            color=INK_GREY,
        ).encode(text="label:N")
    )
    return layers


def _tree_chart(
    candidate_view: Any,
    metadata: dict[str, Any],
    figure: dict[str, Any],
    rendering: dict[str, Any],
    domain: tuple[float, float],
    ticks: tuple[float, ...],
    padding_fraction: float,
    fitness_axis_title: str,
    fitness_tick_format: str,
    *,
    render_tables: dict[str, Any] | None = None,
) -> alt.Chart:
    """Prepare and draw the one tree panel: the simplified Candidate tree.

    The KEM and TCL peak branches are drawn over it in their own colours, each peak
    with its amino-acid genotype above (no marker), and the saddle with a diamond and
    its genotype to the left and a little below. When ``render_tables`` is supplied (from the
    bootstrap_codon_entropy product), the graph preparation is skipped and the
    pre-built tables are used directly (the pure-presentation path).
    """
    if render_tables is not None:
        from types import SimpleNamespace

        from topographer.plotting import TreePlotData

        node = render_tables.get("node_table")
        edge = render_tables.get("edge_table")
        seg = render_tables.get("edge_segment_table")
        attrs = render_tables.get("plot_data_attrs")
        if node is None or edge is None or seg is None or attrs is None:
            raise ValueError("codon render tables are incomplete.")
        plot_data = TreePlotData(
            node_table=node,
            edge_table=edge,
            edge_segment_table=seg,
            root_node_index=int(attrs["root_node_index"]),
            scalar_attr=str(attrs["scalar_attr"]),
            root_kind=str(attrs["root_kind"]),
        )
        wild_type_marker = render_tables.get("wild_type_marker")
        if wild_type_marker is None:
            wild_type_marker = pl.DataFrame()
        tree_analysis = SimpleNamespace(
            plot_data=plot_data,
            branch_table=edge,  # sufficient for count labels
            wild_type_marker_table=wild_type_marker,
        )
    else:
        tree_analysis = candidate_view
        # PR 3 transition: callers that still pass a live view force the legacy
        # path until all figures are migrated.
        plot_data = plot_data_with_metrics(
            selected_view_as_tree(candidate_view, legacy_mixture=True)
        )
    branch_domain = count_domain(plot_data.edge_table)
    # Count colour settings come from the shared table; the figure's own override it.
    colouring = {**rendering, **figure}
    edge_width = float(figure.get("tree_edge_width", figure.get("edge_width", 1.2)))
    highlight_width = float(figure.get("highlight_edge_width", 3.0))
    tree_width = int(figure.get("figure_width", figure.get("tree_width", 520)))
    tree_height = int(figure.get("tree_height", 250))
    fitness_span = domain[1] - domain[0]
    display_domain = (
        domain[0] - fitness_span * padding_fraction,
        domain[1] + fitness_span * padding_fraction,
    )
    # The x domain Topographer's tree scale uses, including horizontal padding, so
    # the count labels are placed against the domain the branches are drawn on.
    shared_geometry = tree_geometry(
        plot_data,
        colouring,
        size=(tree_width, tree_height),
        y_domain=display_domain,
    )
    x_domain = shared_geometry.x_domain
    colors = _branch_colors(figure)
    peaks, saddle = _branch_geometry(plot_data, metadata)
    stop_rendering = {**rendering, **figure}
    # The stop-codon nodes of the drawn tree whose kind is allowed by
    # [rendering] stop_codon_tree_node_kinds (peaks and the minimum by default), read from
    # the tree's own node table. Only a product whose saved node table has no
    # has_stop_codon column keeps its saved maxima-only ``stop_points``.
    saved_stops = None if render_tables is None else render_tables.get("stop_points")
    if "has_stop_codon" in plot_data.node_table.columns:
        stop_points = stop_codon_nodes(plot_data.node_table, stop_rendering)
    elif saved_stops is not None:
        stop_points = saved_stops
    else:
        LOGGER.warning(
            "codon_entropy: the tree node table has no has_stop_codon column and no "
            "saved stop_points; drawing no stop X for this tree."
        )
        stop_points = plot_data.node_table.clear()
    all_pairs = {
        (
            int(row["source_tree_node_index"]),
            int(row["target_tree_node_index"]),
        )
        for row in plot_data.edge_segment_table.to_dicts()
    }
    tree = _full_tree_chart(
        plot_data,
        all_pairs,
        width=tree_width,
        height=tree_height,
        domain=display_domain,
        ticks=ticks,
        padding_fraction=0.0,
        x_domain=x_domain,
        edge_width=edge_width,
        show_fitness_axis=True,
        fitness_axis_title=fitness_axis_title,
        fitness_tick_format=fitness_tick_format,
        rendering=colouring,
        count_domain=branch_domain,
        dimmed_opacity=1.0,
    )
    labels = branch_count_labels(
        plot_data,
        colouring,
        size=(tree_width, tree_height),
        y_domain=display_domain,
        stop_points=stop_points,
        wild_type_source=tree_analysis,
        colorbar_box=count_colorbar_box(
            figure, colouring, branch_domain, width=tree_width
        ),
    )
    highlights = [
        _highlight_layer(
            peaks[name]["segments"],
            colors[name],
            edge_width=highlight_width,
            x_domain=x_domain,
            domain=display_domain,
        )
        for name in BRANCH_NAMES
    ]
    markers = _marker_layers(
        peaks,
        saddle,
        colors,
        figure,
        rendering,
        x_domain=x_domain,
        domain=display_domain,
    )
    marker_table = getattr(tree_analysis, "wild_type_marker_table", None)
    has_wild_type = marker_table is not None and marker_table.height > 0
    wild_type = (
        [
            _pin_explicit_scales(layer, x_domain, display_domain)
            for layer in wild_type_layers(tree_analysis, rendering)
        ]
        if has_wild_type
        else []
    )
    has_stops = stop_points is not None and stop_points.height > 0
    # The shared one-line key that ``tree_chart`` draws on a single tree, naming the
    # wild-type star and the stop crosses.
    keys = (
        [
            annotation_legend(
                tree_width,
                tree_height,
                stop_rendering,
                wild_type=has_wild_type,
                stop_codon=has_stops,
            )
        ]
        if has_wild_type or has_stops
        else []
    )
    chart = compose_tree_layers(
        tree,
        overlays=highlights,
        count_labels=[*labels, *markers],
        stop_markers=[
            _scaled_stop_layer(
                stop_points, stop_rendering, x_domain=x_domain, domain=display_domain
            )
        ],
        keys=keys,
        wild_type=wild_type,
    ).properties(width=tree_width, height=tree_height)
    y_axis_extent = int(
        figure.get("tree_y_axis_extent", rendering.get("tree_y_axis_extent", 60))
    )
    chart = _pin_y_axis_extent(chart, y_axis_extent)
    return overlay_count_colorbar(
        chart, figure, colouring, branch_domain, width=tree_width
    )


def _pin_y_axis_extent(node: Any, extent: int) -> Any:
    """Fix the width of every explicit y axis under ``node`` to ``extent`` pixels.

    Vega-Lite sizes a panel's left margin from its tick labels and title, so panels in
    separate concats would start their plot areas at different x. Equal ``minExtent``
    and ``maxExtent`` give every panel the same margin. Only encodings that already
    carry an axis are touched.
    """
    encoding = getattr(node, "encoding", alt.Undefined)
    if encoding is not alt.Undefined:
        y = getattr(encoding, "y", alt.Undefined)
        if y is not alt.Undefined:
            spec = dict(y) if isinstance(y, dict) else y.to_dict()
            axis = spec.get("axis")
            if isinstance(axis, dict) and axis and hasattr(node, "encode"):
                spec["axis"] = {**axis, "minExtent": extent, "maxExtent": extent}
                node = node.encode(y=spec)
    for composition in ("layer", "hconcat", "vconcat", "concat"):
        children = getattr(node, composition, alt.Undefined)
        if children is not alt.Undefined:
            node[composition] = [_pin_y_axis_extent(child, extent) for child in children]
    return node


def _fitness_axis(spec: dict[str, Any] | None) -> alt.Axis | None:
    """The shared fitness axis of the first sweep panel (fixed width), or no axis."""
    if spec is None:
        return None
    return alt.Axis(
        values=list(spec["values"]),
        format=spec["format"],
        domain=False,
        ticks=True,
        minExtent=spec["extent"],
        maxExtent=spec["extent"],
    )


def _heatmap(
    frame: pl.DataFrame,
    population: pl.DataFrame,
    *,
    width: int,
    height: int,
    domain: tuple[float, float],
    entropy_max: float,
    scheme: str,
    legend_length: int,
    position_axis_title: str,
    entropy_title: list[str],
    legend_orient: str,
    show_position_axis: bool,
    show_legend: bool = True,
    padding_fraction: float = 0.05,
    outside_color: str = OUTSIDE_GREY,
    article_span: tuple[float, float] | None = None,
    rect_opacity: float = 1.0,
    fitness_axis: dict[str, Any] | None = None,
) -> alt.LayerChart:
    """One entropy sweep: amino-acid position (x) against fitness threshold (y).

    ``position_axis_title`` of ``None`` keeps the position labels and drops the title.
    ``fitness_axis`` (``title``, ``values``, ``format``, ``extent``) draws the shared
    fitness axis on this panel; without it the panel has no y axis and relies on the
    panel beside it.
    """
    rows = frame.sort(["aa_position", "threshold_scalar"]).to_dicts()
    population_min = float(population["fitness"].min())
    population_max = float(population["fitness"].max())
    thresholds = sorted(
        {
            float(row["threshold_scalar"])
            for row in rows
            if row.get("entropy_bits") is not None
            and population_min <= float(row["threshold_scalar"]) <= population_max
        }
    )
    midpoints = [(left + right) / 2 for left, right in pairwise(thresholds)]
    # Tile midpoint intervals and clamp the outer edges to the population's
    # exact raw-fitness limits. Null entropy observations remain blank.
    for row in rows:
        threshold = float(row["threshold_scalar"])
        if threshold not in thresholds:
            continue
        index = thresholds.index(threshold)
        row["threshold_lower"] = population_min if index == 0 else midpoints[index - 1]
        row["threshold_upper"] = (
            population_max if index == len(thresholds) - 1 else midpoints[index]
        )
        row["position_label"] = str(int(row["aa_position"]))
    rows = [row for row in rows if row.get("entropy_bits") is not None]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_rect(opacity=rect_opacity)
        .encode(
            x=alt.X(
                "position_label:N",
                sort=["1", "2", "3"],
                title=position_axis_title if show_position_axis else None,
                axis=alt.Axis(labelAngle=0, domain=False)
                if show_position_axis
                else None,
                scale=alt.Scale(paddingInner=0, paddingOuter=0),
            ),
            y=alt.Y(
                "threshold_lower:Q",
                title=fitness_axis["title"] if fitness_axis else None,
                scale=alt.Scale(
                    domain=list(domain), zero=False, padding=padding_fraction
                ),
                axis=_fitness_axis(fitness_axis),
            ),
            y2="threshold_upper:Q",
            color=alt.Color(
                "entropy_bits:Q",
                scale=alt.Scale(scheme=scheme, domain=[0.0, entropy_max]),
                legend=alt.Legend(
                    title=entropy_title,
                    gradientLength=legend_length,
                    orient=legend_orient,
                    **(
                        {"direction": "horizontal"}
                        if legend_orient in ("top", "bottom")
                        else {}
                    ),
                )
                if show_legend
                else None,
            ),
            tooltip=[
                alt.Tooltip("position_label:N", title="Position"),
                alt.Tooltip("threshold_scalar:Q", title="Fitness", format=".3f"),
                alt.Tooltip("entropy_bits:Q", title="Entropy (bits)", format=".3f"),
                alt.Tooltip("active_sequence_count:Q", title="Active sequences"),
            ],
        )
        .properties(width=width, height=height)
    )
    span = article_span or domain
    filler = [
        {"position_label": str(position), "lower": lower, "upper": upper}
        for position in (1, 2, 3)
        for lower, upper in ((span[0], population_min), (population_max, span[1]))
        if lower < upper
    ]
    background = (
        alt.Chart(alt.Data(values=filler))
        .mark_rect(color=outside_color)
        .encode(
            x=alt.X(
                "position_label:N",
                sort=["1", "2", "3"],
                axis=alt.Axis(labelAngle=0, domain=False)
                if show_position_axis
                else None,
                title=position_axis_title if show_position_axis else None,
            ),
            y=alt.Y(
                "lower:Q",
                axis=_fitness_axis(fitness_axis),
                title=fitness_axis["title"] if fitness_axis else None,
                scale=alt.Scale(
                    domain=list(domain), zero=False, padding=padding_fraction
                ),
            ),
            y2="upper:Q",
        )
    )
    return alt.layer(background, chart).properties(width=width, height=height)


LEGEND_LINE_GAP = 3


def vertical_entropy_legend(
    *,
    entropy_max: float,
    scheme: str,
    height: int,
    bar_width: int,
    width: int,
    title: list[str],
    font_size: int,
    bar_fraction: float = 0.5,
) -> alt.LayerChart:
    """A vertical entropy bar in its own pixel box, origin at the top left.

    The box is ``width`` x ``height`` px: the title at the top (one line per item of
    ``title``), then the gradient bar below it, top-aligned, ``bar_fraction`` of the box
    tall (about half the heat map; capped at the room under the title), 0 at the bottom
    and ``entropy_max`` at the top in ``scheme``, with the two end labels to the right
    of the bar's ends.
    """
    cmap = matplotlib.colormaps[scheme]
    stops = [
        {"offset": step, "color": matplotlib.colors.to_hex(cmap(1 - step))}
        for step in (index / 7 for index in range(8))
    ]
    title_height = len(title) * (font_size + 2)
    bar_top = title_height + LEGEND_LINE_GAP + font_size / 2
    bar_height = min(height - bar_top - font_size / 2, bar_fraction * height)
    layers: list[alt.Chart] = []
    for line, text in enumerate(title):
        layers.append(
            alt.Chart(alt.Data(values=[{}])).mark_text(
                x=0,
                y=line * (font_size + 2),
                text=text,
                align="left",
                baseline="top",
                fontSize=font_size,
            )
        )
    layers.append(
        alt.Chart(alt.Data(values=[{}])).mark_rect(
            x=bar_width / 2,
            y=bar_top + bar_height / 2,
            width=bar_width,
            height=bar_height,
            color=alt.Gradient(
                gradient="linear", stops=stops, x1=0, x2=0, y1=0, y2=1
            ),
        )
    )
    for value, y in ((float(entropy_max), bar_top), (0.0, bar_top + bar_height)):
        layers.append(
            alt.Chart(alt.Data(values=[{}])).mark_text(
                x=bar_width + LEGEND_LINE_GAP + 1,
                y=y,
                text=str(round(value, 1)),
                align="left",
                baseline="middle",
                fontSize=font_size,
            )
        )
    return alt.layer(*layers).properties(width=int(width), height=int(height))


SWEEP_ORDER = ("kem", "tcl", "core")
SWEEP_TITLES = {"kem": "KEM branch", "tcl": "TCL branch", "core": "All branches"}


def _tied_to_branch(chart: Any, title: str, color: str, *, bar: float) -> Any:
    """Title a sweep panel in its branch's colour and rule its top edge in it.

    ``bar`` is the rule's stroke width; zero draws no rule (the all-codon panel).
    """
    layers = [chart]
    if bar > 0:
        layers.append(
            # The position lives in encoding value channels: Vega-Lite 6.4.1's
            # normalizer crashes on a unit with mark-level positions and no
            # encoding at all.
            alt.Chart(alt.Data(values=[{}]))
            .mark_rule(color=color, strokeWidth=bar, opacity=1.0)
            .encode(x=alt.value(0), x2=alt.value("width"), y=alt.value(0))
        )
    return alt.layer(*layers).properties(
        title=alt.TitleParams(
            title, anchor="start", color=color, fontWeight="bold", fontSize=15
        )
    )


def build_bootstrap_codon_entropy_figure(
    comparison: Any, settings: dict[str, Any]
) -> alt.Chart:
    """Build the simplified Candidate tree over the three codon-entropy sweeps.

    Row 1 is the simplified Candidate tree (the bootstrap-simplification FDR's
    band-simplified view) at the figure's width, with the KEM and TCL peak branches
    highlighted from the peak down to the shared saddle, each in its own colour used
    nowhere else in the palette. Each peak carries its amino-acid genotype above; the
    saddle carries a diamond with its genotype to the left and a little below. Row 2 is three entropy sweeps side by side: the
    KEM ridge, the TCL ridge and the all-codon sweep. They share the amino-acid
    position axis, one fitness axis (drawn once, on the first panel, at the tree's
    axis width) and one vertical entropy colour bar to the right of all three. The KEM and TCL panels are
    titled in, and ruled along their top edge by, their branch's colour. The two rows
    are the same width.

    ``comparison["candidate_leaf_view"]`` is the band-simplified view of the
    bootstrap checkpoint, and the payload's populations and entropy sweeps are
    computed from that view (``registry._compose_bootstrap_codon``); the renderer
    never rebuilds a view on its behalf. Every count in the row titles is the
    registry's display text (``comparison["manuscript_quantities"]``).
    """
    figure, rendering = _figure_settings(settings)
    metadata, populations = _population_frames(comparison)
    entropy = _entropy_frames(comparison)
    candidate_view = _candidate_leaf_view(comparison)
    _require_keys(comparison, ("manuscript_quantities",), "keys")
    quantities = comparison["manuscript_quantities"]
    all_fitness = [float(value) for value in populations["core"]["fitness"].to_list()]
    padding_fraction = float(rendering.get("fitness_padding_fraction", 0.05))
    tick_step = float(rendering.get("fitness_tick_step", 0.1))
    domain, ticks = _rounded_fitness_domain(all_fitness, tick_step)
    fitness_span = domain[1] - domain[0]
    display_domain = (
        domain[0] - fitness_span * padding_fraction,
        domain[1] + fitness_span * padding_fraction,
    )
    render_tables = comparison.get("codon_render_tables")
    fitness_title = str(rendering.get("tree_axis_title", "Fitness"))
    fitness_format = str(
        figure.get("fitness_tick_format", rendering.get("fitness_tick_format", ".1f"))
    )
    tree = _tree_chart(
        candidate_view,
        metadata,
        figure,
        rendering,
        domain,
        ticks,
        padding_fraction,
        fitness_title,
        fitness_format,
        render_tables=render_tables,
    )

    figure_width = int(figure.get("figure_width", figure.get("tree_width", 520)))
    panel_gap = int(figure.get("panel_gap", 8))
    row_gap = int(figure.get("row_gap", 8))
    sweep_height = int(figure.get("sweep_height", figure.get("tree_height", 250)))
    y_axis_extent = int(
        figure.get("tree_y_axis_extent", rendering.get("tree_y_axis_extent", 60))
    )
    # Three sweeps share the tree's plot width: the first panel's fitness axis takes the
    # same left margin as the tree's, so both rows are the same width.
    sweep_width = (figure_width - panel_gap * (len(SWEEP_ORDER) - 1)) // len(
        SWEEP_ORDER
    )
    entropy_max = MAX_ENTROPY
    scheme = str(figure.get("entropy_scheme", "viridis"))
    entropy_title = list(figure.get("entropy_title", ["Entropy", "(bits)"]))
    position_title = str(figure.get("position_axis_title", "Amino-acid position"))
    heading_font_size = int(rendering.get("heading_font_size", 18))
    subtitle_font_size = int(rendering.get("subtitle_font_size", 14))
    font_size = int(figure.get("font_size", rendering.get("font_size", 15)))
    colors = _branch_colors(figure)
    bar = float(figure.get("sweep_tie_width", 4))
    sweeps = []
    for position, name in enumerate(SWEEP_ORDER):
        heat = _heatmap(
            entropy[name],
            populations[name],
            width=sweep_width,
            height=sweep_height,
            domain=display_domain,
            entropy_max=entropy_max,
            scheme=scheme,
            legend_length=int(figure.get("entropy_legend_length", 110)),
            # Every panel shows the position labels; the title is written once, centred.
            position_axis_title=position_title
            if position == len(SWEEP_ORDER) // 2
            else None,
            entropy_title=entropy_title,
            legend_orient=str(figure.get("entropy_legend_orient", "right")),
            show_position_axis=True,
            show_legend=False,
            padding_fraction=0.0,
            outside_color=str(figure.get("entropy_outside_color", OUTSIDE_GREY)),
            article_span=(min(all_fitness), max(all_fitness)),
            rect_opacity=float(figure.get("entropy_rect_opacity", 1.0)),
            fitness_axis={
                "title": fitness_title,
                "values": list(ticks),
                "format": fitness_format,
                "extent": y_axis_extent,
            }
            if position == 0
            else None,
        )
        sweeps.append(
            _tied_to_branch(
                heat,
                SWEEP_TITLES[name],
                colors.get(name, INK_GREY),
                bar=bar if name in colors else 0,
            ).properties(width=sweep_width, height=sweep_height)
        )

    # One vertical entropy bar to the right of the three sweeps. Its box is as tall as
    # a sweep and carries a blank title of the sweeps' title font so that its top lines
    # up with theirs.
    legend = vertical_entropy_legend(
        entropy_max=entropy_max,
        scheme=scheme,
        height=sweep_height,
        bar_width=int(figure.get("entropy_legend_bar_width", 12)),
        width=int(figure.get("entropy_legend_width", 56)),
        title=entropy_title,
        font_size=font_size,
        bar_fraction=float(figure.get("entropy_legend_bar_fraction", 0.5)),
    ).properties(
        title=alt.TitleParams("\u00a0", anchor="start", fontSize=15)
    )

    core_title = (
        f"{figure.get('candidate_title', 'Band-smoothed candidate graph')} · "
        f"{quantities.display('candidate_graph.functional_count')} functional / "
        f"{quantities.display('candidate_graph.vertex_count')} total sequences"
    )
    sweep_title = (
        f"Codon entropy · KEM ridge "
        f"{quantities.display('ridges.band_view.kem.functional_count')} and TCL ridge "
        f"{quantities.display('ridges.band_view.tcl.functional_count')} "
        "functional sequences"
    )
    # Row titles sit on the rows, not the panels: a title wider than its panel would
    # widen that panel and move the axes.
    rows = [
        alt.hconcat(
            tree,
            title=alt.TitleParams(
                core_title, anchor="start", fontSize=heading_font_size
            ),
        ),
        alt.hconcat(
            *sweeps,
            legend,
            spacing=panel_gap,
            title=alt.TitleParams(
                sweep_title,
                anchor="start",
                fontSize=heading_font_size,
                fontWeight="bold",
            ),
        ).resolve_scale(y="shared", x="independent", color="shared"),
    ]
    return (
        alt.vconcat(*rows, spacing=row_gap)
        .resolve_scale(color="independent", x="independent", y="independent")
        .properties(
            title=alt.TitleParams(
                str(
                    figure.get("entropy_outside_label", "Gray: outside population span")
                ),
                orient="bottom",
                anchor="end",
                offset=10,
                fontSize=subtitle_font_size,
                fontWeight="normal",
            )
        )
        .configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_view(stroke=None)
        .configure_axis(grid=False, labelFontSize=font_size, titleFontSize=font_size)
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
    )
