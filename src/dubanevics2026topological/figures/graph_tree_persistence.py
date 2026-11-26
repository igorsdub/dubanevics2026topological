"""Native Altair panels for the conceptual graph–tree–persistence construction."""

from __future__ import annotations

from collections import deque
from colorsys import hls_to_rgb, rgb_to_hls
from dataclasses import dataclass
import math
from math import cos, radians, sin
from typing import Any

import altair as alt
import numpy as np
import polars as pl
from topographer.topology import build_tree_metrics

from dubanevics2026topological.analysis.manifold_graph_tree import (
    ManifoldGraphTreeConstruction,
    scalar_value,
)
from dubanevics2026topological.figures.palette import INK_GREY
from ._branch_count_style import branch_count_opacity
from ._charts import _legend_entry_layers, dashed_threshold_rule_style

FIGURE_ID = "graph_tree_persistence"
_TREE_Y_DOMAIN = (-0.03, 1.05)
_HEIGHT_TITLE = "Fitness"
_PERSISTENCE_X_TITLE = "Fitness at death"
_PERSISTENCE_Y_TITLE = "Fitness at birth"
# No valley markers (GLOSSARY, Figure marker rules): the global minimum has no
# entry, so it falls back to the plain circle of a regular vertex.
_EVENT_SHAPES = {
    "maximum": "triangle-up",
    "split_saddle": "diamond",
}


@dataclass(frozen=True)
class ProjectedPanelData:
    """Inspectable vector mark data used by the graph panel."""

    threshold: float
    projection: dict[str, float]
    graph_surface: tuple[dict[str, Any], ...]
    sweep_plane: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    graph_edges_inactive: tuple[dict[str, Any], ...]
    graph_edges_active: tuple[dict[str, Any], ...]
    tree_nodes: tuple[dict[str, Any], ...]
    tree_edges: tuple[dict[str, Any], ...]
    trunk_sample_ids: tuple[str, ...]
    fitness_axis: dict[str, Any]


def _appearance_aligned_to_tree_layout(
    appearance: dict[str, Any],
    construction: ManifoldGraphTreeConstruction,
    layout_nodes: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Swap left/right peak colours when the tree layout places them opposite geometry.

    ``subtree_extremum`` can put the geometric right peak on the left of the tree.
    Align presentation colours so the leftmost tree peak uses ``left_peak_color``
    and the rightmost uses ``right_peak_color`` across graph, tree, and persistence.
    """
    xs = {
        str(row["vertex_id"]): float(row["x"]) for row in layout_nodes.values()
    }
    left_id = construction.critical_ids["left_peak"]
    right_id = construction.critical_ids["right_peak"]
    if left_id not in xs or right_id not in xs or xs[left_id] <= xs[right_id]:
        return appearance
    aligned = dict(appearance)
    aligned["left_peak_color"], aligned["right_peak_color"] = (
        appearance["right_peak_color"],
        appearance["left_peak_color"],
    )
    return aligned


def build_graph_tree_persistence_panel_data(
    construction: ManifoldGraphTreeConstruction,
    threshold: float,
    settings: dict[str, Any],
) -> ProjectedPanelData:
    """Project geometry and assemble marker/edge states for one threshold."""
    appearance = _appearance(settings)
    layout_nodes, layout_edges = _layout_records(construction)
    appearance = _appearance_aligned_to_tree_layout(
        appearance, construction, layout_nodes
    )
    threshold = float(threshold)
    state = construction.sweep_state(threshold)
    active_ids = set(state.active_sample_ids)
    saddle = construction.sample_table.filter(
        pl.col("sample_id") == construction.critical_ids["saddle"]
    ).row(0, named=True)
    saddle_value = float(saddle["scalar"])
    basin_boundary = float(saddle["x"] + saddle["y"])
    projection = _projection_parameters(appearance)
    surface_resolution = int(appearance["presentation_surface_resolution"])
    base_axis = np.linspace(0.0, 1.0, surface_resolution)
    critical_samples = construction.sample_table.filter(
        pl.col("sample_id").is_in(list(construction.critical_ids.values()))
    )
    x_axis = np.unique(np.concatenate((base_axis, critical_samples["x"].to_numpy())))
    y_axis = np.unique(np.concatenate((base_axis, critical_samples["y"].to_numpy())))
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    surface_resolution_y, surface_resolution_x = x_grid.shape
    values = np.asarray(
        scalar_value(x_grid, y_grid, settings=construction.scientific_settings)
    )
    projection_fn = _projector(appearance)

    graph_surface: list[dict[str, Any]] = []
    for row in range(surface_resolution_y - 1):
        for column in range(surface_resolution_x - 1):
            vertices = (
                _surface_vertex(
                    x_grid[row, column], y_grid[row, column], values[row, column]
                ),
                _surface_vertex(
                    x_grid[row, column + 1],
                    y_grid[row, column + 1],
                    values[row, column + 1],
                ),
                _surface_vertex(
                    x_grid[row + 1, column + 1],
                    y_grid[row + 1, column + 1],
                    values[row + 1, column + 1],
                ),
                _surface_vertex(
                    x_grid[row + 1, column],
                    y_grid[row + 1, column],
                    values[row + 1, column],
                ),
            )
            for triangle_number, triangle in enumerate(
                (
                    (vertices[0], vertices[1], vertices[2]),
                    (vertices[0], vertices[2], vertices[3]),
                )
            ):
                # A subdued grayscale surface contextualizes the sampled graph.
                full_feature = _surface_feature(
                    triangle,
                    projection_fn,
                    color=appearance["graph_surface_color"],
                    opacity=float(appearance["graph_surface_opacity"]),
                    shade=_surface_shade(
                        triangle, float(appearance["vertical_box_aspect"])
                    ),
                    feature_id=f"{row}-{column}-{triangle_number}-graph",
                )
                if full_feature is not None:
                    graph_surface.append(full_feature)

    graph_surface.sort(key=lambda feature: feature["depth"])

    plane_vertices = tuple(
        _surface_vertex(x, y, threshold)
        for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    sweep_plane = _surface_feature(
        plane_vertices,
        projection_fn,
        color=appearance["sweep_plane_color"],
        opacity=float(appearance["sweep_plane_opacity"]),
        feature_id="sweep-plane",
    )

    samples = []
    sample_by_id: dict[str, dict[str, Any]] = {}
    for sample in construction.sample_table.iter_rows(named=True):
        sample_id = str(sample["sample_id"])
        active = sample_id in active_ids
        event = str(sample["event"])
        marker = _EVENT_SHAPES.get(event) if active else None
        marker = marker or "circle"
        px, py, depth = projection_fn(sample["x"], sample["y"], sample["scalar"])
        fill = (
            _lineage_color(
                sample["x"],
                sample["y"],
                sample["scalar"],
                threshold,
                saddle_value,
                appearance,
                basin_boundary,
            )
            if active
            else appearance["inactive_color"]
        )
        is_surface_event = sample_id in construction.critical_ids.values()
        record = {
            "sample_id": sample_id,
            "x": float(sample["x"]),
            "y": float(sample["y"]),
            "scalar": float(sample["scalar"]),
            "projected_x": float(px),
            "projected_y": float(py),
            "depth": float(depth),
            "event": event,
            "display_event": event if active else "regular",
            "critical_name": sample["critical_name"],
            "active": active,
            "is_critical": is_surface_event,
            "shape": marker,
            "color": fill,
            "outline_color": _outline_color(fill, appearance),
            "opacity": float(
                appearance["active_sample_opacity"]
                if active
                else appearance["inactive_sample_opacity"]
            ),
            "size": int(
                appearance["manifold_critical_node_size"]
                if active and is_surface_event
                else appearance["manifold_regular_node_size"]
            ),
            "graph_size": int(
                appearance["graph_critical_node_size"]
                if active and is_surface_event
                else appearance["graph_regular_node_size"]
            ),
        }
        samples.append(record)
        sample_by_id[sample_id] = record

    inactive_edges: list[dict[str, Any]] = []
    active_edges: list[dict[str, Any]] = []
    for edge_index, (source_id, target_id) in enumerate(
        construction.edge_table.iter_rows()
    ):
        source = sample_by_id[source_id]
        target = sample_by_id[target_id]
        is_active = source_id in active_ids and target_id in active_ids
        color = appearance["edge_color"]
        if is_active:
            if source["color"] != target["color"] or (
                source["scalar"] <= saddle_value and target["scalar"] <= saddle_value
            ):
                color = appearance["shared_region_color"]
            else:
                color = source["color"]
        target_collection = active_edges if is_active else inactive_edges
        for order, point in enumerate((source, target)):
            target_collection.append(
                {
                    "edge_id": f"edge-{edge_index}",
                    "point_order": order,
                    "projected_x": point["projected_x"],
                    "projected_y": point["projected_y"],
                    "depth": (source["depth"] + target["depth"]) / 2,
                    "color": color,
                    "opacity": float(
                        appearance["active_edge_opacity"]
                        if is_active
                        else appearance["inactive_edge_opacity"]
                    ),
                    "width": float(
                        appearance["active_edge_width"]
                        if is_active
                        else appearance["inactive_edge_width"]
                    ),
                }
            )
    inactive_edges.sort(key=lambda row: (row["depth"], row["edge_id"]))
    active_edges.sort(key=lambda row: (row["depth"], row["edge_id"]))

    trunk_ids = _trunk_ids(layout_nodes, layout_edges, construction)
    tree_nodes, tree_edges = _tree_sweep_records(
        construction,
        layout_nodes,
        layout_edges,
        trunk_ids,
        active_ids,
        threshold,
        saddle_value,
        appearance,
        basin_boundary,
    )
    return ProjectedPanelData(
        threshold=threshold,
        projection=projection,
        graph_surface=tuple(graph_surface),
        sweep_plane=sweep_plane,
        samples=tuple(samples),
        graph_edges_inactive=tuple(inactive_edges),
        graph_edges_active=tuple(active_edges),
        tree_nodes=tuple(tree_nodes),
        tree_edges=tuple(tree_edges),
        trunk_sample_ids=tuple(sorted(trunk_ids)),
        fitness_axis=_fitness_axis_records(
            appearance, projection, projection_fn, _height_ticks(construction)
        ),
    )


def build_graph_tree_persistence_panels(
    construction: ManifoldGraphTreeConstruction,
    threshold: float,
    settings: dict[str, Any],
    *,
    show_fitness_axis: bool = True,
    show_death_axis: bool = True,
    show_graph_legend: bool = False,
) -> dict[str, alt.TopLevelMixin]:
    """Return native Altair charts for the graph, tree, and persistence panels."""
    appearance = _appearance(settings)
    layout_nodes, _ = _layout_records(construction)
    appearance = _appearance_aligned_to_tree_layout(
        appearance, construction, layout_nodes
    )
    data = build_graph_tree_persistence_panel_data(construction, threshold, settings)
    projection = {
        "type": "identity",
        "reflectY": True,
        "scale": float(data.projection["scale"]),
        "translate": [
            float(data.projection["translate_x"]),
            float(data.projection["translate_y"]),
        ],
    }
    width = int(appearance["panel_width"])
    height = int(appearance["panel_height"])
    # Critical samples are drawn last so no regular node can hide them.
    regular_samples = tuple(row for row in data.samples if not row["is_critical"])
    critical_samples = tuple(row for row in data.samples if row["is_critical"])
    graph_layers = [
        _polygon_chart(data.graph_surface),
        _line_chart(data.graph_edges_inactive),
        _polygon_chart((data.sweep_plane,)),
        _line_chart(data.graph_edges_active),
        _point_chart(regular_samples, appearance, size_field="graph_size"),
        _point_chart(critical_samples, appearance, size_field="graph_size"),
    ]
    if show_fitness_axis:
        graph_layers.extend(_fitness_axis_charts(data.fitness_axis, appearance))
    if show_graph_legend:
        graph_layers.extend(_graph_legend_layers(appearance, width))
    graph = _projected_chart(graph_layers, width, height, projection)
    tree = _tree_chart(
        # Critical points last, the global minimum on top of them.
        tuple(sorted(data.tree_nodes, key=lambda row: int(row["draw_rank"]))),
        data.tree_edges,
        data.threshold,
        appearance,
        width,
        height,
        _height_ticks(construction),
        show_fitness_axis=show_fitness_axis,
    )
    persistence = _persistence_chart(
        construction,
        threshold,
        appearance,
        show_fitness_axis=show_fitness_axis,
        show_death_axis=show_death_axis,
    )
    return {
        "graph": graph,
        "tree": tree,
        "persistence": persistence,
    }


def build_graph_tree_persistence_preview(
    construction: ManifoldGraphTreeConstruction,
    threshold: float,
    settings: dict[str, Any],
) -> alt.HConcatChart:
    """Return one row of Graph, Tree, and Persistence panels at one threshold."""
    appearance = _appearance(settings)
    rendering = settings["figures"].get("rendering", {})
    panels = build_graph_tree_persistence_panels(construction, threshold, settings)
    columns = [
        panels[key].properties(title=title)
        for key, title in zip(
            ("graph", "tree", "persistence"), appearance["row_titles"], strict=True
        )
    ]
    return (
        alt.hconcat(*columns, spacing=int(rendering.get("panel_gap", 10)))
        .configure_text(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_title(
            font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"),
            fontSize=int(rendering.get("font_size", 15)),
        )
        .configure_axis(
            titleFontSize=int(appearance["axis_title_font_size"]),
            labelFontSize=int(appearance["tick_label_font_size"]),
        )
        .configure_view(stroke=None)
    )


def build_graph_tree_persistence_figure(
    construction: ManifoldGraphTreeConstruction,
    settings: dict[str, Any],
) -> alt.VConcatChart:
    """Compose a landscape Graph, Tree, and Persistence grid over four stages."""
    rendering = settings["figures"].get("rendering", {})
    appearance = _appearance(settings)
    gap = int(rendering.get("panel_gap", 10))
    panel_width = int(appearance["panel_width"])
    panel_height = int(appearance["panel_height"])
    label_width = int(appearance["row_label_width"])
    common = {
        "font": rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"),
        "color": INK_GREY,
    }
    thresholds = construction.thresholds[1:]
    stage_titles = list(appearance["stage_titles"])
    if len(stage_titles) != len(thresholds):
        raise ValueError("stage_titles must have one title per sweep stage")
    row_titles = list(appearance["row_titles"])
    if len(row_titles) != 3:
        raise ValueError("row_titles must name the Graph, Tree, and Persistence rows")

    heading_row = alt.hconcat(
        _label_chart(
            "",
            width=label_width,
            height=int(appearance["heading_height"]),
            font_size=int(appearance["heading_font_size"]),
        ),
        *[
            _label_chart(
                f"{number}. {title}",
                width=panel_width,
                height=int(appearance["heading_height"]),
                font_size=int(appearance["heading_font_size"]),
                weight="bold",
                align="center",
            )
            for number, title in enumerate(stage_titles, start=1)
        ],
        spacing=gap,
    )
    stages = [
        build_graph_tree_persistence_panels(
            construction,
            threshold,
            settings,
            show_fitness_axis=index == 0,
            show_death_axis=False,
            show_graph_legend=index == len(thresholds) - 1,
        )
        for index, threshold in enumerate(thresholds)
    ]
    rows = [
        alt.hconcat(
            _label_chart(
                title,
                width=label_width,
                height=panel_height,
                font_size=int(appearance["row_title_font_size"]),
                weight="bold",
                align="center",
                angle=270,
            ),
            *[panels[key] for panels in stages],
            spacing=gap,
        )
        for title, key in zip(row_titles, ("graph", "tree", "persistence"), strict=True)
    ]
    # One "Fitness at death" title, centred under the whole Persistence row.
    death_title = alt.hconcat(
        _label_chart(
            "",
            width=label_width,
            height=int(appearance["heading_height"]),
            font_size=int(appearance["axis_title_font_size"]),
        ),
        _label_chart(
            str(appearance.get("persistence_x_title", _PERSISTENCE_X_TITLE)),
            width=4 * panel_width + 3 * gap,
            height=int(appearance["heading_height"]),
            font_size=int(appearance["axis_title_font_size"]),
            weight="bold",
            align="center",
        ),
        spacing=gap,
    )
    return (
        alt.vconcat(heading_row, *rows, death_title, spacing=int(appearance["row_gap"]))
        .configure_text(**common)
        .configure_view(stroke=None)
        .configure_axis(
            grid=False,
            domain=False,
            ticks=False,
            labels=False,
            titleFontSize=int(appearance["axis_title_font_size"]),
            labelFontSize=int(appearance["tick_label_font_size"]),
        )
    )


def _label_chart(
    text: str,
    *,
    width: int,
    height: int,
    font_size: int,
    weight: str = "normal",
    align: str = "left",
    angle: float = 0,
) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=[{"label": text}]))
        .mark_text(
            align=align,
            baseline="middle",
            fontSize=font_size,
            fontWeight=weight,
            angle=angle,
        )
        .encode(
            text="label:N",
            x=alt.value(width / 2 if align == "center" else 0),
            y=alt.value(height / 2),
        )
        .properties(width=width, height=height)
    )


def _projected_chart(
    layers: list[alt.TopLevelMixin],
    width: int,
    height: int,
    projection: dict[str, Any],
) -> alt.LayerChart:
    return (
        alt.layer(*layers).properties(width=width, height=height).project(**projection)
    )


def _polygon_chart(records: tuple[dict[str, Any], ...]) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=list(records)))
        .mark_geoshape(stroke=None)
        .encode(
            shape=alt.Shape("geometry:geojson"),
            color=alt.Color("color:N", scale=None, legend=None),
            opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
            order=alt.Order("depth:Q", sort="ascending"),
        )
    )


def _point_chart(
    records: tuple[dict[str, Any], ...],
    appearance: dict[str, Any],
    *,
    size_field: str = "size",
) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=list(records)))
        .mark_point(filled=True, strokeWidth=_outline_width(appearance))
        .encode(
            longitude=alt.Longitude("projected_x:Q"),
            latitude=alt.Latitude("projected_y:Q"),
            shape=alt.Shape("shape:N", scale=None),
            color=alt.Color("color:N", scale=None, legend=None),
            stroke=alt.Stroke("outline_color:N", scale=None, legend=None),
            opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
            size=alt.Size(f"{size_field}:Q", scale=None, legend=None),
            order=alt.Order("depth:Q", sort="ascending"),
            tooltip=["sample_id:N", "display_event:N", "scalar:Q"],
        )
    )


def _line_chart(records: tuple[dict[str, Any], ...]) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=list(records)))
        .mark_line()
        .encode(
            longitude=alt.Longitude("projected_x:Q"),
            latitude=alt.Latitude("projected_y:Q"),
            detail=alt.Detail("edge_id:N"),
            order=alt.Order("point_order:Q"),
            color=alt.Color("color:N", scale=None, legend=None),
            opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
            size=alt.Size("width:Q", scale=None, legend=None),
        )
    )


def _tree_chart(
    nodes: tuple[dict[str, Any], ...],
    edges: tuple[dict[str, Any], ...],
    threshold: float,
    appearance: dict[str, Any],
    width: int,
    height: int,
    ticks: list[float],
    show_fitness_axis: bool = False,
) -> alt.LayerChart:
    y_axis = (
        _fitness_axis(appearance.get("height_axis_title", _HEIGHT_TITLE))
        if show_fitness_axis
        else None
    )
    edge_chart = (
        alt.Chart(alt.Data(values=list(edges)))
        .mark_line()
        .encode(
            x=alt.X(
                "x:Q", scale=alt.Scale(domain=_tree_x_domain(appearance)), axis=None
            ),
            y=alt.Y("y:Q", scale=alt.Scale(domain=_height_domain(ticks)), axis=y_axis),
            detail=alt.Detail("edge_id:N"),
            order=alt.Order("point_order:Q"),
            color=alt.Color("color:N", scale=None, legend=None),
            opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
            size=alt.Size("width:Q", scale=None, legend=None),
        )
    )
    node_chart = (
        alt.Chart(alt.Data(values=list(nodes)))
        .mark_point(filled=True, strokeWidth=_outline_width(appearance))
        .encode(
            x=alt.X(
                "x:Q", scale=alt.Scale(domain=_tree_x_domain(appearance)), axis=None
            ),
            y=alt.Y("y:Q", scale=alt.Scale(domain=_height_domain(ticks)), axis=y_axis),
            shape=alt.Shape("shape:N", scale=None),
            color=alt.Color("color:N", scale=None, legend=None),
            stroke=alt.Stroke("outline_color:N", scale=None, legend=None),
            opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
            size=alt.Size("size:Q", scale=None, legend=None),
            # Explicit mark order, not the data order alone: a higher rank draws on top.
            order=alt.Order("draw_rank:Q", sort="ascending"),
            tooltip=["sample_id:N", "display_event:N", "scalar:Q"],
        )
    )
    threshold_chart = _threshold_rule(threshold, appearance).encode(
        y=alt.Y(
            "threshold:Q",
            scale=alt.Scale(domain=_height_domain(ticks)),
            axis=y_axis,
        )
    )
    return alt.layer(threshold_chart, edge_chart, node_chart).properties(
        width=width, height=height
    )


def _graph_legend_layers(appearance: dict[str, Any], width: int) -> list[alt.Chart]:
    """Peak and saddle key at the top right of the graph panel (no valley entry)."""
    font_size = int(appearance.get("legend_font_size", 15))
    entries = [("▲", "peak"), ("◆", "saddle")]
    inset = int(appearance.get("legend_inset", 8))
    key_width = 14 + math.ceil(0.55 * font_size * max(len(label) for _, label in entries))
    x = width - inset - key_width
    layers: list[alt.Chart] = []
    for index, (marker, label) in enumerate(entries):
        y = inset + font_size / 2 + index * (font_size + 4)
        layers.extend(
            _legend_entry_layers(marker, label, INK_GREY, x, y, font_size)
        )
    return layers


def _persistence_chart(
    construction: ManifoldGraphTreeConstruction,
    threshold: float,
    appearance: dict[str, Any],
    *,
    show_fitness_axis: bool = True,
    show_death_axis: bool = True,
) -> alt.LayerChart:
    """Birth-death panel: alive peaks slide with the threshold, closed peaks stay.

    Every marker is filled and outlined in a darker shade of its own colour (no hollow
    markers; the peak that appears at a stage is filled like the others). An alive peak is filled in its lineage colour; a closed
    peak, merged or the global peak paired with the global minimum, is filled in
    ``persistence_dead_color``.
    """
    ticks = _height_ticks(construction)
    low, high = _height_domain(ticks)
    scale = alt.Scale(domain=[low, high])
    # The axes carry a title and nothing else (no ticks, tick numbers, rule or
    # grid), overriding the figure-wide axis config that hides every other panel's
    # axes. The death axis is titled once by the composed figure (``show_death_axis``);
    # the first column titles the birth axis.
    x_axis = alt.Axis(
        title=(
            appearance.get("persistence_x_title", _PERSISTENCE_X_TITLE)
            if show_death_axis
            else None
        ),
        labels=False,
        ticks=False,
        domain=False,
        grid=False,
    )
    y_axis = _fitness_axis(
        appearance.get("persistence_y_title", _PERSISTENCE_Y_TITLE),
        show=show_fitness_axis,
    )
    x_encoding = alt.X("x:Q", scale=scale, axis=x_axis)
    y_encoding = alt.Y("y:Q", scale=scale, axis=y_axis)
    lineage_colors = {
        "left_peak": str(appearance["left_peak_color"]),
        "right_peak": str(appearance["right_peak_color"]),
    }
    fallback_colour = str(appearance["inactive_color"])
    count_by_peak = getattr(construction.tree, "branch_counts", None)
    if count_by_peak is None:
        metrics = build_tree_metrics(construction.tree).branch_table
        # Reduced-tree indices differ from augmented-layout indices.
        reduced_graph = construction.tree.graph
        count_by_peak = {
            str(reduced_graph.get_node_data(int(row["source_tree_node_index"]))["vertex_id"]):
            int(row["branch_vertex_count"])
            for row in metrics.iter_rows(named=True)
        }
    count_domain = (1.0, float(max(count_by_peak.values(), default=1)))
    filled_rows = []
    for event in construction.persistence_events(threshold):
        lineage = lineage_colors.get(event.peak_name, fallback_colour)
        row = {
            "peak_name": event.peak_name,
            "sample_id": event.birth_sample_id,
            "birth": event.birth,
            "death": event.death,
            "alive": event.alive,
            "y": event.birth,
            "count_opacity": branch_count_opacity(
                count_by_peak[event.birth_sample_id], appearance, count_domain
            ),
        }
        if event.alive:
            row = {**row, "x": float(threshold), "color": lineage}
        else:
            row = {
                **row,
                "x": event.death,
                "color": str(appearance["persistence_dead_color"]),
            }
        filled_rows.append(
            {**row, "outline_color": _outline_color(row["color"], appearance)}
        )
    diagonal = (
        alt.Chart(alt.Data(values=[{"x": low, "y": low}, {"x": high, "y": high}]))
        .mark_line(
            color=appearance["edge_color"],
            strokeWidth=float(appearance["persistence_diagonal_width"]),
        )
        .encode(x=x_encoding, y=y_encoding)
    )
    threshold_chart = _threshold_rule(threshold, appearance).encode(
        y=alt.Y("threshold:Q", scale=scale, axis=y_axis)
    )
    tooltip = ["sample_id:N", "peak_name:N", "birth:Q", "death:Q"]
    filled_chart = (
        alt.Chart(alt.Data(values=filled_rows))
        .mark_point(
            filled=True,
            shape=_EVENT_SHAPES["maximum"],
            size=float(appearance["persistence_marker_size"]),
            strokeWidth=_outline_width(appearance),
        )
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=alt.value(INK_GREY),
            opacity=alt.Opacity("count_opacity:Q", scale=None, legend=None),
            tooltip=tooltip,
        )
    )
    return alt.layer(diagonal, threshold_chart, filled_chart).properties(
        width=int(appearance["panel_width"]), height=int(appearance["panel_height"])
    )


def _threshold_rule(threshold: float, appearance: dict[str, Any]) -> alt.Chart:
    """The sweep threshold as a dashed horizontal rule (Tree and Persistence rows)."""
    return alt.Chart(alt.Data(values=[{"threshold": threshold}])).mark_rule(
        **dashed_threshold_rule_style(appearance)
    )


def _outline_color(color: str, appearance: dict[str, Any]) -> str:
    """A darker shade of ``color``, for the outline of a marker filled with it."""
    return _shade_color(color, float(appearance["marker_outline_shade"]))


def _outline_width(appearance: dict[str, Any]) -> float:
    return float(appearance["marker_outline_width"])


def _height_ticks(construction: ManifoldGraphTreeConstruction) -> list[float]:
    """Return 0 and 1 for [0, 1]-normalised data, else the data minimum and maximum."""
    scalars = construction.sample_table["scalar"]
    low, high = float(scalars.min()), float(scalars.max())
    if np.isclose(low, 0.0) and np.isclose(high, 1.0):
        return [0.0, 1.0]
    return [low, high]


def _height_domain(ticks: list[float]) -> list[float]:
    """Return the height scale domain: the ticks' span with the small fixed padding.

    For [0, 1] data this is ``_TREE_Y_DOMAIN`` itself. Otherwise the same
    padding (as a fraction of the tick span) is applied around the data minimum
    and maximum so both ticks stay on the scale.
    """
    low, high = _TREE_Y_DOMAIN
    if ticks == [0.0, 1.0]:
        return [low, high]
    span = ticks[-1] - ticks[0]
    return [ticks[0] + low * span, ticks[-1] + (high - 1.0) * span]


def _fitness_axis(title: str, *, show: bool = True) -> alt.Axis:
    """Left height axis shared by the Tree and Persistence rows: a title only.

    No ticks, tick numbers, rule or grid; ``show=False`` drops the title too.
    """
    return alt.Axis(
        title=title if show else None,
        labels=False,
        ticks=False,
        domain=False,
        grid=False,
    )


def _fitness_axis_records(
    appearance: dict[str, Any],
    projection: dict[str, float],
    projector,
    ticks: list[float],
) -> dict[str, Any]:
    """Projected title of a left height axis for the graph panel.

    The title sits at the middle of the projected span between the data minimum and
    maximum, on the surface's left-most corner, so it is centred on the same
    heights as the Tree and Persistence rows' axis titles. The axis has no ticks,
    tick numbers or rule.
    """
    scale = float(projection["scale"])
    corner = min(
        ((x, y) for x in (0.0, 1.0) for y in (0.0, 1.0)),
        key=lambda xy: float(projector(xy[0], xy[1], 0.0)[0]),
    )
    low, high = (float(projector(corner[0], corner[1], value)[1]) for value in ticks)
    # Pixel x = 0 is the panel's left edge, where the other rows draw their axes.
    rule_x = -float(projection["translate_x"]) / scale
    middle = {
        "label": appearance.get("height_axis_title", _HEIGHT_TITLE),
        "x": rule_x,
        "y": (low + high) / 2,
    }
    return {"title": [middle]}


def _fitness_axis_charts(
    axis: dict[str, Any], appearance: dict[str, Any]
) -> list[alt.Chart]:
    """The graph panel's height axis: its title and nothing else."""
    return [
        alt.Chart(alt.Data(values=axis["title"]))
        .mark_text(
            color=str(appearance["fitness_axis_color"]),
            baseline="middle",
            clip=False,
            align="center",
            fontSize=int(appearance["axis_title_font_size"]),
            fontWeight="bold",
            angle=270,
            dy=-float(appearance["fitness_axis_title_offset"]),
        )
        .encode(
            text="label:N",
            longitude=alt.Longitude("x:Q"),
            latitude=alt.Latitude("y:Q"),
        )
    ]


def _tree_x_domain(appearance: dict[str, Any]) -> list[float]:
    fraction = float(appearance.get("tree_horizontal_span_fraction", 1.0))
    half_span = 0.58 / fraction
    return [0.5 - half_span, 0.5 + half_span]


def _surface_vertex(x: float, y: float, scalar: float) -> tuple[float, float, float]:
    return float(x), float(y), float(scalar)


def _surface_feature(
    vertices: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    projector,
    *,
    color: str,
    opacity: float,
    shade: float = 1.0,
    feature_id: str,
) -> dict[str, Any] | None:
    projected = [projector(*vertex) for vertex in vertices]
    ring = [[float(point[0]), float(point[1])] for point in projected]
    ring.append(ring[0])
    area = abs(
        sum(
            ring[index][0] * ring[index + 1][1] - ring[index + 1][0] * ring[index][1]
            for index in range(len(ring) - 1)
        )
        / 2
    )
    if area < 1e-10:
        return None
    return {
        "feature_id": feature_id,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "depth": float(np.mean([point[2] for point in projected])),
        "color": _shade_color(color, shade),
        "opacity": opacity,
        "scalar_min": float(min(vertex[2] for vertex in vertices)),
        "scalar_max": float(max(vertex[2] for vertex in vertices)),
    }


def _lineage_color(
    x: float,
    y: float,
    scalar: float,
    threshold: float,
    saddle_value: float,
    appearance: dict[str, Any],
    basin_boundary: float,
) -> str:
    if threshold <= saddle_value and scalar <= saddle_value:
        return str(appearance["shared_region_color"])
    return str(
        appearance["right_peak_color"]
        if x + y > basin_boundary
        else appearance["left_peak_color"]
    )


def _surface_shade(
    vertices: tuple[tuple[float, float, float], ...], vertical_aspect: float
) -> float:
    points = np.asarray(vertices, dtype=float)
    points[:, 2] *= vertical_aspect
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    magnitude = float(np.linalg.norm(normal))
    if magnitude == 0.0:
        return 1.0
    normal /= magnitude
    light = np.asarray((-0.45, -0.35, 0.82), dtype=float)
    light /= np.linalg.norm(light)
    return float(np.clip(0.80 + 0.28 * np.dot(normal, light), 0.70, 1.08))


def _shade_color(color: str, factor: float) -> str:
    red = int(color[1:3], 16) / 255
    green = int(color[3:5], 16) / 255
    blue = int(color[5:7], 16) / 255
    hue, lightness, saturation = rgb_to_hls(red, green, blue)
    red, green, blue = hls_to_rgb(
        hue, min(1.0, max(0.0, lightness * factor)), saturation
    )
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def _projector(appearance: dict[str, Any]):
    elevation = radians(float(appearance["camera_elevation_degrees"]))
    azimuth = radians(float(appearance["camera_azimuth_degrees"]))
    vertical_aspect = float(appearance["vertical_box_aspect"])
    right = np.asarray((-sin(azimuth), cos(azimuth), 0.0))
    up = np.asarray(
        (-sin(elevation) * cos(azimuth), -sin(elevation) * sin(azimuth), cos(elevation))
    )
    toward_camera = np.asarray(
        (cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation))
    )

    def project(x: Any, y: Any, z: Any) -> tuple[Any, Any, Any]:
        points = np.stack(
            np.broadcast_arrays(
                # Rotated 180 degrees in the plane (x -> 1 - x, y -> 1 - y) so the
                # highest peak sits on the left, matching the tree row.
                (1.0 - np.asarray(x, dtype=float)) - 0.5,
                (1.0 - np.asarray(y, dtype=float)) - 0.5,
                (np.asarray(z, dtype=float) - 0.5) * vertical_aspect,
            ),
            axis=-1,
        )
        return (
            np.sum(points * right, axis=-1),
            np.sum(points * up, axis=-1),
            np.sum(points * toward_camera, axis=-1),
        )

    return project


def _projection_parameters(appearance: dict[str, Any]) -> dict[str, float]:
    project = _projector(appearance)
    corners = [
        project(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)
    ]
    xs = np.asarray([point[0] for point in corners])
    ys = np.asarray([point[1] for point in corners])
    width = float(appearance["panel_width"])
    height = float(appearance["panel_height"])
    padding = float(appearance["projection_padding_pixels"])
    scale = min(
        (width - 2 * padding) / float(xs.max() - xs.min()),
        (height - 2 * padding) / float(ys.max() - ys.min()),
    )
    return {
        "scale": scale,
        "translate_x": width / 2,
        "translate_y": height / 2,
    }


def _layout_records(
    construction: ManifoldGraphTreeConstruction,
) -> tuple[dict[int, dict[str, Any]], list[tuple[int, int]]]:
    nodes = {
        int(row["tree_node_index"]): row
        for row in construction.layout.node_table.iter_rows(named=True)
    }
    edges = [
        (int(row["source_tree_node_index"]), int(row["target_tree_node_index"]))
        for row in construction.layout.edge_table.select(
            "source_tree_node_index", "target_tree_node_index"
        ).iter_rows(named=True)
    ]
    return nodes, edges


def _trunk_ids(
    nodes: dict[int, dict[str, Any]],
    edges: list[tuple[int, int]],
    construction: ManifoldGraphTreeConstruction,
) -> set[str]:
    id_to_index = {row["vertex_id"]: index for index, row in nodes.items()}
    root = id_to_index[construction.critical_ids["global_minimum"]]
    saddle = id_to_index[construction.critical_ids["saddle"]]
    adjacency = {index: set() for index in nodes}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    queue = deque([(root, (root,))])
    visited = {root}
    while queue:
        current, path = queue.popleft()
        if current == saddle:
            return {str(nodes[index]["vertex_id"]) for index in path}
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            queue.append((neighbor, (*path, neighbor)))
    raise ValueError("public split-tree layout has no root-to-saddle path")


def _tree_sweep_records(
    construction: ManifoldGraphTreeConstruction,
    layout_nodes: dict[int, dict[str, Any]],
    layout_edges: list[tuple[int, int]],
    trunk_ids: set[str],
    active_ids: set[str],
    threshold: float,
    saddle_value: float,
    appearance: dict[str, Any],
    basin_boundary: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tree_nodes_by_id = {row["vertex_id"]: row for row in layout_nodes.values()}
    trunk_x = float(tree_nodes_by_id[construction.critical_ids["saddle"]]["x"])
    sample_by_id = {
        row["sample_id"]: row for row in construction.sample_table.iter_rows(named=True)
    }
    active_node_ids = active_ids
    tree_nodes: list[dict[str, Any]] = []
    for sample_id, sample in sample_by_id.items():
        layout_node = tree_nodes_by_id[sample_id]
        x = trunk_x if sample_id in trunk_ids else float(layout_node["x"])
        active = sample_id in active_node_ids
        if not active:
            continue
        event = str(sample["event"])
        shape = _EVENT_SHAPES.get(event) if active else None
        shape = shape or "circle"
        is_critical = sample_id in construction.critical_ids.values()
        size = int(
            appearance["tree_critical_node_size"]
            if active and is_critical
            else appearance["tree_regular_node_size"]
        )
        fill = (
            _lineage_color(
                sample["x"],
                sample["y"],
                sample["scalar"],
                threshold,
                saddle_value,
                appearance,
                basin_boundary,
            )
            if active
            else appearance["inactive_color"]
        )
        tree_nodes.append(
            {
                "sample_id": sample_id,
                "x": x,
                "y": float(sample["scalar"]),
                "scalar": float(sample["scalar"]),
                "event": event,
                "display_event": event,
                "shape": shape,
                "active": active,
                "is_critical": is_critical,
                # Draw order: regular vertices, then critical points, the global
                # minimum last so no stacked regular vertex hides it.
                "draw_rank": 2
                if sample_id == construction.critical_ids["global_minimum"]
                else int(is_critical),
                "size": size,
                "color": fill,
                "outline_color": _outline_color(fill, appearance),
                "opacity": float(
                    appearance["active_sample_opacity"]
                    if active
                    else appearance["inactive_sample_opacity"]
                ),
            }
        )

    edge_rows: list[dict[str, Any]] = []
    for edge_index, (first, second) in enumerate(layout_edges):
        first_row, second_row = layout_nodes[first], layout_nodes[second]
        if float(first_row["scalar"]) <= float(second_row["scalar"]):
            parent_index, child_index = first, second
        else:
            parent_index, child_index = second, first
        parent, child = layout_nodes[parent_index], layout_nodes[child_index]
        parent_id, child_id = parent["vertex_id"], child["vertex_id"]
        child_active = child_id in active_node_ids
        if not child_active:
            continue
        parent_active = parent_id in active_node_ids
        parent_on_trunk = parent_id in trunk_ids
        child_on_trunk = child_id in trunk_ids
        if not parent_active:
            points = [
                (float(child["x"]), float(child["scalar"])),
                (float(child["x"]), threshold),
            ]
        elif parent_on_trunk and child_on_trunk:
            points = [
                (float(parent["x"]), float(parent["scalar"])),
                (float(child["x"]), float(child["scalar"])),
            ]
        else:
            points = [
                (float(child["x"]), float(child["scalar"])),
                (float(child["x"]), float(parent["scalar"])),
                (float(parent["x"]), float(parent["scalar"])),
            ]
        if len(set(points)) < 2:
            continue
        child_sample = sample_by_id[child_id]
        is_shared = threshold <= saddle_value and (
            parent_id in trunk_ids and child_id in trunk_ids
        )
        edge_color = (
            str(appearance["shared_region_color"])
            if is_shared
            else _lineage_color(
                child_sample["x"],
                child_sample["y"],
                child_sample["scalar"],
                threshold,
                saddle_value,
                appearance,
                basin_boundary,
            )
        )
        for point_order, (x, y) in enumerate(points):
            edge_rows.append(
                {
                    "edge_id": f"tree-edge-{edge_index}",
                    "point_order": point_order,
                    "x": x,
                    "y": y,
                    "color": edge_color,
                    "opacity": float(appearance["active_sample_opacity"]),
                    "width": float(appearance["tree_active_edge_width"]),
                    "partial": not parent_active,
                    "child_sample_id": child_id,
                    "parent_sample_id": parent_id,
                }
            )
    return tree_nodes, edge_rows


def _appearance(settings: dict[str, Any]) -> dict[str, Any]:
    figures = settings["figures"]
    return {**figures.get("rendering", {}), **figures["figures"][FIGURE_ID]}
