"""Standalone join tree of one landscape graph.

The chart is the Full split-tree panel of main figure 5 (``full_tree_persistence``):
branch vertex-count colour, the wild-type star, and the same fitness axis
(``Fitness (WT = 0)``, outward-rounded bounds and padding from
``publication_scalar_scales``). A join tree is drawn root-up, so its leaves are
the valleys (local minima). The leading saddle band of the Full graph, the
closed interval between the two trailing saddles, is a horizontal band behind
the branches. Its bounds are the ``leading-saddle-band`` stage's
``band_lower`` and ``band_upper``; the printed endpoints are the manuscript
quantity registry's display text.

``build_join_tree_figure`` takes a graph key (``full``, ``published`` or
``candidate``). Only Full is loaded. Published would be the published-threshold
landscape and Candidate the configured BY functional class (the declaration in
``config/analysis.toml``, median-reflected cutoff, nominal q = 0.05). Neither
selection is built here.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import altair as alt
import polars as pl
from dubanevics2026topological.network_analysis import TreeAnalysis
from dubanevics2026topological.papkou2023 import (
    Papkou2023Workflow,
    build_papkou2023_backbone_analysis,
)

from ..analysis import dataset_input_path
from ._charts import (
    count_domain,
    overlay_count_colorbar,
    plot_data_with_metrics,
    publication_scalar_scales,
    tree_chart,
)
from .palette import BAND_GREY

FIGURE_ID = "full_join_tree"
# Later figures reuse this builder. Only ``full`` has a loader.
GRAPH_KEYS = ("full", "published", "candidate")
_COPY = {
    "full": {
        "title": "Full graph",
        "vertex_key": "full_graph.vertex_count",
        "valley_key": "full_graph.persistence.valley_count",
    },
    "published": {
        "title": "Published graph",
        "vertex_key": "published_graph.vertex_count",
        "valley_key": None,
    },
    "candidate": {
        "title": "Candidate graph",
        "vertex_key": "candidate_graph.vertex_count",
        "valley_key": None,
    },
}


def load_full_network(settings: dict[str, Any], context: Any | None = None) -> Any:
    """The Full-graph scalar network the join tree is swept on."""
    dataset = settings["analysis"]["dataset"]
    return Papkou2023Workflow(
        landscape=dataset["landscape"],
        fitness_wt=pl.read_parquet(dataset_input_path(dataset["fitness_wt"], context)),
        fitness_diff=pl.read_parquet(
            dataset_input_path(dataset["fitness_diff"], context)
        ),
    ).build_network()


def join_tree_analysis(network: Any) -> TreeAnalysis:
    """One reduced join tree and the tables ``tree_figure`` draws."""
    built = build_papkou2023_backbone_analysis(network, kind="join")
    _require_valley_leaves(built.tree)
    return TreeAnalysis(
        kind="join",
        scalar_attr="raw_fitness",
        tree=built.tree,
        plot_data=built.plot_data,
        event_table=built.event_table,
        branch_table=built.branch_table,
        branch_span_table=built.branch_span_table,
        persistence_table=built.persistence_table,
        branch_member_table=built.branch_member_table,
        critical_node_table=built.critical_node_table,
        stop_marker_table=built.stop_marker_table,
        all_stop_marker_table=built.all_stop_marker_table,
        wild_type_marker_table=built.wild_type_marker_table,
    )


def analysis_view(network: Any, tree: TreeAnalysis) -> Any:
    """The scalar domain and join tree a tree panel reads."""
    scalars = [float(payload[network.scalar_attr]) for payload in network.graph.nodes()]
    return SimpleNamespace(
        trees={"join": tree},
        scalar_domain=(min(scalars), max(scalars)),
        scalar_label="Raw fitness",
    )


def payload_for_graph(
    graph: str, inputs: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """Compose the join-tree payload for one graph key.

    Published and Candidate are accepted by the chart builder and refused here
    until a loader is wired. The band is always the Full graph's.
    """
    if graph not in GRAPH_KEYS:
        raise ValueError(
            f"graph must be one of {', '.join(GRAPH_KEYS)}, got {graph!r}."
        )
    if graph != "full":
        raise NotImplementedError(
            f"The {graph} join tree is not wired. The chart builder accepts it; "
            "only the Full graph is loaded."
        )
    network = inputs["full_network"]
    tree = join_tree_analysis(network)
    summary = inputs["leading_saddle_band"].summary
    lower = float(summary["band_lower"])
    upper = float(summary["band_upper"])
    if lower > upper:
        raise ValueError(
            "The leading saddle band is inverted: "
            f"band_lower {lower} is above band_upper {upper}."
        )
    return {
        "graph": graph,
        "analysis": analysis_view(network, tree),
        "tree": tree,
        "band_lower": lower,
        "band_upper": upper,
        "manuscript_quantities": inputs["manuscript_quantities"],
    }


def build_join_tree_figure(
    payload: dict[str, Any],
    settings: dict[str, Any],
    variant: str,
) -> Any:
    """Draw one join tree with the leading saddle band behind its branches."""
    if variant != "vanilla":
        raise ValueError("The join tree figure has one canonical vanilla export.")
    graph = payload["graph"]
    if graph not in _COPY:
        raise ValueError(
            f"graph must be one of {', '.join(GRAPH_KEYS)}, got {graph!r}."
        )
    copy = _COPY[graph]
    quantities = payload["manuscript_quantities"]
    figure_settings = settings["figures"]["figures"][FIGURE_ID]
    rendering = settings["figures"].get("rendering", {})
    tree_analysis = payload["tree"]
    analysis = payload["analysis"]
    lower = float(payload["band_lower"])
    upper = float(payload["band_upper"])
    if lower > upper:
        raise ValueError(
            "The leading saddle band is inverted: "
            f"band_lower {lower} is above band_upper {upper}."
        )
    merged = {**rendering, **figure_settings}
    plot_data = plot_data_with_metrics(_upper_targets(tree_analysis))
    domain = count_domain(plot_data.edge_table)
    tree = tree_chart(
        plot_data,
        tree_analysis,
        analysis,
        figure_settings,
        merged,
        domain,
        linked=False,
        annotated=False,
        underlays=[
            _band_layer(
                lower,
                upper,
                int(
                    figure_settings.get("tree", {}).get(
                        "width", rendering.get("tree_width", 420)
                    )
                ),
                figure_settings,
            )
        ],
    )
    tree = overlay_count_colorbar(tree, figure_settings, merged, domain)
    valley = (
        ""
        if copy["valley_key"] is None
        else (f" · {quantities.display(copy['valley_key'])} valleys")
    )
    chart = tree.properties(
        title=alt.TitleParams(
            text=copy["title"],
            subtitle=[
                "Papkou et al. (2023) · valleys are the leaves",
                f"{quantities.display(copy['vertex_key'])} sequences{valley}",
                (
                    "Leading saddle band "
                    f"{quantities.display('leading_saddle_band.band.lower')} to "
                    f"{quantities.display('leading_saddle_band.band.upper')}"
                ),
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
    return publication_scalar_scales(chart, analysis.scalar_domain, rendering)


def _band_layer(
    lower: float, upper: float, width: int, figure_settings: dict[str, Any]
) -> Any:
    """A horizontal band on the fitness axis, behind the branches.

    The x extent is the panel width in pixels. Sharing the branch x scale with
    this rect makes Vega-Lite drop the axis (a scale with no domain), so the
    band is placed in the panel instead.
    """
    return (
        alt.Chart(pl.DataFrame({"y": [lower], "y_end": [upper]}))
        .mark_rect(
            color=figure_settings.get("band_color", BAND_GREY),
            opacity=float(figure_settings.get("band_opacity", 0.35)),
            clip=True,
        )
        .encode(
            y=alt.Y("y:Q"),
            y2="y_end:Q",
            x=alt.value(0),
            x2=alt.value(width),
        )
    )


def _upper_targets(tree_analysis: Any) -> Any:
    """Point each displayed edge at its upper endpoint, as the split-tree panel does.

    A join tree's plot edges run from the higher endpoint down to the valley.
    ``plot_data_with_metrics`` requires the target to be the upper end. Swapping
    the two endpoints leaves the drawn segment in place.
    """
    plot_data = tree_analysis.plot_data
    return replace(
        tree_analysis,
        plot_data=replace(
            plot_data,
            edge_table=_swap_downward(plot_data.edge_table),
            edge_segment_table=_swap_downward(plot_data.edge_segment_table),
        ),
    )


def _swap_downward(table: pl.DataFrame) -> pl.DataFrame:
    pairs = (
        ("source_tree_node_index", "target_tree_node_index"),
        ("source_vertex_index", "target_vertex_index"),
        ("x_source", "x_target"),
        ("y_source", "y_target"),
        ("source_scalar", "target_scalar"),
    )
    present = [(left, right) for left, right in pairs if left in table.columns]
    if "target_scalar" not in table.columns:
        return table
    downward = pl.col("target_scalar") < pl.col("source_scalar")
    swapped = table
    for left, right in present:
        swapped = swapped.with_columns(
            pl.when(downward).then(pl.col(right)).otherwise(pl.col(left)).alias(left),
            pl.when(downward).then(pl.col(left)).otherwise(pl.col(right)).alias(right),
        )
    return swapped


def _require_valley_leaves(tree: Any) -> None:
    """Every leaf of a leaf-to-root join tree is a valley (a minimum)."""
    for index in tree.node_indices():
        if tree.in_degree(index) != 0 or tree.out_degree(index) == 0:
            continue
        event = str(dict(tree.get_node_data(index)).get("event", ""))
        if event != "minimum":
            raise ValueError(
                f"A join-tree leaf is not a valley: node {index} has event {event!r}."
            )
