# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""DrWatson-inspired workflow core and data caching engine.

This module provides the core abstractions for 2-stage computational workflows
following the DrWatson.jl reproducibility pattern (Datseris et al., 2020,
'DrWatson: the perfect sidekick for your scientific inquiries', JOSS 5(54), 2673,
https://doi.org/10.21105/joss.02673).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import polars as pl
import rustworkx as rx

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

from topographer.graph import ScalarNetwork
from topographer.plotting import TreePlotData

if TYPE_CHECKING:
    pass

__all__ = [
    "WORKFLOW_SCHEMA_VERSION",
    "NetworkAnalysis",
    "NetworkWorkflow",
    "TreeAnalysis",
    "load_network_analysis",
    "produce_or_load_analysis",
    "save_network_analysis",
    "savename",
]

WORKFLOW_SCHEMA_VERSION = "1.0"

TreeKind = Literal["join", "split"]


def _dump_toml_val(v: Any) -> str:
    """Format a single python value as TOML value text."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(v, (list, tuple)):
        items = ", ".join(_dump_toml_val(x) for x in v)
        return f"[{items}]"
    raise TypeError(f"Unsupported TOML value type: {type(v).__name__}")


def _dump_toml(d: dict[str, Any]) -> str:
    """Serialize a simple nested dict into valid TOML format."""
    lines: list[str] = []
    subtables: dict[str, dict[str, Any]] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            subtables[k] = v
        else:
            lines.append(f"{k} = {_dump_toml_val(v)}")
    for subk, subv in subtables.items():
        lines.append(f"\n[{subk}]")
        for k, v in subv.items():
            lines.append(f"{k} = {_dump_toml_val(v)}")
    return "\n".join(lines) + "\n"


def savename(
    prefix: str | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    connector: str = "_",
    equals: str = "=",
    val_to_string: Callable[[Any], str] | None = None,
    sort: bool = True,
) -> str:
    """Deterministically map a case prefix and configuration dict to a string identifier.

    Parameters
    ----------
    prefix : str | None
        Optional string prefix (e.g., "cube", "papkou2023").
    config : Mapping[str, Any] | None
        Parameter dictionary of workflow settings.
    connector : str
        String connector between key-value pairs (default "_").
    equals : str
        String separator between key and value (default "=").
    val_to_string : Callable[[Any], str] | None
        Optional custom function to format parameter values to string.
    sort : bool
        Whether to sort parameter keys alphabetically (default True).

    Returns
    -------
    str
        Deterministic string identifier.
    """
    prefix_str = str(prefix).strip() if prefix is not None else ""
    if not config:
        return prefix_str

    items = list(config.items())
    if sort:
        items.sort(key=lambda item: item[0])

    pairs: list[str] = []
    for k, v in items:
        if val_to_string is not None:
            formatted_val = val_to_string(v)
        elif isinstance(v, bool):
            formatted_val = "true" if v else "false"
        else:
            formatted_val = str(v)
        pairs.append(f"{k}{equals}{formatted_val}")

    joined_pairs = connector.join(pairs)
    if prefix_str:
        return f"{prefix_str}{connector}{joined_pairs}"
    return joined_pairs


def _default_root_kind(kind: TreeKind) -> Literal["maximum", "minimum"]:
    """Return the global extremum that roots a tree of this kind.

    A join tree is rooted at the global maximum and a split tree at the global
    minimum. This is the fallback for hand-edited metadata that lacks ``root_kind``;
    ``save_network_analysis`` always writes the field.
    """

    return "maximum" if kind == "join" else "minimum"


@dataclass(slots=True)
class TreeAnalysis:
    """Analytical products for one tree kind (join or split)."""

    kind: TreeKind
    scalar_attr: str
    tree: rx.PyDiGraph
    plot_data: TreePlotData
    event_table: pl.DataFrame
    branch_table: pl.DataFrame
    branch_span_table: pl.DataFrame
    persistence_table: pl.DataFrame
    branch_member_table: pl.DataFrame | None = None
    critical_node_table: pl.DataFrame | None = None
    marker_table: pl.DataFrame | None = None
    stop_marker_table: pl.DataFrame | None = None
    all_stop_marker_table: pl.DataFrame | None = None
    wild_type_marker_table: pl.DataFrame | None = None
    simplified_tree: rx.PyDiGraph | None = None
    simplified_plot_data: TreePlotData | None = None
    qualifying_pairs: pl.DataFrame | None = None
    persistence_threshold: float | None = None


@dataclass(slots=True)
class NetworkAnalysis:
    """Complete analytical snapshot of a network workflow execution."""

    name: str
    config: dict[str, Any]
    scalar_attr: str
    scalar_domain: tuple[float, float]
    scalar_label: str
    input_summary: pl.DataFrame
    trees: dict[TreeKind, TreeAnalysis] = field(default_factory=dict)
    vertex_table: pl.DataFrame | None = None
    edge_table: pl.DataFrame | None = None
    network_persistence_table: pl.DataFrame | None = None
    cycle_representatives: pl.DataFrame | None = None



def save_network_analysis(analysis: NetworkAnalysis, data_dir: str | Path) -> Path:
    """Save NetworkAnalysis checkpoint tables and metadata.toml to data_dir."""
    dest = Path(data_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    analysis.input_summary.write_parquet(dest / "input-summary.parquet")

    if analysis.vertex_table is not None:
        analysis.vertex_table.write_parquet(dest / "vertices.parquet")
    if analysis.edge_table is not None:
        analysis.edge_table.write_parquet(dest / "edges.parquet")
    if analysis.network_persistence_table is not None:
        analysis.network_persistence_table.write_parquet(dest / "network-persistence.parquet")
    if analysis.cycle_representatives is not None:
        analysis.cycle_representatives.write_parquet(dest / "cycle-representatives.parquet")

    tree_kinds = sorted(analysis.trees.keys())
    trees_meta: dict[str, dict[str, Any]] = {}

    for kind, tree_analysis in analysis.trees.items():
        rx.digraph_write_graphml(tree_analysis.tree, str(dest / f"{kind}-tree-backbone.graphml"))

        tables: dict[str, pl.DataFrame] = {
            f"{kind}-node-table.parquet": tree_analysis.plot_data.node_table,
            f"{kind}-edge-table.parquet": tree_analysis.plot_data.edge_table,
            f"{kind}-edge-segment-table.parquet": tree_analysis.plot_data.edge_segment_table,
            f"{kind}-tree-events.parquet": tree_analysis.event_table,
            f"{kind}-tree-branches.parquet": tree_analysis.branch_table,
            f"{kind}-branch-span.parquet": tree_analysis.branch_span_table,
            f"{kind}-persistence.parquet": tree_analysis.persistence_table,
        }

        if tree_analysis.branch_member_table is not None:
            tables[f"{kind}-branch-members.parquet"] = tree_analysis.branch_member_table
        if tree_analysis.critical_node_table is not None:
            tables[f"{kind}-critical-nodes.parquet"] = tree_analysis.critical_node_table
        if tree_analysis.marker_table is not None:
            tables[f"{kind}-markers.parquet"] = tree_analysis.marker_table
        if tree_analysis.stop_marker_table is not None:
            tables[f"{kind}-stop-markers.parquet"] = tree_analysis.stop_marker_table
        if tree_analysis.all_stop_marker_table is not None:
            tables[f"{kind}-all-stop-markers.parquet"] = tree_analysis.all_stop_marker_table
        if tree_analysis.wild_type_marker_table is not None:
            tables[f"{kind}-wild-type-markers.parquet"] = tree_analysis.wild_type_marker_table
        if tree_analysis.qualifying_pairs is not None:
            tables[f"{kind}-qualifying-pairs.parquet"] = tree_analysis.qualifying_pairs

        if tree_analysis.simplified_tree is not None:
            rx.digraph_write_graphml(
                tree_analysis.simplified_tree, str(dest / f"{kind}-simplified-tree.graphml")
            )
        if tree_analysis.simplified_plot_data is not None:
            tables[f"{kind}-simplified-node-table.parquet"] = (
                tree_analysis.simplified_plot_data.node_table
            )
            tables[f"{kind}-simplified-edge-table.parquet"] = (
                tree_analysis.simplified_plot_data.edge_table
            )
            tables[f"{kind}-simplified-edge-segment-table.parquet"] = (
                tree_analysis.simplified_plot_data.edge_segment_table
            )

        for filename, df in tables.items():
            df.write_parquet(dest / filename)

        trees_meta[kind] = {
            "root_kind": tree_analysis.plot_data.root_kind,
            "root_node_index": tree_analysis.plot_data.root_node_index,
        }
        if tree_analysis.persistence_threshold is not None:
            trees_meta[kind]["persistence_threshold"] = tree_analysis.persistence_threshold

    meta_dict: dict[str, Any] = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_name": analysis.name,
        "scalar_attr": analysis.scalar_attr,
        "scalar_label": analysis.scalar_label,
        "scalar_domain": list(analysis.scalar_domain),
        "tree_kinds": tree_kinds,
        "config": analysis.config,
    }
    for kind, tmeta in trees_meta.items():
        meta_dict[f"trees_{kind}"] = tmeta

    (dest / "metadata.toml").write_text(_dump_toml(meta_dict), encoding="utf-8")
    return dest


def load_network_analysis(data_dir: str | Path) -> NetworkAnalysis:
    """Load NetworkAnalysis checkpoint from data_dir."""
    dest = Path(data_dir).resolve()
    if not dest.is_dir():
        raise ValueError(f"Analysis directory does not exist: {dest}")

    metadata_path = dest / "metadata.toml"
    if not metadata_path.exists():
        raise ValueError(f"Invalid analysis checkpoint in {dest}: missing metadata.toml")

    try:
        with metadata_path.open("rb") as f:
            metadata = tomllib.load(f)
    except Exception as err:
        raise ValueError(f"Failed to parse metadata in {metadata_path}: {err}") from err

    version = metadata.get("schema_version")
    if version != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema version {version!r} in {metadata_path}. "
            f"Expected {WORKFLOW_SCHEMA_VERSION!r}."
        )

    required_keys = {
        "schema_version",
        "workflow_name",
        "scalar_attr",
        "scalar_label",
        "scalar_domain",
        "tree_kinds",
        "config",
    }
    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError(
            f"Incomplete snapshot metadata in {metadata_path}: missing {sorted(missing_keys)}"
        )

    input_summary_path = dest / "input-summary.parquet"
    if not input_summary_path.exists():
        raise ValueError(f"Incomplete snapshot in {dest}: missing required input-summary.parquet")
    input_summary = pl.read_parquet(input_summary_path)

    trees: dict[TreeKind, TreeAnalysis] = {}
    for kind in metadata["tree_kinds"]:
        graph_path = dest / f"{kind}-tree-backbone.graphml"
        if not graph_path.exists():
            raise ValueError(f"Incomplete snapshot in {dest}: missing {graph_path.name}")

        try:
            loaded = rx.read_graphml(str(graph_path))
            if not loaded:
                raise ValueError(f"Empty tree graph in {graph_path}")
            tree_graph = loaded[0]
            for idx in tree_graph.node_indices():
                d = dict(tree_graph.get_node_data(idx))
                if "id" in d:
                    d.pop("id")
                    tree_graph[idx] = d
        except Exception as err:
            raise ValueError(f"Failed to load tree graph from {graph_path}: {err}") from err

        required_tables = [
            f"{kind}-node-table.parquet",
            f"{kind}-edge-table.parquet",
            f"{kind}-edge-segment-table.parquet",
            f"{kind}-tree-events.parquet",
            f"{kind}-tree-branches.parquet",
            f"{kind}-branch-span.parquet",
            f"{kind}-persistence.parquet",
        ]
        for tbl in required_tables:
            if not (dest / tbl).exists():
                raise ValueError(f"Incomplete snapshot in {dest}: missing required table {tbl}")

        tmeta = metadata.get(f"trees_{kind}", {})
        # The root_node_index default of 0 is as arbitrary as a missing root_kind; both
        # defaults apply only to hand-edited metadata.
        plot_data = TreePlotData(
            scalar_attr=metadata["scalar_attr"],
            root_kind=tmeta.get("root_kind", _default_root_kind(kind)),
            root_node_index=tmeta.get("root_node_index", 0),
            node_table=pl.read_parquet(dest / f"{kind}-node-table.parquet"),
            edge_table=pl.read_parquet(dest / f"{kind}-edge-table.parquet"),
            edge_segment_table=pl.read_parquet(dest / f"{kind}-edge-segment-table.parquet"),
        )

        tree_analysis = TreeAnalysis(
            kind=kind,
            scalar_attr=metadata["scalar_attr"],
            tree=tree_graph,
            plot_data=plot_data,
            event_table=pl.read_parquet(dest / f"{kind}-tree-events.parquet"),
            branch_table=pl.read_parquet(dest / f"{kind}-tree-branches.parquet"),
            branch_span_table=pl.read_parquet(dest / f"{kind}-branch-span.parquet"),
            persistence_table=pl.read_parquet(dest / f"{kind}-persistence.parquet"),
        )

        # Read optional tables
        for field_name, tbl_name in [
            ("branch_member_table", f"{kind}-branch-members.parquet"),
            ("critical_node_table", f"{kind}-critical-nodes.parquet"),
            ("marker_table", f"{kind}-markers.parquet"),
            ("stop_marker_table", f"{kind}-stop-markers.parquet"),
            ("all_stop_marker_table", f"{kind}-all-stop-markers.parquet"),
            ("wild_type_marker_table", f"{kind}-wild-type-markers.parquet"),
            ("qualifying_pairs", f"{kind}-qualifying-pairs.parquet"),
        ]:
            if (dest / tbl_name).exists():
                setattr(tree_analysis, field_name, pl.read_parquet(dest / tbl_name))

        if (dest / f"{kind}-simplified-tree.graphml").exists():
            sim_loaded = rx.read_graphml(str(dest / f"{kind}-simplified-tree.graphml"))
            if sim_loaded:
                sim_graph = sim_loaded[0]
                for idx in sim_graph.node_indices():
                    d = dict(sim_graph.get_node_data(idx))
                    if "id" in d:
                        d.pop("id")
                        sim_graph[idx] = d
                tree_analysis.simplified_tree = sim_graph

        if (dest / f"{kind}-simplified-node-table.parquet").exists():
            tree_analysis.simplified_plot_data = TreePlotData(
                scalar_attr=metadata["scalar_attr"],
                root_kind=tmeta.get("root_kind", _default_root_kind(kind)),
                root_node_index=tmeta.get("root_node_index", 0),
                node_table=pl.read_parquet(dest / f"{kind}-simplified-node-table.parquet"),
                edge_table=pl.read_parquet(dest / f"{kind}-simplified-edge-table.parquet"),
                edge_segment_table=pl.read_parquet(
                    dest / f"{kind}-simplified-edge-segment-table.parquet"
                ),
            )

        if "persistence_threshold" in tmeta:
            tree_analysis.persistence_threshold = tmeta["persistence_threshold"]

        trees[kind] = tree_analysis

    vertex_table = (
        pl.read_parquet(dest / "vertices.parquet")
        if (dest / "vertices.parquet").exists()
        else None
    )
    edge_table = (
        pl.read_parquet(dest / "edges.parquet") if (dest / "edges.parquet").exists() else None
    )
    net_pers = (
        pl.read_parquet(dest / "network-persistence.parquet")
        if (dest / "network-persistence.parquet").exists()
        else None
    )
    cycle_reps = (
        pl.read_parquet(dest / "cycle-representatives.parquet")
        if (dest / "cycle-representatives.parquet").exists()
        else None
    )

    return NetworkAnalysis(
        name=metadata["workflow_name"],
        config=metadata["config"],
        scalar_attr=metadata["scalar_attr"],
        scalar_domain=tuple(metadata["scalar_domain"]),
        scalar_label=metadata["scalar_label"],
        input_summary=input_summary,
        trees=trees,
        vertex_table=vertex_table,
        edge_table=edge_table,
        network_persistence_table=net_pers,
        cycle_representatives=cycle_reps,
    )


def produce_or_load_analysis(
    workflow: NetworkWorkflow,
    output_dir: str | Path,
    *,
    tree_kind: Literal["split", "join", "both"] = "both",
    force: bool = False,
) -> NetworkAnalysis:
    """Produce or reload NetworkAnalysis checkpoint from output_dir / "data".

    If force is False and output_dir / "data" / "metadata.toml" exists and is valid,
    reloads NetworkAnalysis without invoking workflow.build_network() or workflow.analyze().
    Otherwise, executes workflow.analyze(tree_kind=tree_kind) and writes checkpoint.
    """
    out_path = Path(output_dir).resolve()
    data_dir = out_path / "data"

    if not force and (data_dir / "metadata.toml").exists():
        return load_network_analysis(data_dir)

    analysis = workflow.analyze(tree_kind=tree_kind)
    save_network_analysis(analysis, data_dir)
    return analysis


class NetworkWorkflow(ABC):
    """Base class for DrWatson-inspired network workflows."""

    name: str = "workflow"

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Return parameter config dict for this workflow instance."""
        ...

    def savename(self) -> str:
        """Construct savename identifier for this workflow configuration."""
        return savename(self.name, self.config)

    @abstractmethod
    def build_network(self) -> ScalarNetwork:
        """Build or load the input ScalarNetwork."""
        ...

    @abstractmethod
    def analyze(
        self,
        *,
        tree_kind: Literal["split", "join", "both"] = "both",
    ) -> NetworkAnalysis:
        """Run analysis on the network."""
        ...

    def produce_or_load_analysis(
        self,
        output_dir: str | Path,
        *,
        tree_kind: Literal["split", "join", "both"] = "both",
        force: bool = False,
    ) -> NetworkAnalysis:
        """Produce or reload NetworkAnalysis checkpoint from output_dir/data."""
        return produce_or_load_analysis(
            self,
            output_dir,
            tree_kind=tree_kind,
            force=force,
        )
