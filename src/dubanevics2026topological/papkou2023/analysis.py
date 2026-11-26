# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""Tree analysis and snapshot serialization for Papkou et al. (2023)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Literal, Mapping

import polars as pl
import polars.selectors as cs
import rustworkx as rx

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

from topographer.biology import (
    annotate_stop_codons,
    build_tree_lineage_metrics,
    parse_vertex_chain_json,
)
from topographer.graph import ScalarNetwork
from topographer.plotting import (
    TreePlotData,
    build_tree_plot_data,
    compute_branch_local_swarm,
    select_tree_root_node_index,
)
from topographer.topology import (
    MergeTree,
    ScalarNetworkTopology,
    build_tree_location_tables,
)

STOP_CODON_SYMBOLS = frozenset({"*", "%", "^", "&"})
WILD_TYPE_DNA_SEQUENCE = "GCCGATCTC"
WILD_TYPE_AA_SEQUENCE = "ADL"

# Progress phases reported by ``build_papkou2023_backbone_analysis``.
_PROGRESS_KEYS = frozenset({"tree_sweep", "tree_plot", "branch_metrics", "branch_spans"})
# Sweep node keys kept on the backbone (mirrors _make_tree_node_payload);
# MergeTree's vertex_id/label are dropped.
_BACKBONE_NODE_KEYS = ("vertex_index", "scalar", "event")


@dataclass(frozen=True, slots=True)
class Papkou2023TreeAnalysis:
    """Backbone tree, plot data, branch spans, metrics, and extended persistence."""

    kind: Literal["join", "split"]
    tree: rx.PyDiGraph
    event_table: pl.DataFrame
    plot_data: TreePlotData
    branch_table: pl.DataFrame
    branch_span_table: pl.DataFrame
    branch_member_table: pl.DataFrame
    critical_node_table: pl.DataFrame
    stop_marker_table: pl.DataFrame
    all_stop_marker_table: pl.DataFrame
    wild_type_marker_table: pl.DataFrame
    persistence_table: pl.DataFrame

    def save(self, snapshot_dir: str | Path) -> Path:
        """Save this analysis snapshot to disk."""
        return save_papkou2023_analysis_snapshot(self, snapshot_dir)

    @classmethod
    def load(cls, snapshot_dir: str | Path) -> Papkou2023TreeAnalysis:
        """Load an analysis snapshot from disk."""
        return load_papkou2023_analysis_snapshot(snapshot_dir)


def build_papkou2023_backbone_analysis(
    network: ScalarNetwork,
    *,
    kind: Literal["join", "split"],
    progress_callbacks: Mapping[str, Callable[[int, int], None]] | None = None,
) -> Papkou2023TreeAnalysis:
    """Build one reduced tree and its biological plot/publication branch tables.

    The tree kind is swept and reduced once, through the public
    :class:`~topographer.topology.MergeTree`. The sequence-annotated, leaf-to-root backbone
    that the plots and tables consume is derived from that merge tree.

    ``progress_callbacks`` maps phase names to ``callback(completed, total)`` functions.
    The accepted keys are ``"tree_sweep"``, ``"tree_plot"``, ``"branch_metrics"`` and
    ``"branch_spans"``; any other key raises :class:`ValueError`. ``"tree_sweep"`` is a
    two-point phase, ``(0, 1)`` then ``(1, 1)``, around the whole public merge-tree
    construction (sweep, reduction, orientation and location tables), not only the sweep.
    """
    if kind not in {"join", "split"}:
        raise ValueError("kind must be 'join' or 'split'.")
    unknown = set(progress_callbacks or ()) - _PROGRESS_KEYS
    if unknown:
        raise ValueError(
            f"Unknown progress callback keys {sorted(unknown)}; expected {sorted(_PROGRESS_KEYS)}."
        )
    node_attrs = ("dna_sequence", "aa_sequence", "has_stop_codon")
    sweep_progress = _progress(progress_callbacks, "tree_sweep")
    _report_progress(sweep_progress, 0, 1)
    merge_tree = _compute_merge_tree(network, kind)
    _report_progress(sweep_progress, 1, 1)
    tree = _sequence_backbone_tree(network, merge_tree, node_attrs=node_attrs)
    root_kind = "maximum" if kind == "join" else "minimum"
    plot_data = build_tree_plot_data(
        tree,
        root_kind=root_kind,
        node_attrs=node_attrs,
        horizontal_layout="subtree_extremum",
        progress_callback=_progress(progress_callbacks, "tree_plot"),
    )
    location_tables = build_tree_location_tables(tree)
    branch_member_table = location_tables.branch_member_table
    critical_node_table = location_tables.critical_node_table
    if (
        "has_stop_codon" not in branch_member_table.columns
        and "aa_sequence" in branch_member_table.columns
    ):
        branch_member_table = branch_member_table.pipe(annotate_stop_codons)
    elif "has_stop_codon" not in branch_member_table.columns:
        branch_member_table = branch_member_table.with_columns(
            pl.lit(False).alias("has_stop_codon")
        )
    if (
        "has_stop_codon" not in critical_node_table.columns
        and "aa_sequence" in critical_node_table.columns
    ):
        critical_node_table = critical_node_table.pipe(annotate_stop_codons)
    elif "has_stop_codon" not in critical_node_table.columns:
        critical_node_table = critical_node_table.with_columns(
            pl.lit(False).alias("has_stop_codon")
        )
    lineage_metrics = build_tree_lineage_metrics(
        network,
        merge_tree,
        progress_callback=_progress(progress_callbacks, "branch_metrics"),
    )
    # Object-dtype public-id columns cannot be written to Parquet.
    branch_table = lineage_metrics.branch_table.drop(cs.by_dtype(pl.Object))
    stops = _build_stop_marker_table(tree, plot_data)
    all_stops = _build_all_stop_marker_table(branch_member_table, stops, plot_data)
    wild_type_marker = _build_wild_type_marker_table(tree, plot_data)
    persistence = lineage_metrics.persistence_table.drop(cs.by_dtype(pl.Object))
    branch_spans = _build_branch_span_table(
        tree,
        branch_table,
        kind=kind,
        progress_callback=_progress(progress_callbacks, "branch_spans"),
    )
    return Papkou2023TreeAnalysis(
        kind=kind,
        tree=tree,
        event_table=merge_tree.events,
        plot_data=plot_data,
        branch_table=branch_table,
        branch_span_table=branch_spans,
        branch_member_table=branch_member_table,
        critical_node_table=critical_node_table,
        stop_marker_table=stops,
        all_stop_marker_table=all_stops,
        wild_type_marker_table=wild_type_marker,
        persistence_table=persistence,
    )


def _compute_merge_tree(network: ScalarNetwork, kind: Literal["join", "split"]) -> MergeTree:
    topology = ScalarNetworkTopology(network)
    return topology.join_tree() if kind == "join" else topology.split_tree()


def _sequence_backbone_tree(
    network: ScalarNetwork,
    merge_tree: MergeTree,
    *,
    node_attrs: tuple[str, ...],
) -> rx.PyDiGraph:
    """Return the public merge tree pointing leaf to root, annotated with network attributes.

    Node indices and edge payloads are those of ``merge_tree.graph``. Edges are reversed and
    added in child-index order. Node and vertex-chain payloads keep ``vertex_index``,
    ``scalar`` and ``event`` plus each requested attribute present on the network vertex, so
    keys the merge tree adds for itself (``vertex_id``, ``label``) do not reach the tables.
    """
    source_graph = network.graph

    def annotate(payload: Mapping[str, Any]) -> dict[str, Any]:
        annotated = {key: payload[key] for key in _BACKBONE_NODE_KEYS}
        source = source_graph.get_node_data(int(payload["vertex_index"]))
        annotated.update({attr: source[attr] for attr in node_attrs if attr in source})
        return annotated

    oriented = merge_tree.graph
    backbone = rx.PyDiGraph()
    for index in oriented.node_indices():
        backbone.add_node(annotate(oriented.get_node_data(index)))
    parent_edges = sorted(
        (
            (int(child), int(parent), dict(payload))
            for parent, child, payload in oriented.weighted_edge_list()
        ),
        key=lambda edge: edge[0],
    )
    for child, parent, payload in parent_edges:
        if "vertex_chain_json" in payload:
            chain = parse_vertex_chain_json(payload["vertex_chain_json"])
            payload["vertex_chain_json"] = json.dumps(
                [annotate(entry) for entry in chain], sort_keys=True
            )
        backbone.add_edge(child, parent, payload)
    return backbone


def _build_branch_span_table(
    tree: rx.PyDiGraph,
    branch_table: pl.DataFrame,
    *,
    kind: Literal["join", "split"],
    progress_callback: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    saddle_event = "join_saddle" if kind == "join" else "split_saddle"
    extremum_event = "minimum" if kind == "join" else "maximum"
    root_event = "maximum" if kind == "join" else "minimum"
    allowed_source_events = {extremum_event, saddle_event}
    branch_metrics = {
        (int(row["source_tree_node_index"]), int(row["target_tree_node_index"])): row
        for row in branch_table.to_dicts()
    }
    metric_names = tuple(
        name
        for name in branch_table.columns
        if name not in {"source_tree_node_index", "target_tree_node_index"}
    )
    rows: list[dict[str, Any]] = []
    edges = list(tree.edge_list())
    _report_progress(progress_callback, 0, len(edges))
    for position, (source, target) in enumerate(edges, start=1):
        source_data = dict(tree.get_node_data(source))
        target_data = dict(tree.get_node_data(target))
        source_event = str(source_data.get("event", "regular"))
        target_event = str(target_data.get("event", "regular"))
        is_terminal_minimum = (
            kind == "split" and source_event == saddle_event and tree.out_degree(target) == 0
        )
        if (
            source_event not in allowed_source_events or target_event != saddle_event
        ) and not is_terminal_minimum:
            _report_progress(progress_callback, position, len(edges))
            continue
        source_scalar = float(source_data["scalar"])
        target_scalar = float(target_data["scalar"])
        displayed_target_event = root_event if is_terminal_minimum else target_event
        metrics = branch_metrics[(int(source), int(target))]
        rows.append(
            {
                "source_tree_node_index": int(source),
                "target_tree_node_index": int(target),
                "source_vertex_index": int(source_data.get("vertex_index", source)),
                "target_vertex_index": int(target_data.get("vertex_index", target)),
                "source_label": str(source_data.get("dna_sequence", source)),
                "target_label": str(target_data.get("dna_sequence", target)),
                "source_dna_sequence": str(source_data.get("dna_sequence", "")),
                "target_dna_sequence": str(target_data.get("dna_sequence", "")),
                "source_aa_sequence": str(source_data.get("aa_sequence", "")),
                "target_aa_sequence": str(target_data.get("aa_sequence", "")),
                "source_scalar": source_scalar,
                "target_scalar": target_scalar,
                "branch_span": abs(source_scalar - target_scalar),
                "source_event": source_event,
                "target_event": displayed_target_event,
                "is_global_extremum_self_point": False,
                **{
                    name: value
                    for name, value in metrics.items()
                    if name not in {"source_tree_node_index", "target_tree_node_index"}
                },
            }
        )
        _report_progress(progress_callback, position, len(edges))
    root_index = select_tree_root_node_index(
        tree,
        scalar_attr="scalar",
        root_kind="maximum" if kind == "join" else "minimum",
    )
    root_data = dict(tree.get_node_data(root_index))
    root_scalar = float(root_data["scalar"])
    rows.append(
        {
            "source_tree_node_index": root_index,
            "target_tree_node_index": root_index,
            "source_vertex_index": int(root_data.get("vertex_index", root_index)),
            "target_vertex_index": int(root_data.get("vertex_index", root_index)),
            "source_label": str(root_data.get("dna_sequence", root_index)),
            "target_label": str(root_data.get("dna_sequence", root_index)),
            "source_dna_sequence": str(root_data.get("dna_sequence", "")),
            "target_dna_sequence": str(root_data.get("dna_sequence", "")),
            "source_aa_sequence": str(root_data.get("aa_sequence", "")),
            "target_aa_sequence": str(root_data.get("aa_sequence", "")),
            "source_scalar": root_scalar,
            "target_scalar": root_scalar,
            "branch_span": 0.0,
            "source_event": root_event,
            "target_event": root_event,
            "is_global_extremum_self_point": True,
            **{name: None for name in metric_names},
        }
    )
    return pl.DataFrame(rows).sort(
        ["branch_span", "source_scalar", "source_vertex_index"],
        descending=[True, True, False],
    )


def _build_stop_marker_table(tree: rx.PyDiGraph, plot_data: TreePlotData) -> pl.DataFrame:
    coordinates = {int(row["tree_node_index"]): row for row in plot_data.node_table.to_dicts()}
    markers: list[dict[str, Any]] = []
    for tree_node_index in tree.node_indices():
        node = dict(tree.get_node_data(tree_node_index))
        aa_sequence = node.get("aa_sequence")
        if not _has_stop_codon(aa_sequence):
            continue
        coordinate = coordinates[int(tree_node_index)]
        markers.append(
            {
                "tree_node_index": int(tree_node_index),
                "vertex_index": int(node.get("vertex_index", tree_node_index)),
                "scalar": float(node["scalar"]),
                "event": str(node.get("event", "regular")),
                "dna_sequence": str(node.get("dna_sequence", "")),
                "aa_sequence": str(aa_sequence),
                "x": float(coordinate["x"]),
                "y": float(coordinate["y"]),
            }
        )
    schema = {
        "tree_node_index": pl.Int64,
        "vertex_index": pl.Int64,
        "scalar": pl.Float64,
        "event": pl.String,
        "dna_sequence": pl.String,
        "aa_sequence": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
    }
    return pl.DataFrame(markers, schema=schema).sort("tree_node_index")


def _build_all_stop_marker_table(
    branch_member_table: pl.DataFrame,
    critical_stops: pl.DataFrame,
    plot_data: TreePlotData,
) -> pl.DataFrame:
    swarm = compute_branch_local_swarm(
        branch_member_table,
        plot_data,
        critical_table=critical_stops,
        filter_stop_codons=True,
    )
    if "event" not in swarm.columns:
        swarm = swarm.with_columns(pl.lit("regular").alias("event"))
    else:
        swarm = swarm.with_columns(pl.col("event").fill_null("regular"))
    return swarm


def _build_wild_type_marker_table(tree: rx.PyDiGraph, plot_data: TreePlotData) -> pl.DataFrame:
    schema = {
        "tree_node_index": pl.Int64,
        "vertex_index": pl.Int64,
        "scalar": pl.Float64,
        "event": pl.String,
        "marker_label": pl.String,
        "dna_sequence": pl.String,
        "aa_sequence": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
    }
    node_coordinates = {
        int(row["tree_node_index"]): row for row in plot_data.node_table.to_dicts()
    }
    markers_by_vertex: dict[int, dict[str, Any]] = {}
    for tree_node_index in tree.node_indices():
        node = dict(tree.get_node_data(tree_node_index))
        if node.get("dna_sequence") != WILD_TYPE_DNA_SEQUENCE:
            continue
        coordinate = node_coordinates[int(tree_node_index)]
        vertex_index = _required_vertex_index(node)
        markers_by_vertex[vertex_index] = _wild_type_marker_row(
            node,
            tree_node_index=int(tree_node_index),
            x=float(coordinate["x"]),
        )

    edge_coordinates = {
        frozenset((int(row["source_tree_node_index"]), int(row["target_tree_node_index"]))): row
        for row in plot_data.edge_table.to_dicts()
    }
    for source, target in sorted(tuple(map(int, edge)) for edge in tree.edge_list()):
        edge = dict(tree.get_edge_data(source, target))
        coordinate = edge_coordinates[frozenset((source, target))]
        for node in parse_vertex_chain_json(edge.get("vertex_chain_json")):
            if node.get("dna_sequence") != WILD_TYPE_DNA_SEQUENCE:
                continue
            vertex_index = _required_vertex_index(node)
            if vertex_index in markers_by_vertex:
                continue
            markers_by_vertex[vertex_index] = _wild_type_marker_row(
                node,
                tree_node_index=None,
                x=float(coordinate["x_target"]),
            )

    if len(markers_by_vertex) > 1:
        raise ValueError(f"Multiple vertices match Papkou wild type {WILD_TYPE_DNA_SEQUENCE!r}.")
    return pl.DataFrame(list(markers_by_vertex.values()), schema=schema)


def _required_vertex_index(node: dict[str, Any]) -> int:
    vertex_index = node.get("vertex_index")
    if vertex_index is None:
        raise ValueError("Papkou wild-type marker data is missing vertex_index.")
    return int(vertex_index)


def _wild_type_marker_row(
    node: dict[str, Any], *, tree_node_index: int | None, x: float
) -> dict[str, Any]:
    scalar = float(node["scalar"])
    return {
        "tree_node_index": tree_node_index,
        "vertex_index": _required_vertex_index(node),
        "scalar": scalar,
        "event": str(node.get("event", "regular")),
        "marker_label": "WT",
        "dna_sequence": str(node["dna_sequence"]),
        "aa_sequence": str(node.get("aa_sequence", WILD_TYPE_AA_SEQUENCE)),
        "x": x,
        "y": scalar,
    }


def _has_stop_codon(sequence: Any) -> bool:
    return bool(STOP_CODON_SYMBOLS.intersection(str(sequence or "")))


def _progress(
    callbacks: Mapping[str, Callable[[int, int], None]] | None, name: str
) -> Callable[[int, int], None] | None:
    return None if callbacks is None else callbacks.get(name)


def _report_progress(
    callback: Callable[[int, int], None] | None, completed: int, total: int
) -> None:
    if callback is None:
        return
    step = max(1, total // 1000)
    if completed == 0 or completed == total or completed % step == 0:
        callback(completed, total)


SNAPSHOT_SCHEMA_VERSION = "2.0"
_SNAPSHOT_EXPECTED_TABLES = (
    "event_table.parquet",
    "node_table.parquet",
    "edge_table.parquet",
    "edge_segment_table.parquet",
    "branch_table.parquet",
    "branch_span_table.parquet",
    "branch_member_table.parquet",
    "critical_node_table.parquet",
    "stop_marker_table.parquet",
    "all_stop_marker_table.parquet",
    "wild_type_marker_table.parquet",
    "persistence_table.parquet",
)


def save_papkou2023_analysis_snapshot(
    analysis: Papkou2023TreeAnalysis,
    snapshot_dir: str | Path,
) -> Path:
    """Save a complete Papkou2023TreeAnalysis snapshot to disk."""
    dest = Path(snapshot_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    tree_path = dest / "tree.graphml"
    rx.digraph_write_graphml(analysis.tree, str(tree_path))

    tables = {
        "event_table.parquet": analysis.event_table,
        "node_table.parquet": analysis.plot_data.node_table,
        "edge_table.parquet": analysis.plot_data.edge_table,
        "edge_segment_table.parquet": analysis.plot_data.edge_segment_table,
        "branch_table.parquet": analysis.branch_table,
        "branch_span_table.parquet": analysis.branch_span_table,
        "branch_member_table.parquet": analysis.branch_member_table,
        "critical_node_table.parquet": analysis.critical_node_table,
        "stop_marker_table.parquet": analysis.stop_marker_table,
        "all_stop_marker_table.parquet": analysis.all_stop_marker_table,
        "wild_type_marker_table.parquet": analysis.wild_type_marker_table,
        "persistence_table.parquet": analysis.persistence_table,
    }

    for filename, df in tables.items():
        df.write_parquet(dest / filename)

    table_files_repr = ",\n".join(f'  "{f}"' for f in tables.keys())
    metadata_content = (
        f'schema_version = "{SNAPSHOT_SCHEMA_VERSION}"\n'
        f'kind = "{analysis.kind}"\n'
        f'scalar_attr = "{analysis.plot_data.scalar_attr}"\n'
        f'root_kind = "{analysis.plot_data.root_kind}"\n'
        f"root_node_index = {analysis.plot_data.root_node_index}\n"
        f"tables = [\n{table_files_repr}\n]\n"
    )
    (dest / "metadata.toml").write_text(metadata_content, encoding="utf-8")
    return dest


def load_papkou2023_analysis_snapshot(
    snapshot_dir: str | Path,
) -> Papkou2023TreeAnalysis:
    """Load a Papkou2023TreeAnalysis snapshot from disk."""
    path = Path(snapshot_dir).resolve()
    if not path.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {path}")

    metadata_path = path / "metadata.toml"
    if not metadata_path.exists():
        raise ValueError(f"Invalid snapshot in {path}: missing metadata.toml")

    try:
        with metadata_path.open("rb") as f:
            metadata = tomllib.load(f)
    except Exception as err:
        raise ValueError(f"Failed to parse metadata in {metadata_path}: {err}") from err

    version = metadata.get("schema_version")
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema version {version!r} in {metadata_path}. "
            f"Expected {SNAPSHOT_SCHEMA_VERSION!r}."
        )

    required_keys = {"kind", "scalar_attr", "root_kind", "root_node_index"}
    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError(
            f"Incomplete snapshot metadata in {metadata_path}: missing {sorted(missing_keys)}"
        )

    tree_path = path / "tree.graphml"
    if not tree_path.exists():
        raise ValueError(f"Incomplete snapshot in {path}: missing tree.graphml")

    try:
        loaded_graphs = rx.read_graphml(str(tree_path))
        if not loaded_graphs:
            raise ValueError(f"Empty tree graph in {tree_path}")
        tree = loaded_graphs[0]
        for idx in tree.node_indices():
            data = dict(tree.get_node_data(idx))
            if "id" in data:
                data.pop("id")
                tree[idx] = data
    except Exception as err:
        raise ValueError(f"Failed to load tree graph from {tree_path}: {err}") from err

    loaded_tables: dict[str, pl.DataFrame] = {}
    for filename in _SNAPSHOT_EXPECTED_TABLES:
        file_path = path / filename
        if not file_path.exists():
            raise ValueError(
                f"Incomplete snapshot in {path}: missing required table file '{filename}'"
            )
        try:
            loaded_tables[filename] = pl.read_parquet(file_path)
        except Exception as err:
            raise ValueError(
                f"Failed to read Parquet table '{filename}' in {path}: {err}"
            ) from err

    plot_data = TreePlotData(
        node_table=loaded_tables["node_table.parquet"],
        edge_table=loaded_tables["edge_table.parquet"],
        edge_segment_table=loaded_tables["edge_segment_table.parquet"],
        root_node_index=int(metadata["root_node_index"]),
        scalar_attr=str(metadata["scalar_attr"]),
        root_kind=str(metadata["root_kind"]),
    )

    return Papkou2023TreeAnalysis(
        kind=metadata["kind"],
        tree=tree,
        event_table=loaded_tables["event_table.parquet"],
        plot_data=plot_data,
        branch_table=loaded_tables["branch_table.parquet"],
        branch_span_table=loaded_tables["branch_span_table.parquet"],
        branch_member_table=loaded_tables["branch_member_table.parquet"],
        critical_node_table=loaded_tables["critical_node_table.parquet"],
        stop_marker_table=loaded_tables["stop_marker_table.parquet"],
        all_stop_marker_table=loaded_tables["all_stop_marker_table.parquet"],
        wild_type_marker_table=loaded_tables["wild_type_marker_table.parquet"],
        persistence_table=loaded_tables["persistence_table.parquet"],
    )
