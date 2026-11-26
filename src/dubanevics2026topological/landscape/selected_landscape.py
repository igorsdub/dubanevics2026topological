"""Prepare threshold-selected Papkou landscapes and analyze them at raw fitness."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl
import rustworkx as rx
from dubanevics2026topological.network_analysis import NetworkAnalysis, TreeAnalysis
from dubanevics2026topological.papkou2023 import build_papkou2023_backbone_analysis
from topographer.graph import Graph, ScalarNetwork

SCALAR_EDGE_SELECTION_RULE = "at_least_one_endpoint_raw_fitness_ge_threshold"
CLASS_EDGE_SELECTION_RULE = "at_least_one_endpoint_functional_class"


@dataclass(slots=True)
class PreparedLandscape:
    """A threshold-selected, largest-component landscape and auditable products."""

    network: ScalarNetwork
    threshold: float
    components: pl.DataFrame
    excluded_vertices: pl.DataFrame
    peaks: pl.DataFrame
    diagnostics: dict[str, Any]
    tree_analysis: NetworkAnalysis | None = None


def prepare_clamped_landscape(
    full_network: ScalarNetwork,
    threshold: float,
    *,
    functional_sequences: Collection[str] | None = None,
    functional_classes: Mapping[str, str] | None = None,
) -> PreparedLandscape:
    """Select edges incident to functional vertices, then retain a largest component.

    A vertex is functional by the scalar rule when its raw fitness is at least
    ``threshold``. When ``functional_sequences`` is given, the Functional class of
    the functional-classification stage decides instead: a vertex is functional
    exactly when its DNA identity is in the set, whatever its raw fitness, and
    ``threshold`` only sets the ``clamped_fitness`` metadata. ``functional_classes``
    (DNA identity to ``functional`` / ``non_functional`` / ``uncertain``) labels the
    ``functional_class`` column of ``excluded_vertices``; it is null under the
    scalar rule and for a vertex the mapping does not name.

    Clamped fitness is stored as reference metadata only. The returned network and
    all downstream tree and persistence calculations use the unchanged raw values.

    The tie between equally large components is resolved by the lexicographically
    smallest minimum DNA identity. Input and output vertex identities are sequence
    identities supplied by the ScalarNetwork, never Rustworkx indices.
    """
    if not isinstance(full_network, ScalarNetwork):
        raise TypeError("full_network must be a Topographer ScalarNetwork.")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a real number.")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    functional_set = (
        None if functional_sequences is None else frozenset(functional_sequences)
    )
    source = full_network.graph
    identity_attr = "dna_sequence"
    scalar_attr = full_network.scalar_attr
    node_by_id: dict[str, dict[str, Any]] = {}
    for node_index in source.node_indices():
        payload = dict(source[node_index])
        identity = payload.get(identity_attr)
        if not isinstance(identity, str) or not identity:
            raise ValueError(
                "Every vertex must have a non-empty dna_sequence identity."
            )
        if identity in node_by_id:
            raise ValueError(f"Duplicate DNA sequence identity {identity!r}.")
        raw_value = payload.get(scalar_attr)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"Vertex {identity!r} has a non-numeric raw fitness.")
        if not math.isfinite(float(raw_value)):
            raise ValueError(f"Vertex {identity!r} has non-finite raw fitness.")
        payload["raw_fitness"] = float(raw_value)
        payload["clamped_fitness"] = max(float(raw_value), threshold)
        node_by_id[identity] = payload

    if functional_set is not None:
        absent = sorted(functional_set - node_by_id.keys())
        if absent:
            raise ValueError(
                f"{len(absent)} Functional sequences are not vertices of the Full "
                f"graph, first: {absent[:3]!r}; the functional classification and "
                "the Full graph describe different sequences."
            )

    def is_functional(identity: str) -> bool:
        if functional_set is not None:
            return identity in functional_set
        return node_by_id[identity]["raw_fitness"] >= threshold

    def class_of(identity: str) -> str | None:
        if functional_set is None:
            return None
        if functional_classes is not None:
            return functional_classes.get(identity)
        return "functional" if identity in functional_set else None

    eligible_edges: list[tuple[str, str, dict[str, Any]]] = []
    excluded_low_low_edges = 0
    for edge_index in source.edge_indices():
        left_index, right_index = source.get_edge_endpoints_by_index(edge_index)
        left = str(source[left_index][identity_attr])
        right = str(source[right_index][identity_attr])
        if left == right:
            raise ValueError(f"Self-loop found at sequence {left!r}.")
        if not (is_functional(left) or is_functional(right)):
            excluded_low_low_edges += 1
            continue
        eligible_edges.append(
            (left, right, dict(source.get_edge_data_by_index(edge_index) or {}))
        )

    incident = {
        identity for left, right, _ in eligible_edges for identity in (left, right)
    }
    if not incident:
        raise ValueError(
            "Threshold selection retained no edges; no incident vertices remain."
        )
    adjacency = {identity: set() for identity in incident}
    for left, right, _ in eligible_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[set[str]] = []
    unseen = set(incident)
    while unseen:
        root = min(unseen)
        pending = [root]
        component: set[str] = set()
        unseen.remove(root)
        while pending:
            identity = pending.pop()
            component.add(identity)
            for neighbor in sorted(adjacency[identity] & unseen):
                unseen.remove(neighbor)
                pending.append(neighbor)
        components.append(component)

    selected_component = min(
        components,
        key=lambda component: (-len(component), min(component)),
    )
    component_rows = []
    excluded_rows = []
    for component_index, component in enumerate(
        sorted(components, key=lambda item: (-len(item), min(item))), start=1
    ):
        selected = component == selected_component
        component_rows.append(
            {
                "component_id": component_index,
                "vertex_count": len(component),
                "selected": selected,
                "minimum_sequence": min(component),
                "sequences": sorted(component),
            }
        )
        if not selected:
            excluded_rows.extend(
                {
                    "dna_sequence": identity,
                    "component_id": component_index,
                    "clamped_fitness": node_by_id[identity]["clamped_fitness"],
                    "raw_fitness": node_by_id[identity]["raw_fitness"],
                    "reason": "non_largest_component",
                    "functional_class": class_of(identity),
                }
                for identity in sorted(component)
            )
    excluded_rows.extend(
        {
            "dna_sequence": identity,
            "component_id": None,
            "clamped_fitness": node_by_id[identity]["clamped_fitness"],
            "raw_fitness": node_by_id[identity]["raw_fitness"],
            "reason": "no_retained_incident_edge",
            "functional_class": class_of(identity),
        }
        for identity in sorted(set(node_by_id) - incident)
    )

    selected_edges = [
        (left, right, attrs)
        for left, right, attrs in eligible_edges
        if left in selected_component and right in selected_component
    ]
    prepared_rx = rx.PyGraph(multigraph=False)
    index_by_id: dict[str, int] = {}
    for identity in sorted(selected_component):
        index_by_id[identity] = prepared_rx.add_node(node_by_id[identity])
    for left, right, attrs in sorted(
        selected_edges, key=lambda edge: tuple(sorted(edge[:2]))
    ):
        prepared_rx.add_edge(index_by_id[left], index_by_id[right], attrs)
    vertex_attrs = sorted(
        set().union(*(set(node_by_id[identity]) for identity in selected_component))
        - {identity_attr}
    )
    edge_attrs = (
        sorted(set().union(*(set(attrs) for _, _, attrs in selected_edges)))
        if selected_edges
        else []
    )
    graph = Graph.from_rustworkx(
        prepared_rx,
        vertex_id_attr=identity_attr,
        vertex_attribute_cols=vertex_attrs,
        edge_attribute_cols=edge_attrs,
    )
    network = ScalarNetwork(graph, scalar_attr="raw_fitness")

    peaks = []
    for identity in sorted(selected_component):
        if not any(
            node_by_id[neighbor]["raw_fitness"] > node_by_id[identity]["raw_fitness"]
            for neighbor in adjacency[identity]
            if neighbor in selected_component
        ):
            peaks.append(
                {
                    "dna_sequence": identity,
                    "raw_fitness": node_by_id[identity]["raw_fitness"],
                    "clamped_fitness": node_by_id[identity]["clamped_fitness"],
                }
            )
    component_table = pl.DataFrame(
        component_rows,
        schema={
            "component_id": pl.Int64,
            "vertex_count": pl.Int64,
            "selected": pl.Boolean,
            "minimum_sequence": pl.String,
            "sequences": pl.List(pl.String),
        },
    )
    excluded_table = pl.DataFrame(
        excluded_rows,
        schema={
            "dna_sequence": pl.String,
            "component_id": pl.Int64,
            "clamped_fitness": pl.Float64,
            "raw_fitness": pl.Float64,
            "reason": pl.String,
            "functional_class": pl.String,
        },
    )
    peak_table = pl.DataFrame(
        peaks,
        schema={
            "dna_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "clamped_fitness": pl.Float64,
        },
    )
    diagnostics = {
        "threshold": threshold,
        "edge_selection_rule": (
            SCALAR_EDGE_SELECTION_RULE
            if functional_set is None
            else CLASS_EDGE_SELECTION_RULE
        ),
        "analysis_scalar": "raw_fitness",
        "input_vertex_count": full_network.vertex_count,
        "input_edge_count": full_network.edge_count,
        "low_low_edge_count_excluded": excluded_low_low_edges,
        "incident_vertex_count": len(incident),
        "component_count": len(components),
        "selected_component_vertex_count": len(selected_component),
        "selected_component_edge_count": len(selected_edges),
        "excluded_vertex_count": excluded_table.height,
        "peak_count": peak_table.height,
        "functional_vertex_count": (
            None if functional_set is None else len(functional_set)
        ),
        "functional_set_sha256": hashlib.sha256(
            "\n".join(
                sorted(
                    identity
                    for identity in node_by_id
                    if (
                        identity in functional_set
                        if functional_set is not None
                        else is_functional(identity)
                    )
                )
            ).encode("utf-8")
        ).hexdigest(),
        # Functional vertices of the retained component (the two above are over the
        # whole classification, i.e. the Full graph).
        "selected_component_functional_vertex_count": sum(
            is_functional(identity) for identity in selected_component
        ),
        "uncertain_vertex_count": (
            None
            if functional_set is None or functional_classes is None
            else sum(1 for name in functional_classes.values() if name == "uncertain")
        ),
        "selected_component_tie_break": "largest size, then lexicographically smallest minimum DNA sequence",
    }
    return PreparedLandscape(
        network=network,
        threshold=threshold,
        components=component_table,
        excluded_vertices=excluded_table,
        peaks=peak_table,
        diagnostics=diagnostics,
    )


def analyze_prepared_landscape(
    prepared: PreparedLandscape,
    *,
    name: str,
    wild_type_sequence: str,
) -> NetworkAnalysis:
    """Build a renderer-compatible public analysis for one prepared network."""
    papkou2023_tree = build_papkou2023_backbone_analysis(
        prepared.network,
        kind="split",
    )
    tree = TreeAnalysis(
        kind="split",
        scalar_attr="raw_fitness",
        tree=papkou2023_tree.tree,
        plot_data=papkou2023_tree.plot_data,
        event_table=papkou2023_tree.event_table,
        branch_table=papkou2023_tree.branch_table,
        branch_span_table=papkou2023_tree.branch_span_table,
        persistence_table=papkou2023_tree.persistence_table,
        branch_member_table=papkou2023_tree.branch_member_table,
        critical_node_table=papkou2023_tree.critical_node_table,
        stop_marker_table=papkou2023_tree.stop_marker_table,
        all_stop_marker_table=papkou2023_tree.all_stop_marker_table,
        wild_type_marker_table=_wild_type_marker(papkou2023_tree, wild_type_sequence),
    )
    scalars = [
        float(payload["raw_fitness"]) for payload in prepared.network.graph.nodes()
    ]
    domain = (min(scalars), max(scalars))
    prepared.tree_analysis = NetworkAnalysis(
        name=name,
        config={"threshold": prepared.threshold, "scalar_attr": "raw_fitness"},
        scalar_attr="raw_fitness",
        scalar_domain=(float(domain[0]), float(domain[1])),
        scalar_label="Raw fitness",
        input_summary=pl.DataFrame(
            [
                {"metric": "vertex_count", "value": prepared.network.vertex_count},
                {"metric": "edge_count", "value": prepared.network.edge_count},
                {"metric": "scalar_min", "value": float(domain[0])},
                {"metric": "scalar_max", "value": float(domain[1])},
            ]
        ),
        trees={"split": tree},
    )
    return prepared.tree_analysis


def validate_clamped_reference(
    prepared: PreparedLandscape,
    reference_network: ScalarNetwork,
    reference_peaks: pl.DataFrame,
    *,
    scalar_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Validate identities, undirected adjacency, raw/clamped scalars, and peak IDs.

    Scalar comparisons use absolute tolerance ``1e-12`` to allow GraphML decimal
    serialization differences. Vertex and edge identities and peak identities are
    exact set comparisons.
    """
    if scalar_tolerance < 0 or not math.isfinite(scalar_tolerance):
        raise ValueError("scalar_tolerance must be finite and non-negative.")
    reference_graph = reference_network.graph
    prepared_graph = prepared.network.graph
    ref_ids = set(reference_network.vertex_ids)
    actual_ids = set(prepared.network.vertex_ids)
    missing_ids = sorted(ref_ids - actual_ids)
    extra_ids = sorted(actual_ids - ref_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            f"Prepared sequence identities differ from reference: {len(missing_ids)} missing, "
            f"{len(extra_ids)} extra."
        )
    ref_index_to_id = {
        int(index): str(reference_graph[index]["dna_sequence"])
        for index in reference_graph.node_indices()
    }
    actual_index_to_id = {
        int(index): str(prepared_graph[index]["dna_sequence"])
        for index in prepared_graph.node_indices()
    }
    reference_edges = {
        tuple(sorted((ref_index_to_id[int(left)], ref_index_to_id[int(right)])))
        for left, right in reference_graph.edge_list()
    }
    actual_edges = {
        tuple(sorted((actual_index_to_id[int(left)], actual_index_to_id[int(right)])))
        for left, right in prepared_graph.edge_list()
    }
    if reference_edges != actual_edges:
        raise ValueError(
            "Prepared unordered edge identities differ from reference: "
            f"{len(reference_edges - actual_edges)} missing, "
            f"{len(actual_edges - reference_edges)} extra."
        )
    reference_payloads = {
        ref_index_to_id[int(index)]: dict(reference_graph[index])
        for index in reference_graph.node_indices()
    }
    prepared_payloads = {
        actual_index_to_id[int(index)]: dict(prepared_graph[index])
        for index in prepared_graph.node_indices()
    }
    max_raw_error = max(
        abs(
            float(prepared_payloads[identity]["raw_fitness"])
            - float(reference_payloads[identity][reference_network.scalar_attr])
        )
        for identity in actual_ids
    )
    max_clamped_error = max(
        abs(
            float(prepared_payloads[identity]["clamped_fitness"])
            - float(reference_payloads[identity]["Fitness"])
        )
        for identity in actual_ids
    )
    if max_raw_error > scalar_tolerance:
        raise ValueError(f"Raw fitness differs from reference by {max_raw_error:g}.")
    if max_clamped_error > scalar_tolerance:
        raise ValueError(
            f"Clamped fitness differs from reference by {max_clamped_error:g}."
        )
    if "dna" not in reference_peaks.columns:
        raise ValueError("Reference peak table must contain a `dna` column.")
    reference_peak_ids = set(reference_peaks.get_column("dna").cast(pl.String))
    actual_peak_ids = set(prepared.peaks.get_column("dna_sequence"))
    if actual_peak_ids != reference_peak_ids:
        raise ValueError(
            "Computed peak sequence identities differ from reference: "
            f"{len(reference_peak_ids - actual_peak_ids)} missing, "
            f"{len(actual_peak_ids - reference_peak_ids)} extra."
        )
    return {
        "validated": True,
        "threshold": prepared.threshold,
        "vertex_count": len(actual_ids),
        "edge_count": len(actual_edges),
        "peak_count": len(actual_peak_ids),
        "max_raw_fitness_absolute_error": max_raw_error,
        "max_clamped_fitness_absolute_error": max_clamped_error,
        "scalar_absolute_tolerance": scalar_tolerance,
        "vertex_identities_match": True,
        "undirected_edges_match": True,
        "peak_identities_match": True,
    }


def _wild_type_marker(tree_analysis: Any, wild_type_sequence: str) -> pl.DataFrame:
    """Return the WT location from the reduced tree or its represented edge."""
    plot_data = tree_analysis.plot_data
    coordinates = {
        int(row["tree_node_index"]): row for row in plot_data.node_table.to_dicts()
    }
    for node_index in tree_analysis.tree.node_indices():
        node = dict(tree_analysis.tree[node_index])
        if node.get("dna_sequence") == wild_type_sequence:
            point = coordinates[int(node_index)]
            return _marker_frame(node, point["x"], wild_type_sequence)
    edge_coordinates = {
        frozenset(
            (int(row["source_tree_node_index"]), int(row["target_tree_node_index"]))
        ): row
        for row in plot_data.edge_table.to_dicts()
    }
    for source, target in tree_analysis.tree.edge_list():
        edge = dict(tree_analysis.tree.get_edge_data(source, target))
        chain = edge.get("vertex_chain_json")
        for node in json.loads(chain) if isinstance(chain, str) else (chain or []):
            if node.get("dna_sequence") == wild_type_sequence:
                coordinates_row = edge_coordinates[
                    frozenset((int(source), int(target)))
                ]
                return _marker_frame(
                    node, coordinates_row["x_target"], wild_type_sequence
                )
    return pl.DataFrame(
        schema={
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
    )


def _marker_frame(node: dict[str, Any], x: float, sequence: str) -> pl.DataFrame:
    scalar = float(node["scalar"])
    return pl.DataFrame(
        [
            {
                "tree_node_index": node.get("tree_node_index"),
                "vertex_index": node.get("vertex_index"),
                "scalar": scalar,
                "event": str(node.get("event", "regular")),
                "marker_label": "WT",
                "dna_sequence": sequence,
                "aa_sequence": str(node.get("aa_sequence") or ""),
                "x": float(x),
                "y": scalar,
            }
        ],
        schema_overrides={
            "tree_node_index": pl.Int64,
            "vertex_index": pl.Int64,
            "scalar": pl.Float64,
            "event": pl.String,
            "marker_label": pl.String,
            "dna_sequence": pl.String,
            "aa_sequence": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
        },
    )
