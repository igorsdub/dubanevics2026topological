"""Numeric graph summaries cached with the comparison analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import polars as pl

from .biological_annotations import STOP_CODON_SYMBOLS
from .describe import SUMMARY_SCHEMA
from .leaves import Leaves


@dataclass(frozen=True)
class LandscapeSummaryInput:
    """One selected graph and tree state, with its full-library classifications.

    Component and incident counts describe the graph before largest-component
    selection. BY ``classes`` maps every Full-library DNA sequence to its call;
    point-estimate columns use ``None`` and the exact recorded threshold.
    """

    network: Any
    leaves: Leaves
    threshold: float | None
    classes: Mapping[str, str] | None
    component_count: int
    pre_lcc_vertex_count: int


SUMMARY_COLUMNS = (
    "Full (nominal)",
    "Published (point-estimate nominal)",
    "Median-reflected (point-estimate nominal)",
    "Published BY (nominal)",
    "Published BY (band-simplified)",
    "Median-reflected BY (nominal)",
    "Median-reflected BY (band-simplified)",
)

# The rows a band-simplified tree changes; every other row is a graph quantity.
TREE_LEVEL_METRICS = frozenset(
    {
        "Peak count",
        "Synonymous leaves",
        "Eligible leaves",
        "Ineligible root-only leaves",
        "Synonymous leaves (%)",
    }
)


def landscape_summary_column(
    item: LandscapeSummaryInput, *, full_fitness: Mapping[str, float]
) -> dict[str, int | float | str | None]:
    """Compute graph and tree statistics using explicit sequence populations."""
    full_identities = set(full_fitness)
    graph = item.network.graph  # one deep copy; the property copies on every access
    nodes = [dict(graph[i]) for i in graph.node_indices()]
    ids = [node.get("dna_sequence") for node in nodes]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Every graph vertex needs a DNA identity.")
    if len(set(ids)) != len(ids) or not set(ids) <= full_identities:
        raise ValueError("Selected graph identities must be unique Full members.")
    if not ids or not full_identities:
        raise ValueError("Graph populations must not be empty.")
    if any(
        node.get(item.network.scalar_attr) != full_fitness[node["dna_sequence"]]
        for node in nodes
    ):
        raise ValueError("Selected graph fitness differs from the Full library.")
    if item.pre_lcc_vertex_count < len(ids) or item.component_count < 1:
        raise ValueError("Invalid pre-largest-component diagnostics.")
    metrics = _summarize_network(
        item.network, "Selected", item.threshold, classes=item.classes
    )
    result = dict(metrics)
    if item.classes is None:
        result[UNCERTAIN_METRIC] = "not applicable"
    result.update(
        {
            "Number of components before largest component": item.component_count,
            "Vertices before largest component": item.pre_lcc_vertex_count,
            "Largest component": len(ids),
        }
    )
    if item.classes is not None and item.threshold is None:
        raise ValueError("A classification requires its exact threshold.")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in full_fitness.values()
    ):
        raise ValueError("Full-library fitness values must be finite numbers.")
    for scope, population in (
        ("Full library", full_identities),
        ("Selected graph", set(ids)),
    ):
        if item.classes is not None and set(item.classes) != full_identities:
            raise ValueError("Classification must cover precisely the Full library.")
        if item.classes is not None and set(item.classes.values()) - {
            "functional",
            "uncertain",
            "non_functional",
        }:
            raise ValueError("Unknown classification call.")
        if item.classes is not None and any(
            (full_fitness[identity] < item.threshold) != (call == "non_functional")
            for identity, call in item.classes.items()
        ):
            raise ValueError(
                "Classification calls disagree with the applied threshold."
            )
        for label, call in (
            ("functional", "functional"),
            ("uncertain", "uncertain"),
            ("non-functional", "non_functional"),
        ):
            if item.classes is not None:
                count = sum(item.classes[identity] == call for identity in population)
            elif item.threshold is None or call == "uncertain":
                count = "not applicable"
            else:
                count = sum(
                    (full_fitness[identity] >= item.threshold) == (call == "functional")
                    for identity in population
                )
            result[f"{scope} {label} sequences"] = count
            result[f"{scope} {label} sequences (%)"] = (
                100 * count / len(population) if isinstance(count, int) else count
            )
    for n in (1, 2, 3):
        if any(
            not isinstance(node.get("aa_sequence"), str) or not node["aa_sequence"]
            for node in nodes
        ):
            raise ValueError("Every graph vertex needs an amino-acid sequence.")
        count = sum(
            sum(c in STOP_CODON_SYMBOLS for c in node["aa_sequence"]) == n
            for node in nodes
        )
        result[f"Sequences with exactly {n} stops"] = count
        result[f"Sequences with exactly {n} stops (%)"] = 100 * count / len(ids)
    leaves = item.leaves
    rows = leaves.table.to_dicts()
    if leaves.summary.get("tree_member_count") != len(ids):
        raise ValueError("Leaf tree population differs from selected graph size.")
    members = leaves.members["member_dna_sequence"].to_list()
    if len(members) != len(set(members)) or not set(members) <= set(ids):
        raise ValueError("Leaf membership is duplicated or outside the selected graph.")
    if any(
        row["is_synonymous"] is not None
        if row["is_root_endpoint"]
        else row["is_synonymous"] is None
        for row in rows
    ):
        raise ValueError("Leaf eligibility and synonymous status disagree.")
    eligible = sum(not row["is_root_endpoint"] for row in rows)
    ineligible = len(rows) - eligible
    if (
        leaves.summary.get("leaf_count") != len(rows)
        or leaves.summary.get("eligible_leaf_denominator") != eligible
        or leaves.summary.get("ineligible_endpoint_count") != ineligible
    ):
        raise ValueError("Leaf table and summary disagree.")
    synonymous = sum(row["is_synonymous"] is True for row in rows)
    result.update(
        {
            "Peak count": len(rows),
            "Synonymous leaves": synonymous,
            "Eligible leaves": eligible,
            "Ineligible root-only leaves": ineligible,
            "Synonymous leaves (%)": 100 * synonymous / eligible if eligible else None,
        }
    )
    return result


def landscape_summary(
    inputs: Mapping[str, LandscapeSummaryInput],
    *,
    full_fitness: Mapping[str, float],
    mixture: LandscapeSummaryInput | None = None,
) -> dict[str, dict[str, int | float | str | None]]:
    """Return the seven canonical columns, optionally followed by mixture."""
    if set(inputs) != set(SUMMARY_COLUMNS):
        raise ValueError("All seven canonical landscape states are required.")
    if any(inputs[name].classes is not None for name in SUMMARY_COLUMNS[:3]):
        raise ValueError("Point-estimate states cannot use BY classification calls.")
    if any(inputs[name].classes is None for name in SUMMARY_COLUMNS[3:]):
        raise ValueError("BY states require their saved classification calls.")
    full_graph = inputs[SUMMARY_COLUMNS[0]].network.graph
    if {
        full_graph[index]["dna_sequence"] for index in full_graph.node_indices()
    } != set(full_fitness):
        raise ValueError("Full graph and Full-library identities disagree.")
    for name in SUMMARY_COLUMNS:
        expected_state = "band" if "band-simplified" in name else "nominal"
        if inputs[name].leaves.summary.get("tree_state") != expected_state:
            raise ValueError(f"Leaves state does not match {name}.")
    result = {
        name: landscape_summary_column(inputs[name], full_fitness=full_fitness)
        for name in SUMMARY_COLUMNS
    }
    for nominal, band in (
        (SUMMARY_COLUMNS[3], SUMMARY_COLUMNS[4]),
        (SUMMARY_COLUMNS[5], SUMMARY_COLUMNS[6]),
    ):

        def graph_ids(name: str) -> set[str]:
            graph = inputs[name].network.graph
            return {graph[index]["dna_sequence"] for index in graph.node_indices()}

        if graph_ids(nominal) != graph_ids(band):
            raise ValueError(f"Graph identities differ between {nominal} and {band}.")
        if any(
            result[nominal][key] != result[band][key]
            for key in result[nominal]
            if key not in TREE_LEVEL_METRICS
        ):
            raise ValueError(f"Graph metrics differ between {nominal} and {band}.")
    if mixture is not None:
        result["Mixture (nominal)"] = landscape_summary_column(
            mixture, full_fitness=full_fitness
        )
    return result


METRICS = (
    "Number of nodes",
    "Number of edges",
    "Minimum fitness",
    "Maximum fitness",
    "Mean fitness",
    "Median fitness",
    "Functional threshold",
    "Number of functional sequences",
    "Number of non-functional sequences",
    "Percentage of functional sequences",
)
# Reported beside the metrics, under the Functional-class rule only (the registry
# carries it; Table 1 has no row for it).
UNCERTAIN_METRIC = "Number of uncertain sequences"


def _summarize_network(
    network: Any,
    label: str,
    threshold: float | None,
    *,
    classes: Mapping[str, str] | None = None,
) -> dict[str, float | int | None]:
    """Numeric summary of one graph.

    By default a sequence is functional when its raw fitness is at least
    ``threshold``. With ``classes`` (DNA identity to ``functional`` /
    ``non_functional`` / ``uncertain``, the functional-classification stage) the
    class decides instead: the functional, non-functional and uncertain counts are
    the vertices of the graph in each class, and ``threshold`` is only reported.
    """
    graph = network.graph
    fitness: list[float] = []
    identities: list[Any] = []
    for index in graph.node_indices():
        payload = dict(graph[index])
        identity = payload.get(
            "dna_sequence", payload.get("vertex_id", payload.get("id"))
        )
        value = payload.get(network.scalar_attr)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"{label} graph has missing, non-numeric, or non-finite fitness "
                f"for sequence {identity!r}."
            )
        fitness.append(float(value))
        identities.append(identity)
    if not fitness:
        raise ValueError(f"{label} graph has no nodes.")
    uncertain: int | None = None
    if classes is not None:
        found = []
        for identity in identities:
            if identity not in classes:
                raise ValueError(
                    f"{label} graph sequence {identity!r} has no functional class."
                )
            found.append(classes[identity])
        functional = found.count("functional")
        nonfunctional = found.count("non_functional")
        uncertain = found.count("uncertain")
    else:
        functional = (
            sum(value >= threshold for value in fitness)
            if threshold is not None
            else None
        )
        nonfunctional = len(fitness) - functional if functional is not None else None
    return {
        "Number of nodes": len(fitness),
        "Number of edges": network.edge_count,
        "Minimum fitness": min(fitness),
        "Maximum fitness": max(fitness),
        "Mean fitness": mean(fitness),
        "Median fitness": median(fitness),
        "Functional threshold": threshold,
        "Number of functional sequences": functional,
        "Number of non-functional sequences": nonfunctional,
        "Percentage of functional sequences": (100 * functional / len(fitness))
        if functional is not None
        else None,
        UNCERTAIN_METRIC: uncertain,
    }


def split_tree_peak_count(tree_analysis: Any) -> int:
    """The peak count of a graph: the maxima (leaves) of its split tree.

    ``tree_analysis`` is a Topographer split-tree analysis. Every figure that states
    a Full, Published or Candidate peak count counts the ``maximum`` events of its
    event table, and the manuscript quantity registry reads the same number here.
    """
    events = tree_analysis.event_table
    if "event" not in events.columns:
        raise ValueError("The split-tree event table has no 'event' column.")
    return int(events.filter(pl.col("event") == "maximum").height)


def split_tree_sequence_count(tree_analysis: Any) -> int:
    """The sequences a split tree represents: its critical nodes and branch members.

    Counted by DNA identity, as the figures do (``tree_population``); the
    band-simplified view retains exactly these sequences.
    """
    tables = [
        table.select("dna_sequence")
        for table in (
            tree_analysis.critical_node_table,
            tree_analysis.branch_member_table,
        )
    ]
    return int(pl.concat(tables).unique("dna_sequence").height)


def _summary_payload(
    full: Any,
    published: Any,
    candidate: Any,
    published_functional_threshold: float,
    candidate_functional_threshold: float,
    *,
    full_summary: dict[str, float | int | None] | None = None,
    candidate_classes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SUMMARY_SCHEMA,
        "metrics": {
            "Full": full_summary or _summarize_network(full, "Full", None),
            "Published": _summarize_network(
                published, "Published", published_functional_threshold
            ),
            "Candidate": _summarize_network(
                candidate,
                "Candidate",
                candidate_functional_threshold,
                classes=candidate_classes,
            ),
        },
    }
