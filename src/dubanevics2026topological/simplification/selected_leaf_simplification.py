"""Identity-selected leaf absorption for a publication view of a split tree."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
import rustworkx as rx
from topographer.graph import Graph, ScalarNetwork
from topographer.plotting import make_tree_layout
from topographer.topology import MergeTree, ScalarNetworkTopology


@dataclass(frozen=True)
class SelectedLeafView:
    """Simplified structure with sequence-level reassignment evidence."""

    plot_data: Any
    pairs: pl.DataFrame
    assignments: pl.DataFrame
    wild_type_marker_table: pl.DataFrame
    removed_saddles: tuple[str, ...]
    tree: Any


def absorb_singleton_leaves(
    tree_analysis: Any,
    peak_ids: set[str],
    *,
    horizontal_layout: str = "subtree_extremum",
) -> SelectedLeafView:
    """Absorb nominated terminal singleton branches and contract regular saddles."""
    return _absorb(
        tree_analysis,
        peak_ids,
        singleton_only=True,
        horizontal_layout=horizontal_layout,
    )


def absorb_terminal_branches(
    tree_analysis: Any,
    peak_ids: set[str],
    *,
    horizontal_layout: str = "subtree_extremum",
) -> SelectedLeafView:
    """Absorb nominated terminal branches of any size into their saddles.

    Like :func:`absorb_singleton_leaves` without the singleton restriction: the
    whole branch (peak plus regular vertices) joins the saddle's lower branch.
    Regular vertices of an absorbed branch are labelled ``absorbed_branch`` and
    flagged ``absorbed_branch_member`` (the flag survives cascades into the
    root, where ``location`` becomes ``absorbed_root``). A nominated peak whose
    saddle is the root moves its branch into the root group.
    """
    return _absorb(
        tree_analysis,
        peak_ids,
        singleton_only=False,
        horizontal_layout=horizontal_layout,
    )


def enforce_elder_rule(tree_analysis: Any, peak_ids: set[str]) -> set[str]:
    """Return the nominated terminal peaks that the elder rule protects.

    At each saddle the incoming arm whose upstream subtree holds the highest
    scalar is the elder arm and always survives; the global maximum is the
    elder arm of every saddle on its path and is also guarded explicitly. Ties
    on the subtree maximum are broken deterministically by the smaller
    ``vertex_index`` (the node payload field) of the node holding that maximum,
    matching Topographer's elder convention (``_elder_key`` in
    ``topographer.topology``). ``vertex_index`` is used only to
    order ties and is never exported. Subtree maxima come from the tree graph
    itself (all nodes upstream through the arm).
    """
    graph = tree_analysis.tree
    ids = {graph[i]["dna_sequence"]: i for i in graph.node_indices()}
    if not peak_ids <= ids.keys():
        raise ValueError("Selected peak identity is absent from the tree.")

    scalars = {i: float(graph[i]["scalar"]) for i in graph.node_indices()}
    if not all(math.isfinite(v) for v in scalars.values()):
        raise ValueError("Tree node scalars must be finite for the elder rule.")
    # Edges point upper -> lower, so one topological pass gives every node's
    # subtree-maximum key: the min of its own key and its predecessors' keys.
    missing = [
        str(graph[i]["dna_sequence"])
        for i in graph.node_indices()
        if graph[i].get("vertex_index") is None
    ]
    if missing:
        raise ValueError(
            "Tree nodes lack a vertex_index for elder-rule ties: "
            + ", ".join(sorted(missing)[:20])
        )
    key: dict[int, tuple[float, int]] = {}
    for i in rx.topological_sort(graph):
        own = (-scalars[i], int(graph[i]["vertex_index"]))
        key[i] = min([own, *(key[p] for p in graph.predecessor_indices(i))])
    global_max_vertex = min(key.values())[1]
    global_max = next(
        graph[i]["dna_sequence"]
        for i in graph.node_indices()
        if int(graph[i]["vertex_index"]) == global_max_vertex
    )
    protected: set[str] = set()
    for dna in sorted(peak_ids):
        peak = ids[dna]
        if graph.out_degree(peak) != 1 or graph.in_degree(peak) != 0:
            raise ValueError(f"Selected sequence {dna} is not a terminal peak.")
        if dna == global_max:
            protected.add(dna)
            continue
        saddle = next(iter(graph.successor_indices(peak)))
        elder = min(graph.predecessor_indices(saddle), key=key.__getitem__)
        if elder == peak:
            protected.add(dna)
    return protected


def rebuild_split_tree(
    graph: Any, *, vertex_attribute_cols: Sequence[str] = ("scalar",)
) -> MergeTree:
    """Rebuild a tree of critical vertices as a Topographer split tree.

    ``graph`` is a directed Rustworkx tree whose edges run from the upper to the
    lower vertex and whose node payloads carry ``dna_sequence`` and every column
    of ``vertex_attribute_cols`` (``scalar`` among them). The rebuild is the one
    Topographer computes persistence and Horton–Strahler orders on, shared by
    absorption (pairs of the surviving tree) and by the branch decomposition
    (its orders). Topographer orders vertices by identity when it builds its
    graph, so ties between equal scalars, and with them the elder arm, resolve as
    Topographer canonicalises them whatever order the nodes were inserted in.
    """
    undirected = rx.PyGraph()
    index_map = {i: undirected.add_node(dict(graph[i])) for i in graph.node_indices()}
    for upper, lower in graph.edge_list():
        undirected.add_edge(index_map[upper], index_map[lower], {})
    network = ScalarNetwork(
        Graph.from_rustworkx(
            undirected,
            vertex_id_attr="dna_sequence",
            vertex_attribute_cols=tuple(vertex_attribute_cols),
        ),
        scalar_attr="scalar",
    )
    return ScalarNetworkTopology(network).split_tree()


def _absorb(
    tree_analysis: Any,
    peak_ids: set[str],
    *,
    singleton_only: bool,
    horizontal_layout: str,
) -> SelectedLeafView:
    """Absorb nominated terminal branches and contract regular saddles.

    Work on a defensive graph copy. Branch membership retains every original
    sequence and its measured scalar, including absorbed peaks and saddles.
    Recompute elder-rule pairs from the surviving scalar tree; deleting old
    persistence rows alone would leave incorrect death endpoints.
    """
    graph = tree_analysis.tree.copy()
    nodes = {i: dict(graph[i]) for i in graph.node_indices()}
    ids = {row["dna_sequence"]: i for i, row in nodes.items()}
    if not peak_ids <= ids.keys():
        raise ValueError("Selected peak identity is absent from the tree.")
    members = {(u, v): [nodes[u]] for u, v in graph.edge_list()}
    for row in tree_analysis.branch_member_table.to_dicts():
        members[(row["source_tree_node_index"], row["target_tree_node_index"])].append(
            row
        )
    roots = [i for i in nodes if graph.out_degree(i) == 0]
    if len(roots) != 1:
        raise ValueError("Selected-leaf absorption requires one descending tree root.")
    root = roots[0]
    original_population = [nodes[root]] + [
        r for group in members.values() for r in group
    ]
    original_ids = [r["dna_sequence"] for r in original_population]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError("Input sequence membership is not a partition.")
    # Validate all nominations before changing the copy; the rule is evaluated
    # on the original tree, not iteratively on newly exposed branches.
    for dna in sorted(peak_ids):
        peak = ids[dna]
        if graph.in_degree(peak) != 0 or graph.out_degree(peak) != 1:
            raise ValueError(f"Selected sequence {dna} is not a terminal peak.")
        saddle = next(iter(graph.successor_indices(peak)))
        is_root = saddle == root and graph.in_degree(saddle) >= 2
        is_saddle = graph.in_degree(saddle) >= 2 and graph.out_degree(saddle) == 1
        if not (is_root or is_saddle):
            raise ValueError(f"Selected peak {dna} has no adjacent internal saddle.")
        if singleton_only and len(members[(peak, saddle)]) != 1:
            raise ValueError(f"Selected peak {dna} is not a singleton branch.")
    absorbed_branch: set[str] = set()
    root_members: list[dict[str, Any]] = []
    for dna in sorted(peak_ids):
        peak = ids[dna]
        saddle = next(iter(graph.successor_indices(peak)))
        branch = members.pop((peak, saddle))
        absorbed_branch.update(
            r["dna_sequence"] for r in branch if r["dna_sequence"] != dna
        )
        if saddle == root:
            # The branch joins the root directly: there is no lower branch.
            root_members.extend(branch)
        else:
            lower = next(iter(graph.successor_indices(saddle)))
            members[(saddle, lower)].extend(branch)
        graph.remove_node(peak)
    removed = []
    # When every incoming leaf is selected, the former saddle is not a new
    # biological peak: absorb the emptied branch farther down the trunk.
    while True:
        empty = [
            i
            for i in graph.node_indices()
            if i != root and graph.in_degree(i) == 0 and nodes[i]["event"] != "maximum"
        ]
        if not empty:
            break
        for saddle in empty:
            lower = next(iter(graph.successor_indices(saddle)))
            group = members.pop((saddle, lower))
            if lower == root:
                root_members.extend(group)
            else:
                below = next(iter(graph.successor_indices(lower)))
                members[(lower, below)].extend(group)
            graph.remove_node(saddle)
            removed.append(nodes[saddle]["dna_sequence"])
    if graph.num_nodes() == 1:
        raise ValueError("Selected leaves would remove every surviving peak.")
    for saddle in list(graph.node_indices()):
        if graph.in_degree(saddle) == 1 and graph.out_degree(saddle) == 1:
            upper = next(iter(graph.predecessor_indices(saddle)))
            lower = next(iter(graph.successor_indices(saddle)))
            members[(upper, lower)] = members.pop((upper, saddle)) + members.pop(
                (saddle, lower)
            )
            graph.remove_node(saddle)
            graph.add_edge(upper, lower, {})
            removed.append(nodes[saddle]["dna_sequence"])
    assignments = []
    for (upper, lower), group in members.items():
        for row in group:
            assignments.append(
                {
                    "dna_sequence": row["dna_sequence"],
                    "raw_fitness": float(row["scalar"]),
                    "aa_sequence": str(row["aa_sequence"]),
                    "upper_sequence": nodes[upper]["dna_sequence"],
                    "lower_sequence": nodes[lower]["dna_sequence"],
                    "location": "critical"
                    if row["dna_sequence"] == nodes[upper]["dna_sequence"]
                    else "absorbed_branch"
                    if row["dna_sequence"] in absorbed_branch
                    else "branch",
                    "absorbed_peak": row["dna_sequence"] in peak_ids,
                    "absorbed_branch_member": row["dna_sequence"] in absorbed_branch,
                }
            )
    for row in root_members:
        assignments.append(
            {
                "dna_sequence": row["dna_sequence"],
                "raw_fitness": float(row["scalar"]),
                "aa_sequence": str(row["aa_sequence"]),
                "upper_sequence": nodes[root]["dna_sequence"],
                "lower_sequence": nodes[root]["dna_sequence"],
                "location": "absorbed_root",
                "absorbed_peak": row["dna_sequence"] in peak_ids,
                "absorbed_branch_member": row["dna_sequence"] in absorbed_branch,
            }
        )
    assignments.append(
        {
            "dna_sequence": nodes[root]["dna_sequence"],
            "raw_fitness": float(nodes[root]["scalar"]),
            "aa_sequence": str(nodes[root]["aa_sequence"]),
            "upper_sequence": nodes[root]["dna_sequence"],
            "lower_sequence": nodes[root]["dna_sequence"],
            "location": "root",
            "absorbed_peak": False,
            "absorbed_branch_member": False,
        }
    )
    assignment_table = pl.DataFrame(assignments).sort("dna_sequence")
    if set(assignment_table["dna_sequence"]) != set(
        original_ids
    ) or assignment_table.height != len(original_ids):
        raise ValueError("Simplification lost or duplicated sequence identities.")
    layout = make_tree_layout(
        graph,
        scalar_attr="scalar",
        root_kind="minimum",
        node_attrs=("dna_sequence", "aa_sequence", "has_stop_codon"),
        horizontal_layout=horizontal_layout,
    )
    # Public topology API computes persistence on the small surviving tree.
    simplified = rebuild_split_tree(
        graph, vertex_attribute_cols=("scalar", "aa_sequence", "has_stop_codon")
    )
    pair_table = simplified.persistence()
    # Public vertex IDs, rather than remapped Rustworkx positions, identify pairs.
    payloads = {str(r["vertex_id"]): r for r in simplified.graph.nodes()}
    pair_rows = []
    for row in pair_table.to_dicts():
        birth = payloads[str(row["birth_label"])]
        death = payloads[str(row["death_label"])]
        pair_rows.append(
            {
                **row,
                "birth_label": birth["vertex_id"],
                "death_label": death["vertex_id"],
                "birth_has_stop_codon": bool(
                    nodes[ids[str(birth["vertex_id"])]].get("has_stop_codon", False)
                ),
            }
        )
    pairs = pl.DataFrame(pair_rows).select(
        "birth_label",
        "death_label",
        "birth_scalar",
        "death_scalar",
        "persistence",
        "birth_event",
        "death_event",
        "pair_kind",
        "birth_has_stop_codon",
    )
    wild_type = tree_analysis.wild_type_marker_table
    if not wild_type.is_empty():
        wt = wild_type.row(0, named=True)
        location = assignment_table.filter(
            pl.col("dna_sequence") == wt["dna_sequence"]
        ).row(0, named=True)
        coordinates = {r["dna_sequence"]: r for r in layout.node_table.to_dicts()}
        wt["x"] = coordinates[location["upper_sequence"]]["x"]
        wt["y"] = location["raw_fitness"]
        wild_type = pl.DataFrame([wt], schema=wild_type.schema)
    return SelectedLeafView(
        layout, pairs, assignment_table, wild_type, tuple(sorted(removed)), graph
    )


def candidate_leaf_ids(
    pairs: pl.DataFrame, contrasts: pl.DataFrame, threshold: float
) -> set[str]:
    """Select singleton peak identities from the original adjacent-saddle test."""
    selected = (
        pairs.filter(
            (pl.col("branch_vertex_count") == 1)
            & pl.col("adjacent_saddle_sequence").is_not_null()
        )
        .join(
            contrasts,
            left_on=["birth_label", "adjacent_saddle_sequence"],
            right_on=["dna_sequence", "neighbor_sequence"],
            how="inner",
        )
        .filter(pl.col("q_value") > threshold)
    )
    return set(selected["birth_label"].to_list())


def build_candidate_leaf_view(
    candidate: Any,
    contrasts: pl.DataFrame,
    threshold: float,
    *,
    horizontal_layout: str = "subtree_extremum",
) -> SelectedLeafView:
    """Build the display transformation from unchanged Candidate evidence."""
    from ..analysis.biological_annotations import _build_biological_annotations

    pairs = _build_biological_annotations(
        candidate, landscape="candidate_clamped"
    ).pairs.rename({"birth_sequence": "birth_label", "death_sequence": "death_label"})
    return absorb_singleton_leaves(
        candidate.trees["split"],
        candidate_leaf_ids(pairs, contrasts, threshold),
        horizontal_layout=horizontal_layout,
    )
