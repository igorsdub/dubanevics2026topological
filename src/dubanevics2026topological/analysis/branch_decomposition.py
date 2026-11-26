"""Branch decomposition of the smoothed Candidate split tree (exploratory stage).

The stage reads the bootstrap checkpoint (ADR 0008), builds the smoothed split
tree named by ``[branch_decomposition] tree`` in ``config/analysis.toml`` (the
band-simplified tree by default, step 2 of ADR 0008) and lists every edge of it
as one branch. It is a produce-or-load stage (ADR 0013): the checkpoint is keyed
on the stage's scientific setting, the bootstrap checkpoint's fingerprint and
artefact hashes, and the code that builds the tree and the decomposition
(Topographer's installed sources included), and it is reused whenever those
match.

Decision: a branch is one edge
------------------------------
A **branch** is one edge of the smoothed split tree, terminal or internal: the
stretch between two adjacent critical points, or between a critical point and the
root. This is the per-edge reading of the glossary's Branch span, Branch vertex
count, Hill, Absorbed branch, Kept branch and Terminal branch entries, and of
Topographer's ``build_tree_metrics().branch_table``, which has one row per tree
edge with ``branch_vertex_count`` and ``horton_strahler_order``. An elder-rule
path (a peak's whole run down to the saddle where it meets an older arm) was
tried first and set aside, because its span is ordinary H0 persistence under
another name (the project keeps branch span and H0 persistence distinct), it gives
"branch" a second meaning next to the glossary's, and its per-order histograms
are degenerate (on the band tree, one branch each at the higher orders). No
persistence is computed or exported here, and this stage computes no extended
persistence.

Columns of ``branches`` (one row per edge; identities are ``dna_sequence`` values
and no Rustworkx index leaves the module):

* ``branch_id``: the sequence of the edge's upper endpoint. It is unique, because
  every vertex of a split tree except the root has exactly one lower edge. It is
  the peak for a terminal branch and the saddle for an internal one.
* ``upper_sequence`` and ``lower_sequence``: the endpoints, the lower one the saddle
  or root where the edge ends (``branch_id`` repeats the upper one).
* ``peak_sequence``: the elder peak whose elder path runs through the edge, the
  highest peak above it. Ties between equal peaks are broken by ``vertex_index``,
  the smaller winning, exactly as ``enforce_elder_rule`` breaks them in the
  smoothing step. Topographer's persistence pairs break them by identity order
  instead; the two agree on the Candidate tree, whose ``vertex_index`` is the
  rank of the ``dna_sequence`` among the network's vertices, and a tree where
  they disagree is refused by the check against the pairs, never decomposed
  differently from them. For a terminal branch it is the upper endpoint.
* ``parent_branch_id``: the branch below, the edge whose upper endpoint is this
  edge's lower endpoint; null when the edge ends at the root.
* ``is_terminal``: the upper endpoint is a peak. ``is_root_edge``: the lower
  endpoint is the root.
* ``horton_strahler_order``: Topographer's Horton–Strahler order of the edge,
  computed on the smoothed tree itself, so absorbed hills do not raise it. It is
  1 exactly for terminal branches.
* ``member_sequences`` and ``branch_vertex_count``: the sequences of the edge after
  absorption (its upper endpoint, its interior vertices, and the sequences of
  absorbed hills that now sit on it), excluding the lower endpoint, which
  belongs to the edge below; ``branch_vertex_count`` is their number, the
  glossary's Branch vertex count. Every sequence of the tree except the root is on
  exactly one branch. Sequences absorbed into the root group (a hill whose
  adjacent saddle is the root) sit on no edge; they are reported on the root edge
  that carries the global maximum.
* ``edge_count``: the number of single-mutation adjacencies of the Full graph
  (every sequence of the dataset, before the candidate functional threshold prunes it, the graph
  the Candidate graph is built from) with both ends among the branch's members,
  the edges of the induced subgraph. A single mutation joining two sequences
  below the candidate functional threshold counts, though it is not an edge of the Candidate
  graph (that graph drops every edge whose two ends are below the threshold, which
  would leave low-fitness branches with no edges at all). It is not the vertex
  count less one: a branch whose members form a cycle, or are not all adjacent,
  differs, and an adjacency to a sequence on another branch is counted for
  neither. The stage builds the Full graph anyway (it is the Candidate graph's
  input), so no extra rebuild is needed; only its edges among the Candidate
  graph's vertices are kept.
* ``birth_fitness``, ``death_fitness`` and ``branch_span``: the raw fitness of the
  upper and the lower endpoint and their difference, the glossary's Branch span.
  Span is a different quantity from ordinary H0 persistence (the project keeps them
  apart). Its value coincides with the persistence of the branch's peak on a
  terminal branch whose peak dies at the saddle where the branch ends, and never
  on an internal branch or on the terminal branch of an elder arm, whose peak
  outlives that saddle.

``members`` is the same assignment in long format (``branch_id``, ``dna_sequence``,
``raw_fitness``, ``horton_strahler_order``): exactly the exploded
``member_sequences``.

Topographer supplies the Horton–Strahler orders (``build_tree_metrics``), which
take a ``MergeTree``, so the smoothed tree, a plain graph, is rebuilt as a split
tree over its critical vertices by ``rebuild_split_tree``, the helper absorption
uses to recompute persistence. The stage fails closed (``ValueError``) unless the
rebuilt edges equal the smoothed tree's, every sequence but the root is on exactly
one branch, and the peaks and the ends of their elder paths are those of the
smoothed tree's pairs (only the peaks and the death endpoints are compared, no
span or persistence).

``python -m dubanevics2026topological.build branch-decomposition`` runs the stage;
``load_cached_branch_decomposition`` reads it for notebooks without computing.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import rustworkx as rx
import topographer
from dubanevics2026topological.papkou2023 import Papkou2023Workflow
from topographer.topology import build_tree_metrics

from ..landscape.selected_landscape import analyze_prepared_landscape
from ..landscape.selection import (
    candidate_selection_source,
    prepare_candidate_landscape,
)
from ..provenance import upstream_identity
from ..simplification.bootstrap_smoothed_tree import (
    build_band_simplified_view,
    check_hills_match_tree,
    load_cached_bootstrap_smoothed_tree,
)
from ..simplification.selected_leaf_simplification import (
    SelectedLeafView,
    absorb_terminal_branches,
    rebuild_split_tree,
)
from ..stages import project_stage
from ..threshold.functional_threshold import (
    DECLARATION_KEY,
    configured_declaration,
    declaration_identity,
)
from . import OUTPUT, ROOT, load_settings, package_source_hashes
from .produce_or_load import (
    Checkpoint,
    Stage,
    load_checkpoint,
    produce_or_load,
    result_directory,
)

SECTION = "branch_decomposition"
STAGE_NAME = "branch-decomposition"
RETRY_HINT = "just branch-decomposition"
CHECKPOINT_DIR = OUTPUT / "analysis" / "branch_decomposition"
# The active nominal and direct band-simplified trees.
TREES = ("band", "nominal")
CORE_RETIRED = (
    "The core-rule tree was retired in ticket 11 (#136); "
    "historical core views are legacy-only."
)
SETTING_KEYS = ("tree",)
# The two endpoints of one single-mutation adjacency of the Full graph.
ADJACENCY_COLUMNS = ("left_sequence", "right_sequence")
BRANCH_SCHEMA: dict[str, Any] = {
    "branch_id": pl.String,
    "upper_sequence": pl.String,
    "lower_sequence": pl.String,
    "peak_sequence": pl.String,
    "parent_branch_id": pl.String,
    "is_terminal": pl.Boolean,
    "is_root_edge": pl.Boolean,
    "horton_strahler_order": pl.Int64,
    "branch_vertex_count": pl.Int64,
    "edge_count": pl.Int64,
    "birth_fitness": pl.Float64,
    "death_fitness": pl.Float64,
    "branch_span": pl.Float64,
    "member_sequences": pl.List(pl.String),
}
MEMBER_SCHEMA: dict[str, Any] = {
    "branch_id": pl.String,
    "dna_sequence": pl.String,
    "raw_fitness": pl.Float64,
    "horton_strahler_order": pl.Int64,
}
ARTEFACTS = (
    "branches.parquet",
    "branches.csv",
    "members.parquet",
    "members.csv",
    "summary.json",
)


@dataclass(frozen=True, slots=True)
class BranchDecomposition:
    """One row per branch (tree edge), and its members in long format.

    ``root_sequence`` is the root vertex's identity; it is a member of no branch.
    ``root_group_members`` counts the sequences that smoothing absorbed into the
    root group and that the root edge of the global maximum carries.
    """

    branches: pl.DataFrame
    members: pl.DataFrame
    root_sequence: str
    root_group_members: int


@dataclass(frozen=True, slots=True)
class BranchDecompositionResult:
    """The published tables of the stage with its summary and provenance record."""

    branches: pl.DataFrame
    members: pl.DataFrame
    summary: dict[str, Any]
    provenance: dict[str, Any]
    reused: bool


def branch_decomposition_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``[branch_decomposition]``; return the settings the stage records."""
    unknown = sorted(set(config) - set(SETTING_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown [{SECTION}] settings {unknown}; known: {list(SETTING_KEYS)}."
        )
    if "tree" not in config:
        raise ValueError(f"[{SECTION}] needs `tree`, one of {list(TREES)}.")
    tree = config["tree"]
    if tree == "core":
        raise ValueError(CORE_RETIRED)
    if not isinstance(tree, str) or tree not in TREES:
        raise ValueError(
            f"[{SECTION}] tree must be one of {list(TREES)}, got {tree!r}."
        )
    return {"tree": tree}


def decomposition_view(
    tree: str, tree_analysis: Any, hills: pl.DataFrame
) -> SelectedLeafView:
    """Build the smoothed tree the setting names, through the one path figures use."""
    if tree == "band":
        return build_band_simplified_view(tree_analysis, hills)
    if tree == "core":
        raise ValueError(CORE_RETIRED)
    if tree == "nominal":
        return absorb_terminal_branches(tree_analysis, set())
    raise ValueError(f"tree must be one of {list(TREES)}, got {tree!r}.")


def network_sequences(network: Any) -> set[str]:
    """The ``dna_sequence`` identities of the vertices of a ``ScalarNetwork``."""
    graph = network.graph
    return {str(graph[i]["dna_sequence"]) for i in graph.node_indices()}


def network_adjacency(network: Any, among: Collection[str]) -> pl.DataFrame:
    """The mutation edges of a ``ScalarNetwork`` with both ends in ``among``.

    Pairs of sequence identities, never Rustworkx indices. The Full graph has
    millions of edges, so the filter runs on the indices and only the kept edges
    become strings.
    """
    graph = network.graph
    wanted = set(among)
    identities = {int(i): str(graph[i]["dna_sequence"]) for i in graph.node_indices()}
    kept = {index for index, name in identities.items() if name in wanted}
    return pl.DataFrame(
        [
            (identities[a], identities[b])
            for a, b in graph.edge_list()
            if a in kept and b in kept
        ],
        schema=dict.fromkeys(ADJACENCY_COLUMNS, pl.String),
        orient="row",
    )


def build_candidate_inputs(settings: dict[str, Any]) -> tuple[Any, pl.DataFrame]:
    """The Candidate split tree and the Full graph's adjacencies among its sequences.

    The Full graph is built once: the Candidate graph is made from it by
    ``prepare_clamped_landscape`` (which drops every edge with both ends below
    ``candidate_functional_threshold`` and keeps the largest component), the split tree is built
    from that, and the Full graph's own single-mutation adjacencies among the
    Candidate graph's vertices are kept for ``edge_count``. The steps are those of
    ``candidate_tree_contrast.build_candidate_network`` and ``build_candidate_tree``,
    repeated here because those functions drop the Full graph and editing them would
    change the fingerprints of every stage that hashes their file;
    ``check_hills_match_tree`` refuses
    a tree that differs from the bootstrap checkpoint's.
    """
    scientific = settings["analysis"]
    wt = pl.read_parquet(ROOT / scientific["dataset"]["fitness_wt"])
    diff = pl.read_parquet(ROOT / scientific["dataset"]["fitness_diff"])
    full = Papkou2023Workflow(
        landscape="full", fitness_wt=wt, fitness_diff=diff
    ).build_network()
    prepared, _ = prepare_candidate_landscape(full, settings)
    analysis = analyze_prepared_landscape(
        prepared,
        name="candidate_clamped",
        wild_type_sequence=str(wt.get_column("dna_from")[0]),
    )
    adjacency = network_adjacency(full, network_sequences(prepared.network))
    return analysis.trees["split"], adjacency


def _identities(graph: Any) -> dict[int, str]:
    """Map each node of ``graph`` to its ``dna_sequence``; refuse missing or repeated."""
    ids = {int(i): graph[i].get("dna_sequence") for i in graph.node_indices()}
    if not all(isinstance(value, str) and value for value in ids.values()):
        raise ValueError("Smoothed tree nodes must carry a dna_sequence identity.")
    if len(set(ids.values())) != len(ids):
        raise ValueError(
            "Smoothed tree nodes must have distinct dna_sequence identities."
        )
    return {i: str(value) for i, value in ids.items()}


def _vertex_indices(graph: Any, names: Mapping[int, str]) -> dict[str, int]:
    """The ``vertex_index`` of each sequence, which orders ties between equal peaks."""
    found = {names[int(i)]: graph[i].get("vertex_index") for i in graph.node_indices()}
    missing = sorted(name for name, value in found.items() if value is None)
    if missing:
        raise ValueError(
            "Smoothed tree nodes lack a vertex_index for elder-rule ties: "
            + ", ".join(missing[:20])
        )
    indices = {name: int(value) for name, value in found.items()}
    if len(set(indices.values())) != len(indices):
        raise ValueError(
            "Smoothed tree nodes must have distinct vertex_index values for "
            "elder-rule ties."
        )
    return indices


def _elder_peaks(
    graph: Any,
    names: Mapping[int, str],
    fitness: Mapping[str, float],
    vertex_index: Mapping[str, int],
) -> dict[str, str]:
    """Map each vertex to the highest peak at or above it.

    At a saddle the arm whose subtree holds the highest peak is the elder arm; a
    tie is broken by the smaller ``vertex_index``, as ``enforce_elder_rule`` does.
    """
    best: dict[int, tuple[float, int, str]] = {}
    for i in rx.topological_sort(graph):
        name = names[int(i)]
        arms = [best[p] for p in graph.predecessor_indices(i)]
        best[i] = min(arms) if arms else (-fitness[name], vertex_index[name], name)
    return {names[int(i)]: key[2] for i, key in best.items()}


def _edge_members(
    assignments: pl.DataFrame, edges: set[tuple[str, str]], root: str
) -> tuple[dict[tuple[str, str], list[tuple[str, float]]], list[tuple[str, float]]]:
    """Group the assigned sequences by tree edge; return the root group apart.

    The root group (upper = lower = root) holds the root vertex and the
    sequences absorbed into it; the root vertex is dropped here, so what is
    returned for it are the absorbed sequences only.
    """
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {e: [] for e in edges}
    root_group: list[tuple[str, float]] = []
    for row in assignments.select(
        "dna_sequence", "raw_fitness", "upper_sequence", "lower_sequence"
    ).iter_rows(named=True):
        key = (str(row["upper_sequence"]), str(row["lower_sequence"]))
        item = (str(row["dna_sequence"]), float(row["raw_fitness"]))
        if key in grouped:
            grouped[key].append(item)
        elif key == (root, root):
            if item[0] != root:
                root_group.append(item)
        else:
            raise ValueError(
                f"Sequence {item[0]!r} is assigned to {key[0]!r} -> {key[1]!r}, "
                "which is not an edge of the smoothed tree."
            )
    empty = sorted(f"{u}|{l}" for (u, l), rows in grouped.items() if not rows)
    if empty:
        raise ValueError(
            "Smoothed tree edges carry no sequence: " + ", ".join(empty[:20])
        )
    return grouped, root_group


def _induced_edge_counts(
    members: pl.DataFrame, adjacency: pl.DataFrame, tree_sequences: set[str]
) -> dict[str, int]:
    """Count, per branch, the network edges with both ends among its members.

    ``adjacency`` is refused when it names a sequence the tree does not have: the
    stage keeps only the Full graph's edges among the Candidate graph's vertices,
    which are the tree's sequences, so a stranger means the two came from
    different builds. An edge to the root counts for no branch, the root being on
    none.
    """
    absent = set(ADJACENCY_COLUMNS) - set(adjacency.columns)
    if absent:
        raise ValueError(f"The Full graph adjacency lacks columns {sorted(absent)}.")
    left, right = ADJACENCY_COLUMNS
    pairs = (
        adjacency.select(left, right)
        .filter(pl.col(left) != pl.col(right))
        .with_columns(
            pl.when(pl.col(left) < pl.col(right))
            .then(pl.col(left))
            .otherwise(pl.col(right))
            .alias(left),
            pl.when(pl.col(left) < pl.col(right))
            .then(pl.col(right))
            .otherwise(pl.col(left))
            .alias(right),
        )
        .unique()
    )
    known = sorted(tree_sequences)
    strangers = pairs.filter(~pl.col(left).is_in(known) | ~pl.col(right).is_in(known))
    if strangers.height:
        raise ValueError(
            "The Full graph adjacency has edges to sequences outside the tree, e.g. "
            f"{strangers.row(0)!r}; the adjacency and the tree come from different "
            "builds."
        )
    owner = members.select("dna_sequence", "branch_id")
    inside = (
        pairs.join(
            owner.rename({"dna_sequence": left, "branch_id": "left_branch"}),
            on=left,
            how="inner",
            validate="m:1",
        )
        .join(
            owner.rename({"dna_sequence": right, "branch_id": "right_branch"}),
            on=right,
            how="inner",
            validate="m:1",
        )
        .filter(pl.col("left_branch") == pl.col("right_branch"))
        .group_by("left_branch")
        .agg(pl.len().cast(pl.Int64).alias("edge_count"))
    )
    return dict(inside.iter_rows())


def decompose_view(
    view: SelectedLeafView, adjacency: pl.DataFrame
) -> BranchDecomposition:
    """List every edge of the smoothed tree of ``view`` as a branch.

    ``adjacency`` holds the single-mutation adjacencies of the Full graph among
    the tree's sequences (``left_sequence``, ``right_sequence``), for
    ``edge_count``. Fails closed
    (``ValueError``) unless the tree has one root and one lower edge per other
    vertex, the rebuilt split tree has the smoothed tree's edges, every sequence
    but the root belongs to exactly one branch, and the peaks and the ends of
    their elder paths are those of ``view.pairs``.
    """
    graph = view.tree
    names = _identities(graph)
    fitness = {names[int(i)]: float(graph[i]["scalar"]) for i in graph.node_indices()}
    roots = [names[int(i)] for i in graph.node_indices() if graph.out_degree(i) == 0]
    if len(roots) != 1:
        raise ValueError(f"The smoothed tree needs one root, found {len(roots)}.")
    root = roots[0]
    forked = sorted(
        names[int(i)] for i in graph.node_indices() if graph.out_degree(i) > 1
    )
    if forked:
        raise ValueError(
            "Smoothed tree vertices need exactly one lower neighbour: "
            + ", ".join(forked[:20])
        )
    tree_edges = {(names[int(u)], names[int(v)]) for u, v in graph.edge_list()}
    lower_of = dict(tree_edges)
    vertex_index = _vertex_indices(graph, names)

    merge_tree = rebuild_split_tree(graph)
    rebuilt = merge_tree.graph
    rebuilt_names = {
        int(i): str(rebuilt[i]["vertex_id"]) for i in rebuilt.node_indices()
    }
    # MergeTree edges run from the root to the leaves: (lower, upper).
    rebuilt_edges = {
        (rebuilt_names[int(upper)], rebuilt_names[int(lower)])
        for lower, upper in rebuilt.edge_list()
    }
    if rebuilt_edges != tree_edges:
        differing = sorted(f"{u}|{l}" for u, l in rebuilt_edges ^ tree_edges)
        raise ValueError(
            "The split tree rebuilt from the smoothed tree differs from it in "
            "edges " + ", ".join(differing[:20]) + "."
        )
    order_at = {
        rebuilt_names[int(source)]: int(order)
        for source, order in build_tree_metrics(merge_tree)
        .branch_table.select("source_tree_node_index", "horton_strahler_order")
        .iter_rows()
    }
    peak_of = _elder_peaks(graph, names, fitness, vertex_index)
    terminal = {names[int(i)] for i in graph.node_indices() if graph.in_degree(i) == 0}
    _check_against_pairs(view, peak_of, lower_of, root, terminal)

    grouped, root_group = _edge_members(view.assignments, tree_edges, root)
    if root_group:
        # A hill absorbed straight into the root sits on no edge; the root edge
        # that carries the global maximum takes it.
        carrier = next(
            edge
            for edge in sorted(tree_edges)
            if edge[1] == root and peak_of[edge[0]] == peak_of[root]
        )
        grouped[carrier] = [*grouped[carrier], *root_group]
    rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for upper, lower in sorted(tree_edges):
        found = sorted(grouped[(upper, lower)], key=lambda item: (-item[1], item[0]))
        order = order_at[upper]
        rows.append(
            {
                "branch_id": upper,
                "upper_sequence": upper,
                "lower_sequence": lower,
                "peak_sequence": peak_of[upper],
                "parent_branch_id": None if lower == root else lower,
                "is_terminal": upper in terminal,
                "is_root_edge": lower == root,
                "horton_strahler_order": order,
                "branch_vertex_count": len(found),
                "edge_count": 0,
                "birth_fitness": fitness[upper],
                "death_fitness": fitness[lower],
                "branch_span": fitness[upper] - fitness[lower],
                "member_sequences": [name for name, _ in found],
            }
        )
        long_rows.extend(
            {
                "branch_id": upper,
                "dna_sequence": name,
                "raw_fitness": value,
                "horton_strahler_order": order,
            }
            for name, value in found
        )
    members = pl.DataFrame(long_rows, schema=MEMBER_SCHEMA).sort(
        ["branch_id", "raw_fitness", "dna_sequence"], descending=[False, True, False]
    )
    _check_partition(view, members, root)
    counts = _induced_edge_counts(
        members, adjacency, set(view.assignments["dna_sequence"])
    )
    for row in rows:
        row["edge_count"] = counts.get(row["branch_id"], 0)
    branches = pl.DataFrame(rows, schema=BRANCH_SCHEMA).sort(
        ["birth_fitness", "branch_id"], descending=[True, False]
    )
    return BranchDecomposition(branches, members, root, len(root_group))


def _check_partition(view: SelectedLeafView, members: pl.DataFrame, root: str) -> None:
    """Every assigned sequence except the root belongs to exactly one branch."""
    expected = set(view.assignments["dna_sequence"]) - {root}
    if members.height != len(expected) or set(members["dna_sequence"]) != expected:
        raise ValueError(
            "Branch members are not a partition of the tree's sequences without "
            "the root."
        )


def _check_against_pairs(
    view: SelectedLeafView,
    peak_of: Mapping[str, str],
    lower_of: Mapping[str, str],
    root: str,
    terminal: set[str],
) -> None:
    """The peaks and the ends of their elder paths are those of ``view.pairs``.

    The pairs are computed by Topographer on the surviving tree when it is
    absorbed, with its own tie-break. Only the peaks and the death endpoints are
    compared (the elder path of a peak ends at the saddle, or the root, below its
    lowest edge), never a span or a persistence value: a branch is one edge, and
    its span is a different quantity from a pair's persistence.
    """
    pairs = {
        str(row["birth_label"]): str(row["death_label"])
        for row in view.pairs.select("birth_label", "death_label").iter_rows(named=True)
    }
    if terminal != set(pairs):
        raise ValueError(
            "Branch peaks differ from the peaks of the smoothed tree's pairs: "
            + ", ".join(sorted(terminal ^ set(pairs))[:20])
            + "."
        )
    for peak in sorted(terminal):
        node = peak
        while lower_of[node] != root and peak_of[lower_of[node]] == peak:
            node = lower_of[node]
        end = lower_of[node]
        if pairs[peak] != end:
            raise ValueError(
                f"The elder path of peak {peak!r} ends at {end!r}, but the smoothed "
                f"tree pairs it with {pairs[peak]!r}."
            )


def order_summary(branches: pl.DataFrame) -> pl.DataFrame:
    """One row per Horton–Strahler order present, with its branch counts and sizes.

    ``sequence_count`` is the number of sequences on the branches of that order;
    it sums to every sequence but the root over all orders.
    """
    return (
        branches.group_by("horton_strahler_order")
        .agg(
            pl.len().cast(pl.Int64).alias("branch_count"),
            pl.col("branch_vertex_count").median().alias("median_branch_vertex_count"),
            pl.col("branch_vertex_count").max().alias("maximum_branch_vertex_count"),
            pl.col("branch_vertex_count").sum().alias("sequence_count"),
        )
        .sort("horton_strahler_order")
    )


def decomposition_summary(
    decomposition: BranchDecomposition, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """The counts the stage records beside its tables (``summary.json``)."""
    branches = decomposition.branches
    terminal = int(branches["is_terminal"].sum())
    return {
        "settings": dict(settings),
        "branch_count": branches.height,
        "terminal_branch_count": terminal,
        "internal_branch_count": branches.height - terminal,
        "branches_per_order": {
            str(order): count
            for order, count in order_summary(branches)
            .select("horton_strahler_order", "branch_count")
            .iter_rows()
        },
        "maximum_order": int(branches["horton_strahler_order"].max()),
        "sequence_count": decomposition.members.height + 1,
        "member_count": decomposition.members.height,
        "root_sequence": decomposition.root_sequence,
        "root_edge_count": int(branches["is_root_edge"].sum()),
        "root_group_member_count": decomposition.root_group_members,
    }


def describe(result: BranchDecompositionResult) -> str:
    """One line for the command line: what was decomposed and how it came out."""
    info = result.summary
    return (
        f"Branch decomposition of the {info['settings']['tree']} tree "
        f"{'reused' if result.reused else 'produced'}: {info['branch_count']} branches "
        f"({info['terminal_branch_count']} terminal, "
        f"{info['internal_branch_count']} internal), "
        f"orders {info['branches_per_order']}, {info['sequence_count']} sequences."
    )


def _write_tables(
    directory: Path, decomposition: BranchDecomposition, summary: Mapping[str, Any]
) -> None:
    """Write Parquet (lists kept) and CSV (lists joined by ``;``) plus the summary."""
    branches, members = decomposition.branches, decomposition.members
    branches.write_parquet(directory / "branches.parquet")
    branches.with_columns(pl.col("member_sequences").list.join(";")).write_csv(
        directory / "branches.csv"
    )
    members.write_parquet(directory / "members.parquet")
    members.write_csv(directory / "members.csv")
    (directory / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _stage(
    section: Mapping[str, Any], bootstrap: Any, settings: Mapping[str, Any]
) -> Stage:
    """Declare what the checkpoint depends on, through the project's stage factory."""
    analysis_dir = Path(__file__).resolve().parent
    landscape_dir = analysis_dir.parent / "landscape"
    simplification_dir = analysis_dir.parent / "simplification"
    return project_stage(
        STAGE_NAME,
        settings={
            SECTION: dict(section),
            "bootstrap_checkpoint": upstream_identity(bootstrap.provenance),
            # The Candidate tree is selected at the computed threshold: the rule and
            # the digits. The rounded value is not keyed: it reads data/raw, which a
            # HPC-fetched checkout lacks; threshold implementation is hashed below.
            DECLARATION_KEY: declaration_identity(configured_declaration(settings)),
            # A re-run classification changes the Functional vertex set.
            "candidate_selection_source": candidate_selection_source(settings),
            # The effective Topographer sources (the split-tree rebuild and the
            # orders), so an editable install invalidates too; uv.lock alone does
            # not change under `uv run --with-editable`.
            "topographer_sources": package_source_hashes(topographer),
        },
        code=(
            Path(__file__).resolve(),
            simplification_dir / "__init__.py",
            simplification_dir / "bootstrap_smoothed_tree.py",
            simplification_dir / "bootstrap_chunks.py",
            simplification_dir / "checkpoint_identity.py",
            simplification_dir / "identities.py",
            # build_candidate_inputs repeats the recipe of this file's
            # build_candidate_network, so an edit to either invalidates.
            landscape_dir / "selection.py",
            analysis_dir.parent / "threshold" / "functional_threshold.py",
            analysis_dir / "candidate_tree_contrast.py",
            landscape_dir / "selected_landscape.py",
            landscape_dir / "full_population.py",
            simplification_dir / "selected_leaf_simplification.py",
            ROOT / "uv.lock",
        ),
        root=ROOT,
    )


def _read_result(checkpoint: Checkpoint) -> BranchDecompositionResult:
    """Read a verified checkpoint's tables; call it while the lock is held."""
    directory = checkpoint.path
    return BranchDecompositionResult(
        branches=pl.read_parquet(directory / "branches.parquet"),
        members=pl.read_parquet(directory / "members.parquet"),
        summary=json.loads((directory / "summary.json").read_text(encoding="utf-8")),
        provenance=checkpoint.provenance,
        reused=checkpoint.reused,
    )


def compute_branch_decomposition(
    settings: dict[str, Any] | None = None,
    *,
    force: bool = False,
    checkpoint_dir: Path | None = None,
) -> BranchDecompositionResult:
    """Produce the decomposition, or load the checkpoint whose fingerprint matches.

    The bootstrap checkpoint is only read (``load_cached_bootstrap_smoothed_tree``
    never draws), and the Candidate tree is built only when a checkpoint has to be
    produced. ``force`` reproduces even when a matching checkpoint exists. The
    tables are read under the checkpoint lock, so a concurrent publish cannot mix
    two checkpoints.
    """
    settings = settings or load_settings()
    section = branch_decomposition_settings(settings["analysis"][SECTION])
    bootstrap = load_cached_bootstrap_smoothed_tree(settings)

    def produce(directory: Path) -> None:
        tree_analysis, adjacency = build_candidate_inputs(settings)
        check_hills_match_tree(bootstrap.hills, tree_analysis)
        view = decomposition_view(section["tree"], tree_analysis, bootstrap.hills)
        decomposition = decompose_view(view, adjacency)
        _write_tables(
            directory, decomposition, decomposition_summary(decomposition, section)
        )

    read: list[BranchDecompositionResult] = []
    stage = _stage(section, bootstrap, settings)
    produce_or_load(
        stage,
        produce,
        checkpoint_dir=result_directory(checkpoint_dir or CHECKPOINT_DIR, stage),
        force=force,
        on_ready=lambda ready: read.append(_read_result(ready)),
    )
    return read[0]


def load_cached_branch_decomposition(
    settings: dict[str, Any] | None = None, *, checkpoint_dir: Path | None = None
) -> BranchDecompositionResult:
    """Read the checkpoint for the configured settings; never compute or recover.

    The branch checkpoint is read by ``load_checkpoint``, which only reads: it
    rejects a checkpoint left mid-publication (``RuntimeError``), refuses one that
    does not match the configured settings, the bootstrap checkpoint and the code
    (``FileNotFoundError``, so a stale one is never shown), verifies the artefact
    hashes, reads the tables under the lock, and moves and removes nothing of the
    branch checkpoint. The reader as a whole is not side-effect free: to learn the
    bootstrap checkpoint's identity it first calls
    ``load_cached_bootstrap_smoothed_tree``, which, as ADR 0008 allows, restores
    a processed checkpoint into ``output/`` and moves a non-matching bootstrap
    checkpoint in ``output/`` aside to ``.replaced-*`` (it never draws). That
    loader also holds the bootstrap checkpoint lock and checks the working slot
    against the published checkpoint (ADR 0013, Phase 2), so it raises
    ``RuntimeError`` while a bootstrap run is drawing, while its publication is
    unsettled, or when the slot lags the published checkpoint.
    """
    settings = settings or load_settings()
    section = branch_decomposition_settings(settings["analysis"][SECTION])
    bootstrap = load_cached_bootstrap_smoothed_tree(settings)
    stage = _stage(section, bootstrap, settings)
    return load_checkpoint(
        stage,
        _read_result,
        checkpoint_dir=result_directory(checkpoint_dir or CHECKPOINT_DIR, stage),
        retry_hint=RETRY_HINT,
    )
