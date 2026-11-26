"""Hand-built nominal trees, Full-graph adjacencies and hill tables for the branch-decomposition tests."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from types import SimpleNamespace

import polars as pl
import rustworkx as rx

EVENTS = {"leaf": "maximum", "saddle": "split_saddle", "root": "minimum"}


def chain_adjacency(
    edges: Sequence[tuple[str, str]],
    regular: Sequence[tuple[str, str, str, float]] = (),
    extra: Sequence[tuple[str, str]] = (),
) -> pl.DataFrame:
    """The single-mutation adjacencies of a tree: each edge is a path through its regular vertices.

    An edge ``(upper, lower)`` with regular vertices ``r1, r2`` (in the order of
    ``regular``) becomes ``upper - r1 - r2 - lower``; ``extra`` adds chords.
    """
    on_edge: dict[tuple[str, str], list[str]] = {}
    for upper, lower, dna, _ in regular:
        on_edge.setdefault((upper, lower), []).append(dna)
    pairs = []
    for upper, lower in edges:
        path = [upper, *on_edge.get((upper, lower), []), lower]
        pairs.extend(pairwise(path))
    pairs.extend(extra)
    return pl.DataFrame(
        pairs,
        schema={"left_sequence": pl.String, "right_sequence": pl.String},
        orient="row",
    )


def make_tree(
    nodes: Sequence[tuple[str, float, str]],
    edges: Sequence[tuple[str, str]],
    regular: Sequence[tuple[str, str, str, float]] = (),
    *,
    reverse: bool = False,
    extra_adjacency: Sequence[tuple[str, str]] = (),
) -> SimpleNamespace:
    """A nominal split tree as the stage reads it: ``tree``, member table, ``adjacency``.

    ``nodes`` are ``(dna_sequence, fitness, kind)`` with kind ``leaf``, ``saddle`` or
    ``root``; ``edges`` are ``(upper, lower)``; ``regular`` are
    ``(upper, lower, dna_sequence, fitness)`` vertices lying on an edge. The
    ``vertex_index`` of a node is its position in ``nodes``, so ``reverse=True``
    inserts the nodes in the opposite order (other Rustworkx indices) without
    changing any identity or tie-break. ``adjacency`` is the Full graph's
    single-mutation adjacencies among the tree's sequences (``chain_adjacency``),
    which the decomposition counts edges in. List ``nodes`` in
    alphabetical order of identity to make ``vertex_index`` the identity rank, as it
    is on the Candidate tree.
    """
    graph = rx.PyDiGraph()
    order = list(reversed(range(len(nodes)))) if reverse else range(len(nodes))
    index = {}
    for position in order:
        dna, fitness, kind = nodes[position]
        index[dna] = graph.add_node(
            {
                "vertex_index": position,
                "dna_sequence": dna,
                "aa_sequence": "A",
                "scalar": fitness,
                "event": EVENTS[kind],
                "has_stop_codon": False,
            }
        )
    counts: dict[tuple[str, str], int] = {}
    for upper, lower, _, _ in regular:
        counts[(upper, lower)] = counts.get((upper, lower), 0) + 1
    for upper, lower in edges:
        graph.add_edge(
            index[upper], index[lower], {"regular_count": counts.get((upper, lower), 0)}
        )
    schema = {
        "source_tree_node_index": pl.Int64,
        "target_tree_node_index": pl.Int64,
        "dna_sequence": pl.String,
        "aa_sequence": pl.String,
        "scalar": pl.Float64,
    }
    table = pl.DataFrame(
        [
            {
                "source_tree_node_index": index[upper],
                "target_tree_node_index": index[lower],
                "dna_sequence": dna,
                "aa_sequence": "A",
                "scalar": fitness,
            }
            for upper, lower, dna, fitness in regular
        ],
        schema=schema,
    )
    return SimpleNamespace(
        tree=graph,
        branch_member_table=table,
        wild_type_marker_table=pl.DataFrame(),
        adjacency=chain_adjacency(edges, regular, extra_adjacency),
    )


BALANCED_NODES = [
    ("a", 9.0, "leaf"),
    ("b", 8.0, "leaf"),
    ("c", 7.0, "leaf"),
    ("d", 6.0, "leaf"),
    ("s1", 5.0, "saddle"),
    ("s2", 4.0, "saddle"),
    ("s3", 3.0, "saddle"),
    ("root", 0.0, "root"),
]
BALANCED_EDGES = [
    ("a", "s1"),
    ("b", "s1"),
    ("c", "s2"),
    ("d", "s2"),
    ("s1", "s3"),
    ("s2", "s3"),
    ("s3", "root"),
]
# ra lies on a's edge; rb1 and rb2 on b's.
BALANCED_REGULAR = [
    ("a", "s1", "ra", 8.5),
    ("b", "s1", "rb1", 7.5),
    ("b", "s1", "rb2", 6.5),
]
# Ordinary persistence of each peak of the balanced tree (its fitness minus its
# death saddle's; the global maximum's pair closes at the root).
BALANCED_PERSISTENCE = {"a": 9.0, "b": 3.0, "c": 4.0, "d": 2.0}


# Two chords of the balanced tree's adjacencies: a-s1 lies inside branch a once b is
# absorbed (a cycle there), rb1-c joins two branches and so is counted for neither.
BALANCED_CHORDS = [("a", "s1"), ("rb1", "c")]


def balanced_tree(*, reverse: bool = False) -> SimpleNamespace:
    """Four peaks, two forks and a joining saddle, with regular vertices on a and b.

    Horton–Strahler orders of its tree edges: the four peak edges 1, the edges below
    s1 and s2 2, and the edge below s3 (to the root) 3.
    """
    return make_tree(
        BALANCED_NODES,
        BALANCED_EDGES,
        BALANCED_REGULAR,
        reverse=reverse,
        extra_adjacency=BALANCED_CHORDS,
    )


def hills_table(
    tree: SimpleNamespace,
    persistence: dict[str, float],
    *,
    core_absorbed: Sequence[str] = (),
    two_c: float,
) -> pl.DataFrame:
    """The columns of the bootstrap ``hills`` table that the smoothed views read."""
    regular: dict[str, int] = {}
    for row in tree.branch_member_table.iter_rows(named=True):
        upper = tree.tree[row["source_tree_node_index"]]["dna_sequence"]
        regular[upper] = regular.get(upper, 0) + 1
    peaks = sorted(persistence)
    clears = [persistence[peak] > two_c for peak in peaks]
    return pl.DataFrame(
        {
            "peak_sequence": peaks,
            "outcome": ["absorbed" if p in core_absorbed else "kept" for p in peaks],
            "member_count": [regular.get(p, 0) + 1 for p in peaks],
            "nominal_persistence": [persistence[p] for p in peaks],
            "clears_band": clears,
            "band_outcome": ["kept" if keep else "absorbed" for keep in clears],
        }
    )
