# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""Tree helpers for leaf co-occurrence: terminal branches, diagrams, distances and cores.

These are the pieces `LeafCooccurrence` and the deprecated `bootstrap_terminal_cooccurrence`
share. See ADR 0007 for the statistics and ADR 0008 for the chunked layout.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import numbers

import numpy as np
import polars as pl
import rustworkx as rx
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching

from topographer.graph import ScalarNetwork
from topographer.topology import MergeTree

__all__ = ["extract_cores", "terminal_branch_members"]

_PEAK_EVENT = {"split": "maximum", "join": "minimum"}
_ROLE_LEAF, _ROLE_SADDLE, _ROLE_ROOT = 0, 1, 2


# --- tree helpers (shared by the per-draw trees and the public MergeTree wrapper) -------------


def _find_root(tree: rx.PyDiGraph, *, root_to_leaf: bool) -> int:
    """Return the unique node without a parent."""
    degree = tree.in_degree if root_to_leaf else tree.out_degree
    roots = [int(node) for node in tree.node_indices() if degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(
            f"Merge tree must have exactly one root, found {len(roots)}; "
            "the network must be connected."
        )
    return roots[0]


def _terminal_branches(
    tree: rx.PyDiGraph, *, peak_event: str, root_to_leaf: bool
) -> dict[int, list[int]]:
    """Return ``{leaf extremum vertex: sorted member vertices}`` for a reduced merge tree.

    A terminal branch is a tree edge whose leaf-side node has the ``peak_event``. Its members
    are that extremum plus the regular vertices contracted into the edge (the interior of the
    edge's ``vertex_chain_json``); the root-side endpoint, a saddle or the root, is not a
    member. The same rule serves the leaves-to-root trees of the draws
    (``root_to_leaf=False``) and ``MergeTree.graph`` (``root_to_leaf=True``).
    """
    branches: dict[int, list[int]] = {}
    for first, second, edge in tree.weighted_edge_list():
        leaf = second if root_to_leaf else first
        payload = tree.get_node_data(leaf)
        if payload.get("event") != peak_event:
            continue
        members = [int(payload["vertex_index"])]
        chain_json = edge.get("vertex_chain_json") if edge else None
        if chain_json:
            members.extend(int(node["vertex_index"]) for node in json.loads(chain_json)[1:-1])
        branches[members[0]] = sorted(members)
    return branches


def _critical_roles(
    tree: rx.PyDiGraph, *, peak_event: str, root_to_leaf: bool
) -> tuple[list[int], list[int], int]:
    """Return ``(peak vertices, saddle vertices, root vertex)`` of a reduced merge tree.

    The root takes priority over a peak, and a peak over a saddle, so each vertex is listed
    once.
    """
    root_node = _find_root(tree, root_to_leaf=root_to_leaf)
    peaks: list[int] = []
    saddles: list[int] = []
    for node in tree.node_indices():
        if int(node) == root_node:
            continue
        payload = tree.get_node_data(node)
        event = str(payload.get("event", "regular"))
        if event == peak_event:
            peaks.append(int(payload["vertex_index"]))
        elif "saddle" in event:
            saddles.append(int(payload["vertex_index"]))
    return peaks, saddles, int(tree.get_node_data(root_node)["vertex_index"])


@dataclass(frozen=True)
class _Diagram:
    """H0 persistence pairs of one reduced tree, the global pair included.

    ``points`` is a ``(k, 2)`` float64 array of ``(birth, death)`` scalars, and ``pairs`` holds
    the matching ``(birth vertex, death vertex)`` network vertex indices.
    """

    points: np.ndarray
    pairs: frozenset[tuple[int, int]]


def _persistence_diagram(tree: rx.PyDiGraph, *, tree_kind: str, root_to_leaf: bool) -> _Diagram:
    """Return the elder-rule H0 pairs of a reduced merge tree plus the global pair.

    This is the lean equivalent of ``build_tree_extended_persistence_pair_table`` for trees
    without a ``label`` attribute: sweep from the leaves, let each node inherit the elder
    birth of its children (higher scalar in a split tree, lower in a join tree, ties to the
    smaller vertex index) and let the others die at a node with several children. The elder
    birth that reaches the root is paired with the root scalar.
    """
    sign = -1.0 if tree_kind == "split" else 1.0
    scalar: dict[int, float] = {}
    vertex: dict[int, int] = {}
    for node in tree.node_indices():
        payload = tree.get_node_data(node)
        scalar[int(node)] = float(payload["scalar"])
        vertex[int(node)] = int(payload["vertex_index"])
    children: dict[int, list[int]] = {node: [] for node in scalar}
    for first, second in tree.edge_list():
        leaf, parent = (second, first) if root_to_leaf else (first, second)
        children[int(parent)].append(int(leaf))

    order = [int(node) for node in rx.topological_sort(tree)]
    if root_to_leaf:
        order.reverse()
    elder: dict[int, int] = {}
    births: list[int] = []
    deaths: list[int] = []
    for node in order:
        below = children[node]
        if not below:
            elder[node] = node
            continue
        candidates = sorted({elder[child] for child in below})
        winner = min(candidates, key=lambda c: (sign * scalar[c], vertex[c]))
        for loser in candidates:
            if loser != winner:
                births.append(loser)
                deaths.append(node)
        elder[node] = winner
    root = order[-1]
    births.append(elder[root])
    deaths.append(root)
    points = np.array([[scalar[b], scalar[d]] for b, d in zip(births, deaths)], dtype=np.float64)
    pairs = frozenset((vertex[b], vertex[d]) for b, d in zip(births, deaths))
    return _Diagram(points=points.reshape(-1, 2), pairs=pairs)


def terminal_branch_members(tree: MergeTree) -> dict[int, list[int]]:
    """Return the members of every terminal branch of a merge tree.

    A terminal branch is a reduced-tree edge whose leaf-side node is a leaf extremum: a maximum
    for a split tree, a minimum for a join tree. Its members are that extremum and the regular
    vertices on the edge. The lower endpoint (the saddle or root where the branch ends) is
    never a member, so a saddle or the root belongs to no terminal branch. A path whose
    maximum's edge goes straight to the root has one terminal branch that excludes the root.

    Parameters
    ----------
    tree : MergeTree
        Merge tree of either kind.

    Returns
    -------
    dict[int, list[int]]
        Maps the vertex index of each leaf extremum to the sorted vertex indices of its branch
        members, the extremum included. Vertex indices are Rustworkx node indices of the
        network the tree was built from.
    """
    return _terminal_branches(tree.graph, peak_event=_PEAK_EVENT[tree.kind], root_to_leaf=True)


# --- persistence-diagram distances ------------------------------------------------------------


def _persistence_of(points: np.ndarray) -> np.ndarray:
    return np.abs(points[:, 1] - points[:, 0])


def _wasserstein_1(first: np.ndarray, second: np.ndarray) -> float:
    """Return the 1-Wasserstein distance between two ``(k, 2)`` diagrams of (birth, death).

    The ground metric between points is L1, ``|b1 - b2| + |d1 - d2|``; a point goes to the
    diagonal at cost ``|d - b|`` (its L1 distance to the diagonal), and diagonal-to-diagonal
    costs 0. The spec's optimal matching is a linear assignment on the ``(n + m) x (n + m)``
    matrix with a diagonal copy for every point of the other diagram. It reduces exactly to the
    rectangular ``n x m`` form used here: with ``p_i`` and ``q_j`` the diagonal costs and
    ``c_ij`` the point-to-point costs, matching ``i`` with ``j`` saves ``p_i + q_j - c_ij``
    over parking both on the diagonal, so
    ``W1 = sum(p) + sum(q) + min over assignments of sum(min(c_ij - p_i - q_j, 0))``.
    Clipping at 0 makes "unmatched" free, so any rectangular assignment of the clipped
    matrix is allowed to leave points on the diagonal, and its size is ``O(n * m)`` instead of
    ``O((n + m) ** 2)``.
    """
    n, m = len(first), len(second)
    persistence_first = _persistence_of(first)
    persistence_second = _persistence_of(second)
    total = float(persistence_first.sum() + persistence_second.sum())
    if n == 0 or m == 0:
        return total
    saving = (
        np.abs(first[:, None, 0] - second[None, :, 0])
        + np.abs(first[:, None, 1] - second[None, :, 1])
        - persistence_first[:, None]
        - persistence_second[None, :]
    )
    gain = np.minimum(saving, 0.0)
    rows, columns = linear_sum_assignment(gain)
    return total + float(gain[rows, columns].sum())


def _bottleneck(first: np.ndarray, second: np.ndarray) -> float:
    """Return the bottleneck distance between two ``(k, 2)`` diagrams of (birth, death).

    Points cost ``max(|b1 - b2|, |d1 - d2|)`` against each other and ``|d - b| / 2`` against
    the diagonal (the L-infinity convention). The distance is the smallest candidate cost whose
    thresholded augmented bipartite graph has a perfect matching, found by bisection.

    In the augmented graph every point that may go to the diagonal at the threshold can be
    parked there, and the diagonal copies match one another freely. A perfect matching therefore
    exists exactly when one point-to-point matching covers every point of both diagrams that
    cannot reach the diagonal. By the Mendelsohn-Dulmage theorem such a matching exists when
    each diagram's forced points can be covered separately, which is two bipartite matchings on
    the thresholded point-to-point graph.
    """
    n, m = len(first), len(second)
    if n + m == 0:
        return 0.0
    point_cost = np.zeros((n, m), dtype=np.float64)
    if n and m:
        point_cost = np.maximum(
            np.abs(first[:, None, 0] - second[None, :, 0]),
            np.abs(first[:, None, 1] - second[None, :, 1]),
        )
    to_diagonal_first = _persistence_of(first) / 2.0
    to_diagonal_second = _persistence_of(second) / 2.0
    candidates = np.sort(
        np.concatenate([point_cost.ravel(), to_diagonal_first, to_diagonal_second, [0.0]])
    )

    def covers_forced(near: np.ndarray, forced: np.ndarray) -> bool:
        if not forced.any():
            return True
        rows = near[forced]
        if rows.shape[1] < rows.shape[0]:
            return False
        matched = maximum_bipartite_matching(sp.csr_matrix(rows.astype(np.int8)), "column")
        return bool(np.all(matched >= 0))

    def feasible(threshold: float) -> bool:
        near = point_cost <= threshold
        return covers_forced(near, to_diagonal_first > threshold) and covers_forced(
            near.T, to_diagonal_second > threshold
        )

    low, high = 0, candidates.size - 1  # the largest candidate always admits every edge
    while low < high:
        middle = (low + high) // 2
        if feasible(float(candidates[middle])):
            high = middle
        else:
            low = middle + 1
    return float(candidates[low])


# --- validation -------------------------------------------------------------------------------


def _validate_cutoff(name: str, value: float) -> float:
    """Return ``value`` as a float after checking ``0 < value <= 1``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number in (0, 1].")
    cutoff = float(value)
    if not (math.isfinite(cutoff) and 0.0 < cutoff <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value!r}.")
    return cutoff


def _validate_hills(
    nominal_hills: Mapping[Hashable, Sequence[int]], vertex_count: int
) -> tuple[pl.Series, dict[int, tuple[int, ...]]]:
    """Return the ``hill_id`` Series in mapping order and, per vertex, its hills' positions.

    The Series is built here, before any draw, so ids that polars cannot hold in one column
    (for example ``{1: ..., "a": ...}``) fail fast.
    """
    if len(nominal_hills) == 0:
        raise ValueError("nominal_hills is empty: there are no hills to score.")
    hill_ids: list[Hashable] = []
    hills_of_vertex: dict[int, list[int]] = {}
    for position, (hill_id, members) in enumerate(nominal_hills.items()):
        hill_ids.append(hill_id)
        if len(members) == 0:
            raise ValueError(f"nominal hill {hill_id!r} is empty.")
        for member in members:
            if isinstance(member, bool) or not isinstance(member, numbers.Integral):
                raise ValueError(f"nominal hill {hill_id!r} has a non-integer member {member!r}.")
            if not 0 <= int(member) < vertex_count:
                raise ValueError(
                    f"nominal hill {hill_id!r} has member {member!r} outside "
                    f"0..{vertex_count - 1}."
                )
        for member in {int(member) for member in members}:
            hills_of_vertex.setdefault(member, []).append(position)
    try:
        hill_id_series = pl.Series("hill_id", hill_ids)
    except (TypeError, pl.exceptions.PolarsError) as error:
        raise ValueError(
            f"nominal_hills ids must share one type that polars can store: {error}"
        ) from error
    return hill_id_series, {vertex: tuple(found) for vertex, found in hills_of_vertex.items()}


def _check_reference(reference: MergeTree, network: ScalarNetwork) -> None:
    """Reject a simplified reference or one with other vertex identities than ``network``.

    Scalars are deliberately not compared: a reference built from the same vertices with other
    scalars is accepted, so a flat null can be measured against the real tree.
    """
    if reference.original is not None:
        raise ValueError(
            "reference must be an unsimplified merge tree; pass the original tree, not a "
            "simplified variant."
        )
    vertex_ids = network.vertex_ids
    reference_graph = reference.graph
    for node in reference_graph.node_indices():
        payload = reference_graph.get_node_data(node)
        vertex_index = payload.get("vertex_index")
        if (
            vertex_index is None
            or not 0 <= int(vertex_index) < len(vertex_ids)
            or payload.get("vertex_id") != str(vertex_ids[int(vertex_index)])
        ):
            raise ValueError("reference was not built from this network.")


def _reference_scalars(reference: MergeTree, vertex_count: int) -> np.ndarray:
    """Return the reference tree's own scalar of every vertex, in vertex-index order.

    Critical node payloads and ``branch_member_table`` rows partition the vertices of an
    unsimplified tree; any vertex missing or listed twice means the tree has another vertex
    count than the network.
    """
    scalars = np.full(vertex_count, np.nan, dtype=np.float64)
    seen = 0
    mapping = reference.mapping
    for table in (mapping.critical_vertices, mapping.branch_vertices):
        if table.is_empty():
            continue
        indices = table["vertex_index"].to_numpy()
        if indices.min() < 0 or indices.max() >= vertex_count:
            raise ValueError("reference was not built from this network.")
        scalars[indices] = table["scalar"].to_numpy()
        seen += indices.size
    if seen != vertex_count or np.isnan(scalars).any():
        raise ValueError("reference was not built from this network.")
    return scalars


# --- cores ------------------------------------------------------------------------------------


def _find_set(parent: dict[int, int], item: int) -> int:
    root = item
    while parent[root] != root:
        root = parent[root]
    while parent[item] != root:
        parent[item], item = root, parent[item]
    return root


def extract_cores(
    pair_frequency: pl.DataFrame,
    *,
    cutoff: float,
    scalars: Sequence[float] | np.ndarray | None = None,
) -> pl.DataFrame:
    """Read peak cores off a terminal co-occurrence table.

    A core is a connected component of the graph whose vertices are those with
    ``frequency(v, v) >= cutoff`` and whose edges are the pairs ``(u, v)`` with
    ``frequency(u, v) >= cutoff`` between two such vertices. Cores are disjoint; a core can be a
    single vertex. Re-cut at another cutoff without re-running the draws.

    Parameters
    ----------
    pair_frequency : pl.DataFrame
        Table with columns ``vertex_a``, ``vertex_b`` and ``frequency`` as in
        `BootstrapResult.same_leaf_frequency`, diagonal rows included.
    cutoff : float
        Frequency cutoff in ``(0, 1]``.
    scalars : Sequence[float] | np.ndarray | None, default None
        Optional finite scalar per vertex, indexed by vertex index. When given, ``core_id``
        runs in order of decreasing highest scalar in the core, ties by the smallest vertex
        index; otherwise in order of the smallest vertex index.

    Returns
    -------
    pl.DataFrame
        Columns ``core_id`` and ``vertex_index`` (Int64), sorted by ``(core_id, vertex_index)``.

    Raises
    ------
    ValueError
        If ``cutoff`` is not in ``(0, 1]``, a required column is missing, or ``scalars`` is
        not a finite one-dimensional array covering every vertex of the table.
    """
    cutoff = _validate_cutoff("cutoff", cutoff)
    missing = {"vertex_a", "vertex_b", "frequency"} - set(pair_frequency.columns)
    if missing:
        raise ValueError(f"pair_frequency is missing columns {sorted(missing)}.")
    first = pair_frequency["vertex_a"].to_numpy()
    second = pair_frequency["vertex_b"].to_numpy()
    frequency = pair_frequency["frequency"].to_numpy()
    passes = frequency >= cutoff
    vertices = np.unique(first[passes & (first == second)])

    scalar_values = None
    if scalars is not None:
        scalar_values = np.asarray(scalars, dtype=np.float64)
        if scalar_values.ndim != 1 or not np.isfinite(scalar_values).all():
            raise ValueError("scalars must be a finite one-dimensional array.")
        if vertices.size and vertices.max() >= scalar_values.size:
            raise ValueError("scalars must cover every vertex index of pair_frequency.")

    parent = {int(vertex): int(vertex) for vertex in vertices}
    linked = passes & (first != second) & np.isin(first, vertices) & np.isin(second, vertices)
    for a, b in zip(first[linked].tolist(), second[linked].tolist()):
        root_a, root_b = _find_set(parent, a), _find_set(parent, b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    components: dict[int, list[int]] = {}
    for vertex in sorted(parent):
        components.setdefault(_find_set(parent, vertex), []).append(vertex)

    def order_key(members: list[int]) -> tuple[float, int]:
        top = (
            -max(scalar_values[member] for member in members) if scalar_values is not None else 0.0
        )
        return (top, members[0])

    ordered = sorted(components.values(), key=order_key)
    return pl.DataFrame(
        {
            "core_id": pl.Series(
                [core_id for core_id, members in enumerate(ordered) for _ in members],
                dtype=pl.Int64,
            ),
            "vertex_index": pl.Series(
                [vertex for members in ordered for vertex in members], dtype=pl.Int64
            ),
        }
    )
