# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""Leaf co-occurrence: the tree statistic of the chunked bootstrap (ADR 0007, ADR 0008).

`LeafCooccurrence` plugs into `topographer.statistics.bootstrap`. Each draw rebuilds the split or
join tree and records which vertices share a leaf (terminal) branch, the role of every vertex
and a persistence-diagram comparison with a reference tree. `finish` turns the summed counts
into a `BootstrapResult`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np
import polars as pl

from topographer.graph import ScalarNetwork
from dubanevics2026topological.simplification.cooccurrence_bootstrap import (
    _PEAK_EVENT,
    _ROLE_LEAF,
    _ROLE_ROOT,
    _ROLE_SADDLE,
    _bottleneck,
    _check_reference,
    _critical_roles,
    _Diagram,
    _persistence_diagram,
    _reference_scalars,
    _terminal_branches,
    _validate_cutoff,
    _validate_hills,
    _wasserstein_1,
    extract_cores,
)
from topographer.statistics.bootstrap import Counts, DrawnNetwork
from topographer.topology import MergeTree, ScalarNetworkTopology
from topographer.topology import build_join_tree, build_split_tree, reduce_merge_tree

__all__ = ["BootstrapResult", "LeafCooccurrence"]

_STAT_NAMES = ("leaf_count", "wasserstein_1", "bottleneck", "reproduced_pair_fraction")


@dataclass(frozen=True)
class BootstrapResult:
    """Leaf co-occurrence of network vertices under a parametric bootstrap.

    Vertex indices are Rustworkx node indices of the network, not vertex identities;
    ``network.vertex_ids[index]`` maps an index to its identity.

    Attributes
    ----------
    same_leaf_frequency : pl.DataFrame
        Columns ``vertex_a`` and ``vertex_b`` (Int64, ``vertex_a <= vertex_b``) and ``frequency``
        (Float64): the fraction of draws in which both vertices are members of the same leaf
        branch of the rebuilt tree. Only nonzero pairs are listed, sorted by
        ``(vertex_a, vertex_b)``. Diagonal rows ``(v, v)`` give how often ``v`` is on any leaf
        branch at all.
    vertex_roles : pl.DataFrame
        One row per network node index with ``vertex_index`` (Int64) and the Float64 columns
        ``leaf_frequency``, ``saddle_frequency``, ``regular_frequency`` and ``root_frequency``,
        the fraction of draws in which the vertex has that role in the rebuilt tree. The four
        roles are one-hot per draw with priority root, then leaf (leaf extremum), then saddle,
        else regular, so they sum to 1. ``terminal_frequency`` is the diagonal of
        ``same_leaf_frequency`` and is not part of that set.
    leaf_cores : pl.DataFrame
        Columns ``core_id`` and ``vertex_index`` (Int64): the cores at ``core_cutoff``, from
        `extract_cores`, ordered by the highest reference scalar in the core, then by the
        smallest vertex index.
    draw_stats : pl.DataFrame
        One row per draw, in draw order: ``draw`` (Int64, ``0..n_draws-1``), ``leaf_count``
        (Int64, the number of leaf extrema of the rebuilt tree), ``wasserstein_1``,
        ``bottleneck`` (Float64, distances between the draw's and the reference's H0 persistence
        diagrams, each including the global pair) and ``reproduced_pair_fraction`` (Float64, the
        share of reference ``(leaf vertex, death vertex)`` pairs found identically in the draw,
        the global pair counted as one of them).
    nominal_hill_support : pl.DataFrame | None
        Columns ``hill_id`` and ``hill_support`` (Float64) when ``nominal_hills`` was given: the
        fraction of draws in which any member of the hill is a leaf of the rebuilt tree.
        ``None`` otherwise.
    n_draws : int
        Number of bootstrap draws.
    seed : int | None
        Seed of the bootstrap.
    tree_type : str
        ``"split"`` or ``"join"``.
    core_cutoff : float
        Cutoff at which ``leaf_cores`` was extracted.
    """

    same_leaf_frequency: pl.DataFrame
    vertex_roles: pl.DataFrame
    leaf_cores: pl.DataFrame
    draw_stats: pl.DataFrame
    nominal_hill_support: pl.DataFrame | None
    n_draws: int
    seed: int | None
    tree_type: str
    core_cutoff: float


# --- exact integer counts ---------------------------------------------------------------------
#
# A draw's counts are sparse integer tables whose last column is the count and whose other
# columns are the key: ``pairs`` (vertex_a, vertex_b, count), ``roles`` (vertex, role, count)
# and ``hills`` (hill position, count). Tables are kept sorted by key with unique keys, so a
# sum has one canonical form whatever the grouping or order.


def _coalesce(table: np.ndarray) -> np.ndarray:
    """Return ``table`` sorted by its key columns with equal keys summed."""
    if len(table) == 0:
        return table
    keys = table[:, :-1]
    ordered = table[np.lexsort(keys.T[::-1])]
    changed = np.any(ordered[1:, :-1] != ordered[:-1, :-1], axis=1)
    starts = np.concatenate([[0], np.flatnonzero(changed) + 1])
    totals = np.add.reduceat(ordered[:, -1], starts)
    return np.column_stack([ordered[starts, :-1], totals])


def _pair_table(branches: Mapping[int, Sequence[int]]) -> np.ndarray:
    """Count every pair ``(a <= b)`` of members of each leaf branch, diagonal included."""
    parts = []
    for members in branches.values():
        array = np.asarray(members, dtype=np.int64)
        first, second = np.triu_indices(array.size)
        parts.append(np.column_stack([array[first], array[second], np.ones(first.size, np.int64)]))
    return np.concatenate(parts) if parts else np.zeros((0, 3), dtype=np.int64)


def _role_table(peaks: list[int], saddles: list[int], root: int) -> np.ndarray:
    rows = [(v, _ROLE_LEAF, 1) for v in peaks]
    rows += [(v, _ROLE_SADDLE, 1) for v in saddles]
    rows.append((root, _ROLE_ROOT, 1))
    return np.array(rows, dtype=np.int64)


@dataclass(frozen=True)
class _Bound:
    """What a statistic learns from the network in ``prepare``; small and picklable."""

    nominal: _Diagram
    hills_of_vertex: dict[int, tuple[int, ...]]
    hill_ids: pl.Series | None
    reference_scalars: np.ndarray
    network_hash: str


def _network_hash(network: ScalarNetwork) -> str:
    """Hash the vertex identities (in index order) and the edge list of ``network``."""
    digest = hashlib.sha256()
    digest.update(repr(network.vertex_ids).encode())
    edges = np.asarray(network.graph.edge_list(), dtype="<i8").reshape(-1, 2)
    digest.update(edges.tobytes())
    return digest.hexdigest()


class LeafCooccurrence:
    """Which vertices stick together on a leaf branch of a rebuilt merge tree.

    A `DrawStatistic` for `topographer.statistics.bootstrap.run`. In each draw the split or
    join tree of the drawn landscape is rebuilt. A leaf branch is a tree edge whose leaf-side
    node is a leaf extremum (a maximum for a split tree, a minimum for a join tree); its members
    are that extremum and the regular vertices on the edge, and the saddle or root where it ends
    is not a member. Every pair of members of one leaf branch, each vertex with itself
    included, adds one to the pair count. Each draw also records the vertex roles (root, leaf,
    saddle, regular), the nominal hills that have a leaf, and H0 persistence-diagram distances
    to the reference tree (see ADR 0007 for the conventions).

    `finish` takes ``core_cutoff`` (default 0.95) as an option, so saved chunks can be
    combined again at another cutoff.

    Parameters
    ----------
    tree_type : {"split", "join"}
        Tree kind rebuilt for every draw.
    reference : MergeTree | None, default None
        Unsimplified reference tree of kind ``tree_type`` with the same vertex identities as the
        network; defaults to the network's own split or join tree. It supplies the nominal
        diagram and the scalars that order the cores, so it may come from a network with other
        scalars (for example a flat null measured against the real tree). The caller guarantees
        the same edges.
    nominal_hills : Mapping[Hashable, Sequence[int]] | None, default None
        Maps a hill id to the network node indices of its nominal members; the result then
        carries ``nominal_hill_support``.

    Raises
    ------
    ValueError
        For an invalid ``tree_type``. ``prepare`` raises for invalid ``nominal_hills`` or a
        ``reference`` of another kind, a simplified one, or one that was not built from the
        network.
    """

    def __init__(
        self,
        tree_type: Literal["split", "join"],
        reference: MergeTree | None = None,
        nominal_hills: Mapping[Hashable, Sequence[int]] | None = None,
    ) -> None:
        if tree_type not in _PEAK_EVENT:
            raise ValueError(f"tree_type must be 'split' or 'join', got {tree_type!r}.")
        self.tree_type = tree_type
        self.reference = reference
        self.nominal_hills = nominal_hills
        self._bound: _Bound | None = None

    def prepare(self, network: ScalarNetwork) -> LeafCooccurrence:
        """Return a copy bound to ``network``: validated reference, hills and nominal diagram.

        The copy holds no ``MergeTree`` (only the nominal diagram, the reference scalars and
        the hill tables), so it is cheap to send to worker processes: the copy keeps
        ``reference`` and ``nominal_hills`` as attributes, but drops ``reference`` when pickled.
        Preparing a bound statistic on the same network (same vertex identities and edges)
        returns it unchanged.

        Raises
        ------
        ValueError
            If this statistic is already bound to a different network; prepare the original,
            unbound statistic for each network instead.
        """
        if not isinstance(network, ScalarNetwork):
            raise TypeError("network must be a ScalarNetwork.")
        network_hash = _network_hash(network)
        if self._bound is not None:
            if self._bound.network_hash != network_hash:
                raise ValueError(
                    "this statistic is already bound to another network (different vertex "
                    "ids or edges); call prepare on the unbound statistic."
                )
            return self
        reference = self.reference
        if reference is None:
            topology = ScalarNetworkTopology(network)
            reference = (
                topology.split_tree()
                if self.tree_type == "split"
                else topology.join_tree()
            )
        if reference.kind != self.tree_type:
            raise ValueError(
                f"reference.kind is {reference.kind!r} but tree_type is {self.tree_type!r}."
            )
        _check_reference(reference, network)
        vertex_count = network.vertex_count
        reference_scalars = _reference_scalars(reference, vertex_count)
        hill_ids, hills_of_vertex = (
            _validate_hills(self.nominal_hills, vertex_count)
            if self.nominal_hills is not None
            else (None, {})
        )
        bound = LeafCooccurrence(self.tree_type, self.reference, self.nominal_hills)
        bound._bound = _Bound(
            nominal=_persistence_diagram(
                reference.graph, tree_kind=self.tree_type, root_to_leaf=True
            ),
            hills_of_vertex=hills_of_vertex,
            hill_ids=hill_ids,
            reference_scalars=reference_scalars,
            network_hash=network_hash,
        )
        return bound

    def __getstate__(self) -> dict:
        """Drop the reference tree when pickled; workers need only the bound tables."""
        state = self.__dict__.copy()
        if self._bound is not None:
            state["reference"] = None
        return state

    def _require_bound(self) -> _Bound:
        if self._bound is None:
            raise ValueError("statistic is not bound to a network; call prepare(network) first.")
        return self._bound

    def measure(self, network: DrawnNetwork) -> tuple[Counts, np.ndarray]:
        """Rebuild the tree of one drawn landscape and return its counts and stats row."""
        bound = self._require_bound()
        build = build_split_tree if self.tree_type == "split" else build_join_tree
        tree = reduce_merge_tree(build(network.graph, scalar_attr=network.scalar_attr).tree)
        peak_event = _PEAK_EVENT[self.tree_type]

        branches = _terminal_branches(tree, peak_event=peak_event, root_to_leaf=False)
        peaks, saddles, root = _critical_roles(tree, peak_event=peak_event, root_to_leaf=False)
        hit = {position for vertex in peaks for position in bound.hills_of_vertex.get(vertex, ())}
        counts = {
            "pairs": _coalesce(_pair_table(branches)),
            "roles": _coalesce(_role_table(peaks, saddles, root)),
            "hills": np.array([(p, 1) for p in sorted(hit)], dtype=np.int64).reshape(-1, 2),
        }

        diagram = _persistence_diagram(tree, tree_kind=self.tree_type, root_to_leaf=False)
        reproduced = len(bound.nominal.pairs & diagram.pairs) / len(bound.nominal.pairs)
        row = np.array(
            [
                len(peaks),
                _wasserstein_1(diagram.points, bound.nominal.points),
                _bottleneck(diagram.points, bound.nominal.points),
                reproduced,
            ]
        )
        return counts, row

    def add(self, first: Counts, second: Counts) -> Counts:
        """Return the exact sum of two counts mappings."""
        return {name: _coalesce(np.concatenate([first[name], second[name]])) for name in first}

    def fingerprint(self) -> str:
        """Hash the tree type, the nominal diagram, the hills and the reference scalars."""
        bound = self._require_bound()
        digest = hashlib.sha256()
        digest.update(self.tree_type.encode())
        digest.update(np.ascontiguousarray(bound.nominal.points, dtype="<f8").tobytes())
        digest.update(repr(sorted(bound.nominal.pairs)).encode())
        digest.update(repr(sorted(bound.hills_of_vertex.items())).encode())
        digest.update(
            repr(bound.hill_ids.to_list() if bound.hill_ids is not None else None).encode()
        )
        digest.update(np.ascontiguousarray(bound.reference_scalars, dtype="<f8").tobytes())
        return digest.hexdigest()

    def finish(
        self,
        counts: Counts,
        stats: np.ndarray,
        n_draws: int,
        *,
        seed: int | None = None,
        core_cutoff: float = 0.95,
    ) -> BootstrapResult:
        """Turn summed counts and per-draw stats into a `BootstrapResult`.

        Parameters
        ----------
        counts : Mapping[str, np.ndarray]
            Summed counts of all ``n_draws`` draws.
        stats : np.ndarray
            ``(n_draws, 4)`` rows in draw order.
        n_draws : int
            Number of draws.
        seed : int | None, default None
            Seed of the bootstrap, recorded in the result (`combine` passes the chunks' seed).
        core_cutoff : float, default 0.95
            Frequency cutoff in ``(0, 1]`` for ``leaf_cores``.
        """
        bound = self._require_bound()
        core_cutoff = _validate_cutoff("core_cutoff", core_cutoff)
        if stats.shape != (n_draws, len(_STAT_NAMES)):
            raise ValueError(
                f"stats must have shape ({n_draws}, {len(_STAT_NAMES)}), got {stats.shape}."
            )
        vertex_count = bound.reference_scalars.size

        pairs = counts["pairs"]
        pair_frequency = pl.DataFrame(
            {
                "vertex_a": pl.Series(pairs[:, 0], dtype=pl.Int64),
                "vertex_b": pl.Series(pairs[:, 1], dtype=pl.Int64),
                "frequency": pl.Series(pairs[:, 2] / n_draws, dtype=pl.Float64),
            }
        )
        diagonal = np.zeros(vertex_count, dtype=np.int64)
        on_diagonal = pairs[:, 0] == pairs[:, 1]
        diagonal[pairs[on_diagonal, 0]] = pairs[on_diagonal, 2]

        role_counts = np.zeros((vertex_count, 3), dtype=np.int64)
        roles = counts["roles"]
        role_counts[roles[:, 0], roles[:, 1]] = roles[:, 2]
        frequencies = {
            "leaf_frequency": role_counts[:, _ROLE_LEAF] / n_draws,
            "saddle_frequency": role_counts[:, _ROLE_SADDLE] / n_draws,
            "regular_frequency": (n_draws - role_counts.sum(axis=1)) / n_draws,
            "root_frequency": role_counts[:, _ROLE_ROOT] / n_draws,
            "terminal_frequency": diagonal / n_draws,
        }
        vertex_roles = pl.DataFrame(
            {"vertex_index": pl.Series(np.arange(vertex_count), dtype=pl.Int64)}
        ).with_columns(
            *(pl.Series(name, values, dtype=pl.Float64) for name, values in frequencies.items())
        )

        draw_stats = pl.DataFrame(
            {
                "draw": pl.Series(np.arange(n_draws), dtype=pl.Int64),
                "leaf_count": pl.Series(stats[:, 0].astype(np.int64), dtype=pl.Int64),
                "wasserstein_1": pl.Series(stats[:, 1], dtype=pl.Float64),
                "bottleneck": pl.Series(stats[:, 2], dtype=pl.Float64),
                "reproduced_pair_fraction": pl.Series(stats[:, 3], dtype=pl.Float64),
            }
        )
        hill_support = None
        if bound.hill_ids is not None:
            hill_counts = np.zeros(bound.hill_ids.len(), dtype=np.int64)
            hill_counts[counts["hills"][:, 0]] = counts["hills"][:, 1]
            hill_support = pl.DataFrame(
                {
                    "hill_id": bound.hill_ids,
                    "hill_support": pl.Series(hill_counts / n_draws, dtype=pl.Float64),
                }
            )
        return BootstrapResult(
            same_leaf_frequency=pair_frequency,
            vertex_roles=vertex_roles,
            leaf_cores=extract_cores(
                pair_frequency, cutoff=core_cutoff, scalars=bound.reference_scalars
            ),
            draw_stats=draw_stats,
            nominal_hill_support=hill_support,
            n_draws=n_draws,
            seed=seed,
            tree_type=self.tree_type,
            core_cutoff=core_cutoff,
        )
