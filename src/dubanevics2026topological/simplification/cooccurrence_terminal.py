# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""Deprecated entry point `bootstrap_terminal_cooccurrence`, kept on top of the chunked API.

New code uses `topographer.statistics.bootstrap` with `LeafCooccurrence` (ADR 0008). The
numbers are identical to ADR 0007's; only the names of the old result type are kept here.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
import numbers
from typing import Literal
import warnings

import numpy as np
import polars as pl

from topographer.graph import ScalarNetwork
from dubanevics2026topological.simplification.cooccurrence_bootstrap import _validate_cutoff
from dubanevics2026topological.simplification.cooccurrence_leaf import LeafCooccurrence
from topographer.statistics.bootstrap import combine, run
from topographer.topology import MergeTree

__all__ = ["TerminalCooccurrenceResult", "bootstrap_terminal_cooccurrence"]


@dataclass(frozen=True)
class TerminalCooccurrenceResult:
    """Terminal co-occurrence of network vertices under a parametric bootstrap.

    Vertex indices are Rustworkx node indices of the network, not vertex identities;
    ``network.vertex_ids[index]`` maps an index to its identity.

    Attributes
    ----------
    pair_frequency : pl.DataFrame
        Columns ``vertex_a`` and ``vertex_b`` (Int64, ``vertex_a <= vertex_b``) and ``frequency``
        (Float64): the fraction of draws in which both vertices are members of the same terminal
        branch of the rebuilt tree. Only nonzero pairs are listed, sorted by
        ``(vertex_a, vertex_b)``. Diagonal rows ``(v, v)`` give how often ``v`` is on any
        terminal branch at all.
    vertex_roles : pl.DataFrame
        One row per network node index with ``vertex_index`` (Int64) and the Float64 columns
        ``peak_frequency``, ``saddle_frequency``, ``regular_frequency`` and ``root_frequency``,
        the fraction of draws in which the vertex has that role in the rebuilt tree. The four
        roles are one-hot per draw with priority root, then peak (leaf extremum), then saddle,
        else regular, so they sum to 1. ``terminal_frequency`` is the diagonal of
        ``pair_frequency`` and is not part of that set: a peak is always on a terminal branch,
        and so is every regular vertex of one.
    cores : pl.DataFrame
        Columns ``core_id`` and ``vertex_index`` (Int64): the peak cores at ``core_cutoff``,
        from `extract_cores`, ordered by the highest reference scalar in the core, then by the
        smallest vertex index.
    draw_stats : pl.DataFrame
        One row per draw, in draw order: ``draw`` (Int64, ``0..n_replicates-1``), ``peak_count``
        (Int64, the number of leaf extrema of the rebuilt tree), ``wasserstein_1``,
        ``bottleneck`` (Float64, distances between the draw's and the reference's H0 persistence
        diagrams, each including the global pair) and ``reproduced_pair_fraction`` (Float64, the
        share of reference ``(peak vertex, death vertex)`` pairs found identically in the draw;
        the global ``(peak, root)`` pair counts as one of them, so the denominator is the number
        of nominal hills, the reference's non-global pairs, plus 1; it does not depend on the
        ``nominal_hills`` argument).
    nominal_hill_support : pl.DataFrame | None
        Columns ``hill_id`` and ``hill_support`` (Float64) when ``nominal_hills`` was given: the
        fraction of draws in which any member of the hill is a peak of the rebuilt tree.
        ``None`` otherwise.
    n_replicates : int
        Number of bootstrap draws.
    seed : int | None
        Seed passed to ``numpy.random.SeedSequence``.
    tree_type : str
        ``"split"`` or ``"join"``.
    core_cutoff : float
        Cutoff at which ``cores`` was extracted.
    """

    pair_frequency: pl.DataFrame
    vertex_roles: pl.DataFrame
    cores: pl.DataFrame
    draw_stats: pl.DataFrame
    nominal_hill_support: pl.DataFrame | None
    n_replicates: int
    seed: int | None
    tree_type: str
    core_cutoff: float


def _validate_arguments(
    tree_type: str, n_replicates: int, n_jobs: int, core_cutoff: float
) -> float:
    if tree_type not in ("split", "join"):
        raise ValueError(f"tree_type must be 'split' or 'join', got {tree_type!r}.")
    for name, value in (("n_replicates", n_replicates), ("n_jobs", n_jobs)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    return _validate_cutoff("core_cutoff", core_cutoff)


def bootstrap_terminal_cooccurrence(
    network: ScalarNetwork,
    *,
    se_attr: str = "se",
    tree_type: Literal["split", "join"] = "split",
    n_replicates: int = 1000,
    seed: int | None = None,
    core_cutoff: float = 0.95,
    reference: MergeTree | None = None,
    nominal_hills: Mapping[Hashable, Sequence[int]] | None = None,
    n_jobs: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TerminalCooccurrenceResult:
    """Deprecated: use `bootstrap.run` with `LeafCooccurrence`; same numbers.

    Score which vertices stick together at the top of a hill under a parametric bootstrap.

    Every draw replaces each vertex scalar by ``Normal(scalar, se)`` independently and rebuilds
    the merge tree. In the rebuilt tree a terminal branch is an edge whose leaf-side node is a
    leaf extremum (a maximum for a split tree, a minimum for a join tree); its members are that
    extremum and the regular vertices on the edge, and the saddle or root where it ends is not a
    member. For each pair of members of one terminal branch, including each vertex with itself,
    the pair count grows by one, so the diagonal counts how often a vertex is on any terminal
    branch. After all draws the counts are divided by ``n_replicates``, and peak cores, the
    sets of vertices that stick together, are read off the pair table with `extract_cores`.

    Each draw also yields a persistence-diagram comparison with the reference. The diagram is
    the H0 persistence pairs of the tree including the global extremum paired with the root
    scalar. ``wasserstein_1`` uses the L1 ground metric between points with a point-to-diagonal
    cost equal to its persistence ``|d - b|``; ``bottleneck`` uses the standard L-infinity
    convention (``max(|db|, |dd|)`` between points, ``|d - b| / 2`` to the diagonal). They equal
    ``gudhi.hera.wasserstein_distance(..., order=1, internal_p=1)`` and
    ``gudhi.hera.bottleneck_distance`` respectively. ``wasserstein_1`` is computed on the exact
    reduced ``n x m`` assignment, which is equivalent to the diagonal-augmented assignment in
    which an unmatched point goes to the diagonal at cost equal to its persistence.

    Vertex roles in a draw take one value in priority order: the root of the rebuilt tree, then
    peak (a leaf extremum), then saddle, else regular.

    Draw ``d`` uses ``numpy.random.SeedSequence(seed).spawn(n_replicates)[d]``, and all counts
    are summed as exact integers, so results are identical for every ``n_jobs``. Draws run on a
    minimal copy of the network graph that keeps only the scalar attribute, so the input network
    and the reference tree are never modified.

    With ``n_jobs > 1`` the workers are started with the ``spawn`` method, so a script that calls
    this function must guard its entry point with ``if __name__ == "__main__":``.

    The reference supplies the nominal diagram and the scalars that order the cores, using its
    own scalars rather than the network's, and for ``tree_type="join"`` the cores are still
    ordered by the highest reference scalar. It may therefore come from a network with the same
    vertex identities and graph but other scalars, for example a flat null that sets every
    scalar of a copy of the network to a constant and measures its draws against the real tree.
    The reference tree cannot reveal the network's edges, so the caller guarantees that
    ``network`` and the network of ``reference`` have the same edges.

    Parameters
    ----------
    network : ScalarNetwork
        Network whose scalar attribute is the landscape that is drawn around.
    se_attr : str, default "se"
        Vertex attribute with finite, non-negative standard errors (zero keeps the vertex fixed).
    tree_type : {"split", "join"}, default "split"
        Tree kind rebuilt for every draw.
    n_replicates : int, default 1000
        Number of draws, at least 1; any integer type is accepted.
    seed : int | None, default None
        Seed for ``SeedSequence``; ``None`` draws fresh entropy.
    core_cutoff : float, default 0.95
        Frequency cutoff in ``(0, 1]`` for the cores in the result.
    reference : MergeTree | None, default None
        Unsimplified reference tree of kind ``tree_type`` with the same vertex identities as
        ``network``; defaults to the network's own split or join tree.
    nominal_hills : Mapping[Hashable, Sequence[int]] | None, default None
        Maps a hill id to the network node indices of its nominal members. When given, the
        result carries ``nominal_hill_support``.
    n_jobs : int, default 1
        Worker processes, started with ``spawn``; 1 runs in-process.
    progress_callback : Callable[[int, int], None] | None, default None
        Called as ``progress_callback(completed, total)`` after each draw (``n_jobs == 1``) or
        each finished chunk of draws (``n_jobs > 1``).

    Returns
    -------
    TerminalCooccurrenceResult
        Pair frequencies, vertex roles, cores, per-draw statistics and optional hill support.
        Vertex indices are Rustworkx node indices of ``network``; ``network.vertex_ids[i]`` maps
        node index ``i`` to the vertex identity.

    Raises
    ------
    ValueError
        For an invalid ``tree_type``, ``n_replicates``, ``n_jobs`` (booleans are rejected) or
        ``core_cutoff``, invalid standard errors, invalid ``nominal_hills``, a ``reference``
        whose kind differs from ``tree_type``, a simplified ``reference``, or a ``reference``
        with other vertex identities or another vertex count than ``network``.
    TypeError
        If ``network`` is not a ``ScalarNetwork``.
    """
    warnings.warn(
        "bootstrap_terminal_cooccurrence is deprecated; use topographer.statistics.bootstrap "
        "with LeafCooccurrence (ADR 0008).",
        DeprecationWarning,
        stacklevel=2,
    )
    core_cutoff = _validate_arguments(tree_type, n_replicates, n_jobs, core_cutoff)
    # A run needs an integer seed; None keeps the old "fresh entropy" meaning.
    run_seed = np.random.SeedSequence().entropy if seed is None else seed
    statistic = LeafCooccurrence(tree_type, reference=reference, nominal_hills=nominal_hills)
    bound = statistic.prepare(network)  # validates reference and hills before any draw
    chunk = run(
        network,
        bound,
        se_attr=se_attr,
        n_draws=n_replicates,
        seed=run_seed,
        n_jobs=n_jobs,
        progress=progress_callback,
    )
    result = combine([chunk], bound, core_cutoff=core_cutoff)
    roles = result.vertex_roles.rename({"leaf_frequency": "peak_frequency"})
    return TerminalCooccurrenceResult(
        pair_frequency=result.same_leaf_frequency,
        vertex_roles=roles,
        cores=result.leaf_cores,
        draw_stats=result.draw_stats.rename({"leaf_count": "peak_count"}),
        nominal_hill_support=result.nominal_hill_support,
        n_replicates=result.n_draws,
        seed=seed,
        tree_type=tree_type,
        core_cutoff=core_cutoff,
    )
