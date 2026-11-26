"""Per-leaf tables shared by the two leaf diagnostic figures.

``papkou2023_{graph}_leaf_persistence`` and ``papkou2023_{graph}_leaf_swarm`` draw
the same objects: the leaves (peaks) of a clamped landscape's nominal split tree,
each with its hill. Nothing is recomputed here. The tree, its persistence table and
its branch member table come from the cached comparison checkpoint, and
:func:`~dubanevics2026topological.analysis.describe.describe`
supplies the paired death, the adjacent saddle, the branch vertex count and the stop
flag of every peak; :func:`~dubanevics2026topological.simplification.bootstrap_smoothed_tree.nominal_hills`
assembles each hill (peak plus the members of its terminal branch, the adjacent
saddle excluded). Tables carry sequence identities only, never Rustworkx indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from ._charts import padded_scalar_domain, wild_type_identities

# Landscape key and heading of each graph the leaf figures cover, by ID slug.
GRAPHS: dict[str, tuple[str, str]] = {
    "published": ("study_clamped", "Published graph"),
    "candidate": ("candidate_clamped", "Candidate graph"),
}

# Split-tree persistence tables hold ordinary pairs and exactly one "extended" pair,
# the global maximum's, closed at the global minimum by convention (the rule of
# ``nominal_diagram`` in simplification/bootstrap_smoothed_tree.py). The Full-graph
# component persistence table's "extended_positive" is a different table and is not
# accepted here.
GLOBAL_PAIR_KIND = "extended"
PAIR_KINDS = frozenset({"ordinary", GLOBAL_PAIR_KIND})


@dataclass(frozen=True)
class LeafTables:
    """One row per leaf and one row per member sequence of every leaf's hill."""

    leaves: pl.DataFrame
    members: pl.DataFrame


def figure_graph(figure_id: str, suffix: str) -> tuple[str, str, str]:
    """Return ``(slug, landscape key, heading)`` of ``{slug}_{suffix}``."""
    tail = f"_{suffix}"
    if figure_id.endswith(tail):
        slug = figure_id[: -len(tail)]
        if slug in GRAPHS:
            return (slug, *GRAPHS[slug])
    raise ValueError(f"{figure_id!r} is not a {suffix.replace('_', ' ')} figure ID.")


def leaf_tables(landscape: Any, *, key: str, pairs: pl.DataFrame) -> LeafTables:
    """Assemble the leaf and member tables of one clamped landscape.

    ``landscape`` carries ``trees["split"]`` (``tree``, ``branch_table``,
    ``branch_member_table``, ``persistence_table``, ``plot_data`` and an optional
    ``wild_type_marker_table``); ``key`` names it in the output; ``pairs`` is the
    landscape's saved biological annotation pairs table.

    ``leaves`` has one row per peak, ranked by birth fitness descending with the
    peak's DNA sequence breaking ties (``rank`` counts from 1): ``peak_sequence``,
    ``peak_aa_sequence``, ``birth_scalar``, the paired ``death_sequence`` and
    ``death_scalar`` (never the adjacent saddle), ``is_global_pair`` (the global
    maximum's pair, closed at the global minimum), ``adjacent_saddle_sequence`` and
    ``adjacent_saddle_scalar`` (null where the peak's branch ends at a root that is
    not a saddle), ``member_count`` (the hill's sequence count, equal to the terminal
    branch's vertex count), ``is_singleton``, ``birth_has_stop_codon`` and
    ``is_wild_type``.

    ``leaves`` also has ``adjacent_saddle_aa_sequence`` and ``saddle_is_synonymous`` (the
    adjacent saddle has the peak's amino-acid genotype), ``tree_x`` (the peak's x
    position in the tree drawing's own layout, ``plot_data.node_table``) and
    ``tree_order`` (1 from the left: peaks ordered by ``tree_x``, ties by birth fitness
    descending then DNA sequence; the left-to-right order the peaks appear on the tree
    figure, not the birth-fitness ``rank``).

    ``members`` has one row per sequence of each hill, the peak included: ``rank``,
    ``leaf_peak_sequence``, ``dna_sequence``, ``raw_fitness``, ``is_peak`` and
    ``is_synonymous`` (not the peak, with the peak's amino-acid genotype; the branch member
    table must carry ``aa_sequence``, otherwise a ``ValueError`` names it),
    ``is_stop_codon`` (the sequence has a stop codon: the member table's
    ``has_stop_codon``, the peak's own flag for the peak) and ``tree_order`` (its
    leaf's), ordered by leaf rank then fitness descending then DNA sequence.
    """
    from ..simplification.bootstrap_smoothed_tree import nominal_hills

    tree_analysis = landscape.trees["split"]
    kinds = pairs["pair_kind"].to_list()
    if kinds.count(GLOBAL_PAIR_KIND) != 1 or set(kinds) - PAIR_KINDS:
        raise ValueError(
            "The persistence table must hold ordinary pairs and exactly one global "
            f"pair (pair_kind {GLOBAL_PAIR_KIND!r}), found {sorted(set(kinds))}."
        )
    hills = {hill.peak_sequence: hill for hill in nominal_hills(tree_analysis)}
    if set(hills) != set(pairs["birth_sequence"].to_list()):
        raise ValueError("Every split-tree peak must be exactly one persistence birth.")
    wild_type = wild_type_identities(tree_analysis)
    member_table = tree_analysis.branch_member_table
    if "aa_sequence" not in member_table.columns:
        raise ValueError(
            "The branch member table lacks the column 'aa_sequence', so the synonymous "
            "members of each hill cannot be named."
        )
    fitness = {
        str(dna): float(scalar)
        for dna, scalar in member_table.select("dna_sequence", "scalar").iter_rows()
    }
    aa_of = {
        str(dna): str(aa)
        for dna, aa in member_table.select("dna_sequence", "aa_sequence").iter_rows()
    }
    stops = (
        {
            str(dna)
            for dna, flag in member_table.select(
                "dna_sequence", "has_stop_codon"
            ).iter_rows()
            if flag
        }
        if "has_stop_codon" in member_table.columns
        else set()
    )
    node_table = tree_analysis.plot_data.node_table
    x_of = (
        {
            str(dna): float(x)
            for dna, x in node_table.select("dna_sequence", "x").iter_rows()
            if str(dna) in hills
        }
        if {"dna_sequence", "x"} <= set(node_table.columns)
        else {}
    )
    return _assemble(
        pairs,
        {peak: tuple(hill.members) for peak, hill in hills.items()},
        wild_type,
        fitness,
        aa_of,
        stops=stops,
        x_of=x_of,
    )


def _assemble(
    pairs: pl.DataFrame,
    hills: dict[str, tuple[str, ...]],
    wild_type: Any,
    fitness: dict[str, float],
    aa_of: dict[str, str],
    *,
    stops: Any = frozenset(),
    x_of: dict[str, float] | None = None,
) -> LeafTables:
    """Leaf and member tables from the pairs, each peak's hill members (peak first),
    the wild-type identities and the fitness and amino-acid maps of the non-peak members.

    ``stops`` are the non-peak members with a stop codon; ``x_of`` maps each peak to its
    x in the tree layout (``tree_order`` follows it; without it the order is the
    birth-fitness ``rank``).
    """
    x_of = x_of or {}
    rows = []
    for pair in pairs.to_dicts():
        peak = pair["birth_sequence"]
        count = len(hills[peak])
        if count != pair["branch_vertex_count"]:
            raise ValueError(
                f"Hill of {peak!r} has {count} sequences but its terminal branch "
                f"has branch vertex count {pair['branch_vertex_count']}."
            )
        rows.append(
            {
                "peak_sequence": peak,
                "peak_aa_sequence": pair["birth_aa_sequence"],
                "tree_x": x_of.get(peak),
                "birth_scalar": pair["birth_scalar"],
                "death_sequence": pair["death_sequence"],
                "death_scalar": pair["death_scalar"],
                "is_global_pair": pair["pair_kind"] == GLOBAL_PAIR_KIND,
                "adjacent_saddle_sequence": pair["adjacent_saddle_sequence"],
                "adjacent_saddle_scalar": pair["adjacent_saddle_scalar"],
                "adjacent_saddle_aa_sequence": pair.get("adjacent_saddle_aa_sequence"),
                "saddle_is_synonymous": (
                    pair.get("adjacent_saddle_aa_sequence") is not None
                    and pair.get("adjacent_saddle_aa_sequence")
                    == pair["birth_aa_sequence"]
                ),
                "member_count": count,
                "is_singleton": count == 1,
                "birth_has_stop_codon": pair["birth_has_stop_codon"],
                "is_wild_type": peak in wild_type,
            }
        )
    leaves = (
        pl.DataFrame(
            rows,
            schema={
                "peak_sequence": pl.String,
                "peak_aa_sequence": pl.String,
                "tree_x": pl.Float64,
                "birth_scalar": pl.Float64,
                "death_sequence": pl.String,
                "death_scalar": pl.Float64,
                "is_global_pair": pl.Boolean,
                "adjacent_saddle_sequence": pl.String,
                "adjacent_saddle_scalar": pl.Float64,
                "adjacent_saddle_aa_sequence": pl.String,
                "saddle_is_synonymous": pl.Boolean,
                "member_count": pl.Int64,
                "is_singleton": pl.Boolean,
                "birth_has_stop_codon": pl.Boolean,
                "is_wild_type": pl.Boolean,
            },
        )
        .sort(["birth_scalar", "peak_sequence"], descending=[True, False])
        .with_row_index("rank", offset=1)
        .with_columns(pl.col("rank").cast(pl.Int64))
    )
    if x_of:
        if leaves["tree_x"].null_count():
            raise ValueError("Every peak must have an x position in the tree layout.")
        order = leaves.sort(
            ["tree_x", "birth_scalar", "peak_sequence"], descending=[False, True, False]
        ).with_row_index("tree_order", offset=1)
        leaves = leaves.join(
            order.select("peak_sequence", pl.col("tree_order").cast(pl.Int64)),
            on="peak_sequence",
        )
    else:
        leaves = leaves.with_columns(pl.col("rank").alias("tree_order"))
    leaves = leaves.sort("rank")
    peak_aa = dict(leaves.select("peak_sequence", "peak_aa_sequence").iter_rows())
    member_rows = []
    stop_peaks = set(
        leaves.filter(pl.col("birth_has_stop_codon"))["peak_sequence"].to_list()
    )
    order_of = dict(leaves.select("rank", "tree_order").iter_rows())
    for leaf in leaves.select("rank", "peak_sequence", "birth_scalar").to_dicts():
        peak = leaf["peak_sequence"]
        for dna in hills[peak]:
            member_rows.append(
                {
                    "rank": leaf["rank"],
                    "leaf_peak_sequence": peak,
                    "dna_sequence": dna,
                    "raw_fitness": leaf["birth_scalar"]
                    if dna == peak
                    else fitness[dna],
                    "is_peak": dna == peak,
                    "tree_order": order_of[leaf["rank"]],
                    "is_stop_codon": (peak in stop_peaks)
                    if dna == peak
                    else dna in stops,
                    # Another sequence of the hill with the peak's amino acids.
                    "is_synonymous": dna != peak
                    and aa_of.get(dna) is not None
                    and aa_of[dna] == peak_aa[peak],
                }
            )
    members = pl.DataFrame(
        member_rows,
        schema={
            "rank": pl.Int64,
            "leaf_peak_sequence": pl.String,
            "dna_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "is_peak": pl.Boolean,
            "tree_order": pl.Int64,
            "is_stop_codon": pl.Boolean,
            "is_synonymous": pl.Boolean,
        },
    ).sort(["rank", "raw_fitness", "dna_sequence"], descending=[False, True, False])
    return LeafTables(leaves=leaves, members=members)


def marker_areas(counts: list[int], *, max_area: float, min_area: float) -> list[float]:
    """Return marker areas proportional to ``counts``, with a legibility floor.

    The largest count gets ``max_area``; every other count gets ``max_area * count /
    max(counts)``. ``min_area`` is applied only where that falls below it, so the
    floor never changes a marker that is already legible.
    """
    if not counts:
        return []
    if min(counts) < 1:
        raise ValueError("Hill sequence counts must be at least one.")
    top = max(counts)
    return [max(max_area * count / top, min_area) for count in counts]


def with_marker_area(
    leaves: pl.DataFrame, *, max_area: float, min_area: float
) -> pl.DataFrame:
    """Add ``marker_area`` (see :func:`marker_areas`) to a leaf table."""
    return leaves.with_columns(
        pl.Series(
            "marker_area",
            marker_areas(
                leaves["member_count"].to_list(), max_area=max_area, min_area=min_area
            ),
            dtype=pl.Float64,
        )
    )


def fitness_axis(
    low: float, high: float, appearance: dict[str, Any]
) -> tuple[list[float], list[float]]:
    """Return the padded display domain and the ticks of a fitness axis.

    The domain is the shared one (``fitness_padding_fraction``); the ticks are the
    outward-rounded bounds (``fitness_tick_step``) and zero, as on the other figures.
    """
    step = float(appearance.get("fitness_tick_step", 0.1))
    precision = max(0, math.ceil(-math.log10(step)))
    minimum = round(math.floor(low / step) * step, precision)
    maximum = round(math.ceil(high / step) * step, precision)
    domain = padded_scalar_domain((low, high), appearance)
    return domain, sorted({minimum, 0.0, maximum})


def saved_tree_leaf_tables(inputs: Any) -> LeafTables:
    """Leaf and member tables of a saved tree (``SavedTreeInputs``), nominal or band.

    The band-simplified trees of the D10 views carry no Rustworkx graph, so the
    hills are read from the saved layout: a peak's hill is the ``assignments`` rows
    whose upper sequence is the peak (the peak itself, location ``critical``, and
    the members of its terminal branch; the adjacent saddle is the terminal edge's
    lower node and is not a member). The adjacent saddle and the branch vertex count
    come from that terminal edge of the layout's ``edge_table`` (either direction). The result has the
    columns of :func:`leaf_tables`; nothing is recomputed and no Rustworkx index is
    exposed.
    """
    view = inputs.tree_analysis
    nodes = view.plot_data.node_table
    edges = view.plot_data.edge_table
    assignments = view.assignments
    persistence = inputs.persistence_table
    kinds = persistence["pair_kind"].to_list()
    if kinds.count(GLOBAL_PAIR_KIND) != 1 or set(kinds) - PAIR_KINDS:
        raise ValueError(
            "The persistence table must hold ordinary pairs and exactly one global "
            f"pair (pair_kind {GLOBAL_PAIR_KIND!r}), found {sorted(set(kinds))}."
        )
    node_of = {
        str(row["dna_sequence"]): row for row in nodes.iter_rows(named=True)
    }
    by_index = {int(row["tree_node_index"]): row for row in node_of.values()}
    # The layout's edges may run either way; a terminal edge has one maximum end.
    terminal: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for edge in edges.iter_rows(named=True):
        ends = (
            by_index[int(edge["source_tree_node_index"])],
            by_index[int(edge["target_tree_node_index"])],
        )
        tops = [end for end in ends if end["event"] == "maximum"]
        if len(tops) == 1:
            peak = str(tops[0]["dna_sequence"])
            if peak in terminal:
                raise ValueError(f"Peak {peak!r} has more than one terminal edge.")
            lower = ends[0] if ends[1] is tops[0] else ends[1]
            terminal[peak] = (edge, lower)
    # A band layout's edge_table keeps the nominal layout's counts; the saved tree's own
    # branch counts are on ``branch_table`` (restored by ``tree_layout_view``).
    counts_table = getattr(view, "branch_table", None)
    if counts_table is None:
        counts_table = edges
    counts = {
        frozenset((int(a), int(b))): int(count)
        for a, b, count in counts_table.select(
            "source_tree_node_index", "target_tree_node_index", "branch_vertex_count"
        ).iter_rows()
    }
    births = persistence["birth_label"].to_list()
    if set(terminal) != set(births):
        raise ValueError("Every tree peak must be exactly one persistence birth.")
    rows = []
    for pair in persistence.iter_rows(named=True):
        peak = str(pair["birth_label"])
        edge, saddle = terminal[peak]
        rows.append(
            {
                "birth_sequence": peak,
                "birth_aa_sequence": node_of[peak]["aa_sequence"],
                "birth_scalar": pair["birth_scalar"],
                "death_sequence": pair["death_label"],
                "death_scalar": pair["death_scalar"],
                "pair_kind": pair["pair_kind"],
                "adjacent_saddle_sequence": saddle["dna_sequence"],
                "adjacent_saddle_scalar": saddle["scalar"],
                "adjacent_saddle_aa_sequence": saddle["aa_sequence"],
                "branch_vertex_count": counts[
                    frozenset(
                        (int(edge["source_tree_node_index"]), int(edge["target_tree_node_index"]))
                    )
                ],
                "birth_has_stop_codon": bool(node_of[peak]["has_stop_codon"]),
            }
        )
    pairs = pl.DataFrame(rows)
    members: dict[str, list[str]] = {peak: [peak] for peak in births}
    for dna, upper in assignments.select("dna_sequence", "upper_sequence").iter_rows():
        if upper in members and dna != upper:
            members[upper].append(str(dna))
    fitness = dict(assignments.select("dna_sequence", "raw_fitness").iter_rows())
    aa_of = dict(assignments.select("dna_sequence", "aa_sequence").iter_rows())
    stops = (
        {
            str(dna)
            for dna, flag in assignments.select("dna_sequence", "has_stop_codon").iter_rows()
            if flag
        }
        if "has_stop_codon" in assignments.columns
        else set()
    )
    return _assemble(
        pairs,
        {peak: tuple(found) for peak, found in members.items()},
        wild_type_identities(view),
        {str(k): float(v) for k, v in fitness.items()},
        {str(k): str(v) for k, v in aa_of.items()},
        stops=stops,
        x_of={peak: float(node_of[peak]["x"]) for peak in births},
    )
