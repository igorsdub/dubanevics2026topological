"""Peak-pair distances on a hand-built split tree (6 nt = 2 codons).

Edges point from a node toward the root, as ``save_band_tree`` writes them
(``source_tree_node_identity`` is the upper node). Tree, raw fitness in brackets::

        A [5]   B [4]
          \\     /
           S2 [3.5]     C [3]
              \\         /
               S1 [2]        D [2.5]
                  \\         /
                   R [1] (root)

Sequences (codon 1 | codon 2) and amino acids::

    A  GCT|AAA  AK      S2 GCG|AAA  AK
    B  GCC|AAA  AK      S1 TCT|AAA  SK
    C  GCT|GAA  AE      R  TGT|AAC  CN
    D  TCT|GAC  SD

Leaf order is A, B, C, D: at every fork the child reaching the highest peak
goes first (A 5 > B 4 under S2; S2's subtree 5 > C 3 under S1; S1's 5 > D 2.5).
Every expected value below is derived by hand from the sequences above.
"""

from __future__ import annotations

from itertools import permutations as relabellings

import numpy as np
import polars as pl
import pytest
import rustworkx as rx
from peak_pair_toy import (
    EDGES,
    FITNESS,
    S1,
    S2,
    A,
    B,
    C,
    D,
    R,
    toy_tree,
)
from scipy.stats import spearmanr
from topographer.plotting import make_tree_layout

from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.analysis.peak_pair_distances import (
    MANTEL_COLUMNS,
    OUTPUT_COLUMNS,
    RHO_TOLERANCE,
    _matrix,
    band_tree_tables,
    mantel_test,
    peak_leaf_order,
    peak_pair_distances,
    peak_pair_distances_settings,
    peak_pair_mantel_table,
    peak_pair_tables,
)
from dubanevics2026topological.simplification.selected_leaf_simplification import (
    rebuild_split_tree,
)

_tree = toy_tree


# (peak_1, peak_2): (saddle, nt direct, nt via, aa direct, aa via, fit direct, fit via)
EXPECTED = {
    # A,B differ at site 3 (T/C, synonymous: GCT, GCC both Ala). Saddle S2 GCG
    # differs from both at that site: via 1+1 = 2; amino acids all AK, so 0 and 0.
    # Fitness: |5-4| = 1; (5-3.5)+(4-3.5) = 2 = 1 + 2*(4-3.5).
    (A, B): (S2, 1, 2, 0, 0, 1.0, 2.0),
    # A,C differ at site 4 only. Saddle S1 (TCT AAA) differs from A at site 1 and
    # from C at sites 1 and 4: via 1+2 = 3 (excess 2, site 1 where peaks agree).
    # Amino: AK/AE 1; AK-SK 1, SK-AE 2 -> 3. Fitness: 2; 3+1 = 4 = 2 + 2*(3-2).
    (A, C): (S1, 1, 3, 1, 3, 2.0, 4.0),
    # A,D differ at sites 1,4,6 -> 3. Saddle R (TGT AAC): A-R sites 1,2,6 = 3,
    # R-D sites 2,4 = 2 -> 5. Amino AK/SD 2; AK-CN 2, CN-SD 2 -> 4.
    # Fitness: 2.5; 4+1.5 = 5.5 = 2.5 + 2*(2.5-1).
    (A, D): (R, 3, 5, 2, 4, 2.5, 5.5),
    # B,C differ at sites 3,4 -> 2. Saddle S1: B-S1 sites 1,3 = 2, S1-C = 2 -> 4.
    # Amino AK/AE 1; AK-SK 1, SK-AE 2 -> 3. Fitness: 1; 2+1 = 3 = 1 + 2*(3-2).
    (B, C): (S1, 2, 4, 1, 3, 1.0, 3.0),
    # B,D differ at sites 1,3,4,6 -> 4. Saddle R: B-R sites 1,2,3,6 = 4, R-D = 2
    # -> 6. Amino AK/SD 2; AK-CN 2, CN-SD 2 -> 4.
    # Fitness: 1.5; 3+1.5 = 4.5 = 1.5 + 2*(2.5-1).
    (B, D): (R, 4, 6, 2, 4, 1.5, 4.5),
    # C,D differ at sites 1,6 -> 2. Saddle R: C-R sites 1,2,4,6 = 4, R-D = 2 -> 6.
    # Amino AE/SD 2; AE-CN 2, CN-SD 2 -> 4. Fitness: 0.5; 2+1.5 = 3.5 = 0.5 + 2*(2.5-1).
    (C, D): (R, 2, 6, 2, 4, 0.5, 3.5),
}


def test_leaf_order_puts_highest_peak_first() -> None:
    assert peak_leaf_order(*_tree()) == [A, B, C, D]


def test_pair_count_and_order() -> None:
    table = peak_pair_distances(*_tree())
    assert table.height == 4 * 3 // 2
    pairs = list(zip(table["peak_1_dna"], table["peak_2_dna"]))
    assert pairs == list(EXPECTED)
    assert table["peak_1_leaf_order"].to_list() == [0, 0, 0, 1, 1, 2]
    assert table["peak_2_leaf_order"].to_list() == [1, 2, 3, 2, 3, 3]


def test_hand_computed_values() -> None:
    table = peak_pair_distances(*_tree())
    for row in table.to_dicts():
        saddle, nt_d, nt_v, aa_d, aa_v, f_d, f_v = EXPECTED[
            (row["peak_1_dna"], row["peak_2_dna"])
        ]
        assert row["common_saddle_dna"] == saddle
        assert row["nucleotide_direct_hamming"] == nt_d
        assert row["nucleotide_via_saddle_hamming"] == nt_v
        assert row["amino_acid_direct_hamming"] == aa_d
        assert row["amino_acid_via_saddle_hamming"] == aa_v
        assert row["direct_fitness_difference"] == pytest.approx(f_d)
        assert row["via_saddle_fitness_distance"] == pytest.approx(f_v)


def test_synonymous_pair_has_zero_amino_acid_distance() -> None:
    row = peak_pair_distances(*_tree()).row(0, named=True)
    assert row["nucleotide_direct_hamming"] == 1
    assert row["amino_acid_direct_hamming"] == 0


def test_excess_two_where_saddle_differs_at_agreed_site() -> None:
    rows = {
        (r["peak_1_dna"], r["peak_2_dna"]): r
        for r in peak_pair_distances(*_tree()).to_dicts()
    }
    row = rows[(A, C)]
    assert row["nucleotide_via_saddle_hamming"] - row["nucleotide_direct_hamming"] == 2


def test_nested_saddles_differ() -> None:
    rows = {
        (r["peak_1_dna"], r["peak_2_dna"]): r["common_saddle_dna"]
        for r in peak_pair_distances(*_tree()).to_dicts()
    }
    # S2 lies above S1, which lies above R.
    assert (rows[(A, B)], rows[(A, C)], rows[(A, D)]) == (S2, S1, R)


def test_invariants() -> None:
    for row in peak_pair_distances(*_tree()).to_dicts():
        assert row["nucleotide_via_saddle_hamming"] >= row["nucleotide_direct_hamming"]
        assert row["amino_acid_via_saddle_hamming"] >= row["amino_acid_direct_hamming"]
        lower = min(FITNESS[row["peak_1_dna"]], FITNESS[row["peak_2_dna"]])
        saddle = FITNESS[row["common_saddle_dna"]]
        assert row["via_saddle_fitness_distance"] - row[
            "direct_fitness_difference"
        ] == pytest.approx(2 * (lower - saddle))


def test_no_integer_node_indices() -> None:
    table = peak_pair_distances(*_tree())
    for name in (
        "peak_1_dna",
        "peak_2_dna",
        "common_saddle_dna",
    ):
        assert table.schema[name] == pl.String
    assert not any("index" in column for column in table.columns)
    assert set(table["peak_1_dna"]) | set(table["peak_2_dna"]) <= {A, B, C, D}


# ---------------------------------------------------------------------------
# Leaf order against Topographer's own layout, edge cases and validation.
# ---------------------------------------------------------------------------


def _tables(
    fitness: dict[str, float], edges: list[tuple[str, str]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    critical = pl.DataFrame(
        {"tree_node_identity": list(fitness), "scalar": list(fitness.values())},
        schema={"tree_node_identity": pl.String, "scalar": pl.Float64},
    )
    branches = pl.DataFrame(
        {
            "source_tree_node_identity": [s for s, _ in edges],
            "target_tree_node_identity": [t for _, t in edges],
        },
        schema={
            "source_tree_node_identity": pl.String,
            "target_tree_node_identity": pl.String,
        },
    )
    return critical, branches


def _topographer_order(
    fitness: dict[str, float], edges: list[tuple[str, str]]
) -> list[str]:
    """Leaf order of Topographer's layout on a graph built in reversed order."""
    graph = rx.PyDiGraph()
    index = {
        dna: graph.add_node({"dna_sequence": dna, "scalar": value})
        for dna, value in reversed(list(fitness.items()))
    }
    for child, up in reversed(edges):
        graph.add_edge(index[child], index[up], {})
    layout = make_tree_layout(
        rebuild_split_tree(graph), horizontal_layout="subtree_extremum"
    )
    sources = set(layout.edge_table["source_tree_node_index"].to_list())
    leaves = layout.node_table.filter(~pl.col("tree_node_index").is_in(sources)).sort(
        "x"
    )
    return leaves["vertex_id"].to_list()


def test_leaf_order_ties_on_maximum_fitness_follow_topographer() -> None:
    # Both root subtrees reach 5. The left one has three nodes, the right one a
    # single peak whose DNA sorts first alphabetically, so only Topographer's
    # tie-break (larger subtree first) yields X, Y, Z.
    x, y, z, sa, root = "GCC", "GCA", "AAG", "GCT", "TGT"
    fitness = {root: 1.0, sa: 2.5, x: 5.0, y: 4.0, z: 5.0}
    edges = [(sa, root), (x, sa), (y, sa), (z, root)]
    expected = _topographer_order(fitness, edges)
    assert expected == [x, y, z]
    assert peak_leaf_order(*_tables(fitness, edges)) == expected


def test_leaf_order_fully_tied_siblings_ignore_row_order() -> None:
    p, q, s, root = "GCC", "AAG", "GCT", "TGT"
    fitness = {root: 1.0, s: 2.0, p: 5.0, q: 5.0}
    edges = [(s, root), (p, s), (q, s)]
    expected = _topographer_order(fitness, edges)
    assert peak_leaf_order(*_tables(fitness, edges)) == expected
    reversed_fitness = dict(reversed(list(fitness.items())))
    assert peak_leaf_order(*_tables(reversed_fitness, edges[::-1])) == expected


def test_equal_fitness_peaks_have_zero_direct_fitness_difference() -> None:
    p, q, s, root = "GCC", "GCA", "GCT", "TGT"
    table = peak_pair_distances(
        *_tables({root: 1.0, s: 3.0, p: 5.0, q: 5.0}, [(s, root), (p, s), (q, s)])
    )
    assert table.height == 1
    row = table.row(0, named=True)
    assert row["direct_fitness_difference"] == 0.0
    assert row["via_saddle_fitness_distance"] == pytest.approx(4.0)
    assert row["common_saddle_dna"] == s


def test_two_peak_tree_has_one_row() -> None:
    table = peak_pair_distances(
        *_tables({R: 1.0, S2: 3.5, A: 5.0, B: 4.0}, [(S2, R), (A, S2), (B, S2)])
    )
    assert table.height == 1
    row = table.row(0, named=True)
    assert (row["peak_1_dna"], row["peak_2_dna"]) == (A, B)
    assert row["common_saddle_dna"] == S2
    assert row["nucleotide_via_saddle_hamming"] == 2  # A-S2 1, S2-B 1


def test_one_node_tree_gives_empty_table_with_schema() -> None:
    table = peak_pair_distances(*_tables({R: 1.0}, []))
    assert table.height == 0
    assert tuple(table.columns) == OUTPUT_COLUMNS
    for name, dtype in table.schema.items():
        if "hamming" in name or "leaf_order" in name:
            assert dtype == pl.Int64
        elif "fitness" in name:
            assert dtype == pl.Float64
        else:
            assert dtype == pl.String


def test_degree_one_global_minimum_root_is_not_a_peak() -> None:
    # Real split trees end in a degree-1 global minimum under the trailing saddle:
    # R(1) <- S1(2) <- {S2(3.5) <- {A, B}, C}. Literals from the hand
    # computation in the module docstring (same sequences, no D).
    fitness = {R: 1.0, S1: 2.0, S2: 3.5, A: 5.0, B: 4.0, C: 3.0}
    edges = [(S1, R), (S2, S1), (C, S1), (A, S2), (B, S2)]
    critical, branches = _tables(fitness, edges)
    assert peak_leaf_order(critical, branches) == [A, B, C]
    table = peak_pair_distances(critical, branches)
    rows = {(r["peak_1_dna"], r["peak_2_dna"]): r for r in table.to_dicts()}
    assert set(rows) == {(A, B), (A, C), (B, C)}
    assert rows[(A, B)]["common_saddle_dna"] == S2
    assert rows[(A, C)]["common_saddle_dna"] == S1
    assert rows[(B, C)]["common_saddle_dna"] == S1
    assert rows[(A, C)]["nucleotide_via_saddle_hamming"] == 3
    assert rows[(B, C)]["nucleotide_via_saddle_hamming"] == 4
    assert rows[(B, C)]["via_saddle_fitness_distance"] == pytest.approx(3.0)
    assert R not in set(table["peak_1_dna"]) | set(table["peak_2_dna"])


def test_save_band_tree_column_set_round_trips_through_parquet(tmp_path) -> None:
    fitness = {R: 1.0, S1: 2.0, S2: 3.5, A: 5.0, B: 4.0, C: 3.0}
    edges = [(S1, R), (S2, S1), (C, S1), (A, S2), (B, S2)]
    events = {R: "regular", S1: "split_saddle", S2: "split_saddle"}
    critical = pl.DataFrame(
        [
            {
                "tree_node_identity": dna,
                "scalar": value,
                "event": events.get(dna, "maximum"),
                "vertex_id": dna,
                "label": dna,
                "dna_sequence": dna,
            }
            for dna, value in fitness.items()
        ]
    )
    branches = pl.DataFrame(
        {
            "source_tree_node_identity": [s for s, _ in edges],
            "target_tree_node_identity": [t for _, t in edges],
            "branch_vertex_count": [0] * len(edges),
            "source_scalar": [fitness[s] for s, _ in edges],
            "target_scalar": [fitness[t] for _, t in edges],
        }
    )
    critical.write_parquet(tmp_path / "critical_nodes.parquet")
    branches.write_parquet(tmp_path / "branches.parquet")
    result = peak_pair_distances(
        pl.read_parquet(tmp_path / "critical_nodes.parquet"),
        pl.read_parquet(tmp_path / "branches.parquet"),
    )
    assert result.equals(peak_pair_distances(*_tables(fitness, edges)))
    assert result.height == 3


def test_error_paths() -> None:
    with pytest.raises(ValueError, match="one root"):
        peak_pair_distances(*_tables({A: 2.0, B: 1.0, C: 1.0}, [(A, B)]))
    with pytest.raises(ValueError, match="more than one edge"):
        peak_pair_distances(
            *_tables(
                {A: 5.0, S1: 2.0, S2: 3.0, R: 1.0}, [(A, S1), (A, S2), (S1, R), (S2, R)]
            )
        )
    with pytest.raises(ValueError, match="same length"):
        peak_pair_distances(
            *_tables(
                {"GCTAAA": 1.0, "GCT": 2.0, "GCC": 3.0},
                [("GCT", "GCTAAA"), ("GCC", "GCTAAA")],
            )
        )
    with pytest.raises(ValueError, match="Duplicate"):
        critical = pl.DataFrame(
            {"tree_node_identity": [A, A, R], "scalar": [5.0, 5.0, 1.0]}
        )
        _, branches = _tables({A: 5.0, R: 1.0}, [(A, R)])
        peak_pair_distances(critical, branches)
    with pytest.raises(ValueError, match="missing from critical_nodes"):
        peak_pair_distances(*_tables({A: 5.0, R: 1.0}, [(A, R), (B, R)]))
    with pytest.raises(ValueError, match="missing from critical_nodes"):
        peak_pair_distances(*_tables({A: 5.0, R: 1.0}, [(A, R), (R, S1)]))


# ---------------------------------------------------------------------------
# Mantel test, the comparison table, settings and the band-tree tables.
# ---------------------------------------------------------------------------


def _symmetric(values: list[float], n: int) -> np.ndarray:
    """Square symmetric matrix with ``values`` on the upper triangle (row-major)."""
    matrix = np.zeros((n, n))
    matrix[np.triu_indices(n, k=1)] = values
    return matrix + matrix.T


# Four peaks, six pairs in row-major order: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
X4 = _symmetric([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4)
Y4_MONOTONE = _symmetric([1.0, 4.0, 9.0, 16.0, 25.0, 36.0], 4)
Y4_REVERSED = _symmetric([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 4)


def test_mantel_perfectly_monotone_gives_rho_one() -> None:
    rho, p = mantel_test(X4, Y4_MONOTONE, permutations=99, seed=1)
    assert rho == pytest.approx(1.0)
    # Only the identity relabelling can reach |rho| = 1 for six distinct values
    # unless a symmetry exists; either way p is on the grid and at least 1/100.
    assert 1 / 100 <= p <= 1.0


def test_mantel_reversed_gives_rho_minus_one_and_is_two_sided() -> None:
    rho, p = mantel_test(X4, Y4_REVERSED, permutations=99, seed=1)
    assert rho == pytest.approx(-1.0)
    assert p == mantel_test(X4, Y4_MONOTONE, permutations=99, seed=1)[1]


def test_mantel_p_formula_with_tiny_permutation_counts() -> None:
    for permutations in (1, 2, 3):
        _, p = mantel_test(X4, Y4_MONOTONE, permutations=permutations, seed=5)
        assert p * (1 + permutations) == pytest.approx(round(p * (1 + permutations)))
        assert 1 <= round(p * (1 + permutations)) <= 1 + permutations


def _tied_matrices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs = n * (n - 1) // 2
    return (
        _symmetric(list(rng.integers(0, 4, size=pairs).astype(float)), n),
        _symmetric(list(rng.integers(0, 5, size=pairs).astype(float)), n),
    )


@pytest.mark.parametrize("data_seed", [0, 1, 2])
def test_mantel_p_is_exact_for_the_seeded_permutations(data_seed: int) -> None:
    # Rebuild the generator's relabellings and score each with scipy's spearmanr.
    n, draws, seed = 8, 300, 17
    x, y = _tied_matrices(n, data_seed)
    iu = np.triu_indices(n, k=1)
    rho, p = mantel_test(x, y, permutations=draws, seed=seed)
    assert rho == pytest.approx(spearmanr(x[iu], y[iu]).statistic)
    orders = np.argsort(np.random.default_rng(seed).random((draws, n)), axis=1)
    count = sum(
        abs(spearmanr(x[iu], y[np.ix_(o, o)][iu]).statistic) >= abs(rho) - RHO_TOLERANCE
        for o in orders
    )
    assert p == (1 + count) / (1 + draws)


def test_mantel_chunking_does_not_change_the_result() -> None:
    x, y = _tied_matrices(9, 5)
    whole = mantel_test(x, y, permutations=1234, seed=3, chunk=10_000)
    for chunk in (1, 7, 500, 1234):
        assert mantel_test(x, y, permutations=1234, seed=3, chunk=chunk) == whole
    assert mantel_test(x, y, permutations=1234, seed=3) == whole


def test_mantel_null_matches_the_enumerated_relabellings() -> None:
    # Four peaks have 24 relabellings. Enumerate them: the share with |rho| = 1
    # is the null probability that the seeded sample must approximate.
    iu = np.triu_indices(4, k=1)
    hits = sum(
        abs(spearmanr(X4[iu], Y4_MONOTONE[np.ix_(o, o)][iu]).statistic) >= 1 - 1e-12
        for o in relabellings(range(4))
    )
    assert hits >= 1
    draws = 5000
    _, p = mantel_test(X4, Y4_MONOTONE, permutations=draws, seed=3)
    assert p == pytest.approx((1 + draws * hits / 24) / (1 + draws), abs=0.02)


def test_mantel_is_deterministic_for_a_seed_and_differs_between_seeds() -> None:
    rng = np.random.default_rng(0)
    x = _symmetric(list(rng.normal(size=45)), 10)
    y = _symmetric(list(rng.normal(size=45)), 10)
    first = mantel_test(x, y, permutations=200, seed=7)
    assert mantel_test(x, y, permutations=200, seed=7) == first
    assert first[0] == mantel_test(x, y, permutations=200, seed=8)[0]
    assert first[1] != mantel_test(x, y, permutations=200, seed=8)[1]


def test_mantel_rho_is_spearman_of_the_upper_triangle_with_ties() -> None:
    x = _symmetric([1, 1, 2, 3, 3, 3], 4)
    y = _symmetric([5, 4, 4, 1, 2, 2], 4)
    iu = np.triu_indices(4, k=1)
    rho, _ = mantel_test(x, y, permutations=9, seed=0)
    assert rho == pytest.approx(spearmanr(x[iu], y[iu]).statistic)


def test_mantel_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mantel_test(X4, np.zeros((3, 3)), permutations=9, seed=0)
    with pytest.raises(ValueError, match="at least 3"):
        mantel_test(np.zeros((2, 2)), np.zeros((2, 2)), permutations=9, seed=0)
    with pytest.raises(ValueError, match="permutations"):
        mantel_test(X4, Y4_MONOTONE, permutations=0, seed=0)


def test_constant_matrix_gives_nan_not_an_error() -> None:
    rho, p = mantel_test(X4, np.ones((4, 4)), permutations=9, seed=0)
    assert np.isnan(rho)
    assert np.isnan(p)


def test_constant_comparison_writes_a_nan_row() -> None:
    # Every pair has the same direct fitness difference: that comparison is NaN,
    # the other five rows are unaffected.
    pairs = peak_pair_distances(*_tree()).with_columns(
        direct_fitness_difference=pl.lit(1.0)
    )
    table = peak_pair_mantel_table(pairs, permutations=19, seed=1)
    assert table.height == 6
    row = table.filter(
        (pl.col("comparison") == "direct") & (pl.col("alphabet") == "nucleotide")
    ).row(0, named=True)
    assert np.isnan(row["spearman_rho"])
    assert np.isnan(row["mantel_p"])
    assert (
        table.filter(pl.col("comparison") == "via_saddle")["spearman_rho"]
        .is_not_nan()
        .all()
    )


def test_mantel_table_has_six_rows_with_hand_checked_rho() -> None:
    pairs = peak_pair_distances(*_tree())
    table = peak_pair_mantel_table(pairs, permutations=99, seed=11)
    assert table.columns == list(MANTEL_COLUMNS)
    assert table.height == 6
    assert (
        table["comparison"].to_list()
        == [
            "direct",
            "via_saddle",
            "via_minus_direct",
        ]
        * 2
    )
    assert table["alphabet"].to_list() == ["nucleotide"] * 3 + ["amino_acid"] * 3
    assert set(table["permutations"]) == {99}
    assert set(table["seed"]) == {11}
    assert set(table["peak_count"]) == {4}
    assert set(table["pair_count"]) == {6}
    rows = {(r["comparison"], r["alphabet"]): r for r in table.to_dicts()}
    # Direct nucleotide Hamming in the toy's pair order (A,B) (A,C) (A,D)
    # (B,C) (B,D) (C,D): Hamming 1,1,3,2,4,2 against fitness 1,2,2.5,1,1.5,0.5.
    hamming = [1, 1, 3, 2, 4, 2]
    fitness = [1.0, 2.0, 2.5, 1.0, 1.5, 0.5]
    assert rows[("direct", "nucleotide")]["spearman_rho"] == pytest.approx(
        spearmanr(hamming, fitness).statistic
    )
    via_h = [2, 3, 5, 4, 6, 6]
    via_f = [2.0, 4.0, 5.5, 3.0, 4.5, 3.5]
    assert rows[("via_saddle", "nucleotide")]["spearman_rho"] == pytest.approx(
        spearmanr(via_h, via_f).statistic
    )
    delta_h = [b - a for a, b in zip(hamming, via_h)]
    delta_f = [b - a for a, b in zip(fitness, via_f)]
    assert rows[("via_minus_direct", "nucleotide")]["spearman_rho"] == pytest.approx(
        spearmanr(delta_h, delta_f).statistic
    )
    aa_direct = [0, 1, 2, 1, 2, 2]
    assert rows[("direct", "amino_acid")]["spearman_rho"] == pytest.approx(
        spearmanr(aa_direct, fitness).statistic
    )


def test_peak_pair_tables_return_pairs_and_mantel() -> None:
    critical, branches = _tree()
    pairs, mantel = peak_pair_tables(
        critical,
        branches,
        {"tree": "band", "mantel_permutations": 19, "seed": 3},
    )
    assert pairs.columns == list(OUTPUT_COLUMNS)
    assert mantel.height == 6
    assert set(mantel["permutations"]) == {19}


def test_settings_validation() -> None:
    good = {"tree": "band", "mantel_permutations": 9999, "seed": 20261121}
    assert peak_pair_distances_settings(good) == good
    with pytest.raises(ValueError, match="tree"):
        peak_pair_distances_settings({**good, "tree": "nominal"})
    with pytest.raises(ValueError, match="mantel_permutations"):
        peak_pair_distances_settings({**good, "mantel_permutations": 0})
    with pytest.raises(TypeError, match="seed"):
        peak_pair_distances_settings({**good, "seed": "x"})
    with pytest.raises(ValueError, match="Unknown"):
        peak_pair_distances_settings({**good, "extra": 1})
    with pytest.raises(ValueError, match="needs"):
        peak_pair_distances_settings({"tree": "band"})


def test_configured_settings_validate() -> None:
    section = load_settings()["analysis"]["peak_pair_distances"]
    assert peak_pair_distances_settings(section) == section
    assert section["tree"] == "band"
    assert section["mantel_permutations"] == 9999


def test_band_tree_tables_match_save_band_tree_columns() -> None:
    graph = rx.PyDiGraph()
    index = {
        dna: graph.add_node(
            {"dna_sequence": dna, "scalar": value, "vertex_index": i, "kind": "x"}
        )
        for i, (dna, value) in enumerate(FITNESS.items())
    }
    for upper, lower in EDGES:
        graph.add_edge(index[upper], index[lower], None)

    class View:
        tree = graph

    critical, branches = band_tree_tables(View())
    assert set(critical["tree_node_identity"]) == set(FITNESS)
    assert not any(column.endswith("_index") for column in critical.columns)
    assert set(
        zip(
            branches["source_tree_node_identity"], branches["target_tree_node_identity"]
        )
    ) == set(EDGES)
    assert peak_pair_distances(critical, branches).height == 6


def test_amino_acid_mantel_rows_use_the_amino_acid_hamming_matrices() -> None:
    pairs = peak_pair_distances(*_tree())
    table = peak_pair_mantel_table(pairs, permutations=19, seed=3)
    n = 4
    fitness = {
        "direct": pairs["direct_fitness_difference"],
        "via_saddle": pairs["via_saddle_fitness_distance"],
    }
    fitness["via_minus_direct"] = fitness["via_saddle"] - fitness["direct"]
    amino = {
        "direct": pairs["amino_acid_direct_hamming"].cast(pl.Float64),
        "via_saddle": pairs["amino_acid_via_saddle_hamming"].cast(pl.Float64),
    }
    amino["via_minus_direct"] = amino["via_saddle"] - amino["direct"]
    nucleotide_direct = pairs["nucleotide_direct_hamming"].cast(pl.Float64)
    for comparison in ("direct", "via_saddle", "via_minus_direct"):
        rho, p = mantel_test(
            _matrix(pairs, amino[comparison], n),
            _matrix(pairs, fitness[comparison], n),
            permutations=19,
            seed=3,
        )
        row = table.filter(
            (pl.col("comparison") == comparison) & (pl.col("alphabet") == "amino_acid")
        ).row(0, named=True)
        assert (row["spearman_rho"], row["mantel_p"]) == (rho, p)
    # The toy's amino-acid matrix differs from the nucleotide one, so this is not vacuous.
    assert amino["direct"].to_list() != nucleotide_direct.to_list()
