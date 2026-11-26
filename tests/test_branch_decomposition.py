"""Branch decomposition stage: one branch per tree edge, orders on the smoothed tree, reuse."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import topographer
from _branch_trees import (
    BALANCED_PERSISTENCE,
    balanced_tree,
    hills_table,
    make_tree,
)
from _selection import fixed_selection
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological import build
from dubanevics2026topological.analysis import branch_decomposition as module
from dubanevics2026topological.analysis.branch_decomposition import (
    BRANCH_SCHEMA,
    MEMBER_SCHEMA,
    branch_decomposition_settings,
    compute_branch_decomposition,
    decompose_view,
    decomposition_summary,
    decomposition_view,
    load_cached_branch_decomposition,
    order_summary,
)
from dubanevics2026topological.analysis.checkpoint import checkpoint_lock
from dubanevics2026topological.landscape import selection as candidate_selection
from dubanevics2026topological.simplification.selected_leaf_simplification import (
    absorb_terminal_branches,
    enforce_elder_rule,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "upper_sequence",
    "lower_sequence",
    "peak_sequence",
    "parent_branch_id",
    "is_terminal",
    "is_root_edge",
    "horton_strahler_order",
    "branch_vertex_count",
    "edge_count",
    "birth_fitness",
    "death_fitness",
    "branch_span",
    "member_sequences",
)


def decomposed(tree, absorbed=()):
    view = absorb_terminal_branches(tree, set(absorbed))
    return decompose_view(view, tree.adjacency)


def by_id(decomposition):
    return {
        row["branch_id"]: row for row in decomposition.branches.iter_rows(named=True)
    }


def fields(decomposition):
    """Every row of the branch table as the tuple of ``FIELDS``, keyed by branch."""
    return {
        branch: tuple(row[name] for name in FIELDS)
        for branch, row in by_id(decomposition).items()
    }


def orders(decomposition):
    return {
        branch: row["horton_strahler_order"]
        for branch, row in by_id(decomposition).items()
    }


def real_style_tree(nodes, edges, **options):
    """A tree whose ``vertex_index`` is the identity rank, as on the Candidate tree."""
    return make_tree(sorted(nodes), edges, **options)


# -- Horton–Strahler orders of trees whose orders are known by hand -------------


def test_a_single_fork_gives_two_terminal_branches_of_order_one_and_a_root_edge():
    tree = make_tree(
        [
            ("a", 5.0, "leaf"),
            ("b", 4.0, "leaf"),
            ("s", 2.0, "saddle"),
            ("root", 0.0, "root"),
        ],
        [("a", "s"), ("b", "s"), ("s", "root")],
    )
    result = decomposed(tree)
    assert orders(result) == {"a": 1, "b": 1, "s": 2}
    assert fields(result)["s"][:7] == ("s", "root", "a", None, False, True, 2)


def test_a_balanced_tree_gives_every_edge_its_order():
    result = decomposed(balanced_tree())
    assert orders(result) == {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "s1": 2,
        "s2": 2,
        "s3": 3,
    }
    # Order 1 is exactly the terminal branches, and only they.
    for row in result.branches.iter_rows(named=True):
        assert row["is_terminal"] == (row["horton_strahler_order"] == 1)


def test_a_caterpillar_stays_at_order_two():
    """Tributaries of lower order along one trunk never raise it past 2."""
    tree = make_tree(
        [
            ("a", 9.0, "leaf"),
            ("b", 8.0, "leaf"),
            ("c", 7.0, "leaf"),
            ("s1", 5.0, "saddle"),
            ("s2", 3.0, "saddle"),
            ("root", 0.0, "root"),
        ],
        [("a", "s1"), ("b", "s1"), ("s1", "s2"), ("c", "s2"), ("s2", "root")],
    )
    assert orders(decomposed(tree)) == {"a": 1, "b": 1, "c": 1, "s1": 2, "s2": 2}


def perfect_tree():
    """Eight peaks, joined pairwise three levels deep; the root edge has order 4."""
    nodes = [
        (f"p{i}", fitness, "leaf")
        for i, fitness in enumerate([9.0, 8.0, 7.0, 6.0, 5.5, 5.2, 5.1, 5.0])
    ]
    nodes += [
        ("sa", 4.0, "saddle"),
        ("sb", 3.9, "saddle"),
        ("sc", 3.8, "saddle"),
        ("sd", 3.7, "saddle"),
        ("se", 3.0, "saddle"),
        ("sf", 2.9, "saddle"),
        ("sg", 2.0, "saddle"),
        ("root", 0.0, "root"),
    ]
    edges = [
        ("p0", "sa"),
        ("p1", "sa"),
        ("p2", "sb"),
        ("p3", "sb"),
        ("p4", "sc"),
        ("p5", "sc"),
        ("p6", "sd"),
        ("p7", "sd"),
        ("sa", "se"),
        ("sb", "se"),
        ("sc", "sf"),
        ("sd", "sf"),
        ("se", "sg"),
        ("sf", "sg"),
        ("sg", "root"),
    ]
    return make_tree(nodes, edges)


def test_a_perfect_tree_reaches_order_four_and_counts_branches_by_order():
    result = decomposed(perfect_tree())
    assert orders(result) == {
        **{f"p{i}": 1 for i in range(8)},
        **dict.fromkeys(("sa", "sb", "sc", "sd"), 2),
        **dict.fromkeys(("se", "sf"), 3),
        "sg": 4,
    }
    summary = order_summary(result.branches)
    assert summary["horton_strahler_order"].to_list() == [1, 2, 3, 4]
    assert summary["branch_count"].to_list() == [8, 4, 2, 1]
    # Every sequence but the root sits on exactly one order.
    assert summary["sequence_count"].sum() == result.members.height == 15
    assert summary["maximum_branch_vertex_count"].to_list() == [1, 1, 1, 1]


# -- the fields of a branch ----------------------------------------------------


def test_every_field_of_every_branch_of_the_balanced_tree():
    assert fields(decomposed(balanced_tree())) == {
        # upper, lower, peak, parent, terminal, root edge, order, vertices, edges,
        # birth, death, span, members
        "a": ("a", "s1", "a", "s1", True, False, 1, 2, 1, 9.0, 5.0, 4.0, ["a", "ra"]),
        "b": (
            "b",
            "s1",
            "b",
            "s1",
            True,
            False,
            1,
            3,
            2,
            8.0,
            5.0,
            3.0,
            ["b", "rb1", "rb2"],
        ),
        "c": ("c", "s2", "c", "s2", True, False, 1, 1, 0, 7.0, 4.0, 3.0, ["c"]),
        "d": ("d", "s2", "d", "s2", True, False, 1, 1, 0, 6.0, 4.0, 2.0, ["d"]),
        "s1": ("s1", "s3", "a", "s3", False, False, 2, 1, 0, 5.0, 3.0, 2.0, ["s1"]),
        "s2": ("s2", "s3", "c", "s3", False, False, 2, 1, 0, 4.0, 3.0, 1.0, ["s2"]),
        "s3": ("s3", "root", "a", None, False, True, 3, 1, 0, 3.0, 0.0, 3.0, ["s3"]),
    }


def test_branch_span_is_the_gap_between_the_endpoints_not_a_persistence():
    """Span is its own quantity; the next test says where its value meets persistence."""
    rows = by_id(decomposed(balanced_tree()))
    assert {k: r["branch_span"] for k, r in rows.items()} == {
        "a": 4.0,
        "b": 3.0,
        "c": 3.0,
        "d": 2.0,
        "s1": 2.0,
        "s2": 1.0,
        "s3": 3.0,
    }
    # The peaks' ordinary persistence (a: 9, b: 3, c: 4, d: 2) is another quantity.
    assert rows["a"]["branch_span"] != BALANCED_PERSISTENCE["a"]
    assert "persistence" not in " ".join(BRANCH_SCHEMA)


def test_span_equals_the_peaks_persistence_only_on_a_younger_terminal_branch():
    """Different quantities; the values meet where a peak dies at its own saddle.

    b and d are the younger arms at s1 and s2: each peak dies at the saddle where
    its terminal branch ends, so span and persistence are the same number. The
    elder arms a and c outlive their adjacent saddle (their persistence is larger),
    and an internal branch has no peak of its own at its upper endpoint, so its
    span differs from the persistence of the peak above it.
    """
    tree = balanced_tree()
    view = absorb_terminal_branches(tree, set())
    rows = by_id(decompose_view(view, tree.adjacency))
    persistence = dict(view.pairs.select("birth_label", "persistence").iter_rows())
    assert persistence == BALANCED_PERSISTENCE
    deaths = dict(view.pairs.select("birth_label", "death_label").iter_rows())
    coincide = {
        name
        for name, row in rows.items()
        if row["branch_span"] == persistence[row["peak_sequence"]]
    }
    assert coincide == {"b", "d"}
    for name in coincide:
        assert rows[name]["is_terminal"]
        assert rows[name]["lower_sequence"] == deaths[name]  # dies where it ends
    for name in ("a", "c"):  # terminal branches of the elder arms
        assert rows[name]["is_terminal"]
        assert rows[name]["lower_sequence"] != deaths[name]
        assert rows[name]["branch_span"] < persistence[name]
    for name in ("s1", "s2", "s3"):  # internal branches
        assert not rows[name]["is_terminal"]
        assert rows[name]["branch_span"] != persistence[rows[name]["peak_sequence"]]


def test_edge_count_counts_the_adjacencies_inside_the_branch():
    """Induced edges only: a chord inside a branch counts, an edge across does not."""
    tree = balanced_tree()
    nominal = by_id(decomposed(tree))
    # b holds b-rb1 and rb1-rb2; rb1-c (a chord to branch c) is counted for neither.
    assert nominal["b"]["edge_count"] == 2 and nominal["c"]["edge_count"] == 0
    smoothed = by_id(decomposed(tree, {"b"}))
    # a now holds six sequences joined by six edges (a path of five and the chord
    # a-s1), so the count is neither the vertex count nor the vertex count less one.
    assert smoothed["a"]["branch_vertex_count"] == 6
    assert smoothed["a"]["edge_count"] == 6
    assert smoothed["c"]["edge_count"] == 0


def test_the_summary_counts_what_the_tables_hold():
    result = decomposed(balanced_tree())
    assert decomposition_summary(result, {"tree": "nominal"}) == {
        "settings": {"tree": "nominal"},
        "branch_count": 7,
        "terminal_branch_count": 4,
        "internal_branch_count": 3,
        "branches_per_order": {"1": 4, "2": 2, "3": 1},
        "maximum_order": 3,
        "sequence_count": 11,
        "member_count": 10,
        "root_sequence": "root",
        "root_edge_count": 1,
        "root_group_member_count": 0,
    }
    summary = order_summary(result.branches)
    assert summary["branch_count"].to_list() == [4, 2, 1]
    assert summary["sequence_count"].to_list() == [7, 2, 1]
    assert summary["median_branch_vertex_count"].to_list() == [1.5, 1.0, 1.0]
    assert summary["maximum_branch_vertex_count"].to_list() == [3, 1, 1]


def test_tables_are_sorted_from_the_fittest_upper_endpoint_and_typed():
    result = decomposed(balanced_tree())
    assert result.branches["branch_id"].to_list() == [
        "a",
        "b",
        "c",
        "d",
        "s1",
        "s2",
        "s3",
    ]
    assert dict(result.branches.schema) == BRANCH_SCHEMA
    assert dict(result.members.schema) == MEMBER_SCHEMA
    first = result.members.filter(pl.col("branch_id") == "b")
    assert first["raw_fitness"].to_list() == sorted(first["raw_fitness"], reverse=True)


# -- the long members table is the exploded member_sequences ---------------------


@pytest.mark.parametrize("absorbed", [(), ("b",), ("d",), ("b", "d")])
def test_members_table_is_the_exploded_member_sequences(absorbed):
    tree = balanced_tree()
    view = absorb_terminal_branches(tree, set(absorbed))
    result = decompose_view(view, tree.adjacency)
    fitness = dict(view.assignments.select("dna_sequence", "raw_fitness").iter_rows())
    for row in result.branches.iter_rows(named=True):
        mine = result.members.filter(pl.col("branch_id") == row["branch_id"])
        assert mine["dna_sequence"].to_list() == row["member_sequences"]
        assert mine["horton_strahler_order"].to_list() == [
            row["horton_strahler_order"]
        ] * len(row["member_sequences"])
        assert mine["raw_fitness"].to_list() == [
            fitness[name] for name in row["member_sequences"]
        ]
    exploded = result.branches.select(
        "branch_id", pl.col("member_sequences").alias("dna_sequence")
    ).explode("dna_sequence", empty_as_null=False)
    assert sorted(exploded.iter_rows()) == sorted(
        result.members.select("branch_id", "dna_sequence").iter_rows()
    )
    assert result.members.height == sum(result.branches["branch_vertex_count"])


# -- absorbed branches do not count --------------------------------------------


def test_an_absorbed_hill_no_longer_raises_the_order_and_its_sequences_move():
    tree = balanced_tree()
    nominal = decomposed(tree)
    smoothed = decomposed(tree, {"b"})
    assert orders(nominal)["s3"] == 3
    # b is gone and s1 stops being a confluence (it is contracted), so s3 joins an
    # order-1 arm (a) and an order-2 arm (s2): the root edge drops to order 2.
    assert fields(smoothed) == {
        "a": (
            "a",
            "s3",
            "a",
            "s3",
            True,
            False,
            1,
            6,
            6,
            9.0,
            3.0,
            6.0,
            ["a", "ra", "b", "rb1", "rb2", "s1"],
        ),
        "c": ("c", "s2", "c", "s2", True, False, 1, 1, 0, 7.0, 4.0, 3.0, ["c"]),
        "d": ("d", "s2", "d", "s2", True, False, 1, 1, 0, 6.0, 4.0, 2.0, ["d"]),
        "s2": ("s2", "s3", "c", "s3", False, False, 2, 1, 0, 4.0, 3.0, 1.0, ["s2"]),
        "s3": ("s3", "root", "a", None, False, True, 2, 1, 0, 3.0, 0.0, 3.0, ["s3"]),
    }
    assert "b" not in set(smoothed.branches["branch_id"])
    assert smoothed.members.height == nominal.members.height
    assert set(smoothed.members["dna_sequence"]) == set(nominal.members["dna_sequence"])


def test_absorbed_hills_never_form_branches_and_keep_their_raw_fitness():
    smoothed = decomposed(balanced_tree(), absorbed={"b", "d"})
    assert fields(smoothed) == {
        "a": (
            "a",
            "s3",
            "a",
            "s3",
            True,
            False,
            1,
            6,
            6,
            9.0,
            3.0,
            6.0,
            ["a", "ra", "b", "rb1", "rb2", "s1"],
        ),
        "c": (
            "c",
            "s3",
            "c",
            "s3",
            True,
            False,
            1,
            3,
            2,
            7.0,
            3.0,
            4.0,
            ["c", "d", "s2"],
        ),
        "s3": ("s3", "root", "a", None, False, True, 2, 1, 0, 3.0, 0.0, 3.0, ["s3"]),
    }
    fitness = dict(smoothed.members.select("dna_sequence", "raw_fitness").iter_rows())
    assert fitness["b"] == 8.0 and fitness["rb2"] == 6.5 and fitness["d"] == 6.0


def test_sequences_absorbed_into_the_root_group_sit_on_the_root_edge_of_the_maximum():
    tree = make_tree(
        [("p1", 3.0, "leaf"), ("p2", 2.0, "leaf"), ("root", 0.0, "root")],
        [("p1", "root"), ("p2", "root")],
        [("p2", "root", "r2", 1.0)],
    )
    view = absorb_terminal_branches(tree, {"p2"})
    assert set(
        view.assignments.filter(pl.col("location") == "absorbed_root")["dna_sequence"]
    ) == {"p2", "r2"}
    result = decompose_view(view, tree.adjacency)
    assert result.root_group_members == 2
    assert fields(result) == {
        "p1": (
            "p1",
            "root",
            "p1",
            None,
            True,
            True,
            1,
            3,
            1,
            3.0,
            0.0,
            3.0,
            ["p1", "p2", "r2"],
        )
    }
    assert "root" not in set(result.members["dna_sequence"])


def test_arms_that_meet_at_the_root_are_terminal_root_edges_of_order_one():
    """A root that is itself the merge: no saddle edge, so no internal branch."""
    tree = make_tree(
        [("p1", 3.0, "leaf"), ("p2", 2.0, "leaf"), ("root", 0.0, "root")],
        [("p1", "root"), ("p2", "root")],
    )
    result = decomposed(tree)
    assert fields(result) == {
        "p1": ("p1", "root", "p1", None, True, True, 1, 1, 0, 3.0, 0.0, 3.0, ["p1"]),
        "p2": ("p2", "root", "p2", None, True, True, 1, 1, 0, 2.0, 0.0, 2.0, ["p2"]),
    }


def test_every_sequence_but_the_root_is_on_exactly_one_branch():
    for absorbed in ((), ("b",), ("d",), ("b", "d")):
        tree = balanced_tree()
        view = absorb_terminal_branches(tree, set(absorbed))
        result = decompose_view(view, tree.adjacency)
        everyone = set(view.assignments["dna_sequence"]) - {"root"}
        assert result.members.height == len(everyone) == 10
        assert set(result.members["dna_sequence"]) == everyone
        assert sum(result.branches["branch_vertex_count"]) == result.members.height


# -- the elder peak of an edge ---------------------------------------------------


def three_way_tie():
    return real_style_tree(
        [
            ("a", 4.0, "leaf"),
            ("b", 4.0, "leaf"),
            ("c", 4.0, "leaf"),
            ("s1", 2.0, "saddle"),
            ("s2", 1.0, "saddle"),
            ("root", 0.0, "root"),
        ],
        [("a", "s1"), ("b", "s1"), ("s1", "s2"), ("c", "s2"), ("s2", "root")],
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_tied_peaks_agree_with_the_smoothing_steps_elder_rule(reverse):
    """On a real-style tree (vertex_index is the identity rank) ties go the same way."""
    nodes = [
        ("alpha", 5.0, "leaf"),
        ("zeta", 5.0, "leaf"),
        ("s", 2.0, "saddle"),
        ("root", 0.0, "root"),
    ]
    tree = make_tree(
        sorted(nodes),
        [("alpha", "s"), ("zeta", "s"), ("s", "root")],
        reverse=reverse,
    )
    protected = enforce_elder_rule(tree, {"alpha", "zeta"})
    rows = by_id(decomposed(tree))
    assert protected == {"alpha"}
    assert rows["s"]["peak_sequence"] == "alpha"
    assert rows["alpha"]["peak_sequence"] == "alpha"
    assert rows["zeta"]["peak_sequence"] == "zeta"
    assert rows["zeta"]["parent_branch_id"] == "s"


def test_a_three_way_tie_goes_to_the_smallest_vertex_index_at_every_saddle():
    tree = three_way_tie()
    assert enforce_elder_rule(tree, {"a", "b", "c"}) == {"a"}
    rows = by_id(decomposed(tree))
    assert {k: r["peak_sequence"] for k, r in rows.items()} == {
        "a": "a",
        "b": "b",
        "c": "c",
        "s1": "a",
        "s2": "a",
    }


def test_a_tie_that_the_pairs_break_differently_is_refused_not_decomposed():
    """vertex_index says zeta, Topographer's identity order says alpha: fail closed."""
    tree = make_tree(
        [
            ("zeta", 5.0, "leaf"),
            ("alpha", 5.0, "leaf"),
            ("s", 2.0, "saddle"),
            ("root", 0.0, "root"),
        ],
        [("alpha", "s"), ("zeta", "s"), ("s", "root")],
    )
    assert enforce_elder_rule(tree, {"alpha", "zeta"}) == {"zeta"}
    with pytest.raises(ValueError, match="pairs it with"):
        decomposed(tree)


# -- identities, never indices ---------------------------------------------------


def test_tables_hold_sequence_identities_and_no_indices():
    result = decomposed(balanced_tree())
    identities = {"a", "b", "c", "d", "s1", "s2", "s3", "root", "ra", "rb1", "rb2"}
    for frame in (result.branches, result.members):
        assert not [name for name in frame.columns if "index" in name]
    for column in (
        "branch_id",
        "upper_sequence",
        "lower_sequence",
        "peak_sequence",
    ):
        assert set(result.branches[column]) <= identities
    assert set(result.branches["parent_branch_id"].drop_nulls()) <= identities
    assert set(result.members["branch_id"]) <= identities
    assert set(result.members["dna_sequence"]) <= identities
    listed = {s for row in result.branches["member_sequences"] for s in row}
    assert listed <= identities
    # The only integer columns are counts and the order, never an identifier.
    integers = {
        name for name, dtype in result.branches.schema.items() if dtype == pl.Int64
    }
    assert integers == {"horton_strahler_order", "branch_vertex_count", "edge_count"}
    # branch_id is unique, being the sequence of the upper endpoint.
    assert result.branches["branch_id"].n_unique() == result.branches.height
    assert result.branches["branch_id"].to_list() == (
        result.branches["upper_sequence"].to_list()
    )


def test_other_rustworkx_indices_give_identical_tables():
    forward = decomposed(balanced_tree())
    backward = decomposed(balanced_tree(reverse=True))
    assert backward.branches.equals(forward.branches)
    assert backward.members.equals(forward.members)
    for absorbed in ({"b"}, {"d"}, {"b", "d"}):
        one = decomposed(balanced_tree(), absorbed)
        other = decomposed(balanced_tree(reverse=True), absorbed)
        assert other.branches.equals(one.branches)
        assert other.members.equals(one.members)


# -- the checks fail closed --------------------------------------------------------


def balanced_view():
    tree = balanced_tree()
    return tree, absorb_terminal_branches(tree, set())


def node_index(graph):
    return {graph[i]["dna_sequence"]: i for i in graph.node_indices()}


def test_a_disagreement_with_the_smoothed_trees_pairs_is_refused():
    tree, view = balanced_view()
    wrong_death = view.pairs.with_columns(
        pl.when(pl.col("birth_label") == "c")
        .then(pl.lit("s2"))
        .otherwise(pl.col("death_label"))
        .alias("death_label")
    )
    with pytest.raises(ValueError, match="pairs it with"):
        decompose_view(dataclasses.replace(view, pairs=wrong_death), tree.adjacency)
    missing_peak = view.pairs.filter(pl.col("birth_label") != "d")
    with pytest.raises(ValueError, match="differ from the peaks"):
        decompose_view(dataclasses.replace(view, pairs=missing_peak), tree.adjacency)


def test_a_persistence_that_differs_is_not_what_the_pairs_check_reads():
    """Only peaks and death endpoints are compared, never a span or a persistence."""
    tree, view = balanced_view()
    shifted = view.pairs.with_columns(pl.col("persistence") + 1.0)
    result = decompose_view(dataclasses.replace(view, pairs=shifted), tree.adjacency)
    assert result.branches.height == 7


def test_a_sequence_assigned_twice_or_off_the_tree_is_refused():
    tree, view = balanced_view()
    twice = pl.concat(
        [view.assignments, view.assignments.filter(pl.col("dna_sequence") == "ra")]
    )
    with pytest.raises(ValueError, match="partition"):
        decompose_view(dataclasses.replace(view, assignments=twice), tree.adjacency)
    stray = view.assignments.with_columns(
        pl.when(pl.col("dna_sequence") == "ra")
        .then(pl.lit("s3"))
        .otherwise(pl.col("lower_sequence"))
        .alias("lower_sequence")
    )
    with pytest.raises(ValueError, match="not an edge"):
        decompose_view(dataclasses.replace(view, assignments=stray), tree.adjacency)


def test_an_edge_that_carries_no_sequence_is_refused():
    tree, view = balanced_view()
    without_d = view.assignments.filter(pl.col("dna_sequence") != "d")
    with pytest.raises(ValueError, match=r"carry no sequence: d\|s2"):
        decompose_view(dataclasses.replace(view, assignments=without_d), tree.adjacency)


def test_nodes_without_or_with_repeated_identities_are_refused():
    tree, view = balanced_view()
    graph = view.tree.copy()
    graph[0] = {k: v for k, v in graph[0].items() if k != "dna_sequence"}
    with pytest.raises(ValueError, match="must carry a dna_sequence"):
        decompose_view(dataclasses.replace(view, tree=graph), tree.adjacency)
    graph = view.tree.copy()
    graph[0] = {**graph[0], "dna_sequence": graph[1]["dna_sequence"]}
    with pytest.raises(ValueError, match="distinct dna_sequence"):
        decompose_view(dataclasses.replace(view, tree=graph), tree.adjacency)


def test_nodes_without_or_with_repeated_vertex_indices_are_refused():
    tree, view = balanced_view()
    graph = view.tree.copy()
    graph[0] = {k: v for k, v in graph[0].items() if k != "vertex_index"}
    with pytest.raises(ValueError, match="lack a vertex_index"):
        decompose_view(dataclasses.replace(view, tree=graph), tree.adjacency)
    graph = view.tree.copy()
    graph[0] = {**graph[0], "vertex_index": graph[1]["vertex_index"]}
    with pytest.raises(ValueError, match="distinct vertex_index"):
        decompose_view(dataclasses.replace(view, tree=graph), tree.adjacency)


def test_a_tree_with_two_roots_or_a_forked_vertex_is_refused():
    tree, view = balanced_view()
    ids = node_index(view.tree)
    headless = view.tree.copy()
    headless.remove_edge(ids["s3"], ids["root"])
    with pytest.raises(ValueError, match="needs one root, found 2"):
        decompose_view(dataclasses.replace(view, tree=headless), tree.adjacency)
    forked = view.tree.copy()
    forked.add_edge(ids["a"], ids["s2"], {})
    with pytest.raises(ValueError, match="one lower neighbour: a"):
        decompose_view(dataclasses.replace(view, tree=forked), tree.adjacency)


def test_a_tree_the_split_tree_rebuild_does_not_reproduce_is_refused():
    """A regular vertex left uncontracted on an edge is not a critical vertex."""
    tree, view = balanced_view()
    graph = view.tree.copy()
    ids = node_index(graph)
    graph.remove_edge(ids["a"], ids["s1"])
    extra = graph.add_node(
        {
            "vertex_index": 99,
            "dna_sequence": "x",
            "aa_sequence": "A",
            "scalar": 8.7,
            "event": "regular",
            "has_stop_codon": False,
        }
    )
    graph.add_edge(ids["a"], extra, {})
    graph.add_edge(extra, ids["s1"], {})
    with pytest.raises(ValueError, match=r"differs from it in edges .*x"):
        decompose_view(dataclasses.replace(view, tree=graph), tree.adjacency)


def test_a_full_graph_adjacency_that_does_not_match_the_tree_is_refused():
    tree, view = balanced_view()
    stranger = pl.concat(
        [
            tree.adjacency,
            pl.DataFrame({"left_sequence": ["a"], "right_sequence": ["z"]}),
        ]
    )
    with pytest.raises(ValueError, match="outside the tree"):
        decompose_view(view, stranger)
    with pytest.raises(ValueError, match="lacks columns"):
        decompose_view(view, tree.adjacency.rename({"right_sequence": "other"}))


def test_the_adjacency_is_read_as_undirected_and_without_repeats():
    tree, view = balanced_view()
    noisy = pl.concat(
        [
            tree.adjacency,
            tree.adjacency.rename(
                {"left_sequence": "right_sequence", "right_sequence": "left_sequence"}
            ).select("left_sequence", "right_sequence"),
            pl.DataFrame({"left_sequence": ["a"], "right_sequence": ["a"]}),
        ]
    )
    assert decompose_view(view, noisy).branches.equals(
        decompose_view(view, tree.adjacency).branches
    )


# -- the adjacency comes from the Full graph, not the Candidate graph --------------


def full_graph():
    """A Full graph whose Candidate graph (threshold 0) lacks the edge x-y.

    x and y are both below the threshold, so ``prepare_clamped_landscape`` drops x-y,
    though each is joined to the functional h and so stays in the Candidate graph.
    z and w hang below x with every end below the threshold, so neither has a
    retained edge and neither is a Candidate vertex.
    """
    vertices = pl.DataFrame(
        {
            "dna_sequence": ["h", "x", "y", "z", "w"],
            "fitness": [0.5, -0.5, -0.4, -0.6, -0.7],
            "aa_sequence": ["A"] * 5,
            "has_stop_codon": [False] * 5,
        }
    )
    edges = pl.DataFrame(
        {"source": ["h", "h", "x", "x", "z"], "target": ["x", "y", "y", "z", "w"]}
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "aa_sequence", "has_stop_codon"),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def test_the_adjacency_is_the_full_graphs_among_the_candidate_graphs_vertices(
    tmp_path, monkeypatch
):
    """Low-low edges, which the Candidate graph drops, are kept; strangers are not."""
    pl.DataFrame({"dna_from": ["hhh"]}).write_parquet(tmp_path / "wt.parquet")
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "diff.parquet")
    seen = {}

    class FakeWorkflow:
        def __init__(self, *, landscape, fitness_wt, fitness_diff):
            seen["landscape"] = landscape

        def build_network(self):
            return full_graph()

    def analyze(prepared, *, name, wild_type_sequence):
        seen["candidate"] = prepared
        seen["wild_type"] = wild_type_sequence
        return SimpleNamespace(trees={"split": "tree"})

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        candidate_selection,
        "resolve_candidate_selection",
        fixed_selection(0.0),
    )
    monkeypatch.setattr(module, "Papkou2023Workflow", FakeWorkflow)
    monkeypatch.setattr(module, "analyze_prepared_landscape", analyze)
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    tree, adjacency = module.build_candidate_inputs(settings)
    candidate = seen["candidate"].network
    assert tree == "tree" and seen["landscape"] == "full"
    assert seen["wild_type"] == "hhh"
    assert set(candidate.vertex_ids) == {"h", "x", "y"}
    candidate_edges = {
        frozenset(
            (candidate.graph[a]["dna_sequence"], candidate.graph[b]["dna_sequence"])
        )
        for a, b in candidate.graph.edge_list()
    }
    assert frozenset("xy") not in candidate_edges  # the Candidate graph lacks it
    assert {frozenset(pair) for pair in adjacency.iter_rows()} == {
        frozenset("hx"),
        frozenset("hy"),
        frozenset("xy"),  # kept: the Full graph has it
    }  # and neither x-z nor z-w, whose z and w are not Candidate vertices
    assert adjacency.columns == list(module.ADJACENCY_COLUMNS)


# -- settings and the tree each setting names -------------------------------------


def test_settings_are_validated():
    with pytest.raises(ValueError, match="retired in ticket 11.*legacy-only"):
        branch_decomposition_settings({"tree": "core"})
    with pytest.raises(ValueError, match="one of"):
        branch_decomposition_settings({"tree": "median"})
    with pytest.raises(ValueError, match="needs `tree`"):
        branch_decomposition_settings({})
    with pytest.raises(ValueError, match="Unknown"):
        branch_decomposition_settings({"tree": "band", "depth": 3})


def bootstrap_hills():
    tree = balanced_tree()
    return tree, hills_table(
        tree, BALANCED_PERSISTENCE, core_absorbed=("d",), two_c=3.5
    )


def test_each_tree_setting_decomposes_its_own_tree():
    tree, hills = bootstrap_hills()
    found = {}
    for name in ("nominal", "band"):
        view = decomposition_view(name, tree, hills)
        found[name] = orders(decompose_view(view, tree.adjacency))
    # Nominal: all four peaks. Band (persistence above 2c = 3.5): b and d absorbed.
    assert found["nominal"] == {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "s1": 2,
        "s2": 2,
        "s3": 3,
    }
    assert found["band"] == {"a": 1, "c": 1, "s3": 2}
    with pytest.raises(ValueError, match="retired in ticket 11.*legacy-only"):
        decomposition_view("core", tree, hills)
    with pytest.raises(ValueError, match="one of"):
        decomposition_view("median", tree, hills)


# -- produce or load -----------------------------------------------------------------


@pytest.fixture
def stage(tmp_path, monkeypatch):
    tree, hills = bootstrap_hills()
    bootstrap = SimpleNamespace(
        hills=hills,
        provenance={
            "fingerprint": {"config": {"n_replicates": 10}},
            "artifacts": {"hills.parquet": "0" * 64},
            "created_at": "2026-10-21T00:00:00+00:00",
        },
    )
    builds: list[int] = []

    def build_inputs(settings):
        builds.append(1)
        return tree, tree.adjacency

    monkeypatch.setattr(
        module, "load_cached_bootstrap_smoothed_tree", lambda settings: bootstrap
    )
    monkeypatch.setattr(module, "build_candidate_inputs", build_inputs)

    def settings(tree_name="band"):
        return {
            "analysis": {
                "branch_decomposition": {"tree": tree_name},
                "clamped": {
                    "candidate_functional_threshold": {
                        "rule": "median_reflected",
                        "decimals": 6,
                    }
                },
            }
        }

    return SimpleNamespace(
        builds=builds,
        bootstrap=bootstrap,
        checkpoint_dir=tmp_path / "checkpoints" / "branch_decomposition",
        settings=settings,
    )


def test_stage_fingerprints_moved_simplification_sources(stage):
    settings = stage.settings()
    section = branch_decomposition_settings(
        settings["analysis"]["branch_decomposition"]
    )
    declared = {
        Path(path).resolve()
        for path in module._stage(section, stage.bootstrap, settings).code
    }
    simplification = ROOT / "src/dubanevics2026topological/simplification"
    assert {
        (simplification / name).resolve()
        for name in (
            "__init__.py",
            "bootstrap_smoothed_tree.py",
            "bootstrap_chunks.py",
            "checkpoint_identity.py",
            "identities.py",
            "selected_leaf_simplification.py",
        )
    } <= declared


def run(stage, tree_name="band", **options):
    return compute_branch_decomposition(
        stage.settings(tree_name), checkpoint_dir=stage.checkpoint_dir, **options
    )


def load(stage, tree_name="band"):
    return load_cached_branch_decomposition(
        stage.settings(tree_name), checkpoint_dir=stage.checkpoint_dir
    )


def test_a_matching_checkpoint_is_reused_and_the_tree_is_built_only_to_produce(stage):
    first = run(stage)
    assert not first.reused and len(stage.builds) == 1
    again = run(stage)
    assert again.reused and len(stage.builds) == 1
    assert again.provenance["fingerprint"] == first.provenance["fingerprint"]
    assert again.branches.equals(first.branches)
    assert again.members.equals(first.members)
    assert again.summary == first.summary
    loaded = load(stage)
    assert loaded.reused and len(stage.builds) == 1
    assert loaded.branches.equals(first.branches)
    assert loaded.members.equals(first.members)


def test_the_published_checkpoint_holds_the_tables_with_list_members_intact(stage):
    result = run(stage)
    assert set(result.provenance["artefacts"]) == set(module.ARTEFACTS)
    assert result.provenance["stage"] == module.STAGE_NAME
    tree, hills = bootstrap_hills()
    expected = decompose_view(decomposition_view("band", tree, hills), tree.adjacency)
    assert result.branches.equals(expected.branches)
    assert result.members.equals(expected.members)
    assert result.branches["member_sequences"].dtype == pl.List(pl.String)
    summary = json.loads((_leaf(stage) / "summary.json").read_text())
    assert summary == result.summary
    assert summary["settings"] == {"tree": "band"}
    # band: a, c terminal (order 1), s3 the root edge (order 2).
    assert summary["branches_per_order"] == {"1": 2, "2": 1}
    assert summary["branch_count"] == 3 and summary["maximum_order"] == 2
    assert summary["terminal_branch_count"] == 2
    assert summary["internal_branch_count"] == 1
    assert summary["sequence_count"] == 11 and summary["root_sequence"] == "root"
    assert summary["member_count"] == 10 and summary["root_edge_count"] == 1
    assert summary["root_group_member_count"] == 0


def test_the_csv_tables_equal_the_parquet_tables(stage):
    result = run(stage)
    branches = pl.read_csv(
        _leaf(stage) / "branches.csv",
        schema_overrides={**BRANCH_SCHEMA, "member_sequences": pl.String},
    ).with_columns(pl.col("member_sequences").str.split(";"))
    assert branches.equals(result.branches)
    assert branches.equals(pl.read_parquet(_leaf(stage) / "branches.parquet"))
    members = pl.read_csv(
        _leaf(stage) / "members.csv", schema_overrides=MEMBER_SCHEMA
    )
    assert members.equals(result.members)
    assert members.equals(pl.read_parquet(_leaf(stage) / "members.parquet"))
    exploded = branches.select(
        "branch_id", pl.col("member_sequences").alias("dna_sequence")
    ).explode("dna_sequence", empty_as_null=False)
    assert sorted(exploded.iter_rows()) == sorted(
        members.select("branch_id", "dna_sequence").iter_rows()
    )


def test_force_reproduces_with_identical_bytes(stage):
    first = run(stage)
    forced = run(stage, force=True)
    assert not forced.reused and len(stage.builds) == 2
    assert forced.provenance["artefacts"] == first.provenance["artefacts"]


def test_a_changed_setting_or_upstream_checkpoint_produces_again(stage):
    band = run(stage)
    nominal = run(stage, "nominal")
    assert not nominal.reused and len(stage.builds) == 2
    assert nominal.provenance["fingerprint"] != band.provenance["fingerprint"]
    assert nominal.summary["settings"] == {"tree": "nominal"}
    assert nominal.summary["maximum_order"] == 3
    assert band.summary["maximum_order"] == 2
    # Each tree is its own key, so the band result stays readable.
    assert load(stage, "band").reused
    stage.bootstrap.provenance["artifacts"] = {"hills.parquet": "1" * 64}
    changed = run(stage, "nominal")
    assert not changed.reused and len(stage.builds) == 3


def test_timestamps_of_the_upstream_checkpoint_do_not_invalidate_it(stage):
    run(stage)
    stage.bootstrap.provenance["created_at"] = "2027-01-01T00:00:00+00:00"
    stage.bootstrap.provenance["git"] = {"commit": "abc", "dirty": True}
    assert run(stage).reused


def test_the_stage_declares_topographers_sources(stage):
    declared = module._stage(
        {"tree": "band"}, stage.bootstrap, stage.settings()
    ).settings
    sources = declared["topographer_sources"]
    assert any(name.endswith("_tree_metrics.py") for name in sources)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in sources.values())


def test_an_edited_topographer_source_produces_again(stage, monkeypatch):
    """uv.lock does not change under an editable install; the source hashes do."""
    first = run(stage)
    real, seen = module.package_source_hashes, []

    def edited(*packages):
        seen.append(packages)
        return {**real(*packages), "topographer/topology/_tree_metrics.py": "f" * 64}

    monkeypatch.setattr(module, "package_source_hashes", edited)
    again = run(stage)
    assert not again.reused and len(stage.builds) == 2
    assert again.provenance["fingerprint"] != first.provenance["fingerprint"]
    assert seen[0] == (topographer,)
    assert load(stage).reused  # the new checkpoint is the current one


def test_loading_without_a_checkpoint_refuses_instead_of_computing(stage):
    with pytest.raises(FileNotFoundError, match="just branch-decomposition"):
        load(stage)
    assert stage.builds == []
    assert not stage.checkpoint_dir.parent.exists()


def _leaf(stage) -> Path:
    root = stage.checkpoint_dir
    found = sorted(root.glob("*/provenance.json"))
    if len(found) == 1:
        return found[0].parent
    return root


def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Names, sizes and modification times, apart from the lock file."""
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
        if not path.name.endswith(".lock")
    }


def test_the_loader_rejects_an_unfinished_publish_and_changes_nothing(stage):
    run(stage)
    parent = stage.checkpoint_dir
    leaf = _leaf(stage)
    pending = leaf.parent / f".{leaf.name}.pending"
    pending.write_text("", encoding="utf-8")
    before = snapshot(parent)
    with pytest.raises(
        RuntimeError, match="mid-publication.*just branch-decomposition"
    ):
        load(stage)
    assert snapshot(parent) == before and pending.exists()
    pending.unlink()
    backup = leaf.parent / f".{leaf.name}.previous"
    backup.mkdir()
    (backup / "branches.parquet").write_bytes(b"older")
    before = snapshot(parent)
    with pytest.raises(RuntimeError, match="mid-publication"):
        load(stage)
    assert snapshot(parent) == before and backup.is_dir()
    assert len(stage.builds) == 1


def test_the_loader_verifies_artefact_hashes_and_repairs_nothing(stage):
    run(stage)
    parent = stage.checkpoint_dir.parent
    (_leaf(stage) / "branches.parquet").write_bytes(b"tampered")
    before = snapshot(parent)
    with pytest.raises(RuntimeError, match="does not match its recorded hashes"):
        load(stage)
    assert snapshot(parent) == before
    assert not list(parent.glob(".*unusable*"))
    (_leaf(stage) / "provenance.json").unlink()
    with pytest.raises(RuntimeError, match="unreadable"):
        load(stage)


def test_tables_are_read_under_the_checkpoint_lock(stage, monkeypatch):
    seen = []
    real = module._read_result

    def spy(checkpoint):
        try:
            with checkpoint_lock(_leaf(stage), "spy"):
                free = True
        except RuntimeError:
            free = False
        seen.append(free)
        return real(checkpoint)

    monkeypatch.setattr(module, "_read_result", spy)
    run(stage)  # produced
    run(stage)  # reused
    load(stage)
    assert seen == [False, False, False]


def test_a_hill_table_from_another_tree_is_refused_before_publishing(stage):
    stage.bootstrap.hills = stage.bootstrap.hills.filter(pl.col("peak_sequence") != "d")
    with pytest.raises(ValueError, match="differ"):
        run(stage)
    assert not stage.checkpoint_dir.exists()


# -- the build command and the just recipe ------------------------------------------------


def test_the_build_command_runs_the_stage_and_names_its_tables(
    stage, monkeypatch, capsys
):
    monkeypatch.setattr(
        module,
        "compute_branch_decomposition",
        lambda settings, force=False: run(stage, force=force),
    )
    assert build.main(["branch-decomposition", "--force"]) == 0
    printed = capsys.readouterr().out
    assert "3 branches" in printed and "2 terminal, 1 internal" in printed
    assert "branches.parquet" in printed and "produced" in printed
    assert len(stage.builds) == 1


def test_the_just_recipe_calls_the_build_command_and_not_a_second_cli():
    text = (ROOT / "justfile").read_text(encoding="utf-8")
    recipe = text.split("\nbranch-decomposition:", 1)[1].split("\n\n", 1)[0]
    # The Candidate selection reads the functional classification first.
    assert recipe.startswith(" selection\n")
    assert "-m dubanevics2026topological.build branch-decomposition" in recipe
    assert "dubanevics2026topological.analysis" not in recipe
    assert "def main" not in Path(module.__file__).read_text(encoding="utf-8")


def test_the_loader_needs_no_raw_data_for_the_rounded_threshold(stage, monkeypatch):
    run(stage)

    def no_raw_data(*_args):
        raise FileNotFoundError("data/raw is absent")

    monkeypatch.setattr(
        "dubanevics2026topological.landscape.selection._raw_fitness",
        no_raw_data,
    )
    assert load(stage).reused


def test_a_changed_threshold_declaration_produces_again(stage):
    run(stage)
    assert run(stage).reused
    changed = stage.settings()
    changed["analysis"]["clamped"]["candidate_functional_threshold"]["decimals"] = 3
    assert not compute_branch_decomposition(
        changed, checkpoint_dir=stage.checkpoint_dir
    ).reused
