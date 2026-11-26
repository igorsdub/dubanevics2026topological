from types import SimpleNamespace

import polars as pl
import pytest
import rustworkx as rx

from dubanevics2026topological.simplification.selected_leaf_simplification import (
    absorb_singleton_leaves,
    absorb_terminal_branches,
    enforce_elder_rule,
    rebuild_split_tree,
)


def tree_input(extra_peak=False):
    graph = rx.PyDiGraph()
    rows = [
        ("elder", 5.0, "maximum"),
        ("survivor", 4.0, "maximum"),
        ("other", 3.0, "maximum"),
        ("saddle", 2.0, "split_saddle"),
        ("lower", 1.0, "split_saddle"),
        ("root", 0.0, "minimum"),
    ]
    if extra_peak:
        rows.append(("extra", 3.5, "maximum"))
    for i, (dna, scalar, event) in enumerate(rows):
        graph.add_node(
            {
                "vertex_index": i,
                "dna_sequence": dna,
                "aa_sequence": "A",
                "scalar": scalar,
                "event": event,
                "has_stop_codon": dna == "survivor",
            }
        )
    graph.add_edges_from_no_data([(0, 3), (1, 3), (3, 4), (2, 4), (4, 5)])
    if extra_peak:
        graph.add_edge(6, 3, {})
    return SimpleNamespace(
        tree=graph,
        branch_member_table=pl.DataFrame(),
        wild_type_marker_table=pl.DataFrame(),
    )


def test_absorption_preserves_population_and_recomputes_surviving_pair():
    source = tree_input()
    view = absorb_singleton_leaves(source, {"elder"})
    assert source.tree.num_nodes() == 6
    assert set(view.plot_data.node_table["dna_sequence"]) == {
        "survivor",
        "other",
        "lower",
        "root",
    }
    assert view.removed_saddles == ("saddle",)
    assert view.assignments.height == 6
    assert dict(view.assignments.select("dna_sequence", "raw_fitness").iter_rows()) == {
        source.tree[i]["dna_sequence"]: source.tree[i]["scalar"]
        for i in source.tree.node_indices()
    }
    assert view.pairs.filter(pl.col("pair_kind") == "extended")[
        "birth_label"
    ].to_list() == ["survivor"]
    assert view.pairs.filter(pl.col("birth_label") == "survivor")[
        "birth_has_stop_codon"
    ].to_list() == [True]
    assert view.pairs["death_label"].to_list() == ["root", "lower"]
    absorbed = view.assignments.filter(pl.col("absorbed_peak")).row(0, named=True)
    assert (absorbed["upper_sequence"], absorbed["lower_sequence"]) == (
        "survivor",
        "lower",
    )


def test_layout_puts_the_highest_peak_first():
    # The default horizontal_layout is "subtree_extremum": the child whose subtree
    # reaches the highest peak goes first, so the elder peak (5.0) is left of the
    # survivor (4.0).
    view = absorb_singleton_leaves(tree_input(), {"other"})
    x = dict(view.plot_data.node_table.select("dna_sequence", "x").iter_rows())
    assert x["elder"] < x["survivor"]
    assert (min(x.values()), max(x.values())) == (0.0, 1.0)


def test_layout_follows_the_horizontal_layout_argument():
    # The caller supplies the order: Topographer's default ("leaf_order") puts the
    # lower peak first.
    view = absorb_singleton_leaves(
        tree_input(), {"other"}, horizontal_layout="leaf_order"
    )
    x = dict(view.plot_data.node_table.select("dna_sequence", "x").iter_rows())
    assert x["survivor"] < x["elder"]


def test_the_configured_horizontal_layout_is_subtree_extremum():
    # The helper is figure-side: analysis/__init__.py is fingerprinted.
    from dubanevics2026topological import analysis

    assert not hasattr(analysis, "tree_horizontal_layout")
    from dubanevics2026topological.figures._layout_config import tree_horizontal_layout

    assert tree_horizontal_layout() == "subtree_extremum"


def test_the_shared_rebuild_gives_the_split_tree_absorption_computes_pairs_on():
    source = tree_input()
    merge_tree = rebuild_split_tree(source.tree)
    labels = {
        int(i): str(merge_tree.graph[i]["vertex_id"])
        for i in merge_tree.graph.node_indices()
    }
    assert set(labels.values()) == {
        source.tree[i]["dna_sequence"] for i in source.tree.node_indices()
    }
    # Edges run root to leaves in a MergeTree; they are the tree's, reversed.
    edges = {(labels[int(u)], labels[int(v)]) for u, v in merge_tree.graph.edge_list()}
    assert edges == {
        (source.tree[v]["dna_sequence"], source.tree[u]["dna_sequence"])
        for u, v in source.tree.edge_list()
    }
    # The pairs of the rebuilt tree are the ones absorption reports for a nominal tree.
    view = absorb_terminal_branches(source, set())
    rebuilt = {
        (str(row["birth_label"]), str(row["death_label"]))
        for row in merge_tree.persistence().iter_rows(named=True)
    }
    assert rebuilt == set(view.pairs.select("birth_label", "death_label").iter_rows())


def test_the_rebuild_orders_ties_by_identity_whatever_the_insertion_order():
    tied = SimpleNamespace(tree=rx.PyDiGraph())
    for dna, vertex in (("zed", 0), ("ant", 1)):
        tied.tree.add_node(
            {
                "vertex_index": vertex,
                "dna_sequence": dna,
                "aa_sequence": "A",
                "scalar": 5.0,
                "event": "maximum",
                "has_stop_codon": False,
            }
        )
    tied.tree.add_node(
        {
            "vertex_index": 2,
            "dna_sequence": "mid",
            "aa_sequence": "A",
            "scalar": 1.0,
            "event": "split_saddle",
            "has_stop_codon": False,
        }
    )
    tied.tree.add_node(
        {
            "vertex_index": 3,
            "dna_sequence": "root",
            "aa_sequence": "A",
            "scalar": 0.0,
            "event": "minimum",
            "has_stop_codon": False,
        }
    )
    tied.tree.add_edges_from_no_data([(0, 2), (1, 2), (2, 3)])
    pairs = rebuild_split_tree(tied.tree).persistence()
    # Topographer canonicalises by identity: "ant" is the elder of the tie, not the
    # node inserted first or the one with the smaller vertex_index.
    survivor = pairs.filter(pl.col("death_label") == "root")["birth_label"].to_list()
    assert survivor == ["ant"]


def test_saddle_with_two_remaining_arms_is_retained():
    view = absorb_singleton_leaves(tree_input(extra_peak=True), {"elder"})
    assert view.removed_saddles == ()
    assert "saddle" in view.plot_data.node_table["dna_sequence"]
    assert view.pairs.height == 3


def test_entire_selected_fork_is_absorbed_without_inventing_a_peak():
    view = absorb_singleton_leaves(tree_input(), {"elder", "survivor"})
    assert set(view.plot_data.node_table["dna_sequence"]) == {"other", "root"}
    assert set(view.removed_saddles) == {"saddle", "lower"}
    assert view.pairs["birth_label"].to_list() == ["other"]
    assert view.assignments.height == 6


def test_invalid_selection_rejected():
    with pytest.raises(ValueError, match="not a terminal peak"):
        absorb_singleton_leaves(tree_input(), {"saddle"})


def tree_with_branch():
    """Peak ``elder`` carries two regular vertices on its branch to ``saddle``."""
    source = tree_input()
    source.branch_member_table = pl.DataFrame(
        {
            "source_tree_node_index": [0, 0],
            "target_tree_node_index": [3, 3],
            "dna_sequence": ["reg1", "reg2"],
            "aa_sequence": ["A", "A"],
            "scalar": [4.5, 3.2],
        }
    )
    return source


def test_terminal_branch_mechanism_absorbs_any_nominated_peak_including_the_higher_one():
    source = tree_with_branch()
    view = absorb_terminal_branches(source, {"elder"})
    assert view.assignments.height == 8
    assert view.assignments["dna_sequence"].n_unique() == 8
    assert set(view.plot_data.node_table["dna_sequence"]) == {
        "survivor",
        "other",
        "lower",
        "root",
    }
    rows = {r["dna_sequence"]: r for r in view.assignments.iter_rows(named=True)}
    for dna in ("reg1", "reg2"):
        assert rows[dna]["location"] == "absorbed_branch"
        assert rows[dna]["absorbed_peak"] is False
        assert (rows[dna]["upper_sequence"], rows[dna]["lower_sequence"]) == (
            "survivor",
            "lower",
        )
    assert rows["elder"]["absorbed_peak"] is True
    assert rows["elder"]["location"] == "branch"
    # Raw fitness stays the scalar; the saddle keeps its measured value.
    assert rows["saddle"]["raw_fitness"] == 2.0
    assert rows["reg1"]["raw_fitness"] == 4.5
    # The mechanism is general: with the higher peak nominated, the lower
    # sibling holds the surviving maximum. The elder rule is applied upstream.
    assert view.pairs.filter(pl.col("pair_kind") == "extended")[
        "birth_label"
    ].to_list() == ["survivor"]
    assert view.pairs["birth_scalar"].max() == 4.0


def test_singleton_wrapper_still_rejects_larger_branch():
    with pytest.raises(ValueError, match="not a singleton branch"):
        absorb_singleton_leaves(tree_with_branch(), {"elder"})


def tree_with_root_saddle():
    """Peaks ``p1`` and ``p2`` meet directly at ``root`` (a saddle with in_degree 2)."""
    graph = rx.PyDiGraph()
    rows = [("p1", 3.0, "maximum"), ("p2", 2.0, "maximum"), ("root", 0.0, "minimum")]
    for i, (dna, scalar, event) in enumerate(rows):
        graph.add_node(
            {
                "vertex_index": i,
                "dna_sequence": dna,
                "aa_sequence": "A",
                "scalar": scalar,
                "event": event,
                "has_stop_codon": False,
            }
        )
    graph.add_edges_from_no_data([(0, 2), (1, 2)])
    return SimpleNamespace(
        tree=graph,
        branch_member_table=pl.DataFrame(
            {
                "source_tree_node_index": [1],
                "target_tree_node_index": [2],
                "dna_sequence": ["reg"],
                "aa_sequence": ["A"],
                "scalar": [1.0],
            }
        ),
        wild_type_marker_table=pl.DataFrame(),
    )


def test_peak_whose_saddle_is_the_root_moves_into_root_group():
    view = absorb_terminal_branches(tree_with_root_saddle(), {"p2"})
    assert view.assignments.height == 4
    rows = {r["dna_sequence"]: r for r in view.assignments.iter_rows(named=True)}
    assert rows["p2"]["absorbed_peak"] is True
    assert rows["p2"]["location"] == "absorbed_root"
    assert rows["reg"]["location"] == "absorbed_root"
    assert rows["reg"]["absorbed_branch_member"] is True
    assert rows["p2"]["absorbed_branch_member"] is False
    assert (rows["reg"]["upper_sequence"], rows["reg"]["lower_sequence"]) == (
        "root",
        "root",
    )
    assert set(view.plot_data.node_table["dna_sequence"]) == {"p1", "root"}


def test_absorbing_every_peak_at_the_root_is_refused():
    with pytest.raises(ValueError, match="every surviving peak"):
        absorb_terminal_branches(tree_with_root_saddle(), {"p1", "p2"})


def test_cascade_into_root_keeps_branch_member_flag():
    source = tree_with_branch()
    view = absorb_terminal_branches(source, {"elder", "survivor"})
    rows = {r["dna_sequence"]: r for r in view.assignments.iter_rows(named=True)}
    assert rows["reg1"]["absorbed_branch_member"] is True
    assert rows["saddle"]["absorbed_branch_member"] is False
    assert rows["root"]["absorbed_branch_member"] is False


def test_enforce_elder_rule_protects_highest_arm_at_each_saddle():
    source = tree_input()
    assert enforce_elder_rule(source, {"elder", "survivor"}) == {"elder"}
    assert enforce_elder_rule(source, {"survivor"}) == set()
    # "other" joins a trunk whose upstream subtree peaks at 5.0.
    assert enforce_elder_rule(source, {"other"}) == set()
    assert enforce_elder_rule(source, set()) == set()


def test_enforce_elder_rule_breaks_exact_ties_by_smaller_vertex_index():
    source = tree_input()
    source.tree[1] = {**source.tree[1], "scalar": 5.0}
    assert enforce_elder_rule(source, {"elder", "survivor"}) == {"elder"}
    # The tie follows vertex_index, not dna_sequence: "elder" sorts before
    # "survivor" by name but loses once its vertex_index is larger.
    source.tree[0] = {**source.tree[0], "vertex_index": 9}
    assert enforce_elder_rule(source, {"elder", "survivor"}) == {"survivor"}


def test_enforce_elder_rule_requires_vertex_index():
    source = tree_input()
    payload = dict(source.tree[0])
    payload.pop("vertex_index")
    source.tree[0] = payload
    with pytest.raises(ValueError, match="vertex_index"):
        enforce_elder_rule(source, {"elder"})


def test_enforce_elder_rule_guards_global_maximum_and_validates_input():
    source = tree_input()
    assert enforce_elder_rule(source, {"elder"}) == {"elder"}
    with pytest.raises(ValueError, match="absent"):
        enforce_elder_rule(source, {"missing"})
    with pytest.raises(ValueError, match="not a terminal peak"):
        enforce_elder_rule(source, {"saddle"})


def test_enforce_elder_rule_on_root_saddle():
    source = tree_with_root_saddle()
    assert enforce_elder_rule(source, {"p1", "p2"}) == {"p1"}
    assert enforce_elder_rule(source, {"p2"}) == set()
    assert enforce_elder_rule(source, {"p1"}) == {"p1"}


def _deep_tie_tree(peak_index: int, deep_index: int):
    """Terminal peak and an interior arm whose deeper peak has the same height.

    The names sort the opposite way to the indices on purpose: ``a_deep`` sorts
    first but the tie must follow ``vertex_index``.
    """
    graph = rx.PyDiGraph()
    rows = [
        ("b_peak", 5.0, peak_index),
        ("a_deep", 5.0, deep_index),
        ("x", 1.0, 2),
        ("inner", 3.0, 3),
        ("s", 2.0, 4),
        ("root", 0.0, 5),
    ]
    for dna, scalar, vertex_index in rows:
        graph.add_node(
            {
                "vertex_index": vertex_index,
                "dna_sequence": dna,
                "aa_sequence": "A",
                "scalar": scalar,
                "event": "x",
                "has_stop_codon": False,
            }
        )
    graph.add_edges_from_no_data([(1, 3), (2, 3), (3, 4), (0, 4), (4, 5)])
    return SimpleNamespace(
        tree=graph,
        branch_member_table=pl.DataFrame(),
        wild_type_marker_table=pl.DataFrame(),
    )


def test_enforce_elder_rule_tie_from_deeper_in_an_arm():
    # The interior arm's maximum (a_deep) ties the sibling peak (b_peak); the
    # smaller vertex_index wins, whatever the names.
    sibling_wins = _deep_tie_tree(peak_index=0, deep_index=1)
    assert enforce_elder_rule(sibling_wins, {"b_peak", "x"}) == {"b_peak"}
    # a_deep still beats x at its own saddle, so it is protected there.
    assert enforce_elder_rule(sibling_wins, {"a_deep"}) == {"a_deep"}
    interior_wins = _deep_tie_tree(peak_index=1, deep_index=0)
    assert enforce_elder_rule(interior_wins, {"b_peak", "x"}) == set()
    assert enforce_elder_rule(interior_wins, {"a_deep"}) == {"a_deep"}


def test_enforce_elder_rule_rejects_non_finite_scalar():
    source = tree_input()
    source.tree[1] = {**source.tree[1], "scalar": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        enforce_elder_rule(source, {"elder"})
