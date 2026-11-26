"""Terminal leaf contracts on small, independently inspectable trees."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import rustworkx as rx
from _branch_trees import balanced_tree, make_tree

from dubanevics2026topological.analysis import describe


def test_single_sequence_is_root_only_leaf() -> None:
    graph = rx.PyDiGraph()
    graph.add_node({"dna_sequence": "AAA", "aa_sequence": "*", "scalar": 0.5})
    product = describe(
        products={"leaves"},
        leaf_tree=SimpleNamespace(tree=graph),
        leaf_population="full",
    ).leaves
    leaf = product.table.row(0, named=True)
    assert leaf["member_count"] == 1
    assert leaf["adjacent_saddle_dna_sequence"] is None
    assert leaf["is_synonymous"] is None
    assert leaf["stop_composition_1"] == 1
    assert product.members["member_dna_sequence"].to_list() == ["AAA"]
    assert product.summary["root_member_count"] == 0
    assert product.summary["ineligible_endpoint_count"] == 1
    assert product.summary["top2_shortfall"] == 1


def test_empty_tree_has_explicit_shortfall() -> None:
    product = describe(
        products={"leaves"},
        leaf_tree=SimpleNamespace(tree=rx.PyDiGraph()),
        leaf_population="full",
    ).leaves
    assert product.table.is_empty()
    assert product.members.is_empty()
    assert product.summary["top2_shortfall"] == 2
    assert product.summary["tree_member_count"] == 0


def test_leaf_members_and_internal_root_reconcile() -> None:
    product = describe(
        products={"leaves"}, leaf_tree=balanced_tree(), leaf_population="full"
    ).leaves
    assert product.table["peak_dna_sequence"].to_list() == ["b", "a", "c", "d"]
    assert product.table["member_count"].to_list() == [3, 2, 1, 1]
    assert product.table["is_top2_member_count"].to_list() == [True, True, False, False]
    assert product.members["member_dna_sequence"].n_unique() == 7
    assert product.summary["internal_member_count"] == 3
    assert product.summary["root_member_count"] == 1
    assert product.summary["tree_member_count"] == 11
    assert (
        product.table.filter(pl.col("peak_dna_sequence") == "a")[
            "adjacent_saddle_dna_sequence"
        ].item()
        == "s1"
    )


def test_equal_leaf_sizes_rank_by_peak_dna() -> None:
    tree = make_tree(
        [
            ("a", 3.0, "leaf"),
            ("b", 4.0, "leaf"),
            ("c", 5.0, "leaf"),
            ("s", 1.0, "saddle"),
            ("r", 0.0, "root"),
        ],
        [("a", "s"), ("b", "s"), ("c", "s"), ("s", "r")],
    )
    leaves = describe(
        products={"leaves"}, leaf_tree=tree, leaf_population="full"
    ).leaves
    assert leaves.table["peak_dna_sequence"].to_list() == ["a", "b", "c"]
    assert leaves.table["is_top2_member_count"].to_list() == [True, True, False]


def test_stop_counts_and_synonymous_eligibility() -> None:
    tree = make_tree(
        [
            ("a", 4.0, "leaf"),
            ("b", 3.0, "leaf"),
            ("s", 1.0, "saddle"),
            ("r", 0.0, "root"),
        ],
        [("a", "s"), ("b", "s"), ("s", "r")],
        [("a", "s", "x", 3.5), ("a", "s", "y", 3.2)],
    )
    for index in tree.tree.node_indices():
        if tree.tree[index]["dna_sequence"] == "b":
            tree.tree[index]["aa_sequence"] = "B"
    tree.branch_member_table = tree.branch_member_table.with_columns(
        pl.when(pl.col("dna_sequence") == "x")
        .then(pl.lit("**"))
        .when(pl.col("dna_sequence") == "y")
        .then(pl.lit("***"))
        .otherwise(pl.col("aa_sequence"))
        .alias("aa_sequence")
    )
    product = describe(
        products={"leaves"}, leaf_tree=tree, leaf_population="full"
    ).leaves
    a = product.table.filter(pl.col("peak_dna_sequence") == "a").row(0, named=True)
    b = product.table.filter(pl.col("peak_dna_sequence") == "b").row(0, named=True)
    assert (
        a["stop_composition_1"],
        a["stop_composition_2"],
        a["stop_composition_3"],
    ) == (0, 1, 1)
    assert a["is_synonymous"] is True
    assert b["is_synonymous"] is False
    assert product.summary["eligible_leaf_denominator"] == 2
    root_tree = make_tree([("a", 2.0, "leaf"), ("r", 0.0, "root")], [("a", "r")])
    root_leaf = describe(
        products={"leaves"}, leaf_tree=root_tree, leaf_population="full"
    ).leaves
    assert root_leaf.table["is_synonymous"].item() is None
    assert root_leaf.summary["eligible_leaf_denominator"] == 0


def test_save_and_load_verified_leaf_product(tmp_path: Path) -> None:
    from dubanevics2026topological.landscape.products import _publish
    from dubanevics2026topological.results import RunContext, load

    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    params = {"graph": "full", "tree": "nominal"}
    parent_path, _ = _publish(
        "tree",
        params,
        params,
        context=context,
        input_data_sha256s={},
        parents=(),
        code_paths=(),
        write_artifacts=lambda path: (path / "tree.json").write_text("{}"),
    )
    parent = load("tree", parent_path.name, context=context)
    graph = rx.PyDiGraph()
    graph.add_node({"dna_sequence": "AAA", "aa_sequence": "A", "scalar": 1.0})
    leaves = describe(
        products={"leaves"},
        leaf_tree=SimpleNamespace(tree=graph),
        leaf_population="full",
    ).leaves
    saved = leaves.save(parent=parent, context=context)
    restored = type(leaves).load(saved.name, context=context)
    assert restored.summary == leaves.summary
    assert restored.table.equals(leaves.table)
    assert restored.members.equals(leaves.members)
    loaded = load("leaves", saved.name, context=context)
    for ref in loaded.manifest.parents:
        resolved = load(ref["product_type"], ref["label"], context=context)
        assert resolved.manifest.scientific_id_sha256 == ref["scientific_id_sha256"]


