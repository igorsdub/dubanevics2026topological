"""Selectable horizontal tree layouts: path to the root, WT side, determinism, format."""

from __future__ import annotations

import json
import random

import polars as pl
import pytest
import rustworkx as rx
from topographer.plotting import make_tree_layout

from dubanevics2026topological.analysis import tree_layouts
from dubanevics2026topological.analysis.tree_layouts import (
    HORIZONTAL_LAYOUTS,
    anchor_identity_for_sequence,
    crop_to_subtree,
    layout_tree,
)

ANCHORED = ("persistence_wt_anchored", "tidy")

# name -> (scalar, upper children). Split tree: scalars rise away from "root".
# "wt_peak" carries the wild type on its branch; the global maximum "g" is on
# the other side, so a subtree-extremum order puts the wild type second.
SMALL = {
    "root": (0.0, ["s1"]),
    "s1": (1.0, ["s2", "s3"]),
    "s2": (2.0, ["g", "h"]),
    "g": (10.0, []),
    "h": (5.0, []),
    "s3": (1.5, ["wt_peak", "w2"]),
    "wt_peak": (4.0, []),
    "w2": (3.0, []),
}


SMALL_EDGES = len(SMALL) - 1


def build_tree(spec, order=None, chains=None):
    """Return a PyDiGraph (edges upper -> lower) with payloads in ``order``."""
    names = list(order or spec)
    graph = rx.PyDiGraph()
    index = {
        name: graph.add_node(
            {"dna_sequence": name, "scalar": spec[name][0], "event": "regular"}
        )
        for name in names
    }
    for parent, (_, kids) in spec.items():
        for kid in kids:
            payload = {}
            if chains and kid in chains:
                payload["vertex_chain_json"] = json.dumps(
                    [{"dna_sequence": s} for s in chains[kid]]
                )
            graph.add_edge(index[kid], index[parent], payload)
    return graph


def layout(graph, name, anchor=None, **kwargs):
    return layout_tree(graph, horizontal_layout=name, anchor_identity=anchor, **kwargs)


def x_by_identity(plot_data):
    return dict(zip(plot_data.node_table["dna_sequence"], plot_data.node_table["x"]))


def crossings(plot_data):
    """Count horizontal segments that cross a vertical one inside both."""
    segments = plot_data.edge_segment_table
    vertical = segments.filter(pl.col("segment_kind") == "vertical").to_dicts()
    horizontal = segments.filter(pl.col("segment_kind") == "horizontal").to_dicts()
    eps = 1e-9
    count = 0
    for h in horizontal:
        lo, hi = sorted((h["x_start"], h["x_end"]))
        for v in vertical:
            y_lo, y_hi = sorted((v["y_start"], v["y_end"]))
            if (
                lo + eps < v["x_start"] < hi - eps
                and y_lo + eps < h["y_start"] < y_hi - eps
            ):
                count += 1
    return count


def random_spec(seed, leaves=12):
    """A random split tree: merge ``leaves`` summits pairwise or three at a time."""
    rng = random.Random(seed)
    spec = {}
    pending = []
    for i in range(leaves):
        name = f"p{i:02d}"
        spec[name] = (round(rng.uniform(5.0, 9.0), 6) + i * 1e-4, [])
        pending.append(name)
    saddle = 0
    while len(pending) > 1:
        take = min(len(pending), rng.choice((2, 2, 3)))
        kids = [pending.pop(rng.randrange(len(pending))) for _ in range(take)]
        floor = min(spec[k][0] for k in kids)
        name = f"s{saddle:02d}"
        saddle += 1
        spec[name] = (round(floor - rng.uniform(0.2, 1.0), 6) - saddle * 1e-4, kids)
        pending.append(name)
    top = pending[0]
    spec["root"] = (spec[top][0] - 0.5, [top])
    return spec


def path_to_root(spec, name):
    parent = {kid: p for p, (_, kids) in spec.items() for kid in kids}
    path = [name]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path


def test_layout_names_and_unknown_name_error():
    assert HORIZONTAL_LAYOUTS == (
        "topographer_default",
        "subtree_extremum",
        "persistence_wt_anchored",
        "tidy",
    )
    with pytest.raises(ValueError, match="horizontal_layout must be one of"):
        layout(build_tree(SMALL), "radial")


def test_delegated_layouts_equal_topographer():
    graph = build_tree(SMALL)
    for name, option in (
        ("topographer_default", "leaf_order"),
        ("subtree_extremum", "subtree_extremum"),
    ):
        ours = layout(graph, name, node_attrs=("dna_sequence",))
        theirs = make_tree_layout(
            graph,
            scalar_attr="scalar",
            node_attrs=("dna_sequence",),
            horizontal_layout=option,
        )
        assert ours.node_table.equals(theirs.node_table)
        assert ours.edge_table.equals(theirs.edge_table)
        assert ours.edge_segment_table.equals(theirs.edge_segment_table)
        assert ours.root_node_index == theirs.root_node_index


@pytest.mark.parametrize("name", ANCHORED)
def test_custom_layouts_keep_topographers_table_format(name):
    # Only one edge carries ``vertex_chain_json``, so Topographer's edge rows
    # have different keys and the optional column is null on the other edges.
    graph = build_tree(SMALL, chains={"wt_peak": ["wt_peak", "WT", "other"]})
    edge_attrs = ("vertex_chain_json",)
    reference = make_tree_layout(
        graph,
        node_attrs=("dna_sequence",),
        edge_attrs=edge_attrs,
        horizontal_layout="iplotx",
    )
    ours = layout(graph, name, "wt_peak", edge_attrs=edge_attrs)
    assert "vertex_chain_json" in ours.edge_table.columns
    assert ours.edge_table["vertex_chain_json"].null_count() == SMALL_EDGES - 1
    assert ours.node_table.schema == reference.node_table.schema
    assert ours.edge_table.schema == reference.edge_table.schema
    assert ours.edge_segment_table.schema == reference.edge_segment_table.schema
    assert ours.edge_segment_table.columns == reference.edge_segment_table.columns
    assert ours.node_table.drop("x").equals(reference.node_table.drop("x"))
    assert ours.edge_table.drop("x_source", "x_target").equals(
        reference.edge_table.drop("x_source", "x_target")
    )
    assert ours.edge_segment_table.drop(
        "x_source", "x_target", "x_start", "x_end"
    ).equals(
        reference.edge_segment_table.drop("x_source", "x_target", "x_start", "x_end")
    )
    assert (ours.root_node_index, ours.root_kind, ours.scalar_attr) == (
        reference.root_node_index,
        reference.root_kind,
        reference.scalar_attr,
    )
    # Edge rows and segment rows must agree with the node coordinates.
    node_x = dict(zip(ours.node_table["tree_node_index"], ours.node_table["x"]))
    for row in ours.edge_table.to_dicts():
        assert row["x_source"] == node_x[row["source_tree_node_index"]]
        assert row["x_target"] == node_x[row["target_tree_node_index"]]
    for row in ours.edge_segment_table.to_dicts():
        source_x = node_x[row["source_tree_node_index"]]
        target_x = node_x[row["target_tree_node_index"]]
        assert row["x_start"] == target_x
        assert row["x_end"] == (
            target_x if row["segment_kind"] == "vertical" else source_x
        )


def test_segment_encoding_is_vertical_then_horizontal_per_edge():
    rows = [
        {"x_source": 1.0, "y_source": 2.0, "x_target": 3.0, "y_target": 5.0},
        {"x_source": 1.0, "y_source": 2.0, "x_target": 4.0, "y_target": 7.0},
    ]
    segments = tree_layouts._orthogonal_edge_segments(rows)
    keys = (
        "logical_edge_index",
        "segment_index",
        "segment_kind",
        "x_start",
        "y_start",
        "x_end",
        "y_end",
    )
    got = [tuple(seg[k] for k in keys) for seg in segments]
    assert got == [
        (0, 0, "vertical", 3.0, 5.0, 3.0, 2.0),
        (0, 1, "horizontal", 3.0, 2.0, 1.0, 2.0),
        (1, 0, "vertical", 4.0, 7.0, 4.0, 2.0),
        (1, 1, "horizontal", 4.0, 2.0, 1.0, 2.0),
    ]
    # The edge row's own fields are carried onto both of its segments.
    assert all(seg["x_source"] == 1.0 for seg in segments)


def test_anchored_path_to_root_is_vertical_at_zero_and_the_rest_is_right():
    plot = layout(build_tree(SMALL), "persistence_wt_anchored", "wt_peak")
    x = x_by_identity(plot)
    path = ["root", "s1", "s3", "wt_peak"]
    assert [x[name] for name in path] == [0.0] * 4
    assert all(x[name] > 0 for name in SMALL if name not in path)
    # Branch columns: one per summit, evenly spaced from 0.
    leaves = sorted(x[n] for n in ("g", "h", "w2", "wt_peak"))
    assert leaves == [0.0, 1.0, 2.0, 3.0]
    assert crossings(plot) == 0


def test_without_an_anchor_the_vertical_path_is_the_global_maximum_chain():
    for anchor in (None, "not_in_this_tree"):
        x = x_by_identity(layout(build_tree(SMALL), "persistence_wt_anchored", anchor))
        assert [x[n] for n in ("root", "s1", "s2", "g")] == [0.0] * 4
        assert x["wt_peak"] > 0


def test_subtree_extremum_alone_puts_the_wt_branch_second():
    """The failure the anchored layouts remove: the extremum order ignores the WT."""
    x = x_by_identity(layout(build_tree(SMALL), "subtree_extremum"))
    assert x["wt_peak"] > x["g"]


def test_tidy_puts_the_wt_leaf_first_and_centres_parents():
    plot = layout(build_tree(SMALL), "tidy", "wt_peak")
    x = x_by_identity(plot)
    assert x["wt_peak"] == 0.0 and x["w2"] == 1.0 and x["g"] == 2.0 and x["h"] == 3.0
    assert x["s3"] == 0.5 and x["s2"] == 2.5 and x["s1"] == 1.5
    assert crossings(plot) == 0


def test_tidy_centres_a_saddle_between_its_first_and_last_child_not_their_mean():
    """Topographer places a parent at the mean of its children; tidy at the midpoint."""
    spec = {
        "root": (0.0, ["s"]),
        "s": (1.0, ["a", "b", "c"]),
        "a": (5.0, ["a1", "a2"]),
        "a1": (7.0, []),
        "a2": (6.0, []),
        "b": (4.0, []),
        "c": (3.0, []),
    }
    tidy = x_by_identity(layout(build_tree(spec), "tidy"))
    assert [tidy[n] for n in ("a1", "a2", "b", "c")] == [0.0, 1.0, 2.0, 3.0]
    assert tidy["a"] == 0.5
    assert tidy["s"] == (tidy["a"] + tidy["c"]) / 2 == 1.75
    mean = x_by_identity(layout(build_tree(spec), "subtree_extremum"))
    assert mean["s"] == pytest.approx((0.5 + 2.0 + 3.0) / 3)


@pytest.mark.parametrize("name", ANCHORED)
def test_wt_stays_on_the_same_side_in_a_cropped_subtree(name):
    full = x_by_identity(layout(build_tree(SMALL), name, "wt_peak"))
    cropped_graph = crop_to_subtree(build_tree(SMALL), "s1")
    assert cropped_graph.num_nodes() == len(SMALL) - 1
    cropped = x_by_identity(layout(cropped_graph, name, "wt_peak"))
    # The wild-type leaf is the leftmost leaf in the tree and in its crop.
    leaves = ("g", "h", "w2", "wt_peak")
    for coordinates in (full, cropped):
        assert coordinates["wt_peak"] == min(coordinates[n] for n in leaves)
    if name == "persistence_wt_anchored":
        assert full["wt_peak"] == cropped["wt_peak"] == 0.0
        assert full["s1"] == cropped["s1"] == 0.0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("name", ANCHORED)
def test_random_trees_keep_wt_first_after_cropping_and_cross_nothing(name, seed):
    spec = random_spec(seed)
    summits = sorted(n for n in spec if n.startswith("p"))
    wt = summits[seed % len(summits)]
    full = layout(build_tree(spec), name, wt)
    assert crossings(full) == 0
    x = x_by_identity(full)
    assert x[wt] == min(x[s] for s in summits)
    if name == "persistence_wt_anchored":
        assert all(x[n] == 0.0 for n in path_to_root(spec, wt))
        assert all(v >= 0.0 for v in x.values())
    # Crop at every critical point on the wild type's path to the root.
    for crop_root in path_to_root(spec, wt)[1:-1]:
        cropped = layout(crop_to_subtree(build_tree(spec), crop_root), name, wt)
        assert crossings(cropped) == 0
        cx = x_by_identity(cropped)
        kept = [s for s in summits if s in cx]
        assert cx[wt] == min(cx[s] for s in kept)
        if name == "persistence_wt_anchored":
            assert cx[wt] == cx[crop_root] == 0.0


@pytest.mark.parametrize("name", HORIZONTAL_LAYOUTS)
def test_layout_is_deterministic_and_ignores_node_indices(name):
    spec = random_spec(3)
    names = list(spec)
    shuffled = names[:]
    random.Random(7).shuffle(shuffled)
    anchor = "p04"
    first = layout(build_tree(spec, names), name, anchor, node_attrs=("dna_sequence",))
    again = layout(build_tree(spec, names), name, anchor, node_attrs=("dna_sequence",))
    other = layout(
        build_tree(spec, shuffled), name, anchor, node_attrs=("dna_sequence",)
    )
    assert first.node_table.equals(again.node_table)
    assert first.edge_segment_table.equals(again.edge_segment_table)
    # This tree has no exact scalar ties, so even Topographer's own layouts, whose
    # tie-breaks use node indices, must not depend on the index assignment.
    assert x_by_identity(first) == pytest.approx(x_by_identity(other))


@pytest.mark.parametrize("name", ANCHORED)
def test_equal_persistence_ties_are_broken_by_identity(name):
    spec = {
        "root": (0.0, ["s"]),
        "s": (1.0, ["peak_b", "peak_a", "peak_c"]),
        "peak_a": (3.0, []),
        "peak_b": (3.0, []),
        "peak_c": (3.0, []),
    }
    reverse = ["peak_c", "peak_b", "peak_a", "s", "root"]
    for order in (list(spec), reverse):
        x = x_by_identity(layout(build_tree(spec, order), name))
        assert x["peak_a"] < x["peak_b"] < x["peak_c"]


def test_join_tree_mirrors_the_split_tree():
    spec = random_spec(2)
    split = x_by_identity(layout(build_tree(spec), "persistence_wt_anchored", "p03"))
    flipped = {k: (-v[0], v[1]) for k, v in spec.items()}
    join = x_by_identity(
        layout(
            build_tree(flipped),
            "persistence_wt_anchored",
            "p03",
            root_kind="maximum",
        )
    )
    assert join == pytest.approx(split)


def test_single_node_and_two_node_trees():
    one = {"only": (0.0, [])}
    two = {"low": (0.0, ["high"]), "high": (1.0, [])}
    for name in ANCHORED:
        assert x_by_identity(layout(build_tree(one), name)) == {"only": 0.0}
        assert x_by_identity(layout(build_tree(two), name)) == {"low": 0.0, "high": 0.0}


def test_custom_layouts_reject_a_network_and_missing_identity():
    graph = build_tree(SMALL)
    with pytest.raises(ValueError, match="does not support a network"):
        layout(graph, "tidy", network=object())
    nameless = rx.PyDiGraph()
    a = nameless.add_node({"scalar": 0.0, "dna_sequence": ""})
    b = nameless.add_node({"scalar": 1.0, "dna_sequence": "x"})
    nameless.add_edge(b, a, {})
    with pytest.raises(ValueError, match="non-empty"):
        layout(nameless, "tidy")
    duplicate = build_tree({"a": (0.0, ["b"]), "b": (1.0, [])})
    duplicate[1] = {**duplicate[1], "dna_sequence": "a"}
    with pytest.raises(ValueError, match="unique"):
        layout(duplicate, "tidy")


def test_anchor_identity_for_a_sequence_on_a_branch():
    chains = {"wt_peak": ["wt_peak", "WT", "other"]}
    graph = build_tree(SMALL, chains=chains)
    assert anchor_identity_for_sequence(graph, "s3") == "s3"
    assert anchor_identity_for_sequence(graph, "WT") == "wt_peak"
    assert anchor_identity_for_sequence(graph, "missing") is None
    # A join tree takes the lower endpoint as the one farther from the root.
    flipped = {k: (-v[0], v[1]) for k, v in SMALL.items()}
    join = build_tree(flipped, chains=chains)
    assert anchor_identity_for_sequence(join, "WT", root_kind="maximum") == "wt_peak"


def test_crop_to_subtree_copies_payloads_and_needs_a_digraph():
    graph = build_tree(SMALL)
    cropped = crop_to_subtree(graph, "s3")
    assert sorted(cropped[i]["dna_sequence"] for i in cropped.node_indices()) == [
        "s3",
        "w2",
        "wt_peak",
    ]
    cropped[0]["scalar"] = 99.0
    assert graph[0]["scalar"] == 0.0
    with pytest.raises(TypeError):
        crop_to_subtree(rx.PyGraph(), "s3")
    with pytest.raises(ValueError, match="exactly one node"):
        crop_to_subtree(graph, "nope")
