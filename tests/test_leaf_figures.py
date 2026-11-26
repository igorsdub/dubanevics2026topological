"""Leaf tables, marker areas, the swarm layout and the four leaf diagnostic figures."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import rustworkx as rx
import vl_convert as vlc
from _figure_labels import SENTINEL, label_quantities, spec_text

from dubanevics2026topological import figures
from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures._leaf_data import (
    GLOBAL_PAIR_KIND,
    LeafTables,
    leaf_tables,
    marker_areas,
    with_marker_area,
)
from dubanevics2026topological.build import FOUR_GRAPH_FIGURE_IDS
from dubanevics2026topological.figures.leaf_swarm import (
    FDR_PANEL_IDS,
    PANEL_SLUGS,
    build_leaf_swarm_panel_figure,
    swarm_layout,
    swarm_offsets,
    swarm_rows,
)

ROOT = Path(__file__).resolve().parents[1]
IDS = {
    "persistence": (
        "published_leaf_persistence",
        "candidate_leaf_persistence",
    ),
    "swarm": (
        "published_leaf_swarm",
        "candidate_leaf_swarm",
    ),
}
ALL_IDS = IDS["persistence"] + IDS["swarm"]

# A valid split tree. A (3.0, the global maximum, three sequences), B (2.5, stop-codon
# singleton) and C (2.0, wild type, two sequences) peak above S1 (1.0), where B and C
# meet; S2 (0.5) joins A to them; D (2.0, plain singleton) joins at S3 (0.25) above
# the root. By the elder rule B dies at S2 although its adjacent saddle is S1, and
# C and D tie on birth fitness, so their rank follows peak sequence.
NODES = [
    ("AAAAAA", "KK", 3.0, "maximum"),  # 0 A
    ("CCCCCC", "**", 2.5, "maximum"),  # 1 B
    ("GGGGGG", "PK", 2.0, "maximum"),  # 2 C
    ("TCTCTC", "PP", 2.0, "maximum"),  # 3 D
    ("TTTTTT", "PP", 1.0, "split_saddle"),  # 4 S1
    ("TTTTTA", "PQ", 0.5, "split_saddle"),  # 5 S2
    ("TTTTTG", "QP", 0.25, "split_saddle"),  # 6 S3
    ("TTTTTC", "QQ", 0.0, "minimum"),  # 7 root
]
EDGES = [(0, 5), (1, 4), (2, 4), (4, 5), (5, 6), (3, 6), (6, 7)]
# Each node's x in the tree drawing. The peaks run left to right B, C, D, A: not the
# birth-fitness rank (A, B, C, D), so the tree order is told apart from the rank.
NODE_X = [3.0, 0.0, 1.0, 2.0, 0.5, 2.5, 2.25, 2.0]
# The sequence of a hill member that carries a stop codon.
STOP_MEMBER = "GGGGGA"
MEMBERS = [
    (0, 5, "AAAAAC", 2.7),
    (0, 5, "AAAAAG", 2.0),
    (2, 4, "GGGGGA", 1.5),
]


def _landscape(kind: str = "extended", *, wild_type: str | None = "GGGGGG"):
    graph = rx.PyDiGraph()
    for position, (dna, aa, scalar, event) in enumerate(NODES):
        graph.add_node(
            {
                "vertex_index": position,
                "dna_sequence": dna,
                "aa_sequence": aa,
                "scalar": scalar,
                "event": event,
            }
        )
    for upper, lower in EDGES:
        graph.add_edge(upper, lower, {})
    regular = {(u, l): 0 for u, l in EDGES}
    for upper, lower, _, _ in MEMBERS:
        regular[(upper, lower)] += 1
    branches = pl.DataFrame(
        [
            {
                "source_tree_node_index": u,
                "target_tree_node_index": l,
                "branch_member_count": n,
            }
            for (u, l), n in regular.items()
        ]
    )
    members = pl.DataFrame(
        [
            {
                "source_tree_node_index": u,
                "target_tree_node_index": l,
                "dna_sequence": dna,
                "aa_sequence": "PP",
                "scalar": fitness,
                "has_stop_codon": dna == STOP_MEMBER,
            }
            for u, l, dna, fitness in MEMBERS
        ]
    )
    # B's paired death is S2 although its adjacent saddle is S1: the paired death
    # must be drawn, never the adjacent saddle.
    pairs = pl.DataFrame(
        [
            {
                "birth_tree_node_index": b,
                "death_tree_node_index": d,
                "birth_scalar": graph[b]["scalar"],
                "death_scalar": graph[d]["scalar"],
                "pair_kind": k,
            }
            for b, d, k in [
                (0, 7, kind),
                (1, 5, "ordinary"),
                (2, 4, "ordinary"),
                (3, 6, "ordinary"),
            ]
        ]
    )
    coordinates = pl.DataFrame(
        [
            {
                "tree_node_index": i,
                "dna_sequence": graph[i]["dna_sequence"],
                "x": NODE_X[i],
                "y": graph[i]["scalar"],
            }
            for i in graph.node_indices()
        ]
    )
    marker = (
        pl.DataFrame({"dna_sequence": [wild_type]})
        if wild_type
        else pl.DataFrame({"dna_sequence": []}, schema={"dna_sequence": pl.String})
    )
    return SimpleNamespace(
        trees={
            "split": SimpleNamespace(
                tree=graph,
                branch_table=branches,
                branch_member_table=members,
                persistence_table=pairs,
                plot_data=SimpleNamespace(node_table=coordinates),
                wild_type_marker_table=marker,
            )
        }
    )


def _annotations(landscape, key: str = "study_clamped"):
    """Saved annotation evidence of a fixture landscape, as the export would hold."""
    from dubanevics2026topological.analysis import describe

    return describe(
        products={"annotations"}, landscape=landscape, landscape_name=key
    ).annotations


def _tables(landscape):
    return leaf_tables(
        landscape, key="study_clamped", pairs=_annotations(landscape).pairs
    )


def _payload(kind: str = "extended"):
    return {
        "landscapes": {
            "study_clamped": _landscape(kind),
            "candidate_clamped": _landscape(kind),
        },
        "annotations": {
            "study_clamped": _annotations(_landscape(kind)),
            "candidate_clamped": _annotations(_landscape(kind), "candidate_clamped"),
        },
        "manuscript_quantities": label_quantities(),
    }


def _settings():
    keys = {
        "published_leaf_persistence": {"panel_size": 300},
        "candidate_leaf_persistence": {"panel_size": 300},
        "published_leaf_swarm": {"leaves_per_row": 2, "row_width": 200},
        "candidate_leaf_swarm": {"leaves_per_row": 2, "row_width": 200},
    }
    return {"figures": {"rendering": {"font_size": 15}, "figures": keys}}


# -- data assembly ------------------------------------------------------------------


def test_leaf_table_ranks_by_birth_descending_with_sequence_tie_break() -> None:
    leaves = _tables(_landscape()).leaves
    assert leaves["peak_sequence"].to_list() == [
        "AAAAAA",
        "CCCCCC",
        "GGGGGG",
        "TCTCTC",
    ]
    assert leaves["rank"].to_list() == [1, 2, 3, 4]
    assert leaves["birth_scalar"].to_list() == [3.0, 2.5, 2.0, 2.0]


def test_leaf_table_carries_paired_death_and_distinct_adjacent_saddle() -> None:
    leaves = _tables(_landscape()).leaves
    stop = leaves.filter(pl.col("peak_sequence") == "CCCCCC").row(0, named=True)
    # B dies at S2 (0.5) but its adjacent saddle is S1 (1.0).
    assert stop["death_scalar"] == 0.5
    assert stop["death_sequence"] == "TTTTTA"
    assert stop["adjacent_saddle_scalar"] == 1.0
    assert stop["adjacent_saddle_sequence"] == "TTTTTT"
    wild = leaves.filter(pl.col("peak_sequence") == "GGGGGG").row(0, named=True)
    assert wild["member_count"] == 2
    assert wild["is_wild_type"] and not wild["birth_has_stop_codon"]
    assert (wild["death_scalar"], wild["adjacent_saddle_scalar"]) == (1.0, 1.0)


def test_leaf_table_counts_singletons_and_flags_stop_codon() -> None:
    leaves = _tables(_landscape()).leaves
    assert leaves["member_count"].to_list() == [3, 1, 2, 1]
    assert leaves["is_singleton"].to_list() == [False, True, False, True]
    assert leaves["birth_has_stop_codon"].to_list() == [False, True, False, False]
    assert leaves["is_wild_type"].to_list() == [False, False, True, False]


def test_global_pair_is_flagged_and_other_kinds_are_rejected() -> None:
    leaves = _tables(_landscape(GLOBAL_PAIR_KIND)).leaves
    assert leaves["is_global_pair"].to_list() == [True, False, False, False]
    top = leaves.row(0, named=True)
    assert (top["birth_scalar"], top["death_scalar"]) == (3.0, 0.0)
    # The Full-graph spelling belongs to another table and is not accepted here.
    with pytest.raises(ValueError, match="exactly one global pair"):
        _tables(_landscape("extended_positive"))
    with pytest.raises(ValueError, match="exactly one global pair"):
        _tables(_landscape("ordinary"))


def test_a_hill_size_differing_from_the_branch_vertex_count_is_rejected() -> None:
    landscape = _landscape()
    split = landscape.trees["split"]
    split.branch_table = split.branch_table.with_columns(
        pl.when(pl.col("source_tree_node_index") == 0)
        .then(pl.col("branch_member_count") + 1)
        .otherwise(pl.col("branch_member_count"))
        .alias("branch_member_count")
    )
    with pytest.raises(ValueError, match="branch vertex count"):
        _tables(landscape)


def test_every_peak_must_be_exactly_one_persistence_birth() -> None:
    landscape = _landscape()
    # D no longer counts as a hill's maximum, yet is still a leaf with a pair.
    landscape.trees["split"].tree[3]["event"] = "merge"
    with pytest.raises(ValueError, match="exactly one persistence birth"):
        _tables(landscape)


def test_member_table_holds_every_hill_sequence_with_identities_only() -> None:
    members = _tables(_landscape()).members
    first = members.filter(pl.col("leaf_peak_sequence") == "AAAAAA")
    assert first["dna_sequence"].to_list() == ["AAAAAA", "AAAAAC", "AAAAAG"]
    assert first["raw_fitness"].to_list() == [3.0, 2.7, 2.0]
    assert first["is_peak"].to_list() == [True, False, False]
    assert members["leaf_peak_sequence"].value_counts().height == 4
    assert members.height == 7
    assert not any(column.endswith("_index") for column in members.columns)
    leaves = _tables(_landscape()).leaves
    assert not any(column.endswith("_index") for column in leaves.columns)


# -- marker area --------------------------------------------------------------------


def test_marker_area_is_proportional_to_count_up_to_the_floor() -> None:
    areas = marker_areas([1, 10, 50, 100], max_area=400, min_area=0.0)
    assert areas == pytest.approx([4, 40, 200, 400])
    assert [a / c for a, c in zip(areas, [1, 10, 50, 100])] == pytest.approx([4] * 4)


def test_marker_area_floor_applies_only_below_it() -> None:
    areas = marker_areas([1, 2, 10, 100], max_area=400, min_area=12)
    # 4 and 8 fall below the floor; 40 and 400 are left exactly proportional.
    assert areas == [12, 12, 40, 400]
    assert marker_areas([], max_area=400, min_area=12) == []
    with pytest.raises(ValueError):
        marker_areas([0, 3], max_area=400, min_area=12)


def test_with_marker_area_adds_the_column() -> None:
    leaves = _tables(_landscape()).leaves
    area = with_marker_area(leaves, max_area=300, min_area=0.0)["marker_area"]
    assert area.to_list() == pytest.approx([300, 100, 200, 100])


# -- swarm layout -------------------------------------------------------------------


def _distances(heights, offsets):
    return [
        math.hypot(offsets[i] - offsets[j], heights[i] - heights[j])
        for i in range(len(heights))
        for j in range(i)
    ]


def test_swarm_dots_do_not_overlap_before_the_cap() -> None:
    heights = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.9, 3.0, 3.1, 10.0]
    offsets, overflow = swarm_offsets(heights, diameter=3.0, max_half_width=30.0)
    assert not any(overflow)
    assert min(_distances(heights, offsets)) >= 3.0 - 1e-6


def test_swarm_offsets_are_symmetric_and_deterministic() -> None:
    heights = [5.0] * 5
    offsets, overflow = swarm_offsets(heights, diameter=2.0, max_half_width=10.0)
    assert offsets == [0.0, 2.0, -2.0, 4.0, -4.0]
    assert not any(overflow)
    assert swarm_offsets(heights, diameter=2.0, max_half_width=10.0) == (
        offsets,
        overflow,
    )
    assert sorted(offsets) == [-v for v in sorted(offsets, reverse=True)]


def test_swarm_input_order_decides_ties_only() -> None:
    heights = [3.0, 7.0, 3.0]
    offsets, _ = swarm_offsets(heights, diameter=2.0, max_half_width=10.0)
    # The 7.0 dot is alone; the two equal dots take 0 and +2 in input order.
    assert offsets == [0.0, 0.0, 2.0]


def test_swarm_cap_keeps_offsets_inside_and_flags_the_excess() -> None:
    heights = [0.0] * 12
    offsets, overflow = swarm_offsets(heights, diameter=2.0, max_half_width=4.0)
    assert max(abs(v) for v in offsets) <= 4.0 + 1e-9
    # Five slots 2 apart fit within +-4; the other seven overlap.
    assert sum(overflow) == 7
    assert overflow[:5] == [False] * 5
    # The excess is spread over the slots, not stacked on one.
    assert len(set(offsets)) > 5


def test_swarm_zero_width_cap_stacks_every_dot_on_the_centre() -> None:
    offsets, overflow = swarm_offsets([1.0, 1.0, 1.0], diameter=2.0, max_half_width=0)
    assert offsets == [0.0, 0.0, 0.0]
    assert overflow == [False, True, True]


def test_swarm_layout_wraps_leaves_in_tree_order_into_rows_and_columns() -> None:
    tables = _tables(_landscape())
    layout = swarm_layout(
        tables,
        leaves_per_row=2,
        column_width=100.0,
        row_height=100.0,
        domain=[-0.5, 3.5],
        diameter=3.0,
        half_width_fraction=0.4,
    )
    positions = layout.leaves.select("peak_sequence", "row", "column").rows()
    # The tree draws the peaks B, C, D, A from the left (NODE_X), not by birth rank.
    assert positions == [
        ("AAAAAA", 1, 1),
        ("CCCCCC", 0, 0),
        ("GGGGGG", 0, 1),
        ("TCTCTC", 1, 0),
    ]
    for row in layout.members.iter_rows(named=True):
        # Every dot sits inside its own column.
        assert row["column"] < row["x"] < row["column"] + 1
        assert abs(row["swarm_offset"]) <= 40.0
    # The peak (3.0) is the highest dot of the leaf and takes the centre of its column.
    peak = layout.members.filter(pl.col("dna_sequence") == "AAAAAA").row(0, named=True)
    assert peak["swarm_offset"] == 0.0 and peak["x"] == 1.5
    assert layout.members.height == 7


# -- the figures --------------------------------------------------------------------


def _resolve(spec, layer):
    data = layer.get("data", {})
    return spec["datasets"][data["name"]] if "name" in data else data.get("values", [])


def _marker_layers(spec):
    return [
        layer
        for layer in spec["layer"]
        if layer["mark"]["type"] == "point"
        and layer.get("encoding", {}).get("size", {}).get("field") == "marker_area"
    ]


@pytest.mark.parametrize("figure_id", IDS["persistence"])
def test_persistence_markers_are_all_filled_singletons_included(figure_id) -> None:
    chart = figures.render_payload(
        figures.get_spec(figure_id), _payload(), _settings(), "vanilla"
    )
    spec = chart.to_dict()
    # Paired death on x, birth on y, in fitness units on one shared domain.
    # The diagram is one layered chart: the key is drawn inside it, no row below.
    assert "vconcat" not in spec
    assert spec["width"] == spec["height"] == 300
    plain = {}
    for layer in _marker_layers(spec):
        if layer["mark"]["color"] == palette.EMPHASIS_BLACK:
            continue  # the global pair's black triangle, checked separately
        encoding = layer["encoding"]
        assert encoding["x"]["field"] == "death_scalar"
        assert encoding["y"]["field"] == "birth_scalar"
        # Identity scale: Altair must not rescale the precomputed area.
        assert encoding["size"]["scale"] is None
        for row in _resolve(spec, layer):
            plain[row["peak_sequence"]] = (
                layer["mark"]["filled"],
                layer["mark"]["color"],
                row["marker_area"],
                row["death_scalar"],
            )
    # No hollow markers: every marker is filled, singletons included.
    assert all(filled is True for filled, *_ in plain.values())
    # A is the global pair (filled grey), B the stop-codon singleton (filled crimson),
    # C the wild type (filled red). C's death is its paired 1.0, not its saddle 0.5.
    assert plain["AAAAAA"][:2] == (True, palette.PEAK_GREY)
    assert plain["CCCCCC"][:2] == (True, palette.STOP_CODON_CRIMSON)
    assert plain["GGGGGG"][:2] == (True, palette.WILD_TYPE_RED)
    # D is a plain singleton: filled and mid-grey.
    assert plain["TCTCTC"][:2] == (True, palette.PEAK_GREY)
    assert plain["GGGGGG"][3] == 1.0
    # Area is proportional to the hill's sequence count (3, 1, 2), none below the floor.
    assert plain["AAAAAA"][2] == 400
    assert plain["CCCCCC"][2] == pytest.approx(400 / 3)
    assert plain["GGGGGG"][2] == pytest.approx(400 * 2 / 3)


@pytest.mark.parametrize("figure_id", IDS["persistence"])
def test_persistence_draws_the_diagonal_and_marks_the_global_pair(figure_id) -> None:
    spec = figures.render_payload(
        figures.get_spec(figure_id), _payload(), _settings(), "vanilla"
    ).to_dict()
    rules = [layer for layer in spec["layer"] if layer["mark"]["type"] == "rule"]
    assert len(rules) == 1
    (diagonal,) = _resolve(spec, rules[0])
    assert diagonal["death_scalar"] == diagonal["birth_scalar"]
    assert diagonal["death_end"] == diagonal["birth_end"]
    rings = [
        layer
        for layer in _marker_layers(spec)
        if layer["mark"]["color"] == palette.EMPHASIS_BLACK
    ]
    assert len(rings) == 1
    assert rings[0]["mark"]["filled"] is True  # a filled black triangle, not an outline
    (ring_row,) = _resolve(spec, rings[0])
    assert (ring_row["birth_scalar"], ring_row["death_scalar"]) == (3.0, 0.0)
    labels = [layer["mark"].get("text") for layer in spec["layer"]]
    assert "Global maximum pair" in labels
    # The key sits inside the diagram, bottom right, one entry per line (peaks only:
    # a leaf diagram has no valleys).
    entries = [
        layer["mark"]
        for layer in spec["layer"]
        if layer["mark"]["type"] == "text"
        and layer["mark"].get("text") in {"peak", "valley", "Stop codon containing peaks", "wild type"}
    ]
    assert [mark["text"] for mark in entries] == ["peak", "Stop codon containing peaks", "wild type"]
    ys = [mark["y"] for mark in entries]
    assert ys == sorted(ys) and ys[-1] <= spec["height"]
    assert all(mark["x"] > spec["width"] / 10 for mark in entries)  # long stop label


@pytest.mark.parametrize("figure_id", IDS["swarm"])
def test_swarm_rows_carry_birth_and_adjacent_saddle_marks_per_column(figure_id) -> None:
    spec = figures.render_payload(
        figures.get_spec(figure_id), _payload(), _settings(), "vanilla"
    ).to_dict()
    # Four leaves, two per row: two stacked rows.
    assert len(spec["vconcat"]) == 2
    first, second = spec["vconcat"]
    assert first["height"] == second["height"] == 130
    leaves = spec["datasets"]["leaf_swarm_row_0_leaves"]
    # Columns follow the tree's left-to-right peak order (B, C, D, A), not birth fitness.
    assert [row["peak_sequence"] for row in leaves] == ["CCCCCC", "GGGGGG"]
    assert [row["column"] for row in leaves] == [0, 1]
    assert [row["birth_scalar"] for row in leaves] == [2.5, 2.0]
    assert [row["adjacent_saddle_scalar"] for row in leaves] == [1.0, 1.0]
    assert [row["birth_color"] for row in leaves] == [
        palette.STOP_CODON_CRIMSON,
        palette.WILD_TYPE_RED,
    ]
    plain, grey = spec["datasets"]["leaf_swarm_row_1_leaves"]
    assert plain["birth_color"] == palette.PEAK_GREY
    assert plain["adjacent_saddle_scalar"] == 0.25
    assert grey["peak_sequence"] == "AAAAAA"
    assert grey["adjacent_saddle_scalar"] == 0.5

    def key_texts(row):
        return [
            layer["data"]["values"][0]["label"]
            for layer in row["layer"]
            if layer["mark"]["type"] == "text"
        ]

    # The key lives in the last row, never over the top-ranked leaves, and names the
    # stop-codon and wild-type rules because both are present.
    assert key_texts(first) == []
    assert key_texts(second) == [
        "Member sequence",
        "Birth (peak) fitness",
        "Adjacent saddle fitness",
        "Stop-codon peak",
        "Wild type",
        "Stop-codon sequence",
    ]
    # No solid or dashed lines anywhere: the swarm is dots, the leaf's birth (peak)
    # fitness a triangle and its adjacent saddle a diamond.
    for row in spec["vconcat"]:
        assert not [
            layer for layer in row["layer"] if layer["mark"]["type"] == "rule"
        ]
        assert not [layer for layer in row["layer"] if "strokeDash" in layer["mark"]]
    data_layers = [
        layer
        for layer in second["layer"]
        if layer["encoding"].get("x", {}).get("field") is not None
    ]
    # Plain grey members first, the highlighted (stop-codon, synonymous) ones over them.
    assert [layer["mark"]["type"] for layer in data_layers] == [
        "circle",
        "circle",
        "point",
        "point",
    ]
    plain_dots, marked_dots, saddle, birth = data_layers
    assert plain_dots["transform"] == [{"filter": "!datum.highlighted"}]
    assert marked_dots["transform"] == [{"filter": "datum.highlighted"}]
    assert plain_dots["mark"]["opacity"] == marked_dots["mark"]["opacity"] == 0.5
    assert saddle["mark"]["shape"] == "diamond" and saddle["mark"]["filled"] is True
    assert birth["mark"]["shape"] == "triangle-up" and birth["mark"]["filled"] is True
    # Both stand at the column's centre, at the leaf's own fitness.
    assert saddle["encoding"]["y"]["field"] == "adjacent_saddle_scalar"
    assert birth["encoding"]["y"]["field"] == "birth_scalar"
    assert saddle["encoding"]["x"]["field"] == birth["encoding"]["x"]["field"] == "centre"
    assert birth["encoding"]["color"]["field"] == "birth_color"
    # The key's glyphs repeat the marks: a dot, a triangle, a diamond, then the
    # annotated triangles; none is a stroke.
    glyphs = [
        layer["mark"]["shape"]
        for layer in second["layer"]
        if layer["mark"]["type"] == "point"
        and layer["encoding"].get("x", {}).get("value") is not None
    ]
    assert glyphs == [
        "circle",
        "triangle-up",
        "diamond",
        "triangle-up",
        "triangle-up",
        "circle",
    ]
    assert "stroke" not in glyphs
    # One dot per hill sequence: B (1) and C (2) in the first row, D (1) and A (3) in
    # the second, the tree's order.
    assert len(spec["datasets"]["leaf_swarm_row_0_members"]) == 3
    assert len(spec["datasets"]["leaf_swarm_row_1_members"]) == 4


@pytest.mark.parametrize("figure_id", ALL_IDS)
def test_leaf_figures_render_to_svg(figure_id) -> None:
    chart = figures.render_payload(
        figures.get_spec(figure_id), _payload(), _settings(), "vanilla"
    )
    assert vlc.vegalite_to_svg(chart.to_dict(), vl_version="v5_21").startswith("<svg")


@pytest.mark.parametrize("graph", ["published", "candidate"])
def test_leaf_persistence_subtitle_is_read_from_the_registry(graph) -> None:
    payload = _payload()
    figure_id = f"{graph}_leaf_persistence"
    spec = figures.render_payload(
        figures.get_spec(figure_id), payload, _settings(), "vanilla"
    ).to_dict()
    assert spec["title"]["subtitle"] == [
        f"Papkou et al. (2023) · {SENTINEL[f'{graph}_graph.peak_count']} leaves",
        (
            f"{SENTINEL[f'annotations.{graph}.singleton_peak_count']} singleton hills "
            "· largest hill "
            f"{SENTINEL[f'{graph}_graph.largest_hill_sequence_count']} sequences"
        ),
    ]
    assert set(payload["manuscript_quantities"].requested_keys) == {
        f"{graph}_graph.peak_count",
        f"annotations.{graph}.singleton_peak_count",
        f"{graph}_graph.largest_hill_sequence_count",
    }


@pytest.mark.parametrize("graph", ["published", "candidate"])
def test_leaf_swarm_subtitle_is_read_from_the_registry(graph) -> None:
    payload = _payload()
    figure_id = f"{graph}_leaf_swarm"
    spec = figures.render_payload(
        figures.get_spec(figure_id), payload, _settings(), "vanilla"
    ).to_dict()
    first = spec["title"]["subtitle"][0]
    assert first == (
        f"Papkou et al. (2023) · {SENTINEL[f'{graph}_graph.peak_count']} leaves · "
        f"{SENTINEL[f'{graph}_graph.hill_sequence_count']} sequences in hills"
    )
    # The overlap line is a layout diagnostic, exempt from the registry.
    assert set(payload["manuscript_quantities"].requested_keys) == {
        f"{graph}_graph.peak_count",
        f"{graph}_graph.hill_sequence_count",
    }
    assert SENTINEL[f"{graph}_graph.peak_count"] in spec_text(spec)


# -- registry and configuration -----------------------------------------------------


def test_the_four_leaf_figures_are_registered_vanilla_comparison_figures() -> (
    None
):
    for figure_id in ALL_IDS:
        spec = figures.get_spec(figure_id)
        assert figure_id in figures.FIGURE_IDS
        # The swarms are manuscript figures (S10, S12); the persistence diagrams are opt-in.
        assert spec.default is figure_id.endswith("_leaf_swarm")
        assert spec.inputs == frozenset(
            {"comparison", "manuscript_quantities", "annotations"}
        )
        assert spec.variants == frozenset({"vanilla"})
        assert spec.config_table == figure_id
        with pytest.raises(ValueError, match="canonical vanilla"):
            figures.render_payload(spec, _payload(), _settings(), "annotated")


def test_leaf_figure_config_tables_set_appearance_only() -> None:
    tables = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))["figures"]
    for figure_id in ALL_IDS:
        table = tables[figure_id]
        assert "notebook" not in table
        assert not [key for key in table if key.endswith("_color")]
    for figure_id in IDS["persistence"]:
        assert (
            tables[figure_id]["max_marker_area"] > tables[figure_id]["min_marker_area"]
        )
    for figure_id in IDS["swarm"]:
        assert tables[figure_id]["leaves_per_row"] >= 1


def test_saved_annotation_tables_round_trip_to_the_describe_evidence(tmp_path):
    from dubanevics2026topological.analysis import describe
    from dubanevics2026topological.analysis.describe import load_saved_annotations

    landscape = _landscape()
    built = describe(
        products={"annotations"},
        landscape=landscape,
        landscape_name="study_clamped",
        annotation_directory=tmp_path,
    ).annotations
    saved = load_saved_annotations(tmp_path)
    for name in ("branches", "pairs", "critical_points"):
        assert getattr(saved, name).equals(getattr(built, name))
    assert saved.summary == built.summary
    assert leaf_tables(landscape, key="study_clamped", pairs=saved.pairs).leaves.equals(
        _tables(landscape).leaves
    )


# -- synonymous leaves in green ------------------------------------------------------


def _green_landscape():
    """The fixture with one synonymous saddle and one synonymous member.

    S1 takes B's amino acids (``**``), so B's adjacent saddle is synonymous to it; the
    member AAAAAC takes A's (``KK``), so it is synonymous to its peak.
    """
    landscape = _landscape()
    split = landscape.trees["split"]
    split.tree[4]["aa_sequence"] = "**"
    split.branch_member_table = split.branch_member_table.with_columns(
        pl.when(pl.col("dna_sequence") == "AAAAAC")
        .then(pl.lit("KK"))
        .otherwise(pl.col("aa_sequence"))
        .alias("aa_sequence")
    )
    return landscape


def _green_tables():
    landscape = _green_landscape()
    return leaf_tables(
        landscape, key="study_clamped", pairs=_annotations(landscape).pairs
    )


def test_a_synonymous_saddle_and_member_are_flagged_and_nothing_else() -> None:
    tables = _green_tables()
    flagged = tables.leaves.filter(pl.col("saddle_is_synonymous"))
    assert flagged["peak_sequence"].to_list() == ["CCCCCC"]
    assert flagged["adjacent_saddle_aa_sequence"].to_list() == ["**"]
    synonymous = tables.members.filter(pl.col("is_synonymous"))
    assert synonymous["dna_sequence"].to_list() == ["AAAAAC"]
    # The peak is never its own synonymous member.
    assert not tables.members.filter(pl.col("is_peak"))["is_synonymous"].any()
    plain = _tables(_landscape())
    assert not plain.leaves["saddle_is_synonymous"].any()
    assert not plain.members["is_synonymous"].any()


def test_a_branch_member_table_without_aa_sequence_is_an_error() -> None:
    landscape = _landscape()
    split = landscape.trees["split"]
    split.branch_member_table = split.branch_member_table.drop("aa_sequence")
    with pytest.raises(ValueError, match="aa_sequence"):
        _tables(landscape)


@pytest.mark.parametrize(
    ("frame", "column"),
    [("members", "is_synonymous"), ("leaves", "saddle_is_synonymous")],
)
def test_swarm_rows_without_the_synonymous_columns_raise(frame, column) -> None:
    tables = _tables(_landscape())
    broken = {"members": tables.members, "leaves": tables.leaves}
    broken[frame] = broken[frame].drop(column)
    with pytest.raises(ValueError, match=column):
        swarm_rows(LeafTables(leaves=broken["leaves"], members=broken["members"]), {})


def _green_payload():
    return {
        "landscapes": {
            "study_clamped": _green_landscape(),
            "candidate_clamped": _green_landscape(),
        },
        "annotations": {
            "study_clamped": _annotations(_green_landscape()),
            "candidate_clamped": _annotations(_green_landscape(), "candidate_clamped"),
        },
        "manuscript_quantities": label_quantities(),
    }


@pytest.mark.parametrize("figure_id", IDS["swarm"])
def test_swarm_colours_synonymous_dots_and_saddles_in_the_tree_green(figure_id) -> None:
    spec = figures.render_payload(
        figures.get_spec(figure_id), _green_payload(), _settings(), "vanilla"
    ).to_dict()
    # The tree puts the leaf with the synonymous member (A) in the second row.
    members = spec["datasets"]["leaf_swarm_row_1_members"]
    colours = sorted(row["dot_color"] for row in members)
    assert colours.count(palette.SYNONYMOUS_GREEN) == 1
    assert colours.count(palette.PEAK_GREY) == len(members) - 1
    assert sorted(row["highlighted"] for row in members) == [False] * (len(members) - 1) + [True]
    leaves = spec["datasets"]["leaf_swarm_row_0_leaves"]
    assert [row["saddle_color"] for row in leaves] == [
        palette.SYNONYMOUS_GREEN,
        palette.EMPHASIS_BLACK,
    ]
    second = spec["vconcat"][1]
    texts = [
        layer["data"]["values"][0]["label"]
        for layer in second["layer"]
        if layer["mark"]["type"] == "text"
    ]
    assert texts[-1] == "Synonymous to its peak"
    dots = next(layer for layer in spec["vconcat"][0]["layer"] if layer["mark"]["type"] == "circle")
    assert dots["encoding"]["color"]["field"] == "dot_color"
    saddle = next(
        layer
        for layer in spec["vconcat"][0]["layer"]
        if layer["mark"].get("shape") == "diamond"
    )
    assert saddle["encoding"]["color"]["field"] == "saddle_color"


def test_the_key_names_synonymous_only_when_the_figure_has_one() -> None:
    spec = figures.render_payload(
        figures.get_spec("published_leaf_swarm"), _payload(), _settings(), "vanilla"
    ).to_dict()
    texts = [
        layer["data"]["values"][0]["label"]
        for layer in spec["vconcat"][1]["layer"]
        if layer["mark"]["type"] == "text"
    ]
    assert "Synonymous to its peak" not in texts


def test_the_green_is_a_configurable_rendering_key_not_a_swarm_key() -> None:
    settings = _settings()
    settings["figures"]["rendering"]["synonymous_color"] = "#00ff00"
    spec = figures.render_payload(
        figures.get_spec("published_leaf_swarm"), _green_payload(), settings, "vanilla"
    ).to_dict()
    assert "#00ff00" in {
        row["dot_color"] for row in spec["datasets"]["leaf_swarm_row_1_members"]
    }


# -- the four graphs: the 2x2 and its panels -----------------------------------------


def _four_payload():
    tables = _green_tables()
    return {
        "panels": {
            variant: {"leaf_tables": tables, "branch_table": None}
            for variant, _, _ in PANEL_SLUGS.values()
        }
    }


def _four_settings():
    layout = {"leaves_per_row": 2, "row_width": 200, "row_height": 60}
    keys = {figure_id: layout for figure_id in FDR_PANEL_IDS}
    return {
        "analysis": {"clamped": {"candidate_functional_threshold": {"q": 0.05}}},
        "figures": {"rendering": {"font_size": 15}, "figures": keys},
    }


@pytest.mark.parametrize("figure_id", FDR_PANEL_IDS)
def test_each_by_panel_is_its_own_figure(figure_id) -> None:
    spec = build_leaf_swarm_panel_figure(
        _four_payload(), _four_settings(), figure_id
    ).to_dict()
    graph = "Papkou" if figure_id.startswith("published") else "Topographer"
    assert spec["title"]["text"] == f"{graph} + FDR: sequences of every leaf"
    assert spec["title"]["subtitle"][0] == "FDR: BY, q = 0.05"
    assert spec["title"]["subtitle"][1] == "4 leaves · 7 sequences in hills"
    assert len(spec["vconcat"]) == 2
    assert "leaf_swarm_row_0_members" in spec["datasets"]


def test_a_missing_variant_is_an_error_not_a_stand_in() -> None:
    payload = _four_payload()
    del payload["panels"]["threshold=reflected_fdr=by_q=0.05"]
    with pytest.raises(ValueError, match="lacks the"):
        build_leaf_swarm_panel_figure(payload, _four_settings(), "candidate_fdr_leaf_swarm")


def test_the_new_swarm_figures_are_opt_in_and_read_the_four_graph_input() -> None:
    tables = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))["figures"]
    for figure_id in FDR_PANEL_IDS:
        spec = figures.get_spec(figure_id)
        assert spec.default is False
        assert spec.inputs == frozenset({"four_graph_comparisons"})
        assert spec.variants == frozenset({"vanilla"})
        assert figure_id in FOUR_GRAPH_FIGURE_IDS
        assert not [key for key in tables[figure_id] if key.endswith("_color")]


def test_leaves_follow_the_tree_order_not_the_birth_rank() -> None:
    tables = _tables(_landscape())
    ordered = tables.leaves.sort("tree_order")
    # The tree draws B, C, D, A from the left (NODE_X); birth rank is A, B, C, D.
    assert ordered["peak_sequence"].to_list() == [
        "CCCCCC",
        "GGGGGG",
        "TCTCTC",
        "AAAAAA",
    ]
    assert tables.leaves.sort("rank")["peak_sequence"].to_list() == [
        "AAAAAA",
        "CCCCCC",
        "GGGGGG",
        "TCTCTC",
    ]
    # Every member carries its leaf's order.
    for row in tables.members.iter_rows(named=True):
        leaf = tables.leaves.filter(pl.col("peak_sequence") == row["leaf_peak_sequence"])
        assert row["tree_order"] == leaf["tree_order"][0]


def test_the_swarm_titles_its_x_axis_leaf_order_on_tree() -> None:
    spec = figures.render_payload(
        figures.get_spec("published_leaf_swarm"), _payload(), _settings(), "vanilla"
    ).to_dict()
    last = spec["vconcat"][-1]
    x_titles = [
        layer["encoding"]["x"]["axis"]["title"]
        for layer in last["layer"]
        if layer["encoding"].get("x", {}).get("axis", {}).get("title")
    ]
    assert x_titles and set(x_titles) == {"Leaf order on tree"}
    assert "Leaves in the order they appear on the tree" in spec["title"]["subtitle"][1]
    import tomllib

    shipped = tomllib.loads((ROOT / "config" / "figures.toml").read_text())["figures"]
    for name, table in shipped.items():
        if "rank_axis_title" in table:
            assert table["rank_axis_title"] == "Leaf order on tree", name


def test_stop_codon_members_are_crimson_and_drawn_over_the_plain_grey_ones() -> None:
    spec = figures.render_payload(
        figures.get_spec("published_leaf_swarm"), _payload(), _settings(), "vanilla"
    ).to_dict()
    rows = [
        row
        for index in (0, 1)
        for row in spec["datasets"][f"leaf_swarm_row_{index}_members"]
    ]
    crimson = [row for row in rows if row["dot_color"] == palette.STOP_CODON_CRIMSON]
    # The stop-codon peak B, the stop-codon member GGGGGA (the data's STOP_MEMBER).
    assert len(crimson) == 2 and all(row["highlighted"] for row in crimson)
    assert all(
        not row["highlighted"] for row in rows if row["dot_color"] == palette.PEAK_GREY
    )
