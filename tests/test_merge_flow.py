"""Merge-flow components, events and levels of synthetic merge trees."""

from __future__ import annotations

import polars as pl
import pytest

from dubanevics2026topological.analysis import describe
from dubanevics2026topological.analysis.merge_flow import MergeFlow
from dubanevics2026topological.analysis.ridges import RidgeTree

SETTINGS = {"min_functional_count_K": 1, "level_grid": "exact_breakpoints"}
NAMES = {"AAA": "K", "CCC": "L", "GGG": "M"}
VERTICES = {
    "AAA": 9.0,
    "AAC": 8.0,
    "CCC": 7.0,
    "CCA": 6.0,
    "GGG": 5.0,
    "GGA": 3.0,
    "TTT": 1.0,
}
CLASSES = {
    "AAA": "functional",
    "AAC": "functional",
    "CCC": "functional",
    "CCA": "non_functional",
    "GGG": "functional",
    "GGA": "functional",
    "TTT": "non_functional",
}


def _tree(vertices=VERTICES, members=None, edges=None, nodes=None) -> RidgeTree:
    """Y tree: AAA and CCC merge at GGG, which descends to the root TTT."""
    nodes = nodes or ["AAA", "CCC", "GGG", "TTT"]
    edges = edges or [("AAA", "GGG"), ("CCC", "GGG"), ("GGG", "TTT")]
    members = members or [("AAA", "AAC"), ("CCC", "CCA"), ("GGG", "GGA")]
    return RidgeTree(
        pl.DataFrame(
            {
                "tree_node_identity": nodes,
                "scalar": [vertices[n] for n in nodes],
            }
        ),
        pl.DataFrame(
            {
                "source_tree_node_identity": [e[0] for e in edges],
                "target_tree_node_identity": [e[1] for e in edges],
            }
        ),
        pl.DataFrame(
            {
                "source_tree_node_identity": [m[0] for m in members],
                "target_tree_node_identity": [dict(edges)[m[0]] for m in members],
                "dna_sequence": [m[1] for m in members],
            },
            schema={
                "source_tree_node_identity": pl.String,
                "target_tree_node_identity": pl.String,
                "dna_sequence": pl.String,
            },
        ),
    )


def _pairs(*pairs) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "birth_sequence": [p[0] for p in pairs],
            "death_sequence": [p[1] for p in pairs],
        },
        schema={"birth_sequence": pl.String, "death_sequence": pl.String},
    )


Y_PAIRS = _pairs(("CCC", "GGG"), ("AAA", "TTT"))


def _build(**overrides) -> MergeFlow:
    arguments = {
        "merge_flow_tree": _tree(),
        "merge_flow_state": "published",
        "merge_flow_vertices": VERTICES,
        "merge_flow_classes": CLASSES,
        "merge_flow_names": NAMES,
        "merge_flow_pairs": Y_PAIRS,
        "merge_flow_settings": SETTINGS,
    }
    arguments.update(overrides)
    return describe(products={"merge_flow"}, **arguments).merge_flow


def _row(frame: pl.DataFrame, **match) -> dict:
    expr = pl.lit(True)
    for key, value in match.items():
        expr = expr & (pl.col(key) == value)
    rows = frame.filter(expr).to_dicts()
    assert len(rows) == 1
    return rows[0]


def test_y_tree_arcs_births_deaths_and_balanced_merge():
    flow = _build()
    components = flow.components
    assert components.height == 4
    a = _row(components, component_id="AAA@AAA")
    assert (a["birth_fitness"], a["death_fitness"]) == (9.0, 5.0)
    assert a["death_dna_sequence"] == "GGG"
    assert a["parent_component_id"] == "AAA@GGG"
    assert a["summit_aa_sequence"] == "K"
    assert (a["functional_count_at_death"], a["total_count_at_death"]) == (2, 2)
    c = _row(components, component_id="CCC@CCC")
    assert (c["functional_count_at_death"], c["total_count_at_death"]) == (1, 2)
    saddle = _row(components, component_id="AAA@GGG")
    assert saddle["is_leaf"] is False
    assert (saddle["birth_fitness"], saddle["death_fitness"]) == (5.0, 1.0)
    assert saddle["functional_count_at_death"] == 5
    root = _row(components, component_id="AAA@TTT")
    assert root["death_dna_sequence"] is None
    assert root["parent_component_id"] is None
    assert root["total_count_at_death"] == 7
    event = flow.events.to_dicts()[0]
    assert event["saddle_dna_sequence"] == "GGG"
    assert event["elder_component_id"] == "AAA@AAA"
    assert event["elder_summit_dna_sequence"] == "AAA"
    assert event["younger_component_ids"] == ["CCC@CCC"]
    assert event["child_count"] == 2
    assert event["balanced_merge"] == 1
    assert event["smaller_child_total_count"] == 2
    assert event["largest_child_functional_count"] == 2
    assert event["saddle_drop"] == pytest.approx(2.0)
    assert event["persistence_pair_agrees"] is True
    assert flow.sidecar["elder_check"] == {"pairs": 2, "agree": 2}


def _equal_summits():
    vertices = {**VERTICES, "CCC": 9.0}
    return vertices, {**CLASSES}


def test_equal_summits_smaller_vertex_index_wins_and_agrees_with_pairs():
    vertices, classes = _equal_summits()
    flow = _build(
        merge_flow_tree=_tree(vertices),
        merge_flow_vertices=vertices,
        merge_flow_classes=classes,
        merge_flow_order={
            dna: i for i, dna in enumerate(sorted(vertices, reverse=True))
        },
        merge_flow_pairs=_pairs(("AAA", "GGG"), ("CCC", "TTT")),
    )
    event = flow.events.to_dicts()[0]
    assert event["elder_summit_dna_sequence"] == "CCC"
    assert event["younger_component_ids"] == ["AAA@AAA"]
    assert flow.sidecar["tie_break"] == "vertex_index"


def test_disagreeing_persistence_pairs_raise():
    vertices, classes = _equal_summits()
    with pytest.raises(ValueError, match="disagree"):
        _build(
            merge_flow_tree=_tree(vertices),
            merge_flow_vertices=vertices,
            merge_flow_classes=classes,
            merge_flow_order={
                dna: i for i, dna in enumerate(sorted(vertices, reverse=True))
            },
            merge_flow_pairs=_pairs(("CCC", "GGG"), ("AAA", "TTT")),
        )


def test_uncertain_counts_as_non_functional_and_the_raw_class_is_kept():
    classes = {**CLASSES, "AAC": "uncertain", "CCC": "uncertain"}
    flow = _build(merge_flow_classes=classes)
    a = _row(flow.components, component_id="AAA@AAA")
    assert (a["functional_count_at_death"], a["total_count_at_death"]) == (1, 2)
    c = _row(flow.components, component_id="CCC@CCC")
    assert c["functional_count_at_death"] == 0
    assert flow.sidecar["uncertain_policy"] == "merged_with_non_functional"
    assert flow.sidecar["class_counts"]["uncertain"] == 2


def test_full_state_widths_use_every_sequence():
    flow = _build(
        merge_flow_state="full",
        merge_flow_classes={dna: None for dna in VERTICES},
    )
    assert set(flow.components["width_basis"]) == {"all"}
    c = _row(flow.components, component_id="CCC@CCC")
    assert c["functional_count_at_death"] == c["total_count_at_death"] == 2
    assert flow.sidecar["width_basis"] == "all"


def test_nominal_k_pruning_keeps_the_row_and_adds_a_stub():
    flow = _build(merge_flow_settings={**SETTINGS, "min_functional_count_K": 2})
    c = _row(flow.components, component_id="CCC@CCC")
    assert c["pruned"] is True
    assert c["prune_reason"] == "below_K"
    a = _row(flow.components, component_id="AAA@AAA")
    assert a["pruned"] is False
    assert a["prune_reason"] is None
    event = flow.events.to_dicts()[0]
    assert event["stub_functional_count"] == 1
    assert event["stub_component_count"] == 1
    assert "CCC@CCC" not in set(flow.levels["component_id"])
    assert event["balanced_merge"] == 1  # counts never change with pruning


def test_band_state_marks_nominal_leaves_missing_from_the_band_tree():
    band = _tree(
        nodes=["AAA", "TTT"],
        edges=[("AAA", "TTT")],
        members=[("AAA", "AAC")],
    )
    flow = _build(
        merge_flow_state="published_by_band",
        merge_flow_tree=band,
        merge_flow_nominal=_tree(),
    )
    c = _row(flow.components, component_id="CCC@CCC")
    assert (c["pruned"], c["prune_reason"]) == (True, "band_absent")
    assert not _row(flow.components, component_id="AAA@AAA")["pruned"]
    assert flow.sidecar["band_absent_summits"] == ["CCC"]
    assert flow.sidecar["tree_state"] == "band"
    assert flow.events["stub_functional_count"].to_list() == [1]


def _subtree(component_id: str) -> set[str]:
    return {
        "AAA@AAA": {"AAA", "AAC"},
        "CCC@CCC": {"CCC", "CCA"},
        "AAA@GGG": {"AAA", "AAC", "CCC", "CCA", "GGG", "GGA"},
        "AAA@TTT": set(VERTICES),
    }[component_id]


def test_levels_equal_a_brute_force_superlevel_count_and_are_monotone():
    flow = _build()
    levels = flow.levels
    assert set(levels["level_fitness"]) == set(VERTICES.values())
    for (component_id,), group in levels.group_by("component_id"):
        group = group.sort("level_fitness", descending=True)
        members = _subtree(component_id)
        for row in group.to_dicts():
            level = row["level_fitness"]
            above = {d for d in members if VERTICES[d] >= level}
            assert row["total_count"] == len(above)
            assert row["functional_count"] == sum(
                CLASSES[d] == "functional" for d in above
            )
        for column in ("functional_count", "total_count"):
            counts = group[column].to_list()
            assert counts == sorted(counts)
        birth = _row(flow.components, component_id=component_id)["birth_fitness"]
        assert group["level_fitness"].max() == birth


def test_no_index_columns_and_every_identifier_is_dna_from_the_vertex_set():
    flow = _build()
    for frame in (flow.components, flow.events, flow.levels):
        assert not [c for c in frame.columns if c.endswith("index")]
    dna_columns = [
        "summit_dna_sequence",
        "birth_dna_sequence",
        "death_dna_sequence",
    ]
    for column in dna_columns:
        assert set(flow.components[column].drop_nulls()) <= set(VERTICES)
    identifiers = set(flow.components["component_id"]) | set(
        flow.levels["component_id"]
    )
    for identifier in identifiers:
        assert set(identifier.split("@")) <= set(VERTICES)
    assert set(flow.events["saddle_dna_sequence"]) <= set(VERTICES)


def test_settings_require_the_functional_count_threshold():
    with pytest.raises(ValueError, match="min_functional_count_K"):
        _build(merge_flow_settings={"level_grid": "exact_breakpoints"})


def test_save_and_from_artifacts_round_trip(tmp_path):
    flow = _build()
    flow.save(tmp_path)
    back = MergeFlow.from_artifacts({p.name: p for p in tmp_path.iterdir()})
    assert back.components.equals(flow.components)
    assert back.events.equals(flow.events)
    assert back.levels.equals(flow.levels)
    assert back.sidecar == flow.sidecar


def test_root_members_are_counted_in_the_root_component():
    vertices = {**VERTICES, "TTA": 0.5, "TTC": 0.25}
    classes = {**CLASSES, "TTA": "functional", "TTC": "non_functional"}
    base = _tree(vertices)
    tree = RidgeTree(
        base.nodes,
        base.edges,
        base.branch_members,
        pl.DataFrame({"dna_sequence": ["TTA", "TTC"]}),
    )
    flow = _build(
        merge_flow_tree=tree,
        merge_flow_vertices=vertices,
        merge_flow_classes=classes,
    )
    root = _row(flow.components, component_id="AAA@TTT")
    assert root["total_count_at_death"] == len(vertices)
    assert root["functional_count_at_death"] == 6
    assert flow.sidecar["swept_member_count"] == flow.sidecar["graph_vertex_count"]
    lowest = flow.levels.filter(
        (pl.col("component_id") == "AAA@TTT") & (pl.col("level_fitness") == 0.25)
    )
    assert lowest["total_count"].to_list() == [len(vertices)]
