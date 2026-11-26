"""Scientific construction and sweep checks for the conceptual Figure 1 data."""

from __future__ import annotations

import numpy as np
import pytest

from dubanevics2026topological.analysis.manifold_graph_tree import (
    _assert_vertical_trunk,
    build_manifold_graph_tree,
    scalar_value,
)


@pytest.fixture(scope="module")
def construction():
    return build_manifold_graph_tree()


def test_construction_preserves_lattice_identities_and_public_tree_layout(construction):
    assert construction.sample_table.height == 36
    assert construction.edge_table.height == 60
    assert construction.grid_table.height == 121**2
    assert construction.sample_table[0, "sample_id"] == "sample-r1-c1"
    assert construction.sample_table[35, "sample_id"] == "sample-r6-c6"
    assert construction.critical_ids == {
        "left_peak": "sample-r2-c2",
        "saddle": "sample-r3-c4",
        "right_peak": "sample-r5-c5",
        "global_minimum": "sample-r1-c6",
    }
    assert set(construction.layout.node_table["vertex_id"]) == set(
        construction.sample_table["sample_id"]
    )
    assert construction.layout.node_table.height == 36
    assert construction.tree.kind == "split"


def test_public_tree_layout_keeps_root_to_saddle_trunk_vertical(construction):
    _assert_vertical_trunk(
        construction.layout,
        construction.critical_ids["global_minimum"],
        construction.critical_ids["saddle"],
    )


def _critical_x(construction):
    nodes = construction.layout.node_table
    return {
        name: float(nodes.filter(nodes["vertex_id"] == sample_id)["x"][0])
        for name, sample_id in construction.critical_ids.items()
    }


def test_public_tree_layout_puts_the_highest_peak_first(construction):
    # The default horizontal_layout is "subtree_extremum": the child whose subtree reaches the
    # highest peak goes first, so the global maximum (the right peak of the
    # landscape) is the leftmost leaf, ahead of the lower left peak.
    x = _critical_x(construction)
    assert x["right_peak"] < x["left_peak"]
    assert x["right_peak"] == pytest.approx(0.0)
    assert x["left_peak"] == pytest.approx(1.0)


def test_public_tree_layout_follows_the_horizontal_layout_argument():
    # The caller supplies the order: Topographer's default ("leaf_order") puts the
    # lower peak first.
    x = _critical_x(build_manifold_graph_tree(horizontal_layout="leaf_order"))
    assert x["left_peak"] < x["right_peak"]


def test_critical_points_and_normalized_sample_values_match_source(construction):
    expected = {
        "left_peak": ([0.2513839463525055, 0.2513839463525055], 0.7207588477951963),
        "saddle": ([0.48065800468765063, 0.48065800468765063], 0.3525253669629021),
        "right_peak": ([0.7492936546697476, 0.7492936546697476], 1.0),
        "global_minimum": ([1.0, 0.0], 0.0),
    }
    for name, (coordinate, scalar) in expected.items():
        sample_id = construction.critical_ids[name]
        row = construction.sample_table.filter(
            construction.sample_table["sample_id"] == sample_id
        ).row(0, named=True)
        assert [row["x"], row["y"]] == pytest.approx(coordinate, abs=1e-12)
        assert row["scalar"] == pytest.approx(scalar, abs=1e-12)
    assert construction.sample_table["scalar"].sum() == pytest.approx(
        12.648516172086655, abs=1e-12
    )
    assert construction.sample_table.filter(
        construction.sample_table["critical_name"].is_not_null()
    )["event"].to_list() == [
        "minimum",
        "maximum",
        "split_saddle",
        "maximum",
    ]


def test_five_sweep_checkpoints_have_expected_component_counts(construction):
    assert construction.thresholds == pytest.approx(
        (1.0, 0.99, 0.70, 0.3525253669629021, 0.0), abs=1e-12
    )
    states = [construction.sweep_state(value) for value in construction.thresholds]
    assert [state.component_count for state in states] == [0, 1, 2, 1, 1]
    assert len(states[0].active_sample_ids) == 0
    assert len(states[1].active_sample_ids) == 1
    assert len(states[2].active_sample_ids) > 2
    assert len(states[3].active_sample_ids) > 3
    assert len(states[4].active_sample_ids) == 36
    assert states[0].active_sample_ids == ()
    assert states[1].active_sample_ids == (construction.critical_ids["right_peak"],)
    assert {
        construction.critical_ids["left_peak"],
        construction.critical_ids["right_peak"],
    }.issubset(states[2].active_sample_ids)
    assert {
        construction.critical_ids["left_peak"],
        construction.critical_ids["right_peak"],
        construction.critical_ids["saddle"],
    }.issubset(states[3].active_sample_ids)
    assert set(states[4].active_sample_ids) == set(
        construction.sample_table["sample_id"]
    )


@pytest.mark.parametrize(
    ("critical_name", "is_active_at_equality"),
    [("left_peak", True), ("saddle", True), ("global_minimum", True)],
)
def test_critical_sample_activation_is_inclusive_at_scalar_event(
    construction, critical_name, is_active_at_equality
):
    sample_id = construction.critical_ids[critical_name]
    scalar = construction.sample_table.filter(
        construction.sample_table["sample_id"] == sample_id
    )["scalar"][0]
    epsilon = 1e-8
    assert sample_id not in construction.sweep_state(scalar + epsilon).active_sample_ids
    assert (
        sample_id in construction.sweep_state(scalar).active_sample_ids
    ) is is_active_at_equality
    assert sample_id in construction.sweep_state(scalar - epsilon).active_sample_ids


def test_opening_threshold_is_sentinel_and_maximum_enters_below_one(construction):
    right_peak_id = construction.critical_ids["right_peak"]
    assert right_peak_id not in construction.sweep_state(1.0).active_sample_ids
    assert right_peak_id in construction.sweep_state(1.0 - 1e-8).active_sample_ids
    with pytest.raises(ValueError, match="finite"):
        construction.sweep_state(float("nan"))


def test_scalar_evaluator_uses_surface_normalization(construction):
    center_value = scalar_value(0.7492936546697476, 0.7492936546697476)
    assert center_value == pytest.approx(1.0)
    sample = construction.sample_table.filter(
        construction.sample_table["sample_id"] == construction.critical_ids["saddle"]
    ).row(0, named=True)
    assert scalar_value(sample["x"], sample["y"]) == pytest.approx(sample["scalar"])
    assert np.isfinite(construction.grid_table["scalar"].to_numpy()).all()


def test_persistence_events_follow_the_four_stage_thresholds(construction):
    scalars = dict(
        zip(
            construction.sample_table["critical_name"],
            construction.sample_table["scalar"],
            strict=True,
        )
    )
    first, second, merged, complete = construction.thresholds[1:]

    one = construction.persistence_events(first)
    assert [(e.peak_name, e.death, e.alive) for e in one] == [
        ("right_peak", None, True)
    ]

    two = construction.persistence_events(second)
    assert [(e.peak_name, e.death, e.alive) for e in two] == [
        ("right_peak", None, True),
        ("left_peak", None, True),
    ]

    after_merge = construction.persistence_events(merged)
    assert [(e.peak_name, e.death, e.alive) for e in after_merge] == [
        ("right_peak", None, True),
        ("left_peak", pytest.approx(scalars["saddle"]), False),
    ]

    final = construction.persistence_events(complete)
    assert [(e.peak_name, e.alive) for e in final] == [
        ("right_peak", False),
        ("left_peak", False),
    ]
    assert final[0].death == pytest.approx(scalars["global_minimum"])
    assert final[1].death == pytest.approx(scalars["saddle"])
    assert [e.birth for e in final] == sorted((e.birth for e in final), reverse=True)
    assert [e.pair_kind for e in final] == ["extended", "ordinary"]
    ids = construction.critical_ids
    assert [e.birth_sample_id for e in final] == [ids["right_peak"], ids["left_peak"]]
    assert [e.death_sample_id for e in final] == [
        ids["global_minimum"],
        ids["saddle"],
    ]
    assert after_merge[0].death_sample_id is None


def test_persistence_death_is_inclusive_at_the_pair_scalar(construction):
    saddle = float(
        construction.sample_table.filter(
            construction.sample_table["critical_name"] == "saddle"
        ).item(0, "scalar")
    )
    at_saddle = {e.peak_name: e for e in construction.persistence_events(saddle)}
    assert not at_saddle["left_peak"].alive
    assert at_saddle["left_peak"].pair_kind == "ordinary"
    just_above = {
        e.peak_name: e for e in construction.persistence_events(saddle + 1e-9)
    }
    assert just_above["left_peak"].alive
    assert just_above["left_peak"].death is None

    at_zero = {e.peak_name: e for e in construction.persistence_events(0.0)}
    assert not at_zero["right_peak"].alive
    assert at_zero["right_peak"].pair_kind == "extended"
    almost = {e.peak_name: e for e in construction.persistence_events(1e-9)}
    assert almost["right_peak"].alive
    assert almost["right_peak"].pair_kind == "extended"


def test_persistence_events_are_empty_before_the_sweep(construction):
    assert construction.persistence_events(construction.thresholds[0]) == ()
