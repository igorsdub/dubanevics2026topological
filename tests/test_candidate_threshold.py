"""The Candidate functional threshold is computed, rounded and what selection receives."""

from __future__ import annotations

import json

import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.landscape.selected_landscape import (
    prepare_clamped_landscape,
)
from dubanevics2026topological.threshold.functional_threshold import (
    CandidateThreshold,
    compute_candidate_functional_threshold,
    resolve_candidate_threshold,
    validate_candidate_declaration,
)

DECLARATION = {"rule": "median_reflected", "decimals": 6}


def _network(ids: list[str], values: list[float], edges) -> ScalarNetwork:
    graph = Graph.from_tables(
        pl.DataFrame({"dna_sequence": ids, "fitness": values}),
        pl.DataFrame(
            edges, schema={"source": pl.String, "target": pl.String}, orient="row"
        ),
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness",),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def test_hand_checked_fixture() -> None:
    # g = -1.0, m = -0.5, h = 0.5, T = m + h = 0.0.
    computed = compute_candidate_functional_threshold(
        [-1.0, -0.6, -0.5, -0.2, 0.3], DECLARATION
    )
    assert computed == CandidateThreshold(
        rule="median_reflected",
        decimals=6,
        median=-0.5,
        global_minimum=-1.0,
        median_gap=0.5,
        unrounded=0.0,
        threshold=0.0,
    )


@pytest.mark.parametrize(
    ("decimals", "expected"), [(6, 0.246913), (3, 0.247), (1, 0.2), (0, 0.0)]
)
def test_threshold_is_rounded_to_declared_decimals(decimals, expected) -> None:
    computed = compute_candidate_functional_threshold(
        [0.0, 0.1234567, 0.3], {**DECLARATION, "decimals": decimals}
    )
    assert computed.unrounded == pytest.approx(0.2469134)
    assert computed.threshold == expected


def test_selection_receives_the_rounded_threshold() -> None:
    # m = 0.1232 and g = 0, so the unrounded T is 0.2464 and the 3-decimal T is
    # 0.246: vertex c (0.2462) is functional only under the rounded value.
    network = _network(["a", "b", "c"], [0.0, 0.1232, 0.2462], [("a", "b"), ("b", "c")])
    computed = resolve_candidate_threshold(
        network,
        {
            "analysis": {
                "clamped": {
                    "candidate_functional_threshold": {**DECLARATION, "decimals": 3}
                }
            }
        },
    )
    assert computed.unrounded == pytest.approx(0.2464)
    assert computed.threshold == 0.246
    prepared = prepare_clamped_landscape(network, computed.threshold)
    assert prepared.threshold == 0.246
    identities = {
        prepared.network.graph[i]["dna_sequence"]
        for i in prepared.network.graph.node_indices()
    }
    assert "c" in identities


def test_record_holds_the_registry_inputs() -> None:
    computed = compute_candidate_functional_threshold(
        [0.0, 0.1234567, 0.3], DECLARATION
    )
    record = json.loads(json.dumps(computed.to_json()))
    assert record == {
        "rule": "median_reflected",
        "decimals": 6,
        "median": 0.1234567,
        "global_minimum": 0.0,
        "median_gap": 0.1234567,
        "unrounded_threshold": pytest.approx(0.2469134),
        "rounded_threshold": 0.246913,
    }


@pytest.mark.parametrize(
    "declaration",
    [
        {"rule": "saddle_reflected", "decimals": 6},
        {"rule": "median_reflected", "decimals": -1},
        {"rule": "median_reflected", "decimals": 2.5},
        {"rule": "median_reflected", "decimals": True},
        {"rule": "median_reflected"},
        {"rule": "median_reflected", "decimals": 6, "value": -0.28},
    ],
)
def test_invalid_declarations_raise(declaration) -> None:
    with pytest.raises(ValueError):
        validate_candidate_declaration(declaration)


def test_unknown_rule_error_names_the_rule() -> None:
    with pytest.raises(ValueError, match="saddle_reflected"):
        compute_candidate_functional_threshold(
            [0.0, 1.0], {"rule": "saddle_reflected", "decimals": 6}
        )


def test_checked_in_config_declares_the_rule_not_a_value() -> None:
    clamped = load_settings()["analysis"]["clamped"]
    declaration = clamped["candidate_functional_threshold"]
    assert declaration["rule"] == "fdr_fixed_cutoff"
    assert declaration["decimals"] == 6
    assert declaration["cutoff"] == "median_reflected"
    assert "value" not in declaration
    assert clamped["published_functional_threshold"] == -0.507774


def test_a_typed_value_is_not_a_declaration() -> None:
    with pytest.raises(TypeError, match="must be a table"):
        validate_candidate_declaration(-0.28)
