import marimo as mo
import polars as pl
import pytest

from dubanevics2026topological.figures.candidate_edge_evidence import (
    build_selected_edge_detail_figure,
    select_edges,
    selected_edge_rows,
)
from dubanevics2026topological.figures.registry import get_spec, render_payload


def _edges():
    return pl.DataFrame(
        {
            "edge_id": ["a|b", "b|c", "c|d"],
            "dna_left": ["a", "b", "c"],
            "dna_right": ["b", "c", "d"],
            "fitness_left": [0.1, 0.3, -0.2],
            "fitness_right": [0.3, -0.2, 0.4],
            "fitness_lower": [0.1, -0.2, -0.2],
            "fitness_upper": [0.3, 0.3, 0.4],
            "fitness_delta": [0.2, -0.5, 0.6],
            "absolute_fitness_delta": [0.2, 0.5, 0.6],
            "contrast_id": ["x", "y", "z"],
            "reported_dna_from": ["b", "b", "c"],
            "reported_dna_to": ["a", "c", "d"],
            "reported_delta": [-0.2, 0.5, 0.6],
            "pairwise_standard_error": [0.1, 0.2, 0.3],
            "test_statistic": [2.0, -2.5, 2.0],
            "p_value": [0.05, 0.02, 0.05],
            "q_value": [0.05, 0.2, 1.0],
        }
    )


def test_select_edges_uses_inclusive_band_and_strict_alpha():
    selected = select_edges(_edges(), 0.05, 1.0, 0.05)
    assert selected["edge_id"].to_list() == ["a|b", "b|c", "c|d"]
    assert selected["qualifies"].to_list() == [False, True, True]
    assert selected["reported_raw_sign_disagreement"].to_list() == [False, True, False]
    narrowed = select_edges(_edges(), 0.0, 1.0, 0.0, edge_ids=["c|d"])
    assert narrowed["edge_id"].to_list() == ["c|d"]


@pytest.mark.parametrize(
    ("q_lower", "q_upper", "alpha"),
    [(-0.1, 1.0, 0.0), (0.0, 1.1, 0.0), (0.0, 1.0, float("nan")), (0.8, 0.2, 0.0)],
)
def test_select_edges_rejects_invalid_controls(q_lower, q_upper, alpha):
    with pytest.raises(ValueError):
        select_edges(_edges(), q_lower, q_upper, alpha)


def test_select_edges_retains_zero_and_below_alpha_with_qualification_flag():
    edges = _edges().with_columns(pl.Series("q_value", [0.0, 0.1, 1.0]))
    selected = select_edges(edges, 0.0, 1.0, 0.2)
    assert selected["q_value"].to_list() == [0.0, 0.1, 1.0]
    assert selected["qualifies"].to_list() == [False, False, True]


def test_select_edges_preserves_cached_sign_disagreement_flag():
    edges = _edges().with_columns(
        pl.Series("reported_raw_sign_disagreement", [True, False, True])
    )
    selected = select_edges(edges, 0.0, 1.0, 0.0)
    assert selected["reported_raw_sign_disagreement"].to_list() == [
        True,
        False,
        True,
    ]


def test_selected_marimo_rows_build_both_linked_detail_views():
    edges = _edges().with_columns(pl.Series("q_value", [0.0, 0.1, 1.0]))
    eligible_rows = select_edges(edges, 0.0, 1.0, 0.2)
    table = mo.ui.table(eligible_rows, selection="multi", initial_selection=[0])
    selected = selected_edge_rows(table.value)
    assert selected["edge_id"].to_list() == ["a|b"]
    assert selected["q_value"].to_list() == [0.0]
    assert selected["qualifies"].to_list() == [False]

    spec = build_selected_edge_detail_figure(selected).to_dict()
    assert len(spec["hconcat"]) == 2
    assert spec["hconcat"][0]["encoding"]["x"]["field"] == "q_value"
    assert spec["hconcat"][0]["encoding"]["y"]["field"] == ("absolute_fitness_delta")
    assert spec["hconcat"][1]["encoding"]["x"]["field"] == "q_value"


def test_figure_builds_compact_aggregate_chart():
    settings = {
        "figures": {
            "figures": {
                "candidate_edge_evidence": {
                    "panel_size": 320,
                    "font_size": 13,
                    "q_bins": 50,
                    "effect_bins": 60,
                    "fitness_bins": 60,
                }
            }
        }
    }
    chart = render_payload(
        get_spec("candidate_edge_evidence"),
        {"edges": _edges()},
        settings,
        "vanilla",
    )
    spec = chart.to_dict()
    assert spec["vconcat"][0]["hconcat"][0]["mark"]["orient"] == "vertical"
    assert spec["vconcat"][0]["hconcat"][0]["data"]["values"][-1]["q_end"] == 1.0
    assert len(spec["vconcat"][0]["hconcat"][0]["data"]["values"]) == 50
    assert len(spec["vconcat"][0]["hconcat"][1]["data"]["values"]) == 51
    coverage = spec["vconcat"][1]["hconcat"][1]["data"]["values"]
    assert coverage
    assert all("coverage" in row and "count" not in row for row in coverage)
