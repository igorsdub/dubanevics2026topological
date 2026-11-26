from __future__ import annotations

import csv
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis.graph_summary import (
    SUMMARY_COLUMNS,
    LandscapeSummaryInput,
    landscape_summary,
    landscape_summary_column,
)
from dubanevics2026topological.analysis.leaves import Leaves
from dubanevics2026topological.tables.graph_summary import (
    SUMMARY_DISPLAY,
    export_landscape_summary,
)


class Graph:
    def __init__(self, rows):
        self.rows = rows

    def node_indices(self):
        return range(len(self.rows))

    def __getitem__(self, index):
        return self.rows[index]


FULL_FITNESS = {"a": 0.5, "b": 0.0, "c": -0.5}


def item(*, singleton=False, band=False, by=True):
    rows = [
        {"dna_sequence": "a", "aa_sequence": "K*", "fitness": 0.5},
        {"dna_sequence": "b", "aa_sequence": "**", "fitness": 0.0},
        {"dna_sequence": "c", "aa_sequence": "KK", "fitness": -0.5},
    ][: 1 if singleton else 3]
    leaf_rows = (
        [{"is_root_endpoint": True, "is_synonymous": None}]
        if singleton
        else [
            {"is_root_endpoint": False, "is_synonymous": not band},
            {"is_root_endpoint": True, "is_synonymous": None},
        ]
    )
    leaves = Leaves(
        pl.DataFrame(leaf_rows),
        pl.DataFrame({"member_dna_sequence": ["a"]}),
        {
            "tree_state": "band" if band else "nominal",
            "tree_member_count": len(rows),
            "leaf_count": len(leaf_rows),
            "eligible_leaf_denominator": 0 if singleton else 1,
            "ineligible_endpoint_count": 1,
        },
    )
    return LandscapeSummaryInput(
        SimpleNamespace(
            graph=Graph(rows), scalar_attr="fitness", edge_count=len(rows) - 1
        ),
        leaves,
        0.0,
        {"a": "functional", "b": "uncertain", "c": "non_functional"} if by else None,
        1,
        len(rows),
    )


def test_counts_stop_composition_and_leaf_eligibility():
    result = landscape_summary_column(item(), full_fitness=FULL_FITNESS)
    assert result["Sequences with exactly 1 stops"] == 1
    assert result["Sequences with exactly 2 stops"] == 1
    assert result["Sequences with exactly 3 stops"] == 0
    assert result["Selected graph uncertain sequences (%)"] == pytest.approx(100 / 3)
    assert result["Synonymous leaves"] == 1
    assert result["Eligible leaves"] == 1
    assert result["Ineligible root-only leaves"] == 1
    assert result["Synonymous leaves (%)"] == 100


def test_single_sequence_leaf_is_ineligible():
    result = landscape_summary_column(item(singleton=True), full_fitness=FULL_FITNESS)
    assert result["Peak count"] == 1
    assert result["Eligible leaves"] == 0
    assert result["Synonymous leaves (%)"] is None


def test_seven_columns_mixture_and_exports(tmp_path):
    inputs = {
        name: item(band="band-simplified" in name, by=index >= 3)
        for index, name in enumerate(SUMMARY_COLUMNS)
    }
    result = landscape_summary(inputs, full_fitness=FULL_FITNESS)
    assert tuple(result) == SUMMARY_COLUMNS
    assert result[SUMMARY_COLUMNS[3]]["Synonymous leaves"] == 1
    assert result[SUMMARY_COLUMNS[4]]["Synonymous leaves"] == 0
    assert (
        result[SUMMARY_COLUMNS[1]]["Full library uncertain sequences"]
        == "not applicable"
    )
    assert result[SUMMARY_COLUMNS[1]]["Full library functional sequences"] == 2
    assert (
        len(landscape_summary(inputs, full_fitness=FULL_FITNESS, mixture=item())) == 8
    )
    paths = export_landscape_summary(result, "toy", tmp_path)
    with paths[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # The identity column names stay in the saved values; a reader sees the state labels.
    assert tuple(rows[0]) == ("Metric", *SUMMARY_DISPLAY.values())
    assert tuple(SUMMARY_DISPLAY) == SUMMARY_COLUMNS
    assert "eligible leaves" in paths[1].read_text()
    assert "Full-library" in paths[2].read_text()


def test_export_formats_percentages_and_requires_exactly_seven(tmp_path):
    inputs = {
        name: item(band="band-simplified" in name, by=index >= 3)
        for index, name in enumerate(SUMMARY_COLUMNS)
    }
    result = landscape_summary(inputs, full_fitness=FULL_FITNESS)
    paths = export_landscape_summary(result, "toy", tmp_path)
    assert "33.333" in paths[0].read_text()
    assert "33.33%" in paths[1].read_text()
    assert "33.33%" in paths[2].read_text()
    mixed = landscape_summary(inputs, full_fitness=FULL_FITNESS, mixture=item())
    with pytest.raises(ValueError, match="exactly the seven"):
        export_landscape_summary(mixed, "toy", tmp_path)


def test_rejects_mismatched_graphs_and_incomplete_classifications():
    inputs = {
        name: item(band="band-simplified" in name, by=index >= 3)
        for index, name in enumerate(SUMMARY_COLUMNS)
    }
    inputs[SUMMARY_COLUMNS[4]] = item(singleton=True, band=True)
    with pytest.raises(ValueError, match="Graph identities differ"):
        landscape_summary(inputs, full_fitness=FULL_FITNESS)
    with pytest.raises(ValueError, match="[Cc]lassification|functional class"):
        landscape_summary_column(item(), full_fitness={**FULL_FITNESS, "d": 1.0})
