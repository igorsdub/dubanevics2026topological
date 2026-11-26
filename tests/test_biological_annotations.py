"""Biological endpoint evidence follows topology rather than persistence death."""

from types import SimpleNamespace

import polars as pl
import pytest
import rustworkx as rx

from dubanevics2026topological.analysis.biological_annotations import (
    _build_biological_annotations as build_biological_annotations,
)
from dubanevics2026topological.analysis.biological_annotations import (
    _export_biological_annotations as export_biological_annotations,
)


def _analysis(singleton=False):
    graph = rx.PyDiGraph()
    if singleton:
        graph.add_node(
            {
                "dna_sequence": "TAATAA",
                "aa_sequence": "**",
                "scalar": 0.0,
                "event": "maximum",
            }
        )
        branches = pl.DataFrame(
            schema={
                "source_tree_node_index": pl.Int64,
                "target_tree_node_index": pl.Int64,
                "branch_member_count": pl.Int64,
            }
        )
        pair_rows = [
            {
                "birth_tree_node_index": 0,
                "death_tree_node_index": 0,
                "birth_scalar": 0.0,
                "death_scalar": 0.0,
                "pair_kind": "extended",
            }
        ]
    else:
        for dna, aa, scalar, event in [
            ("AAAAAA", "KK", 3.0, "maximum"),
            ("AAGAAG", "KK", 1.0, "merge"),
            ("CCCAAA", "PK", 0.0, "minimum"),
            ("TAATAA", "**", 2.0, "maximum"),
            ("TAGTAG", "%%", 1.5, "maximum"),
        ]:
            graph.add_node(
                {
                    "dna_sequence": dna,
                    "aa_sequence": aa,
                    "scalar": scalar,
                    "event": event,
                }
            )
        edges = [(0, 1, 0), (3, 1, 0), (4, 1, 2), (1, 2, 1)]
        for upper, lower, _ in edges:
            graph.add_edge(upper, lower, {})
        branches = pl.DataFrame(
            [
                {
                    "source_tree_node_index": u,
                    "target_tree_node_index": l,
                    "branch_member_count": n,
                }
                for u, l, n in edges
            ]
        )
        pair_rows = [
            {
                "birth_tree_node_index": b,
                "death_tree_node_index": d,
                "birth_scalar": graph[b]["scalar"],
                "death_scalar": graph[d]["scalar"],
                "pair_kind": k,
            }
            for b, d, k in [(0, 2, "extended"), (3, 1, "ordinary"), (4, 1, "ordinary")]
        ]
    coordinates = pl.DataFrame(
        [
            {"tree_node_index": i, "x": float(i), "y": graph[i]["scalar"]}
            for i in graph.node_indices()
        ]
    )
    return SimpleNamespace(
        trees={
            "split": SimpleNamespace(
                tree=graph,
                branch_table=branches,
                persistence_table=pl.DataFrame(pair_rows),
                plot_data=SimpleNamespace(node_table=coordinates),
            )
        }
    )


def test_adjacent_saddle_equality_is_distinct_from_paired_death():
    result = build_biological_annotations(_analysis())
    global_pair = result.pairs.filter(pl.col("birth_sequence") == "AAAAAA").row(
        0, named=True
    )
    assert global_pair["adjacent_saddle_sequence"] == "AAGAAG"
    assert global_pair["death_sequence"] == "CCCAAA"
    assert global_pair["synonymous_endpoints"] is True
    assert global_pair["paired_synonymous_endpoints"] is False
    assert global_pair["branch_vertex_count"] == 1
    assert result.summary["singleton_peak_count"] == 2
    assert result.summary["multi_sequence_peak_count"] == 1
    assert result.summary["peak_denominator"] == 3
    assert result.summary["represented_vertex_count"] == 8
    assert not any(
        column.endswith("_index")
        for frame in (result.branches, result.pairs, result.critical_points)
        for column in frame.columns
    )


def test_multiple_stops_count_once_per_sequence_and_distinct_states_stay_distinct():
    result = build_biological_annotations(_analysis())
    assert result.summary["critical_stop_sequence_count"] == 2
    assert result.summary["stop_codon_peak_count"] == 2
    stopped = result.critical_points.filter(pl.col("has_stop_codon"))
    assert stopped["stop_codon_count"].to_list() == [2, 2]
    assert result.pairs.filter(pl.col("birth_has_stop_codon"))[
        "synonymous_endpoints"
    ].to_list() == [False, False]
    # Equal stop-position patterns with different encoded states remain unequal.
    analysis = _analysis()
    analysis.trees["split"].tree[1] = {
        "dna_sequence": "TGATGA",
        "aa_sequence": "^^",
        "scalar": 1.0,
        "event": "merge",
    }
    result = build_biological_annotations(analysis)
    assert result.summary["synonymous_endpoint_peak_count"] == 0


def test_one_vertex_has_no_saddle_or_branch_but_one_peak():
    result = build_biological_annotations(_analysis(singleton=True))
    assert result.branches.is_empty()
    assert result.pairs["adjacent_saddle_sequence"].to_list() == [None]
    assert result.pairs["synonymous_endpoints"].to_list() == [False]
    assert result.summary["adjacent_saddle_peak_denominator"] == 0
    assert result.summary["represented_vertex_count"] == 1
    assert result.summary["stop_codon_peak_count"] == 1


def test_export_round_trip_preserves_identities_and_counts(tmp_path):
    result = build_biological_annotations(_analysis())
    outputs = export_biological_annotations(result, tmp_path)
    assert len(outputs) == 7
    for name in ("branches", "pairs", "critical_points"):
        assert pl.read_parquet(
            tmp_path / f"biological_annotation_{name}.parquet"
        ).equals(getattr(result, name))
    assert (tmp_path / "biological_annotation_counts.json").is_file()


def test_single_peak_root_lower_endpoint_is_not_a_saddle():
    analysis = _analysis(singleton=True)
    tree = analysis.trees["split"]
    tree.tree.add_node(
        {
            "dna_sequence": "TAGTAG",
            "aa_sequence": "**",
            "scalar": -1.0,
            "event": "minimum",
        }
    )
    tree.tree.add_edge(0, 1, {})
    tree.branch_table = pl.DataFrame(
        {
            "source_tree_node_index": [0],
            "target_tree_node_index": [1],
            "branch_member_count": [0],
        }
    )
    tree.plot_data.node_table = pl.DataFrame(
        {"tree_node_index": [0, 1], "x": [0.0, 0.0], "y": [0.0, -1.0]}
    )
    tree.persistence_table = pl.DataFrame(
        {
            "birth_tree_node_index": [0],
            "death_tree_node_index": [1],
            "birth_scalar": [0.0],
            "death_scalar": [-1.0],
            "pair_kind": ["extended"],
        }
    )
    result = build_biological_annotations(analysis)
    assert result.pairs["adjacent_lower_sequence"].to_list() == ["TAGTAG"]
    assert result.pairs["adjacent_saddle_sequence"].to_list() == [None]
    assert result.pairs["synonymous_endpoints"].to_list() == [False]
    assert result.pairs["paired_synonymous_endpoints"].to_list() == [True]
    assert result.summary["singleton_stop_codon_peak_count"] == 1
    assert result.summary["singleton_peak_denominator"] == 1


def test_internal_saddle_cannot_replace_a_peak_in_pair_evidence():
    import pytest

    analysis = _analysis()
    rows = analysis.trees["split"].persistence_table.to_dicts()
    rows[0]["birth_tree_node_index"] = 1
    rows[0]["birth_scalar"] = 1.0
    analysis.trees["split"].persistence_table = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="identify every peak"):
        build_biological_annotations(analysis)


def test_duplicate_critical_sequence_identities_are_rejected():
    import pytest

    analysis = _analysis()
    payload = dict(analysis.trees["split"].tree[1])
    payload["dna_sequence"] = "AAAAAA"
    analysis.trees["split"].tree[1] = payload
    with pytest.raises(ValueError, match="identities must be unique"):
        build_biological_annotations(analysis)


def test_branch_vertex_count_is_read_as_is_and_member_count_gets_one_added():
    current = _analysis()
    current_tree = current.trees["split"]
    legacy = _analysis()
    legacy_tree = legacy.trees["split"]
    assert "branch_member_count" in legacy_tree.branch_table.columns
    current_tree.branch_table = current_tree.branch_table.rename(
        {"branch_member_count": "branch_vertex_count"}
    ).with_columns(pl.col("branch_vertex_count") + 1)
    assert "branch_member_count" not in current_tree.branch_table.columns
    new = build_biological_annotations(current)
    old = build_biological_annotations(legacy)
    assert new.pairs["branch_vertex_count"].to_list() == (
        old.pairs["branch_vertex_count"].to_list()
    )
    assert new.branches["branch_vertex_count"].to_list() == (
        old.branches["branch_vertex_count"].to_list()
    )


def test_a_branch_table_without_either_count_is_refused():
    analysis = _analysis()
    tree = analysis.trees["split"]
    tree.branch_table = tree.branch_table.drop("branch_member_count")
    with pytest.raises(ValueError, match="branch_vertex_count"):
        build_biological_annotations(analysis)
