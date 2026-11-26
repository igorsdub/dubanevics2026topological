"""Checks for full-network component pair identities and threshold diagnostics."""

from __future__ import annotations

import json

import polars as pl
import pytest
from topographer.examples.cube import cube_edge_table, cube_node_table
from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology

import dubanevics2026topological.analysis.component_persistence as component_module
from dubanevics2026topological.analysis.component_persistence import (
    PAIR_SCHEMA,
    _pairs_with_sequence_identities,
    build_threshold_diagnostics,
)
from dubanevics2026topological.figures.component_persistence import (
    descending_component_pairs,
)


def test_component_pair_export_preserves_pair_kinds_and_sequence_endpoints() -> None:
    nodes = cube_node_table()
    edges = cube_edge_table()
    graph = Graph.from_tables(
        nodes,
        edges,
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height",),
    )
    topology = ScalarNetworkTopology(ScalarNetwork(graph, scalar_attr="height"))
    public_result = topology.component_persistence()

    pairs = _pairs_with_sequence_identities(
        public_result.pairs,
        {"001": True},
    )

    assert set(pairs["pair_kind"]) == {"ordinary", "relative", "extended_positive"}
    ordinary = pairs.filter(pl.col("pair_kind") == "ordinary").row(0, named=True)
    assert ordinary["homology_dimension"] == 0
    assert (ordinary["birth_sequence"], ordinary["death_sequence"]) == ("100", "000")
    assert ordinary["birth_scalar"] == 0.1
    assert ordinary["death_scalar"] == 0.5
    assert ordinary["persistence"] == 0.4
    assert ordinary["birth_scalar"] <= ordinary["death_scalar"]
    relative = pairs.filter(pl.col("pair_kind") == "relative").row(0, named=True)
    assert relative["homology_dimension"] == 1
    assert (relative["birth_sequence"], relative["death_sequence"]) == (
        "110",
        "111",
    )
    assert relative["birth_scalar"] == 0.9
    assert relative["death_scalar"] == 0.8
    assert relative["birth_scalar"] >= relative["death_scalar"]
    assert relative["persistence"] == pytest.approx(0.1)
    global_pair = pairs.filter(pl.col("pair_kind") == "extended_positive").row(
        0, named=True
    )
    assert (global_pair["birth_sequence"], global_pair["death_sequence"]) == (
        "001",
        "011",
    )
    assert global_pair["birth_has_stop_codon"] is True
    assert pairs.schema == PAIR_SCHEMA

    for public_row in public_result.pairs.iter_rows(named=True):
        exported = pairs.filter(pl.col("pair_id") == public_row["pair_id"]).row(
            0, named=True
        )
        assert exported["pair_kind"] == public_row["pair_kind"]
        assert exported["homology_dimension"] == public_row["homology_dimension"]
        assert exported["birth_sequence"] == public_row["birth_simplex"][0]
        assert exported["death_sequence"] == public_row["death_simplex"][0]
        assert exported["birth_scalar"] == public_row["birth_scalar"]
        assert exported["death_scalar"] == public_row["death_scalar"]
        assert exported["persistence"] == public_row["persistence"]


def test_tied_component_values_keep_deterministic_global_pair_identity() -> None:
    vertices = pl.DataFrame({"sequence": ["a", "b", "c"], "fitness": [0.5, 0.5, 0.5]})
    edges = pl.DataFrame({"left": ["a", "b"], "right": ["b", "c"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="sequence",
        source_id_col="left",
        target_id_col="right",
        vertex_attribute_cols=("fitness",),
    )
    result = ScalarNetworkTopology(
        ScalarNetwork(graph, scalar_attr="fitness")
    ).component_persistence()

    pairs = _pairs_with_sequence_identities(result.pairs)

    assert pairs.height == 1
    assert pairs["pair_kind"].to_list() == ["extended_positive"]
    assert pairs["birth_sequence"].to_list() == ["a"]
    assert pairs["death_sequence"].to_list() == ["c"]
    assert pairs["birth_scalar"].to_list() == [0.5]
    assert pairs["death_scalar"].to_list() == [0.5]


@pytest.mark.parametrize(
    "heights",
    [
        [0.5, 0.0, 0.6, 1.0, 0.1, 0.7, 0.9, 0.8],
        [0.5, 0.0, 0.6, 1.0, 0.1, 0.7, 0.9, 0.9],
    ],
    ids=["distinct-values", "finite-tie"],
)
def test_descending_view_matches_public_api_on_negated_asymmetric_network(
    heights: list[float],
) -> None:
    """The relabeled pairs agree with public join/split persistence on -fitness."""
    source_nodes = cube_node_table().with_columns(pl.Series("height", heights))
    nodes = source_nodes.with_columns((-pl.col("height")).alias("height"))
    edges = cube_edge_table()
    graph = Graph.from_tables(
        nodes,
        edges,
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height",),
    )
    negated_result = ScalarNetworkTopology(
        ScalarNetwork(graph, scalar_attr="height")
    ).component_persistence()
    negated_export = _pairs_with_sequence_identities(negated_result.pairs)

    raw_graph = Graph.from_tables(
        source_nodes,
        edges,
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height",),
    )
    ascending = _pairs_with_sequence_identities(
        ScalarNetworkTopology(ScalarNetwork(raw_graph, scalar_attr="height"))
        .component_persistence()
        .pairs
    )
    descending = descending_component_pairs(ascending)

    for kind in ("ordinary", "relative"):
        actual = descending.filter(pl.col("pair_kind") == kind).sort("pair_id")
        expected = negated_export.filter(pl.col("pair_kind") == kind)
        assert (
            actual["birth_sequence"].to_list() == expected["birth_sequence"].to_list()
        )
        assert (
            actual["death_sequence"].to_list() == expected["death_sequence"].to_list()
        )
        assert actual["birth_scalar"].to_list() == pytest.approx(
            [-x for x in expected["birth_scalar"].to_list()]
        )
        assert actual["death_scalar"].to_list() == pytest.approx(
            [-x for x in expected["death_scalar"].to_list()]
        )
        assert actual["persistence"].to_list() == pytest.approx(
            expected["persistence"].to_list()
        )
    assert descending.filter(pl.col("pair_kind") == "ordinary")[
        "homology_dimension"
    ].to_list() == [0]
    assert descending.filter(pl.col("pair_kind") == "relative")[
        "homology_dimension"
    ].to_list() == [1]


def test_descending_global_pair_preserves_checkpoint_extrema_under_ties() -> None:
    vertices = pl.DataFrame(
        {"sequence": ["a", "b", "c"], "fitness": [-0.5, -0.5, -0.5]}
    )
    edges = pl.DataFrame({"left": ["a", "b"], "right": ["b", "c"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="sequence",
        source_id_col="left",
        target_id_col="right",
        vertex_attribute_cols=("fitness",),
    )
    ascending_result = ScalarNetworkTopology(
        ScalarNetwork(graph, scalar_attr="fitness")
    ).component_persistence()
    ascending = _pairs_with_sequence_identities(ascending_result.pairs)
    descending = descending_component_pairs(ascending)
    global_pair = descending.row(0, named=True)
    assert (global_pair["birth_sequence"], global_pair["death_sequence"]) == ("c", "a")
    assert global_pair["birth_scalar"] == global_pair["death_scalar"] == -0.5


def test_one_vertex_network_has_one_zero_persistence_global_pair() -> None:
    vertices = pl.DataFrame({"sequence": ["only"], "fitness": [0.25]})
    edges = pl.DataFrame(schema={"left": pl.String, "right": pl.String})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="sequence",
        source_id_col="left",
        target_id_col="right",
        vertex_attribute_cols=("fitness",),
    )
    result = ScalarNetworkTopology(
        ScalarNetwork(graph, scalar_attr="fitness")
    ).component_persistence()

    pairs = _pairs_with_sequence_identities(result.pairs)

    assert pairs.height == 1
    assert pairs["pair_kind"].to_list() == ["extended_positive"]
    assert pairs["birth_sequence"].to_list() == ["only"]
    assert pairs["death_sequence"].to_list() == ["only"]
    assert pairs["persistence"].to_list() == [0.0]


def test_threshold_diagnostics_report_the_join_tree_trailing_saddle() -> None:
    pairs = pl.DataFrame(
        {
            "pair_id": [0, 1, 2],
            "pair_kind": ["ordinary", "ordinary", "extended_positive"],
            "homology_dimension": [0, 0, 0],
            "birth_sequence": ["a", "b", "a"],
            "death_sequence": ["c", "d", "e"],
            "birth_has_stop_codon": [False, False, False],
            "death_has_stop_codon": [False, True, False],
            "birth_scalar": [-1.0, -0.9, -1.2],
            "death_scalar": [-0.713981, -0.8, -0.1],
            "persistence": [0.286019, 0.1, 1.1],
        },
        schema=PAIR_SCHEMA,
    )

    diagnostics = build_threshold_diagnostics(pairs)

    assert diagnostics["ordinary_death_value_frequencies"] == [
        {"death_scalar": -0.8, "count": 1},
        {"death_scalar": -0.713981, "count": 1},
    ]
    assert diagnostics["join_tree_trailing_saddle"] == -0.713981
    assert diagnostics["schema"] == 3
    assert "split_tree_leading_saddle" not in diagnostics
    assert diagnostics["global_minimum"] == -1.2
    assert not any(key.startswith("proposed") for key in diagnostics)
    assert "reflected_candidate_threshold" not in diagnostics
    assert "observed_upper_ordinary_death_boundary" not in diagnostics


def test_threshold_diagnostics_handle_no_ordinary_pairs() -> None:
    pairs = pl.DataFrame(
        {
            "pair_id": [0],
            "pair_kind": ["extended_positive"],
            "homology_dimension": [0],
            "birth_sequence": ["a"],
            "death_sequence": ["a"],
            "birth_has_stop_codon": [False],
            "death_has_stop_codon": [False],
            "birth_scalar": [0.4],
            "death_scalar": [0.4],
            "persistence": [0.0],
        },
        schema=PAIR_SCHEMA,
    )

    diagnostics = build_threshold_diagnostics(pairs)

    assert diagnostics["ordinary_pair_count"] == 0
    assert diagnostics["ordinary_death_value_frequencies"] == []
    assert diagnostics["join_tree_trailing_saddle"] is None


def test_cached_component_analysis_reuses_matching_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_cache_paths(tmp_path, monkeypatch)
    pairs = _single_global_pair()
    _write_cached_component_artifacts(paths, pairs)
    monkeypatch.setattr(
        component_module, "scientific_fingerprint", lambda settings: {"key": "same"}
    )
    monkeypatch.setattr(
        component_module,
        "compute_component_persistence_analysis",
        lambda settings: pytest.fail("matching cache should be reused"),
    )

    loaded_pairs, diagnostics = component_module.load_or_compute_component_persistence()

    assert loaded_pairs.equals(pairs)
    assert diagnostics["pair_count"] == 1


def test_cached_component_analysis_rejects_tampering(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_cache_paths(tmp_path, monkeypatch)
    _write_cached_component_artifacts(paths, _single_global_pair())
    paths["diagnostics"].write_text('{"pair_count": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="do not match their provenance"):
        component_module.load_cached_component_persistence()


def test_pending_component_refresh_blocks_cached_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_cache_paths(tmp_path, monkeypatch)
    _write_cached_component_artifacts(paths, _single_global_pair())
    paths["pending"].write_text("refresh pending\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refresh is pending"):
        component_module.load_cached_component_persistence()


def test_component_artefacts_are_byte_stable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterisation (ADR 0013, Phase 3): the two artefacts are a function of the inputs.

    A forced recompute and a reload must leave ``component_pairs.parquet`` and
    ``component_diagnostics.json`` byte for byte as they were, because Phase 3 puts
    the component stage on the produce-or-load helper and exports these files from
    a versioned checkpoint. ``component_provenance.json`` is left out on purpose:
    the helper's record carries ``created_at`` and a git stamp, so it is not stable
    and is not part of the claim. The real analysis runs on the cube landscape; only
    the workflow that builds the network and the fingerprint are replaced.
    """
    from _manifest import file_digest

    paths = _install_cache_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(component_module, "ROOT", tmp_path)
    for name in ("wt.parquet", "diff.parquet"):
        pl.DataFrame({"dna_from": ["000"]}).write_parquet(tmp_path / name)
    built: list[str] = []

    class CubeWorkflow:
        def __init__(self, *, landscape: str, fitness_wt, fitness_diff) -> None:
            self.landscape = landscape

        def build_network(self) -> ScalarNetwork:
            built.append(self.landscape)
            nodes = cube_node_table().rename({"node_id": "dna_sequence"})
            graph = Graph.from_tables(
                nodes,
                cube_edge_table(),
                vertex_id_col="dna_sequence",
                source_id_col="source",
                target_id_col="target",
                vertex_attribute_cols=("height",),
            )
            return ScalarNetwork(graph, scalar_attr="height")

    monkeypatch.setattr(component_module, "Papkou2023Workflow", CubeWorkflow)
    monkeypatch.setattr(
        component_module, "scientific_fingerprint", lambda settings: {"key": "cube"}
    )
    settings = {
        "analysis": {
            "dataset": {
                "landscape": "full",
                "fitness_wt": "wt.parquet",
                "fitness_diff": "diff.parquet",
            },
            "persistence": {"component": {"proposed_death_boundary": -0.713981}},
        }
    }

    def artefacts() -> dict[str, str]:
        return {name: file_digest(paths[name]) for name in ("pairs", "diagnostics")}

    pairs, diagnostics = component_module.load_or_compute_component_persistence(
        force=True, settings=settings
    )
    assert built == ["full"]
    assert pairs.height == diagnostics["pair_count"] > 1
    first = artefacts()

    again, _ = component_module.load_or_compute_component_persistence(
        force=True, settings=settings
    )
    assert built == ["full", "full"]  # recomputed, not reused
    assert again.equals(pairs)
    assert artefacts() == first

    # A reuse and a plain reload read the same bytes and write none.
    component_module.load_or_compute_component_persistence(settings=settings)
    component_module.load_cached_component_persistence()
    assert built == ["full", "full"]
    assert artefacts() == first
    assert not paths["pending"].exists()


def _single_global_pair() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pair_id": [0],
            "pair_kind": ["extended_positive"],
            "homology_dimension": [0],
            "birth_sequence": ["a"],
            "death_sequence": ["a"],
            "birth_has_stop_codon": [False],
            "death_has_stop_codon": [False],
            "birth_scalar": [0.4],
            "death_scalar": [0.4],
            "persistence": [0.0],
        },
        schema=PAIR_SCHEMA,
    )


def _install_cache_paths(tmp_path, monkeypatch: pytest.MonkeyPatch):
    paths = {
        "pairs": tmp_path / "component_pairs.parquet",
        "diagnostics": tmp_path / "component_diagnostics.json",
        "provenance": tmp_path / "component_provenance.json",
        "pending": tmp_path / ".component-analysis-pending",
    }
    monkeypatch.setattr(
        component_module,
        "component_paths",
        lambda context=None: component_module.ComponentPaths(
            tmp_path,
            paths["pairs"],
            paths["diagnostics"],
            paths["provenance"],
            paths["pending"],
        ),
    )
    return paths


def _write_cached_component_artifacts(paths, pairs: pl.DataFrame) -> None:
    diagnostics = build_threshold_diagnostics(pairs)
    pairs.write_parquet(paths["pairs"])
    paths["diagnostics"].write_text(json.dumps(diagnostics), encoding="utf-8")
    paths["provenance"].write_text(
        json.dumps(
            {
                "schema": 1,
                "fingerprint": {"key": "same"},
                "pairs_sha256": component_module._sha256(paths["pairs"]),
                "diagnostics_sha256": component_module._sha256(paths["diagnostics"]),
            }
        ),
        encoding="utf-8",
    )
