"""Deterministic checks for clamped-landscape preparation and reference validation."""

from __future__ import annotations

import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.landscape.selected_landscape import (
    analyze_prepared_landscape,
    prepare_clamped_landscape,
    validate_clamped_reference,
)


def _network() -> ScalarNetwork:
    vertices = pl.DataFrame(
        {
            "dna_sequence": list("abcdefghij"),
            "fitness": [0.5, -0.1, -0.2, 0.4, 0.3, 0.6, -0.3, 0.1, -0.5, -0.6],
            "aa_sequence": ["AAA"] * 10,
            "has_stop_codon": [False] * 10,
        }
    )
    edges = pl.DataFrame(
        {
            "source": ["a", "b", "c", "d", "e", "c", "b", "i", "g"],
            "target": ["b", "c", "d", "e", "f", "g", "i", "j", "h"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "aa_sequence", "has_stop_codon"),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def test_threshold_excludes_low_low_edges_and_selects_largest_incident_component():
    prepared = prepare_clamped_landscape(_network(), 0.0)

    assert set(prepared.network.vertex_ids) == {"c", "d", "e", "f"}
    assert prepared.network.edge_count == 3
    assert prepared.network.scalar_attr == "raw_fitness"
    assert prepared.network.vertex_ids == ["c", "d", "e", "f"]
    assert prepared.diagnostics["low_low_edge_count_excluded"] == 4
    assert prepared.diagnostics["component_count"] == 3
    assert prepared.diagnostics["selected_component_vertex_count"] == 4
    assert set(prepared.excluded_vertices["dna_sequence"]) == {
        "a",
        "b",
        "g",
        "h",
        "i",
        "j",
    }
    assert set(
        prepared.excluded_vertices.filter(
            pl.col("reason") == "no_retained_incident_edge"
        )["dna_sequence"]
    ) == {"i", "j"}
    assert set(prepared.peaks["dna_sequence"]) == {"d", "f"}
    assert (
        prepared.network.graph[prepared.network.graph.node_indices()[0]]["raw_fitness"]
        == -0.2
    )


def test_equal_largest_components_use_stable_sequence_identity_tie_break():
    vertices = pl.DataFrame(
        {"dna_sequence": ["z", "zz", "a", "aa"], "s": [1.0, -1.0, 1.0, -1.0]}
    )
    edges = pl.DataFrame({"source": ["z", "a", "zz"], "target": ["zz", "aa", "aa"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("s",),
    )
    prepared = prepare_clamped_landscape(ScalarNetwork(graph, scalar_attr="s"), 0.0)

    assert set(prepared.network.vertex_ids) == {"a", "aa"}
    assert (
        prepared.components.filter(pl.col("selected"))["minimum_sequence"].item() == "a"
    )


def test_nonfinite_fitness_is_rejected():
    vertices = pl.DataFrame({"dna_sequence": ["a", "b"], "s": [0.0, float("nan")]})
    edges = pl.DataFrame({"source": ["a"], "target": ["b"]})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("s",),
    )

    with pytest.raises(ValueError, match="is not finite"):
        ScalarNetwork(graph, scalar_attr="s")


def test_reference_validation_requires_exact_peak_identities():
    prepared = prepare_clamped_landscape(_network(), 0.0)
    reference_graph = Graph.from_tables(
        pl.DataFrame(
            {
                "dna_sequence": ["c", "d", "e", "f"],
                "Fitness": [0.0, 0.4, 0.3, 0.6],
                "s": [-0.2, 0.4, 0.3, 0.6],
            }
        ),
        pl.DataFrame({"source": ["c", "d", "e"], "target": ["d", "e", "f"]}),
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("Fitness", "s"),
    )
    reference = ScalarNetwork(reference_graph, scalar_attr="s")

    valid = validate_clamped_reference(
        prepared,
        reference,
        pl.DataFrame({"dna": ["d", "f"]}),
    )
    assert valid["validated"] is True
    assert valid["max_raw_fitness_absolute_error"] == 0.0
    assert valid["max_clamped_fitness_absolute_error"] == 0.0
    with pytest.raises(ValueError, match="peak sequence identities differ"):
        validate_clamped_reference(prepared, reference, pl.DataFrame({"dna": ["d"]}))


def test_reference_validation_rejects_same_count_with_different_adjacency():
    prepared = prepare_clamped_landscape(_network(), 0.0)
    changed_edges = pl.DataFrame({"source": ["c", "d", "c"], "target": ["d", "e", "f"]})
    graph = Graph.from_tables(
        pl.DataFrame(
            {
                "dna_sequence": ["c", "d", "e", "f"],
                "Fitness": [0.0, 0.4, 0.3, 0.6],
                "s": [-0.2, 0.4, 0.3, 0.6],
            }
        ),
        changed_edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("Fitness", "s"),
    )
    reference = ScalarNetwork(graph, scalar_attr="s")

    assert reference.vertex_count == prepared.network.vertex_count
    assert reference.edge_count == prepared.network.edge_count
    with pytest.raises(ValueError, match="unordered edge identities differ"):
        validate_clamped_reference(
            prepared, reference, pl.DataFrame({"dna": ["d", "f"]})
        )


def test_reference_validation_rejects_scalar_change_with_matching_graph():
    prepared = prepare_clamped_landscape(_network(), 0.0)
    graph = Graph.from_tables(
        pl.DataFrame(
            {
                "dna_sequence": ["c", "d", "e", "f"],
                "Fitness": [0.0, 0.40001, 0.3, 0.6],
                "s": [-0.2, 0.4, 0.3, 0.6],
            }
        ),
        pl.DataFrame({"source": ["c", "d", "e"], "target": ["d", "e", "f"]}),
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("Fitness", "s"),
    )
    reference = ScalarNetwork(graph, scalar_attr="s")

    with pytest.raises(ValueError, match="Clamped fitness differs"):
        validate_clamped_reference(
            prepared, reference, pl.DataFrame({"dna": ["d", "f"]})
        )


def test_prepared_analysis_uses_raw_scalar_and_records_wild_type_position():
    prepared = prepare_clamped_landscape(_network(), 0.0)
    analysis = analyze_prepared_landscape(
        prepared,
        name="test-clamped",
        wild_type_sequence="c",
    )

    assert analysis.scalar_attr == "raw_fitness"
    assert analysis.scalar_domain == (-0.2, 0.6)
    assert analysis.trees["split"].wild_type_marker_table["y"].to_list() == [-0.2]
    assert analysis.trees["split"].plot_data.node_table["scalar"].min() <= -0.2
    assert not analysis.trees["split"].persistence_table.is_empty()


def test_threshold_selects_incident_edges_then_largest_expanded_component_immutably():
    vertices = pl.DataFrame(
        {
            "dna_sequence": ["a", "b", "c", "d", "e", "f", "g"],
            "fitness": [0.0, 0.0, -0.2, 0.5, -0.3, 0.4, 0.3],
            "metadata": list("ABCDEFG"),
        }
    )
    edges = pl.DataFrame(
        {
            "source": ["a", "b", "c", "c", "e", "f"],
            "target": ["b", "c", "d", "e", "f", "g"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "metadata"),
    )
    source = ScalarNetwork(graph, scalar_attr="fitness")
    original_nodes = {
        str(source.graph[index]["dna_sequence"]): dict(source.graph[index])
        for index in source.graph.node_indices()
    }
    original_edges = set(source.graph.edge_list())

    prepared = prepare_clamped_landscape(source, 0.0)

    # The exact-threshold pair is functional, and its equal-fitness edge remains.
    # Low c is retained as fringe between that pair and functional d; c-e is
    # excluded because both endpoints are below threshold. Expansion makes abcd
    # the largest component over efg.
    assert set(prepared.network.vertex_ids) == {"a", "b", "c", "d"}
    assert prepared.network.edge_count == 3
    assert {
        float(prepared.network.graph[index]["raw_fitness"])
        for index in prepared.network.graph.node_indices()
    } == {0.0, -0.2, 0.5}
    assert prepared.network.scalar_attr == "raw_fitness"
    assert prepared.diagnostics["low_low_edge_count_excluded"] == 1
    assert prepared.diagnostics["analysis_scalar"] == "raw_fitness"
    assert prepared.diagnostics["selected_component_vertex_count"] == 4
    assert set(prepared.peaks["dna_sequence"]) == {"a", "b", "d"}
    assert set(prepared.excluded_vertices["dna_sequence"]) == {"e", "f", "g"}
    assert {
        str(source.graph[index]["dna_sequence"]): dict(source.graph[index])
        for index in source.graph.node_indices()
    } == original_nodes
    assert set(source.graph.edge_list()) == original_edges


# The Functional class selects the vertices (ADR 2026-10-24) -------------------

CLASSES = {
    "a": "uncertain",
    "b": "non_functional",
    "c": "non_functional",
    "d": "functional",
    "e": "functional",
    "f": "functional",
    "g": "uncertain",
    "h": "functional",
    "i": "non_functional",
    "j": "non_functional",
}
FUNCTIONAL = {name for name, label in CLASSES.items() if label == "functional"}


def test_the_functional_class_selects_edges_with_a_functional_endpoint():
    prepared = prepare_clamped_landscape(
        _network(), 0.0, functional_sequences=FUNCTIONAL, functional_classes=CLASSES
    )

    # c is not Functional but is kept as the neighbour of d; a (raw 0.5, Uncertain)
    # is not kept although its raw fitness is above the threshold.
    assert set(prepared.network.vertex_ids) == {"c", "d", "e", "f"}
    assert prepared.network.edge_count == 3
    assert prepared.diagnostics["edge_selection_rule"] == (
        "at_least_one_endpoint_functional_class"
    )
    assert prepared.diagnostics["functional_vertex_count"] == 4
    assert prepared.diagnostics["uncertain_vertex_count"] == 2
    # a-b, b-c, c-g, b-i and i-j join no Functional vertex.
    assert prepared.diagnostics["low_low_edge_count_excluded"] == 5
    assert prepared.diagnostics["component_count"] == 2
    assert prepared.diagnostics["selected_component_vertex_count"] == 4
    # The raw-fitness tree is unchanged: raw values, clamped ones are metadata only.
    assert prepared.network.scalar_attr == "raw_fitness"
    graph = prepared.network.graph
    by_id = {graph[i]["dna_sequence"]: graph[i] for i in graph.node_indices()}
    assert by_id["c"]["raw_fitness"] == -0.2
    assert by_id["c"]["clamped_fitness"] == 0.0


def test_the_excluded_vertices_carry_their_functional_class():
    prepared = prepare_clamped_landscape(
        _network(), 0.0, functional_sequences=FUNCTIONAL, functional_classes=CLASSES
    )
    excluded = prepared.excluded_vertices
    assert excluded.columns[-1] == "functional_class"
    rows = {
        row["dna_sequence"]: (row["reason"], row["functional_class"])
        for row in excluded.iter_rows(named=True)
    }
    assert rows == {
        "g": ("non_largest_component", "uncertain"),
        "h": ("non_largest_component", "functional"),
        "a": ("no_retained_incident_edge", "uncertain"),
        "b": ("no_retained_incident_edge", "non_functional"),
        "i": ("no_retained_incident_edge", "non_functional"),
        "j": ("no_retained_incident_edge", "non_functional"),
    }


def test_the_scalar_rule_leaves_the_functional_class_null():
    prepared = prepare_clamped_landscape(_network(), 0.0)
    assert prepared.excluded_vertices["functional_class"].null_count() == (
        prepared.excluded_vertices.height
    )
    assert prepared.diagnostics["edge_selection_rule"] == (
        "at_least_one_endpoint_raw_fitness_ge_threshold"
    )
    assert prepared.diagnostics["functional_vertex_count"] is None


def test_the_largest_functional_component_wins_with_the_stable_tie_break():
    # a - b - x - c - d with Functional a and d: the edges b-x and x-c join no
    # Functional vertex, leaving two equal components {a, b} and {c, d}.
    graph = Graph.from_tables(
        pl.DataFrame({"dna_sequence": list("abxcd"), "s": [-1.0] * 5}),
        pl.DataFrame({"source": ["a", "b", "x", "c"], "target": ["b", "x", "c", "d"]}),
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("s",),
    )
    network = ScalarNetwork(graph, scalar_attr="s")
    prepared = prepare_clamped_landscape(network, 0.0, functional_sequences={"a", "d"})
    # No raw value reaches the threshold; the smaller minimum identity wins.
    assert set(prepared.network.vertex_ids) == {"a", "b"}
    # Without the class mapping a Functional vertex is labelled functional, the
    # rest null.
    assert dict(
        zip(
            prepared.excluded_vertices["dna_sequence"],
            prepared.excluded_vertices["functional_class"],
            strict=True,
        )
    ) == {"c": None, "d": "functional", "x": None}


def test_a_functional_sequence_that_is_not_a_vertex_is_refused():
    with pytest.raises(ValueError, match="not vertices of the Full graph"):
        prepare_clamped_landscape(_network(), 0.0, functional_sequences={"d", "zzz"})


def test_no_functional_vertex_selects_no_edge():
    with pytest.raises(ValueError, match="retained no edges"):
        prepare_clamped_landscape(_network(), 0.0, functional_sequences=set())
