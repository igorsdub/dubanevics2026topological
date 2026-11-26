"""The standalone Full join tree: valleys are the leaves, and the band is marked."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest
from _figure_labels import label_quantities, spec_texts
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.figures.join_tree import (
    FIGURE_ID,
    GRAPH_KEYS,
    analysis_view,
    build_join_tree_figure,
    join_tree_analysis,
    payload_for_graph,
)
from dubanevics2026topological.figures.registry import get_spec

BAND_LOWER = -0.4
BAND_UPPER = -0.1


def _network() -> ScalarNetwork:
    """A path of four sequences. The join tree's leaves are the two minima."""
    vertices = pl.DataFrame(
        {
            "dna_sequence": ["AAAAAAAAA", "AAAAAAAAC", "AAAAAACAA", "AAAAAACAC"],
            "raw_fitness": [-0.6, 0.2, -0.9, 1.0],
            "aa_sequence": ["KKK", "KKN", "KNK", "KNN"],
            "has_stop_codon": [False, False, False, False],
        }
    )
    edges = pl.DataFrame(
        {
            "source": ["AAAAAAAAA", "AAAAAAAAC", "AAAAAACAA"],
            "target": ["AAAAAAAAC", "AAAAAACAA", "AAAAAACAC"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("raw_fitness", "aa_sequence", "has_stop_codon"),
    )
    return ScalarNetwork(graph, scalar_attr="raw_fitness")


def _settings() -> dict:
    return {
        "figures": {
            "rendering": {
                "font_size": 15,
                "tree_width": 420,
                "tree_height": 250,
                "tree_axis_title": "Fitness (WT = 0)",
            },
            "figures": {
                FIGURE_ID: {
                    "colorbar_x_fraction": 0.9,
                    "count_legend_ticks": [1, 2],
                    "band_color": "#9e9e9e",
                    "band_opacity": 0.35,
                    "tree": {"width": 420, "height": 250},
                }
            },
        }
    }


def _payload(network: ScalarNetwork | None = None) -> dict:
    network = _network() if network is None else network
    tree = join_tree_analysis(network)
    return {
        "graph": "full",
        "analysis": analysis_view(network, tree),
        "tree": tree,
        "band_lower": BAND_LOWER,
        "band_upper": BAND_UPPER,
        "manuscript_quantities": label_quantities(),
    }


def test_join_tree_leaves_are_valleys() -> None:
    tree = join_tree_analysis(_network())
    leaves = [
        index
        for index in tree.tree.node_indices()
        if tree.tree.in_degree(index) == 0 and tree.tree.out_degree(index) > 0
    ]
    assert leaves
    assert {tree.tree.get_node_data(index)["event"] for index in leaves} == {"minimum"}


def test_join_tree_labels_come_from_the_registry() -> None:
    quantities = label_quantities()
    payload = _payload()
    payload["manuscript_quantities"] = quantities
    build_join_tree_figure(payload, _settings(), "vanilla")
    assert set(quantities.requested_keys) == {
        "full_graph.vertex_count",
        "full_graph.persistence.valley_count",
        "leading_saddle_band.band.lower",
        "leading_saddle_band.band.upper",
    }


def test_full_join_tree_marks_the_band_on_the_fitness_axis() -> None:
    chart = build_join_tree_figure(_payload(), _settings(), "vanilla")
    spec = chart.to_dict()
    texts = " ".join(spec_texts(spec))
    quantities = label_quantities()
    assert "Full graph" in texts
    assert "valleys are the leaves" in texts
    assert quantities.display("leading_saddle_band.band.lower") in texts
    assert quantities.display("leading_saddle_band.band.upper") in texts
    assert quantities.display("full_graph.persistence.valley_count") in texts
    assert "Fitness (WT = 0)" in texts

    rects = _marks(spec, "rect")
    assert rects
    encoded = rects[0]["encoding"]
    assert encoded["y"]["field"] == "y"
    assert encoded["y2"]["field"] == "y_end"
    assert _rows(spec, "y", "y_end") == [{"y": BAND_LOWER, "y_end": BAND_UPPER}]


def test_the_builder_accepts_later_graphs_and_only_full_is_loaded() -> None:
    assert GRAPH_KEYS == ("full", "published", "candidate")
    spec = get_spec(FIGURE_ID)
    assert spec.default is False
    assert spec.variants == frozenset({"vanilla"})
    assert spec.inputs == frozenset(
        {"full_network", "leading_saddle_band", "manuscript_quantities"}
    )
    network = _network()
    tree = join_tree_analysis(network)
    for graph in ("published", "candidate"):
        chart = build_join_tree_figure(
            {
                **_payload(network),
                "graph": graph,
                "tree": tree,
                "analysis": analysis_view(network, tree),
            },
            _settings(),
            "vanilla",
        )
        heading = {
            "full": "Full graph",
            "published": "Papkou graph",
            "candidate": "Topographer + FDR graph",
        }[graph]
        assert heading in " ".join(spec_texts(chart.to_dict()))
    with pytest.raises(NotImplementedError, match="not wired"):
        payload_for_graph(
            "candidate",
            {
                "full_network": network,
                "leading_saddle_band": SimpleNamespace(
                    summary={"band_lower": BAND_LOWER, "band_upper": BAND_UPPER}
                ),
                "manuscript_quantities": label_quantities(),
            },
            {},
        )


def test_annotated_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="vanilla"):
        build_join_tree_figure(_payload(), _settings(), "annotated")


def _marks(node, mark: str) -> list[dict]:
    found = []
    if isinstance(node, dict):
        kind = node.get("mark")
        name = kind.get("type") if isinstance(kind, dict) else kind
        if name == mark and "encoding" in node:
            found.append(node)
        for value in node.values():
            found.extend(_marks(value, mark))
    elif isinstance(node, list):
        for value in node:
            found.extend(_marks(value, mark))
    return found


def _rows(node, *fields: str) -> list[dict]:
    found = []
    if isinstance(node, dict):
        if all(field in node for field in fields) and all(
            isinstance(node[field], (int, float)) for field in fields
        ):
            found.append({field: node[field] for field in fields})
        for value in node.values():
            found.extend(_rows(value, *fields))
    elif isinstance(node, list):
        for value in node:
            found.extend(_rows(value, *fields))
    return found
