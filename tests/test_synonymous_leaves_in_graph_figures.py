"""Stop-codon and synonymous leaves on every graph's tree figure, one code path.

The Published graph's biology figure draws a synonymous leaf (a terminal peak branch
whose adjacent saddle has the peak's amino acids) as a green branch and a green
triangle. The saved-tree view that renders the Published BY, Candidate and Candidate
BY graphs draws the same marks through the same helpers in ``figures/_charts.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest

from dubanevics2026topological.build.d10_tree_inputs import (
    SavedTreeInputs,
    persistence_table_from_pairs,
)
from dubanevics2026topological.figures import _charts
from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures import study_tree_persistence_biology as study
from dubanevics2026topological.figures import tree_persistence_view as view


@dataclass
class _PlotData:
    edge_table: pl.DataFrame
    edge_segment_table: pl.DataFrame
    node_table: pl.DataFrame


def _plot_data() -> _PlotData:
    # Two peaks over one merge saddle, and a root. P1 and its saddle share the amino
    # acids "K" (synonymous leaf); P2 differs ("R").
    nodes = pl.DataFrame(
        {
            "tree_node_index": [0, 1, 2, 3],
            "dna_sequence": ["ROOT", "SADDLE", "P1", "P2"],
            "aa_sequence": ["M", "K", "K", "R"],
            "display_event": ["minimum", "split_saddle", "maximum", "maximum"],
        }
    )
    edges = pl.DataFrame(
        {
            "source_tree_node_index": [0, 1, 1],
            "target_tree_node_index": [1, 2, 3],
            "source_label": ["ROOT", "SADDLE", "SADDLE"],
            "target_label": ["SADDLE", "P1", "P2"],
            "vertex_count": [2, 3, 1],
        }
    )
    segments = pl.DataFrame(
        {
            "source_label": ["SADDLE", "SADDLE"],
            "target_label": ["P1", "P2"],
            "x_start": [0.0, 0.0],
            "x_end": [1.0, 2.0],
            "y_start": [0.0, 0.0],
            "y_end": [0.5, 0.4],
            "logical_edge_index": [1, 2],
        }
    )
    return _PlotData(edges, segments, nodes)


def test_synonymous_leaf_edges_select_peak_branches_with_a_matching_saddle() -> None:
    found = _charts.synonymous_leaf_edges(_plot_data())
    assert found["target_label"].to_list() == ["P1"]
    assert found["source_label"].to_list() == ["SADDLE"]


def test_synonymous_leaf_edges_need_a_peak_over_a_saddle() -> None:
    data = _plot_data()
    # P1 is no longer a maximum: its branch is not a terminal peak branch.
    nodes = data.node_table.with_columns(
        pl.when(pl.col("dna_sequence") == "P1")
        .then(pl.lit("split_saddle"))
        .otherwise(pl.col("display_event"))
        .alias("display_event")
    )
    assert _charts.synonymous_leaf_edges(
        _PlotData(data.edge_table, data.edge_segment_table, nodes)
    ).is_empty()
    # A table without amino acids cannot name a synonymous leaf: it is an error, not an empty result.
    bare = _PlotData(
        data.edge_table, data.edge_segment_table, data.node_table.drop("aa_sequence")
    )
    with pytest.raises(ValueError, match="aa_sequence"):
        _charts.synonymous_leaf_edges(bare)
    no_event = _PlotData(
        data.edge_table, data.edge_segment_table, data.node_table.drop("display_event")
    )
    with pytest.raises(ValueError, match="event"):
        _charts.synonymous_leaf_edges(no_event)


def test_the_biology_figure_and_the_saved_tree_view_share_one_branch_layer() -> None:
    assert view.synonymous_branch_layer is _charts.synonymous_branch_layer
    assert study.synonymous_branch_layer is _charts.synonymous_branch_layer
    assert not hasattr(study, "_synonymous_peak_branch_layer")


def _inputs() -> SavedTreeInputs:
    pairs = pl.DataFrame(
        {
            "birth_label": ["P1", "P2"],
            "death_label": ["ROOT", "SADDLE"],
            "birth_scalar": [0.9, 0.5],
            "death_scalar": [-0.2, 0.1],
            "persistence": [1.1, 0.4],
            "pair_kind": ["ordinary", "ordinary"],
        }
    )
    tree = SimpleNamespace(
        wild_type_marker_table=pl.DataFrame({"x": [0.0], "y": [0.0]}),
        all_stop_marker_table=pl.DataFrame(),
    )
    return SavedTreeInputs(
        tree_label="tree__thr=published_fdr=by_q=0.05_tree=nominal_hash=00000000",
        tree_id="a" * 64,
        layout_id="b" * 64,
        landscape_label="landscape__thr=published_fdr=by_q=0.05_hash=00000000",
        landscape_id="c" * 64,
        tree_analysis=tree,
        persistence_table=persistence_table_from_pairs(pairs),
        scalar_domain=(-0.2, 1.0),
        vertex_count=4,
        fitness_values=pl.Series([-0.1, 0.0, 0.5, 1.0]),
    )


def _patched_view(monkeypatch, style: str):
    seen: dict = {}
    monkeypatch.setattr(view, "plot_data_with_metrics", lambda tree: _plot_data())
    monkeypatch.setattr(
        view,
        "persistence_with_counts",
        lambda pairs, plot, domain: pairs.with_columns(
            pl.Series("vertex_count", [3, 1])
        ),
    )

    def tree_panel(*args, **kwargs):
        seen["overlays"] = kwargs.get("overlays")
        return (
            alt.Chart(pl.DataFrame({"y": [0.0]}))
            .mark_point()
            .encode(y="y:Q")
            .properties(width=50, height=50)
        )

    def diagram_panel(*args, **kwargs):
        return (
            alt.Chart(pl.DataFrame({"birth_scalar": [0.0], "death_scalar": [0.0]}))
            .mark_point()
            .encode(x="death_scalar:Q", y="birth_scalar:Q")
            .properties(width=50, height=50)
        )

    monkeypatch.setattr(view, "tree_chart", tree_panel)
    monkeypatch.setattr(view, "overlay_count_colorbar", lambda tree, *a: tree)
    monkeypatch.setattr(view, "persistence_chart", diagram_panel)
    monkeypatch.setattr(view, "biology_axis_contract", lambda chart, **kw: chart)
    monkeypatch.setattr(view, "publication_scalar_scales", lambda chart, *a: chart)
    settings = {"figures": {"d10": {"tree_persistence": {"histogram_height": 70}}}}
    chart = view.build_tree_persistence_figure(
        _inputs(), settings, style, title="t", subtitle="s"
    )
    return chart, seen


def test_annotated_view_draws_the_synonymous_leaf_green_on_tree_and_diagram(
    monkeypatch,
) -> None:
    chart, seen = _patched_view(monkeypatch, "annotated")
    (overlay,) = seen["overlays"]
    rule = overlay.to_dict()
    assert rule["mark"]["color"] == palette.SYNONYMOUS_GREEN
    data = overlay.data
    assert data["target_label"].to_list() == ["P1"]  # P2 is not synonymous
    layers = chart.hconcat[1].vconcat[1].to_dict()["layer"]
    green = [
        layer
        for layer in layers
        if layer["mark"]["type"] == "point"
        and layer["mark"].get("color") == palette.SYNONYMOUS_GREEN
    ]
    assert len(green) == 1
    assert green[0]["mark"]["shape"] == "triangle-up"
    assert green[0]["mark"]["filled"] is True
    labels = [
        layer["mark"]["text"]
        for layer in layers
        if layer["mark"]["type"] == "text" and layer["mark"]["text"] not in {"▲", "★"}
    ]
    assert "synonymous leaves" in labels
    assert "valley" not in labels


def test_vanilla_view_draws_no_synonymous_leaf(monkeypatch) -> None:
    chart, seen = _patched_view(monkeypatch, "vanilla")
    assert not seen["overlays"]
    layers = chart.hconcat[1].vconcat[1].to_dict()["layer"]
    assert not [
        layer
        for layer in layers
        if layer["mark"].get("color") == palette.SYNONYMOUS_GREEN
    ]
    texts = [layer["mark"].get("text") for layer in layers]
    assert "synonymous leaves" not in texts

