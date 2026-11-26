"""Hand-counted descriptive statistics of a tiny graph."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.tables.graph_descriptive_statistics import (
    BY_VARIANT,
    COLUMNS,
    EN_DASH,
    POINT_VARIANT,
    GraphColumn,
    assemble_columns,
    directed_uphill,
    export_table,
    statistics_table,
    summarize_graph,
)


def _network(
    ids: list[str], values: list[float], edges: list[tuple[str, str]]
) -> ScalarNetwork:
    vertices = pl.DataFrame({"dna_sequence": ids, "fitness": values})
    edge_frame = pl.DataFrame(
        edges, schema={"source": pl.String, "target": pl.String}, orient="row"
    )
    graph = Graph.from_tables(
        vertices,
        edge_frame,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness",),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def _hand() -> ScalarNetwork:
    # a=0, b=1, c=1, d=2. Edges a—b, b—c, c—d.
    return _network(
        ["a", "b", "c", "d"],
        [0.0, 1.0, 1.0, 2.0],
        [("a", "b"), ("c", "b"), ("d", "c")],
    )


def test_counts_match_a_hand_enumeration() -> None:
    column = GraphColumn("Published", _hand(), threshold=1.0)
    stats = summarize_graph(column)
    assert stats["number of nodes"] == 4
    assert stats["number of edges"] == 3
    assert stats["minimum fitness"] == 0.0
    assert stats["maximum fitness"] == 2.0
    assert stats["mean fitness"] == pytest.approx(1.0)
    assert stats["median fitness"] == pytest.approx(1.0)
    # b, c and d are at or above 1; a is not.
    assert stats["functional count"] == 3
    assert stats["non-functional count"] == 1
    assert stats["% functional"] == pytest.approx(75.0)
    assert stats["functional threshold"] == 1.0
    # Degrees: a and d have one edge, b and c have two. Mean undirected = 1.5.
    assert stats["mean degree (undirected)"] == pytest.approx(1.5)
    assert stats["mean in/out degree (in = out)"] == pytest.approx(0.75)


def test_edges_are_oriented_from_lower_fitness_to_higher() -> None:
    arcs = directed_uphill(_hand())
    assert set(arcs) == {("a", "b"), ("b", "c"), ("c", "d")}
    # The stored b—c edge was written c→b; orientation follows fitness, then identity.
    assert ("c", "b") not in arcs


def test_full_graph_threshold_and_functional_counts_are_undefined() -> None:
    columns = tuple(
        GraphColumn(label, _hand(), None if label == "Full" else 1.0)
        for label in COLUMNS[:3]
    )
    by = GraphColumn(
        COLUMNS[3],
        _hand(),
        1.5,
        {"a": "non_functional", "b": "uncertain", "c": "functional", "d": "functional"},
    )
    table = statistics_table((*columns, by))
    for row in (
        "functional threshold",
        "functional count",
        "non-functional count",
        "% functional",
    ):
        assert table[row]["Full"] is None
    # Classes, not the threshold 1.5 (which would call only d functional).
    assert table["functional count"][COLUMNS[3]] == 2
    assert table["non-functional count"][COLUMNS[3]] == 1
    assert table["% functional"][COLUMNS[3]] == pytest.approx(50.0)
    assert table["functional threshold"][COLUMNS[3]] == 1.5


def test_export_writes_an_en_dash_and_the_readme(tmp_path) -> None:
    network = _hand()
    columns = [
        GraphColumn(COLUMNS[0], network, None),
        GraphColumn(COLUMNS[1], network, 1.0),
        GraphColumn(COLUMNS[2], network, 1.0),
        GraphColumn(
            COLUMNS[3],
            network,
            1.0,
            {
                "a": "non_functional",
                "b": "functional",
                "c": "functional",
                "d": "functional",
            },
        ),
    ]
    paths = export_table(
        statistics_table(columns),
        tmp_path,
        provenance={"columns": []},
        dpi=50,
    )
    written = {
        path.name: path.read_text(encoding="utf-8")
        for path in paths
        if path.suffix != ".png"
    }
    assert EN_DASH in written["graph_descriptive_statistics.csv"]
    assert "functional threshold" in written["graph_descriptive_statistics.csv"]
    header, first = written["graph_descriptive_statistics.tsv"].splitlines()[:2]
    assert "\t" in header
    assert first.startswith("number of nodes\t4\t")
    assert "mean in/out degree (in = out)" in written["README.md"]
    png = next(path for path in paths if path.suffix == ".png")
    assert png.read_bytes().startswith(b"\x89PNG")


def test_the_by_column_is_built_through_the_fdr_variant(monkeypatch) -> None:
    loaded: list[str | None] = []

    def fake_load(context=None):
        name = getattr(context, "variant_name", None)
        loaded.append(name)
        return {
            "analysis": {
                "clamped": {
                    "published_functional_threshold": -0.5,
                    "candidate_functional_threshold": {
                        "cutoff": "median_reflected" if name == BY_VARIANT else None
                    },
                }
            },
            "variant": name,
        }

    def fake_prepare(network, settings, context=None):
        variant = settings["variant"]
        if variant == POINT_VARIANT:
            selection = SimpleNamespace(
                rule="median_reflected",
                threshold=SimpleNamespace(threshold=0.25),
                classes=None,
            )
        else:
            selection = SimpleNamespace(
                rule="fdr_fixed_cutoff",
                threshold=SimpleNamespace(threshold=0.25),
                classes={"aa": "functional"},
            )
        return SimpleNamespace(network=f"network:{variant}"), selection

    module = "dubanevics2026topological.tables.graph_descriptive_statistics"
    monkeypatch.setattr(f"{module}._load_settings", fake_load)
    monkeypatch.setattr(
        f"{module}._prepare_clamped",
        lambda network, threshold: SimpleNamespace(network="published"),
    )
    monkeypatch.setattr(f"{module}._prepare_selected", fake_prepare)
    monkeypatch.setattr(
        f"{module}._selection_identity",
        lambda selection: {"rule": selection.rule},
    )
    columns, provenance = assemble_columns(
        {"analysis": {"clamped": {"published_functional_threshold": -0.5}}},
        full_network="FULL",
    )
    assert [column.network for column in columns] == [
        "FULL",
        "published",
        f"network:{POINT_VARIANT}",
        f"network:{BY_VARIANT}",
    ]
    assert loaded == [POINT_VARIANT, BY_VARIANT]
    assert provenance["columns"][3]["variant"] == BY_VARIANT
    assert columns[3].classes == {"aa": "functional"}
    assert columns[2].classes is None


def test_build_full_network_imports_the_repo_workflow(monkeypatch, tmp_path):
    """The Full graph comes from the repo's Papkou2023 workflow, not a topographer
    examples module (the pinned topographer does not ship one)."""
    from dubanevics2026topological import analysis
    from dubanevics2026topological.tables import graph_descriptive_statistics as mod

    seen = {}

    class FakeWorkflow:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def build_network(self):
            return "NETWORK"

    for name in ("wt", "diff"):
        pl.DataFrame({"x": [1]}).write_parquet(tmp_path / f"{name}.parquet")
    monkeypatch.setattr("dubanevics2026topological.papkou2023.Papkou2023Workflow", FakeWorkflow)
    monkeypatch.setattr(
        analysis, "dataset_input_path", lambda path, context, root=None: path
    )
    settings = {
        "analysis": {
            "dataset": {
                "fitness_wt": str(tmp_path / "wt.parquet"),
                "fitness_diff": str(tmp_path / "diff.parquet"),
            }
        }
    }
    assert mod.build_full_network(settings, None) == "NETWORK"
    assert seen["landscape"] == "full"
