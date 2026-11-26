from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.analysis.graph_summary import (
    SUMMARY_SCHEMA,
)
from dubanevics2026topological.analysis.graph_summary import (
    _summarize_network as summarize_network,
)
from dubanevics2026topological.analysis.graph_summary import (
    _summary_payload as summary_payload,
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


def test_summary_uses_each_graph_population_and_inclusive_threshold() -> None:
    full = _network(
        ["a", "b", "c", "x"],
        [0.0, 0.5, -0.5, -1.0],
        [("a", "b"), ("b", "c"), ("c", "x")],
    )
    published = _network(["a", "b", "c"], [0.0, 0.5, -0.5], [("a", "b"), ("b", "c")])
    candidate = _network(["b", "x"], [0.5, -1.0], [("b", "x")])

    payload = summary_payload(full, published, candidate, 0.0, -1.0)
    assert payload["schema"] == SUMMARY_SCHEMA
    assert payload["metrics"]["Full"]["Number of nodes"] == 4
    assert payload["metrics"]["Full"]["Mean fitness"] == pytest.approx(-0.25)
    assert payload["metrics"]["Published"]["Number of nodes"] == 3
    assert payload["metrics"]["Published"]["Number of functional sequences"] == 2
    assert payload["metrics"]["Published"]["Number of non-functional sequences"] == 1
    assert payload["metrics"]["Candidate"]["Number of functional sequences"] == 2
    assert (
        payload["metrics"]["Candidate"]["Percentage of functional sequences"] == 100.0
    )
    assert payload["metrics"]["Full"]["Functional threshold"] is None


def test_the_candidate_counts_follow_the_classes_not_raw_fitness() -> None:
    full = _network(
        ["a", "b", "c", "d", "x"],
        [0.9, 0.5, 0.4, -0.5, -1.0],
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "x")],
    )
    published = _network(["a", "b"], [0.9, 0.5], [("a", "b")])
    candidate = _network(["b", "c", "d"], [0.5, 0.4, -0.5], [("b", "c"), ("c", "d")])
    # Raw fitness at T = 0.45 would call a and b functional; the classes say b is
    # Uncertain (a Functional-class vertex is not one of the Candidate vertices).
    classes = {
        "a": "functional",
        "b": "uncertain",
        "c": "functional",
        "d": "non_functional",
        "x": "non_functional",
    }
    metrics = summary_payload(
        full, published, candidate, 0.0, 0.45, candidate_classes=classes
    )["metrics"]["Candidate"]
    assert metrics["Number of nodes"] == 3
    assert metrics["Number of functional sequences"] == 1
    assert metrics["Number of non-functional sequences"] == 1
    assert metrics["Number of uncertain sequences"] == 1
    assert metrics["Percentage of functional sequences"] == pytest.approx(100 / 3)
    assert metrics["Functional threshold"] == 0.45
    # The raw rule, for comparison, counts b at or above T.
    raw = summary_payload(full, published, candidate, 0.0, 0.45)["metrics"]["Candidate"]
    assert raw["Number of functional sequences"] == 1
    assert raw["Number of uncertain sequences"] is None
    # Every Candidate vertex must be classified.
    with pytest.raises(ValueError, match="no functional class"):
        summary_payload(
            full, published, candidate, 0.0, 0.45, candidate_classes={"a": "functional"}
        )


@pytest.mark.parametrize("fitness", [None, "bad", float("nan"), float("inf")])
def test_invalid_fitness_names_graph_and_sequence(fitness: object) -> None:
    class FakeGraph:
        def node_indices(self) -> list[int]:
            return [5]

        def __getitem__(self, index: int) -> dict[str, object]:
            return {"dna_sequence": "seq-A", "fitness": fitness}

    network = SimpleNamespace(
        graph=FakeGraph(),
        scalar_attr="fitness",
        edge_count=0,
    )
    with pytest.raises(ValueError, match="Published graph.*seq-A"):
        summarize_network(network, "Published", 0.0)


def test_comparison_science_fingerprint_tracks_metrics_not_presentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dubanevics2026topological.analysis.clamped_cache as cache
    from dubanevics2026topological import stages

    monkeypatch.setattr(
        cache,
        "candidate_selection_fingerprint",
        lambda _, **kwargs: {
            "rounded_threshold": -0.25,
            "source": {"rule": "median_reflected"},
        },
    )
    root = tmp_path
    provenance = root / "provenance.json"
    provenance.write_text('{"fingerprint":"baseline"}', encoding="utf-8")
    monkeypatch.setattr(cache, "ROOT", root)
    monkeypatch.setattr(
        cache, "analysis_provenance_path", lambda context=None: provenance
    )
    monkeypatch.setattr(
        cache, "scientific_inputs", lambda settings, *args, **kwargs: []
    )
    monkeypatch.setattr(cache, "require_files", lambda *args: None)
    monkeypatch.setattr(
        cache,
        "value_digest",
        lambda path: "graph" if path.name == "graph.graphml" else "peaks",
    )
    captured: list[dict[str, object]] = []

    def stage(name: str, **kwargs: object) -> object:
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(stages, "project_stage", stage)
    settings = {
        "analysis": {
            "dataset": {"name": "Papkou2023"},
            "clamped": {
                "graphml": "graph.graphml",
                "peaks": "peaks.csv",
                "graphml_value_sha256": "graph",
                "peaks_value_sha256": "peaks",
                "published_functional_threshold": 0.0,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    cache.comparison_stage(settings)
    settings["figures"] = {
        "tables": {"graph_summary": {"fitness_significant_digits": 2}}
    }
    cache.comparison_stage(settings)
    assert captured[0]["settings"] == captured[1]["settings"]
    sources = str(captured[0]["settings"]["sources"])
    assert "/analysis/graph_summary.py" in sources
    assert "/tables/graph_summary.py" not in sources
