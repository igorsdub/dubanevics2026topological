"""The ``fdr_fixed_cutoff`` Selection rule: an authenticated FDR snapshot selects the graph.

A hand-made snapshot (six sequences on a path a-b-c-d-e-f) is written under ``tmp_path``;
nothing here reads ``data/raw/`` or the real tables.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.landscape import selection as cs
from dubanevics2026topological.threshold import fdr_selection as fdr

IDS = ["a", "b", "c", "d", "e", "f"]
FITNESS = [0.9, 0.1, 0.8, -0.2, -0.5, -0.6]
CLASSES = [
    "functional",
    "uncertain",
    "functional",
    "non_functional",
    "non_functional",
    "non_functional",
]
APPLIED = 0.0
TABLE = (
    f"fdr_classification_T={APPLIED}_cutoff=median_reflected_method=by_q=0.05.parquet"
)


def _write(directory: Path, ids=IDS, classes=CLASSES) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TABLE
    pl.DataFrame({"dna_sequence": ids, "fdr_class": classes}).write_parquet(path)
    provenance = {
        "cutoffs": [
            {
                "name": "median_reflected",
                "source_precision_threshold": 0.0000001,
                "applied_threshold": APPLIED,
            },
        ]
    }
    (directory / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(sha: str | None = None, **changes) -> dict:
    declaration = {
        "rule": "fdr_fixed_cutoff",
        "decimals": 6,
        "cutoff": "median_reflected",
        "method": "by",
        "q": 0.05,
        "input_dir": "snapshot",
    }
    return {
        "analysis": {
            "clamped": {"candidate_functional_threshold": {**declaration, **changes}}
        }
    }


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(fdr, "ROOT", tmp_path)
    return tmp_path / "snapshot", tmp_path


def _network(ids=IDS) -> ScalarNetwork:
    edges = [(left, right) for left, right in zip(ids, ids[1:], strict=False)]
    graph = Graph.from_tables(
        pl.DataFrame({"dna_sequence": ids, "fitness": FITNESS[: len(ids)]}),
        pl.DataFrame(
            edges, schema={"source": pl.String, "target": pl.String}, orient="row"
        ),
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness",),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def test_only_functional_class_is_functional_and_applied_threshold_is_used_unrounded(
    snapshot,
):
    directory, _ = snapshot
    settings = _settings(_write(directory))
    selection = cs.resolve_candidate_selection(
        settings, raw_fitness=FITNESS, wild_type="a"
    )
    assert selection.functional_sequences == frozenset({"a", "c"})
    assert selection.uncertain_count == 1
    assert selection.threshold.threshold == APPLIED
    assert selection.source["class_counts"] == {
        "functional": 2,
        "uncertain": 1,
        "non_functional": 3,
    }


def test_graph_keeps_uncertain_and_below_cutoff_neighbours_as_fringe_only(snapshot):
    directory, _ = snapshot
    settings = _settings(_write(directory))
    prepared, selection = cs.prepare_candidate_landscape(_network(), settings)
    # Edges with a functional endpoint: a-b, b-c, c-d. The low-low edges d-e and e-f drop.
    kept = set(
        prepared.network.graph[i]["dna_sequence"]
        for i in prepared.network.graph.node_indices()
    )
    assert kept == {"a", "b", "c", "d"}
    assert selection.functional_sequences == frozenset({"a", "c"})


def test_identity_records_cutoff_method_q_and_table_hash(snapshot):
    directory, _ = snapshot
    sha = _write(directory)
    selection = cs.resolve_candidate_selection(
        _settings(sha), raw_fitness=FITNESS, wild_type="a"
    )
    identity = cs.selection_identity(selection)
    assert identity["fdr_cutoff"] == "median_reflected"
    assert identity["fdr_method"] == "by"
    assert identity["fdr_q"] == 0.05
    assert identity["fdr_table_sha256"] == sha
    assert "fdr_loader_sha256" not in identity
    assert identity["functional_vertex_count"] == 2


def test_loader_source_change_is_provenance_not_scientific_identity(
    snapshot, monkeypatch
):
    directory, _ = snapshot
    settings = _settings(_write(directory))
    before = (
        cs.candidate_selection_fingerprint(settings),
        cs.candidate_selection_source(settings),
        cs.candidate_selection_identity(settings),
    )
    monkeypatch.setattr(fdr, "loader_source_sha256", lambda: "changed")
    after = (
        cs.candidate_selection_fingerprint(settings),
        cs.candidate_selection_source(settings),
        cs.candidate_selection_identity(settings),
    )
    assert before[0] != after[0]
    assert before[1] != after[1]
    assert before[2] == after[2]


def test_a_missing_snapshot_names_the_recipes(snapshot):
    with pytest.raises(FileNotFoundError, match="just all.*just fetch-data"):
        cs.resolve_candidate_selection(
            _settings(), raw_fitness=FITNESS, wild_type="a"
        )


def test_a_missing_table_names_the_recipes(snapshot):
    directory, _ = snapshot
    _write(directory)
    (directory / TABLE).unlink()
    with pytest.raises(FileNotFoundError, match="just classification-probability-scans"):
        cs.resolve_candidate_selection(
            _settings(), raw_fitness=FITNESS, wild_type="a"
        )


def test_the_table_digest_is_recorded_from_the_table_itself(snapshot):
    directory, _ = snapshot
    sha = _write(directory)
    selection = cs.resolve_candidate_selection(
        _settings(), raw_fitness=FITNESS, wild_type="a"
    )
    assert selection.source["table_sha256"] == sha


@pytest.mark.parametrize(
    "ids",
    [
        ["a", "a", "c", "d", "e", "f"],
        ["a", None, "c", "d", "e", "f"],
        ["a", "", "c", "d", "e", "f"],
    ],
)
def test_duplicate_null_or_empty_identities_are_refused(snapshot, ids):
    directory, _ = snapshot
    sha = _write(directory, ids=ids)
    with pytest.raises(ValueError, match="duplicate|null or empty"):
        cs.resolve_candidate_selection(
            _settings(sha), raw_fitness=FITNESS, wild_type="a"
        )


def test_unknown_class_is_refused(snapshot):
    directory, _ = snapshot
    sha = _write(directory, classes=CLASSES[:-1] + ["maybe"])
    with pytest.raises(ValueError, match="unknown classes"):
        cs.resolve_candidate_selection(
            _settings(sha), raw_fitness=FITNESS, wild_type="a"
        )


def test_wild_type_must_be_functional(snapshot):
    directory, _ = snapshot
    sha = _write(directory)
    with pytest.raises(ValueError, match="wild type"):
        cs.resolve_candidate_selection(
            _settings(sha), raw_fitness=FITNESS, wild_type="b"
        )


def test_classification_must_cover_exactly_the_graph_vertices(snapshot):
    directory, _ = snapshot
    # The table drops "f" and adds "z": the sets differ in both directions.
    sha = _write(directory, ids=IDS[:-1] + ["z"])
    settings = _settings(sha)
    with pytest.raises(ValueError, match="different sequences"):
        cs.prepare_candidate_landscape(_network(), settings)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"method": "bh"}, "method must be 'by'"),
        ({"q": 0.1}, "q must be the nominal 0.05"),
        ({"q": "0.05"}, "q must be the nominal"),
        ({"cutoff": "other"}, "cutoff must be one of"),
        ({"input_dir": ""}, "non-empty string"),
    ],
)
def test_declaration_is_validated(snapshot, change, message):
    directory, _ = snapshot
    sha = _write(directory)
    with pytest.raises(ValueError, match=message):
        cs.resolve_candidate_selection(
            _settings(sha, **change), raw_fitness=FITNESS, wild_type="a"
        )


def test_missing_declaration_field_is_refused(snapshot):
    directory, _ = snapshot
    settings = _settings(_write(directory))
    del settings["analysis"]["clamped"]["candidate_functional_threshold"]["q"]
    with pytest.raises(ValueError, match="Missing FDR keys.*q"):
        cs.resolve_candidate_selection(settings, raw_fitness=FITNESS, wild_type="a")
