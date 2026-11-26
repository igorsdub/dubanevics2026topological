"""The Candidate selection: the Functional class (active) or the median rule.

A real classification checkpoint is published into a temporary directory from a
seeded two-bell fixture; nothing here reads ``data/raw/``.
"""

from __future__ import annotations

import copy
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from test_functional_classification import CONFIG, PUBLISHED, growth_table
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.analysis import sequence_growth_rates as growth
from dubanevics2026topological.landscape import selection as cs
from dubanevics2026topological.threshold import functional_classification as fc

MIXTURE = {"rule": "mixture_quantile", "decimals": 6}
MEDIAN = {"rule": "median_reflected", "decimals": 6}


def _settings(declaration: dict[str, Any], **config: Any) -> dict[str, Any]:
    return {
        "analysis": {
            fc.SECTION: {**CONFIG, **config},
            "clamped": {
                "published_functional_threshold": PUBLISHED,
                "candidate_functional_threshold": declaration,
            },
        }
    }


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    # A wide standard error leaves an Uncertain class around T.
    return growth_table([(-1.0, 0.15, 400, 60, 30), (0.0, 0.10, 300, 0, 0)], se=0.06)


@pytest.fixture
def published(
    table: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """Publish a classification checkpoint under ``tmp_path`` and point the module at it."""
    path = tmp_path / "growth_rates.parquet"
    table.write_parquet(path)
    monkeypatch.setattr(fc, "ROOT", Path(tmp_path.anchor))
    monkeypatch.setattr(fc, "growth_rates_path", lambda context=None: path)
    monkeypatch.setattr(
        fc, "checkpoint_dir", lambda context=None: tmp_path / "checkpoints" / "fc"
    )
    monkeypatch.setattr(
        fc, "export_dir", lambda context=None: tmp_path / "analysis" / "fc"
    )
    monkeypatch.setattr(
        growth,
        "load_cached_sequence_growth_rates",
        lambda context=None: SimpleNamespace(growth_rates=pl.read_parquet(path)),
    )
    result = fc.compute_functional_classification(_settings(MIXTURE))
    return SimpleNamespace(result=result, export=tmp_path / "analysis" / "fc")


def _path_network(identities: list[str], raw: list[float]) -> ScalarNetwork:
    graph = Graph.from_tables(
        pl.DataFrame({"dna_sequence": identities, "fitness": raw}),
        pl.DataFrame(
            {"source": identities[:-1], "target": identities[1:]},
            schema={"source": pl.String, "target": pl.String},
        ),
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness",),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def _classes(result: Any) -> dict[str, list[str]]:
    frame = result.classification
    return {
        name: frame.filter(pl.col("functional_class") == name)["dna_sequence"].to_list()
        for name in fc.CLASSES
    }


def test_the_mixture_rule_selects_the_functional_class(published) -> None:
    settings = _settings(MIXTURE)
    selection = cs.resolve_candidate_selection(settings, raw_fitness=[-1.0, 0.0, 1.0])
    classes = _classes(published.result)
    summary = published.result.summary

    assert selection.rule == "mixture_quantile"
    assert selection.functional_sequences == frozenset(classes["functional"])
    assert selection.threshold.threshold == summary["threshold"]["rounded"]
    assert selection.threshold.unrounded == summary["threshold"]["unrounded"]
    assert selection.threshold.rule == "mixture_quantile"
    # The Full-graph numbers keep their meaning beside T.
    assert (selection.threshold.median, selection.threshold.global_minimum) == (
        0.0,
        -1.0,
    )
    assert selection.uncertain_count == len(classes["uncertain"]) > 0
    assert selection.classes is not None
    assert selection.classes[classes["uncertain"][0]] == "uncertain"
    record = selection.to_json()
    assert record["functional_vertex_count"] == len(classes["functional"])
    assert record["uncertain_vertex_count"] == len(classes["uncertain"])
    assert record["rule"] == "mixture_quantile"
    assert record["rounded_threshold"] == summary["threshold"]["rounded"]


def test_the_source_records_the_checkpoint_identity(published) -> None:
    source = cs.resolve_candidate_selection(
        _settings(MIXTURE), raw_fitness=[0.0, 1.0]
    ).source
    assert source["rule"] == "mixture_quantile"
    assert source["stage"] == "functional-classification"
    assert len(source["fingerprint"]) == 64
    assert {"classification.parquet", "summary.json"} <= set(source["artefacts"])
    assert source == cs.candidate_selection_source(_settings(MIXTURE))


def test_a_rerun_classification_changes_the_source(
    published, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = cs.candidate_selection_source(_settings(MIXTURE))
    changed = _settings(MIXTURE, quantile=0.99)
    fc.compute_functional_classification(changed)
    after = cs.candidate_selection_source(changed)
    assert after["fingerprint"] != before["fingerprint"]
    assert after["artefacts"]["summary.json"] != before["artefacts"]["summary.json"]
    # The previous quantile keeps its own folder.
    assert cs.candidate_selection_source(_settings(MIXTURE)) == before


def test_the_median_rule_is_the_comparison_and_selects_by_raw_fitness() -> None:
    selection = cs.resolve_candidate_selection(
        _settings(MEDIAN), raw_fitness=[-1.0, -0.6, -0.5, -0.2, 0.3]
    )
    assert selection.rule == "median_reflected"
    assert selection.functional_sequences is None
    assert selection.classes is None
    assert selection.threshold.threshold == 0.0
    assert selection.source == {"rule": "median_reflected"}
    assert selection.to_json()["functional_vertex_count"] is None
    assert selection.to_json()["uncertain_vertex_count"] is None
    # No checkpoint is read for the comparison rule.
    assert cs.candidate_selection_source(_settings(MEDIAN)) == {
        "rule": "median_reflected"
    }


def test_the_mixture_rule_fails_closed_without_a_current_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fc, "ROOT", Path(tmp_path.anchor))
    monkeypatch.setattr(
        fc, "growth_rates_path", lambda context=None: tmp_path / "g.parquet"
    )
    (tmp_path / "g.parquet").write_bytes(b"")
    monkeypatch.setattr(
        fc, "checkpoint_dir", lambda context=None: tmp_path / "checkpoints" / "fc"
    )
    settings = _settings(MIXTURE)
    for call in (
        lambda: cs.resolve_candidate_selection(settings, raw_fitness=[0.0, 1.0]),
        lambda: cs.candidate_selection_source(settings),
        lambda: cs.candidate_selection_identity(settings),
        lambda: cs.candidate_selection_fingerprint(settings),
    ):
        with pytest.raises(FileNotFoundError, match="just functional-classification"):
            call()


def test_a_stale_checkpoint_is_refused(published) -> None:
    stale = _settings(MIXTURE, n_init=2)
    with pytest.raises(FileNotFoundError, match="just functional-classification"):
        cs.resolve_candidate_selection(stale, raw_fitness=[0.0, 1.0])


def test_decimals_that_disagree_with_the_classification_are_refused(published) -> None:
    # The classification rounds T to 3 decimals; the declaration says 6, so the
    # reported T would differ from the one the classes were cut at.
    coarse = _settings(MIXTURE, decimals=3)
    fc.compute_functional_classification(coarse)
    with pytest.raises(ValueError, match="decimals"):
        cs.resolve_candidate_selection(coarse, raw_fitness=[0.0, 1.0])


def test_the_fingerprint_carries_the_rounded_threshold_and_the_source(
    published,
) -> None:
    value = cs.candidate_selection_fingerprint(_settings(MIXTURE))
    assert (
        value["rounded_threshold"] == published.result.summary["threshold"]["rounded"]
    )
    assert value["source"] == cs.candidate_selection_source(_settings(MIXTURE))


def test_the_identity_is_a_hash_of_the_functional_set_not_of_file_bytes(
    published,
) -> None:
    identity = cs.candidate_selection_identity(_settings(MIXTURE))
    classes = _classes(published.result)
    assert identity["functional_vertex_count"] == len(classes["functional"])
    assert (
        identity["rounded_threshold"]
        == published.result.summary["threshold"]["rounded"]
    )
    assert len(identity["functional_set_sha256"]) == 64
    assert "fingerprint" not in identity and "artefacts" not in identity
    assert cs.candidate_selection_identity(_settings(MEDIAN)) == {
        "rule": "median_reflected"
    }


def _largest_run(identities: list[str], functional: set[str]) -> set[str]:
    """Largest component of a path whose edges need a Functional endpoint."""
    runs: list[list[str]] = []
    for left, right in pairwise(identities):
        if left in functional or right in functional:
            if runs and runs[-1][-1] == left:
                runs[-1].append(right)
            else:
                runs.append([left, right])
    return set(min(runs, key=lambda run: (-len(run), min(run))))


def test_prepare_candidate_landscape_keeps_functional_vertices_and_neighbours(
    published, table: pl.DataFrame
) -> None:
    identities = table["dna_sequence"].to_list()
    classes = _classes(published.result)
    functional = set(classes["functional"])
    network = _path_network(identities, table["s"].to_list())
    prepared, selection = cs.prepare_candidate_landscape(network, _settings(MIXTURE))

    kept = _largest_run(identities, functional)
    assert selection.rule == "mixture_quantile"
    assert set(prepared.network.vertex_ids) == kept
    assert prepared.diagnostics["edge_selection_rule"] == (
        "at_least_one_endpoint_functional_class"
    )
    assert prepared.diagnostics["functional_vertex_count"] == len(functional)
    assert prepared.diagnostics["uncertain_vertex_count"] == len(classes["uncertain"])
    assert prepared.threshold == selection.threshold.threshold
    # Neighbours of a Functional vertex stay although they are not Functional.
    assert kept - functional
    excluded = prepared.excluded_vertices
    assert set(excluded["dna_sequence"]) == set(identities) - kept
    labelled = dict(
        zip(excluded["dna_sequence"], excluded["functional_class"], strict=True)
    )
    expected = {name: label for label, names in classes.items() for name in names}
    assert all(labelled[name] == expected[name] for name in labelled)


def test_prepare_candidate_landscape_under_the_median_rule_is_the_raw_edge_test() -> (
    None
):
    network = _path_network(list("abcd"), [-1.0, -0.2, 0.3, -0.8])
    prepared, selection = cs.prepare_candidate_landscape(network, _settings(MEDIAN))
    # g = -1.0, m = -0.5, T = 0.0: only c is at or above T.
    assert selection.threshold.threshold == 0.0
    assert set(prepared.network.vertex_ids) == {"b", "c", "d"}
    assert prepared.diagnostics["edge_selection_rule"] == (
        "at_least_one_endpoint_raw_fitness_ge_threshold"
    )


def test_resolve_candidate_threshold_delegates_to_the_resolver(published) -> None:
    from dubanevics2026topological.threshold.functional_threshold import (
        resolve_candidate_threshold,
    )

    network = _path_network(["a", "b"], [-1.0, 0.5])
    threshold = resolve_candidate_threshold(network, _settings(MIXTURE))
    assert threshold.rule == "mixture_quantile"
    assert threshold.threshold == published.result.summary["threshold"]["rounded"]
    assert threshold.global_minimum == -1.0


def test_uncertain_sequences_are_exported_with_the_checkpoint(published) -> None:
    exported = pl.read_parquet(published.export / fc.UNCERTAIN_FILE)
    classes = _classes(published.result)
    assert exported.columns == list(fc.UNCERTAIN_COLUMNS)
    assert sorted(exported["dna_sequence"]) == sorted(classes["uncertain"])
    probability = exported["functional_probability"]
    assert ((probability > 0.05) & (probability < 0.95)).all()
    assert fc.UNCERTAIN_FILE in fc.ARTEFACTS


def test_the_comparison_rule_leaves_the_classification_checkpoint_current(
    published,
) -> None:
    # Switching the active Selection rule must not recompute the classification.
    median = _settings(MEDIAN)
    mixture = _settings(MIXTURE)
    assert (
        fc.functional_classification_stage(median).settings
        == fc.functional_classification_stage(mixture).settings
    )
    assert copy.deepcopy(median) != mixture
    assert (
        fc.load_cached_functional_classification(median).summary
        == published.result.summary
    )


def test_the_lock_free_read_never_takes_the_checkpoint_lock(
    published, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological.analysis import produce_or_load as pol

    def refuse(*_: Any, **__: Any) -> Any:
        raise RuntimeError("another process holds the checkpoint")

    monkeypatch.setattr(pol, "checkpoint_lock", refuse)
    with pytest.raises(RuntimeError, match="holds the checkpoint"):
        cs.resolve_candidate_selection(_settings(MIXTURE), raw_fitness=[0.0, 1.0])
    selection = cs.resolve_candidate_selection(
        _settings(MIXTURE), raw_fitness=[0.0, 1.0], lock=False
    )
    assert selection.source["stage"] == "functional-classification"


def test_a_resolved_selection_is_not_read_again(
    published, table: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(MIXTURE)
    selection = cs.resolve_candidate_selection(settings, raw_fitness=[0.0, 1.0])
    identity = cs.candidate_selection_identity(settings)

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("the classification checkpoint was read again")

    monkeypatch.setattr(fc, "load_cached_functional_classification_with_source", refuse)
    assert cs.candidate_selection_identity(settings, selection) == identity
    network = _path_network(table["dna_sequence"].to_list(), table["s"].to_list())
    prepared, again = cs.prepare_candidate_landscape(
        network, settings, selection=selection
    )
    assert again is selection
    functional = set(_classes(published.result)["functional"])
    assert prepared.diagnostics["selected_component_functional_vertex_count"] == len(
        set(prepared.network.vertex_ids) & functional
    )
    assert (
        prepared.diagnostics["selected_component_functional_vertex_count"]
        <= prepared.diagnostics["functional_vertex_count"]
    )


def test_the_wild_type_must_be_functional(published) -> None:
    settings = _settings(MIXTURE)
    classes = _classes(published.result)
    ok = cs.resolve_candidate_selection(
        settings, raw_fitness=[0.0, 1.0], wild_type=classes["functional"][0]
    )
    assert ok.rule == "mixture_quantile"
    for wild_type in (classes["non_functional"][0], classes["uncertain"][0], "ABSENT"):
        with pytest.raises(cs.WildTypeNotFunctionalError, match="wild type"):
            cs.resolve_candidate_selection(
                settings, raw_fitness=[0.0, 1.0], wild_type=wild_type
            )
    # The comparison rule has no classification, so nothing to assert.
    cs.resolve_candidate_selection(
        _settings(MEDIAN), raw_fitness=[0.0, 1.0], wild_type="ABSENT"
    )
