"""Core-free aggregation preserves the pinned bootstrap measurement contract."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_bootstrap_chunks import _network, _statistic
from test_bootstrap_smoothed_tree import _toy
from dubanevics2026topological.simplification import (
    cooccurrence_leaf as _leaf_cooccurrence,
)
from topographer.statistics import bootstrap

from dubanevics2026topological.simplification import bootstrap_smoothed_tree as module
from dubanevics2026topological.simplification.bootstrap_aggregation import (
    CoreFreeLeafCooccurrence,
)
from dubanevics2026topological.simplification.bootstrap_chunks import (
    ChunkedBootstrap,
    ChunkStore,
    chunk_directory,
)

GOLDEN = Path(__file__).parent / "data" / "bootstrap_golden"
N_DRAWS = 30
SEED = 7


def _runner(path: Path, scientific: dict | None = None) -> ChunkedBootstrap:
    store = ChunkStore(
        path,
        scientific=scientific or {"config": {"seed": SEED, "n_replicates": N_DRAWS}},
        n_draws=N_DRAWS,
        seed=SEED,
        chunk_size=7,
        log=lambda _: None,
    )
    return ChunkedBootstrap(store, log=lambda _: None)


def test_adapter_preserves_prepared_statistic_fingerprint():
    network = _network()
    statistic = _statistic(network)
    assert CoreFreeLeafCooccurrence(statistic).prepare(network).fingerprint() == (
        statistic.prepare(network).fingerprint()
    )


def test_finish_skips_core_extraction_and_matches_current_and_golden_draws(
    tmp_path, monkeypatch
):
    network = _network()
    statistic = _statistic(network)
    runner = _runner(tmp_path / "chunks")
    direct = runner.run("observed", network, statistic)
    chunks = runner.store.load("observed")
    old = bootstrap.combine(chunks, statistic.prepare(network), core_cutoff=0.95)

    def refuse_cores(*args, **kwargs):
        raise AssertionError("core extraction was called")

    monkeypatch.setattr(_leaf_cooccurrence, "extract_cores", refuse_cores)
    without_cores = bootstrap.combine(
        chunks, CoreFreeLeafCooccurrence(statistic).prepare(network)
    )
    assert without_cores.draw_stats.equals(old.draw_stats)
    assert direct.draw_stats.equals(old.draw_stats)
    assert without_cores.nominal_hill_support.equals(old.nominal_hill_support)
    golden = pl.read_parquet(GOLDEN / "observed_draw_stats.parquet").rename(
        {"peak_count": "leaf_count"}
    )
    assert direct.draw_stats.schema == golden.schema
    exact = ["draw", "leaf_count", "reproduced_pair_fraction"]
    assert direct.draw_stats.select(exact).equals(golden.select(exact))
    for name in ("wasserstein_1", "bottleneck"):
        assert direct.draw_stats[name].to_numpy() == pytest.approx(
            golden[name].to_numpy(), rel=1e-12, abs=0
        )


@pytest.mark.parametrize("series", ["observed", "flat"])
def test_saved_chunks_match_old_combine_for_both_series(tmp_path, series):
    network = _network()
    statistic = _statistic(network)
    if series == "flat":
        prepared, _ = _toy()
        ids, edges = module._prepared_tables(prepared)
        network = module._make_network(
            ids, edges, np.zeros(len(ids)), np.full(len(ids), 0.05)
        )
    runner = _runner(tmp_path / "chunks")
    result = runner.run(series, network, statistic)
    chunks = runner.store.load(series)
    prepared_statistic = statistic.prepare(network)
    old = bootstrap.combine(chunks, prepared_statistic, core_cutoff=0.95)
    direct = bootstrap.combine(chunks, CoreFreeLeafCooccurrence(prepared_statistic))
    assert direct.draw_stats.equals(old.draw_stats)
    assert direct.nominal_hill_support.equals(old.nominal_hill_support)
    assert result.draw_stats.equals(old.draw_stats)
    assert result.nominal_hill_support.equals(old.nominal_hill_support)


def test_nonfirst_chunk_with_different_statistic_fingerprint_is_refused(
    tmp_path, monkeypatch
):
    network = _network()
    statistic = _statistic(network)
    runner = _runner(tmp_path / "chunks")
    runner.run("observed", network, statistic)
    load = runner.store.load

    def altered(name):
        chunks = load(name)
        if name == "observed":
            assert len(chunks) >= 2
            chunks[1] = dataclasses.replace(chunks[1], statistic_fingerprint="wrong")
        return chunks

    monkeypatch.setattr(runner.store, "load", altered)
    with pytest.raises(ValueError, match="chunk 1.*statistic fingerprint"):
        runner.run("observed", network, statistic)


def test_legacy_chunk_directory_restores_without_draws_or_manifest_changes(
    tmp_path, monkeypatch
):
    network = _network()
    statistic = _statistic(network)
    current = {"config": {"seed": SEED, "n_replicates": N_DRAWS}}
    historical = {
        "config": {
            **current["config"],
            "band_quantile": 0.95,
            "core_threshold": 0.95,
            "consensus_peak_frequency": 0.5,
        }
    }
    directory = chunk_directory(historical, tmp_path)
    first = _runner(directory, historical).run("observed", network, statistic)
    manifest = directory / "manifest.json"
    original = {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.name == "manifest.json" or path.suffix == ".npz"
    }
    assert chunk_directory(current, tmp_path) == directory

    monkeypatch.setenv("FITNESSTREE_BOOTSTRAP_NO_DRAW", "1")

    def refuse_draw(*args, **kwargs):
        raise AssertionError("unexpected draw")

    monkeypatch.setattr(bootstrap, "run", refuse_draw)
    resumed = _runner(chunk_directory(current, tmp_path), current).run(
        "observed", network, statistic
    )
    assert resumed.draw_stats.equals(first.draw_stats)
    assert resumed.nominal_hill_support.equals(first.nominal_hill_support)
    assert {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.name == "manifest.json" or path.suffix == ".npz"
    } == original
    assert json.loads(manifest.read_text())["scientific_fingerprint"] == historical
    mismatched = {"config": {**current["config"], "seed": SEED + 1}}
    assert chunk_directory(mismatched, tmp_path) != directory
