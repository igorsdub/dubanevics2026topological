"""Golden core-era results produced by the Topographer revision preceding v0.5.0."""

from pathlib import Path

import polars as pl
import pytest
from test_bootstrap_chunks import SCIENTIFIC, _network, _statistic
from test_bootstrap_smoothed_tree import DNA
from topographer.statistics import bootstrap

from dubanevics2026topological.simplification import bootstrap_smoothed_tree as module
from dubanevics2026topological.simplification.bootstrap_chunks import (
    ChunkedBootstrap,
    ChunkStore,
)

GOLDEN = Path(__file__).parent / "data" / "bootstrap_golden"


@pytest.mark.parametrize("series", ["observed", "flat"])
def test_legacy_golden_and_direct_measurements(tmp_path, series):
    observed = _network()
    network = (
        observed
        if series == "observed"
        else module._make_network(
            DNA,
            pl.DataFrame({"source": list("abcdef"), "target": list("bcdefg")}),
            pl.Series([module.FLAT_LANDSCAPE_CONSTANT] * 7).to_numpy(),
            pl.Series([0.05] * 7).to_numpy(),
        )
    )
    store = ChunkStore(
        tmp_path / "chunks",
        scientific=SCIENTIFIC,
        n_draws=30,
        seed=7,
        chunk_size=7,
        log=lambda _: None,
    )
    result = ChunkedBootstrap(store, log=lambda _: None).run(
        series, network, _statistic(observed)
    )
    legacy = bootstrap.combine(
        store.load(series), _statistic(observed).prepare(network), core_cutoff=0.95
    )

    def old(name):
        return pl.read_parquet(GOLDEN / f"{series}_{name}.parquet")

    assert legacy.same_leaf_frequency.equals(old("pair_frequency"))
    assert legacy.leaf_cores.equals(old("cores"))
    assert legacy.vertex_roles.equals(
        old("vertex_roles").rename({"peak_frequency": "leaf_frequency"})
    )
    expected = old("draw_stats").rename({"peak_count": "leaf_count"})
    if series == "flat":
        assert result.draw_stats.equals(expected)
    else:
        # Observed chunking changes only the last few bits of three distances.
        assert result.draw_stats.schema == expected.schema
        exact = ["draw", "leaf_count", "reproduced_pair_fraction"]
        assert result.draw_stats.select(exact).equals(expected.select(exact))
        for name in ("wasserstein_1", "bottleneck"):
            assert result.draw_stats[name].to_numpy() == pytest.approx(
                expected[name].to_numpy(), rel=1e-12, abs=0
            )
    assert result.nominal_hill_support.equals(old("nominal_hill_support"))
