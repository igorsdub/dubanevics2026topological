"""Combine unchanged bootstrap measurements without extracting peak cores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from dubanevics2026topological.simplification.cooccurrence_leaf import LeafCooccurrence


@dataclass(frozen=True, slots=True)
class BootstrapDistances:
    """Per-draw distances and nominal-hill support from one bootstrap series."""

    draw_stats: pl.DataFrame
    nominal_hill_support: pl.DataFrame | None
    n_draws: int
    seed: int | None
    tree_type: str


class CoreFreeLeafCooccurrence:
    """Delegate the pinned statistic's measurements, but finish without cores.

    Hill positions use the insertion order of ``nominal_hills``, as in the
    pinned Topographer statistic. Preparing delegates its validation and binding;
    neither the fingerprint nor the chunk counts and measurements change.
    """

    def __init__(self, statistic: LeafCooccurrence) -> None:
        self.statistic = statistic

    def prepare(self, network: Any) -> CoreFreeLeafCooccurrence:
        return CoreFreeLeafCooccurrence(self.statistic.prepare(network))

    def measure(self, network: Any) -> Any:
        return self.statistic.measure(network)

    def add(self, first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> Any:
        return self.statistic.add(first, second)

    def fingerprint(self) -> str:
        return self.statistic.fingerprint()

    def finish(
        self,
        counts: Mapping[str, np.ndarray],
        stats: np.ndarray,
        n_draws: int,
        *,
        seed: int | None = None,
    ) -> BootstrapDistances:
        """Finish the public statistic's counts without its core extraction."""
        # This also refuses an unprepared statistic, matching the upstream seam.
        self.fingerprint()
        if stats.shape != (n_draws, 4):
            raise ValueError(f"stats must have shape ({n_draws}, 4), got {stats.shape}.")
        draw_stats = pl.DataFrame(
            {
                "draw": pl.Series(np.arange(n_draws), dtype=pl.Int64),
                "leaf_count": pl.Series(stats[:, 0].astype(np.int64), dtype=pl.Int64),
                "wasserstein_1": pl.Series(stats[:, 1], dtype=pl.Float64),
                "bottleneck": pl.Series(stats[:, 2], dtype=pl.Float64),
                "reproduced_pair_fraction": pl.Series(stats[:, 3], dtype=pl.Float64),
            }
        )
        support = None
        if self.statistic.nominal_hills is not None:
            hill_ids = pl.Series("hill_id", list(self.statistic.nominal_hills))
            hill_counts = np.zeros(len(hill_ids), dtype=np.int64)
            hill_counts[counts["hills"][:, 0]] = counts["hills"][:, 1]
            support = pl.DataFrame(
                {
                    "hill_id": hill_ids,
                    "hill_support": pl.Series(hill_counts / n_draws, dtype=pl.Float64),
                }
            )
        return BootstrapDistances(
            draw_stats, support, n_draws, seed, self.statistic.tree_type
        )