"""Read historical peak-core evidence without running core classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class LegacyBootstrapCore:
    """Archived real and flat core tables, hill decisions, and original summary."""

    cores: pl.DataFrame
    flat_cores: pl.DataFrame
    hills: pl.DataFrame
    flat_hills: pl.DataFrame
    summary: dict[str, Any]


def read_bootstrap_core(checkpoint: Path) -> LegacyBootstrapCore:
    """Read an archived checkpoint in place, preserving its original identities.

    The reader checks the fields needed to interpret historical decisions; it
    does not mutate the archive, construct cores, or assert scientific match.
    Integrity and compatibility remain the responsibility of the result loader.
    """
    checkpoint = Path(checkpoint)
    cores = pl.read_parquet(checkpoint / "cores.parquet")
    flat_cores = pl.read_parquet(checkpoint / "flat_cores.parquet")
    hills = pl.read_parquet(checkpoint / "hills.parquet")
    flat_hills = pl.read_parquet(checkpoint / "flat_hills.parquet")
    summary = json.loads((checkpoint / "summary.json").read_text(encoding="utf-8"))
    core_fields = {"core_id", "dna_sequence", "nominal_hill"}
    hill_fields = {
        "peak_sequence", "core_ids", "core_relation", "surviving_peak_sequence",
        "outcome", "nominal_persistence", "clears_band", "band_outcome",
    }
    for name, frame, required in (
        ("cores", cores, core_fields), ("flat_cores", flat_cores, core_fields),
        ("hills", hills, hill_fields), ("flat_hills", flat_hills, hill_fields),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Legacy {name} table lacks columns: {sorted(missing)}")
    if not isinstance(summary, dict):
        raise ValueError("Legacy summary must be a JSON object.")
    return LegacyBootstrapCore(cores, flat_cores, hills, flat_hills, summary)