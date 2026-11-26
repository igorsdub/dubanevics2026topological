"""Positional amino-acid entropy of the Full graph and the Published graph.

The sweeps are ``compute_entropy_sweeps`` (the same Shannon entropy, in bits, of
the amino acid at each codon position, on sequences whose raw fitness is at or
above each threshold, that the main-text codon-entropy figure uses). The Full
graph supplies the shared threshold grid, as the candidate population does for
that figure. Nothing about ridges is computed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .codon_entropy import compute_entropy_sweeps
from .codon_entropy_populations import (
    CODON_TRANSLATION,
    _sequence_frame,
    _validate_sequence_translation,
)
from .ridge_entropy import translate

POPULATION_NAMES = ("full", "published")
_DNA_ALPHABET = frozenset("ACGT")


@dataclass(frozen=True)
class GraphCodonEntropy:
    """Entropy sweeps of the Full graph and the Published graph."""

    sweeps: pl.DataFrame
    sidecar: dict[str, Any]

    def save(self, directory: Path) -> None:
        """Write the sweep table and sidecar."""
        self.sweeps.write_parquet(directory / "entropy_sweeps.parquet")
        (directory / "summary.json").write_text(
            json.dumps(self.sidecar, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _population_frame(table: pl.DataFrame) -> pl.DataFrame:
    """Sequence frame the codon-entropy sweeps already accept."""
    required = {"dna_sequence", "raw_fitness"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Graph table is missing columns: {sorted(missing)}")
    if table["dna_sequence"].n_unique() != table.height:
        raise ValueError("Graph sequence identities must be unique.")
    dna = [str(value) for value in table["dna_sequence"].to_list()]
    frame = _sequence_frame(
        pl.DataFrame(
            {
                "dna_sequence": dna,
                "aa_sequence": [translate(sequence) for sequence in dna],
                "scalar": [float(value) for value in table["raw_fitness"].to_list()],
            },
            schema={
                "dna_sequence": pl.String,
                "aa_sequence": pl.String,
                "scalar": pl.Float64,
            },
        )
    )
    _validate_sequence_translation(
        frame,
        dna_alphabet=set(_DNA_ALPHABET),
        aa_alphabet=set(CODON_TRANSLATION.values()),
    )
    return frame


def population_digest(table: pl.DataFrame) -> str:
    """SHA-256 of the sorted sequence identities and raw fitness."""
    rows = table.select("dna_sequence", "raw_fitness").sort("dna_sequence")
    return hashlib.sha256(rows.write_csv().encode()).hexdigest()


def build_graph_codon_entropy(
    full: pl.DataFrame,
    published: pl.DataFrame,
    *,
    sweep_points: int,
) -> GraphCodonEntropy:
    """Entropy versus raw-fitness threshold for the two graphs.

    ``full`` and ``published`` have ``dna_sequence`` and ``raw_fitness``. The
    Published graph's sequences must be vertices of the Full graph. The shared
    grid is the one ``compute_entropy_sweeps`` builds from the Full graph.
    """
    if sweep_points < 1:
        raise ValueError("sweep_points must be positive.")
    if not set(published["dna_sequence"]) <= set(full["dna_sequence"]):
        raise ValueError(
            "Published graph sequences must be vertices of the Full graph."
        )
    frames = {
        "core": _population_frame(full),
        "published": _population_frame(published),
    }
    sweeps = compute_entropy_sweeps(frames, num_thresholds=sweep_points)
    stacked = pl.concat(
        [
            sweeps["core"]
            .with_columns(pl.lit("full").alias("population"))
            .select(
                "population",
                "threshold_scalar",
                "aa_position",
                "entropy_bits",
                "active_sequence_count",
            ),
            sweeps["published"]
            .with_columns(pl.lit("published").alias("population"))
            .select(
                "population",
                "threshold_scalar",
                "aa_position",
                "entropy_bits",
                "active_sequence_count",
            ),
        ]
    )
    sidecar = {
        "sweep_points": sweep_points,
        "counts": {
            name: frames[key].height
            for name, key in (("full", "core"), ("published", "published"))
        },
        "fitness_range": {
            name: None
            if frames[key].is_empty()
            else [
                float(frames[key]["fitness"].min()),
                float(frames[key]["fitness"].max()),
            ]
            for name, key in (("full", "core"), ("published", "published"))
        },
        "grid_size": sweeps["core"]["threshold_scalar"].n_unique(),
        "population_sha256": {
            "full": population_digest(full),
            "published": population_digest(published),
        },
    }
    return GraphCodonEntropy(stacked, sidecar)
