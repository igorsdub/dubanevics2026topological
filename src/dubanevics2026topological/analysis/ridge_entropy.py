"""Positional codon entropy sweeps of the core, KEM and TCL populations of one saved tree.

The populations are read from the DNA-keyed membership table of a saved ``ridges``
product: the core is every selected-graph sequence of the tree's state (the
Candidate population including fringe), and each ridge is the sequences carrying
its ``ridge_name``. The sweeps are the existing ``compute_entropy_sweeps``, one
common raw-fitness grid for the three populations; nothing is redefined here.
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
from .ridges import RIDGE_NAMES

POPULATION_NAMES = ("core", *RIDGE_NAMES)
SWEEPS_FILE = "entropy_sweeps.parquet"
POPULATIONS_FILE = "populations.parquet"
SIDECAR_FILE = "summary.json"
_DNA_ALPHABET = frozenset("ACGT")


def translate(dna: str) -> str:
    """Amino-acid sequence of a 9-nt DNA sequence (stops are ``%``, ``^``, ``&``)."""
    return "".join(CODON_TRANSLATION[dna[i : i + 3]] for i in range(0, len(dna), 3))


@dataclass(frozen=True)
class RidgeEntropy:
    """Populations, entropy sweeps and sidecar of one saved tree."""

    populations: pl.DataFrame
    sweeps: pl.DataFrame
    sidecar: dict[str, Any]

    def save(self, directory: Path) -> None:
        """Write the three artifacts of a saved ``ridge_entropy_sweeps`` product."""
        self.populations.write_parquet(directory / POPULATIONS_FILE)
        self.sweeps.write_parquet(directory / SWEEPS_FILE)
        (directory / SIDECAR_FILE).write_text(
            json.dumps(self.sidecar, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_artifacts(cls, artifacts: dict[str, Path]) -> RidgeEntropy:
        """Read the artifacts of a saved ``ridge_entropy_sweeps`` product."""
        return cls(
            pl.read_parquet(artifacts[POPULATIONS_FILE]),
            pl.read_parquet(artifacts[SWEEPS_FILE]),
            json.loads(Path(artifacts[SIDECAR_FILE]).read_text(encoding="utf-8")),
        )


def population_members(members: pl.DataFrame) -> dict[str, list[str]]:
    """DNA sequences of ``core`` (all rows), ``kem`` and ``tcl`` from a membership table."""
    result = {"core": members["member_dna_sequence"].to_list()}
    for name in RIDGE_NAMES:
        result[name] = members.filter(pl.col("ridge_name") == name)[
            "member_dna_sequence"
        ].to_list()
    return result


def build_ridge_entropy(members: pl.DataFrame, *, sweep_points: int) -> RidgeEntropy:
    """Entropy sweeps of core, KEM and TCL over one common grid of raw fitness."""
    if members["member_dna_sequence"].n_unique() != members.height:
        raise ValueError("Membership table repeats a sequence.")
    fitness = dict(
        zip(members["member_dna_sequence"], members["raw_fitness"], strict=True)
    )
    frames: dict[str, pl.DataFrame] = {}
    for name, dna_list in population_members(members).items():
        frame = _sequence_frame(
            pl.DataFrame(
                {
                    "dna_sequence": dna_list,
                    "aa_sequence": [translate(dna) for dna in dna_list],
                    "scalar": [fitness[dna] for dna in dna_list],
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
        frames[name] = frame
    sweeps = compute_entropy_sweeps(frames, num_thresholds=sweep_points)
    populations = pl.concat(
        [
            frame.select("dna_sequence", "aa_sequence", "raw_fitness").with_columns(
                pl.lit(name).alias("population")
            )
            for name, frame in frames.items()
        ]
    ).select("population", "dna_sequence", "aa_sequence", "raw_fitness")
    sidecar = {
        "sweep_points": sweep_points,
        "counts": {name: frame.height for name, frame in frames.items()},
        "fitness_range": {
            name: None
            if frame.is_empty()
            else [float(frame["fitness"].min()), float(frame["fitness"].max())]
            for name, frame in frames.items()
        },
        "grid_size": sweeps["core"]["threshold_scalar"].n_unique(),
        "sequence_identity_sha256": {
            name: hashlib.sha256(
                "\n".join(sorted(frame["dna_sequence"])).encode()
            ).hexdigest()
            for name, frame in frames.items()
        },
        "ridges_present": all(frames[name].height for name in RIDGE_NAMES),
    }
    return RidgeEntropy(
        populations,
        pl.concat(
            [
                frame.with_columns(pl.lit(name).alias("population")).select(
                    "population",
                    "threshold_scalar",
                    "aa_position",
                    "entropy_bits",
                    "active_sequence_count",
                )
                for name, frame in sweeps.items()
            ]
        ),
        sidecar,
    )
