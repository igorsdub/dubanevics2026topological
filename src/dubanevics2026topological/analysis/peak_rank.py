"""Peak-rank table of the Papkou et al. (2023) published peak set.

The peaks are the Reference peak set (``[clamped] peaks``, ``data/raw/papkou2023_peaks.csv``,
SHA-256 pinned in ``config/analysis.toml``): one 9-nt DNA sequence per peak. Each peak's
fitness is the WT-relative fitness of that sequence in the Full-graph population
(``load_full_network_fitness``). A peak missing from the population is excluded and
counted: ``excluded`` lists its DNA identity, and the summary reports how many.

Rank 1 is the highest fitness; ties break by ``dna_sequence`` so the order is stable.
A stop-codon peak has any of the stop symbols (``STOP_CODON_SYMBOLS``) in its translation.
Nothing here classifies a peak as synonymous: no synonymous annotation is drawn.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from ..rawdata.value_digest import value_digest
from .biological_annotations import STOP_CODON_SYMBOLS
from .codon_entropy_populations import CODON_TRANSLATION

PEAK_COLUMN = "dna"
TABLE_COLUMNS = (
    "rank",
    "dna_sequence",
    "aa_sequence",
    "fitness",
    "has_stop_codon",
)


def write_peak_rank_table(table: pl.DataFrame, summary: dict[str, Any], output_root: Path) -> None:
    """Write the peak rank table and its summary under the analysis output root."""
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table.write_parquet(tables / "peak_rank.parquet")
    (tables / "peak_rank_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def translate(dna: str) -> str:
    """Amino-acid sequence of a DNA sequence whose length is a multiple of 3."""
    if len(dna) % 3:
        raise ValueError(f"DNA sequence {dna!r} is not a whole number of codons.")
    try:
        return "".join(CODON_TRANSLATION[dna[i : i + 3]] for i in range(0, len(dna), 3))
    except KeyError as error:
        raise ValueError(f"DNA sequence {dna!r} has an unknown codon.") from error


def load_published_peaks(path: Path, sha256: str) -> list[str]:
    """DNA identities of the Reference peak set, verified against its pinned value digest (the sorted peak set)."""
    digest = value_digest(path)
    if digest != sha256:
        raise ValueError(f"{path} has value digest {digest}, expected {sha256}.")
    peaks = pl.read_csv(path)[PEAK_COLUMN].cast(pl.String).to_list()
    if len(set(peaks)) != len(peaks):
        raise ValueError("The Reference peak set lists a sequence twice.")
    return peaks


def build_peak_rank_table(
    population: pl.DataFrame, peaks: Sequence[str]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """``(table, summary)`` of ``peaks`` ranked by fitness in ``population``.

    ``population`` has ``dna_sequence`` and ``raw_fitness`` (every vertex once).
    The table has ``TABLE_COLUMNS`` sorted by fitness descending, rank 1 first;
    the summary counts the listed, ranked, excluded and stop-codon peaks.
    """
    if population["dna_sequence"].n_unique() != population.height:
        raise ValueError("The fitness population lists a sequence twice.")
    if not population["raw_fitness"].is_finite().all():
        raise ValueError("The fitness population has non-finite fitness.")
    fitness = dict(
        zip(population["dna_sequence"], population["raw_fitness"], strict=True)
    )
    present = [peak for peak in peaks if peak in fitness]
    excluded = sorted(peak for peak in peaks if peak not in fitness)
    rows = sorted(
        ((peak, fitness[peak]) for peak in present), key=lambda r: (-r[1], r[0])
    )
    table = pl.DataFrame(
        {
            "rank": list(range(1, len(rows) + 1)),
            "dna_sequence": [dna for dna, _ in rows],
            "aa_sequence": [translate(dna) for dna, _ in rows],
            "fitness": [value for _, value in rows],
        },
        schema={
            "rank": pl.Int64,
            "dna_sequence": pl.String,
            "aa_sequence": pl.String,
            "fitness": pl.Float64,
        },
    ).with_columns(
        pl.col("aa_sequence")
        .map_elements(
            lambda aa: any(symbol in STOP_CODON_SYMBOLS for symbol in aa),
            return_dtype=pl.Boolean,
        )
        .alias("has_stop_codon")
    )
    summary = {
        "peaks_listed": len(peaks),
        "peaks_ranked": table.height,
        "peaks_excluded": len(excluded),
        "excluded_sequences": excluded,
        "stop_codon_peaks": int(table["has_stop_codon"].sum()),
        "population_sequences": population.height,
    }
    return table.select(TABLE_COLUMNS), summary
