"""Regenerate the compact Parquet inputs for the 6-nt toy landscape."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from scipy.stats import norm

TARGET = Path(__file__).resolve().parent
WILD_TYPE = "TAAACA"
SHIFT = -0.8
SE = 0.03
FITNESS = {
    "TAATAA": -0.60,
    "CAAAAA": -0.470,
    "TAAAAA": 0.05,
    "TAAAAG": 0.08,
    "TAAACA": 0.80,
}
CODON_AA = {"TAA": "%", "CAA": "Q", "AAA": "K", "AAG": "K", "ACA": "T"}
AA = {dna: CODON_AA[dna[:3]] + CODON_AA[dna[3:]] for dna in FITNESS}
EDGES = [(dna, "TAAAAA") for dna in FITNESS if dna not in {"TAAAAA"}]


def main() -> None:
    shifted = {dna: value + SHIFT for dna, value in FITNESS.items()}
    ordered = sorted(FITNESS)
    wt_rows = [dna for dna in ordered if dna != WILD_TYPE]
    wt = pl.DataFrame(
        {
            "dna_from": [WILD_TYPE] * len(wt_rows),
            "dna_to": wt_rows,
            "aa_from": [AA[WILD_TYPE]] * len(wt_rows),
            "aa_to": [AA[dna] for dna in wt_rows],
            "s": [shifted[dna] for dna in wt_rows],
            "se": [SE] * len(wt_rows),
            "pval": [float(2 * norm.sf(abs(shifted[dna]) / SE)) for dna in wt_rows],
        }
    )
    diff = pl.DataFrame(
        {
            "dna_from": [a for a, _ in EDGES],
            "dna_to": [b for _, b in EDGES],
            "aa_from": [AA[a] for a, _ in EDGES],
            "aa_to": [AA[b] for _, b in EDGES],
            "s": [shifted[b] - shifted[a] for a, b in EDGES],
            "se": [SE] * len(EDGES),
            "pval": [
                float(2 * norm.sf(abs(shifted[b] - shifted[a]) / SE)) for a, b in EDGES
            ],
        }
    )
    stops = {"TAA", "TAG", "TGA"}
    stop_count = {dna: sum(dna[i : i + 3] in stops for i in (0, 3)) for dna in ordered}
    growth = pl.DataFrame(
        {
            "dna_sequence": ordered,
            "sequence_id": [f"toy-{i:02d}" for i in range(len(ordered))],
            "r": [1.0 + shifted[dna] for dna in ordered],
            "s": [shifted[dna] for dna in ordered],
            "se_pooled": [SE] * len(ordered),
            "r_replicate_mean": [1.0 + shifted[dna] for dna in ordered],
            "r_replicate_sd": [SE] * len(ordered),
            "se_replicate": [SE] * len(ordered),
            "n_replicates": [3] * len(ordered),
            "status": ["converged"] * len(ordered),
            "stop_codon_count": [stop_count[dna] for dna in ordered],
        }
    )
    wt.write_parquet(TARGET / "fitness_wt.parquet")
    diff.write_parquet(TARGET / "fitness_diff.parquet")
    growth.write_parquet(TARGET / "growth_rates.parquet")


if __name__ == "__main__":
    main()
