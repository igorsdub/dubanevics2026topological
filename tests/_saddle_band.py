"""Hand-built Full-graph population and ascending pair tables for the leading-saddle-band tests."""

from __future__ import annotations

import polars as pl

from dubanevics2026topological.analysis.component_persistence import PAIR_SCHEMA

DECLARATION = {"rule": "median_reflected", "decimals": 6}
# 21 sequences, raw fitness -1.0 to 1.0 in steps of 0.1 (rounded, so a literal in a
# pair table is the same float). Codon 1 alternates AAA (K) and GAA (E) with the
# index; codons 2 and 3 are leucine codons that vary with the index only to keep
# every sequence unique, so entropy is 0 at positions 2 and 3.
FITNESS = [round(-1.0 + 0.1 * i, 1) for i in range(21)]
_LEUCINE = ("TTA", "TTG", "CTT", "CTC", "CTA", "CTG")
SEQUENCES = [
    ("AAA" if i % 2 == 0 else "GAA") + _LEUCINE[i % 6] + _LEUCINE[(i // 6) % 6]
    for i in range(21)
]


def sequence_at(fitness: float) -> str:
    """The population sequence whose raw fitness is ``fitness``."""
    return SEQUENCES[FITNESS.index(round(fitness, 1))]


def _population() -> pl.DataFrame:
    return pl.DataFrame({"dna_sequence": SEQUENCES, "raw_fitness": FITNESS})


def isolated_graph(population: pl.DataFrame | None = None):
    """A Full-graph stand-in: one node per sequence and no edges."""
    import rustworkx

    frame = _population() if population is None else population
    graph = rustworkx.PyGraph()
    for sequence, fitness in zip(
        frame["dna_sequence"].to_list(),
        frame["raw_fitness"].to_list(),
        strict=True,
    ):
        graph.add_node({"dna_sequence": sequence, "s": float(fitness)})
    return graph


def _pairs(
    split_deaths, join_deaths, split_sequences=None, join_sequences=None
) -> pl.DataFrame:
    """Ascending pairs: relative (split saddle) and ordinary (join saddle) plus the global.

    Each death sequence is the population sequence at that fitness unless
    ``split_sequences`` or ``join_sequences`` names them.
    """
    split_sequences = split_sequences or [sequence_at(d) for d in split_deaths]
    join_sequences = join_sequences or [sequence_at(d) for d in join_deaths]
    rows = [("extended_positive", 0, SEQUENCES[0], SEQUENCES[-1], -1.0, 1.0)]
    for i, (death, sequence) in enumerate(zip(split_deaths, split_sequences)):
        rows.append(("relative", 1, f"PEAK{i}", sequence, 1.0, death))
    for i, (death, sequence) in enumerate(zip(join_deaths, join_sequences)):
        rows.append(("ordinary", 0, f"VALLEY{i}", sequence, -1.0, death))
    return pl.DataFrame(
        [
            {
                "pair_id": i,
                "pair_kind": kind,
                "homology_dimension": dim,
                "birth_sequence": birth_seq,
                "death_sequence": death_seq,
                "birth_has_stop_codon": False,
                "death_has_stop_codon": False,
                "birth_scalar": birth,
                "death_scalar": death,
                "persistence": abs(death - birth),
            }
            for i, (kind, dim, birth_seq, death_seq, birth, death) in enumerate(rows)
        ],
        schema=PAIR_SCHEMA,
    )


# join trailing -0.6 below split trailing -0.4
JOIN_BELOW = _pairs([0.3, 0.1, -0.4], [-0.9, -0.6, -0.7])
# split trailing -0.8 below join trailing -0.5
SPLIT_BELOW = _pairs([0.3, -0.8], [-0.9, -0.5])
