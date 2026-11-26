"""A five-vertex landscape and a stand-in for the Papkou workflow that builds it.

The comparison and peak-uncertainty stages build their full network with
``Papkou2023Workflow(...).build_network()``. The characterisation tests of those
stages (ADR 0013, Phase 3) replace the workflow with ``FakeWorkflow`` and keep
everything else real: the clamped-landscape preparation, the tree analyses, the
parquet and Topographer writers, and the checkpoint publication. They then see the
bytes the real stages write, on a landscape small enough for a unit test.

Topographer's Papkou tree analysis requires every DNA identity to have the same
length and every amino-acid identity too, so the vertices are three-base codons
and one-letter amino acids.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
from topographer.graph import Graph, ScalarNetwork

PEAK, VALLEY, SECOND_PEAK, LOW, LOW_TWO = "AAA", "AAC", "AAG", "AAT", "ACA"
SEQUENCES = [PEAK, VALLEY, SECOND_PEAK, LOW, LOW_TWO]
WILD_TYPE = LOW


def toy_network() -> ScalarNetwork:
    """Two peaks joined through a valley, with two low neighbours of the first."""
    vertices = pl.DataFrame(
        {
            "dna_sequence": SEQUENCES,
            "fitness": [0.5, 0.2, 0.6, 0.1, 0.08],
            "aa_sequence": ["K", "N", "K", "N", "T"],
            "has_stop_codon": [False] * 5,
        }
    )
    edges = pl.DataFrame(
        {
            "source": [PEAK, VALLEY, PEAK, PEAK],
            "target": [VALLEY, SECOND_PEAK, LOW, LOW_TWO],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "aa_sequence", "has_stop_codon"),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


class FakeWorkflow:
    """Stands in for ``Papkou2023Workflow``: the same keywords, the toy network."""

    built: ClassVar[list[str]] = []

    def __init__(
        self, *, landscape: str, fitness_wt: pl.DataFrame, fitness_diff: pl.DataFrame
    ) -> None:
        self.landscape = landscape
        self.fitness_wt = fitness_wt
        self.fitness_diff = fitness_diff

    def build_network(self) -> ScalarNetwork:
        FakeWorkflow.built.append(self.landscape)
        return toy_network()
