"""Raw-fitness population for full-network figure context."""

from __future__ import annotations

from typing import Any

import polars as pl
from dubanevics2026topological.papkou2023 import build_papkou2023_fitness_vertex_table

from ..analysis import ROOT, dataset_input_path, load_settings


def load_full_network_fitness(
    settings: dict[str, Any] | None = None,
    *,
    context: Any | None = None,
) -> pl.DataFrame:
    """Load all full-network vertices, including WT, without graph or topology work.

    The public Papkou vertex builder is also used by its full-network builder.
    It adds the WT reference at zero and preserves every observed sequence;
    no threshold, critical-point selection, or stop-codon exclusion is applied.
    """
    settings = settings or load_settings(context=context)
    dataset = settings["analysis"]["dataset"]
    if dataset["landscape"] != "full":
        raise ValueError("Full-network fitness requires the declared full landscape.")
    vertices = build_papkou2023_fitness_vertex_table(
        pl.read_parquet(dataset_input_path(dataset["fitness_wt"], context, root=ROOT))
    )
    population = vertices.select(
        pl.col("dna_sequence").cast(pl.String),
        pl.col("s").cast(pl.Float64).alias("raw_fitness"),
    )
    if population.is_empty():
        raise ValueError("Full-network fitness population must not be empty.")
    if any(population.null_count().row(0)):
        raise ValueError(
            "Full-network sequence identities and fitness must not be null."
        )
    if population["dna_sequence"].n_unique() != population.height:
        raise ValueError("Full-network sequence identities must be unique.")
    if not population["raw_fitness"].is_finite().all():
        raise ValueError("Full-network raw fitness must be finite.")
    return population
