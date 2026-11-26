"""Write the golden bootstrap tables with the PREVIOUS Topographer (see README.md).

Run from an isolated environment pinned to the Topographer revision preceding v0.5.0, never from the
project environment::

    uv run --no-project --with "topographer @ git+file://<topographer checkout>@<revision preceding v0.5.0>" \
        python tests/data/bootstrap_golden/generate.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from topographer import Graph, ScalarNetwork, ScalarNetworkTopology
from dubanevics2026topological.simplification.cooccurrence_terminal import bootstrap_terminal_cooccurrence

OUT = Path(__file__).parent
DNA = list("abcdefg")
RAW = [-0.1, 0.5, -0.2, 0.4, -0.15, 0.6, 0.1]
SE = 0.05
FLAT = 0.0
HILLS = {"b": [1, 0], "d": [3], "f": [5, 6]}
N_DRAWS, SEED, THRESHOLD = 30, 7, 0.95


def network(raw: list[float]) -> ScalarNetwork:
    vertices = pl.DataFrame(
        {"dna_sequence": DNA, "raw_fitness": raw, "se": [SE] * len(DNA)},
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64, "se": pl.Float64},
    )
    edges = pl.DataFrame({"source": list("abcdef"), "target": list("bcdefg")})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("raw_fitness", "se"),
    )
    return ScalarNetwork(graph, scalar_attr="raw_fitness")


for series, raw in (("observed", RAW), ("flat", [FLAT] * len(DNA))):
    net = network(raw)
    # Both series read the REAL reference tree, as the pipeline does.
    reference = ScalarNetworkTopology(network(RAW)).compute_split_tree()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        found = bootstrap_terminal_cooccurrence(
            net,
            se_attr="se",
            tree_type="split",
            n_replicates=N_DRAWS,
            seed=SEED,
            core_cutoff=THRESHOLD,
            reference=reference,
            nominal_hills=HILLS,
            n_jobs=1,
        )
    for name in (
        "pair_frequency",
        "cores",
        "vertex_roles",
        "draw_stats",
        "nominal_hill_support",
    ):
        table = getattr(found, name)
        assert isinstance(table, pl.DataFrame), name
        table.write_parquet(OUT / f"{series}_{name}.parquet")
    np.testing.assert_equal(found.n_replicates, N_DRAWS)
    print(series, "ok")
