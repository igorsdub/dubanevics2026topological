"""Validated full-family comparison evidence shared by publication analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from topographer.statistics import compute_bh_qvalues


def global_contrasts(
    diff: pl.DataFrame,
    raw_fitness_by_id: dict[str, float],
    needed_edges: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], list[float]]:
    """Validate all tests, run one global BH correction, then materialize used edges."""
    required = {"dna_from", "dna_to", "aa_from", "aa_to", "s", "se", "pval"}
    missing_columns = required - set(diff.columns)
    if missing_columns:
        raise ValueError(
            f"fitness_diff missing required fields: {sorted(missing_columns)!r}."
        )
    if diff.is_empty():
        raise ValueError("fitness_diff contains no pairwise comparisons.")
    if diff.filter(
        pl.col("dna_from").is_null()
        | pl.col("dna_to").is_null()
        | (pl.col("dna_from") == "")
        | (pl.col("dna_to") == "")
    ).height:
        raise ValueError("fitness_diff has rows with missing DNA identities.")
    if diff.filter(pl.col("dna_from") == pl.col("dna_to")).height:
        raise ValueError("Self-loop pairwise comparison found in fitness_diff.")
    if diff.filter(
        pl.col("aa_from").is_null()
        | pl.col("aa_to").is_null()
        | (pl.col("aa_from") == "")
        | (pl.col("aa_to") == "")
    ).height:
        raise ValueError(
            "fitness_diff contains comparisons with missing amino-acid identities."
        )

    numeric = diff.select("s", "se", "pval").to_numpy().astype(float, copy=False)
    finite = np.isfinite(numeric)
    if not finite.all():
        row, column_index = np.argwhere(~finite)[0]
        column = ("s", "se", "pval")[int(column_index)]
        identity = diff.row(int(row), named=True)
        raise ValueError(
            f"fitness_diff {column} must be finite for {identity['dna_from']}|{identity['dna_to']} (row {int(row)})."
        )
    _deltas, standard_errors, pvalues = numeric.T
    if np.any(standard_errors <= 0):
        row = int(np.flatnonzero(standard_errors <= 0)[0])
        identity = diff.row(row, named=True)
        raise ValueError(
            f"fitness_diff standard error must be positive for {identity['dna_from']}|{identity['dna_to']} (row {row})."
        )
    invalid_p = (pvalues < 0.0) | (pvalues > 1.0)
    if invalid_p.any():
        row = int(np.flatnonzero(invalid_p)[0])
        identity = diff.row(row, named=True)
        raise ValueError(
            f"fitness_diff p-value must be in [0, 1] for {identity['dna_from']}|{identity['dna_to']} (row {row})."
        )

    keyed = diff.with_columns(
        pl.when(pl.col("dna_from") < pl.col("dna_to"))
        .then(pl.col("dna_from"))
        .otherwise(pl.col("dna_to"))
        .alias("edge_left"),
        pl.when(pl.col("dna_from") < pl.col("dna_to"))
        .then(pl.col("dna_to"))
        .otherwise(pl.col("dna_from"))
        .alias("edge_right"),
    )
    duplicates = (
        keyed.group_by("edge_left", "edge_right").len().filter(pl.col("len") > 1)
    )
    if duplicates.height:
        identities = [
            f"{row['edge_left']}|{row['edge_right']}"
            for row in duplicates.sort("edge_left", "edge_right").to_dicts()
        ]
        raise ValueError(
            "Duplicate unordered edge comparison is ambiguous: " + ", ".join(identities)
        )

    # Every input comparison participates, including zero deltas and edges
    # outside both selected landscapes.
    qvalues = compute_bh_qvalues(pvalues)
    keyed = keyed.with_columns(pl.Series("q_value", qvalues))

    if needed_edges:
        needed = pl.DataFrame(
            sorted(needed_edges),
            schema={"edge_left": pl.String, "edge_right": pl.String},
            orient="row",
        )
        matched = needed.join(keyed, on=["edge_left", "edge_right"], how="left")
        missing = matched.filter(pl.col("pval").is_null())
        if missing.height:
            identities = [
                f"{row['edge_left']}|{row['edge_right']}" for row in missing.to_dicts()
            ]
            raise ValueError(
                "No pairwise test comparison exists for prepared edges: "
                + ", ".join(identities)
            )
        contrast_by_edge = {}
        for row in matched.to_dicts():
            delta, se = float(row["s"]), float(row["se"])
            edge = (str(row["edge_left"]), str(row["edge_right"]))
            contrast_by_edge[edge] = {
                "edge_key": edge,
                "contrast_id": f"{row['dna_from']}>{row['dna_to']}",
                "reported_dna_from": row["dna_from"],
                "reported_dna_to": row["dna_to"],
                "reported_aa_from": row["aa_from"],
                "reported_aa_to": row["aa_to"],
                "reported_delta": delta,
                "pairwise_standard_error": se,
                "test_statistic": delta / se,
                "p_value": float(row["pval"]),
                "q_value": float(row["q_value"]),
            }
    else:
        contrast_by_edge = {}

    raw_frame = pl.DataFrame(
        {
            "dna_sequence": list(raw_fitness_by_id),
            "raw_fitness": list(raw_fitness_by_id.values()),
        },
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64},
    )
    signed = (
        keyed.select("dna_from", "dna_to", "s", "pval", "q_value")
        .join(
            raw_frame.rename(
                {"dna_sequence": "dna_from", "raw_fitness": "fitness_from"}
            ),
            on="dna_from",
            how="inner",
        )
        .join(
            raw_frame.rename({"dna_sequence": "dna_to", "raw_fitness": "fitness_to"}),
            on="dna_to",
            how="inner",
        )
        .with_columns(
            (pl.col("fitness_to") - pl.col("fitness_from")).alias("raw_delta")
        )
        .with_columns(
            pl.when(pl.col("s") > 0)
            .then(1)
            .when(pl.col("s") < 0)
            .then(-1)
            .otherwise(0)
            .alias("pairwise_sign"),
            pl.when(pl.col("raw_delta") > 0)
            .then(1)
            .when(pl.col("raw_delta") < 0)
            .then(-1)
            .otherwise(0)
            .alias("raw_fitness_sign"),
        )
        .filter(pl.col("pairwise_sign") != pl.col("raw_fitness_sign"))
    )
    disagreements = [
        {
            "contrast_id": f"{row['dna_from']}>{row['dna_to']}",
            "dna_from": row["dna_from"],
            "dna_to": row["dna_to"],
            "pairwise_delta": float(row["s"]),
            "raw_fitness_delta": float(row["raw_delta"]),
            "pairwise_sign": int(row["pairwise_sign"]),
            "raw_fitness_sign": int(row["raw_fitness_sign"]),
            "p_value": float(row["pval"]),
            "q_value": float(row["q_value"]),
        }
        for row in signed.select(
            "dna_from",
            "dna_to",
            "s",
            "raw_delta",
            "pairwise_sign",
            "raw_fitness_sign",
            "pval",
            "q_value",
        ).to_dicts()
    ]
    return (
        contrast_by_edge,
        disagreements,
        [float(standard_errors.min()), float(standard_errors.max())],
    )
