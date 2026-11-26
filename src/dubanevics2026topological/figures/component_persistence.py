"""Descending-sweep view over cached ascending component-persistence pairs."""

from __future__ import annotations

import polars as pl

from ..analysis.component_persistence import (
    DESCENDING_PAIR_KINDS,
    PAIR_SCHEMA,
    _validate_pair_table,
)


def descending_component_pairs(ascending_pairs: pl.DataFrame) -> pl.DataFrame:
    """Express a validated ascending checkpoint using descending-sweep categories.

    Finite split pairs become ordinary peak H0 pairs and finite join pairs become
    relative valley H1 pairs. The global pair is oriented maximum to minimum.
    Scalar values, persistence, and finite-pair endpoint identities are retained.
    On tied global extrema, the checkpoint's deterministic endpoint choices are
    retained and swapped; these can differ from independently recomputing the
    public API on ``-fitness`` because canonical tie-breaking runs in that scalar
    order. Full-network global extrema are unique.
    """
    _validate_pair_table(ascending_pairs)
    rows = []
    for row in ascending_pairs.iter_rows(named=True):
        source_kind = row["pair_kind"]
        if source_kind in DESCENDING_PAIR_KINDS:
            kind, dimension = DESCENDING_PAIR_KINDS[source_kind]
        else:
            kind, dimension = source_kind, row["homology_dimension"]
            row = {
                **row,
                "birth_sequence": row["death_sequence"],
                "death_sequence": row["birth_sequence"],
                "birth_has_stop_codon": row["death_has_stop_codon"],
                "death_has_stop_codon": row["birth_has_stop_codon"],
                "birth_scalar": row["death_scalar"],
                "death_scalar": row["birth_scalar"],
            }
        rows.append({**row, "pair_kind": kind, "homology_dimension": dimension})

    result = (
        pl.DataFrame(rows, schema=PAIR_SCHEMA)
        if rows
        else pl.DataFrame(schema=PAIR_SCHEMA)
    )
    _validate_descending_view(result)
    return result


def _validate_descending_view(pairs: pl.DataFrame) -> None:
    """Check category dimensions, raw scalar directions, and pair lengths."""
    if pairs.schema != PAIR_SCHEMA:
        raise ValueError("Descending component view does not match the pair schema.")
    invalid = pairs.filter(
        ((pl.col("pair_kind") == "ordinary") & (pl.col("homology_dimension") != 0))
        | ((pl.col("pair_kind") == "relative") & (pl.col("homology_dimension") != 1))
        | (
            (pl.col("pair_kind") == "extended_positive")
            & (pl.col("homology_dimension") != 0)
        )
        | (
            (pl.col("pair_kind") == "ordinary")
            & (pl.col("birth_scalar") < pl.col("death_scalar"))
        )
        | (
            (pl.col("pair_kind") == "relative")
            & (pl.col("birth_scalar") > pl.col("death_scalar"))
        )
        | (
            (pl.col("pair_kind") == "extended_positive")
            & (pl.col("birth_scalar") < pl.col("death_scalar"))
        )
        | (
            (
                (pl.col("birth_scalar") - pl.col("death_scalar")).abs()
                - pl.col("persistence")
            ).abs()
            > 1e-10
        )
    )
    if invalid.height:
        raise ValueError(
            "Descending component pair categories or coordinates are invalid."
        )
