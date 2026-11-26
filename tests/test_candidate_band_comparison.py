"""The Candidate split-tree versus band-tree table, on a hand-built tree.

The branch rows are the balanced tree from ``tests/_branch_trees.py`` after
``decompose_view`` (see ``tests/test_branch_decomposition.py``): vertex counts
``[2, 3, 1, 1, 1, 1, 1]``, Horton–Strahler orders 1, 1, 1, 1, 2, 2, 3, and
ordinary H0 persistences 9, 3, 4 and 2. Hamming is computed here by
Topographer's ``hamming_distance`` on the endpoints only.
"""

from __future__ import annotations

import csv
import json

import polars as pl

from dubanevics2026topological.analysis.branch_decomposition import BRANCH_SCHEMA
from dubanevics2026topological.tables.candidate_band_comparison import (
    TreeColumn,
    comparison_rows,
    measure_tree,
    readme_text,
    write_table,
)

# Endpoint sequences. Identities are the balanced tree's. Amino-acid "*" is a
# stop. Nucleotide strings are equal length so hamming_distance applies; the
# fixture identities themselves are not equal length.
AMINO = {
    "a": "*",
    "b": "M",
    "c": "M",
    "d": "M",
    "s1": "M",
    "s2": "M",
    "s3": "M",
    "root": "M",
}
NUCLEOTIDE = {name: "AAA" for name in AMINO}
NUCLEOTIDE["b"] = "AAT"


def _branches(rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(rows)
    for name, dtype in BRANCH_SCHEMA.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(name))
    return frame.select(list(BRANCH_SCHEMA)).cast(BRANCH_SCHEMA)


def _nominal() -> TreeColumn:
    # One row per edge of the balanced tree. Vertex counts include interior
    # vertices (ra on a, rb1 and rb2 on b) and exclude the lower endpoint.
    rows = [
        ("a", "s1", True, 1, 2, 9.0),
        ("b", "s1", True, 1, 3, 8.0),
        ("c", "s2", True, 1, 1, 7.0),
        ("d", "s2", True, 1, 1, 6.0),
        ("s1", "s3", False, 2, 1, 5.0),
        ("s2", "s3", False, 2, 1, 4.0),
        ("s3", "root", False, 3, 1, 3.0),
    ]
    table = _branches(
        [
            {
                "branch_id": upper,
                "upper_sequence": upper,
                "lower_sequence": lower,
                "is_terminal": terminal,
                "horton_strahler_order": order,
                "branch_vertex_count": count,
                "birth_fitness": birth,
                "member_sequences": [upper],
            }
            for upper, lower, terminal, order, count, birth in rows
        ]
    )
    return TreeColumn(
        branches=table,
        persistence=[9.0, 3.0, 4.0, 2.0],
        amino_acid=AMINO,
        nucleotide=NUCLEOTIDE,
    )


def _without_stop_peak() -> TreeColumn:
    """The band tree after absorbing terminal peak ``a`` (vertex count 2)."""
    kept = _nominal().branches.filter(pl.col("branch_id") != "a")
    return TreeColumn(
        branches=kept,
        persistence=[3.0, 4.0, 2.0],
        amino_acid=AMINO,
        nucleotide=NUCLEOTIDE,
    )


def test_nominal_balanced_tree_matches_the_decomposition():
    measured = measure_tree(_nominal())
    assert measured["number of peaks; number of saddles"] == "4; 3"
    assert measured["branch vertex count: min / median / mean / max"] == (
        "1 / 1 / 1.42857142857 / 3"
    )
    # Only a–s1 joins "*" with "M". A path through ra would be a different
    # number; the mean is the mean of the seven endpoint distances, 1/7.
    assert (
        measured["amino-acid Hamming distance along a branch: mean / median"]
        == "0.142857142857 / 0"
    )
    # b (AAT) against s1 (AAA) is the only nucleotide mismatch.
    assert (
        measured["nucleotide Hamming distance along a branch: mean / median"]
        == "0.142857142857 / 0"
    )
    assert (
        measured["Horton-Strahler order 1 / 2 / 3 / 4 branch counts"] == "4 / 2 / 1 / 0"
    )
    assert measured["branch persistence: median / max"] == "3.5 / 9"
    assert measured["stop-codon genotypes on peaks"] == "1"


def test_absorbing_the_stop_peak_drops_it_from_the_band_column():
    rows = comparison_rows(_nominal(), _without_stop_peak())
    by_name = {row["statistic"]: row for row in rows}
    assert by_name["number of peaks; number of saddles"]["split_tree"] == "4; 3"
    assert (
        by_name["number of peaks; number of saddles"]["band_simplified_tree"] == "3; 3"
    )
    assert by_name["stop-codon genotypes on peaks"]["band_simplified_tree"] == "0"
    assert by_name["branch vertex count: min / median / mean / max"][
        "band_simplified_tree"
    ].endswith("/ 3")


def test_endpoint_distance_ignores_a_longer_member_path(tmp_path):
    """Two members between the endpoints would change a path length; they do not."""
    column = TreeColumn(
        branches=_branches(
            [
                {
                    "branch_id": "peak",
                    "upper_sequence": "peak",
                    "lower_sequence": "saddle",
                    "is_terminal": True,
                    "horton_strahler_order": 1,
                    "branch_vertex_count": 3,
                    "birth_fitness": 1.0,
                    "member_sequences": ["peak", "mid1", "mid2"],
                }
            ]
        ),
        persistence=[1.0],
        amino_acid={"peak": "AAA", "saddle": "AAT", "mid1": "TTT", "mid2": "CCC"},
        nucleotide={"peak": "AAA", "saddle": "AAT", "mid1": "TTT", "mid2": "CCC"},
    )
    measured = measure_tree(column)
    assert (
        measured["amino-acid Hamming distance along a branch: mean / median"] == "1 / 1"
    )
    assert measured["stop-codon genotypes on peaks"] == "0"


def test_write_round_trip_records_the_band_settings(tmp_path):
    rows = comparison_rows(_nominal(), _nominal())
    directory = tmp_path / "canonical" / "candidate_band_comparison"
    sidecar = tmp_path / ".candidate-band-comparison-provenance.json"
    write_table(
        directory,
        rows,
        n_replicates=10000,
        seed=20261019,
        band_quantile=0.95,
        provenance={
            "schema": 1,
            "n_replicates": 10000,
            "seed": 20261019,
            "band_quantile": 0.95,
            "draws_executed": 0,
        },
        sidecar=sidecar,
    )
    with (directory / "candidate_band_comparison.csv").open(encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    with (directory / "candidate_band_comparison.tsv").open(encoding="utf-8") as handle:
        tsv_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert csv_rows == tsv_rows == rows
    text = (directory / "README.md").read_text(encoding="utf-8")
    assert "draws (n_replicates) = 10000" in text
    assert "seed = 20261019" in text
    assert "band quantile = 0.95" in text
    assert "two critical endpoints" in text
    assert "not a sum of distances along the path" in text
    assert "not the maximum pairwise distance" in text
    assert "not branch span" in text
    assert (directory / "candidate_band_comparison.png").stat().st_size > 0
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["draws_executed"] == 0
    assert "numbered S6" in readme_text(
        n_replicates=10000, seed=20261019, band_quantile=0.95
    )
