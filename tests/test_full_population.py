"""Checks for the raw-fitness population used by the component figure."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import dubanevics2026topological.landscape.full_population as population_module
from dubanevics2026topological.landscape.full_population import (
    load_full_network_fitness,
)


def _settings(dataset_path: str) -> dict[str, object]:
    return {"analysis": {"dataset": {"landscape": "full", "fitness_wt": dataset_path}}}


def test_load_full_network_fitness_includes_wt_and_every_observed_vertex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = pl.DataFrame(
        {
            "dna_from": ["ATGGAA", "ATGGAA"],
            "dna_to": ["ATGGAC", "ATGTAA"],
            "aa_from": ["ME", "ME"],
            "aa_to": ["MD", "M%"],
            "s": [0.2, -0.4],
            "se": [0.01, 0.02],
            "pval": [0.1, 0.2],
        }
    )
    source_path = tmp_path / "fitness_wt.parquet"
    source.write_parquet(source_path)
    monkeypatch.setattr(population_module, "ROOT", tmp_path)

    observed = load_full_network_fitness(_settings("fitness_wt.parquet"))

    assert observed.columns == ["dna_sequence", "raw_fitness"]
    assert observed.to_dicts() == [
        {"dna_sequence": "ATGGAA", "raw_fitness": 0.0},
        {"dna_sequence": "ATGGAC", "raw_fitness": 0.2},
        {"dna_sequence": "ATGTAA", "raw_fitness": -0.4},
    ]
    assert observed.height == 3
    assert observed["dna_sequence"].n_unique() == observed.height


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"dna_sequence": None, "s": 0.0}], "must not be null"),
        (
            [
                {"dna_sequence": "A", "s": 0.0},
                {"dna_sequence": "A", "s": 1.0},
            ],
            "must be unique",
        ),
        ([{"dna_sequence": "A", "s": float("nan")}], "must be finite"),
        ([], "must not be empty"),
    ],
)
def test_load_full_network_fitness_rejects_invalid_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    source_path = tmp_path / "fitness_wt.parquet"
    pl.DataFrame({"unused": [0]}).write_parquet(source_path)
    monkeypatch.setattr(population_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        population_module,
        "build_papkou2023_fitness_vertex_table",
        lambda _: pl.DataFrame(
            rows, schema={"dna_sequence": pl.String, "s": pl.Float64}
        ),
    )

    with pytest.raises(ValueError, match=message):
        load_full_network_fitness(_settings("fitness_wt.parquet"))


def test_load_full_network_fitness_rejects_non_full_landscape() -> None:
    with pytest.raises(ValueError, match="full landscape"):
        load_full_network_fitness(
            {"analysis": {"dataset": {"landscape": "study", "fitness_wt": "x"}}}
        )
