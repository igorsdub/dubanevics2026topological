"""The FDR variant declarations name their processed classification folder."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("variant", "cutoff", "snapshot"),
    (
        ("threshold=reflected_fdr=by_q=0.05", "median_reflected", "reflected"),
        ("threshold=published_fdr=by_q=0.05", "papkou_original", "published"),
    ),
)
def test_variant_declaration_reads_its_processed_classification(
    variant: str, cutoff: str, snapshot: str
) -> None:
    with (ROOT / "config/variants" / f"{variant}.toml").open("rb") as stream:
        declaration = tomllib.load(stream)["clamped"]["candidate_functional_threshold"]
    assert declaration["cutoff"] == cutoff
    assert declaration["input_dir"] == f"data/processed/classification/{snapshot}"
    assert "table_sha256" not in declaration
