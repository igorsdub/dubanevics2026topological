"""Saved-result input of ``functional_counts_vs_q``.

The figure reads the combined FDR scan written by the classification
probability scan (``just classification-probability-scans``; the stage the
naming ADR calls ``functional_classification_scan``). That file is the
``fdr_q_scan`` result: one row per cutoff, method and FDR target, with the
functional, uncertain and non-functional counts already computed. The figure
does not call the scan's plotters.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ..results import RunContext

# Relative to the run's output root. The recipe writes here; there is no
# separate checkpoint directory for this scan.
SCAN_RELATIVE = (
    "analysis",
    "classification_probability_scans",
    "fdr_scan_all_cutoffs_and_methods.parquet",
)


def scan_path(context: RunContext):
    """Path of the combined FDR scan under ``context``."""
    return context.output_root.joinpath(*SCAN_RELATIVE)


def load_fdr_q_scan(context: RunContext) -> dict[str, Any]:
    """The combined FDR scan table and the file it was read from.

    Raises ``FileNotFoundError`` when the classification-probability scan has
    not written the combined table.
    """
    path = scan_path(context)
    if not path.is_file():
        raise FileNotFoundError(
            "The fdr_q_scan result is missing "
            f"({path}). Run classification-probability-scans "
            "(functional_classification_scan) first."
        )
    return {"scan": pl.read_parquet(path), "source": str(path)}
