"""Nominal peak counts of the four-graph composite's panels, from their checkpoints.

The saved panel draws the band-simplified tree, so it carries no count of the graph's
nominal peaks. Each panel's bootstrap checkpoint records it as ``hill_count`` (one hill
per nominal peak). This lives apart from ``bootstrap_composite_product`` because that
file is part of the panel products' renderer digest.
"""

from __future__ import annotations

from typing import Any

from ..analysis import load_settings
from ..analysis.checkpoint import checkpoint_lock
from ..figures.bootstrap_four_graphs import PANEL_ORDER
from ..results import RunContext
from ..simplification.bootstrap_smoothed_tree import (
    checkpoint_dir,
    load_cached_bootstrap_smoothed_tree,
)

NOMINAL_KEY = "_composite_nominal_peaks"


def nominal_peak_counts(context: RunContext) -> dict[str, int]:
    """Variant -> peaks of that variant's nominal tree, read from its checkpoint."""
    counts: dict[str, int] = {}
    for variant, _ in PANEL_ORDER:
        variant_context = RunContext(
            root=context.root,
            output_root=context.output_root / "variants" / variant,
            processed_root=context.processed_root,
            raw_data_dir=context.raw_data_dir,
            barseq_export_dir=context.barseq_export_dir,
            variant_name=variant,
        )
        settings = load_settings(context=variant_context)
        with checkpoint_lock(checkpoint_dir(variant_context), "bootstrap smoothed tree"):
            checkpoint = load_cached_bootstrap_smoothed_tree(
                settings=settings, context=variant_context, lock=False
            )
            counts[variant] = int(checkpoint.summary["hill_count"])
    return counts


def add_nominal_peaks(panels: dict[str, dict[str, Any]], context: RunContext) -> None:
    """Attach each panel's nominal peak count under ``NOMINAL_KEY`` (in place)."""
    for variant, count in nominal_peak_counts(context).items():
        panels[variant][NOMINAL_KEY] = count
