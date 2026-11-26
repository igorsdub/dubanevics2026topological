"""Saved-comparison input of the four-graph 2x2 figures (leaf swarm, branch vertex counts).

The four graphs are the Candidate-selected graphs of four analysis variants
(``figures/_four_graphs.py``): Published and Candidate thresholds, each nominal and
as the Benjamini-Yekutieli q = 0.05 selection. Each variant's clamped-comparison
checkpoint lives under ``<output root>/variants/<variant>/analysis/clamped_comparison``
(``just variant-comparison-analysis <variant>``); this module only reads those
checkpoints and recomputes no landscape, tree or persistence pair. From each it takes
the Candidate landscape's leaf and member tables (with the annotation pairs the saved
tree gives) and the split-tree branch table with its Horton-Strahler orders.
Sequence identities are DNA strings throughout; no Rustworkx index leaves this module.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ..figures._four_graphs import VARIANTS
from ..figures._leaf_data import leaf_tables
from ..results import RunContext

LANDSCAPE = "candidate_clamped"
BRANCH_TABLE = f"{LANDSCAPE}_branch_table"
INPUT_NAME = "four_graph_comparisons"


def variant_context(base: RunContext, variant: str) -> RunContext:
    """``base`` moved to ``<output root>/variants/<variant>``, the root the variant's stages write."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown four-graph variant {variant!r}.")
    return dataclasses.replace(
        base, output_root=Path(base.output_root) / "variants" / variant, variant_name=variant
    )


def load_four_graph_comparisons(context: RunContext) -> dict[str, Any]:
    """Leaf tables, branch tables and checkpoint fingerprints of the four variants.

    Returns ``{"panels": {variant: {"leaf_tables", "branch_table"}}, "fingerprints":
    {variant: digest}}``. A variant whose comparison checkpoint is absent stops the
    load and names the command that builds it; no panel stands in for another.
    """
    from ..analysis.clamped_cache import load_cached_clamped_comparison
    from ..analysis.describe import describe
    from . import _comparison_analysis_fingerprint

    panels: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for variant in VARIANTS:
        scoped = variant_context(context, variant)
        try:
            comparison = load_cached_clamped_comparison(context=scoped)
            fingerprints[variant] = _comparison_analysis_fingerprint(scoped)
        except FileNotFoundError as error:
            raise ValueError(
                f"The {variant!r} comparison checkpoint is missing under "
                f"{scoped.output_root}: run `just variant-comparison-analysis {variant}`."
            ) from error
        landscape = comparison["landscapes"][LANDSCAPE]
        pairs = describe(
            products={"annotations"}, landscape=landscape, landscape_name=LANDSCAPE
        ).annotations.pairs
        panels[variant] = {
            "leaf_tables": leaf_tables(landscape, key=LANDSCAPE, pairs=pairs),
            "branch_table": comparison["tables"][BRANCH_TABLE],
        }
    return {"panels": panels, "fingerprints": fingerprints}
