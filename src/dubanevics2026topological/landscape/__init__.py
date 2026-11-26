"""Selected genotype landscapes and their nominal topological trees.

The public interface accepts a Full ``ScalarNetwork`` plus a threshold and,
optionally, immutable sequence classifications. It returns graph and nominal-tree
products with stable DNA sequence identities; graph-library indices stay private.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from topographer.graph import ScalarNetwork

from .products import save_landscape_variants, save_selected_products
from .selected_landscape import (
    PreparedLandscape,
    analyze_prepared_landscape,
    prepare_clamped_landscape,
)
from .selection import Selection, prepare_selected_landscape, resolve_selection


def build_selected_landscape(
    full_network: ScalarNetwork,
    threshold: float,
    *,
    functional_sequences: Collection[str] | None = None,
    functional_classes: Mapping[str, str] | None = None,
) -> PreparedLandscape:
    """Build a deterministic selected graph from Full fitness and classifications."""
    return prepare_clamped_landscape(
        full_network,
        threshold,
        functional_sequences=functional_sequences,
        functional_classes=functional_classes,
    )


def extract_nominal_merge_tree(
    landscape: PreparedLandscape,
    *,
    name: str,
    wild_type_sequence: str,
) -> Any:
    """Extract the nominal split tree and ordinary H0 persistence pairs."""
    return analyze_prepared_landscape(
        landscape, name=name, wild_type_sequence=wild_type_sequence
    )


__all__ = [
    "PreparedLandscape",
    "Selection",
    "build_selected_landscape",
    "extract_nominal_merge_tree",
    "prepare_selected_landscape",
    "resolve_selection",
    "save_landscape_variants",
    "save_selected_products",
]
