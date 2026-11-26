"""Identity checks for the four-graph bootstrap composite's saved products.

The composite uses its own product route; this does not alter the reviewed
product pins or their existing selection guards.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ..landscape.selection import selection_identity
from ..simplification.checkpoint_identity import scientific_fingerprint
from ..threshold.functional_threshold import configured_declaration, declaration_identity


def expected_panel_identity(
    variant: str, settings: Mapping[str, Any], selection: Any
) -> dict[str, Any]:
    """Return the declared selection and resolved graph identity for one panel."""
    return {
        "variant": variant,
        "declaration": declaration_identity(configured_declaration(settings)),
        "selected_graph": selection_identity(selection),
        "threshold": float(selection.threshold.threshold),
    }


def require_panel_identity(
    saved_identity: Mapping[str, Any], variant: str, settings: Mapping[str, Any], selection: Any
) -> None:
    """Reject a saved product unless its graph and rule match this panel exactly."""
    expected = expected_panel_identity(variant, settings, selection)
    if dict(saved_identity) != expected:
        raise ValueError("Bootstrap composite panel product has a different scientific identity.")


def require_checkpoint_identity(
    checkpoint: Any, settings: Mapping[str, Any], selection: Any
) -> dict[str, Any]:
    """Check selection, declared rule, and complete draws before publishing a panel.

    The caller must additionally verify that the saved hills match the actual
    tree and bind the resulting band identity to its published artifacts.
    """
    scientific = scientific_fingerprint(checkpoint.summary["fingerprint"])
    if scientific.get("candidate_selection") != selection_identity(selection):
        raise ValueError("Bootstrap checkpoint selected graph identity differs.")
    if scientific.get("candidate_functional_threshold") != declaration_identity(
        configured_declaration(settings)
    ):
        raise ValueError("Bootstrap checkpoint threshold declaration differs.")
    configured = settings["analysis"]["bootstrap_smoothed_tree"]
    draw_count = int(configured["n_replicates"])
    config = scientific["config"]
    if config.get("n_replicates") != draw_count or config.get("seed") != configured["seed"]:
        raise ValueError("Bootstrap checkpoint draw count or seed differs.")
    if float(checkpoint.summary["settings"]["band_quantile"]) != float(configured["band_quantile"]):
        raise ValueError("Bootstrap checkpoint band quantile differs.")
    expected = list(range(draw_count))
    for condition in ("real", "flat"):
        draws = checkpoint.draw_stats.filter(pl.col("condition") == condition)["draw"].sort().to_list()
        if draws != expected:
            raise ValueError(f"Bootstrap checkpoint has incomplete or duplicate {condition} draws.")
    if checkpoint.draw_stats.height != 2 * draw_count:
        raise ValueError("Bootstrap checkpoint has unexpected draw conditions.")
    return scientific