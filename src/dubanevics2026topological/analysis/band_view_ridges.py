"""KEM and TCL ridge populations of the band-simplified tree, for the registry.

The bootstrap codon-entropy figure draws its ridges on the band-simplified view of
the Candidate split tree (the hills that clear the bootstrapped persistence
threshold survive, every other hill is absorbed). The ``codon-entropy`` stage uses
the selected-leaf view instead, so its ridge counts are another population. This
module is the one place the band view's counts are computed: the figure's own
steps (``bootstrap_tree`` check, ``build_band_simplified_view``,
``extract_codon_entropy_populations``) in the same order, so the registry's count
is the count the figure prints.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import polars as pl


def _band_view_ridge_counts(
    comparison: Mapping[str, Any],
    hills: pl.DataFrame,
    tree_analysis: Any,
    config: Mapping[str, Any],
    *,
    functional_sequences: Collection[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Total and functional sequences of the core, KEM and TCL band-view populations.

    ``comparison`` is the cached clamped comparison (its Candidate threshold is the
    functional threshold), ``hills`` the bootstrap checkpoint's hill table,
    ``tree_analysis`` the in-memory float64 Candidate split tree the hills were
    drawn on and ``config`` the ``[codon_entropy]`` settings. A sequence is functional
    when it is in ``functional_sequences`` (the Functional class, under
    ``mixture_quantile``) or, without it (``median_reflected``), when its raw fitness
    is at or above the Candidate threshold.
    """
    from ..simplification.bootstrap_smoothed_tree import (
        build_band_simplified_view,
        check_hills_match_tree,
    )
    from .codon_entropy_populations import extract_codon_entropy_populations

    check_hills_match_tree(hills, tree_analysis)
    view = build_band_simplified_view(tree_analysis, hills)
    populations = extract_codon_entropy_populations(
        comparison=comparison,
        candidate_view=view,
        config=config,
        save_files=False,
        functional_sequences=functional_sequences,
    )
    threshold = float(comparison["thresholds"]["candidate_clamped"])
    members = None if functional_sequences is None else frozenset(functional_sequences)
    counts = {}
    for name in ("core", "kem", "tcl"):
        frame = populations[f"df_{name}"]
        counts[name] = {
            "total_count": int(frame.height),
            "functional_count": int(
                frame.filter(
                    pl.col("fitness") >= threshold
                    if members is None
                    else pl.col("dna_sequence").is_in(members)
                ).height
            ),
        }
    return counts


def _load_band_view_ridge_counts(
    comparison: Mapping[str, Any], bootstrap: Any
) -> dict[str, dict[str, int]]:
    """The counts from the cached products and the raw tables (the real reader).

    Builds the Candidate split tree in memory from the raw fitness tables, as the
    figure does; a missing table raises ``FileNotFoundError``.
    """
    from ..landscape.selection import resolve_candidate_selection
    from . import load_settings
    from .candidate_tree_contrast import build_candidate_tree

    settings = load_settings()
    selection = resolve_candidate_selection(settings)
    tree, _ = build_candidate_tree(settings, selection=selection)
    counts = _band_view_ridge_counts(
        comparison,
        bootstrap.hills,
        tree,
        settings["analysis"]["codon_entropy"],
        functional_sequences=selection.functional_sequences,
    )
    # The population the bootstrap biology figure counts its stop-codon peaks in: the
    # registry's consistency check compares it with the cached tree's count.
    counts["candidate_tree"] = {
        "stop_codon_peak_count": _in_memory_stop_codon_peak_count(tree)
    }
    return counts


def _in_memory_stop_codon_peak_count(tree_analysis: Any) -> int:
    """Peaks with a stop codon in the in-memory Candidate split tree, as the figure counts them."""
    from types import SimpleNamespace

    from .biological_annotations import _build_biological_annotations

    evidence = _build_biological_annotations(
        SimpleNamespace(trees={"split": tree_analysis}), landscape="candidate_clamped"
    )
    return int(evidence.summary["stop_codon_peak_count"])
