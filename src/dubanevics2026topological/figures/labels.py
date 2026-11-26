"""Display strings for the graph states, thresholds and branches of the figures and tables.

This module is the single source of the words a reader sees. Internal state names
(``full``, ``published``, ``published_by``, ``published_by_band``, ``reflected``,
``reflected_by``, ``reflected_by_band``), directory names, configuration keys, figure ids
and exported file names do not change; only the text drawn on a figure or written into a
table comes from here. ``config/figures.toml`` cannot import Python, so the few label
tables it carries repeat these strings; ``tests/test_labels.py`` asserts they match.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FULL = "Full"
PAPKOU = "Papkou"
PAPKOU_FDR = "Papkou + FDR"
PAPKOU_FDR_BOOTSTRAPPED = "Papkou + FDR, bootstrapped"
TOPOGRAPHER = "Topographer"
TOPOGRAPHER_FDR = "Topographer + FDR"
TOPOGRAPHER_FDR_BOOTSTRAPPED = "Topographer + FDR, bootstrapped"

STATE_LABELS: dict[str, str] = {
    "full": FULL,
    "published": PAPKOU,
    "published_by": PAPKOU_FDR,
    "published_by_band": PAPKOU_FDR_BOOTSTRAPPED,
    "reflected": TOPOGRAPHER,
    "reflected_by": TOPOGRAPHER_FDR,
    "reflected_by_band": TOPOGRAPHER_FDR_BOOTSTRAPPED,
}

PAPKOU_THRESHOLD = "Papkou threshold"
TOPOGRAPHER_THRESHOLD = "Topographer threshold"
THRESHOLD_LABELS: dict[str, str] = {
    "published": PAPKOU_THRESHOLD,
    "reflected": TOPOGRAPHER_THRESHOLD,
}

KEM_BRANCH = "KEM branch"
TCL_BRANCH = "TCL branch"
ALL_BRANCHES = "All branches"


def state_label(state: str) -> str:
    """The display label of one graph state, for example ``reflected_by`` -> ``Topographer + FDR``."""
    try:
        return STATE_LABELS[state]
    except KeyError:
        raise KeyError(f"No display label for graph state {state!r}; known: {sorted(STATE_LABELS)}.") from None


def graph_label(state: str) -> str:
    """The state label as a noun phrase, ``Papkou graph`` or ``Topographer + FDR graph``."""
    return f"{state_label(state)} graph"


def threshold_label(state: str) -> str:
    """The threshold label of a state without FDR, ``Papkou threshold`` or ``Topographer threshold``."""
    try:
        return THRESHOLD_LABELS[state]
    except KeyError:
        raise KeyError(f"No threshold label for {state!r}; known: {sorted(THRESHOLD_LABELS)}.") from None


def settings_fdr_q(settings: Mapping[str, Any]) -> float | None:
    """The BY level of the Topographer + FDR selection, read from the analysis settings.

    ``None`` when the settings do not declare one (a figure without FDR panels).
    """
    declared = (
        settings.get("analysis", {})
        .get("clamped", {})
        .get("candidate_functional_threshold", {})
    )
    q = declared.get("q")
    return None if q is None else float(q)


def fdr_level_text(q: float | None) -> str | None:
    """The one-line FDR level a figure prints once: ``FDR: BY, q = 0.05`` (``None`` without q)."""
    if q is None:
        return None
    return f"FDR: BY, q = {float(q)}"


__all__ = [
    "ALL_BRANCHES",
    "FULL",
    "KEM_BRANCH",
    "PAPKOU",
    "PAPKOU_FDR",
    "PAPKOU_FDR_BOOTSTRAPPED",
    "PAPKOU_THRESHOLD",
    "STATE_LABELS",
    "TCL_BRANCH",
    "THRESHOLD_LABELS",
    "TOPOGRAPHER",
    "TOPOGRAPHER_FDR",
    "TOPOGRAPHER_FDR_BOOTSTRAPPED",
    "TOPOGRAPHER_THRESHOLD",
    "fdr_level_text",
    "graph_label",
    "settings_fdr_q",
    "state_label",
    "threshold_label",
]
