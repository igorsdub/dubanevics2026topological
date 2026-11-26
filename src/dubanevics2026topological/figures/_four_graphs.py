"""The four graphs the 2x2 supplementary figures draw, and their grid.

The Papkou graph and the Topographer graph, each as a nominal (no FDR) selection
and as the Benjamini-Yekutieli q = 0.05 selection. Each is an analysis variant of
``config/variants/`` (the four variants of ``peak_uncertainty``): its selected
graph is what the variant's comparison checkpoint calls
``candidate_clamped``. In every 2x2 the columns are the graph (Papkou left,
Topographer right) and the rows the selection (nominal on top, BY q = 0.05 below),
the layout of the main-text figures.
"""

from __future__ import annotations

from .labels import graph_label, state_label

# slug -> (analysis variant, graph name, selection name)
PANELS: dict[str, tuple[str, str, str]] = {
    "published": ("threshold=published_fdr=none", graph_label("published"), "Nominal"),
    "published_fdr": (
        "threshold=published_fdr=by_q=0.05",
        graph_label("published"),
        "BY q = 0.05",
    ),
    "candidate": ("threshold=reflected_fdr=none", graph_label("reflected"), "Nominal"),
    "candidate_fdr": (
        "threshold=reflected_fdr=by_q=0.05",
        graph_label("reflected"),
        "BY q = 0.05",
    ),
}
# slug -> graph state; a panel is titled with the strict state label, not graph and selection.
STATES = {
    "published": "published",
    "published_fdr": "published_by",
    "candidate": "reflected",
    "candidate_fdr": "reflected_by",
}


def panel_title(slug: str) -> str:
    """The panel's strict state label: ``Papkou``, ``Papkou + FDR``, ``Topographer``, ``Topographer + FDR``."""
    return state_label(STATES[slug])


# Rows of the grid, left (Papkou) to right (Topographer).
GRID_ORDER = (("published", "candidate"), ("published_fdr", "candidate_fdr"))
VARIANTS = tuple(variant for variant, _, _ in PANELS.values())
