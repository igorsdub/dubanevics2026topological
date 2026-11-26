"""The four graphs the 2x2 supplementary figures draw, and their grid.

The Published graph and the Candidate graph, each as a nominal (no FDR) selection
and as the Benjamini-Yekutieli q = 0.05 selection. Each is an analysis variant of
``config/variants/`` (the four variants of ``peak_uncertainty``): its Candidate-
selected graph is what the variant's comparison checkpoint calls
``candidate_clamped``. In every 2x2 the columns are the graph (Published left,
Candidate right) and the rows the selection (nominal on top, BY q = 0.05 below),
the layout of the main-text figures.
"""

from __future__ import annotations

# slug -> (analysis variant, graph name, selection name)
PANELS: dict[str, tuple[str, str, str]] = {
    "published": ("threshold=published_fdr=none", "Published graph", "Nominal"),
    "published_fdr": (
        "threshold=published_fdr=by_q=0.05",
        "Published graph",
        "BY q = 0.05",
    ),
    "candidate": ("threshold=reflected_fdr=none", "Candidate graph", "Nominal"),
    "candidate_fdr": (
        "threshold=reflected_fdr=by_q=0.05",
        "Candidate graph",
        "BY q = 0.05",
    ),
}
# Rows of the grid, left (Published) to right (Candidate).
GRID_ORDER = (("published", "candidate"), ("published_fdr", "candidate_fdr"))
VARIANTS = tuple(variant for variant, _, _ in PANELS.values())
