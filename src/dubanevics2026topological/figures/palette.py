"""Shared palette and panel-size fallbacks for the figures (ADR 0012, ADR 0010).

``config/figures.toml`` sets every colour and size; the constants here are the
single fallback the figure modules read when a key is missing, so no module
restates a hex literal. Five hues: red (wild type only), crimson (stop codon),
green (synonymous), orange (elder-protected only) and grey (structure, peaks,
bands); the codon-entropy figure alone adds sky blue and magenta for its two highlighted
branches. Peaks are mid-grey (``PEAK_GREY``) and filled; black
(``EMPHASIS_BLACK``) is emphasis only, for the peaks a figure singles out
(absorbed). No marker is hollow and no figure draws valley markers (GLOSSARY,
Figure marker rules). The palette binds annotations, not figures: each annotation
keeps its colour wherever it appears. On persistence panels colour is the
annotation.
"""

from __future__ import annotations

from typing import Any

# Red means wild type only: a solid star on trees, a solid triangle where the wild
# type is a peak of a persistence panel. Never translucent.
WILD_TYPE_RED = "#d62728"
# Stop codon: dark crimson (Reds, ColorBrewer), distinct from the plasma count ramp
# (which has no dark red) and from the lighter wild-type red. A solid cross on trees
# and a solid triangle on persistence panels, never translucent.
STOP_CODON_CRIMSON = "#a50f15"
SYNONYMOUS_GREEN = "#1b9e77"
# Synonymous peaks and valleys in the combined stop-and-synonymous encoding (full-graph
# and ``*_combined`` figures): teal. ``SYNONYMOUS_TEAL`` is the diagram marker and key
# glyph (its fade comes from the branch-count opacity); the tree's synonymous leaf
# branches run from ``SYNONYMOUS_TEAL_LIGHT`` (count one) to ``SYNONYMOUS_TEAL_DARK``
# (the largest count) on the same log scale as the grey count ramp.
SYNONYMOUS_TEAL = "#00897b"
SYNONYMOUS_TEAL_LIGHT = "#a1dad2"
SYNONYMOUS_TEAL_DARK = "#00362f"
# The two branches the codon-entropy figure singles out, each a hue used by no other
# annotation or figure: the KEM peak branch (sky blue) and the TCL peak branch (magenta).
# Their sweep panels take the same colours. Nothing else is drawn in either.
KEM_BRANCH_COLOR = "#56b4e9"
TCL_BRANCH_COLOR = "#e7298a"
# Peaks of every persistence diagram: mid-grey and filled.
PEAK_GREY = "#737373"
# Emphasis only: the peaks a figure singles out (absorbed by the band, singleton
# leaves). Also the ink of rules and diagonals that are not peaks.
EMPHASIS_BLACK = "#111111"
COUNT_BLACK = "#000000"  # branch vertex count ramp: opacity, not hue, carries the count
DARK_GREY = "#595959"
BAND_GREY = "#9e9e9e"
LIGHT_GREY = "#bdbdbd"
ELDER_PROTECTED_ORANGE = "#ff7f0e"
MEDIAN_GREY = "#555555"
THRESHOLD_GREY = "#555555"  # functional-threshold rule
DIAGONAL_GREY = "#737373"
# Papkou agreement density: light grey for the least-occupied bin, dark grey for
# the most-occupied bin and for the scatter fallback. Same values the figure
# used as literals.
AGREEMENT_BIN_LIGHT = "#d9d9d9"
AGREEMENT_BIN_DARK = "#252525"
COUNT_RAMP_DARK = "#1a1a1a"  # peak-pair count ramp, most pairs

# Structural greys. Each names one role, so two roles may share a value and
# still be restyled independently through their own configuration key.
TREE_GREY = "#b3b3b3"  # tree edges (``biology_tree_color``)
INK_GREY = "#222222"  # text and strokes
MID_GREY = "#777777"  # filled histogram bars
INTERVAL_GREY = "#555555"  # interval segments
OUTSIDE_GREY = "#f5f5f5"  # entropy cells outside the population span
# Horton–Strahler order: one purple ramp, order 1 light lavender to order 4
# near-black purple (lightness falls at every step). Order is a different quantity from the grey
# vertex count, so it has a hue. Every figure that colours by order reads this
# tuple (or the same values under ``horton_strahler_colors``).
HORTON_STRAHLER_COLORS = ("#bcbddc", "#807dba", "#54278f", "#2a0a4a")  # orders 1 to 4
# Black-and-white vertex count: ``COUNT_BLACK`` whose opacity carries the count (see
# ``_branch_count_style``), and the light grey of the sequence-count histogram above
# a persistence diagram. These are the only count colours; no hue.
VERTEX_COUNT_GREY = COUNT_BLACK
HISTOGRAM_GREY = LIGHT_GREY
# Top histogram of every tree-persistence figure: bin count, the two count ticks and the
# headroom of the count axis over the tallest bin (the axis limit is headroom x tallest bin).
TOP_HISTOGRAM_BINS = 200
TOP_HISTOGRAM_COUNT_TICKS = (0, 20_000)
TOP_HISTOGRAM_HEADROOM = 1.05
FDR_RAMP_LIGHT = "#d9d9d9"  # critical-FDR ramp at 100 %, black at 5 %

# Branch outcome states, every one filled: kept = mid-grey (a plain peak),
# absorbed = black (the emphasised peaks), elder-protected = orange. No marker in any
# figure is hollow (GLOSSARY, Figure marker rules).
OUTCOME_COLORS = {
    "kept": PEAK_GREY,
    "absorbed": EMPHASIS_BLACK,
    "elder_protected": ELDER_PROTECTED_ORANGE,
}
SURVIVES_COLOR = PEAK_GREY

# Shared panel sizes (layout pixels): ``rendering.tree_width`` and friends.
TREE_WIDTH = 420
TREE_HEIGHT = 250
PERSISTENCE_WIDTH = 250
PERSISTENCE_HEIGHT = 250


def tree_panel_size(
    rendering: dict[str, Any], appearance: dict[str, Any] | None = None
) -> tuple[int, int]:
    """Return the tree panel ``(width, height)``: a figure's own ``tree`` table
    overrides the shared ``rendering`` keys."""
    own = (appearance or {}).get("tree", {})
    return (
        int(own.get("width", rendering.get("tree_width", TREE_WIDTH))),
        int(own.get("height", rendering.get("tree_height", TREE_HEIGHT))),
    )
