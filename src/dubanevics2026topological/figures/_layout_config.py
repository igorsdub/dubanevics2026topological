"""The configured tree layout order, read on the figure side.

This lives under ``figures/`` and not in ``analysis/__init__.py`` because that file
is hashed by ``scientific_code_paths()`` and by the clamped comparison stage: a
presentation helper there would invalidate the fingerprint of the base analysis.
"""

from __future__ import annotations

from typing import Any

from ..analysis import load_settings


def tree_horizontal_layout(settings: dict[str, Any] | None = None) -> str:
    """Return the ``horizontal_layout`` that tree-building callers pass down.

    It is the one key ``[rendering] tree_horizontal_layout`` of
    ``config/figures.toml``. The analysis builders
    (``build_manifold_graph_tree``, ``build_candidate_leaf_view``) take it as their
    ``horizontal_layout`` argument; Topographer validates the value (``leaf_order``,
    ``iplotx`` or ``subtree_extremum``). Branch order is presentation only: it moves
    x coordinates and changes no scientific quantity.
    """
    rendering = (settings or load_settings())["figures"]["rendering"]
    return str(rendering["tree_horizontal_layout"])
