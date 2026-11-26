"""Saved-product input of the band-tree leaf figures (``*_leaf_persistence_band``, ``*_leaf_swarm_band``).

The two BY band trees are the story-view band trees (``[story_views] band_trees`` in
config/figures.toml, the processed band states the ``bootstrap_band`` view and the band tree
comparisons read). They
are loaded with the D10 loader, ``load_saved_tree_inputs``, and turned into leaf and
member tables by ``figures/_leaf_data.saved_tree_leaf_tables``; nothing is recomputed.
Sequence identities are DNA strings throughout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..figures._leaf_data import saved_tree_leaf_tables
from ..results import RunContext, parse_label
from .d10_tree_inputs import SavedTreeInputs, load_saved_tree_inputs

# Graph slug of each band tree's threshold (the leaf figures' Candidate graph is the
# reflected threshold).
SLUG_OF_THRESHOLD = {"published": "published", "reflected": "candidate"}


def load_band_leaf_inputs(
    context: RunContext,
    settings: Mapping[str, Any],
    loader: Callable[[RunContext, str], SavedTreeInputs] = load_saved_tree_inputs,
) -> dict[str, Any]:
    """Leaf tables, band settings and product ids of the Published and Candidate BY band trees.

    Returns ``{"panels": {slug: {"leaf_tables", "label", "band_quantile", "draws"}},
    "product_ids": {slug: tree scientific id, f"{slug}_landscape": landscape id}}``.
    """
    from .story_view_inputs import story_view_labels

    panels: dict[str, Any] = {}
    product_ids: dict[str, str] = {}
    for label in story_view_labels(settings, context)["band"]:
        params = dict(parse_label(label, validate=False).params)
        slug = SLUG_OF_THRESHOLD[str(params["thr"])]
        if params.get("tree") != "band" or params.get("fdr") != "by":
            raise ValueError(f"{label} is not a BY band tree.")
        saved = loader(context, label)
        panels[slug] = {
            "leaf_tables": saved_tree_leaf_tables(saved),
            "label": label,
            "band_quantile": params["bandq"],
            "draws": int(params["draws"]),
        }
        product_ids[slug] = saved.tree_id
        product_ids[f"{slug}_landscape"] = saved.landscape_id
    return {"panels": panels, "product_ids": product_ids}
