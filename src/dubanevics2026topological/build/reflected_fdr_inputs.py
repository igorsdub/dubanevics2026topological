"""Resolve the two processed nominal trees for Figure 6 without reselection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..results import MissingProductError, PresentationProducts, RunContext
from .landscape_summary_inputs import _snapshot
from .saved_figure_inputs import _landscape_for_tree, _layout
from .tree_layout_products import _tree_for_leaf


def validate_selection(state: str, identity: dict[str, Any]) -> None:
    """Refuse a different selected graph or a bootstrap band tree."""
    expected = {"thr": "reflected", "fdr": "none", "tree": "nominal"}
    if state == "reflected_by":
        expected.update(fdr="by", q=0.05)
    elif state != "reflected":
        raise ValueError(f"Unknown reflected tree state {state!r}.")
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{state}: expected nominal selection {expected}, got {identity}.")


def load_reflected_fdr_inputs(context: RunContext) -> dict[str, Any]:
    """Load verified processed trees, layouts, landscape populations and thresholds."""
    products = PresentationProducts(context)
    rows, ids = {}, {}
    for state in ("reflected", "reflected_by"):
        try:
            tree = _tree_for_leaf(state, context)
            identity = tree.manifest.scientific_identity
            validate_selection(state, identity)
            landscape = _landscape_for_tree(products, tree)
            landscape_identity = landscape.manifest.scientific_identity
            for key in ("thr", "fdr", "q", "landscape_selection"):
                if identity.get(key) != landscape_identity.get(key):
                    raise ValueError(f"{state}: tree and landscape disagree on {key}.")
            if state == "reflected_by":
                def loader(kind, label):
                    return products.load(kind, label)

                path, digest = _snapshot(context, landscape, loader, state)
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != digest:
                    raise ValueError(f"{state}: classification snapshot {path} has a mismatched digest.")
                ids["classification_sha256"] = digest
            layout, layout_id = _layout(products, tree)
            population = products.landscape_population(landscape)
            selection = identity["landscape_selection"]
            threshold = float(selection["threshold"])
            diagnostics = json.loads(landscape.artifacts["diagnostics.json"].read_text())
            if threshold != float(diagnostics["threshold"]):
                raise ValueError(f"{state}: saved threshold and diagnostics disagree.")
            if diagnostics.get("functional_set_sha256") != selection.get("functional_set_sha256"):
                raise ValueError(f"{state}: saved functional-set identity disagrees with diagnostics.")
            if (
                not population.height
                or population["dna_sequence"].null_count()
                or population["dna_sequence"].n_unique() != population.height
                or population.height != diagnostics["selected_component_vertex_count"]
            ):
                raise ValueError(f"{state}: saved population has missing identities.")
            rows[state] = {"layout": layout, "population": population, "threshold": threshold,
                           "method": identity["fdr"], "q": identity.get("q")}
            ids[state] = {"tree": tree.manifest.scientific_id_sha256, "landscape": landscape.manifest.scientific_id_sha256, "layout": layout_id}
        except (FileNotFoundError, KeyError, MissingProductError) as exc:
            raise ValueError(f"{state}: missing processed nominal tree, layout, landscape or classification input; run `just all` or `just fetch-data`: {exc}") from exc
    return {"rows": rows, "product_ids": ids}