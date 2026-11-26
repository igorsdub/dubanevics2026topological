"""Publish one ``merge_flow`` product per state from saved products only."""

from __future__ import annotations

import inspect
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import merge_flow as merge_flow_module
from ..analysis.describe import describe
from ..analysis.ridges import RidgeTree
from ..landscape.products import _publish, _sha256
from ..results import ALLOWED_KEYS, RunContext
from . import landscape_summary_inputs, ridges_product, tree_layout_products
from .landscape_summary_inputs import (
    Loader,
    _classes,
    _landscape_for,
    _parent,
    _verified,
)
from .ridges_product import _reference, _ridge_tree
from ..results.processed import LEAVES, product_loader

_LABEL_KEYS = frozenset(ALLOWED_KEYS) - {"host", "run", "hash", "style"}


def _settings(context: RunContext) -> dict[str, Any]:
    """The ``[merge_flow]`` table of the analysis config; K has no default."""
    with (Path(context.root) / "config" / "analysis.toml").open("rb") as handle:
        section = tomllib.load(handle)["merge_flow"]
    return {
        "min_functional_count_K": int(section["min_functional_count_K"]),
        "level_grid": str(section["level_grid"]),
    }


def _flow_tree(product: Any) -> RidgeTree:
    """The ridge tree plus the root members the tree product saved, if any."""
    tree = _ridge_tree(product)
    path = product.artifacts.get("root_members.parquet")
    if path is None:
        return tree
    root = pl.read_parquet(path).select("dna_sequence")
    return RidgeTree(tree.nodes, tree.edges, tree.branch_members, root)


def _names(tree: Any, leaf: Any) -> dict[str, str]:
    """Amino-acid names of the summits: the tree's nodes, else the leaf product's peaks."""
    names: dict[str, str] = {}
    nodes = pl.read_parquet(tree.artifacts["critical_nodes.parquet"])
    if "aa_sequence" in nodes.columns:
        names.update(
            {
                dna: aa
                for dna, aa in zip(
                    nodes["tree_node_identity"], nodes["aa_sequence"], strict=True
                )
                if aa is not None
            }
        )
    peaks = pl.read_parquet(leaf.artifacts["leaves.parquet"])
    for dna, aa in zip(
        peaks["peak_dna_sequence"], peaks["peak_aa_sequence"], strict=True
    ):
        names.setdefault(dna, aa)
    return names


def build_merge_flow_products(
    context: RunContext,
    *,
    labels: Mapping[str, str] = LEAVES,
    loader: Loader | None = None,
) -> dict[str, Path]:
    """Save each state's merge flow under the label keys of its own tree."""
    if loader is None:
        loader = product_loader(context)

    settings = _settings(context)
    code_paths = (
        Path(inspect.getfile(merge_flow_module)).resolve(),
        Path(inspect.getfile(describe)).resolve(),
        Path(landscape_summary_inputs.__file__).resolve(),
        Path(inspect.getfile(ridges_product)).resolve(),
        Path(inspect.getfile(tree_layout_products)).resolve(),
        Path(__file__).resolve(),
    )
    digest = {path.name: _sha256(path) for path in code_paths}
    outputs: dict[str, Path] = {}
    for state, label in labels.items():
        leaf = _verified(loader, "leaves", label)
        tree = _parent(loader, leaf, "tree", state)
        landscape = _landscape_for(loader, tree, state)
        band = state.endswith("_band")
        nominal = _parent(loader, tree, "tree", state) if band else None
        swept = nominal if band else tree
        info = json.loads(landscape.artifacts["diagnostics.json"].read_text())
        rows = pl.read_parquet(landscape.artifacts["vertices.parquet"])
        vertices = dict(zip(rows["dna_sequence"], rows["raw_fitness"], strict=True))
        order = (
            dict(zip(rows["dna_sequence"], rows["vertex_index"], strict=True))
            if "vertex_index" in rows.columns
            else None
        )
        classification = None
        classes: dict[str, str | None]
        if state == "full":
            classes = dict.fromkeys(vertices)
        elif state.startswith(("published_by", "reflected_by")):
            table, classification = _classes(context, landscape, loader, state)
            classes = {dna: table[dna] for dna in vertices}
        else:
            threshold = info["threshold"]
            classes = {
                dna: "functional" if fitness >= threshold else "non_functional"
                for dna, fitness in vertices.items()
            }
        result = describe(
            products={"merge_flow"},
            merge_flow_tree=_flow_tree(tree),
            merge_flow_state=state,
            merge_flow_vertices=vertices,
            merge_flow_classes=classes,
            merge_flow_names=_names(swept, leaf),
            merge_flow_pairs=pl.read_parquet(
                swept.artifacts["persistence_pairs.parquet"]
            ),
            merge_flow_settings=settings,
            merge_flow_nominal=None if nominal is None else _flow_tree(nominal),
            merge_flow_order=order,
        ).merge_flow
        params = {
            key: value
            for key, value in tree.manifest.scientific_identity.items()
            if key in _LABEL_KEYS
        }
        parents = [_reference(leaf), _reference(tree), _reference(landscape)]
        if nominal is not None:
            parents.append(_reference(nominal))
        identity = {
            **params,
            "state": state,
            "merge_flow_settings": settings,
            "code_digest": digest,
        }
        if classification is not None:
            identity["classification_table_sha256"] = classification
        outputs[state], _ = _publish(
            "merge_flow",
            params,
            identity,
            context=context,
            input_data_sha256s=(
                {}
                if classification is None
                else {"classification_table": classification}
            ),
            parents=parents,
            code_paths=code_paths,
            write_artifacts=result.save,
        )
    return outputs
