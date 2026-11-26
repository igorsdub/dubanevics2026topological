"""Publish one ``ridges`` product per classifiable state from saved products only."""

from __future__ import annotations

import inspect
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import ridges as ridges_module
from ..analysis.describe import describe
from ..analysis.ridges import RIDGE_NAMES, RidgeTree
from ..landscape.products import _publish, _sha256
from ..results import ALLOWED_KEYS, RunContext
from . import landscape_summary_inputs
from .landscape_summary_inputs import (
    Loader,
    _classes,
    _landscape_for,
    _parent,
    _verified,
)
from ..results.processed import LEAVES, product_loader

_LABEL_KEYS = frozenset(ALLOWED_KEYS) - {"host", "run", "hash", "style"}
# Full carries no functional threshold, so it has no functional class to key by.
EXCLUDED_STATES = {"full": "no functional threshold, so no functional_class"}
TOP_LEAF_LABELS = 3


def _summits(context: RunContext) -> dict[str, str]:
    """The nominated KEM and TCL summits from ``[codon_entropy]`` of the analysis config."""
    with (Path(context.root) / "config" / "analysis.toml").open("rb") as handle:
        section = tomllib.load(handle)["codon_entropy"]
    return {name: str(section[f"{name}_summit_dna_sequence"]) for name in RIDGE_NAMES}


def _ridge_tree(product: Any) -> RidgeTree:
    """The saved critical nodes, branches and branch members of a tree product."""
    nodes = pl.read_parquet(product.artifacts["critical_nodes.parquet"])
    edges = pl.read_parquet(product.artifacts["branches.parquet"])
    members = pl.read_parquet(product.artifacts["branch_members.parquet"])
    return RidgeTree(
        nodes.select("tree_node_identity", "scalar"),
        edges.select("source_tree_node_identity", "target_tree_node_identity"),
        members.select(
            "source_tree_node_identity", "target_tree_node_identity", "dna_sequence"
        ),
    )


def _reference(product: Any) -> dict[str, str]:
    return {
        "product_type": product.product_type,
        "label": product.label,
        "scientific_id_sha256": product.manifest.scientific_id_sha256,
    }


def _leaf_hills(leaf: Any, summits: Mapping[str, str]) -> list[dict[str, Any]]:
    """Every terminal leaf by member count, the top three labelled (beyond the spec)."""
    table = pl.read_parquet(leaf.artifacts["leaves.parquet"])
    ridge_of = {dna: name for name, dna in summits.items()}
    return [
        {
            "rank": rank,
            "peak_dna_sequence": row["peak_dna_sequence"],
            "member_count": row["member_count"],
            "top_label": f"top_{rank}" if rank <= TOP_LEAF_LABELS else None,
            "ridge_name": ridge_of.get(row["peak_dna_sequence"]),
        }
        for rank, row in enumerate(table.iter_rows(named=True), 1)
    ]


def build_ridges_products(
    context: RunContext,
    *,
    labels: Mapping[str, str] = LEAVES,
    loader: Loader | None = None,
) -> dict[str, Path]:
    """Save each classifiable state's ridges under the label keys of its own tree."""
    if loader is None:
        loader = product_loader(context)

    summits = _summits(context)
    code_paths = (
        Path(inspect.getfile(ridges_module)).resolve(),
        Path(inspect.getfile(describe)).resolve(),
        Path(landscape_summary_inputs.__file__).resolve(),
        Path(__file__).resolve(),
    )
    digest = {str(i): _sha256(path) for i, path in enumerate(code_paths)}
    outputs: dict[str, Path] = {}
    for state, label in labels.items():
        if state in EXCLUDED_STATES:
            print(f"{state}: skipped, {EXCLUDED_STATES[state]}")
            continue
        leaf = _verified(loader, "leaves", label)
        tree = _parent(loader, leaf, "tree", state)
        landscape = _landscape_for(loader, tree, state)
        band = state.endswith("_band")
        nominal = _parent(loader, tree, "tree", state) if band else None
        info = json.loads(landscape.artifacts["diagnostics.json"].read_text())
        rows = pl.read_parquet(landscape.artifacts["vertices.parquet"])
        vertices = dict(zip(rows["dna_sequence"], rows["raw_fitness"], strict=True))
        classification = None
        if state.startswith(("published_by", "reflected_by")):
            table, classification = _classes(context, landscape, loader, state)
            classes = {dna: table[dna] for dna in vertices}
        else:
            threshold = info["threshold"]
            classes = {
                dna: "functional" if fitness >= threshold else "non_functional"
                for dna, fitness in vertices.items()
            }
        result = describe(
            products={"ridges"},
            ridge_tree=_ridge_tree(tree),
            ridge_population=state.removesuffix("_band"),
            ridge_tree_state="band" if band else "nominal",
            ridge_summits=summits,
            ridge_vertices=vertices,
            ridge_classes=classes,
            ridge_nominal=None if nominal is None else _ridge_tree(nominal),
        ).ridges
        result.sidecar["leaf_hills"] = _leaf_hills(leaf, summits)
        result.sidecar["leaf_hills_note"] = (
            "Addition beyond the ridges spec: every terminal leaf of this tree by "
            "member count; the top three are labelled."
        )
        result.sidecar["excluded_states"] = EXCLUDED_STATES
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
            "summits": summits,
            "code_digest": digest,
        }
        if classification is not None:
            identity["classification_table_sha256"] = classification
        outputs[state], _ = _publish(
            "ridges",
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
