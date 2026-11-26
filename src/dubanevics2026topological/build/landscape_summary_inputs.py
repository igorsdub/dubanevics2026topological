"""Assemble the seven landscape-summary states from verified saved products only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl

from ..analysis.describe import SUMMARY_COLUMNS, LandscapeSummaryInput
from ..analysis.leaves import Leaves
from ..results import RunContext
from ..results.processed import LEAVES, classification_table, product_loader

Loader = Callable[[str, str], Any]
_STATES = tuple(LEAVES)


class SavedGraph:
    """The vertex rows of a saved landscape, in the graph protocol the summary reads."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def node_indices(self):
        return range(len(self.rows))

    def __getitem__(self, index):
        return self.rows[index]


@dataclass(frozen=True)
class LoadedLandscapeSummary:
    """The seven summary inputs plus the saved evidence each was built from."""

    inputs: dict[str, LandscapeSummaryInput]
    full_fitness: dict[str, float]
    diagnostics: dict[str, dict]
    landscape_labels: dict[str, str]
    classification_sha256: dict[str, str]
    leaves: dict[str, Any]
    trees: dict[str, Any]
    landscapes: dict[str, Any]


def _verified(loader: Loader, product: str, label: str) -> Any:
    value = loader(product, label)
    if value.report.status != "full" or not value.report.byte_integrity:
        raise ValueError(f"Unverified or incomplete {product}: {label}")
    return value


def _parent(loader: Loader, child: Any, product: str, what: str) -> Any:
    refs = [r for r in child.manifest.parents if r.get("product_type") == product]
    if len(refs) != 1:
        raise ValueError(f"{what} must identify exactly one {product} parent.")
    parent = _verified(loader, product, refs[0]["label"])
    if parent.manifest.scientific_id_sha256 != refs[0]["scientific_id_sha256"]:
        raise ValueError(f"{what}: {product} parent does not match its recorded id.")
    return parent


def _landscape_for(loader: Loader, tree: Any, state: str) -> Any:
    """The tree's landscape; a band tree reaches it through its nominal tree."""
    if any(r.get("product_type") == "landscape" for r in tree.manifest.parents):
        return _parent(loader, tree, "landscape", state)
    return _parent(loader, _parent(loader, tree, "tree", state), "landscape", state)


def _snapshot(
    context: RunContext, landscape: Any, loader: Loader, state: str
) -> tuple[Path, str]:
    """The processed classification table of the state's variant and the sha256 the landscape pins."""
    pinned = landscape.manifest.data_provenance.get("classification_table")
    if pinned is None:
        raise ValueError(f"{state}: landscape does not pin its classification table.")
    variant = state.removesuffix("_band").removesuffix("_by")
    return classification_table(context, variant), pinned


def _classes(
    context: RunContext, landscape: Any, loader: Loader, state: str
) -> tuple[dict[str, str], str]:
    """Read the BY calls, refusing a table that is not the pinned one."""
    path, pinned = _snapshot(context, landscape, loader, state)
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != pinned:
        raise ValueError(
            f"{state}: snapshot {path.name} has sha256 {actual}, not {pinned}."
        )
    frame = pl.read_parquet(path)
    return dict(zip(frame["dna_sequence"], frame["fdr_class"], strict=True)), pinned


def load_landscape_summary_inputs(
    context: RunContext,
    *,
    labels: Mapping[str, str] = LEAVES,
    loader: Loader | None = None,
) -> LoadedLandscapeSummary:
    """Build the seven states: leaves -> tree -> landscape, classes by sha256."""
    if loader is None:
        loader = product_loader(context)

    inputs, diagnostics, landscape_labels, shas = {}, {}, {}, {}
    leaf_products, tree_products, landscape_products = {}, {}, {}
    full_fitness = None
    for state, name in zip(_STATES, SUMMARY_COLUMNS, strict=True):
        leaf = _verified(loader, "leaves", labels[state])
        tree = _parent(loader, leaf, "tree", state)
        landscape = _landscape_for(loader, tree, state)
        if state.endswith("_band"):
            sibling = _landscape_for(
                loader, _parent(loader, tree, "tree", state), state
            )
            if sibling.manifest.scientific_id_sha256 != (
                landscape.manifest.scientific_id_sha256
            ):
                raise ValueError(f"{state}: band tree uses a different landscape.")
        info = json.loads(landscape.artifacts["diagnostics.json"].read_text())
        rows = pl.read_parquet(landscape.artifacts["vertices.parquet"]).to_dicts()
        network = SimpleNamespace(
            graph=SavedGraph(rows),
            scalar_attr="raw_fitness",
            edge_count=pl.read_parquet(landscape.artifacts["edges.parquet"]).height,
        )
        if state == "full":
            full_fitness = {r["dna_sequence"]: r["raw_fitness"] for r in rows}
        classes = None
        if state.startswith(("published_by", "reflected_by")):
            classes, shas[name] = _classes(context, landscape, loader, state)
        product = Leaves(
            pl.read_parquet(leaf.artifacts["leaves.parquet"]),
            pl.read_parquet(leaf.artifacts["members.parquet"]),
            json.loads(leaf.artifacts["summary.json"].read_text()),
        )
        inputs[name] = LandscapeSummaryInput(
            network,
            product,
            None if state == "full" else info["threshold"],
            classes,
            info["component_count"],
            info["incident_vertex_count"],
        )
        diagnostics[name] = info
        landscape_labels[name] = landscape.label
        leaf_products[name], tree_products[name] = leaf, tree
        landscape_products[name] = landscape
    if full_fitness is None:
        raise ValueError("The Full state is required for the Full-library population.")
    return LoadedLandscapeSummary(
        inputs,
        full_fitness,
        diagnostics,
        landscape_labels,
        shas,
        leaf_products,
        tree_products,
        landscape_products,
    )
