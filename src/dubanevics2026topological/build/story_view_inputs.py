"""Saved-product inputs of the story views (Decision 10): trees, ridge sweeps, graph entropy.

Each loader reads saved products only (the same keyed resolvers as the rest of the build) and
returns a payload ``figures/story_views.py`` draws from; a changed appearance setting
re-renders and never rebuilds science. Which saved trees a view reads is configuration:
``[story_views]`` in ``config/figures.toml`` (``nominal_trees``, ``band_trees``,
``graph_codon_entropy``), each a state of ``data/processed``. Ridge products are every
``data/processed/ridge_entropy_sweeps/<state>`` that exists.

A band tree's "surviving peaks" count is its persistence-pair count, so it includes the
global maximum's pair (closed at the global minimum by convention). ``bootstrap_band`` reads
2c from the checkpoint summary the band tree's manifest names (``build/d10_band_inputs.py``)
and checks the surviving set against it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis.ridge_entropy import RidgeEntropy
from ..figures.band_survival import classify_peaks, peak_nodes
from ..results import PresentationProducts, RunContext, processed
from .d10_band_inputs import load_two_c
from .d10_tree_inputs import (
    SavedTreeInputs,
    landscape_source_tree,
    load_saved_tree_inputs,
)
from .landscape_summary_inputs import _parent, _verified

SECTION = "story_views"


@dataclass(frozen=True)
class BandTreeInputs:
    """A band tree with its nominal parent, the 2c survival classification and peak nodes."""

    nominal_label: str
    nominal: SavedTreeInputs
    band: SavedTreeInputs
    survival: Any
    peaks: Any


def story_view_labels(
    settings: Mapping[str, Any], context: RunContext
) -> dict[str, tuple[str, ...]]:
    """``{"nominal": labels, "band": labels}``: the tree labels of the processed states of
    ``[story_views]`` in config/figures.toml."""
    table = settings["figures"][SECTION]
    return {
        "nominal": tuple(
            processed.load("tree", state, context).label
            for state in table["nominal_trees"]
        ),
        "band": tuple(
            processed.load("tree", state, context).label
            for state in table["band_trees"]
        ),
    }


def band_two_c(context: RunContext, band_label: str) -> float:
    """2c of a band tree, from the checkpoint summary its manifest names."""
    manifest = PresentationProducts(context).load("tree", band_label).manifest
    return load_two_c(
        manifest, [processed.path("bootstrap", root=processed.processed_root(context))]
    )


def band_nominal_label(context: RunContext, band_label: str) -> str:
    """Label of the nominal tree a band tree was simplified from."""
    products = PresentationProducts(context)
    return landscape_source_tree(products, products.load("tree", band_label)).label


def load_band_tree_inputs(
    context: RunContext,
    band_label: str,
    cached: Callable[[str], SavedTreeInputs],
) -> BandTreeInputs:
    """The band tree, its nominal parent, the survival of each peak and the surviving nodes."""
    nominal_label = band_nominal_label(context, band_label)
    nominal = cached(nominal_label)
    band = cached(band_label)
    survival = classify_peaks(
        nominal.persistence_table,
        band.persistence_table,
        band_two_c(context, band_label),
    )
    return BandTreeInputs(
        nominal_label=nominal_label,
        nominal=nominal,
        band=band,
        survival=survival,
        peaks=peak_nodes(band.tree_analysis.plot_data.node_table, survival.surviving),
    )


def band_survival(
    context: RunContext,
    band_label: str,
    loaded: Mapping[str, SavedTreeInputs],
    loader: Callable[[RunContext, str], SavedTreeInputs] = load_saved_tree_inputs,
) -> tuple[SavedTreeInputs, SavedTreeInputs, Any]:
    """``(nominal, band, survival)`` of a band tree; inputs not in ``loaded`` are read."""
    item = load_band_tree_inputs(
        context,
        band_label,
        lambda label: loaded.get(label) or loader(context, label),
    )
    return item.nominal, item.band, item.survival


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def load_tree_view_inputs(
    context: RunContext,
    settings: Mapping[str, Any],
    loader: Callable[[RunContext, str], SavedTreeInputs] = load_saved_tree_inputs,
) -> dict[str, Any]:
    """Nominal and band trees of ``[story_views]``, for the three tree-based families.

    Returns ``{"nominal": {label: SavedTreeInputs}, "band": {label: BandTreeInputs},
    "commit": str | None, "product_ids": {label: tree scientific id}}``; each saved tree
    is read once.
    """
    labels = story_view_labels(settings, context)
    cache: dict[str, SavedTreeInputs] = {}

    def cached(label: str) -> SavedTreeInputs:
        if label not in cache:
            cache[label] = loader(context, label)
        return cache[label]

    nominal = {label: cached(label) for label in labels["nominal"]}
    band = {
        label: load_band_tree_inputs(context, label, cached)
        for label in labels["band"]
    }
    return {
        "nominal": nominal,
        "band": band,
        "commit": _git_commit(context.root),
        "product_ids": {label: item.tree_id for label, item in cache.items()},
    }


def ridge_entropy_states(context: RunContext) -> tuple[str, ...]:
    """The states with a ``processed/ridge_entropy_sweeps/<state>`` product, in ``STATES`` order."""
    root = processed.path(
        "ridge_entropy_sweeps", root=processed.processed_root(context)
    )
    present = {manifest.parent.name for manifest in root.glob("*/manifest.json")}
    return tuple(state for state in processed.STATES if state in present)


def load_ridge_entropy_inputs(context: RunContext) -> dict[str, Any]:
    """``{label: {"entropy": RidgeEntropy, "functional": {ridge: count}}}`` per processed sweep."""
    loader = processed.product_loader(context)
    payload: dict[str, Any] = {}
    for state in ridge_entropy_states(context):
        saved = _verified(loader, "ridge_entropy_sweeps", state)
        entropy = RidgeEntropy.from_artifacts(saved.artifacts)
        ridges = _parent(loader, saved, "ridges", state)
        summary = pl.read_parquet(ridges.artifacts["ridges.parquet"])
        payload[saved.label] = {
            "product_id": saved.manifest.scientific_id_sha256,
            "entropy": entropy,
            "functional": dict(
                zip(summary["ridge_name"], summary["functional_count"], strict=True)
            ),
        }
    return payload


def load_graph_codon_entropy_inputs(
    context: RunContext, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Entropy of the Full and Published landscapes named by ``[story_views.graph_codon_entropy]``."""
    from ..analysis.graph_codon_entropy import build_graph_codon_entropy

    states = dict(settings["figures"][SECTION]["graph_codon_entropy"])
    sweep_points = int(settings["analysis"]["codon_entropy"]["sweep_points"])
    landscapes = {
        name: processed.load("landscape", states[name], context)
        for name in ("full", "published")
    }
    vertices = {
        name: pl.read_parquet(landscapes[name].artifacts["vertices.parquet"]).select(
            "dna_sequence", "raw_fitness"
        )
        for name in landscapes
    }
    return {
        "product_ids": {
            name: landscape.manifest.scientific_id_sha256
            for name, landscape in landscapes.items()
        },
        "entropy": build_graph_codon_entropy(
            vertices["full"], vertices["published"], sweep_points=sweep_points
        ),
        "sweep_points": sweep_points,
        "labels": {name: landscape.label for name, landscape in landscapes.items()},
    }
