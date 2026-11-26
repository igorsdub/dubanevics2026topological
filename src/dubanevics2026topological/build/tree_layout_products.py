"""Build layouts for the seven reviewed leaf/tree states from saved products."""

from __future__ import annotations

import json
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from ..analysis import load_settings
from ..analysis.leaves import direct_band_parent
from ..landscape import build_selected_landscape, extract_nominal_merge_tree
from ..landscape.build_products import _configured_input_path
from ..results import RunContext, processed
from ..simplification.bootstrap_smoothed_tree import check_hills_match_tree
from ..tables import _public_table
from ..threshold.fdr_selection import load_fdr_snapshot, require_full_population
from ..threshold.functional_threshold import (
    compute_candidate_functional_threshold,
    declaration_identity,
)
from .tree_annotations import build_tree_annotations
from .tree_layout import build_tree_layout

# BY state -> (variant overlay, threshold kind). The bootstrap evidence of a BY state is the
# processed bootstrap/<state> product.
BY = {
    "published_by": ("threshold=published_fdr=by_q=0.05.toml", "published"),
    "reflected_by": ("threshold=reflected_fdr=by_q=0.05.toml", "reflected"),
}

BOOTSTRAP_IDENTITY = {"fdr": "by", "q": 0.05, "draws": 10000, "seed": 20261019}


def _progress(state: str, event: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{timestamp} tree-layout {state}: {event}", flush=True)


def _vertex_identities(network: Any) -> dict[int, str]:
    """Read identities from one defensive graph snapshot."""
    graph = network.graph
    return {index: str(graph[index]["dna_sequence"]) for index in graph.node_indices()}


def _tree_for_leaf(name: str, context: RunContext):
    """Load the processed tree of a state, then check that its leaves point at that tree.

    The tree is read from ``processed/tree/<state>`` directly: the layout stage needs the
    leaves only as the parent check, never as the route to the tree.
    """
    parent = processed.load("tree", name, context)
    leaf = processed.load("leaves", name, context)
    refs = [ref for ref in leaf.manifest.parents if ref.get("product_type") == "tree"]
    if len(refs) != 1:
        raise ValueError(f"Leaves for {name} must identify exactly one tree.")
    if parent.manifest.scientific_id_sha256 != refs[0]["scientific_id_sha256"]:
        raise ValueError(f"Leaves for {name} point at a mismatched tree.")
    return parent


def _require_saved_pairs(name: str, tree: Any, analysis: Any, nominal: Any) -> None:
    pairs = _public_table("persistence_table", tree.persistence_table, analysis)
    if not pairs.equals(
        pl.read_parquet(nominal.artifacts["persistence_pairs.parquet"])
    ):
        raise ValueError(
            f"Nominal persistence pairs differ from the saved parent for {name}."
        )


def build_tree_layouts(context: RunContext) -> dict[str, Path]:
    """Rebuild layout inputs, reconcile with the processed trees and publish layouts."""
    settings = load_settings(context=context)
    config = settings["analysis"]
    network = None
    wild_type = None
    thresholds: dict[str, float] = {}

    def ensure_network() -> None:
        nonlocal network, wild_type
        if network is not None:
            return
        _progress("all", "building full network for in-memory tree reconstruction")
        wt = pl.read_parquet(
            _configured_input_path(context, config["dataset"]["fitness_wt"])
        )
        diff = pl.read_parquet(
            _configured_input_path(context, config["dataset"]["fitness_diff"])
        )
        wild_type = str(wt[0, "dna_from"])
        network = Papkou2023Workflow(
            landscape="full", fitness_wt=wt, fitness_diff=diff
        ).build_network()
        raw = [float(node[network.scalar_attr]) for node in network.graph.nodes()]
        thresholds.update(
            full=min(raw) - 1.0,
            published=float(config["clamped"]["published_functional_threshold"]),
            reflected=compute_candidate_functional_threshold(
                raw, {"rule": "median_reflected", "decimals": 6}
            ).threshold,
        )
        _progress("all", "full network ready")

    outputs: dict[str, Path] = {}
    for name in ("full", "published", "reflected", "published_by", "reflected_by"):
        _progress(name, "start")
        nominal = _tree_for_leaf(name, context)
        _progress(name, "parent resolved and verified")
        reconstruction_start = time.monotonic()
        ensure_network()
        classes = None
        functional = None
        if name in BY:
            variant = BY[name][0]
            with (context.root / "config/variants" / variant).open("rb") as stream:
                variant_declaration = declaration_identity(
                    tomllib.load(stream)["clamped"]["candidate_functional_threshold"]
                )
            snapshot = load_fdr_snapshot(
                variant_declaration, wild_type=wild_type, root=context.root
            )
            thresholds[name] = float(snapshot.summary["threshold"]["rounded"])
            classes = dict(
                zip(
                    snapshot.table["dna_sequence"],
                    snapshot.table["fdr_class"],
                    strict=True,
                )
            )
            require_full_population(classes, list(_vertex_identities(network).values()))
            functional = frozenset(
                dna for dna, category in classes.items() if category == "functional"
            )
        prepared = build_selected_landscape(
            network,
            thresholds[name],
            functional_sequences=functional,
            functional_classes=classes,
        )
        _progress(name, "selected landscape ready; building nominal tree and layout")
        analysis = extract_nominal_merge_tree(
            prepared, name=name, wild_type_sequence=wild_type
        )
        vertex_identities = _vertex_identities(prepared.network)
        _progress(
            name,
            f"in-memory reconstruction complete in {time.monotonic() - reconstruction_start:.1f}s",
        )
        tree = analysis.trees["split"]
        _require_saved_pairs(name, tree, analysis, nominal)
        outputs[name] = build_tree_layout(
            tree,
            tree=nominal,
            context=context,
            vertex_identities=vertex_identities,
            progress=lambda event, state=name: _progress(state, event),
        )
        if name not in BY:
            continue
        outputs[f"{name}_annotations"] = build_tree_annotations(
            tree, tree=nominal, layout_label=outputs[name].name, context=context
        )
        _progress(name, "annotations published")
        band_name = f"{name}_band"
        _progress(band_name, "start")
        band = _tree_for_leaf(band_name, context)
        _progress(band_name, "parent resolved and verified")
        threshold_kind = BY[name][1]
        draws = processed.load("bootstrap", name, context)
        expected_identity = {"thr": threshold_kind, **BOOTSTRAP_IDENTITY}
        if (
            any(
                draws.manifest.scientific_identity.get(key) != value
                for key, value in expected_identity.items()
            )
            or nominal.manifest.scientific_identity["landscape_selection"][
                "functional_set_sha256"
            ]
            != draws.manifest.scientific_identity["functional_set_sha256"]
            or (
                prepared is not None
                and prepared.diagnostics["functional_set_sha256"]
                != draws.manifest.scientific_identity["functional_set_sha256"]
            )
        ):
            raise ValueError(f"Bootstrap and selected graph disagree for {name}.")
        summary = json.loads(draws.artifacts["checkpoint/summary.json"].read_text())
        band_identity = direct_band_parent(draws, summary)
        if not any(
            ref.get("product_type") == "direct_band_identity"
            and ref.get("scientific_id_sha256") == band_identity.scientific_id_sha256
            for ref in band.manifest.parents
        ):
            raise ValueError(f"Band identity differs for {name}.")
        hills = pl.read_parquet(draws.artifacts["checkpoint/hills.parquet"])
        check_hills_match_tree(hills, tree)
        _progress(band_name, "bootstrap evidence verified; building band layout")
        outputs[f"{name}_band"] = build_tree_layout(
            tree,
            tree=band,
            nominal_tree=nominal,
            hills=hills,
            context=context,
            vertex_identities=vertex_identities,
            progress=lambda event, state=band_name: _progress(state, event),
        )
    return outputs
