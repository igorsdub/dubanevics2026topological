"""Publish the leaves of the seven states, and the band trees of the two BY states.

Each nominal state's leaves are built from the processed nominal tree (``landscape-products``)
and the selected landscape rebuilt in memory, as ``tree-layout-products`` does. A BY state's
band tree is simplified from the processed bootstrap draw evidence (``bootstrap/<state>``),
whose Candidate vertex set must be the one of the rebuilt graph; the band tree and its leaves
are published beside the nominal ones. Nothing is drawn.
"""

from __future__ import annotations

import json
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from ..analysis import describe, load_settings
from ..analysis.leaves import direct_band_parent, save_band_tree
from ..landscape import build_selected_landscape, extract_nominal_merge_tree
from ..landscape.build_products import _configured_input_path
from ..results import RunContext, processed
from ..results import load as load_product
from ..simplification import build_band_simplified_view
from ..simplification.selected_leaf_simplification import absorb_terminal_branches
from ..threshold.fdr_selection import load_fdr_snapshot, require_full_population
from ..threshold.functional_threshold import (
    compute_candidate_functional_threshold,
    declaration_identity,
)
from .tree_layout_products import BOOTSTRAP_IDENTITY, BY, _vertex_identities

# State -> the population name the leaves summary records.
POPULATIONS = {
    "full": "Full",
    "published": "Published",
    "reflected": "Median_reflected",
    "published_by": "Published_BY",
    "reflected_by": "Median_reflected_BY",
}
NOMINAL_STATES = tuple(POPULATIONS)


def _progress(state: str, event: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{timestamp} leaves {state}: {event}", flush=True)


def _check_assignments(state: str, view: Any, graph_members: set[str]) -> None:
    assigned = view.assignments["dna_sequence"].to_list()
    if len(assigned) != len(set(assigned)) or set(assigned) != graph_members:
        raise ValueError(f"{state}: tree assignments differ from the graph")


def _save_leaves(
    state: str,
    tree: Any,
    parent: Any,
    graph_members: set[str],
    context: RunContext,
    *,
    tree_state: str,
    hills: pl.DataFrame | None = None,
) -> Path:
    product = describe(
        products={"leaves"},
        leaf_tree=tree,
        leaf_population=POPULATIONS[state.removesuffix("_band")],
        leaf_tree_state=tree_state,
        leaf_hills=hills,
    ).leaves
    if product.summary["tree_member_count"] != len(graph_members):
        raise ValueError(f"{state}: graph/tree count differs")
    if not set(product.members["member_dna_sequence"]).issubset(graph_members):
        raise ValueError(f"{state}: members absent from the graph")
    path = product.save(parent=parent, context=context)
    restored = type(product).load(path.name, context=context)
    if not restored.table.equals(product.table) or not restored.members.equals(
        product.members
    ):
        raise ValueError(f"{state}: the saved leaves differ from the computed ones")
    return path


def _band_draws(state: str, prepared: Any, context: RunContext) -> tuple[Any, dict, pl.DataFrame]:
    """The processed draw evidence of a BY state, checked against the rebuilt graph."""
    draws = processed.load("bootstrap", state, context)
    expected = {"thr": BY[state][1], **BOOTSTRAP_IDENTITY}
    identity = draws.manifest.scientific_identity
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{state}: the bootstrap evidence is not the {expected} run")
    if (
        prepared.diagnostics["functional_set_sha256"]
        != identity["functional_set_sha256"]
    ):
        raise ValueError(f"{state}: selected graph differs from the bootstrap selection")
    summary = json.loads(draws.artifacts["checkpoint/summary.json"].read_text())
    hills = pl.read_parquet(draws.artifacts["checkpoint/hills.parquet"])
    return draws, summary, hills


def build_leaves_products(context: RunContext) -> dict[str, Path]:
    """Publish ``leaves`` for every state and ``tree`` plus ``leaves`` for the two band states."""
    settings = load_settings(context=context)
    config = settings["analysis"]
    wt = pl.read_parquet(_configured_input_path(context, config["dataset"]["fitness_wt"]))
    diff = pl.read_parquet(
        _configured_input_path(context, config["dataset"]["fitness_diff"])
    )
    wild_type = str(wt[0, "dna_from"])
    _progress("all", "building full network")
    network = Papkou2023Workflow(
        landscape="full", fitness_wt=wt, fitness_diff=diff
    ).build_network()
    raw = [float(node[network.scalar_attr]) for node in network.graph.nodes()]
    thresholds = {
        "full": min(raw) - 1.0,
        "published": float(config["clamped"]["published_functional_threshold"]),
        "reflected": compute_candidate_functional_threshold(
            raw, {"rule": "median_reflected", "decimals": 6}
        ).threshold,
    }
    outputs: dict[str, Path] = {}
    for state in NOMINAL_STATES:
        began = time.monotonic()
        _progress(state, "start")
        nominal = processed.load("tree", state, context)
        classes = functional = None
        if state in BY:
            with (context.root / "config/variants" / BY[state][0]).open("rb") as stream:
                declaration = declaration_identity(
                    tomllib.load(stream)["clamped"]["candidate_functional_threshold"]
                )
            snapshot = load_fdr_snapshot(declaration, wild_type=wild_type, root=context.root)
            thresholds[state] = float(snapshot.summary["threshold"]["rounded"])
            classes = dict(
                zip(snapshot.table["dna_sequence"], snapshot.table["fdr_class"], strict=True)
            )
            require_full_population(classes, list(_vertex_identities(network).values()))
            functional = frozenset(
                dna for dna, category in classes.items() if category == "functional"
            )
        prepared = build_selected_landscape(
            network,
            thresholds[state],
            functional_sequences=functional,
            functional_classes=classes,
        )
        analysis = extract_nominal_merge_tree(
            prepared, name=state, wild_type_sequence=wild_type
        )
        tree = analysis.trees["split"]
        graph_members = set(_vertex_identities(prepared.network).values())
        _check_assignments(state, absorb_terminal_branches(tree, set()), graph_members)
        outputs[state] = _save_leaves(
            state, tree, nominal, graph_members, context, tree_state="nominal"
        )
        _progress(state, f"nominal leaves published in {time.monotonic() - began:.1f}s")
        if state not in BY:
            continue
        band_state = f"{state}_band"
        draws, summary, hills = _band_draws(state, prepared, context)
        band_parent = direct_band_parent(draws, summary)
        view = build_band_simplified_view(tree, hills)
        _check_assignments(band_state, view, graph_members)
        tree_path = save_band_tree(view, band=band_parent, nominal=nominal, context=context)
        band_tree = load_product("tree", tree_path.name, context=context)
        references = [
            ref
            for ref in band_tree.manifest.parents
            if ref.get("product_type") == "direct_band_identity"
        ]
        if (
            len(references) != 1
            or references[0]["scientific_id_sha256"] != band_parent.scientific_id_sha256
        ):
            raise ValueError(f"{band_state}: band identity parent mismatch")
        outputs[f"{band_state}_tree"] = tree_path
        outputs[band_state] = _save_leaves(
            band_state, tree, band_tree, graph_members, context, tree_state="band", hills=hills
        )
        _progress(band_state, "band tree and leaves published")
    return outputs
