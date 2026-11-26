"""Publish one ``ridge_entropy_sweeps`` product per band tree from saved products only."""

from __future__ import annotations

import inspect
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import codon_entropy as codon_entropy_module
from ..analysis import codon_entropy_populations as codon_populations_module
from ..analysis import ridge_entropy as ridge_entropy_module
from ..analysis import ridges as ridges_module
from ..analysis.ridge_entropy import build_ridge_entropy
from ..analysis.ridges import RIDGE_NAMES
from ..landscape.products import _publish, _sha256
from ..results import RunContext, format_label
from . import landscape_summary_inputs
from .landscape_summary_inputs import Loader, _landscape_for, _parent, _verified
from .ridges_product import _LABEL_KEYS, _reference, _ridge_tree, _summits
from ..results.processed import LEAVES, product_loader

BAND_STATES = tuple(state for state in LEAVES if state.endswith("_band"))


def _sweep_points(context: RunContext) -> int:
    with (Path(context.root) / "config" / "analysis.toml").open("rb") as handle:
        return int(tomllib.load(handle)["codon_entropy"]["sweep_points"])


def check_membership(
    ridge_tree: Any, summits: Mapping[str, str], members: pl.DataFrame
) -> None:
    """Refuse a ridges membership that differs from the ridges the saved tree gives."""
    found = ridges_module._find_ridges(ridge_tree, summits)["ridges"]
    for name in RIDGE_NAMES:
        saved = set(members.filter(pl.col("ridge_name") == name)["member_dna_sequence"])
        if saved != found[name]:
            raise ValueError(
                f"{name} ridge membership of the ridges product differs from the tree."
            )


def build_ridge_entropy_products(
    context: RunContext,
    *,
    labels: Mapping[str, str] = LEAVES,
    loader: Loader | None = None,
) -> dict[str, Path]:
    """Save each band tree's core, KEM and TCL entropy sweeps under its tree's label keys."""
    if loader is None:
        loader = product_loader(context)

    summits = _summits(context)
    sweep_points = _sweep_points(context)
    code_paths = (
        Path(inspect.getfile(ridge_entropy_module)).resolve(),
        Path(inspect.getfile(codon_entropy_module)).resolve(),
        Path(inspect.getfile(codon_populations_module)).resolve(),
        Path(inspect.getfile(ridges_module)).resolve(),
        Path(landscape_summary_inputs.__file__).resolve(),
        Path(__file__).resolve(),
    )
    digest = {str(i): _sha256(path) for i, path in enumerate(code_paths)}
    outputs: dict[str, Path] = {}
    for state in BAND_STATES:
        leaf = _verified(loader, "leaves", labels[state])
        tree = _parent(loader, leaf, "tree", state)
        landscape = _landscape_for(loader, tree, state)
        params = {
            key: value
            for key, value in tree.manifest.scientific_identity.items()
            if key in _LABEL_KEYS
        }
        ridges = _verified(loader, "ridges", format_label("ridges", params))
        members = pl.read_parquet(ridges.artifacts["members.parquet"])
        vertices = pl.read_parquet(landscape.artifacts["vertices.parquet"])
        if set(members["member_dna_sequence"]) != set(vertices["dna_sequence"]):
            raise ValueError(f"{state}: membership is not the landscape's vertex set.")
        check_membership(_ridge_tree(tree), summits, members)
        result = build_ridge_entropy(members, sweep_points=sweep_points)
        identity = {
            **params,
            "state": state,
            "summits": summits,
            "sweep_points": sweep_points,
            "code_digest": digest,
        }
        outputs[state], _ = _publish(
            "ridge_entropy_sweeps",
            params,
            identity,
            context=context,
            input_data_sha256s={},
            parents=[_reference(tree), _reference(landscape), _reference(ridges)],
            code_paths=code_paths,
            write_artifacts=result.save,
        )
    return outputs
