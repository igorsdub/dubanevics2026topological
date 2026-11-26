"""Build and publish the five canonical landscape and nominal-tree products."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from dubanevics2026topological.results import RunContext
from dubanevics2026topological.threshold.fdr_selection import (
    load_fdr_snapshot,
    require_full_population,
)
from dubanevics2026topological.threshold.functional_threshold import (
    compute_candidate_functional_threshold,
)

from . import (
    build_selected_landscape,
    extract_nominal_merge_tree,
    save_landscape_variants,
)

FDR_VARIANTS = {
    "Published BY": "threshold=published_fdr=by_q=0.05.toml",
    "Median-reflected BY": "threshold=reflected_fdr=by_q=0.05.toml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _configured_input_path(context: RunContext, value: str) -> Path:
    """Resolve configured raw-table paths through the context's raw-data directory."""
    path = Path(value)
    if path.parts[:2] == ("data", "raw"):
        return context.raw_data_dir.joinpath(*path.parts[2:])
    return path if path.is_absolute() else context.root / path


def node_payloads(network: Any) -> list[Any]:
    """Return the vertex payloads, reading ``network.graph`` exactly once.

    topographer's ``ScalarNetwork.graph`` deep-copies the whole graph on every access,
    so it must never be read per node.
    """
    graph = network.graph
    return [graph[index] for index in graph.node_indices()]


def build_and_save_landscape_products(
    context: RunContext | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Write all five ADR 0016 landscape/tree pairs using the public module.

    The fitness tables and the processed BY classification tables are read-only
    inputs. Their SHA-256 digests enter each landscape identity.
    """
    context = context or RunContext()
    with (context.root / "config/analysis.toml").open("rb") as stream:
        config = tomllib.load(stream)
    dataset = config["dataset"]
    wt_path = _configured_input_path(context, dataset["fitness_wt"])
    diff_path = _configured_input_path(context, dataset["fitness_diff"])
    wt_table = pl.read_parquet(wt_path)
    wt_sequence = str(wt_table[0, "dna_from"])
    diff_table = pl.read_parquet(diff_path)
    network = Papkou2023Workflow(
        landscape="full", fitness_wt=wt_table, fitness_diff=diff_table
    ).build_network()
    payloads = node_payloads(network)
    raw = [float(payload[network.scalar_attr]) for payload in payloads]
    full_sequences = [payload["dna_sequence"] for payload in payloads]
    raw_inputs = {
        "fitness_wt": _sha256(wt_path),
        "fitness_diff": _sha256(diff_path),
    }
    full = build_selected_landscape(network, min(raw) - 1.0)
    published = build_selected_landscape(
        network, float(config["clamped"]["published_functional_threshold"])
    )
    reflected_threshold = compute_candidate_functional_threshold(
        raw, {"rule": "median_reflected", "decimals": 6}
    ).threshold
    reflected = build_selected_landscape(network, reflected_threshold)

    variants: dict[
        str,
        tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, str], tuple[dict[str, str], ...]],
    ] = {}
    for name, prepared, parameters in (
        ("Full", full, {"graph": "full"}),
        ("Published", published, {"thr": "published", "fdr": "none"}),
        (
            "Median-reflected",
            reflected,
            {"thr": "reflected", "fdr": "none"},
        ),
    ):
        # TODO(ticket 09/14): when point-estimate fitness products are published,
        # reference their real manifests here. Until then the source table digests
        # are the complete provenance; do not invent unresolved parent labels.
        tree = extract_nominal_merge_tree(
            prepared, name=name, wild_type_sequence=wt_sequence
        )
        variants[name] = (
            prepared,
            tree,
            parameters,
            {},
            raw_inputs,
            (),
        )

    for name, variant_file in FDR_VARIANTS.items():
        with (context.root / "config/variants" / variant_file).open("rb") as stream:
            declaration = tomllib.load(stream)["clamped"][
                "candidate_functional_threshold"
            ]
        snapshot = load_fdr_snapshot(
            declaration, wild_type=wt_sequence, root=context.root
        )
        applied_threshold = float(snapshot.summary["threshold"]["rounded"])
        classes = dict(
            zip(
                snapshot.table["dna_sequence"].to_list(),
                snapshot.table["fdr_class"].to_list(),
                strict=True,
            )
        )
        functional = frozenset(
            sequence
            for sequence, category in classes.items()
            if category == "functional"
        )
        require_full_population(classes, full_sequences)
        prepared = build_selected_landscape(
            network,
            applied_threshold,
            functional_sequences=functional,
            functional_classes=classes,
        )
        tree = extract_nominal_merge_tree(
            prepared, name=name, wild_type_sequence=wt_sequence
        )
        table_path = context.root / declaration["input_dir"] / snapshot.source["table"]
        table_hashes = {
            "classification_table": _sha256(table_path)
        }
        threshold_kind = (
            "published"
            if declaration["cutoff"] == "papkou_original"
            else "reflected"
        )
        label_parameters = {
            "thr": threshold_kind,
            "fdr": "by",
            "q": float(declaration["q"]),
        }
        variants[name] = (
            prepared,
            tree,
            label_parameters,
            {"classification_source": snapshot.source["table"]},
            {**raw_inputs, **table_hashes},
            (),
        )

    from dubanevics2026topological.build.common import BUILD_MODULES

    code_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("products.py").resolve(),
        Path(__file__).with_name("selected_landscape.py").resolve(),
        Path(__file__).with_name("selection.py").resolve(),
        *(
            context.root / "src/dubanevics2026topological/build" / f"{name}.py"
            for name in BUILD_MODULES
        ),
        context.root
        / "src/dubanevics2026topological/threshold/fdr_selection.py",
        context.root
        / "src/dubanevics2026topological/threshold/functional_threshold.py",
    )
    return save_landscape_variants(
        variants, context=context, code_paths=code_paths
    )