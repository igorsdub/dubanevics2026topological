"""Publish the conceptual graph/split-tree/persistence construction as a saved product.

The construction is deterministic and cheap, but its split tree and Topographer
layout are scientific results, so the figure must not recompute them. This
product runs ``build_manifold_graph_tree`` once and publishes the sample, edge
and grid tables, the layout node and edge tables, the persistence pairs (in the
producer's order) and the scalar metadata. The loader rebuilds a
``ManifoldGraphTreeConstruction`` from those bytes without calling Topographer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from topographer.topology import build_tree_metrics

from ..analysis import load_settings
from ..analysis import manifold_graph_tree as construction_module
from ..analysis.manifold_graph_tree import (
    ManifoldGraphTreeConstruction,
    build_manifold_graph_tree,
)
from ..figures import _layout_config
from ..figures._layout_config import tree_horizontal_layout
from ..landscape.products import _publish
from ..results import RunContext

ARTIFACT_NAMES = (
    "sample_table.parquet",
    "edge_table.parquet",
    "grid_table.parquet",
    "layout_nodes.parquet",
    "layout_edges.parquet",
    "persistence.parquet",
    "metadata.json",
)


@dataclass(frozen=True)
class _SavedTree:
    """Saved persistence pairs and branch counts keyed by sample identity."""

    table: pl.DataFrame
    branch_counts: dict[str, int]

    def persistence(self) -> pl.DataFrame:
        return self.table


@dataclass(frozen=True)
class _SavedLayout:
    """Stands in for the Topographer layout: only its two tables are read."""

    node_table: pl.DataFrame
    edge_table: pl.DataFrame


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(settings: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "graph_tree_persistence": settings["analysis"]["graph_tree_persistence"],
        "horizontal_layout": tree_horizontal_layout(settings),
    }
    return json.loads(json.dumps(identity, sort_keys=True))


def _input_hashes() -> dict[str, str]:
    paths = {
        "analysis/manifold_graph_tree.py": Path(construction_module.__file__),
        "build/graph_tree_persistence_product.py": Path(__file__),
        "figures/_layout_config.py": Path(_layout_config.__file__),
    }
    return {key: _file_sha256(path.resolve()) for key, path in paths.items()}


def _metadata(construction: ManifoldGraphTreeConstruction) -> dict[str, Any]:
    tree = construction.tree
    metrics = build_tree_metrics(tree).branch_table
    graph = tree.graph
    branch_counts = {
        str(graph.get_node_data(int(row["source_tree_node_index"]))["vertex_id"]):
        int(row["branch_vertex_count"])
        for row in metrics.iter_rows(named=True)
    }
    return {
        "thresholds": list(construction.thresholds),
        "critical_ids": dict(construction.critical_ids),
        "opening_threshold": construction.opening_threshold,
        "scientific_settings": construction.scientific_settings,
        "branch_counts": branch_counts,
    }


def _construction_from_artifacts(
    artifacts: dict[str, Path],
) -> ManifoldGraphTreeConstruction:
    metadata = json.loads(artifacts["metadata.json"].read_text(encoding="utf-8"))
    return ManifoldGraphTreeConstruction(
        sample_table=pl.read_parquet(artifacts["sample_table.parquet"]),
        edge_table=pl.read_parquet(artifacts["edge_table.parquet"]),
        grid_table=pl.read_parquet(artifacts["grid_table.parquet"]),
        thresholds=tuple(float(value) for value in metadata["thresholds"]),
        critical_ids=dict(metadata["critical_ids"]),
        graph=None,  # type: ignore[arg-type]  # never read by figure or composer
        tree=_SavedTree(
            pl.read_parquet(artifacts["persistence.parquet"]),
            {str(key): int(value) for key, value in metadata["branch_counts"].items()},
        ),  # type: ignore[arg-type]
        layout=_SavedLayout(
            pl.read_parquet(artifacts["layout_nodes.parquet"]),
            pl.read_parquet(artifacts["layout_edges.parquet"]),
        ),
        opening_threshold=float(metadata["opening_threshold"]),
        scientific_settings=metadata["scientific_settings"],
    )


def build_graph_tree_persistence_product(*, context: RunContext | None = None) -> Path:
    """Build the graph_tree_persistence product."""
    context = context or RunContext()
    settings = load_settings(context=context)
    identity = _identity(settings)
    construction = build_manifold_graph_tree(
        settings["analysis"]["graph_tree_persistence"],
        horizontal_layout=tree_horizontal_layout(settings),
    )

    def write_artifacts(directory: Path) -> None:
        construction.sample_table.write_parquet(directory / "sample_table.parquet")
        construction.edge_table.write_parquet(directory / "edge_table.parquet")
        construction.grid_table.write_parquet(directory / "grid_table.parquet")
        construction.layout.node_table.write_parquet(directory / "layout_nodes.parquet")
        construction.layout.edge_table.write_parquet(directory / "layout_edges.parquet")
        construction.tree.persistence().write_parquet(directory / "persistence.parquet")
        (directory / "metadata.json").write_text(
            json.dumps(_metadata(construction), sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "graph_tree_persistence",
        {},
        identity,
        context=context,
        input_data_sha256s=_input_hashes(),
        parents=(),
        code_paths=(
            Path(construction_module.__file__).resolve(),
            Path(__file__).resolve(),
        ),
        write_artifacts=write_artifacts,
    )
    return path
