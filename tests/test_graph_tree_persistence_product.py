"""Tests for the graph/tree/persistence construction product (PR 3d)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.analysis.manifold_graph_tree import (
    build_manifold_graph_tree,
)
from dubanevics2026topological.build import graph_tree_persistence_product as product
from dubanevics2026topological.build import saved_figure_inputs as sfi
from dubanevics2026topological.figures._layout_config import tree_horizontal_layout
from dubanevics2026topological.results import (
    MissingProductError,
    RunContext,
    load,
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    settings = load_settings()
    context = RunContext(output_root=tmp_path / "output")
    monkeypatch.setattr(product, "load_settings", lambda context=None: settings)
    return SimpleNamespace(settings=settings, context=context)


def _direct(settings: dict):
    return build_manifold_graph_tree(
        settings["analysis"]["graph_tree_persistence"],
        horizontal_layout=tree_horizontal_layout(settings),
    )


def _load(env: SimpleNamespace, settings: dict | None = None):
    return sfi.load_graph_tree_persistence_product(
        env.context, settings=settings or env.settings
    )


def test_round_trip_and_parity_with_direct_construction(env) -> None:
    path = product.build_graph_tree_persistence_product(context=env.context)
    prod = load("graph_tree_persistence", context=env.context)
    assert prod.report.status == "full" and prod.report.byte_integrity
    assert set(prod.artifacts) >= set(product.ARTIFACT_NAMES)
    loaded = _load(env)["construction"]
    direct = _direct(env.settings)
    assert loaded.sample_table.equals(direct.sample_table)
    assert loaded.edge_table.equals(direct.edge_table)
    assert loaded.grid_table.equals(direct.grid_table)
    assert loaded.thresholds == direct.thresholds
    assert loaded.critical_ids == direct.critical_ids
    assert loaded.opening_threshold == direct.opening_threshold
    assert loaded.layout.node_table.equals(direct.layout.node_table)
    assert loaded.layout.edge_table.equals(direct.layout.edge_table)
    assert loaded.tree.persistence().equals(direct.tree.persistence())
    for threshold in direct.thresholds:
        assert loaded.persistence_events(threshold) == direct.persistence_events(
            threshold
        )
        assert loaded.sweep_state(threshold) == direct.sweep_state(threshold)
    assert product.build_graph_tree_persistence_product(context=env.context) == path


def test_load_does_not_recompute(env, monkeypatch: pytest.MonkeyPatch) -> None:
    product.build_graph_tree_persistence_product(context=env.context)

    def forbidden(*args, **kwargs):
        raise AssertionError("loading must not rebuild the construction")

    monkeypatch.setattr(product, "build_manifold_graph_tree", forbidden)
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.manifold_graph_tree."
        "build_manifold_graph_tree",
        forbidden,
    )
    assert isinstance(_load(env)["construction"].sample_table, pl.DataFrame)


def test_saved_product_renders_identically_to_direct_construction(env) -> None:
    from dubanevics2026topological import build
    from dubanevics2026topological.figures.registry import get_spec
    from topographer.topology import build_tree_metrics

    product.build_graph_tree_persistence_product(context=env.context)
    spec = get_spec("graph_tree_persistence")
    inputs = build.load_figure_inputs((spec,), env.settings, context=env.context)
    direct_construction = _direct(env.settings)
    direct_tree = direct_construction.tree
    counts = {
        str(direct_tree.graph.get_node_data(int(row["source_tree_node_index"]))["vertex_id"]):
        int(row["branch_vertex_count"])
        for row in build_tree_metrics(direct_tree).branch_table.iter_rows(named=True)
    }
    assert inputs["graph_tree_persistence"].tree.branch_counts == counts
    assert load("graph_tree_persistence", context=env.context).manifest.scientific_identity == product._identity(env.settings)
    saved = spec.render(inputs, env.settings, "vanilla").to_dict()
    direct = spec.render(
        {"graph_tree_persistence": direct_construction}, env.settings, "vanilla"
    ).to_dict()
    assert saved == direct


def test_parameter_change_is_an_identity_mismatch(env) -> None:
    product.build_graph_tree_persistence_product(context=env.context)
    changed = {
        **env.settings,
        "analysis": {
            **env.settings["analysis"],
            "graph_tree_persistence": {
                **env.settings["analysis"]["graph_tree_persistence"],
                "gaussian_width": 0.25,
            },
        },
    }
    with pytest.raises(MissingProductError):
        _load(env, changed)


def test_layout_change_is_an_identity_mismatch(env) -> None:
    product.build_graph_tree_persistence_product(context=env.context)
    other = (
        "leaf_order"
        if tree_horizontal_layout(env.settings) != "leaf_order"
        else "subtree_extremum"
    )
    changed = {
        **env.settings,
        "figures": {
            **env.settings["figures"],
            "rendering": {
                **env.settings["figures"]["rendering"],
                "tree_horizontal_layout": other,
            },
        },
    }
    with pytest.raises(MissingProductError):
        _load(env, changed)


def test_missing_product_fails_closed(env) -> None:
    with pytest.raises(MissingProductError, match="graph-tree-persistence-product"):
        _load(env)


def test_tampered_artifact_fails_closed(env) -> None:
    path = product.build_graph_tree_persistence_product(context=env.context)
    with (path / "persistence.parquet").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(MissingProductError):
        _load(env)


def test_stale_code_hash_fails_closed(env, monkeypatch: pytest.MonkeyPatch) -> None:
    product.build_graph_tree_persistence_product(context=env.context)
    hashes = product._input_hashes()
    monkeypatch.setattr(
        product, "_input_hashes", lambda: {**hashes, "figures/_layout_config.py": "0"}
    )
    with pytest.raises(MissingProductError):
        _load(env)
