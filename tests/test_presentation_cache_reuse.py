"""Appearance changes re-render a figure from immutable saved science."""

import shutil

from sklearn.mixture import GaussianMixture
from test_landscape_module import _network

from dubanevics2026topological import analysis as analysis_package
from dubanevics2026topological import landscape
from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.build.tree_layout import build_tree_layout
from dubanevics2026topological.figures.tree_histogram import build_tree_histogram_figure
from dubanevics2026topological.results import PresentationProducts, RunContext, load


def test_appearance_change_reuses_saved_tree_and_population(tmp_path, monkeypatch):
    context = RunContext(root=tmp_path, env={})
    prepared = landscape.build_selected_landscape(_network(), 0.0)
    analysis = landscape.extract_nominal_merge_tree(
        prepared, name="fixture", wild_type_sequence="AAAAAAAAA"
    )
    landscape_path, tree_path = landscape.save_selected_products(
        prepared,
        analysis,
        parameters={"thr": "reflected", "fdr": "by", "q": 0.05},
        context=context,
        input_data_sha256s={"fitness": "a" * 64},
        parents=(),
    )
    tree = load("tree", tree_path.name, context=context)
    selected = load("landscape", landscape_path.name, context=context)
    vertex_ids = {
        index: str(prepared.network.graph[index]["dna_sequence"])
        for index in prepared.network.graph.node_indices()
    }
    layout_path = build_tree_layout(
        analysis.trees["split"],
        tree=tree,
        context=context,
        vertex_identities=vertex_ids,
    )
    product = load("tree_layout", layout_path.name, context=context)
    products = PresentationProducts(context)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering attempted scientific recomputation")

    monkeypatch.setattr(
        "dubanevics2026topological.landscape.selected_landscape.build_papkou2023_backbone_analysis",
        forbidden,
    )
    monkeypatch.setattr(
        "dubanevics2026topological.simplification.bootstrap_smoothed_tree.build_band_simplified_view",
        forbidden,
    )
    monkeypatch.setattr(
        "dubanevics2026topological.threshold.functional_threshold.compute_candidate_functional_threshold",
        forbidden,
    )
    monkeypatch.setattr(GaussianMixture, "fit", forbidden)

    view = products.tree_layout_view(layout_path.name, tree=tree)
    population = products.landscape_population(selected)
    payload = {
        "landscapes": {"candidate_clamped": analysis},
        "diagnostics": {"candidate_clamped": prepared.diagnostics},
        "saved_layouts": {"candidate": view},
        "saved_populations": {"candidate": population},
    }
    baseline_nodes = view.plot_data.node_table.clone()
    baseline_branches = view.branch_table.clone()
    shutil.copytree(analysis_package.ROOT / "config", tmp_path / "config")
    monkeypatch.setattr(analysis_package, "ROOT", tmp_path)
    monkeypatch.delenv("FITNESSTREE_ANALYSIS_VARIANT", raising=False)
    settings = load_settings()
    figure_id = "candidate_tree_histogram"
    figures_toml = tmp_path / "config/figures.toml"
    original = figures_toml.read_text(encoding="utf-8")
    section = f"[figures.{figure_id}.tree]"
    assert original.count(section) == 1
    figures_toml.write_text(
        original.replace(section, section + "\nwidth = 777"), encoding="utf-8"
    )
    altered = load_settings()
    first = build_tree_histogram_figure(payload, settings, figure_id).to_dict()
    second = build_tree_histogram_figure(payload, altered, figure_id).to_dict()
    assert first != second
    assert products.landscape_population(selected).equals(population)
    assert view.plot_data.node_table.equals(baseline_nodes)
    assert view.branch_table.equals(baseline_branches)
    assert (
        load(
            "tree_layout", layout_path.name, context=context
        ).manifest.scientific_id_sha256
        == product.manifest.scientific_id_sha256
    )
