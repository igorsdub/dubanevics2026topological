"""Saved figure inputs must preserve the selected tree's parent lineage."""

from types import SimpleNamespace

import pytest

from dubanevics2026topological.build.saved_figure_inputs import (
    _direct_band_identity,
    _landscape_for_tree,
    load_saved_figure_inputs,
)
from dubanevics2026topological.results import MissingProductError, RunContext


def test_saved_band_requires_matching_direct_band_parent():
    digest = "a" * 64
    manifest = SimpleNamespace(
        parents=[
            {"product_type": "direct_band_identity", "scientific_id_sha256": digest}
        ],
        scientific_identity={"direct_band_identity_sha256": digest},
    )
    tree = SimpleNamespace(manifest=manifest)
    assert _direct_band_identity(tree) == digest
    manifest.scientific_identity["direct_band_identity_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="invalid direct-band identity"):
        _direct_band_identity(tree)


def test_selection_is_resolved_only_for_routed_figures(monkeypatch):
    from dubanevics2026topological.analysis import ROOT

    if not (ROOT / "data/raw/papkou2023_fitness_wt.parquet").is_file():
        pytest.skip("needs the gitignored raw tables under data/raw/ (absent in a clean public clone)")
    from dubanevics2026topological import build

    calls = []
    monkeypatch.setattr(
        build, "load_inputs", lambda names, settings: {"loaded": tuple(names)}
    )
    monkeypatch.setattr(
        build,
        "selection_source",
        lambda settings: calls.append("selection") or {"rule": "mixture_quantile"},
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs.load_saved_figure_inputs",
        lambda context, identity, include_band: {"mode": "legacy_mixture"},
    )
    other = SimpleNamespace(
        figure_id="candidate_edge_evidence", inputs={"edge_evidence"}
    )
    assert "saved_figure_inputs" not in build.load_figure_inputs([other], {})
    assert calls == []
    routed = SimpleNamespace(
        figure_id="tree_comparison", inputs={"comparison"}
    )
    assert (
        build.load_figure_inputs([routed], {})["saved_figure_inputs"]["mode"]
        == "legacy_mixture"
    )
    assert calls == ["selection"]

    # peak case: loads pooled but does not resolve selection
    calls.clear()
    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs.load_pooled_scatter_product",
        lambda *a, **k: SimpleNamespace(
            manifest=SimpleNamespace(scientific_id_sha256="p" * 64)
        ),
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs._load_pooled_scatter_payload",
        lambda *a, **k: {"dummy": True},
    )
    peak = SimpleNamespace(
        figure_id="peak_uncertainty",
        inputs={"pooled_scatter", "manuscript_quantities"},
    )
    res = build.load_figure_inputs([peak], {})
    assert "pooled_scatter" in res.get("saved_figure_inputs", {})
    assert calls == []

    # a peak spec without "pooled_scatter" does not load the product
    peak_no = SimpleNamespace(
        figure_id="peak_uncertainty", inputs={"manuscript_quantities"}
    )
    res_no = build.load_figure_inputs([peak_no], {})
    assert "pooled_scatter" not in res_no.get("saved_figure_inputs", {})


def test_rejects_landscape_with_wrong_parent_identity():
    tree = SimpleNamespace(
        manifest=SimpleNamespace(
            parents=[
                {
                    "product_type": "landscape",
                    "label": "landscape",
                    "scientific_id_sha256": "expected",
                }
            ]
        )
    )
    landscape = SimpleNamespace(
        report=SimpleNamespace(status="full", byte_integrity=True),
        manifest=SimpleNamespace(scientific_id_sha256="different"),
    )
    products = SimpleNamespace(load=lambda *args: landscape)
    with pytest.raises(ValueError, match="mismatched"):
        _landscape_for_tree(products, tree)


def test_mixture_routes_explicitly_without_saved_lookup(tmp_path):
    result = load_saved_figure_inputs(
        RunContext(root=tmp_path, env={}),
        {"rule": "mixture_quantile"},
        include_band=True,
    )
    assert result["mode"] == "legacy_mixture"


def test_reviewed_lookup_missing_never_falls_back(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise MissingProductError("missing reviewed tree")

    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs._tree_for_leaf", missing
    )
    with pytest.raises(MissingProductError, match="missing reviewed tree"):
        load_saved_figure_inputs(
            RunContext(root=tmp_path, env={}),
            {
                "rule": "fdr_fixed_cutoff",
                "method": "by",
                "q": 0.05,
                "cutoff": "papkou_original",
            },
            include_band=True,
        )


def _band_route(tmp_path, monkeypatch, band_id):
    from dubanevics2026topological.build import saved_figure_inputs as route

    nominal_id = "c" * 64
    band = SimpleNamespace(
        report=SimpleNamespace(status="full", byte_integrity=True),
        manifest=SimpleNamespace(
            scientific_id_sha256="d" * 64,
            scientific_identity={"direct_band_identity_sha256": band_id},
            parents=[
                {"product_type": "direct_band_identity", "scientific_id_sha256": band_id},
                {"product_type": "tree", "scientific_id_sha256": nominal_id},
            ],
        ),
    )
    nominal = SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256=nominal_id))
    leaves = SimpleNamespace(artifacts={"leaves.parquet": tmp_path / "leaves.parquet"})
    annotation = SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256="e" * 64))
    loaded = []

    class Products:
        def __init__(self, context):
            pass

        def load(self, kind, *args, **kwargs):
            if kind == "tree_annotations":
                return annotation
            raise AssertionError(kind)

        def landscape_population(self, landscape):
            return object()

        def tree_annotations(self, **kwargs):
            return object()

    def processed_load(kind, state, context):
        loaded.append((kind, state))
        assert kind == "leaves"
        return leaves

    monkeypatch.setattr(route, "PresentationProducts", Products)
    monkeypatch.setattr(route.processed, "load", processed_load)
    monkeypatch.setattr(route, "state_for_selection", lambda *args: "reflected_by")
    monkeypatch.setattr(
        route,
        "_tree_for_leaf",
        lambda name, context: band if name.endswith("_band") else nominal,
    )
    monkeypatch.setattr(route, "_landscape_for_tree", lambda *args: object())
    monkeypatch.setattr(route, "_check_fdr_landscape", lambda *args: None)
    monkeypatch.setattr(route, "_layout", lambda products, tree: (tree, "f" * 64))
    monkeypatch.setattr(
        "polars.read_parquet", lambda path: SimpleNamespace(select=lambda *args: object())
    )
    return route, band, loaded


def test_saved_band_must_match_the_local_bootstrap_identity(tmp_path, monkeypatch):
    band_id = "b" * 64
    route, band, loaded = _band_route(tmp_path, monkeypatch, band_id)
    context = RunContext(root=tmp_path, env={})
    result = route.load_saved_figure_inputs(
        context, {"rule": "fdr_fixed_cutoff"}, include_band=True,
        active_band_identity=band_id,
    )
    assert result["layouts"]["band"] is band
    assert result["direct_band_identity"] == band_id
    assert ("leaves", "reflected_by") in loaded
    with pytest.raises(ValueError, match="band identity differs"):
        route.load_saved_figure_inputs(
            context, {"rule": "fdr_fixed_cutoff"}, include_band=True,
            active_band_identity="a" * 64,
        )


def test_top_level_figure_loader_passes_active_checkpoint_identity(monkeypatch):
    from dubanevics2026topological import build

    checkpoint = object()
    seen = []
    monkeypatch.setattr(build, "load_inputs", lambda names, settings: {"bootstrap_smoothed": checkpoint})
    monkeypatch.setattr(build, "selection_source", lambda settings: {
        "rule": "fdr_fixed_cutoff", "method": "by", "q": 0.05, "cutoff": "median_reflected"})
    monkeypatch.setattr(build, "active_bootstrap_band_identity", lambda value: "a" * 64, raising=False)
    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs.load_saved_figure_inputs",
        lambda context, identity, **kwargs: seen.append(kwargs) or {"mode": "saved"},
    )
    spec = SimpleNamespace(figure_id="candidate_tree_bootstrap_biology", inputs={"comparison", "bootstrap_smoothed"})
    build.load_figure_inputs([spec], {})
    assert seen[-1] == {"include_band": True, "active_band_identity": "a" * 64}
    nominal = SimpleNamespace(figure_id="tree_comparison", inputs={"comparison"})
    build.load_figure_inputs([nominal], {})
    assert seen[-1] == {"include_band": False}


def test_active_checkpoint_identity_refuses_missing_draws():
    import polars as pl
    from dubanevics2026topological.build import active_bootstrap_band_identity

    checkpoint = SimpleNamespace(
        summary={"settings": {"n_replicates": 2, "seed": 1, "band_quantile": 0.95}},
        draw_stats=pl.DataFrame({"condition": ["real", "real", "flat"], "draw": [0, 1, 0]}),
    )
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        active_bootstrap_band_identity(checkpoint)
