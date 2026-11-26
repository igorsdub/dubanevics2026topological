"""Composite panel products must be verified against their own variant."""

from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from dubanevics2026topological.build import bootstrap_composite_product as product


def _case():
    settings = {"analysis": {"clamped": {"candidate_functional_threshold": {"rule": "fixed_cutoff", "decimals": 6, "value": -0.5}}}}
    selection = SimpleNamespace(rule="fixed_cutoff", functional_sequences=None, threshold=SimpleNamespace(threshold=-0.5))
    return settings, selection


def test_peak_se_domain_changes_global_composite_presentation_identity():
    old = {"figures": {"figures": {"peak_uncertainty": {"standard_error_domain": [0.0, 0.28]}}}}
    new = {"figures": {"figures": {"peak_uncertainty": {"standard_error_domain": [-0.006, 0.126]}}}}
    assert product.presentation_digest(old) != product.presentation_digest(new)


def test_loader_accepts_verified_matching_product(tmp_path, monkeypatch):
    settings, selection = _case()
    identity = product.expected_panel_identity("point_papkou_original", settings, selection)
    chart = tmp_path / "panel.json"
    chart.write_text('{"mark":"point","data":{"values":[]}}')
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(scientific_identity={"panel": identity, "band_sha256": "band", "presentation_sha256": product.presentation_digest(settings), "renderer_sha256": "renderer"}, data_provenance={"hills": "digest"}),
        report=SimpleNamespace(status="full", byte_integrity=True),
        artifacts={"panel.json": chart},
    )
    monkeypatch.setattr(product, "load", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(product, "active_band_identity", lambda *args: "band")
    monkeypatch.setattr(product, "renderer_dependency_digest", lambda: "renderer")
    monkeypatch.setattr(product, "_bootstrap_inputs", lambda *args: {"hills": "digest"})
    result = product.load_panel_product(SimpleNamespace(variant_name="point_papkou_original"), settings, selection, object())
    assert result["mark"] == "point"
    assert result["_composite_threshold"] == identity["threshold"]


def test_loader_rejects_different_variant(tmp_path, monkeypatch):
    settings, selection = _case()
    identity = product.expected_panel_identity("different_variant", settings, selection)
    chart = tmp_path / "panel.json"
    chart.write_text('{"mark":"point","data":{"values":[]}}')
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(scientific_identity={"panel": identity, "band_sha256": "band", "presentation_sha256": product.presentation_digest(settings)}),
        report=SimpleNamespace(status="full", byte_integrity=True),
        artifacts={"panel.json": chart},
    )
    monkeypatch.setattr(product, "load", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(product, "active_band_identity", lambda *args: "band")
    with pytest.raises(ValueError, match="different scientific identity"):
        product.load_panel_product(SimpleNamespace(variant_name="point_papkou_original"), settings, selection, object())


def test_loader_rejects_unverified_product(monkeypatch):
    settings, selection = _case()
    loaded = SimpleNamespace(report=SimpleNamespace(status="partial", byte_integrity=False))
    monkeypatch.setattr(product, "load", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(product, "active_band_identity", lambda *args: "band")
    with pytest.raises(ValueError, match="not fully verified"):
        product.load_panel_product(SimpleNamespace(variant_name="point_papkou_original"), settings, selection, object())


def test_loader_rejects_different_band(tmp_path, monkeypatch):
    settings, selection = _case()
    identity = product.expected_panel_identity("point_papkou_original", settings, selection)
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(scientific_identity={"panel": identity, "band_sha256": "old-band"}),
        report=SimpleNamespace(status="full", byte_integrity=True),
        artifacts={},
    )
    monkeypatch.setattr(product, "load", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(product, "active_band_identity", lambda *args: "current-band")
    with pytest.raises(ValueError, match="band identity differs"):
        product.load_panel_product(SimpleNamespace(variant_name="point_papkou_original"), settings, selection, object())


def test_loader_rejects_changed_checkpoint_artifacts(monkeypatch):
    settings, selection = _case()
    identity = product.expected_panel_identity("point_papkou_original", settings, selection)
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            scientific_identity={"panel": identity, "band_sha256": "band", "presentation_sha256": product.presentation_digest(settings), "renderer_sha256": "renderer"},
            data_provenance={"hills": "old"},
        ),
        report=SimpleNamespace(status="full", byte_integrity=True),
        artifacts={},
    )
    monkeypatch.setattr(product, "load", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(product, "active_band_identity", lambda *args: "band")
    monkeypatch.setattr(product, "renderer_dependency_digest", lambda: "renderer")
    monkeypatch.setattr(product, "_bootstrap_inputs", lambda *args: {"hills": "new"})
    with pytest.raises(ValueError, match="checkpoint artifacts differ"):
        product.load_panel_product(SimpleNamespace(variant_name="point_papkou_original"), settings, selection, object())


def test_composite_loader_uses_a_separate_context_for_each_variant(tmp_path, monkeypatch):
    parent = SimpleNamespace(root=tmp_path, output_root=tmp_path / "output", processed_root=tmp_path / "processed", raw_data_dir=tmp_path / "raw", barseq_export_dir=None)
    contexts = []

    def context_factory(**kwargs):
        context = SimpleNamespace(**kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(product, "RunContext", context_factory)
    monkeypatch.setattr(product, "load_settings", lambda context: {"variant": context.variant_name})
    monkeypatch.setattr(product, "resolve_candidate_selection", lambda settings, context: context.variant_name)
    monkeypatch.setattr(product, "checkpoint_dir", lambda context: context.output_root / "checkpoint")
    monkeypatch.setattr(product, "checkpoint_lock", lambda *args: nullcontext())
    monkeypatch.setattr(product, "load_cached_bootstrap_smoothed_tree", lambda **kwargs: kwargs["context"].variant_name)
    monkeypatch.setattr(product, "load_panel_product", lambda context, settings, selection, checkpoint: {"variant": checkpoint})
    panels = product.load_all_panel_products(parent)
    assert [context.variant_name for context in contexts] == [name for name, _ in product.PANEL_ORDER]
    assert len({context.output_root for context in contexts}) == 4
    assert all(panels[context.variant_name] == {"variant": context.variant_name} for context in contexts)


def test_writer_refuses_tree_when_hills_do_not_match(monkeypatch):
    settings, selection = _case()
    context = SimpleNamespace(variant_name="point_papkou_original")
    monkeypatch.setattr(product, "build_candidate_tree", lambda *args, **kwargs: (object(), None))
    monkeypatch.setattr(product, "checkpoint_dir", lambda context: "checkpoint")
    monkeypatch.setattr(product, "checkpoint_lock", lambda *args: nullcontext())
    monkeypatch.setattr(product, "load_cached_bootstrap_smoothed_tree", lambda **kwargs: SimpleNamespace(hills=object()))
    monkeypatch.setattr(product, "active_band_identity", lambda *args: "band")

    def reject(*args):
        raise ValueError("hills do not match tree")

    monkeypatch.setattr(product, "check_hills_match_tree", reject)
    with pytest.raises(ValueError, match="hills do not match tree"):
        product.write_panel_product(context, settings, selection)


def test_renderer_dependency_digest_changes_with_source(tmp_path, monkeypatch):
    source = tmp_path / "renderer.py"
    source.write_text("first\n")
    monkeypatch.setattr(product, "renderer_dependency_paths", lambda: (source,))
    before = product.renderer_dependency_digest()
    source.write_text("second\n")
    assert product.renderer_dependency_digest() != before


def test_panel_product_identity_changes_with_renderer(monkeypatch):
    settings, selection = _case()
    monkeypatch.setattr(product, "renderer_dependency_digest", lambda: "first")
    before = product.panel_product_identity("point_papkou_original", settings, selection, "band")
    monkeypatch.setattr(product, "renderer_dependency_digest", lambda: "second")
    after = product.panel_product_identity("point_papkou_original", settings, selection, "band")
    assert before != after


def test_loader_rejects_stale_renderer(monkeypatch):
    settings, selection = _case()
    identity = product.expected_panel_identity("point_papkou_original", settings, selection)
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            scientific_identity={"panel": identity, "band_sha256": "band", "presentation_sha256": product.presentation_digest(settings), "renderer_sha256": "old"},
            data_provenance={"hills": "digest"},
        ),
        report=SimpleNamespace(status="full", byte_integrity=True),
        artifacts={},
    )
    monkeypatch.setattr(product, "load", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(product, "active_band_identity", lambda *args: "band")
    monkeypatch.setattr(product, "renderer_dependency_digest", lambda: "new")
    monkeypatch.setattr(product, "_bootstrap_inputs", lambda *args: {"hills": "digest"})
    with pytest.raises(ValueError, match="renderer.*differ"):
        product.load_panel_product(SimpleNamespace(variant_name="point_papkou_original"), settings, selection, object())