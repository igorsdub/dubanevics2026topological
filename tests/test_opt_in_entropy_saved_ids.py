"""The opt-in entropy figures register their saved-product ids for the render record.

``build_figures`` reads ``saved_figure_inputs["product_ids"][name]`` for every
``SAVED_PRODUCT_INPUTS`` name a figure declares; ``graph_codon_entropy`` and
``ridge_entropy`` once registered none, so rendering failed with ``KeyError``.
"""

from __future__ import annotations

from types import SimpleNamespace


from dubanevics2026topological import build
from dubanevics2026topological.build import story_view_inputs
from dubanevics2026topological.figures.registry import (
    FIGURE_SPECS,
    SAVED_PRODUCT_INPUTS,
)


def _manifest(identity: str) -> SimpleNamespace:
    return SimpleNamespace(scientific_id_sha256=identity)


def test_load_figure_inputs_registers_an_id_for_every_declared_saved_input(
    monkeypatch,
) -> None:
    ridge = {"label_a": {"product_id": "ridge-a", "entropy": object(), "functional": {}}}
    graph = {"product_ids": {"full": "full-id", "published": "pub-id"}}
    monkeypatch.setattr(story_view_inputs, "load_ridge_entropy_inputs", lambda ctx: ridge)
    monkeypatch.setattr(
        story_view_inputs, "load_graph_codon_entropy_inputs", lambda ctx, settings: graph
    )
    specs = [
        spec
        for spec in FIGURE_SPECS
        if spec.figure_id in {"graph_codon_entropy", "ridge_entropy"}
    ]
    assert len(specs) == 2
    inputs = build.load_figure_inputs(specs, {}, context=SimpleNamespace())
    ids = inputs["saved_figure_inputs"]["product_ids"]
    for spec in specs:
        for name in SAVED_PRODUCT_INPUTS & spec.inputs:
            assert name in ids, (spec.figure_id, name)
    assert ids["ridge_entropy"] == {"label_a": "ridge-a"}
    assert ids["graph_codon_entropy"] == {"full": "full-id", "published": "pub-id"}


def test_ridge_loader_reports_each_sweeps_scientific_id(monkeypatch) -> None:
    import polars as pl

    class Entropy:
        @staticmethod
        def from_artifacts(artifacts):
            return "entropy"

    saved = SimpleNamespace(
        artifacts={}, manifest=_manifest("sweep-id"), label="lab",
        report=SimpleNamespace(status="full", byte_integrity=True),
    )
    monkeypatch.setattr(story_view_inputs, "ridge_entropy_states", lambda context: ("full",))
    monkeypatch.setattr(story_view_inputs, "_verified", lambda loader, product, label: saved)
    monkeypatch.setattr(story_view_inputs, "RidgeEntropy", Entropy)
    monkeypatch.setattr(story_view_inputs, "_parent", lambda *a: SimpleNamespace(artifacts={"ridges.parquet": "x"}))
    monkeypatch.setattr(
        pl, "read_parquet",
        lambda path: pl.DataFrame({"ridge_name": ["KEM"], "functional_count": [3]}),
    )
    payload = story_view_inputs.load_ridge_entropy_inputs(SimpleNamespace())
    assert payload["lab"]["product_id"] == "sweep-id"
    assert payload["lab"]["functional"] == {"KEM": 3}
