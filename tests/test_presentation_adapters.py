"""Presentation adapters preserve lookup and preview-writing contracts."""

from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.build.presentation_io import figure_path, save_preview
from dubanevics2026topological.results import PresentationProducts, RunContext


def test_product_adapter_forwards_identity_and_context(monkeypatch, tmp_path):
    context = RunContext(root=tmp_path, env={})
    sentinel = object()
    calls = []

    def fake_load(product, label, *, context, expected):
        calls.append((product, label, context, expected))
        return sentinel

    monkeypatch.setattr(
        "dubanevics2026topological.results.presentation.load", fake_load
    )
    expected = {"tree": "nominal"}
    assert (
        PresentationProducts(context).load("tree", "label", expected=expected)
        is sentinel
    )
    assert calls == [("tree", "label", context, expected)]


def test_preview_writer_preserves_chart_options(tmp_path):
    context = RunContext(root=tmp_path, env={})
    calls = []

    class Chart:
        def save(self, path, **options):
            calls.append((path, options))
            path.write_bytes(b"preview")

    path = save_preview(context, Chart(), "example", "png", ppi=96)
    assert path == tmp_path / "output/previews/example.png"
    assert path.parent.is_dir()
    assert len(calls) == 1
    assert calls[0][0].parent == path.parent
    assert calls[0][0].suffix == ".png"
    assert calls[0][1] == {"ppi": 96}
    assert path.read_bytes() == b"preview"
    assert list(path.parent.iterdir()) == [path]
    assert (
        figure_path(context, "example", "svg")
        == tmp_path / "output/figures/example.svg"
    )


def test_failed_preview_save_preserves_existing_file(tmp_path):
    context = RunContext(root=tmp_path, env={})
    destination = tmp_path / "output/previews/example.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing preview")

    class BrokenChart:
        def save(self, path, **options):
            path.write_bytes(b"partial")
            raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        save_preview(context, BrokenChart(), "example", "png")

    assert destination.read_bytes() == b"existing preview"
    assert list(destination.parent.iterdir()) == [destination]


def test_landscape_population_reads_verified_saved_vertices(tmp_path):
    path = tmp_path / "vertices.parquet"
    vertices = pl.DataFrame({"dna_sequence": ["B", "A"], "raw_fitness": [0.3, 0.8]})
    vertices.write_parquet(path)
    landscape = SimpleNamespace(
        product_type="landscape",
        report=SimpleNamespace(status="full", byte_integrity=True),
        artifacts={"vertices.parquet": path},
    )
    products = PresentationProducts(RunContext(root=tmp_path, env={}))
    assert products.landscape_population(landscape).equals(
        vertices.rename({"raw_fitness": "scalar"})
    )
    vertices.with_columns(pl.lit("A").alias("dna_sequence")).write_parquet(path)
    with pytest.raises(ValueError, match="invalid identities"):
        products.landscape_population(landscape)
