"""Tests for the bootstrap codon entropy product (PR 3b).

These tests follow the PR 3a template in test_pooled_scatter_product.py.
They use real checkpoint writers where possible so that schema and include_full
expectations are satisfied.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dubanevics2026topological.build.bootstrap_codon_entropy_product import (
    _file_sha256,
    build_bootstrap_codon_entropy_product,
)
from dubanevics2026topological.results import RunContext, load


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "t.bin"
    p.write_bytes(b"hello world")
    assert _file_sha256(p) == _sha256_bytes(b"hello world")
    # Mutating the file changes the hash.
    p.write_bytes(b"goodbye")
    assert _file_sha256(p) != _sha256_bytes(b"hello world")


def test_prepare_plot_tables_uses_enriched_tree_plot_data(monkeypatch):
    from test_bootstrap_smoothed_tree import _toy

    from dubanevics2026topological.analysis import codon_entropy_render_tables as render

    _, tree = _toy()
    monkeypatch.setattr(render, "selected_view_as_tree", lambda view: tree)
    tables = render.prepare_plot_tables(object())

    assert tables["node_table"].equals(tree.plot_data.node_table)
    for name in ("edge_table", "edge_segment_table"):
        assert tables[name].height > 0
        assert tables[name]["vertex_count"].to_list() == tables[name][
            "branch_vertex_count"
        ].to_list()


def test_retained_bootstrap_requires_matching_selection(monkeypatch):
    from dubanevics2026topological.build import bootstrap_codon_entropy_product as product

    scientific = {
        "config": {"n_replicates": 10000, "seed": 20261019},
        "candidate_selection": {
            "rule": "mixture_quantile",
            "functional_set_sha256": "a" * 64,
        },
    }
    identity = dict(scientific["candidate_selection"])
    monkeypatch.setattr(product, "selection_identity", lambda selection: identity)
    product._require_bootstrap_selection(scientific, object())
    changed = {**scientific, "candidate_selection": {**identity, "functional_set_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="Bootstrap checkpoint selection"):
        product._require_bootstrap_selection(changed, object())
    missing = {key: value for key, value in scientific.items() if key != "candidate_selection"}
    with pytest.raises(ValueError, match="Bootstrap checkpoint selection"):
        product._require_bootstrap_selection(missing, object())


def test_build_reads_and_hashes_bootstrap_under_one_lock(tmp_path, monkeypatch):
    from dubanevics2026topological.analysis import clamped_cache
    from dubanevics2026topological.analysis.checkpoint import checkpoint_lock
    from dubanevics2026topological.build import bootstrap_codon_entropy_product as product
    from dubanevics2026topological.simplification import bootstrap_smoothed_tree as bootstrap

    published = tmp_path / "published"
    slot = tmp_path / "slot"
    settings = {"analysis": {"codon_entropy": {}}}
    monkeypatch.setattr(product, "load_settings", lambda **kwargs: settings)
    monkeypatch.setattr(product, "resolve_candidate_selection", lambda *a, **k: None)
    monkeypatch.setattr(clamped_cache, "cache_dirs", lambda context: (tmp_path / "comparison",))
    monkeypatch.setattr(clamped_cache, "_load_clamped_checkpoint", lambda *a, **k: {"provenance": {"fingerprint": "current"}})
    monkeypatch.setattr(clamped_cache, "_fingerprint", lambda *a, **k: "current")
    monkeypatch.setattr(product, "_comparison_inputs", lambda context: {})
    monkeypatch.setattr(product, "bootstrap_checkpoint_dir", lambda context: published)
    monkeypatch.setattr(product, "bootstrap_slot_dir", lambda context: slot)
    monkeypatch.setattr(bootstrap, "checkpoint_dir", lambda context: published)
    monkeypatch.setattr(bootstrap, "cache_dir", lambda context: slot)
    monkeypatch.setattr(bootstrap, "_configured_settings", lambda *a: "wanted")
    monkeypatch.setattr(bootstrap, "publication_state", lambda path: {"pending": False, "backup": False})
    monkeypatch.setattr(bootstrap, "_local_settings", lambda path: "wanted")
    monkeypatch.setattr(bootstrap, "_wanted_scientific", lambda *a, **k: {})
    monkeypatch.setattr(bootstrap.checkpoint_identity, "scientific_of", lambda path: {})
    monkeypatch.setattr(bootstrap.checkpoint_identity, "differences", lambda *a: [])
    monkeypatch.setattr(bootstrap, "_require_current", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_require_exported", lambda context: None)

    def assert_locked():
        with pytest.raises(RuntimeError):
            with checkpoint_lock(published, "contender"):
                pass
        with pytest.raises(RuntimeError):
            with checkpoint_lock(tmp_path / "comparison", "contender"):
                pass

    def read(path):
        assert_locked()
        return object()

    class HashReached(Exception):
        pass

    def hash_inputs(context):
        assert_locked()
        raise HashReached

    monkeypatch.setattr(bootstrap, "_read_checkpoint", read)
    monkeypatch.setattr(product.checkpoint_identity, "scientific_of", lambda path: {"candidate_selection": {}})
    monkeypatch.setattr(product, "selection_identity", lambda selection: {})
    monkeypatch.setattr(product, "_selection_product_identity", lambda selection: {})
    monkeypatch.setattr(product, "_bootstrap_inputs", hash_inputs)
    with pytest.raises(HashReached):
        product.build_bootstrap_codon_entropy_product(context=object())
    with checkpoint_lock(published, "after build"):
        pass


def test_product_publish_load_and_selection_staleness(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import polars as pl

    from dubanevics2026topological import analysis
    from dubanevics2026topological.analysis import clamped_cache
    from dubanevics2026topological.build import bootstrap_codon_entropy_product as product
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_bootstrap_codon_entropy_product,
    )
    from dubanevics2026topological.landscape import selection as selections
    from dubanevics2026topological.results import MissingProductError

    context = RunContext(output_root=tmp_path / "output", root=tmp_path, env={})
    settings = {"analysis": {"codon_entropy": {"sweep_points": 3}}}
    selection = SimpleNamespace(
        rule="median_reflected",
        threshold=SimpleNamespace(threshold=0.4),
        functional_sequences=None,
        source={"rule": "median_reflected"},
    )
    monkeypatch.setattr(product, "load_settings", lambda **kwargs: settings)
    monkeypatch.setattr(analysis, "load_settings", lambda **kwargs: settings)
    monkeypatch.setattr(product, "resolve_candidate_selection", lambda *a, **k: selection)
    monkeypatch.setattr(selections, "resolve_candidate_selection", lambda *a, **k: selection)
    monkeypatch.setattr(clamped_cache, "cache_dirs", lambda context: (tmp_path / "comparison",))
    monkeypatch.setattr(clamped_cache, "_load_clamped_checkpoint", lambda *a, **k: {"provenance": {"fingerprint": "current"}})
    monkeypatch.setattr(clamped_cache, "_fingerprint", lambda *a, **k: "current")
    monkeypatch.setattr(product, "bootstrap_checkpoint_dir", lambda context: tmp_path / "bootstrap")
    monkeypatch.setattr(product, "load_cached_bootstrap_smoothed_tree", lambda **kwargs: SimpleNamespace(hills=None))
    monkeypatch.setattr(product.checkpoint_identity, "scientific_of", lambda path: {"candidate_selection": {"rule": "median_reflected"}})
    monkeypatch.setattr(product, "_comparison_inputs", lambda context: {"data": "a" * 64})
    monkeypatch.setattr(product, "_bootstrap_inputs", lambda context: {"data": "b" * 64})
    def build_tree(*args, **kwargs):
        assert kwargs["selection"] is selection
        return None, None

    monkeypatch.setattr(product, "build_candidate_tree", build_tree)
    monkeypatch.setattr(product, "build_band_simplified_view", lambda *a: object())
    table = pl.DataFrame({"value": [1]})
    original_extractor = product.extract_codon_entropy_populations

    def extract(**kwargs):
        assert kwargs["threshold"] == selection.threshold.threshold
        assert kwargs["functional_sequences"] is selection.functional_sequences
        return {"df_core": table, "df_kem": table, "df_tcl": table, "metadata": {}}

    monkeypatch.setattr(product, "extract_codon_entropy_populations", extract)
    monkeypatch.setattr(product, "compute_entropy_sweeps", lambda *a, **k: {
        name: table for name in ("core", "kem", "tcl")
    })
    monkeypatch.setattr(product, "build_render_tables", lambda *a: {
        **{name: table for name in ("node_table", "edge_table", "edge_segment_table", "ridge_edges", "stop_points", "wild_type_marker")},
        "count_domain": (1.0, 2.0),
        "plot_data_attrs": {"root_node_index": 0, "scalar_attr": "raw_fitness", "root_kind": "minimum"},
    })

    product.build_bootstrap_codon_entropy_product(context)
    loaded = load_bootstrap_codon_entropy_product(context)
    assert loaded["wild_type_marker"].equals(table)
    assert loaded["product"].manifest.scientific_identity["functional_selection"]["rounded_threshold"] == 0.4

    settings["analysis"]["codon_entropy"]["sweep_points"] = 4
    with pytest.raises(MissingProductError):
        load_bootstrap_codon_entropy_product(context)
    settings["analysis"]["codon_entropy"]["sweep_points"] = 3
    selection.threshold.threshold = 0.5
    with pytest.raises(MissingProductError):
        load_bootstrap_codon_entropy_product(context)

    from dubanevics2026topological.analysis import codon_entropy_populations

    monkeypatch.setattr(product.checkpoint_identity, "scientific_of", lambda path: {"candidate_selection": {"rule": "other"}})
    with pytest.raises(ValueError, match="Bootstrap checkpoint selection"):
        product.build_bootstrap_codon_entropy_product(context)
    monkeypatch.setattr(product.checkpoint_identity, "scientific_of", lambda path: {"candidate_selection": {"rule": "median_reflected"}})
    monkeypatch.setattr(clamped_cache, "_fingerprint", lambda *a, **k: "stale")
    with pytest.raises(ValueError, match="Comparison checkpoint is stale"):
        product.build_bootstrap_codon_entropy_product(context)
    monkeypatch.setattr(clamped_cache, "_fingerprint", lambda *a, **k: "current")

    monkeypatch.setattr(product, "extract_codon_entropy_populations", original_extractor)
    monkeypatch.setattr(codon_entropy_populations, "_read_figure_config", lambda config: {})
    monkeypatch.setattr(clamped_cache, "_load_clamped_checkpoint", lambda *a, **k: {
        "thresholds": {"candidate_clamped": 0.4}, "provenance": {"fingerprint": "current"}
    })
    with pytest.raises(ValueError, match="threshold must match"):
        product.build_bootstrap_codon_entropy_product(context)


@pytest.mark.skip(
    reason="requires full comparison + bootstrap fixtures; run on the cluster or with golden data"
)
def test_product_round_trip_and_provenance() -> None:
    ctx = RunContext()
    path = build_bootstrap_codon_entropy_product(context=ctx)
    prod = load("bootstrap_codon_entropy", context=ctx)
    assert prod.report.status == "full"
    assert prod.report.byte_integrity
    # Key artifacts exist.
    for name in (
        "populations_core.parquet",
        "populations_kem.parquet",
        "populations_tcl.parquet",
        "entropy_sweeps.parquet",
        "node_table.parquet",
        "edge_table.parquet",
        "ridge_edges.parquet",
        "stop_points.parquet",
        "metadata.json",
    ):
        assert name in prod.artifacts or any(k.endswith(name) for k in prod.artifacts)
    # Provenance contains raw file hashes for comparison and bootstrap inputs.
    prov = prod.manifest.data_provenance or {}
    assert any(k.startswith("tables/") or k.endswith(".parquet") for k in prov)
    # Rebuilding with the same inputs yields the same scientific id (or at least
    # the input hashes are identical).
    path2 = build_bootstrap_codon_entropy_product(context=ctx)
    assert (
        path == path2
        or prod.manifest.scientific_id_sha256
        == load("bootstrap_codon_entropy", context=ctx).manifest.scientific_id_sha256
    )


def test_tamper_on_comparison_file_is_detected(tmp_path: Path, monkeypatch) -> None:
    # This is a structural test: if a comparison file's bytes change after the
    # lock window, the recorded hash will not match a fresh computation.
    # The real enforcement happens inside the product under lock.
    p = tmp_path / "fake.parquet"
    p.write_bytes(b"0" * 100)
    h1 = _file_sha256(p)
    p.write_bytes(b"1" * 100)
    h2 = _file_sha256(p)
    assert h1 != h2


def test_load_bootstrap_codon_entropy_product_raises_on_tampered_hash(monkeypatch):
    """A product whose recorded data_provenance does not match the freshly
    computed input hashes causes MissingProductError (fail-closed).
    """
    from types import SimpleNamespace

    from dubanevics2026topological.build import saved_figure_inputs as sfi
    from dubanevics2026topological import analysis
    from dubanevics2026topological.landscape import selection
    from dubanevics2026topological.results import MissingProductError

    ctx = RunContext()
    monkeypatch.setattr(analysis, "load_settings", lambda **kwargs: {})
    monkeypatch.setattr(selection, "resolve_candidate_selection", lambda *a, **k: None)
    monkeypatch.setattr(
        "dubanevics2026topological.build.bootstrap_codon_entropy_product._product_identity",
        lambda *a: {},
    )

    fresh_hashes = {"comparison/f": "aaa", "bootstrap/g": "bbb"}

    def fake_hashes(_ctx):
        return dict(fresh_hashes)

    # Patch on the source module; the loader does a local import at call time.
    monkeypatch.setattr(
        "dubanevics2026topological.build.bootstrap_codon_entropy_product._codon_entropy_input_hashes",
        fake_hashes,
    )

    manifest = SimpleNamespace(
        data_provenance={"comparison/f": "tampered", "bootstrap/g": "bbb"}
    )
    fake_product = SimpleNamespace(
        report=SimpleNamespace(status="full", byte_integrity=True),
        manifest=manifest,
        artifacts={},
    )

    def fake_load(name, context=None, expected=None):
        # If the loader passes expected with data_provenance, simulate the
        # results layer rejecting a mismatch.
        if expected and expected.get("data_provenance") != manifest.data_provenance:
            raise MissingProductError("no match")
        return fake_product

    monkeypatch.setattr(sfi, "load", fake_load)

    with pytest.raises(MissingProductError, match="stale|Missing or stale"):
        sfi.load_bootstrap_codon_entropy_product(ctx)
