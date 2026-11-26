"""The shared loader builds the seven states from verified saved products or halts."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis.graph_summary import SUMMARY_COLUMNS
from dubanevics2026topological.build.landscape_summary_inputs import (
    load_landscape_summary_inputs,
)
from dubanevics2026topological.results.processed import LEAVES

STATES = tuple(LEAVES)
TABLE = "fdr_classification_T=-0.25_cutoff=x_method=by_q=0.05.parquet"


def _product(kind, label, root, *, parents=(), artifacts=None, **manifest):
    files = {}
    for name, content in (artifacts or {}).items():
        path = root / f"{label}-{name}"
        if name.endswith(".json"):
            path.write_text(json.dumps(content))
        else:
            content.write_parquet(path)
        files[name] = path
    ref = lambda p: {
        "product_type": p.kind,
        "label": p.label,
        "scientific_id_sha256": p.manifest.scientific_id_sha256,
    }
    return SimpleNamespace(
        kind=kind,
        label=label,
        report=SimpleNamespace(status="full", byte_integrity=True),
        artifacts=files,
        manifest=SimpleNamespace(
            scientific_id_sha256=hashlib.sha256(label.encode()).hexdigest(),
            parents=[ref(p) for p in parents],
            scientific_identity=manifest.get("identity", {}),
            data_provenance=manifest.get("provenance", {}),
        ),
    )


def _store(tmp_path):
    processed_root = tmp_path / "processed"
    calls = pl.DataFrame({"dna_sequence": ["a", "b"], "fdr_class": ["functional"] * 2})
    for variant in ("published", "reflected"):
        directory = processed_root / "classification" / variant
        directory.mkdir(parents=True)
        calls.write_parquet(directory / TABLE)
    digest = hashlib.sha256((directory / TABLE).read_bytes()).hexdigest()
    store = {}

    def add(product):
        store[(product.kind, product.label)] = product
        return product

    vertices = pl.DataFrame({"dna_sequence": ["a", "b"], "raw_fitness": [0.5, -0.5]})
    fitness = add(_product("fitness", "fit", tmp_path))
    for state in ("full", "published", "reflected", "published_by", "reflected_by"):
        by = state.endswith("_by")
        landscape = add(
            _product(
                "landscape",
                f"land-{state}",
                tmp_path,
                parents=(fitness,) if by else (),
                artifacts={
                    "diagnostics.json": {
                        "threshold": 0.0,
                        "component_count": 1,
                        "incident_vertex_count": 2,
                    },
                    "vertices.parquet": vertices,
                    "edges.parquet": pl.DataFrame({"a": [0]}),
                },
                identity={"classification_source": TABLE} if by else {},
                provenance={"classification_table": digest} if by else {},
            )
        )
        nominal = add(_product("tree", f"tree-{state}", tmp_path, parents=(landscape,)))
        trees = {state: nominal}
        if by:
            trees[f"{state}_band"] = add(
                _product("tree", f"tree-{state}-band", tmp_path, parents=(nominal,))
            )
        for key, tree in trees.items():
            add(
                _product(
                    "leaves",
                    LEAVES[key],
                    tmp_path,
                    parents=(tree,),
                    artifacts={
                        "leaves.parquet": pl.DataFrame(
                            {"is_root_endpoint": [True], "is_synonymous": [None]},
                            schema_overrides={"is_synonymous": pl.Boolean},
                        ),
                        "members.parquet": pl.DataFrame({"member_dna_sequence": ["a"]}),
                        "summary.json": {"tree_state": "nominal"},
                    },
                )
            )
    return store, SimpleNamespace(processed_root=processed_root), digest


def _load(tmp_path, store, context):
    return load_landscape_summary_inputs(
        context, loader=lambda kind, label: store[(kind, label)]
    )


def test_builds_seven_states_from_saved_products(tmp_path):
    store, context, digest = _store(tmp_path)
    loaded = _load(tmp_path, store, context)
    assert tuple(loaded.inputs) == SUMMARY_COLUMNS
    assert loaded.inputs[SUMMARY_COLUMNS[0]].threshold is None
    assert loaded.inputs[SUMMARY_COLUMNS[1]].classes is None
    assert loaded.inputs[SUMMARY_COLUMNS[4]].classes == {
        "a": "functional",
        "b": "functional",
    }
    assert loaded.landscape_labels[SUMMARY_COLUMNS[4]] == "land-published_by"
    assert loaded.classification_sha256[SUMMARY_COLUMNS[3]] == digest
    assert loaded.full_fitness == {"a": 0.5, "b": -0.5}


def test_missing_product_halts(tmp_path):
    store, context, _ = _store(tmp_path)
    del store[("tree", "tree-published")]
    with pytest.raises(KeyError):
        _load(tmp_path, store, context)


def test_unverified_product_halts(tmp_path):
    store, context, _ = _store(tmp_path)
    store[("leaves", LEAVES["published"])].report.byte_integrity = False
    with pytest.raises(ValueError, match="Unverified"):
        _load(tmp_path, store, context)


def test_parent_identity_mismatch_halts(tmp_path):
    store, context, _ = _store(tmp_path)
    store[("leaves", LEAVES["full"])].manifest.parents[0]["scientific_id_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="does not match"):
        _load(tmp_path, store, context)


def test_band_tree_on_different_landscape_halts(tmp_path):
    store, context, _ = _store(tmp_path)
    other = store[("landscape", "land-reflected_by")]
    nominal = store[("tree", "tree-published_by")]
    nominal.manifest.parents[0].update(
        label=other.label, scientific_id_sha256=other.manifest.scientific_id_sha256
    )
    # nominal and band now agree with each other, so break the band route instead
    band = store[("tree", "tree-published_by-band")]
    band.manifest.parents.append(
        {
            "product_type": "landscape",
            "label": "land-published_by",
            "scientific_id_sha256": store[
                ("landscape", "land-published_by")
            ].manifest.scientific_id_sha256,
        }
    )
    with pytest.raises(ValueError, match="different landscape"):
        _load(tmp_path, store, context)


def test_snapshot_sha_mismatch_halts(tmp_path):
    store, context, _ = _store(tmp_path)
    store[("landscape", "land-published_by")].manifest.data_provenance[
        "classification_table"
    ] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        _load(tmp_path, store, context)


def test_landscape_without_a_pinned_table_halts(tmp_path):
    store, context, _ = _store(tmp_path)
    del store[("landscape", "land-published_by")].manifest.data_provenance[
        "classification_table"
    ]
    with pytest.raises(ValueError, match="does not pin"):
        _load(tmp_path, store, context)


def test_missing_processed_classification_halts(tmp_path):
    import shutil

    store, context, _ = _store(tmp_path)
    shutil.rmtree(context.processed_root / "classification" / "reflected")
    with pytest.raises(Exception, match="just all"):
        _load(tmp_path, store, context)
