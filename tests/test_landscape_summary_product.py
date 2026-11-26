"""One landscape_summary product per state, keyed on that state's tree keys."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubanevics2026topological.analysis.describe import SUMMARY_COLUMNS
from dubanevics2026topological.build import (
    CONTEXT_AWARE_COMMANDS,
)
from dubanevics2026topological.build import (
    landscape_summary_product as product,
)
from dubanevics2026topological.build.landscape_summary_inputs import (
    LoadedLandscapeSummary,
)
from dubanevics2026topological.results import RunContext, parse_label

REPO_ROOT = Path(__file__).resolve().parents[1]
BAND = {"draws": 10000, "seed": 20261019, "tree": "band", "bandq": 0.95}
IDENTITIES = (
    {"graph": "full", "tree": "nominal"},
    {"thr": "published", "fdr": "none", "tree": "nominal"},
    {"thr": "reflected", "fdr": "none", "tree": "nominal"},
    {"thr": "published", "fdr": "by", "q": 0.05, "tree": "nominal"},
    {"thr": "published", "fdr": "by", "q": 0.05, **BAND},
    {"thr": "reflected", "fdr": "by", "q": 0.05, "tree": "nominal"},
    {"thr": "reflected", "fdr": "by", "q": 0.05, **BAND},
)
SNAPSHOT = "f" * 64


def _fake(kind: str, label: str, identity: dict | None = None):
    return SimpleNamespace(
        product_type=kind,
        label=label,
        manifest=SimpleNamespace(
            scientific_id_sha256=hashlib.sha256(f"{kind}:{label}".encode()).hexdigest(),
            scientific_identity=identity or {},
        ),
    )


def _loaded() -> LoadedLandscapeSummary:
    names = SUMMARY_COLUMNS
    return LoadedLandscapeSummary(
        inputs={name: object() for name in names},
        full_fitness={},
        diagnostics={},
        landscape_labels={},
        classification_sha256={name: SNAPSHOT for name in names[3:]},
        leaves={name: _fake("leaves", f"leaves-{i}") for i, name in enumerate(names)},
        trees={
            name: _fake("tree", f"tree-{i}", IDENTITIES[i])
            for i, name in enumerate(names)
        },
        landscapes={
            name: _fake("landscape", f"landscape-{i}") for i, name in enumerate(names)
        },
    )


@pytest.fixture
def stubbed(monkeypatch):
    columns = {
        name: {"Number of nodes": index, "Functional threshold": None}
        for index, name in enumerate(SUMMARY_COLUMNS)
    }
    monkeypatch.setattr(
        product, "load_landscape_summary_inputs", lambda *a, **k: _loaded()
    )
    monkeypatch.setattr(
        product, "describe", lambda **k: SimpleNamespace(landscape_summary=columns)
    )
    return columns


def test_command_is_context_aware():
    assert "landscape-summary-product" in CONTEXT_AWARE_COMMANDS


def test_publishes_seven_products_with_tree_keys_and_saved_values(tmp_path, stubbed):
    context = RunContext(root=tmp_path, output_root=tmp_path / "out", env={})
    paths = product.build_landscape_summary_products(context)
    assert tuple(paths) == SUMMARY_COLUMNS
    assert len({path.name for path in paths.values()}) == 7
    for name, identity in zip(SUMMARY_COLUMNS, IDENTITIES, strict=True):
        path = paths[name]
        parsed = parse_label(path.name)
        assert parsed.product_name == "landscape_summary"
        for key, value in identity.items():
            assert parsed.scientific_params[key] == value
        rows = json.loads((path / product.SUMMARY_FILE).read_text())
        assert [row["metric"] for row in rows] == list(stubbed[name])
        assert {row["metric"]: row["value"] for row in rows} == stubbed[name]
        manifest = json.loads((path / "manifest.json").read_text())
        kinds = [ref["product_type"] for ref in manifest["parents"]]
        assert kinds[:3] == ["leaves", "tree", "landscape"]
        assert (len(kinds) == 3) == (name == SUMMARY_COLUMNS[0])


def test_by_states_pin_the_snapshot_hash(tmp_path, stubbed):
    context = RunContext(root=tmp_path, output_root=tmp_path / "out", env={})
    paths = product.build_landscape_summary_products(context)
    for name in SUMMARY_COLUMNS:
        manifest = json.loads((paths[name] / "manifest.json").read_text())
        pinned = manifest["data_provenance"].get("classification_table")
        assert pinned == (SNAPSHOT if name in SUMMARY_COLUMNS[3:] else None)


def test_rebuild_is_idempotent(tmp_path, stubbed):
    context = RunContext(root=tmp_path, output_root=tmp_path / "out", env={})
    first = product.build_landscape_summary_products(context)
    assert product.build_landscape_summary_products(context) == first


def test_unverified_input_halts_before_publishing(tmp_path):
    from test_landscape_summary_inputs import LEAVES, _store

    store, _, _ = _store(tmp_path)
    store[("leaves", LEAVES["published"])].report.byte_integrity = False
    context = RunContext(root=tmp_path, output_root=tmp_path / "out", env={})
    with pytest.raises(ValueError, match="Unverified"):
        product.build_landscape_summary_products(
            context, loader=lambda kind, label: store[(kind, label)]
        )
    assert not (tmp_path / "out").exists()


def test_justfile_recipes_exist():
    text = (REPO_ROOT / "justfile").read_text(encoding="utf-8")
    for recipe in (
        "landscape-summary-product",
        "landscape-summary-table",
        "presentation-tables",
    ):
        assert re.search(rf"^{recipe}:", text, re.MULTILINE)
