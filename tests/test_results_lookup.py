from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from dubanevics2026topological.results.context import RunContext
from dubanevics2026topological.results.hashing import compute_scientific_hash
from dubanevics2026topological.results.labels import format_label
from dubanevics2026topological.results.lookup import (
    AmbiguousParentError,
    MissingProductError,
    load,
)
from dubanevics2026topological.results.manifest import Manifest, write_manifest


def _context(root: Path) -> RunContext:
    return RunContext(
        root=root,
        output_root=root / "output",
        raw_data_dir=root / "data" / "raw",
        env={},
    )


def _write_product(
    context: RunContext,
    *,
    directory_name: str = "sample",
    run_id: int | None = None,
    digest_override: str | None = None,
    check_collision: bool = True,
) -> str:
    identity = {"thr": "reflected", "fdr": "by", "q": 0.05, "draws": 10, "seed": 7}
    digest, short = compute_scientific_hash("bootstrap", identity)
    label_params = {**identity, "hash": short}
    directory = context.output_root / "bootstrap" / directory_name
    directory.mkdir(parents=True)
    artifact = directory / "summary.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = Manifest(
        product_type="bootstrap",
        scientific_identity=identity,
        scientific_id_sha256=digest_override or digest,
        parents=[],
        data_provenance={},
        artifacts={
            "summary.json": {
                "sha256": hashlib.sha256(b"{}").hexdigest(),
                "size_bytes": 2,
            }
        },
        run_fields={"host": "local", "run": run_id} if run_id is not None else {},
        code_provenance={},
        environment={},
        created_at="2026-01-01T00:00:00Z",
        evidence_availability={
            "checkpoint_present": True,
            "distances_present": True,
            "chunks_present": False,
        },
    )
    write_manifest(manifest, directory, check_collision=check_collision)
    if run_id is not None:
        label_params.update({"host": "local", "run": run_id})
    return format_label("bootstrap", label_params)


def test_load_reports_full_partial_and_mismatch(tmp_path: Path) -> None:
    context = _context(tmp_path)
    label = _write_product(context)
    identity = {"thr": "reflected", "fdr": "by", "q": 0.05, "draws": 10, "seed": 7}

    full = load("bootstrap", label, context=context, expected=identity)
    assert full.report.status == "full"
    assert full.report.byte_integrity
    assert full.report.checked[0]["expected"] == full.report.checked[0]["observed"]

    partial = load(
        "bootstrap", label, context=context, expected={**identity, "parent": "unknown"}
    )
    assert partial.report.status == "partial"
    assert "parent" in partial.report.unavailable

    mismatch = load(
        "bootstrap", label, context=context, expected={**identity, "thr": "published"}
    )
    assert mismatch.report.status == "mismatch"
    assert mismatch.report.differing["thr"] == {
        "expected": "published",
        "observed": "reflected",
    }

    structured = load(
        "bootstrap",
        label,
        context=context,
        expected={
            "scientific_identity": identity,
            "data_provenance": {},
            "parents": [],
            "scientific_id_sha256": full.manifest.scientific_id_sha256,
        },
    )
    assert structured.report.status == "full"
    structured_mismatch = load(
        "bootstrap",
        label,
        context=context,
        expected={
            "scientific_identity": identity,
            "data_provenance": {"input.parquet": "bad"},
            "parents": [],
            "scientific_id_sha256": full.manifest.scientific_id_sha256,
        },
    )
    assert structured_mismatch.report.status == "mismatch"
    assert "data_provenance" in structured_mismatch.report.differing

    canonical_partial = load(
        "bootstrap", label, context=context, expected={"core_threshold": 0.5}
    )
    assert canonical_partial.report.status == "partial"


def test_no_label_lookup_filters_wrong_manifest_product_type(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_product(context)
    manifest_path = context.output_root / "bootstrap" / "sample" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["product_type"] = "fitness"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MissingProductError, match="No 'bootstrap' product found"):
        load("bootstrap", context=context)


def test_short_hash_collision_is_explicit(tmp_path: Path) -> None:
    context = _context(tmp_path)
    identity = {"thr": "reflected", "fdr": "by", "q": 0.05, "draws": 10, "seed": 7}
    common_prefix = "cafebabe"
    _write_product(
        context,
        directory_name="collision-one",
        digest_override=common_prefix + "1" * 56,
        check_collision=False,
    )
    _write_product(
        context,
        directory_name="collision-two",
        digest_override=common_prefix + "2" * 56,
        check_collision=False,
    )
    collision_label = format_label("bootstrap", {**identity, "hash": common_prefix})
    from dubanevics2026topological.results import ScientificHashCollisionError

    with pytest.raises(ScientificHashCollisionError, match="short-hash collision"):
        load("bootstrap", collision_label, context=context)


def test_unpinned_load_with_data_provenance_filters_when_sci_identity_shared(
    tmp_path: Path,
) -> None:
    """Two products with identical sci identity but different data_provenance are
    disambiguated by passing data_provenance in expected (unpinned path now filters
    on observed fields as _apply_expected builds them).
    """
    context = _context(tmp_path)
    identity = {"thr": "reflected", "fdr": "by", "q": 0.05}
    digest, _ = compute_scientific_hash("bootstrap", identity)
    for dirname, dp in [("prod-a", {"input": "a"}), ("prod-b", {"input": "b"})]:
        d = context.output_root / "bootstrap" / dirname
        d.mkdir(parents=True)
        (d / "summary.json").write_text("{}", encoding="utf-8")
        manifest = Manifest(
            product_type="bootstrap",
            scientific_identity=identity,
            scientific_id_sha256=digest,
            parents=[],
            data_provenance=dp,
            artifacts={
                "summary.json": {
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                    "size_bytes": 2,
                }
            },
            run_fields={},
            code_provenance={},
            environment={},
            created_at="2026-01-01T00:00:00Z",
            evidence_availability={
                "checkpoint_present": False,
                "distances_present": False,
                "chunks_present": False,
            },
        )
        write_manifest(manifest, d, check_collision=False)
    # load with dp of a selects prod-a
    p = load("bootstrap", context=context, expected={"data_provenance": {"input": "a"}})
    assert p.path.name == "prod-a"
    # similarly for b
    p = load("bootstrap", context=context, expected={"data_provenance": {"input": "b"}})
    assert p.path.name == "prod-b"


def test_deduplication_respects_parent_identity_not_dict_order(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_product(context, directory_name="prod-a")
    _write_product(context, directory_name="prod-b")
    paths = [context.output_root / "bootstrap" / name / "manifest.json"
             for name in ("prod-a", "prod-b")]
    parent = {"product_type": "tree", "label": "tree-a", "scientific_id_sha256": "a" * 64}
    for path, ref in zip(paths, (parent, dict(reversed(list(parent.items())))), strict=True):
        manifest = json.loads(path.read_text())
        manifest["parents"] = [ref]
        path.write_text(json.dumps(manifest))
    assert load("bootstrap", context=context).path.name in {"prod-a", "prod-b"}

    manifest = json.loads(paths[1].read_text())
    manifest["parents"][0]["label"] = "tree-b"
    paths[1].write_text(json.dumps(manifest))
    with pytest.raises(AmbiguousParentError):
        load("bootstrap", context=context)


def test_a_product_published_to_processed_is_found_and_deduplicated(tmp_path: Path) -> None:
    context = _context(tmp_path)
    label = _write_product(context, directory_name="staged")
    shutil.copytree(
        context.output_root / "bootstrap" / "staged",
        context.processed_root / "bootstrap" / "published_by",
    )
    found = load("bootstrap", label, context=context)
    assert found.report.status == "full" and found.report.byte_integrity
    only = load("bootstrap", context=context)
    assert only.manifest.scientific_id_sha256 == found.manifest.scientific_id_sha256


def test_a_missing_product_names_the_recipes(tmp_path: Path) -> None:
    with pytest.raises(MissingProductError, match="just all.*just fetch-data"):
        load("bootstrap", context=_context(tmp_path))
