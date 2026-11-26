"""Tests for manifest read/write, schema validation, and collision detection per ADR 0016."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dubanevics2026topological.results.manifest import (
    CURRENT_SCHEMA_VERSION,
    Manifest,
    ScientificHashCollisionError,
    check_hash_collision,
    read_manifest,
    write_manifest,
)


def _sample_manifest(
    full_sha: str = "7a21c3f0ffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    short_hash_override: str | None = None,
) -> Manifest:
    """Helper to produce a valid Manifest instance."""
    sha = full_sha
    if short_hash_override:
        sha = short_hash_override + full_sha[8:]
    return Manifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        product_type="bootstrap",
        scientific_identity={
            "draws": 10000,
            "fdr": "by",
            "q": 0.05,
            "seed": 20261019,
            "thr": "reflected",
        },
        scientific_id_sha256=sha,
        parents=[
            {
                "product_type": "landscape",
                "scientific_id_sha256": "1111111122222222333333334444444455555555666666667777777788888888",
            }
        ],
        data_provenance={
            "papkou2023_fitness_diff.parquet": "44fcd723d360a73df9187fc66e66d2d28d0b9b5985984d0be8f0d56e3c8ffd5d"
        },
        artifacts={
            "distances.parquet": {
                "bytes": 1048576,
                "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            }
        },
        run_fields={"host": "local", "run": 51756602},
        code_provenance={"branch": "main", "commit": "1f1b4a9", "dirty": False},
        environment={
            "platform": "darwin",
            "python_version": "3.13.0",
            "uv_lock_sha256": "14973f71",
        },
        created_at="2026-10-27T14:00:00Z",
        evidence_availability={
            "checkpoint_present": True,
            "chunks_present": False,
            "distances_present": True,
        },
        storage_references={"bucket_uri": "gs://store/results/51756602"},
    )


def test_manifest_round_trip_dict_and_json() -> None:
    """Verify Manifest round-trips through dictionary and JSON serialization."""
    manifest = _sample_manifest()
    m_dict = manifest.to_dict()
    assert m_dict["schema_version"] == 1
    assert m_dict["product_type"] == "bootstrap"
    assert manifest.short_hash == "7a21c3f0"

    rebuilt = Manifest.from_dict(m_dict)
    assert rebuilt == manifest

    json_str = manifest.to_json()
    from_json = Manifest.from_json(json_str)
    assert from_json == manifest


def test_manifest_file_read_write(tmp_path: Path) -> None:
    """Verify writing manifest to a directory or file and reading it back."""
    manifest = _sample_manifest()
    target_dir = tmp_path / "bootstrap_run_1"

    out_file = write_manifest(manifest, target_dir)
    assert out_file == target_dir / "manifest.json"
    assert out_file.exists()

    # Read back from directory
    read_back = read_manifest(target_dir)
    assert read_back == manifest

    # Read back from direct file path
    read_direct = read_manifest(out_file)
    assert read_direct == manifest


def test_manifest_overwrite_protection(tmp_path: Path) -> None:
    """Verify writing fails when file exists unless overwrite=True."""
    manifest = _sample_manifest()
    target_dir = tmp_path / "run"

    write_manifest(manifest, target_dir)

    with pytest.raises(FileExistsError, match="Manifest file already exists"):
        write_manifest(manifest, target_dir, overwrite=False)

    # With overwrite=True, succeeds
    write_manifest(manifest, target_dir, overwrite=True)


def test_collision_detection_raises_error() -> None:
    """Encountering identical 8-hex short hashes with differing full digests raises ScientificHashCollisionError."""
    # Both start with "7a21c3f0", but remainder differs
    m1 = _sample_manifest(
        full_sha="7a21c3f011111111111111111111111111111111111111111111111111111111"
    )
    m2 = _sample_manifest(
        full_sha="7a21c3f022222222222222222222222222222222222222222222222222222222"
    )

    assert m1.short_hash == m2.short_hash == "7a21c3f0"
    assert m1.scientific_id_sha256 != m2.scientific_id_sha256

    with pytest.raises(
        ScientificHashCollisionError,
        match="collision detected for 8-hex prefix '7a21c3f0'",
    ):
        check_hash_collision(m1, m2)


def test_collision_detection_on_write(tmp_path: Path) -> None:
    """write_manifest raises ScientificHashCollisionError when colliding with existing file."""
    m1 = _sample_manifest(
        full_sha="7a21c3f011111111111111111111111111111111111111111111111111111111"
    )
    m2 = _sample_manifest(
        full_sha="7a21c3f022222222222222222222222222222222222222222222222222222222"
    )

    target_dir = tmp_path / "test_run"
    write_manifest(m1, target_dir)

    with pytest.raises(ScientificHashCollisionError):
        write_manifest(m2, target_dir, overwrite=True, check_collision=True)


def test_manifest_validation_scientific_id_sha256() -> None:
    """scientific_id_sha256 must be exactly 64 lowercase hexadecimal characters (L2)."""
    # Uppercase hex is rejected
    with pytest.raises(ValueError, match="must be exactly 64 lowercase hexadecimal"):
        _sample_manifest(
            full_sha="7A21C3F0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        )

    # Invalid length is rejected
    with pytest.raises(ValueError, match="must be exactly 64 lowercase hexadecimal"):
        _sample_manifest(full_sha="7a21c3f0")

    # Non-hex characters rejected
    with pytest.raises(ValueError, match="must be exactly 64 lowercase hexadecimal"):
        _sample_manifest(
            full_sha="7a21c3f0zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
        )


def test_manifest_validation_evidence_availability() -> None:
    """evidence_availability must be a mapping with boolean values for all 3 keys (L2)."""
    base = _sample_manifest()

    # Missing a required key
    with pytest.raises(ValueError, match="evidence_availability missing required keys"):
        Manifest(
            schema_version=base.schema_version,
            product_type=base.product_type,
            scientific_identity=base.scientific_identity,
            scientific_id_sha256=base.scientific_id_sha256,
            parents=base.parents,
            data_provenance=base.data_provenance,
            artifacts=base.artifacts,
            run_fields=base.run_fields,
            code_provenance=base.code_provenance,
            environment=base.environment,
            created_at=base.created_at,
            evidence_availability={"checkpoint_present": True},  # missing 2 keys
        )

    # Non-boolean value
    with pytest.raises(
        TypeError,
        match="evidence_availability\\['distances_present'\\] must be a boolean",
    ):
        Manifest(
            schema_version=base.schema_version,
            product_type=base.product_type,
            scientific_identity=base.scientific_identity,
            scientific_id_sha256=base.scientific_id_sha256,
            parents=base.parents,
            data_provenance=base.data_provenance,
            artifacts=base.artifacts,
            run_fields=base.run_fields,
            code_provenance=base.code_provenance,
            environment=base.environment,
            created_at=base.created_at,
            evidence_availability={
                "checkpoint_present": True,
                "distances_present": "yes",  # not a bool
                "chunks_present": False,
            },
        )


def test_collision_detection_sibling_runs_in_product_dir(tmp_path: Path) -> None:
    """write_manifest detects colliding short hashes in sibling subdirectories (M1)."""
    product_dir = tmp_path / "bootstrap"
    product_dir.mkdir()

    m1 = _sample_manifest(
        full_sha="7a21c3f011111111111111111111111111111111111111111111111111111111"
    )
    m2 = _sample_manifest(
        full_sha="7a21c3f022222222222222222222222222222222222222222222222222222222"
    )

    run_a = product_dir / "run_a"
    run_b = product_dir / "run_b"

    write_manifest(m1, run_a)

    # Writing run_b should detect sibling collision with run_a
    with pytest.raises(
        ScientificHashCollisionError,
        match="collision detected for 8-hex prefix '7a21c3f0'",
    ):
        write_manifest(m2, run_b, check_collision=True)


def test_collision_detection_single_file_sibling_runs(tmp_path: Path) -> None:
    """write_manifest detects colliding short hashes in single-file sibling manifests (M1)."""
    product_dir = tmp_path / "bootstrap"
    product_dir.mkdir()

    m1 = _sample_manifest(
        full_sha="7a21c3f011111111111111111111111111111111111111111111111111111111"
    )
    m2 = _sample_manifest(
        full_sha="7a21c3f022222222222222222222222222222222222222222222222222222222"
    )

    file_a = product_dir / "run_a.json"
    file_b = product_dir / "run_b.json"

    write_manifest(m1, file_a)

    with pytest.raises(
        ScientificHashCollisionError,
        match="collision detected for 8-hex prefix '7a21c3f0'",
    ):
        write_manifest(m2, file_b, check_collision=True)


def test_write_manifest_raises_on_corrupt_manifest(tmp_path: Path) -> None:
    """write_manifest does not swallow errors when an existing or sibling manifest is corrupt (M1)."""
    product_dir = tmp_path / "bootstrap"
    product_dir.mkdir()

    # Corrupt target file
    corrupt_file = product_dir / "run_corrupt" / "manifest.json"
    corrupt_file.parent.mkdir()
    corrupt_file.write_text("{ not valid json", encoding="utf-8")

    m = _sample_manifest()
    with pytest.raises((json.JSONDecodeError, ValueError)):
        write_manifest(m, corrupt_file.parent, overwrite=True, check_collision=True)
