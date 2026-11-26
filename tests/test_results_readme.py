"""Tests for README documentation generation for output roots and product folders."""

from __future__ import annotations

from pathlib import Path

from dubanevics2026topological.results.manifest import (
    CURRENT_SCHEMA_VERSION,
    Manifest,
    write_manifest,
)
from dubanevics2026topological.results.readme import (
    generate_product_readme,
    generate_root_readme,
    write_product_readme,
    write_root_readme,
)


def _write_mock_manifest(
    run_dir: Path,
    short_hash: str = "7a21c3f0",
    host: str = "local",
    run_id: int = 51756602,
    created_at: str = "2026-10-27T14:00:00Z",
) -> None:
    manifest = Manifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        product_type="bootstrap",
        scientific_identity={"draws": 10000, "seed": 20261019},
        scientific_id_sha256=short_hash + "0" * 56,
        parents=[],
        data_provenance={},
        artifacts={
            "summary.json": {"bytes": 100, "sha256": "abc"},
            "distances.parquet": {"bytes": 200, "sha256": "def"},
        },
        run_fields={"host": host, "run": run_id},
        code_provenance={"commit": "1f1b4a9", "dirty": False, "branch": "main"},
        environment={
            "python_version": "3.13.0",
            "platform": "darwin",
            "uv_lock_sha256": "123",
        },
        created_at=created_at,
        evidence_availability={
            "checkpoint_present": True,
            "chunks_present": False,
            "distances_present": True,
        },
    )
    write_manifest(manifest, run_dir)


def test_root_readme_generation(tmp_path: Path) -> None:
    """Root README documents all standard product folders and naming conventions."""
    content = generate_root_readme(tmp_path)
    assert "# FitnessTree Analysis Outputs" in content
    assert "### `fitness/`" in content
    assert "### `landscape/`" in content
    assert "### `bootstrap/`" in content
    assert "### `tree/`" in content
    assert "### `leaves/`" in content
    assert "### `landscape_summary/`" in content
    assert "### `ridges/`" in content
    assert "DrWatson Parameter Labels" in content

    # Test file writing
    out_file = write_root_readme(tmp_path)
    assert out_file == tmp_path / "README.md"
    assert out_file.read_text(encoding="utf-8") == content


def test_product_readme_empty_directory(tmp_path: Path) -> None:
    """Product README without runs explains schema and reports no recorded runs."""
    prod_dir = tmp_path / "bootstrap"
    prod_dir.mkdir()

    content = generate_product_readme(prod_dir)
    assert "# Product: bootstrap" in content
    assert "schema version 1" in content
    assert "_No runs recorded yet._" in content

    out_file = write_product_readme(prod_dir)
    assert out_file == prod_dir / "README.md"
    assert out_file.read_text(encoding="utf-8") == content


def test_product_readme_with_manifest_runs(tmp_path: Path) -> None:
    """Product README compiles runs table from subdirectories containing manifests."""
    prod_dir = tmp_path / "bootstrap"
    prod_dir.mkdir()

    run1 = (
        prod_dir
        / "bootstrap__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_host=local_run=51756602_hash=7a21c3f0"
    )
    _write_mock_manifest(
        run1,
        short_hash="7a21c3f0",
        host="local",
        run_id=51756602,
        created_at="2026-10-27T14:00:00Z",
    )

    run2 = (
        prod_dir
        / "bootstrap__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_host=local_run=1_hash=b4d9e0a1"
    )
    _write_mock_manifest(
        run2,
        short_hash="b4d9e0a1",
        host="local",
        run_id=1,
        created_at="2026-10-27T15:00:00Z",
    )

    content = generate_product_readme(prod_dir)
    assert (
        "| Directory / Label | Short Hash | Host | Run ID | Created (UTC) | Artifacts |"
        in content
    )
    assert "`7a21c3f0`" in content
    assert "51756602" in content
    assert "`b4d9e0a1`" in content
    assert "2 files" in content
