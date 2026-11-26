"""Publish the bootstrap band-quantile scan and survival tables as a saved product.

The scan (``q``, ``c``, ``surviving_peaks``) and survival curve (``tau``,
``surviving_peaks``) are already computed by the ``bootstrap-band-quantile-scan``
stage. This product verifies them against the cached bootstrap checkpoint and
publishes them, so the two scan figures become pure presentation.

Locks are taken in a fixed order: the bootstrap checkpoint, then the scan
directory. The raw bytes of every consumed file are hashed inside those windows.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import bootstrap_band_quantile_scan as scan_module
from ..analysis import load_settings
from ..analysis.checkpoint import checkpoint_lock
from ..landscape.products import _publish
from ..results import RunContext
from ..simplification import checkpoint_identity
from ..simplification.bootstrap_smoothed_tree import (
    cache_dir as bootstrap_cache_dir,
)
from ..simplification.bootstrap_smoothed_tree import (
    checkpoint_dir as bootstrap_checkpoint_dir,
)
from ..simplification.bootstrap_smoothed_tree import (
    load_cached_bootstrap_smoothed_tree,
)

BOOTSTRAP_LABEL = "bootstrap smoothed tree"
SCAN_LABEL = "band quantile scan"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(directory: Path, prefix: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                key = f"{prefix}/{path.relative_to(directory).as_posix()}"
                hashes[key] = _file_sha256(path)
    return hashes


def _bootstrap_inputs(context: RunContext) -> dict[str, str]:
    """Raw-byte sha256 of the bootstrap checkpoint and working slot files.

    The caller must hold the bootstrap checkpoint lock.
    """
    hashes = {
        **_tree_hashes(bootstrap_checkpoint_dir(context), "bootstrap/checkpoint"),
        **_tree_hashes(bootstrap_cache_dir(context), "bootstrap/slot"),
    }
    if not hashes:
        raise FileNotFoundError(
            "Bootstrap checkpoint artifacts missing; run `just bootstrap-smoothed-tree`."
        )
    return hashes


def _band_scan_inputs(context: RunContext) -> dict[str, str]:
    """Raw-byte sha256 of the scan tables and their record.

    The caller must hold the scan lock.
    """
    scan_dir = scan_module.scan_dir(context)
    hashes: dict[str, str] = {}
    for name in (scan_module.SCAN_FILE, scan_module.SURVIVAL_FILE):
        path = scan_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Band quantile table {path} missing; run `just bootstrap-band-quantile-scan`."
            )
        hashes[f"scan/{name}"] = _file_sha256(path)
    record = scan_dir / scan_module.RECORD_NAME
    if not record.is_file():
        raise FileNotFoundError(
            "Band quantile scan has no provenance record; "
            "run `just bootstrap-band-quantile-scan`."
        )
    hashes[f"scan/{scan_module.RECORD_NAME}"] = _file_sha256(record)
    return hashes


def _identity(settings: dict[str, Any], result: Any) -> dict[str, Any]:
    step = scan_module.validate_config(settings["analysis"][scan_module.SECTION])
    scientific = checkpoint_identity.scientific_fingerprint(
        result.provenance["fingerprint"]
    )
    return {
        "grid_step": step,
        "band_quantile": float(
            settings["analysis"]["bootstrap_smoothed_tree"]["band_quantile"]
        ),
        "bootstrap_scientific": json.loads(json.dumps(scientific, sort_keys=True)),
    }


def _verified_tables(
    settings: dict[str, Any], result: Any, context: RunContext
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the scan tables and require them to match the checkpoint."""
    scan_path, survival_path = scan_module.scan_paths(context)
    scan = pl.read_parquet(scan_path)
    survival = pl.read_parquet(survival_path)
    step = scan_module.validate_config(settings["analysis"][scan_module.SECTION])
    if scan.columns != list(scan_module.SCAN_COLUMNS) or survival.columns != list(
        scan_module.SURVIVAL_COLUMNS
    ):
        raise RuntimeError(
            "band quantile tables have unexpected columns; rerun "
            "`just bootstrap-band-quantile-scan`."
        )
    if not scan.equals(scan_module.scan_from_result(result, step)) or (
        not survival.equals(scan_module.survival_from_result(result))
    ):
        raise RuntimeError(
            "stale band quantile tables: they disagree with the bootstrap "
            "checkpoint; rerun `just bootstrap-band-quantile-scan`."
        )
    return scan, survival


def build_bootstrap_band_quantile_scan_product(
    context: RunContext | None = None,
) -> Path:
    """Build the bootstrap_band_quantile_scan product."""
    context = context or RunContext()
    settings = load_settings(context=context)
    scan_dir = scan_module.scan_dir(context)

    # The bootstrap reader takes the checkpoint lock itself, so hash before and
    # after it, inside the lock, and require the bytes to be unchanged.
    with checkpoint_lock(bootstrap_checkpoint_dir(context), BOOTSTRAP_LABEL):
        before = _bootstrap_inputs(context)
    result = load_cached_bootstrap_smoothed_tree(settings=settings, context=context)
    with checkpoint_lock(bootstrap_checkpoint_dir(context), BOOTSTRAP_LABEL):
        bootstrap_hashes = _bootstrap_inputs(context)
        if bootstrap_hashes != before:
            raise RuntimeError("bootstrap checkpoint changed while it was read; retry.")
        # Restoring missing tables locks the scan directory itself (fixed order).
        scan_module.ensure_scan_tables(settings, result, context=context)
        with checkpoint_lock(scan_dir, SCAN_LABEL):
            scan_hashes = _band_scan_inputs(context)
            scan, survival = _verified_tables(settings, result, context)
            artifact_bytes = {
                name: (scan_dir / name).read_bytes()
                for name in (scan_module.SCAN_FILE, scan_module.SURVIVAL_FILE)
            }
            if {
                f"scan/{name}": hashlib.sha256(data).hexdigest()
                for name, data in artifact_bytes.items()
            } != {k: v for k, v in scan_hashes.items() if k.endswith(".parquet")}:
                raise RuntimeError("band quantile tables changed while read; retry.")

    identity = _identity(settings, result)
    label_fields = {"bandq": identity["band_quantile"]}
    input_data_sha256s = {**scan_hashes, **bootstrap_hashes}
    code_paths = (
        Path(inspect.getfile(scan_module)).resolve(),
        Path(__file__).resolve(),
    )

    def write_artifacts(directory: Path) -> None:
        for name, data in artifact_bytes.items():
            (directory / name).write_bytes(data)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "grid_step": identity["grid_step"],
                    "band_quantile": identity["band_quantile"],
                    "scan_rows": scan.height,
                    "survival_rows": survival.height,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "bootstrap_band_quantile_scan",
        label_fields,
        identity,
        context=context,
        input_data_sha256s=input_data_sha256s,
        parents=(),
        code_paths=code_paths,
        write_artifacts=write_artifacts,
    )
    return path


def _band_scan_input_hashes(context: RunContext) -> dict[str, str]:
    """Fresh namespaced input hashes for the saved-figure-inputs loader."""
    with checkpoint_lock(bootstrap_checkpoint_dir(context), BOOTSTRAP_LABEL):
        boot = _bootstrap_inputs(context)
        with checkpoint_lock(scan_module.scan_dir(context), SCAN_LABEL):
            scan = _band_scan_inputs(context)
    return {**scan, **boot}
