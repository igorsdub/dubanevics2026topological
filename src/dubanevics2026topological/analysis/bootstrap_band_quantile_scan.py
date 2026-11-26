"""Sensitivity of the surviving peak count to the bootstrap band quantile.

The confidence band of the bootstrap stage has half-width ``c``, the
``band_quantile`` quantile of the REAL draws' bottleneck distance to the nominal
diagram (:func:`band_width`). Define the persistence threshold ``tau = 2c``: a
nominal peak survives when its persistence is strictly greater than ``tau``.

Two tables come out of this stage:

- ``scan.parquet`` (``q``, ``c``, ``surviving_peaks``): ``q`` swept over a fine
  grid from 0 to 1, ``c(q)`` from ``band_width`` itself (no second quantile
  implementation) and the count of peaks with persistence above ``2c(q)``.
  ``c`` is non-decreasing in ``q``, so the count is non-increasing.
- ``survival.parquet`` (``tau``, ``surviving_peaks``): the full survival curve of
  the nominal persistences, evaluated at ``tau = 0`` and at every distinct
  persistence, so its steps are exact. It needs no bootstrap draws.

Both are written to ``output/analysis/bootstrap_band_quantile_scan/``, followed by
``provenance.json``, the scan's record (its ``config``, the sha256 of the two tables and
the ``bootstrap_source`` the checkpoint was restored from, if any); the record is
written last, so it also marks a finished stage. The tables hold only counts and
thresholds, never sequence identities. The stage reads the finished bootstrap checkpoint
and never recomputes the bootstrap.

Where the tables are kept: a scan of a BY variant (``published_by`` or ``reflected_by``) is
also published to ``data/processed/band_quantile_scan/<state>/`` (``scan.parquet``,
``survival.parquet`` and ``provenance.json``); the scan figures read that folder, and a
missing scan in ``output/`` is restored from it. A selection without a bootstrap draw set
(the point-estimate graphs) has no scan.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..simplification import checkpoint_identity
from ..simplification.bootstrap_smoothed_tree import (
    BootstrapSmoothedTreeResult,
    band_width,
    load_cached_bootstrap_smoothed_tree,
)
from . import get_run_context
from .checkpoint import checkpoint_lock

SECTION = "bootstrap_band_quantile_scan"
SCAN_FILE = "scan.parquet"
SURVIVAL_FILE = "survival.parquet"
RECORD_NAME = "provenance.json"
SCAN_COLUMNS = ("q", "c", "surviving_peaks")
SURVIVAL_COLUMNS = ("tau", "surviving_peaks")
SURVIVAL_SCHEMA = {"tau": pl.Float64, "surviving_peaks": pl.Int64}
SCAN_SCHEMA = {"q": pl.Float64, "c": pl.Float64, "surviving_peaks": pl.Int64}


def scan_dir(context: Any | None = None) -> Path:
    """The scan directory under the context's output root (a context-less call resolves
    a fresh context now)."""
    root = Path((context or get_run_context()).output_root)
    return root / "analysis" / "bootstrap_band_quantile_scan"


def scan_paths(context: Any | None = None) -> tuple[Path, Path]:
    """The scan and survival table paths of ``context``."""
    directory = scan_dir(context)
    return directory / SCAN_FILE, directory / SURVIVAL_FILE


def survives(persistence: Any, c: float) -> Any:
    """Return whether ``persistence`` is strictly above ``2c`` (works on arrays).

    The same strict rule as the stage's ``clears_band`` column in
    ``bootstrap_smoothed_tree``; that module is part of the checkpoint
    fingerprint so it keeps its own copy, pinned to this one by a test.
    """
    return persistence > 2.0 * c


def validate_config(config: dict[str, Any]) -> float:
    """Return the validated ``grid_step``."""
    unknown = sorted(set(config) - {"grid_step"})
    if unknown:
        raise ValueError(
            f"Unknown [{SECTION}] settings {unknown}; known: ['grid_step']."
        )
    step = config["grid_step"]
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(step)
        or not 0 < step <= 1
    ):
        raise ValueError("grid_step must be finite and in (0, 1].")
    cells = 1.0 / step
    if not math.isclose(cells, round(cells), rel_tol=0, abs_tol=1e-9):
        raise ValueError("1 / grid_step must be a whole number.")
    return float(step)


def require_on_grid(grid_step: float, band_quantile: float) -> None:
    """Refuse a ``band_quantile`` that is not a grid point, before anything is written."""
    cells = round(1.0 / grid_step)
    scaled = band_quantile * cells
    if not math.isclose(scaled, round(scaled), rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"band_quantile {band_quantile} is not on the grid of step {grid_step}; "
            "choose a grid_step that divides it."
        )


def quantile_grid(grid_step: float) -> list[float]:
    """Return ``q = 0, step, ..., 1`` as ``k / n`` so that 0.95 is exact for 0.001."""
    cells = round(1.0 / grid_step)
    return [k / cells for k in range(cells + 1)]


def band_quantile_scan(
    draw_stats: pl.DataFrame,
    persistence: Sequence[float],
    grid: Sequence[float],
) -> pl.DataFrame:
    """Count the nominal peaks with persistence above ``2c(q)`` for every ``q``.

    ``draw_stats`` is the stage's draw table (only its ``real`` rows are read,
    through :func:`band_width`); ``persistence`` is the nominal peaks'
    persistence. The comparison is strict: a peak exactly at ``2c`` does not
    survive.
    """
    values = np.asarray(list(persistence), dtype=np.float64)
    rows = []
    for q in grid:
        c = band_width(draw_stats, q)
        rows.append(
            {"q": float(q), "c": c, "surviving_peaks": int(survives(values, c).sum())}
        )
    return pl.DataFrame(rows, schema=SCAN_SCHEMA).select(*SCAN_COLUMNS)


def scan_from_result(
    result: BootstrapSmoothedTreeResult, grid_step: float
) -> pl.DataFrame:
    """Scan the cached bootstrap checkpoint's draws and nominal peaks."""
    return band_quantile_scan(
        result.draw_stats,
        result.hills["nominal_persistence"].to_list(),
        quantile_grid(grid_step),
    )


def survival_curve(persistence: Sequence[float]) -> pl.DataFrame:
    """Return ``tau`` and the number of peaks with persistence strictly above it.

    ``tau`` runs over 0 and every distinct persistence, so the curve is exact: the
    count at a persistence value excludes the peaks tied at that value, and it is 0
    at the largest persistence.
    """
    values = np.asarray(list(persistence), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nominal persistence has non-finite values.")
    taus = np.unique(np.concatenate([[0.0], values]))
    rows = [{"tau": float(t), "surviving_peaks": int((values > t).sum())} for t in taus]
    return pl.DataFrame(rows, schema=SURVIVAL_SCHEMA).select(*SURVIVAL_COLUMNS)


def survival_from_result(result: BootstrapSmoothedTreeResult) -> pl.DataFrame:
    """Survival curve of the cached checkpoint's nominal persistences."""
    return survival_curve(result.hills["nominal_persistence"].to_list())


def _write_atomic(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_scan(scan: pl.DataFrame, path: Path) -> Path:
    """Write the scan atomically."""
    return _write_atomic(scan, path)


def _write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scan_config(
    settings: dict[str, Any], grid_step: float, scientific: dict[str, Any]
) -> dict[str, Any]:
    """What identifies a scan: the checkpoint settings, the scientific fingerprint of the
    bootstrap it was read from and the grid step (JSON-normalised)."""
    return json.loads(
        json.dumps(
            {
                "checkpoint": dict(settings),
                "scientific": scientific,
                "grid_step": grid_step,
            }
        )
    )


def scan_record(
    scan_path: Path,
    survival_path: Path,
    settings: dict[str, Any],
    grid_step: float,
    scientific: dict[str, Any],
) -> dict[str, Any]:
    """The scan's own record: its ``config`` and the sha256 of its two tables."""
    return {
        "config": scan_config(settings, grid_step, scientific),
        "artifacts": {
            path.name: checkpoint_identity.sha256_of(path)
            for path in (scan_path, survival_path)
        },
    }


def processed_scan_dir(state: str, context: Any | None = None) -> Path:
    """``processed/band_quantile_scan/<state>`` of ``context``."""
    from ..results import processed

    return processed.path(
        "band_quantile_scan",
        state,
        root=processed.processed_root(context or get_run_context()),
    )


def publish_scan(state: str, context: Any | None = None) -> Path:
    """Copy the scan tables and their record to ``processed/band_quantile_scan/<state>``."""
    source = scan_dir(context)
    target = processed_scan_dir(state, context)
    target.mkdir(parents=True, exist_ok=True)
    for name in (SCAN_FILE, SURVIVAL_FILE, RECORD_NAME):
        shutil.copyfile(source / name, target / name)
    return target


def compute_bootstrap_band_quantile_scan(
    settings: dict[str, Any],
    *,
    path: Path | None = None,
    survival_path: Path | None = None,
    context: Any | None = None,
) -> pl.DataFrame:
    """Read the bootstrap checkpoint and write the scan and survival tables.

    The two tables are written first and ``provenance.json`` (the scan's record) last,
    beside them. The scan of a BY variant is then published to
    ``processed/band_quantile_scan/<state>``.
    """
    default_path, default_survival = scan_paths(context)
    path = default_path if path is None else path
    survival_path = default_survival if survival_path is None else survival_path
    step = validate_config(settings["analysis"][SECTION])
    require_on_grid(
        step, settings["analysis"]["bootstrap_smoothed_tree"]["band_quantile"]
    )
    result = load_cached_bootstrap_smoothed_tree(settings, context=context)
    scan = scan_from_result(result, step)
    with checkpoint_lock(path.parent, "band quantile scan"):
        # The record marks a finished stage: a record of an earlier run must not
        # outlive the tables it described while the new ones are written.
        (path.parent / RECORD_NAME).unlink(missing_ok=True)
        _write_atomic(survival_from_result(result), survival_path)
        write_scan(scan, path)
        scientific = checkpoint_identity.scientific_fingerprint(
            result.provenance["fingerprint"]
        )
        record = scan_record(
            path, survival_path, result.summary["settings"], step, scientific
        )
        from .band_quantile_scan_panels import panel_for_variant

        panel = panel_for_variant(
            None if context is None else getattr(context, "variant_name", None),
            settings,
        )
        if panel is not None:
            record = {**record, "graph": panel.as_dict()}
        _write_json_atomic(record, path.parent / RECORD_NAME)
    if panel is not None and path.parent == scan_dir(context):
        publish_scan(panel.state, context)
    return scan


def ensure_scan_tables(
    settings: dict[str, Any],
    result: BootstrapSmoothedTreeResult,
    *,
    path: Path | None = None,
    survival_path: Path | None = None,
    context: Any | None = None,
) -> None:
    """Restore missing scan tables from ``processed/band_quantile_scan/<state>``.

    Only the scan of the context's own BY panel is used, and only when its checkpoint
    settings and bootstrap fingerprint equal the loaded checkpoint's; otherwise nothing
    happens and the caller reports the missing tables (``missing_scan_message``).
    """
    default_path, default_survival = scan_paths(context)
    path = default_path if path is None else path
    survival_path = default_survival if survival_path is None else survival_path
    if path.is_file() and survival_path.is_file():
        return
    from .band_quantile_scan_panels import panel_for_variant

    panel = panel_for_variant(
        None if context is None else getattr(context, "variant_name", None), settings
    )
    if panel is None:
        return
    source = processed_scan_dir(panel.state, context)
    try:
        record = json.loads((source / RECORD_NAME).read_text(encoding="utf-8"))
        step = validate_config(settings["analysis"][SECTION])
        expected = scan_config(
            result.summary["settings"],
            step,
            checkpoint_identity.scientific_fingerprint(result.provenance["fingerprint"]),
        )
        tables = (source / SCAN_FILE, source / SURVIVAL_FILE)
        if record["config"] != expected or not all(table.is_file() for table in tables):
            return
    except (OSError, ValueError, KeyError):
        return
    with checkpoint_lock(path.parent, "band quantile scan"):
        for table, destination in zip(
            (*tables, source / RECORD_NAME),
            (path, survival_path, path.parent / RECORD_NAME),
            strict=True,
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.restore")
            shutil.copy2(table, temporary)
            os.replace(temporary, destination)


def missing_scan_message(
    settings: dict[str, Any], result: BootstrapSmoothedTreeResult, path: Path
) -> str:
    """Why no band quantile table is there."""
    step = validate_config(settings["analysis"][SECTION])
    return (
        f"No band quantile table at {path}, and no matching scan under "
        f"data/processed/band_quantile_scan/ (grid_step {step}): run "
        "`just bootstrap-band-quantile-scan`, `just all` or `just fetch-data`."
    )


def load_bootstrap_band_quantile_scan(
    settings: dict[str, Any],
    result: BootstrapSmoothedTreeResult,
    *,
    path: Path | None = None,
    survival_path: Path | None = None,
    context: Any | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read both tables and refuse them unless they match the checkpoint."""
    default_path, default_survival = scan_paths(context)
    path = default_path if path is None else path
    survival_path = default_survival if survival_path is None else survival_path
    ensure_scan_tables(
        settings, result, path=path, survival_path=survival_path, context=context
    )
    for table in (path, survival_path):
        if not table.is_file():
            raise FileNotFoundError(missing_scan_message(settings, result, table))
    scan = pl.read_parquet(path)
    survival = pl.read_parquet(survival_path)
    expected = scan_from_result(result, validate_config(settings["analysis"][SECTION]))
    if (
        scan.columns != list(SCAN_COLUMNS)
        or survival.columns != list(SURVIVAL_COLUMNS)
        or not scan.equals(expected)
        or not survival.equals(survival_from_result(result))
    ):
        raise RuntimeError(
            "stale band quantile tables: they disagree with the bootstrap "
            "checkpoint; rerun `just bootstrap-band-quantile-scan`."
        )
    return scan, survival


def steps_around(
    scan: pl.DataFrame, quantile: float
) -> dict[str, dict[str, float | int] | None]:
    """Return the scan row at ``quantile`` and the nearest count changes each side.

    ``below`` is the largest grid ``q`` under ``quantile`` whose count differs
    from the count at ``quantile``, ``above`` the smallest grid ``q`` over it
    (``None`` when the count never changes on that side).
    """
    at = scan.filter(pl.col("q") == quantile)
    if at.height != 1:
        raise ValueError(f"The scan has no grid point at q = {quantile}.")
    count = int(at["surviving_peaks"][0])
    differs = pl.col("surviving_peaks") != count
    before = scan.filter((pl.col("q") < quantile) & differs)
    after = scan.filter((pl.col("q") > quantile) & differs)
    return {
        "at": at.row(0, named=True),
        "below": before.row(-1, named=True) if before.height else None,
        "above": after.row(0, named=True) if after.height else None,
    }
