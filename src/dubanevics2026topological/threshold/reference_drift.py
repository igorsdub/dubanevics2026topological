"""Drift check of recomputed tables against the reference hashes of the original run.

``config/data.toml`` ``[reference_hashes]`` keeps the byte hashes of the growth-rate table
and of the two BY classification tables the manuscript numbers were first made from.
A refit or a different library build can change those bytes without changing the science,
so a mismatch is never an error: it warns and, when a copy of the reference table is
available (``$FITNESSTREE_REFERENCE_DIR``, searched recursively by file name), writes a
short ``drift_report.json`` with the class-count deltas (functional, uncertain,
non-functional) and the largest absolute value differences, and prints a summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

CLASSES = ("functional", "uncertain", "non_functional")
REFERENCE_DIR_ENV = "FITNESSTREE_REFERENCE_DIR"
REPORT_NAME = "drift_report.json"
KEY_COLUMN = "dna_sequence"


def reference_hashes(config: dict[str, Any] | None = None) -> dict[str, str]:
    """The ``[reference_hashes]`` table of config/data.toml."""
    if config is None:
        from ..rawdata.files import load_config

        config = load_config()
    return dict(config["reference_hashes"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_reference_copy(
    pattern: str, directory: Path | str | None = None
) -> Path | None:
    """The one file matching ``pattern`` under the reference directory, or ``None``.

    A pattern that is an exact relative path to a file is taken as it is. Otherwise the
    pattern is searched recursively and must match exactly one file: with several matches
    the copy is ambiguous, so it warns and returns ``None`` rather than compare to a guess.
    """
    directory = directory or os.environ.get(REFERENCE_DIR_ENV)
    if not directory or not Path(directory).is_dir():
        return None
    exact = Path(directory) / pattern
    if exact.is_file():
        return exact
    found = sorted(Path(directory).rglob(pattern))
    if len(found) > 1:
        warnings.warn(
            f"Ambiguous reference copy for {pattern!r}: {len(found)} matches under "
            f"{directory} ({', '.join(str(f.relative_to(directory)) for f in found)}); "
            "none used. Keep one copy or name its relative path.",
            stacklevel=2,
        )
        return None
    return found[0] if found else None


def _max_differences(new: pl.DataFrame, reference: pl.DataFrame) -> dict[str, float]:
    joined = new.join(reference, on=KEY_COLUMN, how="inner", suffix="__reference")
    out: dict[str, float] = {}
    for column, dtype in new.schema.items():
        other = f"{column}__reference"
        if column == KEY_COLUMN or not dtype.is_float() or other not in joined.columns:
            continue
        difference = (joined[column] - joined[other]).abs()
        finite = difference.filter(difference.is_finite())
        out[column] = float(finite.max()) if finite.len() else 0.0
    return out


def class_counts(table: pl.DataFrame, column: str = "fdr_class") -> dict[str, int] | None:
    """Functional, uncertain and non-functional counts of a classification column.

    ``None`` (with a warning) when the table has no such column.
    """
    if column not in table.columns:
        warnings.warn(f"No class column {column!r} in the table; class counts skipped.", stacklevel=2)
        return None
    counts = table[column].value_counts()
    by_class = dict(zip(counts[column].to_list(), counts["count"].to_list()))
    return {name: int(by_class.get(name, 0)) for name in CLASSES}


def compare_tables(
    new: pl.DataFrame, reference: pl.DataFrame, *, class_column: str | None = None
) -> dict[str, Any]:
    """Row-count, value and (when ``class_column`` is given) class-count differences."""
    report: dict[str, Any] = {
        "rows": {"new": new.height, "reference": reference.height},
        "max_abs_difference": _max_differences(new, reference),
    }
    if class_column is not None:
        now, then = class_counts(new, class_column), class_counts(reference, class_column)
    if class_column is not None and now is not None and then is not None:
        report["class_counts"] = {
            "new": now,
            "reference": then,
            "delta": {name: now[name] - then[name] for name in CLASSES},
        }
        joined = new.join(
            reference.select(KEY_COLUMN, class_column), on=KEY_COLUMN, suffix="__reference"
        )
        mine, theirs = joined[class_column], joined[f"{class_column}__reference"]
        # A null on one side only is a change; two nulls are not.
        changed = (mine != theirs).fill_null(False) | (mine.is_null() != theirs.is_null())
        report["class_changed_rows"] = int(changed.sum())
    return report


def check_drift(
    table: Path,
    *,
    label: str,
    reference_sha256: str,
    reference_pattern: str,
    report_dir: Path,
    class_column: str | None = None,
    reference_dir: Path | str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any] | None:
    """Warn when ``table`` differs from the reference hash; report the deltas if a copy exists.

    Returns ``None`` when the hash matches, otherwise the report (also merged into
    ``report_dir/drift_report.json`` under ``label``). Never raises on a mismatch.
    """
    table = Path(table)
    digest = _sha256(table)
    if digest == reference_sha256:
        return None
    report: dict[str, Any] = {
        "table": table.name,
        "sha256": digest,
        "reference_sha256": reference_sha256,
    }
    log(
        f"WARNING: {label}: {table.name} differs from the reference table "
        f"(sha256 {digest[:8]} != {reference_sha256[:8]}); not an error."
    )
    copy = find_reference_copy(reference_pattern, reference_dir)
    if copy is None:
        report["reference_copy"] = None
        log(
            f"  No reference copy found (set ${REFERENCE_DIR_ENV} to a folder holding "
            f"{reference_pattern}); no delta report."
        )
    else:
        report["reference_copy"] = str(copy)
        report.update(
            compare_tables(
                pl.read_parquet(table), pl.read_parquet(copy), class_column=class_column
            )
        )
        if "class_counts" in report:
            log(f"  class-count deltas: {report['class_counts']['delta']}")
        log(f"  max |difference| per column: {report['max_abs_difference']}")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / REPORT_NAME
    existing = json.loads(target.read_text()) if target.is_file() else {}
    existing[label] = report
    target.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return report
