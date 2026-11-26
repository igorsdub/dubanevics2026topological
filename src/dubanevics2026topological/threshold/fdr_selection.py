"""The ``fdr_fixed_cutoff`` Selection rule: a reviewed FDR classification snapshot.

The snapshot is a directory ``data/processed/classification/<variant>/`` holding the producer's
``provenance.json`` and per-sequence ``fdr_classification_T=<T>_cutoff=<name>_method=
<method>_q=<q>.parquet`` tables (``threshold/classification_probability_scans.py``).
``[clamped] candidate_functional_threshold`` names the cutoff, method, q and the snapshot
directory; ``T`` is never typed and the table's integrity is the folder's ``MANIFEST.tsv``. It is read from the
snapshot's own ``provenance.json`` (the cutoff's ``source_precision_threshold`` and
``applied_threshold``), and the table's file name must carry the applied ``T``.

The Functional set is the table's ``functional`` class only. ``uncertain`` and
``non_functional`` (below-cutoff point-estimate) sequences are not functional; the
graph rule of ``clamped_landscapes.prepare_clamped_landscape`` is unchanged, so they
enter only as fringe neighbours of a functional vertex.

The table is read fail-closed: a missing table, a duplicate or missing DNA identity, an unknown class or a wild type that is not functional raises.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from dubanevics2026topological.analysis import ROOT

FDR_FIXED_CUTOFF = "fdr_fixed_cutoff"
DECLARATION_FIELDS = ("cutoff", "method", "q", "input_dir")
CLASSES = ("functional", "uncertain", "non_functional")
# Authorised scope: BY at nominal q = 0.05 for the two fixed cutoffs. BH stays a separate sensitivity output.
SCOPED_METHOD = "by"
SCOPED_Q = 0.05
SCOPED_CUTOFFS = ("papkou_original", "median_reflected")


@dataclass(frozen=True)
class FdrSnapshot:
    """The verified table, its threshold record and its identity."""

    table: pl.DataFrame
    summary: dict[str, Any]
    source: dict[str, Any]


def loader_source_sha256() -> str:
    """Hash of the loader and threshold-rule sources, so a code change invalidates caches.

    Stage code lists name only some modules and ``scientific_code_paths`` does not follow
    imports, so every consumer reads this through the selection source and identity.
    """
    digest = hashlib.sha256()
    for name in (
        "fdr_selection.py",
        "functional_threshold.py",
        "classification_probability_scans.py",
        "candidate_threshold.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((Path(__file__).with_name(name)).read_bytes())
    return digest.hexdigest()


def table_name(applied: float, cutoff: str, method: str, q: float) -> str:
    return (
        f"fdr_classification_T={applied}_cutoff={cutoff}_method={method}_q={q}.parquet"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cutoff_record(provenance: Mapping[str, Any], cutoff: str) -> Mapping[str, Any]:
    matches = [item for item in provenance["cutoffs"] if item["name"] == cutoff]
    if len(matches) != 1:
        raise ValueError(f"The snapshot provenance names no unique cutoff {cutoff!r}.")
    return matches[0]


def _snapshot_directory(input_dir: str, root: Path | None) -> Path:
    """``input_dir`` under the repository root; ``data/processed/...`` follows
    ``FITNESSTREE_PROCESSED_ROOT`` when that names another processed folder."""
    parts = Path(input_dir).parts
    override = os.environ.get("FITNESSTREE_PROCESSED_ROOT", "").strip()
    if override and parts[:2] == ("data", "processed"):
        return Path(override).joinpath(*parts[2:])
    return (ROOT if root is None else Path(root)) / input_dir


def _summary_and_path(
    declaration: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    missing = [key for key in DECLARATION_FIELDS if key not in declaration]
    if missing:
        raise ValueError(f"fdr_fixed_cutoff declaration lacks {missing}.")
    for key in ("cutoff", "method", "input_dir"):
        if not isinstance(declaration[key], str) or not declaration[key]:
            raise ValueError(f"fdr_fixed_cutoff {key} must be a non-empty string.")
    if declaration["method"] != SCOPED_METHOD:
        raise ValueError(
            f"fdr_fixed_cutoff method must be {SCOPED_METHOD!r} (BY is the primary "
            f"correction), not {declaration['method']!r}; BH outputs stay separate."
        )
    q = declaration["q"]
    if isinstance(q, bool) or not isinstance(q, float) or q != SCOPED_Q:
        raise ValueError(
            f"fdr_fixed_cutoff q must be the nominal {SCOPED_Q}, not {q!r}."
        )
    if declaration["cutoff"] not in SCOPED_CUTOFFS:
        raise ValueError(
            f"fdr_fixed_cutoff cutoff must be one of {SCOPED_CUTOFFS}, "
            f"not {declaration['cutoff']!r}."
        )
    directory = _snapshot_directory(str(declaration["input_dir"]), root)
    provenance_path = directory / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"No FDR classification at {directory}; run `just classification-probability-scans`, "
            "`just all` or `just fetch-data`."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = _cutoff_record(provenance, str(declaration["cutoff"]))
    applied = float(record["applied_threshold"])
    path = directory / table_name(
        applied,
        str(declaration["cutoff"]),
        str(declaration["method"]),
        declaration["q"],
    )
    summary = {
        "threshold": {
            "unrounded": float(record["source_precision_threshold"]),
            "rounded": applied,
        },
        "cutoff": str(declaration["cutoff"]),
        "method": str(declaration["method"]),
        "q": float(declaration["q"]),
    }
    return summary, path, provenance_path


def threshold_summary(
    declaration: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """The ``summary.json``-shaped threshold record, without reading the table."""
    return _summary_and_path(declaration, root=root)[0]


def require_full_population(
    classes: Mapping[str, str], vertices: Collection[str]
) -> None:
    """The classified identities must be exactly the Full graph's vertices."""
    classified, graph = set(classes), set(vertices)
    if classified != graph:
        raise ValueError(
            "The FDR classification and the Full graph describe different sequences: "
            f"{len(classified - graph)} classified but not in the graph (first "
            f"{sorted(classified - graph)[:3]!r}), {len(graph - classified)} in the graph "
            f"but unclassified (first {sorted(graph - classified)[:3]!r})."
        )


def load_fdr_snapshot(
    declaration: Mapping[str, Any],
    *,
    wild_type: str | None = None,
    root: Path | None = None,
) -> FdrSnapshot:
    """Read and verify the declared FDR classification table."""
    summary, path, provenance_path = _summary_and_path(declaration, root=root)
    if not path.is_file():
        raise FileNotFoundError(
            f"FDR classification table {path} is missing; run "
            "`just classification-probability-scans`, `just all` or `just fetch-data`."
        )
    actual = _sha256(path)
    table = pl.read_parquet(path, columns=["dna_sequence", "fdr_class"])
    if table["dna_sequence"].null_count() or (table["dna_sequence"] == "").any():
        raise ValueError(f"{path.name} has a null or empty DNA sequence identity.")
    if table["dna_sequence"].n_unique() != table.height:
        raise ValueError(f"{path.name} has duplicate DNA sequence identities.")
    unknown = set(table["fdr_class"].unique().to_list()) - set(CLASSES)
    if unknown:
        raise ValueError(f"{path.name} has unknown classes {sorted(unknown)}.")
    if wild_type is not None:
        found = table.filter(pl.col("dna_sequence") == wild_type)["fdr_class"].to_list()
        if found != ["functional"]:
            raise ValueError(
                f"The wild type {wild_type!r} is classed {found!r}, not 'functional', "
                "in the FDR classification."
            )
    counts = {name: int((table["fdr_class"] == name).sum()) for name in CLASSES}
    source = {
        "rule": FDR_FIXED_CUTOFF,
        "table": path.name,
        "table_sha256": actual,
        "loader_sha256": loader_source_sha256(),
        "provenance_sha256": _sha256(provenance_path),
        "cutoff": summary["cutoff"],
        "method": summary["method"],
        "q": summary["q"],
        "row_count": table.height,
        "class_counts": counts,
    }
    return FdrSnapshot(table, summary, source)
