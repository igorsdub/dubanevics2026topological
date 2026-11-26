"""Full-network component persistence and threshold diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow
from topographer.topology import ScalarNetworkTopology

from . import (
    ROOT,
    _write_json_atomic,
    get_run_context,
    load_settings,
    scientific_fingerprint,
)
from .checkpoint import checkpoint_lock

PAIR_SCHEMA = {
    "pair_id": pl.Int64,
    "pair_kind": pl.String,
    "homology_dimension": pl.Int64,
    "birth_sequence": pl.String,
    "death_sequence": pl.String,
    "birth_has_stop_codon": pl.Boolean,
    "death_has_stop_codon": pl.Boolean,
    "birth_scalar": pl.Float64,
    "death_scalar": pl.Float64,
    "persistence": pl.Float64,
}
# Schema 3 renamed split_tree_leading_saddle to join_tree_trailing_saddle.
DIAGNOSTICS_SCHEMA = 3
_PAIR_KINDS = {"ordinary", "relative", "extended_positive"}


class ComponentPaths(NamedTuple):
    """Where the component-persistence checkpoint lives under one output root."""

    analysis_dir: Path
    pairs: Path
    diagnostics: Path
    provenance: Path
    pending: Path


def component_paths(context: Any | None = None) -> ComponentPaths:
    """The checkpoint paths of ``context``; a context-less call resolves one now."""
    if context is None:
        context = get_run_context()
    analysis_dir = Path(context.output_root) / "analysis"
    return ComponentPaths(
        analysis_dir,
        analysis_dir / "component_pairs.parquet",
        analysis_dir / "component_diagnostics.json",
        analysis_dir / "component_provenance.json",
        analysis_dir / ".component-analysis-pending",
    )


def compute_component_persistence_analysis(
    settings: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Compute full-network component persistence and exploratory threshold evidence."""
    settings = settings or load_settings()
    dataset = settings["analysis"]["dataset"]
    workflow = Papkou2023Workflow(
        landscape=dataset["landscape"],
        fitness_wt=pl.read_parquet(ROOT / dataset["fitness_wt"]),
        fitness_diff=pl.read_parquet(ROOT / dataset["fitness_diff"]),
    )
    network = workflow.build_network()
    result = ScalarNetworkTopology(network).component_persistence()
    stop_by_sequence = {
        str(payload["dna_sequence"]): bool(payload.get("has_stop_codon", False))
        for payload in network.graph.nodes()
    }
    pairs = _pairs_with_sequence_identities(result.pairs, stop_by_sequence)
    diagnostics = build_threshold_diagnostics(pairs)
    return pairs, diagnostics


def build_threshold_diagnostics(pairs: pl.DataFrame) -> dict[str, Any]:
    """Summarize ordinary deaths and report the join-tree trailing saddle.

    The pairs are the ascending (sublevel) pairs, so the join tree is the merge
    tree. The maximum ordinary death scalar is the saddle where the last ordinary
    join-tree pair closes, adjacent to the root, by which point every sublevel
    component has merged: the join tree's trailing saddle. It is a reported
    quantity, not an input to the Candidate functional threshold, which is
    computed from the Full-graph raw fitness (see ``candidate_threshold``).
    """
    if not set(PAIR_SCHEMA).issubset(pairs.columns):
        missing = sorted(set(PAIR_SCHEMA) - set(pairs.columns))
        raise ValueError(f"Component pair table is missing columns: {missing}.")
    ordinary = pairs.filter(pl.col("pair_kind") == "ordinary")
    global_pairs = pairs.filter(pl.col("pair_kind") == "extended_positive")
    if global_pairs.height != 1:
        raise ValueError("Component persistence must contain exactly one global pair.")
    global_minimum = float(global_pairs["birth_scalar"][0])
    frequencies = (
        ordinary.group_by("death_scalar")
        .len(name="count")
        .sort("death_scalar")
        .to_dicts()
    )
    trailing_saddle = float(ordinary["death_scalar"].max()) if ordinary.height else None
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "pair_count": pairs.height,
        "ordinary_pair_count": ordinary.height,
        "ordinary_death_value_frequencies": [
            {"death_scalar": float(row["death_scalar"]), "count": int(row["count"])}
            for row in frequencies
        ],
        "join_tree_trailing_saddle": trailing_saddle,
        "global_minimum": global_minimum,
    }


# The ascending checkpoint's pair kinds read as a descending sweep: its ``relative``
# pairs are the ordinary H0 pairs and its ``ordinary`` pairs the relative H1 pairs
# (kind, homology dimension). The one place this mapping is written; the figure's
# ``descending_component_pairs`` reads it too. Other kinds (the global pair) keep
# their kind and dimension.
DESCENDING_PAIR_KINDS: dict[str, tuple[str, int]] = {
    "relative": ("ordinary", 0),
    "ordinary": ("relative", 1),
}


def descending_pair_counts(pairs: pl.DataFrame) -> dict[str, int]:
    """The two counts of the Full-graph persistence figure, from the ascending pairs.

    ``pairs`` is the ascending checkpoint. Read as a descending sweep, its
    ``relative`` pairs are the ordinary H0 pairs (peak birth, split-saddle death) and
    its ``ordinary`` pairs are the relative pairs (valley birth, join-saddle death);
    the one ``extended_positive`` pair is the global maximum's, closed at the global
    minimum by convention. ``ordinary_h0_pair_count`` is the ordinary H0 pairs
    including that global pair; ``valley_count`` is the relative pairs plus one (the
    global pair's mirror at the global maximum). These are the numbers
    ``full_tree_persistence`` prints as "peaks" and "valleys".
    """
    _validate_pair_table(pairs)
    kinds = pairs["pair_kind"]
    global_pairs = int((kinds == "extended_positive").sum())
    if global_pairs != 1:
        raise ValueError("Component persistence must contain exactly one global pair.")
    descending = [DESCENDING_PAIR_KINDS.get(kind, (kind, 0))[0] for kind in kinds]
    return {
        "ordinary_h0_pair_count": descending.count("ordinary") + global_pairs,
        "valley_count": descending.count("relative") + global_pairs,
    }


def load_cached_component_persistence(
    *, context: Any | None = None
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load component artifacts without hashing large inputs or recomputing analysis."""
    paths = component_paths(context)
    pairs_path, diagnostics_path = paths.pairs, paths.diagnostics
    pending_path, provenance_path = paths.pending, paths.provenance

    if pending_path.exists():
        raise RuntimeError(
            "Component analysis refresh is pending; rerun `just analysis` first."
        )
    if not all(
        path.is_file() for path in (pairs_path, diagnostics_path, provenance_path)
    ):
        raise FileNotFoundError("No complete component-persistence checkpoint exists.")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(
            "Component-persistence checkpoint metadata is unreadable."
        ) from err
    if provenance.get("schema") != 1 or provenance.get("fingerprint") is None:
        raise ValueError("Component-persistence checkpoint provenance is invalid.")
    if provenance.get("pairs_sha256") != _sha256(pairs_path) or provenance.get(
        "diagnostics_sha256"
    ) != _sha256(diagnostics_path):
        raise ValueError(
            "Component-persistence artifacts do not match their provenance."
        )
    pairs = pl.read_parquet(pairs_path)
    _validate_pair_table(pairs)
    if diagnostics.get("schema") != DIAGNOSTICS_SCHEMA:
        raise ValueError(
            "Component-persistence diagnostics use an old schema; "
            "rerun `just component-analysis`."
        )
    if diagnostics.get("pair_count") != pairs.height:
        raise ValueError(
            "Component-persistence diagnostics do not match the pair table."
        )
    return pairs, diagnostics


def load_or_compute_component_persistence(
    *,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load matching cached component artifacts or compute and publish a fresh pair."""
    settings = settings or load_settings(context=context)
    paths = component_paths(context)
    pairs_path, diagnostics_path = paths.pairs, paths.diagnostics
    pending_path, provenance_path = paths.pending, paths.provenance
    analysis_dir = paths.analysis_dir

    with checkpoint_lock(analysis_dir / "component_persistence", "component analysis"):
        fingerprint = scientific_fingerprint(settings)
        if (
            not force
            and not pending_path.exists()
            and all(
                path.is_file()
                for path in (pairs_path, diagnostics_path, provenance_path)
            )
        ):
            try:
                saved = json.loads(provenance_path.read_text(encoding="utf-8"))
                if saved.get("fingerprint") == fingerprint:
                    return load_cached_component_persistence(context=context)
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        pairs, diagnostics = compute_component_persistence_analysis(settings)
        _validate_pair_table(pairs)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(
            "component analysis refresh pending\n", encoding="utf-8"
        )
        _write_parquet_atomic(pairs_path, pairs)
        _write_json_atomic(diagnostics_path, diagnostics)
        _write_json_atomic(
            provenance_path,
            {
                "schema": 1,
                "fingerprint": fingerprint,
                "topographer_revision": _topographer_revision(),
                "pair_count": pairs.height,
                "pairs_sha256": _sha256(pairs_path),
                "diagnostics_sha256": _sha256(diagnostics_path),
            },
        )
        pending_path.unlink(missing_ok=True)
        return pairs, diagnostics


def _pairs_with_sequence_identities(
    pairs: pl.DataFrame, stop_by_sequence: dict[str, bool] | None = None
) -> pl.DataFrame:
    required = {
        "pair_id",
        "pair_kind",
        "homology_dimension",
        "birth_simplex",
        "death_simplex",
        "birth_scalar",
        "death_scalar",
        "persistence",
    }
    if not required.issubset(pairs.columns):
        raise ValueError(
            f"Topographer component results lack columns: {sorted(required - set(pairs.columns))}."
        )
    rows = []
    stop_by_sequence = stop_by_sequence or {}
    for row in pairs.iter_rows(named=True):
        birth = row["birth_simplex"]
        death = row["death_simplex"]
        if len(birth) != 1 or len(death) != 1:
            raise ValueError(
                "Component persistence endpoints must be single-vertex sequences."
            )
        rows.append(
            {
                "pair_id": int(row["pair_id"]),
                "pair_kind": str(row["pair_kind"]),
                "homology_dimension": int(row["homology_dimension"]),
                "birth_sequence": str(birth[0]),
                "death_sequence": str(death[0]),
                "birth_has_stop_codon": stop_by_sequence.get(str(birth[0]), False),
                "death_has_stop_codon": stop_by_sequence.get(str(death[0]), False),
                "birth_scalar": float(row["birth_scalar"]),
                "death_scalar": float(row["death_scalar"]),
                "persistence": float(row["persistence"]),
            }
        )
    result = (
        pl.DataFrame(rows, schema=PAIR_SCHEMA)
        if rows
        else pl.DataFrame(schema=PAIR_SCHEMA)
    )
    _validate_pair_table(result)
    return result


def _validate_pair_table(pairs: pl.DataFrame) -> None:
    missing = set(PAIR_SCHEMA) - set(pairs.columns)
    if missing:
        raise ValueError(f"Component pair table is missing columns: {sorted(missing)}.")
    if pairs.filter(~pl.col("pair_kind").is_in(sorted(_PAIR_KINDS))).height:
        raise ValueError("Unexpected component persistence pair category.")
    if any(value for value in pairs.null_count().row(0)):
        raise ValueError("Component persistence pair fields must not be null.")
    if pairs["pair_id"].n_unique() != pairs.height:
        raise ValueError("Component persistence pair IDs must be unique.")
    invalid_dimensions = pairs.filter(
        ((pl.col("pair_kind") == "ordinary") & (pl.col("homology_dimension") != 0))
        | ((pl.col("pair_kind") == "relative") & (pl.col("homology_dimension") != 1))
        | (
            (pl.col("pair_kind") == "extended_positive")
            & (pl.col("homology_dimension") != 0)
        )
    )
    if invalid_dimensions.height:
        raise ValueError(
            "Component pair categories do not match their homology dimensions."
        )
    if pairs.filter(
        pl.col("birth_scalar").is_nan()
        | pl.col("death_scalar").is_nan()
        | pl.col("persistence").is_nan()
        | pl.col("birth_scalar").is_infinite()
        | pl.col("death_scalar").is_infinite()
        | pl.col("persistence").is_infinite()
    ).height:
        raise ValueError("Component persistence scalars must be finite.")
    if pairs.filter(
        (pl.col("death_scalar") - pl.col("birth_scalar"))
        .abs()
        .sub(pl.col("persistence"))
        .abs()
        > 1e-10
    ).height:
        raise ValueError(
            "Pair persistence must equal the absolute endpoint scalar difference."
        )
    invalid_directions = pairs.filter(
        (
            (pl.col("pair_kind") == "ordinary")
            & (pl.col("death_scalar") < pl.col("birth_scalar"))
        )
        | (
            (pl.col("pair_kind") == "relative")
            & (pl.col("death_scalar") > pl.col("birth_scalar"))
        )
        | (
            (pl.col("pair_kind") == "extended_positive")
            & (pl.col("death_scalar") < pl.col("birth_scalar"))
        )
    )
    if invalid_directions.height:
        raise ValueError("Component pair endpoint scalars have an invalid direction.")


# Public name for other analysis modules; the underscored one stays for existing callers.
validate_pair_table = _validate_pair_table


def _write_parquet_atomic(path: Path, frame: pl.DataFrame) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _topographer_revision() -> str | None:
    """Read the pinned source revision when installed from the local git checkout."""
    try:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    except OSError:
        return None
    return (
        project.get("tool", {})
        .get("uv", {})
        .get("sources", {})
        .get("topographer", {})
        .get("rev")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
