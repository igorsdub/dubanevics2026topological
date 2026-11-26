"""All-Candidate-edge evidence with an independent atomic checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from ..landscape.selected_landscape import PreparedLandscape
from ..landscape.selection import (
    candidate_selection_fingerprint,
    prepare_candidate_landscape,
)
from . import (
    ROOT,
    dataset_input_key,
    dataset_input_path,
    get_run_context,
    load_settings,
)
from .edge_evidence import global_contrasts


def _output_root(context: Any | None) -> Path:
    """The output root of ``context``; a context-less call resolves a fresh one now."""
    return Path((context or get_run_context()).output_root)


def cache_dir(context: Any | None = None) -> Path:
    """The evidence checkpoint directory under the context's output root."""
    return _output_root(context) / "analysis" / "candidate_edge_evidence"


def build_candidate_edge_evidence(
    diff: pl.DataFrame, prepared: PreparedLandscape, wt: pl.DataFrame | None = None
) -> dict[str, Any]:
    """Match each graph edge exactly once, correcting the complete source family."""
    graph = prepared.network.graph
    vertices = {}
    for index in graph.node_indices():
        node = graph[index]
        identity = node["dna_sequence"]
        value = float(node["raw_fitness"])
        if identity in vertices or not identity or not math.isfinite(value):
            raise ValueError(f"Invalid or duplicate Candidate identity {identity!r}.")
        vertices[identity] = value
    endpoint_se = {}
    if wt is not None:
        if not {"dna_to", "s", "se"}.issubset(wt.columns):
            raise ValueError("fitness_wt must supply dna_to, s and se.")
        for measurement in wt.select("dna_to", "s", "se").to_dicts():
            identity = measurement["dna_to"]
            if identity in endpoint_se:
                raise ValueError(f"Duplicate endpoint measurement {identity!r}.")
            value = float(measurement["se"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid endpoint uncertainty for {identity!r}.")
            if identity in vertices and not math.isclose(
                float(measurement["s"]), vertices[identity], abs_tol=1e-12
            ):
                raise ValueError(f"Endpoint raw-fitness mismatch for {identity!r}.")
            endpoint_se[identity] = value
    pairs = []
    for left, right in graph.edge_list():
        pair = tuple(
            sorted((graph[left]["dna_sequence"], graph[right]["dna_sequence"]))
        )
        if pair[0] == pair[1]:
            raise ValueError(f"Self-loop Candidate edge {pair!r}.")
        pairs.append(pair)
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate Candidate edges violate one-to-one coverage.")
    evidence, disagreements, _ = global_contrasts(diff, vertices, set(pairs))
    disagreement_ids = {item["contrast_id"] for item in disagreements}
    rows = []
    for left, right in sorted(pairs):
        lo, hi = vertices[left], vertices[right]
        row = dict(evidence[(left, right)])
        row.pop("edge_key")
        rows.append(
            {
                "edge_id": f"{left}|{right}",
                "dna_left": left,
                "dna_right": right,
                "fitness_left": lo,
                "fitness_right": hi,
                "fitness_standard_error_left": endpoint_se.get(left),
                "fitness_standard_error_right": endpoint_se.get(right),
                "fitness_lower": min(lo, hi),
                "fitness_upper": max(lo, hi),
                "fitness_delta": hi - lo,
                "absolute_fitness_delta": abs(hi - lo),
                "reported_raw_sign_disagreement": row["contrast_id"]
                in disagreement_ids,
                **row,
            }
        )
    edges = pl.DataFrame(rows)
    if not rows:
        raise ValueError("Candidate graph contains no evidence edges.")
    return {
        "edges": edges,
        "diagnostics": {
            "candidate_functional_threshold": prepared.threshold,
            "vertex_count": len(vertices),
            "edge_count": len(pairs),
            "global_bh_family_size": diff.height,
            "bh_family": "all_fitness_diff_rows_including_zero_deltas",
            "duplicate_policy": "reject_all_duplicate_unordered_source_pairs",
            "fitness_delta_orientation": "dna_right_minus_dna_left",
            "qualification_rule": "strict_q_greater_than_explicit_alpha",
            "reported_raw_sign_disagreement_count": len(disagreements),
            "reported_raw_sign_disagreements": disagreements,
            "candidate_edge_sign_disagreement_count": sum(
                row["reported_raw_sign_disagreement"] for row in rows
            ),
            "graph_selection": prepared.diagnostics,
        },
    }


def qualify_edges(edges: pl.DataFrame, alpha: float) -> pl.DataFrame:
    """Add strict exploratory qualification without changing cached evidence."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1].")
    return edges.with_columns((pl.col("q_value") > alpha).alias("qualifies"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(
    settings: dict[str, Any], *, context: Any | None = None
) -> dict[str, Any]:
    scientific = settings["analysis"]
    dataset = {
        key: dataset_input_path(scientific["dataset"][key], context, root=ROOT)
        for key in ("fitness_wt", "fitness_diff")
    }
    paths = [
        Path(__file__),
        Path(__file__).with_name("edge_evidence.py"),
        Path(__file__).resolve().parents[1] / "landscape" / "selection.py",
        Path(__file__).resolve().parents[1] / "threshold" / "functional_threshold.py",
        Path(__file__).resolve().parents[1] / "landscape" / "selected_landscape.py",
        Path(__file__).resolve().parents[1] / "landscape" / "full_population.py",
        ROOT / "uv.lock",
    ]
    return {
        "correction": scientific.get(
            "candidate_edge_evidence",
            {
                "correction": "benjamini_hochberg",
                "family": "all_fitness_diff_rows_including_zero_deltas",
            },
        ),
        "schema": 1,
        "candidate_functional_threshold": scientific["clamped"][
            "candidate_functional_threshold"
        ],
        "candidate_selection": candidate_selection_fingerprint(
            settings, context=context
        ),
        "files": {
            **{
                dataset_input_key(scientific["dataset"][key], path): _hash(path)
                for key, path in dataset.items()
            },
            **{str(path.relative_to(ROOT)): _hash(path) for path in paths},
        },
    }


def load_cached_candidate_edge_evidence(
    *, context: Any | None = None
) -> dict[str, Any]:
    """Read only a completed checkpoint, checking table coverage and hashes."""
    directory = cache_dir(context)
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, digest in provenance["artifacts"].items():
        if _hash(directory / name) != digest:
            raise ValueError(f"Candidate evidence artifact hash mismatch: {name}.")
    edges = pl.read_parquet(directory / "edges.parquet")
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    if (
        edges.height != diagnostics["edge_count"]
        or edges["edge_id"].n_unique() != edges.height
    ):
        raise ValueError("Candidate evidence coverage mismatch.")
    return {"edges": edges, "diagnostics": diagnostics, "provenance": provenance}


def load_or_compute_candidate_edge_evidence(
    *,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Atomically publish tables and provenance; appearance does not recompute."""
    settings = settings or load_settings(context=context)
    directory = cache_dir(context)
    correction = settings["analysis"].get(
        "candidate_edge_evidence",
        {
            "correction": "benjamini_hochberg",
            "family": "all_fitness_diff_rows_including_zero_deltas",
        },
    )
    if correction != {
        "correction": "benjamini_hochberg",
        "family": "all_fitness_diff_rows_including_zero_deltas",
    }:
        raise ValueError("Candidate evidence requires fixed full-family BH correction.")
    fingerprint = _fingerprint(settings, context=context)
    if not force and (directory / "provenance.json").is_file():
        try:
            cached = load_cached_candidate_edge_evidence(context=context)
            if cached["provenance"]["fingerprint"] == fingerprint:
                return cached
        except (OSError, ValueError, KeyError):
            pass
    scientific = settings["analysis"]
    wt = pl.read_parquet(
        dataset_input_path(scientific["dataset"]["fitness_wt"], context, root=ROOT)
    )
    diff = pl.read_parquet(
        dataset_input_path(scientific["dataset"]["fitness_diff"], context, root=ROOT)
    )
    network = Papkou2023Workflow(
        landscape="full", fitness_wt=wt, fitness_diff=diff
    ).build_network()
    prepared, _ = prepare_candidate_landscape(network, settings, context=context)
    result = build_candidate_edge_evidence(diff, prepared, wt)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".candidate-evidence-", dir=directory.parent))
    backup = directory.with_name(".candidate-evidence-previous")
    try:
        result["edges"].write_parquet(stage / "edges.parquet")
        result["edges"].write_csv(stage / "edges.csv")
        (stage / "diagnostics.json").write_text(
            json.dumps(result["diagnostics"], indent=2)
        )
        artifacts = {
            name: _hash(stage / name)
            for name in ("edges.parquet", "edges.csv", "diagnostics.json")
        }
        (stage / "provenance.json").write_text(
            json.dumps({"fingerprint": fingerprint, "artifacts": artifacts}, indent=2)
        )
        if backup.exists():
            shutil.rmtree(backup)
        if directory.exists():
            os.replace(directory, backup)
        try:
            os.replace(stage, directory)
        except BaseException:
            if backup.exists():
                os.replace(backup, directory)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_cached_candidate_edge_evidence(context=context)
