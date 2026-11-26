"""Neighbour-based peak uncertainty tables and atomic cache management."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import topographer

from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from ..landscape.selected_landscape import PreparedLandscape, prepare_clamped_landscape
from ..landscape.selection import (
    candidate_selection_fingerprint,
    prepare_candidate_landscape,
)
from . import (
    ROOT,
    _write_json_atomic,
    dataset_input_path,
    dedupe_by_resolved,
    get_run_context,
    hash_sources,
    load_settings,
    require_files,
    split_declared_inputs,
)
from .checkpoint import checkpoint_lock, publish_checkpoint, recover_checkpoint
from .edge_evidence import global_contrasts as _global_contrasts
from .produce_or_load import Stage, stage_fingerprint

LANDSCAPE_KEYS = ("study_clamped", "candidate_clamped")
CACHE_SCHEMA = 2
# The command the checkpoint helpers name when a rollback or recovery fails.
RETRY_HINT = "just peak-analysis"
_TABLE_SUFFIXES = ("scatter", "contrasts", "peaks", "decay")
_RAW_SCALAR = "raw_fitness"
_EDGE_SELECTION_RULE = "at_least_one_endpoint_raw_fitness_ge_threshold"


def cache_paths(context: Any | None = None) -> tuple[Path, Path]:
    """The checkpoint directory and pending marker under the context's output root.

    A context-less call resolves a fresh context now, so nothing is frozen at import.
    """
    analysis = Path((context or get_run_context()).output_root) / "analysis"
    return analysis / "peak_uncertainty", analysis / ".peak-uncertainty-pending"


def build_peak_uncertainty(
    wt: pl.DataFrame,
    diff: pl.DataFrame,
    landscapes: dict[str, PreparedLandscape],
) -> dict[str, Any]:
    """Build scatter, neighbour contrasts, peak scores, and exact decay curves.

    The supplied pairwise p-values receive one BH correction over the complete
    input table. Graph membership and local-maximum status come only from each
    prepared landscape's actual undirected adjacency and raw vertex fitness.
    """
    _require_columns(wt, ("dna_from", "dna_to", "aa_to", "s", "se"), "fitness_wt")
    _require_columns(
        diff,
        ("dna_from", "dna_to", "aa_from", "aa_to", "s", "se", "pval"),
        "fitness_diff",
    )
    if set(landscapes) != set(LANDSCAPE_KEYS):
        raise ValueError(
            f"landscapes must contain exactly {LANDSCAPE_KEYS!r}; got {sorted(landscapes)!r}."
        )
    raw_measurements = {
        str(row["dna_to"]): _finite_number(row["s"], "WT-relative raw fitness")
        for row in wt.to_dicts()
        if row.get("dna_to") is not None
    }
    if len(raw_measurements) != wt.height:
        raise ValueError("fitness_wt has missing or duplicate measured DNA identities.")
    wt_sequences = wt.get_column("dna_from").drop_nulls().unique().to_list()
    if len(wt_sequences) != 1:
        raise ValueError(
            "fitness_wt must identify exactly one wild-type dna_from sequence."
        )
    raw_measurements[str(wt_sequences[0])] = 0.0
    wt_sequence = str(wt_sequences[0])
    wt_rows = wt.to_dicts()
    fitness_wt_by_id: dict[str, dict[str, Any]] = {}
    for row in wt_rows:
        sequence = row.get("dna_to")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("Every measured fitness_wt row must have a DNA identity.")
        if sequence == wt_sequence:
            continue
        if sequence in fitness_wt_by_id:
            raise ValueError(
                f"Duplicate measured fitness row for DNA sequence {sequence!r}."
            )
        fitness_wt_by_id[sequence] = row
        _finite_number(row.get("s"), f"fitness_wt raw fitness for {sequence}")
        _finite_number(row.get("se"), f"fitness_wt standard error for {sequence}")
        if float(row["se"]) < 0:
            raise ValueError(f"Negative fitness_wt standard error for {sequence!r}.")

    landscape_graphs = {}
    needed_edges: set[tuple[str, str]] = set()
    for key in LANDSCAPE_KEYS:
        adj, vertices = _landscape_adjacency(landscapes[key])
        for identity, payload in vertices.items():
            if identity not in raw_measurements:
                raise ValueError(
                    f"Prepared vertex {identity!r} has no WT-relative measurement."
                )
            if not math.isclose(
                payload["raw_fitness"],
                raw_measurements[identity],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Prepared raw fitness for {identity!r} disagrees with fitness_wt."
                )
        for identity, neighbors in adj.items():
            needed_edges.update(_edge_key(identity, neighbor) for neighbor in neighbors)
        landscape_graphs[key] = (adj, vertices)
    contrast_by_edge, disagreement, diff_se_range = _global_contrasts(
        diff, raw_measurements, needed_edges
    )

    result_tables: dict[str, pl.DataFrame] = {}
    full_domain = _full_fitness_domain(wt, wt_sequence)
    diagnostics: dict[str, Any] = {
        "analysis_scalar": _RAW_SCALAR,
        "edge_selection_rule": _EDGE_SELECTION_RULE,
        "wild_type_sequence": wt_sequence,
        "fitness_domain": full_domain,
        "fitness_domain_including_wt_zero": full_domain,
        "thresholds": {key: float(landscapes[key].threshold) for key in LANDSCAPE_KEYS},
        "peak_counts": {},
        "wild_type_scatter_excluded": True,
        "wild_type_scatter_excluded_count": sum(
            str(row.get("dna_to")) == wt_sequence for row in wt_rows
        ),
        "fitness_wt_input_row_count": wt.height,
        "fitness_wt_scatter_row_count": len(fitness_wt_by_id),
        "fitness_wt_standard_error_range": _range(
            [float(row["se"]) for row in fitness_wt_by_id.values()]
        ),
        "fitness_diff_input_row_count": diff.height,
        "global_bh_family_size": diff.height,
        "fitness_diff_standard_error_range": diff_se_range,
        "pairwise_raw_fitness_sign_disagreement_count": len(disagreement),
        "pairwise_raw_fitness_sign_disagreement_table": "pairwise_sign_disagreements",
        "landscapes": {},
    }
    result_tables["pairwise_sign_disagreements"] = _disagreement_frame(disagreement)

    for key in LANDSCAPE_KEYS:
        prepared = landscapes[key]
        adj, vertices = landscape_graphs[key]
        raw_by_id = {
            identity: data["raw_fitness"] for identity, data in vertices.items()
        }
        peaks = {
            identity
            for identity, value in raw_by_id.items()
            if all(raw_by_id[neighbor] <= value for neighbor in adj[identity])
        }
        scatter_rows = []
        for identity, row in fitness_wt_by_id.items():
            scatter_rows.append(
                {
                    "dna_sequence": identity,
                    "aa_sequence": str(row.get("aa_to") or ""),
                    "raw_fitness": float(row["s"]),
                    "standard_error": float(row["se"]),
                    "is_peak": identity in peaks,
                }
            )
        contrast_rows = []
        peak_rows = []
        for peak in sorted(peaks):
            neighbors = sorted(adj[peak])
            if not neighbors:
                raise ValueError(f"Peak {peak!r} has an empty neighbourhood in {key}.")
            neighbor_q = []
            for neighbor in neighbors:
                edge_key = _edge_key(peak, neighbor)
                test = contrast_by_edge.get(edge_key)
                if test is None:
                    raise ValueError(
                        f"No pairwise test comparison exists for prepared edge {peak!r}—{neighbor!r}."
                    )
                orientation = 1.0 if test["reported_dna_from"] == peak else -1.0
                pair_delta = orientation * test["reported_delta"]
                statistic = orientation * test["test_statistic"]
                neighbor_q.append(test["q_value"])
                contrast_rows.append(
                    {
                        "dna_sequence": peak,
                        "neighbor_sequence": neighbor,
                        "raw_fitness": raw_by_id[peak],
                        "neighbor_fitness": raw_by_id[neighbor],
                        "pairwise_delta": pair_delta,
                        "pairwise_standard_error": test["pairwise_standard_error"],
                        "test_statistic": statistic,
                        "p_value": test["p_value"],
                        "q_value": test["q_value"],
                        "contrast_id": test["contrast_id"],
                        "reported_dna_from": test["reported_dna_from"],
                        "reported_dna_to": test["reported_dna_to"],
                        "reported_aa_from": test["reported_aa_from"],
                        "reported_aa_to": test["reported_aa_to"],
                        "reported_delta": test["reported_delta"],
                    }
                )
            max_q = max(neighbor_q)
            peak_rows.append(
                {
                    "dna_sequence": peak,
                    "raw_fitness": raw_by_id[peak],
                    "max_neighbor_q": max_q,
                    "min_neighbor_score": _neighbor_score(max_q),
                    "neighbor_count": len(neighbors),
                }
            )
        scatter = _scatter_frame(scatter_rows)
        contrast_frame = _contrast_frame(contrast_rows)
        peak_frame = _peak_frame(peak_rows)
        decay = _decay_frame(peak_rows)
        for suffix, frame in zip(
            _TABLE_SUFFIXES, (scatter, contrast_frame, peak_frame, decay)
        ):
            result_tables[f"{key}_{suffix}"] = frame
        diagnostics["landscapes"][key] = {
            "threshold": float(prepared.threshold),
            "selected_vertex_count": len(vertices),
            "selected_edge_count": sum(map(len, adj.values())) // 2,
            "peak_count": len(peaks),
            "scatter_measured_count": scatter.height,
            "scatter_peak_count": sum(bool(row["is_peak"]) for row in scatter_rows),
            "neighbour_comparison_count": contrast_frame.height,
            "fitness_domain": _range(list(raw_by_id.values())),
        }
        diagnostics["peak_counts"][key] = len(peaks)
    return {"tables": result_tables, "diagnostics": diagnostics}


def _load_peak_checkpoint(
    directory: Path, *, allow_pending: bool = False, context: Any | None = None
) -> dict[str, Any]:
    """Validate and load a checkpoint at ``directory`` without acquiring a lock."""
    if cache_paths(context)[1].exists() and not allow_pending:
        raise RuntimeError(
            "Peak uncertainty refresh is pending; rerun `just peak-analysis`."
        )
    provenance_path = directory / "provenance.json"
    diagnostics_path = directory / "diagnostics.json"
    if not provenance_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError(
            "No peak uncertainty checkpoint exists; run `just peak-analysis`."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != CACHE_SCHEMA
        or provenance.get("analysis_scalar") != _RAW_SCALAR
        or provenance.get("published_edge_selection_rule") != _EDGE_SELECTION_RULE
        or provenance.get("edge_selection_rule")
        != provenance.get("candidate_edge_selection_rule")
        or provenance.get("bh_family") != "all_fitness_diff_rows_including_zero_deltas"
    ):
        raise ValueError(
            "Peak uncertainty checkpoint has an incompatible scientific contract."
        )
    expected = tuple(
        f"{key}_{suffix}" for key in LANDSCAPE_KEYS for suffix in _TABLE_SUFFIXES
    ) + ("pairwise_sign_disagreements",)
    tables: dict[str, pl.DataFrame] = {}
    for name in expected:
        path = directory / "tables" / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Peak uncertainty checkpoint is incomplete: {path.name}."
            )
        tables[name] = pl.read_parquet(path)
    return {
        "tables": tables,
        "diagnostics": json.loads(diagnostics_path.read_text(encoding="utf-8")),
        "provenance": provenance,
    }


def load_cached_peak_uncertainty(*, context: Any | None = None) -> dict[str, Any]:
    """Load a completed checkpoint while excluding concurrent publication."""
    cache_dir, _ = cache_paths(context)
    with checkpoint_lock(cache_dir, "peak uncertainty"):
        return _load_peak_checkpoint(cache_dir, context=context)


def load_or_compute_peak_uncertainty(
    *,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Reuse or atomically refresh peak uncertainty products without building trees."""
    cache_dir, _ = cache_paths(context)
    with checkpoint_lock(cache_dir, "peak uncertainty"):
        return _load_or_compute_peak_uncertainty_locked(
            force=force, settings=settings, context=context
        )


def _load_or_compute_peak_uncertainty_locked(
    *,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings(context=context)
    cache_dir, pending_path = cache_paths(context)
    fingerprint = _fingerprint(settings, context=context)
    recovered = recover_checkpoint(
        cache_dir,
        pending_path,
        "peak uncertainty",
        lambda path: _load_peak_checkpoint(path, allow_pending=True, context=context),
        expected_fingerprint=fingerprint,
        fingerprint_of=lambda value: value["provenance"].get("fingerprint"),
        retry_hint=RETRY_HINT,
    )
    if (
        not force
        and recovered is not None
        and recovered["provenance"].get("fingerprint") == fingerprint
    ):
        return recovered

    analysis_dir = cache_dir.parent
    analysis_dir.mkdir(parents=True, exist_ok=True)
    pending_path.write_text("peak uncertainty refresh pending\n", encoding="utf-8")
    stage = Path(tempfile.mkdtemp(prefix=".peak-uncertainty-stage-", dir=analysis_dir))
    try:
        scientific = settings["analysis"]
        dataset = scientific["dataset"]
        clamped = scientific["clamped"]
        wt = pl.read_parquet(
            dataset_input_path(dataset["fitness_wt"], context, root=ROOT)
        )
        diff = pl.read_parquet(
            dataset_input_path(dataset["fitness_diff"], context, root=ROOT)
        )
        workflow = Papkou2023Workflow(
            landscape="full", fitness_wt=wt, fitness_diff=diff
        )
        print(
            "Building the full Papkou network for peak-neighbour analysis.", flush=True
        )
        network = workflow.build_network()
        print(
            f"Built the full Papkou network ({network.vertex_count} vertices, "
            f"{network.edge_count} edges); preparing two threshold landscapes.",
            flush=True,
        )
        candidate_landscape, selection = prepare_candidate_landscape(
            network, settings, context=context
        )
        candidate_threshold = selection.threshold
        landscapes = {
            "study_clamped": prepare_clamped_landscape(
                network, float(clamped["published_functional_threshold"])
            ),
            "candidate_clamped": candidate_landscape,
        }
        products = build_peak_uncertainty(wt, diff, landscapes)
        print("Computed global BH contrasts and threshold peak tables.", flush=True)
        products["diagnostics"]["clamped_thresholds"] = {
            "study_clamped": float(clamped["published_functional_threshold"]),
            "candidate_clamped": candidate_threshold.threshold,
        }
        table_dir = stage / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in products["tables"].items():
            frame.write_parquet(table_dir / f"{name}.parquet")
        _write_json_atomic(stage / "diagnostics.json", products["diagnostics"])
        provenance = {
            "schema": CACHE_SCHEMA,
            "analysis_scalar": _RAW_SCALAR,
            # The Candidate graph's rule; the Published landscape keeps its own.
            "edge_selection_rule": candidate_landscape.diagnostics[
                "edge_selection_rule"
            ],
            "candidate_edge_selection_rule": candidate_landscape.diagnostics[
                "edge_selection_rule"
            ],
            "published_edge_selection_rule": _EDGE_SELECTION_RULE,
            "bh_family": "all_fitness_diff_rows_including_zero_deltas",
            "fingerprint": fingerprint,
            "thresholds": products["diagnostics"]["clamped_thresholds"],
            "global_bh_family_size": products["diagnostics"]["global_bh_family_size"],
            "tables": sorted(products["tables"]),
        }
        _write_json_atomic(stage / "provenance.json", provenance)
        return publish_checkpoint(
            stage,
            cache_dir,
            pending_path,
            "peak uncertainty",
            lambda path: _load_peak_checkpoint(
                path, allow_pending=True, context=context
            ),
            expected_fingerprint=fingerprint,
            fingerprint_of=lambda value: value["provenance"].get("fingerprint"),
            retry_hint=RETRY_HINT,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _landscape_adjacency(
    prepared: PreparedLandscape,
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    graph = prepared.network.graph
    vertices: dict[str, dict[str, Any]] = {}
    index_to_id: dict[int, str] = {}
    for index in graph.node_indices():
        payload = dict(graph[index])
        identity = payload.get("dna_sequence")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Prepared graph vertex is missing stable DNA identity.")
        if identity in vertices:
            raise ValueError(f"Prepared graph has duplicate DNA identity {identity!r}.")
        value = _finite_number(
            payload.get("raw_fitness"), f"raw fitness for {identity}"
        )
        vertices[identity] = {
            **payload,
            "raw_fitness": value,
        }
        index_to_id[int(index)] = identity
    adjacency = {identity: set() for identity in vertices}
    for left_index, right_index in graph.edge_list():
        left, right = index_to_id[int(left_index)], index_to_id[int(right_index)]
        if left == right:
            raise ValueError(f"Self-loop found in prepared graph at {left!r}.")
        if right in adjacency[left]:
            raise ValueError(f"Duplicate ambiguous prepared edge {left!r}—{right!r}.")
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency, vertices


def _full_fitness_domain(wt: pl.DataFrame, wt_sequence: str) -> list[float]:
    values = [0.0]
    for row in wt.select("dna_to", "s").iter_rows(named=True):
        if row["dna_to"] != wt_sequence:
            values.append(_finite_number(row["s"], "WT-relative raw fitness"))
    return _range(values)


def _disagreement_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "contrast_id": pl.String,
            "dna_from": pl.String,
            "dna_to": pl.String,
            "pairwise_delta": pl.Float64,
            "raw_fitness_delta": pl.Float64,
            "pairwise_sign": pl.Int8,
            "raw_fitness_sign": pl.Int8,
            "p_value": pl.Float64,
            "q_value": pl.Float64,
        },
    )


def _scatter_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "dna_sequence": pl.String,
            "aa_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "standard_error": pl.Float64,
            "is_peak": pl.Boolean,
        },
    )


def _contrast_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "dna_sequence": pl.String,
            "neighbor_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "neighbor_fitness": pl.Float64,
            "pairwise_delta": pl.Float64,
            "pairwise_standard_error": pl.Float64,
            "test_statistic": pl.Float64,
            "p_value": pl.Float64,
            "q_value": pl.Float64,
            "contrast_id": pl.String,
            "reported_dna_from": pl.String,
            "reported_dna_to": pl.String,
            "reported_aa_from": pl.String,
            "reported_aa_to": pl.String,
            "reported_delta": pl.Float64,
        },
    )


def _peak_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "dna_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "max_neighbor_q": pl.Float64,
            "min_neighbor_score": pl.Float64,
            "neighbor_count": pl.Int64,
        },
    )


def _decay_frame(peaks: list[dict[str, Any]]) -> pl.DataFrame:
    points = {0.0, 1.0}
    for row in peaks:
        score = float(row["min_neighbor_score"])
        points.add(score)
        following = float(np.nextafter(score, math.inf))
        if following <= 1.0:
            points.add(following)
    thresholds = sorted(points)
    return pl.DataFrame(
        {
            "threshold": thresholds,
            "retained_peak_count": [
                sum(float(row["min_neighbor_score"]) >= threshold for row in peaks)
                for threshold in thresholds
            ],
        },
        schema={"threshold": pl.Float64, "retained_peak_count": pl.Int64},
    )


def _neighbor_score(q_value: float) -> float:
    """Preserve the strict upper endpoint for positive, extremely small q-values.

    Binary subtraction can round ``1 - q`` to exactly one when q is positive.
    The preceding representable score preserves the correct selection for every
    representable threshold: such a comparison survives immediately below one but
    fails at one. The original q-value remains unchanged in all exported tables.
    """
    score = 1.0 - q_value
    if q_value > 0.0 and score == 1.0:
        return math.nextafter(1.0, 0.0)
    return score


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")


def _finite_number(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric data.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite numeric data.") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite numeric data.")
    return numeric


def _range(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def peak_stage(settings: dict[str, Any], *, context: Any | None = None) -> Stage:
    """Declare the peak uncertainty checkpoint: scientific settings, inputs and code.

    The settings are the parsed config values the checkpoint reads (the dataset
    section and the two thresholds) and the hashes of its code, not the bytes of
    ``config/analysis.toml``, so ``n_jobs`` and ``chunk_size`` never reach the
    fingerprint. ``project_stage`` strips any scheduling key that is declared.
    """
    # Imported here because ``stages`` itself imports the helper in this package.
    import dubanevics2026topological as _package

    from ..stages import project_stage

    scientific = settings["analysis"]
    dataset = scientific["dataset"]
    package_root = Path(topographer.__file__).resolve().parent
    landscape_root = Path(__file__).resolve().parents[1] / "landscape"
    named_inputs = {
        key: dataset_input_path(dataset[key], context, root=ROOT)
        for key in ("fitness_wt", "fitness_diff")
    }
    code = dedupe_by_resolved(
        [
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            Path(__file__).resolve(),
            landscape_root / "selection.py",
            Path(__file__).resolve().parents[1]
            / "threshold"
            / "functional_threshold.py",
            landscape_root / "selected_landscape.py",
            landscape_root / "full_population.py",
            Path(__file__).with_name("edge_evidence.py").resolve(),
            Path(_package.__file__).resolve().parent / "network_analysis.py",
            package_root / "biology/_sequences.py",
            *sorted(
                (Path(_package.__file__).resolve().parent / "papkou2023").glob("*.py")
            ),
            *sorted((package_root / "graph").glob("*.py")),
            *sorted((package_root / "statistics").glob("*.py")),
        ]
    )
    require_files([*named_inputs.values(), *code], "peak uncertainty inputs")
    # Only now hash an input outside the repository: a missing one was reported above.
    inputs, outside_inputs = split_declared_inputs(named_inputs, ROOT)
    return project_stage(
        "peak-uncertainty",
        settings={
            "schema": CACHE_SCHEMA,
            "analysis_scalar": _RAW_SCALAR,
            "edge_selection_rule": _EDGE_SELECTION_RULE,
            "bh_family": "all_fitness_diff_rows_including_zero_deltas",
            "config": {
                "dataset": dataset,
                "published_functional_threshold": scientific["clamped"][
                    "published_functional_threshold"
                ],
                "candidate_functional_threshold": scientific["clamped"][
                    "candidate_functional_threshold"
                ],
                "candidate_selection": candidate_selection_fingerprint(
                    settings, context=context
                ),
            },
            **({"outside_input_sha256": outside_inputs} if outside_inputs else {}),
            "sources": hash_sources(code),
            "topographer_version": getattr(topographer, "__version__", None),
        },
        inputs=inputs,
        root=ROOT,
    )


def _fingerprint(settings: dict[str, Any], *, context: Any | None = None) -> str:
    return stage_fingerprint(peak_stage(settings, context=context))
