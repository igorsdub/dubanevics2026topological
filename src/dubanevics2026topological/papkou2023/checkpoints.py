# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""Safe, provenance-checked checkpoints for Papkou2023 workflows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Literal
import uuid

from dubanevics2026topological.network_analysis import (
    NetworkAnalysis,
    NetworkWorkflow,
    load_network_analysis,
    save_network_analysis,
)

TreeKindRequest = Literal["join", "split", "both"]
PROVENANCE_FILENAME = "papkou-provenance.json"
PENDING_FILENAME = ".analysis-pending"


def produce_or_load_papkou_analysis(
    workflow: NetworkWorkflow,
    output_dir: str | Path,
    *,
    tree_kind: TreeKindRequest = "both",
    force: bool = False,
) -> NetworkAnalysis:
    """Load a compatible Papkou checkpoint or publish a fresh complete generation.

    The doit task graph in dodo.py owns freshness decisions. This method checks the
    requested workflow, configuration, source declarations, checkpoint schema, and exact
    tree kinds.
    A forced update is staged in an immutable generation and made visible by an
    atomic replacement of the ``data`` symlink. Failed analysis leaves a pending
    marker that prevents later calls from treating the previous checkpoint as current.
    """
    out_path = Path(output_dir).resolve()
    data_path = out_path / "data"
    pending_path = out_path / PENDING_FILENAME
    requested_kinds = _requested_kinds(tree_kind)

    if not force and pending_path.exists():
        raise ValueError(
            f"Papkou2023 analysis update is pending in {out_path}; the previous checkpoint "
            "may be stale. Retry with a forced analysis rebuild before rendering."
        )
    if not force and data_path.exists():
        return _load_compatible_checkpoint(workflow, data_path, requested_kinds)

    out_path.mkdir(parents=True, exist_ok=True)
    pending_path.write_text("analysis update pending\n", encoding="utf-8")
    analysis = workflow.analyze(tree_kind=tree_kind)  # type: ignore[arg-type]

    generations_dir = out_path / ".data-generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    generation = generations_dir / f"generation-{uuid.uuid4().hex}"
    try:
        _validate_analysis(analysis, workflow, requested_kinds)
        save_network_analysis(analysis, generation)
        provenance = _provenance(workflow, requested_kinds)
        (generation / PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        _validate_generation(generation, workflow, requested_kinds)
        _publish_generation(data_path, generation)
    except Exception:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    pending_path.unlink(missing_ok=True)
    return analysis


def _requested_kinds(tree_kind: str) -> tuple[str, ...]:
    if tree_kind == "both":
        return ("join", "split")
    if tree_kind in {"join", "split"}:
        return (tree_kind,)
    raise ValueError(f"Unsupported Papkou2023 tree kind {tree_kind!r}; use split, join, or both.")


def _source_declarations(workflow: Any) -> list[str]:
    """Return stable source identifiers; the doit task graph tracks the source file contents."""
    if workflow.landscape == "article":
        source = workflow.article_path or Path("data/external/graph_largest_component.graphml")
        return [str(Path(source).expanduser().resolve())]

    if workflow.fitness_wt is not None or workflow.fitness_diff is not None:
        wt = (
            _memory_source("fitness_wt", workflow.fitness_wt)
            if workflow.fitness_wt is not None
            else "data/raw/papkou2023_fitness_wt.parquet"
        )
        diff = (
            _memory_source("fitness_diff", workflow.fitness_diff)
            if workflow.fitness_diff is not None
            else "data/raw/papkou2023_fitness_diff.parquet"
        )
        return [wt, diff]
    return [
        str(Path("data/raw/papkou2023_fitness_wt.parquet").resolve()),
        str(Path("data/raw/papkou2023_fitness_diff.parquet").resolve()),
    ]


def _memory_source(name: str, frame: Any) -> str:
    """Identify injected tables by content for compatibility, not freshness."""
    content = {
        "schema": [(column, str(dtype)) for column, dtype in frame.schema.items()],
        "rows": frame.to_dicts(),
    }
    encoded = json.dumps(content, sort_keys=True, default=str, allow_nan=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"memory:{name}:sha256={digest}"


def _provenance(workflow: Any, tree_kinds: tuple[str, ...]) -> dict[str, Any]:
    return {
        "checkpoint_kind": "papkou2023-analysis",
        "complete": True,
        "workflow_name": workflow.name,
        "config": workflow.config,
        "source_inputs": _source_declarations(workflow),
        "tree_kinds": list(tree_kinds),
    }


def _load_compatible_checkpoint(
    workflow: Any, data_path: Path, requested_kinds: tuple[str, ...]
) -> NetworkAnalysis:
    provenance_path = data_path / PROVENANCE_FILENAME
    if not provenance_path.exists():
        raise ValueError(
            f"Papkou2023 checkpoint at {data_path} has no provenance marker. "
            "It may be a legacy checkpoint; rebuild it with the forced analysis command."
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(
            f"Papkou2023 checkpoint provenance at {provenance_path} is unreadable; "
            "rebuild it with the forced analysis command."
        ) from err
    expected = _provenance(workflow, requested_kinds)
    if provenance != expected:
        raise ValueError(
            f"Papkou2023 checkpoint at {data_path} does not match the requested workflow, "
            "configuration, source inputs, or exact tree kinds. "
            "Rebuild it with the forced analysis command."
        )
    try:
        _validate_checkpoint_files(data_path, requested_kinds)
        analysis = load_network_analysis(data_path)
    except (OSError, ValueError) as err:
        raise ValueError(
            f"Papkou2023 checkpoint at {data_path} is incomplete or invalid: {err}. "
            "Rebuild it with the forced analysis command."
        ) from err
    _validate_analysis(analysis, workflow, requested_kinds)
    return analysis


def _validate_analysis(
    analysis: NetworkAnalysis, workflow: Any, requested_kinds: tuple[str, ...]
) -> None:
    actual_kinds = tuple(sorted(analysis.trees))
    if analysis.name != workflow.name or analysis.config != workflow.config:
        raise ValueError(
            "Papkou2023 analysis returned a different workflow configuration than requested."
        )
    if actual_kinds != tuple(sorted(requested_kinds)):
        raise ValueError(
            f"Papkou2023 analysis returned tree kinds {actual_kinds!r}; "
            f"expected exactly {requested_kinds!r}."
        )
    for kind, tree in analysis.trees.items():
        required = {
            "branch_member_table": tree.branch_member_table,
            "critical_node_table": tree.critical_node_table,
            "stop_marker_table": tree.stop_marker_table,
            "all_stop_marker_table": tree.all_stop_marker_table,
            "wild_type_marker_table": tree.wild_type_marker_table,
        }
        missing = [name for name, table in required.items() if table is None]
        if missing:
            raise ValueError(
                f"Papkou2023 {kind} analysis is missing required plotting data: "
                f"{', '.join(missing)}."
            )


def _validate_generation(
    generation: Path, workflow: Any, requested_kinds: tuple[str, ...]
) -> None:
    # Reuse the ordinary loader to verify every required table and graph exists.
    _validate_checkpoint_files(generation, requested_kinds)
    analysis = load_network_analysis(generation)
    _validate_analysis(analysis, workflow, requested_kinds)
    marker = json.loads((generation / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    if marker != _provenance(workflow, requested_kinds):
        raise ValueError("Staged Papkou2023 checkpoint provenance does not match the request.")


def _validate_checkpoint_files(data_path: Path, tree_kinds: tuple[str, ...]) -> None:
    required_suffixes = (
        "branch-members.parquet",
        "critical-nodes.parquet",
        "stop-markers.parquet",
        "all-stop-markers.parquet",
        "wild-type-markers.parquet",
    )
    missing = [
        f"{kind}-{suffix}"
        for kind in tree_kinds
        for suffix in required_suffixes
        if not (data_path / f"{kind}-{suffix}").is_file()
    ]
    if missing:
        raise ValueError(
            f"Papkou2023 checkpoint at {data_path} is missing required plotting tables: "
            f"{', '.join(missing)}. Rebuild it with the forced analysis command."
        )


def _publish_generation(data_path: Path, generation: Path) -> None:
    """Atomically point ``data`` at a complete generation, preserving old data."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(generation, data_path.parent)
    pending_link = data_path.parent / f".data-link-{uuid.uuid4().hex}"
    os.symlink(relative_target, pending_link)

    backup: Path | None = None
    try:
        if data_path.is_dir() and not data_path.is_symlink():
            backup = data_path.parent / f".data-legacy-{uuid.uuid4().hex}"
            os.replace(data_path, backup)
        os.replace(pending_link, data_path)
    except Exception:
        pending_link.unlink(missing_ok=True)
        if backup is not None and backup.exists() and not data_path.exists():
            os.replace(backup, data_path)
        raise
