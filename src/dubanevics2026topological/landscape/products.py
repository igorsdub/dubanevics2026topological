"""Publish selected landscapes and nominal trees as labelled result products."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dubanevics2026topological.results import (
    ALLOWED_KEYS,
    Manifest,
    RunContext,
    compute_scientific_hash,
    format_label,
    read_manifest,
    write_product_readme,
    write_root_readme,
)
from dubanevics2026topological.tables import _public_table

from .selected_landscape import PreparedLandscape

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_IDENTITY_KEYS = frozenset(
    {
        *ALLOWED_KEYS,
        "parameters",
        "landscape_selection",
        "full_graph_counts",
        "scientific_identity",
        "data_provenance",
        "parents",
        "scientific_id_sha256",
        "product_type",
        "artifacts",
        "run_fields",
        "code_provenance",
        "environment",
        "created_at",
        "evidence_availability",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_records(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(directory).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _run_fields(context: RunContext) -> tuple[dict[str, Any], dict[str, Any]]:
    fields: dict[str, Any] = {}
    params: dict[str, Any] = {}
    if context.run_id is not None:
        fields = {"host": context.host, "run": context.run_id}
        params = dict(fields)
    return fields, params


def _code_provenance(paths: Sequence[Path]) -> dict[str, str]:
    return {path.as_posix(): _sha256(path) for path in paths}


def _environment(context: RunContext) -> dict[str, Any]:
    lockfile = context.root / "uv.lock"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "uv_lock_sha256": _sha256(lockfile) if lockfile.is_file() else None,
    }


def _existing_product(target: Path, scientific_id: str) -> Path | None:
    if not target.exists():
        return None
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Product path exists without a manifest; refusing to overwrite {target}."
        )
    from dubanevics2026topological.results import read_manifest

    manifest = read_manifest(manifest_path)
    if manifest.scientific_id_sha256 != scientific_id:
        raise FileExistsError(
            f"Product label collision at {target}: the recorded scientific identity differs."
        )
    for relative, record in manifest.artifacts.items():
        path = target / relative
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(
                f"Existing product artifact {relative!r} is missing or has changed."
            )
    return target


def _publish(
    product_type: str,
    base_params: Mapping[str, Any],
    scientific_params: Mapping[str, Any],
    *,
    context: RunContext,
    input_data_sha256s: Mapping[str, str],
    parents: Sequence[Mapping[str, str]],
    code_paths: Sequence[Path],
    write_artifacts: Any,
) -> tuple[Path, Manifest]:
    parent_ids = [str(parent["scientific_id_sha256"]) for parent in parents]
    full_hash, short_hash = compute_scientific_hash(
        product_type,
        scientific_params,
        parent_scientific_ids=parent_ids,
        input_data_sha256s=input_data_sha256s,
    )
    run_fields, run_params = _run_fields(context)
    label_params = {**base_params, **run_params, "hash": short_hash}
    label = format_label(product_type, label_params)
    product_root = context.output_root / product_type
    target = product_root / label
    existing = _existing_product(target, full_hash)

    product_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{label}.", dir=product_root))
    try:
        write_artifacts(staging)
        artifacts = _artifact_records(staging)
        if existing is not None:
            manifest = read_manifest(existing / "manifest.json")
            if artifacts != manifest.artifacts:
                raise ValueError(
                    f"Rebuilt product {label!r} has the same scientific identity but "
                    "different artifact bytes; refusing to replace the immutable product."
                )
            shutil.rmtree(staging)
            write_product_readme(product_root, product_type)
            write_root_readme(context.output_root)
            return existing, manifest
        manifest = Manifest(
            product_type=product_type,
            scientific_identity=dict(scientific_params),
            scientific_id_sha256=full_hash,
            parents=[dict(parent) for parent in parents],
            data_provenance=dict(input_data_sha256s),
            artifacts=artifacts,
            run_fields=run_fields,
            code_provenance={"source_sha256s": _code_provenance(code_paths)},
            environment=_environment(context),
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            evidence_availability={
                "checkpoint_present": False,
                "distances_present": False,
                "chunks_present": False,
            },
        )
        manifest.write(staging / "manifest.json", check_collision=False)
        try:
            os.rename(staging, target)
        except FileExistsError:
            existing = _existing_product(target, full_hash)
            if existing is None:
                raise
            manifest = read_manifest(existing / "manifest.json")
            if artifacts != manifest.artifacts:
                raise ValueError(
                    f"Concurrent publication of {label!r} produced different artifact bytes."
                )
            shutil.rmtree(staging)
        write_product_readme(product_root, product_type)
        write_root_readme(context.output_root)
        return target, manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def save_selected_products(
    landscape: PreparedLandscape,
    nominal_tree: Any,
    *,
    parameters: Mapping[str, Any],
    scientific_parameters: Mapping[str, Any] | None = None,
    context: RunContext | None = None,
    input_data_sha256s: Mapping[str, str],
    parents: Sequence[Mapping[str, str]],
    code_paths: Sequence[Path] | None = None,
) -> tuple[Path, Path]:
    """Save a labelled ``landscape`` product and its nominal ``tree`` product.

    ``parameters`` contains only ADR 0016 label fields in pipeline order (for
    example ``{"thr": "reflected", "fdr": "by", "q": 0.05}``). The same
    scientific choices and exact input checksums enter each manifest. Source-code
    hashes are recorded only under ``code_provenance`` and never enter either
    scientific identity or its digest. ``input_data_sha256s`` is mandatory and
    non-empty; ``parents`` is mandatory but may be empty when source inputs have no
    saved parent product to reference.
    """
    context = context or RunContext()
    if "graph" not in parameters and not {"thr", "fdr"} <= set(parameters):
        raise ValueError("Selected landscape products need graph=full or thr and fdr.")
    inputs = _require_input_hashes(input_data_sha256s)
    parents = _require_parents(parents)
    extra = dict(scientific_parameters or {})
    collisions = _RESERVED_IDENTITY_KEYS & extra.keys()
    if collisions:
        raise ValueError(
            "scientific_parameters cannot override reserved identity fields: "
            f"{sorted(collisions)}."
        )
    code_paths = (
        tuple(code_paths)
        if code_paths is not None
        else (
            Path(__file__).resolve(),
            Path(__file__).with_name("selection.py").resolve(),
            Path(__file__).with_name("selected_landscape.py").resolve(),
        )
    )
    landscape_identity = _scientific_identity(landscape, parameters, extra)
    if parameters.get("graph") == "full":
        full_counts = _validate_full_graph(landscape)
        landscape_identity["full_graph_counts"] = full_counts
    landscape_path, landscape_manifest = _publish(
        "landscape",
        parameters,
        landscape_identity,
        context=context,
        input_data_sha256s=inputs,
        parents=parents,
        code_paths=code_paths,
        write_artifacts=lambda directory: _write_landscape(directory, landscape),
    )

    tree_params = {**parameters, "tree": "nominal"}
    tree_identity = _scientific_identity(landscape, tree_params, extra)
    if parameters.get("graph") == "full":
        tree_identity["full_graph_counts"] = full_counts
    tree_parent = {
        "product_type": "landscape",
        "label": landscape_path.name,
        "scientific_id_sha256": landscape_manifest.scientific_id_sha256,
    }
    tree_path, _ = _publish(
        "tree",
        tree_params,
        tree_identity,
        context=context,
        input_data_sha256s=inputs,
        parents=(tree_parent,),
        code_paths=code_paths,
        write_artifacts=lambda directory: _write_tree(directory, nominal_tree),
    )
    return landscape_path, tree_path


def save_landscape_variants(
    variants: Mapping[
        str,
        tuple[
            PreparedLandscape,
            Any,
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, str],
            Sequence[Mapping[str, str]],
        ],
    ],
    *,
    context: RunContext,
    code_paths: Sequence[Path] | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Publish named graph/tree variants using the same per-graph writer.

    Each value is ``(prepared_landscape, nominal_tree, label_parameters,
    scientific_parameters, input_sha256s, parents)``. This deliberately remains a
    thin loop: selection and threshold methods are resolved by their owning modules
    before publication.
    """
    return {
        name: save_selected_products(
            prepared,
            tree,
            parameters=parameters,
            scientific_parameters=scientific_parameters,
            context=context,
            input_data_sha256s=input_data_sha256s,
            parents=parents,
            code_paths=code_paths,
        )
        for name, (
            prepared,
            tree,
            parameters,
            scientific_parameters,
            input_data_sha256s,
            parents,
        ) in variants.items()
    }


def _require_input_hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not values:
        raise ValueError("At least one input-data SHA-256 is required.")
    result = dict(values)
    invalid = [key for key, digest in result.items() if not _SHA256.fullmatch(digest)]
    if invalid:
        raise ValueError(f"Input-data SHA-256 values are invalid for {invalid}.")
    return result


def _require_parents(values: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    parents = tuple(dict(parent) for parent in values)
    for parent in parents:
        missing = {
            "product_type",
            "label",
            "scientific_id_sha256",
        } - parent.keys()
        if missing or not _SHA256.fullmatch(parent.get("scientific_id_sha256", "")):
            raise ValueError(
                "Each parent needs product_type, label, and a 64-hex "
                f"scientific_id_sha256; missing {sorted(missing)}."
            )
    return parents


def _scientific_identity(
    landscape: PreparedLandscape,
    parameters: Mapping[str, Any],
    scientific_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    is_full = parameters.get("graph") == "full"
    if is_full:
        if set(parameters) not in ({"graph"}, {"graph", "tree"}):
            raise ValueError("Full graph labels contain graph=full and optional tree.")
    else:
        if parameters.get("thr") not in {"published", "reflected", "mixture"}:
            raise ValueError("Selected landscape needs a supported thr label.")
        if parameters.get("fdr") not in {"none", "by"}:
            raise ValueError("Selected landscape needs fdr=none or fdr=by.")
        if (parameters["fdr"] == "by") != ("q" in parameters):
            raise ValueError("A BY landscape requires q; an unadjusted one forbids it.")
    collisions = _RESERVED_IDENTITY_KEYS & scientific_parameters.keys()
    if collisions:
        raise ValueError(
            "scientific_parameters cannot override reserved identity fields: "
            f"{sorted(collisions)}."
        )
    identity = {
        **dict(parameters),
        "landscape_selection": {
            "graph_kind": "full" if is_full else "selected",
            "threshold_kind": None if is_full else parameters["thr"],
            "threshold": None if is_full else float(landscape.threshold),
            "edge_selection_rule": landscape.diagnostics["edge_selection_rule"],
            "functional_vertex_count": (
                landscape.diagnostics["input_vertex_count"]
                if is_full
                else landscape.diagnostics["functional_vertex_count"]
            ),
            "functional_set_sha256": landscape.diagnostics[
                "functional_set_sha256"
            ],
        },
        **dict(scientific_parameters),
    }
    return identity


def _validate_full_graph(landscape: PreparedLandscape) -> dict[str, Any]:
    diagnostics = landscape.diagnostics
    counts = {
        "input_vertices": int(diagnostics["input_vertex_count"]),
        "selected_vertices": int(diagnostics["selected_component_vertex_count"]),
        "input_edges": int(diagnostics["input_edge_count"]),
        "selected_edges": int(diagnostics["selected_component_edge_count"]),
        "excluded_vertices": int(diagnostics["excluded_vertex_count"]),
        "components": int(diagnostics["component_count"]),
    }
    if (
        counts["components"] != 1
        or counts["excluded_vertices"] != 0
        or counts["input_vertices"] != counts["selected_vertices"]
        or counts["input_edges"] != counts["selected_edges"]
    ):
        raise ValueError(
            "graph=full requires every input vertex and edge to be retained in one "
            f"connected component; observed {counts}."
        )
    return {
        **counts,
        "component_handling": "all_vertices_and_edges_retained_in_one_component",
    }


def _write_landscape(directory: Path, landscape: PreparedLandscape) -> None:
    graph = landscape.network.graph
    node_rows = [dict(graph[index]) for index in graph.node_indices()]
    pl.DataFrame(node_rows).sort("dna_sequence").write_parquet(
        directory / "vertices.parquet"
    )
    index_to_id = {
        index: str(graph[index]["dna_sequence"]) for index in graph.node_indices()
    }
    edge_rows = []
    for index in graph.edge_indices():
        left, right = graph.get_edge_endpoints_by_index(index)
        edge_rows.append(
            {
                "source_dna_sequence": index_to_id[left],
                "target_dna_sequence": index_to_id[right],
                **dict(graph.get_edge_data_by_index(index) or {}),
            }
        )
    pl.DataFrame(edge_rows).sort(
        ["source_dna_sequence", "target_dna_sequence"]
    ).write_parquet(directory / "edges.parquet")
    landscape.components.write_parquet(directory / "components.parquet")
    landscape.excluded_vertices.write_parquet(directory / "excluded_vertices.parquet")
    landscape.peaks.write_parquet(directory / "peaks.parquet")
    (directory / "diagnostics.json").write_text(
        json.dumps(landscape.diagnostics, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_tree(directory: Path, analysis: Any) -> None:
    tree = analysis.trees["split"]
    for name, frame in (
        ("critical_nodes", tree.critical_node_table),
        ("persistence_pairs", tree.persistence_table),
        ("branches", tree.branch_table),
        ("branch_members", tree.branch_member_table),
    ):
        _public_table(
            {
                "critical_nodes": "critical_node_table",
                "persistence_pairs": "persistence_table",
                "branches": "branch_table",
                "branch_members": "branch_member_table",
            }[name],
            frame,
            analysis,
        ).write_parquet(directory / f"{name}.parquet")
