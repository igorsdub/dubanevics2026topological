"""Stable-identity supporting tables for the publication analysis."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import OUTPUT, _write_json_atomic

TABLE_DIR = OUTPUT / "tables"
# Identity shelf. The plan's `_by_id` name is `canonical/` (T3 lock).
SHELF_DIR = TABLE_DIR / "canonical"


def _identity_maps(analysis: Any) -> tuple[dict[int, str], dict[int, str]]:
    tree = analysis.trees["split"].tree
    tree_ids: dict[int, str] = {}
    vertex_ids: dict[int, str] = {}
    for index in tree.node_indices():
        data = dict(tree[index])
        identity = next(
            (
                data[key]
                for key in ("dna_sequence", "vertex_id", "id")
                if data.get(key) is not None
            ),
            None,
        )
        if identity is not None:
            tree_ids[int(index)] = str(identity)
        vertex_index = data.get("vertex_index")
        if vertex_index is not None and identity is not None:
            vertex_ids[int(vertex_index)] = str(identity)
    members = getattr(analysis.trees["split"], "branch_member_table", None)
    if members is not None and {"vertex_index", "dna_sequence"}.issubset(
        members.columns
    ):
        for vertex_index, sequence in members.select(
            "vertex_index", "dna_sequence"
        ).iter_rows():
            if vertex_index is not None and sequence is not None:
                vertex_ids[int(vertex_index)] = str(sequence)
    return tree_ids, vertex_ids


def _public_table(name: str, frame: pl.DataFrame, analysis: Any) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return frame
    tree_ids, vertex_ids = _identity_maps(analysis)
    values: dict[str, list[Any]] = {}
    for column in frame.columns:
        data = frame[column].to_list()
        if column.endswith("_tree_node_index"):
            identity_column = column.removesuffix("_index") + "_identity"
            values[identity_column] = [
                tree_ids.get(int(value)) if value is not None else None
                for value in data
            ]
        elif column.endswith("_vertex_index"):
            identity_column = column.removesuffix("_index") + "_identity"
            values[identity_column] = [
                vertex_ids.get(int(value)) if value is not None else None
                for value in data
            ]
        elif column == "vertex_index":
            values["vertex_identity"] = [
                vertex_ids.get(int(value)) if value is not None else None
                for value in data
            ]
        elif column == "tree_node_index":
            values["tree_node_identity"] = [
                tree_ids.get(int(value)) if value is not None else None
                for value in data
            ]
        elif column.endswith("_index"):
            continue
        else:
            values[column] = data
    result = pl.DataFrame(values) if values else pl.DataFrame()
    if name in {"branch_table", "branch_span_table", "branch_member_table"}:
        missing_identity = [
            column
            for column in ("source_tree_node_identity", "target_tree_node_identity")
            if column in result.columns and result[column].null_count()
        ]
        # Some tables carry their stable sequence labels directly. Prefer those over
        # exposing backend indices if a particular Topographer version lacks payloads.
        for endpoint, label in (("source", "source_label"), ("target", "target_label")):
            identity_column = f"{endpoint}_tree_node_identity"
            dna_column = f"{endpoint}_dna_sequence"
            if identity_column in result.columns and label in result.columns:
                result = result.with_columns(
                    pl.coalesce(identity_column, label).alias(identity_column)
                )
            elif identity_column in result.columns and dna_column in result.columns:
                result = result.with_columns(
                    pl.coalesce(identity_column, dna_column).alias(identity_column)
                )
        # Keep unknown identities visibly null rather than inventing sequence labels.
        del missing_identity
    return result


def export_tables(analysis: Any, directory: Path | None = None) -> list[Path]:
    """Export analysis tables as CSV and Parquet with private indices removed.

    ``directory`` defaults to ``output/tables/canonical``.
    """
    table_dir = directory or SHELF_DIR
    table_dir.mkdir(parents=True, exist_ok=True)
    tree = analysis.trees["split"]
    frames = {
        "input_summary": analysis.input_summary,
        "event_table": tree.event_table,
        "branch_table": tree.branch_table,
        "branch_span_table": tree.branch_span_table,
        "branch_member_table": tree.branch_member_table,
        "critical_node_table": tree.critical_node_table,
        "persistence_table": tree.persistence_table,
    }
    outputs: list[Path] = []
    schemas: dict[str, dict[str, str]] = {}
    for name, frame in frames.items():
        if frame is None:
            continue
        public = _public_table(name, frame, analysis)
        schemas[name] = {column: str(dtype) for column, dtype in public.schema.items()}
        for extension in ("csv", "parquet"):
            target = table_dir / f"{name}.{extension}"
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=f".{extension}", dir=table_dir
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                if extension == "csv":
                    public.write_csv(temporary)
                else:
                    public.write_parquet(temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(target)
    _write_json_atomic(table_dir / "schema.json", schemas)
    return outputs


def export_component_persistence_table(
    pairs: pl.DataFrame, directory: Path | None = None
) -> list[Path]:
    """Export component-persistence pairs with their stable sequence identities."""
    required = (
        "pair_id",
        "pair_kind",
        "homology_dimension",
        "birth_sequence",
        "death_sequence",
        "birth_scalar",
        "death_scalar",
        "persistence",
    )
    table_dir = directory or SHELF_DIR
    missing = set(required) - set(pairs.columns)
    if missing:
        raise ValueError(
            "Component persistence table is missing columns: "
            + ", ".join(sorted(missing))
        )
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    table = pairs.select(required)
    for extension in ("csv", "parquet"):
        target = table_dir / f"network_component_persistence.{extension}"
        fd, temporary_name = tempfile.mkstemp(
            prefix=".network_component_persistence.",
            suffix=f".{extension}",
            dir=table_dir,
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            if extension == "csv":
                table.write_csv(temporary)
            else:
                table.write_parquet(temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.append(target)
    return outputs
