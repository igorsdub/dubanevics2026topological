"""Publish tree layout tables against an identified, verified tree parent."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl
import topographer.plotting
from dubanevics2026topological.papkou2023 import build_papkou2023_backbone_analysis

from ..landscape import selected_landscape
from ..landscape.products import _publish, _sha256
from ..results import ALLOWED_KEYS, RunContext
from ..simplification import bootstrap_smoothed_tree, selected_leaf_simplification

_LABEL_KEYS = frozenset(ALLOWED_KEYS) - {"host", "run", "hash", "style"}


def _check_scalar_values(
    state: str, kind: str, values: list[tuple[str, Any, float]]
) -> None:
    mismatches = [
        (dna, actual, expected)
        for dna, actual, expected in values
        if actual != expected
    ]
    if mismatches:
        differences = [
            abs(float(actual) - expected)
            for _, actual, expected in mismatches
            if actual is not None
        ]
        maximum = max(differences, default=float("nan"))
        names = ", ".join(sorted({dna for dna, _, _ in mismatches})[:10])
        raise ValueError(
            f"{state} {kind} scalar mismatch for {names}; maximum absolute difference {maximum!r}."
        )


def _check_internal_scalars(tree_analysis: Any, parent: Any, state: str) -> None:
    saved = _members(
        pl.read_parquet(parent.artifacts["critical_nodes.parquet"]), "scalar"
    )
    graph = tree_analysis.tree
    nodes = [graph[index] for index in graph.node_indices()]
    identities = [str(node["dna_sequence"]) for node in nodes]
    if len(set(identities)) != len(identities) or set(identities) != set(saved):
        raise ValueError(
            f"{state} internal tree node identities differ from the parent tree."
        )
    _check_scalar_values(
        state,
        "internal tree",
        [
            (
                str(node["dna_sequence"]),
                node["scalar"],
                saved[str(node["dna_sequence"])],
            )
            for node in nodes
        ],
    )


def _check_plot_scalars(plot: Any, parent: Any, state: str) -> None:
    saved = _members(
        pl.read_parquet(parent.artifacts["critical_nodes.parquet"]), "scalar"
    )
    nodes = plot.node_table
    indices = dict(nodes.select("tree_node_index", "dna_sequence").iter_rows())
    if len(indices) != nodes.height or set(indices.values()) != set(saved):
        raise ValueError(
            f"{state} plotted node identities differ from the parent tree."
        )
    values = [
        (dna, y, saved[dna]) for dna, y in nodes.select("dna_sequence", "y").iter_rows()
    ]
    for row in plot.edge_segment_table.iter_rows(named=True):
        try:
            source = indices[row["source_tree_node_index"]]
            target = indices[row["target_tree_node_index"]]
        except KeyError as exc:
            raise ValueError(
                f"{state} plotted segment refers to an unknown tree node."
            ) from exc
        if row["segment_kind"] == "vertical":
            values.extend(
                (
                    (target, row["y_start"], saved[target]),
                    (source, row["y_end"], saved[source]),
                )
            )
        elif row["segment_kind"] == "horizontal":
            values.extend(
                (
                    (source, row["y_start"], saved[source]),
                    (source, row["y_end"], saved[source]),
                )
            )
        else:
            raise ValueError(f"{state} has an unknown plotted segment kind.")
    _check_scalar_values(state, "plot", values)


def _members(frame: pl.DataFrame, scalar: str) -> dict[str, float]:
    if (
        "dna_sequence" in frame.columns
        and "vertex_identity" in frame.columns
        and any(
            left is None or right is None or left != right
            for left, right in frame.select(
                "dna_sequence", "vertex_identity"
            ).iter_rows()
        )
    ):
        raise ValueError("Conflicting sequence identities in tree membership.")
    identity = "dna_sequence" if "dna_sequence" in frame.columns else "vertex_identity"
    rows = frame.select(identity, scalar).iter_rows()
    result: dict[str, float] = {}
    for dna, fitness in rows:
        if dna is None or fitness is None or dna in result:
            raise ValueError(
                "Tree membership contains missing or duplicate identities."
            )
        result[str(dna)] = float(fitness)
    return result


def _reconcile_members(tree: Any, tables: dict[str, pl.DataFrame]) -> None:
    saved = _members(
        pl.read_parquet(tree.artifacts["critical_nodes.parquet"]), "scalar"
    )
    additional = _members(
        pl.read_parquet(tree.artifacts["branch_members.parquet"]), "scalar"
    )
    if saved.keys() & additional.keys():
        raise ValueError("Tree membership overlaps critical and branch rows.")
    saved.update(additional)
    if "root_members.parquet" in tree.artifacts:
        root = _members(
            pl.read_parquet(tree.artifacts["root_members.parquet"]), "raw_fitness"
        )
        if any(saved[dna] != root[dna] for dna in saved.keys() & root.keys()):
            raise ValueError("Tree root membership disagrees with critical rows.")
        saved.update(root)
    if "assignments" in tables:
        actual = _members(tables["assignments"], "raw_fitness")
    else:
        actual = _members(tables["critical_node_table"], "scalar")
        branch = _members(tables["branch_member_table"], "scalar")
        if actual.keys() & branch.keys():
            raise ValueError("Layout membership overlaps critical and branch rows.")
        actual.update(branch)
    if actual != saved:
        raise ValueError("Layout membership differs from the parent tree.")
    parent_members = pl.read_parquet(tree.artifacts["branch_members.parquet"])
    identity = (
        "dna_sequence"
        if "dna_sequence" in parent_members.columns
        else "vertex_identity"
    )
    expected_branches = {
        str(dna): frozenset((str(upper), str(lower)))
        for dna, upper, lower in parent_members.select(
            identity, "source_tree_node_identity", "target_tree_node_identity"
        ).iter_rows()
    }
    if "assignments" in tables:
        actual_branches = {
            str(dna): frozenset((str(upper), str(lower)))
            for dna, upper, lower, location in tables["assignments"]
            .select("dna_sequence", "upper_sequence", "lower_sequence", "location")
            .iter_rows()
            if location not in {"critical", "root", "absorbed_root"}
        }
    else:
        indices = dict(
            tables["node_table"].select("tree_node_index", "dna_sequence").iter_rows()
        )
        actual_branches = {
            str(dna): frozenset((indices[upper], indices[lower]))
            for dna, upper, lower in tables["branch_member_table"]
            .select("dna_sequence", "source_tree_node_index", "target_tree_node_index")
            .iter_rows()
        }
    if actual_branches != expected_branches:
        raise ValueError("Layout branch assignments differ from the parent tree.")
    if "assignments" in tables and "location" in parent_members.columns:
        expected_locations = dict(
            parent_members.select(identity, "location").iter_rows()
        )
        actual_locations = {
            dna: location
            for dna, location in tables["assignments"]
            .select("dna_sequence", "location")
            .iter_rows()
            if dna in expected_locations
        }
        if actual_locations != expected_locations:
            raise ValueError("Layout branch locations differ from the parent tree.")


def build_tree_layout(
    tree_analysis: Any,
    *,
    tree: Any,
    context: RunContext,
    nominal_tree: Any | None = None,
    hills: pl.DataFrame | None = None,
    vertex_identities: dict[int, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Capture existing nominal geometry or build the existing direct-band view."""
    members = tree_analysis.branch_member_table
    if "dna_sequence" not in members.columns:
        if vertex_identities is None or "vertex_index" not in members.columns:
            raise ValueError("Tree layout requires an explicit vertex-to-DNA mapping.")
        identities = [vertex_identities.get(index) for index in members["vertex_index"]]
        if any(identity is None for identity in identities):
            raise ValueError("Tree layout contains unmapped branch vertices.")
        members = members.with_columns(
            pl.Series("dna_sequence", identities, dtype=pl.String)
        )
        tree_analysis = replace(tree_analysis, branch_member_table=members)
    state = tree.manifest.scientific_identity.get("tree")
    state_label = tree.label
    _check_internal_scalars(
        tree_analysis,
        nominal_tree if state == "band" and nominal_tree is not None else tree,
        state_label,
    )
    if state == "band":
        if hills is None:
            raise ValueError("Band layout requires saved bootstrap hills.")
        source = bootstrap_smoothed_tree.build_band_simplified_view(
            tree_analysis, hills
        )
    elif state == "nominal" and hills is None:
        source = tree_analysis
    else:
        raise ValueError("Tree state and supplied band evidence disagree.")
    if progress is not None:
        progress("layout builder complete")
    plot = source.plot_data
    _check_plot_scalars(plot, tree, state_label)
    tables = {
        "node_table": plot.node_table,
        "edge_table": plot.edge_table,
        "edge_segment_table": plot.edge_segment_table,
        "wild_type_marker_table": source.wild_type_marker_table,
    }
    metadata: dict[str, str | int | float | bool | None] = {}
    if is_dataclass(plot):
        for field in fields(plot):
            value = getattr(plot, field.name)
            if isinstance(value, pl.DataFrame):
                tables[field.name] = value
            elif value is None or type(value) in (str, int, float, bool):
                metadata[field.name] = value
            else:
                raise TypeError(
                    f"Unsupported tree plot field {field.name!r}: {type(value).__name__}."
                )
    if state == "band":
        tables.update(assignments=source.assignments, pairs=source.pairs)
    else:
        for name in ("branch_table", "critical_node_table", "branch_member_table"):
            tables[name] = getattr(source, name)
    code_paths = tuple(
        Path(module.__file__).resolve()
        for module in (
            topographer.plotting,
            bootstrap_smoothed_tree,
            selected_leaf_simplification,
            selected_landscape,
        )
    ) + (
        Path(inspect.getfile(topographer.plotting.make_tree_layout)).resolve(),
        Path(inspect.getfile(build_papkou2023_backbone_analysis)).resolve(),
        Path(__file__).resolve(),
    )
    return publish_tree_layout(
        tree=tree,
        tables=tables,
        plot_metadata=metadata,
        label_fields={
            key: value
            for key, value in tree.manifest.scientific_identity.items()
            if key in _LABEL_KEYS
        },
        layout_code_paths=code_paths,
        context=context,
        nominal_tree=nominal_tree,
        progress=progress,
    )


def publish_tree_layout(
    *,
    tree: Any,
    tables: dict[str, pl.DataFrame],
    label_fields: dict[str, Any],
    layout_code_paths: tuple[Path, ...],
    context: RunContext,
    nominal_tree: Any | None = None,
    plot_metadata: dict[str, str | int | float | bool | None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Save layout tables; caller builds the tables using the existing layout code.

    The tree and optional nominal parent must already be fully verified. Table
    production is the responsibility of the build stage. Node, edge and member
    identities are checked here before publication.
    """
    parents = (tree,) if nominal_tree is None else (nominal_tree, tree)
    for parent in parents:
        if (
            parent.product_type != "tree"
            or parent.report.status != "full"
            or not parent.report.byte_integrity
        ):
            raise ValueError("Tree layouts require fully verified tree parents.")
    identity = tree.manifest.scientific_identity
    inherited = {key: value for key, value in identity.items() if key in _LABEL_KEYS}
    if label_fields != inherited:
        raise ValueError("Layout label fields differ from the parent tree identity.")
    if inherited.get("tree") == "band":
        if (
            nominal_tree is None
            or nominal_tree.manifest.scientific_identity.get("tree") != "nominal"
        ):
            raise ValueError("Band layouts require their verified nominal tree parent.")
        if not any(
            ref.get("product_type") == "tree"
            and ref.get("scientific_id_sha256")
            == nominal_tree.manifest.scientific_id_sha256
            for ref in tree.manifest.parents
        ):
            raise ValueError("Band tree does not identify the supplied nominal parent.")
    elif nominal_tree is not None:
        raise ValueError("Only band layouts have a separate nominal tree parent.")
    required = {
        "node_table",
        "edge_table",
        "edge_segment_table",
        "wild_type_marker_table",
    }
    if inherited.get("tree") == "band":
        required |= {"assignments", "pairs"}
    else:
        required |= {"branch_table", "critical_node_table", "branch_member_table"}
    if not required <= tables.keys():
        raise ValueError(
            f"Missing tree layout tables: {sorted(required - tables.keys())}."
        )
    if any(
        not name.isidentifier() or not isinstance(frame, pl.DataFrame)
        for name, frame in tables.items()
    ):
        raise ValueError("Layout table names and values must be valid.")
    node_table = tables["node_table"]
    if not {"tree_node_index", "dna_sequence"} <= set(node_table.columns):
        raise ValueError(
            "Layout nodes must explicitly map backend indices to DNA identities."
        )
    if (
        node_table["tree_node_index"].n_unique() != node_table.height
        or node_table["dna_sequence"].n_unique() != node_table.height
        or node_table["tree_node_index"].null_count()
        or node_table["dna_sequence"].null_count()
    ):
        raise ValueError("Layout node mappings must be unique and non-null.")
    saved_nodes = pl.read_parquet(tree.artifacts["critical_nodes.parquet"])
    if set(saved_nodes["dna_sequence"]) != set(node_table["dna_sequence"]):
        raise ValueError("Layout nodes differ from the parent tree.")
    indices = dict(node_table.select("tree_node_index", "dna_sequence").iter_rows())
    saved_branches = pl.read_parquet(tree.artifacts["branches.parquet"])
    expected_edges = sorted(
        tuple(sorted((str(source), str(target))))
        for source, target in saved_branches.select(
            "source_tree_node_identity", "target_tree_node_identity"
        ).iter_rows()
    )
    try:
        actual_edges = sorted(
            tuple(sorted((indices[source], indices[target])))
            for source, target in tables["edge_table"]
            .select("source_tree_node_index", "target_tree_node_index")
            .iter_rows()
        )
    except KeyError as exc:
        raise ValueError("Layout edge refers to an unknown tree node.") from exc
    if actual_edges != expected_edges:
        raise ValueError("Layout edges differ from the parent tree.")
    _reconcile_members(tree, tables)
    if "pairs" in tables and not tables["pairs"].equals(
        pl.read_parquet(tree.artifacts["persistence_pairs.parquet"])
    ):
        raise ValueError("Layout persistence pairs differ from the parent tree.")
    if progress is not None:
        progress("reconciliation complete")
    if not layout_code_paths:
        raise ValueError("Layout code provenance is required.")
    digest = {str(index): _sha256(path) for index, path in enumerate(layout_code_paths)}
    refs = tuple(
        {
            "product_type": "tree",
            "label": parent.label,
            "scientific_id_sha256": parent.manifest.scientific_id_sha256,
        }
        for parent in parents
    )

    def write(directory: Path) -> None:
        for name, frame in sorted(tables.items()):
            frame.write_parquet(directory / f"{name}.parquet")
        (directory / "plot_metadata.json").write_text(
            json.dumps(plot_metadata or {}, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "tree_layout",
        inherited,
        {
            **inherited,
            "tree_scientific_id_sha256": tree.manifest.scientific_id_sha256,
            "layout_code_digest": digest,
        },
        context=context,
        input_data_sha256s={},
        parents=refs,
        code_paths=layout_code_paths,
        write_artifacts=write,
    )
    if progress is not None:
        progress("published")
    return path
