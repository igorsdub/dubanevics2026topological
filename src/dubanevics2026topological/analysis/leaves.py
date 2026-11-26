"""Sequence-keyed terminal branch populations from an existing tree view."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl

from .biological_annotations import STOP_CODON_SYMBOLS


@dataclass(frozen=True)
class Leaves:
    """Leaf and member tables with tree-population reconciliation."""

    table: pl.DataFrame
    members: pl.DataFrame
    summary: dict[str, Any]

    def save(self, *, parent: Any, context: Any) -> Path:
        """Persist this product with its identified parent tree."""
        return save_leaves(self, parent=parent, context=context)

    @classmethod
    def load(cls, label: str, *, context: Any) -> Leaves:
        """Load an identified and verified saved leaves product."""
        return load_leaves(label, context=context)


@dataclass(frozen=True)
class BandParent:
    """Direct-band identity and its verified historical draw evidence."""

    parameters: dict[str, Any]
    scientific_identity: dict[str, Any]
    scientific_id_sha256: str
    label: str
    data_provenance: dict[str, str]
    lineage: dict[str, Any]


def direct_band_parent(
    draws: Any,
    summary: dict[str, Any],
    *,
    equivalence: dict[str, str] | None = None,
) -> BandParent:
    """Derive direct-band identity from a verified restored bootstrap product.

    ``equivalence`` (from ``functional_set_equivalence``) is recorded in the lineage only; it does
    not enter the band identity.
    """
    from ..landscape.products import _sha256
    from ..results import ALLOWED_KEYS, format_label
    from ..simplification.identities import (
        band_identity,
        checkpoint_draw_identity,
        checkpoint_method,
    )

    if (
        draws.product_type != "bootstrap"
        or draws.report.status != "full"
        or not draws.report.byte_integrity
    ):
        raise ValueError("Direct band requires fully verified bootstrap draw evidence.")
    record = draws.manifest.artifacts.get("checkpoint/summary.json")
    source = draws.path / "checkpoint/summary.json"
    if (
        not record
        or _sha256(source) != record["sha256"]
        or json.loads(source.read_text(encoding="utf-8")) != summary
    ):
        raise ValueError("Restored summary differs from verified checkpoint evidence.")
    method = checkpoint_method(summary)
    quantile = float(summary["settings"]["band_quantile"])
    draws_record = draws.manifest.artifacts.get("checkpoint/draw_stats.parquet")
    draws_path = draws.path / "checkpoint/draw_stats.parquet"
    if not draws_record or _sha256(draws_path) != draws_record["sha256"]:
        raise ValueError("Restored draw statistics differ from verified evidence.")
    draw_table = pl.read_parquet(draws_path, columns=["draw", "condition"])
    n_draws = int(summary["settings"]["n_replicates"])
    ids = list(range(n_draws))
    for condition in ("real", "flat"):
        if (
            draw_table.filter(pl.col("condition") == condition)["draw"].sort().to_list()
            != ids
        ):
            raise ValueError(f"Incomplete or duplicated {condition} draw IDs.")
    if draw_table.height != 2 * n_draws:
        raise ValueError("Unexpected draw conditions in checkpoint.")
    draw_fields, draw_id = checkpoint_draw_identity(
        summary["fingerprint"], seed=int(summary["settings"]["seed"]), draw_ids=ids
    )
    identity, digest = band_identity(draw_scientific_id_sha256=draw_id, bandq=quantile)
    params = {
        key: value
        for key, value in draws.manifest.scientific_identity.items()
        if key in ALLOWED_KEYS
        and key not in {"tree", "bandq", "host", "run", "hash", "style"}
    }
    params.update(tree="band", bandq=quantile)
    return BandParent(
        parameters=params,
        scientific_identity=identity,
        scientific_id_sha256=digest,
        label=format_label("tree", {**params, "hash": digest[:8]}),
        data_provenance=draws.manifest.data_provenance,
        lineage={
            "source_checkpoint": str(draws.path / "checkpoint"),
            "source_label": draws.legacy_alias or draws.label,
            "source_schema": summary["fingerprint"]["schema"],
            "historical_checkpoint_method": method,
            "source_fingerprint": summary["fingerprint"],
            "source_artifacts": draws.manifest.artifacts,
            "legacy_synthetic_scientific_id_sha256": draws.manifest.scientific_id_sha256,
            "draw_scientific_identity": draw_fields,
            "draw_scientific_id_sha256": draw_id,
            **({"functional_set_equivalence": equivalence} if equivalence else {}),
        },
    )


def save_band_tree(view: Any, *, band: BandParent, nominal: Any, context: Any) -> Path:
    """Publish a direct-band tree with its nominal and simplification identities."""
    from ..landscape.products import _publish, _sha256

    if (
        nominal.product_type != "tree"
        or nominal.report.status != "full"
        or not nominal.report.byte_integrity
        or nominal.manifest.scientific_identity.get("tree") != "nominal"
    ):
        raise ValueError("A fully verified nominal tree is required.")
    nominal_identity = nominal.manifest.scientific_identity
    for key in ("graph", "thr", "fdr", "q", "p"):
        if nominal_identity.get(key) != band.parameters.get(key):
            raise ValueError(f"Band and nominal tree disagree on {key}.")
    graph = view.tree
    nodes = {index: dict(graph[index]) for index in graph.node_indices()}
    assignments = view.assignments.to_dicts()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in assignments:
        groups.setdefault((row["upper_sequence"], row["lower_sequence"]), []).append(
            row
        )
    branches = []
    for upper, lower in graph.edge_list():
        upper_id, lower_id = nodes[upper]["dna_sequence"], nodes[lower]["dna_sequence"]
        branches.append(
            {
                "source_tree_node_identity": upper_id,
                "target_tree_node_identity": lower_id,
                "branch_vertex_count": len(groups.get((upper_id, lower_id), [])),
                "source_scalar": float(nodes[upper]["scalar"]),
                "target_scalar": float(nodes[lower]["scalar"]),
            }
        )
    interior = [
        {
            "source_tree_node_identity": r["upper_sequence"],
            "target_tree_node_identity": r["lower_sequence"],
            "dna_sequence": r["dna_sequence"],
            "aa_sequence": r["aa_sequence"],
            "scalar": r["raw_fitness"],
            "location": r["location"],
        }
        for r in assignments
        if r["location"] not in {"critical", "root", "absorbed_root"}
    ]
    root_members = [
        r for r in assignments if r["location"] in {"root", "absorbed_root"}
    ]
    critical = [
        {
            "tree_node_identity": node["dna_sequence"],
            **{key: value for key, value in node.items() if not key.endswith("_index")},
        }
        for node in nodes.values()
    ]
    code = Path(__file__).resolve()
    code_paths = (
        code,
        code.parent.parent / "simplification" / "bootstrap_smoothed_tree.py",
        code.parent.parent / "simplification" / "selected_leaf_simplification.py",
        code.parent.parent / "simplification" / "identities.py",
    )
    identity = {
        **band.parameters,
        "direct_band_identity_sha256": band.scientific_id_sha256,
        "stage_code_digest": {path.name: _sha256(path) for path in code_paths},
    }
    parents = (
        {
            "product_type": "tree",
            "label": nominal.label,
            "scientific_id_sha256": nominal.manifest.scientific_id_sha256,
        },
        {
            "product_type": "direct_band_identity",
            "scientific_id_sha256": band.scientific_id_sha256,
            "scientific_identity": band.scientific_identity,
            "source_lineage": band.lineage,
        },
    )

    def write(directory: Path) -> None:
        pl.DataFrame(
            branches,
            schema={
                "source_tree_node_identity": pl.String,
                "target_tree_node_identity": pl.String,
                "branch_vertex_count": pl.Int64,
                "source_scalar": pl.Float64,
                "target_scalar": pl.Float64,
            },
        ).write_parquet(directory / "branches.parquet")
        pl.DataFrame(
            interior,
            schema={
                "source_tree_node_identity": pl.String,
                "target_tree_node_identity": pl.String,
                "dna_sequence": pl.String,
                "aa_sequence": pl.String,
                "scalar": pl.Float64,
                "location": pl.String,
            },
        ).write_parquet(directory / "branch_members.parquet")
        pl.DataFrame(critical).write_parquet(directory / "critical_nodes.parquet")
        view.pairs.write_parquet(directory / "persistence_pairs.parquet")
        pl.DataFrame(root_members, schema=view.assignments.schema).write_parquet(
            directory / "root_members.parquet"
        )

    path, _ = _publish(
        "tree",
        band.parameters,
        identity,
        context=context,
        input_data_sha256s=band.data_provenance,
        parents=parents,
        code_paths=code_paths,
        write_artifacts=write,
    )
    return path


def save_leaves(product: Leaves, *, parent: Any, context: Any) -> Path:
    """Save a leaves product identified by its verified parent tree and code.

    The caller must provide the tree used to compute the product as ``parent``.
    """
    from ..landscape.products import _publish, _sha256
    from ..results import ALLOWED_KEYS

    if isinstance(parent, BandParent):
        raise TypeError("Publish the band tree before saving its leaves.")
    if (
        parent.product_type != "tree"
        or parent.report.status != "full"
        or not parent.report.byte_integrity
    ):
        raise ValueError("Leaves require a fully verified tree parent.")
    else:
        params = {
            key: value
            for key, value in parent.manifest.scientific_identity.items()
            if key in ALLOWED_KEYS and key not in {"host", "run", "hash"}
        }
        parent_ref = {
            "product_type": "tree",
            "label": parent.label,
            "scientific_id_sha256": parent.manifest.scientific_id_sha256,
        }
        for ref in parent.manifest.parents:
            note = ref.get("source_lineage", {}).get("functional_set_equivalence")
            if ref.get("product_type") == "direct_band_identity" and note:
                parent_ref["functional_set_equivalence"] = note
        inputs = parent.manifest.data_provenance
    if params.get("tree") != product.summary["tree_state"]:
        raise ValueError("Leaf state does not match the parent tree.")
    code_path = Path(__file__).resolve()
    code_paths = (
        code_path,
        code_path.with_name("biological_annotations.py"),
        code_path.parent.parent / "simplification" / "selected_leaf_simplification.py",
        code_path.parent.parent / "simplification" / "bootstrap_smoothed_tree.py",
        code_path.parent.parent / "simplification" / "checkpoint_identity.py",
        code_path.parent.parent / "simplification" / "identities.py",
    )
    identity = {
        **params,
        "population": product.summary["population"],
        "population_rule": product.summary["population_rule"],
        "stage_code_digest": {path.name: _sha256(path) for path in code_paths},
    }

    def write(directory: Path) -> None:
        product.table.write_parquet(directory / "leaves.parquet")
        product.members.write_parquet(directory / "members.parquet")
        (directory / "summary.json").write_text(
            json.dumps(product.summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "leaves",
        params,
        identity,
        context=context,
        input_data_sha256s=inputs,
        parents=(parent_ref,),
        code_paths=code_paths,
        write_artifacts=write,
    )
    return path


def load_leaves(label: str, *, context: Any) -> Leaves:
    """Load and verify a saved leaves product using the results lookup."""
    from ..results import load

    loaded = load("leaves", label, context=context)
    if loaded.report.status != "full" or not loaded.report.byte_integrity:
        raise ValueError("Leaves product is not fully verified.")
    return Leaves(
        pl.read_parquet(loaded.artifacts["leaves.parquet"]),
        pl.read_parquet(loaded.artifacts["members.parquet"]),
        json.loads(loaded.artifacts["summary.json"].read_text(encoding="utf-8")),
    )


_LEAF_SCHEMA = {
    "population": pl.String,
    "tree_state": pl.String,
    "peak_dna_sequence": pl.String,
    "peak_aa_sequence": pl.String,
    "birth_fitness": pl.Float64,
    "adjacent_saddle_dna_sequence": pl.String,
    "adjacent_saddle_aa_sequence": pl.String,
    "adjacent_saddle_fitness": pl.Float64,
    "is_root_endpoint": pl.Boolean,
    "member_count": pl.Int64,
    "stop_composition_1": pl.Int64,
    "stop_composition_2": pl.Int64,
    "stop_composition_3": pl.Int64,
    "is_synonymous": pl.Boolean,
    "is_top2_member_count": pl.Boolean,
}
_MEMBER_SCHEMA = {
    "population": pl.String,
    "tree_state": pl.String,
    "peak_dna_sequence": pl.String,
    "member_dna_sequence": pl.String,
    "member_raw_fitness": pl.Float64,
    "member_rank_in_leaf": pl.Int64,
}


def build_leaves(
    tree_analysis: Any,
    *,
    population: str,
    tree_state: str,
    hills: pl.DataFrame | None = None,
) -> Leaves:
    """Describe the nominal or direct band view without rebuilding its graph.

    A single sequence that is both peak and root is one ineligible root-only
    leaf, with the sequence in the leaf and no remaining root population.
    """
    if tree_state not in {"nominal", "band"}:
        raise ValueError("tree_state must be nominal or band.")
    if tree_state == "band" and hills is None:
        raise ValueError("band leaves require bootstrap hills.")
    if tree_state == "nominal" and hills is not None:
        raise ValueError("nominal leaves do not use bootstrap hills.")
    source_tree = tree_analysis.tree
    if source_tree.num_nodes() <= 1:
        assignments = []
        for index in source_tree.node_indices():
            node = source_tree[index]
            assignments.append(
                {
                    "dna_sequence": node["dna_sequence"],
                    "aa_sequence": node["aa_sequence"],
                    "raw_fitness": float(node["scalar"]),
                    "upper_sequence": node["dna_sequence"],
                    "lower_sequence": node["dna_sequence"],
                    "location": "singleton_leaf",
                }
            )
        view = SimpleNamespace(tree=source_tree, assignments=pl.DataFrame(assignments))
    elif tree_state == "band":
        from ..simplification import build_band_simplified_view

        view = build_band_simplified_view(tree_analysis, hills)
    else:
        from ..simplification.selected_leaf_simplification import (
            absorb_terminal_branches,
        )

        view = absorb_terminal_branches(tree_analysis, set())
    graph = view.tree
    nodes = {i: dict(graph[i]) for i in graph.node_indices()}
    assignments = view.assignments.to_dicts()
    identities = [row["dna_sequence"] for row in assignments]
    if len(identities) != len(set(identities)):
        raise ValueError("Tree assignments contain duplicate sequences.")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for assignment in assignments:
        key = (assignment["upper_sequence"], assignment["lower_sequence"])
        groups.setdefault(key, []).append(assignment)
    rows, members = [], []
    for index, peak in nodes.items():
        if graph.in_degree(index) != 0:
            continue
        successors = list(graph.successor_indices(index))
        if len(successors) > 1 or (not successors and len(nodes) != 1):
            raise ValueError("A terminal peak must have one descending branch.")
        lower_index = successors[0] if successors else index
        lower = nodes[lower_index]
        saddle = lower if graph.in_degree(lower_index) > 1 else None
        if saddle is None and graph.out_degree(lower_index) != 0:
            raise ValueError("Terminal branch ends at neither saddle nor root.")
        group = groups.get((peak["dna_sequence"], lower["dna_sequence"]), [])
        group.sort(key=lambda row: (-row["raw_fitness"], row["dna_sequence"]))
        if not group or peak["dna_sequence"] not in {r["dna_sequence"] for r in group}:
            raise ValueError("Terminal branch is missing its peak.")
        counts = [
            sum(
                sum(c in STOP_CODON_SYMBOLS for c in r["aa_sequence"]) == n
                for r in group
            )
            for n in (1, 2, 3)
        ]
        rows.append(
            {
                "population": population,
                "tree_state": tree_state,
                "peak_dna_sequence": peak["dna_sequence"],
                "peak_aa_sequence": peak["aa_sequence"],
                "birth_fitness": float(peak["scalar"]),
                "adjacent_saddle_dna_sequence": saddle["dna_sequence"]
                if saddle
                else None,
                "adjacent_saddle_aa_sequence": saddle["aa_sequence"]
                if saddle
                else None,
                "adjacent_saddle_fitness": float(saddle["scalar"]) if saddle else None,
                "is_root_endpoint": saddle is None,
                "member_count": len(group),
                "stop_composition_1": counts[0],
                "stop_composition_2": counts[1],
                "stop_composition_3": counts[2],
                "is_synonymous": peak["aa_sequence"] == saddle["aa_sequence"]
                if saddle
                else None,
                "is_top2_member_count": False,
            }
        )
        members.extend(
            {
                "population": population,
                "tree_state": tree_state,
                "peak_dna_sequence": peak["dna_sequence"],
                "member_dna_sequence": row["dna_sequence"],
                "member_raw_fitness": row["raw_fitness"],
                "member_rank_in_leaf": rank,
            }
            for rank, row in enumerate(group, 1)
        )
    rows.sort(key=lambda row: (-row["member_count"], row["peak_dna_sequence"]))
    for row in rows[:2]:
        row["is_top2_member_count"] = True
    terminal = {row["member_dna_sequence"] for row in members}
    if len(terminal) != len(members):
        raise ValueError("A sequence belongs to multiple terminal leaves.")
    root_count = sum(
        row["location"] in {"root", "absorbed_root"} for row in assignments
    )
    summary = {
        "population": population,
        "tree_state": tree_state,
        "leaf_count": len(rows),
        "top2_shortfall": max(0, 2 - len(rows)),
        "ineligible_endpoint_count": sum(row["is_root_endpoint"] for row in rows),
        "eligible_leaf_denominator": sum(not row["is_root_endpoint"] for row in rows),
        "terminal_member_count": len(terminal),
        "root_member_count": root_count,
        "internal_member_count": len(assignments) - len(terminal) - root_count,
        "tree_member_count": len(assignments),
        "population_rule": (
            "Terminal branch peak and represented interior, excluding its lower endpoint; "
            "a single peak/root sequence is one root-only leaf with no remaining root members."
        ),
    }
    return Leaves(
        pl.DataFrame(rows, schema=_LEAF_SCHEMA),
        pl.DataFrame(members, schema=_MEMBER_SCHEMA),
        summary,
    )
