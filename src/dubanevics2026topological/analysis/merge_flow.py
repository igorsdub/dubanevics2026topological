"""Merge-flow components of one saved merge tree, keyed by DNA.

A merge-flow component is a tree arc (upper node to lower node) with the subtree
above it (GLOSSARY.md, "merge-flow component"). It is born at the scalar of its
upper endpoint and dies at the scalar of its lower node, where it merges. It is
named by its elder summit, the highest summit among the arcs entering its upper
node, ties broken by the smaller ``vertex_index`` (the landscape vertex payload
order) in the saved product, and by the smaller DNA sequence only as the fallback
when no vertex order is supplied; ``summary.json`` records which (``tie_break``).
The elder cross-check against ``persistence_pairs`` raises on any disagreement.
Counts and names are computed on the full swept tree; pruning only marks rows.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import polars as pl

from .ridges import RidgeTree

CLASSES = frozenset({"functional", "uncertain", "non_functional"})
STATES = ("nominal", "band")
COMPONENTS_FILE = "components.parquet"
EVENTS_FILE = "events.parquet"
LEVELS_FILE = "levels.parquet"
SIDECAR_FILE = "summary.json"
LEVEL_GRID = "exact_breakpoints"
WIDTH_CAVEAT = (
    "Functional-only widths do not add up at merges: non-functional members are "
    "excluded from the width."
)
_BIRTH_COLUMNS = ("birth_sequence", "birth_label", "birth_vertex_identity")
_DEATH_COLUMNS = ("death_sequence", "death_label", "death_vertex_identity")

COMPONENTS_SCHEMA = {
    "population": pl.String,
    "tree_state": pl.String,
    "width_basis": pl.String,
    "component_id": pl.String,
    "summit_dna_sequence": pl.String,
    "summit_aa_sequence": pl.String,
    "summit_raw_fitness": pl.Float64,
    "birth_dna_sequence": pl.String,
    "birth_fitness": pl.Float64,
    "death_dna_sequence": pl.String,
    "death_fitness": pl.Float64,
    "parent_component_id": pl.String,
    "is_leaf": pl.Boolean,
    "own_member_count": pl.Int64,
    "own_functional_count": pl.Int64,
    "total_count_at_death": pl.Int64,
    "functional_count_at_death": pl.Int64,
    "max_functional_count": pl.Int64,
    "pruned": pl.Boolean,
    "prune_reason": pl.String,
    "functional_fitness_mean": pl.Float64,
    "functional_fitness_min": pl.Float64,
    "functional_fitness_max": pl.Float64,
}
EVENTS_SCHEMA = {
    "population": pl.String,
    "tree_state": pl.String,
    "saddle_dna_sequence": pl.String,
    "saddle_fitness": pl.Float64,
    "elder_component_id": pl.String,
    "elder_summit_dna_sequence": pl.String,
    "younger_component_ids": pl.List(pl.String),
    "child_count": pl.Int64,
    "balanced_merge": pl.Int64,
    "smaller_child_total_count": pl.Int64,
    "saddle_drop": pl.Float64,
    "largest_child_functional_count": pl.Int64,
    "stub_functional_count": pl.Int64,
    "stub_component_count": pl.Int64,
    "persistence_pair_agrees": pl.Boolean,
}
LEVELS_SCHEMA = {
    "population": pl.String,
    "tree_state": pl.String,
    "component_id": pl.String,
    "level_fitness": pl.Float64,
    "functional_count": pl.Int64,
    "total_count": pl.Int64,
}


@dataclass(frozen=True)
class MergeFlow:
    """The components, merge events and per-level counts of one population and state."""

    components: pl.DataFrame
    events: pl.DataFrame
    levels: pl.DataFrame
    sidecar: dict[str, Any]

    def save(self, directory: Path) -> None:
        """Write the four artifacts of a saved ``merge_flow`` product."""
        self.components.write_parquet(directory / COMPONENTS_FILE)
        self.events.write_parquet(directory / EVENTS_FILE)
        self.levels.write_parquet(directory / LEVELS_FILE)
        (directory / SIDECAR_FILE).write_text(
            json.dumps(self.sidecar, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_artifacts(cls, artifacts: Mapping[str, Path]) -> MergeFlow:
        """Read the artifacts of a saved ``merge_flow`` product."""
        return cls(
            pl.read_parquet(artifacts[COMPONENTS_FILE]),
            pl.read_parquet(artifacts[EVENTS_FILE]),
            pl.read_parquet(artifacts[LEVELS_FILE]),
            json.loads(Path(artifacts[SIDECAR_FILE]).read_text(encoding="utf-8")),
        )


def _structure(tree: RidgeTree) -> dict[str, Any]:
    scalar = dict(
        zip(
            tree.nodes["tree_node_identity"].to_list(),
            tree.nodes["scalar"].to_list(),
            strict=True,
        )
    )
    lower: dict[str, str] = {}
    uppers: dict[str, list[str]] = {}
    for upper, low in zip(
        tree.edges["source_tree_node_identity"].to_list(),
        tree.edges["target_tree_node_identity"].to_list(),
        strict=True,
    ):
        if upper in lower:
            raise ValueError(f"Tree node {upper} has more than one descending edge.")
        if upper not in scalar or low not in scalar:
            raise ValueError("Every tree edge endpoint must be a critical node.")
        lower[upper] = low
        uppers.setdefault(low, []).append(upper)
    roots = [node for node in scalar if node not in lower]
    if len(roots) != 1:
        raise ValueError(f"The merge tree must have one root, found {len(roots)}.")
    edge_members: dict[str, list[str]] = {}
    for upper, dna in zip(
        tree.branch_members["source_tree_node_identity"].to_list(),
        tree.branch_members["dna_sequence"].to_list(),
        strict=True,
    ):
        edge_members.setdefault(upper, []).append(dna)
    if tree.root_members is not None:
        edge_members.setdefault(roots[0], []).extend(
            tree.root_members["dna_sequence"].to_list()
        )
    return {
        "scalar": scalar,
        "lower": lower,
        "uppers": {node: sorted(items) for node, items in uppers.items()},
        "root": roots[0],
        "members": edge_members,
    }


def _leaves(structure: Mapping[str, Any]) -> set[str]:
    return {node for node in structure["scalar"] if node not in structure["uppers"]}


def _pair_set(pairs: pl.DataFrame) -> set[tuple[str, str]]:
    birth = next((c for c in _BIRTH_COLUMNS if c in pairs.columns), None)
    death = next((c for c in _DEATH_COLUMNS if c in pairs.columns), None)
    if birth is None or death is None:
        raise ValueError("Persistence pairs need DNA birth and death columns.")
    return set(zip(pairs[birth].to_list(), pairs[death].to_list(), strict=True))


def _check_input(
    vertices: Mapping[str, float],
    classes: Mapping[str, str | None],
    settings: Mapping[str, Any],
) -> bool:
    """Validate the inputs and say whether the width basis is all sequences."""
    if "min_functional_count_K" not in settings:
        raise ValueError("merge_flow settings require min_functional_count_K.")
    if settings.get("level_grid") != LEVEL_GRID:
        raise ValueError(f"level_grid must be {LEVEL_GRID!r}.")
    if not set(vertices) <= set(classes):
        raise ValueError("Every selected-graph sequence needs a functional class.")
    values = {classes[dna] for dna in vertices}
    if values == {None}:
        return True
    if None in values or values - CLASSES:
        raise ValueError("Unknown or missing functional class.")
    return False


def build_merge_flow(
    tree: RidgeTree,
    *,
    state: str,
    vertices: Mapping[str, float],
    classes: Mapping[str, str | None],
    names: Mapping[str, str],
    pairs: pl.DataFrame,
    settings: Mapping[str, Any],
    nominal: RidgeTree | None = None,
    vertex_order: Mapping[str, int] | None = None,
) -> MergeFlow:
    """Sweep one state's merge tree top-down into components, events and levels.

    ``state`` is the landscape-summary state name (a ``_band`` suffix marks a
    band state). ``vertices`` maps every selected-graph sequence to its raw fitness
    and ``classes`` to its functional class; a Full state passes null classes, so
    its width basis is every sequence. Uncertain sequences count as non-functional.
    ``names`` maps summit DNA to amino-acid sequence. ``pairs`` is the saved
    persistence table of the swept tree; the elder cross-check raises ``ValueError``
    on any disagreement. ``vertex_order`` maps sequences to ``vertex_index``; when
    omitted, equal-summit ties break by DNA sequence.

    A band state sweeps its ``nominal`` tree, so counts and names never change, and
    ``tree`` (the band tree) only decides which components survive: those whose
    elder summit is not a band-tree leaf are pruned as ``band_absent``. A nominal
    state is pruned as ``below_K`` when its largest functional count is under K.
    """
    band = state.endswith("_band")
    population = state.removesuffix("_band")
    tree_state = "band" if band else "nominal"
    if band and nominal is None:
        raise ValueError("A band state needs its nominal tree.")
    width_all = _check_input(vertices, classes, settings)
    threshold = int(settings["min_functional_count_K"])
    swept = nominal if band else tree
    structure = _structure(swept)
    scalar, lower, uppers = structure["scalar"], structure["lower"], structure["uppers"]
    members_on = structure["members"]
    for node, value in scalar.items():
        if node not in vertices or not math.isclose(
            value, vertices[node], rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"Tree scalar of {node} differs from the graph raw fitness."
            )
    unknown = {
        dna for dna in chain.from_iterable(members_on.values()) if dna not in vertices
    }
    if unknown:
        raise ValueError(f"{len(unknown)} branch members are absent from the graph.")

    def counted(dna: str) -> bool:
        return width_all or classes[dna] == "functional"

    def rank(dna: str) -> tuple[float, Any]:
        order = dna if vertex_order is None else vertex_order[dna]
        return (-scalar[dna], order)

    # Sweep outward from the root, then walk back so every upper node precedes its
    # lower node (descending scalar, without relying on distinct scalars).
    outward = [structure["root"]]
    for node in outward:
        outward.extend(uppers.get(node, ()))
    nodes = outward[::-1]
    summit: dict[str, str] = {}
    for node in nodes:
        if node in uppers:
            summit[node] = min((summit[c] for c in uppers[node]), key=rank)
        else:
            summit[node] = node
    cid = {node: f"{summit[node]}@{node}" for node in scalar}

    # Sorted fitness lists of the cumulative members (all, counted) per arc.
    all_fit: dict[str, list[float]] = {}
    fun_fit: dict[str, list[float]] = {}
    own: dict[str, list[str]] = {}
    for node in nodes:
        mine = [node, *members_on.get(node, ())]
        own[node] = mine
        everything = [vertices[dna] for dna in mine]
        functional = [vertices[dna] for dna in mine if counted(dna)]
        for child in uppers.get(node, ()):
            everything.extend(all_fit[child])
            functional.extend(fun_fit[child])
        all_fit[node] = sorted(everything)
        fun_fit[node] = sorted(functional)

    def count_at(values: list[float], level: float) -> int:
        return len(values) - bisect_left(values, level)

    # Elder cross-check against the saved persistence pairs.
    saved = _pair_set(pairs)
    expected: set[tuple[str, str]] = set()
    younger_of: dict[str, list[str]] = {}
    for node, children in uppers.items():
        ranked = sorted(children, key=lambda c: rank(summit[c]))
        younger_of[node] = ranked[1:]
        for child in ranked[1:]:
            expected.add((summit[child], node))
    missing = expected - saved
    leftover = saved - expected
    root_summit = summit[structure["root"]]
    if missing or len(leftover) > 1 or any(b != root_summit for b, _ in leftover):
        raise ValueError(
            "Persistence pairs disagree with the elder rule: "
            f"{len(missing)} expected pairs missing, "
            f"{len(leftover)} unexplained pairs."
        )
    elder_check = {"pairs": len(saved), "agree": len(saved)}

    band_leaves = _leaves(_structure(tree)) if band else set()
    nominal_leaves = _leaves(structure)

    def count_full(node: str) -> tuple[int, int]:
        level = scalar[lower[node]] if node in lower else min(all_fit[node])
        return count_at(fun_fit[node], level), count_at(all_fit[node], level)

    rows: dict[str, dict[str, Any]] = {}
    for node in nodes:
        low = lower.get(node)
        functional_count, total_count = count_full(node)
        rows[node] = {
            "node": node,
            "component_id": cid[node],
            "low": low,
            "functional": functional_count,
            "total": total_count,
        }
    reason: dict[str, str | None] = {}
    for node in nodes:
        if band:
            absent = summit[node] not in band_leaves
            reason[node] = "band_absent" if absent else None
        else:
            below = rows[node]["functional"] < threshold
            reason[node] = "below_K" if below else None

    components = []
    for node in nodes:
        low = lower.get(node)
        fun_own = [vertices[dna] for dna in own[node] if counted(dna)]
        components.append(
            {
                "population": population,
                "tree_state": tree_state,
                "width_basis": "all" if width_all else "functional",
                "component_id": cid[node],
                "summit_dna_sequence": summit[node],
                "summit_aa_sequence": names.get(summit[node]),
                "summit_raw_fitness": float(vertices[summit[node]]),
                "birth_dna_sequence": node,
                "birth_fitness": float(scalar[node]),
                "death_dna_sequence": low,
                "death_fitness": None if low is None else float(scalar[low]),
                "parent_component_id": None if low is None else cid[low],
                "is_leaf": node not in uppers,
                "own_member_count": len(own[node]),
                "own_functional_count": len(fun_own),
                "total_count_at_death": rows[node]["total"],
                "functional_count_at_death": rows[node]["functional"],
                "max_functional_count": rows[node]["functional"],
                "pruned": reason[node] is not None,
                "prune_reason": reason[node],
                "functional_fitness_mean": (
                    sum(fun_fit[node]) / len(fun_fit[node]) if fun_fit[node] else None
                ),
                "functional_fitness_min": min(fun_fit[node]) if fun_fit[node] else None,
                "functional_fitness_max": max(fun_fit[node]) if fun_fit[node] else None,
            }
        )

    def stub(node: str) -> tuple[int, int]:
        """Mass and component count of a pruned arc and its pruned subtree."""
        return rows[node]["functional"], 1 + sum(
            stub(c)[1] for c in uppers.get(node, ()) if reason[c]
        )

    events = []
    for node in nodes:
        children = uppers.get(node)
        if not children or len(children) < 2:
            continue
        ranked = sorted(children, key=lambda c: rank(summit[c]))
        by_size = sorted(children, key=lambda c: (-rows[c]["functional"], cid[c]))
        pruned_children = [c for c in children if reason[c]]
        events.append(
            {
                "population": population,
                "tree_state": tree_state,
                "saddle_dna_sequence": node,
                "saddle_fitness": float(scalar[node]),
                "elder_component_id": cid[ranked[0]],
                "elder_summit_dna_sequence": summit[ranked[0]],
                "younger_component_ids": [cid[c] for c in ranked[1:]],
                "child_count": len(children),
                "balanced_merge": rows[by_size[1]]["functional"],
                "smaller_child_total_count": rows[by_size[1]]["total"],
                "saddle_drop": float(vertices[summit[ranked[1]]] - scalar[node]),
                "largest_child_functional_count": rows[by_size[0]]["functional"],
                "stub_functional_count": sum(stub(c)[0] for c in pruned_children),
                "stub_component_count": sum(stub(c)[1] for c in pruned_children),
                "persistence_pair_agrees": all(
                    (summit[c], node) in saved for c in ranked[1:]
                ),
            }
        )

    # Exact breakpoints: critical-node scalars and every member fitness.
    breakpoints = sorted(
        {*scalar.values(), *(vertices[d] for d in chain.from_iterable(own.values()))}
    )
    level_rows = []
    for node in nodes:
        if reason[node]:
            continue
        top = scalar[node]
        bottom = scalar[lower[node]] if node in lower else breakpoints[0]
        start = bisect_left(breakpoints, bottom)
        for level in breakpoints[start:]:
            if level > top:
                break
            level_rows.append(
                {
                    "population": population,
                    "tree_state": tree_state,
                    "component_id": cid[node],
                    "level_fitness": float(level),
                    "functional_count": count_at(fun_fit[node], level),
                    "total_count": count_at(all_fit[node], level),
                }
            )

    class_counts: dict[str, int] = {}
    for dna in vertices:
        key = "all" if width_all else str(classes[dna])
        class_counts[key] = class_counts.get(key, 0) + 1
    sidecar: dict[str, Any] = {
        "population": population,
        "tree_state": tree_state,
        "width_basis": "all" if width_all else "functional",
        "settings": {"min_functional_count_K": threshold, "level_grid": LEVEL_GRID},
        "elder_check": elder_check,
        "uncertain_policy": "merged_with_non_functional",
        "width_caveat": WIDTH_CAVEAT,
        "swept_tree": "nominal",
        "tie_break": "dna_sequence" if vertex_order is None else "vertex_index",
        "band_absent_summits": (sorted(nominal_leaves - band_leaves) if band else []),
        "class_counts": class_counts,
        "graph_vertex_count": len(vertices),
        "swept_member_count": len(all_fit[structure["root"]]),
    }
    return MergeFlow(
        pl.DataFrame(components, schema=COMPONENTS_SCHEMA),
        pl.DataFrame(events, schema=EVENTS_SCHEMA),
        pl.DataFrame(level_rows, schema=LEVELS_SCHEMA),
        sidecar,
    )
