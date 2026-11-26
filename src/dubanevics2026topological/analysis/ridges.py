"""KEM and TCL superlevel ridge populations of one saved merge tree, keyed by DNA.

A ridge is every sequence in the superlevel component of its nominated summit
strictly above the common saddle of the two summits (GLOSSARY.md, "KEM/TCL ridge
population"). The common saddle is the lowest common ancestor of the summits in
that tree, so it is found per tree and never assumed. The saddle, the trunk below
it and every other summit-side component belong to no ridge and carry a null
``ridge_name`` in the membership table.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

RIDGE_NAMES = ("kem", "tcl")
CLASSES = frozenset({"functional", "uncertain", "non_functional"})
SUMMARY_FILE = "ridges.parquet"
MEMBERS_FILE = "members.parquet"
SIDECAR_FILE = "summary.json"

SUMMARY_SCHEMA = {
    "population": pl.String,
    "tree_state": pl.String,
    "ridge_name": pl.String,
    "summit_dna_sequence": pl.String,
    "summit_raw_fitness": pl.Float64,
    "common_saddle_dna_sequence": pl.String,
    "common_saddle_fitness": pl.Float64,
    "member_count": pl.Int64,
    "functional_count": pl.Int64,
    "functional_share": pl.Float64,
}
MEMBERS_SCHEMA = {
    "population": pl.String,
    "tree_state": pl.String,
    "ridge_name": pl.String,
    "summit_dna_sequence": pl.String,
    "member_dna_sequence": pl.String,
    "raw_fitness": pl.Float64,
    "functional_class": pl.String,
}


@dataclass(frozen=True)
class RidgeTree:
    """The saved tables of one merge tree, as the ridge rule reads them.

    ``nodes`` holds ``tree_node_identity`` and ``scalar``; ``edges`` holds
    ``source_tree_node_identity`` (upper) and ``target_tree_node_identity``
    (lower); ``branch_members`` holds the regular vertices on each edge with
    ``source_tree_node_identity``, ``target_tree_node_identity`` and
    ``dna_sequence``. The optional ``root_members`` holds the sequences assigned to
    the root (``dna_sequence``), which no branch carries.
    """

    nodes: pl.DataFrame
    edges: pl.DataFrame
    branch_members: pl.DataFrame
    root_members: pl.DataFrame | None = None


@dataclass(frozen=True)
class Ridges:
    """The ridge summary and DNA-keyed membership of one population and state."""

    summary: pl.DataFrame
    members: pl.DataFrame
    sidecar: dict[str, Any]

    def save(self, directory: Path) -> None:
        """Write the three artifacts of a saved ``ridges`` product."""
        self.summary.write_parquet(directory / SUMMARY_FILE)
        self.members.write_parquet(directory / MEMBERS_FILE)
        (directory / SIDECAR_FILE).write_text(
            json.dumps(self.sidecar, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_artifacts(cls, artifacts: Mapping[str, Path]) -> Ridges:
        """Read the artifacts of a saved ``ridges`` product."""
        return cls(
            pl.read_parquet(artifacts[SUMMARY_FILE]),
            pl.read_parquet(artifacts[MEMBERS_FILE]),
            json.loads(Path(artifacts[SIDECAR_FILE]).read_text(encoding="utf-8")),
        )


def _find_ridges(tree: RidgeTree, summits: Mapping[str, str]) -> dict[str, Any]:
    """Locate the summits, their common saddle and the ridge sequences of one tree."""
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
        lower[upper] = low
        uppers.setdefault(low, []).append(upper)
    missing = {
        name: dna for name, dna in summits.items() if dna not in scalar or dna in uppers
    }
    if missing:
        return {
            "missing": missing,
            "scalar": scalar,
            "saddle": None,
            "ridges": {name: set() for name in summits},
            "trunk": set(),
        }

    def path(node: str) -> list[str]:
        nodes = [node]
        while nodes[-1] in lower:
            nodes.append(lower[nodes[-1]])
        return nodes

    paths = {name: path(dna) for name, dna in summits.items()}
    on_second = set(paths["tcl"])
    saddle = next((node for node in paths["kem"] if node in on_second), None)
    if saddle is None or saddle in summits.values():
        raise ValueError("KEM and TCL summits do not have a distinct shared saddle.")
    edge_members: dict[str, list[str]] = {}
    for upper, dna in zip(
        tree.branch_members["source_tree_node_identity"].to_list(),
        tree.branch_members["dna_sequence"].to_list(),
        strict=True,
    ):
        edge_members.setdefault(upper, []).append(dna)
    ridges = {}
    for name in summits:
        subroot = paths[name][paths[name].index(saddle) - 1]
        nodes, pending = set(), [subroot]
        while pending:
            node = pending.pop()
            if node not in nodes:
                nodes.add(node)
                pending.extend(uppers.get(node, ()))
        sequences = set(nodes)
        for node in nodes:
            sequences.update(edge_members.get(node, ()))
        if any(scalar[node] <= scalar[saddle] for node in nodes):
            raise ValueError(f"{name} ridge reaches or falls below the common saddle.")
        ridges[name] = sequences
    trunk = set(path(saddle)[1:])
    for node in path(saddle):
        trunk.update(edge_members.get(node, ()))
    if ridges["kem"] & ridges["tcl"]:
        raise ValueError("KEM and TCL ridge populations overlap.")
    return {
        "missing": {},
        "scalar": scalar,
        "saddle": saddle,
        "ridges": ridges,
        "trunk": trunk,
    }


def build_ridges(
    tree: RidgeTree,
    *,
    population: str,
    tree_state: str,
    summits: Mapping[str, str],
    vertices: Mapping[str, float],
    classes: Mapping[str, str],
    nominal: RidgeTree | None = None,
) -> Ridges:
    """Summarise and key the KEM and TCL ridges of one population and state.

    ``vertices`` maps every selected-graph sequence to its raw fitness and
    ``classes`` to its functional class; the membership table lists them all. A
    band state passes its ``nominal`` tree so the sidecar records which ridge
    sequences the band tree kept and whether the common saddle moved. When a
    summit is not a maximum of the tree (for instance absorbed in a band tree) the
    state has no common saddle and so no ridge: the membership table still lists
    every vertex with a null ``ridge_name``, the summary has a zero-member row
    with null saddle fields for each summit that survives (none for an absent
    one), and the sidecar names the absent summits.
    """
    if tuple(summits) != RIDGE_NAMES:
        raise ValueError("Summits must be given for kem then tcl.")
    if tree_state not in {"nominal", "band"}:
        raise ValueError("tree_state must be nominal or band.")
    if set(classes.values()) - CLASSES:
        raise ValueError("Unknown functional class.")
    if not set(vertices) <= set(classes):
        raise ValueError("Every selected-graph sequence needs a functional class.")
    found = _find_ridges(tree, summits)
    scalar, saddle, ridges = found["scalar"], found["saddle"], found["ridges"]
    sidecar: dict[str, Any] = {
        "population": population,
        "tree_state": tree_state,
        "summit_dna_sequences": dict(summits),
        "graph_vertex_count": len(vertices),
        "graph_functional_count": sum(classes[dna] == "functional" for dna in vertices),
        "ridges_present": saddle is not None,
        "common_saddle_dna_sequence": saddle,
    }
    if found["missing"]:
        sidecar["reason"] = {
            name: "summit absent from the tree or not a maximum"
            for name in found["missing"]
        }
    # The tree's scalar is the raw fitness of its sequence; a mismatch means the
    # tree and the graph describe different data.
    for dna in [*(summits[n] for n in RIDGE_NAMES if n not in found["missing"])] + (
        [saddle] if saddle else []
    ):
        if dna not in vertices or not math.isclose(
            scalar[dna], vertices[dna], rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"Tree scalar of {dna} differs from the graph raw fitness."
            )
    outside = set().union(*ridges.values()) - set(vertices)
    if outside:
        raise ValueError(
            f"{len(outside)} ridge sequences are absent from the selected graph."
        )
    total_functional = sidecar["graph_functional_count"]
    owner = {dna: name for name, dna_set in ridges.items() for dna in dna_set}
    summary = []
    for name in RIDGE_NAMES:
        if name in found["missing"]:
            continue
        functional = sum(classes[dna] == "functional" for dna in ridges[name])
        summary.append(
            {
                "population": population,
                "tree_state": tree_state,
                "ridge_name": name,
                "summit_dna_sequence": summits[name],
                "summit_raw_fitness": float(vertices[summits[name]]),
                "common_saddle_dna_sequence": saddle,
                "common_saddle_fitness": None
                if saddle is None
                else float(vertices[saddle]),
                "member_count": len(ridges[name]),
                "functional_count": functional,
                "functional_share": 100.0 * functional / total_functional
                if total_functional
                else None,
            }
        )
    members = [
        {
            "population": population,
            "tree_state": tree_state,
            "ridge_name": owner.get(dna),
            "summit_dna_sequence": summits[owner[dna]] if dna in owner else None,
            "member_dna_sequence": dna,
            "raw_fitness": float(fitness),
            "functional_class": classes[dna],
        }
        for dna, fitness in sorted(vertices.items())
    ]
    trunk = found["trunk"] & set(vertices)
    sidecar["composition"] = {
        "ridge": len(owner),
        "common_saddle": 0 if saddle is None else 1,
        "trunk": len(trunk - {saddle}),
        "other": len(vertices) - len(owner) - len(trunk | {saddle} - {None}),
    }
    if nominal is not None:
        before = _find_ridges(nominal, summits)
        if before["saddle"] is None:
            sidecar["nominal"] = {"ridges_present": False}
        else:
            sidecar["nominal"] = {
                "ridges_present": True,
                "common_saddle_dna_sequence": before["saddle"],
                "saddle_shifted": before["saddle"] != saddle,
                "ridges": {
                    name: {
                        "nominal_member_count": len(before["ridges"][name]),
                        "band_member_count": len(ridges[name]),
                        "retained_count": len(before["ridges"][name] & ridges[name]),
                    }
                    for name in RIDGE_NAMES
                },
            }
    return Ridges(
        pl.DataFrame(summary, schema=SUMMARY_SCHEMA),
        pl.DataFrame(members, schema=MEMBERS_SCHEMA),
        sidecar,
    )
