"""Sequence-keyed biological evidence from existing split-tree checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from . import _write_json_atomic

# Input AA states preserve the Papkou checkpoint alphabet.
STOP_CODON_SYMBOLS = frozenset({"*", "%", "^", "&"})


@dataclass(frozen=True)
class BiologicalAnnotations:
    """Inspectable annotations, including peak and adjacent-saddle comparisons."""

    branches: pl.DataFrame
    pairs: pl.DataFrame
    critical_points: pl.DataFrame
    summary: dict[str, Any]


def _endpoint(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    dna = payload.get("dna_sequence")
    aa = payload.get("aa_sequence")
    if not isinstance(dna, str) or not dna or not isinstance(aa, str) or not aa:
        raise ValueError(
            "Biological annotations require DNA and AA sequence identities."
        )
    count = sum(symbol in STOP_CODON_SYMBOLS for symbol in aa)
    return {
        f"{prefix}_sequence": dna,
        f"{prefix}_aa_sequence": aa,
        f"{prefix}_has_stop_codon": count > 0,
        f"{prefix}_stop_codon_count": count,
    }


def _branch_vertex_count(row: dict[str, Any]) -> int:
    """Return a branch's vertex count from either Topographer column spelling.

    Current Topographer branch tables carry ``branch_vertex_count``, which already
    includes the branch's upper critical vertex. Tables written before the rename
    carry ``branch_member_count``, which excludes it, so one is added.
    """
    if "branch_vertex_count" in row:
        return int(row["branch_vertex_count"])
    if "branch_member_count" in row:
        return int(row["branch_member_count"]) + 1
    raise ValueError(
        "The split-tree branch table has neither branch_vertex_count nor "
        "branch_member_count."
    )


def _build_biological_annotations(
    analysis: Any, *, landscape: str = "study_clamped"
) -> BiologicalAnnotations:
    """Annotate cached branches, critical nodes, and pairs without filtering science.

    Branch direction comes from the split-tree topology, whose edges descend from
    upper endpoints to lower endpoints. Synonymous equality preserves the input
    AA alphabet, including distinct stop symbols. Peak annotations compare
    the peak to its first adjacent saddle; a root-only endpoint that is not a
    saddle is excluded. Paired death is retained separately and never
    substituted for that adjacent saddle.
    """
    tree_analysis = analysis.trees["split"]
    tree = tree_analysis.tree
    nodes = {int(index): dict(tree[index]) for index in tree.node_indices()}
    for payload in nodes.values():
        _endpoint(payload, "node")
    if len({payload["dna_sequence"] for payload in nodes.values()}) != len(nodes):
        raise ValueError("Critical DNA sequence identities must be unique.")
    coordinates = {
        int(row["tree_node_index"]): row
        for row in tree_analysis.plot_data.node_table.to_dicts()
    }
    branch_rows = []
    descending: dict[int, tuple[int, int]] = {}
    for row in tree_analysis.branch_table.to_dicts():
        upper = int(row["source_tree_node_index"])
        lower = int(row["target_tree_node_index"])
        if not tree.has_edge(upper, lower):
            raise ValueError(
                "Branch table direction must match descending split-tree edges."
            )
        if upper in descending:
            raise ValueError(
                "Split-tree upper endpoint has multiple descending branches."
            )
        count = _branch_vertex_count(row)
        if count < 1:
            raise ValueError("Branch vertex count must be positive.")
        descending[upper] = (lower, count)
        branch_rows.append(
            {
                "landscape": landscape,
                **_endpoint(nodes[upper], "upper"),
                **_endpoint(nodes[lower], "lower"),
                "upper_scalar": float(nodes[upper]["scalar"]),
                "lower_scalar": float(nodes[lower]["scalar"]),
                "branch_vertex_count": count,
                "is_peak_branch": tree.in_degree(upper) == 0,
                "synonymous_endpoints": nodes[upper]["aa_sequence"]
                == nodes[lower]["aa_sequence"],
            }
        )
    pair_rows = []
    for row in tree_analysis.persistence_table.to_dicts():
        birth = int(row["birth_tree_node_index"])
        death = int(row["death_tree_node_index"])
        adjacent = descending.get(birth)
        if adjacent is None and len(nodes) != 1:
            raise ValueError("Persistence birth has no adjacent descending branch.")
        lower, count = adjacent if adjacent else (None, 1)
        saddle = lower if lower is not None and tree.in_degree(lower) > 1 else None
        saddle_fields = (
            _endpoint(nodes[saddle], "adjacent_saddle")
            if saddle is not None
            else {
                "adjacent_saddle_sequence": None,
                "adjacent_saddle_aa_sequence": None,
                "adjacent_saddle_has_stop_codon": None,
                "adjacent_saddle_stop_codon_count": None,
            }
        )
        pair_rows.append(
            {
                "landscape": landscape,
                "pair_id": str(row.get("pair_id", nodes[birth]["dna_sequence"])),
                "pair_kind": str(row["pair_kind"]),
                **_endpoint(nodes[birth], "birth"),
                **_endpoint(nodes[death], "death"),
                **saddle_fields,
                "adjacent_lower_sequence": nodes[lower]["dna_sequence"]
                if lower is not None
                else None,
                "adjacent_lower_is_root": lower is not None
                and tree.out_degree(lower) == 0,
                "birth_scalar": float(row["birth_scalar"]),
                "death_scalar": float(row["death_scalar"]),
                "persistence": abs(
                    float(row["birth_scalar"]) - float(row["death_scalar"])
                ),
                "adjacent_saddle_scalar": float(nodes[saddle]["scalar"])
                if saddle is not None
                else None,
                "branch_vertex_count": count,
                "synonymous_endpoints": saddle is not None
                and nodes[birth]["aa_sequence"] == nodes[saddle]["aa_sequence"],
                "paired_synonymous_endpoints": nodes[birth]["aa_sequence"]
                == nodes[death]["aa_sequence"],
                "has_stop_codon_endpoint": _endpoint(nodes[birth], "birth")[
                    "birth_has_stop_codon"
                ]
                or _endpoint(nodes[death], "death")["death_has_stop_codon"],
            }
        )
    critical_rows = []
    for index, payload in nodes.items():
        coordinate = coordinates[index]
        endpoint = _endpoint(payload, "node")
        critical_rows.append(
            {
                "landscape": landscape,
                "dna_sequence": endpoint["node_sequence"],
                "aa_sequence": endpoint["node_aa_sequence"],
                "has_stop_codon": endpoint["node_has_stop_codon"],
                "stop_codon_count": endpoint["node_stop_codon_count"],
                "scalar": float(payload["scalar"]),
                "event": str(payload.get("event", "unknown")),
                "is_peak": tree.in_degree(index) == 0,
                "is_root": tree.out_degree(index) == 0,
                "x": float(coordinate["x"]),
                "y": float(coordinate["y"]),
            }
        )
    branches = (
        pl.DataFrame(branch_rows)
        if branch_rows
        else pl.DataFrame(
            schema={
                "landscape": pl.String,
                "upper_sequence": pl.String,
                "lower_sequence": pl.String,
                "branch_vertex_count": pl.Int64,
                "is_peak_branch": pl.Boolean,
                "synonymous_endpoints": pl.Boolean,
            }
        )
    )
    pairs = pl.DataFrame(pair_rows)
    critical = pl.DataFrame(critical_rows).sort("dna_sequence")
    peaks = critical.filter(pl.col("is_peak"))
    summary = {
        "landscape": landscape,
        "stop_evidence_population": "critical_nodes",
        "synonymous_pair_definition": "birth_to_adjacent_descending_saddle",
        "amino_acid_source": "cached_input_aa_sequence",
        "stop_alphabet": sorted(STOP_CODON_SYMBOLS),
        "distinct_stop_states_equal": False,
        "critical_node_count": critical.height,
        "critical_stop_sequence_count": critical.filter(
            pl.col("has_stop_codon")
        ).height,
        "peak_count": peaks.height,
        "stop_codon_peak_count": peaks.filter(pl.col("has_stop_codon")).height,
        "peak_denominator": peaks.height,
        "singleton_peak_count": pairs.filter(pl.col("branch_vertex_count") == 1).height,
        "multi_sequence_peak_count": pairs.filter(
            pl.col("branch_vertex_count") > 1
        ).height,
        "synonymous_endpoint_peak_count": pairs.filter(
            pl.col("synonymous_endpoints")
        ).height,
        "adjacent_saddle_peak_denominator": pairs.filter(
            pl.col("adjacent_saddle_sequence").is_not_null()
        ).height,
        "pair_count": pairs.height,
        "branch_count": branches.height,
        "represented_vertex_count": sum(count for _, count in descending.values()) + 1,
    }
    summary["stop_endpoint_pair_count"] = pairs.filter(
        pl.col("has_stop_codon_endpoint")
    ).height
    summary["pair_denominator"] = pairs.height
    for category, predicate in (
        ("singleton", pl.col("branch_vertex_count") == 1),
        ("multi_sequence", pl.col("branch_vertex_count") > 1),
    ):
        group = pairs.filter(predicate)
        summary[f"{category}_peak_denominator"] = group.height
        summary[f"{category}_stop_codon_peak_count"] = group.filter(
            pl.col("birth_has_stop_codon")
        ).height
        summary[f"{category}_synonymous_endpoint_peak_count"] = group.filter(
            pl.col("synonymous_endpoints")
        ).height
        summary[f"{category}_adjacent_saddle_peak_denominator"] = group.filter(
            pl.col("adjacent_saddle_sequence").is_not_null()
        ).height
    if pairs["birth_sequence"].n_unique() != pairs.height or set(
        pairs["birth_sequence"]
    ) != set(peaks["dna_sequence"]):
        raise ValueError("Persistence births must identify every peak exactly once.")
    return BiologicalAnnotations(branches, pairs, critical, summary)


def hill_statistics(pairs: pl.DataFrame) -> dict[str, int]:
    """The hill sizes of an annotated split tree: total and largest sequence counts.

    A hill is a peak with the members of its terminal branch, the adjacent saddle
    excluded; its size is the pair's ``branch_vertex_count`` (the leaf figures check
    that equality). ``hill_sequence_count`` is the sum over all hills and
    ``largest_hill_sequence_count`` the maximum.
    """
    if not pairs.height:
        raise ValueError("A split tree with no persistence pair has no hills.")
    sizes = pairs["branch_vertex_count"]
    return {
        "hill_sequence_count": int(sizes.sum()),
        "largest_hill_sequence_count": int(sizes.max()),
    }


def _export_biological_annotations(
    annotations: BiologicalAnnotations, directory: Path
) -> list[Path]:
    """Save sequence-keyed CSV/Parquet evidence and explicit count denominators."""
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in ("branches", "pairs", "critical_points"):
        frame = getattr(annotations, name)
        for extension in ("csv", "parquet"):
            target = directory / f"biological_annotation_{name}.{extension}"
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                if extension == "csv":
                    frame.write_csv(temporary)
                else:
                    frame.write_parquet(temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(target)
    counts = directory / "biological_annotation_counts.json"
    _write_json_atomic(counts, annotations.summary)
    return [*outputs, counts]
