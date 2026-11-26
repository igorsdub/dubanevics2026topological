"""Prepare reproducible population and site evidence for the codon-entropy figure."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import tomllib
from collections import Counter
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import polars as pl

from dubanevics2026topological.analysis import OUTPUT, ROOT
from dubanevics2026topological.analysis.clamped_cache import (
    load_cached_clamped_comparison,
)

POPULATION_SCHEMA = 3
_POPULATION_KEYS = ("core", "kem", "tcl")
_STOP_TRANSLATION = {"TAA": "%", "TAG": "^", "TGA": "&"}
_CODON_TRANSLATION = {
    codon: amino_acid
    for amino_acid, codons in {
        "F": "TTT TTC",
        "L": "TTA TTG CTT CTC CTA CTG",
        "I": "ATT ATC ATA",
        "M": "ATG",
        "V": "GTT GTC GTA GTG",
        "S": "TCT TCC TCA TCG AGT AGC",
        "P": "CCT CCC CCA CCG",
        "T": "ACT ACC ACA ACG",
        "A": "GCT GCC GCA GCG",
        "Y": "TAT TAC",
        "H": "CAT CAC",
        "Q": "CAA CAG",
        "N": "AAT AAC",
        "K": "AAA AAG",
        "D": "GAT GAC",
        "E": "GAA GAG",
        "C": "TGT TGC",
        "W": "TGG",
        "R": "CGT CGC CGA CGG AGA AGG",
        "G": "GGT GGC GGA GGG",
        **{value: key for key, value in _STOP_TRANSLATION.items()},
    }.items()
    for codon in codons.split()
}
# Public name for other analysis modules; the underscored one stays for existing callers.
CODON_TRANSLATION = _CODON_TRANSLATION


def _read_figure_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    with (ROOT / "config" / "analysis.toml").open("rb") as stream:
        return tomllib.load(stream).get("codon_entropy", {})


def get_published_functional_threshold(
    config: Mapping[str, Any] | None = None,
) -> float:
    """Read the declared codon-entropy functional threshold."""
    if config is not None and "functional_threshold" in config:
        return float(config["functional_threshold"])
    with (ROOT / "config" / "analysis.toml").open("rb") as stream:
        settings = tomllib.load(stream)
    figure_settings = settings.get("codon_entropy", {})
    return float(
        figure_settings.get(
            "functional_threshold",
            settings["clamped"]["published_functional_threshold"],
        )
    )


def _sequence_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Return stable sequence rows with raw fitness and one-based positions."""
    required = {"dna_sequence", "aa_sequence", "scalar"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Tree location table is missing columns: {sorted(missing)}")
    selected = frame.select("dna_sequence", "aa_sequence", "scalar").unique(
        "dna_sequence", keep="first", maintain_order=True
    )
    rows = []
    for dna, aa, scalar in selected.iter_rows():
        dna = str(dna)
        aa = str(aa)
        if len(dna) != 9 or len(aa) != 3:
            raise ValueError(
                f"Codon-entropy analysis requires aligned 9-nt/3-aa sequences; got {dna!r}/{aa!r}."
            )
        row: dict[str, Any] = {
            "dna_sequence": dna,
            "aa_sequence": aa,
            "fitness": float(scalar),
            "raw_fitness": float(scalar),
        }
        for pos in range(3):
            row[f"codon{pos + 1}"] = dna[pos * 3 : (pos + 1) * 3]
            row[f"aa{pos + 1}"] = aa[pos]
        for pos, nt in enumerate(dna, start=1):
            row[f"dna{pos}"] = nt
        rows.append(row)
    schema = {
        "dna_sequence": pl.String,
        "aa_sequence": pl.String,
        "fitness": pl.Float64,
        "raw_fitness": pl.Float64,
        **{f"codon{i}": pl.String for i in range(1, 4)},
        **{f"aa{i}": pl.String for i in range(1, 4)},
        **{f"dna{i}": pl.String for i in range(1, 10)},
    }
    return pl.DataFrame(rows, schema=schema).sort("fitness", descending=True)


def _validate_sequence_translation(
    frame: pl.DataFrame, *, dna_alphabet: set[str], aa_alphabet: set[str]
) -> None:
    for row in frame.iter_rows(named=True):
        if set(row["dna_sequence"]) - dna_alphabet:
            raise ValueError(
                f"DNA sequence {row['dna_sequence']} contains a state outside the declared alphabet."
            )
        if aa_alphabet and set(row["aa_sequence"]) - aa_alphabet:
            raise ValueError(
                f"AA sequence {row['aa_sequence']} contains a state outside the declared alphabet."
            )
        for position in range(3):
            codon = row[f"codon{position + 1}"]
            expected_aa = _CODON_TRANSLATION.get(codon)
            if expected_aa is None:
                raise ValueError(
                    f"Unsupported codon {codon!r} in codon-entropy DNA sequence."
                )
            if row[f"aa{position + 1}"] != expected_aa:
                raise ValueError(
                    f"Codon {codon} translates to {expected_aa!r}, not "
                    f"{row[f'aa{position + 1}']!r}."
                )


def _descendants(tree: Any, root: int) -> set[int]:
    """Return a tree node and all nodes upstream of it, toward the peaks."""
    visited: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(src for src, _, _ in tree.in_edges(node))
    return visited


def ancestor_path(tree: Any, leaf: int) -> list[int]:
    """The nodes from ``leaf`` down to the root of a split tree, ``leaf`` first."""
    path = [leaf]
    current = leaf
    while tree.out_degree(current):
        edges = tree.out_edges(current)
        current = edges[0][1]
        path.append(current)
    return path


def lowest_common_ancestor(tree: Any, first: int, second: int) -> int | None:
    """The first node on ``first``'s path to the root that is also on ``second``'s.

    In a split tree this is the saddle where the two summits' branches merge. It
    is ``None`` only when the nodes lie in different trees. The result can be
    ``first`` or ``second`` itself when one is an ancestor of the other; callers
    that need a distinct shared saddle must check.
    """
    second_path = set(ancestor_path(tree, second))
    return next(
        (node for node in ancestor_path(tree, first) if node in second_path), None
    )


def _ridge_nodes(tree: Any, saddle: int, summit: int) -> tuple[set[int], int]:
    """Get the critical nodes on the summit side, excluding the saddle itself."""
    upstream = _descendants(tree, saddle)
    if summit not in upstream:
        raise ValueError("Configured summit is not above the common saddle.")
    saddle_inputs = [src for src, _, _ in tree.in_edges(saddle)]
    subroot = next(
        (node for node in saddle_inputs if summit in _descendants(tree, node)), None
    )
    if subroot is None:
        raise ValueError("Could not identify the ridge branch entering the saddle.")
    return _descendants(tree, subroot), subroot


def _frames_for_populations(
    populations: Mapping[str, pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build site-frequency, site-entropy, and sequence-assignment evidence."""
    assignments: dict[str, dict[str, Any]] = {}
    frequencies: list[dict[str, Any]] = []
    entropies: list[dict[str, Any]] = []
    for population_id, frame in populations.items():
        sequences = frame.to_dicts()
        for row in sequences:
            dna = row["dna_sequence"]
            assignment = assignments.setdefault(
                dna,
                {
                    "dna_sequence": dna,
                    "aa_sequence": row["aa_sequence"],
                    "raw_fitness": row["raw_fitness"],
                    "fitness": row["fitness"],
                    "is_functional_core": row["is_functional_core"],
                    "population_membership": [],
                },
            )
            assignment["population_membership"].append(population_id)
        groups = [("all", sequences)]
        for group in ("functional", "fringe"):
            groups.append(
                (
                    group,
                    [
                        row
                        for row in sequences
                        if bool(row["is_functional_core"]) == (group == "functional")
                    ],
                )
            )
        for fitness_group, group_rows in groups:
            site_states: list[tuple[str, int, list[str]]] = []
            for site in range(1, 10):
                site_states.append(
                    ("dna", site, [str(row[f"dna{site}"]) for row in group_rows])
                )
            for site in range(1, 4):
                site_states.append(
                    ("amino_acid", site, [str(row[f"aa{site}"]) for row in group_rows])
                )
            for molecule, site, states in site_states:
                counts = Counter(states)
                n = len(states)
                entropy = (
                    -sum(
                        (count / n) * math.log2(count / n) for count in counts.values()
                    )
                    if n
                    else None
                )
                entropies.append(
                    {
                        "population_id": population_id,
                        "fitness_group": fitness_group,
                        "molecule": molecule,
                        "site_index": site,
                        "entropy_bits": entropy,
                        "state_count": len(counts),
                        "population_count": n,
                    }
                )
                for state, count in sorted(counts.items()):
                    frequencies.append(
                        {
                            "population_id": population_id,
                            "fitness_group": fitness_group,
                            "molecule": molecule,
                            "site_index": site,
                            "state": state,
                            "count": count,
                            "population_count": n,
                            "frequency": count / n if n else None,
                        }
                    )
    for item in assignments.values():
        members = item.pop("population_membership")
        ridge_members = [member for member in members if member in {"kem", "tcl"}]
        item["ridge_id"] = ridge_members[0] if ridge_members else "unassigned"
        item["membership_populations"] = ",".join(sorted(members))
    assignment_frame = (
        pl.DataFrame(list(assignments.values()))
        .with_columns(
            pl.col("dna_sequence").cast(pl.String),
            pl.col("aa_sequence").cast(pl.String),
            pl.col("raw_fitness").cast(pl.Float64),
            pl.col("fitness").cast(pl.Float64),
            pl.col("is_functional_core").cast(pl.Boolean),
            pl.col("ridge_id").cast(pl.String),
            pl.col("membership_populations").cast(pl.String),
        )
        .sort("dna_sequence")
    )
    frequency_frame = pl.DataFrame(
        frequencies,
        schema={
            "population_id": pl.String,
            "fitness_group": pl.String,
            "molecule": pl.String,
            "site_index": pl.Int64,
            "state": pl.String,
            "count": pl.Int64,
            "population_count": pl.Int64,
            "frequency": pl.Float64,
        },
    )
    entropy_frame = pl.DataFrame(
        entropies,
        schema={
            "population_id": pl.String,
            "fitness_group": pl.String,
            "molecule": pl.String,
            "site_index": pl.Int64,
            "entropy_bits": pl.Float64,
            "state_count": pl.Int64,
            "population_count": pl.Int64,
        },
    )
    return frequency_frame, entropy_frame, assignment_frame


def _claim_summary(
    populations: Mapping[str, pl.DataFrame],
    assignments: pl.DataFrame,
    threshold: float,
    settings: Mapping[str, Any],
) -> pl.DataFrame:
    core = populations["core"]
    functional = core.filter(pl.col("is_functional_core"))
    ridge_ids = assignments.filter(pl.col("ridge_id").is_in(["kem", "tcl"])).get_column(
        "dna_sequence"
    )
    in_ridges = functional.filter(pl.col("dna_sequence").is_in(ridge_ids.to_list()))
    denominator = functional.height
    numerator = in_ridges.height
    two_functional_ridges_share_minimum = float(
        settings.get("two_functional_ridges_share_minimum", 0.9)
    )
    rows: list[dict[str, Any]] = [
        {
            "claim_id": "functional_sequences_in_two_ridges",
            "numerator": numerator,
            "denominator": denominator,
            "fraction": numerator / denominator if denominator else None,
            "claim_threshold": two_functional_ridges_share_minimum,
            # ">=": a share equal to the configured minimum meets it (the registry's
            # ``ridges.two_functional_share.meets_minimum`` uses the same comparison).
            "claim_meets_minimum": bool(
                denominator
                and numerator / denominator >= two_functional_ridges_share_minimum
            ),
            "claim_exceeds_baseline": None,
            "definition": "functional Candidate-selected sequences assigned to KEM or TCL ridge / all functional Candidate-selected sequences",
            "threshold": threshold,
        }
    ]
    core_states = Counter(functional.get_column("aa2").to_list())
    core_n = denominator
    for ridge in ("kem", "tcl"):
        ridge_frame = populations[ridge]
        ridge_states = Counter(ridge_frame.get_column("aa2").to_list())
        ridge_n = ridge_frame.height
        for state in ("E", "D"):
            baseline = core_states[state] / core_n if core_n else 0.0
            observed = ridge_states[state] / ridge_n if ridge_n else None
            rows.append(
                {
                    "claim_id": f"{ridge}_aa_site_2_{state}_enrichment",
                    "numerator": ridge_states[state],
                    "denominator": ridge_n,
                    "fraction": observed,
                    "claim_threshold": baseline,
                    "claim_meets_minimum": None,
                    # Enrichment is strict: the ridge frequency must exceed the baseline.
                    "claim_exceeds_baseline": bool(
                        observed is not None and baseline and observed > baseline
                    ),
                    "definition": f"{state} frequency at amino-acid position 2 in {ridge} ridge; baseline is the functional Candidate-selected population",
                    "threshold": threshold,
                }
            )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "claim_meets_minimum": pl.Boolean,
            "claim_exceeds_baseline": pl.Boolean,
        },
    )


CLAIM_SUMMARY_COLUMN_DESCRIPTIONS = {
    "claim_meets_minimum": (
        "Share row only (null for enrichment rows): true when fraction >= "
        "claim_threshold, the configured minimum, so a share exactly equal to the "
        "minimum meets it."
    ),
    "claim_exceeds_baseline": (
        "Enrichment rows only (null for the share row): true when fraction > "
        "claim_threshold, the functional Candidate-selected baseline, so a ridge "
        "frequency exactly equal to the baseline is false."
    ),
}


def extract_codon_entropy_populations(
    *,
    threshold: float | None = None,
    output_dir: Path | str | None = None,
    save_files: bool = False,
    comparison: Mapping[str, Any] | None = None,
    candidate_view: Any | None = None,
    config: Mapping[str, Any] | None = None,
    functional_sequences: Collection[str] | None = None,
) -> dict[str, Any]:
    """Extract Candidate simplified-tree and mutually exclusive ridge populations.

    ``candidate_view`` is the selected-leaf view whose tree defines the populations
    (identity-selected by default; the band-simplified view for the bootstrap twin
    figure, which saves nothing). ``functional_sequences`` is the Functional class
    of the resolved Candidate selection under ``mixture_quantile``: a sequence is
    then in the functional core when it is in that set, whatever its raw fitness.
    Without it (``median_reflected``) the functional core is raw fitness at or above
    ``threshold``.
    """
    members = None if functional_sequences is None else frozenset(functional_sequences)
    settings = _read_figure_config(config)
    comparison = comparison or load_cached_clamped_comparison()
    if candidate_view is None:
        candidate_view = comparison.get("candidate_leaf_view")
    if candidate_view is None:
        raise ValueError(
            "Codon-entropy analysis requires the simplified Candidate tree view."
        )
    declared_threshold = comparison.get("thresholds", {}).get("candidate_clamped")
    if threshold is None:
        if declared_threshold is None:
            raise ValueError("Candidate functional threshold is unavailable.")
        threshold = float(declared_threshold)
    if declared_threshold is not None and not math.isclose(
        float(threshold), float(declared_threshold), abs_tol=1e-12
    ):
        raise ValueError(
            "Codon-entropy functional threshold must match the Candidate-selected landscape threshold."
        )
    if (
        settings.get(
            "candidate_population",
            "all_candidate_selected_graph_vertices_including_fringe",
        )
        != "all_candidate_selected_graph_vertices_including_fringe"
    ):
        raise ValueError("Unsupported codon-entropy Candidate population definition.")
    if (
        settings.get(
            "ridge_membership", "strictly_above_common_saddle_excluding_saddle"
        )
        != "strictly_above_common_saddle_excluding_saddle"
    ):
        raise ValueError("Unsupported codon-entropy ridge membership definition.")
    if set(settings.get("stop_codons", _STOP_TRANSLATION)) != set(_STOP_TRANSLATION):
        raise ValueError(
            "Codon-entropy analysis requires the distinct TAA, TAG, and TGA stop codons."
        )
    functional_core = (
        pl.col("fitness") >= threshold
        if members is None
        else pl.col("dna_sequence").is_in(members)
    )
    declared_aa = settings.get("aa_alphabet", [])
    if declared_aa and (
        len(declared_aa) != 23
        or not set(_STOP_TRANSLATION.values()) <= set(declared_aa)
    ):
        raise ValueError(
            "Codon-entropy amino-acid alphabet must declare all 23 distinct states, including stops."
        )
    declared_dna = settings.get("dna_alphabet", ["A", "C", "G", "T"])
    if len(declared_dna) != 4 or set(declared_dna) != {"A", "C", "G", "T"}:
        raise ValueError("Codon-entropy DNA alphabet must contain A, C, G, and T.")
    tree = candidate_view.tree
    nodes: dict[int, dict[str, Any]] = {
        int(idx): dict(tree[idx]) for idx in tree.node_indices()
    }
    peaks = [
        (idx, row)
        for idx, row in nodes.items()
        if tree.in_degree(idx) == 0 and row.get("event") == "maximum"
    ]
    expected_kem = str(settings.get("kem_summit_dna_sequence", "AAGGAAATG"))
    expected_tcl = str(settings.get("tcl_summit_dna_sequence", "ACCTGTCTT"))
    by_dna = {str(row["dna_sequence"]): (idx, row) for idx, row in peaks}
    try:
        kem_idx, kem_peak = by_dna[expected_kem]
        tcl_idx, tcl_peak = by_dna[expected_tcl]
    except KeyError as error:
        raise ValueError(
            f"Configured codon-entropy summit DNA is absent from simplified Candidate tree: {error.args[0]}"
        ) from error
    common_saddle = lowest_common_ancestor(tree, kem_idx, tcl_idx)
    if common_saddle is None or common_saddle in {kem_idx, tcl_idx}:
        raise ValueError("KEM and TCL summits do not have a distinct shared saddle.")
    saddle = nodes[common_saddle]
    reported_saddle = settings.get("reported_saddle_dna_sequence", "AGTTGCCAA")

    kem_nodes, kem_subroot = _ridge_nodes(tree, common_saddle, kem_idx)
    tcl_nodes, tcl_subroot = _ridge_nodes(tree, common_saddle, tcl_idx)
    if kem_nodes & tcl_nodes:
        raise ValueError("KEM and TCL critical-node ridge populations overlap.")
    assignments = candidate_view.assignments
    if not {"dna_sequence", "aa_sequence", "raw_fitness"} <= set(assignments.columns):
        raise ValueError("Simplified Candidate assignments lack sequence evidence.")
    df_core = _sequence_frame(
        assignments.select(
            "dna_sequence",
            "aa_sequence",
            pl.col("raw_fitness").alias("scalar"),
        )
    ).with_columns(functional_core.alias("is_functional_core"))
    dna_alphabet = set(settings.get("dna_alphabet", ["A", "C", "G", "T"]))
    aa_alphabet = set(settings.get("aa_alphabet", []))
    _validate_sequence_translation(
        df_core, dna_alphabet=dna_alphabet, aa_alphabet=aa_alphabet
    )

    def ridge_population(critical_nodes: set[int], subroot: int) -> pl.DataFrame:
        ridge_critical_ids = {str(nodes[idx]["dna_sequence"]) for idx in critical_nodes}
        subroot_id = str(nodes[subroot]["dna_sequence"])
        saddle_id = str(saddle["dna_sequence"])
        selected = assignments.filter(
            (
                pl.col("upper_sequence").is_in(sorted(ridge_critical_ids))
                & pl.col("lower_sequence").is_in(sorted(ridge_critical_ids))
            )
            | (
                (pl.col("upper_sequence") == subroot_id)
                & (pl.col("lower_sequence") == saddle_id)
            )
        )
        return _sequence_frame(
            selected.select(
                "dna_sequence",
                "aa_sequence",
                pl.col("raw_fitness").alias("scalar"),
            )
        ).with_columns(functional_core.alias("is_functional_core"))

    df_kem = ridge_population(kem_nodes, kem_subroot)
    df_tcl = ridge_population(tcl_nodes, tcl_subroot)
    saddle_fitness = float(saddle["scalar"])
    for ridge_id, ridge_frame in (("KEM", df_kem), ("TCL", df_tcl)):
        if ridge_frame.filter(pl.col("fitness") <= saddle_fitness).height:
            raise ValueError(
                f"{ridge_id} ridge contains a sequence at or below the common saddle."
            )
    core_ids = set(df_core.get_column("dna_sequence"))
    kem_ids = set(df_kem.get_column("dna_sequence"))
    tcl_ids = set(df_tcl.get_column("dna_sequence"))
    if not kem_ids <= core_ids or not tcl_ids <= core_ids:
        raise ValueError(
            "A ridge sequence is absent from the simplified Candidate population."
        )
    if kem_ids & tcl_ids:
        raise ValueError("KEM and TCL sequence populations overlap.")
    populations = {"core": df_core, "kem": df_kem, "tcl": df_tcl}
    expected_count = (
        comparison.get("diagnostics", {})
        .get("candidate_clamped", {})
        .get("selected_component_vertex_count")
    )
    if expected_count is not None and len(core_ids) != int(expected_count):
        raise ValueError(
            f"Codon-entropy population has {len(core_ids)} sequences, expected all {expected_count} Candidate graph vertices."
        )
    frequencies, site_entropy, sequence_assignments = _frames_for_populations(
        populations
    )
    claims = _claim_summary(populations, sequence_assignments, threshold, settings)

    core_path = ancestor_path(tree, common_saddle)
    trunk_critical = set(core_path)
    trunk_ids = {str(nodes[idx]["dna_sequence"]) for idx in trunk_critical}
    trunk_edges = {
        (str(nodes[upper]["dna_sequence"]), str(nodes[lower]["dna_sequence"]))
        for upper, lower in itertools.pairwise(core_path)
    }
    trunk_ids.update(
        row["dna_sequence"]
        for row in assignments.to_dicts()
        if (row["upper_sequence"], row["lower_sequence"]) in trunk_edges
    )
    sequence_assignments = sequence_assignments.with_columns(
        pl.when(pl.col("dna_sequence") == str(saddle["dna_sequence"]))
        .then(pl.lit("common_saddle"))
        .when(pl.col("dna_sequence").is_in(trunk_ids))
        .then(pl.lit("shared_trunk"))
        .when(pl.col("dna_sequence") == expected_kem)
        .then(pl.lit("kem_summit"))
        .when(pl.col("dna_sequence") == expected_tcl)
        .then(pl.lit("tcl_summit"))
        .when(pl.col("ridge_id").is_in(["kem", "tcl"]))
        .then(pl.lit("ridge"))
        .otherwise(pl.lit("unassigned"))
        .alias("membership_role")
    ).with_columns(
        pl.when(pl.col("membership_role").is_in(["common_saddle", "shared_trunk"]))
        .then(pl.lit("shared"))
        .otherwise(pl.col("ridge_id"))
        .alias("ridge_id")
    )
    expected_grid = {
        "min_raw_fitness": float(df_core.get_column("fitness").min()),
        "max_raw_fitness": float(df_core.get_column("fitness").max()),
        "regular_point_count": int(settings.get("sweep_points", 100)),
        "endpoint_policy": "union_with_each_population_minimum_and_maximum",
    }
    source_fingerprint = comparison.get("provenance", {}).get("fingerprint")
    metadata: dict[str, Any] = {
        "schema": POPULATION_SCHEMA,
        "analysis_scalar": "raw_fitness",
        "claim_summary_column_descriptions": CLAIM_SUMMARY_COLUMN_DESCRIPTIONS,
        "functional_threshold": float(threshold),
        "functional_rule": (
            "median_reflected: raw fitness at or above the Candidate threshold"
            if members is None
            else "mixture_quantile: the sequence is in the Functional class"
        ),
        "population_definitions": {
            "core": "all sequence identities represented in the simplified Candidate split tree assignments, including below-threshold fringe, at raw fitness",
            "kem": "all critical and branch-member sequences strictly above the common saddle on the configured KEM side; saddle excluded",
            "tcl": "all critical and branch-member sequences strictly above the common saddle on the configured TCL side; saddle excluded",
        },
        "sequence_alphabets": {
            "dna": list(settings.get("dna_alphabet", ["A", "C", "G", "T"])),
            "amino_acid": list(settings.get("aa_alphabet", [])),
            "stop_codons": list(settings.get("stop_codons", ["TAA", "TAG", "TGA"])),
        },
        "common_sweep_grid": expected_grid,
        "source_fingerprint": source_fingerprint,
        "summits": {
            "kem": {
                "dna_sequence": expected_kem,
                "aa_sequence": kem_peak["aa_sequence"],
                "raw_fitness": float(kem_peak["scalar"]),
            },
            "tcl": {
                "dna_sequence": expected_tcl,
                "aa_sequence": tcl_peak["aa_sequence"],
                "raw_fitness": float(tcl_peak["scalar"]),
            },
        },
        "kem_peak": {
            "dna_sequence": expected_kem,
            "aa_sequence": kem_peak["aa_sequence"],
            "fitness": float(kem_peak["scalar"]),
        },
        "tcl_peak": {
            "dna_sequence": expected_tcl,
            "aa_sequence": tcl_peak["aa_sequence"],
            "fitness": float(tcl_peak["scalar"]),
        },
        "common_saddle": {
            "dna_sequence": str(saddle["dna_sequence"]),
            "aa_sequence": str(saddle["aa_sequence"]),
            "raw_fitness": float(saddle["scalar"]),
            "reported_dna_sequence": str(reported_saddle),
            "reported_identity_matches": str(reported_saddle)
            == str(saddle["dna_sequence"]),
        },
        "counts": {
            "core": len(core_ids),
            "kem_ridge": len(kem_ids),
            "tcl_ridge": len(tcl_ids),
            "shared_saddle_and_trunk": len(trunk_ids),
        },
        "sequence_identity_sha256": {
            name: hashlib.sha256(
                "\n".join(sorted(frame.get_column("dna_sequence"))).encode()
            ).hexdigest()
            for name, frame in populations.items()
        },
    }
    for name, frame in populations.items():
        if name == "core":
            continue
        if "is_functional_core" in frame.columns:
            metadata["counts"][f"{name}_functional"] = int(
                frame.get_column("is_functional_core").sum()
            )
    evidence = {
        "site_frequency": frequencies,
        "site_entropy": site_entropy,
        "sequence_assignments": sequence_assignments,
        "claim_summary": claims,
    }
    saved_files: dict[str, str] = {}
    if save_files:
        target_dir = Path(output_dir) if output_dir else OUTPUT / "tables"
        target_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "core": ("papkou2023_candidate_population", df_core),
            "kem": ("papkou2023_kem_ridge_population", df_kem),
            "tcl": ("papkou2023_tcl_ridge_population", df_tcl),
            "site_frequency": ("papkou2023_codon_entropy_site_frequency", frequencies),
            "site_entropy": ("papkou2023_codon_entropy_site_entropy", site_entropy),
            "sequence_assignments": (
                "papkou2023_codon_entropy_sequence_assignments",
                sequence_assignments,
            ),
            "claim_summary": ("papkou2023_codon_entropy_claim_summary", claims),
        }
        for key, (stem, frame) in outputs.items():
            parquet = target_dir / f"{stem}.parquet"
            csv = target_dir / f"{stem}.csv"
            frame.write_parquet(parquet)
            frame.write_csv(csv)
            saved_files[f"{key}_parquet"] = str(parquet)
            saved_files[f"{key}_csv"] = str(csv)
        meta_path = target_dir / "papkou2023_codon_entropy_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        saved_files["metadata_json"] = str(meta_path)
    return {
        "candidate_leaf_view": candidate_view,
        "metadata": metadata,
        "saved_files": saved_files,
        "df_core": df_core,
        "df_kem": df_kem,
        "df_tcl": df_tcl,
        **evidence,
    }
