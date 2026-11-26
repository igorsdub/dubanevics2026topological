"""Legacy mixture presentation pending active-config product stages.

This explicit compatibility path preserves the default figures until the active
mixture landscape, bootstrap, tree and description products are published.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import polars as pl


class LegacyPresentationPayload(dict):
    """Keep legacy routing metadata outside the figure payload's keys."""

    legacy_mixture = True


def tree_population(tree_analysis: Any, population_id: str) -> pl.DataFrame:
    population = pl.concat(
        [
            tree_analysis.critical_node_table.select("dna_sequence", "scalar"),
            tree_analysis.branch_member_table.select("dna_sequence", "scalar"),
        ]
    )
    conflicting = (
        population.group_by("dna_sequence")
        .agg(pl.col("scalar").n_unique().alias("scalar_count"))
        .filter(pl.col("scalar_count") != 1)
    )
    if conflicting.height:
        raise ValueError(
            f"{population_id.title()} tree tables disagree on a vertex's raw fitness."
        )
    return population.unique("dna_sequence", keep="first")


def bootstrap_tree(inputs: dict[str, Any], settings: dict[str, Any]) -> Any:
    from ..analysis.candidate_tree_contrast import build_candidate_tree
    from ..simplification.bootstrap_smoothed_tree import check_hills_match_tree

    if "bootstrap_tree" not in inputs:
        if "candidate_contrast_tree" not in inputs:
            inputs["candidate_contrast_tree"], _ = build_candidate_tree(settings)
        tree = inputs["candidate_contrast_tree"]
        check_hills_match_tree(inputs["bootstrap_smoothed"].hills, tree)
        inputs["bootstrap_tree"] = tree
    return inputs["bootstrap_tree"]


def bootstrap_band_view(inputs: dict[str, Any], settings: dict[str, Any]) -> Any:
    from ..simplification.bootstrap_smoothed_tree import build_band_simplified_view

    if "bootstrap_band_view" not in inputs:
        inputs["bootstrap_band_view"] = build_band_simplified_view(
            bootstrap_tree(inputs, settings), inputs["bootstrap_smoothed"].hills
        )
    return inputs["bootstrap_band_view"]


def annotations(tree: Any) -> Any:
    from ..analysis import describe

    return describe(
        products={"annotations"},
        landscape=SimpleNamespace(trees={"split": tree}),
        landscape_name="candidate_clamped",
    ).annotations


def selected_view_as_tree(view: Any) -> SimpleNamespace:
    plot_data = view.plot_data
    labels = {
        int(row["tree_node_index"]): str(row["dna_sequence"])
        for row in plot_data.node_table.iter_rows(named=True)
    }
    assignments = view.assignments
    on_branch = Counter(
        frozenset((row["upper_sequence"], row["lower_sequence"]))
        for row in assignments.iter_rows(named=True)
        if row["upper_sequence"] != row["lower_sequence"]
    )
    rows = []
    for row in plot_data.edge_table.iter_rows(named=True):
        source, target = (
            int(row["source_tree_node_index"]),
            int(row["target_tree_node_index"]),
        )
        key = frozenset((labels[source], labels[target]))
        if key not in on_branch:
            raise ValueError("A drawn edge of the smoothed tree has no member vertex.")
        rows.append(
            {
                "source_tree_node_index": source,
                "target_tree_node_index": target,
                "branch_vertex_count": on_branch[key],
            }
        )
    fitness = assignments.select("dna_sequence", pl.col("raw_fitness").alias("scalar"))
    return SimpleNamespace(
        plot_data=plot_data,
        branch_table=pl.DataFrame(
            rows,
            schema={
                "source_tree_node_index": pl.Int64,
                "target_tree_node_index": pl.Int64,
                "branch_vertex_count": pl.Int64,
            },
        ),
        critical_node_table=fitness,
        branch_member_table=fitness.clear(),
        wild_type_marker_table=view.wild_type_marker_table,
    )
