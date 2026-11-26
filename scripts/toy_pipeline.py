"""Run the checked-in 6-nt landscape through current analysis seams."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from dubanevics2026topological.analysis import OUTPUT, ROOT, describe
from dubanevics2026topological.analysis.sequence_growth_rates import (
    SequenceGrowthRatesResult,
)
from dubanevics2026topological.landscape import selection as cs
from dubanevics2026topological.landscape.selected_landscape import (
    analyze_prepared_landscape,
    prepare_clamped_landscape,
)
from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
    build_band_simplified_view,
    build_bootstrap_smoothed_tree,
)
from dubanevics2026topological.threshold import classify_functional_sequences
from dubanevics2026topological.threshold import fdr_selection as fdr

DATA = ROOT / "tests/data/toy"
THRESHOLD = -1.3
WT = "TAAACA"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fdr_snapshot(growth: pl.DataFrame) -> dict:
    directory = OUTPUT / "toy_fdr_snapshot"
    directory.mkdir(parents=True, exist_ok=True)
    sequence_ids = growth["sequence_id"].to_list()
    result = classify_functional_sequences(
        growth["s"].to_numpy(),
        growth["se_pooled"].to_numpy(),
        THRESHOLD,
        sequence_ids=sequence_ids,
        library_ids=sequence_ids,
        nominal_q=0.05,
    )
    table = growth.with_columns(
        pl.Series("fdr_class", result.classes),
        pl.Series("gated_p_value", result.gated_p_values),
        pl.Series("adjusted_p_value", result.q_values),
    )
    path = directory / fdr.table_name(THRESHOLD, "median_reflected", "by", 0.05)
    table.select("dna_sequence", pl.col("fdr_class")).write_parquet(path)
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "cutoffs": [
                    {
                        "name": "median_reflected",
                        "source_precision_threshold": THRESHOLD,
                        "applied_threshold": THRESHOLD,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "rule": "fdr_fixed_cutoff",
        "decimals": 6,
        "cutoff": "median_reflected",
        "method": "by",
        "q": 0.05,
        "input_dir": directory.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wt = pl.read_parquet(DATA / "fitness_wt.parquet")
    diff = pl.read_parquet(DATA / "fitness_diff.parquet")
    growth_rates = pl.read_parquet(DATA / "growth_rates.parquet")
    full = Papkou2023Workflow(
        landscape="full", fitness_wt=wt, fitness_diff=diff
    ).build_network()
    declaration = _fdr_snapshot(growth_rates)
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "tests/data/toy/fitness_wt.parquet"},
            "clamped": {"candidate_functional_threshold": declaration},
        }
    }
    prepared, selection = cs.prepare_candidate_landscape(full, settings)
    analysis = analyze_prepared_landscape(
        prepared, name="toy_candidate", wild_type_sequence=WT
    )
    annotations = describe(
        products={"annotations"}, landscape=analysis, landscape_name="toy_candidate"
    ).annotations
    growth = SequenceGrowthRatesResult(
        growth_rates=growth_rates,
        diagnostics={"identity_tolerance": 1e-12, "wild_type_r": 1.0},
        provenance={"source": "tests/data/toy/growth_rates.parquet"},
    )
    bootstrap = build_bootstrap_smoothed_tree(
        prepared.network,
        analysis.trees["split"],
        growth,
        {
            "n_replicates": 50,
            "seed": 127,
            "band_quantile": 0.95,
            "n_jobs": 1,
            "chunk_size": 25,
        },
    )
    band_view = build_band_simplified_view(analysis.trees["split"], bootstrap.hills)

    full_graph = full.graph
    raw = [float(full_graph[node]["s"]) for node in full_graph.node_indices()]
    reflected_threshold = 2 * float(pl.Series(raw).median()) - min(raw)
    reflected = prepare_clamped_landscape(full, reflected_threshold)
    reflected_graph = reflected.network.graph  # property deep-copies; read once
    reflected_analysis = analyze_prepared_landscape(
        reflected, name="toy_median_reflected", wild_type_sequence=WT
    )
    reflected_annotations = describe(
        products={"annotations"},
        landscape=reflected_analysis,
        landscape_name="toy_median_reflected",
    ).annotations

    out = OUTPUT / "analysis" / "toy"
    out.mkdir(parents=True, exist_ok=True)
    growth_rates.write_parquet(out / "growth_rates.parquet")
    annotations.pairs.write_parquet(out / "candidate_pairs.parquet")
    annotations.branches.write_parquet(out / "candidate_branches.parquet")
    bootstrap.hills.write_parquet(out / "bootstrap_hills.parquet")
    bootstrap.draw_stats.write_parquet(out / "bootstrap_draw_stats.parquet")
    band_view.assignments.write_parquet(out / "band_assignments.parquet")
    band_view.pairs.write_parquet(out / "band_pairs.parquet")
    reflected_annotations.pairs.write_parquet(out / "median_reflected_pairs.parquet")
    summary = {
        "threshold": selection.threshold.to_json(),
        "classes": declaration,
        "selected_vertex_count": prepared.network.vertex_count,
        "selected_edge_count": prepared.network.edge_count,
        "nominal_pair_count": annotations.pairs.height,
        "bootstrap_draws_per_series": 50,
        "bootstrap_band": bootstrap.summary["band_simplification"],
        "bootstrap_c": bootstrap.summary["band_simplification"]["c"],
        "bootstrap_summary": bootstrap.summary,
        "band_peak_count": band_view.pairs.height,
        "reflected_threshold": reflected_threshold,
        "reflected_vertex_ids": sorted(
            reflected_graph[node]["dna_sequence"]
            for node in reflected_graph.node_indices()
        ),
        "codon_entropy": "not run (toy sequences are 6 nt)",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
