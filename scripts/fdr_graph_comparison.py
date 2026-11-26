"""Three-variant graph comparison: Papkou point estimates, Papkou + FDR 5%, Topographer + FDR 5%.

Builds each prepared graph through ``prepare_clamped_landscape`` /
``prepare_candidate_landscape`` (the established construction), then records the
population at every construction stage and the sha256 of the selected component's
sorted DNA identities and canonical edge list. The published baseline is the scalar
point-estimate rule at the exact Papkou cutoff; it is the q = 1 gated classification.

Writes ``output/analysis/fdr_graph_comparison/fdr_graph_comparison.{json,csv}`` (under the
active output root). Run: ``PYTHONPATH=src python scripts/fdr_graph_comparison.py``.
"""

from __future__ import annotations

import hashlib
import json
import os

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from dubanevics2026topological.analysis import OUTPUT, ROOT, load_settings
from dubanevics2026topological.landscape.selected_landscape import (
    prepare_clamped_landscape,
)
from dubanevics2026topological.landscape.selection import (
    prepare_candidate_landscape,
)

PAPKOU_T = -0.5077739708744209
VARIANTS = (
    ("published_point_estimates", "Papkou baseline: point estimates (no uncertainty screening)", None),
    ("fdr_by_q0.05_papkou_original", "Papkou threshold: BY, nominal 5% FDR", "papkou_original"),
    ("fdr_by_q0.05_median_reflected", "Topographer threshold: BY, nominal 5% FDR", "median_reflected"),
)


def _sha(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _fingerprints(network) -> tuple[list[str], str, str]:
    graph = network.graph
    dna = sorted(graph[i]["dna_sequence"] for i in graph.node_indices())
    pairs = set()
    for edge in graph.edge_indices():
        left, right = graph.get_edge_endpoints_by_index(edge)
        pairs.add(tuple(sorted((graph[left]["dna_sequence"], graph[right]["dna_sequence"]))))
    edges = ["\t".join(pair) for pair in sorted(pairs)]
    return dna, _sha(dna), _sha(edges)


def _settings_for(variant: str) -> dict:
    os.environ["FITNESSTREE_ANALYSIS_VARIANT"] = variant
    try:
        return load_settings()
    finally:
        del os.environ["FITNESSTREE_ANALYSIS_VARIANT"]


def main() -> None:
    base = load_settings()
    scientific = base["analysis"]["dataset"]
    workflow = Papkou2023Workflow(
        landscape="full",
        fitness_wt=pl.read_parquet(ROOT / scientific["fitness_wt"]),
        fitness_diff=pl.read_parquet(ROOT / scientific["fitness_diff"]),
    )
    full = workflow.build_network()
    graph = full.graph
    baseline_functional = sum(
        graph[i][full.scalar_attr] >= PAPKOU_T for i in graph.node_indices()
    )
    rows = []
    for name, label, cutoff in VARIANTS:
        if cutoff is None:
            prepared = prepare_clamped_landscape(full, PAPKOU_T)
            record = {"cutoff": PAPKOU_T, "rule": "scalar_point_estimate_raw_fitness_ge_cutoff",
                      "method": None, "q": None, "table_sha256": None, "uncertain_total": None}
        else:
            settings = _settings_for(name)
            prepared, selection = prepare_candidate_landscape(full, settings)
            record = {"cutoff": selection.threshold.threshold,
                      "rule": "fdr_fixed_cutoff_functional_class",
                      "method": selection.source["method"], "q": selection.source["q"],
                      "table_sha256": selection.source["table_sha256"],
                      "uncertain_total": selection.uncertain_count}
        d = prepared.diagnostics
        _, dna_sha, edge_sha = _fingerprints(prepared.network)
        classes = prepared.excluded_vertices  # outside the selected component
        rows.append({
            "variant": name, "label": label, **record,
            "full_vertices": d["input_vertex_count"], "full_edges": d["input_edge_count"],
            "functional_total": d["functional_vertex_count"]
            if d["functional_vertex_count"] is not None
            else baseline_functional,
            "low_low_edges_excluded": d["low_low_edge_count_excluded"],
            "incident_vertices": d["incident_vertex_count"],
            "components": d["component_count"],
            "selected_vertices": d["selected_component_vertex_count"],
            "selected_edges": d["selected_component_edge_count"],
            "selected_functional": d["selected_component_functional_vertex_count"],
            "peaks": d["peak_count"],
            "selected_dna_sha256": dna_sha, "selected_edges_sha256": edge_sha,
            "excluded_vertices_table_rows": classes.height,
        })
    out = OUTPUT / "analysis" / "fdr_graph_comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fdr_graph_comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pl.DataFrame(rows).write_csv(out / "fdr_graph_comparison.csv")
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
