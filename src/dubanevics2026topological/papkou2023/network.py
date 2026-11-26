# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""Network loading, normalization, and Papkou2023Workflow class for Papkou et al. (2023)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Literal

import polars as pl
import rustworkx as rx

from topographer.biology import annotate_stop_codons, has_stop_codon
from dubanevics2026topological.network_analysis import (
    NetworkAnalysis,
    NetworkWorkflow,
    TreeAnalysis,
)
from dubanevics2026topological.papkou2023.analysis import (
    build_papkou2023_backbone_analysis,
)
from dubanevics2026topological.papkou2023.checkpoints import produce_or_load_papkou_analysis
from topographer.graph import Graph, ScalarNetwork

VERTEX_INDEX_DTYPE = pl.Int64
ProgressCallback = Callable[[int, int], None]
PAPKOU_DEFAULT_THRESHOLD = -0.253


def normalize_papkou2023_graph(
    graph: rx.PyGraph | rx.PyDiGraph,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> rx.PyGraph | rx.PyDiGraph:
    """Copy a GraphML graph into the canonical Papkou ``s``, ``se``, ``pval`` schema."""
    normalized = graph.__class__()
    total = int(graph.num_nodes() + graph.num_edges())
    _report_progress(progress_callback, 0, total)
    for position, node_index in enumerate(graph.node_indices(), start=1):
        payload = dict(graph.get_node_data(node_index))
        payload["s"] = _finite_scalar(payload.get("s", payload.get("m")), "s")
        _add_article_sequences(payload)
        if payload.get("aa_sequence") is not None:
            payload["has_stop_codon"] = has_stop_codon(payload["aa_sequence"])
        if "m" in payload:
            del payload["m"]
        if "se" not in payload and "m_se" in payload:
            payload["se"] = payload["m_se"]
        if "pval" not in payload and "m_p.value" in payload:
            payload["pval"] = payload["m_p.value"]
        payload.pop("m_se", None)
        payload.pop("m_p.value", None)
        normalized.add_node(payload)
        _report_progress(progress_callback, position, total)
    node_count = int(graph.num_nodes())
    for position, edge_index in enumerate(graph.edge_indices(), start=node_count + 1):
        source, target = graph.get_edge_endpoints_by_index(edge_index)
        normalized.add_edge(
            int(source), int(target), dict(graph.get_edge_data_by_index(edge_index))
        )
        _report_progress(progress_callback, position, total)
    return normalized


def _add_article_sequences(payload: dict[str, Any]) -> None:
    if payload.get("dna_sequence") is None:
        codons: list[str] = []
        pos = 1
        while f"codon{pos}" in payload:
            codon = payload[f"codon{pos}"]
            if isinstance(codon, str) and codon:
                codons.append(codon)
                pos += 1
            else:
                break
        if codons:
            payload["dna_sequence"] = "".join(codons)
    if payload.get("aa_sequence") is None:
        amino_acids: list[str] = []
        pos = 1
        while f"aa{pos}" in payload:
            amino_acid = payload[f"aa{pos}"]
            if isinstance(amino_acid, str) and amino_acid:
                amino_acids.append(amino_acid)
                pos += 1
            else:
                break
        if amino_acids:
            payload["aa_sequence"] = "".join(amino_acids)


def _rx_to_scalar_network(
    graph: rx.PyGraph | rx.PyDiGraph, scalar_attr: str = "s"
) -> ScalarNetwork:
    if isinstance(graph, rx.PyDiGraph):
        graph = graph.to_undirected()
    node_indices = graph.node_indices()
    all_keys: set[str] = set()
    for idx in node_indices:
        data = dict(graph.get_node_data(idx))
        if "dna_sequence" not in data:
            data["dna_sequence"] = f"vid_{idx}"
            graph[idx] = data
        all_keys.update(data.keys())
    all_keys.discard("dna_sequence")
    g = Graph.from_rustworkx(
        graph, vertex_id_attr="dna_sequence", vertex_attribute_cols=tuple(sorted(all_keys))
    )
    return ScalarNetwork(g, scalar_attr=scalar_attr)


def load_papkou2023_article_network(
    path: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> ScalarNetwork:
    """Load the Papkou (2023) GraphML file as a validated ScalarNetwork."""
    graphs = rx.read_graphml(str(path))
    if not graphs:
        raise ValueError(f"No graph was found in {path!s}.")
    norm = normalize_papkou2023_graph(graphs[0], progress_callback=progress_callback)
    return _rx_to_scalar_network(norm, scalar_attr="s")


def _validate_and_derive_sequence_lengths(
    table: pl.DataFrame,
    *,
    dna_cols: tuple[str, ...],
    aa_cols: tuple[str, ...],
) -> tuple[int, int]:
    """Validate sequence lengths across columns and return (dna_length, aa_length).

    Validates:
    - All DNA sequences have the same length.
    - DNA sequence length is a positive multiple of 3.
    - All amino-acid sequences have the same length.
    - Amino-acid sequence length equals DNA sequence length // 3.
    """
    dna_lens: set[int] = set()
    for col in dna_cols:
        if col in table.columns and not table.is_empty():
            dna_lens.update(table.select(pl.col(col).str.len_chars().unique())[col].to_list())
    if not dna_lens:
        return 0, 0
    if len(dna_lens) > 1:
        raise ValueError(
            f"All DNA sequences must have the same length; found distinct lengths: {sorted(dna_lens)}."
        )
    dna_len = next(iter(dna_lens))
    if dna_len == 0 or dna_len % 3 != 0:
        raise ValueError(
            f"DNA sequence length must be a positive multiple of 3; got length {dna_len}."
        )

    expected_aa_len = dna_len // 3
    aa_lens: set[int] = set()
    for col in aa_cols:
        if col in table.columns and not table.is_empty():
            aa_lens.update(table.select(pl.col(col).str.len_chars().unique())[col].to_list())
    if aa_lens:
        if len(aa_lens) > 1 or next(iter(aa_lens)) != expected_aa_len:
            raise ValueError(
                f"Amino acid sequence length must equal DNA sequence length // 3 ({expected_aa_len}); "
                f"found distinct lengths: {sorted(aa_lens)}."
            )
        aa_len = next(iter(aa_lens))
    else:
        aa_len = expected_aa_len

    return dna_len, aa_len


def build_papkou2023_fitness_vertex_table(fitness_wt: pl.DataFrame) -> pl.DataFrame:
    """Build observed Papkou vertices plus a WT reference vertex."""
    if fitness_wt.is_empty():
        return pl.DataFrame(
            schema={
                "vertex_index": VERTEX_INDEX_DTYPE,
                "dna_sequence": pl.String,
                "aa_sequence": pl.String,
                "has_stop_codon": pl.Boolean,
                "s": pl.Float64,
                "se": pl.Float64,
                "pval": pl.Float64,
                "is_reference": pl.Boolean,
            }
        )
    _validate_and_derive_sequence_lengths(
        fitness_wt,
        dna_cols=("dna_from", "dna_to"),
        aa_cols=("aa_from", "aa_to"),
    )
    reference = fitness_wt.select("dna_from", "aa_from").head(1).to_dicts()[0]
    observed = fitness_wt.select(
        pl.col("dna_to").alias("dna_sequence"),
        pl.col("aa_to").alias("aa_sequence"),
        "s",
        "se",
        "pval",
    ).with_columns(is_reference=pl.lit(False))
    return (
        pl.concat(
            [
                pl.DataFrame(
                    [
                        {
                            "dna_sequence": reference["dna_from"],
                            "aa_sequence": reference["aa_from"],
                            "s": 0.0,
                            "se": float("nan"),
                            "pval": float("nan"),
                            "is_reference": True,
                        }
                    ]
                ),
                observed,
            ],
            how="vertical_relaxed",
        )
        .with_columns(sort_key=pl.when(pl.col("is_reference")).then(0).otherwise(1))
        .sort(["sort_key", "dna_sequence"])
        .drop("sort_key")
        .with_row_index("vertex_index")
        .with_columns(pl.col("vertex_index").cast(VERTEX_INDEX_DTYPE))
        .pipe(annotate_stop_codons)
    )


def build_papkou2023_fitness_edge_table(
    fitness_diff: pl.DataFrame, vertex_table: pl.DataFrame
) -> pl.DataFrame:
    """Build oriented Papkou mutation edges with endpoint fitness and Hamming metadata."""
    dna_len, aa_len = 0, 0
    if not vertex_table.is_empty():
        dna_len, aa_len = _validate_and_derive_sequence_lengths(
            vertex_table, dna_cols=("dna_sequence",), aa_cols=("aa_sequence",)
        )
    if not fitness_diff.is_empty():
        diff_dna_len, diff_aa_len = _validate_and_derive_sequence_lengths(
            fitness_diff,
            dna_cols=("dna_from", "dna_to"),
            aa_cols=("aa_from", "aa_to"),
        )
        if dna_len > 0 and (diff_dna_len, diff_aa_len) != (dna_len, aa_len):
            raise ValueError(
                f"Vertex table sequence lengths ({dna_len} nt, {aa_len} aa) do not match "
                f"edge table sequence lengths ({diff_dna_len} nt, {diff_aa_len} aa)."
            )
        dna_len, aa_len = diff_dna_len, diff_aa_len

    oriented = fitness_diff.select(
        "dna_from", "dna_to", "aa_from", "aa_to", "s", "se", "pval"
    ).with_columns(
        pl.when(pl.col("s") < 0)
        .then(pl.col("dna_to"))
        .otherwise(pl.col("dna_from"))
        .alias("source_dna"),
        pl.when(pl.col("s") < 0)
        .then(pl.col("dna_from"))
        .otherwise(pl.col("dna_to"))
        .alias("target_dna"),
        pl.when(pl.col("s") < 0)
        .then(pl.col("aa_to"))
        .otherwise(pl.col("aa_from"))
        .alias("source_aa"),
        pl.when(pl.col("s") < 0)
        .then(pl.col("aa_from"))
        .otherwise(pl.col("aa_to"))
        .alias("target_aa"),
        pl.col("s").abs().alias("s"),
        pl.col("s").abs().alias("s_raw"),
    )
    source = vertex_table.select(
        pl.col("dna_sequence").alias("source_dna"),
        pl.col("vertex_index").alias("source_vertex_index"),
        pl.col("s").alias("source_fitness"),
        pl.col("se").alias("source_se"),
        pl.col("pval").alias("source_pval"),
        pl.col("is_reference").cast(pl.Int8).alias("source_is_reference"),
    )
    target = vertex_table.select(
        pl.col("dna_sequence").alias("target_dna"),
        pl.col("vertex_index").alias("target_vertex_index"),
        pl.col("s").alias("target_fitness"),
        pl.col("se").alias("target_se"),
        pl.col("pval").alias("target_pval"),
        pl.col("is_reference").cast(pl.Int8).alias("target_is_reference"),
    )
    return (
        oriented.join(source, on="source_dna", how="left")
        .join(target, on="target_dna", how="left")
        .with_columns(
            pl.col("source_fitness").is_not_null().alias("source_present"),
            pl.col("target_fitness").is_not_null().alias("target_present"),
            pl.col("s").alias("fitness_delta"),
            pl.sum_horizontal(
                [
                    (
                        pl.col("source_dna").str.slice(index, 1)
                        != pl.col("target_dna").str.slice(index, 1)
                    ).cast(pl.Int8)
                    for index in range(dna_len)
                ]
            ).alias("dna_hamming")
            if dna_len > 0
            else pl.lit(0, dtype=pl.Int8).alias("dna_hamming"),
            pl.sum_horizontal(
                [
                    (
                        pl.col("source_aa").str.slice(index, 1)
                        != pl.col("target_aa").str.slice(index, 1)
                    ).cast(pl.Int8)
                    for index in range(aa_len)
                ]
            ).alias("aa_hamming")
            if aa_len > 0
            else pl.lit(0, dtype=pl.Int8).alias("aa_hamming"),
        )
    )


def build_papkou2023_fitness_graph(
    vertex_table: pl.DataFrame, edge_table: pl.DataFrame
) -> rx.PyDiGraph:
    """Build the directed Papkou graph from deterministic vertex and edge tables."""
    graph = rx.PyDiGraph()
    node_by_vertex = {}
    for row in vertex_table.sort("vertex_index").iter_rows(named=True):
        node_by_vertex[int(row["vertex_index"])] = graph.add_node(
            {
                "vertex_index": int(row["vertex_index"]),
                "dna_sequence": row["dna_sequence"],
                "aa_sequence": row["aa_sequence"],
                "has_stop_codon": bool(row["has_stop_codon"]),
                "s": float(row["s"]),
                "se": float(row["se"]),
                "pval": float(row["pval"]),
                "is_reference": int(bool(row["is_reference"])),
            }
        )
    for row in edge_table.iter_rows(named=True):
        graph.add_edge(
            node_by_vertex[int(row["source_vertex_index"])],
            node_by_vertex[int(row["target_vertex_index"])],
            {
                "s": float(row["s"]),
                "se": float(row["se"]),
                "pval": float(row["pval"]),
                "dna_hamming": int(row["dna_hamming"]),
                "aa_hamming": int(row["aa_hamming"]),
            },
        )
    return graph


def build_papkou2023_full_network(
    fitness_wt: pl.DataFrame,
    fitness_diff: pl.DataFrame,
) -> ScalarNetwork:
    """Build the complete directed Papkou landscape from its two raw tables."""
    vertices = build_papkou2023_fitness_vertex_table(fitness_wt)
    edges = build_papkou2023_fitness_edge_table(fitness_diff, vertices)
    graph = build_papkou2023_fitness_graph(
        vertices, edges.drop_nulls(["source_vertex_index", "target_vertex_index"])
    )
    return _rx_to_scalar_network(graph, scalar_attr="s")


def build_papkou2023_threshold_network(
    fitness_wt: pl.DataFrame,
    fitness_diff: pl.DataFrame,
    *,
    threshold: float = PAPKOU_DEFAULT_THRESHOLD,
) -> ScalarNetwork:
    """Build the functional-induced, largest-component threshold landscape."""
    threshold = _finite_scalar(threshold, "threshold")
    vertices = build_papkou2023_fitness_vertex_table(fitness_wt).filter(pl.col("s") > threshold)
    allowed = vertices.get_column("dna_sequence").implode()
    edges = build_papkou2023_fitness_edge_table(fitness_diff, vertices).filter(
        pl.col("source_dna").is_in(allowed) & pl.col("target_dna").is_in(allowed)
    )
    graph = build_papkou2023_fitness_graph(
        vertices, edges.drop_nulls(["source_vertex_index", "target_vertex_index"])
    )
    selected = select_papkou2023_largest_weak_component(graph)
    return _rx_to_scalar_network(selected, scalar_attr="s")


def select_papkou2023_largest_weak_component(graph: rx.PyGraph | rx.PyDiGraph):
    """Return the largest weak component with deterministic fitness/DNA tie breaking."""
    components = list(rx.weakly_connected_components(graph))
    if not components:
        raise ValueError("Cannot select a component from an empty graph.")

    def key(component: set[int]) -> tuple[int, float, str]:
        nodes = sorted(map(int, component))
        payloads = [dict(graph.get_node_data(index)) for index in nodes]
        return (
            -len(nodes),
            -max(float(payload["s"]) for payload in payloads),
            min(str(payload.get("dna_sequence", "")) for payload in payloads),
        )

    return graph.subgraph(sorted(min(components, key=key)))


def _finite_scalar(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real number.")
    return numeric


def _report_progress(
    callback: Callable[[int, int], None] | None, completed: int, total: int
) -> None:
    if callback is None:
        return
    step = max(1, total // 1000)
    if completed == 0 or completed == total or completed % step == 0:
        callback(completed, total)


@dataclass(slots=True)
class Papkou2023Workflow(NetworkWorkflow):
    """Workflow runner for Papkou et al. (2023) landscapes."""

    name: str = "papkou2023"
    landscape: str = "article"
    threshold: float = PAPKOU_DEFAULT_THRESHOLD
    article_path: str | Path | None = None
    fitness_wt: pl.DataFrame | None = None
    fitness_diff: pl.DataFrame | None = None

    @property
    def config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"landscape": self.landscape}
        if self.landscape in {"threshold", "new-threshold"}:
            cfg["threshold"] = self.threshold
        return cfg

    def build_network(self) -> ScalarNetwork:
        if self.landscape == "article":
            path = (
                self.article_path
                if self.article_path is not None
                else Path("data/external/graph_largest_component.graphml")
            )
            return load_papkou2023_article_network(path)
        wt = (
            self.fitness_wt
            if self.fitness_wt is not None
            else pl.read_parquet("data/raw/papkou2023_fitness_wt.parquet")
        )
        diff = (
            self.fitness_diff
            if self.fitness_diff is not None
            else pl.read_parquet("data/raw/papkou2023_fitness_diff.parquet")
        )
        if self.landscape == "full":
            return build_papkou2023_full_network(wt, diff)
        if self.landscape in {"threshold", "new-threshold"}:
            return build_papkou2023_threshold_network(wt, diff, threshold=self.threshold)
        raise ValueError(f"Unknown Papkou2023 landscape mode: {self.landscape!r}")

    def produce_or_load_analysis(
        self,
        output_dir: str | Path,
        *,
        tree_kind: Literal["split", "join", "both"] = "both",
        force: bool = False,
    ) -> NetworkAnalysis:
        """Load a matching Papkou checkpoint or atomically publish a fresh analysis."""
        return produce_or_load_papkou_analysis(self, output_dir, tree_kind=tree_kind, force=force)

    def analyze(
        self,
        *,
        tree_kind: Literal["split", "join", "both"] = "both",
    ) -> NetworkAnalysis:
        sn = self.build_network()
        kinds: tuple[Literal["join", "split"], ...]
        if tree_kind == "both":
            kinds = ("join", "split")
        elif tree_kind in ("join", "split"):
            kinds = (tree_kind,)
        else:
            raise ValueError(f"Invalid tree_kind: {tree_kind!r}")

        trees: dict[Literal["join", "split"], TreeAnalysis] = {}
        for k in kinds:
            papkou_tree_analysis = build_papkou2023_backbone_analysis(sn, kind=k)
            trees[k] = TreeAnalysis(
                kind=k,
                scalar_attr="s",
                tree=papkou_tree_analysis.tree,
                plot_data=papkou_tree_analysis.plot_data,
                event_table=papkou_tree_analysis.event_table,
                branch_table=papkou_tree_analysis.branch_table,
                branch_span_table=papkou_tree_analysis.branch_span_table,
                persistence_table=papkou_tree_analysis.persistence_table,
                branch_member_table=papkou_tree_analysis.branch_member_table,
                critical_node_table=papkou_tree_analysis.critical_node_table,
                stop_marker_table=papkou_tree_analysis.stop_marker_table,
                all_stop_marker_table=papkou_tree_analysis.all_stop_marker_table,
                wild_type_marker_table=papkou_tree_analysis.wild_type_marker_table,
            )

        scalars = [payload["s"] for payload in sn.graph.nodes()]
        scalar_domain = (float(min(scalars)), float(max(scalars)))

        input_summary = pl.DataFrame(
            [
                {"metric": "vertex_count", "value": sn.vertex_count},
                {"metric": "edge_count", "value": sn.edge_count},
                {"metric": "scalar_min", "value": scalar_domain[0]},
                {"metric": "scalar_max", "value": scalar_domain[1]},
            ]
        )

        return NetworkAnalysis(
            name=self.name,
            config=self.config,
            scalar_attr="s",
            scalar_domain=scalar_domain,
            scalar_label="Fitness",
            input_summary=input_summary,
            trees=trees,
        )
