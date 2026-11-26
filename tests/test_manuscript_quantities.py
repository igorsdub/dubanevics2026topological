"""The manuscript quantity registry: rows, rounding, gaps, report.

Every fixture is synthetic and lives in ``tmp_path``; no stage runs on real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import rustworkx as rx
from _saddle_band import _pairs

from dubanevics2026topological import build
from dubanevics2026topological.analysis import manuscript_quantities as mq
from dubanevics2026topological.analysis.codon_entropy_populations import (
    lowest_common_ancestor,
)

CONFIG = {
    "clamped": {"published_functional_threshold": -0.507774},
    "codon_entropy": {
        "kem_summit_dna_sequence": "AAGGAAATG",
        "tcl_summit_dna_sequence": "ACCTGTCTT",
        "reported_saddle_dna_sequence": "CTCTGCTTG",
        "two_functional_ridges_share_minimum": 0.9,
    },
}
SUMMARY_LABELS = (
    "Number of nodes",
    "Number of edges",
    "Minimum fitness",
    "Maximum fitness",
    "Mean fitness",
    "Median fitness",
    "Functional threshold",
    "Number of functional sequences",
    "Number of non-functional sequences",
    "Percentage of functional sequences",
)


def _tree(saddle_dna: str = "CTCTGCTTG", extra_peaks: int = 28) -> rx.PyDiGraph:
    """Two summits that merge at a saddle, then a root: edges run upper to lower.

    ``extra_peaks`` more peaks join the saddle, below the summits. The first has a
    stop codon in its amino-acid sequence and the second the saddle's (synonymous).
    """
    tree = rx.PyDiGraph()
    kem = tree.add_node(
        {"dna_sequence": "AAGGAAATG", "aa_sequence": "KEM", "scalar": 0.4567}
    )
    tcl = tree.add_node(
        {"dna_sequence": "ACCTGTCTT", "aa_sequence": "TCL", "scalar": 0.3333}
    )
    saddle = tree.add_node(
        {"dna_sequence": saddle_dna, "aa_sequence": "LCL", "scalar": -0.2531585}
    )
    root = tree.add_node(
        {"dna_sequence": "GGGGGGGGG", "aa_sequence": "GGG", "scalar": -1.0}
    )
    tree.add_edge(kem, saddle, None)
    tree.add_edge(tcl, saddle, None)
    tree.add_edge(saddle, root, None)
    for i in range(extra_peaks):
        amino = {0: "P*", 1: "LCL"}.get(i, "PQ")
        peak = tree.add_node(
            {
                "dna_sequence": f"PEAK{i:05d}",
                "aa_sequence": amino,
                "scalar": -0.2 + i / 1e3,
            }
        )
        tree.add_edge(peak, saddle, None)
    return tree


def _split_analysis(tree: rx.PyDiGraph, members: dict[str, int] | None = None) -> Any:
    """A synthetic split-tree analysis over ``tree`` with the tables the registry reads.

    ``members`` maps a peak's DNA sequence to the extra sequences on its terminal
    branch (a hill of ``1 + members`` sequences). The highest peak is the global pair,
    closed at the root; every other peak is paired with its adjacent saddle.
    """
    members = members or {}
    nodes = {i: tree[i] for i in tree.node_indices()}
    peaks = [i for i in nodes if tree.in_degree(i) == 0]
    root = next(i for i in nodes if tree.out_degree(i) == 0)
    top = max(peaks, key=lambda i: nodes[i]["scalar"])
    branch_rows, extra = [], []
    for upper, lower in tree.edge_list():
        dna = nodes[upper]["dna_sequence"]
        count = 1 + members.get(dna, 0)
        branch_rows.append(
            {
                "source_tree_node_index": upper,
                "target_tree_node_index": lower,
                "branch_vertex_count": count,
            }
        )
        extra += [f"{dna}-MEMBER{k}" for k in range(count - 1)]
    pair_rows = []
    for peak in peaks:
        death = root if peak == top else next(iter(tree.successor_indices(peak)))
        pair_rows.append(
            {
                "pair_id": nodes[peak]["dna_sequence"],
                "pair_kind": "extended" if peak == top else "ordinary",
                "birth_tree_node_index": peak,
                "death_tree_node_index": death,
                "birth_scalar": float(nodes[peak]["scalar"]),
                "death_scalar": float(nodes[death]["scalar"]),
            }
        )
    events = [
        "maximum"
        if tree.in_degree(i) == 0
        else "minimum"
        if tree.out_degree(i) == 0
        else "saddle"
        for i in sorted(nodes)
    ]
    return SimpleNamespace(
        tree=tree,
        event_table=pl.DataFrame({"event": events}),
        critical_node_table=pl.DataFrame(
            {"dna_sequence": [nodes[i]["dna_sequence"] for i in sorted(nodes)]}
        ),
        branch_member_table=pl.DataFrame(
            {"dna_sequence": extra}, schema={"dna_sequence": pl.String}
        ),
        plot_data=SimpleNamespace(
            node_table=pl.DataFrame(
                [
                    {
                        "tree_node_index": i,
                        "x": float(i),
                        "y": float(nodes[i]["scalar"]),
                    }
                    for i in sorted(nodes)
                ]
            )
        ),
        branch_table=pl.DataFrame(branch_rows),
        persistence_table=pl.DataFrame(pair_rows),
    )


def _landscape(tree: rx.PyDiGraph, vertices: int | None = None) -> Any:
    """A landscape whose split tree represents ``vertices`` sequences (the summit's hill
    carries the members beyond the tree's own nodes)."""
    members = {}
    if vertices is not None:
        members = {"AAGGAAATG": max(vertices - tree.num_nodes(), 0)}
    return SimpleNamespace(trees={"split": _split_analysis(tree, members)})


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def fake_readers(output: Path, tree: rx.PyDiGraph | None = None) -> dict[str, Any]:
    """Stand-ins for the cached loaders that read the synthetic files of ``output``.

    Each raises ``FileNotFoundError`` when its files are absent, like the real ones.
    """
    clamped = output / "analysis" / "clamped_comparison"
    boot = output / "analysis" / "bootstrap_smoothed_tree"
    branches = output / "analysis" / "branch_decomposition"
    peak = output / "analysis" / "peak_uncertainty"

    def read_clamped() -> dict[str, Any]:
        provenance = _read(clamped / "provenance.json")
        summary = _read(clamped / "graph_summary.json")
        nodes = {
            name: summary["metrics"][name]["Number of nodes"]
            for name in ("Published", "Candidate")
        }
        return {
            "provenance": provenance,
            "diagnostics": _read(clamped / "diagnostics.json"),
            "graph_summary": summary,
            "candidate_threshold": _read(
                clamped / "candidate_functional_threshold.json"
            ),
            "landscapes": {
                "full": _landscape(_tree(extra_peaks=4)),
                "study_clamped": _landscape(
                    _tree(
                        extra_peaks=_read(clamped / "diagnostics.json")[
                            "study_clamped"
                        ]["peak_count"]
                        - 2
                    ),
                    nodes["Published"],
                ),
                "candidate_clamped": _landscape(tree or _tree(), nodes["Candidate"]),
            },
        }

    def read_component() -> tuple[pl.DataFrame, Any]:
        _read(output / "analysis" / "component_provenance.json")
        return (
            pl.read_parquet(output / "analysis" / "component_pairs.parquet"),
            _read(output / "analysis" / "component_diagnostics.json"),
        )

    def read_peak() -> dict[str, Any]:
        return {
            "provenance": _read(peak / "provenance.json"),
            "diagnostics": _read(peak / "diagnostics.json"),
        }

    def read_bootstrap() -> SimpleNamespace:
        return SimpleNamespace(
            summary=_read(boot / "summary.json"),
            provenance={},
            draw_stats=pl.read_parquet(boot / "draw_stats.parquet"),
        )

    def read_branches() -> SimpleNamespace:
        _read(branches / "provenance.json")
        return SimpleNamespace(
            summary=_read(branches / "summary.json"),
            provenance=_read(branches / "provenance.json"),
            branches=pl.read_parquet(branches / "branches.parquet"),
        )

    def read_band() -> SimpleNamespace:
        band = output / "checkpoints" / "leading_saddle_band"
        _read(band / "provenance.json")
        return SimpleNamespace(summary=_read(band / "summary.json"))

    def read_functional() -> SimpleNamespace:
        directory = output / "checkpoints" / "functional_classification"
        _read(directory / "provenance.json")
        return SimpleNamespace(summary=_read(directory / "summary.json"))

    def read_band_view(comparison: Any, bootstrap: Any) -> dict[str, Any]:
        # The reader gets both cached results; the synthetic counts differ from the
        # selected-leaf view's ridges.* so a mix-up of the two populations shows.
        assert comparison["provenance"] and bootstrap.summary
        candidate = comparison["graph_summary"]["metrics"]["Candidate"]
        return {
            "core": {
                "total_count": candidate["Number of nodes"],
                "functional_count": candidate["Number of functional sequences"],
            },
            "kem": {"total_count": 11, "functional_count": 8},
            "tcl": {"total_count": 7, "functional_count": 4},
        }

    return {
        "band_view": read_band_view,
        "clamped": read_clamped,
        "component": read_component,
        "peak": read_peak,
        "bootstrap": read_bootstrap,
        "branches": read_branches,
        "band": read_band,
        "functional_classification": read_functional,
        # The fingerprint the synthetic annotation export record must carry.
        "annotation_fingerprint": lambda: "annotation-fp",
    }


def load(output: Path, tree: rx.PyDiGraph | None = None) -> dict[str, Any]:
    return mq.load_sources(output, readers=fake_readers(output, tree))


def _graph_metrics(
    nodes: int,
    functional: int | None,
    threshold: float | None = None,
    uncertain: int | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "Number of nodes": nodes,
        "Number of edges": nodes * 3,
        "Minimum fitness": -0.2531585,
        "Maximum fitness": 0.456789,
        "Mean fitness": 0.12345,
        "Median fitness": 0.1,
        "Functional threshold": threshold,
        "Number of functional sequences": functional,
        "Number of non-functional sequences": None
        if functional is None
        else nodes - functional,
        "Percentage of functional sequences": None
        if functional is None
        else 100 * functional / nodes,
        "Number of uncertain sequences": uncertain,
    }
    return metrics


def write_upstream(
    output: Path,
    *,
    band: bool = False,
    published_peaks: int = 12,
    clamped_fingerprint: str = "clamped-fp",
    skip: tuple[str, ...] = (),
) -> None:
    """Write synthetic outputs of every upstream stage under ``output``."""
    if "clamped" not in skip:
        directory = output / "analysis" / "clamped_comparison"
        _write(
            directory / "provenance.json",
            {
                "fingerprint": clamped_fingerprint,
                "thresholds": {"study_clamped": -0.507774, "candidate_clamped": 0.7},
            },
        )
        _write(
            directory / "diagnostics.json",
            {
                "full": {"input_vertex_count": 1000},
                "study_clamped": {
                    "component_count": 4,
                    "selected_component_vertex_count": 100,
                    "peak_count": published_peaks,
                },
                "candidate_clamped": {
                    "component_count": 7,
                    "selected_component_vertex_count": 40,
                    "peak_count": 30,
                },
            },
        )
        _write(
            directory / "graph_summary.json",
            {
                "schema": 2,
                "metrics": {
                    "Full": _graph_metrics(1000, None),
                    "Published": _graph_metrics(100, 30, -0.507774),
                    "Candidate": _graph_metrics(40, 25, 0.7, uncertain=0),
                },
            },
        )
        _write(
            directory / "candidate_functional_threshold.json",
            {
                "rule": "median_reflected",
                "decimals": 6,
                "median": -0.1,
                "global_minimum": -0.9,
                "median_gap": 0.8,
                "unrounded_threshold": 0.7000004,
                "rounded_threshold": 0.7,
            },
        )
        (directory / "candidate_clamped" / "data").mkdir(parents=True, exist_ok=True)
    if "component" not in skip:
        _write(
            output / "analysis" / "component_provenance.json",
            {"fingerprint": "component-fp"},
        )
        _write(
            output / "analysis" / "component_diagnostics.json",
            {"join_tree_trailing_saddle": -0.3, "global_minimum": -0.9},
        )
        # Three split pairs and the global pair read as four ordinary H0 pairs; two
        # join pairs and the mirrored global pair read as three valleys.
        _pairs([0.3, 0.1, -0.4], [-0.9, -0.6]).write_parquet(
            output / "analysis" / "component_pairs.parquet"
        )
    if "peak" not in skip:
        directory = output / "analysis" / "peak_uncertainty"
        _write(directory / "provenance.json", {"fingerprint": "peak-fp"})
        _write(
            directory / "diagnostics.json",
            {
                "fitness_wt_scatter_row_count": 999,
                "peak_counts": {
                    "study_clamped": published_peaks,
                    "candidate_clamped": 30,
                },
            },
        )
    if "bootstrap" not in skip:
        directory = output / "analysis" / "bootstrap_smoothed_tree"
        _write(directory / "summary.json", bootstrap_summary())
        # 21 nominal-landscape draws: the 0.95 quantile is the 20th, c = 0.0123455
        # (the two largest values below the maximum are equal, so the linear
        # interpolation is exact); the smallest is 0.0001 and the largest 0.05.
        real = [0.0001, *[0.0002 * k for k in range(1, 18)], 0.0123455, 0.0123455, 0.05]
        flat = [0.1 * (k + 1) for k in range(21)]
        pl.DataFrame(
            {
                "condition": ["real"] * 21 + ["flat"] * 21,
                "bottleneck": [*real, *flat],
            }
        ).write_parquet(directory / "draw_stats.parquet")
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if "codon" not in skip:
        _write(
            tables / mq.CODON_METADATA,
            {
                "summits": {
                    "kem": {
                        "dna_sequence": "AAGGAAATG",
                        "aa_sequence": "KEM",
                        "raw_fitness": 0.4567,
                    },
                    "tcl": {
                        "dna_sequence": "ACCTGTCTT",
                        "aa_sequence": "TCL",
                        "raw_fitness": 0.3333,
                    },
                },
                "common_saddle": {
                    "dna_sequence": "CTCTGCTTG",
                    "aa_sequence": "LCL",
                    "raw_fitness": -0.2531585,
                },
                "counts": {
                    "core": 40,
                    "kem_ridge": 10,
                    "tcl_ridge": 6,
                    "kem_functional": 9,
                    "tcl_functional": 5,
                },
                "functional_threshold": 0.7,
                "source_fingerprint": "clamped-fp",
            },
        )
        pl.DataFrame(
            [
                {
                    "claim_id": "functional_sequences_in_two_ridges",
                    "numerator": 14,
                    "denominator": 25,
                    "fraction": 0.56,
                }
            ]
        ).write_parquet(tables / mq.CODON_CLAIMS)
        entropy = [
            {
                "population_id": population,
                "fitness_group": group,
                "molecule": "amino_acid",
                "site_index": site,
                "entropy_bits": 0.5 * site,
            }
            for population in ("core", "kem", "tcl")
            for group in ("all", "functional", "fringe")
            for site in (1, 2, 3)
        ]
        pl.DataFrame(entropy).write_parquet(tables / mq.CODON_SITE_ENTROPY)
        _write(tables / mq.CODON_SIDECAR, {"fingerprint": "codon-fp"})
    if "annotations" not in skip:
        _write(
            tables / mq.ANNOTATION_COUNTS,
            {
                "landscape": "study_clamped",
                "peak_count": published_peaks,
                "peak_denominator": published_peaks,
                "stop_codon_peak_count": 2,
                "singleton_peak_count": 6,
                "synonymous_endpoint_peak_count": 3,
                "adjacent_saddle_peak_denominator": 11,
            },
        )
        pl.DataFrame(
            {"paired_synonymous_endpoints": [True, False, True, False]}
        ).write_parquet(tables / mq.ANNOTATION_PAIRS)
        _write(tables / mq.ANNOTATION_SIDECAR, {"fingerprint": "annotation-fp"})
    if "branches" not in skip:
        directory = output / "analysis" / "branch_decomposition"
        _write(directory / "provenance.json", {"fingerprint": "branch-fp"})
        _write(
            directory / "summary.json",
            {
                "settings": {"tree": "band"},
                "terminal_branch_count": 3,
                "internal_branch_count": 2,
            },
        )
        pl.DataFrame(
            {
                "is_terminal": [True, True, True, False, False],
                "branch_vertex_count": [5, 7, 1, 20, 4],
            }
        ).write_parquet(directory / "branches.parquet")
    if band:
        write_band(output)


def bootstrap_summary() -> dict[str, Any]:
    """A synthetic bootstrap ``summary.json``: settings, band simplification, verdict."""
    return {
        "fingerprint": {"a": 1, "schema": 6},
        "settings": {"n_replicates": 10000, "seed": 20261019, "band_quantile": 0.95},
        "band_simplification": {
            "c": 0.0123455,
            "two_c": 0.024691,
            "peaks_before": 30,
            "peaks_after_core_rule": 20,
            "peaks_after": 8,
            "saddles_removed": 31,
        },
        "diagnostics": {
            "global_verdict": {
                "band_quantile": 0.95,
                "c": 0.0123455,
                "two_c": 0.024691,
                "nominal_point_count": 30,
                "significant_point_count": 8,
                "relative_wasserstein_1": {"median": 0.08345},
                "sanity_check": {
                    "real": {"median": 0.0102},
                    "flat": {"median": 0.15215},
                    "flat_bottleneck_p5": 0.0987,
                    "real_share_below_flat_bottleneck_p5": 0.9987,
                },
            }
        },
    }


def band_summary() -> dict[str, Any]:
    """A synthetic ``summary.json`` in the layout of the leading-saddle-band stage."""
    fitness = {
        "global_minimum": -0.9,
        "split_trailing_saddle": -0.35,
        "join_trailing_saddle": -0.3,
        "median": -0.1,
        "split_leading_saddle": 0.2,
        "join_leading_saddle": 0.25,
        "published_functional_threshold": -0.507774,
        "candidate_functional_threshold": 0.7,
    }
    sequences = {
        "global_minimum": "AAAAAAAAA",
        "split_trailing_saddle": "CCCCCCCCC",
        "join_trailing_saddle": "GGGGGGGGG",
        "split_leading_saddle": "TTTTTTTTT",
        "join_leading_saddle": "ACACACACA",
    }
    order = [
        "global_minimum__published_functional_threshold",
        "published_functional_threshold__split_trailing_saddle",
        "join_trailing_saddle__median",
    ]
    summary: dict[str, Any] = {"schema": 1}
    for key, value in fitness.items():
        summary[f"{key}_fitness"] = value
        summary[f"{key}_sequence"] = sequences.get(key)
    summary.update(
        band_lower_landmark="split_trailing_saddle",
        band_lower=-0.35,
        band_upper=-0.3,
        median_inside_band=False,
        band_count=1234,
        band_percentage=0.4707,
        band_entropy_sum=5.5,
        split_trunk_count=100,
        join_trunk_count=200,
        split_trunk_percentage=0.0381,
        join_trunk_percentage=0.0763,
        split_trunk_entropy_sum=4.25,
        join_trunk_entropy_sum=None,
        interval_order=order,
        interval_counts=dict(zip(order, (11, 22, 33))),
        interval_percentages=dict(zip(order, (0.25, 0.35, 0.45))),
        interval_entropy_sums=dict(zip(order, (1.5, None, 2.5))),
        below_first_landmark_count=7,
        above_last_landmark_count=8,
        interval_convention="half-open [lower, upper), the last interval closed",
        total_sequence_count=262144,
        median_gap=0.8,
        candidate_unrounded_threshold=0.7000004,
        candidate_threshold_rule="median_reflected",
        candidate_threshold_decimals=6,
    )
    return summary


def write_band(output: Path, summary: dict[str, Any] | None = None) -> None:
    directory = output / "checkpoints" / "leading_saddle_band"
    _write(directory / "summary.json", summary or band_summary())
    _write(
        directory / "provenance.json",
        {"fingerprint": "band-fp", "artefacts": {"summary.json": "abc"}},
    )


def rows_of(
    output: Path, tree: rx.PyDiGraph | None = None
) -> dict[str, dict[str, Any]]:
    sources = load(output, tree or _tree())
    rows = mq.build_rows(sources, CONFIG, git_commit="abc123")
    return {row["key"]: row for row in rows}


# ---------------------------------------------------------------- keys


def test_key_format_and_uniqueness_are_enforced() -> None:
    good = {"key": "candidate_graph.peak_count"}
    mq.validate_rows([good, {"key": "thresholds.candidate_functional.value"}])
    for bad in ("Candidate.peak", "nodots", "a..b", "a.B", "a.b-c", ".a.b", "a.b."):
        with pytest.raises(ValueError, match="not dotted"):
            mq.validate_rows([{"key": bad}])
    with pytest.raises(ValueError, match="more than once"):
        mq.validate_rows([good, good])


def _rows_under(tmp_path: Path, rule: str | None) -> dict[str, dict[str, Any]]:
    write_upstream(tmp_path, band=True)
    config = CONFIG
    if rule is not None:
        config = {
            **CONFIG,
            "clamped": {
                **CONFIG["clamped"],
                "candidate_functional_threshold": {"rule": rule, "decimals": 6},
            },
        }
    rows = mq.build_rows(load(tmp_path, _tree()), config, git_commit=None)
    return {row["key"]: row for row in rows}


@pytest.mark.parametrize("rule", [None, "mixture_quantile"])
def test_the_candidate_rows_name_the_mixture_rule(
    tmp_path: Path, rule: str | None
) -> None:
    # mixture_quantile is the default when the configuration does not say.
    rows = _rows_under(tmp_path, rule)
    text = rows["candidate_graph.largest_component_vertex_count"]["selection_rule"]
    assert text.startswith("mixture_quantile:")
    assert "raw fitness >=" not in text
    components = rows["candidate_graph.component_count"]["selection_rule"]
    assert components.startswith("mixture_quantile:")
    assert "Functional-class endpoint" in components
    assert "Functional class" in rows["candidate_graph.vertex_count"]["selection_rule"]
    # The Published graph keeps its raw-fitness rule.
    assert rows["published_graph.vertex_count"]["selection_rule"].startswith(
        "raw fitness >= published threshold"
    )


def test_the_candidate_rows_name_the_median_comparison_rule(tmp_path: Path) -> None:
    rows = _rows_under(tmp_path, "median_reflected")
    text = rows["candidate_graph.largest_component_vertex_count"]["selection_rule"]
    assert text.startswith("median_reflected:")
    components = rows["candidate_graph.component_count"]["selection_rule"]
    assert "endpoint at or above the candidate threshold" in components
    assert rows["candidate_graph.vertex_count"]["selection_rule"].startswith(
        "median_reflected: raw fitness >= candidate threshold"
    )


def test_the_ridge_functional_counts_say_what_they_count(tmp_path: Path) -> None:
    mixture = _rows_under(tmp_path / "mixture", "mixture_quantile")
    for key in ("ridges.kem.functional_count", "ridges.two_functional_share.fraction"):
        text = mixture[key]["selection_rule"]
        assert "Functional class (mixture_quantile)" in text
        assert "raw fitness" not in text
    median = _rows_under(tmp_path / "median", "median_reflected")
    for key in ("ridges.kem.functional_count", "ridges.two_functional_share.fraction"):
        assert (
            "raw fitness at or above the Candidate threshold"
            in (median[key]["selection_rule"])
        )


def test_the_candidate_counts_say_what_they_count(tmp_path: Path) -> None:
    mixture = _rows_under(tmp_path / "mixture", "mixture_quantile")
    count = mixture["candidate_graph.functional_count"]["selection_rule"]
    assert "Functional class" in count and "not raw fitness" in count
    assert (
        "Non-functional class"
        in (mixture["candidate_graph.nonfunctional_count"]["selection_rule"])
    )
    assert (
        "Functional class"
        in (mixture["candidate_graph.functional_percent"]["selection_rule"])
    )
    uncertain = mixture["candidate_graph.uncertain_count"]
    assert (
        uncertain["value"] == "0" and "Uncertain class" in uncertain["selection_rule"]
    )
    assert mixture["candidate_graph.functional_count"]["value"] == "25"
    # The Published graph keeps the raw-fitness count at its threshold.
    assert (
        mixture["published_graph.functional_count"]["selection_rule"]
        == mq.FUNCTIONAL_RULE
    )
    assert "published_graph.uncertain_count" not in mixture
    median = _rows_under(tmp_path / "median", "median_reflected")
    assert (
        median["candidate_graph.functional_count"]["selection_rule"]
        == mq.FUNCTIONAL_RULE
    )
    assert "candidate_graph.uncertain_count" not in median


def test_the_classification_rows_state_their_own_rule(tmp_path: Path) -> None:
    rows = _rows_under(tmp_path, None)
    for key in (
        "functional_classification.functional_class_count",
        "functional_classification.uncertain_class_count",
        "functional_classification.double_stop_fraction_below_threshold",
    ):
        assert rows[key]["selection_rule"].startswith("mixture_quantile:"), key


def test_every_emitted_key_is_valid_and_unique(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=True)
    sources = load(tmp_path, _tree())
    rows = mq.build_rows(sources, CONFIG, git_commit=None)
    keys = [row["key"] for row in rows]
    assert len(keys) == len(set(keys))
    assert all(mq.KEY_PATTERN.fullmatch(key) for key in keys)
    assert list(rows[0]) == list(mq.COLUMNS)


def test_manuscript_order_names_only_emitted_keys_with_and_without_band(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path / "with", band=True)
    write_upstream(tmp_path / "without", band=False)
    for name in ("with", "without"):
        keys = set(rows_of(tmp_path / name))
        assert set(mq.MANUSCRIPT_ORDER) <= keys
    assert len(mq.MANUSCRIPT_ORDER) == len(set(mq.MANUSCRIPT_ORDER))


# ------------------------------------------------------------ rounding


@pytest.mark.parametrize(
    ("text", "unit", "role", "expected"),
    [
        ("0.12345", "fitness", None, (6, "0.123450")),
        ("-0.2531585", "fitness", None, (6, "-0.253159")),  # half up, not -0.253158
        ("0.2531585", "fitness", None, (6, "0.253159")),
        ("0.123456789", "fitness", None, (6, "0.123457")),
        ("33.35", "percent", None, (1, "33.4")),  # exact text: no binary float error
        ("12.25", "percent", None, (1, "12.3")),
        ("0.4567", "fitness", "summit_fitness", (2, "0.46")),
        ("-0.2531585", "fitness", "summit_fitness", (2, "-0.25")),
        ("12345", "count", None, (0, "12,345")),  # counts carry separators
        ("262144", "count", None, (0, "262,144")),
        ("999", "count", None, (0, "999")),
        ("-1234", "count", None, (0, "-1,234")),
        ("0", "count", None, (0, "0")),
        ("20261019", "seed", None, (0, "20261019")),  # a seed stays verbatim
        ("0.95", "quantile", None, (None, "0.95")),
        ("true", "boolean", None, (None, "true")),
        ("AAGGAAATG", "dna_sequence", None, (None, "AAGGAAATG")),
        ("0.9", "fraction", "verbatim", (None, "0.9")),
        ("0.5625", "fraction", None, (4, "0.5625")),
        ("-0.0000001", "fitness", None, (6, "0.000000")),
    ],
)
def test_display_rounds_half_up_and_leaves_counts_alone(
    text: str, unit: str, role: str | None, expected: tuple[int | None, str]
) -> None:
    assert mq.display_spec(text, unit, role) == expected


def test_a_display_value_never_carries_a_unit_symbol() -> None:
    for text, unit in (("33.35", "percent"), ("12", "count"), ("0.5", "fraction")):
        shown = mq.display_spec(text, unit)[1]
        assert shown is not None and "%" not in shown


def test_the_schema_states_what_display_value_holds() -> None:
    text = mq.schema_description()["display_value"]
    assert "unit symbol" in text and "thousands separators" in text


def test_display_has_no_rule_for_an_unknown_unit_and_passes_none_through() -> None:
    assert mq.display_spec(None, "fitness") == (None, None)
    with pytest.raises(ValueError, match="No display rule"):
        mq.display_spec("1.0", "furlongs")


def test_value_text_is_exact_and_single_typed() -> None:
    assert mq.value_text(7) == "7"
    assert mq.value_text(0.1) == "0.1"
    assert mq.value_text(True) == "true" and mq.value_text(False) == "false"
    assert mq.value_text("AC") == "AC"
    with pytest.raises(ValueError, match="finite"):
        mq.value_text(float("nan"))


def test_rows_carry_exact_value_and_rounded_display(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    saddle_fitness = rows["common_saddle.raw_fitness"]
    assert saddle_fitness["value"] == "-0.2531585"
    assert saddle_fitness["display_value"] == "-0.25"
    assert saddle_fitness["display_decimals"] == 2
    assert rows["candidate_graph.vertex_count"]["display_value"] == "40"
    assert rows["candidate_graph.vertex_count"]["display_decimals"] == 0
    assert rows["candidate_graph.min_fitness"]["display_value"] == "-0.253159"
    assert rows["candidate_graph.functional_percent"]["display_value"] == "62.5"
    assert rows["summits.kem.raw_fitness"]["display_value"] == "0.46"
    assert rows["thresholds.published_functional.value"]["display_value"] == "-0.507774"
    assert rows["bootstrap.seed"]["git_commit"] == "abc123"


# ----------------------------------------------------------- the topics


def test_thresholds_graph_sizes_and_ordering(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    assert rows["thresholds.candidate_functional.value"]["display_value"] == "0.700000"
    assert rows["thresholds.join_tree_trailing_saddle.value"]["value"] == "-0.3"
    ordering = rows[
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
    ]
    assert ordering["value"] == "true"  # -0.9 < -0.3 < -0.1 < 0.7
    assert ordering["source_stage"] == f"{mq.CLAMPED}+{mq.COMPONENT}"
    assert ordering["source_fingerprint"] == "clamped-fp+component-fp"
    assert "thresholds.full_median.value" in ordering["settings"]
    assert rows["published_graph.component_count"]["value"] == "4"
    assert rows["candidate_graph.largest_component_vertex_count"]["value"] == "40"
    assert rows["candidate_graph.peak_count"]["value"] == "30"
    assert (
        "largest connected component"
        in rows["published_graph.vertex_count"]["selection_rule"]
    )
    assert rows["full_graph.vertex_count"]["selection_rule"].startswith("no threshold")
    # Functional populations carry an explicit denominator and their threshold.
    assert rows["candidate_graph.functional_denominator"]["value"] == "40"
    assert rows["candidate_graph.functional_count"]["settings"] == (
        "thresholds.candidate_functional.value"
    )
    assert rows["candidate_graph.nonfunctional_count"]["value"] == "15"


def test_ordering_is_false_when_the_trailing_saddle_leaves_the_order(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    _write(
        tmp_path / "analysis" / "component_diagnostics.json",
        {"join_tree_trailing_saddle": -0.5, "global_minimum": -0.9},
    )
    assert (
        rows_of(tmp_path)[
            "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
        ]["value"]
        == "true"
    )
    _write(
        tmp_path / "analysis" / "component_diagnostics.json",
        {"join_tree_trailing_saddle": 0.9, "global_minimum": -0.9},
    )
    assert (
        rows_of(tmp_path)[
            "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
        ]["value"]
        == "false"
    )


def test_disagreeing_global_minimum_or_peak_counts_raise(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    _write(
        tmp_path / "analysis" / "component_diagnostics.json",
        {"join_tree_trailing_saddle": -0.3, "global_minimum": -0.8},
    )
    with pytest.raises(ValueError, match="global_minimum"):
        rows_of(tmp_path)
    write_upstream(tmp_path)
    _write(
        tmp_path / "analysis" / "peak_uncertainty" / "diagnostics.json",
        {
            "fitness_wt_scatter_row_count": 999,
            "peak_counts": {"study_clamped": 99, "candidate_clamped": 30},
        },
    )
    with pytest.raises(ValueError, match="peak count differs"):
        rows_of(tmp_path)


def test_bootstrap_states_and_absorbed_counts(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    assert rows["bootstrap.candidate.nominal.peak_count"]["value"] == "30"
    assert rows["bootstrap.candidate.after_core_rule.peak_count"]["value"] == "20"
    assert (
        rows["bootstrap.candidate.after_core_rule.peak_count"]["selection_rule"]
        == "after the peak-core rule (step 1)"
    )
    assert rows["bootstrap.candidate.after_band_rule.peak_count"]["value"] == "8"
    assert rows["bootstrap.candidate.after_band_rule.peak_count"]["selection_rule"] == (
        "after the band rule (step 2): persistence strictly greater than the value "
        "(nominal persistence > 2c)"
    )
    assert rows["bootstrap.candidate.after_core_rule.absorbed_count"]["value"] == "10"
    assert rows["bootstrap.candidate.after_band_rule.absorbed_count"]["value"] == "12"
    assert rows["bootstrap.draws"]["value"] == "10000"
    assert rows["bootstrap.seed"]["display_value"] == "20261019"
    assert rows["bootstrap.band_quantile"]["display_value"] == "0.95"
    assert rows["bootstrap.persistence_threshold.radius"]["display_value"] == "0.012346"
    assert rows["bootstrap.persistence_threshold.value"]["display_value"] == "0.024691"
    assert (
        "strictly greater"
        in rows["bootstrap.persistence_threshold.value"]["selection_rule"]
    )
    assert not [key for key in rows if "selected_leaf" in key]


def test_ridge_counts_share_and_boolean(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    assert rows["ridges.kem.total_count"]["value"] == "10"
    assert rows["ridges.kem.nonfunctional_count"]["value"] == "1"
    assert rows["ridges.tcl.nonfunctional_count"]["value"] == "1"
    assert rows["ridges.union.total_count"]["value"] == "16"
    assert rows["ridges.union.functional_count"]["value"] == "14"
    assert rows["ridges.two_functional_share.numerator"]["value"] == "14"
    assert rows["ridges.two_functional_share.denominator"]["value"] == "25"
    assert rows["ridges.two_functional_share.fraction"]["display_value"] == "0.5600"
    assert rows["ridges.two_functional_share.minimum"]["display_value"] == "0.9"
    assert rows["ridges.two_functional_share.meets_minimum"]["value"] == "false"
    # The share is exactly the minimum: >= holds.
    claims = pl.DataFrame(
        [
            {
                "claim_id": "functional_sequences_in_two_ridges",
                "numerator": 14,
                "denominator": 14,
                "fraction": 0.9,
            }
        ]
    )
    claims.write_parquet(tmp_path / "tables" / mq.CODON_CLAIMS)
    assert (
        rows_of(tmp_path)["ridges.two_functional_share.meets_minimum"]["value"]
        == "true"
    )


def test_ridge_numerator_must_match_the_ridge_functional_counts(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    pl.DataFrame(
        [
            {
                "claim_id": "functional_sequences_in_two_ridges",
                "numerator": 13,
                "denominator": 25,
                "fraction": 0.52,
            }
        ]
    ).write_parquet(tmp_path / "tables" / mq.CODON_CLAIMS)
    with pytest.raises(ValueError, match="numerator"):
        rows_of(tmp_path)


def test_entropy_rows_state_their_population_and_sum_the_positions(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    assert "entropy.kem_ridge.all_sequences.aa_site_2" not in rows
    assert rows["ridges.kem.functional_count"]["gap_reason"] is None
    assert rows["ridges.tcl.nonfunctional_count"]["gap_reason"] is None
    assert rows["candidate_graph.functional_count"]["gap_reason"] is None
    table = mq.position_entropy_frame(load(tmp_path, _tree()))
    site = table.filter(
        (pl.col("population") == "kem_ridge")
        & (pl.col("fitness_group") == "all_sequences")
        & (pl.col("site") == "aa_site_2")
    )
    assert site["entropy_bits"][0] == 1.0
    total = table.filter(
        (pl.col("population") == "kem_ridge")
        & (pl.col("fitness_group") == "all_sequences")
        & (pl.col("site") == "aa_sum")
    )
    assert total["entropy_bits"][0] == 3.0
    assert total["source_fingerprint"][0] == "codon-fp"
    assert table.filter(pl.col("population") == "candidate").height > 0


def test_annotations_branches_and_summits(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    assert rows["annotations.published.stop_codon_peak_count"]["value"] == "2"
    assert rows["annotations.published.stop_codon_peak_denominator"]["value"] == "12"
    assert rows["annotations.published.synonymous_peak_count"]["value"] == "3"
    assert rows["annotations.published.synonymous_peak_denominator"]["value"] == "11"
    assert rows["annotations.published.synonymous_partner_peak_count"]["value"] == "2"
    assert (
        rows["annotations.published.synonymous_partner_peak_denominator"]["value"]
        == "4"
    )
    assert rows["annotations.published.singleton_peak_count"]["value"] == "6"
    # The Candidate counts come from the same annotation function on its split tree:
    # thirty peaks, one with a stop codon, one synonymous with the saddle, 29 singletons.
    candidate = {
        key.removeprefix("annotations.candidate."): row
        for key, row in rows.items()
        if key.startswith("annotations.candidate.")
    }
    assert {
        name: candidate[name]["value"]
        for name in (
            "stop_codon_peak_count",
            "stop_codon_peak_denominator",
            "synonymous_peak_count",
            "synonymous_peak_denominator",
            "synonymous_partner_peak_count",
            "synonymous_partner_peak_denominator",
            "singleton_peak_count",
            "singleton_peak_denominator",
        )
    } == {
        "stop_codon_peak_count": "1",
        "stop_codon_peak_denominator": "30",
        "synonymous_peak_count": "1",
        "synonymous_peak_denominator": "30",
        "synonymous_partner_peak_count": "1",
        "synonymous_partner_peak_denominator": "30",
        "singleton_peak_count": "29",
        "singleton_peak_denominator": "30",
    }
    assert candidate["stop_codon_peak_count"]["source_stage"] == mq.CLAMPED
    assert candidate["stop_codon_peak_count"]["source_fingerprint"] == "clamped-fp"
    assert candidate["stop_codon_peak_count"]["display_value"] == "1"
    # The core-rule and band-rule trees are not annotated: named gaps.
    for state in ("candidate_bootstrap_core", "candidate_bootstrap_band"):
        gap = rows[f"annotations.{state}.singleton_peak_count"]
        assert gap["value"] is None and gap["display_value"] is None
        assert "Published and Candidate graphs only" in gap["gap_reason"]
    assert rows["branches.terminal_count"]["value"] == "3"
    assert rows["branches.internal_count"]["value"] == "2"
    assert rows["branches.terminal_sequence_count"]["value"] == "13"
    assert rows["branches.internal_sequence_count"]["value"] == "24"
    assert rows["summits.kem.dna_sequence"]["value"] == "AAGGAAATG"
    assert rows["summits.tcl.aa_sequence"]["value"] == "TCL"
    assert rows["common_saddle.dna_sequence"]["value"] == "CTCTGCTTG"
    assert rows["common_saddle.aa_sequence"]["value"] == "LCL"


# ------------------------------------------------------------ the saddle


def test_lowest_common_ancestor_on_a_small_tree() -> None:
    tree = _tree()
    by_dna = {tree[i]["dna_sequence"]: i for i in tree.node_indices()}
    kem, tcl = by_dna["AAGGAAATG"], by_dna["ACCTGTCTT"]
    saddle, root = by_dna["CTCTGCTTG"], by_dna["GGGGGGGGG"]
    assert lowest_common_ancestor(tree, kem, tcl) == saddle
    assert lowest_common_ancestor(tree, tcl, kem) == saddle
    assert lowest_common_ancestor(tree, kem, saddle) == saddle
    assert lowest_common_ancestor(tree, kem, root) == root


def test_common_saddle_is_computed_and_must_equal_the_reported_sequence() -> None:
    found = mq.common_saddle(_tree(), "AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG")
    assert found == {
        "dna_sequence": "CTCTGCTTG",
        "aa_sequence": "LCL",
        "raw_fitness": -0.2531585,
    }
    with pytest.raises(ValueError, match="differs from the reported"):
        mq.common_saddle(_tree(), "AAGGAAATG", "ACCTGTCTT", "AAAAAAAAA")
    with pytest.raises(ValueError, match="differs from the reported"):
        mq.common_saddle(_tree("TTTTTTTTT"), "AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG")
    with pytest.raises(ValueError, match="absent"):
        mq.common_saddle(_tree(), "AAGGAAATG", "CCCCCCCCC", "CTCTGCTTG")
    with pytest.raises(ValueError, match="distinct shared saddle"):
        mq.common_saddle(_tree(), "AAGGAAATG", "CTCTGCTTG", "CTCTGCTTG")


def test_the_stage_raises_when_the_tree_saddle_differs_from_the_configuration(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    with pytest.raises(ValueError, match="differs from the reported"):
        rows_of(tmp_path, _tree("TTTTTTTTT"))


def test_the_computed_saddle_must_match_the_codon_entropy_metadata(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    path = tmp_path / "tables" / mq.CODON_METADATA
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["common_saddle"]["dna_sequence"] = "AAAAAAAAA"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="codon-entropy"):
        rows_of(tmp_path)


# ---------------------------------------------------------------- gaps


def test_absent_upstreams_become_named_gap_rows(tmp_path: Path) -> None:
    write_upstream(tmp_path, skip=("clamped", "bootstrap", "branches", "codon"))
    sources = load(tmp_path, _tree())
    assert sources["clamped"] is None and sources["candidate_tree"] is None
    rows = {r["key"]: r for r in mq.build_rows(sources, CONFIG, git_commit="x")}
    for key in (
        "thresholds.candidate_functional.value",
        "candidate_graph.vertex_count",
        "bootstrap.persistence_threshold.value",
        "bootstrap.candidate.after_band_rule.absorbed_count",
        "branches.terminal_count",
        "summits.kem.dna_sequence",
        "ridges.union.total_count",
        "ridges.kem.functional_count",
        "common_saddle.raw_fitness",
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value",
    ):
        row = rows[key]
        assert row["value"] is None and row["display_value"] is None, key
        assert row["display_decimals"] is None
        assert row["gap_reason"], key
        assert row["source_fingerprint"] is None
    # Sourced rows of present upstreams are untouched, config rows still sourced.
    assert rows["thresholds.join_tree_trailing_saddle.value"]["gap_reason"] is None
    assert rows["thresholds.published_functional.value"]["gap_reason"] is None
    assert rows["annotations.published.stop_codon_peak_count"]["value"] == "2"


def test_full_graph_components_are_named_gaps(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    for key in (
        "full_graph.component_count",
        "full_graph.largest_component_vertex_count",
    ):
        # Sourced from the saved landscape summary product: a not-run gap until it exists.
        assert rows[key]["value"] is None
        assert mq.is_stage_not_run(rows[key]["gap_reason"])
        assert mq.LANDSCAPE_SUMMARY in rows[key]["gap_reason"]


def test_leading_saddle_band_absent_gives_named_gap_rows(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=False)
    rows = rows_of(tmp_path)
    band_keys = [key for key in rows if key.startswith("leading_saddle_band.")]
    for key in (
        "leading_saddle_band.split_trailing_saddle.fitness",
        "leading_saddle_band.join_trailing_saddle.sequence",
        "leading_saddle_band.median_inside_band",
        "leading_saddle_band.band.count",
        "leading_saddle_band.split_trunk.entropy_sum",
        "leading_saddle_band.join_trunk.percentage",
        "leading_saddle_band.total_sequence_count",
        "leading_saddle_band.intervals.count",
    ):
        assert key in band_keys, key
    for key in band_keys:
        assert rows[key]["value"] is None and rows[key]["display_value"] is None
        assert rows[key]["gap_reason"].startswith(
            "missing: stage leading-saddle-band not run: "
        )
        assert "rerun just leading-saddle-band" in rows[key]["gap_reason"]
        assert rows[key]["source_stage"] == "leading-saddle-band"
    # Per-position entropy is a table file. Ridge class counts stay registry rows.
    assert rows["ridges.kem.functional_count"]["gap_reason"] is None
    table = mq.position_entropy_frame(load(tmp_path, _tree()))
    assert table.filter(pl.col("site") == "aa_sum")["entropy_bits"].null_count() == 0


def test_leading_saddle_band_rows_follow_the_stage_summary_layout(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path, band=True)
    sources = load(tmp_path, _tree())
    assert sources["band"].fingerprint == "band-fp"
    assert sources["band"].identity == {
        "fingerprint": "band-fp",
        "artefacts": {"summary.json": "abc"},
    }
    rows = rows_of(tmp_path)
    prefix = "leading_saddle_band."
    band_rows = {k: r for k, r in rows.items() if k.startswith(prefix)}
    summary = band_summary()
    # One row per landmark fitness, and per sequence where the landmark has one.
    for landmark in mq.BAND_LANDMARKS:
        row = band_rows[f"{prefix}{landmark}.fitness"]
        assert row["value"] == repr(summary[f"{landmark}_fitness"])
        assert row["unit"] == "fitness" and row["display_decimals"] == 6
        assert row["source_stage"] == "leading-saddle-band"
        assert row["source_fingerprint"] == "band-fp"
        assert row["gap_reason"] is None
    assert band_rows[f"{prefix}global_minimum.sequence"]["value"] == "AAAAAAAAA"
    assert f"{prefix}median.sequence" not in band_rows  # null in the summary
    assert (
        band_rows[f"{prefix}split_trailing_saddle.fitness"]["display_value"]
        == "-0.350000"
    )
    assert band_rows[f"{prefix}median_gap"]["value"] == "0.8"
    assert band_rows[f"{prefix}candidate_threshold_rule"]["value"] == "median_reflected"
    assert band_rows[f"{prefix}candidate_threshold_decimals"]["value"] == "6"
    assert band_rows[f"{prefix}total_sequence_count"]["value"] == "262144"
    # Band, trunks, outside counts, convention.
    assert band_rows[f"{prefix}band.lower_landmark"]["value"] == "split_trailing_saddle"
    assert band_rows[f"{prefix}band.lower"]["value"] == "-0.35"
    assert band_rows[f"{prefix}band.upper"]["value"] == "-0.3"
    assert band_rows[f"{prefix}median_inside_band"]["value"] == "false"
    assert band_rows[f"{prefix}band.count"]["value"] == "1234"
    assert band_rows[f"{prefix}band.percentage"]["display_value"] == "0.5"
    assert band_rows[f"{prefix}band.entropy_sum"]["display_value"] == "5.5000"
    assert band_rows[f"{prefix}split_trunk.count"]["value"] == "100"
    assert band_rows[f"{prefix}join_trunk.count"]["value"] == "200"
    assert band_rows[f"{prefix}split_trunk.entropy_sum"]["value"] == "4.25"
    assert band_rows[f"{prefix}below_first_landmark_count"]["value"] == "7"
    assert band_rows[f"{prefix}above_last_landmark_count"]["value"] == "8"
    assert "half-open" in band_rows[f"{prefix}interval_convention"]["value"]
    # A null entropy sum is a gap with its own reason, not the "absent" reason.
    null_entropy = band_rows[f"{prefix}join_trunk.entropy_sum"]
    assert (
        null_entropy["value"] is None
        and "empty population" in null_entropy["gap_reason"]
    )
    # One count, percentage and entropy row per interval, named by the stage.
    for name, count, percentage, entropy in (
        ("global_minimum__published_functional_threshold", "11", "0.3", "1.5000"),
        ("join_trailing_saddle__median", "33", "0.5", "2.5000"),
    ):
        base = f"{prefix}interval.{name}"
        assert band_rows[f"{base}.count"]["value"] == count
        assert band_rows[f"{base}.percentage"]["display_value"] == percentage
        assert band_rows[f"{base}.entropy_sum"]["display_value"] == entropy
    middle = band_rows[
        f"{prefix}interval.published_functional_threshold__split_trailing_saddle.entropy_sum"
    ]
    assert middle["value"] is None and "is empty" in middle["gap_reason"]
    assert f"{prefix}intervals.count" not in band_rows  # the absent-run placeholder
    # Nothing from per-ridge entropy leaks into the band rows.
    assert not [k for k in band_rows if "ridge" in k]


def test_a_malformed_present_file_raises(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    (tmp_path / "analysis" / "component_diagnostics.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        load(tmp_path, _tree())
    write_upstream(tmp_path)
    _write(
        tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json",
        {
            "fingerprint": "x",
            "settings": {},
            "band_simplification": {"peaks_before": 30},
        },
    )
    with pytest.raises(KeyError, match="n_replicates"):
        rows_of(tmp_path)
    write_upstream(tmp_path)
    (tmp_path / "tables" / mq.CODON_CLAIMS).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load(tmp_path, _tree())


def test_source_identities_distinguish_present_from_absent(tmp_path: Path) -> None:
    write_upstream(tmp_path, skip=("peak",))
    sources = load(tmp_path, _tree())
    identities = mq.source_identities(sources)
    assert identities["peak"] is None
    clamped = identities["clamped"]
    assert clamped["stage"] == mq.CLAMPED and clamped["fingerprint"] == "clamped-fp"
    assert set(clamped["files"]) >= {
        "analysis/clamped_comparison/provenance.json",
        "analysis/clamped_comparison/diagnostics.json",
    }
    assert identities["bootstrap"]["fingerprint"] == mq.fingerprint_text(
        bootstrap_summary()["fingerprint"]
    )


# ----------------------------------------------------- report and changes


def _row(key: str, shown: str | None, reason: str | None = None) -> dict[str, Any]:
    return {
        "key": key,
        "display_value": shown,
        "unit": "count",
        "gap_reason": reason,
    }


def test_changes_flag_changed_new_and_removed_keys() -> None:
    previous = {"a.one": "1", "a.two": "2", "a.old": "9", "a.gap": None}
    rows = [
        _row("a.one", "1"),
        _row("a.two", "3"),
        _row("a.new", "5"),
        _row("a.gap", "4"),
    ]
    flags, removed = mq.changes(rows, previous)
    assert flags == {
        "a.two": "changed (was 2)",
        "a.new": "new",
        "a.gap": "changed (was gap)",
    }
    assert removed == ["a.old"]
    assert mq.changes(rows, None) == ({}, [])


def test_report_has_topic_headings_flags_and_the_ordered_numbers(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    rows = list(rows_of(tmp_path).values())
    previous = {row["key"]: row["display_value"] for row in rows}
    previous["candidate_graph.vertex_count"] = "39"
    previous["retired.key"] = "1"
    report = mq.render_report(rows, previous)
    for heading, _ in mq.TOPICS:
        assert f"## {heading}" in report
    assert "changed (was 39)" in report
    assert "`retired.key`" in report
    assert "gap: " in report
    numbers = report.split("## Manuscript numbers")[1]
    lines = [ln for ln in numbers.splitlines() if " = " in ln]
    keys = [ln.split(" = ")[0] for ln in lines]
    listed = [key for key in mq.MANUSCRIPT_ORDER if key in set(keys)]
    assert keys[: len(listed)] == listed
    assert "candidate_graph.vertex_count = 40" in lines
    assert "full_graph.peak_count = 6" in lines
    assert not any("full_graph.component_count" in ln for ln in lines)  # a gap
    assert "No previous run" in mq.render_report(rows, None)


def test_unlisted_keys_follow_the_manuscript_order() -> None:
    ordered = mq.manuscript_order(
        ["zzz.extra", "bootstrap.seed", "thresholds.median_gap.value"]
    )
    assert ordered == ["thresholds.median_gap.value", "bootstrap.seed", "zzz.extra"]


# ----------------------------------------------- the stage, end to end


def _stage_settings() -> dict[str, Any]:
    return {
        "analysis": {
            "dataset": {"name": "Papkou2023"},
            "clamped": CONFIG["clamped"],
            "codon_entropy": CONFIG["codon_entropy"],
        },
    }


def test_stage_exports_tables_and_a_report_that_flags_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    write_upstream(output)
    monkeypatch.setattr(build, "OUTPUT", output)
    monkeypatch.setattr(build, "load_settings", _stage_settings)
    real = mq.load_sources
    monkeypatch.setattr(
        mq,
        "load_sources",
        lambda directory: real(directory, readers=fake_readers(directory)),
    )
    destination = output / "tables" / mq.TABLE_SUBDIRECTORY

    paths = build.build_manuscript_quantities()
    assert {p.name for p in paths} == {
        "manuscript_quantities.csv",
        "manuscript_quantities.parquet",
        "manuscript_quantities.schema.json",
        "manuscript_quantities_report.md",
        "entropy_by_position.parquet",
    }
    assert all(p.parent == destination and p.is_file() for p in paths)
    frame = pl.read_parquet(destination / "manuscript_quantities.parquet")
    assert tuple(frame.columns) == mq.COLUMNS
    assert frame.schema["value"] == pl.String
    first = (destination / "manuscript_quantities_report.md").read_text(
        encoding="utf-8"
    )
    assert "No previous run" in first
    schema = json.loads((destination / "manuscript_quantities.schema.json").read_text())
    assert schema["rounding"].startswith("ROUND_HALF_UP")

    # A reused run neither rewrites the report nor loses its flags.
    build.build_manuscript_quantities()
    assert (destination / "manuscript_quantities_report.md").read_text(
        encoding="utf-8"
    ) == first

    # Change one upstream value and add a quantity's removal: the report flags it.
    write_upstream(output, published_peaks=13, clamped_fingerprint="clamped-fp-2")
    build.build_manuscript_quantities()
    report = (destination / "manuscript_quantities_report.md").read_text(
        encoding="utf-8"
    )
    assert "changed (was 12)" in report
    assert "published_graph.peak_count" in report
    flagged = [ln for ln in report.splitlines() if "changed (was" in ln]
    # The peak count moved (cross-checked by both sources) and so did the two
    # annotation denominators that must equal it, and the largest hill (the
    # synthetic Published tree grows a peak with it).
    assert len(flagged) == 4


def test_the_stage_fingerprint_keys_on_upstream_presence_and_identity(
    tmp_path: Path,
) -> None:
    from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint

    def fingerprint(**kwargs: Any) -> str:
        output = tmp_path / str(len(list(tmp_path.iterdir())))
        write_upstream(output, **kwargs)
        sources = load(output, _tree())
        stage, _ = build._manuscript_quantities_stage(_stage_settings(), sources)
        return stage_fingerprint(stage)

    assert fingerprint() == fingerprint()
    assert fingerprint() != fingerprint(skip=("bootstrap",))
    assert fingerprint() != fingerprint(band=True)


def test_command_and_recipes_are_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "justfile").read_text(encoding="utf-8")
    header, recipe = text.split("\nmanuscript-quantities:", 1)[1].split("\n", 1)
    recipe = recipe.split("\n\n", 1)[0]
    # The band stage is a declared upstream of the registry.
    assert "leading-saddle-band" in header.split()
    # The registry reads the annotation tables, so they are built first.
    assert "biological-annotation-tables" in header.split()
    assert "build manuscript-quantities" in recipe
    annotation = text.split("\nbiological-annotation-tables:", 1)[1].split("\n\n", 1)[0]
    assert "comparison-analysis" in annotation.splitlines()[0].split()
    assert "build biological-annotation-tables" in annotation
    tables = text.split("\ntables:", 1)[1].split("\n\n", 1)[0]
    assert "leading-saddle-band" in tables.splitlines()[0].split()
    assert "biological-annotation-tables" in tables.splitlines()[0].split()
    assert tables.index("build manuscript-quantities") < tables.index("build tables")
    # Table 1 (the three-column graph summary) is retired for the seven-column table.
    assert "graph-summary" not in text
    assert "build landscape-summary-table" in tables


# ------------------------------------------- stale upstreams and consistency


def _update(path: Path, **changes: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(changes)
    path.write_text(json.dumps(value), encoding="utf-8")


def _clamped_dir(output: Path) -> Path:
    return output / "analysis" / "clamped_comparison"


@pytest.mark.parametrize(
    ("reader", "gap_key"),
    [
        ("clamped", "published_graph.vertex_count"),
        ("component", "thresholds.join_tree_trailing_saddle.value"),
        ("peak", None),
        ("bootstrap", "bootstrap.draws"),
        ("branches", "branches.terminal_count"),
        ("band", "leading_saddle_band.median_inside_band"),
    ],
)
def test_a_cached_loader_not_found_is_a_gap_and_any_other_failure_raises(
    tmp_path: Path, reader: str, gap_key: str | None
) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)

    def not_found() -> None:
        raise FileNotFoundError("no current checkpoint")

    readers[reader] = not_found
    sources = mq.load_sources(tmp_path, readers=readers)
    assert sources[reader] is None
    if gap_key is not None:
        rows = {r["key"]: r for r in mq.build_rows(sources, CONFIG, git_commit=None)}
        reason = rows[gap_key]["gap_reason"]
        assert reason is not None
        # The fixture's files exist (stale) except the band's, which write_upstream
        # leaves out (missing).
        kind = "missing" if reader == "band" else "stale"
        assert reason.startswith(f"{kind}: ")
        assert "no current checkpoint; rerun just " in reason
        assert rows[gap_key]["value"] is None
    for error in (
        RuntimeError("left mid-publication"),
        ValueError("incompatible contract"),
    ):

        def refuse(error: Exception = error) -> None:
            raise error

        readers[reader] = refuse
        with pytest.raises(type(error), match=str(error)):
            mq.load_sources(tmp_path, readers=readers)


def test_the_default_readers_are_the_cached_loaders() -> None:
    from dubanevics2026topological.analysis.branch_decomposition import (
        load_cached_branch_decomposition,
    )
    from dubanevics2026topological.analysis.clamped_cache import (
        load_cached_clamped_comparison,
    )

    readers = mq.cached_readers()
    # The Full peak count is stated, so the reader must load the Full analysis.
    assert readers["clamped"].func is load_cached_clamped_comparison
    assert readers["clamped"].keywords == {"include_full": True}
    assert readers["branches"] is load_cached_branch_decomposition
    assert set(readers) == {
        "landscape_summary",
        "band_view",
        "clamped",
        "component",
        "peak",
        "bootstrap",
        "branches",
        "band",
        "functional_classification",
        "band_directory",
        "functional_directory",
        "functional_threshold_scan",
        "annotation_fingerprint",
        "full_fitness_height",
    }
    # The band reader is the build module's cached loader (it refuses stale or
    # mid-publication output and raises FileNotFoundError when none is current).
    assert readers["band"].__name__ == "load_cached_band"


def test_keyed_checkpoint_folders_are_read_not_the_step_root(tmp_path: Path) -> None:
    """produce_or_load publishes under a keyed folder; no provenance.json at the root."""
    write_upstream(tmp_path)
    write_band(tmp_path)
    write_functional(tmp_path)
    readers = dict(fake_readers(tmp_path))
    for step, name, key in (
        ("leading_saddle_band", "band", "band_directory"),
        (
            "functional_classification",
            "functional_classification",
            "functional_directory",
        ),
    ):
        root = tmp_path / "checkpoints" / step
        keyed = root / "key=abcd1234"
        keyed.mkdir()
        for file in ("summary.json", "provenance.json"):
            (root / file).rename(keyed / file)
        assert not (root / "provenance.json").exists()
        readers[key] = lambda keyed=keyed: keyed
        readers[name] = lambda keyed=keyed: SimpleNamespace(
            summary=_read(keyed / "summary.json")
        )
    sources = mq.load_sources(tmp_path, readers=readers)
    for name in ("band", "functional_classification"):
        assert sources[name] is not None
        assert all("key=abcd1234" in path for path in sources[name].files)
        assert all(len(digest) == 64 for digest in sources[name].files.values())


def test_a_clamped_result_without_its_full_analysis_is_refused_not_none(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    readers = dict(fake_readers(tmp_path))
    inner = readers["clamped"]

    def without_full() -> Any:
        result = inner()
        result["landscapes"]["full"] = None
        return result

    readers["clamped"] = without_full
    with pytest.raises(ValueError, match="without its Full analysis"):
        mq.load_clamped(tmp_path, readers)


def test_a_band_without_a_summary_is_absent_even_with_a_pending_marker(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    assert load(tmp_path)["band"] is None


def _raises_naming(tmp_path: Path, first: str, second: str, rerun: str) -> None:
    with pytest.raises(ValueError) as caught:
        rows_of(tmp_path)
    message = str(caught.value)
    assert first in message and second in message and f"`{rerun}`" in message


def test_a_published_threshold_that_differs_from_the_comparison_raises(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    _update(
        _clamped_dir(tmp_path) / "provenance.json",
        thresholds={"study_clamped": -0.4, "candidate_clamped": 0.7},
    )
    _raises_naming(
        tmp_path,
        "config/analysis.toml published_functional_threshold",
        "provenance thresholds.study_clamped",
        "just comparison-analysis",
    )


def test_the_graph_summary_thresholds_must_match_the_config_and_the_candidate_record(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    path = _clamped_dir(tmp_path) / "graph_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["metrics"]["Published"]["Functional threshold"] = -0.1
    path.write_text(json.dumps(summary), encoding="utf-8")
    _raises_naming(
        tmp_path,
        "published_functional_threshold",
        "graph summary Published functional threshold",
        "just comparison-analysis",
    )
    write_upstream(tmp_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["metrics"]["Candidate"]["Functional threshold"] = 0.65
    path.write_text(json.dumps(summary), encoding="utf-8")
    _raises_naming(
        tmp_path,
        "rounded_threshold",
        "graph summary Candidate functional threshold",
        "just comparison-analysis",
    )


def test_a_candidate_threshold_that_differs_from_the_provenance_raises(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    _update(
        _clamped_dir(tmp_path) / "provenance.json",
        thresholds={"study_clamped": -0.507774, "candidate_clamped": 0.65},
    )
    _raises_naming(
        tmp_path,
        "rounded_threshold",
        "provenance thresholds.candidate_clamped",
        "just comparison-analysis",
    )


def test_the_codon_entropy_threshold_must_match_the_candidate_threshold(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    _update(tmp_path / "tables" / mq.CODON_METADATA, functional_threshold=0.65)
    _raises_naming(
        tmp_path,
        "rounded_threshold",
        "codon-entropy metadata functional_threshold",
        "just codon-entropy-analysis",
    )


def test_the_nominal_bootstrap_peak_count_must_equal_the_candidate_peak_count(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    path = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["band_simplification"]["peaks_before"] = 49
    path.write_text(json.dumps(summary), encoding="utf-8")
    _raises_naming(
        tmp_path,
        "Candidate graph peak_count",
        "bootstrap nominal peaks_before",
        "just bootstrap-smoothed-tree",
    )


@pytest.mark.parametrize(
    ("field", "value", "label"),
    [
        ("published_functional_threshold_fitness", -0.5, "published_functional"),
        ("candidate_functional_threshold_fitness", 0.65, "rounded_threshold"),
        ("global_minimum_fitness", -0.8, "global_minimum"),
        ("median_fitness", -0.2, "median"),
        ("join_trailing_saddle_fitness", -0.25, "join_tree_trailing_saddle"),
        ("candidate_threshold_rule", "other", "rule"),
        ("candidate_threshold_decimals", 5, "decimals"),
    ],
)
def test_the_band_landmarks_must_equal_the_registry_thresholds(
    tmp_path: Path, field: str, value: Any, label: str
) -> None:
    write_upstream(tmp_path, band=True)
    rows_of(tmp_path)  # consistent as written
    summary = band_summary()
    summary[field] = value
    write_band(tmp_path, summary)
    _raises_naming(
        tmp_path,
        label,
        f"leading-saddle-band {'summary ' if 'published' not in field else ''}{field}",
        "just leading-saddle-band",
    )


def test_the_ordering_row_keeps_b_as_the_join_tree_trailing_saddle(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path, band=True)
    rows = rows_of(tmp_path)
    # b = join tree trailing saddle (-0.3), not the band's split saddle (-0.35).
    assert rows["thresholds.join_tree_trailing_saddle.value"]["value"] == "-0.3"
    assert rows["leading_saddle_band.join_trailing_saddle.fitness"]["value"] == "-0.3"


def test_an_incomplete_clamped_diagnostics_file_raises(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    path = _clamped_dir(tmp_path) / "diagnostics.json"
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    del diagnostics["study_clamped"]
    path.write_text(json.dumps(diagnostics), encoding="utf-8")
    with pytest.raises(KeyError, match="study_clamped"):
        rows_of(tmp_path)


def test_no_ordinary_pair_makes_the_trailing_saddle_undefined_not_absent(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    _write(
        tmp_path / "analysis" / "component_diagnostics.json",
        {"join_tree_trailing_saddle": None, "global_minimum": -0.9},
    )
    row = rows_of(tmp_path)[
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
    ]
    assert row["value"] is None
    assert "trailing saddle b is undefined" in row["gap_reason"]
    assert "absent" not in row["gap_reason"]
    write_upstream(tmp_path / "other", skip=("component",))
    absent = rows_of(tmp_path / "other")[
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
    ]
    assert mq.is_stage_not_run(absent["gap_reason"])


def test_the_graph_summary_and_band_schemas_are_checked(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=True)
    _update(_clamped_dir(tmp_path) / "graph_summary.json", schema=3)
    with pytest.raises(ValueError, match="graph summary has schema 3"):
        rows_of(tmp_path)
    write_upstream(tmp_path, band=True)
    summary = band_summary()
    summary["schema"] = 2
    write_band(tmp_path, summary)
    with pytest.raises(ValueError, match="band summary has schema 2"):
        rows_of(tmp_path)


# ------------------------------------------------------- upstream identity


def test_a_changed_input_file_changes_the_stage_fingerprint_with_the_sidecar_fixed(
    tmp_path: Path,
) -> None:
    from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint

    write_upstream(tmp_path)

    def fingerprint() -> str:
        stage, _ = build._manuscript_quantities_stage(_stage_settings(), load(tmp_path))
        return stage_fingerprint(stage)

    before = fingerprint()
    assert fingerprint() == before
    # The annotation sidecar fingerprint and every parsed count stay the same; only
    # the raw bytes of the counts file change.
    path = tmp_path / "tables" / mq.ANNOTATION_COUNTS
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert fingerprint() != before
    after_annotations = fingerprint()
    # A codon table replaced with identical rows but a different file.
    path = tmp_path / "tables" / mq.CODON_CLAIMS
    frame = pl.read_parquet(path)
    frame.write_parquet(path, compression="lz4")
    assert fingerprint() != after_annotations
    # With the sidecar missing as well, the bytes still count.
    (tmp_path / "tables" / mq.CODON_SIDECAR).unlink()
    without_sidecar = fingerprint()
    path = tmp_path / "tables" / mq.CODON_SITE_ENTROPY
    pl.read_parquet(path).write_parquet(path, compression="uncompressed")
    assert fingerprint() != without_sidecar


def test_the_band_and_cached_upstreams_carry_file_hashes(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=True)
    identities = mq.source_identities(load(tmp_path))
    for name in ("band", "bootstrap", "branches", "component", "peak", "codon"):
        assert identities[name]["files"], name
    assert all(len(digest) == 64 for digest in identities["band"]["files"].values())


# ------------------------------------------------- interrupted-export report


def test_an_interrupted_export_never_leaves_a_stale_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    write_upstream(output)
    monkeypatch.setattr(build, "OUTPUT", output)
    monkeypatch.setattr(build, "load_settings", _stage_settings)
    real = mq.load_sources
    monkeypatch.setattr(
        mq,
        "load_sources",
        lambda directory: real(directory, readers=fake_readers(directory)),
    )
    destination = output / "tables" / mq.TABLE_SUBDIRECTORY
    report = destination / mq.REPORT_NAME
    record = output / "tables" / ".manuscript-quantities-provenance.json"
    build.build_manuscript_quantities()
    assert record.is_file()

    # An interrupted export: the record is gone and the report is out of date.
    record.unlink()
    report.write_text("STALE REPORT", encoding="utf-8")
    build.build_manuscript_quantities()
    assert "STALE REPORT" not in report.read_text(encoding="utf-8")
    assert record.is_file()

    # A crash while rendering leaves no stale report either.
    record.unlink()
    monkeypatch.setattr(
        mq, "render_report", lambda *args: (_ for _ in ()).throw(OSError("killed"))
    )
    with pytest.raises(OSError, match="killed"):
        build.build_manuscript_quantities()
    assert not report.exists()
    assert not record.exists()


def test_the_report_is_written_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "report.md"
    build._write_text_atomic(target, "first")
    assert target.read_text(encoding="utf-8") == "first"

    def fail(*_args: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(build.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        build._write_text_atomic(target, "second")
    assert target.read_text(encoding="utf-8") == "first"
    assert [p.name for p in tmp_path.iterdir()] == ["report.md"]


# ------------------------------------------------------------------ nits


@pytest.mark.parametrize(
    "bad", ["aa.1b", "aa._bb", "aa.bb._cc", "1a.bb", "aa.bb.9", "aa.bb2.9"]
)
def test_every_key_segment_must_start_with_a_letter(bad: str) -> None:
    with pytest.raises(ValueError, match="not dotted"):
        mq.validate_rows([{"key": bad}])
    mq.validate_rows([{"key": "aa.b2.cd_ef"}])


@pytest.mark.parametrize(
    "bad",
    [
        "x.value",
        "thresholds.g_b.value",
        "thresholds.ordering_g_b_median_t.value",
        "aa.b",
        "aa.bb_c.dd",
        "aa.c_bb.dd",
        "aa.bb.dd_e",
    ],
)
def test_no_key_segment_or_word_may_be_a_single_letter(bad: str) -> None:
    with pytest.raises(ValueError, match="single letter"):
        mq.validate_rows([{"key": bad}])


@pytest.mark.parametrize(
    "good", ["entropy.core.all.aa_site_1", "bootstrap.persistence_threshold.value"]
)
def test_numeric_suffix_words_stay_legal(good: str) -> None:
    mq.validate_rows([{"key": good}])


@pytest.mark.parametrize(
    ("value", "text", "shown"),
    [
        (1e-07, "1e-07", "0.000000"),
        (1.5e-05, "1.5e-05", "0.000015"),
        (-4e-07, "-4e-07", "0.000000"),
        (2.5e-06, "2.5e-06", "0.000003"),
    ],
)
def test_exponent_form_floats_display_by_exact_decimal_rounding(
    value: float, text: str, shown: str
) -> None:
    assert mq.value_text(value) == text
    assert mq.display_spec(text, "fitness") == (6, shown)
    assert mq.display_spec(mq.value_text(value * 1e6), "percent")[0] == 1


def test_a_count_must_arrive_as_an_integer_even_when_the_float_is_integral() -> None:
    assert mq.display_spec(mq.value_text(12), "count") == (0, "12")
    assert mq.display_spec(mq.value_text(12345), "count") == (0, "12,345")
    for bad in (12.0, 12.5):
        with pytest.raises(ValueError, match="must be an integer"):
            mq.display_spec(mq.value_text(bad), "count")
    with pytest.raises(ValueError, match="must be an integer"):
        mq.display_spec(mq.value_text(20261019.0), "seed")
    rows = mq._Rows(None)
    with pytest.raises(ValueError, match="must be an integer"):
        rows.put(
            "x.count",
            12.0,
            "count",
            "p",
            "r",
            up=mq.Upstream("s", "f"),
            stage="s",
        )


# ------------------------------------------- band values and the ordering row


def test_g_m_h_t_and_b_come_from_the_band_summary_when_it_ran(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=True)
    rows = rows_of(tmp_path)
    for key in (
        "thresholds.candidate_functional.value",
        "thresholds.candidate_functional.unrounded_value",
        "thresholds.full_median.value",
        "thresholds.full_global_minimum.value",
        "thresholds.median_gap.value",
        "thresholds.join_tree_trailing_saddle.value",
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value",
    ):
        assert rows[key]["source_stage"] == "leading-saddle-band", key
        assert rows[key]["source_fingerprint"] == "band-fp", key
    assert rows["thresholds.median_gap.value"]["value"] == "0.8"
    assert (
        rows["thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"][
            "value"
        ]
        == "true"
    )
    # Without the band the same rows come from the clamped and component outputs.
    write_upstream(tmp_path / "plain")
    plain = rows_of(tmp_path / "plain")
    assert plain["thresholds.full_median.value"]["source_stage"] == "clamped-comparison"
    assert (
        plain["thresholds.join_tree_trailing_saddle.value"]["source_stage"]
        == "component-analysis"
    )


@pytest.mark.parametrize(
    ("field", "value", "clamped_label"),
    [
        ("candidate_unrounded_threshold", 0.71, "unrounded_threshold"),
        ("median_gap", 0.81, "median_gap"),
        ("join_trailing_saddle_fitness", -0.31, "component analysis"),
    ],
)
def test_band_and_other_upstreams_must_agree_exactly(
    tmp_path: Path, field: str, value: float, clamped_label: str
) -> None:
    write_upstream(tmp_path, band=True)
    rows_of(tmp_path)  # consistent as written
    summary = band_summary()
    summary[field] = value
    write_band(tmp_path, summary)
    with pytest.raises(ValueError, match="just leading-saddle-band") as caught:
        rows_of(tmp_path)
    assert clamped_label in str(caught.value)


def test_a_false_ordering_is_recorded_and_visible_in_the_report(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path, band=True)
    summary = band_summary()
    # b_join above the median: g < b < m fails.
    summary["join_trailing_saddle_fitness"] = 0.0
    write_band(tmp_path, summary)
    _update(
        tmp_path / "analysis" / "component_diagnostics.json",
        join_tree_trailing_saddle=0.0,
    )
    rows = rows_of(tmp_path)
    row = rows[
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
    ]
    assert row["value"] == "false"
    assert row["display_value"] == "false"
    assert row["gap_reason"] is None
    report = mq.render_report(list(rows.values()), None)
    line = next(
        ln
        for ln in report.splitlines()
        if "ordering_minimum_join_trailing_saddle_median_candidate" in ln
    )
    assert "| false |" in line
    assert (
        "ordering_minimum_join_trailing_saddle_median_candidate.value = false" in report
    )


def test_the_band_stage_layout_is_the_real_summary_of_the_stage() -> None:
    from dubanevics2026topological.analysis.leading_saddle_band import (
        LANDMARK_KEYS,
        STAGE_NAME,
        SUMMARY_SCHEMA,
    )

    assert mq.BAND == STAGE_NAME
    assert mq.BAND_SCHEMA == SUMMARY_SCHEMA
    assert mq.BAND_LANDMARKS == LANDMARK_KEYS
    summary = band_summary()
    for landmark in LANDMARK_KEYS:
        assert f"{landmark}_fitness" in summary
        assert f"{landmark}_sequence" in summary


def test_claim_column_and_registry_agree_at_exactly_the_minimum(tmp_path: Path) -> None:
    from dubanevics2026topological.analysis.codon_entropy_populations import (
        _claim_summary,
    )

    def population(count: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "dna_sequence": [f"S{index}" for index in range(count)],
                "aa2": ["E"] * count,
                "is_functional_core": [True] * count,
            }
        )

    minimum = 0.9
    core = population(10)
    assignments = pl.DataFrame(
        {
            "dna_sequence": [f"S{index}" for index in range(10)],
            "ridge_id": ["kem"] * 9 + ["none"],
        }
    )
    populations = {"core": core, "kem": population(9), "tcl": population(0)}
    claims = _claim_summary(
        populations, assignments, 0.7, {"two_functional_ridges_share_minimum": minimum}
    )
    share = claims.filter(pl.col("claim_id") == "functional_sequences_in_two_ridges")
    assert "claim_exceeds_threshold" not in claims.columns
    assert share["fraction"][0] == minimum
    assert share["claim_meets_minimum"][0] is True
    assert share["claim_exceeds_baseline"][0] is None
    # The registry's boolean, read from the same claim table, agrees.
    write_upstream(tmp_path)
    metadata = tmp_path / "tables" / mq.CODON_METADATA
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["counts"].update(kem_functional=9, tcl_functional=0)
    metadata.write_text(json.dumps(document), encoding="utf-8")
    share.select("claim_id", "numerator", "denominator", "fraction").write_parquet(
        tmp_path / "tables" / mq.CODON_CLAIMS
    )
    registry = rows_of(tmp_path)["ridges.two_functional_share.meets_minimum"]
    assert registry["value"] == "true"
    assert (registry["value"] == "true") is share["claim_meets_minimum"][0]


def test_enrichment_row_equal_to_its_baseline_does_not_exceed_it() -> None:
    from dubanevics2026topological.analysis.codon_entropy_populations import (
        _claim_summary,
    )

    def population(states: list[str]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "dna_sequence": [f"S{index}" for index in range(len(states))],
                "aa2": states,
                "is_functional_core": [True] * len(states),
            }
        )

    # Baseline E frequency 0.5; the KEM ridge has exactly 0.5, the TCL ridge 1.0.
    core = population(["E", "D", "E", "D"])
    kem = population(["E", "D"])
    tcl = population(["E", "E"])
    assignments = pl.DataFrame({"dna_sequence": ["S0"], "ridge_id": ["kem"]})
    claims = _claim_summary(
        {"core": core, "kem": kem, "tcl": tcl}, assignments, 0.7, {}
    )

    def row(claim_id: str) -> dict:
        return claims.filter(pl.col("claim_id") == claim_id).row(0, named=True)

    equal = row("kem_aa_site_2_E_enrichment")
    assert equal["fraction"] == equal["claim_threshold"] == 0.5
    assert equal["claim_exceeds_baseline"] is False
    assert equal["claim_meets_minimum"] is None
    assert row("tcl_aa_site_2_E_enrichment")["claim_exceeds_baseline"] is True


# ------------------------------------------------ stale annotation counts (1)


def _counts_path(output: Path) -> Path:
    return output / "tables" / mq.ANNOTATION_COUNTS


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"landscape": "candidate_clamped"}, "landscape"),
        ({"peak_denominator": 99}, "peak_denominator"),
        ({"peak_count": 99}, "peak_count"),
    ],
)
def test_annotation_counts_of_another_graph_raise(
    tmp_path: Path, changes: dict[str, Any], expected: str
) -> None:
    write_upstream(tmp_path)
    _update(_counts_path(tmp_path), **changes)
    with pytest.raises(ValueError, match=expected) as caught:
        rows_of(tmp_path)
    message = str(caught.value)
    assert "just biological-annotation-tables" in message
    if expected != "landscape":
        assert "clamped comparison diagnostics study_clamped peak_count" in message
        assert "biological annotation counts" in message


def test_annotation_counts_must_record_their_landscape(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    counts = json.loads(_counts_path(tmp_path).read_text(encoding="utf-8"))
    del counts["landscape"]
    _write(_counts_path(tmp_path), counts)
    with pytest.raises(KeyError, match="counts.landscape"):
        rows_of(tmp_path)


def test_a_stale_threshold_change_is_caught_through_the_peak_count(
    tmp_path: Path,
) -> None:
    """The scenario: the threshold changes, the comparison is rebuilt, the tables
    of the old graph remain."""
    write_upstream(tmp_path, published_peaks=12)
    # Rebuild the clamped record and peak analysis only (13 peaks now).
    fresh = tmp_path / "fresh"
    write_upstream(fresh, published_peaks=13)
    for name in ("analysis",):
        for path in (fresh / name).rglob("*"):
            if path.is_file():
                target = tmp_path / path.relative_to(fresh)
                target.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="peak_denominator"):
        rows_of(tmp_path)


def test_the_annotation_export_record_must_carry_the_current_fingerprint(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)
    readers["annotation_fingerprint"] = lambda: "annotation-fp-now"
    with pytest.raises(ValueError, match="stale") as caught:
        mq.load_sources(tmp_path, readers=readers)
    message = str(caught.value)
    assert "'annotation-fp'" in message and "'annotation-fp-now'" in message
    assert ".biological-annotation-provenance.json" in message
    assert "just biological-annotation-tables" in message
    readers["annotation_fingerprint"] = lambda: "annotation-fp"
    assert mq.load_sources(tmp_path, readers=readers)["annotations"] is not None


def test_a_missing_annotation_export_record_raises(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    (tmp_path / "tables" / mq.ANNOTATION_SIDECAR).unlink()
    with pytest.raises(RuntimeError, match="no export record"):
        load(tmp_path)


def test_an_uncomputable_annotation_fingerprint_is_a_stale_gap(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)

    def cannot() -> str:
        raise FileNotFoundError("comparison provenance is missing")

    readers["annotation_fingerprint"] = cannot
    sources = mq.load_sources(tmp_path, readers=readers)
    assert sources["annotations"] is None
    rows = {r["key"]: r for r in mq.build_rows(sources, CONFIG, git_commit=None)}
    reason = rows["annotations.published.stop_codon_peak_count"]["gap_reason"]
    assert reason.startswith("stale: ")
    assert "just biological-annotation-tables" in reason


def test_the_annotation_stage_fingerprint_is_computable_without_running_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    write_upstream(output)
    monkeypatch.setattr(build, "OUTPUT", output)
    first = build.current_biological_annotation_fingerprint()
    assert first == build.current_biological_annotation_fingerprint()
    _update(_clamped_dir(output) / "provenance.json", fingerprint="another")
    assert build.current_biological_annotation_fingerprint() != first


def test_the_stage_code_includes_the_graph_summary_module() -> None:
    sources = {name: None for name in ("clamped", "band")}
    stage, _ = build._manuscript_quantities_stage(_stage_settings(), sources)
    assert any(Path(path).name == "graph_summary.py" for path in stage.code)


# ------------------------------------------------------- crash paths (7)


def test_an_absorbed_count_with_a_null_input_is_a_gap_row(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    boot = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(boot.read_text(encoding="utf-8"))
    summary["band_simplification"]["peaks_after_core_rule"] = None
    _write(boot, summary)
    rows = rows_of(tmp_path)
    for state in ("after_core_rule", "after_band_rule"):
        row = rows[f"bootstrap.candidate.{state}.absorbed_count"]
        assert row["value"] is None
        assert row["gap_reason"].startswith("missing: ")
        assert "peak count" in row["gap_reason"]
    assert rows["bootstrap.candidate.after_core_rule.peak_count"]["value"] is None
    assert rows["bootstrap.candidate.nominal.peak_count"]["value"] == "30"


@pytest.mark.parametrize("stale_core_field", [False, True])
def test_direct_band_summary_has_no_core_stage(
    tmp_path: Path, stale_core_field
) -> None:
    from dubanevics2026topological.simplification.identities import band_identity

    write_upstream(tmp_path)
    boot = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(boot.read_text(encoding="utf-8"))
    summary["fingerprint"]["schema"] = 7
    fields, digest = band_identity(draw_scientific_id_sha256="a" * 64, bandq=0.95)
    summary["scientific_identities"] = {
        "band": {
            "product_type": "tree",
            "scientific_identity": fields,
            "scientific_id_sha256": digest,
        }
    }
    if not stale_core_field:
        summary["band_simplification"].pop("peaks_after_core_rule")
    _write(boot, summary)
    rows = rows_of(tmp_path)
    assert "bootstrap.candidate.after_core_rule.peak_count" not in rows
    assert "bootstrap.candidate.after_core_rule.absorbed_count" not in rows
    assert "bootstrap.candidate.after_band_rule.absorbed_count" not in rows
    assert rows["bootstrap.candidate.total_absorbed_count"]["value"] == str(
        summary["band_simplification"]["peaks_before"]
        - summary["band_simplification"]["peaks_after"]
    )


def test_legacy_summary_missing_core_count_remains_a_gap(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    boot = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(boot.read_text(encoding="utf-8"))
    summary["band_simplification"].pop("peaks_after_core_rule")
    _write(boot, summary)
    rows = rows_of(tmp_path)
    for state in ("after_core_rule", "after_band_rule"):
        row = rows[f"bootstrap.candidate.{state}.absorbed_count"]
        assert row["value"] is None
        assert row["gap_reason"].startswith("missing: ")


@pytest.mark.parametrize(
    ("schema", "method"), [(6, "direct_band"), (7, "band_after_core"), (7, "unknown")]
)
def test_bootstrap_summary_refuses_conflicting_method(tmp_path: Path, schema, method):
    write_upstream(tmp_path)
    boot = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(boot.read_text(encoding="utf-8"))
    summary["fingerprint"]["schema"] = schema
    summary["scientific_identities"] = {
        "band": {"scientific_identity": {"simplification_method": method}}
    }
    _write(boot, summary)
    with pytest.raises(ValueError, match="conflicts with simplification method"):
        rows_of(tmp_path)


def test_bootstrap_summary_refuses_unknown_schema(tmp_path: Path):
    write_upstream(tmp_path)
    boot = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(boot.read_text(encoding="utf-8"))
    summary["fingerprint"]["schema"] = 8
    _write(boot, summary)
    with pytest.raises(ValueError, match="Unsupported bootstrap checkpoint schema"):
        rows_of(tmp_path)


def test_the_band_saddle_is_checked_against_the_component_without_the_comparison(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path, band=True, skip=("clamped", "peak"))
    rows = rows_of(tmp_path)
    saddle = rows["thresholds.join_tree_trailing_saddle.value"]
    assert saddle["gap_reason"] is None
    assert "not cross-checked" not in saddle["selection_rule"]
    _write(
        tmp_path / "analysis" / "component_diagnostics.json",
        {"join_tree_trailing_saddle": -0.25, "global_minimum": -0.9},
    )
    with pytest.raises(ValueError, match="join_tree_trailing_saddle") as caught:
        rows_of(tmp_path)
    assert "just leading-saddle-band" in str(caught.value)


def test_a_band_saddle_without_the_component_analysis_says_it_was_not_checked(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path, band=True, skip=("component",))
    rows = rows_of(tmp_path)
    saddle = rows["thresholds.join_tree_trailing_saddle.value"]
    assert saddle["value"] == "-0.3"
    assert saddle["gap_reason"] is None
    assert saddle["source_stage"] == "leading-saddle-band"
    assert "not cross-checked" in saddle["selection_rule"]
    ordering = rows[
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
    ]
    assert "not cross-checked" in ordering["selection_rule"]
    # With the component analysis the note is absent.
    write_upstream(tmp_path, band=True)
    assert (
        "not cross-checked"
        not in rows_of(tmp_path)["thresholds.join_tree_trailing_saddle.value"][
            "selection_rule"
        ]
    )


# ------------------------------------------------------- gap kinds (9)


def test_every_gap_reason_starts_with_a_kind(tmp_path: Path) -> None:
    write_upstream(tmp_path, skip=("bootstrap", "codon", "annotations"))
    readers = fake_readers(tmp_path)

    def not_current() -> None:
        raise FileNotFoundError("no current checkpoint")

    readers["branches"] = not_current
    sources = mq.load_sources(tmp_path, readers=readers)
    rows = mq.build_rows(sources, CONFIG, git_commit=None)
    gaps = [row for row in rows if row["gap_reason"] is not None]
    assert gaps
    assert all(row["gap_reason"].startswith(mq.GAP_KINDS) for row in gaps)
    by_key = {row["key"]: row["gap_reason"] for row in gaps}
    # An output that exists but is refused is stale; an absent one is missing.
    assert by_key["branches.terminal_count"].startswith("stale: ")
    assert by_key["bootstrap.draws"].startswith("missing: ")
    assert by_key["summits.kem.dna_sequence"].startswith("missing: ")
    assert by_key["annotations.published.stop_codon_peak_count"].startswith("missing: ")
    # The saved landscape summary product has not been read.
    assert by_key["full_graph.component_count"].startswith(
        "missing: stage landscape-summary-product not run"
    )


def test_a_joined_row_of_a_stale_and_a_missing_stage_is_stale(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path, skip=("component",))
    readers = fake_readers(tmp_path)

    def not_current() -> None:
        raise FileNotFoundError("no current checkpoint")

    readers["clamped"] = not_current
    sources = mq.load_sources(tmp_path, readers=readers)
    rows = {r["key"]: r for r in mq.build_rows(sources, CONFIG, git_commit=None)}
    reason = rows[
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value"
    ]["gap_reason"]
    assert reason.startswith("stale: ")
    assert "missing: " not in reason
    assert "clamped-comparison" in reason and "component-analysis" in reason


def test_an_empty_population_is_undefined_not_missing(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=True)
    rows = rows_of(tmp_path)
    reason = rows["leading_saddle_band.join_trunk.entropy_sum"]["gap_reason"]
    assert reason.startswith("undefined: ")


def test_the_report_header_counts_stale_and_missing_gap_rows(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path, skip=("bootstrap",))
    readers = fake_readers(tmp_path)

    def not_current() -> None:
        raise FileNotFoundError("no current checkpoint")

    readers["branches"] = not_current
    rows = mq.build_rows(
        mq.load_sources(tmp_path, readers=readers), CONFIG, git_commit=None
    )
    stale = sum(str(r["gap_reason"]).startswith("stale:") for r in rows)
    missing = sum(str(r["gap_reason"]).startswith("missing:") for r in rows)
    assert stale == 4 and missing > 0
    report = mq.render_report(rows, None)
    lines = report.splitlines()
    assert lines[0] == "# Manuscript quantities"
    header = lines[2]
    assert header.startswith("**Gap rows: ")
    assert f"{stale} `stale:`" in header and f"{missing} `missing:`" in header
    assert f"**WARNING: {stale} rows are `stale:`" in "\n".join(lines[:8])
    # A run with nothing stale shows the counts and no warning.
    clean = mq.render_report(
        [
            _row("aa.bb", "1"),
            _row("aa.cc", None, "missing: no stage computes it"),
        ],
        None,
    )
    assert "**Gap rows: 0 `stale:`" in clean and "1 `missing:`" in clean
    assert "WARNING" not in clean


# ------------------------------------------------ quantities behind figure labels

# Each new key with the value and display text the synthetic upstreams give it.
FIGURE_LABEL_ROWS = {
    "full_graph.peak_count": ("6", "6"),
    "full_graph.persistence.ordinary_h0_pair_count": ("4", "4"),
    "full_graph.persistence.valley_count": ("3", "3"),
    "published_graph.hill_sequence_count": ("98", "98"),
    "published_graph.largest_hill_sequence_count": ("87", "87"),
    "candidate_graph.hill_sequence_count": ("38", "38"),
    "candidate_graph.largest_hill_sequence_count": ("9", "9"),
    "peak_uncertainty.measured_sequence_count": ("1000", "1,000"),
    "bootstrap.candidate.total_absorbed_count": ("22", "22"),
    "bootstrap.candidate.after_band_rule.retained_sequence_count": ("40", "40"),
    "bootstrap.candidate.after_band_rule.saddles_removed_count": ("31", "31"),
    "bootstrap.band_quantile_percent": ("95", "95"),
    "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min": (
        "0.0002",
        "0.000200",
    ),
    "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max": (
        "0.1",
        "0.100000",
    ),
    "bootstrap.global_stability.relative_wasserstein_1_median": ("0.08345", "0.0835"),
    "bootstrap.global_stability.sanity_check.nominal_bottleneck_distance_median": (
        "0.0102",
        "0.010200",
    ),
    "bootstrap.global_stability.sanity_check.flat_bottleneck_distance_median": (
        "0.15215",
        "0.152150",
    ),
    "bootstrap.global_stability.sanity_check.flat_bottleneck_distance_fifth_percentile": (
        "0.0987",
        "0.098700",
    ),
}
SHARE_KEY = (
    "bootstrap.global_stability.sanity_check."
    "nominal_landscape_draws_below_flat_fifth_percentile_percent"
)
# The upstream whose absence turns each new key into a gap.
FIGURE_LABEL_SKIPS = {
    "clamped": [
        "full_graph.peak_count",
        "published_graph.hill_sequence_count",
        "candidate_graph.largest_hill_sequence_count",
        "annotations.candidate.stop_codon_peak_count",
        "bootstrap.candidate.after_band_rule.retained_sequence_count",
    ],
    "component": [
        "full_graph.persistence.ordinary_h0_pair_count",
        "full_graph.persistence.valley_count",
    ],
    "peak": ["peak_uncertainty.measured_sequence_count"],
    "bootstrap": [
        "bootstrap.candidate.total_absorbed_count",
        "bootstrap.candidate.after_band_rule.saddles_removed_count",
        "bootstrap.band_quantile_percent",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max",
        "bootstrap.global_stability.relative_wasserstein_1_median",
        SHARE_KEY,
    ],
}


def test_every_figure_label_quantity_has_a_value_and_a_display_value(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    for key, (value, shown) in FIGURE_LABEL_ROWS.items():
        assert rows[key]["value"] == value, key
        assert rows[key]["display_value"] == shown, key
        assert rows[key]["gap_reason"] is None, key
    share = rows[SHARE_KEY]
    assert share["unit"] == "percent" and share["display_value"] == "99.9"
    assert float(share["value"]) == pytest.approx(99.87)
    assert share["source_fingerprint"] == mq.fingerprint_text(
        bootstrap_summary()["fingerprint"]
    )


@pytest.mark.parametrize("skipped", sorted(FIGURE_LABEL_SKIPS))
def test_figure_label_quantities_are_named_gaps_when_their_upstream_is_absent(
    tmp_path: Path, skipped: str
) -> None:
    write_upstream(tmp_path, skip=(skipped,))
    rows = rows_of(tmp_path)
    for key in FIGURE_LABEL_SKIPS[skipped]:
        row = rows[key]
        assert row["value"] is None and row["display_value"] is None, key
        assert row["display_decimals"] is None and row["source_fingerprint"] is None
        assert row["gap_reason"], key


def test_figure_label_keys_are_listed_in_the_report_in_their_own_topics() -> None:
    new = {*FIGURE_LABEL_ROWS, SHARE_KEY}
    assert new <= set(mq.MANUSCRIPT_ORDER)
    assert mq.OTHER_TOPIC not in {mq.topic_of(key) for key in new}
    assert mq.topic_of("full_graph.persistence.valley_count") == (
        "Full graph persistence"
    )
    assert mq.topic_of("candidate_graph.hill_sequence_count") == "Hills"
    assert mq.topic_of(SHARE_KEY) == "Bootstrap global stability"
    assert mq.topic_of("peak_uncertainty.measured_sequence_count") == "Peak uncertainty"
    assert mq.topic_of("bootstrap.band_quantile_percent") == (
        "Bootstrapped persistence threshold"
    )
    assert mq.topic_of("bootstrap.candidate.total_absorbed_count") == "Peak counts"
    assert mq.topic_of("ridges.band_view.kem.functional_count") == (
        "Ridge populations and positional entropy"
    )


def test_the_full_peak_count_is_the_split_tree_maxima_and_distinct_from_pairs(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    full = rows["full_graph.peak_count"]
    assert "maxima of the Full graph's split tree" in full["selection_rule"]
    assert full["source_stage"] == mq.CLAMPED
    # The persistence figure's count comes from the component pairs, a separate row.
    pairs = rows["full_graph.persistence.ordinary_h0_pair_count"]
    assert pairs["source_stage"] == mq.COMPONENT
    assert "global maximum's pair" in pairs["selection_rule"]
    assert pairs["source_fingerprint"] == "component-fp"
    assert (
        "relative pairs"
        in rows["full_graph.persistence.valley_count"]["selection_rule"]
    )


def test_persistence_pair_counts_equal_what_the_figure_prints() -> None:
    from _saddle_band import JOIN_BELOW, SPLIT_BELOW

    from dubanevics2026topological.analysis.component_persistence import (
        descending_pair_counts,
    )
    from dubanevics2026topological.figures.component_persistence import (
        descending_component_pairs,
    )

    for pairs in (JOIN_BELOW, SPLIT_BELOW, _pairs([0.3], [])):
        scientific = descending_component_pairs(pairs)
        printed_peaks = scientific.filter(pl.col("pair_kind") != "relative").height
        printed_valleys = (
            scientific.filter(pl.col("pair_kind") == "relative").height + 1
        )
        assert descending_pair_counts(pairs) == {
            "ordinary_h0_pair_count": printed_peaks,
            "valley_count": printed_valleys,
        }
    assert descending_pair_counts(JOIN_BELOW) == {
        "ordinary_h0_pair_count": 4,
        "valley_count": 4,
    }
    with pytest.raises(ValueError, match="exactly one global pair"):
        descending_pair_counts(
            JOIN_BELOW.filter(pl.col("pair_kind") != "extended_positive")
        )


def test_hill_statistics_count_the_sequences_of_every_terminal_branch() -> None:
    from dubanevics2026topological.analysis.biological_annotations import (
        _build_biological_annotations as build_biological_annotations,
    )
    from dubanevics2026topological.analysis.biological_annotations import (
        hill_statistics,
    )

    analysis = _split_analysis(_tree(extra_peaks=1), {"AAGGAAATG": 4})
    pairs = build_biological_annotations(
        SimpleNamespace(trees={"split": analysis}), landscape="candidate_clamped"
    ).pairs
    assert sorted(pairs["branch_vertex_count"].to_list()) == [1, 1, 5]
    assert hill_statistics(pairs) == {
        "hill_sequence_count": 7,
        "largest_hill_sequence_count": 5,
    }
    with pytest.raises(ValueError, match="no hills"):
        hill_statistics(pairs.head(0))


def test_split_tree_peak_and_sequence_counts() -> None:
    from dubanevics2026topological.analysis.graph_summary import (
        split_tree_peak_count,
        split_tree_sequence_count,
    )

    analysis = _split_analysis(_tree(extra_peaks=2), {"AAGGAAATG": 3})
    assert split_tree_peak_count(analysis) == 4
    assert split_tree_sequence_count(analysis) == 6 + 3
    with pytest.raises(ValueError, match="event"):
        split_tree_peak_count(SimpleNamespace(event_table=pl.DataFrame({"x": [1]})))


def test_the_band_quantile_percent_is_exact_and_shown_as_stored(tmp_path: Path) -> None:
    for quantile, text in ((0.95, "95"), (0.975, "97.5"), (0.29, "29"), (1.0, "100")):
        assert mq.quantile_percent(quantile) == text
    write_upstream(tmp_path)
    row = rows_of(tmp_path)["bootstrap.band_quantile_percent"]
    assert row["unit"] == "percent" and row["display_decimals"] is None
    assert row["settings"] == "bootstrap.band_quantile"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("c", 0.02),
        ("two_c", 0.05),
        ("nominal_point_count", 49),
        ("significant_point_count", 9),
        ("band_quantile", 0.9),
    ],
)
def test_the_bootstrap_summary_blocks_must_agree(
    tmp_path: Path, name: str, value: float
) -> None:
    write_upstream(tmp_path)
    path = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["diagnostics"]["global_verdict"][name] = value
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=f"global_verdict {name}") as error:
        rows_of(tmp_path)
    assert "just bootstrap-smoothed-tree" in str(error.value)


def test_the_summary_radius_must_equal_the_draw_table_at_the_band_quantile(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    path = tmp_path / "analysis" / "bootstrap_smoothed_tree" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["band_simplification"]["c"] = 0.02
    summary["diagnostics"]["global_verdict"]["c"] = 0.02
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="band_width of the draw table"):
        rows_of(tmp_path)


def test_the_candidate_tree_must_represent_every_candidate_vertex(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)
    reader = readers["clamped"]

    def oversized() -> dict[str, Any]:
        result = reader()
        result["landscapes"]["candidate_clamped"] = _landscape(_tree(), 45)
        return result

    readers["clamped"] = oversized
    sources = mq.load_sources(tmp_path, readers=readers)
    with pytest.raises(
        ValueError, match="Candidate split tree sequence count"
    ) as error:
        mq.build_rows(sources, CONFIG, git_commit=None)
    assert "just comparison-analysis" in str(error.value)


def test_band_view_ridge_counts_are_the_band_views_not_the_selected_leaf_views(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    for ridge, expected in (("kem", "8"), ("tcl", "4")):
        row = rows[f"ridges.band_view.{ridge}.functional_count"]
        assert row["value"] == expected and row["display_value"] == expected
        assert row["gap_reason"] is None
        assert row["source_stage"] == mq.BAND_VIEW
        assert row["source_fingerprint"] == "clamped-fp+" + mq.fingerprint_text(
            bootstrap_summary()["fingerprint"]
        )
        assert "band-simplified" in row["selection_rule"]
        assert row["settings"] == "thresholds.candidate_functional.value"
    # The stage's own ridge rows are a different population and stay as they were.
    assert rows["ridges.kem.functional_count"]["value"] == "9"


def test_band_view_ridge_counts_are_gaps_without_their_inputs(tmp_path: Path) -> None:
    for skip in ("clamped", "bootstrap"):
        write_upstream(tmp_path / skip, skip=(skip,))
        rows = rows_of(tmp_path / skip)
        for ridge in ("kem", "tcl"):
            row = rows[f"ridges.band_view.{ridge}.functional_count"]
            assert row["value"] is None
            assert row["gap_reason"].startswith("missing: ")
    # The raw tables behind the in-memory tree are not there: a named gap, no raise.
    write_upstream(tmp_path / "raw")
    readers = fake_readers(tmp_path / "raw")

    def no_tables(*_: Any) -> None:
        raise FileNotFoundError("fitness_wt.parquet")

    readers["band_view"] = no_tables
    sources = mq.load_sources(tmp_path / "raw", readers=readers)
    rows = {r["key"]: r for r in mq.build_rows(sources, CONFIG, git_commit=None)}
    row = rows["ridges.band_view.kem.functional_count"]
    assert row["value"] is None and "fitness_wt.parquet" in row["gap_reason"]
    assert row["gap_reason"].startswith("missing: ")


def test_band_view_counts_follow_the_figures_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counts run the figure's own steps in order and count fitness >= threshold."""
    from dubanevics2026topological.analysis import (
        codon_entropy_populations as cep,
    )
    from dubanevics2026topological.analysis import describe
    from dubanevics2026topological.simplification import (
        bootstrap_smoothed_tree as bst,
    )

    calls: list[str] = []
    monkeypatch.setattr(
        bst, "check_hills_match_tree", lambda h, t: calls.append("check")
    )
    monkeypatch.setattr(
        bst,
        "build_band_simplified_view",
        lambda t, h: calls.append("band") or "view",
    )

    def populations(**kwargs: Any) -> dict[str, pl.DataFrame]:
        calls.append(f"extract:{kwargs['candidate_view']}:{kwargs['save_files']}")
        frame = pl.DataFrame({"fitness": [0.1, 0.7, 0.9]})
        return {"df_core": frame, "df_kem": frame.head(2), "df_tcl": frame.head(1)}

    monkeypatch.setattr(cep, "extract_codon_entropy_populations", populations)
    counts = describe(
        products={"ridge_counts"},
        comparison={"thresholds": {"candidate_clamped": 0.7}},
        hills=pl.DataFrame(),
        tree_analysis=object(),
        ridge_config={},
    ).ridge_counts
    assert calls == ["check", "band", "extract:view:False"]
    assert counts == {
        "core": {"total_count": 3, "functional_count": 2},
        "kem": {"total_count": 2, "functional_count": 1},
        "tcl": {"total_count": 1, "functional_count": 0},
    }


def test_band_view_counts_follow_the_class_under_the_mixture_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Functional is membership of the Functional class, whatever the raw fitness."""
    from dubanevics2026topological.analysis import (
        codon_entropy_populations as cep,
    )
    from dubanevics2026topological.analysis.band_view_ridges import (
        _band_view_ridge_counts as band_view_ridge_counts,
    )
    from dubanevics2026topological.simplification import (
        bootstrap_smoothed_tree as bst,
    )

    monkeypatch.setattr(bst, "check_hills_match_tree", lambda h, t: None)
    monkeypatch.setattr(bst, "build_band_simplified_view", lambda t, h: "view")
    frame = pl.DataFrame({"dna_sequence": ["a", "b", "c"], "fitness": [0.1, 0.7, 0.9]})
    monkeypatch.setattr(
        cep,
        "extract_codon_entropy_populations",
        lambda **kwargs: {
            "df_core": frame,
            "df_kem": frame.head(2),
            "df_tcl": frame.head(1),
        },
    )
    comparison = {"thresholds": {"candidate_clamped": 0.7}}
    # Raw fitness >= 0.7 would count b and c; the class set is a alone.
    by_class = band_view_ridge_counts(
        comparison, pl.DataFrame(), None, {}, functional_sequences={"a"}
    )
    assert by_class == {
        "core": {"total_count": 3, "functional_count": 1},
        "kem": {"total_count": 2, "functional_count": 1},
        "tcl": {"total_count": 1, "functional_count": 1},
    }
    by_fitness = band_view_ridge_counts(comparison, pl.DataFrame(), None, {})
    assert by_fitness["core"]["functional_count"] == 2


def test_registry_currency_lives_in_the_analysis_module(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no export record"):
        mq.read_registry_fingerprint(tmp_path)
    record = mq.registry_record_path(tmp_path)
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"fingerprint": "fp-1"}), encoding="utf-8")
    assert mq.read_registry_fingerprint(tmp_path) == "fp-1"
    assert mq.require_current_registry(tmp_path, "fp-1") == "fp-1"
    with pytest.raises(RuntimeError, match="stale") as error:
        mq.require_current_registry(tmp_path, "fp-2")
    assert "just manuscript-quantities" in str(error.value)
    assert mq.gap_kind("stale: x") == "stale:" and mq.gap_kind("odd") is None


def test_scan_range_and_radius_use_the_stages_own_band_width(tmp_path: Path) -> None:
    from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
        band_width,
    )

    write_upstream(tmp_path)
    sources = load(tmp_path)
    draws = pl.read_parquet(
        tmp_path / "analysis" / "bootstrap_smoothed_tree" / "draw_stats.parquet"
    )
    data = sources["bootstrap"].data
    assert data["bottleneck_range"] == {
        "minimum": band_width(draws, 0.0),
        "maximum": band_width(draws, 1.0),
    }
    assert data["radius_at_band_quantile"] == band_width(draws, 0.95)
    assert "analysis/bootstrap_smoothed_tree/draw_stats.parquet" in (
        sources["bootstrap"].files
    )
    assert "analysis/component_pairs.parquet" in sources["component"].files


# ------------------------------------------------------- the design document


def _figure_label_table() -> list[tuple[str, str, str]]:
    """The (figure, label, key) rows of the design document's Figure labels table."""
    import re

    text = (Path(__file__).resolve().parents[1] / "docs" / DESIGN_DOCUMENT).read_text(
        encoding="utf-8"
    )
    section = text.split("## Figure labels", 1)[1].split("### Population checks", 1)[0]
    return re.findall(
        r"^\| `([^`]+)` \| ([^|]+?) \| `([^`]+)` \|$", section, re.MULTILINE
    )


DESIGN_DOCUMENT = "manuscript-quantity-registry-design.md"


def test_every_documented_figure_label_key_is_a_registry_key(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=True)
    keys = set(rows_of(tmp_path))
    table = _figure_label_table()
    assert len(table) > 80
    assert {key for _, _, key in table} <= keys
    pairs = [(figure, label, key) for figure, label, key in table]
    assert len(pairs) == len(set(pairs))  # one row per printed label and key


def test_documented_figures_exist_and_labels_are_spelled_out() -> None:
    from dubanevics2026topological.figures import registry

    known = {spec.figure_id for spec in registry.FIGURE_SPECS}
    table = _figure_label_table()
    assert {figure for figure, _, _ in table} <= known
    for _, label, key in table:
        assert not any("Ͱ" <= char <= "Ͽ" for char in label), label
        assert "cutoff" not in label.lower() + key
        assert not any(
            segment in {"tau", "q", "c", "p5", "two_c"} for segment in key.split(".")
        ), key


# ------------------------------------------------------ population checks


def _with(sources: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**sources, **changes}


def test_the_full_graph_vertex_count_must_equal_the_full_fitness_table(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    sources = load(tmp_path)
    agree = mq.Upstream(mq.FULL_FITNESS, None, {"height": 1000})
    mq.build_rows(_with(sources, full_fitness=agree), CONFIG, git_commit=None)
    differ = mq.Upstream(mq.FULL_FITNESS, None, {"height": 999})
    with pytest.raises(ValueError, match="Full fitness table height") as error:
        mq.build_rows(_with(sources, full_fitness=differ), CONFIG, git_commit=None)
    assert "just comparison-analysis" in str(error.value)


def test_the_full_fitness_table_is_read_through_its_own_reader(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)
    assert mq.load_sources(tmp_path, readers=readers)["full_fitness"] is None
    readers["full_fitness_height"] = lambda: 1000
    source = mq.load_sources(tmp_path, readers=readers)["full_fitness"]
    assert source is not None and source.identity == {"height": 1000}

    def absent() -> int:
        raise FileNotFoundError("fitness_wt.parquet")

    readers["full_fitness_height"] = absent
    assert mq.load_sources(tmp_path, readers=readers)["full_fitness"] is None


def test_the_candidate_stop_codon_count_must_match_the_in_memory_tree(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)
    plain = readers["band_view"]

    def counts(count: int) -> Any:
        return lambda *args: {
            **plain(*args),
            "candidate_tree": {"stop_codon_peak_count": count},
        }

    readers["band_view"] = counts(1)
    sources = mq.load_sources(tmp_path, readers=readers)
    mq.build_rows(sources, CONFIG, git_commit=None)
    readers["band_view"] = counts(2)
    sources = mq.load_sources(tmp_path, readers=readers)
    with pytest.raises(ValueError, match="in-memory Candidate split tree") as error:
        mq.build_rows(sources, CONFIG, git_commit=None)
    assert "just comparison-analysis" in str(error.value)
    # Absent raw tables are a named gap, not a failed check.

    def no_tables(*_: Any) -> None:
        raise FileNotFoundError("fitness_wt.parquet")

    readers["band_view"] = no_tables
    sources = mq.load_sources(tmp_path, readers=readers)
    mq.build_rows(sources, CONFIG, git_commit=None)


# ------------------------------------------------- not-run versus other gaps


def test_only_an_unrun_stage_is_a_not_run_gap(tmp_path: Path) -> None:
    write_upstream(tmp_path, skip=("band", "clamped", "bootstrap"))
    rows = rows_of(tmp_path)
    for key in (
        "leading_saddle_band.global_minimum.fitness",
        "bootstrap.draws",
        "full_graph.peak_count",
        "ridges.band_view.kem.functional_count",
    ):
        assert mq.is_stage_not_run(rows[key]["gap_reason"]), key
    # The Full components await the saved landscape summary product.
    assert mq.is_stage_not_run(rows["full_graph.component_count"]["gap_reason"])


def test_absent_raw_tables_are_not_an_unrun_stage(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)

    def no_tables(*_: Any) -> None:
        raise FileNotFoundError("fitness_wt.parquet")

    readers["band_view"] = no_tables
    rows = mq.build_rows(
        mq.load_sources(tmp_path, readers=readers), CONFIG, git_commit=None
    )
    reason = {row["key"]: row["gap_reason"] for row in rows}[
        "ridges.band_view.kem.functional_count"
    ]
    assert reason.startswith("missing: the raw fitness tables")
    assert not mq.is_stage_not_run(reason)


def test_a_stale_stage_is_not_a_not_run_gap(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)

    def refuse() -> None:
        raise FileNotFoundError("checkpoint not current")

    readers["bootstrap"] = refuse
    rows = {
        row["key"]: row
        for row in mq.build_rows(
            mq.load_sources(tmp_path, readers=readers), CONFIG, git_commit=None
        )
    }
    for key in ("bootstrap.draws", "ridges.band_view.kem.functional_count"):
        reason = rows[key]["gap_reason"]
        assert reason.startswith("stale:") and not mq.is_stage_not_run(reason), key


# ---------------------------------------------- measured and core populations


def test_peak_uncertainty_measured_sequences_must_equal_the_full_graph(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    # 999 scatter rows plus the wild type is the fixture's 1000 Full vertices.
    assert rows_of(tmp_path)["peak_uncertainty.measured_sequence_count"]["value"] == (
        "1000"
    )
    directory = tmp_path / "analysis" / "peak_uncertainty"
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    diagnostics["fitness_wt_scatter_row_count"] = 1000
    _write(directory / "diagnostics.json", diagnostics)
    with pytest.raises(ValueError, match="plus the wild type") as error:
        rows_of(tmp_path)
    assert "just peak-analysis" in str(error.value)


def test_the_band_view_core_counts_must_equal_the_candidate_graphs(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path)
    plain = readers["band_view"]

    def core(total: int | None, functional: int | None) -> Any:
        def read(*args: Any) -> dict[str, Any]:
            counts = plain(*args)
            counts["core"] = {
                "total_count": total or counts["core"]["total_count"],
                "functional_count": functional or counts["core"]["functional_count"],
            }
            return counts

        return read

    sources = mq.load_sources(tmp_path, readers=readers)
    mq.build_rows(sources, CONFIG, git_commit=None)
    for bad, pattern in (
        (core(41, None), "Number of nodes"),
        (core(None, 3), "Number of functional sequences"),
    ):
        readers["band_view"] = bad
        sources = mq.load_sources(tmp_path, readers=readers)
        with pytest.raises(ValueError, match=pattern) as error:
            mq.build_rows(sources, CONFIG, git_commit=None)
        assert "just comparison-analysis" in str(error.value)


# ---------------------------------------------- functional classification


def functional_summary(**changes: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "threshold": {"unrounded": -0.4123456789, "rounded": -0.412346},
        "mixture_k": 3,
        "class_counts": {"functional": 100, "non_functional": 150, "uncertain": 7},
        "replication_cell": {"threshold": {"unrounded": -0.9, "rounded": -0.9}},
        "stop_codon_fractions_below_threshold": {
            "1": {"sequences": 10, "below_threshold": 5, "fraction": 0.5},
            "2": {"sequences": 8, "below_threshold": 7, "fraction": 0.875},
            "3": {"sequences": 0, "below_threshold": 0, "fraction": None},
        },
    }
    summary.update(changes)
    return summary


def write_functional(output: Path, summary: dict[str, Any] | None = None) -> None:
    directory = output / "checkpoints" / "functional_classification"
    _write(directory / "summary.json", summary or functional_summary())
    _write(
        directory / "provenance.json",
        {"fingerprint": "fc-fp", "artefacts": {"summary.json": "abc"}},
    )


FUNCTIONAL_KEYS = (
    "mixture_quantile_functional_threshold_fitness",
    "mixture_quantile_functional_threshold_unrounded_fitness",
    "mixture_component_count",
    "functional_class_count",
    "non_functional_class_count",
    "uncertain_class_count",
    "papkou_replication_functional_threshold_fitness",
    "double_stop_fraction_below_threshold",
)


def test_functional_classification_rows_state_population_and_rule(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    write_functional(tmp_path)
    rows = rows_of(tmp_path)
    keys = [f"functional_classification.{name}" for name in FUNCTIONAL_KEYS]
    for key in keys:
        row = rows[key]
        assert row["gap_reason"] is None and row["value"] is not None
        assert row["population"].startswith("All sequences with wild-type-relative")
        assert row["source_stage"] == "functional-classification"
        assert row["source_fingerprint"] == "fc-fp"
        assert mq.topic_of(key) == "Functional classification"
    threshold = rows[keys[0]]
    assert threshold["selection_rule"].startswith("mixture_quantile")
    assert threshold["display_value"] == "-0.412346"
    assert rows[keys[1]]["value"] == "-0.4123456789"
    assert rows[keys[2]]["display_value"] == "3"
    assert rows[keys[3]]["value"] == "100" and rows[keys[5]]["value"] == "7"
    assert rows[keys[6]]["selection_rule"] == (
        "Papkou k = 3 lowest-mean component (fixed 0.999 quantile)"
    )
    assert rows[keys[7]]["value"] == "0.875" and rows[keys[7]]["unit"] == "fraction"
    # The active rule's threshold row is untouched.
    assert "thresholds.candidate_functional.value" in rows


def test_functional_classification_absent_gives_named_gap_rows(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    rows = rows_of(tmp_path)
    for name in FUNCTIONAL_KEYS:
        row = rows[f"functional_classification.{name}"]
        assert row["value"] is None and row["display_value"] is None
        assert row["gap_reason"].startswith(
            "missing: stage functional-classification not run"
        )
        assert "rerun just functional-classification" in row["gap_reason"]
        assert row["population"] and row["selection_rule"]


def test_an_empty_double_stop_class_is_an_undefined_gap(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    fractions = functional_summary()["stop_codon_fractions_below_threshold"]
    fractions["2"] = {"sequences": 0, "below_threshold": 0, "fraction": None}
    write_functional(
        tmp_path, functional_summary(stop_codon_fractions_below_threshold=fractions)
    )
    row = rows_of(tmp_path)[
        "functional_classification.double_stop_fraction_below_threshold"
    ]
    assert row["value"] is None and row["gap_reason"].startswith("undefined:")


# ------------------------------------------- functional threshold sensitivity scan

SCAN_READER = "functional_threshold_scan"
SCAN_ROWS = [f"functional_threshold_scan.{name}" for name in mq.SCAN_KEYS]


def scan_result(
    *,
    passed: tuple[bool, bool, bool | None] = (True, True, True),
    interval: dict[str, Any] | None = None,
    axes: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """A synthetic scan: the pieces ``load_functional_threshold_scan`` returns."""
    a, b, c = passed
    low, high = -0.62, -0.58
    interval = interval or {
        "family": "gaussian",
        "k": 3,
        "threshold_low": low,
        "threshold_high": high,
        "width": high - low,
        "n_converged": 195,
        "n_resamples": 200,
    }
    axes = axes or {
        "family": {"max_flip_fraction": 0.021, "passed": True, "neighbours": []},
        "k": {"max_flip_fraction": 0.034, "passed": True, "neighbours": []},
        "quantile": {"max_flip_fraction": 0.008, "passed": True, "neighbours": []},
        "confidence": {"max_flip_fraction": None, "passed": True, "neighbours": []},
        "component_rule": {"max_flip_fraction": 0.0, "passed": True, "neighbours": []},
    }
    keys = ("threshold_low", "threshold_high", "width")
    return SimpleNamespace(
        acceptance={
            "canonical": {
                "family": "gaussian",
                "k": 3,
                "component_rule": "stop_anchored",
                "quantile": 0.999,
                "confidence": 0.95,
                "threshold": -0.6,
            },
            "a": {"frac_2stop_below": 0.995, "minimum": 0.99, "passed": a},
            "b": {
                "family": "gaussian",
                "k": 3,
                "median_se_pooled": 0.05,
                "limit": 0.196,
                "passed": b,
                **{key: interval[key] for key in keys},
            },
            "c": {
                "flip_fraction_limit": 0.05,
                "canonical_functional_count": 1000,
                "axes": axes,
                "passed": c,
            },
            "passed": {"a": a, "b": b, "c": c, "all": a and b and c},
        },
        intervals=pl.DataFrame([interval]),
        provenance={
            "scan_config": {"n_resamples": 200, "random_seed": 1},
            "classification_config": {"k_range": [1, 5]},
            "classification_source": {"fingerprint": "fc-fp"},
            "growth_rates_sha256": "g" * 8,
            "artifacts": {"cells.parquet": "abc"},
        },
    )


def scan_rows(tmp_path: Path, result: SimpleNamespace | None):
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path, _tree())
    if result is not None:
        readers[SCAN_READER] = lambda: result
    sources = mq.load_sources(tmp_path, readers=readers)
    rows = mq.build_rows(sources, CONFIG, git_commit="abc123")
    return {row["key"]: row for row in rows}, sources


def test_scan_rows_carry_the_acceptance_checks_and_the_canonical_interval(
    tmp_path: Path,
) -> None:
    rows, sources = scan_rows(tmp_path, scan_result())
    assert [k for k in rows if k.startswith("functional_threshold_scan.")] == SCAN_ROWS
    for key in SCAN_ROWS:
        row = rows[key]
        assert row["gap_reason"] is None and row["value"] is not None, key
        assert row["source_stage"] == "functional-threshold-scan"
        assert row["source_fingerprint"] == sources[SCAN_READER].fingerprint
        assert row["population"] and row["selection_rule"]
        assert mq.topic_of(key) == "Functional threshold sensitivity scan"
        # a setting a row names is itself a row
        for setting in (row["settings"] or "").split(mq.SETTINGS_SEPARATOR):
            assert not setting or setting in rows, (key, setting)
    value = {k.split(".")[1]: rows[k]["value"] for k in SCAN_ROWS}
    assert (
        value["acceptance_a_passed"] == "true"
        and value["acceptance_all_passed"] == "true"
    )
    assert rows["functional_threshold_scan.acceptance_all_passed"]["unit"] == "boolean"
    assert (
        rows["functional_threshold_scan.acceptance_b_passed"]["display_value"] == "true"
    )
    assert value["canonical_threshold_interval_low"] == "-0.62"
    assert value["canonical_threshold_interval_high"] == "-0.58"
    assert float(value["canonical_threshold_interval_width"]) == pytest.approx(0.04)
    assert value["interval_width_limit"] == "0.196"
    assert value["resample_count"] == "200"
    assert rows["functional_threshold_scan.resample_count"]["display_value"] == "200"
    # the worst axis, skipping the axis with no neighbour in the scan
    assert value["max_flip_fraction_one_step"] == "0.034"
    assert value["flip_fraction_limit"] == "0.05"
    assert value["double_stop_below_minimum"] == "0.99"
    assert "(gaussian, k = 3)" in rows[SCAN_ROWS[7]]["selection_rule"]


def test_a_failed_check_is_a_false_row_not_a_gap(tmp_path: Path) -> None:
    rows, _ = scan_rows(tmp_path, scan_result(passed=(True, False, True)))
    assert rows["functional_threshold_scan.acceptance_b_passed"]["value"] == "false"
    assert rows["functional_threshold_scan.acceptance_all_passed"]["value"] == "false"
    assert rows["functional_threshold_scan.acceptance_a_passed"]["value"] == "true"
    assert all(rows[k]["gap_reason"] is None for k in SCAN_ROWS)


def test_file_hashes_skip_atomic_write_temporaries_but_not_named_dotfiles(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "analysis" / "scan"
    scan.mkdir(parents=True)
    (scan / "cells.parquet").write_bytes(b"cells")
    (scan / ".cells.k3j2.parquet").write_bytes(b"half written")
    sidecar = tmp_path / "analysis" / ".sidecar.json"
    sidecar.write_text("{}")
    hashes = mq._hashes(tmp_path, scan, sidecar)
    assert sorted(hashes) == ["analysis/.sidecar.json", "analysis/scan/cells.parquet"]


def test_an_undefined_check_is_an_undefined_gap_not_a_false_row(
    tmp_path: Path,
) -> None:
    result = scan_result(passed=(True, True, None))
    result.acceptance["c"]["reason"] = "confidence: no_neighbours"
    rows, _ = scan_rows(tmp_path, result)
    c = rows["functional_threshold_scan.acceptance_c_passed"]
    assert c["value"] is None
    assert c["gap_reason"].startswith("undefined: check c has no verdict")
    assert "no_neighbours" in c["gap_reason"]
    every = rows["functional_threshold_scan.acceptance_all_passed"]
    assert every["value"] is None and every["gap_reason"].startswith("undefined:")
    assert rows["functional_threshold_scan.acceptance_a_passed"]["value"] == "true"


def test_a_scan_that_has_not_run_gives_named_gap_rows(tmp_path: Path) -> None:
    rows, sources = scan_rows(tmp_path, None)
    assert sources[SCAN_READER] is None
    for key in SCAN_ROWS:
        row = rows[key]
        assert row["value"] is None and row["display_value"] is None, key
        assert row["gap_reason"].startswith(
            "missing: stage functional-threshold-scan not run"
        )
        assert row["population"] and row["selection_rule"]
        assert row["source_stage"] == "functional-threshold-scan"


def test_a_scan_the_loader_refuses_is_a_gap_that_names_its_rerun(
    tmp_path: Path,
) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path, _tree())

    def refuse() -> None:
        raise FileNotFoundError("stale functional-threshold scan at x")

    readers[SCAN_READER] = refuse
    sources = mq.load_sources(tmp_path, readers=readers)
    rows = {r["key"]: r for r in mq.build_rows(sources, CONFIG, git_commit=None)}
    reason = rows[SCAN_ROWS[3]]["gap_reason"]
    assert reason.startswith("missing: stage functional-threshold-scan not run")
    assert "rerun just functional-threshold-scan" in reason
    # with an output present, the same refusal is a stale gap
    scan_dir = tmp_path.joinpath(*mq.SCAN_DIR)
    scan_dir.mkdir(parents=True)
    (scan_dir / "cells.parquet").write_text("x")
    sources = mq.load_sources(tmp_path, readers=readers)
    rows = {r["key"]: r for r in mq.build_rows(sources, CONFIG, git_commit=None)}
    assert rows[SCAN_ROWS[3]]["gap_reason"].startswith("stale:")


def test_an_empty_interval_and_no_scan_step_are_undefined_gaps(
    tmp_path: Path,
) -> None:
    no_interval = {
        "family": "gaussian",
        "k": 3,
        "threshold_low": None,
        "threshold_high": None,
        "width": None,
        "n_converged": 0,
        "n_resamples": 200,
    }
    no_steps = {
        name: {"max_flip_fraction": None, "passed": True, "neighbours": []}
        for name in ("family", "k", "quantile", "confidence", "component_rule")
    }
    rows, _ = scan_rows(
        tmp_path,
        scan_result(passed=(True, False, True), interval=no_interval, axes=no_steps),
    )
    for name in (
        "canonical_threshold_interval_low",
        "canonical_threshold_interval_high",
        "canonical_threshold_interval_width",
        "max_flip_fraction_one_step",
    ):
        row = rows[f"functional_threshold_scan.{name}"]
        assert row["value"] is None and row["gap_reason"].startswith("undefined:")
    # the limit and the checks still have values
    assert rows["functional_threshold_scan.interval_width_limit"]["value"] == "0.196"
    assert rows["functional_threshold_scan.acceptance_b_passed"]["value"] == "false"


def test_an_interval_table_that_disagrees_with_the_acceptance_record_raises(
    tmp_path: Path,
) -> None:
    result = scan_result()
    result.acceptance["b"]["width"] = 0.5
    with pytest.raises(ValueError, match="differs from check b of acceptance.json"):
        scan_rows(tmp_path, result)


def test_only_the_acceptance_check_keys_may_use_a_single_letter() -> None:
    for letter in "abc":
        mq.validate_rows(
            [{"key": f"functional_threshold_scan.acceptance_{letter}_passed"}]
        )
    for bad in (
        "functional_threshold_scan.acceptance_d_passed",
        "functional_threshold_scan.acceptance_a_passed.extra",
        "thresholds.acceptance_a_passed",
    ):
        with pytest.raises(ValueError, match="single letter"):
            mq.validate_rows([{"key": bad}])


def test_the_scan_is_part_of_the_registry_identity(tmp_path: Path) -> None:
    _, absent = scan_rows(tmp_path, None)
    assert mq.source_identities(absent)[SCAN_READER] is None
    scan_dir = tmp_path.joinpath(*mq.SCAN_DIR)
    scan_dir.mkdir(parents=True)
    (scan_dir / "cells.parquet").write_text("one")
    _, first = scan_rows(tmp_path, scan_result())
    identity = mq.source_identities(first)[SCAN_READER]
    assert identity["stage"] == "functional-threshold-scan"
    assert identity["identity"]["scan_config"]["n_resamples"] == 200
    assert list(identity["files"]) == [
        "analysis/functional_threshold_scan/cells.parquet"
    ]
    (scan_dir / "cells.parquet").write_text("two")
    _, second = scan_rows(tmp_path, scan_result())
    assert mq.source_identities(second)[SCAN_READER] != identity
    assert second[SCAN_READER].fingerprint == first[SCAN_READER].fingerprint


def test_the_cached_scan_loader_turns_a_stale_scan_into_not_found(monkeypatch) -> None:
    from dubanevics2026topological.analysis import functional_threshold_scan as fts

    assert mq.cached_readers()[SCAN_READER] is fts.load_cached_functional_threshold_scan

    def stale(settings, *, scan_dir):
        raise RuntimeError("stale functional-threshold scan at x: scan_config differ")

    monkeypatch.setattr(fts, "load_functional_threshold_scan", stale)
    with pytest.raises(FileNotFoundError, match="stale functional-threshold scan"):
        fts.load_cached_functional_threshold_scan()


def test_the_scan_keys_are_in_the_manuscript_order() -> None:
    assert set(SCAN_ROWS) <= set(mq.MANUSCRIPT_ORDER)
    assert [k for k in mq.MANUSCRIPT_ORDER if k in SCAN_ROWS] == SCAN_ROWS
