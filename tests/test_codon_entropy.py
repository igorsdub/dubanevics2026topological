from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis.codon_entropy import (
    compute_positional_entropy_sweep,
    process_codon_entropy_datasets,
)
from dubanevics2026topological.analysis.codon_entropy_populations import (
    _frames_for_populations,
    _sequence_frame,
    _validate_sequence_translation,
    extract_codon_entropy_populations,
)


def _entropy_frame(rows: list[tuple[float, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fitness": [row[0] for row in rows],
            "aa1": [row[1] for row in rows],
            "aa2": [row[2] for row in rows],
            "aa3": [row[3] for row in rows],
        }
    )


def test_positional_entropy_balanced_two_state_is_one_bit() -> None:
    result = compute_positional_entropy_sweep(
        _entropy_frame([(1.0, "A", "E", "M"), (1.0, "K", "D", "M")]),
        thresholds=[1.0],
    )

    assert result.filter(pl.col("aa_position") == 1)["entropy_bits"][
        0
    ] == pytest.approx(1.0)
    assert result.filter(pl.col("aa_position") == 2)["entropy_bits"][
        0
    ] == pytest.approx(1.0)
    assert result.filter(pl.col("aa_position") == 3)["entropy_bits"][
        0
    ] == pytest.approx(0.0)


def test_shared_grid_masks_values_outside_population_range() -> None:
    result = compute_positional_entropy_sweep(
        _entropy_frame([(0.2, "A", "E", "M"), (0.8, "K", "D", "M")]),
        thresholds=[0.1, 0.2, 0.8, 0.9],
    )

    low = result.filter(pl.col("threshold_scalar") == 0.1)
    high = result.filter(pl.col("threshold_scalar") == 0.9)
    assert low["active_sequence_count"].to_list() == [0, 0, 0]
    assert high["active_sequence_count"].to_list() == [0, 0, 0]
    assert low["entropy_bits"].to_list() == [None, None, None]
    assert high["entropy_bits"].to_list() == [None, None, None]
    assert result.filter(pl.col("threshold_scalar") == 0.2)[
        "active_sequence_count"
    ].to_list() == [2, 2, 2]


def test_empty_population_has_null_entropy_and_zero_counts() -> None:
    empty = _entropy_frame([])
    result = compute_positional_entropy_sweep(empty, thresholds=[0.1, 0.2])

    assert result.height == 6
    assert result["entropy_bits"].to_list() == [None] * 6
    assert result["active_sequence_count"].to_list() == [0] * 6


def test_population_sweeps_share_grid_and_include_each_population_endpoint(
    tmp_path,
) -> None:
    populations = {
        "core": [(0.0, "A", "E", "M"), (1.0, "K", "D", "M")],
        "kem": [(0.25, "A", "E", "M"), (0.75, "K", "D", "M")],
        "tcl": [(0.5, "A", "E", "M")],
    }
    stems = {
        "core": "papkou2023_candidate_population.parquet",
        "kem": "papkou2023_kem_ridge_population.parquet",
        "tcl": "papkou2023_tcl_ridge_population.parquet",
    }
    for name, rows in populations.items():
        _entropy_frame(rows).write_parquet(tmp_path / stems[name])

    results = process_codon_entropy_datasets(
        input_dir=tmp_path, save_files=False, num_thresholds=2
    )

    for frame in results.values():
        assert frame["threshold_scalar"].unique().sort().to_list() == [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ]
    assert (
        results["kem"].filter(pl.col("threshold_scalar") == 0.25)[
            "active_sequence_count"
        ][0]
        == 2
    )


def test_site_evidence_keeps_stop_codons_as_distinct_dna_and_aa_states() -> None:
    seqs = _sequence_frame(
        pl.DataFrame(
            {
                "dna_sequence": [
                    "TAA GAAATG".replace(" ", ""),
                    "TAGGAAATG",
                    "TGAGAAATG",
                    "ACGGAAATG",
                ],
                "aa_sequence": ["%EM", "^EM", "&EM", "TEM"],
                "scalar": [1.0, 0.8, 0.6, 0.4],
            }
        )
    ).with_columns(pl.lit(True).alias("is_functional_core"))
    frequencies, entropies, _ = _frames_for_populations({"core": seqs})

    codon_first_states = frequencies.filter(
        (pl.col("molecule") == "dna")
        & (pl.col("site_index") == 1)
        & (pl.col("fitness_group") == "all")
    )
    aa_first_states = frequencies.filter(
        (pl.col("molecule") == "amino_acid")
        & (pl.col("site_index") == 1)
        & (pl.col("fitness_group") == "all")
    )
    assert set(codon_first_states["state"]) == {"T", "A"}
    assert set(aa_first_states["state"]) == {"%", "^", "&", "T"}
    assert (
        entropies.filter(
            (pl.col("molecule") == "amino_acid") & (pl.col("site_index") == 1)
        )["state_count"][0]
        == 4
    )


def test_sequence_translation_rejects_mismatched_stop_and_sense_codons() -> None:
    mismatch = _sequence_frame(
        pl.DataFrame(
            {
                "dna_sequence": ["TAAGAAATG", "ACGGAAATG"],
                "aa_sequence": ["^EM", "KEM"],
                "scalar": [1.0, 0.5],
            }
        )
    )

    with pytest.raises(ValueError, match="translates to"):
        _validate_sequence_translation(
            mismatch,
            dna_alphabet=set("ACGT"),
            aa_alphabet=set("ACDEFGHIKLMNPQRSTVWY%^&"),
        )


def test_balanced_four_state_dna_site_has_two_bits_entropy() -> None:
    from dubanevics2026topological.analysis.codon_entropy_populations import (
        _sequence_frame,
    )

    sequences = _sequence_frame(
        pl.DataFrame(
            {
                "dna_sequence": ["ACGGAAATG", "CCGGAAATG", "GCGGAAATG", "TCGGAAATG"],
                "aa_sequence": ["TEM", "PEM", "AEM", "SEM"],
                "scalar": [1.0, 0.9, 0.8, 0.7],
            }
        )
    ).with_columns(pl.lit(True).alias("is_functional_core"))
    _, entropies, _ = _frames_for_populations({"core": sequences})

    observed = entropies.filter(
        (pl.col("population_id") == "core")
        & (pl.col("fitness_group") == "all")
        & (pl.col("molecule") == "dna")
        & (pl.col("site_index") == 1)
    )["entropy_bits"][0]
    assert observed == pytest.approx(2.0)


def _simplified_view_fixture():
    import rustworkx as rx

    tree = rx.PyDiGraph()
    nodes = [
        ("AAGGAAATG", "KEM", 1.39, "maximum"),
        ("ACCTGTCTT", "TCL", 0.41, "maximum"),
        ("CTCTGCTTG", "LCL", -0.23, "split_saddle"),
        ("GCTGCTGCT", "AAA", -0.7, "minimum"),
    ]
    for index, (dna, aa, scalar, event) in enumerate(nodes):
        tree.add_node(
            {
                "dna_sequence": dna,
                "aa_sequence": aa,
                "scalar": scalar,
                "event": event,
            }
        )
    tree.add_edges_from_no_data([(0, 2), (1, 2), (2, 3)])
    source_rows = [
        ("AAGGAAATG", "KEM", 1.39, "AAGGAAATG", "CTCTGCTTG"),
        ("AAGGAGATG", "KEM", 0.2, "AAGGAAATG", "CTCTGCTTG"),
        ("ACCTGTCTT", "TCL", 0.41, "ACCTGTCTT", "CTCTGCTTG"),
        ("ACCTGTCTC", "TCL", -0.1, "ACCTGTCTT", "CTCTGCTTG"),
        ("CTCTGCTTG", "LCL", -0.23, "CTCTGCTTG", "GCTGCTGCT"),
        ("GCTGCTGCT", "AAA", -0.7, "GCTGCTGCT", "GCTGCTGCT"),
        ("GCTGCTGCC", "AAA", -0.8, "GCTGCTGCT", "GCTGCTGCT"),
    ]
    view = SimpleNamespace(
        tree=tree,
        assignments=pl.DataFrame(
            {
                "dna_sequence": [r[0] for r in source_rows],
                "aa_sequence": [r[1] for r in source_rows],
                "raw_fitness": [r[2] for r in source_rows],
                "upper_sequence": [r[3] for r in source_rows],
                "lower_sequence": [r[4] for r in source_rows],
            }
        ),
    )

    comparison = {
        "thresholds": {"candidate_clamped": -0.5},
        "diagnostics": {"candidate_clamped": {"selected_component_vertex_count": 7}},
        "provenance": {"fingerprint": {"source": "fixture"}},
    }
    config = {
        "candidate_population": "all_candidate_selected_graph_vertices_including_fringe",
        "kem_summit_dna_sequence": "AAGGAAATG",
        "tcl_summit_dna_sequence": "ACCTGTCTT",
        "reported_saddle_dna_sequence": "CTCTGCTTG",
        "aa_alphabet": list("ACDEFGHIKLMNPQRSTVWY") + ["%", "^", "&"],
        "dna_alphabet": list("ACGT"),
        "stop_codons": ["TAA", "TAG", "TGA"],
    }

    return comparison, view, config


def test_candidate_simplified_populations_include_fringe_and_exclude_saddle() -> None:
    comparison, view, config = _simplified_view_fixture()
    result = extract_codon_entropy_populations(
        comparison=comparison, candidate_view=view, config=config
    )

    assert result["df_core"].height == 7
    assert (
        result["df_core"].filter(pl.col("dna_sequence") == "GCTGCTGCC")[
            "is_functional_core"
        ][0]
        is False
    )
    assert "CTCTGCTTG" not in result["df_kem"]["dna_sequence"].to_list()
    assert "CTCTGCTTG" not in result["df_tcl"]["dna_sequence"].to_list()
    assert set(result["df_kem"]["dna_sequence"]).isdisjoint(
        result["df_tcl"]["dna_sequence"]
    )
    assert result["metadata"]["common_saddle"]["raw_fitness"] == pytest.approx(-0.23)
    assert (
        result["sequence_assignments"].filter(pl.col("dna_sequence") == "CTCTGCTTG")[
            "ridge_id"
        ][0]
        == "shared"
    )
    assert "median_reflected" in result["metadata"]["functional_rule"]
    assert result["metadata"]["schema"] == 3


def test_functional_core_follows_the_class_under_the_mixture_rule() -> None:
    comparison, view, config = _simplified_view_fixture()
    # Raw fitness at or above -0.5 would call five sequences functional; the class
    # set is the TCL summit and one sequence below the threshold.
    members = {"ACCTGTCTT", "GCTGCTGCC"}
    result = extract_codon_entropy_populations(
        comparison=comparison,
        candidate_view=view,
        config=config,
        functional_sequences=members,
    )
    core = result["df_core"]
    flagged = set(core.filter(pl.col("is_functional_core"))["dna_sequence"])
    assert flagged == members
    assert result["metadata"]["counts"]["tcl_functional"] == 1
    assert result["metadata"]["counts"]["kem_functional"] == 0
    assert "Functional class" in result["metadata"]["functional_rule"]
    row = result["claim_summary"].filter(
        pl.col("claim_id") == "functional_sequences_in_two_ridges"
    )
    assert (row["numerator"][0], row["denominator"][0]) == (1, 2)


def test_the_kem_ridge_reaches_the_shared_saddle_through_its_own_saddle() -> None:
    """The KEM peak's edge ends at a saddle of its own, above the shared one.

    The KEM ridge population still runs down to the shared saddle: the sequences of
    the edge between its own saddle and the shared one belong to it, as the TCL
    ridge's edge into the shared saddle does, so both sweeps cover the span down to
    the saddle (saddle excluded).
    """
    import rustworkx as rx

    tree = rx.PyDiGraph()
    nodes = [
        ("AAGGAAATG", "KEM", 1.39, "maximum"),
        ("ACCTGTCTT", "TCL", 0.41, "maximum"),
        ("CTCTGCTTG", "LCL", -0.23, "split_saddle"),
        ("GCTGCTGCT", "AAA", -0.7, "minimum"),
        ("AAGGAAAGG", "KER", -0.1, "split_saddle"),
    ]
    for dna, aa, scalar, event in nodes:
        tree.add_node(
            {"dna_sequence": dna, "aa_sequence": aa, "scalar": scalar, "event": event}
        )
    tree.add_edges_from_no_data([(0, 4), (4, 2), (1, 2), (2, 3)])
    rows = [
        ("AAGGAAATG", "KEM", 1.39, "AAGGAAATG", "AAGGAAAGG"),
        ("AAGGAAAGG", "KER", -0.1, "AAGGAAAGG", "CTCTGCTTG"),
        ("AAGGAAAGC", "KES", -0.22, "AAGGAAAGG", "CTCTGCTTG"),
        ("ACCTGTCTT", "TCL", 0.41, "ACCTGTCTT", "CTCTGCTTG"),
        ("ACCTGTCTC", "TCL", -0.21, "ACCTGTCTT", "CTCTGCTTG"),
        ("CTCTGCTTG", "LCL", -0.23, "CTCTGCTTG", "GCTGCTGCT"),
        ("GCTGCTGCT", "AAA", -0.7, "GCTGCTGCT", "GCTGCTGCT"),
    ]
    view = SimpleNamespace(
        tree=tree,
        assignments=pl.DataFrame(
            {
                "dna_sequence": [r[0] for r in rows],
                "aa_sequence": [r[1] for r in rows],
                "raw_fitness": [r[2] for r in rows],
                "upper_sequence": [r[3] for r in rows],
                "lower_sequence": [r[4] for r in rows],
            }
        ),
    )
    comparison, _view, config = _simplified_view_fixture()
    comparison["diagnostics"]["candidate_clamped"]["selected_component_vertex_count"] = 7
    result = extract_codon_entropy_populations(
        comparison=comparison, candidate_view=view, config=config
    )
    saddle = result["metadata"]["common_saddle"]["raw_fitness"]
    assert saddle == pytest.approx(-0.23)
    for name in ("df_kem", "df_tcl"):
        frame = result[name]
        lowest = float(frame["fitness"].min())
        # Strictly above the saddle, and within 0.03 of it: the span down to the saddle.
        assert saddle < lowest < saddle + 0.03
    assert "AAGGAAAGC" in result["df_kem"]["dna_sequence"].to_list()
