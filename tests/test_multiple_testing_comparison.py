"""BH and Bonferroni adjustments, and the BY count against the scan's classes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from statsmodels.stats.multitest import multipletests

from dubanevics2026topological.figures.multiple_testing_comparison import (
    bh_adjusted_p_values,
    bonferroni_adjusted_p_values,
    build_multiple_testing_comparison_figure,
    comparison_counts,
    functional_call_count,
)
from dubanevics2026topological.threshold.classification_probability_scans import (
    FDR_PRIMARY_TARGET,
    adjust_p_values_fdr,
    assign_fdr_classes,
    create_fdr_classification_table,
)

_TOY_P = np.array([0.001, 0.01, 0.03, 0.04, 0.2, 0.5], dtype=np.float64)


def _hand_bonferroni(p_values: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, p_values * p_values.size)


def _hand_bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    n = p_values.size
    raw = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = np.minimum(monotone, 1.0)
    return adjusted


def test_bonferroni_matches_the_hand_values_and_statsmodels() -> None:
    got = bonferroni_adjusted_p_values(_TOY_P)
    np.testing.assert_allclose(got, _hand_bonferroni(_TOY_P))
    rejected, expected, _, _ = multipletests(_TOY_P, method="bonferroni")
    np.testing.assert_allclose(got, expected)
    assert rejected.tolist() == (got <= FDR_PRIMARY_TARGET).tolist()


def test_bh_matches_the_hand_values_and_statsmodels() -> None:
    got = bh_adjusted_p_values(_TOY_P)
    np.testing.assert_allclose(got, _hand_bh(_TOY_P), rtol=1e-10, atol=1e-12)
    _, expected, _, _ = multipletests(_TOY_P, method="fdr_bh")
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-12)


def test_by_count_matches_the_pipelines_classification() -> None:
    growth = pl.DataFrame(
        {
            "dna_sequence": ["ATGATGATG", "CCCGCCGCC", "TTATTATTA", "AAACCCGGG"],
            "sequence_id": ["a", "b", "c", "d"],
            "s": [0.4, 0.0, -0.2, -1.0],
            "se_pooled": [0.05, 0.05, 0.05, 0.05],
            "stop_codon_count": [0, 0, 1, 0],
        }
    )
    table = create_fdr_classification_table(
        growth, threshold=-0.25, method="by", target=FDR_PRIMARY_TARGET
    )
    stored = int((table["fdr_class"] == "functional").sum())
    counted = functional_call_count(
        table["gated_p_value"].to_numpy(),
        table["is_baseline_candidate"].to_numpy(),
        "by",
        FDR_PRIMARY_TARGET,
    )
    _, log_adjusted = adjust_p_values_fdr(
        table["log_gated_p_value"].to_numpy(), method="by"
    )
    _, pipeline, _, _ = assign_fdr_classes(
        table["is_baseline_candidate"].to_numpy(),
        log_adjusted,
        FDR_PRIMARY_TARGET,
    )
    assert counted == stored == pipeline
    frame = comparison_counts(
        {"papkou_original": table, "median_reflected": table}, FDR_PRIMARY_TARGET
    )
    by_rows = frame.filter(pl.col("method") == "Benjamini-Yekutieli")
    assert by_rows["functional_count"].to_list() == [stored, stored]


def test_builder_draws_both_thresholds_and_three_corrections() -> None:
    growth = pl.DataFrame(
        {
            "dna_sequence": ["ATGATGATG", "CCCGCCGCC", "TTATTATTA"],
            "sequence_id": ["a", "b", "c"],
            "s": [1.0, -0.1, -1.0],
            "se_pooled": [0.05, 0.05, 0.2],
            "stop_codon_count": [0, 0, 0],
        }
    )
    published = create_fdr_classification_table(growth, threshold=-0.5, method="by")
    candidate = create_fdr_classification_table(growth, threshold=-0.2, method="by")
    settings = {
        "figures": {
            "rendering": {"font_size": 12, "tree_height": 250},
            "figures": {"multiple_testing_comparison": {"width": 420, "height": 250}},
        }
    }
    chart = build_multiple_testing_comparison_figure(
        {
            "tables": {
                "papkou_original": published,
                "median_reflected": candidate,
            },
            "nominal_q": FDR_PRIMARY_TARGET,
        },
        settings,
        "vanilla",
    )
    plotted = pl.DataFrame(chart.data)
    assert set(plotted["threshold"]) == {"Published", "Candidate"}
    assert set(plotted["method"]) == {
        "Bonferroni",
        "Benjamini-Hochberg",
        "Benjamini-Yekutieli",
    }
    assert plotted.height == 6
    assert chart.title is not None


def test_by_mismatch_is_refused() -> None:
    table = pl.DataFrame(
        {
            "gated_p_value": [0.01, 1.0],
            "is_baseline_candidate": [True, False],
            "fdr_class": ["uncertain", "non_functional"],
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        comparison_counts(
            {"papkou_original": table, "median_reflected": table}, FDR_PRIMARY_TARGET
        )
