"""Threshold public-interface and fixed-threshold contract tests."""

from __future__ import annotations

import pickle
import tomllib
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from dubanevics2026topological import threshold
from dubanevics2026topological.threshold import (
    candidate_threshold as legacy_threshold_module,
)
from dubanevics2026topological.threshold import (
    classification_probability_scans as scans,
)
from dubanevics2026topological.threshold import fdr_selection
from dubanevics2026topological.threshold import (
    functional_threshold as candidate_threshold,
)
from dubanevics2026topological.threshold import functional_threshold as threshold_impl


def test_threshold_facade_exports_implementation_objects() -> None:
    assert threshold_impl is legacy_threshold_module
    assert (
        threshold.resolve_functional_threshold
        is candidate_threshold.resolve_candidate_threshold
    )
    assert threshold.load_fdr_snapshot is fdr_selection.load_fdr_snapshot
    assert threshold.adjust_threshold_p_values is scans.adjust_p_values_fdr
    assert threshold.FunctionalThreshold is threshold_impl.FunctionalThreshold
    assert threshold_impl.CandidateThreshold is threshold.FunctionalThreshold
    assert (
        pickle.loads(
            b"cdubanevics2026topological.threshold.candidate_threshold\n"
            b"CandidateThreshold\n."
        )
        is threshold.FunctionalThreshold
    )


def test_published_threshold_keeps_exact_full_precision_value() -> None:
    published = next(
        item
        for item in threshold.get_active_threshold_definitions()
        if item.name == "papkou_original"
    )
    assert published.source_precision_threshold == -0.5077739708744209
    assert published.applied_threshold == -0.5077739708744209


def test_published_point_estimate_and_by_precision() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/analysis.toml"
    with config_path.open("rb") as stream:
        published_point = tomllib.load(stream)["clamped"][
            "published_functional_threshold"
        ]
    published_by = next(
        item.applied_threshold
        for item in threshold.get_active_threshold_definitions()
        if item.name == "papkou_original"
    )
    assert published_point == -0.507774
    assert published_by == norm.ppf(0.999, loc=-0.721, scale=0.069)
    assert round(published_by, 6) == published_point
    assert abs(published_by - published_point) < 1e-6


def test_gated_one_sided_p_values_and_by_family_use_all_rows() -> None:
    values = np.array([-1.4, -1.270, -0.75, -0.72, 0.0, 0.0])
    se = np.full(len(values), 0.03)
    is_above, _, _, _, gated, log_gated = threshold.compute_threshold_gated_p_values(
        values, se, -1.3
    )
    assert gated[0] == 1.0
    assert gated[1] == norm.sf((values[1] + 1.3) / 0.03)
    # The adjustment's harmonic factor is the number of rows passed to it, here N=6.
    adjusted, adjusted_log = threshold.adjust_threshold_p_values(log_gated, method="by")
    assert adjusted_log.shape == (6,)
    assert is_above.tolist() == [False, True, True, True, True, True]
    expected_harmonic = sum(1 / rank for rank in range(1, 7))
    assert np.isclose(threshold.harmonic_number(6), expected_harmonic)
    order = np.argsort(gated)
    expected_sorted = np.minimum.accumulate(
        (np.sort(gated) * 6 * expected_harmonic / np.arange(1, 7))[::-1]
    )[::-1]
    expected_sorted = np.minimum(expected_sorted, 1.0)
    np.testing.assert_allclose(adjusted[order], expected_sorted)


def test_by_adjustment_uses_the_complete_full_library_family() -> None:
    family_size = 261_333
    threshold_value = -0.5
    values = np.full(family_size, threshold_value - 0.1)
    values[0] = threshold_value + 0.33
    errors = np.full(family_size, 0.03)

    library_ids = [f"sequence-{index}" for index in range(family_size)]
    result = threshold.classify_functional_sequences(
        values,
        errors,
        threshold_value,
        sequence_ids=library_ids,
        library_ids=library_ids,
    )

    expected_q = (
        norm.sf(0.33 / 0.03) * family_size * threshold.harmonic_number(family_size)
    )
    assert result.family_size == family_size
    assert result.gated_p_values[1] == 1.0
    assert result.q_values[0] == pytest.approx(expected_q, rel=1e-12)


def test_approved_toy_threshold_classes_match_expected_values() -> None:
    # The WT is TAAACA (s=0); the five rows are the complete BY family (N=5).
    sequences = ["TAATAA", "CAAAAA", "TAAAAA", "TAAAAG", "TAAACA"]
    values = np.array([-1.4, -1.270, -0.75, -0.72, 0.0])
    errors = np.full(5, 0.03)
    threshold_value = -1.3
    result = threshold.classify_functional_sequences(
        values,
        errors,
        threshold_value,
        sequence_ids=sequences,
        library_ids=sequences,
        nominal_q=0.05,
    )

    assert result.family_size == 5
    assert result.gated_p_values[1] == pytest.approx(0.1586552539)
    assert result.q_values[1] == pytest.approx(0.4528285373)
    assert dict(zip(sequences, result.classes, strict=True)) == {
        "TAATAA": "non_functional",
        "CAAAAA": "uncertain",
        "TAAAAA": "functional",
        "TAAAAG": "functional",
        "TAAACA": "functional",
    }


@pytest.mark.parametrize(
    ("sequence_ids", "library_ids", "message"),
    [
        (["a", "b"], ["a", "b", "c"], "missing=1, extra=0"),
        (["a", "b", "x"], ["a", "b", "c"], "missing=1, extra=1"),
        (["a", "b", "b"], ["a", "b", "c"], "duplicate"),
        (["a", "c"], ["a", "b", "c"], "missing=1, extra=0"),
    ],
)
def test_classification_rejects_incomplete_or_invalid_library_coverage(
    sequence_ids: list[str], library_ids: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        threshold.classify_functional_sequences(
            [-1.0] * len(sequence_ids),
            [0.1] * len(sequence_ids),
            -0.5,
            sequence_ids=sequence_ids,
            library_ids=library_ids,
        )


def test_classification_result_arrays_are_readonly_non_aliasing_copies() -> None:
    ids = ["a", "b"]
    values = np.array([-0.4, -0.6])
    errors = np.array([0.1, 0.1])
    result = threshold.classify_functional_sequences(
        values,
        errors,
        -0.5,
        sequence_ids=ids,
        library_ids=ids,
    )
    classes = result.classes.copy()
    p_values = result.gated_p_values.copy()
    q_values = result.q_values.copy()

    values[:] = 10.0
    errors[:] = 10.0

    np.testing.assert_array_equal(result.classes, classes)
    np.testing.assert_array_equal(result.gated_p_values, p_values)
    np.testing.assert_array_equal(result.q_values, q_values)
    with pytest.raises(ValueError, match="read-only"):
        result.classes[0] = "mutated"
    with pytest.raises(ValueError, match="read-only"):
        result.gated_p_values[0] = 0.0
