"""The label text helpers print exactly what the inline format specs printed."""

import pytest

from dubanevics2026topological.figures._count_labels import (
    count_label_text,
    fixed_label_text,
    general_label_text,
)


@pytest.mark.parametrize("value", [0.0, 0.1234567, 2.0, 12.34565, 1234567.891])
def test_fixed_and_general_text_match_the_inline_specs(value: float) -> None:
    assert fixed_label_text(value, 3) == f"{value:.3f}"
    assert general_label_text(value) == f"{value:.6g}"


@pytest.mark.parametrize("count", [0, 7, 999, 1000, 1234567])
def test_count_text_matches_the_thousands_spec(count: int) -> None:
    assert count_label_text(count) == f"{count:,}"
