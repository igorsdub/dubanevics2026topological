"""Historical core evidence stays readable without re-running its method."""

from test_bootstrap_smoothed_tree import _decide



def test_historical_hills_without_core_are_nominated():
    hills = _decide([["elder", "e1"]])
    assert hills["mid"]["core_relation"] == hills["other"]["core_relation"] == "none"
    assert hills["mid"]["core_ids"] == []
    assert hills["mid"]["surviving_peak_sequence"] is None
    assert hills["mid"]["split_candidate"] is False
    assert hills["mid"]["outcome"] == hills["other"]["outcome"] == "absorbed"
    assert hills["elder"]["outcome"] == "kept"


def test_historical_core_inside_hill_keeps_it():
    hills = _decide([["elder", "e1"], ["mid"], ["other"]])
    for peak in ("elder", "mid", "other"):
        row = hills[peak]
        assert (row["core_relation"], row["outcome"]) == ("inside", "kept")
        assert row["split_candidate"] is False
        assert row["surviving_peak_sequence"] is None
    assert hills["mid"]["core_ids"] == [1]


