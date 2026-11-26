"""Each bootstrap composite panel must use its own scientific selection."""

from types import SimpleNamespace

import pytest
import polars as pl

from dubanevics2026topological.build.bootstrap_composite_inputs import (
    expected_panel_identity,
    require_checkpoint_identity,
    require_panel_identity,
)


def _panel(rule, value):
    declaration = {"rule": rule, "decimals": 6}
    if rule == "fixed_cutoff":
        declaration["value"] = value
    settings = {"analysis": {"clamped": {"candidate_functional_threshold": declaration}}}
    selection = SimpleNamespace(rule=rule, functional_sequences=None, threshold=SimpleNamespace(threshold=value))
    return settings, selection


def test_matching_panel_identity_is_accepted():
    settings, selection = _panel("fixed_cutoff", -0.5)
    require_panel_identity(expected_panel_identity("published", settings, selection), "published", settings, selection)


def test_different_selection_is_rejected():
    first_settings, first_selection = _panel("fixed_cutoff", -0.5)
    second_settings, second_selection = _panel("fixed_cutoff", -0.4)
    with pytest.raises(ValueError, match="different scientific identity"):
        require_panel_identity(expected_panel_identity("published", first_settings, first_selection), "candidate", second_settings, second_selection)


def test_fixed_cutoff_is_distinct_from_median_reflected():
    fixed_settings, fixed_selection = _panel("fixed_cutoff", -0.5)
    median_settings, median_selection = _panel("median_reflected", -0.5)
    assert expected_panel_identity("same", fixed_settings, fixed_selection) != expected_panel_identity("same", median_settings, median_selection)


def test_same_selection_cannot_be_reused_across_variants():
    settings, selection = _panel("fixed_cutoff", -0.5)
    saved = expected_panel_identity("first", settings, selection)
    with pytest.raises(ValueError, match="different scientific identity"):
        require_panel_identity(saved, "second", settings, selection)


def test_checkpoint_requires_each_draw_and_matching_rule():
    settings, selection = _panel("fixed_cutoff", -0.5)
    settings["analysis"]["bootstrap_smoothed_tree"] = {
        "n_replicates": 2, "seed": 17, "band_quantile": 0.95
    }
    checkpoint = SimpleNamespace(
        summary={
            "fingerprint": {
                "config": {"n_replicates": 2, "seed": 17},
                "candidate_functional_threshold": {"rule": "fixed_cutoff", "decimals": 6, "value": -0.5},
                "candidate_selection": {"rule": "fixed_cutoff", "value": -0.5},
            },
            "settings": {"band_quantile": 0.95},
        },
        draw_stats=pl.DataFrame({"condition": ["real", "real", "flat", "flat"], "draw": [0, 1, 0, 1]}),
    )
    require_checkpoint_identity(checkpoint, settings, selection)
    checkpoint.draw_stats = checkpoint.draw_stats.head(3)
    with pytest.raises(ValueError, match="flat draws"):
        require_checkpoint_identity(checkpoint, settings, selection)