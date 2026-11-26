"""The fixed point-estimate rule preserves its declared value and identity."""

import pytest

from dubanevics2026topological.landscape.selection import (
    resolve_selection,
    selected_graph_identity,
)
from dubanevics2026topological.threshold.functional_threshold import (
    declaration_identity,
    validate_functional_threshold_declaration,
)


def _settings(declaration):
    return {"analysis": {"clamped": {"candidate_functional_threshold": declaration}}}


@pytest.mark.parametrize("value", [None, True, "-0.5", float("nan"), float("inf")])
def test_reject_invalid_fixed_cutoff(value):
    with pytest.raises(ValueError, match="finite number"):
        validate_functional_threshold_declaration({"rule": "fixed_cutoff", "decimals": 16, "value": value})


def test_cutoff_identity_and_precision():
    declaration = {"value": -0.5077739708744209, "decimals": 6, "rule": "fixed_cutoff"}
    settings = _settings(declaration)
    selection = resolve_selection(settings, raw_fitness=[-1.0, 0.0])
    assert selection.threshold.threshold == declaration["value"]
    assert selection.functional_sequences is None
    assert declaration_identity(declaration) == {"rule": "fixed_cutoff", "decimals": 6, "value": declaration["value"]}
    assert list(declaration_identity(declaration)) == ["rule", "decimals", "value"]
    assert selected_graph_identity(settings) == selected_graph_identity(settings, selection)
    assert selected_graph_identity(settings) != {"rule": "median_reflected"}
    assert selected_graph_identity(_settings({**declaration, "value": -0.5})) != selected_graph_identity(settings)