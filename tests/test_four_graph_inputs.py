"""The saved-comparison input of the four-graph figures: scoping and failure messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from dubanevics2026topological import build
from dubanevics2026topological.build import four_graph_inputs
from dubanevics2026topological.figures._four_graphs import GRID_ORDER, PANELS, VARIANTS
from dubanevics2026topological.results import RunContext


def test_the_four_variants_are_the_peak_uncertainty_variants() -> None:
    from dubanevics2026topological.figures.peak_uncertainty import PANEL_VARIANTS

    assert set(VARIANTS) == set(PANEL_VARIANTS)
    assert len(VARIANTS) == len(set(VARIANTS)) == 4


def test_the_grid_is_published_left_candidate_right_nominal_above() -> None:
    assert GRID_ORDER == (("published", "candidate"), ("published_fdr", "candidate_fdr"))
    assert PANELS["published"][1:] == ("Papkou graph", "Nominal")
    assert PANELS["candidate_fdr"][1:] == ("Topographer graph", "BY q = 0.05")


def test_a_variant_context_moves_only_the_output_root(tmp_path: Path) -> None:
    base = RunContext(output_root=tmp_path / "output", root=tmp_path)
    scoped = four_graph_inputs.variant_context(base, "threshold=published_fdr=none")
    assert scoped.output_root == tmp_path / "output" / "variants" / "threshold=published_fdr=none"
    assert scoped.variant_name == "threshold=published_fdr=none"
    assert scoped.root == base.root
    with pytest.raises(ValueError, match="Unknown four-graph variant"):
        four_graph_inputs.variant_context(base, "nope")


def test_a_missing_variant_checkpoint_names_the_command_that_builds_it(
    tmp_path: Path, monkeypatch
) -> None:
    def missing(*, context):
        raise FileNotFoundError("No clamped comparison checkpoint exists")

    monkeypatch.setattr(
        "dubanevics2026topological.analysis.clamped_cache.load_cached_clamped_comparison",
        missing,
    )
    base = RunContext(output_root=tmp_path / "output", root=tmp_path)
    with pytest.raises(ValueError, match="just variant-comparison-analysis threshold="):
        four_graph_inputs.load_four_graph_comparisons(base)


def test_load_inputs_has_a_branch_for_the_four_graph_input(monkeypatch, tmp_path: Path) -> None:
    seen = []
    monkeypatch.setattr(
        four_graph_inputs,
        "load_four_graph_comparisons",
        lambda context: seen.append(context) or {"panels": {}, "fingerprints": {}},
    )
    context = RunContext(output_root=tmp_path, root=tmp_path)
    loaded = build.load_inputs(["four_graph_comparisons"], context=context)
    assert loaded == {"four_graph_comparisons": {"panels": {}, "fingerprints": {}}}
    assert seen == [context]
