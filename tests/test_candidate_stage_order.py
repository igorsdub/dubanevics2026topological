"""Every Candidate stage runs after the growth rates and the classification.

The Candidate graph is the Functional class of the functional-classification stage,
which reads the growth-rate checkpoint (ADR 2026-10-24-mixture-quantile-functional-
classification): growth rates, then classification, then selection. The ``selection``
recipe is that chain plus the BY classification the shipped configuration reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ("sequence-growth-rates", "functional-classification")
SELECTION = ["sequence-growth-rates", "functional-classification", "classification-probability-scans"]
CANDIDATE_RECIPES = (
    "comparison-analysis",
    "peak-analysis",
    "candidate-edge-analysis",
    "candidate-tree-contrast",
    "bootstrap-smoothed-tree",
    "leading-saddle-band",
    "branch-decomposition",
    "tables",
)


def _recipes() -> dict[str, list[str]]:
    """Recipe name to its prerequisites, from the justfile."""
    recipes: dict[str, list[str]] = {}
    for line in (ROOT / "justfile").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-z][a-z0-9-]*)(?: [^:]*)?:\s*([^=].*)?$", line)
        if match and not line.startswith(("#", " ", "\t")):
            recipes[match.group(1)] = (match.group(2) or "").split()
    return recipes


@pytest.mark.parametrize("recipe", CANDIDATE_RECIPES)
def test_a_candidate_recipe_runs_after_growth_rates_and_classification(
    recipe: str,
) -> None:
    prerequisites = _recipes()[recipe]
    assert prerequisites[0] == "selection" or tuple(prerequisites[:2]) == UPSTREAM


def test_the_selection_recipe_is_the_growth_rates_the_classification_and_the_by_tables() -> None:
    assert _recipes()["selection"] == SELECTION


def test_figures_has_no_compute_prerequisite() -> None:
    assert _recipes()["figures"] == ["check-manifest"]


def test_the_classification_recipe_runs_after_the_growth_rates() -> None:
    assert _recipes()["functional-classification"] == ["sequence-growth-rates"]


def test_every_figure_chain_inherits_the_prerequisites() -> None:
    recipes = _recipes()

    def closure(name: str, seen: set[str] | None = None) -> set[str]:
        seen = set() if seen is None else seen
        for dependency in recipes.get(name, []):
            if dependency not in seen:
                seen.add(dependency)
                closure(dependency, seen)
        return seen

    for recipe in ("all", "figures-svg", "full-tree-figure", "codon-entropy-analysis"):
        assert set(UPSTREAM) <= closure(recipe), recipe
    for recipe in recipes:
        if recipe.endswith("-figure") and recipe.startswith(
            ("comparison", "candidate-histogram", "study", "peak")
        ):
            assert set(UPSTREAM) <= closure(recipe), recipe
