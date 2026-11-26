"""Test helpers for the Candidate selection resolver."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from dubanevics2026topological.landscape.selection import CandidateSelection
from dubanevics2026topological.threshold.functional_threshold import CandidateThreshold


def fixed_selection(threshold: float) -> Callable[..., CandidateSelection]:
    """A stand-in for ``resolve_candidate_selection`` under the scalar rule.

    Returns a selection whose threshold record has only ``threshold`` set, so a test
    controls the value the landscape is prepared at without data or checkpoints.
    """

    def resolve(*_: Any, **__: Any) -> CandidateSelection:
        return CandidateSelection(
            "median_reflected",
            SimpleNamespace(threshold=threshold, to_json=dict),  # type: ignore[arg-type]
            None,
            {"rule": "median_reflected"},
        )

    return resolve


# A small classification with every class: ``a`` Non-functional, ``b`` Uncertain, ``c``
# and ``d`` Functional. Tests whose Full graph has these identities use it as is.
MIXTURE_CLASSES = {
    "a": "non_functional",
    "b": "uncertain",
    "c": "functional",
    "d": "functional",
}


def fixed_mixture_selection(
    classes: Mapping[str, str] | None = None, threshold: float = 0.2
) -> Callable[..., CandidateSelection]:
    """A stand-in for ``resolve_candidate_selection`` under ``mixture_quantile``.

    The Functional vertex set is the ``functional`` entries of ``classes`` (default
    ``MIXTURE_CLASSES``), whatever the raw fitness, so a test sees the class rule and
    not the scalar one; no classification checkpoint or data is read. ``threshold`` is
    the reported ``T``.
    """
    table = dict(MIXTURE_CLASSES if classes is None else classes)

    def resolve(*_: Any, **__: Any) -> CandidateSelection:
        return CandidateSelection(
            "mixture_quantile",
            CandidateThreshold(
                rule="mixture_quantile",
                decimals=6,
                median=0.1,
                global_minimum=-1.0,
                median_gap=1.1,
                unrounded=threshold,
                threshold=threshold,
            ),
            frozenset(name for name, label in table.items() if label == "functional"),
            {"rule": "mixture_quantile", "stage": "functional-classification"},
            classes=table,
            uncertain_count=sum(1 for label in table.values() if label == "uncertain"),
        )

    return resolve
