"""Functional-threshold production for configured selection rules.

``config/analysis.toml`` declares the Selection rule and the rounding digits under
``[clamped] candidate_functional_threshold``; it never holds the value.

The ``fdr_fixed_cutoff`` rule is the active publication Selection rule (user lock
2026-11-12): a reviewed BY classification snapshot at nominal ``q = 0.05`` selects
the Functional class under a named fixed cutoff (``median_reflected`` or
``papkou_original``). See ``threshold/fdr_selection.py``.

The ``mixture_quantile`` rule remains available (``config/variants/mixture_quantile.toml``):
``T`` is a quantile of the stop-anchored non-functional mixture and comes from the
``functional-classification`` stage. It is not computed from the Full-graph raw
fitness here; ``produce_functional_threshold`` takes the classification's
``summary.json`` for it, and ``landscape/selection.py`` is the one place that loads it.

The ``median_reflected`` point-estimate rule (no FDR) is the comparison rule. It
reflects the median ``m`` through the Median gap ``h = m - g`` (``g`` is the global
minimum): ``T = m + h``, rounded to ``decimals`` places. Under every rule the record
keeps ``m``, ``g`` and ``h`` of the Full graph (the Topological noise band's numbers).

``fixed_cutoff`` uses its declared finite value without rounding or refitting.
For the Papkou overlay this is the 0.999 quantile of the bell-shaped component of
the published three-component Gaussian mixture (see ADR 2026-10-24-mixture-quantile-
functional-classification and classification_probability_scans.py). Bootstrap
draws use the same fixed selection; refitting per draw is a different analysis.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

MEDIAN_REFLECTED = "median_reflected"
FIXED_CUTOFF = "fixed_cutoff"
MIXTURE_QUANTILE = "mixture_quantile"
FDR_FIXED_CUTOFF = "fdr_fixed_cutoff"
RULES = (MEDIAN_REFLECTED, MIXTURE_QUANTILE, FDR_FIXED_CUTOFF, FIXED_CUTOFF)
FDR_KEYS = ("cutoff", "method", "q", "input_dir")
DECLARATION_KEY = "candidate_functional_threshold"


@dataclass(frozen=True)
class FunctionalThreshold:
    """A resolved threshold and the inputs behind it."""

    rule: str
    decimals: int
    median: float
    global_minimum: float
    median_gap: float
    unrounded: float
    threshold: float

    def to_json(self) -> dict[str, Any]:
        """The record written beside the clamped cache for the quantity registry."""
        return {
            "rule": self.rule,
            "decimals": self.decimals,
            "median": self.median,
            "global_minimum": self.global_minimum,
            "median_gap": self.median_gap,
            "unrounded_threshold": self.unrounded,
            "rounded_threshold": self.threshold,
        }


# Compatibility name for existing imports and serialized Python references.
CandidateThreshold = FunctionalThreshold


def comparison_declaration(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Project a threshold declaration onto the median-reflected rule."""
    _, decimals = validate_functional_threshold_declaration(declaration)
    return {"rule": MEDIAN_REFLECTED, "decimals": decimals}


median_reflected_declaration = comparison_declaration


def declaration_identity(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the declaration fields recorded in scientific identities."""
    rule, _ = validate_functional_threshold_declaration(declaration)
    keys = ("rule", "decimals")
    if rule == FDR_FIXED_CUTOFF:
        keys += FDR_KEYS
    elif rule == FIXED_CUTOFF:
        keys += ("value",)
    return {key: declaration[key] for key in keys}


def configured_declaration(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read and validate the configured Candidate threshold declaration."""
    declaration = settings["analysis"]["clamped"][DECLARATION_KEY]
    validate_functional_threshold_declaration(declaration)
    return declaration


def validate_functional_threshold_declaration(declaration: Any) -> tuple[str, int]:
    """Return ``(rule, decimals)`` from the threshold configuration declaration.

    Raises ``TypeError`` for a declaration that is not a table and ``ValueError``
    for an unknown key, rule or invalid digits.
    """
    if not isinstance(declaration, Mapping):
        raise TypeError(
            "[clamped] candidate_functional_threshold must be a table "
            '{ rule = "median_reflected" | "mixture_quantile" | "fdr_fixed_cutoff", '
            "decimals = <int>[, FDR keys…] }, "
            f"not {declaration!r}; the value is computed, never typed."
        )
    allowed = {"rule", "decimals"}
    if declaration.get("rule") == FDR_FIXED_CUTOFF:
        allowed |= set(FDR_KEYS)
    if declaration.get("rule") == FIXED_CUTOFF:
        allowed.add("value")
    unknown = set(declaration) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in candidate_functional_threshold: {sorted(unknown)}."
        )
    rule = declaration.get("rule")
    if rule not in RULES:
        raise ValueError(
            f"Unknown Candidate functional threshold rule {rule!r}; "
            f"expected one of {list(RULES)}."
        )
    if rule == FDR_FIXED_CUTOFF:
        missing = set(FDR_KEYS) - set(declaration)
        if missing:
            raise ValueError(f"Missing FDR keys in candidate_functional_threshold: {sorted(missing)}.")
    if rule == FIXED_CUTOFF:
        value = declaration.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("fixed_cutoff value must be a finite number.")
    decimals = declaration.get("decimals")
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError(
            f"candidate_functional_threshold decimals must be a non-negative integer, got {decimals!r}."
        )
    return str(rule), int(decimals)


def produce_functional_threshold(
    raw_fitness: Iterable[float],
    declaration: Mapping[str, Any],
    *,
    classification_summary: Mapping[str, Any] | None = None,
) -> FunctionalThreshold:
    """Produce a threshold from the existing configured threshold rule."""
    return compute_candidate_functional_threshold(
        raw_fitness, declaration, classification_summary=classification_summary
    )


def validate_candidate_declaration(declaration: Any) -> tuple[str, int]:
    """Validate the legacy-named declaration and preserve its former error text."""
    try:
        return validate_functional_threshold_declaration(declaration)
    except ValueError as error:
        if str(error).startswith("Unknown functional threshold rule"):
            rule = declaration.get("rule") if isinstance(declaration, Mapping) else None
            raise ValueError(
                f"Unknown Candidate functional threshold rule {rule!r}; "
                f"expected one of {list(RULES)}."
            ) from None
        raise


def compute_functional_threshold(
    raw_fitness: Iterable[float],
    declaration: Mapping[str, Any],
    *,
    classification_summary: Mapping[str, Any] | None = None,
) -> FunctionalThreshold:
    """Compute the configured functional threshold from Full-graph fitness.

    ``declaration`` is the ``[clamped] candidate_functional_threshold`` table.
    The ``median_reflected`` value is computed from the Full-graph raw fitness. The
    ``mixture_quantile`` value is read from ``classification_summary`` (the
    ``summary.json`` of the functional-classification checkpoint), and the record
    keeps the Full-graph ``m``, ``g`` and ``h`` beside it. Without the summary the
    mixture rule raises, since the raw fitness alone cannot give it.
    """
    rule, decimals = validate_functional_threshold_declaration(declaration)
    if rule in (MIXTURE_QUANTILE, FDR_FIXED_CUTOFF) and classification_summary is None:
        raise ValueError(
            f"The {rule} threshold is not computed from the Full-graph raw fitness; "
            "pass the classification summary (`classification_summary=`) or resolve "
            "it through `resolve_candidate_selection` "
            "(`just functional-classification` for mixture_quantile; the FDR "
            "snapshot for fdr_fixed_cutoff)."
        )
    values = [float(value) for value in raw_fitness]
    if not values:
        raise ValueError("The Full graph has no raw fitness values.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Full-graph raw fitness values must be finite.")
    median = float(statistics.median(values))
    global_minimum = float(min(values))
    median_gap = median - global_minimum
    if rule == MEDIAN_REFLECTED:
        unrounded = median + median_gap
        threshold = round(unrounded, decimals)
    elif rule == FIXED_CUTOFF:
        unrounded = threshold = float(declaration["value"])
    else:
        record = classification_summary["threshold"]  # type: ignore[index]
        unrounded = float(record["unrounded"])
        threshold = float(record["rounded"])
        # The FDR rule's applied T is authenticated by the snapshot provenance and is
        # used exactly as produced (Papkou: exact formula value, median: 6 decimals).
        if rule != FDR_FIXED_CUTOFF and round(unrounded, decimals) != threshold:
            raise ValueError(
                f"The functional classification rounds T to {threshold!r}, but "
                f"[clamped] candidate_functional_threshold decimals = {decimals} "
                f"rounds {unrounded!r} differently; make the two decimals agree "
                "and rerun `just functional-classification`."
            )
    return FunctionalThreshold(
        rule=rule,
        decimals=decimals,
        median=median,
        global_minimum=global_minimum,
        median_gap=median_gap,
        unrounded=unrounded,
        threshold=threshold,
    )


compute_candidate_functional_threshold = compute_functional_threshold


def network_raw_fitness(full_network: Any) -> list[float]:
    """Raw fitness of every vertex of a Topographer ``ScalarNetwork``."""
    graph = full_network.graph
    scalar_attr = full_network.scalar_attr
    return [float(graph[index][scalar_attr]) for index in graph.node_indices()]


def resolve_candidate_threshold(
    full_network: Any, settings: Mapping[str, Any]
) -> CandidateThreshold:
    """The configured functional threshold for a Full network and settings.

    Delegates to ``resolve_candidate_selection``, so the active rule decides.
    """
    from dubanevics2026topological.landscape.selection import (
        resolve_candidate_selection,
    )

    return resolve_candidate_selection(settings, full_network=full_network).threshold


resolve_functional_threshold = resolve_candidate_threshold


def rounded_candidate_threshold(settings: Mapping[str, Any]) -> float:
    """Return the configured rounded functional threshold of the Full graph.

    Delegates to ``resolve_candidate_selection``. Under ``mixture_quantile`` it
    reads only the classification checkpoint; under ``median_reflected`` it reads
    the Full-graph raw fitness (no graph or topology work).
    """
    from dubanevics2026topological.landscape.selection import (
        candidate_selection_fingerprint,
    )

    return candidate_selection_fingerprint(settings)["rounded_threshold"]


rounded_functional_threshold = rounded_candidate_threshold


__all__ = [
    "MEDIAN_REFLECTED",
    "MIXTURE_QUANTILE",
    "CandidateThreshold",
    "FunctionalThreshold",
    "comparison_declaration",
    "compute_candidate_functional_threshold",
    "compute_functional_threshold",
    "configured_declaration",
    "declaration_identity",
    "median_reflected_declaration",
    "network_raw_fitness",
    "produce_functional_threshold",
    "resolve_candidate_threshold",
    "resolve_functional_threshold",
    "rounded_candidate_threshold",
    "rounded_functional_threshold",
    "validate_candidate_declaration",
    "validate_functional_threshold_declaration",
]
