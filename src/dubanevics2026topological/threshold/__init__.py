"""Threshold production and sequence classification interface.

Threshold implementations and their public interfaces live in this package.
"""

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from dubanevics2026topological.threshold.classification_probability_scans import (
    ThresholdDefinition,
    get_active_threshold_definitions,
    harmonic_number,
)
from dubanevics2026topological.threshold.classification_probability_scans import (
    adjust_p_values_fdr as _adjust_p_values,
)
from dubanevics2026topological.threshold.classification_probability_scans import (
    assign_fdr_classes as _assign_classes,
)
from dubanevics2026topological.threshold.classification_probability_scans import (
    compute_one_sided_gated_p_values as _gated_p_values,
)
from dubanevics2026topological.threshold.fdr_selection import (
    FdrSnapshot,
    load_fdr_snapshot,
    require_full_population,
    threshold_summary,
)
from dubanevics2026topological.threshold.functional_classification import (
    classify_growth_rates as classify_mixture_sequences,
)
from dubanevics2026topological.threshold.functional_classification import (
    compute_functional_classification as compute_mixture_threshold,
)
from dubanevics2026topological.threshold.functional_threshold import (
    FunctionalThreshold,
    produce_functional_threshold,
    resolve_functional_threshold,
)

# Domain-first names for new callers. Historical names remain in forwarding modules.
compute_threshold_gated_p_values = _gated_p_values
adjust_threshold_p_values = _adjust_p_values
assign_threshold_classes = _assign_classes


@dataclass(frozen=True)
class ThresholdClassification:
    """Sequence classifications and statistical evidence for one threshold."""

    threshold: float
    family_size: int
    sequence_ids: tuple[str, ...]
    gated_p_values: np.ndarray
    q_values: np.ndarray
    classes: np.ndarray

    def __post_init__(self) -> None:
        for field in ("gated_p_values", "q_values", "classes"):
            value = np.array(getattr(self, field), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, field, value)


def classify_functional_sequences(
    s: object,
    se: object,
    threshold: float,
    *,
    sequence_ids: Collection[str],
    library_ids: Collection[str],
    nominal_q: float = 0.05,
) -> ThresholdClassification:
    """Classify a complete Full-library family using fixed-threshold BY.

    ``sequence_ids`` must exactly cover the declared ``library_ids`` family.
    Classification is performed before any graph or component selection.
    """
    observed_ids = list(sequence_ids)
    declared_ids = list(library_ids)
    if any(not isinstance(item, str) or not item for item in observed_ids):
        raise ValueError("sequence_ids must contain non-empty sequence identities.")
    if any(not isinstance(item, str) or not item for item in declared_ids):
        raise ValueError("library_ids must contain non-empty sequence identities.")
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("sequence_ids contains duplicate identities.")
    if len(set(declared_ids)) != len(declared_ids):
        raise ValueError("library_ids contains duplicate identities.")
    observed_set = set(observed_ids)
    declared_set = set(declared_ids)
    missing = declared_set - observed_set
    extra = observed_set - declared_set
    if missing or extra:
        raise ValueError(
            "Classification rows must exactly cover the declared Full-library "
            f"family; missing={len(missing)}, extra={len(extra)}."
        )

    above, _, _, _, gated, log_gated = _gated_p_values(
        s, se, threshold, sequence_ids=observed_ids
    )
    if len(gated) != len(declared_ids):
        raise ValueError(
            "Fitness and standard-error rows do not match sequence_ids length."
        )
    adjusted, log_adjusted = _adjust_p_values(log_gated, method="by")
    classes, _, _, _ = _assign_classes(above, log_adjusted, nominal_q)
    return ThresholdClassification(
        threshold=float(threshold),
        family_size=len(declared_ids),
        sequence_ids=tuple(observed_ids),
        gated_p_values=gated,
        q_values=adjusted,
        classes=classes,
    )


__all__ = [
    "FdrSnapshot",
    "FunctionalThreshold",
    "ThresholdClassification",
    "ThresholdDefinition",
    "adjust_threshold_p_values",
    "assign_threshold_classes",
    "classify_functional_sequences",
    "classify_mixture_sequences",
    "compute_mixture_threshold",
    "compute_threshold_gated_p_values",
    "get_active_threshold_definitions",
    "harmonic_number",
    "load_fdr_snapshot",
    "produce_functional_threshold",
    "require_full_population",
    "resolve_functional_threshold",
    "threshold_summary",
]
