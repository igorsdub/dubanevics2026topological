"""The ``[functional_classification]`` settings table and its validator.

Kept apart from ``functional_classification`` so ``load_settings`` can validate the
table without importing scikit-learn or scipy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SECTION = "functional_classification"
CONFIG_KEYS = (
    "bic_k_range",
    "stop_anchor_min_responsibility",
    "stop_codon_count_for_anchor",
    "quantile",
    "functional_confidence",
    "decimals",
    "random_seed",
    "n_init",
    "max_iter",
    "tol",
)


@dataclass(frozen=True)
class FunctionalClassificationSettings:
    """The validated ``[functional_classification]`` table."""

    k_range: tuple[int, int]
    stop_anchor_min_responsibility: float
    stop_codon_count_for_anchor: int
    quantile: float
    functional_confidence: float
    decimals: int
    random_seed: int
    n_init: int
    max_iter: int
    tol: float


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_config(config: Any) -> FunctionalClassificationSettings:
    """Validate the ``[functional_classification]`` table.

    Raises ``TypeError`` for a table that is not a mapping and ``ValueError`` for an
    unknown or missing key or a value of the wrong type or range.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"[{SECTION}] must be a table, not {config!r}.")
    unknown = sorted(set(config) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown [{SECTION}] settings {unknown}; known: {list(CONFIG_KEYS)}."
        )
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"[{SECTION}] lacks settings {missing}.")

    k_range = config["bic_k_range"]
    if (
        not isinstance(k_range, (list, tuple))
        or len(k_range) != 2
        or not all(_is_int(k) for k in k_range)
        or not 1 <= k_range[0] <= k_range[1]
    ):
        raise ValueError(
            "bic_k_range must be [low, high] integers with 1 <= low <= high, "
            f"got {k_range!r}."
        )
    share = config["stop_anchor_min_responsibility"]
    if not _is_real(share) or not 0 < share < 1:
        raise ValueError(
            f"stop_anchor_min_responsibility must be in (0, 1), got {share!r}."
        )
    anchor = config["stop_codon_count_for_anchor"]
    if not _is_int(anchor) or anchor < 1:
        raise ValueError(
            f"stop_codon_count_for_anchor must be an integer >= 1, got {anchor!r}."
        )
    quantile = config["quantile"]
    if not _is_real(quantile) or not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}.")
    confidence = config["functional_confidence"]
    if not _is_real(confidence) or not 0.5 < confidence < 1:
        raise ValueError(
            f"functional_confidence must be in (0.5, 1), got {confidence!r}."
        )
    decimals = config["decimals"]
    if not _is_int(decimals) or decimals < 0:
        raise ValueError(f"decimals must be a non-negative integer, got {decimals!r}.")
    seed = config["random_seed"]
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"random_seed must be a non-negative integer, got {seed!r}.")
    n_init = config["n_init"]
    if not _is_int(n_init) or n_init < 1:
        raise ValueError(f"n_init must be an integer >= 1, got {n_init!r}.")
    max_iter = config["max_iter"]
    if not _is_int(max_iter) or max_iter < 1:
        raise ValueError(f"max_iter must be an integer >= 1, got {max_iter!r}.")
    tol = config["tol"]
    if not _is_real(tol) or tol <= 0:
        raise ValueError(f"tol must be a number > 0, got {tol!r}.")
    return FunctionalClassificationSettings(
        k_range=(int(k_range[0]), int(k_range[1])),
        stop_anchor_min_responsibility=float(share),
        stop_codon_count_for_anchor=int(anchor),
        quantile=float(quantile),
        functional_confidence=float(confidence),
        decimals=int(decimals),
        random_seed=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
        tol=float(tol),
    )
