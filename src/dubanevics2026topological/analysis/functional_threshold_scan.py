"""Sensitivity scan of the functional threshold (ADR 2026-10-24, Decisions 7 and 8).

The classification stage fixes one rule: a Gaussian mixture, ``k`` by BIC, the
stop-anchored non-functional components, the 0.999 quantile, confidence 0.95. This
module reports how far the classification moves when each of those choices changes one
step. The scan describes sensitivity; it is never used to pick the rule, and a failed
acceptance check is reported, never re-tuned.

Tables (all keyed by cell, none by sequence; written to ``SCAN_DIR``):

- ``cells.parquet`` (:func:`point_scan`): one row per (family, k, component rule,
  quantile, confidence) on the real data.
- ``comparator.parquet`` (:func:`comparator_cells`): model-free empirical quantiles of the
  double- and triple-stop sets.
- ``resamples.parquet`` (:func:`resample_thresholds`): ``T`` of every (family, k) in each
  nonparametric resample (sequences drawn with replacement) at the stop-anchored rule
  and the classification's quantile.
- ``intervals.parquet`` (:func:`interval_table`): the 2.5% and 97.5% quantiles of those.
- ``acceptance.json`` (:func:`acceptance_checks`): checks (a) to (c) of Decision 8.
- ``provenance.json``: written last, so it also marks a finished scan. It records the
  settings, the sha256 of this module and ``mixture_families`` (a code change makes a
  scan stale), the classification checkpoint's identity, the growth table's sha256, the
  library versions, the git stamp and the sha256 of every table. The loader
  (:func:`load_functional_threshold_scan`) refuses tables whose provenance differs from
  the current settings or classification checkpoint.

Seeds. Resample ``r`` draws its sequences with ``default_rng(random_seed + r)`` and
fits with scikit-learn seed ``random_seed + r``, so any process computing index ``r``
gets the same draw and fits: a Slurm array task computes a sub-range
(:func:`compute_resample_slice`) and a merge job combines the slices
(:func:`merge_resample_slices`, then :func:`compute_functional_threshold_scan`). The
point scan fits with the classification's ``random_seed`` (and the scan's ``n_init``,
``max_iter`` and ``tol``, which must equal the classification's for the canonical cell
to reproduce the stage; :func:`point_scan` fails when its Gaussian cell does not).

Definitions fixed here because the ADR leaves them open:

- The threshold used to classify is rounded to the classification's ``decimals`` like the
  stage's; ``threshold`` in the tables is that rounded value.
- "One scan step" from the canonical cell, other axes held at canonical: the family axis
  is each other family at its own BIC-chosen ``k``; the ``k`` axis is the canonical
  family at ``k`` +/- 1 around its BIC-chosen ``k``; the quantile and confidence axes
  are the neighbouring entries of the sorted configured lists; the component rule axis
  is the other rule.
- A resampled interval uses the fits that converged; ``n_converged`` and ``n_resamples``
  are reported beside it.
- Resample rows also export BIC and selected component identities. The original-data
  ``is_bic_chosen`` remains unchanged; ``is_resample_bic_chosen`` marks a new BIC
  choice within that resample. ``model_selection_complete`` requires every candidate
  k of its family to converge with finite BIC. These additional fields describe
  selection stability, not biological sampling coverage or a replacement interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from ..threshold import functional_classification as fc
from ..threshold.functional_classification_config import (
    FunctionalClassificationSettings,
)
from ..threshold.functional_classification_config import (
    validate_config as validate_classification,
)
from . import OUTPUT, ROOT, _write_json_atomic
from . import mixture_families as mf
from .git_stamp import tag

SECTION = "functional_threshold_scan"
SCAN_DIR = OUTPUT / "analysis" / "functional_threshold_scan"
CELLS_NAME = "cells.parquet"
COMPARATOR_NAME = "comparator.parquet"
RESAMPLES_NAME = "resamples.parquet"
INTERVALS_NAME = "intervals.parquet"
ACCEPTANCE_NAME = "acceptance.json"
PROVENANCE_NAME = "provenance.json"
SLICE_NAME = "slice.json"
TABLE_NAMES = (CELLS_NAME, COMPARATOR_NAME, RESAMPLES_NAME, INTERVALS_NAME)
ARTIFACT_NAMES = (*TABLE_NAMES, ACCEPTANCE_NAME)
SCHEMA = 2
RETRY_HINT = "just functional-threshold-scan"

STOP_ANCHORED = "stop_anchored"
LOWEST_MEAN = "lowest_mean"
COMPONENT_RULES = (STOP_ANCHORED, LOWEST_MEAN)
COMPARATOR_QUANTILES = (0.99, 0.999)
COMPARATOR_STOP_COUNTS = (2, 3)
INTERVAL_LOW = 0.025
INTERVAL_HIGH = 0.975
# 2 * 1.96 standard errors: the width of a 95% interval of one measurement.
WIDTH_FACTOR = 2.0 * 1.96

CONFIG_KEYS = (
    "families",
    "k_range",
    "quantiles",
    "confidences",
    "component_rules",
    "n_resamples",
    "random_seed",
    "n_init",
    "max_iter",
    "tol",
    "flip_fraction_limit",
    "double_stop_below_min",
)
# Scheduling keys: they change how the resamples are computed, never what they are, so
# they are optional, checked here and left out of every recorded identity (ADR 0013's
# ``EXECUTION_KEYS`` pattern; ``stages.EXECUTION_KEYS`` lists the same name).
EXECUTION_KEYS = ("n_jobs",)
DEFAULT_N_JOBS = 8
ENV_N_JOBS = "FITNESSTREE_N_JOBS"

CELL_SCHEMA = {
    "family": pl.String,
    "k": pl.Int64,
    "bic_chosen": pl.Int64,
    "component_rule": pl.String,
    "quantile": pl.Float64,
    "confidence": pl.Float64,
    "threshold": pl.Float64,
    "bic": pl.Float64,
    "converged": pl.Boolean,
    "selected_components": pl.List(pl.Int64),
    "frac_1stop_below": pl.Float64,
    "frac_2stop_below": pl.Float64,
    "frac_3stop_below": pl.Float64,
    "functional_count": pl.Int64,
    "non_functional_count": pl.Int64,
    "uncertain_count": pl.Int64,
    "functional_flips": pl.Int64,
    "functional_jaccard": pl.Float64,
    "is_canonical": pl.Boolean,
}
COMPARATOR_SCHEMA = {
    "stop_count": pl.Int64,
    "quantile": pl.Float64,
    "confidence": pl.Float64,
    "threshold": pl.Float64,
    "n_sequences": pl.Int64,
    "frac_1stop_below": pl.Float64,
    "frac_2stop_below": pl.Float64,
    "frac_3stop_below": pl.Float64,
    "functional_count": pl.Int64,
    "non_functional_count": pl.Int64,
    "uncertain_count": pl.Int64,
    "functional_flips": pl.Int64,
    "functional_jaccard": pl.Float64,
}
RESAMPLE_SCHEMA = {
    "resample": pl.Int64,
    "family": pl.String,
    "k": pl.Int64,
    "threshold": pl.Float64,
    "converged": pl.Boolean,
    "is_bic_chosen": pl.Boolean,
    "bic": pl.Float64,
    "selected_components": pl.List(pl.Int64),
    "is_resample_bic_chosen": pl.Boolean,
    "model_selection_complete": pl.Boolean,
}
INTERVAL_SCHEMA = {
    "family": pl.String,
    "k": pl.Int64,
    "threshold_low": pl.Float64,
    "threshold_high": pl.Float64,
    "width": pl.Float64,
    "n_converged": pl.Int64,
    "n_resamples": pl.Int64,
}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalThresholdScanSettings:
    """The validated ``[functional_threshold_scan]`` table."""

    families: tuple[str, ...]
    k_range: tuple[int, int]
    quantiles: tuple[float, ...]
    confidences: tuple[float, ...]
    component_rules: tuple[str, ...]
    n_resamples: int
    random_seed: int
    n_init: int
    max_iter: int
    tol: float
    flip_fraction_limit: float
    double_stop_below_min: float

    @property
    def ks(self) -> range:
        return range(self.k_range[0], self.k_range[1] + 1)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _names(
    config: Mapping[str, Any], key: str, known: Sequence[str]
) -> tuple[str, ...]:
    value = config[key]
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(item, str) for item in value)
        or len(set(value)) != len(value)
        or not set(value) <= set(known)
    ):
        raise ValueError(
            f"{key} must be a non-empty list of distinct names from {list(known)}, "
            f"got {value!r}."
        )
    return tuple(value)


def _levels(config: Mapping[str, Any], key: str, low: float, high: float) -> tuple:
    value = config[key]
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(_is_real(item) and low < item < high for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{key} must be a non-empty list of distinct numbers in ({low}, {high}), "
            f"got {value!r}."
        )
    return tuple(float(item) for item in value)


def validate_config(config: Any) -> FunctionalThresholdScanSettings:
    """Validate the ``[functional_threshold_scan]`` table.

    ``TypeError`` for a table that is not a mapping, ``ValueError`` for an unknown or
    missing key or a value of the wrong type or range.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"[{SECTION}] must be a table, not {config!r}.")
    unknown = sorted(set(config) - set(CONFIG_KEYS) - set(EXECUTION_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown [{SECTION}] settings {unknown}; known: "
            f"{[*CONFIG_KEYS, *EXECUTION_KEYS]}."
        )
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"[{SECTION}] lacks settings {missing}.")
    if "n_jobs" in config and (not _is_int(config["n_jobs"]) or config["n_jobs"] < 1):
        raise ValueError(f"n_jobs must be an integer >= 1, got {config['n_jobs']!r}.")
    families = _names(config, "families", mf.FAMILY_NAMES)
    rules = _names(config, "component_rules", COMPONENT_RULES)
    k_range = config["k_range"]
    if (
        not isinstance(k_range, (list, tuple))
        or len(k_range) != 2
        or not all(_is_int(k) for k in k_range)
        or not 1 <= k_range[0] <= k_range[1]
    ):
        raise ValueError(
            f"k_range must be [low, high] integers with 1 <= low <= high, got {k_range!r}."
        )
    quantiles = _levels(config, "quantiles", 0.0, 1.0)
    confidences = _levels(config, "confidences", 0.5, 1.0)
    n_resamples = config["n_resamples"]
    if not _is_int(n_resamples) or n_resamples < 1:
        raise ValueError(f"n_resamples must be an integer >= 1, got {n_resamples!r}.")
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
    limit = config["flip_fraction_limit"]
    if not _is_real(limit) or not 0 < limit <= 1:
        raise ValueError(f"flip_fraction_limit must be in (0, 1], got {limit!r}.")
    minimum = config["double_stop_below_min"]
    if not _is_real(minimum) or not 0 < minimum <= 1:
        raise ValueError(f"double_stop_below_min must be in (0, 1], got {minimum!r}.")
    return FunctionalThresholdScanSettings(
        families=families,
        k_range=(int(k_range[0]), int(k_range[1])),
        quantiles=quantiles,
        confidences=confidences,
        component_rules=rules,
        n_resamples=int(n_resamples),
        random_seed=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
        tol=float(tol),
        flip_fraction_limit=float(limit),
        double_stop_below_min=float(minimum),
    )


def _as_settings(config: Any) -> FunctionalThresholdScanSettings:
    return (
        config
        if isinstance(config, FunctionalThresholdScanSettings)
        else validate_config(config)
    )


class ScanSettingsMismatch(ValueError):
    """The scan's ``n_init``, ``max_iter`` or ``tol`` differ from the classification's."""


MATCHED_KEYS = ("n_init", "max_iter", "tol")


def require_matching_settings(
    config: Any, classification_config: Any
) -> tuple[FunctionalThresholdScanSettings, FunctionalClassificationSettings]:
    """Validate both tables and require equal ``n_init``, ``max_iter`` and ``tol``.

    The scan's Gaussian cell reproduces the stage's only with the stage's fit
    settings. Called before any fit; :func:`point_scan` keeps its own check of the
    resulting threshold as a second line. Raises :class:`ScanSettingsMismatch`.
    """
    scan = _as_settings(config)
    cc = _as_classification(classification_config)
    differ = {
        key: (getattr(scan, key), getattr(cc, key))
        for key in MATCHED_KEYS
        if getattr(scan, key) != getattr(cc, key)
    }
    if differ:
        detail = "; ".join(
            f"{key}: scan {a!r} against classification {b!r}"
            for key, (a, b) in differ.items()
        )
        raise ScanSettingsMismatch(
            f"[{SECTION}] must equal [{fc.SECTION}] on {list(MATCHED_KEYS)} ({detail})."
        )
    return scan, cc


def _as_classification(config: Any) -> FunctionalClassificationSettings:
    return (
        config
        if isinstance(config, FunctionalClassificationSettings)
        else validate_classification(config)
    )


# --------------------------------------------------------------------------
# Inputs and shared cell arithmetic
# --------------------------------------------------------------------------

_REQUIRED = ("sequence_id", "s", "se_pooled", "stop_codon_count")


@dataclass(frozen=True)
class _Inputs:
    ids: np.ndarray
    s: np.ndarray
    se: np.ndarray
    stops: np.ndarray


def _inputs(growth: pl.DataFrame) -> _Inputs:
    absent = [name for name in _REQUIRED if name not in growth.columns]
    if absent:
        raise ValueError(f"The growth-rate table lacks columns {absent}.")
    if growth.is_empty():
        raise ValueError("The growth-rate table is empty.")
    s = growth["s"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    if not np.isfinite(s).all():
        raise ValueError("s must be finite for every sequence.")
    # The stage's se rule: fail closed on a null, non-finite or non-positive se.
    se_list = [None if v is None else float(v) for v in growth["se_pooled"]]
    fc.functional_probability(s, se_list, 0.0, growth["sequence_id"].to_list())
    return _Inputs(
        growth["sequence_id"].to_numpy(),
        s,
        np.array(se_list, dtype=float),
        growth["stop_codon_count"].to_numpy(),
    )


def _canonical_mask(data: _Inputs, canonical: fc.FunctionalClassificationResult):
    """The canonical Functional set as a boolean mask in the growth table's order."""
    table = canonical.classification
    if table.height != data.ids.size:
        raise ValueError(
            "The classification and the growth-rate table have different sequence "
            f"counts ({table.height} against {data.ids.size})."
        )
    frame = pl.DataFrame({"sequence_id": data.ids}).join(
        table.select("sequence_id", "functional_class"), on="sequence_id", how="left"
    )
    if frame["functional_class"].null_count():
        raise ValueError(
            "The classification does not cover every sequence of the growth-rate table."
        )
    return (frame["functional_class"] == fc.FUNCTIONAL).to_numpy()


def _flips(functional: np.ndarray, canonical: np.ndarray) -> tuple[int, float]:
    both = int((functional & canonical).sum())
    either = int((functional | canonical).sum())
    return int((functional ^ canonical).sum()), (both / either if either else 1.0)


def _fraction(data: _Inputs, threshold: float, count: int) -> float | None:
    return fc.stop_fractions(data.s, data.stops, threshold)[str(count)]["fraction"]


def _class_columns(
    data: _Inputs,
    threshold: float,
    confidence: float,
    canonical: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    classes = fc.assign_class(probability, confidence)
    functional = classes == fc.FUNCTIONAL
    flips, jaccard = _flips(functional, canonical)
    return {
        "functional_count": int(functional.sum()),
        "non_functional_count": int((classes == fc.NON_FUNCTIONAL).sum()),
        "uncertain_count": int((classes == fc.UNCERTAIN).sum()),
        "functional_flips": flips,
        "functional_jaccard": jaccard,
    }


def _stop_columns(data: _Inputs, threshold: float) -> dict[str, Any]:
    return {
        f"frac_{count}stop_below": _fraction(data, threshold, count)
        for count in fc.STOP_COUNTS_REPORTED
    }


def _select(
    fit: mf.FamilyFit,
    s: np.ndarray,
    stops: np.ndarray,
    rule: str,
    cc: FunctionalClassificationSettings,
) -> np.ndarray:
    """Boolean mask of the non-functional components under ``rule``."""
    if rule == LOWEST_MEAN:
        mask = np.zeros(fit.k, dtype=bool)
        mask[int(np.argmin(fit.component_means))] = True
        return mask
    anchored = stops == cc.stop_codon_count_for_anchor
    if not anchored.any():
        raise ValueError(
            f"No sequence has {cc.stop_codon_count_for_anchor} stop codons, so no "
            "mixture component can be anchored to the non-functional distribution."
        )
    shares = fit.responsibilities(s[anchored]).mean(axis=0)
    return fc.select_non_functional(shares, cc.stop_anchor_min_responsibility)


def _fit_all(
    values: np.ndarray,
    scan: FunctionalThresholdScanSettings,
    seed: int,
) -> dict[tuple[str, int], mf.FamilyFit]:
    """One fit per (family, k); the non-Gaussian EM starts from the Gaussian fit."""
    fits: dict[tuple[str, int], mf.FamilyFit] = {}
    for k in scan.ks:
        gaussian = mf.get_family(mf.GAUSSIAN).fit(
            values,
            k,
            seed=seed,
            n_init=scan.n_init,
            max_iter=scan.max_iter,
            tol=scan.tol,
        )
        for name in scan.families:
            fits[(name, k)] = (
                gaussian
                if name == mf.GAUSSIAN
                else mf.get_family(name).fit(
                    values,
                    k,
                    seed=seed,
                    n_init=scan.n_init,
                    max_iter=scan.max_iter,
                    tol=scan.tol,
                    init=gaussian,
                )
            )
    return fits


def _bic_chosen(
    fits: Mapping[tuple[str, int], mf.FamilyFit], families: Iterable[str]
) -> dict[str, int]:
    return {
        name: mf.choose_by_bic([fit for (f, _), fit in fits.items() if f == name]).k
        for name in families
    }


# --------------------------------------------------------------------------
# Point scan and comparator
# --------------------------------------------------------------------------


def point_scan(
    growth: pl.DataFrame,
    config: Any,
    classification_config: Any,
    canonical: fc.FunctionalClassificationResult,
) -> pl.DataFrame:
    """One row per (family, k, component rule, quantile, confidence) on the real data.

    One fit per (family, k) is reused across the rules, quantiles and confidences.
    ``canonical`` is the classification stage's result; ``functional_flips`` and
    ``functional_jaccard`` compare each cell's Functional set with its Functional class.
    ``is_canonical`` marks the one cell equal to the stage's settings (Gaussian, the
    BIC-chosen ``k``, stop-anchored, the stage's quantile and confidence). The scan
    must contain it, and its rounded threshold must equal the stage's, or
    ``ValueError`` (the scan's ``n_init``, ``max_iter`` and ``tol`` then differ from
    the classification's).
    """
    scan, cc = require_matching_settings(config, classification_config)
    data = _inputs(growth)
    canon = _canonical_mask(data, canonical)
    canonical_k = int(canonical.summary["mixture_k"])
    fits = _fit_all(data.s, scan, cc.random_seed)
    chosen = _bic_chosen(fits, scan.families)
    rows: list[dict[str, Any]] = []
    for (name, k), fit in fits.items():
        for rule in scan.component_rules:
            mask = _select(fit, data.s, data.stops, rule, cc)
            selected = [int(j) for j in np.flatnonzero(mask)]
            for quantile in scan.quantiles:
                threshold = round(fit.union_quantile(mask, quantile), cc.decimals)
                probability = norm.cdf((data.s - threshold) / data.se)
                stops = _stop_columns(data, threshold)
                for confidence in scan.confidences:
                    rows.append(
                        {
                            "family": name,
                            "k": k,
                            "bic_chosen": chosen[name],
                            "component_rule": rule,
                            "quantile": quantile,
                            "confidence": confidence,
                            "threshold": threshold,
                            "bic": fit.bic,
                            "converged": fit.converged,
                            "selected_components": selected,
                            **stops,
                            **_class_columns(
                                data, threshold, confidence, canon, probability
                            ),
                            "is_canonical": (
                                name == mf.GAUSSIAN
                                and k == canonical_k
                                and rule == STOP_ANCHORED
                                and quantile == cc.quantile
                                and confidence == cc.functional_confidence
                            ),
                        }
                    )
    cells = pl.DataFrame(rows, schema=CELL_SCHEMA)
    marked = cells.filter(pl.col("is_canonical"))
    if marked.height != 1:
        raise ValueError(
            "The scan has no canonical cell: it needs the gaussian family, "
            f"k = {canonical_k}, the stop_anchored rule, quantile {cc.quantile} and "
            f"confidence {cc.functional_confidence} (found {marked.height})."
        )
    stage_threshold = float(canonical.summary["threshold"]["rounded"])
    if marked["threshold"][0] != stage_threshold or marked["functional_flips"][0]:
        raise ValueError(
            "The scan's Gaussian cell does not reproduce the classification "
            f"(threshold {marked['threshold'][0]} against {stage_threshold}, "
            f"{marked['functional_flips'][0]} flips); the scan's n_init, max_iter and "
            "tol must equal [functional_classification]'s."
        )
    return cells


def comparator_cells(
    growth: pl.DataFrame,
    quantiles: Sequence[float] = COMPARATOR_QUANTILES,
    *,
    canonical: fc.FunctionalClassificationResult,
    classification_config: Any,
    stop_counts: Sequence[int] = COMPARATOR_STOP_COUNTS,
) -> pl.DataFrame:
    """Model-free comparator cells: empirical quantiles of the double- and triple-stop sets.

    One row per (stop count, quantile): ``threshold`` is ``numpy.quantile`` (linear
    interpolation) of ``s`` over the sequences with that many stop codons, rounded
    like the stage's. The class counts use the classification's confidence level and
    flips and Jaccard are against ``canonical``. ``ValueError`` when no sequence has a
    requested stop count.
    """
    cc = _as_classification(classification_config)
    data = _inputs(growth)
    canon = _canonical_mask(data, canonical)
    rows = []
    for count in stop_counts:
        members = data.s[data.stops == count]
        if members.size == 0:
            raise ValueError(f"No sequence has {count} stop codons for the comparator.")
        for quantile in quantiles:
            threshold = round(float(np.quantile(members, quantile)), cc.decimals)
            probability = norm.cdf((data.s - threshold) / data.se)
            rows.append(
                {
                    "stop_count": int(count),
                    "quantile": float(quantile),
                    "confidence": cc.functional_confidence,
                    "threshold": threshold,
                    "n_sequences": int(members.size),
                    **_stop_columns(data, threshold),
                    **_class_columns(
                        data, threshold, cc.functional_confidence, canon, probability
                    ),
                }
            )
    return pl.DataFrame(rows, schema=COMPARATOR_SCHEMA)


# --------------------------------------------------------------------------
# Resamples and intervals
# --------------------------------------------------------------------------


def resample_thresholds(
    growth: pl.DataFrame,
    config: Any,
    resample_indices: range,
    canonical_k_by_family: Mapping[str, int] | None = None,
    *,
    classification_config: Any,
) -> pl.DataFrame:
    """``T`` of every (family, k) in each resample of ``resample_indices``.

    Resample ``r`` draws ``n`` sequences with replacement using
    ``default_rng(random_seed + r)`` and fits with scikit-learn seed
    ``random_seed + r``, so the result for ``r`` does not depend on which range or
    process computed it. ``T`` is the stop-anchored rule at the classification's
    quantile, rounded like the stage's. ``canonical_k_by_family`` (the BIC-chosen ``k``
    per family from the point scan) only fills ``is_bic_chosen``, null when not given.
    ``classification_config`` supplies the anchor settings and the quantile. A fit that
    fails raises; one that has not converged is recorded with ``converged`` false.
    """
    scan, cc = require_matching_settings(config, classification_config)
    if canonical_k_by_family is not None:
        _check_known_families(canonical_k_by_family, scan)
    data = _inputs(growth)
    return _resample_rows(
        data.s, data.stops, scan, cc, resample_indices, canonical_k_by_family
    )


def _check_known_families(
    canonical_k_by_family: Mapping[str, int], scan: FunctionalThresholdScanSettings
) -> None:
    unknown = sorted(set(canonical_k_by_family) - set(scan.families))
    if unknown:
        raise ValueError(f"canonical_k_by_family names unknown families {unknown}.")


def _resample_rows(
    s: np.ndarray,
    stops: np.ndarray,
    scan: FunctionalThresholdScanSettings,
    cc: FunctionalClassificationSettings,
    resample_indices: range,
    canonical_k_by_family: Mapping[str, int] | None,
) -> pl.DataFrame:
    """The resamples of ``resample_indices`` from the arrays ``s`` and ``stops`` alone.

    What a ``joblib`` task receives: the growth frame's other columns (sequence ids,
    standard errors, DNA) are not needed to fit and select, so they are not sent.
    """
    n = s.size
    rows: list[dict[str, Any]] = []
    for r in resample_indices:
        draw = np.random.default_rng(scan.random_seed + r).integers(0, n, size=n)
        sample_s, sample_stops = s[draw], stops[draw]
        fits = _fit_all(sample_s, scan, scan.random_seed + r)
        # Select among every fitted k, including unconverged competitors: dropping
        # one silently changes the selection rule. Completeness is reported so
        # downstream summaries can distinguish the descriptive choice from a
        # usable, fully converged model-selection replicate.
        chosen = _bic_chosen(fits, scan.families)
        complete = {
            name: all(
                fit.converged and math.isfinite(fit.bic)
                for (family, _), fit in fits.items()
                if family == name
            )
            for name in scan.families
        }
        for (name, k), fit in fits.items():
            mask = _select(fit, sample_s, sample_stops, STOP_ANCHORED, cc)
            rows.append(
                {
                    "resample": int(r),
                    "family": name,
                    "k": k,
                    "threshold": round(
                        fit.union_quantile(mask, cc.quantile), cc.decimals
                    ),
                    "converged": fit.converged,
                    "bic": fit.bic,
                    "selected_components": [int(j) for j in np.flatnonzero(mask)],
                    "is_resample_bic_chosen": chosen[name] == k,
                    "model_selection_complete": complete[name],
                    "is_bic_chosen": (
                        None
                        if canonical_k_by_family is None
                        else canonical_k_by_family.get(name) == k
                    ),
                }
            )
    return pl.DataFrame(rows, schema=RESAMPLE_SCHEMA)


def resolve_n_jobs(
    config: Mapping[str, Any], environ: Mapping[str, str] | None = None
) -> int:
    """Worker count: ``$FITNESSTREE_N_JOBS`` when set, else the table's ``n_jobs``, else 8.

    A blank variable asks for nothing; any other value must be a positive integer in
    plain ASCII digits, or ``ValueError`` names the variable. The count never changes a
    result: every resample is seeded by its own index.
    """
    text = (os.environ if environ is None else environ).get(ENV_N_JOBS, "").strip()
    if text:
        if not (text.isascii() and text.isdigit()) or int(text) < 1:
            raise ValueError(f"{ENV_N_JOBS} must be a positive integer, got {text!r}.")
        return int(text)
    validate_config(config)
    return int(config.get("n_jobs", DEFAULT_N_JOBS))


def resample_thresholds_parallel(
    growth: pl.DataFrame,
    config: Any,
    resample_indices: range,
    canonical_k_by_family: Mapping[str, int] | None = None,
    *,
    classification_config: Any,
    n_jobs: int = 1,
) -> pl.DataFrame:
    """:func:`resample_thresholds`, one resample per ``joblib`` task over ``n_jobs`` workers.

    Resample ``r`` depends only on ``r`` and the seed, so the table equals the one the
    serial function gives for every ``n_jobs`` (rows in resample order). With
    ``n_jobs`` 1 or a single resample it simply calls the serial function.
    """
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}.")
    scan, cc = require_matching_settings(config, classification_config)
    indices = list(resample_indices)
    if n_jobs == 1 or len(indices) <= 1:
        return resample_thresholds(
            growth,
            config,
            resample_indices,
            canonical_k_by_family,
            classification_config=classification_config,
        )
    from joblib import Parallel, delayed

    if canonical_k_by_family is not None:
        _check_known_families(canonical_k_by_family, scan)
    data = _inputs(growth)
    frames = Parallel(n_jobs=min(n_jobs, len(indices)))(
        delayed(_resample_rows)(
            data.s, data.stops, scan, cc, range(r, r + 1), canonical_k_by_family
        )
        for r in indices
    )
    return pl.concat(frames)


def interval_table(resamples: pl.DataFrame) -> pl.DataFrame:
    """Per (family, k): the 2.5% and 97.5% quantiles of ``T`` over converged resamples.

    ``numpy.quantile`` (linear). ``n_converged`` counts the fits the interval is
    computed from and ``n_resamples`` every row; the bounds are null when none converged.
    """
    rows = []
    for (name, k), group in resamples.group_by(["family", "k"], maintain_order=True):
        usable = group.filter(pl.col("converged") & pl.col("threshold").is_not_null())[
            "threshold"
        ].to_numpy()
        low = float(np.quantile(usable, INTERVAL_LOW)) if usable.size else None
        high = float(np.quantile(usable, INTERVAL_HIGH)) if usable.size else None
        rows.append(
            {
                "family": name,
                "k": int(k),
                "threshold_low": low,
                "threshold_high": high,
                "width": None if low is None else high - low,
                "n_converged": int(usable.size),
                "n_resamples": group.height,
            }
        )
    return pl.DataFrame(rows, schema=INTERVAL_SCHEMA).sort("family", "k")


# --------------------------------------------------------------------------
# Slices for a job array
# --------------------------------------------------------------------------


def _check_coverage(resamples: pl.DataFrame, expected_n: int) -> None:
    """Every index 0..expected_n-1 exactly once per (family, k), no other index."""
    if resamples.columns != list(RESAMPLE_SCHEMA):
        raise ValueError(
            f"Resample columns are {resamples.columns}, expected {list(RESAMPLE_SCHEMA)}."
        )
    found = set(resamples["resample"].to_list())
    expected = set(range(expected_n))
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    duplicated = (
        resamples.group_by(["resample", "family", "k"]).len().filter(pl.col("len") > 1)
    )
    cells = resamples.select("family", "k").unique().height
    short = (
        resamples.group_by("resample").len().filter(pl.col("len") != cells)["resample"]
    )
    problems = []
    if missing:
        problems.append(f"missing resample indices {_head(missing)}")
    if extra:
        problems.append(f"unexpected resample indices {_head(extra)}")
    if duplicated.height:
        problems.append(
            "repeated resample indices "
            f"{_head(sorted(duplicated['resample'].unique().to_list()))}"
        )
    if short.len() and not duplicated.height:
        problems.append(
            f"resample indices without all {cells} (family, k) fits "
            f"{_head(sorted(short.to_list()))}"
        )
    if problems:
        raise ValueError(
            f"The resamples do not cover 0..{expected_n - 1} exactly once: "
            + "; ".join(problems)
            + "."
        )


def _head(values: list[int], limit: int = 20) -> str:
    shown = ", ".join(str(v) for v in values[:limit])
    return f"[{shown}{', ...' if len(values) > limit else ''}] ({len(values)})"


def merge_resample_slices(
    slice_dirs: Sequence[Path],
    expected_n: int,
    *,
    current: Mapping[str, Any],
    cells: Collection[tuple[str, int]],
) -> pl.DataFrame:
    """Concatenate the slices' ``resamples.parquet``, check them and the coverage.

    Every slice must hold a ``slice.json`` whose scan settings, classification
    settings, classification source, growth-table hash and code hashes equal
    ``current`` (:func:`current_slice_identity`); the error names the slice and the
    keys that differ. The (family, k) pairs must equal ``cells`` (:func:`scan_cells`),
    and every index ``0..expected_n - 1`` must appear exactly once per pair; the error
    names the missing, repeated and unexpected indices. The result is sorted by
    ``resample``, ``family``, ``k``.
    """
    frames = []
    for directory in slice_dirs:
        path = Path(directory) / RESAMPLES_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"No {RESAMPLES_NAME} in slice directory {directory}."
            )
        record = Path(directory) / SLICE_NAME
        if not record.is_file():
            raise FileNotFoundError(
                f"No {SLICE_NAME} in slice directory {directory}; the slice cannot "
                "be checked against the current settings."
            )
        full = json.loads(record.read_text(encoding="utf-8"))
        differ = [key for key in _SLICE_IDENTITY if full.get(key) != current.get(key)]
        if differ:
            raise ValueError(
                f"Slice {directory} was computed with other {', '.join(differ)} than "
                "the current ones; recompute it."
            )
        frames.append(pl.read_parquet(path))
    if not frames:
        raise ValueError("No slices to merge.")
    merged = pl.concat(frames).sort("resample", "family", "k")
    found = {
        (row["family"], int(row["k"]))
        for row in merged.select("family", "k").unique().to_dicts()
    }
    if found != set(cells):
        raise ValueError(
            "The slices' (family, k) fits differ from the scan's families x ks: "
            f"missing {sorted(set(cells) - found)}, unexpected {sorted(found - set(cells))}."
        )
    _check_coverage(merged, expected_n)
    return merged


def scan_cells(settings: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The (family, k) pairs the resamples must hold: ``families x ks``."""
    scan = validate_config(settings["analysis"][SECTION])
    return [(name, k) for name in scan.families for k in scan.ks]


def current_slice_identity(settings: Mapping[str, Any]) -> dict[str, Any]:
    """What a slice must record to be merged now (read without the checkpoint lock)."""
    _, source, growth_sha = _load_inputs(settings, lock=False)
    return {
        **_identity(settings),
        "classification_source": _plain(source),
        "growth_rates_sha256": growth_sha,
    }


_SLICE_IDENTITY = (
    "scan_config",
    "classification_config",
    "classification_source",
    "growth_rates_sha256",
    "code",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    """``value`` as it reads back from JSON (tuples to lists, keys sorted)."""
    return json.loads(json.dumps(value, sort_keys=True))


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _libraries() -> dict[str, str]:
    return {
        name: metadata.version(name)
        for name in ("numpy", "polars", "scikit-learn", "scipy")
    }


def _code_hashes() -> dict[str, str]:
    """sha256 of the two modules that decide the scan's numbers."""
    return {
        "mixture_families": _sha256(Path(mf.__file__)),
        "functional_threshold_scan": _sha256(Path(__file__)),
    }


def _identity(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The scan and classification settings and the code a record is compared on."""
    scientific = settings["analysis"]
    require_matching_settings(scientific[SECTION], scientific[fc.SECTION])
    scan_config = {
        key: value
        for key, value in scientific[SECTION].items()
        if key not in EXECUTION_KEYS
    }
    return {
        "scan_config": _plain(scan_config),
        "classification_config": _plain(scientific[fc.SECTION]),
        "code": _code_hashes(),
    }


def _load_inputs(
    settings: Mapping[str, Any], *, lock: bool
) -> tuple[pl.DataFrame, dict[str, Any], str]:
    path = fc.growth_rates_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"No sequence growth-rate table at {path}; run `just sequence-growth-rates`."
        )
    _, source = fc.load_cached_functional_classification_with_source(
        dict(settings), tables=False, lock=lock
    )
    return pl.read_parquet(path), source, _sha256(path)


def compute_resample_slice(
    indices: range, settings: Mapping[str, Any], out_dir: Path
) -> pl.DataFrame:
    """Compute the resamples ``indices`` and write ``resamples.parquet`` to ``out_dir``.

    The unit of work of one array task. It reads the growth-rate table and the
    classification's identity (``summary.json`` only, without the checkpoint lock, so
    concurrent tasks do not refuse each other) and records both in ``slice.json`` for
    the merge to compare.
    """
    identity = _identity(settings)
    scientific = settings["analysis"]
    growth, source, growth_sha = _load_inputs(settings, lock=False)
    frame = resample_thresholds(
        growth,
        scientific[SECTION],
        indices,
        classification_config=scientific[fc.SECTION],
    )
    out_dir = Path(out_dir)
    _write_parquet_atomic(frame, out_dir / RESAMPLES_NAME)
    _write_json_atomic(
        out_dir / SLICE_NAME,
        {
            **identity,
            "classification_source": _plain(source),
            "growth_rates_sha256": growth_sha,
            "indices": {
                "start": indices.start,
                "stop": indices.stop,
                "step": indices.step,
            },
        },
    )
    return frame


# --------------------------------------------------------------------------
# Acceptance checks
# --------------------------------------------------------------------------

_CELL_KEYS = ("family", "k", "component_rule", "quantile", "confidence")


NO_NEIGHBOURS = "no_neighbours"
NEIGHBOUR_NOT_CONVERGED = "neighbour_not_converged"


def _all_defined(verdicts: Sequence[bool | None]) -> bool | None:
    """Three-valued "all": False if any is False, None if any is None, else True."""
    if any(v is False for v in verdicts):
        return False
    return None if any(v is None for v in verdicts) else True


def _neighbour(levels: Sequence[float], value: float) -> list[float]:
    ordered = sorted(levels)
    i = ordered.index(value)
    return [ordered[j] for j in (i - 1, i + 1) if 0 <= j < len(ordered)]


def acceptance_checks(
    point: pl.DataFrame,
    intervals: pl.DataFrame,
    growth: pl.DataFrame,
    config: Any,
) -> dict[str, Any]:
    """The three acceptance checks of Decision 8, as a JSON-ready dictionary.

    (a) the canonical cell's ``frac_2stop_below`` is at least ``double_stop_below_min``;
    (b) the canonical (family, k) interval is narrower than
    ``2 * 1.96 * median(se_pooled)`` (a missing interval fails and says so);
    (c) every cell one scan step from canonical on each axis flips less than
    ``flip_fraction_limit`` of the canonical Functional set (see the module docstring
    for the steps). ``passed`` holds ``a``, ``b``, ``c`` and ``all``. A failed check is
    reported, never re-tuned.

    Check (b) records ``n_converged`` and ``n_resamples`` beside the interval. Each
    check (c) neighbour carries ``converged``; an axis with no neighbour in the scan
    (``"no_neighbours"``) or with a neighbour whose fit did not converge
    (``"neighbour_not_converged"``) has ``passed`` null and a ``reason``. Check c fails
    if any axis fails, is null (with that reason) if none fails but one is null, and
    passes otherwise; ``passed.all`` follows the same rule over a, b and c, so it is
    true only when every check is true and none is null.
    """
    scan = _as_settings(config)
    marked = point.filter(pl.col("is_canonical"))
    if marked.height != 1:
        raise ValueError("The point scan must hold exactly one canonical cell.")
    canon = marked.row(0, named=True)
    count = int(canon["functional_count"])
    if count == 0:
        raise ValueError("The canonical Functional set is empty; flips are undefined.")

    a_value = canon["frac_2stop_below"]
    a_passed = a_value is not None and a_value >= scan.double_stop_below_min

    median_se = float(growth["se_pooled"].cast(pl.Float64).median())
    limit = WIDTH_FACTOR * median_se
    row = intervals.filter(
        (pl.col("family") == canon["family"]) & (pl.col("k") == canon["k"])
    )
    width = row["width"][0] if row.height else None
    b_passed = width is not None and width < limit
    check_b = {
        "family": canon["family"],
        "k": canon["k"],
        "threshold_low": row["threshold_low"][0] if row.height else None,
        "threshold_high": row["threshold_high"][0] if row.height else None,
        "width": width,
        "n_converged": int(row["n_converged"][0]) if row.height else None,
        "n_resamples": int(row["n_resamples"][0]) if row.height else None,
        "median_se_pooled": median_se,
        "limit": limit,
        "passed": bool(b_passed),
    }

    chosen = {
        item["family"]: int(item["bic_chosen"])
        for item in point.select("family", "bic_chosen")
        .unique()
        .sort("family")
        .to_dicts()
    }
    fam, k = canon["family"], int(canon["k"])
    rule, quantile, confidence = (
        canon["component_rule"],
        canon["quantile"],
        canon["confidence"],
    )
    steps: dict[str, list[tuple[str, int, str, float, float]]] = {
        "family": [
            (f, chosen[f], rule, quantile, confidence) for f in chosen if f != fam
        ],
        "k": [(fam, kk, rule, quantile, confidence) for kk in (k - 1, k + 1)],
        "quantile": [
            (fam, k, rule, q, confidence) for q in _neighbour(scan.quantiles, quantile)
        ],
        "confidence": [
            (fam, k, rule, quantile, c)
            for c in _neighbour(scan.confidences, confidence)
        ],
        "component_rule": [
            (fam, k, other, quantile, confidence)
            for other in scan.component_rules
            if other != rule
        ],
    }
    axes: dict[str, Any] = {}
    for axis, cells in steps.items():
        neighbours = []
        for key in cells:
            hit = point.filter(
                pl.all_horizontal(pl.col(n) == v for n, v in zip(_CELL_KEYS, key))
            )
            if hit.height:
                flips = int(hit["functional_flips"][0])
                neighbours.append(
                    {
                        **dict(zip(_CELL_KEYS, key)),
                        "converged": bool(hit["converged"][0]),
                        "functional_flips": flips,
                        "flip_fraction": flips / count,
                    }
                )
        worst = max((n["flip_fraction"] for n in neighbours), default=None)
        reason = None
        if not neighbours:
            reason = NO_NEIGHBOURS
        elif not all(n["converged"] for n in neighbours):
            reason = NEIGHBOUR_NOT_CONVERGED
        axes[axis] = {
            "max_flip_fraction": worst,
            # A flip fraction is only a result when its neighbour's fit converged;
            # an axis without a usable neighbour has no verdict (None).
            "passed": (None if reason else bool(worst < scan.flip_fraction_limit)),
            "reason": reason,
            "neighbours": neighbours,
        }
    c_passed = _all_defined([item["passed"] for item in axes.values()])
    c_reason = "; ".join(
        f"{axis}: {item['reason']}"
        for axis, item in axes.items()
        if item["passed"] is None
    )
    all_passed = _all_defined([a_passed, b_passed, c_passed])
    return {
        "canonical": {key: canon[key] for key in _CELL_KEYS}
        | {"threshold": canon["threshold"]},
        "a": {
            "frac_2stop_below": a_value,
            "minimum": scan.double_stop_below_min,
            "passed": bool(a_passed),
        },
        "b": check_b,
        "c": {
            "flip_fraction_limit": scan.flip_fraction_limit,
            "canonical_functional_count": count,
            "axes": axes,
            "passed": c_passed,
            "reason": c_reason if c_passed is None else None,
        },
        "passed": {
            "a": bool(a_passed),
            "b": bool(b_passed),
            "c": c_passed,
            "all": all_passed,
        },
    }


# --------------------------------------------------------------------------
# Writing, computing and loading the scan
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FunctionalThresholdScan:
    """The scan's tables, acceptance record and provenance."""

    cells: pl.DataFrame
    comparator: pl.DataFrame
    resamples: pl.DataFrame
    intervals: pl.DataFrame
    acceptance: dict[str, Any]
    provenance: dict[str, Any]


def write_functional_threshold_scan(
    scan_dir: Path,
    *,
    cells: pl.DataFrame,
    comparator: pl.DataFrame,
    resamples: pl.DataFrame,
    intervals: pl.DataFrame,
    acceptance: Mapping[str, Any],
    settings: Mapping[str, Any],
    classification_source: Mapping[str, Any],
    growth_rates_sha256: str,
) -> dict[str, Any]:
    """Write the tables and ``acceptance.json`` atomically, then ``provenance.json``.

    A provenance of an earlier run is removed first, so it never outlives the tables
    it described while the new ones are written. Returns the provenance record.
    """
    scan_dir = Path(scan_dir)
    (scan_dir / PROVENANCE_NAME).unlink(missing_ok=True)
    for name, frame in zip(
        TABLE_NAMES, (cells, comparator, resamples, intervals), strict=True
    ):
        _write_parquet_atomic(frame, scan_dir / name)
    _write_json_atomic(scan_dir / ACCEPTANCE_NAME, acceptance)
    record = tag(
        {
            "schema": SCHEMA,
            **_identity(settings),
            "classification_source": _plain(classification_source),
            "growth_rates_sha256": growth_rates_sha256,
            "libraries": _libraries(),
            "artifacts": {name: _sha256(scan_dir / name) for name in ARTIFACT_NAMES},
        },
        ROOT,
    )
    _write_json_atomic(scan_dir / PROVENANCE_NAME, record)
    return record


def compute_functional_threshold_scan(
    settings: Mapping[str, Any],
    *,
    resamples: pl.DataFrame | None = None,
    scan_dir: Path = SCAN_DIR,
    lock: bool = True,
    n_jobs: int = 1,
) -> FunctionalThresholdScan:
    """Compute the point scan, comparator, intervals and checks, and write the scan.

    ``resamples`` is the merged slice table; when ``None`` every resample is computed
    here, over ``n_jobs`` worker processes (:func:`resample_thresholds_parallel`; a job
    array and :func:`merge_resample_slices` is the way on a cluster). It must cover
    ``0..n_resamples - 1`` exactly once. Either way the table's ``is_bic_chosen`` is
    set here from the point scan and its rows are sorted by ``resample``, ``family``,
    ``k``, so a merged table equals a computed one.
    """
    scientific = settings["analysis"]
    classification = scientific[fc.SECTION]
    scan, _ = require_matching_settings(scientific[SECTION], classification)
    growth, source, growth_sha = _load_inputs(settings, lock=lock)
    canonical, _ = fc.load_cached_functional_classification_with_source(
        dict(settings), tables=True, lock=lock
    )
    cells = point_scan(growth, scan, classification, canonical)
    comparator = comparator_cells(
        growth, canonical=canonical, classification_config=classification
    )
    chosen = {
        name: int(cells.filter(pl.col("family") == name)["bic_chosen"][0])
        for name in scan.families
    }
    if resamples is None:
        resamples = resample_thresholds_parallel(
            growth,
            scan,
            range(scan.n_resamples),
            chosen,
            classification_config=classification,
            n_jobs=n_jobs,
        )
    _check_coverage(resamples, scan.n_resamples)
    resamples = (
        resamples.with_columns(
            is_bic_chosen=pl.struct("family", "k").map_elements(
                lambda row: chosen.get(row["family"]) == row["k"],
                return_dtype=pl.Boolean,
            )
        )
        .select(list(RESAMPLE_SCHEMA))
        .sort("resample", "family", "k")
    )
    intervals = interval_table(resamples)
    acceptance = acceptance_checks(cells, intervals, growth, scan)
    provenance = write_functional_threshold_scan(
        scan_dir,
        cells=cells,
        comparator=comparator,
        resamples=resamples,
        intervals=intervals,
        acceptance=acceptance,
        settings=settings,
        classification_source=source,
        growth_rates_sha256=growth_sha,
    )
    return FunctionalThresholdScan(
        cells, comparator, resamples, intervals, acceptance, provenance
    )


def load_functional_threshold_scan(
    settings: Mapping[str, Any],
    *,
    scan_dir: Path = SCAN_DIR,
    classification_source: Mapping[str, Any] | None = None,
) -> FunctionalThresholdScan:
    """Read the scan and refuse it unless it matches the current settings.

    ``FileNotFoundError`` when there is no ``provenance.json`` (no finished scan) or a
    table is missing; ``RuntimeError`` (a stale scan) when the provenance's scan
    settings, classification settings or classification checkpoint identity differ from
    the current ones, or a table's sha256 differs from the provenance's.
    ``classification_source`` is read from the current classification checkpoint when
    not given.
    """
    scan_dir = Path(scan_dir)
    record_path = scan_dir / PROVENANCE_NAME
    if not record_path.is_file():
        raise FileNotFoundError(
            f"No finished functional-threshold scan at {scan_dir} (no "
            f"{PROVENANCE_NAME}); run `{RETRY_HINT}`."
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    identity = _identity(settings)
    if classification_source is None:
        _, classification_source = fc.load_cached_functional_classification_with_source(
            dict(settings), tables=False
        )
    stale = []
    if record.get("schema") != SCHEMA:
        stale.append("schema")
    for key, current in identity.items():
        if record.get(key) != current:
            stale.append(key)
    if record.get("classification_source") != _plain(classification_source):
        stale.append("classification_source")
    for name in ARTIFACT_NAMES:
        path = scan_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"The scan at {scan_dir} lacks {name}; run `{RETRY_HINT}`."
            )
        if record.get("artifacts", {}).get(name) != _sha256(path):
            stale.append(name)
    if stale:
        raise RuntimeError(
            f"stale functional-threshold scan at {scan_dir}: {', '.join(stale)} "
            f"differ from the current settings or classification; rerun `{RETRY_HINT}`."
        )
    return FunctionalThresholdScan(
        cells=pl.read_parquet(scan_dir / CELLS_NAME),
        comparator=pl.read_parquet(scan_dir / COMPARATOR_NAME),
        resamples=pl.read_parquet(scan_dir / RESAMPLES_NAME),
        intervals=pl.read_parquet(scan_dir / INTERVALS_NAME),
        acceptance=json.loads((scan_dir / ACCEPTANCE_NAME).read_text(encoding="utf-8")),
        provenance=record,
    )


def load_cached_functional_threshold_scan() -> FunctionalThresholdScan:
    """The scan for the current settings, as a cached loader of the registry.

    Reads ``SCAN_DIR`` against ``config/analysis.toml``. A scan that exists but no
    longer matches the settings or the classification checkpoint raises
    ``FileNotFoundError`` (the registry's "no current output" signal, like every other
    ``load_cached_*``), so the registry names a ``stale:`` gap row instead of failing;
    an absent scan raises it too.
    """
    from . import load_settings

    try:
        return load_functional_threshold_scan(load_settings(), scan_dir=SCAN_DIR)
    except RuntimeError as error:
        raise FileNotFoundError(str(error)) from error


__all__ = [
    "CONFIG_KEYS",
    "EXECUTION_KEYS",
    "SCAN_DIR",
    "SECTION",
    "FunctionalThresholdScan",
    "FunctionalThresholdScanSettings",
    "ScanSettingsMismatch",
    "acceptance_checks",
    "comparator_cells",
    "compute_functional_threshold_scan",
    "compute_resample_slice",
    "current_slice_identity",
    "interval_table",
    "load_cached_functional_threshold_scan",
    "load_functional_threshold_scan",
    "merge_resample_slices",
    "point_scan",
    "require_matching_settings",
    "resample_thresholds",
    "resample_thresholds_parallel",
    "resolve_n_jobs",
    "scan_cells",
    "validate_config",
    "write_functional_threshold_scan",
]


def main(argv: Sequence[str] | None = None) -> int:
    """``n-resamples``: print ``[functional_threshold_scan] n_resamples`` (for a job's prep).

    Slice ranges come from ``simplification.bootstrap_slices range K SLICES N``, whose
    arithmetic (``range(K*N//S, (K+1)*N//S)``) the scan shares.
    """
    import argparse

    from . import load_settings

    parser = argparse.ArgumentParser(prog="functional_threshold_scan")
    parser.add_argument("command", choices=("n-resamples",))
    parser.parse_args(argv)
    print(validate_config(load_settings()["analysis"][SECTION]).n_resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
