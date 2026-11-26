"""Pooled-standard-error scatter of the peak-uncertainty figure (ADR 2026-10-22-pooled-standard-error-for-bootstrap).

The peak checkpoint's scatter tables hold the published wild-type-relative fitness
``s`` and standard error ``se`` of every measured sequence, never the wild type.
The figure shows the pooled standard error of the growth rate from our own
barseqfit fit instead, and the wild type with its own, so this module composes its
payload from two cached products at render time and recomputes nothing:

* the peak checkpoint (sequence identities, peak flags, thresholds), and
* the growth-rate table (``r`` and ``se_pooled`` of every published sequence and
  the wild type).

For a sequence with a fit the point is ``x = se_pooled``, ``y = r - r_wt``. The
table covers every published sequence, so in practice each measured sequence has
a fit. A sequence without one keeps the published ``s`` (equal to ``r - r_wt`` by
the identity the growth-rate stage checks) and takes the pooled standard error as
the published one with the wild-type term removed in quadrature,
``sqrt(se**2 - se_wt**2)``: the term Papkou et al. add to every published
standard error (ADR 2026-10-22-pooled-standard-error-for-bootstrap). Both steps
are validated on the sequences that have a fit and fail closed outside their
tolerances.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import polars as pl

SCATTER_SUFFIX = "_scatter"
PEAKS_SUFFIX = "_peaks"
# Standard genetic code, codons ordered T, C, A, G at each position; ``*`` is a stop.
_CODE = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_BASES = "TCAG"
_SCATTER_COLUMNS = (
    "dna_sequence",
    "aa_sequence",
    "raw_fitness",
    "standard_error",
    "is_peak",
)


def translate_dna(sequence: str) -> str:
    """Amino-acid sequence of a DNA sequence read in frame from its first base."""
    if len(sequence) % 3 or set(sequence) - set(_BASES):
        raise ValueError(f"Not a codon-aligned DNA sequence: {sequence!r}.")
    codons = (sequence[i : i + 3] for i in range(0, len(sequence), 3))
    return "".join(
        _CODE[16 * _BASES.index(a) + 4 * _BASES.index(b) + _BASES.index(c)]
        for a, b, c in codons
    )


def pooled_scatter(
    peak: Mapping[str, Any],
    growth_rates: pl.DataFrame,
    *,
    identity_tolerance: float,
    standard_error_tolerance: float,
) -> dict[str, Any]:
    """Compose the figure's scatter tables and diagnostics from cached products.

    ``peak`` is the peak checkpoint (``tables`` and ``diagnostics``) and
    ``growth_rates`` the growth-rate table. ``identity_tolerance`` bounds
    ``|(r - r_wt) - s|`` on every sequence that has a fit (the growth-rate stage's
    own ``identity_tolerance``); ``standard_error_tolerance`` bounds the relative
    gap between ``se_pooled`` and the published standard error with the wild-type
    term removed. Each landscape's table is its published one with every point
    moved to the pooled scale and the wild type added; sequence identities are
    ``dna_sequence`` values throughout.
    """
    for name, value in (
        ("identity_tolerance", identity_tolerance),
        ("standard_error_tolerance", standard_error_tolerance),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    diagnostics = dict(peak["diagnostics"])
    tables = peak["tables"]
    wild_type = diagnostics.get("wild_type_sequence")
    if not isinstance(wild_type, str) or not wild_type:
        raise ValueError("Peak diagnostics do not name the wild-type sequence.")
    fits, r_wt, se_wt = _fits(growth_rates, wild_type)
    keys = sorted(
        name.removesuffix(SCATTER_SUFFIX)
        for name in tables
        if name.endswith(SCATTER_SUFFIX)
    )
    if not keys:
        raise ValueError("Peak result has no scatter table.")

    pooled: dict[str, pl.DataFrame] = {}
    identity_gap = standard_error_gap = 0.0
    fitted: dict[str, int] = {}
    reconstructed: dict[str, int] = {}
    for key in keys:
        frame, summary = _pooled_frame(
            key,
            tables,
            fits,
            wild_type=wild_type,
            r_wt=r_wt,
            se_wt=se_wt,
            identity_tolerance=identity_tolerance,
            standard_error_tolerance=standard_error_tolerance,
        )
        pooled[f"{key}{SCATTER_SUFFIX}"] = frame
        identity_gap = max(identity_gap, summary["identity_gap"])
        standard_error_gap = max(standard_error_gap, summary["standard_error_gap"])
        fitted[key], reconstructed[key] = summary["fitted"], summary["reconstructed"]

    fitness = [float(v) for frame in pooled.values() for v in frame["raw_fitness"]]
    diagnostics.update(
        {
            "fitness_domain": [min(fitness), max(fitness)],
            "fitness_domain_including_wt_zero": [min(fitness), max(fitness)],
            "wild_type_scatter_excluded": False,
            "wild_type_scatter_excluded_count": 0,
            "pooled_standard_error": {
                "wild_type_standard_error": se_wt,
                "fitted_sequence_count": fitted,
                "reconstructed_sequence_count": reconstructed,
                "max_identity_gap": identity_gap,
                "max_relative_reconstruction_gap": standard_error_gap,
                "identity_tolerance": float(identity_tolerance),
                "standard_error_tolerance": float(standard_error_tolerance),
            },
        }
    )
    return {"tables": pooled, "diagnostics": diagnostics}


def _fits(
    growth_rates: pl.DataFrame, wild_type: str
) -> tuple[pl.DataFrame, float, float]:
    """The validated fits and the wild type's ``r`` and pooled standard error."""
    missing = sorted({"dna_sequence", "r", "se_pooled"} - set(growth_rates.columns))
    if missing:
        raise ValueError(f"Growth-rate table is missing columns: {missing}.")
    fits = growth_rates.select("dna_sequence", "r", "se_pooled").with_columns(
        pl.col("r").cast(pl.Float64), pl.col("se_pooled").cast(pl.Float64)
    )
    duplicated = fits.filter(pl.col("dna_sequence").is_duplicated())
    if duplicated.height:
        raise ValueError(
            "Duplicate growth-rate rows for sequences "
            + _names(duplicated["dna_sequence"])
        )
    usable = pl.col("r").is_finite() & (
        pl.col("se_pooled").is_finite() & (pl.col("se_pooled") > 0)
    )
    bad = fits.filter(~usable.fill_null(False))
    if bad.height:
        raise ValueError(
            "Growth rate or pooled standard error is not finite and positive for "
            + _names(bad["dna_sequence"])
        )
    wild = fits.filter(pl.col("dna_sequence") == wild_type)
    if wild.height != 1:
        raise ValueError(
            f"Growth-rate table has no fit for the wild type {wild_type!r}."
        )
    return fits, float(wild["r"][0]), float(wild["se_pooled"][0])


def _pooled_frame(
    key: str,
    tables: Mapping[str, Any],
    fits: pl.DataFrame,
    *,
    wild_type: str,
    r_wt: float,
    se_wt: float,
    identity_tolerance: float,
    standard_error_tolerance: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    scatter = tables[f"{key}{SCATTER_SUFFIX}"]
    if not isinstance(scatter, pl.DataFrame):
        scatter = pl.DataFrame(scatter)
    missing = sorted(set(_SCATTER_COLUMNS) - set(scatter.columns))
    if missing:
        raise ValueError(f"Scatter table {key!r} is missing columns: {missing}.")
    peaks = tables.get(f"{key}{PEAKS_SUFFIX}")
    if peaks is None:
        raise KeyError(f"Peak result is missing table {key + PEAKS_SUFFIX!r}.")
    wild_type_is_peak = wild_type in set(peaks["dna_sequence"].to_list())
    # The wild type is added from its fit below, never read from the published table.
    scatter = scatter.select(_SCATTER_COLUMNS).filter(
        pl.col("dna_sequence") != wild_type
    )
    joined = scatter.join(fits, on="dna_sequence", how="left")
    has_fit = joined["r"].is_not_null().to_numpy()
    published_y = joined["raw_fitness"].to_numpy().astype(float)
    published_x = joined["standard_error"].to_numpy().astype(float)
    r = joined["r"].fill_null(float("nan")).to_numpy().astype(float)
    pooled_x = joined["se_pooled"].fill_null(float("nan")).to_numpy().astype(float)

    # The growth-rate stage's identity, r_i - r_wt = s_i, on every fitted sequence.
    gap = np.abs((r - r_wt) - published_y)[has_fit]
    off = ~(gap <= identity_tolerance)
    if off.any():
        offenders = joined.filter(pl.Series(has_fit))["dna_sequence"].filter(
            pl.Series(off)
        )
        raise ValueError(
            "Growth rates break the identity r_i - r_wt = s_i in the scatter: "
            f"{offenders.len()} sequences differ by more than "
            f"{identity_tolerance:g} (largest gap {float(np.nanmax(gap)):.3g}): "
            + _names(offenders)
        )

    # The published standard error is the pooled one plus the wild-type term.
    excess = published_x**2 - se_wt**2
    if (excess <= 0).any():
        raise ValueError(
            "Published standard error is not above the wild type's pooled standard "
            "error, so the wild-type term cannot be removed for "
            + _names(joined.filter(pl.Series(excess <= 0))["dna_sequence"])
        )
    removed = np.sqrt(excess)
    reconstructed = int((~has_fit).sum())
    standard_error_gap = 0.0
    if has_fit.any():
        relative = np.abs(removed[has_fit] - pooled_x[has_fit]) / pooled_x[has_fit]
        standard_error_gap = float(relative.max())
        if standard_error_gap > standard_error_tolerance:
            raise ValueError(
                "The published standard error is not the pooled standard error with "
                "the wild-type term in quadrature (ADR 2026-10-22-pooled-standard-error-for-bootstrap): relative gap "
                f"{standard_error_gap:.3g} exceeds {standard_error_tolerance:g}."
            )
    elif reconstructed:
        raise ValueError(
            "No scatter sequence has a growth-rate fit, so the wild-type term "
            "removed from the published standard error cannot be validated."
        )

    body = pl.DataFrame(
        {
            "dna_sequence": joined["dna_sequence"],
            "aa_sequence": joined["aa_sequence"],
            "raw_fitness": np.where(has_fit, r - r_wt, published_y),
            "standard_error": np.where(has_fit, pooled_x, removed),
            "is_peak": joined["is_peak"],
        }
    )
    wild = pl.DataFrame(
        {
            "dna_sequence": [wild_type],
            "aa_sequence": [translate_dna(wild_type)],
            "raw_fitness": [r_wt - r_wt],
            "standard_error": [se_wt],
            "is_peak": [wild_type_is_peak],
        },
        schema=body.schema,
    )
    summary = {
        "identity_gap": float(gap.max()) if gap.size else 0.0,
        "standard_error_gap": standard_error_gap,
        "fitted": int(has_fit.sum()) + 1,
        "reconstructed": reconstructed,
    }
    return pl.concat([body, wild]), summary


def _names(sequences: Iterable[str], shown: int = 5) -> str:
    """Name the first few sequence identities and count the rest."""
    names = sorted(sequences)
    text = ", ".join(repr(name) for name in names[:shown])
    return text + (f" and {len(names) - shown} more" if len(names) > shown else "")
