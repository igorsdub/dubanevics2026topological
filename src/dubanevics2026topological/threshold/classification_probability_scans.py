"""Measurement-uncertainty functional classification and probability requirement scans.

Performs uncertainty-aware classification for two fixed fitness thresholds:
1. Papkou original threshold:
   - Source: scipy.stats.norm.ppf(0.999, loc=-0.721, scale=0.069) = -0.5077739708744209
   - Applied: -0.5077739708744209 (exact formula precision; published rounded reference is -0.507774)
2. Median-reflected threshold:
   - Source: T = m + (m - g) from full-graph raw fitness (N = 261,333)
     m = -0.7139811713950942, g = -1.17480288668069, h = m - g = 0.4608217152855959
     unrounded T = -0.2531594561094983
   - Applied: -0.253159 (6-decimal rounded)

For each fixed threshold T and sequence i with fitness s_i and pooled standard error se_i:
    P_i = Phi((s_i - T) / se_i)
At confidence / probability requirement q (q > 0.5):
    - Functional:     P_i >= q
    - Non-functional: P_i <= round(1.0 - q, 12)
    - Uncertain:      round(1.0 - q, 12) < P_i < q

Scans q from strictly above 0.5 up to 0.99999, marking 0.95, 0.99, 0.999, 0.9999, 0.99999.
Exports per-sequence baseline (q = 0.95) classifications, scan tables, summary JSON,
publication-quality figures, and cryptographic provenance with DrWatson-style key=value naming.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import polars as pl
import vl_convert as vlc
from scipy.stats import norm

from dubanevics2026topological.analysis import OUTPUT, ROOT
from dubanevics2026topological.analysis.git_stamp import git_stamp

from . import functional_classification as fc
from . import reference_drift

# Default paths
STAGE_NAME = "classification-probability-scans"
PROCESSED_ROOT = Path(os.environ.get("FITNESSTREE_PROCESSED_ROOT") or ROOT / "data" / "processed")
DEFAULT_SOURCE_PATH = PROCESSED_ROOT / "growth_rates" / "growth_rates.parquet"
DEFAULT_OUTPUT_DIR = OUTPUT / "analysis" / "classification_probability_scans"
# processed/classification/<variant> holds the BY table of that variant's cutoff.
VARIANT_CUTOFFS = {"published": "papkou_original", "reflected": "median_reflected"}

# Numerical anchor for validation check (deferred mixture cutoff, not an active mode)
VALIDATION_ANCHOR_T = -0.49739479076248694
VALIDATION_ANCHOR_Q = 0.95
VALIDATION_ANCHOR_COUNTS = {
    "functional": 16496,
    "non_functional": 239104,
    "uncertain": 5733,
}

# Approved marked confidence levels
MARKED_Q_VALUES: tuple[float, ...] = (0.95, 0.99, 0.999, 0.9999, 0.99999)
BASELINE_Q: float = 0.95

# Visual palette (accessible high-contrast scientific styling)
COLOR_FUNCTIONAL = "#1b9e77"  # Teal green
COLOR_NON_FUNCTIONAL = "#7570b3"  # Slate purple / grey
COLOR_UNCERTAIN = "#d95f02"  # Amber orange

# FDR Multiple Testing Screening Specifications
FDR_PRIMARY_TARGET: float = 0.05
FDR_MARKED_TARGETS: tuple[float, ...] = (
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.50,
    0.75,
    0.90,
    1.0,
)
FDR_METHODS: tuple[str, ...] = ("by", "bh")
FDR_METHOD_NAMES: dict[str, str] = {
    "by": "Benjamini-Yekutieli (BY)",
    "bh": "Benjamini-Hochberg (BH)",
}
FDR_METHOD_DESCS: dict[str, str] = {
    "by": "Arbitrary dependence c(N)=13.05 (Primary)",
    "bh": "Sensitivity analysis (PRDS / independence)",
}


@dataclass(frozen=True)
class ThresholdDefinition:
    """Scientific specification and provenance of a fixed functional threshold."""

    name: str
    display_name: str
    formula: str
    source_precision_threshold: float
    applied_threshold: float
    decimals: int | None
    metadata: dict[str, Any]


def get_active_threshold_definitions() -> list[ThresholdDefinition]:
    """Return the two fixed thresholds used by the existing BY analyses."""
    # 1. Papkou original: evaluated at exact formula precision from original rounded parameters
    papkou_unrounded = float(norm.ppf(0.999, loc=-0.721, scale=0.069))
    papkou_applied = papkou_unrounded  # -0.5077739708744209 (exact formula precision)
    papkou_def = ThresholdDefinition(
        name="papkou_original",
        display_name="Papkou et al. (2023) Original",
        formula="scipy.stats.norm.ppf(0.999, loc=-0.721, scale=0.069)",
        source_precision_threshold=papkou_unrounded,
        applied_threshold=papkou_applied,
        decimals=None,
        metadata={
            "description": (
                "Papkou et al. (2023) original chosen cutoff evaluated at exact formula "
                "precision from rounded published parameters loc=-0.721, scale=0.069 at 0.999 quantile."
            ),
            "distribution": "Normal",
            "loc": -0.721,
            "scale": 0.069,
            "quantile": 0.999,
            "applied_precision_mode": "exact_formula",
            "published_rounded_reference": -0.507774,
            "config_clamped_published_functional_threshold": -0.507774,
        },
    )

    # 2. Median reflected
    # Full-graph raw fitness (N = 261,333) parameters from data/raw/papkou2023_fitness_wt.parquet
    median_val = -0.7139811713950942
    min_val = -1.17480288668069
    gap_val = median_val - min_val  # 0.4608217152855959
    unrounded_t = median_val + gap_val  # -0.2531594561094983
    applied_t = round(unrounded_t, 6)  # -0.253159
    median_def = ThresholdDefinition(
        name="median_reflected",
        display_name="Full-Graph Median Reflected",
        formula="T = m + (m - g)",
        source_precision_threshold=unrounded_t,
        applied_threshold=applied_t,
        decimals=6,
        metadata={
            "description": (
                "Median-reflected threshold reflecting the full-graph raw fitness "
                "median m through the median gap h = m - g (g is global minimum)."
            ),
            "raw_median_source_file": "data/raw/papkou2023_fitness_wt.parquet",
            "raw_median_source_sha256": "a3f3fd421b7ad7d05c7ae870c190f2d0b5a4e998bfffb3b8f76652a37882e4b4",
            "median_m": median_val,
            "global_minimum_g": min_val,
            "median_gap_h": gap_val,
            "population_size": 261333,
            "unrounded_threshold": unrounded_t,
            "applied_rule": "median_reflected",
            "applied_threshold": applied_t,
            "applied_decimals": 6,
            "fitted_s_median_in_growth_rates": -0.7139811713950914,
            "fitted_s_min_in_growth_rates": -1.174802886680697,
            "s_vs_raw_discrepancy_note": (
                "Fitted s median and min in growth_rates.parquet agree with full raw vertex table within ~1e-14."
            ),
        },
    )

    return [papkou_def, median_def]


def savename(params: Mapping[str, Any], prefix: str = "", suffix: str = "") -> str:
    """DrWatson Julia savename-style sorted key=value naming.

    Deterministic formatting: keys are sorted alphabetically, formatted as key=value,
    and joined by underscores. Floats are converted with str() to preserve exact IEEE-754
    precision for active cutoffs and requirements.
    """
    parts: list[str] = []
    for k in sorted(params.keys()):
        val = params[k]
        val_str = str(val)
        parts.append(f"{k}={val_str}")
    name = "_".join(parts)
    if prefix:
        name = f"{prefix}_{name}" if name else prefix
    if suffix:
        name = f"{name}{suffix}"
    return name


def build_probability_scan_grid(
    marked: Sequence[float] = MARKED_Q_VALUES,
) -> np.ndarray:
    """Construct a dense scan grid strictly above 0.5 up to 0.99999.

    Avoids boundary overlap at 0.5 by starting at 0.505.
    Dense linear spacing over main range, transitioning to logarithmic/logit spacing
    in the upper tail so high-confidence behavior is finely sampled.
    Guarantees exact inclusion of all `marked` points.
    """
    grid: list[float] = []
    # 0.505 to 0.950 with step 0.005
    grid.extend(np.round(np.arange(0.505, 0.950, 0.005), 6))
    # 0.950 to 0.990 with step 0.001
    grid.extend(np.round(np.arange(0.950, 0.990, 0.001), 6))
    # 0.990 to 0.999 with step 0.0001
    grid.extend(np.round(np.arange(0.990, 0.999, 0.0001), 6))
    # 0.999 to 0.9999 with step 0.00001
    grid.extend(np.round(np.arange(0.999, 0.9999, 0.00001), 6))
    # 0.9999 to 0.99999 with step 0.000001
    grid.extend(np.round(np.arange(0.9999, 0.999991, 0.000001), 6))

    # Ensure marked points are exactly included
    for m in marked:
        grid.append(float(m))

    q_array = np.array(sorted(set(grid)), dtype=np.float64)
    # Verification checks
    if (q_array <= 0.5).any():
        raise ValueError("Grid points must be strictly above 0.5.")
    if (q_array > 0.99999 + 1e-8).any():
        raise ValueError("Grid points must not exceed 0.99999.")
    for m in marked:
        if not np.isclose(q_array, m, atol=1e-9).any():
            raise ValueError(f"Marked point {m} is missing from grid.")
    return q_array


def compute_functional_probabilities(
    s: Sequence[float] | np.ndarray,
    se: Sequence[float] | np.ndarray,
    threshold: float,
    sequence_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute analytic Normal measurement-uncertainty functional probabilities P_i.

    Delegates to `dubanevics2026topological.threshold.functional_classification.functional_probability`,
    enforcing fail-closed validation on non-finite, zero, or negative SE.
    """
    return fc.functional_probability(s, se, threshold, sequence_ids=sequence_ids)


def scan_probability_requirements(
    probabilities: np.ndarray,
    grid: Sequence[float],
) -> pl.DataFrame:
    """Fast, exact scan of sequence counts across probability requirements q.

    For each q:
        lower = round(1.0 - q, 12)
        Functional (F):     P >= q
        Non-functional (NF): P <= lower
        Uncertain (U):      lower < P < q

    Computes counts via `np.searchsorted` on sorted probabilities, verified against
    the explicit 3-way partition.
    Enforces non-strict monotonicity:
        F non-increasing with q
        NF non-increasing with q
        U non-decreasing with q
    """
    p_sorted = np.sort(np.asarray(probabilities, dtype=np.float64))
    n_total = len(p_sorted)
    if n_total == 0:
        raise ValueError("Probabilities array must not be empty.")

    records: list[dict[str, Any]] = []
    prev_f = n_total + 1
    prev_nf = n_total + 1
    prev_u = -1

    for q in grid:
        q_val = float(q)
        lower = round(1.0 - q_val, 12)

        # Non-functional: P <= lower
        nf_count = int(np.searchsorted(p_sorted, lower, side="right"))
        # Functional: P >= q
        f_count = int(n_total - np.searchsorted(p_sorted, q_val, side="left"))
        # Uncertain: everything else
        u_count = n_total - f_count - nf_count

        # Exact partition check
        if f_count + nf_count + u_count != n_total:
            raise ValueError(
                f"Partition violation at q={q_val}: F={f_count}, NF={nf_count}, "
                f"U={u_count}, sum={f_count + nf_count + u_count} != {n_total}"
            )

        # Monotonicity check (plateaus allowed)
        if f_count > prev_f:
            raise ValueError(
                f"F count increased from {prev_f} to {f_count} at q={q_val}"
            )
        if nf_count > prev_nf:
            raise ValueError(
                f"NF count increased from {prev_nf} to {nf_count} at q={q_val}"
            )
        if u_count < prev_u:
            raise ValueError(
                f"U count decreased from {prev_u} to {u_count} at q={q_val}"
            )

        prev_f = f_count
        prev_nf = nf_count
        prev_u = u_count

        records.append(
            {
                "q": q_val,
                "lower_bound": lower,
                "functional_count": f_count,
                "non_functional_count": nf_count,
                "uncertain_count": u_count,
                "total_count": n_total,
                "functional_fraction": f_count / n_total,
                "non_functional_fraction": nf_count / n_total,
                "uncertain_fraction": u_count / n_total,
            }
        )

    return pl.DataFrame(records)


def baseline_sequence_classification(
    growth: pl.DataFrame,
    threshold: float,
    confidence: float = BASELINE_Q,
) -> pl.DataFrame:
    """Per-sequence baseline classification table for one fixed threshold."""
    required = ("dna_sequence", "sequence_id", "s", "se_pooled", "stop_codon_count")
    for col in required:
        if col not in growth.columns:
            raise ValueError(f"Missing required column: {col}")

    table = growth.select(*required)
    s = table["s"].to_numpy()
    se = table["se_pooled"].to_list()
    seq_ids = table["sequence_id"].to_list()

    proba = fc.functional_probability(s, se, threshold, sequence_ids=seq_ids)
    classes = fc.assign_class(proba, confidence)

    return table.with_columns(
        pl.Series("functional_probability", proba, dtype=pl.Float64),
        pl.Series("functional_class", classes, dtype=pl.String),
    )


def plot_probability_scans(
    scan_tables: Mapping[str, pl.DataFrame],
    cutoffs: Sequence[ThresholdDefinition],
    point_estimate_counts: Mapping[str, dict[str, int]],
    output_png: Path,
    output_pdf: Path | None = None,
    output_html: Path | None = None,
    marked_q: Sequence[float] = MARKED_Q_VALUES,
) -> Any:
    """Generate high-quality stacked count figures using Altair and export via vl_convert.

    Produces stacked count bars as high as total sequences (total N = 261,333):
    - First categorical bar: 'Original rule (s >= T)' / 'Point estimate (s >= T)'
      where Functional is direct s >= T, Non-functional is s < T, and Uncertain = 0.
      (Equivalent to q = 0.5 in the continuous Normal error model, not q = 0).
    - Subsequent categorical bars: actual probability requirements q in (0.95, 0.99, 0.999, 0.9999, 0.99999).
    - Stacking order (bottom to top): Functional (0 to F), Uncertain (F to F+U), Non-functional (F+U to N).
    """
    order_map = {"Functional": 0, "Uncertain": 1, "Non-functional": 2}
    color_scale = alt.Scale(
        domain=["Functional", "Uncertain", "Non-functional"],
        range=[COLOR_FUNCTIONAL, COLOR_UNCERTAIN, COLOR_NON_FUNCTIONAL],
    )

    panels = []
    for cutoff in cutoffs:
        c_title = cutoff.display_name
        t_val = cutoff.applied_threshold
        dec = cutoff.decimals
        t_str = (
            f"{t_val:.{dec}f} [{dec}-dec rounded]"
            if dec is not None
            else f"{t_val:.16g} [exact formula]"
        )

        pe = point_estimate_counts[cutoff.name]
        f_pe = pe["functional"]
        nf_pe = pe["non_functional"]
        u_pe = pe["uncertain"]
        tot = f_pe + nf_pe + u_pe

        pe_label = "No uncertainty (s >= T)"
        sort_order = [pe_label]

        rows: list[dict[str, Any]] = []
        for cat, count in [
            ("Functional", f_pe),
            ("Uncertain", u_pe),
            ("Non-functional", nf_pe),
        ]:
            rows.append(
                {
                    "mode": "point_estimate",
                    "probability": None,
                    "equivalent_continuous_normal_q": 0.5,
                    "q_label": pe_label,
                    "category": cat,
                    "count": count,
                    "fraction": count / tot,
                    "order": order_map[cat],
                }
            )

        df_scan = scan_tables[cutoff.name]
        for m_q in marked_q:
            q_label = f"q = {m_q:g}"
            sort_order.append(q_label)
            row = df_scan.filter(pl.col("q") == m_q).to_dicts()[0]
            for cat, count, frac in [
                ("Functional", row["functional_count"], row["functional_fraction"]),
                ("Uncertain", row["uncertain_count"], row["uncertain_fraction"]),
                (
                    "Non-functional",
                    row["non_functional_count"],
                    row["non_functional_fraction"],
                ),
            ]:
                rows.append(
                    {
                        "mode": "probability_threshold",
                        "probability": m_q,
                        "equivalent_continuous_normal_q": m_q,
                        "q_label": q_label,
                        "category": cat,
                        "count": count,
                        "fraction": frac,
                        "order": order_map[cat],
                    }
                )

        df_panel = pl.DataFrame(rows)
        p = (
            alt.Chart(df_panel)
            .mark_bar(size=26)
            .encode(
                x=alt.X(
                    "q_label:N",
                    title="Probability requirement",
                    sort=sort_order,
                    axis=alt.Axis(
                        labelAngle=-30,
                        labelFontSize=10,
                        titleFontSize=11,
                        titlePadding=8,
                    ),
                ),
                y=alt.Y(
                    "count:Q",
                    stack="zero",
                    title=f"Sequence count (total N = {tot:,})",
                    scale=alt.Scale(domain=[0, tot]),
                    axis=alt.Axis(
                        format=",d", labelFontSize=10, titleFontSize=11, tickCount=6
                    ),
                ),
                color=alt.Color(
                    "category:N",
                    scale=color_scale,
                    legend=alt.Legend(
                        title="Classification",
                        orient="top",
                        titleFontSize=11,
                        labelFontSize=10,
                        direction="horizontal",
                    ),
                ),
                order=alt.Order("order:Q", sort="ascending"),
                tooltip=[
                    alt.Tooltip("q_label:N", title="Requirement"),
                    alt.Tooltip("category:N", title="Class"),
                    alt.Tooltip("count:Q", title="Count", format=","),
                    alt.Tooltip("fraction:Q", title="Fraction", format=".2%"),
                ],
            )
            .properties(
                width=340,
                height=360,
                title=alt.TitleParams(
                    text=[c_title, f"Applied T = {t_str}"],
                    subtitle=[
                        "First bar: 'No uncertainty' direct point estimate s >= T (equivalent to q = 0.5 under continuous Normal error model, not q = 0).",
                        "Subsequent bars: probability requirement q filters borderline sequences into Uncertain as confidence rises.",
                    ],
                    fontSize=11.5,
                    fontWeight="bold",
                    subtitleFontSize=8.5,
                    subtitleColor="#555555",
                    anchor="middle",
                    offset=10,
                ),
            )
        )
        panels.append(p)

    if len(panels) == 1:
        chart = panels[0].configure_view(stroke=None)
    else:
        chart = (
            alt.hconcat(*panels)
            .resolve_scale(color="shared", y="shared")
            .configure_view(stroke=None)
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    spec_dict = chart.to_dict()
    png_bytes = vlc.vegalite_to_png(spec_dict, scale=2)
    output_png.write_bytes(png_bytes)

    if output_pdf is not None:
        pdf_bytes = vlc.vegalite_to_pdf(spec_dict)
        output_pdf.write_bytes(pdf_bytes)

    if output_html is not None:
        chart.save(str(output_html), format="html")

    return chart


def harmonic_number(n: int) -> float:
    """Exact harmonic number c(N) = sum_{j=1}^N (1/j) used for Benjamini-Yekutieli correction."""
    if n <= 0:
        raise ValueError(f"Harmonic number requires n > 0, got {n}")
    return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))


def compute_one_sided_gated_p_values(
    s: np.ndarray | Sequence[float],
    se: np.ndarray | Sequence[float | None],
    threshold: float,
    sequence_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-sided raw and gated p-values testing H0: s <= T vs H1: s > T.

    Validates that all standard errors are strictly positive and finite.
    - Candidate indicator: is_candidate = s >= threshold.
    - z = (s - threshold) / se.
    - Raw one-sided p-values: raw_p = norm.sf(z), log_raw_p = norm.logsf(z) for all sequences.
    - Gated p-values: gated_p = raw_p if is_candidate else 1.0; log_gated_p = log_raw_p if is_candidate else 0.0.
    Returns: (is_candidate, z_scores, raw_p, log_raw_p, gated_p, log_gated_p)
    """
    s_arr = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s_arr).all():
        raise ValueError("s must be finite for every sequence.")
    raw_se = list(se)
    errors = np.array([np.nan if item is None else float(item) for item in raw_se])
    bad = ~(np.isfinite(errors) & (errors > 0))
    if bad.any():
        index = np.flatnonzero(bad)[: fc.NAMED_LIMIT]
        named = (
            ", ".join(repr(str(sequence_ids[i])) for i in index)
            if sequence_ids is not None
            else ", ".join(f"row {i}" for i in index)
        )
        raise fc.NonPositiveStandardError(
            f"se_pooled must be finite and strictly positive: {int(bad.sum())} "
            f"invalid rows, first: {named}."
        )
    n = len(s_arr)

    is_cand = s_arr >= threshold
    z = (s_arr - threshold) / errors

    raw_p = norm.sf(z)
    log_raw_p = norm.logsf(z)

    gated_p = np.ones(n, dtype=np.float64)
    log_gated_p = np.zeros(n, dtype=np.float64)

    if np.any(is_cand):
        gated_p[is_cand] = raw_p[is_cand]
        log_gated_p[is_cand] = log_raw_p[is_cand]

    return is_cand, z, raw_p, log_raw_p, gated_p, log_gated_p


def adjust_p_values_fdr(
    log_p: np.ndarray,
    method: str = "by",
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY) FDR adjustment in log space.

    Adjusts across the full family of N gated p-values.
    Enforces right-to-left cumulative minimum monotonicity and caps adjusted values at 1.0 (log 0.0).
    Authority resides in log_adjusted_p without artificial positive floor clamping.
    Float adjusted_p legitimately underflows to 0.0 for extreme log values.
    Returns: (adjusted_p, log_adjusted_p)
    """
    method_lower = method.lower()
    if method_lower not in ("bh", "by"):
        raise ValueError(
            f"Unknown FDR adjustment method '{method}'; expected 'bh' or 'by'."
        )

    n = len(log_p)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    log_n = np.log(n)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    log_ranks = np.log(ranks)

    # Sort ascending by log_p (i.e. smallest p-value first)
    order = np.argsort(log_p)
    sorted_log_p = log_p[order]

    # BH raw step: log_p + log(N) - log(rank)
    log_adj_raw = sorted_log_p + log_n - log_ranks

    if method_lower == "by":
        c_n = harmonic_number(n)
        log_adj_raw = log_adj_raw + np.log(c_n)

    # Monotonicity: cumulative minimum from right to left (largest rank down to 1)
    log_adj_mono = np.minimum.accumulate(log_adj_raw[::-1])[::-1]
    # Cap at ln(1.0) = 0.0
    log_adj_capped = np.minimum(log_adj_mono, 0.0)

    # Invert permutation back to original array order
    rev_order = np.empty_like(order)
    rev_order[order] = np.arange(n)
    log_adjusted = log_adj_capped[rev_order]

    # Evaluate float64 adjusted p with legitimate underflow to 0.0 for extreme negative log values
    with np.errstate(under="ignore"):
        adjusted = np.exp(np.minimum(log_adjusted, 0.0))

    return adjusted, log_adjusted


def bonferroni_log_adjusted(
    log_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni adjustment min(1, N p) on the scan's log gated p-values.

    Same family size and the same cap at 1 as
    ``figures.multiple_testing_comparison.bonferroni_adjusted_p_values``.
    BH and BY are not touched.
    """
    values = np.asarray(log_p, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    log_adjusted = np.minimum(0.0, values + np.log(float(n)))
    with np.errstate(under="ignore"):
        adjusted = np.exp(log_adjusted)
    return adjusted, log_adjusted


def assign_fdr_classes(
    is_candidate: np.ndarray,
    log_adjusted_p: np.ndarray,
    target: float,
) -> tuple[np.ndarray, int, int, int]:
    """Assign sequence classification at specified FDR target q in [0, 1] using log-space decision authority.

    - Target q = 0.0: strictly F = 0, U = sum(is_candidate), NF = N - sum(is_candidate).
    - Target q = 1.0: candidate gate recovers baseline F = sum(is_candidate), U = 0, NF = N - sum(is_candidate).
    - Target 0 < q < 1:
        F: is_candidate & (log_adjusted_p <= log(target))
        U: is_candidate & ~(log_adjusted_p <= log(target))
        NF: ~is_candidate (fixed original non-candidates below threshold)
    Returns: (classes_array, f_count, u_count, nf_count)
    """
    target_val = float(target)
    if not (0.0 <= target_val <= 1.0):
        raise ValueError(f"FDR target {target_val} must be in [0.0, 1.0].")

    n = len(is_candidate)
    n_cand = int(np.sum(is_candidate))
    n_non_cand = n - n_cand

    classes = np.full(n, fc.NON_FUNCTIONAL, dtype=object)

    if target_val == 0.0:
        f_mask = np.zeros(n, dtype=bool)
    elif target_val == 1.0:
        f_mask = is_candidate.copy()
    else:
        log_target = np.log(target_val)
        f_mask = is_candidate & (log_adjusted_p <= log_target)

    u_mask = is_candidate & (~f_mask)

    classes[f_mask] = fc.FUNCTIONAL
    classes[u_mask] = fc.UNCERTAIN

    f_count = int(np.sum(f_mask))
    u_count = int(np.sum(u_mask))
    nf_count = n_non_cand

    if f_count + u_count + nf_count != n:
        raise ValueError(
            f"FDR class partition error: {f_count} + {u_count} + {nf_count} != {n}"
        )

    return classes, f_count, u_count, nf_count


def uniform_fdr_q_grid(step: float = 0.01) -> np.ndarray:
    """FDR targets from 0 to 1 inclusive at ``step`` (101 values at 0.01).

    Values are rounded to 10 decimal places so each step is the decimal it
    names. This grid is the one the supplementary proportion figures draw.
    """
    if step <= 0 or step > 1:
        raise ValueError(f"FDR step {step} must lie in (0, 1].")
    n_steps = int(round(1.0 / step))
    if abs(n_steps * step - 1.0) > 1e-12:
        raise ValueError(f"FDR step {step} does not divide 1.")
    return np.round(np.linspace(0.0, 1.0, n_steps + 1), 10)


def build_fdr_scan_grid() -> np.ndarray:
    """Build dense FDR target scan grid covering [0, 1] including marked points.

    The historical dense samples stay. The uniform 0.01 grid is included so
    the saved scan contains every step the proportion figures read.
    """
    fine_low = np.linspace(0.001, 0.10, 50)
    fine_high = np.linspace(0.10, 1.0, 60)
    grid = np.unique(
        np.concatenate(
            ([0.0], fine_low, fine_high, FDR_MARKED_TARGETS, uniform_fdr_q_grid())
        )
    )
    return np.sort(grid)


def counts_at_primary_fdr(scan: pl.DataFrame) -> dict[str, int]:
    """Functional, uncertain, and non-functional counts at the primary FDR target.

    Read from a scan table so every method, including Bonferroni, records the
    same ``counts_at_primary_fdr_0_05`` fields the summary printer reads.
    """
    matched = scan.filter(pl.col("fdr_target") == FDR_PRIMARY_TARGET)
    if matched.height != 1:
        raise ValueError(
            "FDR scan must contain exactly one row at "
            f"q={FDR_PRIMARY_TARGET}, found {matched.height}."
        )
    row = matched.to_dicts()[0]
    return {
        "functional": int(row["functional_count"]),
        "uncertain": int(row["uncertain_count"]),
        "non_functional": int(row["non_functional_count"]),
    }


def scan_fdr_targets(
    is_candidate: np.ndarray,
    log_adjusted_p: np.ndarray,
    grid: Sequence[float] | np.ndarray,
) -> pl.DataFrame:
    """Scan FDR targets across grid, evaluating decisions in log space for exact tail precision."""
    n = len(is_candidate)
    n_cand = int(np.sum(is_candidate))
    n_non_cand = n - n_cand

    cand_log_adj = np.sort(log_adjusted_p[is_candidate])

    records: list[dict[str, Any]] = []
    prev_f = -1
    prev_u = n_cand + 1

    for q in grid:
        q_val = float(q)
        if q_val == 0.0:
            f_count = 0
        elif q_val == 1.0:
            f_count = n_cand
        else:
            log_q = np.log(q_val)
            f_count = int(np.searchsorted(cand_log_adj, log_q, side="right"))

        u_count = n_cand - f_count
        nf_count = n_non_cand

        if f_count + u_count + nf_count != n:
            raise ValueError(f"FDR scan partition violation at q={q_val}")

        if f_count < prev_f:
            raise ValueError(
                f"FDR scan: F decreased from {prev_f} to {f_count} at q={q_val}"
            )
        if u_count > prev_u:
            raise ValueError(
                f"FDR scan: U increased from {prev_u} to {u_count} at q={q_val}"
            )

        prev_f = f_count
        prev_u = u_count

        records.append(
            {
                "fdr_target": q_val,
                "functional_count": f_count,
                "uncertain_count": u_count,
                "non_functional_count": nf_count,
                "total_count": n,
                "functional_fraction": f_count / n,
                "uncertain_fraction": u_count / n,
                "non_functional_fraction": nf_count / n,
            }
        )

    return pl.DataFrame(records)


def create_fdr_classification_table(
    growth: pl.DataFrame,
    threshold: float,
    method: str = "by",
    target: float = FDR_PRIMARY_TARGET,
) -> pl.DataFrame:
    """Create per-sequence FDR classification table at specified target."""
    required = ("dna_sequence", "sequence_id", "s", "se_pooled", "stop_codon_count")
    for col in required:
        if col not in growth.columns:
            raise ValueError(f"Missing required column: {col}")

    table = growth.select(*required)
    s = table["s"].to_numpy()
    se = table["se_pooled"].to_list()
    seq_ids = table["sequence_id"].to_list()

    is_cand, z, raw_p, log_raw_p, p_gated, log_p_gated = (
        compute_one_sided_gated_p_values(s, se, threshold, sequence_ids=seq_ids)
    )
    adj_p, log_adj_p = adjust_p_values_fdr(log_p_gated, method=method)
    classes, _, _, _ = assign_fdr_classes(is_cand, log_adj_p, target)

    return table.with_columns(
        pl.Series("z_score", z, dtype=pl.Float64),
        pl.Series("raw_one_sided_p_value", raw_p, dtype=pl.Float64),
        pl.Series("log_raw_one_sided_p_value", log_raw_p, dtype=pl.Float64),
        pl.Series("gated_p_value", p_gated, dtype=pl.Float64),
        pl.Series("log_gated_p_value", log_p_gated, dtype=pl.Float64),
        pl.Series("adjusted_p_value", adj_p, dtype=pl.Float64),
        pl.Series("log_adjusted_p_value", log_adj_p, dtype=pl.Float64),
        pl.Series("is_baseline_candidate", is_cand, dtype=pl.Boolean),
        pl.Series("fdr_class", classes, dtype=pl.String),
    )


def plot_fdr_scans_2x2(
    fdr_scan_tables: Mapping[tuple[str, str], pl.DataFrame],
    cutoffs: Sequence[ThresholdDefinition],
    methods: Sequence[str] = FDR_METHODS,
    output_png: Path | None = None,
    output_pdf: Path | None = None,
    output_html: Path | None = None,
    marked_targets: Sequence[float] = FDR_MARKED_TARGETS,
) -> Any:
    """Generate 2x2 split-axis publication figure (Two cutoffs x Two corrections).

    Upper band: 240,000 to 261,333 showing fixed baseline Non-functional segment reaching total N.
    Break banner: explicitly marking omitted interval 20,000–240,000.
    Lower band: 0 to 20,000 showing Functional and Uncertain segments stacked up to candidate baseline,
    plus Non-functional continuation to the break mark.
    """
    total_n = 261333
    color_scale = alt.Scale(
        domain=["Functional", "Uncertain", "Non-functional (below T)"],
        range=[COLOR_FUNCTIONAL, COLOR_UNCERTAIN, COLOR_NON_FUNCTIONAL],
    )

    t_order = [f"{q:g}" if q != 0.05 else "0.05 (5% FDR)" for q in marked_targets]

    panel_rows: list[list[Any]] = []

    for cutoff in cutoffs:
        panel_row: list[Any] = []
        for method in methods:
            df_scan = fdr_scan_tables[(cutoff.name, method)]
            df_marked = df_scan.filter(pl.col("fdr_target").is_in(list(marked_targets)))

            n_cand = int(
                df_marked.filter(pl.col("fdr_target") == 1.0)["functional_count"][0]
            )
            nf_count = total_n - n_cand

            rows_low: list[dict[str, Any]] = []
            rows_up: list[dict[str, Any]] = []

            for row in df_marked.to_dicts():
                q_val = row["fdr_target"]
                q_label = f"{q_val:g}" if q_val != 0.05 else "0.05 (5% FDR)"
                f_cnt = row["functional_count"]
                u_cnt = row["uncertain_count"]

                # Lower panel cumulative segments (0 to 20,000):
                if f_cnt > 0:
                    rows_low.append(
                        {
                            "target": q_label,
                            "category": "Functional",
                            "y0": 0,
                            "y1": f_cnt,
                            "count": f_cnt,
                            "fraction": f_cnt / total_n,
                        }
                    )
                if u_cnt > 0:
                    rows_low.append(
                        {
                            "target": q_label,
                            "category": "Uncertain",
                            "y0": f_cnt,
                            "y1": n_cand,
                            "count": u_cnt,
                            "fraction": u_cnt / total_n,
                        }
                    )
                # Non-functional continuation from n_cand up to lower domain limit (20,000)
                rows_low.append(
                    {
                        "target": q_label,
                        "category": "Non-functional (below T)",
                        "y0": n_cand,
                        "y1": 20000,
                        "count": nf_count,
                        "fraction": nf_count / total_n,
                    }
                )

                # Upper panel segment: 240,000 to total_n (261,333)
                rows_up.append(
                    {
                        "target": q_label,
                        "category": "Non-functional (below T)",
                        "y0": 240000,
                        "y1": total_n,
                        "count": nf_count,
                        "fraction": nf_count / total_n,
                    }
                )

            df_low = pl.DataFrame(rows_low)
            df_up = pl.DataFrame(rows_up)

            dec = cutoff.decimals
            t_str = (
                f"{cutoff.applied_threshold:.{dec}f} [{dec}-dec rounded]"
                if dec is not None
                else f"{cutoff.applied_threshold:.16g} [exact formula]"
            )
            m_title = FDR_METHOD_NAMES.get(method, method.upper())
            m_desc = FDR_METHOD_DESCS.get(method, "")

            c_up = (
                alt.Chart(df_up)
                .mark_bar(size=18, clip=True)
                .encode(
                    x=alt.X(
                        "target:N",
                        sort=t_order,
                        axis=alt.Axis(labels=False, ticks=False, title=None),
                    ),
                    y=alt.Y(
                        "y0:Q",
                        scale=alt.Scale(domain=[240000, total_n], clamp=True),
                        axis=alt.Axis(format=",d", tickCount=2, title="Total N"),
                    ),
                    y2=alt.Y2("y1:Q"),
                    color=alt.Color("category:N", scale=color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("count:Q", title="Full NF count", format=","),
                    ],
                )
                .properties(
                    width=310,
                    height=60,
                    title=alt.TitleParams(
                        text=f"{cutoff.display_name} | {m_title}",
                        subtitle=[
                            f"Applied T = {t_str} | {m_desc}",
                            f"Fixed baseline NF = {nf_count:,} (s < T) | All bar tops = {total_n:,}",
                        ],
                        fontSize=11,
                        fontWeight="bold",
                        subtitleFontSize=8.5,
                        subtitleColor="#555555",
                    ),
                )
            )

            break_bars = (
                alt.Chart(pl.DataFrame([{"target": t, "label": "//"} for t in t_order]))
                .mark_text(
                    fontSize=10,
                    fontWeight="bold",
                    color="#756bb1",
                    align="center",
                    baseline="middle",
                    dy=4,
                )
                .encode(
                    x=alt.X("target:N", sort=t_order, axis=None),
                    text="label:N",
                )
            )

            axis_break = (
                alt.Chart(pl.DataFrame([{"label": "//"}]))
                .mark_text(
                    fontSize=10,
                    fontWeight="bold",
                    color="#333333",
                    align="center",
                    baseline="middle",
                    dx=0,
                    dy=4,
                )
                .encode(
                    text="label:N",
                )
            )

            omitted_text = (
                alt.Chart(
                    pl.DataFrame(
                        {
                            "break_label": [
                                "≈  Omitted interval: 20,000 – 240,000 (fixed Non-functional)  ≈"
                            ]
                        }
                    )
                )
                .mark_text(
                    fontSize=7.5,
                    fontStyle="italic",
                    color="#666666",
                    align="center",
                    baseline="middle",
                    dy=-5,
                )
                .encode(
                    text="break_label:N",
                )
            )

            break_banner = (omitted_text + break_bars + axis_break).properties(
                width=310, height=18
            )

            c_low = (
                alt.Chart(df_low)
                .mark_bar(size=18, clip=True)
                .encode(
                    x=alt.X(
                        "target:N",
                        sort=t_order,
                        title="Nominal FDR target (q)",
                        axis=alt.Axis(
                            labelAngle=-30,
                            labelFontSize=9,
                            titleFontSize=10,
                            titlePadding=6,
                        ),
                    ),
                    y=alt.Y(
                        "y0:Q",
                        scale=alt.Scale(domain=[0, 20000], clamp=True),
                        axis=alt.Axis(format=",d", tickCount=5, title="Sequence count"),
                    ),
                    y2=alt.Y2("y1:Q"),
                    color=alt.Color("category:N", scale=color_scale, legend=None),
                    tooltip=[
                        alt.Tooltip("target:N", title="Target q"),
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("count:Q", title="Count", format=","),
                        alt.Tooltip(
                            "fraction:Q", title="Fraction of total", format=".2%"
                        ),
                    ],
                )
                .properties(
                    width=310,
                    height=190,
                )
            )

            panel = alt.vconcat(c_up, break_banner, c_low, spacing=2)
            panel_row.append(panel)
        panel_rows.append(panel_row)

    legend_df = pl.DataFrame(
        [
            {
                "x": 0.4,
                "y": 0,
                "color": COLOR_FUNCTIONAL,
                "label": "Functional (s >= T, adjusted p <= q)",
            },
            {
                "x": 2.9,
                "y": 0,
                "color": COLOR_UNCERTAIN,
                "label": "Uncertain (s >= T, adjusted p > q)",
            },
            {
                "x": 5.4,
                "y": 0,
                "color": COLOR_NON_FUNCTIONAL,
                "label": "Non-functional (s < T, fixed baseline)",
            },
        ]
    )
    swatches = (
        alt.Chart(legend_df)
        .mark_rect(width=14, height=14)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 8.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-0.5, 0.5]), axis=None),
            color=alt.Color("color:N", scale=None),
        )
    )
    labels = (
        alt.Chart(legend_df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=12,
            fontSize=10,
            fontWeight="bold",
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 8.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-0.5, 0.5]), axis=None),
            text="label:N",
        )
    )
    legend_header = (swatches + labels).properties(width=645, height=22)

    grid = (
        alt.vconcat(
            legend_header,
            alt.hconcat(*panel_rows[0], spacing=25),
            alt.hconcat(*panel_rows[1], spacing=25),
            spacing=18,
        )
        .properties(
            title=alt.TitleParams(
                text="One-Sided Normal Measurement-Uncertainty Screening (H0: s <= T vs H1: s > T) across Nominal FDR Targets",
                subtitle=[
                    "Nominal 5% FDR target; BY primary (arbitrary dependence c(N)=13.05), BH sensitivity analysis (PRDS/independence).",
                    "Error-control interpretation depends on validity of plug-in Normal p-values; calibration has not been established.",
                    "Dependence structure unassessed; BY robust to arbitrary dependence given valid marginal p. Does not repair overdispersion, does not account for uncertainty in estimated SE, or frozen same-data T.",
                    "Split y-axis: lower band (0–20k) shows candidates partitioned into Functional (green) and Uncertain (orange); upper band (240k–261k) shows fixed Non-functional (purple, s < T).",
                ],
                fontSize=12,
                fontWeight="bold",
                subtitleFontSize=8.5,
                subtitleColor="#444444",
                anchor="middle",
                offset=15,
            )
        )
        .configure_view(stroke=None)
    )

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = vlc.vegalite_to_png(grid.to_dict(), scale=2)
        output_png.write_bytes(png_bytes)

    if output_pdf is not None:
        pdf_bytes = vlc.vegalite_to_pdf(grid.to_dict())
        output_pdf.write_bytes(pdf_bytes)

    if output_html is not None:
        grid.save(str(output_html), format="html")

    return grid


def plot_fdr_scans_candidates_focus_2x2(
    fdr_scan_tables: Mapping[tuple[str, str], pl.DataFrame],
    cutoffs: Sequence[ThresholdDefinition],
    methods: Sequence[str] = FDR_METHODS,
    output_png: Path | None = None,
    output_pdf: Path | None = None,
    output_html: Path | None = None,
    marked_targets: Sequence[float] = FDR_MARKED_TARGETS,
) -> Any:
    """Generate 2x2 stacked bar chart focused strictly on the candidate pool (0 - 20,000)."""
    total_n = 261333
    color_scale = alt.Scale(
        domain=["Functional", "Uncertain"],
        range=[COLOR_FUNCTIONAL, COLOR_UNCERTAIN],
    )
    t_order = [f"{q:g}" if q != 0.05 else "0.05 (5% FDR)" for q in marked_targets]

    panel_rows: list[list[Any]] = []
    for cutoff in cutoffs:
        panel_row: list[Any] = []
        for method in methods:
            df_scan = fdr_scan_tables[(cutoff.name, method)]
            df_marked = df_scan.filter(pl.col("fdr_target").is_in(list(marked_targets)))
            n_cand = int(
                df_marked.filter(pl.col("fdr_target") == 1.0)["functional_count"][0]
            )
            nf_count = total_n - n_cand

            rows: list[dict[str, Any]] = []
            for row in df_marked.to_dicts():
                q_val = row["fdr_target"]
                q_label = f"{q_val:g}" if q_val != 0.05 else "0.05 (5% FDR)"
                f_cnt = row["functional_count"]
                u_cnt = row["uncertain_count"]
                rows.append(
                    {
                        "target": q_label,
                        "category": "Functional",
                        "count": f_cnt,
                        "order": 0,
                    }
                )
                rows.append(
                    {
                        "target": q_label,
                        "category": "Uncertain",
                        "count": u_cnt,
                        "order": 1,
                    }
                )

            dec = cutoff.decimals
            t_str = (
                f"{cutoff.applied_threshold:.{dec}f} [{dec}-dec rounded]"
                if dec is not None
                else f"{cutoff.applied_threshold:.16g} [exact formula]"
            )
            m_title = FDR_METHOD_NAMES.get(method, method.upper())

            c = (
                alt.Chart(pl.DataFrame(rows))
                .mark_bar(size=18)
                .encode(
                    x=alt.X(
                        "target:N",
                        sort=t_order,
                        title="FDR target (q)",
                        axis=alt.Axis(labelAngle=-30),
                    ),
                    y=alt.Y(
                        "count:Q",
                        stack="zero",
                        scale=alt.Scale(domain=[0, 20000]),
                        axis=alt.Axis(
                            format=",d", title=f"Candidate count (total = {n_cand:,})"
                        ),
                    ),
                    color=alt.Color("category:N", scale=color_scale),
                    order=alt.Order("order:Q", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("target:N", title="Target q"),
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("count:Q", title="Count", format=","),
                    ],
                )
                .properties(
                    width=310,
                    height=250,
                    title=alt.TitleParams(
                        text=f"{cutoff.display_name} | {m_title}",
                        subtitle=[
                            f"Applied T = {t_str} | Candidates = {n_cand:,}",
                            f"Fixed baseline NF = {nf_count:,} (s < T)",
                        ],
                        fontSize=11,
                        fontWeight="bold",
                        subtitleFontSize=8.5,
                    ),
                )
            )
            panel_row.append(c)
        panel_rows.append(panel_row)

    grid = (
        alt.vconcat(
            alt.hconcat(*panel_rows[0], spacing=25),
            alt.hconcat(*panel_rows[1], spacing=25),
            spacing=25,
        )
        .resolve_scale(color="shared")
        .configure_view(stroke=None)
    )

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_png.write_bytes(vlc.vegalite_to_png(grid.to_dict(), scale=2))
    if output_pdf is not None:
        output_pdf.write_bytes(vlc.vegalite_to_pdf(grid.to_dict()))
    if output_html is not None:
        grid.save(str(output_html), format="html")

    return grid


def plot_fdr_scans_full_scale_2x2(
    fdr_scan_tables: Mapping[tuple[str, str], pl.DataFrame],
    cutoffs: Sequence[ThresholdDefinition],
    methods: Sequence[str] = FDR_METHODS,
    output_png: Path | None = None,
    output_pdf: Path | None = None,
    output_html: Path | None = None,
    marked_targets: Sequence[float] = FDR_MARKED_TARGETS,
) -> Any:
    """Generate 2x2 stacked bar chart across the unbroken full population axis (0 - 261,333)."""
    total_n = 261333
    color_scale = alt.Scale(
        domain=["Functional", "Uncertain", "Non-functional (below T)"],
        range=[COLOR_FUNCTIONAL, COLOR_UNCERTAIN, COLOR_NON_FUNCTIONAL],
    )
    t_order = [f"{q:g}" if q != 0.05 else "0.05 (5% FDR)" for q in marked_targets]

    panel_rows: list[list[Any]] = []
    for cutoff in cutoffs:
        panel_row: list[Any] = []
        for method in methods:
            df_scan = fdr_scan_tables[(cutoff.name, method)]
            df_marked = df_scan.filter(pl.col("fdr_target").is_in(list(marked_targets)))
            n_cand = int(
                df_marked.filter(pl.col("fdr_target") == 1.0)["functional_count"][0]
            )
            nf_count = total_n - n_cand

            rows: list[dict[str, Any]] = []
            for row in df_marked.to_dicts():
                q_val = row["fdr_target"]
                q_label = f"{q_val:g}" if q_val != 0.05 else "0.05 (5% FDR)"
                f_cnt = row["functional_count"]
                u_cnt = row["uncertain_count"]
                rows.append(
                    {
                        "target": q_label,
                        "category": "Functional",
                        "count": f_cnt,
                        "order": 0,
                    }
                )
                rows.append(
                    {
                        "target": q_label,
                        "category": "Uncertain",
                        "count": u_cnt,
                        "order": 1,
                    }
                )
                rows.append(
                    {
                        "target": q_label,
                        "category": "Non-functional (below T)",
                        "count": nf_count,
                        "order": 2,
                    }
                )

            dec = cutoff.decimals
            t_str = (
                f"{cutoff.applied_threshold:.{dec}f} [{dec}-dec rounded]"
                if dec is not None
                else f"{cutoff.applied_threshold:.16g} [exact formula]"
            )
            m_title = FDR_METHOD_NAMES.get(method, method.upper())

            c = (
                alt.Chart(pl.DataFrame(rows))
                .mark_bar(size=18)
                .encode(
                    x=alt.X(
                        "target:N",
                        sort=t_order,
                        title="FDR target (q)",
                        axis=alt.Axis(labelAngle=-30),
                    ),
                    y=alt.Y(
                        "count:Q",
                        stack="zero",
                        scale=alt.Scale(domain=[0, total_n]),
                        axis=alt.Axis(
                            format=",d", title=f"Sequence count (total N = {total_n:,})"
                        ),
                    ),
                    color=alt.Color("category:N", scale=color_scale),
                    order=alt.Order("order:Q", sort="ascending"),
                    tooltip=[
                        alt.Tooltip("target:N", title="Target q"),
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("count:Q", title="Count", format=","),
                    ],
                )
                .properties(
                    width=310,
                    height=250,
                    title=alt.TitleParams(
                        text=f"{cutoff.display_name} | {m_title}",
                        subtitle=[
                            f"Applied T = {t_str} | Unbroken 0–{total_n:,} scale",
                            f"Fixed baseline NF = {nf_count:,} (s < T)",
                        ],
                        fontSize=11,
                        fontWeight="bold",
                        subtitleFontSize=8.5,
                    ),
                )
            )
            panel_row.append(c)
        panel_rows.append(panel_row)

    grid = (
        alt.vconcat(
            alt.hconcat(*panel_rows[0], spacing=25),
            alt.hconcat(*panel_rows[1], spacing=25),
            spacing=25,
        )
        .resolve_scale(color="shared")
        .configure_view(stroke=None)
    )

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_png.write_bytes(vlc.vegalite_to_png(grid.to_dict(), scale=2))
    if output_pdf is not None:
        output_pdf.write_bytes(vlc.vegalite_to_pdf(grid.to_dict()))
    if output_html is not None:
        grid.save(str(output_html), format="html")

    return grid


def plot_fdr_sensitivity_curves_2x2(
    fdr_scan_tables: Mapping[tuple[str, str], pl.DataFrame],
    cutoffs: Sequence[ThresholdDefinition],
    methods: Sequence[str] = FDR_METHODS,
    output_png: Path | None = None,
    output_pdf: Path | None = None,
    output_html: Path | None = None,
) -> Any:
    """Generate 2x2 continuous sensitivity curves of Functional and Uncertain counts over q in [0, 1]."""
    panel_rows: list[list[Any]] = []
    color_scale = alt.Scale(
        domain=["Functional", "Uncertain"],
        range=[COLOR_FUNCTIONAL, COLOR_UNCERTAIN],
    )
    rule_df = pl.DataFrame({"target": [0.05]})

    for cutoff in cutoffs:
        panel_row: list[Any] = []
        for method in methods:
            df_scan = fdr_scan_tables[(cutoff.name, method)]
            n_cand = int(
                df_scan.filter(pl.col("fdr_target") == 1.0)["functional_count"][0]
            )

            rows: list[dict[str, Any]] = []
            for row in df_scan.to_dicts():
                q_val = row["fdr_target"]
                rows.append(
                    {
                        "fdr_target": q_val,
                        "category": "Functional",
                        "count": row["functional_count"],
                    }
                )
                rows.append(
                    {
                        "fdr_target": q_val,
                        "category": "Uncertain",
                        "count": row["uncertain_count"],
                    }
                )

            df_curve = pl.DataFrame(rows)
            dec = cutoff.decimals
            t_str = (
                f"{cutoff.applied_threshold:.{dec}f} [{dec}-dec rounded]"
                if dec is not None
                else f"{cutoff.applied_threshold:.16g} [exact formula]"
            )
            m_title = FDR_METHOD_NAMES.get(method, method.upper())

            lines = (
                alt.Chart(df_curve)
                .mark_line(strokeWidth=2.0)
                .encode(
                    x=alt.X(
                        "fdr_target:Q",
                        scale=alt.Scale(domain=[0.0, 1.0]),
                        title="FDR target (q)",
                    ),
                    y=alt.Y(
                        "count:Q",
                        scale=alt.Scale(domain=[0, 20000]),
                        axis=alt.Axis(format=",d", title="Sequence count"),
                    ),
                    color=alt.Color("category:N", scale=color_scale),
                    strokeDash=alt.condition(
                        alt.datum.category == "Uncertain",
                        alt.value([4, 4]),
                        alt.value([0]),
                    ),
                )
            )

            rule = (
                alt.Chart(rule_df)
                .mark_rule(color="#333333", strokeDash=[3, 3], strokeWidth=1.5)
                .encode(x="target:Q")
            )

            rule_text = (
                alt.Chart(rule_df)
                .mark_text(
                    text="5% FDR",
                    angle=270,
                    align="right",
                    baseline="bottom",
                    dx=-5,
                    dy=-10,
                    fontSize=9,
                    color="#333333",
                )
                .encode(x="target:Q", y=alt.value(10))
            )

            panel = (lines + rule + rule_text).properties(
                width=310,
                height=220,
                title=alt.TitleParams(
                    text=f"{cutoff.display_name} | {m_title}",
                    subtitle=[
                        f"Applied T = {t_str} | Candidates = {n_cand:,}",
                        "Sensitivity curves across full q in [0, 1]",
                    ],
                    fontSize=11,
                    fontWeight="bold",
                    subtitleFontSize=8.5,
                ),
            )
            panel_row.append(panel)
        panel_rows.append(panel_row)

    grid = (
        alt.vconcat(
            alt.hconcat(*panel_rows[0], spacing=25),
            alt.hconcat(*panel_rows[1], spacing=25),
            spacing=25,
        )
        .resolve_scale(color="shared")
        .configure_view(stroke=None)
    )

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_png.write_bytes(vlc.vegalite_to_png(grid.to_dict(), scale=2))
    if output_pdf is not None:
        output_pdf.write_bytes(vlc.vegalite_to_pdf(grid.to_dict()))
    if output_html is not None:
        grid.save(str(output_html), format="html")

    return grid


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_of_file(path: Path) -> str:
    """Compute hex SHA256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def publish_classification(
    output_dir: Path, processed_root: Path = PROCESSED_ROOT
) -> dict[str, Path]:
    """Copy each variant's BY table and the scan provenance to ``processed/classification/<variant>``."""
    published: dict[str, Path] = {}
    for variant, cutoff in VARIANT_CUTOFFS.items():
        tables = sorted(
            output_dir.glob(
                f"fdr_classification_T=*_cutoff={cutoff}_method=by_q={FDR_PRIMARY_TARGET}.parquet"
            )
        )
        if len(tables) != 1:
            raise FileNotFoundError(
                f"Expected one BY table for cutoff {cutoff!r} in {output_dir}, found {len(tables)}."
            )
        target = processed_root / "classification" / variant
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        shutil.copyfile(tables[0], target / tables[0].name)
        shutil.copyfile(output_dir / "provenance.json", target / "provenance.json")
        published[variant] = target
    return published


def check_reference_drift(
    source_path: Path, published: Mapping[str, Path], output_dir: Path
) -> None:
    """Warn, and report deltas when a reference copy exists, if a table differs from the original run."""
    hashes = reference_drift.reference_hashes()
    reference_drift.check_drift(
        source_path,
        label="growth_rates",
        reference_sha256=hashes["growth_rates_sha256"],
        reference_pattern="growth_rates.parquet",
        report_dir=output_dir,
    )
    for variant, cutoff in VARIANT_CUTOFFS.items():
        for table in sorted(Path(published[variant]).glob("fdr_classification_*.parquet")):
            reference_drift.check_drift(
                table,
                label=f"classification_{cutoff}",
                reference_sha256=hashes[f"classification_{cutoff}_sha256"],
                reference_pattern=f"fdr_classification_*_cutoff={cutoff}_method=by_q=*.parquet",
                report_dir=output_dir,
                class_column="fdr_class",
            )


def run_pipeline(
    source_path: Path = DEFAULT_SOURCE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    processed_root: Path = PROCESSED_ROOT,
) -> dict[str, Any]:
    """Execute the uncertainty-aware functional classification and probability scan pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source verification
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source growth-rate table not found: {source_path}; "
            "run `just sequence-growth-rates`, `just all` or `just fetch-data`."
        )
    source_sha256 = sha256_of_file(source_path)

    growth = pl.read_parquet(source_path)
    total_sequences = growth.height
    cutoffs = get_active_threshold_definitions()
    grid = build_probability_scan_grid()

    # 2. Validation anchor check (deferred mixture threshold)
    anchor_s = growth["s"].to_numpy()
    anchor_se = growth["se_pooled"].to_list()
    anchor_seq_ids = growth["sequence_id"].to_list()
    anchor_p = fc.functional_probability(
        anchor_s, anchor_se, VALIDATION_ANCHOR_T, sequence_ids=anchor_seq_ids
    )
    anchor_classes = fc.assign_class(anchor_p, VALIDATION_ANCHOR_Q)
    anchor_counts = {
        "functional": int((anchor_classes == fc.FUNCTIONAL).sum()),
        "non_functional": int((anchor_classes == fc.NON_FUNCTIONAL).sum()),
        "uncertain": int((anchor_classes == fc.UNCERTAIN).sum()),
    }
    if anchor_counts != VALIDATION_ANCHOR_COUNTS:
        raise ValueError(
            f"Validation anchor mismatch at T={VALIDATION_ANCHOR_T}: "
            f"got {anchor_counts}, expected {VALIDATION_ANCHOR_COUNTS}"
        )

    # 3. Process both active cutoffs
    s_values = growth["s"].to_numpy()
    se_series = growth["se_pooled"]
    point_estimates: dict[str, dict[str, int]] = {}
    for cutoff in cutoffs:
        f_pe = int((s_values >= cutoff.applied_threshold).sum())
        nf_pe = total_sequences - f_pe
        point_estimates[cutoff.name] = {
            "functional": f_pe,
            "non_functional": nf_pe,
            "uncertain": 0,
        }

    scan_tables: dict[str, pl.DataFrame] = {}
    baseline_tables: dict[str, pl.DataFrame] = {}
    summary_baseline: dict[str, Any] = {
        "schema_version": 2,
        "population_size": total_sequences,
        "baseline_confidence": BASELINE_Q,
        "marked_confidences": list(MARKED_Q_VALUES),
        "cutoffs": {},
    }

    generated_files: dict[str, Path] = {}

    for cutoff in cutoffs:
        t_applied = cutoff.applied_threshold

        # Baseline per-sequence classification (q = 0.95)
        df_base = baseline_sequence_classification(
            growth, t_applied, confidence=BASELINE_Q
        )
        base_name = savename(
            {"cutoff": cutoff.name, "q": BASELINE_Q, "T": t_applied},
            prefix="classification",
            suffix=".parquet",
        )
        base_path = output_dir / base_name
        df_base.write_parquet(base_path)
        generated_files[base_name] = base_path
        baseline_tables[cutoff.name] = df_base

        # Probability scan
        s = growth["s"].to_numpy()
        se = growth["se_pooled"].to_list()
        seq_ids = growth["sequence_id"].to_list()
        proba = fc.functional_probability(s, se, t_applied, sequence_ids=seq_ids)
        df_scan = scan_probability_requirements(proba, grid)

        scan_csv_name = savename(
            {"cutoff": cutoff.name, "T": t_applied},
            prefix="scan",
            suffix=".csv",
        )
        scan_parquet_name = savename(
            {"cutoff": cutoff.name, "T": t_applied},
            prefix="scan",
            suffix=".parquet",
        )
        scan_csv_path = output_dir / scan_csv_name
        scan_parquet_path = output_dir / scan_parquet_name
        df_scan.write_csv(scan_csv_path)
        df_scan.write_parquet(scan_parquet_path)
        generated_files[scan_csv_name] = scan_csv_path
        generated_files[scan_parquet_name] = scan_parquet_path
        scan_tables[cutoff.name] = df_scan

        # Summary records at marked confidence levels
        marked_records: dict[str, Any] = {}
        for m_q in MARKED_Q_VALUES:
            row = df_scan.filter(pl.col("q") == m_q).to_dicts()[0]
            marked_records[f"q={m_q}"] = {
                "q": m_q,
                "functional_count": row["functional_count"],
                "non_functional_count": row["non_functional_count"],
                "uncertain_count": row["uncertain_count"],
                "functional_fraction": row["functional_fraction"],
                "non_functional_fraction": row["non_functional_fraction"],
                "uncertain_fraction": row["uncertain_fraction"],
            }

        base_row = df_scan.filter(pl.col("q") == BASELINE_Q).to_dicts()[0]
        summary_baseline["cutoffs"][cutoff.name] = {
            "display_name": cutoff.display_name,
            "formula": cutoff.formula,
            "source_precision_threshold": cutoff.source_precision_threshold,
            "applied_threshold": cutoff.applied_threshold,
            "decimals": cutoff.decimals,
            "metadata": cutoff.metadata,
            "point_estimate_baseline_no_uncertainty": {
                "mode": "point_estimate",
                "probability": None,
                "equivalent_continuous_normal_model_q": 0.5,
                "rule": "s >= T (no uncertainty; equivalent to q = 0.5 in continuous Normal model)",
                "functional_count": point_estimates[cutoff.name]["functional"],
                "non_functional_count": point_estimates[cutoff.name]["non_functional"],
                "uncertain_count": 0,
                "total_count": total_sequences,
            },
            "baseline_counts_q_0_95": {
                "functional": base_row["functional_count"],
                "non_functional": base_row["non_functional_count"],
                "uncertain": base_row["uncertain_count"],
            },
            "marked_confidence_counts": marked_records,
        }

    # Add published rounded reference point estimate check for Papkou
    f_pub_rounded = int((s_values >= -0.507774).sum())
    summary_baseline["cutoffs"]["papkou_original"]["metadata"][
        "published_rounded_point_estimate"
    ] = {
        "threshold": -0.507774,
        "mode": "point_estimate",
        "probability": None,
        "equivalent_continuous_normal_model_q": 0.5,
        "rule": "s >= -0.507774",
        "functional_count": f_pub_rounded,
        "non_functional_count": total_sequences - f_pub_rounded,
        "uncertain_count": 0,
        "diff_vs_exact_formula": f_pub_rounded
        - point_estimates["papkou_original"]["functional"],
    }

    # Combined scan table
    combined_scan_list: list[pl.DataFrame] = []
    for cutoff in cutoffs:
        df_c = scan_tables[cutoff.name].with_columns(
            pl.lit(cutoff.name).alias("cutoff"),
            pl.lit(cutoff.applied_threshold).alias("applied_threshold"),
        )
        combined_scan_list.append(df_c)
    df_combined_scan = pl.concat(combined_scan_list)
    comb_csv_name = "scan_all_cutoffs.csv"
    comb_parquet_name = "scan_all_cutoffs.parquet"
    comb_csv_path = output_dir / comb_csv_name
    comb_parquet_path = output_dir / comb_parquet_name
    df_combined_scan.write_csv(comb_csv_path)
    df_combined_scan.write_parquet(comb_parquet_path)
    generated_files[comb_csv_name] = comb_csv_path
    generated_files[comb_parquet_name] = comb_parquet_path

    # Altair Stacked Count Figures (PNG, PDF, HTML)
    fig_png_name = "figure_classification_probability_scans.png"
    fig_pdf_name = "figure_classification_probability_scans.pdf"
    fig_html_name = "figure_classification_probability_scans.html"
    fig_png_path = output_dir / fig_png_name
    fig_pdf_path = output_dir / fig_pdf_name
    fig_html_path = output_dir / fig_html_name
    plot_probability_scans(
        scan_tables,
        cutoffs,
        point_estimate_counts=point_estimates,
        output_png=fig_png_path,
        output_pdf=fig_pdf_path,
        output_html=fig_html_path,
        marked_q=MARKED_Q_VALUES,
    )
    generated_files[fig_png_name] = fig_png_path
    generated_files[fig_pdf_name] = fig_pdf_path
    generated_files[fig_html_name] = fig_html_path

    # Individual Altair figures per cutoff (PNG, PDF, HTML)
    for cutoff in cutoffs:
        indiv_png_name = savename(
            {"cutoff": cutoff.name, "T": cutoff.applied_threshold},
            prefix="figure",
            suffix=".png",
        )
        indiv_pdf_name = savename(
            {"cutoff": cutoff.name, "T": cutoff.applied_threshold},
            prefix="figure",
            suffix=".pdf",
        )
        indiv_html_name = savename(
            {"cutoff": cutoff.name, "T": cutoff.applied_threshold},
            prefix="figure",
            suffix=".html",
        )
        indiv_png_path = output_dir / indiv_png_name
        indiv_pdf_path = output_dir / indiv_pdf_name
        indiv_html_path = output_dir / indiv_html_name
        plot_probability_scans(
            {cutoff.name: scan_tables[cutoff.name]},
            [cutoff],
            point_estimate_counts={cutoff.name: point_estimates[cutoff.name]},
            output_png=indiv_png_path,
            output_pdf=indiv_pdf_path,
            output_html=indiv_html_path,
            marked_q=MARKED_Q_VALUES,
        )
        generated_files[indiv_png_name] = indiv_png_path
        generated_files[indiv_pdf_name] = indiv_pdf_path
        generated_files[indiv_html_name] = indiv_html_path

    # =========================================================================
    # 4. One-Sided FDR Multiple Testing Screening Analyses
    # Four analyses: 2 cutoffs (Papkou original, Median reflected) x 2 methods (BY, BH)
    # Family denominator: full population N = 261,333 (separate per cutoff)
    # One-sided hypothesis: H0: s_i <= T vs H1: s_i > T
    # Candidate gate: p_i* = norm.sf((s_i - T)/se_i) if s_i >= T else 1.0
    # =========================================================================
    fdr_grid = build_fdr_scan_grid()
    fdr_scan_tables: dict[tuple[str, str], pl.DataFrame] = {}
    fdr_classification_tables: dict[tuple[str, str], pl.DataFrame] = {}
    fdr_summary: dict[str, Any] = {
        "schema_version": 1,
        "population_size": total_sequences,
        "primary_fdr_target": FDR_PRIMARY_TARGET,
        "marked_fdr_targets": list(FDR_MARKED_TARGETS),
        "methods": {
            "by": {
                "name": FDR_METHOD_NAMES["by"],
                "description": FDR_METHOD_DESCS["by"],
                "harmonic_constant_c_N": harmonic_number(total_sequences),
            },
            "bh": {
                "name": FDR_METHOD_NAMES["bh"],
                "description": FDR_METHOD_DESCS["bh"],
            },
        },
        "cutoffs": {},
    }

    combined_fdr_scan_list: list[pl.DataFrame] = []

    for cutoff in cutoffs:
        t_applied = cutoff.applied_threshold
        is_cand, z, raw_p, log_raw_p, p_gated, log_p_gated = (
            compute_one_sided_gated_p_values(
                s_values,
                se_series.to_list(),
                t_applied,
                sequence_ids=growth["sequence_id"].to_list(),
            )
        )
        n_cand = int(np.sum(is_cand))
        n_non_cand = total_sequences - n_cand

        cutoff_fdr_summary: dict[str, Any] = {
            "display_name": cutoff.display_name,
            "applied_threshold": t_applied,
            "decimals": cutoff.decimals,
            "baseline_candidates_count": n_cand,
            "baseline_non_candidates_count": n_non_cand,
            "methods": {},
        }

        # Raw alpha = 0.05 unadjusted baseline check
        raw_sig = is_cand & (raw_p <= 0.05)
        raw_f = int(np.sum(raw_sig))
        cutoff_fdr_summary["raw_alpha_0_05_unadjusted_reference"] = {
            "functional_count": raw_f,
            "uncertain_count": n_cand - raw_f,
            "non_functional_count": n_non_cand,
            "rule": "s >= T & p_raw <= 0.05 (unadjusted individual p-values)",
        }

        for method in FDR_METHODS:
            adj_p, log_adj_p = adjust_p_values_fdr(log_p_gated, method=method)

            # Per-sequence classification table at target 5% FDR
            classes_5pct, f_5pct, u_5pct, nf_5pct = assign_fdr_classes(
                is_cand, log_adj_p, FDR_PRIMARY_TARGET
            )

            df_fdr_class = growth.select(
                "dna_sequence", "sequence_id", "s", "se_pooled", "stop_codon_count"
            ).with_columns(
                pl.Series("z_score", z, dtype=pl.Float64),
                pl.Series("raw_one_sided_p_value", raw_p, dtype=pl.Float64),
                pl.Series("log_raw_one_sided_p_value", log_raw_p, dtype=pl.Float64),
                pl.Series("gated_p_value", p_gated, dtype=pl.Float64),
                pl.Series("log_gated_p_value", log_p_gated, dtype=pl.Float64),
                pl.Series("adjusted_p_value", adj_p, dtype=pl.Float64),
                pl.Series("log_adjusted_p_value", log_adj_p, dtype=pl.Float64),
                pl.Series("is_baseline_candidate", is_cand, dtype=pl.Boolean),
                pl.Series("fdr_class", classes_5pct, dtype=pl.String),
            )
            fdr_class_name = savename(
                {
                    "cutoff": cutoff.name,
                    "method": method,
                    "q": FDR_PRIMARY_TARGET,
                    "T": t_applied,
                },
                prefix="fdr_classification",
                suffix=".parquet",
            )
            fdr_class_path = output_dir / fdr_class_name
            df_fdr_class.write_parquet(fdr_class_path)
            generated_files[fdr_class_name] = fdr_class_path
            fdr_classification_tables[(cutoff.name, method)] = df_fdr_class

            # Full scan across FDR grid (evaluated with log-space decision authority)
            df_fdr_scan = scan_fdr_targets(is_cand, log_adj_p, fdr_grid)
            fdr_scan_csv_name = savename(
                {"cutoff": cutoff.name, "method": method, "T": t_applied},
                prefix="fdr_scan",
                suffix=".csv",
            )
            fdr_scan_parquet_name = savename(
                {"cutoff": cutoff.name, "method": method, "T": t_applied},
                prefix="fdr_scan",
                suffix=".parquet",
            )
            fdr_scan_csv_path = output_dir / fdr_scan_csv_name
            fdr_scan_parquet_path = output_dir / fdr_scan_parquet_name
            df_fdr_scan.write_csv(fdr_scan_csv_path)
            df_fdr_scan.write_parquet(fdr_scan_parquet_path)
            generated_files[fdr_scan_csv_name] = fdr_scan_csv_path
            generated_files[fdr_scan_parquet_name] = fdr_scan_parquet_path
            fdr_scan_tables[(cutoff.name, method)] = df_fdr_scan

            # For combined table
            df_fdr_c = df_fdr_scan.with_columns(
                pl.lit(cutoff.name).alias("cutoff"),
                pl.lit(t_applied).alias("applied_threshold"),
                pl.lit(method).alias("method"),
            )
            combined_fdr_scan_list.append(df_fdr_c)

            # Marked records summary
            cand_adj_p = adj_p[is_cand]
            max_cand_adj = float(np.max(cand_adj_p)) if len(cand_adj_p) > 0 else 1.0
            marked_fdr_records: dict[str, Any] = {}
            for mq in FDR_MARKED_TARGETS:
                row = df_fdr_scan.filter(pl.col("fdr_target") == mq).to_dicts()[0]
                marked_fdr_records[f"q={mq:g}"] = {
                    "fdr_target": mq,
                    "functional_count": row["functional_count"],
                    "uncertain_count": row["uncertain_count"],
                    "non_functional_count": row["non_functional_count"],
                    "functional_fraction": row["functional_fraction"],
                    "uncertain_fraction": row["uncertain_fraction"],
                    "non_functional_fraction": row["non_functional_fraction"],
                }

            cutoff_fdr_summary["methods"][method] = {
                "method_name": FDR_METHOD_NAMES.get(method, method.upper()),
                "saturation_target_q": max_cand_adj,
                "counts_at_primary_fdr_0_05": {
                    "functional": f_5pct,
                    "uncertain": u_5pct,
                    "non_functional": nf_5pct,
                },
                "marked_targets_counts": marked_fdr_records,
            }

        # Bonferroni on the same gated p-values and the same q grid. The 2x2
        # BY/BH plotters stay on FDR_METHODS, and no per-sequence
        # classification table is written. The summary still records the
        # primary-target counts from this scan.
        bonf_adj, bonf_log = bonferroni_log_adjusted(log_p_gated)
        df_bonf = scan_fdr_targets(is_cand, bonf_log, fdr_grid)
        bonf_key = {"cutoff": cutoff.name, "method": "bonferroni", "T": t_applied}
        bonf_parquet = output_dir / savename(bonf_key, prefix="fdr_scan", suffix=".parquet")
        bonf_csv = output_dir / savename(bonf_key, prefix="fdr_scan", suffix=".csv")
        df_bonf.write_parquet(bonf_parquet)
        df_bonf.write_csv(bonf_csv)
        generated_files[bonf_parquet.name] = bonf_parquet
        generated_files[bonf_csv.name] = bonf_csv
        fdr_scan_tables[(cutoff.name, "bonferroni")] = df_bonf
        combined_fdr_scan_list.append(
            df_bonf.with_columns(
                pl.lit(cutoff.name).alias("cutoff"),
                pl.lit(t_applied).alias("applied_threshold"),
                pl.lit("bonferroni").alias("method"),
            )
        )
        cand_bonf = bonf_adj[is_cand]
        cutoff_fdr_summary["methods"]["bonferroni"] = {
            "method_name": "Bonferroni",
            "saturation_target_q": (
                float(np.max(cand_bonf)) if len(cand_bonf) > 0 else 1.0
            ),
            "counts_at_primary_fdr_0_05": counts_at_primary_fdr(df_bonf),
        }

        fdr_summary["cutoffs"][cutoff.name] = cutoff_fdr_summary

    # Combined FDR scan table
    df_combined_fdr_scan = pl.concat(combined_fdr_scan_list)
    comb_fdr_csv_name = "fdr_scan_all_cutoffs_and_methods.csv"
    comb_fdr_parquet_name = "fdr_scan_all_cutoffs_and_methods.parquet"
    comb_fdr_csv_path = output_dir / comb_fdr_csv_name
    comb_fdr_parquet_path = output_dir / comb_fdr_parquet_name
    df_combined_fdr_scan.write_csv(comb_fdr_csv_path)
    df_combined_fdr_scan.write_parquet(comb_fdr_parquet_path)
    generated_files[comb_fdr_csv_name] = comb_fdr_csv_path
    generated_files[comb_fdr_parquet_name] = comb_fdr_parquet_path

    # Altair 2x2 Figures (PNG, PDF, HTML)
    # 1. Primary Split-Axis 2x2
    fig_fdr_split_png = output_dir / "figure_fdr_scans_split_axis_2x2.png"
    fig_fdr_split_pdf = output_dir / "figure_fdr_scans_split_axis_2x2.pdf"
    fig_fdr_split_html = output_dir / "figure_fdr_scans_split_axis_2x2.html"
    plot_fdr_scans_2x2(
        fdr_scan_tables,
        cutoffs,
        methods=FDR_METHODS,
        output_png=fig_fdr_split_png,
        output_pdf=fig_fdr_split_pdf,
        output_html=fig_fdr_split_html,
        marked_targets=FDR_MARKED_TARGETS,
    )
    generated_files[fig_fdr_split_png.name] = fig_fdr_split_png
    generated_files[fig_fdr_split_pdf.name] = fig_fdr_split_pdf
    generated_files[fig_fdr_split_html.name] = fig_fdr_split_html

    # 2. Candidate Focus 2x2
    fig_fdr_focus_png = output_dir / "figure_fdr_scans_candidates_focus_2x2.png"
    fig_fdr_focus_pdf = output_dir / "figure_fdr_scans_candidates_focus_2x2.pdf"
    fig_fdr_focus_html = output_dir / "figure_fdr_scans_candidates_focus_2x2.html"
    plot_fdr_scans_candidates_focus_2x2(
        fdr_scan_tables,
        cutoffs,
        methods=FDR_METHODS,
        output_png=fig_fdr_focus_png,
        output_pdf=fig_fdr_focus_pdf,
        output_html=fig_fdr_focus_html,
        marked_targets=FDR_MARKED_TARGETS,
    )
    generated_files[fig_fdr_focus_png.name] = fig_fdr_focus_png
    generated_files[fig_fdr_focus_pdf.name] = fig_fdr_focus_pdf
    generated_files[fig_fdr_focus_html.name] = fig_fdr_focus_html

    # 3. Full-Scale 2x2
    fig_fdr_full_png = output_dir / "figure_fdr_scans_full_scale_2x2.png"
    fig_fdr_full_pdf = output_dir / "figure_fdr_scans_full_scale_2x2.pdf"
    fig_fdr_full_html = output_dir / "figure_fdr_scans_full_scale_2x2.html"
    plot_fdr_scans_full_scale_2x2(
        fdr_scan_tables,
        cutoffs,
        methods=FDR_METHODS,
        output_png=fig_fdr_full_png,
        output_pdf=fig_fdr_full_pdf,
        output_html=fig_fdr_full_html,
        marked_targets=FDR_MARKED_TARGETS,
    )
    generated_files[fig_fdr_full_png.name] = fig_fdr_full_png
    generated_files[fig_fdr_full_pdf.name] = fig_fdr_full_pdf
    generated_files[fig_fdr_full_html.name] = fig_fdr_full_html

    # 4. Sensitivity Curves 2x2
    fig_fdr_curves_png = output_dir / "figure_fdr_sensitivity_curves_2x2.png"
    fig_fdr_curves_pdf = output_dir / "figure_fdr_sensitivity_curves_2x2.pdf"
    fig_fdr_curves_html = output_dir / "figure_fdr_sensitivity_curves_2x2.html"
    plot_fdr_sensitivity_curves_2x2(
        fdr_scan_tables,
        cutoffs,
        methods=FDR_METHODS,
        output_png=fig_fdr_curves_png,
        output_pdf=fig_fdr_curves_pdf,
        output_html=fig_fdr_curves_html,
    )
    generated_files[fig_fdr_curves_png.name] = fig_fdr_curves_png
    generated_files[fig_fdr_curves_pdf.name] = fig_fdr_curves_pdf
    generated_files[fig_fdr_curves_html.name] = fig_fdr_curves_html

    # Write dedicated summary_fdr_and_scans.json
    summary_fdr_name = "summary_fdr_and_scans.json"
    summary_fdr_path = output_dir / summary_fdr_name
    with summary_fdr_path.open("w", encoding="utf-8") as f:
        json.dump(fdr_summary, f, indent=2)
    generated_files[summary_fdr_name] = summary_fdr_path

    # Update summary_baseline_and_scans.json with pointer to FDR analyses
    summary_baseline["fdr_screening_analyses"] = fdr_summary

    # Write summary JSON
    summary_name = "summary_baseline_and_scans.json"
    summary_path = output_dir / summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_baseline, f, indent=2)
    generated_files[summary_name] = summary_path

    # Compute sha256 for all generated files
    file_hashes = {name: sha256_of_file(path) for name, path in generated_files.items()}

    # Explicit standard error audit on input table
    se_series = growth["se_pooled"]
    se_audit = {
        "total_sequences": total_sequences,
        "observed_invalid_se_count": int(
            (~(se_series.is_finite() & (se_series > 0))).sum()
        ),
        "null_se_count": int(se_series.null_count()),
        "non_positive_se_count": int((se_series <= 0).sum()),
        "non_finite_se_count": int((~se_series.is_finite()).sum()),
        "min_se_pooled": float(se_series.min()),
        "max_se_pooled": float(se_series.max()),
        "se_rule": "fail-closed (every sequence must have finite se_pooled > 0)",
    }

    # Provenance record
    provenance: dict[str, Any] = {
        "stage": STAGE_NAME,
        "git_stamp": git_stamp(ROOT),
        "python_version": sys.version,
        "platform": platform.platform(),
        "input_source": {
            "path": _relative_to_root(source_path),
            "sha256": source_sha256,
            "population_size": total_sequences,
        },
        "standard_error_audit": se_audit,
        "model_definition": {
            "type": "Analytic Normal measurement-uncertainty probability",
            "formula": "P_i = Phi((s_i - T) / se_pooled_i)",
            "classes": {
                "functional": "P_i >= q",
                "non_functional": "P_i <= round(1.0 - q, 12)",
                "uncertain": "round(1.0 - q, 12) < P_i < q",
            },
            "scientific_boundary_note": (
                "Analytic Normal measurement-uncertainty probability evaluates individual measurement "
                "standard errors and is distinct from threshold bootstrap stability or tree bootstrap."
            ),
        },
        "validation_anchor": {
            "threshold_T": VALIDATION_ANCHOR_T,
            "confidence_q": VALIDATION_ANCHOR_Q,
            "expected_counts": VALIDATION_ANCHOR_COUNTS,
            "verified": True,
        },
        "cutoffs": [asdict(c) for c in cutoffs],
        "marked_confidences": list(MARKED_Q_VALUES),
        "baseline_confidence": BASELINE_Q,
        "fdr_screening_model": {
            "type": "One-sided Normal measurement-uncertainty screening across nominal FDR targets",
            "hypothesis": "H0: mean(s_i) <= T vs H1: mean(s_i) > T",
            "family_denominator": "Full library N = 261,333 separate per cutoff",
            "gating_rule": "p_i* = norm.sf((s_i - T)/se_i) if s_i >= T else 1.0 (non-candidates fixed at 1.0)",
            "primary_correction": f"Benjamini-Yekutieli (BY) arbitrary dependence c(N) = {harmonic_number(total_sequences):.6f}",
            "sensitivity_correction": "Benjamini-Hochberg (BH) sensitivity analysis (PRDS / independence)",
            "primary_target_fdr": FDR_PRIMARY_TARGET,
            "marked_fdr_targets": list(FDR_MARKED_TARGETS),
            "caveats_and_limitations": [
                "Nominal approximate model-based screening, not a calibrated guarantee, posterior, or time.",
                "Dependence structure across sequences is unassessed; BY is robust to arbitrary dependence conditional on valid marginal p-values; BH is provided as a sensitivity analysis.",
                "WT-relative normalization alone does not establish covariance or prove BH inappropriate; true dependence structure remains unassessed.",
                "Estimated standard errors treated as fixed plug-in values; overdispersion, uncertainty in estimated standard errors, and wild-type normalization uncertainty are not accounted for or repaired by BY.",
                "Threshold T is frozen at its realized point-estimate; joint threshold resampling or refitting uncertainty is not modeled.",
                "Non-candidates (s < T) are fixed as below-cutoff point estimate, reflecting original screening baseline, not formal statistical proof of non-functionality.",
                "Discovery subset relation: BY functional discoveries are a subset (non-strict count inequality F_BY <= F_BH) of BH discoveries at any given nominal target.",
            ],
        },
        "generated_artifacts": file_hashes,
    }

    prov_name = "provenance.json"
    prov_path = output_dir / prov_name
    with prov_path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    published = publish_classification(output_dir, processed_root)
    check_reference_drift(source_path, published, output_dir)

    # Durable delivery bundles in gitignored workspace directories (.context and delivery)
    bundle_name = "classification_probability_scans_bundle.zip"
    bundle_paths: list[Path] = []
    for dest_dir in [ROOT / ".context", ROOT / "delivery"]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = dest_dir / bundle_name
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, fpath in generated_files.items():
                zf.write(fpath, arcname=arcname)
            zf.write(prov_path, arcname=prov_name)
        bundle_paths.append(bundle_path)

    return {
        "summary": summary_baseline,
        "provenance": provenance,
        "files": generated_files,
        "bundles": bundle_paths,
        "classification": published,
    }


def main() -> None:
    """CLI entrypoint."""
    print("Running classification probability scans pipeline...")
    result = run_pipeline()
    print("Completed successfully.")
    print("\n--- Symmetric Baseline Counts at q = 0.95 ---")
    cutoffs_summary = result["summary"]["cutoffs"]
    for c_name, c_data in cutoffs_summary.items():
        counts = c_data["baseline_counts_q_0_95"]
        print(
            f"  [{c_name} | T={c_data['applied_threshold']}]: F={counts['functional']}, NF={counts['non_functional']}, U={counts['uncertain']}"
        )
    print("\n--- One-Sided FDR Screening Counts at Primary Target 5% (q = 0.05) ---")
    fdr_summary = result["summary"]["fdr_screening_analyses"]["cutoffs"]
    for c_name, c_data in fdr_summary.items():
        print(
            f"  [{c_name} | T={c_data['applied_threshold']}]: Candidates={c_data['baseline_candidates_count']}, Non-candidates={c_data['baseline_non_candidates_count']}"
        )
        for m_data in c_data["methods"].values():
            fdr_counts = m_data["counts_at_primary_fdr_0_05"]
            print(
                f"    {m_data['method_name']}: F={fdr_counts['functional']}, U={fdr_counts['uncertain']}, NF={fdr_counts['non_functional']}"
            )
    print(f"\nArtifacts written to: {DEFAULT_OUTPUT_DIR}")
    for variant, target in result.get("classification", {}).items():
        print(f"Classification for {variant}: {target}")


if __name__ == "__main__":
    main()
