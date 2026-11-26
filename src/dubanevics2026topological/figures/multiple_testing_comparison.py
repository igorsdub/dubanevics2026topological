"""Functional calls under three corrections at the two fixed thresholds.

Supplementary comparison of how many sequences the pipeline calls Functional
at nominal q = 0.05. The Papkou and Topographer panels are the Papkou
original and median-reflected cutoffs. Benjamini-Yekutieli is the class
already stored on the reviewed classification snapshot. Benjamini-Hochberg
and Bonferroni are applied here to that snapshot's gated p-values, the same
family the scan adjusts. The scan-stage Altair renderers are not used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ._title_flag import figure_heading
from .labels import PAPKOU, TOPOGRAPHER
from .palette import COUNT_BLACK, MID_GREY

FIGURE_ID = "multiple_testing_comparison"
INPUT_NAME = "multiple_testing_comparison"
# The snapshot's method is BY; BH and Bonferroni are derived from its p-values.
SNAPSHOT_METHOD = "by"
CUTOFFS = ("papkou_original", "median_reflected")
THRESHOLD_LABELS = {
    "papkou_original": PAPKOU,
    "median_reflected": TOPOGRAPHER,
}
METHODS = (
    ("bonferroni", "Bonferroni"),
    ("bh", "Benjamini-Hochberg"),
    ("by", "Benjamini-Yekutieli"),
)
_CLASS_COLUMNS = (
    "gated_p_value",
    "is_baseline_candidate",
    "fdr_class",
)


def bonferroni_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni adjustment min(1, N p), family size N = len(p_values)."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p-values must be a one-dimensional family.")
    if values.size == 0:
        return values.copy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("p-values must be finite and on [0, 1].")
    return np.minimum(1.0, values * values.size)


def bh_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values from the scan's log-space adjustment."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p-values must be a one-dimensional family.")
    if values.size == 0:
        return values.copy()
    if not np.isfinite(values).all() or (values <= 0).any() or (values > 1).any():
        raise ValueError("BH p-values must be finite and on (0, 1].")
    adjusted, _ = _adjust(np.log(values), "bh")
    return adjusted


def functional_call_count(
    gated_p_values: np.ndarray,
    is_candidate: np.ndarray,
    method: str,
    nominal_q: float,
) -> int:
    """Sequences at or above the threshold whose adjusted p-value is at most q.

    ``method`` is ``bonferroni``, ``bh`` or ``by``. BH and BY use the scan's
    adjustment and its Functional rule; Bonferroni uses the same rule on
    ``bonferroni_adjusted_p_values``.
    """
    gated = np.asarray(gated_p_values, dtype=np.float64)
    candidates = np.asarray(is_candidate, dtype=bool)
    if gated.shape != candidates.shape or gated.ndim != 1:
        raise ValueError("gated p-values and the candidate mask must share one shape.")
    if method == "bonferroni":
        adjusted = bonferroni_adjusted_p_values(gated)
        return int(np.sum(candidates & (adjusted <= nominal_q)))
    if method not in {"bh", "by"}:
        raise ValueError(f"Unknown correction {method!r}.")
    _, log_adjusted = _adjust(np.log(gated), method)
    from dubanevics2026topological.threshold.classification_probability_scans import (
        assign_fdr_classes,
    )

    _, functional, _, _ = assign_fdr_classes(candidates, log_adjusted, nominal_q)
    return functional


def comparison_counts(
    tables: Mapping[str, pl.DataFrame], nominal_q: float
) -> pl.DataFrame:
    """One row per threshold and correction.

    The BY row must equal the snapshot's stored Functional class. A mismatch
    means the gated p-values and the saved classification are not the same run.
    """
    rows: list[dict[str, object]] = []
    for cutoff in CUTOFFS:
        table = tables[cutoff]
        missing = [name for name in _CLASS_COLUMNS if name not in table.columns]
        if missing:
            raise ValueError(f"{cutoff} classification is missing {missing}.")
        gated = table["gated_p_value"].to_numpy()
        candidates = table["is_baseline_candidate"].to_numpy()
        label = THRESHOLD_LABELS[cutoff]
        counts = {
            method: functional_call_count(gated, candidates, method, nominal_q)
            for method, _ in METHODS
        }
        stored = int((table["fdr_class"] == "functional").sum())
        if counts["by"] != stored:
            raise ValueError(
                f"{label} BY count {counts['by']} does not match the stored "
                f"Functional class ({stored})."
            )
        for method, method_label in METHODS:
            rows.append(
                {
                    "threshold": label,
                    "method": method_label,
                    "functional_count": counts[method],
                }
            )
    return pl.DataFrame(rows)


def load_multiple_testing_comparison(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Read the two BY classification tables of the declared snapshot."""
    from dubanevics2026topological.analysis import ROOT
    from dubanevics2026topological.threshold.classification_probability_scans import (
        FDR_PRIMARY_TARGET,
        VARIANT_CUTOFFS,
    )
    from dubanevics2026topological.threshold.fdr_selection import table_name

    declaration = settings["analysis"]["clamped"]["candidate_functional_threshold"]
    if declaration.get("method") != SNAPSHOT_METHOD:
        raise ValueError("The classification snapshot must be the BY tables.")
    if declaration.get("q") != FDR_PRIMARY_TARGET:
        raise ValueError(
            "The classification snapshot q does not match the scan's nominal q."
        )
    # input_dir is processed/classification/<variant>; its sibling holds the other variant.
    base = (ROOT / str(declaration["input_dir"])).parent
    tables: dict[str, pl.DataFrame] = {}
    digest = hashlib.sha256()
    for variant, cutoff in VARIANT_CUTOFFS.items():
        directory = base / variant
        provenance_path = directory / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"No FDR classification at {directory}; run `just classification-probability-scans`, "
                "`just all` or `just fetch-data`."
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        record = {item["name"]: item for item in provenance["cutoffs"]}[cutoff]
        path = directory / table_name(
            float(record["applied_threshold"]),
            cutoff,
            SNAPSHOT_METHOD,
            FDR_PRIMARY_TARGET,
        )
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
        tables[cutoff] = pl.read_parquet(path)
    return {
        "tables": tables,
        "nominal_q": FDR_PRIMARY_TARGET,
        "product_id": digest.hexdigest(),
    }


def build_multiple_testing_comparison_figure(
    payload: Mapping[str, Any],
    settings: Mapping[str, Any],
    variant: str,
) -> alt.Chart:
    """Grouped bars: Functional calls by correction, one colour per threshold."""
    del variant
    frame = comparison_counts(payload["tables"], float(payload["nominal_q"]))
    rendering = settings["figures"]["rendering"]
    figure = settings["figures"]["figures"].get(FIGURE_ID, {})
    width = int(figure.get("width", 420))
    height = int(figure.get("height", rendering.get("tree_height", 250)))
    colours = [
        str(figure.get("published_color", COUNT_BLACK)),
        str(figure.get("candidate_color", MID_GREY)),
    ]
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(
                "method:N",
                sort=[label for _, label in METHODS],
                title="Correction",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("threshold:N", sort=list(THRESHOLD_LABELS.values())),
            y=alt.Y(
                "functional_count:Q",
                title="Functional calls",
                axis=alt.Axis(format="d"),
            ),
            color=alt.Color(
                "threshold:N",
                title=None,
                sort=list(THRESHOLD_LABELS.values()),
                scale=alt.Scale(domain=list(THRESHOLD_LABELS.values()), range=colours),
            ),
            tooltip=["threshold:N", "method:N", "functional_count:Q"],
        )
        .properties(
            width=width,
            height=height,
            title=figure_heading(
                "Functional calls",
                [
                    "Bonferroni, Benjamini-Hochberg and Benjamini-Yekutieli",
                    "at nominal q = 0.05",
                ],
            ),
        )
    )
    return chart.configure_axis(labelFontSize=int(rendering.get("font_size", 15)))


def _adjust(log_p: np.ndarray, method: str):
    from dubanevics2026topological.threshold.classification_probability_scans import (
        adjust_p_values_fdr,
    )

    return adjust_p_values_fdr(log_p, method=method)
