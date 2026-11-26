"""Sentinel manuscript quantities for the figure-label tests.

Every key a converted figure prints is given a display value that no synthetic input
can produce, made by the registry's own ``display_spec`` rule (comma-grouped counts,
six-decimal fitness, one-decimal percentages, an ungrouped seed), so the tests show
what the figures really print and a label the figure recomputed itself would fail
the comparison. ``SENTINEL[key]`` is the text
a figure must print for ``key``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from dubanevics2026topological.analysis.manuscript_quantities import (
    display_spec,
    value_text,
)
from dubanevics2026topological.figures._quantities import (
    ManuscriptQuantities,
    from_rows,
)

LANDMARKS = (
    "global_minimum",
    "split_trailing_saddle",
    "join_trailing_saddle",
    "median",
    "split_leading_saddle",
    "join_leading_saddle",
    "published_functional_threshold",
    "candidate_functional_threshold",
)
GRAPH_PEAK_KEYS = tuple(
    f"{graph}_graph.peak_count" for graph in ("full", "published", "candidate")
)
LABEL_KEYS = (
    *(f"leading_saddle_band.{name}.fitness" for name in LANDMARKS),
    "leading_saddle_band.total_sequence_count",
    "leading_saddle_band.band.count",
    "leading_saddle_band.band.percentage",
    "leading_saddle_band.median_inside_band",
    *GRAPH_PEAK_KEYS,
    "full_graph.vertex_count",
    *(
        f"{graph}_graph.{suffix}"
        for graph in ("published", "candidate")
        for suffix in (
            "functional_count",
            "vertex_count",
            "hill_sequence_count",
            "largest_hill_sequence_count",
        )
    ),
    "thresholds.published_functional.value",
    "thresholds.candidate_functional.value",
    "full_graph.persistence.ordinary_h0_pair_count",
    "full_graph.persistence.valley_count",
    "peak_uncertainty.measured_sequence_count",
    *(
        f"annotations.{graph}.{suffix}"
        for graph in ("published", "candidate")
        for suffix in (
            "stop_codon_peak_count",
            "synonymous_peak_count",
            "singleton_peak_count",
        )
    ),
    # The second batch: the codon-entropy ridges and the bootstrap figures.
    *(f"ridges.band_view.{ridge}.functional_count" for ridge in ("kem", "tcl")),
    "bootstrap.candidate.nominal.peak_count",
    "bootstrap.candidate.after_band_rule.peak_count",
    "bootstrap.candidate.total_absorbed_count",
    "bootstrap.candidate.after_band_rule.retained_sequence_count",
    "bootstrap.candidate.after_band_rule.saddles_removed_count",
    "bootstrap.draws",
    "bootstrap.seed",
    "bootstrap.band_quantile",
    "bootstrap.band_quantile_percent",
    "bootstrap.persistence_threshold.value",
    "bootstrap.persistence_threshold.radius",
    "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min",
    "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max",
    "bootstrap.global_stability.relative_wasserstein_1_median",
    *(
        f"bootstrap.global_stability.sanity_check.{name}"
        for name in (
            "nominal_landscape_draws_below_flat_fifth_percentile_percent",
            "flat_bottleneck_distance_fifth_percentile",
            "nominal_bottleneck_distance_median",
            "flat_bottleneck_distance_median",
        )
    ),
    # Appended so earlier sentinels keep their index.
    "leading_saddle_band.band.lower",
    "leading_saddle_band.band.upper",
)
# ``peak_uncertainty`` is not here: its sequence counts and largest components are its
# variant products' diagnostics (ADR 2026-11-19-peak-analysis-per-variant), not registry
# keys. ``peak_uncertainty.measured_sequence_count`` stays above only so that no later
# key's sentinel value moves; the registry stage still emits it.


def _unit(key: str) -> str:
    """The registry unit of a label key (the display rule is the unit's)."""
    if key.endswith("median_inside_band"):
        return "boolean"
    if key == "bootstrap.seed":
        return "seed"
    if key == "bootstrap.band_quantile":
        return "quantile"
    if key.endswith(("percentage", "_percent")):
        return "percent"
    if key.endswith("wasserstein_1_median"):
        return "fraction"
    if key.endswith(
        (
            ".fitness",
            ".value",
            ".radius",
            ".lower",
            ".upper",
            "_min",
            "_max",
            "_median",
            "fifth_percentile",
        )
    ):
        return "fitness"
    return "count"


def _sentinel(index: int, key: str) -> str:
    """A display text no synthetic input produces, shaped by the registry's own rule.

    The exact value goes through ``display_spec``, so a count carries its comma
    separators, a fitness its six decimals and a percentage its one decimal, as the
    exported registry shows them; a figure that reformats one would differ.
    """
    unit = _unit(key)
    exact = {
        "boolean": "true",
        "seed": f"{20261000 + index}",
        "quantile": f"0.7{index:05d}",
        "percent": f"{40 + index}.7",
        "fraction": f"0.7{index:05d}",
        "fitness": f"-0.5{index:05d}"
        if key.endswith((".fitness", ".value"))
        else f"0.7{index:05d}",
        "count": f"{1000 + 111 * index + 234}",
    }[unit]
    shown = display_spec(exact, unit)[1]
    assert shown is not None
    return shown


SENTINEL: dict[str, str] = {
    key: _sentinel(index, key) for index, key in enumerate(LABEL_KEYS)
}


# Every view ``label_quantities`` hands out, in creation order. The doc-versus-code
# drift test (``test_figure_label_registry_doc.py``) runs each figure's own label test
# and reads the keys the figure asked for from the views that test created.
CREATED: list[ManuscriptQuantities] = []


def scan_values(scan: pl.DataFrame, quantile: float = 0.95) -> dict[str, str]:
    """The exact registry values a scan's drawn marks must equal (see ``label_quantities``)."""
    prefix = "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_"
    row = scan.filter(pl.col("q") == quantile)
    if row.height != 1:
        return {}  # the figure refuses a scan without the grid point first
    return {
        "bootstrap.persistence_threshold.value": value_text(2.0 * float(row["c"][0])),
        "bootstrap.candidate.after_band_rule.peak_count": value_text(
            int(row["surviving_peaks"][0])
        ),
        prefix + "min": value_text(2.0 * float(scan["c"].min())),
        prefix + "max": value_text(2.0 * float(scan["c"].max())),
    }


def label_quantities(values: Mapping[str, str] | None = None) -> ManuscriptQuantities:
    """The registry view holding every label key with its sentinel display value.

    ``values`` gives a key's exact ``value`` text (the sentinels are display text
    only); a figure that compares a drawn number with the registry reads it.
    """
    quantities = from_rows(
        {"key": key, "display_value": shown, "value": (values or {}).get(key)}
        for key, shown in SENTINEL.items()
    )
    CREATED.append(quantities)
    return quantities


def with_labels(payload: dict[str, Any]) -> dict[str, Any]:
    """``payload`` plus the sentinel registry under ``manuscript_quantities``."""
    return {**payload, "manuscript_quantities": label_quantities()}


def spec_texts(spec: Any) -> list[str]:
    """Every string a Vega-Lite spec dict carries (titles, subtitles, text marks, data)."""
    found: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(spec)
    return found


def spec_text(spec: Any) -> str:
    """All of ``spec_texts`` joined with newlines, for substring checks."""
    return "\n".join(spec_texts(spec))
