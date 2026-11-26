"""The manuscript quantity registry: every reported number from one generated table.

``docs/manuscript-quantity-registry-design.md`` is the specification. This module
has two layers that never mix:

- **Pure row building** (``build_rows`` and the ``_topic`` functions): takes the
  upstream outputs as plain Python data (``Upstream``) and returns table rows. It
  reads no file, so a test feeds synthetic inputs.
- **Reading** (``load_sources``): reads the finished outputs of the upstream
  stages under an ``output`` directory into ``Upstream`` values. An upstream that
  is wholly absent becomes ``None`` and its quantities become named *gap rows*; a
  present file that is malformed or incomplete raises.

A quantity the stage cannot source is a gap row (``value`` and ``display_value``
null, ``gap_reason`` set). No number is ever typed in here: the only literals are
keys, units, rules and the display decimals of ``display_spec``.

Report and Table 1: ``render_report`` and ``read_display_values`` are the helpers
the build layer uses; the report is written at export time, outside the
checkpoint (see the design document).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from ..threshold.fdr_selection import threshold_summary
from ..threshold.functional_threshold import (
    DECLARATION_KEY,
    FDR_FIXED_CUTOFF,
    MIXTURE_QUANTILE,
    RULES,
)
from .codon_entropy_populations import lowest_common_ancestor
from .describe import describe
from .graph_summary import (
    SUMMARY_SCHEMA,
    UNCERTAIN_METRIC,
    split_tree_peak_count,
    split_tree_sequence_count,
)

STAGE_NAME = "manuscript-quantities"
TABLE_STEM = "manuscript_quantities"
REPORT_NAME = "manuscript_quantities_report.md"
TABLE_SUBDIRECTORY = "manuscript_quantities"

COLUMNS = (
    "key",
    "value",
    "display_decimals",
    "display_value",
    "unit",
    "population",
    "selection_rule",
    "settings",
    "source_stage",
    "source_fingerprint",
    "git_commit",
    "gap_reason",
)
SCHEMA = {
    "key": pl.String,
    "value": pl.String,
    "display_decimals": pl.Int64,
    "display_value": pl.String,
    "unit": pl.String,
    "population": pl.String,
    "selection_rule": pl.String,
    "settings": pl.String,
    "source_stage": pl.String,
    "source_fingerprint": pl.String,
    "git_commit": pl.String,
    "gap_reason": pl.String,
}
# Lowercase snake segments joined by dots, at least two. A segment starts with a
# letter; each underscore-separated word in it is at least two characters (a letter
# then letters or digits) or all digits (``aa_site_1``), so no word is a single
# letter and a bare ``x`` or ``g_b`` is rejected.
_WORD = r"(?:[a-z][a-z0-9]+|[0-9]+)"
_SEGMENT = rf"[a-z][a-z0-9]+(?:_+{_WORD})*"
# The one exception to "no single-letter word": the scan's acceptance checks are named a, b
# and c in the ADR (Decision 8) and the keys follow it (acceptance_a_passed and so on).
_SCAN_CHECK_KEY = r"functional_threshold_scan\.acceptance_[abc]_passed"
KEY_PATTERN = re.compile(rf"^(?:{_SEGMENT}(?:\.{_SEGMENT})+|{_SCAN_CHECK_KEY})$")
SETTINGS_SEPARATOR = ";"

# Every ``gap_reason`` starts with one of these kinds. ``stale:`` an upstream output
# exists but is not current; ``missing:`` the output is absent or no stage computes
# the quantity; ``undefined:`` the stage ran and the quantity has no value for its
# data (an empty population, no ordinary pair). The report header counts them.
STALE = "stale:"
MISSING = "missing:"
UNDEFINED = "undefined:"
GAP_KINDS = (STALE, MISSING, UNDEFINED)

# Upstream stage names, as recorded in ``source_stage``.
CLAMPED = "clamped-comparison"
COMPONENT = "component-analysis"
PEAK = "peak-uncertainty"
BOOTSTRAP = "bootstrap-smoothed-tree"
CODON = "codon-entropy"
ANNOTATIONS = "biological-annotation-tables"
# The saved per-state ``landscape_summary`` products (``build/landscape_summary_product.py``),
# pinned in ``[tables.landscape_summary.labels]`` of ``config/figures.toml``. The registry
# reads only the seven-column states the manuscript quotes: the Full library and the
# Published point-estimate graph. The Candidate graph (mixture_quantile) is not one of the
# seven states and keeps its own sources.
LANDSCAPE_SUMMARY = "landscape-summary-product"
CITED_STATES = {
    "full": "Full (nominal)",
    "published": "Published (point-estimate nominal)",
}
BRANCHES = "branch-decomposition"
BAND = "leading-saddle-band"
BAND_VIEW = "bootstrap-band-view"
FUNCTIONAL = "functional-classification"
THRESHOLD_SCAN = "functional-threshold-scan"
FULL_FITNESS = "full-fitness-table"
CONFIG = "config/analysis.toml"

# Where the upstream outputs live under ``output/``.
CLAMPED_DIR = ("analysis", "clamped_comparison")
COMPONENT_DIAGNOSTICS = ("analysis", "component_diagnostics.json")
COMPONENT_PAIRS = ("analysis", "component_pairs.parquet")
COMPONENT_PROVENANCE = ("analysis", "component_provenance.json")
PEAK_DIR = ("analysis", "peak_uncertainty")
BOOTSTRAP_SUMMARY = ("analysis", "bootstrap_smoothed_tree", "summary.json")
BOOTSTRAP_DRAWS = ("analysis", "bootstrap_smoothed_tree", "draw_stats.parquet")
BRANCH_DIR = ("analysis", "branch_decomposition")
TABLES_DIR = ("tables",)
CODON_METADATA = "papkou2023_codon_entropy_metadata.json"
CODON_CLAIMS = "papkou2023_codon_entropy_claim_summary.parquet"
CODON_SITE_ENTROPY = "papkou2023_codon_entropy_site_entropy.parquet"
CODON_SIDECAR = ".codon-entropy-provenance.json"
ANNOTATION_COUNTS = "biological_annotation_counts.json"
ANNOTATION_PAIRS = "biological_annotation_pairs.parquet"
ANNOTATION_SIDECAR = ".biological-annotation-provenance.json"
# The leading-saddle-band stage (``analysis/leading_saddle_band.py``): its checkpoint
# holds the three tables, ``summary.json`` and the helper's ``provenance.json``
# (fingerprint, artefacts); ``build.load_cached_leading_saddle_band`` reads it.
BAND_DIR = ("checkpoints", "leading_saddle_band")
# The functional-classification stage (``threshold/functional_classification.py``).
FUNCTIONAL_DIR = ("checkpoints", "functional_classification")
FUNCTIONAL_POPULATION = (
    "All sequences with wild-type-relative fitness (growth-rate refit)"
)
# The functional-threshold sensitivity scan (``analysis/functional_threshold_scan.py``): a
# tables-only export that is not a checkpoint. Its cells stay in its own tables; only the
# acceptance checks and the canonical interval are manuscript quantities.
SCAN_DIR = ("analysis", "functional_threshold_scan")
SCAN_READER = "functional_threshold_scan"
SCAN_PREFIX = "functional_threshold_scan."
SCAN_CHECKS = ("a", "b", "c")
# Row names under ``SCAN_PREFIX`` in manuscript order: the settings the checks use, the
# three checks and their conjunction, then the canonical interval and the two limits.
SCAN_KEYS = (
    "resample_count",
    "double_stop_below_minimum",
    "flip_fraction_limit",
    "acceptance_a_passed",
    "acceptance_b_passed",
    "acceptance_c_passed",
    "acceptance_all_passed",
    "canonical_threshold_interval_low",
    "canonical_threshold_interval_high",
    "canonical_threshold_interval_width",
    "interval_width_limit",
    "max_flip_fraction_one_step",
)
FUNCTIONAL_CLASS_RULE = "mixture_quantile"
FUNCTIONAL_PAPKOU_RULE = "Papkou k = 3 lowest-mean component"
BAND_SCHEMA = 1
# The command that refreshes each upstream, named in consistency errors.
RERUN = {
    CLAMPED: "just comparison-analysis",
    COMPONENT: "just component-analysis",
    PEAK: "just peak-analysis",
    BOOTSTRAP: "just bootstrap-smoothed-tree",
    BRANCHES: "just branch-decomposition",
    CODON: "just codon-entropy-analysis",
    BAND: "just leading-saddle-band",
    FUNCTIONAL: "just functional-classification",
    THRESHOLD_SCAN: "just functional-threshold-scan",
    BAND_VIEW: "just bootstrap-smoothed-tree",
    ANNOTATIONS: "just biological-annotation-tables",
    LANDSCAPE_SUMMARY: "just landscape-summary-product",
}
# Landmarks of the summary (``<key>_fitness``; ``<key>_sequence`` where it has one).
BAND_LANDMARKS = (
    "global_minimum",
    "split_trailing_saddle",
    "join_trailing_saddle",
    "median",
    "split_leading_saddle",
    "join_leading_saddle",
    "published_functional_threshold",
    "candidate_functional_threshold",
)
BAND_SEQUENCE_LANDMARKS = (
    "global_minimum",
    "split_trailing_saddle",
    "join_trailing_saddle",
    "split_leading_saddle",
    "join_leading_saddle",
)
# Band and trunks: (key segment, summary field prefix, rule).
BAND_POPULATIONS = (
    (
        "band",
        "band",
        "Full graph sequences between the lower and the upper trailing saddle (closed)",
    ),
    (
        "split_trunk",
        "split_trunk",
        "Full graph sequences at or below the split-tree trailing saddle (closed trunk)",
    ),
    (
        "join_trunk",
        "join_trunk",
        "Full graph sequences at or above the join-tree trailing saddle (closed trunk)",
    ),
)
# Candidate-threshold record name -> the band summary field that holds the same value.
BAND_THRESHOLD_FIELDS = {
    "rounded_threshold": "candidate_functional_threshold_fitness",
    "unrounded_threshold": "candidate_unrounded_threshold",
    "median": "median_fitness",
    "global_minimum": "global_minimum_fitness",
    "median_gap": "median_gap",
}
BAND_ENTROPY_RULE = (
    "sum over the three codon positions of the Shannon entropy in bits of the amino "
    "acid"
)

GRAPHS = ("full", "published", "candidate")
# The landscape key of the clamped comparison and the peak analysis for each graph.
LANDSCAPE_KEYS = {"published": "study_clamped", "candidate": "candidate_clamped"}
GRAPH_LABELS = {"full": "Full", "published": "Published", "candidate": "Candidate"}
# Table 1 metric label -> (key suffix, unit); the clamped graph summary uses the
# same labels.
SUMMARY_METRICS: dict[str, tuple[str, str]] = {
    "Number of nodes": ("vertex_count", "count"),
    "Number of edges": ("edge_count", "count"),
    "Minimum fitness": ("min_fitness", "fitness"),
    "Maximum fitness": ("max_fitness", "fitness"),
    "Mean fitness": ("mean_fitness", "fitness"),
    "Median fitness": ("median_fitness", "fitness"),
    "Number of functional sequences": ("functional_count", "count"),
    "Number of non-functional sequences": ("nonfunctional_count", "count"),
    "Percentage of functional sequences": ("functional_percent", "percent"),
}
GRAPH_RULES = {
    "full": "no threshold: every sequence with a measured raw fitness",
    "published": (
        "raw fitness >= published threshold; edges with at least one such "
        "endpoint; largest connected component"
    ),
    "candidate": (
        "median_reflected: raw fitness >= candidate threshold; edges with at least "
        "one such endpoint; largest connected component"
    ),
}
# The Candidate graph's rule when the declared Selection rule is mixture_quantile:
# the Functional class of the functional-classification stage selects the vertices.
CANDIDATE_CLASS_RULE = (
    "mixture_quantile: Functional class (Functional probability at or above the "
    "declared confidence level); edges with at least one Functional endpoint; "
    "largest connected component"
)
# Selection rules under which a Functional class, not raw fitness against T, selects the Candidate graph.
CLASS_RULES = (MIXTURE_QUANTILE, FDR_FIXED_CUTOFF)
FDR_CUTOFF_DESCRIPTIONS = {
    "papkou_original": "the Papkou et al. original cutoff, reconstructed",
    "median_reflected": "the historical applied median-reflected cutoff",
}
THRESHOLD_KEY = {
    "published": "thresholds.published_functional.value",
    "candidate": "thresholds.candidate_functional.value",
}
NOT_CROSS_CHECKED = "; b was not cross-checked: the component analysis output is absent"
FUNCTIONAL_RULE = "raw fitness at or above the threshold (>=)"
# The Candidate graph's functional counts under mixture_quantile: the class decides,
# so the count is of vertices of the Candidate graph in a class of the
# functional-classification stage, never raw fitness against T.
CANDIDATE_CLASS_COUNT_RULES = {
    "Number of functional sequences": (
        "vertices of the Candidate graph in the Functional class "
        "(mixture_quantile), not raw fitness at or above the threshold"
    ),
    "Number of non-functional sequences": (
        "vertices of the Candidate graph in the Non-functional class "
        "(mixture_quantile); the Uncertain class is counted separately"
    ),
    "Percentage of functional sequences": (
        "vertices of the Candidate graph in the Functional class (mixture_quantile) "
        "as a percentage of all its vertices"
    ),
    UNCERTAIN_METRIC: (
        "vertices of the Candidate graph in the Uncertain class (mixture_quantile): "
        "neither Functional nor Non-functional at the declared confidence level"
    ),
}
# The GLOSSARY definition of Peak: "A local fitness maximum in the declared graph. The
# peak count of a graph equals the number of maxima (leaves) of its split tree.
# Equal-fitness neighbours form one split-tree leaf; ties never create extra peaks."
PEAK_RULE = (
    "local fitness maximum in the declared graph; the peak count equals the number "
    "of maxima (leaves) of its split tree; equal-fitness neighbours form one "
    "split-tree leaf and ties never create extra peaks (GLOSSARY: Peak)"
)
FULL_PEAK_RULE = (
    "maxima of the Full graph's split tree (a vertex with no neighbour of higher "
    "raw fitness)"
)
# The landscape key of the clamped comparison for each graph that is annotated.
EVIDENCE_KEYS = ("study_clamped", "candidate_clamped")

# Topic order of design section 3, with the keys of each topic.
TOPICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Functional thresholds", re.compile(r"^thresholds\.")),
    (
        "Graph sizes",
        re.compile(
            r"^(full|published|candidate)_graph\."
            r"(vertex_count|edge_count|component_count|largest_component_vertex_count"
            r"|min_fitness|max_fitness|mean_fitness|median_fitness)$"
        ),
    ),
    (
        "Functional populations",
        re.compile(
            r"^(published|candidate)_graph\.(functional|nonfunctional)_"
            r"|^published_graph\.full_library_"
        ),
    ),
    (
        "Peak counts",
        re.compile(
            r"^(full|published|candidate)_graph\.(peak_count|leaf_count)$"
            r"|^bootstrap\.candidate\."
        ),
    ),
    (
        "Bootstrapped persistence threshold",
        re.compile(r"^bootstrap\.(draws|seed|band_quantile|persistence_threshold)"),
    ),
    (
        "Full graph persistence",
        re.compile(r"^full_graph\.persistence\."),
    ),
    (
        "Hills",
        re.compile(
            r"^(published|candidate)_graph\."
            r"(hill_sequence_count|largest_hill_sequence_count)$"
        ),
    ),
    ("Bootstrap global stability", re.compile(r"^bootstrap\.global_stability\.")),
    ("Peak uncertainty", re.compile(r"^peak_uncertainty\.")),
    ("Summits and common saddle", re.compile(r"^(summits|common_saddle)\.")),
    ("Ridge populations and positional entropy", re.compile(r"^(ridges|entropy)\.")),
    ("Annotations", re.compile(r"^annotations\.")),
    ("Branches", re.compile(r"^branches\.")),
    ("Leading saddle band", re.compile(r"^leading_saddle_band\.")),
    ("Functional classification", re.compile(r"^functional_classification\.")),
    (
        "Functional threshold sensitivity scan",
        re.compile(r"^functional_threshold_scan\."),
    ),
)
OTHER_TOPIC = "Other quantities"

_SUMMIT_KEYS = ("dna_sequence", "aa_sequence", "raw_fitness")
_RIDGE_KEYS = ("kem", "tcl", "union")
_ENTROPY_POPULATIONS = (
    ("core", "candidate"),
    ("kem", "kem_ridge"),
    ("tcl", "tcl_ridge"),
)
_ENTROPY_GROUPS = (("all", "all_sequences"), ("functional", "functional_sequences"))
_ANNOTATION_GRAPHS = (
    "published",
    "candidate",
    "candidate_bootstrap_core",
    "candidate_bootstrap_band",
)
_GRAPH_SUFFIXES = (
    "vertex_count",
    "edge_count",
    "component_count",
    "largest_component_vertex_count",
    "peak_count",
    "min_fitness",
    "max_fitness",
    "mean_fitness",
    "median_fitness",
)
_FUNCTIONAL_SUFFIXES = (
    "functional_count",
    "nonfunctional_count",
    "functional_denominator",
    "functional_percent",
)
# The saved tree's leaf count of the cited states, beside the split-tree peak count.
LEAF_COUNT_KEYS = ("full_graph.leaf_count", "published_graph.leaf_count")
# Published point-estimate classification over the Full library (the denominator is the
# Full library, not the selected graph): (key suffix, saved metric, unit).
FULL_LIBRARY_ROWS = (
    (
        "full_library_functional_count",
        "Full library functional sequences",
        "count",
    ),
    (
        "full_library_functional_percent",
        "Full library functional sequences (%)",
        "percent",
    ),
    (
        "full_library_nonfunctional_count",
        "Full library non-functional sequences",
        "count",
    ),
    (
        "full_library_nonfunctional_percent",
        "Full library non-functional sequences (%)",
        "percent",
    ),
)
FULL_LIBRARY_KEYS = tuple(
    f"published_graph.{suffix}" for suffix, _, _ in FULL_LIBRARY_ROWS
)
_RIDGE_SUFFIXES = ("total_count", "functional_count", "nonfunctional_count")
_ANNOTATION_SUFFIXES = (
    "stop_codon_peak_count",
    "stop_codon_peak_denominator",
    "synonymous_peak_count",
    "synonymous_peak_denominator",
    "synonymous_partner_peak_count",
    "synonymous_partner_peak_denominator",
    "singleton_peak_count",
    "singleton_peak_denominator",
)

# Global-stability quantities of the bootstrap: (key, unit) in manuscript order. The
# values come from ``summary["diagnostics"]["global_verdict"]``.
GLOBAL_STABILITY_KEYS = (
    "bootstrap.global_stability.relative_wasserstein_1_median",
    "bootstrap.global_stability.sanity_check.nominal_bottleneck_distance_median",
    "bootstrap.global_stability.sanity_check.flat_bottleneck_distance_median",
    "bootstrap.global_stability.sanity_check.flat_bottleneck_distance_fifth_percentile",
    (
        "bootstrap.global_stability.sanity_check."
        "nominal_landscape_draws_below_flat_fifth_percentile_percent"
    ),
)

# Manuscript order of the "Manuscript numbers" list: the topic order above, one
# explicit constant. Keys the stage emits that are not listed are appended.
MANUSCRIPT_ORDER: tuple[str, ...] = (
    "thresholds.published_functional.value",
    "thresholds.candidate_functional.value",
    "thresholds.candidate_functional.unrounded_value",
    "thresholds.full_median.value",
    "thresholds.full_global_minimum.value",
    "thresholds.median_gap.value",
    "thresholds.join_tree_trailing_saddle.value",
    "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value",
    *(
        f"{graph}_graph.{suffix}"
        for graph in GRAPHS
        for suffix in _GRAPH_SUFFIXES
        if suffix != "peak_count"
    ),
    *(
        f"{graph}_graph.{suffix}"
        for graph in ("published", "candidate")
        for suffix in _FUNCTIONAL_SUFFIXES
    ),
    *(f"{graph}_graph.peak_count" for graph in GRAPHS),
    *LEAF_COUNT_KEYS,
    *FULL_LIBRARY_KEYS,
    "full_graph.persistence.ordinary_h0_pair_count",
    "full_graph.persistence.valley_count",
    *(
        f"{graph}_graph.{suffix}"
        for graph in ("published", "candidate")
        for suffix in ("hill_sequence_count", "largest_hill_sequence_count")
    ),
    "peak_uncertainty.measured_sequence_count",
    *(
        f"bootstrap.candidate.{state}.{suffix}"
        for state, suffixes in (
            ("nominal", ("peak_count",)),
            ("after_core_rule", ("peak_count", "absorbed_count")),
            (
                "after_band_rule",
                (
                    "peak_count",
                    "absorbed_count",
                    "retained_sequence_count",
                    "saddles_removed_count",
                ),
            ),
        )
        for suffix in suffixes
    ),
    "bootstrap.candidate.total_absorbed_count",
    "bootstrap.draws",
    "bootstrap.seed",
    "bootstrap.band_quantile",
    "bootstrap.band_quantile_percent",
    "bootstrap.persistence_threshold.radius",
    "bootstrap.persistence_threshold.value",
    "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min",
    "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max",
    *GLOBAL_STABILITY_KEYS,
    *(
        f"{who}.{field_name}"
        for who in ("summits.kem", "summits.tcl", "common_saddle")
        for field_name in _SUMMIT_KEYS
    ),
    *(f"ridges.{who}.{suffix}" for who in _RIDGE_KEYS for suffix in _RIDGE_SUFFIXES),
    *(f"ridges.band_view.{ridge}.functional_count" for ridge in ("kem", "tcl")),
    # Per-position entropy is entropy_by_position.parquet, not a registry row.
    "ridges.two_functional_share.numerator",
    "ridges.two_functional_share.denominator",
    "ridges.two_functional_share.fraction",
    "ridges.two_functional_share.minimum",
    "ridges.two_functional_share.meets_minimum",
    *(
        f"annotations.{graph}.{suffix}"
        for graph in _ANNOTATION_GRAPHS
        for suffix in _ANNOTATION_SUFFIXES
    ),
    "branches.terminal_count",
    "branches.internal_count",
    "branches.terminal_sequence_count",
    "branches.internal_sequence_count",
    *(f"leading_saddle_band.{landmark}.fitness" for landmark in BAND_LANDMARKS),
    *(
        f"leading_saddle_band.{landmark}.sequence"
        for landmark in BAND_SEQUENCE_LANDMARKS
    ),
    "leading_saddle_band.median_gap",
    "leading_saddle_band.candidate_unrounded_threshold",
    "leading_saddle_band.candidate_threshold_rule",
    "leading_saddle_band.candidate_threshold_decimals",
    "leading_saddle_band.total_sequence_count",
    "leading_saddle_band.band.lower_landmark",
    "leading_saddle_band.band.lower",
    "leading_saddle_band.band.upper",
    "leading_saddle_band.median_inside_band",
    *(
        f"leading_saddle_band.{key}.{suffix}"
        for key, _, _ in BAND_POPULATIONS
        for suffix in ("count", "percentage", "entropy_sum")
    ),
    "leading_saddle_band.below_first_landmark_count",
    "leading_saddle_band.above_last_landmark_count",
    "leading_saddle_band.interval_convention",
    *(
        f"functional_classification.{name}"
        for name in (
            "mixture_quantile_functional_threshold_fitness",
            "mixture_quantile_functional_threshold_unrounded_fitness",
            "mixture_component_count",
            "functional_class_count",
            "non_functional_class_count",
            "uncertain_class_count",
            "papkou_replication_functional_threshold_fitness",
            "double_stop_fraction_below_threshold",
        )
    ),
    *(f"{SCAN_PREFIX}{name}" for name in SCAN_KEYS),
)


# --------------------------------------------------------------------------
# Display rules
# --------------------------------------------------------------------------

# Decimals of a displayed quantity, by unit; counts and seeds are integers.
UNIT_DECIMALS = {
    "count": 0,
    "seed": 0,
    "percent": 1,
    "fitness": 6,
    "fraction": 4,
    "bits": 4,
}
# Units shown exactly as stored.
VERBATIM_UNITS = frozenset(
    {"boolean", "dna_sequence", "aa_sequence", "quantile", "text"}
)
# Role overrides: a summit's and the common saddle's fitness shows two decimals.
ROLE_DECIMALS = {"summit_fitness": 2}
VERBATIM_ROLE = "verbatim"


def value_text(value: Any) -> str:
    """The exact text of a value: ints as ints, floats by ``repr``, booleans lowercase."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"A registry value must be finite, got {value!r}.")
        return repr(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported registry value type {type(value).__name__}.")


def display_spec(
    value: str | None, unit: str, role: str | None = None
) -> tuple[int | None, str | None]:
    """The (decimals, display text) of an exact value: the one place rules live.

    Numbers round half up (away from zero on a tie) from the exact decimal text
    of ``value``, never by truncation and never through a binary float. Counts
    and seeds must be integers and are shown unrounded; a count carries comma
    thousands separators ("262,144"), a seed does not. Booleans, sequences,
    quantiles and any ``role="verbatim"`` quantity are shown as stored.

    ``display_value`` is the complete printable number: rounding, separators and
    sign live in it. A unit symbol ("%", "draws") does not; the consumer appends it
    from ``unit``.
    """
    if value is None:
        return None, None
    if role == VERBATIM_ROLE or unit in VERBATIM_UNITS:
        return None, value
    if role in ROLE_DECIMALS:
        decimals = ROLE_DECIMALS[role]
    elif unit in UNIT_DECIMALS:
        decimals = UNIT_DECIMALS[unit]
    else:
        raise ValueError(f"No display rule for unit {unit!r}.")
    if unit in {"count", "seed"}:
        if not re.fullmatch(r"-?\d+", value):
            raise ValueError(
                f"A {unit} must be an integer, got {value!r} (not even an integral "
                "float is accepted: the upstream value has the wrong type)."
            )
        return 0, f"{int(value):,}" if unit == "count" else str(int(value))
    exact = Decimal(value)
    rounded = exact.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    text = format(rounded, "f")
    if rounded == 0 and text.startswith("-"):
        text = text[1:]
    return decimals, text


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Raise unless every key matches ``KEY_PATTERN`` and no key repeats."""
    seen: set[str] = set()
    for row in rows:
        key = row["key"]
        if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
            raise ValueError(
                f"Registry key {key!r} is not dotted lowercase snake_case segments "
                "(no segment or underscore-separated word may be a single letter)."
            )
        if key in seen:
            raise ValueError(f"Registry key {key!r} appears more than once.")
        seen.add(key)


def topic_of(key: str) -> str:
    """The report topic a key belongs to."""
    for heading, pattern in TOPICS:
        if pattern.search(key):
            return heading
    return OTHER_TOPIC


# --------------------------------------------------------------------------
# Upstream values and the row builder
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Upstream:
    """One upstream output: its stage, its fingerprint text and its parsed data."""

    stage: str
    fingerprint: str | None
    data: Mapping[str, Any] = field(default_factory=dict)
    # What a downstream stage fingerprint keys on when more than the fingerprint
    # text identifies the output (a checkpoint's fingerprint and artefact hashes).
    identity: Any = None
    # sha256 of every file the loader read, keyed by path under ``output/``: a
    # changed file changes the registry fingerprint even when the upstream's own
    # fingerprint is missing or was reused.
    files: Mapping[str, str] = field(default_factory=dict)
    # The loader's own result, kept for a source derived from two upstreams.
    raw: Any = field(default=None, repr=False, compare=False)


def fingerprint_text(value: Any) -> str | None:
    """A fingerprint as text: a string as is, any other JSON value by sha256."""
    if value is None or isinstance(value, str):
        return value
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def combine(*parts: Upstream | None) -> Upstream | None:
    """A derived row's source: the stages and fingerprints joined, or ``None``."""
    if any(part is None for part in parts):
        return None
    present = [part for part in parts if part is not None]
    return Upstream(
        "+".join(part.stage for part in present),
        "+".join(part.fingerprint or "none" for part in present),
    )


def _dig(upstream: Upstream | None, *path: str) -> Any:
    """Nested data of ``upstream``; ``None`` when absent; ``KeyError`` when malformed."""
    if upstream is None:
        return None
    current: Any = upstream.data
    for step in path:
        try:
            current = current[step]
        except (KeyError, TypeError, IndexError) as error:
            raise KeyError(
                f"{upstream.stage} output lacks {'.'.join(path)!r}; it is incomplete "
                "or from an incompatible version."
            ) from error
    return current


def strip_gap_kind(reason: str) -> str:
    """``reason`` without its leading kind prefix."""
    for kind in GAP_KINDS:
        if reason.startswith(kind):
            return reason[len(kind) :].lstrip()
    return reason


def stage_not_run_reason(stage: str) -> str:
    """The one canonical ``missing:`` reason for a stage whose output does not exist.

    Only an input stage that has not run (or whose output is wholly absent) gets this
    text; ``missing:`` also covers a quantity no stage computes and absent raw
    tables, which are not awaiting a run. ``is_stage_not_run`` is its one predicate.
    """
    names = [name for name in stage.split("+") if not name.startswith("config/")]
    return f"{MISSING} stage {'+'.join(names) or stage} not run"


def is_stage_not_run(reason: str) -> bool:
    """Whether a gap reason is a ``stage_not_run_reason`` (the output awaits a run)."""
    return (
        re.match(rf"^{re.escape(MISSING)} stage \S+ not run(?:$|[:;])", reason)
        is not None
    )


def gap_reason(reason: str | None, stage: str) -> str:
    """``reason`` with a kind prefix; an unprefixed one is ``missing:``.

    No reason at all means the stage's output is absent: the stage has not run.
    """
    if not reason:
        return stage_not_run_reason(stage)
    return reason if reason.startswith(GAP_KINDS) else f"{MISSING} {reason}"


class _Rows:
    """Collects registry rows; the one place a row is assembled."""

    def __init__(
        self, git_commit: str | None, absent: Mapping[str, str] | None = None
    ) -> None:
        self.git_commit = git_commit
        # Gap reasons of upstreams whose cached loader found no current checkpoint,
        # keyed by stage name.
        self.absent = dict(absent or {})
        self.rows: list[dict[str, Any]] = []
        # The declared Selection rule of the Candidate graph (``build_rows`` sets it).
        self.candidate_rule = MIXTURE_QUANTILE
        # The declaration itself: the FDR rule's cutoff name, method and q come from it.
        self.candidate_declaration: Mapping[str, Any] = {}

    @property
    def by_class(self) -> bool:
        """Whether a Functional class (not raw fitness against T) selects the Candidate graph."""
        return self.candidate_rule in CLASS_RULES

    @property
    def class_tag(self) -> str:
        """The parenthesised rule tag the class-based row texts carry."""
        if self.candidate_rule != FDR_FIXED_CUTOFF:
            return "(mixture_quantile)"
        d = self.candidate_declaration
        return f"(fdr_fixed_cutoff, {str(d['method']).upper()}, nominal q = {d['q']:g})"

    def candidate_threshold_rule(self, *, unrounded: bool = False) -> str:
        """The ``selection_rule`` text of the Candidate threshold row (rounded or unrounded)."""
        if self.candidate_rule == FDR_FIXED_CUTOFF:
            d = self.candidate_declaration
            cutoff = str(d.get("cutoff", ""))
            desc = FDR_CUTOFF_DESCRIPTIONS.get(cutoff)
            tag = f"{cutoff}: {desc}" if desc else cutoff
            base = (
                f"Fixed Candidate cutoff T declared by the fdr_fixed_cutoff variant "
                f"({tag}); not refitted"
            )
            return f"{base}, before rounding" if unrounded else base
        if unrounded:
            return "median plus the Median gap, before rounding"
        return (
            "median plus the Median gap of the Full graph raw fitness, rounded to the "
            "configured decimals (ADR 2026-10-23)"
        )

    def class_text(self, text: str) -> str:
        """``text`` written for the mixture rule, with its tag and semantics adapted for the active rule."""
        if self.candidate_rule == FDR_FIXED_CUTOFF:
            d = self.candidate_declaration
            method = str(d.get("method", "by")).upper()
            q = d.get("q", 0.05)
            if text == CANDIDATE_CLASS_COUNT_RULES[UNCERTAIN_METRIC]:
                return (
                    f"vertices of the Candidate graph in the Uncertain class {self.class_tag}: "
                    f"at or above cutoff T but not retained by the full-library {method} "
                    f"adjustment at nominal q = {q:g}; excluded from the retained Functional set"
                )
            if (
                text
                == CANDIDATE_CLASS_COUNT_RULES["Number of non-functional sequences"]
            ):
                return (
                    f"vertices of the Candidate graph in the Non-functional class {self.class_tag}: "
                    "raw fitness below cutoff T (point-estimate category; not "
                    "statistically confirmed nonfunctional)"
                )
        return text.replace("(mixture_quantile)", self.class_tag)

    def graph_rule(self, graph: str) -> str:
        """The ``selection_rule`` text of one graph's rows."""
        if graph == "candidate" and self.candidate_rule == FDR_FIXED_CUTOFF:
            d = self.candidate_declaration
            threshold = threshold_summary(d)["threshold"]["rounded"]
            return (
                "fdr_fixed_cutoff: retained Functional class from reviewed full-library "
                f"{str(d['method']).upper()} adjustment at nominal q = {d['q']:g}, cutoff "
                f"T = {threshold!r} ({d['cutoff']}); edges with at least one retained "
                "Functional endpoint; largest connected component"
            )
        if graph == "candidate" and self.candidate_rule == MIXTURE_QUANTILE:
            return CANDIDATE_CLASS_RULE
        return GRAPH_RULES[graph]

    def _absent_reason(self, stage: str) -> str | None:
        """The loader-supplied reason(s) for the stages in ``stage``, if any.

        Joined stages give one reason whose kind is ``stale:`` when any of them is
        stale, else ``missing:``.
        """
        reasons = [
            self.absent[name] for name in stage.split("+") if name in self.absent
        ]
        if len(reasons) < 2:
            return reasons[0] if reasons else None
        kind = STALE if any(r.startswith(STALE) for r in reasons) else MISSING
        return f"{kind} " + "; ".join(strip_gap_kind(r) for r in reasons)

    def put(
        self,
        key: str,
        value: Any,
        unit: str,
        population: str,
        selection_rule: str,
        *,
        up: Upstream | None,
        stage: str,
        reason: str | None = None,
        settings: Iterable[str] = (),
        role: str | None = None,
    ) -> None:
        """Add a sourced row, or a gap row when ``up`` is absent or ``value`` is None."""
        settings_text = SETTINGS_SEPARATOR.join(settings) or None
        gap = up is None or value is None
        if gap and reason is None and up is None:
            reason = self._absent_reason(stage)
        text = None if gap else value_text(value)
        decimals, shown = display_spec(text, unit, role)
        self.rows.append(
            {
                "key": key,
                "value": text,
                "display_decimals": decimals,
                "display_value": shown,
                "unit": unit,
                "population": population,
                "selection_rule": selection_rule,
                "settings": settings_text,
                "source_stage": up.stage if up is not None else stage,
                "source_fingerprint": up.fingerprint if up is not None else None,
                "git_commit": self.git_commit,
                "gap_reason": gap_reason(reason, stage) if gap else None,
            }
        )


# --------------------------------------------------------------------------
# Topics
# --------------------------------------------------------------------------


def _config_upstream(config: Mapping[str, Any]) -> Upstream:
    return Upstream(CONFIG, None, config)


def _thresholds(
    rows: _Rows, sources: Mapping[str, Upstream | None], config: Mapping[str, Any]
) -> None:
    clamped, component = sources.get("clamped"), sources.get("component")
    band = sources.get("band")
    published = _config_upstream(config)
    candidate = _dig(clamped, "candidate_threshold")

    def threshold(name: str) -> Any:
        """g, m, h and T: the band stage's value when it ran, else the clamped record.

        Both record the same numbers (``check_consistency`` raises when they differ,
        exactly), so the band is the single source wherever it provides them.
        """
        if band is not None:
            return _dig(band, "summary", BAND_THRESHOLD_FIELDS[name])
        return None if candidate is None else candidate[name]

    threshold_up, threshold_stage = (
        (band, BAND) if band is not None else (clamped, CLAMPED)
    )

    if candidate is not None:
        _require_keys(
            candidate, ("median", "global_minimum", "median_gap", "rounded_threshold")
        )
    rows.put(
        "thresholds.published_functional.value",
        config["clamped"]["published_functional_threshold"],
        "fitness",
        "Full graph raw fitness",
        "published functional threshold of Papkou et al. 2023",
        up=published,
        stage=CONFIG,
    )
    candidate_text = rows.candidate_threshold_rule(unrounded=False)
    unrounded_text = rows.candidate_threshold_rule(unrounded=True)
    specs = (
        ("thresholds.candidate_functional.value", "rounded_threshold", candidate_text),
        (
            "thresholds.candidate_functional.unrounded_value",
            "unrounded_threshold",
            unrounded_text,
        ),
        (
            "thresholds.full_median.value",
            "median",
            "median of the Full graph raw fitness",
        ),
        (
            "thresholds.full_global_minimum.value",
            "global_minimum",
            "minimum of the Full graph raw fitness (g)",
        ),
        (
            "thresholds.median_gap.value",
            "median_gap",
            "median minus the global minimum (the Median gap)",
        ),
    )
    for key, name, rule in specs:
        rows.put(
            key,
            threshold(name),
            "fitness",
            "Full graph raw fitness",
            rule,
            up=threshold_up,
            stage=threshold_stage,
        )
    # The band's b_join is checked against the component analysis in
    # ``check_consistency`` whenever the component output is present; when it is
    # absent the value is kept and the rows that use it say it was not checked.
    unchecked = NOT_CROSS_CHECKED if band is not None and component is None else ""
    if band is not None:
        saddle = _dig(band, "summary", "join_trailing_saddle_fitness")
        saddle_up, saddle_stage = band, BAND
    else:
        saddle = _dig(component, "diagnostics", "join_tree_trailing_saddle")
        saddle_up, saddle_stage = component, COMPONENT
    rows.put(
        "thresholds.join_tree_trailing_saddle.value",
        saddle,
        "fitness",
        "Full graph join tree",
        "maximum death of the ordinary join-tree pairs (b); reported, not an input "
        "to the Candidate threshold" + unchecked,
        up=saddle_up,
        stage=saddle_stage,
        reason=None
        if saddle_up is None
        else f"{UNDEFINED} the join tree has no ordinary pair",
    )
    ordering = None
    ordering_up = combine(clamped, component)
    ordering_stage = f"{CLAMPED}+{COMPONENT}"
    if band is not None:
        # g, b (= b_join), m and T all come from the band summary.
        ordering_up, ordering_stage = band, BAND
        values = (
            threshold("global_minimum"),
            saddle,
            threshold("median"),
            threshold("rounded_threshold"),
        )
        if all(value is not None for value in values):
            g, b, m, t = (float(value) for value in values)
            ordering = bool(g < b < m < t)
    elif candidate is not None and saddle is not None:
        ordering = bool(
            float(candidate["global_minimum"])
            < float(saddle)
            < float(candidate["median"])
            < float(candidate["rounded_threshold"])
        )
    ordering_reason = None
    if saddle is None and (
        band is not None or (clamped is not None and component is not None)
    ):
        ordering_reason = (
            f"{UNDEFINED} the join tree has no ordinary pair, so its trailing saddle "
            "b is undefined"
        )
    rows.put(
        "thresholds.ordering_minimum_join_trailing_saddle_median_candidate.value",
        ordering,
        "boolean",
        "Full graph raw fitness",
        "global minimum g < join-tree trailing saddle b < median < candidate threshold T"
        + unchecked,
        up=ordering_up,
        stage=ordering_stage,
        reason=ordering_reason,
        settings=(
            "thresholds.full_global_minimum.value",
            "thresholds.join_tree_trailing_saddle.value",
            "thresholds.full_median.value",
            "thresholds.candidate_functional.value",
        ),
    )


def _require_keys(mapping: Mapping[str, Any], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise KeyError(f"Upstream record lacks {missing}; it is incomplete.")


def _graphs(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    clamped, peak = sources.get("clamped"), sources.get("peak")
    summary = _dig(clamped, "graph_summary", "metrics")
    if (
        clamped is not None
        and _dig(clamped, "graph_summary", "schema") != SUMMARY_SCHEMA
    ):
        raise ValueError(
            f"The clamped comparison graph summary has schema "
            f"{_dig(clamped, 'graph_summary', 'schema')!r}, expected {SUMMARY_SCHEMA}; "
            f"rerun `{RERUN[CLAMPED]}`."
        )
    diagnostics = _dig(clamped, "diagnostics")
    peak_counts = _dig(peak, "diagnostics", "peak_counts")
    for graph in GRAPHS:
        label = GRAPH_LABELS[graph]
        population = f"{label} graph"
        rule = rows.graph_rule(graph)
        threshold = (THRESHOLD_KEY[graph],) if graph in THRESHOLD_KEY else ()
        metrics = None if summary is None else summary[label]
        for metric, (suffix, unit) in SUMMARY_METRICS.items():
            if suffix.startswith(("functional", "nonfunctional")):
                continue
            rows.put(
                f"{graph}_graph.{suffix}",
                None if metrics is None else metrics[metric],
                unit,
                population,
                rule,
                up=clamped,
                stage=CLAMPED,
            )
        landscape_key = LANDSCAPE_KEYS.get(graph)
        landscape = (
            None
            if diagnostics is None or landscape_key is None
            else _dig(clamped, "diagnostics", landscape_key)
        )
        if graph == "full":
            # The Full graph's component rows come from the saved Full landscape
            # summary (``_landscape_summary``).
            rows.put(
                "full_graph.peak_count",
                _dig(clamped, "full_peak_count"),
                "count",
                population,
                FULL_PEAK_RULE,
                up=clamped,
                stage=CLAMPED,
                reason=None
                if clamped is None
                else f"{MISSING} the clamped comparison result carries no Full analysis",
            )
            continue
        by_class = graph == "candidate" and rows.by_class
        prefix = f"{rows.candidate_rule}: " if graph == "candidate" else ""
        component_rule = prefix + (
            "connected components of the graph of edges with at least one "
            + (
                "Functional-class endpoint"
                if by_class
                else f"endpoint at or above the {graph} threshold"
            )
        )
        largest_rule = prefix + (
            "vertices of the largest of those components (ties: lexicographically "
            "smallest minimum DNA sequence)"
        )
        rows.put(
            f"{graph}_graph.component_count",
            None if landscape is None else landscape["component_count"],
            "count",
            population,
            component_rule,
            up=clamped,
            stage=CLAMPED,
            settings=threshold,
        )
        largest = (
            None if landscape is None else landscape["selected_component_vertex_count"]
        )
        if (
            largest is not None
            and metrics is not None
            and largest != metrics["Number of nodes"]
        ):
            raise ValueError(
                f"{label} graph: the largest component has {largest} vertices but "
                f"the graph summary counts {metrics['Number of nodes']}."
            )
        rows.put(
            f"{graph}_graph.largest_component_vertex_count",
            largest,
            "count",
            population,
            largest_rule,
            up=clamped,
            stage=CLAMPED,
            settings=threshold,
        )
        peaks = None if landscape is None else landscape["peak_count"]
        if peaks is not None and peak_counts is not None:
            other = peak_counts[LANDSCAPE_KEYS[graph]]
            if other != peaks:
                raise ValueError(
                    f"{label} graph peak count differs between the clamped comparison "
                    f"({peaks}) and the peak uncertainty analysis ({other})."
                )
        rows.put(
            f"{graph}_graph.peak_count",
            peaks,
            "count",
            population,
            PEAK_RULE,
            up=clamped,
            stage=CLAMPED,
            settings=threshold,
        )
        hills = _dig(clamped, "evidence", landscape_key)
        hill_rule = (
            "sequences on the terminal branches (hills) of the split tree: each peak "
            "with the members of its terminal branch, the adjacent saddle excluded"
        )
        for suffix, field_name, rule in (
            ("hill_sequence_count", "hill_sequence_count", hill_rule),
            (
                "largest_hill_sequence_count",
                "largest_hill_sequence_count",
                "sequences of the largest hill: " + hill_rule,
            ),
        ):
            rows.put(
                f"{graph}_graph.{suffix}",
                None if hills is None else hills["hills"][field_name],
                "count",
                population,
                rule,
                up=clamped,
                stage=CLAMPED,
                settings=threshold,
            )
        # Functional populations: counts, denominator and percentage.
        for metric in (
            "Number of functional sequences",
            "Number of non-functional sequences",
            "Percentage of functional sequences",
        ):
            suffix, unit = SUMMARY_METRICS[metric]
            rows.put(
                f"{graph}_graph.{suffix}",
                None if metrics is None else metrics[metric],
                unit,
                population,
                rows.class_text(CANDIDATE_CLASS_COUNT_RULES[metric])
                if by_class
                else FUNCTIONAL_RULE,
                up=clamped,
                stage=CLAMPED,
                settings=threshold,
            )
        if by_class:
            rows.put(
                f"{graph}_graph.uncertain_count",
                None if metrics is None else metrics.get(UNCERTAIN_METRIC),
                "count",
                population,
                rows.class_text(CANDIDATE_CLASS_COUNT_RULES[UNCERTAIN_METRIC]),
                up=clamped,
                stage=CLAMPED,
                settings=threshold,
            )
        rows.put(
            f"{graph}_graph.functional_denominator",
            None if metrics is None else metrics["Number of nodes"],
            "count",
            population,
            "all vertices of the graph (the denominator of the percentage)",
            up=clamped,
            stage=CLAMPED,
        )


def _saved(up: Upstream | None, state: str, metric: str) -> Any:
    """One saved value of a cited state's column; ``None`` while the products are absent."""
    return _dig(up, "columns", CITED_STATES[state], metric)


def _landscape_summary(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    """Rows read from the saved ``landscape_summary`` products of the cited states.

    Full graph components (the Full landscape's diagnostics, ``graph=full`` requires one
    component holding every vertex), the saved trees' leaf counts and the Published
    point-estimate classification over the Full library. Every value is checked
    against the source that already carries the same quantity in
    ``_check_landscape_summary``.
    """
    up = sources.get("landscape_summary")
    stage = LANDSCAPE_SUMMARY
    full_rule = "Full landscape diagnostics saved with the Full (nominal) summary"
    for key, metric in (
        (
            "full_graph.component_count",
            "Number of components before largest component",
        ),
        ("full_graph.largest_component_vertex_count", "Largest component"),
    ):
        rows.put(
            key,
            _saved(up, "full", metric),
            "count",
            "Full graph",
            f"{full_rule}: {metric.lower()}",
            up=up,
            stage=stage,
        )
    leaf_rule = (
        "leaves of the saved nominal split tree (terminal branches); "
        "equal to the split-tree maxima counted by the peak_count row"
    )
    for state, graph in (("full", "Full"), ("published", "Published")):
        rows.put(
            f"{state}_graph.leaf_count",
            _saved(up, state, "Peak count"),
            "count",
            f"{graph} graph",
            leaf_rule,
            up=up,
            stage=stage,
        )
    for suffix, metric, unit in FULL_LIBRARY_ROWS:
        rows.put(
            f"published_graph.{suffix}",
            _saved(up, "published", metric),
            unit,
            "Full library",
            (
                "Published point-estimate classification of every Full library "
                "sequence by raw fitness against the published threshold; the "
                "percentage divides by the Full library, not the selected graph"
            ),
            up=up,
            stage=stage,
            settings=(THRESHOLD_KEY["published"],),
        )


def quantile_percent(quantile: Any) -> str:
    """A quantile as an exact percentage text: ``0.95`` gives ``95``, never ``95.00000001``."""
    exact = Decimal(repr(float(quantile))) * 100
    return format(exact.normalize(), "f")


def _check_bootstrap_agreement(boot: Upstream | None) -> None:
    """Raise unless the summary's blocks agree on the band they all describe.

    The band simplification, the global verdict and the draw table state the same
    radius, threshold, quantile and peak counts; a disagreement means the summary
    belongs to another run than its tables.
    """
    if boot is None:
        return
    band = _dig(boot, "summary", "band_simplification")
    verdict = _dig(boot, "summary", "diagnostics", "global_verdict")
    settings = _dig(boot, "summary", "settings")
    for name, other, verdict_name in (
        ("band_simplification c", band["c"], "c"),
        ("band_simplification two_c", band["two_c"], "two_c"),
        (
            "band_simplification peaks_before",
            band["peaks_before"],
            "nominal_point_count",
        ),
        (
            "band_simplification peaks_after",
            band["peaks_after"],
            "significant_point_count",
        ),
        ("settings band_quantile", settings["band_quantile"], "band_quantile"),
    ):
        agree(
            (f"bootstrap summary {name}", other),
            (f"bootstrap summary global_verdict {verdict_name}", verdict[verdict_name]),
            BOOTSTRAP,
        )
    radius = _dig(boot, "radius_at_band_quantile")
    if radius is not None and not math.isclose(
        float(band["c"]), float(radius), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"bootstrap summary band_simplification c ({band['c']!r}) differs from "
            f"the band_width of the draw table at the band quantile ({radius!r}); "
            f"an upstream output is stale: rerun `{RERUN[BOOTSTRAP]}`."
        )


def _global_stability(rows: _Rows, boot: Upstream | None, population: str) -> None:
    """The global-stability quantities of the verdict (the band is the verdict)."""
    verdict = _dig(boot, "summary", "diagnostics", "global_verdict")
    sanity = None if verdict is None else verdict["sanity_check"]
    prefix = "bootstrap.global_stability."
    rows.put(
        prefix + "relative_wasserstein_1_median",
        None if verdict is None else verdict["relative_wasserstein_1"]["median"],
        "fraction",
        population,
        "median over the nominal-landscape draws of the Wasserstein-1 distance to the "
        "nominal diagram divided by the nominal diagram's total persistence (global "
        "pair included); descriptive only",
        up=boot,
        stage=BOOTSTRAP,
    )
    draws = "nominal-landscape draws (the bootstrap's own landscape)"
    flat = "flat-landscape draws (a sanity check, not the verdict)"
    for name, value, rule in (
        (
            "nominal_bottleneck_distance_median",
            None if sanity is None else sanity["real"]["median"],
            f"median bottleneck distance to the nominal diagram over the {draws}",
        ),
        (
            "flat_bottleneck_distance_median",
            None if sanity is None else sanity["flat"]["median"],
            f"median bottleneck distance to the nominal diagram over the {flat}",
        ),
        (
            "flat_bottleneck_distance_fifth_percentile",
            None if sanity is None else sanity["flat_bottleneck_p5"],
            f"5th percentile of the bottleneck distance over the {flat}",
        ),
    ):
        rows.put(
            f"{prefix}sanity_check.{name}",
            value,
            "fitness",
            population,
            rule,
            up=boot,
            stage=BOOTSTRAP,
        )
    share = None if sanity is None else sanity["real_share_below_flat_bottleneck_p5"]
    rows.put(
        prefix
        + "sanity_check.nominal_landscape_draws_below_flat_fifth_percentile_percent",
        None if share is None else 100.0 * float(share),
        "percent",
        population,
        f"percent of the {draws} whose bottleneck distance is strictly below the 5th "
        f"percentile of the {flat}",
        up=boot,
        stage=BOOTSTRAP,
        settings=(prefix + "sanity_check.flat_bottleneck_distance_fifth_percentile",),
    )


def _bootstrap(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    boot = sources.get("bootstrap")
    population = "Candidate graph bootstrap"
    settings = _dig(boot, "summary", "settings")
    band = _dig(boot, "summary", "band_simplification")
    value_rule = (
        "persistence strictly greater than the value (nominal persistence > 2c)"
    )
    parameters = (
        (
            "bootstrap.draws",
            None if settings is None else settings["n_replicates"],
            "count",
            "number of bootstrap draws",
        ),
        (
            "bootstrap.seed",
            None if settings is None else settings["seed"],
            "seed",
            "seed of the bootstrap draws",
        ),
        (
            "bootstrap.band_quantile",
            None if settings is None else settings["band_quantile"],
            "quantile",
            "quantile of the draws' bottleneck distances that sets the radius c",
        ),
    )
    for key, value, unit, rule in parameters:
        rows.put(key, value, unit, population, rule, up=boot, stage=BOOTSTRAP)
    _check_bootstrap_agreement(boot)
    percent_rule = (
        "the band quantile as a percentage (nominal band level; coverage not independently established)"
        if rows.candidate_rule == FDR_FIXED_CUTOFF
        else "the band quantile as a percentage (the confidence level of the band)"
    )
    rows.put(
        "bootstrap.band_quantile_percent",
        None if settings is None else quantile_percent(settings["band_quantile"]),
        "percent",
        population,
        percent_rule,
        up=boot,
        stage=BOOTSTRAP,
        settings=("bootstrap.band_quantile",),
        role=VERBATIM_ROLE,
    )
    rows.put(
        "bootstrap.persistence_threshold.radius",
        None if band is None else band["c"],
        "fitness",
        population,
        "radius c: the band quantile of the bottleneck distance to the nominal diagram",
        up=boot,
        stage=BOOTSTRAP,
        settings=("bootstrap.band_quantile",),
    )
    rows.put(
        "bootstrap.persistence_threshold.value",
        None if band is None else band["two_c"],
        "fitness",
        population,
        value_rule + "; the value is 2c",
        up=boot,
        stage=BOOTSTRAP,
        settings=("bootstrap.persistence_threshold.radius",),
    )
    before = None if band is None else band["peaks_before"]
    if boot is None:
        legacy_core = True  # missing upstream retains the existing gap rows
    else:
        from ..simplification.identities import checkpoint_method

        legacy_core = (
            checkpoint_method(
                boot.data["summary"], getattr(boot.raw, "provenance", None)
            )
            == "band_after_core"
        )
    core = None if band is None else band.get("peaks_after_core_rule")
    after = None if band is None else band["peaks_after"]
    states = [("nominal", before, "nominal split tree, no simplification", ())]
    if legacy_core:
        states.append(
            ("after_core_rule", core, "after the peak-core rule (step 1)", ())
        )
    states.append(
        (
            "after_band_rule",
            after,
            (
                "after the band rule (step 2): "
                if legacy_core
                else "after the persistence band: "
            )
            + value_rule,
            ("bootstrap.persistence_threshold.value",),
        )
    )
    for state, count, rule, settings_keys in states:
        rows.put(
            f"bootstrap.candidate.{state}.peak_count",
            count,
            "count",
            population,
            rule,
            up=boot,
            stage=BOOTSTRAP,
            settings=settings_keys,
        )
    absorbed = []
    if legacy_core:
        absorbed.append(
            (
                "after_core_rule",
                before,
                core,
                "peaks of the nominal tree minus peaks after the core rule",
            )
        )
        absorbed.append(
            (
                "after_band_rule",
                core,
                after,
                "peaks after the core rule minus peaks after the band rule",
            )
        )
    for state, first, second, rule in absorbed:
        incomplete = first is None or second is None
        rows.put(
            f"bootstrap.candidate.{state}.absorbed_count",
            None if incomplete else first - second,
            "count",
            population,
            rule,
            up=boot,
            stage=BOOTSTRAP,
            reason=f"{MISSING} the bootstrap summary records no peak count for one "
            "of the two states, so the difference is undefined"
            if incomplete and boot is not None
            else None,
            settings=(f"bootstrap.candidate.{state}.peak_count",),
        )
    rows.put(
        "bootstrap.candidate.total_absorbed_count",
        None if before is None else before - after,
        "count",
        population,
        "peaks of the nominal tree minus peaks after the band rule: every nominal "
        "peak that does not survive the bootstrapped persistence threshold",
        up=boot,
        stage=BOOTSTRAP,
        settings=(
            "bootstrap.candidate.nominal.peak_count",
            "bootstrap.candidate.after_band_rule.peak_count",
        ),
    )
    rows.put(
        "bootstrap.candidate.after_band_rule.saddles_removed_count",
        None if band is None else band["saddles_removed"],
        "count",
        population,
        "saddles of the nominal tree that the band-simplified tree no longer has",
        up=boot,
        stage=BOOTSTRAP,
        settings=("bootstrap.candidate.after_band_rule.peak_count",),
    )
    clamped = sources.get("clamped")
    rows.put(
        "bootstrap.candidate.after_band_rule.retained_sequence_count",
        _dig(clamped, "candidate_tree_sequence_count"),
        "count",
        population,
        "sequences of the band-simplified tree: every sequence of the nominal tree "
        "stays on a branch (absorbed members are reassigned, none is dropped)",
        up=combine(boot, clamped),
        stage=f"{BOOTSTRAP}+{CLAMPED}",
        settings=("bootstrap.candidate.after_band_rule.peak_count",),
    )
    scan = _dig(boot, "bottleneck_range")
    scan_rule = (
        "twice the {end} bottleneck distance of the nominal-landscape draws to the "
        "nominal diagram (the bootstrapped persistence threshold at band quantile {q})"
    )
    for suffix, name, end, q in (
        ("min", "minimum", "smallest", "0"),
        ("max", "maximum", "largest", "1"),
    ):
        rows.put(
            f"bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_{suffix}",
            None if scan is None else 2.0 * scan[name],
            "fitness",
            population,
            scan_rule.format(end=end, q=q),
            up=boot,
            stage=BOOTSTRAP,
        )
    _global_stability(rows, boot, population)


def common_saddle(
    tree: Any,
    kem_dna: str,
    tcl_dna: str,
    reported_dna: str,
    *,
    allow_difference: bool = False,
) -> dict[str, Any]:
    """The lowest common ancestor saddle of two summits in a Candidate split tree.

    Raises when a summit is absent, when the two summits share no distinct saddle,
    or when the saddle's DNA sequence differs from ``reported_dna`` (unless
    ``allow_difference=True``). Returns the node's ``dna_sequence``, ``aa_sequence``
    and ``raw_fitness``.
    """
    nodes = {int(index): dict(tree[index]) for index in tree.node_indices()}
    by_dna = {str(row["dna_sequence"]): index for index, row in nodes.items()}
    try:
        kem, tcl = by_dna[kem_dna], by_dna[tcl_dna]
    except KeyError as error:
        raise ValueError(
            f"Summit {error.args[0]!r} is absent from the Candidate split tree."
        ) from error
    saddle = lowest_common_ancestor(tree, kem, tcl)
    if saddle is None or saddle in {kem, tcl}:
        raise ValueError("The summits do not have a distinct shared saddle.")
    payload = nodes[saddle]
    found = str(payload["dna_sequence"])
    if found != reported_dna and not allow_difference:
        raise ValueError(
            f"The computed common saddle {found!r} differs from the reported "
            f"saddle {reported_dna!r}."
        )
    return {
        "dna_sequence": found,
        "aa_sequence": str(payload["aa_sequence"]),
        "raw_fitness": float(payload["scalar"]),
    }


def _lookup_fdr_sequence(
    dna_sequence: str,
    declaration: Mapping[str, Any],
) -> tuple[float, str]:
    """Look up raw fitness and FDR class of ``dna_sequence`` in the declared FDR snapshot.

    Fails closed: raises if the declaration is invalid, the table file is missing,
    the sequence is absent from
    the snapshot, or the sequence matches more than one row.
    """
    from ..threshold.fdr_selection import _summary_and_path

    _, path, _ = _summary_and_path(declaration)
    if not path.is_file():
        raise FileNotFoundError(f"FDR classification table {path} is missing.")
    df = pl.read_parquet(path, columns=["dna_sequence", "s", "fdr_class"]).filter(
        pl.col("dna_sequence") == dna_sequence
    )
    if df.height == 0:
        raise ValueError(
            f"Sequence {dna_sequence!r} is absent from FDR classification table {path.name}."
        )
    if df.height > 1:
        raise ValueError(
            f"Sequence {dna_sequence!r} matches multiple rows ({df.height}) in FDR classification table {path.name}."
        )
    return float(df["s"][0]), str(df["fdr_class"][0])


def _summits(
    rows: _Rows, sources: Mapping[str, Upstream | None], config: Mapping[str, Any]
) -> None:
    codon, tree_source = sources.get("codon"), sources.get("candidate_tree")
    entropy_config = config["codon_entropy"]
    summits = _dig(codon, "metadata", "summits")
    for name in ("kem", "tcl"):
        for field_name in _SUMMIT_KEYS:
            rows.put(
                f"summits.{name}.{field_name}",
                None if summits is None else summits[name][field_name],
                {"dna_sequence": "dna_sequence", "aa_sequence": "aa_sequence"}.get(
                    field_name, "fitness"
                ),
                "Candidate split tree",
                f"configured {name.upper()} summit sequence; raw fitness at two decimals",
                up=codon,
                stage=CODON,
                role="summit_fitness" if field_name == "raw_fitness" else None,
            )
    saddle = None
    reported_dna = str(entropy_config["reported_saddle_dna_sequence"])
    if tree_source is not None:
        allow_difference = rows.candidate_rule == FDR_FIXED_CUTOFF
        saddle = common_saddle(
            _dig(tree_source, "tree"),
            str(entropy_config["kem_summit_dna_sequence"]),
            str(entropy_config["tcl_summit_dna_sequence"]),
            reported_dna,
            allow_difference=allow_difference,
        )
        recorded = _dig(codon, "metadata", "common_saddle")
        if recorded is not None and recorded["dna_sequence"] != saddle["dna_sequence"]:
            raise ValueError(
                "The common saddle computed from the Candidate split tree "
                f"({saddle['dna_sequence']!r}) differs from the codon-entropy "
                f"metadata ({recorded['dna_sequence']!r})."
            )
    saddle_differs = saddle is not None and saddle["dna_sequence"] != reported_dna
    for field_name in _SUMMIT_KEYS:
        rule = (
            "lowest common ancestor saddle of the KEM and TCL summits in the "
            "Candidate split tree; raw fitness at two decimals"
            if saddle_differs
            else (
                "lowest common ancestor saddle of the KEM and TCL summits, asserted equal "
                "to the configured sequence; raw fitness at two decimals"
            )
        )
        rows.put(
            f"common_saddle.{field_name}",
            None if saddle is None else saddle[field_name],
            {"dna_sequence": "dna_sequence", "aa_sequence": "aa_sequence"}.get(
                field_name, "fitness"
            ),
            "Candidate split tree",
            rule,
            up=tree_source,
            stage=CLAMPED,
            reason=None,
            role="summit_fitness" if field_name == "raw_fitness" else None,
        )
    if saddle_differs:
        reported_fitness, fdr_class = _lookup_fdr_sequence(
            reported_dna, rows.candidate_declaration
        )
        computed_fitness = float(saddle["raw_fitness"])
        merge_diff = reported_fitness - computed_fitness
        d = rows.candidate_declaration
        method = str(d.get("method", "by")).upper()
        q = d.get("q", 0.05)
        computed_seq = str(saddle["dna_sequence"])
        disp_comp = display_spec(
            value_text(computed_fitness), "fitness", "summit_fitness"
        )[1]
        disp_rep = display_spec(
            value_text(reported_fitness), "fitness", "summit_fitness"
        )[1]
        disp_diff = display_spec(
            value_text(abs(merge_diff)), "fitness", "summit_fitness"
        )[1]
        # merge_diff is reported minus computed: positive means the computed saddle
        # lies below the reported one, negative above it.
        if merge_diff > 0:
            position = f"{disp_diff} below"
        elif merge_diff < 0:
            position = f"{disp_diff} above"
        else:
            position = "level with"
        mechanism_note = (
            f"The reported saddle {reported_dna} is classed {fdr_class} under the {method}, "
            f"nominal q = {q:g} screen at cutoff T; in this Candidate graph (edges need "
            f"≥1 retained Functional endpoint) KEM and TCL first merge at {computed_seq} "
            f"(s = {disp_comp}), {position} the reported saddle (s = {disp_rep})."
        )
        rows.put(
            "common_saddle.reported_dna_sequence",
            reported_dna,
            "dna_sequence",
            "Candidate split tree",
            "configured sequence of the reported common saddle",
            up=tree_source,
            stage=CLAMPED,
        )
        rows.put(
            "common_saddle.differs_from_reported",
            True,
            "boolean",
            "Candidate split tree",
            "whether the computed common saddle differs from the reported sequence",
            up=tree_source,
            stage=CLAMPED,
        )
        rows.put(
            "common_saddle.reported_raw_fitness",
            reported_fitness,
            "fitness",
            "Candidate split tree",
            "raw fitness of the reported common saddle sequence; two decimals",
            up=tree_source,
            stage=CLAMPED,
            role="summit_fitness",
        )
        rows.put(
            "common_saddle.merge_height_difference",
            merge_diff,
            "fitness",
            "Candidate split tree",
            "signed difference (reported minus computed) between reported and computed common saddle raw fitness; two decimals",
            up=tree_source,
            stage=CLAMPED,
            role="summit_fitness",
        )
        rows.put(
            "common_saddle.mechanism",
            mechanism_note,
            "text",
            "Candidate split tree",
            "mechanism note for the saddle difference under the FDR candidate cutoff and edge rule",
            up=tree_source,
            stage=CLAMPED,
        )


def _ridges(
    rows: _Rows, sources: Mapping[str, Upstream | None], config: Mapping[str, Any]
) -> None:
    codon = sources.get("codon")
    counts = _dig(codon, "metadata", "counts")
    claim = _dig(codon, "claim")
    minimum = config["codon_entropy"]["two_functional_ridges_share_minimum"]
    rule = (
        "sequences strictly above the common saddle on the ridge's side, saddle "
        "excluded; nonfunctional sequences included; functional means "
        + (
            f"in the Functional class {rows.class_tag}"
            if rows.by_class
            else "raw fitness at or above the Candidate threshold (median_reflected)"
        )
    )
    totals: dict[str, int | None] = {}
    functional: dict[str, int | None] = {}
    for ridge in ("kem", "tcl"):
        totals[ridge] = None if counts is None else counts[f"{ridge}_ridge"]
        functional[ridge] = None if counts is None else counts[f"{ridge}_functional"]
    totals["union"] = None if counts is None else totals["kem"] + totals["tcl"]
    functional["union"] = (
        None if counts is None else functional["kem"] + functional["tcl"]
    )
    for ridge in _RIDGE_KEYS:
        label = ridge.upper() if ridge != "union" else "KEM and TCL union"
        scope = f"{label} ridge" if ridge != "union" else label
        for suffix in _RIDGE_SUFFIXES:
            if suffix == "total_count":
                value = totals[ridge]
            elif suffix == "functional_count":
                value = functional[ridge]
            else:
                value = None if counts is None else totals[ridge] - functional[ridge]
            rows.put(
                f"ridges.{ridge}.{suffix}",
                value,
                "count",
                f"{scope} sequences",
                rule,
                up=codon,
                stage=CODON,
                settings=()
                if suffix == "total_count"
                else (THRESHOLD_KEY["candidate"],),
            )
    row = None
    if claim is not None:
        row = next(
            (
                item
                for item in claim
                if item["claim_id"] == "functional_sequences_in_two_ridges"
            ),
            None,
        )
        if row is None:
            raise KeyError("The codon-entropy claim summary lacks the two-ridges row.")
        if row["numerator"] != functional["union"]:
            raise ValueError(
                "The two-ridges numerator differs from the sum of the ridges' "
                "functional counts."
            )
    share_rule = (
        "functional sequences in either ridge over the functional "
        "Candidate-selected sequences; functional means "
        + (
            f"in the Functional class {rows.class_tag}"
            if rows.by_class
            else "raw fitness at or above the Candidate threshold (median_reflected)"
        )
    )
    share_population = "Functional Candidate-selected sequences"
    for name, field_name, unit in (
        ("numerator", "numerator", "count"),
        ("denominator", "denominator", "count"),
        ("fraction", "fraction", "fraction"),
    ):
        rows.put(
            f"ridges.two_functional_share.{name}",
            None if row is None else row[field_name],
            unit,
            share_population,
            share_rule,
            up=codon,
            stage=CODON,
            reason=f"{UNDEFINED} the codon-entropy claim summary has no fraction "
            "(empty denominator)"
            if codon is not None
            else None,
            settings=(THRESHOLD_KEY["candidate"],),
        )
    config_up = _config_upstream(config)
    rows.put(
        "ridges.two_functional_share.minimum",
        minimum,
        "fraction",
        share_population,
        "configured minimum share of the two functional ridges",
        up=config_up,
        stage=CONFIG,
        role=VERBATIM_ROLE,
    )
    meets = None
    if row is not None and row["fraction"] is not None:
        meets = bool(float(row["fraction"]) >= float(minimum))
    rows.put(
        "ridges.two_functional_share.meets_minimum",
        meets,
        "boolean",
        share_population,
        "share >= configured minimum; the code gives no verdict text",
        up=combine(codon, config_up),
        stage=f"{CODON}+{CONFIG}",
        settings=(
            "ridges.two_functional_share.fraction",
            "ridges.two_functional_share.minimum",
        ),
    )


def _band_view_ridges(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    """Functional KEM and TCL sequences of the band-simplified tree's ridges.

    The bootstrap codon-entropy figure builds its ridges on the band-simplified
    view, which absorbs more hills than the selected-leaf view of the
    ``codon-entropy`` stage, so these are a different population from
    ``ridges.kem.*``; ``band_view_ridge_counts`` is the figure's own computation. The
    in-memory Candidate tree is built from the raw tables, so a missing table makes
    them gap rows.
    """
    view = sources.get("band_view")
    counts = None
    # No view: the clamped comparison or the bootstrap checkpoint is not current, so
    # the loaders' own reasons (stale, or stage not run) say why.
    reason = rows._absent_reason(f"{CLAMPED}+{BOOTSTRAP}") or (
        f"{MISSING} the band-simplified view needs both the clamped comparison and "
        "the bootstrap checkpoint"
    )
    if view is not None:
        try:
            counts = _dig(view, "counts")
        except FileNotFoundError as error:
            reason = f"{MISSING} the raw fitness tables behind the band view: {error}"
    for ridge in ("kem", "tcl"):
        rows.put(
            f"ridges.band_view.{ridge}.functional_count",
            None if counts is None else counts[ridge]["functional_count"],
            "count",
            f"{ridge.upper()} ridge sequences of the band-simplified tree",
            "sequences strictly above the common saddle on the ridge's side of the "
            "band-simplified tree, saddle excluded, "
            + (
                f"in the Functional class {rows.class_tag}"
                if rows.by_class
                else "raw fitness at or above the Candidate threshold"
            ),
            up=view if counts is not None else None,
            stage=BAND_VIEW,
            reason=reason,
            settings=(THRESHOLD_KEY["candidate"],),
        )


def _full_persistence(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    """The two counts of the Full-graph persistence figure (component analysis)."""
    component = sources.get("component")
    counts = _dig(component, "pair_counts")
    population = "Full graph"
    rows.put(
        "full_graph.persistence.ordinary_h0_pair_count",
        None if counts is None else counts["ordinary_h0_pair_count"],
        "count",
        population,
        "ordinary H0 pairs of the descending sweep (peak birth, split-saddle death) "
        "plus the global maximum's pair, closed at the global minimum by convention "
        "(the pair is counted: ordinary H0 persistence includes it)",
        up=component,
        stage=COMPONENT,
    )
    rows.put(
        "full_graph.persistence.valley_count",
        None if counts is None else counts["valley_count"],
        "count",
        population,
        "relative pairs of the descending sweep (valley birth, join-saddle death) "
        "plus one for the global pair's mirror at the global maximum",
        up=component,
        stage=COMPONENT,
    )


def _peak_uncertainty(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    """The measured-sequence count of the peak-uncertainty scatter."""
    peak = sources.get("peak")
    measured = _dig(peak, "diagnostics", "fitness_wt_scatter_row_count")
    rows.put(
        "peak_uncertainty.measured_sequence_count",
        None if measured is None else int(measured) + 1,
        "count",
        "measured sequences of the dataset (both panels draw the same table)",
        "every row of the wild-type-relative fitness table, plus the wild type itself",
        up=peak,
        stage=PEAK,
    )


ENTROPY_BY_POSITION = "entropy_by_position.parquet"


def position_entropy_frame(
    sources: Mapping[str, Upstream | None],
) -> pl.DataFrame:
    """Per-position amino-acid entropy. A table file, not registry rows.

    The registry keeps functional and non-functional vertex counts for the whole
    simplified Candidate tree and for the KEM and TCL ridges. This frame is one
    row per population, fitness group and amino-acid site, plus the three-site sum.
    """
    codon = sources.get("codon")
    sites = _dig(codon, "site_entropy")
    records: list[dict[str, Any]] = []
    for population_id, name in _ENTROPY_POPULATIONS:
        for group_id, group in _ENTROPY_GROUPS:
            selected: dict[int, float | None] = {}
            if sites is not None:
                for item in sites:
                    if (
                        item["population_id"] == population_id
                        and item["fitness_group"] == group_id
                        and item["molecule"] == "amino_acid"
                    ):
                        selected[int(item["site_index"])] = item["entropy_bits"]
                if sorted(selected) != [1, 2, 3]:
                    raise KeyError(
                        f"Codon-entropy site entropy lacks the amino-acid sites of "
                        f"{population_id}/{group_id}."
                    )
            values = {site: selected.get(site) for site in (1, 2, 3)}
            for site, value in values.items():
                records.append(
                    {
                        "population": name,
                        "fitness_group": group,
                        "site": f"aa_site_{site}",
                        "entropy_bits": None if value is None else float(value),
                        "source_fingerprint": None
                        if codon is None
                        else fingerprint_text(codon.fingerprint),
                    }
                )
            total = (
                None
                if codon is None or any(value is None for value in values.values())
                else float(sum(float(value) for value in values.values()))
            )
            records.append(
                {
                    "population": name,
                    "fitness_group": group,
                    "site": "aa_sum",
                    "entropy_bits": total,
                    "source_fingerprint": None
                    if codon is None
                    else fingerprint_text(codon.fingerprint),
                }
            )
    return pl.DataFrame(records)


def paper_quantity_keys() -> set[str]:
    """Registry keys a main-text figure prints, plus the per-branch class counts.

    The per-branch counts are the functional and non-functional vertex counts of
    the simplified Candidate tree and of the KEM and TCL ridges.
    """
    import re

    keys: set[str] = set()
    figures = Path(__file__).resolve().parents[1] / "figures"
    pattern = re.compile(r"""\.display\(\s*['\"]([^'\"]+)['\"]""")
    for path in figures.glob("*.py"):
        keys.update(pattern.findall(path.read_text(encoding="utf-8")))
    for graph in ("full", "published", "candidate"):
        keys.add(f"{graph}_graph.peak_count")
        keys.add(f"{graph}_graph.vertex_count")
        keys.add(f"{graph}_graph.functional_count")
        keys.add(f"thresholds.{graph}_functional.value")
    keys.update(
        {
            "candidate_graph.nonfunctional_count",
            "ridges.kem.functional_count",
            "ridges.kem.nonfunctional_count",
            "ridges.tcl.functional_count",
            "ridges.tcl.nonfunctional_count",
        }
    )
    return keys


def gap_paper_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Paper keys that are missing or a gap row."""
    by_key = {row["key"]: row for row in rows}
    gaps: list[str] = []
    for key in sorted(paper_quantity_keys()):
        row = by_key.get(key)
        if row is None or row.get("gap_reason"):
            gaps.append(key)
    return gaps


def _annotations(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    annotations, clamped = sources.get("annotations"), sources.get("clamped")
    evidence = _dig(clamped, "evidence", "candidate_clamped")
    # The Published counts come from the annotation tables stage; the Candidate
    # counts from the same function (``build_biological_annotations``) run on the
    # cached Candidate split tree, which the bootstrap figure also counts.
    origin = {
        "published": (
            _dig(annotations, "counts"),
            _dig(annotations, "partner"),
            annotations,
            ANNOTATIONS,
        ),
        "candidate": (
            None if evidence is None else evidence["summary"],
            None if evidence is None else evidence["partner"],
            clamped,
            CLAMPED,
        ),
    }
    for graph in _ANNOTATION_GRAPHS:
        c, p, source, source_stage = origin.get(graph, (None, None, None, ANNOTATIONS))
        sourced = graph in origin
        specs: tuple[tuple[str, Any, str], ...] = (
            (
                "stop_codon_peak_count",
                None if c is None else c["stop_codon_peak_count"],
                "peaks whose sequence has a stop codon",
            ),
            (
                "stop_codon_peak_denominator",
                None if c is None else c["peak_denominator"],
                "all peaks",
            ),
            (
                "synonymous_peak_count",
                None if c is None else c["synonymous_endpoint_peak_count"],
                (
                    "peaks whose amino-acid sequence equals that of the adjacent "
                    "descending saddle"
                ),
            ),
            (
                "synonymous_peak_denominator",
                None if c is None else c["adjacent_saddle_peak_denominator"],
                "peaks with an adjacent descending saddle",
            ),
            (
                "synonymous_partner_peak_count",
                None if p is None else p["synonymous_count"],
                (
                    "secondary: peaks whose amino-acid sequence equals that of their "
                    "persistence partner (death)"
                ),
            ),
            (
                "synonymous_partner_peak_denominator",
                None if p is None else p["pair_count"],
                "all persistence pairs",
            ),
            (
                "singleton_peak_count",
                None if c is None else c["singleton_peak_count"],
                "peaks whose branch holds exactly one sequence",
            ),
            (
                "singleton_peak_denominator",
                None if c is None else c["peak_denominator"],
                "all peaks",
            ),
        )
        for suffix, value, rule in specs:
            rows.put(
                f"annotations.{graph}.{suffix}",
                value,
                "count",
                f"{graph.replace('_', ' ')} peaks",
                rule,
                up=source,
                stage=source_stage,
                reason=None
                if sourced
                else f"{MISSING} no stage annotates this tree (annotation counts exist "
                "for the Published and Candidate graphs only)",
            )


def _branches(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    branches = sources.get("branches")
    summary = _dig(branches, "summary")
    sequences = _dig(branches, "sequence_counts")
    tree = None if summary is None else summary["settings"]["tree"]
    population = "Candidate smoothed split tree" + (f" ({tree})" if tree else "")
    for key, value, rule in (
        (
            "terminal_count",
            None if summary is None else summary["terminal_branch_count"],
            "branches ending in a peak (tree edges)",
        ),
        (
            "internal_count",
            None if summary is None else summary["internal_branch_count"],
            "branches that do not end in a peak",
        ),
        (
            "terminal_sequence_count",
            None if sequences is None else sequences["terminal"],
            "sequences carried by terminal branches (root excluded)",
        ),
        (
            "internal_sequence_count",
            None if sequences is None else sequences["internal"],
            "sequences carried by internal branches (root excluded)",
        ),
    ):
        rows.put(
            f"branches.{key}",
            value,
            "count",
            population,
            rule,
            up=branches,
            stage=BRANCHES,
        )


def _band(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    """The leading-saddle-band quantities, one row per quantity of its summary.

    A gap run (stage absent) names every fixed quantity; the intervals are named
    by the stage's sorted landmarks, so absent they are three placeholder gaps.
    """
    band = sources.get("band")
    population = "Full graph raw fitness"
    # The loader's message when it supplied one (absent or stale), else the plain text.
    band_absent = rows.absent.get(BAND) or stage_not_run_reason(BAND)
    if band is not None and _dig(band, "summary", "schema") != BAND_SCHEMA:
        raise ValueError(
            f"The leading-saddle-band summary has schema "
            f"{_dig(band, 'summary', 'schema')!r}, expected {BAND_SCHEMA}; "
            f"rerun `{RERUN[BAND]}`."
        )

    def summary(name: str) -> Any:
        return _dig(band, "summary", name)

    def put(key: str, name: str, unit: str, rule: str, role: str | None = None) -> None:
        rows.put(
            f"leading_saddle_band.{key}",
            summary(name),
            unit,
            population,
            rule,
            up=band,
            stage=BAND,
            reason=band_absent
            if band is None
            else f"{UNDEFINED} the stage recorded no value (an empty population)",
            role=role,
        )

    for landmark in BAND_LANDMARKS:
        label = landmark.replace("_", " ")
        put(
            f"{landmark}.fitness",
            f"{landmark}_fitness",
            "fitness",
            f"raw fitness of the {label}",
        )
    for landmark in BAND_SEQUENCE_LANDMARKS:
        label = landmark.replace("_", " ")
        put(
            f"{landmark}.sequence",
            f"{landmark}_sequence",
            "dna_sequence",
            f"DNA sequence of the {label}",
        )
    put(
        "median_gap",
        "median_gap",
        "fitness",
        "median minus the global minimum (h = m - g)",
    )
    put(
        "candidate_unrounded_threshold",
        "candidate_unrounded_threshold",
        "fitness",
        "candidate threshold before rounding",
    )
    put(
        "candidate_threshold_rule",
        "candidate_threshold_rule",
        "text",
        "rule that computes the candidate threshold",
    )
    put(
        "candidate_threshold_decimals",
        "candidate_threshold_decimals",
        "count",
        "decimals the candidate threshold is rounded to",
    )
    put(
        "total_sequence_count",
        "total_sequence_count",
        "count",
        "all Full graph sequences",
    )
    put(
        "band.lower_landmark",
        "band_lower_landmark",
        "text",
        "the lower of the two trailing saddles (split on a tie)",
    )
    put("band.lower", "band_lower", "fitness", "lower end of the band (closed)")
    put("band.upper", "band_upper", "fitness", "upper end of the band (closed)")
    put(
        "median_inside_band",
        "median_inside_band",
        "boolean",
        "band lower <= Full graph median <= band upper (closed band)",
    )
    for key, prefix, rule in BAND_POPULATIONS:
        put(f"{key}.count", f"{prefix}_count", "count", rule)
        put(
            f"{key}.percentage",
            f"{prefix}_percentage",
            "percent",
            rule + "; percent of all Full graph sequences",
        )
        put(
            f"{key}.entropy_sum",
            f"{prefix}_entropy_sum",
            "bits",
            f"{BAND_ENTROPY_RULE}; {rule}",
        )
    put(
        "below_first_landmark_count",
        "below_first_landmark_count",
        "count",
        "sequences below the lowest landmark (outside every interval)",
    )
    put(
        "above_last_landmark_count",
        "above_last_landmark_count",
        "count",
        "sequences above the highest landmark (outside every interval)",
    )
    put(
        "interval_convention",
        "interval_convention",
        "text",
        "how the intervals include their ends",
    )
    order = summary("interval_order")
    interval_rule = (
        "Full graph sequences in the interval between consecutive landmarks sorted "
        "by fitness (half-open, the last closed)"
    )
    if order is None:
        for suffix, unit in (
            ("count", "count"),
            ("percentage", "percent"),
            ("entropy_sum", "bits"),
        ):
            rows.put(
                f"leading_saddle_band.intervals.{suffix}",
                None,
                unit,
                population,
                interval_rule,
                up=None,
                stage=BAND,
                reason=f"{band_absent}; interval names depend on the sorted landmarks",
            )
        return
    for name in order:
        for suffix, field_name, unit in (
            ("count", "interval_counts", "count"),
            ("percentage", "interval_percentages", "percent"),
            ("entropy_sum", "interval_entropy_sums", "bits"),
        ):
            rows.put(
                f"leading_saddle_band.interval.{name}.{suffix}",
                summary(field_name)[name],
                unit,
                population,
                interval_rule
                + (f"; {BAND_ENTROPY_RULE}" if suffix == "entropy_sum" else ""),
                up=band,
                stage=BAND,
                reason=f"{UNDEFINED} the interval {name} is empty, so its entropy is "
                "undefined",
            )


def _functional_classification(
    rows: _Rows, sources: Mapping[str, Upstream | None]
) -> None:
    """The mixture-quantile threshold, class counts and Papkou replication cell."""
    up = sources.get("functional_classification")
    summary = _dig(up, "summary")
    prefix = "functional_classification."

    def pick(*path: str) -> Any:
        return None if summary is None else _dig(up, "summary", *path)

    def put(
        name: str, value: Any, unit: str, rule: str, reason: str | None = None
    ) -> None:
        rows.put(
            prefix + name,
            value,
            unit,
            FUNCTIONAL_POPULATION,
            rule,
            up=up,
            stage=FUNCTIONAL,
            reason=reason,
        )

    threshold_rule = (
        "mixture_quantile: the declared quantile of the union of the Non-functional "
        "components (stop-anchored), rounded to the configured decimals"
    )
    put(
        "mixture_quantile_functional_threshold_fitness",
        pick("threshold", "rounded"),
        "fitness",
        threshold_rule,
    )
    put(
        "mixture_quantile_functional_threshold_unrounded_fitness",
        pick("threshold", "unrounded"),
        "fitness",
        threshold_rule + "; before rounding",
    )
    put(
        "mixture_component_count",
        pick("mixture_k"),
        "count",
        "number of mixture components with the lowest BIC in the declared range",
    )
    for name, key in (
        ("functional_class_count", "functional"),
        ("non_functional_class_count", "non_functional"),
        ("uncertain_class_count", "uncertain"),
    ):
        put(
            name,
            pick("class_counts", key),
            "count",
            f"{FUNCTIONAL_CLASS_RULE}: sequences of the {key.replace('_', '-')} class at the "
            "declared confidence level",
        )
    put(
        "papkou_replication_functional_threshold_fitness",
        pick("replication_cell", "threshold", "rounded"),
        "fitness",
        FUNCTIONAL_PAPKOU_RULE + " (fixed 0.999 quantile)",
    )
    double = pick("stop_codon_fractions_below_threshold", "2")
    fraction = None if double is None else double["fraction"]
    put(
        "double_stop_fraction_below_threshold",
        fraction,
        "fraction",
        FUNCTIONAL_CLASS_RULE + ": share of double-stop-codon sequences with fitness "
        "below the threshold",
        reason=(
            f"{UNDEFINED} no sequence has two stop codons"
            if up is not None and fraction is None
            else None
        ),
    )


def _threshold_scan(rows: _Rows, sources: Mapping[str, Upstream | None]) -> None:
    """The sensitivity scan's acceptance checks and the canonical threshold interval.

    The scan's cells stay in its own tables; only these are quoted by the manuscript.
    Every row is a gap when the scan has not run (or is not current).
    """
    up = sources.get("functional_threshold_scan")
    acceptance = _dig(up, "acceptance")
    prefix = SCAN_PREFIX
    scan_settings = _dig(up, "scan_config")

    def check(name: str) -> Any:
        return None if acceptance is None else acceptance[name]

    def put(
        name: str,
        value: Any,
        unit: str,
        population: str,
        rule: str,
        *,
        settings: Iterable[str] = (),
        reason: str | None = None,
    ) -> None:
        rows.put(
            prefix + name,
            value,
            unit,
            population,
            rule,
            up=up,
            stage=THRESHOLD_SCAN,
            reason=reason,
            settings=settings,
        )

    resamples = f"{prefix}resample_count"
    minimum = f"{prefix}double_stop_below_minimum"
    limit = f"{prefix}flip_fraction_limit"
    put(
        "resample_count",
        None if scan_settings is None else scan_settings["n_resamples"],
        "count",
        "Nonparametric resamples of the sequences (drawn with replacement)",
        "number of resamples behind each threshold interval",
    )
    put(
        "double_stop_below_minimum",
        None if acceptance is None else check("a")["minimum"],
        "fraction",
        FUNCTIONAL_POPULATION,
        "declared minimum share of double-stop-codon sequences below the threshold "
        "(acceptance check a)",
    )
    put(
        "flip_fraction_limit",
        None if acceptance is None else check("c")["flip_fraction_limit"],
        "fraction",
        FUNCTIONAL_POPULATION,
        "declared largest share of the canonical Functional set that one scan step may "
        "flip (acceptance check c)",
    )
    canonical = None if acceptance is None else check("canonical")
    cell = (
        ""
        if canonical is None
        else f" for the canonical cell ({canonical['family']}, k = {canonical['k']})"
    )
    checks = (
        (
            "a",
            "Double-stop-codon sequences",
            (
                "the canonical cell's share of double-stop sequences below the threshold "
                "is at least the declared minimum"
            ),
            (minimum,),
        ),
        (
            "b",
            FUNCTIONAL_POPULATION,
            (
                "the 95% interval of the threshold over the resamples is narrower than "
                "2 x 1.96 x the median pooled standard error"
            ),
            (resamples,),
        ),
        (
            "c",
            "Canonical Functional set",
            (
                "one scan step from the canonical cell on any axis flips less than the "
                "declared share of the Functional set"
            ),
            (limit,),
        ),
    )

    def undefined(name: str) -> str | None:
        """The gap reason of a check the record leaves undefined (``passed`` null)."""
        if acceptance is None or check("passed")[name] is not None:
            return None
        if name == "all":
            return f"{UNDEFINED} an acceptance check has no verdict"
        detail = check(name).get("reason") or "no verdict recorded"
        return f"{UNDEFINED} check {name} has no verdict ({detail})"

    for name, population, rule, settings in checks:
        put(
            f"acceptance_{name}_passed",
            None if acceptance is None else check("passed")[name],
            "boolean",
            population,
            "acceptance check " + name + ": " + rule,
            settings=settings,
            reason=undefined(name),
        )
    put(
        "acceptance_all_passed",
        None if acceptance is None else check("passed")["all"],
        "boolean",
        FUNCTIONAL_POPULATION,
        "acceptance checks a, b and c all pass (true only when none is false or "
        "undefined); a failed check is reported, never re-tuned",
        reason=undefined("all"),
        settings=(
            f"{prefix}acceptance_a_passed",
            f"{prefix}acceptance_b_passed",
            f"{prefix}acceptance_c_passed",
        ),
    )
    interval = None if up is None else _dig(up, "canonical_interval")
    interval_rule = (
        "2.5% and 97.5% quantiles of the threshold over the converged resamples"
        + cell
        + " (stop-anchored rule, the classification's quantile)"
    )
    no_interval = (
        f"{UNDEFINED} no resample fit of the canonical cell converged"
        if up is not None and (interval is None or interval["threshold_low"] is None)
        else None
    )
    for name, field_name in (
        ("canonical_threshold_interval_low", "threshold_low"),
        ("canonical_threshold_interval_high", "threshold_high"),
        ("canonical_threshold_interval_width", "width"),
    ):
        put(
            name,
            None if interval is None else interval[field_name],
            "fitness",
            FUNCTIONAL_POPULATION,
            interval_rule + ("; high minus low" if field_name == "width" else ""),
            settings=(resamples,),
            reason=no_interval,
        )
    put(
        "interval_width_limit",
        None if acceptance is None else check("b")["limit"],
        "fitness",
        FUNCTIONAL_POPULATION,
        "2 x 1.96 x the median pooled standard error: the widest interval check b accepts",
    )
    axes = None if acceptance is None else check("c")["axes"]
    worst = (
        None
        if axes is None
        else max(
            (
                axis["max_flip_fraction"]
                for axis in axes.values()
                if axis["max_flip_fraction"] is not None
            ),
            default=None,
        )
    )
    put(
        "max_flip_fraction_one_step",
        worst,
        "fraction",
        "Canonical Functional set",
        "largest share of the canonical Functional set flipped by one scan step on "
        "any axis (family, k, quantile, confidence, component rule)",
        settings=(limit,),
        reason=(
            f"{UNDEFINED} no scan step from the canonical cell is in the scan"
            if axes is not None and worst is None
            else None
        ),
    )


def agree(first: tuple[str, Any], second: tuple[str, Any], stage: str) -> None:
    """Raise unless two sources record the same value (exact equality).

    The one equality rule of the registry: ``check_consistency`` uses it between
    upstream outputs and the figures use it between a drawn value and the
    registry's exact text.
    """
    if first[1] != second[1]:
        raise ValueError(
            f"{first[0]} ({first[1]!r}) differs from {second[0]} ({second[1]!r}); "
            f"an upstream output is stale: rerun `{RERUN[stage]}`."
        )


def _check_annotations(annotations: Upstream, clamped: Upstream) -> None:
    """Raise unless the annotation counts describe the Published graph of the clamped record.

    The annotation tables are rebuilt only by ``just biological-annotation-tables``,
    so they can outlive a change of threshold: their landscape must be the
    published (``study_clamped``) one and their peak denominator (and peak count,
    when recorded) must equal the clamped record's Published peak count.
    """
    counts = _dig(annotations, "counts")
    rerun = RERUN[ANNOTATIONS]
    landscape = _dig(annotations, "counts", "landscape")
    if landscape != LANDSCAPE_KEYS["published"]:
        raise ValueError(
            f"biological annotation counts landscape ({landscape!r}) is not the "
            f"published landscape ({LANDSCAPE_KEYS['published']!r}) of the clamped "
            f"comparison; the annotation tables are stale: rerun `{rerun}`."
        )
    expected = (
        "clamped comparison diagnostics study_clamped peak_count",
        _dig(clamped, "diagnostics", LANDSCAPE_KEYS["published"], "peak_count"),
    )
    for name in ("peak_denominator", "peak_count"):
        if name in counts:
            agree(
                expected,
                (f"biological annotation counts {name}", counts[name]),
                ANNOTATIONS,
            )


def _check_populations(sources: Mapping[str, Upstream | None]) -> None:
    """Raise when two figures' populations of one quantity differ.

    The Full graph's vertex count against the peak uncertainty's measured sequences
    (its scatter rows plus the wild type) and against the Full fitness table the
    Full-graph persistence figure sizes itself by; the band view's core total and
    functional counts against the Candidate graph's vertex and functional counts the
    codon-entropy heading prints; and the Candidate stop-codon peak count of the
    cached split tree against the tree the bootstrap biology figure builds in
    memory from the raw tables. Each is skipped when its source is absent.
    """
    clamped, full, view, peak = (
        sources.get(name) for name in ("clamped", "full_fitness", "band_view", "peak")
    )
    if clamped is not None and peak is not None:
        # The scatter table holds every measured sequence but the wild type (its
        # rows are the wild-type-relative table minus the wild type's own), and the
        # Full graph's vertices are every measured sequence including the wild type.
        measured = (
            "peak uncertainty fitness_wt scatter row count plus the wild type",
            int(_dig(peak, "diagnostics", "fitness_wt_scatter_row_count")) + 1,
        )
        agree(
            (
                "clamped graph summary Full Number of nodes",
                _dig(clamped, "graph_summary", "metrics", "Full", "Number of nodes"),
            ),
            measured,
            PEAK,
        )
    if clamped is not None and full is not None:
        agree(
            (
                "clamped graph summary Full Number of nodes",
                _dig(clamped, "graph_summary", "metrics", "Full", "Number of nodes"),
            ),
            ("Full fitness table height", _dig(full, "height")),
            CLAMPED,
        )
    if clamped is None or view is None:
        return
    try:
        counts = _dig(view, "counts")
    except FileNotFoundError:
        return  # the raw tables are absent: the band-view rows are named gaps
    candidate_metrics = _dig(clamped, "graph_summary", "metrics", "Candidate")
    for name, label, summary_key in (
        ("total_count", "total", "Number of nodes"),
        ("functional_count", "functional", "Number of functional sequences"),
    ):
        agree(
            (
                f"clamped graph summary Candidate {summary_key}",
                candidate_metrics[summary_key],
            ),
            (f"band-view core {label} sequence count", counts["core"][name]),
            CLAMPED,
        )
    if "candidate_tree" in counts:
        agree(
            (
                "cached Candidate split tree stop-codon peak count",
                _dig(
                    clamped,
                    "evidence",
                    "candidate_clamped",
                    "summary",
                    "stop_codon_peak_count",
                ),
            ),
            (
                "in-memory Candidate split tree stop-codon peak count",
                counts["candidate_tree"]["stop_codon_peak_count"],
            ),
            CLAMPED,
        )


def _same_quantity(first: tuple[str, Any], second: tuple[str, Any]) -> None:
    """Raise, naming both numbers, unless two sources of one quantity are equal.

    A difference is a finding, not a stale output: neither value is chosen and no code
    is adjusted to force the equality; both numbers are reported for investigation.
    """
    if first[1] != second[1]:
        raise ValueError(
            f"{first[0]} ({first[1]!r}) differs from {second[0]} ({second[1]!r}); "
            "both are the same quantity and neither is chosen: report both numbers "
            "and investigate before changing any code."
        )


def _check_landscape_summary(sources: Mapping[str, Upstream | None]) -> None:
    """Check the saved summary products against the sources that carry the same quantity.

    The saved tree's leaf count equals the registry's peak count (the clamped
    split-tree maxima); the leaves product's synonymous and eligible leaf counts are
    the annotation tables' ``synonymous_peak_*`` numerator and denominator; the graph
    sizes equal the clamped graph summary's; the Full graph is one component that
    holds every vertex.
    """
    summary, clamped = sources.get("landscape_summary"), sources.get("clamped")
    if summary is None:
        return
    _same_quantity(
        ("Full saved Largest component", _saved(summary, "full", "Largest component")),
        ("saved Number of nodes", _saved(summary, "full", "Number of nodes")),
    )
    _same_quantity(
        (
            "Full saved Number of components before largest component",
            _saved(
                summary,
                "full",
                "Number of components before largest component",
            ),
        ),
        ("the one component of the Full graph", 1),
    )
    if clamped is not None:
        metrics = _dig(clamped, "graph_summary", "metrics")
        for state, label in (("full", "Full"), ("published", "Published")):
            _same_quantity(
                (
                    f"{label} saved Number of nodes",
                    _saved(summary, state, "Number of nodes"),
                ),
                (
                    f"clamped graph summary {label} Number of nodes",
                    metrics[label]["Number of nodes"],
                ),
            )
        _same_quantity(
            (
                "Full graph leaf count of the saved tree",
                _saved(summary, "full", "Peak count"),
            ),
            (
                "clamped comparison Full split-tree maxima peak count",
                _dig(clamped, "full_peak_count"),
            ),
        )
        _same_quantity(
            (
                "Published graph leaf count of the saved tree",
                _saved(summary, "published", "Peak count"),
            ),
            (
                "clamped comparison Published peak count",
                _dig(clamped, "diagnostics", LANDSCAPE_KEYS["published"], "peak_count"),
            ),
        )
    annotations = sources.get("annotations")
    if annotations is not None:
        for metric, name in (
            ("Synonymous leaves", "synonymous_endpoint_peak_count"),
            ("Eligible leaves", "adjacent_saddle_peak_denominator"),
        ):
            _same_quantity(
                (
                    f"Published saved {metric} of the leaves product",
                    _saved(summary, "published", metric),
                ),
                (
                    f"annotation tables counts {name}",
                    _dig(annotations, "counts", name),
                ),
            )


def check_consistency(
    sources: Mapping[str, Upstream | None], config: Mapping[str, Any]
) -> None:
    """Raise when upstream outputs disagree on a quantity they all record.

    Checked wherever the sources are present: the published threshold of
    ``config/analysis.toml`` against the clamped comparison (provenance and graph
    summary), the Candidate threshold against the clamped provenance, graph summary,
    codon-entropy metadata and band summary, the nominal bootstrap peak count against
    the Candidate graph's, and the band's Full-graph landmarks against the registry's
    thresholds, the component analysis's global minimum and b_join against the
    clamped record and the band, and the annotation counts' landscape and peak
    counts against the clamped record's Published graph, the Full graph's vertex
    count against the peak uncertainty's measured sequences and the Full fitness
    table, the band view's core counts against the Candidate graph's, and the
    Candidate stop-codon peak count of the cached split tree against the in-memory
    tree (``_check_populations``).
    The bootstrap and the
    branch decomposition record only the Candidate threshold declaration, not its
    value, so no value of theirs is compared.
    """
    clamped, component = sources.get("clamped"), sources.get("component")
    codon, boot, band = (sources.get(name) for name in ("codon", "bootstrap", "band"))
    published = (
        "config/analysis.toml published_functional_threshold",
        config["clamped"]["published_functional_threshold"],
    )
    candidate = None
    if clamped is not None:
        thresholds = _dig(clamped, "provenance", "thresholds")
        record = _dig(clamped, "candidate_threshold")
        metrics = _dig(clamped, "graph_summary", "metrics")
        candidate = (
            "clamped comparison candidate_functional_threshold.json rounded_threshold",
            record["rounded_threshold"],
        )
        agree(
            published,
            (
                "clamped comparison provenance thresholds.study_clamped",
                thresholds["study_clamped"],
            ),
            CLAMPED,
        )
        agree(
            published,
            (
                "clamped graph summary Published functional threshold",
                metrics["Published"]["Functional threshold"],
            ),
            CLAMPED,
        )
        agree(
            candidate,
            (
                "clamped comparison provenance thresholds.candidate_clamped",
                thresholds["candidate_clamped"],
            ),
            CLAMPED,
        )
        agree(
            candidate,
            (
                "clamped graph summary Candidate functional threshold",
                metrics["Candidate"]["Functional threshold"],
            ),
            CLAMPED,
        )
        # The figures count the maxima of the split tree and the registry the
        # graph's peak table: the same set (GLOSSARY: Peak), enforced here.
        for key in EVIDENCE_KEYS:
            agree(
                (
                    f"clamped comparison diagnostics {key} peak_count",
                    _dig(clamped, "diagnostics", key, "peak_count"),
                ),
                (
                    f"{key} split-tree maxima",
                    _dig(clamped, "split_tree_peak_counts", key),
                ),
                CLAMPED,
            )
        # The band-simplified tree retains every sequence of the nominal tree, and
        # the nominal tree represents every vertex of the Candidate graph.
        agree(
            (
                "clamped graph summary Candidate Number of nodes",
                metrics["Candidate"]["Number of nodes"],
            ),
            (
                "Candidate split tree sequence count",
                _dig(clamped, "candidate_tree_sequence_count"),
            ),
            CLAMPED,
        )
    if clamped is not None and component is not None:
        agree(
            (
                "component analysis global_minimum",
                _dig(component, "diagnostics", "global_minimum"),
            ),
            (
                "clamped comparison candidate_functional_threshold.json global_minimum",
                _dig(clamped, "candidate_threshold", "global_minimum"),
            ),
            COMPONENT,
        )
    annotations = sources.get("annotations")
    if annotations is not None and clamped is not None:
        _check_annotations(annotations, clamped)
    if codon is not None and candidate is not None:
        agree(
            candidate,
            (
                "codon-entropy metadata functional_threshold",
                _dig(codon, "metadata", "functional_threshold"),
            ),
            CODON,
        )
    if boot is not None and clamped is not None:
        agree(
            (
                "clamped comparison Candidate graph peak_count",
                _dig(clamped, "diagnostics", "candidate_clamped", "peak_count"),
            ),
            (
                "bootstrap nominal peaks_before",
                _dig(boot, "summary", "band_simplification", "peaks_before"),
            ),
            BOOTSTRAP,
        )
    _check_populations(sources)
    _check_landscape_summary(sources)
    if band is None:
        return
    summary = _dig(band, "summary")
    agree(
        published,
        (
            "leading-saddle-band published_functional_threshold_fitness",
            summary["published_functional_threshold_fitness"],
        ),
        BAND,
    )
    if component is not None:
        # The band's b_join against the component analysis directly, whether or not
        # the clamped comparison is present.
        agree(
            (
                "component analysis join_tree_trailing_saddle",
                _dig(component, "diagnostics", "join_tree_trailing_saddle"),
            ),
            (
                "leading-saddle-band summary join_trailing_saddle_fitness",
                summary["join_trailing_saddle_fitness"],
            ),
            BAND,
        )
    if clamped is None:
        return
    record = _dig(clamped, "candidate_threshold")
    pairs = (
        ("candidate_functional_threshold_fitness", "rounded_threshold"),
        ("global_minimum_fitness", "global_minimum"),
        ("median_fitness", "median"),
        ("candidate_unrounded_threshold", "unrounded_threshold"),
        ("median_gap", "median_gap"),
        ("candidate_threshold_rule", "rule"),
        ("candidate_threshold_decimals", "decimals"),
    )
    for band_name, record_name in pairs:
        agree(
            (
                f"clamped comparison candidate_functional_threshold.json {record_name}",
                record[record_name],
            ),
            (f"leading-saddle-band summary {band_name}", summary[band_name]),
            BAND,
        )


def build_rows(
    sources: Mapping[str, Upstream | None],
    config: Mapping[str, Any],
    *,
    git_commit: str | None,
) -> list[dict[str, Any]]:
    """Build every registry row from the upstream values; validate the keys.

    ``sources`` names: ``clamped``, ``component``, ``peak``, ``bootstrap``,
    ``codon``, ``annotations``, ``branches``, ``band`` and ``candidate_tree``; a
    missing name or ``None`` means the upstream is absent and its quantities
    become gap rows. ``config`` holds the ``[clamped]`` and ``[codon_entropy]``
    tables of ``config/analysis.toml``.
    """
    check_consistency(sources, config)
    rows = _Rows(git_commit, getattr(sources, "absent_reasons", None))
    # Quantity extraction also accepts partial historical configurations.
    declared = config.get("clamped", {}).get(DECLARATION_KEY)
    if isinstance(declared, Mapping) and declared.get("rule") in RULES:
        rows.candidate_rule = declared["rule"]
        rows.candidate_declaration = declared
    _thresholds(rows, sources, config)
    _graphs(rows, sources)
    _landscape_summary(rows, sources)
    _bootstrap(rows, sources)
    _summits(rows, sources, config)
    _ridges(rows, sources, config)
    _band_view_ridges(rows, sources)
    _full_persistence(rows, sources)
    _peak_uncertainty(rows, sources)
    build_rows.position_entropy = position_entropy_frame(sources)
    _annotations(rows, sources)
    _branches(rows, sources)
    _band(rows, sources)
    _functional_classification(rows, sources)
    _threshold_scan(rows, sources)
    validate_rows(rows.rows)
    return rows.rows


def rows_frame(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """The registry rows as a frame with the fixed column order and types."""
    return pl.DataFrame(
        [{name: row[name] for name in COLUMNS} for row in rows], schema=SCHEMA
    )


def schema_description() -> dict[str, Any]:
    """The contents of ``manuscript_quantities.schema.json``."""
    return {
        "columns": {name: str(dtype) for name, dtype in SCHEMA.items()},
        "key_pattern": KEY_PATTERN.pattern,
        "settings_separator": SETTINGS_SEPARATOR,
        "units": sorted({*UNIT_DECIMALS, *VERBATIM_UNITS}),
        "unit_decimals": dict(UNIT_DECIMALS),
        "role_decimals": dict(ROLE_DECIMALS),
        "rounding": "ROUND_HALF_UP from the exact decimal text; never truncated",
        "display_value": (
            "the complete printable number: rounding, comma thousands separators of a "
            "count (unit count only; a seed and an identifier stay verbatim) and sign "
            "live here; a unit symbol (%, draws) does not, the consumer appends it "
            "from unit"
        ),
        "value": "exact value as text: ints as ints, floats by repr, booleans true/false",
        "gap_reason": (
            "null for a sourced row; the reason when value is null, starting with a "
            "kind: 'stale:' an upstream output exists but is not current; 'missing:' "
            "the output is absent or no stage computes the quantity; 'undefined:' the "
            "stage ran and the quantity has no value for its data"
        ),
    }


def write_tables(directory: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the registry as CSV, Parquet and the schema description."""
    frame = rows_frame(rows)
    frame.write_csv(directory / f"{TABLE_STEM}.csv")
    frame.write_parquet(directory / f"{TABLE_STEM}.parquet")
    (directory / f"{TABLE_STEM}.schema.json").write_text(
        json.dumps(schema_description(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Report and Table 1 helpers
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Currency of the exported registry (Table 1 and the figures share it)
# --------------------------------------------------------------------------

RECORD_NAME = ".manuscript-quantities-provenance.json"


def registry_record_path(output: Path) -> Path:
    """The export record of the registry under ``output/tables``."""
    return output / "tables" / RECORD_NAME


def read_registry_fingerprint(output: Path) -> str:
    """The fingerprint the registry export recorded; ``FileNotFoundError`` without one.

    An export removes the record while it copies, so a missing record means the last
    export was interrupted or never finished.
    """
    record = registry_record_path(output)
    if not record.is_file():
        raise FileNotFoundError(
            f"The manuscript quantity registry has no export record ({record}); "
            "the last export was interrupted or never finished: rerun "
            "`just manuscript-quantities`."
        )
    return str(json.loads(record.read_text(encoding="utf-8"))["fingerprint"])


def require_current_registry(output: Path, current: str) -> str:
    """Return the recorded fingerprint, raising unless it equals ``current``.

    ``current`` is the stage fingerprint computed as ``build_manuscript_quantities``
    declares the stage. This is the one registry-currency check: Table 1 calls it and
    the figures read the record through ``read_registry_fingerprint``.
    """
    recorded = read_registry_fingerprint(output)
    if recorded != current:
        raise RuntimeError(
            "The manuscript quantity registry is stale: its record "
            f"{registry_record_path(output)} has fingerprint {recorded!r}, the "
            f"current stage fingerprint is {current!r}; rerun "
            "`just manuscript-quantities`."
        )
    return recorded


def gap_kind(reason: str) -> str | None:
    """The kind (``stale:``, ``missing:`` or ``undefined:``) a gap reason starts with."""
    return next((kind for kind in GAP_KINDS if reason.startswith(kind)), None)


def read_display_values(path: Path) -> dict[str, str | None]:
    """Map ``key`` to ``display_value`` from an exported registry (CSV or Parquet)."""
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
    else:
        frame = pl.read_csv(path, infer_schema=False)
    return dict(frame.select("key", "display_value").iter_rows())


def previous_display_values(path: Path) -> dict[str, str | None] | None:
    """The previous run's display values, or ``None`` when there is no previous CSV."""
    return read_display_values(path) if path.is_file() else None


def manuscript_order(keys: Iterable[str]) -> list[str]:
    """``MANUSCRIPT_ORDER`` restricted to ``keys``, then the unlisted ones as given."""
    present = list(keys)
    known = set(present)
    listed = [key for key in MANUSCRIPT_ORDER if key in known]
    return listed + [key for key in present if key not in set(listed)]


def changes(
    rows: Sequence[Mapping[str, Any]], previous: Mapping[str, str | None] | None
) -> tuple[dict[str, str], list[str]]:
    """Per key ``changed (was X)`` or ``new``, and the keys removed since ``previous``."""
    if previous is None:
        return {}, []
    flags: dict[str, str] = {}
    for row in rows:
        key = row["key"]
        if key not in previous:
            flags[key] = "new"
        elif previous[key] != row["display_value"]:
            was = "gap" if previous[key] is None else previous[key]
            flags[key] = f"changed (was {was})"
    current = {row["key"] for row in rows}
    return flags, sorted(set(previous) - current)


def _cell(text: str | None) -> str:
    return "" if text is None else text.replace("|", "\\|")


def render_report(
    rows: Sequence[Mapping[str, Any]], previous: Mapping[str, str | None] | None
) -> str:
    """The Markdown report: a table per topic, flagged changes, the numbers list."""
    flags, removed = changes(rows, previous)
    gaps = [row for row in rows if row["gap_reason"] is not None]
    kinds = {
        kind: sum(row["gap_reason"].startswith(kind) for row in gaps)
        for kind in GAP_KINDS
    }
    lines = ["# Manuscript quantities", ""]
    lines += [
        (
            f"**Gap rows: {kinds[STALE]} `stale:` (upstream output exists but is not "
            f"current), {kinds[MISSING]} `missing:` (output absent or no stage "
            f"computes it), {kinds[UNDEFINED]} `undefined:` (no value for the "
            "data).**"
        ),
        "",
    ]
    if kinds[STALE]:
        lines += [
            (
                f"**WARNING: {kinds[STALE]} rows are `stale:`; the numbers below do "
                "not include them. Rerun the named upstream stages.**"
            ),
            "",
        ]
    if previous is None:
        lines += ["No previous run: nothing is flagged as changed.", ""]
    else:
        lines += [
            f"{len(flags)} of {len(rows)} rows are new or changed since the previous run.",
            "",
        ]
    lines += [f"{len(gaps)} of {len(rows)} rows are gaps (no source).", ""]
    headings = [heading for heading, _ in TOPICS] + [OTHER_TOPIC]
    for heading in headings:
        topic_rows = [row for row in rows if topic_of(row["key"]) == heading]
        if not topic_rows:
            continue
        lines += [
            f"## {heading}",
            "",
            "| key | display value | unit | status |",
            "|---|---:|---|---|",
        ]
        for row in topic_rows:
            status = flags.get(row["key"], "")
            if row["gap_reason"] is not None:
                status = "; ".join(filter(None, [status, f"gap: {row['gap_reason']}"]))
            lines.append(
                f"| `{row['key']}` | {_cell(row['display_value'])} | {row['unit']} "
                f"| {_cell(status)} |"
            )
        lines.append("")
    lines += ["## Removed keys", ""]
    lines += [f"- `{key}`" for key in removed] or ["None."]
    lines += ["", "## Manuscript numbers", "", "```"]
    sourced = {
        row["key"]: row["display_value"] for row in rows if row["gap_reason"] is None
    }
    for key in manuscript_order(sourced):
        lines.append(f"{key} = {sourced[key]}")
    lines += ["```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Reading the upstream outputs
# --------------------------------------------------------------------------


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(paths: Iterable[Path], label: str) -> None:
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"The {label} output is incomplete: missing {', '.join(missing)}."
        )


def _frame_rows(path: Path) -> list[dict[str, Any]]:
    return pl.read_parquet(path).to_dicts()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes(output: Path, *paths: Path) -> dict[str, str]:
    """sha256 of each existing file (a directory: every file in it), keyed by path
    relative to ``output``.

    A dotfile found by walking a directory is skipped: it is an atomic-write
    temporary (``.<name>.<random>.parquet``), not an artefact. A dotfile named as a
    path (the codon-entropy sidecar) is hashed.
    """
    hashes: dict[str, str] = {}
    for path in paths:
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if path.is_dir() and any(
                part.startswith(".") for part in file.relative_to(path).parts
            ):
                continue
            if file.is_file():
                hashes[file.relative_to(output).as_posix()] = _sha256(file)
    return hashes


# The cached loaders of the upstream stages: each refuses a stale checkpoint and one
# left mid-publication (RuntimeError, ValueError), and raises FileNotFoundError when
# there is no current checkpoint. Imported lazily; they pull in Topographer.


def cached_readers() -> dict[str, Callable[[], Any]]:
    """The real cached loader of each upstream that has one."""
    from ..simplification.bootstrap_smoothed_tree import (
        load_cached_bootstrap_smoothed_tree,
    )
    from ..threshold.functional_classification import (
        load_cached_functional_classification,
    )
    from .branch_decomposition import load_cached_branch_decomposition
    from .clamped_cache import load_cached_clamped_comparison
    from .component_persistence import load_cached_component_persistence
    from .functional_threshold_scan import load_cached_functional_threshold_scan
    from .peak_uncertainty import load_cached_peak_uncertainty

    def load_cached_band() -> Any:
        # Imported here: the build module imports this one.
        from ..build import load_cached_leading_saddle_band

        return load_cached_leading_saddle_band()

    def band_directory() -> Path:
        from ..build import leading_saddle_band_directory

        return leading_saddle_band_directory()

    def functional_directory() -> Path:
        from ..threshold.functional_classification import (
            functional_classification_directory,
        )

        return functional_classification_directory()

    def current_annotation_fingerprint() -> str:
        from ..build import current_biological_annotation_fingerprint

        return current_biological_annotation_fingerprint()

    def full_fitness_height() -> int:
        from ..landscape.full_population import load_full_network_fitness

        return load_full_network_fitness().height

    def load_cited_summary_columns() -> Any:
        from ..results import RunContext
        from ..results.lookup import MissingProductError
        from ..tables.landscape_summary_table import load_summary_columns
        from . import load_settings

        labels = load_settings()["figures"]["tables"]["landscape_summary"]["labels"]
        try:
            return load_summary_columns(
                RunContext(), labels, names=tuple(CITED_STATES.values())
            )
        except MissingProductError as error:
            raise FileNotFoundError(str(error)) from error

    return {
        "landscape_summary": load_cited_summary_columns,
        # The Full peak count is stated, so the Full analysis loads whatever the context.
        "clamped": partial(load_cached_clamped_comparison, include_full=True),
        "component": load_cached_component_persistence,
        "peak": load_cached_peak_uncertainty,
        "bootstrap": load_cached_bootstrap_smoothed_tree,
        "branches": load_cached_branch_decomposition,
        "band": load_cached_band,
        # Not loaders: the keyed folders the two checkpoints are published in.
        "band_directory": band_directory,
        "functional_directory": functional_directory,
        "functional_classification": load_cached_functional_classification,
        # A tables-only export, not a checkpoint: the scan's acceptance record and intervals.
        SCAN_READER: load_cached_functional_threshold_scan,
        # Derived from the clamped comparison and the bootstrap: takes both results.
        "band_view": lambda comparison, bootstrap: (
            describe(
                products={"ridge_counts"}, comparison=comparison, bootstrap=bootstrap
            ).ridge_counts
        ),
        # Not a loader: the fingerprint the annotation tables must carry to be current.
        "annotation_fingerprint": current_annotation_fingerprint,
        # Not a loader either: the Full fitness table's height, for a population check.
        "full_fitness_height": full_fitness_height,
    }


def _stage_directory(
    output: Path,
    readers: Mapping[str, Callable[[], Any]],
    name: str,
    parts: tuple[str, ...],
) -> Path:
    """The folder a checkpoint stage was published in.

    ``produce_or_load`` writes beneath the step root in a parameter-keyed folder, so
    the resolver in ``readers`` (``name``) gives it; without one, the step root.
    """
    resolve = readers.get(name)
    return resolve() if resolve is not None else output.joinpath(*parts)


def _cached(readers: Mapping[str, Callable[[], Any]], name: str) -> Any | None:
    """The cached loader's result; ``None`` when it finds no current checkpoint.

    Only "not found" becomes a gap: a stale or half-published output raises.
    """
    try:
        return readers[name]()
    except FileNotFoundError:
        return None


def load_clamped(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> tuple[Upstream | None, Upstream | None]:
    """The clamped comparison and its Candidate split tree (each ``None`` if absent)."""
    result = _cached(readers, "clamped")
    if result is None:
        return None, None
    directory = output.joinpath(*CLAMPED_DIR)
    files = _hashes(
        output,
        *(
            directory / name
            for name in (
                "provenance.json",
                "diagnostics.json",
                "graph_summary.json",
                "candidate_functional_threshold.json",
            )
        ),
    )
    landscapes = result["landscapes"]
    full = landscapes.get("full")
    if full is None:
        raise ValueError(
            "The clamped comparison was read without its Full analysis, so the Full "
            "split-tree maxima peak count is missing; load it with include_full=True."
        )
    candidate_split = landscapes["candidate_clamped"].trees["split"]
    clamped = Upstream(
        CLAMPED,
        fingerprint_text(result["provenance"].get("fingerprint")),
        {
            "diagnostics": result["diagnostics"],
            "graph_summary": result["graph_summary"],
            "candidate_threshold": result["candidate_threshold"],
            "provenance": result["provenance"],
            "full_peak_count": split_tree_peak_count(full.trees["split"]),
            "candidate_tree_sequence_count": split_tree_sequence_count(candidate_split),
            "split_tree_peak_counts": {
                key: split_tree_peak_count(landscapes[key].trees["split"])
                for key in EVIDENCE_KEYS
            },
            "evidence": {
                key: _landscape_evidence(landscapes[key], key) for key in EVIDENCE_KEYS
            },
        },
        files={
            **files,
            **_hashes(output, directory / "study_clamped" / "data"),
        },
        raw=result,
    )
    tree_source = Upstream(
        CLAMPED,
        clamped.fingerprint,
        {"tree": candidate_split.tree},
        files=_hashes(output, directory / "candidate_clamped" / "data"),
    )
    return clamped, tree_source


def _landscape_evidence(landscape: Any, key: str) -> dict[str, Any]:
    """Annotation counts, partner counts and hill sizes of one cached split tree.

    The same ``build_biological_annotations`` the figures and the annotation tables
    run, so a count read here is the count a figure draws.
    """
    from .biological_annotations import hill_statistics

    evidence = describe(
        products={"annotations"}, landscape=landscape, landscape_name=key
    ).annotations
    return {
        "summary": evidence.summary,
        "partner": {
            "synonymous_count": int(
                evidence.pairs["paired_synonymous_endpoints"].sum()
            ),
            "pair_count": evidence.pairs.height,
        },
        "hills": hill_statistics(evidence.pairs),
    }


def load_component(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> Upstream | None:
    result = _cached(readers, "component")
    if result is None:
        return None
    from .component_persistence import descending_pair_counts

    provenance = output.joinpath(*COMPONENT_PROVENANCE)
    return Upstream(
        COMPONENT,
        fingerprint_text(_json(provenance).get("fingerprint")),
        {"diagnostics": result[1], "pair_counts": descending_pair_counts(result[0])},
        files=_hashes(
            output,
            provenance,
            output.joinpath(*COMPONENT_DIAGNOSTICS),
            output.joinpath(*COMPONENT_PAIRS),
        ),
    )


def load_peak(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> Upstream | None:
    result = _cached(readers, "peak")
    if result is None:
        return None
    directory = output.joinpath(*PEAK_DIR)
    return Upstream(
        PEAK,
        fingerprint_text(result["provenance"].get("fingerprint")),
        {"diagnostics": result["diagnostics"]},
        files=_hashes(
            output, directory / "provenance.json", directory / "diagnostics.json"
        ),
    )


def load_bootstrap(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> Upstream | None:
    result = _cached(readers, "bootstrap")
    if result is None:
        return None
    from ..simplification.bootstrap_smoothed_tree import band_width

    # The scan's range is the bottleneck-distance quantile at 0 and at 1 of the
    # nominal-landscape draws, by the stage's own ``band_width``; the radius at the
    # band quantile is read the same way, to be checked against the summary.
    quantile = result.summary.get("settings", {}).get("band_quantile")
    draws = result.draw_stats
    return Upstream(
        BOOTSTRAP,
        fingerprint_text(
            result.summary.get("fingerprint") or result.provenance.get("fingerprint")
        ),
        {
            "summary": result.summary,
            "bottleneck_range": {
                "minimum": band_width(draws, 0.0),
                "maximum": band_width(draws, 1.0),
            },
            "radius_at_band_quantile": None
            if quantile is None
            else band_width(draws, float(quantile)),
        },
        files=_hashes(
            output,
            output.joinpath(*BOOTSTRAP_SUMMARY),
            output.joinpath(*BOOTSTRAP_DRAWS),
        ),
        raw=result,
    )


def load_codon(output: Path) -> Upstream | None:
    directory = output.joinpath(*TABLES_DIR)
    metadata = directory / CODON_METADATA
    if not metadata.is_file():
        return None
    _require(
        [directory / CODON_CLAIMS, directory / CODON_SITE_ENTROPY], "codon entropy"
    )
    parsed = _json(metadata)
    sidecar = directory / CODON_SIDECAR
    fingerprint = (
        _json(sidecar).get("fingerprint")
        if sidecar.is_file()
        else parsed.get("source_fingerprint")
    )
    return Upstream(
        CODON,
        fingerprint_text(fingerprint),
        {
            "metadata": parsed,
            "claim": _frame_rows(directory / CODON_CLAIMS),
            "site_entropy": _frame_rows(directory / CODON_SITE_ENTROPY),
        },
        files=_hashes(
            output,
            metadata,
            directory / CODON_CLAIMS,
            directory / CODON_SITE_ENTROPY,
            sidecar,
        ),
    )


def _require_current_annotations(sidecar: Path, current: Callable[[], str]) -> None:
    """Raise unless the export record's fingerprint is the current stage fingerprint."""
    rerun = RERUN[ANNOTATIONS]
    if not sidecar.is_file():
        raise RuntimeError(
            f"The biological annotation tables have no export record ({sidecar}); "
            f"the last export was interrupted: rerun `{rerun}`."
        )
    recorded = _json(sidecar).get("fingerprint")
    expected = current()
    if recorded != expected:
        raise ValueError(
            f"biological annotation export record {sidecar.name} has fingerprint "
            f"{recorded!r}, the current annotation stage fingerprint is "
            f"{expected!r}; the annotation tables are stale: rerun `{rerun}`."
        )


def load_annotations(
    output: Path,
    readers: Mapping[str, Callable[[], Any]] | None = None,
    reasons: dict[str, str] | None = None,
) -> Upstream | None:
    """The annotation tables, refused when their export record is not current.

    ``readers["annotation_fingerprint"]``, when given, computes the fingerprint the
    tables must carry (see ``cached_readers``); a mismatch or a missing record
    raises. When it cannot be computed (the comparison provenance it keys on is
    missing, ``FileNotFoundError``) the tables are not current: the upstream is
    absent and its gap reason is ``stale:``. Without it only ``check_consistency``
    (landscape and peak counts against the clamped record) guards the tables.
    """
    directory = output.joinpath(*TABLES_DIR)
    counts = directory / ANNOTATION_COUNTS
    if not counts.is_file():
        return None
    _require([directory / ANNOTATION_PAIRS], "biological annotation")
    current = (readers or {}).get("annotation_fingerprint")
    if current is not None:
        try:
            _require_current_annotations(directory / ANNOTATION_SIDECAR, current)
        except FileNotFoundError as error:
            if reasons is not None:
                reasons[ANNOTATIONS] = (
                    f"{STALE} {ANNOTATIONS} output exists but its current "
                    f"fingerprint cannot be computed: {error}; rerun "
                    f"{RERUN[CLAMPED]} then {RERUN[ANNOTATIONS]}"
                )
            return None
    pairs = pl.read_parquet(directory / ANNOTATION_PAIRS)
    if "paired_synonymous_endpoints" not in pairs.columns:
        raise KeyError("The annotation pairs table lacks paired_synonymous_endpoints.")
    sidecar = directory / ANNOTATION_SIDECAR
    return Upstream(
        ANNOTATIONS,
        fingerprint_text(_json(sidecar).get("fingerprint"))
        if sidecar.is_file()
        else None,
        {
            "counts": _json(counts),
            "partner": {
                "synonymous_count": int(pairs["paired_synonymous_endpoints"].sum()),
                "pair_count": pairs.height,
            },
        },
        files=_hashes(output, counts, directory / ANNOTATION_PAIRS, sidecar),
    )


def load_landscape_summary(
    readers: Mapping[str, Callable[[], Any]],
) -> Upstream | None:
    """The saved Full and Published summary columns (``None`` when the products are absent).

    A reader set without ``landscape_summary`` reads no products. The upstream's
    identity is the two scientific ids, so a repinned or rebuilt product changes the
    registry fingerprint.
    """
    if "landscape_summary" not in readers:
        return None
    result = _cached(readers, "landscape_summary")
    if result is None:
        return None
    columns, products = result
    identity = {
        name: products[name]["scientific_id_sha256"] for name in CITED_STATES.values()
    }
    return Upstream(
        LANDSCAPE_SUMMARY,
        fingerprint_text(identity),
        {"columns": {name: columns[name] for name in identity}},
        identity=identity,
    )


def load_full_fitness(readers: Mapping[str, Callable[[], Any]]) -> Upstream | None:
    """The height of the Full fitness table, for a population check; ``None`` if unread.

    Read through ``readers["full_fitness_height"]`` (absent in readers that name no
    such reader); a missing raw table is an absent source, not an error. The height
    is the source's identity, so the registry fingerprint keys on it.
    """
    read = readers.get("full_fitness_height")
    if read is None:
        return None
    try:
        height = int(read())
    except FileNotFoundError:
        return None
    return Upstream(FULL_FITNESS, None, {"height": height}, identity={"height": height})


def load_branches(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> Upstream | None:
    result = _cached(readers, "branches")
    if result is None:
        return None
    branches = result.branches
    terminal = int(branches.filter(pl.col("is_terminal"))["branch_vertex_count"].sum())
    internal = int(branches.filter(~pl.col("is_terminal"))["branch_vertex_count"].sum())
    directory = output.joinpath(*BRANCH_DIR)
    return Upstream(
        BRANCHES,
        fingerprint_text(result.provenance.get("fingerprint")),
        {
            "summary": result.summary,
            "sequence_counts": {"terminal": terminal, "internal": internal},
        },
        files=_hashes(
            output,
            directory / "summary.json",
            directory / "provenance.json",
            directory / "branches.parquet",
        ),
    )


def load_band(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> Upstream | None:
    """The leading-saddle-band checkpoint through its cached loader.

    The loader refuses a stale or mid-publication checkpoint and raises
    ``FileNotFoundError`` when none is current (a gap). The summary comes from the
    loaded result; the fingerprint and the file hashes from ``provenance.json``.
    """
    result = _cached(readers, "band")
    if result is None:
        return None
    directory = _stage_directory(output, readers, "band_directory", BAND_DIR)
    provenance = _json(directory / "provenance.json")
    return Upstream(
        BAND,
        fingerprint_text(provenance.get("fingerprint")),
        {"summary": result.summary},
        identity={
            "fingerprint": provenance.get("fingerprint"),
            "artefacts": provenance.get("artefacts"),
        },
        files=_hashes(
            output, directory / "summary.json", directory / "provenance.json"
        ),
    )


def load_functional_classification(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> Upstream | None:
    """The functional-classification checkpoint through its cached loader.

    ``None`` when the loader finds no current checkpoint or ``readers`` has none.
    The summary comes from the loaded result; the fingerprint and file hashes from
    the checkpoint's ``provenance.json``.
    """
    if "functional_classification" not in readers:
        return None
    result = _cached(readers, "functional_classification")
    if result is None:
        return None
    directory = _stage_directory(
        output, readers, "functional_directory", FUNCTIONAL_DIR
    )
    provenance = _json(directory / "provenance.json")
    return Upstream(
        FUNCTIONAL,
        fingerprint_text(provenance.get("fingerprint")),
        {"summary": result.summary},
        identity={
            "fingerprint": provenance.get("fingerprint"),
            "artefacts": provenance.get("artefacts"),
        },
        files=_hashes(
            output, directory / "summary.json", directory / "provenance.json"
        ),
    )


def load_threshold_scan(
    output: Path, readers: Mapping[str, Callable[[], Any]]
) -> Upstream | None:
    """The functional-threshold scan through its cached loader.

    ``None`` when the loader finds no current scan or ``readers`` has none. The data
    are the acceptance record, the settings it was run with and the canonical cell's
    row of ``intervals.parquet``; that row must equal the one ``acceptance.json``
    records for check b (they are written together), or the scan is damaged and this
    raises. The upstream's fingerprint is the digest of the scan's identity and table
    hashes (a scan has no helper fingerprint).
    """
    if SCAN_READER not in readers:
        return None
    result = _cached(readers, SCAN_READER)
    if result is None:
        return None
    acceptance = result.acceptance
    check_b = acceptance["b"]
    found = [
        row
        for row in result.intervals.to_dicts()
        if (row["family"], row["k"]) == (check_b["family"], check_b["k"])
    ]
    interval = (
        None
        if not found
        else {
            key: found[0][key] for key in ("threshold_low", "threshold_high", "width")
        }
    )
    recorded = {
        key: check_b[key] for key in ("threshold_low", "threshold_high", "width")
    }
    agree(
        (
            "the canonical interval of intervals.parquet",
            interval or dict.fromkeys(recorded),
        ),
        ("check b of acceptance.json", recorded),
        THRESHOLD_SCAN,
    )
    provenance = result.provenance
    identity = {
        key: provenance.get(key)
        for key in (
            "scan_config",
            "classification_config",
            "classification_source",
            "growth_rates_sha256",
            "artifacts",
        )
    }
    return Upstream(
        THRESHOLD_SCAN,
        fingerprint_text(identity),
        {
            "acceptance": acceptance,
            "scan_config": provenance["scan_config"],
            "canonical_interval": interval,
        },
        identity=identity,
        files=_hashes(output, output.joinpath(*SCAN_DIR)),
    )


class _LazyCounts(Mapping[str, Any]):
    """A mapping computed on first access, so reading fingerprints costs nothing."""

    def __init__(self, compute: Callable[[], Mapping[str, Any]]) -> None:
        self._compute = compute
        self._value: Mapping[str, Any] | None = None

    def _load(self) -> Mapping[str, Any]:
        if self._value is None:
            self._value = self._compute()
        return self._value

    def __getitem__(self, key: str) -> Any:
        return self._load()[key]

    def __iter__(self) -> Any:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


def load_band_view(
    readers: Mapping[str, Callable[..., Any]],
    clamped: Upstream | None,
    bootstrap: Upstream | None,
) -> Upstream | None:
    """The band-simplified view's ridge counts: derived from the comparison and bootstrap.

    The source's fingerprint joins theirs (the raw tables behind the comparison are
    in its fingerprint); the counts themselves are computed when first read, by
    ``band_view_ridge_counts`` from the two cached results.
    """
    if clamped is None or bootstrap is None or "band_view" not in readers:
        return None
    read = readers["band_view"]
    return Upstream(
        BAND_VIEW,
        f"{clamped.fingerprint}+{bootstrap.fingerprint}",
        _LazyCounts(lambda: {"counts": read(clamped.raw, bootstrap.raw)}),
    )


class Sources(dict[str, "Upstream | None"]):
    """The upstreams by name, plus why each cached upstream is missing.

    ``absent_reasons`` maps a stage name to the gap reason built from its cached
    loader's ``FileNotFoundError`` (the loader also raises it for a stale checkpoint).
    """

    absent_reasons: dict[str, str]


# Where each cached upstream keeps its checkpoint under ``output/``: when any of
# these holds a file, an output exists, so a loader's "not found" means it is not
# current (``stale:``); when none does, it is ``missing:``.
STAGE_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    CLAMPED: (CLAMPED_DIR,),
    COMPONENT: (COMPONENT_PROVENANCE, COMPONENT_DIAGNOSTICS),
    PEAK: (PEAK_DIR,),
    BOOTSTRAP: (
        ("checkpoints", "bootstrap_smoothed_tree"),
        ("analysis", "bootstrap_smoothed_tree"),
    ),
    BRANCHES: (BRANCH_DIR,),
    BAND: (BAND_DIR,),
    FUNCTIONAL: (FUNCTIONAL_DIR,),
    THRESHOLD_SCAN: (SCAN_DIR,),
    LANDSCAPE_SUMMARY: (("landscape_summary",),),
}


def _has_output(output: Path, stage: str) -> bool:
    """Whether ``stage`` has any file under its checkpoint paths in ``output``."""
    for parts in STAGE_PATHS.get(stage, ()):
        path = output.joinpath(*parts)
        if path.is_file() or (
            path.is_dir() and any(item.is_file() for item in path.rglob("*"))
        ):
            return True
    return False


def _reasoned(
    readers: Mapping[str, Callable[[], Any]], reasons: dict[str, str], output: Path
) -> dict[str, Callable[[], Any]]:
    """``readers`` that record why the loader found no current checkpoint.

    The reason starts ``stale:`` when the stage has an output under ``output``
    (the loader refused it as not current) and ``missing:`` when it has none.
    Readers that are not loaders (``annotation_fingerprint``) pass through.
    """
    stages = {
        "clamped": CLAMPED,
        "component": COMPONENT,
        "peak": PEAK,
        "bootstrap": BOOTSTRAP,
        "branches": BRANCHES,
        "band": BAND,
        "functional_classification": FUNCTIONAL,
        SCAN_READER: THRESHOLD_SCAN,
        "landscape_summary": LANDSCAPE_SUMMARY,
    }

    def wrap(name: str, read: Callable[[], Any]) -> Callable[[], Any]:
        stage = stages[name]

        def wrapped() -> Any:
            try:
                return read()
            except FileNotFoundError as error:
                state = (
                    f"{STALE} {stage} output exists but is not current"
                    if _has_output(output, stage)
                    else stage_not_run_reason(stage)
                )
                reasons[stage] = f"{state}: {error}; rerun {RERUN[stage]}"
                raise

        return wrapped

    return {
        name: wrap(name, read) if name in stages else read
        for name, read in readers.items()
    }


def load_sources(
    output: Path, *, readers: Mapping[str, Callable[[], Any]] | None = None
) -> Sources:
    """Read every upstream output; a wholly absent one is ``None``.

    The upstreams that have a cached loader are read through it (``readers``
    names them, see ``cached_readers``); the codon-entropy and annotation tables and
    the band stage's loader reads its checkpoint from ``output``.
    """
    reasons: dict[str, str] = {}
    readers = _reasoned(
        cached_readers() if readers is None else readers, reasons, output
    )
    clamped, tree = load_clamped(output, readers)
    bootstrap = load_bootstrap(output, readers)
    sources = Sources(
        {
            "clamped": clamped,
            "candidate_tree": tree,
            "component": load_component(output, readers),
            "peak": load_peak(output, readers),
            "bootstrap": bootstrap,
            "band_view": load_band_view(readers, clamped, bootstrap),
            "codon": load_codon(output),
            "annotations": load_annotations(output, readers, reasons),
            "landscape_summary": load_landscape_summary(readers),
            "full_fitness": load_full_fitness(readers),
            "branches": load_branches(output, readers),
            "band": load_band(output, readers),
            "functional_classification": load_functional_classification(
                output, readers
            ),
            "functional_threshold_scan": load_threshold_scan(output, readers),
        }
    )
    sources.absent_reasons = reasons
    return sources


def source_identities(sources: Mapping[str, Upstream | None]) -> dict[str, Any]:
    """The upstream identities a stage fingerprint keys on (``None`` when absent)."""
    return {
        name: None
        if up is None
        else {
            "stage": up.stage,
            "fingerprint": up.fingerprint,
            **({} if up.identity is None else {"identity": up.identity}),
            **({"files": dict(sorted(up.files.items()))} if up.files else {}),
        }
        for name, up in sorted(sources.items())
    }
