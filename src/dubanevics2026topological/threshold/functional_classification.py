"""The ``functional-classification`` stage: the mixture-quantile Candidate threshold.

Implements Decision items 2-6 of ADR 2026-10-24 (mixture-quantile functional
classification; status: proposed). The stage reads the per-sequence growth-rate
table (``output/analysis/sequence_growth_rates/growth_rates.parquet``: ``s``,
``se_pooled`` and ``stop_codon_count`` for every sequence, the wild type included),
computes the Candidate functional threshold ``T`` by the ``mixture_quantile`` rule and
classifies every sequence. It does NOT change which vertices the Candidate graph
selects: ``median_reflected`` stays the active Selection rule until a later change
rewires the consumers.

Steps (pure functions below, so tests run on arrays and small frames):

1. ``fit_mixtures``: one-dimensional Gaussian mixtures (full covariance) of ``s`` for
   every ``k`` in ``bic_k_range``; the chosen ``k`` has the lowest BIC (a tie goes to
   the smaller ``k``). Components are numbered by ascending mean, so output is
   deterministic.
2. ``double_stop_shares`` and ``select_non_functional``: stop-anchored component
   identification. A component's share is its summed posterior responsibility over the
   double-stop-codon sequences (``stop_codon_count == stop_codon_count_for_anchor``)
   divided by their number; every component with share at least
   ``stop_anchor_min_responsibility`` is a Non-functional component. No anchor
   sequence, or no selected component, raises ``ValueError``.
3. ``mixture_quantile``: ``T`` is the ``quantile`` of the union of the selected
   components, i.e. the root of ``F(x) = sum_j w_j Phi((x - mu_j) / sigma_j) /
   sum_j w_j`` (``j`` over the union, weights renormalised) found with ``brentq``.
   The unrounded and the ``decimals``-rounded values are both reported; classification
   uses the rounded ``T``, as the median-reflected threshold is used.
4. Replication cell: Papkou et al.'s exact rule on the same data (three components,
   lowest mean only, the fixed 0.999 quantile, whatever ``quantile`` is configured), always reported beside the canonical value,
   together with the published threshold and the median-reflected threshold of the
   same population.
5. ``classify``: ``functional_probability = Phi((s - T) / se_pooled)``; the class is
   ``functional`` at or above ``functional_confidence``, ``non_functional`` at or below
   its complement and ``uncertain`` between.

Standard-error rule (the "se rule"). The stage fails closed: any ``se_pooled`` that is
null, NaN, infinite, zero or negative raises ``NonPositiveStandardError`` (a
``ValueError``) naming up to ``NAMED_LIMIT`` ``sequence_id`` values. The growth-rate
stage guarantees ``se_pooled > 0`` for every row it writes (it fails closed on any
other value, and the wild type is fitted like any other sequence: ``s = 0`` by
construction, small positive ``se_pooled``), so there is no point-mass branch. The
wild type's probability is ``Phi((0 - T) / se_pooled)``, close to 1 when ``T < 0``.

Outputs (a produce-or-load checkpoint, ADR 0013, exported to
``output/analysis/functional_classification/``):

- ``classification.parquet``: ``dna_sequence, sequence_id, s, se_pooled,
  stop_codon_count, functional_probability, functional_class``, keyed by sequence
  identity.
- ``components.parquet``: one row per fitted component of the canonical mixture
  (``fit = "canonical"``) and of the Replication cell's three-component mixture
  (``fit = "replication"``): ``fit, component, weight, mean, sd,
  double_stop_responsibility_share, selected``.
- ``summary.json``: ``T``, the rule, the chosen ``k``, BIC per ``k``, class counts, the
  fraction of 1-, 2- and 3-stop sequences below ``T``, the Replication cell and the
  settings.

Convergence: a fit that has not converged within ``max_iter`` at ``tol`` is not a
result. The BIC-chosen fit and the Replication cell's fit raise
``MixtureNotConverged``; the other fits of the BIC loop only record ``converged``.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

from dubanevics2026topological.analysis import ROOT, load_settings
from dubanevics2026topological.analysis.produce_or_load import (
    Checkpoint,
    Stage,
    load_checkpoint,
    produce_or_load,
    result_directory,
)

from .functional_classification_config import (
    CONFIG_KEYS,
    SECTION,
    FunctionalClassificationSettings,
    validate_config,
)
from .functional_threshold import (
    DECLARATION_KEY,
    MEDIAN_REFLECTED,
    MIXTURE_QUANTILE,
    comparison_declaration,
    compute_candidate_functional_threshold,
)

if TYPE_CHECKING:
    from sklearn.mixture import GaussianMixture

STAGE_NAME = "functional-classification"
RETRY_HINT = "just functional-classification"
SUMMARY_SCHEMA = 2
CLASSIFICATION_FILE = "classification.parquet"
COMPONENTS_FILE = "components.parquet"
SUMMARY_FILE = "summary.json"
UNCERTAIN_FILE = "uncertain_sequences.parquet"
ARTEFACTS = (CLASSIFICATION_FILE, COMPONENTS_FILE, SUMMARY_FILE, UNCERTAIN_FILE)
UNCERTAIN_COLUMNS = (
    "dna_sequence",
    "sequence_id",
    "s",
    "se_pooled",
    "functional_probability",
)
NAMED_LIMIT = 5
DEFAULT_MAX_ITER = 1000
DEFAULT_TOL = 1e-6

FUNCTIONAL = "functional"
NON_FUNCTIONAL = "non_functional"
UNCERTAIN = "uncertain"
CLASSES = (FUNCTIONAL, NON_FUNCTIONAL, UNCERTAIN)
CANONICAL = "canonical"
REPLICATION = "replication"
# Papkou et al. (2023): three components, the lowest-mean one, the 0.999 quantile.
# Fixed: the Replication cell does not follow the configured ``quantile``.
REPLICATION_K = 3
REPLICATION_QUANTILE = 0.999
# The Candidate declaration the median-reflected comparison falls back to when the
# caller gives none (the checked-in ``[clamped] candidate_functional_threshold``).
DEFAULT_CANDIDATE_DECLARATION = comparison_declaration({"rule": MEDIAN_REFLECTED, "decimals": 6})
STOP_COUNTS_REPORTED = (1, 2, 3)

CLASSIFICATION_COLUMNS = (
    "dna_sequence",
    "sequence_id",
    "s",
    "se_pooled",
    "stop_codon_count",
    "functional_probability",
    "functional_class",
)
COMPONENT_COLUMNS = (
    "fit",
    "component",
    "weight",
    "mean",
    "sd",
    "double_stop_responsibility_share",
    "selected",
)
CLASSIFICATION_SCHEMA = {
    "dna_sequence": pl.String,
    "sequence_id": pl.String,
    "s": pl.Float64,
    "se_pooled": pl.Float64,
    "stop_codon_count": pl.Int64,
    "functional_probability": pl.Float64,
    "functional_class": pl.String,
}
COMPONENT_SCHEMA = {
    "fit": pl.String,
    "component": pl.Int64,
    "weight": pl.Float64,
    "mean": pl.Float64,
    "sd": pl.Float64,
    "double_stop_responsibility_share": pl.Float64,
    "selected": pl.Boolean,
}
# --------------------------------------------------------------------------
# Mixtures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureFit:
    """One fitted mixture, components numbered by ascending mean."""

    k: int
    bic: float
    converged: bool
    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    # Index into the model's own component order, ascending by mean.
    order: np.ndarray
    model: GaussianMixture

    def responsibilities(self, values: np.ndarray) -> np.ndarray:
        """Posterior responsibilities, one column per component in mean order."""
        proba = self.model.predict_proba(np.asarray(values, float).reshape(-1, 1))
        return proba[:, self.order]


class MixtureNotConverged(RuntimeError):
    """The mixture fit that decides ``T`` did not converge within ``max_iter``."""


def _fit_one(
    column: np.ndarray,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> MixtureFit:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    with warnings.catch_warnings():
        # Non-convergence is recorded (``converged``) and checked by the caller.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        ).fit(column)
    order = np.argsort(model.means_.reshape(-1), kind="stable")
    return MixtureFit(
        k=k,
        bic=float(model.bic(column)),
        converged=bool(model.converged_),
        weights=model.weights_[order].astype(float),
        means=model.means_.reshape(-1)[order].astype(float),
        sds=np.sqrt(model.covariances_.reshape(-1)[order]).astype(float),
        order=order,
        model=model,
    )


def fit_mixtures(
    s: Any,
    k_range: tuple[int, int],
    seed: int,
    n_init: int,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> list[MixtureFit]:
    """Fit a Gaussian mixture of ``s`` for every ``k`` in ``k_range`` (inclusive)."""
    values = np.asarray(s, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("s must be a one-dimensional array of finite values.")
    low, high = k_range
    if values.size < high:
        raise ValueError(f"Cannot fit {high} components to {values.size} sequences.")
    column = values.reshape(-1, 1)
    return [
        _fit_one(column, k, seed, n_init, max_iter, tol) for k in range(low, high + 1)
    ]


def require_converged(fit: MixtureFit, role: str, max_iter: int, tol: float) -> None:
    """Raise ``MixtureNotConverged`` unless ``fit`` converged."""
    if not fit.converged:
        raise MixtureNotConverged(
            f"The {role} mixture fit (k = {fit.k}) did not converge within "
            f"max_iter = {max_iter} at tol = {tol}; raise max_iter or loosen tol in "
            f"[{SECTION}] and rerun `{RETRY_HINT}`."
        )


def choose_by_bic(fits: Sequence[MixtureFit]) -> MixtureFit:
    """The fit with the lowest BIC; a tie goes to the smaller ``k``."""
    if not fits:
        raise ValueError("No mixture fits to choose from.")
    ordered = sorted(fits, key=lambda item: item.k)
    best = ordered[0]
    for fit in ordered[1:]:
        if fit.bic < best.bic:
            best = fit
    return best


def double_stop_shares(
    fit: MixtureFit, s: Any, stop_codon_count: Any, anchor_count: int
) -> np.ndarray:
    """Each component's share of the anchor sequences' summed responsibility.

    The anchor sequences are those with ``stop_codon_count == anchor_count``. Raises
    ``ValueError`` when there are none.
    """
    values = np.asarray(s, dtype=float)
    anchored = np.asarray(stop_codon_count) == anchor_count
    if not anchored.any():
        raise ValueError(
            f"No sequence has {anchor_count} stop codons, so no mixture component "
            "can be anchored to the non-functional distribution."
        )
    return fit.responsibilities(values[anchored]).sum(axis=0) / int(anchored.sum())


def select_non_functional(shares: np.ndarray, min_share: float) -> np.ndarray:
    """Boolean mask of the components with ``share >= min_share``."""
    selected = np.asarray(shares, dtype=float) >= min_share
    if not selected.any():
        raise ValueError(
            f"No mixture component holds a share of at least {min_share} of the "
            "anchor sequences' responsibility."
        )
    return selected


def mixture_quantile(weights: Any, means: Any, sds: Any, quantile: float) -> float:
    """The ``quantile`` of the weight-renormalised union of Gaussian components."""
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(means, dtype=float)
    sd = np.asarray(sds, dtype=float)
    if w.size == 0 or not (w.size == mu.size == sd.size):
        raise ValueError("weights, means and sds must be non-empty and equal length.")
    if (w <= 0).any() or (sd <= 0).any():
        raise ValueError("Component weights and sds must be positive.")
    w = w / w.sum()

    def gap(x: float) -> float:
        return float(np.sum(w * norm.cdf((x - mu) / sd))) - quantile

    reach = 50.0 * float(sd.max())
    return float(
        brentq(
            gap,
            float(mu.min()) - reach,
            float(mu.max()) + reach,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=500,
        )
    )


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def assign_class(probability: Any, confidence: float) -> np.ndarray:
    """Class of each Functional probability at the declared confidence level."""
    p = np.asarray(probability, dtype=float)
    # 1.0 - 0.95 is 0.050000000000000044; round it so p = 0.05 is Non-functional.
    lower = round(1.0 - confidence, 12)
    return np.where(
        p >= confidence,
        FUNCTIONAL,
        np.where(p <= lower, NON_FUNCTIONAL, UNCERTAIN),
    )


class NonPositiveStandardError(ValueError):
    """A ``se_pooled`` that is null, non-finite, zero or negative."""


class NonFiniteGrowthRate(ValueError):
    """An ``s`` that is null, NaN or infinite."""


def functional_probability(
    s: Any, se: Any, threshold: float, sequence_ids: Sequence[str] | None = None
) -> np.ndarray:
    """``Phi((s - T) / se)`` per sequence.

    This is a conditional Normal plug-in exceedance score that treats ``T`` and
    ``se`` as fixed. It is neither a mixture posterior nor a calibrated probability
    of biological function, and the 0.95 class cut-off is a classification
    convention, not an FDR.

    ``se`` is a sequence whose nulls are ``None``. A null, NaN, infinite, zero or
    negative entry raises ``NonPositiveStandardError`` naming up to ``NAMED_LIMIT``
    ids (or row numbers when ``sequence_ids`` is not given).
    """
    values = np.asarray(s, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("s must be finite for every sequence.")
    raw = list(se)
    errors = np.array([np.nan if item is None else float(item) for item in raw])
    bad = ~(np.isfinite(errors) & (errors > 0))
    if bad.any():
        index = np.flatnonzero(bad)[:NAMED_LIMIT]
        named = (
            ", ".join(repr(str(sequence_ids[i])) for i in index)
            if sequence_ids is not None
            else ", ".join(f"row {i}" for i in index)
        )
        raise NonPositiveStandardError(
            f"se_pooled must be finite and strictly positive: {int(bad.sum())} "
            f"invalid rows, first: {named}."
        )
    return norm.cdf((values - threshold) / errors)


def stop_fractions(
    s: Any, stop_codon_count: Any, threshold: float
) -> dict[str, dict[str, Any]]:
    """Counts and fraction of 1-, 2- and 3-stop sequences with ``s < threshold``."""
    values = np.asarray(s, dtype=float)
    stops = np.asarray(stop_codon_count)
    report: dict[str, dict[str, Any]] = {}
    for count in STOP_COUNTS_REPORTED:
        mask = stops == count
        total = int(mask.sum())
        below = int((values[mask] < threshold).sum())
        report[str(count)] = {
            "sequences": total,
            "below_threshold": below,
            "fraction": below / total if total else None,
        }
    return report


@dataclass(slots=True)
class FunctionalClassificationResult:
    """Classification table, component table and summary of one checkpoint."""

    classification: pl.DataFrame
    components: pl.DataFrame
    summary: dict[str, Any]


def _component_rows(
    name: str, fit: MixtureFit, shares: np.ndarray, selected: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "fit": name,
            "component": j,
            "weight": float(fit.weights[j]),
            "mean": float(fit.means[j]),
            "sd": float(fit.sds[j]),
            "double_stop_responsibility_share": float(shares[j]),
            "selected": bool(selected[j]),
        }
        for j in range(fit.k)
    ]


def _threshold_record(unrounded: float, decimals: int) -> dict[str, float]:
    return {"unrounded": unrounded, "rounded": round(unrounded, decimals)}


def classify_growth_rates(
    growth: pl.DataFrame,
    config: Mapping[str, Any],
    published_threshold: float,
    candidate_declaration: Mapping[str, Any] | None = None,
) -> FunctionalClassificationResult:
    """Compute ``T`` and classify every row of the growth-rate table.

    ``candidate_declaration`` is the ``[clamped] candidate_functional_threshold``
    table: its ``decimals`` round the median-reflected comparison value, which is
    always computed with the ``median_reflected`` rule whichever rule is active.
    """
    settings = validate_config(config)
    required = ("dna_sequence", "sequence_id", "s", "se_pooled", "stop_codon_count")
    absent = [name for name in required if name not in growth.columns]
    if absent:
        raise ValueError(f"The growth-rate table lacks columns {absent}.")
    table = growth.select(*required)
    if table.is_empty():
        raise ValueError("The growth-rate table is empty.")
    for column in ("dna_sequence", "sequence_id"):
        if table[column].n_unique() != table.height:
            raise ValueError(f"The growth-rate table repeats a {column}.")
    s = table["s"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    bad = ~np.isfinite(s)
    if bad.any():
        named = ", ".join(
            repr(str(item))
            for item in table["sequence_id"].to_numpy()[bad][:NAMED_LIMIT]
        )
        raise NonFiniteGrowthRate(
            f"s must be finite for every sequence: {int(bad.sum())} invalid rows, "
            f"first: {named}."
        )
    stops = table["stop_codon_count"].to_numpy()

    fits = fit_mixtures(
        s,
        settings.k_range,
        settings.random_seed,
        settings.n_init,
        settings.max_iter,
        settings.tol,
    )
    chosen = choose_by_bic(fits)
    require_converged(chosen, "BIC-chosen", settings.max_iter, settings.tol)
    shares = double_stop_shares(chosen, s, stops, settings.stop_codon_count_for_anchor)
    selected = select_non_functional(shares, settings.stop_anchor_min_responsibility)
    unrounded = mixture_quantile(
        chosen.weights[selected],
        chosen.means[selected],
        chosen.sds[selected],
        settings.quantile,
    )
    threshold = _threshold_record(unrounded, settings.decimals)

    # Replication cell: Papkou's rule, reusing the BIC loop's k = 3 fit when it ran.
    papkou = next((fit for fit in fits if fit.k == REPLICATION_K), None)
    if papkou is None:
        papkou = _fit_one(
            s.reshape(-1, 1),
            REPLICATION_K,
            settings.random_seed,
            settings.n_init,
            settings.max_iter,
            settings.tol,
        )
    require_converged(papkou, "Replication cell", settings.max_iter, settings.tol)
    papkou_shares = double_stop_shares(
        papkou, s, stops, settings.stop_codon_count_for_anchor
    )
    papkou_selected = np.zeros(papkou.k, dtype=bool)
    papkou_selected[0] = True
    papkou_unrounded = mixture_quantile(
        papkou.weights[papkou_selected],
        papkou.means[papkou_selected],
        papkou.sds[papkou_selected],
        REPLICATION_QUANTILE,
    )
    declaration = (
        DEFAULT_CANDIDATE_DECLARATION
        if candidate_declaration is None
        else candidate_declaration
    )
    median_reflected = compute_candidate_functional_threshold(
        s.tolist(), comparison_declaration(declaration)
    )

    # The se rule (module docstring): fail closed on any se_pooled that is not finite
    # and positive; the growth-rate stage guarantees it, wild type included.
    se_values = [
        None if value is None else float(value) for value in table["se_pooled"]
    ]
    probability = functional_probability(
        s, se_values, threshold["rounded"], table["sequence_id"].to_list()
    )
    classes = assign_class(probability, settings.functional_confidence)
    classification = pl.DataFrame(
        {
            "dna_sequence": table["dna_sequence"],
            "sequence_id": table["sequence_id"],
            "s": table["s"].cast(pl.Float64),
            "se_pooled": table["se_pooled"].cast(pl.Float64),
            "stop_codon_count": table["stop_codon_count"].cast(pl.Int64),
            "functional_probability": pl.Series(probability, dtype=pl.Float64),
            "functional_class": pl.Series(classes.tolist(), dtype=pl.String),
        },
        schema=CLASSIFICATION_SCHEMA,
    ).select(*CLASSIFICATION_COLUMNS)

    components = pl.DataFrame(
        _component_rows(CANONICAL, chosen, shares, selected)
        + _component_rows(REPLICATION, papkou, papkou_shares, papkou_selected),
        schema=COMPONENT_SCHEMA,
    ).select(*COMPONENT_COLUMNS)

    counts = {name: int((classes == name).sum()) for name in CLASSES}
    summary = {
        "schema": SUMMARY_SCHEMA,
        "rule": MIXTURE_QUANTILE,
        "threshold": threshold,
        "mixture_k": chosen.k,
        "converged": chosen.converged,
        "bic": [
            {"k": fit.k, "bic": fit.bic, "converged": fit.converged} for fit in fits
        ],
        "selected_components": [int(j) for j in np.flatnonzero(selected)],
        "class_counts": counts,
        "sequence_count": table.height,
        "stop_codon_fractions_below_threshold": stop_fractions(
            s, stops, threshold["rounded"]
        ),
        "replication_cell": {
            "rule": "papkou_lowest_mean_component",
            "k": REPLICATION_K,
            "quantile": REPLICATION_QUANTILE,
            "converged": papkou.converged,
            "threshold": _threshold_record(papkou_unrounded, settings.decimals),
            "published_functional_threshold": float(published_threshold),
            "median_reflected_threshold": {
                "unrounded": median_reflected.unrounded,
                "rounded": median_reflected.threshold,
            },
        },
        "settings": {
            "bic_k_range": list(settings.k_range),
            "stop_anchor_min_responsibility": (settings.stop_anchor_min_responsibility),
            "stop_codon_count_for_anchor": settings.stop_codon_count_for_anchor,
            "quantile": settings.quantile,
            "functional_confidence": settings.functional_confidence,
            "decimals": settings.decimals,
            "random_seed": settings.random_seed,
            "n_init": settings.n_init,
            "max_iter": settings.max_iter,
            "tol": settings.tol,
        },
        "components_artefact": COMPONENTS_FILE,
    }
    return FunctionalClassificationResult(classification, components, summary)


def describe(result: FunctionalClassificationResult) -> str:
    """The one-line summary the command prints."""
    summary = result.summary
    counts = summary["class_counts"]
    return (
        f"Functional classification ready: T = {summary['threshold']['rounded']} "
        f"({summary['rule']}, k = {summary['mixture_k']}); "
        f"{counts[FUNCTIONAL]} functional, {counts[NON_FUNCTIONAL]} non-functional, "
        f"{counts[UNCERTAIN]} uncertain of {summary['sequence_count']} sequences."
    )


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def write_functional_classification(
    directory: Path, result: FunctionalClassificationResult
) -> list[Path]:
    """Write the four artefacts into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    result.classification.write_parquet(directory / CLASSIFICATION_FILE)
    result.components.write_parquet(directory / COMPONENTS_FILE)
    # The Uncertain class is reported, never selected: list it beside the stage.
    result.classification.filter(pl.col("functional_class") == UNCERTAIN).select(
        *UNCERTAIN_COLUMNS
    ).write_parquet(directory / UNCERTAIN_FILE)
    (directory / SUMMARY_FILE).write_text(
        json.dumps(result.summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return [directory / name for name in ARTEFACTS]


def read_functional_classification(
    directory: Path,
) -> FunctionalClassificationResult:
    """Read the artefacts of a checkpoint or export; refuse another schema."""
    classification = pl.read_parquet(directory / CLASSIFICATION_FILE)
    components = pl.read_parquet(directory / COMPONENTS_FILE)
    summary = json.loads((directory / SUMMARY_FILE).read_text(encoding="utf-8"))
    if classification.columns != list(CLASSIFICATION_COLUMNS):
        raise ValueError(
            f"Classification columns are {classification.columns}, expected "
            f"{list(CLASSIFICATION_COLUMNS)}; rerun `{RETRY_HINT}`."
        )
    if components.columns != list(COMPONENT_COLUMNS):
        raise ValueError(
            f"Component columns are {components.columns}, expected "
            f"{list(COMPONENT_COLUMNS)}; rerun `{RETRY_HINT}`."
        )
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError(f"Unsupported summary schema; rerun `{RETRY_HINT}`.")
    return FunctionalClassificationResult(classification, components, summary)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def _output_root(context: Any | None) -> Path:
    """The output root of ``context``; a context-less call resolves a fresh one now."""
    if context is None:
        from dubanevics2026topological.results import RunContext

        context = RunContext()
    return Path(context.output_root)


def checkpoint_dir(context: Any | None = None) -> Path:
    """The classification checkpoint directory under the context's output root."""
    return _output_root(context) / "checkpoints" / "functional_classification"


def export_dir(context: Any | None = None) -> Path:
    """The exported-artefact directory under the context's output root."""
    return _output_root(context) / "analysis" / "functional_classification"


def growth_rates_path(context: Any | None = None) -> Path:
    """The growth-rate table this stage reads (its one declared input)."""
    return (
        _output_root(context)
        / "analysis"
        / "sequence_growth_rates"
        / "growth_rates.parquet"
    )


def _library_versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in ("scikit-learn", "scipy")}


def functional_classification_stage(
    settings: Mapping[str, Any],
    *,
    root: Path | None = None,
    growth_rates: Path | None = None,
) -> Stage:
    """Declare the stage: scientific settings, the growth-rate table and the code.

    ``project_stage`` records only the factory's libraries, so the versions of
    scikit-learn and scipy (which decide the fit) enter the fingerprint as settings.
    No scheduling key exists in ``[functional_classification]``.
    """
    # Imported here because ``stages`` itself imports the helper in this package.
    from dubanevics2026topological.stages import project_stage

    scientific = settings["analysis"]
    validate_config(scientific[SECTION])
    threshold_dir = Path(__file__).resolve().parents[1] / "threshold"
    return project_stage(
        STAGE_NAME,
        settings={
            "schema": SUMMARY_SCHEMA,
            "config": scientific[SECTION],
            "published_functional_threshold": scientific["clamped"][
                "published_functional_threshold"
            ],
            # Rounds the median-reflected comparison value in the summary. The rule
            # is pinned, so switching the active Selection rule leaves the
            # classification checkpoint current.
            "candidate_declaration": comparison_declaration(
                scientific["clamped"][DECLARATION_KEY]
            ),
            "libraries": _library_versions(),
        },
        inputs=(growth_rates if growth_rates is not None else growth_rates_path(),),
        code=(
            Path(__file__).resolve(),
            threshold_dir / "functional_threshold.py",
            threshold_dir / "functional_classification_config.py",
        ),
        root=ROOT if root is None else root,
    )


def _require_growth_rates(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"No sequence growth-rate table at {path}; run "
            "`just sequence-growth-rates` first."
        )


def _require_section(scientific: Mapping[str, Any]) -> None:
    """``FileNotFoundError`` (a registry gap) when the table is not configured."""
    if SECTION not in scientific:
        raise FileNotFoundError(
            f"config/analysis.toml has no [{SECTION}] table, so there is no "
            f"functional classification; add it and run `{RETRY_HINT}`."
        )


def compute_functional_classification(
    settings: dict[str, Any] | None = None,
    *,
    force: bool = False,
    context: Any | None = None,
) -> FunctionalClassificationResult:
    """Load the current checkpoint or compute, publish and export a new one.

    Reads the cached growth-rate checkpoint (fail closed when it is missing or
    pending), never recomputes it.
    """
    from dubanevics2026topological.analysis.sequence_growth_rates import (
        load_cached_sequence_growth_rates,
    )

    settings = settings or load_settings(context=context)
    scientific = settings["analysis"]
    _require_section(scientific)
    rates_path = growth_rates_path(context=context)
    _require_growth_rates(rates_path)
    growth = load_cached_sequence_growth_rates(context=context).growth_rates
    published = float(scientific["clamped"]["published_functional_threshold"])
    export_to = export_dir(context)

    def produce(directory: Path) -> None:
        write_functional_classification(
            directory,
            classify_growth_rates(
                growth,
                scientific[SECTION],
                published,
                scientific["clamped"][DECLARATION_KEY],
            ),
        )

    def export(ready: Checkpoint) -> None:
        ready.export_to(export_to)

    stage = functional_classification_stage(settings, growth_rates=rates_path)
    checkpoint = produce_or_load(
        stage,
        produce,
        checkpoint_dir=result_directory(checkpoint_dir(context), stage),
        force=force,
        on_ready=export,
    )
    return read_functional_classification(checkpoint.path)


def load_cached_functional_classification(
    settings: dict[str, Any] | None = None,
    *,
    context: Any | None = None,
) -> FunctionalClassificationResult:
    """Read the current checkpoint; never compute, recover or move anything."""
    settings = settings or load_settings(context=context)
    _require_section(settings["analysis"])
    rates_path = growth_rates_path(context=context)
    stage = functional_classification_stage(settings, growth_rates=rates_path)
    return load_checkpoint(
        stage,
        lambda ready: read_functional_classification(ready.path),
        checkpoint_dir=result_directory(checkpoint_dir(context), stage),
        retry_hint=RETRY_HINT,
    )


def functional_classification_directory(
    settings: dict[str, Any] | None = None, *, context: Any | None = None
) -> Path:
    """The keyed folder of the current classification checkpoint (it may not exist)."""
    settings = settings or load_settings(context=context)
    _require_section(settings["analysis"])
    stage = functional_classification_stage(
        settings, growth_rates=growth_rates_path(context=context)
    )
    return result_directory(checkpoint_dir(context), stage)


def checkpoint_source(ready: Checkpoint) -> dict[str, Any]:
    """The identity of a classification checkpoint, for downstream fingerprints.

    The stage fingerprint and the sha256 of every artefact, so a re-run
    classification (new settings, inputs, code or bytes) changes it.
    """
    return {
        "stage": STAGE_NAME,
        "fingerprint": ready.provenance["fingerprint"],
        "artefacts": dict(sorted(ready.provenance["artefacts"].items())),
    }


def load_cached_functional_classification_with_source(
    settings: dict[str, Any] | None = None,
    *,
    tables: bool = True,
    lock: bool = True,
    context: Any | None = None,
) -> tuple[FunctionalClassificationResult, dict[str, Any]]:
    """The current checkpoint and its identity; fail closed like the plain loader.

    ``FileNotFoundError`` when there is no current checkpoint (missing or stale),
    ``RuntimeError`` when it is unreadable or unfinished. With ``tables=False`` only
    ``summary.json`` is read and the two frames are empty, for a fingerprint that
    needs the identity and ``T`` but not 261,333 rows. ``lock=False`` skips the
    checkpoint lock (nonblocking and exclusive, so concurrent readers would refuse
    each other); only a caller that knows no process writes the checkpoint may pass
    it, like the Slurm array tasks of a sliced bootstrap.
    """
    settings = settings or load_settings(context=context)
    _require_section(settings["analysis"])

    def read(ready: Checkpoint) -> tuple[FunctionalClassificationResult, dict]:
        if tables:
            result = read_functional_classification(ready.path)
        else:
            summary = json.loads(
                (ready.path / SUMMARY_FILE).read_text(encoding="utf-8")
            )
            if summary.get("schema") != SUMMARY_SCHEMA:
                raise ValueError(f"Unsupported summary schema; rerun `{RETRY_HINT}`.")
            result = FunctionalClassificationResult(
                pl.DataFrame(schema=CLASSIFICATION_SCHEMA),
                pl.DataFrame(schema=COMPONENT_SCHEMA),
                summary,
            )
        return result, checkpoint_source(ready)

    stage = functional_classification_stage(
        settings, growth_rates=growth_rates_path(context)
    )
    return load_checkpoint(
        stage,
        read,
        checkpoint_dir=result_directory(checkpoint_dir(context), stage),
        retry_hint=RETRY_HINT,
        lock=lock,
    )


__all__ = [
    "CONFIG_KEYS",
    "REPLICATION_QUANTILE",
    "SECTION",
    "FunctionalClassificationResult",
    "FunctionalClassificationSettings",
    "MixtureFit",
    "MixtureNotConverged",
    "NonFiniteGrowthRate",
    "NonPositiveStandardError",
    "assign_class",
    "checkpoint_source",
    "choose_by_bic",
    "classify_growth_rates",
    "compute_functional_classification",
    "describe",
    "double_stop_shares",
    "fit_mixtures",
    "functional_classification_stage",
    "functional_probability",
    "load_cached_functional_classification",
    "load_cached_functional_classification_with_source",
    "mixture_quantile",
    "read_functional_classification",
    "require_converged",
    "select_non_functional",
    "stop_fractions",
    "validate_config",
    "write_functional_classification",
]
