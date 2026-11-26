"""Mixture-quantile threshold, stop-anchored identification and classification.

All fixtures are small seeded numpy draws; nothing here reads ``data/raw/``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint
from dubanevics2026topological.threshold import functional_classification as fc
from dubanevics2026topological.threshold import functional_threshold as ct

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "bic_k_range": [1, 5],
    "stop_anchor_min_responsibility": 0.10,
    "stop_codon_count_for_anchor": 2,
    "quantile": 0.999,
    "functional_confidence": 0.95,
    "decimals": 6,
    "random_seed": 20261024,
    "n_init": 3,
    "max_iter": 1000,
    "tol": 1e-6,
}
DECLARATION = {"rule": "median_reflected", "decimals": 6}
PUBLISHED = -0.507774


def growth_table(
    groups: list[tuple[float, float, int, int, int]],
    seed: int = 7,
    se: float = 0.02,
) -> pl.DataFrame:
    """Rows from ``(mean, sd, n, n_double_stop, n_single_stop)`` per bell."""
    rng = np.random.default_rng(seed)
    s: list[float] = []
    stops: list[int] = []
    for mean, sd, n, doubles, singles in groups:
        s.extend(rng.normal(mean, sd, n))
        stops.extend([2] * doubles + [1] * singles + [0] * (n - doubles - singles))
    return pl.DataFrame(
        {
            "dna_sequence": [f"SEQ{i:05d}" for i in range(len(s))],
            "sequence_id": [f"id{i:05d}" for i in range(len(s))],
            "s": s,
            "se_pooled": [se] * len(s),
            "stop_codon_count": stops,
        }
    )


@pytest.fixture(scope="module")
def two_bells() -> pl.DataFrame:
    # Non-functional bell around -1 holding the double-stop rows; functional around 0.
    return growth_table([(-1.0, 0.15, 1500, 200, 100), (0.0, 0.10, 1500, 0, 0)])


@pytest.fixture(scope="module")
def split_bells() -> pl.DataFrame:
    # The non-functional bell is two close components; double stops sit in both.
    return growth_table(
        [
            (-1.2, 0.10, 1200, 150, 60),
            (-0.8, 0.10, 1200, 150, 60),
            (0.3, 0.10, 1500, 0, 0),
        ]
    )


@pytest.fixture(scope="module")
def skewed_bell() -> pl.DataFrame:
    """One skewed non-functional bell (carrying every stop row) and a functional bell.

    The non-functional values are a shifted gamma(3, 0.12), right-skewed, so a forced
    k = 3 Gaussian fit splits it into a narrow core and a wide shoulder. Checked at
    seeds 7 to 10 and at 2 and 3 gamma shapes: the two halves' double-stop shares
    stay between 0.29 and 0.71, far from the 0.10 selection edge, so the fixed seed
    below is not a lucky draw.
    """
    rng = np.random.default_rng(7)
    nonfunctional = -1.3 + rng.gamma(3.0, 0.12, 3000)
    functional = rng.normal(0.3, 0.10, 1500)
    s = np.concatenate([nonfunctional, functional])
    stops = [2] * 400 + [1] * 200 + [0] * 2400 + [0] * 1500
    return pl.DataFrame(
        {
            "dna_sequence": [f"SEQ{i:05d}" for i in range(s.size)],
            "sequence_id": [f"id{i:05d}" for i in range(s.size)],
            "s": s,
            "se_pooled": [0.02] * s.size,
            "stop_codon_count": stops,
        }
    )


def canonical_union(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights, means and sds of the selected canonical components of a result."""
    rows = result.components.filter((pl.col("fit") == "canonical") & pl.col("selected"))
    return (
        rows["weight"].to_numpy(),
        rows["mean"].to_numpy(),
        rows["sd"].to_numpy(),
    )


def bisect_union_quantile(
    weights: np.ndarray, means: np.ndarray, sds: np.ndarray, q: float
) -> float:
    """An independent root finder: plain bisection on a hand-written CDF."""
    w = weights / weights.sum()

    def cdf(x: float) -> float:
        return sum(
            wj * 0.5 * (1 + math.erf((x - m) / (sd * math.sqrt(2))))
            for wj, m, sd in zip(w, means, sds, strict=True)
        )

    low, high = float(means.min() - 20 * sds.max()), float(means.max() + 20 * sds.max())
    for _ in range(200):
        mid = (low + high) / 2
        low, high = (mid, high) if cdf(mid) < q else (low, mid)
    return (low + high) / 2


# 1. Two components ---------------------------------------------------------


def test_bic_picks_two_components_and_the_stop_anchor_selects_the_low_one(
    two_bells: pl.DataFrame,
) -> None:
    s = two_bells["s"].to_numpy()
    fits = fc.fit_mixtures(s, (1, 4), CONFIG["random_seed"], CONFIG["n_init"])
    chosen = fc.choose_by_bic(fits)
    assert chosen.k == 2
    assert chosen.means[0] < chosen.means[1]
    shares = fc.double_stop_shares(chosen, s, two_bells["stop_codon_count"], 2)
    assert shares[0] > 0.99 and shares[1] < 0.01
    selected = fc.select_non_functional(shares, 0.10)
    assert selected.tolist() == [True, False]
    threshold = fc.mixture_quantile(
        chosen.weights[selected], chosen.means[selected], chosen.sds[selected], 0.999
    )
    expected = norm.ppf(0.999, chosen.means[0], chosen.sds[0])
    assert threshold == pytest.approx(expected, abs=1e-8)
    assert threshold == pytest.approx(-1.0 + 3.09 * 0.15, abs=0.08)


def test_the_lowest_mean_rule_agrees_with_the_stop_anchor_on_two_bells(
    two_bells: pl.DataFrame,
) -> None:
    result = fc.classify_growth_rates(
        two_bells, {**CONFIG, "bic_k_range": [2, 2]}, PUBLISHED
    )
    canonical = result.components.filter(pl.col("fit") == "canonical")
    assert canonical["selected"].to_list() == [True, False]
    # k = 3 lowest-mean component is a part of the low bell, so only the k = 2 value
    # is compared: the canonical T is the quantile of the single low component.
    low = canonical.row(0, named=True)
    assert result.summary["threshold"]["unrounded"] == pytest.approx(
        norm.ppf(0.999, low["mean"], low["sd"]), abs=1e-8
    )


def test_the_chosen_k_breaks_a_bic_tie_toward_the_smaller_k() -> None:
    def fit(k: int, bic: float) -> fc.MixtureFit:
        one = np.ones(k)
        return fc.MixtureFit(k, bic, True, one / k, one, one, np.arange(k), None)  # type: ignore[arg-type]

    assert fc.choose_by_bic([fit(3, 5.0), fit(2, 5.0), fit(1, 9.0)]).k == 2
    assert fc.choose_by_bic([fit(1, 9.0), fit(2, 5.0), fit(3, 4.0)]).k == 3


# 2. A split non-functional bell --------------------------------------------


def test_split_non_functional_bell_merges_both_halves(
    split_bells: pl.DataFrame,
) -> None:
    s = split_bells["s"].to_numpy()
    (fit,) = fc.fit_mixtures(s, (3, 3), CONFIG["random_seed"], CONFIG["n_init"])
    shares = fc.double_stop_shares(fit, s, split_bells["stop_codon_count"], 2)
    selected = fc.select_non_functional(shares, 0.10)
    assert selected.tolist() == [True, True, False]
    merged = fc.mixture_quantile(
        fit.weights[selected], fit.means[selected], fit.sds[selected], 0.999
    )
    independent = bisect_union_quantile(
        fit.weights[selected], fit.means[selected], fit.sds[selected], 0.999
    )
    assert merged == pytest.approx(independent, abs=1e-9)
    lowest_only = fc.mixture_quantile(
        fit.weights[:1], fit.means[:1], fit.sds[:1], 0.999
    )
    assert lowest_only < merged - 0.2


def test_the_stage_function_reports_papkou_lower_than_the_merged_threshold(
    split_bells: pl.DataFrame,
) -> None:
    result = fc.classify_growth_rates(
        split_bells, {**CONFIG, "bic_k_range": [3, 3]}, PUBLISHED
    )
    summary = result.summary
    assert summary["mixture_k"] == 3
    assert summary["selected_components"] == [0, 1]
    papkou = summary["replication_cell"]["threshold"]["rounded"]
    assert papkou < summary["threshold"]["rounded"] - 0.2
    # The canonical T is the union quantile, not any single component's quantile.
    weights, means, sds = canonical_union(result)
    assert len(weights) == 2
    assert summary["threshold"]["unrounded"] == pytest.approx(
        bisect_union_quantile(weights, means, sds, 0.999), abs=1e-9
    )


def test_one_skewed_non_functional_bell_split_in_two_merges_both_halves(
    skewed_bell: pl.DataFrame,
) -> None:
    s = skewed_bell["s"].to_numpy()
    (fit,) = fc.fit_mixtures(s, (3, 3), CONFIG["random_seed"], CONFIG["n_init"])
    shares = fc.double_stop_shares(fit, s, skewed_bell["stop_codon_count"], 2)
    # Both halves of the one non-functional bell carry real double-stop mass.
    assert shares[0] >= 0.10 and shares[1] >= 0.10 and shares[2] < 0.01
    assert fc.select_non_functional(shares, 0.10).tolist() == [True, True, False]

    result = fc.classify_growth_rates(
        skewed_bell, {**CONFIG, "bic_k_range": [3, 3]}, PUBLISHED
    )
    assert result.summary["selected_components"] == [0, 1]
    weights, means, sds = canonical_union(result)
    independent = bisect_union_quantile(weights, means, sds, 0.999)
    assert result.summary["threshold"]["unrounded"] == pytest.approx(
        independent, abs=1e-9
    )
    # Papkou's lowest-mean-only quantile is a different, lower number.
    lowest = bisect_union_quantile(weights[:1], means[:1], sds[:1], 0.999)
    papkou = result.summary["replication_cell"]["threshold"]["unrounded"]
    assert papkou == pytest.approx(lowest, abs=1e-9)
    assert papkou < result.summary["threshold"]["unrounded"] - 0.2


def test_bic_prefers_the_split_on_well_separated_halves() -> None:
    table = growth_table(
        [
            (-1.5, 0.05, 1500, 200, 0),
            (-0.5, 0.05, 1500, 200, 0),
            (0.5, 0.05, 1500, 0, 0),
        ]
    )
    fits = fc.fit_mixtures(
        table["s"].to_numpy(), (1, 5), CONFIG["random_seed"], CONFIG["n_init"]
    )
    assert fc.choose_by_bic(fits).k == 3


# 3. Replication cell --------------------------------------------------------


def test_the_replication_cell_is_reported_beside_the_canonical_threshold(
    two_bells: pl.DataFrame,
) -> None:
    with_three = fc.classify_growth_rates(two_bells, CONFIG, PUBLISHED).summary
    without_three = fc.classify_growth_rates(
        two_bells, {**CONFIG, "bic_k_range": [1, 2]}, PUBLISHED
    ).summary
    for summary in (with_three, without_three):
        cell = summary["replication_cell"]
        assert cell["k"] == 3
        assert cell["published_functional_threshold"] == PUBLISHED
        assert math.isfinite(cell["threshold"]["unrounded"])
        assert cell["threshold"]["rounded"] == round(cell["threshold"]["unrounded"], 6)
        expected = ct.compute_candidate_functional_threshold(
            two_bells["s"].to_list(), {"rule": "median_reflected", "decimals": 6}
        )
        assert cell["median_reflected_threshold"]["rounded"] == expected.threshold
        assert cell["median_reflected_threshold"]["unrounded"] == expected.unrounded
    # The same seed and n_init give the same k = 3 fit in the loop and apart from it.
    assert (
        with_three["replication_cell"]["threshold"]
        == without_three["replication_cell"]["threshold"]
    )


def test_the_components_table_holds_both_fits(two_bells: pl.DataFrame) -> None:
    result = fc.classify_growth_rates(two_bells, CONFIG, PUBLISHED)
    components = result.components
    assert components.columns == list(fc.COMPONENT_COLUMNS)
    replication = components.filter(pl.col("fit") == "replication")
    assert replication["component"].to_list() == [0, 1, 2]
    assert replication["selected"].to_list() == [True, False, False]
    assert replication["mean"].is_sorted()
    canonical = components.filter(pl.col("fit") == "canonical")
    assert canonical["weight"].sum() == pytest.approx(1.0)
    assert canonical["double_stop_responsibility_share"].sum() == pytest.approx(1.0)


def test_summary_counts_and_stop_fractions_are_consistent(
    two_bells: pl.DataFrame,
) -> None:
    result = fc.classify_growth_rates(two_bells, CONFIG, PUBLISHED)
    summary = result.summary
    table = result.classification
    assert table.columns == list(fc.CLASSIFICATION_COLUMNS)
    assert summary["sequence_count"] == table.height == two_bells.height
    assert sum(summary["class_counts"].values()) == table.height
    assert summary["class_counts"]["functional"] == (
        table.filter(pl.col("functional_class") == "functional").height
    )
    assert "point_mass_count" not in summary
    threshold = summary["threshold"]["rounded"]
    doubles = summary["stop_codon_fractions_below_threshold"]["2"]
    assert doubles["sequences"] == 200
    assert doubles["below_threshold"] == int(
        (table.filter(pl.col("stop_codon_count") == 2)["s"] < threshold).sum()
    )
    assert doubles["fraction"] == doubles["below_threshold"] / 200
    assert summary["stop_codon_fractions_below_threshold"]["3"]["fraction"] is None
    assert [item["k"] for item in summary["bic"]] == [1, 2, 3, 4, 5]


def test_no_double_stop_sequences_raises() -> None:
    table = growth_table([(-1.0, 0.15, 500, 0, 50), (0.0, 0.10, 500, 0, 0)])
    with pytest.raises(ValueError, match="No sequence has 2 stop codons"):
        fc.classify_growth_rates(table, CONFIG, PUBLISHED)


def test_a_missing_column_and_a_repeated_sequence_raise(
    two_bells: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        fc.classify_growth_rates(two_bells.drop("se_pooled"), CONFIG, PUBLISHED)
    repeated = two_bells.with_columns(pl.lit("SAME").alias("dna_sequence"))
    with pytest.raises(ValueError, match="repeats"):
        fc.classify_growth_rates(repeated, CONFIG, PUBLISHED)


def test_the_replication_cell_keeps_the_fixed_0_999_quantile(
    two_bells: pl.DataFrame,
) -> None:
    config = {**CONFIG, "quantile": 0.99, "bic_k_range": [2, 2]}
    summary = fc.classify_growth_rates(two_bells, config, PUBLISHED).summary
    cell = summary["replication_cell"]
    assert fc.REPLICATION_QUANTILE == 0.999
    assert cell["quantile"] == 0.999 and cell["k"] == 3
    components = fc.classify_growth_rates(two_bells, config, PUBLISHED).components
    lowest = components.filter(pl.col("fit") == "replication").row(0, named=True)
    assert cell["threshold"]["unrounded"] == pytest.approx(
        norm.ppf(0.999, lowest["mean"], lowest["sd"]), abs=1e-8
    )
    # The canonical threshold does follow the configured quantile.
    low = components.filter(pl.col("fit") == "canonical").row(0, named=True)
    assert summary["threshold"]["unrounded"] == pytest.approx(
        norm.ppf(0.99, low["mean"], low["sd"]), abs=1e-8
    )
    assert summary["settings"]["quantile"] == 0.99


def test_the_median_reflected_comparison_uses_the_candidate_declaration(
    two_bells: pl.DataFrame,
) -> None:
    # Different decimals in the two tables, and the declared rule is the other one:
    # the comparison is still the median_reflected rule at the declared decimals.
    config = {**CONFIG, "decimals": 6}
    declaration = {"rule": "mixture_quantile", "decimals": 2}
    cell = fc.classify_growth_rates(two_bells, config, PUBLISHED, declaration).summary[
        "replication_cell"
    ]
    expected = ct.compute_candidate_functional_threshold(
        two_bells["s"].to_list(), {"rule": "median_reflected", "decimals": 2}
    )
    assert cell["median_reflected_threshold"] == {
        "unrounded": expected.unrounded,
        "rounded": expected.threshold,
    }
    assert expected.threshold == round(expected.unrounded, 2)
    assert expected.threshold != round(expected.unrounded, 6)


@pytest.mark.parametrize("rule", ["fdr_fixed_cutoff", "mixture_quantile"])
def test_classification_preserves_sequence_calls_for_threshold_declarations(
    two_bells: pl.DataFrame, rule: str
) -> None:
    with (ROOT / "config/analysis.toml").open("rb") as stream:
        fdr = tomllib.load(stream)["clamped"]["candidate_functional_threshold"]
    assert fdr["rule"] == "fdr_fixed_cutoff"
    declaration = fdr if rule == "fdr_fixed_cutoff" else {
        "rule": rule, "decimals": fdr["decimals"]
    }
    baseline = fc.classify_growth_rates(two_bells, CONFIG, PUBLISHED, DECLARATION)
    actual = fc.classify_growth_rates(two_bells, CONFIG, PUBLISHED, declaration)
    assert actual.classification.equals(baseline.classification)
    assert actual.summary["replication_cell"] == baseline.summary["replication_cell"]
    assert ct.median_reflected_declaration(declaration) == DECLARATION


# 3b. Convergence ------------------------------------------------------------


def test_a_non_converged_chosen_fit_fails_closed(two_bells: pl.DataFrame) -> None:
    config = {**CONFIG, "max_iter": 1, "tol": 1e-12}
    with pytest.raises(fc.MixtureNotConverged, match="BIC-chosen") as error:
        fc.classify_growth_rates(two_bells, config, PUBLISHED)
    assert isinstance(error.value, RuntimeError)
    assert "max_iter = 1" in str(error.value)


def test_a_non_converged_replication_fit_fails_closed(
    two_bells: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fc._fit_one

    def fit(*args: Any, **kwargs: Any) -> fc.MixtureFit:
        result = original(*args, **kwargs)
        if result.k == fc.REPLICATION_K:
            return dataclasses.replace(result, converged=False)
        return result

    monkeypatch.setattr(fc, "_fit_one", fit)
    # k = 3 lies outside the BIC range, so only the Replication cell fits it.
    with pytest.raises(fc.MixtureNotConverged, match="Replication cell"):
        fc.classify_growth_rates(
            two_bells, {**CONFIG, "bic_k_range": [1, 2]}, PUBLISHED
        )


def test_a_non_chosen_non_converged_fit_is_recorded_not_fatal(
    two_bells: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fc.fit_mixtures

    def fits(*args: Any, **kwargs: Any) -> list[fc.MixtureFit]:
        return [
            dataclasses.replace(fit, converged=False) if fit.k == 1 else fit
            for fit in original(*args, **kwargs)
        ]

    monkeypatch.setattr(fc, "fit_mixtures", fits)
    summary = fc.classify_growth_rates(two_bells, CONFIG, PUBLISHED).summary
    assert summary["mixture_k"] == 2 and summary["converged"] is True
    assert {item["k"]: item["converged"] for item in summary["bic"]}[1] is False


# 3c. Inputs -------------------------------------------------------------------


def test_a_repeated_sequence_id_raises(two_bells: pl.DataFrame) -> None:
    repeated = two_bells.with_columns(pl.lit("SAME").alias("sequence_id"))
    with pytest.raises(ValueError, match="repeats a sequence_id"):
        fc.classify_growth_rates(repeated, CONFIG, PUBLISHED)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_non_finite_s_fails_before_fitting_naming_up_to_five_ids(
    two_bells: pl.DataFrame, bad: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("fitting must not start")

    monkeypatch.setattr(fc, "fit_mixtures", refuse)
    table = two_bells.with_columns(
        pl.when(pl.int_range(pl.len()) < 7).then(bad).otherwise(pl.col("s")).alias("s")
    )
    with pytest.raises(fc.NonFiniteGrowthRate, match="7 invalid rows") as error:
        fc.classify_growth_rates(table, CONFIG, PUBLISHED)
    assert isinstance(error.value, ValueError)
    message = str(error.value)
    assert "'id00000'" in message and "'id00004'" in message
    assert "id00005" not in message


# 4. Classification boundaries ----------------------------------------------


def test_class_boundaries_are_inclusive_at_the_declared_confidence() -> None:
    # 1 - 0.95 is 0.050000000000000044 in floating point; the lower boundary is
    # rounded to 12 decimals, so a Functional probability of exactly 0.05 is
    # Non-functional and the first representable value above it is Uncertain.
    p = [0.95, 0.05, 0.05000000000000004, 0.9499999, 0.0500001, 0.5, 1.0, 0.0]
    assert 1.0 - 0.95 > 0.05
    assert fc.assign_class(p, 0.95).tolist() == [
        "functional",
        "non_functional",
        "uncertain",
        "uncertain",
        "uncertain",
        "uncertain",
        "functional",
        "non_functional",
    ]


def test_probability_is_phi_and_s_equal_to_t_is_uncertain() -> None:
    threshold = -0.5
    se = 0.1
    s = np.array([threshold, threshold + 1.0, threshold - 1.0])
    p = fc.functional_probability(s, [se] * 3, threshold)
    assert p[0] == pytest.approx(0.5)
    assert p == pytest.approx(norm.cdf((s - threshold) / se))
    assert fc.assign_class(p, 0.95).tolist() == [
        "uncertain",
        "functional",
        "non_functional",
    ]
    # One-sided 0.95 boundary: s = T + z * se has probability 0.95.
    edge = fc.functional_probability([threshold + norm.ppf(0.95) * se], [se], threshold)
    assert edge[0] == pytest.approx(0.95)


@pytest.mark.parametrize(
    "bad", [0.0, None, -0.1, float("nan"), float("inf"), float("-inf")]
)
def test_null_zero_negative_or_non_finite_se_raises_naming_the_sequences(
    bad: float | None,
) -> None:
    ids = [f"id{i}" for i in range(8)]
    se: list[float | None] = [0.1] * 8
    se[2], se[5] = bad, bad
    with pytest.raises(fc.NonPositiveStandardError, match="id2") as error:
        fc.functional_probability(np.zeros(8), se, -0.5, ids)
    assert isinstance(error.value, ValueError)
    assert "id5" in str(error.value) and "2 invalid rows" in str(error.value)


def test_the_error_names_at_most_a_few_sequences() -> None:
    ids = [f"id{i}" for i in range(20)]
    with pytest.raises(ValueError) as error:
        fc.functional_probability(np.zeros(20), [-1.0] * 20, 0.0, ids)
    assert "id4" in str(error.value) and "id5" not in str(error.value)


def test_the_wild_type_row_is_functional_when_t_is_negative() -> None:
    table = growth_table([(-1.0, 0.15, 800, 100, 50), (0.0, 0.10, 800, 0, 0)])
    wild_type = pl.DataFrame(
        {
            "dna_sequence": ["WILDTYPE"],
            "sequence_id": ["wt"],
            "s": [0.0],
            "se_pooled": [0.02],
            "stop_codon_count": [0],
        }
    )
    result = fc.classify_growth_rates(pl.concat([table, wild_type]), CONFIG, PUBLISHED)
    assert result.summary["threshold"]["rounded"] < 0
    row = result.classification.filter(pl.col("dna_sequence") == "WILDTYPE")
    assert row["functional_class"][0] == "functional"
    assert row["functional_probability"][0] > 0.95


def test_a_null_or_zero_se_row_fails_the_whole_classification(
    two_bells: pl.DataFrame,
) -> None:
    for value in (None, 0.0):
        table = two_bells.with_columns(
            pl.when(pl.int_range(pl.len()) < 3)
            .then(value)
            .otherwise(pl.col("se_pooled"))
            .alias("se_pooled")
        )
        with pytest.raises(fc.NonPositiveStandardError, match="id00000"):
            fc.classify_growth_rates(table, CONFIG, PUBLISHED)


# 5. Config validation -------------------------------------------------------


def real_config() -> dict[str, Any]:
    with (ROOT / "config/analysis.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_the_checked_in_config_is_valid() -> None:
    analysis = real_config()
    settings = fc.validate_config(analysis[fc.SECTION])
    assert settings.k_range == (1, 5)
    assert settings.stop_anchor_min_responsibility == 0.10
    assert settings.stop_codon_count_for_anchor == 2
    assert settings.quantile == 0.999
    assert settings.functional_confidence == 0.95
    assert settings.decimals == 6
    assert settings.max_iter == 1000 and settings.tol == 1e-6
    # fdr_fixed_cutoff (median_reflected, BY q=0.05) is the active Selection rule
    # (user lock 2026-11-12); mixture_quantile remains as config/variants/mixture_quantile.toml.
    declaration = analysis["clamped"]["candidate_functional_threshold"]
    assert declaration["rule"] == "fdr_fixed_cutoff"
    assert declaration["cutoff"] == "median_reflected"
    assert declaration["method"] == "by"
    assert declaration["q"] == 0.05
    ct.validate_candidate_declaration(declaration)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bic_k_range", [0, 3]),
        ("bic_k_range", [4, 2]),
        ("bic_k_range", [1]),
        ("bic_k_range", [1.0, 3]),
        ("bic_k_range", "1-5"),
        ("bic_k_range", [True, 3]),
        ("stop_anchor_min_responsibility", 0.0),
        ("stop_anchor_min_responsibility", 1.0),
        ("stop_anchor_min_responsibility", "0.1"),
        ("stop_codon_count_for_anchor", 0),
        ("stop_codon_count_for_anchor", 2.0),
        ("quantile", 0.0),
        ("quantile", 1.0),
        ("quantile", float("nan")),
        ("functional_confidence", 0.5),
        ("functional_confidence", 1.0),
        ("decimals", -1),
        ("decimals", 6.0),
        ("decimals", True),
        ("random_seed", -1),
        ("random_seed", 1.5),
        ("n_init", 0),
        ("n_init", True),
        ("max_iter", 0),
        ("max_iter", 10.0),
        ("max_iter", True),
        ("tol", 0.0),
        ("tol", -1e-6),
        ("tol", float("inf")),
        ("tol", "1e-6"),
    ],
)
def test_bad_config_values_are_rejected(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=key):
        fc.validate_config({**CONFIG, key: value})


def test_unknown_missing_and_non_table_config_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        fc.validate_config({**CONFIG, "n_jobs": 4})
    with pytest.raises(ValueError, match="lacks"):
        fc.validate_config({k: v for k, v in CONFIG.items() if k != "n_init"})
    with pytest.raises(TypeError):
        fc.validate_config([1, 5])
    assert fc.validate_config(CONFIG).n_init == 3
    with pytest.raises(ValueError, match="lacks"):
        fc.validate_config({k: v for k, v in CONFIG.items() if k != "max_iter"})


def test_load_settings_validates_without_importing_scikit_learn() -> None:
    code = (
        "import sys\n"
        "from dubanevics2026topological.analysis import load_settings\n"
        "load_settings()\n"
        # The threshold package imports its classification module, but fitting
        # dependencies must remain lazy when settings are loaded.
        "assert 'sklearn' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, cwd=ROOT)


def test_the_candidate_declaration_accepts_both_rules_and_rejects_others() -> None:
    assert ct.RULES == ("median_reflected", "mixture_quantile", "fdr_fixed_cutoff", "fixed_cutoff")
    for rule in ct.RULES:
        declaration = {"rule": rule, "decimals": 6}
        if rule == "fdr_fixed_cutoff":
            declaration.update(
                cutoff="median_reflected", method="by", q=0.05,
                input_dir="reference",
            )
        if rule == "fixed_cutoff":
            declaration["value"] = -0.5
        assert ct.validate_candidate_declaration(declaration) == (
            rule,
            6,
        )
    with pytest.raises(ValueError, match="Unknown Candidate functional threshold rule"):
        ct.validate_candidate_declaration({"rule": "lowest_mean", "decimals": 6})
    with pytest.raises(ValueError, match="Unknown keys"):
        ct.validate_candidate_declaration(
            {"rule": "median_reflected", "decimals": 6, "quantile": 0.9}
        )
    with pytest.raises(ValueError, match="Unknown keys"):
        ct.validate_candidate_declaration(
            {"rule": "fdr_fixed_cutoff", "decimals": 6, "quantile": 0.9}
        )
    with pytest.raises(ValueError, match="Unknown keys"):
        ct.validate_candidate_declaration(
            {"rule": "fixed_cutoff", "decimals": 6, "value": -0.5, "extra": 1}
        )


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), float("-inf")])
def test_candidate_fixed_cutoff_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="finite number"):
        ct.validate_candidate_declaration({"rule": "fixed_cutoff", "decimals": 6, "value": value})


def test_candidate_fixed_cutoff_requires_value() -> None:
    with pytest.raises(ValueError, match="finite number"):
        ct.validate_candidate_declaration({"rule": "fixed_cutoff", "decimals": 6})


def test_the_raw_fitness_computation_refuses_the_mixture_rule_without_a_summary() -> (
    None
):
    declaration = {"rule": "mixture_quantile", "decimals": 6}
    with pytest.raises(ValueError, match="classification summary"):
        ct.compute_candidate_functional_threshold([0.0, 1.0], declaration)
    from_summary = ct.compute_candidate_functional_threshold(
        [-1.0, 0.0, 1.0],
        declaration,
        classification_summary={
            "threshold": {"unrounded": -0.5000004, "rounded": -0.5}
        },
    )
    # T comes from the summary; m, g and h are still the Full graph's.
    assert (from_summary.threshold, from_summary.unrounded) == (-0.5, -0.5000004)
    assert (from_summary.median, from_summary.global_minimum) == (0.0, -1.0)
    value = ct.compute_candidate_functional_threshold(
        [-1.0, 0.0, 1.0], {"rule": "median_reflected", "decimals": 6}
    )
    assert value.threshold == 1.0


# 6. Stage fingerprint -------------------------------------------------------


def stage_settings(**changes: Any) -> dict[str, Any]:
    analysis = {
        fc.SECTION: {**CONFIG, **changes.pop("config", {})},
        "clamped": {
            "published_functional_threshold": changes.pop("published", PUBLISHED),
            "candidate_functional_threshold": changes.pop("declaration", DECLARATION),
        },
    }
    return {"analysis": analysis}


def test_the_fingerprint_is_stable_and_follows_scientific_settings(
    tmp_path: Path,
) -> None:
    table = tmp_path / "growth_rates.parquet"
    pl.DataFrame({"s": [0.0]}).write_parquet(table)

    def fingerprint(settings: dict[str, Any]) -> str:
        return stage_fingerprint(
            fc.functional_classification_stage(
                settings, root=Path(tmp_path.anchor), growth_rates=table
            )
        )

    base = fingerprint(stage_settings())
    assert base == fingerprint(copy.deepcopy(stage_settings()))
    for change in (
        {"config": {"quantile": 0.99}},
        {"config": {"bic_k_range": [1, 4]}},
        {"config": {"random_seed": 1}},
        {"config": {"max_iter": 500}},
        {"config": {"tol": 1e-8}},
        {"declaration": {"rule": "median_reflected", "decimals": 3}},
        {"config": {"functional_confidence": 0.9}},
        {"published": -0.5},
    ):
        assert fingerprint(stage_settings(**change)) != base
    pl.DataFrame({"s": [0.0, 1.0]}).write_parquet(table)
    assert fingerprint(stage_settings()) != base


def test_the_stage_is_built_through_the_project_factory(tmp_path: Path) -> None:
    table = tmp_path / "g.parquet"
    pl.DataFrame({"s": [0.0]}).write_parquet(table)
    stage = fc.functional_classification_stage(
        stage_settings(), root=Path(tmp_path.anchor), growth_rates=table
    )
    assert stage.name == "functional-classification"
    assert set(stage.execution_keys) == {"n_jobs", "chunk_size"}
    assert tuple(stage.inputs) == (table,)
    assert {path.name for path in stage.code} == {
        "functional_classification.py",
        "functional_threshold.py",
        "functional_classification_config.py",
    }
    assert set(stage.settings["libraries"]) == {"scikit-learn", "scipy"}


def test_the_stage_fingerprint_tracks_the_moved_threshold_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = tmp_path / "src/pkg/analysis"
    threshold_dir = tmp_path / "src/pkg/threshold"
    analysis_dir.mkdir(parents=True)
    threshold_dir.mkdir(parents=True)
    classifier = analysis_dir / "functional_classification.py"
    classifier.write_text("# classifier\n", encoding="utf-8")
    implementation = threshold_dir / "functional_threshold.py"
    implementation.write_text("# threshold-v1\n", encoding="utf-8")
    config = threshold_dir / "functional_classification_config.py"
    config.write_text("# config\n", encoding="utf-8")
    table = tmp_path / "growth_rates.parquet"
    pl.DataFrame({"s": [0.0]}).write_parquet(table)
    monkeypatch.setattr(fc, "__file__", str(classifier))

    stage = fc.functional_classification_stage(
        stage_settings(), root=tmp_path, growth_rates=table
    )
    assert implementation.resolve() in {Path(path).resolve() for path in stage.code}
    baseline = stage_fingerprint(stage)
    assert baseline == stage_fingerprint(
        fc.functional_classification_stage(
            stage_settings(), root=tmp_path, growth_rates=table
        )
    )

    from dubanevics2026topological.analysis.produce_or_load import (
        _declared,
        _label_block,
    )

    before_code = _label_block(stage, _declared(stage))["code"]
    implementation.write_text("# threshold-v2\n", encoding="utf-8")
    changed = fc.functional_classification_stage(
        stage_settings(), root=tmp_path, growth_rates=table
    )
    assert stage_fingerprint(changed) == baseline
    assert _label_block(changed, _declared(changed))["code"] != before_code


def test_the_stage_publishes_exports_and_reuses(
    two_bells: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological.analysis import sequence_growth_rates as growth

    table = tmp_path / "growth_rates.parquet"
    two_bells.write_parquet(table)
    # The temporary table lies outside the repository, so key paths against `/`.
    monkeypatch.setattr(fc, "ROOT", Path(tmp_path.anchor))
    monkeypatch.setattr(fc, "growth_rates_path", lambda context=None: table)
    monkeypatch.setattr(
        fc, "checkpoint_dir", lambda context=None: tmp_path / "checkpoints" / "fc"
    )
    monkeypatch.setattr(
        fc, "export_dir", lambda context=None: tmp_path / "analysis" / "fc"
    )
    monkeypatch.setattr(
        growth,
        "load_cached_sequence_growth_rates",
        lambda context=None: SimpleNamespace(growth_rates=pl.read_parquet(table)),
    )
    settings = stage_settings()
    calls: list[int] = []
    produce = fc.classify_growth_rates

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return produce(*args, **kwargs)

    monkeypatch.setattr(fc, "classify_growth_rates", counting)
    first = fc.compute_functional_classification(settings)
    assert len(calls) == 1
    exported = tmp_path / "analysis" / "fc"
    assert sorted(path.name for path in exported.iterdir()) == sorted(fc.ARTEFACTS)
    assert pl.read_parquet(exported / "classification.parquet").columns == list(
        fc.CLASSIFICATION_COLUMNS
    )
    again = fc.compute_functional_classification(settings)
    assert len(calls) == 1, "the second run must load the checkpoint, not compute"
    assert again.summary == first.summary
    assert fc.load_cached_functional_classification(settings).summary == first.summary
    assert len(calls) == 1
    # A changed scientific setting is a different fingerprint: it computes again.
    fc.compute_functional_classification(stage_settings(config={"n_init": 2}))
    assert len(calls) == 2
    assert "T = " in fc.describe(first)


def test_the_stage_fails_closed_without_the_growth_rate_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fc, "growth_rates_path", lambda context=None: tmp_path / "absent.parquet"
    )
    with pytest.raises(FileNotFoundError, match="just sequence-growth-rates"):
        fc.compute_functional_classification(stage_settings())


def test_a_missing_section_is_a_registry_gap_not_a_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dubanevics2026topological.analysis import manuscript_quantities as mq

    settings = {"analysis": {"clamped": {}}}
    with pytest.raises(FileNotFoundError, match="functional_classification"):
        fc.load_cached_functional_classification(settings)
    with pytest.raises(FileNotFoundError, match="functional_classification"):
        fc.compute_functional_classification(settings)
    readers = {
        "functional_classification": lambda: fc.load_cached_functional_classification(
            settings
        )
    }
    # The registry treats FileNotFoundError as "no current checkpoint": a gap row.
    assert mq._cached(readers, "functional_classification") is None
