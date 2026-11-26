"""The mixture families of the functional-threshold sensitivity scan.

Small seeded draws only; nothing here reads ``data/raw/``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.optimize import check_grad
from scipy.stats import skewnorm
from scipy.stats import t as student
from test_functional_classification import (
    CONFIG,
    DECLARATION,
    PUBLISHED,
    bisect_union_quantile,
    growth_table,
)

from dubanevics2026topological.analysis import mixture_families as mf
from dubanevics2026topological.threshold import functional_classification as fc


@pytest.fixture(scope="module")
def two_bells() -> pl.DataFrame:
    return growth_table([(-1.0, 0.15, 1500, 200, 100), (0.0, 0.10, 1500, 0, 0)])


FIT = {"seed": 11, "n_init": 2, "max_iter": 500, "tol": 1e-7}


def fit_family(name: str, values: np.ndarray, k: int, **changes):
    return mf.get_family(name).fit(values, k, **{**FIT, **changes})


def bisect_cdf(cdf, low: float, high: float, q: float) -> float:
    for _ in range(300):
        mid = (low + high) / 2
        low, high = (mid, high) if cdf(mid) < q else (low, mid)
    return (low + high) / 2


# 1. Gaussian: identical to the stage ---------------------------------------


def test_the_gaussian_family_reproduces_the_stages_threshold_exactly(
    two_bells: pl.DataFrame,
) -> None:
    result = fc.classify_growth_rates(two_bells, CONFIG, PUBLISHED, DECLARATION)
    s = two_bells["s"].to_numpy()
    stops = two_bells["stop_codon_count"].to_numpy()
    fit = mf.get_family("gaussian").fit(
        s,
        result.summary["mixture_k"],
        seed=CONFIG["random_seed"],
        n_init=CONFIG["n_init"],
        max_iter=CONFIG["max_iter"],
        tol=CONFIG["tol"],
    )
    anchored = stops == CONFIG["stop_codon_count_for_anchor"]
    shares = fit.responsibilities(s[anchored]).mean(axis=0)
    selected = fc.select_non_functional(
        shares, CONFIG["stop_anchor_min_responsibility"]
    )
    assert (
        fit.union_quantile(selected, CONFIG["quantile"])
        == (result.summary["threshold"]["unrounded"])
    )
    assert fit.converged and fit.family == "gaussian"
    assert fit.param_names == ("mean", "sd")


# 2. Skew-normal -------------------------------------------------------------


@pytest.mark.parametrize("shape", [6.0, -6.0])
def test_skew_normal_recovers_the_sign_of_a_known_shape(shape: float) -> None:
    x = skewnorm.rvs(shape, loc=0.3, scale=0.5, size=4000, random_state=3)
    fit = fit_family("skew_normal", x, 1)
    loc, scale, a = fit.params[0]
    assert np.sign(a) == np.sign(shape) and abs(a) > 2.0
    assert loc == pytest.approx(0.3, abs=0.1)
    assert scale == pytest.approx(0.5, rel=0.15)
    assert fit.converged
    # Never worse than the nested Gaussian (a = 0) on its own data.
    gaussian = fit_family("gaussian", x, 1)
    assert fit.log_likelihood > gaussian.log_likelihood


def test_skew_normal_logpdf_matches_scipy() -> None:
    x = np.linspace(-2, 3, 41)
    p = np.array([0.2, 0.7, -3.0])
    assert mf.skew_normal_logpdf(x, p) == pytest.approx(
        skewnorm.logpdf(x, p[2], loc=p[0], scale=p[1]), abs=1e-12
    )


# 3. Student-t ---------------------------------------------------------------


def test_student_t_recovers_heavy_tails_and_goes_large_on_normal_data() -> None:
    heavy = student.rvs(3.0, loc=0.2, scale=0.4, size=6000, random_state=5)
    fit = fit_family("student_t", heavy, 1)
    assert fit.converged
    assert mf.DF_MIN <= fit.params[0, 2] < 6.0
    assert fit.params[0, 0] == pytest.approx(0.2, abs=0.05)
    normal = np.random.default_rng(6).normal(0.0, 1.0, 6000)
    assert fit_family("student_t", normal, 1).params[0, 2] > 30.0


def test_student_t_df_is_bounded_below() -> None:
    cauchy = student.rvs(1.0, size=3000, random_state=8)
    fit = fit_family("student_t", cauchy, 1)
    assert fit.params[0, 2] >= mf.DF_MIN - 1e-9


def test_student_t_logpdf_matches_scipy() -> None:
    x = np.linspace(-3, 3, 41)
    p = np.array([0.1, 0.8, 4.5])
    assert mf.student_t_logpdf(x, p) == pytest.approx(
        student.logpdf(x, p[2], loc=p[0], scale=p[1]), abs=1e-12
    )


# 4. Gradients ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["skew_normal", "student_t"])
def test_the_m_step_gradient_matches_finite_differences(name: str) -> None:
    spec = mf._SPECS[name]
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 300) + rng.gamma(2.0, 0.3, 300)
    r = rng.uniform(0.1, 1.0, 300)
    theta = np.array([0.3, np.log(0.9), 2.0 if name == "skew_normal" else 6.0])
    error = check_grad(
        lambda t: spec.nll(t, x, r)[0], lambda t: spec.nll(t, x, r)[1], theta
    )
    assert error < 1e-5


# 5. Union quantile ----------------------------------------------------------


def three_component_data(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(-1.0, 0.12, 1500),
            -0.4 + rng.gamma(2.5, 0.08, 1000),
            rng.normal(0.5, 0.15, 1500),
        ]
    )


def test_the_gaussian_union_quantile_matches_a_bisection_oracle() -> None:
    fit = fit_family("gaussian", three_component_data(), 3)
    mask = np.array([True, True, False])
    rows = fit.params[mask]
    assert fit.union_quantile(mask, 0.999) == pytest.approx(
        bisect_union_quantile(fit.weights[mask], rows[:, 0], rows[:, 1], 0.999),
        abs=1e-9,
    )


@pytest.mark.parametrize(
    ("name", "cdf"),
    [
        ("skew_normal", lambda x, p: skewnorm.cdf(x, p[2], loc=p[0], scale=p[1])),
        ("student_t", lambda x, p: student.cdf(x, p[2], loc=p[0], scale=p[1])),
    ],
)
@pytest.mark.parametrize("quantile", [0.95, 0.999, 0.9999])
def test_the_union_quantile_matches_bisection_for_non_gaussian_families(
    name, cdf, quantile: float
) -> None:
    fit = fit_family(name, three_component_data(), 3)
    mask = np.array([True, True, False])
    w = fit.weights[mask] / fit.weights[mask].sum()
    rows = fit.params[mask]

    def union(x: float) -> float:
        return float(sum(wj * cdf(x, row) for wj, row in zip(w, rows)))

    oracle = bisect_cdf(union, -50.0, 50.0, quantile)
    assert fit.union_quantile(mask, quantile) == pytest.approx(oracle, abs=1e-8)


def test_the_union_quantile_brackets_a_far_heavy_tail() -> None:
    fit = fit_family("student_t", student.rvs(2.2, size=3000, random_state=1), 1)
    far = fit.union_quantile([True], 0.9999)
    assert far > fit.union_quantile([True], 0.99) > fit.union_quantile([True], 0.95)
    assert student.cdf(
        far, fit.params[0, 2], loc=fit.params[0, 0], scale=fit.params[0, 1]
    ) == pytest.approx(0.9999, abs=1e-9)


def test_a_bad_mask_or_quantile_is_rejected() -> None:
    fit = fit_family("skew_normal", three_component_data(), 2)
    with pytest.raises(ValueError, match="entries"):
        fit.union_quantile([True], 0.9)
    with pytest.raises(ValueError, match="No component"):
        fit.union_quantile([False, False], 0.9)
    with pytest.raises(ValueError, match="quantile"):
        fit.union_quantile([True, False], 1.0)


# 6. Fit bookkeeping ---------------------------------------------------------


@pytest.mark.parametrize("name", list(mf.FAMILY_NAMES))
def test_fits_are_deterministic_ordered_and_consistent(name: str) -> None:
    x = three_component_data()
    first = fit_family(name, x, 3)
    again = fit_family(name, x, 3)
    assert np.array_equal(first.params, again.params)
    assert first.bic == again.bic and first.log_likelihood == again.log_likelihood
    means = first.component_means
    assert np.all(np.diff(means) > 0)
    assert first.weights.sum() == pytest.approx(1.0)
    resp = first.responsibilities(x)
    assert resp.shape == (x.size, 3)
    assert resp.sum(axis=1) == pytest.approx(1.0)
    # The lowest-mean component owns the low bell.
    assert resp[:1500, 0].mean() > 0.9
    cdf = first.component_cdf(x, 0)
    assert cdf.shape == x.shape and cdf.min() >= 0 and cdf.max() <= 1
    n_params = len(first.param_names)
    assert first.bic == pytest.approx(
        -2 * first.log_likelihood + (3 * n_params + 2) * np.log(x.size), rel=1e-9
    )


def test_a_gaussian_init_is_reused_and_must_match() -> None:
    x = three_component_data()
    base = fit_family("gaussian", x, 2)
    reused = fit_family("student_t", x, 2, init=base)
    assert reused.k == 2
    with pytest.raises(ValueError, match="init"):
        fit_family("student_t", x, 3, init=base)


def test_a_fit_that_has_not_converged_says_so() -> None:
    fit = fit_family("skew_normal", three_component_data(), 3, max_iter=1)
    assert fit.converged is False and fit.n_iter == 1


@pytest.mark.parametrize("name", ["skew_normal", "student_t"])
def test_convergence_is_the_absolute_mean_log_likelihood_change(name: str) -> None:
    # sd = 1 / sqrt(2 pi e) puts the mean log-likelihood at about zero, where a rule
    # relative to |LL| would never be met.
    x = np.random.default_rng(4).normal(0.0, 1.0 / np.sqrt(2 * np.pi * np.e), 3000)
    fit = fit_family(name, x, 1, tol=1e-6, max_iter=500)
    assert abs(fit.log_likelihood) / x.size < 0.01
    assert fit.converged and fit.n_iter < 500


@pytest.mark.parametrize("name", ["skew_normal", "student_t"])
def test_a_degenerate_component_keeps_its_scale_above_the_floor(name: str) -> None:
    rng = np.random.default_rng(12)
    x = np.concatenate([np.full(200, 0.5), rng.normal(0.0, 0.3, 800)])
    fit = fit_family(name, x, 2)
    assert np.isfinite(fit.log_likelihood) and np.isfinite(fit.bic)
    assert fit.params[:, 1].min() >= np.exp(mf.LOG_SCALE_MIN) * (1 - 1e-9)


def test_large_samples_are_binned_but_report_the_exact_likelihood() -> None:
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.normal(-0.7, 0.07, 30000), rng.normal(0.0, 0.2, 30000)])
    assert x.size > mf.BIN_THRESHOLD
    fit = fit_family("student_t", x, 2)
    exact = sum(
        w * np.exp(mf.student_t_logpdf(x, row))
        for w, row in zip(fit.weights, fit.params)
    )
    assert fit.log_likelihood == pytest.approx(float(np.log(exact).sum()), rel=1e-9)
    assert fit.params[0, 0] == pytest.approx(-0.7, abs=0.01)


def test_input_validation_and_family_lookup() -> None:
    with pytest.raises(ValueError, match="Unknown mixture family"):
        mf.get_family("laplace")
    with pytest.raises(ValueError, match="finite"):
        fit_family("gaussian", np.array([0.0, np.nan, 1.0]), 1)
    with pytest.raises(ValueError, match="Cannot fit"):
        fit_family("skew_normal", np.array([0.0, 1.0]), 3)
    fit = fit_family("gaussian", three_component_data(), 2)
    with pytest.raises(IndexError):
        fit.component_cdf([0.0], 2)


def test_choose_by_bic_prefers_the_lowest_bic() -> None:
    x = three_component_data()
    fits = [fit_family("gaussian", x, k) for k in (1, 2, 3)]
    assert mf.choose_by_bic(fits).k == 3
    assert mf.choose_by_bic(list(reversed(fits))).k == 3
    with pytest.raises(ValueError, match="No mixture fits"):
        mf.choose_by_bic([])
