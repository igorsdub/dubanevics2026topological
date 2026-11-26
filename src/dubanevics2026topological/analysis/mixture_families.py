"""One-dimensional mixture families for the functional-threshold sensitivity scan.

The classification stage (``functional_classification``) fits Gaussian mixtures only.
The sensitivity scan (ADR 2026-10-24-mixture-quantile-functional-classification,
Decision 7) also asks how far the threshold ``T`` moves when the non-functional
component is allowed to be skewed or heavy-tailed. This module gives the three
families one interface so the scan treats them alike:

- ``gaussian``: delegates to the stage's own scikit-learn code
  (``functional_classification._fit_one`` and ``mixture_quantile``), so a fit and its
  ``T`` are identical to the stage's for the same seed, ``n_init``, ``max_iter`` and
  ``tol``.
- ``skew_normal``: ``scipy.stats.skewnorm`` components (location, scale, shape).
- ``student_t``: ``scipy.stats.t`` components (location, scale, df), df in
  ``[DF_MIN, DF_MAX]``.

Non-Gaussian fits use generalised expectation-maximisation (EM). The E-step is one
vectorised ``n x k`` log-density matrix and a log-sum-exp. The M-step is, per
component, a weighted maximum-likelihood fit by L-BFGS-B on the weighted negative
log-likelihood with an analytic gradient (checked against finite differences in the
tests), warm-started from the previous parameters and capped at ``MSTEP_MAXITER``
iterations, which is enough for the likelihood to rise without solving each M-step to
the end. Points whose responsibility for a component is below ``RESP_FLOOR`` are left
out of that component's M-step (they carry less than ``n * RESP_FLOOR`` total weight).

Initialisation is the Gaussian fit of the same ``k``: per component, the weighted
method-of-moments skew-normal (a symmetric start is a saddle) or a Student-t with df 10
and the Gaussian's variance, so
the only random input is the Gaussian fit's ``seed``; the EM itself is deterministic.
``n_init`` therefore means the number of Gaussian restarts. A caller that has already
fitted the Gaussian of that ``k`` passes it as ``init`` and no second Gaussian fit is
made.

Convergence is scikit-learn's rule: the absolute change of the mean log-likelihood per
value, ``|dLL| / n <= tol``, within ``max_iter`` EM iterations (so ``tol`` means what
it does in ``[functional_classification]`` even where the total log-likelihood is
near zero). ``FamilyFit.converged`` records it; a fit that has not
converged is returned, never hidden, and the caller decides.

BIC counts ``k * n_params + (k - 1)`` parameters (location, scale and shape or df per
component, plus the free weights), ``n_params`` 3 for the two non-Gaussian families
and 2 for the Gaussian, as scikit-learn counts them.

Cost and binning. Above ``BIN_THRESHOLD`` values the EM runs on ``BIN_COUNT`` bin means
with their counts (each bin keeps its first moment, and is about 1e-4 of the range
wide, far below any component's scale), which makes an EM iteration independent of
``n``; the exact log-likelihood (and so BIC) is evaluated once on the full data at the
end. Measured on a laptop core on 261,333 synthetic values shaped like the growth
table (``n_init`` 3): Gaussian (scikit-learn) 0.1 s at k = 1 and 14-20 s at k = 3-5;
skew-normal and Student-t EM from a given Gaussian fit 0.03-11 s, at most about 11 s
at k = 5 (4 s for Student-t).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import digamma, gammaln, log_ndtr, logsumexp
from scipy.stats import norm, skewnorm
from scipy.stats import t as student

from ..threshold import functional_classification as fc

GAUSSIAN = "gaussian"
SKEW_NORMAL = "skew_normal"
STUDENT_T = "student_t"
FAMILY_NAMES = (GAUSSIAN, SKEW_NORMAL, STUDENT_T)

DF_MIN = 2.01
DF_MAX = 1000.0
DF_INIT = 10.0
SHAPE_BOUND = 30.0
MSTEP_MAXITER = 25
RESP_FLOOR = 1e-8
MIN_POINTS = 5
BIN_THRESHOLD = 20000
BIN_COUNT = 20000
# Lower bound of the M-step's log-scale: scikit-learn adds reg_covar = 1e-6 to every
# variance, so its component sd is at least sqrt(1e-6); this mirrors it and stops a
# component collapsing onto a few points (log-likelihood going to infinity).
LOG_SCALE_MIN = 0.5 * float(np.log(1e-6))
_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_2 = float(np.log(2.0))


# --------------------------------------------------------------------------
# Component distributions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    """Densities of one family: log-pdf, cdf, mean and the weighted-NLL for the M-step.

    ``params`` is ``(location, scale, extra)`` (``(mean, sd)`` for the Gaussian).
    ``theta`` is the optimiser's parametrisation: ``(location, log scale, extra)``.
    """

    name: str
    param_names: tuple[str, ...]
    logpdf: Callable[[np.ndarray, np.ndarray], np.ndarray]
    cdf: Callable[[Any, np.ndarray], Any]
    mean: Callable[[np.ndarray], float]
    # (theta, x, r) -> (negative weighted mean log-likelihood, gradient wrt theta)
    nll: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[float, np.ndarray]] | None
    bounds: tuple[tuple[float | None, float | None], ...] | None
    # init_extra: the extra parameter a Gaussian component starts from.
    init_extra: float


def _gaussian_logpdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    z = (x - p[0]) / p[1]
    return -0.5 * z * z - 0.5 * _LOG_2PI - np.log(p[1])


def _gaussian_cdf(x: Any, p: np.ndarray) -> Any:
    return norm.cdf(x, loc=p[0], scale=p[1])


def skew_normal_logpdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """``log skewnorm.pdf(x, a, loc, scale)`` with ``p = (loc, scale, a)``."""
    loc, scale, a = p
    z = (x - loc) / scale
    return _LOG_2 - np.log(scale) - 0.5 * z * z - 0.5 * _LOG_2PI + log_ndtr(a * z)


def _skew_normal_nll(
    theta: np.ndarray, x: np.ndarray, r: np.ndarray
) -> tuple[float, np.ndarray]:
    loc, u, a = theta
    scale = np.exp(u)
    z = (x - loc) / scale
    az = a * z
    lcdf = log_ndtr(az)
    # Inverse Mills ratio phi(az) / Phi(az), in logs so that az << 0 does not overflow.
    ratio = np.exp(-0.5 * az * az - 0.5 * _LOG_2PI - lcdf)
    logp = _LOG_2 - u - 0.5 * z * z - 0.5 * _LOG_2PI + lcdf
    total = r.sum()
    value = -float(np.dot(r, logp)) / total
    grad = (
        -np.array(
            [
                np.dot(r, (z - a * ratio) / scale),
                np.dot(r, -1.0 + z * z - a * z * ratio),
                np.dot(r, z * ratio),
            ]
        )
        / total
    )
    return value, grad


def _skew_normal_mean(p: np.ndarray) -> float:
    loc, scale, a = p
    delta = a / np.sqrt(1.0 + a * a)
    return float(loc + scale * delta * np.sqrt(2.0 / np.pi))


def student_t_logpdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """``log t.pdf(x, df, loc, scale)`` with ``p = (loc, scale, df)``."""
    loc, scale, nu = p
    z = (x - loc) / scale
    return (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
        - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    )


def _student_t_nll(
    theta: np.ndarray, x: np.ndarray, r: np.ndarray
) -> tuple[float, np.ndarray]:
    loc, u, nu = theta
    scale = np.exp(u)
    z = (x - loc) / scale
    z2 = z * z
    logp = (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - u
        - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    )
    total = r.sum()
    value = -float(np.dot(r, logp)) / total
    denominator = nu + z2
    d_nu = 0.5 * (
        digamma(0.5 * (nu + 1.0))
        - digamma(0.5 * nu)
        - 1.0 / nu
        - np.log1p(z2 / nu)
        + (nu + 1.0) * z2 / (nu * denominator)
    )
    grad = (
        -np.array(
            [
                np.dot(r, (nu + 1.0) * z / (denominator * scale)),
                np.dot(r, -1.0 + (nu + 1.0) * z2 / denominator),
                np.dot(r, d_nu),
            ]
        )
        / total
    )
    return value, grad


_SPECS = {
    GAUSSIAN: _Spec(
        GAUSSIAN,
        ("mean", "sd"),
        _gaussian_logpdf,
        _gaussian_cdf,
        lambda p: float(p[0]),
        None,
        None,
        0.0,
    ),
    SKEW_NORMAL: _Spec(
        SKEW_NORMAL,
        ("loc", "scale", "shape"),
        skew_normal_logpdf,
        lambda x, p: skewnorm.cdf(x, p[2], loc=p[0], scale=p[1]),
        _skew_normal_mean,
        _skew_normal_nll,
        ((None, None), (LOG_SCALE_MIN, None), (-SHAPE_BOUND, SHAPE_BOUND)),
        0.0,
    ),
    STUDENT_T: _Spec(
        STUDENT_T,
        ("loc", "scale", "df"),
        student_t_logpdf,
        lambda x, p: student.cdf(x, p[2], loc=p[0], scale=p[1]),
        lambda p: float(p[0]),
        _student_t_nll,
        ((None, None), (LOG_SCALE_MIN, None), (DF_MIN, DF_MAX)),
        DF_INIT,
    ),
}


# --------------------------------------------------------------------------
# The fitted mixture
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FamilyFit:
    """One fitted mixture; components are numbered by ascending mean.

    ``params`` has one row per component and one column per entry of
    ``param_names`` (``(mean, sd)``, ``(loc, scale, shape)`` or ``(loc, scale, df)``).
    """

    family: str
    k: int
    bic: float
    converged: bool
    weights: np.ndarray
    params: np.ndarray
    log_likelihood: float
    n_iter: int
    param_names: tuple[str, ...]
    _spec: _Spec = field(repr=False)
    _gaussian: fc.MixtureFit | None = field(default=None, repr=False)

    @property
    def component_means(self) -> np.ndarray:
        """Each component's mean (for the lowest-mean rule)."""
        return np.array([self._spec.mean(row) for row in self.params], dtype=float)

    def responsibilities(self, values: Any) -> np.ndarray:
        """Posterior responsibilities, ``len(values) x k``, columns by component."""
        x = np.asarray(values, dtype=float)
        if self._gaussian is not None:
            return self._gaussian.responsibilities(x)
        logp = _log_joint(self._spec, x, self.weights, self.params)
        return np.exp(logp - logsumexp(logp, axis=1, keepdims=True))

    def component_cdf(self, values: Any, i: int) -> np.ndarray:
        """CDF of component ``i`` at ``values``."""
        if not 0 <= i < self.k:
            raise IndexError(f"Component {i} is outside 0..{self.k - 1}.")
        return np.asarray(
            self._spec.cdf(np.asarray(values, dtype=float), self.params[i])
        )

    def union_quantile(self, selected_mask: Any, quantile: float) -> float:
        """The ``quantile`` of the weight-renormalised union of the selected components.

        The root of ``sum_j w_j F_j(x) / sum_j w_j - quantile`` by ``brentq``, ``j``
        over the selected components. The Gaussian family calls the stage's own
        ``mixture_quantile``.
        """
        mask = np.asarray(selected_mask, dtype=bool)
        if mask.shape != (self.k,):
            raise ValueError(f"selected_mask must have {self.k} entries.")
        if not mask.any():
            raise ValueError("No component is selected.")
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile!r}.")
        weights = self.weights[mask]
        rows = self.params[mask]
        if self.family == GAUSSIAN:
            return fc.mixture_quantile(weights, rows[:, 0], rows[:, 1], quantile)
        weights = weights / weights.sum()
        spec = self._spec

        def gap(x: float) -> float:
            return (
                float(sum(w * spec.cdf(x, row) for w, row in zip(weights, rows)))
                - quantile
            )

        low = float(rows[:, 0].min())
        high = float(rows[:, 0].max())
        step = float(rows[:, 1].max())
        for _ in range(200):
            if gap(low) <= 0:
                break
            low -= step
            step *= 2.0
        else:
            raise RuntimeError("Could not bracket the union quantile from below.")
        step = float(rows[:, 1].max())
        for _ in range(200):
            if gap(high) >= 0:
                break
            high += step
            step *= 2.0
        else:
            raise RuntimeError("Could not bracket the union quantile from above.")
        return float(brentq(gap, low, high, xtol=1e-13, rtol=1e-13, maxiter=500))


@runtime_checkable
class MixtureFamily(Protocol):
    """A mixture family the scan can fit."""

    name: str

    def fit(
        self,
        values: Any,
        k: int,
        *,
        seed: int,
        n_init: int,
        max_iter: int,
        tol: float,
        init: FamilyFit | None = None,
    ) -> FamilyFit: ...


def _log_joint(
    spec: _Spec, x: np.ndarray, weights: np.ndarray, params: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            np.log(max(float(w), 1e-300)) + spec.logpdf(x, row)
            for w, row in zip(weights, params)
        ]
    )


def _check_values(values: Any, k: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("values must be a one-dimensional array of finite numbers.")
    if k < 1 or x.size < k:
        raise ValueError(f"Cannot fit {k} components to {x.size} values.")
    return x


def _gaussian_fit(fit: fc.MixtureFit, n: int, log_likelihood: float) -> FamilyFit:
    return FamilyFit(
        family=GAUSSIAN,
        k=fit.k,
        bic=fit.bic,
        converged=fit.converged,
        weights=fit.weights,
        params=np.column_stack([fit.means, fit.sds]),
        log_likelihood=log_likelihood,
        n_iter=int(fit.model.n_iter_),
        param_names=_SPECS[GAUSSIAN].param_names,
        _spec=_SPECS[GAUSSIAN],
        _gaussian=fit,
    )


class GaussianFamily:
    """Gaussian mixture by the stage's scikit-learn code (identical results)."""

    name = GAUSSIAN

    def fit(
        self,
        values: Any,
        k: int,
        *,
        seed: int,
        n_init: int,
        max_iter: int,
        tol: float,
        init: FamilyFit | None = None,
    ) -> FamilyFit:
        x = _check_values(values, k)
        column = x.reshape(-1, 1)
        mixture = fc._fit_one(column, k, seed, n_init, max_iter, tol)
        return _gaussian_fit(
            mixture, x.size, float(mixture.model.score(column)) * x.size
        )


def _bic(log_likelihood: float, n: int, k: int, n_params: int) -> float:
    return -2.0 * log_likelihood + (k * n_params + (k - 1)) * float(np.log(n))


def _mstep(spec: _Spec, x: np.ndarray, r: np.ndarray, params: np.ndarray) -> np.ndarray:
    keep = r > RESP_FLOOR
    if int(keep.sum()) < MIN_POINTS or r[keep].sum() <= 0:
        return params
    theta0 = np.array(
        [params[0], max(np.log(params[1]), LOG_SCALE_MIN), params[2]], dtype=float
    )
    assert spec.nll is not None
    result = minimize(
        spec.nll,
        theta0,
        args=(x[keep], r[keep]),
        jac=True,
        method="L-BFGS-B",
        bounds=spec.bounds,
        options={"maxiter": MSTEP_MAXITER, "ftol": 1e-12, "gtol": 1e-9},
    )
    theta = result.x
    if not np.all(np.isfinite(theta)):
        return params
    return np.array([theta[0], np.exp(theta[1]), theta[2]], dtype=float)


def _skew_normal_start(
    x: np.ndarray, r: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Method-of-moments skew-normal ``(loc, scale, shape)`` of one weighted component.

    A symmetric start (shape 0 at the Gaussian optimum) is a saddle of the likelihood
    (the shape gradient is zero there), so the start carries the component's weighted
    sample skewness, clipped inside the skew-normal's attainable range.
    """
    total = float(r.sum())
    if total <= 0:
        return fallback
    mean = float(np.dot(r, x)) / total
    centred = x - mean
    var = float(np.dot(r, centred * centred)) / total
    if var <= 0:
        return fallback
    skew = float(np.dot(r, centred**3)) / total / var**1.5
    limit = 0.99 * 0.9952719
    skew = float(np.clip(skew, -limit, limit))
    if abs(skew) < 1e-3:
        skew = 1e-3 if skew >= 0 else -1e-3
    b = np.sqrt(2.0 / np.pi)
    t23 = (2.0 * abs(skew) / (4.0 - np.pi)) ** (2.0 / 3.0)
    delta = float(np.sign(skew)) * np.sqrt(t23 / (1.0 + t23)) / b
    delta = float(np.clip(delta, -0.99, 0.99))
    scale = np.sqrt(var / (1.0 - 2.0 * delta * delta / np.pi))
    loc = mean - scale * delta * b
    return np.array([loc, scale, delta / np.sqrt(1.0 - delta * delta)])


def _start_params(spec: _Spec, x: np.ndarray, init: FamilyFit) -> np.ndarray:
    """Starting ``(loc, scale, extra)`` per component from the Gaussian fit."""
    gaussian = init.params
    if spec.name == SKEW_NORMAL:
        resp = init.responsibilities(x)
        rows = [
            _skew_normal_start(
                x, resp[:, j], np.array([gaussian[j, 0], gaussian[j, 1], 0.5])
            )
            for j in range(init.k)
        ]
        return np.array(rows, dtype=float)
    shrink = np.sqrt((DF_INIT - 2.0) / DF_INIT)
    return np.column_stack(
        [gaussian[:, 0], gaussian[:, 1] * shrink, np.full(init.k, DF_INIT)]
    )


def _bin(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x`` itself, or at most ``BIN_COUNT`` bin means and counts for a large ``x``.

    Bin means (not centres) keep each bin's first moment exact; the bin width is
    ``range / BIN_COUNT`` (about 1e-4 of the range, far below any component's scale).
    """
    if x.size <= BIN_THRESHOLD:
        return x, np.ones(x.size)
    low, high = float(x.min()), float(x.max())
    if high <= low:
        return x, np.ones(x.size)
    index = np.minimum(
        ((x - low) / (high - low) * BIN_COUNT).astype(int), BIN_COUNT - 1
    )
    counts = np.bincount(index, minlength=BIN_COUNT).astype(float)
    sums = np.bincount(index, weights=x, minlength=BIN_COUNT)
    keep = counts > 0
    return sums[keep] / counts[keep], counts[keep]


class _EMFamily:
    """Skew-normal or Student-t mixture by generalised EM from the Gaussian fit."""

    def __init__(self, spec: _Spec) -> None:
        self._spec = spec
        self.name = spec.name

    def fit(
        self,
        values: Any,
        k: int,
        *,
        seed: int,
        n_init: int,
        max_iter: int,
        tol: float,
        init: FamilyFit | None = None,
    ) -> FamilyFit:
        x = _check_values(values, k)
        spec = self._spec
        if init is None:
            init = GaussianFamily().fit(
                x, k, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol
            )
        if init.family != GAUSSIAN or init.k != k:
            raise ValueError("init must be the Gaussian FamilyFit of the same k.")
        weights = init.weights.astype(float).copy()
        params = _start_params(spec, x, init)
        n = x.size
        # Above BIN_THRESHOLD values the EM runs on BIN_COUNT bin means with their
        # counts (the exact likelihood is then evaluated once at the end).
        points, counts = _bin(x)

        def expect() -> tuple[float, np.ndarray]:
            logp = _log_joint(spec, points, weights, params)
            total = logsumexp(logp, axis=1)
            return float(np.dot(counts, total)), np.exp(logp - total[:, None])

        log_likelihood, resp = expect()
        converged = False
        iteration = 0
        for iteration in range(1, max_iter + 1):
            mass = resp * counts[:, None]
            weights = np.maximum(mass.sum(axis=0) / n, 1e-12)
            weights = weights / weights.sum()
            for j in range(k):
                params[j] = _mstep(spec, points, mass[:, j], params[j])
            updated, resp = expect()
            # scikit-learn's rule: absolute change in the mean log-likelihood per value.
            done = abs(updated - log_likelihood) / n <= tol
            log_likelihood = updated
            if done:
                converged = True
                break
        if points.size != n:
            log_likelihood = float(
                logsumexp(_log_joint(spec, x, weights, params), axis=1).sum()
            )
        order = np.argsort([spec.mean(row) for row in params], kind="stable")
        return FamilyFit(
            family=spec.name,
            k=k,
            bic=_bic(log_likelihood, n, k, len(spec.param_names)),
            converged=converged,
            weights=weights[order],
            params=params[order].copy(),
            log_likelihood=log_likelihood,
            n_iter=iteration,
            param_names=spec.param_names,
            _spec=spec,
        )


FAMILIES: dict[str, MixtureFamily] = {
    GAUSSIAN: GaussianFamily(),
    SKEW_NORMAL: _EMFamily(_SPECS[SKEW_NORMAL]),
    STUDENT_T: _EMFamily(_SPECS[STUDENT_T]),
}


def get_family(name: str) -> MixtureFamily:
    """The family called ``name``; ``ValueError`` naming the known ones otherwise."""
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown mixture family {name!r}; known: {list(FAMILY_NAMES)}."
        ) from None


def choose_by_bic(fits: Sequence[FamilyFit]) -> FamilyFit:
    """The fit with the lowest BIC; a tie goes to the smaller ``k``."""
    if not fits:
        raise ValueError("No mixture fits to choose from.")
    best = None
    for item in sorted(fits, key=lambda fit: fit.k):
        if best is None or item.bic < best.bic:
            best = item
    assert best is not None
    return best


__all__ = [
    "DF_MAX",
    "DF_MIN",
    "FAMILIES",
    "FAMILY_NAMES",
    "FamilyFit",
    "GaussianFamily",
    "MixtureFamily",
    "choose_by_bic",
    "get_family",
    "skew_normal_logpdf",
    "student_t_logpdf",
]
