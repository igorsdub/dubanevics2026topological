"""The functional-threshold sensitivity scan.

Small seeded tables only; nothing here reads ``data/raw/`` or a checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm
from test_functional_classification import (
    CONFIG as CLASSIFICATION,
)
from test_functional_classification import (
    DECLARATION,
    PUBLISHED,
    growth_table,
)

from dubanevics2026topological.analysis import functional_threshold_scan as fts
from dubanevics2026topological.analysis import mixture_families as mf
from dubanevics2026topological.threshold import functional_classification as fc

ROOT = Path(__file__).resolve().parents[1]
SCAN = {
    "families": ["gaussian", "skew_normal", "student_t"],
    "k_range": [1, 2],
    "quantiles": [0.95, 0.99, 0.999, 0.9999],
    "confidences": [0.95, 0.99, 0.999],
    "component_rules": ["stop_anchored", "lowest_mean"],
    "n_resamples": 3,
    "random_seed": 20261025,
    "n_init": CLASSIFICATION["n_init"],
    "max_iter": CLASSIFICATION["max_iter"],
    "tol": CLASSIFICATION["tol"],
    "flip_fraction_limit": 0.05,
    "double_stop_below_min": 0.99,
}
N_CELLS = 3 * 2 * 2 * 4 * 3


@pytest.fixture(scope="module")
def growth() -> pl.DataFrame:
    # Non-functional bell around -1 (double-, single- and some triple-stop rows).
    table = growth_table([(-1.0, 0.15, 1500, 200, 100), (0.0, 0.10, 1500, 0, 0)])
    stops = table["stop_codon_count"].to_list()
    stops[:20] = [3] * 20
    return table.with_columns(stop_codon_count=pl.Series(stops, dtype=pl.Int64))


@pytest.fixture(scope="module")
def canonical(growth: pl.DataFrame) -> fc.FunctionalClassificationResult:
    result = fc.classify_growth_rates(growth, CLASSIFICATION, PUBLISHED, DECLARATION)
    assert result.summary["mixture_k"] == 2
    return result


@pytest.fixture(scope="module")
def cells(growth, canonical) -> pl.DataFrame:
    return fts.point_scan(growth, SCAN, CLASSIFICATION, canonical)


@pytest.fixture(scope="module")
def resamples3(growth) -> pl.DataFrame:
    return fts.resample_thresholds(
        growth, SCAN, range(3), None, classification_config=CLASSIFICATION
    )


# 1. Config ------------------------------------------------------------------


def test_the_checked_in_config_is_valid() -> None:
    with (ROOT / "config/analysis.toml").open("rb") as stream:
        analysis = tomllib.load(stream)
    settings = fts.validate_config(analysis[fts.SECTION])
    assert settings.families == ("gaussian", "skew_normal", "student_t")
    assert settings.k_range == (1, 5)
    assert settings.quantiles == (0.95, 0.99, 0.999, 0.9999)
    assert settings.confidences == (0.95, 0.99, 0.999)
    assert settings.component_rules == ("stop_anchored", "lowest_mean")
    assert settings.n_resamples == 200 and settings.random_seed == 20261025
    assert settings.flip_fraction_limit == 0.05
    assert settings.double_stop_below_min == 0.99
    # The scan's Gaussian cell can only reproduce the stage with the stage's settings.
    stage = analysis["functional_classification"]
    for key in ("n_init", "max_iter", "tol"):
        assert analysis[fts.SECTION][key] == stage[key]
    assert set(analysis[fts.SECTION]) == {*fts.CONFIG_KEYS, *fts.EXECUTION_KEYS}
    assert analysis[fts.SECTION]["n_jobs"] == fts.DEFAULT_N_JOBS


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("families", []),
        ("families", ["gaussian", "gaussian"]),
        ("families", ["gaussian", "laplace"]),
        ("families", "gaussian"),
        ("k_range", [0, 3]),
        ("k_range", [4, 2]),
        ("k_range", [1]),
        ("k_range", [1.0, 3]),
        ("k_range", [True, 3]),
        ("quantiles", []),
        ("quantiles", [1.0]),
        ("quantiles", [0.0, 0.5]),
        ("quantiles", [0.9, 0.9]),
        ("quantiles", [float("nan")]),
        ("confidences", [0.5]),
        ("confidences", [1.0]),
        ("component_rules", ["median"]),
        ("component_rules", []),
        ("n_resamples", 0),
        ("n_resamples", 2.0),
        ("n_resamples", True),
        ("random_seed", -1),
        ("random_seed", 1.5),
        ("n_init", 0),
        ("max_iter", 0),
        ("tol", 0.0),
        ("tol", "1e-6"),
        ("flip_fraction_limit", 0.0),
        ("flip_fraction_limit", 1.5),
        ("double_stop_below_min", 0.0),
        ("double_stop_below_min", "0.99"),
    ],
)
def test_bad_config_values_are_rejected(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=key):
        fts.validate_config({**SCAN, key: value})


def test_unknown_missing_and_non_table_config_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        fts.validate_config({**SCAN, "workers": 4})
    with pytest.raises(ValueError, match="lacks"):
        fts.validate_config({k: v for k, v in SCAN.items() if k != "n_init"})
    with pytest.raises(TypeError):
        fts.validate_config([1, 5])
    assert fts.validate_config(SCAN).n_resamples == 3


# 2. Point scan --------------------------------------------------------------


def test_the_point_scan_has_one_row_per_cell_and_the_declared_schema(cells) -> None:
    assert cells.height == N_CELLS
    assert cells.schema == pl.Schema(fts.CELL_SCHEMA)
    key = ["family", "k", "component_rule", "quantile", "confidence"]
    assert cells.select(key).unique().height == N_CELLS
    assert (
        cells["functional_count"]
        + cells["non_functional_count"]
        + cells["uncertain_count"]
    ).unique().to_list() == [3000]
    assert cells["selected_components"].list.len().min() >= 1
    lowest = cells.filter(pl.col("component_rule") == "lowest_mean")
    assert lowest["selected_components"].list.len().unique().to_list() == [1]


def test_the_canonical_cell_has_no_flips_and_jaccard_one(cells, canonical) -> None:
    marked = cells.filter(pl.col("is_canonical"))
    assert marked.height == 1
    row = marked.row(0, named=True)
    assert (row["family"], row["k"], row["component_rule"]) == (
        "gaussian",
        2,
        "stop_anchored",
    )
    assert (row["quantile"], row["confidence"]) == (0.999, 0.95)
    assert row["functional_flips"] == 0 and row["functional_jaccard"] == 1.0
    assert row["threshold"] == canonical.summary["threshold"]["rounded"]
    assert row["functional_count"] == canonical.summary["class_counts"]["functional"]
    assert row["bic_chosen"] == 2 and row["converged"]


def test_bic_chosen_is_per_family_and_constant_within_it(cells) -> None:
    per_family = cells.group_by("family").agg(pl.col("bic_chosen").n_unique())
    assert per_family["bic_chosen"].to_list() == [1, 1, 1]
    for name in SCAN["families"]:
        rows = cells.filter(pl.col("family") == name).select("k", "bic").unique()
        best = rows.sort("bic", "k")["k"][0]
        chosen = cells.filter(pl.col("family") == name)["bic_chosen"][0]
        assert chosen == best


def test_thresholds_rise_with_the_quantile_and_flips_with_the_confidence(cells) -> None:
    ordered = cells.filter(
        (pl.col("family") == "gaussian")
        & (pl.col("k") == 2)
        & (pl.col("component_rule") == "stop_anchored")
        & (pl.col("confidence") == 0.95)
    ).sort("quantile")
    assert ordered["threshold"].is_sorted()
    assert ordered["threshold"].n_unique() == 4
    # A higher threshold leaves fewer sequences above it; a stricter confidence fewer
    # functional ones.
    same_t = cells.filter(
        (pl.col("family") == "gaussian")
        & (pl.col("k") == 2)
        & (pl.col("component_rule") == "stop_anchored")
        & (pl.col("quantile") == 0.999)
    ).sort("confidence")
    assert same_t["functional_count"].to_list() == sorted(
        same_t["functional_count"].to_list(), reverse=True
    )


def test_a_non_canonical_cell_matches_an_independent_computation(
    growth, canonical, cells
) -> None:
    s = growth["s"].to_numpy()
    se = growth["se_pooled"].to_numpy()
    fit = mf.get_family("skew_normal").fit(
        s,
        2,
        seed=CLASSIFICATION["random_seed"],
        n_init=SCAN["n_init"],
        max_iter=SCAN["max_iter"],
        tol=SCAN["tol"],
    )
    mask = np.zeros(2, dtype=bool)
    mask[int(np.argmin(fit.component_means))] = True
    threshold = round(fit.union_quantile(mask, 0.99), 6)
    p = norm.cdf((s - threshold) / se)
    functional = p >= 0.99
    expected = canonical.classification["functional_class"].to_numpy() == "functional"
    row = cells.filter(
        (pl.col("family") == "skew_normal")
        & (pl.col("k") == 2)
        & (pl.col("component_rule") == "lowest_mean")
        & (pl.col("quantile") == 0.99)
        & (pl.col("confidence") == 0.99)
    ).row(0, named=True)
    assert row["threshold"] == threshold
    assert row["functional_count"] == int(functional.sum())
    assert row["functional_flips"] == int((functional ^ expected).sum())
    both = int((functional & expected).sum())
    assert row["functional_jaccard"] == pytest.approx(
        both / int((functional | expected).sum())
    )
    stops = growth["stop_codon_count"].to_numpy()
    assert row["frac_2stop_below"] == pytest.approx(
        float((s[stops == 2] < threshold).mean())
    )


def test_the_point_scan_refuses_a_canonical_it_cannot_reproduce(
    growth, canonical
) -> None:
    wrong_threshold = dataclasses.replace(
        canonical,
        summary={**canonical.summary, "threshold": {"unrounded": 0.0, "rounded": 0.0}},
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        fts.point_scan(growth, SCAN, CLASSIFICATION, wrong_threshold)
    absent_k = dataclasses.replace(
        canonical, summary={**canonical.summary, "mixture_k": 5}
    )
    with pytest.raises(ValueError, match="no canonical cell"):
        fts.point_scan(growth, SCAN, CLASSIFICATION, absent_k)


@pytest.mark.parametrize(
    ("key", "value"), [("n_init", 9), ("max_iter", 7), ("tol", 1e-3)]
)
def test_scan_fit_settings_must_equal_the_classifications_before_any_fit(
    key: str, value: Any, growth, canonical, monkeypatch, tmp_path: Path
) -> None:
    def no_fit(*args, **kwargs):
        raise AssertionError("a fit started")

    monkeypatch.setattr(fts, "_fit_all", no_fit)
    scan = {**SCAN, key: value}
    with pytest.raises(fts.ScanSettingsMismatch, match=key):
        fts.point_scan(growth, scan, CLASSIFICATION, canonical)
    with pytest.raises(fts.ScanSettingsMismatch, match=key):
        fts.resample_thresholds(
            growth, scan, range(1), classification_config=CLASSIFICATION
        )
    with pytest.raises(fts.ScanSettingsMismatch, match=key):
        fts.resample_thresholds_parallel(
            growth, scan, range(2), classification_config=CLASSIFICATION, n_jobs=2
        )
    settings = {"analysis": {fts.SECTION: scan, fc.SECTION: dict(CLASSIFICATION)}}
    with pytest.raises(fts.ScanSettingsMismatch, match=key):
        fts.compute_resample_slice(range(1), settings, tmp_path / "slice")
    with pytest.raises(fts.ScanSettingsMismatch, match=key):
        fts.compute_functional_threshold_scan(settings, scan_dir=tmp_path / "scan")
    assert not (tmp_path / "slice").exists() and not (tmp_path / "scan").exists()


def test_the_point_scan_rejects_bad_inputs(growth, canonical) -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        fts.point_scan(growth.drop("s"), SCAN, CLASSIFICATION, canonical)
    bad = growth.with_columns(
        se_pooled=pl.when(pl.int_range(pl.len()) == 0)
        .then(0.0)
        .otherwise(pl.col("se_pooled"))
    )
    with pytest.raises(fc.NonPositiveStandardError):
        fts.point_scan(bad, SCAN, CLASSIFICATION, canonical)
    with pytest.raises(ValueError, match="sequence counts"):
        fts.point_scan(growth.head(10), SCAN, CLASSIFICATION, canonical)


# 3. Comparator --------------------------------------------------------------


def test_comparator_cells_use_empirical_stop_quantiles(growth, canonical) -> None:
    table = fts.comparator_cells(
        growth, canonical=canonical, classification_config=CLASSIFICATION
    )
    assert table.schema == pl.Schema(fts.COMPARATOR_SCHEMA)
    assert table.select("stop_count", "quantile").rows() == [
        (2, 0.99),
        (2, 0.999),
        (3, 0.99),
        (3, 0.999),
    ]
    s = growth["s"].to_numpy()
    stops = growth["stop_codon_count"].to_numpy()
    for row in table.iter_rows(named=True):
        members = s[stops == row["stop_count"]]
        assert row["n_sequences"] == members.size
        assert row["threshold"] == round(
            float(np.quantile(members, row["quantile"])), 6
        )
        assert row["confidence"] == 0.95
        assert (
            row["functional_count"]
            + row["non_functional_count"]
            + row["uncertain_count"]
            == 3000
        )
        functional = (
            norm.cdf((s - row["threshold"]) / growth["se_pooled"].to_numpy()) >= 0.95
        )
        expected = (
            canonical.classification["functional_class"].to_numpy() == "functional"
        )
        assert row["functional_flips"] == int((functional ^ expected).sum())
    assert table.filter(pl.col("stop_count") == 3)["n_sequences"][0] == 20


def test_comparator_needs_sequences_with_the_stop_count(growth, canonical) -> None:
    without_triples = growth.with_columns(
        stop_codon_count=pl.col("stop_codon_count").replace(3, 2)
    )
    with pytest.raises(ValueError, match="3 stop codons"):
        fts.comparator_cells(
            without_triples, canonical=canonical, classification_config=CLASSIFICATION
        )


# 4. Resamples ---------------------------------------------------------------


def resample(growth, indices: range, chosen=None) -> pl.DataFrame:
    return fts.resample_thresholds(
        growth, SCAN, indices, chosen, classification_config=CLASSIFICATION
    )


def test_a_resample_is_the_same_whichever_range_computes_it(growth, resamples3) -> None:
    second = resample(growth, range(1, 2))
    assert resamples3.schema == pl.Schema(fts.RESAMPLE_SCHEMA)
    assert resamples3.height == 3 * 6
    assert resamples3.filter(pl.col("resample") == 1).equals(second)


def test_different_resample_indices_draw_different_samples(resamples3) -> None:
    table = resamples3.filter((pl.col("family") == "gaussian") & (pl.col("k") == 1))
    thresholds = table.sort("resample")["threshold"].to_list()
    assert len(set(thresholds)) == 3


def test_a_gaussian_k1_resample_matches_an_independent_draw(growth, resamples3) -> None:
    s = growth["s"].to_numpy()
    r = 2
    draw = np.random.default_rng(SCAN["random_seed"] + r).integers(0, s.size, s.size)
    sample = s[draw]
    expected = round(float(sample.mean() + sample.std() * norm.ppf(0.999)), 6)
    row = resamples3.filter(
        (pl.col("resample") == r)
        & (pl.col("family") == "gaussian")
        & (pl.col("k") == 1)
    ).row(0, named=True)
    assert row["threshold"] == pytest.approx(expected, abs=1e-4)
    assert row["converged"] is True


def test_is_bic_chosen_is_null_unless_the_choice_is_given(growth, resamples3) -> None:
    assert resamples3["is_bic_chosen"].null_count() == resamples3.height
    table = resample(
        growth, range(1), {"gaussian": 2, "skew_normal": 1, "student_t": 2}
    )
    chosen = table.filter(pl.col("is_bic_chosen")).select("family", "k").sort("family")
    assert chosen.rows() == [("gaussian", 2), ("skew_normal", 1), ("student_t", 2)]
    with pytest.raises(ValueError, match="unknown families"):
        resample(growth, range(1), {"laplace": 2})


# 5. Intervals ---------------------------------------------------------------


def test_resample_bic_selection_is_per_draw_and_auditable(resamples3) -> None:
    for _, group in resamples3.group_by(["resample", "family"]):
        expected = group.sort("bic", "k").row(0, named=True)
        chosen = group.filter(pl.col("is_resample_bic_chosen"))
        assert chosen.height == 1
        assert chosen["k"][0] == expected["k"]
        assert group["model_selection_complete"].unique().to_list() == [
            all(group["converged"]) and all(np.isfinite(group["bic"].to_numpy()))
        ]
        assert all(len(components) >= 1 for components in group["selected_components"])


def test_the_interval_table_takes_quantiles_of_the_converged_fits() -> None:
    values = np.arange(101.0)
    frame = pl.DataFrame(
        {
            "resample": list(range(101)) + [101, 0, 1],
            "family": ["gaussian"] * 102 + ["student_t"] * 2,
            "k": [2] * 102 + [1, 1],
            "threshold": list(values) + [1e9, 5.0, 7.0],
            "converged": [True] * 101 + [False, False, False],
            "is_bic_chosen": [None] * 104,
            "bic": [0.0] * 104,
            "selected_components": [[0]] * 104,
            "is_resample_bic_chosen": [False] * 104,
            "model_selection_complete": [False] * 104,
        },
        schema=fts.RESAMPLE_SCHEMA,
    )
    table = fts.interval_table(frame)
    assert table.schema == pl.Schema(fts.INTERVAL_SCHEMA)
    gaussian = table.filter(pl.col("family") == "gaussian").row(0, named=True)
    assert gaussian["threshold_low"] == pytest.approx(2.5)
    assert gaussian["threshold_high"] == pytest.approx(97.5)
    assert gaussian["width"] == pytest.approx(95.0)
    assert (gaussian["n_converged"], gaussian["n_resamples"]) == (101, 102)
    never = table.filter(pl.col("family") == "student_t").row(0, named=True)
    assert never["threshold_low"] is None and never["width"] is None
    assert (never["n_converged"], never["n_resamples"]) == (0, 2)


def test_intervals_from_real_resamples_are_ordered(resamples3) -> None:
    table = fts.interval_table(resamples3)
    assert table.height == 6
    assert (table["threshold_low"] <= table["threshold_high"]).all()
    assert (table["n_resamples"] == 3).all()


# 6. Acceptance checks -------------------------------------------------------

COUNT = 1000


def fake_point(
    *,
    frac_2stop: float = 0.995,
    flips: dict[tuple, int] | None = None,
    k: int = 2,
    unconverged: set[tuple] | None = None,
) -> pl.DataFrame:
    """A point table with the columns the checks read, canonical = gaussian k=2."""
    flips = flips or {}
    chosen = {"gaussian": k, "skew_normal": 2, "student_t": 3}
    rows = []
    for family in SCAN["families"]:
        for kk in (1, 2, 3):
            for rule in SCAN["component_rules"]:
                for q in SCAN["quantiles"]:
                    for c in SCAN["confidences"]:
                        key = (family, kk, rule, q, c)
                        canon = key == ("gaussian", k, "stop_anchored", 0.999, 0.95)
                        rows.append(
                            {
                                "family": family,
                                "k": kk,
                                "bic_chosen": chosen[family],
                                "component_rule": rule,
                                "quantile": q,
                                "confidence": c,
                                "threshold": -0.5,
                                "converged": key not in (unconverged or set()),
                                "frac_2stop_below": frac_2stop if canon else 0.5,
                                "functional_count": COUNT,
                                "functional_flips": flips.get(key, 0),
                                "is_canonical": canon,
                            }
                        )
    return pl.DataFrame(rows)


def fake_intervals(width: float | None = 0.05) -> pl.DataFrame:
    low = None if width is None else -0.5
    high = None if width is None else -0.5 + width
    return pl.DataFrame(
        [
            {
                "family": "gaussian",
                "k": 2,
                "threshold_low": low,
                "threshold_high": high,
                "width": width,
                "n_converged": 3,
                "n_resamples": 3,
            }
        ],
        schema=fts.INTERVAL_SCHEMA,
    )


def checks(point=None, intervals=None, growth=None, config=SCAN) -> dict[str, Any]:
    if growth is None:
        growth = pl.DataFrame({"se_pooled": [0.03, 0.04, 0.05]})  # median 0.04
    return fts.acceptance_checks(
        fake_point() if point is None else point,
        fake_intervals() if intervals is None else intervals,
        growth,
        config,
    )


def test_every_check_passes_on_a_clean_scan() -> None:
    result = checks()
    assert result["passed"] == {"a": True, "b": True, "c": True, "all": True}
    json.dumps(result, allow_nan=False)
    assert set(result["c"]["axes"]) == {
        "family",
        "k",
        "quantile",
        "confidence",
        "component_rule",
    }


def test_check_a_fails_below_the_double_stop_minimum() -> None:
    assert checks(fake_point(frac_2stop=0.99))["passed"]["a"] is True
    result = checks(fake_point(frac_2stop=0.9899))
    assert result["a"] == {
        "frac_2stop_below": 0.9899,
        "minimum": 0.99,
        "passed": False,
    }
    assert result["passed"]["all"] is False


def test_check_b_compares_the_interval_width_with_the_median_se() -> None:
    limit = 2 * 1.96 * 0.04
    ok = checks(intervals=fake_intervals(limit - 1e-6))
    assert ok["passed"]["b"] is True
    assert ok["b"]["median_se_pooled"] == pytest.approx(0.04)
    assert ok["b"]["limit"] == pytest.approx(limit)
    assert ok["b"]["width"] == pytest.approx(limit - 1e-6)
    wide = checks(intervals=fake_intervals(limit + 1e-6))
    assert wide["passed"]["b"] is False and wide["passed"]["all"] is False
    assert checks(intervals=fake_intervals(None))["passed"]["b"] is False
    assert checks(intervals=fake_intervals().clear())["b"]["width"] is None


CANON = ("gaussian", 2, "stop_anchored", 0.999, 0.95)


@pytest.mark.parametrize(
    ("axis", "key"),
    [
        ("family", ("skew_normal", 2, "stop_anchored", 0.999, 0.95)),
        ("family", ("student_t", 3, "stop_anchored", 0.999, 0.95)),
        ("k", ("gaussian", 1, "stop_anchored", 0.999, 0.95)),
        ("k", ("gaussian", 3, "stop_anchored", 0.999, 0.95)),
        ("quantile", ("gaussian", 2, "stop_anchored", 0.99, 0.95)),
        ("quantile", ("gaussian", 2, "stop_anchored", 0.9999, 0.95)),
        ("confidence", ("gaussian", 2, "stop_anchored", 0.999, 0.99)),
        ("component_rule", ("gaussian", 2, "lowest_mean", 0.999, 0.95)),
    ],
)
def test_check_c_flags_a_neighbour_at_the_limit_only(axis: str, key: tuple) -> None:
    assert checks(fake_point(flips={key: 49}))["passed"]["c"] is True
    result = checks(fake_point(flips={key: 50}))
    assert result["passed"]["c"] is False and result["passed"]["all"] is False
    assert result["c"]["axes"][axis]["passed"] is False
    assert result["c"]["axes"][axis]["max_flip_fraction"] == pytest.approx(0.05)
    others = [name for name in result["c"]["axes"] if name != axis]
    assert all(result["c"]["axes"][name]["passed"] for name in others)
    assert result["c"]["canonical_functional_count"] == COUNT


def test_check_b_records_how_many_resamples_converged() -> None:
    result = checks()
    assert (result["b"]["n_converged"], result["b"]["n_resamples"]) == (3, 3)
    missing = checks(intervals=fake_intervals().clear())
    assert (missing["b"]["n_converged"], missing["b"]["n_resamples"]) == (None, None)


def test_check_c_neighbours_carry_their_convergence() -> None:
    result = checks()
    neighbours = [
        n for axis in result["c"]["axes"].values() for n in axis["neighbours"]
    ]
    assert neighbours and all(n["converged"] is True for n in neighbours)
    assert all(axis["reason"] is None for axis in result["c"]["axes"].values())


def test_an_unconverged_neighbour_leaves_its_axis_without_a_verdict() -> None:
    key = ("gaussian", 2, "stop_anchored", 0.99, 0.95)
    result = checks(fake_point(unconverged={key}))
    axis = result["c"]["axes"]["quantile"]
    assert axis["passed"] is None and axis["reason"] == "neighbour_not_converged"
    assert [n["converged"] for n in axis["neighbours"]] == [False, True]
    assert (
        result["c"]["passed"] is None
        and "quantile: neighbour_not_converged" in (result["c"]["reason"])
    )
    assert result["passed"] == {"a": True, "b": True, "c": None, "all": None}
    json.dumps(result, allow_nan=False)
    # A failed axis still makes check c fail outright.
    worse = checks(fake_point(unconverged={key}, flips={CANON[:4] + (0.99,): 60}))
    assert worse["passed"]["c"] is False and worse["passed"]["all"] is False


def test_an_axis_without_neighbours_is_undefined_not_passed() -> None:
    result = checks(config={**SCAN, "confidences": [0.95]})
    axis = result["c"]["axes"]["confidence"]
    assert axis["neighbours"] == [] and axis["max_flip_fraction"] is None
    assert axis["passed"] is None and axis["reason"] == "no_neighbours"
    assert result["passed"]["c"] is None and result["passed"]["all"] is None


def test_check_c_ignores_cells_more_than_one_step_away() -> None:
    far = {
        ("gaussian", 2, "stop_anchored", 0.95, 0.95): 900,  # two quantile steps
        ("gaussian", 2, "stop_anchored", 0.999, 0.999): 900,  # two confidence steps
        ("skew_normal", 1, "stop_anchored", 0.999, 0.95): 900,  # not its BIC k
        ("gaussian", 2, "lowest_mean", 0.99, 0.95): 900,  # two axes at once
    }
    assert checks(fake_point(flips=far))["passed"]["c"] is True


def test_check_c_k_axis_follows_the_bic_choice_and_the_edge_of_the_range() -> None:
    edge = fake_point(k=1, flips={("gaussian", 2, "stop_anchored", 0.999, 0.95): 0})
    result = checks(edge)
    assert [n["k"] for n in result["c"]["axes"]["k"]["neighbours"]] == [2]


def test_acceptance_needs_one_canonical_cell_and_a_nonempty_set() -> None:
    with pytest.raises(ValueError, match="exactly one canonical"):
        checks(fake_point().with_columns(is_canonical=pl.lit(False)))
    empty = fake_point().with_columns(functional_count=pl.lit(0))
    with pytest.raises(ValueError, match="empty"):
        checks(empty)


def test_acceptance_on_a_real_point_scan_is_json_ready(
    growth, cells, resamples3
) -> None:
    intervals = fts.interval_table(resamples3)
    result = fts.acceptance_checks(cells, intervals, growth, SCAN)
    json.dumps(result, allow_nan=False)
    assert result["a"]["passed"] is True
    assert result["c"]["axes"]["k"]["neighbours"]
    assert result["canonical"]["family"] == "gaussian"


def test_acceptance_record_does_not_depend_on_row_order(
    growth, cells, resamples3
) -> None:
    intervals = fts.interval_table(resamples3)
    first = fts.acceptance_checks(cells, intervals, growth, SCAN)
    families = [n["family"] for n in first["c"]["axes"]["family"]["neighbours"]]
    assert families == sorted(families)
    # The point scan's rows in another order give the same record (a sliced and an
    # unsliced run must write the same acceptance.json).
    shuffled = cells.sample(fraction=1.0, shuffle=True, seed=7)
    assert fts.acceptance_checks(shuffled, intervals, growth, SCAN) == first


# 7. Slices and merge --------------------------------------------------------


CURRENT = {
    "scan_config": {"n_resamples": 3},
    "classification_config": {},
    "classification_source": {"fingerprint": "x"},
    "growth_rates_sha256": "abc",
    "code": {"mixture_families": "m1", "functional_threshold_scan": "s1"},
}
CELLS = [(f, k) for f in SCAN["families"] for k in (1, 2)]


def write_slice(
    frame: pl.DataFrame, directory: Path, indices: range, record=CURRENT
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    part = frame.filter(pl.col("resample").is_in(list(indices)))
    fts._write_parquet_atomic(part, directory / fts.RESAMPLES_NAME)
    if record is not None:
        (directory / fts.SLICE_NAME).write_text(json.dumps(record), encoding="utf-8")
    return directory


def merge(dirs, n: int, current=CURRENT, cells=CELLS) -> pl.DataFrame:
    return fts.merge_resample_slices(dirs, n, current=current, cells=cells)


def test_slices_merge_into_the_full_resample_table(resamples3, tmp_path: Path) -> None:
    first = write_slice(resamples3, tmp_path / "a", range(2))
    second = write_slice(resamples3, tmp_path / "b", range(2, 3))
    merged = merge([second, first], 3)
    assert merged["resample"].unique().sort().to_list() == [0, 1, 2]
    assert merged.equals(resamples3.sort("resample", "family", "k"))


def test_a_missing_resample_index_fails_naming_it(resamples3, tmp_path: Path) -> None:
    first = write_slice(resamples3, tmp_path / "a", range(1))
    third = write_slice(resamples3, tmp_path / "c", range(2, 3))
    with pytest.raises(ValueError, match=r"missing resample indices \[1\]"):
        merge([first, third], 3)
    with pytest.raises(ValueError, match=r"missing resample indices \[3, 4\]"):
        merge([write_slice(resamples3, tmp_path / "d", range(3))], 5)


def test_repeated_and_unexpected_indices_fail(resamples3, tmp_path: Path) -> None:
    first = write_slice(resamples3, tmp_path / "a", range(2))
    overlap = write_slice(resamples3, tmp_path / "b", range(1, 3))
    with pytest.raises(ValueError, match=r"repeated resample indices \[1\]"):
        merge([first, overlap], 3)
    with pytest.raises(ValueError, match=r"unexpected resample indices \[2\]"):
        merge([first, overlap], 2)


def test_an_incomplete_resample_is_not_accepted(resamples3, tmp_path: Path) -> None:
    first = write_slice(resamples3, tmp_path / "a", range(2))
    table = pl.read_parquet(first / fts.RESAMPLES_NAME)
    thin = table.filter(
        ~((pl.col("resample") == 1) & (pl.col("family") == "student_t"))
    )
    fts._write_parquet_atomic(thin, first / fts.RESAMPLES_NAME)
    with pytest.raises(ValueError, match=r"without all 6 \(family, k\) fits \[1\]"):
        merge([first], 2)


@pytest.mark.parametrize(
    "key",
    [
        "scan_config",
        "classification_config",
        "classification_source",
        "growth_rates_sha256",
    ],
)
def test_a_slice_from_other_settings_does_not_merge(
    key: str, resamples3, tmp_path: Path
) -> None:
    first = write_slice(resamples3, tmp_path / "a", range(2))
    other = write_slice(
        resamples3, tmp_path / "b", range(2, 3), {**CURRENT, key: {"other": 1}}
    )
    with pytest.raises(ValueError, match=rf"Slice .*b was computed with other {key}"):
        merge([first, other], 3)
    # All slices agreeing with each other is not enough: they must match the current.
    stale = {**CURRENT, key: {"other": 1}}
    with pytest.raises(ValueError, match=rf"Slice .*a was computed with other {key}"):
        merge([first], 3, current=stale)


def test_a_slice_without_its_record_does_not_merge(resamples3, tmp_path: Path) -> None:
    first = write_slice(resamples3, tmp_path / "a", range(2))
    bare = write_slice(resamples3, tmp_path / "b", range(2, 3), None)
    with pytest.raises(FileNotFoundError, match=r"No slice\.json in .*b"):
        merge([first, bare], 3)


def test_the_slices_family_k_set_must_equal_the_scans(
    resamples3, tmp_path: Path
) -> None:
    thin = resamples3.filter(pl.col("family") != "student_t")
    first = write_slice(thin, tmp_path / "a", range(3))
    with pytest.raises(ValueError, match=r"missing \[\('student_t', 1\)"):
        merge([first], 3)
    full = write_slice(resamples3, tmp_path / "b", range(3))
    with pytest.raises(ValueError, match=r"unexpected \[\('student_t', 2\)"):
        merge([full], 3, cells=CELLS[:-1])


def test_merge_needs_slices_and_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No slices"):
        merge([], 1)
    with pytest.raises(FileNotFoundError):
        merge([tmp_path], 1)


# 8. Writing, computing and loading ------------------------------------------


@pytest.fixture
def settings() -> dict[str, Any]:
    return {
        "analysis": {fts.SECTION: dict(SCAN), fc.SECTION: dict(CLASSIFICATION)},
    }


@pytest.fixture
def offline(monkeypatch, growth, canonical):
    """Serve the growth table and the classification without any checkpoint."""
    source = {
        "stage": "functional_classification",
        "fingerprint": "f1",
        "artefacts": {},
    }
    state = {"source": source}

    def load_inputs(settings, *, lock):
        return growth, state["source"], "growth-sha"

    def load_classification(settings, *, tables=True, lock=True):
        return canonical, state["source"]

    monkeypatch.setattr(fts, "_load_inputs", load_inputs)
    monkeypatch.setattr(
        fc, "load_cached_functional_classification_with_source", load_classification
    )
    return state


def test_a_slice_task_writes_its_range_and_the_merge_equals_one_run(
    offline, settings, resamples3, tmp_path: Path
) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    first = fts.compute_resample_slice(range(2), settings, a)
    fts.compute_resample_slice(range(2, 3), settings, b)
    assert first.equals(pl.read_parquet(a / fts.RESAMPLES_NAME))
    record = json.loads((a / fts.SLICE_NAME).read_text(encoding="utf-8"))
    assert record["indices"] == {"start": 0, "stop": 2, "step": 1}
    assert record["growth_rates_sha256"] == "growth-sha"
    merged = fts.merge_resample_slices(
        [a, b],
        3,
        current=fts.current_slice_identity(settings),
        cells=fts.scan_cells(settings),
    )
    assert merged.equals(resamples3.sort("resample", "family", "k"))


def test_the_scan_is_written_with_its_provenance_last_and_loads_back(
    offline, settings, resamples3, tmp_path: Path
) -> None:
    scan_dir = tmp_path / "scan"
    result = fts.compute_functional_threshold_scan(
        settings, resamples=resamples3, scan_dir=scan_dir
    )
    assert sorted(p.name for p in scan_dir.iterdir()) == sorted(
        [*fts.ARTIFACT_NAMES, fts.PROVENANCE_NAME]
    )
    record_time = (scan_dir / fts.PROVENANCE_NAME).stat().st_mtime_ns
    assert all(
        (scan_dir / name).stat().st_mtime_ns <= record_time
        for name in fts.ARTIFACT_NAMES
    )
    provenance = result.provenance
    assert provenance["growth_rates_sha256"] == "growth-sha"
    assert provenance["classification_source"]["fingerprint"] == "f1"
    assert provenance["scan_config"]["n_resamples"] == 3
    assert set(provenance["libraries"]) == {"numpy", "polars", "scikit-learn", "scipy"}
    assert set(provenance["git"]) == {"commit", "dirty"}
    assert set(provenance["artifacts"]) == set(fts.ARTIFACT_NAMES)
    assert result.cells.height == N_CELLS and result.intervals.height == 6
    assert result.acceptance["passed"]["a"] is True

    loaded = fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)
    assert loaded.cells.equals(result.cells)
    assert loaded.comparator.equals(result.comparator)
    assert loaded.resamples.equals(result.resamples)
    assert loaded.intervals.equals(result.intervals)
    assert loaded.acceptance == result.acceptance
    assert loaded.provenance == provenance


def test_the_loader_refuses_a_stale_or_unfinished_scan(
    offline, settings, resamples3, tmp_path: Path
) -> None:
    scan_dir = tmp_path / "scan"
    with pytest.raises(FileNotFoundError, match="No finished"):
        fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)
    fts.compute_functional_threshold_scan(
        settings, resamples=resamples3, scan_dir=scan_dir
    )
    fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)

    changed = {
        "analysis": {
            fts.SECTION: {**SCAN, "flip_fraction_limit": 0.5},
            fc.SECTION: dict(CLASSIFICATION),
        }
    }
    with pytest.raises(RuntimeError, match="stale.*scan_config"):
        fts.load_functional_threshold_scan(changed, scan_dir=scan_dir)
    retuned = {
        "analysis": {
            fts.SECTION: dict(SCAN),
            fc.SECTION: {**CLASSIFICATION, "quantile": 0.99},
        }
    }
    with pytest.raises(RuntimeError, match="stale.*classification_config"):
        fts.load_functional_threshold_scan(retuned, scan_dir=scan_dir)

    offline["source"] = {**offline["source"], "fingerprint": "f2"}
    with pytest.raises(RuntimeError, match="stale.*classification_source"):
        fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)
    offline["source"] = {**offline["source"], "fingerprint": "f1"}

    (scan_dir / fts.CELLS_NAME).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match=r"stale.*cells\.parquet"):
        fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)
    (scan_dir / fts.INTERVALS_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="intervals"):
        fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)


def test_a_code_change_makes_the_scan_and_its_slices_stale(
    offline, settings, resamples3, tmp_path: Path, monkeypatch
) -> None:
    scan_dir = tmp_path / "scan"
    result = fts.compute_functional_threshold_scan(
        settings, resamples=resamples3, scan_dir=scan_dir
    )
    code = result.provenance["code"]
    assert set(code) == {"mixture_families", "functional_threshold_scan"}
    assert code["mixture_families"] == fts._sha256(Path(mf.__file__))
    assert code["functional_threshold_scan"] == fts._sha256(Path(fts.__file__))
    fts.compute_resample_slice(range(1), settings, tmp_path / "a")
    record = json.loads((tmp_path / "a" / fts.SLICE_NAME).read_text())
    assert record["code"] == code

    monkeypatch.setattr(
        fts, "_code_hashes", lambda: {**code, "mixture_families": "edited"}
    )
    with pytest.raises(RuntimeError, match="stale.*code"):
        fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)
    with pytest.raises(ValueError, match=r"Slice .*a was computed with other code"):
        fts.merge_resample_slices(
            [tmp_path / "a"],
            1,
            current=fts.current_slice_identity(settings),
            cells=fts.scan_cells(settings),
        )


def test_a_rerun_removes_the_old_provenance_before_writing(
    offline, settings, resamples3, tmp_path: Path, monkeypatch
) -> None:
    scan_dir = tmp_path / "scan"
    fts.compute_functional_threshold_scan(
        settings, resamples=resamples3, scan_dir=scan_dir
    )
    seen = []
    real = fts._write_parquet_atomic

    def spy(frame, path):
        seen.append((scan_dir / fts.PROVENANCE_NAME).exists())
        return real(frame, path)

    monkeypatch.setattr(fts, "_write_parquet_atomic", spy)
    fts.compute_functional_threshold_scan(
        settings, resamples=resamples3, scan_dir=scan_dir
    )
    assert seen and not any(seen)


def test_the_scan_rejects_resamples_that_do_not_cover_the_range(
    offline, settings, resamples3, tmp_path: Path
) -> None:
    partial = resamples3.filter(pl.col("resample") < 2)
    with pytest.raises(ValueError, match=r"missing resample indices \[2\]"):
        fts.compute_functional_threshold_scan(
            settings, resamples=partial, scan_dir=tmp_path / "scan"
        )
    assert not (tmp_path / "scan").exists()


def test_the_scan_accepts_merged_resamples(
    offline, settings, resamples3, tmp_path: Path
) -> None:
    result = fts.compute_functional_threshold_scan(
        settings, resamples=resamples3, scan_dir=tmp_path / "scan"
    )
    # Only is_bic_chosen differs: the scan fills it from the point scan.
    assert result.resamples.drop("is_bic_chosen").equals(
        resamples3.sort("resample", "family", "k").drop("is_bic_chosen")
    )
    chosen = result.resamples.filter(pl.col("is_bic_chosen"))
    assert chosen.group_by("family").len()["len"].to_list() == [3, 3, 3]


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "8"])
def test_n_jobs_is_validated_but_optional(bad: Any) -> None:
    assert fts.validate_config({**SCAN, "n_jobs": 3}).n_resamples == 3
    with pytest.raises(ValueError, match="n_jobs"):
        fts.validate_config({**SCAN, "n_jobs": bad})


def test_n_jobs_is_in_no_recorded_identity(
    offline, settings, resamples3, tmp_path: Path
) -> None:
    settings["analysis"][fts.SECTION]["n_jobs"] = 2
    scan_dir = tmp_path / "scan"
    result = fts.compute_functional_threshold_scan(
        settings, resamples=resamples3, scan_dir=scan_dir
    )
    assert "n_jobs" not in result.provenance["scan_config"]
    settings["analysis"][fts.SECTION]["n_jobs"] = 5
    fts.load_functional_threshold_scan(settings, scan_dir=scan_dir)  # not stale


def test_resolve_n_jobs_prefers_the_environment_then_the_table_then_eight() -> None:
    assert fts.resolve_n_jobs(SCAN, {}) == 8
    assert fts.resolve_n_jobs({**SCAN, "n_jobs": 3}, {}) == 3
    assert fts.resolve_n_jobs({**SCAN, "n_jobs": 3}, {"FITNESSTREE_N_JOBS": "6"}) == 6
    assert fts.resolve_n_jobs({**SCAN, "n_jobs": 3}, {"FITNESSTREE_N_JOBS": " "}) == 3
    for bad in ("0", "-1", "x", "1.5"):
        with pytest.raises(ValueError, match="FITNESSTREE_N_JOBS"):
            fts.resolve_n_jobs(SCAN, {"FITNESSTREE_N_JOBS": bad})


def test_parallel_resamples_equal_the_serial_ones(growth, resamples3) -> None:
    parallel = fts.resample_thresholds_parallel(
        growth, SCAN, range(3), None, classification_config=CLASSIFICATION, n_jobs=2
    )
    assert parallel.equals(resamples3)
    with pytest.raises(ValueError, match="n_jobs"):
        fts.resample_thresholds_parallel(
            growth, SCAN, range(3), None, classification_config=CLASSIFICATION, n_jobs=0
        )


def test_the_scan_computes_its_own_resamples_in_parallel(
    offline, settings, resamples3, tmp_path: Path
) -> None:
    result = fts.compute_functional_threshold_scan(
        settings, scan_dir=tmp_path / "scan", n_jobs=2
    )
    assert result.resamples.drop("is_bic_chosen").equals(
        resamples3.sort("resample", "family", "k").drop("is_bic_chosen")
    )
