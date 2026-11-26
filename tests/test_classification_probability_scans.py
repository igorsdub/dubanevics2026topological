"""Focused unit tests for measurement-uncertainty functional classification and scans."""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from dubanevics2026topological.threshold import (
    classification_probability_scans as scans,
)
from dubanevics2026topological.threshold.classification_probability_scans import (
    MARKED_Q_VALUES,
    VALIDATION_ANCHOR_COUNTS,
    VALIDATION_ANCHOR_Q,
    VALIDATION_ANCHOR_T,
    adjust_p_values_fdr,
    assign_fdr_classes,
    baseline_sequence_classification,
    bonferroni_log_adjusted,
    build_fdr_scan_grid,
    build_probability_scan_grid,
    compute_functional_probabilities,
    compute_one_sided_gated_p_values,
    counts_at_primary_fdr,
    create_fdr_classification_table,
    get_active_threshold_definitions,
    harmonic_number,
    main,
    savename,
    scan_fdr_targets,
    scan_probability_requirements,
    uniform_fdr_q_grid,
)
from dubanevics2026topological.threshold.functional_classification import (
    FUNCTIONAL,
    NON_FUNCTIONAL,
    UNCERTAIN,
    NonPositiveStandardError,
    assign_class,
    functional_probability,
)


def test_active_cutoff_definitions_match_specification() -> None:
    """Verify formulas, source precision and applied rounding for the two active cutoffs."""
    cutoffs = {c.name: c for c in get_active_threshold_definitions()}
    assert set(cutoffs.keys()) == {"papkou_original", "median_reflected"}

    # 1. Papkou original (applied at exact formula precision)
    papkou = cutoffs["papkou_original"]
    expected_papkou_unrounded = float(norm.ppf(0.999, loc=-0.721, scale=0.069))
    assert np.isclose(
        papkou.source_precision_threshold, expected_papkou_unrounded, atol=1e-15
    )
    assert np.isclose(
        papkou.source_precision_threshold, -0.5077739708744209, atol=1e-15
    )
    assert papkou.applied_threshold == -0.5077739708744209
    assert papkou.decimals is None
    assert papkou.metadata["published_rounded_reference"] == -0.507774

    # 2. Median reflected (applied at 6-decimal rounding)
    median_ref = cutoffs["median_reflected"]
    m = -0.7139811713950942
    g = -1.17480288668069
    h = m - g
    expected_median_unrounded = m + h
    assert np.isclose(
        median_ref.source_precision_threshold, expected_median_unrounded, atol=1e-15
    )
    assert np.isclose(
        median_ref.source_precision_threshold, -0.2531594561094983, atol=1e-15
    )
    assert median_ref.applied_threshold == -0.253159
    assert median_ref.decimals == 6
    assert (
        median_ref.metadata["raw_median_source_sha256"]
        == "a3f3fd421b7ad7d05c7ae870c190f2d0b5a4e998bfffb3b8f76652a37882e4b4"
    )


def test_drwatson_savename_is_deterministic_and_sorted() -> None:
    """Verify DrWatson savename produces sorted key=value pairs without collisions."""
    params = {"cutoff": "papkou_original", "T": -0.507774, "q": 0.95}
    name = savename(params, prefix="classification", suffix=".parquet")
    assert name == "classification_T=-0.507774_cutoff=papkou_original_q=0.95.parquet"

    # Permuted dictionary order must produce identical output
    params_reordered = {"q": 0.95, "cutoff": "papkou_original", "T": -0.507774}
    assert (
        savename(params_reordered, prefix="classification", suffix=".parquet") == name
    )


def test_scan_grid_properties() -> None:
    """Verify scan grid bounds, ordering, and exact presence of marked points."""
    grid = build_probability_scan_grid()
    assert len(grid) > 100
    assert (grid > 0.5).all(), (
        "Grid must be strictly above 0.5 to prevent boundary overlap"
    )
    assert (grid <= 0.99999 + 1e-9).all()
    assert (np.diff(grid) > 0).all(), "Grid must be strictly increasing"

    for m in MARKED_Q_VALUES:
        assert np.isclose(grid, m, atol=1e-9).any(), f"Marked point {m} must be in grid"


def test_fail_closed_on_invalid_standard_errors() -> None:
    """Non-positive or non-finite SE must raise NonPositiveStandardError."""
    s = [0.0, -0.5, 0.2]
    # Zero SE
    with pytest.raises(NonPositiveStandardError):
        compute_functional_probabilities(s, [0.01, 0.0, 0.02], threshold=-0.5)
    # Negative SE
    with pytest.raises(NonPositiveStandardError):
        compute_functional_probabilities(s, [0.01, -0.01, 0.02], threshold=-0.5)
    # Null / None SE
    with pytest.raises(NonPositiveStandardError):
        compute_functional_probabilities(s, [0.01, None, 0.02], threshold=-0.5)
    # Infinite / NaN SE
    with pytest.raises(NonPositiveStandardError):
        compute_functional_probabilities(s, [0.01, np.nan, 0.02], threshold=-0.5)


def test_assign_class_boundary_and_rounding() -> None:
    """Verify assign_class exact 12-decimal rounding for the lower bound."""
    # At q = 0.95, lower = round(1.0 - 0.95, 12) = 0.05
    classes = assign_class([0.95, 0.949999999, 0.0500000000000, 0.050000000001], 0.95)
    assert list(classes) == [FUNCTIONAL, UNCERTAIN, NON_FUNCTIONAL, UNCERTAIN]


def test_scan_partition_and_monotonicity_on_synthetic_data() -> None:
    """Verify exact partition and non-strict monotonicity on synthetic distributions."""
    rng = np.random.default_rng(42)
    p_synthetic = rng.uniform(0.0, 1.0, size=5000)
    grid = [0.51, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999]

    df_scan = scan_probability_requirements(p_synthetic, grid)
    assert df_scan.height == len(grid)

    f_counts = df_scan["functional_count"].to_list()
    nf_counts = df_scan["non_functional_count"].to_list()
    u_counts = df_scan["uncertain_count"].to_list()
    totals = df_scan["total_count"].to_list()

    # Exact partition
    for f, nf, u, tot in zip(f_counts, nf_counts, u_counts, totals):
        assert f + nf + u == 5000
        assert tot == 5000

    # Monotonicity: F non-increasing, NF non-increasing, U non-decreasing
    for i in range(len(grid) - 1):
        assert f_counts[i] >= f_counts[i + 1]
        assert nf_counts[i] >= nf_counts[i + 1]
        assert u_counts[i] <= u_counts[i + 1]


def test_validation_anchor_on_reference_data() -> None:
    """Verify numerical validation anchor T=-0.49739479076248694 at q=0.95."""
    source_path = scans.DEFAULT_SOURCE_PATH
    if not source_path.exists():
        pytest.skip(f"Processed growth-rate table not present: {source_path}")

    growth = pl.read_parquet(source_path)
    s = growth["s"].to_numpy()
    se = growth["se_pooled"].to_list()
    p = functional_probability(s, se, VALIDATION_ANCHOR_T)
    classes = assign_class(p, VALIDATION_ANCHOR_Q)

    counts = {
        "functional": int((classes == FUNCTIONAL).sum()),
        "non_functional": int((classes == NON_FUNCTIONAL).sum()),
        "uncertain": int((classes == UNCERTAIN).sum()),
    }
    assert counts == VALIDATION_ANCHOR_COUNTS


def test_reference_data_baseline_and_marked_counts() -> None:
    """Verify exact counts at marked points for both active cutoffs on reference data."""
    source_path = scans.DEFAULT_SOURCE_PATH
    if not source_path.exists():
        pytest.skip("Processed growth-rate table not present")

    growth = pl.read_parquet(source_path)
    s = growth["s"].to_numpy()
    se = growth["se_pooled"].to_list()

    # 1. Papkou original applied exact formula T = -0.5077739708744209
    t_papkou_formula = -0.5077739708744209
    p_papkou = functional_probability(s, se, t_papkou_formula)
    # Direct formula check at baseline q = 0.95
    c_papkou_base = assign_class(p_papkou, 0.95)
    f_pap = int((c_papkou_base == FUNCTIONAL).sum())
    nf_pap = int((c_papkou_base == NON_FUNCTIONAL).sum())
    u_pap = int((c_papkou_base == UNCERTAIN).sum())
    assert (f_pap, nf_pap, u_pap) == (16690, 237678, 6965)
    assert f_pap + nf_pap + u_pap == 261333

    # Fast scan check across marked points
    df_scan_pap = scan_probability_requirements(p_papkou, MARKED_Q_VALUES)
    pap_marked = {
        row["q"]: (
            row["functional_count"],
            row["non_functional_count"],
            row["uncertain_count"],
        )
        for row in df_scan_pap.to_dicts()
    }
    assert pap_marked[0.95] == (16690, 237678, 6965)
    assert pap_marked[0.99] == (16338, 231345, 13650)
    assert pap_marked[0.999] == (16023, 218629, 26681)
    assert pap_marked[0.9999] == (15800, 202481, 43052)
    assert pap_marked[0.99999] == (15627, 183757, 61949)

    # Verify the 4 exact grid points where exact formula vs rounded -0.507774 changes counts by NF+1, U-1
    diff_q_points = [0.99907, 0.9997, 0.999903, 0.999912]
    p_papkou_rounded = functional_probability(s, se, -0.507774)
    df_diff_formula = scan_probability_requirements(p_papkou, diff_q_points)
    df_diff_rounded = scan_probability_requirements(p_papkou_rounded, diff_q_points)
    for r_row, f_row in zip(df_diff_rounded.to_dicts(), df_diff_formula.to_dicts()):
        assert f_row["functional_count"] == r_row["functional_count"]
        assert f_row["non_functional_count"] == r_row["non_functional_count"] + 1
        assert f_row["uncertain_count"] == r_row["uncertain_count"] - 1

    # 2. Median reflected applied T = -0.253159
    p_median = functional_probability(s, se, -0.253159)
    # Direct formula check at baseline q = 0.95
    c_median_base = assign_class(p_median, 0.95)
    f_med = int((c_median_base == FUNCTIONAL).sum())
    nf_med = int((c_median_base == NON_FUNCTIONAL).sum())
    u_med = int((c_median_base == UNCERTAIN).sum())
    assert (f_med, nf_med, u_med) == (13301, 247373, 659)
    assert f_med + nf_med + u_med == 261333

    df_scan_med = scan_probability_requirements(p_median, MARKED_Q_VALUES)
    med_marked = {
        row["q"]: (
            row["functional_count"],
            row["non_functional_count"],
            row["uncertain_count"],
        )
        for row in df_scan_med.to_dicts()
    }
    assert med_marked[0.95] == (13301, 247373, 659)
    assert med_marked[0.99] == (13166, 247189, 978)
    assert med_marked[0.999] == (13032, 246887, 1414)
    assert med_marked[0.9999] == (12933, 246357, 2043)
    assert med_marked[0.99999] == (12856, 245579, 2898)


def test_baseline_sequence_classification_schema() -> None:
    """Verify per-sequence baseline classification DataFrame columns and types."""
    tiny_growth = pl.DataFrame(
        {
            "dna_sequence": ["ACTG", "AAAA"],
            "sequence_id": ["seq1", "seq2"],
            "s": [-0.1, -0.9],
            "se_pooled": [0.05, 0.04],
            "stop_codon_count": [0, 2],
        }
    )
    classified = baseline_sequence_classification(
        tiny_growth, threshold=-0.5, confidence=0.95
    )
    assert classified.columns == [
        "dna_sequence",
        "sequence_id",
        "s",
        "se_pooled",
        "stop_codon_count",
        "functional_probability",
        "functional_class",
    ]
    assert classified["functional_class"].to_list() == [FUNCTIONAL, NON_FUNCTIONAL]


def test_harmonic_number_exact() -> None:
    """Verify exact harmonic number calculation c(N) = sum(1/j)."""
    assert harmonic_number(1) == 1.0
    assert np.isclose(harmonic_number(4), 25.0 / 12.0)
    c_large = harmonic_number(261333)
    assert np.isclose(c_large, 13.05076757, atol=1e-6)


def test_one_sided_gated_p_values_and_se_validation() -> None:
    """Verify one-sided gated p-values testing H0: s <= T and fail-closed SE check."""
    s = [0.0, -0.5, -0.6]
    se = [0.1, 0.1, 0.1]
    is_cand, z, raw_p, log_raw_p, gated_p, log_gated_p = (
        compute_one_sided_gated_p_values(s, se, threshold=-0.5)
    )

    assert list(is_cand) == [True, True, False]
    assert np.isclose(z[0], 5.0)
    assert np.isclose(z[1], 0.0)
    assert np.isclose(gated_p[1], 0.5)
    # Non-candidate s = -0.6 < -0.5 has gated p* = 1.0, log_p* = 0.0
    assert gated_p[2] == 1.0
    assert log_gated_p[2] == 0.0
    # True raw p-value for non-candidate s = -0.6 is norm.sf(-1.0) ~ 0.8413
    assert np.isclose(raw_p[2], norm.sf(-1.0))
    assert np.isclose(log_raw_p[2], norm.logsf(-1.0))

    # Non-positive SE must raise NonPositiveStandardError
    with pytest.raises(NonPositiveStandardError):
        compute_one_sided_gated_p_values(s, [0.1, 0.0, 0.1], threshold=-0.5)
    with pytest.raises(NonPositiveStandardError):
        compute_one_sided_gated_p_values(s, [0.1, -0.05, 0.1], threshold=-0.5)


def test_adjust_p_values_fdr_against_statsmodels() -> None:
    """Verify vectorized log-space BH and BY match statsmodels reference."""
    from statsmodels.stats.multitest import multipletests

    # Synthetic p-values including ties, extremes, and 1.0
    p_synthetic = np.array(
        [1e-15, 1e-6, 0.001, 0.01, 0.04, 0.05, 0.06, 0.2, 0.5, 1.0, 1.0]
    )
    log_p = np.log(p_synthetic)

    # 1. BH
    adj_bh, log_adj_bh = adjust_p_values_fdr(log_p, method="bh")
    _, sm_bh, _, _ = multipletests(p_synthetic, method="fdr_bh")
    assert np.allclose(adj_bh, sm_bh, atol=1e-12)
    assert np.all(np.diff(np.sort(adj_bh)) >= -1e-15), (
        "Adjusted p-values must be monotone"
    )

    # 2. BY
    adj_by, log_adj_by = adjust_p_values_fdr(log_p, method="by")
    _, sm_by, _, _ = multipletests(p_synthetic, method="fdr_by")
    assert np.allclose(adj_by, sm_by, atol=1e-12)
    assert np.all(adj_by >= adj_bh - 1e-15), "BY must be conservative relative to BH"


def test_scan_grid_contains_the_uniform_hundredth_steps() -> None:
    """The saved scan must include q = 0, 0.01, ..., 1 without dropping marked targets."""
    uniform = uniform_fdr_q_grid()
    assert uniform.size == 101
    assert uniform[0] == 0.0 and uniform[-1] == 1.0
    assert np.allclose(np.diff(uniform), 0.01)
    grid = build_fdr_scan_grid()
    missing = [float(q) for q in uniform if not np.any(np.isclose(grid, q, atol=0))]
    assert missing == []
    for marked in (0.0, 0.025, 0.05, 1.0):
        assert any(float(q) == marked or np.isclose(q, marked, atol=0) for q in grid)


def test_bonferroni_adjustment_is_min_one_n_p() -> None:
    log_p = np.log(np.array([0.01, 0.2, 1.0]))
    adjusted, log_adjusted = bonferroni_log_adjusted(log_p)
    expected = np.minimum(1.0, np.exp(log_p) * 3)
    assert np.allclose(adjusted, expected)
    assert np.allclose(log_adjusted, np.log(expected))


def test_candidate_threshold_is_the_six_decimal_median_reflection() -> None:
    reflected = next(
        item
        for item in get_active_threshold_definitions()
        if item.name == "median_reflected"
    )
    assert round(reflected.applied_threshold, 6) == -0.253159


def test_fdr_partition_and_monotonicity_on_synthetic_data() -> None:
    """Verify partition, monotonicity, and boundary recoveries for FDR scans."""
    n = 1000
    n_cand = 200
    is_cand = np.zeros(n, dtype=bool)
    is_cand[:n_cand] = True

    # Candidate log_p ascending from -50 to -0.05; non-candidates 0.0
    log_p = np.zeros(n, dtype=np.float64)
    log_p[:n_cand] = np.linspace(-50, -0.05, n_cand)

    adj_bh, log_adj_bh = adjust_p_values_fdr(log_p, method="bh")
    adj_by, log_adj_by = adjust_p_values_fdr(log_p, method="by")

    grid = build_fdr_scan_grid()
    df_bh = scan_fdr_targets(is_cand, log_adj_bh, grid)
    df_by = scan_fdr_targets(is_cand, log_adj_by, grid)

    # 1. Partition & boundaries for BH
    row_0_bh = df_bh.filter(pl.col("fdr_target") == 0.0).to_dicts()[0]
    assert row_0_bh["functional_count"] == 0
    assert row_0_bh["uncertain_count"] == n_cand
    assert row_0_bh["non_functional_count"] == n - n_cand

    row_1_bh = df_bh.filter(pl.col("fdr_target") == 1.0).to_dicts()[0]
    assert row_1_bh["functional_count"] == n_cand
    assert row_1_bh["uncertain_count"] == 0
    assert row_1_bh["non_functional_count"] == n - n_cand

    # 2. BY discoveries subset of BH
    f_bh = df_bh["functional_count"].to_numpy()
    f_by = df_by["functional_count"].to_numpy()
    assert np.all(f_by <= f_bh), (
        "BY functional discoveries must be a subset of BH at every target"
    )


def test_fdr_extreme_tail_log_precision() -> None:
    """Verify log-space decision authority correctly handles extreme tail thresholds like 1e-323 without floor."""
    is_cand = np.array([True, True, False])
    # Extreme log adjusted values: -1000, -500, 0
    log_adj = np.array([-1000.0, -500.0, 0.0])

    # Target 1e-323: log(1e-323) ~ -743.7
    target_extreme = 1e-323
    classes, f_cnt, u_cnt, nf_cnt = assign_fdr_classes(is_cand, log_adj, target_extreme)
    # Row 0: -1000 <= -743.7 -> Functional
    # Row 1: -500 > -743.7 -> Uncertain
    # Row 2: non-candidate -> Non-functional
    assert list(classes) == [FUNCTIONAL, UNCERTAIN, NON_FUNCTIONAL]
    assert (f_cnt, u_cnt, nf_cnt) == (1, 1, 1)


def test_fdr_reference_data_exact_spec_counts() -> None:
    """Verify exact counts for the four analyses on reference data matching the approved specification."""
    source_path = scans.DEFAULT_SOURCE_PATH
    if not source_path.exists():
        pytest.skip("Processed growth-rate table not present")

    growth = pl.read_parquet(source_path)
    s = growth["s"].to_numpy()
    se = growth["se_pooled"].to_numpy()
    n_total = len(s)
    assert n_total == 261333

    # 1. Papkou original (T = -0.5077739708744209)
    t_papkou = -0.5077739708744209
    cand_pap, z_pap, raw_p_pap, log_raw_p_pap, p_gated_pap, log_p_pap = (
        compute_one_sided_gated_p_values(s, se, t_papkou)
    )
    assert int(cand_pap.sum()) == 18030
    assert n_total - int(cand_pap.sum()) == 243303

    # Raw alpha=0.05 unadjusted baseline check cited in prompt
    raw_sig_pap = cand_pap & (raw_p_pap <= 0.05)
    assert int(raw_sig_pap.sum()) == 16690
    assert 18030 - int(raw_sig_pap.sum()) == 1340

    # Papkou BH at 5% FDR
    adj_bh_pap, log_adj_bh_pap = adjust_p_values_fdr(log_p_pap, method="bh")
    _, f_bh_pap, u_bh_pap, nf_bh_pap = assign_fdr_classes(
        cand_pap, log_adj_bh_pap, 0.05
    )
    assert (f_bh_pap, u_bh_pap, nf_bh_pap) == (16160, 1870, 243303)

    # Papkou BY at 5% FDR (Primary)
    adj_by_pap, log_adj_by_pap = adjust_p_values_fdr(log_p_pap, method="by")
    _, f_by_pap, u_by_pap, nf_by_pap = assign_fdr_classes(
        cand_pap, log_adj_by_pap, 0.05
    )
    assert (f_by_pap, u_by_pap, nf_by_pap) == (15851, 2179, 243303)
    assert f_by_pap <= f_bh_pap

    # 2. Median reflected (applied 6-dec T = -0.253159)
    t_med = -0.253159
    cand_med, z_med, raw_p_med, log_raw_p_med, p_gated_med, log_p_med = (
        compute_one_sided_gated_p_values(s, se, t_med)
    )
    assert int(cand_med.sum()) == 13626
    assert n_total - int(cand_med.sum()) == 247707

    # Raw alpha=0.05 unadjusted baseline check cited in prompt
    raw_sig_med = cand_med & (raw_p_med <= 0.05)
    assert int(raw_sig_med.sum()) == 13301
    assert 13626 - int(raw_sig_med.sum()) == 325

    # Median BH at 5% FDR
    adj_bh_med, log_adj_bh_med = adjust_p_values_fdr(log_p_med, method="bh")
    _, f_bh_med, u_bh_med, nf_bh_med = assign_fdr_classes(
        cand_med, log_adj_bh_med, 0.05
    )
    assert (f_bh_med, u_bh_med, nf_bh_med) == (13078, 548, 247707)

    # Median BY at 5% FDR (Primary)
    adj_by_med, log_adj_by_med = adjust_p_values_fdr(log_p_med, method="by")
    _, f_by_med, u_by_med, nf_by_med = assign_fdr_classes(
        cand_med, log_adj_by_med, 0.05
    )
    assert (f_by_med, u_by_med, nf_by_med) == (12960, 666, 247707)
    assert f_by_med <= f_bh_med


def test_fdr_classification_table_schema() -> None:
    """Verify per-sequence FDR classification table schema and columns."""
    tiny_growth = pl.DataFrame(
        {
            "dna_sequence": ["ACTG", "AAAA"],
            "sequence_id": ["seq1", "seq2"],
            "s": [-0.1, -0.9],
            "se_pooled": [0.05, 0.04],
            "stop_codon_count": [0, 2],
        }
    )
    table = create_fdr_classification_table(
        tiny_growth, threshold=-0.5, method="by", target=0.05
    )
    expected_cols = [
        "dna_sequence",
        "sequence_id",
        "s",
        "se_pooled",
        "stop_codon_count",
        "z_score",
        "raw_one_sided_p_value",
        "log_raw_one_sided_p_value",
        "gated_p_value",
        "log_gated_p_value",
        "adjusted_p_value",
        "log_adjusted_p_value",
        "is_baseline_candidate",
        "fdr_class",
    ]
    assert table.columns == expected_cols
    assert table["fdr_class"].to_list() == [FUNCTIONAL, NON_FUNCTIONAL]


def test_main_prints_primary_fdr_counts_for_by_bh_and_bonferroni(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary printer walks every method, including Bonferroni."""
    n = 20
    is_cand = np.array([True] * 8 + [False] * 12)
    log_p = np.linspace(-8.0, 0.0, n)
    grid = build_fdr_scan_grid()
    methods: dict[str, dict[str, object]] = {}
    expected_lines: list[str] = []
    for method, display in (
        ("by", "Benjamini-Yekutieli (BY)"),
        ("bh", "Benjamini-Hochberg (BH)"),
        ("bonferroni", "Bonferroni"),
    ):
        if method == "bonferroni":
            _adjusted, log_adjusted = bonferroni_log_adjusted(log_p)
        else:
            _adjusted, log_adjusted = adjust_p_values_fdr(log_p, method=method)
        scan = scan_fdr_targets(is_cand, log_adjusted, grid)
        counts = counts_at_primary_fdr(scan)
        _classes, f_count, u_count, nf_count = assign_fdr_classes(
            is_cand, log_adjusted, 0.05
        )
        assert counts == {
            "functional": f_count,
            "uncertain": u_count,
            "non_functional": nf_count,
        }
        methods[method] = {
            "method_name": display,
            "counts_at_primary_fdr_0_05": counts,
        }
        expected_lines.append(
            f"    {display}: F={f_count}, U={u_count}, NF={nf_count}"
        )

    summary = {
        "cutoffs": {
            "papkou_original": {
                "applied_threshold": -0.5,
                "baseline_counts_q_0_95": {
                    "functional": 1,
                    "non_functional": 2,
                    "uncertain": 3,
                },
            }
        },
        "fdr_screening_analyses": {
            "cutoffs": {
                "papkou_original": {
                    "applied_threshold": -0.5,
                    "baseline_candidates_count": 8,
                    "baseline_non_candidates_count": 12,
                    "methods": methods,
                }
            }
        },
    }

    assert "counts_at_primary_fdr(df_bonf)" in inspect.getsource(scans.run_pipeline)

    def _pipeline() -> dict[str, object]:
        return {"summary": summary}

    monkeypatch.setattr(scans, "run_pipeline", _pipeline)
    main()
    printed = capsys.readouterr().out
    for line in expected_lines:
        assert line in printed


def _fake_scan_output(directory, cutoffs=("papkou_original", "median_reflected")):
    directory.mkdir(parents=True)
    for cutoff in cutoffs:
        for method in ("by", "bh"):
            (directory / f"fdr_classification_T=-0.5_cutoff={cutoff}_method={method}_q=0.05.parquet").write_bytes(
                f"{cutoff}-{method}".encode()
            )
    (directory / "provenance.json").write_text("{}")


def test_each_variant_gets_its_own_by_table_and_the_scan_provenance(tmp_path) -> None:
    out, processed = tmp_path / "out", tmp_path / "processed"
    _fake_scan_output(out)
    published = scans.publish_classification(out, processed)
    assert set(published) == {"published", "reflected"}
    for variant, cutoff in scans.VARIANT_CUTOFFS.items():
        files = sorted(path.name for path in published[variant].iterdir())
        assert files == [
            f"fdr_classification_T=-0.5_cutoff={cutoff}_method=by_q=0.05.parquet",
            "provenance.json",
        ]


def test_republishing_replaces_the_variant_directory(tmp_path) -> None:
    out, processed = tmp_path / "out", tmp_path / "processed"
    _fake_scan_output(out)
    stale = processed / "classification" / "published" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    scans.publish_classification(out, processed)
    assert not stale.exists()


def test_publishing_needs_one_by_table_per_cutoff(tmp_path) -> None:
    out = tmp_path / "out"
    _fake_scan_output(out, cutoffs=("papkou_original",))
    with pytest.raises(FileNotFoundError, match="median_reflected"):
        scans.publish_classification(out, tmp_path / "processed")


def test_missing_growth_rates_names_the_recipes(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="just sequence-growth-rates"):
        scans.run_pipeline(tmp_path / "missing.parquet", tmp_path / "out", tmp_path / "p")
