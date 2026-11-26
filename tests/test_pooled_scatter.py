"""The peak-uncertainty scatter moves to the pooled scale and gains the wild type."""

from __future__ import annotations

import math

import polars as pl
import pytest

from dubanevics2026topological.analysis.pooled_scatter import (
    pooled_scatter,
    translate_dna,
)

WILD_TYPE = "GCCGATCTC"
PEAK = "AAAAAAAAC"
VALLEY = "AAAAAAAAG"
UNFITTED = "AAAAAAACA"  # measured and published, but with no growth-rate fit
R_WT = 0.038
SE_WT = 0.001
# (sequence, r - r_wt, se_pooled) of the fitted sequences; the others are unfitted.
FITTED = ((PEAK, 0.5, 0.04), (VALLEY, -0.2, 0.08))
UNFITTED_S, UNFITTED_SE_POOLED = -0.3, 0.05


def _published_se(se_pooled: float) -> float:
    """Papkou et al. add the wild type's term in quadrature to every standard error."""
    return math.sqrt(se_pooled**2 + SE_WT**2)


def peak_checkpoint() -> dict:
    """A peak checkpoint as the analysis writes it: published scale, no wild type."""
    rows = [
        (PEAK, "KKT", 0.5, _published_se(0.04), True),
        (VALLEY, "KKK", -0.2, _published_se(0.08), False),
        (UNFITTED, "KKT", UNFITTED_S, _published_se(UNFITTED_SE_POOLED), False),
    ]
    scatter = pl.DataFrame(
        rows,
        schema=[
            "dna_sequence",
            "aa_sequence",
            "raw_fitness",
            "standard_error",
            "is_peak",
        ],
        orient="row",
    )
    peaks = pl.DataFrame({"dna_sequence": [PEAK]})
    return {
        "tables": {
            "study_clamped_scatter": scatter,
            "candidate_clamped_scatter": scatter,
            "study_clamped_peaks": peaks,
            "candidate_clamped_peaks": peaks,
        },
        "diagnostics": {
            "wild_type_sequence": WILD_TYPE,
            "fitness_domain": [-0.2, 0.5],
            "thresholds": {"study_clamped": -0.5, "candidate_clamped": -0.25},
            "wild_type_scatter_excluded": True,
            "wild_type_scatter_excluded_count": 0,
        },
    }


def growth_table() -> pl.DataFrame:
    """The growth-rate table: the wild type and the fitted sequences only."""
    rows = [(WILD_TYPE, R_WT, SE_WT, 0.004)]
    rows += [(dna, R_WT + s, se, 2 * se) for dna, s, se in FITTED]
    return pl.DataFrame(
        rows,
        schema=["dna_sequence", "r", "se_pooled", "se_replicate"],
        orient="row",
    )


def compose(peak=None, growth=None, **tolerances):
    return pooled_scatter(
        peak or peak_checkpoint(),
        growth_table() if growth is None else growth,
        **{"identity_tolerance": 1e-6, "standard_error_tolerance": 1e-3, **tolerances},
    )


def rows(result, key="study_clamped"):
    return {
        row["dna_sequence"]: row
        for row in result["tables"][f"{key}_scatter"].to_dicts()
    }


def test_the_wild_type_is_plotted_at_zero_with_its_own_pooled_standard_error() -> None:
    for key in ("study_clamped", "candidate_clamped"):
        wild_type = rows(compose(), key)[WILD_TYPE]
        assert wild_type["raw_fitness"] == 0.0
        assert wild_type["standard_error"] == SE_WT
        assert wild_type["aa_sequence"] == "ADL"
        assert wild_type["is_peak"] is False


def test_the_wild_type_peak_flag_comes_from_the_peak_table() -> None:
    peak = peak_checkpoint()
    for key in ("study_clamped", "candidate_clamped"):
        peak["tables"][f"{key}_peaks"] = pl.DataFrame({"dna_sequence": [WILD_TYPE]})
    assert rows(compose(peak))[WILD_TYPE]["is_peak"] is True


def test_fitted_sequences_take_the_pooled_standard_error_and_r_minus_r_wt() -> None:
    by_id = rows(compose())
    for dna, s, se_pooled in FITTED:
        assert by_id[dna]["standard_error"] == se_pooled  # not the published one
        assert by_id[dna]["raw_fitness"] == pytest.approx((R_WT + s) - R_WT, abs=0)


def test_growth_rows_outside_the_scatter_are_ignored() -> None:
    """The full-set table holds sequences that are on no scatter."""
    growth = growth_table()
    extra = pl.DataFrame(
        [("AAAAAAGGG", R_WT + 0.9, 0.2, 0.3), ("AAAAAAGGC", R_WT - 0.9, 0.2, 0.3)],
        schema=growth.schema,
        orient="row",
    )
    plain = compose()
    bigger = compose(growth=pl.concat([growth, extra]))
    for key in ("study_clamped", "candidate_clamped"):
        assert bigger["tables"][f"{key}_scatter"].equals(
            plain["tables"][f"{key}_scatter"]
        )
    assert bigger["diagnostics"] == plain["diagnostics"]


def test_unfitted_sequences_keep_s_and_lose_the_wild_type_term() -> None:
    row = rows(compose())[UNFITTED]
    assert row["raw_fitness"] == UNFITTED_S
    assert row["standard_error"] == pytest.approx(UNFITTED_SE_POOLED, rel=1e-12)


def test_every_measured_sequence_stays_once_with_its_identity_and_peak_flag() -> None:
    peak = peak_checkpoint()
    result = compose(peak)
    for key in ("study_clamped", "candidate_clamped"):
        frame = result["tables"][f"{key}_scatter"]
        assert frame["dna_sequence"].to_list() == [PEAK, VALLEY, UNFITTED, WILD_TYPE]
        assert frame.columns == [
            "dna_sequence",
            "aa_sequence",
            "raw_fitness",
            "standard_error",
            "is_peak",
        ]
        assert frame["is_peak"].to_list() == [True, False, False, False]
        assert frame["aa_sequence"].to_list()[:3] == ["KKT", "KKK", "KKT"]


def test_a_published_wild_type_row_is_replaced_by_the_fit_not_duplicated() -> None:
    peak = peak_checkpoint()
    for key in ("study_clamped", "candidate_clamped"):
        frame = peak["tables"][f"{key}_scatter"]
        peak["tables"][f"{key}_scatter"] = pl.concat(
            [
                frame,
                pl.DataFrame(
                    {
                        "dna_sequence": [WILD_TYPE],
                        "aa_sequence": ["ADL"],
                        "raw_fitness": [0.0],
                        "standard_error": [0.5],
                        "is_peak": [False],
                    }
                ),
            ]
        )
    wild_type = rows(compose(peak))[WILD_TYPE]
    assert wild_type["standard_error"] == SE_WT
    assert compose(peak)["tables"]["study_clamped_scatter"].height == 4


def test_the_inputs_are_not_modified() -> None:
    peak = peak_checkpoint()
    before = peak["tables"]["study_clamped_scatter"].clone()
    compose(peak)
    assert peak["tables"]["study_clamped_scatter"].equals(before)
    assert peak["diagnostics"]["wild_type_scatter_excluded"] is True


def test_diagnostics_say_what_was_measured_and_what_was_reconstructed() -> None:
    diagnostics = compose()["diagnostics"]
    assert diagnostics["wild_type_sequence"] == WILD_TYPE
    assert diagnostics["wild_type_scatter_excluded"] is False
    assert diagnostics["fitness_domain"] == [UNFITTED_S, 0.5]
    assert diagnostics["thresholds"] == {
        "study_clamped": -0.5,
        "candidate_clamped": -0.25,
    }
    pooled = diagnostics["pooled_standard_error"]
    assert pooled["wild_type_standard_error"] == SE_WT
    # Counted per landscape: two sequences and the wild type, one reconstructed.
    assert pooled["fitted_sequence_count"] == {
        "study_clamped": 3,
        "candidate_clamped": 3,
    }
    assert pooled["reconstructed_sequence_count"] == {
        "study_clamped": 1,
        "candidate_clamped": 1,
    }
    assert pooled["max_identity_gap"] == pytest.approx(0.0, abs=1e-12)
    assert 0 <= pooled["max_relative_reconstruction_gap"] < 1e-9
    assert (pooled["identity_tolerance"], pooled["standard_error_tolerance"]) == (
        1e-6,
        1e-3,
    )


def test_the_identity_r_minus_r_wt_equals_s_is_still_checked_per_sequence() -> None:
    growth = growth_table().with_columns(
        pl.when(pl.col("dna_sequence") == VALLEY)
        .then(pl.col("r") + 1e-4)
        .otherwise(pl.col("r"))
        .alias("r")
    )
    with pytest.raises(ValueError, match="r_i - r_wt = s_i") as error:
        compose(growth=growth)
    assert VALLEY in str(error.value) and PEAK not in str(error.value)
    # Within the config tolerance the same table passes.
    compose(growth=growth, identity_tolerance=1e-3)


def test_a_published_standard_error_that_is_not_pooled_plus_wild_type_fails() -> None:
    peak = peak_checkpoint()
    for key in ("study_clamped", "candidate_clamped"):
        frame = peak["tables"][f"{key}_scatter"]
        peak["tables"][f"{key}_scatter"] = frame.with_columns(
            pl.when(pl.col("dna_sequence") == PEAK)
            .then(pl.col("standard_error") * 1.01)
            .otherwise(pl.col("standard_error"))
            .alias("standard_error")
        )
    with pytest.raises(
        ValueError, match="ADR 2026-10-22-pooled-standard-error-for-bootstrap"
    ):
        compose(peak)
    compose(peak, standard_error_tolerance=0.05)


def test_a_published_standard_error_at_or_below_the_wild_types_fails() -> None:
    peak = peak_checkpoint()
    for key in ("study_clamped", "candidate_clamped"):
        frame = peak["tables"][f"{key}_scatter"]
        peak["tables"][f"{key}_scatter"] = frame.with_columns(
            pl.when(pl.col("dna_sequence") == UNFITTED)
            .then(SE_WT)
            .otherwise(pl.col("standard_error"))
            .alias("standard_error")
        )
    with pytest.raises(ValueError, match="cannot be removed") as error:
        compose(peak)
    assert UNFITTED in str(error.value)


def test_nothing_is_reconstructed_without_a_fit_to_validate_it_against() -> None:
    peak = peak_checkpoint()
    growth = growth_table().filter(pl.col("dna_sequence") == WILD_TYPE)
    with pytest.raises(ValueError, match="cannot be validated"):
        compose(peak, growth)


def test_the_growth_table_must_fit_the_wild_type() -> None:
    growth = growth_table().filter(pl.col("dna_sequence") != WILD_TYPE)
    with pytest.raises(ValueError, match="no fit for the wild type"):
        compose(growth=growth)


def test_a_bad_growth_table_fails_closed() -> None:
    with pytest.raises(ValueError, match="Duplicate growth-rate rows"):
        compose(growth=pl.concat([growth_table(), growth_table().head(1)]))
    with pytest.raises(ValueError, match="not finite and positive"):
        compose(
            growth=growth_table().with_columns(
                pl.when(pl.col("dna_sequence") == PEAK)
                .then(0.0)
                .otherwise(pl.col("se_pooled"))
                .alias("se_pooled")
            )
        )
    with pytest.raises(ValueError, match="missing columns"):
        compose(growth=growth_table().drop("se_pooled"))


def test_bad_settings_and_a_bad_checkpoint_fail_closed() -> None:
    for name in ("identity_tolerance", "standard_error_tolerance"):
        with pytest.raises(ValueError, match=name):
            compose(**{name: -1.0})
        with pytest.raises(ValueError, match=name):
            compose(**{name: float("nan")})
    no_wild_type = peak_checkpoint()
    del no_wild_type["diagnostics"]["wild_type_sequence"]
    with pytest.raises(ValueError, match="wild-type sequence"):
        compose(no_wild_type)
    no_peaks = peak_checkpoint()
    del no_peaks["tables"]["study_clamped_peaks"]
    with pytest.raises(KeyError, match="study_clamped_peaks"):
        compose(no_peaks)
    no_scatter = peak_checkpoint()
    no_scatter["tables"] = {}
    with pytest.raises(ValueError, match="no scatter table"):
        compose(no_scatter)
    narrow = peak_checkpoint()
    narrow["tables"]["study_clamped_scatter"] = narrow["tables"][
        "study_clamped_scatter"
    ].drop("is_peak")
    with pytest.raises(ValueError, match="missing columns"):
        compose(narrow)


def test_translate_dna_reads_the_standard_code_in_frame() -> None:
    assert translate_dna(WILD_TYPE) == "ADL"
    assert translate_dna("AAAAAAACA") == "KKT"
    assert translate_dna("TAATGATAG") == "***"
    assert translate_dna("") == ""
    for bad in ("GC", "GCCGATCTN", "gccgatctc"):
        with pytest.raises(ValueError, match="codon-aligned"):
            translate_dna(bad)
