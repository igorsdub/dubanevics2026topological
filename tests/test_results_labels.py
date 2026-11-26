"""Tests for result label formatting, parsing, and grammar validation per ADR 0016."""

from __future__ import annotations

import pytest

from dubanevics2026topological.results.labels import (
    MalformedLabelError,
    parse_label,
)


def test_canonical_examples_round_trip() -> None:
    """Verify canonical examples from ADR 0016 lines 198-201 round-trip identically."""
    examples = [
        "tree__graph=full_tree=nominal_hash=a1b2c3d4",
        "tree_layout__graph=full_tree=nominal_hash=a1b2c3d4",
        "fitness__thr=reflected_fdr=by_q=0.05_hash=c23f8986",
        (
            "bootstrap__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_"
            "host=local_run=51756602_hash=7a21c3f0"
        ),
        (
            "tree__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_"
            "tree=band_bandq=0.95_hash=b4d9e0a1"
        ),
        (
            "tree_layout__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_"
            "tree=band_bandq=0.95_hash=b4d9e0a1"
        ),
    ]
    for label_str in examples:
        parsed = parse_label(label_str)
        formatted = parsed.format()
        assert formatted == label_str
        # Re-parse formatted to verify stability
        assert parse_label(formatted) == parsed


def test_tree_layout_requires_tree_state_and_excludes_appearance() -> None:
    with pytest.raises(MalformedLabelError):
        parse_label("tree_layout__graph=full_hash=a1b2c3d4")
    with pytest.raises(MalformedLabelError):
        parse_label("tree_layout__graph=full_tree=nominal_hash=a1b2c3d4_style=vanilla")


def test_round_trip_with_negative_and_full_precision_floats() -> None:
    """Verify that negative floats and full-precision numbers round-trip precisely on numeric keys."""
    # Negative and full-precision float under q
    label_str = "fitness__thr=reflected_fdr=by_q=-0.5077739708744209_hash=abcdef12"
    parsed = parse_label(label_str)
    assert parsed.params["thr"] == "reflected"
    assert parsed.params["q"] == -0.5077739708744209
    assert parsed.format() == label_str

    # Full-precision positive float under q
    label_q = "fitness__thr=reflected_fdr=by_q=0.0500000000000001_hash=abcdef12"
    parsed_q = parse_label(label_q)
    assert parsed_q.params["q"] == 0.0500000000000001
    assert parsed_q.format() == label_q

    # Negative float in bandq
    band_label = "tree__graph=full_draws=500_seed=42_tree=band_bandq=-0.1234567890123456_hash=12345678"
    parsed_band = parse_label(band_label)
    assert parsed_band.params["bandq"] == -0.1234567890123456
    assert parsed_band.format() == band_label

    # Negative float under mixture p
    mix_label = "fitness__thr=mixture_fdr=by_q=0.05_p=-0.2531590000000001_hash=12345678"
    parsed_mix = parse_label(mix_label)
    assert parsed_mix.params["p"] == -0.2531590000000001
    assert parsed_mix.format() == mix_label


def test_reject_numeric_threshold() -> None:
    """thr only takes published|reflected|mixture; exact numerical value is never in the label."""
    with pytest.raises(MalformedLabelError, match="cannot contain digits"):
        parse_label("fitness__thr=-0.5077739708744209_fdr=by_q=0.05_hash=abcdef12")

    with pytest.raises(MalformedLabelError, match="cannot contain digits"):
        parse_label("fitness__thr=0.05_fdr=by_q=0.05_hash=abcdef12")

    with pytest.raises(MalformedLabelError, match="'thr' must be one of"):
        parse_label("fitness__thr=unknown_fdr=by_q=0.05_hash=abcdef12")


def test_all_product_types_and_pipeline_order() -> None:
    """Verify all standard products format in strict pipeline data-flow order."""
    # leaves nominal
    l_nom = "leaves__graph=full_tree=nominal_hash=11112222"
    assert parse_label(l_nom).format() == l_nom

    # leaves band
    l_band = "leaves__thr=published_fdr=none_draws=100_seed=1_tree=band_bandq=0.9_hash=33334444"
    assert parse_label(l_band).format() == l_band

    # landscape_summary
    s_label = (
        "landscape_summary__thr=mixture_fdr=by_q=0.01_p=0.95_tree=nominal_hash=55556666"
    )
    assert parse_label(s_label).format() == s_label

    # ridges
    r_label = "ridges__thr=reflected_fdr=by_q=0.05_draws=1000_seed=2026_tree=band_bandq=0.95_hash=77778888"
    assert parse_label(r_label).format() == r_label


def test_figure_labels_with_style_after_hash() -> None:
    """Verify single-run figure labels allow style=annotated/vanilla after hash=."""
    fig_annotated = (
        "tree_overview__thr=reflected_fdr=by_q=0.05_hash=c23f8986_style=annotated"
    )
    parsed = parse_label(fig_annotated)
    assert parsed.style == "annotated"
    assert parsed.short_hash == "c23f8986"
    assert parsed.format() == fig_annotated

    fig_vanilla = (
        "persistence_diagram__graph=full_tree=nominal_hash=a1b2c3d4_style=vanilla"
    )
    assert parse_label(fig_vanilla).style == "vanilla"


def test_properties_and_scientific_params() -> None:
    """Verify ResultLabel convenience properties and scientific_params extraction."""
    label_str = (
        "bootstrap__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_"
        "host=local_run=51756602_hash=7a21c3f0"
    )
    parsed = parse_label(label_str)
    assert parsed.product_name == "bootstrap"
    assert parsed.short_hash == "7a21c3f0"
    assert parsed.host == "local"
    assert parsed.run == 51756602
    assert parsed.style is None

    # scientific_params excludes host, run, hash
    sci = parsed.scientific_params
    assert "host" not in sci
    assert "run" not in sci
    assert "hash" not in sci
    assert sci == {
        "thr": "reflected",
        "fdr": "by",
        "q": 0.05,
        "draws": 10000,
        "seed": 20261019,
    }


def test_reject_missing_or_multiple_double_underscores() -> None:
    """Reject labels without exactly one '__' separator."""
    with pytest.raises(MalformedLabelError, match="must contain '__'"):
        parse_label("fitness_thr=reflected_fdr=none_hash=12345678")

    with pytest.raises(MalformedLabelError, match="exactly one '__' separator"):
        parse_label("fitness__thr=reflected__fdr=none_hash=12345678")


def test_reject_empty_name_or_params() -> None:
    """Reject empty product name or parameter string."""
    with pytest.raises(MalformedLabelError, match="Empty product/view name"):
        parse_label("__thr=reflected_hash=12345678")

    with pytest.raises(MalformedLabelError, match="Empty parameters"):
        parse_label("fitness__")


def test_reject_unknown_key() -> None:
    """Reject keys not in ALLOWED_KEYS."""
    with pytest.raises(MalformedLabelError, match="Unknown parameter key 'cutoff'"):
        parse_label("fitness__cutoff=0.05_fdr=none_hash=12345678")


def test_reject_invalid_key_characters() -> None:
    """Keys must be lowercase alphabetic only."""
    # Uppercase
    with pytest.raises(MalformedLabelError, match="Unknown parameter key 'THR'"):
        parse_label("fitness__THR=reflected_fdr=none_hash=12345678")


def test_reject_underscores_and_hyphens_inside_values() -> None:
    """Values cannot contain underscores or hyphens, except leading minus on numbers."""
    from dubanevics2026topological.results.labels import format_label

    with pytest.raises(MalformedLabelError, match="cannot contain underscores"):
        format_label(
            "fitness",
            {"thr": "median_reflected", "fdr": "none", "hash": "12345678"},
        )

    with pytest.raises(MalformedLabelError, match="only permitted as a leading minus"):
        parse_label("fitness__thr=median-reflected_fdr=none_hash=12345678")

    with pytest.raises(MalformedLabelError, match="only permitted as a leading minus"):
        parse_label("tree__graph=full_draws=100-200_seed=1_tree=nominal_hash=12345678")

    with pytest.raises(MalformedLabelError, match="only permitted on numeric values"):
        parse_label("fitness__thr=-reflected_fdr=none_hash=12345678")


def test_reject_numbers_not_after_equals() -> None:
    """Numbers are forbidden in non-numeric parameter tokens."""
    with pytest.raises(MalformedLabelError, match="cannot contain digits"):
        parse_label("fitness__thr=reflected2_fdr=none_hash=12345678")

    with pytest.raises(MalformedLabelError, match="cannot contain digits"):
        parse_label("tree__graph=full_tree=nominal1_hash=12345678")


def test_reject_keys_out_of_pipeline_order() -> None:
    """Keys must be in pipeline data-flow order, not alphabetical or random."""
    # thr after fdr
    with pytest.raises(MalformedLabelError, match="out of pipeline data-flow order"):
        parse_label("fitness__fdr=by_thr=reflected_q=0.05_hash=12345678")

    # q before fdr
    with pytest.raises(MalformedLabelError, match="out of pipeline data-flow order"):
        parse_label("fitness__thr=reflected_q=0.05_fdr=by_hash=12345678")

    # hash before run
    with pytest.raises(MalformedLabelError, match="out of pipeline data-flow order"):
        parse_label(
            "bootstrap__thr=reflected_fdr=by_q=0.05_draws=100_seed=1_hash=12345678_host=local_run=123"
        )


def test_reject_duplicate_keys() -> None:
    """Duplicate keys are strictly rejected."""
    with pytest.raises(MalformedLabelError, match="Duplicate parameter key 'thr'"):
        parse_label("fitness__thr=reflected_thr=published_fdr=none_hash=12345678")


def test_reject_missing_equals() -> None:
    """Tokens must contain '='."""
    with pytest.raises(MalformedLabelError, match="missing '='"):
        parse_label("fitness__reflected_fdr=none_hash=12345678")


def test_reject_missing_required_product_keys() -> None:
    """Product contracts require specific keys per product type."""
    # fitness requires thr and fdr
    with pytest.raises(MalformedLabelError, match="requires threshold key 'thr'"):
        parse_label("fitness__fdr=none_hash=12345678")

    with pytest.raises(
        MalformedLabelError, match="requires multiple testing key 'fdr'"
    ):
        parse_label("fitness__thr=reflected_hash=12345678")

    # fitness fdr=by requires q
    with pytest.raises(MalformedLabelError, match="requires nominal level 'q'"):
        parse_label("fitness__thr=reflected_fdr=by_hash=12345678")

    # fitness fdr=none must omit q
    with pytest.raises(MalformedLabelError, match="with 'fdr=none' must omit 'q'"):
        parse_label("fitness__thr=reflected_fdr=none_q=0.05_hash=12345678")

    # bootstrap requires draws and seed
    with pytest.raises(MalformedLabelError, match="requires 'draws' and 'seed' keys"):
        parse_label("bootstrap__graph=full_draws=100_hash=12345678")

    # band tree requires bandq
    with pytest.raises(
        MalformedLabelError, match="requires persistence band quantile 'bandq'"
    ):
        parse_label("tree__graph=full_draws=100_seed=1_tree=band_hash=12345678")


def test_reject_forbidden_product_keys() -> None:
    """Product contracts forbid stage-irrelevant or inappropriate keys."""
    # fitness cannot have tree or bandq
    with pytest.raises(MalformedLabelError, match="must not contain key 'tree'"):
        parse_label("fitness__thr=reflected_fdr=none_tree=nominal_hash=12345678")

    # landscape cannot combine graph=full with thr
    with pytest.raises(MalformedLabelError, match="cannot combine 'graph=full'"):
        parse_label("landscape__graph=full_thr=reflected_fdr=none_hash=12345678")

    # bootstrap cannot contain tree or bandq
    with pytest.raises(MalformedLabelError, match="draw identity only"):
        parse_label("bootstrap__graph=full_draws=100_seed=1_tree=band_hash=12345678")

    # nominal tree cannot contain draws or bandq
    with pytest.raises(MalformedLabelError, match="must not contain key 'draws'"):
        parse_label("tree__graph=full_draws=100_tree=nominal_hash=12345678")

    # style on scientific data products
    with pytest.raises(
        MalformedLabelError, match="not permitted on scientific data product"
    ):
        parse_label("tree__graph=full_tree=nominal_hash=12345678_style=annotated")


def test_reject_unpaired_run_fields() -> None:
    """host and run must both be present if either is specified."""
    with pytest.raises(MalformedLabelError, match="must both be present"):
        parse_label("fitness__thr=reflected_fdr=none_host=local_hash=12345678")

    with pytest.raises(MalformedLabelError, match="must both be present"):
        parse_label("fitness__thr=reflected_fdr=none_run=51756602_hash=12345678")


def test_reject_invalid_hash() -> None:
    """hash must be exactly 8 lowercase hex characters."""
    with pytest.raises(MalformedLabelError, match="exactly 8 lowercase hexadecimal"):
        parse_label("tree__graph=full_tree=nominal_hash=1234")

    with pytest.raises(MalformedLabelError, match="exactly 8 lowercase hexadecimal"):
        parse_label("tree__graph=full_tree=nominal_hash=1234567Z")


def test_mixture_threshold_requires_p_and_forbidden_otherwise() -> None:
    """thr=mixture requires p=; p is forbidden for thr=published|reflected (L1)."""
    # thr=mixture with p is valid
    mix_valid = "fitness__thr=mixture_fdr=by_q=0.05_p=0.95_hash=12345678"
    parsed = parse_label(mix_valid)
    assert parsed.params["p"] == 0.95
    assert parsed.format() == mix_valid

    # thr=mixture without p is rejected
    with pytest.raises(MalformedLabelError, match="requires mixture probability 'p'"):
        parse_label("fitness__thr=mixture_fdr=by_q=0.05_hash=12345678")

    # thr=reflected with p is rejected
    with pytest.raises(
        MalformedLabelError, match="Parameter 'p' is forbidden when thr='reflected'"
    ):
        parse_label("fitness__thr=reflected_fdr=by_q=0.05_p=0.95_hash=12345678")

    # thr=published with p is rejected
    with pytest.raises(
        MalformedLabelError, match="Parameter 'p' is forbidden when thr='published'"
    ):
        parse_label("fitness__thr=published_fdr=none_p=0.95_hash=12345678")

    # graph=full with p is rejected
    with pytest.raises(MalformedLabelError, match="cannot combine 'graph=full'"):
        parse_label("tree__graph=full_p=0.95_tree=nominal_hash=12345678")


def test_float_formatting_fixed_point_decimal_no_scientific_notation() -> None:
    """format label floats in fixed-point decimal (shortest round-trip, no scientific notation) (L3)."""
    from dubanevics2026topological.results.labels import format_label

    # 1e-05 must format as 0.00001
    label = format_label(
        "fitness",
        {"thr": "reflected", "fdr": "by", "q": 1e-05, "hash": "12345678"},
    )
    assert label == "fitness__thr=reflected_fdr=by_q=0.00001_hash=12345678"
    assert parse_label(label).params["q"] == 1e-05

    # 1e-06 must format as 0.000001
    band_label = format_label(
        "tree",
        {
            "graph": "full",
            "draws": 100,
            "seed": 1,
            "tree": "band",
            "bandq": 1e-06,
            "hash": "12345678",
        },
    )
    assert (
        band_label
        == "tree__graph=full_draws=100_seed=1_tree=band_bandq=0.000001_hash=12345678"
    )
    assert parse_label(band_label).params["bandq"] == 1e-06


def test_reject_non_finite_numeric_values() -> None:
    """Reject non-finite numbers (NaN, inf) in label parsing and formatting (M2)."""
    from dubanevics2026topological.results.labels import format_label

    # Nan/inf in parse_label
    with pytest.raises(MalformedLabelError, match="Non-finite numeric value"):
        parse_label("fitness__thr=reflected_fdr=by_q=nan_hash=12345678")

    with pytest.raises(MalformedLabelError, match="Non-finite numeric value"):
        parse_label("fitness__thr=reflected_fdr=by_q=inf_hash=12345678")

    with pytest.raises(MalformedLabelError, match="Non-finite numeric value"):
        parse_label("fitness__thr=reflected_fdr=by_q=-inf_hash=12345678")

    # Nan/inf in format_label
    with pytest.raises(MalformedLabelError, match="Non-finite float value"):
        format_label(
            "fitness",
            {
                "thr": "reflected",
                "fdr": "by",
                "q": float("nan"),
                "hash": "12345678",
            },
        )

    with pytest.raises(MalformedLabelError, match="Non-finite float value"):
        format_label(
            "fitness",
            {
                "thr": "reflected",
                "fdr": "by",
                "q": float("inf"),
                "hash": "12345678",
            },
        )
