"""Tests for canonical JSON, scientific hash calculation, and exclusion rules per ADR 0016."""

from __future__ import annotations

import json

import pytest

from dubanevics2026topological.results.hashing import (
    canonical_json,
    compute_scientific_hash,
)


def test_canonical_json_recursive_sorting_and_compact_separators() -> None:
    """Canonical JSON must sort keys recursively at all levels and use (',', ':') with no spaces."""
    doc = {
        "z": 1,
        "a": {
            "d": 4,
            "b": 2,
            "c": [3, {"y": 20, "x": 10}],
        },
    }
    canon = canonical_json(doc)
    # Check key ordering
    assert canon == '{"a":{"b":2,"c":[3,{"x":10,"y":20}],"d":4},"z":1}'
    # Must not contain space after comma or colon
    assert ", " not in canon
    assert ": " not in canon


def test_canonical_json_float_shortest_round_trip() -> None:
    """Float representation in canonical JSON must use shortest round-trip representation."""
    doc = {
        "q": 0.05,
        "thr": -0.5077739708744209,
        "zero": -0.0,
    }
    canon = canonical_json(doc)
    parsed = json.loads(canon)
    assert parsed["q"] == 0.05
    assert parsed["thr"] == -0.5077739708744209
    assert parsed["zero"] == 0.0
    assert '"thr":-0.5077739708744209' in canon

    # Non-finite floats (NaN, inf) must be rejected
    with pytest.raises(ValueError, match="Non-finite float value"):
        canonical_json({"q": float("nan")})

    with pytest.raises(ValueError, match="Non-finite float value"):
        canonical_json({"q": float("inf")})

    with pytest.raises(ValueError, match="Non-finite float value"):
        canonical_json({"q": float("-inf")})


def test_golden_value_scientific_hash() -> None:
    """Golden-value test with hardcoded literal constants to detect regressions."""
    product_type = "bootstrap"
    scientific_parameters = {
        "draws": 10000,
        "fdr": "by",
        "q": 0.05,
        "seed": 20261019,
        "thr": "reflected",
    }
    parent_scientific_ids = [
        "1111111122222222333333334444444455555555666666667777777788888888"
    ]
    input_data_sha256s = {
        "papkou2023_fitness_diff.parquet": (
            "44fcd723d360a73df9187fc66e66d2d28d0b9b5985984d0be8f0d56e3c8ffd5d"
        ),
        "papkou2023_fitness_wt.parquet": (
            "a3f3fd421b7ad7d05c7ae870c190f2d0b5a4e998bfffb3b8f76652a37882e4b4"
        ),
    }

    full_hash, short_hash = compute_scientific_hash(
        product_type=product_type,
        scientific_parameters=scientific_parameters,
        parent_scientific_ids=parent_scientific_ids,
        input_data_sha256s=input_data_sha256s,
    )

    # Assert literal 64-hex and 8-hex constants (L4)
    expected_literal_full = (
        "c15846de4b6cfb1e44cca87e5c36b8e23ca19af4549d4109c30080da2f59c972"
    )
    expected_literal_short = "c15846de"

    assert full_hash == expected_literal_full
    assert short_hash == expected_literal_short
    assert len(full_hash) == 64
    assert len(short_hash) == 8


def test_exclusion_invariance() -> None:
    """Run execution fields, timestamps, environment, and storage are excluded from hash."""
    base_params = {
        "thr": "reflected",
        "fdr": "by",
        "q": 0.05,
    }
    base_full, base_short = compute_scientific_hash(
        product_type="fitness",
        scientific_parameters=base_params,
    )

    # Add run fields, git stamp, environment, timestamps to parameters
    contaminated_params = dict(base_params)
    contaminated_params.update(
        {
            "host": "local",
            "run": 51756602,
            "created_at": "2026-10-27T14:00:00Z",
            "code_provenance": {"commit": "abc1234", "dirty": False},
            "environment": {"python_version": "3.13.0"},
            "storage_references": {"bucket_uri": "gs://store/results/123"},
            "artifacts": {"table.parquet": {"sha256": "1234"}},
        }
    )

    new_full, new_short = compute_scientific_hash(
        product_type="fitness",
        scientific_parameters=contaminated_params,
    )

    # Hashes must remain bit-identical
    assert new_full == base_full
    assert new_short == base_short


def test_hash_sensitivity_to_scientific_changes() -> None:
    """Any change to a scientific parameter alters the hash."""
    base_full, _ = compute_scientific_hash(
        product_type="fitness",
        scientific_parameters={"thr": "reflected", "fdr": "by", "q": 0.05},
    )

    diff_q, _ = compute_scientific_hash(
        product_type="fitness",
        scientific_parameters={"thr": "reflected", "fdr": "by", "q": 0.01},
    )
    assert diff_q != base_full

    diff_type, _ = compute_scientific_hash(
        product_type="landscape",
        scientific_parameters={"thr": "reflected", "fdr": "by", "q": 0.05},
    )
    assert diff_type != base_full

    diff_parent, _ = compute_scientific_hash(
        product_type="fitness",
        scientific_parameters={"thr": "reflected", "fdr": "by", "q": 0.05},
        parent_scientific_ids=[
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        ],
    )
    assert diff_parent != base_full
