"""``ridge_entropy``: cells, panels, styles and palette colours."""

from __future__ import annotations

import itertools
import json

import polars as pl
import pytest

from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.analysis.ridge_entropy import build_ridge_entropy
from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures.view_naming import single_result_figure_name
from dubanevics2026topological.figures.ridge_entropy_view import (
    build_ridge_entropy_figure,
    entropy_cells,
)

DNA = ["AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG", "AAAAAAAAA", "AAGGAAAAA"]
FITNESS = [1.0, 0.8, 0.4, 0.1, 0.9]
RIDGE = ["kem", "tcl", None, None, "kem"]


def _entropy():
    return build_ridge_entropy(
        pl.DataFrame(
            {
                "member_dna_sequence": DNA,
                "raw_fitness": FITNESS,
                "ridge_name": RIDGE,
            },
            schema={
                "member_dna_sequence": pl.String,
                "raw_fitness": pl.Float64,
                "ridge_name": pl.String,
            },
        ),
        sweep_points=5,
    )


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _sweep(entropy, name):
    return entropy.sweeps.filter(pl.col("population") == name)


def test_cells_tile_the_population_span_without_gaps() -> None:
    entropy = _entropy()
    low, high = entropy.sidecar["fitness_range"]["kem"]
    cells = [
        cell
        for cell in entropy_cells(_sweep(entropy, "kem"), low, high)
        if cell["position"] == "1"
    ]
    assert cells[0]["lower"] == low
    assert cells[-1]["upper"] == high
    for before, after in itertools.pairwise(cells):
        assert before["upper"] == after["lower"]


def test_cells_outside_the_span_or_null_draw_nothing() -> None:
    entropy = _entropy()
    low, high = entropy.sidecar["fitness_range"]["tcl"]
    cells = entropy_cells(_sweep(entropy, "tcl"), low, high)
    assert cells
    assert all(low <= cell["threshold"] <= high for cell in cells)
    assert all(cell["entropy_bits"] is not None for cell in cells)


@pytest.mark.parametrize("style", ["annotated", "vanilla"])
def test_figure_has_core_kem_and_tcl_panels(settings, style) -> None:
    chart = build_ridge_entropy_figure(
        _entropy(), {"kem": 2, "tcl": 1}, settings, style, title="t", subtitle="s"
    )
    spec = chart.to_dict()
    assert len(spec["hconcat"]) == 3
    text = json.dumps(spec)
    assert ("functional" in text) is (style == "annotated")
    assert "KEM branch" in text and "TCL branch" in text


def test_unknown_style_is_refused(settings) -> None:
    with pytest.raises(ValueError, match="style"):
        build_ridge_entropy_figure(
            _entropy(), {}, settings, "fancy", title="t", subtitle="s"
        )


def test_an_absent_ridge_is_a_grey_panel(settings) -> None:
    members = pl.DataFrame(
        {
            "member_dna_sequence": DNA[:3],
            "raw_fitness": FITNESS[:3],
            "ridge_name": ["kem", None, None],
        },
        schema={
            "member_dna_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "ridge_name": pl.String,
        },
    )
    entropy = build_ridge_entropy(members, sweep_points=5)
    chart = build_ridge_entropy_figure(
        entropy, {"kem": 1}, settings, "annotated", title="t", subtitle="s"
    )
    assert "absent" in json.dumps(chart.to_dict())


def test_outside_the_span_is_left_blank(settings) -> None:
    for style in ("vanilla", "annotated"):
        text = json.dumps(
            build_ridge_entropy_figure(
                _entropy(), {}, settings, style, title="t", subtitle="s"
            ).to_dict()
        )
        assert "#f5f5f5" not in text and "outside" not in text.lower()
    assert not hasattr(palette, "OUTSIDE_GREY")
    assert "#" not in json.dumps(settings["figures"]["d10"]["ridge_entropy"])


def test_names_use_the_band_key_set() -> None:
    params = {
        "thr": "published",
        "fdr": "by",
        "q": 0.05,
        "draws": 10000,
        "seed": 20261019,
        "tree": "band",
        "bandq": 0.95,
        "hash": "5ce7993a",
    }
    name = single_result_figure_name("ridge_entropy", params, "annotated", "png")
    assert name.startswith("ridge_entropy__thr=published_fdr=by_q=0.05_draws=10000")
    assert name.endswith("_hash=5ce7993a_style=annotated.png")
