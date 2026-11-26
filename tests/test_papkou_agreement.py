"""Supplementary S1: match, correlation and registry of papkou_agreement."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dubanevics2026topological.figures import registry
from dubanevics2026topological.figures._hexbin import HEXAGON_PATH
from dubanevics2026topological.figures.papkou_agreement import (
    FIGURE_ID,
    build_papkou_agreement_figure,
    correlations,
    matched_genotypes,
)


def _fits() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dna_sequence": ["AAC", "ACA", "AGA", "AAA"],
            "s": [0.2, 0.4, 0.1, 0.0],
            "se_pooled": [0.01, 0.05, 0.02, 0.03],
        }
    )


def _published() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dna_from": ["AAA", "AAA", "AAA"],
            "dna_to": ["AAC", "ACA", "TAA"],
            "s": [0.2, 0.3, -0.5],
            "se": [0.01, 0.04, 0.08],
        }
    )


def test_match_is_an_inner_join_on_dna_sequence() -> None:
    matched = matched_genotypes(_fits(), _published())
    assert matched["dna_sequence"].to_list() == ["AAC", "ACA"]
    assert matched["fitted_s"].to_list() == [0.2, 0.4]
    assert matched["published_m"].to_list() == [0.2, 0.3]
    assert matched["se_pooled"].to_list() == [0.01, 0.05]
    assert matched["published_m_se"].to_list() == [0.01, 0.04]


def test_duplicate_genotype_fails() -> None:
    published = pl.concat([_published(), _published().head(1)])
    with pytest.raises(ValueError, match="Duplicate published"):
        matched_genotypes(_fits(), published)


def test_correlations_on_a_known_sample() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 5.0])
    stats = correlations(x, y)
    assert stats["n"] == 4
    assert stats["pearson"] == pytest.approx(0.9827076298239908)
    assert stats["spearman"] == pytest.approx(1.0)
    perfect = correlations(x, x)
    assert perfect["pearson"] == pytest.approx(1.0)
    assert perfect["spearman"] == pytest.approx(1.0)


def test_correlation_rejects_a_short_sample() -> None:
    with pytest.raises(ValueError, match="at least two"):
        correlations(np.array([1.0]), np.array([1.0]))


def test_registry_hooks_the_figure() -> None:
    spec = registry.get_spec(FIGURE_ID)
    assert spec.figure_id == "papkou_agreement"
    assert spec.default is False
    assert spec.inputs == frozenset({"growth_rates", "published_fitness"})
    assert spec.variants == frozenset({"vanilla"})
    assert spec in registry.FIGURE_SPECS


def test_builder_reports_correlation_and_n() -> None:
    settings = {
        "figures": {
            "rendering": {"font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {
                FIGURE_ID: {
                    "title": "Agreement with Papkou et al. 2023",
                    "hexbin_min": 2500,
                }
            },
        }
    }
    chart = build_papkou_agreement_figure(
        {"growth_rates": _fits(), "published": _published()}, settings
    )
    spec = chart.to_dict()
    subtitle = spec["title"]["subtitle"]
    assert subtitle[0].startswith("A  Pearson r = 1.000000, Spearman")
    assert "n = 2 genotypes" in subtitle[0]
    assert "n = 2 genotypes" in subtitle[1]
    panels = spec["hconcat"]
    assert panels[0]["title"]["text"] == "A  Fitness"
    assert "n = 2 genotypes" in panels[0]["title"]["subtitle"][0]
    assert panels[1]["title"]["text"] == "B  Standard error"
    assert "Pearson r = 1.000000" in subtitle[0]


def _cloud(n: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    fitted = rng.normal(0.0, 1.0, n)
    published = fitted * 0.6 + rng.normal(0.0, 0.4, n)
    se = np.abs(rng.normal(0.02, 0.005, n)) + 0.001
    published_se = se * 0.5 + rng.normal(0.0, 0.004, n)
    names = [f"g{i:04d}" for i in range(n)]
    growth = pl.DataFrame({"dna_sequence": names, "s": fitted, "se_pooled": se})
    published_table = pl.DataFrame(
        {"dna_to": names, "s": published, "se": published_se}
    )
    return growth, published_table


def _walk(node, found: list) -> None:
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            _walk(value, found)
    elif isinstance(node, list):
        for value in node:
            _walk(value, found)


def test_binned_density_has_marks_and_a_finite_color_domain() -> None:
    """A panel past hexbin_min must keep its bin rows and a finite legend."""
    growth, published = _cloud(80, seed=7)
    settings = {
        "figures": {
            "rendering": {"font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {
                FIGURE_ID: {
                    "title": "Agreement with Papkou et al. 2023",
                    "hexbin_min": 10,
                    "hexbin_bins": 8,
                    "panel_width": 360,
                    "panel_height": 360,
                    "point_size": 36,
                    "point_color": "#252525",
                }
            },
        }
    }
    chart = build_papkou_agreement_figure(
        {"growth_rates": growth, "published": published}, settings
    )
    spec = chart.to_dict()
    rows = [
        table
        for table in spec["datasets"].values()
        if table and "n_genotypes" in table[0]
    ]
    assert rows, "binned panels embedded no genotype counts"
    assert all(row["n_genotypes"] >= 1 for table in rows for row in table)
    nodes: list = []
    _walk(spec, nodes)
    domains = [
        node["scale"]["domain"]
        for node in nodes
        if node.get("field") == "n_genotypes" and "scale" in node
    ]
    assert len(domains) == 2
    for domain in domains:
        assert len(domain) == 2
        assert all(np.isfinite(value) for value in domain)
        assert domain[0] < domain[1]
    # The density is hexagons (a custom-path point mark), not rectangles; sparse bins are
    # drawn as circles as well.
    marks = [node["mark"] for node in nodes if "mark" in node]
    types = [mark["type"] if isinstance(mark, dict) else mark for mark in marks]
    assert "rect" not in types
    assert "circle" in types
    assert "line" in types
    hexagons = [
        mark
        for mark in marks
        if isinstance(mark, dict) and mark.get("shape") == HEXAGON_PATH
    ]
    assert len(hexagons) == 2
    assert all(mark["filled"] is True for mark in hexagons)


def test_each_axis_is_cropped_to_its_own_data_extent() -> None:
    """No shared square: x and y domains hug their own data (a 2 % pad)."""
    growth, published = _cloud(80, seed=7)
    settings = {
        "figures": {
            "rendering": {"font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {
                FIGURE_ID: {"hexbin_min": 10, "hexbin_bins": 8, "title": "t"}
            },
        }
    }
    spec = build_papkou_agreement_figure(
        {"growth_rates": growth, "published": published}, settings
    ).to_dict()
    nodes: list = []
    _walk(spec, nodes)
    domains = {
        node["field"]: node["scale"]["domain"]
        for node in nodes
        if node.get("field") in {"fitted_s", "published_m"}
        and "scale" in node
        and "domain" in node["scale"]
    }
    matched = matched_genotypes(growth, published)
    for field in ("fitted_s", "published_m"):
        values = matched[field].to_numpy()
        low, high = domains[field]
        assert low < values.min() and high > values.max()
        assert high - low < (values.max() - values.min()) * 1.1
    # The two ranges differ, so the panel is not forced square.
    assert domains["fitted_s"] != domains["published_m"]
