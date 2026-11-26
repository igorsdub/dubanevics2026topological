"""Supplementary S1: match, correlation and registry of papkou_agreement."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dubanevics2026topological.figures import registry
from dubanevics2026topological.figures._hexbin import HEXAGON_PATH
from dubanevics2026topological.figures.papkou_agreement import (
    FIGURE_ID,
    STEM,
    build_papkou_agreement_figure,
    correlations,
    matched_genotypes,
    LOG_TICK_LABEL_EXPR,
    LOG_TICKS,
    wild_type_point,
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
    assert spec.default is True  # Supplementary S1
    assert spec.inputs == frozenset({"growth_rates", "published_fitness"})
    assert spec.variants == frozenset({"vanilla"})
    assert spec.multi is True
    assert spec in registry.FIGURE_SPECS


def test_output_stem_is_the_canonical_name() -> None:
    assert STEM == "papkou2023_fitness_and_se_barseqfit_agreement"
    spec = registry.get_spec(FIGURE_ID)
    outputs = list(spec.render(
            {"growth_rates": _fits(), "published_fitness": _published()},
            _settings(),
            "vanilla",
        ))
    assert [output.stem for output in outputs] == [STEM]


def _settings(**own) -> dict:
    return {
        "figures": {
            "rendering": {"font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {
                "peak_uncertainty": {"panel_size": 300, "font_size": 15},
                FIGURE_ID: {"title": "t", "hexbin_min": 2500, **own},
            },
        }
    }


def test_builder_is_a_two_by_two_with_stats() -> None:
    spec = build_papkou_agreement_figure(
        {"growth_rates": _fits(), "published": _published()}, _settings()
    ).to_dict()
    # One 2-column grid, so Vega aligns every column and row.
    assert spec["columns"] == 2 and len(spec["concat"]) == 4
    top, bottom = spec["concat"][:2], spec["concat"][2:]
    assert top[0]["title"]["text"] == "Papkou et al. (2023)"
    assert top[1]["title"]["text"] == "BarSeqFit"
    assert len(top[0]["title"]["subtitle"]) == 1
    assert top[0]["title"]["subtitle"][0].startswith("n = ")
    assert "Spearman" not in str(top[0]["title"]) + str(top[1]["title"])
    assert bottom[0]["title"]["text"] == "Fitness agreement"
    assert bottom[1]["title"]["text"] == "Standard error agreement"
    stats_line, count_line = bottom[0]["title"]["subtitle"]
    assert stats_line.startswith("Pearson r = 1.000000, Spearman")
    assert count_line == "n = 2 genotypes"
    # Every y axis has the same extent, so the plot areas start at one x.
    assert spec["config"]["axisY"] == {"minExtent": 110, "maxExtent": 110}
    nodes: list = []
    _walk(spec, nodes)
    sizes = {(n["width"], n["height"]) for n in nodes if "width" in n and "height" in n}
    assert sizes == {(300, 300)}


def _cloud(n: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    fitted = rng.normal(0.0, 1.0, n)
    published = fitted * 0.6 + rng.normal(0.0, 0.4, n)
    se = np.abs(rng.normal(0.02, 0.005, n)) + 0.001
    published_se = se * 0.5 + rng.normal(0.0, 0.004, n)
    names = [f"g{i:04d}" for i in range(n)]
    growth = pl.DataFrame(
        {
            "dna_sequence": [*names, "WTWTWT"],
            "s": [*fitted, 0.0],
            "se_pooled": [*se, 0.01],
        }
    )
    published_table = pl.DataFrame(
        {
            "dna_from": ["WTWTWT"] * n,
            "dna_to": names,
            "s": published,
            "se": published_se,
        }
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
    assert len(domains) == 4
    # One common colour domain for all four panels.
    assert all(domain == domains[0] for domain in domains)
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
    assert types.count("line") == 2  # y = x in the two agreement panels only
    hexagons = [
        mark
        for mark in marks
        if isinstance(mark, dict) and mark.get("shape") == HEXAGON_PATH
    ]
    assert len(hexagons) == 4
    assert all(mark["filled"] is True for mark in hexagons)
    # A single colour bar: exactly one panel declares a legend, scales are shared.
    legends = [
        node["legend"]
        for node in nodes
        if node.get("field") == "n_genotypes" and node.get("legend")
    ]
    assert len(legends) == 1 and legends[0]["orient"] == "top-right"
    assert legends[0]["labelExpr"] == LOG_TICK_LABEL_EXPR and legends[0]["gradientLength"] == 140
    assert legends[0]["labelOverlap"] is False
    # Logarithmic common scale from 1, with decade ticks.
    colour = [
        node
        for node in nodes
        if node.get("field") == "n_genotypes" and "scale" in node
    ]
    assert all(node["scale"]["type"] == "log" for node in colour)
    assert all(node["scale"]["domain"][0] == 1 for node in colour)
    ticks = next(n["legend"]["values"] for n in colour if n.get("legend"))
    assert ticks == [tick for tick in LOG_TICKS if tick <= colour[0]["scale"]["domain"][1]]
    assert ticks[0] == 1
    assert ticks[-1] <= colour[0]["scale"]["domain"][1]
    # The wild type: one red star, in the BarSeqFit panel only (top right).
    panels = spec["concat"]
    assert len(panels) == 4

    def is_star(layer: dict) -> bool:
        mark = layer["mark"]
        return (
            isinstance(mark, dict)
            and mark.get("type") == "point"
            and mark.get("color") == "#d62728"
            and "field" in layer["encoding"].get("x", {})
        )

    stars = [[layer for layer in panel["layer"] if is_star(layer)] for panel in panels]
    assert [len(found) for found in stars] == [0, 1, 0, 0]
    assert stars[1][0]["mark"]["filled"] is True
    # A "Wild type" entry under the colour bar, in the BarSeqFit panel only.
    entries = [
        [
            layer
            for layer in panel["layer"]
            if isinstance(layer["mark"], dict)
            and layer["mark"].get("type") == "text"
            and layer["encoding"]["text"].get("value") == "Wild type"
        ]
        for panel in panels
    ]
    assert [len(found) for found in entries] == [0, 1, 0, 0]


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


def test_wild_type_point_is_the_growth_rate_row_or_none() -> None:
    growth = pl.DataFrame(
        {
            "dna_sequence": ["AAA", "AAC", "ACA"],
            "s": [0.01, 0.2, 0.4],
            "se_pooled": [0.003, 0.01, 0.05],
        }
    )
    published = pl.DataFrame(
        {
            "dna_from": ["AAA", "AAA"],
            "dna_to": ["AAC", "ACA"],
            "s": [0.2, 0.3],
            "se": [0.01, 0.04],
        }
    )
    assert wild_type_point(growth, published) == (0.003, 0.01)
    spec = build_papkou_agreement_figure(
        {"growth_rates": growth, "published": published}, _settings()
    ).to_dict()
    points = []
    for table in spec["datasets"].values():
        points += [
            row for row in table if row.get("se_pooled") == 0.003 and row.get("fitted_s") == 0.01
        ]
    assert len(points) == 1  # BarSeqFit panel only; no invented published values
    assert (points[0]["se_pooled"], points[0]["fitted_s"]) == (0.003, 0.01)


def test_missing_wild_type_draws_no_star_and_does_not_raise() -> None:
    growth = _fits().filter(pl.col("dna_sequence") != "AAA")
    published = _published()
    assert wild_type_point(growth, published) is None
    spec = build_papkou_agreement_figure(
        {"growth_rates": growth, "published": published}, _settings()
    ).to_dict()
    nodes: list = []
    _walk(spec, nodes)
    assert not [n for n in nodes if n.get("value") == "Wild type"]


def test_the_matched_cloud_keeps_the_wild_type_genotype() -> None:
    """Statistics equal the merge-base figure's: no genotype is dropped for being WT."""
    published = pl.DataFrame(
        {
            "dna_from": ["AAA"] * 3,
            "dna_to": ["AAC", "ACA", "AAA"],
            "s": [0.2, 0.3, 0.0],
            "se": [0.01, 0.04, 0.02],
        }
    )
    spec = build_papkou_agreement_figure(
        {"growth_rates": _fits(), "published": published}, _settings()
    ).to_dict()
    assert "n = 3 genotypes" in spec["concat"][2]["title"]["subtitle"][1]


def test_barseqfit_count_adds_the_wild_type_row_and_papkou_keeps_the_matched_set() -> None:
    growth = pl.DataFrame(
        {
            "dna_sequence": ["AAA", "AAC", "ACA", "ACC"],
            "s": [0.01, 0.2, 0.4, 0.5],
            "se_pooled": [0.003, 0.01, 0.05, 0.06],
        }
    )
    published = pl.DataFrame(
        {
            "dna_from": ["AAA"] * 3,
            "dna_to": ["AAC", "ACA", "ACC"],
            "s": [0.2, 0.3, 0.45],
            "se": [0.01, 0.04, 0.05],
        }
    )
    spec = build_papkou_agreement_figure(
        {"growth_rates": growth, "published": published}, _settings()
    ).to_dict()
    top = spec["concat"][:2]
    assert top[0]["title"]["subtitle"] == ["n = 3 genotypes"]
    assert top[1]["title"]["subtitle"] == ["n = 4 genotypes"]
    assert spec["concat"][2]["title"]["subtitle"][1] == "n = 3 genotypes"
    # Without the wild type both panels count the matched set.
    absent = build_papkou_agreement_figure(
        {"growth_rates": growth.filter(pl.col("dna_sequence") != "AAA"), "published": published},
        _settings(),
    ).to_dict()
    assert [c["title"]["subtitle"] for c in absent["concat"][:2]] == [["n = 3 genotypes"]] * 2
