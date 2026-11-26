"""Shared palette and top-histogram helpers (Tree 1.3, bucket 1)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures._top_histogram import (
    count_axis_limit,
    top_histogram_bins,
    top_histogram_chart,
)

ROOT = Path(__file__).resolve().parents[1]


def test_horton_strahler_colors_run_from_light_lavender_to_near_black_purple() -> None:
    colors = palette.HORTON_STRAHLER_COLORS
    assert len(colors) == 4
    rgb = [
        [int(color[i : i + 2], 16) for i in (1, 3, 5)] for color in colors
    ]
    for red, green, blue in rgb:
        # Purple: blue dominates.
        assert blue > max(red, green)
    lightness = [sum(channels) for channels in rgb]
    assert lightness == sorted(lightness, reverse=True)
    assert len(set(colors)) == 4


def test_configured_horton_strahler_colors_are_the_palette() -> None:
    figures = tomllib.loads((ROOT / "config" / "figures.toml").read_text())
    metrics = figures["figures"]["study_tree_persistence_metrics"]
    assert tuple(metrics["horton_strahler_colors"]) == palette.HORTON_STRAHLER_COLORS


def test_vertex_count_palette_is_black_and_white_only() -> None:
    assert palette.VERTEX_COUNT_GREY == "#000000"
    for color in (palette.VERTEX_COUNT_GREY, palette.HISTOGRAM_GREY):
        assert len({color[1:3], color[3:5], color[5:7]}) == 1
    figures = tomllib.loads((ROOT / "config" / "figures.toml").read_text())
    assert figures["rendering"]["count_color"] == palette.VERTEX_COUNT_GREY
    assert figures["rendering"]["top_histogram_bins"] == palette.TOP_HISTOGRAM_BINS
    assert tuple(figures["rendering"]["top_histogram_count_ticks"]) == (
        palette.TOP_HISTOGRAM_COUNT_TICKS
    )
    assert figures["rendering"]["top_histogram_headroom"] == palette.TOP_HISTOGRAM_HEADROOM


def test_no_plasma_scheme_in_figure_sources_or_config() -> None:
    hits = [
        path.name
        for path in [
            *(ROOT / "src" / "dubanevics2026topological" / "figures").glob("*.py"),
            ROOT / "config" / "figures.toml",
        ]
        if re.search(r"scheme\s*=\s*[\"']plasma", path.read_text())
    ]
    assert hits == []


def test_bins_default_to_200_and_count_every_value() -> None:
    fitness = pl.Series([-1.0, 0.0, 0.5, 0.5, 1.0])
    table = top_histogram_bins(fitness, (-1.2, 1.4))
    assert table.height == 200
    assert int(table["count"].sum()) == fitness.len()


def test_axis_limit_is_headroom_over_the_tallest_shared_bin() -> None:
    assert count_axis_limit([20_209, 100], {}) == pytest.approx(1.05 * 20_209)


def test_axis_limit_never_drops_below_the_last_tick() -> None:
    assert count_axis_limit([500], {}) == 20_000
    assert count_axis_limit([], {}) == 20_000


def test_axis_limit_reads_headroom_and_ticks_from_the_rendering() -> None:
    rendering = {"top_histogram_headroom": 1.1, "top_histogram_count_ticks": [0, 100]}
    assert count_axis_limit([1_000], rendering) == pytest.approx(1_100)


@pytest.mark.parametrize("hidden", [False, True])
def test_chart_clips_nothing_and_keeps_two_ticks(hidden) -> None:
    fitness = pl.Series([0.1] * 50_000 + [0.9] * 5)
    chart = top_histogram_chart(
        fitness, (-1.2, 1.4), {}, width=250, height=70, hidden=hidden
    )
    spec = chart.to_dict()
    y = spec["encoding"]["y"]
    assert y["scale"]["domain"] == [0, pytest.approx(52_500)]
    assert y["axis"]["values"] == [0, 20_000]
    data = pl.DataFrame(spec["datasets"][spec["data"]["name"]])
    assert data.height == 200
    assert data["count"].max() == 50_000


def test_shared_count_limit_is_used_when_given() -> None:
    fitness = pl.Series([0.1] * 100)
    chart = top_histogram_chart(
        fitness, (-1.2, 1.4), {}, width=250, height=70, count_limit=30_000
    )
    assert chart.to_dict()["encoding"]["y"]["scale"]["domain"] == [0, 30_000]


def test_fixed_count_axis_ticks_are_the_configured_ones() -> None:
    from dubanevics2026topological.figures._charts import fixed_count_axis

    axis = fixed_count_axis((0, 20_000), title="Vertex count").to_dict()
    assert axis["values"] == [0, 20_000]


def test_tree_histogram_axis_has_headroom_over_a_tall_bin() -> None:
    from dubanevics2026topological.figures._charts import histogram_chart

    population = pl.DataFrame({"scalar": [0.1] * 20_209 + [0.9] * 5})
    chart = histogram_chart(
        population, None, (-1.2, 1.4), None, {}, {}, count_axis=None
    )
    spec = chart.to_dict(validate=False)
    assert spec["encoding"]["x"]["scale"]["domain"] == [0.0, pytest.approx(1.05 * 20_209)]


def test_count_axis_limit_rejects_empty_ticks() -> None:
    import pytest

    from dubanevics2026topological.figures._top_histogram import count_axis_limit

    with pytest.raises(ValueError, match="top_histogram_count_ticks"):
        count_axis_limit([3], {"top_histogram_count_ticks": []})
