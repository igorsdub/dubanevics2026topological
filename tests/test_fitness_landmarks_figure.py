"""The fitness-landmarks figure renders from fake stage outputs; registry wiring."""

from __future__ import annotations

import tomllib
from pathlib import Path

import altair as alt
import polars as pl
import pytest
import vl_convert as vlc
from _figure_labels import LANDMARKS, SENTINEL, label_quantities, spec_text
from _saddle_band import DECLARATION, JOIN_BELOW, _population, isolated_graph

from dubanevics2026topological.analysis.leading_saddle_band import (
    compute_leading_saddle_band,
)
from dubanevics2026topological.figures import registry
from dubanevics2026topological.figures.fitness_landmarks import (
    FIGURE_ID,
    build_fitness_landmarks_figure,
    histogram_table,
    label_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _settings() -> dict:
    config = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))
    return {"figures": config}


def _payload() -> dict:
    stage = compute_leading_saddle_band(
        JOIN_BELOW,
        _population(),
        graph=isolated_graph(),
        published_threshold=-0.5,
        candidate_declaration=DECLARATION,
    )
    return {
        "leading_saddle_band": stage,
        "full_fitness": _population(),
        "manuscript_quantities": label_quantities(),
    }


def test_histogram_table_counts_every_value_once() -> None:
    table = histogram_table(pl.Series([0.0, 0.1, 0.9, 1.0]), 2)
    assert table["count"].to_list() == [2, 2]
    assert table["start"][0] == 0.0 and table["end"][-1] == 1.0
    with pytest.raises(ValueError):
        histogram_table(pl.Series([0.0]), 0)


def test_label_rows_are_ranked_by_fitness_with_the_registrys_values() -> None:
    payload = _payload()
    landmarks = payload["leading_saddle_band"].landmarks
    rows = label_rows(landmarks, {"median": "Median"}, payload["manuscript_quantities"])
    assert rows["row"].to_list() == list(range(landmarks.height))
    assert rows["fitness"].to_list() == sorted(rows["fitness"].to_list())
    # The number is the registry's display text as is (six decimals, not three), never
    # the table's fitness formatted here.
    assert (
        f"Median {SENTINEL['leading_saddle_band.median.fitness']}"
        in rows["label"].to_list()
    )
    for name in landmarks["landmark"].to_list():
        assert any(
            label.endswith(" " + SENTINEL[f"leading_saddle_band.{name}.fitness"])
            for label in rows["label"].to_list()
        )


def test_every_label_and_the_subtitle_come_from_the_registry() -> None:
    payload = _payload()
    spec = build_fitness_landmarks_figure(payload, _settings()).to_dict()
    texts = spec_text(spec)
    for name in LANDMARKS:
        assert SENTINEL[f"leading_saddle_band.{name}.fitness"] in texts
    labels = [
        row["label"]
        for layer in spec["layer"]
        if layer["mark"]["type"] == "text"
        for row in spec["datasets"][layer["data"]["name"]]
    ]
    assert len(labels) == len(LANDMARKS)
    # Names in words: no single-letter shorthand like "g" or "b_split".
    assert all(not label.startswith(("g ", "b_")) for label in labels)
    assert spec["title"]["subtitle"] == [
        (
            f"{SENTINEL['leading_saddle_band.total_sequence_count']} sequences · "
            "leading saddle band "
            f"{SENTINEL['leading_saddle_band.band.count']} "
            f"({SENTINEL['leading_saddle_band.band.percentage']}%) · "
            "median inside the band"
        )
    ]
    assert set(payload["manuscript_quantities"].requested_keys) == {
        *(f"leading_saddle_band.{name}.fitness" for name in LANDMARKS),
        "leading_saddle_band.total_sequence_count",
        "leading_saddle_band.band.count",
        "leading_saddle_band.band.percentage",
        "leading_saddle_band.median_inside_band",
    }


def test_the_decimals_setting_is_gone_from_the_figure_table() -> None:
    table = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))
    assert "label_decimals" not in table["figures"][FIGURE_ID]


def test_figure_has_histogram_band_rules_and_one_label_per_landmark() -> None:
    chart = build_fitness_landmarks_figure(_payload(), _settings())
    spec = chart.to_dict()
    layers = spec["layer"]
    marks = [layer["mark"]["type"] for layer in layers]
    assert marks[0] == "bar" and marks[1] == "rect" and marks[-1] == "text"
    assert marks.count("rule") == 3  # solid, dashed and dotted rules
    text = [layer for layer in layers if layer["mark"]["type"] == "text"]
    # The candidate threshold, at the right edge, is written to the left of its rule.
    assert [layer["mark"]["align"] for layer in text] == ["left", "right"]
    assert sum(len(spec["datasets"][layer["data"]["name"]]) for layer in text) == 8
    band = spec["datasets"][layers[1]["data"]["name"]][0]
    assert (band["lower"], band["upper"]) == (-0.6, -0.4)
    assert isinstance(chart, alt.LayerChart | alt.Chart)
    assert vlc.vegalite_to_svg(chart.to_dict(), vl_version="v5_21").startswith("<svg")


def test_figure_refuses_a_population_that_does_not_match_the_stage() -> None:
    payload = _payload()
    payload["full_fitness"] = payload["full_fitness"].head(5)
    with pytest.raises(ValueError, match="does not match"):
        build_fitness_landmarks_figure(payload, _settings())


def test_spec_is_registered_opt_in_and_reads_the_stage() -> None:
    spec = registry.get_spec(FIGURE_ID)
    assert spec.default is False
    assert spec.inputs == {
        "leading_saddle_band",
        "full_fitness",
        "manuscript_quantities",
    }
    assert spec.variants == {"vanilla"}
    assert registry.renderer_file(spec) == "fitness_landmarks.py"
    payload = _payload()
    chart = spec.render(payload, _settings(), "vanilla")
    assert chart is not None
