from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from _figure_labels import SENTINEL, label_quantities, scan_values, spec_text

from dubanevics2026topological.analysis import (
    bootstrap_band_quantile_scan as scan_module,
)
from dubanevics2026topological.figures import bootstrap_band_quantile_scan as fig
from dubanevics2026topological.simplification import bootstrap_smoothed_tree as stage

PERSISTENCE = [0.2, 0.5, 0.65, 0.9, 1.5]
SCAN = pl.DataFrame(
    {
        "q": [0.0, 0.5, 0.95, 1.0],
        "c": [0.1, 0.3, 0.45, 0.5],
        "surviving_peaks": [4, 3, 1, 1],
    }
)


def _payload(scan=SCAN):
    cached = SimpleNamespace(
        summary={"settings": {"band_quantile": 0.95, "n_replicates": 1000, "seed": 7}},
        hills=pl.DataFrame({"nominal_persistence": PERSISTENCE}),
    )
    return {
        "bootstrap_smoothed": cached,
        "band_quantile_scan": scan,
        "survival": scan_module.survival_curve(PERSISTENCE),
        "manuscript_quantities": label_quantities(scan_values(scan)),
    }


def _settings(**extra):
    return {"figures": {"figures": {fig.FIGURE_ID: dict(extra)}}}


def _spec(**extra):
    chart = fig.build_bootstrap_band_quantile_scan_figure(
        _payload(), _settings(**extra)
    )
    return chart.to_dict()


def _layer(spec, kind):
    return next(
        layer
        for layer in spec["layer"]
        if (layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"])
        == kind
    )


def _data(spec, layer):
    return spec["datasets"][layer["data"]["name"]][0]


def test_step_after_line_shading_and_marker() -> None:
    spec = _spec()
    assert _layer(spec, "line")["mark"]["interpolate"] == "step-after"
    rect = _layer(spec, "rect")
    data = _data(spec, rect)
    assert data["tau"] == pytest.approx(0.2) and data["tau_end"] == pytest.approx(1.0)
    assert rect["mark"]["clip"] is True
    rule_data = _data(spec, _layer(spec, "rule"))
    assert rule_data["tau"] == pytest.approx(0.9)
    assert rule_data["surviving_peaks"] == 1
    # The marker names the registry's numbers, never the scan's own (0.95, 0.9, 1).
    assert rule_data["label"] == (
        f"{SENTINEL['bootstrap.band_quantile']} quantile: "
        f"{SENTINEL['bootstrap.persistence_threshold.value']}, "
        f"{SENTINEL['bootstrap.candidate.after_band_rule.peak_count']} peaks"
    )


def test_x_domain_reaches_tau_high_and_total_is_the_hill_count() -> None:
    line = _layer(_spec(tau_axis_max=0.4), "line")
    assert line["encoding"]["x"]["scale"]["domain"] == [0, 1.0]
    assert line["encoding"]["y"]["scale"]["domain"] == [0, len(PERSISTENCE)]


def test_top_axis_ticks_interpolate_the_scan() -> None:
    ticks = fig.top_axis_ticks(SCAN, [0.0, 0.25, 1.0])
    assert ticks["tau"].to_list() == pytest.approx([0.2, 0.4, 1.0])
    assert ticks["label"].to_list() == ["0", "0.25", "1"]


def test_top_axis_tick_marks_are_drawn_above_the_plot_not_clipped() -> None:
    # A clipped tick sits outside the plot area and is not drawn at all.
    spec = _spec(top_axis_quantiles=[0.0, 0.5, 1.0], top_tick_length=6)
    tick = _layer(spec, "tick")
    assert tick["mark"]["clip"] is False
    assert tick["mark"]["size"] == 6
    assert tick["mark"]["yOffset"] == -3  # the mark spans 0 to 6 px above the edge
    assert tick["encoding"]["y"] == {"value": 0}  # the shading's top edge
    assert [row["tau"] for row in spec["datasets"][tick["data"]["name"]]] == (
        pytest.approx([0.2, 0.6, 1.0])
    )
    assert [row["label"] for row in spec["datasets"][tick["data"]["name"]]] == [
        "0",
        "0.5",
        "1",
    ]


def test_top_axis_ticks_range_and_grid_errors() -> None:
    with pytest.raises(ValueError, match="within the scanned range"):
        fig.top_axis_ticks(SCAN, [1.5])
    with pytest.raises(ValueError, match="within the scanned range"):
        fig.top_axis_ticks(SCAN, [-0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        fig.top_axis_ticks(SCAN.reverse(), [0.5])
    with pytest.raises(ValueError, match="non-decreasing"):
        fig.top_axis_ticks(SCAN.with_columns(pl.col("c").reverse()), [0.5])


def test_text_names_the_quantities_in_words_and_never_half_band() -> None:
    text = json.dumps(_spec(), ensure_ascii=False)
    assert "half band" not in text.lower()
    # The subtitle wraps by width, so a phrase can break across two lines: compare the
    # subtitle lines joined as one sentence.
    spec = _spec()
    joined = " ".join(spec["title"]["subtitle"])
    assert "bootstrapped persistence threshold" in joined
    assert "band quantile" in text
    for symbol in ("τ", "tau =", "tau(q)", "2c", "q = "):
        assert symbol not in text


def test_the_subtitle_prints_the_registry_numbers_as_given() -> None:
    quantities = label_quantities(scan_values(SCAN))
    chart = fig.build_bootstrap_band_quantile_scan_figure(
        {**_payload(), "manuscript_quantities": quantities}, _settings()
    ).to_dict()
    text = spec_text(chart)
    for key in (
        "bootstrap.draws",
        "bootstrap.seed",
        "bootstrap.persistence_threshold.value",
        "bootstrap.band_quantile",
        "bootstrap.candidate.after_band_rule.peak_count",
        "bootstrap.candidate.nominal.peak_count",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max",
    ):
        assert SENTINEL[key] in text, key
    assert set(quantities.requested_keys) == {
        "bootstrap.draws",
        "bootstrap.seed",
        "bootstrap.persistence_threshold.value",
        "bootstrap.band_quantile",
        "bootstrap.candidate.after_band_rule.peak_count",
        "bootstrap.candidate.nominal.peak_count",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max",
    }


def test_missing_band_quantile_grid_point_is_refused() -> None:
    payload = _payload(SCAN.filter(pl.col("q") != 0.95))
    with pytest.raises(ValueError, match="no grid point"):
        fig.build_bootstrap_band_quantile_scan_figure(payload, _settings())


def test_scan_row_at_band_quantile_matches_the_stage_rule() -> None:
    # bootstrap_smoothed_tree is checkpoint-fingerprinted, so it keeps its own
    # `nominal_persistence > two_c`; this pins the shared helper to that rule.
    draws = pl.DataFrame(
        {"condition": ["real"] * 5, "bottleneck": [0.1, 0.2, 0.3, 0.4, 0.5]}
    )
    two_c = 2.0 * stage.band_width(draws, 0.95)
    clears_band = np.asarray(PERSISTENCE) > two_c
    scan = scan_module.band_quantile_scan(draws, PERSISTENCE, [0.95])
    assert scan["surviving_peaks"][0] == int(clears_band.sum())
    assert scan["c"][0] == two_c / 2
    assert not scan_module.survives(two_c / 2 * 2, two_c / 2)


def test_band_quantile_must_lie_on_the_grid(tmp_path, monkeypatch) -> None:
    scan_module.require_on_grid(0.001, 0.95)
    with pytest.raises(ValueError, match="not on the grid"):
        scan_module.require_on_grid(0.1, 0.95)
    called = []
    monkeypatch.setattr(
        scan_module, "load_cached_bootstrap_smoothed_tree", lambda *_: called.append(1)
    )
    settings = {
        "analysis": {
            "bootstrap_band_quantile_scan": {"grid_step": 0.1},
            "bootstrap_smoothed_tree": {"band_quantile": 0.95},
        }
    }
    with pytest.raises(ValueError, match="not on the grid"):
        scan_module.compute_bootstrap_band_quantile_scan(
            settings, path=tmp_path / "a.parquet", survival_path=tmp_path / "b.parquet"
        )
    assert not called and not list(tmp_path.iterdir())


def test_single_run_figure_layer_order() -> None:
    # The figure layers, in order: (shade, line, rule, note, ticks, tick labels,
    # axis title). No point mark stands where the dashed rule meets the curve.
    spec = _spec()
    kinds = [
        layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
        for layer in spec["layer"]
    ]
    assert kinds[:4] == ["rect", "line", "rule", "text"]
    assert kinds[4] == "tick"
    assert "point" not in kinds


def test_panel_is_the_shared_tree_size_with_no_grid() -> None:
    spec = _spec()
    assert (spec["width"], spec["height"]) == (420, 250)
    assert spec["config"]["axis"]["grid"] is False
    assert spec["config"]["view"]["strokeWidth"] == 0


def test_configured_top_axis_ticks_are_the_four_quantiles() -> None:
    import tomllib

    from dubanevics2026topological.analysis import ROOT

    with (ROOT / "config/figures.toml").open("rb") as stream:
        table = tomllib.load(stream)["figures"][fig.FIGURE_ID]
    assert table["top_axis_quantiles"] == [0.0, 0.05, 0.95, 0.99]
    assert "panel_width" not in table and "panel_height" not in table
    assert table["line_color"] == "#111111"  # black, not blue (ADR 0010)
    # The gap above the top axis is only the offset below the subtitle.
    assert table["title_offset"] <= 12
    assert table["top_tick_length"] > 0


def _shared_settings():
    import tomllib

    from dubanevics2026topological.analysis import ROOT

    return {"figures": tomllib.loads((ROOT / "config/figures.toml").read_text())}


def test_panel_and_font_sizes_are_those_of_the_linked_tree_figure() -> None:
    settings = _shared_settings()
    tables = settings["figures"]["figures"]
    rendering = settings["figures"]["rendering"]
    # The linked tree and the scan set no font size of their own: [rendering] rules.
    sizes = {"font_size", "heading_font_size", "subtitle_font_size", "legend_font_size"}
    assert not sizes & set(tables["linked_tree_persistence"])
    assert not sizes & set(tables[fig.FIGURE_ID])
    chart = fig.build_bootstrap_band_quantile_scan_figure(_payload(), settings)
    spec = chart.to_dict()
    assert (spec["width"], spec["height"]) == (
        rendering["tree_width"],
        rendering["tree_height"],
    )
    assert spec["config"]["axis"]["labelFontSize"] == rendering["font_size"]
    assert spec["config"]["axis"]["titleFontSize"] == rendering["font_size"]
    assert spec["config"]["title"]["fontSize"] == rendering["heading_font_size"]
    assert (
        spec["config"]["title"]["subtitleFontSize"] == rendering["subtitle_font_size"]
    )
    # Every text mark (the marker note, the top tick labels and the top axis title)
    # is set at the axis font size.
    text_sizes = {
        layer["mark"]["fontSize"]
        for layer in spec["layer"]
        if isinstance(layer["mark"], dict) and layer["mark"]["type"] == "text"
    }
    assert text_sizes == {rendering["font_size"]}


def test_top_tick_label_rows_and_axis_title_clear_each_other_at_any_font_size() -> None:
    def lifts(font_size):
        spec = _spec(font_size=font_size)
        texts = [
            layer["mark"]["dy"]
            for layer in spec["layer"]
            if isinstance(layer["mark"], dict)
            and layer["mark"]["type"] == "text"
            and "dy" in layer["mark"]
            and layer["encoding"]["x"].get("field") == "tau"
            and layer["encoding"]["y"].get("value") == 0
        ]
        return texts

    for font_size in (10, 15, 22):
        lower, upper, title = sorted(lifts(font_size), reverse=True)
        assert lower - upper >= font_size  # the upper row clears the lower row
        assert upper - title >= font_size  # the axis title clears the upper row


def test_subtitle_wraps_each_phrase_to_the_configured_width() -> None:
    wide = _spec(subtitle_wrap=200)["title"]["subtitle"]
    narrow = _spec(subtitle_wrap=30)["title"]["subtitle"]
    assert len(narrow) > len(wide)
    assert all(len(line) <= 30 for line in narrow)
    assert " ".join(narrow).split() == " ".join(wide).split()  # nothing is lost


def test_scan_config_sets_no_point_size_and_wraps_its_subtitle() -> None:
    table = _shared_settings()["figures"]["figures"][fig.FIGURE_ID]
    assert "point_size" not in table  # the point mark is gone
    assert table["subtitle_wrap"] > 0


def test_a_drawn_value_that_differs_from_the_registry_is_refused() -> None:
    exact = scan_values(SCAN)
    for key in (
        "bootstrap.persistence_threshold.value",
        "bootstrap.candidate.after_band_rule.peak_count",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_min",
        "bootstrap.band_quantile_scan.bootstrapped_persistence_threshold_max",
    ):
        payload = {
            **_payload(),
            "manuscript_quantities": label_quantities({**exact, key: "123456"}),
        }
        with pytest.raises(ValueError, match=rf"registry {key}.*stale"):
            fig.build_bootstrap_band_quantile_scan_figure(payload, _settings())


def test_the_registry_text_is_compared_exactly_not_by_its_rounded_display() -> None:
    exact = scan_values(SCAN)
    key = "bootstrap.persistence_threshold.value"
    nearly = {**exact, key: repr(float(exact[key]) + 1e-12)}
    payload = {**_payload(), "manuscript_quantities": label_quantities(nearly)}
    with pytest.raises(ValueError, match="differs"):
        fig.build_bootstrap_band_quantile_scan_figure(payload, _settings())
