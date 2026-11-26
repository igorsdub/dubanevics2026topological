from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from _fake_figures import install_fake_registry, write_renderer_sources
from _figure_labels import SENTINEL, label_quantities, scan_values, spec_text

from dubanevics2026topological.analysis import (
    bootstrap_band_quantile_scan as scan_module,
)
from dubanevics2026topological.figures import bootstrap_band_quantile_scan_seeds as fig

PERSISTENCE = [0.2, 0.5, 0.65, 0.9, 1.5]
LIVE = pl.DataFrame(
    {
        "q": [0.0, 0.5, 0.95, 1.0],
        "c": [0.1, 0.3, 0.45, 0.5],
        "surviving_peaks": [4, 3, 1, 1],
    }
)
def _settings(**extra):
    return {"figures": {"figures": {fig.FIGURE_ID: dict(extra)}}}


def _payload():
    cached = SimpleNamespace(
        summary={
            "settings": {"band_quantile": 0.95, "n_replicates": 10000, "seed": 20261019}
        },
        hills=pl.DataFrame({"nominal_persistence": PERSISTENCE}),
    )
    return {
        "bootstrap_smoothed": cached,
        "band_quantile_scan": LIVE,
        "survival": scan_module.survival_curve(PERSISTENCE),
        "manuscript_quantities": label_quantities(scan_values(LIVE)),
    }


def test_figure_draws_the_single_live_run_with_labels_from_the_registry() -> None:
    payload = _payload()
    chart = fig.build_bootstrap_band_quantile_scan_seeds_figure(payload, _settings())
    spec = chart.to_dict()
    text = json.dumps(spec, ensure_ascii=False)
    # The frozen 1,000-draw reference run is neither read nor drawn; the run's draws
    # and seed are the registry's (the toy checkpoint's 10,000 and 20261019 are not).
    assert "frozen reference" not in text and "1,000 draws" not in text
    assert (
        f"current run, {SENTINEL['bootstrap.draws']} draws, "
        f"seed {SENTINEL['bootstrap.seed']}"
    ) in text
    assert "10,000 draws" not in text and "20261019" not in text
    kinds = [
        layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
        for layer in spec["layer"]
    ]
    # No point mark stands where the dashed rule meets a step line: the marker
    # label alone names it.
    # A white rectangle sits behind each marker label so a step line cannot
    # cross the glyphs. The shade rect is the earlier rect.
    assert kinds == ["rect", "line", "rule", "rect", "text"]
    backings = [
        layer
        for layer in spec["layer"]
        if layer["mark"]["type"] == "rect" and layer["mark"].get("color") == "white"
    ]
    assert len(backings) == 1
    legend = spec["layer"][1]["encoding"]["color"]["legend"]
    assert legend["orient"] in ("top-right", "bottom-left")  # inside the plot area
    assert legend["labelLimit"] >= 300 and legend["symbolOpacity"] == 1
    assert legend["title"] is None and "fillColor" not in legend
    lines = [layer for layer in spec["layer"] if layer["mark"]["type"] == "line"]
    assert all(layer["encoding"]["x"]["field"] == "q" for layer in lines)
    notes = [
        spec["datasets"][layer["data"]["name"]][0]
        for layer in spec["layer"]
        if layer["mark"]["type"] == "text"
    ]
    assert [n["surviving_peaks"] for n in notes] == [1]
    assert [n["q"] for n in notes] == [0.95]
    assert [n["label"] for n in notes] == [
        (
            f"{SENTINEL['bootstrap.persistence_threshold.value']}, "
            f"{SENTINEL['bootstrap.candidate.after_band_rule.peak_count']} peaks"
        ),
    ]
    assert SENTINEL["bootstrap.band_quantile"] in spec_text(spec)
    assert payload["manuscript_quantities"].requested_keys == (
        "bootstrap.draws",
        "bootstrap.seed",
        "bootstrap.persistence_threshold.value",
        "bootstrap.candidate.after_band_rule.peak_count",
        "bootstrap.band_quantile",
    )


def test_shaded_interval_is_the_runs_holding_range() -> None:
    scan = pl.DataFrame(
        {
            "q": [0.93, 0.94, 0.95, 0.96, 0.97, 0.98],
            "c": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
            "surviving_peaks": [4, 3, 3, 3, 2, 2],
        }
    )
    assert fig.holding_range(scan, 0.95) == (0.94, 0.96)
    payload = _payload()
    payload["band_quantile_scan"] = scan.with_columns(
        pl.Series("surviving_peaks", [4, 4, 2, 2, 1, 1])
    )
    payload["manuscript_quantities"] = label_quantities(
        scan_values(payload["band_quantile_scan"])
    )
    spec = fig.build_bootstrap_band_quantile_scan_seeds_figure(
        payload, _settings()
    ).to_dict()
    rects = [
        spec["datasets"][layer["data"]["name"]][0]
        for layer in spec["layer"]
        if layer["mark"]["type"] == "rect"
    ]
    # Step-after lines draw the count at q = 0.96 until the next grid point.
    assert (rects[0]["q"], rects[0]["q_end"]) == (0.95, 0.97)


def _legend_corner(spec) -> str:
    return spec["layer"][1]["encoding"]["color"]["legend"]["orient"]


def test_legend_is_inside_the_plot_area_in_an_allowed_corner() -> None:
    payload = _payload()
    chart = fig.build_bootstrap_band_quantile_scan_seeds_figure
    assert _legend_corner(chart(payload, _settings()).to_dict()) in (
        "top-right",
        "bottom-left",
    )
    # The allowed corners come from the config.
    for corner in ("top-right", "bottom-left"):
        only = _settings(legend_corners=[corner])
        assert _legend_corner(chart(payload, only).to_dict()) == corner


WINDOW = pl.DataFrame(
    {
        "q": np.linspace(0.90, 1.0, 21),
        "surviving_peaks": np.linspace(50, 10, 21).astype(int),
    }
)  # falls left to right, leaving the top-right and bottom-left corners free
X_DOMAIN, Y_DOMAIN = (0.90, 1.0), (5.0, 55.0)
MIDDLE = {"x_start": 0.93, "x_end": 0.97, "y_start": 30.0, "y_end": 30.0}


def _corner(*labels, **rendering) -> str:
    return fig.legend_orient([WINDOW], list(labels), rendering, X_DOMAIN, Y_DOMAIN)


def test_legend_takes_the_corner_the_marker_labels_leave_free() -> None:
    top_right = {"x_start": 0.96, "x_end": 1.0, "y_start": 50.0, "y_end": 50.0}
    bottom_left = {"x_start": 0.90, "x_end": 0.94, "y_start": 10.0, "y_end": 10.0}
    assert _corner(MIDDLE) == "top-right"  # both free: the first listed corner
    assert _corner(MIDDLE, top_right) == "bottom-left"
    assert _corner(MIDDLE, bottom_left) == "top-right"
    assert _corner(MIDDLE, bottom_left, legend_corners=["bottom-left"]) == (
        "bottom-left"
    )


def test_legend_takes_the_corner_the_step_lines_leave_free() -> None:
    def corner(count: int) -> str:
        level = WINDOW.with_columns(pl.lit(count).alias("surviving_peaks"))
        return fig.legend_orient([level], [MIDDLE], {}, X_DOMAIN, Y_DOMAIN)

    assert corner(50) == "bottom-left"  # a high level line runs through the top
    assert corner(10) == "top-right"  # a low one runs through the bottom


def test_step_segments_run_level_then_step() -> None:
    window = pl.DataFrame({"q": [0.9, 0.95, 1.0], "surviving_peaks": [5, 3, 2]})
    segments = fig.step_segments(window).to_dicts()
    assert segments == [
        {"x_start": 0.9, "x_end": 0.95, "y_start": 5.0, "y_end": 5.0},
        {"x_start": 0.95, "x_end": 1.0, "y_start": 3.0, "y_end": 3.0},
        {"x_start": 0.95, "x_end": 0.95, "y_start": 5.0, "y_end": 3.0},
        {"x_start": 1.0, "x_end": 1.0, "y_start": 3.0, "y_end": 2.0},
    ]


def test_label_segment_lies_on_the_configured_side_of_its_marker() -> None:
    rendering = {
        "font_size": 9,  # a marker label is set at the figure's font size
        "panel_width": 400,
        "panel_height": 200,
        "label_char_width": 0.5,
    }
    domains = ((0.0, 0.1), (0.0, 100.0))  # 0.00025 per pixel, 0.5 per pixel
    text = "0123456789"  # 10 characters of 9 px at 0.5: 45 px wide
    above = fig.label_segment(
        (0.95, 40.0), text, ["left", 7, -9, "bottom"], rendering, *domains
    )
    assert above["x_start"] == pytest.approx(0.95 + 7 * 0.00025)
    assert above["x_end"] == pytest.approx(above["x_start"] + 45 * 0.00025)
    assert above["y_start"] == above["y_end"] == pytest.approx(40 + 13.5 * 0.5)
    below = fig.label_segment(
        (0.95, 40.0), text, ["right", -7, 9, "top"], rendering, *domains
    )
    assert below["x_end"] == pytest.approx(0.95 - 7 * 0.00025)
    assert below["x_start"] == pytest.approx(below["x_end"] - 45 * 0.00025)
    assert below["y_start"] == pytest.approx(40 - 13.5 * 0.5)


def test_label_template_comes_from_config() -> None:
    chart = fig.build_bootstrap_band_quantile_scan_seeds_figure(
        _payload(), _settings(run_label_template="n={draws}, s={seed}")
    )
    text = json.dumps(chart.to_dict())
    assert f"n={SENTINEL['bootstrap.draws']}, s={SENTINEL['bootstrap.seed']}" in text


def test_the_archived_run_code_paths_are_gone() -> None:
    for name in (
        "load_archived_run",
        "archived_run_files",
        "archived_run_label",
        "archived_marker_text",
        "check_same_grid",
        "REFERENCE_ROOT",
        "ARCHIVED_TEXT_FUNCTIONS",
    ):
        assert not hasattr(fig, name), name


def test_holding_extent_ends_at_the_step_edge_or_the_last_grid_point() -> None:
    scan = pl.DataFrame(
        {
            "q": [0.93, 0.94, 0.95, 0.96, 0.97, 0.98],
            "c": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
            "surviving_peaks": [4, 3, 3, 3, 2, 2],
        }
    )
    assert fig.holding_extent(scan, 0.95) == (0.94, 0.97)
    flat = scan.with_columns(pl.lit(3).alias("surviving_peaks"))
    assert fig.holding_extent(flat, 0.95) == (0.93, 0.98)


# -- the build: no archived files enter the fingerprint ---------------------------


def _build_settings():
    settings = _settings()
    settings["figures"].update(variants=["annotated"], formats=["svg"], rendering={})
    settings["analysis"] = {"bootstrap_band_quantile_scan": {"grid_step": 0.001}}
    return settings


def _patch_build(monkeypatch, tmp_path, rendered):
    from dubanevics2026topological import build
    from dubanevics2026topological.analysis import (
        bootstrap_band_quantile_scan as scan_stage,
    )
    from dubanevics2026topological.build import saved_figure_inputs
    from dubanevics2026topological.figures import registry

    # The registry-currency check is exercised in test_figure_quantities.py.
    monkeypatch.setattr(build, "require_current_registry", lambda settings: None)
    source = tmp_path / "src/dubanevics2026topological/figures"
    write_renderer_sources(source, "# renderer")
    (tmp_path / "output/analysis").mkdir(parents=True)
    stage = tmp_path / "src/dubanevics2026topological/analysis"
    stage.mkdir(parents=True)
    (stage / "bootstrap_band_quantile_scan.py").write_text("# stage")
    for name in ("scan.parquet", "survival.parquet"):
        (tmp_path / name).write_bytes(b"table")
    spec = next(s for s in registry.FIGURE_SPECS if s.figure_id == fig.FIGURE_ID)
    fake = dataclasses.replace(
        spec, render=lambda inputs, settings, variant: rendered.append(1) or object()
    )
    install_fake_registry(monkeypatch)
    monkeypatch.setattr(build, "FIGURE_SPECS", [fake])
    settings = _build_settings()
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(
        build,
        "load_inputs",
        lambda names, settings=None: {
            "bootstrap_smoothed": SimpleNamespace(),
            "manuscript_quantities": label_quantities(),
        },
    )
    monkeypatch.setattr(build, "_bootstrap_digests", lambda *args: {})
    # The saved-product load is exercised in the band quantile scan product tests.
    monkeypatch.setattr(
        saved_figure_inputs,
        "load_bootstrap_band_quantile_scan_product",
        lambda context, *, settings, bootstrap: {
            "product": SimpleNamespace(
                manifest=SimpleNamespace(scientific_id_sha256="0" * 64)
            )
        },
    )
    monkeypatch.setattr(build, "_digest_file", lambda path: path.read_bytes().hex())
    monkeypatch.setattr(build, "_render_one", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        scan_stage,
        "scan_paths",
        lambda context=None: (tmp_path / "scan.parquet", tmp_path / "survival.parquet"),
    )
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "figures/.records")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")
    return build


def test_build_does_not_fingerprint_or_require_an_archived_run(
    tmp_path, monkeypatch
) -> None:
    rendered: list[int] = []
    build = _patch_build(monkeypatch, tmp_path, rendered)
    build.build_figures((fig.FIGURE_ID,), variants=("annotated",))
    assert rendered == [1]
    record = json.loads(
        (tmp_path / f"figures/.records/{fig.FIGURE_ID}.svg.json").read_text()
    )
    assert not any("archived" in key for key in record)


def test_seeds_config_uses_greys_and_the_shared_tree_size() -> None:
    import tomllib

    from dubanevics2026topological.analysis import ROOT

    with (ROOT / "config/figures.toml").open("rb") as stream:
        table = tomllib.load(stream)["figures"][
            "bootstrap_band_quantile_scan_seeds"
        ]
    assert "panel_width" not in table and "panel_height" not in table
    assert table["live_color"] == "#111111"
    # Nothing of the frozen 1,000-draw reference run is configured any more.
    assert not [key for key in table if key.startswith("archived")]
    assert "label_placements" not in table


def _shared_settings():
    import tomllib

    from dubanevics2026topological.analysis import ROOT

    return {"figures": tomllib.loads((ROOT / "config/figures.toml").read_text())}


def test_panel_and_font_sizes_are_those_of_the_linked_tree_figure() -> None:
    settings = _shared_settings()
    tables = settings["figures"]["figures"]
    rendering = settings["figures"]["rendering"]
    sizes = {"font_size", "heading_font_size", "subtitle_font_size", "legend_font_size"}
    assert not sizes & set(tables["linked_tree_persistence"])
    assert not sizes & set(tables[fig.FIGURE_ID])
    chart = fig.build_bootstrap_band_quantile_scan_seeds_figure(_payload(), settings)
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
    # The shipped single run has no legend.
    assert spec["layer"][1]["encoding"]["color"]["legend"] is None
    text_sizes = {
        layer["mark"]["fontSize"]
        for layer in spec["layer"]
        if layer["mark"]["type"] == "text"
    }
    assert text_sizes == {rendering["font_size"]}


def test_run_legend_breaks_each_label_into_two_lines_when_configured() -> None:
    payload = _payload()
    chart = fig.build_bootstrap_band_quantile_scan_seeds_figure
    plain = chart(payload, _settings()).to_dict()
    split = chart(payload, _settings(legend_label_split=", ")).to_dict()
    assert "labelExpr" not in plain["layer"][1]["encoding"]["color"]["legend"]
    legend = split["layer"][1]["encoding"]["color"]["legend"]
    assert legend["labelExpr"] == 'split(datum.label, ", ")'
    assert legend["rowPadding"] == 6
    # The run label itself keeps its comma: the split is in the legend only.
    assert "current run, " in json.dumps(split, ensure_ascii=False)


def test_seeds_config_has_no_point_size_and_wraps_its_subtitle() -> None:
    table = _shared_settings()["figures"]["figures"][fig.FIGURE_ID]
    assert "point_size" not in table
    assert table["subtitle_wrap"] > 0
    assert table["legend_label_split"] == ", "


def test_the_current_run_must_equal_the_registry_threshold_and_count() -> None:
    exact = scan_values(LIVE)
    for key in (
        "bootstrap.persistence_threshold.value",
        "bootstrap.candidate.after_band_rule.peak_count",
    ):
        payload = _payload()
        payload["manuscript_quantities"] = label_quantities({**exact, key: "9"})
        with pytest.raises(ValueError, match=rf"registry {key}.*stale"):
            fig.build_bootstrap_band_quantile_scan_seeds_figure(payload, _settings())


def test_a_single_run_needs_no_legend_and_the_shipped_config_leaves_it_out() -> None:
    import tomllib
    from pathlib import Path

    shipped = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )["figures"][fig.FIGURE_ID]
    assert shipped["show_legend"] is False
    spec = fig.build_bootstrap_band_quantile_scan_seeds_figure(
        _payload(), _settings(show_legend=False)
    ).to_dict()
    assert spec["layer"][1]["encoding"]["color"]["legend"] is None
    # The marker label still names the threshold and the count.
    assert "peaks" in json.dumps(spec, ensure_ascii=False)
