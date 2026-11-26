"""The four band-quantile panels (no FDR and BY, per threshold) and the figure built from fixture scans."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.analysis import (
    bootstrap_band_quantile_scan as scan_module,
)
from dubanevics2026topological.analysis.band_quantile_scan_panels import (
    PANELS,
    load_grid,
    panel_for_variant,
    scan_panel_for_variant,
    variant_declaration,
)
from dubanevics2026topological.figures import bootstrap_band_quantile_scan as fig

ROOT = Path(__file__).resolve().parents[1]


def _write(directory: Path, panel, persistence: list[float]) -> None:
    directory.mkdir(parents=True)
    scan_module.survival_curve(persistence).write_parquet(directory / "survival.parquet")
    pl.DataFrame(
        {
            "q": [0.0, 0.95, 1.0],
            "c": [0.05, 0.2, 0.4],
            "surviving_peaks": [len(persistence), 1, 0],
        }
    ).write_parquet(directory / "scan.parquet")
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "graph": panel.as_dict(),
                "config": {
                    "checkpoint": {
                        "band_quantile": 0.95,
                        "n_replicates": 4,
                        "seed": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_the_panels_are_the_four_graphs_of_the_2x2() -> None:
    assert [(panel.threshold, panel.fdr, panel.state) for panel in PANELS] == [
        ("published", "none", "published"),
        ("published", "by", "published_by"),
        ("candidate", "none", "reflected"),
        ("candidate", "by", "reflected_by"),
    ]
    by = variant_declaration("threshold=reflected_fdr=by_q=0.05")
    assert by["method"] == "by" and by["q"] == 0.05
    assert by["cutoff"] == "median_reflected"
    assert variant_declaration("threshold=published_fdr=none")["rule"] == "fixed_cutoff"


def test_a_default_job_is_the_candidate_by_panel_only_when_the_declaration_matches() -> None:
    settings = {
        "analysis": {
            "clamped": {
                "candidate_functional_threshold": variant_declaration(
                    "threshold=reflected_fdr=by_q=0.05"
                )
            }
        }
    }
    assert panel_for_variant(None, settings).state == "reflected_by"
    assert panel_for_variant("threshold=published_fdr=by_q=0.05").state == "published_by"
    # The bootstrap stage publishes BY draw evidence only; the scan stage maps all four cells.
    assert panel_for_variant("threshold=published_fdr=none") is None
    assert scan_panel_for_variant("threshold=published_fdr=none").state == "published"
    assert scan_panel_for_variant("threshold=reflected_fdr=none").state == "reflected"
    assert scan_panel_for_variant("threshold=reflected_fdr=by_q=0.05").state == "reflected_by"
    assert scan_panel_for_variant(None, settings).state == "reflected_by"
    assert scan_panel_for_variant("mixture_quantile") is None
    assert panel_for_variant("mixture_quantile") is None
    assert panel_for_variant(None, {"analysis": {}}) is None


def test_load_grid_reads_the_four_state_directories_and_names_gaps(tmp_path: Path) -> None:
    for index, panel in enumerate(PANELS):
        _write(
            tmp_path / "band_quantile_scan" / panel.state, panel, [0.2 + index, 0.8]
        )
    loaded = load_grid(tmp_path)
    assert [item["panel"].state for item in loaded] == [
        "published",
        "published_by",
        "reflected",
        "reflected_by",
    ]
    missing = tmp_path / "band_quantile_scan" / "published_by"
    for path in missing.rglob("*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(FileNotFoundError, match="published_by.*just all"):
        load_grid(tmp_path)


def test_a_directory_that_is_not_a_panel_state_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "band_quantile_scan" / "full", PANELS[0], [0.2, 0.8])
    with pytest.raises(FileNotFoundError):
        load_grid(tmp_path)


def test_absent_no_fdr_scans_leave_the_by_panels(tmp_path: Path) -> None:
    for panel in PANELS:
        if panel.fdr == "by":
            _write(tmp_path / "band_quantile_scan" / panel.state, panel, [0.2, 0.8])
    loaded = load_grid(tmp_path)
    assert [item["panel"].state for item in loaded] == ["published_by", "reflected_by"]


def test_a_present_no_fdr_scan_is_included_and_a_missing_by_scan_still_stops(
    tmp_path: Path,
) -> None:
    for panel in PANELS:
        if panel.state != "reflected_by":
            _write(tmp_path / "band_quantile_scan" / panel.state, panel, [0.2, 0.8])
    with pytest.raises(FileNotFoundError, match=r"\(reflected_by\)"):
        load_grid(tmp_path)


def _grid_panels(selected) -> list[dict]:
    panels = []
    for index, panel in enumerate(selected):
        persistence = [0.3 + 0.1 * index, 1.2]
        panels.append(
            {
                "panel": panel,
                "scan": pl.DataFrame(
                    {
                        "q": [0.0, 0.95, 1.0],
                        "c": [0.05, 0.15 + 0.05 * index, 0.5],
                        "surviving_peaks": [2, 1, 0],
                    }
                ),
                "survival": scan_module.survival_curve(persistence),
                "summary_settings": {
                    "band_quantile": 0.95,
                    "n_replicates": 8,
                    "seed": 3,
                },
            }
        )
    return panels


def test_grid_figure_without_no_fdr_scans_is_one_row_of_the_by_panels() -> None:
    by = [panel for panel in PANELS if panel.fdr == "by"]
    spec = fig.build_bootstrap_band_quantile_scan_grid(
        _grid_panels(by),
        {
            "analysis": {"clamped": {"candidate_functional_threshold": {"q": 0.05}}},
            "figures": {"figures": {fig.GRID_FIGURE_ID: {"panel_size": 180}}},
        },
    ).to_dict()
    (framed,) = spec["vconcat"]
    (grid,) = framed["hconcat"]
    (row,) = grid["vconcat"]
    assert [cell["title"]["text"] for cell in row["hconcat"]] == [
        "Papkou + FDR",
        "Topographer + FDR",
    ]


def test_grid_figure_prints_the_fdr_level_once_from_the_settings() -> None:
    settings = {
        "analysis": {"clamped": {"candidate_functional_threshold": {"q": 0.1}}},
        "figures": {"figures": {fig.GRID_FIGURE_ID: {"title": "T"}}},
    }
    spec = fig.build_bootstrap_band_quantile_scan_grid(
        _grid_panels(PANELS), settings
    ).to_dict()
    assert spec["title"]["subtitle"] == "FDR: BY, q = 0.1"
    import json as _json

    assert _json.dumps(spec).count("FDR: BY") == 1


def test_grid_figure_is_the_2x2_titled_by_state() -> None:
    panels = _grid_panels(PANELS)
    spec = fig.build_bootstrap_band_quantile_scan_grid(
        panels,
        {"figures": {"figures": {fig.GRID_FIGURE_ID: {"panel_size": 180, "tau_axis_max": 1.5}}}},
    ).to_dict()
    left = spec
    assert left["title"]["orient"] == "left"
    assert left["title"]["text"] == "Peaks with persistence > threshold"
    (grid,) = left["hconcat"]
    assert grid["title"]["orient"] == "bottom"
    assert grid["title"]["text"] == "Persistence threshold"
    top, bottom = grid["vconcat"]
    rows = [top["hconcat"], bottom["hconcat"]]
    assert [[cell["title"]["text"] for cell in row] for row in rows] == [
        ["Papkou", "Topographer"],
        ["Papkou + FDR", "Topographer + FDR"],
    ]
    cells = [cell for row in rows for cell in row]
    assert grid["resolve"]["scale"] == {"x": "shared", "y": "independent"}
    for cell in cells:
        assert cell["width"] == cell["height"] == 180
        for layer in cell["layer"]:
            encoding = layer["encoding"]
            if "x" in encoding and "scale" in encoding["x"]:
                assert encoding["x"]["scale"]["domain"] == [0, 1.5]
                assert encoding["x"]["axis"]["grid"] is False
                assert encoding["x"]["axis"]["title"] is None
            if "y" in encoding and "axis" in encoding["y"]:
                assert encoding["y"]["axis"]["grid"] is False
                assert encoding["y"]["axis"]["title"] is None
    import vl_convert as vlc

    svg = vlc.vegalite_to_svg(spec)
    assert svg.count(">Persistence threshold<") == 1
    assert svg.count(">Peaks with persistence > threshold<") + svg.count(">Peaks with persistence &gt; threshold<") == 1


def test_the_grid_needs_the_by_panels() -> None:
    with pytest.raises(ValueError, match="two BY panels"):
        fig.build_bootstrap_band_quantile_scan_grid([], {})


def _fake_result(persistence):
    return type(
        "R",
        (),
        {
            "draw_stats": pl.DataFrame({"kind": ["real"], "distance": [0.1]}),
            "hills": pl.DataFrame({"nominal_persistence": persistence}),
            "summary": {"settings": {"band_quantile": 0.5, "n_replicates": 1, "seed": 1}},
            "provenance": {"fingerprint": {"config": {"band_quantile": 0.5}}},
        },
    )()


def _by_settings(variant: str) -> dict:
    return {
        "analysis": {
            "bootstrap_band_quantile_scan": {"grid_step": 0.5},
            "bootstrap_smoothed_tree": {"band_quantile": 0.5},
            "clamped": {"candidate_functional_threshold": variant_declaration(variant)},
        }
    }


def _by_context(tmp_path: Path, variant: str):
    return type(
        "C",
        (),
        {
            "variant_name": variant,
            "output_root": tmp_path / "output",
            "processed_root": tmp_path / "processed",
        },
    )()


def test_a_by_scan_is_recorded_and_published_to_its_processed_state(tmp_path, monkeypatch) -> None:
    result = _fake_result([0.2, 0.5])
    monkeypatch.setattr(
        scan_module, "load_cached_bootstrap_smoothed_tree", lambda given, **_: result
    )
    monkeypatch.setattr(scan_module, "band_width", lambda draws, q: 0.1)
    variant = "threshold=published_fdr=by_q=0.05"
    context = _by_context(tmp_path, variant)
    scan_module.compute_bootstrap_band_quantile_scan(
        _by_settings(variant), context=context
    )
    published = tmp_path / "processed" / "band_quantile_scan" / "published_by"
    assert sorted(path.name for path in published.iterdir()) == [
        "provenance.json",
        "scan.parquet",
        "survival.parquet",
    ]
    record = json.loads((published / "provenance.json").read_text(encoding="utf-8"))
    assert record["graph"]["threshold"] == "published"
    assert record["graph"]["fdr"] == "by"
    assert load_grid_one(published) is not None


def load_grid_one(directory: Path):
    from dubanevics2026topological.analysis.band_quantile_scan_panels import _read_panel

    return _read_panel(directory, PANELS[0])


def test_a_no_fdr_scan_is_published_to_its_processed_state(tmp_path, monkeypatch) -> None:
    result = _fake_result([0.2, 0.5])
    monkeypatch.setattr(
        scan_module, "load_cached_bootstrap_smoothed_tree", lambda given, **_: result
    )
    monkeypatch.setattr(scan_module, "band_width", lambda draws, q: 0.1)
    variant = "threshold=published_fdr=none"
    scan_module.compute_bootstrap_band_quantile_scan(
        _by_settings(variant), context=_by_context(tmp_path, variant)
    )
    published = tmp_path / "processed" / "band_quantile_scan" / "published"
    record = json.loads((published / "provenance.json").read_text(encoding="utf-8"))
    assert record["graph"]["fdr"] == "none"
    assert (published / "scan.parquet").is_file()


def test_a_job_outside_the_four_variants_publishes_no_scan(tmp_path, monkeypatch) -> None:
    result = _fake_result([0.2, 0.5])
    monkeypatch.setattr(
        scan_module, "load_cached_bootstrap_smoothed_tree", lambda given, **_: result
    )
    monkeypatch.setattr(scan_module, "band_width", lambda draws, q: 0.1)
    scan_module.compute_bootstrap_band_quantile_scan(
        _by_settings("threshold=published_fdr=none"),
        context=_by_context(tmp_path, "mixture_quantile"),
    )
    assert not (tmp_path / "processed").exists()


def test_missing_output_tables_are_restored_from_the_processed_scan(tmp_path, monkeypatch) -> None:
    result = _fake_result([0.2, 0.5])
    monkeypatch.setattr(
        scan_module, "load_cached_bootstrap_smoothed_tree", lambda given, **_: result
    )
    monkeypatch.setattr(scan_module, "band_width", lambda draws, q: 0.1)
    variant = "threshold=reflected_fdr=by_q=0.05"
    context = _by_context(tmp_path, variant)
    settings = _by_settings(variant)
    expected = scan_module.compute_bootstrap_band_quantile_scan(settings, context=context)
    for table in scan_module.scan_paths(context):
        table.unlink()
    scan, survival = scan_module.load_bootstrap_band_quantile_scan(
        settings, result, context=context
    )
    assert scan.equals(expected)


def test_a_missing_scan_names_the_recipes(tmp_path, monkeypatch) -> None:
    result = _fake_result([0.2, 0.5])
    variant = "threshold=reflected_fdr=by_q=0.05"
    with pytest.raises(FileNotFoundError, match="just bootstrap-band-quantile-scan.*just fetch-data"):
        scan_module.load_bootstrap_band_quantile_scan(
            _by_settings(variant), result, context=_by_context(tmp_path, variant)
        )


def test_the_shipped_grid_config_draws_no_gridlines() -> None:
    import tomllib
    from pathlib import Path

    shipped = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )
    assert shipped["figures"]["bootstrap_band_quantile_scan_grid"]["grid"] is False
