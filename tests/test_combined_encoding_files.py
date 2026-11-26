"""The combined stop-and-synonymous files: names, manuscript entry, defaults, retirements."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.figures import _charts, palette, tree_comparison_view
from dubanevics2026topological.figures import registry as figures
from dubanevics2026topological.figures import story_views
from dubanevics2026topological.figures.view_naming import (
    D10_VIEWS,
    figure_relative_path,
    single_result_figure_name,
    stable_figure_path,
)
from dubanevics2026topological.output_paths import shelf_stem

ROOT = Path(__file__).parents[1]
BAND_PARAMS = {
    "thr": "reflected",
    "fdr": "by",
    "q": "0.05",
    "draws": "10000",
    "seed": "20261019",
    "tree": "band",
    "bandq": "0.95",
    "hash": "9b3f0432",
}


def test_the_combined_files_have_one_shelf_token_each() -> None:
    nominal = single_result_figure_name(
        "tree_persistence_combined",
        {"thr": "reflected", "fdr": "by", "q": "0.05", "tree": "nominal", "hash": "a1b2c3d4"},
        "annotated",
        "png",
    )
    assert shelf_stem(nominal.replace("_hash=a1b2c3d4", "")).startswith(
        "tree_persistence_combined__thr=reflected_fdr=by_q=0.05_tree=nominal"
    )
    band = single_result_figure_name(
        "bootstrap_band_combined", BAND_PARAMS, "annotated", "png"
    )
    assert figure_relative_path("bootstrap_band_combined", band)
    stem = shelf_stem(band.replace("_hash=9b3f0432", ""))
    # The same shelf pattern as bootstrap_stops and bootstrap_synonymous.
    assert stem.startswith("bootstrap_combined__thr=reflected_fdr=by_q=0.05_draws=10000")
    assert shelf_stem("bootstrap_band_stops__x") == "bootstrap_stops__x"
    assert stable_figure_path(Path("out"), "bootstrap_band_combined", band).name == stem


def test_the_combined_views_are_registered_single_result_views() -> None:
    for name in ("tree_persistence_combined", "bootstrap_band_combined"):
        assert D10_VIEWS[name].kind == "single"


def test_the_combined_files_use_the_encoded_layout_and_are_always_written() -> None:
    assert story_views.layout_outputs("bootstrap_band", "annotated")[-1] == (
        "bootstrap_band_combined",
        "encoded",
    )
    assert story_views.layout_outputs("tree_persistence", "annotated")[-1] == (
        "tree_persistence_combined",
        "encoded",
    )


def test_the_manuscript_keeps_the_two_row_published_fdr_tree_and_no_combined_reflected_one() -> None:
    entries = tomllib.loads((ROOT / "config/manuscript.toml").read_text("utf-8"))["figure"]
    supplementary = {
        entry["desc"]: entry.get("source")
        for entry in entries
        if entry.get("section") == "supplementary"
    }
    assert supplementary["published_fdr_tree"] == (
        "tree_persistence__thr=published_fdr=by_q=0.05_tree=nominal_style=annotated"
    )
    assert not any(
        (source or "").startswith("tree_persistence_combined__thr=reflected_fdr")
        for source in supplementary.values()
    )


def test_leaf_swarm_2x2_is_retired() -> None:
    with pytest.raises(Exception):
        figures.get_spec("leaf_swarm_2x2")
    assert "leaf_swarm_2x2" not in (ROOT / "config/figures.toml").read_text("utf-8")
    assert not (ROOT / "output/figures/canonical/leaf_swarm_2x2.png").exists()
    for figure_id in ("published_leaf_swarm", "candidate_leaf_swarm", "published_fdr_leaf_swarm"):
        assert figures.get_spec(figure_id)


def test_the_manuscript_comparisons_draw_no_peak_genotype_labels() -> None:
    d10 = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))["d10"]
    assert d10["tree_comparison"]["peak_identity_labels"] is False


def _labelled_panel_layers(monkeypatch, setting):
    seen = []
    monkeypatch.setattr(
        tree_comparison_view,
        "identity_label_layers",
        lambda *a, **k: seen.append(1) or [],
    )
    monkeypatch.setattr(tree_comparison_view, "tree_chart", lambda *a, **k: _Chart())
    monkeypatch.setattr(
        tree_comparison_view, "publication_scalar_scales", lambda chart, *a, **k: chart
    )
    monkeypatch.setattr(tree_comparison_view, "tree_x_domain", lambda *a: (0.0, 1.0))
    panel = SimpleNamespace(
        title="t",
        subtitle="s",
        inputs=SimpleNamespace(threshold=None),
        peak_labels=pl.DataFrame({"x": [0.0], "y": [0.0]}),
    )
    data = SimpleNamespace(edge_segment_table=None)
    tree_comparison_view.panel_trees(
        [(panel, SimpleNamespace(), data)],
        (0.0, 1.0),
        (1.0, 2.0),
        {"peak_identity_labels": setting},
        {},
        panel_size=(100, 100),
        y_axis_extent=60,
        stops=False,
        synonymous=False,
        labelled=True,
    )
    return seen


class _Chart:
    def properties(self, **kwargs):
        return self


@pytest.mark.parametrize(("setting", "drawn"), [(False, 0), (True, 1)])
def test_peak_identity_labels_follow_the_setting(monkeypatch, setting, drawn) -> None:
    assert len(_labelled_panel_layers(monkeypatch, setting)) == drawn


def test_teal_hexes_live_only_in_the_palette_and_the_config() -> None:
    assert palette.SYNONYMOUS_TEAL != palette.SYNONYMOUS_GREEN
    rendering = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))["rendering"]
    assert rendering["synonymous_teal"] == palette.SYNONYMOUS_TEAL
    assert rendering["synonymous_teal_light"] == palette.SYNONYMOUS_TEAL_LIGHT
    assert rendering["synonymous_teal_dark"] == palette.SYNONYMOUS_TEAL_DARK


def test_the_stop_cross_is_the_tree_cross_drawn_over_a_marker() -> None:
    rows = pl.DataFrame({"death_scalar": [0.1], "birth_scalar": [0.5], "birth_label": ["A"]})
    rendering = {
        "stop_codon_color": "#a50f15",
        "stop_codon_tree_size": 60,
        "stop_codon_tree_opacity": 0.7,
        "stop_codon_stroke_width": 0.5,
    }
    spec = _charts.stop_cross_layer(rows, rendering).to_dict()
    assert spec["mark"]["shape"] == _charts.OPEN_X
    assert spec["mark"]["color"] == "#a50f15" and spec["mark"]["filled"] is False
    # Exactly the tree's cross: size, opacity and stroke width from the tree's settings.
    tree = _charts.stop_tree_points_layer(
        pl.DataFrame({"x": [0.0], "y": [0.0], "dna_sequence": ["A"], "aa_sequence": ["K"]}),
        rendering,
    ).to_dict()
    for key in ("shape", "color", "size", "opacity", "strokeWidth", "filled"):
        assert spec["mark"][key] == tree["mark"][key], key
    assert spec["mark"]["size"] == 60 and _charts.tree_marker_size(rendering) == 60
