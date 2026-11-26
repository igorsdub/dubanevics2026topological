"""Figure marker rules (GLOSSARY): valleys only on the full graph, no hollow markers, orange order."""

from __future__ import annotations

import colorsys
import re
import tomllib
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures._charts import (
    biology_persistence_marker_layer,
    peak_marker_layers,
    persistence_legend_layers,
)

ROOT = Path(__file__).parents[1]
FIGURES = ROOT / "src" / "dubanevics2026topological" / "figures"
# The two stop-codon crosses (``shape="x"``) are strokes, not fillable marks: they have
# no interior to leave hollow. Every other mark is filled.
CROSS_FILES = {"_charts.py", "peak_rank.py"}


def _sources():
    return {path.name: path.read_text() for path in sorted(FIGURES.glob("*.py"))}


def test_no_figure_module_draws_an_unfilled_mark_except_the_stop_crosses() -> None:
    offenders = {
        name
        for name, text in _sources().items()
        if re.search(r"filled\s*=\s*False", text) and name not in CROSS_FILES
    }
    assert offenders == set()
    for name in CROSS_FILES:
        for match in re.finditer(r"filled\s*=\s*False", (FIGURES / name).read_text()):
            context = (FIGURES / name).read_text()[max(0, match.start() - 400) : match.start()]
            assert "stop_codon" in context  # only the stop cross is unfilled


# The full-graph figures are the only ones that draw valleys: a filled ▼ on their
# tree and persistence diagram (``full_tree_persistence.py``) and its legend entry
# (``persistence_legend_layers(valley=True)`` in ``_charts.py``).
VALLEY_FILES = {"full_tree_persistence.py", "_charts.py"}


def test_only_the_full_graph_figure_draws_a_valley_marker_or_legend_entry() -> None:
    for name, text in _sources().items():
        if name in VALLEY_FILES:
            continue
        assert "triangle-down" not in text, name
        assert "▽" not in text and "▼" not in text, name
        assert '"valley"' not in text.replace("(no valley", ""), name
    for name, text in _sources().items():
        assert "▽" not in text, name  # no hollow glyph anywhere
    assert "triangle-down" in (FIGURES / "full_tree_persistence.py").read_text()


def test_marker_helpers_have_no_hollow_option() -> None:
    table = pl.DataFrame(
        {
            "death_scalar": [0.0],
            "birth_scalar": [1.0],
            "birth_label": ["p"],
            "death_label": ["s"],
            "birth_has_stop_codon": [False],
        }
    )
    with pytest.raises(TypeError):
        biology_persistence_marker_layer(
            table, {}, color="#737373", annotation="a", filled=False
        )
    with pytest.raises(TypeError):
        peak_marker_layers(table, {}, color="#737373", annotation="a", filled=False)
    for layer in peak_marker_layers(table, {}, color="#737373", annotation="a"):
        assert layer.to_dict()["mark"]["filled"] is True


def test_persistence_legend_has_a_filled_valley_entry_only_when_asked() -> None:
    for peak in (True, False):
        for stop in (True, False):
            for synonymous in (True, False):
                kwargs = dict(
                    peak=peak, stop_codon=stop, synonymous=synonymous, wild_type=True
                )
                default = persistence_legend_layers(250, 250, {}, **kwargs)
                texts = {layer.to_dict()["mark"]["text"] for layer in default}
                assert "valley" not in texts and "▼" not in texts
                asked = persistence_legend_layers(250, 250, {}, valley=True, **kwargs)
                marks = [layer.to_dict()["mark"] for layer in asked]
                assert [m["text"] for m in marks if m["text"] == "valley"] == ["valley"]
                glyphs = [m["text"] for m in marks if m["text"] in {"▲", "▼", "▽"}]
                assert "▼" in glyphs and "▽" not in glyphs
                # Grey like the peak entry, never a hollow glyph.
                valley_glyph = next(m for m in marks if m["text"] == "▼")
                assert valley_glyph["color"] == palette.PEAK_GREY


def test_config_has_no_valley_or_hollow_switches() -> None:
    text = (ROOT / "config" / "figures.toml").read_text()
    for key in ("valley_filled", "persistence_open_stroke_width", "relative_point_size"):
        assert key not in text


def _hue_degrees(color: str) -> float:
    red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(red, green, blue)[0] * 360


def test_horton_strahler_order_is_one_orange_ramp_everywhere() -> None:
    colors = palette.HORTON_STRAHLER_COLORS
    assert len(colors) == 4
    for color in colors:
        assert 10 <= _hue_degrees(color) <= 30  # orange
    lightness = [
        colorsys.rgb_to_hls(*(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)))[1]
        for c in colors
    ]
    assert lightness == sorted(lightness, reverse=True)  # order 4 is the darkest
    # The configured palette is the module's, for every figure that reads it.
    figures = tomllib.loads((ROOT / "config" / "figures.toml").read_text())
    configured = figures["figures"]["study_tree_persistence_metrics"]
    assert tuple(configured["horton_strahler_colors"]) == colors


def test_figures_colour_horton_strahler_order_only_from_the_palette() -> None:
    # A literal colour list next to "horton" or "strahler" would be a second palette.
    for name, text in _sources().items():
        if name == "palette.py":
            continue
        for match in re.finditer(r"horton_strahler_colors|HORTON_STRAHLER_COLORS", text):
            line = text[: match.start()].count("\n")
            window = "\n".join(text.splitlines()[line : line + 2])
            assert not re.search(r'\[\s*"#[0-9a-fA-F]{6}"', window), (name, window)
