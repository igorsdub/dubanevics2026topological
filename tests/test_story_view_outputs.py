"""The story views as multi-output registry figures: one render record per output.

A multi-output family (``tree_persistence``, ``bootstrap_band``, ``tree_comparison_views``,
``ridge_entropy``, ``graph_codon_entropy``) writes one file per saved tree, layout and style.
Each file, and its titled twin, must get a render record under ``.records/`` named by its
shelf stem.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest

from dubanevics2026topological import build
from dubanevics2026topological.figures import registry, story_views
from dubanevics2026topological.figures.story_views import FigureOutput

BAND = (
    "tree__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019"
    "_tree=band_bandq=0.95_hash=1c35ccfe"
)
MAIN_STEM = (
    "bootstrap_stops__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019"
    "_tree=band_bandq=0.95_style=annotated"
)


def _chart(text: str) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_point()
        .encode(x="x", y="y")
        .properties(title=text, width=40, height=40)
    )


def _family_outputs(payload, settings, variant):
    """Stands in for the real family: the figure-8 file, a combined file, a sidecar one."""
    yield FigureOutput(MAIN_STEM, lambda: _chart("stops"))
    yield FigureOutput(
        "bootstrap_combined__thr=reflected_fdr=by_q=0.05_style=annotated",
        lambda: _chart("combined"),
    )
    yield FigureOutput("tree_comparison_band_main", lambda: _chart("comparison"), {"k": 1})
    yield FigureOutput("will_fail", lambda: (_ for _ in ()).throw(ValueError("no tree")))


@pytest.fixture
def family_build(tmp_path, monkeypatch):
    """``build_figures`` over one synthetic multi-output family, writing under ``tmp_path``."""
    real = registry.get_spec("bootstrap_band")
    spec = replace(
        real,
        inputs=frozenset(),
        variants=frozenset({"annotated"}),
        render=lambda inputs, settings, variant: _family_outputs(inputs, settings, variant),
    )
    settings = {
        "figures": {
            "formats": ["png"],
            "variants": ["annotated"],
            "figures": {spec.figure_id: {}},
        }
    }
    monkeypatch.setattr(build, "FIGURE_SPECS", (spec,))
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(build, "load_figure_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(build, "_figure_dependencies", lambda *args: {"fixed": True})
    monkeypatch.setattr(build, "register_font_directory", lambda **kwargs: None)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "output" / "figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "output" / "figures" / ".records")
    return tmp_path / "output" / "figures"


def test_every_output_gets_a_file_a_twin_and_a_render_record(family_build) -> None:
    with pytest.raises(build.FiguresFailedError, match="bootstrap_band"):
        build.build_figures(("bootstrap_band",), context=SimpleNamespace(variant_name=None))
    records = family_build / ".records"
    for stem in (MAIN_STEM, "tree_comparison_band_main"):
        assert (family_build / "canonical" / f"{stem}.png").is_file()
        assert (family_build / "canonical" / "titled" / f"{stem}.png").is_file()
        assert (records / f"{stem}.png.json").is_file()
        assert (records / "titled" / f"{stem}.png.json").is_file()
    assert json.loads((records / f"{MAIN_STEM}.png.json").read_text())["output"] == MAIN_STEM
    # The sidecar sits beside the figure; a failed output leaves no file and no record.
    sidecar = family_build / "canonical" / "tree_comparison_band_main.json"
    assert json.loads(sidecar.read_text()) == {"k": 1}
    assert not (family_build / "canonical" / "will_fail.png").exists()
    assert not (records / "will_fail.png.json").exists()


def test_a_failed_output_does_not_stop_the_others(family_build) -> None:
    with pytest.raises(build.FiguresFailedError):
        build.build_figures(("bootstrap_band",), context=SimpleNamespace(variant_name=None))
    written = {path.name for path in (family_build / "canonical").glob("*.png")}
    assert written == {
        f"{MAIN_STEM}.png",
        "bootstrap_combined__thr=reflected_fdr=by_q=0.05_style=annotated.png",
        "tree_comparison_band_main.png",
    }


def test_the_story_view_families_declare_their_inputs_and_renderer_files() -> None:
    for figure_id in story_views.STORY_VIEW_FIGURE_IDS:
        spec = registry.get_spec(figure_id)
        assert spec.multi
        assert "story_views.py" in registry.renderer_files(spec)
        assert "view_naming.py" in registry.renderer_files(spec)
