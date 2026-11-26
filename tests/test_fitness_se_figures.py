"""Fitness against SE: the Spearman panel, and Papkou published beside our estimate."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dubanevics2026topological.figures import registry
from dubanevics2026topological.figures._charts import FIVE_POINT_STAR
from dubanevics2026topological.figures.fitness_pooled_se_pair import (
    FIGURE_ID as PAIR_ID,
)
from dubanevics2026topological.figures.fitness_pooled_se_pair import (
    OURS_TITLE,
    PUBLISHED_TITLE,
    build_fitness_vs_pooled_se_figure,
    our_frame,
    published_frame,
    wild_type_sequence,
)
from dubanevics2026topological.figures.fitness_se_correlation import (
    FIGURE_ID as CORRELATION_ID,
)
from dubanevics2026topological.figures.fitness_se_correlation import (
    build_fitness_se_correlation_figure,
    fitness_and_se,
    spearman_rho,
)
from dubanevics2026topological.figures.palette import WILD_TYPE_RED

WILD_TYPE = "AAA"


def _growth() -> pl.DataFrame:
    # The wild type is in the table with s = 0 and its own pooled SE.
    return pl.DataFrame(
        {
            "dna_sequence": [WILD_TYPE, "AAC", "ACA", "AGA"],
            "s": [0.0, 0.2, -0.4, 0.1],
            "se_pooled": [0.02, 0.01, 0.05, 0.03],
        }
    )


def _published() -> pl.DataFrame:
    # Papkou's table measures every genotype against the wild type and lacks it.
    return pl.DataFrame(
        {
            "dna_from": [WILD_TYPE] * 3,
            "dna_to": ["AAC", "ACA", "AGA"],
            "s": [0.2, -0.4, 0.1],
            "se": [0.011, 0.052, 0.031],
        }
    )


def _settings(figure_id: str, **own) -> dict:
    block = {
        "panel_size": 120,
        "fitness_domain": [-0.5, 0.5],
        "standard_error_domain": [0.0, 0.06],
        "hexbin_min": 2,
        "hexbin_bins": 6,
        **own,
    }
    return {
        "figures": {
            "rendering": {"font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {figure_id: block},
        }
    }


# -- fitness_se_correlation ---------------------------------------------------------


def test_spearman_rho_on_known_samples() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_rho(x, x**2) == pytest.approx(1.0)
    assert spearman_rho(x, -x) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="at least two"):
        spearman_rho(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="constant"):
        spearman_rho(x, np.ones(4))


def test_fitness_and_se_keeps_finite_genotypes_only() -> None:
    frame = _growth().vstack(
        pl.DataFrame({"dna_sequence": ["TTT"], "s": [float("nan")], "se_pooled": [0.1]})
    )
    kept = fitness_and_se(frame)
    assert kept["dna_sequence"].to_list() == [WILD_TYPE, "AAC", "ACA", "AGA"]
    with pytest.raises(ValueError, match="missing columns"):
        fitness_and_se(pl.DataFrame({"dna_sequence": ["A"]}))


def test_the_correlation_panel_prints_spearman_rho_and_n() -> None:
    spec = build_fitness_se_correlation_figure(
        {"growth_rates": _growth()}, _settings(CORRELATION_ID)
    ).to_dict()
    rho = spearman_rho(_growth()["s"].to_numpy(), _growth()["se_pooled"].to_numpy())
    subtitle = spec["title"]["subtitle"][0]
    assert subtitle == f"Spearman ρ = {rho:.3f}, n = 4 genotypes"
    # The same hexagon density as the agreement figure, no identity line.
    marks = []

    def walk(node):
        if isinstance(node, dict):
            if "mark" in node:
                marks.append(node["mark"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    types = [m["type"] if isinstance(m, dict) else m for m in marks]
    assert "line" not in types and "rect" not in types


def _layers(spec):
    return spec["layer"] if "layer" in spec else [spec]


def test_the_correlation_panel_is_square_and_takes_its_style_from_peak_uncertainty() -> None:
    settings = {
        "figures": {
            "rendering": {"font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {
                "peak_uncertainty": {
                    "panel_size": 300,
                    "font_size": 15,
                    "axis_label_padding": 4,
                    "axis_title_padding": 8,
                    "grid": False,
                    "heading_font_size": 18,
                },
                CORRELATION_ID: {"hexbin_min": 2, "hexbin_bins": 6},
            },
        }
    }
    spec = build_fitness_se_correlation_figure(
        {"growth_rates": _growth()}, settings
    ).to_dict()
    assert spec["width"] == spec["height"] == 300
    axis = spec["config"]["axis"]
    assert axis["labelFontSize"] == axis["titleFontSize"] == 15
    assert axis["labelPadding"] == 4 and axis["titlePadding"] == 8
    assert axis["grid"] is False and axis["domain"] is False
    assert spec["config"]["title"]["fontSize"] == 18
    # Pooled SE on x, fitness on y.
    layers = _layers(spec)
    for layer in layers:
        encoding = layer["encoding"]
        assert encoding["x"]["field"] == "se"
        assert encoding["y"]["field"] == "fitness"
    # The colour bar is inside the panel; the sparsest bins are small filled light-grey dots.
    from dubanevics2026topological.figures.palette import LIGHT_GREY

    hexes = [layer for layer in layers if "color" in layer["encoding"]]
    assert hexes
    assert hexes[0]["encoding"]["color"]["legend"]["orient"] == "top-right"
    dots = [layer for layer in layers if layer["mark"]["type"] == "circle"]
    assert dots
    for dot in dots:
        assert dot["mark"]["color"] == LIGHT_GREY
        assert dot["mark"]["size"] == 12
        assert dot["mark"].get("filled", True) is not False


def test_the_correlation_figure_is_a_default_vanilla_figure_with_the_same_inputs() -> None:
    spec = registry.get_spec(CORRELATION_ID)
    assert spec.default is True  # manuscript source
    assert spec.inputs == frozenset({"growth_rates", "published_fitness"})
    assert spec.variants == frozenset({"vanilla"})
    assert spec.config_table == CORRELATION_ID


# -- fitness_vs_pooled_se -----------------------------------------------------------


def test_published_panel_has_no_wild_type_and_ours_has_it() -> None:
    assert wild_type_sequence(_published()) == WILD_TYPE
    assert WILD_TYPE not in published_frame(_published())["dna_sequence"].to_list()
    ours = our_frame(_growth())
    assert WILD_TYPE in ours["dna_sequence"].to_list()
    zero = ours.filter(pl.col("dna_sequence") == WILD_TYPE)
    assert zero["raw_fitness"].to_list() == [0.0]
    assert zero["standard_error"].to_list() == [0.02]


def test_a_published_table_with_two_origins_is_refused() -> None:
    table = _published().with_columns(
        pl.Series("dna_from", [WILD_TYPE, "TTT", WILD_TYPE])
    )
    with pytest.raises(ValueError, match="one wild type"):
        wild_type_sequence(table)


def _star_layers(panel: dict) -> list[dict]:
    return [
        layer
        for layer in panel["layer"]
        if isinstance(layer["mark"], dict)
        and layer["mark"].get("shape") == FIVE_POINT_STAR
    ]


def test_two_panels_published_left_ours_right_with_a_red_wild_type_star_on_ours_only() -> None:
    spec = build_fitness_vs_pooled_se_figure(
        {"growth_rates": _growth(), "published": _published()}, _settings(PAIR_ID)
    ).to_dict()
    left, right = spec["hconcat"]
    assert left["title"]["text"] == PUBLISHED_TITLE
    assert right["title"]["text"] == OURS_TITLE
    assert _star_layers(left) == []
    (star,) = _star_layers(right)
    assert star["mark"]["color"] == WILD_TYPE_RED
    assert star["mark"]["filled"] is True
    texts = [
        layer["encoding"]["text"]["field"]
        for layer in right["layer"]
        if layer["mark"]["type"] == "text"
    ]
    assert texts == ["label"]
    assert not [layer for layer in left["layer"] if layer["mark"]["type"] == "text"]
    # Both panels draw the same raster cloud on the same scales.
    for panel in (left, right):
        (image,) = [layer for layer in panel["layer"] if layer["mark"]["type"] == "image"]
        assert image["encoding"]["x"]["scale"]["domain"] == [0.0, 0.06]
        assert image["encoding"]["y"]["scale"]["domain"] == [-0.5, 0.5]
    # Only the left panel carries the fitness axis title.
    assert left["layer"][0]["encoding"]["y"]["axis"]["title"] == "Fitness (WT = 0)"
    assert right["layer"][0]["encoding"]["y"]["axis"]["title"] is None


def test_the_axis_ranges_default_to_those_of_peak_uncertainty() -> None:
    settings = _settings(PAIR_ID)
    block = settings["figures"]["figures"][PAIR_ID]
    del block["fitness_domain"], block["standard_error_domain"]
    settings["figures"]["figures"]["peak_uncertainty"] = {
        "fitness_domain": [-1.2, 1.4],
        "standard_error_domain": [-0.006, 0.126],
    }
    spec = build_fitness_vs_pooled_se_figure(
        {"growth_rates": _growth(), "published": _published()}, settings
    ).to_dict()
    left, _ = spec["hconcat"]
    (image,) = [layer for layer in left["layer"] if layer["mark"]["type"] == "image"]
    assert image["encoding"]["y"]["scale"]["domain"] == [-1.2, 1.4]


def test_the_pair_figure_is_an_opt_in_vanilla_figure_with_the_same_inputs() -> None:
    spec = registry.get_spec(PAIR_ID)
    assert spec.default is False
    assert spec.inputs == frozenset({"growth_rates", "published_fitness"})
    assert spec.variants == frozenset({"vanilla"})
    assert spec.config_table == PAIR_ID
