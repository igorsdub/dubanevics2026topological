"""ADR 0012 palette (amending ADR 0010), shared panel sizes and band trapezoid."""

import tomllib

import polars as pl
import pytest

from dubanevics2026topological.analysis import ROOT, load_settings
from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures._charts import (
    band_trapezoid_frame,
    biology_persistence_marker_layer,
)
from dubanevics2026topological.figures.candidate_tree_biology import (
    persistence_band_layers,
)


@pytest.fixture(scope="module")
def figures():
    return load_settings()["figures"]


def _luminance(hex_color: str) -> float:
    """Return the WCAG relative luminance of a ``#rrggbb`` colour."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def test_stop_codon_is_solid_crimson_and_red_is_wild_type_only(figures):
    rendering = figures["rendering"]
    assert palette.STOP_CODON_CRIMSON == "#a50f15"  # dark crimson (Reds)
    assert rendering["stop_codon_color"] == palette.STOP_CODON_CRIMSON
    assert rendering["wild_type_color"] == palette.WILD_TYPE_RED
    assert palette.STOP_CODON_CRIMSON != palette.WILD_TYPE_RED
    # The crimson is the darker red, the lightness cue between the two (ADR 0012).
    assert _luminance(palette.STOP_CODON_CRIMSON) < _luminance(palette.WILD_TYPE_RED)
    # Wild type and stop codon are solid: no opacity setting exists for either, on
    # trees or on diagrams, and the lighter red and its opacity are gone.
    for key in (
        "wild_type_opacity",
        "stop_codon_opacity",
        "stop_codon_persistence_opacity",
        "stop_codon_persistence_size",
    ):
        assert key not in rendering, key
    assert not hasattr(palette, "STOP_CODON_RED")
    assert not hasattr(palette, "STOP_CODON_PERSISTENCE_OPACITY")
    text = str(figures).lower()
    assert "#e57373" not in text
    for name, table in figures["figures"].items():
        assert "biology_stop_color" not in table, name
        for key in ("stop_codon_tree_opacity", "stop_codon_persistence_opacity"):  # no per-figure override
            assert key not in table, (name, key)


def test_each_annotation_has_one_colour_in_the_config_and_the_palette(figures):
    rendering = figures["rendering"]
    assert rendering["synonymous_color"] == palette.SYNONYMOUS_GREEN
    assert rendering["wild_type_color"] == palette.WILD_TYPE_RED
    assert rendering["wild_type_label_color"] == palette.WILD_TYPE_RED
    for name, table in figures["figures"].items():
        for key, expected in (
            ("synonymous_color", palette.SYNONYMOUS_GREEN),
            ("wild_type_color", palette.WILD_TYPE_RED),
            ("stop_codon_color", palette.STOP_CODON_CRIMSON),
        ):
            assert table.get(key, expected) == expected, (name, key)


def test_persistence_markers_default_to_mid_grey_and_black_is_emphasis(figures):
    tables = figures["figures"]
    for name, table in tables.items():
        if "biology_persistence_color" in table:
            assert table["biology_persistence_color"] == palette.PEAK_GREY, name
    full = tables["full_tree_persistence"]["persistence"]
    assert full["point_color"] == palette.PEAK_GREY
    # Black is emphasis only: the absorbed peaks of the nominal-versus-bootstrap
    # figure.
    bootstrap = tables["candidate_tree_bootstrap_biology"]
    assert bootstrap["survives_color"] == palette.PEAK_GREY
    assert bootstrap["absorbed_color"] == palette.EMPHASIS_BLACK


def test_wild_type_and_stop_codon_markers_ignore_any_opacity_setting():
    from dubanevics2026topological.figures._charts import marker_config

    translucent = {"wild_type_opacity": 0.3, "stop_codon_opacity": 0.3}
    wild_type = marker_config(translucent, "wild_type", default_shape="star")
    stop = marker_config(translucent, "stop_codon", default_shape="x")
    assert wild_type.opacity == stop.opacity == 1.0
    assert stop.color == palette.STOP_CODON_CRIMSON
    assert wild_type.color == palette.WILD_TYPE_RED


def test_outcome_colours_follow_the_emphasis_rule(figures):
    colours = {
        key: table[key]
        for table in figures["figures"].values()
        for key in ("kept_color", "absorbed_color", "elder_protected_color")
        if key in table
    }
    # Kept is a plain mid-grey peak, absorbed is the black emphasis, elder-protected
    # is orange; every outcome is filled.
    assert palette.OUTCOME_COLORS == {
        "kept": palette.PEAK_GREY,
        "absorbed": palette.EMPHASIS_BLACK,
        "elder_protected": "#ff7f0e",
    }
    assert palette.PEAK_GREY == "#737373"
    assert palette.EMPHASIS_BLACK == "#111111"
    assert palette.SURVIVES_COLOR == palette.PEAK_GREY
    for table in figures["figures"].values():
        for outcome, expected in palette.OUTCOME_COLORS.items():
            assert table.get(f"{outcome}_color", expected) == expected
    assert colours  # the figures state their outcome colours explicitly
    text = str(figures).lower()
    assert "#17becf" not in text  # teal is retired
    assert not hasattr(palette, "OPEN_OUTCOMES")  # no hollow marker exists
    assert not hasattr(palette, "HOLLOW_WHITE")


def test_shared_panel_sizes_are_set_once():
    with (ROOT / "config/figures.toml").open("rb") as stream:
        figures = tomllib.load(stream)
    rendering = figures["rendering"]
    assert (rendering["tree_width"], rendering["tree_height"]) == (420, 250)
    assert (rendering["persistence_width"], rendering["persistence_height"]) == (
        250,
        250,
    )
    raw = figures["figures"]
    overrides = {"tree_comparison"}
    for name, table in raw.items():
        if name in overrides:
            continue
        assert "width" not in table.get("tree", {}), name
        assert "width" not in table.get("persistence", {}), name


def test_load_settings_fills_panel_sizes_from_the_shared_keys():
    settings = load_settings()["figures"]["figures"]
    biology = settings["candidate_tree_bootstrap_biology"]
    assert biology["tree"]["width"] == 420
    assert biology["tree"]["height"] == 250
    assert biology["persistence"]["width"] == 250
    assert (
        settings["tree_comparison"]["tree"]["width"] == 520
    )  # override kept


def test_tree_panel_size_prefers_the_figure_override():
    assert palette.tree_panel_size({}) == (420, 250)
    assert palette.tree_panel_size({"tree_width": 300}) == (300, 250)
    assert palette.tree_panel_size({}, {"tree": {"height": 90}}) == (420, 90)


def test_band_has_no_edge_keys_and_is_translucent_grey(figures):
    table = figures["figures"]["candidate_tree_bootstrap_biology"]
    assert not [key for key in table if key.startswith("band_edge")]
    assert table["band_color"] == palette.BAND_GREY
    assert table["band_opacity"] < 1


@pytest.mark.parametrize("above", [True, False])
def test_band_trapezoid_is_clipped_to_the_diagonal_extent(above):
    frame = band_trapezoid_frame(-1.0, 1.0, 0.5, above=above)
    assert frame.height == 3
    assert frame["along"].min() == -1.0 and frame["along"].max() == 1.0
    assert frame["diagonal"].to_list() == frame["along"].to_list()
    assert frame["edge"].min() >= -1.0 and frame["edge"].max() <= 1.0
    offset = (frame["edge"] - frame["diagonal"]).to_list()
    if above:
        assert offset == pytest.approx([0.5, 0.5 - 0.0, 0.0])
    else:
        assert offset == pytest.approx([0.0, -0.5, -0.5])


def test_band_wider_than_the_diagonal_stays_inside_the_extent():
    frame = band_trapezoid_frame(0.0, 1.0, 5.0)
    assert frame["edge"].to_list() == [1.0, 1.0, 1.0]
    assert frame["along"].to_list() == [0.0, 0.0, 1.0]


def test_persistence_markers_are_always_filled():
    table = pl.DataFrame(
        {
            "death_scalar": [0.0],
            "birth_scalar": [1.0],
            "birth_label": ["p"],
            "death_label": ["s"],
        }
    )
    with pytest.raises(TypeError):
        biology_persistence_marker_layer(
            table, {}, color=palette.PEAK_GREY, annotation="a", filled=False
        )
    filled_layer = biology_persistence_marker_layer(
        table, {}, color=palette.PEAK_GREY, annotation="k"
    ).to_dict()
    assert filled_layer["mark"]["shape"] == "triangle-up"
    # Filled and fully opaque by default.
    assert filled_layer["mark"]["filled"] is True
    assert filled_layer["mark"]["opacity"] == 1.0


def test_global_stability_has_no_band_edge_keys_and_a_grey_peak_colour(figures):
    table = figures["figures"]["bootstrap_global_stability"]
    assert not [key for key in table if key.startswith("band_edge")]
    assert table["peak_color"] == palette.PEAK_GREY
    assert table["histogram_width"] == 385


def test_codon_entropy_figures_dim_every_branch_but_the_focused_ridge(figures):
    dimmed = [
        table
        for table in figures["figures"].values()
        if "dimmed_branch_opacity" in table
    ]
    assert dimmed
    for table in dimmed:
        assert table["dimmed_branch_opacity"] == pytest.approx(0.25)
        # The heat map is data, not context: it stays fully opaque.
        assert "entropy_rect_opacity" not in table
    # The key it replaces is gone from the configuration, whichever table held it.
    assert not [
        name
        for name, table in figures["figures"].items()
        if "non_bridge_branch_opacity" in table
    ]
    assert "non_bridge_branch_opacity" not in figures["rendering"]


def test_full_graph_noise_band_fill_is_the_bootstrap_band_fill(figures):
    full = figures["figures"]["full_tree_persistence"]["persistence"]
    bootstrap = figures["figures"]["candidate_tree_bootstrap_biology"]
    assert full["band_color"] == bootstrap["band_color"] == palette.BAND_GREY
    assert full["band_opacity"] == bootstrap["band_opacity"] < 1
    assert not [key for key in full if key.startswith("band_edge")]
    assert full["threshold_color"] == palette.THRESHOLD_GREY


def test_diagonal_line_spans_the_diagram_range_plus_padding():
    from dubanevics2026topological.figures.bootstrap_global_stability import (
        diagonal_line,
    )

    points = pl.DataFrame({"birth": [0.5, 1.0], "death": [-1.0, 0.0]})
    line = diagonal_line(points, 0.25)
    assert line.columns == ["birth", "diagonal"]  # no band edge column
    assert line["birth"].to_list() == [-1.25, 1.25]
    assert line["diagonal"].to_list() == line["birth"].to_list()


def test_palette_greys_match_the_configured_values(figures):
    rendering = figures["rendering"]
    tables = figures["figures"]
    assert rendering["count_color"] == palette.COUNT_BLACK
    assert 0 < rendering["count_alpha_min"] < rendering["count_alpha_max"] <= 1
    for name, table in tables.items():
        if "biology_tree_color" in table:
            assert table["biology_tree_color"] == palette.TREE_GREY, name
    # Branches of the codon-entropy trees are coloured by count, so the grey ridge
    # and context colours are gone with their palette constants.
    entropy = [table for table in tables.values() if "dimmed_branch_opacity" in table]
    assert entropy
    for table in entropy:
        assert not {"ridge_color", "inactive_ridge_color", "context_color"} & set(table)
        assert "entropy_outside_color" not in table
    assert not hasattr(palette, "RIDGE_GREY")
    assert not hasattr(palette, "INACTIVE_RIDGE_GREY")
    metrics = tables["study_tree_persistence_metrics"]
    assert tuple(metrics["horton_strahler_colors"]) == palette.HORTON_STRAHLER_COLORS
    # The scatter figure keeps black as the emphasis of its peaks (ADR 0012).
    assert tables["peak_uncertainty"]["peak_color"] == palette.EMPHASIS_BLACK


def test_smoothed_tree_figure_draws_the_base_tree_in_structural_light_grey(figures):
    # The base tree is the palette's light grey, never black or the peak grey, so
    # the black absorbed branches and the grey survivors drawn over it stand out.
    bootstrap = figures["figures"]["candidate_tree_bootstrap_biology"]
    assert bootstrap["biology_tree_color"] == palette.TREE_GREY
    assert bootstrap["biology_persistence_color"] == palette.PEAK_GREY
    assert palette.TREE_GREY not in {
        palette.EMPHASIS_BLACK,
        palette.PEAK_GREY,
        palette.ELDER_PROTECTED_ORANGE,
    }
    assert bootstrap["absorbed_color"] == palette.EMPHASIS_BLACK


def test_no_grey_hex_literal_outside_the_palette_module():
    """Greys live in ``palette.py``; figure modules read them by name."""
    import re

    figure_dir = ROOT / "src/dubanevics2026topological/figures"
    offenders = []
    for path in sorted(figure_dir.glob("*.py")):
        if path.name == "palette.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for code in re.findall(r"[\"']#([0-9a-fA-F]{6})[\"']", line):
                channels = [int(code[i : i + 2], 16) for i in (0, 2, 4)]
                if max(channels) - min(channels) <= 32:
                    offenders.append(f"{path.name}:{number} #{code}")
    assert not offenders, offenders


def test_linked_figure_uses_the_shared_count_bar_ticks(figures):
    table = figures["figures"]
    ticks = table["full_tree_persistence"]["count_legend_ticks"]
    assert ticks == [1, 100, 10000]
    assert table["linked_tree_persistence"]["count_legend_ticks"] == ticks


def test_persistence_band_stops_where_the_diagonal_stops():
    # The diagonal is drawn across the unpadded scalar domain; the axes extend
    # past it by the publication padding, and the band must not.
    layers = persistence_band_layers(
        0.2, (-0.5, 1.0), {"fitness_padding_fraction": 0.05}
    )
    (rows,) = layers[0].to_dict()["datasets"].values()
    assert min(row["death_scalar"] for row in rows) == pytest.approx(-0.5)
    assert max(row["death_scalar"] for row in rows) == pytest.approx(1.0)
    assert max(row["birth_scalar"] for row in rows) == pytest.approx(1.0)


def test_persistence_band_label_is_read_from_the_appearance():
    domain = (-0.5, 1.0)
    plain = persistence_band_layers(0.2, domain, {})
    labelled = persistence_band_layers(
        0.2, domain, {"band_label": "bootstrap band (2c)"}
    )
    assert len(plain) == 1
    assert len(labelled) == 2
    (rows,) = labelled[1].to_dict()["datasets"].values()
    assert rows[0]["label"] == "bootstrap band (2c)"
