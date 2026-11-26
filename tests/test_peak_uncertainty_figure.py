"""Renderer checks for the four-graph peak-uncertainty publication figure."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from xml.etree import ElementTree

import altair as alt
import polars as pl
import pytest
import vl_convert as vlc

from dubanevics2026topological import figures
from dubanevics2026topological.figures.peak_uncertainty import (
    PANEL_GRID,
    PANEL_VARIANTS,
)

FIGURE_ID = "peak_uncertainty"
ROOT = Path(__file__).resolve().parents[1]

# Distinct synthetic numbers per variant, so a panel that read another variant's data
# or repeated one panel's numbers would show. The figure prints them as given.
VALUES = {
    "threshold=published_fdr=none": {
        "threshold": -0.5, "peak_count": 11, "functional_sequence_count": 1111,
        "nonfunctional_sequence_count": 2222, "uncertain_sequence_count": None,
        "largest_component_vertex_count": 3333, "fdr_method": None, "fdr_q": None,
    },
    "threshold=reflected_fdr=none": {
        "threshold": -0.25, "peak_count": 22, "functional_sequence_count": 4444,
        "nonfunctional_sequence_count": 5555, "uncertain_sequence_count": None,
        "largest_component_vertex_count": 6666, "fdr_method": None, "fdr_q": None,
    },
    "threshold=published_fdr=by_q=0.05": {
        "threshold": -0.4, "peak_count": 33, "functional_sequence_count": 7777,
        "nonfunctional_sequence_count": 8888, "uncertain_sequence_count": 99,
        "largest_component_vertex_count": 12345, "fdr_method": "by", "fdr_q": 0.05,
    },
    "threshold=reflected_fdr=by_q=0.05": {
        "threshold": -0.3, "peak_count": 44, "functional_sequence_count": 100,
        "nonfunctional_sequence_count": 200, "uncertain_sequence_count": 300,
        "largest_component_vertex_count": 400, "fdr_method": "by", "fdr_q": 0.05,
    },
}
THRESHOLDS = {variant: values["threshold"] for variant, values in VALUES.items()}


def _panel(index: int, variant: str) -> dict:
    """One variant's saved panel: its own scatter (a peak at its own fitness) and diagnostics."""
    scatter = pl.DataFrame(
        {
            "dna_sequence": ["wt", f"peak_{variant}", f"valley_{variant}"],
            "aa_sequence": ["WT", "P", "V"],
            "raw_fitness": [0.0, 0.1 * (index + 1), -0.2],
            "standard_error": [0.06, 0.04, 0.08],
            "is_peak": [False, True, False],
        }
    )
    return {
        "scatter": scatter,
        "diagnostics": {**VALUES[variant], "wild_type_sequence": "wt", "variant": variant},
    }


def _panels() -> dict:
    return {variant: _panel(index, variant) for index, variant in enumerate(PANEL_VARIANTS)}


def _payload() -> dict:
    return {"panels": _panels()}


def _render(payload=None, settings=None):
    return figures.render_payload(
        figures.get_spec(FIGURE_ID), payload or _payload(), settings or _settings(), "vanilla"
    )


def _cells(spec):
    """The four cells in row-major order: columns are graphs, rows are selections."""
    columns = spec["hconcat"]
    return [column["vconcat"][row] for row in range(2) for column in columns]


def _panels_of(spec):
    return [_scatter(cell) for cell in _cells(spec)]


def test_public_figure_route_builds_2x2_with_shared_scales_and_dashed_threshold() -> None:
    chart = _render()
    assert isinstance(chart, alt.TopLevelMixin)
    spec = chart.to_dict()

    # 2 columns (hconcat) x 2 rows (vconcat each)
    assert len(spec["hconcat"]) == 2
    col0, col1 = spec["hconcat"]
    assert len(col0["vconcat"]) == len(col1["vconcat"]) == 2
    # Columns are the graph (Published | Candidate), rows the selection (no FDR above BY FDR).
    assert col0["title"]["text"] == "Published graph"
    assert col1["title"]["text"] == "Candidate graph"
    assert [cell["title"]["text"] for cell in col0["vconcat"]] == ["Nominal", "BY q = 0.05"]
    assert [cell["title"]["text"] for cell in col1["vconcat"]] == ["Nominal", "BY q = 0.05"]
    left_scatter = col0["vconcat"][0]
    right_scatter = col1["vconcat"][0]
    assert left_scatter["width"] == right_scatter["width"] == 300
    assert left_scatter["height"] == right_scatter["height"] == 300
    assert all(
        panel["width"] == panel["height"] == 300 for panel in _panels_of(spec)
    )

    # The raster cloud sits on the same scales as the vector marks.
    cloud = next(
        layer
        for layer in left_scatter["layer"]
        if layer.get("mark", {}).get("type") == "image"
    )
    assert cloud["mark"]["align"] == "left"
    assert cloud["mark"]["baseline"] == "top"
    left_x = cloud["encoding"]["x"]["scale"]["domain"]
    right_x = right_scatter["layer"][0]["encoding"]["x"]["scale"]["domain"]
    left_y = cloud["encoding"]["y"]["scale"]["domain"]
    right_y = right_scatter["layer"][0]["encoding"]["y"]["scale"]["domain"]
    assert left_x == right_x
    assert left_y == right_y

    peak_layer = next(
        layer
        for layer in left_scatter["layer"]
        if layer.get("mark", {}).get("shape") == "triangle-up"
        and layer.get("encoding", {}).get("x", {}).get("field") == "standard_error"
    )
    plotted = spec["datasets"][peak_layer["data"]["name"]]
    wild_type = [row for row in plotted if row["dna_sequence"] == "wt"]
    assert [row["raw_fitness"] for row in wild_type] == [0.0]
    # The non-peak cloud is not an inline point table.
    assert all(row["is_peak"] or row["dna_sequence"] == "wt" for row in plotted)
    assert any(row.get("is_peak") for row in plotted)

    # Dashed threshold rule on every panel (rule strokeDash), no dotted rule.
    rule_dashes = [
        layer["mark"]["strokeDash"]
        for panel in _panels_of(spec)
        for layer in panel["layer"]
        if layer.get("mark", {}).get("type") == "rule"
    ]
    assert rule_dashes == [[5, 4]] * 4
    text_spec = str(spec)
    assert "q-value" not in text_spec
    assert "retained_peak_count" not in text_spec


def test_each_panel_reads_its_own_variant_and_no_panel_repeats_another() -> None:
    spec = _render().to_dict()
    names = []
    order = [variant for row in PANEL_GRID for variant in row]
    for variant, panel in zip(order, _panels_of(spec), strict=True):
        peak_layer = next(
            layer
            for layer in panel["layer"]
            if layer.get("mark", {}).get("shape") == "triangle-up"
            and layer.get("encoding", {}).get("x", {}).get("field") == "standard_error"
        )
        name = peak_layer["data"]["name"]
        names.append(name)
        assert name == f"{variant}_measurements"
        plotted = spec["datasets"][name]
        peak_rows = [row for row in plotted if row["is_peak"]]
        # This variant's own peak, at this variant's own fitness.
        fitness = 0.1 * (PANEL_VARIANTS.index(variant) + 1)
        assert [(row["dna_sequence"], row["raw_fitness"]) for row in peak_rows] == [
            (f"peak_{variant}", fitness)
        ]
        rule = next(l for l in panel["layer"] if l.get("mark", {}).get("type") == "rule")
        assert spec["datasets"][rule["data"]["name"]][0]["functional_threshold"] == THRESHOLDS[variant]
    assert len(set(names)) == 4
    # Four different thresholds, four different datasets.
    assert len({tuple(map(str, spec["datasets"][name])) for name in names}) == 4


def test_a_variant_missing_from_the_payload_is_an_error_not_a_repeat() -> None:
    payload = _payload()
    del payload["panels"]["threshold=reflected_fdr=none"]
    with pytest.raises(ValueError, match="threshold=reflected_fdr=none"):
        _render(payload)


def test_top_row_is_nominal_and_bottom_row_fdr_corrected() -> None:
    assert PANEL_GRID == (
        ("threshold=published_fdr=none", "threshold=reflected_fdr=none"),
        ("threshold=published_fdr=by_q=0.05", "threshold=reflected_fdr=by_q=0.05"),
    )
    payload = _payload()
    payload["panels"]["threshold=reflected_fdr=by_q=0.05"]["diagnostics"]["fdr_method"] = None
    with pytest.raises(ValueError, match="disagree|FDR-corrected"):
        _render(payload)


def test_the_figure_no_longer_reads_the_manuscript_quantity_registry() -> None:
    import inspect

    from dubanevics2026topological.figures import peak_uncertainty as module

    assert "manuscript_quantities" not in inspect.getsource(module)
    assert "manuscript_quantities" not in figures.get_spec(FIGURE_ID).inputs
    assert figures.get_spec(FIGURE_ID).inputs == {"peak_panels"}


def test_titles_and_subtitles_state_threshold_fdr_setting_peaks_and_sequence_counts() -> None:
    spec = _render().to_dict()
    panels = {
        variant: cell
        for variant, cell in zip(
            [variant for row in PANEL_GRID for variant in row],
            _cells(spec),
            strict=True,
        )
    }
    threshold_line = "Papkou et al. (2023) · Functional threshold = {}"
    assert panels["threshold=published_fdr=none"]["title"]["subtitle"] == [
        "Nominal · 11 peaks",
        threshold_line.format("-0.500"),
        "1,111 functional sequences",
        "2,222 non-functional sequences",
        "Largest component: 3,333 sequences",
    ]
    assert panels["threshold=reflected_fdr=none"]["title"]["subtitle"] == [
        "Nominal · 22 peaks",
        threshold_line.format("-0.250"),
        "4,444 functional sequences",
        "5,555 non-functional sequences",
        "Largest component: 6,666 sequences",
    ]
    # The Uncertain class exists only where the variant has it (the BY selections).
    assert panels["threshold=published_fdr=by_q=0.05"]["title"]["subtitle"] == [
        "BY q = 0.05 · 33 peaks",
        threshold_line.format("-0.400"),
        "7,777 functional sequences",
        "99 uncertain sequences",
        "8,888 non-functional sequences",
        "Largest component: 12,345 sequences",
    ]
    assert panels["threshold=reflected_fdr=by_q=0.05"]["title"]["subtitle"] == [
        "BY q = 0.05 · 44 peaks",
        threshold_line.format("-0.300"),
        "100 functional sequences",
        "300 uncertain sequences",
        "200 non-functional sequences",
        "Largest component: 400 sequences",
    ]
    assert not any("uncertain" in line for line in panels["threshold=published_fdr=none"]["title"]["subtitle"])
    # One-sentence note under each column title, naming the graph.
    notes = [column["title"]["subtitle"] for column in spec["hconcat"]]
    assert "Published graph" in notes[0] and "Candidate graph" in notes[1]
    assert "Same selection" not in str(spec)


def test_a_row_with_unequal_line_counts_is_padded_so_its_plots_stay_level() -> None:
    payload = _payload()
    payload["panels"]["threshold=reflected_fdr=none"]["diagnostics"]["uncertain_sequence_count"] = 5
    spec = _render(payload).to_dict()
    left = spec["hconcat"][0]["vconcat"][0]
    right = spec["hconcat"][1]["vconcat"][0]
    assert len(left["title"]["subtitle"]) == len(right["title"]["subtitle"])
    assert left["title"]["subtitle"][-1].strip() == ""


def test_rendered_svg_has_the_scatter_axis_titles_and_no_q_value_axis() -> None:
    chart = _render()
    svg = ElementTree.fromstring(vlc.vegalite_to_svg(chart.to_dict()))
    origins = _square_panel_origins(svg)
    assert len(origins) == 4  # 2x2
    # Y axis title on the left column of each row.
    axes = [
        node
        for node in svg.iter()
        if node.tag.endswith("}g") and "role-axis" in node.attrib.get("class", "")
    ]
    axis_titles = [node.attrib.get("aria-label", "") for node in axes]
    assert (
        sum("X-axis titled 'Pooled fitness SE'" in title for title in axis_titles) == 2
    )
    assert (
        sum("Y-axis titled 'Fitness (WT = 0)'" in title for title in axis_titles) == 2
    )
    assert not any("q-value" in title or "# peaks" in title for title in axis_titles)


def _square_panel_origins(svg):
    origins = []

    def visit(node, x=0.0, y=0.0):
        translation = re.fullmatch(
            r"translate\(([-+0-9.eE]+),([-+0-9.eE]+)\)",
            node.attrib.get("transform", ""),
        )
        if translation:
            x += float(translation[1])
            y += float(translation[2])
        if (
            node.attrib.get("class") == "background"
            and node.attrib.get("d") == "M0,0h300v300h-300Z"
        ):
            origins.append((x, y))
        for child in node:
            visit(child, x, y)

    visit(svg)
    return origins


def _settings():
    return {
        "figures": {
            "rendering": {"font_size": 15, "fitness_tick_step": 0.1},
            "figures": {
                "peak_uncertainty": {
                    "panel_size": 300,
                    "fitness_domain": [-1.2, 1.4],
                    "standard_error_domain": [0.0, 0.28],
                    "standard_error_title": "Pooled fitness SE",
                    "fitness_axis_title": "Fitness (WT = 0)",
                    # As configured: a dashed rule with no text.
                    "threshold_dash": [5, 4],
                }
            },
        }
    }


def _scatter(cell):
    """The 300px scatter. The legend panel is that scatter above a legend chart."""
    if "vconcat" in cell:
        return cell["vconcat"][0]
    return cell


def _legend_marks(panel):
    # Legend marks sit at fixed pixel offsets in both directions; the threshold
    # label has a pixel x but takes its y from the fitness scale.
    return [
        layer
        for layer in panel["layer"]
        if layer["encoding"].get("x", {}).get("value") is not None
        and layer["encoding"].get("y", {}).get("value") is not None
    ]


def test_one_legend_sits_in_the_top_right_panel() -> None:
    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), _settings(), "vanilla"
    ).to_dict()
    # Top row (no FDR), right column (Candidate graph).
    left_col, right_col = spec["hconcat"]
    right = right_col["vconcat"][0]
    assert right["title"]["text"] == "Nominal"
    # Row titles name the selection in both columns; the graph is the column title.
    for column in (left_col, right_col):
        assert [cell["title"]["text"] for cell in column["vconcat"]] == [
            "Nominal",
            "BY q = 0.05",
        ]
        assert column["vconcat"][1]["title"]["subtitle"][0].startswith("BY q = 0.05")
        assert column["vconcat"][0]["title"]["subtitle"][0].startswith("Nominal")

    assert _legend_marks(left_col["vconcat"][0]) == []
    assert _legend_marks(left_col["vconcat"][1]) == []
    assert _legend_marks(right_col["vconcat"][1]) == []
    marks = _legend_marks(right)
    points = [layer for layer in marks if layer["mark"]["type"] == "point"]
    texts = [layer for layer in marks if layer["mark"]["type"] == "text"]
    # The glyphs repeat the plotted marks: a filled black triangle, a grey circle and
    # the repository's red wild-type star.
    from dubanevics2026topological.figures._charts import FIVE_POINT_STAR
    from dubanevics2026topological.figures.palette import WILD_TYPE_RED

    assert [layer["mark"]["filled"] for layer in points] == [True, True, True]
    assert [layer["mark"]["shape"] for layer in points] == [
        "triangle-up",
        "circle",
        FIVE_POINT_STAR,
    ]
    assert [layer["mark"]["color"] for layer in points] == [
        "#111111",
        "#bdbdbd",
        WILD_TYPE_RED,
    ]
    assert [layer["mark"]["opacity"] for layer in points] == [1.0, 0.65, 1.0]
    labels = [layer["data"]["values"][0]["label"] for layer in texts]
    assert labels == ["Peak", "Not peak", "Wild type"]
    # The legend font is the title font size.
    assert {layer["mark"]["fontSize"] for layer in texts} == {
        right["title"]["fontSize"]
    }
    # Inside the plot, upper-right corner.
    for layer in marks:
        x, y = layer["encoding"]["x"]["value"], layer["encoding"]["y"]["value"]
        assert right["width"] / 2 < x < right["width"]
        assert 0 < y < right["height"] / 2

    # Three rows: peak above not peak above wild type (a smaller y is higher up),
    # one font height and the row gap apart, and each glyph stands level with its own
    # label; the top row clears the top edge by the padding.
    font = right["title"]["fontSize"]
    peak_y, other_y, wild_y = (layer["encoding"]["y"]["value"] for layer in texts)
    assert 0 < peak_y < other_y < wild_y
    assert other_y - peak_y == font + 2
    assert wild_y - other_y == font + 2
    assert peak_y == 8 + font / 2  # legend_padding, then half a row
    assert [layer["encoding"]["y"]["value"] for layer in points] == [
        peak_y,
        other_y,
        wild_y,
    ]
    # Both glyphs stand in one column, left of one column of labels, and the
    # block is inset from the right edge by the padding.
    assert len({layer["encoding"]["x"]["value"] for layer in points}) == 1
    assert len({layer["encoding"]["x"]["value"] for layer in texts}) == 1
    assert points[0]["encoding"]["x"]["value"] < texts[0]["encoding"]["x"]["value"]
    # The longest label ("Wild type") sets the column.
    assert texts[0]["encoding"]["x"]["value"] == 300 - 8 - len("Wild type") * font * 0.6

    # The legend is not inside the scatter, so the two headings stay level.
    assert right["title"]["offset"] == left_col["vconcat"][0]["title"]["offset"] == 10
    # Peaks are filled black triangles on the topmost data layer.
    peak_layer = right["layer"][2]
    assert peak_layer["mark"]["shape"] == "triangle-up"
    assert peak_layer["mark"]["filled"] is True
    assert right["layer"][1]["mark"]["type"] == "rule"
    # The rule carries no text: the peaks follow it directly.
    assert right["layer"][2]["mark"]["type"] == "point"


def test_rendered_legend_stands_in_the_top_right_plot() -> None:
    chart = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), _settings(), "vanilla"
    )
    svg = ElementTree.fromstring(vlc.vegalite_to_svg(chart.to_dict()))
    # The legend is in the top-right panel (second origin).
    origins = _square_panel_origins(svg)
    right_origin = max(origins, key=lambda origin: (origin[0], -origin[1]))
    assert right_origin[1] == min(origin[1] for origin in origins)
    placed = _text_positions(svg)
    for label in ("Peak", "Not peak"):
        x, y = placed[label]
        # Inside the top-right plot square, in its upper half.
        assert right_origin[0] < x < right_origin[0] + 300
        assert right_origin[1] < y < right_origin[1] + 150


def test_legend_text_follows_the_title_size_and_peak_glyphs_are_sized_triangles() -> (
    None
):
    settings = _settings()
    settings["figures"]["rendering"]["heading_font_size"] = 22
    settings["figures"]["figures"]["peak_uncertainty"]["peak_point_size"] = (
        40
    )
    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), settings, "vanilla"
    ).to_dict()
    # Legend is in the top row, right panel
    right = spec["hconcat"][1]["vconcat"][0]
    assert right["title"]["fontSize"] == 22
    texts = [layer for layer in _legend_marks(right) if layer["mark"]["type"] == "text"]
    assert [layer["mark"]["fontSize"] for layer in texts] == [22, 22, 22]
    assert len(texts) == 3


def test_peak_triangles_are_four_times_the_former_area_and_the_q_value_keys_are_gone() -> (
    None
):
    with (ROOT / "config/figures.toml").open("rb") as stream:
        table = tomllib.load(stream)["figures"][FIGURE_ID]
    # Altair size is an area: 112 = 4 x 28, twice the former linear size.
    assert table["peak_point_size"] == 4 * 28
    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), _settings(), "vanilla"
    ).to_dict()
    # pick a right panel
    right = spec["hconcat"][1]["vconcat"][0]
    # find the peak point layer (triangle-up, filled)
    peak_layer = next(
        (l for l in right.get("layer", []) if l.get("mark", {}).get("shape") == "triangle-up"),
        None,
    )
    assert peak_layer is not None
    assert peak_layer["mark"]["size"] == 112
    assert not {
        "count_tick_step",
        "q_axis_title",
        "count_axis_title",
        "decay_color",
        "decay_width",
        "legend_entry_gap",
    } & set(table)


def _threshold_text_layers(panel):
    return [
        layer
        for layer in panel.get("layer", [])
        if layer.get("mark", {}).get("type") == "text"
        and layer.get("encoding", {}).get("y", {}).get("field")
        == "functional_threshold"
    ]


def test_both_panels_draw_a_dashed_rule_with_no_text() -> None:
    settings = _settings()
    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), settings, "vanilla"
    ).to_dict()
    panels = _cells(spec)
    for panel, expected in zip(panels, (-0.5, -0.25, -0.4, -0.3), strict=True):
        panel = _scatter(panel)
        rule = next(
            (l for l in panel["layer"] if l.get("mark", {}).get("type") == "rule"),
            None,
        )
        assert rule is not None
        assert rule["mark"]["strokeDash"] == [5, 4]
        assert rule["encoding"]["y"]["field"] == "functional_threshold"
        row_ds = spec["datasets"][rule["data"]["name"]][0]
        assert row_ds["functional_threshold"] == expected
        # Neither the Published nor the Candidate panel names the rule in text.
        assert _threshold_text_layers(panel) == []


def _text_positions(svg):
    """Absolute (x, y) of each text node of the rendered SVG, by its content."""
    found = {}

    def visit(node, x=0.0, y=0.0):
        translation = re.fullmatch(
            r"translate\(([-+0-9.eE]+),([-+0-9.eE]+)\)",
            node.attrib.get("transform", ""),
        )
        if translation:
            x += float(translation[1])
            y += float(translation[2])
        if node.tag.endswith("}text") and node.text:
            found[node.text] = (x, y)
        for child in node:
            visit(child, x, y)

    visit(svg)
    return found


def _svg_lines(svg) -> list[str]:
    """Every text element and every tspan (one subtitle line) of the SVG.

    Vega writes a multi-line title as one text element with a tspan per line, so a
    text node's joined string is the whole subtitle; the tspans are the lines.
    """
    return [
        "".join(node.itertext())
        for node in svg.iter()
        if node.tag.endswith("}text") or node.tag.endswith("}tspan")
    ]


def test_no_rule_is_labelled_in_the_rendered_figure() -> None:
    chart = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), _settings(), "vanilla"
    )
    svg = ElementTree.fromstring(vlc.vegalite_to_svg(chart.to_dict()))
    texts = _svg_lines(svg)
    # The rule carries no text; the subtitles still state the value.
    assert texts.count("Functional threshold") == 0
    subtitles = [
        text
        for text in texts
        if text and text.startswith("Papkou et al. (2023) · Functional threshold = ")
    ]
    assert len(subtitles) >= 2


def test_the_threshold_rule_is_dashed_and_unlabelled_in_the_figure_table() -> None:
    with (ROOT / "config/figures.toml").open("rb") as stream:
        figures_table = tomllib.load(stream)["figures"]
    table = figures_table[FIGURE_ID]
    assert "threshold_label_panels" not in table
    assert "threshold_label_below" not in table
    assert "threshold_label_font_size" not in table
    assert "threshold_label_padding" not in table
    # The legend is inside the plot, so the old reservation above it is gone.
    assert "legend_gap" not in table
    assert table["title_offset"] == 10
    assert table["legend_padding"] == 8
    # The repository-wide rule: threshold = dashed.
    assert table["threshold_dash"] == [5, 4]
    assert figures_table["tree_comparison"]["threshold_label_padding"] == 8


def _axes(spec):
    """Every (panel, layer index, channel, axis) the Vega-Lite spec declares (supports 2x2 vconcat)."""
    panels = _panels_of(spec)
    return [
        (i, j, channel, layer["encoding"][channel]["axis"])
        for i, panel in enumerate(panels)
        for j, layer in enumerate(panel.get("layer", []))
        for channel in ("x", "y")
        if isinstance(layer.get("encoding", {}).get(channel, {}).get("axis"), dict)
    ]


def test_points_outside_the_configured_domains_are_counted(caplog) -> None:
    payload = _payload()
    extra = pl.DataFrame(
        {
            "dna_sequence": ["low", "wide", "lower", "upper"],
            "aa_sequence": ["L", "W", "A", "B"],
            "raw_fitness": [-2.0, 0.0, -1.2, 1.4],
            "standard_error": [0.1, 0.9, 0.0, 0.28],
            "is_peak": [False, False, False, False],
        }
    )
    for variant in ("threshold=published_fdr=none", "threshold=published_fdr=by_q=0.05"):
        panel = payload["panels"][variant]
        payload["panels"][variant] = {**panel, "scatter": pl.concat([panel["scatter"], extra])}
    with caplog.at_level("WARNING"):
        _render(payload)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    counts = (
        "1 point outside the configured fitness domain "
        "[-1.2, 1.4], 1 point outside the configured SE domain [0.0, 0.28]"
    )
    assert warnings == [
        f"Published graph / Nominal: {counts}",
        f"Published graph / BY q = 0.05: {counts}",
    ]


def test_exported_spec_keeps_titles_and_axis_labels() -> None:
    """Titles and tick labels stay on when the axis domain line is off.

    A bare ``Arial`` face also drops every glyph from the PNG, so the export
    font has to name a fallback.
    """
    with (ROOT / "config/figures.toml").open("rb") as stream:
        table = tomllib.load(stream)["figures"][FIGURE_ID]
    settings = _settings()
    settings["figures"]["figures"][FIGURE_ID] = {
        **settings["figures"]["figures"][FIGURE_ID],
        **table,
    }
    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), settings, "vanilla"
    ).to_dict()
    font = spec["config"]["font"]
    assert isinstance(font, str) and "," in font
    assert spec["hconcat"][0]["title"]["text"] == "Published graph"
    assert spec["hconcat"][1]["title"]["text"] == "Candidate graph"
    top_left = spec["hconcat"][0]["vconcat"][0]
    assert top_left["title"]["text"] == "Nominal"
    assert top_left["title"]["subtitle"]
    bottom_left = _scatter(spec["hconcat"][0]["vconcat"][1])
    axes = {
        channel: layer["encoding"][channel]["axis"]
        for layer in bottom_left["layer"]
        for channel in ("x", "y")
        if isinstance(layer.get("encoding", {}).get(channel, {}).get("axis"), dict)
        and layer["encoding"][channel]["axis"].get("title")
    }
    assert axes["x"]["title"] == "Pooled fitness SE"
    assert axes["y"]["title"] == "Fitness (WT = 0)"
    for axis in axes.values():
        assert axis["domain"] is False
        assert axis["ticks"] is True
        assert axis.get("labels") is not False
    legend_text = [
        layer
        for layer in _legend_marks(spec["hconcat"][1]["vconcat"][0])
        if layer["mark"]["type"] == "text"
    ]
    assert [layer["encoding"]["text"]["field"] for layer in legend_text] == [
        "label",
        "label",
        "label",
    ]


def test_no_grid_is_drawn_on_either_axis_only_ticks_and_labels() -> None:
    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), _settings(), "vanilla"
    ).to_dict()
    # Nothing turns a grid on: the chart-wide default is off, and so is every
    # axis a layer declares (Vega-Lite merges layers' axes, and a layer that left
    # ``grid`` unset would fall back to the default of the data type).
    assert spec["config"]["axis"]["grid"] is False
    axes = _axes(spec)
    assert axes
    assert all(axis["grid"] is False for *_, axis in axes)
    svg = vlc.vegalite_to_svg(spec)
    assert "role-axis-grid" not in svg
    # What remains of the axes are their ticks, tick labels, titles and baselines.
    for role in ("role-axis-tick", "role-axis-label", "role-axis-title"):
        assert role in svg


def test_the_grid_switch_is_the_one_config_key_and_reaches_every_axis() -> None:
    with (ROOT / "config/figures.toml").open("rb") as stream:
        assert tomllib.load(stream)["figures"][FIGURE_ID]["grid"] is False
    settings = _settings()
    settings["figures"]["figures"][FIGURE_ID]["grid"] = True
    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), settings, "vanilla"
    ).to_dict()
    # The key is not decorative: switching it on turns the grid on everywhere, so
    # no axis carries a hard-coded value that would override the config.
    assert spec["config"]["axis"]["grid"] is True
    assert all(axis["grid"] is True for *_, axis in _axes(spec))
    assert "role-axis-grid" in vlc.vegalite_to_svg(spec)


def test_the_wild_type_is_the_repository_star_on_top_of_all_panels() -> None:
    from dubanevics2026topological.figures._charts import FIVE_POINT_STAR
    from dubanevics2026topological.figures.palette import WILD_TYPE_RED

    spec = figures.render_payload(
        figures.get_spec(FIGURE_ID), _payload(), _settings(), "vanilla"
    ).to_dict()
    for panel in _panels_of(spec):
        star = panel["layer"][-1]  # the topmost layer of the scatter
        assert star["mark"]["shape"] == FIVE_POINT_STAR
        assert star["mark"]["color"] == WILD_TYPE_RED
        assert star["mark"]["size"] == 200
        assert star["mark"]["filled"] is True
        assert star["transform"] == [{"filter": 'datum.dna_sequence == "wt"'}]
        assert star["encoding"]["x"]["field"] == "standard_error"
        assert star["encoding"]["y"]["field"] == "raw_fitness"
        # The cloud is one image on the star's scales. The triangle layer leaves
        # the wild type out: it is drawn once, as the star.
        images = [
            layer
            for layer in panel["layer"]
            if layer.get("mark", {}).get("type") == "image"
        ]
        assert len(images) == 1
        assert images[0]["encoding"]["x"]["scale"]["domain"] == (
            star["encoding"]["x"]["scale"]["domain"]
        )
        assert images[0]["encoding"]["y"]["scale"]["domain"] == (
            star["encoding"]["y"]["scale"]["domain"]
        )
        others = [
            layer
            for layer in panel["layer"][:-1]
            if layer.get("transform")
            and layer["encoding"].get("x", {}).get("field") == "standard_error"
        ]
        assert len(others) == 1
        assert others[0]["mark"]["shape"] == "triangle-up"
        assert 'datum.dna_sequence != "wt"' in others[0]["transform"][0]["filter"]
        assert "datum.is_peak == true" in others[0]["transform"][0]["filter"]


def test_a_payload_without_the_wild_type_sequence_draws_no_star() -> None:
    payload = _payload()
    for panel in payload["panels"].values():
        panel["diagnostics"] = {
            key: value
            for key, value in panel["diagnostics"].items()
            if key != "wild_type_sequence"
        }
    spec = _render(payload).to_dict()
    for panel in _panels_of(spec):
        assert not any(
            "dna_sequence ==" in str(layer.get("transform")) for layer in panel["layer"]
        )


def test_shared_axis_ranges_come_from_the_figure_table() -> None:
    payload = _payload()
    for panel in payload["panels"].values():
        panel["scatter"] = panel["scatter"].with_columns(
            standard_error=pl.Series([0.001, 0.5, 0.3])
        )
    spec = _render(payload).to_dict()
    for panel in _panels_of(spec):
        x_low, x_high = panel["layer"][0]["encoding"]["x"]["scale"]["domain"]
        y_low, y_high = panel["layer"][0]["encoding"]["y"]["scale"]["domain"]
        assert [x_low, x_high] == [0.0, 0.28]
        assert [y_low, y_high] == [-1.2, 1.4]
        assert panel["layer"][0]["encoding"]["x"]["axis"]["domain"] is False
        assert panel["layer"][0]["encoding"]["y"]["axis"]["ticks"] is True


def test_the_shipped_x_title_is_pooled_fitness_se() -> None:
    config = tomllib.loads((ROOT / "config/figures.toml").read_text())
    table = config["figures"][FIGURE_ID]
    assert table["standard_error_title"] == "Pooled fitness SE"
    assert table["standard_error_domain"] == [-0.006, 0.126]



def test_the_candidate_by_subtitle_reaches_the_rendered_svg() -> None:
    payload = _payload()
    diagnostics = payload["panels"]["threshold=reflected_fdr=by_q=0.05"]["diagnostics"]
    diagnostics.update({"threshold": -0.253159, "peak_count": 0})
    svg = ElementTree.fromstring(vlc.vegalite_to_svg(_render(payload).to_dict()))
    text = _svg_lines(svg)
    assert any("Functional threshold = -0.253" in item for item in text)
    assert any(item == "BY q = 0.05 · 0 peaks" for item in text)


def test_the_payload_is_required_and_the_error_names_the_stage() -> None:
    from dubanevics2026topological.results import MissingProductError

    with pytest.raises(MissingProductError, match="peak-panels-product"):
        figures.compose_payload(figures.get_spec(FIGURE_ID), {}, {})
    with pytest.raises(MissingProductError, match="peak-panels-product"):
        figures.compose_payload(
            figures.get_spec(FIGURE_ID), {"saved_figure_inputs": {}}, {}
        )


def test_compose_hands_the_saved_panels_to_the_figure_unchanged() -> None:
    panels = _panels()
    payload = figures.compose_payload(
        figures.get_spec(FIGURE_ID), {"saved_figure_inputs": {"peak_panels": panels}}, {}
    )
    assert payload == {"panels": panels}


def test_every_printed_number_comes_from_the_panel_diagnostics_not_the_scatter() -> None:
    # The scatter has three rows and one peak in every panel; the printed counts are the
    # product's, so none of them can come from counting the table.
    spec = _render().to_dict()
    text = str(spec)
    for number in ("1,111", "2,222", "3,333", "12,345", "8,888", "7,777"):
        assert number in text


def _y_axes(panel):
    return [
        layer["encoding"]["y"]["axis"]
        for layer in panel["layer"]
        if isinstance(layer.get("encoding", {}).get("y", {}).get("axis"), dict)
    ]


def test_fitness_axis_has_three_ticks_and_each_panels_own_threshold_tick() -> None:
    spec = _render().to_dict()
    expected_left = [[-1.2, -0.5, 0.0, 1.4], [-1.2, -0.4, 0.0, 1.4]]
    expected_right = [[-0.25], [-0.3]]
    cells = _cells(spec)
    for row, left_values, right_values in zip(
        (cells[:2], cells[2:]), expected_left, expected_right, strict=True
    ):
        left, right = (_scatter(cell) for cell in row)
        # Left column: minimum, the panel's own threshold, 0 and maximum, on every layer's axis.
        left_axes = _y_axes(left)
        assert left_axes
        for axis in left_axes:
            assert axis["values"] == left_values
            assert axis["title"] == "Fitness (WT = 0)"
            assert "'.3f'" in axis["labelExpr"] and "'.1f'" in axis["labelExpr"]
        # Right column: only its own threshold, three decimals, no axis title.
        right_axes = _y_axes(right)
        assert right_axes
        for axis in right_axes:
            assert axis["values"] == right_values
            assert axis["labels"] is True
            assert axis["title"] is None
            assert repr(right_values[0]) in axis["labelExpr"]
        assert spec["resolve"]["axis"]["y"] == "independent"


def test_the_rendered_axes_show_three_decimal_threshold_ticks() -> None:
    svg = ElementTree.fromstring(vlc.vegalite_to_svg(_render().to_dict()))
    labels = ["".join(node.itertext()) for node in svg.iter() if node.tag.endswith("}text")]
    for tick in ("0.500", "0.250", "0.400", "0.300"):
        assert any(text.endswith(tick) for text in labels), tick
    # The regular ticks keep one decimal.
    assert "1.4" in labels and "0.0" in labels
