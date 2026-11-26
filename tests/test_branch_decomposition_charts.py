"""Charts, appearance settings and notebook of the exploratory branch decomposition."""

from __future__ import annotations

import ast
import copy
import importlib.util
import math
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _branch_trees import balanced_tree

from dubanevics2026topological import analysis as analysis_package
from dubanevics2026topological.analysis import branch_decomposition as stage
from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.figures import FIGURE_IDS, FIGURE_SPECS
from dubanevics2026topological.figures import branch_decomposition_charts as charts
from dubanevics2026topological.simplification.selected_leaf_simplification import (
    absorb_terminal_branches,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "papkou2023_branch_decomposition.py"
NOTEBOOK_ID = "papkou2023_branch_decomposition"
EN_DASH = "–"


@pytest.fixture(scope="module")
def decomposition():
    tree = balanced_tree()
    return stage.decompose_view(absorb_terminal_branches(tree, set()), tree.adjacency)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def tuned(settings, appearance=None, rendering=None):
    """A copy of the settings with some appearance and rendering values replaced."""
    copied = copy.deepcopy(settings)
    copied["figures"]["exploratory"]["branch_decomposition"].update(appearance or {})
    copied["figures"]["rendering"].update(rendering or {})
    return copied


def inline_datasets(chart) -> list[list[dict]]:
    """Every inline data table of the compiled chart, in drawing order."""
    found: list[list[dict]] = []

    def walk(node):
        if isinstance(node, dict):
            data = node.get("data")
            if isinstance(data, dict) and "values" in data:
                found.append(data["values"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(chart.to_dict())
    return found


def total(panel: dict) -> int:
    """Sum of ``count`` over the bars of one compiled panel."""
    return sum(row["count"] for row in panel["data"]["values"])


def bars_with(panel: dict, count: int) -> list[tuple[float, float]]:
    return [
        (row["start"], row["end"])
        for row in panel["data"]["values"]
        if row["count"] == count
    ]


# -- binning ---------------------------------------------------------------------


def test_fitness_edges_cover_the_domain_and_widen_to_hold_every_value():
    assert list(charts.fitness_edges([0.0, 1.0], 4)) == [0.0, 0.25, 0.5, 0.75, 1.0]
    widened = charts.fitness_edges([0.0, 1.0], 2, [-1.0, 3.0])
    assert widened[0] == -1.0 and widened[-1] == 3.0
    with pytest.raises(ValueError, match="at least one bin"):
        charts.fitness_edges([0.0, 1.0], 0)
    with pytest.raises(ValueError, match="real range"):
        charts.fitness_edges([1.0, 1.0], 3)


def test_vertex_count_edges_are_equal_in_log10_from_one():
    edges = charts.vertex_count_edges(100_000, 5)
    assert edges[0] == 1.0 and edges[-1] == 100_000
    assert np.allclose(np.diff(np.log10(edges)), 1.0)
    assert charts.vertex_count_edges(1, 3)[-1] == 10.0
    with pytest.raises(ValueError, match="at least one bin"):
        charts.vertex_count_edges(10, 0)


@pytest.mark.parametrize("maximum", [13, 14, 18, 30, 50_000, 103_226])
def test_the_last_vertex_count_edge_is_the_largest_branch_exactly(maximum):
    """``10 ** log10(n)`` rounds below ``n`` for these, which dropped the largest branch."""
    edges = charts.vertex_count_edges(maximum, 20)
    assert edges[-1] == float(maximum)
    assert edges[0] == 1.0 and np.all(np.diff(edges) > 0)
    rows = charts.histogram_rows([1, 7, maximum], edges)
    assert sum(row["count"] for row in rows) == 3
    assert rows[-1]["count"] >= 1  # the largest branch sits in the closed last bin


def test_histogram_rows_count_every_value_once_and_refuse_values_outside():
    edges = np.array([0.0, 1.0, 2.0])
    rows = charts.histogram_rows([0.0, 0.5, 1.0, 2.0], edges)
    assert [row["count"] for row in rows] == [2, 2]  # the last bin is closed
    assert [(row["start"], row["end"]) for row in rows] == [(0.0, 1.0), (1.0, 2.0)]
    with pytest.raises(ValueError, match="do not cover"):
        charts.histogram_rows([0.5, 3.0], edges)


# -- appearance --------------------------------------------------------------------


def test_the_appearance_table_lives_outside_the_figure_registry(settings):
    appearance = charts.appearance_settings(settings)
    assert {
        "panel_width",
        "panel_height",
        "fitness_bins",
        "vertex_count_bins",
        "histogram_color",
        "bar_seam_width",
        "log_count_axis",
        "log_count_constant",
        "sequence_axis_title",
        "branch_axis_title",
        "show_branch_facets",
        "facet_columns",
        "facet_panel_width",
        "facet_panel_height",
        "facet_font_size",
        "png_directory",
    } <= set(appearance)
    assert "bar_color" not in appearance
    assert isinstance(appearance["show_branch_facets"], bool)
    assert appearance["log_count_axis"] is True  # on by default
    assert 0 < appearance["log_count_constant"] <= 1
    assert appearance["bar_seam_width"] >= 0
    raw = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))
    assert "branch_decomposition" not in raw["figures"]
    assert NOTEBOOK_ID not in raw["figures"]
    # The PNGs go where git ignores them.
    assert appearance["png_directory"].startswith("output/")
    assert not appearance["png_directory"].startswith("output/figures")


def test_a_missing_appearance_table_is_named():
    with pytest.raises(KeyError, match=r"\[exploratory.branch_decomposition\]"):
        charts.appearance_settings({"figures": {}})


def test_axis_titles_come_from_the_configuration(decomposition, settings):
    plain = charts.order_histograms(
        decomposition.branches, decomposition.members, [1], settings
    ).to_dict()
    rendering = settings["figures"]["rendering"]
    left, right = plain["vconcat"][0]["hconcat"]
    assert left["encoding"]["x"]["axis"]["title"] == rendering["tree_axis_title"]
    assert left["encoding"]["y"]["axis"]["title"] == "Sequences"
    # The two-line legend title of [rendering] is one line on an axis.
    assert right["encoding"]["x"]["axis"]["title"] == "Branch vertex count"
    assert right["encoding"]["y"]["axis"]["title"] == "Branches"
    other = tuned(
        settings,
        {"sequence_axis_title": "Seqs", "branch_axis_title": "Edges"},
        {"tree_axis_title": "Fitness!", "branch_count_title": "Vertices\nper\nedge"},
    )
    left, right = charts.order_histograms(
        decomposition.branches, decomposition.members, [1], other
    ).to_dict()["vconcat"][0]["hconcat"]
    assert left["encoding"]["x"]["axis"]["title"] == "Fitness!"
    assert left["encoding"]["y"]["axis"]["title"] == "Seqs"
    assert right["encoding"]["x"]["axis"]["title"] == "Vertices per edge"
    assert right["encoding"]["y"]["axis"]["title"] == "Edges"


def test_the_histogram_colour_comes_from_the_configuration(decomposition, settings):
    other = tuned(settings, {"histogram_color": "#123456"})
    for chart in (
        charts.order_histograms(
            decomposition.branches, decomposition.members, [1], other
        ),
        charts.branch_facets(decomposition.branches, decomposition.members, [1], other),
    ):
        colours = {
            mark["color"]
            for mark in find_all(chart.to_dict(), "mark")
            if isinstance(mark, dict)
        }
        assert colours == {"#123456"}


def find_all(node, key):
    """Every value stored under ``key`` anywhere in a compiled spec."""
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                yield value
            yield from find_all(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from find_all(value, key)


# -- charts ------------------------------------------------------------------------


def test_each_selected_order_gets_a_row_with_the_counts_of_its_own_panels(
    decomposition, settings
):
    chart = charts.order_histograms(
        decomposition.branches, decomposition.members, [1, 2, 3], settings
    )
    rows = chart.to_dict()["vconcat"]
    assert [row["title"]["text"] for row in rows] == [
        f"Horton{EN_DASH}Strahler order 1: 4 branches, 7 sequences",
        f"Horton{EN_DASH}Strahler order 2: 2 branches, 2 sequences",
        f"Horton{EN_DASH}Strahler order 3: 1 branch, 1 sequence",
    ]
    # Per row, left: the sequences on the branches of the order; right: the branches.
    assert [(total(left), total(right)) for left, right in map_panels(rows)] == [
        (7, 4),
        (2, 2),
        (1, 1),
    ]
    only = charts.order_histograms(
        decomposition.branches, decomposition.members, [3], settings
    ).to_dict()["vconcat"]
    assert [row["title"]["text"] for row in only] == [
        f"Horton{EN_DASH}Strahler order 3: 1 branch, 1 sequence"
    ]


def map_panels(rows):
    return [tuple(row["hconcat"]) for row in rows]


def test_each_panel_holds_the_values_of_its_order(decomposition, settings):
    rows = charts.order_histograms(
        decomposition.branches, decomposition.members, [1, 2, 3], settings
    ).to_dict()["vconcat"]
    (_, size_one), (_, size_two), (fitness_three, size_three) = map_panels(rows)
    # Order 3 holds the one sequence s3 (fitness 3.0): one bar of height 1 around it.
    (start, end), *none = bars_with(fitness_three, 1)
    assert not none and start <= 3.0 <= end
    # Its single branch has one vertex, so the bar sits in the first bin (from 1).
    assert bars_with(size_three, 1)[0][0] == 1.0
    # Order 1 has branches of 2, 3, 1 and 1 vertices; order 2 two of 1.
    assert total(size_one) == 4 and bars_with(size_one, 2)
    assert [row["count"] for row in size_two["data"]["values"] if row["count"]] == [2]


def test_every_row_shares_its_bin_edges(decomposition, settings):
    chart = charts.order_histograms(
        decomposition.branches, decomposition.members, [1, 2, 3], settings
    )
    edges = {
        tuple((row["start"], row["end"]) for row in rows)
        for rows in inline_datasets(chart)
    }
    # One set of fitness edges and one of vertex-count edges, whatever the order.
    assert len(edges) == 2


def test_selecting_no_present_order_is_refused(decomposition, settings):
    with pytest.raises(ValueError, match="No branch"):
        charts.order_histograms(
            decomposition.branches, decomposition.members, [9], settings
        )
    with pytest.raises(ValueError, match="No branch"):
        charts.branch_facets(
            decomposition.branches, decomposition.members, [], settings
        )


def order_rows(chart):
    """The rows of the order chart as lists of compiled panels, each with its data."""
    rows = []
    for row in chart.to_dict()["vconcat"]:
        panels = row["hconcat"]
        if "data" in row:  # Altair hoists the data of a row with a single panel
            panels = [{**panel, "data": row["data"]} for panel in panels]
        rows.append(panels)
    return rows


def facet_panels(chart):
    """The facet grid as rows of compiled panels, ``columns`` to a row, with data."""
    spec = chart.to_dict()
    panels = spec["concat"]
    if "data" in spec:  # Altair hoists the data of a grid with a single panel
        panels = [{**panel, "data": spec["data"]} for panel in panels]
    columns = spec["columns"]
    return [panels[start : start + columns] for start in range(0, len(panels), columns)]


def test_one_facet_per_branch_laid_out_without_empty_cells(decomposition, settings):
    other = tuned(settings, {"facet_columns": 3})
    chart = charts.branch_facets(
        decomposition.branches, decomposition.members, [1, 2, 3], other
    )
    rows = facet_panels(chart)
    # One grid, not a stack of separate rows: its columns line up across rows.
    spec = chart.to_dict()
    assert spec["columns"] == 3 and "vconcat" not in spec and "hconcat" not in spec
    # Seven branches, three to a row: the last row holds only the one panel left.
    assert [len(row) for row in rows] == [3, 3, 1]
    # From the highest order down, the fittest upper endpoint first within an order.
    assert [panel["title"]["text"] for row in rows for panel in row] == [
        "s3 · order 3 · 1 sequence",
        "s1 · order 2 · 1 sequence",
        "s2 · order 2 · 1 sequence",
        "a · order 1 · 2 sequences",
        "b · order 1 · 3 sequences",
        "c · order 1 · 1 sequence",
        "d · order 1 · 1 sequence",
    ]
    # Each panel is one histogram of its own branch, so its counts are its sequences.
    assert [total(panel) for row in rows for panel in row] == [1, 1, 1, 2, 3, 1, 1]


def test_facet_axis_titles_are_not_repeated_on_every_panel(decomposition, settings):
    other = tuned(settings, {"facet_columns": 3})
    rows = facet_panels(
        charts.branch_facets(
            decomposition.branches, decomposition.members, [1, 2, 3], other
        )
    )
    flat = [panel for row in rows for panel in row]
    y_titles = [panel["encoding"]["y"]["axis"]["title"] for panel in flat]
    x_titles = [panel["encoding"]["x"]["axis"]["title"] for panel in flat]
    # The count title is on the first panel of each row only.
    assert y_titles == ["Sequences", None, None] * 2 + ["Sequences"]
    # The fitness title is on every panel with no panel below it: the last row, and
    # the columns the short last row leaves open.
    fitness_title = other["figures"]["rendering"]["tree_axis_title"]
    assert x_titles == [None] * 4 + [fitness_title] * 3
    # Tick labels stay on: no panel switches its axes off.
    assert all(panel["encoding"]["x"]["axis"].get("labels", True) for panel in flat)


def test_facet_fonts_follow_the_panel_size(decomposition, settings):
    other = tuned(settings, {"facet_font_size": 7})
    spec = charts.branch_facets(
        decomposition.branches, decomposition.members, [1, 2, 3], other
    ).to_dict()
    assert spec["config"]["axis"]["labelFontSize"] == 7
    assert spec["config"]["axis"]["titleFontSize"] == 7
    assert spec["config"]["title"]["fontSize"] == 7
    sized = [
        title["fontSize"]
        for title in find_all(spec, "title")
        if isinstance(title, dict) and "fontSize" in title
    ]
    assert sized and set(sized) == {7}
    plain = charts.order_histograms(
        decomposition.branches, decomposition.members, [1], other
    ).to_dict()
    assert plain["config"]["axis"]["labelFontSize"] == int(
        settings["figures"]["rendering"]["font_size"]
    )


def test_a_subset_of_orders_draws_only_those_facets(decomposition, settings):
    chart = charts.branch_facets(
        decomposition.branches, decomposition.members, [1], settings
    )
    titles = [panel["title"]["text"] for row in facet_panels(chart) for panel in row]
    assert titles == [
        "a · order 1 · 2 sequences",
        "b · order 1 · 3 sequences",
        "c · order 1 · 1 sequence",
        "d · order 1 · 1 sequence",
    ]


def test_save_pngs_writes_each_chart_to_the_directory(
    decomposition, settings, tmp_path
):
    chart = charts.order_histograms(
        decomposition.branches, decomposition.members, [1], settings
    )
    written = charts.save_pngs(
        {"branch_decomposition_orders": chart}, tmp_path / "out" / "nested", settings
    )
    assert [path.name for path in written] == ["branch_decomposition_orders.png"]
    assert written[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# -- count axes: ticks, domains, the log option ---------------------------------


def test_count_ticks_are_integers_at_round_steps_or_powers_of_ten():
    assert charts.count_ticks(0, 180, log=False) == [0]
    assert charts.count_ticks(800, 180, log=False) == [0, 200, 400, 600, 800]
    assert charts.count_ticks(55_895, 180, log=False) == [
        0,
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
    ]
    assert charts.count_ticks(3, 90, log=False) == [0, 1, 2, 3]
    assert charts.count_ticks(15_261, 180, log=True) == [0, 1, 10, 100, 1_000, 10_000]
    assert charts.count_ticks(9, 180, log=True) == [0, 1]
    assert charts.count_ticks(1, 180, log=True) == [0, 1]
    assert charts.count_ticks(0, 180, log=True) == [0]


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("maximum", [1, 2, 9, 10, 99, 800, 4_473, 15_261, 55_895])
@pytest.mark.parametrize("height,font", [(180, 15), (90, 9)])
def test_the_domain_keeps_the_top_tick_label_inside_the_panel(
    maximum, log, height, font
):
    """A tick at the very top would raise its panel's heading by half a label."""
    constant = 0.1
    ticks = charts.count_ticks(maximum, height, log=log)
    top = charts.count_domain_top(
        maximum, ticks, height, font, log=log, constant=constant
    )
    assert top >= maximum  # the largest bar fits

    def position(value):
        if log:
            return math.log1p(value / constant) / math.log1p(top / constant)
        return value / top

    assert (1 - position(ticks[-1])) * height >= font / 2 + 2 - 1e-9
    assert position(maximum) <= 1 + 1e-12


def test_the_axis_extent_follows_the_widest_tick_label():
    narrow = charts.count_axis_extent([40, 7], 180, 15, log=False)
    wide = charts.count_axis_extent([40, 55_895], 180, 15, log=False)
    assert wide > narrow
    assert wide == charts.count_axis_extent([55_895], 180, 15, log=False)
    # "50,000" is five digits and a comma: 5 * 0.6 + 0.3 em, plus tick and padding.
    assert wide == math.ceil(3.3 * 15) + charts.TICK_SIZE + charts.LABEL_PADDING
    assert charts.count_axis_extent([10], 180, 10, log=True) == 12 + 7


def sized_tables():
    """Two orders of very different sizes: 800 and 30,000 sequences."""
    branches = pl.DataFrame(
        {
            "branch_id": ["u1", "u2"],
            "horton_strahler_order": [1, 2],
            "branch_vertex_count": [800, 30_000],
            "birth_fitness": [1.0, 0.5],
        }
    )
    members = pl.DataFrame(
        {
            "branch_id": ["u1"] * 800 + ["u2"] * 30_000,
            "horton_strahler_order": [1] * 800 + [2] * 30_000,
            "raw_fitness": [0.1] * 800 + [-0.7] * 30_000,
        }
    )
    return branches, members


def test_every_row_of_the_order_chart_has_the_same_y_title_offset(settings):
    """Tick labels of "800" and "30,000" must not move the row headings."""
    branches, members = sized_tables()
    rows = order_rows(charts.order_histograms(branches, members, [1, 2], settings))
    for column in (0, 1):
        extents = {row[column]["encoding"]["y"]["axis"]["minExtent"] for row in rows}
        assert len(extents) == 1
    # The left extent holds the widest label of any row ("10,000" on the log axis).
    left = rows[0][0]["encoding"]["y"]["axis"]["minExtent"]
    appearance = settings["figures"]["exploratory"]["branch_decomposition"]
    size = int(settings["figures"]["rendering"]["font_size"])
    height = appearance["panel_height"]
    assert left == charts.count_axis_extent([800, 30_000], height, size, log=True)
    assert left > charts.count_axis_extent([800], height, size, log=True)
    # The facets put the offset on the titled axes only, equal across the grid.
    grid = facet_panels(charts.branch_facets(branches, members, [1, 2], settings))
    titled = {
        panel["encoding"]["y"]["axis"]["minExtent"]
        for row in grid
        for panel in row
        if "minExtent" in panel["encoding"]["y"]["axis"]
    }
    assert len(titled) == 1


def test_the_fitness_count_axis_is_symmetric_log_by_default(settings):
    branches, members = sized_tables()
    chart = charts.order_histograms(branches, members, [1, 2], settings)
    constant = settings["figures"]["exploratory"]["branch_decomposition"][
        "log_count_constant"
    ]
    for left, right in order_rows(chart):
        scale = left["encoding"]["y"]["scale"]
        assert scale["type"] == "symlog" and scale["constant"] == constant
        assert scale["domain"][0] == 0 and scale["nice"] is False
        assert left["encoding"]["y"]["axis"]["values"][:2] == [0, 1]
        # Branch counts stay on a linear axis.
        assert "type" not in right["encoding"]["y"]["scale"]
    # Per-branch facets follow the same option.
    for row in facet_panels(charts.branch_facets(branches, members, [1, 2], settings)):
        for panel in row:
            assert panel["encoding"]["y"]["scale"]["type"] == "symlog"


def test_the_log_count_axis_can_be_switched_off_or_retuned(settings):
    branches, members = sized_tables()
    linear = tuned(settings, {"log_count_axis": False})
    for left, _ in order_rows(
        charts.order_histograms(branches, members, [1, 2], linear)
    ):
        scale = left["encoding"]["y"]["scale"]
        assert "type" not in scale and "constant" not in scale
        assert left["encoding"]["y"]["axis"]["values"][1] > 1  # round linear steps
    for row in facet_panels(charts.branch_facets(branches, members, [1, 2], linear)):
        for panel in row:
            assert "type" not in panel["encoding"]["y"]["scale"]
    retuned = tuned(settings, {"log_count_constant": 0.5})
    scale = order_rows(charts.order_histograms(branches, members, [1], retuned))[0][0][
        "encoding"
    ]["y"]["scale"]
    assert scale["constant"] == 0.5


def test_small_counts_stay_visible_next_to_thousands_on_the_log_axis(settings):
    """30,000 and 5 sequences: on a linear axis the 5 is 0.017 of the panel."""
    branches = pl.DataFrame(
        {
            "branch_id": ["u"],
            "horton_strahler_order": [1],
            "branch_vertex_count": [30_005],
            "birth_fitness": [1.0],
        }
    )
    members = pl.DataFrame(
        {
            "branch_id": ["u"] * 30_005,
            "horton_strahler_order": [1] * 30_005,
            "raw_fitness": [-0.7] * 30_000 + [0.5] * 5,
        }
    )
    panel = facet_panels(charts.branch_facets(branches, members, [1], settings))[0][0]
    scale = panel["encoding"]["y"]["scale"]
    top, constant = scale["domain"][1], scale["constant"]
    height = settings["figures"]["exploratory"]["branch_decomposition"][
        "facet_panel_height"
    ]
    share = math.log1p(5 / constant) / math.log1p(top / constant)
    assert share * height > 20  # pixels; linear would give 1.5


def test_empty_bins_are_not_drawn_but_stay_in_the_data(decomposition, settings):
    chart = charts.order_histograms(
        decomposition.branches, decomposition.members, [1, 2, 3], settings
    )
    for row in order_rows(chart):
        for panel in row:
            assert panel["transform"] == [{"filter": "datum.count > 0"}]
            assert any(bar["count"] == 0 for bar in panel["data"]["values"])
    for row in facet_panels(
        charts.branch_facets(
            decomposition.branches, decomposition.members, [1, 2, 3], settings
        )
    ):
        for panel in row:
            assert panel["transform"] == [{"filter": "datum.count > 0"}]


def test_bars_carry_a_stroke_in_their_fill_colour_to_hide_the_seams(
    decomposition, settings
):
    other = tuned(settings, {"histogram_color": "#123456", "bar_seam_width": 0.75})
    for chart in (
        charts.order_histograms(
            decomposition.branches, decomposition.members, [1], other
        ),
        charts.branch_facets(decomposition.branches, decomposition.members, [1], other),
    ):
        marks = [m for m in find_all(chart.to_dict(), "mark") if isinstance(m, dict)]
        assert marks
        assert {(m["color"], m["stroke"], m["strokeWidth"]) for m in marks} == {
            ("#123456", "#123456", 0.75)
        }
        assert {m["binSpacing"] for m in marks} == {0}


# -- the notebook ------------------------------------------------------------------


def test_the_notebook_is_exploratory_and_not_a_registered_figure():
    assert NOTEBOOK.is_file()
    assert NOTEBOOK_ID not in FIGURE_IDS
    assert Path("notebooks") / NOTEBOOK.name not in {s.notebook for s in FIGURE_SPECS}
    inventory = (ROOT / "docs/figure-inventory.md").read_text("utf-8")
    assert NOTEBOOK.name in inventory and "exploratory analysis stage" in inventory
    tree = ast.parse(NOTEBOOK.read_text("utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not {"render_payload", "get_spec", "save_appearance_settings"} & imported


def run_notebook(monkeypatch, decomposition, settings, *, facets: bool):
    """Run the notebook on a cached decomposition, with the facet checkbox set."""
    patched = tuned(settings, {"show_branch_facets": facets})
    result = stage.BranchDecompositionResult(
        branches=decomposition.branches,
        members=decomposition.members,
        summary=stage.decomposition_summary(decomposition, {"tree": "band"}),
        provenance={},
        reused=True,
    )
    monkeypatch.setattr(analysis_package, "load_settings", lambda: patched)
    monkeypatch.setattr(
        stage, "load_cached_branch_decomposition", lambda settings=None: result
    )
    spec = importlib.util.spec_from_file_location("nb_branch_decomposition", NOTEBOOK)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    _, defs = notebook.app.run()
    return defs


def test_the_notebook_runs_on_a_cached_decomposition_with_facets_off(
    decomposition, settings, monkeypatch
):
    defs = run_notebook(monkeypatch, decomposition, settings, facets=False)
    assert defs["selected_orders"] == [1, 2, 3]
    assert defs["order_table"]["branch_count"].to_list() == [4, 2, 1]
    assert defs["facets"] is None
    assert isinstance(defs["result"].branches, pl.DataFrame)
    assert [
        total(panel) for row in order_rows(defs["histograms"]) for panel in row
    ] == [
        7,
        4,
        2,
        2,
        1,
        1,
    ]


def test_the_notebook_draws_the_facets_when_the_checkbox_starts_ticked(
    decomposition, settings, monkeypatch
):
    defs = run_notebook(monkeypatch, decomposition, settings, facets=True)
    panels = [panel for row in facet_panels(defs["facets"]) for panel in row]
    assert [total(panel) for panel in panels] == [1, 1, 1, 2, 3, 1, 1]
