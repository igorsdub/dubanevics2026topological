"""Branch vertex counts of the four graphs as a 2x2 of overlaid bars by Horton-Strahler order."""

from __future__ import annotations

import polars as pl
import pytest

from dubanevics2026topological.build import FOUR_GRAPH_FIGURE_IDS, NON_CANDIDATE_FIGURE_IDS
from dubanevics2026topological.figures._four_graphs import PANELS
from dubanevics2026topological.figures.branch_vertex_counts_grid import (
    FIGURE_ID,
    LEGEND_TITLE,
    ORDERS,
    build_branch_vertex_counts_grid,
    draw_ranks,
    panel_table,
    shared_edges,
    stacked_rows,
)
from dubanevics2026topological.figures.palette import HORTON_STRAHLER_COLORS
from dubanevics2026topological.figures.registry import get_spec

# Heavy-tailed counts per order for each graph; the Candidate FDR graph is smallest.
COUNTS = {
    "published": {1: [1, 1, 1, 2, 2, 3], 2: [4, 8], 3: [30], 4: [400]},
    "published_fdr": {1: [1, 1, 2], 2: [5], 3: [60], 4: [300]},
    "candidate": {1: [1, 1, 1, 1, 2, 2, 3], 2: [4, 8, 9], 3: [20, 40], 4: [200]},
    "candidate_fdr": {1: [1, 2], 2: [6], 3: [25], 4: [100]},
}


def _table(counts: dict[int, list[int]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"branch_vertex_count": count, "horton_strahler_order": order}
            for order, values in counts.items()
            for count in values
        ]
    )


def _payload() -> dict:
    return {
        "panels": {
            PANELS[slug][0]: {"branch_table": _table(counts), "leaf_tables": None}
            for slug, counts in COUNTS.items()
        }
    }


def _settings() -> dict:
    return {
        "figures": {
            "rendering": {"font_size": 15, "font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {
                FIGURE_ID: {
                    "title": "Branch vertex counts, four graphs",
                    "panel_width": 200,
                    "panel_height": 120,
                    "vertex_count_bins": 8,
                    "bar_seam_width": 0.5,
                }
            },
        }
    }


def _cells(spec: dict) -> list[list[dict]]:
    """Each panel as its title plus its bar layer's mark, encoding and data."""
    return [
        [
            {**cell["layer"][0], "title": cell["title"], "total": cell["layer"][1]}
            for cell in row["hconcat"]
        ]
        for row in spec["vconcat"]
    ]


def test_the_grid_is_published_left_candidate_right_nominal_above_by_below() -> None:
    spec = build_branch_vertex_counts_grid(_payload(), _settings()).to_dict()
    titles = [[cell["title"]["text"] for cell in row] for row in _cells(spec)]
    assert titles == [
        ["Published graph · Nominal", "Candidate graph · Nominal"],
        ["Published graph · BY q = 0.05", "Candidate graph · BY q = 0.05"],
    ]
    assert spec["title"]["text"] == "Branch vertex counts, four graphs"


def test_each_panel_writes_its_branch_total_inside_the_plot() -> None:
    spec = build_branch_vertex_counts_grid(_payload(), _settings()).to_dict()
    branches = {
        "Published graph · Nominal": 10,
        "Candidate graph · Nominal": 13,
        "Published graph · BY q = 0.05": 6,
        "Candidate graph · BY q = 0.05": 5,
    }
    for row in _cells(spec):
        for cell in row:
            assert "subtitle" not in cell["title"]
            total = cell["total"]
            assert total["data"]["values"] == [
                {"label": f"{branches[cell['title']['text']]:,} branches"}
            ]
            assert total["mark"]["type"] == "text" and total["mark"]["align"] == "left"
            # Top left, inside the plot area.
            assert total["encoding"]["x"] == {"value": 0}
            assert total["encoding"]["y"] == {"value": 0}


def test_bars_count_branches_by_order_on_shared_edges() -> None:
    tables = {slug: _table(counts) for slug, counts in COUNTS.items()}
    edges, log_scale = shared_edges(list(tables.values()), bins=8)
    assert log_scale
    assert edges[0] == 1.0 and edges[-1] == 400.0
    for slug, table in tables.items():
        rows = stacked_rows(table, edges)
        for order in ORDERS:
            total = sum(r["count"] for r in rows if r["order"] == str(order))
            assert total == len(COUNTS[slug][order])
        # Every order shares the bin edges, empty bins included.
        assert {(r["start"], r["end"]) for r in rows} == set(zip(edges[:-1], edges[1:], strict=True))


def test_panels_overlay_in_the_purple_palette_and_share_axes() -> None:
    spec = build_branch_vertex_counts_grid(_payload(), _settings()).to_dict()
    domains, ymaxes = set(), set()
    for row in _cells(spec):
        for cell in row:
            assert cell["mark"]["type"] == "bar"
            color = cell["encoding"]["color"]
            assert color["scale"]["range"] == list(HORTON_STRAHLER_COLORS)
            assert color["scale"]["domain"] == ["1", "2", "3", "4"]
            # Overlaid, not stacked, at 70 % opacity.
            assert cell["encoding"]["y"]["stack"] is None
            assert cell["mark"]["opacity"] == 0.7
            x = cell["encoding"]["x"]
            assert x["scale"]["type"] == "log"
            domains.add(tuple(x["scale"]["domain"]))
            ymaxes.add(tuple(cell["encoding"]["y"]["scale"]["domain"]))
            # y is the fraction of the panel's tallest bar: its maximum is exactly 1.
            assert cell["encoding"]["y"]["field"] == "fraction"
            fractions = [r["fraction"] for r in cell["data"]["values"]]
            assert max(fractions) == 1.0 and min(fractions) >= 0.0
    assert len(domains) == 1 and ymaxes == {(0, 1.05)}


def test_the_order_with_the_fewest_branches_is_drawn_in_front() -> None:
    tables = [_table(counts) for counts in COUNTS.values()]
    ranks = draw_ranks(tables)
    # Branch totals over the four graphs: order 1 has 18, order 2 has 7, order 3 has 5
    # and order 4 has 4, so order 4 is drawn last (in front) and order 1 first (at the back).
    assert ranks == {"1": 0, "2": 1, "3": 2, "4": 3}
    # Ties go to the higher order at the back.
    tie = [_table({1: [1], 2: [1], 3: [1, 1], 4: [1, 1]})]
    assert draw_ranks(tie) == {"4": 0, "3": 1, "2": 2, "1": 3}
    spec = build_branch_vertex_counts_grid(_payload(), _settings()).to_dict()
    for row in _cells(spec):
        for cell in row:
            order = cell["encoding"]["order"]
            assert order["field"] == "draw_rank" and order["sort"] == "ascending"
            drawn = {r["order"]: r["draw_rank"] for r in cell["data"]["values"]}
            assert drawn == ranks
    # Vega-Lite compiles that channel into the mark's sort, so rank 3 is painted last.
    import vl_convert as vlc

    vega = vlc.vegalite_to_vega(spec)
    marks = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "rect" and "sort" in node:
                marks.append(node["sort"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(vega)
    assert marks and all(
        sort == {"field": ['datum["draw_rank"]'], "order": ["ascending"]} for sort in marks
    )


def test_the_default_is_one_hundred_bins() -> None:
    settings = _settings()
    del settings["figures"]["figures"][FIGURE_ID]["vertex_count_bins"]
    spec = build_branch_vertex_counts_grid(_payload(), settings).to_dict()
    cell = _cells(spec)[0][0]
    rows = cell["data"]["values"]
    assert len({r["start"] for r in rows}) == 100


def test_one_horizontal_legend_titled_by_order_sits_inside_the_top_right_panel() -> None:
    spec = build_branch_vertex_counts_grid(_payload(), _settings()).to_dict()
    legends = [
        (r, c, cell["encoding"]["color"].get("legend"))
        for r, row in enumerate(_cells(spec))
        for c, cell in enumerate(row)
    ]
    shown = [(r, c, legend) for r, c, legend in legends if legend]
    assert [(r, c) for r, c, _ in shown] == [(0, 1)]
    legend = shown[0][2]
    # Placed by pixel inside the panel's top right (orient "none"), not outside it.
    assert legend["orient"] == "none"
    assert 0 < legend["legendX"] < 360 and legend["legendY"] >= 0
    # Drawn on top of the bars.
    assert legend["zindex"] == 1
    assert legend["direction"] == "horizontal"
    assert legend["columns"] == len(ORDERS)
    assert legend["title"] == LEGEND_TITLE == "Horton–Strahler order"
    # The panels' legends stay their own, so none is hoisted outside the plot.
    assert spec["resolve"]["scale"]["color"] == "independent"


def test_axis_titles_sit_on_the_outer_edges_only() -> None:
    spec = build_branch_vertex_counts_grid(_payload(), _settings()).to_dict()
    for r, row in enumerate(_cells(spec)):
        for c, cell in enumerate(row):
            x_title = cell["encoding"]["x"].get("title")
            y_title = cell["encoding"]["y"].get("title")
            assert (x_title == "Branch vertex count") is (r == 1)
            assert (y_title == "Branches (fraction of max)") is (c == 0)


def test_a_missing_graph_or_a_bad_table_is_refused() -> None:
    payload = _payload()
    del payload["panels"][PANELS["candidate_fdr"][0]]
    with pytest.raises(ValueError, match="lacks the"):
        build_branch_vertex_counts_grid(payload, _settings())
    bad = _payload()
    bad["panels"][PANELS["published"][0]]["branch_table"] = _table({5: [3]})
    with pytest.raises(ValueError, match="orders outside"):
        panel_table(bad, "published")
    short = _payload()
    short["panels"][PANELS["published"][0]]["branch_table"] = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="missing"):
        panel_table(short, "published")


def test_the_figure_is_a_default_vanilla_figure_of_the_four_graph_input() -> None:
    spec = get_spec(FIGURE_ID)
    assert spec.default is True  # manuscript source
    assert spec.inputs == frozenset({"four_graph_comparisons"})
    assert spec.variants == frozenset({"vanilla"})
    assert FIGURE_ID in FOUR_GRAPH_FIGURE_IDS and FIGURE_ID in NON_CANDIDATE_FIGURE_IDS
