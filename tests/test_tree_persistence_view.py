"""Story-view tree_persistence: saved-product adapter, death histogram, view and names."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest

from dubanevics2026topological.figures.story_views import population_title
from dubanevics2026topological.figures.view_naming import single_result_figure_name
from dubanevics2026topological.build.d10_tree_inputs import (
    SavedTreeInputs,
    persistence_table_from_pairs,
    scalar_domain_from_population,
    stop_marker_table,
)
from dubanevics2026topological.figures import tree_persistence_view as view
from dubanevics2026topological.results.labels import (
    STANDARD_PRODUCT_TYPES,
    MalformedLabelError,
)


def _pairs(**changes) -> pl.DataFrame:
    base = {
        "birth_label": ["A", "B"],
        "death_label": ["R", "R"],
        "birth_scalar": [0.9, 0.5],
        "death_scalar": [0.2, -0.1],
        "persistence": [0.7, 0.6],
        "pair_kind": ["ordinary", "ordinary"],
        "lineage_vertex_count": [3, 1],
    }
    return pl.DataFrame({**base, **changes})


def test_persistence_table_keeps_the_topographer_schema_columns() -> None:
    table = persistence_table_from_pairs(_pairs())
    assert table.columns == list(view_columns())
    assert table.schema["death_scalar"] == pl.Float64


def view_columns():
    from dubanevics2026topological.build.d10_tree_inputs import PERSISTENCE_COLUMNS

    return PERSISTENCE_COLUMNS


@pytest.mark.parametrize(
    "bad",
    [
        {"death_scalar": [float("nan"), 0.0]},
        {"birth_label": ["A", "A"]},
        {"death_scalar": [1.5, 0.0]},
    ],
)
def test_persistence_table_refuses_invalid_pairs(bad) -> None:
    with pytest.raises(ValueError, match="Persistence pairs|dies above"):
        persistence_table_from_pairs(_pairs(**bad))


def test_persistence_table_refuses_missing_columns() -> None:
    with pytest.raises(ValueError, match="lack columns"):
        persistence_table_from_pairs(_pairs().drop("pair_kind"))


def test_scalar_domain_is_the_population_extent() -> None:
    population = pl.DataFrame({"dna_sequence": ["A", "B", "C"], "scalar": [-1.0, 0, 2]})
    assert scalar_domain_from_population(population) == (-1.0, 2.0)


def test_stop_markers_come_from_the_layout_node_table() -> None:
    nodes = pl.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.1, 0.2],
            "dna_sequence": ["AAA", "TAA"],
            "aa_sequence": ["K", "*"],
            "has_stop_codon": [False, True],
            "display_event": ["maximum", "split_saddle"],
        }
    )
    tree = SimpleNamespace(plot_data=SimpleNamespace(node_table=nodes))
    table = stop_marker_table(tree)
    # Every stop node is kept with its kind; the figure layer picks the kinds drawn.
    assert table["dna_sequence"].to_list() == ["TAA"]
    assert table["display_event"].to_list() == ["split_saddle"]


def test_fitness_bins_count_every_sequence_once() -> None:
    bins = view.fitness_bins(pl.Series([-0.1, 0.0, 0.5, 1.0]), (-0.2, 1.0), 6)
    assert bins["count"].sum() == 4
    assert bins["bin_min"][0] == pytest.approx(-0.2)
    assert bins["bin_max"][-1] == pytest.approx(1.0)


def _names(label: str) -> dict[str, str]:
    from dubanevics2026topological.results import parse_label

    params = dict(parse_label(label, validate=False).params)
    return {
        style: single_result_figure_name("tree_persistence", params, style, "png")
        for style in view.STYLES
    }


def test_names_follow_the_tree_label_and_both_styles() -> None:
    names = _names("tree__thr=published_fdr=none_tree=nominal_hash=65d22b16")
    assert names == {
        "annotated": "tree_persistence__thr=published_fdr=none_tree=nominal"
        "_hash=65d22b16_style=annotated.png",
        "vanilla": "tree_persistence__thr=published_fdr=none_tree=nominal"
        "_hash=65d22b16_style=vanilla.png",
    }


def test_band_trees_have_no_tree_persistence_name() -> None:
    label = (
        "tree__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019"
        "_tree=band_bandq=0.95_hash=9b3f0432"
    )
    with pytest.raises(MalformedLabelError):
        _names(label)


def test_population_titles() -> None:
    assert population_title({"graph": "full"}) == "Full graph"
    assert (
        population_title({"thr": "reflected", "fdr": "none"}) == "Candidate threshold"
    )
    assert (
        population_title({"thr": "published", "fdr": "by", "q": 0.05})
        == "Published threshold, BY q = 0.05"
    )


@dataclass
class _PlotData:
    edge_table: pl.DataFrame
    node_table: pl.DataFrame


def _inputs() -> SavedTreeInputs:
    tree = SimpleNamespace(
        wild_type_marker_table=pl.DataFrame({"x": [0.0], "y": [0.0]}),
        all_stop_marker_table=pl.DataFrame(),
    )
    return SavedTreeInputs(
        tree_label="tree__graph=full_tree=nominal_hash=e6b7c141",
        tree_id="a" * 64,
        layout_id="b" * 64,
        landscape_label="landscape__graph=full_hash=7b651442",
        landscape_id="c" * 64,
        tree_analysis=tree,
        persistence_table=persistence_table_from_pairs(_pairs()),
        scalar_domain=(-0.2, 1.0),
        vertex_count=3,
        fitness_values=pl.Series([-0.1, 0.0, 0.5, 1.0]),
    )


def _patch_panels(monkeypatch, seen) -> None:
    edges = pl.DataFrame({"vertex_count": [1, 4]})
    monkeypatch.setattr(
        view,
        "plot_data_with_metrics",
        lambda tree: seen.append(tree) or _PlotData(
            edges, pl.DataFrame({"dna_sequence": ["A"], "has_stop_codon": [False]})
        ),
    )
    monkeypatch.setattr(
        view, "persistence_with_counts", lambda pairs, plot, domain: pairs
    )

    def panel(*args, **kwargs):
        return (
            alt.Chart(pl.DataFrame({"y": [0.0], "birth_scalar": [0.0]}))
            .mark_point()
            .encode(y="y:Q")
            .properties(width=50, height=50)
        )

    monkeypatch.setattr(view, "tree_chart", panel)
    monkeypatch.setattr(view, "overlay_count_colorbar", lambda tree, *a: tree)
    monkeypatch.setattr(view, "persistence_chart", panel)
    monkeypatch.setattr(view, "biology_axis_contract", lambda chart, **kw: chart)
    monkeypatch.setattr(view, "publication_scalar_scales", lambda chart, *a: chart)


SETTINGS = {"figures": {"d10": {"tree_persistence": {"histogram_height": 70}}}}


@pytest.mark.parametrize("style", view.STYLES)
def test_view_aligns_tree_and_diagram_below_equal_height_cells(
    monkeypatch, style
) -> None:
    seen: list = []
    _patch_panels(monkeypatch, seen)
    chart = view.build_tree_persistence_figure(
        _inputs(), SETTINGS, style, title="t", subtitle="s"
    )
    left, right = chart.hconcat
    assert left.vconcat[0].height == right.vconcat[0].height == 70
    assert has_wild_type(seen[0]) is (style == "annotated")


@pytest.mark.parametrize("style", view.STYLES)
def test_diagram_key_is_inside_the_diagram_not_a_row_below(monkeypatch, style) -> None:
    _patch_panels(monkeypatch, [])
    chart = view.build_tree_persistence_figure(
        _inputs(), SETTINGS, style, title="t", subtitle="s"
    )
    right = chart.hconcat[1]
    # Marginal histogram and diagram only: no legend row under the diagram.
    assert len(right.vconcat) == 2
    marks = [
        layer["mark"]
        for layer in right.vconcat[1].to_dict()["layer"]
        if layer.get("mark", {}).get("type") == "text"
    ]
    labels = [mark["text"] for mark in marks if mark["text"] not in {"▲", "▽", "★"}]
    assert labels == ["peak"]  # no valley entry
    ys = [mark["y"] for mark in marks if mark["text"] in labels]
    assert ys == sorted(ys) and len(set(ys)) == 1


def has_wild_type(tree_analysis) -> bool:
    return hasattr(tree_analysis, "wild_type_marker_table")


def test_view_refuses_an_unknown_style() -> None:
    with pytest.raises(ValueError, match="style must be one of"):
        view.build_tree_persistence_figure(
            _inputs(), SETTINGS, "bold", title="t", subtitle="s"
        )


def test_only_leaves_is_named_like_a_data_product() -> None:
    # `leaves` is a known wart: figure names skip the data-product contract check
    # (the presentation specification, figures/view_naming.py), so the clash is benign.
    from dubanevics2026topological.figures.view_naming import D10_VIEWS

    assert set(D10_VIEWS) & set(STANDARD_PRODUCT_TYPES) == {"leaves"}


def test_slide_mode_has_no_title_and_plain_coloring_has_no_colorbar(
    monkeypatch,
) -> None:
    _patch_panels(monkeypatch, [])
    colorbars: list = []
    monkeypatch.setattr(
        view, "overlay_count_colorbar", lambda tree, *a: colorbars.append(1) or tree
    )
    coloured = view.build_tree_persistence_figure(
        _inputs(), SETTINGS, "annotated", title=None, subtitle=None
    )
    assert "title" not in coloured.to_dict()
    assert colorbars == [1]
    plain = view.build_tree_persistence_figure(
        _inputs(), SETTINGS, "annotated", title=None, subtitle=None, coloring="none"
    )
    assert "title" not in plain.to_dict()
    assert colorbars == [1]


def test_view_refuses_an_unknown_coloring() -> None:
    with pytest.raises(ValueError, match="coloring must be one of"):
        view.build_tree_persistence_figure(
            _inputs(), SETTINGS, "annotated", title="t", subtitle="s", coloring="hue"
        )


def test_shared_scalar_domain_replaces_the_tree_extent(monkeypatch) -> None:
    _patch_panels(monkeypatch, [])
    domains: list = []
    monkeypatch.setattr(
        view,
        "publication_scalar_scales",
        lambda chart, domain, *a: domains.append(domain) or chart,
    )
    view.build_tree_persistence_figure(
        _inputs(),
        SETTINGS,
        "vanilla",
        title=None,
        subtitle=None,
        scalar_domain=(-1.2, 1.4),
    )
    assert domains == [(-1.2, 1.4)]
