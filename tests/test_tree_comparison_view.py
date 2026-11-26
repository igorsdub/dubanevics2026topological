"""Story-view comparison views: declarations, panel resolution, layout and hashing."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest

from dubanevics2026topological.figures.story_views import (
    comparison_specification,
    panel_title,
    resolve_panel_labels,
    split_comparison_id,
)
from dubanevics2026topological.figures import tree_persistence_view
from dubanevics2026topological.figures.view_naming import (
    D10_VIEWS,
    comparison_hash,
)
from dubanevics2026topological.figures.tree_comparison_view import (
    ComparisonPanel,
    grid_rows,
    union_scalar_domain,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "tree__graph=full_tree=nominal_hash=e6b7c141",
    "tree__thr=published_fdr=none_tree=nominal_hash=65d22b16",
    "tree__thr=reflected_fdr=none_tree=nominal_hash=dbdd8df3",
    "tree__thr=published_fdr=by_q=0.05_tree=nominal_hash=53e9bb63",
    "tree__thr=reflected_fdr=by_q=0.05_tree=nominal_hash=d338cf99",
    "tree__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019"
    "_tree=band_bandq=0.95_hash=9b3f0432",
    "tree__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019"
    "_tree=band_bandq=0.95_hash=554140c0",
)


def _declared() -> dict:
    with (ROOT / "config/figures.toml").open("rb") as stream:
        return tomllib.load(stream)["comparisons"]


def test_every_declared_comparison_names_a_comparison_view() -> None:
    for comparison_id in _declared():
        view, name = split_comparison_id(comparison_id)
        assert D10_VIEWS[view].kind == "comparison"
        assert name.isalnum()


def test_split_refuses_other_views() -> None:
    with pytest.raises(ValueError):
        split_comparison_id("tree_persistence_main")


@pytest.mark.parametrize(
    ("comparison_id", "expected"),
    [
        (
            "tree_comparison_thresholds_main",
            [LABELS[0], LABELS[1], LABELS[2]],
        ),
        (
            "tree_comparison_fdr_main",
            [LABELS[1], LABELS[3], LABELS[2], LABELS[4]],
        ),
        ("tree_comparison_band_main", [LABELS[5], LABELS[6]]),
    ],
)
def test_declared_panels_resolve_in_declared_order(comparison_id, expected) -> None:
    assert resolve_panel_labels(_declared()[comparison_id], LABELS) == expected


def test_a_panel_must_match_exactly_one_tree() -> None:
    comparison = {"tree_state": "nominal", "panels": [{"thr": "published"}]}
    with pytest.raises(ValueError, match="exactly one"):
        resolve_panel_labels(comparison, LABELS)
    ambiguous = {"tree_state": "nominal", "panels": [{"graph": "full"}]}
    with pytest.raises(ValueError, match="exactly one"):
        resolve_panel_labels(
            ambiguous, (LABELS[0], LABELS[0].replace("e6b7c141", "e6b7c142"))
        )


def test_nominal_state_never_picks_a_band_tree() -> None:
    comparison = {
        "tree_state": "nominal",
        "panels": [{"thr": "published", "fdr": "by", "q": 0.05}],
    }
    assert resolve_panel_labels(comparison, LABELS) == [LABELS[3]]


def test_panel_titles_name_the_band_state() -> None:
    assert (
        panel_title({"thr": "published", "fdr": "by", "q": 0.05, "tree": "band"})
        == "Papkou + FDR, bootstrapped"
    )
    assert panel_title({"graph": "full", "tree": "nominal"}) == "Full"


def test_grid_layout_per_view() -> None:
    charts = list("abcd")
    assert grid_rows("tree_comparison_thresholds", charts) == [
        ["a"],
        ["b"],
        ["c"],
        ["d"],
    ]
    assert grid_rows("tree_comparison_fdr", charts) == [["a", "b"], ["c", "d"]]
    assert grid_rows("tree_comparison_band", charts[:2]) == [["a", "b"]]
    with pytest.raises(ValueError):
        grid_rows("tree_comparison_fdr", charts[:3])


def test_union_domain_covers_every_panel() -> None:
    panels = [
        ComparisonPanel("a", "", SimpleNamespace(scalar_domain=(-1.0, 0.5))),
        ComparisonPanel("b", "", SimpleNamespace(scalar_domain=(-0.5, 2.0))),
    ]
    assert union_scalar_domain(panels) == (-1.0, 2.0)


def test_hash_depends_on_input_order_and_declaration_not_appearance() -> None:
    comparison = _declared()["tree_comparison_band_main"]
    spec = comparison_specification("tree_comparison_band", "main", comparison)
    a, b = "a" * 64, "b" * 64
    assert comparison_hash([a, b], spec) != comparison_hash([b, a], spec)
    changed = comparison_specification(
        "tree_comparison_band", "main", {**comparison, "tree_state": "nominal"}
    )
    assert comparison_hash([a, b], spec) != comparison_hash([a, b], changed)
    assert comparison_hash([a, b], spec) == comparison_hash([a, b], dict(spec))


def test_fitness_histogram_counts_every_sequence() -> None:
    fitness = pl.Series([0.1, 0.2, 5.0])
    with pytest.raises(ValueError, match="count every sequence"):
        # 5.0 lies outside the fitness extent, so it would go uncounted.
        tree_persistence_view.fitness_histogram_chart(fitness, (0.0, 1.0), {}, {})


def test_comparison_band_wires_peak_labels_into_the_tree(monkeypatch) -> None:
    from dubanevics2026topological.figures import tree_comparison_view as view

    seen: list = []

    def fake_tree_chart(*args, **kwargs):
        seen.append(kwargs.get("count_labels"))
        return alt.Chart(pl.DataFrame({"x": [0]})).mark_point().encode(x="x:Q")

    monkeypatch.setattr(view, "tree_chart", fake_tree_chart)
    monkeypatch.setattr(
        view,
        "plot_data_with_metrics",
        lambda analysis: SimpleNamespace(
            edge_table=None, edge_segment_table=pl.DataFrame()
        ),
    )
    monkeypatch.setattr(view, "count_domain", lambda edges: (1, 3))
    monkeypatch.setattr(
        view,
        "overlay_count_colorbar",
        lambda tree, settings, rendering, domain, **kw: tree,
    )
    peaks = pl.DataFrame(
        {"x": [1.0], "y": [0.5], "aa_sequence": ["KEM"], "dna_sequence": ["AAAGAAATG"]}
    )

    def inputs():
        return SimpleNamespace(
            scalar_domain=(0.0, 1.0), threshold=None, tree_analysis=SimpleNamespace()
        )

    panels = [
        view.ComparisonPanel("a", "", inputs(), peak_labels=peaks),
        view.ComparisonPanel("b", "", inputs()),
    ]
    settings = {"figures": {"rendering": {}, "d10": {"tree_comparison": {}}}}
    view.build_tree_comparison_figure(
        "tree_comparison_band", panels, settings, title="t", subtitle="s"
    )
    assert len(seen[0]) == 1  # the labelled panel carries its identity layer
    assert seen[1] == []  # the unlabelled one carries none


def test_each_comparison_panel_draws_only_its_own_threshold_as_a_dashed_rule(
    monkeypatch,
) -> None:
    """One dashed rule, no text, per thresholded panel; its value is a 3-decimal tick."""
    from dubanevics2026topological.figures import tree_comparison_view as view

    underlays: list = []
    labels: list = []

    def fake_tree_chart(*args, **kwargs):
        underlays.append(kwargs.get("underlays"))
        labels.append(kwargs.get("count_labels"))
        return (
            alt.Chart(pl.DataFrame({"x": [0], "y": [0.0]}))
            .mark_point()
            .encode(x="x:Q", y="y:Q")
        )

    monkeypatch.setattr(view, "tree_chart", fake_tree_chart)
    monkeypatch.setattr(
        view,
        "plot_data_with_metrics",
        lambda analysis: SimpleNamespace(
            edge_table=None, edge_segment_table=pl.DataFrame()
        ),
    )
    monkeypatch.setattr(view, "count_domain", lambda edges: (1, 3))
    monkeypatch.setattr(
        view,
        "overlay_count_colorbar",
        lambda tree, settings, rendering, domain, **kw: tree,
    )

    def inputs(threshold, label):
        return SimpleNamespace(
            scalar_domain=(-1.15, 1.35),
            threshold=threshold,
            tree_label=label,
            tree_analysis=SimpleNamespace(),
        )

    panels = [
        view.ComparisonPanel("full", "", inputs(None, "tree__graph=full")),
        view.ComparisonPanel(
            "pub", "", inputs(-0.507774, "tree__thr=published_fdr=none")
        ),
        view.ComparisonPanel(
            "cand", "", inputs(-0.253159, "tree__thr=reflected_fdr=none")
        ),
    ]
    settings = {"figures": {"rendering": {}, "d10": {"tree_comparison": {}}}}
    spec = view.build_tree_comparison_figure(
        "tree_comparison_thresholds", panels, settings, title="t", subtitle="s"
    ).to_dict()

    # The Full panel has no threshold; the others draw one rule each, never both.
    assert [len(layers) for layers in underlays] == [0, 1, 1]
    assert labels == [[], [], []]
    for (rule,) in (layers for layers in underlays if layers):
        mark = rule.mark.to_dict()
        assert mark["type"] == "rule" and mark["strokeDash"] == [5, 4]
    drawn = [layers[0].data["functional_threshold"][0] for layers in underlays if layers]
    assert drawn == [-0.507774, -0.253159]
    assert "Functional threshold" not in str(spec)

    # Three regular ticks (the outward-rounded domain ends and 0), plus the panel's
    # own threshold with three decimals.
    expected = [[-1.2, 0.0, 1.4], [-1.2, -0.507774, 0.0, 1.4], [-1.2, -0.253159, 0.0, 1.4]]
    for row, values in zip(spec["vconcat"], expected, strict=True):
        # The first row is hconcat([hconcat(panels), colour bar]); descend to the panel.
        panel = row
        while "hconcat" in panel:
            panel = panel["hconcat"][0]
        axis = panel["encoding"]["y"]["axis"]
        assert axis["values"] == pytest.approx(values)
        assert axis["format"] == ".1f"
        assert ("labelExpr" in axis) == (len(values) == 4)


def _y_axes(spec) -> list[dict]:
    found: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            y = node.get("encoding", {}).get("y")
            if isinstance(y, dict) and y.get("field") == "y":
                found.append(y.get("axis") or {})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return found


def test_publication_scales_can_pin_every_y_axis_to_one_extent() -> None:
    from dubanevics2026topological.figures._charts import publication_scalar_scales

    frame = pl.DataFrame({"x": [0.0, 1.0], "y": [-0.5, 0.4]})
    panels = [
        publication_scalar_scales(
            alt.Chart(frame).mark_line().encode(x="x:Q", y="y:Q"),
            (-0.5, 0.4),
            {},
            extra_ticks=extra,
            exact_domain=True,
            y_axis_extent=60,
        )
        for extra in ({}, {"y": [-0.508]})
    ]
    axes = [axis for panel in panels for axis in _y_axes(panel.to_dict())]
    assert len(axes) == 2
    assert all(a["minExtent"] == a["maxExtent"] == 60 for a in axes)


def test_comparison_view_puts_the_colorbar_inside_the_top_right_panel(monkeypatch) -> None:
    from dubanevics2026topological.figures import tree_comparison_view as view

    placed: list[tuple[str, int]] = []

    def fake_tree_chart(*args, **kwargs):
        return alt.Chart(pl.DataFrame({"x": [0], "y": [0.0]})).mark_point().encode(
            x="x:Q", y="y:Q"
        )

    def fake_overlay(tree, settings, rendering, domain, **kwargs):
        placed.append(("overlay", kwargs["width"]))
        return tree

    monkeypatch.setattr(view, "tree_chart", fake_tree_chart)
    monkeypatch.setattr(view, "overlay_count_colorbar", fake_overlay)
    monkeypatch.setattr(
        view,
        "plot_data_with_metrics",
        lambda analysis: SimpleNamespace(
            edge_table=None, edge_segment_table=pl.DataFrame()
        ),
    )
    monkeypatch.setattr(view, "count_domain", lambda edges: (1, 3))

    def inputs():
        return SimpleNamespace(
            scalar_domain=(0.0, 1.0), threshold=None, tree_analysis=SimpleNamespace()
        )

    # Every comparison view: the one bar goes on exactly one panel, the last of the first row.
    for name, count in (
        ("tree_comparison_band", 2),
        ("tree_comparison_fdr", 4),
        ("tree_comparison_thresholds", 3),
    ):
        placed.clear()
        panels = [view.ComparisonPanel(str(i), "", inputs()) for i in range(count)]
        settings = {
            "figures": {
                "rendering": {},
                "d10": {"tree_comparison": {"colorbar_x_fraction": 0.9}},
            }
        }
        view.build_tree_comparison_figure(name, panels, settings, title="t", subtitle="s")
        assert len(placed) == 1, name


def test_comparison_view_has_no_outside_colorbar() -> None:
    import inspect

    from dubanevics2026topological.figures import _charts
    from dubanevics2026topological.figures import tree_comparison_view as view

    source = inspect.getsource(view.build_tree_comparison_figure)
    assert "outside_count_colorbar" not in source
    assert not hasattr(_charts, "outside_count_colorbar")
    assert "y_axis_extent=y_axis_extent" in source
    with (ROOT / "config/figures.toml").open("rb") as stream:
        appearance = tomllib.load(stream)["d10"]["tree_comparison"]
    assert appearance["tree_y_axis_extent"] > 0
    assert "colorbar_outside_gap" not in appearance
    assert 0 < appearance["colorbar_x_fraction"] < 1


def test_band_nominal_row_keeps_its_fitness_axis(monkeypatch) -> None:
    from dubanevics2026topological.figures import bootstrap_band_view as view

    seen: dict = {}

    def fake_layer(*args, **kwargs):
        seen.update(kwargs)
        return alt.Chart(pl.DataFrame({"a": [1]})).mark_rule()

    monkeypatch.setattr(view, "peak_branch_layer", fake_layer)
    monkeypatch.setattr(view, "peak_branch_segments", lambda *a, **k: None)
    monkeypatch.setattr(view, "plot_data_with_metrics", lambda analysis: SimpleNamespace(edge_table=None))
    monkeypatch.setattr(view, "count_domain", lambda table: (1.0, 2.0))
    monkeypatch.setattr(
        view, "tree_chart", lambda *a, **k: alt.Chart(pl.DataFrame({"a": [1]})).mark_line()
    )
    monkeypatch.setattr(view, "overlay_count_colorbar", lambda tree, *a, **k: tree)
    inputs = SimpleNamespace(tree_analysis=SimpleNamespace())
    absorbed = pl.DataFrame({"dna_sequence": ["AAA"]})
    view._tree_panel(inputs, {}, {}, (-1.0, 1.0), False, absorbed)
    assert seen["keep_fitness_axis"] is True
    with (ROOT / "config/figures.toml").open("rb") as stream:
        assert tomllib.load(stream)["d10"]["bootstrap_band"]["tree_y_axis_extent"] > 0


def _band_view_stubs(monkeypatch):
    """Stub the band view's collaborators; return the call records."""
    from dubanevics2026topological.figures import bootstrap_band_view as view

    calls: dict = {"panels": [], "legends": 0}
    mark = lambda: alt.Chart(pl.DataFrame({"a": [1]})).mark_line()  # noqa: E731

    def fake_panel(item, *args, keyed=True, **kwargs):
        calls["panels"].append(keyed)
        return mark(), SimpleNamespace()

    def fake_legend(*args, **kwargs):
        calls["legends"] += 1
        return [mark()]

    monkeypatch.setattr(view, "_tree_panel", fake_panel)
    monkeypatch.setattr(view, "_diagram", lambda *a, **k: mark())
    monkeypatch.setattr(view, "persistence_legend_layers", fake_legend)
    monkeypatch.setattr(view, "persistence_panel_size", lambda *a: (10, 10))
    monkeypatch.setattr(view, "wild_type_identities", lambda analysis: frozenset())
    monkeypatch.setattr(view, "peak_nodes", lambda table, names: pl.DataFrame())
    monkeypatch.setattr(view, "publication_scalar_scales", lambda chart, *a, **k: chart)
    return view, calls


def _band_view_inputs():
    plot_data = SimpleNamespace(
        node_table=None, edge_table=SimpleNamespace(height=3)
    )
    item = SimpleNamespace(
        scalar_domain=(-1.0, 1.0),
        tree_analysis=SimpleNamespace(plot_data=plot_data),
        persistence_table=pl.DataFrame({"birth_label": ["a"]}),
    )
    survival = SimpleNamespace(
        nominal_peaks=4, surviving=["a"], absorbed=["b"], two_c=0.1
    )
    settings = {"figures": {"d10": {"bootstrap_band": {}}, "rendering": {}}}
    return item, survival, settings


def test_band_view_keys_appear_once_on_the_first_row(monkeypatch) -> None:
    view, calls = _band_view_stubs(monkeypatch)
    item, survival, settings = _band_view_inputs()
    view.build_bootstrap_band_figure(
        item, item, survival, settings, "annotated", title="T", subtitle="S"
    )
    # Two rows: the count colour bar and tree key (keyed) and the diagram key on row 1 only.
    assert calls["panels"] == [True, False]
    assert calls["legends"] == 1


def test_band_view_slide_mode_also_keys_the_first_row_only(monkeypatch) -> None:
    view, calls = _band_view_stubs(monkeypatch)
    item, survival, settings = _band_view_inputs()
    view.build_bootstrap_band_figure(
        item, item, survival, settings, "vanilla", title=None, subtitle=None
    )
    assert calls["panels"] == [True, False]
    assert calls["legends"] == 1


def test_band_tree_panel_without_key_has_no_colorbar_and_no_tree_key(monkeypatch) -> None:
    from dubanevics2026topological.figures import bootstrap_band_view as view

    seen: list = []
    colorbars: list = []
    monkeypatch.setattr(view, "plot_data_with_metrics", lambda analysis: SimpleNamespace(edge_table=None))
    monkeypatch.setattr(view, "count_domain", lambda table: (1.0, 2.0))

    def fake_tree(*args, **kwargs):
        seen.append(kwargs["annotation_key"])
        return alt.Chart(pl.DataFrame({"a": [1]})).mark_line()

    monkeypatch.setattr(view, "tree_chart", fake_tree)
    monkeypatch.setattr(
        view, "overlay_count_colorbar", lambda tree, *a, **k: colorbars.append(1) or tree
    )
    inputs = SimpleNamespace(tree_analysis=SimpleNamespace())
    view._tree_panel(inputs, {}, {}, (-1.0, 1.0), False, None, keyed=True)
    view._tree_panel(inputs, {}, {}, (-1.0, 1.0), False, None, keyed=False)
    assert seen == [True, False]
    assert colorbars == [1]
