from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import polars as pl
import pytest
import rustworkx as rx
from _figure_labels import SENTINEL, with_labels
from topographer.plotting import make_tree_layout

from dubanevics2026topological.figures._count_labels import (
    LABEL_COLOR,
    LEADER_WIDTH,
)
from dubanevics2026topological.figures.registry import get_spec, render_payload


def _walk_specs(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_specs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_specs(child)


def _ridge_tree():
    tree = rx.PyDiGraph()
    nodes = [
        {"dna_sequence": "SAD", "aa_sequence": "sad-aa", "scalar": -0.8, "has_stop_codon": False},
        {"dna_sequence": "KEM", "aa_sequence": "kem-aa", "scalar": -0.2, "has_stop_codon": False},
        {"dna_sequence": "K1A", "aa_sequence": "k1a-aa", "scalar": -0.3, "has_stop_codon": False},
        # Two stop-codon maxima: one on the KEM ridge's side branch, the TCL peak.
        {"dna_sequence": "K2A", "aa_sequence": "k2a-aa", "scalar": -0.25, "has_stop_codon": True},
        {"dna_sequence": "TCL", "aa_sequence": "tcl-aa", "scalar": 0.2, "has_stop_codon": True},
        {"dna_sequence": "WT1", "aa_sequence": "wt1-aa", "scalar": -0.1, "has_stop_codon": False},
    ]
    indices = [tree.add_node(node) for node in nodes]
    saddle, kem_peak, kem_branch, kem_sibling, tcl_peak, wt = indices
    # Tree direction runs from leaves toward the shared saddle/root.
    tree.add_edge(kem_peak, kem_branch, {})
    tree.add_edge(kem_sibling, kem_branch, {})
    tree.add_edge(kem_branch, saddle, {})
    tree.add_edge(tcl_peak, saddle, {})
    tree.add_edge(wt, saddle, {})
    return tree, (saddle, kem_peak, kem_branch, kem_sibling, tcl_peak, wt)


def _assignments():
    """Sequences on the toy tree's branches (``upper`` to ``lower``).

    The KEM branch holds two sequences, every other branch one; the saddle is the
    root group. These are the post-absorption counts the figure colours by.
    """
    rows = [
        ("KEM", -0.2, "KEM", "K1A"),
        ("KEMx", -0.25, "KEM", "K1A"),
        ("K2A", -0.25, "K2A", "K1A"),
        ("K1A", -0.3, "K1A", "SAD"),
        ("TCL", 0.2, "TCL", "SAD"),
        ("WT1", -0.1, "WT1", "SAD"),
        ("SAD", -0.8, "SAD", "SAD"),
    ]
    return pl.DataFrame(
        {
            "dna_sequence": [row[0] for row in rows],
            "raw_fitness": [row[1] for row in rows],
            "upper_sequence": [row[2] for row in rows],
            "lower_sequence": [row[3] for row in rows],
        }
    )


def _codon_payload(extra_thresholds=()):
    """Payload shaped like ``compose_payload`` output for the codon figure."""
    tree, (saddle, kem_peak, kem_branch, kem_sibling, tcl_peak, _) = _ridge_tree()
    plot_data = make_tree_layout(
        tree,
        scalar_attr="scalar",
        root_kind="minimum",
        node_attrs=("dna_sequence", "aa_sequence", "has_stop_codon"),
    )
    # The toy layout marks no critical points; the leaves are the maxima.
    plot_data = replace(
        plot_data,
        node_table=plot_data.node_table.with_columns(
            [
                pl.when(pl.col("dna_sequence").is_in(["KEM", "K2A", "TCL", "WT1"]))
                .then(pl.lit("maximum"))
                .otherwise(pl.col(column))
                .alias(column)
                for column in ("event", "display_event")
            ]
        ),
    )
    tree_analysis = SimpleNamespace(
        tree=tree,
        plot_data=plot_data,
        wild_type_marker_table=pl.DataFrame(
            {"x": [0.5], "y": [-0.1], "marker_label": ["WT"]}
        ),
    )
    candidate_view = SimpleNamespace(
        tree=tree,
        plot_data=plot_data,
        assignments=_assignments(),
        wild_type_marker_table=tree_analysis.wild_type_marker_table,
    )
    core = pl.DataFrame({"fitness": [-0.8, -0.4, -0.2, -0.1, 0.2]})
    kem = pl.DataFrame({"fitness": [-0.4, -0.3, -0.2]})
    tcl = pl.DataFrame({"fitness": [0.1, 0.2]})
    thresholds = sorted([-0.8, -0.4, -0.2, -0.1, 0.2, *extra_thresholds])
    entropy = {
        name: pl.DataFrame(
            {
                "threshold_scalar": thresholds,
                "aa_position": [1] * len(thresholds),
                "entropy_bits": [1.0] * len(thresholds),
                "active_sequence_count": [1] * len(thresholds),
            }
        )
        for name in ("core", "kem", "tcl")
    }
    comparison = with_labels(
        {
            "metadata": {
                "functional_threshold": -0.5,
                "summits": {
                    "kem": {"dna_sequence": "KEM"},
                    "tcl": {"dna_sequence": "TCL"},
                },
                "common_saddle": {"dna_sequence": "SAD"},
            },
            "df_core": core,
            "df_kem": kem,
            "df_tcl": tcl,
            "entropy": entropy,
            "candidate_leaf_view": candidate_view,
        }
    )
    settings = {
        "figures": {
            "rendering": {
                "fitness_tick_step": 0.1,
                "heading_font_size": 18,
                "subtitle_font_size": 14,
            },
            "figures": {
                "bootstrap_codon_entropy": {
                    "tree_width": 520,
                    "figure_width": 520,
                    "tree_height": 250,
                    "sweep_height": 250,
                    "tree_edge_width": 1.2,
                    "colorbar_x_fraction": 0.9,
                    "branch_count_label_mode": "tips",
                    "branch_count_label_max": 1,
                    "entropy_legend_width": 260,
                }
            },
        }
    }
    return (
        comparison,
        settings,
        tree,
        plot_data,
        saddle,
        kem_peak,
        kem_branch,
        kem_sibling,
        tcl_peak,
    )


def _render_codon(**figure_overrides):
    comparison, settings, *_ = _codon_payload()
    settings["figures"]["figures"]["bootstrap_codon_entropy"].update(
        figure_overrides
    )
    return render_payload(
        get_spec("bootstrap_codon_entropy"),
        comparison,
        settings,
        "annotated",
    ).to_dict()


@pytest.fixture(scope="module")
def rendered():
    """The toy figure in the tips mode at a maximum of one, rendered once."""
    return _render_codon()


def _tree_panel(result):
    return result["vconcat"][0]["hconcat"][0]


def _sweep_panels(result):
    """The three heat maps; the fourth hconcat item is the one entropy bar."""
    panels = result["vconcat"][1]["hconcat"]
    assert len(panels) == 4
    return panels[:3]


def _entropy_bar(result):
    return result["vconcat"][1]["hconcat"][3]


def _entropy_color(heat_spec):
    colors = [
        item["encoding"]["color"]
        for item in _walk_specs(heat_spec)
        if isinstance(item.get("encoding"), dict)
        and item["encoding"].get("color", {}).get("field") == "entropy_bits"
    ]
    assert len(colors) == 1
    return colors[0]


def _highlight_layers(tree_spec):
    """The solid-colour branch layers drawn over the count-coloured tree."""
    return [
        item
        for item in _walk_specs(tree_spec)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("type") == "rule"
        and isinstance(item["mark"].get("color"), str)
        and item["mark"].get("strokeWidth") == 3.0
        and isinstance(item.get("data"), dict)
    ]


def _rows(item, datasets):
    data = item["data"]
    return datasets[data["name"]] if "name" in data else data["values"]


def test_build_draws_the_tree_over_three_sweeps():
    result = _render_codon()
    assert result["resolve"]["scale"]["color"] == "independent"
    # Row 1 the tree, row 2 the three sweeps and, to their right, the one entropy bar.
    assert len(result["vconcat"]) == 2
    tree_row, sweep_row = result["vconcat"]
    assert len(tree_row["hconcat"]) == 1
    assert len(sweep_row["hconcat"]) == 4
    tree = _tree_panel(result)
    assert (tree["width"], tree["height"]) == (520, 250)
    sweeps = _sweep_panels(result)
    # Equal panels: with the two 8 px gaps they fill the tree's width, so the two rows
    # (each with the tree's 60 px fitness axis on its left) are the same width.
    assert {(panel["width"], panel["height"]) for panel in sweeps} == {(168, 250)}
    assert 3 * 168 + 2 * 8 <= tree["width"]
    assert sweep_row["spacing"] == 8
    assert "title" not in tree  # a tree-level title would widen the panel
    assert tree_row["title"]["text"] == (
        "Band-smoothed candidate graph · "
        f"{SENTINEL['candidate_graph.functional_count']} functional / "
        f"{SENTINEL['candidate_graph.vertex_count']} total sequences"
    )
    assert sweep_row["title"]["text"] == (
        f"Codon entropy · KEM ridge "
        f"{SENTINEL['ridges.band_view.kem.functional_count']} and TCL ridge "
        f"{SENTINEL['ridges.band_view.tcl.functional_count']} functional sequences"
    )
    assert _render_requested_keys() == (
        "candidate_graph.functional_count",
        "candidate_graph.vertex_count",
        "ridges.band_view.kem.functional_count",
        "ridges.band_view.tcl.functional_count",
    )
    assert result["config"]["axis"]["labelFontSize"] == 15
    assert result["config"]["legend"]["titleFontSize"] == 15
    assert result["title"]["orient"] == "bottom"  # the gray-region note stays


def _render_requested_keys():
    comparison, settings, *_ = _codon_payload()
    render_payload(
        get_spec("bootstrap_codon_entropy"), comparison, settings, "annotated"
    )
    return comparison["manuscript_quantities"].requested_keys


def test_the_sweeps_are_kem_tcl_and_all_codons_in_that_order(rendered):
    titles = [panel["title"] for panel in _sweep_panels(rendered)]
    assert [title["text"] for title in titles] == [
        "KEM branch",
        "TCL branch",
        "All branches",
    ]


def test_the_sweeps_share_one_fitness_axis_one_position_axis_and_one_colour_bar(
    rendered,
):
    sweeps = _sweep_panels(rendered)
    domains = []
    for index, panel in enumerate(sweeps):
        y_axes = [
            item["encoding"]["y"]["axis"]
            for item in _walk_specs(panel)
            if isinstance(item.get("encoding"), dict)
            and item["encoding"].get("y", {}).get("field")
            in {"threshold_lower", "lower"}
            and item["encoding"]["y"].get("axis")
        ]
        # The fitness axis is drawn on the first panel only, at the tree's axis width.
        if index == 0:
            assert y_axes
            assert all(axis["minExtent"] == axis["maxExtent"] == 60 for axis in y_axes)
        else:
            assert not y_axes
        domains.append(
            {
                tuple(item["encoding"]["y"]["scale"]["domain"])
                for item in _walk_specs(panel)
                if isinstance(item.get("encoding"), dict)
                and item["encoding"].get("y", {}).get("field")
                in {"threshold_lower", "lower"}
            }
        )
        color = _entropy_color(panel)
        assert color["scale"]["domain"] == [0.0, 4.523561956057013]
        # No encoding legend: one pixel bar serves all three sweeps.
        assert color["legend"] is None
    assert domains[0] == domains[1] == domains[2]
    ((low, high),) = domains[0]
    assert [low, high] == pytest.approx([-0.85, 0.25])
    # Position labels on every panel, the title written once, on the middle panel.
    titled = [
        [
            item["encoding"]["x"]["title"]
            for item in _walk_specs(panel)
            if isinstance(item.get("encoding"), dict)
            and item["encoding"].get("x", {}).get("field") == "position_label"
            and item["encoding"]["x"].get("title")
        ]
        for panel in sweeps
    ]
    assert titled[0] == [] and titled[2] == []
    assert titled[1] and set(titled[1]) == {"Amino-acid position"}
    # The one entropy bar: a vertical gradient (top is the maximum) with its title
    # and its end labels, to the right of all three sweeps.
    bar = _entropy_bar(rendered)
    assert bar["height"] == 250
    texts = [
        item["mark"].get("text")
        for item in _walk_specs(bar)
        if isinstance(item.get("mark"), dict) and item["mark"].get("type") == "text"
    ]
    assert {"Entropy", "(bits)", "0.0", "4.5"} <= set(texts)
    gradients = [
        item
        for item in _walk_specs(bar)
        if isinstance(item.get("mark"), dict)
        and isinstance(item["mark"].get("color"), dict)
        and "stops" in item["mark"]["color"]
    ]
    assert len(gradients) == 1
    gradient = gradients[0]["mark"]["color"]
    assert (gradient["x1"], gradient["x2"]) == (0, 0)
    assert (gradient["y1"], gradient["y2"]) == (0, 1)
    assert "Entropy" not in [
        item["mark"].get("text")
        for panel in sweeps
        for item in _walk_specs(panel)
        if isinstance(item.get("mark"), dict) and item["mark"].get("type") == "text"
    ]


def _config_figures():
    import tomllib
    from pathlib import Path

    return tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )


def test_the_figure_is_as_wide_as_the_tree_figure():
    figures = _config_figures()["figures"]
    own = figures["bootstrap_codon_entropy"]
    assert own["figure_width"] == own["tree_width"]
    assert own["figure_width"] == figures["tree_comparison"]["tree"]["width"]


def test_the_branch_colours_are_new_and_the_panels_are_tied_to_them(rendered):
    from dubanevics2026topological.figures import palette

    config = _config_figures()
    own = config["figures"]["bootstrap_codon_entropy"]
    kem, tcl = own["kem_color"], own["tcl_color"]
    assert (kem, tcl) == (palette.KEM_BRANCH_COLOR, palette.TCL_BRANCH_COLOR)
    assert kem != tcl
    # Used by no other table of the figures configuration, nor by the palette.
    tables = {
        name: table
        for name, table in config["figures"].items()
        if name != "bootstrap_codon_entropy"
    }
    text = str({**config, "figures": tables}).lower()
    assert kem.lower() not in text and tcl.lower() not in text
    other_palette = {
        value.lower()
        for name, value in vars(palette).items()
        if isinstance(value, str)
        and value.startswith("#")
        and name not in {"KEM_BRANCH_COLOR", "TCL_BRANCH_COLOR"}
    }
    assert kem.lower() not in other_palette and tcl.lower() not in other_palette
    # The KEM and TCL sweep panels carry their branch's colour in the title and on a rule.
    titles = [panel["title"] for panel in _sweep_panels(rendered)]
    assert [title["color"] for title in titles[:2]] == [kem, tcl]
    for panel, colour in zip(_sweep_panels(rendered)[:2], (kem, tcl), strict=True):
        rules = [
            item["mark"]
            for item in _walk_specs(panel)
            if isinstance(item.get("mark"), dict)
            and item["mark"].get("type") == "rule"
            and item["mark"].get("color") == colour
        ]
        assert len(rules) == 1
    assert titles[2]["color"] not in (kem, tcl)


def test_exactly_two_peak_branches_are_highlighted_in_their_own_colours(rendered):
    comparison, _settings, _tree, plot_data, saddle, kem_peak, kem_branch, _s, tcl_peak = (
        _codon_payload()
    )
    layers = _highlight_layers(_tree_panel(rendered))
    assert len(layers) == 2
    from dubanevics2026topological.figures import palette

    assert {layer["mark"]["color"] for layer in layers} == {
        palette.KEM_BRANCH_COLOR,
        palette.TCL_BRANCH_COLOR,
    }
    by_color = {layer["mark"]["color"]: layer for layer in layers}
    # Each runs from its peak down to the shared saddle, and nothing else: the KEM
    # branch takes its own edge and the edge on to the saddle (not its sibling), the
    # TCL branch the one edge that already ends at the saddle.
    expected = {
        palette.KEM_BRANCH_COLOR: [kem_peak, kem_branch],
        palette.TCL_BRANCH_COLOR: [tcl_peak],
    }
    segments = plot_data.edge_segment_table.to_dicts()
    for colour, targets in expected.items():
        rows = _rows(by_color[colour], rendered["datasets"])
        own = {
            row["logical_edge_index"]
            for row in segments
            if int(row["target_tree_node_index"]) in targets
        }
        assert own
        assert {row["logical_edge_index"] for row in rows} == own
        assert {int(row["target_tree_node_index"]) for row in rows} == set(targets)
        # The lowest end of the branch is the saddle's fitness, whichever peak.
        lowest = min(min(row["y_start"], row["y_end"]) for row in rows)
        assert lowest == pytest.approx(-0.8)
        assert saddle in {int(row["source_tree_node_index"]) for row in rows}


def test_peak_genotypes_are_labels_without_markers_and_the_saddle_is_a_filled_diamond(
    rendered,
):
    from dubanevics2026topological.figures import palette

    datasets = rendered["datasets"]
    tree = _tree_panel(rendered)
    points = [
        item
        for item in _walk_specs(tree)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("type") == "point"
        and item["mark"].get("shape") in {"triangle-up", "diamond"}
    ]
    shapes = sorted(item["mark"]["shape"] for item in points)
    # No peak triangle: only the saddle's diamond, filled; no valleys.
    assert shapes == ["diamond"]
    assert all(item["mark"]["filled"] is True for item in points)
    assert "triangle-down" not in str(rendered)
    assert "triangle-up" not in str(_tree_panel(rendered))
    labels = {
        item["mark"]["color"]: item
        for item in _walk_specs(tree)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("type") == "text"
        and item.get("encoding", {}).get("text", {}).get("field") == "label"
    }
    # A peak's amino-acid genotype above its marker, in the branch's colour.
    for colour, aa in (
        (palette.KEM_BRANCH_COLOR, "kem-aa"),
        (palette.TCL_BRANCH_COLOR, "tcl-aa"),
    ):
        item = labels[colour]
        assert item["mark"]["baseline"] == "bottom" and item["mark"]["dy"] < 0
        assert [row["label"] for row in _rows(item, datasets)] == [aa]
    # The saddle's genotype to the left of its diamond and a little below it.
    (saddle,) = [
        item for colour, item in labels.items() if colour == palette.INK_GREY
    ]
    assert saddle["mark"]["align"] == "right"
    assert saddle["mark"]["dx"] < 0 and saddle["mark"]["dy"] > 0
    assert [row["label"] for row in _rows(saddle, datasets)] == ["sad-aa"]
    # Sequence identities travel with every marker; no tree index is exposed.
    for item in (*points, *labels.values()):
        rows = _rows(item, datasets)
        assert rows and all("dna_sequence" in row for row in rows)
        assert not any("tree_node_index" in row for row in rows)


def test_a_peak_missing_from_the_tree_is_named_by_its_sequence():
    comparison, settings, *_ = _codon_payload()
    comparison["metadata"]["summits"]["kem"] = {"dna_sequence": "NOPE"}
    with pytest.raises(ValueError, match="NOPE"):
        render_payload(
            get_spec("bootstrap_codon_entropy"), comparison, settings, "annotated"
        )


def test_heatmap_clips_rows_to_population_fitness_endpoints():
    comparison, settings, *_ = _codon_payload(extra_thresholds=(-0.95, 0.3))
    result = render_payload(
        get_spec("bootstrap_codon_entropy"),
        comparison,
        settings,
        "annotated",
    ).to_dict()
    all_codons = _sweep_panels(result)[2]
    values = next(
        item["data"]["values"]
        for item in _walk_specs(all_codons)
        if isinstance(item.get("encoding"), dict)
        and item["encoding"].get("color", {}).get("field") == "entropy_bits"
    )
    outside = [r for r in values if not -0.8 <= r["threshold_scalar"] <= 0.2]
    inside = [r for r in values if -0.8 <= r["threshold_scalar"] <= 0.2]
    assert {r["threshold_scalar"] for r in outside} == {-0.95, 0.3}
    assert all("threshold_lower" not in r for r in outside)
    assert min(r["threshold_lower"] for r in inside) == -0.8
    assert max(r["threshold_upper"] for r in inside) == 0.2


def _label_layers(node, datasets):
    """Return ``(mark, rows)`` of every count-label layer under ``node``.

    A count label is peak-grey text; the wild-type label is red, a genotype label is
    its branch's colour and the colour bar's text carries no colour of its own.
    """
    found = []
    for item in _walk_specs(node):
        mark = item.get("mark")
        if (
            isinstance(mark, dict)
            and mark.get("type") == "text"
            and mark.get("color") == "#737373"
            and isinstance(item.get("data"), dict)
        ):
            data = item["data"]
            found.append(
                (mark, datasets[data["name"]] if "name" in data else data["values"])
            )
    return found


def test_count_labels_mark_the_small_tips_in_peak_grey(rendered):
    datasets = rendered["datasets"]
    (tip,) = _label_layers(_tree_panel(rendered), datasets)
    mark, rows = tip
    assert mark["fontSize"] == 13  # the figure's body font less two
    assert mark["opacity"] == 1.0
    # The tips mode at a maximum of one: the three one-sequence tips are labelled,
    # the KEM tip (two sequences) is not; no amino-acid name in the text.
    assert sorted((row["dna_sequence"], row["label"]) for row in rows) == [
        ("K2A", "1"),
        ("TCL", "1"),
        ("WT1", "1"),
    ]
    # The sweeps carry no branch count labels.
    for panel in _sweep_panels(rendered):
        assert not _label_layers(panel, datasets)


def test_the_tip_labels_follow_the_maximum():
    quiet = _render_codon(branch_count_label_max=0)  # no tip is at most zero
    assert not _label_layers(_tree_panel(quiet), quiet["datasets"])
    wide = _render_codon(branch_count_label_max=2)
    ((_, rows),) = _label_layers(_tree_panel(wide), wide["datasets"])
    assert sorted(row["dna_sequence"] for row in rows) == ["K2A", "KEM", "TCL", "WT1"]


def test_the_all_mode_labels_every_branch_of_the_tree_once():
    everything = _render_codon(branch_count_label_mode="all")
    ((_, rows),) = _label_layers(_tree_panel(everything), everything["datasets"])
    # Every branch of the band tree once: the four tips and the internal branch K1A.
    assert sorted((row["dna_sequence"], row["label"]) for row in rows) == [
        ("K1A", "1"),
        ("K2A", "1"),
        ("KEM", "2"),
        ("TCL", "1"),
        ("WT1", "1"),
    ]


def test_the_rotate_variant_turns_every_count_label_and_keeps_the_branches():
    turned = _render_codon(
        branch_count_label_mode="all", branch_count_label_collision="rotate"
    )
    plain = _render_codon(branch_count_label_mode="all")
    layers = _label_layers(turned, turned["datasets"])
    assert len(layers) == len(_label_layers(plain, plain["datasets"])) > 0
    assert all(mark["angle"] == 270 for mark, _ in layers)
    assert not any(
        "angle" in mark for mark, _ in _label_layers(plain, plain["datasets"])
    )
    # The same branches carry the labels; only their orientation and spot change.
    assert sorted(row["dna_sequence"] for _, rows in layers for row in rows) == sorted(
        row["dna_sequence"]
        for _, rows in _label_layers(plain, plain["datasets"])
        for row in rows
    )


def _mark_order(node):
    """Index of each mark in the order Vega draws them (first drawn first)."""
    return [
        item["mark"] for item in _walk_specs(node) if isinstance(item.get("mark"), dict)
    ]


def test_layers_stack_branches_highlights_labels_crosses_and_the_star(rendered):
    items = [
        item
        for item in _walk_specs(_tree_panel(rendered))
        if isinstance(item.get("mark"), dict)
    ]
    marks = [item["mark"] for item in items]
    plain = [
        i
        for i, item in enumerate(items)
        if item["mark"].get("type") == "rule"
        and item.get("encoding", {}).get("color", {}).get("field") == "vertex_count"
    ]
    highlights = [
        i
        for i, mark in enumerate(marks)
        if mark.get("type") == "rule" and mark.get("strokeWidth") == 3.0
    ]
    count_label = next(
        index
        for index, mark in enumerate(marks)
        if mark.get("type") == "text" and mark.get("color") == "#737373"
    )
    star = next(
        index
        for index, mark in enumerate(marks)
        if mark.get("type") == "point" and mark.get("color") == "#d62728"
    )
    cross = [
        i
        for i, mark in enumerate(marks)
        if mark.get("type") == "point" and mark.get("shape") == STOP_CROSS_SHAPE
    ]
    assert plain and len(highlights) == 2 and cross
    # Highlights sit over the plain branches, under the labels; the stop crosses sit
    # over the labels and markers; nothing hides the star.
    assert max(plain) < min(highlights)
    assert max(highlights) < count_label < min(cross)
    assert max(cross) < star


def test_only_one_count_colour_bar_and_nothing_is_clipped(rendered):
    bars = [
        item
        for item in _walk_specs(rendered)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("text") == "Branch vertex count"
    ]
    assert len(bars) == 1
    marks = [
        item["mark"]
        for item in _walk_specs(_tree_panel(rendered))
        if isinstance(item.get("mark"), dict)
    ]
    assert marks
    assert all(mark.get("clip") is not True for mark in marks)
    assert all(
        mark.get("strokeWidth", 1.2) in {1.2, 3.0}
        for mark in marks
        if mark.get("type") in {"line", "rule"}
        # a count label's thin leader line is not a branch
        and not (
            mark.get("color") == LABEL_COLOR and mark.get("strokeWidth") == LEADER_WIDTH
        )
    )


def _key_texts(node):
    return [
        item["mark"]["text"]
        for item in _walk_specs(node)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("type") == "text"
        and item["mark"].get("text") in {"Wild type", "Stop codon containing peaks"}
    ]


def test_the_figure_has_one_tree_key_and_the_tree_axis_has_the_shared_width(rendered):
    assert sorted(_key_texts(rendered)) == ["Stop codon containing peaks", "Wild type"]
    assert sorted(_key_texts(_tree_panel(rendered))) == [
        "Stop codon containing peaks",
        "Wild type",
    ]
    extents = {
        (
            item["encoding"]["y"]["axis"].get("minExtent"),
            item["encoding"]["y"]["axis"].get("maxExtent"),
        )
        for item in _walk_specs(_tree_panel(rendered))
        if isinstance(item.get("encoding"), dict)
        and isinstance(item["encoding"].get("y"), dict)
        and item["encoding"]["y"].get("axis")
    }
    assert extents == {(60, 60)}
    y_encodings = [
        item["encoding"]["y"]
        for item in _walk_specs(_tree_panel(rendered))
        if isinstance(item.get("encoding"), dict)
        and isinstance(item["encoding"].get("y"), dict)
        and item["encoding"]["y"].get("axis")
    ]
    assert y_encodings[0]["axis"]["values"] == [-0.8, 0.0, 0.2]
    assert y_encodings[0]["scale"]["domain"] == pytest.approx([-0.85, 0.25])


# Vega-Lite spells the ``x`` point shape as this path.
STOP_CROSS_SHAPE = "M-1,-1L1,1M-1,1L1,-1"


def _stop_layer_marks(node):
    """Return the stop-cross layers (crimson, unfilled ``x`` points) under ``node``."""
    return [
        item
        for item in _walk_specs(node)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("type") == "point"
        and item["mark"].get("shape") == STOP_CROSS_SHAPE
    ]


def test_the_tree_crosses_hold_every_stop_maximum(rendered):
    (layer,) = _stop_layer_marks(_tree_panel(rendered))
    data = layer["data"]
    rows = rendered["datasets"][data["name"]] if "name" in data else data["values"]
    assert {row["dna_sequence"] for row in rows} == {"K2A", "TCL"}


def test_the_shared_config_labels_only_the_small_tips():
    config = _config_figures()
    assert config["rendering"]["branch_count_label_mode"] == "tips"
    assert config["rendering"]["branch_count_label_max"] == 9
    # No figure overrides the one global setting.
    for name, figure in config["figures"].items():
        assert "branch_count_label_mode" not in figure, name


def test_the_untitled_twin_keeps_the_two_row_headings_and_the_note():
    from dubanevics2026topological.figures._title_flag import without_headings

    comparison, settings, *_ = _codon_payload()
    chart = render_payload(
        get_spec("bootstrap_codon_entropy"), comparison, settings, "annotated"
    )
    plain, removed = without_headings(chart)
    assert removed == 0  # no figure-level heading: the root title is the bottom note
    result = plain.to_dict()
    assert len(result["vconcat"]) == 2
    # the Candidate and sweep row headings (with counts) are content and stay
    assert all("title" in row for row in result["vconcat"][:2])
    assert result["vconcat"][0]["title"]["text"].startswith("Band-smoothed")
    assert result["title"]["orient"] == "bottom"  # the gray-region note stays
    assert chart.to_dict()["vconcat"][0]["title"]["text"].startswith("Band-smoothed")


def test_the_figure_converts_to_png_through_vl_convert():
    """Vega-Lite's normalizer crashed on a rule layer that had no encoding."""
    import vl_convert as vlc

    comparison, settings, *_ = _codon_payload()
    chart = render_payload(
        get_spec("bootstrap_codon_entropy"), comparison, settings, "annotated"
    )
    png = vlc.vegalite_to_png(chart.to_dict())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_vertical_entropy_bar_is_half_the_box_height_and_top_aligned():
    from dubanevics2026topological.figures.codon_entropy import vertical_entropy_legend

    def bar(fraction):
        legend = vertical_entropy_legend(
            entropy_max=4.5,
            scheme="viridis",
            height=200,
            bar_width=12,
            width=56,
            title=["Entropy", "(bits)"],
            font_size=15,
            bar_fraction=fraction,
        ).to_dict()
        (rect,) = [
            layer["mark"] for layer in legend["layer"] if layer["mark"]["type"] == "rect"
        ]
        return rect

    half, full = bar(0.5), bar(1.0)
    assert half["height"] == 100
    # Top-aligned: the bar's top edge is the same whatever its height.
    assert half["y"] - half["height"] / 2 == full["y"] - full["height"] / 2
    assert full["height"] < 200  # capped at the room under the title
