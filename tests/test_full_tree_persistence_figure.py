"""Full-graph tree, persistence and histogram figure composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import altair as alt
import polars as pl
import pytest
import vl_convert as vlc
from _figure_labels import SENTINEL, label_quantities, spec_text

from dubanevics2026topological.figures import palette
from dubanevics2026topological.figures._charts import padded_scalar_domain
from dubanevics2026topological.figures.full_tree_persistence import (
    topological_noise_band,
)
from dubanevics2026topological.figures.registry import (
    compose_payload,
    get_spec,
    render_payload,
)
from dubanevics2026topological.landscape.selection import (
    candidate_threshold_from_summary,
)
from dubanevics2026topological.threshold.functional_threshold import (
    MEDIAN_REFLECTED,
    comparison_declaration,
    compute_candidate_functional_threshold,
)

FIGURE_ID = "full_tree_persistence"
VERTICAL_ID = "full_tree_persistence_vertical_histogram"
DECLARATION = {"rule": "median_reflected", "decimals": 6}


@dataclass
class _PlotData:
    edge_table: pl.DataFrame
    edge_segment_table: pl.DataFrame
    node_table: pl.DataFrame


def _analysis():
    displayed = pl.DataFrame(
        {
            "source_tree_node_index": [0, 2],
            "target_tree_node_index": [1, 3],
            "source_scalar": [0.2, 0.3],
            "target_scalar": [0.9, 0.8],
            "x_source": [0.0, 1.0],
            "x_target": [0.5, 1.5],
            "y_source": [0.2, 0.3],
            "y_target": [0.9, 0.8],
        }
    )
    tree = SimpleNamespace(
        plot_data=_PlotData(
            displayed,
            displayed,
            pl.DataFrame(
                {
                    "tree_node_index": [0, 1, 2, 3, 4],
                    "dna_sequence": ["ROOT", "MAX1", "MERGER", "MAX2", "JOIN*"],
                    "vertex_index": [0, 1, 2, 3, 4],
                    "x": [0.0, 0.5, 1.0, 1.5, 0.7],
                    "scalar": [0.1, 0.9, 0.2, 0.8, -0.2],
                }
            ),
        ),
        branch_table=pl.DataFrame(
            {
                "source_tree_node_index": [1, 3],
                "target_tree_node_index": [0, 2],
                "branch_vertex_count": [1, 4],
                "synonymous_endpoints": [1, 0],
            }
        ),
        all_stop_marker_table=pl.DataFrame(
            {
                "vertex_id": [2],
                "dna_sequence": ["MAX1"],
                "aa_sequence": ["K"],
                "x": [1.0],
                "y": [0.9],
                "display_event": ["split_saddle"],
            }
        ),
        wild_type_marker_table=pl.DataFrame(
            {
                "dna_sequence": ["MAX1"],
                "aa_sequence": ["K"],
                "marker_label": ["WT"],
                "x": [1.0],
                "y": [0.9],
            }
        ),
        critical_node_table=pl.DataFrame(
            {
                "dna_sequence": ["ROOT", "MAX1", "MERGER", "MAX2"],
                "scalar": [0.1, 0.9, 0.2, 0.8],
            }
        ),
        branch_member_table=pl.DataFrame(
            {"dna_sequence": ["INTERIOR"], "scalar": [0.3]}
        ),
    )
    return SimpleNamespace(
        trees={"split": tree}, scalar_domain=(-1.0, 1.1), scalar_label="Fitness"
    )


def _payload():
    pairs = pl.DataFrame(
        {
            "pair_id": [0, 1, 2],
            "pair_kind": ["ordinary", "relative", "extended_positive"],
            "homology_dimension": [0, 1, 0],
            "birth_sequence": ["VALLEY", "PEAK*", "GLOBAL_VALLEY*"],
            "death_sequence": ["JOIN*", "SPLIT", "GLOBAL_PEAK"],
            "birth_has_stop_codon": [False, True, True],
            "death_has_stop_codon": [True, False, False],
            "birth_scalar": [-0.8, 0.9, -1.0],
            "death_scalar": [-0.2, 0.1, 1.1],
            "persistence": [0.6, 0.8, 2.1],
        }
    )
    full_fitness = pl.DataFrame(
        {
            "dna_sequence": [
                "VALLEY",
                "JOIN*",
                "PEAK*",
                "SPLIT",
                "GLOBAL_VALLEY*",
                "GLOBAL_PEAK",
                "NONCRITICAL_A",
                "NONCRITICAL_B",
            ],
            "raw_fitness": [-0.8, -0.2, 0.9, 0.1, -1.0, 1.1, -0.35, 0.4],
        },
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64},
    )
    return {
        "analysis": _analysis(),
        "pairs": pairs,
        "diagnostics": {"global_minimum": -1.0, "sequence_count": 8},
        "full_fitness": full_fitness,
        "manuscript_quantities": label_quantities(),
        "threshold_diagnostics": _make_threshold_diagnostics(full_fitness, DECLARATION),
    }


def _make_threshold_diagnostics(
    full_fitness: pl.DataFrame,
    declaration: dict[str, Any] | None = None,
    *,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a threshold_diagnostics dict carrying real CandidateThreshold objects.

    Reproduces exactly the stage logic: candidate = from_summary(settings, raw_fitness=...);
    computed = candidate or compute(...); reflected = compute(...). Same calls, order, rule.
    """
    if settings is None:
        if declaration is None:
            declaration = DECLARATION
        settings = {
            "analysis": {"clamped": {"candidate_functional_threshold": declaration}}
        }
    else:
        declaration = settings["analysis"]["clamped"]["candidate_functional_threshold"]
    values = full_fitness["raw_fitness"].to_list()
    candidate = candidate_threshold_from_summary(settings, raw_fitness=values)
    computed = candidate or compute_candidate_functional_threshold(values, declaration)
    rule_name = declaration["rule"] if candidate is None else candidate.rule
    if rule_name == MEDIAN_REFLECTED:
        reflected = compute_candidate_functional_threshold(
            values, comparison_declaration(declaration)
        )
    else:
        reflected = None
    d = {
        "candidate": computed,
        "lowest_split_saddle_death": None,
        "highest_join_saddle_death": None,
    }
    if reflected is not None:
        d["reflected"] = reflected
    return d


def _settings(bins=6):
    return {
        "analysis": {"clamped": {"candidate_functional_threshold": DECLARATION}},
        "figures": {
            "rendering": {
                "stop_codon_color": "#595959",
                "fitness_tick_step": 0.1,
                "fitness_padding_fraction": 0.05,
                "font_size": 15,
                "branch_count_title": "Branch\\nvertex count",
                "colorbar_top_inset": 8,
            },
            "figures": {
                FIGURE_ID: {
                    "colorbar_x_fraction": 0.9,
                    "histogram_bins": bins,
                    "tree": {"width": 420, "height": 250},
                    "persistence": {"width": 250},
                },
                VERTICAL_ID: {"persistence": {"histogram_width": 90}},
                # The dashed rule's style is shared with this figure.
                "graph_tree_persistence": {
                    "tree_threshold_color": "#777777",
                    "tree_threshold_opacity": 0.7,
                    "tree_threshold_width": 1.2,
                    "tree_threshold_dash": [4, 3],
                },
            },
        },
    }


@pytest.fixture(autouse=True)
def _fake_topographer_tree(monkeypatch):
    from dubanevics2026topological.figures import _charts as charts

    def fake_plot_tree(plot_data, *, config, presentation, **kwargs):
        domain = list(config.edge_color_scale.domain)
        # Two points on a hidden x axis, like the real tree: Vega can lay it out.
        return (
            alt.Chart(
                pl.DataFrame({"x": [0.0, 1.0], "y": [0.2, 0.9], "vertex_count": [1, 2]})
            )
            .mark_line()
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q"),
                color=alt.Color("vertex_count:Q", scale=alt.Scale(domain=domain)),
            )
            .properties(width=config.width, height=config.height)
        )

    monkeypatch.setattr(charts, "plot_tree", fake_plot_tree)


def _render(variant="vanilla", **kwargs):
    return render_payload(
        get_spec(FIGURE_ID), _payload(), _settings(**kwargs), variant
    ).to_dict(validate=True)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _assert_square_at_the_back(scatter, spec):
    """One grey noise square, the diagram's first layer, from the minimum to the threshold."""
    rects = [i for i, layer in enumerate(scatter["layer"]) if layer["mark"]["type"] == "rect"]
    assert rects == [0]
    square = scatter["layer"][0]
    assert square["mark"]["color"] == palette.BAND_GREY  # the bootstrap band's fill
    assert square["mark"]["opacity"] == 0.3
    (row,) = spec["datasets"][square["data"]["name"]]
    assert row["death_scalar"] == row["birth_scalar"] < row["square_high"]


def _points_layer(scatter):
    """The first point layer of the diagram: the plain (grey) peaks."""
    return next(layer for layer in scatter["layer"] if layer["mark"]["type"] == "point")


def test_spec_declares_inputs_and_composes_payload():
    spec = get_spec(FIGURE_ID)
    assert spec.inputs == {
        "split_analysis",
        "component_persistence",
        "full_fitness",
        "manuscript_quantities",
        "threshold_diagnostics",
    }
    assert spec.variants == {"vanilla", "annotated"}
    payload = _payload()
    inputs = {
        "split_analysis": payload["analysis"],
        "component_persistence": (payload["pairs"], payload["diagnostics"]),
        "full_fitness": payload["full_fitness"],
        "manuscript_quantities": payload["manuscript_quantities"],
        "threshold_diagnostics": payload["threshold_diagnostics"],
    }
    composed = compose_payload(spec, inputs, _settings())
    assert composed["analysis"] is payload["analysis"]
    assert composed["pairs"] is payload["pairs"]
    assert composed["full_fitness"] is payload["full_fitness"]
    assert composed["manuscript_quantities"] is payload["manuscript_quantities"]


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_title_counts_are_the_registrys_and_say_ordinary_h0_pairs_not_peaks(variant):
    payload = _payload()
    spec = render_payload(get_spec(FIGURE_ID), payload, _settings(), variant).to_dict(
        validate=True
    )
    subtitle = spec["title"]["subtitle"]
    pairs = SENTINEL["full_graph.persistence.ordinary_h0_pair_count"]
    valleys = SENTINEL["full_graph.persistence.valley_count"]
    total = SENTINEL["full_graph.vertex_count"]
    assert subtitle == [
        (
            f"Papkou et al. (2023) · Persistence: {pairs} ordinary H0 pairs · "
            f"{valleys} valleys"
        ),
        f"{total} total sequences",
    ]
    # "peak" keeps one meaning (a split-tree maximum): the persistence count is not it.
    assert "peaks" not in " ".join(subtitle)
    assert set(payload["manuscript_quantities"].requested_keys) == {
        "full_graph.persistence.ordinary_h0_pair_count",
        "full_graph.persistence.valley_count",
        "full_graph.vertex_count",
    }
    # Nothing else the figure draws prints a recomputed count or threshold.
    assert f"{payload['full_fitness'].height:,} total sequences" not in spec_text(spec)


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_two_panels_share_the_fitness_axis(variant):
    spec = _render(variant)
    assert len(spec["hconcat"]) == 2  # no right-side marginal histogram
    assert "y" not in spec["resolve"]["scale"]
    domains = {
        tuple(item["encoding"]["y"]["scale"]["domain"])
        for item in _walk(spec)
        if item.get("encoding", {}).get("y", {}).get("field")
        in {"y", "birth_scalar", "bin_min", "hist_bin_min", "scalar"}
        and "domain" in item["encoding"]["y"].get("scale", {})
    }
    assert len(domains) == 1
    # Every panel carries an explicit domain, so alignment is by equality.
    panel_domains = [
        {
            tuple(item["encoding"]["y"]["scale"]["domain"])
            for item in _walk(panel)
            if item.get("encoding", {}).get("y", {}).get("field")
            in {"y", "birth_scalar", "bin_min", "hist_bin_min", "scalar"}
            and "domain" in item["encoding"]["y"].get("scale", {})
        }
        for panel in spec["hconcat"]
    ]
    assert all(panel == domains for panel in panel_domains)
    # Duplicate y axes are hidden on the middle diagram.
    middle = spec["hconcat"][1]
    for item in _walk(middle):
        y = item.get("encoding", {}).get("y", {})
        if y.get("field") == "birth_scalar":
            assert y.get("axis") is None
    assert not any(
        item.get("encoding", {}).get("y", {}).get("field") == "bin_min"
        for item in _walk(spec)
    )


def test_persistence_panel_has_no_tree_text_labels_and_a_marginal_histogram():
    spec = _render()
    middle = spec["hconcat"][1]
    labels = {
        row["label"]
        for rows in spec["datasets"].values()
        for row in rows
        if "label" in row
    }
    assert not {"Split tree", "Join tree"} & labels
    assert "extended" not in str(spec["title"]).lower()
    assert "Persistence" in str(spec["title"])
    # No legend row sits below the diagram: the key is drawn inside it.
    marginal, scatter = middle["vconcat"]
    assert any(layer["mark"]["type"] == "bar" for layer in marginal["layer"])
    assert not any(layer["mark"]["type"] == "bar" for layer in scatter["layer"])
    # The diagram is square: width equals the tree height.
    assert scatter["width"] == scatter["height"] == 250


def test_marginal_histogram_has_tick_only_count_axis_and_a_median_marker():
    spec = _render()
    marginal = spec["hconcat"][1]["vconcat"][0]
    bars = next(layer for layer in marginal["layer"] if layer["mark"]["type"] == "bar")
    axis = bars["encoding"]["y"]["axis"]
    assert axis["title"] == "Count"
    assert axis["domain"] is False and axis["grid"] is False
    assert axis["ticks"] is True
    # Fixed count axis: 0 to 20K, ticks at 0 and 20K only.
    assert bars["encoding"]["y"]["scale"]["domain"] == [0, 20000]
    assert axis["values"] == [0, 20000]
    assert any(layer["mark"]["type"] == "rule" for layer in marginal["layer"])


def test_tree_stop_markers_skip_saddles_and_the_key_names_them():
    spec = _render("annotated")
    tree = spec["hconcat"][0]["vconcat"][1]
    texts = {
        item["mark"].get("text")
        for item in _walk(tree)
        if isinstance(item.get("mark"), dict)
    }
    # The one-line key under the merge tree names the markers it draws.
    assert "Stop codon containing peaks" in texts
    assert "Wild type" in texts
    # The only stop-table row (MAX1) is a saddle, which carries no cross by default.
    crosses = [
        spec["datasets"][item["data"]["name"]]
        for item in _walk(tree)
        if isinstance(item.get("data"), dict)
        and item["data"].get("name") in spec["datasets"]
        and spec["datasets"][item["data"]["name"]]
        and "aa_sequence" in spec["datasets"][item["data"]["name"]][0]
        and "marker_label" not in spec["datasets"][item["data"]["name"]][0]
    ]
    assert all(not rows for rows in crosses)


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_tree_key_is_one_horizontal_line_under_the_merge_tree(variant):
    tree = _render(variant)["hconcat"][0]["vconcat"][1]
    entries = {
        item["mark"]["text"]: item["mark"]
        for item in _walk(tree)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("text") in {"Wild type", "Stop codon containing peaks"}
    }
    assert "Wild type" in entries
    assert ("Stop codon containing peaks" in entries) == (variant == "annotated")
    # One row: every entry shares one y; entries differ in x.
    assert len({mark["y"] for mark in entries.values()}) == 1
    assert len({mark["x"] for mark in entries.values()}) == len(entries)


def test_persistence_diagram_draws_peaks_up_and_valleys_down_all_filled():
    spec = _render()
    scatter = spec["hconcat"][1]["vconcat"][1]
    points = _points_layer(scatter)
    assert points["encoding"]["x"]["field"] == "death_scalar"
    assert points["encoding"]["y"]["field"] == "birth_scalar"
    layers = _point_layers(scatter)
    assert all("shape" not in layer["encoding"] for layer in layers)
    assert all(layer["mark"]["filled"] is True for layer in layers)
    rows = {
        shape: [
            row
            for layer in layers
            if layer["mark"]["shape"] == shape
            for row in spec["datasets"][layer["data"]["name"]]
        ]
        for shape in ("triangle-up", "triangle-down")
    }
    assert {r["critical_category"] for r in rows["triangle-up"]} == {"Peaks"}
    assert {r["critical_category"] for r in rows["triangle-down"]} == {"Valleys"}
    # Peaks above the diagonal; valleys (the join-tree pair and the mirrored global
    # pair, the printed valley count) below it.
    assert len(rows["triangle-up"]) == 2
    assert len(rows["triangle-down"]) == 2
    assert all(r["death_scalar"] <= r["birth_scalar"] for r in rows["triangle-up"])
    assert all(r["death_scalar"] >= r["birth_scalar"] for r in rows["triangle-down"])


def test_the_tree_marks_join_tree_valley_saddles_with_filled_down_triangles():
    spec = _render()
    layers = [
        item
        for item in _walk(spec["hconcat"][0])
        if item.get("mark", {}).get("shape") == "triangle-down"
    ]
    assert len(layers) == 1
    assert layers[0]["mark"]["filled"] is True
    # The colour is a per-row field: mid-grey, or teal for a synonymous valley.
    assert layers[0]["encoding"]["color"]["field"] == "marker_color"
    rows = spec["datasets"][layers[0]["data"]["name"]]
    assert [row["marker_color"] for row in rows] == [palette.PEAK_GREY]
    assert [row["dna_sequence"] for row in rows] == ["JOIN*"]
    assert (rows[0]["x"], rows[0]["y"]) == (0.7, -0.2)


@pytest.mark.parametrize("bins", [4, 9])
def test_histogram_bins_key_drives_the_marginal_histogram(bins):
    spec = _render(bins=bins)
    marginal = spec["hconcat"][1]["vconcat"][0]
    bars = next(layer for layer in marginal["layer"] if layer["mark"]["type"] == "bar")
    assert len(spec["datasets"][bars["data"]["name"]]) == bins


def test_count_colorbar_is_inside_the_tree_panel():
    spec = _render()
    tree = spec["hconcat"][0]["vconcat"][1]
    assert "layer" in tree
    legend = tree["layer"][1]
    title = next(
        item["mark"]
        for item in _walk(legend)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("type") == "text"
        and "Branch" in str(item["mark"].get("text"))
    )
    assert 0 < title["x"] < 420
    assert title["x"] == pytest.approx(0.9 * 420, abs=40)
    # No separate legend panel sits beside the tree.
    assert len(spec["hconcat"]) == 2


def test_variants_differ_only_by_annotations():
    vanilla = _render("vanilla")
    annotated = _render("annotated")
    assert "Stop codon containing" not in str(vanilla)
    assert "Stop codon containing" in str(annotated)
    assert len(vanilla["hconcat"]) == len(annotated["hconcat"]) == 2


def test_median_tick_survives_on_the_death_axis():
    spec = _render()
    scatter = spec["hconcat"][1]["vconcat"][1]
    points = _points_layer(scatter)
    median = float(_payload()["full_fitness"]["raw_fitness"].median())
    values = points["encoding"]["x"]["axis"]["values"]
    assert median in values
    assert {-1.0, 0.0, 1.1} <= set(values)


def _point_layers(scatter):
    return [layer for layer in scatter["layer"] if layer["mark"]["type"] == "point"]


CROSS = "M-1,-1L1,1M-1,1L1,-1"


def test_annotated_stop_codons_are_a_red_cross_on_top_of_a_grey_marker():
    spec = _render("annotated")
    scatter = spec["hconcat"][1]["vconcat"][1]
    layers = _point_layers(scatter)
    crosses = [layer for layer in layers if layer["mark"].get("shape") == CROSS]
    # One cross layer for the stop peak and one for the stop valley (the mirror of
    # the global pair carries the global minimum's stop codon).
    assert len(crosses) == 2
    for layer in crosses:
        assert layer["mark"]["color"] == "#595959"  # stop_codon_color of the fixture
        assert layer["mark"]["filled"] is False  # a stroke, no interior
        # The tree's cross style (defaults here, shipped values in the config).
        assert layer["mark"]["opacity"] == 0.7
        assert layer["mark"]["strokeWidth"] == 0.5
    rows = [row for layer in crosses for row in spec["datasets"][layer["data"]["name"]]]
    assert sorted(r["birth_sequence"] for r in rows) == ["GLOBAL_VALLEY*", "PEAK*"]
    # The markers beneath stay grey triangles (never crimson) and the crosses are drawn
    # after every marker.
    markers = [
        layer
        for layer in layers
        if layer["mark"].get("shape") in {"triangle-up", "triangle-down"}
    ]
    assert all(layer["mark"]["color"] == palette.PEAK_GREY for layer in markers)
    assert min(layers.index(layer) for layer in crosses) > max(
        layers.index(layer) for layer in markers
    )


def test_vanilla_draws_no_stop_cross_and_no_teal_marker():
    spec = _render("vanilla")
    scatter = spec["hconcat"][1]["vconcat"][1]
    layers = _point_layers(scatter)
    assert not any(layer["mark"].get("shape") == CROSS for layer in layers)
    assert not any(layer["mark"].get("color") == palette.SYNONYMOUS_TEAL for layer in layers)


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_plain_persistence_markers_are_count_faded(variant):
    spec = _render(variant)
    scatter = spec["hconcat"][1]["vconcat"][1]
    grey = [
        layer
        for layer in _point_layers(scatter)
        if layer["mark"].get("color") == palette.PEAK_GREY
    ]
    assert [layer["mark"]["shape"] for layer in grey] == ["triangle-up", "triangle-down"]
    assert all(layer["mark"]["filled"] is True for layer in grey)
    assert all(
        layer["encoding"]["opacity"]["field"] == "count_opacity" for layer in grey
    )


def _threshold_rule(scatter):
    """The dashed threshold rule layers of the persistence diagram (y encodings)."""
    return [
        layer
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "rule"
        and layer.get("encoding", {}).get("y", {}).get("field") == "functional_threshold"
    ]


def test_annotated_threshold_is_a_dashed_rule_with_one_axis_label_and_a_caption():
    band = _band_numbers()
    spec = _render("annotated")
    scatter = spec["hconcat"][1]["vconcat"][1]
    # The grey noise square sits at the very back; no square apex marker.
    _assert_square_at_the_back(scatter, spec)
    assert not [
        layer for layer in scatter["layer"] if layer["mark"].get("shape") == "square"
    ]
    (rule,) = _threshold_rule(scatter)
    (row,) = scatter_rows(spec, rule)
    assert row["functional_threshold"] == pytest.approx(band.noise_band_upper)
    # The style is the dashed rule of graph_tree_persistence.
    assert rule["mark"]["strokeDash"] == [4.0, 3.0]
    assert rule["mark"]["color"] == "#777777"
    assert rule["mark"]["opacity"] == pytest.approx(0.7)
    assert rule["mark"]["strokeWidth"] == pytest.approx(1.2)
    # The rule spans the diagram only: it is a layer of the scatter panel.
    assert "x" not in rule["encoding"]
    # The left axis shows exactly the threshold, three decimals, without ticks.
    axis = rule["encoding"]["y"]["axis"]
    assert axis["values"] == [pytest.approx(band.noise_band_upper)]
    assert axis["format"] == ".3f"
    assert axis["ticks"] is False
    assert axis["title"] is None
    other_axes = [
        layer["encoding"]["y"].get("axis", "absent")
        for layer in scatter["layer"]
        if layer is not rule and "y" in layer.get("encoding", {})
    ]
    assert all(axis is None for axis in other_axes)
    # The caption sits above the rule at the diagram's right edge.
    (caption,) = [
        layer
        for layer in scatter["layer"]
        if layer["mark"].get("type") == "text"
        and isinstance(layer["mark"].get("text"), list)
    ]
    # A list of lines: Vega draws one line per item (a "\n" in one string would not wrap).
    assert caption["mark"]["text"] == ["Median-reflected", "threshold"]
    mark = caption["mark"]
    assert (mark["align"], mark["baseline"], mark["color"]) == (
        "right",
        "bottom",
        "#555555",
    )
    assert mark["dy"] < 0
    assert caption["encoding"]["x"]["value"] == scatter["width"]
    assert caption["encoding"]["y"]["field"] == "functional_threshold"


def test_vanilla_draws_no_threshold_rule_label_text_or_apex_but_the_noise_square():
    spec = _render("vanilla")
    scatter = spec["hconcat"][1]["vconcat"][1]
    _assert_square_at_the_back(scatter, spec)
    assert not [
        layer for layer in scatter["layer"] if layer["mark"].get("shape") == "square"
    ]
    assert "Median-reflected\nthreshold" not in spec_text(spec)
    assert not [
        layer
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "rule" and "y" in layer.get("encoding", {})
    ]
    # The diagram shows no y axis at all: no label, no tick.
    assert all(
        layer["encoding"]["y"].get("axis", "absent") is None
        for layer in scatter["layer"]
        if "y" in layer.get("encoding", {})
    )


def test_annotated_wild_type_peak_is_a_solid_red_triangle_over_a_stop_peak():
    payload = _payload()
    payload["analysis"].trees["split"].wild_type_marker_table = pl.DataFrame(
        {
            "dna_sequence": ["PEAK*"],
            "aa_sequence": ["K"],
            "marker_label": ["WT"],
            "x": [1.0],
            "y": [0.9],
        }
    )
    spec = render_payload(
        get_spec(FIGURE_ID), payload, _settings(), "annotated"
    ).to_dict(validate=True)
    scatter = spec["hconcat"][1]["vconcat"][1]
    layers = _point_layers(scatter)
    wild_type = next(layer for layer in layers if layer["mark"].get("color") == palette.WILD_TYPE_RED)
    assert [
        r["birth_sequence"] for r in spec["datasets"][wild_type["data"]["name"]]
    ] == ["PEAK*"]
    # The wild type takes the stop peak's place: no crimson layer repeats it.
    assert not [
        layer
        for layer in layers
        if layer["mark"].get("color") == "#595959"
        and layer["mark"]["shape"] == "triangle-up"
    ]
    assert wild_type["mark"]["color"] == palette.WILD_TYPE_RED
    assert wild_type["encoding"]["opacity"]["value"] == 1.0
    assert wild_type["mark"]["filled"] is True


def test_persistence_height_follows_tree_height():
    settings = _settings()
    settings["figures"]["figures"][FIGURE_ID]["tree"]["height"] = 300
    spec = render_payload(get_spec(FIGURE_ID), _payload(), settings, "vanilla").to_dict(
        validate=True
    )
    assert spec["hconcat"][1]["vconcat"][1]["height"] == 300


def test_marginal_histogram_defaults_to_the_tree_histograms_200_bins():
    settings = _settings()
    del settings["figures"]["figures"][FIGURE_ID]["histogram_bins"]
    spec = render_payload(get_spec(FIGURE_ID), _payload(), settings, "vanilla").to_dict(
        validate=True
    )
    marginal = spec["hconcat"][1]["vconcat"][0]
    bars = next(layer for layer in marginal["layer"] if layer["mark"]["type"] == "bar")
    assert len(spec["datasets"][bars["data"]["name"]]) == 200


def test_shipped_config_matches_the_tree_histogram_bins_and_line_styles():
    import tomllib
    from pathlib import Path

    config = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )["figures"]
    figure = config[FIGURE_ID]
    assert figure["histogram_bins"] == 200
    assert (
        figure["histogram_bins"]
        == config["full_tree_histogram"]["histogram_bins"]
    )
    persistence = figure["persistence"]
    # Dashed threshold, dotted median: the dash is longer, the colour and width match.
    assert persistence["threshold_dash"][0] > persistence["median_dash"][0]
    assert persistence["threshold_color"] == persistence["median_color"]
    assert persistence["threshold_width"] == persistence["median_width"]


def test_shipped_config_has_no_separate_persistence_height():
    import tomllib
    from pathlib import Path

    config = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )
    persistence = config["figures"][FIGURE_ID]["persistence"]
    assert "height" not in persistence


def test_rejects_population_without_global_extrema():
    payload = _payload()
    payload["full_fitness"] = pl.DataFrame(
        {
            "dna_sequence": payload["full_fitness"]["dna_sequence"].to_list(),
            "raw_fitness": [-0.8, -0.2, 0.9, 0.1, -1.0, 1.1, -0.35, 2.0],
        },
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64},
    )
    with pytest.raises(ValueError, match="full-population extrema"):
        render_payload(get_spec(FIGURE_ID), payload, _settings(), "vanilla")


def _band_numbers():
    payload = _payload()
    diags = payload["threshold_diagnostics"]
    return topological_noise_band(
        payload["pairs"],
        payload["full_fitness"],
        DECLARATION,
        candidate=diags["candidate"],
        reflected=diags["reflected"],
    )


def test_topological_noise_band_numbers():
    band = _band_numbers()
    assert band.global_minimum == -1.0  # g
    assert band.median == pytest.approx(-0.05)  # m
    assert band.median_gap == pytest.approx(0.95)  # h = m - g
    assert band.functional_threshold == pytest.approx(0.9)  # m + h
    # Descending ordinary pairs end at split saddles, relative pairs at join saddles.
    assert band.lowest_split_saddle_death == pytest.approx(0.1)
    assert band.highest_join_saddle_death == pytest.approx(-0.2)


def test_topological_noise_band_without_saddle_pairs_or_spread():
    pairs = pl.DataFrame(
        {
            "pair_id": [0],
            "pair_kind": ["extended_positive"],
            "homology_dimension": [0],
            "birth_sequence": ["LOW"],
            "death_sequence": ["HIGH"],
            "birth_has_stop_codon": [False],
            "death_has_stop_codon": [False],
            "birth_scalar": [1.0],
            "death_scalar": [1.0],
            "persistence": [0.0],
        }
    )
    fitness = pl.DataFrame(
        {"dna_sequence": ["LOW", "HIGH"], "raw_fitness": [1.0, 1.0]},
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64},
    )
    values = fitness["raw_fitness"].to_list()
    c = compute_candidate_functional_threshold(values, DECLARATION)
    r = compute_candidate_functional_threshold(
        values, comparison_declaration(DECLARATION)
    )
    band = topological_noise_band(pairs, fitness, DECLARATION, candidate=c, reflected=r)
    assert band.lowest_split_saddle_death is None
    assert band.highest_join_saddle_death is None
    assert band.median_gap == 0.0
    assert band.functional_threshold == 1.0


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_every_variant_draws_the_grey_noise_square_at_the_very_back(variant):
    spec = _render(variant)
    scatter = spec["hconcat"][1]["vconcat"][1]
    _assert_square_at_the_back(scatter, spec)
    # Behind the dotted median rule, the diagonal and every point.
    kinds = [layer["mark"]["type"] for layer in scatter["layer"]]
    assert kinds.index("rect") < kinds.index("line") < kinds.index("point")
    assert not any(layer["mark"]["type"] == "area" for layer in scatter["layer"])


def scatter_rows(spec, layer):
    return spec["datasets"][layer["data"]["name"]]


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_the_mixture_rule_draws_no_noise_band(variant):
    """GLOSSARY: the Topological noise band is retained for the comparison rule only."""
    from dubanevics2026topological.threshold.functional_threshold import (
        CandidateThreshold,
    )

    mixture = CandidateThreshold(
        rule="mixture_quantile",
        decimals=6,
        median=-0.05,
        global_minimum=-1.0,
        median_gap=0.95,
        unrounded=0.321,
        threshold=0.321,
    )
    payload = dict(_payload())
    payload["threshold_diagnostics"] = {
        "candidate": mixture,
        "lowest_split_saddle_death": None,
        "highest_join_saddle_death": None,
    }
    settings = _settings()
    settings["analysis"]["clamped"]["candidate_functional_threshold"] = {
        "rule": "mixture_quantile",
        "decimals": 6,
    }
    spec = render_payload(get_spec(FIGURE_ID), payload, settings, variant).to_dict(
        validate=True
    )
    scatter = spec["hconcat"][1]["vconcat"][1]
    assert not [layer for layer in scatter["layer"] if layer["mark"]["type"] == "rect"]
    assert not [
        layer
        for layer in scatter["layer"]
        if layer["mark"].get("shape") == "square"
    ]
    assert "Median-reflected\nthreshold" not in spec_text(spec)
    assert not _threshold_rule(scatter)


def _shipped_geometry_render(variant, **persistence):
    """Render with the shipped persistence table over a realistic population.

    The population spans fitness -1.2 to 1.4 (the shipped ``fitness_domain``), its
    median is -0.7265 and its functional threshold ``2 m - g`` is -0.253: the rule
    lands about 156 px down a 250 px diagram, where the full graph's rule does.
    """
    import tomllib
    from pathlib import Path

    config = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )
    payload = _payload()
    values = {
        "VALLEY": -0.8,
        "JOIN*": -0.2,
        "PEAK*": 0.9,
        "SPLIT": 0.1,
        "GLOBAL_VALLEY*": -1.2,
        "GLOBAL_PEAK": 1.4,
        "NONCRITICAL_A": -0.85,
        "NONCRITICAL_B": -0.78,
        "NONCRITICAL_C": -0.753,
        "NONCRITICAL_D": -0.7,
    }
    payload["full_fitness"] = pl.DataFrame(
        {"dna_sequence": list(values), "raw_fitness": list(values.values())},
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64},
    )
    payload["pairs"] = payload["pairs"].with_columns(
        pl.Series("birth_scalar", [-0.8, 0.9, -1.2]),
        pl.Series("death_scalar", [-0.2, 0.1, 1.4]),
        pl.Series("persistence", [0.6, 0.8, 2.6]),
    )
    payload["analysis"].scalar_domain = (-1.2, 1.4)
    settings = _settings()
    rendering = {**settings["figures"]["rendering"], **config["rendering"]}
    settings["figures"]["rendering"] = rendering
    figure = settings["figures"]["figures"][FIGURE_ID]
    figure["persistence"] = {
        "width": 250,
        **config["figures"][FIGURE_ID]["persistence"],
        **persistence,
    }
    # Rebuild threshold_diagnostics exactly as the product stage would, for the
    # fitness and declaration in the current settings (same from_summary + or + compute
    # calls and rule). This ensures the rendered figure uses the correct T and rule
    # for the shipped geometry test expectations.
    payload["threshold_diagnostics"] = _make_threshold_diagnostics(
        payload["full_fitness"], settings=settings
    )
    spec = render_payload(get_spec(FIGURE_ID), payload, settings, variant).to_dict(
        validate=True
    )
    return spec, rendering, figure["persistence"]


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_shipped_config_draws_the_square_and_an_unlabelled_dashed_rule(variant):
    spec, _, persistence = _shipped_geometry_render(variant)
    scatter = spec["hconcat"][1]["vconcat"][1]
    assert "Functional threshold" not in spec_text(spec)
    _assert_square_at_the_back(scatter, spec)
    if variant == "annotated":
        (rule,) = _threshold_rule(scatter)
        assert rule["mark"]["strokeDash"] == [4.0, 3.0]
    else:
        assert not _threshold_rule(scatter)
    # The key is inside the diagram (the shipped inset keeps it in the plot area).
    keys = [
        layer["mark"]
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "text"
        and isinstance(layer["mark"].get("text"), str)
        and layer["mark"].get("text") in {"peak", "valley", "≥1 stop codon"}
    ]
    assert keys and all(0 < mark["x"] < scatter["width"] for mark in keys)
    assert all(0 < mark["y"] <= scatter["height"] for mark in keys)


def test_shipped_config_has_no_threshold_label_keys():
    import tomllib
    from pathlib import Path

    config = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )
    persistence = config["figures"][FIGURE_ID]["persistence"]
    assert not [key for key in persistence if key.startswith("threshold_label")]


def test_persistence_layers_stack_rules_and_diagonal_under_the_points():
    # Both variants share the median rule and diagonal beneath the points, the
    # in-diagram key (a glyph and a label per entry) is drawn last, and the annotated
    # export puts its threshold rule and caption just under the key.
    def kinds(variant):
        scatter = _render(variant)["hconcat"][1]["vconcat"][1]
        return [layer["mark"]["type"] for layer in scatter["layer"]]

    for variant, entries in (("vanilla", 2), ("annotated", 3)):
        layers = kinds(variant)
        key = 2 * entries
        # The grey noise square is first, then the median rule and the diagonal.
        assert layers[:3] == ["rect", "rule", "line"]
        body = layers[3:-key]
        if variant == "annotated":
            assert body[-2:] == ["rule", "text"]
            body = body[:-2]
        assert all(kind == "point" for kind in body)
        assert layers[-key:] == ["text"] * key


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_threshold_layers_leave_the_y_axis_to_the_tree_but_the_one_label(variant):
    # Only the annotated threshold rule carries a y axis (its single label); every
    # other y encoding of the diagram has none.
    scatter = _render(variant)["hconcat"][1]["vconcat"][1]
    axes = [
        layer["encoding"]["y"].get("axis", "absent")
        for layer in scatter["layer"]
        if "y" in layer.get("encoding", {})
    ]
    visible = [axis for axis in axes if axis is not None]
    if variant == "annotated":
        assert len(visible) == 1
        assert visible[0]["values"] and visible[0]["ticks"] is False
    else:
        assert not [a for a in visible if isinstance(a, dict) and a.get('values') and a.get('ticks') is False]


def _layout(spec):
    """Vega's laid-out plot areas and marks: ``({group: (x, y, w, h)}, {group: [...]})``.

    Coordinates are layout pixels; each mark is ``(marktype, x, y)`` relative to
    the plot area that owns it.
    """
    root = vlc.vegalite_to_scenegraph(spec)["scenegraph"]
    boxes: dict[str, tuple[float, float, float, float]] = {}
    absolute: dict[str, list[tuple[str, float, float]]] = {}

    def visit(mark, origin_x, origin_y, owner):
        for item in mark.get("items", []):
            x, y = origin_x + item.get("x", 0), origin_y + item.get("y", 0)
            if mark.get("marktype") != "group":
                absolute.setdefault(owner, []).append((mark["marktype"], x, y))
                continue
            name = item.get("name") or mark.get("name") or ""
            if name.endswith("_group"):
                boxes[name] = (x, y, item.get("width"), item.get("height"))
                owner = name
            for child in item.get("items", []):
                visit(child, x, y, owner)

    visit(root, 0, 0, "")
    marks = {
        name: [(kind, x - boxes[name][0], y - boxes[name][1]) for kind, x, y in entries]
        for name, entries in absolute.items()
        if name in boxes
    }
    return boxes, marks


# The tree's plot area and the diagram's plot area, as Vega names them.
TREE_AREA = "concat_0_concat_1_group"
DIAGRAM_AREA = "concat_1_concat_1_group"


def _y_domains(panel):
    return {
        tuple(item["encoding"]["y"]["scale"]["domain"])
        for item in _walk(panel)
        if item.get("encoding", {}).get("y", {}).get("field")
        in {"y", "birth_scalar", "scalar"}
        and "domain" in item["encoding"]["y"].get("scale", {})
    }


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_diagram_birth_axis_is_the_trees_fitness_axis_in_pixels(variant):
    spec = _render(variant)
    domain = tuple(
        padded_scalar_domain(
            _analysis().scalar_domain, _settings()["figures"]["rendering"]
        )
    )
    # Same domain on both panels...
    assert _y_domains(spec["hconcat"][0]) == {domain}
    assert _y_domains(spec["hconcat"][1]) == {domain}
    # ...and the same range: the plot areas start at the same height and are the
    # same height, so a fitness value lands on the same row on both panels.
    boxes, marks = _layout(spec)
    _, tree_top, _, tree_height = boxes[TREE_AREA]
    _, diagram_top, diagram_width, diagram_height = boxes[DIAGRAM_AREA]
    assert diagram_top == pytest.approx(tree_top)
    assert diagram_height == tree_height == 250
    # The functional threshold rule sits where the shared mapping puts m + h.
    payload = _payload()
    diags = payload["threshold_diagnostics"]
    threshold = topological_noise_band(
        payload["pairs"],
        payload["full_fitness"],
        DECLARATION,
        candidate=diags["candidate"],
        reflected=diags["reflected"],
    ).functional_threshold
    expected = (domain[1] - threshold) / (domain[1] - domain[0]) * tree_height
    rule_rows = [y for kind, _, y in marks[DIAGRAM_AREA] if kind == "rule"]
    if variant == "annotated":
        assert any(row == pytest.approx(expected, abs=0.01) for row in rule_rows)
    assert diagram_width == diagram_height


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_diagram_is_square_in_data_units(variant):
    spec = _render(variant)
    boxes, marks = _layout(spec)
    _, _, width, height = boxes[DIAGRAM_AREA]
    # The diagonal runs corner to corner of the plot area. Both axes use one
    # domain, so an x range narrower than the y range (the 5 px inset bar marks
    # give a shared x scale) shows up as a diagonal that stops short of the edges.
    corners = [(x, y) for kind, x, y in marks[DIAGRAM_AREA] if kind == "line"]
    assert sorted(corners) == [(0, height), (width, 0)]
    domains = {
        tuple(item["encoding"]["x"]["scale"]["domain"])
        for item in _walk(spec["hconcat"][1])
        if item.get("encoding", {}).get("x", {}).get("field") == "death_scalar"
        and "domain" in item["encoding"]["x"].get("scale", {})
    }
    assert domains == _y_domains(spec["hconcat"][1])


def test_fdr_variant_under_the_median_reflected_cutoff_draws_the_square_only():
    import tomllib

    from dubanevics2026topological.analysis import ROOT

    with (ROOT / "config/variants/threshold=reflected_fdr=by_q=0.05.toml").open(
        "rb"
    ) as stream:
        overlay = tomllib.load(stream)

    settings = _settings()
    settings["analysis"] = overlay

    # Provide threshold_diagnostics built exactly as the stage for these FDR settings
    # (from_summary + or + compute on the fitness). The rule is named
    # ``fdr_fixed_cutoff`` but its declared cutoff is the median-reflected one, so the
    # grey square is drawn.
    p = _payload()
    p = dict(p)
    p["threshold_diagnostics"] = _make_threshold_diagnostics(
        p["full_fitness"], settings=settings
    )
    assert p["threshold_diagnostics"]["candidate"].rule == "fdr_fixed_cutoff"
    spec = render_payload(get_spec(FIGURE_ID), p, settings, "vanilla").to_dict(
        validate=True
    )

    # The vanilla export draws no threshold marking, only the noise square.
    scatter = spec["hconcat"][1]["vconcat"][1]
    _assert_square_at_the_back(scatter, spec)


def _fdr_payload(cutoff):
    """A payload whose Candidate rule is ``fdr_fixed_cutoff`` with the given cutoff.

    The stage stores no ``reflected`` record for this rule, so none is given; the
    applied threshold (0.9 = m + h on this population) is the reflected one.
    """
    from dubanevics2026topological.threshold.functional_threshold import (
        FDR_FIXED_CUTOFF,
        CandidateThreshold,
    )

    payload = dict(_payload())
    payload["threshold_diagnostics"] = {
        "candidate": CandidateThreshold(
            rule=FDR_FIXED_CUTOFF,
            decimals=6,
            median=-0.05,
            global_minimum=-1.0,
            median_gap=0.95,
            unrounded=0.9,
            threshold=0.9,
        ),
        "lowest_split_saddle_death": None,
        "highest_join_saddle_death": None,
    }
    settings = _settings()
    settings["analysis"]["clamped"]["candidate_functional_threshold"] = {
        "rule": FDR_FIXED_CUTOFF,
        "decimals": 6,
        "cutoff": cutoff,
        "method": "by",
        "q": 0.05,
        "input_dir": "unused",
    }
    return payload, settings


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_threshold_rule_is_drawn_whenever_the_cutoff_is_median_reflected(variant):
    payload, settings = _fdr_payload("median_reflected")
    spec = render_payload(get_spec(FIGURE_ID), payload, settings, variant).to_dict(
        validate=True
    )
    scatter = spec["hconcat"][1]["vconcat"][1]
    _assert_square_at_the_back(scatter, spec)
    if variant == "vanilla":
        assert not [
            layer
            for layer in scatter["layer"]
            if layer["mark"].get("shape") == "square"
            or layer["mark"].get("text") == "Median-reflected\nthreshold"
        ]
        return
    (rule,) = _threshold_rule(scatter)
    (row,) = scatter_rows(spec, rule)
    assert row["functional_threshold"] == pytest.approx(0.9)


def test_no_square_or_caption_is_drawn_for_the_other_fdr_cutoff():
    payload, settings = _fdr_payload("papkou_original")
    spec = render_payload(get_spec(FIGURE_ID), payload, settings, "vanilla").to_dict(
        validate=True
    )
    scatter = spec["hconcat"][1]["vconcat"][1]
    assert not [layer for layer in scatter["layer"] if layer["mark"]["type"] == "rect"]
    assert not [
        layer
        for layer in scatter["layer"]
        if layer.get("encoding", {}).get("text", {}).get("field") == "label"
    ]


def test_median_reflected_cutoff_reads_the_rule_and_the_declared_cutoff():
    from dubanevics2026topological.figures.full_tree_persistence import (
        median_reflected_cutoff,
    )

    assert median_reflected_cutoff("median_reflected", {})
    assert median_reflected_cutoff(
        "fdr_fixed_cutoff", {"cutoff": "median_reflected"}
    )
    assert not median_reflected_cutoff(
        "fdr_fixed_cutoff", {"cutoff": "papkou_original"}
    )
    assert not median_reflected_cutoff("mixture_quantile", {})


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_the_full_graph_draws_no_functional_threshold_label(variant):
    """No ``Functional threshold`` text; only the annotated export marks the cutoff."""
    spec = render_payload(
        get_spec(FIGURE_ID), _payload(), _settings(), variant
    ).to_dict(validate=True)
    assert "Functional threshold" not in spec_text(spec)
    scatter = spec["hconcat"][1]["vconcat"][1]
    assert bool(_threshold_rule(scatter)) == (variant == "annotated")


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("vanilla", ["peak", "valley"]),
        ("annotated", ["peak", "valley", "≥1 stop codon"]),
    ],
)
def test_persistence_legend_is_inside_the_diagram_bottom_right_one_entry_per_line(
    variant, expected
):
    import tomllib
    from pathlib import Path

    shipped = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )["figures"][FIGURE_ID]["persistence"]
    inset_x = shipped["persistence_legend_inset_x"]
    inset_y = shipped["persistence_legend_inset_y"]
    settings = _settings()
    settings["figures"]["figures"][FIGURE_ID]["persistence"].update(
        persistence_legend_inset_x=inset_x,
        persistence_legend_inset_y=inset_y,
        persistence_legend_font_size=11,  # the shipped [rendering] value
    )
    spec = render_payload(get_spec(FIGURE_ID), _payload(), settings, variant).to_dict(
        validate=True
    )
    middle = spec["hconcat"][1]
    # No legend row under the diagram: only the marginal histogram and the diagram.
    assert len(middle["vconcat"]) == 2
    scatter = middle["vconcat"][1]
    entries = [
        layer["mark"]
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "text"
        and isinstance(layer["mark"].get("text"), str)
        and layer["mark"].get("text") in {"peak", "valley", "≥1 stop codon", "wild type"}
    ]
    assert [mark["text"] for mark in entries] == expected
    ys = [mark["y"] for mark in entries]
    # One entry per line, top to bottom, left-aligned as a block at the right.
    assert ys == sorted(ys) and len(set(ys)) == len(ys)
    assert len({mark["x"] for mark in entries}) == 1
    # Bottom right, inside the diagram: each label starts where the longest label's
    # estimated width (0.55 * font size per character, rounded up) ends at the right
    # inset, and each glyph sits 14 px left of its label.
    font_size = entries[0]["fontSize"]
    longest = max(len(mark["text"]) for mark in entries)
    label_x = scatter["width"] - inset_x - math.ceil(0.55 * font_size * longest)
    assert all(abs(mark["x"] - label_x) <= 1 for mark in entries)
    glyphs = [
        layer["mark"]
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "text"
        and isinstance(layer["mark"].get("text"), str)
        and layer["mark"].get("text") in {"▲", "▼", "★", "×", "●"}
        and layer["mark"]["y"] in set(ys)
    ]
    assert len(glyphs) == len(entries)
    assert all(0 < mark["x"] < scatter["width"] for mark in entries + glyphs)
    assert all(mark["x"] > scatter["width"] / 10 for mark in glyphs)  # long stop label
    # The last line sits well above the bottom edge and the block well left of the right
    # edge (the shipped insets, 50 and 45 px), so the global-minimum corner point (about 11 px
    # from both edges) is not read as a legend symbol: a gap of at least 25 px each way.
    assert ys[-1] <= scatter["height"] - (inset_y - 10)
    assert all(mark["x"] <= scatter["width"] - inset_x - 14 for mark in entries)
    assert inset_x >= 25 and inset_y >= 25
    old = ("peak (split tree)", "valley (join tree)", "No stop codon", "stop codon (≥1)")
    assert not [label for label in old if label in spec_text(spec)]
    if variant == "vanilla":
        assert "stop codon" not in spec_text(spec)


def _left_axes(node, found=None):
    """Every left-oriented Vega axis of a compiled specification."""
    found = [] if found is None else found
    if isinstance(node, dict):
        found.extend(a for a in node.get("axes", []) if a.get("orient") == "left")
        for value in node.values():
            _left_axes(value, found)
    elif isinstance(node, list):
        for value in node:
            _left_axes(value, found)
    return found


def test_annotated_persistence_panel_compiles_to_one_untickd_threshold_label():
    band = _band_numbers()
    vega = vlc.vegalite_to_vega(_render("annotated"))
    axes = [
        axis
        for axis in _left_axes(vega)
        if axis.get("values") == [pytest.approx(band.noise_band_upper)]
    ]
    assert len(axes) == 1
    (axis,) = axes
    assert axis["format"] == ".3f"
    assert axis["ticks"] is False
    assert axis["domain"] is False
    assert axis.get("title") in (None, "")
    # The vanilla panel keeps no y axis of its own.
    plain = vlc.vegalite_to_vega(_render("vanilla"))
    assert not [
        a for a in _left_axes(plain) if a.get("values") == [pytest.approx(band.noise_band_upper)]
    ]



@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_tree_y_axis_does_not_carry_the_threshold_value(variant):
    """The threshold label belongs to the persistence diagram's axis only."""
    threshold = _band_numbers().noise_band_upper
    spec = _render(variant)
    tree_axes = [
        item["encoding"]["y"]["axis"]
        for item in _walk(spec["hconcat"][0])
        if isinstance(item.get("encoding", {}).get("y", {}).get("axis"), dict)
    ]
    assert tree_axes
    for axis in tree_axes:
        assert not any(
            value == pytest.approx(threshold) for value in axis.get("values", [])
        )
    vega = vlc.vegalite_to_vega(spec)
    labelled = [
        a for a in _left_axes(vega) if a.get("values") == [pytest.approx(threshold)]
    ]
    assert len(labelled) == (1 if variant == "annotated" else 0)



# -- the vertical-histogram variant ------------------------------------------------


def _render_vertical(variant="vanilla"):
    payload = _payload()
    spec = render_payload(get_spec(VERTICAL_ID), payload, _settings(), variant).to_dict(
        validate=True
    )
    # The histogram and tree sit in a nested concat (close together); the tests read
    # the three panels as one row.
    first = spec["hconcat"][0]
    if "hconcat" in first:
        spec["hconcat"] = [*first["hconcat"], *spec["hconcat"][1:]]
    return spec, payload


def _y_ticks(panel):
    return {
        tuple(item["encoding"]["y"]["axis"]["values"])
        for item in _walk(panel)
        if isinstance(item.get("encoding", {}).get("y", {}).get("axis"), dict)
        and "values" in item["encoding"]["y"]["axis"]
    }


def test_vertical_variant_is_a_registered_default_figure_with_its_own_id():
    spec = get_spec(VERTICAL_ID)
    assert spec.default is True  # Figure 5
    assert spec.variants == {"vanilla", "annotated"}
    assert spec.config_table == VERTICAL_ID


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_vertical_variant_lays_out_histogram_tree_diagram_left_to_right(variant):
    spec, payload = _render_vertical(variant)
    fitness = payload["full_fitness"]["raw_fitness"]
    histogram, tree, diagram = spec["hconcat"]
    bars = next(i for i in _walk(histogram) if i.get("mark", {}).get("type") == "bar")
    assert bars["mark"]["orient"] == "horizontal"
    assert bars["encoding"]["y"]["field"] == "hist_bin_min"  # fitness on the one y axis
    assert bars["encoding"]["x"]["field"] == "sequence_count"
    assert bars["encoding"]["x"]["scale"]["reverse"] is True  # grows away from the tree
    # The histogram owns the figure's one fitness axis, titled, ticked at the data
    # maximum, 0, the median and the data minimum with three decimals.
    axis = bars["encoding"]["y"]["axis"]
    assert axis["title"] == "Fitness (WT = 0)" and axis["format"] == ".3f"
    assert axis["values"] == [fitness.max(), 0.0, fitness.median(), fitness.min()]
    # The Count axis is centred under its ticks; the histogram draws no median rule.
    assert bars["encoding"]["x"]["axis"]["labelAlign"] == "center"
    assert bars["encoding"]["x"]["axis"]["labelFlush"] is False
    assert not [i for i in _walk(histogram) if i.get("mark", {}).get("type") == "rule"]
    # The tree and the diagram show no y axis of their own.
    # (The diagram's one exception is the labelled median rule: its left axis carries
    # only the median and threshold values.)
    for panel in (tree, diagram):
        y_encodings = [
            item["encoding"]["y"]
            for item in _walk(panel)
            if "field" in item.get("encoding", {}).get("y", {})
            and item["encoding"]["y"]["field"] != "median_fitness"
        ]
        assert y_encodings and all(e.get("axis", "absent") is None for e in y_encodings)
    # The same explicit fitness domain on all three panels.
    domains = {
        tuple(item["encoding"]["y"]["scale"]["domain"])
        for panel in spec["hconcat"]
        for item in _walk(panel)
        if item.get("encoding", {}).get("y", {}).get("field")
        in {"y", "birth_scalar", "bin_min", "hist_bin_min", "scalar"}
        and "domain" in item["encoding"]["y"].get("scale", {})
    }
    assert len(domains) == 1
    # No top histogram any more: the diagram is a plain layered panel.
    assert "vconcat" not in diagram
    assert not any(
        item.get("mark", {}).get("type") == "bar" for item in _walk(tree)
    )


def test_vertical_variant_draws_a_median_rule_on_the_tree_and_the_diagram_only():
    spec, payload = _render_vertical()
    median = float(payload["full_fitness"]["raw_fitness"].median())
    histogram, tree, diagram = spec["hconcat"]
    assert not [i for i in _walk(histogram) if i.get("mark", {}).get("type") == "rule"]
    for panel, field in ((tree, "y"), (diagram, "median_fitness")):
        rules = [
            item
            for item in _walk(panel)
            if item.get("mark", {}).get("type") == "rule"
            and item.get("encoding", {}).get("y", {}).get("field") == field
        ]
        assert len(rules) == 1, panel
        rows = spec["datasets"][rules[0]["data"]["name"]]
        assert rows == [{field: median}]
        assert rules[0]["mark"]["strokeDash"] == [1.0, 3.0]  # dotted, as everywhere
    # No number is drawn on the tree; the diagram writes the median at the left of its
    # line (see the test below).
    assert not [
        item
        for item in _walk(tree)
        if item.get("mark", {}).get("type") == "text"
        and item.get("encoding", {}).get("text", {}).get("field") == "y"
    ]


def test_vertical_variant_has_a_dotted_vertical_median_rule_on_the_diagram_and_no_label():
    spec, payload = _render_vertical()
    median = float(payload["full_fitness"]["raw_fitness"].median())
    diagram = spec["hconcat"][2]
    rules = [
        item
        for item in _walk(diagram)
        if item.get("mark", {}).get("type") == "rule"
        and item.get("encoding", {}).get("x", {}).get("field") == "median_fitness"
    ]
    assert len(rules) == 1
    assert spec["datasets"][rules[0]["data"]["name"]] == [{"median_fitness": median}]
    assert rules[0]["mark"]["strokeDash"] == [1.0, 3.0]
    # The dotted horizontal rule of the diagram is its own and starts at its plot area.
    horizontal = [
        item
        for item in _walk(diagram)
        if item.get("mark", {}).get("type") == "rule"
        and item.get("encoding", {}).get("y", {}).get("field") == "median_fitness"
    ]
    assert len(horizontal) == 1


def test_vertical_variant_has_the_grey_noise_square_behind_the_points():
    spec, payload = _render_vertical()
    fitness = payload["full_fitness"]["raw_fitness"]
    threshold = payload["threshold_diagnostics"]["reflected"].threshold
    diagram = spec["hconcat"][2]
    layers = diagram["layer"]
    squares = [
        i
        for i, layer in enumerate(layers)
        if layer["mark"]["type"] == "rect"
        and layer["mark"].get("color") == palette.BAND_GREY
    ]
    assert squares == [0]  # drawn first: behind the rules, the diagonal and the points
    square = layers[0]
    assert square["mark"]["color"] == palette.BAND_GREY  # the bootstrap band's fill
    assert square["mark"]["opacity"] == 0.3
    rows = spec["datasets"][square["data"]["name"]]
    assert rows == [
        {
            "death_scalar": float(fitness.min()),
            "birth_scalar": float(fitness.min()),
            "square_high": threshold,
        }
    ]
    # The diagram's own field names, so the diagram keeps its "Fitness at death" axis.
    assert square["encoding"]["x"]["field"] == "death_scalar"
    assert square["encoding"]["y"]["field"] == "birth_scalar"
    assert square["encoding"]["x2"]["field"] == "square_high"
    assert square["encoding"]["y2"]["field"] == "square_high"
    # Centred on the median: the minimum and the threshold are reflections of it.
    median = float(fitness.median())
    assert median - float(fitness.min()) == pytest.approx(threshold - median)
    first_point = next(
        i for i, layer in enumerate(layers) if layer["mark"]["type"] == "point"
    )
    assert squares[0] < first_point
    assert palette.LIGHT_GREY not in [
        item.get("mark", {}).get("color") for item in _walk(spec["hconcat"][1])
    ]


def test_vertical_variant_has_one_fitness_axis_ticked_at_the_data_extremes():
    spec, payload = _render_vertical()
    fitness = payload["full_fitness"]["raw_fitness"]
    (ticks,) = _y_ticks(spec["hconcat"][0])
    assert ticks == (
        float(fitness.max()),
        0.0,
        float(fitness.median()),
        float(fitness.min()),
    )
    assert not _y_ticks(spec["hconcat"][1])
    # The diagram's only y axis labels the median and the threshold, not the fitness ticks.
    (labels,) = _y_ticks(spec["hconcat"][2])
    assert float(fitness.min()) not in labels and 0.0 not in labels


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_vertical_variant_writes_the_median_and_threshold_values_at_the_left_of_their_lines(
    variant,
):
    spec, payload = _render_vertical(variant)
    median = float(payload["full_fitness"]["raw_fitness"].median())
    threshold = payload["threshold_diagnostics"]["reflected"].threshold
    diagram = spec["hconcat"][2]
    # The values are the labels of the diagram's own left axis (black, no ticks), not
    # text inside the plot area; the median rule owns that axis.
    axes = [
        layer["encoding"]["y"]["axis"]
        for layer in diagram["layer"]
        if isinstance(layer.get("encoding", {}).get("y", {}).get("axis"), dict)
    ]
    assert len(axes) == 1
    # Only the threshold: the median value is on the histogram's axis, not the diagram's.
    assert axes[0]["values"] == [threshold]
    assert axes[0]["format"] == ".3f"
    assert axes[0]["ticks"] is False and axes[0]["labelColor"] == "#000000"
    # The x axis carries no median tick.
    x_axis = next(
        layer["encoding"]["x"]["axis"]
        for layer in diagram["layer"]
        if layer.get("encoding", {}).get("x", {}).get("field") == "death_scalar"
        and isinstance(layer["encoding"]["x"].get("axis"), dict)
    )
    assert median not in x_axis["values"]
    fitness = payload["full_fitness"]["raw_fitness"]
    assert x_axis["values"] == [float(fitness.min()), 0.0, float(fitness.max())]
    assert x_axis["format"] == ".3f"
    # The dashed rule at the threshold is drawn in both variants, owning no axis.
    (rule,) = [
        layer
        for layer in diagram["layer"]
        if layer["mark"]["type"] == "rule"
        and layer.get("encoding", {}).get("y", {}).get("field") == "functional_threshold"
    ]
    assert rule["encoding"]["y"]["axis"] is None
    # The two-line caption stays on the right, annotated only.
    captions = [
        layer
        for layer in diagram["layer"]
        if layer["mark"]["type"] == "text" and layer["mark"].get("align") == "right"
    ]
    assert len(captions) == (variant == "annotated")
    # No separate median value between the tree and the diagram.
    assert not [
        item
        for item in _walk(spec["hconcat"][1])
        if item.get("mark", {}).get("type") == "text"
        and item.get("encoding", {}).get("text", {}).get("field") == "y"
    ]


@pytest.mark.parametrize("variant", ["vanilla", "annotated"])
def test_vertical_variant_keeps_valleys_stop_markers_and_the_legend(variant):
    spec, _ = _render_vertical(variant)
    diagram = spec["hconcat"][2]
    point_layers = _point_layers(diagram)
    shapes = {layer["mark"]["shape"] for layer in point_layers}
    assert shapes == {"triangle-up", "triangle-down"} | (
        {CROSS} if variant == "annotated" else set()
    )
    assert all(
        layer["mark"]["filled"] is True
        for layer in point_layers
        if layer["mark"]["shape"] != CROSS
    )
    texts = [
        layer["mark"]["text"]
        for layer in diagram["layer"]
        if layer["mark"]["type"] == "text"
        and isinstance(layer["mark"].get("text"), str)
        and layer["mark"].get("text") in {"peak", "valley", "≥1 stop codon"}
    ]
    expected = ["peak", "valley"] + (
        ["≥1 stop codon"] if variant == "annotated" else []
    )
    assert texts == expected
    tree_marks = {
        item.get("mark", {}).get("shape") for item in _walk(spec["hconcat"][1])
    }
    assert "triangle-down" in tree_marks
    stop_crosses = [
        item
        for item in _walk(spec["hconcat"][1])
        if item.get("mark", {}).get("filled") is False
        and item["mark"].get("color") == "#595959"
    ]
    assert bool(stop_crosses) == (variant == "annotated")


def test_vertical_variant_labels_come_from_the_registry():
    payload = _payload()
    spec = render_payload(get_spec(VERTICAL_ID), payload, _settings(), "vanilla").to_dict(
        validate=True
    )
    assert set(payload["manuscript_quantities"].requested_keys) == {
        "full_graph.persistence.ordinary_h0_pair_count",
        "full_graph.persistence.valley_count",
        "full_graph.vertex_count",
    }
    assert spec["title"]["text"] == "Full graph"


def test_vertical_variant_compiles_to_png():
    spec, _ = _render_vertical("annotated")
    assert vlc.vegalite_to_png(spec)[:8] == b"\x89PNG\r\n\x1a\n"


def test_vertical_variant_key_straddles_the_dotted_median_rule_in_four_rows(monkeypatch):
    monkeypatch.setattr(
        "dubanevics2026topological.figures.full_tree_persistence."
        "synonymous_valley_sequences",
        lambda valleys: frozenset({"VALLEY"}),
    )
    spec, payload = _render_vertical("annotated")
    scatter = spec["hconcat"][2]
    labels = ["peak", "valley", "synonymous", "≥1 stop codon"]
    keys = {
        layer["mark"]["text"]: layer["mark"]
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "text" and layer["mark"].get("text") in labels
    }
    assert list(keys) == labels  # one row per label, in reading order
    low, high = padded_scalar_domain(
        (-1.0, 1.1), {"fitness_tick_step": 0.1, "fitness_padding_fraction": 0.05}
    )
    median = float(payload["full_fitness"]["raw_fitness"].median())
    median_y = scatter["height"] * (high - median) / (high - low)
    # Peak and valley just above the rule, synonymous and stop codon just below it.
    assert keys["peak"]["y"] < keys["valley"]["y"] < median_y
    assert median_y < keys["synonymous"]["y"] < keys["≥1 stop codon"]["y"]
    assert keys["valley"]["y"] > median_y - 10 and keys["synonymous"]["y"] < median_y + 25
    assert all(mark["y"] < scatter["height"] for mark in keys.values())
    # One column, left-aligned at one x inside the diagram, on the right of it.
    assert len({mark["x"] for mark in keys.values()}) == 1
    assert scatter["width"] / 2 < keys["peak"]["x"] < scatter["width"]
    # The font size is the threshold caption's, and there is no backing box.
    caption = next(
        layer["mark"]
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "text" and layer["mark"].get("align") == "right"
    )
    assert {mark["fontSize"] for mark in keys.values()} == {caption["fontSize"]}
    rects = [layer for layer in scatter["layer"] if layer["mark"]["type"] == "rect"]
    assert len(rects) == 1  # the grey noise square only


def test_vertical_variant_closes_the_tree_histogram_seam_by_the_configured_overlap():
    settings = _settings()
    settings["figures"]["figures"][VERTICAL_ID]["persistence"]["tree_left_overlap_px"] = 30
    spec = render_payload(get_spec(VERTICAL_ID), _payload(), settings, "annotated").to_dict(
        validate=True
    )
    left, diagram = spec["hconcat"]
    assert spec["spacing"] == 0  # nothing between the (histogram + tree) pair and the diagram
    assert left["spacing"] == -30  # the "WT" label overlaps the histogram's empty edge
    assert [bool(i) for i in (left["hconcat"], diagram)] == [True, True]


def test_a_tree_table_can_pin_the_trees_x_extent_exactly():
    from dubanevics2026topological.figures import _charts

    segments = pl.DataFrame(
        {
            "x_start": [0.0, 5.0, 7.0, 3.0],
            "x_end": [5.0, 5.0, 9.0, 3.0],
            "y_start": [0.0, 0.0, 1.0, 2.0],
            "y_end": [0.0, 1.0, 1.0, 2.0],  # the last segment has zero length
        }
    )
    plot_data = SimpleNamespace(edge_segment_table=segments)
    assert _charts._tight_horizontal_domain(plot_data, {"tree": {}}) is None
    tight = {"tree": {"tight_horizontal": True}}
    assert _charts._tight_horizontal_domain(plot_data, tight) == (0.0, 9.0)
    config = _charts.tree_plot_config(
        SimpleNamespace(scalar_domain=(0.0, 1.0)),
        tight,
        {},
        segments,
        (1.0, 2.0),
        horizontal_domain=(0.0, 9.0),
    )
    assert config.horizontal_domain == (0.0, 9.0) and config.horizontal_padding == 0
    chart = alt.Chart(pl.DataFrame({"x": [0.0, 9.0]})).mark_line().encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0.0, 9.0], padding=0))
    )
    scale = _charts._without_nice_x(chart).to_dict()["encoding"]["x"]["scale"]
    assert scale["nice"] is False and scale["domain"] == [0.0, 9.0]


def test_stop_crosses_and_triangles_share_the_trees_marker_size_and_style():
    from dubanevics2026topological.figures import _charts

    spec, _ = _render_vertical("annotated")
    rendering = _settings()["figures"]["rendering"]
    size = _charts.tree_marker_size(rendering)
    diagram = spec["hconcat"][2]
    triangles = [
        layer
        for layer in _point_layers(diagram)
        if layer["mark"]["shape"] in {"triangle-up", "triangle-down"}
    ]
    crosses_diagram = [
        layer["mark"] for layer in _point_layers(diagram) if layer["mark"]["shape"] == CROSS
    ]
    crosses_tree = [
        item["mark"]
        for item in _walk(spec["hconcat"][1])
        if item.get("mark", {}).get("filled") is False
    ]
    assert triangles and crosses_diagram and crosses_tree
    assert {layer["mark"]["size"] for layer in triangles} == {size}
    for mark in crosses_diagram:
        for key in ("size", "strokeWidth", "opacity", "color", "shape"):
            assert mark[key] == crosses_tree[0][key], key
    # And the tree's own valley triangles and crosses use it too.
    assert {
        item["mark"]["size"]
        for item in _walk(spec["hconcat"][1])
        if item.get("mark", {}).get("shape") in {"triangle-down", CROSS}
    } == {size}


def test_a_synonymous_valley_is_a_teal_down_triangle_with_its_key_entry(monkeypatch):
    monkeypatch.setattr(
        "dubanevics2026topological.figures.full_tree_persistence."
        "synonymous_valley_sequences",
        lambda valleys: frozenset({"VALLEY"}),
    )
    spec = _render("annotated")
    scatter = spec["hconcat"][1]["vconcat"][1]
    layers = _point_layers(scatter)
    teal = [
        layer
        for layer in layers
        if layer["mark"].get("color") == palette.SYNONYMOUS_TEAL
    ]
    assert [layer["mark"]["shape"] for layer in teal] == ["triangle-down"]
    assert [r["birth_sequence"] for r in spec["datasets"][teal[0]["data"]["name"]]] == [
        "VALLEY"
    ]
    # A stop valley stays a grey marker with its cross; only the synonymous one is teal.
    crosses = [layer for layer in layers if layer["mark"].get("shape") == CROSS]
    assert crosses
    texts = {
        layer["mark"]["text"]
        for layer in scatter["layer"]
        if layer["mark"]["type"] == "text" and isinstance(layer["mark"].get("text"), str)
    }
    assert {"peak", "valley", "synonymous", "≥1 stop codon", "●", "×"} <= texts


def test_vanilla_marks_no_synonymous_valley(monkeypatch):
    monkeypatch.setattr(
        "dubanevics2026topological.figures.full_tree_persistence."
        "synonymous_valley_sequences",
        lambda valleys: frozenset({"VALLEY"}),
    )
    spec = _render("vanilla")
    layers = _point_layers(spec["hconcat"][1]["vconcat"][1])
    assert not [l for l in layers if l["mark"].get("color") == palette.SYNONYMOUS_TEAL]
