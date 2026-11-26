"""Tests for the shared chart vocabulary in ``figures/_charts.py``."""

from dataclasses import dataclass
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest
from _figure_labels import SENTINEL, label_quantities

from dubanevics2026topological.figures._charts import (
    THRESHOLD_LABEL_PADDING,
    functional_threshold_layers,
    functional_threshold_text,
    histogram_count_axis,
    landscape_population_subtitle,
    plot_data_with_metrics,
    publication_scalar_scales,
    study_biology_persistence_chart,
    threshold_label_padding,
)


def test_branch_vertex_counts_follow_reversed_display_edges():
    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        node_table: pl.DataFrame

    edges = pl.DataFrame(
        {
            "source_tree_node_index": [0, 1, 1],
            "target_tree_node_index": [1, 2, 6],
            "source_scalar": [0.0, 0.3, 0.3],
            "target_scalar": [0.3, 0.9, 0.8],
        }
    )
    nodes = pl.DataFrame(
        {
            "tree_node_index": list(range(7)),
            "dna_sequence": ["ROOT", "MERGE", "AAA", "I1", "I2", "I3", "AAG"],
        }
    )
    branches = pl.DataFrame(
        {
            "source_tree_node_index": [1, 2, 6],
            "target_tree_node_index": [0, 1, 1],
            "branch_vertex_count": [1, 1, 4],
            "synonymous_endpoints": [0, 1, 0],
        }
    )
    result = plot_data_with_metrics(
        SimpleNamespace(plot_data=PlotData(edges, edges, nodes), branch_table=branches)
    )
    for table in (result.edge_table, result.edge_segment_table):
        assert table["branch_vertex_count"].to_list() == [1, 1, 4]
        assert table["vertex_count"].to_list() == [1, 1, 4]
        assert table["source_label"].to_list() == ["ROOT", "MERGE", "MERGE"]
        assert table["target_label"].to_list() == ["MERGE", "AAA", "AAG"]

    # The root-to-merger and direct maximum-to-merger branches each own one
    # upper endpoint; the third owns its maximum and three interior vertices.
    # The root is separately owned, so all seven vertices are accounted for.
    assert sum(result.edge_table["vertex_count"].to_list()) + 1 == nodes.height


def test_persistence_births_use_immediate_displayed_branch_counts():
    from dubanevics2026topological.figures._charts import persistence_with_counts

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        node_table: pl.DataFrame

    displayed = pl.DataFrame(
        {
            "target_label": ["AAA", "AAC"],
            "vertex_count": [1, 4],
        }
    )
    plot_data = PlotData(
        displayed,
        pl.DataFrame({"dna_sequence": ["AAA", "AAC", "AAT"]}),
    )
    pairs = pl.DataFrame(
        {
            "birth_label": ["AAA", "AAC"],
            "death_label": ["AAT", "AAA"],
            "pair_kind": ["ordinary", "extended"],
            "birth_scalar": [0.9, 0.8],
            "death_scalar": [0.2, 0.1],
        }
    )
    counted = persistence_with_counts(pairs, plot_data, (1.0, 4.0))
    assert counted["vertex_count"].to_list() == [1, 4]
    assert counted.select("birth_label", "birth_scalar", "death_scalar").equals(
        pairs.select("birth_label", "birth_scalar", "death_scalar")
    )


def test_singleton_persistence_birth_uses_count_one():
    from dubanevics2026topological.figures._charts import persistence_with_counts

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        node_table: pl.DataFrame

    plot_data = PlotData(
        pl.DataFrame(schema={"target_label": pl.String, "vertex_count": pl.Int64}),
        pl.DataFrame({"dna_sequence": ["AAA"]}),
    )
    pairs = pl.DataFrame(
        {"birth_label": ["AAA"], "birth_scalar": [0.5], "death_scalar": [0.5]}
    )
    assert persistence_with_counts(pairs, plot_data, (1.0, 1.0))[
        "vertex_count"
    ].to_list() == [1]


def test_plot_data_with_metrics_reads_a_real_topographer_branch_table():
    """Regression: Topographer's branch table names the column ``branch_vertex_count``.

    The tree is a real Topographer split tree (the toy path of the bootstrap
    tests), not a hand-made frame. Its hills are b = {b, a}, d = {d} and
    f = {f, g}; the branch vertex count includes the upper critical vertex, so
    ``vertex_count`` must equal it rather than gain a second ``+ 1``.
    """
    from test_bootstrap_smoothed_tree import _toy

    _, tree = _toy()
    assert "branch_member_count" not in tree.branch_table.columns
    result = plot_data_with_metrics(tree)

    expected = {
        (int(row["source_tree_node_index"]), int(row["target_tree_node_index"])): int(
            row["branch_vertex_count"]
        )
        for row in tree.branch_table.iter_rows(named=True)
    }
    assert sorted(expected.values()) == [1, 1, 2, 2]
    for table in (result.edge_table, result.edge_segment_table):
        got = {
            frozenset(
                (int(row["source_tree_node_index"]), int(row["target_tree_node_index"]))
            ): int(row["vertex_count"])
            for row in table.iter_rows(named=True)
        }
        assert got == {frozenset(key): value for key, value in expected.items()}
        assert table["vertex_count"].to_list() == table["branch_vertex_count"].to_list()
    # Every sequence of the path is owned by one displayed branch except the root.
    assert result.edge_table["vertex_count"].sum() + 1 == 7


def test_branch_metric_join_rejects_negative_member_count():
    from dubanevics2026topological.figures._charts import add_metrics

    edges = pl.DataFrame({"source_tree_node_index": [0], "target_tree_node_index": [1]})
    metrics = pl.DataFrame(
        {
            "source_tree_node_index": [0],
            "target_tree_node_index": [1],
            "branch_vertex_count": [-1],
        }
    )
    with pytest.raises(ValueError, match="nonnegative integers"):
        add_metrics(edges, metrics)


def test_branch_metric_join_rejects_unmatched_or_null_count():
    from dubanevics2026topological.figures._charts import add_metrics

    edges = pl.DataFrame({"source_tree_node_index": [0], "target_tree_node_index": [1]})
    metrics = pl.DataFrame(
        {
            "source_tree_node_index": [2],
            "target_tree_node_index": [3],
            "branch_vertex_count": [4],
        }
    )
    with pytest.raises(ValueError, match="matching scientific branch metric"):
        add_metrics(edges, metrics)


def test_birth_matching_rejects_duplicate_displayed_endpoint_identity():
    from dubanevics2026topological.figures._charts import persistence_with_counts

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        node_table: pl.DataFrame

    plot_data = PlotData(
        pl.DataFrame(
            {
                "target_label": ["AAA", "AAA"],
                "vertex_count": [1, 4],
            }
        ),
        pl.DataFrame({"dna_sequence": ["AAA"]}),
    )
    pairs = pl.DataFrame({"birth_label": ["AAA"]})
    with pytest.raises(ValueError, match="Duplicate branch endpoint identities"):
        persistence_with_counts(pairs, plot_data, (1.0, 4.0))


def test_persistence_annotation_defaults_are_the_palette_entries():
    from dubanevics2026topological.figures import palette
    from dubanevics2026topological.figures._charts import peak_annotation_style

    # Colour means annotation (ADR 0012): solid, whatever opacity a stale key says.
    stale = {"stop_codon_persistence_opacity": 0.4}
    assert peak_annotation_style("stop_codon", stale, color="own") == (
        palette.STOP_CODON_CRIMSON,
        1.0,
    )
    assert peak_annotation_style("wild_type", {}, color="own") == (
        palette.WILD_TYPE_RED,
        1.0,
    )
    assert peak_annotation_style(None, {}, color="own") == ("own", 1.0)
    configured = {"stop_codon_color": "#4a4a4a", "wild_type_color": "#111111"}
    assert peak_annotation_style("stop_codon", configured, color="own")[0] == "#4a4a4a"
    assert peak_annotation_style("wild_type", configured, color="own")[0] == "#111111"


def test_a_peak_carries_one_annotation_wild_type_then_stop():
    from dubanevics2026topological.figures._charts import (
        annotated_peaks,
        peak_annotation_kinds,
        unannotated_peaks,
    )

    pairs = pl.DataFrame(
        {
            "birth_label": ["WT", "STOPWT", "STOP", "PLAIN"],
            "birth_has_stop_codon": [True, True, True, False],
            # The paired endpoint alone never annotates a peak.
            "death_has_stop_codon": [False, False, False, True],
        }
    )
    kwargs = {"wild_type": frozenset({"WT"})}
    assert peak_annotation_kinds(pairs, **kwargs) == [
        "wild_type",
        "stop_codon",
        "stop_codon",
        None,
    ]
    assert peak_annotation_kinds(pairs) == [
        "stop_codon",
        "stop_codon",
        "stop_codon",
        None,
    ]
    assert unannotated_peaks(pairs, **kwargs)["birth_label"].to_list() == ["PLAIN"]
    assert annotated_peaks(pairs, **kwargs)["birth_label"].to_list() == [
        "WT",
        "STOPWT",
        "STOP",
    ]


def test_peak_marker_layers_keep_the_outcome_fill_and_annotation_colour():
    from dubanevics2026topological.figures import palette
    from dubanevics2026topological.figures._charts import (
        FIVE_POINT_STAR,
        OPEN_X,
        peak_marker_layers,
    )

    table = pl.DataFrame(
        {
            "death_scalar": [0.1, 0.2, 0.3],
            "birth_scalar": [0.9, 0.8, 0.7],
            "birth_label": ["plain", "stop", "wt"],
            "death_label": ["x", "y", "z"],
            "birth_has_stop_codon": [False, True, False],
        }
    )
    layers = peak_marker_layers(
        table,
        {},
        color=palette.EMPHASIS_BLACK,
        annotation="absorbed",
        wild_type=frozenset({"wt"}),
    )
    marks = [layer.to_dict() for layer in layers]
    # Plain peak, crimson stop peak (above it) and wild-type star; no × layer.
    assert [mark["mark"]["shape"] for mark in marks] == [
        "triangle-up", "triangle-up", FIVE_POINT_STAR
    ]
    assert [mark["mark"]["color"] for mark in marks] == [
        palette.EMPHASIS_BLACK, palette.STOP_CODON_CRIMSON, palette.WILD_TYPE_RED
    ]
    assert not any(OPEN_X == mark["mark"]["shape"] for mark in marks)
    # A table without stop flags yields one plain layer; an empty one, none.
    assert (
        len(
            peak_marker_layers(
                table.drop("birth_has_stop_codon"), {}, color="#000000", annotation="a"
            )
        )
        == 1
    )
    assert peak_marker_layers(table.clear(), {}, color="#000000", annotation="a") == []


def test_vanilla_tree_variants_omit_markers_and_structural_events(monkeypatch):
    import altair as alt
    from topographer.plotting import TreePresentationConfig

    from dubanevics2026topological.figures import _charts as charts

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        node_table: pl.DataFrame

    edge = pl.DataFrame(
        {
            "source_tree_node_index": [0],
            "target_tree_node_index": [1],
            "source_scalar": [0.5],
            "target_scalar": [0.9],
            "x_source": [0.0],
            "x_target": [1.0],
            "y_source": [0.9],
            "y_target": [0.5],
        }
    )
    raw = PlotData(
        edge,
        edge,
        pl.DataFrame({"tree_node_index": [0, 1], "dna_sequence": ["AAA", "AAT"]}),
    )
    tree_analysis = SimpleNamespace(
        plot_data=raw,
        branch_table=pl.DataFrame(
            {
                "source_tree_node_index": [1],
                "target_tree_node_index": [0],
                "branch_vertex_count": [1],
                "synonymous_endpoints": [1],
            }
        ),
        all_stop_marker_table=pl.DataFrame(
            schema={"dna_sequence": pl.String, "x": pl.Float64, "y": pl.Float64}
        ),
        wild_type_marker_table=pl.DataFrame(
            {"dna_sequence": ["AAA"], "aa_sequence": ["K"]}
        ),
    )
    analysis = SimpleNamespace(scalar_domain=(0.0, 1.0), scalar_label="Fitness")
    calls = []

    def fake_plot_tree(plot_data, *, presentation, **kwargs):
        calls.append((presentation, kwargs))
        return alt.Chart(pl.DataFrame({"scalar": [0.5]})).mark_line()

    monkeypatch.setattr(charts, "plot_tree", fake_plot_tree)
    display_data = charts.plot_data_with_metrics(tree_analysis)
    for linked in (False, True):
        charts.tree_chart(
            display_data,
            tree_analysis,
            analysis,
            {"tree": {"width": 420, "height": 250}},
            {},
            (1.0, 1.0),
            linked=linked,
            annotated=False,
        )

    assert len(calls) == 2
    for presentation, kwargs in calls:
        assert isinstance(presentation, TreePresentationConfig)
        assert presentation.annotation_layers == ()
        assert kwargs["show_structural_events"] is False


def test_annotated_marker_contract_uses_solid_star_and_solid_x():
    from dubanevics2026topological.figures._charts import marker_config

    rendering = {
        "wild_type_shape": "star",
        "wild_type_color": "#d62728",
        "wild_type_size": 200,
        "stop_codon_shape": "x",
        "stop_codon_color": "#4a4a4a",
        "stop_codon_size": 110,
        "persistence_point_size": 70,
    }
    wild_type = marker_config(rendering, "wild_type", default_shape="star")
    stop = marker_config(rendering, "stop_codon", default_shape="x")
    assert wild_type.shape.startswith("M")
    assert wild_type.shape.count("L") == 9
    assert wild_type.color == "#d62728"
    assert wild_type.opacity == stop.opacity == 1.0  # solid, never translucent
    assert (stop.shape, stop.color, stop.size, stop.filled) == (
        "M-1,-1L1,1M-1,1L1,-1",
        "#4a4a4a",
        110,
        False,
    )
    assert stop.size > rendering["persistence_point_size"]


def test_fitness_scales_override_presentation_padding():
    import altair as alt

    from dubanevics2026topological.figures import _charts as charts

    panel = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_point()
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[-2, 2])))
    )
    chart = charts.publication_scalar_scales(panel, (-1.2, 1.4))
    scale = chart.to_dict()["encoding"]["y"]["scale"]
    assert scale["domain"] == pytest.approx([-1.33, 1.53])
    assert scale["nice"] is False


def test_study_metric_scales_use_rounded_padded_fitness_ticks():
    import altair as alt

    from dubanevics2026topological.figures._charts import publication_scalar_scales

    chart = alt.Chart(pl.DataFrame({"y": [0.2]})).mark_point().encode(y="y:Q")
    spec = publication_scalar_scales(
        chart, (-1.17480288668069, 1.395791684395264), {}
    ).to_dict()
    axis = spec["encoding"]["y"]
    assert axis["scale"]["domain"] == pytest.approx([-1.33, 1.53])
    assert axis["axis"]["values"] == [-1.2, 0.0, 1.4]
    assert axis["axis"]["format"] == ".1f"


def test_exact_domain_keeps_the_display_domain_and_extra_tick_has_three_decimals():
    from dubanevics2026topological.figures._charts import publication_scalar_scales

    chart = alt.Chart(pl.DataFrame({"y": [0.2]})).mark_point().encode(y="y:Q")
    spec = publication_scalar_scales(
        chart,
        (-1.2, 1.4),
        {},
        exact_domain=True,
        extra_ticks={"y": [-0.507774]},
    ).to_dict()
    axis = spec["encoding"]["y"]
    # The drawn domain is the configured one, unpadded; three regular ticks plus the
    # threshold, which alone is written with three decimals.
    assert axis["scale"]["domain"] == pytest.approx([-1.2, 1.4])
    assert axis["axis"]["values"] == [-1.2, -0.507774, 0.0, 1.4]
    assert axis["axis"]["format"] == ".1f"
    assert axis["axis"]["labelExpr"] == (
        "format(datum.value, (abs(datum.value - (-0.507774)) < 1e-9) ? '.3f' : '.1f')"
    )


def test_three_regular_ticks_need_no_label_expression_without_extra_ticks():
    from dubanevics2026topological.figures._charts import publication_scalar_scales

    chart = alt.Chart(pl.DataFrame({"y": [0.2]})).mark_point().encode(y="y:Q")
    axis = publication_scalar_scales(
        chart, (-1.2, 1.4), {}, exact_domain=True
    ).to_dict()["encoding"]["y"]["axis"]
    assert axis["values"] == [-1.2, 0.0, 1.4]
    assert "labelExpr" not in axis


def _key_text_marks(layer):
    found = []

    def visit(node):
        if isinstance(node, dict):
            mark = node.get("mark")
            if isinstance(mark, dict) and mark.get("type") == "text":
                found.append(mark)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(layer.to_dict())
    return found


def test_tree_key_is_one_horizontal_line_wild_type_then_stop_codon():
    from dubanevics2026topological.figures._charts import annotation_legend

    marks = _key_text_marks(
        annotation_legend(520, 250, {"annotation_legend_spacing": 190})
    )
    assert [mark["text"] for mark in marks] == [
        "★",
        "Wild type",
        "×",
        "Stop codon containing peaks",
    ]
    # One row: every glyph and label shares one y, near the panel's bottom edge.
    assert {mark["y"] for mark in marks} == {250 - 12}
    xs = [mark["x"] for mark in marks]
    assert xs == sorted(xs) and len(set(xs)) == 4
    # Entries are one spacing apart, side by side.
    assert xs[2] - xs[0] == 190


def test_tree_key_names_only_the_markers_the_panel_draws():
    from dubanevics2026topological.figures._charts import annotation_legend

    wild_only = _key_text_marks(annotation_legend(520, 250, {}, stop_codon=False))
    assert [mark["text"] for mark in wild_only] == ["★", "Wild type"]
    stop_only = _key_text_marks(annotation_legend(520, 250, {}, wild_type=False))
    assert [mark["text"] for mark in stop_only] == ["×", "Stop codon containing peaks"]


def test_study_metric_scalar_scales_update_altair_unit_spec():
    import altair as alt

    from dubanevics2026topological.figures._charts import publication_scalar_scales

    unit = alt.UnitSpec(
        mark="point",
        data=alt.Data(values=[{"y": 0.2}]),
        encoding=alt.Encoding(y=alt.Y("y:Q")),
    )
    spec = publication_scalar_scales(unit, (-1.0, 1.0)).to_dict()
    assert spec["encoding"]["y"]["scale"]["domain"] == [-1.1, 1.1]


def test_publication_scalar_domains_do_not_overwrite_legend_layout_scales():
    import altair as alt

    from dubanevics2026topological.figures._charts import publication_scalar_scales

    scalar_panel = (
        alt.Chart(pl.DataFrame({"death_scalar": [0.2], "birth_scalar": [0.8]}))
        .mark_point()
        .encode(
            x=alt.X("death_scalar:Q"),
            y=alt.Y("birth_scalar:Q"),
        )
    )
    legend_panel = (
        alt.Chart(pl.DataFrame({"legend_x": [10], "legend_y": [20], "label": ["1"]}))
        .mark_text()
        .encode(
            x=alt.X("legend_x:Q", scale=alt.Scale(domain=[0, 250])),
            y=alt.Y("legend_y:Q", scale=alt.Scale(domain=[0, 250])),
            text="label:N",
        )
    )
    spec = publication_scalar_scales(
        alt.layer(scalar_panel, legend_panel), (-1.2, 1.4)
    ).to_dict()

    encodings = [layer["encoding"] for layer in spec["layer"] if "encoding" in layer]
    scalar_encoding = next(
        encoding
        for encoding in encodings
        if "birth_scalar" in encoding.get("y", {}).get("field", "")
    )
    legend_encoding = next(
        encoding
        for encoding in encodings
        if encoding.get("y", {}).get("field") == "legend_y"
    )
    assert scalar_encoding["x"]["scale"]["domain"] == pytest.approx([-1.33, 1.53])
    assert scalar_encoding["y"]["scale"]["domain"] == pytest.approx([-1.33, 1.53])
    assert legend_encoding["x"]["scale"]["domain"] == [0, 250]
    assert legend_encoding["y"]["scale"]["domain"] == [0, 250]


def test_tree_and_persistence_share_full_tree_count_scale_and_legend():
    from dubanevics2026topological.figures._charts import tree_plot_config

    rendering = {
        "branch_count_title": "Branch\\nvertex\\ncount",
        "legend_length": 150,
    }
    analysis = SimpleNamespace(
        scalar_domain=(0.0, 1.0),
        scalar_label="Fitness",
    )
    figure_settings = {"tree": {"width": 420, "height": 250}}
    edge_table = pl.DataFrame({"vertex_count": [1, 100]})
    count_domain = (1.0, 100.0)
    tree_scale = tree_plot_config(
        analysis, figure_settings, rendering, edge_table, count_domain
    ).edge_color_scale
    from dubanevics2026topological.figures._charts import color_scale

    persistence_scale = color_scale(rendering, count_domain)
    assert tree_scale.attribute == persistence_scale.attribute == "vertex_count"
    assert tree_scale.domain == persistence_scale.domain == (1.0, 100.0)
    assert tree_scale.title == persistence_scale.title == "Branch\nvertex\ncount"
    assert tree_scale.one_value is None
    assert tree_scale.colors == persistence_scale.colors


def test_publication_scalar_scales_round_bounds_and_pad_by_five_percent() -> None:
    chart = (
        alt.Chart(pl.DataFrame({"birth_scalar": [0.8], "death_scalar": [0.2]}))
        .mark_point()
        .encode(x="death_scalar:Q", y="birth_scalar:Q")
    )

    spec = publication_scalar_scales(chart, (-0.83, 0.94), {}).to_dict()

    assert spec["encoding"]["x"]["scale"]["domain"] == pytest.approx([-0.995, 1.095])
    assert spec["encoding"]["y"]["scale"]["domain"] == pytest.approx([-0.995, 1.095])
    assert spec["encoding"]["x"]["axis"]["values"] == [-0.9, 0.0, 1.0]
    assert spec["encoding"]["x"]["axis"]["format"] == ".1f"


def test_branch_metrics_join_by_undirected_identity_at_scalar_ties():
    from dataclasses import dataclass

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        node_table: pl.DataFrame

    edges = pl.DataFrame(
        {
            "source_tree_node_index": [0, 1, 2],
            "target_tree_node_index": [1, 2, 3],
            "source_scalar": [0.0, 0.5, 0.5],
            "target_scalar": [0.5, 0.5, 0.9],
            "x_source": [0.0, 1.0, 2.0],
            "x_target": [1.0, 2.0, 3.0],
            "y_source": [0.0, 0.5, 0.5],
            "y_target": [0.5, 0.5, 0.9],
        }
    )
    analysis = SimpleNamespace(
        plot_data=PlotData(
            edges,
            edges,
            pl.DataFrame(
                {
                    "tree_node_index": [0, 1, 2, 3],
                    "dna_sequence": ["ROOT", "A", "B", "C"],
                }
            ),
        ),
        branch_table=pl.DataFrame(
            {
                "source_tree_node_index": [1, 1, 3],
                "target_tree_node_index": [0, 2, 2],
                "branch_vertex_count": [1, 1, 2],
                "synonymous_endpoints": [0, 1, 0],
            }
        ),
    )

    result = plot_data_with_metrics(analysis)
    assert result.edge_table["vertex_count"].to_list() == [1, 1, 2]
    assert result.edge_table["synonymous_endpoints"].to_list() == [0, 1, 0]
    assert result.edge_segment_table["vertex_count"].to_list() == [1, 1, 2]
    # The one metric-only member represents a regular vertex omitted from the
    # reduced display tree, in addition to the separately owned root.
    assert sum(result.edge_table["vertex_count"].to_list()) + 1 == 5


def test_histogram_count_axis_uses_sparse_compact_integer_ticks():
    comparison = histogram_count_axis(
        (0.0, 20209 * 1.05), title="Vertex count"
    ).to_dict()
    candidate = histogram_count_axis((0.0, 7048 * 1.05), title="Vertex count").to_dict()
    small = histogram_count_axis((0.0, 1.05), title="Vertex count").to_dict()
    assert comparison["values"] == [0, 10000, 20000]
    assert candidate["values"] == [0, 5000]
    assert small["values"] == [0, 1]
    assert comparison["format"] == "~s"
    assert "datum.value === 0 ? '0'" in comparison["labelExpr"]


def test_biology_persistence_baseline_is_mid_grey_without_metric_encoding():
    table = pl.DataFrame(
        {
            "birth_scalar": [1.0],
            "death_scalar": [0.0],
            "persistence": [1.0],
            "birth_label": ["peak"],
            "death_label": ["root"],
        }
    )
    chart = study_biology_persistence_chart(
        table,
        SimpleNamespace(scalar_domain=(0.0, 1.0)),
        {"persistence": {"point_size": 113}},
        {},
    )
    spec = chart.to_dict()
    points = [
        node
        for node in _walk(spec)
        if isinstance(node.get("mark"), dict) and node["mark"].get("type") == "point"
    ]
    assert len(points) == 1
    assert points[0]["mark"]["color"] == "#737373"  # peaks are filled mid-grey
    assert points[0]["mark"]["shape"] == "triangle-up"
    assert points[0]["mark"]["filled"] is True
    assert points[0]["mark"]["size"] == 113
    assert points[0]["mark"]["opacity"] == 1.0
    assert all(
        "color" not in node.get("encoding", {})
        and "shape" not in node.get("encoding", {})
        for node in _walk(spec)
    )


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _legend_marks(layer_spec, mark_type):
    return [
        layer["mark"]
        for layer in layer_spec["layer"]
        if layer["mark"].get("type") == mark_type
    ]


def test_overlay_colorbar_layers_horizontal_legend_at_pixel_anchor():
    from dubanevics2026topological.figures._charts import (
        horizontal_count_legend,
        overlay_colorbar,
    )

    tree = alt.layer(
        alt.Chart(pl.DataFrame({"a": [1.0]})).mark_point().encode(x="a:Q")
    ).properties(width=420, height=250)
    legend = horizontal_count_legend(
        {"legend_length": 55, "branch_count_title": "Count"},
        (1.0, 1000.0),
        zero_color="#737373",
    )
    chart = overlay_colorbar(tree, legend, x_fraction=0.5, top_inset=8, width=420)

    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    assert (spec["width"], spec["height"]) == (420, 250)
    assert spec["resolve"]["scale"]["color"] == "independent"
    legend_spec = spec["layer"][1]
    texts = _legend_marks(legend_spec, "text")
    # Title above the bar, then swatch and tick labels below it.
    assert texts[0]["text"] == "Count"
    assert {mark["text"] for mark in texts} >= {"0", "1", "10", "100", "1,000"}
    assert texts[0]["y"] == 8
    assert all(mark["y"] > texts[0]["y"] for mark in texts[1:])
    rects = _legend_marks(legend_spec, "rect")
    cells = [r for r in rects if isinstance(r.get("color"), str)]
    assert len(cells) == 1  # the zero swatch only
    ramp = [r for r in rects if isinstance(r.get("color"), dict)]
    assert len(ramp) == 1
    assert ramp[0]["color"]["gradient"] == "linear"
    bar_left = min(m["x"] - m["width"] / 2 for m in ramp)
    bar_right = max(m["x"] + m["width"] / 2 for m in ramp)
    swatch_left = min(m["x"] - m["width"] / 2 for m in cells)
    # The whole legend (swatches plus bar) is centred at x_fraction * width.
    assert texts[0]["x"] == pytest.approx(0.5 * 420)  # the title marks the centre
    assert swatch_left < bar_left < bar_right
    assert bar_right - bar_left == pytest.approx(55)
    # The source legend keeps its local frame; the overlay works on a copy.
    assert legend.width != alt.Undefined


def test_single_count_legend_has_uniform_bar_and_tick():
    from dubanevics2026topological.figures._charts import horizontal_count_legend

    legend = horizontal_count_legend(
        {"legend_length": 55, "count_legend_ticks": [10], "count_alpha_min": 0.2},
        (1.0, 1.0),
    ).to_dict()
    rects = _legend_marks(legend, "rect")
    assert len(rects) == 1
    assert rects[0]["width"] == 55
    assert rects[0]["color"] == "rgba(0,0,0,0.2)"
    tick = next(mark for mark in _legend_marks(legend, "text") if mark["text"] == "1")
    assert tick["x"] == rects[0]["x"]


def test_overlay_colorbar_keeps_legend_inside_the_panel():
    from dubanevics2026topological.figures._charts import (
        horizontal_count_legend,
        overlay_colorbar,
    )

    tree = alt.layer(alt.Chart(pl.DataFrame({"a": [1.0]})).mark_point())
    tree = tree.properties(width=420, height=250)
    legend = horizontal_count_legend({"legend_length": 110}, (1.0, 1000.0))
    spec = overlay_colorbar(
        tree, legend, x_fraction=1.0, top_inset=8, width=420
    ).to_dict()
    marks = [layer["mark"] for layer in spec["layer"][1]["layer"]]
    right_edges = [
        mark["x"] + mark.get("width", 0) / 2 for mark in marks if mark["type"] == "rect"
    ]
    assert max(right_edges) <= 420
    # Text is centred, so allow half its estimated width (0.55 em per glyph).
    for mark in marks:
        if mark["type"] == "text":
            half = 0.55 * mark["fontSize"] * len(mark["text"]) / 2
            assert 0 <= mark["x"] - half and mark["x"] + half <= 420


def test_persistence_chart_and_tree_draw_higher_counts_last(monkeypatch):
    from dubanevics2026topological.figures import _charts as charts

    persistence_kwargs = {}
    tree_tables = []

    def fake_persistence(table, **kwargs):
        persistence_kwargs.update(kwargs)
        return alt.Chart(table).mark_point()

    def fake_tree(plot_data, **kwargs):
        tree_tables.append(plot_data.edge_table)
        return alt.Chart(pl.DataFrame({"scalar": [0.5]})).mark_line()

    monkeypatch.setattr(charts, "plot_persistence", fake_persistence)
    monkeypatch.setattr(charts, "plot_tree", fake_tree)
    table = pl.DataFrame(
        {
            "birth_scalar": [0.8, 0.7],
            "death_scalar": [0.2, 0.1],
            "vertex_count": [9, 2],
        }
    )
    charts.persistence_chart(
        table, {}, {}, (2.0, 9.0), (0.0, 1.0), include_config=False
    )
    assert persistence_kwargs["draw_order_by"] == "vertex_count"

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        node_table: pl.DataFrame

    edges = pl.DataFrame({"vertex_count": [5, 1, 9, 2], "source_label": list("abcd")})
    charts.tree_chart(
        PlotData(edges, edges, pl.DataFrame()),
        SimpleNamespace(),
        SimpleNamespace(scalar_domain=(0.0, 1.0)),
        {"tree": {"width": 420, "height": 250}},
        {},
        (1.0, 9.0),
        linked=False,
        annotated=False,
    )
    assert tree_tables[0]["vertex_count"].to_list() == [1, 2, 5, 9]


def _role_layer(role):
    """A stand-in layer whose mark carries its ``role`` as its description."""
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_point(description=role)


def _marks_bottom_to_top(spec):
    """The marks of a layered spec in draw order (the first drawn is at the bottom)."""
    if "layer" in spec:
        for layer in spec["layer"]:
            yield from _marks_bottom_to_top(layer)
    else:
        yield spec["mark"]


def test_compose_tree_layers_fixes_the_stacking_order_whatever_the_call_order():
    from dubanevics2026topological.figures._charts import compose_tree_layers

    chart = compose_tree_layers(
        _role_layer("branches"),
        underlays=[_role_layer("threshold rule")],
        wild_type=[_role_layer("star"), _role_layer("star label")],
        keys=[_role_layer("key")],
        stop_markers=[_role_layer("stop crosses")],
        count_labels=[_role_layer("count labels")],
        overlays=[_role_layer("absorbed branches"), _role_layer("highlight")],
    )
    roles = [mark["description"] for mark in _marks_bottom_to_top(chart.to_dict())]
    assert roles == [
        "threshold rule",
        "branches",
        "absorbed branches",
        "highlight",
        "count labels",
        "stop crosses",
        "key",
        "star",
        "star label",
    ]
    # A role left out leaves the others where they were.
    bare = compose_tree_layers(
        _role_layer("branches"),
        wild_type=[_role_layer("star")],
        overlays=[_role_layer("highlight")],
    )
    assert [m["description"] for m in _marks_bottom_to_top(bare.to_dict())] == [
        "branches",
        "highlight",
        "star",
    ]


def test_tree_chart_stacks_overlays_labels_stop_crosses_and_star_in_order(
    monkeypatch,
):
    from dubanevics2026topological.figures import _charts as charts

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        node_table: pl.DataFrame

    edges = pl.DataFrame({"vertex_count": [1, 2], "source_label": ["a", "b"]})
    tree_analysis = SimpleNamespace(
        all_stop_marker_table=pl.DataFrame(
            {
                "dna_sequence": ["TAA"],
                "aa_sequence": ["*"],
                "x": [0.2],
                "y": [0.5],
                "display_event": ["maximum"],
            }
        ),
        wild_type_marker_table=pl.DataFrame(
            {
                "dna_sequence": ["AAA"],
                "aa_sequence": ["K"],
                "marker_label": ["WT"],
                "x": [0.1],
                "y": [0.0],
            }
        ),
    )
    monkeypatch.setattr(charts, "plot_tree", lambda *a, **k: _role_layer("branches"))
    # The overlays and labels are handed in written in the wrong order on purpose:
    # ``tree_chart`` places them, the caller does not.
    chart = charts.tree_chart(
        PlotData(edges, edges, pl.DataFrame()),
        tree_analysis,
        SimpleNamespace(scalar_domain=(0.0, 1.0)),
        {"tree": {"width": 420, "height": 250}},
        {},
        (1.0, 2.0),
        linked=False,
        annotated=True,
        count_labels=[_role_layer("count labels")],
        overlays=[_role_layer("absorbed branches")],
    )
    marks = list(_marks_bottom_to_top(chart.to_dict()))

    def positions(wanted):
        return [i for i, mark in enumerate(marks) if wanted(mark)]

    def role(name):
        return positions(lambda mark: mark.get("description") == name)

    cross = positions(lambda mark: mark.get("shape") == charts.OPEN_X)
    star = positions(lambda mark: mark.get("shape") == charts.FIVE_POINT_STAR)
    key = positions(lambda mark: mark.get("type") == "text" and "text" in mark)
    assert role("branches") == [0]
    assert cross and star and key
    assert (
        max(role("branches"))
        < max(role("absorbed branches"))
        < max(role("count labels"))
        < min(cross)
        < min(key)
        < min(star)
    )
    # The wild-type label is the last mark of all.
    assert marks[-1]["type"] == "text"


def test_functional_threshold_text_prints_the_registry_display_text_as_given():
    # The text is the registry's, never reformatted: six decimals stay six.
    assert functional_threshold_text("-0.507774") == "Functional threshold = -0.507774"
    assert functional_threshold_text("-0.5") == "Functional threshold = -0.5"


@pytest.mark.parametrize(
    ("graph", "heading"), [("published", "Papkou"), ("candidate", "Topographer + FDR")]
)
def test_population_subtitle_is_read_from_the_registry_not_recomputed(graph, heading):
    quantities = label_quantities()
    subtitle = landscape_population_subtitle(quantities, graph)
    peaks = SENTINEL[f"{graph}_graph.peak_count"]
    functional = SENTINEL[f"{graph}_graph.functional_count"]
    total = SENTINEL[f"{graph}_graph.vertex_count"]
    threshold = SENTINEL[f"thresholds.{graph}_functional.value"]
    assert subtitle == [
        f"Papkou et al. (2023) · {peaks} peaks",
        (
            f"{functional} functional / {total} total sequences · "
            f"Functional threshold = {threshold}"
        ),
    ]
    assert quantities.requested_keys == (
        f"{graph}_graph.peak_count",
        f"{graph}_graph.vertex_count",
        f"{graph}_graph.functional_count",
        f"thresholds.{graph}_functional.value",
    )
    with_graph = landscape_population_subtitle(quantities, graph, include_graph=True)
    assert with_graph[0] == f"{heading} graph · {subtitle[0]}"
    assert with_graph[1:] == subtitle[1:]


def test_functional_threshold_layers_draw_a_dashed_rule_and_its_label():
    scale = alt.Scale(domain=[-1.0, 1.0], nice=False, zero=False)
    rule, label = (
        layer.to_dict()
        for layer in functional_threshold_layers(
            -0.25,
            scale=scale,
            color="#555555",
            dash=[2.0, 3.0],
            width=1.2,
            font_size=14,
            label_x=292,
        )
    )
    assert rule["mark"] == {
        "type": "rule",
        "color": "#555555",
        "strokeDash": [2.0, 3.0],
        "strokeWidth": 1.2,
    }
    assert rule["encoding"]["y"]["field"] == "functional_threshold"
    assert rule["encoding"]["y"]["scale"]["domain"] == [-1.0, 1.0]
    assert rule["encoding"]["y"]["axis"] is None
    # The label is right-aligned above the rule, in the rule's colour.
    assert label["mark"] == {
        "type": "text",
        "text": "Functional threshold",
        "align": "right",
        "baseline": "bottom",
        "dy": -3,
        "fontSize": 14,
        "color": "#555555",
    }
    assert label["encoding"]["x"] == {"value": 292}
    assert label["encoding"]["y"] == rule["encoding"]["y"]


def test_functional_threshold_label_can_sit_below_the_rule_and_leave_the_axis():
    scale = alt.Scale(domain=[-1.0, 1.0])
    rule, label = (
        layer.to_dict()
        for layer in functional_threshold_layers(
            0.5,
            scale=scale,
            axis=alt.Undefined,
            color="#555555",
            dash=[2.0, 3.0],
            width=1.2,
            font_size=14,
            label_x=8,
            label_align="left",
            label_below=True,
        )
    )
    assert "axis" not in rule["encoding"]["y"]
    assert label["mark"]["align"] == "left"
    assert label["mark"]["baseline"] == "top"
    assert label["mark"]["dy"] == 3


def test_functional_threshold_label_offset_sets_the_gap_to_the_rule():
    for below, baseline, dy in ((False, "bottom", -8), (True, "top", 8)):
        _, label = (
            layer.to_dict()
            for layer in functional_threshold_layers(
                0.5,
                scale=alt.Scale(domain=[-1.0, 1.0]),
                color="#555555",
                dash=[2.0, 3.0],
                width=1.2,
                font_size=14,
                label_x=292,
                label_below=below,
                label_offset=8,
            )
        )
        assert (label["mark"]["baseline"], label["mark"]["dy"]) == (baseline, dy)


def test_threshold_label_padding_reads_the_figure_table_with_one_fallback():
    assert threshold_label_padding({}) == THRESHOLD_LABEL_PADDING
    assert threshold_label_padding({"threshold_label_padding": 12}) == 12


def test_functional_threshold_label_backing_sits_between_rule_and_label():
    layers = functional_threshold_layers(
        0.5,
        scale=alt.Scale(domain=[-1.0, 1.0]),
        color="#555555",
        dash=[2.0, 3.0],
        width=1.2,
        font_size=14,
        label_x=8,
        label_align="left",
        label_backing="white",
    )
    rule, backing, label = (layer.to_dict() for layer in layers)
    assert rule["mark"]["type"] == "rule"
    assert backing["mark"]["text"] == label["mark"]["text"]
    assert backing["mark"]["stroke"] == "white"
    assert backing["mark"]["strokeWidth"] == 4
    assert backing["mark"]["color"] == "white"
    assert backing["encoding"]["x"] == label["encoding"]["x"] == {"value": 8}
    assert "stroke" not in label["mark"]


def test_persistence_legend_is_inside_the_diagram_bottom_right_one_entry_per_line():
    from dubanevics2026topological.figures import palette
    from dubanevics2026topological.figures._charts import persistence_legend_layers

    layers = persistence_legend_layers(
        250, 250, {"legend_font_size": 15}, wild_type=True
    )
    marks = [layer.to_dict()["mark"] for layer in layers]
    labels = [mark for mark in marks if mark["text"] not in {"▲", "▽", "★"}]
    # Peak, then stop codon, then wild type: one entry per line, never a valley.
    assert [mark["text"] for mark in labels] == ["peak", "Stop codon containing peaks", "wild type"]
    ys = [mark["y"] for mark in labels]
    assert ys == [250 - 10 - 19 * step for step in (2, 1, 0)]
    # One left-aligned block whose right edge sits 10 px inside the right edge.
    assert len({mark["x"] for mark in labels}) == 1
    widest = max(len(mark["text"]) for mark in labels)
    assert labels[0]["x"] + 0.55 * 15 * widest <= 250 - 10 + 1
    glyphs = [mark for mark in marks if mark["text"] in {"▲", "▽", "★"}]
    assert {mark["x"] for mark in glyphs} == {labels[0]["x"] - 14}
    assert [mark["color"] for mark in glyphs][1] == palette.STOP_CODON_CRIMSON


def test_persistence_legend_leaves_out_the_entries_a_diagram_does_not_draw():
    from dubanevics2026topological.figures._charts import persistence_legend_layers

    def labels(**entries):
        layers = persistence_legend_layers(250, 250, {}, **entries)
        return [
            layer.to_dict()["mark"]["text"]
            for layer in layers
            if layer.to_dict()["mark"]["text"] not in {"▲", "▼", "★"}
        ]

    assert labels(stop_codon=False) == ["peak"]
    assert labels(peak=False, stop_codon=False, synonymous=True) == [
        "synonymous leaves"
    ]
    assert labels(peak=False, stop_codon=False) == []
    # The valley entry is opt-in (the full-graph figures only), never a default.
    assert "valley" not in labels(wild_type=True, synonymous=True)
    assert labels(valley=True) == ["peak", "valley", "Stop codon containing peaks"]


def test_stop_marker_size_is_one_configured_value_on_every_tree_figure():
    import tomllib
    from pathlib import Path

    from dubanevics2026topological.figures._charts import stop_tree_points_layer

    config = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )
    size = config["rendering"]["stop_codon_tree_size"]
    # An area in px^2, the size of the persistence triangles (70); set nowhere else.
    assert size == config["rendering"]["persistence_point_size"]

    def overrides(node, path=()):
        for key, value in node.items():
            if key == "stop_codon_tree_size" and path != ("rendering",):
                yield ".".join(path)
            if isinstance(value, dict):
                yield from overrides(value, (*path, key))

    assert not list(overrides(config))
    points = pl.DataFrame(
        {"x": [0.0], "y": [0.5], "dna_sequence": ["A"], "aa_sequence": ["K"]}
    )
    mark = stop_tree_points_layer(points, config["rendering"]).to_dict()["mark"]
    assert mark["size"] == size


def test_count_opacity_floor_keeps_count_one_visible_and_the_maximum_opaque():
    import tomllib
    from pathlib import Path

    from dubanevics2026topological.figures._branch_count_style import (
        branch_count_opacity,
    )

    rendering = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )["rendering"]
    assert rendering["count_alpha_min"] == pytest.approx(0.3)
    assert branch_count_opacity(1, rendering, (1.0, 15261.0)) == pytest.approx(0.3)
    assert branch_count_opacity(15261, rendering, (1.0, 15261.0)) == pytest.approx(1.0)
    assert branch_count_opacity(1, {}, (1.0, 100.0)) == pytest.approx(0.3)


def test_biology_and_full_graph_tree_stop_layers_share_the_configured_size():
    import tomllib
    from pathlib import Path

    from dubanevics2026topological.figures._charts import stop_tree_points_layer

    config = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )
    points = pl.DataFrame(
        {"x": [0.0], "y": [0.5], "dna_sequence": ["A"], "aa_sequence": ["K"]}
    )
    sizes = []
    for figure_id in ("study_tree_persistence_biology", "full_tree_persistence"):
        rendering = {**config["rendering"], **config["figures"].get(figure_id, {})}
        sizes.append(stop_tree_points_layer(points, rendering).to_dict()["mark"]["size"])
    assert sizes == [config["rendering"]["stop_codon_tree_size"]] * 2


def test_persistence_legend_inset_y_lifts_the_block_without_moving_its_right_edge():
    from dubanevics2026topological.figures._charts import persistence_legend_layers

    layers = persistence_legend_layers(250, 250, {"persistence_legend_inset_y": 30})
    labels = [
        layer.to_dict()["mark"]
        for layer in layers
        if layer.to_dict()["mark"]["text"] in {"peak", "Stop codon containing peaks"}
    ]
    assert [mark["y"] for mark in labels] == [250 - 30 - 19, 250 - 30]


def test_persistence_legend_inset_x_moves_the_block_left_and_defaults_to_the_inset():
    from dubanevics2026topological.figures._charts import persistence_legend_layers

    def glyph_x(rendering):
        layers = persistence_legend_layers(250, 250, rendering, stop_codon=False)
        return layers[0].to_dict()["mark"]["x"]

    base = glyph_x({"persistence_legend_inset": 10})
    assert glyph_x({"persistence_legend_inset": 10, "persistence_legend_inset_x": 10}) == base
    assert glyph_x({"persistence_legend_inset": 10, "persistence_legend_inset_x": 45}) == base - 35


def _stop_nodes():
    return pl.DataFrame(
        {
            "dna_sequence": ["PEAK", "VALLEY", "SADDLE", "REGULAR", "PLAIN"],
            "aa_sequence": ["*", "*", "*", "*", "K"],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.9, -0.5, 0.2, 0.1, 0.7],
            "display_event": ["maximum", "minimum", "split_saddle", "regular", "maximum"],
            "has_stop_codon": [True, True, True, True, False],
        }
    )


def test_stop_codon_nodes_default_keeps_peaks_and_valleys_only():
    from dubanevics2026topological.figures._charts import stop_codon_nodes

    kept = stop_codon_nodes(_stop_nodes(), {})
    assert kept["dna_sequence"].to_list() == ["PEAK", "VALLEY"]
    configured = {"stop_codon_tree_node_kinds": ["maximum", "minimum"]}
    assert stop_codon_nodes(_stop_nodes(), configured).equals(kept)


def test_stop_codon_nodes_follow_the_configured_kinds():
    from dubanevics2026topological.figures._charts import stop_codon_nodes

    peaks = stop_codon_nodes(_stop_nodes(), {"stop_codon_tree_node_kinds": ["maximum"]})
    assert peaks["dna_sequence"].to_list() == ["PEAK"]
    every = ["maximum", "minimum", "saddle", "regular"]
    allowed = stop_codon_nodes(_stop_nodes(), {"stop_codon_tree_node_kinds": every})
    assert allowed["dna_sequence"].to_list() == ["PEAK", "VALLEY", "SADDLE", "REGULAR"]
    critical = ["maximum", "minimum", "saddle"]
    assert stop_codon_nodes(
        _stop_nodes(), {"stop_codon_tree_node_kinds": critical}
    )["dna_sequence"].to_list() == ["PEAK", "VALLEY", "SADDLE"]


def test_stop_codon_nodes_refuse_an_unknown_kind():
    from dubanevics2026topological.figures._charts import stop_codon_nodes

    for bad in (["peak"], ["maximum", "valley"], "maximum"):
        with pytest.raises(ValueError, match="stop_codon_tree_node_kinds"):
            stop_codon_nodes(_stop_nodes(), {"stop_codon_tree_node_kinds": bad})


def test_stop_codon_nodes_read_event_and_the_root_of_a_critical_point_table():
    from dubanevics2026topological.figures._charts import stop_codon_nodes

    critical = pl.DataFrame(
        {
            "dna_sequence": ["PEAK", "SADDLE", "ROOT"],
            "event": ["maximum", "split_saddle", "split_saddle"],
            "is_root": [False, False, True],
            "has_stop_codon": [True, True, True],
            "x": [0.0, 1.0, 2.0],
            "y": [0.9, 0.2, -0.5],
        }
    )
    kept = stop_codon_nodes(critical, {})
    assert kept["dna_sequence"].to_list() == ["PEAK", "ROOT"]
    with pytest.raises(ValueError, match="display_event or event"):
        stop_codon_nodes(critical.drop("event"), {})


def test_stop_marker_layer_draws_no_cross_on_a_saddle_by_default():
    from dubanevics2026topological.figures._charts import stop_marker_layer

    tree = SimpleNamespace(all_stop_marker_table=_stop_nodes())
    spec = stop_marker_layer(tree, {}).to_dict()
    rows = next(iter(spec["datasets"].values()))
    assert [row["dna_sequence"] for row in rows] == ["PEAK", "VALLEY"]
    only_saddles = SimpleNamespace(
        all_stop_marker_table=_stop_nodes().filter(pl.col("dna_sequence") == "SADDLE")
    )
    spec = stop_marker_layer(only_saddles, {}).to_dict()
    assert not any(next(iter(spec["datasets"].values()), []))


def _full_graph_tree():
    # Topographer's full-graph stop table: event but no display_event / is_root.
    stops = pl.DataFrame(
        {
            "tree_node_index": [0, 1, None],
            "dna_sequence": ["ROOT", "SADDLE", "MEMBER"],
            "aa_sequence": ["*", "*", "*"],
            "event": ["split_saddle", "split_saddle", "regular"],
            "x": [0.0, 1.0, 2.0],
            "y": [-0.5, 0.2, 0.1],
        }
    )
    nodes = pl.DataFrame(
        {
            "tree_node_index": [0, 1, 2],
            "event": ["split_saddle", "split_saddle", "maximum"],
            "display_event": ["minimum", "split_saddle", "maximum"],
        }
    )
    return SimpleNamespace(
        all_stop_marker_table=stops, plot_data=SimpleNamespace(node_table=nodes)
    )


def test_stop_marker_layer_reads_the_root_kind_from_the_plot_data_node_table():
    from dubanevics2026topological.figures._charts import stop_marker_layer

    def drawn(rendering):
        spec = stop_marker_layer(_full_graph_tree(), rendering).to_dict()
        rows = next(iter(spec["datasets"].values()))
        return [row["dna_sequence"] for row in rows]

    assert drawn({}) == ["ROOT"]
    assert drawn({"stop_codon_tree_node_kinds": ["maximum"]}) == []


def test_stop_codon_nodes_refuse_a_table_without_has_stop_codon_unless_stop_only():
    from dubanevics2026topological.figures._charts import stop_codon_nodes

    table = _stop_nodes().drop("has_stop_codon")
    with pytest.raises(ValueError, match="has_stop_codon"):
        stop_codon_nodes(table, {})
    kept = stop_codon_nodes(table, {}, stop_only=True)
    assert kept["dna_sequence"].to_list() == ["PEAK", "VALLEY", "PLAIN"]


def test_the_shipped_default_allows_peaks_and_valleys_only():
    import tomllib
    from pathlib import Path

    config = tomllib.loads(
        (Path(__file__).parents[1] / "config" / "figures.toml").read_text()
    )
    assert config["rendering"]["stop_codon_tree_node_kinds"] == ["maximum", "minimum"]
