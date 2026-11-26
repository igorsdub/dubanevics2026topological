"""Focused rendering contract for the three-landscape tree comparison."""

from dataclasses import dataclass
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest
from _figure_labels import SENTINEL, label_quantities


def _analysis(maximum, title, domain=(0.0, 1.0)):
    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        node_table: pl.DataFrame

    edge = pl.DataFrame(
        {
            "source_tree_node_index": [0],
            "target_tree_node_index": [1],
            "source_scalar": [0.1],
            "target_scalar": [0.9],
            "x_source": [0.0],
            "x_target": [1.0],
            "y_source": [0.1],
            "y_target": [0.9],
        }
    )
    branches = pl.DataFrame(
        {
            "source_tree_node_index": [1],
            "target_tree_node_index": [0],
            "branch_vertex_count": [maximum],
        }
    )
    tree = SimpleNamespace(
        scalar_attr="raw_fitness",
        plot_data=PlotData(
            edge,
            edge,
            pl.DataFrame({"tree_node_index": [0, 1], "dna_sequence": ["ROOT", "WT"]}),
        ),
        branch_table=branches,
        critical_node_table=pl.DataFrame({"dna_sequence": ["WT"], "scalar": [0.5]}),
        branch_member_table=pl.DataFrame({"dna_sequence": ["ROOT"], "scalar": [0.1]}),
        event_table=pl.DataFrame({"event": ["maximum"]}),
        wild_type_marker_table=pl.DataFrame(
            {
                "dna_sequence": ["WT"],
                "aa_sequence": ["K"],
                "marker_label": ["WT"],
                "x": [0.5],
                "y": [0.5],
            }
        ),
        all_stop_marker_table=pl.DataFrame(
            {
                "dna_sequence": ["STOP"],
                "x": [0.2],
                "y": [0.2],
                "display_event": ["maximum"],
            }
        ),
    )
    return SimpleNamespace(
        trees={"split": tree},
        scalar_domain=domain,
        scalar_label=title,
        scalar_attr="raw_fitness",
    )


@pytest.mark.parametrize(
    "figure_id", ["tree_comparison", "full_tree_histogram", "published_tree_histogram", "candidate_tree_histogram"]
)
def test_context_figure_loader_renders_with_no_full_analysis(tmp_path, monkeypatch, figure_id):
    from contextlib import nullcontext
    from pathlib import Path
    import tomllib

    from dubanevics2026topological import build
    from dubanevics2026topological.analysis import clamped_cache
    from dubanevics2026topological.build import saved_figure_inputs
    from dubanevics2026topological.figures import _charts as charts
    from dubanevics2026topological.figures import _quantities
    from dubanevics2026topological.figures.registry import get_spec
    from dubanevics2026topological.results import RunContext

    analyses = {
        "full": _analysis(4, "Raw fitness"),
        "study_clamped": _analysis(20, "Raw fitness", (-0.7, 0.9)),
        "candidate_clamped": _analysis(100, "Raw fitness", (-0.4, 0.8)),
    }
    comparison = {
        "landscapes": {**analyses, "full": None},
        "thresholds": {"study_clamped": -0.507774, "candidate_clamped": -0.253159},
    }
    names = {"full": "full", "article": "study_clamped", "candidate": "candidate_clamped"}
    saved = {
        "mode": "saved",
        "layouts": {key: analyses[name].trees["split"] for key, name in names.items()},
        "populations": {
            key: pl.concat([
                analyses[name].trees["split"].critical_node_table,
                analyses[name].trees["split"].branch_member_table,
            ]).unique("dna_sequence", keep="first")
            for key, name in names.items()
        },
    }
    seen = []

    def load_checkpoint(*args, **kwargs):
        assert kwargs["include_full"] is False
        return comparison

    monkeypatch.setattr(clamped_cache, "checkpoint_lock", lambda *args: nullcontext())
    monkeypatch.setattr(clamped_cache, "_load_clamped_checkpoint", load_checkpoint)
    monkeypatch.setattr(build, "selection_source", lambda settings: {"rule": "median_reflected"})
    monkeypatch.setattr(saved_figure_inputs, "load_saved_figure_inputs", lambda *args, **kwargs: saved)
    monkeypatch.setattr(_quantities, "load_manuscript_quantities", lambda output: label_quantities())

    def plot_tree(plot_data, *, config, **kwargs):
        seen.append(config.scalar_domain)
        return alt.Chart(pl.DataFrame({"x": [0.0], "y": [0.0]})).mark_point().encode(x="x:Q", y="y:Q")

    monkeypatch.setattr(charts, "plot_tree", plot_tree)
    figures_path = Path(__file__).resolve().parents[1] / "config/figures.toml"
    settings = {"figures": tomllib.loads(figures_path.read_text(encoding="utf-8"))}
    assert "colorbar_x_fraction" in settings["figures"]["figures"][figure_id]
    spec = get_spec(figure_id)
    context = RunContext(root=tmp_path, env={})
    inputs = build.load_figure_inputs((spec,), settings, context=context)
    spec.render(inputs, settings, "annotated").to_dict()
    expected = {"tree_comparison": [(0.1, 0.5), (-0.7, 0.9), (-0.4, 0.8)],
                "full_tree_histogram": [(0.1, 0.5)],
                "published_tree_histogram": [(-0.7, 0.9)],
                "candidate_tree_histogram": [(-0.4, 0.8)]}
    assert seen == expected[figure_id]


def test_comparison_stacks_independent_wt_only_tree_panels(monkeypatch):
    from dubanevics2026topological.figures import _charts as charts
    from dubanevics2026topological.figures.registry import get_spec

    analyses = {
        "full": _analysis(4, "Raw fitness"),
        "study_clamped": _analysis(20, "Raw fitness", (-0.7, 0.9)),
        "candidate_clamped": _analysis(100, "Raw fitness", (-0.4, 0.8)),
    }
    # Unequal repeated-value populations must share the largest bin count.
    analyses["full"].trees["split"].critical_node_table = pl.DataFrame(
        {"dna_sequence": ["WT", "OTHER", "THIRD"], "scalar": [0.5, 0.5, 0.5]}
    )
    captured_domains = []
    captured_scalar_domains = []

    def fake_plot_tree(plot_data, *, config, **kwargs):
        captured_domains.append(config.edge_color_scale.domain)
        captured_scalar_domains.append(config.scalar_domain)
        return (
            alt.Chart(pl.DataFrame({"scalar": [0.5], "y": [0.5], "vertex_count": [2]}))
            .mark_line()
            .encode(
                x=alt.X("scalar:Q", scale=alt.Scale(domain=list(config.scalar_domain))),
                y="y:Q",
                color=alt.Color(
                    "vertex_count:Q",
                    scale=alt.Scale(domain=list(config.edge_color_scale.domain)),
                ),
            )
        )

    monkeypatch.setattr(charts, "plot_tree", fake_plot_tree)
    settings = {
        "figures": {
            "rendering": {"branch_count_title": "Branch\\nvertex\\ncount"},
            "figures": {
                "tree_comparison": {
                    "tree": {"width": 520, "height": 250},
                    "histogram_bins": 200,
                    "histogram_width": 140,
                    "colorbar_x_fraction": 1.0,
                },
                "full_tree_histogram": {"colorbar_x_fraction": 0.9},
                "published_tree_histogram": {"colorbar_x_fraction": 0.5},
                "candidate_tree_histogram": {"colorbar_x_fraction": 0.8},
            },
        }
    }
    quantities = label_quantities()
    chart = get_spec("tree_comparison").render(
        {
            "comparison": {
                "landscapes": analyses,
                "thresholds": {
                    "study_clamped": -0.507774,
                    "candidate_clamped": -0.253159,
                },
            },
            "manuscript_quantities": quantities,
            "saved_figure_inputs": {
                "mode": "saved",
                "layouts": {
                    key: analyses[name].trees["split"]
                    for key, name in (
                        ("full", "full"),
                        ("article", "study_clamped"),
                        ("candidate", "candidate_clamped"),
                    )
                },
                "populations": {
                    key: pl.concat(
                        [
                            analyses[name].trees["split"].critical_node_table,
                            analyses[name].trees["split"].branch_member_table,
                        ]
                    ).unique("dna_sequence", keep="first")
                    for key, name in (
                        ("full", "full"),
                        ("article", "study_clamped"),
                        ("candidate", "candidate_clamped"),
                    )
                },
            },
        },
        settings,
        "annotated",
    )
    spec = chart.to_dict()

    assert captured_domains == [(1.0, 100.0)] * 3
    assert captured_scalar_domains == [(0.0, 1.0), (-0.7, 0.9), (-0.4, 0.8)]
    rows = spec["vconcat"]
    assert len(rows) == 3
    assert all(row["resolve"]["scale"]["y"] == "shared" for row in rows)
    # No separate colour-bar axis: the bar is inside the top tree panel only.
    top_tree = rows[0]["hconcat"][0]
    assert "layer" in top_tree
    assert "Branch vertex count" in str(top_tree)
    assert "Branch vertex count" not in str(rows[1:])
    colorbar = [
        layer
        for layer in top_tree["layer"][-1]["layer"]  # the bar is the last layer
        if "mark" in layer
    ]
    legend_text = [
        layer["mark"]["text"] for layer in colorbar if layer["mark"]["type"] == "text"
    ]
    assert legend_text[0] == "Branch vertex count"
    # Ticks come from the union of the data: the largest count of any panel.
    assert "10,000" not in legend_text and "100" in legend_text
    gradients = [
        layer["mark"]
        for layer in colorbar
        if layer["mark"]["type"] == "rect"
        and isinstance(layer["mark"].get("color"), dict)
        and layer["mark"]["color"].get("gradient") == "linear"
    ]
    assert len(gradients) == 1
    # Top right: the title is centred in the right half of the 520 px tree panel.
    title = next(layer["mark"] for layer in colorbar if layer["mark"]["type"] == "text")
    assert 520 / 2 < title["x"] < 520
    assert title["y"] < 250 / 4

    scalar_domains = []

    def collect_scalar_domains(node):
        if isinstance(node, dict):
            for channel in ("x", "y"):
                encoding = node.get("encoding", {}).get(channel, {})
                if encoding.get("field") == "scalar":
                    scalar_domains.append(encoding.get("scale", {}).get("domain"))
            for value in node.values():
                collect_scalar_domains(value)
        elif isinstance(node, list):
            for value in node:
                collect_scalar_domains(value)

    collect_scalar_domains(spec)
    assert scalar_domains
    assert all(domain == pytest.approx([-1.2, 1.4]) for domain in scalar_domains)
    serialized = str(spec)
    assert "Full graph" in serialized
    assert "Papkou graph" in serialized
    assert "Topographer + FDR graph" in serialized
    assert "Papkou et al. (2023)" in serialized
    # Every number in the subtitles is the registry's display text: the sentinels
    # differ from anything these synthetic landscapes yield, so a recomputed count or
    # a three-decimal threshold would fail.
    assert "cutoff" not in serialized
    subtitles = [row["hconcat"][0]["title"]["subtitle"] for row in rows]
    full, published, candidate = (
        SENTINEL[f"{g}_graph.peak_count"] for g in ("full", "published", "candidate")
    )
    assert subtitles[0] == [
        f"Papkou et al. (2023) · {full} peaks",
        f"{SENTINEL['full_graph.vertex_count']} sequences",
    ]
    for subtitle, graph, peaks in (
        (subtitles[1], "published", published),
        (subtitles[2], "candidate", candidate),
    ):
        assert subtitle == [
            f"Papkou et al. (2023) · {peaks} peaks",
            (
                f"{SENTINEL[f'{graph}_graph.functional_count']} functional / "
                f"{SENTINEL[f'{graph}_graph.vertex_count']} total sequences · "
                "Functional threshold = "
                f"{SENTINEL[f'thresholds.{graph}_functional.value']}"
            ),
        ]
    assert set(quantities.requested_keys) == {
        "full_graph.peak_count",
        "full_graph.vertex_count",
        *(
            f"{graph}_graph.{suffix}"
            for graph in ("published", "candidate")
            for suffix in ("peak_count", "functional_count", "vertex_count")
        ),
        "thresholds.published_functional.value",
        "thresholds.candidate_functional.value",
    }
    assert "-0.253159" not in "\n".join(text for row in subtitles for text in row)
    assert "Vertex count" in serialized

    # The Published and Candidate rows draw only their own functional threshold as a
    # dashed rule with no text, across the tree and, the same rule, across the
    # histogram; both read the tree's fitness domain. The value is an extra
    # three-decimal tick on the fitness axis. The full-graph row has no threshold.
    full_row, published_row, candidate_row = rows
    assert _threshold_marks(full_row) == ([], [])
    assert _fitness_axes(full_row) and all(
        axis["values"] == pytest.approx([-1.2, 0.0, 1.4]) and "labelExpr" not in axis
        for axis in _fitness_axes(full_row)
    )
    assert "Functional threshold" not in _mark_texts(spec)
    for row, expected in ((published_row, -0.507774), (candidate_row, -0.253159)):
        tree_panel, histogram_panel = row["hconcat"]
        tree_rules, tree_labels = _threshold_marks(tree_panel)
        histogram_rules, histogram_labels = _threshold_marks(histogram_panel)
        assert len(tree_rules) == len(histogram_rules) == 1
        assert tree_labels == [] and histogram_labels == []
        for rule in (*tree_rules, *histogram_rules):
            assert spec["datasets"][rule["data"]["name"]][0][
                "functional_threshold"
            ] == pytest.approx(expected)
            assert rule["encoding"]["y"]["scale"]["domain"] == pytest.approx(
                [-1.2, 1.4]
            )
            assert rule["mark"]["strokeDash"] == [5.0, 4.0]
        # Exactly three regular ticks (the domain ends and 0, one decimal) plus the
        # panel's own threshold, written with three decimals.
        axes = _fitness_axes(row)
        assert axes
        for axis in axes:
            assert axis["values"] == pytest.approx(sorted([-1.2, 0.0, 1.4, expected]))
            assert axis["format"] == ".1f"
            assert f"({expected!r})" in axis["labelExpr"]
            assert "'.3f' : '.1f'" in axis["labelExpr"]
        # The histogram's rule is the tree's: same colour and width, cropped.
        (tree_rule,), (histogram_rule,) = tree_rules, histogram_rules
        for key in ("color", "strokeWidth", "strokeDash"):
            assert histogram_rule["mark"][key] == tree_rule["mark"][key]
        assert histogram_rule["mark"]["clip"] is True
        # The row's y scale is shared, so the two rules land at the same value.
        assert row["resolve"]["scale"]["y"] == "shared"
        # The rule is drawn behind the tree's branches; the wild-type star and its
        # label are last, and no text layer names the threshold.
        layers = tree_panel["layer"]
        kinds = [layer.get("mark", {}).get("type", "layer") for layer in layers]
        assert kinds[0] == "rule"
        assert not [
            layer
            for layer in layers
            if layer.get("mark", {}).get("type") == "text"
            and layer.get("encoding", {}).get("y", {}).get("field")
            == "functional_threshold"
        ]
        star = [
            index
            for index, layer in enumerate(layers)
            if layer.get("mark", {}).get("color") == "#d62728"
        ]
        assert star and max(star) == len(layers) - 1
        # Over the histogram it is the top layer, above the bars.
        kinds = [layer["mark"]["type"] for layer in histogram_panel["layer"]]
        assert kinds == ["bar", "rule"]
    histograms = []

    def collect_histograms(node):
        if isinstance(node, dict):
            encoding = node.get("encoding", {})
            if encoding.get("y", {}).get("field") == "bin_min":
                histograms.append(node)
            for value in node.values():
                collect_histograms(value)
        elif isinstance(node, list):
            for value in node:
                collect_histograms(value)

    collect_histograms(spec)
    assert len(histograms) == 3
    domains = [chart["encoding"]["x"]["scale"]["domain"] for chart in histograms]
    assert domains == [[0.0, 20_000.0]] * 3
    assert [chart["encoding"]["x"]["axis"]["title"] for chart in histograms] == [
        None,
        None,
        "Vertex count",
    ]
    assert "Clamped fitness" not in serialized
    assert "wild_type_marker_table" not in serialized
    assert "Stop codon" not in serialized
    assert "Synonymous" not in serialized
    assert "#d62728" in serialized
    assert "star" in serialized


def test_the_threshold_label_padding_is_read_from_the_figure_table():
    from dubanevics2026topological.figures.tree_comparison import _threshold_layers

    for rendering, expected in (({}, 8), ({"threshold_label_padding": 14}, 14)):
        layered = alt.layer(*_threshold_layers(-0.5, (-1.2, 1.4), rendering)).to_dict()
        (label,) = _threshold_marks(layered)[1]
        assert label["encoding"]["x"] == {"value": expected}
        assert label["mark"]["align"] == "left"


def _threshold_marks(node):
    """Return the functional-threshold rule layers and label layers under ``node``."""
    rules, labels = [], []

    def visit(item):
        if isinstance(item, dict):
            if item.get("encoding", {}).get("y", {}).get("field") == (
                "functional_threshold"
            ):
                kind = item["mark"]["type"]
                (rules if kind == "rule" else labels).append(item)
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(node)
    return rules, labels


def _fitness_axes(node):
    """Return the axis of every fitness (``y`` / ``bin_min``) encoding under ``node``."""
    axes = []

    def visit(item):
        if isinstance(item, dict):
            channel = item.get("encoding", {}).get("y", {})
            if channel.get("field") in ("y", "bin_min", "scalar") and isinstance(
                channel.get("axis"), dict
            ):
                axes.append(channel["axis"])
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(node)
    return axes


def _mark_texts(node):
    """Return the literal text of every text mark under ``node``."""
    texts = []

    def visit(item):
        if isinstance(item, dict):
            mark = item.get("mark")
            if isinstance(mark, dict) and "text" in mark:
                texts.append(mark["text"])
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(node)
    return texts


def _histogram():
    return (
        alt.Chart(pl.DataFrame({"bin_min": [0.0], "bin_max": [1.0], "count": [3]}))
        .mark_bar()
        .encode(
            y=alt.Y("bin_min:Q", scale=alt.Scale(domain=[-1.2, 1.4])),
            y2="bin_max:Q",
            x="count:Q",
        )
        .properties(width=140, height=250)
    )


def test_the_histogram_threshold_is_the_trees_rule_without_its_label():
    from dubanevics2026topological.figures.tree_comparison import (
        _threshold_layers,
        _with_histogram_threshold,
    )

    rendering = {
        "threshold_color": "#123456",
        "threshold_width": 2.5,
        "threshold_dash": [6, 2],
    }
    threshold = -0.4
    domain = (-1.2, 1.4)
    tree_rules, tree_labels = _threshold_marks(
        alt.layer(*_threshold_layers(threshold, domain, rendering)).to_dict()
    )
    (bare,) = _threshold_layers(threshold, domain, rendering, labelled=False)
    assert bare.mark.to_dict()["type"] == "rule"
    layered = _with_histogram_threshold(_histogram(), threshold, domain, rendering)
    rules, labels = _threshold_marks(layered.to_dict())
    assert labels == [] and len(tree_labels) == 1
    (rule,), (tree_rule,) = rules, tree_rules
    assert rule["mark"] == {**tree_rule["mark"], "clip": True}
    assert rule["mark"]["color"] == "#123456"
    assert rule["mark"]["strokeWidth"] == 2.5
    assert rule["mark"]["strokeDash"] == [6.0, 2.0]
    assert rule["encoding"]["y"] == tree_rule["encoding"]["y"]
    assert rule["encoding"]["y"]["scale"] == {
        "domain": [-1.2, 1.4],
        "nice": False,
        "zero": False,
    }
    # The rule has no x channel, so it spans the histogram's width and leaves the
    # count axis alone; the bars keep their size.
    assert "x" not in rule["encoding"]
    assert layered.to_dict()["layer"][0]["mark"]["type"] == "bar"
    assert layered.to_dict()["width"] == 140 and layered.to_dict()["height"] == 250


def test_the_shipped_threshold_style_is_dashed():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with (root / "config/figures.toml").open("rb") as stream:
        table = tomllib.load(stream)["figures"]["tree_comparison"]
    # Functional threshold = dashed, the global line-style rule (median = dotted).
    assert table["threshold_dash"] == [5, 4]
