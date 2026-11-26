"""Population and shared-axis contract for the Candidate histogram figure."""

from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest

from dubanevics2026topological.figures.registry import get_spec, render_payload


def _saved(analysis):
    names = {
        "full": "full",
        "study_clamped": "article",
        "candidate_clamped": "candidate",
    }
    layouts, populations = {}, {}
    for name, item in analysis["landscapes"].items():
        tree = item.trees["split"]
        key = names[name]
        layouts[key] = tree
        populations[key] = pl.concat(
            [tree.critical_node_table, tree.branch_member_table]
        ).unique("dna_sequence", keep="first")
    return {"mode": "saved", "layouts": layouts, "populations": populations}


def test_candidate_histogram_counts_unique_tree_population_and_shares_fitness_axis(
    monkeypatch,
):
    from dubanevics2026topological.figures import tree_histogram

    tree = SimpleNamespace(
        critical_node_table=pl.DataFrame(
            {"dna_sequence": ["A", "C"], "scalar": [0.0, 2.0]}
        ),
        branch_member_table=pl.DataFrame(
            {"dna_sequence": ["B", "C"], "scalar": [1.0, 2.0]}
        ),
        scalar_domain=(0.0, 2.0),
    )
    candidate = SimpleNamespace(trees={"split": tree}, scalar_domain=(0.0, 2.0))
    analysis = {
        "landscapes": {"candidate_clamped": candidate},
        "diagnostics": {"candidate_clamped": {"selected_component_vertex_count": 3}},
    }
    monkeypatch.setattr(
        tree_histogram,
        "plot_data_with_metrics",
        lambda _: SimpleNamespace(edge_table=pl.DataFrame({"vertex_count": [2, 7]})),
    )
    captured_count_domains = []

    def fake_tree_chart(*args, **kwargs):
        count_domain = args[5]
        captured_count_domains.append(count_domain)
        return (
            alt.Chart(pl.DataFrame({"scalar": [1.0], "vertex_count": [2]}))
            .mark_line()
            .encode(
                x="vertex_count:Q",
                y=alt.Y("scalar:Q"),
                color=alt.Color(
                    "vertex_count:Q", scale=alt.Scale(domain=list(count_domain))
                ),
            )
        )

    monkeypatch.setattr(tree_histogram, "tree_chart", fake_tree_chart)
    settings = {
        "figures": {
            "rendering": {"branch_count_title": "Branch\\nvertex\\ncount"},
            "figures": {
                "candidate_tree_histogram": {
                    "tree": {"width": 420, "height": 250},
                    "histogram_bins": 200,
                    "histogram_width": 140,
                    "legend_length": 55,
                    "colorbar_x_fraction": 0.8,
                }
            },
        }
    }

    spec = (
        get_spec("candidate_tree_histogram")
        .render(
            {"comparison": analysis, "saved_figure_inputs": _saved(analysis)},
            settings,
            "annotated",
        )
        .to_dict()
    )
    assert captured_count_domains == [(1.0, 7.0)]
    tree_spec, histogram_spec = spec["hconcat"]
    legend_spec = tree_spec["layer"][1]
    ramps = [
        layer["mark"]
        for layer in legend_spec["layer"]
        if layer.get("mark", {}).get("type") == "rect"
        and isinstance(layer["mark"].get("color"), dict)
        and layer["mark"]["color"].get("gradient") == "linear"
    ]
    assert len(ramps) == 1
    assert ramps[0]["width"] == pytest.approx(55)
    assert histogram_spec["encoding"]["x"]["axis"]["title"] == "Vertex count"
    assert histogram_spec["mark"]["binSpacing"] == 0
    histogram_dataset = next(
        rows
        for rows in spec["datasets"].values()
        if rows and {"bin_min", "bin_max", "count"}.issubset(rows[0])
    )
    assert len(histogram_dataset) == 200
    assert sum(row["count"] for row in histogram_dataset) == 3
    assert sum(row["count"] for row in histogram_dataset if row["count"]) == 3
    assert histogram_spec["encoding"]["y"]["scale"]["domain"] == [-1.2, 1.4]
    assert histogram_spec["encoding"]["y"]["axis"]["labels"] is False
    assert histogram_spec["encoding"]["y"]["axis"]["grid"] is False
    tree_domains = []

    def collect_tree_domain(node):
        if isinstance(node, dict):
            y_encoding = node.get("encoding", {}).get("y", {})
            if y_encoding.get("field") == "scalar":
                tree_domains.append(y_encoding.get("scale", {}).get("domain"))
            for value in node.values():
                collect_tree_domain(value)
        elif isinstance(node, list):
            for value in node:
                collect_tree_domain(value)

    collect_tree_domain(tree_spec["layer"][0])
    assert tree_domains == [[-1.2, 1.4]]
    assert spec["resolve"]["scale"]["y"] == "shared"
    # The paper figure (not only a slide) carries exactly three fitness ticks, the
    # domain's two ends and 0, written with one decimal; no threshold, so no extra.
    tree_axes = []

    def collect_tree_axes(node):
        if isinstance(node, dict):
            y_encoding = node.get("encoding", {}).get("y", {})
            if y_encoding.get("field") == "scalar" and isinstance(
                y_encoding.get("axis"), dict
            ):
                tree_axes.append(y_encoding["axis"])
            for value in node.values():
                collect_tree_axes(value)
        elif isinstance(node, list):
            for value in node:
                collect_tree_axes(value)

    collect_tree_axes(tree_spec["layer"][0])
    assert tree_axes
    for axis in tree_axes:
        assert axis["values"] == [-1.2, 0.0, 1.4]
        assert axis["format"] == ".1f"
        assert "labelExpr" not in axis


def test_candidate_histogram_rejects_missing_represented_vertices():
    tree = SimpleNamespace(
        critical_node_table=pl.DataFrame({"dna_sequence": ["A"], "scalar": [0.0]}),
        branch_member_table=pl.DataFrame({"dna_sequence": ["B"], "scalar": [1.0]}),
    )
    candidate = SimpleNamespace(trees={"split": tree}, scalar_domain=(0.0, 1.0))
    analysis = {
        "landscapes": {"candidate_clamped": candidate},
        "diagnostics": {"candidate_clamped": {"selected_component_vertex_count": 3}},
    }
    analysis["saved_layouts"] = _saved(analysis)["layouts"]
    analysis["saved_populations"] = _saved(analysis)["populations"]
    settings = {
        "figures": {
            "rendering": {},
            "figures": {
                "candidate_tree_histogram": {"colorbar_x_fraction": 0.8}
            },
        }
    }
    try:
        render_payload(
            get_spec("candidate_tree_histogram"),
            analysis,
            settings,
            "annotated",
        )
    except ValueError as error:
        assert "every selected graph vertex" in str(error)
    else:
        raise AssertionError("missing graph population identities were accepted")


def test_full_and_article_figures_keep_observed_bins_and_independent_count_limits(
    monkeypatch,
):
    from dubanevics2026topological.figures import tree_histogram

    values = {
        "full": [0.0, 1.0, 2.0, 3.0],
        "study_clamped": [0.0, 0.0, 1.0, 1.0],
    }
    landscapes = {}
    diagnostics = {}
    for key, scalars in values.items():
        table = pl.DataFrame(
            {"dna_sequence": [f"{key}-{i}" for i in range(4)], "scalar": scalars}
        )
        tree = SimpleNamespace(
            critical_node_table=table.head(2),
            branch_member_table=table.tail(2),
        )
        landscapes[key] = SimpleNamespace(trees={"split": tree})
        diagnostics[key] = {"selected_component_vertex_count": 4}

    monkeypatch.setattr(
        tree_histogram,
        "plot_data_with_metrics",
        lambda _: SimpleNamespace(edge_table=pl.DataFrame({"vertex_count": [2]})),
    )
    monkeypatch.setattr(
        tree_histogram,
        "tree_chart",
        lambda *args, **kwargs: (
            alt.Chart(pl.DataFrame({"scalar": [0.5], "vertex_count": [2]}))
            .mark_line()
            .encode(x="vertex_count:Q", y="scalar:Q")
        ),
    )
    analysis = {
        "landscapes": landscapes,
        "diagnostics": diagnostics,
    }

    def render(domain):
        settings = {
            "figures": {
                "fitness_domain": domain,
                "figures": {
                    "full_tree_histogram": {
                        "histogram_bins": 200,
                        "colorbar_x_fraction": 0.9,
                    },
                    "published_tree_histogram": {
                        "histogram_bins": 200,
                        "colorbar_x_fraction": 0.5,
                    },
                },
            }
        }
        result = {}
        for figure_id in (
            "full_tree_histogram",
            "published_tree_histogram",
        ):
            spec = (
                get_spec(figure_id)
                .render(
                    {"comparison": analysis, "saved_figure_inputs": _saved(analysis)},
                    settings,
                    "annotated",
                )
                .to_dict()
            )
            tree, histogram = spec["hconcat"]
            tree = tree["layer"][0]
            bins = next(
                rows
                for rows in spec["datasets"].values()
                if rows and {"bin_min", "bin_max", "count"}.issubset(rows[0])
            )
            result[figure_id] = (tree, histogram, bins)
        return result

    broad = render([-1.2, 3.2])
    narrow = render([0.25, 0.75])
    for figure_id in broad:
        tree, histogram, bins = broad[figure_id]
        cropped_tree, cropped_histogram, cropped_bins = narrow[figure_id]
        assert len(bins) == len(cropped_bins) == 200
        assert [(row["bin_min"], row["bin_max"], row["count"]) for row in bins] == [
            (row["bin_min"], row["bin_max"], row["count"]) for row in cropped_bins
        ]
        assert sum(row["count"] for row in bins) == 4
        assert histogram["encoding"]["y"]["scale"]["domain"] == [-1.2, 3.2]
        # exact_domain rounds the ends outward to one decimal so end ticks sit on labels
        assert cropped_histogram["encoding"]["y"]["scale"]["domain"] == [0.2, 0.8]
        assert cropped_histogram["mark"]["clip"] is True
        assert cropped_tree["mark"]["clip"] is True
    assert broad["full_tree_histogram"][1]["encoding"]["x"]["scale"][
        "domain"
    ] == [0.0, 20000.0]  # fixed count axis, not per-figure limits
    assert broad["published_tree_histogram"][1]["encoding"]["x"]["scale"][
        "domain"
    ] == [0.0, 20000.0]


def _walk_nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_nodes(value)


def _threshold_figures(monkeypatch):
    """Render the three standalone tree histograms with declared thresholds."""
    from dubanevics2026topological.figures import tree_histogram

    landscapes, diagnostics = {}, {}
    for key in ("full", "study_clamped", "candidate_clamped"):
        table = pl.DataFrame(
            {"dna_sequence": [f"{key}-{i}" for i in range(4)], "scalar": [0.0, 1.0, 0.5, 0.2]}
        )
        tree = SimpleNamespace(
            critical_node_table=table.head(2), branch_member_table=table.tail(2)
        )
        landscapes[key] = SimpleNamespace(trees={"split": tree}, scalar_domain=(0.0, 1.0))
        diagnostics[key] = {"selected_component_vertex_count": 4}
    monkeypatch.setattr(
        tree_histogram,
        "plot_data_with_metrics",
        lambda _: SimpleNamespace(edge_table=pl.DataFrame({"vertex_count": [2]})),
    )
    underlays = {}

    def fake_tree_chart(*args, **kwargs):
        underlays[len(underlays)] = list(kwargs.get("underlays", []))
        return (
            alt.Chart(pl.DataFrame({"y": [0.5], "vertex_count": [2]}))
            .mark_line()
            .encode(x="vertex_count:Q", y="y:Q")
        )

    monkeypatch.setattr(tree_histogram, "tree_chart", fake_tree_chart)
    analysis = {
        "landscapes": landscapes,
        "diagnostics": diagnostics,
        "thresholds": {"study_clamped": -0.5077739, "candidate_clamped": -0.253159},
    }
    settings = {
        "figures": {
            "rendering": {},
            "figures": {
                fid: {"histogram_bins": 50, "colorbar_x_fraction": 0.9}
                for fid in (
                    "full_tree_histogram",
                    "published_tree_histogram",
                    "candidate_tree_histogram",
                )
            },
        }
    }
    out = {}
    for fid in settings["figures"]["figures"]:
        before = len(underlays)
        spec = (
            get_spec(fid)
            .render(
                {"comparison": analysis, "saved_figure_inputs": _saved(analysis)},
                settings,
                "annotated",
            )
            .to_dict()
        )
        out[fid] = (spec, underlays[before])
    return out


@pytest.mark.parametrize(
    ("figure_id", "tick"),
    [("published_tree_histogram", -0.508), ("candidate_tree_histogram", -0.253)],
)
def test_standalone_tree_histograms_draw_their_own_dashed_unlabelled_threshold(
    monkeypatch, figure_id, tick
):
    spec, tree_underlays = _threshold_figures(monkeypatch)[figure_id]
    # The tree carries one unlabelled dashed rule behind its branches.
    assert len(tree_underlays) == 1
    rule = tree_underlays[0].to_dict()
    assert rule["mark"]["type"] == "rule"
    assert rule["mark"]["strokeDash"]
    assert not [n for n in _walk_nodes(rule) if n.get("type") == "text"]
    # The histogram continues the same dashed rule across its bars, no text.
    _, histogram = spec["hconcat"]
    assert "layer" in histogram
    rules = [
        layer
        for layer in histogram["layer"]
        if layer.get("mark", {}).get("type") == "rule"
    ]
    assert len(rules) == 1 and rules[0]["mark"]["strokeDash"]
    assert not [
        layer for layer in histogram["layer"] if layer.get("mark", {}).get("type") == "text"
    ]
    # Fitness ticks: [lower, T, 0, upper], T written with three decimals.
    axes = [
        node["encoding"]["y"]["axis"]
        for node in _walk_nodes(spec)
        if isinstance(node.get("encoding", {}).get("y", {}).get("axis"), dict)
        and "values" in node["encoding"]["y"]["axis"]
    ]
    assert axes
    for axis in axes:
        assert axis["values"] == pytest.approx([-1.2, tick, 0.0, 1.4], abs=1e-3)
        assert "'.3f' : '.1f'" in axis["labelExpr"]


def test_full_tree_histogram_has_no_threshold_rule(monkeypatch):
    spec, tree_underlays = _threshold_figures(monkeypatch)["full_tree_histogram"]
    assert tree_underlays == []
    assert not [n for n in _walk_nodes(spec) if n.get("mark", {}).get("type") == "rule"]
    for node in _walk_nodes(spec):
        axis = node.get("encoding", {}).get("y", {}).get("axis")
        if isinstance(axis, dict) and "values" in axis:
            assert axis["values"] == pytest.approx([-1.2, 0.0, 1.4])
            assert "labelExpr" not in axis
