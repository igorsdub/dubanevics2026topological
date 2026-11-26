from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest
from _figure_labels import SENTINEL, label_quantities

from dubanevics2026topological.figures import (
    study_tree_persistence_metrics as metrics_module,
)
from dubanevics2026topological.figures._charts import plot_data_with_saved_metrics
from dubanevics2026topological.figures.palette import HORTON_STRAHLER_COLORS
from dubanevics2026topological.figures.registry import get_spec
from dubanevics2026topological.figures.study_tree_persistence_metrics import (
    study_metric_persistence_table,
    study_metric_scale,
)


@dataclass
class FixturePlotData:
    node_table: pl.DataFrame
    edge_table: pl.DataFrame
    edge_segment_table: pl.DataFrame


def _saved_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    branches = pl.DataFrame(
        {
            "source_tree_node_identity": ["AAA", "AAG"],
            "target_tree_node_identity": ["AAG", "AAT"],
            "branch_vertex_count": [1, 3],
            "aa_hamming": [1, 1],
            "horton_strahler_order": [2, 1],
        }
    )
    pairs = pl.DataFrame(
        {
            "birth_label": ["AAA", "AAG"],
            "death_label": ["AAT", "AAA"],
            "birth_scalar": [0.9, 0.8],
            "death_scalar": [0.1, 0.2],
            "persistence": [0.8, 0.6],
            "aa_hamming": [0, 1],
            "horton_strahler_order": [4, 3],
        }
    )
    return branches, pairs


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_tree_and_pair_metrics_use_stable_endpoint_ownership() -> None:
    branches, pairs = _saved_tables()
    plot_data = FixturePlotData(
        node_table=pl.DataFrame(
            {
                "tree_node_index": [0, 1, 2],
                "dna_sequence": ["AAA", "AAG", "AAT"],
            }
        ),
        edge_table=pl.DataFrame(
            {
                "source_tree_node_index": [1, 2],
                "target_tree_node_index": [0, 1],
                "source_scalar": [0.8, 0.7],
                "target_scalar": [0.9, 0.8],
            }
        ),
        edge_segment_table=pl.DataFrame(),
    )
    tree_analysis = SimpleNamespace(plot_data=plot_data)

    enriched = plot_data_with_saved_metrics(tree_analysis, branches)
    assert enriched.edge_table["branch_vertex_count"].to_list() == [1, 3]
    assert enriched.edge_table["aa_hamming"].to_list() == [1, 1]
    assert enriched.edge_table["horton_strahler_order"].to_list() == [2, 1]

    counted = study_metric_persistence_table(
        pairs, branches, metric="branch_vertex_count"
    )
    assert counted["branch_vertex_count"].to_list() == [1, 3]
    assert study_metric_persistence_table(pairs, branches, metric="aa_hamming")[
        "aa_hamming"
    ].to_list() == [0, 1]


def test_pair_branch_count_requires_unique_birth_branch() -> None:
    branches, pairs = _saved_tables()
    duplicated = pl.concat([branches, branches.head(1)])
    with pytest.raises(ValueError, match="Duplicate birth endpoints"):
        study_metric_persistence_table(pairs, duplicated, metric="branch_vertex_count")


def test_metric_figure_composes_three_shared_scale_tree_pair_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branches, pairs = _saved_tables()
    plot_data = FixturePlotData(
        node_table=pl.DataFrame(
            {
                "tree_node_index": [0, 1, 2],
                "dna_sequence": ["AAA", "AAG", "AAT"],
            }
        ),
        edge_table=pl.DataFrame(
            {
                "source_tree_node_index": [1, 2],
                "target_tree_node_index": [0, 1],
                "source_scalar": [0.8, 0.7],
                "target_scalar": [0.9, 0.8],
            }
        ),
        edge_segment_table=pl.DataFrame(),
    )
    split = SimpleNamespace(
        plot_data=plot_data,
        critical_node_table=pl.DataFrame(
            {
                "dna_sequence": ["AAA", "AAG", "AAT"],
                "scalar": [0.9, 0.8, 0.1],
            }
        ),
        branch_member_table=pl.DataFrame(
            {"dna_sequence": ["INTERIOR"], "scalar": [0.3]}
        ),
        event_table=pl.DataFrame({"event": ["maximum", "maximum", "split_saddle"]}),
        wild_type_marker_table=pl.DataFrame(
            {
                "dna_sequence": ["AAA"],
                "aa_sequence": ["K"],
                "marker_label": ["WT"],
                "x": [0.0],
                "y": [0.0],
            }
        ),
    )
    study = SimpleNamespace(
        trees={"split": split},
        scalar_domain=(-1.17480288668069, 1.395791684395264),
        scalar_label="Raw fitness",
    )
    payload = {
        "landscapes": {"study_clamped": study},
        "thresholds": {"study_clamped": -0.507774},
        "tables": {
            "study_clamped_branch_table": branches,
            "study_clamped_persistence_table": pairs,
        },
    }
    settings = {
        "figures": {
            "rendering": {},
            "figures": {
                "study_tree_persistence_metrics": {
                    "font_size": 19,
                    "colorbar_x_fraction": 0.9,
                    "branch_count_title": "Branch vertex count",
                    "tree": {},
                    "persistence": {},
                }
            },
        }
    }
    seen: list[tuple[str, str, tuple[float, float] | None]] = []

    def fake_tree(data, *, config, **kwargs):
        scale = config.edge_color_scale
        seen.append(("tree", scale.attribute, scale.domain))
        return (
            alt.Chart(data.edge_table)
            .mark_line()
            .encode(
                x="source_scalar:Q",
                y=alt.Y("scalar:Q", title=config.scalar_axis_title),
                color=alt.Color(
                    f"{scale.attribute}:Q",
                    scale=alt.Scale(domain=scale.domain or [1, 4]),
                ),
            )
        )

    def fake_persistence(data, *, config, color_by, **kwargs):
        scale = config.color_scale
        seen.append(("persistence", color_by, scale.domain))
        return (
            alt.Chart(data)
            .mark_point()
            .encode(
                x=alt.X("death_scalar:Q", title=config.death_axis_title),
                y=alt.Y("birth_scalar:Q", title=config.birth_axis_title),
                color=alt.Color(
                    f"{color_by}:Q", scale=alt.Scale(domain=scale.domain or [1, 4])
                ),
            )
        )

    monkeypatch.setattr(metrics_module, "plot_tree", fake_tree)
    from dubanevics2026topological.figures import _charts

    monkeypatch.setattr(_charts, "plot_persistence", fake_persistence)
    monkeypatch.setattr(metrics_module, "plot_persistence", fake_persistence)
    quantities = label_quantities()
    chart = get_spec("study_tree_persistence_metrics").render(
        {"comparison": payload, "manuscript_quantities": quantities},
        settings,
        "vanilla",
    )
    spec = chart.to_dict()
    assert "vconcat" in spec
    assert len(spec["vconcat"]) == 3
    assert all(len(row["hconcat"]) == 2 for row in spec["vconcat"])
    assert all(row["resolve"]["scale"]["y"] == "shared" for row in spec["vconcat"])
    assert spec["title"]["text"] == "Published graph"
    # Registry provenance: the sentinels differ from the two peaks this synthetic
    # tree has, and no recomputed three-decimal threshold is printed.
    assert spec["title"]["subtitle"] == [
        f"Papkou et al. (2023) · {SENTINEL['published_graph.peak_count']} peaks",
        (
            f"{SENTINEL['published_graph.functional_count']} functional / "
            f"{SENTINEL['published_graph.vertex_count']} total sequences · "
            "Functional threshold = "
            f"{SENTINEL['thresholds.published_functional.value']}"
        ),
    ]
    assert set(quantities.requested_keys) == {
        "published_graph.peak_count",
        "published_graph.vertex_count",
        "published_graph.functional_count",
        "thresholds.published_functional.value",
    }
    assert [entry[1] for entry in seen] == [
        "branch_vertex_count",
        "vertex_count",
        "aa_hamming",
        "aa_hamming",
        "horton_strahler_order",
        "horton_strahler_order",
    ]
    for index, row in enumerate(spec["vconcat"]):
        tree_panel, persistence_panel = row["hconcat"]
        # Every row's colour bar is overlaid on its tree, none on the persistence panel.
        assert len(tree_panel["layer"]) == 2
        tree_y = tree_panel["layer"][0]["layer"][0]["encoding"]["y"]
        # Only the first diagram is keyed (a layer over the diagram); the others are
        # bare diagrams. No legend row sits below any of them.
        assert "vconcat" not in persistence_panel
        persistence_chart = persistence_panel["layer"][0] if index == 0 else persistence_panel
        persistence_base = persistence_chart["encoding"]
        diagram_text = {
            item["mark"].get("text")
            for item in _walk(persistence_panel)
            if isinstance(item.get("mark"), dict) and item["mark"].get("type") == "text"
        }
        tree_text = {
            item["mark"].get("text")
            for item in _walk(tree_panel["layer"][0])
            if isinstance(item.get("mark"), dict) and item["mark"].get("type") == "text"
        }
        if index == 0:
            # "peak" alone, inside the first diagram; one wild-type key under the tree.
            assert {"peak", "wild type"} & diagram_text == {"peak"}
            assert "valley" not in diagram_text and "stop codon" not in diagram_text
            assert "Wild type" in tree_text
        else:
            assert not diagram_text
            assert not {"Wild type", "Stop codon containing peaks"} & tree_text
            # The markers take the branch colours of their row.
            assert persistence_base["color"]["field"] == (
                "aa_hamming" if index == 1 else "horton_strahler_order"
            )
        titles = {
            item["mark"].get("text")
            for item in _walk(tree_panel["layer"][1])
            if isinstance(item.get("mark"), dict) and item["mark"].get("type") == "text"
        }
        assert titles >= {
            ["Branch vertex count", "AA Hamming distance", "Horton–Strahler order"][
                index
            ]
        }
        assert tree_y.get("title", tree_y.get("axis", {}).get("title")) == (
            "Fitness (at birth, WT = 0)"
        )
        assert persistence_base["y"]["axis"] is None
        assert (
            persistence_base["x"].get(
                "title", persistence_base["x"].get("axis", {}).get("title")
            )
            == "Fitness at death"
        )
        assert tree_y["scale"]["domain"] == persistence_base["y"]["scale"]["domain"]
    assert spec["title"]["text"] == "Published graph"
    assert spec["config"]["axis"]["labelFontSize"] == 19
    assert spec["config"]["legend"]["labelFontSize"] == 19


def test_study_metric_scales_keep_singleton_and_small_ranges_valid():
    count = study_metric_scale(
        {}, "count", "branch_vertex_count", "Count", pl.Series([1])
    )
    hamming = study_metric_scale(
        {}, "aa_hamming", "aa_hamming", "AA distance", pl.Series([0])
    )
    ordinal = study_metric_scale(
        {}, "horton_strahler_order", "horton_strahler_order", "H-S", pl.Series([1])
    )
    assert count.domain == (1.0, 1.001)
    assert count.one_value is None
    assert hamming.categories == (0,)  # observed range only
    assert hamming.domain is None
    assert hamming.scale_type == "ordinal"
    assert hamming.scheme == "bluepurple"
    assert len(hamming.colors) == 1
    assert ordinal.categories == (1, 2, 3, 4)
    assert ordinal.colors == HORTON_STRAHLER_COLORS


def test_hamming_colour_scale_is_discrete_with_width_one_bins_over_the_observed_range():
    scale = study_metric_scale(
        {}, "aa_hamming", "aa_hamming", "AA distance", pl.Series([1, 4, 2, 4])
    )
    assert scale.categories == (1, 2, 3, 4)
    assert scale.scale_type == "ordinal" and scale.domain is None
    assert len(scale.colors) == 4 and len(set(scale.colors)) == 4
    assert scale.scheme == "bluepurple"
    # Blue at the low distances, purple at the high ones: the red channel stays
    # high and the green one falls along the scheme.
    channels = [
        [int(color[i : i + 2], 16) for i in (1, 3, 5)] for color in scale.colors
    ]
    assert [c[1] for c in channels] == sorted((c[1] for c in channels), reverse=True)
    assert channels[0][2] > channels[0][0]  # the first bin is a blue
    assert channels[-1][0] > channels[-1][1]  # the last bin is a purple


@pytest.mark.parametrize("count", [1, 2, 4, 7])
def test_bluepurple_bins_are_the_colours_vega_draws_for_that_scheme(count):
    """The explicit colours match Altair's own ``bluepurple`` ordinal scale."""
    import re

    import vl_convert as vlc

    from dubanevics2026topological.figures.study_tree_persistence_metrics import (
        discrete_scheme_colors,
    )

    chart = (
        alt.Chart(alt.Data(values=[{"bin": i} for i in range(count)]))
        .mark_rect()
        .encode(
            x="bin:O",
            color=alt.Color("bin:O", scale=alt.Scale(scheme="bluepurple"), legend=None),
        )
        .properties(width=40 * count, height=20)
    )
    svg = vlc.vegalite_to_svg(chart.to_dict())
    vega = [
        tuple(int(value) for value in match.split(","))
        for match in re.findall(r'fill="rgb\(([^)]*)\)"', svg)
    ]
    ours = [
        tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
        for color in discrete_scheme_colors("bluepurple", count)
    ]
    assert len(vega) == len(ours) == count
    for vega_color, our_color in zip(vega, ours, strict=True):
        assert max(abs(a - b) for a, b in zip(vega_color, our_color)) <= 1


def test_matplotlib_scheme_names_still_sample_first_to_last():
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    from dubanevics2026topological.figures.study_tree_persistence_metrics import (
        discrete_scheme_colors,
    )

    colors = discrete_scheme_colors("cividis", 4)
    assert colors[0] == to_hex(colormaps["cividis"](0.0))
    assert colors[-1] == to_hex(colormaps["cividis"](1.0))


def test_study_metric_count_colour_bar_has_the_shared_three_ticks() -> None:
    """The 110 px count bar labels 1, 100 and 10,000 only (not 1,000 over 10,000)."""
    import tomllib
    from pathlib import Path

    from dubanevics2026topological.figures._charts import horizontal_count_legend

    config = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "config/figures.toml").read_text("utf-8")
    )
    figure = config["figures"]["study_tree_persistence_metrics"]
    assert figure["colorbar_length"] == 110
    assert figure["count_legend_ticks"] == [1, 100, 10000]
    rendering = {
        **config["rendering"],
        **figure,
        "legend_length": figure["colorbar_length"],
    }
    legend = horizontal_count_legend(rendering, (2.0, 20000.0))
    labels = {
        node["mark"]["text"]
        for node in legend.to_dict()["layer"]
        if node.get("mark", {}).get("type") == "text"
        and node["mark"]["text"] in {"1", "10", "100", "1,000", "10,000"}
    }
    assert labels == {"1", "100", "10,000"}
