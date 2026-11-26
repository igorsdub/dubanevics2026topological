"""Linked split-tree figure renderer driven through the figure registry."""

from dataclasses import dataclass
from types import SimpleNamespace

import polars as pl
import pytest
from _figure_labels import SENTINEL, label_quantities

from dubanevics2026topological.figures.registry import get_spec, render_payload

SPLIT_FIGURE_IDS = ("linked_tree_persistence",)


def test_built_figure_variants_encode_one_full_tree_count_domain(monkeypatch):
    import altair as alt

    from dubanevics2026topological.figures import _charts as charts

    @dataclass
    class PlotData:
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        node_table: pl.DataFrame

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
    tree_analysis = SimpleNamespace(
        plot_data=PlotData(
            displayed,
            displayed,
            pl.DataFrame(
                {
                    "tree_node_index": [0, 1, 2, 3],
                    "dna_sequence": ["ROOT", "MAX1", "MERGER", "MAX2"],
                    "vertex_index": [0, 1, 2, 3],
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
        persistence_table=pl.DataFrame(
            {
                "birth_label": ["MAX1", "MAX2"],
                "death_label": ["MERGER", "ROOT"],
                "birth_scalar": [0.9, 0.8],
                "death_scalar": [0.2, 0.1],
                "pair_kind": ["ordinary", "extended"],
                "persistence": [0.7, 0.7],
                "has_stop_codon": [False, False],
                "synonymous_endpoints": [False, True],
            }
        ),
        all_stop_marker_table=pl.DataFrame(
            {
                "vertex_id": [2],
                "dna_sequence": ["MAX1"],
                "aa_sequence": ["K"],
                "x": [1.0],
                "y": [0.9],
                "display_event": ["maximum"],
            }
        ),
        wild_type_marker_table=pl.DataFrame(
            {
                "dna_sequence": ["WILDTYPE"],
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
        event_table=pl.DataFrame(
            {"event": ["maximum", "maximum", "split_saddle", "regular"]}
        ),
    )
    analysis = SimpleNamespace(
        trees={"split": tree_analysis},
        scalar_domain=(0.0, 1.0),
        scalar_label="Fitness",
    )
    settings = {
        "figures": {
            "formats": ["svg", "png"],
            "variants": ["vanilla", "annotated"],
            "rendering": {
                "branch_count_title": "Branch\\nvertex\\ncount",
                "legend_x": 300,
                "tree_legend_x": 300,
                "persistence_legend_x": 10,
                "persistence_point_size": 70,
                "persistence_point_shape": "triangle-up",
                "wild_type_shape": "star",
                "wild_type_color": "#d62728",
                "stop_codon_shape": "x",
                "stop_codon_color": "#4a4a4a",
                "stop_codon_size": 110,
            },
            "figures": {
                figure_id: {
                    "title": "Legacy title to suppress",
                    "colorbar_x_fraction": 0.9,
                    "count_legend_ticks": [2, 3],
                    "tree": {"width": 420, "height": 250},
                    "persistence": {"width": 250, "height": 250},
                }
                for figure_id in SPLIT_FIGURE_IDS
            },
        }
    }

    def fake_plot_tree(plot_data, *, config, presentation, **kwargs):
        domain = list(config.edge_color_scale.domain)
        base = (
            alt.Chart(pl.DataFrame({"scalar": [0.5], "vertex_count": [1]}))
            .mark_line()
            .encode(
                x=alt.X("scalar:Q"),
                y=alt.Y("scalar:Q"),
                color=alt.Color("vertex_count:Q", scale=alt.Scale(domain=domain)),
            )
        )
        marker_layers = [
            alt.Chart(annotation.marker_table)
            .mark_point(
                shape=annotation.marker_config.shape,
                color=annotation.marker_config.color,
                size=annotation.marker_config.size,
                filled=annotation.marker_config.filled,
            )
            .encode(x="x:Q", y="y:Q")
            for annotation in presentation.annotation_layers
        ]
        return alt.layer(base, *marker_layers)

    def fake_plot_persistence(table, *, config, **kwargs):
        domain = list(config.color_scale.domain)
        point = (
            alt.Chart(table)
            .mark_point(size=config.point_size)
            .encode(
                x=alt.X("death_scalar:Q"),
                y=alt.Y("birth_scalar:Q"),
                color=alt.Color("vertex_count:Q", scale=alt.Scale(domain=domain)),
            )
        )
        return alt.layer(point)

    monkeypatch.setattr(charts, "plot_tree", fake_plot_tree)
    monkeypatch.setattr(charts, "plot_persistence", fake_plot_persistence)
    quantities = label_quantities()
    inputs = {"split_analysis": analysis, "manuscript_quantities": quantities}
    charts = [
        get_spec(figure_id).render(inputs, settings, "vanilla")
        for figure_id in SPLIT_FIGURE_IDS
    ]

    def vertex_count_domains(node):
        found = []
        if isinstance(node, dict):
            color = node.get("encoding", {}).get("color", {})
            if color.get("field") == "vertex_count":
                found.append(color.get("scale", {}).get("domain"))
            for value in node.values():
                found.extend(vertex_count_domains(value))
        elif isinstance(node, list):
            for value in node:
                found.extend(vertex_count_domains(value))
        return found

    def colored_point_shapes(node):
        found = []
        if isinstance(node, dict):
            mark = node.get("mark", {})
            color = node.get("encoding", {}).get("color", {})
            if mark.get("type") == "point" and color.get("field") == "vertex_count":
                found.append(mark.get("shape"))
            for value in node.values():
                found.extend(colored_point_shapes(value))
        elif isinstance(node, list):
            for value in node:
                found.extend(colored_point_shapes(value))
        return found

    for chart in charts:
        domains = vertex_count_domains(chart.to_dict())
        assert domains
        assert set(map(tuple, domains)) == {(1.0, 4.0)}
        assert chart.to_dict()["title"]["text"] == "Full graph"
    assert "triangle-up" in colored_point_shapes(charts[0].to_dict())

    linked = charts[0].to_dict()
    assert linked["title"]["text"] == "Full graph"
    # The title's counts are the registry's display text (the sentinels match nothing
    # this synthetic tree yields), asked for by key.
    assert linked["title"]["subtitle"] == [
        "Papkou et al. (2023) · full dataset",
        (
            f"{SENTINEL['full_graph.peak_count']} peaks · "
            f"{SENTINEL['full_graph.vertex_count']} sequences"
        ),
    ]
    assert set(quantities.requested_keys) == {
        "full_graph.peak_count",
        "full_graph.vertex_count",
    }

    def text_marks(node):
        found = []
        if isinstance(node, dict):
            mark = node.get("mark")
            if isinstance(mark, dict) and mark.get("type") == "text":
                found.append(mark.get("text"))
            for value in node.values():
                found.extend(text_marks(value))
        elif isinstance(node, list):
            for value in node:
                found.extend(text_marks(value))
        return found

    # The figure's own count_legend_ticks reach the colour bar, not only the
    # global rendering table (the default ticks here would be just 2).
    assert {"2", "3"} <= set(text_marks(linked))

    def gradient_rects(node):
        if isinstance(node, dict):
            mark = node.get("mark")
            here = (
                [mark]
                if isinstance(mark, dict)
                and mark.get("type") == "rect"
                and isinstance(mark.get("color"), dict)
                and mark["color"].get("gradient") == "linear"
                else []
            )
            return here + [
                rect for value in node.values() for rect in gradient_rects(value)
            ]
        if isinstance(node, list):
            return [rect for value in node for rect in gradient_rects(value)]
        return []

    assert len(gradient_rects(linked)) == 1

    def point_marks(node):
        found = []
        if isinstance(node, dict):
            mark = node.get("mark")
            if isinstance(mark, dict) and mark.get("type") == "point":
                found.append(mark)
            for value in node.values():
                found.extend(point_marks(value))
        elif isinstance(node, list):
            for value in node:
                found.extend(point_marks(value))
        return found

    # Vanilla: no stop-codon cross and no synonymous square; the only fixed-colour
    # marks are the red wild-type star on the tree. The key under the tree names
    # only what is drawn, so it holds the wild type and no stop-codon entry.
    colours = {mark.get("color") for mark in point_marks(linked) if "color" in mark}
    assert colours == {"#d62728"}
    assert not {"×", "Stop codon containing peaks", "Synonymous"} & set(text_marks(linked))
    assert {"★", "Wild type"} <= set(text_marks(linked))
    persistence_marks = point_marks(linked["hconcat"][1])
    assert {mark["shape"] for mark in persistence_marks} == {"triangle-up"}
    assert not [mark for mark in persistence_marks if "color" in mark]
    # The diagram key is inside the diagram, bottom right, peak entry only (no
    # valley entry): a layer over the diagram, never a row below it.
    diagram = linked["hconcat"][1]
    assert "vconcat" not in diagram
    entries = [
        layer["mark"]
        for layer in diagram["layer"]
        if layer.get("mark", {}).get("type") == "text"
        and layer["mark"].get("text") in {"peak", "valley"}
    ]
    assert [mark["text"] for mark in entries] == ["peak"]
    assert entries[0]["y"] <= 250
    assert all(mark["x"] > 125 for mark in entries)


def test_linked_figure_has_only_a_vanilla_variant():
    spec = get_spec("linked_tree_persistence")
    assert spec.variants == {"vanilla"}
    with pytest.raises(ValueError, match="canonical vanilla"):
        render_payload(spec, SimpleNamespace(), {"figures": {}}, "annotated")
