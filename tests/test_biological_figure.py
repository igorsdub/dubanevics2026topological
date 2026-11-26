"""Figure composition keeps biology markers above neutral study structure."""

from types import SimpleNamespace
from xml.etree import ElementTree

import altair as alt
import polars as pl
import pytest
import vl_convert as vlc
from _figure_labels import SENTINEL, label_quantities

from dubanevics2026topological.figures import (
    study_tree_persistence_biology as biology_module,
)
from dubanevics2026topological.figures.registry import get_spec, render_payload


def test_biology_figure_uses_maximum_only_overlays_and_no_count_legend(monkeypatch):
    edges = pl.DataFrame(
        {
            "source_label": ["saddle-stop", "saddle-syn", "root", "root"],
            "target_label": ["peak-stop", "peak-syn", "internal", "root-only"],
            "x_source": [0.0, 1.0, 2.0, 3.0],
            "x_target": [0.2, 1.2, 2.2, 3.2],
            "y": [1.0, 0.9, 0.5, 0.4],
            "y_source": [1.0, 0.9, 0.5, 0.4],
            "y_target": [0.7, 0.6, 0.0, 0.0],
            "branch_vertex_count": [3, 2, 4, 1],
            "vertex_count": [3, 2, 4, 1],
        }
    )
    segments = pl.DataFrame(
        {
            "target_label": [
                "peak-stop",
                "peak-syn",
                "peak-syn",
                "internal",
                "root-only",
            ],
            "source_label": ["saddle-stop", "saddle-syn", "saddle-syn", "root", "root"],
            "logical_edge_index": [0, 1, 1, 2, 3],
            "x_start": [0.0, 1.0, 1.0, 2.0, 3.0],
            "x_end": [0.0, 1.0, 1.2, 2.0, 3.0],
            "y_start": [0.7, 0.9, 0.6, 0.0, 0.0],
            "y_end": [1.0, 0.6, 0.6, 0.5, 0.4],
        }
    )
    pairs = pl.DataFrame(
        {
            "birth_sequence": ["peak-stop", "peak-syn", "root-only"],
            "death_sequence": ["saddle-stop", "root", "root"],
            "adjacent_saddle_sequence": ["saddle-stop", "saddle-syn", None],
            "birth_scalar": [1.0, 0.9, 0.4],
            "death_scalar": [0.7, 0.0, 0.0],
            "persistence": [0.3, 0.9, 0.4],
            "branch_vertex_count": [3, 2, 1],
            "birth_has_stop_codon": [True, False, False],
            "death_has_stop_codon": [True, True, True],
            "synonymous_endpoints": [False, True, False],
            "paired_synonymous_endpoints": [False, False, True],
        }
    )
    evidence = SimpleNamespace(
        branches=pl.DataFrame(
            {
                "upper_sequence": ["peak-syn", "internal", "root-only"],
                "lower_sequence": ["saddle-syn", "root", "root"],
                "is_peak_branch": [True, False, True],
                "synonymous_endpoints": [True, True, True],
            }
        ),
        pairs=pairs,
        critical_points=pl.DataFrame(
            {
                "dna_sequence": ["peak-stop", "peak-syn", "internal", "root"],
                "aa_sequence": ["*", "A", "%", "A"],
                "has_stop_codon": [True, False, True, True],
                "is_peak": [True, True, False, False],
                "event": ["maximum", "maximum", "split_saddle", "split_saddle"],
                "is_root": [False, False, False, True],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [1.0, 0.9, 0.5, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        biology_module,
        "plot_data_with_saved_metrics",
        lambda *_: SimpleNamespace(
            edge_table=edges,
            edge_segment_table=segments,
            node_table=pl.DataFrame(
                {
                    "dna_sequence": [
                        "peak-stop",
                        "peak-syn",
                        "internal",
                        "root-only",
                        "saddle-stop",
                        "saddle-syn",
                        "root",
                    ]
                }
            ),
        ),
    )
    monkeypatch.setattr(
        biology_module,
        "study_biology_tree_chart",
        lambda *_args, **_kwargs: (
            alt.Chart(edges)
            .mark_line(color="#b3b3b3")
            .encode(x=alt.X("x_source:Q", axis=None), y="y:Q")
        ),
    )
    persistence_heights = []

    def persistence_chart(data, _analysis, figure_settings, _rendering):
        persistence_heights.append(figure_settings["persistence"]["height"])
        return (
            alt.Chart(data)
            .mark_point(color="#737373")
            .encode(x="death_scalar:Q", y="birth_scalar:Q")
        )

    monkeypatch.setattr(
        biology_module, "study_biology_persistence_chart", persistence_chart
    )
    wild_type = pl.DataFrame(
        {
            "dna_sequence": ["wild-type"],
            "aa_sequence": ["K"],
            "marker_label": ["WT"],
            "x": [0.1],
            "y": [0.0],
        }
    )
    study = SimpleNamespace(
        trees={"split": SimpleNamespace(wild_type_marker_table=wild_type)},
        scalar_domain=(0, 1),
    )
    payload = {
        "landscapes": {"study_clamped": study},
        "tables": {"study_clamped_branch_table": pl.DataFrame()},
    }
    settings = {
        "figures": {
            "rendering": {},
            "figures": {
                "study_tree_persistence_biology": {
                    "font_size": 19,
                    "stop_codon_tree_size": 21,
                    "stop_codon_color": "#a50f15",
                    "tree": {"height": 210},
                    "persistence": {"point_size": 113, "marker_size": 150, "marker_opacity_floor": 0.6, "height": 100},
                }
            },
        }
    }
    quantities = label_quantities()
    spec = (
        get_spec("study_tree_persistence_biology")
        .render(
            {
                "comparison": payload,
                "annotations": {"study_clamped": evidence},
                "manuscript_quantities": quantities,
            },
            settings,
            "annotated",
        )
        .to_dict()
    )

    assert len(spec["vconcat"]) == 2
    assert all(len(row["hconcat"]) == 2 for row in spec["vconcat"])
    assert all(
        not (isinstance(node, dict) and "legend" in node.get("encoding", {}))
        for node in _walk(spec)
    )
    top_tree, top_persistence = spec["vconcat"][0]["hconcat"]
    bottom_tree, bottom_persistence = spec["vconcat"][1]["hconcat"]
    # Row titles and subtitles are the registry's display text, not the saved
    # evidence's own counts (one stop-codon peak, one synonymous leaf).
    stop_count = SENTINEL["annotations.published.stop_codon_peak_count"]
    synonymous_count = SENTINEL["annotations.published.synonymous_peak_count"]
    assert (
        spec["vconcat"][0]["title"]["text"]
        == f"{stop_count} stop-codon containing peaks"
    )
    assert (
        spec["vconcat"][1]["title"]["text"] == f"{synonymous_count} synonymous leaves"
    )
    for row in spec["vconcat"]:
        assert row["title"]["subtitle"] == [
            (
                "Papkou graph · Papkou et al. (2023) · "
                f"{SENTINEL['published_graph.peak_count']} peaks"
            ),
            (
                f"{SENTINEL['published_graph.functional_count']} functional / "
                f"{SENTINEL['published_graph.vertex_count']} total sequences · "
                "Functional threshold = "
                f"{SENTINEL['thresholds.published_functional.value']}"
            ),
        ]
    assert set(quantities.requested_keys) == {
        "annotations.published.stop_codon_peak_count",
        "annotations.published.synonymous_peak_count",
        "published_graph.peak_count",
        "published_graph.vertex_count",
        "published_graph.functional_count",
        "thresholds.published_functional.value",
    }
    assert all(row["resolve"]["scale"]["y"] == "shared" for row in spec["vconcat"])
    assert persistence_heights == [210, 210]

    def axis_for(panel, field, channel):
        return next(
            node["encoding"][channel].get("axis", "missing")
            for node in _walk(panel)
            if channel in node.get("encoding", {})
            and node["encoding"][channel].get("field") == field
        )

    for tree_panel in (top_tree, bottom_tree):
        assert axis_for(tree_panel, "y", "y")["title"] == "Fitness (at birth, WT = 0)"
    assert (
        axis_for(bottom_tree, "y_start", "y")["title"] == "Fitness (at birth, WT = 0)"
    )
    for persistence_panel in (top_persistence, bottom_persistence):
        assert axis_for(persistence_panel, "birth_scalar", "y") is None
    assert axis_for(top_persistence, "death_scalar", "x") is None
    assert (
        axis_for(bottom_persistence, "death_scalar", "x")["title"] == "Fitness at death"
    )

    compiled = vlc.vegalite_to_vega(spec)
    assert compiled["$schema"].endswith("vega/v6.json")
    svg = ElementTree.fromstring(vlc.vegalite_to_svg(spec))
    axes = [
        node
        for node in svg.iter()
        if node.tag.endswith("}g") and "role-axis" in node.attrib.get("class", "")
    ]
    axis_labels = [node.attrib.get("aria-label", "") for node in axes]
    tree_y_axes = [
        node
        for node in axes
        if "Y-axis titled 'Fitness (at birth, WT = 0)'"
        in node.attrib.get("aria-label", "")
    ]
    assert len(tree_y_axes) == 2
    assert all(
        any(
            child.tag.endswith("}g")
            and "role-axis-label" in child.attrib.get("class", "")
            for child in axis.iter()
        )
        for axis in tree_y_axes
    )
    assert all(
        any(
            child.tag.endswith("}g")
            and "role-axis-tick" in child.attrib.get("class", "")
            for child in axis.iter()
        )
        for axis in tree_y_axes
    )
    assert sum("Y-axis titled 'Birth fitness'" in label for label in axis_labels) == 0
    assert (
        sum("X-axis titled 'Fitness at death'" in label for label in axis_labels) == 1
    )

    stop_layer = top_tree["layer"][1]
    stop_data = spec["datasets"][stop_layer["data"]["name"]]
    # The saddle with a stop codon ("internal") carries no cross; the root (the global
    # minimum, a valley) does, with the default node kinds.
    assert [row["dna_sequence"] for row in stop_data] == ["peak-stop", "root"]
    stop_mark = stop_layer["mark"]
    assert stop_mark["color"] == "#a50f15"
    assert stop_mark["opacity"] == 0.7
    assert stop_mark["size"] == 21
    assert stop_mark["shape"] == "M-1,-1L1,1M-1,1L1,-1"

    branch_layer = bottom_tree["layer"][1]
    highlighted = spec["datasets"][branch_layer["data"]["name"]]
    assert {row["target_label"] for row in highlighted} == {"peak-syn"}
    assert len(highlighted) == 2
    assert all(
        row["x_start"] == row["x_end"] or row["y_start"] == row["y_end"]
        for row in highlighted
    )
    assert branch_layer["mark"]["color"] == "#1b9e77"
    assert branch_layer["mark"]["opacity"] == 1.0
    wild_type_layer = bottom_tree["layer"][-2]
    assert wild_type_layer["mark"]["shape"].startswith("M")
    assert wild_type_layer["mark"]["color"] == "#d62728"

    # No legend row below either diagram: the diagrams are layered charts and the
    # key is drawn inside them.
    assert "vconcat" not in top_persistence and "vconcat" not in bottom_persistence

    def keyed_labels(panel):
        return [
            layer["mark"]["text"]
            for layer in panel["layer"]
            if layer.get("mark", {}).get("type") == "text"
            and layer["mark"]["text"] not in {"▲", "▽", "★"}
        ]

    # Stop row: peak, stop codon (never a valley). Synonymous row: synonymous leaves only.
    assert keyed_labels(top_persistence) == ["peak", "Stop codon containing peaks"]
    assert keyed_labels(bottom_persistence) == ["synonymous leaves"]
    key_glyphs = [
        (layer["mark"]["text"], layer["mark"]["color"])
        for layer in bottom_persistence["layer"]
        if layer.get("mark", {}).get("type") == "text"
        and layer["mark"]["text"] == "▲"
    ]
    assert key_glyphs == [("▲", "#1b9e77")]  # green

    def points(panel):
        return [
            layer
            for layer in panel["layer"]
            if layer.get("mark", {}).get("type") == "point"
        ]

    stop_layer = points(top_persistence)[-1]
    synonym_layer = points(bottom_persistence)[-1]
    stop_points = spec["datasets"][stop_layer["data"]["name"]]
    synonym_points = spec["datasets"][synonym_layer["data"]["name"]]
    assert [row["birth_label"] for row in stop_points] == ["peak-stop"]
    assert [row["birth_label"] for row in synonym_points] == ["peak-syn"]
    assert synonym_points[0]["death_label"] == "root"
    assert synonym_points[0]["death_scalar"] == 0.0
    # A stop-codon peak is a solid crimson triangle (no cross on a diagram), and
    # it replaces the grey triangle instead of sitting under it.
    stop_mark = stop_layer["mark"]
    assert stop_mark["shape"] == "triangle-up"
    assert stop_mark["color"] == "#a50f15"
    assert stop_points[0]["vertex_count"] == 3
    assert stop_mark["filled"] is True
    assert "M-1,-1L1,1M-1,1L1,-1" not in str(top_persistence)
    top_base = top_persistence["layer"][0]
    assert {
        row["birth_label"] for row in spec["datasets"][top_base["data"]["name"]]
    } == {"peak-syn", "root-only"}
    # The synonymous peak keeps its green triangle.
    assert synonym_layer["mark"]["shape"] == "triangle-up"
    assert synonym_layer["mark"]["color"] == "#1b9e77"
    assert synonym_points[0]["vertex_count"] == 2
    for layer in (stop_layer, synonym_layer):
        opacity = layer["encoding"]["opacity"]
        assert opacity["field"] == "count_opacity"
        assert opacity["scale"] is None
        assert opacity["legend"] is None
    # No count-driven opacity falls below the floor (the stop point has 3 vertices and
    # the synonymous point 2, both under the 0.3-to-1 ramp's 0.6 mark).
    for layer in (stop_layer, synonym_layer):
        data = spec["datasets"][layer["data"]["name"]]
        assert all(row["count_opacity"] >= 0.6 for row in data)
    # The stop and synonymous markers are larger than the plain points (113).
    assert stop_layer["mark"]["size"] == 150
    assert synonym_layer["mark"]["size"] == 150
    # Both trees carry a key under them: wild type on both rows, stop codon only on
    # the stop row.
    def tree_key(tree):
        return {
            item["mark"]["text"]
            for item in _walk(tree)
            if isinstance(item.get("mark"), dict)
            and item["mark"].get("text") in {"Wild type", "Stop codon containing peaks"}
        }

    assert tree_key(top_tree) == {"Wild type", "Stop codon containing peaks"}
    assert tree_key(bottom_tree) == {"Wild type"}


def test_direct_biology_vanilla_rejected_before_analysis_access():
    with pytest.raises(ValueError, match="canonical annotated"):
        render_payload(
            get_spec("study_tree_persistence_biology"), None, {}, "vanilla"
        )


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
