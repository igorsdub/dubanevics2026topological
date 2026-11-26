"""The bootstrap tree-histogram twin and the band-smoothed codon-entropy figure."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import rustworkx as rx
from _figure_labels import with_labels
from _selection import fixed_mixture_selection
from test_codon_entropy import _entropy_frame
from test_codon_entropy_plot import _assignments, _ridge_tree
from topographer.plotting import make_tree_layout

from dubanevics2026topological import build
from dubanevics2026topological.analysis import codon_entropy as entropy_module
from dubanevics2026topological.figures import registry, tree_histogram
from dubanevics2026topological.figures.codon_entropy import (
    build_bootstrap_codon_entropy_figure,
)
from dubanevics2026topological.results.presentation import SavedTreeLayout

HISTOGRAM_ID = "bootstrap_tree_histogram"
ENTROPY_ID = "bootstrap_codon_entropy"
TWINS = (HISTOGRAM_ID, ENTROPY_ID)


def _tables():
    path = Path(build.ROOT) / "config/figures.toml"
    return tomllib.loads(path.read_text("utf-8"))["figures"]


# -- registry and configuration ------------------------------------------------


@pytest.mark.parametrize("figure_id", TWINS)
def test_twins_are_default_annotated_and_configured(figure_id):
    spec = registry.get_spec(figure_id)
    assert spec.default is True  # manuscript source
    assert spec.variants == {"annotated"}
    expected = {"comparison", "bootstrap_smoothed"}
    if figure_id == ENTROPY_ID:
        expected |= {"manuscript_quantities", "bootstrap_codon_entropy"}
    assert spec.inputs == expected
    assert spec.notebook is None
    assert spec.config_table == figure_id
    assert figure_id in _tables()
    assert figure_id in build.BOOTSTRAP_FIGURE_IDS
    assert registry.renderer_file(spec) in {"tree_histogram.py", "codon_entropy.py"}
    with pytest.raises(ValueError, match="canonical annotated"):
        registry.render_payload(spec, {}, {"figures": {}}, "vanilla")


def test_the_histogram_twin_table_repeats_its_source_figure_table():
    tables = _tables()
    assert tables[HISTOGRAM_ID] == {
        key: value
        for key, value in tables["candidate_tree_histogram"].items()
        if key != "notebook"
    }


def test_the_codon_entropy_table_names_the_band_smoothed_graph():
    assert _tables()[ENTROPY_ID]["candidate_title"] == "Band-smoothed candidate graph"


def test_the_entropy_figure_dims_every_branch_but_the_focused_ridge():
    table = _tables()[ENTROPY_ID]
    assert table["dimmed_branch_opacity"] == 0.25
    assert "non_bridge_branch_opacity" not in table  # replaced by the key above
    assert "entropy_rect_opacity" not in table
    # Branch colours are the count scale, with its colour bar on the top tree.
    assert table["colorbar_x_fraction"] == 0.9
    assert table["count_legend_ticks"] == [1, 100, 10000]


# -- tree histogram twin ------------------------------------------------------


def _view():
    tree = rx.PyDiGraph()
    saddle, kem, side, tcl = (
        tree.add_node(
            {
                "dna_sequence": name,
                "aa_sequence": "A",
                "has_stop_codon": False,
                "scalar": scalar,
            }
        )
        for name, scalar in (("SAD", -0.8), ("KEM", -0.2), ("K1A", -0.3), ("TCL", 0.2))
    )
    # Edges run from a peak or branch toward the saddle it drains into.
    tree.add_edge(kem, side, {})
    tree.add_edge(side, saddle, {})
    tree.add_edge(tcl, saddle, {})
    plot_data = make_tree_layout(
        tree,
        scalar_attr="scalar",
        root_kind="minimum",
        node_attrs=("dna_sequence", "aa_sequence", "has_stop_codon"),
    )
    # Rows sit on the branch from ``upper`` to ``lower``; the root group has none.
    rows = [
        ("SAD", -0.8, "SAD", "SAD"),
        ("KEM", -0.2, "KEM", "K1A"),
        ("X1", -0.25, "KEM", "K1A"),
        ("X2", -0.28, "KEM", "K1A"),
        ("K1A", -0.3, "K1A", "SAD"),
        ("TCL", 0.2, "TCL", "SAD"),
        ("R1", -0.7, "SAD", "SAD"),
    ]
    assignments = pl.DataFrame(
        {
            "dna_sequence": [row[0] for row in rows],
            "raw_fitness": [row[1] for row in rows],
            "upper_sequence": [row[2] for row in rows],
            "lower_sequence": [row[3] for row in rows],
        }
    )
    wild_type = pl.DataFrame({"x": [0.5], "y": [-0.1], "marker_label": ["WT"]})
    return SavedTreeLayout(
        plot_data=plot_data,
        assignments=assignments,
        branch_table=pl.DataFrame(
            {
                "source_tree_node_index": [kem, side, tcl],
                "target_tree_node_index": [side, saddle, saddle],
                "branch_vertex_count": [3, 1, 1],
            }
        ),
        wild_type_marker_table=wild_type,
    )


def test_a_smoothed_view_counts_each_branch_from_its_assignments():
    view = _view()
    shim = tree_histogram.selected_view_as_tree(view)
    labels = dict(
        zip(
            view.plot_data.node_table["tree_node_index"].to_list(),
            view.plot_data.node_table["dna_sequence"].to_list(),
            strict=True,
        )
    )
    counts = {
        frozenset(
            (
                labels[row["source_tree_node_index"]],
                labels[row["target_tree_node_index"]],
            )
        ): row["branch_vertex_count"]
        for row in shim.branch_table.iter_rows(named=True)
    }
    assert counts == {
        frozenset(("KEM", "K1A")): 3,  # the peak and two absorbed regular vertices
        frozenset(("K1A", "SAD")): 1,
        frozenset(("TCL", "SAD")): 1,
    }
    assert shim.branch_table is view.branch_table
    assert shim.wild_type_marker_table is view.wild_type_marker_table


def test_band_rendering_does_not_recount_assignments():
    view = _view()
    view.assignments = view.assignments.filter(pl.col("dna_sequence") != "TCL")
    assert tree_histogram.selected_view_as_tree(view).branch_table is view.branch_table


def test_selected_view_rejects_unrecognized_type():
    with pytest.raises(
        TypeError, match="saved tree layout or an in-memory SelectedLeafView"
    ):
        tree_histogram.selected_view_as_tree(SimpleNamespace())


def test_the_histogram_twin_reuses_the_candidate_builder_on_the_band_tree(monkeypatch):
    view = _view()
    candidate = SimpleNamespace(trees={"split": object()}, scalar_domain=(-0.9, 0.3))
    analysis = {
        "landscapes": {"candidate_clamped": candidate},
        "diagnostics": {"candidate_clamped": {"selected_component_vertex_count": 7}},
    }
    monkeypatch.setattr(registry, "bootstrap_band_view", lambda *_: view)
    calls = []
    real = tree_histogram._tree_histogram_figure

    def spy(
        analysis, settings, figure_id, item, tree_analysis, population_id, **kwargs
    ):
        calls.append((figure_id, item, population_id, kwargs))
        return real(
            analysis, settings, figure_id, item, tree_analysis, population_id, **kwargs
        )

    monkeypatch.setattr(tree_histogram, "_tree_histogram_figure", spy)
    settings = {
        "figures": {
            "rendering": {},
            "figures": {
                HISTOGRAM_ID: {"colorbar_x_fraction": 0.9, "tree": {"height": 200}}
            },
        }
    }
    chart = registry.get_spec(HISTOGRAM_ID).render(
        {
            "comparison": analysis,
            "bootstrap_smoothed": object(),
            "saved_figure_inputs": {
                "mode": "saved",
                "layouts": {},
                "populations": {
                    "candidate": view.assignments.select(
                        "dna_sequence", pl.col("raw_fitness").alias("scalar")
                    )
                },
            },
        },
        settings,
        "annotated",
    )
    # The band tree is the twin's own: its branches carry count labels.
    assert calls == [(HISTOGRAM_ID, candidate, "candidate", {"count_labels": True})]
    spec = chart.to_dict()
    histogram = next(
        rows
        for rows in spec["datasets"].values()
        if rows and {"bin_min", "bin_max", "count"}.issubset(rows[0])
    )
    assert sum(row["count"] for row in histogram) == 7  # every vertex once
    assert len(spec["hconcat"]) == 2


def _histogram_spec(monkeypatch, **label_keys):
    view = _view()
    candidate = SimpleNamespace(trees={"split": object()}, scalar_domain=(-0.9, 0.3))
    analysis = {
        "landscapes": {"candidate_clamped": candidate},
        "diagnostics": {"candidate_clamped": {"selected_component_vertex_count": 7}},
    }
    monkeypatch.setattr(registry, "bootstrap_band_view", lambda *_: view)
    settings = {
        "figures": {
            "rendering": label_keys,
            "figures": {
                HISTOGRAM_ID: {"colorbar_x_fraction": 0.9, "tree": {"height": 200}}
            },
        }
    }
    return (
        registry.get_spec(HISTOGRAM_ID)
        .render(
            {
                "comparison": analysis,
                "bootstrap_smoothed": object(),
                "saved_figure_inputs": {
                    "mode": "saved",
                    "layouts": {},
                    "populations": {
                        "candidate": view.assignments.select(
                            "dna_sequence", pl.col("raw_fitness").alias("scalar")
                        )
                    },
                },
            },
            settings,
            "annotated",
        )
        .to_dict()
    )


def _count_label_rows(spec):
    rows = []
    for item in _walk(spec):
        mark = item.get("mark")
        if (
            isinstance(mark, dict)
            and mark.get("type") == "text"
            and mark.get("color") == "#737373"
        ):
            rows.append((mark, spec["datasets"][item["data"]["name"]]))
    return rows


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_the_histogram_twin_colours_branches_by_count_and_labels_the_small_tips(
    monkeypatch,
):
    spec = _histogram_spec(
        monkeypatch, branch_count_label_mode="tips", branch_count_label_max=1
    )
    tree = spec["hconcat"][0]
    # Branch colour is the post-absorption count (the KEM branch folds in X1 and X2)
    # on the log black opacity ramp over the tree's counts.
    (edges,) = [
        item
        for item in _walk(tree)
        if isinstance(item.get("encoding"), dict)
        and item["encoding"].get("color", {}).get("field") == "vertex_count"
        and "scale" in item["encoding"]["color"]
    ]
    color = edges["encoding"]["color"]
    assert color["scale"] == {
        "scheme": "greys",
        "type": "log",
        "domain": [1.0, 3.0],
        "range": ["rgba(0,0,0,0.3)", "rgba(0,0,0,1.0)"],
    }
    assert "condition" not in color
    by_branch = {
        (row["source_label"], row["target_label"]): row["vertex_count"]
        for item in _walk(tree)
        if isinstance(item.get("data"), dict)
        for row in item["data"].get("values", [])
        if "vertex_count" in row and "source_label" in row
    }
    assert by_branch == {("K1A", "KEM"): 3, ("SAD", "K1A"): 1, ("SAD", "TCL"): 1}
    # Count labels in the tips mode: only the TCL tip (one vertex) is at most the
    # maximum (the KEM tip has three; K1A is not a tip), in the peak grey and the
    # body font less two.
    ((mark, rows),) = _count_label_rows(spec)
    assert mark["fontSize"] == 13
    assert [(row["dna_sequence"], row["label"]) for row in rows] == [("TCL", "1")]
    # A maximum of three takes in the KEM tip; below one, no labels.
    three = _histogram_spec(
        monkeypatch, branch_count_label_mode="tips", branch_count_label_max=3
    )
    ((_, rows),) = _count_label_rows(three)
    assert sorted((row["dna_sequence"], row["label"]) for row in rows) == [
        ("KEM", "3"),
        ("TCL", "1"),
    ]
    assert not _count_label_rows(
        _histogram_spec(
            monkeypatch, branch_count_label_mode="tips", branch_count_label_max=0
        )
    )


def test_the_histogram_twin_labels_every_branch_in_the_all_mode(monkeypatch):
    spec = _histogram_spec(
        monkeypatch, branch_count_label_mode="all", branch_count_label_max=0
    )
    ((_, rows),) = _count_label_rows(spec)
    # The tips and the internal branch K1A, each with its post-absorption count.
    assert sorted((row["dna_sequence"], row["label"]) for row in rows) == [
        ("K1A", "1"),
        ("KEM", "3"),
        ("TCL", "1"),
    ]
    # The collision variant is a config choice: rotated, the same branches.
    turned = _histogram_spec(
        monkeypatch,
        branch_count_label_mode="all",
        branch_count_label_collision="rotate",
    )
    ((mark, turned_rows),) = _count_label_rows(turned)
    assert mark["angle"] == 270
    assert sorted(row["dna_sequence"] for row in turned_rows) == ["K1A", "KEM", "TCL"]


def test_the_candidate_histogram_has_no_count_labels(monkeypatch):
    view = _view()
    candidate = SimpleNamespace(trees={"split": object()}, scalar_domain=(-0.9, 0.3))
    analysis = {
        "landscapes": {"candidate_clamped": candidate},
        "diagnostics": {"candidate_clamped": {"selected_component_vertex_count": 7}},
        "saved_populations": {
            "candidate": view.assignments.select(
                "dna_sequence", pl.col("raw_fitness").alias("scalar")
            )
        },
    }
    settings = {
        "figures": {
            "rendering": {"branch_count_label_mode": "all"},
            "figures": {
                HISTOGRAM_ID: {"colorbar_x_fraction": 0.9, "tree": {"height": 200}}
            },
        }
    }
    chart = tree_histogram._tree_histogram_figure(
        analysis,
        settings,
        HISTOGRAM_ID,
        candidate,
        tree_histogram.selected_view_as_tree(view),
        "candidate",
    ).to_dict()
    assert not _count_label_rows(chart)


# -- codon entropy twin --------------------------------------------------------


def test_the_sweeps_are_shared_with_the_table_stage(tmp_path):
    frame = _entropy_frame([(0.0, "A", "E", "M"), (1.0, "K", "D", "M")])
    direct = entropy_module.compute_entropy_sweeps(
        {"core": frame, "kem": frame, "tcl": frame}, num_thresholds=3
    )
    for name in ("candidate", "kem_ridge", "tcl_ridge"):
        frame.write_parquet(tmp_path / f"papkou2023_{name}_population.parquet")
    staged = entropy_module.process_codon_entropy_datasets(
        input_dir=tmp_path, save_files=False, num_thresholds=3
    )
    assert set(direct) == {"core", "kem", "tcl"}
    for name in direct:
        assert direct[name].equals(staged[name])


def test_the_codon_composer_runs_the_table_stage_functions_on_the_band_view(
    monkeypatch,
):
    from dubanevics2026topological.analysis import (
        codon_entropy_populations as populations,
    )

    view = object()
    frames = {
        "df_core": pl.DataFrame({"fitness": [0.0, 1.0], "dna_sequence": ["A", "B"]}),
        "df_kem": pl.DataFrame({"fitness": [1.0], "dna_sequence": ["B"]}),
        "df_tcl": pl.DataFrame({"fitness": [0.0], "dna_sequence": ["A"]}),
    }
    seen = {}

    def extract(**kwargs):
        seen["extract"] = kwargs
        return {"metadata": {"m": 1}, "candidate_leaf_view": view, **frames}

    def sweeps(populations_in, *, num_thresholds):
        seen["sweeps"] = (populations_in, num_thresholds)
        return {"core": 1, "kem": 2, "tcl": 3}

    monkeypatch.setattr(populations, "extract_codon_entropy_populations", extract)
    monkeypatch.setattr(entropy_module, "compute_entropy_sweeps", sweeps)
    monkeypatch.setattr(registry, "_legacy_codon_band_view", lambda *_: view)
    monkeypatch.setattr(
        "dubanevics2026topological.landscape.selection.resolve_candidate_selection",
        fixed_mixture_selection({"A": "functional", "B": "non_functional"}),
    )
    config = {"sweep_points": 7}
    quantities = object()
    payload = registry.compose_payload(
        registry.get_spec(ENTROPY_ID),
        {
            "comparison": {"landscapes": "c"},
            "bootstrap_smoothed": object(),
            "manuscript_quantities": quantities,
        },
        {"analysis": {"codon_entropy": config}},
    )
    assert seen["extract"] == {
        "comparison": {"landscapes": "c"},
        "candidate_view": view,
        "config": config,
        "save_files": False,
        "functional_sequences": frozenset({"A"}),
    }
    assert seen["sweeps"][1] == 7
    assert set(seen["sweeps"][0]) == {"core", "kem", "tcl"}
    assert payload["candidate_leaf_view"] is view
    assert payload["entropy"] == {"core": 1, "kem": 2, "tcl": 3}
    assert payload["landscapes"] == "c"
    assert payload["metadata"] == {"m": 1}
    assert payload["manuscript_quantities"] is quantities


def _render_payload(with_view: bool):
    tree, _ = _ridge_tree()
    for index in tree.node_indices():
        tree[index].setdefault("has_stop_codon", False)
        tree[index].setdefault("aa_sequence", "A")
    plot_data = make_tree_layout(
        tree,
        scalar_attr="scalar",
        root_kind="minimum",
        node_attrs=("dna_sequence", "aa_sequence", "has_stop_codon"),
    )
    view = SimpleNamespace(
        tree=tree,
        plot_data=plot_data,
        assignments=_assignments(),
        wild_type_marker_table=pl.DataFrame(
            {"x": [0.5], "y": [-0.1], "marker_label": ["WT"]}
        ),
    )
    thresholds = [-0.8, -0.4, -0.2, -0.1, 0.2]
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
    payload = with_labels(
        {
            "metadata": {
                "functional_threshold": -0.5,
                "summits": {
                    "kem": {"dna_sequence": "KEM"},
                    "tcl": {"dna_sequence": "TCL"},
                },
                "common_saddle": {"dna_sequence": "SAD"},
            },
            "df_core": pl.DataFrame({"fitness": [-0.8, -0.4, -0.2, -0.1, 0.2]}),
            "df_kem": pl.DataFrame({"fitness": [-0.4, -0.3, -0.2]}),
            "df_tcl": pl.DataFrame({"fitness": [0.1, 0.2]}),
            "entropy": entropy,
        }
    )
    if with_view:
        payload["candidate_leaf_view"] = view
    return payload


_SHIPPED_CODON = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "config" / "figures.toml").read_text()
)["figures"][ENTROPY_ID]

_CODON_SETTINGS = {
    "figures": {
        "rendering": {"fitness_tick_step": 0.1},
        "figures": {
            ENTROPY_ID: {
                "candidate_title": "Band-smoothed candidate graph",
                "ridge_labels": ["KEM ridge", "TCL ridge"],
                "colorbar_x_fraction": 0.9,
                "entropy_legend_orient": _SHIPPED_CODON["entropy_legend_orient"],
                "entropy_legend_length": _SHIPPED_CODON["entropy_legend_length"],
            },
        },
    }
}


def test_the_codon_figure_keeps_row_headings_out_of_the_panels():
    figure = build_bootstrap_codon_entropy_figure(
        _render_payload(True), _CODON_SETTINGS
    ).to_dict()
    tree_row, sweep_row = figure["vconcat"]
    legend_row = sweep_row["hconcat"][3]
    # Every heading belongs to its row: the tree carries none that could widen it.
    assert all("title" not in panel for panel in tree_row["hconcat"])
    for row in (tree_row, sweep_row):
        title = row["title"]
        assert title["anchor"] == "start" and "limit" not in title
    # No encoding legend: one vertical bar, drawn in pixels right of the three sweeps.
    def _walk(node):
        if isinstance(node, dict):
            yield node
            for child in node.values():
                yield from _walk(child)
        elif isinstance(node, list):
            for child in node:
                yield from _walk(child)

    legends = [
        item["encoding"]["color"].get("legend")
        for item in _walk(sweep_row)
        if isinstance(item.get("encoding"), dict)
        and item["encoding"].get("color", {}).get("field") == "entropy_bits"
    ]
    assert len(legends) == 3 and all(legend is None for legend in legends)
    texts = [
        item["mark"].get("text")
        for item in _walk(legend_row)
        if isinstance(item.get("mark"), dict) and item["mark"].get("type") == "text"
    ]
    assert {"Entropy", "(bits)"} <= set(texts)
    # The bar is vertical: its gradient runs top to bottom (y1 -> y2, no x change).
    gradients = [
        item["mark"]["color"]
        for item in _walk(legend_row)
        if isinstance(item.get("mark"), dict)
        and item["mark"].get("type") == "rect"
        and isinstance(item["mark"].get("color"), dict)
        and "stops" in item["mark"]["color"]
    ]
    assert gradients
    assert all(
        (g["x1"], g["x2"], g["y1"], g["y2"]) == (0, 0, 0, 1) for g in gradients
    )


def test_the_codon_renderer_requires_its_selected_leaf_view():
    with pytest.raises(KeyError, match="candidate_leaf_view"):
        build_bootstrap_codon_entropy_figure(_render_payload(False), _CODON_SETTINGS)
    payload = {**_render_payload(True), "candidate_leaf_view": None}
    with pytest.raises(ValueError, match="candidate_leaf_view"):
        build_bootstrap_codon_entropy_figure(payload, _CODON_SETTINGS)


def test_the_codon_twin_draws_with_its_own_table():
    settings = {
        "figures": {
            "rendering": {"fitness_tick_step": 0.1},
            "figures": {
                ENTROPY_ID: {
                    "candidate_title": "Band-smoothed candidate graph",
                    "dimmed_branch_opacity": 0.1,
                    "colorbar_x_fraction": 0.9,
                },
            },
        }
    }
    payload = _render_payload(True)
    chart = registry.render_payload(
        registry.get_spec(ENTROPY_ID), payload, settings, "annotated"
    ).to_dict()
    assert chart["vconcat"][0]["title"]["text"].startswith(
        "Band-smoothed candidate graph · "
    )

    def marks(chart):
        found = []

        def walk(node):
            if isinstance(node, dict):
                mark = node.get("mark")
                if isinstance(mark, dict) and mark.get("type") in {"rule", "rect"}:
                    found.append((mark["type"], mark.get("opacity")))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(chart)
        return found

    # One tree: every branch and both highlighted peak branches are solid; the
    # sweeps stay opaque.
    opacities = marks(chart)
    assert ("rule", 1.0) in opacities
    assert ("rule", 0.1) not in opacities
    assert ("rect", 0.1) not in opacities
