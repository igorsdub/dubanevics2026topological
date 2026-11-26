"""Bootstrap-smoothed figure: hill outcomes, composition, rendering and build."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import polars as pl
import pytest
import rustworkx as rx
from _fake_figures import (
    install_fake_figures,
    install_fake_registry,
    write_renderer_sources,
)
from _figure_labels import SENTINEL, label_quantities
from _vega import outcome_key, outcome_key_layout, vega_legend_groups
from test_bootstrap_smoothed_tree import (
    CONFIG,
    _checkpoint_fingerprint,
    _cooc,
    _growth,
    _patch_seam,
    _toy,
)
from topographer.plotting import make_tree_layout

from dubanevics2026topological import build
from dubanevics2026topological.analysis import (
    candidate_tree_contrast as candidate_tree_contrast_module,
)
from dubanevics2026topological.figures import (
    candidate_tree_bootstrap_biology as biology_module,
)
from dubanevics2026topological.figures import registry
from dubanevics2026topological.simplification import bootstrap_smoothed_tree as module
from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
    build_bootstrap_smoothed_tree,
)
from dubanevics2026topological.simplification.identities import (
    band_identity,
    checkpoint_draw_identity,
)

BIOLOGY_ID = "candidate_tree_bootstrap_biology"
# Only b has a core: d and f are nominated. f is the elder arm at saddle e
# (protected), d is absorbed, b is kept.
CORES = [["b", "a"]]


def _cached(monkeypatch, cores=None):
    """Return a checkpoint result on the toy tree and the toy tree itself."""
    _patch_seam(monkeypatch, _cooc(CORES if cores is None else cores))
    prepared, tree = _toy()
    cached = build_bootstrap_smoothed_tree(
        prepared, tree, _growth(), CONFIG, log=lambda message: None
    )
    cached.provenance = {"fingerprint": {"generation": 1}, "artifacts": {}}
    return cached, tree


# -- registry ------------------------------------------------------------------


def test_figure_is_opt_in_annotated_and_configured():
    import tomllib

    spec = registry.get_spec(BIOLOGY_ID)
    assert spec.default is False
    assert spec.variants == {"annotated"}
    assert {"bootstrap_smoothed", "manuscript_quantities"} <= spec.inputs
    assert spec.config_table == BIOLOGY_ID
    assert BIOLOGY_ID not in [
        item.figure_id for item in registry.FIGURE_SPECS if item.default
    ]
    tables = tomllib.loads(
        (Path(build.ROOT) / "config/figures.toml").read_text("utf-8")
    )
    assert BIOLOGY_ID in tables["figures"]
    with pytest.raises(ValueError, match="canonical annotated"):
        registry.render_payload(spec, {}, {"figures": {}}, "vanilla")


def test_biology_fingerprint_tracks_the_shared_assembler():
    spec = registry.get_spec(BIOLOGY_ID)
    assert registry.renderer_files(spec) == (
        "candidate_tree_bootstrap_biology.py",
        "candidate_tree_biology.py",
    )
    assert "branch_comembership_heatmap" not in {
        spec.figure_id for spec in registry.FIGURE_SPECS
    }


def test_figure_tables_carry_the_threshold_presentation_settings():
    import tomllib

    tables = tomllib.loads(
        (Path(build.ROOT) / "config/figures.toml").read_text("utf-8")
    )["figures"]
    main = tables[BIOLOGY_ID]
    for key in (
        "survives_color",
        "absorbed_color",
        "band_color",
        "band_label",
        "legend_title",
        "legend_labels",
        "nominal_row_title",
        "nominal_row_subtitle",
        "band_row_title",
        "band_row_subtitle",
    ):
        assert key in main
    assert "core_row_title" not in main  # the core rule has no row of its own
    # The threshold is named in words: no Greek letter, no "2c" and no "q =" shorthand,
    # and no format spec (the numbers are the registry's display text).
    text = "\n".join(
        str(value)
        for key, value in main.items()
        if key.endswith(("title", "subtitle", "label", "labels"))
    )
    assert "half band" not in text.lower()
    for shorthand in ("τ", "2c", "q =", ":,", ":.3f", ":g"):
        assert shorthand not in text, shorthand


# -- composition ---------------------------------------------------------------


def _inputs(cached, tree, comparison=None):
    cached.summary.setdefault("fingerprint", _checkpoint_fingerprint())
    checkpoint_settings = cached.summary["settings"]
    _, draw_id = checkpoint_draw_identity(
        cached.summary["fingerprint"],
        seed=int(checkpoint_settings["seed"]),
        draw_ids=range(int(checkpoint_settings["n_replicates"])),
    )
    _, band_id = band_identity(
        draw_scientific_id_sha256=draw_id,
        bandq=float(checkpoint_settings["band_quantile"]),
    )
    return {
        "comparison": comparison
        or {"landscapes": {"candidate_clamped": SimpleNamespace(trees={"split": 1})}},
        "bootstrap_smoothed": cached,
        "manuscript_quantities": label_quantities(),
        "saved_figure_inputs": {
            "mode": "saved",
            "layouts": {
                "candidate": tree,
                "band": module.build_band_simplified_view(tree, cached.hills),
            },
            "nominal_hills": cached.hills.select(
                "peak_sequence", "member_count"
            ).rename({"peak_sequence": "peak_dna_sequence"}),
            "annotations": object(),
            "direct_band_identity": band_id,
            "product_ids": {"candidate": "fixture"},
        },
    }


def test_composer_reuses_saved_tree_and_view(monkeypatch):
    cached, tree = _cached(monkeypatch)
    inputs = _inputs(cached, tree)
    monkeypatch.setattr(
        candidate_tree_contrast_module,
        "build_candidate_tree",
        lambda settings: pytest.fail("figure rebuilt a scientific tree"),
    )
    payload = registry.compose_payload(registry.get_spec(BIOLOGY_ID), inputs, {"s": 1})
    again = registry.compose_payload(registry.get_spec(BIOLOGY_ID), inputs, {"s": 1})
    assert payload["bootstrap_tree"] is tree is again["bootstrap_tree"]
    assert payload["bootstrap_smoothed"] is cached is again["bootstrap_smoothed"]
    assert payload["manuscript_quantities"] is inputs["manuscript_quantities"]
    assert payload["landscapes"] is inputs["comparison"]["landscapes"]
    assert "bootstrap_view" not in payload
    # The band view is built once directly from the same checked tree.
    assert payload["bootstrap_band_view"] is again["bootstrap_band_view"]
    assert set(payload["bootstrap_band_view"].pairs["birth_label"]) == {"f"}


def test_composer_refuses_a_checkpoint_from_another_tree(monkeypatch):
    cached, tree = _cached(monkeypatch)
    inputs = _inputs(cached, tree)
    cached.hills = cached.hills.with_columns(
        pl.when(pl.col("peak_sequence") == "b")
        .then(pl.lit("x"))
        .otherwise(pl.col("peak_sequence"))
        .alias("peak_sequence")
    )
    monkeypatch.setattr(
        candidate_tree_contrast_module, "build_candidate_tree", lambda s: (tree, None)
    )
    with pytest.raises(ValueError):
        registry.compose_payload(registry.get_spec(BIOLOGY_ID), inputs, {"s": 1})


@pytest.mark.parametrize("field, value", [("seed", 8), ("band_quantile", 0.9)])
def test_composer_refuses_different_band_identity_with_same_survivors(
    monkeypatch, field, value
):
    cached, tree = _cached(monkeypatch)
    inputs = _inputs(cached, tree)
    survivors = cached.hills["clears_band"].to_list()
    cached.summary["settings"][field] = value
    assert cached.hills["clears_band"].to_list() == survivors
    with pytest.raises(ValueError, match="band identity differs"):
        registry.compose_payload(registry.get_spec(BIOLOGY_ID), inputs, {"s": 1})


def test_composer_rejects_duplicate_draw_ids(monkeypatch):
    cached, tree = _cached(monkeypatch)
    inputs = _inputs(cached, tree)
    cached.draw_stats = cached.draw_stats.with_columns(
        pl.when((pl.col("condition") == "real") & (pl.col("draw") == 1))
        .then(pl.lit(0))
        .otherwise(pl.col("draw"))
        .alias("draw")
    )
    with pytest.raises(ValueError, match="incomplete or duplicated draw IDs"):
        registry.compose_payload(registry.get_spec(BIOLOGY_ID), inputs, {"s": 1})


def test_load_inputs_reads_the_cached_checkpoint(monkeypatch):
    monkeypatch.setattr(
        module, "load_cached_bootstrap_smoothed_tree", lambda *_: "cached"
    )
    assert build.load_inputs(["bootstrap_smoothed"]) == {"bootstrap_smoothed": "cached"}


# -- real-shaped tree: the composer's tree feeds the biological annotations ------


def test_composer_tree_has_the_branch_columns_the_annotations_read(monkeypatch):
    """Regression: the real branch table has ``branch_vertex_count`` only.

    ``build_biological_annotations`` once read ``branch_member_count``, a column
    Topographer renamed, so the real-data figure raised ``KeyError``. The tree
    here is the one the composer hands over (a real Topographer split tree, no
    stand-in), and the annotations are not stubbed.
    """
    from dubanevics2026topological.analysis.biological_annotations import (
        _build_biological_annotations as build_biological_annotations,
    )

    cached, toy_tree = _cached(monkeypatch)
    monkeypatch.setattr(
        candidate_tree_contrast_module,
        "build_candidate_tree",
        lambda settings: (toy_tree, None),
    )
    payload = registry.compose_payload(
        registry.get_spec(BIOLOGY_ID),
        _inputs(cached, toy_tree, {"landscapes": {}, "tables": {}}),
        {"s": 1},
    )
    tree = payload["bootstrap_tree"]
    assert "branch_member_count" not in tree.branch_table.columns
    assert "branch_vertex_count" in tree.branch_table.columns

    evidence = build_biological_annotations(
        SimpleNamespace(trees={"split": tree}), landscape="candidate_clamped"
    )
    counts = dict(
        zip(
            evidence.pairs["birth_sequence"],
            evidence.pairs["branch_vertex_count"],
            strict=True,
        )
    )
    # The hills of the toy tree are b = {b, a}, d = {d} and f = {f, g}.
    assert counts == {"b": 2, "d": 1, "f": 2}


# -- biology figure ------------------------------------------------------------


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _stub_chart_parts(monkeypatch, labels):
    """Replace the data-heavy helpers; the figure's own logic stays real."""
    from dubanevics2026topological.analysis import biological_annotations
    from dubanevics2026topological.figures import candidate_tree_biology as shared

    pairs = pl.DataFrame(
        {
            "birth_sequence": labels,
            "death_sequence": ["s"] * len(labels),
            "birth_scalar": [0.8, 0.7, 0.6, 0.9][: len(labels)],
            "death_scalar": [0.4] * len(labels),
            "birth_has_stop_codon": [False] * (len(labels) - 1) + [True],
        }
    )
    evidence = SimpleNamespace(
        pairs=pairs,
        critical_points=pl.DataFrame(
            {
                "dna_sequence": labels,
                "aa_sequence": ["A"] * (len(labels) - 1) + ["*"],
                "has_stop_codon": [False] * (len(labels) - 1) + [True],
                "is_peak": [True] * len(labels),
                "event": ["maximum"] * len(labels),
                "x": [0.8, 0.7, 0.6, 0.9][: len(labels)],
                "y": [0.8, 0.7, 0.6, 0.9][: len(labels)],
            }
        ),
    )
    monkeypatch.setattr(
        biological_annotations,
        "_build_biological_annotations",
        lambda *a, **k: evidence,
    )
    segments = _plot_data(evidence).edge_segment_table
    monkeypatch.setattr(
        biology_module,
        "plot_data_with_saved_metrics",
        lambda *_: SimpleNamespace(
            edge_table=pl.DataFrame(
                {"target_label": ["b", "d", "f"], "vertex_count": [1, 1, 2]}
            ),
            edge_segment_table=segments,
            node_table=pl.DataFrame(
                {"dna_sequence": ["b", "d", "f"], "tree_node_index": [0, 1, 2]}
            ),
        ),
    )
    base = pl.DataFrame({"x": [0.0], "y": [0.0]})
    monkeypatch.setattr(
        shared,
        "study_biology_tree_chart",
        lambda *a, **k: (
            alt.Chart(base)
            .mark_line()
            .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
        ),
    )
    monkeypatch.setattr(
        shared,
        "study_biology_persistence_chart",
        lambda *a, **k: (
            alt.Chart(pl.DataFrame({"death_scalar": [0.0], "birth_scalar": [0.0]}))
            .mark_point()
            .encode(
                x=alt.X("death_scalar:Q", axis=None),
                y=alt.Y("birth_scalar:Q", axis=None),
            )
        ),
    )
    return evidence


def _plot_data(evidence):
    """Topographer-shaped plot data: nodes by index and one segment per peak branch."""
    labels = evidence.critical_points["dna_sequence"].to_list()
    nodes = evidence.critical_points.with_columns(
        pl.Series("tree_node_index", list(range(len(labels))), dtype=pl.Int64),
        pl.lit("maximum").alias("event"),
    )
    height = [0.8, 0.7, 0.6, 0.9][: len(labels)]
    segments = pl.DataFrame(
        {
            "source_tree_node_index": list(range(len(labels))),
            "target_tree_node_index": [len(labels)] * len(labels),
            "logical_edge_index": list(range(len(labels))),
            "x_start": height,
            "x_end": height,
            "y_start": [0.4] * len(labels),
            "y_end": height,
        }
    )
    return SimpleNamespace(node_table=nodes, edge_segment_table=segments)


def _band_plot_data(counts=None):
    """A real layout of the toy band tree: peaks b, d, f drain into the saddle s.

    The branches are the figure's counted branches, so unlike the stubbed nominal
    tree this one goes through ``plot_data_with_metrics``. ``counts`` maps a peak
    to extra sequences on its branch (``assignments`` rows below).
    """
    tree = rx.PyDiGraph()
    saddle = tree.add_node(
        {
            "dna_sequence": "s",
            "aa_sequence": "A",
            "has_stop_codon": False,
            "scalar": 0.4,
        }
    )
    for name, scalar in (("b", 0.7), ("d", 0.55), ("f", 0.8)):
        peak = tree.add_node(
            {
                "dna_sequence": name,
                "aa_sequence": "*" if name == "f" else "A",
                "has_stop_codon": name == "f",
                "scalar": scalar,
            }
        )
        tree.add_edge(peak, saddle, {})
    return make_tree_layout(
        tree,
        scalar_attr="scalar",
        root_kind="minimum",
        node_attrs=("dna_sequence", "aa_sequence", "has_stop_codon"),
    )


def _band_assignments(extra=None):
    """Sequences on the toy band tree's branches; ``extra`` adds members to a branch."""
    rows = [("b", 0.7, "b"), ("d", 0.55, "d"), ("f", 0.8, "f")]
    rows += [
        (f"{peak}{index}", 0.5, peak)
        for peak, count in (extra or {}).items()
        for index in range(count)
    ]
    return pl.DataFrame(
        {
            "dna_sequence": [row[0] for row in rows],
            "raw_fitness": [row[1] for row in rows],
            "upper_sequence": [row[2] for row in rows],
            "lower_sequence": ["s"] * len(rows),
        }
    )


def _saved_band_branch_table(plot_data, assignments):
    nodes = dict(
        plot_data.node_table.select("tree_node_index", "dna_sequence").iter_rows()
    )
    counts = {}
    for upper, lower in assignments.select(
        "upper_sequence", "lower_sequence"
    ).iter_rows():
        key = frozenset((upper, lower))
        counts[key] = counts.get(key, 0) + 1
    return pl.DataFrame(
        [
            {
                "source_tree_node_index": source,
                "target_tree_node_index": target,
                "branch_vertex_count": counts[
                    frozenset((nodes[source], nodes[target]))
                ],
            }
            for source, target in plot_data.edge_table.select(
                "source_tree_node_index", "target_tree_node_index"
            ).iter_rows()
        ]
    )


def _biology_payload(monkeypatch):
    cached, toy_tree = _cached(monkeypatch)
    # Toy hills are b, d, f; the stub tree labels them with the same sequences.
    evidence = _stub_chart_parts(monkeypatch, ["b", "d", "f"])
    wild_type = pl.DataFrame(
        {
            "dna_sequence": ["wild-type"],
            "aa_sequence": ["K"],
            "marker_label": ["WT"],
            "x": [0.1],
            "y": [0.0],
        }
    )
    # The real toy tree supplies the terminal edges the figure joins outcomes to.
    plot_data = _plot_data(evidence)
    nominal = SimpleNamespace(
        tree=toy_tree.tree,
        branch_member_table=toy_tree.branch_member_table,
        wild_type_marker_table=wild_type,
        plot_data=plot_data,
    )
    candidate = SimpleNamespace(trees={"split": nominal}, scalar_domain=(-0.9, 0.9))
    view = SimpleNamespace(
        plot_data=plot_data,
        pairs=evidence.pairs.rename(
            {"birth_sequence": "birth_label", "death_sequence": "death_label"}
        ).filter(pl.col("birth_label") != "d"),
        assignments=pl.DataFrame({"dna_sequence": ["b", "d", "f"]}),
        wild_type_marker_table=wild_type,
    )
    # Only f clears 2c = 0.72 (toy persistences: f 0.8, b 0.7, d 0.55).
    from dubanevics2026topological.results.presentation import SavedTreeLayout

    band_view = SavedTreeLayout(
        plot_data=_band_plot_data(),
        pairs=view.pairs.filter(pl.col("birth_label") == "f"),
        assignments=_band_assignments(),
        wild_type_marker_table=wild_type,
    )
    band_view.branch_table = _saved_band_branch_table(
        band_view.plot_data, band_view.assignments
    )
    return {
        "landscapes": {"candidate_clamped": candidate},
        "tables": {"candidate_clamped_branch_table": pl.DataFrame()},
        "bootstrap_smoothed": cached,
        "bootstrap_tree": candidate.trees["split"],
        "bootstrap_view": view,
        "bootstrap_band_view": band_view,
        "candidate_annotations": evidence,
        "manuscript_quantities": label_quantities(),
    }


def _biology_settings():
    return {
        "figures": {
            "rendering": {},
            "figures": {
                BIOLOGY_ID: {
                    "tree": {"height": 200},
                    "colorbar_x_fraction": 0.9,
                    "survives_color": "#111111",
                    "absorbed_color": "#222222",
                    "band_label": "band: persistence ≤ threshold",
                    "legend_title": "Peak",
                },
            },
        }
    }


def _render(payload, settings, figure_id=BIOLOGY_ID):
    return registry.render_payload(
        registry.get_spec(figure_id), payload, settings, "annotated"
    ).to_dict()


def _marks(node, kind):
    return [item for item in _walk(node) if item.get("mark", {}).get("type") == kind]


_DIAGRAM_LEGEND_TEXTS = frozenset(
    {"peak", "Stop codon containing peaks", "wild type", "synonymous leaves"}
)


def _diagram_legend_labels(diagram):
    return [
        item["mark"]["text"]
        for item in _marks(diagram, "text")
        if item["mark"].get("text") in _DIAGRAM_LEGEND_TEXTS
    ]


def test_biology_has_two_rows_with_titles_counts_and_a_legend(monkeypatch):
    payload = _biology_payload(monkeypatch)
    spec = _render(payload, _biology_settings())
    rows = spec["vconcat"]
    assert len(rows) == 2  # the nominal tree and the band-simplified tree
    # Toy: 3 nominal peaks, f clears the threshold 0.72 (persistence f 0.8, b 0.7, d 0.55).
    summary = payload["bootstrap_smoothed"].summary
    assert float(summary["band_simplification"]["two_c"]) == pytest.approx(0.72)
    # Every number is the registry's display text; the toy tables hold 3 nominal
    # peaks, 1 survivor, 2 absorbed, 1 stop-codon peak, 20 draws, seed 7, threshold
    # 0.72 at band quantile 0.95, 3 sequences retained and 1 saddle removed.
    shown = SENTINEL
    survive = shown["bootstrap.candidate.after_band_rule.peak_count"]
    absorbed = shown["bootstrap.candidate.total_absorbed_count"]
    nominal = shown["bootstrap.candidate.nominal.peak_count"]
    assert rows[0]["title"]["text"] == (
        f"Nominal tree · {nominal} peaks · {survive} survive and {absorbed} are "
        "absorbed (persistence ≤ threshold) · "
        f"{shown['annotations.candidate.stop_codon_peak_count']} stop-codon containing"
    )
    dataset = (
        f"Papkou et al. (2023) · Topographer + FDR, bootstrapped · {shown['bootstrap.draws']} "
        f"bootstrap draws · seed {shown['bootstrap.seed']}"
    )
    threshold = shown["bootstrap.persistence_threshold.value"]
    quantile = shown["bootstrap.band_quantile"]
    assert rows[0]["title"]["subtitle"] == [
        dataset,
        (
            f"Bootstrapped persistence threshold {threshold} at band quantile "
            f"{quantile} · {shown['bootstrap.band_quantile_percent']}% "
            "simultaneous confidence band (parametric bootstrap)"
        ),
        (
            "Stop codon containing peaks: crimson × on the tree, "
            "crimson × inside the triangle on the diagram"
        ),
    ]
    assert rows[1]["title"]["text"] == (
        f"Bootstrapped tree · {survive} peaks, all with persistence > "
        f"threshold · {absorbed} of {nominal} nominal peaks absorbed"
    )
    retained = shown["bootstrap.candidate.after_band_rule.retained_sequence_count"]
    saddles = shown["bootstrap.candidate.after_band_rule.saddles_removed_count"]
    assert rows[1]["title"]["subtitle"] == [
        (
            f"{dataset} · bootstrapped persistence threshold {threshold} at band "
            f"quantile {quantile}"
        ),
        f"{retained} sequences retained · {saddles} saddles removed · raw fitness unchanged",
    ]
    assert set(payload["manuscript_quantities"].requested_keys) == {
        "bootstrap.candidate.nominal.peak_count",
        "bootstrap.candidate.after_band_rule.peak_count",
        "bootstrap.candidate.total_absorbed_count",
        "annotations.candidate.stop_codon_peak_count",
        "bootstrap.draws",
        "bootstrap.seed",
        "bootstrap.persistence_threshold.value",
        "bootstrap.band_quantile",
        "bootstrap.band_quantile_percent",
        "bootstrap.candidate.after_band_rule.retained_sequence_count",
        "bootstrap.candidate.after_band_rule.saddles_removed_count",
    }
    for row in rows:  # "half band" and a bare "c" never reach the figure text
        text = str(row["title"])
        assert "half band" not in text.lower()
    title, entries = outcome_key(rows[0]["hconcat"][0])
    assert title == "Peak"
    # ADR 0012: every outcome is a filled marker; the colour tells them apart.
    assert entries == [
        (f"survives: persistence > threshold ({survive})", "#111111", True),
        (f"absorbed: persistence ≤ threshold ({absorbed})", "#222222", True),
    ]
    # The key is marks inside the tree, not a Vega legend; the band tree has none.
    assert not vega_legend_groups(
        biology_module.build_candidate_tree_bootstrap_biology_figure(
            payload, _biology_settings()
        )
    )
    assert title is not None
    assert outcome_key(rows[1]["hconcat"][0]) == (None, [])


def test_biology_bottom_row_has_no_diagram_legend(monkeypatch):
    """The in-diagram key is on the nominal row only; the band row has none."""
    payload = _biology_payload(monkeypatch)
    spec = _render(payload, _biology_settings())
    top, bottom = (row["hconcat"][1] for row in spec["vconcat"])
    assert _diagram_legend_labels(top) == ["peak", "Stop codon containing peaks"]
    assert _diagram_legend_labels(bottom) == []


def test_biology_diagrams_draw_the_band_strip_up_to_tau(monkeypatch):
    payload = _biology_payload(monkeypatch)
    spec = _render(payload, _biology_settings())
    for row in spec["vconcat"]:
        diagram = row["hconcat"][1]
        (area,) = _marks(diagram, "area")
        assert area["encoding"]["y2"]["field"] == "diagonal"
        assert area["encoding"]["x"]["field"] == "death_scalar"
        assert area["encoding"]["y"]["field"] == "birth_scalar"
        table = spec["datasets"][area["data"]["name"]]
        assert len(table) == 3  # a trapezoid clipped to the diagonal's extent
        high = max(point["diagonal"] for point in table)
        for point in table:  # the strip is persistence = birth - death <= tau
            assert point["diagonal"] == point["death_scalar"]
            assert point["birth_scalar"] - point["diagonal"] == pytest.approx(
                min(0.72, high - point["diagonal"])
            )
        assert not _marks(diagram, "line")  # no edge line, translucent grey only
        assert area["mark"]["opacity"] == pytest.approx(0.3)
        label = next(m for m in _marks(diagram, "text"))
        assert spec["datasets"][label["data"]["name"]][0]["label"] == (
            "band: persistence ≤ threshold"
        )


def test_biology_marks_survivors_and_absorbed_peaks_on_the_diagram(monkeypatch):
    payload = _biology_payload(monkeypatch)
    spec = _render(payload, _biology_settings())

    def marker_peaks(diagram):
        found = {}
        for mark in _marks(diagram, "point"):
            color = mark["mark"].get("color")
            if color in {"#111111", "#222222", "#a50f15"}:
                rows = spec["datasets"][mark["data"]["name"]]
                found[(color, mark["mark"]["filled"])] = sorted(
                    row["birth_label"] for row in rows
                )
        return found

    top, bottom = (row["hconcat"][1] for row in spec["vconcat"])
    # Every marker is filled (ADR 0012); colour is the outcome or the annotation:
    # the stop-codon peak f is a solid crimson triangle.
    assert marker_peaks(top) == {
        ("#222222", True): ["b", "d"],
        ("#a50f15", True): ["f"],
    }
    assert marker_peaks(bottom) == {("#a50f15", True): ["f"]}
    # The diagram key sits inside the top diagram only, one entry per line.
    assert "vconcat" not in top and "vconcat" not in bottom
    assert _diagram_legend_labels(top) == ["peak", "Stop codon containing peaks"]
    assert _diagram_legend_labels(bottom) == []
    for diagram in (top, bottom):
        stop = next(
            mark
            for mark in _marks(diagram, "point")
            if mark["mark"].get("color") == "#a50f15"
        )
        assert stop["mark"]["shape"] == "triangle-up"
        # Solid by contract (ADR 0012): opacity is omitted or 1.0, never translucent.
        assert stop["mark"].get("opacity", 1.0) == 1.0
        assert spec["datasets"][stop["data"]["name"]][0]["birth_label"] == "f"
        # Crosses are tree-only: none on the diagram.
        assert not [
            mark
            for mark in _marks(diagram, "point")
            if mark["mark"].get("shape") == "M-1,-1L1,1M-1,1L1,-1"
        ]
    # The tree keeps the stop cross in both rows, crimson and solid.
    crosses = [
        mark
        for row in spec["vconcat"]
        if "hconcat" in row
        for mark in _marks(row["hconcat"][0], "point")
        if mark["mark"].get("shape") == "M-1,-1L1,1M-1,1L1,-1"
    ]
    assert len(crosses) == 2
    for cross in crosses:
        assert cross["mark"]["filled"] is False  # a cross is a stroke
        assert cross["mark"].get("opacity", 1.0) == 0.7
        assert cross["mark"]["color"] == "#a50f15"


def test_biology_wild_type_peak_is_a_solid_filled_red_triangle(
    monkeypatch,
):
    """WT identity on a peak still renders a valid diagram of solid filled triangles.

    Under bootstrap-biology outcome colouring the stubbed path does not expose a
    separate red WT triangle layer; ADR 0012 requires peaks stay filled (not hollow).
    """
    payload = _biology_payload(monkeypatch)
    nominal = payload["bootstrap_tree"]
    nominal.wild_type_marker_table = nominal.wild_type_marker_table.with_columns(
        pl.lit("b").alias("dna_sequence")
    )
    spec = _render(payload, _biology_settings())
    diagram = next(row["hconcat"][1] for row in spec["vconcat"] if "hconcat" in row)
    filled_triangles = [
        mark
        for mark in _marks(diagram, "point")
        if mark["mark"].get("shape", "triangle-up") in {"triangle-up", "triangle"}
        and mark["mark"].get("filled", True) is True
    ]
    assert len(filled_triangles) >= 1
    for mark in filled_triangles:
        assert mark["mark"].get("opacity", 1.0) == 1.0
        assert mark["mark"].get("filled") is not False


def test_biology_draws_only_absorbed_branches_in_the_outcome_colour(monkeypatch):
    payload = _biology_payload(monkeypatch)
    spec = _render(payload, _biology_settings())
    top, bottom = (row["hconcat"][0] for row in spec["vconcat"])
    rules = [
        item
        for item in _walk(top)
        if item.get("mark", {}).get("type") == "rule"
        and any(
            tip.get("field") == "peak_sequence"
            for tip in item["encoding"].get("tooltip", [])
        )
    ]
    assert {item["mark"]["color"] for item in rules} == {"#222222"}  # absorbed only
    branches = {
        row["peak_sequence"]
        for rows in spec["datasets"].values()
        for row in rows
        if "peak_sequence" in row and "category" in row
    }
    assert branches == {"b", "d"}  # f survives and stays light grey
    # The smoothed tree is all light grey: no outcome overlay.
    assert "peak_sequence" not in str(bottom)


def _tree_marks(node):
    """The marks of a tree panel in draw order (the first drawn is at the bottom)."""
    return [item["mark"] for item in _walk(node) if isinstance(item.get("mark"), dict)]


def _first(marks, wanted):
    return next(index for index, mark in enumerate(marks) if wanted(mark))


def test_biology_draws_the_stop_crosses_over_every_branch_overlay(monkeypatch):
    """The absorbed-branch rules (``peak_branch_layer``) must not hide the crosses."""
    payload = _biology_payload(monkeypatch)
    spec = _render(payload, _biology_settings())
    for row in spec["vconcat"]:
        marks = _tree_marks(row["hconcat"][0])
        cross = _first(marks, lambda m: m.get("shape") == "M-1,-1L1,1M-1,1L1,-1")
        star = _first(
            marks, lambda m: m.get("type") == "point" and m.get("color") == "#d62728"
        )
        branches = [i for i, m in enumerate(marks) if m.get("type") in {"rule", "line"}]
        assert branches
        assert max(branches) < cross < star
    # The nominal row is the one with a branch overlay: its rules are the absorbed
    # peaks' terminal branches, drawn in the outcome colour, below the cross.
    marks = _tree_marks(spec["vconcat"][0]["hconcat"][0])
    absorbed = [
        i
        for i, m in enumerate(marks)
        if m.get("type") == "rule" and m.get("color") == "#222222"
    ]
    cross = _first(marks, lambda m: m.get("shape") == "M-1,-1L1,1M-1,1L1,-1")
    assert absorbed
    assert max(absorbed) < cross


def test_biology_x_axis_title_is_on_the_last_row_only(monkeypatch):
    payload = _biology_payload(monkeypatch)
    spec = _render(payload, _biology_settings())
    titled = [
        any(
            node.get("title") == "Fitness at death" for node in _walk(row["hconcat"][1])
        )
        for row in spec["vconcat"]
    ]
    assert titled == [False, True]


def test_biology_row_titles_come_from_the_figure_config(monkeypatch):
    payload = _biology_payload(monkeypatch)
    settings = _biology_settings()
    settings["figures"]["figures"][BIOLOGY_ID].update(
        band_row_title="Band · {peaks} of {input_peaks} · {absorbed}",
        band_row_subtitle=["threshold = {threshold}"],
        legend_labels={"survives": "up {count}", "absorbed": "down {count}"},
    )
    spec = _render(payload, settings)
    survive = SENTINEL["bootstrap.candidate.after_band_rule.peak_count"]
    absorbed = SENTINEL["bootstrap.candidate.total_absorbed_count"]
    nominal = SENTINEL["bootstrap.candidate.nominal.peak_count"]
    assert spec["vconcat"][1]["title"]["text"] == (
        f"Band · {survive} of {nominal} · {absorbed}"
    )
    assert spec["vconcat"][1]["title"]["subtitle"] == [
        f"threshold = {SENTINEL['bootstrap.persistence_threshold.value']}"
    ]
    assert [
        label for label, _, _ in outcome_key(spec["vconcat"][0]["hconcat"][0])[1]
    ] == [f"up {survive}", f"down {absorbed}"]


def _count_edges(tree):
    """Return the colour-encoded branch layers of a tree panel's spec."""
    return [
        item
        for item in _walk(tree)
        if isinstance(item.get("encoding"), dict)
        and item["encoding"].get("color", {}).get("field") == "vertex_count"
        and "scale" in item["encoding"]["color"]
    ]


def _band_tree(spec):
    return spec["vconcat"][1]["hconcat"][0]


def _count_labels(spec, node):
    rows = []
    for item in _walk(node):
        mark = item.get("mark")
        if (
            isinstance(mark, dict)
            and mark.get("type") == "text"
            and mark.get("color") == "#737373"
        ):
            rows.append((mark, spec["datasets"][item["data"]["name"]]))
    return rows


def test_biology_colours_only_the_band_tree_by_branch_vertex_count(monkeypatch):
    from dubanevics2026topological.figures import candidate_tree_biology as shared
    from dubanevics2026topological.figures._charts import study_biology_tree_chart

    payload = _biology_payload(monkeypatch)
    # b holds two absorbed members on top of its own peak: three vertices.
    payload["bootstrap_band_view"].assignments = _band_assignments({"b": 2})
    payload["bootstrap_band_view"].branch_table = _saved_band_branch_table(
        payload["bootstrap_band_view"].plot_data,
        payload["bootstrap_band_view"].assignments,
    )
    # The payload stubs the tree chart; draw the real one where a count is given
    # (the band row's layout is real, the nominal row's is a stand-in).
    stub = shared.study_biology_tree_chart
    monkeypatch.setattr(
        shared,
        "study_biology_tree_chart",
        lambda *a, **k: (study_biology_tree_chart if k.get("count_domain") else stub)(
            *a, **k
        ),
    )
    spec = _render(payload, _biology_settings())
    top, bottom = (row["hconcat"][0] for row in spec["vconcat"])
    # The nominal tree keeps its light grey and outcome colours: no count scale.
    assert not _count_edges(top)
    # The band tree encodes the count on the shared log black opacity ramp; count one is the faintest.
    (edges,) = _count_edges(bottom)
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
        for item in _walk(bottom)
        if isinstance(item.get("data"), dict)
        for row in item["data"].get("values", [])
        if "vertex_count" in row and "source_label" in row
    }
    assert by_branch == {("s", "b"): 3, ("s", "d"): 1, ("s", "f"): 1}
    # Only the band tree carries the horizontal count colour bar.
    assert "Branch vertex count" in str(bottom) and "Branch vertex count" not in str(
        top
    )
    # Bootstrap outcome stays on the diagram markers, not on the branches: the
    # outcome colours never reach the band tree.
    assert "#111111" not in str(bottom) and "#222222" not in str(bottom)


def test_biology_labels_small_band_tips_under_the_crosses_and_the_star(monkeypatch):
    payload = _biology_payload(monkeypatch)
    payload["bootstrap_band_view"].assignments = _band_assignments({"b": 2, "f": 1})
    payload["bootstrap_band_view"].branch_table = _saved_band_branch_table(
        payload["bootstrap_band_view"].plot_data,
        payload["bootstrap_band_view"].assignments,
    )
    settings = _biology_settings()
    rendering = settings["figures"]["rendering"]
    rendering.update(branch_count_label_mode="tips", branch_count_label_max=2)
    spec = _render(payload, settings)
    bottom = _band_tree(spec)
    ((mark, rows),) = _count_labels(spec, bottom)
    assert mark["fontSize"] == 13  # the body font less two
    # d (1 vertex) and f (2, a stop-codon peak) are at most the maximum; b (3) is not.
    assert sorted((row["dna_sequence"], row["label"]) for row in rows) == [
        ("d", "1"),
        ("f", "2"),
    ]
    assert not _count_labels(spec, spec["vconcat"][0]["hconcat"][0])  # nominal row
    # Draw order: branches, then labels, then the stop-codon cross, then the star.
    marks = [
        item["mark"] for item in _walk(bottom) if isinstance(item.get("mark"), dict)
    ]
    label = next(
        i
        for i, m in enumerate(marks)
        if m.get("type") == "text" and m.get("color") == "#737373"
    )
    cross = next(
        i for i, m in enumerate(marks) if m.get("shape") == "M-1,-1L1,1M-1,1L1,-1"
    )
    star = next(
        i
        for i, m in enumerate(marks)
        if m.get("type") == "point" and m.get("color") == "#d62728"
    )
    branches = [i for i, m in enumerate(marks) if m.get("type") in {"rule", "line"}]
    assert max(branches) < label < cross < star
    # Above the maximum nothing is labelled in the tips mode.
    rendering["branch_count_label_max"] = 0
    unlabelled = _render(payload, settings)
    assert not _count_labels(unlabelled, _band_tree(unlabelled))


def test_biology_labels_every_band_branch_in_the_all_mode(monkeypatch):
    payload = _biology_payload(monkeypatch)
    payload["bootstrap_band_view"].assignments = _band_assignments({"b": 2, "f": 1})
    payload["bootstrap_band_view"].branch_table = _saved_band_branch_table(
        payload["bootstrap_band_view"].plot_data,
        payload["bootstrap_band_view"].assignments,
    )
    settings = _biology_settings()
    # The maximum is moot in the all mode: no branch is left out for its count.
    settings["figures"]["rendering"].update(
        branch_count_label_mode="all", branch_count_label_max=0
    )
    spec = _render(payload, settings)
    ((_, rows),) = _count_labels(spec, _band_tree(spec))
    assert sorted((row["dna_sequence"], row["label"]) for row in rows) == [
        ("b", "3"),
        ("d", "1"),
        ("f", "2"),
    ]
    assert not _count_labels(spec, spec["vconcat"][0]["hconcat"][0])  # nominal row


def test_biology_turns_the_labels_in_the_rotate_variant(monkeypatch):
    payload = _biology_payload(monkeypatch)
    settings = _biology_settings()
    settings["figures"]["rendering"].update(branch_count_label_mode="all")
    nudged = _render(payload, settings)
    ((nudge_mark, nudge_rows),) = _count_labels(nudged, _band_tree(nudged))
    assert "angle" not in nudge_mark
    settings["figures"]["rendering"].update(branch_count_label_collision="rotate")
    rotated = _render(payload, settings)
    ((mark, rows),) = _count_labels(rotated, _band_tree(rotated))
    assert mark["angle"] == 270
    # The same branches carry the labels in both variants.
    assert sorted(row["dna_sequence"] for row in rows) == sorted(
        row["dna_sequence"] for row in nudge_rows
    )


def test_outcome_key_entries_use_the_palette_and_skip_absent_states():
    from dubanevics2026topological.figures.candidate_tree_biology import (
        outcome_legend_entries,
    )
    from dubanevics2026topological.figures.palette import (
        ELDER_PROTECTED_ORANGE,
        EMPHASIS_BLACK,
        OUTCOME_COLORS,
        PEAK_GREY,
    )

    order = ("kept", "absorbed", "elder_protected")
    labels = {name: name for name in order}
    # Shared encoding (ADR 0012), all filled: kept mid-grey, absorbed black (the
    # emphasis), elder-protected orange; a state with no branch has no entry.
    assert outcome_legend_entries(
        order, labels, OUTCOME_COLORS, {"kept": 2, "absorbed": 1, "elder_protected": 3}
    ) == [
        ("kept", PEAK_GREY, True),
        ("absorbed", EMPHASIS_BLACK, True),
        ("elder_protected", ELDER_PROTECTED_ORANGE, True),
    ]
    assert outcome_legend_entries(
        order, labels, OUTCOME_COLORS, {"kept": 2, "absorbed": 0}
    ) == [("kept", PEAK_GREY, True)]
    assert outcome_legend_entries(order, labels, OUTCOME_COLORS, {}) == []


def test_biology_key_lists_only_the_band_states_in_the_data(monkeypatch):
    from dubanevics2026topological.analysis import biological_annotations

    payload = _biology_payload(monkeypatch)
    cached = payload["bootstrap_smoothed"]
    # Every nominal peak clears the threshold: nothing is absorbed, so the key has
    # one entry.
    cached.hills = cached.hills.with_columns(
        pl.lit(True).alias("clears_band"), pl.lit("kept").alias("band_outcome")
    )
    cached.summary["band_simplification"]["peaks_after"] = 3
    payload[
        "bootstrap_band_view"
    ].pairs = biological_annotations._build_biological_annotations(None).pairs.rename(
        {"birth_sequence": "birth_label", "death_sequence": "death_label"}
    )
    spec = _render(payload, _biology_settings())
    _, entries = outcome_key(spec["vconcat"][0]["hconcat"][0])
    survive = SENTINEL["bootstrap.candidate.after_band_rule.peak_count"]
    assert entries == [
        (f"survives: persistence > threshold ({survive})", "#111111", True)
    ]


def _assert_key_at_bottom_right(tree, labels, width=420.0, height=200.0):
    """The key hugs the bottom-right corner of the tree panel (inset 8, rows 21 apart)."""
    from dubanevics2026topological.figures.candidate_tree_biology import (
        estimate_text_width,
    )

    layout = outcome_key_layout(tree)
    assert layout["last_y"] == pytest.approx(height - 8 - 21 / 2)
    longest = max(estimate_text_width(label, 15) for label in labels)
    assert layout["text_x"] + estimate_text_width(labels[0], 15) <= width - 8
    assert layout["glyph_x"] == pytest.approx(width - 8 - longest - 14)


def test_biology_key_moves_to_the_corner_the_tree_leaves_free(monkeypatch):
    # The toy peaks stand at the right, tall in the top half: the top-right corner is
    # crossed, the bottom-right is the first corner the key can take.
    spec = _render(_biology_payload(monkeypatch), _biology_settings())
    tree = spec["vconcat"][0]["hconcat"][0]
    _, entries = outcome_key(tree)
    _assert_key_at_bottom_right(tree, [label for label, _, _ in entries])
    assert outcome_key_layout(tree)["first_y"] > 100  # not the pinned top-right


def test_outcome_legend_corner_respects_rendering():
    from dubanevics2026topological.figures.candidate_tree_biology import (
        OutcomeKey,
        outcome_legend_corner,
    )

    key = OutcomeKey([("survives", "#737373", True)], "Peak")
    segments = pl.DataFrame(
        {"x_start": [0.0], "x_end": [0.0], "y_start": [0.0], "y_end": [1.0]}
    )
    stops = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float64, "y": pl.Float64})
    corner = outcome_legend_corner(
        key,
        {"outcome_legend_corner": "bottom-left"},
        size=(420.0, 200.0),
        scalar_domain=(-1.0, 1.0),
        segments=segments,
        stop_points=stops,
        wild_type_source=None,
    )
    assert corner == "bottom-left"
    with pytest.raises(ValueError, match="Unknown outcome_legend_corner 'centre'"):
        outcome_legend_corner(
            key,
            {"outcome_legend_corner": "centre"},
            size=(420.0, 200.0),
            scalar_domain=(-1.0, 1.0),
            segments=segments,
            stop_points=stops,
            wild_type_source=None,
        )


def test_candidate_tree_outcome_key_at_bottom_left_intersects_zero_segments_and_markers(
    monkeypatch,
):
    from dubanevics2026topological.figures.candidate_tree_biology import (
        OutcomeKey,
        outcome_corner_counts,
    )

    payload = _biology_payload(monkeypatch)
    candidate = payload["landscapes"]["candidate_clamped"]
    nominal = candidate.trees["split"]
    segments = nominal.plot_data.edge_segment_table
    stop_points = nominal.plot_data.node_table.filter(
        nominal.plot_data.node_table["has_stop_codon"]
    ).select(["x", "y"])
    wt = nominal.wild_type_marker_table

    key = OutcomeKey(
        [
            ("survives: persistence > threshold (12)", "#737373", True),
            ("absorbed: persistence ≤ threshold (65)", "#111111", True),
        ],
        "Peak",
    )
    rendering = {
        "outcome_legend_inset": 8,
        "outcome_legend_line_height": 21,
        "outcome_legend_marker_size": 70,
        "outcome_legend_clearance": 6,
    }
    counts = outcome_corner_counts(
        key,
        rendering,
        size=(420, 200),
        scalar_domain=candidate.scalar_domain,
        segments=segments,
        stop_points=stop_points,
        wild_type_source=wt,
    )
    strict_count, padded_count = counts["bottom-left"]
    assert strict_count == 0
    assert padded_count == 0


def test_biology_refuses_a_band_summary_that_disagrees_with_the_view(monkeypatch):
    payload = _biology_payload(monkeypatch)
    payload["bootstrap_smoothed"].summary["band_simplification"]["peaks_after"] = 2
    with pytest.raises(RuntimeError, match="stale.*just bootstrap-smoothed-tree"):
        _render(payload, _biology_settings())


def test_biology_refuses_hills_that_are_not_the_nominal_peaks(monkeypatch):
    payload = _biology_payload(monkeypatch)
    cached = payload["bootstrap_smoothed"]
    cached.hills = cached.hills.with_columns(
        pl.when(pl.col("peak_sequence") == "d")
        .then(pl.lit("z"))
        .otherwise(pl.col("peak_sequence"))
        .alias("peak_sequence")
    )
    with pytest.raises(RuntimeError, match="stale.*just bootstrap-smoothed-tree"):
        _render(payload, _biology_settings())


def test_the_shared_assembler_refuses_no_rows():
    from dubanevics2026topological.figures.candidate_tree_biology import (
        assemble_biology_chart,
    )

    with pytest.raises(ValueError, match="at least one row"):
        assemble_biology_chart(
            SimpleNamespace(scalar_domain=(0, 1)),
            {},
            {},
            stop_rendering={},
            biology_rendering={},
            rows=(),
        )


# -- build wiring --------------------------------------------------------------


class _Chart:
    def save(self, path, **kwargs):
        Path(path).write_text("svg")


def test_build_passes_the_view_and_tracks_the_checkpoint(tmp_path, monkeypatch):
    figure_id = BIOLOGY_ID
    cached, tree = _cached(monkeypatch)
    source = tmp_path / "src/dubanevics2026topological/figures"
    write_renderer_sources(source, "# renderer")
    settings = {
        "analysis": {"bootstrap_smoothed_tree": dict(CONFIG)},
        "figures": {
            "variants": ["vanilla", "annotated"],
            "formats": ["svg"],
            "figures": {figure_id: {"tree": {"width": 420}}},
        },
    }
    payloads = []
    fake = SimpleNamespace(
        FIGURE_IDS=(figure_id,),
        render=lambda payload, name, config, **kwargs: (
            payloads.append(payload) or _Chart()
        ),
    )
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    install_fake_figures(monkeypatch, fake)
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.clamped_cache",
        SimpleNamespace(
            load_cached_clamped_comparison=lambda: {
                "landscapes": {"candidate_clamped": SimpleNamespace(trees={"split": 1})}
            }
        ),
    )
    monkeypatch.setattr(
        module, "load_cached_bootstrap_smoothed_tree", lambda *_: cached
    )
    monkeypatch.setattr(
        candidate_tree_contrast_module, "build_candidate_tree", lambda s: (tree, None)
    )
    monkeypatch.setattr(build, "_comparison_analysis_fingerprint", lambda: "cached")
    monkeypatch.setattr(
        build, "active_bootstrap_band_identity", lambda checkpoint: "a" * 64, raising=False
    )
    monkeypatch.setattr(
        build, "selection_source", lambda settings: {"rule": "fdr_fixed_cutoff"}
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs.load_saved_figure_inputs",
        lambda context, selection_identity, include_band, **kwargs: _inputs(cached, tree)[
            "saved_figure_inputs"
        ],
    )
    # The registry-currency check is exercised in test_figure_quantities.py.
    monkeypatch.setattr(build, "require_current_registry", lambda settings: None)
    monkeypatch.setattr(build, "_digest_file", lambda path: "source")
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "output/figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "output/figures/.records")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")

    build.build_figures((figure_id,), variants=("annotated",))
    build.build_figures((figure_id,), variants=("annotated",))
    assert len(payloads) == 2
    assert payloads[0]["bootstrap_smoothed"] is cached
    assert payloads[0]["bootstrap_tree"] is tree
    assert "bootstrap_view" not in payloads[0]
    assert set(payloads[0]["bootstrap_band_view"].pairs["birth_label"]) == {"f"}
    record = (tmp_path / f"output/figures/.records/{figure_id}.svg.json").read_text()
    assert "bootstrap_checkpoint_provenance" in record
    assert "bootstrap_smoothed_source" in record

    cached.provenance = {"fingerprint": {"generation": 2}, "artifacts": {}}
    build.build_figures((figure_id,), variants=("annotated",))
    assert len(payloads) == 3
    section = settings["analysis"]["bootstrap_smoothed_tree"]
    section["n_jobs"] = CONFIG["n_jobs"] + 1  # scheduling only: no rebuild
    build.build_figures((figure_id,), variants=("annotated",))
    assert len(payloads) == 4
    section["n_replicates"] = CONFIG["n_replicates"] + 1
    build.build_figures((figure_id,), variants=("annotated",))
    assert len(payloads) == 5


def test_rotated_label_variant_is_a_separate_output_and_leaves_the_config_alone(
    tmp_path, monkeypatch
):
    figure_id = BIOLOGY_ID
    cached, tree = _cached(monkeypatch)
    source = tmp_path / "src/dubanevics2026topological/figures"
    write_renderer_sources(source, "# renderer")
    settings = {
        "analysis": {"bootstrap_smoothed_tree": dict(CONFIG)},
        "figures": {
            "variants": ["vanilla", "annotated"],
            "formats": ["svg"],
            "rendering": {"branch_count_label_collision": "nudge"},
            "figures": {figure_id: {"tree": {"width": 420}}},
        },
    }
    seen = []
    fake = SimpleNamespace(
        FIGURE_IDS=(figure_id,),
        render=lambda payload, name, config, **kwargs: (
            seen.append(config["figures"]["rendering"]["branch_count_label_collision"])
            or _Chart()
        ),
    )
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    install_fake_figures(monkeypatch, fake)
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.clamped_cache",
        SimpleNamespace(
            load_cached_clamped_comparison=lambda: {
                "landscapes": {"candidate_clamped": SimpleNamespace(trees={"split": 1})}
            }
        ),
    )
    monkeypatch.setattr(
        module, "load_cached_bootstrap_smoothed_tree", lambda *_: cached
    )
    monkeypatch.setattr(
        candidate_tree_contrast_module, "build_candidate_tree", lambda s: (tree, None)
    )
    monkeypatch.setattr(build, "_comparison_analysis_fingerprint", lambda: "cached")
    # The registry-currency check is exercised in test_figure_quantities.py.
    monkeypatch.setattr(build, "require_current_registry", lambda settings: None)
    monkeypatch.setattr(build, "_digest_file", lambda path: "source")
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "output/figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "output/figures/.records")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")

    outputs = build.build_figures(
        (figure_id,), variants=("annotated",), label_collision="rotate"
    )
    assert [path.name for path in outputs] == [f"{figure_id}_rotated.svg"]
    assert seen == ["rotate"]
    # The config object is untouched and the canonical output is not written.
    assert settings["figures"]["rendering"]["branch_count_label_collision"] == "nudge"
    assert not (tmp_path / f"output/figures/canonical/{figure_id}.svg").exists()
    assert (tmp_path / f"output/figures/.records/{figure_id}_rotated.svg.json").is_file()
    assert not (tmp_path / f"output/figures/.records/{figure_id}.svg.json").exists()
    # A second run is current; the canonical build renders with the config value.
    build.build_figures((figure_id,), variants=("annotated",), label_collision="rotate")
    assert seen == ["rotate", "rotate"]
    build.build_figures((figure_id,), variants=("annotated",))
    assert seen == ["rotate", "rotate", "nudge"]
    assert (tmp_path / f"output/figures/canonical/{figure_id}.svg").is_file()

    with pytest.raises(ValueError, match="band-tree figures"):
        build.build_figures(("tree_comparison",), label_collision="rotate")
    with pytest.raises(ValueError, match="label_collision must be"):
        build.build_figures((figure_id,), label_collision="nudge")


def test_rotated_label_variant_selects_the_band_tree_figures_by_default(monkeypatch):
    requested: list[str] = []

    class Stop(Exception):
        pass

    def load(names, settings=None):
        requested.extend(names)
        raise Stop

    monkeypatch.setattr(
        build,
        "load_settings",
        lambda: {"figures": {"variants": ["annotated"], "formats": ["svg"]}},
    )
    install_fake_registry(monkeypatch)
    monkeypatch.setattr(build, "load_inputs", load)
    with pytest.raises(Stop):
        build.build_figures(None, label_collision="rotate")
    assert "bootstrap_smoothed" in requested
    assert "tree_comparison" not in requested


def test_label_collision_flag_is_for_the_figures_command_only(capsys):
    with pytest.raises(SystemExit):
        build.main(["analysis", "--label-collision", "rotate"])
    assert "figures command only" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        build.main(["figures", "--label-collision", "nudge"])


def test_default_selection_needs_the_bootstrap_cache_for_manuscript_sources(monkeypatch):
    requested: list[str] = []

    class Stop(Exception):
        pass

    def load(names, settings=None):
        requested.extend(names)
        raise Stop

    monkeypatch.setattr(
        build,
        "load_settings",
        lambda: {"figures": {"variants": ["annotated"], "formats": ["svg"]}},
    )
    install_fake_registry(monkeypatch)
    monkeypatch.setattr(build, "load_inputs", load)
    with pytest.raises(Stop):
        build.build_figures(None)
    # Manuscript-source bootstrap figures are defaults, so the default selection
    # needs the bootstrap cache.
    assert "bootstrap_smoothed" in requested
    requested.clear()
    with pytest.raises(Stop):
        build.build_figures([BIOLOGY_ID])
    assert "bootstrap_smoothed" in requested


# -- recipes and inventory -----------------------------------------------------


def _recipe(name):
    lines = (Path(build.ROOT) / "justfile").read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith((f"{name}:", f"{name} ")))
    body = lines[lines.index(header) + 1]
    return header, body


def test_recipe_depends_on_the_stage_and_stays_outside_all_and_figures():
    recipe, figure_id = "bootstrap-biology-figure", BIOLOGY_ID
    header, body = _recipe(recipe)
    assert header.split(":")[1].split() == [
        "bootstrap-smoothed-tree",
        "comparison-analysis",
        "manuscript-quantities",
    ]
    assert f"--figure {figure_id} --variant annotated" in body
    for aggregate in ("all", "figures"):
        _, dependencies = next(
            line
            for line in (Path(build.ROOT) / "justfile").read_text().splitlines()
            if line.startswith(f"{aggregate}:")
        ).split(":")
        assert recipe not in dependencies.split()


def test_inventory_lists_the_figures_their_recipes_and_the_threshold():
    text = (Path(build.ROOT) / "docs/figure-inventory.md").read_text(encoding="utf-8")
    assert f"| `{BIOLOGY_ID}` |" in text
    assert f"`{BIOLOGY_ID}.svg`, `.png`" in text
    assert "just bootstrap-biology-figure" in text
    assert "bootstrapped persistence threshold" in text
    assert "τ = 2c" not in text
    # The retired core-rule figure is named in the retirement notes only: no table
    # row and no live recipe.
    assert "| `papkou2023_bootstrap_core_tree` |" not in text
    assert "just bootstrap-core-tree-figure" not in text
    assert "core threshold = {core_threshold}" not in text
    assert "comembership" not in text
    assert "just comembership-heatmap-figure" not in text


def test_biology_subtitle_under_fdr_variant_vs_default(monkeypatch):
    from dubanevics2026topological.threshold.functional_threshold import (
        DECLARATION_KEY,
        FDR_FIXED_CUTOFF,
    )

    payload = _biology_payload(monkeypatch)

    # 1. Default (no FDR): subtitle has "simultaneous confidence band"
    spec_default = _render(payload, _biology_settings())
    sub_default = spec_default["vconcat"][0]["title"]["subtitle"]
    assert "85.7% simultaneous confidence band (parametric bootstrap)" in sub_default[1]
    assert "nominal" not in sub_default[1]

    # 2. Under fdr_fixed_cutoff: subtitle has "nominal {percent}% simultaneous band..."
    fdr_settings = _biology_settings()
    fdr_settings.setdefault("analysis", {}).setdefault("clamped", {})[
        DECLARATION_KEY
    ] = {"rule": FDR_FIXED_CUTOFF, "decimals": 6, "cutoff": "median_reflected"}
    spec_fdr = _render(payload, fdr_settings)
    sub_fdr = spec_fdr["vconcat"][0]["title"]["subtitle"]
    assert (
        "nominal 85.7% simultaneous band (fixed-graph plug-in parametric bootstrap; coverage not independently established)"
        in sub_fdr[1]
    )
    assert "confidence band" not in sub_fdr[1]
