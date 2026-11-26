"""Bootstrap global stability figure (ADR 0008, C4)."""

from __future__ import annotations

import dataclasses
import textwrap
import tomllib
from pathlib import Path

import polars as pl
import pytest
from _fake_figures import install_fake_registry, write_renderer_sources
from _figure_labels import SENTINEL, label_quantities, with_labels
from test_bootstrap_smoothed_tree import (
    BAND_C,
    CONFIG,
    DNA,
    _cooc,
    _growth,
    _patch_seam,
    _toy,
)

from dubanevics2026topological import build
from dubanevics2026topological.analysis import (
    candidate_tree_contrast as candidate_tree_contrast_module,
)
from dubanevics2026topological.figures import registry
from dubanevics2026topological.figures.bootstrap_global_stability import (
    bottleneck_histogram,
    diagram_points,
    wrap_balanced,
    wrap_phrases,
)
from dubanevics2026topological.figures.candidate_tree_biology import estimate_text_width
from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
    build_bootstrap_smoothed_tree,
)

STABILITY_ID = "bootstrap_global_stability"
ROOT = Path(build.ROOT)
# Toy hills: b = {b, a}, d = {d}, f = {f, g}; saddle e and root c are on no hill.
# The core {b, a} lies on hill b; {e} is a core on no hill (a critical node).
CORES = [["b", "a"], ["e"]]
FREQUENCIES = {
    ("b", "b"): 0.9,  # the peak is its own diagonal
    ("a", "b"): 0.6,  # a member with its peak (stage vertex order a < b)
    ("a", "a"): 0.7,
    ("d", "d"): 0.5,
    ("f", "f"): 0.8,  # g has no pair with f
    ("g", "g"): 0.95,
    ("e", "e"): 0.97,
}


def _pairs(values: dict[tuple[str, str], float]) -> pl.DataFrame:
    """Return a stage-style pair table: vertex_a <= vertex_b by network order."""
    rows = sorted(
        (min(DNA.index(a), DNA.index(b)), max(DNA.index(a), DNA.index(b)), value)
        for (a, b), value in values.items()
    )
    return pl.DataFrame(
        {
            "vertex_a": [row[0] for row in rows],
            "vertex_b": [row[1] for row in rows],
            "frequency": [row[2] for row in rows],
        },
        schema={
            "vertex_a": pl.Int64,
            "vertex_b": pl.Int64,
            "frequency": pl.Float64,
        },
    )


def _draws(n, wasserstein, peaks, bottleneck=None):
    """Return a result-style ``draw_stats`` frame (the stage adds the condition)."""
    return pl.DataFrame(
        {
            "draw": list(range(n)),
            "leaf_count": peaks if peaks is not None else [3] * n,
            "wasserstein_1": wasserstein if wasserstein is not None else [0.0] * n,
            "bottleneck": bottleneck if bottleneck is not None else [BAND_C] * n,
            "reproduced_pair_fraction": [1.0] * n,
        }
    )


def _cached(
    monkeypatch,
    cores=CORES,
    frequencies=FREQUENCIES,
    *,
    real_w=None,
    flat_w=None,
    real_peaks=None,
    flat_peaks=None,
    real_bottleneck=None,
    flat_bottleneck=None,
):
    """Return a real checkpoint result on the toy tree, plus the toy tree."""
    n = CONFIG["n_replicates"]
    real = dataclasses.replace(
        _cooc(cores),
        same_leaf_frequency=_pairs(frequencies),
        draw_stats=_draws(n, real_w, real_peaks, real_bottleneck),
    )
    flat = dataclasses.replace(
        _cooc(cores), draw_stats=_draws(n, flat_w, flat_peaks, flat_bottleneck)
    )
    _patch_seam(monkeypatch, real, flat)
    prepared, tree = _toy()
    cached = build_bootstrap_smoothed_tree(
        prepared, tree, _growth(), CONFIG, log=lambda message: None
    )
    cached.provenance = {"fingerprint": {"generation": 1}, "artifacts": {}}
    return cached, tree


def _settings(stability=None):
    figures = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))
    figures["figures"][STABILITY_ID].update(stability or {})
    return {"figures": figures, "analysis": {"bootstrap_smoothed_tree": dict(CONFIG)}}


# -- registry ------------------------------------------------------------------


def test_figure_is_a_default_annotated_and_configured():
    spec = registry.get_spec(STABILITY_ID)
    assert spec.default is True  # manuscript source
    assert spec.variants == {"annotated"}
    assert spec.inputs == {"bootstrap_smoothed", "manuscript_quantities"}
    assert spec.config_table == STABILITY_ID
    assert STABILITY_ID in [
        item.figure_id for item in registry.FIGURE_SPECS if item.default
    ]
    tables = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))
    assert STABILITY_ID in tables["figures"]
    with pytest.raises(ValueError, match="canonical annotated"):
        registry.render_payload(spec, {}, {"figures": {}}, "vanilla")


def test_stability_needs_only_the_draws_and_the_registry():
    spec = registry.get_spec(STABILITY_ID)
    assert spec.inputs == {"bootstrap_smoothed", "manuscript_quantities"}
    assert registry.renderer_files(spec)[1] == "candidate_tree_biology.py"
    assert registry.renderer_file(spec) == "bootstrap_global_stability.py"


def test_the_bootstrap_tree_is_checked_against_the_checkpoint(monkeypatch):
    cached, tree = _cached(monkeypatch)
    calls = []
    monkeypatch.setattr(
        candidate_tree_contrast_module,
        "build_candidate_tree",
        lambda settings: calls.append(settings) or (tree, None),
    )
    quantities = label_quantities()
    inputs = {
        "bootstrap_smoothed": cached,
        "manuscript_quantities": quantities,
        "saved_figure_inputs": {"mode": "legacy_mixture"},
    }
    assert registry.bootstrap_tree(inputs, {"s": 1}) is tree
    stability = registry.compose_payload(
        registry.get_spec(STABILITY_ID), inputs, {"s": 1}
    )
    assert stability == {
        "bootstrap_smoothed": cached,
        "manuscript_quantities": quantities,
    }
    assert len(calls) == 1  # the stability figure never builds a tree

    cached.hills = cached.hills.with_columns(
        pl.when(pl.col("peak_sequence") == "b")
        .then(pl.lit("x"))
        .otherwise(pl.col("peak_sequence"))
        .alias("peak_sequence")
    )
    with pytest.raises(ValueError):
        registry.bootstrap_tree(
            {
                "bootstrap_smoothed": cached,
                "saved_figure_inputs": {"mode": "legacy_mixture"},
            },
            {"s": 1},
        )


# -- stability figure ----------------------------------------------------------
# Toy diagram (ordinary pairs plus the global pair closed at the root c = -0.2):
# b (0.5, -0.2) persistence 0.7, d (0.4, -0.15) persistence 0.55, f (0.6, -0.2)
# persistence 0.8. With CORES, b is kept, d absorbed and f elder-protected. Every
# real draw has bottleneck 0.3, so c = 0.3, 2c = 0.6: b and f are significant.
BOTTLENECK = 0.3
# Flat draws sit far above the real ones (0.80 .. 0.99 against 0.3), as on Papkou.
FLAT_BOTTLENECK = [0.8 + 0.01 * i for i in range(20)]
OUTCOME_COLORS = {
    "absorbed": "#101010",
    "elder_protected": "#202020",
    "kept": "#303030",
}


def _stats(real, flat):
    """Return a ``draw_stats`` table from per-condition bottleneck lists."""
    frames = []
    for condition, values in (("real", real), ("flat", flat)):
        frames.append(
            pl.DataFrame(
                {
                    "draw": list(range(len(values))),
                    "condition": [condition] * len(values),
                    "peak_count": [3] * len(values),
                    "wasserstein_1": [0.0] * len(values),
                    "bottleneck": values,
                    "reproduced_pair_fraction": [1.0] * len(values),
                }
            )
        )
    return pl.concat(frames)


def test_bottleneck_histogram_bins_one_condition_on_its_own_range():
    stats = _stats(real=[0.1, 0.1, 0.2, 0.4], flat=[500.0, 600.0])
    table = bottleneck_histogram(stats, 3)
    assert table.columns == ["bin_start", "bin_end", "count"]
    assert table["count"].sum() == 4  # the flat draws do not enter the real bins
    assert table["bin_start"].min() == 0.1 and table["bin_end"].max() == 0.4
    assert (table["bin_end"] - table["bin_start"]).round(9).n_unique() == 1
    flat = bottleneck_histogram(stats, 3, "flat")
    assert flat["count"].sum() == 2
    assert flat["bin_start"].min() == 500.0 and flat["bin_end"].max() == 600.0
    flat_only = stats.filter(pl.col("condition") == "flat")
    with pytest.raises(ValueError, match="no 'real' draws"):
        bottleneck_histogram(flat_only, 3)
    real_only = stats.filter(pl.col("condition") == "real")
    with pytest.raises(ValueError, match="no 'flat' draws"):
        bottleneck_histogram(real_only, 3, "flat")
    with pytest.raises(ValueError, match="non-finite"):
        bottleneck_histogram(_stats(real=[float("nan")], flat=[1.0]), 3)
    with pytest.raises(ValueError, match="flat bottleneck has non-finite"):
        bottleneck_histogram(_stats(real=[1.0], flat=[float("inf")]), 3, "flat")


def test_bottleneck_histogram_can_share_bin_edges_across_conditions():
    stats = _stats(real=[0.1, 0.2], flat=[0.8, 0.9])
    real = bottleneck_histogram(stats, 4, "real", (0.1, 0.9))
    flat = bottleneck_histogram(stats, 4, "flat", (0.1, 0.9))
    assert real["bin_start"].min() == pytest.approx(0.1)
    assert flat["bin_end"].max() == pytest.approx(0.9)
    edges = {round(v, 9) for v in (*real["bin_start"], *flat["bin_start"])}
    assert edges <= {0.1, 0.3, 0.5, 0.7}


def _stability(monkeypatch, settings=None, *, bottleneck=None, tamper=None):
    n = CONFIG["n_replicates"]
    cached, _ = _cached(
        monkeypatch,
        real_w=[0.1] * n,
        flat_w=[0.9] * n,
        real_bottleneck=bottleneck if bottleneck is not None else [BOTTLENECK] * n,
        flat_bottleneck=FLAT_BOTTLENECK,
    )
    if tamper:
        tamper(cached)
    settings = settings or _settings()
    return cached, registry.render_payload(
        registry.get_spec(STABILITY_ID),
        with_labels({"bootstrap_smoothed": cached}),
        settings,
        "annotated",
    ).to_dict()


def _stability_chart(monkeypatch):
    n = CONFIG["n_replicates"]
    cached, _ = _cached(
        monkeypatch,
        real_w=[0.1] * n,
        flat_w=[0.9] * n,
        real_bottleneck=[BOTTLENECK] * n,
        flat_bottleneck=FLAT_BOTTLENECK,
    )
    return registry.render_payload(
        registry.get_spec(STABILITY_ID),
        with_labels({"bootstrap_smoothed": cached}),
        _settings(),
        "annotated",
    )


def _panels(spec):
    """Return the overlaid histogram and the diagram of the one stacked figure."""
    histogram, diagram = spec["vconcat"]
    return histogram, diagram


def _layer_rows(spec, panel, index):
    layer = panel["layer"][index]
    data = layer.get("data") or panel["data"]
    return spec["datasets"][data["name"]]


def test_diagram_points_carry_outcome_and_significance(monkeypatch):
    cached, _ = _stability(monkeypatch)
    points = diagram_points(cached.hills)
    rows = {r["peak_sequence"]: r for r in points.iter_rows(named=True)}
    assert {k: v["band_outcome"] for k, v in rows.items()} == {
        "b": "kept",
        "d": "absorbed",
        "f": "kept",
    }
    assert {k: v["status"] for k, v in rows.items()} == {
        "b": "significant",
        "d": "within the band",
        "f": "significant",
    }
    assert (rows["d"]["birth"], rows["d"]["death"]) == (0.4, -0.15)
    assert rows["d"]["persistence"] == pytest.approx(0.55)
    broken = cached.hills.with_columns(pl.lit("merged").alias("band_outcome"))
    with pytest.raises(ValueError, match="Unknown hill outcomes"):
        diagram_points(broken)


def test_stability_figure_stacks_one_histogram_over_the_diagram(monkeypatch):
    cached, spec = _stability(monkeypatch)
    histogram, diagram = _panels(spec)
    n = CONFIG["n_replicates"]
    bars = histogram["layer"][0]
    rows = _layer_rows(spec, histogram, 0)
    assert bars["mark"]["type"] == "rect"
    assert {r["condition"] for r in rows} == {
        "Nominal landscape",
        "Flat landscape",
    }
    for label in ("Nominal landscape", "Flat landscape"):
        assert sum(r["count"] for r in rows if r["condition"] == label) == n
    assert bars["encoding"]["y2"] == {"datum": 0}
    # x axes are not shared between the histogram and the diagram.
    assert spec["resolve"]["scale"]["x"] == "independent"
    assert (histogram["width"], histogram["height"]) == (250, 100)
    assert (diagram["width"], diagram["height"]) == (250, 250)
    assert [layer["mark"]["type"] for layer in diagram["layer"]] == [
        "area",
        "line",
        "point",
        "point",
    ]
    assert cached.summary["diagnostics"]["global_verdict"]["c"] == BOTTLENECK


def test_stability_histograms_share_bins_and_take_two_greys_from_the_config(
    monkeypatch,
):
    tables = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))
    config = tables["figures"][STABILITY_ID]
    _, spec = _stability(monkeypatch)
    histogram, _ = _panels(spec)
    color = histogram["layer"][0]["encoding"]["color"]
    assert color["scale"]["range"] == [config["real_color"], config["flat_color"]]
    assert config["real_color"] != config["flat_color"]
    assert color["scale"]["domain"] == ["Nominal landscape", "Flat landscape"]
    rows = _layer_rows(spec, histogram, 0)
    widths = {round(r["bin_end"] - r["bin_start"], 9) for r in rows}
    assert len(widths) == 1  # one bin width for both conditions
    settings = _settings(stability={"real_color": "#010203", "flat_color": "#040506"})
    _, spec = _stability(monkeypatch, settings)
    histogram, _ = _panels(spec)
    assert histogram["layer"][0]["encoding"]["color"]["scale"]["range"] == [
        "#010203",
        "#040506",
    ]


def test_stability_legend_has_two_entries_centred_at_the_top_inside_the_histogram(
    monkeypatch,
):
    _, spec = _stability(monkeypatch)
    histogram, _ = _panels(spec)
    color = histogram["layer"][0]["encoding"]["color"]
    legend = color["legend"]
    # Only the two conditions are listed, and the rule at c has no entry.
    assert color["scale"]["domain"] == ["Nominal landscape", "Flat landscape"]
    assert "color" not in histogram["layer"][1]["encoding"]
    assert legend["orient"] == "none"  # placed by hand, inside the panel
    assert legend["direction"] == "vertical"
    width, height = histogram["width"], histogram["height"]
    # Centred: the legend's estimated width leaves equal margins on both sides.
    font = legend["labelFontSize"]
    text = estimate_text_width("Nominal landscape", font)
    legend_width = 10 + 4 + text  # symbol, label offset, longest label
    assert legend["legendX"] == pytest.approx((width - legend_width) / 2)
    assert 0 <= legend["legendX"] and legend["legendX"] + legend_width <= width
    # At the top of the panel, inside it (not above it).
    assert 0 <= legend["legendY"] < height / 4


def test_stability_diagram_x_axis_is_titled_depth_saddle_or_root(monkeypatch):
    _, spec = _stability(monkeypatch)
    _, diagram = _panels(spec)
    assert diagram["layer"][3]["encoding"]["x"]["title"] == "Depth, saddle or root"
    assert diagram["layer"][3]["encoding"]["y"]["title"] == (
        "Birth (peak fitness, WT = 0)"
    )


def test_stability_marks_the_radius_with_a_rule_on_the_histogram(monkeypatch):
    _, spec = _stability(monkeypatch)
    histogram, _ = _panels(spec)
    assert [layer["mark"]["type"] for layer in histogram["layer"]] == [
        "rect",
        "rule",
    ]
    (row,) = _layer_rows(spec, histogram, 1)
    assert row["bin_start"] == BOTTLENECK


def test_stability_has_no_grid_and_no_spines(monkeypatch):
    _, spec = _stability(monkeypatch)
    assert spec["config"]["axis"]["grid"] is False
    assert spec["config"]["axis"]["domain"] is False
    assert spec["config"]["view"]["strokeWidth"] == 0


def test_stability_diagram_has_death_on_x_birth_on_y_and_a_cone_above_the_diagonal(
    monkeypatch,
):
    _, spec = _stability(monkeypatch)
    _, diagram = _panels(spec)
    band = _layer_rows(spec, diagram, 0)
    two_c = 2 * BOTTLENECK
    assert len(band) == 3  # a trapezoid clipped to the diagonal's extent
    high = max(row["death"] for row in band)
    for row in band:  # the cone lies above the diagonal, 2c tall in birth
        assert row["diagonal"] == pytest.approx(row["death"])
        assert row["band_edge"] == pytest.approx(min(row["death"] + two_c, high))
    assert diagram["layer"][0]["encoding"]["y2"] == {"field": "diagonal"}
    assert diagram["layer"][0]["mark"]["opacity"] < 1  # translucent
    # Both axes share one fitness domain so the diagonal stays at 45 degrees.
    x = diagram["layer"][2]["encoding"]["x"]["scale"]["domain"]
    y = diagram["layer"][2]["encoding"]["y"]["scale"]["domain"]
    assert x == y
    assert x[0] < -0.2 and x[1] > 0.6


def test_stability_peaks_are_filled_triangles_grey_above_the_threshold_black_below(
    monkeypatch,
):
    _, spec = _stability(monkeypatch)
    _, diagram = _panels(spec)
    open_layer, filled_layer = diagram["layer"][2], diagram["layer"][3]
    # No hollow markers: both states are filled and differ by colour.
    assert open_layer["mark"]["filled"] is True
    assert filled_layer["mark"]["filled"] is True
    assert open_layer["mark"]["color"] == "#111111"  # inside the band: black
    assert filled_layer["mark"]["color"] == "#737373"  # above 2c: mid-grey, ADR 0012
    for layer in (open_layer, filled_layer):
        assert layer["mark"]["shape"] == "triangle-up"
        assert layer["mark"]["opacity"] == 1.0  # fully opaque (point_opacity)
        assert "color" not in layer["encoding"]  # no outcome colours, no legend
    inside = {r["peak_sequence"] for r in _layer_rows(spec, diagram, 2)}
    outside = {r["peak_sequence"] for r in _layer_rows(spec, diagram, 3)}
    assert (inside, outside) == ({"d"}, {"b", "f"})
    rows = {r["peak_sequence"]: r for r in _layer_rows(spec, diagram, 3)}
    assert (rows["b"]["birth"], rows["b"]["death"]) == (0.5, -0.2)
    assert filled_layer["encoding"]["x"]["field"] == "death"
    assert filled_layer["encoding"]["y"]["field"] == "birth"
    assert "legend" not in str(spec["vconcat"][1])


def test_stability_subtitle_prints_the_registry_numbers_in_words(monkeypatch):
    cached, spec = _stability(monkeypatch)
    verdict = cached.summary["diagnostics"]["global_verdict"]
    # The toy tables hold radius 0.3, threshold 0.6, 2 of 3 peaks and 20 draws; the
    # subtitle prints the registry's (sentinel) display text and none of those.
    assert verdict["significant_point_count"] == 2
    text = " ".join(spec["title"]["subtitle"])
    sanity = "bootstrap.global_stability.sanity_check."
    assert (
        f"Radius {SENTINEL['bootstrap.persistence_threshold.radius']} (band "
        f"quantile {SENTINEL['bootstrap.band_quantile']} of the bottleneck "
        f"distance over {SENTINEL['bootstrap.draws']} nominal-landscape draws); "
        "bootstrapped persistence threshold "
        f"{SENTINEL['bootstrap.persistence_threshold.value']}, twice the radius"
    ) in text
    assert (
        f"{SENTINEL['bootstrap.candidate.after_band_rule.peak_count']} of "
        f"{SENTINEL['bootstrap.candidate.nominal.peak_count']} nominal peaks have "
        "persistence above the threshold (grey); black peaks lie in the shaded "
        "cone. Relative Wasserstein-1 median "
        f"{SENTINEL['bootstrap.global_stability.relative_wasserstein_1_median']}"
    ) in text
    below = SENTINEL[
        sanity + "nominal_landscape_draws_below_flat_fifth_percentile_percent"
    ]
    assert (
        f"Sanity check, not the verdict: {below}% of nominal-landscape draws lie "
        "below the flat-landscape fifth percentile "
        f"({SENTINEL[sanity + 'flat_bottleneck_distance_fifth_percentile']})"
    ) in text
    assert (
        f"medians {SENTINEL[sanity + 'nominal_bottleneck_distance_median']} "
        f"(nominal) and {SENTINEL[sanity + 'flat_bottleneck_distance_median']} (flat)"
    ) in text
    assert all(len(line) <= 52 for line in spec["title"]["subtitle"])  # wrapped
    for symbol in ("2c", "τ", "c = "):
        assert symbol not in text
    assert "extended" not in str(spec).lower()
    assert "flat" not in " ".join(spec["title"]["text"]).lower()


def test_stability_asks_the_registry_for_exactly_its_subtitle_numbers(monkeypatch):
    n = CONFIG["n_replicates"]
    cached, _ = _cached(
        monkeypatch,
        real_w=[0.1] * n,
        flat_w=[0.9] * n,
        real_bottleneck=[BOTTLENECK] * n,
        flat_bottleneck=FLAT_BOTTLENECK,
    )
    payload = with_labels({"bootstrap_smoothed": cached})
    registry.render_payload(
        registry.get_spec(STABILITY_ID), payload, _settings(), "annotated"
    )
    assert set(payload["manuscript_quantities"].requested_keys) == {
        "bootstrap.persistence_threshold.radius",
        "bootstrap.band_quantile",
        "bootstrap.draws",
        "bootstrap.persistence_threshold.value",
        "bootstrap.candidate.after_band_rule.peak_count",
        "bootstrap.candidate.nominal.peak_count",
        "bootstrap.global_stability.relative_wasserstein_1_median",
        (
            "bootstrap.global_stability.sanity_check."
            "nominal_landscape_draws_below_flat_fifth_percentile_percent"
        ),
        "bootstrap.global_stability.sanity_check.flat_bottleneck_distance_fifth_percentile",
        "bootstrap.global_stability.sanity_check.nominal_bottleneck_distance_median",
        "bootstrap.global_stability.sanity_check.flat_bottleneck_distance_median",
    }


def test_wrap_balances_lines_so_no_lone_word_is_left_on_a_line():
    # Greedy wrapping at 44 leaves "0.108" and "(flat)" alone on their own lines.
    median = (
        "42 of 169 nominal peaks have persistence above 2c (grey); black peaks "
        "lie in the shaded cone. Relative Wasserstein-1 median 0.108"
    )
    sanity = (
        "draws lie below the flat 5th percentile (0.803); medians 0.158 (nominal) "
        "and 0.907 (flat)"
    )
    for text in (median, sanity):
        greedy = textwrap.wrap(text, width=44)
        assert len(greedy[-1].split()) == 1  # the orphan this fixes
        lines = wrap_balanced(text, 44)
        assert len(lines) == len(greedy)
        assert all(len(line) <= 44 for line in lines)
        assert all(len(line.split()) > 1 for line in lines)
        assert " ".join(lines) == text
    assert wrap_balanced("short", 44) == ["short"]
    # A hyphenated word is not split across lines.
    hyphenated = wrap_balanced("x " * 14 + "Wasserstein-1 median", 30)
    assert "Wasserstein-1" in " ".join(hyphenated)
    assert not any(line.endswith("-") for line in hyphenated)


def test_wrap_phrases_wraps_each_phrase_on_its_own_line_start():
    lines = wrap_phrases(["aaa bbb ccc ddd eee fff", "ggg"], 12)
    assert lines[-1] == "ggg"  # the second phrase never joins the first one's line
    assert all(len(line) <= 12 for line in lines)


def test_stability_subtitle_has_no_orphan_lines(monkeypatch):
    _, spec = _stability(monkeypatch)
    subtitle = spec["title"]["subtitle"]
    assert subtitle
    # No line is a lone word or number such as "0.108" or "(flat)".
    assert all(len(line.split()) > 1 for line in subtitle)
    assert max(len(line) for line in subtitle) <= 52


def test_stability_refuses_a_summary_that_disagrees_with_the_tables(monkeypatch):
    def move_c(cached):
        cached.summary["diagnostics"]["global_verdict"]["c"] = 0.1

    def drop_count(cached):
        cached.summary["diagnostics"]["global_verdict"]["significant_point_count"] = 1

    def move_flat_p5(cached):
        sanity = cached.summary["diagnostics"]["global_verdict"]["sanity_check"]
        sanity["flat_bottleneck_p5"] = 0.1

    def move_share(cached):
        sanity = cached.summary["diagnostics"]["global_verdict"]["sanity_check"]
        sanity["real_share_below_flat_bottleneck_p5"] = 0.5

    for tamper in (move_c, drop_count, move_flat_p5, move_share):
        with pytest.raises(RuntimeError, match="stale.*just bootstrap-smoothed-tree"):
            _stability(monkeypatch, tamper=tamper)


def test_stability_needs_the_flat_draws_for_panel_a(monkeypatch):
    cached, _ = _stability(monkeypatch)
    cached.draw_stats = cached.draw_stats.filter(pl.col("condition") == "real")
    with pytest.raises(ValueError, match="no 'flat' draws"):
        registry.render_payload(
            registry.get_spec(STABILITY_ID),
            with_labels({"bootstrap_smoothed": cached}),
            _settings(),
            "annotated",
        )


# -- rendering through the build -----------------------------------------------


def test_build_renders_a_file_through_the_registry(tmp_path, monkeypatch):
    figure_id = STABILITY_ID
    cached, tree = _cached(monkeypatch)
    settings = _settings()
    settings["figures"].update(variants=["annotated"], formats=["svg"], rendering={})
    source = tmp_path / "src/dubanevics2026topological/figures"
    write_renderer_sources(source, "# renderer")
    loaded: list[list[str]] = []
    inputs = {"bootstrap_smoothed": cached}
    install_fake_registry(monkeypatch)
    inputs["manuscript_quantities"] = label_quantities()
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(
        build,
        "load_inputs",
        lambda names, settings=None: loaded.append(sorted(names)) or inputs,
    )
    monkeypatch.setattr(
        candidate_tree_contrast_module, "build_candidate_tree", lambda s: (tree, None)
    )
    monkeypatch.setattr(build, "_comparison_analysis_fingerprint", lambda: "cached")
    # The registry-currency check is exercised in test_figure_quantities.py.
    monkeypatch.setattr(build, "require_current_registry", lambda settings: None)
    monkeypatch.setattr(build, "_analysis_fingerprint", lambda: pytest.fail("cache"))
    monkeypatch.setattr(build, "_digest_file", lambda path: "source")
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "output/figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "output/figures/.records")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")

    (target,) = build.build_figures((figure_id,), variants=("annotated",))
    assert target == tmp_path / "output/figures/canonical" / f"{figure_id}.svg"
    assert target.stat().st_size > 0
    assert target.read_text(encoding="utf-8").lstrip().startswith(("<svg", "<?xml"))
    assert loaded == [["bootstrap_smoothed", "manuscript_quantities"]]
    record = (tmp_path / f"output/figures/.records/{figure_id}.svg.json").read_text()
    assert "bootstrap_checkpoint_provenance" in record
    assert "bootstrap_smoothed_source" in record


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
        build.build_figures([STABILITY_ID])
    assert "bootstrap_smoothed" in requested


# -- recipes and inventory -----------------------------------------------------


def _recipe(name):
    lines = (ROOT / "justfile").read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith(f"{name}:"))
    return header, lines[lines.index(header) + 1], lines[lines.index(header) - 2 :]


def test_recipe_depends_on_the_stage_and_stays_outside_all_and_figures():
    recipe, figure_id = "bootstrap-global-stability-figure", STABILITY_ID
    header, body, context = _recipe(recipe)
    assert header.split(":")[1].split() == [
        "bootstrap-smoothed-tree",
        "comparison-analysis",
        "manuscript-quantities",
    ]
    assert f"--figure {figure_id} --variant annotated" in body
    assert "Not part of `all` or `figures`" in "\n".join(context)
    for aggregate in ("all", "figures"):
        _, dependencies = next(
            line
            for line in (ROOT / "justfile").read_text().splitlines()
            if line.startswith(f"{aggregate}:")
        ).split(":")
        assert recipe not in dependencies.split()


def test_inventory_lists_the_figure_and_its_recipe():
    text = (ROOT / "docs/figure-inventory.md").read_text(encoding="utf-8")
    assert f"| `{STABILITY_ID}` |" in text
    assert f"`{STABILITY_ID}.svg`, `.png`" in text
    assert "just bootstrap-global-stability-figure" in text
    assert "comembership" not in text


def test_presentation_keys_live_in_the_figure_config_tables():
    tables = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))["figures"]
    stability = tables[STABILITY_ID]
    assert {
        "real_color",
        "flat_color",
        "real_title",
        "flat_title",
        "histogram_width",
        "histogram_height",
        "bottleneck_bins",
        "rule_color",
        "rule_dash",
        "band_color",
        "band_opacity",
        "diagonal_color",
        "birth_axis_title",
        "death_axis_title",
    } <= set(stability)
    # Sizes come from the shared keys; the teal/outcome legend is gone.
    assert not {"panel_width", "panel_height", "legend_title", "point_size"} & set(
        stability
    )
    # Only the bottleneck is binned now; the Wasserstein-1 panels are gone.
    assert not {"wasserstein_bins", "wasserstein_title", "bar_color"} & set(stability)
    # Outcome colours live in the bootstrap biology table and are not repeated.
    assert not {"absorbed_color", "kept_color", "elder_protected_color"} & set(
        stability
    )


def test_stability_title_under_fdr_variant(monkeypatch):
    from dubanevics2026topological.threshold.functional_threshold import (
        DECLARATION_KEY,
        FDR_FIXED_CUTOFF,
    )

    # 1. Default settings: title retains "confidence band"
    _, default_spec = _stability(monkeypatch)
    default_title = " ".join(default_spec["title"]["text"])
    assert (
        "Nominal persistence diagram against the bootstrap confidence band"
        in default_title
    )

    # 2. FDR variant: "confidence band" replaced by "band"
    fdr_settings = _settings()
    fdr_settings.setdefault("analysis", {}).setdefault("clamped", {})[
        DECLARATION_KEY
    ] = {"rule": FDR_FIXED_CUTOFF, "decimals": 6, "cutoff": "median_reflected"}
    _, fdr_spec = _stability(monkeypatch, settings=fdr_settings)
    fdr_title = " ".join(fdr_spec["title"]["text"])
    assert "Nominal persistence diagram against the bootstrap band" in fdr_title
    assert "confidence" not in fdr_title
