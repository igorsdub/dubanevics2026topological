"""Peak-pair figures: registry, config, matrices, labels and the scatter."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from peak_pair_toy import toy_tree

from dubanevics2026topological import build
from dubanevics2026topological.analysis.peak_pair_distances import (
    peak_pair_distances,
    peak_pair_mantel_table,
)
from dubanevics2026topological.figures import registry
from dubanevics2026topological.figures.peak_pair_figures import (
    COMPARISONS,
    DIRECT_VIA_ID,
    MATRICES_ID,
    SCATTER_ID,
    build_peak_pair_direct_via_saddle_figure,
    build_peak_pair_hamming_fitness_scatter_figure,
    build_peak_pair_matrices_figure,
    diagonal_summary,
    direct_via_cells,
    fitness_ticks,
    jittered,
    legend_ticks,
    matrix_cells,
    padded_range,
    peak_labels,
    peak_order,
    scatter_points,
)

IDS = (MATRICES_ID, SCATTER_ID, DIRECT_VIA_ID)
FIGURES = tomllib.loads((Path(build.ROOT) / "config/figures.toml").read_text("utf-8"))


@pytest.fixture
def payload() -> dict:
    pairs = peak_pair_distances(*toy_tree())
    return {
        "pairs": pairs,
        "mantel": peak_pair_mantel_table(pairs, permutations=19, seed=1),
    }


def _settings(figure_id: str, **overrides) -> dict:
    table = {**FIGURES["figures"][figure_id], **overrides}
    return {
        "figures": {"rendering": FIGURES["rendering"], "figures": {figure_id: table}}
    }


@pytest.mark.parametrize("figure_id", IDS)
def test_registry_and_config(figure_id: str) -> None:
    spec = registry.get_spec(figure_id)
    assert spec.default is False
    assert spec.inputs == frozenset({"peak_pair_distances"})
    assert spec.variants == frozenset({"vanilla"})
    assert spec.config_table == figure_id
    assert figure_id in FIGURES["figures"]
    assert registry.renderer_file(spec) == "peak_pair_figures.py"
    assert figure_id in build.NON_CANDIDATE_FIGURE_IDS


def test_config_values() -> None:
    matrices = FIGURES["figures"][MATRICES_ID]
    assert matrices["label_mode"] == "amino_acid"
    scatter = FIGURES["figures"][SCATTER_ID]
    assert scatter["jitter"] == 0.15
    assert isinstance(scatter["jitter_seed"], int)


def test_peak_order_and_labels_toy(payload) -> None:
    order = peak_order(payload["pairs"])
    assert order == ["GCTAAA", "GCCAAA", "GCTGAA", "TCTGAC"]
    assert peak_labels(order, "dna") == order
    assert peak_labels(order, "amino_acid") == [
        "AK",
        "AK",
        "AE",
        "SD",
    ]  # collision kept
    assert peak_labels(order, "both")[0] == "GCTAAA AK"
    with pytest.raises(ValueError, match="label_mode"):
        peak_labels(order, "nope")


def test_stop_codons_use_existing_characters() -> None:
    assert peak_labels(["TAATAGTGA"], "amino_acid") == ["%^&"]


def test_matrix_cells_triangles_and_diagonal(payload) -> None:
    cells = matrix_cells(
        payload["pairs"], "nucleotide_direct_hamming", "nucleotide_via_saddle_hamming"
    )
    assert cells.height == 4 + 2 * 6
    assert cells.filter(pl.col("triangle") == "diagonal")["value"].to_list() == [0] * 4
    lookup = {(r["row"], r["column"]): r["value"] for r in cells.iter_rows(named=True)}
    # pair (A, D) = positions (0, 3): direct Hamming 3 below, via saddle 5 above.
    assert lookup[(3, 0)] == 3
    assert lookup[(0, 3)] == 5


@pytest.mark.parametrize("mode", ["amino_acid", "dna", "both"])
def test_matrices_render_in_each_label_mode(payload, mode: str) -> None:
    chart = build_peak_pair_matrices_figure(
        payload, _settings(MATRICES_ID, label_mode=mode)
    )
    spec = chart.to_dict()
    assert len(spec["hconcat"]) == 3
    text = json.dumps(spec)
    first = peak_labels(peak_order(payload["pairs"]), mode)[0]
    assert first in text


def test_scatter_points_and_jitter(payload) -> None:
    pairs = payload["pairs"]
    points = scatter_points(pairs, "direct", "nucleotide", 0.15, 7)
    assert points.height == pairs.height
    assert (points["hamming"] - points["hamming_exact"]).abs().max() <= 0.15
    assert (points["hamming"] != points["hamming_exact"]).any()
    again = scatter_points(pairs, "direct", "nucleotide", 0.15, 7)
    assert points["hamming"].to_list() == again["hamming"].to_list()
    other = scatter_points(pairs, "direct", "nucleotide", 0.15, 8)
    assert points["hamming"].to_list() != other["hamming"].to_list()
    assert np.array_equal(jittered([1, 2], 0.15, 3), jittered([1, 2], 0.15, 3))
    diff = scatter_points(pairs, "via_minus_direct", "nucleotide", 0.0, 7)
    assert diff["hamming_exact"].to_list() == [1, 2, 2, 2, 2, 4]


PAIR_ROWS = 6


def _finite_mantel(payload: dict) -> pl.DataFrame:
    """The Mantel table with finite, distinct statistics per (comparison, alphabet)."""
    mantel = payload["mantel"]
    count = mantel.height
    return mantel.with_columns(
        pl.Series("spearman_rho", [0.1 * (i + 1) - 0.05 for i in range(count)]),
        pl.Series("mantel_p", [0.0125 * (i + 1) for i in range(count)]),
    )


ALPHABETS = ("nucleotide", "amino_acid")
NOUNS = {"nucleotide": "nucleotides", "amino_acid": "amino acids"}


def _grid(chart) -> list[list[dict]]:
    """The 2 x 3 panel specs, row by row."""
    return [row["hconcat"] for row in chart.to_dict()["vconcat"]]


def _hamming_columns(pairs: pl.DataFrame, comparison: str, alphabet: str) -> list:
    direct = pairs[f"{alphabet}_direct_hamming"]
    via = pairs[f"{alphabet}_via_saddle_hamming"]
    return {"direct": direct, "via_saddle": via, "via_minus_direct": via - direct}[
        comparison
    ].to_list()


def _fitness_column(pairs: pl.DataFrame, comparison: str) -> list:
    direct = pairs["direct_fitness_difference"]
    via = pairs["via_saddle_fitness_distance"]
    return {"direct": direct, "via_saddle": via, "via_minus_direct": via - direct}[
        comparison
    ].to_list()


def test_scatter_is_two_rows_by_three_columns_of_filled_markers(payload) -> None:
    chart = build_peak_pair_hamming_fitness_scatter_figure(
        payload, _settings(SCATTER_ID)
    )
    grid = _grid(chart)
    assert [len(row) for row in grid] == [3, 3]
    for row in grid:
        for panel in row:
            mark = panel["mark"]
            assert mark["filled"] is True
            assert mark["type"] == "point"
            assert "fill" not in panel["encoding"]
    text = json.dumps(chart.to_dict())
    assert "transparent" not in text


def test_scatter_panels_use_each_rows_alphabet_and_the_right_y(payload) -> None:
    pairs = payload["pairs"]
    chart = build_peak_pair_hamming_fitness_scatter_figure(
        payload, _settings(SCATTER_ID)
    )
    spec = chart.to_dict()
    for row_index, alphabet in enumerate(ALPHABETS):
        for col_index, comparison in enumerate(COMPARISONS):
            panel = _grid(chart)[row_index][col_index]
            data = spec["datasets"][panel["data"]["name"]]
            assert [r["hamming_exact"] for r in data] == _hamming_columns(
                pairs, comparison, alphabet
            )
            assert [r["fitness"] for r in data] == pytest.approx(
                _fitness_column(pairs, comparison)
            )
            assert NOUNS[alphabet] in panel["encoding"]["x"]["title"]
            ticks = panel["encoding"]["x"]["axis"]["values"]
            assert ticks == list(range(min(ticks), max(ticks) + 1))
            assert panel["encoding"]["x"]["axis"]["grid"] is False
            assert panel["encoding"]["y"]["axis"]["grid"] is False
    # The toy's two alphabets differ, or the check proves nothing.
    assert _hamming_columns(pairs, "direct", "nucleotide") != _hamming_columns(
        pairs, "direct", "amino_acid"
    )


def test_scatter_columns_share_their_y_range(payload) -> None:
    chart = build_peak_pair_hamming_fitness_scatter_figure(
        payload, _settings(SCATTER_ID)
    )
    grid = _grid(chart)
    for col in range(3):
        top, bottom = (grid[r][col]["encoding"]["y"]["scale"]["domain"] for r in (0, 1))
        assert top == bottom
        fitness = _fitness_column(payload["pairs"], COMPARISONS[col])
        assert top[0] <= min(fitness) and top[1] >= max(fitness)
        if COMPARISONS[col] != "via_minus_direct":
            # Direct and via columns are anchored at 0 with 5% padding.
            high = max(fitness)
            assert top == pytest.approx([-0.05 * high, 1.05 * high])
    assert grid[0][0]["encoding"]["y"]["title"] == "Direct fitness difference"
    assert grid[0][1]["encoding"]["y"]["title"] == "Via-saddle fitness distance"
    assert grid[0][2]["encoding"]["y"]["title"] == "Via saddle minus direct fitness"
    assert (
        grid[1][0]["encoding"]["x"]["title"] == "Direct Hamming distance (amino acids)"
    )
    assert (
        grid[0][0]["encoding"]["x"]["title"] == "Direct Hamming distance (nucleotides)"
    )


def test_scatter_statistics_are_the_formatted_mantel_values_per_panel(payload) -> None:
    mantel = _finite_mantel(payload)
    chart = build_peak_pair_hamming_fitness_scatter_figure(
        {"pairs": payload["pairs"], "mantel": mantel}, _settings(SCATTER_ID)
    )
    seen = set()
    for row_index, alphabet in enumerate(ALPHABETS):
        for col_index, comparison in enumerate(COMPARISONS):
            row = mantel.filter(
                (pl.col("comparison") == comparison) & (pl.col("alphabet") == alphabet)
            ).row(0, named=True)
            expected = (
                f"Spearman \u03c1 = {row['spearman_rho']:.3f} "
                f"(Mantel p = {row['mantel_p']:.3f})"
            )
            subtitle = _grid(chart)[row_index][col_index]["title"]["subtitle"]
            assert subtitle == expected
            seen.add(subtitle)
    assert len(seen) == 6
    assert (
        _grid(chart)[0][0]["title"]["subtitle"]
        == "Spearman \u03c1 = 0.050 (Mantel p = 0.013)"
    )
    text = json.dumps(chart.to_dict()).lower()
    assert "significan" not in text and "cause" not in text


def test_scatter_small_mantel_p_prints_less_than_0_001(payload) -> None:
    mantel = _finite_mantel(payload).with_columns(
        pl.when(
            (pl.col("comparison") == "direct") & (pl.col("alphabet") == "nucleotide")
        )
        .then(0.0004)
        .otherwise(pl.col("mantel_p"))
        .alias("mantel_p")
    )
    chart = build_peak_pair_hamming_fitness_scatter_figure(
        {"pairs": payload["pairs"], "mantel": mantel}, _settings(SCATTER_ID)
    )
    assert _grid(chart)[0][0]["title"]["subtitle"] == (
        "Spearman \u03c1 = 0.050 (Mantel p < 0.001)"
    )
    assert "p = 0.001" not in json.dumps(chart.to_dict())


def test_scatter_x_domain_clears_the_farthest_jittered_point(payload) -> None:
    pairs = payload["pairs"]
    chart = build_peak_pair_hamming_fitness_scatter_figure(
        payload, _settings(SCATTER_ID)
    )
    amount = FIGURES["figures"][SCATTER_ID].get("jitter", 0.15)
    for row_index, alphabet in enumerate(ALPHABETS):
        for col_index, comparison in enumerate(COMPARISONS):
            values = _hamming_columns(pairs, comparison, alphabet)
            low, high = _grid(chart)[row_index][col_index]["encoding"]["x"]["scale"][
                "domain"
            ]
            assert low < min(values) - amount
            assert high > max(values) + amount


def test_scatter_toy_hamming_literals(payload) -> None:
    pairs = payload["pairs"]
    assert _hamming_columns(pairs, "direct", "amino_acid") == [0, 1, 2, 1, 2, 2]
    assert _hamming_columns(pairs, "via_saddle", "amino_acid") == [0, 3, 4, 3, 4, 4]
    assert _hamming_columns(pairs, "via_minus_direct", "amino_acid") == [
        0,
        2,
        2,
        2,
        2,
        2,
    ]


def test_scatter_statistic_undefined_case(payload) -> None:
    mantel = _finite_mantel(payload).with_columns(
        pl.when(
            (pl.col("comparison") == "direct") & (pl.col("alphabet") == "amino_acid")
        )
        .then(float("nan"))
        .otherwise(pl.col("spearman_rho"))
        .alias("spearman_rho")
    )
    chart = build_peak_pair_hamming_fitness_scatter_figure(
        {"pairs": payload["pairs"], "mantel": mantel}, _settings(SCATTER_ID)
    )
    grid = _grid(chart)
    assert grid[1][0]["title"]["subtitle"] == "Spearman \u03c1 and Mantel p undefined"
    for row_index, col_index in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]:
        assert "undefined" not in grid[row_index][col_index]["title"]["subtitle"]


def test_legend_ticks_are_whole_units_only() -> None:
    assert legend_ticks(3.6) == [0, 1, 2, 3]
    assert legend_ticks(3.0) == [0, 1, 2, 3]
    assert legend_ticks(2.99) == [0, 1, 2]
    assert legend_ticks(0.5) == [0]


def _colour_legends(chart) -> list[dict]:
    found: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            color = node.get("encoding", {}).get("color")
            if color and color.get("type") == "quantitative":
                found.append(color)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(chart.to_dict())
    return found


def test_continuous_colour_bars_span_zero_to_the_data_maximum(payload) -> None:
    chart = build_peak_pair_matrices_figure(payload, _settings(MATRICES_ID))
    legends = _colour_legends(chart)
    assert len(legends) == 2
    pairs = payload["pairs"]
    maxima = [
        float(pairs["via_saddle_fitness_distance"].max()),
        float(
            (
                pairs["via_saddle_fitness_distance"]
                - pairs["direct_fitness_difference"]
            ).max()
        ),
    ]
    fitness_max = float(
        max(
            pairs["direct_fitness_difference"].max(),
            pairs["via_saddle_fitness_distance"].max(),
        )
    )
    for color, high in zip(legends, (fitness_max, maxima[1]), strict=True):
        assert color["scale"]["domain"] == [0.0, high]
        assert color["scale"]["nice"] is False
        values = color["legend"]["values"]
        assert values == list(range(int(high // 1) + 1))
        assert color["legend"]["format"] == ".0f"


def test_matrices_legend_symbol_is_a_square(payload) -> None:
    matrices = build_peak_pair_matrices_figure(payload, _settings(MATRICES_ID))
    assert matrices.to_dict()["config"]["legend"]["symbolType"] == "square"


def test_load_inputs_reads_the_stage_variant(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from dubanevics2026topological.analysis import peak_pair_distances as stage
    from dubanevics2026topological.results import RunContext

    seen: list[object] = []
    monkeypatch.setattr(
        stage,
        "load_cached_peak_pair_distances",
        lambda ctx: seen.append(ctx) or "cached",
    )
    context = RunContext(
        output_root=tmp_path / "out", root=tmp_path, variant_name="vanilla"
    )
    inputs = build.load_inputs(["peak_pair_distances"], context=context)
    assert inputs == {"peak_pair_distances": "cached"}
    (ctx,) = seen
    assert ctx.variant_name == stage.VARIANT
    assert (
        ctx.output_root == (tmp_path / "output" / "variants" / stage.VARIANT).resolve()
    )
    # A context already on the stage variant is passed through unchanged.
    seen.clear()
    own = RunContext(
        output_root=tmp_path / "own", root=tmp_path, variant_name=stage.VARIANT
    )
    build.load_inputs(["peak_pair_distances"], context=own)
    assert seen == [own]
    assert isinstance(own, RunContext) and not isinstance(own, SimpleNamespace)


@pytest.mark.parametrize("figure_id", IDS)
def test_render_record_depends_on_the_stage_record(
    figure_id: str, payload, monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    provenance = {"stage": "peak-pair-distances", "artifacts": {"pairs": "digest-1"}}
    result = SimpleNamespace(provenance=provenance, **payload)
    output = tmp_path / "output"
    monkeypatch.setattr(build, "OUTPUT", output)
    monkeypatch.setattr(build, "FIGURE_DIR", output / "figures")
    monkeypatch.setattr(build, "RECORD_DIR", output / "figures/.records")
    build.build_figures(
        (figure_id,),
        _inputs_override={"peak_pair_distances": result},
        formats=["svg"],
        titled_twins=False,
    )
    records = list((output / "figures/.records").rglob(f"{figure_id}.svg.json"))
    assert len(records) == 1
    text = records[0].read_text("utf-8")
    assert "peak_pair_distances" in text and "digest-1" in text


def test_missing_column_is_an_error(payload) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_peak_pair_matrices_figure(
            {"pairs": payload["pairs"].drop("peak_1_dna"), "mantel": payload["mantel"]},
            _settings(MATRICES_ID),
        )


# -- direct against via-saddle scatter ----------------------------------------
# Hand-computed from tests/peak_pair_toy.py. Pairs (direct, via):
#   nucleotide  (1,2) (1,3) (3,5) (2,4) (4,6) (2,6)   excess 1 2 2 2 2 4
#   amino acid  (0,0) (1,3) (2,4) (1,3) (2,4) (2,4)   excess 0 2 2 2 2 2
#   fitness     (1,2) (2,4) (2.5,5.5) (1,3) (1.5,4.5) (0.5,3.5)   excess 1 2 3 2 3 3
# Median of six values is the mean of the third and fourth sorted values.


def _direct_via_panels(chart) -> list[dict]:
    return chart.to_dict()["hconcat"]


def test_direct_via_toy_cells_are_the_hand_computed_pairs(payload) -> None:
    pairs = payload["pairs"]
    nucleotide = direct_via_cells(pairs, "nucleotide")
    assert sorted(
        zip(
            nucleotide["direct"].to_list(),
            nucleotide["via_saddle"].to_list(),
            strict=True,
        )
    ) == sorted([(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (2, 6)])
    assert nucleotide["pairs"].sum() == PAIR_ROWS
    assert nucleotide["pairs"].to_list() == [1] * PAIR_ROWS
    amino = direct_via_cells(pairs, "amino_acid")
    counts = {
        (r["direct"], r["via_saddle"]): r["pairs"] for r in amino.iter_rows(named=True)
    }
    # One row per occupied cell; (1,3) holds two pairs and (2,4) three.
    assert counts == {(0, 0): 1, (1, 3): 2, (2, 4): 3}
    assert amino.height == 3 and amino["pairs"].sum() == PAIR_ROWS


def test_direct_via_diagonal_fraction_mean_and_median_excess_on_the_toy(
    payload,
) -> None:
    pairs = payload["pairs"]
    nt = diagonal_summary(
        pairs["nucleotide_direct_hamming"].to_list(),
        pairs["nucleotide_via_saddle_hamming"].to_list(),
    )
    aa = diagonal_summary(
        pairs["amino_acid_direct_hamming"].to_list(),
        pairs["amino_acid_via_saddle_hamming"].to_list(),
    )
    fit = diagonal_summary(
        pairs["direct_fitness_difference"].to_list(),
        pairs["via_saddle_fitness_distance"].to_list(),
    )
    # Excess sums 13, 10, 14 over six pairs (hand-computed above the helpers).
    assert nt == (0.0, pytest.approx(13 / 6), 2.0)
    assert aa == (1 / 6, pytest.approx(10 / 6), 2.0)
    assert fit == (0.0, pytest.approx(14 / 6), 2.5)


def test_median_excess_of_an_even_count_is_the_mean_of_the_two_middle_values() -> None:
    assert diagonal_summary([0, 0, 0, 0], [1, 2, 5, 9]) == (0.0, 4.25, 3.5)
    assert diagonal_summary([0, 0, 0], [1, 2, 9]) == (0.0, 4.0, 2.0)
    assert diagonal_summary([1, 1], [1, 4]) == (0.5, 1.5, 1.5)


def test_fitness_diagonal_is_exact_equality() -> None:
    assert diagonal_summary([1.0], [1.0])[0] == 1.0
    assert diagonal_summary([1.0], [1.0 + 1e-12])[0] == 0.0


def test_direct_via_figure_layout_and_text(payload) -> None:
    chart = build_peak_pair_direct_via_saddle_figure(payload, _settings(DIRECT_VIA_ID))
    panels = _direct_via_panels(chart)
    assert len(panels) == 3
    subtitles = [panel["title"]["subtitle"] for panel in panels]
    assert subtitles == [
        ["Pairs on the diagonal: 0 of 6 (0%)", "Mean / median excess: 2.2 / 2"],
        ["Pairs on the diagonal: 1 of 6 (17%)", "Mean / median excess: 1.7 / 2"],
        ["Pairs on the diagonal: 0 of 6 (0%)", "Mean / median excess: 2.333 / 2.5"],
    ]
    text = json.dumps(chart.to_dict())
    for forbidden in (
        "\u03c1",
        "Spearman",
        "Mantel",
        "p-value",
        "p =",
        "jitter",
        "tolerance",
        "raw fitness units",
        "via saddle minus direct)",
    ):
        assert forbidden not in text
    for title in (
        "Direct Hamming distance (nucleotides)",
        "Via-saddle Hamming distance (nucleotides)",
        "Direct Hamming distance (amino acids)",
        "Via-saddle Hamming distance (amino acids)",
        "Direct fitness difference",
        "Via-saddle fitness distance",
    ):
        assert title in text


def test_direct_via_panels_are_square_dashed_padded_gridless_with_identical_ticks(
    payload,
) -> None:
    chart = build_peak_pair_direct_via_saddle_figure(payload, _settings(DIRECT_VIA_ID))
    expected_max = [6, 4, 5.5]
    for index, (panel, high) in enumerate(
        zip(_direct_via_panels(chart), expected_max, strict=True)
    ):
        assert panel["width"] == panel["height"]
        layers = panel["layer"]
        rules = [layer for layer in layers if layer["mark"]["type"] == "line"]
        assert len(rules) == 1 and rules[0]["mark"]["strokeDash"]
        pad = 0.05 * high
        expected = [pytest.approx(-pad), pytest.approx(high + pad)]
        for layer in layers:
            x, y = layer["encoding"]["x"], layer["encoding"]["y"]
            assert x["scale"]["domain"] == expected
            assert y["scale"]["domain"] == expected
            assert x["axis"]["grid"] is False and y["axis"]["grid"] is False
            assert x["axis"]["values"] == y["axis"]["values"]
        ticks = layers[0]["encoding"]["x"]["axis"]["values"]
        if index < 2:
            assert ticks == list(range(high + 1))
        else:
            assert ticks == [0, 2, 4]
        # The dashed y = x line spans the padded domain.
        data = chart.to_dict()["datasets"][rules[0]["data"]["name"]]
        assert [row["direct"] for row in data] == pytest.approx(
            [-0.05 * high, 1.05 * high]
        )
    config = chart.to_dict()["config"]
    assert config["axis"]["grid"] is False


@pytest.mark.parametrize(
    ("high", "expected"),
    [
        (3.6, [0, 1, 2, 3]),
        (0.37, [0, 0.1, 0.2, 0.3]),
        (10, [0, 2, 4, 6, 8, 10]),
        (0, [0.0]),
    ],
)
def test_fitness_ticks_rule(high: float, expected: list[float]) -> None:
    assert fitness_ticks(high) == pytest.approx(expected)


def test_padded_range_survives_zero_range() -> None:
    low, top = padded_range(0, 0)
    assert low < 0 < top
    low, top = padded_range(0, -1)
    assert low < 0 < top
    assert padded_range(0, 10) == pytest.approx([-0.5, 10.5])


def test_fitness_markers_are_filled_and_transparent(payload) -> None:
    chart = build_peak_pair_direct_via_saddle_figure(payload, _settings(DIRECT_VIA_ID))
    layers = _direct_via_panels(chart)[2]["layer"]
    marks = [layer["mark"] for layer in layers if layer["mark"]["type"] == "point"]
    assert len(marks) == 1
    assert marks[0]["filled"] is True and marks[0]["shape"] == "circle"
    assert marks[0]["opacity"] < 1.0  # filled and semi-transparent, not opaque
    for panel in _direct_via_panels(chart)[:2]:
        assert any(layer["mark"]["type"] == "circle" for layer in panel["layer"])


def test_no_peak_pair_axis_or_colour_bar_title_says_raw_fitness_units(payload) -> None:
    for figure_id, builder in (
        (SCATTER_ID, build_peak_pair_hamming_fitness_scatter_figure),
        (MATRICES_ID, build_peak_pair_matrices_figure),
        (DIRECT_VIA_ID, build_peak_pair_direct_via_saddle_figure),
    ):
        text = json.dumps(builder(payload, _settings(figure_id)).to_dict())
        assert "raw fitness units" not in text


def test_direct_via_hamming_cells_are_fixed_size_and_coloured_by_log_count(
    payload,
) -> None:
    chart = build_peak_pair_direct_via_saddle_figure(payload, _settings(DIRECT_VIA_ID))
    spec = chart.to_dict()
    table = FIGURES["figures"][DIRECT_VIA_ID]
    for panel in _direct_via_panels(chart)[:2]:
        circles = [
            layer for layer in panel["layer"] if layer["mark"]["type"] == "circle"
        ]
        assert len(circles) == 1
        assert circles[0]["mark"]["size"] == table["marker_size"]
        assert "size" not in circles[0]["encoding"]
        colour = circles[0]["encoding"]["color"]
        assert colour["field"] == "pairs" and colour["type"] == "quantitative"
        assert colour["scale"]["type"] == "log"
        # Toy maximum cell count is 3; legend ticks are powers of ten up to it, so [1].
        assert colour["scale"]["domain"] == [1, 3]
        assert colour["legend"]["values"] == [1]
        assert colour["legend"]["format"] == "d"
    assert spec["resolve"]["scale"]["color"] == "shared"
    fitness_circles = [
        layer
        for layer in _direct_via_panels(chart)[2]["layer"]
        if layer["mark"]["type"] == "circle"
    ]
    assert fitness_circles == []
    datasets = list(spec["datasets"].values())
    xs = sorted({row["direct"] for data in datasets for row in data if "pairs" in row})
    assert all(float(x).is_integer() for x in xs)


def test_direct_via_legend_ticks_on_real_like_counts(payload) -> None:
    pairs = pl.concat([payload["pairs"]] * 150)  # largest cell count becomes 450
    chart = build_peak_pair_direct_via_saddle_figure(
        {**payload, "pairs": pairs}, _settings(DIRECT_VIA_ID)
    )
    for panel in _direct_via_panels(chart)[:2]:
        circle = next(l for l in panel["layer"] if l["mark"]["type"] == "circle")
        colour = circle["encoding"]["color"]
        assert colour["scale"]["domain"] == [1, 450]
        assert colour["legend"]["values"] == [1, 10, 100]


def test_direct_via_config_values() -> None:
    table = FIGURES["figures"][DIRECT_VIA_ID]
    for key in ("panel_size", "marker_size", "dash", "point_color", "diagonal_color"):
        assert key in table
    assert "jitter" not in table
