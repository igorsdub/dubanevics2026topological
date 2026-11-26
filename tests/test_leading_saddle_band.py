"""Saddles, landmarks, intervals, trunks, entropy and outputs of the leading saddle band."""

from __future__ import annotations

import json
import math

import polars as pl
import pytest
from _saddle_band import (
    DECLARATION,
    FITNESS,
    JOIN_BELOW,
    SEQUENCES,
    SPLIT_BELOW,
    _pairs,
    _population,
    isolated_graph,
    sequence_at,
)
from topographer.examples.cube import cube_edge_table, cube_node_table
from topographer.graph import Graph, ScalarNetwork
from topographer.topology import ScalarNetworkTopology

from dubanevics2026topological.analysis import leading_saddle_band as band_module
from dubanevics2026topological.analysis.codon_entropy import positional_entropy
from dubanevics2026topological.analysis.component_persistence import (
    _pairs_with_sequence_identities,
    build_threshold_diagnostics,
)
from dubanevics2026topological.analysis.leading_saddle_band import (
    ARTEFACTS,
    compute_leading_saddle_band,
    global_minimum,
    read_leading_saddle_band,
    tree_saddles,
    write_leading_saddle_band,
)


def _compute(pairs, published=-0.5, population=None, graph=None):
    frame = _population() if population is None else population
    return compute_leading_saddle_band(
        pairs,
        frame,
        graph=isolated_graph(frame) if graph is None else graph,
        published_threshold=published,
        candidate_declaration=DECLARATION,
    )


def test_split_leading_and_trailing_saddles_carry_identities() -> None:
    saddles = tree_saddles(JOIN_BELOW)
    assert (saddles.split_leading.sequence, saddles.split_leading.fitness) == (
        sequence_at(0.3),
        0.3,
    )
    assert (saddles.split_trailing.sequence, saddles.split_trailing.fitness) == (
        sequence_at(-0.4),
        -0.4,
    )


def test_join_leading_and_trailing_saddles_carry_identities() -> None:
    saddles = tree_saddles(JOIN_BELOW)
    assert (saddles.join_leading.sequence, saddles.join_leading.fitness) == (
        sequence_at(-0.9),
        -0.9,
    )
    assert (saddles.join_trailing.sequence, saddles.join_trailing.fitness) == (
        sequence_at(-0.6),
        -0.6,
    )
    assert (
        saddles.join_trailing.fitness
        == build_threshold_diagnostics(JOIN_BELOW)["join_tree_trailing_saddle"]
    )


def test_equal_saddles_break_ties_by_sequence() -> None:
    pairs = _pairs([0.2, 0.2, 0.2], [-0.5], split_sequences=["B", "A", "C"])
    assert tree_saddles(pairs).split_leading.sequence == "A"
    assert tree_saddles(pairs).split_trailing.sequence == "A"


def test_missing_tree_pairs_give_none_and_the_stage_refuses() -> None:
    pairs = _pairs([], [-0.5])
    assert tree_saddles(pairs).split_trailing is None
    with pytest.raises(ValueError, match="no ordinary pair"):
        _compute(pairs)


def test_global_minimum_is_the_pair_birth() -> None:
    minimum = global_minimum(JOIN_BELOW)
    assert (minimum.sequence, minimum.fitness) == (SEQUENCES[0], -1.0)


def test_join_trailing_saddle_must_match_the_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        band_module,
        "build_threshold_diagnostics",
        lambda pairs: {"join_tree_trailing_saddle": -0.123},
    )
    with pytest.raises(ValueError, match="differs from the threshold diagnostics"):
        _compute(JOIN_BELOW)


def test_landmarks_table_values_and_sequences() -> None:
    result = _compute(JOIN_BELOW)
    table = {row["landmark"]: row for row in result.landmarks.iter_rows(named=True)}
    assert result.landmarks["landmark"].to_list() == list(band_module.LANDMARK_KEYS)
    assert table["global_minimum"]["fitness"] == -1.0
    assert table["split_trailing_saddle"]["fitness"] == -0.4
    assert table["join_trailing_saddle"]["fitness"] == -0.6
    assert table["median"]["fitness"] == 0.0
    assert table["split_leading_saddle"]["fitness"] == 0.3
    assert table["join_leading_saddle"]["fitness"] == -0.9
    assert table["published_functional_threshold"]["fitness"] == -0.5
    assert table["candidate_functional_threshold"]["fitness"] == 1.0
    assert table["global_minimum"]["sequence"] == SEQUENCES[0]
    assert table["split_leading_saddle"]["sequence"] == sequence_at(0.3)
    # The median 0.0 is held by exactly one vertex, so it carries that sequence.
    assert table["median"]["sequence"] == sequence_at(0.0)
    assert table["published_functional_threshold"]["sequence"] is None
    assert table["candidate_functional_threshold"]["sequence"] is None


def test_landmarks_table_records_band_kind_and_median_inside_band() -> None:
    for fixture, kind in ((JOIN_BELOW, "overlap"), (SPLIT_BELOW, "gap")):
        result = _compute(fixture)
        for column, expected in (
            ("band_kind", result.summary["band_kind"]),
            ("median_inside_band", result.summary["median_inside_band"]),
        ):
            assert result.landmarks[column].unique().to_list() == [expected]
        assert result.landmarks["band_kind"][0] == kind


def test_band_orientation_join_below_split() -> None:
    summary = _compute(JOIN_BELOW).summary
    assert summary["band_lower_landmark"] == "join_trailing_saddle"
    assert summary["band_kind"] == "overlap"
    assert (summary["band_lower"], summary["band_upper"]) == (-0.6, -0.4)
    assert summary["median_inside_band"] is False


def test_band_orientation_split_below_join() -> None:
    summary = _compute(SPLIT_BELOW).summary
    assert summary["band_lower_landmark"] == "split_trailing_saddle"
    assert summary["band_kind"] == "gap"
    assert (summary["band_lower"], summary["band_upper"]) == (-0.8, -0.5)


def test_median_on_the_band_edge_is_inside() -> None:
    # Join trailing 0.0 equals the median; the closed band holds it.
    summary = _compute(_pairs([0.3, 0.2], [-0.9, 0.0])).summary
    assert (summary["band_lower"], summary["band_upper"]) == (0.0, 0.2)
    assert summary["median_inside_band"] is True
    assert summary["band_count"] == 3  # 0.0, 0.1, 0.2


def test_interval_counts_include_values_exactly_on_a_landmark() -> None:
    intervals = _compute(JOIN_BELOW).intervals
    assert intervals["interval"].to_list() == [
        "global_minimum__join_leading_saddle",
        "join_leading_saddle__join_trailing_saddle",
        "join_trailing_saddle__published_functional_threshold",
        "published_functional_threshold__split_trailing_saddle",
        "split_trailing_saddle__median",
        "median__split_leading_saddle",
        "split_leading_saddle__candidate_functional_threshold",
    ]
    assert intervals["sequence_count"].to_list() == [1, 3, 1, 1, 4, 3, 8]
    assert intervals["upper_closed"].to_list() == [False] * 6 + [True]
    assert intervals["percentage"].to_list() == pytest.approx(
        [100 * n / 21 for n in (1, 3, 1, 1, 4, 3, 8)]
    )
    assert intervals["interval_index"].to_list() == list(range(7))


def test_tied_landmarks_keep_table_order_and_a_zero_width_interval() -> None:
    # Published threshold equal to the split trailing saddle (-0.4).
    result = _compute(JOIN_BELOW, published=-0.4)
    names = result.intervals["interval"].to_list()
    assert "split_trailing_saddle__published_functional_threshold" in names
    row = result.intervals.filter(
        pl.col("interval") == "split_trailing_saddle__published_functional_threshold"
    ).row(0, named=True)
    assert row["sequence_count"] == 0
    assert result.intervals["sequence_count"].sum() == 21


def test_values_outside_the_landmarks_are_counted_apart() -> None:
    # A candidate threshold below the top fitness leaves values above the last landmark.
    result = compute_leading_saddle_band(
        JOIN_BELOW,
        _population(),
        graph=isolated_graph(),
        published_threshold=-0.5,
        candidate_declaration={"rule": "median_reflected", "decimals": 0},
    )
    summary = result.summary
    last = max(row["fitness"] for row in result.landmarks.iter_rows(named=True))
    assert summary["below_first_landmark_count"] == 0
    assert summary["above_last_landmark_count"] == sum(v > last for v in FITNESS)
    assert (
        result.intervals["sequence_count"].sum() + summary["above_last_landmark_count"]
        == 21
    )


def test_trunk_and_band_populations() -> None:
    summary = _compute(JOIN_BELOW).summary
    assert summary["total_sequence_count"] == 21
    assert summary["band_count"] == 3  # -0.6, -0.5, -0.4
    assert summary["split_trunk_count"] == 7  # at or below -0.4 is -1.0 .. -0.4
    assert summary["join_trunk_count"] == 17  # at or above -0.6 is -0.6 .. 1.0
    assert summary["band_percentage"] == pytest.approx(100 * 3 / 21)
    assert summary["split_trunk_percentage"] == pytest.approx(100 * 7 / 21)
    assert summary["join_trunk_percentage"] == pytest.approx(100 * 17 / 21)


def _membership(summary_pairs):
    """The fitness values (rounded) that the stage puts in each population."""
    result = _compute(summary_pairs)
    summary = result.summary
    low, high = summary["band_lower"], summary["band_upper"]
    b_split = summary["split_trailing_saddle_fitness"]
    b_join = summary["join_trailing_saddle_fitness"]
    band = {v for v in FITNESS if low <= v <= high}
    split = {v for v in FITNESS if v <= b_split}
    join = {v for v in FITNESS if v >= b_join}
    return summary, band, split, join


def test_overlap_band_is_the_intersection_of_the_closed_trunks() -> None:
    # b_join -0.6 < b_split -0.4; both saddles sit exactly on population values.
    summary, band, split, join = _membership(JOIN_BELOW)
    assert summary["band_kind"] == "overlap"
    assert band == split & join == {-0.6, -0.5, -0.4}
    assert -0.4 in split and -0.4 in band  # b_split is in the split trunk
    assert -0.6 in join and -0.6 in band  # b_join is in the join trunk
    assert summary["band_count"] == len(band)
    assert summary["split_trunk_count"] == len(split)
    assert summary["join_trunk_count"] == len(join)
    # Not a partition: the populations overlap, so the counts exceed the total.
    assert (
        summary["split_trunk_count"] + summary["join_trunk_count"]
        == summary["total_sequence_count"] + summary["band_count"]
    )


def test_gap_band_lies_in_neither_trunk_but_its_endpoints() -> None:
    # b_split -0.8 < b_join -0.5; the band [-0.8, -0.5] has -0.7 and -0.6 in neither.
    summary, band, split, join = _membership(SPLIT_BELOW)
    assert summary["band_kind"] == "gap"
    assert band == {-0.8, -0.7, -0.6, -0.5}
    assert split == {-1.0, -0.9, -0.8}
    assert join == {v for v in FITNESS if v >= -0.5}
    assert band & split == {-0.8}  # only the closed endpoint b_split
    assert band & join == {-0.5}  # only the closed endpoint b_join
    assert band - split - join == {-0.7, -0.6}
    assert summary["band_count"] == 4
    assert summary["split_trunk_count"] == 3
    assert summary["join_trunk_count"] == 16
    # The two trunks plus the band's interior cover every sequence once.
    assert (
        summary["split_trunk_count"]
        + summary["join_trunk_count"]
        + len(band - split - join)
        == summary["total_sequence_count"]
    )


def test_equal_trailing_saddles_are_filed_as_overlap() -> None:
    summary = _compute(_pairs([0.3, -0.5], [-0.9, -0.5])).summary
    assert summary["band_kind"] == "overlap"
    assert (summary["band_lower"], summary["band_upper"]) == (-0.5, -0.5)
    assert summary["band_count"] == 1
    assert summary["split_trunk_count"] == 6  # -1.0 .. -0.5
    assert summary["join_trunk_count"] == 16  # -0.5 .. 1.0


def test_entropy_of_a_tiny_population() -> None:
    result = _compute(JOIN_BELOW)
    two_to_one = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    band = result.entropy.filter(pl.col("population") == "band")
    assert band["position"].to_list() == [1, 2, 3]
    assert band["entropy"].to_list() == pytest.approx([two_to_one, 0.0, 0.0])
    assert band["sequence_count"].to_list() == [3, 3, 3]
    assert result.summary["band_entropy_sum"] == pytest.approx(two_to_one)
    # The interval [-0.9, -0.6) holds indices 1 to 3: E, K, E.
    name = "join_leading_saddle__join_trailing_saddle"
    assert result.summary["interval_entropy_sums"][name] == pytest.approx(two_to_one)
    assert result.summary["interval_counts"][name] == 3
    assert result.summary["interval_order"][1] == name
    assert set(result.entropy["population"]) >= {
        "band",
        "split_trunk",
        "join_trunk",
        name,
    }


def test_positional_entropy_of_an_empty_population_is_null() -> None:
    empty = pl.DataFrame(schema={"aa1": pl.String, "aa2": pl.String, "aa3": pl.String})
    assert positional_entropy(empty) == {1: None, 2: None, 3: None}


def test_unsupported_codons_are_refused() -> None:
    population = pl.DataFrame({"dna_sequence": ["NNNATGATG"], "raw_fitness": [0.0]})
    with pytest.raises(ValueError, match="Unsupported codon"):
        band_module._with_amino_acids(population)


def test_writer_outputs_schema_and_summary(tmp_path) -> None:
    result = _compute(JOIN_BELOW)
    written = write_leading_saddle_band(tmp_path, result)
    assert {path.name for path in written} == set(ARTEFACTS)
    assert all(path.is_file() for path in written)
    for name, frame in (
        (band_module.LANDMARKS_TABLE, result.landmarks),
        (band_module.INTERVALS_TABLE, result.intervals),
        (band_module.ENTROPY_TABLE, result.entropy),
    ):
        schema = json.loads((tmp_path / f"{name}.schema.json").read_text())
        assert list(schema["columns"]) == frame.columns
        assert schema["description"]
        if name == band_module.LANDMARKS_TABLE:
            assert "band_kind" in schema["description"]
            assert "band_kind" in schema["columns"]
        assert pl.read_csv(tmp_path / f"{name}.csv").height == frame.height
        assert pl.read_parquet(tmp_path / f"{name}.parquet").equals(frame)
    assert (
        "half-open"
        in json.loads(
            (tmp_path / "leading_saddle_band_intervals.schema.json").read_text()
        )["description"]
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    for key in (
        "total_sequence_count",
        "band_kind",
        "band_lower_landmark",
        "band_lower",
        "band_upper",
        "median_inside_band",
        "band_count",
        "split_trunk_count",
        "join_trunk_count",
        "band_entropy_sum",
        "split_trunk_entropy_sum",
        "join_trunk_entropy_sum",
        "interval_order",
        "interval_counts",
        "interval_percentages",
        "interval_entropy_sums",
        "split_trailing_saddle_sequence",
        "candidate_functional_threshold_fitness",
        "n_components",
        "largest_component_size",
        "largest_component_fraction",
        "n_singletons",
        "band_internal_edges",
    ):
        assert key in summary
    for key in band_module.LANDMARK_KEYS:
        assert f"{key}_fitness" in summary and f"{key}_sequence" in summary
    assert read_leading_saddle_band(tmp_path).summary == summary


def test_summary_and_intervals_describe_the_band_and_trunk_rules(tmp_path) -> None:
    result = _compute(JOIN_BELOW)
    summary = result.summary["description"]
    for text in ("band_kind", "not a partition", "intersect", "band_count"):
        assert text in summary
    for table in (band_module.INTERVALS_TABLE, band_module.ENTROPY_TABLE):
        description = band_module.DESCRIPTIONS[table]
        assert "not a partition" in description
    assert "need not equal" in band_module.DESCRIPTIONS[band_module.INTERVALS_TABLE]


@pytest.mark.parametrize("extra_fitness", [0.0, 0.05])
def test_median_sequence_is_null_unless_exactly_one_vertex_has_it(
    extra_fitness,
) -> None:
    # A 22nd sequence makes the median either shared by two vertices (0.0) or
    # a value between two vertices (0.025); neither is a unique vertex.
    population = pl.concat(
        [
            _population(),
            pl.DataFrame(
                {"dna_sequence": ["AAAATGATG"], "raw_fitness": [extra_fitness]}
            ),
        ]
    )
    result = _compute(JOIN_BELOW, population=population)
    median = result.landmarks.filter(pl.col("landmark") == "median").row(0, named=True)
    assert median["fitness"] == pytest.approx(0.0 if extra_fitness == 0.0 else 0.025)
    assert median["sequence"] is None
    assert result.summary["median_sequence"] is None


def test_unique_median_vertex_records_its_sequence() -> None:
    summary = _compute(JOIN_BELOW).summary
    assert summary["median_sequence"] == sequence_at(0.0)


def test_a_saddle_outside_the_population_is_refused() -> None:
    pairs = _pairs(
        [0.3, -0.4], [-0.9, -0.6], split_sequences=["GHOST", sequence_at(-0.4)]
    )
    with pytest.raises(ValueError, match="not in the Full-graph fitness population"):
        _compute(pairs)


def test_a_saddle_with_another_fitness_than_its_pair_is_refused() -> None:
    # The sequence exists but its raw fitness is 0.0, not -0.4.
    pairs = _pairs(
        [0.3, -0.4], [-0.9, -0.6], split_sequences=[sequence_at(0.3), sequence_at(0.0)]
    )
    with pytest.raises(ValueError, match="not its pair scalar"):
        _compute(pairs)


def test_band_connectivity_ignores_an_out_of_band_bridge() -> None:
    # Band [-0.6, -0.2]: two edges, a singleton, and a bridge outside the band
    # that would join the two edges if it were counted. A self-loop and a
    # parallel copy of an in-band edge are not extra edges.
    pairs = _pairs([0.3, -0.2], [-0.9, -0.6])
    graph = isolated_graph()
    index = {
        graph[node]["dna_sequence"]: node for node in graph.node_indices()
    }
    band = [sequence_at(fitness) for fitness in (-0.6, -0.5, -0.4, -0.3, -0.2)]
    graph.add_edge(index[band[0]], index[band[1]], None)
    graph.add_edge(index[band[0]], index[band[1]], None)
    graph.add_edge(index[band[0]], index[band[0]], None)
    graph.add_edge(index[band[2]], index[band[3]], None)
    outside = sequence_at(-1.0)
    graph.add_edge(index[outside], index[band[0]], None)
    graph.add_edge(index[outside], index[band[2]], None)
    graph.add_edge(index[outside], index[band[4]], None)
    summary = _compute(pairs, graph=graph).summary
    assert summary["band_count"] == 5
    assert summary["n_components"] == 3
    assert summary["largest_component_size"] == 2
    assert summary["largest_component_fraction"] == 2 / 5
    assert summary["n_singletons"] == 1
    assert summary["band_internal_edges"] == 2


def test_zero_persistence_pairs_are_not_filtered() -> None:
    # A split pair born and dying at 1.0 and a join pair born and dying at -1.0
    # have persistence 0; they still count as the leading saddles.
    pairs = _pairs([1.0, 0.3, -0.4], [-1.0, -0.9, -0.6])
    assert pairs.filter(pl.col("persistence") == 0.0).height == 2
    saddles = tree_saddles(pairs)
    assert (saddles.split_leading.sequence, saddles.split_leading.fitness) == (
        sequence_at(1.0),
        1.0,
    )
    assert (saddles.join_leading.sequence, saddles.join_leading.fitness) == (
        sequence_at(-1.0),
        -1.0,
    )
    assert saddles.split_trailing.fitness == -0.4
    assert saddles.join_trailing.fitness == -0.6


def test_tree_saddles_on_real_component_persistence_of_the_cube() -> None:
    graph = Graph.from_tables(
        cube_node_table(),
        cube_edge_table(),
        vertex_id_col="node_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("height",),
    )
    result = ScalarNetworkTopology(
        ScalarNetwork(graph, scalar_attr="height")
    ).component_persistence()
    pairs = _pairs_with_sequence_identities(result.pairs, {})
    saddles = tree_saddles(pairs)
    # One ordinary (join saddle 000 at 0.5) and one relative (split saddle 111 at 0.8).
    assert (saddles.join_leading.sequence, saddles.join_leading.fitness) == ("000", 0.5)
    assert saddles.join_trailing == saddles.join_leading
    assert (saddles.split_leading.sequence, saddles.split_leading.fitness) == (
        "111",
        0.8,
    )
    assert saddles.split_trailing == saddles.split_leading
    assert global_minimum(pairs).sequence == "001"
