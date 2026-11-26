"""Peak-rank table and the three peak-rank figures."""

from __future__ import annotations

from dubanevics2026topological.rawdata.value_digest import value_digest


import polars as pl
import pytest

from dubanevics2026topological.analysis.peak_rank import (
    build_peak_rank_table,
    load_published_peaks,
    translate,
)
from dubanevics2026topological.figures.peak_rank import (
    PEAK_RANK_ANNOTATED_ID,
    PEAK_RANK_HISTOGRAM_ID,
    PEAK_RANK_ID,
    build_peak_rank_figure,
    rank_ticks,
)

POPULATION = pl.DataFrame(
    {
        "dna_sequence": ["AAAAAAAAA", "TAATTTTTT", "GGGGGGGGG", "CCCCCCCCC", "TTTTTTTTT"],
        "raw_fitness": [0.5, 0.9, 0.5, -0.2, 0.0],
    }
)


def test_translate_uses_the_repository_codon_table() -> None:
    assert translate("AAAGGGTAA") == "KG%"
    with pytest.raises(ValueError, match="whole number of codons"):
        translate("AAAG")


def test_ranks_descend_by_fitness_and_ties_break_by_sequence() -> None:
    peaks = ["CCCCCCCCC", "GGGGGGGGG", "AAAAAAAAA", "TAATTTTTT"]
    table, _ = build_peak_rank_table(POPULATION, peaks)
    assert table["rank"].to_list() == [1, 2, 3, 4]
    assert table["dna_sequence"].to_list() == [
        "TAATTTTTT",
        "AAAAAAAAA",
        "GGGGGGGGG",
        "CCCCCCCCC",
    ]


def test_stop_codon_peaks_and_exclusions_are_counted() -> None:
    peaks = ["TAATTTTTT", "AAAAAAAAA", "ACGTACGTA"]
    table, summary = build_peak_rank_table(POPULATION, peaks)
    assert table["has_stop_codon"].to_list() == [True, False]
    assert summary["peaks_listed"] == 3
    assert summary["peaks_ranked"] == 2
    assert summary["peaks_excluded"] == 1
    assert summary["excluded_sequences"] == ["ACGTACGTA"]
    assert summary["stop_codon_peaks"] == 1


def test_published_peaks_are_verified_against_the_pinned_hash(tmp_path) -> None:
    path = tmp_path / "peaks.csv"
    path.write_text("dna\nAAAAAAAAA\nGGGGGGGGG\n")
    digest = value_digest(path)
    assert load_published_peaks(path, digest) == ["AAAAAAAAA", "GGGGGGGGG"]
    with pytest.raises(ValueError, match="expected"):
        load_published_peaks(path, "0" * 64)
    path.write_text("dna\nAAAAAAAAA\nAAAAAAAAA\n")
    digest = value_digest(path)
    with pytest.raises(ValueError, match="twice"):
        load_published_peaks(path, digest)


def test_rank_ticks_are_integers_from_one() -> None:
    assert rank_ticks(514, 100) == [1, 100, 200, 300, 400, 500]


def _settings() -> dict:
    figure = {"peak_size": 24, "rank_tick_step": 100}
    return {
        "figures": {
            "fitness_domain": [-1.2, 1.4],
            "rendering": {
                "tree_width": 420,
                "tree_height": 250,
                "stop_codon_shape": "x",
                "stop_codon_color": "#a50f15",
                "legend_font_size": 15,
            },
            "figures": {
                PEAK_RANK_ID: figure,
                PEAK_RANK_ANNOTATED_ID: figure,
                PEAK_RANK_HISTOGRAM_ID: {
                    **figure,
                    "histogram_bins": 20,
                    "histogram_width": 140,
                },
            },
        }
    }


def _payload() -> dict:
    table, _ = build_peak_rank_table(POPULATION, ["TAATTTTTT", "AAAAAAAAA", "GGGGGGGGG"])
    return {"peak_rank": table, "full_fitness": POPULATION}


def test_plain_figure_has_peaks_only_and_no_title() -> None:
    spec = build_peak_rank_figure(_payload(), _settings(), PEAK_RANK_ID).to_dict()
    assert "title" not in spec
    assert len(spec["layer"]) == 1


def test_annotated_figure_adds_stop_crosses_from_config() -> None:
    spec = build_peak_rank_figure(
        _payload(), _settings(), PEAK_RANK_ANNOTATED_ID
    ).to_dict()
    assert "title" not in spec
    _, stops = spec["layer"]
    assert stops["mark"]["filled"] is False
    assert stops["encoding"]["color"]["scale"]["range"] == ["#a50f15"]
    data = spec["datasets"][stops["data"]["name"]]
    assert [row["dna_sequence"] for row in data] == ["TAATTTTTT"]
    assert "synonymous" not in str(spec).lower()


def test_histogram_figure_puts_the_marginal_on_the_shared_fitness_axis() -> None:
    spec = build_peak_rank_figure(
        _payload(), _settings(), PEAK_RANK_HISTOGRAM_ID
    ).to_dict()
    assert "title" not in spec
    assert len(spec["hconcat"]) == 2
    assert spec["resolve"]["scale"]["y"] == "shared"


def test_nan_fitness_is_refused() -> None:
    bad = pl.DataFrame({"dna_sequence": ["AAAAAAAAA"], "raw_fitness": [float("nan")]})
    with pytest.raises(ValueError, match="non-finite"):
        build_peak_rank_table(bad, ["AAAAAAAAA"])


def test_out_of_domain_peak_is_refused_not_clipped() -> None:
    extra = pl.DataFrame({"dna_sequence": ["ACGACGACG"], "raw_fitness": [2.0]})
    population = pl.concat([POPULATION, extra])
    table, _ = build_peak_rank_table(population, ["ACGACGACG", "AAAAAAAAA"])
    with pytest.raises(ValueError, match="outside figures.fitness_domain"):
        build_peak_rank_figure(
            {"peak_rank": table, "full_fitness": population}, _settings(), PEAK_RANK_ID
        )


def test_fitness_axis_uses_the_shared_padded_scales_and_tree_title() -> None:
    from dubanevics2026topological.figures._charts import padded_scalar_domain

    settings = _settings()
    spec = build_peak_rank_figure(_payload(), settings, PEAK_RANK_ID).to_dict()
    y = spec["layer"][0]["encoding"]["y"]
    assert y["scale"]["domain"] == padded_scalar_domain(
        (-1.2, 1.4), settings["figures"]["rendering"]
    )
    assert y["title"] == "Fitness (at birth, WT = 0)"


def test_histogram_counts_every_sequence_once() -> None:
    spec = build_peak_rank_figure(
        _payload(), _settings(), PEAK_RANK_HISTOGRAM_ID
    ).to_dict()
    totals = [
        sum(row["count"] for row in rows)
        for rows in spec["datasets"].values()
        if rows and "count" in rows[0]
    ]
    assert totals == [POPULATION.height]


def test_unknown_figure_id_is_refused() -> None:
    with pytest.raises(ValueError, match="not a peak-rank figure"):
        build_peak_rank_figure(_payload(), _settings(), "papkou2023_other")
