"""Supplementary S2: class proportions against the BY FDR target."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.build.functional_counts_vs_q import (
    SCAN_RELATIVE,
    load_fdr_q_scan,
)
from dubanevics2026topological.figures.functional_counts_vs_q import (
    BH_ID,
    BONFERRONI_ID,
    CANDIDATE_THRESHOLD,
    FIGURE_ID,
    FIGURE_IDS,
    build_bh_counts_figure,
    build_bonferroni_counts_figure,
    build_functional_counts_vs_q_figure,
    by_scan,
    candidate_threshold,
    long_proportions,
)
from dubanevics2026topological.figures.registry import get_spec
from dubanevics2026topological.results import RunContext
from dubanevics2026topological.threshold.classification_probability_scans import (
    uniform_fdr_q_grid,
)


def _scan(*, break_step: bool = False, drop_threshold: bool = False) -> pl.DataFrame:
    """A toy combined scan on the 101-step grid. BH rows must be ignored."""
    rows = []
    targets = [float(q) for q in uniform_fdr_q_grid()]
    if break_step:
        targets = [q for q in targets if q != 0.5]
    for cutoff, functional_span, threshold in (
        ("papkou_original", 9, -0.5077739708744209),
        ("median_reflected", 6, CANDIDATE_THRESHOLD),
    ):
        for method, scale in (("by", 1), ("bh", 3), ("bonferroni", 2)):
            span = functional_span * scale
            for q in targets:
                functional = int(span * q)
                uncertain = span - functional
                non_functional = 5 * scale
                total = functional + uncertain + non_functional
                row = {
                    "cutoff": cutoff,
                    "method": method,
                    "fdr_target": q,
                    "functional_count": functional,
                    "uncertain_count": uncertain,
                    "non_functional_count": non_functional,
                    "total_count": total,
                    "functional_fraction": functional / total,
                    "uncertain_fraction": uncertain / total,
                    "non_functional_fraction": non_functional / total,
                }
                if not drop_threshold:
                    row["applied_threshold"] = threshold
                rows.append(row)
    return pl.DataFrame(rows)


def _settings(figure_id: str = FIGURE_ID) -> dict:
    block = {
        "panel_width": 280,
        "panel_height": 200,
        "functional_color": "#1b9e77",
        "uncertain_color": "#d95f02",
        "non_functional_color": "#7570b3",
        "title": "Class proportions against the FDR target",
        "subtitle": "Benjamini-Yekutieli, q from 0 to 1 in steps of 0.01",
    }
    return {
        "figures": {
            "rendering": {"font_size": 12},
            "figures": {figure_id: block},
        }
    }


def test_candidate_threshold_matches_the_six_decimal_rule() -> None:
    assert candidate_threshold() == CANDIDATE_THRESHOLD
    assert round(candidate_threshold(), 6) == -0.253159


def test_by_rows_keep_the_saved_counts_and_drop_bh() -> None:
    kept = by_scan(_scan())
    assert set(kept["method"].unique()) == {"by"}
    published = kept.filter(pl.col("cutoff") == "papkou_original")
    assert published.height == 101
    assert published["functional_count"].to_list()[0] == 0
    assert published["functional_count"].to_list()[-1] == 9
    reflected = kept.filter(pl.col("cutoff") == "median_reflected")
    assert reflected["uncertain_count"].to_list()[0] == 6
    assert reflected["uncertain_count"].to_list()[-1] == 0


def test_a_falling_functional_count_is_refused() -> None:
    table = _scan()
    broken = table.with_columns(
        pl.when(
            (pl.col("method") == "by")
            & (pl.col("cutoff") == "papkou_original")
            & (pl.col("fdr_target") == 1.0)
        )
        .then(pl.lit(0))
        .otherwise(pl.col("functional_count"))
        .alias("functional_count")
    )
    with pytest.raises(ValueError, match="must not fall"):
        by_scan(broken)


def test_a_missing_hundredth_is_refused() -> None:
    with pytest.raises(ValueError, match="0.01 grid"):
        by_scan(_scan(break_step=True))


def test_a_different_saved_candidate_threshold_is_refused() -> None:
    table = _scan().with_columns(
        pl.when(pl.col("cutoff") == "median_reflected")
        .then(pl.lit(-0.2))
        .otherwise(pl.col("applied_threshold"))
        .alias("applied_threshold")
    )
    with pytest.raises(ValueError, match="Topographer threshold"):
        by_scan(table)


def test_stacked_chart_encodes_both_thresholds_and_the_three_proportions() -> None:
    chart = build_functional_counts_vs_q_figure(
        {"scan": _scan(), "source": "toy"}, _settings()
    )
    spec = chart.to_dict()
    assert spec["title"]["text"].startswith("Class proportions")
    panels = spec["hconcat"]
    assert [panel["title"]["text"] for panel in panels] == [
        "Papkou threshold",
        "Topographer threshold",
    ]
    published = spec["datasets"][panels[0]["data"]["name"]]
    assert {row["class"] for row in published} == {
        "Functional",
        "Uncertain",
        "Non-functional",
    }
    functional = [
        row["proportion"] for row in published if row["class"] == "Functional"
    ]
    assert len(functional) == 101
    assert functional[0] == 0.0
    assert {round(row["fdr_target"], 2) for row in published} == {
        round(float(q), 2) for q in uniform_fdr_q_grid()
    }
    assert long_proportions(_scan()).height == 101 * 2 * 3
    assert panels[0]["mark"]["type"] == "bar"
    # The Benjamini-Yekutieli scan is clipped at a tenth like the BH and Bonferroni ones.
    assert panels[0]["mark"]["clip"] is True
    assert panels[0]["encoding"]["y"]["scale"]["domain"] == [0.0, 0.1]
    assert panels[0]["encoding"]["x"]["title"] == "Benjamini-Yekutieli FDR target q"


def test_loader_reads_the_combined_scan(tmp_path: Path) -> None:
    path = tmp_path.joinpath(*SCAN_RELATIVE)
    path.parent.mkdir(parents=True)
    _scan().write_parquet(path)
    loaded = load_fdr_q_scan(RunContext(output_root=tmp_path, root=tmp_path))
    assert loaded["source"] == str(path)
    assert loaded["scan"].height == _scan().height


def test_loader_names_the_missing_stage(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="functional_classification_scan"):
        load_fdr_q_scan(RunContext(output_root=tmp_path, root=tmp_path))


def _clipped_settings(figure_id: str) -> dict:
    settings = _settings(figure_id)
    settings["figures"]["figures"][figure_id]["y_max"] = 0.1
    return settings


@pytest.mark.parametrize(
    ("figure_id", "builder", "label"),
    [
        (BH_ID, build_bh_counts_figure, "Benjamini-Hochberg"),
        (BONFERRONI_ID, build_bonferroni_counts_figure, "Bonferroni"),
    ],
)
def test_bh_and_bonferroni_are_stacked_like_by_with_y_clipped_at_a_tenth(
    figure_id: str, builder, label: str
) -> None:
    spec = builder(
        {"scan": _scan(), "source": "toy"}, _clipped_settings(figure_id)
    ).to_dict()
    panels = spec["hconcat"]
    assert [panel["title"]["text"] for panel in panels] == [
        "Papkou threshold",
        "Topographer threshold",
    ]
    for panel in panels:
        assert panel["mark"]["type"] == "bar"
        assert panel["mark"]["clip"] is True
        y = panel["encoding"]["y"]
        assert y["scale"]["domain"] == [0.0, 0.1]
        assert y["stack"] == "zero"
        assert y["title"] == "Proportion of all genotypes"
        assert label in panel["encoding"]["x"]["title"]
    published = spec["datasets"][panels[0]["data"]["name"]]
    functional = [row["proportion"] for row in published if row["class"] == "Functional"]
    assert len(functional) == 101
    assert functional[-1] > functional[0]
    assert {row["method"] for row in published} == {
        "bh" if figure_id == BH_ID else "bonferroni"
    }


def test_the_heatmap_and_combined_figures_are_gone() -> None:
    assert FIGURE_IDS == {FIGURE_ID, BH_ID, BONFERRONI_ID}
    for retired in (
        "functional_counts_vs_q_heatmap",
        "functional_counts_vs_q_bh_bonferroni",
        "functional_counts_vs_q_bh_bonferroni_heatmap",
    ):
        with pytest.raises(ValueError, match="Unknown figure ID"):
            get_spec(retired)


@pytest.mark.parametrize("figure_id", sorted(FIGURE_IDS))
def test_spec_reads_the_scan_and_only_the_manuscript_source_is_default(
    figure_id: str,
) -> None:
    spec = get_spec(figure_id)
    assert spec.default is (figure_id == FIGURE_ID)
    assert spec.inputs == frozenset({"fdr_q_scan"})
    assert spec.variants == frozenset({"vanilla"})
    assert spec.notebook is None
