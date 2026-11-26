"""Tests for the variant footer on figures rendered under fdr_fixed_cutoff variants."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import altair as alt

from dubanevics2026topological.analysis import ROOT
from dubanevics2026topological.build import (
    BOOTSTRAP_FIGURE_IDS,
    CANDIDATE_FIGURE_IDS,
    NON_CANDIDATE_FIGURE_IDS,
    _render_one,
    apply_figure_footer,
    extract_bootstrap_draws,
    load_settings,
    split_footer_lines,
    variant_figure_footer,
)
from dubanevics2026topological.figures import FIGURE_SPECS
from dubanevics2026topological.results import RunContext


def _variant_settings(variant_name: str) -> dict:
    with (ROOT / "config/variants" / f"{variant_name}.toml").open("rb") as stream:
        overlay = tomllib.load(stream)
    return {"analysis": overlay, "figures": {}}


# 1. Candidate threshold tests (with and without bootstrap count, text and lines)


def test_variant_footer_median_reflected_without_bootstrap() -> None:
    settings = _variant_settings("threshold=reflected_fdr=by_q=0.05")
    footer = variant_figure_footer(settings, has_bootstrap=False)
    expected = (
        "Candidate graph: Candidate threshold (T = −0.253159) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component"
    )
    assert footer == expected
    assert "bootstrap" not in footer
    assert "FDR of 5%" not in footer

    lines = variant_figure_footer(settings, has_bootstrap=False, as_lines=True)
    assert lines == [
        "Candidate graph: Candidate threshold (T = −0.253159) · BY, nominal q = 0.05",
        "≥1 retained Functional endpoint, largest component",
    ]


def test_variant_footer_median_reflected_with_bootstrap() -> None:
    settings = _variant_settings("threshold=reflected_fdr=by_q=0.05")
    footer = variant_figure_footer(settings, has_bootstrap=True, bootstrap_draws=10000)
    expected = (
        "Candidate graph: Candidate threshold (T = −0.253159) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component · "
        "10,000 bootstrap draws"
    )
    assert footer == expected
    assert "FDR of 5%" not in footer

    lines = variant_figure_footer(
        settings, has_bootstrap=True, bootstrap_draws=10000, as_lines=True
    )
    assert lines == [
        "Candidate graph: Candidate threshold (T = −0.253159) · BY, nominal q = 0.05",
        "≥1 retained Functional endpoint, largest component · 10,000 bootstrap draws",
    ]


# 2. Papkou threshold tests (with and without bootstrap count, text and lines)


def test_variant_footer_papkou_original_without_bootstrap() -> None:
    settings = _variant_settings("threshold=published_fdr=by_q=0.05")
    footer = variant_figure_footer(settings, has_bootstrap=False)
    expected = (
        "Candidate graph: Papkou threshold (T = −0.5077739708744209) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component"
    )
    assert footer == expected
    assert "bootstrap" not in footer
    assert "FDR of 5%" not in footer

    lines = variant_figure_footer(settings, has_bootstrap=False, as_lines=True)
    assert lines == [
        "Candidate graph: Papkou threshold (T = −0.5077739708744209) · BY, nominal q = 0.05",
        "≥1 retained Functional endpoint, largest component",
    ]


def test_variant_footer_papkou_original_with_bootstrap() -> None:
    settings = _variant_settings("threshold=published_fdr=by_q=0.05")
    footer = variant_figure_footer(settings, has_bootstrap=True, bootstrap_draws=10000)
    expected = (
        "Candidate graph: Papkou threshold (T = −0.5077739708744209) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component · "
        "10,000 bootstrap draws"
    )
    assert footer == expected
    assert "FDR of 5%" not in footer

    lines = variant_figure_footer(
        settings, has_bootstrap=True, bootstrap_draws=10000, as_lines=True
    )
    assert lines == [
        "Candidate graph: Papkou threshold (T = −0.5077739708744209) · BY, nominal q = 0.05",
        "≥1 retained Functional endpoint, largest component · 10,000 bootstrap draws",
    ]


# 3. Two-line wrapping logic


def test_split_footer_lines_balanced() -> None:
    # 3 segments
    three = (
        "Candidate graph: Candidate threshold (T = −0.253159) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component"
    )
    assert split_footer_lines(three) == [
        "Candidate graph: Candidate threshold (T = −0.253159) · BY, nominal q = 0.05",
        "≥1 retained Functional endpoint, largest component",
    ]

    # 4 segments
    four = (
        "Candidate graph: Candidate threshold (T = −0.253159) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component · "
        "10,000 bootstrap draws"
    )
    assert split_footer_lines(four) == [
        "Candidate graph: Candidate threshold (T = −0.253159) · BY, nominal q = 0.05",
        "≥1 retained Functional endpoint, largest component · 10,000 bootstrap draws",
    ]

    # Short / single segment
    assert split_footer_lines("Short line") == ["Short line"]


# 4. No variant and non-FDR variant tests


def test_variant_footer_default_and_empty_settings() -> None:
    settings = load_settings()
    assert variant_figure_footer(settings) is not None
    assert variant_figure_footer({}) is None


def test_variant_footer_non_fdr_variant_returns_none() -> None:
    settings = {
        "analysis": {
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "mixture_quantile",
                    "decimals": 6,
                }
            }
        }
    }
    assert variant_figure_footer(settings) is None
    mixture = load_settings(context=RunContext(variant_name="mixture_quantile"))
    assert variant_figure_footer(mixture) is None


# 5. Figure classification tests (candidate vs full/published)


def test_every_figure_spec_is_classified_as_candidate_or_non_candidate() -> None:
    all_specs = {spec.figure_id for spec in FIGURE_SPECS}
    # Figure 6 is decided non-Candidate: an opt-in figure with per-panel labels and no
    # generic Candidate footer (variant_figure_footer already returns None for it).
    # Listing it in NON_CANDIDATE_FIGURE_IDS edits build/__init__.py, which is hashed
    # into saved stage fingerprints, so that move is a follow-up ticket. This allowance
    # names this one figure; any other unclassified spec still fails.
    pending_classification = {"reflected_fdr_trees"}
    assert (
        CANDIDATE_FIGURE_IDS | NON_CANDIDATE_FIGURE_IDS | pending_classification
        == all_specs
    )
    assert CANDIDATE_FIGURE_IDS.isdisjoint(NON_CANDIDATE_FIGURE_IDS)
    assert pending_classification.isdisjoint(
        CANDIDATE_FIGURE_IDS | NON_CANDIDATE_FIGURE_IDS
    ), "reflected_fdr_trees is now classified; remove the allowance"
    assert (
        variant_figure_footer(
            _variant_settings("threshold=reflected_fdr=by_q=0.05"),
            figure_id="reflected_fdr_trees",
        )
        is None
    )
    assert BOOTSTRAP_FIGURE_IDS <= CANDIDATE_FIGURE_IDS


def test_peak_uncertainty_four_graphs_has_no_variant_footer() -> None:
    figure_id = "peak_uncertainty"
    assert figure_id in NON_CANDIDATE_FIGURE_IDS
    assert figure_id not in CANDIDATE_FIGURE_IDS
    assert (
        variant_figure_footer(
            _variant_settings("threshold=reflected_fdr=by_q=0.05"), figure_id=figure_id
        )
        is None
    )


def test_mixed_composite_has_no_generic_candidate_footer() -> None:
    figure_id = "bootstrap_four_graphs"
    settings = _variant_settings("threshold=reflected_fdr=by_q=0.05")
    assert figure_id in NON_CANDIDATE_FIGURE_IDS
    assert figure_id not in BOOTSTRAP_FIGURE_IDS
    assert variant_figure_footer(settings, figure_id=figure_id) is None


def test_variant_footer_attaches_only_to_candidate_figures() -> None:
    settings = _variant_settings("threshold=reflected_fdr=by_q=0.05")

    # Non-candidate figures get None
    for fig_id in sorted(NON_CANDIDATE_FIGURE_IDS):
        footer = variant_figure_footer(settings, figure_id=fig_id)
        assert footer is None, f"Expected None for non-candidate figure {fig_id}"

    # Candidate figures get the footer
    for fig_id in sorted(CANDIDATE_FIGURE_IDS):
        footer = variant_figure_footer(
            settings,
            figure_id=fig_id,
            has_bootstrap=fig_id in BOOTSTRAP_FIGURE_IDS,
            bootstrap_draws=10000 if fig_id in BOOTSTRAP_FIGURE_IDS else None,
        )
        assert footer is not None, f"Expected footer for candidate figure {fig_id}"
        assert footer.startswith("Candidate graph: ")
        if fig_id in BOOTSTRAP_FIGURE_IDS:
            assert "10,000 bootstrap draws" in footer
        else:
            assert "bootstrap" not in footer


# 6. Bootstrap draw count extraction from restored provenance / summary


def test_extract_bootstrap_draws_from_provenance() -> None:
    bootstrap = SimpleNamespace(
        provenance={
            "declared": {
                "settings": {
                    "scientific": {"config": {"n_replicates": 10000, "seed": 20261019}}
                }
            }
        },
        summary={"settings": {"n_replicates": 10000}},
    )
    assert extract_bootstrap_draws(bootstrap) == 10000


def test_extract_bootstrap_draws_from_summary_fallback() -> None:
    bootstrap = SimpleNamespace(
        summary={"settings": {"n_replicates": 5000}},
    )
    assert extract_bootstrap_draws(bootstrap) == 5000


def test_variant_footer_reads_bootstrap_draws_from_inputs() -> None:
    settings = _variant_settings("threshold=reflected_fdr=by_q=0.05")
    bootstrap = SimpleNamespace(
        provenance={
            "declared": {
                "settings": {
                    "scientific": {"config": {"n_replicates": 10000, "seed": 20261019}}
                }
            }
        },
        summary={"settings": {"n_replicates": 10000}},
    )
    inputs = {"bootstrap_smoothed": bootstrap}
    footer = variant_figure_footer(settings, has_bootstrap=True, inputs=inputs)
    assert footer is not None
    assert "10,000 bootstrap draws" in footer


# 7. Behavior when footer is None vs present: no artist added when None


def test_apply_figure_footer_none_returns_identical_chart() -> None:
    chart = alt.Chart().mark_point()
    result = apply_figure_footer(chart, None)
    assert result is chart
    assert "title" not in result.to_dict()


def test_apply_figure_footer_attaches_bottom_title() -> None:
    chart = (
        alt.Chart(alt.Data(values=[{"x": 1, "y": 2}]))
        .mark_point()
        .encode(x="x:Q", y="y:Q")
    )
    footer_text = (
        "Candidate graph: Candidate threshold (T = −0.253159) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component"
    )
    result = apply_figure_footer(chart, footer_text)
    assert result is not chart
    spec = result.to_dict()
    # Expect text wrapped into 2 balanced lines
    expected_lines = [
        "Candidate graph: Candidate threshold (T = −0.253159) · BY, nominal q = 0.05",
        "≥1 retained Functional endpoint, largest component",
    ]
    assert spec["title"]["text"] == expected_lines
    assert spec["title"]["orient"] == "bottom"
    assert spec["title"]["anchor"] == "start"


def test_apply_figure_footer_preserves_existing_top_title() -> None:
    chart = (
        alt.Chart(alt.Data(values=[{"x": 1, "y": 2}]))
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .properties(title=alt.TitleParams(text="Top Title", subtitle="My Subtitle"))
    )
    footer_text = (
        "Candidate graph: Papkou threshold (T = −0.5077739708744209) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component"
    )
    result = apply_figure_footer(chart, footer_text)
    import vl_convert

    svg = vl_convert.vegalite_to_svg(result.to_dict())
    assert "Top Title" in svg
    assert "My Subtitle" in svg
    assert "Candidate graph: Papkou threshold" in svg
    assert "≥1 retained Functional endpoint" in svg


# 8. _render_one integration test


def test_render_one_with_and_without_footer(tmp_path: Path) -> None:
    chart = (
        alt.Chart(alt.Data(values=[{"x": 1, "y": 2}]))
        .mark_point()
        .encode(x="x:Q", y="y:Q")
    )
    footer_lines = ["Line 1 of footer", "Line 2 of footer"]
    target_no_footer = tmp_path / "fig_no_footer.svg"
    target_with_footer = tmp_path / "fig_with_footer.svg"

    _render_one(chart, target_no_footer, "svg", footer=None)
    assert target_no_footer.is_file()
    assert "Line 1 of footer" not in target_no_footer.read_text(encoding="utf-8")

    _render_one(chart, target_with_footer, "svg", footer=footer_lines)
    assert target_with_footer.is_file()
    svg_content = target_with_footer.read_text(encoding="utf-8")
    assert "Line 1 of footer" in svg_content
    assert "Line 2 of footer" in svg_content


# 9. Verification of figure IDs carrying bootstrap count


def test_bootstrap_figure_ids_exact_set() -> None:
    expected = {
        "candidate_tree_bootstrap_biology",
        "bootstrap_tree_histogram",
        "bootstrap_codon_entropy",
        "bootstrap_global_stability",
        "bootstrap_band_quantile_scan",
        "bootstrap_band_quantile_scan_seeds",
    }
    assert BOOTSTRAP_FIGURE_IDS == expected


# 10. Top-level properties preservation (datasets, usermeta, config, $schema)


def test_apply_figure_footer_preserves_and_migrates_top_level_datasets() -> None:
    data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    chart = alt.Chart("named_data").mark_point().encode(x="x:Q", y="y:Q")
    chart.datasets = {"named_data": data}
    footer_text = (
        "Candidate graph: Candidate threshold (T = −0.253159) · "
        "BY, nominal q = 0.05 · "
        "≥1 retained Functional endpoint, largest component"
    )
    result = apply_figure_footer(chart, footer_text)
    assert hasattr(result, "datasets")
    assert result.datasets == {"named_data": data}
    import vl_convert

    svg = vl_convert.vegalite_to_svg(result.to_dict())
    assert "Candidate graph: Candidate threshold" in svg
