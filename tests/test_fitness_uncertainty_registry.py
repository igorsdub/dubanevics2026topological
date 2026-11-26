"""The opt-in fitness-uncertainty specs, and that registering them leaves every other figure alone.

``registry.py`` and ``build/__init__.py`` are hashed into every figure's dependency record, so
editing them re-keys all figures by design (the PNGs are not re-rendered unless a job is run).
What must not change is what the other figures declare and read: their specs, their defaults,
their configuration and the files and settings of their dependency records.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from dubanevics2026topological import build
from dubanevics2026topological.figures import registry

NEW_IDS: tuple[str, ...] = ()  # fitness_uncertainty_2x2 was retired on 2026-11-24
# Registered after the fitness-uncertainty figures (PR 216); also opt-in.
PEAK_RANK_IDS = (
    "peak_rank",
    "peak_rank_annotated",
    "peak_rank_histogram",
)
# Opt-in before the fitness-uncertainty figures were registered.
EXISTING_OPT_IN = frozenset(
    {
        "full_tree_fitness_landmarks",
        "candidate_tree_bootstrap_biology",
        "bootstrap_band_quantile_scan",
    }
)
# Opt-in then, default since PR 320 (manuscript sources).
MANUSCRIPT_DEFAULTS = frozenset(
    {
        "bootstrap_tree_histogram",
        "bootstrap_codon_entropy",
        "bootstrap_global_stability",
        "bootstrap_band_quantile_scan_seeds",
        "functional_counts_vs_q",
        "fitness_se_correlation",
        "branch_vertex_counts_2x2",
        "bootstrap_band_quantile_scan_grid",
        "reflected_fdr_trees",
        "full_tree_persistence_vertical_histogram",
        "candidate_leaf_persistence_band",
        "candidate_leaf_swarm_band",
    }
)
# The registered figures before this change, in order; new ones are appended after them.
EXISTING_IDS = (
    "candidate_edge_evidence",
    "tree_comparison",
    "full_tree_histogram",
    "published_tree_histogram",
    "candidate_tree_histogram",
    "linked_tree_persistence",
    "full_tree_persistence",
    "full_tree_fitness_landmarks",
    "study_tree_persistence_metrics",
    "study_tree_persistence_biology",
    "candidate_tree_bootstrap_biology",
    "bootstrap_tree_histogram",
    "bootstrap_codon_entropy",
    "bootstrap_global_stability",
    "bootstrap_band_quantile_scan",
    "bootstrap_band_quantile_scan_seeds",
    "peak_uncertainty",
    "published_leaf_persistence",
    "candidate_leaf_persistence",
    "published_leaf_swarm",
    "candidate_leaf_swarm",
    "graph_tree_persistence",
)
FIGURES = tomllib.loads(
    (Path(build.ROOT) / "config" / "figures.toml").read_text("utf-8")
)


def test_default_selection_is_unchanged() -> None:
    defaults = {spec.figure_id for spec in registry.FIGURE_SPECS if spec.default}
    # The story views (Decision 10) are default multi-output families.
    story_views = {"tree_persistence", "bootstrap_band", "tree_comparison_views"}
    assert defaults == (
        (set(EXISTING_IDS) - EXISTING_OPT_IN) | story_views | MANUSCRIPT_DEFAULTS
    )
    opt_in = {spec.figure_id for spec in registry.FIGURE_SPECS if not spec.default}
    assert opt_in == EXISTING_OPT_IN | set(NEW_IDS) | set(PEAK_RANK_IDS) | {
        "bootstrap_four_graphs",
        "full_join_tree",
        "functional_counts_vs_q_bh",
        "functional_counts_vs_q_bonferroni",
        "graph_codon_entropy",
        "ridge_entropy",
        "multiple_testing_comparison",
        "papkou_agreement",
        "fitness_vs_pooled_se",
        "published_fdr_leaf_swarm",
        "candidate_fdr_leaf_swarm",
        "published_leaf_persistence_band",
        "published_leaf_swarm_band",
        "peak_pair_matrices",
        "peak_pair_hamming_fitness_scatter",
        "peak_pair_direct_via_saddle_scatter",
    }


def test_existing_specs_declare_what_they_did() -> None:
    for figure_id in EXISTING_IDS:
        spec = registry.get_spec(figure_id)
        assert "fitness_uncertainty" not in spec.inputs
        assert spec.config_table == figure_id
        assert spec.default is (
            figure_id not in EXISTING_OPT_IN or figure_id in MANUSCRIPT_DEFAULTS
        )


@pytest.fixture
def quantities_fingerprint(monkeypatch):
    monkeypatch.setattr(
        "dubanevics2026topological.figures._quantities.registry_fingerprint",
        lambda output: "registry-fingerprint",
    )


def test_existing_dependency_records_do_not_read_the_new_figures(
    quantities_fingerprint,
) -> None:
    from dubanevics2026topological.analysis import load_settings

    settings = load_settings()
    for figure_id in EXISTING_IDS:
        spec = registry.get_spec(figure_id)
        variant = "vanilla" if "vanilla" in spec.variants else "annotated"
        record = build._figure_dependencies(figure_id, variant, settings)
        assert record["figure_config"] == settings["figures"]["figures"][figure_id]
        text = json.dumps(record, default=str)
        # ``d10.fitness_uncertainty`` is shared appearance and was already in the record.
        assert all(new_id not in text for new_id in NEW_IDS), figure_id
        assert "fitness_uncertainty_sweep" not in text, figure_id
