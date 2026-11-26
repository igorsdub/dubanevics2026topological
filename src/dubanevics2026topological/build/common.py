"""Constants and small helpers shared by the build command groups."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..analysis import OUTPUT, ROOT, load_settings
from ..analysis.produce_or_load import Stage
from ..figures.peak_rank import PEAK_RANK_IDS
from ..figures.registry import (
    LEAF_PERSISTENCE_BAND_IDS,
    LEAF_PERSISTENCE_IDS,
    LEAF_SWARM_BAND_IDS,
    LEAF_SWARM_IDS,
)
from ..provenance import without_git
from ..stages import project_stage

FIGURE_DIR = OUTPUT / "figures"
RECORD_DIR = FIGURE_DIR / ".records"
EDGE_FIGURE_ID = "candidate_edge_evidence"
LANDMARKS_FIGURE_ID = "full_tree_fitness_landmarks"
PEAK_FIGURE_ID = "peak_uncertainty"
MANIFOLD_FIGURE_ID = "graph_tree_persistence"
BOOTSTRAP_BIOLOGY_FIGURE_ID = "candidate_tree_bootstrap_biology"
LEAF_FIGURE_IDS = frozenset((*LEAF_PERSISTENCE_IDS, *LEAF_SWARM_IDS))
# The leaf figures of the BY band-simplified D10 trees (saved products only).
BAND_LEAF_FIGURE_IDS = frozenset((*LEAF_PERSISTENCE_BAND_IDS, *LEAF_SWARM_BAND_IDS))
BOOTSTRAP_STABILITY_FIGURE_ID = "bootstrap_global_stability"
BAND_QUANTILE_SCAN_FIGURE_ID = "bootstrap_band_quantile_scan"
BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID = "bootstrap_band_quantile_scan_seeds"
BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID = "bootstrap_tree_histogram"
BOOTSTRAP_CODON_ENTROPY_FIGURE_ID = "bootstrap_codon_entropy"
# The opt-in rotated count-label variant: the collision mode it forces, the suffix
# of its output stem, and the figures that carry branch count labels.
ROTATED_LABEL_COLLISION = "rotate"
ROTATED_STEM_SUFFIX = "_rotated"
LABEL_COLLISION_FIGURE_IDS = frozenset(
    {
        BOOTSTRAP_BIOLOGY_FIGURE_ID,
        BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID,
        BOOTSTRAP_CODON_ENTROPY_FIGURE_ID,
    }
)
BOOTSTRAP_FIGURE_IDS = frozenset(
    {
        BOOTSTRAP_BIOLOGY_FIGURE_ID,
        BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID,
        BOOTSTRAP_CODON_ENTROPY_FIGURE_ID,
        BOOTSTRAP_STABILITY_FIGURE_ID,
        BAND_QUANTILE_SCAN_FIGURE_ID,
        BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID,
    }
)
CANDIDATE_FIGURE_IDS = frozenset(
    {
        "candidate_edge_evidence",
        "tree_comparison",
        "candidate_tree_histogram",
        "full_tree_fitness_landmarks",
        BOOTSTRAP_BIOLOGY_FIGURE_ID,
        BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID,
        BOOTSTRAP_CODON_ENTROPY_FIGURE_ID,
        BOOTSTRAP_STABILITY_FIGURE_ID,
        BAND_QUANTILE_SCAN_FIGURE_ID,
        BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID,
        "candidate_leaf_persistence",
        "candidate_leaf_swarm",
    }
)
# The 2x2 figures (and the single panels) that read the four variants' saved comparison
# checkpoints (build/four_graph_inputs.py).
FOUR_GRAPH_FIGURE_IDS = frozenset(
    {
        "published_fdr_leaf_swarm",
        "candidate_fdr_leaf_swarm",
        "branch_vertex_counts_2x2",
    }
)
NON_CANDIDATE_FIGURE_IDS = frozenset(
    {
        "bootstrap_four_graphs",
        "bootstrap_band_quantile_scan_grid",
        # Four panels (Published/Candidate x No FDR/BY), each titled with its own
        # threshold, FDR setting and counts; one Candidate footer would mislead.
        "peak_uncertainty",
        "full_tree_histogram",
        "published_tree_histogram",
        "linked_tree_persistence",
        "full_tree_persistence",
        "full_tree_persistence_vertical_histogram",
        "full_join_tree",
        "study_tree_persistence_metrics",
        "study_tree_persistence_biology",
        "published_leaf_persistence",
        "published_leaf_swarm",
        # Band-tree leaf figures: each title names its graph, BY and band; no Candidate footer.
        *BAND_LEAF_FIGURE_IDS,
        "graph_tree_persistence",
        "functional_counts_vs_q",
        "functional_counts_vs_q_bh",
        "functional_counts_vs_q_bonferroni",
        "graph_codon_entropy",
        "ridge_entropy",
        # Story views: each title names its own graph, BY and band; no Candidate footer.
        "tree_persistence",
        "bootstrap_band",
        "tree_comparison_views",
        *PEAK_RANK_IDS,
        # Published and Candidate side by side; a single Candidate footer would
        # describe only one of the two thresholds.
        "multiple_testing_comparison",
        "papkou_agreement",
        "fitness_se_correlation",
        "fitness_vs_pooled_se",
        # Four graphs in one figure (or one panel of it): no single Candidate footer.
        *FOUR_GRAPH_FIGURE_IDS,
        "peak_pair_matrices",
        "peak_pair_hamming_fitness_scatter",
        "peak_pair_direct_via_saddle_scatter",
    }
)
# The opt-in bootstrap figures `build figures --with-bootstrap` adds to the defaults: all
# of them. The two band-quantile-scan figures load the registered
# `bootstrap_band_quantile_scan` product (`bootstrap-band-quantile-scan-product`),
# which publishes the restore-only scan tables. They fail loudly when that product
# is missing (ADR 0014, amendment 2026-10-24).
BAND_SCAN_FIGURE_IDS = frozenset(
    {BAND_QUANTILE_SCAN_FIGURE_ID, BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID}
)
WITH_BOOTSTRAP_FIGURE_IDS = BOOTSTRAP_FIGURE_IDS
# Default figures that may be left out while a product they read is missing
# (`[missing_product_exclusions]` in config/figures.toml).
MISSING_PRODUCT_FIGURE_IDS = frozenset({"peak_uncertainty"})
ENV_FONT_DIR = "FITNESSTREE_FONT_DIR"
# Top-level keys of config/figures.toml that are not appearance settings shared
# by the registered figures: the per-figure tables, the build controls, and the
# exploratory tables that no registered figure reads (``exploratory`` for the
# notebooks, ``layout_exploration`` for scripts/layout_exploration.py, and
# ``bootstrap_identity_exclusions``, which only selects figures), so editing
# one never re-renders a publication figure.
NOT_SHARED_FIGURE_KEYS = frozenset(
    {
        "figures",
        "formats",
        "variants",
        "exploratory",
        "layout_exploration",
        "tables",
        "bootstrap_identity_exclusions",
        "missing_product_exclusions",
        "story_views",
    }
)
BUILD_MODULES = (
    "__init__",
    "common",
    "analysis_stages",
    "tables",
    "products",
    "bootstrap",
    "figures",
    "cli",
)


def _build_sources() -> tuple[Path, ...]:
    """The build modules under the current ``ROOT``, so stage code keys stay root-relative."""
    package = ROOT / "src/dubanevics2026topological/build"
    return tuple(package / f"{name}.py" for name in BUILD_MODULES)


def _context_kwargs(context: Any | None) -> dict[str, Any]:
    """``{"context": context}``, or nothing: a context-less call stays as it was."""
    return {} if context is None else {"context": context}
def _load_settings(context: Any | None = None) -> dict[str, Any]:
    """The settings of ``context``; without one, the environment's, as before."""
    return load_settings() if context is None else load_settings(context=context)
def _stage(
    name: str,
    *,
    settings: Mapping[str, Any],
    inputs: Sequence[Path] = (),
    code: Sequence[Path] = (),
    root: Path | None = None,
) -> Stage:
    """Declare a stage under the current ``ROOT`` through the project factory.

    ``ROOT`` is read at call time because tests repoint it.
    The execution keys and libraries come from ``stages.project_stage``.
    """
    effective_root = ROOT if root is None else root
    return project_stage(
        name, settings=settings, inputs=inputs, code=code, root=effective_root
    )
def _digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
def _digest_file(path: Path) -> str:
    return _digest_bytes(path.read_bytes())
def _provenance_digest(provenance: Mapping[str, Any]) -> str:
    """Digest a whole provenance record, leaving out its commit stamp (ADR 0013 D3).

    Every digest of a whole upstream record goes through here, so a commit
    change or a dirty tree never reaches a figure's dependency record.
    """
    return _digest_bytes(
        json.dumps(without_git(provenance), sort_keys=True, default=str).encode("utf-8")
    )
# Where each landscape's annotation tables sit under ``output/tables``.
ANNOTATION_TABLE_DIRECTORIES = {
    "study_clamped": "",
    "candidate_clamped": "candidate_annotations",
}
def _peak_pair_context(context: Any) -> Any:
    """``context`` on the peak-pair stage's own variant, its output root under ``context.root``."""

    from ..analysis.peak_pair_distances import VARIANT

    if context.variant_name == VARIANT:
        return context
    return dataclasses.replace(
        context,
        variant_name=VARIANT,
        output_root=context.root / "output" / "variants" / VARIANT,
    )
def _display_path(path: Path) -> Path:
    """``path`` relative to the repository when it lies under it, else as is."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path
