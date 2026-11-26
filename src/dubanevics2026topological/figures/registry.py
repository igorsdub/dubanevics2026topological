"""Explicit registry of the figures the build can render.

Each :class:`FigureSpec` declares the cached inputs a figure needs, the
variants it allows, its configuration table, its notebook, and one
``render(inputs, settings, variant)`` callable. The build owns input loading
and variant resolution; renderers never read ``output/``.

Every renderer module is imported at the top of this module and mapped to its
source file in ``_RENDERER_FILES``, which the build hashes per figure. A figure
whose builder imports helpers from another figure module maps to a tuple of
files (its own first), so edits to those helpers invalidate its fingerprint.

``FigureSpec.default`` marks whether a figure is part of the default
selection (``build figures`` with no ``--figure``). Opt-in figures
(``default=False``) need extra caches that a fresh checkout lacks; they render
only when named with ``--figure``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ._leaf_data import GRAPHS
from .bootstrap_band_quantile_scan import (
    build_bootstrap_band_quantile_scan_figure,
    build_bootstrap_band_quantile_scan_grid,
)
from .bootstrap_band_quantile_scan_seeds import (
    build_bootstrap_band_quantile_scan_seeds_figure,
)
from .bootstrap_four_graphs import build_bootstrap_four_graphs_figure
from .bootstrap_global_stability import build_bootstrap_global_stability_figure
from .branch_vertex_counts_grid import FIGURE_ID as BRANCH_VERTEX_COUNTS_GRID_ID
from .branch_vertex_counts_grid import build_branch_vertex_counts_grid
from .candidate_edge_evidence import build_candidate_edge_evidence_figure
from .candidate_tree_bootstrap_biology import (
    build_candidate_tree_bootstrap_biology_figure,
)
from .codon_entropy import build_bootstrap_codon_entropy_figure
from .fitness_landmarks import build_fitness_landmarks_figure
from .fitness_pooled_se_pair import FIGURE_ID as FITNESS_VS_POOLED_SE_ID
from .fitness_pooled_se_pair import build_fitness_vs_pooled_se_figure
from .fitness_se_correlation import FIGURE_ID as FITNESS_SE_CORRELATION_ID
from .fitness_se_correlation import build_fitness_se_correlation_figure
from .full_tree_persistence import (
    build_full_tree_persistence_figure,
    build_full_tree_persistence_vertical_histogram_figure,
)
from .functional_counts_vs_q import (
    BH_ID as FUNCTIONAL_COUNTS_VS_Q_BH_ID,
)
from .functional_counts_vs_q import (
    BONFERRONI_ID as FUNCTIONAL_COUNTS_VS_Q_BONFERRONI_ID,
)
from .functional_counts_vs_q import (
    build_bh_counts_figure,
    build_bonferroni_counts_figure,
    build_functional_counts_vs_q_figure,
)
from .functional_counts_vs_q import FIGURE_ID as FUNCTIONAL_COUNTS_VS_Q_ID
from .graph_tree_persistence import build_graph_tree_persistence_figure
from .join_tree import FIGURE_ID as JOIN_TREE_FIGURE_ID
from .join_tree import build_join_tree_figure, payload_for_graph
from .leaf_persistence import (
    build_leaf_persistence_band_figure,
    build_leaf_persistence_figure,
)
from .leaf_swarm import (
    FDR_PANEL_IDS as LEAF_SWARM_FDR_IDS,
)
from .leaf_swarm import (
    build_leaf_swarm_band_figure,
    build_leaf_swarm_figure,
    build_leaf_swarm_panel_figure,
)
from .multiple_testing_comparison import (
    INPUT_NAME as MULTIPLE_TESTING_INPUT,
)
from .multiple_testing_comparison import (
    build_multiple_testing_comparison_figure,
)
from .papkou_agreement import (
    FIGURE_ID as PAPKOU_AGREEMENT_ID,
)
from .papkou_agreement import (
    STEM as PAPKOU_AGREEMENT_STEM,
)
from .papkou_agreement import (
    build_papkou_agreement_figure,
)
from .peak_pair_figures import (
    DIRECT_VIA_ID as PEAK_PAIR_DIRECT_VIA_ID,
)
from .peak_pair_figures import (
    MATRICES_ID as PEAK_PAIR_MATRICES_ID,
)
from .peak_pair_figures import (
    SCATTER_ID as PEAK_PAIR_SCATTER_ID,
)
from .peak_pair_figures import (
    build_peak_pair_direct_via_saddle_figure,
    build_peak_pair_hamming_fitness_scatter_figure,
    build_peak_pair_matrices_figure,
)
from .peak_rank import PEAK_RANK_IDS, build_peak_rank_figure
from .peak_uncertainty import build_peak_uncertainty_figure
from .reflected_fdr_trees import build_reflected_fdr_trees_figure
from .split_tree import build_split_tree_figure
from .study_tree_persistence_biology import (
    build_study_tree_persistence_biology_figure,
)
from .study_tree_persistence_metrics import (
    build_study_tree_persistence_metrics_figure,
)
from .story_views import (
    BOOTSTRAP_BAND_ID,
    GRAPH_CODON_ENTROPY_ID,
    RIDGE_ENTROPY_ID,
    TREE_COMPARISON_VIEWS_ID,
    TREE_PERSISTENCE_ID,
    FigureOutput,
    bootstrap_band_outputs,
    graph_codon_entropy_outputs,
    ridge_entropy_outputs,
    tree_comparison_outputs,
    tree_persistence_outputs,
)
from .tree_comparison import build_tree_comparison_figure
from .tree_histogram import (
    build_bootstrap_tree_histogram_figure,
    build_tree_histogram_figure,
)

SAVED_PRODUCT_INPUTS = frozenset(
    {
        "pooled_scatter",
        "peak_panels",
        "threshold_diagnostics",
        "bootstrap_codon_entropy",
        "bootstrap_band_quantile_scan",
        "graph_tree_persistence",
        "band_leaf_tables",
        "bootstrap_composite_panels",
        "reflected_fdr_trees",
        "graph_codon_entropy",
        "ridge_entropy",
        "tree_views",
        MULTIPLE_TESTING_INPUT,
    }
)
INPUT_NAMES = (
    frozenset(
        {
            "split_analysis",
            "comparison",
            "peak",
            "growth_rates",
            "edge_evidence",
            "component_persistence",
            "full_fitness",
            "leading_saddle_band",
            "full_network",
            "annotations",
            "bootstrap_smoothed",
            "manuscript_quantities",
            "fdr_q_scan",
            "published_fitness",
            "band_quantile_scan_grid",
            "four_graph_comparisons",
            "peak_pair_distances",
        }
    )
    | SAVED_PRODUCT_INPUTS
)
VARIANT_NAMES = frozenset({"vanilla", "annotated"})
SAVED_FIGURE_IDS = frozenset(
    {
        "tree_comparison",
        "full_tree_histogram",
        "published_tree_histogram",
        "candidate_tree_histogram",
        "candidate_tree_bootstrap_biology",
        "bootstrap_tree_histogram",
    }
)

Render = Callable[[dict[str, Any], dict[str, Any], str], Any]
PayloadBuilder = Callable[[Any, dict[str, Any], str], Any]
# A multi-output family's builder: (payload, settings, variant) -> its files.
OutputsBuilder = Callable[[Any, dict[str, Any], str], Iterable[FigureOutput]]


@dataclass(frozen=True)
class FigureSpec:
    """One renderable figure and the inputs, variants and config it needs."""

    figure_id: str
    inputs: frozenset[str]
    variants: frozenset[str]
    config_table: str
    notebook: Path | None
    render: Render
    default: bool = True
    # A multi-output family: ``render`` returns the figure's ``FigureOutput`` files (per saved
    # tree, layout and style) instead of one chart. ``figure_id`` names the family; each
    # output's shelf stem is its file and render-record name.
    multi: bool = False


# -- payload builders: (payload, settings, variant) -> chart -----------------


def _split_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_split_tree_figure(
            payload["analysis"], figure_id, settings, payload["manuscript_quantities"]
        )

    return build


def _edge_builder(payload, settings, variant):
    return build_candidate_edge_evidence_figure(payload, settings)


def _peak_builder(payload, settings, variant):
    return build_peak_uncertainty_figure(payload, settings)


def _full_tree_persistence_builder(payload, settings, variant):
    return build_full_tree_persistence_figure(payload, settings, variant)


def _full_tree_persistence_vertical_builder(payload, settings, variant):
    return build_full_tree_persistence_vertical_histogram_figure(
        payload, settings, variant
    )


def _peak_rank_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_peak_rank_figure(payload, settings, figure_id)

    return build


def _fitness_landmarks_builder(payload, settings, variant):
    return build_fitness_landmarks_figure(payload, settings)


def _join_tree_builder(payload, settings, variant):
    return build_join_tree_figure(payload, settings, variant)


def _functional_counts_vs_q_builder(payload, settings, variant):
    return build_functional_counts_vs_q_figure(payload, settings)


def _bh_builder(payload, settings, variant):
    return build_bh_counts_figure(payload, settings)


def _bonferroni_builder(payload, settings, variant):
    return build_bonferroni_counts_figure(payload, settings)


def _peak_pair_matrices_builder(payload, settings, variant):
    return build_peak_pair_matrices_figure(payload, settings)


def _peak_pair_scatter_builder(payload, settings, variant):
    return build_peak_pair_hamming_fitness_scatter_figure(payload, settings)


def _peak_pair_direct_via_builder(payload, settings, variant):
    return build_peak_pair_direct_via_saddle_figure(payload, settings)


def _compose_peak_pair_distances(inputs: dict[str, Any], settings: dict[str, Any]) -> Any:
    result = inputs["peak_pair_distances"]
    return {"pairs": result.pairs, "mantel": result.mantel}


def _papkou_agreement_builder(payload, settings, variant):
    """One output whose file stem differs from the figure id."""
    yield FigureOutput(
        PAPKOU_AGREEMENT_STEM,
        lambda: build_papkou_agreement_figure(payload, settings),
    )


def _fitness_se_correlation_builder(payload, settings, variant):
    return build_fitness_se_correlation_figure(payload, settings)


def _fitness_vs_pooled_se_builder(payload, settings, variant):
    return build_fitness_vs_pooled_se_figure(payload, settings)


def _leaf_swarm_panel_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_leaf_swarm_panel_figure(payload, settings, figure_id)

    return build


def _branch_vertex_counts_grid_builder(payload, settings, variant):
    return build_branch_vertex_counts_grid(payload, settings)


def _compose_papkou_agreement(inputs: dict[str, Any], settings: dict[str, Any]) -> Any:
    rates = inputs["growth_rates"]
    frame = rates.growth_rates if hasattr(rates, "growth_rates") else rates
    return {"growth_rates": frame, "published": inputs["published_fitness"]}


def _bootstrap_codon_builder(payload, settings, variant):
    return build_bootstrap_codon_entropy_figure(payload, settings)


def _bootstrap_tree_histogram_builder(payload, settings, variant):
    return build_bootstrap_tree_histogram_figure(payload, settings)


def _comparison_builder(payload, settings, variant):
    return build_tree_comparison_figure(payload, settings)


def _reflected_fdr_builder(payload, settings, variant):
    return build_reflected_fdr_trees_figure(payload, settings)


def _histogram_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_tree_histogram_figure(payload, settings, figure_id)

    return build


def _metrics_builder(payload, settings, variant):
    return build_study_tree_persistence_metrics_figure(payload, settings)


def _biology_builder(payload, settings, variant):
    return build_study_tree_persistence_biology_figure(payload, settings)


def _candidate_bootstrap_biology_builder(payload, settings, variant):
    return build_candidate_tree_bootstrap_biology_figure(payload, settings)


def _bootstrap_global_stability_builder(payload, settings, variant):
    return build_bootstrap_global_stability_figure(payload, settings)


def _bootstrap_four_graphs_builder(payload, settings, variant):
    return build_bootstrap_four_graphs_figure(payload, settings)


def _bootstrap_band_quantile_scan_builder(payload, settings, variant):
    return build_bootstrap_band_quantile_scan_figure(payload, settings)


def _bootstrap_band_quantile_scan_seeds_builder(payload, settings, variant):
    return build_bootstrap_band_quantile_scan_seeds_figure(payload, settings)


def _bootstrap_band_quantile_scan_grid_builder(payload, settings, variant):
    return build_bootstrap_band_quantile_scan_grid(payload, settings)


def _leaf_persistence_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_leaf_persistence_figure(payload, settings, figure_id)

    return build


def _leaf_persistence_band_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_leaf_persistence_band_figure(payload, settings, figure_id)

    return build


def _leaf_swarm_band_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_leaf_swarm_band_figure(payload, settings, figure_id)

    return build


def _leaf_swarm_builder(figure_id: str) -> PayloadBuilder:
    def build(payload, settings, variant):
        return build_leaf_swarm_figure(payload, settings, figure_id)

    return build


def _manifold_builder(payload, settings, variant):
    return build_graph_tree_persistence_figure(payload, settings)


def _multiple_testing_builder(payload, settings, variant):
    return build_multiple_testing_comparison_figure(payload, settings, variant)


# The multi-output families: (payload, settings, variant) -> the family's FigureOutput files.
_ridge_entropy_builder: OutputsBuilder = ridge_entropy_outputs
_graph_codon_entropy_builder: OutputsBuilder = graph_codon_entropy_outputs
_tree_persistence_builder: OutputsBuilder = tree_persistence_outputs
_bootstrap_band_builder: OutputsBuilder = bootstrap_band_outputs
_tree_comparison_views_builder: OutputsBuilder = tree_comparison_outputs


# -- composers: (inputs, settings) -> payload --------------------------------


def candidate_contrast_tree(inputs: dict[str, Any], settings: dict[str, Any]) -> Any:
    """Return the verified saved Candidate layout."""
    return inputs["saved_figure_inputs"]["layouts"]["candidate"]


def bootstrap_tree(inputs: dict[str, Any], settings: dict[str, Any]) -> Any:
    """Return the verified nominal layout for the bootstrap's selected tree."""
    if inputs["saved_figure_inputs"]["mode"] == "legacy_mixture":
        from ._legacy_mixture import bootstrap_tree as legacy_bootstrap_tree

        return legacy_bootstrap_tree(inputs, settings)
    expected = inputs["saved_figure_inputs"]["nominal_hills"]
    hills = inputs["bootstrap_smoothed"].hills
    if set(expected.iter_rows()) != set(
        hills.select("peak_sequence", "member_count").iter_rows()
    ):
        raise ValueError("Bootstrap hills differ from the saved nominal leaf product.")
    return candidate_contrast_tree(inputs, settings)


def bootstrap_band_view(inputs: dict[str, Any], settings: dict[str, Any]) -> Any:
    """Return the verified saved direct-band layout."""
    if inputs["saved_figure_inputs"]["mode"] == "legacy_mixture":
        from ._legacy_mixture import bootstrap_band_view as legacy_band_view

        return legacy_band_view(inputs, settings)
    bootstrap_tree(inputs, settings)
    from ..simplification.identities import band_identity, checkpoint_draw_identity

    checkpoint = inputs["bootstrap_smoothed"]
    checkpoint_settings = checkpoint.summary["settings"]
    draw_ids = list(range(int(checkpoint_settings["n_replicates"])))
    for condition in ("real", "flat"):
        actual = (
            checkpoint.draw_stats.filter(
                checkpoint.draw_stats["condition"] == condition
            )["draw"]
            .sort()
            .to_list()
        )
        if actual != draw_ids:
            raise ValueError(
                "Bootstrap checkpoint has incomplete or duplicated draw IDs."
            )
    if checkpoint.draw_stats.height != 2 * len(draw_ids):
        raise ValueError("Bootstrap checkpoint has unexpected draw conditions.")
    _, draw_id = checkpoint_draw_identity(
        checkpoint.summary["fingerprint"],
        seed=int(checkpoint_settings["seed"]),
        draw_ids=draw_ids,
    )
    _, active_band_id = band_identity(
        draw_scientific_id_sha256=draw_id,
        bandq=float(checkpoint_settings["band_quantile"]),
    )
    if active_band_id != inputs["saved_figure_inputs"]["direct_band_identity"]:
        raise ValueError(
            "Active bootstrap band identity differs from the saved band tree."
        )
    view = inputs["saved_figure_inputs"]["layouts"]["band"]
    hills = inputs["bootstrap_smoothed"].hills
    if set(view.pairs["birth_label"]) != set(
        hills.filter(hills["clears_band"])["peak_sequence"]
    ):
        raise ValueError("Bootstrap survivors differ from the saved band tree.")
    return view


def _compose_saved_comparison(inputs, settings):
    saved = inputs["saved_figure_inputs"]
    if saved["mode"] == "legacy_mixture":
        from ._legacy_mixture import LegacyPresentationPayload

        return LegacyPresentationPayload(inputs["comparison"])
    return {
        **inputs["comparison"],
        "saved_layouts": saved["layouts"],
        "saved_populations": saved["populations"],
    }


def _compose_saved_comparison_with_quantities(inputs, settings):
    payload = _compose_saved_comparison(inputs, settings)
    payload["manuscript_quantities"] = inputs["manuscript_quantities"]
    return payload


def _compose_comparison_with_quantities(inputs, settings):
    """The comparison payload with the registry the figure's labels are read from."""
    return {
        **inputs["comparison"],
        "manuscript_quantities": inputs["manuscript_quantities"],
    }


def _compose_comparison_with_annotations(inputs, settings):
    """The comparison payload with the saved annotation tables of both landscapes."""
    return {
        **_compose_comparison_with_quantities(inputs, settings),
        "annotations": inputs["annotations"],
    }


def _compose_split_tree(inputs, settings):
    return {
        "analysis": inputs["split_analysis"],
        "manuscript_quantities": inputs["manuscript_quantities"],
    }


def _compose_peak_uncertainty(inputs, settings):
    """The four saved panels of the peak figure, one per analysis variant.

    Each variant's pooled_scatter and peak_panel_summary products are written by
    `peak-panels-product` under ``output/variants/<variant>/`` and loaded by
    ``build.peak_panels.load_peak_panels`` (which also checks them against the
    variant's current peak checkpoint). Nothing is computed here, and the manuscript
    quantity registry is not read: every number is the product's own diagnostics.
    """
    saved = inputs.get("saved_figure_inputs") or {}
    panels = saved.get("peak_panels")
    if panels is None:
        from ..results import MissingProductError

        raise MissingProductError(
            "Peak uncertainty needs the four saved variant panels; "
            "run `just peak-panels-product`."
        )
    return {"panels": panels}


def _compose_full_tree_persistence(inputs, settings):
    pairs, diagnostics = inputs["component_persistence"]
    payload = {
        "analysis": inputs["split_analysis"],
        "pairs": pairs,
        "diagnostics": diagnostics,
        "full_fitness": inputs["full_fitness"],
        "manuscript_quantities": inputs["manuscript_quantities"],
    }
    saved = inputs.get("saved_figure_inputs") or {}
    if "threshold_diagnostics" in saved:
        payload["threshold_diagnostics"] = saved["threshold_diagnostics"]
    elif "threshold_diagnostics" in inputs:
        payload["threshold_diagnostics"] = inputs["threshold_diagnostics"]
    return payload


def _compose_peak_rank(inputs, settings):
    """The ranked Reference peak set and the Full population its histogram counts."""
    from ..analysis import OUTPUT, dataset_input_path
    from ..analysis.peak_rank import (
        build_peak_rank_table,
        load_published_peaks,
        write_peak_rank_table,
    )

    clamped = settings["analysis"]["clamped"]
    peaks = load_published_peaks(
        dataset_input_path(clamped["peaks"]), clamped["peaks_value_sha256"]
    )
    table, summary = build_peak_rank_table(inputs["full_fitness"], peaks)
    write_peak_rank_table(table, summary, OUTPUT)
    print(
        "peak rank: "
        f"{summary['peaks_ranked']} peaks ranked of {summary['peaks_listed']} listed, "
        f"{summary['stop_codon_peaks']} stop-codon, "
        f"{summary['peaks_excluded']} excluded (absent from the fitness table)",
        flush=True,
    )
    return {
        "peak_rank": table,
        "peak_rank_summary": summary,
        "full_fitness": inputs["full_fitness"],
    }


def _compose_full_join_tree(inputs, settings):
    return payload_for_graph("full", inputs, settings)


def _compose_fitness_landmarks(inputs, settings):
    return {
        "leading_saddle_band": inputs["leading_saddle_band"],
        "full_fitness": inputs["full_fitness"],
        "manuscript_quantities": inputs["manuscript_quantities"],
    }


def _compose_bootstrap_tree_histogram(inputs, settings):
    payload = _compose_saved_comparison(inputs, settings)
    payload["bootstrap_smoothed"] = inputs["bootstrap_smoothed"]
    payload["bootstrap_band_view"] = bootstrap_band_view(inputs, settings)
    return payload


def _legacy_codon_band_view(inputs, settings):
    """PR 3 exception: preserve the existing codon band's tree construction."""
    from ._legacy_mixture import bootstrap_band_view as legacy_band_view

    return legacy_band_view(inputs, settings)


def _compose_bootstrap_codon(inputs, settings):
    """Load precomputed codon populations/sweeps/render tables when available.

    If the bootstrap_codon_entropy product exists and is verified, return a
    payload assembled from its saved tables (pure presentation path). Otherwise
    fall back to the legacy in-memory band reconstruction (mixture_quantile
    default until a band tree product exists for that rule).
    """
    saved = inputs.get("saved_figure_inputs") or {}
    product = saved.get("bootstrap_codon_entropy")
    if product is not None:
        # Product path: the figure must accept pre-built tables.
        # We surface df_* under the names the figure already expects, plus
        # the pre-built tables under a new key the figure will prefer.
        core = product.get("populations_core")
        kem = product.get("populations_kem")
        tcl = product.get("populations_tcl")
        entropy_table = product.get("entropy_sweeps")
        # Reconstruct a minimal entropy dict the figure understands.
        entropy: dict[str, Any] = {}
        if entropy_table is not None and not entropy_table.is_empty():
            for pop in ("core", "kem", "tcl"):
                pop_frame = entropy_table.filter(pl.col("population") == pop).drop(
                    "population"
                )
                if not pop_frame.is_empty():
                    entropy[pop] = pop_frame
        payload = {
            **inputs.get("comparison", {}),
            "df_core": core,
            "df_kem": kem,
            "df_tcl": tcl,
            "metadata": product.get("metadata", {}),
            "entropy": entropy,
            # Provide the pre-built tables so the figure can skip graph work.
            "codon_render_tables": {
                "node_table": product.get("node_table"),
                "edge_table": product.get("edge_table"),
                "edge_segment_table": product.get("edge_segment_table"),
                "ridge_edges": product.get("ridge_edges"),
                "stop_points": product.get("stop_points"),
                "wild_type_marker": product.get("wild_type_marker"),
                "count_domain": product.get("metadata", {}).get("count_domain"),
                "plot_data_attrs": product.get("metadata", {}).get("plot_data_attrs"),
            },
            "manuscript_quantities": inputs.get("manuscript_quantities"),
        }
        return payload

    # Legacy path (mixture_quantile etc. until a band product is available).
    from ..analysis.codon_entropy import compute_entropy_sweeps
    from ..analysis.codon_entropy_populations import extract_codon_entropy_populations
    from ..landscape.selection import resolve_candidate_selection

    view = _legacy_codon_band_view(inputs, settings)
    config = settings["analysis"]["codon_entropy"]
    populations = extract_codon_entropy_populations(
        comparison=inputs["comparison"],
        candidate_view=view,
        config=config,
        save_files=False,
        functional_sequences=resolve_candidate_selection(settings).functional_sequences,
    )
    entropy = compute_entropy_sweeps(
        {
            "core": populations["df_core"],
            "kem": populations["df_kem"],
            "tcl": populations["df_tcl"],
        },
        num_thresholds=int(config["sweep_points"]),
    )
    return {
        **inputs["comparison"],
        **populations,
        "entropy": entropy,
        "candidate_leaf_view": view,
        "manuscript_quantities": inputs["manuscript_quantities"],
    }


def _compose_candidate_bootstrap_biology(inputs, settings):
    tree = bootstrap_tree(inputs, settings)
    if inputs["saved_figure_inputs"]["mode"] == "legacy_mixture":
        from ._legacy_mixture import annotations

        evidence = annotations(tree)
    else:
        evidence = inputs["saved_figure_inputs"]["annotations"]
    payload = {
        **inputs["comparison"],
        "bootstrap_smoothed": inputs["bootstrap_smoothed"],
        "bootstrap_tree": tree,
        "bootstrap_band_view": bootstrap_band_view(inputs, settings),
        "candidate_annotations": evidence,
        "manuscript_quantities": inputs["manuscript_quantities"],
    }
    if inputs["saved_figure_inputs"]["mode"] == "legacy_mixture":
        from ._legacy_mixture import LegacyPresentationPayload

        return LegacyPresentationPayload(payload)
    return payload


def _compose_bootstrap_global_stability(inputs, settings):
    return {
        "bootstrap_smoothed": inputs["bootstrap_smoothed"],
        "manuscript_quantities": inputs["manuscript_quantities"],
    }


def _compose_bootstrap_band_quantile_scan(inputs, settings):
    from ..results import MissingProductError

    product = (inputs.get("saved_figure_inputs") or {}).get(
        "bootstrap_band_quantile_scan"
    )
    if product is None:
        raise MissingProductError(
            "Missing bootstrap_band_quantile_scan product; "
            "run `bootstrap-band-quantile-scan-product` first."
        )
    return {
        "bootstrap_smoothed": inputs["bootstrap_smoothed"],
        "band_quantile_scan": product["scan"],
        "survival": product["survival"],
        "manuscript_quantities": inputs["manuscript_quantities"],
    }


def _compose_bootstrap_band_quantile_scan_grid(inputs, settings):
    return inputs["band_quantile_scan_grid"]


_PAYLOAD_BUILDERS: dict[str, PayloadBuilder] = {}
_COMPOSERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]] = {}

# The leaf diagnostic figures: one per graph (Published, Candidate) and diagram.
LEAF_PERSISTENCE_IDS = tuple(f"{graph}_leaf_persistence" for graph in GRAPHS)
LEAF_SWARM_IDS = tuple(f"{graph}_leaf_swarm" for graph in GRAPHS)
# The same diagrams for each graph's BY band-simplified tree (opt-in; D10 saved trees).
LEAF_PERSISTENCE_BAND_IDS = tuple(f"{graph}_leaf_persistence_band" for graph in GRAPHS)
LEAF_SWARM_BAND_IDS = tuple(f"{graph}_leaf_swarm_band" for graph in GRAPHS)
# The band leaf figures of the manuscript (the swarms of the Papkou + FDR and Topographer + FDR
# graphs, S11 and S13) are built by default.
MANUSCRIPT_BAND_LEAF_IDS = frozenset(
    {"published_leaf_swarm_band", "candidate_leaf_swarm_band"}
)

# Renderer module under figures/ that defines each figure's builder. Every
# spec must be listed; ``renderer_file`` raises for an unmapped figure.
_RENDERER_FILES: dict[str, str | tuple[str, ...]] = {
    "reflected_fdr_trees": "reflected_fdr_trees.py",
    "bootstrap_four_graphs": ("bootstrap_four_graphs.py", "tree_histogram.py"),
    "candidate_edge_evidence": "candidate_edge_evidence.py",
    "peak_uncertainty": "peak_uncertainty.py",
    "tree_comparison": "tree_comparison.py",
    "full_tree_histogram": "tree_histogram.py",
    "published_tree_histogram": "tree_histogram.py",
    "candidate_tree_histogram": "tree_histogram.py",
    # The count labels reach the tree geometry helpers of candidate_tree_biology.py.
    "bootstrap_tree_histogram": (
        "tree_histogram.py",
        "candidate_tree_biology.py",
    ),
    "bootstrap_codon_entropy": (
        "codon_entropy.py",
        "candidate_tree_biology.py",
        "candidate_tree_bootstrap_biology.py",
    ),
    "study_tree_persistence_metrics": "study_tree_persistence_metrics.py",
    "study_tree_persistence_biology": "study_tree_persistence_biology.py",
    "full_tree_persistence": "full_tree_persistence.py",
    "full_tree_persistence_vertical_histogram": "full_tree_persistence.py",
    "full_tree_fitness_landmarks": (
        "fitness_landmarks.py",
        "_leaf_data.py",
    ),
    JOIN_TREE_FIGURE_ID: "join_tree.py",
    "candidate_tree_bootstrap_biology": (
        "candidate_tree_bootstrap_biology.py",
        "candidate_tree_biology.py",
    ),
    "bootstrap_global_stability": (
        "bootstrap_global_stability.py",
        "candidate_tree_biology.py",
    ),
    "bootstrap_band_quantile_scan": (
        "bootstrap_band_quantile_scan.py",
        "candidate_tree_biology.py",
    ),
    "bootstrap_band_quantile_scan_seeds": (
        "bootstrap_band_quantile_scan_seeds.py",
        "bootstrap_band_quantile_scan.py",
        "candidate_tree_biology.py",
    ),
    "bootstrap_band_quantile_scan_grid": "bootstrap_band_quantile_scan.py",
    "linked_tree_persistence": "split_tree.py",
    "graph_tree_persistence": "graph_tree_persistence.py",
    FUNCTIONAL_COUNTS_VS_Q_ID: "functional_counts_vs_q.py",
    FUNCTIONAL_COUNTS_VS_Q_BH_ID: "functional_counts_vs_q.py",
    FUNCTIONAL_COUNTS_VS_Q_BONFERRONI_ID: "functional_counts_vs_q.py",
    PAPKOU_AGREEMENT_ID: ("papkou_agreement.py", "_hexbin.py"),
    FITNESS_SE_CORRELATION_ID: (
        "fitness_se_correlation.py",
        "papkou_agreement.py",
        "_hexbin.py",
    ),
    FITNESS_VS_POOLED_SE_ID: (
        "fitness_pooled_se_pair.py",
        "peak_uncertainty.py",
    ),
    **{
        figure_id: ("leaf_swarm.py", "_leaf_data.py", "_four_graphs.py")
        for figure_id in LEAF_SWARM_FDR_IDS
    },
    BRANCH_VERTEX_COUNTS_GRID_ID: (
        "branch_vertex_counts_grid.py",
        "branch_decomposition_charts.py",
        "_four_graphs.py",
    ),
    PEAK_PAIR_MATRICES_ID: "peak_pair_figures.py",
    PEAK_PAIR_SCATTER_ID: "peak_pair_figures.py",
    PEAK_PAIR_DIRECT_VIA_ID: "peak_pair_figures.py",
    GRAPH_CODON_ENTROPY_ID: ("graph_codon_entropy.py", "story_views.py", "view_naming.py"),
    RIDGE_ENTROPY_ID: ("ridge_entropy_view.py", "story_views.py", "view_naming.py"),
    TREE_PERSISTENCE_ID: ("tree_persistence_view.py", "story_views.py", "view_naming.py"),
    BOOTSTRAP_BAND_ID: ("bootstrap_band_view.py", "story_views.py", "view_naming.py"),
    TREE_COMPARISON_VIEWS_ID: (
        "tree_comparison_view.py",
        "story_views.py",
        "view_naming.py",
    ),
    "multiple_testing_comparison": "multiple_testing_comparison.py",
    **{figure_id: "peak_rank.py" for figure_id in PEAK_RANK_IDS},
    **{
        figure_id: ("leaf_persistence.py", "_leaf_data.py")
        for figure_id in LEAF_PERSISTENCE_IDS
    },
    **{figure_id: ("leaf_swarm.py", "_leaf_data.py") for figure_id in LEAF_SWARM_IDS},
    **{
        figure_id: ("leaf_persistence.py", "_leaf_data.py")
        for figure_id in LEAF_PERSISTENCE_BAND_IDS
    },
    **{
        figure_id: ("leaf_swarm.py", "_leaf_data.py")
        for figure_id in LEAF_SWARM_BAND_IDS
    },
}

# Message raised when a figure is requested with no variant it allows.
_VARIANT_MESSAGES: dict[str, str] = {
    "candidate_edge_evidence": "Topographer + FDR graph evidence has one vanilla export.",
    "peak_uncertainty": (
        "The peak uncertainty figure has one canonical vanilla export."
    ),
    "linked_tree_persistence": (
        "The linked tree and persistence figure has one canonical vanilla export."
    ),
    "tree_comparison": (
        "The Papkou tree comparison has one canonical wild-type annotated export."
    ),
    "candidate_tree_histogram": (
        "Tree histogram figures have one canonical annotated export."
    ),
    "bootstrap_tree_histogram": (
        "The bootstrap tree histogram has one canonical annotated export."
    ),
    "bootstrap_codon_entropy": (
        "The bootstrap codon entropy figure has one canonical annotated export."
    ),
    "full_tree_histogram": (
        "Tree histogram figures have one canonical annotated export."
    ),
    "published_tree_histogram": (
        "Tree histogram figures have one canonical annotated export."
    ),
    "full_tree_fitness_landmarks": (
        "The fitness landmarks figure has one canonical vanilla export."
    ),
    JOIN_TREE_FIGURE_ID: "The join tree figure has one canonical vanilla export.",
    "study_tree_persistence_metrics": (
        "The study tree persistence metrics figure has one canonical vanilla export."
    ),
    "study_tree_persistence_biology": (
        "The biological figure has one canonical annotated export."
    ),
    "candidate_tree_bootstrap_biology": (
        "The bootstrap-smoothed candidate biology figure has one canonical "
        "annotated export."
    ),
    "bootstrap_global_stability": (
        "The bootstrap global stability figure has one canonical annotated export."
    ),
    "bootstrap_band_quantile_scan": (
        "The bootstrap band quantile scan has one canonical annotated export."
    ),
    "bootstrap_band_quantile_scan_seeds": (
        "The band quantile scan seed overlay has one canonical annotated export."
    ),
    "peak_rank": "The peak-rank plot has one canonical vanilla export.",
    "peak_rank_annotated": (
        "The annotated peak-rank plot has one canonical annotated export."
    ),
    "peak_rank_histogram": (
        "The peak-rank histogram figure has one canonical annotated export."
    ),
    "multiple_testing_comparison": (
        "The multiple-testing comparison has one canonical vanilla export."
    ),
    **{
        figure_id: "The leaf diagnostic figures have one canonical vanilla export."
        for figure_id in (
            *LEAF_PERSISTENCE_IDS,
            *LEAF_SWARM_IDS,
            *LEAF_PERSISTENCE_BAND_IDS,
            *LEAF_SWARM_BAND_IDS,
        )
    },
}

# Figures whose direct (notebook) render has always rejected other variants.
# The remaining figures ignore ``variant`` when called through render_payload.
_STRICT_PAYLOAD_VARIANTS = frozenset(
    {
        "candidate_edge_evidence",
        "peak_uncertainty",
        "multiple_testing_comparison",
        "linked_tree_persistence",
        "bootstrap_tree_histogram",
        "bootstrap_codon_entropy",
        "study_tree_persistence_biology",
        "candidate_tree_bootstrap_biology",
        "bootstrap_global_stability",
        "bootstrap_band_quantile_scan",
        "bootstrap_band_quantile_scan_seeds",
        "full_tree_fitness_landmarks",
        JOIN_TREE_FIGURE_ID,
        *LEAF_PERSISTENCE_IDS,
        *LEAF_SWARM_IDS,
        *LEAF_PERSISTENCE_BAND_IDS,
        *LEAF_SWARM_BAND_IDS,
    }
)

_BOTH = frozenset({"vanilla", "annotated"})
_VANILLA = frozenset({"vanilla"})
_ANNOTATED = frozenset({"annotated"})


def _spec(
    figure_id: str,
    inputs: set[str],
    variants: frozenset[str],
    notebook: str | None,
    builder: PayloadBuilder,
    compose: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None,
    *,
    default: bool = True,
    config_table: str | None = None,
    multi: bool = False,
) -> FigureSpec:
    _PAYLOAD_BUILDERS[figure_id] = builder
    if compose is None:
        (only,) = inputs

        def compose(inputs, settings, _only=only):
            return inputs[_only]

    _COMPOSERS[figure_id] = compose

    def render(inputs, settings, variant):
        return builder(compose(inputs, settings), settings, variant)

    return FigureSpec(
        figure_id=figure_id,
        inputs=frozenset(inputs),
        variants=variants,
        config_table=config_table or figure_id,
        notebook=Path(notebook) if notebook else None,
        render=render,
        default=default,
        multi=multi,
    )


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    _spec(
        "reflected_fdr_trees",
        {"reflected_fdr_trees"},
        _ANNOTATED,
        None,
        _reflected_fdr_builder,
        default=False,
        config_table="tree_comparison",
    ),
    _spec(
        "bootstrap_four_graphs",
        {"bootstrap_composite_panels"},
        _ANNOTATED,
        None,
        _bootstrap_four_graphs_builder,
        default=False,
    ),
    _spec(
        "candidate_edge_evidence",
        {"edge_evidence"},
        _VANILLA,
        "notebooks/papkou2023_candidate_edge_evidence.py",
        _edge_builder,
    ),
    _spec(
        "tree_comparison",
        {"comparison", "manuscript_quantities"},
        _ANNOTATED,
        "notebooks/papkou2023_tree_comparison.py",
        _comparison_builder,
        _compose_saved_comparison_with_quantities,
    ),
    _spec(
        "full_tree_histogram",
        {"comparison"},
        _ANNOTATED,
        "notebooks/papkou2023_full_tree_histogram.py",
        _histogram_builder("full_tree_histogram"),
        _compose_saved_comparison,
        default=False,
    ),
    _spec(
        "published_tree_histogram",
        {"comparison"},
        _ANNOTATED,
        "notebooks/papkou2023_published_tree_histogram.py",
        _histogram_builder("published_tree_histogram"),
        _compose_saved_comparison,
        default=False,
    ),
    _spec(
        "candidate_tree_histogram",
        {"comparison"},
        _ANNOTATED,
        "notebooks/papkou2023_candidate_tree_histogram.py",
        _histogram_builder("candidate_tree_histogram"),
        _compose_saved_comparison,
        default=False,
    ),
    _spec(
        "linked_tree_persistence",
        {"split_analysis", "manuscript_quantities"},
        _VANILLA,
        "notebooks/papkou2023_linked_tree_persistence.py",
        _split_builder("linked_tree_persistence"),
        _compose_split_tree,
    ),
    _spec(
        "full_tree_persistence",
        {
            "split_analysis",
            "component_persistence",
            "full_fitness",
            "manuscript_quantities",
            "threshold_diagnostics",
        },
        _BOTH,
        "notebooks/papkou2023_full_tree_persistence.py",
        _full_tree_persistence_builder,
        _compose_full_tree_persistence,
        default=False,
    ),
    _spec(
        "full_tree_persistence_vertical_histogram",
        {
            "split_analysis",
            "component_persistence",
            "full_fitness",
            "manuscript_quantities",
            "threshold_diagnostics",
        },
        _BOTH,
        None,
        _full_tree_persistence_vertical_builder,
        _compose_full_tree_persistence,
        default=True,
    ),
    _spec(
        "full_tree_fitness_landmarks",
        {"leading_saddle_band", "full_fitness", "manuscript_quantities"},
        _VANILLA,
        None,
        _fitness_landmarks_builder,
        _compose_fitness_landmarks,
        default=False,
    ),
    _spec(
        JOIN_TREE_FIGURE_ID,
        {"full_network", "leading_saddle_band", "manuscript_quantities"},
        _VANILLA,
        None,
        _join_tree_builder,
        _compose_full_join_tree,
        default=False,
    ),
    _spec(
        "study_tree_persistence_metrics",
        {"comparison", "manuscript_quantities"},
        _VANILLA,
        "notebooks/papkou2023_study_tree_persistence_metrics.py",
        _metrics_builder,
        _compose_comparison_with_quantities,
    ),
    _spec(
        "study_tree_persistence_biology",
        {"comparison", "manuscript_quantities", "annotations"},
        _ANNOTATED,
        "notebooks/papkou2023_study_tree_persistence_biology.py",
        _biology_builder,
        _compose_comparison_with_annotations,
    ),
    _spec(
        "candidate_tree_bootstrap_biology",
        {"comparison", "bootstrap_smoothed", "manuscript_quantities"},
        _ANNOTATED,
        None,
        _candidate_bootstrap_biology_builder,
        _compose_candidate_bootstrap_biology,
        default=False,
    ),
    _spec(
        "bootstrap_tree_histogram",
        {"comparison", "bootstrap_smoothed"},
        _ANNOTATED,
        None,
        _bootstrap_tree_histogram_builder,
        _compose_bootstrap_tree_histogram,
        default=False,
    ),
    _spec(
        "bootstrap_codon_entropy",
        {
            "comparison",
            "bootstrap_smoothed",
            "manuscript_quantities",
            "bootstrap_codon_entropy",
        },
        _ANNOTATED,
        None,
        _bootstrap_codon_builder,
        _compose_bootstrap_codon,
        default=True,
    ),
    _spec(
        "bootstrap_global_stability",
        {"bootstrap_smoothed", "manuscript_quantities"},
        _ANNOTATED,
        None,
        _bootstrap_global_stability_builder,
        _compose_bootstrap_global_stability,
        default=True,
    ),
    _spec(
        "bootstrap_band_quantile_scan",
        {
            "bootstrap_smoothed",
            "manuscript_quantities",
            "bootstrap_band_quantile_scan",
        },
        _ANNOTATED,
        None,
        _bootstrap_band_quantile_scan_builder,
        _compose_bootstrap_band_quantile_scan,
        default=False,
    ),
    _spec(
        "bootstrap_band_quantile_scan_seeds",
        {
            "bootstrap_smoothed",
            "manuscript_quantities",
            "bootstrap_band_quantile_scan",
        },
        _ANNOTATED,
        None,
        _bootstrap_band_quantile_scan_seeds_builder,
        _compose_bootstrap_band_quantile_scan,
        default=True,
    ),
    _spec(
        "bootstrap_band_quantile_scan_grid",
        {"band_quantile_scan_grid"},
        _ANNOTATED,
        None,
        _bootstrap_band_quantile_scan_grid_builder,
        _compose_bootstrap_band_quantile_scan_grid,
        default=True,
    ),
    _spec(
        "peak_uncertainty",
        {"peak_panels"},
        _VANILLA,
        "notebooks/papkou2023_peak_uncertainty.py",
        _peak_builder,
        _compose_peak_uncertainty,
    ),
    *(
        _spec(
            figure_id,
            {"comparison", "manuscript_quantities", "annotations"},
            _VANILLA,
            None,
            _leaf_persistence_builder(figure_id),
            _compose_comparison_with_annotations,
            default=False,
        )
        for figure_id in LEAF_PERSISTENCE_IDS
    ),
    *(
        _spec(
            figure_id,
            {"comparison", "manuscript_quantities", "annotations"},
            _VANILLA,
            None,
            _leaf_swarm_builder(figure_id),
            _compose_comparison_with_annotations,
        )
        for figure_id in LEAF_SWARM_IDS
    ),
    # Opt-in: the same leaf figures on the BY band-simplified trees. They read the two saved
    # D10 band trees (build/band_leaf_inputs.py), not the comparison checkpoint.
    *(
        _spec(
            figure_id,
            {"band_leaf_tables"},
            _VANILLA,
            None,
            _leaf_persistence_band_builder(figure_id),
            default=figure_id in MANUSCRIPT_BAND_LEAF_IDS,
        )
        for figure_id in LEAF_PERSISTENCE_BAND_IDS
    ),
    *(
        _spec(
            figure_id,
            {"band_leaf_tables"},
            _VANILLA,
            None,
            _leaf_swarm_band_builder(figure_id),
            default=figure_id in MANUSCRIPT_BAND_LEAF_IDS,
        )
        for figure_id in LEAF_SWARM_BAND_IDS
    ),
    # Opt-in: the two Benjamini-Yekutieli singles and the branch vertex count 2x2 read the four variants'
    # saved comparison checkpoints (build/four_graph_inputs.py).
    *(
        _spec(
            figure_id,
            {"four_graph_comparisons"},
            _VANILLA,
            None,
            _leaf_swarm_panel_builder(figure_id),
            default=False,
        )
        for figure_id in LEAF_SWARM_FDR_IDS
    ),
    _spec(
        "graph_tree_persistence",
        {"graph_tree_persistence"},
        _VANILLA,
        "notebooks/graph_tree_persistence.py",
        _manifold_builder,
    ),
    # Opt-in: the Reference peak set (data/raw) beside the Full-graph fitness population;
    #  names them with --figure.
    *(
        _spec(
            figure_id,
            {"full_fitness"},
            _VANILLA if figure_id == "peak_rank" else _ANNOTATED,
            None,
            _peak_rank_builder(figure_id),
            _compose_peak_rank,
            default=False,
        )
        for figure_id in PEAK_RANK_IDS
    ),
    # Historical ridge view (KEM and TCL). Supplementary S14 is graph_codon_entropy.
    _spec(
        RIDGE_ENTROPY_ID,
        {"ridge_entropy"},
        _BOTH,
        None,
        _ridge_entropy_builder,
        default=False,
        multi=True,
    ),
    # Supplementary S14: positional amino-acid entropy of the Full graph and the
    # Papkou graph against raw-fitness threshold. No tree panel. A manuscript
    # source, so `just figures` renders it with the other defaults.
    _spec(
        GRAPH_CODON_ENTROPY_ID,
        {"graph_codon_entropy"},
        _ANNOTATED,
        None,
        _graph_codon_entropy_builder,
        default=True,
        multi=True,
    ),
    _spec(
        "multiple_testing_comparison",
        {MULTIPLE_TESTING_INPUT},
        _VANILLA,
        None,
        _multiple_testing_builder,
        default=False,
    ),
    # Opt-in: the combined FDR scan of classification-probability-scans. Name it
    # with --figure. It reads saved counts and does not run the scan.
    _spec(
        FUNCTIONAL_COUNTS_VS_Q_ID,
        {"fdr_q_scan"},
        _VANILLA,
        None,
        _functional_counts_vs_q_builder,
        default=True,
    ),
    _spec(
        FUNCTIONAL_COUNTS_VS_Q_BH_ID,
        {"fdr_q_scan"},
        _VANILLA,
        None,
        _bh_builder,
        default=False,
    ),
    _spec(
        FUNCTIONAL_COUNTS_VS_Q_BONFERRONI_ID,
        {"fdr_q_scan"},
        _VANILLA,
        None,
        _bonferroni_builder,
        default=False,
    ),
    # Supplementary S1. It reads the growth-rate
    # checkpoint and the published fitness_wt table and refits nothing.
    _spec(
        PAPKOU_AGREEMENT_ID,
        {"growth_rates", "published_fitness"},
        _VANILLA,
        None,
        _papkou_agreement_builder,
        _compose_papkou_agreement,
        default=True,
        multi=True,
    ),
    # Opt-in: fitness against its SE per genotype with Spearman rho, and Papkou
    # published beside our estimate (wild type a red star on ours). Same inputs.
    _spec(
        FITNESS_SE_CORRELATION_ID,
        {"growth_rates", "published_fitness"},
        _VANILLA,
        None,
        _fitness_se_correlation_builder,
        _compose_papkou_agreement,
        default=False,
    ),
    _spec(
        FITNESS_VS_POOLED_SE_ID,
        {"growth_rates", "published_fitness"},
        _VANILLA,
        None,
        _fitness_vs_pooled_se_builder,
        _compose_papkou_agreement,
        default=False,
    ),
    # Opt-in: stacked histogram of branch vertex counts for all four graphs as a 2x2.
    _spec(
        BRANCH_VERTEX_COUNTS_GRID_ID,
        {"four_graph_comparisons"},
        _VANILLA,
        None,
        _branch_vertex_counts_grid_builder,
        default=True,
    ),
    # Opt-in: peak-pair distance figures. They read the peak-pair-distances stage
    # tables (`just peak-pair-distances`) and compute nothing. Name them with --figure.
    _spec(
        PEAK_PAIR_MATRICES_ID,
        {"peak_pair_distances"},
        _VANILLA,
        None,
        _peak_pair_matrices_builder,
        _compose_peak_pair_distances,
        default=False,
    ),
    _spec(
        PEAK_PAIR_SCATTER_ID,
        {"peak_pair_distances"},
        _VANILLA,
        None,
        _peak_pair_scatter_builder,
        _compose_peak_pair_distances,
        default=False,
    ),
    _spec(
        PEAK_PAIR_DIRECT_VIA_ID,
        {"peak_pair_distances"},
        _VANILLA,
        None,
        _peak_pair_direct_via_builder,
        _compose_peak_pair_distances,
        default=False,
    ),
    # The story views (Decision 10): each family writes one file per saved tree (or comparison)
    # and layout; the stems are those of the manuscript shelf. They read saved products only.
    _spec(
        TREE_PERSISTENCE_ID,
        {"tree_views"},
        _BOTH,
        None,
        _tree_persistence_builder,
        multi=True,
    ),
    _spec(
        BOOTSTRAP_BAND_ID,
        {"tree_views"},
        _BOTH,
        None,
        _bootstrap_band_builder,
        multi=True,
    ),
    _spec(
        TREE_COMPARISON_VIEWS_ID,
        {"tree_views"},
        _ANNOTATED,
        None,
        _tree_comparison_views_builder,
        multi=True,
    ),
)

FIGURE_IDS = tuple(spec.figure_id for spec in FIGURE_SPECS)
_BY_ID = {spec.figure_id: spec for spec in FIGURE_SPECS}


def get_spec(figure_id: str) -> FigureSpec:
    """Return the spec registered for ``figure_id`` or raise ValueError."""
    try:
        return _BY_ID[figure_id]
    except KeyError:
        raise ValueError(
            f"Unknown figure ID {figure_id!r}; choose from {FIGURE_IDS}."
        ) from None


def variant_error(spec: FigureSpec) -> ValueError:
    """Error for a figure requested with no variant it allows."""
    allowed = ", ".join(sorted(spec.variants))
    return ValueError(
        _VARIANT_MESSAGES.get(
            spec.figure_id,
            f"Figure {spec.figure_id!r} allows only variant(s): {allowed}.",
        )
    )


# Modules imported by more than one renderer; they are hashed into every
# figure's dependency fingerprint next to the figure's own renderer file.
SHARED_RENDERER_FILES: tuple[str, ...] = (
    "_charts.py",
    "_legacy_mixture.py",
    "_branch_count_style.py",
    "_top_histogram.py",
    "_count_labels.py",
    "component_persistence.py",
    "candidate_tree_biology.py",
    "tree_persistence_view.py",
    "registry.py",
)


def renderer_files(spec: FigureSpec) -> tuple[str, ...]:
    """Files under figures/ the spec's builder lives in, its own module first."""
    try:
        mapped = _RENDERER_FILES[spec.figure_id]
    except KeyError:
        raise ValueError(
            f"No renderer file mapped for figure {spec.figure_id!r}."
        ) from None
    return (mapped,) if isinstance(mapped, str) else tuple(mapped)


def renderer_file(spec: FigureSpec) -> str:
    """Name of the file under figures/ defining the spec's builder."""
    return renderer_files(spec)[0]


def render_payload(
    spec: FigureSpec, payload: Any, settings: dict[str, Any], variant: str
) -> Any:
    """Render from an already composed payload (notebook calling convention)."""
    if variant not in _BOTH:
        raise ValueError("variant must be 'vanilla' or 'annotated'.")
    if spec.figure_id in _STRICT_PAYLOAD_VARIANTS and variant not in spec.variants:
        raise variant_error(spec)
    return _PAYLOAD_BUILDERS[spec.figure_id](payload, settings, variant)


def compose_payload(
    spec: FigureSpec, inputs: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Compose the payload a spec's renderer consumes from loaded inputs."""
    return _COMPOSERS[spec.figure_id](inputs, settings)
