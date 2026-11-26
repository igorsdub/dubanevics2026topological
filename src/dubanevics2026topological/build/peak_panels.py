"""Peak analysis, pooled scatter and panel summary for each of the four graphs of ``peak_uncertainty``.

The figure draws four graphs: a Published or Candidate threshold, each as a point estimate
or as the Benjamini-Yekutieli selection. Each is an analysis variant of
``config/variants/`` (the ones ``bootstrap_four_graphs`` uses), so each variant gets, under
``<output root>/variants/<variant>/``:

* the peak checkpoint (``analysis/peak_uncertainty``), from the unchanged
  ``load_or_compute_peak_uncertainty`` run under the variant's declaration;
* the ``pooled_scatter`` product, from the unchanged ``build_pooled_scatter_product``;
* a ``peak_panel_summary`` product (this module): the panel's threshold, peak count,
  SEQUENCE counts (functional, non-functional, uncertain where the variant has the class)
  and largest-component size, with the pooled_scatter product as its parent.

Nothing here edits a stage that already exists: the peak checkpoint key, the pooled_scatter
identity and the registry hash what they hashed before (ADR 2026-11-19-peak-analysis-per-variant).
The figure reads the saved products only (``load_peak_panels``) and recomputes nothing.
"""

from __future__ import annotations

import dataclasses
import gc
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow

from ..analysis import ROOT, dataset_input_path, load_settings
from ..analysis.checkpoint import checkpoint_lock
from ..analysis.peak_uncertainty import (
    _fingerprint,
    _load_peak_checkpoint,
    cache_paths,
    load_or_compute_peak_uncertainty,
)
from ..figures.peak_uncertainty import PANEL_VARIANTS, SCATTER_TABLE
from ..landscape.products import _publish
from ..landscape.selection import prepare_candidate_landscape, selection_identity
from ..results import MissingProductError, RunContext, load
from ..threshold.functional_threshold import configured_declaration
from .pooled_scatter_product import _peak_inputs
from .saved_figure_inputs import load_pooled_scatter_product

PRODUCT_TYPE = "peak_panel_summary"
GRAPH = "candidate_clamped"
RETRY_HINT = "just peak-panels-product"

# What each variant must declare (``config/variants/<variant>.toml`` merged over
# ``config/analysis.toml``): a variant file that drifts from its name stops the stage.
PEAK_PANEL_RULES: dict[str, dict[str, Any]] = {
    "threshold=published_fdr=none": {
        "rule": "fixed_cutoff", "cutoff": None, "method": None, "q": None,
    },
    "threshold=reflected_fdr=none": {
        "rule": "median_reflected", "cutoff": None, "method": None, "q": None,
    },
    "threshold=published_fdr=by_q=0.05": {
        "rule": "fdr_fixed_cutoff", "cutoff": "papkou_original", "method": "by", "q": 0.05,
    },
    "threshold=reflected_fdr=by_q=0.05": {
        "rule": "fdr_fixed_cutoff", "cutoff": "median_reflected", "method": "by", "q": 0.05,
    },
}


def variant_context(base: RunContext, variant: str) -> RunContext:
    """``base`` moved to ``<output root>/variants/<variant>`` with that variant's settings.

    Every other field (host, run, processed and raw roots)
    is the base context's, so the products carry the labels the figure chain's other
    stages write. This is the root the existing ``fdr-pooled-scatter-product`` stage
    uses for the median-reflected BY variant.
    """
    if variant not in PEAK_PANEL_RULES:
        raise ValueError(f"Unknown peak panel variant {variant!r}.")
    root = Path(base.output_root) / "variants" / variant
    return dataclasses.replace(base, output_root=root, variant_name=variant)


def variant_reader(context: RunContext) -> RunContext:
    """``context`` (a variant context) for reading its saved products.

    The products are self-contained under the variant root (the same reading context
    the FDR pooled_scatter loader always used).
    """
    return context


def reader_context(base: RunContext, variant: str) -> RunContext:
    """The reading context of ``variant`` under ``base``'s output root."""
    return variant_reader(variant_context(base, variant))


def require_declaration(variant: str, settings: Mapping[str, Any]) -> Mapping[str, Any]:
    """The variant's declared Candidate threshold rule, checked against its name."""
    declaration = configured_declaration(settings)
    expected = PEAK_PANEL_RULES[variant]
    found = {key: declaration.get(key) for key in expected}
    if found != expected:
        raise ValueError(
            f"Variant {variant!r} declares {found}, but a {variant} panel needs {expected}."
        )
    return declaration


def selected_sequence_counts(prepared: Any, selection: Any) -> dict[str, int | None]:
    """SEQUENCE counts of the selected graph (its largest component), by class.

    Under a classification (the BY variants) the class of each vertex decides; under a
    scalar rule a vertex is functional at or above the threshold and there is no
    Uncertain class. These are the counts the graph summary of the comparison stage
    gives for the same graph; they count vertices, never peaks.
    """
    graph = prepared.network.graph
    scalar = prepared.network.scalar_attr
    classes = selection.classes
    counts = {"functional": 0, "non_functional": 0, "uncertain": 0}
    total = 0
    for index in graph.node_indices():
        payload = graph[index]
        total += 1
        if classes is not None:
            label = classes[payload["dna_sequence"]]
        else:
            label = (
                "functional"
                if float(payload[scalar]) >= float(prepared.threshold)
                else "non_functional"
            )
        counts[label] += 1
    expected_total = int(prepared.diagnostics["selected_component_vertex_count"])
    if total != expected_total or sum(counts.values()) != total:
        raise ValueError(
            f"Selected graph has {total} vertices, classes sum to {sum(counts.values())}, "
            f"diagnostics say {expected_total}."
        )
    if counts["functional"] != int(
        prepared.diagnostics["selected_component_functional_vertex_count"]
    ):
        raise ValueError("Functional vertex count differs from the landscape diagnostics.")
    return {
        "functional": counts["functional"],
        "non_functional": counts["non_functional"],
        "uncertain": None if classes is None else counts["uncertain"],
        "vertices": total,
    }


def _tolerances(settings: Mapping[str, Any]) -> dict[str, float]:
    analysis = settings["analysis"]
    return {
        "identity_tolerance": float(
            analysis["sequence_growth_rates"]["identity_tolerance"]
        ),
        "standard_error_tolerance": float(
            analysis["peak_uncertainty_scatter"]["standard_error_tolerance"]
        ),
    }


def _prepare_selected_graph(
    settings: Mapping[str, Any], context: RunContext
) -> tuple[Any, Any]:
    """The variant's selected Candidate graph and its selection, from the Full network.

    The same reads and the same builder the peak analysis uses, so the vertices counted
    here are the vertices whose peaks the checkpoint recorded.
    """
    dataset = settings["analysis"]["dataset"]
    wt = pl.read_parquet(dataset_input_path(dataset["fitness_wt"], context, root=ROOT))
    diff = pl.read_parquet(dataset_input_path(dataset["fitness_diff"], context, root=ROOT))
    network = Papkou2023Workflow(
        landscape="full", fitness_wt=wt, fitness_diff=diff
    ).build_network()
    return prepare_candidate_landscape(network, settings, context=context)


class UnverifiedPeakPanelError(MissingProductError):
    """A saved summary exists but is not fully verified: stale or damaged, not absent.

    A ``MissingProductError`` so the stage's reuse check treats it as "rebuild"; the
    figure loader lets it through unwrapped, as it says more than "missing".
    """


def _verified_summary(
    reader: RunContext,
    variant: str,
    fingerprint: str,
    pooled: Any,
    current: Mapping[str, str],
) -> Any:
    """The saved ``peak_panel_summary`` of ``variant`` if it matches the identity and verifies.

    The one verification the figure-side loader and the stage's reuse check share: the
    summary must carry the variant, the peak checkpoint fingerprint, the pooled_scatter
    id and the current input hashes, be fully verified with intact bytes, and descend
    from that pooled_scatter. ``MissingProductError`` when absent or stale; ``ValueError``
    when present but inconsistent.
    """
    summary = load(
        "peak_panel_summary",
        context=reader,
        expected={
            "scientific_identity": {
                "variant": variant,
                "graph": GRAPH,
                "peak_checkpoint_fingerprint": fingerprint,
                "pooled_scatter_id": pooled.manifest.scientific_id_sha256,
            },
            "data_provenance": current,
        },
    )
    if summary.report.status != "full" or not summary.report.byte_integrity:
        raise UnverifiedPeakPanelError(
            f"peak_panel_summary of {variant} is not fully verified (status "
            f"{summary.report.status!r}, byte_integrity {summary.report.byte_integrity}); "
            f"it is stale or damaged, run `{RETRY_HINT}`."
        )
    parents = {parent.get("scientific_id_sha256") for parent in summary.manifest.parents}
    if parents != {pooled.manifest.scientific_id_sha256}:
        raise ValueError(f"peak_panel_summary of {variant} does not descend from its pooled_scatter.")
    return summary


def build_peak_panel_summary(
    context: RunContext, settings: Mapping[str, Any], pooled: Any
) -> Path:
    """Publish the ``peak_panel_summary`` of ``context``'s variant.

    ``pooled`` is the variant's loaded ``pooled_scatter`` product. The counts come from
    the selected graph itself (``prepare_candidate_landscape``, the one path every
    Candidate network builder uses); the peak count is the pooled scatter's ``is_peak``
    sum, which must equal the checkpoint's and the landscape's.
    """
    variant = str(context.variant_name)
    declaration = require_declaration(variant, settings)
    peak_dir, _ = cache_paths(context)
    with checkpoint_lock(peak_dir, "peak uncertainty"):
        checkpoint = _load_peak_checkpoint(peak_dir, context=context)
        input_hashes = _peak_inputs(context)
    fingerprint = checkpoint["provenance"]["fingerprint"]

    # A repeat run reuses the saved summary of this exact identity instead of rebuilding
    # the Full network and the selection.
    try:
        existing = _verified_summary(
            variant_reader(context), variant, fingerprint, pooled, input_hashes
        )
    except (MissingProductError, ValueError):
        existing = None
    if existing is not None:
        return Path(existing.path)

    prepared, selection = _prepare_selected_graph(settings, context)
    counts = selected_sequence_counts(prepared, selection)

    scatter = pl.read_parquet(pooled.artifacts[f"{SCATTER_TABLE}.parquet"])
    peak_count = int(scatter["is_peak"].sum())
    recorded = int(checkpoint["diagnostics"]["peak_counts"][GRAPH])
    landscape_peaks = int(prepared.diagnostics["peak_count"])
    if not peak_count == recorded == landscape_peaks:
        raise ValueError(
            f"Peak counts of {variant} disagree: scatter {peak_count}, checkpoint "
            f"{recorded}, landscape {landscape_peaks}."
        )
    if variant == "threshold=published_fdr=none":
        # The article's Published graph at its rounded cutoff is the Candidate graph of
        # this variant: both peak counts must agree, or the exact and rounded cutoffs differ.
        study_peaks = int(checkpoint["diagnostics"]["peak_counts"]["study_clamped"])
        if study_peaks != landscape_peaks:
            cutoffs = checkpoint["diagnostics"].get("clamped_thresholds", {})
            raise ValueError(
                f"Published graph of {variant} has {study_peaks} peaks at cutoff "
                f"{cutoffs.get('study_clamped')} but the Candidate graph has "
                f"{landscape_peaks} peaks at cutoff {selection.threshold.threshold}."
            )
    pooled_diagnostics = json.loads(
        pooled.artifacts["diagnostics.json"].read_text(encoding="utf-8")
    )
    threshold = float(selection.threshold.threshold)
    if float(pooled_diagnostics["thresholds"][GRAPH]) != threshold:
        raise ValueError(f"Threshold of {variant} differs between the scatter and the selection.")

    diagnostics = {
        "variant": variant,
        "graph": GRAPH,
        "rule": declaration["rule"],
        "cutoff": declaration.get("cutoff"),
        "fdr_method": declaration.get("method"),
        "fdr_q": declaration.get("q"),
        "threshold": threshold,
        "peak_count": peak_count,
        "functional_sequence_count": counts["functional"],
        "nonfunctional_sequence_count": counts["non_functional"],
        "uncertain_sequence_count": counts["uncertain"],
        "largest_component_vertex_count": counts["vertices"],
        "wild_type_sequence": pooled_diagnostics["wild_type_sequence"],
    }
    identity = {
        "variant": variant,
        "graph": GRAPH,
        "selection": selection_identity(selection),
        "peak_checkpoint_fingerprint": fingerprint,
        "pooled_scatter_id": pooled.manifest.scientific_id_sha256,
    }

    def write_artifacts(directory: Path) -> None:
        (directory / "diagnostics.json").write_text(
            json.dumps(diagnostics, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    landscape = Path(__file__).resolve().parents[1] / "landscape"
    path, _ = _publish(
        "peak_panel_summary",
        {},
        identity,
        context=context,
        input_data_sha256s=input_hashes,
        parents=(
            {
                "product_type": "pooled_scatter",
                "label": pooled.label,
                "scientific_id_sha256": pooled.manifest.scientific_id_sha256,
            },
        ),
        code_paths=(
            Path(__file__).resolve(),
            landscape / "selection.py",
            landscape / "selected_landscape.py",
        ),
        write_artifacts=write_artifacts,
    )
    return path


def build_variant_panel(
    context: RunContext, build_pooled: Callable[..., Path]
) -> Path:
    """Peak checkpoint, pooled_scatter and summary of ``context``'s variant, in that order.

    Each step reuses what a matching earlier run left (the checkpoint by fingerprint, the
    products by identity), so a variant that exists is not recomputed.
    """
    settings = load_settings(context=context)
    require_declaration(str(context.variant_name), settings)
    load_or_compute_peak_uncertainty(settings=settings, context=context)
    build_pooled(context=context)
    reader = variant_reader(context)
    fingerprint = _fingerprint(settings, context=reader)
    pooled = load_pooled_scatter_product(
        reader,
        expected=_tolerances(settings),
        expected_checkpoint_fingerprint=fingerprint,
    )
    return build_peak_panel_summary(context, settings, pooled)


def build_peak_panels(
    base: RunContext,
    build_pooled: Callable[..., Path],
    variants: Sequence[str] = PANEL_VARIANTS,
) -> dict[str, Path]:
    """Build every panel variant under ``<output root>/variants/``; no bootstrap draw."""
    built: dict[str, Path] = {}
    for variant in variants:
        built[variant] = build_variant_panel(variant_context(base, variant), build_pooled)
        gc.collect()
    return built


def load_peak_panels(
    base: RunContext, settings: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """The four saved panels of the figure: ``({variant: payload}, {variant: summary id})``.

    A payload is ``{"scatter": DataFrame, "diagnostics": dict}``: the pooled_scatter
    product's candidate scatter table and the summary's diagnostics. Each product must
    match the variant's current peak checkpoint (fingerprint and table hashes) and the
    summary must descend from that pooled_scatter; a missing or stale one raises
    ``MissingProductError`` naming the stage that builds it.
    """
    panels: dict[str, dict[str, Any]] = {}
    ids: dict[str, str] = {}
    tolerances = _tolerances(settings)
    for variant in PANEL_VARIANTS:
        reader = reader_context(base, variant)
        variant_settings = load_settings(context=reader)
        require_declaration(variant, variant_settings)
        fingerprint = _fingerprint(variant_settings, context=reader)
        try:
            pooled = load_pooled_scatter_product(
                reader, expected=tolerances, expected_checkpoint_fingerprint=fingerprint
            )
            peak_dir, _ = cache_paths(reader)
            with checkpoint_lock(peak_dir, "peak uncertainty"):
                current = _peak_inputs(reader)
            summary = _verified_summary(
                reader, variant, fingerprint, pooled, current
            )
        except UnverifiedPeakPanelError:
            raise
        except MissingProductError as exc:
            raise MissingProductError(
                f"Missing or stale peak panel for variant {variant}; run `{RETRY_HINT}`."
            ) from exc
        diagnostics = json.loads(
            summary.artifacts["diagnostics.json"].read_text(encoding="utf-8")
        )
        if diagnostics.get("variant") != variant:
            raise ValueError(f"peak_panel_summary under {variant} describes {diagnostics.get('variant')!r}.")
        panels[variant] = {
            "scatter": pl.read_parquet(pooled.artifacts[f"{SCATTER_TABLE}.parquet"]),
            "diagnostics": diagnostics,
        }
        ids[variant] = summary.manifest.scientific_id_sha256
    return panels, ids
