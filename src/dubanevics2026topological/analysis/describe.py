"""Single public entry point for existing scientific description products."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

Product = Literal[
    "landscape_summary",
    "annotations",
    "ridge_counts",
    "leaves",
    "ridges",
    "merge_flow",
]
SUMMARY_SCHEMA = 2


@dataclass(frozen=True)
class DescribedProducts:
    """Existing description outputs, unchanged from their producer APIs."""

    landscape_summary: dict[str, Any] | None = None
    annotations: Any | None = None
    ridge_counts: dict[str, dict[str, int]] | None = None
    annotation_files: tuple[Path, ...] | None = None
    leaves: Any | None = None
    ridges: Any | None = None
    merge_flow: Any | None = None


def describe(
    *,
    products: Collection[Product] | None = None,
    full_network: Any | None = None,
    published_network: Any | None = None,
    candidate_network: Any | None = None,
    published_functional_threshold: float | None = None,
    candidate_functional_threshold: float | None = None,
    full_summary: dict[str, Any] | None = None,
    candidate_classes: Mapping[str, str] | None = None,
    landscape: Any | None = None,
    landscape_name: str = "study_clamped",
    annotation_directory: Path | None = None,
    comparison: Mapping[str, Any] | None = None,
    bootstrap: Any | None = None,
    hills: pl.DataFrame | None = None,
    tree_analysis: Any | None = None,
    ridge_config: Mapping[str, Any] | None = None,
    functional_sequences: Collection[str] | None = None,
    leaf_tree: Any | None = None,
    leaf_population: str | None = None,
    leaf_tree_state: str = "nominal",
    leaf_hills: pl.DataFrame | None = None,
    landscape_inputs: Mapping[str, Any] | None = None,
    full_fitness: Mapping[str, float] | None = None,
    mixture: Any | None = None,
    ridge_tree: Any | None = None,
    ridge_population: str | None = None,
    ridge_tree_state: str = "nominal",
    ridge_summits: Mapping[str, str] | None = None,
    ridge_vertices: Mapping[str, float] | None = None,
    ridge_classes: Mapping[str, str] | None = None,
    ridge_nominal: Any | None = None,
    merge_flow_tree: Any | None = None,
    merge_flow_state: str | None = None,
    merge_flow_vertices: Mapping[str, float] | None = None,
    merge_flow_classes: Mapping[str, str | None] | None = None,
    merge_flow_names: Mapping[str, str] | None = None,
    merge_flow_pairs: pl.DataFrame | None = None,
    merge_flow_settings: Mapping[str, Any] | None = None,
    merge_flow_nominal: Any | None = None,
    merge_flow_order: Mapping[str, int] | None = None,
) -> DescribedProducts:
    """Compute selected existing products without rebuilding upstream science.

    Inputs are already-built networks, landscape analyses, and (for ridge counts)
    the existing comparison, bootstrap hills, tree analysis, and ridge config.
    For the seven-state landscape summary, supply ``landscape_inputs`` (mapping
    from the seven SUMMARY_COLUMNS names to LandscapeSummaryInput instances)
    together with ``full_fitness`` (and optional ``mixture``).
    ``products=None`` requests every product supported by the supplied inputs.
    Explicitly requested products fail clearly when their inputs are absent.
    """
    requested = set(products) if products is not None else None
    allowed = {
        "landscape_summary",
        "annotations",
        "ridge_counts",
        "leaves",
        "ridges",
        "merge_flow",
    }
    unknown = (requested or set()) - allowed
    if unknown:
        raise ValueError(f"Unknown description products: {sorted(unknown)}")

    summary_inputs = (
        full_network,
        published_network,
        candidate_network,
        published_functional_threshold,
        candidate_functional_threshold,
    )
    summary_available = all(value is not None for value in summary_inputs)
    landscape_seven_available = (
        landscape_inputs is not None and full_fitness is not None
    )
    annotations_available = landscape is not None
    direct_ridge_inputs = (hills, tree_analysis, ridge_config, functional_sequences)
    cached_ridge_available = (
        bootstrap is not None
        and comparison is not None
        and not any(value is not None for value in direct_ridge_inputs)
    )
    direct_ridge_available = bootstrap is None and all(
        value is not None for value in (comparison, hills, tree_analysis, ridge_config)
    )
    ridges_available = cached_ridge_available or direct_ridge_available
    selected = (
        requested
        if requested is not None
        else {
            name
            for name, available in (
                ("landscape_summary", summary_available or landscape_seven_available),
                ("annotations", annotations_available),
                ("ridge_counts", ridges_available),
                ("leaves", leaf_tree is not None and leaf_population is not None),
                ("ridges", ridge_tree is not None and ridge_population is not None),
            )
            if available
        }
    )
    outputs: dict[str, Any] = {}

    if "leaves" in selected:
        if leaf_tree is None or leaf_population is None:
            raise ValueError("leaves requires leaf_tree and leaf_population.")
        from .leaves import build_leaves

        outputs["leaves"] = build_leaves(
            leaf_tree,
            population=leaf_population,
            tree_state=leaf_tree_state,
            hills=leaf_hills,
        )

    if "ridges" in selected:
        if any(
            value is None
            for value in (
                ridge_tree,
                ridge_population,
                ridge_summits,
                ridge_vertices,
                ridge_classes,
            )
        ):
            raise ValueError(
                "ridges requires ridge_tree, ridge_population, ridge_summits, "
                "ridge_vertices, and ridge_classes."
            )
        from .ridges import build_ridges

        outputs["ridges"] = build_ridges(
            ridge_tree,
            population=ridge_population,
            tree_state=ridge_tree_state,
            summits=ridge_summits,
            vertices=ridge_vertices,
            classes=ridge_classes,
            nominal=ridge_nominal,
        )

    if "merge_flow" in selected:
        if any(
            value is None
            for value in (
                merge_flow_tree,
                merge_flow_state,
                merge_flow_vertices,
                merge_flow_classes,
                merge_flow_names,
                merge_flow_pairs,
                merge_flow_settings,
            )
        ):
            raise ValueError(
                "merge_flow requires merge_flow_tree, merge_flow_state, "
                "merge_flow_vertices, merge_flow_classes, merge_flow_names, "
                "merge_flow_pairs, and merge_flow_settings."
            )
        from .merge_flow import build_merge_flow

        outputs["merge_flow"] = build_merge_flow(
            merge_flow_tree,
            state=merge_flow_state,
            vertices=merge_flow_vertices,
            classes=merge_flow_classes,
            names=merge_flow_names,
            pairs=merge_flow_pairs,
            settings=merge_flow_settings,
            nominal=merge_flow_nominal,
            vertex_order=merge_flow_order,
        )

    if "landscape_summary" in selected:
        has_graph = summary_available
        has_seven = landscape_seven_available
        if has_graph and has_seven:
            raise ValueError(
                "landscape_summary inputs are ambiguous: supply either the network "
                "thresholds or the landscape_inputs+full_fitness, not both."
            )
        if has_seven:
            from .graph_summary import landscape_summary as _landscape_summary

            outputs["landscape_summary"] = _landscape_summary(
                landscape_inputs, full_fitness=full_fitness, mixture=mixture
            )
        elif has_graph:
            from .graph_summary import _summary_payload

            outputs["landscape_summary"] = _summary_payload(
                full_network,
                published_network,
                candidate_network,
                published_functional_threshold,
                candidate_functional_threshold,
                full_summary=full_summary,
                candidate_classes=candidate_classes,
            )
        else:
            raise ValueError(
                "landscape_summary requires full_network, published_network, "
                "candidate_network, and both functional thresholds."
            )

    annotation_files = None
    if "annotations" in selected:
        if not annotations_available:
            raise ValueError("annotations requires an existing landscape analysis.")
        from .biological_annotations import (
            _build_biological_annotations,
            _export_biological_annotations,
        )

        annotations = (
            _build_biological_annotations(landscape)
            if landscape_name == "study_clamped"
            else _build_biological_annotations(landscape, landscape=landscape_name)
        )
        outputs["annotations"] = annotations
        if annotation_directory is not None:
            exported = _export_biological_annotations(annotations, annotation_directory)
            annotation_files = None if exported is None else tuple(exported)
    elif annotation_directory is not None:
        raise ValueError("annotation_directory requires the annotations product.")

    if "ridge_counts" in selected:
        direct_requested = any(value is not None for value in direct_ridge_inputs)
        if bootstrap is not None and direct_requested:
            raise ValueError(
                "ridge_counts inputs are ambiguous: choose cached bootstrap or "
                "direct hills/tree/config inputs, not both."
            )
        if bootstrap is not None and comparison is None:
            raise ValueError("cached ridge_counts requires comparison and bootstrap.")
        if bootstrap is None and (
            comparison is None
            or hills is None
            or tree_analysis is None
            or ridge_config is None
        ):
            raise ValueError(
                "direct ridge_counts requires comparison, hills, tree_analysis, "
                "and ridge_config."
            )
        if bootstrap is not None:
            from .band_view_ridges import _load_band_view_ridge_counts

            outputs["ridge_counts"] = _load_band_view_ridge_counts(
                comparison, bootstrap
            )
        else:
            from .band_view_ridges import _band_view_ridge_counts

            outputs["ridge_counts"] = _band_view_ridge_counts(
                comparison,
                hills,
                tree_analysis,
                ridge_config,
                functional_sequences=functional_sequences,
            )

    return DescribedProducts(
        landscape_summary=outputs.get("landscape_summary"),
        annotations=outputs.get("annotations"),
        ridge_counts=outputs.get("ridge_counts"),
        annotation_files=annotation_files,
        leaves=outputs.get("leaves"),
        ridges=outputs.get("ridges"),
        merge_flow=outputs.get("merge_flow"),
    )


def load_saved_annotations(directory: Path) -> Any:
    """Read the annotation tables ``describe(annotation_directory=...)`` saved."""
    from .biological_annotations import BiologicalAnnotations

    frames = {
        name: pl.read_parquet(directory / f"biological_annotation_{name}.parquet")
        for name in ("branches", "pairs", "critical_points")
    }
    summary = json.loads(
        (directory / "biological_annotation_counts.json").read_text(encoding="utf-8")
    )
    return BiologicalAnnotations(summary=summary, **frames)


def __getattr__(name: str) -> Any:
    """Lazy re-export of landscape-summary symbols.

    This lets tables/ and validation scripts import SUMMARY_COLUMNS,
    LandscapeSummaryInput, and landscape_summary_column from the describe
    module (avoiding any direct reference to the graph_summary producer
    module in their source) without pulling graph_summary at describe
    import time (which would create a partial-init cycle through
    analysis.__init__).
    """
    if name in {
        "SUMMARY_COLUMNS",
        "TREE_LEVEL_METRICS",
        "LandscapeSummaryInput",
        "landscape_summary",
        "landscape_summary_column",
    }:
        from . import graph_summary

        return getattr(graph_summary, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
