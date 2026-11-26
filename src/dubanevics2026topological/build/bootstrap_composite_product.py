"""Variant-scoped, identity-checked composite panel products."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import altair as alt
import topographer

from ..analysis.candidate_tree_contrast import build_candidate_tree
from ..analysis.checkpoint import checkpoint_lock
from ..analysis import load_settings
from ..figures._legacy_mixture import LegacyPresentationPayload
from ..figures.bootstrap_four_graphs import PANEL_ORDER
from ..figures.tree_histogram import build_bootstrap_tree_histogram_figure
from ..landscape.products import _publish
from ..landscape.selection import resolve_candidate_selection
from ..results import RunContext, load
from ..simplification.bootstrap_smoothed_tree import (
    build_band_simplified_view,
    check_hills_match_tree,
    checkpoint_dir,
    load_cached_bootstrap_smoothed_tree,
)
from ..simplification.identities import band_identity, checkpoint_draw_identity, checkpoint_method
from .bootstrap_codon_entropy_product import _bootstrap_inputs
from .bootstrap_composite_inputs import (
    expected_panel_identity,
    require_checkpoint_identity,
    require_panel_identity,
)

PRODUCT_TYPE = "bootstrap_composite_panel"


def renderer_dependency_paths() -> tuple[Path, ...]:
    """Sources that can change a saved tree-histogram panel specification."""
    figure_dir = Path(__file__).resolve().parent.parent / "figures"
    names = (
        "tree_histogram.py", "_charts.py", "_count_labels.py", "_label_layout.py",
        "_branch_count_style.py", "_legacy_mixture.py", "_top_histogram.py",
        "candidate_tree_biology.py", "palette.py", "slide_tree.py",
    )
    package_dir = Path(topographer.__file__).resolve().parent
    return (
        Path(__file__).resolve(),
        *(figure_dir / name for name in names),
        Path(__file__).resolve().parents[3] / "uv.lock",
        *sorted(package_dir.rglob("*.py")),
    )


def renderer_dependency_digest() -> str:
    """Hash source bytes in stable order without embedding machine-specific paths."""
    digest = hashlib.sha256()
    for index, path in enumerate(renderer_dependency_paths()):
        digest.update(f"{index}:{path.name}\0".encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def presentation_digest(settings: Mapping[str, Any]) -> str:
    encoded = json.dumps(settings.get("figures", {}), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def panel_product_identity(variant: str, settings: Mapping[str, Any], selection: Any, band: str) -> dict[str, Any]:
    """Bind a panel's graph and band to the presentation and renderer sources."""
    return {
        "panel": expected_panel_identity(variant, settings, selection),
        "band_sha256": band,
        "presentation_sha256": presentation_digest(settings),
        "renderer_sha256": renderer_dependency_digest(),
    }


def write_panel_product(context: Any, settings: dict[str, Any], selection: Any) -> Path:
    """Build a variant's panel from its verified checkpoint and selected graph.

    The product stores a Vega-Lite panel specification; its presentation digest
    is part of the identity so a changed figure configuration cannot reuse it.
    The checkpoint stays locked while its tables and their digests are read.
    """
    variant = context.variant_name
    if not variant:
        raise ValueError("Bootstrap composite panels require an explicit variant.")
    tree, _ = build_candidate_tree(settings, selection=selection, context=context)
    with checkpoint_lock(checkpoint_dir(context), "bootstrap smoothed tree"):
        checkpoint = load_cached_bootstrap_smoothed_tree(settings=settings, context=context, lock=False)
        band = active_band_identity(checkpoint, settings, selection)
        check_hills_match_tree(checkpoint.hills, tree)
        hashes = _bootstrap_inputs(context)
        view = build_band_simplified_view(tree, checkpoint.hills)
        scalars = view.assignments["raw_fitness"]
        payload = LegacyPresentationPayload({
            "bootstrap_band_view": view,
            "landscapes": {"candidate_clamped": SimpleNamespace(scalar_domain=(float(scalars.min()), float(scalars.max())))},
        })
        previous_consolidation = alt.data_transformers.consolidate_datasets
        try:
            # Inline datasets so the saved spec can be nested inside the 2x2 chart.
            alt.data_transformers.consolidate_datasets = False
            with alt.data_transformers.enable("default", max_rows=None):
                chart = build_bootstrap_tree_histogram_figure(payload, settings).to_dict()
        finally:
            alt.data_transformers.consolidate_datasets = previous_consolidation
    identity = panel_product_identity(variant, settings, selection, band)

    def write_artifacts(directory: Path) -> None:
        (directory / "panel.json").write_text(json.dumps(chart, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    path, _ = _publish(
        PRODUCT_TYPE,
        {},
        identity,
        context=context,
        input_data_sha256s=hashes,
        parents=(),
        code_paths=(Path(__file__).resolve(), Path(__file__).parent.parent / "figures" / "tree_histogram.py"),
        write_artifacts=write_artifacts,
    )
    return path


def active_band_identity(checkpoint: Any, settings: Mapping[str, Any], selection: Any) -> str:
    """Verify the current checkpoint and derive its direct-band identity."""
    require_checkpoint_identity(checkpoint, settings, selection)
    summary = checkpoint.summary
    if checkpoint_method(summary) != "direct_band":
        raise ValueError("Bootstrap composite requires a direct-band checkpoint.")
    cfg = summary["settings"]
    _, draws = checkpoint_draw_identity(
        summary["fingerprint"], seed=int(cfg["seed"]), draw_ids=range(int(cfg["n_replicates"]))
    )
    _, band = band_identity(draw_scientific_id_sha256=draws, bandq=float(cfg["band_quantile"]))
    if summary["scientific_identities"]["band"]["scientific_id_sha256"] != band:
        raise ValueError("Bootstrap checkpoint direct-band identity differs.")
    return band


def load_panel_product(context: Any, settings: Mapping[str, Any], selection: Any, checkpoint: Any) -> dict[str, Any]:
    """Read an intact panel whose saved selection exactly matches this variant.

    The active checkpoint is checked and its direct-band identity must match.
    No reviewed product pins are modified.
    """
    variant = context.variant_name
    if not variant:
        raise ValueError("Bootstrap composite panels require an explicit variant.")
    identity = expected_panel_identity(variant, settings, selection)
    band = active_band_identity(checkpoint, settings, selection)
    presentation = presentation_digest(settings)
    renderer = renderer_dependency_digest()
    product = load(
        PRODUCT_TYPE,
        context=context,
        expected={"scientific_identity": {"panel": identity, "band_sha256": band, "presentation_sha256": presentation, "renderer_sha256": renderer}},
    )
    if product.report.status != "full" or not product.report.byte_integrity:
        raise ValueError("Bootstrap composite panel product is not fully verified.")
    require_panel_identity(product.manifest.scientific_identity["panel"], variant, settings, selection)
    if product.manifest.scientific_identity.get("band_sha256") != band:
        raise ValueError("Bootstrap composite panel band identity differs.")
    if product.manifest.scientific_identity.get("presentation_sha256") != presentation:
        raise ValueError("Bootstrap composite panel presentation settings differ.")
    if product.manifest.scientific_identity.get("renderer_sha256") != renderer:
        raise ValueError("Bootstrap composite panel renderer sources differ.")
    if product.manifest.data_provenance != _bootstrap_inputs(context):
        raise ValueError("Bootstrap composite panel checkpoint artifacts differ.")
    if "panel.json" not in product.artifacts:
        raise ValueError("Bootstrap composite panel product lacks panel.json.")
    panel = json.loads(product.artifacts["panel.json"].read_text(encoding="utf-8"))
    panel["_composite_threshold"] = product.manifest.scientific_identity["panel"]["threshold"]
    return panel


def load_all_panel_products(context: RunContext) -> dict[str, dict[str, Any]]:
    """Resolve each panel through its own explicit variant and checkpoint root."""
    panels: dict[str, dict[str, Any]] = {}
    for variant, _ in PANEL_ORDER:
        variant_context = RunContext(
            root=context.root,
            output_root=context.output_root / "variants" / variant,
            processed_root=context.processed_root,
            raw_data_dir=context.raw_data_dir,
            barseq_export_dir=context.barseq_export_dir,
            variant_name=variant,
        )
        settings = load_settings(context=variant_context)
        selection = resolve_candidate_selection(settings, context=variant_context)
        with checkpoint_lock(checkpoint_dir(variant_context), "bootstrap smoothed tree"):
            checkpoint = load_cached_bootstrap_smoothed_tree(
                settings=settings, context=variant_context, lock=False
            )
            panels[variant] = load_panel_product(variant_context, settings, selection, checkpoint)
    return panels