"""Saved-product commands (landscape, tree-layout, scatter and bootstrap products)."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..analysis import ROOT
from .common import _load_settings
from .tables import build_peak_tables

FDR_POOLED_VARIANT = "threshold=reflected_fdr=by_q=0.05"


FDR_POOLED_DECLARATION = {
    "rule": "fdr_fixed_cutoff",
    "decimals": 6,
    "cutoff": "median_reflected",
    "method": "by",
    "q": 0.05,
    "input_dir": "data/processed/classification/reflected",
}


def _fdr_peak_settings(context: Any) -> dict[str, Any]:
    settings = _load_settings(context)
    declaration = settings["analysis"]["clamped"]["candidate_functional_threshold"]
    if declaration != FDR_POOLED_DECLARATION:
        raise ValueError("FDR peak selection does not match the approved median-reflected snapshot.")
    return settings


def _build_under_fdr_variant(build_pooled: Callable[..., Path]) -> Path:
    """Build the FDR variant's peak checkpoint and pooled_scatter under its own output root.

    The variant env is set only for this build; the default ``output/`` is never written.
    """
    from ..analysis.peak_uncertainty import load_or_compute_peak_uncertainty
    from ..results import RunContext

    root = f"output/variants/{FDR_POOLED_VARIANT}"
    keys = ("FITNESSTREE_ANALYSIS_VARIANT", "FITNESSTREE_OUTPUT_ROOT")
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["FITNESSTREE_ANALYSIS_VARIANT"] = FDR_POOLED_VARIANT
    os.environ["FITNESSTREE_OUTPUT_ROOT"] = root
    try:
        context = RunContext()
        result = load_or_compute_peak_uncertainty(context=context)
        build_peak_tables(result, context=context)
        return build_pooled(context=context)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _fdr_pooled_output_root() -> Path:
    """Output root that holds the FDR pooled_scatter product on disk.

    The safer script sets ``FITNESSTREE_FDR_POOLED_ROOT`` to the
    ``FITNESSTREE_OUTPUT_ROOT`` of the pooled write
    (``output/variants/<variant>``). A process already using that variant root
    is read as-is. Otherwise the median-reflected variant directory is used.
    """
    explicit = os.environ.get("FITNESSTREE_FDR_POOLED_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    active = os.environ.get("FITNESSTREE_OUTPUT_ROOT", "").strip()
    if active and "variants" in Path(active).parts:
        return Path(active)
    variant = os.environ.get("FITNESSTREE_ANALYSIS_VARIANT", "").strip()
    if not variant:
        variant = FDR_POOLED_VARIANT
    return ROOT / "output" / "variants" / variant


def cmd_landscape_products(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..landscape.build_products import build_and_save_landscape_products

    saved = build_and_save_landscape_products(context=context)
    for name, (landscape_path, tree_path) in saved.items():
        print(f"{name}: {landscape_path} ; {tree_path}")
    return 0


def cmd_tree_layout_products(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .tree_layout_products import build_tree_layouts

    for name, path in build_tree_layouts(context).items():
        print(f"{name}: {path}")
    return 0


def cmd_leaves_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .leaves_product import build_leaves_products

    for name, path in build_leaves_products(context).items():
        print(f"{name}: {path}")
    return 0


def cmd_landscape_summary_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .landscape_summary_product import build_landscape_summary_products

    for name, path in build_landscape_summary_products(context).items():
        print(f"{name}: {path}")
    return 0


def cmd_ridges_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .ridges_product import build_ridges_products

    for name, path in build_ridges_products(context).items():
        print(f"{name}: {path}")
    return 0


def cmd_merge_flow_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .merge_flow_product import build_merge_flow_products

    for name, path in build_merge_flow_products(context).items():
        print(f"{name}: {path}")
    return 0


def cmd_ridge_entropy_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .ridge_entropy_product import build_ridge_entropy_products

    for name, path in build_ridge_entropy_products(context).items():
        print(f"{name}: {path}")
    return 0


def cmd_pooled_scatter_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .pooled_scatter_product import build_pooled_scatter_product

    path = build_pooled_scatter_product(context=context)
    print(f"pooled_scatter: {path}")
    return 0


def cmd_fdr_pooled_scatter_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .pooled_scatter_product import build_pooled_scatter_product

    path = _build_under_fdr_variant(build_pooled_scatter_product)
    print(f"fdr pooled_scatter: {path}")
    return 0


def cmd_peak_panels_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .peak_panels import build_peak_panels
    from .pooled_scatter_product import build_pooled_scatter_product

    for variant, path in build_peak_panels(
        context, build_pooled_scatter_product
    ).items():
        print(f"peak_panel_summary {variant}: {path}")
    return 0


def cmd_threshold_diagnostics_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .threshold_diagnostics_product import build_threshold_diagnostics_product

    path = build_threshold_diagnostics_product(context=context)
    print(f"threshold_diagnostics: {path}")
    return 0


def cmd_bootstrap_codon_entropy_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .bootstrap_codon_entropy_product import (
        build_bootstrap_codon_entropy_product,
    )

    path = build_bootstrap_codon_entropy_product(context=context)
    print(f"bootstrap_codon_entropy: {path}")
    return 0


def cmd_bootstrap_band_quantile_scan_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .bootstrap_band_quantile_scan_product import (
        build_bootstrap_band_quantile_scan_product,
    )

    path = build_bootstrap_band_quantile_scan_product(context=context)
    print(f"bootstrap_band_quantile_scan: {path}")
    return 0


def cmd_graph_tree_persistence_product(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from .graph_tree_persistence_product import (
        build_graph_tree_persistence_product,
    )

    path = build_graph_tree_persistence_product(context=context)
    print(f"graph_tree_persistence: {path}")
    return 0
