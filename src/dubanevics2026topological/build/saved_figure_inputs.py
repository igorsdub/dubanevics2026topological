"""Resolve reviewed tree layouts and populations for figure composition."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any

import polars as pl

from ..analysis import peak_uncertainty
from ..analysis.checkpoint import checkpoint_lock
from ..analysis.peak_uncertainty import _load_peak_checkpoint
from ..results import (
    MissingProductError,
    PresentationProducts,
    RunContext,
    load,
    processed,
)
from ..results.processed import state_for_selection
from ..threshold.functional_threshold import MEDIAN_REFLECTED, CandidateThreshold
from .pooled_scatter_product import _peak_inputs
from .threshold_diagnostics_product import _threshold_inputs
from .tree_layout_products import _tree_for_leaf


def _landscape_for_tree(products: PresentationProducts, tree: Any) -> Any:
    refs = [
        ref for ref in tree.manifest.parents if ref.get("product_type") == "landscape"
    ]
    if len(refs) != 1:
        raise ValueError("Nominal tree must identify exactly one landscape parent.")
    ref = refs[0]
    landscape = products.load("landscape", ref["label"])
    if (
        landscape.report.status != "full"
        or not landscape.report.byte_integrity
        or landscape.manifest.scientific_id_sha256 != ref["scientific_id_sha256"]
    ):
        raise ValueError("Tree landscape parent is missing or mismatched.")
    return landscape


def _check_fdr_landscape(landscape: Any, selection_identity: dict[str, Any]) -> None:
    configured = selection_identity.get("table_sha256")
    provenance = landscape.manifest.data_provenance or {}
    key = "classification_table"
    if key not in provenance:
        table = selection_identity.get("table")
        keys = [
            item
            for item in provenance
            if isinstance(item, str)
            and isinstance(table, str)
            and PurePosixPath(item).name == table
        ]
        if len(keys) != 1:
            raise ValueError(
                f"Reviewed landscape {landscape.label!r} classification table {table!r} "
                f"has {'zero' if not keys else 'multiple'} matching keys: {keys!r}; "
                f"configured {configured!r}, observed None."
            )
        key = keys[0]
    observed = provenance.get(key)
    if configured != observed:
        raise ValueError(
            f"Reviewed landscape {landscape.label!r} classification table {key!r} digest "
            f"mismatch: configured {configured!r}, observed {observed!r}."
        )


def _layout(products: PresentationProducts, tree: Any) -> tuple[Any, str]:
    product = products.load(
        "tree_layout",
        expected={"tree_scientific_id_sha256": tree.manifest.scientific_id_sha256},
    )
    return (
        products.tree_layout_view(product.label, tree=tree),
        product.manifest.scientific_id_sha256,
    )


def _direct_band_identity(tree: Any) -> str:
    """Read the verified band tree's unique direct-band parent identity."""
    refs = [
        ref
        for ref in tree.manifest.parents
        if ref.get("product_type") == "direct_band_identity"
    ]
    if len(refs) != 1:
        raise ValueError(
            "Saved band tree must identify exactly one direct-band parent."
        )
    digest = refs[0].get("scientific_id_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or tree.manifest.scientific_identity.get("direct_band_identity_sha256")
        != digest
    ):
        raise ValueError("Saved band tree has an invalid direct-band identity.")
    return digest


def load_saved_figure_inputs(
    context: RunContext, selection_identity: dict[str, Any], *, include_band: bool,
    active_band_identity: str | None = None,
) -> dict[str, Any]:
    """Resolve explicitly by selection identity, never by a lookup failure."""
    candidate = state_for_selection(selection_identity, include_band)
    if candidate == "legacy_mixture":
        # Pending active-config product stages; no product lookup is attempted for
        # tree layouts etc.
        return {
            "mode": "legacy_mixture",
            "product_ids": {"selection": selection_identity},
        }
    products = PresentationProducts(context)
    result: dict[str, Any] = {
        "mode": "saved",
        "layouts": {},
        "populations": {},
        "product_ids": {},
    }
    for key, name in (
        ("full", "full"),
        ("article", "published"),
        ("candidate", candidate),
    ):
        tree = _tree_for_leaf(name, context)
        landscape = _landscape_for_tree(products, tree)
        if key == "candidate" and candidate == "reflected_by":
            _check_fdr_landscape(landscape, selection_identity)
        try:
            view, layout_id = _layout(products, tree)
        except MissingProductError as exc:
            raise MissingProductError(
                f"Missing layout for {name}; run `tree-layout-products` first."
            ) from exc
        result["layouts"][key] = view
        result["populations"][key] = products.landscape_population(landscape)
        result["product_ids"][key] = tree.manifest.scientific_id_sha256
        result["product_ids"][f"{key}_layout"] = layout_id
        if key == "candidate" and include_band:
            leaves = processed.load("leaves", name, context)

            result["nominal_hills"] = pl.read_parquet(
                leaves.artifacts["leaves.parquet"]
            ).select("peak_dna_sequence", "member_count")
            try:
                result["annotations"] = products.tree_annotations(tree=tree)
                annotations = products.load(
                    "tree_annotations",
                    expected={
                        "tree_scientific_id_sha256": tree.manifest.scientific_id_sha256
                    },
                )
            except MissingProductError as exc:
                raise MissingProductError(
                    f"Missing annotations for {name}; run `tree-layout-products` first."
                ) from exc
            result["product_ids"]["candidate_annotations"] = (
                annotations.manifest.scientific_id_sha256
            )
    if not include_band:
        return result
    band = _tree_for_leaf(f"{candidate}_band", context)
    if (
        active_band_identity is not None
        and _direct_band_identity(band) != active_band_identity
    ):
        raise ValueError(
            "Active bootstrap band identity differs from the saved band tree."
        )
    result["direct_band_identity"] = _direct_band_identity(band)
    if not any(
        ref.get("product_type") == "tree"
        and ref.get("scientific_id_sha256") == result["product_ids"]["candidate"]
        for ref in band.manifest.parents
    ):
        raise ValueError("Band tree does not descend from the selected nominal tree.")
    try:
        result["layouts"]["band"], band_layout_id = _layout(products, band)
    except MissingProductError as exc:
        raise MissingProductError(
            f"Missing layout for {candidate}_band; run `tree-layout-products` first."
        ) from exc
    result["product_ids"]["band"] = band.manifest.scientific_id_sha256
    result["product_ids"]["band_layout"] = band_layout_id
    return result


def load_pooled_scatter_product(
    context: RunContext, *, expected: dict[str, Any] | None = None,
    expected_checkpoint_fingerprint: str | None = None,
) -> Any:
    """Load the pooled_scatter product (for callers that also need its manifest/id)."""
    # Compute current inputs FIRST, under the same lock/validation/canonical as the
    # stage (and as before), so we can pass data_provenance in the results expected
    # to filter precisely and avoid Ambiguous when sci id is shared.
    peak_dir, _ = peak_uncertainty.cache_paths(context)
    with checkpoint_lock(peak_dir, "peak uncertainty"):
        try:
            checkpoint = _load_peak_checkpoint(peak_dir, context=context)
        except (FileNotFoundError, ValueError) as exc:
            if expected_checkpoint_fingerprint is None:
                raise
            raise MissingProductError(
                "Missing or invalid FDR peak checkpoint; run `just fdr-pooled-scatter-product`."
            ) from exc
        except RuntimeError as exc:
            if expected_checkpoint_fingerprint is None:
                raise
            raise RuntimeError(
                "FDR peak checkpoint refresh is pending; run `just fdr-pooled-scatter-product`."
            ) from exc
        if expected_checkpoint_fingerprint is not None and (
            checkpoint["provenance"].get("fingerprint") != expected_checkpoint_fingerprint
        ):
            raise MissingProductError(
                "FDR peak checkpoint fingerprint mismatch; run `just fdr-pooled-scatter-product`."
            )
        current_inputs = _peak_inputs(context)

    # Build sci_identity from tolerances (passed expected or current settings) and
    # include data_provenance for disambiguation on rebuild.
    tols = expected or {}
    id_tol = tols.get("identity_tolerance")
    se_tol = tols.get("standard_error_tolerance")
    if id_tol is None or se_tol is None:
        from ..analysis import load_settings as _load_settings

        s = _load_settings(context=context)
        id_tol = id_tol or s["analysis"]["sequence_growth_rates"]["identity_tolerance"]
        se_tol = (
            se_tol
            or s["analysis"]["peak_uncertainty_scatter"]["standard_error_tolerance"]
        )
    sci = {
        "identity_tolerance": float(id_tol),
        "standard_error_tolerance": float(se_tol),
    }
    try:
        product = load(
            "pooled_scatter",
            context=context,
            expected={"scientific_identity": sci, "data_provenance": current_inputs},
        )
    except MissingProductError as exc:
        raise MissingProductError(
            "Missing or stale pooled_scatter product; run `pooled-scatter-product` first."
        ) from exc
    if product.report.status != "full" or not product.report.byte_integrity:
        raise MissingProductError(
            "pooled_scatter product is not fully verified; run `pooled-scatter-product`."
        )
    # Belt-and-braces explicit check (filter should have selected the right one or zero).
    recorded = product.manifest.data_provenance
    if current_inputs != recorded:
        raise MissingProductError(
            "pooled_scatter product is stale; run `pooled-scatter-product`."
        )
    # Internal consistency only (self-contained product): exactly one growth_rates
    # parent with a 64-hex ID. The canonical parent identity is verified at build time.
    # Do not load the growth_rates archive here.
    gr_parents = [
        p for p in product.manifest.parents if p.get("product_type") == "growth_rates"
    ]
    if len(gr_parents) != 1:
        raise ValueError(
            "pooled_scatter product must have exactly one growth_rates parent."
        )
    gparent = gr_parents[0]
    sid = gparent.get("scientific_id_sha256")
    if not (isinstance(sid, str) and re.fullmatch(r"[0-9a-f]{64}", sid)):
        raise ValueError("growth_rates parent must have a 64-hex scientific_id_sha256.")
    return product


def _load_pooled_scatter_payload(
    context: RunContext, *, product: Any | None = None
) -> dict[str, Any]:
    """Load the pooled_scatter product (self-contained tables) via results.load.

    If absent the error names the build subcommand with no fallback.
    The caller (peak figure) verifies parent via the product's manifest.
    Pass a pre-loaded `product` to avoid a second load.
    """
    if product is None:
        product = load_pooled_scatter_product(context)
    tables: dict[str, pl.DataFrame] = {}
    for name, path in product.artifacts.items():
        if name.endswith(".parquet"):
            tables[name.removesuffix(".parquet")] = pl.read_parquet(path)
    diagnostics: dict[str, Any] = {}
    if "diagnostics.json" in product.artifacts:
        diagnostics = json.loads(
            product.artifacts["diagnostics.json"].read_text(encoding="utf-8")
        )
    return {"tables": tables, "diagnostics": diagnostics}


def load_threshold_diagnostics_product(
    context: RunContext, settings: dict[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    """Load the precomputed threshold diagnostics (candidate/reflected T and saddles).

    Selects the product for the current variant using the declaration in settings.
    """
    if settings is None:
        from ..analysis import load_settings as _load_settings

        settings = _load_settings(context=context)
    from ..threshold.functional_threshold import (
        configured_declaration,
        declaration_identity,
    )

    declaration = declaration_identity(configured_declaration(settings))
    hashes, _, _ = _threshold_inputs(context, settings)
    # Use nested scientific_identity (as _requested_identity expects) + data_provenance
    # so unpinned lookup filters on both; stale (same decl/summary but different data)
    # yields zero matches -> wrapped Missing.
    sci = {"candidate_functional_threshold_declaration": declaration}
    if "fdr_threshold_summary" in hashes:
        sci["classification_summary_sha256"] = hashes["fdr_threshold_summary"]
    elif "functional_classification_summary" in hashes:
        sci["classification_summary_sha256"] = hashes[
            "functional_classification_summary"
        ]
    try:
        product = load(
            "threshold_diagnostics",
            context=context,
            expected={"scientific_identity": sci, "data_provenance": hashes},
        )
    except MissingProductError as exc:
        raise MissingProductError(
            "Missing or stale threshold_diagnostics product; run `threshold-diagnostics-product` first."
        ) from exc
    if product.report.status != "full" or not product.report.byte_integrity:
        raise MissingProductError(
            "threshold_diagnostics product is not fully verified; "
            "run `threshold-diagnostics-product`."
        )
    if "threshold_diagnostics.json" not in product.artifacts:
        raise MissingProductError(
            "threshold_diagnostics product missing threshold_diagnostics.json; "
            "run `threshold-diagnostics-product`."
        )
    # staleness
    recorded = product.manifest.data_provenance
    if hashes != recorded:
        raise MissingProductError(
            "threshold_diagnostics product is stale; run `threshold-diagnostics-product`."
        )
    diags = json.loads(
        product.artifacts["threshold_diagnostics.json"].read_text(encoding="utf-8")
    )
    c = diags["candidate"]
    rule = c["rule"]
    has_reflected = "reflected" in diags and diags.get("reflected") is not None
    if (rule == MEDIAN_REFLECTED) != has_reflected:
        raise MissingProductError(
            f"threshold_diagnostics product is inconsistent: rule={rule} "
            f"but reflected is {'present' if has_reflected else 'absent'}; "
            "run `threshold-diagnostics-product`."
        )
    diags = dict(diags)
    diags["candidate"] = CandidateThreshold(
        rule=c["rule"],
        decimals=c["decimals"],
        median=c["median"],
        global_minimum=c["global_minimum"],
        median_gap=c["median_gap"],
        unrounded=c["unrounded"],
        threshold=c["threshold"],
    )
    if has_reflected:
        r = diags["reflected"]
        diags["reflected"] = CandidateThreshold(
            rule=r["rule"],
            decimals=r["decimals"],
            median=r["median"],
            global_minimum=r["global_minimum"],
            median_gap=r["median_gap"],
            unrounded=r["unrounded"],
            threshold=r["threshold"],
        )
    else:
        diags.pop("reflected", None)
    return diags, product.manifest.scientific_id_sha256


def load_bootstrap_codon_entropy_product(
    context: RunContext,
) -> dict[str, Any]:
    """Load the precomputed bootstrap codon entropy product.

    Computes fresh input hashes and loads with expected data_provenance.
    Raises MissingProductError (with a clear message naming the build step)
    if missing, non-full, not byte-integral, or stale (hash mismatch).
    The caller must not fall back to render-time science.
    """
    from ..analysis import load_settings
    from ..landscape.selection import resolve_candidate_selection
    from ..results import MissingProductError
    from .bootstrap_codon_entropy_product import (
        _codon_entropy_input_hashes,
        _product_identity,
    )

    hashes = _codon_entropy_input_hashes(context)
    settings = load_settings(context=context)
    selection = resolve_candidate_selection(settings, context=context)
    identity = _product_identity(settings, selection)

    try:
        product = load(
            "bootstrap_codon_entropy",
            context=context,
            expected={"data_provenance": hashes, "scientific_identity": identity},
        )
    except MissingProductError as exc:
        raise MissingProductError(
            "Missing or stale bootstrap_codon_entropy product; "
            "run `bootstrap-codon-entropy-product` first."
        ) from exc

    if product.report.status != "full" or not product.report.byte_integrity:
        raise MissingProductError(
            "bootstrap_codon_entropy product is not fully verified; "
            "run `bootstrap-codon-entropy-product`."
        )

    # Staleness check (defense in depth, in case load expected filter was bypassed)
    recorded = product.manifest.data_provenance or {}
    if hashes != recorded:
        raise MissingProductError(
            "bootstrap_codon_entropy product is stale; "
            "run `bootstrap-codon-entropy-product`."
        )

    if any(
        product.manifest.scientific_identity.get(k) != v for k, v in identity.items()
    ):
        raise MissingProductError(
            "bootstrap_codon_entropy product has stale settings or selection; "
            "run `bootstrap-codon-entropy-product`."
        )

    tables: dict[str, pl.DataFrame] = {}
    for name, path in product.artifacts.items():
        if name.endswith(".parquet"):
            tables[name.removesuffix(".parquet")] = pl.read_parquet(path)

    metadata: dict[str, Any] = {}
    if "metadata.json" in product.artifacts:
        metadata = json.loads(
            product.artifacts["metadata.json"].read_text(encoding="utf-8")
        )
    # Also expose the raw input hashes for fast staleness checks.
    if "input_data_sha256s.json" in product.artifacts:
        metadata["_input_data_sha256s"] = json.loads(
            product.artifacts["input_data_sha256s.json"].read_text(encoding="utf-8")
        )

    return {
        "populations_core": tables.get("populations_core"),
        "populations_kem": tables.get("populations_kem"),
        "populations_tcl": tables.get("populations_tcl"),
        "entropy_sweeps": tables.get("entropy_sweeps"),
        "node_table": tables.get("node_table"),
        "edge_table": tables.get("edge_table"),
        "edge_segment_table": tables.get("edge_segment_table"),
        "ridge_edges": tables.get("ridge_edges"),
        "stop_points": tables.get("stop_points"),
        "wild_type_marker": tables.get("wild_type_marker"),
        "metadata": metadata,
        "product": product,
    }


def load_bootstrap_band_quantile_scan_product(
    context: RunContext,
    *,
    settings: dict[str, Any],
    bootstrap: Any,
) -> dict[str, Any]:
    """Load the saved band quantile scan and survival tables, failing closed.

    Fresh input hashes and the scientific identity (grid step, band quantile and
    bootstrap fingerprint of the loaded checkpoint) must match the product, which
    must be full and byte-intact; otherwise ``MissingProductError``. The caller
    must not fall back to render-time science.
    """
    from .bootstrap_band_quantile_scan_product import (
        _band_scan_input_hashes,
        _identity,
    )

    hashes = _band_scan_input_hashes(context)
    identity = _identity(settings, bootstrap)
    try:
        product = load(
            "bootstrap_band_quantile_scan",
            context=context,
            expected={"data_provenance": hashes, "scientific_identity": identity},
        )
    except MissingProductError as exc:
        raise MissingProductError(
            "Missing or stale bootstrap_band_quantile_scan product; "
            "run `bootstrap-band-quantile-scan-product` first."
        ) from exc
    if product.report.status != "full" or not product.report.byte_integrity:
        raise MissingProductError(
            "bootstrap_band_quantile_scan product is not fully verified; "
            "run `bootstrap-band-quantile-scan-product`."
        )
    if (product.manifest.data_provenance or {}) != hashes:
        raise MissingProductError(
            "bootstrap_band_quantile_scan product is stale; "
            "run `bootstrap-band-quantile-scan-product`."
        )
    for name in ("scan.parquet", "survival.parquet", "metadata.json"):
        if name not in product.artifacts:
            raise MissingProductError(
                f"bootstrap_band_quantile_scan product missing {name}; "
                "run `bootstrap-band-quantile-scan-product`."
            )
    return {
        "scan": pl.read_parquet(product.artifacts["scan.parquet"]),
        "survival": pl.read_parquet(product.artifacts["survival.parquet"]),
        "metadata": json.loads(
            product.artifacts["metadata.json"].read_text(encoding="utf-8")
        ),
        "product": product,
    }


def load_graph_tree_persistence_product(
    context: RunContext | None = None, *, settings: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load the saved graph/tree/persistence construction, failing closed.

    The scientific identity (settings and layout order) and the raw-byte hashes of
    the producing code must match; otherwise ``MissingProductError``. Nothing is
    recomputed. Returns ``{"construction": ..., "product": ...}``.
    """
    from ..analysis import load_settings as _load_settings
    from .graph_tree_persistence_product import (
        ARTIFACT_NAMES,
        _construction_from_artifacts,
        _identity,
        _input_hashes,
    )

    context = context or RunContext()
    if settings is None:
        settings = _load_settings(context=context)
    hashes = _input_hashes()
    try:
        product = load(
            "graph_tree_persistence",
            context=context,
            expected={
                "scientific_identity": _identity(settings),
                "data_provenance": hashes,
            },
        )
    except MissingProductError as exc:
        raise MissingProductError(
            "Missing or stale graph_tree_persistence product; "
            "run `graph-tree-persistence-product` first."
        ) from exc
    if product.report.status != "full" or not product.report.byte_integrity:
        raise MissingProductError(
            "graph_tree_persistence product is not fully verified; "
            "run `graph-tree-persistence-product`."
        )
    if (product.manifest.data_provenance or {}) != hashes:
        raise MissingProductError(
            "graph_tree_persistence product is stale; "
            "run `graph-tree-persistence-product`."
        )
    for name in ARTIFACT_NAMES:
        if name not in product.artifacts:
            raise MissingProductError(
                f"graph_tree_persistence product missing {name}; "
                "run `graph-tree-persistence-product`."
            )
    try:
        construction = _construction_from_artifacts(product.artifacts)
    except Exception as exc:
        raise MissingProductError(
            "graph_tree_persistence product has unreadable artifacts; "
            "run `graph-tree-persistence-product`."
        ) from exc
    return {"construction": construction, "product": product}
