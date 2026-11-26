"""Results module: labels, scientific hash, manifests, and directory documentation."""

from __future__ import annotations

from dubanevics2026topological.results.context import (
    RunContext,
    add_run_context_arguments,
    parse_run_context,
    serialize_run_context,
)
from dubanevics2026topological.results.hashing import (
    EXCLUDED_FROM_SCIENTIFIC_HASH,
    build_identity_document,
    canonical_json,
    compute_scientific_hash,
)
from dubanevics2026topological.results.labels import (
    ALLOWED_KEYS,
    ALLOWED_STYLE_VALUES,
    ALLOWED_THR_VALUES,
    ALLOWED_TREE_VALUES,
    PIPELINE_KEY_ORDER,
    STANDARD_PRODUCT_TYPES,
    MalformedLabelError,
    ResultLabel,
    format_label,
    parse_label,
)
from dubanevics2026topological.results.lookup import (
    AmbiguousParentError,
    CompatibilityReport,
    LoadedProduct,
    MissingProductError,
    load,
)
from dubanevics2026topological.results.manifest import (
    Manifest,
    ScientificHashCollisionError,
    read_manifest,
    write_manifest,
)
from dubanevics2026topological.results.presentation import PresentationProducts
from dubanevics2026topological.results.readme import (
    generate_product_readme,
    generate_root_readme,
    write_product_readme,
    write_root_readme,
)

__all__ = [
    "ALLOWED_KEYS",
    "ALLOWED_STYLE_VALUES",
    "ALLOWED_THR_VALUES",
    "ALLOWED_TREE_VALUES",
    "EXCLUDED_FROM_SCIENTIFIC_HASH",
    "PIPELINE_KEY_ORDER",
    "STANDARD_PRODUCT_TYPES",
    "AmbiguousParentError",
    "CompatibilityReport",
    "LoadedProduct",
    "MalformedLabelError",
    "Manifest",
    "MissingProductError",
    "PresentationProducts",
    "ResultLabel",
    "RunContext",
    "ScientificHashCollisionError",
    "add_run_context_arguments",
    "build_identity_document",
    "canonical_json",
    "compute_scientific_hash",
    "format_label",
    "generate_product_readme",
    "generate_root_readme",
    "load",
    "parse_label",
    "parse_run_context",
    "read_manifest",
    "serialize_run_context",
    "write_manifest",
    "write_product_readme",
    "write_root_readme",
]
