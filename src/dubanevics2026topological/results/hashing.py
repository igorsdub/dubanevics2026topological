"""Scientific identity and hash calculation per ADR 0016 and Spec 1.

The scientific identity of any product is governed by a cryptographic digest
(``scientific_id_sha256``), the SHA-256 hash of a canonical JSON document over:
- ``product_type``
- ``scientific_parameters``
- ``parent_scientific_ids``
- ``input_data_sha256s``
- (optional) ``stage_code_digest`` (non-bootstrap stages only)

Execution run fields (``host``, ``run``), timestamps (``created_at``), git stamp / code
provenance, environment, artifacts, and remote storage references are strictly excluded.
The label prefix ``hash=`` uses the first 8 hex characters of ``scientific_id_sha256``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

EXCLUDED_FROM_SCIENTIFIC_HASH = frozenset(
    {
        "artifacts",
        "code_provenance",
        "context",
        "created_at",
        "environment",
        "evidence_availability",
        "git_stamp",
        "hash",
        "host",
        "run",
        "run_fields",
        "schema_version",
        "scientific_id_sha256",
        "storage_references",
    }
)


def _canonicalize_value(value: Any) -> Any:
    """Recursively canonicalize values for JSON representation."""
    if isinstance(value, Mapping):
        return {k: _canonicalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite float value '{value}' is not allowed in canonical JSON"
            )
        # Normalize negative zero to positive zero
        if value == 0.0:
            return 0.0
        return value
    return value


def canonical_json(document: Mapping[str, Any]) -> str:
    """Serialize a document to canonical JSON.

    Properties:
    - Keys sorted recursively at all depths.
    - Compact separators (',', ':') with no extraneous whitespace.
    - UTF-8 encoding (no ascii escaping for unicode).
    - Floats formatted in shortest round-trip representation (NaN and inf rejected).
    """
    clean_doc = _canonicalize_value(document)
    return json.dumps(
        clean_doc,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def build_identity_document(
    product_type: str,
    scientific_parameters: Mapping[str, Any],
    parent_scientific_ids: Sequence[str] | None = None,
    input_data_sha256s: Mapping[str, str] | None = None,
    stage_code_digest: str | None = None,
) -> dict[str, Any]:
    """Construct the canonical identity document for a scientific product.

    Excluded fields (run, environment, timestamps, storage) are omitted.
    Stage-irrelevant fields should be omitted from ``scientific_parameters`` prior to
    calling or filtered here.
    """
    # Clean scientific parameters: filter out any excluded execution fields if present
    clean_params = {
        k: v
        for k, v in scientific_parameters.items()
        if k not in EXCLUDED_FROM_SCIENTIFIC_HASH
    }

    doc: dict[str, Any] = {
        "input_data_sha256s": dict(sorted((input_data_sha256s or {}).items())),
        "parent_scientific_ids": list(parent_scientific_ids or []),
        "product_type": str(product_type),
        "scientific_parameters": clean_params,
    }
    if stage_code_digest is not None:
        doc["stage_code_digest"] = str(stage_code_digest)
    return doc


def compute_scientific_hash(
    product_type: str,
    scientific_parameters: Mapping[str, Any],
    parent_scientific_ids: Sequence[str] | None = None,
    input_data_sha256s: Mapping[str, str] | None = None,
    stage_code_digest: str | None = None,
) -> tuple[str, str]:
    """Compute the full 64-hex ``scientific_id_sha256`` and 8-hex ``short_hash``.

    Returns:
        (scientific_id_sha256, short_hash)
    """
    doc = build_identity_document(
        product_type=product_type,
        scientific_parameters=scientific_parameters,
        parent_scientific_ids=parent_scientific_ids,
        input_data_sha256s=input_data_sha256s,
        stage_code_digest=stage_code_digest,
    )
    payload = canonical_json(doc).encode("utf-8")
    full_hash = hashlib.sha256(payload).hexdigest().lower()
    return full_hash, full_hash[:8]
