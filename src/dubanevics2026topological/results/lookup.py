"""Read-only result lookup with provenance-based compatibility checks.

Products are found by their manifests: ``output/<kind>/*/manifest.json`` and
``data/processed/<kind>/*/manifest.json``. A manifest is verified (artifact bytes,
sizes and its recomputed scientific identity) before a product is returned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubanevics2026topological.results.context import RunContext
from dubanevics2026topological.results.hashing import compute_scientific_hash
from dubanevics2026topological.results.labels import (
    ALLOWED_KEYS,
    ResultLabel,
    format_label,
    parse_label,
)
from dubanevics2026topological.results.manifest import (
    Manifest,
    ScientificHashCollisionError,
    read_manifest,
)


# Product types whose processed folder is named differently (data/processed/<folder>/<state>/).
PROCESSED_DIRECTORY = {
    "bootstrap_band_quantile_scan": "band_quantile_scan",
    "peak_panel_summary": "peak_panels",
}


class AmbiguousParentError(LookupError):
    """Raised when multiple stored products satisfy a lookup."""


class MissingProductError(LookupError):
    """Raised when a requested product is unavailable or unverified."""


@dataclass(frozen=True)
class CompatibilityReport:
    """Scientific compatibility is independent of byte and schema integrity."""

    status: str
    byte_integrity: bool
    schema_version: int | None
    checked: tuple[dict[str, Any], ...]
    unavailable: tuple[str, ...]
    differing: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class LoadedProduct:
    """A resolved product, its manifest, and its independent verification report."""

    product_type: str
    path: Path
    manifest: Manifest
    report: CompatibilityReport

    @property
    def artifacts(self) -> dict[str, Path]:
        """Resolved registered artifact paths."""
        return {name: self.path / name for name in self.manifest.artifacts}

    @property
    def label(self) -> str:
        """Canonical result label for this manifest."""
        params = {
            key: value
            for key, value in self.manifest.scientific_identity.items()
            if key in ALLOWED_KEYS
        }
        return format_label(
            self.product_type,
            {
                **params,
                **self.manifest.run_fields,
                "hash": self.manifest.short_hash,
            },
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _requested_identity(
    label: ResultLabel | None, expected: dict[str, Any] | None
) -> dict[str, Any] | None:
    if expected is not None:
        if "scientific_identity" in expected:
            return dict(expected["scientific_identity"])
        if any(
            key in expected
            for key in ("data_provenance", "parents", "scientific_id_sha256")
        ):
            return {}
        return expected
    return None if label is None else label.scientific_params


def _compatibility_report(
    observed: dict[str, Any],
    *,
    expected: dict[str, Any] | None,
    unavailable: tuple[str, ...],
    differing: dict[str, dict[str, Any]],
    byte_integrity: bool,
    schema_version: int | None,
    checks: tuple[dict[str, Any], ...] = (),
) -> CompatibilityReport:
    wanted = {} if expected is None else expected
    checked: list[dict[str, Any]] = []
    missing = list(unavailable)
    for key, value in wanted.items():
        if key not in observed:
            missing.append(key)
        else:
            checked.append({"field": key, "expected": value, "observed": observed[key]})
            if observed[key] != value:
                differing[key] = {"expected": value, "observed": observed[key]}
    status = "mismatch" if differing else "partial" if missing else "full"
    return CompatibilityReport(
        status,
        byte_integrity,
        schema_version,
        (*checks, *checked),
        tuple(dict.fromkeys(missing)),
        differing,
    )


def _manifest_product(path: Path) -> tuple[Manifest, CompatibilityReport]:
    manifest = read_manifest(path / "manifest.json")
    byte_ok = True
    missing: list[str] = []
    differing: dict[str, dict[str, Any]] = {}
    if not manifest.artifacts:
        byte_ok = False
        missing.append("manifest artifacts")
    for name, record in manifest.artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            byte_ok = False
            missing.append(f"unsafe artifact path:{name}")
            continue
        target = path / relative
        if not target.is_file():
            byte_ok = False
            missing.append(f"artifact:{name}")
            continue
        actual = _sha256(target)
        if actual != record.get("sha256"):
            byte_ok = False
        size = record.get("size_bytes", record.get("bytes"))
        if size is not None and target.stat().st_size != size:
            byte_ok = False
    report = _compatibility_report(
        manifest.scientific_identity,
        expected=manifest.scientific_identity,
        unavailable=tuple(missing),
        differing=differing,
        byte_integrity=byte_ok,
        schema_version=manifest.schema_version,
    )
    parent_ids = [
        str(parent["scientific_id_sha256"])
        for parent in manifest.parents
        if "scientific_id_sha256" in parent
    ]
    recomputed, _ = compute_scientific_hash(
        manifest.product_type,
        manifest.scientific_identity,
        parent_scientific_ids=parent_ids,
        input_data_sha256s=manifest.data_provenance,
        stage_code_digest=manifest.code_provenance.get("stage_code_digest"),
    )
    if recomputed != manifest.scientific_id_sha256:
        report = CompatibilityReport(
            "mismatch",
            report.byte_integrity,
            report.schema_version,
            report.checked,
            report.unavailable,
            {
                **report.differing,
                "scientific_id_sha256": {
                    "expected": manifest.scientific_id_sha256,
                    "observed": recomputed,
                },
            },
        )
    return manifest, report


def _matches(
    manifest: Manifest, label: ResultLabel, *, check_hash: bool = True
) -> bool:
    if manifest.product_type != label.product_name:
        return False
    identity = label.scientific_params
    return (
        all(
            manifest.scientific_identity.get(key) == value
            for key, value in identity.items()
        )
        and all(
            str(manifest.run_fields.get(key)) == str(label.params.get(key))
            for key in ("host", "run")
            if key in label.params
        )
        and (
            not check_hash
            or label.short_hash is None
            or manifest.short_hash == label.short_hash
        )
    )


def _matching_labels(
    candidates: list[LoadedProduct], label: ResultLabel
) -> list[LoadedProduct]:
    if label.short_hash is not None:
        prefix_digests = {
            candidate.manifest.scientific_id_sha256
            for candidate in candidates
            if candidate.manifest.short_hash == label.short_hash
        }
        if len(prefix_digests) > 1:
            raise ScientificHashCollisionError(
                f"Scientific short-hash collision for {label.short_hash}: "
                f"{sorted(prefix_digests)}"
            )
    identity_matches = [
        candidate
        for candidate in candidates
        if _matches(candidate.manifest, label, check_hash=False)
    ]
    if label.short_hash is None:
        return identity_matches
    prefix_matches = [
        candidate
        for candidate in identity_matches
        if candidate.manifest.short_hash == label.short_hash
    ]
    return prefix_matches


def _identity_subset(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(observed.get(key) == value for key, value in expected.items())


def _available_candidates(context: RunContext, product: str) -> list[LoadedProduct]:
    candidates: list[LoadedProduct] = []
    paths = set((context.output_root / product).glob("*/manifest.json"))
    processed_folder = PROCESSED_DIRECTORY.get(product, product)
    paths.update((context.processed_root / processed_folder).glob("*/manifest.json"))
    for manifest_path in sorted(paths):
        manifest, report = _manifest_product(manifest_path.parent)
        if manifest.product_type != product:
            continue
        candidates.append(
            LoadedProduct(product, manifest_path.parent, manifest, report)
        )
    # One product published under both roots (output/ and its processed copy) is one product.
    unique: dict[tuple[str, str, str, str, str], LoadedProduct] = {}
    for candidate in candidates:
        key = (
            candidate.manifest.scientific_id_sha256,
            str(sorted(candidate.manifest.artifacts.items())),
            str(sorted(candidate.manifest.run_fields.items())),
            str(sorted(candidate.manifest.data_provenance.items())),
            json.dumps(candidate.manifest.parents, sort_keys=True),
        )
        previous = unique.get(key)
        if previous is None or (
            candidate.report.status == "full"
            and candidate.report.byte_integrity
            and (previous.report.status != "full" or not previous.report.byte_integrity)
        ):
            unique[key] = candidate
    return list(unique.values())


def load(
    product: str,
    label: str | None = None,
    *,
    context: RunContext,
    expected: dict[str, Any] | None = None,
) -> LoadedProduct:
    """Load one exact product, or resolve an unambiguous one by its identity."""
    if label is not None:
        parsed = parse_label(label)
        if parsed.product_name != product:
            raise ValueError(
                f"Requested {product!r}, but label identifies {parsed.product_name!r}."
            )
        matches = _matching_labels(_available_candidates(context, product), parsed)
        if not matches:
            raise MissingProductError(
                f"No {product!r} product matches {label!r} (zero matches); "
                "run `just all` or `just fetch-data`."
            )
        if len(matches) > 1:
            raise AmbiguousParentError(
                f"Multiple products match {label!r}: "
                + ", ".join(str(item.path) for item in matches)
            )
        candidate = _apply_expected(
            matches[0], expected if expected is not None else parsed.scientific_params
        )
        if parsed.short_hash and candidate.manifest.short_hash != parsed.short_hash:
            raise MissingProductError(f"Short hash mismatch for {label!r}.")
        return candidate

    candidates = _available_candidates(context, product)
    expected_identity = _requested_identity(None, expected)
    if expected_identity is not None:
        candidates = [
            candidate
            for candidate in candidates
            if _identity_subset(
                candidate.manifest.scientific_identity, expected_identity
            )
        ]
    # When expected includes data_provenance/parents/scientific_id_sha256 (in addition
    # to or instead of scientific_identity), filter candidates on the observed fields
    # built exactly as _apply_expected does. This prevents AmbiguousParentError when
    # multiple products share sci identity but differ in provenance (e.g. after rebuild
    # with changed inputs).
    if expected is not None:
        extra = {
            k: expected[k]
            for k in ("data_provenance", "parents", "scientific_id_sha256")
            if k in expected
        }
        if extra:
            candidates = [
                c
                for c in candidates
                if _identity_subset(
                    {
                        **c.manifest.scientific_identity,
                        "data_provenance": c.manifest.data_provenance,
                        "parents": c.manifest.parents,
                        "scientific_id_sha256": c.manifest.scientific_id_sha256,
                    },
                    extra,
                )
            ]
    if not candidates:
        raise MissingProductError(
            f"No {product!r} product found for {expected!r}; "
            "run `just all` or `just fetch-data`."
        )
    if len(candidates) > 1:
        raise AmbiguousParentError(
            f"Multiple candidate {product!r} products: "
            + ", ".join(str(item.path) for item in candidates)
        )
    return _apply_expected(candidates[0], expected)


def _apply_expected(
    product: LoadedProduct, expected: dict[str, Any] | None
) -> LoadedProduct:
    if expected is None:
        return product
    expected_identity = _requested_identity(None, expected)
    observed = {
        **product.manifest.scientific_identity,
        "data_provenance": product.manifest.data_provenance,
        "parents": product.manifest.parents,
        "scientific_id_sha256": product.manifest.scientific_id_sha256,
    }
    expected_report: dict[str, Any] | None = expected_identity
    if "scientific_identity" in expected:
        expected_report = {
            **(expected_identity or {}),
            **{
                key: expected[key]
                for key in (
                    "data_provenance",
                    "parents",
                    "scientific_id_sha256",
                )
                if key in expected
            },
        }
    elif any(
        key in expected
        for key in ("data_provenance", "parents", "scientific_id_sha256")
    ):
        expected_report = {
            key: expected[key]
            for key in ("data_provenance", "parents", "scientific_id_sha256")
            if key in expected
        }
    report = _compatibility_report(
        observed,
        expected=expected_report,
        unavailable=product.report.unavailable,
        differing=dict(product.report.differing),
        byte_integrity=product.report.byte_integrity,
        schema_version=product.report.schema_version,
    )
    return LoadedProduct(
        product.product_type,
        product.path,
        product.manifest,
        report,
    )
