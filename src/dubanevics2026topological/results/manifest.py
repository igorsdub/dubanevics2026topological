"""Product manifest read/write and schema validation per ADR 0016 and Spec 1.

Every generated product directory contains an immutable ``manifest.json`` documenting:
- Schema version (currently 1)
- Product type
- Full-precision scientific identity mapping
- Full 64-hex ``scientific_id_sha256``
- Upstream parent product digests
- Input raw data checksums
- Artifact relative file paths, digests, and byte sizes
- Execution run fields (``host``, ``run``)
- Code provenance (commit, dirty, branch)
- Software environment (python_version, platform, uv_lock_sha256)
- UTC creation timestamp (ISO 8601)
- Optional remote bucket storage references
- Evidence availability flags (checkpoint, distances, chunks)

Collision policy:
Encountering identical 8-hex prefixes with differing full digests raises an explicit
``ScientificHashCollisionError``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CURRENT_SCHEMA_VERSION = 1
_HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_EVIDENCE_KEYS = frozenset(
    {"checkpoint_present", "distances_present", "chunks_present"}
)


class ScientificHashCollisionError(ValueError):
    """Raised when two distinct scientific products produce colliding 8-hex short hashes."""


@dataclass(frozen=True)
class Manifest:
    """Immutable product manifest representation per Spec 1 Section 2.D."""

    product_type: str
    scientific_identity: dict[str, Any]
    scientific_id_sha256: str
    parents: list[dict[str, str]]
    data_provenance: dict[str, str]
    artifacts: dict[str, dict[str, Any]]
    run_fields: dict[str, Any]
    code_provenance: dict[str, Any]
    environment: dict[str, Any]
    created_at: str
    evidence_availability: dict[str, Any]
    storage_references: dict[str, Any] | None = None
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError(f"Invalid schema_version: {self.schema_version}")

        if not isinstance(self.scientific_id_sha256, str) or not _HEX_64_PATTERN.match(
            self.scientific_id_sha256
        ):
            raise ValueError(
                f"scientific_id_sha256 must be exactly 64 lowercase hexadecimal characters, "
                f"got '{self.scientific_id_sha256}'"
            )

        if not isinstance(self.evidence_availability, Mapping):
            raise TypeError(
                f"evidence_availability must be a mapping, got {type(self.evidence_availability).__name__}"
            )

        missing_evidence = _REQUIRED_EVIDENCE_KEYS - set(
            self.evidence_availability.keys()
        )
        if missing_evidence:
            raise ValueError(
                f"evidence_availability missing required keys: {', '.join(sorted(missing_evidence))}"
            )

        for k in _REQUIRED_EVIDENCE_KEYS:
            val = self.evidence_availability[k]
            if not isinstance(val, bool):
                raise TypeError(
                    f"evidence_availability['{k}'] must be a boolean, got {type(val).__name__} ({val!r})"
                )

    @property
    def short_hash(self) -> str:
        """First 8 hexadecimal characters of the scientific digest."""
        return self.scientific_id_sha256[:8]

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a serializable dictionary."""
        d = asdict(self)
        if self.storage_references is None:
            d["storage_references"] = None
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Instantiate a Manifest from a dictionary."""
        required_keys = (
            "schema_version",
            "product_type",
            "scientific_identity",
            "scientific_id_sha256",
            "parents",
            "data_provenance",
            "artifacts",
            "run_fields",
            "code_provenance",
            "environment",
            "created_at",
            "evidence_availability",
        )
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Manifest missing required keys: {', '.join(missing)}")

        return cls(
            schema_version=int(data["schema_version"]),
            product_type=str(data["product_type"]),
            scientific_identity=dict(data["scientific_identity"]),
            scientific_id_sha256=str(data["scientific_id_sha256"]),
            parents=[dict(p) for p in data["parents"]],
            data_provenance=dict(data["data_provenance"]),
            artifacts={k: dict(v) for k, v in data["artifacts"].items()},
            run_fields=dict(data["run_fields"]),
            code_provenance=dict(data["code_provenance"]),
            environment=dict(data["environment"]),
            created_at=str(data["created_at"]),
            evidence_availability=dict(data["evidence_availability"]),
            storage_references=dict(data["storage_references"])
            if data.get("storage_references")
            else None,
        )

    @classmethod
    def from_json(cls, json_str: str) -> Manifest:
        """Instantiate a Manifest from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def write(
        self,
        target_path: Path | str,
        overwrite: bool = False,
        check_collision: bool = True,
    ) -> Path:
        """Write manifest.json to file, checking for collision if target exists."""
        return write_manifest(
            self,
            target_path=target_path,
            overwrite=overwrite,
            check_collision=check_collision,
        )


def read_manifest(path: Path | str) -> Manifest:
    """Read and validate a manifest from a file path or directory."""
    target = Path(path)
    if target.is_dir():
        target = target / "manifest.json"

    if not target.exists():
        raise FileNotFoundError(f"Manifest file not found: {target}")

    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return Manifest.from_dict(data)


def check_hash_collision(existing: Manifest, candidate: Manifest) -> None:
    """Verify that identical 8-hex short hashes share the exact same 64-hex digest."""
    if (
        existing.short_hash == candidate.short_hash
        and existing.scientific_id_sha256 != candidate.scientific_id_sha256
    ):
        raise ScientificHashCollisionError(
            f"Scientific hash collision detected for 8-hex prefix '{existing.short_hash}': "
            f"existing digest '{existing.scientific_id_sha256}' differs from "
            f"candidate digest '{candidate.scientific_id_sha256}'."
        )


def write_manifest(
    manifest: Manifest,
    target_path: Path | str,
    overwrite: bool = False,
    check_collision: bool = True,
) -> Path:
    """Write a manifest to disk.

    Args:
        manifest: The manifest instance to write.
        target_path: File path or parent directory (where manifest.json will be created).
        overwrite: Whether to overwrite an existing manifest file.
        check_collision: If True, check existing file and sibling runs in the product folder
                         for colliding 8-hex short hashes with differing full digests.

    Returns:
        Path to the written manifest file.
    """
    path = Path(target_path)
    if path.is_dir() or path.suffix != ".json":
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "manifest.json"
        product_dir = path.parent
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_path = path
        if path.name == "manifest.json":
            product_dir = path.parent.parent
        else:
            product_dir = path.parent

    resolved_target = file_path.resolve()

    if check_collision:
        # 1. Check existing target file if present
        if file_path.exists():
            existing = read_manifest(file_path)
            check_hash_collision(existing, manifest)

        # 2. Check sibling runs in product folder:
        # scan <product_dir>/*/manifest.json and single-file <product_dir>/*.json
        if product_dir.exists() and product_dir.is_dir():
            candidates: set[Path] = set()
            candidates.update(product_dir.glob("*/manifest.json"))
            candidates.update(product_dir.glob("*.json"))

            for candidate in sorted(candidates):
                if candidate.resolve() == resolved_target:
                    continue
                # A corrupt existing manifest raises, not passed!
                sibling = read_manifest(candidate)
                check_hash_collision(sibling, manifest)

    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest file already exists at '{file_path}'. Set overwrite=True to overwrite."
        )

    payload = manifest.to_json(indent=2) + "\n"
    file_path.write_text(payload, encoding="utf-8")
    return file_path
