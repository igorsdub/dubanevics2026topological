"""Scientific identities for reusable bootstrap draws and derived band trees."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dubanevics2026topological.results.hashing import compute_scientific_hash


def checkpoint_draw_identity(
    fingerprint: Mapping[str, Any], *, seed: int, draw_ids: Iterable[int]
) -> tuple[dict[str, Any], str]:
    """Build the canonical draw identity from a checkpoint fingerprint."""
    from ..results.hashing import canonical_json
    from .checkpoint_identity import scientific_fingerprint

    scientific = scientific_fingerprint(fingerprint)
    selected_graph = scientific.get("candidate_selection")
    datasets = scientific.get("datasets")
    if not selected_graph or not datasets or "noise_model" not in scientific["config"]:
        raise ValueError(
            "Checkpoint lacks selected graph, fitness inputs or noise model."
        )
    graph_sha256 = hashlib.sha256(
        canonical_json(
            {
                "candidate_selection": selected_graph,
                "candidate_functional_threshold": scientific.get(
                    "candidate_functional_threshold"
                ),
            }
        ).encode("utf-8")
    ).hexdigest()
    return draw_identity(
        selected_graph_sha256=graph_sha256,
        growth_identity=scientific.get("growth"),
        fitness_input_sha256s=datasets,
        perturbation_method=f"normal-{scientific['config']['noise_model']}",
        seed=seed,
        draw_ids=draw_ids,
    )


def checkpoint_method(
    summary: Mapping[str, Any], provenance: Mapping[str, Any] | None = None
) -> str:
    """Identify historical or direct-band semantics from schema and method.

    Schema 6 predates the optional method marker in some archives. Schema 7
    requires the direct-band marker; conflicting or unknown identities fail closed.
    """
    summary_fingerprint = summary.get("fingerprint")
    if not isinstance(summary_fingerprint, Mapping):
        raise TypeError("Bootstrap summary lacks its fingerprint mapping.")
    schema = summary_fingerprint.get("schema")
    if provenance and provenance.get("fingerprint"):
        record_fingerprint = provenance["fingerprint"]
        if (
            not isinstance(record_fingerprint, Mapping)
            or record_fingerprint.get("schema") != schema
        ):
            raise ValueError("Bootstrap summary and record schemas disagree.")
    if type(schema) is not int or schema not in (6, 7):
        raise ValueError(f"Unsupported bootstrap checkpoint schema: {schema!r}.")
    identities = summary.get("scientific_identities")
    method = None
    if identities is not None:
        try:
            method = identities["band"]["scientific_identity"]["simplification_method"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Bootstrap band identity lacks its simplification method."
            ) from error
    expected = "band_after_core" if schema == 6 else "direct_band"
    if method != expected and not (schema == 6 and method is None):
        raise ValueError(
            f"Bootstrap schema {schema} conflicts with simplification method {method!r}."
        )
    return expected


def _merge_identity_fields(*parts: Mapping[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for part in parts:
        collision = identity.keys() & part.keys()
        if collision:
            raise ValueError(
                "Scientific identity fields collide: " + ", ".join(sorted(collision))
            )
        identity.update(part)
    return identity


def draw_identity(
    *,
    selected_graph_sha256: str,
    growth_identity: Mapping[str, Any] | None,
    fitness_input_sha256s: Mapping[str, str],
    perturbation_method: str,
    seed: int,
    draw_ids: Iterable[int],
) -> tuple[dict[str, Any], str]:
    """Return band-independent draw identity fields and their full SHA256 ID.

    Code, environment and presentation settings are intentionally absent. The
    draw index set is explicit so partial/sliced archives cannot masquerade as a
    complete draw product.
    """
    ids = tuple(draw_ids)
    if not ids or any(
        isinstance(value, bool) or not isinstance(value, int) for value in ids
    ):
        raise ValueError("draw_ids must contain one or more integer draw IDs")
    if tuple(sorted(set(ids))) != ids:
        raise ValueError("draw_ids must be unique and in ascending order")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not perturbation_method:
        raise ValueError("perturbation_method must not be empty")
    inputs = dict(fitness_input_sha256s)
    parameters = _merge_identity_fields(
        {
            "selected_graph_sha256": selected_graph_sha256,
            "growth_identity": growth_identity,
            "perturbation_method": perturbation_method,
            "seed": seed,
            "draw_ids": list(ids),
        },
        {"fitness_input_sha256s": inputs},
    )
    digest, _ = compute_scientific_hash(
        "bootstrap", parameters, input_data_sha256s=inputs
    )
    return parameters, digest


def band_identity(
    *,
    draw_scientific_id_sha256: str,
    bandq: float,
) -> tuple[dict[str, Any], str]:
    """Return the quantile-specific derived identity, parented by draw evidence."""
    if len(draw_scientific_id_sha256) != 64:
        raise ValueError("draw scientific ID must be a full 64-character SHA256")
    if not 0 < bandq <= 1:
        raise ValueError("bandq must be in (0, 1]")
    parameters = {
        "draw_scientific_id_sha256": draw_scientific_id_sha256,
        "bandq": float(bandq),
        "simplification_method": "direct_band",
    }
    digest, _ = compute_scientific_hash(
        "tree",
        parameters,
        parent_scientific_ids=(draw_scientific_id_sha256,),
    )
    return parameters, digest


def evidence_availability(
    *, checkpoint: Path, distances: Iterable[Path], chunks: Iterable[Path]
) -> dict[str, bool]:
    """Record checkpoint, saved-distance, and chunk-payload availability separately."""
    checkpoint_present = checkpoint.is_file() or (
        checkpoint.is_dir() and any(checkpoint.iterdir())
    )
    return {
        "checkpoint_present": checkpoint_present,
        "distances_present": any(path.is_file() for path in distances),
        "chunks_present": any(path.is_file() for path in chunks),
    }
