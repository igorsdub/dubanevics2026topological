"""Publish the pooled-scatter tables as an identified product from the processed growth-rate table and the peak checkpoint (ADR 2026-10-22)."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import polars as pl

from ..analysis import load_settings, peak_uncertainty
from ..analysis.checkpoint import checkpoint_lock
from ..analysis.peak_uncertainty import _load_peak_checkpoint
from ..analysis.pooled_scatter import pooled_scatter
from ..landscape.products import _publish
from ..results import RunContext, processed

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_inputs(context: RunContext) -> dict[str, str]:
    """Return the sha256 of the 9 peak tables plus diagnostics.json, read
    under the peak checkpoint_lock (caller ensures), failing if any is missing.
    """
    peak_dir, _ = peak_uncertainty.cache_paths(context)
    input_data_sha256s: dict[str, str] = {}
    table_names: list[str] = []
    for key in ("study_clamped", "candidate_clamped"):
        for suffix in ("scatter", "contrasts", "peaks", "decay"):
            table_names.append(f"{key}_{suffix}")
    table_names.append("pairwise_sign_disagreements")
    for n in table_names:
        rel = f"tables/{n}.parquet"
        p = peak_dir / rel
        if not p.is_file():
            raise FileNotFoundError(f"Peak checkpoint file missing for hashing: {rel}")
        input_data_sha256s[rel] = _file_sha256(p)
    rel = "diagnostics.json"
    p = peak_dir / rel
    if not p.is_file():
        raise FileNotFoundError(f"Peak checkpoint file missing for hashing: {rel}")
    input_data_sha256s[rel] = _file_sha256(p)
    return input_data_sha256s


def build_pooled_scatter_product(context: RunContext | None = None) -> Path:
    """Build the pooled_scatter product.

    Reads the growth-rate table of ``processed/growth_rates``, loads the peak checkpoint,
    computes fresh SHA-256 for every file read from the checkpoint plus the
    checkpoint's recorded fingerprint (legacy checkpoints have no artifact hashes),
    and publishes the tables produced by the unchanged pooled_scatter computation.
    The input hashes ensure that changed peak bytes yield a new product identity.
    Writes are atomic via the shared _publish helper.
    """
    context = context or RunContext()
    settings = load_settings(context=context)
    analysis = settings["analysis"]
    identity_tolerance = analysis["sequence_growth_rates"]["identity_tolerance"]
    standard_error_tolerance = analysis["peak_uncertainty_scatter"][
        "standard_error_tolerance"
    ]

    growth_path, growth_sha256 = processed.growth_rates(context)
    growth_df = pl.read_parquet(growth_path)
    # The point-estimate Published threshold is the label of this product.
    label_fields = {"thr": "published", "fdr": "none"}

    peak_dir, _ = peak_uncertainty.cache_paths(context)
    with checkpoint_lock(peak_dir, "peak uncertainty"):
        peak = _load_peak_checkpoint(peak_dir, context=context)
        input_data_sha256s = _peak_inputs(context)

    # Parent reference for the manifest: the processed growth-rate table, by its sha256.
    parent_ref = {
        "product_type": "growth_rates",
        "label": "growth_rates",
        "scientific_id_sha256": growth_sha256,
    }

    code_paths = (
        Path(inspect.getfile(pooled_scatter)).resolve(),
        Path(inspect.getfile(_load_peak_checkpoint)).resolve().parent
        / "peak_uncertainty.py",
        Path(__file__).resolve(),
    )

    identity = {
        **label_fields,
        "growth_rates_scientific_id_sha256": growth_sha256,
        "identity_tolerance": float(identity_tolerance),
        "standard_error_tolerance": float(standard_error_tolerance),
    }

    scatter_payload = pooled_scatter(
        peak,
        growth_df,
        identity_tolerance=identity_tolerance,
        standard_error_tolerance=standard_error_tolerance,
    )

    def write_artifacts(directory: Path) -> None:
        for name, frame in sorted(scatter_payload["tables"].items()):
            if not isinstance(frame, pl.DataFrame):
                frame = pl.DataFrame(frame)
            frame.write_parquet(directory / f"{name}.parquet")
        (directory / "diagnostics.json").write_text(
            json.dumps(scatter_payload["diagnostics"], sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "pooled_scatter",
        label_fields,
        identity,
        context=context,
        input_data_sha256s=input_data_sha256s,
        parents=(parent_ref,),
        code_paths=code_paths,
        write_artifacts=write_artifacts,
    )
    return path
