"""Publish the bootstrap codon entropy populations, entropy sweeps, and render tables.

This product moves the remaining render-time science from
``figures/registry.py::_compose_bootstrap_codon`` into a saved, identified build
product. At build time (under canonical checkpoint locks) it:

* Loads the clamped comparison and the bootstrap smoothed tree hills.
* Hashes every consumed file's raw bytes inside the lock windows (comparison
  lock then bootstrap lock, in that order).
* Builds the band-simplified view of the candidate graph using the deterministic
  legacy path (build_band_simplified_view).
* Extracts core/KEM/TCL ridge populations and computes positional entropy sweeps.
* Prepares the plot tables, ridge edge pairs, stop points, and wild-type marker
  that the figure renderer needs.
* Publishes everything so that figure composition becomes pure presentation.

The product follows the PR 3a pattern established by pooled_scatter_product.py.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import load_settings
from ..analysis.candidate_tree_contrast import build_candidate_tree
from ..analysis.checkpoint import checkpoint_lock
from ..analysis.codon_entropy import compute_entropy_sweeps
from ..analysis.codon_entropy_populations import extract_codon_entropy_populations
from ..analysis.codon_entropy_render_tables import build_render_tables
from ..landscape.products import _publish
from ..landscape.selection import resolve_candidate_selection, selection_identity
from ..results import RunContext
from ..simplification import build_band_simplified_view
from ..simplification.bootstrap_smoothed_tree import (
    checkpoint_dir as bootstrap_checkpoint_dir,
)
from ..simplification.bootstrap_smoothed_tree import (
    load_cached_bootstrap_smoothed_tree,
)
from ..simplification import checkpoint_identity
from ..simplification.bootstrap_smoothed_tree import cache_dir as bootstrap_slot_dir

_LABEL_KEYS = frozenset(
    {
        "graph",
        "thr",
        "fdr",
        "q",
        "p",
        "draws",
        "seed",
        "tree",
        "bandq",
        "host",
        "run",
        "hash",
        "style",
    }
) - {"host", "run", "hash", "style"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_product_identity(selection: Any) -> dict[str, Any]:
    """Identify the resolved selection and its source without rereading it."""
    return {
        **selection_identity(selection),
        "rounded_threshold": float(selection.threshold.threshold),
        "source": selection.source,
    }


def _product_identity(settings: dict[str, Any], selection: Any) -> dict[str, Any]:
    return {
        "codon_entropy_sweep_points": int(settings["analysis"]["codon_entropy"]["sweep_points"]),
        "functional_selection": _selection_product_identity(selection),
    }


def _require_bootstrap_selection(scientific: dict[str, Any], selection: Any) -> None:
    if scientific.get("candidate_selection") != selection_identity(selection):
        raise ValueError("Bootstrap checkpoint selection differs from the resolved selection.")


def _comparison_inputs(context: RunContext) -> dict[str, str]:
    """Return raw-byte sha256 for every file in the clamped comparison checkpoint.

    The caller must hold the comparison checkpoint_lock.
    """
    from ..analysis.clamped_cache import cache_dirs

    COMPARISON_CACHE_DIR = cache_dirs(context)[0]
    hashes: dict[str, str] = {}
    # The comparison writes several parquet tables and a diagnostics JSON.
    # We hash every .parquet and the key JSON sidecars under the cache dir.
    for rel in sorted(COMPARISON_CACHE_DIR.rglob("*")):
        if not rel.is_file():
            continue
        if rel.name.startswith("."):
            continue
        if rel.suffix not in {".parquet", ".json"}:
            continue
        key = str(rel.relative_to(COMPARISON_CACHE_DIR))
        hashes[key] = _file_sha256(rel)
    if not hashes:
        raise FileNotFoundError(
            "Clamped comparison checkpoint appears empty; run `just comparison-analysis`."
        )
    return hashes


def _bootstrap_inputs(context: RunContext) -> dict[str, str]:
    """Return raw-byte sha256 for the bootstrap checkpoint files we consume.

    The caller must hold the bootstrap checkpoint_lock. We hash the hills and
    summary artifacts (under the analysis working dir or checkpoint) that feed
    the band view construction.
    """
    hashes: dict[str, str] = {}
    # Hash actual files that the build reads: prefer the analysis working slot,
    # fall back to the published checkpoint layout.
    analysis_dir = context.output_root / "analysis" / "bootstrap_smoothed_tree"
    candidates = [
        "hills.parquet",
        "summary.json",
        "draw_stats.parquet",
        "provenance.json",
        "checkpoint/hills.parquet",
        "checkpoint/summary.json",
        "checkpoint/draw_stats.parquet",
    ]
    base_dirs = [analysis_dir, bootstrap_checkpoint_dir(context)]
    for base in base_dirs:
        for rel in candidates:
            p = base / rel
            if p.is_file():
                key = f"{base.name}/{rel}" if base == analysis_dir else rel
                hashes[key] = _file_sha256(p)
    if not hashes:
        raise FileNotFoundError(
            "Bootstrap checkpoint artifacts missing; run `just bootstrap-smoothed-tree`."
        )
    return hashes


def build_bootstrap_codon_entropy_product(
    context: RunContext | None = None,
) -> Path:
    """Build the bootstrap_codon_entropy product.

    Acquires the comparison lock then the bootstrap lock (fixed order), loads
    via the public non-locking readers, hashes the raw bytes of every consumed
    file inside those locks, builds the band view, extracts populations and
    entropy, prepares render tables, and publishes the product.
    """
    context = context or RunContext()
    settings = load_settings(context=context)
    analysis_cfg = settings["analysis"]
    codon_cfg = analysis_cfg["codon_entropy"]

    # Resolve the functional selection for provenance and for the extractor.
    selection = resolve_candidate_selection(settings, context=context)

    from ..analysis.clamped_cache import cache_dirs

    COMPARISON_CACHE_DIR = cache_dirs(context)[0]

    # Hold both locks while reading and validating the two checkpoint generations.
    with checkpoint_lock(COMPARISON_CACHE_DIR, "comparison"):
        from ..analysis.clamped_cache import _fingerprint, _load_clamped_checkpoint

        with checkpoint_lock(bootstrap_checkpoint_dir(context), "bootstrap smoothed tree"):
            comparison = _load_clamped_checkpoint(COMPARISON_CACHE_DIR, include_full=True)
            if comparison["provenance"].get("fingerprint") != _fingerprint(settings, context=context):
                raise ValueError("Comparison checkpoint is stale; run `just comparison-analysis`.")
            bootstrap = load_cached_bootstrap_smoothed_tree(
                settings=settings, context=context, lock=False
            )
            bootstrap_scientific = checkpoint_identity.scientific_of(bootstrap_slot_dir(context))
            _require_bootstrap_selection(bootstrap_scientific, selection)
            current_selection = resolve_candidate_selection(settings, context=context)
            if _selection_product_identity(current_selection) != _selection_product_identity(selection):
                raise ValueError("Candidate selection changed while reading checkpoints; retry.")
            comparison_hashes = _comparison_inputs(context)
            bootstrap_hashes = _bootstrap_inputs(context)

    # Build (or obtain) the candidate contrast tree. build_candidate_tree is
    # deterministic given the comparison inputs we have already hashed.
    candidate_tree, _wt = build_candidate_tree(
        settings, selection=selection, context=context
    )
    hills = bootstrap.hills
    band_view = build_band_simplified_view(candidate_tree, hills)

    # Extract populations using the same deterministic routine the registry used.
    populations = extract_codon_entropy_populations(
        comparison=comparison,
        candidate_view=band_view,
        threshold=selection.threshold.threshold,
        config=codon_cfg,
        save_files=False,
        functional_sequences=selection.functional_sequences,
    )

    # Compute entropy sweeps over the three populations.
    entropy = compute_entropy_sweeps(
        {
            "core": populations["df_core"],
            "kem": populations["df_kem"],
            "tcl": populations["df_tcl"],
        },
        num_thresholds=int(codon_cfg["sweep_points"]),
    )

    # Prepare the render tables (node/edge tables, ridge edges, stops, WT marker).
    # The band view carries the tree and assignments the render tables need.
    metadata = populations.get("metadata", {})
    render_tables = build_render_tables(band_view, metadata)

    # Assemble label fields from the comparison (graph/thr/fdr/q etc.).
    label_fields: dict[str, Any] = {}
    # The comparison carries a scientific identity; copy the stable keys.
    for key in _LABEL_KEYS:
        if key in comparison:
            label_fields[key] = comparison[key]
        elif key in getattr(comparison, "scientific_identity", {}):
            label_fields[key] = comparison["scientific_identity"][key]

    # Scientific identity for the product itself.
    identity: dict[str, Any] = {
        **{k: v for k, v in label_fields.items() if k in _LABEL_KEYS},
        **_product_identity(settings, selection),
    }

    # Code paths for provenance (the extractor, the sweeps, the render tables,
    # and this product file).
    code_paths = (
        Path(inspect.getfile(extract_codon_entropy_populations)).resolve(),
        Path(inspect.getfile(compute_entropy_sweeps)).resolve(),
        Path(inspect.getfile(build_render_tables)).resolve(),
        Path(__file__).resolve(),
    )

    # Merge input hashes (comparison first, then bootstrap artifacts).
    # Namespace to prevent collisions between the two checkpoints.
    comparison_hashes = {f"comparison/{k}": v for k, v in comparison_hashes.items()}
    bootstrap_hashes = {f"bootstrap/{k}": v for k, v in bootstrap_hashes.items()}
    input_data_sha256s = {**comparison_hashes, **bootstrap_hashes}

    # No tree-layout parent for mixture_quantile; the raw bootstrap + comparison
    # hashes are the provenance (as decided for PR 3b).
    parents: tuple[dict[str, Any], ...] = ()

    def write_artifacts(directory: Path) -> None:
        # Populations
        populations["df_core"].write_parquet(directory / "populations_core.parquet")
        populations["df_kem"].write_parquet(directory / "populations_kem.parquet")
        populations["df_tcl"].write_parquet(directory / "populations_tcl.parquet")
        # Entropy sweeps (one frame per population under a single file or split;
        # we emit a single table with a population column for simplicity).
        entropy_frames = []
        for pop_name, frame in entropy.items():
            entropy_frames.append(
                frame.with_columns(pl.lit(pop_name).alias("population"))
            )
        if entropy_frames:
            pl.concat(entropy_frames).write_parquet(
                directory / "entropy_sweeps.parquet"
            )
        # Render tables
        render_tables["node_table"].write_parquet(directory / "node_table.parquet")
        render_tables["edge_table"].write_parquet(directory / "edge_table.parquet")
        render_tables["edge_segment_table"].write_parquet(
            directory / "edge_segment_table.parquet"
        )
        render_tables["ridge_edges"].write_parquet(directory / "ridge_edges.parquet")
        render_tables["stop_points"].write_parquet(directory / "stop_points.parquet")
        wt = render_tables.get("wild_type_marker")
        if wt is None:
            wt = pl.DataFrame()
        wt.write_parquet(directory / "wild_type_marker.parquet")
        # Metadata for the figure (common_saddle, summits, count_domain, etc.)
        meta_out = dict(metadata)
        meta_out["count_domain"] = list(render_tables["count_domain"])
        meta_out["plot_data_attrs"] = render_tables["plot_data_attrs"]
        (directory / "metadata.json").write_text(
            json.dumps(meta_out, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Light provenance helper for fast loader verification
        (directory / "input_data_sha256s.json").write_text(
            json.dumps(input_data_sha256s, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "bootstrap_codon_entropy",
        label_fields,
        identity,
        context=context,
        input_data_sha256s=input_data_sha256s,
        parents=parents,
        code_paths=code_paths,
        write_artifacts=write_artifacts,
    )
    return path


# Convenience for tests and wiring
def _codon_entropy_input_hashes(context: RunContext) -> dict[str, str]:
    """Lightweight snapshot used by the saved-figure-inputs loader verification.

    Returns namespaced keys (comparison/..., bootstrap/...) to match the
    product's recorded data_provenance.
    """
    from ..analysis.clamped_cache import cache_dirs

    COMPARISON_CACHE_DIR = cache_dirs(context)[0]

    with checkpoint_lock(COMPARISON_CACHE_DIR, "comparison"):
        with checkpoint_lock(bootstrap_checkpoint_dir(context), "bootstrap smoothed tree"):
            comp = _comparison_inputs(context)
            boot = _bootstrap_inputs(context)
    comp = {f"comparison/{k}": v for k, v in comp.items()}
    boot = {f"bootstrap/{k}": v for k, v in boot.items()}
    return {**comp, **boot}
