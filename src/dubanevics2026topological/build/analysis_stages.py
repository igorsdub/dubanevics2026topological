"""Stage fingerprints, dependency digests and the analysis-stage commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..analysis import (
    CHECKPOINT_DIR,
    OUTPUT,
    PROVENANCE_PATH,
    ROOT,
    _write_json_atomic,
    dataset_input_path,
    load_cached_analysis,
    load_or_compute_analysis,
    load_settings,
    read_analysis_record,
    require_files,
    split_declared_inputs,
)
from ..analysis.component_persistence import load_or_compute_component_persistence
from ..analysis.produce_or_load import (
    Checkpoint,
    Stage,
    load_checkpoint,
    produce_or_load,
    result_directory,
    stage_fingerprint,
)
from ..landscape.selection import selection_source
from ..provenance import upstream_identity
from .common import (
    ANNOTATION_TABLE_DIRECTORIES,
    _build_sources,
    _context_kwargs,
    _digest_bytes,
    _digest_file,
    _display_path,
    _load_settings,
    _peak_pair_context,
    _provenance_digest,
    _stage,
)


def _analysis_fingerprint() -> str:
    """Digest the identity of the published analysis: its fingerprint and artefacts.

    The record is checked against the published checkpoint first, so a pending
    publication or an interrupted record update raises instead of keying a figure
    on stale science. Volatile fields (``created_at``, the workflow dump) change
    on a no-op re-publish and are left out, as for the table stages.
    """
    record = read_analysis_record(PROVENANCE_PATH, CHECKPOINT_DIR)
    identity = {
        "fingerprint": record["fingerprint"],
        "artefacts": record.get("artefacts", {}),
    }
    return _digest_bytes(json.dumps(identity, sort_keys=True).encode("utf-8"))


def _comparison_provenance_path(context: Any | None = None) -> Path:
    output = OUTPUT if context is None else context.output_root
    path = output / "analysis" / "clamped_comparison" / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Clamped comparison provenance is missing; run `just comparison-analysis`."
        )
    return path


def _comparison_analysis_fingerprint(context: Any | None = None) -> str:
    """Hash prepared-landscape provenance without rereading large input files.

    The digest leaves out a commit stamp and is the one identity of this record:
    figure records and the table stages that read the comparison checkpoint all
    key on it, never on the file's bytes (ADR 0013, D3).
    """
    payload = json.loads(
        _comparison_provenance_path(context).read_text(encoding="utf-8")
    )
    return _provenance_digest(payload)


def _peak_provenance_path(context: Any | None = None) -> Path:
    output = OUTPUT if context is None else context.output_root
    path = output / "analysis" / "peak_uncertainty" / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Peak uncertainty provenance is missing; run `just peak-analysis`."
        )
    return path


def _peak_analysis_fingerprint(context: Any | None = None) -> str:
    """Digest the peak provenance record, leaving out a commit stamp if it has one.

    Like the comparison digest, and unlike the file's bytes, it is the same
    whether or not the record is stamped, so the peak figures and the table stages
    that read it do not rebuild when the peak writer starts to stamp its record.
    """
    payload = json.loads(_peak_provenance_path(context).read_text(encoding="utf-8"))
    return _provenance_digest(payload)


def _codon_entropy_stage(
    settings: dict[str, Any] | None = None, context: Any | None = None
) -> Stage:
    """Declare what the identity codon-entropy tables depend on.

    The comparison and peak checkpoints enter as the digests of their records
    without a commit stamp, never as the raw bytes of their files (ADR 0013, D3).
    """
    scientific = (settings or _load_settings(context))["analysis"]
    return _stage(
        "codon-entropy",
        settings={
            "codon_entropy": scientific["codon_entropy"],
            "candidate_peak_biology": scientific["candidate_peak_biology"],
            "comparison_checkpoint": _comparison_analysis_fingerprint(
                **_context_kwargs(context)
            ),
            "peak_checkpoint": _peak_analysis_fingerprint(**_context_kwargs(context)),
        },
        code=(
            ROOT
            / "src/dubanevics2026topological/analysis/codon_entropy_populations.py",
            ROOT / "src/dubanevics2026topological/analysis/codon_entropy.py",
            ROOT
            / "src/dubanevics2026topological/simplification/selected_leaf_simplification.py",
            *_build_sources(),
        ),
    )


def _publish_stage_tables(
    stage: Stage,
    produce: Callable[[Path], None],
    *,
    checkpoint_name: str,
    record_path: Path,
    force: bool,
    current_message: str | None = None,
    subdirectory: str | None = None,
    before_export: Callable[[Path], Any] | None = None,
    after_export: Callable[[Checkpoint, Path, list[str], Any], None] | None = None,
    names: Mapping[str, str] | None = None,
    context: Any | None = None,
) -> list[Path]:
    """Produce or load a stage checkpoint and export it to ``output/tables``.

    ``names`` maps an artefact to a different exported name.

    The checkpoint holds the authoritative bytes; ``output/tables`` receives
    byte-identical copies (missing or altered files are restored). The sidecar
    record is removed before copying and rewritten last, so an interrupted
    export leaves no record claiming the tables are current.

    ``subdirectory`` exports into ``output/tables/<subdirectory>`` instead.
    ``before_export(destination)`` runs first, while the record is still present
    (so it can tell an export that follows an interruption, when the record is
    missing, from a reuse), and its result is passed on; ``after_export(ready, destination, written, context)`` runs after it,
    still before the record is rewritten, for files that must not be part of the
    checkpoint (``written`` names the artefacts the copy changed).
    """
    output_root = context.output_root if context is not None else OUTPUT
    table_dir = output_root / "tables"
    destination = table_dir / subdirectory if subdirectory else table_dir
    destination.mkdir(parents=True, exist_ok=True)
    target_record_path = (
        (output_root / "tables" / record_path.name)
        if (context is not None and not str(record_path).startswith(str(output_root)))
        else record_path
    )

    def export(ready: Checkpoint) -> None:
        # Runs under the checkpoint lock, so no other process can replace the
        # checkpoint between verification and the copy.
        if ready.reused and current_message:
            print(current_message)
        cb_context = before_export(destination) if before_export else None
        target_record_path.unlink(missing_ok=True)
        written = ready.export_to(destination, names)
        if after_export:
            after_export(ready, destination, written, cb_context)
        _write_json_atomic(target_record_path, ready.provenance)

    step = output_root / "checkpoints" / checkpoint_name
    checkpoint = produce_or_load(
        stage,
        produce,
        checkpoint_dir=result_directory(step, stage),
        force=force,
        on_ready=export,
    )
    return [
        destination / (names or {}).get(name, name)
        for name in checkpoint.provenance["artefacts"]
    ]


def _bootstrap_digests(
    cached: Any, settings: dict[str, Any], context: Any | None = None
) -> dict[str, Any]:
    """Digest the bootstrap checkpoint, its result settings and the stage source.

    ``n_jobs`` is left out: it changes how the draws are scheduled, not the result.
    """
    from ..simplification.bootstrap_smoothed_tree import (
        restored_source,
        result_settings,
    )

    digests = {
        "bootstrap_checkpoint_provenance": _provenance_digest(cached.provenance),
        "bootstrap_smoothed_tree": result_settings(
            settings["analysis"]["bootstrap_smoothed_tree"]
        ),
    }
    simplification_dir = ROOT / "src/dubanevics2026topological/simplification"
    for name in (
        "__init__.py",
        "bootstrap_smoothed_tree.py",
        "bootstrap_chunks.py",
        "checkpoint_identity.py",
        "identities.py",
        "selected_leaf_simplification.py",
    ):
        key = (
            "bootstrap_smoothed_source"
            if name == "bootstrap_smoothed_tree.py"
            else f"{name}_source"
        )
        digests[key] = _digest_file(simplification_dir / name)
    source = restored_source(context=context)
    if source is not None:
        # The processed bootstrap state the slot was restored from; absent for a
        # slot computed here.
        digests["bootstrap_source"] = source
    return digests


def _band_scan_digests(
    settings: dict[str, Any], cached: Any, context: Any | None = None
) -> dict[str, Any]:
    """Digest the band-quantile scan tables a figure reads, restoring them when missing.

    The tables are restored from ``data/processed/band_quantile_scan/<state>`` when its
    configuration matches the loaded bootstrap; otherwise the build fails with a message
    that names the recipes that produce them.
    """
    from ..analysis.bootstrap_band_quantile_scan import (
        ensure_scan_tables,
        missing_scan_message,
        scan_paths,
    )

    scan_path, survival_path = scan_paths(context)
    ensure_scan_tables(
        settings, cached, path=scan_path, survival_path=survival_path, context=context
    )
    for table in (scan_path, survival_path):
        if not table.is_file():
            raise FileNotFoundError(missing_scan_message(settings, cached, table))
    digests: dict[str, Any] = {
        "band_quantile_scan_config": settings["analysis"][
            "bootstrap_band_quantile_scan"
        ],
        "band_quantile_scan_table": _digest_file(scan_path),
        "band_quantile_survival_table": _digest_file(survival_path),
        "band_quantile_scan_source": _digest_file(
            ROOT
            / "src/dubanevics2026topological/analysis/bootstrap_band_quantile_scan.py"
        ),
    }
    return digests


def _pooled_scatter_dependency(
    inputs: Mapping[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """What the peak figure's pooled scatter depends on besides the peak checkpoint.

    The scatter is composed at render time from the growth-rate table (ADR 2026-10-22-pooled-standard-error-for-bootstrap),
    so the figure keys on that checkpoint's identity (fingerprint and artefact
    hashes, never its timestamps), on the two tolerances that validate the join
    and on the module that composes it.
    """
    analysis = settings["analysis"]
    return {
        "growth_rate_checkpoint": upstream_identity(inputs["growth_rates"].provenance),
        "identity_tolerance": analysis["sequence_growth_rates"]["identity_tolerance"],
        **analysis["peak_uncertainty_scatter"],
        "source": _digest_file(
            ROOT / "src/dubanevics2026topological/analysis/pooled_scatter.py"
        ),
    }


def ensure_analysis_checkpoint(*, context: Any | None = None):
    """Verify the published checkpoint cheaply and rebuild stale or broken state."""
    settings = load_settings() if context is None else load_settings(context=context)
    scientific = settings["analysis"]
    dataset = scientific["dataset"]
    if dataset.get("landscape") != "full":
        raise ValueError(
            "The initial publication build supports only the full landscape."
        )
    if scientific.get("tree", {}).get("kind") != "split":
        raise ValueError("The initial publication build supports only a split tree.")
    if scientific.get("persistence", {}).get("kind") != "extended_h0":
        raise ValueError(
            "The initial publication build supports only persistence.kind = "
            "'extended_h0' (ordinary H0 with the global pair closed at the global "
            "minimum)."
        )
    # Content hashes also detect dependency switches to an older editable checkout.
    return (
        load_or_compute_analysis()
        if context is None
        else load_or_compute_analysis(context=context)
    )


MANUSCRIPT_QUANTITIES_RECORD = ".manuscript-quantities-provenance.json"


# The analysis modules whose functions compute a registry value (the loaders only
# read upstream outputs, which the upstream identities already key on). A test
# checks this list against the transitive package-internal imports of
# manuscript_quantities and band_view_ridges (storage and upstream-stage modules excluded).
REGISTRY_CODE_MODULES = (
    "manuscript_quantities.py",
    "band_view_ridges.py",
    "barseq_inputs.py",
    "biological_annotations.py",
    "describe.py",
    "candidate_tree_contrast.py",
    "codon_entropy_populations.py",
    "component_persistence.py",
    "graph_summary.py",
    "leaves.py",
    "merge_flow.py",
    "ridges.py",
)


REGISTRY_SIMPLIFICATION_MODULES = (
    "__init__.py",
    "bootstrap_aggregation.py",
    "bootstrap_smoothed_tree.py",
    "bootstrap_chunks.py",
    "checkpoint_identity.py",
    "cooccurrence_core.py",
    "identities.py",
    "selected_leaf_simplification.py",
)


REGISTRY_LANDSCAPE_MODULES = (
    "selection.py",
    "full_population.py",
    "selected_landscape.py",
)


REGISTRY_THRESHOLD_MODULES = (
    "functional_threshold.py",
    "classification_probability_scans.py",
    "fdr_selection.py",
    "functional_classification.py",
    "functional_classification_config.py",
)


def require_current_registry(
    settings: dict[str, Any], output: Path | None = None
) -> None:
    """Raise unless the exported registry's record matches the current stage fingerprint.

    The current fingerprint is computed as ``build_manuscript_quantities`` declares
    the stage (same settings, upstream identities and code), without producing.
    """
    from ..analysis.manuscript_quantities import (
        load_sources,
        read_registry_fingerprint,
    )
    from ..analysis.manuscript_quantities import (
        require_current_registry as require_current_analysis_registry,
    )

    output = OUTPUT if output is None else output
    # The record is read first so a missing one is named before the sources load.
    read_registry_fingerprint(output)
    stage, _ = _manuscript_quantities_stage(settings, load_sources(output))
    require_current_analysis_registry(output, stage_fingerprint(stage))


def _manuscript_quantities_stage(
    settings: dict[str, Any], sources: Mapping[str, Any]
) -> tuple[Stage, dict[str, Any]]:
    """Declare the registry stage: its config, the upstream identities and its code."""
    from ..analysis.manuscript_quantities import STAGE_NAME, source_identities
    from ..threshold.functional_threshold import DECLARATION_KEY, declaration_identity

    scientific = settings["analysis"]
    clamped_settings = scientific.get("clamped", {})
    clamped_config: dict[str, Any] = {
        "published_functional_threshold": clamped_settings[
            "published_functional_threshold"
        ]
    }
    if DECLARATION_KEY in clamped_settings:
        clamped_config[DECLARATION_KEY] = declaration_identity(clamped_settings[DECLARATION_KEY])
    config = {
        "clamped": clamped_config,
        "codon_entropy": scientific["codon_entropy"],
    }
    analysis_dir = ROOT / "src/dubanevics2026topological/analysis"
    simplification_dir = ROOT / "src/dubanevics2026topological/simplification"
    threshold_dir = ROOT / "src/dubanevics2026topological/threshold"
    landscape_dir = ROOT / "src/dubanevics2026topological/landscape"
    stage = _stage(
        STAGE_NAME,
        settings={"config": config, "upstream": source_identities(sources)},
        code=(
            *(analysis_dir / name for name in REGISTRY_CODE_MODULES),
            *(simplification_dir / name for name in REGISTRY_SIMPLIFICATION_MODULES),
            *(landscape_dir / name for name in REGISTRY_LANDSCAPE_MODULES),
            *(threshold_dir / name for name in REGISTRY_THRESHOLD_MODULES),
            *_build_sources(),
        ),
    )
    return stage, config


def _full_fitness_dependency(settings: dict[str, Any]) -> dict[str, str]:
    """The Full-graph fitness population's identity: its input bytes and loader code."""
    return {
        "fitness_wt": _digest_file(
            ROOT / settings["analysis"]["dataset"]["fitness_wt"]
        ),
        "source": _digest_file(
            ROOT / "src/dubanevics2026topological/landscape/full_population.py"
        ),
    }


LEADING_SADDLE_BAND_CHECKPOINT = "leading_saddle_band"


EXPORTED_SUMMARY = "leading_saddle_band_summary.json"


def _leading_saddle_band_stage(
    settings: dict[str, Any], *, context: Any | None = None
) -> Stage:
    """Declare what the leading-saddle-band tables depend on.

    The component checkpoint enters as its provenance record's fingerprint and
    artefact hashes (which cover the raw inputs), never as the raw pair bytes.
    """
    from ..analysis import component_persistence
    from ..analysis.leading_saddle_band import STAGE_NAME
    from ..threshold.functional_threshold import DECLARATION_KEY, declaration_identity

    record = json.loads(
        component_persistence.component_paths(context).provenance.read_text(
            encoding="utf-8"
        )
    )
    clamped = settings["analysis"]["clamped"]
    dataset = settings["analysis"]["dataset"]
    fitness = dataset_input_path(dataset["fitness_wt"], context, root=ROOT)
    fitness_diff = dataset_input_path(dataset["fitness_diff"], context, root=ROOT)
    require_files([fitness, fitness_diff], "leading saddle band inputs")
    inputs, outside_inputs = split_declared_inputs(
        {"fitness_wt": fitness, "fitness_diff": fitness_diff}, ROOT
    )
    analysis_dir = ROOT / "src/dubanevics2026topological/analysis"
    threshold_dir = ROOT / "src/dubanevics2026topological/threshold"
    landscape_dir = ROOT / "src/dubanevics2026topological/landscape"
    return _stage(
        STAGE_NAME,
        settings={
            **({"outside_input_sha256": outside_inputs} if outside_inputs else {}),
            "published_functional_threshold": clamped["published_functional_threshold"],
            DECLARATION_KEY: declaration_identity(clamped[DECLARATION_KEY]),
            # The classification's T is the Candidate threshold landmark.
            "candidate_selection_source": selection_source(settings, context=context),
            "component_checkpoint": {
                key: record[key]
                for key in ("fingerprint", "pairs_sha256", "diagnostics_sha256")
            },
        },
        # The Full-graph population (its input bytes and loader) enters the
        # landmarks, intervals, trunks and entropy.
        inputs=inputs,
        code=(
            analysis_dir / "leading_saddle_band.py",
            landscape_dir / "full_population.py",
            landscape_dir / "selection.py",
            threshold_dir / "functional_threshold.py",
            analysis_dir / "codon_entropy.py",
            analysis_dir / "codon_entropy_populations.py",
            analysis_dir / "component_persistence.py",
            *_build_sources(),
        ),
    )


def _leading_saddle_band_identity(context: Any | None = None) -> dict[str, Any]:
    """The checkpoint's fingerprint and artefact hashes (no timestamp or commit stamp)."""
    output_root = context.output_root if context is not None else OUTPUT
    stage = _leading_saddle_band_stage(_load_settings(context), context=context)
    leaf = result_directory(
        output_root / "checkpoints" / LEADING_SADDLE_BAND_CHECKPOINT, stage
    )
    record = json.loads((leaf / "provenance.json").read_text(encoding="utf-8"))
    return {key: record[key] for key in ("fingerprint", "artefacts")}


def leading_saddle_band_directory(
    settings: dict[str, Any] | None = None, *, context: Any | None = None
) -> Path:
    """The keyed folder of the current leading-saddle-band checkpoint (it may not exist)."""
    output_root = context.output_root if context is not None else OUTPUT
    stage = _leading_saddle_band_stage(
        settings or _load_settings(context), context=context
    )
    return result_directory(
        output_root / "checkpoints" / LEADING_SADDLE_BAND_CHECKPOINT, stage
    )


def load_cached_leading_saddle_band(
    settings: dict[str, Any] | None = None, *, context: Any | None = None
) -> Any:
    """Read the stage checkpoint for the configured settings; never compute."""
    from ..analysis.leading_saddle_band import RETRY_HINT, read_leading_saddle_band

    output_root = context.output_root if context is not None else OUTPUT
    stage = _leading_saddle_band_stage(
        settings or _load_settings(context), context=context
    )
    step = output_root / "checkpoints" / LEADING_SADDLE_BAND_CHECKPOINT
    return load_checkpoint(
        stage,
        lambda ready: read_leading_saddle_band(ready.path),
        checkpoint_dir=result_directory(step, stage),
        retry_hint=RETRY_HINT,
    )


def _biological_annotation_stage(context: Any | None = None) -> Stage:
    """Declare the annotation-table stage; its fingerprint needs no run."""
    return _stage(
        "biological-annotation-tables",
        settings={
            "landscapes": list(ANNOTATION_TABLE_DIRECTORIES),
            "comparison_checkpoint": _comparison_analysis_fingerprint(context),
        },
        code=(
            ROOT / "src/dubanevics2026topological/analysis/biological_annotations.py",
            ROOT / "src/dubanevics2026topological/analysis/describe.py",
            *_build_sources(),
        ),
    )


def current_biological_annotation_fingerprint() -> str:
    """The fingerprint the annotation tables must carry to be current.

    Computed from the stage declaration (the comparison provenance digest and the
    code), so the manuscript registry can compare it with the export record
    without rendering the study-biology figure. Raises ``FileNotFoundError`` when
    the comparison provenance is missing.
    """
    return stage_fingerprint(_biological_annotation_stage())


def cmd_status(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.produce_or_load import saved_results

    root = context.output_root / "checkpoints"
    rows = saved_results(root)
    if not rows:
        print(f"No saved results under {root}.")
        return 0
    for row in rows:
        warning = f"  {row['warning']}" if row["warning"] else ""
        print(f"{row['step']}\t{row['folder']}\t{row['commit']}{warning}")
    return 0


def cmd_analysis(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    print(
        "Checking full Papkou2023 inputs and scientific fingerprint...", flush=True
    )
    ensure_analysis_checkpoint(
        context=context
    ) if not args.force else load_or_compute_analysis(force=True, context=context)
    analysis = load_cached_analysis(context=context)
    print(f"Analysis ready: {analysis.input_summary.height} summary rows.")
    return 0


def cmd_verify_analysis(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    analysis = ensure_analysis_checkpoint(context=context)
    print(f"Verified checkpoint for {analysis.input_summary.height} summary rows.")
    return 0


def cmd_component_analysis(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    pairs, diagnostics = load_or_compute_component_persistence(
        force=args.force, context=context
    )
    print(
        "Component persistence ready: "
        f"{pairs.height} pairs; join-tree trailing saddle "
        f"{diagnostics.get('join_tree_trailing_saddle')}."
    )
    return 0


def cmd_branch_decomposition(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.branch_decomposition import (
        ARTEFACTS,
        CHECKPOINT_DIR,
        compute_branch_decomposition,
        describe,
    )

    decomposition = compute_branch_decomposition(load_settings(), force=args.force)
    for name in ARTEFACTS:
        print(f"Table: {(CHECKPOINT_DIR / name).relative_to(ROOT)}")
    print(describe(decomposition))
    return 0


def cmd_peak_pair_distances(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.peak_pair_distances import (
        ARTEFACTS,
        compute_peak_pair_distances,
    )

    context = _peak_pair_context(context)
    result = compute_peak_pair_distances(context, force=bool(args.force))
    for name in ARTEFACTS:
        print(f"Table: {_display_path(result.path / name)}")
    print(
        f"Peak-pair distances {'reused' if result.reused else 'produced'}: "
        f"{int(result.mantel['peak_count'][0])} peaks, {result.pairs.height} pairs."
    )
    return 0
