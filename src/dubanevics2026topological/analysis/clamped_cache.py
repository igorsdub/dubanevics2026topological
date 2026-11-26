"""Cached build orchestration for the clamped Papkou tree comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from dubanevics2026topological.network_analysis import load_network_analysis, save_network_analysis
from dubanevics2026topological.papkou2023 import (
    Papkou2023Workflow,
    load_papkou2023_article_network,
)

from ..landscape.selected_landscape import (
    analyze_prepared_landscape,
    prepare_clamped_landscape,
    validate_clamped_reference,
)
from ..landscape.selection import (
    candidate_selection_fingerprint,
    prepare_selected_landscape,
)
from ..rawdata.value_digest import value_digest
from ..tables import _public_table
from . import (
    ROOT,
    _write_json_atomic,
    dataset_input_path,
    dedupe_by_resolved,
    fingerprint_key,
    get_run_context,
    hash_sources,
    load_cached_analysis,
    load_or_compute_analysis,
    load_settings,
    require_files,
    scientific_code_paths,
    scientific_inputs,
    split_declared_inputs,
)
from .checkpoint import checkpoint_lock, publish_checkpoint, recover_checkpoint
from .describe import SUMMARY_SCHEMA, describe
from .produce_or_load import Stage, stage_fingerprint


def _output_root(context: Any | None) -> Path:
    """The output root of ``context``; a context-less call resolves a fresh one now."""
    return Path((context or get_run_context()).output_root)


def analysis_provenance_path(context: Any | None = None) -> Path:
    """The Full analysis provenance record the comparison fingerprint builds on."""
    return _output_root(context) / "analysis" / "provenance.json"


def cache_dirs(context: Any | None = None) -> tuple[Path, Path]:
    """The comparison checkpoint directory and its pending marker for ``context``."""
    analysis = _output_root(context) / "analysis"
    return analysis / "clamped_comparison", analysis / ".clamped-comparison-pending"


# The computed Candidate functional threshold and its inputs, for the registry.
CANDIDATE_THRESHOLD_FILE = "candidate_functional_threshold.json"
CANDIDATE_THRESHOLD_KEYS = frozenset(
    {
        "rule",
        "decimals",
        "median",
        "global_minimum",
        "median_gap",
        "unrounded_threshold",
        "rounded_threshold",
        "functional_vertex_count",
        "uncertain_vertex_count",
        "selected_component_functional_vertex_count",
    }
)
LANDSCAPE_KEYS = ("study_clamped", "candidate_clamped")
CACHE_SCHEMA = 6
# The command the checkpoint helpers name when a rollback or recovery fails.
RETRY_HINT = "just comparison-analysis"
RAW_ANALYSIS_SCALAR = "raw_fitness"
EDGE_SELECTION_RULE = "at_least_one_endpoint_raw_fitness_ge_threshold"


def _tree_support_tables(name: str, analysis: Any) -> dict[str, pl.DataFrame]:
    """Export stable-identity branches and pairs from the cached tree products."""
    tree_analysis = analysis.trees["split"]
    branches = _public_table("branch_table", tree_analysis.branch_table, analysis)
    if "branch_member_count" in branches.columns:
        branches = branches.with_columns(
            (pl.col("branch_member_count") + 1).alias("branch_vertex_count")
        )
    persistence = _public_table(
        "persistence_table", tree_analysis.persistence_table, analysis
    )
    return {
        f"{name}_branch_table": branches,
        f"{name}_persistence_table": persistence,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_stage(settings: dict[str, Any], *, context: Any | None = None) -> Stage:
    """Declare the comparison checkpoint: scientific settings, inputs and code.

    The settings are the parsed config sections the checkpoint reads (``dataset``
    and ``clamped``), the hashes of its code and the full analysis it builds on,
    not the bytes of ``config/analysis.toml``, so ``n_jobs`` and ``chunk_size``
    never reach the fingerprint. The graph and peak reference files are verified
    against the digests declared in the ``clamped`` section, which is part of the
    settings, so they are not hashed a second time as stage inputs.
    """
    # Imported here because ``stages`` itself imports the helper in this package.
    from ..stages import project_stage

    scientific = settings["analysis"]
    clamped = scientific["clamped"]
    all_inputs = scientific_inputs(settings, context, root=ROOT)
    reference = [ROOT / clamped["graphml"], ROOT / clamped["peaks"]]
    code = dedupe_by_resolved(
        [
            *scientific_code_paths(),
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1] / "landscape" / "selected_landscape.py",
            Path(__file__).resolve().parents[1] / "landscape" / "selection.py",
            Path(__file__).resolve().parents[1] / "landscape" / "full_population.py",
            Path(__file__).resolve().parents[1]
            / "threshold"
            / "functional_threshold.py",
            Path(__file__).with_name("graph_summary.py").resolve(),
            Path(__file__).with_name("describe.py").resolve(),
            Path(__file__).with_name("__init__.py").resolve(),
            Path(__file__).resolve().parents[1] / "tables" / "__init__.py",
        ]
    )
    require_files([*all_inputs, *reference, *code], "clamped comparison inputs")
    inputs, outside_inputs = split_declared_inputs(
        dict(zip(("fitness_wt", "fitness_diff"), all_inputs)), ROOT
    )
    local_code = [
        path for path in code if path.resolve().is_relative_to(ROOT.resolve())
    ]
    for path, setting in zip(reference, ("graphml_value_sha256", "peaks_value_sha256")):
        digest = value_digest(path)
        if digest != clamped[setting]:
            raise ValueError(
                f"Reference input {fingerprint_key(path)!r} has value digest {digest}, "
                f"expected {clamped[setting]} from config/analysis.toml "
                "(values are compared, not bytes)."
            )
    try:
        baseline = json.loads(
            analysis_provenance_path(context).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise FileNotFoundError(
            "Full Papkou analysis provenance is missing; run `just analysis`."
        ) from error
    return project_stage(
        "clamped-comparison",
        settings={
            "schema": CACHE_SCHEMA,
            "analysis_scalar": RAW_ANALYSIS_SCALAR,
            "edge_selection_rule": EDGE_SELECTION_RULE,
            "config": {"dataset": scientific["dataset"], "clamped": clamped},
            "candidate_selection": candidate_selection_fingerprint(
                settings, context=context
            ),
            "sources": hash_sources(code),
            "full_analysis": baseline.get("fingerprint"),
            **({"outside_input_sha256": outside_inputs} if outside_inputs else {}),
        },
        inputs=inputs,
        code=local_code,
        root=ROOT,
    )


def _fingerprint(settings: dict[str, Any], *, context: Any | None = None) -> str:
    return stage_fingerprint(comparison_stage(settings, context=context))


def _table_paths(directory: Path, keys: tuple[str, ...]) -> list[Path]:
    return [directory / "tables" / f"{name}.parquet" for name in keys]


def _load_clamped_checkpoint(
    directory: Path,
    *,
    allow_pending: bool = False,
    include_full: bool = True,
    pending_path: Path | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Validate and load a checkpoint at ``directory`` without acquiring a lock."""
    if pending_path is None:
        pending_path = cache_dirs(context)[1]
    if pending_path.exists() and not allow_pending:
        raise RuntimeError(
            "Clamped comparison refresh is pending; rerun `just comparison-analysis`."
        )
    provenance_path = directory / "provenance.json"
    diagnostics_path = directory / "diagnostics.json"
    summary_path = directory / "graph_summary.json"
    expected_tables = tuple(
        f"{landscape}_{table}"
        for landscape in LANDSCAPE_KEYS
        for table in (
            "components",
            "excluded_vertices",
            "peaks",
            "branch_table",
            "persistence_table",
        )
    )
    if not provenance_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError(
            "No clamped comparison checkpoint exists; run `just comparison-analysis`."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != CACHE_SCHEMA
        or provenance.get("analysis_scalar") != RAW_ANALYSIS_SCALAR
        or provenance.get("published_edge_selection_rule") != EDGE_SELECTION_RULE
        or provenance.get("edge_selection_rule")
        != (provenance.get("candidate_edge_selection_rule"))
    ):
        raise ValueError(
            "Prepared landscape checkpoint uses an incompatible scientific contract; "
            "run `just comparison-analysis` to rebuild it."
        )
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Clamped comparison graph summary is missing; "
            "run `just comparison-analysis` to rebuild it."
        )
    graph_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not (directory / CANDIDATE_THRESHOLD_FILE).is_file():
        raise FileNotFoundError(
            "Candidate functional threshold record is missing; "
            "run `just comparison-analysis` to rebuild it."
        )
    candidate_threshold = json.loads(
        (directory / CANDIDATE_THRESHOLD_FILE).read_text(encoding="utf-8")
    )
    recorded = provenance.get("thresholds", {}).get("candidate_clamped")
    if (
        not isinstance(candidate_threshold, dict)
        or not CANDIDATE_THRESHOLD_KEYS <= candidate_threshold.keys()
        or candidate_threshold["rounded_threshold"] != recorded
    ):
        raise ValueError(
            "Candidate functional threshold record is incomplete or disagrees with "
            "the checkpoint's Candidate threshold; "
            "run `just comparison-analysis` to rebuild it."
        )
    if graph_summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError(
            "Prepared landscape graph summary uses an incompatible schema; "
            "run `just comparison-analysis` to rebuild it."
        )
    for landscape in LANDSCAPE_KEYS:
        if not (directory / landscape / "data" / "metadata.toml").is_file():
            raise FileNotFoundError(
                f"Clamped landscape checkpoint is incomplete: {landscape}."
            )
    tables = {}
    for name, path in zip(expected_tables, _table_paths(directory, expected_tables)):
        if not path.is_file():
            raise FileNotFoundError(
                f"Clamped comparison table is missing: {path.name}."
            )
        tables[name] = pl.read_parquet(path)
    landscapes = {
        "full": load_cached_analysis(context=context) if include_full else None,
        **{
            name: load_network_analysis(directory / name / "data")
            for name in LANDSCAPE_KEYS
        },
    }
    for name in LANDSCAPE_KEYS:
        analysis = landscapes[name]
        scalar_attr = getattr(analysis, "scalar_attr", None)
        tree_scalar_attr = getattr(analysis.trees.get("split"), "scalar_attr", None)
        if (
            scalar_attr != RAW_ANALYSIS_SCALAR
            or tree_scalar_attr != RAW_ANALYSIS_SCALAR
        ):
            raise ValueError(
                f"Prepared landscape {name!r} uses analysis/tree scalars "
                f"{scalar_attr!r}/{tree_scalar_attr!r}; "
                "raw_fitness is required. Rebuild with `just comparison-analysis`."
            )
    return {
        "landscapes": landscapes,
        "thresholds": provenance["thresholds"],
        "diagnostics": json.loads(diagnostics_path.read_text(encoding="utf-8")),
        "tables": tables,
        "provenance": provenance,
        "graph_summary": graph_summary,
        "candidate_threshold": candidate_threshold,
    }


def load_cached_clamped_comparison(
    *,
    _allow_pending: bool = False,
    context: Any | None = None,
    include_full: bool | None = None,
) -> dict[str, Any]:
    """Load the last completed comparison while excluding concurrent publication.

    ``include_full`` loads the Full analysis into ``landscapes["full"]``; left
    ``None`` it follows the default (loaded without a context, skipped with one,
    since context-aware annotation builds consume only the study landscape). A
    reader that states a Full quantity, as manuscript quantities does, passes ``True``.
    """
    cache_dir, pending = cache_dirs(context)
    with checkpoint_lock(cache_dir, "comparison"):
        # Context-aware annotation builds only consume the study landscape.
        return _load_clamped_checkpoint(
            cache_dir,
            allow_pending=_allow_pending,
            include_full=context is None if include_full is None else include_full,
            pending_path=pending,
            context=context,
        )


def load_or_compute_clamped_comparison(
    *,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """Reuse or atomically refresh both threshold-specific tree checkpoints."""
    if context is None:
        context = get_run_context()
    with checkpoint_lock(cache_dirs(context)[0], "comparison"):
        return _load_or_compute_clamped_comparison_locked(
            force=force, settings=settings, context=context
        )


def _load_or_compute_clamped_comparison_locked(
    *,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings(context=context)
    cache_dir, pending_path = cache_dirs(context)
    # This is the existing full/raw analysis and its graph input workflow. Keep
    # its figure object as the first panel and derive both prepared graphs from
    # the same full network built by the public Papkou workflow.
    full_analysis = load_or_compute_analysis(settings=settings, context=context)
    fingerprint = _fingerprint(settings, context=context)
    recovered = recover_checkpoint(
        cache_dir,
        pending_path,
        "comparison",
        lambda path: _load_clamped_checkpoint(
            path, allow_pending=True, include_full=False
        ),
        expected_fingerprint=fingerprint,
        fingerprint_of=lambda value: value["provenance"].get("fingerprint"),
        retry_hint=RETRY_HINT,
    )
    if (
        not force
        and recovered is not None
        and recovered["provenance"].get("fingerprint") == fingerprint
    ):
        recovered["landscapes"]["full"] = full_analysis
        return recovered

    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text("clamped comparison refresh pending\n", encoding="utf-8")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".clamped-comparison-stage-", dir=cache_dir.parent)
    )
    try:
        scientific = settings["analysis"]
        clamped = scientific["clamped"]
        workflow = Papkou2023Workflow(
            landscape="full",
            fitness_wt=pl.read_parquet(
                dataset_input_path(
                    scientific["dataset"]["fitness_wt"], context, root=ROOT
                )
            ),
            fitness_diff=pl.read_parquet(
                dataset_input_path(
                    scientific["dataset"]["fitness_diff"], context, root=ROOT
                )
            ),
        )
        full_network = workflow.build_network()
        reference_network = load_papkou2023_article_network(ROOT / clamped["graphml"])
        reference_peaks = pl.read_csv(ROOT / clamped["peaks"])
        wild_type_sequence = str(workflow.fitness_wt.get_column("dna_from")[0])

        study = prepare_clamped_landscape(
            full_network, float(clamped["published_functional_threshold"])
        )
        reference = validate_clamped_reference(
            study, reference_network, reference_peaks
        )
        candidate, selection = prepare_selected_landscape(
            full_network, settings, context=context
        )
        candidate_threshold = selection.threshold
        prepared = {"study_clamped": study, "candidate_clamped": candidate}
        diagnostics = {
            "full": {
                "input_vertex_count": full_network.vertex_count,
                "input_edge_count": full_network.edge_count,
                "selected_component_vertex_count": full_network.vertex_count,
                "selected_component_edge_count": full_network.edge_count,
            },
            "study_clamped": {
                **prepared["study_clamped"].diagnostics,
                "reference": reference,
            },
            "candidate_clamped": prepared["candidate_clamped"].diagnostics,
        }
        thresholds = {
            "full": None,
            "study_clamped": float(clamped["published_functional_threshold"]),
            "candidate_clamped": candidate_threshold.threshold,
        }
        graph_summary = describe(
            products={"landscape_summary"},
            full_network=full_network,
            published_network=study.network,
            candidate_network=candidate.network,
            published_functional_threshold=float(
                clamped["published_functional_threshold"]
            ),
            candidate_functional_threshold=candidate_threshold.threshold,
            candidate_classes=selection.classes,
        ).landscape_summary
        analyses = {"full": full_analysis}
        for name in LANDSCAPE_KEYS:
            analyses[name] = analyze_prepared_landscape(
                prepared[name], name=name, wild_type_sequence=wild_type_sequence
            )
            save_network_analysis(analyses[name], stage / name / "data")
        table_dir = stage / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        tables = {}
        for name in LANDSCAPE_KEYS:
            for table_name, frame in (
                ("components", prepared[name].components),
                ("excluded_vertices", prepared[name].excluded_vertices),
                ("peaks", prepared[name].peaks),
            ):
                key = f"{name}_{table_name}"
                tables[key] = frame
                frame.write_parquet(table_dir / f"{key}.parquet")
            tree_tables = _tree_support_tables(name, analyses[name])
            tables.update(tree_tables)
            for key, frame in tree_tables.items():
                frame.write_parquet(table_dir / f"{key}.parquet")
        _write_json_atomic(stage / "diagnostics.json", diagnostics)
        _write_json_atomic(stage / "graph_summary.json", graph_summary)
        _write_json_atomic(
            stage / CANDIDATE_THRESHOLD_FILE,
            {
                **selection.to_json(),
                "selected_component_functional_vertex_count": candidate.diagnostics[
                    "selected_component_functional_vertex_count"
                ],
            },
        )
        _write_json_atomic(
            stage / "provenance.json",
            {
                "schema": CACHE_SCHEMA,
                "analysis_scalar": RAW_ANALYSIS_SCALAR,
                # The Candidate graph's rule, the one the checkpoint's headline
                # graph was selected by; the Published landscape keeps its own.
                "edge_selection_rule": candidate.diagnostics["edge_selection_rule"],
                "candidate_edge_selection_rule": candidate.diagnostics[
                    "edge_selection_rule"
                ],
                "published_edge_selection_rule": EDGE_SELECTION_RULE,
                "candidate_selection_rule": selection.rule,
                "fingerprint": fingerprint,
                "thresholds": thresholds,
                "reference_source_topographer_commit": clamped[
                    "reference_topographer_commit"
                ],
                # `_fingerprint` verified both files against these digests.
                "reference_graphml_value_sha256": clamped["graphml_value_sha256"],
                "reference_peaks_value_sha256": clamped["peaks_value_sha256"],
                "study_reference_validation": reference,
            },
        )
        result = publish_checkpoint(
            stage,
            cache_dir,
            pending_path,
            "comparison",
            lambda path: _load_clamped_checkpoint(
                path, allow_pending=True, include_full=False
            ),
            expected_fingerprint=fingerprint,
            fingerprint_of=lambda value: value["provenance"].get("fingerprint"),
            retry_hint=RETRY_HINT,
        )
        result["landscapes"]["full"] = full_analysis
        result["tables"] = tables
        result["graph_summary"] = graph_summary
        result["candidate_threshold"] = {
            **selection.to_json(),
            "selected_component_functional_vertex_count": candidate.diagnostics[
                "selected_component_functional_vertex_count"
            ],
        }
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
