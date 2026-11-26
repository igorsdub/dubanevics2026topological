"""Table builders (the ``produce_or_load`` stage publishers) and the table commands."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import (
    OUTPUT,
    ROOT,
    _write_json_atomic,
    dataset_input_path,
    load_settings,
)
from ..analysis.component_persistence import (
    load_cached_component_persistence,
    load_or_compute_component_persistence,
)
from ..analysis.produce_or_load import Checkpoint
from ..landscape.full_population import load_full_network_fitness
from ..landscape.selection import threshold_from_summary
from ..provenance import upstream_identity
from ..tables import export_component_persistence_table, export_tables
from .analysis_stages import (
    EXPORTED_SUMMARY,
    LEADING_SADDLE_BAND_CHECKPOINT,
    _biological_annotation_stage,
    _codon_entropy_stage,
    _comparison_analysis_fingerprint,
    _leading_saddle_band_stage,
    _manuscript_quantities_stage,
    _peak_analysis_fingerprint,
    _publish_stage_tables,
    ensure_analysis_checkpoint,
    load_cached_leading_saddle_band,
)
from .common import (
    ANNOTATION_TABLE_DIRECTORIES,
    _build_sources,
    _context_kwargs,
    _display_path,
    _load_settings,
    _stage,
)

CODON_ENTROPY_TABLES = (
    "papkou2023_candidate_population",
    "papkou2023_kem_ridge_population",
    "papkou2023_tcl_ridge_population",
    "papkou2023_codon_entropy_site_frequency",
    "papkou2023_codon_entropy_site_entropy",
    "papkou2023_codon_entropy_sequence_assignments",
    "papkou2023_codon_entropy_claim_summary",
    "papkou2023_candidate_aa_entropy",
    "papkou2023_kem_aa_entropy",
    "papkou2023_tcl_aa_entropy",
)


CODON_ENTROPY_FILES = tuple(
    f"{name}.{extension}"
    for name in CODON_ENTROPY_TABLES
    for extension in ("csv", "parquet")
) + ("papkou2023_codon_entropy_metadata.json",)


def _produce_codon_entropy_tables(
    directory: Path,
    *,
    comparison: dict[str, Any] | None,
    settings: dict[str, Any],
    peak_data: dict[str, Any],
    context: Any | None = None,
) -> None:
    """Write the codon-entropy tables into ``directory``.

    ``peak_data`` is loaded by the caller before the stage is declared (see
    ``build_codon_entropy_tables``).
    """
    analysis_settings = settings["analysis"]["codon_entropy"]
    if comparison is None:
        from ..analysis.clamped_cache import load_cached_clamped_comparison

        comparison = load_cached_clamped_comparison(**_context_kwargs(context))
    from ..analysis.codon_entropy import process_codon_entropy_datasets
    from ..analysis.codon_entropy_populations import extract_codon_entropy_populations
    from ..simplification.selected_leaf_simplification import build_candidate_leaf_view

    candidate_view = build_candidate_leaf_view(
        comparison["landscapes"]["candidate_clamped"],
        peak_data["tables"]["candidate_clamped_contrasts"],
        float(settings["analysis"]["candidate_peak_biology"]["q_value_threshold"]),
    )
    from ..landscape.selection import resolve_selection

    extract_codon_entropy_populations(
        comparison=comparison,
        candidate_view=candidate_view,
        config=analysis_settings,
        output_dir=directory,
        save_files=True,
        functional_sequences=resolve_selection(
            settings, **_context_kwargs(context)
        ).functional_sequences,
    )
    process_codon_entropy_datasets(
        input_dir=directory,
        output_dir=directory,
        save_files=True,
        num_thresholds=int(analysis_settings["sweep_points"]),
    )
    missing = [name for name in CODON_ENTROPY_FILES if not (directory / name).is_file()]
    if missing:
        raise RuntimeError("Codon entropy export is missing: " + ", ".join(missing))


def build_codon_entropy_tables(
    comparison: dict[str, Any] | None = None,
    *,
    force: bool = False,
    context: Any | None = None,
) -> list[Path]:
    """Prepare codon-entropy populations, entropy sweeps, and evidence tables."""
    from ..analysis.peak_uncertainty import load_or_compute_peak_uncertainty

    settings = _load_settings(context)
    # Bring the peak checkpoint up to date first: the stage keys on its record, so
    # a refresh during produce would make the sidecar stale on arrival.
    peak_data = load_or_compute_peak_uncertainty(settings=settings, context=context)
    return _publish_stage_tables(
        _codon_entropy_stage(settings, **_context_kwargs(context)),
        lambda directory: _produce_codon_entropy_tables(
            directory,
            comparison=comparison,
            settings=settings,
            peak_data=peak_data,
            context=context,
        ),
        checkpoint_name="codon_entropy",
        record_path=(context.output_root if context is not None else OUTPUT)
        / "tables"
        / ".codon-entropy-provenance.json",
        force=force,
        current_message="Codon entropy tables are current; reusing exported files.",
        context=context,
    )


def build_peak_tables(
    result: dict[str, Any] | None = None,
    *,
    force: bool = False,
    context: Any | None = None,
) -> list[Path]:
    """Export sequence-keyed peak evidence independently of figure appearance."""
    if result is None:
        from ..analysis.peak_uncertainty import load_or_compute_peak_uncertainty

        result = load_or_compute_peak_uncertainty(force=force, context=context)
    tables = result["tables"]
    for name in tables:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Unsafe peak evidence table name: {name!r}.")

    def produce(directory: Path) -> None:
        for name, frame in tables.items():
            frame.write_csv(directory / f"peak_uncertainty_{name}.csv")
            frame.write_parquet(directory / f"peak_uncertainty_{name}.parquet")
        _write_json_atomic(
            directory / "peak_uncertainty_diagnostics.json", result["diagnostics"]
        )

    stage = _stage(
        "peak-uncertainty-tables",
        settings={
            "tables": sorted(tables),
            "peak_checkpoint": _peak_analysis_fingerprint(context),
        },
        code=_build_sources(),
    )
    return _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="peak_uncertainty_tables",
        record_path=(context.output_root if context is not None else OUTPUT)
        / "tables"
        / ".peak-uncertainty-provenance.json",
        force=force,
        context=context,
    )


def _table_record_path(context: Any | None = None) -> Path:
    output_root = context.output_root if context is not None else OUTPUT
    return output_root / "tables" / ".provenance.json"


def build_tables(*, force: bool = False, context: Any | None = None) -> list[Path]:
    """Export stable-identity tables only when analysis or export code changed."""
    analysis = (
        ensure_analysis_checkpoint()
        if context is None
        else ensure_analysis_checkpoint(context=context)
    )
    component_pairs, _ = (
        load_or_compute_component_persistence(force=force)
        if context is None
        else load_or_compute_component_persistence(force=force, context=context)
    )
    output_root = context.output_root if context is not None else OUTPUT
    provenance_path = output_root / "analysis" / "provenance.json"
    # The analysis fingerprint is the stable identity of the science; the rest of
    # provenance.json (created_at, workflow dump) changes on a no-op re-publish.
    analysis_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    pairs_path = output_root / "analysis" / "component_pairs.parquet"

    def produce(directory: Path) -> None:
        export_tables(analysis, directory)
        export_component_persistence_table(component_pairs, directory)

    stage = _stage(
        "analysis-tables",
        settings={"analysis_fingerprint": analysis_provenance["fingerprint"]},
        inputs=(pairs_path,),
        code=(
            ROOT / "src/dubanevics2026topological/tables/__init__.py",
            *_build_sources(),
        ),
    )
    return _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="analysis_tables",
        record_path=_table_record_path(context=context),
        force=force,
        current_message="Tables are current; reusing exported files.",
        context=context,
    )


def build_comparison_tables(
    result: dict[str, Any] | None = None,
    *,
    force: bool = False,
    context: Any | None = None,
) -> list[Path]:
    """Export clamped-comparison supporting tables in CSV and Parquet formats."""
    if result is None:
        from ..analysis.clamped_cache import load_or_compute_clamped_comparison

        result = load_or_compute_clamped_comparison(force=force, context=context)
    tables = result.get("tables", {})
    export_frames: dict[str, Any] = {}
    csv_frames: dict[str, Any] = {}
    for name, frame in tables.items():
        export_frames[name] = frame
        csv_frames[name] = frame
        if "sequences" in frame.columns:
            csv_frames[name] = frame.drop("sequences")
            export_frames[f"{name}_members"] = frame.select(
                "component_id", pl.col("sequences").alias("dna_sequence")
            ).explode("dna_sequence", empty_as_null=True)
            csv_frames[f"{name}_members"] = export_frames[f"{name}_members"]
    for name in export_frames:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Unsafe comparison table name: {name!r}.")

    def produce(directory: Path) -> None:
        for name, frame in export_frames.items():
            csv_frames[name].write_csv(directory / f"clamped_{name}.csv")
            frame.write_parquet(directory / f"clamped_{name}.parquet")

    stage = _stage(
        "clamped-comparison-tables",
        settings={
            "tables": sorted(export_frames),
            "comparison_checkpoint": _comparison_analysis_fingerprint(context),
        },
        code=_build_sources(),
    )
    return _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="clamped_comparison_tables",
        record_path=(context.output_root if context is not None else OUTPUT)
        / "tables"
        / ".clamped-comparison-provenance.json",
        force=force,
        context=context,
    )


def build_landscape_summary_table(context: Any) -> list[Path]:
    """Export the seven-column landscape summary from its pinned saved products.

    Reads the products named in ``[tables.landscape_summary.labels]`` of
    ``config/figures.toml`` and recomputes nothing; the sidecar carries the
    seven scientific ids, so appearance settings never change the identity.
    """
    from ..tables.landscape_summary_table import export_summary_table

    settings = load_settings()
    section = settings["figures"]["tables"]["landscape_summary"]
    dataset = str(settings["analysis"]["dataset"]["name"]).lower()
    return export_summary_table(
        context,
        section["labels"],
        dataset,
        context.output_root / "tables",
        percent_decimals=int(section.get("percent_decimals", 2)),
    )


def build_presentation_tables(context: Any) -> list[Path]:
    """Export manuscript presentation tables sliced from the landscape summary.

    Table 1 Full, Published and Candidate, Table 2 Candidate
    Nominal|FDR|Bootstrap, and the supplementary Published simplification table.
    Same pinned products as
    ``landscape-summary-table``; column selection is presentation only.
    """
    from ..figures._quantities import (
        load_current_manuscript_quantities,
        registry_fingerprint,
        registry_path,
    )
    from ..tables.presentation_tables import export_presentation_tables

    settings = load_settings()
    section = settings["figures"]["tables"]["landscape_summary"]
    dataset = str(settings["analysis"]["dataset"]["name"]).lower()

    valley_source = None
    if registry_path(context.output_root).is_file():
        registry = load_current_manuscript_quantities(context.output_root, settings)
        valley_source = (
            registry.value("full_graph.persistence.valley_count"),
            registry_fingerprint(context.output_root),
        )
    outputs = export_presentation_tables(
        context,
        section["labels"],
        dataset,
        context.output_root / "tables",
        percent_decimals=int(section.get("percent_decimals", 2)),
        valley_source=valley_source,
    )
    outputs.extend(
        export_presentation_tables(
            context,
            section["labels"],
            dataset,
            context.output_root / "tables" / "canonical",
            percent_decimals=int(section.get("percent_decimals", 2)),
            valley_source=valley_source,
        )
    )
    return outputs


def _write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_manuscript_quantities(
    *, force: bool = False, context: Any | None = None
) -> list[Path]:
    """Build the manuscript quantity registry and its report.

    The registry reads the finished outputs of the upstream stages; an upstream
    that is absent becomes named gap rows (nothing is computed or typed here).
    The checkpoint holds the table; the report is written at export time beside
    it, after reading the previously exported CSV, so it can flag what changed
    since the last run. The report is not part of the fingerprint.
    """
    from ..analysis.git_stamp import git_stamp
    from ..analysis.manuscript_quantities import (
        REPORT_NAME,
        TABLE_STEM,
        TABLE_SUBDIRECTORY,
        build_rows,
        gap_paper_rows,
        load_sources,
        previous_display_values,
        render_report,
        write_tables,
    )

    output = OUTPUT if context is None else context.output_root
    # A test may replace load_settings with a zero-argument stub.

    accepts_context = "context" in inspect.signature(load_settings).parameters
    settings = (
        load_settings(context=context)
        if context is not None and accepts_context
        else load_settings()
    )
    sources = load_sources(output)
    stage, config = _manuscript_quantities_stage(settings, sources)

    def produce(directory: Path) -> None:
        commit = git_stamp(ROOT)["commit"]
        rows = build_rows(
            sources, config, git_commit=commit if isinstance(commit, str) else None
        )
        if os.environ.get("FITNESSTREE_REQUIRE_PAPER_ROWS", "").strip():
            gaps = gap_paper_rows(rows)
            if gaps:
                raise RuntimeError(
                    "A production run cannot leave a paper row as a gap: "
                    + ", ".join(gaps)
                )
        write_tables(directory, rows)
        build_rows.position_entropy.write_parquet(
            directory / "entropy_by_position.parquet"
        )

    record_path = output / "tables" / ".manuscript-quantities-provenance.json"

    def before(destination: Path) -> dict[str, str | None] | None:
        # No record means the last export was interrupted (or never happened): the
        # report beside it may belong to an older registry, so it is removed and
        # rendered again below.
        if not record_path.is_file():
            (destination / REPORT_NAME).unlink(missing_ok=True)
        return previous_display_values(destination / f"{TABLE_STEM}.csv")

    def after(
        ready: Checkpoint,
        destination: Path,
        written: list[str],
        previous: dict[str, str | None] | None,
    ) -> None:
        report = destination / REPORT_NAME
        if not written and report.is_file():
            return
        current = pl.read_parquet(ready.path / f"{TABLE_STEM}.parquet").to_dicts()
        _write_text_atomic(report, render_report(current, previous))

    destination = output / "tables" / TABLE_SUBDIRECTORY
    paths = _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="manuscript_quantities",
        record_path=record_path,
        force=force,
        current_message="Manuscript quantities are current; reusing exported files.",
        subdirectory=TABLE_SUBDIRECTORY,
        before_export=before,
        after_export=after,
    )
    return [*paths, destination / REPORT_NAME]


def build_leading_saddle_band_tables(
    *, force: bool = False, context: Any | None = None
) -> list[Path]:
    """Export the Full-graph landmark, interval and entropy tables with provenance.

    Reads the cached component pairs (never recomputes them), the Full-graph
    raw fitness and the Full graph built from ``fitness_wt`` and ``fitness_diff``.
    """
    from ..analysis.leading_saddle_band import (
        compute_leading_saddle_band,
        write_leading_saddle_band,
    )
    from ..threshold.functional_threshold import configured_declaration

    settings = _load_settings(context)
    pairs, _ = load_cached_component_persistence(**_context_kwargs(context))
    clamped = settings["analysis"]["clamped"]
    output_root = context.output_root if context is not None else OUTPUT

    def produce(directory: Path) -> None:
        population = load_full_network_fitness(settings, **_context_kwargs(context))
        result = compute_leading_saddle_band(
            pairs,
            population,
            graph=_full_graph(settings, context),
            published_threshold=clamped["published_functional_threshold"],
            candidate_declaration=configured_declaration(settings),
            candidate=threshold_from_summary(
                settings,
                raw_fitness=population["raw_fitness"].to_list(),
                **_context_kwargs(context),
            ),
        )
        write_leading_saddle_band(directory, result)

    return _publish_stage_tables(
        _leading_saddle_band_stage(settings, **_context_kwargs(context)),
        produce,
        checkpoint_name=LEADING_SADDLE_BAND_CHECKPOINT,
        record_path=output_root / "tables" / ".leading-saddle-band-provenance.json",
        force=force,
        context=context,
        current_message="Leading saddle band tables are current; reusing exported files.",
        # summary.json is too generic a name for the shared tables directory.
        names={"summary.json": EXPORTED_SUMMARY},
    )


def _full_graph(settings: dict[str, Any], context: Any | None) -> Any:
    """The Full-graph rustworkx ``PyGraph`` (payloads carry ``dna_sequence`` and ``s``)."""
    from dubanevics2026topological.papkou2023 import Papkou2023Workflow

    dataset = settings["analysis"]["dataset"]
    network = Papkou2023Workflow(
        landscape=dataset["landscape"],
        fitness_wt=pl.read_parquet(
            dataset_input_path(dataset["fitness_wt"], context, root=ROOT)
        ),
        fitness_diff=pl.read_parquet(
            dataset_input_path(dataset["fitness_diff"], context, root=ROOT)
        ),
    ).build_network()
    return network.graph


def _report_leading_saddle_band(*, force: bool, context: Any | None) -> None:
    """Publish the stage tables, then describe the checkpoint just written.

    The checkpoint lives in the stage's own folder (``result_directory``), so it
    is read back through ``load_cached_leading_saddle_band``, never from the step
    root.
    """
    from ..analysis.leading_saddle_band import describe

    for path in build_leading_saddle_band_tables(force=force, context=context):
        print(f"Table: {_display_path(path)}")
    print(describe(load_cached_leading_saddle_band(context=context)))


def build_biological_tables(
    result: dict[str, Any], *, force: bool = False, context: Any | None = None
) -> list[Path]:
    """Export biological evidence from the cached study landscape independently of style.

    Run by the ``biological-annotation-tables`` command (``just
    biological-annotation-tables``), which the manuscript quantity registry
    depends on: the registry reads these tables, so no figure build writes them.
    """
    from ..analysis.describe import describe

    def produce(directory: Path) -> None:
        expected = [
            f"biological_annotation_{name}.{extension}"
            for name in ("branches", "pairs", "critical_points")
            for extension in ("csv", "parquet")
        ] + ["biological_annotation_counts.json"]
        for landscape, location in ANNOTATION_TABLE_DIRECTORIES.items():
            target = directory / location
            describe(
                products={"annotations"},
                landscape=result["landscapes"][landscape],
                landscape_name=landscape,
                annotation_directory=target,
            )
            missing = [name for name in expected if not (target / name).is_file()]
            if missing:
                raise RuntimeError(
                    "Biological annotation export did not produce all evidence files: "
                    + ", ".join(missing)
                )

    stage = _biological_annotation_stage(context)
    record_path = (
        (context.output_root / "tables" / ".biological-annotation-provenance.json")
        if context is not None
        else (OUTPUT / "tables" / ".biological-annotation-provenance.json")
    )
    return _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="biological_annotation_tables",
        record_path=record_path,
        force=force,
        context=context,
    )


def build_candidate_tree_contrast_tables(
    result: Any, *, force: bool = False, context: Any | None = None
) -> list[Path]:
    """Export the tree-edge pair list and contrast table with sidecar provenance.

    The stage fingerprint keys on the contrast checkpoint's fingerprint and
    artefact hashes (not its timestamps), so the tables are reused exactly while
    that checkpoint is unchanged.
    """

    def produce(directory: Path) -> None:
        _write_frames(
            directory,
            (
                ("papkou2023_candidate_tree_contrast_pairs", result.pairs),
                ("papkou2023_candidate_tree_contrast_edges", result.edges),
            ),
        )

    stage = _stage(
        "candidate-tree-contrast-tables",
        settings={"contrast_checkpoint": upstream_identity(result.provenance)},
        code=_build_sources(),
    )
    return _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="candidate_tree_contrast_tables",
        record_path=(context.output_root if context is not None else OUTPUT)
        / "tables"
        / ".candidate-tree-contrast-provenance.json",
        force=force,
        current_message=(
            "Candidate tree contrast tables are current; reusing exported files."
        ),
        context=context,
    )


def build_sequence_growth_rate_tables(
    result: Any, *, force: bool = False, context: Any | None = None
) -> list[Path]:
    """Export the per-sequence growth rates with sidecar provenance.

    Keyed on the growth-rate checkpoint's fingerprint and artefact hashes, like
    the contrast tables.
    """

    def produce(directory: Path) -> None:
        _write_frames(
            directory, (("papkou2023_sequence_growth_rates", result.growth_rates),)
        )

    stage = _stage(
        "sequence-growth-rate-tables",
        settings={"growth_rate_checkpoint": upstream_identity(result.provenance)},
        code=_build_sources(),
        root=context.root if context is not None else ROOT,
    )
    return _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="sequence_growth_rate_tables",
        record_path=(context.output_root if context is not None else OUTPUT)
        / "tables"
        / ".sequence-growth-rates-provenance.json",
        force=force,
        current_message=(
            "Sequence growth-rate tables are current; reusing exported files."
        ),
        context=context,
    )


def _write_frames(directory: Path, frames: Iterable[tuple[str, pl.DataFrame]]) -> None:
    """Write each frame as ``<name>.csv`` and ``<name>.parquet`` into ``directory``."""
    for name, frame in frames:
        frame.write_csv(directory / f"{name}.csv")
        frame.write_parquet(directory / f"{name}.parquet")


def build_bootstrap_smoothed_tree_tables(
    result: Any, *, force: bool = False, context: Any | None = None
) -> list[Path]:
    """Export direct-band hill and draw tables with checkpoint provenance.

    Core and vertex-role exports are retired; historical evidence remains in
    its original archive. The stage keys on the checkpoint's identity and hashes.
    """

    def produce(directory: Path) -> None:
        from ..simplification.bootstrap_smoothed_tree import HILL_COLUMNS

        _write_frames(
            directory,
            (
                ("papkou2023_bootstrap_hills", result.hills.select(HILL_COLUMNS)),
                ("papkou2023_bootstrap_draw_stats", result.draw_stats),
            ),
        )

    def refuse_retired_exports(directory: Path) -> None:
        retired = [
            directory / f"papkou2023_bootstrap_{table}.{extension}"
            for table in ("cores", "vertex_roles")
            for extension in ("csv", "parquet")
            if (directory / f"papkou2023_bootstrap_{table}.{extension}").exists()
        ]
        if retired:
            raise RuntimeError(
                "Retired core-era bootstrap table exports remain in output/tables: "
                + ", ".join(map(str, retired))
                + ". Remove or relocate these stale files before exporting direct-band tables."
            )

    provenance = result.provenance
    stage = _stage(
        "bootstrap-smoothed-tree-tables",
        settings={"bootstrap_checkpoint": upstream_identity(provenance)},
        code=_build_sources(),
    )
    return _publish_stage_tables(
        stage,
        produce,
        checkpoint_name="bootstrap_smoothed_tree_tables",
        record_path=(context.output_root if context is not None else OUTPUT)
        / "tables"
        / ".bootstrap-smoothed-tree-provenance.json",
        force=force,
        before_export=refuse_retired_exports,
        context=context,
    )


def cmd_comparison_analysis(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.clamped_cache import load_or_compute_clamped_comparison

    result = load_or_compute_clamped_comparison(force=args.force, context=context)
    build_comparison_tables(result, force=args.force, context=context)
    print("Clamped comparison ready: " + ", ".join(result["diagnostics"].keys()))
    return 0


def cmd_four_graph_comparisons(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    # The clamped comparison of each of the four graphs the 2x2 figures draw, each under
    # <output root>/variants/<variant>/ (the checkpoints build/four_graph_inputs.py reads).
    from ..analysis.clamped_cache import load_or_compute_clamped_comparison
    from .four_graph_inputs import VARIANTS, variant_context

    for variant in VARIANTS:
        scoped = variant_context(context, variant)
        result = load_or_compute_clamped_comparison(force=args.force, context=scoped)
        build_comparison_tables(result, force=args.force, context=scoped)
        print(f"Clamped comparison ready for {variant}: " + ", ".join(result["diagnostics"].keys()))
    return 0


def cmd_landscape_summary_table(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    for path in build_landscape_summary_table(context):
        print(f"Table: {_display_path(path)}")
    return 0


def cmd_presentation_tables(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    for path in build_presentation_tables(context):
        print(f"Table: {_display_path(path)}")
    return 0


def cmd_graph_descriptive_statistics(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..tables.graph_descriptive_statistics import publish

    for path in publish(context):
        print(f"Table: {_display_path(path)}")
    return 0


def cmd_candidate_band_comparison_table(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..results import RunContext
    from ..tables.candidate_band_comparison import (
        VARIANT,
        build_candidate_band_comparison,
    )

    if context.variant_name != VARIANT:
        context = RunContext(
            output_root=ROOT / "output" / "variants" / VARIANT,
            variant_name=VARIANT,
        )
    for path in build_candidate_band_comparison(context):
        print(f"Table: {_display_path(path)}")
    return 0


def cmd_manuscript_quantities(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    for path in build_manuscript_quantities(force=args.force, context=context):
        print(f"Table: {path.relative_to(ROOT)}")
    return 0


def cmd_fdr_peak_selection(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.peak_uncertainty import load_or_compute_peak_uncertainty

    result = load_or_compute_peak_uncertainty(force=args.force, context=context)
    build_peak_tables(result, force=args.force, context=context)
    print("FDR peak selection saved.")
    return 0


def cmd_biological_annotation_tables(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.clamped_cache import load_cached_clamped_comparison

    for path in build_biological_tables(
        load_cached_clamped_comparison(context=context),
        force=args.force,
        context=context,
    ):
        print(f"Table: {path.relative_to(context.output_root)}")
    return 0


def cmd_tables(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    for path in build_tables(force=args.force, context=context):
        print(f"Table: {path.relative_to(context.output_root)}")
    return 0


def cmd_leading_saddle_band(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    _report_leading_saddle_band(force=args.force, context=context)
    return 0


def cmd_peak_analysis(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.peak_uncertainty import load_or_compute_peak_uncertainty

    result = load_or_compute_peak_uncertainty(force=args.force, context=context)
    build_peak_tables(result, force=args.force, context=context)
    print("Peak uncertainty analysis and evidence tables ready.")
    return 0


def cmd_codon_entropy_analysis(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    for path in build_codon_entropy_tables(force=args.force, context=context):
        print(f"Table: {_display_path(path)}")
    return 0
