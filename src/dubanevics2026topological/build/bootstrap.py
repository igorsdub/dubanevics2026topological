"""Bootstrap, Candidate-graph and functional-classification commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


from ..analysis import ROOT, load_settings
from .common import _display_path
from .tables import (
    build_bootstrap_smoothed_tree_tables,
    build_candidate_tree_contrast_tables,
    build_sequence_growth_rate_tables,
)


def active_bootstrap_band_identity(checkpoint: Any) -> str:
    """Derive the direct-band identity only after checking complete draw coverage."""
    from ..simplification.identities import band_identity, checkpoint_draw_identity

    config = checkpoint.summary["settings"]
    draw_ids = list(range(int(config["n_replicates"])))
    for condition in ("real", "flat"):
        table = checkpoint.draw_stats
        observed = table.filter(table["condition"] == condition)["draw"].sort().to_list()
        if observed != draw_ids:
            raise ValueError("Bootstrap checkpoint has incomplete or duplicated draw IDs.")
    if checkpoint.draw_stats.height != 2 * len(draw_ids):
        raise ValueError("Bootstrap checkpoint has unexpected draw conditions.")
    _, draw_id = checkpoint_draw_identity(checkpoint.summary["fingerprint"],
                                          seed=int(config["seed"]), draw_ids=draw_ids)
    _, band_id = band_identity(draw_scientific_id_sha256=draw_id, bandq=float(config["band_quantile"]))
    return band_id


def _functional_threshold_scan(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    """The ``functional-threshold-scan`` command in its three forms.

    Not a ``produce_or_load`` stage: a tables-only export like
    ``bootstrap-band-quantile-scan``, written to ``SCAN_DIR`` with ``provenance.json``
    last. Unsliced: every resample here over ``n_jobs`` workers (``$FITNESSTREE_N_JOBS``,
    else ``[functional_threshold_scan] n_jobs``), skipped when a scan matching the
    current settings exists unless ``--force``. ``--slice START STOP --out DIR``
    computes a sub-range into ``DIR``; ``--merge DIR...`` combines slices and finishes.
    """
    from ..analysis import functional_threshold_scan as scan_module

    settings = load_settings()
    scan_dir = scan_module.SCAN_DIR
    section = settings["analysis"][scan_module.SECTION]
    n_resamples = scan_module.validate_config(section).n_resamples
    if args.slice is not None:
        start, stop = args.slice
        if not 0 <= start < stop <= n_resamples:
            parser.error(
                f"--slice {start} {stop} is not inside 0..{n_resamples} "
                "(n_resamples; STOP is exclusive and must exceed START)."
            )
        scan_module.compute_resample_slice(range(start, stop), settings, args.out)
        print(f"Slice: {args.out} (resamples {start} to {stop - 1})")
        return 0
    if args.merge is not None:
        try:
            merged = scan_module.merge_resample_slices(
                args.merge,
                n_resamples,
                current=scan_module.current_slice_identity(settings),
                cells=scan_module.scan_cells(settings),
            )
        except (FileNotFoundError, ValueError) as error:
            parser.exit(1, f"functional-threshold-scan: cannot merge: {error}\n")
        scan = scan_module.compute_functional_threshold_scan(
            settings, resamples=merged, scan_dir=scan_dir
        )
    else:
        if not args.force:
            try:
                current = scan_module.load_functional_threshold_scan(
                    settings, scan_dir=scan_dir
                )
            except (FileNotFoundError, RuntimeError):
                current = None
            if current is not None:
                print(
                    "Functional threshold scan is current; "
                    f"{_describe_scan(current)} (--force recomputes)."
                )
                return 0
        scan = scan_module.compute_functional_threshold_scan(
            settings,
            scan_dir=scan_dir,
            n_jobs=scan_module.resolve_n_jobs(section),
        )
    for name in scan_module.ARTIFACT_NAMES:
        path = scan_dir / name
        print(f"Table: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"Functional threshold scan ready: {_describe_scan(scan)}.")
    return 0


def _describe_scan(scan) -> str:
    passed = scan.acceptance["passed"]
    words = {True: "pass", False: "FAIL", None: "undefined"}
    checks = ", ".join(f"{key} {words[passed[key]]}" for key in "abc")
    return f"{scan.cells.height} cells, acceptance {checks}"


def cmd_candidate_edge_analysis(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.candidate_edge_evidence import (
        load_or_compute_candidate_edge_evidence,
    )

    result = load_or_compute_candidate_edge_evidence(
        force=args.force, context=context
    )
    print(f"Candidate evidence ready: {result['edges'].height} edges.")
    return 0


def cmd_candidate_tree_contrast(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.candidate_tree_contrast import compute_candidate_tree_contrast

    contrast = compute_candidate_tree_contrast(
        load_settings(context=context), force=args.force, context=context
    )
    for path in build_candidate_tree_contrast_tables(
        contrast, force=args.force, context=context
    ):
        print(f"Table: {path.relative_to(context.output_root)}")
    print(
        "Candidate tree contrast ready: "
        f"{contrast.diagnostics['test_count']} edges, "
        f"{contrast.diagnostics['qualifying_count']} with q > alpha."
    )
    return 0


def cmd_sequence_growth_rates(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.sequence_growth_rates import compute_sequence_growth_rates

    rates = compute_sequence_growth_rates(
        load_settings(context=context), force=args.force, context=context
    )
    for path in build_sequence_growth_rate_tables(
        rates, force=args.force, context=context
    ):
        print(f"Table: {path.relative_to(context.output_root)}")
    from ..threshold import reference_drift

    cache_dir = Path(context.output_root) / "analysis" / "sequence_growth_rates"
    table = cache_dir / "growth_rates.parquet"
    if table.is_file():
        reference_drift.check_drift(
            table,
            label="growth_rates",
            reference_sha256=reference_drift.reference_hashes()["growth_rates_sha256"],
            reference_pattern="growth_rates.parquet",
            report_dir=Path(context.output_root) / "analysis" / "drift_reports",
        )
    info = rates.diagnostics
    print(
        "Sequence growth rates ready: "
        f"{info['sequence_count']} sequences, "
        f"max identity gap {info['max_identity_gap']:.2e}."
    )
    return 0


def cmd_functional_classification(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..threshold.functional_classification import (
        compute_functional_classification,
        describe,
    )

    print(
        describe(
            compute_functional_classification(
                load_settings(context=context), force=args.force, context=context
            )
        )
    )
    return 0


def cmd_bootstrap_smoothed_tree(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..simplification.bootstrap_smoothed_tree import (
        compute_bootstrap_smoothed_tree,
    )

    bootstrap = compute_bootstrap_smoothed_tree(
        load_settings(context=context),
        force=args.force,
        fresh=args.fresh,
        context=context,
    )
    for path in build_bootstrap_smoothed_tree_tables(
        bootstrap, force=args.force, context=context
    ):
        print(f"Table: {path.relative_to(ROOT)}")
    info = bootstrap.summary
    band = info["band_simplification"]
    print(
        "Bootstrap-smoothed tree ready: "
        f"{band['peaks_before'] - band['peaks_after']} hills absorbed, "
        f"{band['peaks_after']} kept; {band['peaks_before']} peaks before, "
        f"{band['peaks_after']} after the persistence band."
    )
    return 0


def cmd_bootstrap_band_quantile_scan(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    from ..analysis.bootstrap_band_quantile_scan import (
        compute_bootstrap_band_quantile_scan,
        scan_paths,
        steps_around,
    )

    settings = load_settings(context=context)
    scan = compute_bootstrap_band_quantile_scan(settings, context=context)
    quantile = settings["analysis"]["bootstrap_smoothed_tree"]["band_quantile"]
    steps = steps_around(scan, quantile)
    for table in scan_paths(context):
        print(f"Table: {_display_path(table)}")
    print(f"Band quantile scan ready: {scan.height} grid points; {steps}.")
    return 0


def cmd_functional_threshold_scan(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    return _functional_threshold_scan(args, parser)


def cmd_classification_probability_scans(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    # Same entry point as `just classification-probability-scans`.
    from ..threshold.classification_probability_scans import main as scan_main

    scan_main()
    return 0
