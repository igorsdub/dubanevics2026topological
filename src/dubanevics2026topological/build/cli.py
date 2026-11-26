"""Argument parsing and the command table of ``python -m dubanevics2026topological.build``."""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..analysis import OUTPUT, ROOT
from .analysis_stages import (
    cmd_analysis,
    cmd_branch_decomposition,
    cmd_component_analysis,
    cmd_peak_pair_distances,
    cmd_status,
    cmd_verify_analysis,
)
from .bootstrap import (
    cmd_bootstrap_band_quantile_scan,
    cmd_bootstrap_smoothed_tree,
    cmd_candidate_edge_analysis,
    cmd_candidate_tree_contrast,
    cmd_classification_probability_scans,
    cmd_functional_classification,
    cmd_functional_threshold_scan,
    cmd_sequence_growth_rates,
)
from .common import ROTATED_LABEL_COLLISION
from .figures import cmd_all, cmd_figures, register_font_directory
from .products import (
    cmd_bootstrap_band_quantile_scan_product,
    cmd_bootstrap_codon_entropy_product,
    cmd_fdr_pooled_scatter_product,
    cmd_graph_tree_persistence_product,
    cmd_landscape_products,
    cmd_landscape_summary_product,
    cmd_merge_flow_product,
    cmd_peak_panels_product,
    cmd_pooled_scatter_product,
    cmd_ridge_entropy_product,
    cmd_ridges_product,
    cmd_threshold_diagnostics_product,
    cmd_leaves_product,
    cmd_tree_layout_products,
)
from .tables import (
    cmd_biological_annotation_tables,
    cmd_candidate_band_comparison_table,
    cmd_codon_entropy_analysis,
    cmd_comparison_analysis,
    cmd_fdr_peak_selection,
    cmd_four_graph_comparisons,
    cmd_graph_descriptive_statistics,
    cmd_landscape_summary_table,
    cmd_leading_saddle_band,
    cmd_manuscript_quantities,
    cmd_peak_analysis,
    cmd_presentation_tables,
    cmd_tables,
)

OUTPUT_AT_IMPORT = OUTPUT


ANALYSIS_VARIANT_AT_IMPORT = (
    os.environ.get("FITNESSTREE_ANALYSIS_VARIANT", "").strip() or None
)


def _watch() -> None:
    from watchfiles import watch

    subprocess.run(["just", "all"], cwd=ROOT, check=True)
    watched = [ROOT / path for path in ("src", "config", "notebooks", "data/raw")]
    watched.extend((ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / "justfile"))
    ignored_parts = {"output", ".git", ".venv", "__pycache__", ".marimo"}
    print("Watching source, configuration, notebooks, and raw data; Ctrl+C to stop.")
    for changes in watch(*watched, debounce=700, step=250):
        paths = [Path(path) for _, path in changes]
        relevant = [p for p in paths if not ignored_parts.intersection(p.parts)]
        if not relevant:
            continue
        print("Changes detected; rebuilding affected outputs.")
        science_changed = any(
            p in {ROOT / "uv.lock", ROOT / "pyproject.toml"}
            or p.is_relative_to(ROOT / "data/raw")
            or p == ROOT / "config/analysis.toml"
            or p.is_relative_to(ROOT / "src/dubanevics2026topological/analysis")
            for p in relevant
        )
        target = (
            "all"
            if science_changed
            else (
                "tables"
                if all(
                    p.is_relative_to(ROOT / "src/dubanevics2026topological/tables")
                    for p in relevant
                )
                else "figures"
            )
        )
        subprocess.run(["just", target], cwd=ROOT, check=False)


def main(
    argv: list[str] | None = None,
    *,
    context: Any | None = None,
) -> int:
    from ..results import add_run_context_arguments

    parser = argparse.ArgumentParser(prog="fitnesstree-build")
    add_run_context_arguments(parser)
    parser.add_argument("command", choices=tuple(COMMANDS))
    parser.add_argument(
        "--force",
        action="append",
        nargs="?",
        const="*",
        default=[],
        metavar="STEP",
        help=(
            "Recompute STEP even when its key matches, replacing that folder "
            "in place. Bare --force recomputes the command being run. Downstream "
            "steps rerun only when the forced step's output changed."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "bootstrap-smoothed-tree: move the saved bootstrap chunks aside and "
            "redraw everything (the default resumes from them); implies --force."
        ),
    )
    parser.add_argument(
        "--slice",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help=(
            "functional-threshold-scan: slice mode. Compute only resamples START to "
            "STOP (STOP exclusive, within 0..n_resamples) into --out; nothing else is "
            "written (combine the slices with --merge)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="functional-threshold-scan --slice: the directory for the slice's files.",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        metavar="SLICE_DIR",
        help=(
            "functional-threshold-scan: merge mode. Combine these slice directories "
            "(every resample once), then compute the point scan, comparator, intervals "
            "and acceptance checks and write the scan."
        ),
    )
    parser.add_argument("--figure", action="append", dest="figures")
    parser.add_argument(
        "--keep-going", action="store_true",
        help="figures: attempt every figure variant and report failures before exiting nonzero.",
    )
    parser.add_argument(
        "--slides",
        action="store_true",
        help=(
            "figures: after the prerequisites, render the frameless slide versions of "
            "the registry tree figures into figures/slides/{tree_only,full}/ (no titles; "
            "vanilla/annotated, count/none coloring) instead of the figures."
        ),
    )
    parser.add_argument(
        "--with-bootstrap",
        action="store_true",
        help=(
            "figures: also render the bootstrap figures (the two band-quantile-scan "
            "ones included, from a matching scan dataset) beside the default "
            "figures; restores the bootstrap by fingerprint and never draws. Not "
            "with --figure."
        ),
    )
    parser.add_argument(
        "--opt-in",
        action="store_true",
        help=(
            "figures: render every registered figure outside the default selection "
            "(the ones no manuscript entry needs; `just figures-exploratory`). Not "
            "with --figure, --with-bootstrap or --label-collision."
        ),
    )
    parser.add_argument(
        "--label-collision",
        choices=(ROTATED_LABEL_COLLISION,),
        help=(
            "figures: render the band-tree figures with their branch count labels "
            "turned a quarter turn (overriding branch_count_label_collision for "
            "this run only) into <figure_id>_rotated.<format>; the canonical PNGs "
            "and `just figures` are untouched. Without --figure it renders the "
            "three band-tree figures."
        ),
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=("svg", "png"),
        dest="formats",
        help="Export only this format, overriding config/figures.toml (repeatable).",
    )
    parser.add_argument(
        "--no-titled-twins",
        action="store_false",
        dest="titled_twins",
        default=True,
        help=(
            "figures: write only the untitled render to canonical/. By default each "
            "figure is also drawn with its figure heading to canonical/titled/."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=("vanilla", "annotated"),
        dest="variants",
        help="Build only the selected figure variant (may be repeated).",
    )
    args = parser.parse_args(argv)
    if context is None:
        # Preserve the legacy BarSeq `.env` source while making the constructed
        # context the sole environment snapshot consumed by the stages.
        # Bind the output folder and variant from the environment at call time.
        # A cluster job sets them before the process starts; a caller that already
        # passed a context is left on the bindings it set.
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        package = sys.modules[__package__]
        env_root = os.environ.get("FITNESSTREE_OUTPUT_ROOT", "output").strip() or "output"
        env_path = Path(env_root)
        package.OUTPUT_AT_IMPORT = (
            env_path if env_path.is_absolute() else ROOT / env_path
        ).resolve()
        package.ANALYSIS_VARIANT_AT_IMPORT = (
            os.environ.get("FITNESSTREE_ANALYSIS_VARIANT", "").strip() or None
        )
        from ..results import parse_run_context

        context = parse_run_context(args, root=ROOT)

    has_context_override = context.output_root != OUTPUT_AT_IMPORT or (
        context.variant_name != ANALYSIS_VARIANT_AT_IMPORT
    )
    if has_context_override and args.command not in CONTEXT_AWARE_COMMANDS:
        parser.error(
            f"Run-context overrides are not yet supported by the {args.command!r} "
            "command; this command still uses import-time output paths. Overrides "
            "are supported for: " + ", ".join(sorted(CONTEXT_AWARE_COMMANDS)) + "."
        )

    with context.activated():
        register_font_directory(context=context)
        return _dispatch_command(args, parser, context=context)


def _dispatch_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    if args.keep_going and (args.command != "figures" or args.slides):
        parser.error("--keep-going applies to the figures command without --slides.")
    if args.label_collision is not None and args.command != "figures":
        parser.error("--label-collision applies to the figures command only.")
    if args.with_bootstrap and (
        args.command != "figures" or args.figures or args.label_collision
    ):
        parser.error(
            "--with-bootstrap applies to the figures command alone, without "
            "--figure or --label-collision."
        )
    if args.opt_in and (
        args.command != "figures"
        or args.figures
        or args.with_bootstrap
        or args.label_collision
    ):
        parser.error(
            "--opt-in applies to the figures command alone, without --figure, "
            "--with-bootstrap or --label-collision."
        )
    scan_modes = [m for m in (args.slice, args.merge) if m is not None]
    if (scan_modes or args.out is not None) and args.command != (
        "functional-threshold-scan"
    ):
        parser.error("--slice, --out and --merge apply to functional-threshold-scan.")
    if args.command == "functional-threshold-scan":
        if len(scan_modes) > 1:
            parser.error("--slice and --merge exclude each other.")
        if args.slice is not None and args.out is None:
            parser.error("--slice needs --out DIR.")
        if args.out is not None and args.slice is None:
            parser.error("--out applies to --slice.")
        if scan_modes and args.force:
            parser.error(
                "--slice and --merge take no --force: a slice is "
                "recomputed every time and only the merge writes the scan."
            )
    requested = list(args.force or [])

    def step_force(*names: str) -> bool:
        return "*" in requested or any(name in requested for name in names)

    # A single command is forced by bare --force or by its own name. ``all``
    # keeps the set so each step can be named on its own.
    if args.command != "all":
        args.force = step_force(args.command)
    else:
        args.force = "*" in requested
    args.force_steps = requested
    return COMMANDS[args.command].handler(args, parser, context=context)


def cmd_watch(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    _watch()
    return 0


@dataclasses.dataclass(frozen=True)
class Command:
    """One build command: its handler and whether it honours run-context overrides."""

    handler: Callable[..., int]
    context_aware: bool


COMMANDS: dict[str, Command] = {
    "analysis": Command(cmd_analysis, context_aware=True),
    "verify-analysis": Command(cmd_verify_analysis, context_aware=True),
    "component-analysis": Command(cmd_component_analysis, context_aware=True),
    "comparison-analysis": Command(cmd_comparison_analysis, context_aware=True),
    "four-graph-comparisons": Command(cmd_four_graph_comparisons, context_aware=True),
    "landscape-products": Command(cmd_landscape_products, context_aware=True),
    "leaves-product": Command(cmd_leaves_product, context_aware=True),
    "tree-layout-products": Command(cmd_tree_layout_products, context_aware=True),
    "landscape-summary-product": Command(cmd_landscape_summary_product, context_aware=True),
    "ridges-product": Command(cmd_ridges_product, context_aware=True),
    "merge-flow-product": Command(cmd_merge_flow_product, context_aware=True),  # historical / retired from publication (Ticket 19 / PR #214); kept for legacy rebuilds only; NOT in D10_VIEWS or `all`/`figures`
    "ridge-entropy-product": Command(cmd_ridge_entropy_product, context_aware=True),
    "landscape-summary-table": Command(cmd_landscape_summary_table, context_aware=True),
    "presentation-tables": Command(cmd_presentation_tables, context_aware=True),
    "graph-descriptive-statistics": Command(cmd_graph_descriptive_statistics, context_aware=True),
    "candidate-band-comparison-table": Command(cmd_candidate_band_comparison_table, context_aware=False),
    "pooled-scatter-product": Command(cmd_pooled_scatter_product, context_aware=False),
    "fdr-pooled-scatter-product": Command(cmd_fdr_pooled_scatter_product, context_aware=False),
    "peak-panels-product": Command(cmd_peak_panels_product, context_aware=False),
    "threshold-diagnostics-product": Command(cmd_threshold_diagnostics_product, context_aware=False),
    "bootstrap-codon-entropy-product": Command(cmd_bootstrap_codon_entropy_product, context_aware=True),
    "bootstrap-band-quantile-scan-product": Command(cmd_bootstrap_band_quantile_scan_product, context_aware=True),
    "graph-tree-persistence-product": Command(cmd_graph_tree_persistence_product, context_aware=False),
    "manuscript-quantities": Command(cmd_manuscript_quantities, context_aware=True),
    "fdr-peak-selection": Command(cmd_fdr_peak_selection, context_aware=True),
    "biological-annotation-tables": Command(cmd_biological_annotation_tables, context_aware=True),
    "peak-analysis": Command(cmd_peak_analysis, context_aware=True),
    "candidate-edge-analysis": Command(cmd_candidate_edge_analysis, context_aware=True),
    "candidate-tree-contrast": Command(cmd_candidate_tree_contrast, context_aware=True),
    "sequence-growth-rates": Command(cmd_sequence_growth_rates, context_aware=True),
    "functional-classification": Command(cmd_functional_classification, context_aware=True),
    "bootstrap-smoothed-tree": Command(cmd_bootstrap_smoothed_tree, context_aware=True),
    "bootstrap-band-quantile-scan": Command(cmd_bootstrap_band_quantile_scan, context_aware=True),
    "functional-threshold-scan": Command(cmd_functional_threshold_scan, context_aware=False),
    "classification-probability-scans": Command(cmd_classification_probability_scans, context_aware=False),
    "branch-decomposition": Command(cmd_branch_decomposition, context_aware=False),
    "peak-pair-distances": Command(cmd_peak_pair_distances, context_aware=True),
    "leading-saddle-band": Command(cmd_leading_saddle_band, context_aware=True),
    "codon-entropy-analysis": Command(cmd_codon_entropy_analysis, context_aware=True),
    "tables": Command(cmd_tables, context_aware=True),
    "figures": Command(cmd_figures, context_aware=False),
    "all": Command(cmd_all, context_aware=False),
    "watch": Command(cmd_watch, context_aware=False),
    "status": Command(cmd_status, context_aware=True),
}
# Commands that honour --output-root/--variant overrides; every other command
# still writes under the import-time output paths.
CONTEXT_AWARE_COMMANDS = frozenset(
    name for name, command in COMMANDS.items() if command.context_aware
)
