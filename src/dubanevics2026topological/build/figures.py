"""Figure input loading, figure rendering and the ``figures`` and ``all`` commands."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..analysis import (
    OUTPUT,
    ROOT,
    _write_json_atomic,
    dataset_input_path,
    load_cached_analysis,
    load_settings,
)
from ..analysis.component_persistence import load_cached_component_persistence
from ..analysis.manuscript_quantities import is_stage_not_run
from ..figure_exclusions import (
    MISSING_PRODUCT_TABLE,
    TABLE,
    bootstrap_identity_exclusions,
)
from ..figures._quantities import QuantityError, QuantityGapError
from ..figures._title_flag import without_headings
from ..figures.labels import PAPKOU_THRESHOLD, TOPOGRAPHER_THRESHOLD, graph_label
from ..figures.peak_rank import PEAK_RANK_IDS
from ..figures.registry import (
    FIGURE_SPECS,
    LEAF_PERSISTENCE_BAND_IDS,
    SAVED_FIGURE_IDS,
    SAVED_PRODUCT_INPUTS,
    SHARED_RENDERER_FILES,
    FigureSpec,
    renderer_files,
    variant_error,
)
from ..figures.story_views import STORY_VIEW_FIGURE_IDS
from ..landscape.full_population import load_full_network_fitness
from ..landscape.selection import selection_source
from ..output_paths import figure_path
from ..provenance import without_git
from .analysis_stages import (
    _analysis_fingerprint,
    _band_scan_digests,
    _bootstrap_digests,
    _comparison_analysis_fingerprint,
    _full_fitness_dependency,
    _leading_saddle_band_identity,
    _peak_analysis_fingerprint,
    _pooled_scatter_dependency,
    ensure_analysis_checkpoint,
    load_cached_leading_saddle_band,
    require_current_registry,
)
from .bootstrap import active_bootstrap_band_identity
from .common import (
    ANNOTATION_TABLE_DIRECTORIES,
    BAND_LEAF_FIGURE_IDS,
    BAND_QUANTILE_SCAN_FIGURE_ID,
    BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID,
    BOOTSTRAP_BIOLOGY_FIGURE_ID,
    BOOTSTRAP_CODON_ENTROPY_FIGURE_ID,
    BOOTSTRAP_FIGURE_IDS,
    BOOTSTRAP_STABILITY_FIGURE_ID,
    BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID,
    BUILD_MODULES,
    CANDIDATE_FIGURE_IDS,
    EDGE_FIGURE_ID,
    ENV_FONT_DIR,
    FIGURE_DIR,
    FOUR_GRAPH_FIGURE_IDS,
    LABEL_COLLISION_FIGURE_IDS,
    LANDMARKS_FIGURE_ID,
    LEAF_FIGURE_IDS,
    MANIFOLD_FIGURE_ID,
    MISSING_PRODUCT_FIGURE_IDS,
    NOT_SHARED_FIGURE_KEYS,
    PEAK_FIGURE_ID,
    RECORD_DIR,
    ROTATED_LABEL_COLLISION,
    ROTATED_STEM_SUFFIX,
    WITH_BOOTSTRAP_FIGURE_IDS,
    _context_kwargs,
    _digest_bytes,
    _digest_file,
    _display_path,
    _load_settings,
    _peak_pair_context,
)
from .products import FDR_POOLED_VARIANT, _fdr_peak_settings, _fdr_pooled_output_root
from .tables import (
    build_biological_tables,
    build_codon_entropy_tables,
    build_comparison_tables,
    build_leading_saddle_band_tables,
    build_manuscript_quantities,
    build_peak_tables,
    build_sequence_growth_rate_tables,
    build_tables,
)


def _figure_dependencies(
    figure_id: str, variant: str, settings: dict[str, Any]
) -> dict[str, Any]:
    spec = next((item for item in FIGURE_SPECS if item.figure_id == figure_id), None)
    if spec is None:
        raise KeyError(f"No figure spec registered for {figure_id!r}.")
    figure_settings = settings["figures"].get("figures", {}).get(spec.config_table)
    if figure_settings is None:
        raise KeyError(f"No figure settings configured for {figure_id!r}.")
    figure_dir = ROOT / "src/dubanevics2026topological/figures"
    own = renderer_files(spec)
    sources = [figure_dir / name for name in own]
    sources.extend(
        figure_dir / name
        for name in SHARED_RENDERER_FILES
        if name not in own and (figure_dir / name).is_file()
    )
    files = {}
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Figure renderer source is missing: {path.name}.")
        files[str(path)] = _digest_file(path)
    if figure_id in {
        "candidate_tree_histogram",
        BOOTSTRAP_TREE_HISTOGRAM_FIGURE_ID,
        "full_tree_histogram",
        "published_tree_histogram",
        "tree_comparison",
        "reflected_fdr_trees",
        "full_tree_persistence",
        "full_tree_persistence_vertical_histogram",
    }:
        from topographer.plotting import (
            compute_fitness_histogram_table,
            plot_fitness_histogram,
        )

        histogram_source = inspect.getsourcefile(compute_fitness_histogram_table)
        plot_source = inspect.getsourcefile(plot_fitness_histogram)
        for source in {histogram_source, plot_source}:
            if source is None or not Path(source).is_file():
                raise FileNotFoundError(
                    "Topographer fitness histogram source is unavailable."
                )
            source_path = Path(source).resolve()
            files[str(source_path)] = _digest_file(source_path)
    if spec.notebook:
        notebook_path = ROOT / spec.notebook
        if notebook_path.is_file():
            files[str(notebook_path)] = _digest_file(notebook_path)
    if "annotations" in spec.inputs:
        # The figure draws the saved annotation tables: changed tables re-render it.
        for location in ANNOTATION_TABLE_DIRECTORIES.values():
            directory = OUTPUT / "tables" / location
            for path in sorted(directory.glob("biological_annotation_*.*")):
                if path.suffix == ".csv":
                    continue
                files[str(path)] = _digest_file(path)
    if "manuscript_quantities" in spec.inputs:
        # Labels come from the registry: a changed registry re-renders the figure.
        from ..figures._quantities import registry_fingerprint

        registry = {"manuscript_quantities": registry_fingerprint(OUTPUT)}
    else:
        registry = {}
    return {
        **registry,
        "variant": variant,
        "figure_files": files,
        "figure_config": figure_settings,
        "shared_config": {
            key: value
            for key, value in settings["figures"].items()
            if key not in NOT_SHARED_FIGURE_KEYS
        },
        "render_source": _digest_bytes(
            b"".join(
                (Path(__file__).resolve().parent / f"{name}.py").read_bytes()
                for name in BUILD_MODULES
            )
        ),
    }


def extract_bootstrap_draws(bootstrap: Any) -> int | None:
    """Extract the integer bootstrap draw count from a restored checkpoint or provenance."""
    if bootstrap is None:
        return None
    if hasattr(bootstrap, "provenance") and isinstance(bootstrap.provenance, Mapping):
        prov = bootstrap.provenance
        draws = (
            prov.get("declared", {})
            .get("settings", {})
            .get("scientific", {})
            .get("config", {})
            .get("n_replicates")
        )
        if draws is not None:
            return int(draws)
    if hasattr(bootstrap, "summary") and isinstance(bootstrap.summary, Mapping):
        draws = bootstrap.summary.get("settings", {}).get(
            "n_replicates"
        ) or bootstrap.summary.get("n_replicates")
        if draws is not None:
            return int(draws)
    if isinstance(bootstrap, Mapping):
        draws = (
            bootstrap.get("declared", {})
            .get("settings", {})
            .get("scientific", {})
            .get("config", {})
            .get("n_replicates")
            or bootstrap.get("settings", {}).get("n_replicates")
            or bootstrap.get("n_replicates")
        )
        if draws is not None:
            return int(draws)
    return None


def split_footer_lines(text: str) -> list[str]:
    """Split a long footer string at ' · ' into two balanced lines."""
    parts = text.split(" · ")
    if len(parts) >= 3:
        line1 = " · ".join(parts[:2])
        line2 = " · ".join(parts[2:])
        return [line1, line2]
    return [text]


def variant_figure_footer(
    settings: Mapping[str, Any],
    *,
    figure_id: str | None = None,
    has_bootstrap: bool = False,
    bootstrap_draws: int | None = None,
    inputs: Mapping[str, Any] | None = None,
    as_lines: bool = False,
) -> str | list[str] | None:
    """Return the variant footer line if an ``fdr_fixed_cutoff`` variant is active, else ``None``.

    Constructs a short, consistent footer line from the variant declaration and authenticated
    provenance, without hardcoded values:
    - Prefix: the graph label, "Topographer + FDR graph: " or "Papkou + FDR graph: "
    - Cutoff text: e.g. "Topographer + FDR graph: Topographer threshold (T = −0.253159)" or "Papkou + FDR graph: Papkou threshold (T = −0.5077739708744209)"
    - Method and nominal q: e.g. "BY, nominal q = 0.05"
    - Retained graph rule: "≥1 retained Functional endpoint, largest component"
    - Bootstrap draw count: "10,000 bootstrap draws" included only on figures using bootstrap output.

    If ``figure_id`` is supplied and not in ``CANDIDATE_FIGURE_IDS``, returns ``None``.
    If ``as_lines=True``, returns a list of two balanced lines split at " · " for Altair's
    list-of-lines title text.
    """
    if figure_id is not None and figure_id not in CANDIDATE_FIGURE_IDS:
        return None

    analysis = settings.get("analysis", settings)
    if not isinstance(analysis, Mapping):
        return None
    clamped = analysis.get("clamped", {})
    if not isinstance(clamped, Mapping):
        return None
    declaration = clamped.get("candidate_functional_threshold")
    if (
        not isinstance(declaration, Mapping)
        or declaration.get("rule") != "fdr_fixed_cutoff"
    ):
        return None

    from ..threshold.fdr_selection import threshold_summary

    summary = threshold_summary(declaration)
    applied = summary["threshold"]["rounded"]
    applied_str = f"{applied}".replace("-", "−")
    cutoff = summary["cutoff"]
    if cutoff == "median_reflected":
        cutoff_label = TOPOGRAPHER_THRESHOLD
        graph_state = "reflected_by"
    elif cutoff == "papkou_original":
        cutoff_label = PAPKOU_THRESHOLD
        graph_state = "published_by"
    else:
        # Another cutoff keeps the footer's earlier prefix, which always named the
        # Topographer-side graph; only its cutoff text is the declared name.
        cutoff_label = f"{cutoff.replace('_', ' ').capitalize()} threshold"
        graph_state = "reflected_by"

    cutoff_part = f"{graph_label(graph_state)}: {cutoff_label} (T = {applied_str})"
    method = str(summary["method"]).upper()
    q = summary["q"]
    q_part = (
        f"{method}, nominal q = 0.05" if q == 0.05 else f"{method}, nominal q = {q:g}"
    )
    graph_part = "≥1 retained Functional endpoint, largest component"

    parts = [cutoff_part, q_part, graph_part]

    if has_bootstrap:
        draws = bootstrap_draws
        if draws is None and inputs is not None and "bootstrap_smoothed" in inputs:
            draws = extract_bootstrap_draws(inputs["bootstrap_smoothed"])
        if draws is not None:
            parts.append(f"{draws:,} bootstrap draws")

    full_text = " · ".join(parts)
    if as_lines:
        return split_footer_lines(full_text)
    return full_text


def apply_figure_footer(
    chart: Any,
    footer: str | Sequence[str] | None,
    *,
    offset: int = 15,
    font_size: int = 11,
    color: str = "#222222",
) -> Any:
    """Attach ``footer`` to the bottom of ``chart`` without overlapping axes.

    If ``footer`` is ``None``, returns ``chart`` unmodified (no artist or element added).
    Accepts either a single string or a sequence/list of lines (Altair list-of-lines title text).
    Top-level-only properties on ``chart`` (such as ``datasets``, ``usermeta``, ``config``,
    and ``$schema``) are migrated to the top-level ``vconcat`` wrapper.
    """
    if footer is None:
        return chart

    if isinstance(footer, str):
        title_text: list[str] | str = split_footer_lines(footer)
    else:
        title_text = list(footer)

    import altair as alt

    chart_copy = copy.deepcopy(chart)
    top_level_props: dict[str, Any] = {}
    for prop in ("datasets", "usermeta", "config", "$schema"):
        val = getattr(chart_copy, prop, alt.Undefined)
        if val is not alt.Undefined:
            top_level_props[prop] = val
            setattr(chart_copy, prop, alt.Undefined)

    comp = alt.vconcat(chart_copy).properties(
        title=alt.TitleParams(
            text=title_text,
            orient="bottom",
            anchor="start",
            offset=offset,
            fontSize=font_size,
            fontWeight="normal",
            color=color,
        )
    )
    for prop, val in top_level_props.items():
        setattr(comp, prop, val)
    return comp


class FiguresFailedError(RuntimeError):
    """One or more figures failed to build or render; the rest were rendered."""

    def __init__(
        self, figure_ids: list[str], causes: list[BaseException] | None = None
    ) -> None:
        self.figure_ids = list(figure_ids)
        self.causes = list(causes or [])
        details = "".join(
            f"\n  {type(cause).__name__}: {cause}" for cause in self.causes
        )
        super().__init__(
            f"{len(self.figure_ids)} figure(s) failed: "
            + ", ".join(self.figure_ids)
            + details
        )


def _record_figure_failure(
    failures: list[str],
    causes: list[BaseException],
    figure_id: str,
    variant: str,
    exc: BaseException,
    leftovers: Iterable[Path],
) -> None:
    """Report one failed figure on stderr, remove its output files, note its id."""
    print(
        f"FIGURE FAILED: {figure_id} ({variant}): {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(exc, file=sys.stderr)
    sys.stderr.flush()
    for path in leftovers:
        path.unlink(missing_ok=True)
    causes.append(exc)
    if figure_id not in failures:
        failures.append(figure_id)


def _render_one(
    chart: Any,
    target: Path,
    format_name: str,
    *,
    png_dpi: int = 300,
    footer: str | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=f".{format_name}", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temp_name)
    try:
        options = {"ppi": png_dpi} if format_name == "png" else {}
        chart_to_save = (
            apply_figure_footer(chart, footer) if footer is not None else chart
        )
        chart_to_save.save(temporary, format=format_name, **options)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Renderer produced no data for {target.name}.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_inputs(
    names: Iterable[str],
    settings: dict[str, Any] | None = None,
    *,
    context: Any | None = None,
) -> dict[str, Any]:
    """Read each named cached input once via its existing ``load_cached_*`` reader."""
    names = tuple(names)
    inputs: dict[str, Any] = {}
    for name in names:
        if name in inputs:
            continue
        if name == "split_analysis":
            inputs[name] = load_cached_analysis(**_context_kwargs(context))
        elif name == "comparison":
            from ..analysis.clamped_cache import load_cached_clamped_comparison

            inputs[name] = load_cached_clamped_comparison(**_context_kwargs(context))
        elif name == "peak":
            from ..analysis.peak_uncertainty import load_cached_peak_uncertainty

            inputs[name] = load_cached_peak_uncertainty(**_context_kwargs(context))
        elif name == "growth_rates":
            from ..analysis.sequence_growth_rates import (
                load_cached_sequence_growth_rates,
            )

            inputs[name] = load_cached_sequence_growth_rates(**_context_kwargs(context))
        elif name == "published_fitness":
            from ..analysis.sequence_growth_rates import load_fitness_wt

            inputs[name] = load_fitness_wt(
                settings or _load_settings(context), context=context
            )
        elif name == "edge_evidence":
            from ..analysis.candidate_edge_evidence import (
                load_cached_candidate_edge_evidence,
            )

            inputs[name] = load_cached_candidate_edge_evidence(
                **_context_kwargs(context)
            )
        elif name == "component_persistence":
            inputs[name] = load_cached_component_persistence(**_context_kwargs(context))
        elif name == "full_fitness":
            inputs[name] = load_full_network_fitness(
                settings or _load_settings(context), **_context_kwargs(context)
            )
        elif name == "leading_saddle_band":
            inputs[name] = load_cached_leading_saddle_band(
                settings, **_context_kwargs(context)
            )
        elif name == "full_network":
            from ..figures.join_tree import load_full_network

            inputs[name] = load_full_network(
                settings or _load_settings(context), context
            )
        elif name == "bootstrap_smoothed":
            from ..simplification.bootstrap_smoothed_tree import (
                load_cached_bootstrap_smoothed_tree,
            )

            inputs[name] = load_cached_bootstrap_smoothed_tree(
                settings or _load_settings(context), **_context_kwargs(context)
            )
        elif name == "annotations":
            inputs[name] = load_saved_biological_annotations(**_context_kwargs(context))
        elif name == "manuscript_quantities":
            from ..figures._quantities import load_manuscript_quantities

            inputs[name] = load_manuscript_quantities(OUTPUT)
        elif name == "fdr_q_scan":
            from ..results import RunContext
            from .functional_counts_vs_q import load_fdr_q_scan

            inputs[name] = load_fdr_q_scan(context or RunContext())
        elif name == "four_graph_comparisons":
            from ..results import RunContext
            from .four_graph_inputs import load_four_graph_comparisons

            inputs[name] = load_four_graph_comparisons(context or RunContext())
        elif name == "band_quantile_scan_grid":
            from ..analysis.band_quantile_scan_panels import load_grid
            from ..results import RunContext

            ctx = context or RunContext()
            inputs[name] = load_grid(Path(ctx.processed_root))
        elif name == "peak_pair_distances":
            from ..analysis.peak_pair_distances import (
                load_cached_peak_pair_distances,
            )
            from ..results import RunContext

            # The stage always runs on its own variant (see the CLI command).
            ctx = _peak_pair_context(context or RunContext())
            inputs[name] = load_cached_peak_pair_distances(ctx)
        else:
            raise ValueError(f"Unknown figure input {name!r}.")
    return inputs


def load_saved_biological_annotations(*, context: Any | None = None) -> dict[str, Any]:
    """The saved annotation tables of each landscape, by landscape key.

    Written by ``biological-annotation-tables``; a missing export is an error, never
    a reason to rebuild the annotations here.
    """
    from ..analysis.describe import load_saved_annotations

    directory = (OUTPUT if context is None else Path(context.output_root)) / "tables"
    annotations = {}
    for key, location in ANNOTATION_TABLE_DIRECTORIES.items():
        path = directory / location
        if not (path / "biological_annotation_pairs.parquet").is_file():
            raise FileNotFoundError(
                f"The {key} biological annotation tables are missing ({path}): "
                "run `just biological-annotation-tables`."
            )
        annotations[key] = load_saved_annotations(path)
    return annotations


def load_figure_inputs(
    specs: Iterable[FigureSpec], settings: dict[str, Any], *, context: Any | None = None,
) -> dict[str, Any]:
    """Load declared inputs and resolve selection only for figures that need it."""
    specs = tuple(specs)
    needed = sorted(set().union(*(spec.inputs for spec in specs)))
    # pooled_scatter (and other saved products) handled specially; exclude from load_inputs.
    needed = [n for n in needed if n not in SAVED_PRODUCT_INPUTS]
    inputs = load_inputs(needed, settings, **_context_kwargs(context))
    routed = [
        spec
        for spec in specs
        if spec.figure_id in SAVED_FIGURE_IDS and "comparison" in spec.inputs
    ]
    if routed:
        from ..results import RunContext
        from .saved_figure_inputs import load_saved_figure_inputs

        ctx = context or RunContext()
        include_band = any("bootstrap_smoothed" in spec.inputs for spec in routed)
        declaration = selection_source(settings)
        band_options = {}
        if include_band and "bootstrap_smoothed" in inputs:
            # The saved band tree must descend from the draws this run holds.
            band_options["active_band_identity"] = active_bootstrap_band_identity(
                inputs["bootstrap_smoothed"]
            )
        inputs["saved_figure_inputs"] = load_saved_figure_inputs(
            ctx,
            declaration,
            include_band=include_band,
            **band_options,
        )
    # Load pooled_scatter product only for specs that declare it in their inputs
    # (declaration-driven, not just by figure_id). This avoids forcing it for
    # stubbed test inputs in figure_quantities tests.
    if any(
        spec.figure_id == PEAK_FIGURE_ID and "pooled_scatter" in spec.inputs
        for spec in specs
    ):
        # Peak loads its pooled_scatter via the saved path, but does not require
        # selection routing (the product is tied to the fixed published growth_rates).
        from ..results import RunContext
        from .saved_figure_inputs import (
            _load_pooled_scatter_payload,
            load_pooled_scatter_product,
        )

        id_tol = (
            settings.get("analysis", {})
            .get("sequence_growth_rates", {})
            .get("identity_tolerance", 1e-6)
        )
        se_tol = (
            settings.get("analysis", {})
            .get("peak_uncertainty_scatter", {})
            .get("standard_error_tolerance", 1e-3)
        )

        # Load under a main (non-variant) context so the nominal "pooled_scatter"
        # for the 2x2 is always the main one. Safer-script runs set
        # FITNESSTREE_ANALYSIS_VARIANT + OUTPUT_ROOT per variant and write the
        # FDR pooled under output/variants/<var>; we must not let that pollute
        # the vanilla side.
        import os

        old_var = os.environ.get("FITNESSTREE_ANALYSIS_VARIANT")
        old_root = os.environ.get("FITNESSTREE_OUTPUT_ROOT")
        for k in ("FITNESSTREE_ANALYSIS_VARIANT", "FITNESSTREE_OUTPUT_ROOT"):
            os.environ.pop(k, None)
        try:
            ctx_main = context or RunContext()
            prod = load_pooled_scatter_product(
                ctx_main,
                expected={
                    "identity_tolerance": float(id_tol),
                    "standard_error_tolerance": float(se_tol),
                },
            )
        finally:
            if old_var is not None:
                os.environ["FITNESSTREE_ANALYSIS_VARIANT"] = old_var
            if old_root is not None:
                os.environ["FITNESSTREE_OUTPUT_ROOT"] = old_root

        ctx = context or RunContext()
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved["pooled_scatter"] = _load_pooled_scatter_payload(ctx, product=prod)
        saved.setdefault("product_ids", {})["pooled_scatter"] = (
            prod.manifest.scientific_id_sha256
        )
        # The pooled write lands under the variant output root. The figures
        # process keeps the default output root, so read that product from disk
        # and inject it. A missing product stays absent; compose raises.
        from ..analysis.peak_uncertainty import _fingerprint
        from ..results import RunContext
        from ..results.lookup import MissingProductError

        fdr_root = _fdr_pooled_output_root()
        fdr_ctx = RunContext(
            output_root=fdr_root,
            processed_root=ctx.processed_root,
            raw_data_dir=ctx.raw_data_dir,
            barseq_export_dir=ctx.barseq_export_dir,
            root=ctx.root,
            variant_name=FDR_POOLED_VARIANT,
            env={},
        )
        fdr_settings = _fdr_peak_settings(fdr_ctx)
        fingerprint = _fingerprint(fdr_settings, context=fdr_ctx)
        tolerances = {
            "identity_tolerance": float(id_tol),
            "standard_error_tolerance": float(se_tol),
        }
        try:
            fdr_prod = load_pooled_scatter_product(
                fdr_ctx, expected=tolerances,
                expected_checkpoint_fingerprint=fingerprint,
            )
        except MissingProductError as exc:
            raise MissingProductError(
                "Missing or stale median-reflected FDR pooled scatter; run `just fdr-pooled-scatter-product`."
            ) from exc
        saved["fdr_pooled_scatter"] = _load_pooled_scatter_payload(fdr_ctx, product=fdr_prod)
        saved.setdefault("product_ids", {})["fdr_pooled_scatter"] = (
            fdr_prod.manifest.scientific_id_sha256
        )
        saved["fdr_peak_checkpoint_fingerprint"] = fingerprint
    if any(
        spec.figure_id == PEAK_FIGURE_ID and "peak_panels" in spec.inputs
        for spec in specs
    ):
        # The four graphs of the peak figure, each from its own variant's saved
        # products under <output root>/variants/<variant>/ (build/peak_panels.py).
        from ..results import RunContext
        from .peak_panels import load_peak_panels

        ctx = context or RunContext()
        saved = inputs.setdefault("saved_figure_inputs", {})
        panels, panel_ids = load_peak_panels(ctx, settings)
        saved["peak_panels"] = panels
        saved.setdefault("product_ids", {})["peak_panels"] = panel_ids
    if any("threshold_diagnostics" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .saved_figure_inputs import load_threshold_diagnostics_product

        ctx = context or RunContext()
        saved = inputs.setdefault("saved_figure_inputs", {})
        diags, diag_id = load_threshold_diagnostics_product(ctx, settings=settings)
        saved["threshold_diagnostics"] = diags
        saved.setdefault("product_ids", {})["threshold_diagnostics"] = diag_id
    if any("bootstrap_codon_entropy" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .saved_figure_inputs import load_bootstrap_codon_entropy_product

        ctx = context or RunContext()
        saved = inputs.setdefault("saved_figure_inputs", {})
        prod = load_bootstrap_codon_entropy_product(ctx)
        saved["bootstrap_codon_entropy"] = prod
        saved.setdefault("product_ids", {})["bootstrap_codon_entropy"] = prod[
            "product"
        ].manifest.scientific_id_sha256
    if any("bootstrap_band_quantile_scan" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .saved_figure_inputs import load_bootstrap_band_quantile_scan_product

        ctx = context or RunContext()
        saved = inputs.setdefault("saved_figure_inputs", {})
        prod = load_bootstrap_band_quantile_scan_product(
            ctx, settings=settings, bootstrap=inputs["bootstrap_smoothed"]
        )
        saved["bootstrap_band_quantile_scan"] = prod
        saved.setdefault("product_ids", {})["bootstrap_band_quantile_scan"] = prod[
            "product"
        ].manifest.scientific_id_sha256
    if any("graph_tree_persistence" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .saved_figure_inputs import load_graph_tree_persistence_product

        ctx = context or RunContext()
        saved = inputs.setdefault("saved_figure_inputs", {})
        prod = load_graph_tree_persistence_product(ctx, settings=settings)
        inputs["graph_tree_persistence"] = prod["construction"]
        saved.setdefault("product_ids", {})["graph_tree_persistence"] = prod[
            "product"
        ].manifest.scientific_id_sha256
    if any("reflected_fdr_trees" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .reflected_fdr_inputs import load_reflected_fdr_inputs

        payload = load_reflected_fdr_inputs(context or RunContext())
        inputs["reflected_fdr_trees"] = payload
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved.setdefault("product_ids", {})["reflected_fdr_trees"] = payload["product_ids"]
    if any("bootstrap_composite_panels" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .bootstrap_composite_nominal import add_nominal_peaks
        from .bootstrap_composite_product import load_all_panel_products

        panels = load_all_panel_products(context or RunContext())
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved.setdefault("product_ids", {})["bootstrap_composite_panels"] = {
            variant: _digest_bytes(json.dumps(panel, sort_keys=True, allow_nan=False).encode())
            for variant, panel in panels.items()
        }
        # After the digest: the count comes from the same verified checkpoints.
        add_nominal_peaks(panels, context or RunContext())
        inputs["bootstrap_composite_panels"] = panels
    if any("band_leaf_tables" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .band_leaf_inputs import load_band_leaf_inputs

        payload = load_band_leaf_inputs(context or RunContext(), settings)
        inputs["band_leaf_tables"] = payload
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved.setdefault("product_ids", {})["band_leaf_tables"] = payload["product_ids"]
    if any("tree_views" in spec.inputs for spec in specs):
        # The story views' saved trees (config/figures.toml [story_views]); read once.
        from ..results import RunContext
        from .story_view_inputs import load_tree_view_inputs

        payload = load_tree_view_inputs(context or RunContext(), settings)
        inputs["tree_views"] = payload
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved.setdefault("product_ids", {})["tree_views"] = payload["product_ids"]
    if any("ridge_entropy" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .story_view_inputs import load_ridge_entropy_inputs

        payload = load_ridge_entropy_inputs(context or RunContext())
        inputs["ridge_entropy"] = payload
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved.setdefault("product_ids", {})["ridge_entropy"] = {
            label: item["product_id"] for label, item in payload.items()
        }
    if any("graph_codon_entropy" in spec.inputs for spec in specs):
        from ..results import RunContext
        from .story_view_inputs import load_graph_codon_entropy_inputs

        payload = load_graph_codon_entropy_inputs(context or RunContext(), settings)
        inputs["graph_codon_entropy"] = payload
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved.setdefault("product_ids", {})["graph_codon_entropy"] = payload[
            "product_ids"
        ]
    if any("multiple_testing_comparison" in spec.inputs for spec in specs):
        from ..figures.multiple_testing_comparison import (
            load_multiple_testing_comparison,
        )

        payload = load_multiple_testing_comparison(settings)
        inputs["multiple_testing_comparison"] = payload
        saved = inputs.setdefault("saved_figure_inputs", {})
        saved.setdefault("product_ids", {})["multiple_testing_comparison"] = payload[
            "product_id"
        ]
    return inputs


def register_font_directory(context: Any | None = None) -> Path | None:
    """Register font directory with the renderer (vl-convert), if set in context or env.

    A machine without Arial or Liberation Sans points ``FITNESSTREE_FONT_DIR`` at a folder
    of fonts. A set but missing directory raises: rendering with a silent fallback font
    would change the figures.
    """
    directory: Path | None = None
    if context is not None:
        if getattr(context, "font_dir", None) is None:
            return None
        directory = Path(context.font_dir).resolve()
    else:
        value = os.environ.get(ENV_FONT_DIR)
        if not value:
            return None
        directory = Path(value).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{ENV_FONT_DIR} is not a directory: {directory}")
    import vl_convert

    vl_convert.register_font_directory(str(directory))
    return directory


def default_selection(
    specs: Mapping[str, Any],
    *,
    with_bootstrap: bool = False,
    excluded: Mapping[str, str] | None = None,
    missing_product: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Ids of the default figures, plus the bootstrap figures when asked.

    ``with_bootstrap`` adds ``WITH_BOOTSTRAP_FIGURE_IDS`` (every bootstrap figure,
    the band-scan ones included); the order stays the registry's. ``excluded`` maps
    a figure id to the reason it is left out of that addition
    (``[bootstrap_identity_exclusions]`` in config/figures.toml); each omission is
    printed with its reason; it applies to a bootstrap figure that is a default figure
    as well as to an addition. A figure named with ``--figure`` is never excluded here.
    """
    excluded = bootstrap_identity_exclusions(
        {} if excluded is None else {TABLE: excluded}, WITH_BOOTSTRAP_FIGURE_IDS
    )
    missing_product = bootstrap_identity_exclusions(
        {} if missing_product is None else {MISSING_PRODUCT_TABLE: missing_product},
        MISSING_PRODUCT_FIGURE_IDS,
        MISSING_PRODUCT_TABLE,
    )
    chosen = []
    for figure_id, spec in specs.items():
        if figure_id in missing_product:
            print(
                f"Skipping {figure_id} from the default figures: "
                f"{missing_product[figure_id]}",
                file=sys.stderr,
            )
        elif figure_id in excluded and (spec.default or with_bootstrap):
            # A guarded bootstrap figure is left out whether it is a default figure or
            # an addition of --with-bootstrap.
            origin = "the default figures" if spec.default else "--with-bootstrap"
            print(
                f"Skipping {figure_id} from {origin}: {excluded[figure_id]}",
                file=sys.stderr,
            )
        elif spec.default:
            chosen.append(figure_id)
        elif with_bootstrap and figure_id in WITH_BOOTSTRAP_FIGURE_IDS:
            chosen.append(figure_id)
    return tuple(chosen)


def opt_in_selection(specs: Mapping[str, Any]) -> tuple[str, ...]:
    """Ids of every registered figure outside the default selection (registry order).

    This is the list ``build figures --opt-in`` renders: the figures no manuscript entry
    needs (``just figures-exploratory``)."""
    return tuple(figure_id for figure_id, spec in specs.items() if not spec.default)


def build_figures_keep_going(
    selected: Iterable[str] | None = None,
    *,
    force: bool = False,
    variants: Iterable[str] | None = None,
    formats: Iterable[str] | None = None,
    label_collision: str | None = None,
    with_bootstrap: bool = False,
    titled_twins: bool = True,
    context: Any,
) -> bool:
    """Attempt each figure variant independently and write a failure report."""
    if with_bootstrap and (selected is not None or label_collision is not None):
        raise ValueError(
            "with_bootstrap cannot be combined with a figure selection or label_collision."
        )
    specs = {spec.figure_id: spec for spec in FIGURE_SPECS}
    settings = load_settings()
    # Validated even for an explicit selection, which is never filtered.
    exclusions = bootstrap_identity_exclusions(
        settings["figures"], WITH_BOOTSTRAP_FIGURE_IDS
    )
    missing_product = bootstrap_identity_exclusions(
        settings["figures"],
        MISSING_PRODUCT_FIGURE_IDS,
        MISSING_PRODUCT_TABLE,
    )
    if label_collision is not None:
        if label_collision != ROTATED_LABEL_COLLISION:
            raise ValueError(f"Unknown label_collision: {label_collision!r}.")
        if selected is None:
            selected = [name for name in specs if name in LABEL_COLLISION_FIGURE_IDS]
        if set(selected) - LABEL_COLLISION_FIGURE_IDS:
            raise ValueError("label_collision applies only to the band-tree figures.")
    selection = tuple(
        selected
        or default_selection(
            specs,
            with_bootstrap=with_bootstrap,
            excluded=exclusions,
            missing_product=missing_product,
        )
    )
    unknown = set(selection) - set(specs)
    if unknown:
        raise ValueError(f"Unknown figure id(s): {', '.join(sorted(unknown))}.")
    configured = tuple(settings["figures"].get("variants", ["annotated"]))
    requested = tuple(variants or configured)
    invalid = set(requested) - {"vanilla", "annotated"}
    if invalid:
        raise ValueError("Unknown figure variant(s): " + ", ".join(sorted(invalid)))
    if set(requested) - set(configured):
        raise ValueError("Requested figure variant is not enabled in config/figures.toml.")
    planned = [specs[name] for name in selection if set(requested) & specs[name].variants]
    shared_settings = settings
    if label_collision is not None:
        shared_settings = copy.deepcopy(settings)
        shared_settings["figures"].setdefault("rendering", {})[
            "branch_count_label_collision"
        ] = label_collision
    failures = []
    try:
        shared_inputs = load_figure_inputs(planned, shared_settings, context=context)
    except Exception as exc:  # isolate a shared input failure by trying each figure below
        failures.append(
            {
                "figure_id": None,
                "variant": None,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "scope": "shared_inputs",
            }
        )
        shared_inputs = None
    report_formats = tuple(
        dict.fromkeys(formats or settings["figures"].get("formats", ["png"]))
    )
    for figure_id in selection:
        spec = specs[figure_id]
        applicable = tuple(v for v in requested if v in spec.variants)
        if not applicable and selected is not None and variants is not None:
            raise variant_error(spec)
        for variant in applicable:
            try:
                build_figures(
                    (figure_id,), force=force, variants=(variant,), formats=formats,
                    label_collision=label_collision, context=context,
                    titled_twins=titled_twins,
                    _inputs_override=shared_inputs, _raise_skipped=True,
                    _incomplete_provenance=shared_inputs is None,
                )
            except Exception as exc:  # noqa: BLE001 - report each independent figure failure
                if isinstance(exc, FiguresFailedError) and len(exc.causes) == 1:
                    exc = exc.causes[0]  # report the figure's own error
                stem = (
                    figure_id
                    if len(spec.variants) == 1 or variant == "annotated"
                    else f"{figure_id}_vanilla"
                ) + (ROTATED_STEM_SUFFIX if label_collision is not None else "")
                for format_name in report_formats:
                    (RECORD_DIR / f"{stem}.{format_name}.json").unlink(missing_ok=True)
                    (RECORD_DIR / "titled" / f"{stem}.{format_name}.json").unlink(
                        missing_ok=True
                    )
                failures.append(
                    {
                        "figure_id": figure_id,
                        "variant": variant,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(exc)
                        ),
                    }
                )
    report_path = context.output_root / "figures" / "figure-failures.json"
    _write_json_atomic(report_path, {"failures": failures})
    print(f"Figure failure report: {_display_path(report_path)}", flush=True)
    for failure in failures:
        if failure.get("scope") == "shared_inputs":
            print(
                f"FAILED shared figure inputs: {failure['exception_type']}: "
                f"{failure['message']} (fallback provenance is incomplete)", flush=True,
            )
            continue
        print(
            f"FAILED {failure['figure_id']} ({failure['variant']}): "
            f"{failure['exception_type']}: {failure['message']}", flush=True,
        )
    return not failures


def _render_outputs(
    produced: Iterable[Any],
    *,
    formats: tuple[str, ...],
    titled_twins: bool,
    png_dpi: int,
    dependency: dict[str, Any],
    rendered_now: set[Path],
    outputs: list[Path],
) -> list[tuple[str, BaseException]]:
    """Write every ``FigureOutput`` of a multi-output figure, each with its render record.

    An output is its untitled file in ``canonical/``, its titled twin in
    ``canonical/titled/`` (``titled_twins``), its JSON sidecar ``canonical/<stem>.json`` when it
    has one, and a render record per file in ``.records/`` (twins in ``.records/titled/``),
    named by the output's shelf stem, so ``figure_return`` fetches it. Outputs fail on their
    own: the others are still written, and the failures are returned as ``(stem, error)``
    after the files of each failed output and their records are removed. An error of the
    iteration itself (before any output is named) is raised.
    """
    failed: list[tuple[str, BaseException]] = []
    for output in produced:
        try:
            chart = output.build()
            stripped = without_headings(chart)
            if stripped is None:
                print(
                    f"Headings: {output.stem} could not be stripped; canonical keeps "
                    "today's chart and no titled twin is written"
                )
            plain = chart if stripped is None else stripped[0]
            for format_name in formats:
                for twin in ((False, True) if titled_twins else (False,)):
                    target = figure_path(OUTPUT, output.stem, ext=format_name, titled=twin)
                    record_path = (
                        RECORD_DIR / ("titled" if twin else "") / f"{output.stem}.{format_name}.json"
                    )
                    if twin and stripped is None:
                        target.unlink(missing_ok=True)
                        record_path.unlink(missing_ok=True)
                        continue
                    # Remove the success record before rendering: a failure leaves none.
                    record_path.unlink(missing_ok=True)
                    target.unlink(missing_ok=True)
                    _render_one(chart if twin else plain, target, format_name, png_dpi=png_dpi)
                    rendered_now.add(target)
                    _write_json_atomic(record_path, {**dependency, "output": output.stem})
                    if not twin:
                        outputs.append(target)
                    print(f"Rendered: {_display_path(target)}")
            if output.sidecar is not None:
                sidecar = FIGURE_DIR / "canonical" / f"{output.stem}.json"
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(
                    json.dumps(output.sidecar, indent=2) + "\n", encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001 - each output fails on its own
            print(
                f"FIGURE OUTPUT FAILED: {output.stem}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(exc, file=sys.stderr)
            for format_name in formats:
                for twin in (False, True):
                    figure_path(OUTPUT, output.stem, ext=format_name, titled=twin).unlink(
                        missing_ok=True
                    )
                    (
                        RECORD_DIR / ("titled" if twin else "") / f"{output.stem}.{format_name}.json"
                    ).unlink(missing_ok=True)
            failed.append((output.stem, exc))
    return failed


def build_figures(
    selected: Iterable[str] | None = None,
    *,
    force: bool = False,
    variants: Iterable[str] | None = None,
    formats: Iterable[str] | None = None,
    label_collision: str | None = None,
    with_bootstrap: bool = False,
    context: Any | None = None,
    titled_twins: bool = True,
    _inputs_override: dict[str, Any] | None = None,
    _raise_skipped: bool = False,
    _incomplete_provenance: bool = False,
) -> list[Path]:
    """Render selected figure variants, reusing current per-format records.

    ``formats`` overrides the configured list. Records are per format, so an
    SVG-only build never touches existing PNG outputs or their records.

    ``label_collision="rotate"`` is the opt-in rotated count-label variant of the
    band-tree figures (``LABEL_COLLISION_FIGURE_IDS``): it overrides
    ``branch_count_label_collision`` for this build only, without editing the
    config, and writes ``<figure_id>_rotated.<format>`` with its own record, so the
    canonical output is never touched. Without a selection it renders just those
    figures; any other figure in the selection is an error.

    Every figure is drawn twice (``titled_twins``, the default): the untitled variant,
    without its figure-level heading only, to ``canonical/`` and the titled twin (today's
    chart) to ``canonical/titled/`` (``figures/_title_flag.without_headings`` is the one
    place that drops the heading; panel and row titles stay). ``titled_twins=False``
    writes the untitled variant alone; no route writes a titled chart into
    ``canonical/``. A chart it cannot handle keeps today's chart in ``canonical/``, gets
    no twin and any stale twin and twin record are removed; nothing raises. Twin
    records live in ``.records/titled/``. The returned list holds the untitled
    (``canonical/``) paths only; each twin sits beside them under ``canonical/titled/``.

    ``with_bootstrap`` adds the bootstrap figures (the band-quantile-scan ones
    included) to the default selection; it cannot be combined with a selection or with
    ``label_collision``. Fonts named by ``FITNESSTREE_FONT_DIR`` are registered
    before anything is rendered.
    """
    from ..threshold.functional_threshold import (
        configured_declaration,
        declaration_identity,
    )

    if with_bootstrap and (selected is not None or label_collision is not None):
        raise ValueError(
            "with_bootstrap extends the default selection; it cannot be combined "
            "with a figure selection or label_collision."
        )
    settings = load_settings()
    # Validated even for an explicit selection, which is never filtered.
    exclusions = bootstrap_identity_exclusions(
        settings["figures"], WITH_BOOTSTRAP_FIGURE_IDS
    )
    missing_product = bootstrap_identity_exclusions(
        settings["figures"],
        MISSING_PRODUCT_FIGURE_IDS,
        MISSING_PRODUCT_TABLE,
    )
    specs = {spec.figure_id: spec for spec in FIGURE_SPECS}
    stem_suffix = ""
    if label_collision is not None:
        if label_collision != ROTATED_LABEL_COLLISION:
            raise ValueError(
                f"label_collision must be {ROTATED_LABEL_COLLISION!r} or None, "
                f"got {label_collision!r}."
            )
        if selected is None:
            selected = [sid for sid in specs if sid in LABEL_COLLISION_FIGURE_IDS]
        others = sorted(set(selected) - LABEL_COLLISION_FIGURE_IDS)
        if others:
            raise ValueError(
                "label_collision applies only to the band-tree figures "
                f"({', '.join(sorted(LABEL_COLLISION_FIGURE_IDS))}); got "
                f"{', '.join(others)}."
            )
        settings = copy.deepcopy(settings)
        settings["figures"].setdefault("rendering", {})[
            "branch_count_label_collision"
        ] = label_collision
        stem_suffix = ROTATED_STEM_SUFFIX
    selection = tuple(
        selected
        or default_selection(
            specs,
            with_bootstrap=with_bootstrap,
            excluded=exclusions,
            missing_product=missing_product,
        )
    )
    unknown = set(selection) - set(specs)
    if unknown:
        raise ValueError(f"Unknown figure id(s): {', '.join(sorted(unknown))}.")
    configured_variants = tuple(settings["figures"].get("variants", ["annotated"]))
    requested_variants = tuple(variants or configured_variants)
    invalid_variants = set(requested_variants) - {"vanilla", "annotated"}
    if invalid_variants:
        raise ValueError(
            "Unknown figure variant(s): " + ", ".join(sorted(invalid_variants))
        )
    if set(requested_variants) - set(configured_variants):
        raise ValueError(
            "Requested figure variant is not enabled in config/figures.toml."
        )
    formats = tuple(
        dict.fromkeys(formats or settings["figures"].get("formats", ["png"]))
    )
    if not formats or any(fmt not in {"svg", "png"} for fmt in formats):
        raise ValueError("Supported figure formats are 'svg' and 'png'.")
    png_dpi = int(settings["figures"].get("png_dpi", 300))
    if png_dpi <= 0:
        raise ValueError("PNG DPI must be positive.")

    # Resolve variants: render the intersection of requested and allowed.
    plan: list[tuple[FigureSpec, tuple[str, ...]]] = []
    for figure_id in selection:
        spec = specs[figure_id]
        figure_variants = tuple(v for v in requested_variants if v in spec.variants)
        if not figure_variants and selected is not None and variants is not None:
            raise variant_error(spec)
        if figure_variants:
            plan.append((spec, figure_variants))

    inputs = (
        _inputs_override
        if _inputs_override is not None
        else load_figure_inputs((spec for spec, _ in plan), settings, context=context)
    )
    if any("manuscript_quantities" in spec.inputs for spec, _ in plan):
        # The one registry-currency check (Table 1 runs it too): file hashes, the
        # config and a missing fingerprint, which the per-key check cannot see.
        require_current_registry(settings)
    register_font_directory()
    outputs: list[Path] = []
    failures: list[str] = []
    causes: list[BaseException] = []
    for spec, figure_variants in plan:
        figure_id = spec.figure_id
        print(f"Rendering: {figure_id}", flush=True)
        for variant in figure_variants:
            rendered_now: set[Path] = set()  # written by this attempt, still valid
            try:
                dependency = _figure_dependencies(figure_id, variant, settings)
                if _incomplete_provenance:
                    dependency["keep_going_provenance"] = "incomplete: shared input loading failed"
                all_pids = (inputs.get("saved_figure_inputs") or {}).get("product_ids", {})
                if spec.figure_id in SAVED_FIGURE_IDS and "comparison" in spec.inputs:
                    dependency["saved_products"] = inputs["saved_figure_inputs"][
                        "product_ids"
                    ]
                declared_saved = SAVED_PRODUCT_INPUTS & spec.inputs
                if declared_saved:
                    dependency.setdefault("saved_products", {})
                    dependency["saved_products"] = {
                        **dependency["saved_products"],
                        **{k: all_pids[k] for k in sorted(declared_saved)},
                    }
                footer = variant_figure_footer(
                    settings,
                    figure_id=figure_id,
                    has_bootstrap=figure_id in BOOTSTRAP_FIGURE_IDS,
                    inputs=inputs,
                    as_lines=True,
                )
                if figure_id in LEAF_PERSISTENCE_BAND_IDS:
                    from ..figures.leaf_persistence import band_footer

                    footer = band_footer(figure_id, inputs["band_leaf_tables"])
                if footer is not None:
                    dependency["variant_footer"] = footer
                if figure_id in {
                    "full_tree_persistence",
                    "full_tree_persistence_vertical_histogram",
                }:
                    # The tree panel comes from the split-tree analysis checkpoint.
                    dependency["analysis"] = _analysis_fingerprint()
                    dependency["component_pairs"] = _digest_file(
                        OUTPUT / "analysis" / "component_pairs.parquet"
                    )
                    dependency["component_diagnostics"] = _digest_file(
                        OUTPUT / "analysis" / "component_diagnostics.json"
                    )
                    dependency["full_fitness"] = _full_fitness_dependency(settings)
                    # The saddles come from ``tree_saddles``, which nothing else hashes.
                    dependency["tree_saddles_source"] = _digest_file(
                        ROOT
                        / "src/dubanevics2026topological/analysis/leading_saddle_band.py"
                    )
                    # The noise band computes the Candidate functional threshold from
                    # the declaration, so the rule, the digits and the code re-render.
                    dependency["candidate_threshold"] = {
                        "declaration": declaration_identity(configured_declaration(settings)),
                        "source": _digest_file(
                            ROOT
                            / "src/dubanevics2026topological/threshold/functional_threshold.py"
                        ),
                        "selection_source": selection_source(settings),
                        "selection_code": _digest_file(
                            ROOT / "src/dubanevics2026topological/landscape/selection.py"
                        ),
                    }
                elif figure_id == "full_join_tree":
                    # The band is the stage's interval; the tree is swept on the Full network.
                    dependency["leading_saddle_band"] = _leading_saddle_band_identity(
                        **_context_kwargs(context)
                    )
                    dataset = settings["analysis"]["dataset"]
                    dependency["full_network"] = {
                        name: _digest_file(
                            dataset_input_path(dataset[name], context, root=ROOT)
                        )
                        for name in ("fitness_wt", "fitness_diff")
                    }
                elif figure_id == LANDMARKS_FIGURE_ID:
                    # Reads the stage's tables, keyed on their checkpoint, and bins the
                    # Full-graph population itself.
                    dependency["leading_saddle_band"] = _leading_saddle_band_identity(
                        **_context_kwargs(context)
                    )
                    dependency["full_fitness"] = _full_fitness_dependency(settings)
                elif figure_id in PEAK_RANK_IDS:
                    # The Reference peak set and the Full population; the table code ranks them.
                    dependency["full_fitness"] = _full_fitness_dependency(settings)
                    dependency["peaks_value_sha256"] = settings["analysis"]["clamped"][
                        "peaks_value_sha256"
                    ]
                    dependency["peak_rank_source"] = _digest_file(
                        ROOT / "src/dubanevics2026topological/analysis/peak_rank.py"
                    )
                elif figure_id == EDGE_FIGURE_ID:
                    dependency["edge_analysis"] = without_git(
                        inputs["edge_evidence"]["provenance"]
                    )
                elif figure_id == PEAK_FIGURE_ID and "peak_panels" in spec.inputs:
                    # Each panel's peak checkpoint, pooled_scatter and summary are in the
                    # identity of its peak_panel_summary (saved_products, from above).
                    pass
                elif figure_id == PEAK_FIGURE_ID:
                    dependency["peak_analysis"] = _peak_analysis_fingerprint()
                    if "pooled_scatter" in spec.inputs:  # only when declared
                        if "saved_figure_inputs" in inputs and "product_ids" in inputs.get(
                            "saved_figure_inputs", {}
                        ):
                            dependency["saved_products"] = inputs[
                                "saved_figure_inputs"
                            ].get("product_ids", {})
                            fingerprint = inputs["saved_figure_inputs"].get(
                                "fdr_peak_checkpoint_fingerprint"
                            )
                            if fingerprint is not None:
                                dependency["fdr_peak_checkpoint_fingerprint"] = fingerprint
                        else:
                            dependency["pooled_scatter"] = _pooled_scatter_dependency(
                                inputs, settings
                            )
                elif figure_id == MANIFOLD_FIGURE_ID:
                    dependency["graph_tree_persistence"] = settings["analysis"][
                        "graph_tree_persistence"
                    ]
                    dependency["manifold_source"] = _digest_file(
                        ROOT
                        / "src/dubanevics2026topological/analysis/manifold_graph_tree.py"
                    )
                elif figure_id in {
                    "papkou_agreement",
                    "fitness_se_correlation",
                    "fitness_vs_pooled_se",
                }:
                    dependency["growth_rates"] = without_git(
                        inputs["growth_rates"].provenance
                    )
                    dataset = settings["analysis"]["dataset"]
                    dependency["fitness_wt"] = _digest_file(
                        dataset_input_path(dataset["fitness_wt"], context, root=ROOT)
                    )
                elif figure_id in {
                    "peak_pair_matrices",
                    "peak_pair_hamming_fitness_scatter",
                    "peak_pair_direct_via_saddle_scatter",
                }:
                    # Reads only the stage tables; their identity is the stage record.
                    dependency["peak_pair_distances"] = without_git(
                        inputs["peak_pair_distances"].provenance
                    )
                elif figure_id in {
                    "functional_counts_vs_q",
                    "functional_counts_vs_q_bh",
                    "functional_counts_vs_q_bonferroni",
                }:
                    # The combined FDR scan written by classification-probability-scans.
                    source = Path(inputs["fdr_q_scan"]["source"])
                    dependency["fdr_q_scan"] = _digest_file(source)
                    dependency["fdr_q_scan_loader"] = _digest_file(
                        ROOT
                        / "src/dubanevics2026topological/build/functional_counts_vs_q.py"
                    )
                elif figure_id in BAND_LEAF_FIGURE_IDS:
                    # Reads only the saved D10 band trees (their ids are the saved
                    # products above); the loaders and the leaf-table code are keyed too.
                    build_dir = ROOT / "src/dubanevics2026topological/build"
                    for name in ("band_leaf_inputs.py", "d10_tree_inputs.py", "story_view_inputs.py"):
                        dependency[f"{name}_source"] = _digest_file(build_dir / name)
                elif figure_id in STORY_VIEW_FIGURE_IDS:
                    # Reads only saved products (their ids are the saved products above); the
                    # loaders and the naming grammar are keyed too.
                    build_dir = ROOT / "src/dubanevics2026topological/build"
                    for name in ("story_view_inputs.py", "d10_tree_inputs.py", "d10_band_inputs.py"):
                        dependency[f"{name}_source"] = _digest_file(build_dir / name)
                elif figure_id == "reflected_fdr_trees":
                    build_dir = ROOT / "src/dubanevics2026topological/build"
                    for name in ("reflected_fdr_inputs.py", "saved_figure_inputs.py", "landscape_summary_inputs.py", "tree_layout_products.py"):
                        dependency[f"{name}_source"] = _digest_file(build_dir / name)
                elif figure_id == "bootstrap_four_graphs":
                    # Four verified panel products are already included above.
                    pass
                elif figure_id in FOUR_GRAPH_FIGURE_IDS:
                    # Each variant's comparison checkpoint (read through its own provenance
                    # record) and the loader that turns it into leaf and branch tables.
                    dependency["four_graph_comparisons"] = inputs["four_graph_comparisons"][
                        "fingerprints"
                    ]
                    dependency["four_graph_loader_source"] = _digest_file(
                        ROOT / "src/dubanevics2026topological/build/four_graph_inputs.py"
                    )
                elif figure_id == "multiple_testing_comparison":
                    # The BY snapshot tables are the saved product above. BH and
                    # Bonferroni are computed at render from those p-values.
                    pass
                elif figure_id in {
                    BOOTSTRAP_STABILITY_FIGURE_ID,
                    BAND_QUANTILE_SCAN_FIGURE_ID,
                    BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID,
                }:
                    # Reads only the bootstrap checkpoint, digested with the other
                    # bootstrap figures below; no analysis checkpoint is involved.
                    pass
                else:
                    dependency["analysis"] = (
                        _comparison_analysis_fingerprint()
                        if "comparison" in spec.inputs
                        else _analysis_fingerprint()
                    )
                if figure_id in BOOTSTRAP_FIGURE_IDS:
                    dependency.update(
                        _bootstrap_digests(
                            inputs["bootstrap_smoothed"],
                            settings,
                            **_context_kwargs(context),
                        )
                    )
                if figure_id == BOOTSTRAP_CODON_ENTROPY_FIGURE_ID:
                    # Populations and sweeps are computed in memory at render time.
                    dependency["codon_entropy"] = settings["analysis"]["codon_entropy"]
                    analysis_dir = ROOT / "src/dubanevics2026topological/analysis"
                    simplification_dir = (
                        ROOT / "src/dubanevics2026topological/simplification"
                    )
                    for source in (
                        analysis_dir / "codon_entropy.py",
                        analysis_dir / "codon_entropy_populations.py",
                        simplification_dir / "__init__.py",
                        simplification_dir / "selected_leaf_simplification.py",
                    ):
                        dependency[f"{source.name}_source"] = _digest_file(source)
                if figure_id in {
                    BAND_QUANTILE_SCAN_FIGURE_ID,
                    BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID,
                }:
                    dependency.update(
                        _band_scan_digests(
                            settings,
                            inputs["bootstrap_smoothed"],
                            **_context_kwargs(context),
                        )
                    )
                    dependency["saved_products"] = {
                        "bootstrap_band_quantile_scan": inputs["saved_figure_inputs"][
                            "product_ids"
                        ]["bootstrap_band_quantile_scan"]
                    }
                if figure_id in {
                    "study_tree_persistence_biology",
                    BOOTSTRAP_BIOLOGY_FIGURE_ID,
                    *LEAF_FIGURE_IDS,
                }:
                    dependency["biological_source"] = _digest_file(
                        ROOT
                        / "src/dubanevics2026topological/analysis/biological_annotations.py"
                    )
                if figure_id in LEAF_FIGURE_IDS:
                    # The leaf tables assemble hills with ``nominal_hills``.
                    dependency["hill_source"] = _digest_file(
                        ROOT
                        / "src/dubanevics2026topological/simplification/bootstrap_smoothed_tree.py"
                    )
                if spec.multi:
                    dependency["headings"] = (
                        "untitled in canonical/ (root heading only), titled twin in canonical/titled/"
                        if titled_twins
                        else "untitled in canonical/ (root heading only)"
                    )
                    failed_outputs = _render_outputs(
                        spec.render(inputs, settings, variant),
                        formats=formats,
                        titled_twins=titled_twins,
                        png_dpi=png_dpi,
                        dependency=dependency,
                        rendered_now=rendered_now,
                        outputs=outputs,
                    )
                    if failed_outputs:
                        raise RuntimeError(
                            f"{len(failed_outputs)} output(s) of {figure_id} failed: "
                            + "; ".join(
                                f"{stem}: {type(error).__name__}: {error}"
                                for stem, error in failed_outputs
                            )
                        )
                    continue
                chart = None
                plain = None  # the headingless chart; False when it cannot be made
                stem = (
                    figure_id
                    if len(spec.variants) == 1 or variant == "annotated"
                    else f"{figure_id}_vanilla"
                ) + stem_suffix
                kinds = ("untitled", "titled") if titled_twins else ("untitled",)
                dependency["headings"] = (
                    "untitled in canonical/ (root heading only), titled twin in canonical/titled/"
                    if titled_twins
                    else "untitled in canonical/ (root heading only)"
                )
                for format_name, kind in [(f, k) for f in formats for k in kinds]:
                    is_twin = kind == "titled"
                    target = figure_path(OUTPUT, stem, ext=format_name, titled=is_twin)
                    record_path = (
                        RECORD_DIR / ("titled" if is_twin else "") / f"{stem}.{format_name}.json"
                    )
                    if is_twin and plain is False:
                        # The untitled variant could not be made: canonical/ keeps today's
                        # chart, so no outdated twin or twin record may survive.
                        target.unlink(missing_ok=True)
                        record_path.unlink(missing_ok=True)
                        continue
                    # Figures are not cached. A build renders what it is asked,
                    # every time; a renderer edit is not a key.
                    if chart is None:
                        try:
                            chart = spec.render(inputs, settings, variant)
                        except QuantityGapError as gap:
                            if spec.default:
                                raise
                            if not is_stage_not_run(gap.reason):
                                raise QuantityError(
                                    f"{figure_id} ({variant}) cannot be skipped: "
                                    f"{gap.key} is not awaiting a stage, it is "
                                    f"{gap.reason!r}"
                                ) from gap
                            # An opt-in figure whose stage has not run yet: skip it, and
                            # remove only its dependency records. The figure files stay:
                            # output/figures/*.png is git-tracked (HPC-rendered), and
                            # figure_return copies back only PNGs that have a record, so
                            # a stale PNG cannot be copied back without one.
                            for stale_format in formats:
                                (RECORD_DIR / f"{stem}.{stale_format}.json").unlink(
                                    missing_ok=True
                                )
                                (RECORD_DIR / "titled" / f"{stem}.{stale_format}.json").unlink(
                                    missing_ok=True
                                )
                            if _raise_skipped:
                                raise
                            print(
                                f"Skipped {figure_id} ({variant}): awaiting HPC run: "
                                f"{gap.key}: {gap.reason}"
                            )
                            break
                    to_save = chart
                    if kind == "untitled":
                        if plain is None:
                            stripped = without_headings(chart)
                            plain = False if stripped is None else stripped[0]
                            if stripped is None:
                                print(
                                    f"Headings: {figure_id} ({variant}) could not be "
                                    "stripped; canonical keeps today's chart and no titled twin is written"
                                )
                            elif stripped[1] == 0:
                                print(f"Headings: {figure_id} ({variant}) has none to drop")
                        if plain is not False:
                            to_save = plain
                    # Remove the success record before rendering. A failure will therefore
                    # leave the target stale and the next build will retry it.
                    record_path.unlink(missing_ok=True)
                    # Drop the tracked or stale file first: a failure leaves none.
                    target.unlink(missing_ok=True)
                    _render_one(to_save, target, format_name, png_dpi=png_dpi, footer=footer)
                    rendered_now.add(target)
                    _write_json_atomic(record_path, dependency)
                    if not is_twin:
                        outputs.append(target)
                    print(f"Rendered: {_display_path(target)}")
            except Exception as exc:  # noqa: BLE001 - collect every figure failure
                if _raise_skipped and isinstance(exc, QuantityGapError):
                    raise
                failed_stem = (
                    figure_id
                    if len(spec.variants) == 1 or variant == "annotated"
                    else f"{figure_id}_vanilla"
                ) + stem_suffix
                _record_figure_failure(
                    failures, causes, figure_id, variant, exc,
                    [
                        figure_path(OUTPUT, failed_stem, ext=f, titled=t)
                        for f in formats for t in (False, True)
                        if figure_path(OUTPUT, failed_stem, ext=f, titled=t)
                        not in rendered_now
                    ],
                )
                # A target deleted for this failure must not keep its render record, or
                # figure_return would refuse "<png> is missing but has a record".
                for f in formats:
                    for t in (False, True):
                        if figure_path(OUTPUT, failed_stem, ext=f, titled=t) in rendered_now:
                            continue
                        (
                            RECORD_DIR / ("titled" if t else "") / f"{failed_stem}.{f}.json"
                        ).unlink(missing_ok=True)
    print(
        f"Figures rendered: {len(outputs)}; failed: {len(failures)}"
        + (": " + ", ".join(failures) if failures else ""),
        flush=True,
    )
    if failures:
        raise FiguresFailedError(failures, causes)
    return outputs


def build_slides(*, context: Any | None = None) -> list[Path]:
    """Render the frameless slides of the registry tree figures (``build figures --slides``).

    They go to ``output/figures/slides/{tree_only,full}/``, never over the canonical
    figures; the inputs are those of the figures they redraw.
    """
    from ..figures.registry import _COMPOSERS
    from .registry_slides import SLIDE_FIGURE_IDS, build_registry_slides

    settings = load_settings()
    specs = {spec.figure_id: spec for spec in FIGURE_SPECS}
    chosen = [specs[figure_id] for figure_id in SLIDE_FIGURE_IDS]
    inputs = load_figure_inputs(chosen, settings, context=context)
    require_current_registry(settings)
    register_font_directory()
    return build_registry_slides(inputs, settings, FIGURE_DIR, _COMPOSERS)


def cmd_figures(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    if getattr(args, "opt_in", False):
        args.figures = list(
            opt_in_selection({spec.figure_id: spec for spec in FIGURE_SPECS})
        )
    if args.keep_going:
        return 0 if build_figures_keep_going(
            args.figures, force=args.force, variants=args.variants,
            formats=args.formats, label_collision=args.label_collision,
            with_bootstrap=args.with_bootstrap, context=context,
            titled_twins=args.titled_twins,
        ) else 1
    build_figures(
        args.figures,
        force=args.force,
        variants=args.variants,
        formats=args.formats,
        label_collision=args.label_collision,
        with_bootstrap=args.with_bootstrap,
        titled_twins=args.titled_twins,
        context=context,
    )
    return 0


def cmd_all(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, context: Any
) -> int:
    requested = list(args.force_steps)

    def step_force(*names: str) -> bool:
        return "*" in requested or any(name in requested for name in names)

    # The Candidate graph is the Functional class of the classification, which
    # reads the growth rates (the justfile chain: sequence-growth-rates,
    # functional-classification, then every Candidate stage). Both are current
    # or computed here; neither is forced by `--force`, which is for the rebuild.
    from ..analysis.sequence_growth_rates import compute_sequence_growth_rates
    from ..threshold.functional_classification import (
        compute_functional_classification,
    )

    build_sequence_growth_rate_tables(
        compute_sequence_growth_rates(load_settings())
    )
    compute_functional_classification(load_settings())
    if args.figures is None or EDGE_FIGURE_ID in args.figures:
        from ..analysis.candidate_edge_evidence import (
            load_or_compute_candidate_edge_evidence,
        )

        load_or_compute_candidate_edge_evidence(force=args.force)
    ensure_analysis_checkpoint()
    build_tables(force=step_force("tables", "codon-entropy", "analysis-tables"))
    from ..analysis.clamped_cache import load_or_compute_clamped_comparison

    comparison = load_or_compute_clamped_comparison(
        force=step_force("comparison-analysis", "clamped-comparison")
    )
    build_comparison_tables(
        comparison, force=step_force("clamped-comparison-tables", "tables")
    )
    build_codon_entropy_tables(
        comparison, force=step_force("codon-entropy", "tables")
    )
    if args.figures is None or PEAK_FIGURE_ID in args.figures:
        build_peak_tables(force=step_force("peak-analysis", "peak-uncertainty-tables"))
        from .pooled_scatter_product import build_pooled_scatter_product

        build_pooled_scatter_product(context=context)
    # The registry reads these two stages (the justfile chain: leading-saddle-band,
    # biological-annotation-tables, manuscript-quantities, then figures), and the
    # figures that read labels from it check it is current. The bootstrap and branch
    # outputs are never produced here: the registry reads them when present and
    # names a gap row otherwise, as `just all` does.
    build_leading_saddle_band_tables(force=args.force)
    build_biological_tables(comparison, force=args.force)
    # PR 3b: the bootstrap codon entropy product must run after band/tree
    # layout products and before manuscript-quantities so the figure can
    # load the precomputed tables.
    if args.figures is None or BOOTSTRAP_CODON_ENTROPY_FIGURE_ID in (
        args.figures or ()
    ):
        from .bootstrap_codon_entropy_product import (
            build_bootstrap_codon_entropy_product,
        )

        build_bootstrap_codon_entropy_product(context=context)
    # The band quantile scan figures are opt-in (they need the bootstrap
    # checkpoint), so their product is built only when they are requested.
    if {BAND_QUANTILE_SCAN_FIGURE_ID, BAND_QUANTILE_SCAN_SEEDS_FIGURE_ID} & set(
        args.figures or ()
    ):
        from .bootstrap_band_quantile_scan_product import (
            build_bootstrap_band_quantile_scan_product,
        )

        build_bootstrap_band_quantile_scan_product(context=context)
    if args.figures is None or MANIFOLD_FIGURE_ID in (args.figures or ()):
        from .graph_tree_persistence_product import (
            build_graph_tree_persistence_product,
        )

        build_graph_tree_persistence_product(context=context)
    build_manuscript_quantities(force=args.force)
    if args.figures is None or {
        "full_tree_persistence",
        "full_tree_persistence_vertical_histogram",
    } & set(args.figures or ()):
        from .threshold_diagnostics_product import (
            build_threshold_diagnostics_product,
        )

        build_threshold_diagnostics_product(context=context)
    if args.slides:
        build_slides(context=context)
        return 0
    build_figures(
        args.figures,
        force=args.force,
        variants=args.variants,
        formats=args.formats,
        context=context,
    )
    return 0
