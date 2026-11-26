"""Publish the threshold diagnostics (candidate and reflected T with unrounded, saddles) for the full-graph figure from raw fitness and legacy component persistence (canonical location rule)."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import ROOT, load_settings
from ..analysis.checkpoint import checkpoint_lock
from ..analysis.component_persistence import (
    component_paths,
    load_cached_component_persistence,
)
from ..analysis.leading_saddle_band import tree_saddles
from ..landscape.full_population import load_full_network_fitness
from ..landscape.products import _publish
from ..landscape.selection import candidate_threshold_from_summary
from ..results import RunContext
from ..threshold.fdr_selection import _summary_and_path, threshold_summary
from ..threshold.functional_threshold import (
    FDR_FIXED_CUTOFF,
    MEDIAN_REFLECTED,
    comparison_declaration,
    compute_candidate_functional_threshold,
    configured_declaration,
    declaration_identity,
)


def _table_sha256(declaration: Any) -> str:
    """The sha256 of the declared processed classification table."""
    _, table_path, _ = _summary_and_path(declaration)
    return _file_sha256(table_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _configured_input_path(context: RunContext | None, value: str) -> Path:
    """Resolve configured raw-table paths through the context's raw-data directory."""
    path = Path(value)
    if path.parts[:2] == ("data", "raw"):
        if context is not None and getattr(context, "raw_data_dir", None):
            return context.raw_data_dir.joinpath(*path.parts[2:])
        return ROOT / value
    return path if path.is_absolute() else ROOT / value


def _threshold_inputs(
    context: RunContext, settings: dict[str, Any]
) -> tuple[dict[str, str], pl.DataFrame, Any]:
    """Return (hashes, pairs, summary) doing every read once, in specified order."""
    input_data: dict[str, str] = {}
    paths = component_paths(context)
    with checkpoint_lock(
        paths.analysis_dir / "component_persistence", "component analysis"
    ):
        pairs, _ = load_cached_component_persistence(context=context)
        input_data["component_pairs.parquet"] = _file_sha256(paths.pairs)
        input_data["component_diagnostics.json"] = _file_sha256(paths.diagnostics)
    dataset = settings["analysis"]["dataset"]
    wt_path = _configured_input_path(context, dataset["fitness_wt"])
    diff_path = _configured_input_path(context, dataset["fitness_diff"])
    input_data["fitness_wt"] = _file_sha256(wt_path)
    input_data["fitness_diff"] = _file_sha256(diff_path)
    declaration = configured_declaration(settings)
    rule = declaration.get("rule")
    summary = None
    if rule == MEDIAN_REFLECTED:
        pass
    elif rule == FDR_FIXED_CUTOFF:
        snap = threshold_summary(declaration)
        summary_json = json.dumps(snap, sort_keys=True).encode("utf-8")
        summary_sha = hashlib.sha256(summary_json).hexdigest()
        input_data["fdr_threshold_summary"] = summary_sha
        input_data["fdr_table_sha256"] = _table_sha256(declaration)
        summary = snap
    else:
        from ..threshold.functional_classification import (
            load_cached_functional_classification_with_source,
        )

        result, _ = load_cached_functional_classification_with_source(
            dict(settings), tables=False, lock=True, context=context
        )
        summary = result.summary
        summary_json = json.dumps(summary, sort_keys=True).encode("utf-8")
        summary_sha = hashlib.sha256(summary_json).hexdigest()
        input_data["functional_classification_summary"] = summary_sha
    return input_data, pairs, summary


def build_threshold_diagnostics_product(context: RunContext | None = None) -> Path:
    """Build the threshold_diagnostics product for the full graph.

    Loads full fitness from the configured raw tables (hashing wt and diff plus
    loader code), reads the legacy component-persistence checkpoint under its lock
    (after verifying canonical location), calls the threshold fit functions
    verbatim on the full-fitness values and the full declaration, and saves the
    exact values (including unrounded) and saddle deaths the figure uses.
    """
    context = context or RunContext()
    settings = load_settings(context=context)
    dataset = settings["analysis"]["dataset"]
    wt_path = _configured_input_path(context, dataset["fitness_wt"])
    diff_path = _configured_input_path(context, dataset["fitness_diff"])
    root_wt = ROOT / dataset["fitness_wt"]
    root_diff = ROOT / dataset["fitness_diff"]
    if (
        wt_path.resolve() != root_wt.resolve()
        or diff_path.resolve() != root_diff.resolve()
    ):
        raise ValueError(
            "raw data override not supported for threshold-diagnostics-product; "
            "the component persistence checkpoint is only available at the default "
            "data root"
        )
    input_data_sha256s, pairs, summary = _threshold_inputs(context, settings)
    full_fitness = load_full_network_fitness(settings)
    values = full_fitness["raw_fitness"].to_list()

    declaration = configured_declaration(settings)

    computed = compute_candidate_functional_threshold(
        values, declaration, classification_summary=summary
    )
    rule_name = computed.rule
    if rule_name == MEDIAN_REFLECTED:
        reflected = compute_candidate_functional_threshold(
            values, comparison_declaration(declaration)
        )
    else:
        reflected = None

    # capture summary sha (from the hashes returned by helper)
    summary_sha = None
    if "fdr_threshold_summary" in input_data_sha256s:
        summary_sha = input_data_sha256s["fdr_threshold_summary"]
    elif "functional_classification_summary" in input_data_sha256s:
        summary_sha = input_data_sha256s["functional_classification_summary"]

    saddles = tree_saddles(pairs)
    lowest_split_saddle_death = (
        None if saddles.split_trailing is None else saddles.split_trailing.fitness
    )
    highest_join_saddle_death = (
        None if saddles.join_trailing is None else saddles.join_trailing.fitness
    )

    threshold_diagnostics = {
        "candidate": {
            "rule": computed.rule,
            "decimals": computed.decimals,
            "median": computed.median,
            "global_minimum": computed.global_minimum,
            "median_gap": computed.median_gap,
            "unrounded": computed.unrounded,
            "threshold": computed.threshold,
        },
        "lowest_split_saddle_death": lowest_split_saddle_death,
        "highest_join_saddle_death": highest_join_saddle_death,
    }
    if reflected is not None:
        threshold_diagnostics["reflected"] = {
            "rule": reflected.rule,
            "decimals": reflected.decimals,
            "median": reflected.median,
            "global_minimum": reflected.global_minimum,
            "median_gap": reflected.median_gap,
            "unrounded": reflected.unrounded,
            "threshold": reflected.threshold,
        }

    code_paths = (
        Path(inspect.getfile(compute_candidate_functional_threshold)).resolve(),
        Path(inspect.getfile(candidate_threshold_from_summary)).resolve(),
        Path(inspect.getfile(tree_saddles)).resolve(),
        Path(inspect.getfile(load_full_network_fitness)).resolve(),
        Path(__file__).resolve(),
    )

    label_fields = {"graph": "full"}

    identity = {
        **label_fields,
        "candidate_functional_threshold_declaration": declaration_identity(declaration),
    }
    if summary_sha:
        identity["classification_summary_sha256"] = summary_sha

    def write_artifacts(directory: Path) -> None:
        (directory / "threshold_diagnostics.json").write_text(
            json.dumps(threshold_diagnostics, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    path, _ = _publish(
        "threshold_diagnostics",
        label_fields,
        identity,
        context=context,
        input_data_sha256s=input_data_sha256s,
        parents=(),
        code_paths=code_paths,
        write_artifacts=write_artifacts,
    )
    return path
