"""Resolve the Selected graph's threshold and classification.

``resolve_selection`` is the one place that turns the ``[clamped]
candidate_functional_threshold`` declaration into a selection.

- ``mixture_quantile`` (active): the Functional class of the
  ``functional-classification`` checkpoint selects the vertices and its ``T`` is the
  reported threshold. The checkpoint is read fail-closed: a missing or stale one
  raises ``FileNotFoundError`` naming ``just functional-classification``.
- ``median_reflected`` (comparison): ``T = m + h`` from the Full-graph raw fitness,
  and every vertex at or above it is functional (``functional_sequences`` is
  ``None``, which ``prepare_clamped_landscape`` reads as the raw-fitness edge test).

Under ``mixture_quantile`` the wild type must be Functional (the rule's anchor: the
classification has gone wrong otherwise); ``WildTypeNotFunctionalError`` names it.

``source`` is the identity of whatever the selection read, for stage fingerprints:
a re-run classification changes it, so every downstream checkpoint goes stale.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..analysis import ROOT, dataset_input_path
from ..threshold.fdr_selection import (
    load_fdr_snapshot,
    require_full_population,
    threshold_summary,
)
from ..threshold.functional_threshold import (
    FDR_FIXED_CUTOFF,
    FIXED_CUTOFF,
    MEDIAN_REFLECTED,
    MIXTURE_QUANTILE,
    CandidateThreshold,
    compute_candidate_functional_threshold,
    configured_declaration,
    network_raw_fitness,
    validate_candidate_declaration,
)
from .selected_landscape import PreparedLandscape, prepare_clamped_landscape


class WildTypeNotFunctionalError(ValueError):
    """The wild type is not in the Functional class of the classification."""


@dataclass(frozen=True)
class Selection:
    """The active Selection rule, its threshold and the Functional vertex set."""

    rule: str
    threshold: CandidateThreshold
    functional_sequences: frozenset[str] | None
    source: dict[str, Any]
    # Class of every classified sequence (functional, non_functional, uncertain),
    # or ``None`` under the scalar rule. Not part of equality or the repr.
    classes: Mapping[str, str] | None = field(default=None, compare=False, repr=False)
    # Uncertain sequences, or ``None`` under the scalar rule.
    uncertain_count: int | None = None

    def to_json(self) -> dict[str, Any]:
        """The threshold record plus the vertex counts, for the registry."""
        return {
            **self.threshold.to_json(),
            "functional_vertex_count": (
                None
                if self.functional_sequences is None
                else len(self.functional_sequences)
            ),
            "uncertain_vertex_count": self.uncertain_count,
        }


def _declared_rule(settings: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    declaration = configured_declaration(settings)
    rule, _ = validate_candidate_declaration(declaration)
    return rule, declaration


def _raw_fitness(
    settings: Mapping[str, Any],
    full_network: Any,
    raw_fitness: Sequence[float] | None,
    context: Any | None = None,
) -> Sequence[float]:
    if raw_fitness is not None:
        return raw_fitness
    if full_network is not None:
        return network_raw_fitness(full_network)
    # Imported here because ``full_population`` imports the analysis package.
    from .full_population import load_full_network_fitness

    return load_full_network_fitness(dict(settings), context=context)[
        "raw_fitness"
    ].to_list()


def wild_type_sequence(
    settings: Mapping[str, Any], context: Any | None = None
) -> str | None:
    """The wild-type DNA sequence of the configured ``fitness_wt`` table.

    ``None`` when the settings declare no dataset (a fixture that has no table).
    """
    dataset = settings["analysis"].get("dataset")
    if not dataset or "fitness_wt" not in dataset:
        return None
    table = pl.read_parquet(
        dataset_input_path(dataset["fitness_wt"], context, root=ROOT),
        columns=["dna_from"],
    )
    return str(table.get_column("dna_from")[0])


def _require_wild_type_functional(
    classes: Mapping[str, str], wild_type: str | None
) -> None:
    if wild_type is None:
        return
    found = classes.get(wild_type)
    if found != "functional":
        raise WildTypeNotFunctionalError(
            f"The wild type {wild_type!r} is classed {found!r}, not 'functional', "
            "so the functional classification is not usable as the Candidate "
            "selection; check the growth rates and rerun "
            "`just functional-classification`."
        )


def resolve_selection(
    settings: Mapping[str, Any],
    *,
    full_network: Any = None,
    raw_fitness: Sequence[float] | None = None,
    lock: bool = True,
    wild_type: str | None = None,
    context: Any | None = None,
) -> Selection:
    """Resolve the declared Selection rule against the current checkpoints.

    ``full_network`` or ``raw_fitness`` give the Full-graph fitness the ``m``, ``g``
    and ``h`` of the record come from; with neither, the configured Full graph is
    read. Raises ``FileNotFoundError`` when ``mixture_quantile`` is declared and the
    classification checkpoint is missing or stale. ``lock=False`` reads the
    checkpoint without its lock (see ``load_cached_functional_classification_with_
    source``). Under ``mixture_quantile`` the wild type (``wild_type``, else the one
    of the configured ``fitness_wt`` table, if any) must be Functional.
    """
    rule, declaration = _declared_rule(settings)
    values = _raw_fitness(settings, full_network, raw_fitness, context)
    if rule in (MEDIAN_REFLECTED, FIXED_CUTOFF):
        threshold = compute_candidate_functional_threshold(values, declaration)
        return Selection(rule, threshold, None, {"rule": rule})

    from ..threshold.functional_classification import (
        FUNCTIONAL,
        UNCERTAIN,
        load_cached_functional_classification_with_source,
    )

    if rule == FDR_FIXED_CUTOFF:
        snapshot = load_fdr_snapshot(
            declaration,
            wild_type=wild_type
            if wild_type is not None
            else wild_type_sequence(settings, context),
        )
        threshold = compute_candidate_functional_threshold(
            values, declaration, classification_summary=snapshot.summary
        )
        table = snapshot.table.rename({"fdr_class": "functional_class"})
        source = snapshot.source
    else:
        result, source = load_cached_functional_classification_with_source(
            dict(settings), lock=lock, context=context
        )
        threshold = compute_candidate_functional_threshold(
            values, declaration, classification_summary=result.summary
        )
        table = result.classification
    functional = frozenset(
        table.filter(pl.col("functional_class") == FUNCTIONAL)["dna_sequence"]
    )
    classes = dict(
        zip(
            table["dna_sequence"].to_list(),
            table["functional_class"].to_list(),
            strict=True,
        )
    )
    _require_wild_type_functional(
        classes,
        wild_type if wild_type is not None else wild_type_sequence(settings, context),
    )
    return Selection(
        rule,
        threshold,
        functional,
        {"rule": rule, **source},
        classes=classes,
        uncertain_count=int((table["functional_class"] == UNCERTAIN).sum()),
    )


def threshold_from_summary(
    settings: Mapping[str, Any],
    *,
    raw_fitness: Sequence[float] | None = None,
    lock: bool = True,
    context: Any | None = None,
) -> CandidateThreshold:
    """The Candidate threshold record alone, from the classification summary.

    For a caller that needs ``T`` and the Full-graph ``m``, ``g`` and ``h`` but not the
    vertex set (a figure's rule, a table's band): under ``mixture_quantile`` only
    ``summary.json`` of the checkpoint is read (``tables=False``), not its 261,333
    rows. Same failure modes as ``resolve_selection``.
    """
    rule, declaration = _declared_rule(settings)
    values = _raw_fitness(settings, None, raw_fitness, context)
    if rule in (MEDIAN_REFLECTED, FIXED_CUTOFF):
        return compute_candidate_functional_threshold(values, declaration)
    if rule == FDR_FIXED_CUTOFF:
        return compute_candidate_functional_threshold(
            values, declaration, classification_summary=threshold_summary(declaration)
        )
    from ..threshold.functional_classification import (
        load_cached_functional_classification_with_source,
    )

    result, _ = load_cached_functional_classification_with_source(
        dict(settings), tables=False, lock=lock, context=context
    )
    return compute_candidate_functional_threshold(
        values, declaration, classification_summary=result.summary
    )


def prepare_selected_landscape(
    full_network: Any,
    settings: Mapping[str, Any],
    *,
    selection: Selection | None = None,
    context: Any | None = None,
) -> tuple[PreparedLandscape, Selection]:
    """Select the Candidate graph of a Full network: the one shared path.

    Every Candidate network builder calls this, so the rule, the threshold and the
    Functional vertex set are never recomputed in a second place. A caller that has
    already resolved the selection (the sliced bootstrap, which reads the
    classification once and without the lock) passes it as ``selection``.
    """
    if selection is None:
        selection = resolve_candidate_selection(
            settings, full_network=full_network, context=context
        )
    if selection.rule == FDR_FIXED_CUTOFF:
        graph = full_network.graph
        require_full_population(
            selection.classes,
            [graph[index]["dna_sequence"] for index in graph.node_indices()],
        )
    prepared = prepare_clamped_landscape(
        full_network,
        selection.threshold.threshold,
        functional_sequences=selection.functional_sequences,
        functional_classes=selection.classes,
    )
    return prepared, selection


def selection_source(
    settings: Mapping[str, Any], *, context: Any | None = None
) -> dict[str, Any]:
    """The identity of what the selection reads, without reading raw data.

    ``{"rule": ...}`` under ``median_reflected``; under ``mixture_quantile`` also the
    classification checkpoint's fingerprint and artefact hashes. Raises
    ``FileNotFoundError`` for a missing or stale classification.
    """
    rule, _ = _declared_rule(settings)
    if rule in (MEDIAN_REFLECTED, FIXED_CUTOFF):
        return {"rule": rule}
    return _classification_identity(settings, context=context)[1]


def _classification_identity(
    settings: Mapping[str, Any], *, context: Any | None = None
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    rule, declaration = _declared_rule(settings)
    if rule == FDR_FIXED_CUTOFF:
        snapshot = load_fdr_snapshot(
            declaration, wild_type=wild_type_sequence(settings)
        )
        return snapshot.summary, snapshot.source
    from ..threshold.functional_classification import (
        load_cached_functional_classification_with_source,
    )

    result, source = load_cached_functional_classification_with_source(
        dict(settings), tables=False, context=context
    )
    return result.summary, {"rule": MIXTURE_QUANTILE, **source}


def _fdr_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    """Scientific evidence for an FDR snapshot, excluding loader-code provenance."""
    return {
        "fdr_cutoff": source["cutoff"],
        "fdr_method": source["method"],
        "fdr_q": source["q"],
        "fdr_table_sha256": source["table_sha256"],
    }


def selection_identity(selection: Selection) -> dict[str, Any]:
    """The identity of a resolved selection (see ``selected_graph_identity``)."""
    if selection.rule == FIXED_CUTOFF:
        return {"rule": selection.rule, "value": float(selection.threshold.threshold)}
    if selection.rule == MEDIAN_REFLECTED or selection.functional_sequences is None:
        return {"rule": selection.rule}
    functional = sorted(selection.functional_sequences)
    digest = hashlib.sha256("\n".join(functional).encode("utf-8")).hexdigest()
    extra = (
        _fdr_identity(selection.source) if selection.rule == FDR_FIXED_CUTOFF else {}
    )
    return {
        **extra,
        "rule": selection.rule,
        "rounded_threshold": float(selection.threshold.threshold),
        "functional_vertex_count": len(functional),
        "functional_set_sha256": digest,
    }


def selected_graph_identity(
    settings: Mapping[str, Any],
    selection: Selection | None = None,
    *,
    context: Any | None = None,
) -> dict[str, Any]:
    """What of the selection decides a result, free of numerics and library versions.

    For a scientific identity that must match across machines (the bootstrap
    mirror): the rule and, under ``mixture_quantile``, the rounded ``T``, the
    Functional vertex count and the sha256 of the sorted Functional DNA sequences.
    The classification checkpoint's own fingerprint is left out because it hashes
    the growth-rate table's bytes. Raises ``FileNotFoundError`` for a missing or
    stale classification. With a resolved ``selection`` nothing is read, so a caller
    that already holds one does not read the checkpoint a second time.
    """
    if selection is not None:
        return selection_identity(selection)
    rule, declaration = _declared_rule(settings)
    if rule == FIXED_CUTOFF:
        return {"rule": rule, "value": float(declaration["value"])}
    if rule == MEDIAN_REFLECTED:
        return {"rule": rule}
    from ..threshold.functional_classification import (
        FUNCTIONAL,
        load_cached_functional_classification_with_source,
    )

    extra: dict[str, Any] = {}
    if rule == FDR_FIXED_CUTOFF:
        snapshot = load_fdr_snapshot(
            declaration, wild_type=wild_type_sequence(settings)
        )
        summary = snapshot.summary
        table = snapshot.table.rename({"fdr_class": "functional_class"})
        extra = _fdr_identity(snapshot.source)
    else:
        result, _ = load_cached_functional_classification_with_source(
            dict(settings), context=context
        )
        summary, table = result.summary, result.classification
    functional = sorted(
        table.filter(pl.col("functional_class") == FUNCTIONAL)["dna_sequence"]
    )
    digest = hashlib.sha256("\n".join(functional).encode("utf-8")).hexdigest()
    return {
        **extra,
        "rule": rule,
        "rounded_threshold": float(summary["threshold"]["rounded"]),
        "functional_vertex_count": len(functional),
        "functional_set_sha256": digest,
    }


def selection_fingerprint(
    settings: Mapping[str, Any], *, context: Any | None = None
) -> dict[str, Any]:
    """What a stage fingerprint carries about the selection: ``T`` and the source.

    Cheap: under ``mixture_quantile`` it reads only the classification's
    ``summary.json`` and provenance, under ``median_reflected`` the Full-graph raw
    fitness. A downstream checkpoint goes stale when the classification is re-run.
    """
    rule, declaration = _declared_rule(settings)
    if rule in (MEDIAN_REFLECTED, FIXED_CUTOFF):
        threshold = compute_candidate_functional_threshold(
            _raw_fitness(settings, None, None, context), declaration
        )
        return {"rounded_threshold": threshold.threshold, "source": {"rule": rule}}
    summary, source = _classification_identity(settings, context=context)
    return {
        "rounded_threshold": float(summary["threshold"]["rounded"]),
        "source": source,
    }


__all__ = [
    "MIXTURE_QUANTILE",
    "CandidateSelection",
    "Selection",
    "WildTypeNotFunctionalError",
    "candidate_selection_fingerprint",
    "candidate_selection_identity",
    "candidate_selection_source",
    "candidate_threshold_from_summary",
    "prepare_candidate_landscape",
    "prepare_selected_landscape",
    "resolve_candidate_selection",
    "resolve_selection",
    "selected_graph_identity",
    "selection_fingerprint",
    "selection_identity",
    "selection_source",
    "threshold_from_summary",
    "wild_type_sequence",
]

# Historical imports and serialized names remain resolvable through the analysis
# forwarding module. New callers use the domain-first names above.
CandidateSelection = Selection
candidate_selection_fingerprint = selection_fingerprint
candidate_selection_identity = selected_graph_identity
candidate_selection_source = selection_source
candidate_threshold_from_summary = threshold_from_summary
prepare_candidate_landscape = prepare_selected_landscape


def resolve_candidate_selection(
    settings: Mapping[str, Any],
    *,
    full_network: Any = None,
    raw_fitness: Sequence[float] | None = None,
    lock: bool = True,
    wild_type: str | None = None,
    context: Any | None = None,
) -> Selection:
    """Compatibility adapter for the former analysis-module resolver name."""
    return resolve_selection(
        settings,
        full_network=full_network,
        raw_fitness=raw_fitness,
        lock=lock,
        wild_type=wild_type,
        context=context,
    )
