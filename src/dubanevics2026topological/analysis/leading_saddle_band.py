"""The leading saddle band of the Full graph: landmarks, intervals and entropy.

The Full graph's split tree and join tree each close their ordinary H0 pairs at
saddles (GLOSSARY: Leading saddle, Trailing saddle). The band is the closed
interval between the two trailing saddles. This module reads those four saddles from the cached full-network component pairs
(``analysis/component_persistence.py``; nothing is rebuilt), places them among
the other landmarks of the fitness axis, counts the sequences between
consecutive landmarks and measures positional entropy in the band and in each
trunk. It computes ordinary H0 pair deaths only: no branch span and no extended
persistence.

Quantities (all raw fitness; every saddle carries its DNA sequence identity):

* Split tree, from the pairs of the descending sweep (the cached ascending pairs
  of kind ``relative``): the **leading saddle** is the highest split saddle death
  and the **trailing saddle** ``b_split`` the lowest. This is the computation
  behind ``TopologicalNoiseBand.lowest_split_saddle_death`` of the noise-band
  figure, which now calls :func:`tree_saddles` too.
* Join tree, from the cached ascending pairs of kind ``ordinary``: the leading
  saddle is the lowest join saddle death and the trailing saddle ``b_join`` the
  highest. It must equal ``join_tree_trailing_saddle`` of
  ``build_threshold_diagnostics`` (checked; the rule is not repeated silently).
* Ties between saddles of equal fitness are broken by the smaller DNA sequence.

Landmarks (``LANDMARK_KEYS``): the global minimum ``g``, the two trailing
saddles, the median ``m``, the two leading saddles, the Published functional
threshold and the computed Candidate functional threshold (rounded, from
``candidate_threshold``). The minimum and the saddles are vertices and carry a ``sequence``. The
thresholds are values of the axis and never do. The median carries one only when
exactly one vertex has raw fitness equal to it (for instance an odd-sized
population whose middle value is unique); otherwise it is null.

Every saddle, and the minimum, must be a member of the Full-graph fitness
population with ``raw_fitness`` equal to the pair's scalar; the stage raises
otherwise. Zero-persistence pairs are not filtered: a pair whose death equals its
birth still counts as a saddle, consistent with ``join_tree_trailing_saddle`` of
``build_threshold_diagnostics``.

The **band** is the closed interval ``[band_lower, band_upper]`` between the two
trailing saddles, whichever tree has the lower one (``band_lower_landmark``);
``median_inside_band`` tests it. The **split-tree trunk** is every sequence with
raw fitness at or below ``b_split`` (toward the tree's root, the global minimum;
the trailing saddle itself belongs to the trunk). The **join-tree trunk** is every
sequence at or above ``b_join`` (toward the join tree's root, the global maximum;
the saddle belongs to the trunk). Both trunks are closed.

The band's connectivity is the undirected simple subgraph the Full graph
induces on that same closed interval. Vertices are matched by ``dna_sequence``.
Self-loops are excluded and parallel edges count once. ``n_components``,
``largest_component_size``, ``largest_component_fraction``
(``largest_component_size / band_count``), ``n_singletons`` and
``band_internal_edges`` record it. The graph is passed in; this module does not
build or read it.

``band_kind`` records how the band sits against the trunks:

* ``"overlap"`` when ``b_join <= b_split``: the band is the intersection of the
  two trunks. Equal saddles are filed here, so the band is the single value both
  trunks reach and ``band_count`` counts the sequences at exactly that fitness.
* ``"gap"`` when ``b_join > b_split``: the band lies in neither trunk, apart from
  its two closed endpoints, ``b_split`` (in the split trunk) and ``b_join`` (in
  the join trunk).

The band and trunk populations are therefore not a partition of the Full graph: in
the overlap case they intersect, so their counts do not sum to the total. In the
gap case the two trunks and the band's interior together cover every sequence.

Interval convention: the landmarks sorted by fitness (ties keep the table order)
bound consecutive intervals, half-open ``[lower, upper)`` except the last, which
is closed ``[lower, upper]``. A sequence exactly on a landmark therefore belongs
to the interval that starts there. ``below_first_landmark_count`` and
``above_last_landmark_count`` count what lies outside all intervals; together with
the intervals they cover every sequence once.

Entropy: Shannon entropy in bits of the amino acid at each of the three codon
positions (stop codons keep their distinct symbols), for the band, for each
trunk and for each landmark interval (same half-open convention), through ``codon_entropy.positional_entropy``. A population without
sequences has null entropy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rustworkx

from ..threshold.functional_threshold import (
    CandidateThreshold,
    compute_candidate_functional_threshold,
)
from . import _write_json_atomic
from .codon_entropy import positional_entropy
from .codon_entropy_populations import CODON_TRANSLATION
from .component_persistence import build_threshold_diagnostics, validate_pair_table

STAGE_NAME = "leading-saddle-band"
RETRY_HINT = "just leading-saddle-band"
SUMMARY_SCHEMA = 1
LANDMARKS_TABLE = "leading_saddle_band_landmarks"
INTERVALS_TABLE = "leading_saddle_band_intervals"
ENTROPY_TABLE = "leading_saddle_band_entropy"
TABLE_NAMES = (LANDMARKS_TABLE, INTERVALS_TABLE, ENTROPY_TABLE)
ARTEFACTS = tuple(
    f"{name}.{suffix}"
    for name in TABLE_NAMES
    for suffix in ("csv", "parquet", "schema.json")
) + ("summary.json",)

# The landmarks in table order. Saddle and minimum landmarks carry a sequence.
LANDMARK_KEYS = (
    "global_minimum",
    "split_trailing_saddle",
    "join_trailing_saddle",
    "median",
    "split_leading_saddle",
    "join_leading_saddle",
    "published_functional_threshold",
    "candidate_functional_threshold",
)
SPLIT_TRAILING = "split_trailing_saddle"
JOIN_TRAILING = "join_trailing_saddle"
BAND_OVERLAP = "overlap"
BAND_GAP = "gap"
POPULATIONS = ("band", "split_trunk", "join_trunk")
# An interval is named ``<lower landmark>__<upper landmark>``; its entropy rows use
# that name as their population.
INTERVAL_SEPARATOR = "__"
POSITIONS = (1, 2, 3)

LANDMARK_SCHEMA = {
    "landmark": pl.String,
    "fitness": pl.Float64,
    "sequence": pl.String,
    "source": pl.String,
    "band_kind": pl.String,
    "median_inside_band": pl.Boolean,
}
INTERVAL_SCHEMA = {
    "interval_index": pl.Int64,
    "interval": pl.String,
    "lower_landmark": pl.String,
    "upper_landmark": pl.String,
    "lower_fitness": pl.Float64,
    "upper_fitness": pl.Float64,
    "upper_closed": pl.Boolean,
    "sequence_count": pl.Int64,
    "percentage": pl.Float64,
}
ENTROPY_SCHEMA = {
    "population": pl.String,
    "position": pl.Int64,
    "entropy": pl.Float64,
    "sequence_count": pl.Int64,
}
DESCRIPTIONS = {
    LANDMARKS_TABLE: (
        "One row per landmark of the Full-graph fitness axis, raw fitness. "
        "sequence is the DNA identity of the global minimum and of the four saddles "
        "and null for the two functional thresholds, which are not vertices. For "
        "the median it is set only when exactly one vertex has raw fitness equal "
        "to the median, and null otherwise. source names where the value comes "
        "from. band_kind (overlap or gap) and median_inside_band (closed band) "
        "describe the band, not the landmark: they are constant down the column "
        "and repeat the summary values."
    ),
    INTERVALS_TABLE: (
        "Sequences of the Full graph between consecutive landmarks sorted by "
        "fitness (ties keep the landmark table order). Intervals are half-open "
        "[lower, upper) except the last, which is closed [lower, upper] "
        "(upper_closed); a sequence exactly on a landmark falls in the interval "
        "that starts there. interval is '<lower landmark>__<upper landmark>' and "
        "interval_index its rank. percentage is of all Full-graph sequences. The "
        "intervals are half-open while the band of the summary is closed, so "
        "band_count need not equal the sum of the interval counts between the "
        "two trailing saddles. The band and the trunks of the summary and the "
        "entropy table are not a partition: in the overlap case they "
        "intersect."
    ),
    ENTROPY_TABLE: (
        "Shannon entropy in bits of the amino acid at each codon position "
        "(position 1 to 3) of the band population (closed [band_lower, "
        "band_upper], between the two trailing saddles), the split-tree trunk "
        "(fitness at or below the split-tree trailing saddle), the join-tree "
        "trunk (fitness at or above the join-tree trailing saddle) and each "
        "landmark interval (population named as in the intervals table, same "
        "half-open convention). The band and the two trunks are not a "
        "partition: when the join trailing saddle is at or below the split one "
        "(band_kind overlap) the band is the intersection of the trunks, and "
        "when it is above (gap) the band lies between them. entropy is null "
        "for an empty population."
    ),
}
SUMMARY_DESCRIPTION = (
    "Scalars of the leading saddle band of the Full graph, raw fitness. The band "
    "is the closed interval [band_lower, band_upper] between the two trailing "
    "saddles; the split trunk is fitness at or below the split trailing saddle "
    "and the join trunk fitness at or above the join trailing saddle. band_kind "
    "is 'overlap' when the join trailing saddle is at or below the split one "
    "(the band is then the intersection of the two trunks) and 'gap' when it is "
    "above (the band lies between the trunks, apart from its closed endpoints). "
    "The band and the trunks are not a partition: in the overlap case they "
    "intersect. The band is closed and the intervals are half-open, so "
    "band_count need not equal the sum of the interval counts between the two "
    "trailing saddles. median_sequence is set only when exactly one vertex has "
    "raw fitness equal to the median, and null otherwise. n_components, "
    "largest_component_size, largest_component_fraction "
    "(largest_component_size / band_count), n_singletons and "
    "band_internal_edges are the undirected simple subgraph induced by the band "
    "on the Full graph, vertices matched by dna_sequence, self-loops excluded."
)
SCHEMAS = {
    LANDMARKS_TABLE: LANDMARK_SCHEMA,
    INTERVALS_TABLE: INTERVAL_SCHEMA,
    ENTROPY_TABLE: ENTROPY_SCHEMA,
}
INTERVAL_CONVENTION = (
    "half-open [lower, upper) except the last interval, closed [lower, upper]"
)


@dataclass(frozen=True)
class Saddle:
    """A critical point of a merge tree: its DNA sequence identity and raw fitness."""

    sequence: str
    fitness: float


@dataclass(frozen=True)
class TreeSaddles:
    """Leading and trailing saddles of the split and join trees.

    Each is ``None`` when the pair table has no ordinary pair of that tree.
    """

    split_leading: Saddle | None
    split_trailing: Saddle | None
    join_leading: Saddle | None
    join_trailing: Saddle | None


@dataclass(frozen=True)
class LeadingSaddleBand:
    """The three tables and the summary of one stage run."""

    landmarks: pl.DataFrame
    intervals: pl.DataFrame
    entropy: pl.DataFrame
    summary: dict[str, Any]


def _extreme(deaths: pl.DataFrame, *, highest: bool) -> Saddle | None:
    """The saddle with the highest or lowest death scalar, smaller sequence on a tie."""
    if deaths.is_empty():
        return None
    row = deaths.sort(
        ["death_scalar", "death_sequence"], descending=[highest, False]
    ).row(0, named=True)
    return Saddle(str(row["death_sequence"]), float(row["death_scalar"]))


def tree_saddles(pairs: pl.DataFrame) -> TreeSaddles:
    """Read the four saddles from the cached full-network component pairs.

    ``pairs`` is the ascending checkpoint. Its ``relative`` pairs close at split
    saddles (the ordinary pairs of the descending sweep) and its ``ordinary`` pairs
    at join saddles; the ``extended_positive`` global pair is no saddle and is left
    out.
    """
    validate_pair_table(pairs)
    split = pairs.filter(pl.col("pair_kind") == "relative")
    join = pairs.filter(pl.col("pair_kind") == "ordinary")
    return TreeSaddles(
        split_leading=_extreme(split, highest=True),
        split_trailing=_extreme(split, highest=False),
        join_leading=_extreme(join, highest=False),
        join_trailing=_extreme(join, highest=True),
    )


def global_minimum(pairs: pl.DataFrame) -> Saddle:
    """The global minimum vertex: the birth of the one ascending global pair."""
    validate_pair_table(pairs)
    global_pairs = pairs.filter(pl.col("pair_kind") == "extended_positive")
    if global_pairs.height != 1:
        raise ValueError("Component persistence must contain exactly one global pair.")
    row = global_pairs.row(0, named=True)
    return Saddle(str(row["birth_sequence"]), float(row["birth_scalar"]))


def _with_amino_acids(population: pl.DataFrame) -> pl.DataFrame:
    """Add ``aa1`` to ``aa3``, the translated codons of each 9-nt sequence."""
    columns: dict[str, list[str]] = {f"aa{position}": [] for position in POSITIONS}
    for sequence in population["dna_sequence"].to_list():
        if len(sequence) != 9:
            raise ValueError(
                f"Entropy needs 9-nt sequences; got {sequence!r} ({len(sequence)} nt)."
            )
        for position in POSITIONS:
            codon = sequence[(position - 1) * 3 : position * 3]
            if codon not in CODON_TRANSLATION:
                raise ValueError(f"Unsupported codon {codon!r} in {sequence!r}.")
            columns[f"aa{position}"].append(CODON_TRANSLATION[codon])
    return population.with_columns(
        [pl.Series(name, values, dtype=pl.String) for name, values in columns.items()]
    )


def _validate_vertices(
    vertices: Mapping[str, Saddle | None], population: pl.DataFrame
) -> None:
    """Refuse a vertex that is not a Full-graph sequence with the stated fitness."""
    lookup = dict(
        zip(
            population["dna_sequence"].to_list(),
            population["raw_fitness"].to_list(),
            strict=True,
        )
    )
    for name, vertex in vertices.items():
        if vertex.sequence not in lookup:
            raise ValueError(
                f"The {name} sequence {vertex.sequence!r} is not in the Full-graph "
                "fitness population."
            )
        if lookup[vertex.sequence] != vertex.fitness:
            raise ValueError(
                f"The {name} sequence {vertex.sequence!r} has Full-graph raw fitness "
                f"{lookup[vertex.sequence]!r}, not its pair scalar {vertex.fitness!r}."
            )


def _landmark_rows(
    pairs: pl.DataFrame,
    saddles: TreeSaddles,
    population: pl.DataFrame,
    published_threshold: float,
    candidate_declaration: Mapping[str, Any],
    candidate: CandidateThreshold | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    fitness = population["raw_fitness"]
    minimum = global_minimum(pairs)
    if minimum.fitness != float(fitness.min()):
        raise ValueError("The global pair must start at the Full-graph minimum.")
    if candidate is None:
        candidate = compute_candidate_functional_threshold(
            fitness.to_list(), candidate_declaration
        )
    missing = [
        name
        for name, saddle in (
            ("split_leading", saddles.split_leading),
            ("split_trailing", saddles.split_trailing),
            ("join_leading", saddles.join_leading),
            ("join_trailing", saddles.join_trailing),
        )
        if saddle is None
    ]
    if missing:
        raise ValueError(
            "The component pairs have no ordinary pair for: " + ", ".join(missing) + "."
        )
    _validate_vertices(
        {
            "global minimum": minimum,
            "split leading saddle": saddles.split_leading,
            "split trailing saddle": saddles.split_trailing,
            "join leading saddle": saddles.join_leading,
            "join trailing saddle": saddles.join_trailing,
        },
        population,
    )
    diagnostics = build_threshold_diagnostics(pairs)
    if saddles.join_trailing.fitness != diagnostics["join_tree_trailing_saddle"]:
        raise ValueError(
            "The join-tree trailing saddle differs from the threshold diagnostics: "
            f"{saddles.join_trailing.fitness!r} against "
            f"{diagnostics['join_tree_trailing_saddle']!r}."
        )
    pair_source = "component_pairs"
    # The median is a vertex only when exactly one sequence has that fitness.
    at_median = population.filter(pl.col("raw_fitness") == candidate.median)
    median_sequence = (
        str(at_median["dna_sequence"][0]) if at_median.height == 1 else None
    )
    rows = [
        (
            "global_minimum",
            minimum,
            f"{pair_source}: birth of the extended_positive pair",
        ),
        (
            SPLIT_TRAILING,
            saddles.split_trailing,
            f"{pair_source}: lowest death of the relative (split saddle) pairs",
        ),
        (
            JOIN_TRAILING,
            saddles.join_trailing,
            f"{pair_source}: highest death of the ordinary (join saddle) pairs",
        ),
        (
            "median",
            Saddle("", candidate.median),
            (
                "median raw fitness of the Full graph; sequence set only when "
                "exactly one vertex has that fitness"
            ),
        ),
        (
            "split_leading_saddle",
            saddles.split_leading,
            f"{pair_source}: highest death of the relative (split saddle) pairs",
        ),
        (
            "join_leading_saddle",
            saddles.join_leading,
            f"{pair_source}: lowest death of the ordinary (join saddle) pairs",
        ),
        (
            "published_functional_threshold",
            Saddle("", float(published_threshold)),
            "config [clamped] published_functional_threshold",
        ),
        (
            "candidate_functional_threshold",
            Saddle("", candidate.threshold),
            (
                f"candidate_threshold, rule {candidate.rule}, "
                f"rounded to {candidate.decimals} decimals"
            ),
        ),
    ]
    vertices = {
        "global_minimum",
        SPLIT_TRAILING,
        JOIN_TRAILING,
        "split_leading_saddle",
        "join_leading_saddle",
    }
    return [
        {
            "landmark": key,
            "fitness": saddle.fitness,
            "sequence": (
                median_sequence
                if key == "median"
                else saddle.sequence
                if key in vertices
                else None
            ),
            "source": source,
        }
        for key, saddle, source in rows
    ], candidate


def _interval_mask(
    values: np.ndarray, low: float, high: float, *, closed: bool
) -> np.ndarray:
    """Membership in ``[low, high)``, or ``[low, high]`` when ``closed``."""
    return (values >= low) & ((values <= high) if closed else (values < high))


def _interval_rows(
    landmarks: pl.DataFrame, fitness: np.ndarray
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """The interval table and each interval's membership mask, keyed by name."""
    order = sorted(range(landmarks.height), key=lambda i: (landmarks["fitness"][i], i))
    total = fitness.size
    rows = []
    masks = {}
    for index in range(len(order) - 1):
        lower, upper = order[index], order[index + 1]
        low, high = landmarks["fitness"][lower], landmarks["fitness"][upper]
        closed = index == len(order) - 2
        name = (
            f"{landmarks['landmark'][lower]}{INTERVAL_SEPARATOR}"
            f"{landmarks['landmark'][upper]}"
        )
        mask = _interval_mask(fitness, low, high, closed=closed)
        masks[name] = mask
        count = int(mask.sum())
        rows.append(
            {
                "interval_index": index,
                "interval": name,
                "lower_landmark": landmarks["landmark"][lower],
                "upper_landmark": landmarks["landmark"][upper],
                "lower_fitness": low,
                "upper_fitness": high,
                "upper_closed": closed,
                "sequence_count": count,
                "percentage": 100.0 * count / total,
            }
        )
    return pl.DataFrame(rows, schema=INTERVAL_SCHEMA), masks


def band_induced_connectivity(
    band_sequences: set[str], graph: Any
) -> dict[str, int | float]:
    """Connectivity of the undirected simple subgraph induced by the band.

    ``graph`` is the rustworkx ``PyGraph`` from
    ``Papkou2023Workflow(...).build_network().graph``. A vertex is in the band
    when its ``dna_sequence`` is in ``band_sequences`` (the closed
    ``[band_lower, band_upper]`` mask). Self-loops are dropped and parallel edges
    count once. ``largest_component_fraction`` divides by ``len(band_sequences)``,
    which is ``band_count``.
    """
    inband: dict[int, int] = {}
    seen: set[str] = set()
    for index in graph.node_indices():
        sequence = str(graph[index]["dna_sequence"])
        if sequence not in band_sequences:
            continue
        if sequence in seen:
            raise ValueError(f"The Full graph repeats band sequence {sequence!r}.")
        seen.add(sequence)
        inband[index] = 0
    missing = band_sequences - seen
    if missing:
        sample = ", ".join(repr(sequence) for sequence in sorted(missing)[:3])
        raise ValueError(
            f"{len(missing)} band sequence(s) are missing from the Full graph"
            f" ({sample})."
        )
    induced = rustworkx.PyGraph()
    for index in inband:
        inband[index] = induced.add_node(None)
    internal: set[tuple[int, int]] = set()
    for edge in graph.edge_list():
        left, right = int(edge[0]), int(edge[1])
        if left == right or left not in inband or right not in inband:
            continue
        key = (left, right) if left < right else (right, left)
        if key in internal:
            continue
        internal.add(key)
        induced.add_edge(inband[left], inband[right], None)
    sizes = sorted(
        (len(component) for component in rustworkx.connected_components(induced)),
        reverse=True,
    )
    largest = sizes[0] if sizes else 0
    band_count = len(band_sequences)
    return {
        "n_components": len(sizes),
        "largest_component_size": largest,
        "largest_component_fraction": largest / band_count if band_count else 0.0,
        "n_singletons": sum(size == 1 for size in sizes),
        "band_internal_edges": len(internal),
    }


def compute_leading_saddle_band(
    pairs: pl.DataFrame,
    full_fitness: pl.DataFrame,
    *,
    graph: Any,
    published_threshold: float,
    candidate_declaration: Mapping[str, Any],
    candidate: CandidateThreshold | None = None,
) -> LeadingSaddleBand:
    """Compute the landmarks, intervals, entropy and summary of the Full graph.

    ``pairs`` is the cached ascending component-pair table, ``full_fitness`` the
    Full-graph population (``dna_sequence``, ``raw_fitness``), ``graph`` the Full
    graph (a rustworkx ``PyGraph`` whose payloads carry ``dna_sequence``),
    ``published_threshold`` the ``[clamped] published_functional_threshold`` and
    ``candidate_declaration`` the ``[clamped] candidate_functional_threshold``
    table. ``candidate`` is the resolved Candidate functional threshold
    (``resolve_candidate_selection``); omitted, it is computed from the
    declaration, which only the ``median_reflected`` rule allows. Nothing is
    read or written.
    """
    fitness = full_fitness["raw_fitness"]
    if full_fitness.is_empty() or not fitness.is_finite().all():
        raise ValueError("The Full-graph population must be non-empty and finite.")
    saddles = tree_saddles(pairs)
    rows, candidate = _landmark_rows(
        pairs,
        saddles,
        full_fitness,
        published_threshold,
        candidate_declaration,
        candidate,
    )
    by_key = {row["landmark"]: row for row in rows}
    split_value = by_key[SPLIT_TRAILING]["fitness"]
    join_value = by_key[JOIN_TRAILING]["fitness"]
    lower_key = SPLIT_TRAILING if split_value <= join_value else JOIN_TRAILING
    band_lower, band_upper = sorted((split_value, join_value))
    band_kind = BAND_OVERLAP if join_value <= split_value else BAND_GAP
    median = candidate.median
    median_inside_band = bool(band_lower <= median <= band_upper)
    landmarks = pl.DataFrame(
        [
            {**row, "band_kind": band_kind, "median_inside_band": median_inside_band}
            for row in rows
        ],
        schema=LANDMARK_SCHEMA,
    )

    values = fitness.to_numpy()
    total = int(values.size)
    intervals, interval_masks = _interval_rows(landmarks, values)
    first = float(landmarks["fitness"].min())
    last = float(landmarks["fitness"].max())
    below_first = int((values < first).sum())
    above_last = int((values > last).sum())
    if int(intervals["sequence_count"].sum()) + below_first + above_last != total:
        raise ValueError("The intervals must cover every Full-graph sequence once.")

    masks = {
        "band": (values >= band_lower) & (values <= band_upper),
        "split_trunk": values <= split_value,
        "join_trunk": values >= join_value,
    }
    population = _with_amino_acids(full_fitness)
    entropy_rows = []
    summary: dict[str, Any] = {}
    interval_entropy_sum: dict[str, float | None] = {}
    for name, mask in {**masks, **interval_masks}.items():
        members = population.filter(pl.Series(mask))
        count = members.height
        entropies = positional_entropy(members, positions=POSITIONS)
        for position in POSITIONS:
            entropy_rows.append(
                {
                    "population": name,
                    "position": position,
                    "entropy": entropies[position],
                    "sequence_count": count,
                }
            )
        entropy_sum = None if not count else float(sum(entropies.values()))
        if name in interval_masks:
            interval_entropy_sum[name] = entropy_sum
            continue
        summary[f"{name}_count"] = count
        summary[f"{name}_percentage"] = 100.0 * count / total
        summary[f"{name}_entropy_sum"] = entropy_sum
    entropy = pl.DataFrame(entropy_rows, schema=ENTROPY_SCHEMA)
    band_sequences = set(
        population.filter(pl.Series(masks["band"]))["dna_sequence"].to_list()
    )
    summary.update(band_induced_connectivity(band_sequences, graph))

    for row in rows:
        summary[f"{row['landmark']}_fitness"] = row["fitness"]
        summary[f"{row['landmark']}_sequence"] = row["sequence"]
    summary.update(
        {
            "schema": SUMMARY_SCHEMA,
            "description": SUMMARY_DESCRIPTION,
            "total_sequence_count": total,
            "band_lower_landmark": lower_key,
            "band_kind": band_kind,
            "band_lower": band_lower,
            "band_upper": band_upper,
            "median_inside_band": median_inside_band,
            "below_first_landmark_count": below_first,
            "above_last_landmark_count": above_last,
            "interval_convention": INTERVAL_CONVENTION,
            "interval_order": intervals["interval"].to_list(),
            "interval_counts": dict(
                zip(intervals["interval"], intervals["sequence_count"], strict=True)
            ),
            "interval_percentages": dict(
                zip(intervals["interval"], intervals["percentage"], strict=True)
            ),
            "interval_entropy_sums": interval_entropy_sum,
            "median_gap": candidate.median_gap,
            "candidate_unrounded_threshold": candidate.unrounded,
            "candidate_threshold_rule": candidate.rule,
            "candidate_threshold_decimals": candidate.decimals,
        }
    )
    return LeadingSaddleBand(landmarks, intervals, entropy, summary)


def _schema_record(name: str, frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "description": DESCRIPTIONS[name],
        "columns": {column: str(dtype) for column, dtype in frame.schema.items()},
    }


def write_leading_saddle_band(directory: Path, result: LeadingSaddleBand) -> list[Path]:
    """Write each table as ``.csv``, ``.parquet`` and ``.schema.json``, and ``summary.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in (
        (LANDMARKS_TABLE, result.landmarks),
        (INTERVALS_TABLE, result.intervals),
        (ENTROPY_TABLE, result.entropy),
    ):
        frame.write_csv(directory / f"{name}.csv")
        frame.write_parquet(directory / f"{name}.parquet")
        # Column order is part of the schema, so the keys are not sorted.
        (directory / f"{name}.schema.json").write_text(
            json.dumps(_schema_record(name, frame), indent=2) + "\n", encoding="utf-8"
        )
        written += [
            directory / f"{name}.{suffix}"
            for suffix in ("csv", "parquet", "schema.json")
        ]
    _write_json_atomic(directory / "summary.json", result.summary)
    return [*written, directory / "summary.json"]


def read_leading_saddle_band(directory: Path) -> LeadingSaddleBand:
    """Read a stage directory written by :func:`write_leading_saddle_band`."""
    return LeadingSaddleBand(
        landmarks=pl.read_parquet(directory / f"{LANDMARKS_TABLE}.parquet"),
        intervals=pl.read_parquet(directory / f"{INTERVALS_TABLE}.parquet"),
        entropy=pl.read_parquet(directory / f"{ENTROPY_TABLE}.parquet"),
        summary=json.loads((directory / "summary.json").read_text(encoding="utf-8")),
    )


def describe(result: LeadingSaddleBand) -> str:
    """One line for the command line: band size and the median test."""
    info = result.summary
    return (
        f"Leading saddle band: {info['band_count']} of "
        f"{info['total_sequence_count']} sequences between "
        f"{info['band_lower']:.6f} and {info['band_upper']:.6f}; "
        f"median inside band: {info['median_inside_band']}."
    )
