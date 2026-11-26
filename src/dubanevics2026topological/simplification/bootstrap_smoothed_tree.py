"""Fixed-graph bootstrap and direct nominal-to-band simplification.

Each draw perturbs raw fitness using pooled standard errors and measures
distances to the fixed nominal persistence diagram. A flat-landscape series
provides descriptive checks. Peak survival is determined directly from nominal
ordinary H0 persistence, strictly above twice the real-draw band radius.

The global verdict does not compare real with flat distances (the flat draws
carry tens of thousands of noise peaks, so a sum over diagram points is not like
for like). It uses the bootstrap band construction of Fasy et al. 2014: ``c`` is
the ``band_quantile`` quantile of the REAL draws' bottleneck distance to the
nominal diagram, and a nominal point survives the declared plug-in perturbation
band when its persistence is strictly greater than ``2c``. This provides neither
calibrated coverage nor a peak-level FDR (ADR 0008 supersession note,
2026-10-31). The nominal diagram is the ordinary H0 diagram of the
nominal tree with the global maximum's pair closed at the root (finite births
and deaths), the very diagram Topographer measures every draw against.

The one comparison with the flat landscape that survives is a sanity check, not
the verdict: the bottleneck distance of the real draws and of the flat draws,
both to the REAL nominal diagram (``global_verdict`` ``sanity_check``). Real
draws should sit far below the flat ones; the share of real draws under the flat
5th percentile says by how much. Wasserstein-1 is never compared across
conditions.

Identities are ``dna_sequence`` values throughout: Rustworkx node indices from
Topographer are mapped through the network's vertex order and never exported.
``vertex_index`` is used only internally and for the elder tie-break.

The draws come from Topographer's chunked bootstrap (``bootstrap.run`` with a
``LeafCooccurrence`` statistic), run by ``bootstrap_chunks.ChunkedBootstrap``:
chunks of ``chunk_size`` draws are saved as they finish and a rerun resumes from
them. The core-free aggregation adapter preserves the underlying measurements
and yields draw distances and nominal hill support. The series are called
``observed`` and ``flat``.

Publication goes through ``produce_or_load`` (ADR 0013, Phase 2). The stage key
is the run's scientific identity (``bootstrap_stage``), so the published
checkpoint ``output/checkpoints/bootstrap_smoothed_tree`` and the processed
``data/processed/bootstrap/<state>`` stay reusable across code edits. The working slot
``output/analysis/bootstrap_smoothed_tree`` keeps the layout that readers and the
processed checkpoint use; it is a byte-for-byte export of the checkpoint, written under
the checkpoint lock with its ``provenance.json`` last. The helper stays generic: chunk
resume, ``--force`` and ``--fresh`` live here, around the call.

``n_jobs > 1`` starts Topographer's workers with the ``spawn`` method, so every
script entry point that reaches this stage must sit behind an
``if __name__ == "__main__":`` guard (``python -m dubanevics2026topological.build``
does).

``$FITNESSTREE_N_JOBS`` (a positive integer) replaces the configured ``n_jobs``
when set (``n_jobs_override``), so a cluster job sizes its workers to its
allocation without editing ``config/analysis.toml``. Like ``n_jobs`` it schedules
workers only: result settings, keys and fingerprints are unchanged, and
``summary.json`` records the effective count under ``timing``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import topographer
from topographer.graph import Graph, ScalarNetwork
from dubanevics2026topological.simplification.cooccurrence_leaf import (
    BootstrapResult,
    LeafCooccurrence,
)
from dubanevics2026topological.simplification.cooccurrence_bootstrap import (
    terminal_branch_members,
)
from topographer.topology import ScalarNetworkTopology

from ..analysis import (
    ROOT,
    dataset_input_key,
    dataset_input_path,
    get_run_context,
    load_settings,
    package_source_hashes,
    source_key,
)
from ..analysis.barseq_inputs import _quoted
from ..analysis.candidate_tree_contrast import (
    _sha256,
    build_candidate_network,
    enumerate_tree_edges,
)
from ..analysis.checkpoint import checkpoint_lock
from ..analysis.produce_or_load import (
    PROVENANCE_NAME,
    Checkpoint,
    Stage,
    produce_or_load,
    publication_state,
    result_directory,
)
from ..analysis.sequence_growth_rates import (
    SequenceGrowthRatesResult,
    load_cached_sequence_growth_rates,
)
from ..analysis.sequence_growth_rates import (
    _fingerprint_without_export as _growth_fingerprint_without_export,
)
from ..landscape.selected_landscape import analyze_prepared_landscape
from ..landscape.selection import Selection
from ..landscape.selection import (
    selected_graph_identity as candidate_selection_identity,
)
from ..provenance import without_git
from ..threshold.functional_threshold import (
    configured_declaration,
    declaration_identity,
)
from . import bootstrap_chunks, checkpoint_identity
from .bootstrap_aggregation import BootstrapDistances
from .selected_leaf_simplification import (
    SelectedLeafView,
    absorb_terminal_branches,
    enforce_elder_rule,
)

SECTION = "bootstrap_smoothed_tree"
ENV_N_JOBS = "FITNESSTREE_N_JOBS"
STAGE = "bootstrap-smoothed-tree"
LABEL = "Bootstrap smoothed tree"
# The published checkpoint (ADR 0013): ``produce_or_load`` keeps the one verified
# copy here, keyed by the run's scientific identity.
# It lives under the run context's output root (``checkpoint_dir``).
# The working slot (``cache_dir``): the layout readers, the mirror and the figure
# build use. It is a byte-for-byte export of the checkpoint, rewritten under its lock.
# ``provenance.json`` is reserved for the helper's own record at the top level of
# a checkpoint, so the stage's record (the same bytes as the slot's
# ``provenance.json``) travels in the payload under another name.
SLOT_RECORD = "provenance.json"
RECORD_NAME = "bootstrap_provenance.json"
RESTORE_MARKER = ".restored_from.json"
CACHE_SCHEMA = 7
# Step 2 of the pipeline (the confidence band) only keeps or absorbs a hill.
BAND_OUTCOMES = ("kept", "absorbed")
# A surviving pair of the band view must keep its nominal persistence to this
# absolute tolerance (the same float operands give the same difference).
BAND_PERSISTENCE_TOLERANCE = 1e-9
CONDITIONS = ("real", "flat")
DEFAULT_CHUNK_SIZE = 250
ARTIFACTS = (
    "hills.parquet",
    "draw_stats.parquet",
    "summary.json",
)
PAIR_COLUMNS = ("sequence_a", "sequence_b", "frequency")
CORE_COLUMNS = ("core_id", "dna_sequence", "raw_fitness", "nominal_hill", "is_orphan")
HILL_COLUMNS = (
    "peak_sequence",
    "member_count",
    "hill_support",
    "nominal_birth",
    "nominal_death",
    "nominal_persistence",
    "clears_band",
    "band_outcome",
)
DIAGRAM_COLUMNS = ("peak_sequence", "birth", "death", "persistence")
# numpy's default quantile estimator, stated in the summary.
QUANTILE_METHOD = "linear"
ROLE_COLUMNS = (
    "peak_frequency",
    "saddle_frequency",
    "regular_frequency",
    "root_frequency",
    "terminal_frequency",
)
DRAW_COLUMNS = (
    "draw",
    "condition",
    "peak_count",
    "wasserstein_1",
    "bottleneck",
    "reproduced_pair_fraction",
)
DISTANCE_STATISTICS = ("wasserstein_1", "bottleneck", "peak_count")
# What ``draw_stats`` measures, as Topographer v0.5.0 defines it.
DISTANCE_CONVENTIONS = {
    "wasserstein_1": (
        "W1 between ordinary H0 persistence diagrams (the global maximum's pair "
        "closed at the root, so every pair has a finite birth and death) with the "
        "L1 ground metric; a point matched to the diagonal costs its persistence."
    ),
    "bottleneck": (
        "Bottleneck distance between ordinary H0 persistence diagrams with the "
        "L-infinity ground metric; a point matched to the diagonal costs half "
        "its persistence."
    ),
    "reproduced_pair_fraction": (
        "Share of nominal ordinary H0 (peak, death vertex) pairs found identically "
        "in the draw; the denominator counts the global maximum's (peak, root) "
        "pair together with the ordinary pairs, so it equals the number of "
        "nominal hills (one pair per peak)."
    ),
}
# Every vertex of the flat-landscape rerun is drawn around this one constant.
# Only the real standard errors matter, so the value is arbitrary; 0 is the
# wild-type level of the wild-type-relative scale.
FLAT_LANDSCAPE_CONSTANT = 0.0
# Which standard error drives the draws (ADR 2026-10-22-pooled-standard-error-for-
# bootstrap). Not a setting: it is recorded in the fingerprint so that a pooled run
# can never match, restore or overwrite a run made with the replicate standard error.
NOISE_MODEL = "pooled"
# Settings that decide the result. ``n_jobs`` is not among them: Topographer
# derives every draw's generator from the seed and the draw index, so every
# ``n_jobs`` gives the same draws; neither is ``chunk_size``, which only cuts the
# draws into resumable pieces. ``consensus_peak_frequency``: a vertex is a
# consensus peak when it is a peak in at least this share of draws (a diagnostic,
# but part of the checkpoint fingerprint).
RESULT_SETTINGS = (
    "n_replicates",
    "seed",
    "band_quantile",
)


class StaleCheckpointError(ValueError):
    """A checkpoint whose settings match but whose inputs differ from the configuration."""


@dataclass(slots=True)
class BootstrapSmoothedTreeResult:
    """Direct band hill evidence, draw statistics, summary and provenance."""

    hills: pl.DataFrame
    draw_stats: pl.DataFrame
    summary: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NominalHill:
    """One nominal terminal branch, by sequence identity.

    ``members`` are the peak first, then the regular vertices on the branch.
    ``peak_vertex_index`` is the peak's network node index, used only to break
    exact height ties (smaller wins, Topographer's elder convention).
    """

    peak_sequence: str
    peak_vertex_index: int
    peak_raw_fitness: float
    members: tuple[str, ...]


def _load_settings(context: Any | None = None) -> dict[str, Any]:
    """The settings of ``context``; without one, the environment's, as before."""
    return load_settings() if context is None else load_settings(context=context)


def _ctx(context: Any | None) -> dict[str, Any]:
    """``{"context": context}``, or nothing: a context-less call stays as it was."""
    return {} if context is None else {"context": context}


def _output_root(context: Any | None) -> Path:
    """The output root of ``context``; a context-less call resolves a fresh one now."""
    return Path((context or get_run_context()).output_root)


def step_dir(context: Any | None = None) -> Path:
    """The step directory that holds one folder per bootstrap key."""
    return _output_root(context) / "checkpoints" / "bootstrap_smoothed_tree"


def checkpoint_dir(context: Any | None = None) -> Path:
    """The folder of the current bootstrap key under ``step_dir``.

    Callers that replace this with a fixed path publish into that path.
    """

    step = step_dir(context)
    try:
        settings = _load_settings(context)
        stage = bootstrap_stage(
            _wanted_scientific(settings, warn=False, context=context)
        )
        return result_directory(step, stage)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return step


def _published_leaf(context: Any | None = None) -> Path:
    """The folder of the current bootstrap key."""
    return checkpoint_dir(context)


def cache_dir(context: Any | None = None) -> Path:
    """The working slot under the context's output root."""
    return _output_root(context) / "analysis" / "bootstrap_smoothed_tree"


def _processed_root(context: Any | None) -> Path:
    """The processed-data folder whose ``bootstrap/<state>/checkpoint`` folders a run restores from."""
    from ..results import processed

    return processed.processed_root(context)


def _chunk_base(context: Any | None) -> Path | None:
    """The context's chunk directory; ``None`` keeps the store's own default.

    Without one the store's environment lookup (``XDG_DATA_HOME`` included) applies,
    as it did before the stage took a context.
    """
    if context is None or getattr(context, "bootstrap_chunk_dir", None) is None:
        return None
    return bootstrap_chunks.chunk_base(context=context)


def n_jobs_override(
    environ: Mapping[str, str] | None = None,
    *,
    context: Any | None = None,
) -> int | None:
    """The worker count ``$FITNESSTREE_N_JOBS`` or RunContext asks for, else ``None``.

    An unset or blank variable asks for nothing. Any other value must be a
    positive integer in plain ASCII digits, or ``ValueError`` names the variable.
    """
    if context is not None:
        value = getattr(context, "worker_concurrency", None)
        if value is None:
            return None
        from ..results.context import positive_ascii_integer

        return positive_ascii_integer(value)
    text = (os.environ if environ is None else environ).get(ENV_N_JOBS, "").strip()
    if not text:
        return None
    from ..results.context import positive_ascii_integer

    return positive_ascii_integer(text)


def _validate_config(
    config: dict[str, Any], context: Any | None = None
) -> dict[str, Any]:
    known = {*RESULT_SETTINGS, "n_jobs", "chunk_size"}
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            f"Unknown [{SECTION}] settings {unknown}; known: {sorted(known)}."
        )

    def whole(name: str) -> int:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        return value

    def open_share(name: str) -> float:
        value = config[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(f"{name} must be finite and in (0, 1).")
        return float(value)

    n_replicates, n_jobs, seed = (
        whole("n_replicates"),
        whole("n_jobs"),
        whole("seed"),
    )
    chunk_size = whole("chunk_size") if "chunk_size" in config else DEFAULT_CHUNK_SIZE
    if n_replicates < 1 or n_jobs < 1 or chunk_size < 1:
        raise ValueError(
            "n_replicates, n_jobs and chunk_size must be positive integers."
        )
    if seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    # The one place the override is read: every reader of the worker count, the
    # draws and the summary alike, goes through these validated settings.
    n_jobs = n_jobs_override(context=context) or n_jobs
    return {
        "n_replicates": n_replicates,
        "seed": seed,
        "band_quantile": open_share("band_quantile"),
        "n_jobs": n_jobs,
        "chunk_size": chunk_size,
    }


def result_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return the validated settings that decide the result (everything but ``n_jobs``)."""
    settings = _validate_config(config)
    return {name: settings[name] for name in RESULT_SETTINGS}


def _config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Full checkpoint settings plus the constant noise-model marker.

    ``checkpoint_identity.scientific_fingerprint`` projects this checkpoint identity
    to draw identity by removing the derived band quantile.
    """
    return {**result_settings(config), "noise_model": NOISE_MODEL}


def _prepared_tables(network: ScalarNetwork) -> tuple[list[str], pl.DataFrame]:
    """Return the network's identities in node order and its edge list by identity."""
    graph = network.graph
    ids = [str(graph[i]["dna_sequence"]) for i in graph.node_indices()]
    if ids != [str(vertex_id) for vertex_id in network.vertex_ids]:
        raise ValueError("Network vertex identities are not in node-index order.")
    edges = pl.DataFrame(
        {
            "source": [ids[int(a)] for a, _ in graph.edge_list()],
            "target": [ids[int(b)] for _, b in graph.edge_list()],
        },
        schema={"source": pl.String, "target": pl.String},
    )
    return ids, edges


def _make_network(
    ids: list[str], edges: pl.DataFrame, raw_fitness: np.ndarray, se: np.ndarray
) -> ScalarNetwork:
    """Build a ScalarNetwork with ``raw_fitness`` and ``se`` vertex attributes."""
    vertices = pl.DataFrame(
        {"dna_sequence": ids, "raw_fitness": raw_fitness, "se": se},
        schema={
            "dna_sequence": pl.String,
            "raw_fitness": pl.Float64,
            "se": pl.Float64,
        },
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("raw_fitness", "se"),
    )
    network = ScalarNetwork(graph, scalar_attr="raw_fitness")
    if [str(v) for v in network.vertex_ids] != ids:
        raise ValueError("The bootstrap network reordered the Candidate vertices.")
    return network


def _pooled_se(
    ids: list[str], growth: SequenceGrowthRatesResult, raw_fitness: np.ndarray
) -> np.ndarray:
    """Return ``se_pooled`` per vertex, checking the checkpoint fits this network."""
    rates = growth.growth_rates
    missing = sorted(set(ids) - set(rates["dna_sequence"]))
    if missing:
        raise ValueError(
            "Sequences are absent from the growth-rate checkpoint (rerun "
            "`just sequence-growth-rates`): " + _quoted(missing)
        )
    r_of = dict(zip(rates["dna_sequence"], rates["r"], strict=True))
    se_of = dict(zip(rates["dna_sequence"], rates["se_pooled"], strict=True))
    se = np.array([np.nan if se_of[d] is None else se_of[d] for d in ids], dtype=float)
    r = np.array([r_of[d] for d in ids], dtype=float)
    bad = ~(np.isfinite(se) & (se > 0))
    if bad.any():
        raise ValueError(
            "Pooled standard errors are missing, non-finite or not positive for "
            + _quoted([ids[i] for i in np.flatnonzero(bad)])
        )
    tolerance = float(growth.diagnostics["identity_tolerance"])
    gap = np.abs((r - float(growth.diagnostics["wild_type_r"])) - raw_fitness)
    stale = ~(gap <= tolerance)
    if stale.any():
        raise ValueError(
            "The growth-rate checkpoint disagrees with the Candidate network's "
            f"raw fitness by more than {tolerance:g} (rerun "
            "`just sequence-growth-rates`): "
            + _quoted([ids[i] for i in np.flatnonzero(stale)])
        )
    return se


def _edge_pairs(graph: Any, node_dna: dict[int, str]) -> set[tuple[str, str]]:
    return {(node_dna[int(a)], node_dna[int(b)]) for a, b in graph.edge_list()}


def check_identical_tree(network: ScalarNetwork, tree_analysis: Any) -> Any:
    """Return the network's split tree after proving it equals the nominal tree.

    Every critical node must carry the same ``dna_sequence``, ``vertex_index``,
    height and event, and the edge sets must agree. This guards the step that
    rebuilds the network from tables. The returned tree is the bootstrap's
    reference, so the split tree is built only once.
    """
    reference = ScalarNetworkTopology(network).split_tree()
    rebuilt = reference.graph  # root -> leaf
    nominal = tree_analysis.tree  # leaf -> root
    rebuilt_nodes = {
        str(rebuilt[i]["vertex_id"]): (
            int(rebuilt[i]["vertex_index"]),
            float(rebuilt[i]["scalar"]),
            str(rebuilt[i]["event"]),
        )
        for i in rebuilt.node_indices()
    }
    nominal_nodes = {
        str(nominal[i]["dna_sequence"]): (
            int(nominal[i]["vertex_index"]),
            float(nominal[i]["scalar"]),
            str(nominal[i]["event"]),
        )
        for i in nominal.node_indices()
    }
    problems: list[str] = []
    if rebuilt_nodes.keys() != nominal_nodes.keys():
        extra = sorted(rebuilt_nodes.keys() - nominal_nodes.keys())
        absent = sorted(nominal_nodes.keys() - rebuilt_nodes.keys())
        problems.append(
            f"critical sequences differ (rebuilt only: {_quoted(extra)}; "
            f"nominal only: {_quoted(absent)})"
        )
    else:
        differing = sorted(
            d for d in nominal_nodes if rebuilt_nodes[d] != nominal_nodes[d]
        )
        if differing:
            problems.append(
                "vertex index, height or event differ at " + _quoted(differing)
            )
    rebuilt_dna = {int(i): str(rebuilt[i]["vertex_id"]) for i in rebuilt.node_indices()}
    nominal_dna = {
        int(i): str(nominal[i]["dna_sequence"]) for i in nominal.node_indices()
    }
    # The rebuilt tree points root -> leaf, the nominal tree leaf -> root.
    rebuilt_edges = {(b, a) for a, b in _edge_pairs(rebuilt, rebuilt_dna)}
    nominal_edges = _edge_pairs(nominal, nominal_dna)
    if rebuilt_edges != nominal_edges:
        problems.append(
            f"edge sets differ ({len(rebuilt_edges - nominal_edges)} rebuilt only, "
            f"{len(nominal_edges - rebuilt_edges)} nominal only)"
        )
    if problems:
        raise ValueError(
            "The network rebuilt for the bootstrap does not give the nominal "
            "Candidate split tree: " + "; ".join(problems) + "."
        )
    return reference


def nominal_hills(tree_analysis: Any) -> list[NominalHill]:
    """Return every terminal branch of the nominal tree as a hill, by identity.

    A terminal edge has a maximum with in-degree 0 as its upper node; its hill
    is that peak plus the ``branch_member_table`` rows of the edge (the lower
    endpoint is not a member). ``peak_vertex_index`` is the node payload's
    ``vertex_index``, as in the elder rule. Hills are sorted by peak sequence.
    """
    graph = tree_analysis.tree
    regular: dict[tuple[int, int], list[str]] = {}
    table = tree_analysis.branch_member_table
    if table.height:
        for source, target, dna in table.select(
            "source_tree_node_index", "target_tree_node_index", "dna_sequence"
        ).iter_rows():
            regular.setdefault((int(source), int(target)), []).append(str(dna))
    hills = []
    for node in graph.node_indices():
        payload = graph[node]
        if graph.in_degree(node) or payload["event"] != "maximum":
            continue
        (lower,) = graph.successor_indices(node)
        peak = str(payload["dna_sequence"])
        hills.append(
            NominalHill(
                peak_sequence=peak,
                peak_vertex_index=int(payload["vertex_index"]),
                peak_raw_fitness=float(payload["scalar"]),
                members=(peak, *regular.get((int(node), int(lower)), [])),
            )
        )
    return sorted(hills, key=lambda hill: hill.peak_sequence)


def hill_vertex_indices(
    hills: Sequence[NominalHill], ids: Sequence[str]
) -> dict[str, list[int]]:
    """Return each hill's member network node indices, keyed by peak sequence."""
    index_of = {dna: i for i, dna in enumerate(ids)}
    return {h.peak_sequence: [index_of[d] for d in h.members] for h in hills}


def _check_disjoint(cores: Mapping[int, frozenset[str]]) -> None:
    """Require every sequence to be in at most one core."""
    owner: dict[str, int] = {}
    for core_id, members in cores.items():
        for dna in sorted(members):
            if owner.setdefault(dna, core_id) != core_id:
                raise ValueError(
                    f"Cores must be disjoint, but {dna!r} is in cores "
                    f"{owner[dna]} and {core_id}."
                )


def _hills_met(
    hills: Sequence[NominalHill], cores: Mapping[int, frozenset[str]]
) -> dict[int, list[str]]:
    """Return, per core, the sorted peaks of the nominal hills it has a member in."""
    hill_of = {dna: h.peak_sequence for h in hills for dna in h.members}
    return {
        core_id: sorted({hill_of[d] for d in members if d in hill_of})
        for core_id, members in cores.items()
    }


def classify_hills(
    hills: Sequence[NominalHill], cores: Mapping[int, frozenset[str]]
) -> pl.DataFrame:
    """Apply the decision table of ADR 0008 to every nominal hill (no elder rule).

    A core *meets* a hill when it has a member in the hill's member set; it
    *spans* when it meets two or more hills. For hill H:

    - no core meets H: ``core_relation = "none"``, nominated;
    - a spanning core meets H: ``"spanning"``. Each spanning core has one winner,
      the hill with the highest nominal peak ``raw_fitness`` among the hills it
      meets (ties: smaller peak ``vertex_index``). H is nominated unless it is
      the winner of every spanning core meeting it. ``surviving_peak_sequence``
      is the highest-peak winner among those cores (H's own peak when it wins);
    - otherwise exactly one core meets H: ``"inside"``, kept (also when the core
      extends onto vertices on no nominal hill);
    - otherwise two or more cores meet H, none spanning: ``"split_candidate"``,
      kept.

    ``split_candidate`` is true for every hill met by two or more cores, so a
    spanning hill that is also met by several cores records both facts. The
    output has ``peak_sequence``, ``core_ids``, ``core_relation``,
    ``split_candidate``, ``surviving_peak_sequence`` and ``nominated``.
    """
    _check_disjoint(cores)
    by_peak = {h.peak_sequence: h for h in hills}

    def rank(peak: str) -> tuple[float, int]:
        hill = by_peak[peak]
        return (hill.peak_raw_fitness, -hill.peak_vertex_index)

    met = _hills_met(hills, cores)
    winner = {
        core_id: max(peaks, key=rank)
        for core_id, peaks in met.items()
        if len(peaks) > 1
    }
    meeting: dict[str, list[int]] = {h.peak_sequence: [] for h in hills}
    for core_id, peaks in met.items():
        for peak in peaks:
            meeting[peak].append(core_id)
    rows = []
    for hill in hills:
        core_ids = sorted(meeting[hill.peak_sequence])
        spanning = [c for c in core_ids if c in winner]
        survivor = None
        if not core_ids:
            relation, nominated = "none", True
        elif spanning:
            relation = "spanning"
            winners = {winner[c] for c in spanning}
            nominated = winners != {hill.peak_sequence}
            survivor = max(winners, key=rank)
        else:
            relation, nominated = (
                ("inside" if len(core_ids) == 1 else "split_candidate"),
                False,
            )
        rows.append(
            {
                "peak_sequence": hill.peak_sequence,
                "core_ids": core_ids,
                "core_relation": relation,
                "split_candidate": len(core_ids) >= 2,
                "surviving_peak_sequence": survivor,
                "nominated": nominated,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "peak_sequence": pl.String,
            "core_ids": pl.List(pl.Int64),
            "core_relation": pl.String,
            "split_candidate": pl.Boolean,
            "surviving_peak_sequence": pl.String,
            "nominated": pl.Boolean,
        },
    )


def decide_hills(
    tree_analysis: Any,
    hills: Sequence[NominalHill],
    cores: Mapping[int, frozenset[str]],
) -> pl.DataFrame:
    """Classify the hills, then give each an ``outcome`` through the elder rule.

    ``nominated`` hills pass through ``enforce_elder_rule``: the protected ones
    are ``elder_protected``, the rest ``absorbed``; every other hill is ``kept``.
    """
    classified = classify_hills(hills, cores)
    nominated = set(classified.filter(pl.col("nominated"))["peak_sequence"])
    protected = enforce_elder_rule(tree_analysis, nominated)
    return classified.with_columns(
        pl.when(~pl.col("nominated"))
        .then(pl.lit("kept"))
        .when(pl.col("peak_sequence").is_in(sorted(protected)))
        .then(pl.lit("elder_protected"))
        .otherwise(pl.lit("absorbed"))
        .alias("outcome")
    ).drop("nominated")


def nominal_diagram(tree_analysis: Any, hills: Sequence[NominalHill]) -> pl.DataFrame:
    """Return the nominal ordinary H0 persistence diagram, one row per point.

    This is the diagram Topographer measures every draw against: the elder-rule
    pairs of the nominal split tree (birth = peak scalar, death = pairing-saddle
    scalar) plus the global maximum's pair closed at the global minimum (the
    root scalar), so every birth and death is finite. It is read from the tree's
    ``persistence_table`` (Topographer's extended pair table; its
    ``pair_kind == "extended"`` row is the global maximum's pair closed at the global minimum, ordinary H0 by this
    project's convention) and checked:

    - one point per nominal hill, i.e. the ordinary pairs (one per non-global
      peak) plus the single global pair, with births exactly the hills' peaks;
    - the global pair starts at the highest peak and ends at the root scalar;
    - ``persistence = birth - death`` is non-negative for every point.

    ``peak_sequence`` is the birth peak's ``dna_sequence`` (the diagram point and
    the hill are the same object); ``birth``, ``death`` and ``persistence`` are in
    fitness units. Sorted by ``peak_sequence``.
    """
    table = tree_analysis.persistence_table
    graph = tree_analysis.tree
    kinds = dict(table.group_by("pair_kind").len().iter_rows())
    if kinds.get("extended", 0) != 1 or set(kinds) - {"ordinary", "extended"}:
        raise ValueError(
            "The nominal persistence table must hold ordinary pairs and exactly one "
            f"global pair (pair_kind 'extended'), found {kinds}."
        )
    peaks = [
        str(graph[int(node)]["dna_sequence"])
        for node in table["birth_tree_node_index"].to_list()
    ]
    diagram = pl.DataFrame(
        {
            "peak_sequence": peaks,
            "birth": table["birth_scalar"],
            "death": table["death_scalar"],
            "is_global": table["pair_kind"] == "extended",
        },
        schema={
            "peak_sequence": pl.String,
            "birth": pl.Float64,
            "death": pl.Float64,
            "is_global": pl.Boolean,
        },
    ).with_columns((pl.col("birth") - pl.col("death")).alias("persistence"))
    hill_peaks = {h.peak_sequence for h in hills}
    if diagram.height != len(hills) or set(peaks) != hill_peaks:
        raise ValueError(
            f"The nominal diagram has {diagram.height} points for {len(hills)} hills "
            "(expected the ordinary pairs plus the global pair, one per peak): "
            + _quoted(sorted(set(peaks) ^ hill_peaks))
        )
    if (
        not diagram["persistence"].is_finite().all()
        or (diagram["persistence"] < 0).any()
    ):
        raise ValueError("Nominal persistence must be finite and non-negative.")
    roots = [
        float(graph[int(i)]["scalar"])
        for i in graph.node_indices()
        if graph.out_degree(i) == 0
    ]
    (global_pair,) = diagram.filter(pl.col("is_global")).iter_rows(named=True)
    if len(roots) != 1 or global_pair["death"] != roots[0]:
        raise ValueError("The global pair must be closed at the root scalar.")
    if global_pair["birth"] != max(h.peak_raw_fitness for h in hills):
        raise ValueError("The global pair must start at the highest nominal peak.")
    return diagram.select(*DIAGRAM_COLUMNS).sort("peak_sequence")


def band_width(draw_stats: pl.DataFrame, quantile: float) -> float:
    """Return ``c``, the ``quantile`` of the REAL draws' bottleneck distance.

    The bottleneck distance of a draw's diagram to the nominal diagram is the
    statistic of Fasy et al. (2014); its ``quantile`` over the real draws is the
    half-width ``c`` of the confidence band (numpy's default linear estimator).
    A nominal point within ``c`` of the diagonal in the bottleneck metric, which
    matches half its persistence to the diagonal, has persistence at most
    ``2c``.
    """
    frame = draw_stats.filter(pl.col("condition") == "real")
    if not frame.height:
        raise ValueError("Draw statistics have no 'real' draws.")
    values = frame["bottleneck"].cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("real bottleneck has non-finite values.")
    return float(np.quantile(values, quantile, method=QUANTILE_METHOD))


def _hill_table(
    hills: Sequence[NominalHill],
    support: Mapping[str, float],
    diagram: pl.DataFrame,
    two_c: float,
) -> pl.DataFrame:
    missing = sorted(
        h.peak_sequence for h in hills if support.get(h.peak_sequence) is None
    )
    if missing:
        raise ValueError("Hill support is missing for the hills of " + _quoted(missing))
    counts = {h.peak_sequence: len(h.members) for h in hills}
    return (
        diagram.rename(
            {
                "birth": "nominal_birth",
                "death": "nominal_death",
                "persistence": "nominal_persistence",
            }
        )
        .with_columns(
            (pl.col("nominal_persistence") > two_c).alias("clears_band"),
            pl.when(pl.col("nominal_persistence") > two_c)
            .then(pl.lit("kept"))
            .otherwise(pl.lit("absorbed"))
            .alias("band_outcome"),
            pl.col("peak_sequence")
            .replace_strict(counts, return_dtype=pl.Int64)
            .alias("member_count"),
            pl.col("peak_sequence")
            .replace_strict(
                {k: float(v) for k, v in support.items()}, return_dtype=pl.Float64
            )
            .alias("hill_support"),
        )
        .select(*HILL_COLUMNS)
    )


def false_hill_rate(flat_hills: pl.DataFrame) -> float | None:
    """Share of nominal hills the flat landscape's cores keep (outcome ``kept``).

    ``flat_hills`` is the checkpoint's ``flat_hills`` table; ``None`` when there
    are no hills. This is the one place the rate is computed, so the summary's
    value can be recomputed from the persisted table.
    """
    if not flat_hills.height:
        return None
    return int((flat_hills["outcome"] == "kept").sum()) / flat_hills.height


def check_hills_match_tree(hills: pl.DataFrame, tree_analysis: Any) -> None:
    """Require the checkpoint's hills to be the nominal tree's terminal branches."""
    edges = enumerate_tree_edges(tree_analysis).filter(pl.col("is_terminal"))
    if set(edges["upper_sequence"]) != set(hills["peak_sequence"]):
        raise ValueError(
            "Bootstrap hills differ from the nominal tree's terminal branches: "
            + _quoted(
                sorted(set(edges["upper_sequence"]) ^ set(hills["peak_sequence"]))
            )
            + "."
        )
    joined = edges.join(
        hills, left_on="upper_sequence", right_on="peak_sequence", how="inner"
    )
    differs = joined.filter(pl.col("regular_count") + 1 != pl.col("member_count"))
    if differs.height:
        raise ValueError(
            "Bootstrap hills disagree with the nominal tree on member count for "
            + _quoted(sorted(differs["upper_sequence"]))
        )


def exact_ties(values: Any) -> dict[str, int]:
    """Count float64 values that are exactly equal (no tolerance).

    ``group_count`` is the number of distinct values shared by two or more
    entries and ``tied_count`` the number of entries in those groups.
    """
    _, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    shared = counts[counts > 1]
    return {"group_count": int(shared.size), "tied_count": int(shared.sum())}


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
    }


SANITY_CHECK_ROLE = "sanity check, not the verdict"


def _sanity_check(draw_stats: pl.DataFrame) -> dict[str, Any]:
    """Compare the real and flat draws' bottleneck distances (a sanity check).

    Both conditions are measured against the REAL nominal diagram. The block is
    labelled by ``role`` as a sanity check: the verdict is the confidence band.
    ``real_share_below_flat_bottleneck_p5`` is the share of real draws whose
    bottleneck is strictly below the flat 5th percentile (``flat_bottleneck_p5``).
    Only the bottleneck distance is compared; Wasserstein-1 is not. Reported,
    never asserted.
    """
    values: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        frame = draw_stats.filter(pl.col("condition") == condition)
        if not frame.height:
            raise ValueError(f"Draw statistics have no {condition!r} draws.")
        column = frame["bottleneck"].cast(pl.Float64).to_numpy()
        if not np.isfinite(column).all():
            raise ValueError(f"{condition} bottleneck has non-finite values.")
        values[condition] = column
    flat_p5 = float(np.percentile(values["flat"], 5))
    return {
        "role": SANITY_CHECK_ROLE,
        "statistic": "bottleneck",
        "reference": "both conditions measured against the real nominal diagram",
        "real": _interval(values["real"]),
        "flat": _interval(values["flat"]),
        "flat_bottleneck_p5": flat_p5,
        "real_share_below_flat_bottleneck_p5": float(np.mean(values["real"] < flat_p5)),
    }


def global_verdict(
    draw_stats: pl.DataFrame, hills: pl.DataFrame, band_quantile: float
) -> dict[str, Any]:
    """Judge the nominal diagram against the declared plug-in perturbation band.

    ``c`` is the ``band_quantile`` quantile of the real draws' bottleneck distance
    (:func:`band_width`). A nominal diagram point survives the band when its
    persistence is strictly greater than ``2c``; a point exactly at ``2c`` does not.
    This is neither calibrated coverage nor a peak-level FDR; the ``significant``
    field names are kept for compatibility. ``hills`` carries one row per diagram
    point (``nominal_persistence`` and the hill's ``outcome``), so the surviving and
    non-surviving points are cross-tabulated by outcome. ``relative_wasserstein_1`` is each real draw's
    ``wasserstein_1`` divided by the nominal diagram's total persistence (the L1
    mass of the nominal diagram, global pair included); it is descriptive only.
    The real draws' medians and 95% intervals of ``wasserstein_1``,
    ``bottleneck`` and ``peak_count`` are kept. The band is the verdict; the
    ``sanity_check`` block (real against flat bottleneck, see
    :func:`_sanity_check`) is labelled as a sanity check, and there is no
    real-versus-flat Wasserstein-1. Reported, never asserted.
    """
    c = band_width(draw_stats, band_quantile)
    two_c = 2.0 * c
    real = draw_stats.filter(pl.col("condition") == "real")
    values = {
        name: real[name].cast(pl.Float64).to_numpy() for name in DISTANCE_STATISTICS
    }
    for name, column in values.items():
        if not np.isfinite(column).all():
            raise ValueError(f"real {name} has non-finite values.")
    persistence = hills["nominal_persistence"]
    total = float(persistence.sum())
    if not total > 0:
        raise ValueError("The nominal diagram has no persistence to scale by.")
    significant = persistence > two_c
    return {
        "band_quantile": band_quantile,
        "quantile_method": QUANTILE_METHOD,
        "c": c,
        "two_c": two_c,
        "significance_rule": "nominal persistence > 2c (strict)",
        "nominal_point_count": hills.height,
        "significant_point_count": int(significant.sum()),
        "nominal_total_persistence": total,
        "relative_wasserstein_1": _interval(values["wasserstein_1"] / total),
        "real": {name: _interval(column) for name, column in values.items()},
        "sanity_check": _sanity_check(draw_stats),
    }


def _check_result(
    result: BootstrapDistances, settings: dict[str, Any], label: str
) -> None:
    """Require the result to echo the requested settings."""
    expected = {
        "n_draws": settings["n_replicates"],
        "seed": settings["seed"],
        "tree_type": "split",
    }
    found = {name: getattr(result, name) for name in expected}
    if found != expected:
        raise ValueError(f"The {label} bootstrap did not run as requested: {found}.")
    if result.draw_stats.height != settings["n_replicates"]:
        raise ValueError(
            f"The {label} bootstrap returned {result.draw_stats.height} draw rows "
            f"for {settings['n_replicates']} draws."
        )


def _condition_draws(result: BootstrapResult, condition: str) -> pl.DataFrame:
    """Return one condition's per-draw statistics with the stage's column types."""
    return result.draw_stats.select(
        pl.col("draw").cast(pl.Int64),
        pl.lit(condition).alias("condition"),
        pl.col("leaf_count").cast(pl.Int64).alias("peak_count"),
        pl.col("wasserstein_1").cast(pl.Float64),
        pl.col("bottleneck").cast(pl.Float64),
        pl.col("reproduced_pair_fraction").cast(pl.Float64),
    )


def _draw_stats(real: BootstrapResult, flat: BootstrapResult) -> pl.DataFrame:
    frames = [
        _condition_draws(result, condition)
        for condition, result in (("real", real), ("flat", flat))
    ]
    return pl.concat(frames).select(*DRAW_COLUMNS)


def check_reproduced_pair_denominator(
    draw_stats: pl.DataFrame, point_count: int, tolerance: float = 1e-6
) -> None:
    """Require every ``reproduced_pair_fraction`` to be a multiple of 1 / point_count.

    Topographer divides the number of reproduced nominal pairs by the number of
    points of the nominal diagram, so ``fraction * point_count`` is a whole number
    for every draw when ``point_count`` is that diagram's size. This is a guard
    that the stage's diagram has Topographer's size.
    """
    scaled = draw_stats["reproduced_pair_fraction"].to_numpy() * point_count
    if (
        not np.isfinite(scaled).all()
        or (np.abs(scaled - np.round(scaled)) > tolerance).any()
    ):
        raise ValueError(
            "reproduced_pair_fraction is not a multiple of 1 / "
            f"{point_count}: the nominal diagram does not have the size Topographer "
            "measures draws against."
        )


def _check_same_graph(real: ScalarNetwork, flat: ScalarNetwork) -> None:
    """Require the flat network to have the real network's vertices and edges.

    ``LeafCooccurrence`` reads the REAL reference tree for the
    flat run and cannot check edges, so this is the only guard.
    """
    real_ids = [str(v) for v in real.vertex_ids]
    flat_ids = [str(v) for v in flat.vertex_ids]
    if real_ids != flat_ids:
        raise ValueError(
            "The flat-landscape network does not have the real network's vertex "
            "identities in the same order."
        )
    real_edges = [(int(a), int(b)) for a, b in real.graph.edge_list()]
    flat_edges = [(int(a), int(b)) for a, b in flat.graph.edge_list()]
    if real_edges != flat_edges:
        raise ValueError(
            "The flat-landscape network does not have the real network's edges "
            f"({len(real_edges)} real, {len(flat_edges)} flat)."
        )


def _peak_count(graph: Any) -> int:
    return sum(1 for i in graph.node_indices() if graph.in_degree(i) == 0)


def _hill_support(result: BootstrapResult) -> dict[str, float]:
    """Return the result's ``hill_support`` keyed by hill peak sequence."""
    table = result.nominal_hill_support
    return dict(
        zip(table["hill_id"].to_list(), table["hill_support"].to_list(), strict=True)
    )


def check_hills_match_reference(
    hill_indices: Mapping[str, Sequence[int]], ids: Sequence[str], reference: Any
) -> None:
    """Require the nominal hills to be Topographer's terminal branches of ``reference``.

    The hills come from ``branch_member_table`` of the in-memory nominal tree, the
    bootstrap reads them against ``terminal_branch_members`` of the rebuilt
    reference tree. Both must name the same peaks with the same member sets.
    """
    index_of = {dna: i for i, dna in enumerate(ids)}
    ours = {
        index_of[peak]: frozenset(members) for peak, members in hill_indices.items()
    }
    theirs = {
        int(peak): frozenset(int(v) for v in members)
        for peak, members in terminal_branch_members(reference).items()
    }
    if ours == theirs:
        return
    names = [str(ids[i]) for i in sorted(ours.keys() ^ theirs.keys())]
    differing = sorted(
        ids[i] for i in ours.keys() & theirs.keys() if ours[i] != theirs[i]
    )
    raise ValueError(
        "The nominal hills built from the branch member table differ from "
        "Topographer's terminal branches of the reference tree: peaks on one side "
        f"only {_quoted(names)}; member sets differ for {_quoted(differing)}."
    )


def _per_draw(seconds: float, draws: int) -> float | None:
    """Seconds per draw actually made in this session, or None when none was."""
    return seconds / draws if draws else None


@dataclass(frozen=True)
class _SeriesSetup:
    """The inputs both bootstrap series (observed, flat) are drawn from."""

    ids: list[str]
    edges: Any
    raw: np.ndarray
    se: np.ndarray
    network: Any
    reference: Any
    hills: Any
    hill_indices: Any

    def statistic(self) -> LeafCooccurrence:
        # Both series read the REAL reference tree and the REAL hills.
        return LeafCooccurrence(
            "split", reference=self.reference, nominal_hills=self.hill_indices
        )

    def flat_network(self) -> Any:
        # Topographer cannot see a reference tree's edges, so the caller guarantees
        # the flat copy is the same graph: same tables, only raw_fitness replaced.
        flat = _make_network(
            self.ids,
            self.edges,
            np.full(len(self.ids), FLAT_LANDSCAPE_CONSTANT),
            self.se,
        )
        _check_same_graph(self.network, flat)
        return flat


def _series_setup(
    prepared_network: ScalarNetwork,
    tree_analysis: Any,
    growth: SequenceGrowthRatesResult,
) -> _SeriesSetup:
    """Build and check the real network, reference tree and nominal hills."""
    ids, edges = _prepared_tables(prepared_network)
    graph = prepared_network.graph
    raw = np.array([float(graph[i]["raw_fitness"]) for i in graph.node_indices()])
    se = _pooled_se(ids, growth, raw)
    network = _make_network(ids, edges, raw, se)
    reference = check_identical_tree(network, tree_analysis)
    hills = nominal_hills(tree_analysis)
    hill_indices = hill_vertex_indices(hills, ids)
    check_hills_match_reference(hill_indices, ids, reference)
    return _SeriesSetup(ids, edges, raw, se, network, reference, hills, hill_indices)


def build_bootstrap_smoothed_tree(
    prepared_network: ScalarNetwork,
    tree_analysis: Any,
    growth: SequenceGrowthRatesResult,
    config: dict[str, Any],
    *,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
    chunks: bootstrap_chunks.ChunkedBootstrap | None = None,
) -> BootstrapSmoothedTreeResult:
    """Bootstrap the nominal tree, classify its hills and build the smoothed view.

    ``chunks`` runs the draws chunk by chunk and keeps them on disk for a resumed run; by
    default the chunks are kept in memory only.
    """
    settings = _validate_config(config)
    started = time.perf_counter()

    setup = _series_setup(prepared_network, tree_analysis, growth)
    ids, raw = setup.ids, setup.raw
    network, hills = setup.network, setup.hills
    log(
        f"Bootstrap: {len(ids)} vertices, {len(hills)} nominal hills; "
        f"{settings['n_replicates']} draws in chunks of {settings['chunk_size']} on "
        f"{settings['n_jobs']} worker(s), "
        f"seed {settings['seed']}."
    )
    if chunks is None:
        chunks = bootstrap_chunks.ChunkedBootstrap(
            bootstrap_chunks.ChunkStore(
                None,
                scientific=None,
                n_draws=settings["n_replicates"],
                seed=settings["seed"],
                chunk_size=settings["chunk_size"],
            ),
            n_jobs=settings["n_jobs"],
            log=log,
        )

    statistic = setup.statistic

    drawn = time.perf_counter()
    result = chunks.run(
        "observed",
        network,
        statistic(),
        label="Bootstrap",
    )
    real_seconds = time.perf_counter() - drawn
    _check_result(result, settings, "real")

    # Everything that needs only the real result runs before the flat rerun, so a
    # failing check does not waste the flat draws.
    if result.nominal_hill_support is None:
        raise ValueError("The bootstrap returned no hill support.")
    support = _hill_support(result)
    diagram = nominal_diagram(tree_analysis, hills)
    band = band_width(_condition_draws(result, "real"), settings["band_quantile"])
    hill_rows = _hill_table(hills, support, diagram, 2.0 * band)
    absorbed_count = int((~hill_rows["clears_band"]).sum())
    view = build_band_simplified_view(tree_analysis, hill_rows)

    nominal = tree_analysis.tree
    population = nominal.num_nodes() + tree_analysis.branch_member_table.height
    if view.assignments.height != population or population != len(ids):
        raise ValueError(
            f"Smoothed view assigns {view.assignments.height} sequences, but the "
            f"Candidate population is {len(ids)} (tree population {population})."
        )
    flagged = int(view.assignments["absorbed_peak"].sum())
    if flagged != absorbed_count:
        raise ValueError(
            f"Assignments flag {flagged} absorbed peaks, but {absorbed_count} "
            "hills were absorbed."
        )
    peaks_before, peaks_after = _peak_count(nominal), _peak_count(view.tree)
    if peaks_after != peaks_before - absorbed_count:
        raise ValueError("Surviving peaks do not equal nominal peaks minus absorbed.")

    log("Flat landscape: rerunning the bootstrap with one constant scalar.")
    flat_started = time.perf_counter()
    flat_network = setup.flat_network()
    flat_result = chunks.run(
        "flat",
        flat_network,
        statistic(),
        label="Flat landscape",
    )
    flat_seconds = time.perf_counter() - flat_started
    _check_result(flat_result, settings, "flat")
    if flat_result.nominal_hill_support is None:
        raise ValueError("The flat bootstrap returned no hill support.")

    hill_count = len(hills)
    draw_stats = _draw_stats(result, flat_result)
    check_reproduced_pair_denominator(draw_stats, diagram.height)
    verdict = global_verdict(draw_stats, hill_rows, settings["band_quantile"])
    log(
        f"Confidence band: c = {verdict['c']:.6g} ({settings['band_quantile']:g} "
        f"quantile of the real bottleneck distances), 2c = {verdict['two_c']:.6g}; "
        f"{verdict['significant_point_count']} of {verdict['nominal_point_count']} "
        "nominal diagram points have persistence above 2c."
    )

    band_info = {
        "c": verdict["c"],
        "two_c": verdict["two_c"],
        "rule": "nominal persistence > 2c (strict)",
        "peaks_before": peaks_before,
        "peaks_after": peaks_after,
        "saddles_removed": len(view.removed_saddles),
        "band_tree_node_count": int(view.tree.num_nodes()),
    }
    log(
        f"Band-simplified tree: {band_info['peaks_after']} peaks, "
        f"{band_info['saddles_removed']} saddles removed."
    )

    session_draws = {
        name: chunks.session_draws.get(name, 0) for name in ("observed", "flat")
    }
    chunk_directory = chunks.store.directory
    chunk_payloads_present = bool(
        chunk_directory
        and chunk_directory.is_dir()
        and any(
            path.is_file() and path.suffix == ".npz"
            for path in chunk_directory.iterdir()
        )
    )
    summary = {
        "settings": {name: settings[name] for name in RESULT_SETTINGS},
        "evidence_availability": {
            "checkpoint_present": False,
            "distances_present": False,
            "chunks_present": chunk_payloads_present,
        },
        "vertex_count": len(ids),
        "hill_count": hill_count,
        "peaks_before": peaks_before,
        "peaks_after": peaks_after,
        "saddles_removed": len(view.removed_saddles),
        "band_simplification": band_info,
        "sequences_retained": view.assignments.height,
        "absorbed_regular_vertex_count": int(
            view.assignments["absorbed_branch_member"].sum()
        ),
        "tree_type": result.tree_type,
        "diagnostics": {
            "distance_conventions": DISTANCE_CONVENTIONS,
            "global_verdict": verdict,
            "flat_landscape": {
                "constant": FLAT_LANDSCAPE_CONSTANT,
                "n_replicates": flat_result.n_draws,
                "seed": flat_result.seed,
            },
            "exact_ties": {
                "tree_node_height": exact_ties(
                    [float(nominal[i]["scalar"]) for i in nominal.node_indices()]
                ),
                "network_raw_fitness": exact_ties(raw),
            },
        },
        "timing": {
            # Workers of the run that produced these timings; not a result
            # setting, so it is not in ``settings`` or the fingerprint.
            "n_jobs": settings["n_jobs"],
            "real_seconds": real_seconds,
            # Draws made in THIS session per series (the seconds' denominators).
            "real_draws": session_draws["observed"],
            # Per draw made in THIS session; None when every draw was resumed.
            "real_seconds_per_draw": _per_draw(real_seconds, session_draws["observed"]),
            "flat_seconds": flat_seconds,
            "flat_draws": session_draws["flat"],
            "flat_seconds_per_draw": _per_draw(flat_seconds, session_draws["flat"]),
            "total_seconds": time.perf_counter() - started,
        },
    }
    json.dumps(summary, allow_nan=False)  # strict JSON before anything is written
    return BootstrapSmoothedTreeResult(
        hills=hill_rows,
        draw_stats=draw_stats,
        summary=summary,
        provenance={},
    )


def build_band_simplified_view(
    tree_analysis: Any,
    hills: pl.DataFrame,
) -> SelectedLeafView:
    """Absorb every hill whose ordinary H0 persistence does not exceed ``2c``.

    The band is simultaneous: one threshold ``2c`` (``band_outcome`` ``absorbed`` for
    ``clears_band`` false) cuts the nominal diagram. All the absorbed hills go
    through ``absorb_terminal_branches`` in one call, so the surviving pairs are
    recomputed from the surviving tree.

    Fails closed (``ValueError``) unless the surviving peaks are exactly the
    ``clears_band`` hills, every surviving pair keeps its nominal persistence,
    independently of historical core outcomes.
    """
    unknown = sorted(set(hills["band_outcome"]) - set(BAND_OUTCOMES))
    if unknown:
        raise ValueError(f"Unknown band outcomes {unknown}.")
    clears = hills["clears_band"].to_list()
    if (hills["band_outcome"] == "kept").to_list() != clears:
        raise ValueError("band_outcome disagrees with clears_band.")
    expected = set(hills.filter(pl.col("clears_band"))["peak_sequence"])
    absorbed = set(hills.filter(~pl.col("clears_band"))["peak_sequence"])
    view = absorb_terminal_branches(tree_analysis, absorbed)

    peaks = set(view.pairs["birth_label"])
    if peaks != expected:
        raise ValueError(
            "Band view peaks differ from the hills that clear the band: "
            + _quoted(sorted(peaks ^ expected))
            + "."
        )
    nominal = dict(
        zip(hills["peak_sequence"], hills["nominal_persistence"], strict=True)
    )
    changed = sorted(
        row["birth_label"]
        for row in view.pairs.iter_rows(named=True)
        if abs(float(row["persistence"]) - nominal[row["birth_label"]])
        > BAND_PERSISTENCE_TOLERANCE
    )
    if changed:
        raise ValueError(
            "Band view changed the persistence of surviving peaks: "
            + _quoted(changed)
            + "."
        )
    return view


def _dataset_paths(
    scientific: dict[str, Any], context: Any | None = None
) -> list[Path]:
    return [
        dataset_input_path(scientific["dataset"][key], context, root=ROOT)
        for key in ("fitness_wt", "fitness_diff")
    ]


def _fingerprint(
    settings: dict[str, Any],
    growth_provenance: dict[str, Any],
    selection: Selection | None = None,
    *,
    context: Any | None = None,
) -> dict[str, Any]:
    scientific = settings["analysis"]
    simplification_dir = Path(__file__).parent
    analysis_dir = simplification_dir.parent / "analysis"
    landscape_dir = analysis_dir.parent / "landscape"
    files = {
        dataset_input_key(value, path): _sha256(path)
        for value, path in zip(
            (scientific["dataset"][key] for key in ("fitness_wt", "fitness_diff")),
            _dataset_paths(scientific, context),
            strict=True,
        )
    }
    files |= {
        source_key(path): _sha256(path)
        for path in (
            Path(__file__).resolve(),
            simplification_dir.joinpath("bootstrap_aggregation.py").resolve(),
            simplification_dir.joinpath("bootstrap_chunks.py").resolve(),
            simplification_dir.joinpath("checkpoint_identity.py").resolve(),
            analysis_dir.joinpath("candidate_tree_contrast.py").resolve(),
            landscape_dir.joinpath("selection.py").resolve(),
            landscape_dir.joinpath("selected_landscape.py").resolve(),
            landscape_dir.joinpath("full_population.py").resolve(),
            simplification_dir.joinpath("selected_leaf_simplification.py").resolve(),
            ROOT / "uv.lock",
        )
    }
    # The effective Topographer sources, so an editable install invalidates too.
    files.update(package_source_hashes(topographer))
    return {
        "schema": CACHE_SCHEMA,
        "config": _config_identity(scientific[SECTION]),
        "candidate_functional_threshold": declaration_identity(configured_declaration(settings)),
        # Keep the historical module-level seam: downstream integrations and
        # fingerprint tests replace this adapter when they provide a resolved
        # selection without a classification checkpoint on disk.
        "candidate_selection": candidate_selection_identity(
            settings, selection, **_ctx(context)
        ),
        "growth_rates_checkpoint": json.loads(
            json.dumps(without_git(growth_provenance))
        ),
        "files": files,
    }


def _check_growth_current(
    settings: dict[str, Any],
    growth: SequenceGrowthRatesResult,
    *,
    context: Any | None = None,
) -> None:
    """Refuse a growth-rate checkpoint built from different inputs or code."""
    stored = growth.provenance["fingerprint"]
    stored = {
        **stored,
        "files": {
            name: digest
            for name, digest in stored.get("files", {}).items()
            if not name.startswith("barseq_export/")
        },
    }
    current = json.loads(
        json.dumps(_growth_fingerprint_without_export(settings, **_ctx(context)))
    )
    if stored != current:
        raise ValueError(
            "stale growth-rate checkpoint: its fingerprint differs from the "
            "current inputs; rerun `just sequence-growth-rates`."
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_checkpoint(
    path: Path, record: str = SLOT_RECORD
) -> BootstrapSmoothedTreeResult:
    """Read and verify a checkpoint directory whose provenance record is ``record``."""
    provenance = json.loads((path / record).read_text(encoding="utf-8"))
    schema = provenance["fingerprint"].get("schema")
    if type(schema) is not int or schema not in (6, CACHE_SCHEMA):
        raise ValueError(f"Unsupported bootstrap checkpoint schema: {schema!r}.")
    from .identities import checkpoint_method

    if schema == 6:
        from ..simplification.cooccurrence_core import read_bootstrap_core

        legacy_artifacts = (
            *ARTIFACTS,
            "cores.parquet",
            "flat_cores.parquet",
            "flat_hills.parquet",
            "pair_frequency.parquet",
            "vertex_roles.parquet",
        )
        missing = [
            name for name in legacy_artifacts if name not in provenance["artifacts"]
        ]
        if missing:
            raise ValueError(f"Legacy bootstrap checkpoint lacks artifacts: {missing}.")
        for name in legacy_artifacts:
            if _sha256(path / name) != provenance["artifacts"][name]:
                raise ValueError(f"Bootstrap artifact hash mismatch: {name}.")
        legacy = read_bootstrap_core(path)
        pairs = pl.read_parquet(path / "pair_frequency.parquet")
        roles = pl.read_parquet(path / "vertex_roles.parquet")
        draws = pl.read_parquet(path / "draw_stats.parquet")
        summary = legacy.summary
        checkpoint_method(summary, provenance)
        hills = legacy.hills
        if (
            pairs.columns != list(PAIR_COLUMNS)
            or roles.columns != ["dna_sequence", *ROLE_COLUMNS]
            or legacy.cores.columns != list(CORE_COLUMNS)
            or legacy.flat_cores.columns != list(CORE_COLUMNS)
            or draws.columns != list(DRAW_COLUMNS)
            or hills.height != summary["hill_count"]
            or legacy.flat_hills.height != hills.height
            or set(legacy.flat_hills["peak_sequence"]) != set(hills["peak_sequence"])
            or draws.height != 2 * summary["settings"]["n_replicates"]
            or set(draws["condition"]) != set(CONDITIONS)
            or int(hills["clears_band"].sum())
            != summary["diagnostics"]["global_verdict"]["significant_point_count"]
            or (hills["band_outcome"] == "kept").to_list()
            != hills["clears_band"].to_list()
            or (legacy.flat_hills["band_outcome"] == "kept").to_list()
            != legacy.flat_hills["clears_band"].to_list()
            or summary["band_simplification"]["peaks_after"]
            != int(hills["clears_band"].sum())
            or summary["diagnostics"]["global_verdict"].get("sanity_check")
            != _sanity_check(draws)
        ):
            raise ValueError("Legacy bootstrap checkpoint coverage mismatch.")
        return BootstrapSmoothedTreeResult(hills, draws, summary, provenance)
    missing = [name for name in ARTIFACTS if name not in provenance["artifacts"]]
    if missing:
        raise ValueError(
            f"The bootstrap checkpoint predates its current artefacts {missing}; "
            "rerun `just bootstrap-smoothed-tree`."
        )
    for name in ARTIFACTS:
        if _sha256(path / name) != provenance["artifacts"][name]:
            raise ValueError(f"Bootstrap artifact hash mismatch: {name}.")
    hills = pl.read_parquet(path / "hills.parquet")
    draws = pl.read_parquet(path / "draw_stats.parquet")
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    checkpoint_method(summary, provenance)
    if (
        hills.columns != list(HILL_COLUMNS)
        or draws.columns != list(DRAW_COLUMNS)
        or hills.height != summary["hill_count"]
        or draws.height != 2 * summary["settings"]["n_replicates"]
        or set(draws["condition"]) != set(CONDITIONS)
        or int(hills["clears_band"].sum())
        != summary["diagnostics"]["global_verdict"]["significant_point_count"]
        or (hills["band_outcome"] == "kept").to_list() != hills["clears_band"].to_list()
        or summary["band_simplification"]["peaks_after"]
        != int(hills["clears_band"].sum())
        or summary["diagnostics"]["global_verdict"].get("sanity_check")
        != _sanity_check(draws)
    ):
        raise ValueError("Bootstrap checkpoint coverage mismatch.")
    return BootstrapSmoothedTreeResult(hills, draws, summary, provenance)


def _write_checkpoint(
    result: BootstrapSmoothedTreeResult,
    stage: Path,
    fingerprint: dict[str, Any],
    record: str = SLOT_RECORD,
) -> None:
    """Write the direct-band artefacts and stage record named ``record``."""
    from .identities import band_identity, checkpoint_draw_identity

    draw_fields, draw_id = checkpoint_draw_identity(
        fingerprint,
        seed=int(result.summary["settings"]["seed"]),
        draw_ids=range(int(result.summary["settings"]["n_replicates"])),
    )
    band_fields, band_id = band_identity(
        draw_scientific_id_sha256=draw_id,
        bandq=float(result.summary["settings"]["band_quantile"]),
    )
    result.summary["scientific_identities"] = {
        "draw": {
            "product_type": "bootstrap",
            "scientific_identity": draw_fields,
            "scientific_id_sha256": draw_id,
        },
        "band": {
            "product_type": "tree",
            "scientific_identity": band_fields,
            "scientific_id_sha256": band_id,
        },
    }
    result.hills.write_parquet(stage / "hills.parquet")
    result.draw_stats.write_parquet(stage / "draw_stats.parquet")
    result.summary["evidence_availability"].update(
        checkpoint_present=True,
        distances_present=(stage / "draw_stats.parquet").is_file(),
    )
    result.summary["fingerprint"] = fingerprint
    _write_json(stage / "summary.json", result.summary)
    provenance = {
        "fingerprint": fingerprint,
        "artifacts": {name: _sha256(stage / name) for name in ARTIFACTS},
    }
    _write_json(stage / record, provenance)
    result.provenance = provenance


def build_nominal_tree(
    settings: dict[str, Any],
    *,
    selection: Selection | None = None,
    context: Any | None = None,
) -> tuple[Any, Any]:
    """Build the clamped Candidate network and its nominal split tree in memory.

    ``selection`` is an already resolved Candidate selection, so a caller that holds
    one does not read the classification checkpoint again.
    """
    prepared, wt = build_candidate_network(
        settings, selection=selection, **_ctx(context)
    )
    analysis = analyze_prepared_landscape(
        prepared,
        name="candidate_clamped",
        wild_type_sequence=str(wt.get_column("dna_from")[0]),
    )
    return prepared.network, analysis.trees["split"]


def _configured_settings(
    settings: dict[str, Any] | None, context: Any | None = None
) -> dict[str, Any]:
    settings = settings or _load_settings(context)
    return result_settings(settings["analysis"][SECTION])


def _require_current(
    found: dict[str, Any], wanted: dict[str, Any], where: Path
) -> None:
    """Refuse a checkpoint whose settings differ from ``config/analysis.toml``."""
    if found != wanted:
        raise StaleCheckpointError(
            f"Stale bootstrap checkpoint at {where}: it was built with {found} but "
            f"config/analysis.toml asks for {wanted}; rerun "
            "`just bootstrap-smoothed-tree`."
        )


def _wanted_scientific(
    settings: dict[str, Any], *, warn: bool = True, context: Any | None = None
) -> dict[str, Any]:
    """The scientific fingerprint ``config/analysis.toml`` asks for, possibly partial.

    Parts that cannot be computed are omitted (``checkpoint_identity.unchecked_parts``
    names them) and, with ``warn``, announced: ``datasets`` when ``data/raw`` is
    missing, the growth BarSeq digests when the growth-rate checkpoint is absent.
    The growth settings come from ``config/analysis.toml`` (exactly what the growth
    stage stores as its fingerprint config), so a stale growth checkpoint never
    overrides the configuration.
    """
    scientific = settings["analysis"]
    value: dict[str, Any] = {
        "config": _config_identity(scientific[SECTION]),
        "candidate_functional_threshold": declaration_identity(configured_declaration(settings)),
        "files": {},
    }
    try:
        value["candidate_selection"] = candidate_selection_identity(
            settings, **_ctx(context)
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if warn:
            warnings.warn(
                f"Functional classification unavailable ({error}); the Candidate "
                "vertex set was not compared with the mirror.",
                RuntimeWarning,
                stacklevel=2,
            )
    unchecked = False
    try:
        value["files"] = {
            dataset_input_key(scientific["dataset"][key], path): _sha256(path)
            for key, path in zip(
                ("fitness_wt", "fitness_diff"),
                _dataset_paths(scientific, context),
                strict=True,
            )
        }
    except FileNotFoundError as error:
        unchecked = True
        if warn:
            warnings.warn(
                f"Input dataset digests could not be checked ({error}).",
                RuntimeWarning,
                stacklevel=2,
            )
    wanted = checkpoint_identity.scientific_fingerprint(value)
    if unchecked:
        del wanted["datasets"]
    growth: dict[str, Any] = {}
    if "sequence_growth_rates" in scientific:
        growth["config"] = json.loads(json.dumps(scientific["sequence_growth_rates"]))
    try:
        identity = checkpoint_identity.growth_identity(
            load_cached_sequence_growth_rates(**_ctx(context)).provenance
        )
        if identity is not None:
            growth["datasets"] = identity["datasets"]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if warn:
            warnings.warn(
                f"Growth-rate checkpoint unavailable ({error}); its BarSeq digests "
                "were not compared with the mirror.",
                RuntimeWarning,
                stacklevel=2,
            )
    if growth:
        wanted["growth"] = growth
    return wanted


def _local_settings(path: Path) -> dict[str, Any] | None:
    """Settings of the checkpoint at ``path``, or ``None`` when there is none."""
    if not (path / "provenance.json").is_file():
        return None
    return checkpoint_identity.current_checkpoint_settings(path)


def _refuse_checkpoint_only_band_change(
    wanted: Mapping[str, Any],
    scientific: Mapping[str, Any],
    *,
    context: Any | None = None,
) -> None:
    """Fail closed when a matching draw checkpoint has no reusable draw evidence."""
    candidates: list[tuple[Path, str]] = [(cache_dir(context), SLOT_RECORD)]
    candidates.extend(
        (directory, SLOT_RECORD)
        for directory in sorted(
            (_processed_root(context) / "bootstrap").glob("*/checkpoint")
        )
    )
    for path, record_name in candidates:
        try:
            record = json.loads((path / record_name).read_text(encoding="utf-8"))
            saved_scientific = checkpoint_identity.scientific_fingerprint(
                record["fingerprint"]
            )
            if checkpoint_identity.differences(saved_scientific, scientific):
                continue
            saved_settings = checkpoint_identity.current_checkpoint_settings(path)
            before = {**saved_settings, "band_quantile": None}
            after = {**dict(wanted), "band_quantile": None}
            if before != after or saved_settings.get("band_quantile") == wanted.get(
                "band_quantile"
            ):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        chunk_dir = bootstrap_chunks.chunk_directory(scientific, _chunk_base(context))
        has_chunks = chunk_dir.is_dir() and any(chunk_dir.glob("*.npz"))
        if not has_chunks:
            raise RuntimeError(
                "A checkpoint matches the bootstrap draw identity but its band "
                "settings differ, and no saved draw chunks are available to "
                "rebuild the band result. Evidence required: preserve or restore "
                "the bootstrap chunks before changing band settings; refusing to "
                "silently draw again."
            )


# Errors that mean a checkpoint directory cannot be read as it stands.
_READ_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    pl.exceptions.PolarsError,
)


def bootstrap_stage(scientific: Mapping[str, Any]) -> Stage:
    """Declare the stage: its key is the run's scientific identity and nothing else.

    ``scientific`` is what decides the result (``checkpoint_identity.scientific_fingerprint``:
    the result settings, the candidate functional threshold, the input dataset digests and
    the growth-rate settings and BarSeq digests), never the digests of code, the lock
    file or the growth checkpoint, so a checkpoint, or a processed checkpoint
    of the run, stays reusable across code edits instead of costing hours of draws.
    The full fingerprint stays in the stage's own record. Code files are still
    listed so their digests can be recorded as a label; they are not part of
    ``stage_fingerprint``. Before this change ``_declared`` hashed that tuple
    into the key, which contradicted this docstring and ADR 0013. No input file
    is declared: the identity already holds the dataset digests, and
    ``project_stage`` strips ``n_jobs`` and ``chunk_size`` at any depth.
    """
    from ..stages import project_stage

    return project_stage(
        STAGE,
        settings={"scientific": scientific},
        root=ROOT,
        code=(
            Path(__file__).resolve(),
            Path(__file__).with_name("__init__.py").resolve(),
            Path(__file__).with_name("bootstrap_chunks.py").resolve(),
            Path(__file__).with_name("checkpoint_identity.py").resolve(),
            Path(__file__).with_name("selected_leaf_simplification.py").resolve(),
            Path(__file__).with_name("identities.py").resolve(),
            ROOT / "src/dubanevics2026topological/analysis/barseq_inputs.py",
            ROOT / "src/dubanevics2026topological/analysis/candidate_tree_contrast.py",
            ROOT / "src/dubanevics2026topological/analysis/checkpoint.py",
            ROOT / "src/dubanevics2026topological/analysis/produce_or_load.py",
            ROOT / "src/dubanevics2026topological/analysis/sequence_growth_rates.py",
            ROOT / "src/dubanevics2026topological/landscape/selection.py",
            ROOT / "src/dubanevics2026topological/landscape/selected_landscape.py",
        ),
    )


def _use_processed(
    wanted: dict[str, Any],
    scientific: Mapping[str, Any] | None,
    use: Callable[[Path], None],
    *,
    context: Any | None = None,
) -> bool:
    """Call ``use`` on the first matching processed checkpoint that validates; return whether one did.

    The candidates are ``<processed>/bootstrap/<state>/checkpoint`` folders
    (``checkpoint_identity.iter_processed_checkpoints``); nothing here writes to them.
    A candidate that fails validation warns and the next one is tried; an ambiguous
    match raises ``AmbiguousCheckpointError``. When candidates match but every one fails,
    it raises ``RuntimeError``: a rerun would draw for hours.
    """
    found = failed = 0
    root = _processed_root(context)
    for candidate in checkpoint_identity.iter_processed_checkpoints(
        wanted, scientific, root
    ):
        found += 1
        try:
            _read_checkpoint(candidate, _record_of(candidate))  # validate first
        except _READ_ERRORS as error:
            failed += 1
            warnings.warn(
                f"Processed checkpoint {candidate} failed validation ({error}); "
                "trying the next candidate.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        use(candidate)
        return True
    if found and found == failed:
        raise RuntimeError(
            f"Found {found} matching processed bootstrap checkpoint(s) in "
            f"{root / 'bootstrap'} and all {failed} failed validation; do not rerun "
            "the bootstrap. Inspect or repair them (see the warnings above), or run "
            "`just fetch-data` again."
        )
    return False


def _usable_slot(
    wanted: dict[str, Any],
    scientific: Mapping[str, Any],
    *,
    context: Any | None = None,
) -> bool:
    """Whether the working slot holds a readable checkpoint of exactly this run.

    An unreadable slot warns and is not used; the stage then looks for a mirror
    and, failing that, recomputes (``_export_to_slot`` keeps the unreadable one aside).
    """
    slot = cache_dir(context)
    try:
        if _local_settings(slot) != wanted:
            return False
        if checkpoint_identity.differences(
            checkpoint_identity.scientific_of(slot), scientific
        ):
            return False
        _read_checkpoint(slot)
    except _READ_ERRORS as error:
        warnings.warn(
            f"The bootstrap checkpoint at {slot} cannot be read ({error}); "
            "falling back to a mirror or a recompute (pass --force to skip the "
            "reuse check).",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


def _copy_checkpoint(source: Path, directory: Path) -> None:
    """Copy a checkpoint in the working-slot layout into a stage payload.

    The record travels as ``bootstrap_provenance.json``. The copy is verified
    against its own record, so a torn read of ``source`` never gets published.
    """
    provenance = json.loads((source / _record_of(source)).read_text(encoding="utf-8"))
    names = list(ARTIFACTS)
    if provenance["fingerprint"].get("schema") == 6:
        names.extend(
            (
                "cores.parquet",
                "flat_cores.parquet",
                "flat_hills.parquet",
                "pair_frequency.parquet",
                "vertex_roles.parquet",
            )
        )
    for name in names:
        shutil.copyfile(source / name, directory / name)
    shutil.copyfile(source / _record_of(source), directory / RECORD_NAME)
    _read_checkpoint(directory, RECORD_NAME)


def _record_of(checkpoint: Path) -> str:
    """The file holding the stage record of a slot or a processed checkpoint."""
    return SLOT_RECORD


def _slot_matches(slot: Path, artefacts: Mapping[str, str]) -> bool:
    """Whether ``slot`` (the working slot or a mirror) already holds these bytes.

    ``artefacts`` is the ``artefacts`` map of a checkpoint's record, artefact name to
    sha256, with the stage's record under ``RECORD_NAME``; ``slot`` keeps that record
    as ``provenance.json``.
    """
    record = _record_of(slot)
    for name, digest in artefacts.items():
        target = slot / (record if name == RECORD_NAME else name)
        if not target.is_file() or checkpoint_identity.sha256_of(target) != digest:
            return False
    return True


def _slot_differences(slot: Path, path: Path) -> list[str]:
    """What makes the readable ``slot`` a different run from the checkpoint at ``path``.

    Empty when both hold the same run: equal result settings and no difference in
    the scientific identity (``checkpoint_identity.differences``). Code and ``uv.lock``
    digests are outside the identity, so a slot that a re-classification or
    a forced rerun of the same run replaces is not another run. An identity that
    cannot be read counts as a difference. ``path`` need not be verified: only its
    summary and its record are read.
    """
    try:
        names: list[str] = []
        if checkpoint_identity.checkpoint_settings(
            slot
        ) != checkpoint_identity.checkpoint_settings(path):
            names.append("settings")
        record = json.loads((path / RECORD_NAME).read_text(encoding="utf-8"))
        names += checkpoint_identity.differences(
            checkpoint_identity.scientific_of(slot, _record_of(slot)),
            checkpoint_identity.scientific_fingerprint(record["fingerprint"]),
        )
    except _READ_ERRORS:
        return ["unreadable identity"]
    return names


def _export_to_slot(checkpoint: Checkpoint, context: Any | None = None) -> None:
    """Make the working slot a byte-for-byte copy of the published checkpoint.

    Runs under the checkpoint lock. The slot's ``provenance.json`` is the commit
    marker readers look for: it is removed first and put in place last, so an
    interrupted export never leaves a record that claims files that are not yet
    there. A slot that has a record but cannot be read as a checkpoint, or that
    reads as another run (its settings or scientific identity differ from the
    checkpoint's), is moved to a ``.replaced-*`` sibling and never deleted: the
    publish evicts the other run's checkpoint, so the slot may be the last local
    copy. A readable slot of the same run is overwritten, as the checkpoint is.
    """
    slot = cache_dir(context)
    if _slot_matches(slot, checkpoint.provenance["artefacts"]):
        return
    record = slot / SLOT_RECORD
    if record.is_file():
        why: str | None = None
        try:
            _read_checkpoint(slot)
        except _READ_ERRORS as error:
            why = f"cannot be read ({error})"
        else:
            if parts := _slot_differences(slot, checkpoint.path):
                differing = ", ".join(parts)
                why = f"is of another run than the checkpoint (differing {differing})"
        if why:
            aside = slot.with_name(f".{slot.name}.replaced-{uuid.uuid4().hex}")
            os.replace(slot, aside)
            warnings.warn(
                f"The bootstrap checkpoint at {slot} {why}; moved it aside to "
                f"{aside} and exporting the verified checkpoint {checkpoint.path} "
                "in its place; delete it by hand once it is not needed.",
                RuntimeWarning,
                stacklevel=2,
            )
    record.unlink(missing_ok=True)
    incoming = ".provenance.json.incoming"
    checkpoint.export_to(slot, names={RECORD_NAME: incoming})
    os.replace(slot / incoming, record)


def _published_artefacts(
    context: Any | None = None,
) -> tuple[Path, dict[str, str]] | None:
    """The published checkpoint's path and the artefact hashes its record lists.

    Or ``None`` when there is no readable record. The record is read as it stands
    and the files are not verified, so this is no ``Checkpoint`` (only
    ``produce_or_load`` hands one out); whoever uses the bytes verifies them.
    """
    published = _published_leaf(context)
    try:
        record = json.loads((published / PROVENANCE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("artefacts"), dict):
        return None
    return published, record["artefacts"]


def _disagrees_with_published(slot: Path, context: Any | None = None) -> bool:
    """Whether ``slot`` holds the run of the published checkpoint in other bytes.

    A published checkpoint of another run, or one that cannot be read, says nothing
    about ``slot``: the configuration decides whether it is used.
    """
    published = _published_artefacts(context)
    if published is None:
        return False
    path, artefacts = published
    return not _slot_differences(slot, path) and not _slot_matches(slot, artefacts)


def _require_exported(context: Any | None = None) -> None:
    """Refuse a working slot that lags the published checkpoint of the same run.

    The slot is a byte-for-byte export of the checkpoint, written after it is
    published, so a run killed in between leaves the previous slot beside newer
    published bytes. The two must agree, record and artefact hashes, when they hold
    the same run.
    """
    if _disagrees_with_published(cache_dir(context), **_ctx(context)):
        raise RuntimeError(
            f"The bootstrap slot at {cache_dir(context)} does not match the published "
            f"checkpoint at {checkpoint_dir(context)} (an export that was "
            "interrupted?); run "
            "`just bootstrap-smoothed-tree` to export it "
            "again."
        )


def _restore_processed(source: Path, context: Any | None = None) -> None:
    """Restore a validated processed checkpoint into the working slot, unless that would lag.

    A processed checkpoint that holds the run of the published checkpoint in other
    bytes would give a slot that ``_require_exported`` then refuses. It is refused
    first, so the refusal costs no slot write and moves no slot aside.
    """
    if _disagrees_with_published(source, **_ctx(context)):
        raise RuntimeError(
            f"The processed checkpoint at {source} does not match the published "
            f"checkpoint at {checkpoint_dir(context)}: it holds the same run in other "
            "bytes, and restoring it would leave a slot that differs from the "
            "checkpoint. Run `just bootstrap-smoothed-tree` to export the published "
            "checkpoint instead."
        )
    checkpoint_identity.restore_directory(source, cache_dir(context))
    _write_restore_marker(source, context)


def _write_restore_marker(checkpoint: Path, context: Any | None = None) -> None:
    """Record in the slot which processed ``bootstrap/<state>`` it was restored from.

    The marker carries the sha256 of the slot's record, so ``restored_source`` stops
    reporting the source once the slot is overwritten by another run. A failed write
    only loses the provenance note; it never fails the restore.
    """
    slot = cache_dir(context)
    try:
        marker = {
            "source": checkpoint.parent.name,
            "record_sha256": _sha256(slot / SLOT_RECORD),
        }
        temporary = slot / f"{RESTORE_MARKER}.tmp"
        _write_json(temporary, marker)
        os.replace(temporary, slot / RESTORE_MARKER)
    except OSError as error:
        warnings.warn(
            f"Could not note the restore source in {slot} ({error}).",
            RuntimeWarning,
            stacklevel=2,
        )


def restored_source(
    slot: Path | None = None, *, context: Any | None = None
) -> str | None:
    """The processed state (``bootstrap/<state>``) the working slot was restored from.

    ``None`` when the slot was computed or exported locally, or was overwritten after
    the restore (the marker's record digest no longer matches the slot's record).
    """
    slot = cache_dir(context) if slot is None else slot
    try:
        marker = json.loads((slot / RESTORE_MARKER).read_text(encoding="utf-8"))
        if marker["record_sha256"] == _sha256(slot / SLOT_RECORD):
            return str(marker["source"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def load_cached_bootstrap_smoothed_tree(
    settings: dict[str, Any] | None = None,
    *,
    context: Any | None = None,
    lock: bool = True,
) -> BootstrapSmoothedTreeResult:
    """Read only a completed checkpoint; never compute or publish.

    The working slot in ``output/analysis`` is used when its settings equal the
    configured ones; otherwise the ``checkpoint`` of a ``data/processed/bootstrap/<state>``
    whose fingerprint matches the configuration is restored into the slot first (the
    published checkpoint is left to the next ``just bootstrap-smoothed-tree``). A checkpoint
    that is read but differs from the configuration raises ``ValueError``. Nothing
    is restored while the helper reports an unsettled publication
    (``publication_state``: a pending marker or a kept backup of the published
    checkpoint). The slot must also be the export of the published checkpoint when
    that holds the same run: a slot that differs from it (a run killed between its
    publish and its export) raises ``RuntimeError`` until ``just bootstrap-smoothed-tree``
    exports it again. A mirror that differs from
    the published checkpoint of the same run is refused the same way before it is
    restored, so the refusal writes nothing. The checkpoint lock is held, so a run
    that is still drawing makes this raise ``RuntimeError`` (retry afterwards).
    With ``lock=False`` the caller must already hold the published checkpoint lock
    for the entire read and any dependent operations.
    """
    settings = settings or _load_settings(context)
    wanted = _configured_settings(settings, context)
    published = _published_leaf(context)
    slot = cache_dir(context)
    with checkpoint_lock(published, LABEL) if lock else contextlib.nullcontext():
        state = publication_state(published)
        if state["pending"] or state["backup"]:
            raise RuntimeError(
                "Bootstrap smoothed tree refresh is pending; rerun "
                "`just bootstrap-smoothed-tree`."
            )
        local = _local_settings(slot)
        stale_inputs: list[str] = []
        if local is not None and local == wanted:
            # A checkpoint in output/ must not escape the digest checks either.
            stale_inputs = checkpoint_identity.differences(
                checkpoint_identity.scientific_of(slot),
                _wanted_scientific(settings, warn=False, context=context),
            )
        needs_mirror = local is None or local != wanted or bool(stale_inputs)
        if needs_mirror and _use_processed(
            wanted,
            _wanted_scientific(settings, context=context),
            lambda source: _restore_processed(source, context),
            context=context,
        ):
            local, stale_inputs = wanted, []
        if local is None:
            raise FileNotFoundError(
                f"No bootstrap-smoothed tree checkpoint at {slot} or in "
                f"{_processed_root(context) / 'bootstrap'}; run "
                "`just bootstrap-smoothed-tree`, `just all` or `just fetch-data`."
            )
        if stale_inputs:
            raise StaleCheckpointError(
                f"Stale bootstrap checkpoint at {slot}: its settings match "
                "config/analysis.toml but its inputs differ "
                f"({', '.join(stale_inputs)}); rerun `just bootstrap-smoothed-tree`."
            )
        _require_current(local, wanted, slot)
        result = _read_checkpoint(slot)
        _require_exported(context)
        return result


def _differing_parts(found: dict[str, Any], wanted: dict[str, Any]) -> list[str]:
    """Names of the fingerprint parts of ``found`` that differ from ``wanted``."""
    parts: list[str] = []
    for key in sorted(found.keys() | wanted.keys()):
        if key == "files":
            old, new = found.get(key) or {}, wanted.get(key) or {}
            names = sorted(
                n for n in old.keys() | new.keys() if old.get(n) != new.get(n)
            )
            if names:
                shown = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
                parts.append(f"file digests ({shown})")
        elif found.get(key) != wanted.get(key):
            parts.append(
                "config (result settings or noise model)" if key == "config" else key
            )
    return parts


def _publish_processed(
    checkpoint: Checkpoint, settings: dict[str, Any], context: Any | None = None
) -> Path | None:
    """Publish the finished checkpoint of a BY run to ``processed/bootstrap/<state>``.

    The product is the bootstrap draw evidence of that state: ``checkpoint/`` in the
    working-slot layout (the stage record as ``provenance.json``) and a ``manifest.json``
    that names the draws, the seed and the Candidate vertex set it was drawn for. A run
    that is not a BY variant (a point-estimate selection) publishes nothing.
    """
    from ..analysis.band_quantile_scan_panels import panel_for_variant
    from ..landscape.products import _publish
    from ..results import RunContext, processed

    panel = panel_for_variant(
        None if context is None else getattr(context, "variant_name", None), settings
    )
    if panel is None:
        return None
    record = json.loads((checkpoint.path / RECORD_NAME).read_text(encoding="utf-8"))
    fingerprint = record["fingerprint"]
    scientific = checkpoint_identity.scientific_fingerprint(fingerprint)
    run = checkpoint_identity.checkpoint_settings(checkpoint.path)
    params = {
        "thr": "published" if panel.threshold == "published" else "reflected",
        "fdr": "by",
        "q": panel.q,
        "draws": int(run["n_replicates"]),
        "seed": int(run["seed"]),
    }
    selection = dict(scientific.get("candidate_selection") or {})
    identity = {**params, **selection, "scientific_fingerprint": scientific}
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{checkpoint.path.name}.processed-", dir=checkpoint.path.parent
        )
    )
    try:
        checkpoint.export_to(staging, names={RECORD_NAME: SLOT_RECORD})
        digests = {
            f"checkpoint/{path.name}": _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }

        def write_artifacts(directory: Path) -> None:
            target = directory / "checkpoint"
            target.mkdir()
            for path in sorted(staging.iterdir()):
                if path.is_file():
                    shutil.copyfile(path, target / path.name)

        run_context = context if isinstance(context, RunContext) else RunContext()
        product, _ = _publish(
            "bootstrap",
            params,
            identity,
            context=run_context,
            input_data_sha256s=digests,
            parents=(),
            code_paths=(Path(__file__).resolve(),),
            write_artifacts=write_artifacts,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return processed.publish(
        product, "bootstrap", panel.state, root=processed.processed_root(context)
    )


def _stands(
    ready: Checkpoint,
    fingerprint: dict[str, Any],
    run_settings: dict[str, Any],
    scientific: dict[str, Any],
    *,
    context: Any | None = None,
) -> bool:
    """Whether a reused checkpoint stands as it is; ``False`` asks for a re-classification.

    A checkpoint whose scientific identity matches is reused as it stands. Code
    digests, ``uv.lock`` and Topographer's install path are not a reason to
    re-classify; a code fix is an explicit ``--force``. A different scientific
    identity is still re-classified from saved chunks when they are complete,
    and reused with a warning otherwise. Nothing here rewrites the checkpoint.
    """
    record = json.loads((ready.path / RECORD_NAME).read_text(encoding="utf-8"))
    found = record["fingerprint"]
    # Code, uv.lock and the installed Topographer are a label. They used to
    # force a re-classification; a code fix is now an explicit ``--force``.
    if checkpoint_identity.scientific_fingerprint(found) == scientific:
        stored_band = found.get("config", {}).get("band_quantile")
        wanted_band = run_settings.get("band_quantile")
        # The band quantile is not part of the draw identity, but it does change
        # the classified tree. A different band is re-classified from saved chunks.
        if stored_band is None or stored_band == wanted_band:
            print(
                f"Reusing the bootstrap checkpoint at {cache_dir(context)} (no draws); "
                "pass --force "
                "to recompute.",
                flush=True,
            )
            return True
    if found != fingerprint:
        if bootstrap_chunks.chunks_complete(
            scientific,
            n_draws=run_settings["n_replicates"],
            seed=run_settings["seed"],
            chunk_size=run_settings["chunk_size"],
            base=_chunk_base(context),
        ):
            print(
                "The bootstrap checkpoint predates the current code or inputs; "
                "re-classifying it from the saved chunks (no redraw).",
                flush=True,
            )
            return False
        parts = _differing_parts(found, fingerprint)
        warnings.warn(
            f"Reusing the bootstrap checkpoint at {cache_dir(context)} although it "
            "was made "
            f"with different {', '.join(parts)} and no saved chunks allow "
            f"re-classifying it. Pass --force to recompute.",
            RuntimeWarning,
            stacklevel=2,
        )
    print(
        f"Reusing the bootstrap checkpoint at {cache_dir(context)} (no draws); "
        "pass --force "
        "to recompute.",
        flush=True,
    )
    return True


def _draw_into(
    directory: Path,
    settings: dict[str, Any],
    fingerprint: dict[str, Any],
    scientific: dict[str, Any],
    growth: SequenceGrowthRatesResult,
    *,
    fresh: bool,
    context: Any | None = None,
) -> None:
    """Draw, classify and write the artefacts and the stage's record into ``directory``.

    The draws go through the chunk store (``bootstrap_chunks``): every finished chunk
    is saved, a crash or a stop signal keeps them and publishes nothing, and a rerun
    redraws only the missing ones. ``fresh`` moves the saved chunks aside first.
    """
    config = settings["analysis"][SECTION]
    run_settings = _validate_config(config, context)
    network, tree_analysis = build_nominal_tree(settings, **_ctx(context))
    with bootstrap_chunks.ChunkStore(
        bootstrap_chunks.chunk_directory(scientific, _chunk_base(context)),
        scientific=scientific,
        n_draws=run_settings["n_replicates"],
        seed=run_settings["seed"],
        chunk_size=run_settings["chunk_size"],
        fresh=fresh,
        hold_lock=True,
        log=lambda message: print(message, flush=True),
    ) as store:
        runner = bootstrap_chunks.ChunkedBootstrap(
            store,
            n_jobs=run_settings["n_jobs"],
            log=lambda message: print(message, flush=True),
            no_draw=None
            if context is None
            else bootstrap_chunks.no_draw_requested(context=context),
        )
        with runner.stop.installed():
            result = build_bootstrap_smoothed_tree(
                network,
                tree_analysis,
                growth,
                config,
                chunks=runner,
            )
        if store._legacy_manifest:
            assert store.directory is not None
            manifest_path = store.directory / bootstrap_chunks.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result.summary["legacy_evidence_lineage"] = {
                "kind": "legacy_bootstrap_chunks",
                "chunk_directory_key": store.directory.name,
                "manifest_sha256": _sha256(manifest_path),
                "scientific_fingerprint": manifest["scientific_fingerprint"],
            }
    _write_checkpoint(result, directory, fingerprint, RECORD_NAME)


def compute_bootstrap_smoothed_tree(
    settings: dict[str, Any] | None = None,
    *,
    force: bool = False,
    fresh: bool = False,
    context: Any | None = None,
) -> BootstrapSmoothedTreeResult:
    """Load the current checkpoint or bootstrap, classify and atomically publish one.

    The stage goes through ``produce_or_load`` (ADR 0013): the published checkpoint
    in ``output/checkpoints/bootstrap_smoothed_tree`` is keyed by the scientific
    fingerprint (``bootstrap_stage``) and the working slot in ``output/analysis`` is
    exported from it under the checkpoint lock. A checkpoint of the configured run,
    or failing that a readable slot or the checkpoint of a matching
    ``data/processed/bootstrap/<state>``, is reused unless ``force``
    is set: as it is (restored, never re-stamped) when its full fingerprint matches,
    or, when code or ``uv.lock`` digests differ and the draws are not all
    saved as chunks, with a warning; when they are, it is re-classified from the
    chunks (no redraw). A finished run of a BY variant is published to
    ``data/processed/bootstrap/<state>`` from the published checkpoint before the slot
    is exported, so a failing export cannot leave a fresh draw unpublished.

    The draws are saved chunk by chunk (``bootstrap_chunks``) and a rerun resumes
    from them; ``force`` recomputes the checkpoint but still reuses the chunks,
    ``fresh`` moves them aside and redraws everything (it implies ``force``: a
    checkpoint built from the discarded chunks must not be reused). SIGTERM,
    SIGUSR1 or SIGINT stops the run after the current chunk with exit code 124.
    """
    force = force or fresh
    settings = settings or _load_settings(context)
    config = settings["analysis"][SECTION]
    run_settings = _validate_config(config, context)
    growth = load_cached_sequence_growth_rates(**_ctx(context))
    _check_growth_current(settings, growth, **_ctx(context))
    fingerprint = _fingerprint(settings, growth.provenance, **_ctx(context))
    scientific = checkpoint_identity.scientific_fingerprint(fingerprint)
    wanted = result_settings(config)
    if not fresh:
        _refuse_checkpoint_only_band_change(wanted, scientific, **_ctx(context))
    stage = bootstrap_stage(scientific)
    drew = False
    reclassify = False
    result: BootstrapSmoothedTreeResult | None = None

    def run(forced: bool) -> None:
        def produce(directory: Path) -> None:
            nonlocal drew
            if not forced:
                if _usable_slot(wanted, scientific, **_ctx(context)):
                    _copy_checkpoint(cache_dir(context), directory)
                    return
                if _use_processed(
                    wanted,
                    scientific,
                    lambda source: _copy_checkpoint(source, directory),
                    context=context,
                ):
                    return
            drew = True
            _draw_into(
                directory,
                settings,
                fingerprint,
                scientific,
                growth,
                fresh=fresh,
                **_ctx(context),
            )

        def settle(ready: Checkpoint) -> None:
            nonlocal reclassify, result
            if not drew and not _stands(
                ready, fingerprint, run_settings, scientific, **_ctx(context)
            ):
                reclassify = True
                return
            # The processed product comes first, from the published bytes: a slot
            # export that fails must not cost a fresh multi-hour draw its only copy.
            _publish_processed(ready, settings, context)
            _export_to_slot(ready, **_ctx(context))
            result = _read_checkpoint(cache_dir(context))

        produce_or_load(
            stage,
            produce,
            checkpoint_dir=checkpoint_dir(context),
            force=forced,
            on_ready=settle,
        )

    run(force)
    if reclassify:
        run(True)  # the chunks are all saved: classification only, no redraw
    assert result is not None
    return result

