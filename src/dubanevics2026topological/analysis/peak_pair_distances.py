"""Peak-pair distances on a band-simplified split tree.

For every unordered pair of peaks (the tree's leaves) this measures sequence
and fitness distance directly and via their common saddle, the lowest common
ancestor of the two peaks. See ``docs/peak-pair-distance-design.md`` and the
GLOSSARY entries Common saddle, Direct Hamming distance, Via-saddle Hamming
distance, Direct fitness difference and Via-saddle fitness distance.

The input is the pair of tables ``save_band_tree`` writes: ``critical_nodes``
(``tree_node_identity`` is the node's DNA sequence, ``scalar`` its raw fitness)
and ``branches`` (``source_tree_node_identity`` is the upper node,
``target_tree_node_identity`` the node toward the root). Everything is keyed
by DNA sequence; no graph indices appear.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rustworkx as rx
import topographer
from scipy.stats import rankdata
from topographer.plotting import make_tree_layout

from ..provenance import upstream_identity
from ..sequence_distances import hamming_distance
from ..simplification.selected_leaf_simplification import rebuild_split_tree
from ..stages import project_stage
from . import ROOT, load_settings, package_source_hashes
from .peak_rank import translate
from .produce_or_load import (
    Checkpoint,
    load_checkpoint,
    produce_or_load,
    result_directory,
)

SECTION = "peak_pair_distances"
STAGE_NAME = "peak-pair-distances"
RETRY_HINT = "just peak-pair-distances"
# The Candidate graph of the median-reflected Benjamini-Yekutieli variant.
VARIANT = "threshold=reflected_fdr=by_q=0.05"
SETTING_KEYS = ("tree", "mantel_permutations", "seed")
ARTEFACTS = ("peak_pair_distances.parquet", "peak_pair_mantel.parquet")

OUTPUT_COLUMNS = (
    "peak_1_dna",
    "peak_2_dna",
    "common_saddle_dna",
    "peak_1_leaf_order",
    "peak_2_leaf_order",
    "nucleotide_direct_hamming",
    "nucleotide_via_saddle_hamming",
    "amino_acid_direct_hamming",
    "amino_acid_via_saddle_hamming",
    "direct_fitness_difference",
    "via_saddle_fitness_distance",
)


MANTEL_COLUMNS = (
    "comparison",
    "alphabet",
    "spearman_rho",
    "mantel_p",
    "permutations",
    "seed",
    "peak_count",
    "pair_count",
)
# Each comparison names the Hamming column and the fitness column it sets against
# one another, per alphabet; "via_minus_direct" is the excess of via-saddle over
# direct distance on both axes.
ALPHABETS = ("nucleotide", "amino_acid")
COMPARISONS = ("direct", "via_saddle", "via_minus_direct")
# Float noise on a correlation that is exactly equal must not break the tie.
RHO_TOLERANCE = 1e-12
# Permutations drawn per block: bounds memory at (chunk x pairs) floats.
PERMUTATION_CHUNK = 500


def _structure(
    critical_nodes: pl.DataFrame, branches: pl.DataFrame
) -> tuple[dict[str, float], dict[str, str], dict[str, list[str]], str]:
    """Fitness by node, parent by node, children by node and the root."""
    identities = critical_nodes["tree_node_identity"].to_list()
    if len(set(identities)) != len(identities):
        duplicated = sorted({n for n in identities if identities.count(n) > 1})
        raise ValueError(f"Duplicate tree_node_identity: {duplicated[:3]}.")
    fitness = dict(
        zip(
            identities,
            (float(v) for v in critical_nodes["scalar"].to_list()),
            strict=True,
        )
    )
    parent: dict[str, str] = {}
    children: dict[str, list[str]] = {node: [] for node in fitness}
    for child, up in zip(
        branches["source_tree_node_identity"].to_list(),
        branches["target_tree_node_identity"].to_list(),
        strict=True,
    ):
        for endpoint in (child, up):
            if endpoint not in fitness:
                raise ValueError(
                    f"Branch endpoint {endpoint!r} is missing from critical_nodes."
                )
        if child in parent:
            raise ValueError(f"Node {child!r} has more than one edge toward the root.")
        parent[child] = up
        children[up].append(child)
    roots = [node for node in fitness if node not in parent]
    if len(roots) != 1:
        raise ValueError(f"A split tree has one root, found {len(roots)}.")
    reached, frontier = {roots[0]}, [roots[0]]
    while frontier:
        for kid in children[frontier.pop()]:
            reached.add(kid)
            frontier.append(kid)
    if len(reached) != len(fitness):
        raise ValueError("The branches contain a cycle detached from the root.")
    return fitness, parent, children, roots[0]


def peak_leaf_order(critical_nodes: pl.DataFrame, branches: pl.DataFrame) -> list[str]:
    """Peak DNA sequences in the drawn tree's left-to-right leaf order.

    The tree is rebuilt from the tables as a Topographer split tree
    (``rebuild_split_tree``) and laid out by Topographer's public
    ``make_tree_layout`` with ``horizontal_layout="subtree_extremum"``, the
    repository's tree layout. Leaves (nodes with no child; the root is never a
    peak) are sorted by layout x and mapped to DNA at once, so no graph index
    leaves this function.
    """
    fitness, _, children, _ = _structure(critical_nodes, branches)
    if len(fitness) < 2:
        return []
    graph = rx.PyDiGraph()
    index = {
        dna: graph.add_node({"dna_sequence": dna, "scalar": value})
        for dna, value in fitness.items()
    }
    for up, kids in children.items():
        for kid in kids:
            graph.add_edge(index[kid], index[up], {})
    layout = make_tree_layout(
        rebuild_split_tree(graph), horizontal_layout="subtree_extremum"
    )
    parents = set(layout.edge_table["source_tree_node_index"].to_list())
    leaves = layout.node_table.filter(
        ~pl.col("tree_node_index").is_in(parents)
        & (pl.col("tree_node_index") != layout.root_node_index)
    ).sort("x")
    return leaves["vertex_id"].to_list()


def _path_to_root(node: str, parent: dict[str, str]) -> list[str]:
    path = [node]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path


def peak_pair_distances(
    critical_nodes: pl.DataFrame, branches: pl.DataFrame
) -> pl.DataFrame:
    """One row per unordered peak pair, ``peak_1`` earlier in leaf order.

    Columns are listed in ``OUTPUT_COLUMNS``. Via-saddle Hamming distance is the
    distance from peak 1 to the common saddle plus that from the saddle to
    peak 2; via-saddle fitness distance is ``(f(p) - f(s)) + (f(q) - f(s))``.
    """
    fitness, parent, _, _ = _structure(critical_nodes, branches)
    peaks = peak_leaf_order(critical_nodes, branches)
    rows = []
    for (i, first), (j, second) in combinations(enumerate(peaks), 2):
        on_second = set(_path_to_root(second, parent))
        saddle = next(n for n in _path_to_root(first, parent) if n in on_second)
        amino = {n: translate(n) for n in (first, second, saddle)}
        rows.append(
            {
                "peak_1_dna": first,
                "peak_2_dna": second,
                "common_saddle_dna": saddle,
                "peak_1_leaf_order": i,
                "peak_2_leaf_order": j,
                "nucleotide_direct_hamming": hamming_distance(first, second),
                "nucleotide_via_saddle_hamming": hamming_distance(first, saddle)
                + hamming_distance(saddle, second),
                "amino_acid_direct_hamming": hamming_distance(
                    amino[first], amino[second]
                ),
                "amino_acid_via_saddle_hamming": hamming_distance(
                    amino[first], amino[saddle]
                )
                + hamming_distance(amino[saddle], amino[second]),
                "direct_fitness_difference": abs(fitness[first] - fitness[second]),
                "via_saddle_fitness_distance": (fitness[first] - fitness[saddle])
                + (fitness[second] - fitness[saddle]),
            }
        )
    schema = {
        name: pl.Int64
        if "hamming" in name or "leaf_order" in name
        else pl.Float64
        if "fitness" in name
        else pl.String
        for name in OUTPUT_COLUMNS
    }
    return pl.DataFrame(rows, schema=schema)


def _upper_triangle(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def mantel_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int,
    seed: int,
    chunk: int = PERMUTATION_CHUNK,
) -> tuple[float, float]:
    """Spearman rho of two symmetric distance matrices and its Mantel p-value.

    ``rho`` is the Pearson correlation of the ranks (ties share their mean rank)
    of the upper-triangle entries of ``x`` and ``y``. The null permutes the
    labels of ``y``, rows and columns together, ``permutations`` times with a
    ``numpy`` ``Generator`` seeded by ``seed``. The p-value is two-sided,
    ``(1 + #{|rho_perm| >= |rho_obs|}) / (1 + permutations)``; it never falls
    below ``1 / (1 + permutations)``. Ranks are taken once: a permuted ``y`` has
    the same ranks in a different order, so each permutation is one dot product.
    Permutations are drawn ``chunk`` at a time (row-major draws from one generator,
    so the result does not depend on ``chunk``). A constant matrix has no rank
    correlation: the result is ``(nan, nan)``.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("Mantel matrices must be square and of the same shape.")
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"A Mantel test needs at least 3 peaks, got {n}.")
    if permutations < 1:
        raise ValueError(f"permutations must be at least 1, got {permutations}.")
    rows, cols = _upper_triangle(n)
    x_ranks = rankdata(x[rows, cols])
    y_ranks = np.zeros((n, n))
    y_ranks[rows, cols] = rankdata(y[rows, cols])
    y_ranks = y_ranks + y_ranks.T
    x_centred = x_ranks - x_ranks.mean()
    y_centred = y_ranks[rows, cols] - y_ranks[rows, cols].mean()
    scale = float(np.linalg.norm(x_centred) * np.linalg.norm(y_centred))
    if scale == 0.0:
        return float("nan"), float("nan")
    observed = float(x_centred @ y_centred / scale)
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, permutations, chunk):
        size = min(chunk, permutations - start)
        order = np.argsort(rng.random((size, n)), axis=1)
        permuted = y_ranks[order[:, rows], order[:, cols]]
        null = (permuted - permuted.mean(axis=1, keepdims=True)) @ x_centred / scale
        extreme += int(np.count_nonzero(np.abs(null) >= abs(observed) - RHO_TOLERANCE))
    return observed, (1 + extreme) / (1 + permutations)


def _matrix(pairs: pl.DataFrame, values: pl.Series, n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    first = pairs["peak_1_leaf_order"].to_numpy()
    second = pairs["peak_2_leaf_order"].to_numpy()
    matrix[first, second] = values.to_numpy()
    return matrix + matrix.T


def peak_pair_mantel_table(
    pairs: pl.DataFrame, *, permutations: int, seed: int
) -> pl.DataFrame:
    """Mantel tests of Hamming against fitness distance, per alphabet.

    Three comparisons per alphabet: direct Hamming against direct fitness
    difference, via-saddle against via-saddle, and (via - direct) against
    (via - direct). Raw fitness; every test uses the same ``seed``.
    """
    n = max(pairs["peak_1_leaf_order"].max(), pairs["peak_2_leaf_order"].max(), -1)
    n = int(n) + 1
    fitness = {
        "direct": pairs["direct_fitness_difference"],
        "via_saddle": pairs["via_saddle_fitness_distance"],
    }
    fitness["via_minus_direct"] = fitness["via_saddle"] - fitness["direct"]
    rows = []
    for alphabet in ALPHABETS:
        hamming = {
            "direct": pairs[f"{alphabet}_direct_hamming"].cast(pl.Float64),
            "via_saddle": pairs[f"{alphabet}_via_saddle_hamming"].cast(pl.Float64),
        }
        hamming["via_minus_direct"] = hamming["via_saddle"] - hamming["direct"]
        for comparison in COMPARISONS:
            rho, p = mantel_test(
                _matrix(pairs, hamming[comparison], n),
                _matrix(pairs, fitness[comparison], n),
                permutations=permutations,
                seed=seed,
            )
            rows.append(
                {
                    "comparison": comparison,
                    "alphabet": alphabet,
                    "spearman_rho": rho,
                    "mantel_p": p,
                    "permutations": permutations,
                    "seed": seed,
                    "peak_count": n,
                    "pair_count": pairs.height,
                }
            )
    string_columns = {"comparison", "alphabet"}
    float_columns = {"spearman_rho", "mantel_p"}
    return pl.DataFrame(
        rows,
        schema={
            name: pl.String
            if name in string_columns
            else pl.Float64
            if name in float_columns
            else pl.Int64
            for name in MANTEL_COLUMNS
        },
    )


def peak_pair_distances_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``[peak_pair_distances]``; return the settings the stage records."""
    unknown = sorted(set(config) - set(SETTING_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown [{SECTION}] settings {unknown}; known: {list(SETTING_KEYS)}."
        )
    missing = [key for key in SETTING_KEYS if key not in config]
    if missing:
        raise ValueError(f"[{SECTION}] needs {missing}.")
    if config["tree"] != "band":
        raise ValueError(
            f"[{SECTION}] tree must be 'band' (peaks of the band-simplified tree), "
            f"got {config['tree']!r}."
        )
    for key in ("mantel_permutations", "seed"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"[{SECTION}] {key} must be an integer, got {value!r}.")
    if config["mantel_permutations"] < 1:
        raise ValueError(f"[{SECTION}] mantel_permutations must be at least 1.")
    return {key: config[key] for key in SETTING_KEYS}


def band_tree_tables(view: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``critical_nodes`` and ``branches`` of a band view, as ``save_band_tree`` writes.

    Only the identity, fitness and edge columns this module reads are needed, but
    the node attributes are kept as ``save_band_tree`` keeps them (minus Rustworkx
    indices), so the pair is interchangeable with the saved tree's tables.

    This is a deliberate copy of ``save_band_tree`` in ``analysis/leaves.py`` (its
    twin, which writes these tables to disk). Importing it instead would change
    the hash of ``leaves.py`` and invalidate the fingerprints of every stage that
    hashes that file. Keep the two in step.
    """
    graph = view.tree
    nodes = {index: dict(graph[index]) for index in graph.node_indices()}
    critical = pl.DataFrame(
        [
            {
                "tree_node_identity": node["dna_sequence"],
                **{
                    key: value
                    for key, value in node.items()
                    if not key.endswith("_index")
                },
            }
            for node in nodes.values()
        ]
    )
    branches = pl.DataFrame(
        [
            {
                "source_tree_node_identity": nodes[upper]["dna_sequence"],
                "target_tree_node_identity": nodes[lower]["dna_sequence"],
                "source_scalar": float(nodes[upper]["scalar"]),
                "target_scalar": float(nodes[lower]["scalar"]),
            }
            for upper, lower in graph.edge_list()
        ],
        schema={
            "source_tree_node_identity": pl.String,
            "target_tree_node_identity": pl.String,
            "source_scalar": pl.Float64,
            "target_scalar": pl.Float64,
        },
    )
    return critical, branches


def peak_pair_tables(
    critical_nodes: pl.DataFrame,
    branches: pl.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The pair table and the Mantel table of one tree."""
    section = peak_pair_distances_settings(settings)
    pairs = peak_pair_distances(critical_nodes, branches)
    mantel = peak_pair_mantel_table(
        pairs, permutations=section["mantel_permutations"], seed=section["seed"]
    )
    return pairs, mantel


@dataclass(frozen=True, slots=True)
class PeakPairDistancesResult:
    """The published tables of the stage with its provenance record."""

    pairs: pl.DataFrame
    mantel: pl.DataFrame
    provenance: dict[str, Any]
    reused: bool
    path: Path


def _stage(section: Mapping[str, Any], bootstrap: Any, settings: Mapping[str, Any]):
    """Declare what the checkpoint depends on, through the project's stage factory."""
    from ..landscape.selection import candidate_selection_source
    from ..threshold.functional_threshold import (
        DECLARATION_KEY,
        configured_declaration,
        declaration_identity,
    )

    analysis_dir = Path(__file__).resolve().parent
    package = analysis_dir.parent
    simplification_dir = package / "simplification"
    landscape_dir = package / "landscape"
    return project_stage(
        STAGE_NAME,
        settings={
            SECTION: dict(section),
            "bootstrap_checkpoint": upstream_identity(bootstrap.provenance),
            DECLARATION_KEY: declaration_identity(configured_declaration(settings)),
            "candidate_selection_source": candidate_selection_source(settings),
            "topographer_sources": package_source_hashes(topographer),
        },
        code=(
            Path(__file__).resolve(),
            analysis_dir / "branch_decomposition.py",
            analysis_dir / "peak_rank.py",
            package / "sequence_distances.py",
            simplification_dir / "__init__.py",
            simplification_dir / "bootstrap_smoothed_tree.py",
            simplification_dir / "bootstrap_chunks.py",
            simplification_dir / "checkpoint_identity.py",
            simplification_dir / "identities.py",
            simplification_dir / "selected_leaf_simplification.py",
            landscape_dir / "selection.py",
            landscape_dir / "selected_landscape.py",
            landscape_dir / "full_population.py",
            package / "threshold" / "functional_threshold.py",
            analysis_dir / "candidate_tree_contrast.py",
            ROOT / "uv.lock",
        ),
        root=ROOT,
    )


def _checkpoint_root(context: Any) -> Path:
    return Path(context.output_root) / "analysis" / SECTION


def _read_result(checkpoint: Checkpoint) -> PeakPairDistancesResult:
    directory = checkpoint.path
    return PeakPairDistancesResult(
        pairs=pl.read_parquet(directory / ARTEFACTS[0]),
        mantel=pl.read_parquet(directory / ARTEFACTS[1]),
        provenance=checkpoint.provenance,
        reused=checkpoint.reused,
        path=directory,
    )


def compute_peak_pair_distances(
    context: Any, *, force: bool = False
) -> PeakPairDistancesResult:
    """Produce the pair and Mantel tables, or load the matching checkpoint.

    The bootstrap checkpoint is only read (it never draws); the Candidate split
    tree is built only when a checkpoint has to be produced. Heavy then: it builds
    the Full graph.
    """
    from ..simplification.bootstrap_smoothed_tree import (
        check_hills_match_tree,
        load_cached_bootstrap_smoothed_tree,
    )
    from .branch_decomposition import build_candidate_inputs, decomposition_view

    settings = load_settings(context)
    section = peak_pair_distances_settings(settings["analysis"][SECTION])
    bootstrap = load_cached_bootstrap_smoothed_tree(settings, context=context)

    def produce(directory: Path) -> None:
        tree_analysis, _ = build_candidate_inputs(settings)
        check_hills_match_tree(bootstrap.hills, tree_analysis)
        view = decomposition_view(section["tree"], tree_analysis, bootstrap.hills)
        critical, branches = band_tree_tables(view)
        pairs, mantel = peak_pair_tables(critical, branches, section)
        pairs.write_parquet(directory / ARTEFACTS[0])
        mantel.write_parquet(directory / ARTEFACTS[1])
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "settings": section,
                    "variant": getattr(context, "variant_name", None),
                    "peak_count": int(mantel["peak_count"][0]),
                    "pair_count": pairs.height,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    read: list[PeakPairDistancesResult] = []
    stage = _stage(section, bootstrap, settings)
    produce_or_load(
        stage,
        produce,
        checkpoint_dir=result_directory(_checkpoint_root(context), stage),
        force=force,
        on_ready=lambda ready: read.append(_read_result(ready)),
    )
    return read[0]


def load_cached_peak_pair_distances(context: Any) -> PeakPairDistancesResult:
    """Read the checkpoint for the configured settings; never compute."""
    from ..simplification.bootstrap_smoothed_tree import (
        load_cached_bootstrap_smoothed_tree,
    )

    settings = load_settings(context)
    section = peak_pair_distances_settings(settings["analysis"][SECTION])
    bootstrap = load_cached_bootstrap_smoothed_tree(settings, context=context)
    stage = _stage(section, bootstrap, settings)
    return load_checkpoint(
        stage,
        _read_result,
        checkpoint_dir=result_directory(_checkpoint_root(context), stage),
        retry_hint=RETRY_HINT,
    )
