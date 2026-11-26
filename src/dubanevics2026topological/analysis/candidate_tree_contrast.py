"""Tree-edge contrasts for the Candidate split tree with a locked checkpoint.

Every edge of the Candidate split tree joins two critical points. Each edge is
fitted as one direct BarSeqFit pair contrast (``from`` is the lower endpoint,
``to`` the upper endpoint), and the Wald p-values are adjusted together as one
Benjamini-Hochberg family over tree edges only (ADR 0006, superseded by ADR 0008).
The adjustment is nominal: edges are selected from observed fitness and the tests
are dependent. The contrast-smoothed tree that ADR 0006 built on these q-values is
retired; the bootstrap stage no longer reads them (its singleton check was removed,
ADR 2026-10-22-pooled-standard-error-for-bootstrap).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import barseqfit
import numpy as np
import polars as pl
from dubanevics2026topological.papkou2023 import Papkou2023Workflow
from topographer.statistics import compute_bh_qvalues

from ..landscape.selected_landscape import (
    PreparedLandscape,
    analyze_prepared_landscape,
)
from ..landscape.selection import (
    CandidateSelection,
    candidate_selection_fingerprint,
    prepare_candidate_landscape,
)
from . import (
    ROOT,
    dataset_input_key,
    dataset_input_path,
    get_run_context,
    load_settings,
    source_key,
)
from .barseq_inputs import (
    BarseqExport,
    _quoted,
    barseqfit_version,
    load_barseq_export,
    map_sequence_ids,
)
from .checkpoint import checkpoint_lock, publish_checkpoint, recover_checkpoint


def cache_paths(context: Any | None = None) -> tuple[Path, Path]:
    """The contrast checkpoint directory and its pending marker for ``context``."""
    analysis = Path((context or get_run_context()).output_root) / "analysis"
    return (
        analysis / "candidate_tree_contrast",
        analysis / ".candidate-tree-contrast.pending",
    )


CACHE_SCHEMA = 1
BH_FAMILY = "candidate_split_tree_edges"
QUALIFICATION_RULE = "strict_q_greater_than_alpha"
# barseqfit's pair-fit vocabulary is "converged", "not_converged" and "failed";
# only "converged" is accepted. ("reference" belongs to fit_wt only.)
OK_STATUS = "converged"
ARTIFACTS = (
    "pairs.parquet",
    "pairs.csv",
    "edges.parquet",
    "edges.csv",
    "diagnostics.json",
)
PAIR_COLUMNS = (
    "edge_id",
    "upper_sequence",
    "lower_sequence",
    "upper_event",
    "lower_event",
    "upper_fitness",
    "lower_fitness",
    "raw_fitness_delta",
    "regular_count",
    "is_terminal",
    "lower_in_degree",
)


@dataclass(slots=True)
class CandidateTreeContrastResult:
    """Pair list, evidence table, diagnostics and provenance of one checkpoint."""

    pairs: pl.DataFrame
    edges: pl.DataFrame
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


def enumerate_tree_edges(tree_analysis: Any) -> pl.DataFrame:
    """List every split-tree edge, upper (higher) endpoint first, in a stable order."""
    graph = tree_analysis.tree
    nodes = graph.node_indices()
    roots = [index for index in nodes if graph.out_degree(index) == 0]
    if len(roots) != 1:
        described = ", ".join(
            f"(event={graph[index].get('event')!r}, "
            f"scalar={graph[index].get('scalar')!r}, "
            f"dna_sequence={graph[index].get('dna_sequence')!r})"
            for index in roots
        )
        raise ValueError(
            f"Candidate split tree must have one root, found {len(roots)}: {described}."
        )
    missing = [index for index in nodes if not graph[index].get("dna_sequence")]
    if missing:
        raise ValueError(f"Tree nodes {missing} lack a dna_sequence identity.")
    rows = []
    for upper, lower, payload in graph.weighted_edge_list():
        top, bottom = graph[upper], graph[lower]
        rows.append(
            {
                "edge_id": f"{top['dna_sequence']}|{bottom['dna_sequence']}",
                "upper_sequence": top["dna_sequence"],
                "lower_sequence": bottom["dna_sequence"],
                "upper_event": top["event"],
                "lower_event": bottom["event"],
                "upper_fitness": float(top["scalar"]),
                "lower_fitness": float(bottom["scalar"]),
                "raw_fitness_delta": float(top["scalar"]) - float(bottom["scalar"]),
                "regular_count": int(payload["regular_count"]),
                "is_terminal": graph.in_degree(upper) == 0
                and top["event"] == "maximum",
                "lower_in_degree": graph.in_degree(lower),
            }
        )
    if not rows:
        raise ValueError("Candidate split tree has no edges to test.")
    inverted = sorted(
        row["edge_id"] for row in rows if row["upper_fitness"] < row["lower_fitness"]
    )
    if inverted:
        raise ValueError(
            "Tree edges have upper_fitness below lower_fitness: " + _quoted(inverted)
        )
    edges = pl.DataFrame(
        rows,
        schema={
            "edge_id": pl.String,
            "upper_sequence": pl.String,
            "lower_sequence": pl.String,
            "upper_event": pl.String,
            "lower_event": pl.String,
            "upper_fitness": pl.Float64,
            "lower_fitness": pl.Float64,
            "raw_fitness_delta": pl.Float64,
            "regular_count": pl.Int64,
            "is_terminal": pl.Boolean,
            "lower_in_degree": pl.Int64,
        },
    ).sort("edge_id")
    if edges["edge_id"].n_unique() != edges.height:
        raise ValueError("Candidate split tree has duplicate edge identities.")
    return edges


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1].")
    return float(alpha)


def _map_sequence_ids(pairs: pl.DataFrame, sequences: pl.DataFrame) -> pl.DataFrame:
    """Attach BarSeq ``from``/``to`` ids to tree edges, naming edges on failure."""

    def edges_of(bad: set[str]) -> str:
        owned = pairs.filter(
            pl.col("upper_sequence").is_in(bad) | pl.col("lower_sequence").is_in(bad)
        )["edge_id"]
        return "edges " + _quoted(sorted(owned))

    ids = map_sequence_ids(
        sequences,
        set(pairs["upper_sequence"]) | set(pairs["lower_sequence"]),
        describe=edges_of,
    )
    return pairs.with_columns(
        pl.col("lower_sequence")
        .replace_strict(ids, return_dtype=pl.String)
        .alias("from"),
        pl.col("upper_sequence")
        .replace_strict(ids, return_dtype=pl.String)
        .alias("to"),
    )


def _check_fits(pairs: pl.DataFrame, fits: pl.DataFrame) -> pl.DataFrame:
    """Return fits aligned to pairs, or raise naming every offending edge."""
    required = {
        "from",
        "to",
        "m",
        "se",
        "p",
        "status",
        "diagnostics",
        "elapsed_time",
        "time_unit",
    }
    if not required.issubset(fits.columns):
        raise ValueError(
            f"Pair fit lacks columns {sorted(required - set(fits.columns))}."
        )
    key = pairs.select("edge_id", "from", "to")
    counts = fits.group_by("from", "to").len()
    matched = key.join(
        counts, on=["from", "to"], how="left", maintain_order="left"
    ).with_columns(pl.col("len").fill_null(0))
    problems: dict[str, list[str]] = {
        "missing fit": matched.filter(pl.col("len") == 0)["edge_id"].to_list(),
        "duplicate fit": matched.filter(pl.col("len") > 1)["edge_id"].to_list(),
    }
    extra = fits.join(key, on=["from", "to"], how="anti")
    if extra.height:
        raise ValueError(f"Pair fit returned {extra.height} rows for untested pairs.")
    if not any(problems.values()):
        joined = key.join(fits, on=["from", "to"], how="left", maintain_order="left")
        m, se, p = (joined[name].cast(pl.Float64) for name in ("m", "se", "p"))
        problems["status is not " + repr(OK_STATUS)] = joined.filter(
            pl.col("status").is_null() | (pl.col("status") != OK_STATUS)
        )["edge_id"].to_list()
        problems["non-finite contrast estimate"] = joined.filter(
            ~m.is_finite().fill_null(False)
        )["edge_id"].to_list()
        problems["standard error not finite and positive"] = joined.filter(
            ~(se.is_finite() & (se > 0)).fill_null(False)
        )["edge_id"].to_list()
        problems["p-value outside [0, 1]"] = joined.filter(
            ~(p.is_finite() & (p >= 0) & (p <= 1)).fill_null(False)
        )["edge_id"].to_list()
    failed = {reason: ids for reason, ids in problems.items() if ids}
    if failed:
        raise ValueError(
            "Tree-edge contrast fit failed closed before adjustment: "
            + "; ".join(
                f"{reason}: {_quoted(sorted(ids))}" for reason, ids in failed.items()
            )
        )
    return key.join(fits, on=["from", "to"], how="left", maintain_order="left")


def _diagnostics_json(value: Any) -> str:
    """Serialise one fit-diagnostics record as strict JSON (NaN/inf raise)."""
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _sign_disagreement(m: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Flag edges whose contrast and raw fitness delta have opposite signs.

    A zero on either side is no evidence of disagreement, so the flag is
    ``sign(m) != sign(delta)`` only when both ``m`` and ``delta`` are nonzero.
    """
    return (m != 0) & (delta != 0) & (np.sign(m) != np.sign(delta))


def build_candidate_tree_contrast(
    tree_analysis: Any,
    wt: pl.DataFrame,
    export: BarseqExport,
    config: dict[str, Any],
) -> CandidateTreeContrastResult:
    """Fit, validate and adjust one contrast per tree edge of one split tree."""
    alpha = _validate_alpha(config["alpha"])
    if config.get("correction") != "benjamini_hochberg" or (
        config.get("family") != BH_FAMILY
    ):
        raise ValueError("Tree-edge contrast requires the fixed tree-edge BH family.")
    enumerated = enumerate_tree_edges(tree_analysis)
    pairs = _map_sequence_ids(enumerated, export.sequences)
    fits = barseqfit.fit_pair(
        export.counts,
        export.samples,
        export.sequences,
        pairs=pairs.select("from", "to"),
        elapsed_time=config["elapsed_time"],
        time_unit=config["time_unit"],
        # 0.2.0 requires these; they were the defaults of the previous pin
        # (every zero count replaced by 1; binomial dispersion fixed at 1).
        pseudocount="all_zeros",
        dispersion="fixed",
    )
    fitted = _check_fits(pairs, fits)
    p_values = fitted["p"].to_numpy().astype(float)
    q_values = np.asarray(compute_bh_qvalues(p_values), dtype=float)
    if q_values.shape != p_values.shape:
        raise ValueError("BH adjustment must return one q-value per tree edge.")

    if not {"dna_to", "se"}.issubset(wt.columns):
        raise ValueError("fitness_wt must supply dna_to and se.")
    needed = set(pairs["upper_sequence"]) | set(pairs["lower_sequence"])
    measured = wt.filter(pl.col("dna_to").is_in(list(needed))).select("dna_to", "se")
    if measured["dna_to"].n_unique() != measured.height:
        raise ValueError("Duplicate endpoint measurements in fitness_wt.")
    endpoint_se = dict(zip(measured["dna_to"], measured["se"], strict=True))
    for identity, value in endpoint_se.items():
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid endpoint uncertainty for {identity!r}.")

    delta = pairs["raw_fitness_delta"].to_numpy()
    m = fitted["m"].to_numpy().astype(float)
    edges = pairs.with_columns(
        pl.col("upper_sequence")
        .replace_strict(endpoint_se, default=None, return_dtype=pl.Float64)
        .alias("fitness_standard_error_upper"),
        pl.col("lower_sequence")
        .replace_strict(endpoint_se, default=None, return_dtype=pl.Float64)
        .alias("fitness_standard_error_lower"),
        pl.Series("contrast_estimate", m),
        fitted["se"].alias("contrast_standard_error"),
        fitted["p"].alias("p_value"),
        pl.Series("q_value", q_values),
        pl.Series("qualifies", q_values > alpha),
        pl.Series("contrast_gap", m - delta),
        pl.Series("sign_disagreement", _sign_disagreement(m, delta)),
        fitted["status"].alias("status"),
        fitted["elapsed_time"].alias("elapsed_time"),
        fitted["time_unit"].alias("time_unit"),
        fitted["diagnostics"]
        .map_elements(_diagnostics_json, return_dtype=pl.String)
        .alias("diagnostics"),
    )
    disagreements = edges.filter(pl.col("sign_disagreement"))["edge_id"].to_list()
    graph = tree_analysis.tree
    diagnostics = {
        "tree_node_count": graph.num_nodes(),
        "tree_edge_count": graph.num_edges(),
        "test_count": edges.height,
        "alpha": alpha,
        "bh_family": BH_FAMILY,
        "bh_family_size": edges.height,
        "qualification_rule": QUALIFICATION_RULE,
        "qualifying_count": int(edges["qualifies"].sum()),
        "terminal_qualifying_count": int(
            edges.filter(pl.col("is_terminal"))["qualifies"].sum()
        ),
        "sign_disagreement_count": len(disagreements),
        "sign_disagreement_edge_ids": disagreements,
        "ok_status": OK_STATUS,
        "elapsed_time": float(config["elapsed_time"]),
        "time_unit": config["time_unit"],
        "barseqfit": barseqfit_version(),
        "adjustment_note": "nominal: edges selected from observed fitness; "
        "tests are dependent",
    }
    return CandidateTreeContrastResult(
        pairs=pairs.select(*PAIR_COLUMNS, "from", "to"),
        edges=edges,
        diagnostics=diagnostics,
        provenance={},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_without_export(
    settings: dict[str, Any], *, context: Any | None = None
) -> dict[str, Any]:
    """Fingerprint of everything except the BarSeq export files."""
    scientific = settings["analysis"]
    files = {
        dataset_input_key(scientific["dataset"][key], path): _sha256(path)
        for key in ("fitness_wt", "fitness_diff")
        for path in (
            dataset_input_path(scientific["dataset"][key], context, root=ROOT),
        )
    }
    files |= {
        source_key(path): _sha256(path)
        for path in (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1] / "landscape" / "selection.py",
            Path(__file__).resolve().parents[1]
            / "threshold"
            / "functional_threshold.py",
            Path(__file__).resolve().parents[1] / "landscape" / "selected_landscape.py",
            Path(__file__).resolve().parents[1] / "landscape" / "full_population.py",
            Path(__file__).with_name("barseq_inputs.py").resolve(),
            ROOT / "uv.lock",
        )
    }
    return {
        "schema": CACHE_SCHEMA,
        "config": scientific["candidate_tree_contrast"],
        "candidate_functional_threshold": scientific["clamped"][
            "candidate_functional_threshold"
        ],
        "candidate_selection": candidate_selection_fingerprint(
            settings, context=context
        ),
        "files": files,
    }


def _fingerprint(
    settings: dict[str, Any], export: BarseqExport, *, context: Any | None = None
) -> dict[str, Any]:
    fingerprint = _fingerprint_without_export(settings, context=context)
    for name in ("counts", "samples", "sequences"):
        fingerprint["files"][f"barseq_export/{name}.parquet"] = _sha256(
            export.source_dir / f"{name}.parquet"
        )
    return fingerprint


def _read_checkpoint(path: Path) -> CandidateTreeContrastResult:
    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    for name in ARTIFACTS:
        if _sha256(path / name) != provenance["artifacts"][name]:
            raise ValueError(f"Tree-contrast artifact hash mismatch: {name}.")
    pairs = pl.read_parquet(path / "pairs.parquet")
    edges = pl.read_parquet(path / "edges.parquet")
    diagnostics = json.loads((path / "diagnostics.json").read_text(encoding="utf-8"))
    if (
        edges.height != diagnostics["test_count"]
        or edges.height != diagnostics["tree_edge_count"]
        or pairs.height != edges.height
        or edges["edge_id"].n_unique() != edges.height
        or set(pairs["edge_id"]) != set(edges["edge_id"])
    ):
        raise ValueError("Tree-contrast coverage mismatch.")
    return CandidateTreeContrastResult(pairs, edges, diagnostics, provenance)


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    result = _read_checkpoint(path)
    return {"result": result, "fingerprint": result.provenance["fingerprint"]}


def _write_checkpoint(
    result: CandidateTreeContrastResult, stage: Path, fingerprint: dict[str, Any]
) -> None:
    result.pairs.write_parquet(stage / "pairs.parquet")
    result.pairs.write_csv(stage / "pairs.csv")
    result.edges.write_parquet(stage / "edges.parquet")
    result.edges.write_csv(stage / "edges.csv")
    (stage / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    provenance = {
        "fingerprint": fingerprint,
        "artifacts": {name: _sha256(stage / name) for name in ARTIFACTS},
    }
    (stage / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result.provenance = provenance


def build_candidate_network(
    settings: dict[str, Any],
    *,
    selection: CandidateSelection | None = None,
    context: Any | None = None,
) -> tuple[PreparedLandscape, pl.DataFrame]:
    """Build the clamped Candidate network (no tree) and read ``fitness_wt``.

    ``selection`` is an already resolved Candidate selection (see
    ``prepare_candidate_landscape``); without it the selection is resolved here.

    The prepared landscape's ``network`` holds every Candidate vertex with its
    raw fitness; identities are ``dna_sequence`` values.
    """
    scientific = settings["analysis"]
    wt = pl.read_parquet(
        dataset_input_path(scientific["dataset"]["fitness_wt"], context, root=ROOT)
    )
    diff = pl.read_parquet(
        dataset_input_path(scientific["dataset"]["fitness_diff"], context, root=ROOT)
    )
    network = Papkou2023Workflow(
        landscape="full", fitness_wt=wt, fitness_diff=diff
    ).build_network()
    prepared, _ = prepare_candidate_landscape(
        network, settings, selection=selection, context=context
    )
    return prepared, wt


def build_candidate_tree(
    settings: dict[str, Any],
    *,
    selection: CandidateSelection | None = None,
    context: Any | None = None,
) -> tuple[Any, pl.DataFrame]:
    """Build the Candidate split tree in memory from the raw fitness tables."""
    prepared, wt = build_candidate_network(
        settings, selection=selection, context=context
    )
    analysis = analyze_prepared_landscape(
        prepared,
        name="candidate_clamped",
        wild_type_sequence=str(wt.get_column("dna_from")[0]),
    )
    return analysis.trees["split"], wt


def load_cached_candidate_tree_contrast(
    *, context: Any | None = None
) -> CandidateTreeContrastResult:
    """Read only a completed checkpoint; never compute or publish."""
    cache_dir, pending_path = cache_paths(context)
    if not (cache_dir / "provenance.json").is_file():
        raise FileNotFoundError(
            f"No tree-edge contrast checkpoint at {cache_dir}; run "
            "`just candidate-tree-contrast`."
        )
    with checkpoint_lock(cache_dir, "Candidate tree contrast"):
        if pending_path.exists():
            raise RuntimeError(
                "Candidate tree contrast refresh is pending; rerun "
                "`just candidate-tree-contrast`."
            )
        return _read_checkpoint(cache_dir)


def compute_candidate_tree_contrast(
    settings: dict[str, Any] | None = None,
    *,
    force: bool = False,
    context: Any | None = None,
) -> CandidateTreeContrastResult:
    """Load the current checkpoint or fit, adjust and atomically publish a new one."""
    settings = settings or load_settings(context=context)
    export = (
        load_barseq_export(settings)
        if context is None
        else load_barseq_export(settings, context=context)
    )
    fingerprint = _fingerprint(settings, export, context=context)
    label = "Candidate tree contrast"
    cache_dir, pending_path = cache_paths(context)
    with checkpoint_lock(cache_dir, label):
        recovered = recover_checkpoint(
            cache_dir,
            pending_path,
            label,
            _validate_checkpoint,
            expected_fingerprint=fingerprint,
            fingerprint_of=lambda value: value["fingerprint"],
        )
        if (
            not force
            and recovered is not None
            and recovered["fingerprint"] == fingerprint
        ):
            return recovered["result"]

        tree_analysis, wt = build_candidate_tree(settings, context=context)
        result = build_candidate_tree_contrast(
            tree_analysis, wt, export, settings["analysis"]["candidate_tree_contrast"]
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("Candidate tree contrast refresh pending\n")
        stage = Path(
            tempfile.mkdtemp(prefix=".candidate-tree-contrast-", dir=cache_dir.parent)
        )
        try:
            _write_checkpoint(result, stage, fingerprint)
            publish_checkpoint(
                stage,
                cache_dir,
                pending_path,
                label,
                _validate_checkpoint,
                expected_fingerprint=fingerprint,
                fingerprint_of=lambda value: value["fingerprint"],
            )
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return result
