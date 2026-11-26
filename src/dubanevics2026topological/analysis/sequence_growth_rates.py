"""Per-sequence growth rates for every sequence of the published fitness table.

The stage fits all ``dna_to`` sequences of ``[dataset] fitness_wt`` (261,332
measured sequences) plus the wild type (``dna_from``), before and independently
of any threshold or Candidate selection: it reads no Candidate network. Each
sequence gets its own growth rate ``r`` (the slope per ``time_unit`` of
``logit(count / sample_total)``) from ``barseqfit.fit_sequences``, with a pooled
and an across-replicate standard error. The published wild-type-relative
fitness ``s`` must equal ``r_i - r_wt``: that identity is checked for every
sequence and the stage fails closed on any gap above ``identity_tolerance``
(ADR 0008). A sequence of the published table that is absent from the BarSeq
export fails closed, named. Identities are ``dna_sequence`` values throughout;
the BarSeq ``sequence_id`` is carried only as a column.

Columns of the table (``GROWTH_COLUMNS``):

``dna_sequence``
    The sequence identity.
``sequence_id``
    The BarSeq export id, carried only as a label.
``r``
    Growth rate per ``time_unit`` (pooled fit over all replicates).
``s``
    ``r - r_wt`` from the fitted ``r`` (0 for the wild type by construction),
    not the published ``s`` copied through; the two agree within
    ``identity_tolerance``.
``se_pooled``
    Pooled standard error of ``r`` (see the glossary).
``r_replicate_mean``, ``r_replicate_sd``, ``se_replicate``, ``n_replicates``
    The per-replicate fits and the replicate standard error, a diagnostic.
``status``
    barseqfit's fit status; the stage fails closed unless every row is
    ``converged``.
``stop_codon_count``
    Number of in-frame codons of ``dna_sequence`` in the configured stop-codon
    list (``[codon_entropy] stop_codons``); see ``stop_codon_count``.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import barseqfit
import numpy as np
import polars as pl

from . import OUTPUT, ROOT, load_settings, package_source_hashes, source_key
from .barseq_inputs import (
    BarseqExport,
    _quoted,
    barseqfit_version,
    load_barseq_export,
    map_sequence_ids,
)
from .checkpoint import (
    _sha256,
    checkpoint_lock,
    publish_checkpoint,
    recover_checkpoint,
)

SECTION = "sequence_growth_rates"
CACHE_DIR = OUTPUT / "analysis/sequence_growth_rates"
PENDING_PATH = OUTPUT / "analysis/.sequence-growth-rates.pending"
CACHE_SCHEMA = 2
# barseqfit's fit vocabulary is "converged", "not_converged" and "failed".
OK_STATUS = "converged"
# A replicate SE more than this many times the pooled SE flags overdispersion.
RATIO_THRESHOLD = 2.0
ARTIFACTS = ("growth_rates.parquet", "diagnostics.json")
GROWTH_COLUMNS = (
    "dna_sequence",
    "sequence_id",
    "r",
    "s",
    "se_pooled",
    "r_replicate_mean",
    "r_replicate_sd",
    "se_replicate",
    "n_replicates",
    "status",
    "stop_codon_count",
)
FIT_COLUMNS = (
    "sequence_id",
    "r",
    "se_pooled",
    "r_replicate_mean",
    "r_replicate_sd",
    "se_replicate",
    "n_replicates",
    "n_replicates_converged",
    "status",
)


@dataclass(slots=True)
class SequenceGrowthRatesResult:
    """Growth-rate table, diagnostics and provenance of one checkpoint."""

    growth_rates: pl.DataFrame
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


def load_fitness_wt(
    settings: dict[str, Any], context: Any | None = None
) -> pl.DataFrame:
    """Read the published wild-type-relative fitness table (``fitness_wt``)."""
    raw_rel = settings["analysis"]["dataset"]["fitness_wt"]
    if context is not None:
        raw_path = context.raw_data_dir / Path(raw_rel).name
        if not raw_path.is_file():
            raw_path = context.root / raw_rel
    else:
        raw_path = ROOT / raw_rel
    return pl.read_parquet(raw_path)


def _validate_stop_codons(stop_codons: Iterable[str]) -> frozenset[str]:
    codons = list(stop_codons)
    bad = sorted(
        {
            repr(codon)
            for codon in codons
            if not isinstance(codon, str) or len(codon) != 3
        }
    )
    if bad or not codons:
        raise ValueError(
            "Stop codons must be a non-empty list of 3-nt strings; got "
            + (", ".join(bad) if bad else "an empty list")
            + "."
        )
    return frozenset(codons)


def _count_stops(sequence: str, stops: frozenset[str]) -> int:
    """Count in-frame stops against an already validated stop-codon set."""
    if len(sequence) % 3:
        raise ValueError(
            f"Sequence {sequence!r} has length {len(sequence)}, not a multiple of 3."
        )
    return sum(sequence[i : i + 3] in stops for i in range(0, len(sequence), 3))


def stop_codon_count(sequence: str, stop_codons: Iterable[str]) -> int:
    """Count the in-frame codons of ``sequence`` that are in ``stop_codons``.

    The frame starts at the first base; every stop is counted, not only the
    first. Fails closed (``ValueError``) when the sequence length is not a
    multiple of 3 or a stop codon is not 3 nt long.
    """
    return _count_stops(sequence, _validate_stop_codons(stop_codons))


def _validate_config(config: dict[str, Any]) -> float:
    tolerance = config["identity_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("identity_tolerance must be finite and non-negative.")
    return float(tolerance)


def _published_s(wt: pl.DataFrame, wild_type: str) -> dict[str, float]:
    """Return the published wild-type-relative ``s`` by ``dna_sequence``."""
    if not {"dna_from", "dna_to", "s"}.issubset(wt.columns):
        raise ValueError("fitness_wt must supply dna_from, dna_to and s.")
    origins = set(wt["dna_from"].unique().to_list())
    if origins != {wild_type}:
        raise ValueError(
            "fitness_wt must measure every sequence against the one wild type "
            f"{wild_type!r}; found origins {_quoted(sorted(origins))}."
        )
    if wt["dna_to"].null_count():
        raise ValueError("fitness_wt has rows with a null dna_to.")
    self_rows = wt.filter(pl.col("dna_to") == pl.col("dna_from"))["dna_to"]
    if self_rows.len():
        raise ValueError(
            "fitness_wt has a row of the wild type against itself (dna_to equals "
            f"dna_from): {_quoted(self_rows.to_list())}."
        )
    duplicated = wt.filter(pl.col("dna_to").is_duplicated())["dna_to"].unique()
    if duplicated.len():
        raise ValueError(
            "Duplicate fitness_wt rows for sequences "
            + _quoted(sorted(duplicated.to_list()))
        )
    return dict(zip(wt["dna_to"], wt["s"], strict=True))


def _check_fits(fits: pl.DataFrame, expected: dict[str, str]) -> pl.DataFrame:
    """Return fits keyed by ``sequence_id``, or raise naming every offender.

    ``expected`` maps ``sequence_id`` to ``dna_sequence`` so the error lists
    sequence identities, never export-internal ids.
    """
    missing_columns = sorted(set(FIT_COLUMNS) - set(fits.columns))
    if missing_columns:
        raise ValueError(f"Growth-rate fit lacks columns {missing_columns}.")
    seen = fits.group_by("sequence_id").len()
    extra = set(seen["sequence_id"]) - expected.keys()
    if extra:
        raise ValueError(
            f"Growth-rate fit returned {len(extra)} rows for unrequested sequences."
        )
    counts = dict(zip(seen["sequence_id"], seen["len"], strict=True))

    def sequences(ids: Iterable[str]) -> list[str]:
        return sorted(expected[sequence_id] for sequence_id in ids)

    problems: dict[str, list[str]] = {
        "missing fit": sequences(i for i in expected if counts.get(i, 0) == 0),
        "duplicate fit": sequences(i for i in expected if counts.get(i, 0) > 1),
    }
    if not any(problems.values()):
        r, se_pooled, se_replicate = (
            fits[name].cast(pl.Float64) for name in ("r", "se_pooled", "se_replicate")
        )

        def bad(mask: pl.Series) -> list[str]:
            return sequences(fits.filter(mask)["sequence_id"])

        problems["status is not " + repr(OK_STATUS)] = bad(
            (fits["status"].is_null() | (fits["status"] != OK_STATUS)).fill_null(True)
        )
        problems["growth rate not finite"] = bad(~r.is_finite().fill_null(False))
        problems["pooled standard error not finite and positive"] = bad(
            ~(se_pooled.is_finite() & (se_pooled > 0)).fill_null(False)
        )
        problems["replicate standard error missing or not finite and non-negative"] = (
            bad(~(se_replicate.is_finite() & (se_replicate >= 0)).fill_null(False))
        )
    failed = {reason: ids for reason, ids in problems.items() if ids}
    if failed:
        raise ValueError(
            "Growth-rate fit failed closed: "
            + "; ".join(f"{reason}: {_quoted(ids)}" for reason, ids in failed.items())
        )
    return fits


def build_sequence_growth_rates(
    wt: pl.DataFrame,
    export: BarseqExport,
    config: dict[str, Any],
    stop_codons: Iterable[str],
) -> SequenceGrowthRatesResult:
    """Fit ``r`` for every published sequence plus the wild type.

    The set is every ``dna_to`` of ``wt`` and its ``dna_from``; ``r_i - r_wt``
    is checked against the published ``s_i``.
    """
    tolerance = _validate_config(config)
    if wt.is_empty():
        raise ValueError("fitness_wt is empty.")
    wild_type = str(wt.get_column("dna_from")[0])
    published = _published_s(wt, wild_type)
    stops = _validate_stop_codons(stop_codons)
    wanted = {*published, wild_type}
    ragged = sorted(dna for dna in wanted if len(dna) % 3)
    if ragged:
        raise ValueError(
            "Sequence length is not a multiple of 3 for: " + _quoted(ragged)
        )

    ids = map_sequence_ids(export.sequences, wanted)
    dna_of = {sequence_id: dna for dna, sequence_id in ids.items()}
    if len(dna_of) != len(ids):
        raise ValueError("BarSeq export maps several sequences to one sequence_id.")
    ordered = sorted(wanted)
    sequences = pl.DataFrame(
        {"sequence_id": [ids[dna] for dna in ordered], "sequence": ordered},
        schema={"sequence_id": pl.String, "sequence": pl.String},
    )
    counts = export.counts.filter(
        pl.col("sequence_id").is_in(sequences["sequence_id"].to_list())
    )
    fits = _check_fits(
        barseqfit.fit_sequences(
            counts,
            export.samples,
            sequences,
            elapsed_time=config["elapsed_time"],
            time_unit=config["time_unit"],
            # 0.2.0 requires the published zero-replacement rule as a keyword.
            # It was the default of the previous pin.
            pseudocount="all_zeros",
        ),
        dna_of,
    )

    table = (
        fits.select(*FIT_COLUMNS)
        .with_columns(
            pl.col("sequence_id")
            .replace_strict(dna_of, return_dtype=pl.String)
            .alias("dna_sequence")
        )
        .sort("dna_sequence")
    )
    r_wt = float(table.filter(pl.col("dna_sequence") == wild_type)["r"][0])
    table = table.with_columns(
        (pl.col("r") - r_wt).alias("s"),
        pl.Series(
            "stop_codon_count",
            [_count_stops(dna, stops) for dna in table["dna_sequence"]],
            dtype=pl.Int64,
        ),
    )
    s = np.array(
        [0.0 if dna == wild_type else published[dna] for dna in table["dna_sequence"]],
        dtype=float,
    )
    gap = np.abs(table["s"].to_numpy().astype(float) - s)
    off = ~(gap <= tolerance)
    if off.any():
        worst = table.filter(pl.Series(off))["dna_sequence"].to_list()
        raise ValueError(
            "Growth rates break the identity r_i - r_wt = s_i: "
            f"{len(worst)} sequences differ by more than {tolerance:g} "
            f"(largest gap {float(np.nanmax(gap)):.3g}): " + _quoted(worst)
        )

    ratio = (table["se_replicate"] / table["se_pooled"]).to_numpy().astype(float)
    diagnostics = {
        "sequence_count": table.height,
        "wild_type_sequence": wild_type,
        "wild_type_r": r_wt,
        "identity_tolerance": tolerance,
        "max_identity_gap": float(gap.max()),
        "median_se_ratio": float(np.median(ratio)),
        "ratio_threshold": RATIO_THRESHOLD,
        "share_se_ratio_above_threshold": float(np.mean(ratio > RATIO_THRESHOLD)),
        "partial_replicate_sequence_count": int(
            (table["n_replicates"] != table["n_replicates_converged"]).sum()
        ),
        "ok_status": OK_STATUS,
        "elapsed_time": float(config["elapsed_time"]),
        "time_unit": config["time_unit"],
        "barseqfit": barseqfit_version(),
    }
    return SequenceGrowthRatesResult(
        growth_rates=table.select(*GROWTH_COLUMNS),
        diagnostics=diagnostics,
        provenance={},
    )


def _fingerprint_without_export(
    settings: dict[str, Any], context: Any | None = None
) -> dict[str, Any]:
    """Fingerprint of everything except the BarSeq export files."""
    scientific = settings["analysis"]
    analysis_dir = Path(__file__).parent
    root = context.root if context is not None else ROOT
    raw_rel = scientific["dataset"]["fitness_wt"]
    if context is not None:
        wt_path = context.raw_data_dir / Path(raw_rel).name
        if not wt_path.is_file():
            wt_path = root / raw_rel
    else:
        wt_path = ROOT / raw_rel

    def _key(p: Path) -> str:
        if (
            context is not None
            and getattr(context, "root", None) is not None
            and context.root != ROOT
        ):
            return source_key(p, root=root)
        return source_key(p)

    files = {
        _key(path): _sha256(path)
        for path in (
            Path(__file__).resolve(),
            analysis_dir.joinpath("barseq_inputs.py").resolve(),
            root / "uv.lock",
        )
    }
    files[Path(raw_rel).as_posix()] = _sha256(wt_path)
    # The effective sources of the package this stage computes with (barseqfit): a
    # lockfile alone misses an editable install. Topographer is not used here.
    if (
        context is not None
        and getattr(context, "root", None) is not None
        and context.root != ROOT
    ):
        files.update(package_source_hashes(barseqfit, root=root))
    else:
        files.update(package_source_hashes(barseqfit))
    return {
        "schema": CACHE_SCHEMA,
        "config": scientific[SECTION],
        "stop_codons": list(scientific["codon_entropy"]["stop_codons"]),
        "files": files,
    }


def _fingerprint(
    settings: dict[str, Any], export: BarseqExport, context: Any | None = None
) -> dict[str, Any]:
    if context is not None:
        fingerprint = _fingerprint_without_export(settings, context=context)
    else:
        fingerprint = _fingerprint_without_export(settings)
    for name in ("counts", "samples", "sequences"):
        fingerprint["files"][f"barseq_export/{name}.parquet"] = _sha256(
            export.source_dir / f"{name}.parquet"
        )
    return fingerprint


def _read_checkpoint(path: Path) -> SequenceGrowthRatesResult:
    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    for name in ARTIFACTS:
        if _sha256(path / name) != provenance["artifacts"][name]:
            raise ValueError(f"Growth-rate artifact hash mismatch: {name}.")
    rates = pl.read_parquet(path / "growth_rates.parquet")
    diagnostics = json.loads((path / "diagnostics.json").read_text(encoding="utf-8"))
    if rates.columns != list(GROWTH_COLUMNS):
        raise ValueError(
            "Growth-rate checkpoint predates the current schema (its columns "
            f"are {rates.columns}, expected {list(GROWTH_COLUMNS)}); rerun "
            "`just sequence-growth-rates`."
        )
    if (
        rates.height != diagnostics["sequence_count"]
        or rates["dna_sequence"].n_unique() != rates.height
    ):
        raise ValueError("Growth-rate coverage mismatch.")
    return SequenceGrowthRatesResult(rates, diagnostics, provenance)


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    result = _read_checkpoint(path)
    return {"result": result, "fingerprint": result.provenance["fingerprint"]}


def _write_checkpoint(
    result: SequenceGrowthRatesResult,
    stage: Path,
    fingerprint: dict[str, Any],
    context: Any | None = None,
) -> None:
    result.growth_rates.write_parquet(stage / "growth_rates.parquet")
    (stage / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    provenance = {
        "fingerprint": fingerprint,
        "artifacts": {name: _sha256(stage / name) for name in ARTIFACTS},
    }
    if context is not None:
        from ..results.context import context_record

        provenance["execution_context"] = context_record(context)
    (stage / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result.provenance = provenance


def load_cached_sequence_growth_rates(
    *, lock: bool = True, context: Any | None = None
) -> SequenceGrowthRatesResult:
    """Read only a completed checkpoint; never compute or publish.

    ``lock=False`` skips the checkpoint lock, which is a nonblocking exclusive
    lock: concurrent readers would refuse each other. Only a caller that knows
    no process writes the checkpoint may pass it (the Slurm array tasks of a
    sliced bootstrap, which run after the job that produced it).
    """
    cache_dir = (
        Path(context.output_root) / "analysis" / "sequence_growth_rates"
        if context is not None
        else CACHE_DIR
    )
    pending_path = (
        Path(context.output_root) / "analysis" / ".sequence-growth-rates.pending"
        if context is not None
        else PENDING_PATH
    )
    if not (cache_dir / "provenance.json").is_file():
        raise FileNotFoundError(
            f"No sequence growth-rate checkpoint at {cache_dir}; run "
            "`just sequence-growth-rates`."
        )
    with (
        checkpoint_lock(cache_dir, "Sequence growth rates")
        if lock
        else contextlib.nullcontext()
    ):
        if pending_path.exists():
            raise RuntimeError(
                "Sequence growth-rate refresh is pending; rerun "
                "`just sequence-growth-rates`."
            )
        return _read_checkpoint(cache_dir)


def compute_sequence_growth_rates(
    settings: dict[str, Any] | None = None,
    *,
    force: bool = False,
    context: Any | None = None,
) -> SequenceGrowthRatesResult:
    """Load the current checkpoint or fit, check and atomically publish a new one."""
    if settings is None:
        settings = (
            load_settings() if context is None else load_settings(context=context)
        )
    export = (
        load_barseq_export(settings, SECTION)
        if context is None
        else load_barseq_export(settings, SECTION, context=context)
    )
    fingerprint = (
        _fingerprint(settings, export, context=context)
        if context is not None
        else _fingerprint(settings, export)
    )
    label = "Sequence growth rates"
    cache_dir = (
        Path(context.output_root) / "analysis" / "sequence_growth_rates"
        if context is not None
        else CACHE_DIR
    )
    pending_path = (
        Path(context.output_root) / "analysis" / ".sequence-growth-rates.pending"
        if context is not None
        else PENDING_PATH
    )
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

        fitness_wt = (
            load_fitness_wt(settings, context=context)
            if context is not None
            else load_fitness_wt(settings)
        )

        result = build_sequence_growth_rates(
            fitness_wt,
            export,
            settings["analysis"][SECTION],
            settings["analysis"]["codon_entropy"]["stop_codons"],
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("Sequence growth-rate refresh pending\n")
        stage = Path(
            tempfile.mkdtemp(prefix=".sequence-growth-rates-", dir=cache_dir.parent)
        )
        try:
            _write_checkpoint(result, stage, fingerprint, context=context)
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
