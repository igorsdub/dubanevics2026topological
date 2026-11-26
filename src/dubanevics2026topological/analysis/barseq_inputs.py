"""Load the BarSeq export tables and map sequences to their export ids."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv

from . import ROOT

ENV_FILE = ROOT / ".env"

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "counts": ("sequence_id", "sample_id", "count"),
    "samples": ("sample_id", "replicate_id", "condition", "time", "sample_total"),
    "sequences": ("sequence_id", "sequence"),
}


@dataclass(frozen=True)
class BarseqExport:
    """Count, sample and sequence tables exported by barseqfit, with their folder."""

    counts: pl.DataFrame
    samples: pl.DataFrame
    sequences: pl.DataFrame
    source_dir: Path


def load_barseq_export(
    settings: dict[str, Any],
    section: str = "candidate_tree_contrast",
    *,
    context: Any | None = None,
) -> BarseqExport:
    """Read the three barseqfit export tables from the configured folder.

    The folder comes from ``context.barseq_export_dir`` if set, else from the
    environment variable named by ``analysis.<section>.barseq_export_env``
    (the contrast stage by default; ``sequence_growth_rates`` names the same
    variable). A ``.env`` file in the repository root supplies it when the
    shell has not already exported it.
    """
    if context is not None:
        if getattr(context, "barseq_export_dir", None) is None:
            raise ValueError(
                "RunContext has no barseq_export_dir; provide it when constructing "
                "the context or call without a context to use the legacy environment."
            )
        directory = Path(context.barseq_export_dir)
        if not directory.is_dir():
            raise ValueError(
                f"RunContext barseq_export_dir points to a missing directory: {directory}"
            )
    else:
        load_dotenv(ENV_FILE, override=False)
        env_name = settings["analysis"][section]["barseq_export_env"]
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise ValueError(
                f"Environment variable {env_name} is not set. Set it in .env to the "
                "folder written by `just export-papkou` in the barseqfit repository."
            )
        directory = Path(value).expanduser()
        if not directory.is_dir():
            raise ValueError(f"{env_name} points to a missing directory: {directory}")

    tables: dict[str, pl.DataFrame] = {}
    problems: list[str] = []
    for name, required in REQUIRED_COLUMNS.items():
        path = directory / f"{name}.parquet"
        if not path.is_file():
            raise ValueError(
                f"Missing {path.name} in BarSeq export directory {directory}"
            )
        frame = pl.read_parquet(path)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            problems.append(f"{name} is missing columns {missing}")
        tables[name] = frame
    if problems:
        raise ValueError(
            f"Invalid BarSeq export in {directory}: " + "; ".join(problems)
        )
    return BarseqExport(**tables, source_dir=directory.resolve())


def _quoted(values: Iterable[str], limit: int = 20) -> str:
    items = list(values)
    shown = ", ".join(repr(value) for value in items[:limit])
    extra = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return shown + extra


def map_sequence_ids(
    sequences: pl.DataFrame,
    wanted: Iterable[str],
    *,
    describe: Callable[[set[str]], str] | None = None,
) -> dict[str, str]:
    """Map DNA sequences to the export's ``sequence_id`` values.

    ``wanted`` is any collection of DNA sequences (tree endpoints, network
    vertices, the wild type); identities stay the DNA sequences themselves.
    Raises ``ValueError`` when a wanted sequence is listed under several
    ``sequence_id`` values or is absent from the export. ``describe`` turns the
    offending sequences into the phrase that ends the message (for example the
    edges they belong to); by default the sequences are listed.
    """
    needed = set(wanted)
    describe = describe or (lambda bad: "sequences " + _quoted(sorted(bad)))
    table = sequences.filter(pl.col("sequence").is_in(list(needed)))
    duplicated = set(table.filter(pl.col("sequence").is_duplicated())["sequence"])
    if duplicated:
        raise ValueError(
            "BarSeq export lists several sequence_id values for " + describe(duplicated)
        )
    ids = dict(zip(table["sequence"], table["sequence_id"], strict=True))
    absent = needed - ids.keys()
    if absent:
        raise ValueError(
            "Sequences are absent from the BarSeq export for " + describe(absent)
        )
    return ids


def barseqfit_version() -> dict[str, Any]:
    """Installed barseqfit version and the git revision it was built from."""
    info: dict[str, Any] = {
        "version": importlib.metadata.version("barseqfit"),
        "rev": None,
    }
    try:
        direct = importlib.metadata.distribution("barseqfit").read_text(
            "direct_url.json"
        )
        if direct:
            info["rev"] = json.loads(direct).get("vcs_info", {}).get("commit_id")
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError):
        pass
    if info["rev"] is None:
        try:
            lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
            match = re.search(
                r'name = "barseqfit"\n.*?\?rev=([0-9a-f]+)', lock, re.DOTALL
            )
            info["rev"] = match.group(1) if match else None
        except OSError:
            pass
    return info
