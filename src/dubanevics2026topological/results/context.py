"""Explicit, immutable run context replacing import-time globals (Spec 1, ADR 0015, ADR 0016)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# Environment variable names mapped to RunContext fields
ENV_OUTPUT_ROOT = "FITNESSTREE_OUTPUT_ROOT"
ENV_OUTPUT_LEGACY = "OUTPUT"
ENV_ANALYSIS_VARIANT = "FITNESSTREE_ANALYSIS_VARIANT"
ENV_BOOTSTRAP_CHUNK_DIR = "FITNESSTREE_BOOTSTRAP_CHUNK_DIR"
ENV_BOOTSTRAP_NO_DRAW = "FITNESSTREE_BOOTSTRAP_NO_DRAW"
ENV_RAW_DIR = "FITNESSTREE_RAW_DIR"
ENV_PROCESSED_ROOT = "FITNESSTREE_PROCESSED_ROOT"
ENV_FONT_DIR = "FITNESSTREE_FONT_DIR"
ENV_N_JOBS = "FITNESSTREE_N_JOBS"
ENV_BARSEQ_EXPORT = "PAPKOU2023_BARSEQ_EXPORT"
ENV_HOST = "FITNESSTREE_HOST"
ENV_RUN_ID = "FITNESSTREE_RUN_ID"
ENV_RUN_LEGACY = "FITNESSTREE_RUN"


def positive_ascii_integer(value: str | int, *, name: str = ENV_N_JOBS) -> int:
    """Parse a positive integer from an int or plain ASCII decimal string."""
    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 -- invalid values use one actionable error
            f"{name} must be a positive ASCII integer, got {value!r}."
        )
    if isinstance(value, int):
        if value > 0:
            return value
    elif isinstance(value, str) and value and value.isascii() and value.isdigit():
        parsed = int(value)
        if parsed > 0:
            return parsed
    raise ValueError(f"{name} must be a positive ASCII integer, got {value!r}.")


@dataclass(frozen=True)
class RunContext:
    """Explicit, immutable execution context for pipeline stages.

    Holds filesystem locations, concurrency, run identity, and execution flags,
    eliminating import-time global state.

    The single constructor reads current environment variables as its default source
    when fields are omitted.
    """

    output_root: Path
    raw_data_dir: Path
    processed_root: Path
    barseq_export_dir: Path | None = None
    bootstrap_chunk_dir: Path | None = None
    font_dir: Path | None = None
    host: str = "local"
    run_id: str | None = None
    worker_concurrency: int | None = None
    variant_name: str | None = None
    no_draw_allowed: bool = False
    root: Path = REPO_ROOT

    def __init__(
        self,
        *,
        output_root: str | Path | None = None,
        raw_data_dir: str | Path | None = None,
        processed_root: str | Path | None = None,
        barseq_export_dir: str | Path | None = None,
        bootstrap_chunk_dir: str | Path | None = None,
        chunk_dir: str | Path | None = None,
        font_dir: str | Path | None = None,
        host: str | None = None,
        run_id: str | None = None,
        worker_concurrency: int | None = None,
        n_jobs: int | None = None,
        variant_name: str | None = None,
        no_draw_allowed: bool | None = None,
        root: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        source_env = os.environ if env is None else env

        # Resolve repository root
        resolved_root = Path(root).resolve() if root is not None else REPO_ROOT
        object.__setattr__(self, "root", resolved_root)

        # 1. output_root: Spec 1 table (OUTPUT / FITNESSTREE_OUTPUT_ROOT)
        if output_root is not None:
            resolved_output = Path(output_root)
            if not resolved_output.is_absolute():
                resolved_output = resolved_root / resolved_output
        else:
            env_out = source_env.get(ENV_OUTPUT_ROOT) or source_env.get(
                ENV_OUTPUT_LEGACY
            )
            if env_out and env_out.strip():
                p = Path(env_out.strip())
                resolved_output = p if p.is_absolute() else (resolved_root / p)
            else:
                resolved_output = resolved_root / "output"
        object.__setattr__(self, "output_root", resolved_output.resolve())

        # 3. raw_data_dir: Spec 1 table (FITNESSTREE_RAW_DIR)
        if raw_data_dir is not None:
            resolved_raw = Path(raw_data_dir)
            if not resolved_raw.is_absolute():
                resolved_raw = resolved_root / resolved_raw
        else:
            env_raw = source_env.get(ENV_RAW_DIR)
            if env_raw and env_raw.strip():
                p = Path(env_raw.strip())
                resolved_raw = p if p.is_absolute() else (resolved_root / p)
            else:
                resolved_raw = resolved_root / "data" / "raw"
        resolved_raw = resolved_raw.resolve()
        object.__setattr__(self, "raw_data_dir", resolved_raw)

        # 3b. processed_root: the single processed-data folder (FITNESSTREE_PROCESSED_ROOT)
        if processed_root is not None:
            resolved_processed = Path(processed_root)
            if not resolved_processed.is_absolute():
                resolved_processed = resolved_root / resolved_processed
        else:
            env_processed = source_env.get(ENV_PROCESSED_ROOT)
            if env_processed and env_processed.strip():
                p = Path(env_processed.strip())
                resolved_processed = p if p.is_absolute() else (resolved_root / p)
            else:
                resolved_processed = resolved_root / "data" / "processed"
        object.__setattr__(self, "processed_root", resolved_processed.resolve())

        # 4. barseq_export_dir: Spec 1 table (PAPKOU2023_BARSEQ_EXPORT)
        if barseq_export_dir is not None:
            resolved_barseq = Path(barseq_export_dir)
            if not resolved_barseq.is_absolute():
                resolved_barseq = (resolved_root / resolved_barseq).resolve()
            else:
                resolved_barseq = resolved_barseq.resolve()
        else:
            env_barseq = source_env.get(ENV_BARSEQ_EXPORT)
            if env_barseq and env_barseq.strip():
                p = Path(env_barseq.strip()).expanduser()
                resolved_barseq = (
                    (resolved_root / p).resolve()
                    if not p.is_absolute()
                    else p.resolve()
                )
            elif (resolved_raw / "barseq_export").is_dir():
                resolved_barseq = (resolved_raw / "barseq_export").resolve()
            else:
                resolved_barseq = None
        object.__setattr__(self, "barseq_export_dir", resolved_barseq)

        # 5. bootstrap_chunk_dir: Spec 1 table (FITNESSTREE_BOOTSTRAP_CHUNK_DIR)
        effective_chunk = (
            bootstrap_chunk_dir if bootstrap_chunk_dir is not None else chunk_dir
        )
        if effective_chunk is not None:
            resolved_chunk = Path(effective_chunk)
            if not resolved_chunk.is_absolute():
                resolved_chunk = (resolved_root / resolved_chunk).resolve()
            else:
                resolved_chunk = resolved_chunk.resolve()
        else:
            env_chunk = source_env.get(ENV_BOOTSTRAP_CHUNK_DIR)
            if env_chunk and env_chunk.strip():
                p = Path(env_chunk.strip()).expanduser()
                resolved_chunk = (
                    (resolved_root / p).resolve()
                    if not p.is_absolute()
                    else p.resolve()
                )
            else:
                resolved_chunk = None
        object.__setattr__(self, "bootstrap_chunk_dir", resolved_chunk)

        # 6. font_dir: Spec 1 table (FITNESSTREE_FONT_DIR)
        if font_dir is not None:
            resolved_font = Path(font_dir)
            if not resolved_font.is_absolute():
                resolved_font = (resolved_root / resolved_font).resolve()
            else:
                resolved_font = resolved_font.resolve()
        else:
            env_font = source_env.get(ENV_FONT_DIR)
            if env_font and env_font.strip():
                p = Path(env_font.strip()).expanduser()
                resolved_font = (
                    (resolved_root / p).resolve()
                    if not p.is_absolute()
                    else p.resolve()
                )
            else:
                resolved_font = None
        object.__setattr__(self, "font_dir", resolved_font)

        # 7. host: Spec 1 / ADR 0016 run fields
        if host is not None:
            object.__setattr__(self, "host", str(host))
        else:
            env_host = source_env.get(ENV_HOST)
            object.__setattr__(
                self,
                "host",
                env_host.strip() if env_host and env_host.strip() else "local",
            )

        # 8. run_id: Spec 1 / ADR 0016 run fields
        if run_id is not None:
            object.__setattr__(self, "run_id", str(run_id))
        else:
            env_run = source_env.get(ENV_RUN_ID) or source_env.get(ENV_RUN_LEGACY)
            object.__setattr__(
                self, "run_id", env_run.strip() if env_run and env_run.strip() else None
            )

        # 9. worker_concurrency: Spec 1 table (FITNESSTREE_N_JOBS)
        effective_n_jobs = (
            worker_concurrency if worker_concurrency is not None else n_jobs
        )
        if effective_n_jobs is not None:
            object.__setattr__(
                self,
                "worker_concurrency",
                positive_ascii_integer(effective_n_jobs),
            )
        else:
            env_jobs = source_env.get(ENV_N_JOBS)
            if env_jobs and env_jobs.strip():
                object.__setattr__(
                    self,
                    "worker_concurrency",
                    positive_ascii_integer(env_jobs.strip()),
                )
            else:
                object.__setattr__(self, "worker_concurrency", None)

        # 11. variant_name: Spec 1 table (FITNESSTREE_ANALYSIS_VARIANT)
        if variant_name is not None:
            object.__setattr__(
                self, "variant_name", str(variant_name) if variant_name else None
            )
        else:
            env_variant = source_env.get(ENV_ANALYSIS_VARIANT)
            object.__setattr__(
                self,
                "variant_name",
                env_variant.strip() if env_variant and env_variant.strip() else None,
            )

        # 12. no_draw_allowed: Spec 1 table (FITNESSTREE_BOOTSTRAP_NO_DRAW)
        if no_draw_allowed is not None:
            object.__setattr__(self, "no_draw_allowed", bool(no_draw_allowed))
        else:
            env_no_draw = source_env.get(ENV_BOOTSTRAP_NO_DRAW)
            if env_no_draw is not None and env_no_draw.strip():
                val = env_no_draw.strip().lower()
                object.__setattr__(
                    self, "no_draw_allowed", val not in {"", "0", "false", "no", "off"}
                )
            else:
                object.__setattr__(self, "no_draw_allowed", False)

    @property
    def chunk_dir(self) -> Path | None:
        """Alias for bootstrap_chunk_dir."""
        return self.bootstrap_chunk_dir

    @property
    def raw_dir(self) -> Path:
        """Alias matching the Spec 1 name for the raw input directory."""
        return self.raw_data_dir

    @property
    def n_jobs(self) -> int | None:
        """Alias for worker_concurrency."""
        return self.worker_concurrency

    @property
    def analysis_dir(self) -> Path:
        """Standard analysis output directory under output_root."""
        return self.output_root / "analysis"

    @property
    def tables_dir(self) -> Path:
        """Standard tables output directory under output_root."""
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        """Standard figures output directory under output_root."""
        return self.output_root / "figures"

    def checkpoint_dir(self, stage_name: str) -> Path:
        """Standard checkpoint directory for a given stage."""
        return self.output_root / "checkpoints" / stage_name

    def product_dir(self, product_type: str, directory_name: str) -> Path:
        """Standard product directory under output_root."""
        return self.output_root / product_type / directory_name

    def env_vars(self) -> dict[str, str]:
        """Convert this context to an environment variable dictionary for child processes."""
        vars_dict: dict[str, str] = {
            ENV_OUTPUT_ROOT: str(self.output_root),
            ENV_OUTPUT_LEGACY: str(self.output_root),
            ENV_RAW_DIR: str(self.raw_data_dir),
            ENV_PROCESSED_ROOT: str(self.processed_root),
            ENV_HOST: self.host,
            ENV_BOOTSTRAP_NO_DRAW: "1" if self.no_draw_allowed else "0",
            ENV_BARSEQ_EXPORT: ""
            if self.barseq_export_dir is None
            else str(self.barseq_export_dir),
            ENV_BOOTSTRAP_CHUNK_DIR: ""
            if self.bootstrap_chunk_dir is None
            else str(self.bootstrap_chunk_dir),
            ENV_FONT_DIR: "" if self.font_dir is None else str(self.font_dir),
            ENV_RUN_ID: "" if self.run_id is None else self.run_id,
            ENV_RUN_LEGACY: "" if self.run_id is None else self.run_id,
            ENV_N_JOBS: ""
            if self.worker_concurrency is None
            else str(self.worker_concurrency),
            ENV_ANALYSIS_VARIANT: ""
            if self.variant_name is None
            else self.variant_name,
        }
        return vars_dict

    @contextmanager
    def activated(self) -> Iterator[RunContext]:
        """Temporarily set process environment variables to match this context."""
        old_env: dict[str, str | None] = {}
        for key, value in self.env_vars().items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            yield self
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with paths relative to project root.

        Host and run_id are retired from the context representation.
        """
        root = self.root.resolve()
        return {
            "output_root": _rel_path_str(self.output_root, root),
            "raw_data_dir": _rel_path_str(self.raw_data_dir, root),
            "processed_root": _rel_path_str(self.processed_root, root),
            "barseq_export_dir": _rel_path_str(self.barseq_export_dir, root),
            "bootstrap_chunk_dir": _rel_path_str(self.bootstrap_chunk_dir, root),
            "font_dir": _rel_path_str(self.font_dir, root),
            "worker_concurrency": self.worker_concurrency,
            "variant_name": self.variant_name,
            "no_draw_allowed": self.no_draw_allowed,
        }


def _rel_path_str(path: Path | None, root: Path) -> str | None:
    """Return path relative to root if inside root, else posix string or None."""
    if path is None:
        return None
    resolved_path = path.resolve() if not path.is_absolute() else path
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return path.as_posix()


def context_record(context: RunContext) -> dict[str, Any]:
    """Record the fully resolved execution context and its canonical digest."""
    resolved = context.to_dict()
    resolved["root"] = str(context.root.resolve())
    for key in (
        "output_root",
        "raw_data_dir",
        "processed_root",
        "barseq_export_dir",
        "bootstrap_chunk_dir",
        "font_dir",
    ):
        value = getattr(context, key)
        resolved[key] = None if value is None else str(value.resolve())
    return context_record_from_mapping(resolved)


def context_record_from_mapping(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Hash a resolved context independently of mapping insertion order."""
    payload = json.dumps(
        resolved,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "resolved_context": dict(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def serialize_run_context(context: RunContext) -> list[str]:
    """Serialize a RunContext into CLI flags with paths relative to the project root.

    Retired execution fields (host, run_id) are excluded.
    """
    flags: list[str] = []
    root = context.root.resolve()
    if root != REPO_ROOT.resolve():
        flags.extend(["--project-root", str(root)])

    rel_out = _rel_path_str(context.output_root, root)
    if rel_out is not None:
        flags.extend(["--output-root", rel_out])

    rel_raw = _rel_path_str(context.raw_data_dir, root)
    if rel_raw is not None:
        flags.extend(["--raw-data-dir", rel_raw])

    rel_processed = _rel_path_str(context.processed_root, root)
    if rel_processed is not None:
        flags.extend(["--processed-root", rel_processed])

    rel_barseq = _rel_path_str(context.barseq_export_dir, root)
    if rel_barseq is not None:
        flags.extend(["--barseq-export-dir", rel_barseq])

    rel_chunk = _rel_path_str(context.bootstrap_chunk_dir, root)
    if rel_chunk is not None:
        flags.extend(["--bootstrap-chunk-dir", rel_chunk])

    rel_font = _rel_path_str(context.font_dir, root)
    if rel_font is not None:
        flags.extend(["--font-dir", rel_font])

    if context.worker_concurrency is not None:
        flags.extend(["--worker-concurrency", str(context.worker_concurrency)])

    if context.variant_name is not None:
        flags.extend(["--analysis-variant", str(context.variant_name)])

    if context.no_draw_allowed:
        flags.append("--no-draw")
    else:
        flags.append("--allow-draw")

    return flags


def add_run_context_arguments(parser: argparse.ArgumentParser) -> None:
    """Register standard RunContext CLI flags on an ArgumentParser."""
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Repository project root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root directory for outputs (replaces OUTPUT / FITNESSTREE_OUTPUT_ROOT).",
    )
    parser.add_argument(
        "--raw-data-dir",
        "--raw-dir",
        dest="raw_data_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Raw input directory (replaces FITNESSTREE_RAW_DIR).",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Processed-data folder (replaces FITNESSTREE_PROCESSED_ROOT).",
    )
    parser.add_argument(
        "--barseq-export-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="BarSeq export directory (replaces PAPKOU2023_BARSEQ_EXPORT).",
    )
    parser.add_argument(
        "--bootstrap-chunk-dir",
        "--chunk-dir",
        dest="bootstrap_chunk_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Bootstrap chunk directory (replaces FITNESSTREE_BOOTSTRAP_CHUNK_DIR).",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Font directory for figures (replaces FITNESSTREE_FONT_DIR).",
    )
    parser.add_argument(
        "--worker-concurrency",
        "--n-jobs",
        dest="worker_concurrency",
        type=positive_ascii_integer,
        default=None,
        metavar="N",
        help="Worker concurrency limit (replaces FITNESSTREE_N_JOBS).",
    )
    parser.add_argument(
        "--analysis-variant",
        default=None,
        metavar="NAME",
        help="Analysis variant overlay (replaces FITNESSTREE_ANALYSIS_VARIANT).",
    )
    draw = parser.add_mutually_exclusive_group()
    draw.add_argument(
        "--no-draw",
        action="store_true",
        default=None,
        dest="no_draw",
        help="Forbid drawing bootstrap resamples (replaces FITNESSTREE_BOOTSTRAP_NO_DRAW).",
    )
    draw.add_argument(
        "--allow-draw",
        action="store_false",
        dest="no_draw",
        help="Allow drawing bootstrap resamples.",
    )


def parse_run_context(
    args: argparse.Namespace | Sequence[str],
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RunContext:
    """Reconstruct a RunContext from CLI flags or an argparse.Namespace."""
    if not isinstance(args, argparse.Namespace):
        parser = argparse.ArgumentParser(add_help=False)
        add_run_context_arguments(parser)
        parsed, _ = parser.parse_known_args(args)
    else:
        parsed = args

    effective_root = getattr(parsed, "project_root", None) or root
    no_draw_flag = getattr(parsed, "no_draw", None)
    no_draw_allowed: bool | None
    if no_draw_flag is True:
        no_draw_allowed = True
    elif no_draw_flag is False:
        no_draw_allowed = False
    else:
        no_draw_allowed = None

    return RunContext(
        output_root=getattr(parsed, "output_root", None),
        raw_data_dir=getattr(parsed, "raw_data_dir", None),
        processed_root=getattr(parsed, "processed_root", None),
        barseq_export_dir=getattr(parsed, "barseq_export_dir", None),
        bootstrap_chunk_dir=getattr(parsed, "bootstrap_chunk_dir", None),
        font_dir=getattr(parsed, "font_dir", None),
        worker_concurrency=getattr(parsed, "worker_concurrency", None),
        variant_name=getattr(parsed, "analysis_variant", None),
        no_draw_allowed=no_draw_allowed,
        root=effective_root,
        env=env,
    )


if __name__ == "__main__":
    import json

    cli_parser = argparse.ArgumentParser(description="RunContext serializer CLI")
    cli_parser.add_argument(
        "--job-json",
        type=Path,
        default=None,
        help="Read context from job.json and serialize to flags",
    )
    cli_parser.add_argument(
        "--json", action="store_true", help="Output context as JSON"
    )
    add_run_context_arguments(cli_parser)
    parsed_cli = cli_parser.parse_args()

    if parsed_cli.job_json:
        with parsed_cli.job_json.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        ctx_data = data.get("context", {})
        ctx_data = ctx_data.get("resolved_context", ctx_data)
        ctx_instance = RunContext(
            output_root=ctx_data.get("output_root"),
            raw_data_dir=ctx_data.get("raw_data_dir"),
            processed_root=ctx_data.get("processed_root"),
            barseq_export_dir=ctx_data.get("barseq_export_dir"),
            bootstrap_chunk_dir=ctx_data.get("bootstrap_chunk_dir"),
            font_dir=ctx_data.get("font_dir"),
            worker_concurrency=ctx_data.get("worker_concurrency"),
            variant_name=ctx_data.get("variant_name"),
            no_draw_allowed=ctx_data.get("no_draw_allowed"),
            root=parsed_cli.project_root,
        )
    else:
        ctx_instance = parse_run_context(parsed_cli)

    if parsed_cli.json:
        print(json.dumps(ctx_instance.to_dict(), indent=2))
    else:
        for flag_item in serialize_run_context(ctx_instance):
            print(flag_item)
