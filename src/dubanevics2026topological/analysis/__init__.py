"""Reproducible Papkou2023 analysis inputs and checkpoint management."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl
import topographer

from dubanevics2026topological.papkou2023 import Papkou2023Workflow
from dubanevics2026topological.network_analysis import load_network_analysis

from . import produce_or_load as _produce_or_load
from .describe import DescribedProducts as DescribedProducts
from .describe import describe as describe
from .produce_or_load import PROVENANCE_NAME, Checkpoint, Stage

ROOT = Path(__file__).resolve().parents[3]
# A variant run keeps every checkpoint, table and figure out of the default ``output/``:
# ``FITNESSTREE_OUTPUT_ROOT`` (a path under ROOT, or absolute) replaces it for the whole
# process, and ``FITNESSTREE_ANALYSIS_VARIANT`` names ``config/variants/<name>.toml``,
# merged table by table over ``config/analysis.toml``. Both unset: the default behaviour.
ENV_OUTPUT_ROOT = "FITNESSTREE_OUTPUT_ROOT"
ENV_ANALYSIS_VARIANT = "FITNESSTREE_ANALYSIS_VARIANT"
# Nicknames that named the four variant files before they took the parameter string.
LEGACY_VARIANT_NAMES = {
    "point_papkou_original": "threshold=published_fdr=none",
    "point_median_reflected": "threshold=reflected_fdr=none",
    "fdr_by_q0.05_papkou_original": "threshold=published_fdr=by_q=0.05",
    "fdr_by_q0.05_median_reflected": "threshold=reflected_fdr=by_q=0.05",
}
OUTPUT = ROOT / os.environ.get(ENV_OUTPUT_ROOT, "output")
ANALYSIS_DIR = OUTPUT / "analysis"
DATA_DIR = ANALYSIS_DIR / "data"
PROVENANCE_PATH = ANALYSIS_DIR / "provenance.json"
CHECKPOINT_DIR = OUTPUT / "checkpoints" / "analysis"
# The parsed sections of config/analysis.toml that the analysis and its component
# persistence read. Every other section belongs to a later stage.
ANALYSIS_SECTIONS = ("dataset", "tree", "persistence")


def get_run_context() -> Any:
    """Return a RunContext constructed from the current environment."""
    from dubanevics2026topological.results import RunContext

    return RunContext()


def load_settings(context: Any | None = None) -> dict[str, Any]:
    """Load the checked-in scientific and presentation TOML configuration."""
    with (ROOT / "config/analysis.toml").open("rb") as stream:
        analysis = tomllib.load(stream)
    variant = (
        context.variant_name or ""
        if context is not None
        else os.environ.get(ENV_ANALYSIS_VARIANT, "").strip()
    )
    if variant:
        variant = LEGACY_VARIANT_NAMES.get(variant, variant)
        if (
            not variant.replace("_", "")
            .replace("-", "")
            .replace(".", "")
            .replace("=", "")
            .isalnum()
        ):
            raise ValueError(f"{ENV_ANALYSIS_VARIANT} {variant!r} is not a plain name.")
        with (ROOT / "config" / "variants" / f"{variant}.toml").open("rb") as stream:
            overlay = tomllib.load(stream)
        for section, values in overlay.items():
            if isinstance(values, dict) and isinstance(analysis.get(section), dict):
                analysis[section] = {**analysis[section], **values}
            else:
                analysis[section] = values
    from ..threshold.functional_threshold import (
        DECLARATION_KEY,
        validate_candidate_declaration,
    )

    validate_candidate_declaration(analysis["clamped"][DECLARATION_KEY])
    from ..threshold.functional_classification_config import SECTION, validate_config

    # Optional here (trimmed test configs omit it); the stage itself requires it.
    if SECTION in analysis:
        validate_config(analysis[SECTION])
    with (ROOT / "config/figures.toml").open("rb") as stream:
        figures = tomllib.load(stream)
    _apply_shared_panel_sizes(figures)
    return {"analysis": analysis, "figures": figures}


def _apply_shared_panel_sizes(figures: dict[str, Any]) -> None:
    """Fill each figure's ``tree`` and ``persistence`` panel size from the shared keys.

    ``[rendering]`` holds the one tree size (``tree_width`` x ``tree_height``) and
    persistence size (``persistence_width`` x ``persistence_height``). A figure's
    ``tree`` or ``persistence`` table references them by leaving ``width`` and
    ``height`` out; a size set in the table overrides the shared one.
    """
    rendering = figures.get("rendering", {})
    for figure in figures.get("figures", {}).values():
        for panel in ("tree", "persistence"):
            table = figure.get(panel)
            if not isinstance(table, dict):
                continue
            for axis in ("width", "height"):
                shared = rendering.get(f"{panel}_{axis}")
                if shared is not None:
                    table.setdefault(axis, shared)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(
    path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None
) -> str:
    """Return the POSIX key of ``path`` relative to ``ROOT``, without symlinks.

    The path is normalised with ``os.path.abspath`` (relative paths are taken
    against ``ROOT``) and is deliberately *not* resolved: a file under ``ROOT``
    that is a symlink to storage elsewhere, such as ``data/raw/*``, keeps its
    in-repository key. Raises ``ValueError`` for a path outside ``ROOT``.
    """
    effective_root = os.path.abspath(ROOT if root is None else root)
    absolute = os.path.abspath(os.path.join(effective_root, os.fspath(path)))
    if os.path.commonpath([effective_root, absolute]) != effective_root:
        raise ValueError(
            f"{absolute} lies outside the repository root {effective_root}."
        )
    return Path(os.path.relpath(absolute, effective_root)).as_posix()


def fingerprint_key(
    path: Path,
    package: str = "topographer",
    root: str | os.PathLike[str] | None = None,
) -> str:
    """Key a fingerprinted file: ``source_key``, or ``<package>:`` if external.

    Only files that genuinely live outside ``ROOT`` (installed package sources,
    Topographer's by default) take the ``<package>:<resolved path>`` form.
    """
    try:
        return source_key(path, root=root)
    except ValueError:
        return f"{package}:{path.resolve()}"


def dedupe_by_resolved(paths: Iterable[Path]) -> list[Path]:
    """Drop paths naming the same file, keeping the first spelling of each."""
    unique: dict[Path, Path] = {}
    for path in paths:
        unique.setdefault(path.resolve(), path)
    return list(unique.values())


def package_source_hashes(
    *packages: ModuleType, root: str | os.PathLike[str] | None = None
) -> dict[str, str]:
    """Hash every Python source of the effective (installed or editable) packages.

    The effective location is the one the interpreter imported, so an edit under
    ``uv run --with-editable ../topographer`` changes the hashes. Keys come from
    ``fingerprint_key`` labelled with the package name.
    """
    hashes: dict[str, str] = {}
    for package in packages:
        if package.__file__ is None:
            raise ValueError(f"{package.__name__} has no source location to hash.")
        pkg_root = Path(package.__file__).resolve().parent
        for path in dedupe_by_resolved(sorted(pkg_root.rglob("*.py"))):
            hashes[fingerprint_key(path, package.__name__, root=root)] = _sha256(path)
    return hashes


def scientific_inputs(
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
    *,
    root: Path | None = None,
) -> list[Path]:
    """Return the raw data files the analysis reads: under ``root``, or ``context.raw_data_dir``."""
    dataset = (settings or load_settings(context=context))["analysis"]["dataset"]
    return [
        dataset_input_path(dataset[key], context, root=root)
        for key in ("fitness_wt", "fitness_diff")
    ]


def dataset_input_path(
    value: str | os.PathLike[str],
    context: Any | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve a configured dataset path; a ``data/raw`` path follows ``context.raw_data_dir``.

    Without a context the path is resolved under ``root`` (default ``ROOT``), as before.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    if context is None:
        return (ROOT if root is None else root) / path
    if path.parts[:2] == ("data", "raw"):
        default_raw = Path(context.root) / "data" / "raw"
        if Path(context.raw_data_dir) == default_raw.resolve():
            # The default raw directory, possibly a symlink: keep the in-repository
            # spelling so its fingerprint key matches the published ones.
            return Path(context.root) / path
        return Path(context.raw_data_dir).joinpath(*path.parts[2:])
    return Path(context.root) / path


def split_declared_inputs(
    named: dict[str, Path], root: Path
) -> tuple[list[Path], dict[str, str]]:
    """Separate stage inputs under ``root`` from those that live elsewhere.

    A stage keys a declared input by its path under the stage root and refuses any
    other. An input outside it (a ``raw_data_dir`` elsewhere) is hashed here under its
    configured name instead, so the fingerprint does not depend on where it sits.
    """
    inside: list[Path] = []
    outside: dict[str, str] = {}
    for name, path in named.items():
        try:
            source_key(path, root=root)
        except ValueError:
            outside[name] = _produce_or_load._sha256(path)
        else:
            inside.append(path)
    return inside, outside


def dataset_input_key(value: str, path: Path, root: Path | None = None) -> str:
    """Key a fingerprinted dataset file: its path under ``root``, else its configured name.

    A ``raw_data_dir`` elsewhere keys by the configured ``value``, never by where it sits.
    """
    try:
        return source_key(path, root=root)
    except ValueError:
        return f"dataset:{value}"


def scientific_code_paths() -> list[Path]:
    """Return the lockfiles, local analysis code and effective Topographer sources.

    ``config/analysis.toml`` is deliberately not hashed: a stage declares the
    parsed sections it reads as settings, so scheduling keys such as ``n_jobs``
    and a comment edit leave its fingerprint alone.
    """
    package_root = Path(topographer.__file__).resolve().parent
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        Path(__file__).resolve(),
        Path(__file__).with_name("component_persistence.py").resolve(),
    ]
    for package in ("graph", "topology", "biology", "plotting"):
        source_dir = package_root / package
        paths.extend(
            path
            for path in source_dir.rglob("*.py")
            if not (package == "topology" and "_reeb" in path.parts)
        )
    import dubanevics2026topological as package_root_module

    package_dir = Path(package_root_module.__file__).resolve().parent
    paths.extend(
        [
            package_dir / "network_analysis.py",
            *sorted((package_dir / "papkou2023").rglob("*.py")),
        ]
    )
    return dedupe_by_resolved(paths)


def require_files(paths: Iterable[Path], label: str) -> None:
    """Raise ``FileNotFoundError`` naming every path that is not a file."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required {label} are missing: " + ", ".join(missing))


def hash_sources(paths: Iterable[Path]) -> dict[str, str]:
    """Hash code files keyed by ``fingerprint_key`` (relative to ``ROOT`` if inside)."""
    return {fingerprint_key(path): _sha256(path) for path in paths}


def analysis_stage(
    settings: dict[str, Any] | None = None, *, context: Any | None = None
) -> Stage:
    """Declare the main analysis: scientific settings, inputs and code only.

    The settings are the parsed config sections the analysis reads
    (``ANALYSIS_SECTIONS``), not the bytes of ``config/analysis.toml`` and not
    the hashes of code. Code-file hashes, and Topographer's version and commit,
    are recorded in the provenance label. ``project_stage`` strips the
    scheduling keys, so ``n_jobs`` and ``chunk_size`` never reach the
    fingerprint. The raw data files are the stage inputs.
    """
    # Imported here because ``stages`` itself imports the helper in this package.
    from ..stages import project_stage

    settings = settings or load_settings(context=context)
    scientific = settings["analysis"]
    if scientific.get("tree", {}).get("kind") != "split":
        raise ValueError("This analysis currently supports only tree.kind = 'split'.")
    if scientific.get("persistence", {}).get("kind") != "extended_h0":
        raise ValueError(
            "This analysis currently supports only persistence.kind = 'extended_h0'."
        )
    all_inputs = scientific_inputs(settings, context, root=ROOT)
    code = scientific_code_paths()
    require_files([*all_inputs, *code], "analysis source/input files")
    inputs, outside_inputs = split_declared_inputs(
        dict(zip(("fitness_wt", "fitness_diff"), all_inputs, strict=True)), ROOT
    )
    local_code = []
    for path in code:
        try:
            source_key(path)
        except ValueError:
            continue
        local_code.append(path)
    return project_stage(
        "analysis",
        settings={
            "config": {name: scientific[name] for name in ANALYSIS_SECTIONS},
            "tree_kind": "split",
            **({"outside_input_sha256": outside_inputs} if outside_inputs else {}),
        },
        inputs=inputs,
        code=local_code,
        root=ROOT,
    )


def scientific_fingerprint(
    settings: dict[str, Any] | None = None, *, context: Any | None = None
) -> str:
    """Return the analysis stage's fingerprint (a hex sha256).

    It covers the raw inputs and the config sections the analysis reads, never
    scheduling and never code. Component persistence shares it. Topographer's
    version and commit are a label on the published provenance, not part of
    this value.
    """
    return _produce_or_load.stage_fingerprint(analysis_stage(settings, context=context))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_analysis_record(
    record_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    *,
    context: Any | None = None,
) -> dict[str, Any]:
    """Return the caller's record once it agrees with the published checkpoint.

    The checkpoint holds the authoritative bytes in a fixed slot, so its content
    changes on publication, before ``output/analysis/provenance.json`` is
    rewritten. A reader therefore refuses, with a clear error, while the helper
    reports a pending publication (``publication_state``) and whenever the
    record's fingerprint or its artefact hashes differ from the checkpoint's own
    provenance (an interrupted refresh, or a checkpoint that was not published).
    ``just analysis`` repairs both.
    """
    output_root = context.output_root if context is not None else OUTPUT
    record_path = (
        (output_root / "analysis" / "provenance.json")
        if record_path is None
        else record_path
    )
    checkpoint_dir = (
        (output_root / "checkpoints" / "analysis")
        if checkpoint_dir is None
        else checkpoint_dir
    )
    leaf = _produce_or_load.result_directory(
        checkpoint_dir, analysis_stage(context=context)
    )
    if _produce_or_load.publication_state(leaf)["pending"]:
        raise RuntimeError("Analysis refresh is pending; rerun `just analysis` first.")
    published = leaf / PROVENANCE_NAME
    checkpoint_dir = leaf
    if not record_path.is_file():
        raise FileNotFoundError("Analysis provenance is missing; run `just analysis`.")
    if not published.is_file():
        raise FileNotFoundError(
            "No analysis checkpoint exists yet; run `just analysis`."
        )
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        checkpoint = json.loads(published.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Analysis provenance is unreadable; rerun `just analysis`."
        ) from error
    if (
        not isinstance(record, dict)
        or not isinstance(checkpoint, dict)
        or "fingerprint" not in record
    ):
        raise RuntimeError(
            "Analysis provenance has no fingerprint; run `just analysis`."
        )
    if record["fingerprint"] != checkpoint.get("fingerprint") or record.get(
        "artefacts"
    ) != checkpoint.get("artefacts"):
        raise RuntimeError(
            "The analysis record does not match the published checkpoint "
            "(an interrupted refresh?); rerun `just analysis` to repair it."
        )
    return record


def load_cached_analysis(*, context: Any | None = None):
    """Load the last published checkpoint without reading or hashing raw inputs.

    Refuses while a publication is pending or when the record disagrees with the
    checkpoint (see ``read_analysis_record``).
    """
    output_root = context.output_root if context is not None else OUTPUT
    data_dir = output_root / "analysis" / "data"
    read_analysis_record(context=context)
    if not (data_dir / "metadata.toml").is_file():
        raise FileNotFoundError(
            "No analysis checkpoint exists yet; run `just analysis`."
        )
    return load_network_analysis(data_dir)


def _produce_analysis(
    directory: Path, settings: dict[str, Any], context: Any | None = None
) -> None:
    """Run Topographer's public workflow and place its ``data`` tree in ``directory``."""
    dataset = settings["analysis"]["dataset"]
    workspace = directory.parent / "workflow"
    workspace.mkdir()
    workflow = Papkou2023Workflow(
        landscape=dataset["landscape"],
        fitness_wt=pl.read_parquet(dataset_input_path(dataset["fitness_wt"], context)),
        fitness_diff=pl.read_parquet(
            dataset_input_path(dataset["fitness_diff"], context)
        ),
    )
    # Use the Papkou workflow's public checkpoint API, including its own schema and
    # source/config validation, in a private workspace.
    workflow.produce_or_load_analysis(workspace, tree_kind="split", force=True)
    staged_data = workspace / "data"
    if not (staged_data / "metadata.toml").is_file():
        raise RuntimeError("Topographer did not create a complete analysis checkpoint.")
    # Topographer publishes `data` as a symlink into `.data-generations`; move the
    # resolved generation into the payload so the helper hashes real files.
    os.replace(staged_data.resolve(strict=True), directory / "data")


def _point_outputs_at(
    checkpoint: Checkpoint,
    settings: dict[str, Any],
    *,
    analysis_dir: Path | None = None,
    data_dir: Path | None = None,
    provenance_path: Path | None = None,
) -> None:
    """Link ``output/analysis/data`` to the checkpoint and write the caller's record.

    Runs under the checkpoint lock, so no other process can publish between the
    verification and the update. The checkpoint is the single copy of the bytes.
    The record keeps the keys callers read (``fingerprint``, ``created_at``,
    ``workflow``, ``tree_kind``) and is rewritten only when it would change. It is
    removed first and written last, so an interrupted run leaves no record
    claiming a link that is not in place.
    """
    target_analysis_dir = ANALYSIS_DIR if analysis_dir is None else analysis_dir
    target_data_dir = DATA_DIR if data_dir is None else data_dir
    target_provenance_path = (
        PROVENANCE_PATH if provenance_path is None else provenance_path
    )

    # The workflow reports its own configuration (a threshold landscape also names
    # its threshold); built without the data frames, it only describes itself.
    workflow = Papkou2023Workflow(
        landscape=settings["analysis"]["dataset"]["landscape"]
    )
    record = {
        **checkpoint.provenance,
        "workflow": workflow.config,
        "tree_kind": "split",
    }
    target = Path(os.path.relpath(checkpoint.path / "data", target_analysis_dir))
    linked = (
        target_data_dir.is_symlink() and Path(os.readlink(target_data_dir)) == target
    )
    try:
        current = json.loads(target_provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = None
    if linked and current == record:
        return
    if target_data_dir.exists() and not target_data_dir.is_symlink():
        raise RuntimeError(
            f"{target_data_dir} is a real directory; move it aside before rebuilding."
        )
    target_analysis_dir.mkdir(parents=True, exist_ok=True)
    target_provenance_path.unlink(missing_ok=True)
    temporary_link = target_analysis_dir / ".data-current"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(target)
    os.replace(temporary_link, target_data_dir)
    _write_json_atomic(target_provenance_path, record)


def load_or_compute_analysis(
    *,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    context: Any | None = None,
):
    """Load a current public Topographer checkpoint or compute and publish one.

    This explicit command checks the stage fingerprint (raw inputs, config
    sections and effective code) before loading a saved checkpoint. Figure builds
    use ``load_cached_analysis``, avoiding repeated reads of the large Parquet
    tables. Reuse, locking and atomic publication come from ``produce_or_load``;
    the published checkpoint under ``output/checkpoints`` is the single copy of
    the bytes and ``output/analysis/data`` is a link to it. The link and the
    caller's record are updated under the checkpoint lock.
    """
    settings = settings or load_settings(context=context)
    output_root = context.output_root if context is not None else OUTPUT
    step = output_root / "checkpoints" / "analysis"
    analysis_dir = output_root / "analysis"
    data_dir = analysis_dir / "data"
    provenance_path = analysis_dir / "provenance.json"
    stage = analysis_stage(settings, context=context)
    _produce_or_load.produce_or_load(
        stage,
        lambda directory: _produce_analysis(directory, settings, context),
        checkpoint_dir=_produce_or_load.result_directory(step, stage),
        force=force,
        on_ready=(
            (lambda ready: _point_outputs_at(ready, settings))
            if context is None
            else lambda ready: _point_outputs_at(
                ready,
                settings,
                analysis_dir=analysis_dir,
                data_dir=data_dir,
                provenance_path=provenance_path,
            )
        ),
    )
    return load_network_analysis(data_dir)
