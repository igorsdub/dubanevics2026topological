"""The single processed-data folder, ``data/processed/``.

``just all`` writes it, the Zenodo deposit is a copy of it and ``just figures`` only
reads it. The directory name is the identity of a product: ``<kind>/<state>/`` (or
``<kind>/<variant>/``), never a hash-suffixed label. Every product keeps its own
``manifest.json``; ``MANIFEST.tsv`` at the root lists every file (path, bytes, sha256)
and is written last.
"""

from __future__ import annotations

import argparse
import hashlib
import fnmatch
import json
import os
import re
import shutil
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import lookup as _lookup
from .context import REPO_ROOT, RunContext
from .lookup import (
    PROCESSED_DIRECTORY,
    LoadedProduct,
    MissingProductError,
    _manifest_product,
)

PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
MANIFEST_NAME = "MANIFEST.tsv"
MANIFEST_HEADER = ("path", "bytes", "sha256")

# The seven reviewed landscape/tree states, in publication order.
STATES = (
    "full",
    "published",
    "reflected",
    "published_by",
    "published_by_band",
    "reflected_by",
    "reflected_by_band",
)

# The leaves product of a state is addressed by the state name itself.
LEAVES = {state: state for state in STATES}

MISSING_HINT = "run `just all` or `just fetch-data`"

# The four raw files the analysis reads, in the deposit as ``raw/``: the figures fingerprint
# their inputs by these bytes, so ``just figures`` restores them to ``data/raw/``.
RAW_DIRECTORY = "raw"
RAW_FILES = (
    "papkou2023_fitness_wt.parquet",
    "papkou2023_fitness_diff.parquet",
    "papkou2023_graph.graphml",
    "papkou2023_peaks.csv",
)
RAW_README = "README.md"

DEPOSIT_README = """\
# Processed data

The finished results of the analysis, written by `just all` and checked against
`MANIFEST.tsv` (path, bytes, sha256) by `just check-manifest`. `just figures` renders
every figure and table from this folder alone and computes nothing.

## Credit and licence

`raw/` holds files derived from Papkou et al. 2023, "A rugged yet easily navigable
fitness landscape", Science, doi:10.1126/science.adh3860. The data are Zenodo record
13837982 (doi:10.5281/zenodo.13837982), licensed CC-BY-4.0. The derived files are
credited to Papkou et al.; the licence allows redistribution with that credit.
`raw/README.md` says how they are rebuilt from the record.
"""


def processed_root(context: RunContext | None = None) -> Path:
    """Return the processed root of a context, or the repository default."""
    return PROCESSED_ROOT if context is None else Path(context.processed_root)


def path(kind: str, state: str | None = None, *, root: Path | None = None) -> Path:
    """Return ``<root>/<kind>[/<state>]`` without touching the filesystem."""
    base = (PROCESSED_ROOT if root is None else Path(root)) / kind
    return base if state is None else base / state


def load(kind: str, state: str, context: RunContext) -> LoadedProduct:
    """Load and verify one processed product against its own ``manifest.json``."""
    directory = path(kind, state, root=processed_root(context))
    if not (directory / "manifest.json").is_file():
        raise MissingProductError(
            f"No processed {kind!r} product for {state!r} at {directory}; {MISSING_HINT}."
        )
    manifest, report = _manifest_product(directory)
    if report.status != "full" or not report.byte_integrity:
        raise MissingProductError(
            f"Processed {kind!r} product {state!r} at {directory} failed verification "
            f"({report.status}, byte_integrity={report.byte_integrity}); {MISSING_HINT}."
        )
    return LoadedProduct(kind, directory, manifest, report)


def product_loader(context: RunContext):
    """Return ``loader(product, label)``: a state name reads ``processed/<product>/<state>``, any
    other label is resolved among the products the context can see."""

    def loader(product: str, label: str) -> LoadedProduct:
        if label in STATES:
            return load(product, label, context)
        return _lookup.load(product, label, context=context)

    return loader


def state_for_selection(declaration: dict, include_band: bool = False) -> str:
    """Name the processed state a candidate-selection declaration reads."""
    rule = declaration.get("rule")
    if rule == "mixture_quantile":
        return "legacy_mixture"
    if rule == "median_reflected":
        candidate = "reflected"
    elif (
        rule == "fdr_fixed_cutoff"
        and declaration.get("method") == "by"
        and declaration.get("q") == 0.05
    ):
        candidate = {
            "papkou_original": "published_by",
            "median_reflected": "reflected_by",
        }.get(declaration.get("cutoff"))
    else:
        candidate = None
    if candidate is None:
        raise ValueError(f"No processed figure products for selection {declaration!r}.")
    if include_band and candidate == "reflected":
        raise ValueError("No band tree exists for the reflected point selection.")
    return candidate


def classification_table(context: RunContext, variant: str) -> Path:
    """The BY classification table of ``processed/classification/<variant>``."""
    directory = path("classification", variant, root=processed_root(context))
    tables = sorted(directory.glob("fdr_classification_*_method=by_q=0.05.parquet"))
    if len(tables) != 1:
        raise MissingProductError(
            f"Expected one BY classification table in {directory}, found {len(tables)}; "
            f"{MISSING_HINT}."
        )
    return tables[0]


def growth_rates(context: RunContext) -> tuple[Path, str]:
    """``processed/growth_rates/growth_rates.parquet`` and its sha256, which the products built from it record."""
    file = path("growth_rates", root=processed_root(context)) / "growth_rates.parquet"
    if not file.is_file():
        raise MissingProductError(
            f"No processed growth-rate table at {file}; {MISSING_HINT}."
        )
    return file, _sha256(file)


def publish(source: Path, kind: str, state: str | None, *, root: Path | None = None) -> Path:
    """Copy a finished product directory into ``<root>/<kind>[/<state>]``, replacing it."""
    target = path(kind, state, root=root)
    manifest = Path(source) / "manifest.json"
    if (
        manifest.is_file()
        and (target / "manifest.json").is_file()
        and (target / "manifest.json").read_bytes() == manifest.read_bytes()
    ):
        return target  # the same product is already there
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


# -- state of a product -------------------------------------------------------------------

# The four graphs of the peak figure, by analysis variant, are named by their state.
STATE_OF_VARIANT = {
    "threshold=published_fdr=none": "published",
    "threshold=reflected_fdr=none": "reflected",
    "threshold=published_fdr=by_q=0.05": "published_by",
    "threshold=reflected_fdr=by_q=0.05": "reflected_by",
}
# The state of the default analysis (config/analysis.toml): the median-reflected BY graph.
DEFAULT_STATE = "reflected_by"

# Products named by the state of their tree: output/<kind>/<label>/ -> <kind>/<state>/.
STATE_KINDS = (
    "landscape",
    "tree",
    "leaves",
    "tree_layout",
    "tree_annotations",
    "landscape_summary",
    "ridges",
    "ridge_entropy_sweeps",
    "merge_flow",
)
# Products named by the analysis variant they were built under (output/variants/<variant>/).
SHARED_SCAN_FOLDER = "band_quantile_scan"  # also holds the scan tables the stage publishes
VARIANT_KINDS = ("pooled_scatter", "peak_panel_summary", "bootstrap_band_quantile_scan")
# Products that exist once.
SINGLE_KINDS = (
    "threshold_diagnostics",
    "bootstrap_codon_entropy",
    "graph_tree_persistence",
)


def state_of_identity(identity: Mapping[str, object]) -> str:
    """The state a scientific identity belongs to (``thr``/``fdr``/``graph``/``tree``)."""
    if identity.get("graph") == "full":
        base = "full"
    elif identity.get("thr") in {"published", "reflected"}:
        base = str(identity["thr"])
        if identity.get("fdr") == "by":
            base += "_by"
        elif identity.get("fdr") not in {None, "none"}:
            raise ValueError(f"Unknown fdr {identity.get('fdr')!r} in {identity!r}.")
    else:
        raise ValueError(f"No processed state for the identity {identity!r}.")
    return base + "_band" if identity.get("tree") == "band" else base


def _created(directory: Path) -> str:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
        "created_at"
    ]


def _manifest_directories(output_root: Path, product: str) -> list[tuple[Path, Path]]:
    """``(root, product directory)`` for ``output/`` and every ``output/variants/<variant>/``."""
    roots = [output_root, *sorted((output_root / "variants").glob("*"))]
    return [
        (root, manifest.parent)
        for root in roots
        for manifest in sorted((root / product).glob("*/manifest.json"))
    ]


# The key of the default output root. It is not the key of the variant that builds the same
# graph (``variants/threshold=reflected_fdr=by_q=0.05`` is ``reflected_by``): two sources
# never share a folder. A band-quantile scan is the exception, by design: the default scan
# is published to ``band_quantile_scan/reflected_by`` by its own stage, and its survival-table
# product joins the scan tables there.
DEFAULT_ROOT_STATE = "default"
DEFAULT_ROOT_STATE_OF_KIND = {"bootstrap_band_quantile_scan": DEFAULT_STATE}


def _variant_state(root: Path, output_root: Path, kind: str | None = None) -> str:
    if root == output_root:
        return DEFAULT_ROOT_STATE_OF_KIND.get(kind, DEFAULT_ROOT_STATE)
    return STATE_OF_VARIANT.get(root.name, root.name)


@dataclass(frozen=True)
class FileSet:
    """Plain files of a stage output that the processed folder holds under ``kind[/state]``."""

    kind: str
    state: str | None
    output_dir: str
    patterns: tuple[str, ...]
    # Files picked out of a shared folder (``output/tables``): copied by name, and the
    # processed folder is not emptied first.
    flat: bool = False


def _comparison_sets() -> tuple[FileSet, ...]:
    sets = [
        FileSet(
            "comparison", state, f"variants/{variant}/analysis/clamped_comparison", ("**/*",)
        )
        for variant, state in STATE_OF_VARIANT.items()
    ]
    sets.append(
        FileSet("comparison", DEFAULT_ROOT_STATE, "analysis/clamped_comparison", ("**/*",))
    )
    return tuple(sets)


FILE_SETS = (
    FileSet("growth_rates", None, "analysis/sequence_growth_rates", ("*",)),
    *_comparison_sets(),
    FileSet("manuscript_quantities", None, "tables/manuscript_quantities", ("**/*",)),
    FileSet(
        "presentation_tables",
        None,
        "tables",
        ("*_full_vs_published.*", "*_candidate_corrections.*", "*_published_simplification.*"),
        flat=True,
    ),
    FileSet(
        "graph_descriptive_statistics",
        None,
        "tables/canonical/graph_descriptive_statistics",
        ("**/*",),
    ),
    FileSet("codon_entropy", None, "tables", ("papkou2023_codon_entropy*",), flat=True),
)


# Pending markers, quarantined and half-written state never belong in the deposit.
_JUNK = (
    ".DS_Store",
    "__pycache__",
    ".*pending*",
    ".data-current",
    "*.unusable-*",
    ".replaced-*",
    "*.tmp",
    "*.part",
)


def _is_junk(relative: Path) -> bool:
    return any(
        fnmatch.fnmatchcase(part, pattern) for part in relative.parts for pattern in _JUNK
    )


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """A glob over ``/``-separated paths: ``*`` stays within a segment, ``**/`` and a final ``/**`` cross them."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("/**", i) and i + 3 == len(pattern):
            out, i = out + "/.*", i + 3
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def _walk_files(base: Path) -> list[tuple[Path, str]]:
    """``(relative path, file name)`` of every file under ``base``, following symlinked directories.

    ``Path.glob`` does not descend into a symlinked directory, but ``output/analysis/data``
    is one (a link to its checkpoint). A link back to one of its own ancestors is not
    entered again.
    """
    base = Path(base)
    found: list[Path] = []
    for directory, names, files in os.walk(base, followlinks=True):
        here = Path(directory)
        ancestors = {
            parent.resolve()
            for parent in here.parents
            if parent == base or base in parent.parents
        }
        if here.resolve() in ancestors:
            names[:] = []
            continue
        names.sort()
        found += [
            (here / name).relative_to(base)
            for name in files
            if (here / name).is_file()
        ]
    return sorted(found)


def _walk(base: Path) -> list[Path]:
    """Every file under ``base`` except junk (``_JUNK``), sorted, relative to it."""
    return [path for path in _walk_files(base) if not _is_junk(path)]


def _matching(base: Path, patterns: Sequence[str], exclude: Sequence[str] = ()) -> list[Path]:
    """Files under ``base`` whose relative path matches a pattern and no ``exclude`` pattern."""
    include = [_glob_regex(pattern) for pattern in patterns]
    skip = [_glob_regex(pattern) for pattern in exclude]
    return [
        relative
        for relative in _walk(base)
        if any(rule.match(relative.as_posix()) for rule in include)
        and not any(rule.match(relative.as_posix()) for rule in skip)
    ]


def _matched(output_root: Path, file_set: FileSet) -> list[Path]:
    base = output_root / file_set.output_dir
    return [base / relative for relative in _matching(base, file_set.patterns)]


def _copy_files(files: Sequence[Path], base: Path, target: Path, *, replace: bool) -> None:
    if replace and target.exists():
        shutil.rmtree(target)
    for file in files:
        destination = target / file.relative_to(base)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)


# -- what the figures read from output/ ---------------------------------------------------

# Where the verbatim copies sit in the processed folder: ``stage_outputs/<path under output/>``.
STAGE_OUTPUTS = "stage_outputs"

# Parts of the default output root that the figures (and the registry's currency check) read,
# beyond the products and ``FILE_SETS`` above (which hold the saved comparison): the Full
# analysis and its record, the component checkpoint, the candidate-edge evidence, the peak checkpoint, the
# stage checkpoints (leading saddle band, functional classification), the threshold scan,
# the combined FDR scan, the biological annotation tables and the stage records kept beside the
# tables. The bootstrap is not here: its checkpoint is ``bootstrap/<state>/checkpoint``.
ROOT_MIRROR = (
    "analysis/provenance.json",
    "analysis/component_*",
    "analysis/drift_reports/**",
    "analysis/classification_probability_scans/drift_report.json",
    "analysis/peak_uncertainty/**",
    "analysis/candidate_edge_evidence/**",
    "analysis/branch_decomposition/**",
    "analysis/functional_threshold_scan/**",
    "analysis/classification_probability_scans/fdr_scan_all_cutoffs_and_methods.parquet",
    "checkpoints/**",
    "tables/.*.json",
    "tables/biological_annotation_*",
    "tables/candidate_annotations/**",
    "tables/leading_saddle_band*",
)
# ``analysis/data`` is a link to ``checkpoints/analysis/<key>/data``, which is mirrored with the
# other checkpoints: the data is stored once and ``restore`` recreates the link.
ROOT_MIRROR_EXCLUDE = ("checkpoints/bootstrap*/**",)
# What a variant root (``variants/<variant>/``) holds that the figures read, beside its saved
# comparison (``comparison/<state>``): the record that comparison builds on, the peak checkpoint
# and the peak-pair distances.
VARIANT_MIRROR = (
    "analysis/provenance.json",
    "analysis/peak_uncertainty/**",
    "analysis/peak_pair_distances/**",
)


def _mirror_roots(output_root: Path) -> list[tuple[str, Path, tuple[str, ...], tuple[str, ...]]]:
    """``(prefix under output/, root, include patterns, exclude patterns)`` of the default root and each variant."""
    roots = [("", output_root, ROOT_MIRROR, ROOT_MIRROR_EXCLUDE)]
    variants = output_root / "variants"
    if variants.is_dir():
        roots += [
            (f"variants/{root.name}/", root, VARIANT_MIRROR, ())
            for root in sorted(variants.iterdir())
            if root.is_dir()
        ]
    return roots


def collect(context: RunContext) -> list[Path]:
    """Copy the finished stage outputs of ``output/`` into ``data/processed/``, by name.

    Manifest products go to ``<kind>/<state>`` (the newest of those that map to one
    state), plain stage outputs to the folder their ``FILE_SETS`` entry names. Stage
    outputs that were not produced are skipped; ``MANIFEST.tsv`` is written by
    ``write_manifest`` afterwards.
    """
    output_root = Path(context.output_root)
    root = processed_root(context)
    written: list[Path] = []
    for kind in STATE_KINDS:
        newest: dict[str, Path] = {}
        for _, directory in _manifest_directories(output_root, kind):
            identity = json.loads((directory / "manifest.json").read_text())[
                "scientific_identity"
            ]
            try:
                state = state_of_identity(identity)
            except ValueError as error:
                warnings.warn(f"Not collected: {directory} ({error})", stacklevel=2)
                continue
            if state not in STATES:
                warnings.warn(
                    f"Not collected: {directory} has the state {state!r}, which is not "
                    f"one of {', '.join(STATES)}.",
                    stacklevel=2,
                )
                continue
            if state not in newest or _created(directory) >= _created(newest[state]):
                newest[state] = directory
        for state, directory in sorted(newest.items()):
            written.append(publish(directory, kind, state, root=root))
    for kind in VARIANT_KINDS:
        newest = {}
        for variant_root, directory in _manifest_directories(output_root, kind):
            state = _variant_state(variant_root, output_root, kind)
            if state not in newest or _created(directory) >= _created(newest[state]):
                newest[state] = directory
        folder = PROCESSED_DIRECTORY.get(kind, kind)
        for state, directory in sorted(newest.items()):
            target = path(folder, state, root=root)
            # A scan folder already holds the scan tables and their record: merge into it;
            # every other variant folder is replaced so a stale file does not survive.
            if folder != SHARED_SCAN_FOLDER and target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(directory, target, dirs_exist_ok=True)
            written.append(target)
    for kind in SINGLE_KINDS:
        directories = [d for _, d in _manifest_directories(output_root, kind)]
        if directories:
            written.append(publish(max(directories, key=_created), kind, None, root=root))
    seen: set[tuple[str, str | None]] = set()
    for file_set in FILE_SETS:
        key = (file_set.kind, file_set.state)
        files = _matched(output_root, file_set)
        if key in seen or not files:
            continue
        seen.add(key)
        target = path(file_set.kind, file_set.state, root=root)
        # The target folder is the kind's own, so it is replaced (a flat set only picks its
        # files out of a shared source folder).
        _copy_files(files, output_root / file_set.output_dir, target, replace=True)
        written.append(target)
    shutil.rmtree(root / STAGE_OUTPUTS, ignore_errors=True)  # a re-collect never keeps a stale file
    if _collect_mirror(output_root, root):
        written.append(root / STAGE_OUTPUTS)
    written += _collect_raw(Path(context.raw_data_dir), root)
    return written


def _collect_raw(raw_dir: Path, root: Path) -> list[Path]:
    """Copy the four raw files (and their README, with the credit and licence) into ``raw/``."""
    target = root / RAW_DIRECTORY
    copied = False
    for name in (*RAW_FILES, RAW_README):
        source = raw_dir / name
        if source.is_file():
            target.mkdir(parents=True, exist_ok=True)
            if not (target / name).is_file() or _sha256(target / name) != _sha256(source):
                shutil.copyfile(source, target / name)
            copied = True
    return [target] if copied else []


def missing_raw(context: RunContext) -> list[str]:
    """The raw files that are in neither ``data/raw/`` nor the processed ``raw/``."""
    raw_dir, root = Path(context.raw_data_dir), processed_root(context)
    return [
        name
        for name in RAW_FILES
        if not (raw_dir / name).is_file() and not (root / RAW_DIRECTORY / name).is_file()
    ]


def _restore_raw(raw_dir: Path, root: Path) -> list[Path]:
    """Copy ``raw/`` back to ``data/raw/``; the deposit's bytes win over a local rebuild."""
    written: list[Path] = []
    for name in (*RAW_FILES, RAW_README):
        source = root / RAW_DIRECTORY / name
        destination = raw_dir / name
        if not source.is_file():
            continue
        if destination.is_file() and _sha256(destination) == _sha256(source):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        written.append(destination)
    return written


def _collect_mirror(output_root: Path, root: Path) -> list[Path]:
    """Copy what the figures read from ``output/`` verbatim (links followed) into ``stage_outputs/``."""
    written: list[Path] = []
    for prefix, base, include, exclude in _mirror_roots(output_root):
        for relative in _matching(base, include, exclude):
            destination = root / STAGE_OUTPUTS / (prefix + relative.as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / relative, destination)  # follows a symlink: resolved content
            written.append(destination)
    return written


def _restore_mirror(output_root: Path, root: Path) -> list[Path]:
    """Copy ``stage_outputs/`` back to the same paths under ``output/``; identical files stay."""
    written: list[Path] = []
    source = root / STAGE_OUTPUTS
    if not source.is_dir():
        return written
    real_root = output_root.resolve()
    for relative in _walk(source):
        destination = output_root / relative
        # Never write through a symlinked ancestor (e.g. a link at output/analysis/data).
        if destination.parent.resolve() != real_root / relative.parent:
            warnings.warn(
                f"Not restored: {relative} would be written through a symlinked directory.",
                stacklevel=2,
            )
            continue
        if destination.is_file() and _sha256(destination) == _sha256(source / relative):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
        written.append(destination)
    written += _link_analysis_data(output_root)
    return written


def _link_analysis_data(output_root: Path) -> list[Path]:
    """Point ``output/analysis/data`` at its restored checkpoint, as the analysis stage does."""
    checkpoints = output_root / "checkpoints" / "analysis"
    leaves = sorted(d.parent for d in checkpoints.glob("*/data") if d.is_dir())
    if not leaves:
        return []
    record = output_root / "analysis" / "provenance.json"
    try:
        fingerprint = json.loads(record.read_text(encoding="utf-8")).get("fingerprint")
    except (OSError, ValueError):
        fingerprint = None
    chosen = [leaf for leaf in leaves if _fingerprint_of(leaf) == fingerprint] or (
        leaves if len(leaves) == 1 else []
    )
    if len(chosen) != 1:
        warnings.warn(
            "output/analysis/data not linked: no unique restored analysis checkpoint.",
            stacklevel=2,
        )
        return []
    link = output_root / "analysis" / "data"
    target = Path(os.path.relpath(chosen[0] / "data", link.parent))
    if link.is_symlink() and Path(os.readlink(link)) == target:
        return []
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            warnings.warn(
                f"{link} is a real directory; move it aside, then rerun restore.", stacklevel=2
            )
            return []
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return [link]


def _fingerprint_of(leaf: Path) -> object:
    try:
        return json.loads((leaf / "provenance.json").read_text(encoding="utf-8")).get("fingerprint")
    except (OSError, ValueError):
        return None


def restore(context: RunContext) -> list[Path]:
    """Copy the plain-file folders of ``data/processed/`` back to where the figure inputs read them.

    The inverse of ``collect`` for ``FILE_SETS``; manifest products are read in place
    and need no restore. Files already identical are left alone.
    """
    output_root = Path(context.output_root)
    root = processed_root(context)
    written: list[Path] = []
    seen: set[tuple[str, str | None]] = set()
    for file_set in FILE_SETS:
        key = (file_set.kind, file_set.state)
        source = path(file_set.kind, file_set.state, root=root)
        # One state has one folder; the first output location of it is the one restored.
        if key in seen or not source.is_dir():
            continue
        seen.add(key)
        for relative in _walk(source):
            file = source / relative
            destination = output_root / file_set.output_dir / relative
            if destination.is_file() and _sha256(destination) == _sha256(file):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, destination)
            written.append(destination)
    written += _restore_mirror(output_root, root)
    written += _restore_raw(Path(context.raw_data_dir), root)
    return written


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    # Every file, dotfiles and files behind a symlinked directory included.
    return [
        root / relative
        for relative in _walk_files(root)
        if relative.as_posix() != MANIFEST_NAME and relative.name != ".DS_Store"
    ]


def write_manifest(root: Path) -> Path:
    """Write ``MANIFEST.tsv`` (path, bytes, sha256) for every file under ``root``."""
    root = Path(root)
    lines = ["\t".join(MANIFEST_HEADER)]
    for file in _files(root):
        lines.append(
            f"{file.relative_to(root).as_posix()}\t{file.stat().st_size}\t{_sha256(file)}"
        )
    target = root / MANIFEST_NAME
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_deposit_readme(root: Path) -> Path:
    """Write the deposit README (with the Papkou et al. credit and licence) into ``root``."""
    target = Path(root) / "README.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEPOSIT_README, encoding="utf-8")
    return target


def _read_manifest(root: Path) -> dict[str, tuple[int, str]]:
    target = root / MANIFEST_NAME
    if not target.is_file():
        raise FileNotFoundError(f"{target} is missing; {MISSING_HINT}.")
    rows = target.read_text(encoding="utf-8").splitlines()
    if not rows or tuple(rows[0].split("\t")) != MANIFEST_HEADER:
        raise ValueError(f"{target} has an unexpected header.")
    entries: dict[str, tuple[int, str]] = {}
    for row in rows[1:]:
        name, size, digest = row.split("\t")
        entries[name] = (int(size), digest)
    return entries


def check_manifest(root: Path) -> list[str]:
    """Return every problem found when comparing ``root`` with its ``MANIFEST.tsv``.

    Problems are ``missing``, ``extra``, ``size`` and ``hash``; an empty list means
    the folder is exactly what the manifest records.
    """
    root = Path(root)
    expected = _read_manifest(root)
    present = {file.relative_to(root).as_posix(): file for file in _files(root)}
    problems = [f"missing: {name}" for name in sorted(expected.keys() - present.keys())]
    problems += [f"extra: {name}" for name in sorted(present.keys() - expected.keys())]
    for name in sorted(expected.keys() & present.keys()):
        size, digest = expected[name]
        file = present[name]
        if file.stat().st_size != size:
            problems.append(f"size: {name}")
        elif _sha256(file) != digest:
            problems.append(f"hash: {name}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("collect", "restore", "write", "check"))
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command in {"collect", "restore"}:
        context = RunContext(processed_root=args.root) if args.root else RunContext()
        for item in (collect if args.command == "collect" else restore)(context):
            print(item)
        if args.command == "restore" and (missing := missing_raw(context)):
            print(
                "Missing raw input(s) "
                + ", ".join(missing)
                + f": neither data/raw nor the processed raw/ has them; {MISSING_HINT}.",
                file=sys.stderr,
            )
            return 1
        return 0
    if args.root is None:
        args.root = RunContext().processed_root
    if args.command == "write":
        write_deposit_readme(args.root)
        print(write_manifest(args.root))
        return 0
    try:
        problems = check_manifest(args.root)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
