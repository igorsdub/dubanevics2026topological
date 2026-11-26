"""Produce-or-load for build stages, keyed by fingerprint (ADR 0013, D1).

A Python counterpart of DrWatson.jl's ``produce_or_load``. A stage declares its
scientific settings, input files and code files as a ``Stage``.
``produce_or_load`` derives a fingerprint from those declarations, reuses a
published checkpoint whose fingerprint matches and whose artefact hashes verify,
and otherwise produces into a private temporary directory and publishes it
atomically through ``checkpoint.py``. A published checkpoint is replaced in place only when the same key is forced.
A different key is a new folder beside the old one.

Public surface, and nothing more:

- ``Stage(name, settings, inputs, code, root, execution_keys, libraries)``: the
  declarations a fingerprint is derived from.
- ``stage_fingerprint(stage)``: the sha256 over the stage name, its scientific
  settings and the content hashes of its inputs. Code-file hashes are not part
  of it.
- ``savename(settings, name_keys, *, execution_keys=(), stage=None)``: the
  readable ``key=value`` label of a result folder. ``name_keys`` maps each
  alias to a dotted path in ``settings``. The label is not part of the
  fingerprint.
- ``result_directory(step_root, stage)``: ``<step_root>/<savename>_key=<8 hex>``,
  or ``<step_root>/key=<8 hex>`` when the stage has no aliases.
- ``saved_results(root)``: one row per published provenance under ``root``
  (the step, its folder, the commit label, and a source-changed warning).
- ``produce_or_load(stage, produce, *, checkpoint_dir, force, on_ready)``:
  returns a ``Checkpoint``. The ``produce`` callable receives an empty directory
  to write artefacts into.
- ``publication_state(checkpoint_dir)``: ``{"pending": bool, "backup": bool}``,
  whether a publication of that checkpoint is unsettled (its pending marker
  exists, or the backup of the previous checkpoint is still kept). A reader that
  must not trust a checkpoint mid-publication asks this instead of spelling the
  helper's file names itself.
- ``load_checkpoint(stage, read, *, checkpoint_dir, retry_hint)``: the read-only
  counterpart for readers that must never produce, recover or move anything. It
  returns what ``read(checkpoint)`` returns, called under the lock, and refuses
  while ``publication_state`` reports an unsettled publication.
- ``Checkpoint(path, provenance, reused)`` and ``Checkpoint.export_to(destination,
  names)``: a verified, published checkpoint, and a byte-for-byte copy of its
  artefacts to paths a stage already uses.
- ``PROVENANCE_NAME``: the name of the provenance record, reserved at the top
  level of a checkpoint.
- ``SCHEMA``: the version of that record.
- ``IGNORED_NAMES``: operating-system metadata file names that are neither
  produced nor verified. It is defined in ``checkpoint.py`` and re-exported
  here, so the version digest and the artefact hashes skip the same names.

Git stamping, DrWatson's ``tag``/``tagsave``, lives in the sibling ``git_stamp``
module (standard library only: ``GIT_KEY``, ``git_stamp``, ``tag``, ``untag``);
this module imports it but does not re-export any of it. ``produce_or_load`` tags
every provenance record it writes with the commit and a dirty flag of the stage
root, under ``GIT_KEY`` (``"git"``), taken before ``produce`` runs, so it
describes the state the work started from. The stamp is a pointer for people. It
never enters a fingerprint, a reuse decision or the validation of a published
checkpoint, so a record written before stamping existed is reused like any other.
A caller that digests a whole provenance record must drop the stamp first
(``untag``).

Extraction boundary: this module is meant to be lifted into a standalone package
later, so three rules hold. (1) It imports only the standard library and its
sibling ``checkpoint`` and ``git_stamp`` modules, which themselves import only the
standard library.
It knows nothing of a host project's configuration files, scientific libraries
or task runner. (2) It has no project-specific defaults: the library names
recorded in the provenance versions and the execution keys stripped before
hashing are supplied by the caller and empty by default (like DrWatson's
ignores). A host project passes its values once, through its own stage factory,
and builds every stage through that factory rather than by calling ``Stage``
directly. (3) Its public surface is the generic one above and nothing more.
Behaviour that belongs to one kind of stage (resuming partial work, keeping
reference copies, command-line flags) is layered on by the caller and is never
pushed down into this helper. The host project's tests enforce these rules.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from . import git_stamp as _stamping
from .checkpoint import (
    IGNORED_NAMES,
    backup_path,
    checkpoint_lock,
    publish_checkpoint,
    recover_checkpoint,
)

PROVENANCE_NAME = "provenance.json"
SCHEMA = 1


@dataclass(frozen=True)
class Stage:
    """What a stage depends on, and nothing about how it is scheduled.

    ``settings`` holds scientific settings only. ``execution_keys`` names the
    scheduling settings (worker counts, chunk sizes, ...) that the caller wants
    ignored; they are stripped at any depth of ``settings`` before hashing, so
    changing how a stage is scheduled never changes its fingerprint. Nothing is
    ignored by default. ``inputs`` and ``code`` are files hashed by content.
    Paths are keyed relative to ``root`` when inside it, so the fingerprint does
    not depend on where the repository is checked out. ``libraries`` lists
    distribution names whose installed versions are recorded in the provenance
    (not hashed); none by default. ``execution_keys`` and ``libraries`` must be
    collections of names: a bare string is rejected, since it would be matched
    by substring (``execution_keys``) or read one character at a time
    (``libraries``).
    """

    name: str
    settings: Mapping[str, Any]
    inputs: Sequence[Path] = ()
    code: Sequence[Path] = ()
    root: Path | None = None
    execution_keys: Collection[str] = ()
    libraries: Sequence[str] = ()
    name_keys: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("execution_keys", "libraries"):
            if isinstance(getattr(self, name), str | bytes):
                raise TypeError(
                    f"Stage {name} must be a collection of names, not a bare "
                    f"string; wrap a single name in a tuple."
                )
        keys = self.name_keys
        if isinstance(keys, str | bytes) or not isinstance(keys, Mapping):
            raise TypeError(
                "Stage name_keys must be a mapping of alias to path, not a bare "
                "string; a string would be read one character at a time."
            )
        copied: dict[str, str] = {}
        for alias, path in keys.items():
            if not isinstance(alias, str) or not isinstance(path, str):
                raise TypeError(
                    "Stage name_keys must be a mapping of alias to path."
                )
            copied[alias] = path
        object.__setattr__(self, "name_keys", copied)


@dataclass(frozen=True)
class Checkpoint:
    """A verified, published checkpoint directory and its provenance record."""

    path: Path
    provenance: dict[str, Any]
    reused: bool = field(default=False)

    def export_to(
        self, destination: Path, names: Mapping[str, str] | None = None
    ) -> list[str]:
        """Copy artefacts byte-for-byte into ``destination``; return names written.

        This lets a stage keep its existing output paths. Each file is copied
        to a temporary sibling and moved with ``os.replace``; a destination
        file with identical bytes is left untouched. ``names`` maps artefact
        paths to different destination paths. The provenance file is not copied.
        """
        written: list[str] = []
        base = destination.resolve()
        source_base = self.path.resolve()
        for relative, digest in self.provenance["artefacts"].items():
            source = _contained(source_base, relative, "artefact")
            name = (names or {}).get(relative, relative)
            target = _contained(base, name, "destination name")
            if target.is_file() and _sha256(target) == digest:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source, temporary)
                if _sha256(temporary) != digest:
                    raise ValueError(
                        f"Artefact {relative!r} does not match its recorded "
                        "sha256; refusing to export it."
                    )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            written.append(relative)
        return written


def _contained(base: Path, name: str, what: str) -> Path:
    """Resolve ``name`` under ``base``; reject absolute names and escapes."""
    candidate = Path(name)
    if candidate.is_absolute() or not candidate.parts:
        raise ValueError(f"Unsafe {what} {name!r}: names must be relative.")
    resolved = (base / candidate).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Unsafe {what} {name!r}: it would escape {base}.")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scientific(value: Any, execution_keys: Collection[str]) -> Any:
    """Copy ``value`` as strict JSON data, dropping execution keys at any depth."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Stage settings keys must be strings, got {type(key).__name__}."
                )
            if key not in execution_keys:
                result[key] = _scientific(item, execution_keys)
        return result
    if type(value) in (list, tuple):
        return [_scientific(item, execution_keys) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Stage settings must not contain NaN or infinity.")
        return value
    raise TypeError(
        f"Stage settings must be plain JSON data, got {type(value).__name__}."
    )


def _key(path: Path, root: Path | None) -> str:
    """Key ``path`` relative to ``root``; reject anything outside it.

    A path that lies under ``root`` lexically keeps its in-root key even when it
    is a symlink to storage elsewhere (a data directory linked to a cluster
    volume); its content is still hashed through the link. Only when the lexical
    test fails are both paths resolved, so a symlinked checkout still keys the
    same as the real one. A path (or root) with a ``..`` part is never judged
    lexically: ``link/../x`` is the parent of the link's target, not of the
    link, so collapsing it by name could key one file as another or let a file
    outside the root pass as inside it. Such paths are always resolved.
    """
    if root is None:
        raise ValueError(
            f"Stage declares {str(path)!r} but has no root to key it against."
        )
    if ".." not in Path(path).parts and ".." not in Path(root).parts:
        lexical_base = Path(os.path.abspath(root))
        lexical = Path(os.path.abspath(path))
        if lexical.is_relative_to(lexical_base):
            return lexical.relative_to(lexical_base).as_posix()
    base = Path(root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(
            f"Declared path {str(path)!r} is outside the stage root {base}."
        )
    return resolved.relative_to(base).as_posix()


def _declared(stage: Stage) -> dict[str, Any]:
    def hashed(paths: Sequence[Path]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        declared_as: dict[str, Path] = {}
        for declared in paths:
            path = Path(declared)
            key = _key(path, stage.root)
            if key in declared_as:
                raise ValueError(
                    f"Stage {stage.name!r} declares two paths with the same key "
                    f"{key!r}: {str(declared_as[key])!r} and {str(path)!r}."
                )
            declared_as[key] = path
            hashes[key] = _sha256(path)
        return hashes

    return {
        "stage": stage.name,
        "settings": _scientific(stage.settings, stage.execution_keys),
        "inputs": hashed(stage.inputs),
        "code": hashed(stage.code),
    }


def _canonical_digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key_material(declared: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a declaration that is the key: no code-file hashes."""
    return {
        "stage": declared["stage"],
        "settings": declared["settings"],
        "inputs": declared["inputs"],
    }


def stage_fingerprint(stage: Stage) -> str:
    """Return the sha256 of the stage name, its settings and its input hashes.

    Code-file hashes stay in the provenance label. Editing a code file, including
    a docstring, does not change this value.
    """
    return _canonical_digest(_key_material(_declared(stage)))


_ALIAS = re.compile(r"[a-z][a-z0-9]*")
_LABEL_LIMIT = 100


def _format_leaf(value: Any, stage: str | None) -> str:
    """Write one savename value. Floats keep three significant digits."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(_where(stage, "a savename value must be finite"))
        if value == 0.0:
            return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
        text = format(value, ".3g")
        if "e" in text:
            exponent = int(text.partition("e")[2])
            if -4 < exponent < 6:
                fixed = format(float(text), "f").rstrip("0").rstrip(".")
                return fixed if "." in fixed else f"{fixed}.0"
            return text
        return text if "." in text else f"{text}.0"
    if isinstance(value, str):
        # ``urllib.parse.quote`` leaves ``~`` alone. The label grammar encodes it.
        pieces: list[str] = []
        for character in value:
            if character.isascii() and (character.isalnum() or character in "._+-"):
                pieces.append(character)
            else:
                pieces.append("".join(f"%{byte:02X}" for byte in character.encode()))
        return "".join(pieces)
    raise TypeError(_where(stage, "a savename value must be a scalar"))


def _where(stage: str | None, message: str) -> str:
    if stage:
        return f"{message} ({stage!r})"
    return message


def savename(
    settings: Mapping[str, Any],
    name_keys: Mapping[str, str],
    *,
    execution_keys: Collection[str] = (),
    stage: str | None = None,
) -> str:
    """Return ``alias=value`` pairs joined by ``_``, sorted as whole strings.

    ``name_keys`` maps an alias to a dotted path in ``settings``. Execution keys
    are dropped before the path is read. The label is for the folder name; it
    is never hashed into a fingerprint. An empty ``name_keys`` returns ``""``.
    """
    if isinstance(execution_keys, str | bytes):
        raise TypeError(
            _where(
                stage,
                "execution_keys must be a collection of names, not a bare string",
            )
        )
    if isinstance(name_keys, str | bytes) or not isinstance(name_keys, Mapping):
        raise TypeError(
            _where(stage, "name_keys must be a mapping of alias to path")
        )
    if not name_keys:
        return ""
    try:
        scientific = _scientific(settings, execution_keys)
    except (TypeError, ValueError) as error:
        raise type(error)(_where(stage, str(error))) from error
    pairs: list[str] = []
    for alias, path in name_keys.items():
        if not isinstance(alias, str) or not isinstance(path, str):
            raise TypeError(
                _where(stage, "name_keys must be a mapping of alias to path")
            )
        if _ALIAS.fullmatch(alias) is None:
            raise ValueError(_where(stage, f"alias {alias!r} is not a savename alias"))
        parts = path.split(".")
        if not path or any(part == "" for part in parts):
            raise ValueError(_where(stage, f"path {path!r} is not a dotted path"))
        if any(part in execution_keys for part in parts):
            raise ValueError(
                _where(stage, f"alias {alias!r} path {path!r} crosses an execution key")
            )
        cursor: Any = scientific
        walked: list[str] = []
        for part in parts:
            walked.append(part)
            if not isinstance(cursor, Mapping) or part not in cursor:
                raise KeyError(
                    _where(
                        stage,
                        f"alias {alias!r} path {path!r} is missing at {'.'.join(walked)}",
                    )
                )
            cursor = cursor[part]
        try:
            written = _format_leaf(cursor, stage)
        except (TypeError, ValueError) as error:
            raise type(error)(
                _where(stage, f"alias {alias!r} path {path!r}: {error}")
            ) from error
        pairs.append(f"{alias}={written}")
    label = "_".join(sorted(pairs))
    if len(label) > _LABEL_LIMIT:
        raise ValueError(
            _where(stage, f"savename is longer than {_LABEL_LIMIT} characters")
        )
    return label


def result_directory(step_root: Path, stage: Stage) -> Path:
    """The folder of ``stage`` under ``step_root``: parameters, then ``key=``.

    The folder carries the first eight hex characters of the key. The full key
    is stored in the provenance. A stage with no ``name_keys`` has no parameter
    pairs and no leading underscore.
    """
    fingerprint = stage_fingerprint(stage)
    label = savename(
        stage.settings,
        stage.name_keys,
        execution_keys=stage.execution_keys,
        stage=stage.name,
    )
    short = fingerprint[:8]
    name = f"{label}_key={short}" if label else f"key={short}"
    return Path(step_root) / name


def _write_json_atomic(path: Path, value: Any) -> None:
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


def _versions(libraries: Sequence[str]) -> dict[str, str]:
    """Python plus each named distribution that is installed (absent ones skip)."""
    versions = {"python": platform.python_version()}
    for library in libraries:
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _artefact_hashes(directory: Path) -> dict[str, str]:
    """Hash every file except the top-level record and ``IGNORED_NAMES``."""
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.name in IGNORED_NAMES:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink {path.name!r} is not allowed in a checkpoint.")
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and relative != PROVENANCE_NAME:
            hashes[relative] = _sha256(path)
    return hashes


def _validate(directory: Path) -> dict[str, Any]:
    """Structural check: the record parses and every artefact hash verifies."""
    record = json.loads((directory / PROVENANCE_NAME).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError("provenance record is not an object")
    if record.get("schema") != SCHEMA or not record.get("artefacts"):
        raise ValueError("unsupported or empty provenance record")
    if not isinstance(record.get("fingerprint"), str):
        raise TypeError("provenance record has no fingerprint")
    if _artefact_hashes(directory) != record["artefacts"]:
        raise ValueError("artefacts do not match their recorded hashes")
    return record


def _fingerprint_of(record: dict[str, Any]) -> str | None:
    return record.get("fingerprint")


def _pending_path(checkpoint_dir: Path) -> Path:
    """Where the pending marker of ``checkpoint_dir`` lives: beside it, hidden."""
    return checkpoint_dir.parent / f".{checkpoint_dir.name}.pending"


def publication_state(checkpoint_dir: Path) -> dict[str, bool]:
    """Say whether a publication of ``checkpoint_dir`` is unsettled.

    ``pending`` is true while the helper's marker exists: a publication is in
    flight or was interrupted, and the checkpoint may not be the one the next
    ``produce_or_load`` call settles on. ``backup`` is true while the previous
    checkpoint is still kept beside it, which happens during a publication and
    after a crash or a failed cleanup. Both are false for a settled checkpoint
    and for one that does not exist. This only looks at the file system: it takes
    no lock and creates nothing. Without ``checkpoint_lock`` the answer can go
    stale at once, so a caller that acts on it holds the lock first.
    """
    return {
        "pending": _pending_path(checkpoint_dir).exists(),
        "backup": backup_path(checkpoint_dir).exists(),
    }


def _identity_of(declared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": declared["stage"],
        "parameters": declared["settings"],
        "upstream": declared["inputs"],
    }


def _lock_digest(stage: Stage) -> str | None:
    """sha256 of ``uv.lock`` when the stage root has one. A label, not a key."""
    if stage.root is None:
        return None
    path = Path(stage.root) / "uv.lock"
    if not path.is_file():
        return None
    return _sha256(path)


def _distribution_commit(library: str) -> str | None:
    """The VCS commit of an installed distribution, when its metadata records one."""
    try:
        dist = metadata.distribution(library)
    except metadata.PackageNotFoundError:
        return None
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    commit = (payload.get("vcs_info") or {}).get("commit_id")
    return commit if isinstance(commit, str) else None


def _label_block(stage: Stage, declared: Mapping[str, Any]) -> dict[str, Any]:
    """Code digests, the lockfile and dependency commits. Never part of the key."""
    dependencies = {}
    for library in stage.libraries:
        commit = _distribution_commit(library)
        if commit is not None:
            dependencies[library] = {"commit": commit}
    return {
        "code": declared["code"],
        "uv_lock": _lock_digest(stage),
        "dependencies": dependencies,
    }


def _same_identity(record: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    stored = record.get("identity")
    return isinstance(stored, Mapping) and dict(stored) == dict(identity)


def _refuse_short_key_collision(
    step: Path, leaf: Path, fingerprint: str, identity: Mapping[str, Any]
) -> None:
    """Two different keys that share eight hex characters are an error."""
    if not step.is_dir():
        return
    short = fingerprint[:8]
    for child in step.iterdir():
        if not child.is_dir() or child.resolve() == leaf.resolve():
            continue
        name = child.name
        if name != f"key={short}" and not name.endswith(f"_key={short}"):
            continue
        record_path = child / PROVENANCE_NAME
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Checkpoint {child} shares key={short} but its provenance "
                f"cannot be read ({error})."
            ) from error
        if record.get("fingerprint") != fingerprint or not _same_identity(
            record, identity
        ):
            raise RuntimeError(
                f"Short key key={short} is shared by {leaf.name} and {child.name} "
                "but the full key or the stored identity differs. Nothing was "
                "loaded and nothing was computed."
            )


def saved_results(root: Path) -> list[dict[str, Any]]:
    """List published results under ``root`` without reading them as current.

    Each row has ``step``, ``folder``, ``commit`` and ``warning``. ``warning``
    is ``source changed since`` when a declared code file or ``uv.lock`` no
    longer matches the label, or git history moved those files after the
    labelled commit.     It is ``upstream replaced since`` when the record still cites a fingerprint
    that a forced rewrite recorded under ``replaces``. The listing never
    recomputes a stage.
    """
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    records = [root / PROVENANCE_NAME, *sorted(root.glob("*/" + PROVENANCE_NAME))]
    records.extend(sorted(root.glob("*/*/" + PROVENANCE_NAME)))
    seen: set[Path] = set()
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in records:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or "fingerprint" not in record:
            continue
        folder = path.parent
        step = folder.parent.name if folder.parent != root else folder.name
        if folder == root:
            step = root.name
        warning = _source_warning(record, folder)
        row = {
            "step": record.get("stage", step),
            "folder": str(folder),
            "commit": (record.get("git") or {}).get("commit"),
            "warning": warning,
        }
        rows.append(row)
        loaded.append((row, record))
    replaced = set()
    for _, record in loaded:
        previous = record.get("replaces")
        if isinstance(previous, Mapping) and isinstance(previous.get("fingerprint"), str):
            replaced.add(previous["fingerprint"])
    if replaced:
        for row, record in loaded:
            if row["warning"]:
                continue
            text = json.dumps(_key_material(record["declared"])) if isinstance(
                record.get("declared"), Mapping
            ) else ""
            own = record.get("fingerprint")
            if any(old in text and old != own for old in replaced):
                row["warning"] = "upstream replaced since"
    return rows


def _source_warning(record: Mapping[str, Any], folder: Path) -> str | None:
    label = record.get("label") if isinstance(record.get("label"), Mapping) else {}
    code = label.get("code") if isinstance(label, Mapping) else None
    if not isinstance(code, Mapping):
        declared = record.get("declared")
        code = declared.get("code") if isinstance(declared, Mapping) else {}
    if not isinstance(code, Mapping):
        code = {}
    root = _git_root(folder)
    changed = False
    for key, digest in code.items():
        path = Path(str(key))
        if not path.is_file() and root is not None:
            path = root / str(key)
        if path.is_file() and isinstance(digest, str) and _sha256(path) != digest:
            changed = True
            break
    lock_digest = label.get("uv_lock") if isinstance(label, Mapping) else None
    if root is not None and isinstance(lock_digest, str):
        lock = root / "uv.lock"
        if lock.is_file() and _sha256(lock) != lock_digest:
            changed = True
    commit = (record.get("git") or {}).get("commit") if isinstance(record.get("git"), Mapping) else None
    if root is not None and isinstance(commit, str) and _git_touched(root, commit, code):
        changed = True
    if changed:
        return "source changed since"
    return None


def _git_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_touched(root: Path, commit: str, code: Mapping[str, Any]) -> bool:
    paths = [str(key) for key in code if not str(key).startswith("/")]
    if not paths:
        return False
    try:
        completed = subprocess.run(
            ["git", "log", "--oneline", f"{commit}..HEAD", "--", *paths],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return bool(completed.stdout.strip())


def load_checkpoint[T](
    stage: Stage,
    read: Callable[[Checkpoint], T],
    *,
    checkpoint_dir: Path,
    retry_hint: str | None = None,
    lock: bool = True,
) -> T:
    """Read the current checkpoint of ``stage``; never produce, recover or move.

    The read-only counterpart of ``produce_or_load``, for notebooks and other
    readers. It holds ``checkpoint_lock`` while it reads, so no writer can publish
    between the checks and ``read``, and it changes nothing on disk (apart from
    the lock file). It refuses, rather than repairs:

    - ``RuntimeError`` while a publication is unfinished (``publication_state``
      reports a pending marker or a kept backup, asked under the lock):
      recovery belongs to an explicit command, so the reader leaves the pending
      state as it is;
    - ``FileNotFoundError`` when there is no checkpoint, or when the one there has
      a different fingerprint (other settings, inputs or code), so a stale result
      is never shown;
    - ``RuntimeError`` when the checkpoint is unreadable or its artefacts no
      longer match their recorded sha256 hashes.

    ``read(checkpoint)`` runs under the lock on the verified checkpoint and its
    result is returned; read every table inside it. ``retry_hint`` names the
    command that produces the checkpoint, for the messages. ``lock=False`` skips
    the lock, which is nonblocking and exclusive, so concurrent readers would
    refuse each other; only a caller that knows no process writes the checkpoint
    may pass it (concurrent jobs that start together and read a checkpoint that
    was published before any of them began).
    """
    declared = _declared(stage)
    fingerprint = _canonical_digest(_key_material(declared))
    identity = _identity_of(declared)
    parent = checkpoint_dir.parent
    label = stage.name
    command = (
        f"`{retry_hint}`" if retry_hint else "the command that produces this checkpoint"
    )
    missing = FileNotFoundError(
        f"No current {label} checkpoint for these settings and inputs in "
        f"{checkpoint_dir}; run {command}."
    )
    if not parent.is_dir():
        raise missing
    with checkpoint_lock(checkpoint_dir, label) if lock else contextlib.nullcontext():
        if any(publication_state(checkpoint_dir).values()):
            raise RuntimeError(
                f"The {label} checkpoint was left mid-publication; rerun "
                f"{command} to finish recovery."
            )
        if not checkpoint_dir.is_dir():
            raise missing
        try:
            record = _validate(checkpoint_dir)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RuntimeError(
                f"The {label} checkpoint in {checkpoint_dir} is unreadable or does "
                f"not match its recorded hashes ({error}); rerun {command}."
            ) from error
        if record["fingerprint"] != fingerprint or (
            "identity" in record and not _same_identity(record, identity)
        ):
            raise missing
        return read(Checkpoint(checkpoint_dir, record, reused=True))


def produce_or_load(
    stage: Stage,
    produce: Callable[[Path], None],
    *,
    checkpoint_dir: Path,
    force: bool = False,
    on_ready: Callable[[Checkpoint], None] | None = None,
) -> Checkpoint:
    """Load the checkpoint for ``stage`` or produce and publish a new one.

    ``produce(directory)`` must write the stage's artefacts (any files and
    subdirectories, but not ``provenance.json``) into the fresh, empty
    ``directory`` it is given. The helper then hashes them, writes
    ``provenance.json`` (fingerprint, declarations, artefact sha256s,
    ``created_at``, library versions, and the git stamp of ``stage.root``, or of
    the checkpoint's parent directory when the stage has no root, under
    ``GIT_KEY``, taken before ``produce`` runs) and publishes the directory to
    ``checkpoint_dir`` by atomic rename, holding ``checkpoint_lock`` throughout.
    A failing ``produce`` leaves the existing checkpoint untouched and removes
    its staging directory; only the lock file and the parent directory remain
    created. ``force`` reproduces even when a matching checkpoint exists; the
    old one is replaced only by a validated new one. ``on_ready(checkpoint)``,
    when given, runs on the verified checkpoint (reused or freshly published)
    while the lock is still held, so an export cannot race another process's
    publish; an exception from it propagates after the lock is released.
    Raises ``RuntimeError`` when another process holds the checkpoint.
    """
    # Declare once: hashing the inputs twice would cost a second read of large
    # files and let the record's declarations drift from the fingerprint.
    declared = _declared(stage)
    fingerprint = _canonical_digest(_key_material(declared))
    identity = _identity_of(declared)
    parent = checkpoint_dir.parent
    created_parent = not parent.exists()
    parent.mkdir(parents=True, exist_ok=True)
    _refuse_short_key_collision(parent, checkpoint_dir, fingerprint, identity)
    pending = _pending_path(checkpoint_dir)
    label = stage.name
    try:
        with checkpoint_lock(checkpoint_dir, label):
            current = recover_checkpoint(
                checkpoint_dir,
                pending,
                label,
                _validate,
                expected_fingerprint=fingerprint,
                fingerprint_of=_fingerprint_of,
            )
            if (
                current is not None
                and not force
                and current["fingerprint"] == fingerprint
                and ("identity" not in current or _same_identity(current, identity))
            ):
                ready = Checkpoint(checkpoint_dir, current, reused=True)
                if on_ready is not None:
                    on_ready(ready)
                return ready

            # Stamp first: the commit and dirty flag then describe the state the work
            # started from, not whatever the producer or a concurrent edit left behind.
            stamp = _stamping.git_stamp(
                stage.root if stage.root is not None else parent
            )
            staging = parent / f".{checkpoint_dir.name}.stage-{uuid.uuid4().hex}"
            try:
                payload = staging / "payload"
                payload.mkdir(parents=True)
                produce(payload)
                if (payload / PROVENANCE_NAME).exists():
                    raise ValueError(f"{PROVENANCE_NAME} is reserved for the helper.")
                artefacts = _artefact_hashes(payload)
                if not artefacts:
                    raise ValueError(f"Stage {label!r} produced no artefacts.")
                record = {
                    "schema": SCHEMA,
                    "stage": stage.name,
                    "fingerprint": fingerprint,
                    "identity": identity,
                    "declared": declared,
                    "artefacts": artefacts,
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "versions": _versions(stage.libraries),
                    "label": _label_block(stage, declared),
                    _stamping.GIT_KEY: stamp,
                }
                if force and current is not None:
                    record["replaces"] = {
                        "fingerprint": current.get("fingerprint"),
                        "git": current.get("git"),
                        "content": _canonical_digest(current.get("artefacts")),
                    }
                _write_json_atomic(payload / PROVENANCE_NAME, record)
                # The marker exists only while publication can be half-done, so a
                # failing producer leaves no trace.
                pending.write_text("", encoding="utf-8")
                published = publish_checkpoint(
                    payload,
                    checkpoint_dir,
                    pending,
                    label,
                    _validate,
                    expected_fingerprint=fingerprint,
                    fingerprint_of=_fingerprint_of,
                )
                ready = Checkpoint(checkpoint_dir, published, reused=False)
                if on_ready is not None:
                    on_ready(ready)
                return ready
            finally:
                shutil.rmtree(staging, ignore_errors=True)
    finally:
        # A refused produce must not leave the step directory behind: the lock
        # file is the only thing parent.mkdir created, and callers treat a
        # missing step as "nothing was published".
        if created_parent and parent.is_dir() and not checkpoint_dir.exists():
            lock_path = parent / f".{checkpoint_dir.name}.lock"
            lock_path.unlink(missing_ok=True)
            try:
                parent.rmdir()
            except OSError:
                pass
