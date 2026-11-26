"""Identity of a bootstrap checkpoint: what decides its scientific result, and the helpers
that compare, read and restore checkpoint directories.

A checkpoint is the working slot ``output/analysis/bootstrap_smoothed_tree/`` (or the
``checkpoint/`` folder of a processed ``bootstrap/<state>/`` product, which has the same
layout). Its scientific fingerprint is the result settings, the candidate functional
threshold, the input dataset digests and the growth-rate settings; never the digests of
code files, which change without changing the question asked. ``iter_processed_checkpoints``
finds the processed checkpoints that match a configuration, so a run restores from
``data/processed`` instead of drawing again.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import time
import uuid
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..analysis.checkpoint import checkpoint_lock

CHECKPOINT_SUBDIR = "checkpoint"
SLOT_RECORD = "provenance.json"
LOCK_TIMEOUT = 60.0  # seconds to wait for another process holding a lock


class AmbiguousCheckpointError(ValueError):
    """Several processed checkpoints match and none can be chosen safely."""


class InconsistentCheckpointError(ValueError):
    """A checkpoint whose summary settings disagree with its own fingerprint."""


def digest8(value: Any) -> str:
    """First 8 hex of the sha256 of ``value``'s sorted-key JSON.

    The suffix of a mirror directory that shares its name with another run, and the
    ``fingerprint8`` of a result dataset (``result_dataset``). Of a scientific
    fingerprint it equals the first 8 hex of the chunk directory name.
    """
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _is_volatile(key: str) -> bool:
    """Per-file digests of code (sources, lock file, installed packages)."""
    return key.endswith(".py") or key == "uv.lock" or ":" in key


def scientific_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """What decides the scientific result of a checkpoint.

    The result settings (with the constant ``noise_model`` marker, so a run drawn
    with the pooled standard error never matches one drawn with the replicate
    standard error), the candidate functional threshold and the input dataset digests; the
    digests of code files and the derived-checkpoint provenance are left out because
    they change without changing the question asked. The growth-rate checkpoint
    contributes its settings and input digests (see ``growth_identity``); its output
    hash is excluded.
    """
    draw_config = dict(fingerprint["config"])
    # Historical stage fingerprints included this derived-tree setting. It never
    # changes sampled landscapes, so translate old and current records to the same
    # draw identity while leaving it in checkpoint summary/settings for re-filtering.
    draw_config.pop("band_quantile", None)
    draw_config.pop("core_threshold", None)
    draw_config.pop("consensus_peak_frequency", None)
    value = {
        "config": draw_config,
        "candidate_functional_threshold": fingerprint.get(
            "candidate_functional_threshold"
        ),
        "datasets": {
            key: digest
            for key, digest in fingerprint.get("files", {}).items()
            if not _is_volatile(key)
        },
    }
    if fingerprint.get("candidate_selection") is not None:
        selection = dict(fingerprint["candidate_selection"])
        # Older fingerprints included an implementation digest here. It is stage
        # provenance, not evidence that the selected functional set differs.
        selection.pop("fdr_loader_sha256", None)
        value["candidate_selection"] = selection
    growth = growth_identity(fingerprint.get("growth_rates_checkpoint"))
    if growth is not None:
        value["growth"] = growth
    return json.loads(json.dumps(value))


def growth_identity(
    growth_provenance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """What of the growth-rate checkpoint decides the draws' ``se_pooled``.

    Its settings (e.g. ``elapsed_time``) and non-volatile input digests (BarSeq
    export); never the sha256 of ``growth_rates.parquet``, which depends on numerics
    and library versions and would keep laptop and cluster mirrors from matching.
    """
    if not growth_provenance:
        return None
    inner = growth_provenance.get("fingerprint") or {}
    return {
        "config": inner.get("config"),
        "datasets": {
            key: digest
            for key, digest in inner.get("files", {}).items()
            if not _is_volatile(key)
        },
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_settings(directory: Path) -> dict[str, Any]:
    """The result settings a checkpoint directory was built with (its summary)."""
    return dict(_read_json(directory / "summary.json")["settings"])


def current_checkpoint_settings(directory: Path) -> dict[str, Any]:
    """Project historical settings onto the current band-result settings.

    This is solely for compatibility checks; the stored settings and identity
    are not changed and unknown fields remain subject to comparison.
    """
    settings = checkpoint_settings(directory)
    settings.pop("core_threshold", None)
    settings.pop("consensus_peak_frequency", None)
    return settings


def scientific_of(directory: Path, record: str = SLOT_RECORD) -> dict[str, Any]:
    """The scientific fingerprint of a checkpoint directory, from its stage ``record``."""
    return scientific_fingerprint(_read_json(directory / record)["fingerprint"])


def differences(found: Any, wanted: Mapping[str, Any]) -> list[str]:
    """Names of the parts of ``found`` that disagree with ``wanted``.

    ``wanted`` may be partial: a missing ``datasets`` (or a ``growth`` without
    ``config`` or ``datasets``) is not compared. ``growth`` is dropped from both
    sides when either lacks it, so mirrors written before it was added still match.
    """
    if not isinstance(found, dict):
        return ["unreadable fingerprint"]
    if "growth" not in found or "growth" not in wanted:
        found = {k: v for k, v in found.items() if k != "growth"}
        wanted = {k: v for k, v in wanted.items() if k != "growth"}
    names: list[str] = []
    if "datasets" in wanted:
        mine, theirs = found.get("datasets") or {}, wanted["datasets"] or {}
        names += [
            k for k in sorted(set(mine) | set(theirs)) if mine.get(k) != theirs.get(k)
        ]
    if isinstance(wanted.get("growth"), dict):
        have, want = found.get("growth") or {}, wanted["growth"]
        if "datasets" in want:
            mine, theirs = have.get("datasets") or {}, want["datasets"] or {}
            names += [
                f"growth:{k}"
                for k in sorted(set(mine) | set(theirs))
                if mine.get(k) != theirs.get(k)
            ]
        if "config" in want and have.get("config") != want["config"]:
            names.append("growth:config")
    for part in ("config", "candidate_functional_threshold", "candidate_selection"):
        if part in wanted and found.get(part) != wanted[part]:
            names.append(part)
    return names


def unchecked_parts(wanted: Mapping[str, Any] | None) -> list[str]:
    """What a (possibly partial) ``wanted`` fingerprint leaves unchecked."""
    if wanted is None:
        return ["everything but the result settings"]
    missing: list[str] = []
    if "datasets" not in wanted:
        missing.append("input dataset digests")
    if "candidate_selection" not in wanted:
        missing.append("Candidate vertex set (functional classification)")
    growth = wanted.get("growth")
    if not isinstance(growth, dict):
        missing.append("growth-rate settings and BarSeq digests")
    else:
        if "config" not in growth:
            missing.append("growth-rate settings")
        if "datasets" not in growth:
            missing.append("BarSeq digests")
    return missing


def _fsync_path(path: Path) -> None:
    """Flush a file or directory to disk (best effort where unsupported)."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            _fsync_path(path)
    _fsync_path(directory)


def restore_directory(source: Path, destination: Path) -> None:
    """Copy ``source`` to ``destination`` atomically (temp sibling, then rename).

    An existing ``destination`` is moved aside with a unique name, announced in a
    warning, rather than deleted, since it may hold a different, valuable run. A
    failed rename puts it back.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.restore-{uuid.uuid4().hex}")
    shutil.copytree(source, temp)
    aside: Path | None = None
    try:
        _fsync_tree(temp)
        if destination.exists():
            aside = destination.with_name(
                f".{destination.name}.replaced-{uuid.uuid4().hex}"
            )
            os.replace(destination, aside)
        try:
            os.replace(temp, destination)
        except BaseException:
            if aside is not None:
                os.replace(aside, destination)
                aside = None
            raise
        _fsync_path(destination.parent)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    if aside is not None:
        warnings.warn(
            f"Moved the existing {destination} aside to {aside} before restoring "
            f"the mirror from {source}; delete it by hand once it is not needed.",
            RuntimeWarning,
            stacklevel=2,
        )


@contextmanager
def _timed_lock(
    name: Path, *, lock_path: Path | None = None, what: str = "Bootstrap mirror"
) -> Iterator[bool]:
    """Blocking exclusive lock that gives up after ``LOCK_TIMEOUT`` seconds.

    Yields whether the lock is actually held.

    Where the filesystem cannot lock at all (ENOSYS, ENOLCK, EOPNOTSUPP, as on some
    network mounts) a warning is issued and the work proceeds unlocked. Run one
    bootstrap per repo checkout at a time: the lock is not reliable across nodes.
    ``lock_path`` overrides the default ``<parent>/.<name>.lock`` file and ``what``
    names the guarded thing in messages.
    """
    if os.name == "nt":
        with checkpoint_lock(name, what):
            yield True
        return
    import fcntl

    name.parent.mkdir(parents=True, exist_ok=True)
    lock_path = lock_path or name.parent / f".{name.name}.lock"
    while True:
        stream = lock_path.open("a+b")
        locked = False
        try:
            deadline = time.monotonic() + LOCK_TIMEOUT
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError as error:
                    if error.errno in {errno.ENOSYS, errno.ENOLCK, errno.EOPNOTSUPP}:
                        warnings.warn(
                            f"Cannot lock {lock_path} ({error}); working without a "
                            "lock. Do not run two bootstraps at once.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                        break
                    if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"{what} is locked by another process for more "
                            f"than {LOCK_TIMEOUT:g} s; retry after it finishes."
                        ) from error
                    time.sleep(0.1)
            if locked and not _is_current(lock_path, stream):
                # The directory was moved aside (--fresh) while we waited: this
                # lock guards a file nobody else looks at. Take the new one.
                continue
            yield locked
            return
        finally:
            if locked:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def _is_current(lock_path: Path, stream: Any) -> bool:
    """Whether ``lock_path`` still names the file ``stream`` holds."""
    try:
        now = os.stat(lock_path)
    except FileNotFoundError:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        return False
    held = os.fstat(stream.fileno())
    return (now.st_dev, now.st_ino) == (held.st_dev, held.st_ino)


def iter_processed_checkpoints(
    settings: Mapping[str, Any],
    scientific: Mapping[str, Any] | None,
    processed_root: Path,
) -> Iterator[Path]:
    """Checkpoints of ``<processed_root>/bootstrap/<state>/checkpoint`` that match.

    A candidate matches when its result settings equal ``settings`` and its
    scientific fingerprint has no difference from ``scientific``. ``scientific`` may be
    partial (see ``differences``) or ``None`` (match on the settings alone); with a
    partial fingerprint more than one match is ambiguous and raises
    ``AmbiguousCheckpointError``, exactly one is used with a warning. Candidates that
    match the settings but not the fingerprint are named in a warning once the
    iterator is exhausted.
    """
    mismatched: list[tuple[Path, list[str]]] = []
    matches: list[Path] = []
    for checkpoint in sorted((Path(processed_root) / "bootstrap").glob("*/checkpoint")):
        try:
            if not (
                (checkpoint / SLOT_RECORD).is_file()
                and (checkpoint / "summary.json").is_file()
                and current_checkpoint_settings(checkpoint) == dict(settings)
            ):
                continue
            if scientific is None:
                matches.append(checkpoint)
                continue
            names = differences(scientific_of(checkpoint), scientific)
            if names:
                mismatched.append((checkpoint, names))
            else:
                matches.append(checkpoint)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    unchecked = unchecked_parts(scientific)
    if unchecked and len(matches) > 1:
        raise AmbiguousCheckpointError(
            "Several processed bootstrap checkpoints match the settings and the checks "
            f"that could be made, so none can be chosen safely: {', '.join(map(str, matches))}. "
            f"Not checked: {', '.join(unchecked)}."
        )
    if unchecked and matches:
        warnings.warn(
            f"Restoring {matches[0]} although {', '.join(unchecked)} could not be "
            "checked; it is the only processed checkpoint matching the settings. "
            "Verify it is the run you expect.",
            RuntimeWarning,
            stacklevel=2,
        )
    yield from matches
    for checkpoint, names in mismatched:
        warnings.warn(
            f"Processed checkpoint {checkpoint} matches the settings but not the "
            f"scientific fingerprint; differing: {', '.join(names) or 'unknown'}.",
            RuntimeWarning,
            stacklevel=2,
        )
