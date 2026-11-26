"""Filesystem locking, recovery and versioning for multi-file checkpoints.

Part of the produce-or-load extraction boundary (see ``produce_or_load.py``):
this module imports only the standard library and carries no project names, so
it can be lifted together with the helper. Callers supply anything
project-specific, such as the ``retry_hint`` of ``recover_checkpoint`` and
``publish_checkpoint``.

Versions. A checkpoint slot ``<slot>`` can be a relative link to one immutable
version directory ``<slot>.versions/<name>``. A version name is
``[savename-]key-digest``: an optional readable label, the first 8 hex characters
of the fingerprint (``fingerprint_tag``) and the first 8 of the digest of the
version's files (``tree_digest``). A name is a label and never the identity of
what it holds, so a reader confirms the full fingerprint from the record. When
two different versions would share a short name, both tags are written in full
(64 hex characters). The primitives here only name, digest, point at and list
versions: ``version_name`` and ``parse_version_name``, ``fingerprint_tag``,
``tree_digest``, ``switch_link`` (the one atomic switch of the link),
``current_version`` (the version a link names, read lexically) and
``list_versions``. None of them removes anything.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import time
import uuid
import warnings
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

VERSIONS_SUFFIX = ".versions"
# Operating-system metadata that a file manager may drop into a checkpoint
# directory. It is neither produced nor verified, so it never invalidates one.
IGNORED_NAMES = frozenset({".DS_Store"})

_HEX64 = re.compile(r"[0-9a-f]{64}")
# A tag is 8 hex characters, or the full 64 once the short form is taken.
_TAG = re.compile(r"[0-9a-f]{8}(?:[0-9a-f]{56})?")
# A savename starts with a letter or digit, so a version directory is never hidden.
_SAVENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+%=-]*")
_MAX_NAME_LENGTH = 255


def _lock_file(lock_path: Path, label: str):
    """Return a held, nonblocking OS lock; the lock file itself is persistent."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN} or getattr(
                    error, "winerror", None
                ) in {33, 36}:
                    raise RuntimeError(
                        f"{label} checkpoint is already in use; retry after the "
                        "other command finishes."
                    ) from error
                raise
            return stream

        import fcntl

        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RuntimeError(
                    f"{label} checkpoint is already in use; retry after the "
                    "other command finishes."
                ) from error
            raise
        return stream
    except BaseException:
        stream.close()
        raise


@contextmanager
def checkpoint_lock(cache_dir: Path, label: str) -> Iterator[None]:
    """Serialize all access to one checkpoint and release automatically on exit."""
    lock_path = cache_dir.parent / f".{cache_dir.name}.lock"
    stream = _lock_file(lock_path, label)
    try:
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def backup_path(cache_dir: Path) -> Path:
    """Where ``publish_checkpoint`` keeps the previous checkpoint while it publishes."""
    return cache_dir.with_name(f".{cache_dir.name}.previous")


def _quarantine(path: Path, label: str) -> Path:
    destination = path.with_name(f".{path.name}.unusable-{uuid.uuid4().hex}")
    os.replace(path, destination)
    warnings.warn(
        f"Moved unusable {label} checkpoint state to {destination.name!r} for "
        "inspection.",
        RuntimeWarning,
        stacklevel=2,
    )
    return destination


def _validate[T](path: Path, validator: Callable[[Path], T]) -> T | None:
    if not path.is_dir():
        return None
    try:
        return validator(path)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return None


def _retry_command(retry_hint: str | None) -> str:
    """Name the command to retry: the caller's hint, else the producer generically."""
    return (
        f"`{retry_hint}`" if retry_hint else "the command that produced this checkpoint"
    )


def _remove_pending(
    pending_path: Path, label: str, retry_hint: str | None = None
) -> None:
    try:
        pending_path.unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"The {label} checkpoint is valid, but its pending marker could not "
            f"be cleared. Retry {_retry_command(retry_hint)} to finish recovery."
        ) from error


def recover_checkpoint[T](
    cache_dir: Path,
    pending_path: Path,
    label: str,
    validator: Callable[[Path], T],
    *,
    expected_fingerprint: object | None = None,
    fingerprint_of: Callable[[T], object | None] | None = None,
    retry_hint: str | None = None,
) -> T | None:
    """Reconcile interrupted publication while the caller holds ``checkpoint_lock``.

    Validity is structural and provenance-based. A matching requested fingerprint
    is preferred during recovery; otherwise pending remains set until a matching
    generation is published. ``retry_hint``, when given, is the command named in
    the error if the pending marker cannot be cleared; without it the note names
    the producing command generically.
    """
    backup = backup_path(cache_dir)
    current_value = _validate(cache_dir, validator)
    backup_value = _validate(backup, validator)
    pending = pending_path.exists()
    current_matches = current_value is not None and (
        expected_fingerprint is None
        or (
            fingerprint_of is not None
            and fingerprint_of(current_value) == expected_fingerprint
        )
    )
    backup_matches = backup_value is not None and (
        expected_fingerprint is None
        or (
            fingerprint_of is not None
            and fingerprint_of(backup_value) == expected_fingerprint
        )
    )

    if expected_fingerprint is not None and backup_matches and not current_matches:
        if cache_dir.exists():
            _quarantine(cache_dir, f"stale {label}")
        os.replace(backup, cache_dir)
        restored = _validate(cache_dir, validator)
        if (
            restored is None
            or fingerprint_of is None
            or (fingerprint_of(restored) != expected_fingerprint)
        ):
            raise RuntimeError(
                f"Restored {label} checkpoint failed expected-fingerprint "
                f"validation at {cache_dir}."
            )
        _remove_pending(pending_path, label, retry_hint)
        return restored

    if current_value is not None:
        # A complete current directory is the latest published generation. A
        # backup can now be retired safely, including after a crash post-commit.
        if not pending or current_matches:
            _remove_pending(pending_path, label, retry_hint)
        if backup.exists():
            if backup_value is None:
                _quarantine(backup, f"{label} backup")
            else:
                try:
                    shutil.rmtree(backup)
                except OSError:
                    warnings.warn(
                        f"Could not remove the older {label} checkpoint backup; "
                        "the validated current checkpoint remains usable.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return current_value

    if backup_value is not None:
        # Preserve an invalid current tree for diagnosis before restoring the
        # last complete checkpoint.
        if cache_dir.exists():
            _quarantine(cache_dir, label)
        os.replace(backup, cache_dir)
        restored = _validate(cache_dir, validator)
        if restored is None:
            raise RuntimeError(
                f"Restored {label} checkpoint failed validation at {cache_dir}."
            )
        if (
            not pending
            or expected_fingerprint is None
            or fingerprint_of is None
            or fingerprint_of(restored) == expected_fingerprint
        ):
            _remove_pending(pending_path, label, retry_hint)
        return restored

    # On a first build there is no previous checkpoint. Preserve damaged state,
    # then let the explicit writer build a fresh stage while pending stays set.
    if cache_dir.exists():
        _quarantine(cache_dir, label)
    if backup.exists():
        _quarantine(backup, f"{label} backup")
    if pending or not cache_dir.exists():
        return None
    return current_value


def publish_checkpoint[T](
    stage: Path,
    cache_dir: Path,
    pending_path: Path,
    label: str,
    validator: Callable[[Path], T],
    *,
    expected_fingerprint: object | None = None,
    fingerprint_of: Callable[[T], object | None] | None = None,
    retry_hint: str | None = None,
) -> T:
    """Validate and promote a staged directory, rolling back on failure.

    The caller must hold ``checkpoint_lock`` and have created the pending marker.
    The existing valid checkpoint is retained until the staged checkpoint has
    replaced and passed validation at the canonical path. ``retry_hint``, when
    given, is the command the caller wants named in the error if the rollback
    itself fails (after a failed rename or a failed validation) or the pending
    marker cannot be cleared; without it the note names the producing command
    generically.
    """
    staged_value = _validate(stage, validator)
    if staged_value is None or (
        expected_fingerprint is not None
        and fingerprint_of is not None
        and fingerprint_of(staged_value) != expected_fingerprint
    ):
        raise RuntimeError(f"Staged {label} checkpoint failed validation: {stage}.")

    backup = backup_path(cache_dir)
    if backup.exists():
        # recover_checkpoint has already validated current state. The previous
        # backup is older than that current checkpoint and can be retired.
        shutil.rmtree(backup)

    moved_current = False
    if cache_dir.exists():
        os.replace(cache_dir, backup)
        moved_current = True

    try:
        os.replace(stage, cache_dir)
    except BaseException as publication_error:
        if moved_current and backup.exists():
            try:
                os.replace(backup, cache_dir)
                restored = _validate(cache_dir, validator)
                if restored is None:
                    raise RuntimeError(
                        f"Restored {label} checkpoint failed validation at {cache_dir}."
                    )
                if (
                    expected_fingerprint is None
                    or fingerprint_of is None
                    or fingerprint_of(restored) == expected_fingerprint
                ):
                    _remove_pending(pending_path, label, retry_hint)
            except (OSError, RuntimeError) as rollback_error:
                publication_error.add_note(
                    f"Rollback also failed: {rollback_error!r}. The backup and "
                    f"pending marker were preserved where possible; retry "
                    f"{_retry_command(retry_hint)} for recovery."
                )
        raise

    published_value = _validate(cache_dir, validator)
    if published_value is None or (
        expected_fingerprint is not None
        and fingerprint_of is not None
        and fingerprint_of(published_value) != expected_fingerprint
    ):
        validation_error = RuntimeError(
            f"Published {label} checkpoint failed validation at {cache_dir}."
        )
        if moved_current and backup.exists():
            try:
                _quarantine(cache_dir, f"invalid {label}")
                os.replace(backup, cache_dir)
                restored = _validate(cache_dir, validator)
                if restored is None:
                    raise RuntimeError(
                        f"Restored {label} checkpoint failed validation at {cache_dir}."
                    )
                if (
                    expected_fingerprint is None
                    or fingerprint_of is None
                    or fingerprint_of(restored) == expected_fingerprint
                ):
                    _remove_pending(pending_path, label, retry_hint)
            except (OSError, RuntimeError) as rollback_error:
                validation_error.add_note(
                    f"Rollback also failed: {rollback_error!r}. The backup and "
                    f"pending marker were preserved where possible; retry "
                    f"{_retry_command(retry_hint)} for recovery."
                )
        raise validation_error

    # Clearing pending commits the validated new checkpoint. Backup cleanup is
    # post-commit and can be retried on the next explicit writer invocation.
    _remove_pending(pending_path, label, retry_hint)
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            warnings.warn(
                f"Could not remove the previous {label} checkpoint backup; the "
                "new validated checkpoint is published and remains usable.",
                RuntimeWarning,
                stacklevel=2,
            )
    return published_value


# --- versions ---------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions_dir(link: Path) -> Path:
    """The directory that holds the versions of the slot ``link``."""
    return link.with_name(f"{link.name}{VERSIONS_SUFFIX}")


def fingerprint_tag(fingerprint: object, length: int = 8) -> str:
    """Return the first ``length`` hex characters that stand for ``fingerprint``.

    A 64-character lowercase hex string stands for itself, so the tag of a
    sha256 fingerprint can be found by searching the record that holds it. Any
    other JSON value, such as a dict, is hashed through its canonical JSON
    (sorted keys, no spaces), so the tag does not depend on key order.
    ``length=64`` gives the full form. A missing fingerprint cannot name a
    version and raises ``ValueError``.
    """
    if not 1 <= length <= 64:
        raise ValueError(f"A tag length must be between 1 and 64, got {length!r}.")
    if fingerprint is None:
        raise ValueError("A version cannot be named from a missing fingerprint.")
    if isinstance(fingerprint, str) and _HEX64.fullmatch(fingerprint):
        return fingerprint[:length]
    text = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class VersionName:
    """A parsed version name: the optional savename and the two tags as written."""

    savename: str | None
    key: str
    digest: str


def version_name(
    fingerprint: object,
    digest: str,
    savename: str | None = None,
    *,
    full: bool = False,
) -> str:
    """Return ``[savename-]key-digest`` for a version.

    ``fingerprint`` is turned into the key tag by ``fingerprint_tag`` and
    ``digest`` is the 64-hex ``tree_digest`` of the version's files. ``full``
    writes both tags in 64 hex characters, the form a caller uses when the short
    name is already taken by another version. A savename (empty means none) may
    use letters, digits and ``._+%=-`` and must start with a letter or digit, so
    the directory is never hidden.
    """
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise ValueError(f"A version digest must be 64 lowercase hex, got {digest!r}.")
    length = 64 if full else 8
    name = f"{fingerprint_tag(fingerprint, length)}-{digest[:length]}"
    if savename:
        if not isinstance(savename, str) or not _SAVENAME.fullmatch(savename):
            raise ValueError(
                "A savename must start with a letter or digit and use only "
                f"letters, digits and '._+%=-', got {savename!r}."
            )
        name = f"{savename}-{name}"
    if len(name) > _MAX_NAME_LENGTH:
        raise ValueError(
            f"The version name is {len(name)} characters, over the "
            f"{_MAX_NAME_LENGTH} a directory name can hold; shorten the savename."
        )
    return name


def parse_version_name(name: str) -> VersionName:
    """Split a version name into savename, key and digest; raise ``ValueError``.

    The name is read from the right: the last two ``-`` fields are the tags and
    everything before them is the savename, which may itself hold ``-`` and
    ``_``. The tags must both be 8 or both be 64 lowercase hex characters.
    """
    fields = name.rsplit("-", 2)
    if len(fields) == 2:
        savename, (key, digest) = None, fields
    elif len(fields) == 3:
        savename, key, digest = fields
        if not _SAVENAME.fullmatch(savename):
            raise ValueError(f"Not a version name (bad savename): {name!r}.")
    else:
        raise ValueError(f"Not a version name (no tags): {name!r}.")
    for tag in (key, digest):
        if not _TAG.fullmatch(tag):
            raise ValueError(f"Not a version name (bad tag {tag!r}): {name!r}.")
    if len(key) != len(digest):
        raise ValueError(f"Not a version name (mixed tag lengths): {name!r}.")
    return VersionName(savename, key, digest)


def _quoted(relative: str) -> tuple[str, str]:
    """Return ``(marker, name)``, escaped as ``sha256sum`` does, so a name that
    holds a newline cannot pass for two lines."""
    if "\\" in relative or "\n" in relative:
        escaped = relative.replace("\\", "\\\\").replace("\n", "\\n")
        return "\\", escaped
    return "", relative


def tree_digest(directory: Path, *, exclude: Collection[str] = ()) -> str:
    """Return one sha256 over the files below ``directory``.

    The digest covers the sorted lines ``"<sha256>  <relpath>\\n"``, one per
    regular file, with ``/`` separators; it does not depend on the order files
    were written. A symbolic link is recorded by its target text and never
    followed (a linked directory is not entered), so the digest names the bytes
    this tree holds and not whatever a link happens to point at today. Names in
    ``IGNORED_NAMES`` are skipped. ``exclude`` lists relative file paths to leave out,
    for example a record kept beside the files it describes; it matches file
    paths only, so a directory entry does not exclude its contents. Empty directories
    and entries that are neither regular files nor links are not part of the
    digest.
    """
    if isinstance(exclude, str | bytes):
        raise TypeError(
            "exclude must be a collection of relative paths, not a bare string; "
            "wrap a single path in a tuple."
        )
    skipped = set(exclude)
    base = Path(directory)

    def fail(error: OSError) -> None:
        raise error

    lines: list[str] = []
    for current, subdirectories, names in os.walk(base, onerror=fail):
        here = Path(current)
        linked = [name for name in subdirectories if (here / name).is_symlink()]
        for name in (*linked, *names):
            if name in IGNORED_NAMES:
                continue
            path = here / name
            relative = path.relative_to(base).as_posix()
            if relative in skipped:
                continue
            if path.is_symlink():
                value = f"L {json.dumps(os.readlink(path), ensure_ascii=False)}"
            elif path.is_file():
                value = _sha256(path)
            else:
                continue
            marker, quoted = _quoted(relative)
            lines.append(f"{marker}{value}  {quoted}\n")
    text = "".join(sorted(lines))
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


_READLINK_ATTEMPTS = 20


def _readlink(link: Path) -> str:
    """Read the text of ``link``, retrying while another process replaces it.

    Some file systems (APFS) answer ``EINVAL`` to ``readlink`` and to a path
    that runs through the link for an instant while a link is replaced, although
    ``lstat`` already shows the new one. The caller has seen ``link`` is a link,
    so ``EINVAL`` here is that instant and a short retry gets the answer.
    """
    for _ in range(_READLINK_ATTEMPTS - 1):
        try:
            return os.readlink(link)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise
            time.sleep(0.001)
    return os.readlink(link)


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry to disk where the platform allows it."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
    finally:
        os.close(descriptor)


def _version_target(link: Path, target: str, what: str) -> tuple[str, str]:
    """Return the ``(versions directory, name)`` a target spells, or raise.

    The only accepted form is ``<slot>.versions/<visible name>``: relative, two
    parts, no ``..``, and a name in the version grammar.
    """
    path = Path(target)
    parts = path.parts
    expected = _versions_dir(link).name
    if path.is_absolute() or len(parts) != 2 or parts[0] != expected or ".." in parts:
        raise ValueError(
            f"The {what} {target!r} is not '{expected}/<version name>'; refusing it."
        )
    if parts[1].startswith("."):
        raise ValueError(f"The {what} {target!r} names a hidden entry; refusing it.")
    try:
        parse_version_name(parts[1])
    except ValueError as error:
        raise ValueError(f"The {what} {target!r} is not a version: {error}") from error
    return parts[0], parts[1]


def switch_link(link: Path, version: Path) -> None:
    """Point ``link`` at ``version`` with one atomic replace of the link.

    ``version`` must be an existing, real directory
    ``<slot>.versions/<version name>`` beside ``link``. The link is written
    relative (``<slot>.versions/<name>``), so the tree can be copied or moved as
    a unit. A temporary link is created beside it and moved over it with
    ``os.replace``; the link itself is never unlinked first, so a reader sees
    either the old target or the new one, never none. Nothing happens when the
    link already points there. A real directory (or file) at the link path is
    refused with ``RuntimeError``: it holds a checkpoint that must be adopted
    into the versions directory first, never replaced. An existing link whose
    target ``current_version`` would refuse (outside ``<slot>.versions/``) is
    refused with ``ValueError`` and left as it is. The caller must hold the
    checkpoint lock; nothing here locks.

    A reader should resolve the link once with ``current_version`` and read
    every file from the version path it returns: a version is never modified, so
    that path stays valid across a switch, while a path that runs through the
    link can fail with ``EINVAL`` for an instant on APFS (``current_version``
    retries that).
    """
    link, version = Path(link), Path(version)
    relative = os.path.relpath(version, link.parent)
    _version_target(link, relative, "version")
    if version.is_symlink() or not version.is_dir():
        raise FileNotFoundError(
            errno.ENOENT, "The version is not an existing directory", str(version)
        )
    if link.is_symlink():
        existing = _readlink(link)
        if existing == relative:
            return
        _version_target(link, existing, "link target")  # refuse a foreign link
    elif os.path.lexists(link):
        raise RuntimeError(
            f"{link} is not a link (a real directory or file sits there); adopt it "
            f"into {_versions_dir(link)} before switching the link."
        )
    temporary = link.with_name(f".{link.name}.link-{uuid.uuid4().hex}")
    os.symlink(relative, temporary)
    try:
        os.replace(temporary, link)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(link.parent)


def current_version(link: Path) -> Path | None:
    """Return the version directory ``link`` names, read lexically, or ``None``.

    The path is ``link.parent / <slot>.versions / <name>``, built from the text
    of the link and never resolved, so a parent directory that is itself a link
    keeps the spelling the caller used. ``None`` means there is no current
    version: the link is missing, or dangling because its target is not a
    directory. A target that is not exactly ``<slot>.versions/<visible name>``
    (absolute, containing ``..``, hidden, outside the grammar, in another
    directory or nested deeper) raises ``ValueError``, as does a path that is not
    a link, or a version that is itself a link.
    """
    link = Path(link)
    if not os.path.lexists(link):
        return None
    if not link.is_symlink():
        raise ValueError(
            f"{link} is not a link; a real directory there is a checkpoint that "
            "has not been adopted into the versions directory."
        )
    directory, name = _version_target(link, _readlink(link), "link target")
    version = link.parent / directory / name
    if version.is_symlink():
        raise ValueError(f"The version {version} is itself a link; refusing it.")
    if not version.is_dir():
        return None
    return version


def list_versions(link: Path) -> list[Path]:
    """List the real, visible version directories of the slot ``link``, by name.

    Hidden entries (quarantined or set-aside versions, temporary names), files,
    links and names outside the version grammar are skipped. A missing versions
    directory gives an empty list.
    """
    directory = _versions_dir(Path(link))
    try:
        with os.scandir(directory) as entries:
            found = sorted(entries, key=lambda entry: entry.name)
    except (FileNotFoundError, NotADirectoryError):
        return []
    versions: list[Path] = []
    for entry in found:
        if entry.name.startswith(".") or entry.is_symlink():
            continue
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            parse_version_name(entry.name)
        except ValueError:
            continue
        versions.append(directory / entry.name)
    return versions
