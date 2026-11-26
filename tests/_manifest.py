"""A byte manifest of a directory tree, shared by the Phase 3 characterisation tests.

``tree_manifest(root)`` maps every regular file under ``root`` to the sha256 of its
bytes, keyed by its path relative to ``root`` with ``/`` separators. It follows
symbolic links on purpose: the question the tests ask is "which bytes does a reader
see at this path", and a slot that is a link into a versions directory must answer
exactly as the real directory it replaced did (ADR 0013, Phase 3).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def file_digest(path: Path) -> str:
    """The sha256 of the bytes of ``path`` (read through a link)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    """Return ``{relative path: sha256}`` for every regular file under ``root``.

    ``root`` itself may be a link to a directory; links to directories and to files
    below it are followed. A missing ``root`` gives an empty manifest, so a test can
    compare "nothing" with "nothing".
    """
    manifest: dict[str, str] = {}
    base = Path(root)
    if not base.is_dir():
        return manifest
    for directory, _names, files in os.walk(base, followlinks=True):
        for name in files:
            path = Path(directory) / name
            if path.is_file():  # a dangling link is not a file
                manifest[path.relative_to(base).as_posix()] = file_digest(path)
    return dict(sorted(manifest.items()))
