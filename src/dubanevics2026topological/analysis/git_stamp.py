"""Git stamp for provenance records: a pointer to the repository state.

A Python counterpart of DrWatson.jl's ``tag``/``tagsave``. The stamp says which
commit a result was produced from and whether tracked files had uncommitted
changes. It is a pointer for people only; it is never part of an identity. A
host project keeps it out of every fingerprint, reuse decision and digest of a
whole record, so a dirty tree or a docs-only commit never invalidates a result.

Public surface, and nothing more:

- ``GIT_KEY``: the key a stamped record carries the stamp under.
- ``git_stamp(root)``: ``{"commit": sha | None, "dirty": bool | None}``.
- ``tag(record, root)``: a copy of ``record`` with that stamp under ``GIT_KEY``.
- ``untag(record)``: a copy of ``record`` without the stamp, the inverse of ``tag``.

Extraction boundary: like the sibling ``produce_or_load`` module this one imports
only the standard library and knows nothing of a host project. It never raises:
a missing ``git`` executable, a directory outside any repository, a timeout or
an unreadable repository all give ``None`` values. So does a repository that git
refuses as having dubious ownership when the only ``safe.directory`` setting the
caller had was passed through ``GIT_CONFIG_*`` variables, which the stamp drops
(see ``_CONFIGURATION_VARIABLES``): a stamp that is absent, never an error.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GIT_KEY = "git"
_TIMEOUT_SECONDS = 30
_OBJECT_NAME = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# The stamp must describe ``root`` and nothing the caller's environment says about
# git, so two groups of variables are dropped before every call. Variables that
# only limit discovery (``GIT_CEILING_DIRECTORIES``) are kept: they can make a
# stamp absent but never point it at another repository or change what it says.
#
# First, variables that tell git which repository, work tree or index to use.
# They are set by git itself while it runs a hook, or by a caller that drives a
# different repository, and must never redirect the stamp of ``root``.
_REPOSITORY_VARIABLES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    }
)


# Variables that inject configuration, or swap the configuration files git reads.
# A caller's ``-c`` options reach a child git through ``GIT_CONFIG_PARAMETERS`` and
# ``GIT_CONFIG_COUNT`` with its ``GIT_CONFIG_KEY_<n>``/``GIT_CONFIG_VALUE_<n>``
# pairs; ``GIT_CONFIG_GLOBAL`` and ``GIT_CONFIG_SYSTEM`` replace the files. Any of
# them can make git fail or answer differently, so the stamp reads git's own
# configuration only. That also drops a ``safe.directory`` entry passed this way:
# if the stamp then reports ``commit`` as ``None`` for a repository git treats as
# having dubious ownership, that is the expected degraded mode, not a bug.
_CONFIGURATION_VARIABLES = frozenset(
    {
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
    }
)
_CONFIGURATION_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")


def _environment() -> dict[str, str]:
    """The caller's environment minus what would redirect or reconfigure git."""
    return {
        name: value
        for name, value in os.environ.items()
        if name not in _REPOSITORY_VARIABLES
        and name not in _CONFIGURATION_VARIABLES
        and not name.startswith(_CONFIGURATION_PREFIXES)
    }


def _run(root: Path, *arguments: str) -> str | None:
    """Standard output of ``git -C root <arguments>``, or ``None`` on any failure."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_TIMEOUT_SECONDS,
            env=_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def git_stamp(root: Path) -> dict[str, str | bool | None]:
    """Return ``{"commit": sha | None, "dirty": bool | None}`` for ``root``.

    ``dirty`` is true when tracked files have staged or unstaged changes;
    untracked files are ignored, so large local inputs and outputs do not count.
    Both values are ``None``, and nothing raises, when git is absent, ``root``
    is not inside a repository (for example a ``git archive`` export) or the
    repository has no commit yet. ``dirty`` alone is ``None`` when the status
    query fails or times out.
    """
    commit = _run(root, "rev-parse", "--verify", "HEAD")
    if commit is None or _OBJECT_NAME.fullmatch(commit.strip()) is None:
        return {"commit": None, "dirty": None}
    status = _run(
        root, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"
    )
    return {
        "commit": commit.strip(),
        "dirty": None if status is None else bool(status.strip()),
    }


def tag(record: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Return a copy of ``record`` with the git stamp of ``root`` under ``GIT_KEY``.

    ``record`` itself is left unchanged. An existing ``GIT_KEY`` entry is
    replaced by the fresh stamp.
    """
    return {**record, GIT_KEY: git_stamp(root)}


def untag(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``record`` without its git stamp: the inverse of ``tag``.

    ``record`` itself is left unchanged, and a record that carries no stamp comes
    back equal to itself. Code that digests or compares whole records passes them
    through this first, so the stamp (a pointer for people) never enters an
    identity.
    """
    return {key: value for key, value in record.items() if key != GIT_KEY}
