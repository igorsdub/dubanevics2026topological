"""This project's names for reading an upstream provenance record (ADR 0013, D3).

``analysis/produce_or_load.py`` stamps every provenance record it writes with the
repository commit and a dirty flag (``analysis/git_stamp.py``). The stamp is a
pointer for people, so no identity may depend on it: a dirty tree or a docs-only
commit must never invalidate a result. Code that digests, compares or keys on a
whole provenance record passes it through ``without_git`` first. Code that keys a
downstream stage on an upstream checkpoint takes ``upstream_identity``, which
keeps only the fingerprint and the artefact hashes (so it never sees the stamp,
a timestamp or a library version).

``without_git`` is ``git_stamp.untag`` under the name this project's call sites
use; the logic lives with ``tag``, its inverse. The name is kept here, and not
imported from ``analysis`` at each call site, because some of those call sites
are source files that fingerprints hash: a rename there would change them.
``upstream_identity`` lives here for the same reason and so that the table stages
in ``build`` and the stages under ``analysis`` share one rule for what identifies
an upstream checkpoint.

Like ``stages.py`` this module sits beside, not inside, the ``analysis``,
``figures`` and ``tables`` trees whose sources some fingerprints hash, so
editing it does not count as a science change. ``tests/test_stages.py`` fails if
any module names its path or hashes the package that holds it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .analysis.git_stamp import untag as without_git

__all__ = ["upstream_identity", "without_git"]


def upstream_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Keep an upstream checkpoint's fingerprint and artefact hashes only.

    Volatile fields such as ``created_at``, library versions or the commit stamp
    say nothing about the upstream bytes, so they must not key a downstream
    stage.

    Raises ``KeyError`` when ``provenance`` lacks ``fingerprint`` or
    ``artifacts``: dropping either silently would let a downstream stage reuse
    its results after the upstream bytes changed.
    """
    identity = {}
    for key in ("fingerprint", "artifacts"):
        if key not in provenance:
            raise KeyError(
                f"Upstream checkpoint provenance has no {key!r} "
                f"(found keys: {sorted(provenance)}); rebuild the upstream stage."
            )
        identity[key] = provenance[key]
    return identity
