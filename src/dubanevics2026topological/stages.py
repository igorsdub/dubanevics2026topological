"""The project's one place that declares build stages (ADR 0013).

``analysis/produce_or_load.py`` is generic: its ``Stage`` has no defaults for the
scheduling keys that stay out of a fingerprint or for the libraries whose
versions a checkpoint records. This module holds this project's values and
``project_stage``, the only factory that constructs a ``Stage``. Every stage in
the package is built through it, so no call site can forget a value;
``tests/test_stages.py`` fails on any other ``Stage(...)`` call outside the
helper.

Keep this module out of every stage fingerprint (declared ``code`` files, and the
``analysis``, ``figures`` and ``tables`` trees that some provenance records
hash). It lives beside, not inside, those trees for that reason, and it imports
nothing but the helper.

Because no source hash covers this file, the watch loop (``build._watch``) does
not count an edit here as a science change and runs only the figures target. A
changed ``EXECUTION_KEYS`` still changes what the table stages fingerprint, and
takes effect at the next ``just tables``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .analysis.produce_or_load import Stage

# Scheduling keys that never enter a fingerprint: they change how a stage runs,
# not what it computes.
EXECUTION_KEYS = ("n_jobs", "chunk_size")
# Distributions whose installed versions a checkpoint's provenance records.
STAGE_LIBRARIES = ("polars", "numpy", "pyarrow", "topographer")
# Folder-name aliases. The mapping is display data: it is not part of the key.
# Paths are dotted lookups into the stage's declared settings.
SAVENAME_KEYS = {
    "analysis": {
        "landscape": "config.dataset.landscape",
        "tree": "config.tree.kind",
    },
    "bootstrap-smoothed-tree": {
        "draws": "scientific.config.n_replicates",
        "seed": "scientific.config.seed",
    },
    "codon-entropy": {
        "qthreshold": "candidate_peak_biology.q_value_threshold",
        "sweep": "codon_entropy.sweep_points",
    },
}


def project_stage(
    name: str,
    *,
    settings: Mapping[str, Any],
    root: Path,
    inputs: Sequence[Path] = (),
    code: Sequence[Path] = (),
) -> Stage:
    """Declare a stage with this project's execution keys and libraries.

    ``root`` is the repository root that declared paths are keyed against; the
    caller passes it so a test can point it at a temporary tree.
    """
    return Stage(
        name,
        settings=settings,
        inputs=inputs,
        code=code,
        root=root,
        execution_keys=EXECUTION_KEYS,
        libraries=STAGE_LIBRARIES,
        name_keys=dict(SAVENAME_KEYS.get(name, {})),
    )
