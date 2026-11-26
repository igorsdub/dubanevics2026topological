"""The project's real stage declarations, built without any ``output/`` state.

Three stages carry savename aliases (``stages.SAVENAME_KEYS``). Tests that need to
see the settings those stages really declare build them here, through their own
factories, so an alias is checked against the settings the factory produces and
not against a hand-copied dictionary that could drift from it.

The analysis stage reads the raw data files of the checkout. The bootstrap stage
takes the run's scientific identity, which is built from the checked-in config in
the way ``bootstrap_smoothed_tree._wanted_scientific`` builds it, but without
``candidate_selection`` and without dataset digests: naming does not need them, so
they are left out of the identity. The codon-entropy stage reads two upstream
records from ``output/``; their digests are stubbed because only the declared
settings matter here.
"""

from __future__ import annotations

from typing import Any

import pytest

from dubanevics2026topological import build
from dubanevics2026topological.analysis import (
    analysis_stage,
    load_settings,
    scientific_inputs,
)
from dubanevics2026topological.analysis.produce_or_load import Stage
from dubanevics2026topological.simplification import checkpoint_identity
from dubanevics2026topological.simplification import (
    bootstrap_smoothed_tree as bootstrap,
)

# The raw files the analysis stage declares as inputs; without them it cannot be built.
needs_raw_data = pytest.mark.skipif(
    not all(path.is_file() for path in scientific_inputs(load_settings())),
    reason="needs data/raw",
)


def config() -> dict[str, Any]:
    """The checked-in scientific configuration, parsed."""
    return load_settings()["analysis"]


def build_bootstrap_stage() -> Stage:
    """The bootstrap stage; it reads no raw data."""
    scientific = load_settings()["analysis"]
    identity = checkpoint_identity.scientific_fingerprint(
        {
            "config": bootstrap._config_identity(scientific[bootstrap.SECTION]),
            "candidate_functional_threshold": scientific["clamped"][
                "candidate_functional_threshold"
            ],
            "files": {},
        }
    )
    return bootstrap.bootstrap_stage(identity)


def build_registry_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, Stage]:
    """Return ``{stage name: Stage}`` for every stage that has savename aliases."""
    settings = load_settings()
    monkeypatch.setattr(build, "_comparison_analysis_fingerprint", lambda: "comparison")
    monkeypatch.setattr(
        build, "_peak_analysis_fingerprint", lambda context=None: "peak"
    )
    return {
        "analysis": analysis_stage(settings),
        bootstrap.STAGE: build_bootstrap_stage(),
        "codon-entropy": build._codon_entropy_stage(settings),
    }
