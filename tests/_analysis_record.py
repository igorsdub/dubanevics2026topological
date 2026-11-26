"""Test helper: a caller record plus the published analysis checkpoint it names.

``build._analysis_fingerprint`` and ``analysis.load_cached_analysis`` refuse a
record that disagrees with the checkpoint's provenance, so a test that stands in
only a record must write both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_analysis_record(
    output: Path,
    fingerprint: str,
    *,
    checkpoint: bool = True,
    checkpoint_extra: dict[str, Any] | None = None,
    **extra: Any,
) -> Path:
    """Write ``output/analysis/provenance.json`` and the matching checkpoint record.

    ``extra`` keys are added to the caller's record only, ``checkpoint_extra`` keys
    to the published checkpoint's record only (a commit stamp, say). With
    ``checkpoint=False`` the published checkpoint is left out, which is the legacy
    layout. Returns the caller record's path.
    """
    artefacts = {"data/metadata.toml": "0" * 64}
    record = output / "analysis" / "provenance.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps({"fingerprint": fingerprint, "artefacts": artefacts, **extra}),
        encoding="utf-8",
    )
    if checkpoint:
        # The published checkpoint lives in the parameter-keyed leaf. Readers
        # resolve that leaf from the stage; a record sitting on the step
        # directory itself is not a checkpoint.
        import pytest

        from dubanevics2026topological.analysis import (
            ROOT,
            analysis_stage,
            scientific_inputs,
        )
        from dubanevics2026topological.analysis.produce_or_load import (
            result_directory,
        )

        missing = [p for p in scientific_inputs(root=ROOT) if not p.is_file()]
        if missing:
            pytest.skip(
                "needs the gitignored raw fitness tables under data/raw/ "
                f"(absent in a clean public clone): {missing[0].name}"
            )
        step = output / "checkpoints" / "analysis"
        published = result_directory(step, analysis_stage()) / "provenance.json"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "artefacts": artefacts,
                    **(checkpoint_extra or {}),
                }
            ),
            encoding="utf-8",
        )
    return record
