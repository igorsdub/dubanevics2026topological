"""The band radius of a saved band tree (a second input beyond the tree products).

The persistence strip of ``bootstrap_band`` needs 2c, which no saved tree product
carries: it is in the bootstrap checkpoint's ``summary.json``
(``band_simplification.two_c``). The band tree's manifest records that file's sha256
under ``source_lineage.source_artifacts``. This loader finds the file by that hash
in the given roots and fails if it is absent or differs; there is no fallback. A
saved-product field for 2c would replace it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SUMMARY_KEY = "checkpoint/summary.json"


def summary_sha256(manifest: Any) -> str:
    """The checkpoint summary hash a band tree's manifest records."""
    for parent in manifest.parents:
        artifacts = parent.get("source_lineage", {}).get("source_artifacts", {})
        if SUMMARY_KEY in artifacts:
            return str(artifacts[SUMMARY_KEY]["sha256"])
    raise ValueError("Band tree manifest records no checkpoint summary hash.")


def load_two_c(manifest: Any, roots: Sequence[Path]) -> float:
    """2c of the band whose checkpoint summary hashes to the manifest's record."""
    wanted = summary_sha256(manifest)
    for root in roots:
        for path in sorted(Path(root).rglob("summary.json")):
            if hashlib.sha256(path.read_bytes()).hexdigest() == wanted:
                summary = json.loads(path.read_text(encoding="utf-8"))
                return float(summary["band_simplification"]["two_c"])
    raise ValueError(
        f"No checkpoint summary.json with sha256 {wanted[:12]} under "
        f"{[str(root) for root in roots]}."
    )
