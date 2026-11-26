"""The two graphs of the band-quantile scan figure.

Columns are the Published graph and the Candidate graph, each under BY at nominal
q = 0.05. Only these two selections have a bootstrap draw set, so only they have a
scan: the point-estimate graphs (no FDR) have none. Each panel is the scan of one
processed state (``data/processed/band_quantile_scan/<state>/``); this module only
names the panels and reads scan tables that were already written.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SCAN_DIR_NAME = "band_quantile_scan"


@dataclass(frozen=True)
class ScanPanel:
    """One cell of the figure."""

    threshold: str
    fdr: str
    q: float | None
    variant: str
    state: str
    row_title: str
    column_title: str

    def as_dict(self) -> dict[str, Any]:
        """Identity recorded on a scan, with no threshold number."""
        record = {"threshold": self.threshold, "fdr": self.fdr, "variant": self.variant}
        if self.q is not None:
            record["q"] = self.q
        return record


# Left to right: Published BY, then Candidate BY.
PANELS: tuple[ScanPanel, ...] = (
    ScanPanel(
        "published",
        "by",
        0.05,
        "threshold=published_fdr=by_q=0.05",
        "published_by",
        "Published graph",
        "BY q = 0.05",
    ),
    ScanPanel(
        "candidate",
        "by",
        0.05,
        "threshold=reflected_fdr=by_q=0.05",
        "reflected_by",
        "Candidate graph",
        "BY q = 0.05",
    ),
)
BY_VARIANT = {panel.variant: panel for panel in PANELS}
BY_STATE = {panel.state: panel for panel in PANELS}


def variant_declaration(variant: str) -> dict[str, Any]:
    """The candidate-threshold declaration of ``config/variants/<variant>.toml``."""
    from ..analysis import LEGACY_VARIANT_NAMES

    variant = LEGACY_VARIANT_NAMES.get(variant, variant)
    path = ROOT / "config" / "variants" / f"{variant}.toml"
    with path.open("rb") as stream:
        table = tomllib.load(stream)
    return dict(table["clamped"]["candidate_functional_threshold"])


def panel_for_variant(
    variant: str | None, settings: dict[str, Any] | None = None
) -> ScanPanel | None:
    """Return the panel a scan job is producing.

    A named variant is one of the four cells. A job with no variant is the
    Candidate BY panel only when its declaration equals that variant file.
    """
    if variant:
        from ..analysis import LEGACY_VARIANT_NAMES

        return BY_VARIANT.get(LEGACY_VARIANT_NAMES.get(variant, variant))
    if settings is None:
        return None
    declared = settings.get("analysis", {}).get("clamped", {}).get(
        "candidate_functional_threshold"
    )
    candidate_by = BY_VARIANT["threshold=reflected_fdr=by_q=0.05"]
    if declared == variant_declaration(candidate_by.variant):
        return candidate_by
    return None


def _tables(directory: Path) -> bool:
    return (directory / "scan.parquet").is_file() and (
        directory / "survival.parquet"
    ).is_file()


def _read_panel(directory: Path, panel: ScanPanel) -> dict[str, Any]:
    record_path = directory / "provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    checkpoint = record.get("config", {}).get("checkpoint", {})
    return {
        "panel": panel,
        "scan": pl.read_parquet(directory / "scan.parquet"),
        "survival": pl.read_parquet(directory / "survival.parquet"),
        "summary_settings": {
            "band_quantile": float(checkpoint.get("band_quantile", 0.95)),
            "n_replicates": int(checkpoint.get("n_replicates", 0)),
            "seed": int(checkpoint.get("seed", 0)),
        },
    }


def iter_scan_dirs(processed_root: Path) -> list[Path]:
    """Directories ``<processed_root>/band_quantile_scan/<state>/`` that hold a scan and a survival table."""
    return [
        directory
        for directory in sorted((processed_root / SCAN_DIR_NAME).glob("*/"))
        if directory.is_dir() and _tables(directory)
    ]


def load_grid(processed_root: Path) -> list[dict[str, Any]]:
    """Load the panels in figure order.

    The directory name is the panel's identity (``published_by``, ``reflected_by``).
    Raises ``FileNotFoundError`` naming every missing panel. Nothing is recomputed.
    """
    located = {
        directory.name: _read_panel(directory, BY_STATE[directory.name])
        for directory in iter_scan_dirs(processed_root)
        if directory.name in BY_STATE
    }
    missing = [panel for panel in PANELS if panel.state not in located]
    if missing:
        names = ", ".join(panel.state for panel in missing)
        raise FileNotFoundError(
            f"Band-quantile scan panels are missing ({names}) under "
            f"{processed_root / SCAN_DIR_NAME}: run `just bootstrap-band-quantile-scan` "
            "for each BY variant, `just all` or `just fetch-data`."
        )
    return [located[panel.state] for panel in PANELS]
