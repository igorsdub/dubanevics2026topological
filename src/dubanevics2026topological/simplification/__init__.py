"""Fixed-graph bootstrap draws and tree simplification.

The implementation is being migrated from :mod:`analysis` while its historic
bootstrap behavior remains available through compatibility modules.
"""

from .bootstrap_smoothed_tree import (
    BootstrapSmoothedTreeResult,
    build_band_simplified_view,
    build_bootstrap_smoothed_tree,
    load_cached_bootstrap_smoothed_tree,
)
from .identities import band_identity, draw_identity, evidence_availability

__all__ = [
    "BootstrapSmoothedTreeResult",
    "band_identity",
    "build_band_simplified_view",
    "build_bootstrap_smoothed_tree",
    "draw_identity",
    "evidence_availability",
    "load_cached_bootstrap_smoothed_tree",
]
