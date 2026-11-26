"""Public entry points for the merge-tree fitness-landscape manuscript figures."""

from __future__ import annotations

from typing import Any

from .appearance import preview_settings, save_appearance_settings
from .registry import (
    FIGURE_IDS,
    FIGURE_SPECS,
    FigureSpec,
    compose_payload,
    get_spec,
    render_payload,
)


def __getattr__(name: str) -> Any:
    if name in {
        "apply_figure_footer",
        "extract_bootstrap_draws",
        "variant_figure_footer",
    }:
        from ..build import (
            apply_figure_footer,
            extract_bootstrap_draws,
            variant_figure_footer,
        )

        return {
            "apply_figure_footer": apply_figure_footer,
            "extract_bootstrap_draws": extract_bootstrap_draws,
            "variant_figure_footer": variant_figure_footer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FIGURE_IDS",
    "FIGURE_SPECS",
    "FigureSpec",
    "apply_figure_footer",
    "compose_payload",
    "extract_bootstrap_draws",
    "get_spec",
    "preview_settings",
    "render_payload",
    "save_appearance_settings",
    "variant_figure_footer",
]
