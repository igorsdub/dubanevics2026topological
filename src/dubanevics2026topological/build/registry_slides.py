"""Frameless slides of the registry tree figures: ``figures/slides/{tree_only,full}/``.

The registry flavours (Full, Published, Candidate) draw their trees from the saved layouts
of the figures stage, so they render inside ``build figures --slides``, after its
prerequisites, and not from the saved D10 products (``build/slide_trees.py``).

- ``tree_only``: each population's tree panel alone, in the D10 slide geometry
  (``figures/slide_tree.py``), so it lines up with the D10 tree-only slides;
- ``full``: the registry figure itself (``papkou2023_{full,article,candidate}_tree_histogram``,
  ``tree_comparison``, ``full_tree_persistence``) without
  titles, in vanilla and annotated style and count and none coloring. The
  full-graph persistence figure has one style (``variant``) per run.

Uncoloured files are ``<stem>_color=none_style=<style>.png``; coloured ones
``<stem>_style=<style>.png``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..figures.full_tree_persistence import (
    build_full_tree_persistence_figure,
)
from ..figures.slide_tree import (
    build_tree_only_figure,
    slide_figure_name,
    slide_stop_overrides,
    slide_tree_analysis,
)
from ..figures.tree_comparison import build_tree_comparison_figure
from ..figures.tree_histogram import build_tree_histogram_figure
from ..figures.tree_persistence_view import COLORINGS, STYLES

HISTOGRAM_IDS = {
    "full_tree_histogram": "full",
    "published_tree_histogram": "article",
    "candidate_tree_histogram": "candidate",
}
COMPARISON_ID = "tree_comparison"
FULL_GRAPH_ID = "full_tree_persistence"
SLIDE_FIGURE_IDS = (*HISTOGRAM_IDS, COMPARISON_ID, FULL_GRAPH_ID)


def slide_target(root: Path, group: str, stem: str, style: str, coloring: str) -> Path:
    """``<root>/slides/<group>/<stem>[_color=none]_style=<style>.png``."""
    return root / "slides" / group / slide_figure_name(f"{stem}_style={style}.png", coloring)


def tree_only_chart(
    payload: dict[str, Any],
    settings: dict[str, Any],
    population_id: str,
    style: str,
    coloring: str,
) -> Any:
    """One population's tree panel in the D10 slide geometry."""
    if getattr(payload, "legacy_mixture", False):
        landscape = {"full": "full", "article": "study_clamped", "candidate": "candidate_clamped"}
        tree_analysis = payload["landscapes"][landscape[population_id]].trees["split"]
    else:
        tree_analysis = payload["saved_layouts"][population_id]
    domain = tuple(float(x) for x in settings["figures"]["fitness_domain"])
    prepared, _ = slide_tree_analysis(tree_analysis, style)
    inputs = SimpleNamespace(
        tree_analysis=prepared,
        scalar_domain=domain,
        analysis=SimpleNamespace(scalar_domain=domain),
    )
    return build_tree_only_figure(
        inputs,
        settings,
        style,
        coloring=coloring,
        scalar_domain=domain,
        rendering_overrides=slide_stop_overrides(settings, population_id == "full"),
    )


def build_registry_slides(
    inputs: dict[str, Any],
    settings: dict[str, Any],
    out_dir: Path,
    composers: Any,
    *,
    full_graph: bool = True,
) -> list[Path]:
    """Render every registry slide into ``out_dir`` and check the group pixel sizes."""
    from . import _render_one

    dpi = int(settings["figures"].get("png_dpi", 300))
    written: list[Path] = []

    def emit(chart: Any, group: str, stem: str, style: str, coloring: str) -> None:
        target = slide_target(out_dir, group, stem, style, coloring)
        target.parent.mkdir(parents=True, exist_ok=True)
        _render_one(chart, target, "png", png_dpi=dpi)
        written.append(target)
        print(f"Rendered slide: {target}", flush=True)

    for figure_id, population_id in HISTOGRAM_IDS.items():
        payload = composers[figure_id](inputs, settings)
        for style in STYLES:
            for coloring in COLORINGS:
                slide = {"style": style, "coloring": coloring}
                emit(
                    tree_only_chart(payload, settings, population_id, style, coloring),
                    "tree_only",
                    f"papkou2023_{population_id}_tree",
                    style,
                    coloring,
                )
                emit(
                    build_tree_histogram_figure(
                        payload, settings, figure_id, slide=slide
                    ),
                    "full",
                    figure_id,
                    style,
                    coloring,
                )
    payload = composers[COMPARISON_ID](inputs, settings)
    for style in STYLES:
        for coloring in COLORINGS:
            emit(
                build_tree_comparison_figure(
                    payload, settings, slide={"style": style, "coloring": coloring}
                ),
                "full",
                COMPARISON_ID,
                style,
                coloring,
            )
    if not full_graph:
        check_sizes(written)
        return written
    payload = composers[FULL_GRAPH_ID](inputs, settings)
    for style in STYLES:
        for coloring in COLORINGS:
            emit(
                build_full_tree_persistence_figure(
                    payload, settings, style, slide={"coloring": coloring}
                ),
                "full",
                FULL_GRAPH_ID,
                style,
                coloring,
            )
    check_sizes(written)
    return written


def _png_size(path: Path) -> tuple[int, int]:
    return struct.unpack(">II", path.read_bytes()[16:24])


def check_sizes(paths: list[Path]) -> None:
    """All tree-only slides share one pixel size; full slides share one per figure."""
    sizes: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for path in paths:
        group = path.parent.name
        stem = path.name.split("_style=")[0].replace("_color=none", "")
        key = ("tree_only", "") if group == "tree_only" else ("full", stem)
        sizes.setdefault(key, set()).add(_png_size(path))
    odd = {key: found for key, found in sizes.items() if len(found) != 1}
    if odd:
        raise ValueError(f"Slide PNGs differ in pixel size within a group: {odd}")
