"""Frameless slide trees from saved products: ``figures/slides/{tree_only,full}/``.

For each D10 tree and band tree it renders, in every style (``vanilla``,
``annotated``) and coloring (``count``, ``none``):

- ``tree_only``: the tree panel alone (``figures/slide_tree.py``);
- ``full``: the existing view (``tree_persistence`` or ``bootstrap_band``) with its
  histogram and persistence diagram, with every title, subtitle and row heading
  removed.

Both groups share one fitness axis (``figures.fitness_domain``) and one panel size,
so every file of a group has the same pixel size and the same plot-area offset.
Uncoloured files are ``<stem>_color=none_style=<style>.png``; coloured files keep
the D10 stem.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..figures.view_naming import single_result_figure_name, stable_figure_path
from ..figures.slide_tree import (
    build_tree_only_figure,
    slide_figure_name,
    slide_fitness_domain,
    slide_stop_overrides,
)
from ..figures.tree_persistence_view import (
    COLORINGS,
    STYLES,
    build_tree_persistence_figure,
)
from ..figures.bootstrap_band_view import build_bootstrap_band_figure
from ..results import RunContext, parse_label
from .d10_tree_inputs import SavedTreeInputs, load_saved_tree_inputs
from .story_view_inputs import band_survival

GROUPS = ("tree_only", "full")


def slide_path(figures_root: Path, group: str, view: str, params, style, coloring, ext):
    """``<root>/figures/slides/<group>/<name>`` for one slide render."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}.")
    stable = stable_figure_path(
        figures_root, view, single_result_figure_name(view, params, style, ext)
    )
    return (
        figures_root
        / "figures"
        / "slides"
        / group
        / slide_figure_name(stable.name, coloring)
    )


def build_slide_trees(
    context: RunContext,
    settings: dict[str, Any],
    figures_root: Path,
    tree_labels: tuple[str, ...],
    band_labels: tuple[str, ...],
    *,
    formats: tuple[str, ...] = ("png",),
    loader: Callable[[RunContext, str], SavedTreeInputs] = load_saved_tree_inputs,
) -> list[Path]:
    """Render every slide file of the given D10 trees and band trees."""
    from . import _render_one, register_font_directory

    register_font_directory(context=context)
    dpi = int(settings["figures"].get("png_dpi", 300))
    written: list[Path] = []

    def emit(chart: Any, group, view, params, style, coloring) -> None:
        for ext in formats:
            target = slide_path(figures_root, group, view, params, style, coloring, ext)
            target.parent.mkdir(parents=True, exist_ok=True)
            _render_one(chart, target, ext, png_dpi=dpi)
            written.append(target)

    for label in tree_labels:
        inputs = loader(context, label)
        params = dict(parse_label(label, validate=False).params)
        domain = slide_fitness_domain(settings, inputs)
        overrides = slide_stop_overrides(settings, params.get("graph") == "full")
        for style in STYLES:
            for coloring in COLORINGS:
                emit(
                    build_tree_only_figure(
                        inputs, settings, style, coloring=coloring, scalar_domain=domain,
                        rendering_overrides=overrides,
                    ),
                    "tree_only",
                    "tree_persistence",
                    params,
                    style,
                    coloring,
                )
                emit(
                    build_tree_persistence_figure(
                        inputs,
                        settings,
                        style,
                        title=None,
                        subtitle=None,
                        coloring=coloring,
                        scalar_domain=domain,
                        rendering_overrides=overrides,
                    ),
                    "full",
                    "tree_persistence",
                    params,
                    style,
                    coloring,
                )
    for label in band_labels:
        nominal, band, survival = band_survival(context, label, {}, loader)
        params = dict(parse_label(label, validate=False).params)
        domain = slide_fitness_domain(settings, nominal)
        slide_fitness_domain(settings, band)
        for style in STYLES:
            for coloring in COLORINGS:
                emit(
                    build_tree_only_figure(
                        band,
                        settings,
                        style,
                        coloring=coloring,
                        scalar_domain=domain,
                        view="bootstrap_band",
                    ),
                    "tree_only",
                    "bootstrap_band",
                    params,
                    style,
                    coloring,
                )
                emit(
                    build_bootstrap_band_figure(
                        nominal,
                        band,
                        survival,
                        settings,
                        style,
                        title=None,
                        subtitle=None,
                        coloring=coloring,
                        scalar_domain=domain,
                    ),
                    "full",
                    "bootstrap_band",
                    params,
                    style,
                    coloring,
                )
    check_slide_sizes(written)
    return written


def _png_size(path: Path) -> tuple[int, int]:
    return struct.unpack(">II", path.read_bytes()[16:24])


def check_slide_sizes(paths: list[Path]) -> None:
    """Every PNG of a group (and, for ``full``, of a view) must have one pixel size."""
    sizes: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for path in paths:
        if path.suffix != ".png":
            continue
        group = path.parent.name
        view = path.name.split("__")[0] if group == "full" else ""
        sizes.setdefault((group, view), set()).add(_png_size(path))
    odd = {key: found for key, found in sizes.items() if len(found) != 1}
    if odd:
        raise ValueError(f"Slide PNGs differ in pixel size within a group: {odd}")
