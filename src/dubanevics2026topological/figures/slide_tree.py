"""Frameless slide trees: the tree panel alone, with no title, subtitle or companions.

A slide render drops every title block and shares one fitness axis
(``figures.fitness_domain``), so stacked slides keep their plot area at the same
pixel offset. ``coloring="none"`` is a separate axis from the glossary styles
(``vanilla`` = no markers, ``annotated`` = stop-codon crosses and the wild-type
star): it draws every branch in ``plain_edge_color`` and omits the count colour bar,
which sits over the panel and so never moves it. Everything is read from
``SavedTreeInputs``; nothing is rebuilt.
"""

from __future__ import annotations

from typing import Any

from ._charts import (
    count_domain,
    overlay_count_colorbar,
    plot_data_with_metrics,
    publication_scalar_scales,
    tree_chart,
)
from .tree_persistence_view import (
    COLORINGS,
    STYLES,
    configure_view_chart,
    vanilla_analysis,
)


def slide_fitness_domain(settings: dict[str, Any], inputs: Any) -> tuple[float, float]:
    """The shared slide fitness extent, which must contain the tree's own extent."""
    low, high = (float(x) for x in settings["figures"]["fitness_domain"])
    own_low, own_high = inputs.scalar_domain
    if own_low < low or own_high > high:
        raise ValueError(
            f"Tree fitness extent {inputs.scalar_domain} lies outside the shared "
            f"slide domain {(low, high)}; widen figures.fitness_domain."
        )
    return low, high


def build_tree_only_figure(
    inputs: Any,
    settings: dict[str, Any],
    style: str,
    *,
    coloring: str = "count",
    scalar_domain: tuple[float, float] | None = None,
    view: str = "tree_persistence",
    stop_identities: set[str] | None = None,
    rendering_overrides: dict[str, Any] | None = None,
) -> Any:
    """One saved tree as a single title-less panel.

    ``stop_identities`` restricts the stop crosses to those sequences; the node kinds
    that carry one are ``[rendering] stop_codon_tree_node_kinds``.

    ``view`` names the appearance table under ``d10`` (``tree_persistence`` or
    ``bootstrap_band``); both carry the same tree geometry, so the panel has the
    same size either way.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}.")
    if coloring not in COLORINGS:
        raise ValueError(f"coloring must be one of {COLORINGS}, got {coloring!r}.")
    figures = settings["figures"]
    figure_settings = figures["d10"][view]
    rendering = {
        **figures.get("rendering", {}),
        **figure_settings,
        **(rendering_overrides or {}),
    }
    domain = scalar_domain or slide_fitness_domain(settings, inputs)
    annotated = style == "annotated"
    tree_analysis = vanilla_analysis(inputs.tree_analysis, annotated)
    plot_data = plot_data_with_metrics(tree_analysis)
    branch_domain = count_domain(plot_data.edge_table)
    tree = tree_chart(
        plot_data,
        tree_analysis,
        inputs.analysis,
        figure_settings,
        rendering,
        branch_domain,
        linked=True,
        annotated=annotated,
        coloring=coloring,
        stop_identities=stop_identities,
    )
    if coloring == "count":
        tree = overlay_count_colorbar(tree, figure_settings, rendering, branch_domain)
    return publication_scalar_scales(
        configure_view_chart(tree, figure_settings, rendering), domain, rendering
    )


def slide_figure_name(stable_name: str, coloring: str) -> str:
    """``<stem>_color=none_style=<s>.<ext>`` for an uncoloured tree; coloured keep the stem.

    ``stable_name`` is a D10 figure file name ``<stem>_style=<s>.<ext>``.
    """
    if coloring not in COLORINGS:
        raise ValueError(f"coloring must be one of {COLORINGS}, got {coloring!r}.")
    if coloring == "count":
        return stable_name
    stem, separator, tail = stable_name.rpartition("_style=")
    if not separator:
        raise ValueError(f"{stable_name!r} has no _style= part.")
    return f"{stem}_color=none_style={tail}"


def slide_stop_overrides(settings: dict[str, Any], full_graph: bool) -> dict[str, Any]:
    """Smaller stop crosses for the Full-graph slides only (`[slides]` in figures.toml)."""
    if not full_graph:
        return {}
    table = settings["figures"].get("slides", {})
    return {
        key: table[f"full_graph_{key}"]
        for key in ("stop_codon_tree_size", "stop_codon_stroke_width")
        if f"full_graph_{key}" in table
    }


def slide_tree_analysis(tree_analysis: Any, style: str) -> tuple[Any, bool]:
    """``(analysis, annotated)`` of a registry tree for a slide ``style``.

    The annotated style needs the stop-codon table (built from the layout's node table
    when the saved layout carries none); the vanilla style drops the wild-type marker.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}.")
    annotated = style == "annotated"
    if annotated and not hasattr(tree_analysis, "all_stop_marker_table"):
        nodes = tree_analysis.plot_data.node_table
        # Every stop node with its kind; the figure layer keeps the allowed kinds.
        tree_analysis.all_stop_marker_table = nodes.filter(
            nodes["has_stop_codon"]
        ).select("x", "y", "dna_sequence", "aa_sequence", "display_event")
    return vanilla_analysis(tree_analysis, annotated), annotated
