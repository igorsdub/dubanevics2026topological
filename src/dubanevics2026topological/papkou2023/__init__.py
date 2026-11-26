"""Papkou et al. (2023) landscape: network construction, backbone analysis and checkpoints.

Ported from the last Topographer development revision that contained it (MIT License,
Copyright (c) 2026 Igors Dubanevics), which is not in the v0.5.0 release.
"""

from __future__ import annotations

from dubanevics2026topological.papkou2023.analysis import (
    Papkou2023TreeAnalysis,
    build_papkou2023_backbone_analysis,
    load_papkou2023_analysis_snapshot,
    save_papkou2023_analysis_snapshot,
)
from dubanevics2026topological.papkou2023.network import (
    Papkou2023Workflow,
    build_papkou2023_fitness_edge_table,
    build_papkou2023_fitness_graph,
    build_papkou2023_fitness_vertex_table,
    build_papkou2023_full_network,
    build_papkou2023_threshold_network,
    load_papkou2023_article_network,
    normalize_papkou2023_graph,
    select_papkou2023_largest_weak_component,
)

__all__ = [
    "Papkou2023TreeAnalysis",
    "Papkou2023Workflow",
    "build_papkou2023_backbone_analysis",
    "build_papkou2023_fitness_edge_table",
    "build_papkou2023_fitness_graph",
    "build_papkou2023_fitness_vertex_table",
    "build_papkou2023_full_network",
    "build_papkou2023_threshold_network",
    "load_papkou2023_analysis_snapshot",
    "load_papkou2023_article_network",
    "normalize_papkou2023_graph",
    "save_papkou2023_analysis_snapshot",
    "select_papkou2023_largest_weak_component",
]
