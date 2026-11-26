# Ported verbatim from the last Topographer development revision that contained it
# (https://github.com/oist/topographer, MIT License, Copyright (c) 2026 Igors Dubanevics).
# Not in the Topographer v0.5.0 release. Imports of removed modules point here; other imports use 0.5.0.
"""Scientific source and input dependencies for Papkou analysis checkpoints."""

from __future__ import annotations

from pathlib import Path

LANDSCAPE_INPUTS = {
    "article": ("data/external/graph_largest_component.graphml",),
    "full": (
        "data/raw/papkou2023_fitness_wt.parquet",
        "data/raw/papkou2023_fitness_diff.parquet",
    ),
    "new-threshold": (
        "data/raw/papkou2023_fitness_wt.parquet",
        "data/raw/papkou2023_fitness_diff.parquet",
    ),
}


def papkou_analysis_dependencies(landscape: str) -> tuple[str, ...]:
    """Return repository-relative raw and scientific source prerequisites."""
    if landscape not in LANDSCAPE_INPUTS:
        raise ValueError(f"Unknown Papkou landscape {landscape!r}.")
    core_sources = [
        Path("src/topographer/examples/papkou2023/network.py"),
        Path("src/topographer/examples/papkou2023/analysis.py"),
        Path("src/topographer/examples/papkou2023/checkpoints.py"),
        Path("src/topographer/examples/papkou2023/dependencies.py"),
        Path("src/topographer/examples/_workflow.py"),
        Path("src/topographer/plotting/__init__.py"),
        Path("src/topographer/plotting/_data.py"),
        Path("src/topographer/plotting/_layout.py"),
        Path("src/topographer/plotting/_swarm.py"),
    ]
    for package in ("graph", "topology", "biology"):
        source_directory = Path("src/topographer") / package
        core_sources.extend(sorted(source_directory.glob("*.py")))
    # Reeb modules are outside the merge-tree calculation used by this workflow.
    core_sources = [
        path
        for path in core_sources
        if not path.as_posix().startswith("src/topographer/topology/_reeb")
    ]
    return (*LANDSCAPE_INPUTS[landscape], *(path.as_posix() for path in core_sources))
