"""Draw one comparison PNG per horizontal tree layout (exploration only).

For each layout in ``dubanevics2026topological.analysis.tree_layouts`` the figure has
four panels, left to right:

1. the full-graph split tree;
2. the published-threshold tree (``study_clamped``);
3. the candidate-threshold tree (``candidate_clamped``);
4. the KEM/TCL excerpt: the selected-leaf simplified Candidate tree cropped to the
   common saddle of the KEM and TCL summits and everything above it, laid out as a
   tree of its own.

The wild type (WT) is a red star on its branch, and a red ring marks the summit of
that branch. A blue line is the path to the root of that summit (the path from the
root up to it), the line the layouts are judged on. Each panel title carries four
numbers (see ``measure``).

This script only reads the cached analyses (``just comparison-analysis`` and
``just peak-analysis`` build them) and never recomputes them. Appearance comes from
``[layout_exploration]`` in ``config/figures.toml``. PNGs go to
``output/figures/layout_exploration/<layout>.png``, with ``metrics.csv`` beside them.

    PYTHONPATH=src uv run python scripts/layout_exploration.py [--layout NAME ...]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.collections import LineCollection

from dubanevics2026topological.analysis import ROOT, load_settings
from dubanevics2026topological.analysis.tree_layouts import (
    HORIZONTAL_LAYOUTS,
    anchor_identity_for_sequence,
    crop_to_subtree,
    layout_tree,
)
from dubanevics2026topological.build import load_inputs
from dubanevics2026topological.figures._layout_config import tree_horizontal_layout
from dubanevics2026topological.simplification.selected_leaf_simplification import (
    build_candidate_leaf_view,
)

OUTPUT_DIR = ROOT / "output" / "figures" / "layout_exploration"
IDENTITY = "dna_sequence"
EPSILON = 1e-9
LABEL_OFFSET_PT = 7.0  # gap between the WT star and the right edge of its label
EDGE_GAP_PT = 4.0  # clear space between a drawn object and the edge of the axes


@dataclass(frozen=True)
class Panel:
    """One tree to lay out: a graph, where the WT sits, and how to title it."""

    key: str
    title: str
    graph: Any
    anchor: str | None
    wt_scalar: float
    note: str


def _common_saddle(graph: Any, first: str, second: str) -> str:
    """Identity of the first node shared by the two summits' paths to the root."""
    by_identity = {graph[i][IDENTITY]: i for i in graph.node_indices()}

    def path(identity: str) -> list[int]:
        node = by_identity[identity]
        nodes = [node]
        while graph.out_degree(node):
            node = next(iter(graph.successor_indices(node)))
            nodes.append(node)
        return nodes

    second_path = set(path(second))
    shared = next((n for n in path(first) if n in second_path), None)
    if shared is None:
        raise ValueError("The two summits do not share a saddle.")
    return str(graph[shared][IDENTITY])


def build_panels(settings: dict[str, Any]) -> list[Panel]:
    """Build the four panels from the cached analyses (nothing is recomputed)."""
    inputs = load_inputs(["comparison", "peak"], settings)
    landscapes = inputs["comparison"]["landscapes"]
    panels: list[Panel] = []
    wild_type = None
    for key, title in (
        ("full", "Full graph"),
        ("study_clamped", "Published functional threshold"),
        ("candidate_clamped", "Candidate functional threshold"),
    ):
        tree_analysis = landscapes[key].trees["split"]
        marker = tree_analysis.wild_type_marker_table
        if marker.is_empty():
            raise ValueError(f"The {key} tree has no wild-type marker.")
        wild_type = str(marker[IDENTITY][0])
        panels.append(
            Panel(
                key,
                title,
                tree_analysis.tree,
                anchor_identity_for_sequence(tree_analysis.tree, wild_type),
                float(marker["scalar"][0]),
                "",
            )
        )
    view = build_candidate_leaf_view(
        landscapes["candidate_clamped"],
        inputs["peak"]["tables"]["candidate_clamped_contrasts"],
        float(settings["analysis"]["candidate_peak_biology"]["q_value_threshold"]),
        horizontal_layout=tree_horizontal_layout(settings),
    )
    codon = settings["analysis"]["codon_entropy"]
    saddle = _common_saddle(
        view.tree, codon["kem_summit_dna_sequence"], codon["tcl_summit_dna_sequence"]
    )
    excerpt = crop_to_subtree(view.tree, saddle)
    located = view.assignments.filter(pl.col(IDENTITY) == wild_type)
    anchor = str(located["upper_sequence"][0]) if located.height else None
    in_excerpt = {excerpt[i][IDENTITY] for i in excerpt.node_indices()}
    note = ""
    if anchor not in in_excerpt:
        anchor, note = None, "WT is not in this excerpt"
    panels.append(
        Panel(
            "excerpt",
            "KEM/TCL excerpt",
            excerpt,
            anchor,
            float(located["raw_fitness"][0]) if located.height else 0.0,
            note,
        )
    )
    return panels


def _summit(graph: Any, anchor: str) -> str:
    """The highest node at or above the anchor (ties by identity)."""
    by_identity = {graph[i][IDENTITY]: i for i in graph.node_indices()}
    pending, best = [by_identity[anchor]], None
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        key = (-float(graph[node]["scalar"]), str(graph[node][IDENTITY]))
        best = key if best is None or key < best else best
        pending.extend(graph.predecessor_indices(node))
    assert best is not None
    return best[1]


def _crossings(segments: pl.DataFrame) -> int:
    """Count horizontal segments that cross a vertical one inside both."""
    vertical = segments.filter(pl.col("segment_kind") == "vertical")
    horizontal = segments.filter(pl.col("segment_kind") == "horizontal")
    vx = vertical["x_start"].to_numpy()
    v_ends = vertical.select("y_start", "y_end").to_numpy()
    v_lo, v_hi = v_ends.min(axis=1), v_ends.max(axis=1)
    hy = horizontal["y_start"].to_numpy()
    h_ends = horizontal.select("x_start", "x_end").to_numpy()
    h_lo, h_hi = h_ends.min(axis=1), h_ends.max(axis=1)
    total = 0
    for start in range(0, len(hy), 400):
        part = slice(start, start + 400)
        inside_x = (h_lo[part, None] + EPSILON < vx[None, :]) & (
            vx[None, :] < h_hi[part, None] - EPSILON
        )
        inside_y = (v_lo[None, :] + EPSILON < hy[part, None]) & (
            hy[part, None] < v_hi[None, :] - EPSILON
        )
        total += int((inside_x & inside_y).sum())
    return total


def measure(plot: Any, panel: Panel, summit: str | None) -> dict[str, Any]:
    """Numbers that judge a layout against the invariant.

    ``wt_x`` is the WT branch's x as a fraction of the tree's width (0 = left edge,
    1 = right edge). ``path_to_root_drift`` is the x range along the path to the
    root of the WT branch's summit, as a fraction of the width (0 = vertical).
    ``leaves_left`` and ``leaves_right`` are the shares of leaves left and right of
    the root's x (a path to the root with leaves on both sides is wrapped by
    them). ``crossings`` counts connectors that cross a
    branch they do not belong to.
    """
    nodes = plot.node_table
    x = dict(zip(nodes["tree_node_index"], nodes["x"]))
    identity = dict(zip(nodes[IDENTITY], nodes["tree_node_index"]))
    width = max(x.values()) - min(x.values()) or 1.0
    parent = dict(
        zip(
            plot.edge_table["target_tree_node_index"],
            plot.edge_table["source_tree_node_index"],
        )
    )
    children = set(parent.values())
    leaves = [n for n in x if n not in children]
    root_x = x[plot.root_node_index]
    row: dict[str, Any] = {
        "panel": panel.key,
        "nodes": len(x),
        "leaves": len(leaves),
        "wt_x": None,
        "path_to_root_drift": None,
        "leaves_left": sum(x[n] < root_x - EPSILON for n in leaves) / len(leaves),
        "leaves_right": sum(x[n] > root_x + EPSILON for n in leaves) / len(leaves),
        "crossings": _crossings(plot.edge_segment_table),
    }
    if panel.anchor is not None and summit is not None:
        row["wt_x"] = (x[identity[panel.anchor]] - min(x.values())) / width
        path, node = [], identity[summit]
        while True:
            path.append(x[node])
            if node not in parent:
                break
            node = parent[node]
        row["path_to_root_drift"] = (max(path) - min(path)) / width
    return row


def _draw(
    ax: Any, plot: Any, panel: Panel, summit: str | None, look: dict, red: str
) -> Any:
    """Draw one tree on ``ax``; return the WT label (``None`` without a WT)."""
    segments = plot.edge_segment_table
    lines = [
        ((r["x_start"], r["y_start"]), (r["x_end"], r["y_end"]))
        for r in segments.to_dicts()
    ]
    ax.add_collection(
        LineCollection(lines, colors=look["edge_color"], linewidths=look["edge_width"])
    )
    nodes = plot.node_table
    x = dict(zip(nodes["tree_node_index"], nodes["x"]))
    identity = dict(zip(nodes[IDENTITY], nodes["tree_node_index"]))
    if summit is not None:
        parent = dict(
            zip(
                plot.edge_table["target_tree_node_index"],
                plot.edge_table["source_tree_node_index"],
            )
        )
        on_path, node = set(), identity[summit]
        while node in parent:
            on_path.add((parent[node], node))
            node = parent[node]
        path_segments = [
            ((r["x_start"], r["y_start"]), (r["x_end"], r["y_end"]))
            for r in segments.to_dicts()
            if (r["source_tree_node_index"], r["target_tree_node_index"]) in on_path
        ]
        ax.add_collection(
            LineCollection(
                path_segments,
                colors=look["path_to_root_color"],
                linewidths=look["path_to_root_width"],
            )
        )
        summit_node = identity[summit]
        ax.scatter(
            [x[summit_node]],
            [float(nodes.filter(pl.col("tree_node_index") == summit_node)["y"][0])],
            s=look["summit_size"],
            facecolors="none",
            edgecolors=red,
            linewidths=1.2,
            zorder=4,
            clip_on=False,  # the global summit sits on the top edge of the axes
        )
    if panel.anchor is not None:
        wt_x = x[identity[panel.anchor]]
        ax.scatter(
            [wt_x],
            [panel.wt_scalar],
            marker="*",
            s=look["wild_type_size"],
            color=red,
            zorder=5,
            clip_on=False,
        )
        return ax.annotate(
            "WT",
            (wt_x, panel.wt_scalar),
            xytext=(-LABEL_OFFSET_PT, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=red,
            fontsize=look["font_size"],
            zorder=5,
            annotation_clip=False,
        )
    return None


def _set_x_limits(
    fig: Any, ax: Any, x_min: float, x_max: float, label: Any | None
) -> None:
    """Pad the x range by margins fixed in points, so the WT label fits.

    The WT label sits to the left of the WT's branch, which is at the left edge
    of the tree when the WT is on the path drawn at x = 0. The left margin is the
    label offset, the label's measured width and a gap, so the label clears the
    axis spine whatever the tree's width. Call after ``fig.tight_layout()``,
    which fixes the axes width in points (no ticks, so x limits do not move it).
    """
    renderer = fig.canvas.get_renderer()
    points_per_pixel = 72.0 / fig.dpi
    left_pt = EDGE_GAP_PT
    if label is not None:
        label_pt = label.get_window_extent(renderer).width * points_per_pixel
        left_pt += LABEL_OFFSET_PT + label_pt
    axes_pt = ax.get_window_extent(renderer).width * points_per_pixel
    span = (x_max - x_min) or 1.0
    per_point = span / (axes_pt - left_pt - EDGE_GAP_PT)
    ax.set_xlim(x_min - left_pt * per_point, x_max + EDGE_GAP_PT * per_point)


def render_layout(
    name: str, panels: list[Panel], settings: dict[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    """Draw one four-panel PNG for ``name`` and return its metric rows."""
    look = settings["figures"]["layout_exploration"]
    rendering = settings["figures"]["rendering"]
    domain = settings["figures"]["fitness_domain"]
    red = rendering["wild_type_color"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": look["font_size"]})
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(look["panel_width_in"] * len(panels), look["panel_height_in"]),
    )
    rows = []
    pending: list[tuple[Any, float, float, Any]] = []
    for ax, panel in zip(axes, panels):
        plot = layout_tree(
            panel.graph,
            horizontal_layout=name,
            anchor_identity=panel.anchor,
            identity_attr=IDENTITY,
            scalar_attr="scalar",
            root_kind="minimum",
        )
        summit = _summit(panel.graph, panel.anchor) if panel.anchor else None
        label = _draw(ax, plot, panel, summit, look, red)
        metrics = measure(plot, panel, summit)
        rows.append({"layout": name, **metrics})
        x_values = plot.node_table["x"]
        pending.append((ax, float(x_values.min()), float(x_values.max()), label))
        ax.set_ylim(*domain)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        if ax is not axes[0]:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
        else:
            ax.set_ylabel(rendering["tree_axis_title"])
        wt_x = metrics["wt_x"]
        lines = [
            f"{metrics['leaves']:,} leaves",
            "WT x/width "
            + ("-" if wt_x is None else f"{wt_x:.2f}")
            + "  path-to-root drift "
            + (
                "-"
                if metrics["path_to_root_drift"] is None
                else f"{metrics['path_to_root_drift']:.2f}"
            ),
            (
                f"leaves left/right of root {metrics['leaves_left']:.2f}/"
                f"{metrics['leaves_right']:.2f}  crossings {metrics['crossings']:,}"
            ),
        ]
        if panel.note:
            lines.append(panel.note)
        ax.set_title(
            panel.title + "\n" + "\n".join(lines),
            fontsize=look["font_size"],
            loc="left",
        )
    fig.suptitle(f"horizontal_layout = {name}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    for ax, x_min, x_max, label in pending:
        _set_x_limits(fig, ax, x_min, x_max, label)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=look["dpi"])
    plt.close(fig)
    print(f"wrote {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", action="append", choices=HORIZONTAL_LAYOUTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    settings = load_settings()
    panels = build_panels(settings)
    rows: list[dict[str, Any]] = []
    for name in args.layout or HORIZONTAL_LAYOUTS:
        rows.extend(render_layout(name, panels, settings, args.output_dir))
    table = pl.DataFrame(rows)
    table.write_csv(args.output_dir / "metrics.csv")
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=40):
        print(table)


if __name__ == "__main__":
    main()
