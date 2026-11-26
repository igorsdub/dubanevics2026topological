"""Selectable horizontal layouts for split and join trees.

Topographer's ``make_tree_layout`` takes a fixed set of ``horizontal_layout``
names and no callback for the order of the children or for the x coordinates
(see the layout notes kept with the author's working files). This module is the wrapper
the exploration needs: :func:`layout_tree` returns the same ``TreePlotData``
tables as ``make_tree_layout`` for every layout in :data:`HORIZONTAL_LAYOUTS`.

``topographer_default`` and ``subtree_extremum`` are delegated to Topographer.
``persistence_wt_anchored`` and ``tidy`` take the tables Topographer builds
(so every derived column, such as branch vertex counts, is Topographer's own)
and replace only the horizontal coordinates.

Every ordering rule here is keyed on a stable sequence identity (the node
payload attribute ``identity_attr``, ``dna_sequence`` by default) and on the
scalar, never on a Rustworkx node index, so the same tree drawn from a graph
whose indices were assigned in another order gets the same picture. The tree
nodes of a split tree are critical points; a sequence that is not itself a
critical point sits on a branch and is anchored by that branch's upper
critical point (:func:`anchor_identity_for_sequence`).

This module is exploratory. No production figure calls it. The production call
sites take a ``horizontal_layout`` argument (default ``subtree_extremum``; callers pass
``figures._layout_config.tree_horizontal_layout()``, the ``[rendering]
tree_horizontal_layout`` key of ``config/figures.toml``) for ``make_tree_layout``; the Papkou
backbone trees take ``subtree_extremum`` from Topographer's example.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
import rustworkx as rx
from topographer.plotting import TreePlotData, make_tree_layout

HORIZONTAL_LAYOUTS: tuple[str, ...] = (
    "topographer_default",
    "subtree_extremum",
    "persistence_wt_anchored",
    "tidy",
)
"""Layout names accepted by :func:`layout_tree`."""

_TOPOGRAPHER_OPTION = {
    # ``leaf_order`` is ``make_tree_layout``'s own default: children by ascending
    # scalar, x normalised to [0, 1].
    "topographer_default": "leaf_order",
    "subtree_extremum": "subtree_extremum",
}


def layout_tree(
    tree: Any,
    *,
    horizontal_layout: str = "subtree_extremum",
    anchor_identity: str | None = None,
    identity_attr: str = "dna_sequence",
    scalar_attr: str | None = None,
    network: Any | None = None,
    root_kind: str | None = None,
    node_attrs: Sequence[str] = (),
    edge_attrs: Sequence[str] = (),
    progress_callback: Callable[[int, int], None] | None = None,
) -> TreePlotData:
    """Lay out a tree like ``make_tree_layout`` with a selectable horizontal rule.

    Parameters
    ----------
    horizontal_layout
        One of :data:`HORIZONTAL_LAYOUTS`.

        ``topographer_default``
            ``make_tree_layout(horizontal_layout="leaf_order")``: children by
            ascending scalar, x in [0, 1].
        ``subtree_extremum``
            Topographer's ADR 0011 order: the child whose subtree reaches
            farthest from the root goes first (leftmost); iplotx x coordinates
            (leaves at 0, 1, 2, ..., every internal node at the mean of its
            children).
        ``persistence_wt_anchored``
            Branch decomposition by the elder rule. The path to the root of
            the summit of the anchor's branch (the path from the root up to
            that summit) is one vertical line at ``x = 0``. Every other branch
            is one column to its right. At each node the children are ordered
            by the reach of their arm, largest first, which is the subtree
            extremum order; ties go to the extremum's identity. The child on
            the path to the anchor goes first whatever its reach.
        ``tidy``
            Centred like a Reingold-Tilford drawing but not a true one (there
            is no contour packing): leaves at 0, 1, 2, ... in depth-first
            order, each internal node at the midpoint of its first and last
            child, the child holding the anchor first and the others in the
            same arm-reach order.
    anchor_identity
        Identity of the critical node that must lie on the vertical path to
        the root (the upper critical point of the branch that holds the wild
        type). Used by the two anchored layouts only; ignored by the others.
        ``None``, or an identity that is not a node of this tree, anchors on
        the global extremum instead: the vertical path is the chain of elder
        arms.
    identity_attr
        Node payload attribute holding the unique sequence identity. It is
        copied into the node table (added to ``node_attrs`` when missing).

    The other parameters are those of ``make_tree_layout``. The custom
    layouts do not support ``network`` (its ``vertex_table`` is keyed on x
    coordinates that they replace) and raise ``ValueError`` when it is given.

    Returns
    -------
    TreePlotData
        The tables ``make_tree_layout`` returns, with the same columns.
    """
    if horizontal_layout not in HORIZONTAL_LAYOUTS:
        raise ValueError(
            f"horizontal_layout must be one of {', '.join(HORIZONTAL_LAYOUTS)}; "
            f"got {horizontal_layout!r}."
        )
    attrs = tuple(node_attrs) + (
        () if identity_attr in node_attrs else (identity_attr,)
    )
    if horizontal_layout in _TOPOGRAPHER_OPTION:
        return make_tree_layout(
            tree,
            scalar_attr=scalar_attr,
            network=network,
            root_kind=root_kind,
            node_attrs=attrs,
            edge_attrs=edge_attrs,
            horizontal_layout=_TOPOGRAPHER_OPTION[horizontal_layout],
            progress_callback=progress_callback,
        )
    if network is not None:
        raise ValueError(
            f"horizontal_layout={horizontal_layout!r} does not support a network "
            "argument: the vertex table would keep Topographer's x coordinates."
        )
    base = make_tree_layout(
        tree,
        scalar_attr=scalar_attr,
        root_kind=root_kind,
        node_attrs=attrs,
        edge_attrs=edge_attrs,
        horizontal_layout="iplotx",
        progress_callback=progress_callback,
    )
    rooted = _rooted_tree(base, identity_attr)
    anchor = rooted.node_by_identity.get(anchor_identity) if anchor_identity else None
    ordered = _children_in_anchor_order(rooted, anchor)
    place = (
        _place_branch_columns
        if horizontal_layout == "persistence_wt_anchored"
        else _place_tidy
    )
    return _with_positions(base, place(rooted, ordered))


def anchor_identity_for_sequence(
    graph: Any,
    sequence: str,
    *,
    identity_attr: str = "dna_sequence",
    scalar_attr: str = "scalar",
    root_kind: str = "minimum",
) -> str | None:
    """Return the identity of the critical node that carries ``sequence``.

    A sequence that is itself a node of ``graph`` is its own anchor. A sequence
    carried on a branch (listed in the edge's ``vertex_chain_json``) is anchored
    by the branch's endpoint farther from the root: the higher one for a split
    tree (``root_kind="minimum"``), the lower one for a join tree. Returns
    ``None`` when the sequence is in neither place.
    """
    sign = _sign(root_kind)
    for index in graph.node_indices():
        if graph[index].get(identity_attr) == sequence:
            return str(graph[index][identity_attr])
    for source, target in graph.edge_list():
        chain = (graph.get_edge_data(source, target) or {}).get("vertex_chain_json")
        if isinstance(chain, str):
            if sequence not in chain:
                continue
            chain = json.loads(chain)
        if any(member.get(identity_attr) == sequence for member in chain or ()):
            outer = max(
                (source, target), key=lambda i: sign * float(graph[i][scalar_attr])
            )
            return str(graph[outer][identity_attr])
    return None


def crop_to_subtree(
    graph: Any, root_identity: str, *, identity_attr: str = "dna_sequence"
) -> Any:
    """Return a copy of the part of a tree at or above one critical node.

    ``graph`` is a ``rustworkx.PyDiGraph`` whose edges point from the upper
    critical point to the lower one (the orientation of every tree in this
    repository). The result holds the node named by ``root_identity`` and every
    node reachable by walking up from it, with copied payloads, so the named
    node becomes the new root. It is the "cropped" tree of a ridge excerpt.
    """
    if not isinstance(graph, rx.PyDiGraph):
        raise TypeError("crop_to_subtree needs a rustworkx.PyDiGraph (upper -> lower).")
    matches = [
        i for i in graph.node_indices() if graph[i].get(identity_attr) == root_identity
    ]
    if len(matches) != 1:
        raise ValueError(f"{root_identity!r} must name exactly one node.")
    keep: dict[int, int] = {}
    pending = [matches[0]]
    while pending:
        node = pending.pop()
        if node in keep:
            continue
        keep[node] = -1
        pending.extend(graph.predecessor_indices(node))
    cropped = rx.PyDiGraph()
    for node in sorted(keep):
        keep[node] = cropped.add_node(dict(graph[node]))
    for source, target in graph.edge_list():
        if source in keep and target in keep:
            cropped.add_edge(
                keep[source],
                keep[target],
                dict(graph.get_edge_data(source, target) or {}),
            )
    return cropped


# -- rooted tree read back from the layout tables ---------------------------


@dataclass(frozen=True)
class _Rooted:
    """A rooted tree keyed on tree node index, ordered only through identity."""

    root: int
    sign: float
    children: dict[int, tuple[int, ...]]
    identity: dict[int, str]
    height: dict[int, float]
    node_by_identity: dict[str, int]


def _sign(root_kind: str) -> float:
    """Return +1 when scalars rise away from the root (split tree), else -1."""
    if root_kind == "minimum":
        return 1.0
    if root_kind == "maximum":
        return -1.0
    raise ValueError("root_kind must be 'minimum' or 'maximum'.")


def _rooted_tree(plot_data: TreePlotData, identity_attr: str) -> _Rooted:
    nodes = plot_data.node_table
    if identity_attr not in nodes.columns:
        raise ValueError(
            f"The node table has no {identity_attr!r} column; name the identity "
            "attribute of the tree nodes with identity_attr."
        )
    identity = {
        int(i): v
        for i, v in zip(nodes["tree_node_index"], nodes[identity_attr], strict=True)
    }
    if any(not isinstance(v, str) or not v for v in identity.values()):
        raise ValueError(f"Every tree node needs a non-empty {identity_attr!r}.")
    node_by_identity = {v: i for i, v in identity.items()}
    if len(node_by_identity) != len(identity):
        raise ValueError(f"Tree node {identity_attr!r} values must be unique.")
    sign = _sign(plot_data.root_kind)
    height = {
        int(i): sign * float(s)
        for i, s in zip(nodes["tree_node_index"], nodes["scalar"], strict=True)
    }
    children: dict[int, list[int]] = {i: [] for i in identity}
    edges = plot_data.edge_table
    if edges.height:
        for parent, child in zip(
            edges["source_tree_node_index"],
            edges["target_tree_node_index"],
            strict=True,
        ):
            children[int(parent)].append(int(child))
    return _Rooted(
        root=int(plot_data.root_node_index),
        sign=sign,
        children={
            i: tuple(sorted(c, key=identity.__getitem__)) for i, c in children.items()
        },
        identity=identity,
        height=height,
        node_by_identity=node_by_identity,
    )


def _preorder(rooted: _Rooted, children: dict[int, Any]) -> list[int]:
    order: list[int] = []
    stack = [rooted.root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children[node]))
    return order


def _children_in_anchor_order(
    rooted: _Rooted, anchor: int | None
) -> dict[int, list[int]]:
    """Order every node's children: anchor side first, then arm reach, largest first.

    The reach of the arm to child ``c`` of node ``v`` is the height that its
    whole subtree reaches above ``v``. For a non-elder arm this equals the
    ordinary H0 persistence of the arm's summit, paired with ``v`` as its merge
    saddle. For the elder arm it does not: the elder summit's ordinary
    persistence is closed at an ancestor of ``v`` (or at the global minimum by
    convention), so there reach and persistence are different quantities.
    Sorting siblings by reach equals sorting them by the subtree extremum, the
    key of ADR 0011. Exact ties go to the identity of the first node holding
    the extremum, so no Rustworkx index decides anything.
    """
    # best[v] = (-reach, identity): the lexicographic minimum over v's subtree
    # of (-height, identity), taken children before parents.
    best: dict[int, tuple[float, str]] = {}
    for node in reversed(_preorder(rooted, rooted.children)):
        own = (-rooted.height[node], rooted.identity[node])
        best[node] = min([own, *(best[c] for c in rooted.children[node])])

    def arm_reach(parent: int, child: int) -> float:
        return -best[child][0] - rooted.height[parent]

    toward_anchor: dict[int, int] = {}
    if anchor is not None:
        parents = {c: p for p, cs in rooted.children.items() for c in cs}
        node = anchor
        while node != rooted.root:
            toward_anchor[parents[node]] = node
            node = parents[node]
    ordered: dict[int, list[int]] = {}
    for node, children in rooted.children.items():
        arms = sorted(children, key=lambda c: (-arm_reach(node, c), best[c][1]))
        first = toward_anchor.get(node)
        if first is not None:
            arms.remove(first)
            arms.insert(0, first)
        ordered[node] = arms
    return ordered


def _place_branch_columns(
    rooted: _Rooted, ordered: dict[int, list[int]]
) -> dict[int, float]:
    """One column per branch: the first child continues its parent's column.

    The first child of every node, which is the anchor side on the path to
    the anchor's summit and the arm of greatest reach elsewhere, shares its
    parent's x, so each branch is one vertical line. Every other child opens
    the next free column, and a child's whole subtree is placed before its
    next sibling, so subtrees are contiguous intervals and a connector at a
    saddle's height crosses no other branch. The path from the root to the
    anchor's summit is column 0 and everything else lies to its right.
    """
    x = {rooted.root: 0.0}
    next_column = 1
    stack: list[tuple[int, int]] = []

    def push(node: int) -> None:
        stack.extend((c, node) for c in reversed(ordered[node]))

    push(rooted.root)
    while stack:
        node, parent = stack.pop()
        if node == ordered[parent][0]:
            x[node] = x[parent]
        else:
            x[node] = float(next_column)
            next_column += 1
        push(node)
    return x


def _place_tidy(rooted: _Rooted, ordered: dict[int, list[int]]) -> dict[int, float]:
    """Leaves at 0, 1, 2, ... in order; each parent centred over its children."""
    order = _preorder(rooted, ordered)
    x: dict[int, float] = {}
    next_leaf = 0
    for node in order:
        if not ordered[node]:
            x[node] = float(next_leaf)
            next_leaf += 1
    for node in reversed(order):
        if ordered[node]:
            x[node] = (x[ordered[node][0]] + x[ordered[node][-1]]) / 2
    return x


# -- writing positions back into Topographer's tables -----------------------


def _orthogonal_edge_segments(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Encode each edge as its vertical and horizontal segment.

    The same encoding as Topographer's own layouts; a test pins it against
    hand-computed segments.
    """
    segments: list[dict[str, Any]] = []
    for logical_edge_index, edge in enumerate(edge_rows):
        common = {**edge, "logical_edge_index": logical_edge_index}
        segments.append(
            {
                **common,
                "segment_index": 0,
                "segment_kind": "vertical",
                "x_start": edge["x_target"],
                "y_start": edge["y_target"],
                "x_end": edge["x_target"],
                "y_end": edge["y_source"],
            }
        )
        segments.append(
            {
                **common,
                "segment_index": 1,
                "segment_kind": "horizontal",
                "x_start": edge["x_target"],
                "y_start": edge["y_source"],
                "x_end": edge["x_source"],
                "y_end": edge["y_source"],
            }
        )
    return segments


def _with_positions(base: TreePlotData, x: dict[int, float]) -> TreePlotData:
    """Return ``base`` with its x coordinates replaced by ``x`` (by node index)."""
    node_table = base.node_table.with_columns(
        pl.Series(
            "x",
            [x[int(i)] for i in base.node_table["tree_node_index"]],
            dtype=pl.Float64,
        )
    )
    edge_table = base.edge_table
    if edge_table.height:
        edge_table = edge_table.with_columns(
            pl.Series(
                "x_source",
                [x[int(i)] for i in edge_table["source_tree_node_index"]],
                dtype=pl.Float64,
            ),
            pl.Series(
                "x_target",
                [x[int(i)] for i in edge_table["target_tree_node_index"]],
                dtype=pl.Float64,
            ),
        )
    return dataclasses.replace(
        base,
        node_table=node_table,
        edge_table=edge_table,
        # The schema fixes Topographer's column order and dtypes: with edge
        # attributes present on only some edges, the order inferred from the
        # rebuilt rows differs from Topographer's own table.
        edge_segment_table=pl.DataFrame(
            _orthogonal_edge_segments(edge_table.to_dicts()),
            schema=base.edge_segment_table.schema,
        ),
    )
