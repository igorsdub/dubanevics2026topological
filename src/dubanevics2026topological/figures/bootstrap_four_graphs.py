"""A four-graph comparison of bootstrap band-simplified tree histograms."""

from __future__ import annotations

import copy
from typing import Any

import altair as alt

from ._charts import tree_histogram_chart_config
from ._count_labels import count_label_text
from .tree_histogram import build_bootstrap_tree_histogram_figure

FIGURE_ID = "bootstrap_four_graphs"
# Text of the pixel-positioned keys the first panel alone keeps: the count colour bar
# title and the wild-type entry of the tree key.
KEY_TEXTS = frozenset({"Branch vertex count", "Wild type", "Stop codon containing peaks"})

PANEL_ORDER = (
    ("threshold=published_fdr=none", "Published graph · fixed Papkou cutoff · point estimate"),
    ("threshold=reflected_fdr=none", "Candidate graph · Candidate threshold · point estimate"),
    ("threshold=published_fdr=by_q=0.05", "Published graph · Papkou cutoff · BY FDR q=0.05"),
    ("threshold=reflected_fdr=by_q=0.05", "Candidate graph · Candidate threshold · BY FDR q=0.05"),
)


def panel_title(variant: str, threshold: float) -> str:
    """Format a panel title from that panel's verified numerical threshold."""
    if variant not in {name for name, _ in PANEL_ORDER}:
        raise ValueError("Unknown bootstrap composite panel.")
    published = "threshold=published" in variant
    graph = "Published graph" if published else "Candidate graph"
    cutoff = f"T={threshold:.3f}" if published else f"Candidate threshold, T={threshold:.3f}"
    method = "BY FDR q=0.05" if "fdr=by" in variant else "point estimate"
    return f"{graph} · {cutoff} · {method}"


def panel_peaks_subtitle(kept: int | None, nominal: int | None = None) -> str:
    """The subtitle stating a panel's peaks kept by the bootstrap band.

    ``kept`` is read from the drawn (band-simplified) tree; ``nominal`` is the peak
    count of the graph's nominal tree, carried by the panel's source checkpoint. A
    no-break space when ``kept`` is unknown: every panel carries a subtitle line, so
    the four plot areas stay aligned.
    """
    if kept is None:
        return "\u00a0"
    if nominal is None:
        return f"{count_label_text(kept)} peaks after the bootstrap band"
    return f"{count_label_text(kept)} of {count_label_text(nominal)} peaks kept by the bootstrap band"


def _edge_pairs(node: Any, pairs: set[tuple[int, int]]) -> None:
    if isinstance(node, list):
        for item in node:
            _edge_pairs(item, pairs)
    elif isinstance(node, dict):
        if "source_tree_node_index" in node and "target_tree_node_index" in node:
            try:
                pairs.add((int(node["source_tree_node_index"]), int(node["target_tree_node_index"])))
            except (TypeError, ValueError):
                pass
        for value in node.values():
            _edge_pairs(value, pairs)


def peak_count_from_spec(spec: Any) -> int | None:
    """Peaks (tips) of the tree a panel draws, read from its own edge rows.

    A peak is the upper end of a branch that is the lower end of no other branch
    (the maxima of the split tree, as ``terminal_tips`` finds them). ``None`` when
    the specification carries no branch rows.
    """
    pairs: set[tuple[int, int]] = set()
    _edge_pairs(spec, pairs)
    if not pairs:
        return None
    lower_ends = {source for source, _ in pairs}
    return len({target for _, target in pairs if target not in lower_ends})


def _nested_panel_spec(saved: dict[str, Any]) -> dict[str, Any]:
    """Inline named data and remove properties allowed only on a top-level chart."""
    spec = copy.deepcopy(saved)
    datasets = spec.pop("datasets", {})

    def inline(value: Any) -> Any:
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: inline(item) for key, item in value.items()}
        data = result.get("data")
        if isinstance(data, dict) and data.get("name") in datasets:
            result["data"] = {**data, "values": datasets[data["name"]]}
            del result["data"]["name"]
        return result

    for key in ("$schema", "config", "background", "autosize", "padding"):
        spec.pop(key, None)
    return inline(spec)


def _leaf_marks(node: Any) -> list[dict[str, Any]]:
    """Every leaf view (a dict with a ``mark``) under ``node``."""
    if isinstance(node, list):
        return [leaf for item in node for leaf in _leaf_marks(item)]
    if not isinstance(node, dict):
        return []
    if "mark" in node:
        return [node]
    return [leaf for key in ("layer", "hconcat", "vconcat") for leaf in _leaf_marks(node.get(key, []))]


def _is_pixel_key(node: Any) -> bool:
    """True for a layer made only of pixel-positioned marks that carries a key text.

    Such layers (the count colour bar, the tree key) have no ``encoding`` on any
    leaf; the tree itself always has encoded leaves, so it is never matched.
    """
    leaves = _leaf_marks(node)
    if not leaves or any(leaf.get("encoding") for leaf in leaves):
        return False
    texts = {
        leaf["mark"].get("text")
        for leaf in leaves
        if isinstance(leaf["mark"], dict)
    }
    return bool(texts & KEY_TEXTS)


def pin_tree_y_axis(node: Any, extent: int) -> None:
    """Fix the width of every visible y axis under ``node`` to ``extent`` pixels.

    Vega-Lite sizes a panel's left margin from its own tick labels (the threshold
    tick is written with three decimals, the others with one) and axis title, so
    panels drawn in separate concats start their plot areas at different x. Equal
    ``minExtent`` and ``maxExtent`` give every panel the same margin.
    """
    if isinstance(node, list):
        for item in node:
            pin_tree_y_axis(item, extent)
        return
    if not isinstance(node, dict):
        return
    y = node.get("encoding", {}).get("y") if isinstance(node.get("encoding"), dict) else None
    if isinstance(y, dict) and "field" in y and y.get("axis", {}) is not None:
        y["axis"] = {**(y.get("axis") or {}), "minExtent": extent, "maxExtent": extent}
    for value in node.values():
        pin_tree_y_axis(value, extent)


def compact_panel_spec(
    saved: dict[str, Any],
    *,
    show_legend: bool,
    show_histogram_title: bool,
    y_axis_extent: int | None = None,
) -> dict[str, Any]:
    """Reduce repeated labels while retaining each graph's own count scale.

    ``show_legend`` False drops the count colour bar and the tree key, so the figure
    carries one of each (on its first panel).
    """
    spec = copy.deepcopy(saved)

    def compact(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                compact(item)
            return
        if not isinstance(value, dict):
            return
        layers = value.get("layer")
        if not show_legend and isinstance(layers, list):
            value["layer"] = [item for item in layers if not _is_pixel_key(item)]
        encoding = value.get("encoding", {})
        if isinstance(encoding, dict):
            text = encoding.get("text", {})
            x = encoding.get("x", {})
            is_count_label = isinstance(text, dict) and text.get("field") == "label"
            is_leader = isinstance(x, dict) and x.get("field") == "leader_x"
            data = value.get("data", {})
            if (is_count_label or is_leader) and isinstance(data, dict) and isinstance(data.get("values"), list):
                data["values"] = [row for row in data["values"] if not isinstance(row, dict) or row.get("vertex_count", 5) >= 5]
        for item in value.values():
            compact(item)

    compact(spec)
    children = spec.get("hconcat", [])
    if y_axis_extent is not None and children:
        pin_tree_y_axis(children[0], int(y_axis_extent))
    if len(children) == 2 and isinstance(children[1], dict):
        histogram = children[1]
        if isinstance(histogram.get("width"), (int, float)):
            histogram["width"] = max(1, round(histogram["width"] * 0.55))
        if not show_histogram_title:
            axis = histogram.get("encoding", {}).get("x", {}).get("axis")
            if isinstance(axis, dict):
                axis["title"] = None
    return spec


def _rendering(settings: dict[str, Any] | None) -> dict[str, Any]:
    """The shared rendering table merged with this figure's own table; ``{}`` if absent."""
    figures = (settings or {}).get("figures", {})
    return {**figures.get("rendering", {}), **figures.get("figures", {}).get(FIGURE_ID, {})}


def build_bootstrap_four_graphs_figure(
    panels: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> Any:
    """Compose four already identity-verified panel payloads in the declared order."""
    expected = {key for key, _ in PANEL_ORDER}
    if set(panels) != expected:
        raise ValueError("The bootstrap composite requires exactly the four graph variants.")
    charts = []
    y_axis_extent = int(_rendering(settings).get("tree_y_axis_extent", 60))
    for index, (key, _) in enumerate(PANEL_ORDER):
        saved = panels[key]
        peaks = None
        nominal = None
        if isinstance(saved, dict):
            saved = copy.deepcopy(saved)
            threshold = saved.pop("_composite_threshold")
            nominal = saved.pop("_composite_nominal_peaks", None)
            peaks = peak_count_from_spec(_nested_panel_spec(saved))
            compact = compact_panel_spec(
                _nested_panel_spec(saved),
                show_legend=index == 0,
                show_histogram_title=index >= 2,
                y_axis_extent=y_axis_extent,
            )
            panel = alt.Chart.from_dict(compact).copy(deep=True)
        else:
            payload, settings = saved
            threshold = settings["analysis"]["clamped"]["candidate_functional_threshold"]["value"]
            panel = build_bootstrap_tree_histogram_figure(payload, settings).copy(deep=True)
            try:
                peaks = peak_count_from_spec(_nested_panel_spec(panel.to_dict()))
            except Exception:  # count unavailable: the panel keeps its blank subtitle
                peaks = None
        # Vega-Lite permits a chart configuration only on the outermost composite.
        panel.config = alt.Undefined
        # The count sits on a subtitle: the longest one-line title would not fit the panel.
        title = alt.TitleParams(
            panel_title(key, float(threshold)), subtitle=panel_peaks_subtitle(peaks, nominal)
        )
        charts.append(panel.properties(title=title))
    # One 2x2 grid (two columns), not two separate rows: Vega-Lite aligns the cells of a
    # column, so each column's plot areas and axes line up whatever the panels' own
    # margins (the first panel carries the colour bar and key). Each panel keeps its own
    # scales and axes (the y-axis of every tree stays visible).
    chart = (
        alt.concat(*charts, columns=2, align="all", bounds="full")
        .resolve_scale(x="independent", y="independent", color="independent")
        .resolve_axis(x="independent", y="independent")
    )
    # One chart configuration (font, axis and legend sizes, no view frame), at the top
    # level only; Vega-Lite rejects it on a nested chart.
    return tree_histogram_chart_config(chart, _rendering(settings))
