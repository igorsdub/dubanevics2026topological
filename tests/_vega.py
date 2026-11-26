"""Helpers that compile an Altair chart to Vega and locate its legends."""

from __future__ import annotations

from typing import Any

import vl_convert as vlc


def _title_text(legend: dict[str, Any]) -> str:
    title = legend.get("title")
    return title if isinstance(title, str) else ""


def vega_legend_groups(chart: Any) -> list[tuple[str, str, bool]]:
    """Return ``(group_name, legend_title, is_top_level)`` for every legend.

    ``is_top_level`` is true for a legend Vega hoisted to the whole figure (a
    shared scale) and false for one that lives inside a panel's group.
    """
    vega = vlc.vegalite_to_vega(chart.to_dict())
    found: list[tuple[str, str, bool]] = [
        ("root", _title_text(legend), True) for legend in vega.get("legends", [])
    ]

    def walk(marks: list[dict[str, Any]]) -> None:
        for mark in marks:
            if mark.get("type") != "group":
                continue
            found.extend(
                (mark.get("name", ""), _title_text(legend), False)
                for legend in mark.get("legends", [])
            )
            walk(mark.get("marks", []))

    walk(vega.get("marks", []))
    return found


def legend_titles(legends: list[tuple[str, str, bool]], *, top_level: bool = False):
    """Count legends by title, keeping the top-level ones or the in-group ones."""
    counts: dict[str, int] = {}
    for _, title, hoisted in legends:
        if hoisted is top_level:
            counts[title] = counts.get(title, 0) + 1
    return counts


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def outcome_key(node: Any) -> tuple[str | None, list[tuple[str, str, bool]]]:
    """Read the pixel-positioned outcome key in a tree panel's Vega-Lite spec.

    Returns ``(title, entries)`` where each entry is ``(label, colour, filled)``,
    ordered top to bottom; the title is the bold text mark, if any. Pixel-placed
    text with no marker beside it (the count colour bar's labels) is not a row.
    """
    placed = [
        item
        for item in _walk(node)
        if isinstance(item.get("mark"), dict) and "x" in item["mark"]
    ]
    points = {
        item["mark"]["y"]: item["mark"]
        for item in placed
        if item["mark"]["type"] == "point"
    }
    title = None
    entries = []
    for item in sorted(
        (item for item in placed if item["mark"]["type"] == "text"),
        key=lambda item: item["mark"]["y"],
    ):
        mark = item["mark"]
        if mark.get("fontWeight") == "bold":
            title = mark["text"]
        elif mark["y"] in points:
            point = points[mark["y"]]
            entries.append((mark["text"], point["color"], point["filled"]))
    return title, entries


def outcome_key_layout(node: Any) -> dict[str, float]:
    """Return where the outcome key's marks sit in the tree panel, in pixels.

    ``glyph_x`` and ``text_x`` are the left edges of the marker and label
    columns, ``first_y`` and ``last_y`` the vertical centres of the top and
    bottom rows (the title is the top row when there is one).
    """
    placed = [
        item["mark"]
        for item in _walk(node)
        if isinstance(item.get("mark"), dict) and "x" in item["mark"]
    ]
    glyphs = [mark for mark in placed if mark["type"] == "point"]
    labels = [
        mark
        for mark in placed
        if mark["type"] == "text" and mark.get("fontWeight") != "bold"
    ]
    return {
        "glyph_x": min(mark["x"] for mark in glyphs),
        "text_x": min(mark["x"] for mark in labels),
        "first_y": min(mark["y"] for mark in placed),
        "last_y": max(mark["y"] for mark in placed),
    }
