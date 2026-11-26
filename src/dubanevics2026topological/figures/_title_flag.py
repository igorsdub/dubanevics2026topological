"""Publication versus titled figure headings.

`titled=True` keeps the figure title and subtitle (today's pixels). `titled=False`
drops that figure-level heading only; panel and row titles are content and stay. Axis titles, legends,
colour-bar titles and in-plot labels stay in both variants.
"""

from __future__ import annotations


def figure_heading(
    text: str,
    subtitle: list[str] | None = None,
    *,
    titled: bool = True,
) -> dict[str, object] | None:
    """Return an Altair title mapping, or None for the publication variant."""
    if not titled:
        return None
    heading: dict[str, object] = {"text": text, "anchor": "start"}
    if subtitle:
        heading["subtitle"] = list(subtitle)
    return heading


def _is_heading(title: object) -> bool:
    """Whether a chart-level title is a heading (top) rather than a bottom, left or right note."""
    orient = getattr(title, "orient", None)
    if orient is None:
        orient = title.get("orient") if isinstance(title, dict) else None
    return orient in (None, "top") or type(orient).__name__ == "UndefinedType"


def without_headings(chart: object) -> tuple[object, int] | None:
    """Return a copy of an Altair chart without its figure-level heading, and how many went (0 or 1).

    The one place that turns today's titled chart into the untitled variant: it clears
    only the ROOT chart's ``title`` (title and subtitle) when it is a top heading. Panel,
    row and layer titles are content (sequence counts, thresholds, row names) and stay,
    as do axis titles, legends, colour-bar titles, in-plot labels and bottom, left and
    right notes. A figure whose real heading sits on a child is untitled through
    ``figure_heading(..., titled=False)`` in its builder instead.
    It never raises: ``None`` means the chart could not be handled, and the caller then
    keeps today's chart and writes no twin.
    """
    import copy

    import altair as alt

    try:
        copied = copy.deepcopy(chart)
        title = getattr(copied, "title", alt.Undefined)
        if title is not alt.Undefined and title is not None and _is_heading(title):
            copied.title = alt.Undefined
            return copied, 1
        return copied, 0
    except Exception:
        return None
