"""Shared publication style for branch vertex count colors and labels."""

from __future__ import annotations

import math
from typing import Any

import altair as alt
from topographer.plotting import ColorScaleConfig

from .palette import COUNT_BLACK, SYNONYMOUS_TEAL_DARK, SYNONYMOUS_TEAL_LIGHT

# In the "tips" label mode a terminal branch is labelled with its count when the
# count is at most this (the small branches, whose count the colour alone cannot
# tell apart); the ``branch_count_label_max`` key of ``config/figures.toml``
# overrides it.
DEFAULT_BRANCH_COUNT_LABEL_MAX = 9
# Vega scheme named beside the explicit black opacity range: if a renderer ever read
# the scheme instead of the range, the count would still be grey, never plasma.
COUNT_SCHEME = "greys"
# Which branches carry a count label: only the terminal tips of at most the maximum
# count ("tips"), or every branch ("all"). The ``branch_count_label_mode`` key sets it.
BRANCH_COUNT_LABEL_MODES = ("tips", "all")
DEFAULT_BRANCH_COUNT_LABEL_MODE = "tips"
# How a label that collides with another is moved: shifted vertically, clear of its
# neighbours, with a thin leader line back to its branch ("nudge"), or turned a
# quarter turn so that it takes up less width, with the same vertical nudge for
# any collision left ("rotate"). The ``branch_count_label_collision`` key sets it.
BRANCH_COUNT_LABEL_COLLISIONS = ("nudge", "rotate")
DEFAULT_BRANCH_COUNT_LABEL_COLLISION = "nudge"


def branch_count_domain(count_domain: tuple[float, float]) -> tuple[float, float]:
    """Return the log colour domain, from count one to the largest count."""
    maximum = max(1.0, float(count_domain[1]))
    return (1.0, maximum) if maximum > 1 else (1.0, 1.001)


def branch_count_colors(rendering: dict[str, Any]) -> tuple[str, str]:
    """Return the two ends of the count ramp: one colour, translucent to opaque.

    The ``count_color`` key sets the colour (black); ``count_alpha_min`` is the
    opacity at count one and ``count_alpha_max`` at the largest count.
    """
    hex_color = str(rendering.get("count_color", COUNT_BLACK)).lstrip("#")
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    low = float(rendering.get("count_alpha_min", 0.3))
    high = float(rendering.get("count_alpha_max", 1.0))
    return (f"rgba({red},{green},{blue},{low})", f"rgba({red},{green},{blue},{high})")


def branch_count_color_scale(
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    *,
    attribute: str = "vertex_count",
    show_legend: bool = False,
) -> ColorScaleConfig:
    """Build the shared log scale: one colour, more opaque as the count grows."""
    return ColorScaleConfig(
        attribute=attribute,
        title=rendering.get("branch_count_title", "Branch\\nvertex\\ncount").replace(
            "\\n", "\n"
        ),
        domain=branch_count_domain(count_domain),
        scheme=COUNT_SCHEME,
        scale_type="log",
        colors=branch_count_colors(rendering),
        show_legend=show_legend,
    )


def branch_count_color_encoding(
    rendering: dict[str, Any],
    count_domain: tuple[float, float],
    *,
    attribute: str = "vertex_count",
) -> Any:
    """Return the Altair colour encoding of ``branch_count_color_scale``, for marks
    Topographer does not draw (a cropped excerpt's own segments).

    It is the encoding Topographer builds from the same scale config for a tree's
    edges: the log opacity ramp over ``count_domain`` (``attribute`` is the count
    column). Using one config for both keeps an excerpt's branch colours identical
    to the full tree's.
    """
    scale = branch_count_color_scale(rendering, count_domain, attribute=attribute)
    return alt.Color(
        f"{attribute}:Q",
        scale=alt.Scale(
            scheme=scale.scheme,
            type=scale.scale_type,
            domain=list(scale.domain),
            range=list(scale.colors),
        ),
        legend=None,
    )


def branch_count_fraction(count: float, count_domain: tuple[float, float]) -> float:
    """Return where ``count`` sits on the log count scale: 0 at one, 1 at the largest count."""
    top = branch_count_domain(count_domain)[1]
    fraction = math.log(max(float(count), 1.0)) / math.log(top)
    return min(max(fraction, 0.0), 1.0)


def branch_count_opacity(
    count: float, rendering: dict[str, Any], count_domain: tuple[float, float]
) -> float:
    """Return the opacity a marker of branch ``count`` takes on the tree's count ramp.

    The same log ramp as ``branch_count_colors``: ``count_alpha_min`` at count one,
    ``count_alpha_max`` at the domain's largest count.
    """
    low = float(rendering.get("count_alpha_min", 0.3))
    high = float(rendering.get("count_alpha_max", 1.0))
    return low + (high - low) * branch_count_fraction(count, count_domain)


def branch_count_teal(
    count: float, rendering: dict[str, Any], count_domain: tuple[float, float]
) -> str:
    """Return the teal of a synonymous leaf branch of vertex ``count``, as ``#rrggbb``.

    The same log scale as the grey count ramp (``branch_count_fraction``): the light
    teal at count one, the dark teal at the largest count, so more vertices is darker.
    The ends are ``synonymous_teal_light`` and ``synonymous_teal_dark`` of the rendering.
    """
    light = str(rendering.get("synonymous_teal_light", SYNONYMOUS_TEAL_LIGHT)).lstrip("#")
    dark = str(rendering.get("synonymous_teal_dark", SYNONYMOUS_TEAL_DARK)).lstrip("#")
    fraction = branch_count_fraction(count, count_domain)
    channels = (
        round(int(light[i : i + 2], 16) * (1 - fraction) + int(dark[i : i + 2], 16) * fraction)
        for i in (0, 2, 4)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def branch_count_label_max(rendering: dict[str, Any]) -> int:
    """Return the largest branch vertex count a terminal tip is labelled at.

    Only the "tips" label mode reads it: the "all" mode labels every branch.
    """
    return int(rendering.get("branch_count_label_max", DEFAULT_BRANCH_COUNT_LABEL_MAX))


def _choice(
    rendering: dict[str, Any], key: str, allowed: tuple[str, ...], default: str
) -> str:
    value = str(rendering.get(key, default))
    if value not in allowed:
        raise ValueError(
            f"{key} must be one of {', '.join(map(repr, allowed))}, got {value!r}."
        )
    return value


def branch_count_label_mode(rendering: dict[str, Any]) -> str:
    """Return which branches are labelled: ``"tips"`` or ``"all"``."""
    return _choice(
        rendering,
        "branch_count_label_mode",
        BRANCH_COUNT_LABEL_MODES,
        DEFAULT_BRANCH_COUNT_LABEL_MODE,
    )


def branch_count_label_collision(rendering: dict[str, Any]) -> str:
    """Return how colliding labels are resolved: ``"nudge"`` or ``"rotate"``."""
    return _choice(
        rendering,
        "branch_count_label_collision",
        BRANCH_COUNT_LABEL_COLLISIONS,
        DEFAULT_BRANCH_COUNT_LABEL_COLLISION,
    )
