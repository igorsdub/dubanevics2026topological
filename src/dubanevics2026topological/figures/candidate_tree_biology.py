"""Shared row, outcome-key and legend helpers for the smoothed-tree biology figures.

The bootstrap biology figure (``candidate_tree_bootstrap_biology``) and the
figures that reuse its two-panel layout build their rows, outcome key and inset
legends from the helpers here. The module keeps the name of the figure it was
written for, ``papkou2023_candidate_tree_biology``, which is retired.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ._charts import (
    annotated_peaks,
    band_trapezoid_frame,
    band_trapezoid_layer,
    biology_axis_contract,
    compose_tree_layers,
    marker_config,
    overlay_count_colorbar,
    padded_scalar_domain,
    peak_marker_layers,
    persistence_legend_layers,
    persistence_panel_size,
    publication_scalar_scales,
    stop_tree_points_layer,
    study_biology_persistence_chart,
    study_biology_tree_chart,
    unannotated_peaks,
    wild_type_identities,
    wild_type_layers,
)
from .palette import (
    DARK_GREY,
    PEAK_GREY,
    TREE_WIDTH,
    tree_panel_size,
)


def biology_renderings(
    settings: dict[str, Any], figure_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return appearance, rendering, stop-marker and persistence-marker settings."""
    appearance = settings["figures"]["figures"][figure_id]
    rendering = {**settings["figures"].get("rendering", {}), **appearance}
    font_size = int(appearance.get("font_size", rendering.get("font_size", 15)))
    rendering = {
        **rendering,
        "font_size": font_size,
        "axis_label_font_size": font_size,
        "legend_font_size": font_size,
    }
    stop_rendering = rendering
    biology_rendering = {
        **rendering,
        "biology_persistence_point_size": int(
            appearance.get("persistence", {}).get("point_size", 70)
        ),
    }
    return appearance, rendering, stop_rendering, biology_rendering


def wrap_balanced(text: str, width: int) -> list[str]:
    """Wrap ``text`` to at most ``width`` characters with balanced lines.

    Greedy wrapping can leave a lone short word (an orphan such as ``0.108`` or
    ``(flat)``) on the last line. Keep the greedy line count and narrow the width
    to the smallest one that still gives that many lines, so the lines even out.
    Words are never split, not even at a hyphen.
    """

    def wrap_at(limit: int) -> list[str]:
        return textwrap.wrap(
            text, width=limit, break_long_words=False, break_on_hyphens=False
        )

    lines = wrap_at(width)
    longest_word = max(map(len, text.split()), default=0)
    for limit in range(longest_word, width):
        balanced = wrap_at(limit)
        if len(balanced) == len(lines):
            return balanced
    return lines


def wrap_phrases(phrases: list[str], width: int) -> list[str]:
    """Wrap each phrase on its own, so a line break never runs two phrases together."""
    return [line for phrase in phrases for line in wrap_balanced(phrase, width)]


@dataclass(frozen=True, slots=True)
class OutcomeKey:
    """The outcome key of a tree panel: ``(label, colour, filled)`` entries and a title.

    Each entry repeats the persistence marker of its outcome: a filled triangle in
    the outcome's colour. The third element is always true (no hollow markers) and
    keeps the entry shape. The title is optional.
    """

    entries: Sequence[tuple[str, str, bool]]
    title: str | None = None


@dataclass(slots=True)
class BiologyRow:
    """One row of a stacked biology figure: a tree beside its persistence plot."""

    data: Any
    pairs: pl.DataFrame
    wild_type_source: Any
    stop_points: pl.DataFrame
    title: str
    subtitle: Any
    # Layers drawn on the branches (highlighted or absorbed branches): below the
    # count labels, the stop-codon crosses and the wild-type star, whatever the row.
    tree_overlays: list[Any] = field(default_factory=list)
    # Outcome key of the tree panel. ``assemble_biology_chart`` draws it at the
    # corner of the panel that holds the least of the tree (``outcome_legend_corner``).
    outcome_key: OutcomeKey | None = None
    persistence_markers: list[Any] = field(default_factory=list)
    # Persistence threshold: shades the strip of pairs with persistence <= band.
    band: float | None = None
    # In-diagram key (peak, stop codon, wild type). The bootstrap biology
    # figure draws it on the nominal row only; the band-simplified row leaves it out.
    diagram_legend: bool = True
    # Every peak carries an outcome marker, so the base layer of grey triangles is
    # drawn invisible instead of showing through the outcome markers.
    # The markers also draw the stop-codon and wild-type peaks in their annotation
    # colour (``peak_marker_layers``); a row without them gets those peaks drawn
    # here, over a base layer of the plain peaks only.
    outcome_markers: bool = False
    # ``(min, max)`` of the tree's ``vertex_count`` (``data`` carries the column, as
    # ``plot_data_with_metrics`` leaves it): the branches are then coloured by
    # branch vertex count on the shared count scale instead of light grey, and the
    # horizontal count colour bar sits inside the tree panel.
    count_domain: tuple[float, float] | None = None
    # Count-label layers (``_count_labels.branch_count_labels``), drawn over the
    # branches and their overlays and under the stop-codon crosses and the star.
    tree_labels: list[Any] = field(default_factory=list)


def assemble_biology_chart(
    candidate: Any,
    appearance: dict[str, Any],
    rendering: dict[str, Any],
    *,
    stop_rendering: dict[str, Any],
    biology_rendering: dict[str, Any],
    rows: Sequence[BiologyRow],
) -> Any:
    """Stack the rows, share the scalar axes and apply the figure config.

    Any number of rows (at least one) is stacked top to bottom; the persistence
    x-axis title is drawn on the last row only.
    """
    if not rows:
        raise ValueError("A biology chart needs at least one row.")
    charts = []
    for row_index, spec in enumerate(rows):
        tree = study_biology_tree_chart(
            spec.data, candidate, appearance, rendering, count_domain=spec.count_domain
        )
        panel_width = tree_panel_size(rendering, appearance)[0]
        # The row hands its layers over by role; the stacking order (branch overlays,
        # count labels, stop crosses, key, star) is ``compose_tree_layers``'s.
        tree = compose_tree_layers(
            tree,
            overlays=spec.tree_overlays,
            count_labels=spec.tree_labels,
            stop_markers=[stop_tree_points_layer(spec.stop_points, stop_rendering)],
            keys=outcome_key_overlay(spec, candidate, appearance, rendering),
            wild_type=wild_type_layers(spec.wild_type_source, rendering),
        ).resolve_scale(color="independent")
        tree = biology_axis_contract(tree, y_title="Fitness (at birth, WT = 0)")
        if spec.count_domain is not None:
            tree = overlay_count_colorbar(
                tree, appearance, rendering, spec.count_domain, width=panel_width
            )
        persistence_appearance = {
            **appearance,
            "persistence": {
                **appearance.get("persistence", {}),
                "height": tree_panel_size(rendering, appearance)[1],
            },
        }
        base_rendering = (
            {**rendering, "biology_persistence_opacity": 0.0}
            if spec.outcome_markers
            else rendering
        )
        wild_type = wild_type_identities(spec.wild_type_source)
        # Annotated peaks are drawn in their annotation colour, not over black.
        persistence = study_biology_persistence_chart(
            unannotated_peaks(spec.pairs, wild_type=wild_type),
            candidate,
            persistence_appearance,
            base_rendering,
        )
        markers = (
            spec.persistence_markers
            if spec.outcome_markers
            else peak_marker_layers(
                annotated_peaks(spec.pairs, wild_type=wild_type),
                biology_rendering,
                color=PEAK_GREY,
                annotation="Peak",
                wild_type=wild_type,
            )
        )
        band_layers = (
            persistence_band_layers(spec.band, candidate.scalar_domain, rendering)
            if spec.band is not None
            else []
        )
        # The key sits inside the diagram, bottom right, one entry per line.
        # A row with ``diagram_legend`` false (the band-simplified row of
        # ``candidate_tree_bootstrap_biology``) draws none of it.
        legend_layers = (
            persistence_legend_layers(
                *persistence_panel_size(persistence_appearance, rendering),
                rendering,
                wild_type=bool(wild_type & set(spec.pairs["birth_label"].to_list())),
                stop_codon=(
                    "birth_has_stop_codon" in spec.pairs.columns
                    and bool(spec.pairs["birth_has_stop_codon"].any())
                ),
            )
            if spec.diagram_legend
            else []
        )
        persistence = alt.layer(
            *band_layers, persistence, *markers, *legend_layers
        ).resolve_scale(color="independent")
        persistence = biology_axis_contract(
            persistence,
            y_title=None,
            y_tick_labels=bool(rendering.get("persistence_y_tick_labels", False)),
            x_title=rendering.get("persistence_x_title", "Fitness at death")
            if row_index == len(rows) - 1
            else None,
        )
        row = alt.hconcat(
            tree, persistence, spacing=int(rendering.get("panel_gap", 10))
        ).resolve_scale(x="independent", y="shared", color="independent")
        charts.append(
            row.properties(
                title=alt.TitleParams(
                    text=spec.title, subtitle=spec.subtitle, anchor="start"
                )
            )
        )
    chart = alt.vconcat(*charts, spacing=int(rendering.get("comparison_panel_gap", 16)))
    chart = chart.resolve_scale(x="independent", y="independent", color="independent")
    chart = publication_scalar_scales(chart, candidate.scalar_domain, rendering)
    return (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(rendering.get("font_size", 15)),
            titleFontSize=int(rendering.get("font_size", 15)),
        )
        .configure_legend(
            labelFontSize=int(rendering.get("legend_font_size", 15)),
            titleFontSize=int(rendering.get("legend_font_size", 15)),
        )
        .configure_title(
            fontSize=int(
                rendering.get("heading_font_size", rendering.get("font_size", 15))
            ),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


def persistence_band_layers(
    threshold: float, scalar_domain: tuple[float, float], rendering: dict[str, Any]
) -> list[Any]:
    """Shade the strip of persistence <= ``threshold`` above the diagonal.

    A pair plots at (death, birth) with birth above death, so the strip is the
    trapezoid between ``birth = death`` and ``birth = death + threshold``, clipped
    to the diagonal's extent: the unpadded ``scalar_domain``, which Topographer's
    ``plot_persistence`` draws the diagonal across (the axes extend past it by the
    publication padding). It is a translucent grey with no edge line. A rotated
    label sits below the diagonal, where no pair can be.
    """
    low, high = float(scalar_domain[0]), float(scalar_domain[1])
    frame = band_trapezoid_frame(low, high, threshold).rename(
        {"along": "death_scalar", "edge": "birth_scalar"}
    )
    x = alt.X("death_scalar:Q", axis=None)
    y = alt.Y("birth_scalar:Q", axis=None)
    layers = [band_trapezoid_layer(frame, x, y, rendering)]
    label = rendering.get("band_label")
    if label:
        span = high - low
        anchor = low + float(rendering.get("band_label_position", 0.62)) * span
        text = pl.DataFrame(
            {
                "death_scalar": [anchor],
                "birth_scalar": [
                    anchor - float(rendering.get("band_label_offset", 0.07)) * span
                ],
                "label": [str(label)],
            }
        )
        layers.append(
            alt.Chart(text)
            .mark_text(
                angle=315,
                align="center",
                baseline="middle",
                color=str(rendering.get("band_label_color", DARK_GREY)),
                fontSize=int(rendering.get("band_label_font_size", 12)),
            )
            .encode(x=x, y=y, text="label:N")
        )
    return layers


INSET_CORNERS = ("top-right", "bottom-right", "top-left", "bottom-left")


def segment_sample_points(
    segments: pl.DataFrame,
    x_domain: Sequence[float] | None = None,
    y_domain: Sequence[float] | None = None,
    step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points along line segments, proportionally to their drawn length.

    Returns ``(xs, ys, segment_index)`` in the coordinates the segments are
    plotted in. A segment gets one sample per ``step`` of axes length (the
    length measured as a fraction of the given domains, so a long branch is
    sampled densely and a short one sparsely), and always at least its two ends.
    Without domains the segments' own extent is the unit.
    """
    if segments.is_empty():
        return np.empty(0), np.empty(0), np.empty(0, dtype=int)
    x_start, x_end = (segments[name].to_numpy() for name in ("x_start", "x_end"))
    y_start, y_end = (segments[name].to_numpy() for name in ("y_start", "y_end"))
    if x_domain is None:
        x_domain = (min(x_start.min(), x_end.min()), max(x_start.max(), x_end.max()))
    if y_domain is None:
        y_domain = (min(y_start.min(), y_end.min()), max(y_start.max(), y_end.max()))
    x_span = max(float(x_domain[1]) - float(x_domain[0]), np.finfo(float).tiny)
    y_span = max(float(y_domain[1]) - float(y_domain[0]), np.finfo(float).tiny)
    length = np.hypot((x_end - x_start) / x_span, (y_end - y_start) / y_span)
    counts = np.maximum(2, np.ceil(length / step).astype(int) + 1)
    xs, ys, owners = [], [], []
    for index, count in enumerate(counts):
        fractions = np.linspace(0.0, 1.0, int(count))
        xs.append(x_start[index] + fractions * (x_end[index] - x_start[index]))
        ys.append(y_start[index] + fractions * (y_end[index] - y_start[index]))
        owners.append(np.full(int(count), index))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(owners)


def _nice_bounds(low: float, high: float, count: int = 10) -> tuple[float, float]:
    """Round a domain outward to the step Vega's ``nice`` (d3) would pick."""
    if high <= low:
        return low, high
    raw = (high - low) / count
    power = np.floor(np.log10(raw))
    error = raw / 10**power
    factor = (
        10
        if error >= 50**0.5
        else 5
        if error >= 10**0.5
        else 2
        if error >= 2**0.5
        else 1
    )
    step = factor * 10**power
    return float(np.floor(low / step) * step), float(np.ceil(high / step) * step)


def tree_x_domain(
    segments: pl.DataFrame, padding: float = 0.0, width: float | None = None
) -> tuple[float, float]:
    """Approximate the domain of the tree's horizontal scale.

    Topographer gives the x scale no explicit domain, so Vega-Lite's default
    applies: the data extent including zero. Topographer's ``horizontal_padding``
    (``padding`` pixels on each side of a ``width`` pixel axis) widens that
    domain about its centre first, as Vega's ``padding`` does, and ``nice``
    (about ten ticks) then rounds the result outward. Without ``padding`` and
    ``width`` the extent is only rounded.
    """
    if segments.is_empty():
        return 0.0, 1.0
    xs = np.concatenate([segments["x_start"].to_numpy(), segments["x_end"].to_numpy()])
    low, high = min(float(xs.min()), 0.0), max(float(xs.max()), 0.0)
    if padding > 0 and width:
        pad = min(float(padding), float(width) / 4)
        factor = width / (width - 2 * pad)
        centre = (low + high) / 2
        low, high = centre + (low - centre) * factor, centre + (high - centre) * factor
    return _nice_bounds(low, high)


def _has_extent(
    xs: np.ndarray, x_domain: Sequence[float], y_domain: Sequence[float]
) -> bool:
    """Whether there are data and both domains have a positive span."""
    return bool(
        xs.size > 0
        and float(x_domain[1]) > float(x_domain[0])
        and float(y_domain[1]) > float(y_domain[0])
    )


def corner_counts(
    xs: np.ndarray,
    ys: np.ndarray,
    x_domain: Sequence[float],
    y_domain: Sequence[float],
    rendering: dict[str, Any],
    corners: Sequence[str] = INSET_CORNERS,
    *,
    owners: np.ndarray | None = None,
) -> dict[str, int]:
    """Count the data in each corner's box of the plot area.

    The data are plotted points (or points sampled along lines) in axis
    coordinates. Each corner is a box of ``legend_box_fraction`` (width, height)
    of the axes. With ``owners`` (the segment each sample belongs to) a corner is
    scored by the distinct segments that cross its box, otherwise by raw points.
    """
    width_fraction, height_fraction = (
        float(value) for value in rendering.get("legend_box_fraction", (0.42, 0.3))
    )
    x_low, x_high = float(x_domain[0]), float(x_domain[1])
    y_low, y_high = float(y_domain[0]), float(y_domain[1])
    u = (np.asarray(xs, dtype=float) - x_low) / (x_high - x_low)
    v = (np.asarray(ys, dtype=float) - y_low) / (y_high - y_low)
    counts = {}
    for corner in corners:
        vertical, horizontal = corner.split("-")
        in_x = u >= 1 - width_fraction if horizontal == "right" else u <= width_fraction
        in_y = v >= 1 - height_fraction if vertical == "top" else v <= height_fraction
        inside = in_x & in_y
        counts[corner] = (
            int(np.count_nonzero(inside))
            if owners is None
            else int(np.unique(np.asarray(owners)[inside]).size)
        )
    return counts


def emptiest_corner(
    xs: np.ndarray,
    ys: np.ndarray,
    x_domain: Sequence[float],
    y_domain: Sequence[float],
    rendering: dict[str, Any],
    corners: Sequence[str] = INSET_CORNERS,
    *,
    owners: np.ndarray | None = None,
) -> str:
    """Pick the corner among ``corners`` that the data leave emptiest.

    Ties go to the order of ``corners``. Without data (or with a degenerate
    domain) the first corner wins. See ``corner_counts`` for how a corner is
    scored.
    """
    if not _has_extent(xs, x_domain, y_domain):
        return corners[0]
    counts = corner_counts(
        xs, ys, x_domain, y_domain, rendering, corners, owners=owners
    )
    return min(corners, key=lambda corner: counts[corner])


def choose_legend_orient(
    xs: np.ndarray,
    ys: np.ndarray,
    x_domain: Sequence[float],
    y_domain: Sequence[float],
    rendering: dict[str, Any],
    *,
    owners: np.ndarray | None = None,
    total: int | None = None,
) -> str:
    """Pick the corner of the plot area that the data leave emptiest.

    The emptiest of ``INSET_CORNERS`` wins (see ``emptiest_corner``). If even
    that box holds more than ``legend_collision_fraction`` of the data the
    legend would sit on branches, so the figure falls back to
    ``legend_fallback_orient`` (a strip below the tree by default).

    With ``owners`` (the segment each sample belongs to) and ``total`` (the
    number of segments), a corner is scored by the distinct segments that cross
    its box and the collision limit is a fraction of ``total``; otherwise by
    raw points and a fraction of their number.
    """
    best = emptiest_corner(xs, ys, x_domain, y_domain, rendering, owners=owners)
    if not _has_extent(xs, x_domain, y_domain):
        return best
    counts = corner_counts(xs, ys, x_domain, y_domain, rendering, owners=owners)
    population = xs.size if owners is None or total is None else total
    limit = float(rendering.get("legend_collision_fraction", 0.02)) * population
    if counts[best] > limit:
        return str(rendering.get("legend_fallback_orient", "bottom"))
    return best


def resolve_legend_orient(
    rendering: dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    x_domain: Sequence[float],
    y_domain: Sequence[float],
) -> str:
    """Return ``legend_orient``, choosing the emptiest corner when it is ``auto``."""
    orient = str(rendering.get("legend_orient", "auto"))
    if orient != "auto":
        return orient
    return choose_legend_orient(xs, ys, x_domain, y_domain, rendering)


def inset_legend(
    title: str | None,
    rendering: dict[str, Any],
    *,
    orient: str | None = None,
    **properties: Any,
) -> alt.Legend:
    """Terminal-branch legend: inside the plot area at a corner, with no box.

    Vega draws the four corner orients over the plot area. They carry no fill
    and no stroke, so the branches behind stay visible. Any other orient (for
    example the ``bottom`` fallback strip) keeps the usual outside placement.
    A ``title`` of ``None`` draws no title; ``properties`` are further Vega
    legend properties (label font size, symbol opacity) passed through.
    """
    orient = orient or str(rendering.get("legend_orient", "right"))
    extra: dict[str, Any] = {}
    if orient in INSET_CORNERS:
        extra["offset"] = float(rendering.get("legend_inset_offset", 6))
    options: dict[str, Any] = {
        "titleLimit": float(rendering.get("legend_title_limit", 320)),
        "labelLimit": float(rendering.get("legend_label_limit", 240)),
        **extra,
        **properties,
    }
    return alt.Legend(title=title, orient=orient, **options)


def outcome_legend_entries(
    order: Sequence[str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    counts: Mapping[str, int],
) -> list[tuple[str, str, bool]]:
    """Return the key entries ``(label, colour, filled)`` of the outcome states in the data.

    The shared encoding (palette ``OUTCOME_COLORS``): kept
    is filled mid-grey, absorbed filled black, elder-protected filled orange.
    Only states with at least one branch in ``counts`` are listed, in ``order``,
    so a key never explains a mark the figure does not draw.
    """
    return [
        (labels[name], colors[name], True)
        for name in order
        if counts.get(name, 0) > 0
    ]


# Advance widths of Helvetica (and so Arial, which the figures set) in thousandths
# of an em, for the ASCII characters from space to tilde, then the few others the
# keys use; any other character counts as a digit's width.
_ASCII_WIDTHS = (
    "278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278"
    " 556 556 556 556 556 556 556 556 556 556 278 278 584 584 584 556"
    " 1015 667 667 722 722 667 611 778 722 278 500 667 556 833 722 778"
    " 667 778 722 667 611 722 667 944 667 667 611 278 278 278 469 556"
    " 333 556 556 500 556 556 278 556 556 222 222 500 222 833 556 556"
    " 556 556 333 500 278 556 500 722 500 500 500 334 260 334 584"
)
_TEXT_WIDTHS = {
    **{
        chr(32 + index): int(width) for index, width in enumerate(_ASCII_WIDTHS.split())
    },
    "α": 580,
    "τ": 500,
    "≤": 549,
    "≥": 549,
    "×": 584,
    "−": 584,
    "·": 278,
}
_DEFAULT_TEXT_WIDTH = 556
# Bold runs a little wider than regular text; the factor errs on the wide side.
_BOLD_TEXT_FACTOR = 1.06


def estimate_text_width(text: str, font_size: float, *, bold: bool = False) -> float:
    """Estimate the pixel width of ``text`` set in Arial at ``font_size`` pixels.

    The figures render in Arial (Helvetica metrics), so summing each character's
    advance width is accurate to about a pixel; ``bold`` widens it for the key
    title. It lets a key be flush against a panel edge without measuring text.
    """
    em = sum(_TEXT_WIDTHS.get(char, _DEFAULT_TEXT_WIDTH) for char in text) / 1000
    return em * font_size * (_BOLD_TEXT_FACTOR if bold else 1.0)


@dataclass(frozen=True, slots=True)
class OutcomeKeyMetrics:
    """Pixel geometry of an outcome key: one marker and label per row.

    ``text_width`` is the longest label's estimated width (``estimate_text_width``).
    """

    font_size: int
    inset: float
    line_height: float
    glyph_gap: float
    marker_half: float
    text_width: float
    rows: int

    @property
    def width(self) -> float:
        """Pixels from the marker's near edge to the end of the longest label."""
        return self.marker_half + self.glyph_gap + self.text_width

    @property
    def height(self) -> float:
        """Pixels from the first row's top to the last row's bottom."""
        return self.rows * self.line_height


def outcome_key_metrics(
    key: OutcomeKey, rendering: dict[str, Any]
) -> OutcomeKeyMetrics:
    """Return a key's pixel geometry from the ``outcome_legend_*`` settings."""
    font_size = int(rendering.get("legend_font_size", 15))
    marker_size = int(rendering.get("outcome_legend_marker_size", 70))
    widths = [estimate_text_width(label, font_size) for label, _, _ in key.entries]
    if key.title:
        widths.append(estimate_text_width(key.title, font_size, bold=True))
    return OutcomeKeyMetrics(
        font_size=font_size,
        inset=float(rendering.get("outcome_legend_inset", 8)),
        line_height=float(rendering.get("outcome_legend_line_height", font_size + 6)),
        glyph_gap=float(rendering.get("outcome_legend_glyph_gap", 14)),
        marker_half=math.sqrt(marker_size) / 2,
        text_width=max(widths, default=0.0),
        rows=len(key.entries) + (1 if key.title else 0),
    )


def outcome_legend_layer(
    entries: Sequence[tuple[str, str, bool]],
    rendering: dict[str, Any],
    *,
    width: float,
    height: float,
    title: str | None = None,
    corner: str = "top-right",
):
    """Draw an outcome key inside a tree panel, in the given corner.

    ``entries`` are ``(label, colour, filled)`` triples: always a filled triangle
    (``filled`` is true; no hollow markers), matching the persistence markers. The key is pixel-positioned marks (no Vega legend),
    ``width`` and ``height`` being the panel's; ``corner`` is one of
    ``INSET_CORNERS`` (``outcome_legend_corner`` picks the emptiest).
    ``outcome_legend_inset``, ``outcome_legend_line_height`` and
    ``outcome_legend_glyph_gap`` set its geometry.
    """
    if corner not in INSET_CORNERS:
        raise ValueError(f"Unknown outcome key corner {corner!r}.")
    if not entries:
        return alt.layer()
    metrics = outcome_key_metrics(OutcomeKey(entries, title), rendering)
    vertical, horizontal = corner.split("-")
    if horizontal == "right":
        text_left = width - metrics.inset - metrics.text_width
        glyph_x = text_left - metrics.glyph_gap
    else:
        glyph_x = metrics.inset + metrics.marker_half
        text_left = glyph_x + metrics.glyph_gap
    top = (
        metrics.inset if vertical == "top" else height - metrics.inset - metrics.height
    )
    layers = []
    row = 0
    if title:
        layers.append(
            alt.Chart(alt.Data(values=[{"v": title}])).mark_text(
                x=glyph_x,
                y=top + metrics.line_height * (row + 0.5),
                text=title,
                align="left",
                baseline="middle",
                fontSize=metrics.font_size,
                fontWeight="bold",
            )
        )
        row += 1
    for label, color, filled in entries:
        y = top + metrics.line_height * (row + 0.5)
        layers.append(
            alt.Chart(alt.Data(values=[{"v": label}])).mark_point(
                x=glyph_x,
                y=y,
                shape="triangle-up",
                filled=True,
                color=color,
                size=int(rendering.get("outcome_legend_marker_size", 70)),
                strokeWidth=float(rendering.get("open_marker_stroke_width", 1.6)),
                opacity=1.0,
            )
        )
        layers.append(
            alt.Chart(alt.Data(values=[{"v": label}])).mark_text(
                x=text_left,
                y=y,
                text=label,
                align="left",
                baseline="middle",
                fontSize=metrics.font_size,
            )
        )
        row += 1
    return alt.layer(*layers)


_INK_COLUMNS = ("x_start", "x_end", "y_start", "y_end")


def _mark_runs(
    points: pl.DataFrame, half_height: float = 0.0, half_width: float = 0.0
) -> pl.DataFrame:
    """Return a point's extent as ink segments: a run at each edge and the middle.

    ``points`` has columns ``x`` and ``y``; the extent is ``half_width`` either
    side horizontally and ``half_height`` either side vertically. Without an
    extent each point is one zero-length segment.
    """
    x = points["x"].cast(pl.Float64)
    y = points["y"].cast(pl.Float64)
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "x_start": x - half_width,
                    "x_end": x + half_width,
                    "y_start": y + offset,
                    "y_end": y + offset,
                }
            )
            for offset in sorted({-half_height, 0.0, half_height})
        ]
    )


def tree_ink(
    segments: pl.DataFrame,
    stop_points: pl.DataFrame,
    wild_type_source: Any,
    rendering: dict[str, Any],
    *,
    x_per_pixel: float,
    y_per_pixel: float,
) -> pl.DataFrame:
    """Return everything a tree panel draws as segments, in plotted coordinates.

    The branches, each stop-codon cross and the wild-type star with its extent
    (a marker of ``size`` square pixels is about ``sqrt(size)`` pixels across), and
    the wild-type label (the run to the star's left that ``wild_type_layers``
    draws, with the height of a line of text). The columns are ``x_start``,
    ``x_end``, ``y_start`` and ``y_end``, ready for ``segment_sample_points``; the
    ``*_per_pixel`` arguments convert a pixel extent to those units.
    """
    font_size = int(rendering.get("font_size", 15))

    def runs(points: pl.DataFrame, half_x: float, half_y: float) -> pl.DataFrame:
        return _mark_runs(points, half_y * y_per_pixel, half_x * x_per_pixel)

    stop = marker_config(rendering, "stop_codon", default_shape="x")
    stop_half = math.sqrt(int(rendering.get("stop_codon_tree_size", stop.size))) / 2
    parts = [
        visible_segments(segments).select(_INK_COLUMNS).cast(pl.Float64),
        runs(stop_points, stop_half, stop_half),
    ]
    wild_type = getattr(wild_type_source, "wild_type_marker_table", None)
    if wild_type is not None and not wild_type.is_empty():
        marker = marker_config(rendering, "wild_type", default_shape="star")
        star_half = math.sqrt(marker.size) / 2
        parts.append(runs(wild_type, star_half, star_half))
        if "marker_label" in wild_type.columns:
            for row in wild_type.iter_rows(named=True):
                half_width = (
                    estimate_text_width(str(row["marker_label"]), font_size) / 2
                )
                # The label is right-aligned at the star's x plus ``label_dx``.
                centre = pl.DataFrame(
                    {
                        "x": [
                            float(row["x"])
                            + (marker.label_dx - half_width) * x_per_pixel
                        ],
                        "y": [float(row["y"])],
                    }
                )
                parts.append(runs(centre, half_width, font_size / 2))
    return pl.concat(parts)


def _key_corner_query(
    key: OutcomeKey,
    rendering: dict[str, Any],
    *,
    size: tuple[float, float],
    scalar_domain: tuple[float, float],
    segments: pl.DataFrame,
    stop_points: pl.DataFrame,
    wild_type_source: Any,
) -> tuple[Any, ...]:
    """Return the arguments ``corner_counts`` takes to score a key's corner boxes.

    ``(xs, ys, x_domain, y_domain, owners, boxes)``: the tree's ink sampled in
    plotted coordinates, the panel's domains and the ``rendering`` of each of two
    boxes. The key's own box is its pixel extent (``outcome_key_metrics``) and its
    inset from the corner, as a fraction of the ``size`` (width, height) panel; the
    padded box adds ``outcome_legend_clearance`` pixels on the sides away from
    the corner.
    """
    width, height = float(size[0]), float(size[1])
    metrics = outcome_key_metrics(key, rendering)
    clearance = float(rendering.get("outcome_legend_clearance", 6))
    x_domain = _tree_axes_x_domain(segments, rendering, width)
    y_domain = padded_scalar_domain(scalar_domain, rendering)
    ink = tree_ink(
        segments,
        stop_points,
        wild_type_source,
        rendering,
        x_per_pixel=(x_domain[1] - x_domain[0]) / width,
        y_per_pixel=(y_domain[1] - y_domain[0]) / height,
    )
    xs, ys, owners = segment_sample_points(
        ink,
        x_domain,
        y_domain,
        step=float(rendering.get("legend_sample_step", 0.01)),
    )
    boxes = tuple(
        {
            **rendering,
            "legend_box_fraction": (
                min(1.0, (metrics.inset + metrics.width + margin) / width),
                min(1.0, (metrics.inset + metrics.height + margin) / height),
            ),
        }
        for margin in (0.0, clearance)
    )
    return xs, ys, x_domain, y_domain, owners, boxes


def outcome_corner_counts(
    key: OutcomeKey, rendering: dict[str, Any], **tree: Any
) -> dict[str, tuple[int, int]]:
    """Score each corner as ``(overlapping, nearby)`` marks (see ``outcome_legend_corner``)."""
    xs, ys, x_domain, y_domain, owners, boxes = _key_corner_query(
        key, rendering, **tree
    )
    strict, padded = (
        corner_counts(xs, ys, x_domain, y_domain, box, INSET_CORNERS, owners=owners)
        for box in boxes
    )
    return {corner: (strict[corner], padded[corner]) for corner in INSET_CORNERS}


def outcome_legend_corner(
    key: OutcomeKey,
    rendering: dict[str, Any],
    *,
    size: tuple[float, float],
    scalar_domain: tuple[float, float],
    segments: pl.DataFrame,
    stop_points: pl.DataFrame,
    wild_type_source: Any = None,
) -> str:
    """Pick the corner of a tree panel where the outcome key covers the least ink.

    All four corners are eligible. A corner is scored by ``corner_counts`` over the
    key's own box (see ``_key_corner_query``): the distinct branches, stop crosses
    and wild-type marks (star, label) that reach it. The corner with the fewest
    wins, so a corner crossed by anything loses to one that is not and, when every
    corner is crossed, the one with the fewest wins. The count over the box padded by
    ``outcome_legend_clearance`` pixels breaks a tie, then ``top-right`` does (the
    order of ``INSET_CORNERS``).
    """
    forced = rendering.get("outcome_legend_corner")
    if forced:
        if forced not in INSET_CORNERS:
            raise ValueError(f"Unknown outcome_legend_corner {forced!r}.")
        return forced
    scores = outcome_corner_counts(
        key,
        rendering,
        size=size,
        scalar_domain=scalar_domain,
        segments=segments,
        stop_points=stop_points,
        wild_type_source=wild_type_source,
    )
    return min(INSET_CORNERS, key=lambda corner: scores[corner])


def outcome_key_overlay(
    spec: BiologyRow,
    candidate: Any,
    appearance: dict[str, Any],
    rendering: dict[str, Any],
) -> list[Any]:
    """Return the layer of a row's outcome key at its emptiest corner, or none."""
    key = spec.outcome_key
    if key is None or not key.entries:
        return []
    width, height = tree_panel_size(rendering, appearance)
    corner = outcome_legend_corner(
        key,
        rendering,
        size=(width, height),
        scalar_domain=candidate.scalar_domain,
        segments=spec.data.edge_segment_table,
        stop_points=spec.stop_points,
        wild_type_source=spec.wild_type_source,
    )
    return [
        outcome_legend_layer(
            key.entries,
            rendering,
            width=width,
            height=height,
            title=key.title,
            corner=corner,
        )
    ]


def _tree_axes_x_domain(
    segments: pl.DataFrame, rendering: dict[str, Any], tree_width: float | None = None
) -> tuple[float, float]:
    """Return the x domain that spans the tree panel's width, in data units.

    The tree scale (see ``tree_x_domain``) with Topographer's pixel padding, so a
    pixel offset in the panel converts to data units by the domain's span over
    the ``tree_width``.
    """
    width = float(tree_width or rendering.get("tree_width", TREE_WIDTH))
    padding = float(rendering.get("tree_horizontal_padding", 18))
    return tree_x_domain(segments, padding, width)


def tree_legend_orient(
    segments: pl.DataFrame,
    scalar_domain: Sequence[float],
    rendering: dict[str, Any],
    *,
    tree_width: float | None = None,
) -> str:
    """Resolve the legend orient from a tree's segments.

    The corner counts are of distinct branches crossing each corner box, not of
    samples, so one long branch cannot outweigh many short ones. The x domain is
    the tree scale's (see ``tree_x_domain``) with Topographer's
    ``horizontal_padding`` pixels on each side of the ``tree_width`` axes; the y
    domain is the padded scalar domain every scalar axis uses.
    """
    visible = visible_segments(segments)
    y_domain = padded_scalar_domain(scalar_domain, rendering)
    if visible.is_empty():
        return resolve_legend_orient(
            rendering, np.empty(0), np.empty(0), (0, 1), y_domain
        )
    x_domain = _tree_axes_x_domain(segments, rendering, tree_width)
    orient = str(rendering.get("legend_orient", "auto"))
    if orient != "auto":
        return orient
    xs, ys, owners = segment_sample_points(
        visible,
        x_domain,
        y_domain,
        step=float(rendering.get("legend_sample_step", 0.01)),
    )
    return choose_legend_orient(
        xs, ys, x_domain, y_domain, rendering, owners=owners, total=visible.height
    )


def visible_segments(segments: pl.DataFrame) -> pl.DataFrame:
    """Drop zero-length segments, which draw nothing."""
    return segments.filter(
        (pl.col("x_start") != pl.col("x_end")) | (pl.col("y_start") != pl.col("y_end"))
    )
