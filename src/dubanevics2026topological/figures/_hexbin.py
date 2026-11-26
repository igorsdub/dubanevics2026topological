"""Hexagonal binning for density panels (Altair has no 2D density transform).

The bins are computed here, in pixel space, so a hexagon is regular on the page
whatever the two axes' data ranges are. Each occupied bin is one centre and a
count; the panel draws one hexagon mark per bin (a custom SVG path on
``mark_point``, the same device ``_charts`` uses for its triangles).
"""

from __future__ import annotations

import math

import altair as alt
import numpy as np
import polars as pl

# Pointy-top regular hexagon with circumradius 1: Vega scales a custom path so its
# [-1, 1] box is ``sqrt(size)`` px wide, as for the built-in square.
HEXAGON_PATH = "M0,-1L0.866,-0.5L0.866,0.5L0,1L-0.866,0.5L-0.866,-0.5Z"
_SQRT3 = math.sqrt(3.0)


def data_domain(values: np.ndarray, pad_fraction: float = 0.02) -> tuple[float, float]:
    """The data extent of ``values`` with a small pad; a constant sample gets a unit pad."""
    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        pad = 1.0 if high == 0.0 else abs(high) * 0.05
        return low - pad, high + pad
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def hexbin_counts(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    width: float,
    height: float,
    bins_across: int,
) -> tuple[pl.DataFrame, float]:
    """Count points per hexagon on a ``bins_across``-wide pointy-top grid.

    Returns ``(bins, radius_px)``. ``bins`` has one row per occupied hexagon with
    columns ``hx`` and ``hy`` (the centre, in data units) and ``n_genotypes``;
    ``radius_px`` is the circumradius in pixels. The grid is laid on the panel's
    pixel box, so the hexagons are regular on the page.
    """
    if bins_across < 1:
        raise ValueError("bins_across must be at least 1.")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x_low, x_high = x_domain
    y_low, y_high = y_domain
    x_span = x_high - x_low
    y_span = y_high - y_low
    radius = float(width) / (bins_across * _SQRT3)
    px = (x - x_low) / x_span * float(width)
    py = (y - y_low) / y_span * float(height)
    # Axial coordinates of the containing hexagon, rounded in cube space.
    q_frac = (_SQRT3 / 3.0 * px - py / 3.0) / radius
    r_frac = (2.0 / 3.0 * py) / radius
    s_frac = -q_frac - r_frac
    q = np.rint(q_frac)
    r = np.rint(r_frac)
    s = np.rint(s_frac)
    dq = np.abs(q - q_frac)
    dr = np.abs(r - r_frac)
    ds = np.abs(s - s_frac)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    q = np.where(fix_q, -r - s, q)
    r = np.where(fix_r, -q - s, r)
    counts = (
        pl.DataFrame({"q": q.astype(np.int64), "r": r.astype(np.int64)})
        .group_by("q", "r")
        .len()
        .rename({"len": "n_genotypes"})
    )
    cx = radius * _SQRT3 * (counts["q"].to_numpy() + counts["r"].to_numpy() / 2.0)
    cy = radius * 1.5 * counts["r"].to_numpy()
    bins = counts.select("n_genotypes").with_columns(
        pl.Series("hx", cx / float(width) * x_span + x_low),
        pl.Series("hy", cy / float(height) * y_span + y_low),
    )
    return bins, radius


def hexagon_marks(
    bins: pl.DataFrame,
    *,
    radius_px: float,
    x_field: str,
    y_field: str,
    x_scale: alt.Scale,
    y_scale: alt.Scale,
    x_title: str,
    y_title: str,
    color: alt.Color,
) -> alt.Chart:
    """One filled hexagon per bin; ``bins`` carries ``hx``, ``hy`` and ``n_genotypes``.

    ``x_field`` and ``y_field`` name the columns the hexagon centres are renamed to,
    so a layered line or point sharing the axes uses the same field names.
    """
    # Drawn slightly larger than the cell so antialiasing leaves no seam.
    size = (2.0 * radius_px * 1.03) ** 2
    frame = bins.rename({"hx": x_field, "hy": y_field})
    return (
        alt.Chart(frame)
        .mark_point(shape=HEXAGON_PATH, size=size, filled=True, strokeWidth=0)
        .encode(
            x=alt.X(f"{x_field}:Q", scale=x_scale, title=x_title),
            y=alt.Y(f"{y_field}:Q", scale=y_scale, title=y_title),
            color=color,
        )
    )
