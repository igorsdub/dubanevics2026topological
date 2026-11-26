"""Hexagonal binning of the density panels (pixel-space grid, regular hexagons)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from dubanevics2026topological.figures._hexbin import (
    HEXAGON_PATH,
    data_domain,
    hexagon_marks,
    hexbin_counts,
)

SQRT3 = math.sqrt(3.0)


def _cloud(n: int = 4000):
    rng = np.random.default_rng(3)
    return rng.normal(0.0, 1.0, n), rng.normal(5.0, 0.2, n)


def test_every_point_is_counted_exactly_once() -> None:
    xs, ys = _cloud()
    bins, radius = hexbin_counts(
        xs,
        ys,
        x_domain=data_domain(xs),
        y_domain=data_domain(ys),
        width=360,
        height=300,
        bins_across=40,
    )
    assert int(bins["n_genotypes"].sum()) == xs.size
    assert bins["n_genotypes"].min() >= 1
    assert radius == pytest.approx(360 / (40 * SQRT3))


def test_a_finer_grid_has_more_occupied_bins() -> None:
    xs, ys = _cloud()
    kwargs = {
        "x_domain": data_domain(xs),
        "y_domain": data_domain(ys),
        "width": 360,
        "height": 300,
    }
    coarse, _ = hexbin_counts(xs, ys, bins_across=10, **kwargs)
    fine, _ = hexbin_counts(xs, ys, bins_across=60, **kwargs)
    assert fine.height > coarse.height


def test_centres_form_a_regular_pointy_top_hexagonal_lattice_in_pixels() -> None:
    xs, ys = _cloud()
    x_domain, y_domain = data_domain(xs), data_domain(ys)
    width, height = 360.0, 300.0
    bins, radius = hexbin_counts(
        xs,
        ys,
        x_domain=x_domain,
        y_domain=y_domain,
        width=width,
        height=height,
        bins_across=30,
    )
    px = (bins["hx"].to_numpy() - x_domain[0]) / (x_domain[1] - x_domain[0]) * width
    py = (bins["hy"].to_numpy() - y_domain[0]) / (y_domain[1] - y_domain[0]) * height
    # Rows sit 1.5 radii apart; within a row centres are a whole 2 * 0.866 radii apart.
    rows = py / (1.5 * radius)
    assert np.allclose(rows, np.rint(rows))
    odd = np.rint(rows).astype(int) % 2 == 1
    columns = px / (SQRT3 * radius) - np.where(odd, 0.5, 0.0)
    assert np.allclose(columns, np.rint(columns))


def test_a_point_lies_within_one_radius_of_its_centre() -> None:
    xs = np.array([0.0, 1.0, 2.0, 0.5])
    ys = np.array([0.0, 2.0, 1.0, 1.5])
    x_domain, y_domain = (0.0, 2.0), (0.0, 2.0)
    bins, radius = hexbin_counts(
        xs, ys, x_domain=x_domain, y_domain=y_domain, width=100, height=100, bins_across=5
    )
    centres = np.column_stack([bins["hx"], bins["hy"]]) * 50.0
    for x, y in zip(xs * 50.0, ys * 50.0, strict=True):
        assert np.min(np.hypot(centres[:, 0] - x, centres[:, 1] - y)) <= radius + 1e-9


def test_a_constant_sample_gets_a_padded_domain() -> None:
    low, high = data_domain(np.array([2.0, 2.0]))
    assert low < 2.0 < high
    zero_low, zero_high = data_domain(np.array([0.0, 0.0]))
    assert (zero_low, zero_high) == (-1.0, 1.0)


def test_zero_bins_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        hexbin_counts(
            np.array([0.0]),
            np.array([0.0]),
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            width=10,
            height=10,
            bins_across=0,
        )


def test_hexagon_marks_are_filled_custom_paths_sized_to_the_cell() -> None:
    import altair as alt

    bins = pl.DataFrame({"hx": [0.1], "hy": [0.2], "n_genotypes": [3]})
    chart = hexagon_marks(
        bins,
        radius_px=10.0,
        x_field="x",
        y_field="y",
        x_scale=alt.Scale(domain=[0, 1]),
        y_scale=alt.Scale(domain=[0, 1]),
        x_title="x",
        y_title="y",
        color=alt.Color("n_genotypes:Q"),
    )
    mark = chart.to_dict()["mark"]
    assert mark["shape"] == HEXAGON_PATH and mark["filled"] is True
    assert mark["size"] == pytest.approx((2 * 10.0 * 1.03) ** 2)
