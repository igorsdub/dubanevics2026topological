"""Fitness against its standard error, one point per genotype.

Supplementary. One square panel: the pooled standard error ``se_pooled`` (x)
against our fitted fitness ``s`` (y) for every genotype of the growth-rate table,
a fine hexagonal density with the Spearman rank correlation printed on the panel.
The sparsest bins are small filled light-grey dots. The colour bar stands inside
the panel. Panel size, fonts and axis styling come from the ``peak_uncertainty``
table of ``config/figures.toml`` (overridable in this figure's own table). It reads
the same cached inputs as ``papkou_agreement`` and refits nothing.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from dubanevics2026topological.figures._count_labels import (
    count_label_text,
    fixed_label_text,
)
from dubanevics2026topological.figures.palette import LIGHT_GREY
from dubanevics2026topological.figures.papkou_agreement import apply_density_style, density_panel

FIGURE_ID = "fitness_se_correlation"
_HEXBIN_MIN = 2500
_HEXBIN_BINS = 70
# The figure whose panel size, fonts and axis styling this panel shares.
STYLE_SOURCE = "peak_uncertainty"


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length finite samples."""
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Correlation inputs must be equal-length 1-d samples.")
    if left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Correlation needs at least two finite values.")
    rho = float(spearmanr(left, right).statistic)
    if not np.isfinite(rho):
        raise ValueError("Correlation is undefined for a constant sample.")
    return rho


def fitness_and_se(growth_rates: pl.DataFrame) -> pl.DataFrame:
    """Genotype, fitness and pooled SE, finite rows only."""
    missing = sorted({"dna_sequence", "s", "se_pooled"} - set(growth_rates.columns))
    if missing:
        raise ValueError(f"Growth-rate table is missing columns: {missing}.")
    return growth_rates.select(
        pl.col("dna_sequence").cast(pl.String),
        pl.col("s").cast(pl.Float64).alias("fitness"),
        pl.col("se_pooled").cast(pl.Float64).alias("se"),
    ).filter(pl.col("fitness").is_finite() & pl.col("se").is_finite())


def build_fitness_se_correlation_figure(
    payload: dict[str, Any], settings: dict[str, Any]
) -> alt.Chart:
    """Draw the fitness-against-SE panel with its Spearman rho."""
    figure_tables = settings["figures"]["figures"]
    # Size and style follow peak_uncertainty; this figure's own table wins.
    appearance = {
        **settings["figures"].get("rendering", {}),
        **figure_tables.get(STYLE_SOURCE, {}),
        **figure_tables[FIGURE_ID],
    }
    frame = fitness_and_se(payload["growth_rates"])
    if frame.height < 2:
        raise ValueError("Fewer than two genotypes have a finite fitness and SE.")
    rho = spearman_rho(frame["se"].to_numpy(), frame["fitness"].to_numpy())
    font_size = int(appearance.get("font_size", 15))
    panel_size = int(appearance.get("panel_size", 300))
    panel = density_panel(
        frame,
        x="se",
        y="fitness",
        x_title=str(appearance.get("standard_error_title", "Pooled standard error")),
        y_title=str(appearance.get("fitness_axis_title", "Fitness s")),
        panel_title=str(appearance.get("title", "Fitness against standard error")),
        subtitle=(
            f"Spearman ρ = {fixed_label_text(rho, 3)}, "
            f"n = {count_label_text(frame.height)} genotypes"
        ),
        width=panel_size,
        height=panel_size,
        point_size=float(appearance.get("point_size", 12)),
        hexbin_min=int(appearance.get("hexbin_min", _HEXBIN_MIN)),
        hexbin_bins=int(appearance.get("hexbin_bins", _HEXBIN_BINS)),
        color=str(appearance.get("point_color", LIGHT_GREY)),
        identity_line=False,
        legend_orient=str(appearance.get("colorbar_orient", "top-right")),
    )
    return apply_density_style(panel, appearance)
