"""Class proportions against the Benjamini-Yekutieli FDR target.

Supplementary S2. Two panels, Papkou threshold and Topographer threshold.
Each panel shows the saved scan's Functional, Uncertain and Non-functional
proportions at q = 0, 0.01, ..., 1, one stacked bar per q.
``functional_counts_vs_q`` is against the Benjamini-Yekutieli target, with the
proportion axis clipped at 0.1 like the other two;
``functional_counts_vs_q_bh`` and ``functional_counts_vs_q_bonferroni`` draw the
same stacked chart against the Benjamini-Hochberg and Bonferroni q, with the
proportion axis (of all genotypes) clipped at 0.1. The counts and fractions are
the ``fdr_q_scan`` table; nothing here recomputes a p-value, an adjusted q-value
or a class. The heat-map versions were removed (2026-11).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from dubanevics2026topological.threshold.classification_probability_scans import (
    get_active_threshold_definitions,
    uniform_fdr_q_grid,
)

from ._count_labels import fixed_label_text
from .labels import PAPKOU_THRESHOLD, TOPOGRAPHER_THRESHOLD
from .palette import ELDER_PROTECTED_ORANGE, SYNONYMOUS_GREEN

FIGURE_ID = "functional_counts_vs_q"
BH_ID = "functional_counts_vs_q_bh"
BONFERRONI_ID = "functional_counts_vs_q_bonferroni"
FIGURE_IDS = frozenset({FIGURE_ID, BH_ID, BONFERRONI_ID})
# The clipped top of the proportion axis of the BH and Bonferroni figures.
Y_MAX = 0.1
METHOD_LABELS = {
    "by": "Benjamini-Yekutieli",
    "bh": "Benjamini-Hochberg",
    "bonferroni": "Bonferroni",
}
# The six-decimal Topographer cutoff the median-reflected rule applies.
CANDIDATE_THRESHOLD = -0.253159
# Cutoff names written by the scan, in panel order.
CUTOFFS = (
    ("papkou_original", PAPKOU_THRESHOLD),
    ("median_reflected", TOPOGRAPHER_THRESHOLD),
)
CLASSES = ("Functional", "Uncertain", "Non-functional")
_COUNT = {
    "Functional": "functional_count",
    "Uncertain": "uncertain_count",
    "Non-functional": "non_functional_count",
}
_FRACTION = {
    "Functional": "functional_fraction",
    "Uncertain": "uncertain_fraction",
    "Non-functional": "non_functional_fraction",
}
_DEFAULT_COLORS = {
    "functional_color": SYNONYMOUS_GREEN,
    "uncertain_color": ELDER_PROTECTED_ORANGE,
    "non_functional_color": "#7570b3",
}


def candidate_threshold() -> float:
    """Applied median-reflected threshold from the scan's threshold definitions.

    Stops when that value, read to six decimal places, is not -0.253159.
    """
    reflected = next(
        item
        for item in get_active_threshold_definitions()
        if item.name == "median_reflected"
    )
    applied = float(reflected.applied_threshold)
    if round(applied, 6) != CANDIDATE_THRESHOLD:
        raise ValueError(
            f"The {TOPOGRAPHER_THRESHOLD} is "
            f"{applied}, not {CANDIDATE_THRESHOLD} to six decimal places."
        )
    return applied


def _uniform_targets() -> list[float]:
    return [float(q) for q in uniform_fdr_q_grid()]


def by_scan(table: pl.DataFrame, method: str = "by") -> pl.DataFrame:
    """One method's rows on the uniform q grid, both thresholds.

    ``table`` is the scan result as saved. Rows for any other method, and
    targets off the 0.01 grid, are dropped. Both thresholds must be present.
    Every hundredth from 0 to 1 must be present. Within a threshold the saved
    functional count must not fall as q rises (the uncertain count must not
    rise). The Candidate panel's saved applied threshold must be the
    six-decimal median-reflected value. The counts are not recomputed.
    """
    required = {
        "cutoff",
        "method",
        "fdr_target",
        "functional_count",
        "uncertain_count",
        "non_functional_count",
        "total_count",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            "The FDR scan is missing columns: " + ", ".join(sorted(missing)) + "."
        )
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown correction {method!r}.")
    label = METHOD_LABELS[method]
    by = table.filter(pl.col("method") == method)
    if by.is_empty():
        raise ValueError(f"The FDR scan has no {label} rows.")
    targets = _uniform_targets()
    frames = []
    for name, _title in CUTOFFS:
        panel = by.filter(pl.col("cutoff") == name).sort("fdr_target")
        if panel.is_empty():
            raise ValueError(
                f"The FDR scan has no {label} rows for {name}."
            )
        if name == "median_reflected" and "applied_threshold" in panel.columns:
            saved = {float(value) for value in panel["applied_threshold"].to_list()}
            if any(round(value, 6) != CANDIDATE_THRESHOLD for value in saved):
                raise ValueError(
                    f"The saved {TOPOGRAPHER_THRESHOLD} "
                    f"{sorted(saved)} is not {CANDIDATE_THRESHOLD} to six decimal places."
                )
        kept_rows = []
        seen: list[float] = []
        for record in panel.iter_rows(named=True):
            q = float(record["fdr_target"])
            if any(np.isclose(q, target, rtol=0, atol=1e-10) for target in targets):
                record = dict(record)
                record["fdr_target"] = float(
                    targets[int(np.argmin([abs(q - target) for target in targets]))]
                )
                kept_rows.append(record)
                seen.append(record["fdr_target"])
        if seen != targets:
            missing_q = [q for q in targets if q not in set(seen)]
            raise ValueError(
                f"{name} is missing {label} targets at the 0.01 grid: "
                + ", ".join(fixed_label_text(q, 2) for q in missing_q[:8])
                + "."
            )
        panel = pl.DataFrame(kept_rows)
        q_values = panel["fdr_target"].to_list()
        if any(right <= left for left, right in pairwise(q_values)):
            raise ValueError(f"{name} FDR targets must be strictly increasing.")
        functional = panel["functional_count"].to_list()
        uncertain = panel["uncertain_count"].to_list()
        if any(right < left for left, right in pairwise(functional)):
            raise ValueError(f"{name} functional counts must not fall as q rises.")
        if any(right > left for left, right in pairwise(uncertain)):
            raise ValueError(f"{name} uncertain counts must not rise as q rises.")
        total = panel["total_count"].to_list()
        non_functional = panel["non_functional_count"].to_list()
        for f_count, u_count, nf_count, n in zip(
            functional, uncertain, non_functional, total, strict=True
        ):
            if f_count + u_count + nf_count != n:
                raise ValueError(
                    f"{name} class counts do not partition the scan total."
                )
        frames.append(panel)
    return pl.concat(frames)


def _proportion(record: dict[str, Any], label: str) -> float:
    """Saved fraction when the scan stored it, otherwise count / total."""
    column = _FRACTION[label]
    count = int(record[_COUNT[label]])
    total = int(record["total_count"])
    if total <= 0:
        raise ValueError("The FDR scan total must be positive.")
    computed = count / total
    if column in record and record[column] is not None:
        saved = float(record[column])
        if abs(saved - computed) > 1e-9:
            raise ValueError(
                f"Saved {column} {saved} does not match {count}/{total}."
            )
        return saved
    return computed


def long_proportions(scan: pl.DataFrame, method: str = "by") -> pl.DataFrame:
    """One row per threshold, uniform q step and class, with the saved proportion."""
    candidate_threshold()
    rows: list[dict[str, Any]] = []
    titles = dict(CUTOFFS)
    for record in by_scan(scan, method).iter_rows(named=True):
        for label in CLASSES:
            rows.append(
                {
                    "cutoff": record["cutoff"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "panel": titles[record["cutoff"]],
                    "fdr_target": record["fdr_target"],
                    "class": label,
                    "proportion": _proportion(record, label),
                    "stack_rank": CLASSES.index(label),
                }
            )
    frame = pl.DataFrame(rows)
    for panel in frame["panel"].unique().to_list():
        for q in frame.filter(pl.col("panel") == panel)["fdr_target"].unique().to_list():
            total = float(
                frame.filter((pl.col("panel") == panel) & (pl.col("fdr_target") == q))[
                    "proportion"
                ].sum()
            )
            if abs(total - 1.0) > 1e-8:
                raise ValueError(
                    f"{panel} proportions at q={q} sum to {total}, not 1."
                )
    return frame


def _appearance(settings: dict[str, Any], figure_id: str) -> dict[str, Any]:
    return {
        **settings["figures"].get("rendering", {}),
        **settings["figures"]["figures"][figure_id],
    }


def _colors(appearance: dict[str, Any]) -> list[str]:
    return [
        str(appearance.get(key, default))
        for key, default in _DEFAULT_COLORS.items()
    ]


def _heading(appearance: dict[str, Any], fallback_title: str) -> Any:
    from ._title_flag import figure_heading

    return figure_heading(
        str(appearance.get("title", fallback_title)),
        [str(appearance.get("subtitle", "Benjamini-Yekutieli, q from 0 to 1"))],
    )


def _finish(chart: alt.Chart, appearance: dict[str, Any]) -> alt.Chart:
    font_size = int(appearance.get("font_size", 12))
    return (
        chart.configure(font=appearance.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size, grid=False)
        .configure_legend(labelFontSize=font_size, symbolType="square")
        .configure_title(
            fontSize=int(appearance.get("heading_font_size", 16)),
            subtitleFontSize=int(appearance.get("subtitle_font_size", 12)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )


def build_functional_counts_vs_q_figure(
    payload: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Stacked bars against the Benjamini-Yekutieli q, one panel per threshold, y clipped."""
    return _method_stacked_figure(
        payload, settings, figure_id=FIGURE_ID, method="by"
    )


def _method_stacked_figure(
    payload: dict[str, Any],
    settings: dict[str, Any],
    *,
    figure_id: str,
    method: str,
) -> Any:
    """Stacked bars of one correction, one panel per threshold, y clipped.

    The same chart as ``functional_counts_vs_q`` for Benjamini-Yekutieli, drawn for
    ``method`` (``bh`` or ``bonferroni``). The y axis is the proportion of all
    genotypes and ends at ``y_max`` (0.1): a stack taller than that is cut at the
    top edge (``clip``), so the Functional and Uncertain bands are readable and
    the Non-functional band is not drawn in full.
    """
    appearance = _appearance(settings, figure_id)
    label = METHOD_LABELS[method]
    frame = long_proportions(payload["scan"], method)
    width = int(appearance.get("panel_width", 420))
    height = int(appearance.get("panel_height", 280))
    y_max = float(appearance.get("y_max", Y_MAX))
    colors = _colors(appearance)
    panels = []
    for _name, title in CUTOFFS:
        panel_frame = frame.filter(pl.col("panel") == title)
        bars = (
            alt.Chart(panel_frame)
            .mark_bar(size=float(appearance.get("bar_size", 2)), clip=True)
            .encode(
                x=alt.X(
                    "fdr_target:Q",
                    title=f"{label} FDR target q",
                    scale=alt.Scale(domain=[0.0, 1.0], nice=False, zero=True),
                    axis=alt.Axis(values=[0.0, 0.25, 0.5, 0.75, 1.0]),
                ),
                y=alt.Y(
                    "proportion:Q",
                    title="Proportion of all genotypes",
                    scale=alt.Scale(domain=[0.0, y_max], nice=False, zero=True),
                    stack="zero",
                ),
                color=alt.Color(
                    "class:N",
                    title=None,
                    scale=alt.Scale(domain=list(CLASSES), range=colors),
                    sort=list(CLASSES),
                    legend=alt.Legend(orient="bottom", direction="horizontal"),
                ),
                order=alt.Order("stack_rank:Q", sort="ascending"),
            )
        )
        panels.append(
            bars.properties(
                width=width,
                height=height,
                title=alt.TitleParams(text=title, anchor="start"),
            )
        )
    chart = alt.hconcat(
        *panels, spacing=int(appearance.get("panel_gap", 24))
    ).resolve_scale(y="shared", color="shared")
    heading = _heading(
        appearance,
        f"Functional, uncertain and non-functional genotypes against the {label} q",
    )
    if heading is not None:
        chart = chart.properties(title=alt.TitleParams(**heading))
    return _finish(chart, appearance)


def build_bh_counts_figure(payload: dict[str, Any], settings: dict[str, Any]) -> Any:
    """Stacked class proportions against the Benjamini-Hochberg q, y clipped at 0.1."""
    return _method_stacked_figure(
        payload, settings, figure_id=BH_ID, method="bh"
    )


def build_bonferroni_counts_figure(
    payload: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Stacked class proportions against the Bonferroni q, y clipped at 0.1."""
    return _method_stacked_figure(
        payload, settings, figure_id=BONFERRONI_ID, method="bonferroni"
    )
