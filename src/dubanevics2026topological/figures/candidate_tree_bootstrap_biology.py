"""Candidate biology figure with the band-simplified tree (ADR 0008).

The main bootstrap figure has two rows, tree beside ordinary H0 persistence
diagram. Row 1 is the nominal tree; every peak is marked by whether its
persistence clears the bootstrapped persistence threshold (survives: filled
mid-grey) or not (absorbed: filled black, ADR 0012) on the diagram, and the tree
draws only the absorbed terminal branches in black (all else light grey), with a
key at the emptiest corner of its panel. The simultaneous confidence band
(parametric bootstrap) is the strip of the diagram where persistence is at most
the threshold. Row 2 is the band-simplified tree
(every nominal peak that survives, nothing else), its branches coloured by
branch vertex count (absorbed members folded in) on the grey scale with the
horizontal count colour bar inside its panel; every branch carries its count
(``branch_count_label_mode``; ``_count_labels.py``). Bootstrap outcome
stays on the diagram markers. The core
rule (step 1) has no figure of its own; the band-simplified tree is checked to
nest inside the core-rule view.

Tree and diagram of a row share the fitness axis: a peak's branch top and its
diagram point (birth = peak fitness) sit at the same height.

Every number a title, subtitle or key prints is the manuscript quantity registry's
display text (``analysis["manuscript_quantities"]``); the figure counts peaks only
to decide what to draw and to check its tables against the summary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import altair as alt
import polars as pl

from ..threshold.functional_threshold import DECLARATION_KEY, FDR_FIXED_CUTOFF
from ._charts import (
    count_colorbar_box,
    count_domain,
    empty_rule_layer,
    padded_scalar_domain,
    peak_marker_layers,
    persistence_with_counts,
    plot_data_with_metrics,
    plot_data_with_saved_metrics,
    selected_view_as_tree,
    stop_codon_nodes,
    wild_type_identities,
)
from ._count_labels import branch_count_labels
from .candidate_tree_biology import (
    BiologyRow,
    OutcomeKey,
    assemble_biology_chart,
    biology_renderings,
    outcome_legend_entries,
    visible_segments,
)
from .palette import OUTCOME_COLORS, SURVIVES_COLOR, tree_panel_size

FIGURE_ID = "candidate_tree_bootstrap_biology"
# The core-rule outcomes a hill of the checkpoint can have (the global-stability
# figure checks the checkpoint's hills against them) and the band outcomes drawn
# here.
OUTCOME_ORDER = ("absorbed", "elder_protected", "kept")
BAND_ORDER = ("survives", "absorbed")


# Fallback colour of an absorbed peak; ``figures.toml`` sets it with
# ``absorbed_color``, and a surviving peak's with ``survives_color``.
DEFAULT_OUTCOME_COLORS = OUTCOME_COLORS
DEFAULT_SURVIVES_COLOR = SURVIVES_COLOR


# Row titles, subtitles and legend labels; ``figures.toml`` overrides them (the
# ``str.format`` placeholders are the keyword arguments named below, all display
# text of the registry).
DEFAULT_NOMINAL_ROW_TITLE = (
    "Nominal tree · {peaks} peaks · {survive} survive and {absorbed} are "
    "absorbed (persistence ≤ threshold) · {stops} stop-codon containing"
)
DEFAULT_NOMINAL_ROW_SUBTITLE = (
    "{dataset} · {draws} bootstrap draws · seed {seed}",
    (
        "Bootstrapped persistence threshold {threshold} at band quantile "
        "{band_quantile} · {percent}% simultaneous confidence band "
        "(parametric bootstrap)"
    ),
    "Stop codon containing peaks: crimson × on the tree, crimson × inside the triangle on the diagram",
)
DEFAULT_BAND_ROW_TITLE = (
    "Band-simplified tree · {peaks} peaks, all with persistence > threshold · "
    "{absorbed} of {input_peaks} nominal peaks absorbed"
)
DEFAULT_BAND_ROW_SUBTITLE = (
    (
        "{dataset} · {draws} bootstrap draws · seed {seed} · bootstrapped "
        "persistence threshold {threshold} at band quantile {band_quantile}"
    ),
    "{retained} sequences retained · {saddles} saddles removed · raw fitness unchanged",
)
DEFAULT_LEGEND_TITLE = "Peak"
DEFAULT_LEGEND_LABELS = {
    "survives": "survives: persistence > threshold ({count})",
    "absorbed": "absorbed: persistence ≤ threshold ({count})",
}


def format_lines(template: Any, **values: Any) -> str | list[str]:
    """Fill a title template, or each line of a list of templates."""
    if isinstance(template, str):
        return template.format(**values)
    return [line.format(**values) for line in template]


def band_colors(rendering: dict[str, Any]) -> dict[str, str]:
    """Return the colour of a peak that survives the threshold and of one that is absorbed."""
    return {
        "survives": str(rendering.get("survives_color", DEFAULT_SURVIVES_COLOR)),
        "absorbed": str(
            rendering.get("absorbed_color", DEFAULT_OUTCOME_COLORS["absorbed"])
        ),
    }


def legend_labels(
    rendering: dict[str, Any], counts: Mapping[str, str]
) -> dict[str, str]:
    """Return each band outcome's legend label with its peak count (display text) filled in."""
    templates = {
        **DEFAULT_LEGEND_LABELS,
        **dict(rendering.get("legend_labels", {})),
    }
    return {name: templates[name].format(count=counts[name]) for name in counts}


def dataset_label(rendering: dict[str, Any]) -> str:
    """Name the dataset in row subtitles."""
    return str(rendering.get("dataset_label", "Papkou et al. (2023) · candidate graph"))


def peak_branch_segments(
    plot_data: Any, category_by_peak: Mapping[str, str]
) -> pl.DataFrame:
    """Return the drawn segments of each peak's terminal branch with a category.

    A peak is a leaf of the tree, so its terminal branch is the segment set that
    touches the peak's node; the node table maps the node index to the peak's
    ``dna_sequence`` (the identity, never the node index, is exported).
    """
    nodes = plot_data.node_table.select("tree_node_index", "dna_sequence")
    index_to_peak = {
        int(row["tree_node_index"]): str(row["dna_sequence"])
        for row in nodes.iter_rows(named=True)
        if row["dna_sequence"] in category_by_peak
    }
    rows = []
    for row in plot_data.edge_segment_table.iter_rows(named=True):
        for end in ("source_tree_node_index", "target_tree_node_index"):
            peak = index_to_peak.get(int(row[end]))
            if peak is not None:
                rows.append({**row, "peak_sequence": peak})
                break
    segments = pl.DataFrame(rows, schema_overrides={"peak_sequence": pl.String})
    if segments.is_empty():
        return segments
    return segments.with_columns(
        pl.col("peak_sequence")
        .replace_strict(dict(category_by_peak), return_dtype=pl.String)
        .alias("category")
    )


def peak_branch_layer(
    segments: pl.DataFrame,
    draw: tuple[str, ...],
    colors: Mapping[str, str],
    rendering: dict[str, Any],
    *,
    keep_fitness_axis: bool = False,
):
    """Draw the terminal branches of the ``draw`` categories in their colour.

    The layer states ``axis=None`` on its fitness (y) channel, which Vega-Lite merges with the
    tree's own axis into no axis at all; ``keep_fitness_axis`` leaves the channel's axis
    undefined so the tree keeps its fitness axis (the frameless slides need it).

    Every other branch keeps the tree's light grey: the outcome key is the
    separate outcome key (``OutcomeKey``), so no colour legend is drawn here.
    """
    visible = visible_segments(segments)
    if visible.is_empty():
        return empty_rule_layer()
    layers = []
    for category in draw:
        chosen = visible.filter(pl.col("category") == category)
        if chosen.is_empty():
            continue
        layers.append(
            alt.Chart(chosen)
            .mark_rule(
                color=colors[category],
                strokeWidth=float(rendering.get("edge_width", 1.1)),
                opacity=float(rendering.get("synonymous_opacity", 1.0)),
            )
            .encode(
                x=alt.X("x_start:Q", axis=None),
                x2="x_end:Q",
                y=alt.Y("y_start:Q")
                if keep_fitness_axis
                else alt.Y("y_start:Q", axis=None),
                y2="y_end:Q",
                detail="logical_edge_index:N",
                tooltip=["peak_sequence:N", "category:N"],
            )
        )
    return alt.layer(*layers) if layers else empty_rule_layer()


def persistence_markers(
    pairs: pl.DataFrame,
    category_by_peak: Mapping[str, str],
    order: tuple[str, ...],
    colors: Mapping[str, str],
    annotations: Mapping[str, str],
    biology_rendering: dict[str, Any],
    wild_type: frozenset[str] = frozenset(),
) -> list[Any]:
    """Return the diagram-marker layers of every category, in ``order``.

    A category's peaks are filled in the category's colour (ADR 0012: mid-grey
    survives, black absorbed), except stop-codon and wild-type peaks, which take
    their annotation's colour.
    """
    return [
        layer
        for category in order
        for layer in peak_marker_layers(
            pairs.filter(
                pl.col("birth_label").is_in(
                    [
                        peak
                        for peak, name in category_by_peak.items()
                        if name == category
                    ]
                )
            ),
            biology_rendering,
            color=colors[category],
            annotation=annotations[category],
            wild_type=wild_type,
        )
    ]


def stale(reason: str) -> RuntimeError:
    """Return the error for a summary that disagrees with its tables."""
    return RuntimeError(
        f"stale bootstrap-smoothed summary: {reason}; "
        "rerun `just bootstrap-smoothed-tree`."
    )


def view_stop_points(view: Any) -> pl.DataFrame:
    """Return the maxima of a derived view that contain a stop codon."""
    return view.plot_data.node_table.filter(
        (pl.col("event") == "maximum") & pl.col("has_stop_codon")
    )


def build_candidate_tree_bootstrap_biology_figure(
    analysis: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Colour peaks by survival of the bootstrapped threshold over the nominal and band-simplified tree."""
    appearance, rendering, stop_rendering, biology_rendering = biology_renderings(
        settings, FIGURE_ID
    )
    quantities = analysis["manuscript_quantities"]
    candidate = analysis["landscapes"]["candidate_clamped"]
    cached = analysis["bootstrap_smoothed"]
    summary = cached.summary
    # Row 1 reads this in-memory tree (float64 fitness, same identities as the
    # bootstrap checkpoint); the cached comparison tree is float32-rounded.
    tree_analysis = analysis["bootstrap_tree"]
    band_view = analysis["bootstrap_band_view"]
    band_summary = summary["band_simplification"]
    hills = cached.hills
    input_peaks = hills.height
    survivors = set(hills.filter(pl.col("clears_band"))["peak_sequence"])
    absorbed = int(input_peaks - len(survivors))
    if (
        len(survivors) != band_view.pairs.height
        or len(survivors) != int(band_summary["peaks_after"])
        or input_peaks != int(band_summary["peaks_before"])
        or set(band_view.pairs["birth_label"]) != survivors
    ):
        raise stale("its band simplification disagrees with the band view")

    # Geometry only: the cone's edge. The printed threshold is the registry's.
    tau = float(band_summary["two_c"])

    evidence = analysis["candidate_annotations"]
    data = plot_data_with_saved_metrics(
        tree_analysis, analysis["tables"]["candidate_clamped_branch_table"]
    )
    pairs = evidence.pairs.rename(
        {"birth_sequence": "birth_label", "death_sequence": "death_label"}
    )
    nominal_counts = count_domain(data.edge_table)
    pairs = persistence_with_counts(pairs, data, nominal_counts)
    biology_rendering = {**biology_rendering, "biology_count_domain": nominal_counts}
    if set(pairs["birth_label"]) != set(hills["peak_sequence"]):
        raise stale("its hill table lists other peaks than the nominal tree")
    stop_points = stop_codon_nodes(evidence.critical_points, rendering)

    colors = band_colors(rendering)
    legend_title = str(rendering.get("legend_title", DEFAULT_LEGEND_TITLE))
    annotations = {
        "survives": "Persistence above the threshold (survives)",
        "absorbed": "Persistence at most the threshold (absorbed)",
    }
    nominal_category = {
        row["peak_sequence"]: "survives" if row["clears_band"] else "absorbed"
        for row in hills.select("peak_sequence", "clears_band").iter_rows(named=True)
    }
    nominal_counts = {"survives": len(survivors), "absorbed": absorbed}
    band_category = {peak: "survives" for peak in survivors}
    shown_survivors = quantities.display(
        "bootstrap.candidate.after_band_rule.peak_count"
    )
    shown_absorbed = quantities.display("bootstrap.candidate.total_absorbed_count")
    shown_nominal = quantities.display("bootstrap.candidate.nominal.peak_count")
    common = {
        "dataset": dataset_label(rendering),
        "draws": quantities.display("bootstrap.draws"),
        "seed": quantities.display("bootstrap.seed"),
        "threshold": quantities.display("bootstrap.persistence_threshold.value"),
        "band_quantile": quantities.display("bootstrap.band_quantile"),
        "percent": quantities.display("bootstrap.band_quantile_percent"),
    }

    nominal_labels = legend_labels(
        rendering, {"survives": shown_survivors, "absorbed": shown_absorbed}
    )
    key = OutcomeKey(
        outcome_legend_entries(BAND_ORDER, nominal_labels, colors, nominal_counts),
        legend_title,
    )
    # Figure callers may supply partial presentation settings without a declaration.
    clamped_cfg = settings.get("analysis", {}).get("clamped", {})
    declaration = clamped_cfg.get(DECLARATION_KEY, {})
    is_fdr = (
        isinstance(declaration, Mapping) and declaration.get("rule") == FDR_FIXED_CUTOFF
    )
    nominal_subtitle_template = rendering.get(
        "nominal_row_subtitle", DEFAULT_NOMINAL_ROW_SUBTITLE
    )
    if is_fdr:
        old_phrase = "{percent}% simultaneous confidence band (parametric bootstrap)"
        new_phrase = (
            "nominal {percent}% simultaneous band (fixed-graph plug-in parametric "
            "bootstrap; coverage not independently established)"
        )
        if isinstance(nominal_subtitle_template, (list, tuple)):
            nominal_subtitle_template = [
                line.replace(old_phrase, new_phrase)
                for line in nominal_subtitle_template
            ]
        elif isinstance(nominal_subtitle_template, str):
            nominal_subtitle_template = nominal_subtitle_template.replace(
                old_phrase, new_phrase
            )

    top = BiologyRow(
        data=data,
        pairs=pairs,
        wild_type_source=tree_analysis,
        stop_points=stop_points,
        tree_overlays=[
            peak_branch_layer(
                peak_branch_segments(tree_analysis.plot_data, nominal_category),
                ("absorbed",),
                colors,
                rendering,
            ),
        ],
        outcome_key=key,
        persistence_markers=persistence_markers(
            pairs,
            nominal_category,
            BAND_ORDER,
            colors,
            annotations,
            biology_rendering,
            wild_type_identities(tree_analysis),
        ),
        band=tau,
        outcome_markers=True,
        title=format_lines(
            rendering.get("nominal_row_title", DEFAULT_NOMINAL_ROW_TITLE),
            peaks=shown_nominal,
            survive=shown_survivors,
            absorbed=shown_absorbed,
            stops=quantities.display("annotations.candidate.stop_codon_peak_count"),
            **common,
        ),
        subtitle=format_lines(
            nominal_subtitle_template,
            **common,
        ),
    )
    # Row 2 is coloured by post-absorption branch vertex count (absorbed members
    # folded into the surviving branches), on the scale of its own counts.
    band_data = plot_data_with_metrics(
        selected_view_as_tree(
            band_view,
            legacy_mixture=getattr(analysis, "legacy_mixture", False),
        )
    )
    band_nodes = band_view.plot_data.node_table
    band_stops = (
        stop_codon_nodes(band_nodes, rendering)
        if "has_stop_codon" in band_nodes.columns
        else band_nodes.clear()
    )
    band_counts = count_domain(band_data.edge_table)
    band_pairs = persistence_with_counts(band_view.pairs, band_data, band_counts)
    panel_size = tree_panel_size(rendering, appearance)
    bottom = BiologyRow(
        data=band_data,
        count_domain=band_counts,
        tree_labels=branch_count_labels(
            band_data,
            rendering,
            size=panel_size,
            y_domain=padded_scalar_domain(candidate.scalar_domain, rendering),
            stop_points=band_stops,
            wild_type_source=band_view,
            colorbar_box=count_colorbar_box(
                appearance, rendering, band_counts, width=panel_size[0]
            ),
        ),
        pairs=band_pairs,
        wild_type_source=band_view,
        stop_points=band_stops,
        persistence_markers=persistence_markers(
            band_pairs,
            band_category,
            ("survives",),
            colors,
            annotations,
            {**biology_rendering, "biology_count_domain": band_counts},
            wild_type_identities(band_view),
        ),
        band=tau,
        outcome_markers=True,
        diagram_legend=False,
        title=format_lines(
            rendering.get("band_row_title", DEFAULT_BAND_ROW_TITLE),
            peaks=shown_survivors,
            input_peaks=shown_nominal,
            absorbed=shown_absorbed,
            **common,
        ),
        subtitle=format_lines(
            rendering.get("band_row_subtitle", DEFAULT_BAND_ROW_SUBTITLE),
            retained=quantities.display(
                "bootstrap.candidate.after_band_rule.retained_sequence_count"
            ),
            saddles=quantities.display(
                "bootstrap.candidate.after_band_rule.saddles_removed_count"
            ),
            **common,
        ),
    )
    return assemble_biology_chart(
        candidate,
        appearance,
        rendering,
        stop_rendering=stop_rendering,
        biology_rendering=biology_rendering,
        rows=(top, bottom),
    )
