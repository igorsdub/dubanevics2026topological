"""Published-threshold stop-codon and synonymous-peak biology figure."""

from __future__ import annotations

from typing import Any

import altair as alt
import polars as pl

from ._charts import (
    annotated_peaks,
    annotation_legend,
    biology_axis_contract,
    biology_persistence_marker_layer,
    compose_tree_layers,
    landscape_population_subtitle,
    leaf_row_title,
    peak_marker_layers,
    plot_data_with_saved_metrics,
    count_domain,
    persistence_legend_layers,
    persistence_panel_size,
    persistence_with_counts,
    publication_scalar_scales,
    stack_leaf_rows,
    stop_codon_nodes,
    stop_tree_points_layer,
    synonymous_branch_layer,
    study_biology_persistence_chart,
    study_biology_tree_chart,
    unannotated_peaks,
    wild_type_identities,
    wild_type_layers,
)
from .palette import PEAK_GREY, SYNONYMOUS_GREEN, tree_panel_size


def build_study_tree_persistence_biology_figure(
    analysis: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Build maximum-stop and adjacent-saddle synonym rows from saved evidence.

    Every printed count and the threshold are the registry's display text
    (``analysis["manuscript_quantities"]``); the saved evidence only draws.
    """
    quantities = analysis["manuscript_quantities"]
    appearance = settings["figures"]["figures"][
        "study_tree_persistence_biology"
    ]
    rendering = {**settings["figures"].get("rendering", {}), **appearance}
    font_size = int(appearance.get("font_size", rendering.get("font_size", 15)))
    rendering = {
        **rendering,
        "font_size": font_size,
        "axis_label_font_size": font_size,
        "legend_font_size": font_size,
    }
    study = analysis["landscapes"]["study_clamped"]
    evidence = analysis["annotations"]["study_clamped"]
    tree_analysis = study.trees["split"]
    data = plot_data_with_saved_metrics(
        tree_analysis, analysis["tables"]["study_clamped_branch_table"]
    )
    pairs = evidence.pairs.rename(
        {"birth_sequence": "birth_label", "death_sequence": "death_label"}
    )
    counts = count_domain(data.edge_table)
    pairs = persistence_with_counts(pairs, data, counts)
    rendering = {**rendering, "biology_count_domain": counts}
    # A branch is green only when it is both a terminal peak branch and its
    # adjacent lower endpoint is a saddle with matching amino acids. This keeps
    # root-only singleton branches out of the visual synonym definition.
    synonymous_pairs = pairs.filter(pl.col("synonymous_endpoints"))
    synonymous_keys = {
        tuple(sorted((row["birth_label"], row["adjacent_saddle_sequence"])))
        for row in synonymous_pairs.to_dicts()
        if row.get("adjacent_saddle_sequence") is not None
    }
    peak_synonymous_branches = {
        tuple(sorted((row["upper_sequence"], row["lower_sequence"])))
        for row in evidence.branches.filter(
            pl.col("is_peak_branch") & pl.col("synonymous_endpoints")
        ).to_dicts()
    }
    synonymous_keys &= peak_synonymous_branches
    stop_points = stop_codon_nodes(evidence.critical_points, rendering)
    stop_rendering = rendering
    wild_type = wild_type_identities(tree_analysis)
    biology_rendering = {
        **rendering,
        # The stop and synonymous markers are larger than the plain points.
        "biology_persistence_point_size": int(
            appearance.get("persistence", {}).get(
                "marker_size", appearance.get("persistence", {}).get("point_size", 70)
            )
        ),
        # Count-driven opacity of these highlighted markers never falls below this.
        "biology_marker_opacity_floor": float(
            appearance.get("persistence", {}).get("marker_opacity_floor", 0.0)
        ),
    }
    segments = data.edge_segment_table.filter(
        pl.Series(
            "synonymous_peak_branch",
            [
                tuple(sorted((row["source_label"], row["target_label"])))
                in synonymous_keys
                for row in data.edge_segment_table.to_dicts()
            ],
            dtype=pl.Boolean,
        )
    )
    rows = []
    for row_index, stops in enumerate((True, False)):
        tree = study_biology_tree_chart(data, study, appearance, rendering)
        stop_markers = (
            [stop_tree_points_layer(stop_points, stop_rendering)] if stops else []
        )
        overlays = [] if stops else [synonymous_branch_layer(segments, rendering)]
        wild_type_markers = wild_type_layers(tree_analysis, rendering)
        tree_width, tree_height = tree_panel_size(rendering, appearance)
        tree = compose_tree_layers(
            tree,
            overlays=overlays,
            stop_markers=stop_markers,
            keys=[
                annotation_legend(
                    tree_width,
                    tree_height,
                    rendering,
                    wild_type=bool(wild_type_markers),
                    stop_codon=stops,
                )
            ]
            if stops or wild_type_markers
            else [],
            wild_type=wild_type_markers,
        )
        tree = tree.resolve_scale(color="independent")
        tree = biology_axis_contract(tree, y_title="Fitness (at birth, WT = 0)")
        persistence_appearance = {
            **appearance,
            "persistence": {
                **appearance.get("persistence", {}),
                "height": tree_panel_size(rendering, appearance)[1],
            },
        }
        if stops:
            # Stop-codon peaks are solid crimson triangles in place of the grey
            # ones (ADR 0012); the base layer keeps only the other peaks.
            persistence = study_biology_persistence_chart(
                unannotated_peaks(pairs, wild_type=wild_type),
                study,
                persistence_appearance,
                rendering,
            )
            markers = peak_marker_layers(
                annotated_peaks(pairs, wild_type=wild_type),
                biology_rendering,
                color=rendering.get("peak_color", PEAK_GREY),
                annotation="Stop-codon containing peak",
                wild_type=wild_type,
            )
        else:
            persistence = study_biology_persistence_chart(
                pairs, study, persistence_appearance, rendering
            )
            # The synonymous peak stays a green triangle over the grey one.
            markers = [
                biology_persistence_marker_layer(
                    synonymous_pairs,
                    biology_rendering,
                    color=rendering.get("synonymous_color", SYNONYMOUS_GREEN),
                    annotation="Synonymous peak branch",
                )
            ]
        # The key sits inside the diagram, bottom right, one entry per line: peak,
        # and stop codon on the stop row, synonymous leaves alone on the other.
        legend_layers = persistence_legend_layers(
            *persistence_panel_size(persistence_appearance, rendering),
            rendering,
            peak=stops,
            wild_type=stops and bool(wild_type & set(pairs["birth_label"].to_list())),
            stop_codon=stops,
            synonymous=not stops,
        )
        persistence = alt.layer(persistence, *markers, *legend_layers).resolve_scale(
            color="independent"
        )
        persistence = biology_axis_contract(
            persistence,
            y_title=None,
            x_title=(
                rendering.get("persistence_x_title", "Fitness at death")
                if row_index == 1
                else None
            ),
        )
        row = alt.hconcat(
            tree, persistence, spacing=int(rendering.get("panel_gap", 10))
        ).resolve_scale(x="independent", y="shared", color="shared")
        rows.append(
            row.properties(
                title=leaf_row_title(
                    "stops" if stops else "synonymous",
                    quantities.display(
                        "annotations.published.stop_codon_peak_count"
                        if stops
                        else "annotations.published.synonymous_peak_count"
                    ),
                    landscape_population_subtitle(
                        quantities, "published", include_graph=True
                    ),
                )
            )
        )
    chart = stack_leaf_rows(rows, rendering)
    chart = publication_scalar_scales(chart, study.scalar_domain, rendering)
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
