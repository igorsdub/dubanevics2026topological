"""D10 view ``bootstrap_band``: nominal and band-simplified tree with their diagrams.

Spec 2, Section 6, view 6. Row 1 is the nominal split tree beside its persistence
diagram with the bootstrap band, the strip of vertical height 2c above the
diagonal; peaks with persistence above 2c survive (grey) and the peaks inside the
strip are absorbed (black on the diagram; on the tree only the absorbed leaf's own
branch, tip to merge node, is black, with no tip marker). Row 2 is the
band-simplified split tree beside its surviving persistence points. Both rows share
one fitness axis. No marginal histogram: it belongs to the tree persistence plots. Everything is read from saved products
(``SavedTreeInputs``, ``BandSurvival``); nothing is rebuilt.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import altair as alt
import polars as pl

from ._charts import (
    biology_axis_contract,
    count_domain,
    overlay_count_colorbar,
    persistence_with_counts,
    persistence_legend_layers,
    persistence_panel_size,
    biology_persistence_marker_layer,
    leaf_row_label,
    peak_marker_layers,
    plot_data_with_metrics,
    publication_scalar_scales,
    stack_leaf_rows,
    study_biology_persistence_chart,
    stop_cross_layer,
    tree_marker_size,
    synonymous_branch_layer,
    synonymous_leaf_edges,
    synonymous_leaf_segments,
    synonymous_teal_branch_layer,
    tree_chart,
    wild_type_identities,
)
from ._count_labels import count_label_text, fixed_label_text
from .band_survival import BandSurvival, peak_nodes
from .candidate_tree_biology import persistence_band_layers
from .candidate_tree_bootstrap_biology import peak_branch_layer, peak_branch_segments
from .palette import EMPHASIS_BLACK, PEAK_GREY, SYNONYMOUS_GREEN, SYNONYMOUS_TEAL
from .tree_persistence_view import STYLES

# "rows": the nominal and band rows for stops, then again for synonymous leaves (four
# rows); "stops" and "synonymous": the two rows of one leaf kind, the files of the
# annotated style; "combined": the nominal and band rows marking stops (slides).
# "encoded": the nominal and band rows in the combined stop-and-synonymous encoding (the
# file ``bootstrap_combined``): teal synonymous leaves and peaks, a red cross for a stop.
LAYOUTS = ("combined", "rows", "stops", "synonymous", "encoded")


def _tree_panel(
    inputs: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    domain: tuple[float, float],
    annotated: bool,
    absorbed: pl.DataFrame | None,
    coloring: str = "count",
    slide: bool = False,
    keyed: bool = True,
    leaves: str = "stops",
):
    """One tree, coloured by branch count (``coloring="none"``: plain edges); each
    absorbed leaf's own branch is black.

    ``leaves`` is the leaf kind the row highlights: ``"stops"`` draws the stop-codon
    crosses (annotated style), ``"synonymous"`` draws the synonymous leaves' branches
    green, above the absorbed black, and no crosses.

    ``keyed=False`` draws neither the count colour bar nor the tree key (Wild type /
    stop entry): the figure shows them once, on its first row. Both sit over the
    panel, so leaving them out moves no plot area."""
    analysis = inputs.tree_analysis
    if not annotated:
        analysis = SimpleNamespace(
            **{k: v for k, v in vars(analysis).items() if k != "wild_type_marker_table"}
        )
    data = plot_data_with_metrics(analysis)
    branches = count_domain(data.edge_table)
    overlays = []
    if absorbed is not None and absorbed.height:
        overlays.append(
            peak_branch_layer(
                peak_branch_segments(
                    data, dict.fromkeys(absorbed["dna_sequence"], "absorbed")
                ),
                ("absorbed",),
                {"absorbed": str(rendering.get("absorbed_color", EMPHASIS_BLACK))},
                rendering,
                keep_fitness_axis=True,
            )
        )
    if leaves in ("synonymous", "encoded"):
        edges = synonymous_leaf_edges(data)
        if not edges.is_empty():
            segments = synonymous_leaf_segments(data, edges)
            overlays.append(
                synonymous_teal_branch_layer(
                    segments, rendering, branches, keep_fitness_axis=True
                )
                if leaves == "encoded"
                else synonymous_branch_layer(
                    segments, rendering, keep_fitness_axis=True
                )
            )
    tree = tree_chart(
        data,
        analysis,
        SimpleNamespace(scalar_domain=domain),
        figure_settings,
        rendering,
        branches,
        linked=True,
        annotated=annotated and leaves in ("stops", "encoded"),
        overlays=overlays,
        coloring=coloring,
        annotation_key=keyed,
    )
    if coloring == "none" or not keyed:
        return tree, data
    return overlay_count_colorbar(tree, figure_settings, rendering, branches), data


def _diagram(
    inputs: Any,
    data: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    domain: tuple[float, float],
    survival: BandSurvival,
    *,
    nominal: bool,
    annotated: bool,
    leaves: str = "stops",
):
    """Persistence diagram: the strip and the absorbed peaks only on the nominal row.

    ``leaves="stops"`` marks the stop-codon peaks (crimson) and the wild type;
    ``"synonymous"`` marks the synonymous leaves' peaks green (an annotation colour
    replaces the outcome colour, as a stop's does) and leaves stops and wild type plain.
    """
    encoded = leaves == "encoded"
    synonymous = leaves in ("synonymous", "encoded")
    pairs = persistence_with_counts(
        inputs.persistence_table, data, count_domain(data.edge_table)
    )
    layers = []
    if nominal:
        layers.extend(persistence_band_layers(survival.two_c, domain, rendering))
    counts = count_domain(data.edge_table)
    rendering = {**rendering, "biology_count_domain": counts}
    if encoded:
        # The diagram's triangles are as large as the tree's stop cross.
        size = tree_marker_size(rendering)
        rendering = {**rendering, "biology_persistence_point_size": size}
        figure_settings = {
            **figure_settings,
            "persistence": {**figure_settings.get("persistence", {}), "point_size": size},
        }
    marks_stops = annotated and (not synonymous or encoded)
    stops = set(data.node_table.filter(pl.col("has_stop_codon"))["dna_sequence"].to_list()) if marks_stops and "has_stop_codon" in data.node_table.columns else set()
    stop_pairs = pairs.filter(pl.col("birth_label").is_in(sorted(stops)))
    # In the encoded layout a stop is a cross on top of the marker, not a crimson peak.
    pairs = pairs.with_columns(
        pl.col("birth_label").is_in(sorted(stops)).alias("birth_has_stop_codon")
        if not encoded
        else pl.lit(False).alias("birth_has_stop_codon")
    )
    wild_type = (
        wild_type_identities(inputs.tree_analysis)
        if marks_stops
        else frozenset()
    )
    absorbed = set(survival.absorbed) if nominal else set()
    plain = pairs.filter(
        ~pl.col("birth_has_stop_codon")
        & ~pl.col("birth_label").is_in(sorted(wild_type | absorbed))
    )
    teal_peaks: set[str] = set()
    if encoded:
        edges = synonymous_leaf_edges(data)
        teal_peaks = (
            set(edges["target_label"].to_list()) - wild_type
            if "target_label" in edges.columns
            else set()
        )
        plain = plain.filter(~pl.col("birth_label").is_in(sorted(teal_peaks)))
    layers.append(
        study_biology_persistence_chart(
            plain, SimpleNamespace(scalar_domain=domain), figure_settings, rendering
        )
    )
    for is_absorbed, color in ((False, PEAK_GREY), (True, str(rendering.get("absorbed_color", EMPHASIS_BLACK)))):
        marked = pairs.filter(
            (pl.col("birth_has_stop_codon") | pl.col("birth_label").is_in(sorted(wild_type | absorbed)))
            & (pl.col("birth_label").is_in(sorted(absorbed)) == is_absorbed)
        )
        layers.extend(peak_marker_layers(marked, rendering, color=color,
                                        annotation="absorbed" if is_absorbed else "peak",
                                        wild_type=wild_type))
    if synonymous:
        edges = synonymous_leaf_edges(data)
        leaf_labels = (
            set(edges["target_label"].to_list())
            if "target_label" in edges.columns
            else set()
        )
        green = pairs.filter(pl.col("birth_label").is_in(sorted(leaf_labels)))
        if not green.is_empty():
            layers.append(
                biology_persistence_marker_layer(
                    green,
                    rendering,
                    color=str(
                        rendering.get("synonymous_teal", SYNONYMOUS_TEAL)
                        if encoded
                        else rendering.get("synonymous_color", SYNONYMOUS_GREEN)
                    ),
                    annotation="Synonymous peak branch",
                )
            )
    if encoded:
        crossed = stop_pairs.filter(~pl.col("birth_label").is_in(sorted(wild_type)))
        if not crossed.is_empty():
            layers.append(
                stop_cross_layer(crossed, rendering)
            )
    chart = alt.layer(*layers).resolve_scale(color="independent")
    return biology_axis_contract(
        chart,
        y_title=None,
        x_title=rendering.get("persistence_x_title", "Fitness at death"),
    )


def _row_title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(text=text, subtitle=[subtitle], anchor="start")


def _encoded_flags(inputs: Any, data: Any) -> tuple[bool, bool]:
    """``(has synonymous peaks, has stop-codon peaks)`` among a tree's persistence peaks."""
    births = set(inputs.persistence_table["birth_label"].to_list())
    edges = synonymous_leaf_edges(data)
    synonymous = (
        "target_label" in edges.columns
        and bool(set(edges["target_label"].to_list()) & births)
    )
    stops = "has_stop_codon" in data.node_table.columns and bool(
        set(data.node_table.filter(pl.col("has_stop_codon"))["dna_sequence"].to_list())
        & births
    )
    return synonymous, stops


def _leaf_count(inputs: Any, data: Any, leaves: str, annotated: bool) -> int:
    """How many of a tree's peaks the ``leaves`` row highlights."""
    births = set(inputs.persistence_table["birth_label"].to_list())
    if leaves == "synonymous":
        edges = synonymous_leaf_edges(data)
        found = (
            set(edges["target_label"].to_list())
            if "target_label" in edges.columns
            else set()
        )
        return len(found & births)
    if annotated and "has_stop_codon" in data.node_table.columns:
        stops = data.node_table.filter(pl.col("has_stop_codon"))["dna_sequence"]
        return len(set(stops.to_list()) & births)
    return 0


def build_bootstrap_band_figure(
    nominal: Any,
    band: Any,
    survival: BandSurvival,
    settings: dict[str, Any],
    style: str,
    *,
    title: str | None,
    subtitle: str | None,
    coloring: str = "count",
    scalar_domain: tuple[float, float] | None = None,
    layout: str = "combined",
) -> Any:
    """The ``bootstrap_band`` figure in ``annotated`` or ``vanilla`` style.

    ``layout`` (annotated style only) is ``"combined"``, the nominal and band rows
    marking stop codons (the slides' chart and the extra named ``bootstrap_band_combined``),
    ``"stops"`` or ``"synonymous"``: the two rows (nominal, band) marking that leaf
    kind, the files ``bootstrap_band_stops`` and ``bootstrap_band_synonymous``, or
    ``"rows"``: both blocks stacked, four rows (no longer written by the build).

    ``title=None`` is slide mode: no figure title and no row headings. ``coloring="none"``
    draws the trees in one plain edge colour with no count colour bar. ``scalar_domain``
    replaces the shared fitness extent of the two trees.

    ``settings`` is the figures configuration (``settings["figures"]``); the view's
    appearance table is ``d10.bootstrap_band``. ``nominal`` and ``band`` are
    ``SavedTreeInputs`` of the nominal tree and of its band-simplified tree.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}.")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}.")
    figures = settings["figures"]
    figure_settings = figures["d10"]["bootstrap_band"]
    rendering = {**figures.get("rendering", {}), **figure_settings}
    annotated = style == "annotated"
    lows, highs = zip(nominal.scalar_domain, band.scalar_domain, strict=True)
    domain = scalar_domain or (min(lows), max(highs))
    gap = int(rendering.get("panel_gap", 10))
    absorbed = peak_nodes(nominal.tree_analysis.plot_data.node_table, survival.absorbed)

    states = (
        (
            nominal,
            True,
            f"Nominal tree: {count_label_text(survival.nominal_peaks)} peaks",
            (
                f"{count_label_text(len(survival.surviving))} survive "
                f"(persistence > 2c = {fixed_label_text(survival.two_c, 3)}, grey); "
                f"{count_label_text(len(survival.absorbed))} absorbed "
                "(inside the band, black)"
            ),
        ),
        (
            band,
            False,
            (
                "Bootstrapped tree: "
                f"{count_label_text(len(survival.surviving))} surviving peaks"
            ),
            (
                "absorbed branches folded into parent saddles; "
                f"{count_label_text(band.tree_analysis.plot_data.edge_table.height)}"
                " branches"
            ),
        ),
    )
    # Stop-codon rows first; the "rows" layout adds the same two rows for synonymous leaves.
    if annotated and layout == "rows":
        kinds = ("stops", "synonymous")
    elif annotated and layout == "synonymous":
        kinds = ("synonymous",)
    elif annotated and layout == "encoded":
        kinds = ("encoded",)
    else:
        kinds = ("stops",)
    # Row headings name the leaf kind in the row layouts of the annotated style.
    labelled = annotated and layout not in ("combined", "encoded")
    rows = []
    # The count colour bar and the tree key appear once, on the first row; the diagram
    # key on the first row of each block.
    for block, kind in enumerate(kinds):
        for item, is_nominal, heading, note in states:
            tree, data = _tree_panel(
                item,
                figure_settings,
                rendering,
                domain,
                annotated,
                absorbed if is_nominal else None,
                coloring,
                title is None,
                keyed=is_nominal and block == 0,
                leaves=kind,
            )
            diagram = _diagram(
                item,
                data,
                figure_settings,
                rendering,
                domain,
                survival,
                nominal=is_nominal,
                annotated=annotated,
                leaves=kind,
            )
            count = _leaf_count(item, data, kind, annotated) if labelled else 0
            # The key sits inside the block's first diagram, bottom right, one entry per line.
            if is_nominal and kind == "stops":
                diagram = alt.layer(
                    diagram,
                    *persistence_legend_layers(
                        *persistence_panel_size(figure_settings, rendering),
                        rendering,
                        stop_codon=annotated,
                        wild_type=annotated
                        and bool(
                            wild_type_identities(item.tree_analysis)
                            & set(item.persistence_table["birth_label"].to_list())
                        ),
                    ),
                )
            elif is_nominal and kind == "encoded":
                has_synonymous, has_stops = _encoded_flags(item, data)
                diagram = alt.layer(
                    diagram,
                    *persistence_legend_layers(
                        *persistence_panel_size(figure_settings, rendering),
                        rendering,
                        stop_codon=has_stops,
                        synonymous=has_synonymous,
                        peak_color=PEAK_GREY,
                        encoded=True,
                        wild_type=bool(
                            wild_type_identities(item.tree_analysis)
                            & set(item.persistence_table["birth_label"].to_list())
                        ),
                    ),
                )
            elif is_nominal:
                diagram = alt.layer(
                    diagram,
                    *persistence_legend_layers(
                        *persistence_panel_size(figure_settings, rendering),
                        rendering,
                        peak=False,
                        stop_codon=False,
                        synonymous=count > 0,
                    ),
                )
            row = alt.hconcat(tree, diagram, spacing=gap).resolve_scale(
                color="independent"
            )
            # Every tree draws its own fitness axis (a shared y axis hides the upper one).
            row = row.resolve_axis(y="independent")
            if title is not None:
                row = row.properties(
                    title=alt.TitleParams(
                        text=heading,
                        subtitle=[note, leaf_row_label(kind, count_label_text(count))],
                        anchor="start",
                    )
                    if labelled
                    else _row_title(heading, note)
                )
            rows.append(row)
    if labelled:
        chart = stack_leaf_rows(
            rows, {"comparison_panel_gap": rendering.get("row_gap", 16)}
        )
    else:
        chart = alt.vconcat(
            *rows, spacing=int(rendering.get("row_gap", 16))
        ).resolve_scale(color="independent")
    if title is not None:
        chart = chart.properties(
            title=alt.TitleParams(text=title, subtitle=[subtitle], anchor="start")
        )
    font_size = int(
        figure_settings.get("tree", {}).get("font_size", rendering.get("font_size", 15))
    )
    chart = (
        chart.configure(font=rendering.get("font_family", "Arial, Liberation Sans, Helvetica, sans-serif"))
        .configure_axis(
            labelFontSize=int(rendering.get("axis_label_font_size", font_size)),
            titleFontSize=font_size,
        )
        .configure_legend(
            labelFontSize=int(rendering.get("legend_font_size", 15)),
            titleFontSize=int(rendering.get("legend_font_size", 15)),
        )
        .configure_title(
            fontSize=int(rendering.get("heading_font_size", font_size)),
            subtitleFontSize=int(rendering.get("subtitle_font_size", 14)),
            anchor="start",
        )
        .configure_view(strokeWidth=0)
    )
    return publication_scalar_scales(
        chart,
        domain,
        rendering,
        y_axis_extent=int(rendering.get("tree_y_axis_extent", 60)),
    )


__all__ = ["PEAK_GREY", "STYLES", "build_bootstrap_band_figure"]
