"""D10 view ``tree_persistence``: a split tree beside its persistence diagram.

Spec 2, Section 6, view 1. The tree and the diagram share the vertical fitness
axis; a marginal histogram of every sequence's fitness (stop codons included)
sits above the diagram on its fitness axis, with a
blank spacer of the same height above the tree so the plot areas stay aligned.
The ``annotated`` style adds the stop-codon crosses and the wild-type star; the
``vanilla`` style draws neither. Everything is read from ``SavedTreeInputs``
(``build/d10_tree_inputs.py``); nothing is rebuilt.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import altair as alt
import polars as pl

from ._branch_count_style import branch_count_opacity
from ._count_labels import count_label_text
from ._charts import (
    biology_axis_contract,
    count_domain,
    overlay_count_colorbar,
    persistence_chart,
    persistence_legend_layers,
    persistence_panel_size,
    persistence_with_counts,
    biology_persistence_marker_layer,
    stop_cross_layer,
    tree_marker_size,
    synonymous_teal_branch_layer,
    plot_data_with_metrics,
    publication_scalar_scales,
    set_plain_point_color,
    leaf_row_title,
    stack_leaf_rows,
    synonymous_branch_layer,
    synonymous_leaf_edges,
    synonymous_leaf_segments,
    tree_chart,
    wild_type_identities,
)
from ._top_histogram import top_histogram_bins, top_histogram_chart
from .palette import (
    COUNT_BLACK,
    PEAK_GREY,
    PERSISTENCE_WIDTH,
    SYNONYMOUS_GREEN,
    SYNONYMOUS_TEAL,
    TREE_WIDTH,
)

STYLES = ("annotated", "vanilla")
LAYOUTS = ("combined", "rows", "encoded")
COLORINGS = ("count", "none")


fitness_bins = top_histogram_bins


def fitness_histogram_chart(
    fitness: pl.Series,
    scalar_domain: tuple[float, float],
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    *,
    hidden_width: int | None = None,
    y_title: str = "Sequences",
) -> Any:
    """Marginal histogram of all sequences' fitness, on the diagram's fitness axis.

    It draws no x tick labels (the diagram below carries them) and a count axis
    of two ticks, 0 and 20K (the axis runs to 1.05 times the tallest bin). With ``hidden_width`` the same chart is
    drawn invisibly at that width: the cell above the tree, whose axis labels
    overhang the plot area exactly as the visible marginal's do, so the tree and
    the diagram below stay aligned to the pixel.
    """
    persistence = figure_settings.get("persistence", {})
    return top_histogram_chart(
        fitness,
        scalar_domain,
        rendering,
        width=hidden_width
        or int(
            persistence.get(
                "width", rendering.get("persistence_width", PERSISTENCE_WIDTH)
            )
        ),
        height=int(figure_settings.get("histogram_height", 70)),
        bins=int(figure_settings["histogram_bins"])
        if "histogram_bins" in figure_settings
        else None,
        hidden=bool(hidden_width),
        title=y_title,
    )


def build_tree_persistence_figure(
    inputs: Any,
    settings: dict[str, Any],
    style: str,
    *,
    title: str | None,
    subtitle: str | None,
    coloring: str = "count",
    scalar_domain: tuple[float, float] | None = None,
    stop_identities: set[str] | None = None,
    rendering_overrides: dict[str, Any] | None = None,
    layout: str = "combined",
) -> Any:
    """The ``tree_persistence`` figure in ``annotated`` or ``vanilla`` style.

    ``title=None`` draws no title block (slide mode). ``coloring="none"`` draws the
    tree in one plain edge colour with no count colour bar and the persistence
    points in the same colour; the panels keep their size. ``scalar_domain``
    replaces the tree's own fitness extent on every fitness axis (slides share one).

    ``layout`` (annotated style only; vanilla draws neither leaf kind) is
    ``"encoded"``, one tree beside one diagram in the combined stop-and-synonymous
    encoding (synonymous leaf branches teal by vertex count, teal markers on the
    diagram, a stop codon a red cross on top of the marker; the file
    ``tree_persistence_combined``), ``"combined"``, one tree beside one diagram marking stop codons and synonymous
    leaves together (the slides' chart, and the extra named ``tree_persistence_combined``),
    or ``"rows"``, the layout of the Published graph's
    ``study_tree_persistence_biology``: a row of stop-codon peaks and, below it, a
    row of synonymous leaves, never both in one chart (the published default).
    The marginal fitness histogram is drawn once, above the first row.

    ``settings`` is the figures configuration (``settings["figures"]``); the
    view's own table is ``d10.tree_persistence``. ``title`` and ``subtitle`` are
    the caller's display text (the build derives them from the product labels).
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}.")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}.")
    figures = settings["figures"]
    figure_settings = figures["d10"]["tree_persistence"]
    rendering = {
        **figures.get("rendering", {}),
        **figure_settings,
        **(rendering_overrides or {}),
    }
    if coloring not in COLORINGS:
        raise ValueError(f"coloring must be one of {COLORINGS}, got {coloring!r}.")
    domain = scalar_domain or inputs.scalar_domain
    annotated = style == "annotated"
    if inputs.fitness_values is None:
        raise ValueError("The tree inputs carry no population fitness values.")
    # (kind, stop-codon peaks drawn, synonymous leaves drawn) per row.
    kinds = (
        (("stops", True, False), ("synonymous", False, True))
        if annotated and layout == "rows"
        else (("combined", annotated, annotated),)
    )
    encoded = annotated and layout == "encoded"
    rows = []
    for index, (kind, with_stops, with_synonymous) in enumerate(kinds):
        row, count = tree_persistence_row(
            inputs,
            figure_settings,
            rendering,
            annotated=annotated,
            with_stops=with_stops,
            with_synonymous=with_synonymous,
            coloring=coloring,
            domain=domain,
            stop_identities=stop_identities,
            marginal=index == 0,
            encoded=encoded,
        )
        if len(kinds) > 1 and title is not None:
            row = row.properties(title=leaf_row_title(kind, count_label_text(count)))
        rows.append(row)
    chart = rows[0] if len(rows) == 1 else stack_leaf_rows(rows, rendering)
    if encoded and title is not None:
        # The one-row combined file carries its heading (graph / threshold / FDR and the
        # population line) on the row, a panel title, so the untitled canonical file keeps
        # it as the two-row files keep their row headings; the titled twin shows it once.
        chart = stack_leaf_rows(
            [
                rows[0].properties(
                    title=alt.TitleParams(text=title, subtitle=[subtitle], anchor="start")
                )
            ],
            rendering,
        )
    elif title is not None:
        chart = chart.properties(
            title=alt.TitleParams(text=title, subtitle=[subtitle], anchor="start")
        )
    return publication_scalar_scales(
        configure_view_chart(chart, figure_settings, rendering), domain, rendering
    )


def tree_persistence_row(
    inputs: Any,
    figure_settings: dict[str, Any],
    rendering: dict[str, Any],
    *,
    annotated: bool,
    with_stops: bool,
    with_synonymous: bool,
    coloring: str,
    domain: tuple[float, float],
    stop_identities: set[str] | None = None,
    marginal: bool = True,
    encoded: bool = False,
) -> tuple[Any, int]:
    """One tree beside its diagram, and the count of the leaves the row highlights.

    ``with_stops`` draws the stop-codon crosses and crimson peaks and the wild-type
    star on the diagram; ``with_synonymous`` draws the green synonymous leaves, as
    the Published graph's biology figure does. A row with only synonymous leaves still
    carries the wild-type star on its tree (as the biology figure's second row does) and
    draws the wild-type peak as a plain one. ``marginal=False`` leaves the marginal
    fitness histogram and its spacer out (the second row of a stacked layout).
    The count is the row's stop-codon peaks, or its synonymous leaves when that is
    all the row draws.

    ``encoded`` is the combined stop-and-synonymous encoding: the synonymous leaf
    branches are teal by vertex count, the diagram's synonymous peaks teal ▲ and every
    other peak grey ▲ (count fade), a stop codon a red cross on top of its marker, and
    the key reads ▲ peak, ● synonymous, × stop codon.
    """
    plain = coloring == "none"
    if encoded:
        # The diagram's triangles are as large as the tree's stop cross.
        figure_settings = {
            **figure_settings,
            "persistence": {
                **figure_settings.get("persistence", {}),
                "point_size": tree_marker_size(rendering),
            },
        }
    tree_analysis = vanilla_analysis(inputs.tree_analysis, annotated)
    plot_data = plot_data_with_metrics(tree_analysis)
    branch_domain = count_domain(plot_data.edge_table)
    pairs = persistence_with_counts(inputs.persistence_table, plot_data, branch_domain)
    wild_type = wild_type_identities(tree_analysis) if with_stops else frozenset()
    # Synonymous leaves, as the Published graph's biology figure draws them: the
    # terminal peak branch whose adjacent saddle has the peak's amino acids is green
    # on the tree and its peak a green triangle on the diagram.
    synonymous_edges = (
        synonymous_leaf_edges(plot_data)
        if with_synonymous
        and {"source_label", "target_label"} <= set(plot_data.edge_table.columns)
        else pl.DataFrame(schema={"source_label": pl.String, "target_label": pl.String})
    )
    synonymous_overlays = []
    if not synonymous_edges.is_empty():
        synonymous_overlays.append(
            synonymous_teal_branch_layer(
                synonymous_leaf_segments(plot_data, synonymous_edges),
                rendering,
                branch_domain,
                keep_fitness_axis=True,
            )
            if encoded
            else synonymous_branch_layer(
                synonymous_leaf_segments(plot_data, synonymous_edges),
                rendering,
                keep_fitness_axis=True,
            )
        )

    tree = tree_chart(
        plot_data,
        tree_analysis,
        inputs.analysis,
        figure_settings,
        rendering,
        branch_domain,
        linked=True,
        annotated=with_stops,
        overlays=synonymous_overlays,
        coloring=coloring,
        stop_identities=stop_identities,
    )
    if not plain:
        tree = overlay_count_colorbar(tree, figure_settings, rendering, branch_domain)
    teal_labels = (
        set(synonymous_edges["target_label"].to_list()) - wild_type if encoded else set()
    )
    persistence = persistence_chart(
        pairs.filter(~pl.col("birth_label").is_in(sorted(wild_type | teal_labels))),
        figure_settings,
        rendering,
        branch_domain,
        domain,
        include_config=False,
    )
    if plain:
        persistence = set_plain_point_color(
            persistence, str(rendering.get("plain_edge_color", COUNT_BLACK))
        )
    stop_mask = []
    stops: set[str] = set()
    wild_rows = (
        pairs.filter(pl.col("birth_label").is_in(sorted(wild_type)))
        if wild_type
        else pairs.clear()
    )
    if not wild_rows.is_empty():
        from ._charts import FIVE_POINT_STAR

        stop_mask.append(biology_persistence_marker_layer(
            wild_rows, rendering, color=rendering.get("wild_type_color", "#d62728"),
            annotation="Wild type", shape=FIVE_POINT_STAR,
        ))
    if with_stops and "has_stop_codon" in plot_data.node_table.columns:
        # A peak with a stop codon is a crimson triangle with its count opacity.
        stops = set(
            plot_data.node_table.filter(pl.col("has_stop_codon"))["dna_sequence"]
            .drop_nulls()
            .to_list()
        )
        if stop_identities is not None:
            stops &= set(stop_identities)
        stop_rows = pairs.filter(pl.col("birth_label").is_in(sorted(stops - wild_type)))
        if encoded:
            # The stop codon is a red cross on top of the (grey or teal) marker.
            if not stop_rows.is_empty():
                stop_mask.append(
                    stop_cross_layer(stop_rows, rendering)
                )
        elif not stop_rows.is_empty():
            # Crimson points with the branch-count opacity, drawn above the plain
            # points; the diagram has no × overlay.
            persistence = persistence_chart(
                pairs.filter(~pl.col("birth_label").is_in(sorted(stops | wild_type))),
                figure_settings, rendering, branch_domain, domain,
                include_config=False,
            )
            if plain:
                persistence = set_plain_point_color(
                    persistence, str(rendering.get("plain_edge_color", COUNT_BLACK))
                )
            stop_rows = stop_rows.with_columns(
                pl.col("vertex_count").map_elements(
                    lambda count: branch_count_opacity(count, rendering, branch_domain),
                    return_dtype=pl.Float64,
                ).alias("count_opacity")
            )
            stop_mask.append(
                alt.Chart(stop_rows).mark_point(
                    shape="triangle-up", filled=True,
                    color=rendering.get("stop_codon_color", "#dc143c"),
                    size=int(figure_settings.get("persistence", {}).get(
                        "point_size", rendering.get("persistence_point_size", 65))),
                ).encode(x="death_scalar:Q", y="birth_scalar:Q",
                         opacity=alt.Opacity("count_opacity:Q", scale=None, legend=None))
            )
    synonymous_rows = pairs.filter(
        pl.col("birth_label").is_in(
            sorted(set(synonymous_edges["target_label"].to_list()) - wild_type)
        )
    )
    synonymous_mask = []
    if not synonymous_rows.is_empty():
        synonymous_mask.append(
            biology_persistence_marker_layer(
                synonymous_rows,
                {
                    **rendering,
                    "biology_count_domain": branch_domain,
                    "biology_persistence_point_size": int(
                        figure_settings.get("persistence", {}).get(
                            "point_size", rendering.get("persistence_point_size", 65)
                        )
                    ),
                },
                color=rendering.get(
                    "synonymous_teal" if encoded else "synonymous_color",
                    SYNONYMOUS_TEAL if encoded else SYNONYMOUS_GREEN,
                ),
                annotation="Synonymous peak branch",
            )
        )
    # Markers, then the stop crosses (and the wild-type star) on top.
    stop_mask = [*synonymous_mask, *stop_mask]
    stop_peaks = stops & set(pairs["birth_label"].to_list())
    # The key sits inside the diagram, bottom right, one entry per line. A row that
    # draws only synonymous leaves names only those.
    diagram_width, diagram_height = persistence_panel_size(figure_settings, rendering)
    legend_layers = persistence_legend_layers(
        diagram_width,
        diagram_height,
        rendering,
        peak=with_stops or not with_synonymous,
        wild_type=not wild_rows.is_empty(),
        stop_codon=with_stops and bool(stop_peaks),
        synonymous=not synonymous_rows.is_empty(),
        peak_color=PEAK_GREY if encoded else COUNT_BLACK,
        encoded=encoded,
    )
    persistence = alt.layer(persistence, *stop_mask, *legend_layers)
    diagram = biology_axis_contract(
        persistence,
        y_title=None,
        x_title=rendering.get("persistence_x_title", "Fitness at death"),
    )
    gap = int(rendering.get("panel_gap", 10))
    count = (
        synonymous_rows.height
        if with_synonymous and not with_stops
        else len(stop_peaks)
    )
    if not marginal:
        row = alt.hconcat(tree, diagram, spacing=gap).resolve_scale(color="independent")
        return row, count
    # The combined-encoding row names the histogram's axis "Count", as the full graph does.
    y_title = (
        str(figure_settings.get("encoded_histogram_y_title", "Count"))
        if encoded
        else "Sequences"
    )
    histogram = fitness_histogram_chart(
        inputs.fitness_values, domain, figure_settings, rendering, y_title=y_title
    )
    tree_width = int(
        figure_settings.get("tree", {}).get(
            "width", rendering.get("tree_width", TREE_WIDTH)
        )
    )
    spacer = fitness_histogram_chart(
        inputs.fitness_values,
        domain,
        figure_settings,
        rendering,
        hidden_width=tree_width,
        y_title=y_title,
    )
    row = alt.hconcat(
        alt.vconcat(spacer, tree, spacing=gap).resolve_scale(color="independent"),
        alt.vconcat(histogram, diagram, spacing=gap).resolve_scale(
            x="shared", color="independent"
        ),
        spacing=gap,
    ).resolve_scale(color="independent")
    return row, count


def vanilla_analysis(tree_analysis: Any, annotated: bool) -> Any:
    """The tree analysis, without its wild-type marker table unless ``annotated``.

    The vanilla style draws no wild-type star: ``tree_chart`` skips it when the
    analysis carries no marker table.
    """
    if annotated:
        return tree_analysis
    return SimpleNamespace(
        **{k: v for k, v in _attributes(tree_analysis).items() if k != "wild_type_marker_table"}
    )


def _attributes(obj: Any) -> dict[str, Any]:
    """Public data attributes of ``obj``, whether it has a ``__dict__`` or only slots."""
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    found = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        value = getattr(obj, name, None)
        if not callable(value):
            found[name] = value
    return found


def configure_view_chart(
    chart: Any, figure_settings: dict[str, Any], rendering: dict[str, Any]
) -> Any:
    """The fonts, axis, legend and title configuration shared by the D10 tree views."""
    font_size = int(
        figure_settings.get("tree", {}).get("font_size", rendering.get("font_size", 15))
    )
    return (
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
