import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from dubanevics2026topological.analysis import load_settings
    from dubanevics2026topological.analysis.candidate_edge_evidence import (
        load_cached_candidate_edge_evidence,
    )
    from dubanevics2026topological.figures import get_spec, render_payload
    from dubanevics2026topological.figures.candidate_edge_evidence import (
        build_selected_edge_detail_figure,
        select_edges,
        selected_edge_rows,
    )

    FIGURE_ID = "candidate_edge_evidence"
    settings = load_settings()
    result = load_cached_candidate_edge_evidence()
    return (
        FIGURE_ID,
        build_selected_edge_detail_figure,
        get_spec,
        mo,
        render_payload,
        result,
        select_edges,
        selected_edge_rows,
        settings,
    )


@app.cell
def _(mo, result):
    diagnostics = result["diagnostics"]
    explanation = mo.md(r"""
    # Candidate mutation-edge evidence

    Each row in the cached table represents one undirected mutation edge in the
    Candidate graph. The overview summarizes the q-value band selected below:
    a linear q-value histogram includes both 0 and 1, and the curve counts edges
    with **q > alpha** at sampled exploratory thresholds. The lower plots show
    absolute raw-fitness difference by q-value and fitness-interval coverage by
    q-value. Each interval contributes once to every fitness bin it
    intersects, so coverage counts overlap across fitness bins and must not be
    summed as a total edge count. No edge-level points are sampled into these
    overviews.

    Alpha starts at 0 as an exploratory control; there is no canonical value.
    It does not alter cached p-values or q-values. q-value bounds narrow the
    evidence table while retaining every edge in that band, with a `qualifies`
    flag for the strict q > alpha rule. The optional edge ID field can restrict
    the table to exact stable edge identities. Select table rows to draw their
    effect sizes and raw-fitness intervals. At most 100 selected rows are drawn
    at once; all matching evidence rows remain in the paginated table.
    """)
    _diagnostic_summary = mo.ui.table(
        [
            {"Diagnostic": "Candidate vertices", "Value": diagnostics["vertex_count"]},
            {
                "Diagnostic": "Candidate mutation edges",
                "Value": diagnostics["edge_count"],
            },
            {
                "Diagnostic": "Global BH comparison family",
                "Value": diagnostics["global_bh_family_size"],
            },
            {
                "Diagnostic": "Candidate edge sign disagreements",
                "Value": diagnostics["candidate_edge_sign_disagreement_count"],
            },
        ]
    )
    mo.vstack([explanation, _diagnostic_summary])


@app.cell
def _(mo):
    q_lower = mo.ui.slider(0.0, 1.0, step=0.01, value=0.0, label="Minimum q-value")
    q_upper = mo.ui.slider(0.0, 1.0, step=0.01, value=1.0, label="Maximum q-value")
    alpha = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.0,
        label="Exploratory alpha (q > alpha; initial value 0)",
    )
    edge_ids_text = mo.ui.text_area(label="Optional edge IDs, one per line", rows=4)
    mo.vstack([q_lower, q_upper, alpha, edge_ids_text])
    return alpha, edge_ids_text, q_lower, q_upper


@app.cell
def _(
    FIGURE_ID,
    get_spec,
    mo,
    q_lower,
    q_upper,
    render_payload,
    result,
    settings,
):
    import polars as _overview_polars

    if q_lower.value > q_upper.value:
        q_range_edges = result["edges"].head(0)
        overview = mo.md("Minimum q-value must not exceed maximum q-value.")
    else:
        q_range_edges = result["edges"].filter(
            _overview_polars.col("q_value").is_between(
                q_lower.value, q_upper.value, closed="both"
            )
        )
    if q_lower.value <= q_upper.value and q_range_edges.is_empty():
        overview = mo.md("No edges fall inside the selected q-value band.")
    elif q_lower.value <= q_upper.value:
        overview_result = {**result, "edges": q_range_edges}
        overview = render_payload(
            get_spec(FIGURE_ID), overview_result, settings, "vanilla"
        )
    mo.vstack(
        [
            mo.md(
                f"**Overview edges in selected q-value band:** {q_range_edges.height:,} "
                f"/ **full Candidate edge family:** {result['edges'].height:,}"
            ),
            overview,
        ]
    )


@app.cell
def _(alpha, edge_ids_text, mo, q_lower, q_upper, result, select_edges):
    import polars as _selection_polars

    ids = [value.strip() for value in edge_ids_text.value.splitlines() if value.strip()]
    if q_lower.value > q_upper.value:
        selected = (
            result["edges"]
            .head(0)
            .with_columns(_selection_polars.lit(False).alias("qualifies"))
        )
        control_message = mo.md("Minimum q-value must not exceed maximum q-value.")
    else:
        selected = select_edges(
            result["edges"],
            q_lower.value,
            q_upper.value,
            alpha.value,
            edge_ids=ids or None,
        )
        control_message = mo.md(
            f"**Edges in q-value band:** {selected.height:,} · "
            f"**q > alpha:** {int(selected['qualifies'].sum()):,}"
        )
    display_columns = [
        "edge_id",
        "dna_left",
        "dna_right",
        "qualifies",
        "reported_raw_sign_disagreement",
        "fitness_left",
        "fitness_right",
        "fitness_standard_error_left",
        "fitness_standard_error_right",
        "fitness_lower",
        "fitness_upper",
        "fitness_delta",
        "absolute_fitness_delta",
        "contrast_id",
        "reported_dna_from",
        "reported_dna_to",
        "reported_delta",
        "pairwise_standard_error",
        "test_statistic",
        "p_value",
        "q_value",
    ]
    columns = [column for column in display_columns if column in selected.columns]
    row_table = mo.ui.table(
        selected.select(columns),
        selection="multi",
        pagination=True,
        page_size=25,
        label="Select edges to draw their effect sizes and fitness intervals",
    )
    mo.vstack(
        [
            control_message,
            mo.md("### Candidate edge identities and evidence"),
            row_table,
        ]
    )
    return row_table, selected


@app.cell
def _(
    alpha,
    build_selected_edge_detail_figure,
    mo,
    row_table,
    selected,
    selected_edge_rows,
):
    chosen = selected_edge_rows(row_table.value)
    if chosen.height > 100:
        detail_view = mo.md(
            "Select no more than 100 table rows to draw interval segments."
        )
        displayed_count = 0
    elif chosen.is_empty():
        detail_view = mo.md(
            "Select one or more evidence rows to view effect sizes and fitness intervals."
        )
        displayed_count = 0
    else:
        detail_view = build_selected_edge_detail_figure(chosen)
        displayed_count = chosen.height
    caption = mo.md(
        f"**Selected edges:** {chosen.height:,} · "
        f"**interval segments displayed:** {displayed_count:,} · "
        f"exploratory alpha {alpha.value:.2f} (qualifies when q > alpha)"
    )
    mo.vstack([caption, detail_view])


if __name__ == "__main__":
    app.run()
