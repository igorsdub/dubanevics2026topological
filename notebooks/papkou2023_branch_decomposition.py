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
    from dubanevics2026topological.analysis.branch_decomposition import (
        load_cached_branch_decomposition,
        order_summary,
    )
    from dubanevics2026topological.figures.branch_decomposition_charts import (
        appearance_settings,
        branch_facets,
        order_histograms,
        save_pngs,
    )

    settings = load_settings()
    appearance = appearance_settings(settings)
    try:
        result = load_cached_branch_decomposition(settings)
    except FileNotFoundError as error:
        mo.stop(True, mo.md(f"**{error}**"))
    return (
        ROOT,
        appearance,
        branch_facets,
        mo,
        order_histograms,
        order_summary,
        result,
        save_pngs,
        settings,
    )


@app.cell
def _(mo, result):
    info = result.summary
    mo.md(f"""
    # Branch decomposition of the {info["settings"]["tree"]} tree

    Exploratory analysis, not a publication figure. The stage
    (`just branch-decomposition`) lists every edge of the smoothed Candidate
    split tree as one branch: the stretch between two adjacent critical points, or
    between a critical point and the root. A branch is named by the sequence at its
    upper endpoint (a peak for a terminal branch, a saddle for an internal one) and
    carries the sequences on the edge, including the sequences of absorbed hills
    that the smoothing assigned to it. The Horton–Strahler order is computed on the
    smoothed tree itself, so an absorbed hill no longer raises it, and a branch has
    order 1 exactly when it is terminal.

    **{info["branch_count"]:,} branches** ({info["terminal_branch_count"]:,}
    terminal, {info["internal_branch_count"]:,} internal) carry
    **{info["member_count"]:,} sequences**; the root sequence
    (`{info["root_sequence"]}`) belongs to none. A branch's span is the fitness
    drop between its endpoints (the glossary's branch span), a different quantity
    from ordinary H0 persistence. Its value coincides with the persistence of the
    branch's peak on a terminal branch whose peak dies at the saddle where the
    branch ends, and never on an internal branch or on the terminal branch of an
    elder arm.
    """)
    return (info,)


@app.cell
def _(mo, order_summary, result):
    order_table = order_summary(result.branches)
    mo.vstack(
        [
            mo.md("### Branches per Horton–Strahler order"),
            mo.ui.table(order_table, selection=None),
        ]
    )
    return (order_table,)


@app.cell
def _(appearance, mo, order_table):
    _orders = [str(order) for order in order_table["horton_strahler_order"].to_list()]
    order_select = mo.ui.multiselect(
        options=_orders, value=_orders, label="Horton–Strahler orders shown"
    )
    facet_flag = mo.ui.checkbox(
        value=bool(appearance["show_branch_facets"]),
        label="Per-branch facets (one fitness histogram per branch)",
    )
    mo.vstack([order_select, facet_flag])
    return facet_flag, order_select


@app.cell
def _(appearance, mo, order_histograms, order_select, result, settings):
    selected_orders = [int(order) for order in order_select.value]
    mo.stop(not selected_orders, mo.md("Select at least one order."))
    histograms = order_histograms(
        result.branches, result.members, selected_orders, settings
    )
    _scale_note = (
        "The count axis of the fitness histograms is symmetric-log (ticks at 0 "
        "and the powers of ten), so a few functional sequences show beside "
        "thousands of low-fitness ones; bar heights are not proportional to "
        "counts. `log_count_axis = false` in `config/figures.toml` draws it "
        "linear."
        if appearance["log_count_axis"]
        else ""
    )
    mo.vstack(
        [
            mo.md(
                "### Fitness of the sequences and vertex counts of the branches, "
                "per order"
            ),
            mo.md(_scale_note),
            histograms,
        ]
    )
    return histograms, selected_orders


@app.cell
def _(branch_facets, facet_flag, mo, result, selected_orders, settings):
    if facet_flag.value:
        facets = branch_facets(
            result.branches, result.members, selected_orders, settings
        )
        facet_view = mo.vstack(
            [mo.md("### Fitness of the sequences, per branch"), facets]
        )
    else:
        facets = None
        facet_view = mo.md("Tick *Per-branch facets* to draw one panel per branch.")
    facet_view
    return (facets,)


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save PNGs")
    save_button
    return (save_button,)


@app.cell
def _(ROOT, appearance, facets, histograms, mo, save_button, save_pngs, settings):
    mo.stop(not save_button.value)
    _charts = {"branch_decomposition_orders": histograms}
    if facets is not None:
        _charts["branch_decomposition_facets"] = facets
    _directory = ROOT / appearance["png_directory"]
    _written = save_pngs(_charts, _directory, settings)
    mo.md(
        "Saved "
        + ", ".join(f"`{path.relative_to(ROOT)}`" for path in _written)
        + " (output/ is gitignored)."
    )


if __name__ == "__main__":
    app.run()
