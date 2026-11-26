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
    from dubanevics2026topological.analysis import load_cached_analysis, load_settings
    from dubanevics2026topological.analysis.component_persistence import (
        load_cached_component_persistence,
    )
    from dubanevics2026topological.figures import (
        get_spec,
        preview_settings,
        render_payload,
        save_appearance_settings,
    )
    from dubanevics2026topological.figures._quantities import (
        load_current_manuscript_quantities,
    )
    from dubanevics2026topological.landscape.full_population import (
        load_full_network_fitness,
    )

    FIGURE_ID = "full_tree_persistence"
    settings = load_settings()
    pairs, diagnostics = load_cached_component_persistence()
    component_data = {
        "analysis": load_cached_analysis(),
        "pairs": pairs,
        "diagnostics": diagnostics,
        "full_fitness": load_full_network_fitness(settings),
        "manuscript_quantities": load_current_manuscript_quantities(ROOT / "output"),
    }
    return (
        FIGURE_ID,
        ROOT,
        component_data,
        diagnostics,
        get_spec,
        mo,
        preview_settings,
        render_payload,
        save_appearance_settings,
        settings,
    )


@app.cell
def _(diagnostics, mo):
    mo.md("""
    # Full graph: tree and persistence

    Two panels share one fitness axis. The left panel is the raw split tree of
    the full graph, coloured by branch vertex count; its colour bar sits inside
    the tree. The right panel is the persistence diagram: x is fitness at death
    and y is fitness at birth. Peaks (upward triangles, above the diagonal) pair
    with split saddles and read as the split tree; no valley markers are drawn.
    The global maximum-to-minimum pair is one of the peaks. The marginal
    histogram above it (`histogram_bins` bins) and the dotted median guide use all
    full-network raw fitness values; there is no right-hand histogram. Crimson
    (`#a50f15`) marks identify a represented birth extremum containing at least
    one stop codon.

    The grey square on the diagram is the topological noise band, from the global
    minimum fitness `g` to its reflection through the median, `m + (m - g)`, on
    both axes; the horizontal rule at that value is the functional threshold. The
    band assumes non-functional variation is symmetric about the median and is
    not a calibrated error estimate.

    Branch span, ordinary H0 persistence and extended persistence (not computed)
    remain distinct quantities: the diagram is labelled simply "Persistence",
    and the global pair is the full-network sweep's `extended_positive` pair.
    Relative H1 describes component connectivity in the later phase; it does not
    count independent graph cycles.

    The separate threshold diagnostic below is unchanged and still reports the
    original ascending checkpoint's ordinary-pair death distribution and
    join-tree trailing saddle. The Candidate functional threshold is computed from the
    Full-graph raw fitness, not from this checkpoint.
    """)
    diagnostic_rows = [
        {
            "quantity": "Ascending ordinary pair count",
            "value": diagnostics.get("ordinary_pair_count"),
        },
        {
            "quantity": "Join-tree trailing saddle",
            "value": diagnostics.get("join_tree_trailing_saddle"),
        },
        {"quantity": "Global minimum", "value": diagnostics.get("global_minimum")},
    ]
    mo.ui.table(diagnostic_rows)
    return (diagnostic_rows,)


@app.cell
def _(diagnostics, mo):
    frequency_rows = diagnostics.get("ordinary_death_value_frequencies", [])
    mo.md("Ascending ordinary-pair death-value frequencies")
    mo.ui.table(frequency_rows)


@app.cell
def _(FIGURE_ID, mo, settings):
    figure_config = settings["figures"]["figures"][FIGURE_ID]
    persistence = figure_config["persistence"]
    variant = mo.ui.dropdown(
        options=["annotated", "vanilla"], value="annotated", label="Figure variant"
    )
    tree_width = mo.ui.slider(
        250, 900, value=figure_config["tree"]["width"], label="Tree width"
    )
    height = mo.ui.slider(
        180, 900, value=figure_config["tree"]["height"], label="Panel height"
    )
    persistence_width = mo.ui.slider(
        180, 900, value=persistence["width"], label="Persistence width"
    )
    bins = mo.ui.slider(
        10, 300, value=figure_config["histogram_bins"], label="Histogram bins"
    )
    ordinary_size = mo.ui.slider(
        20,
        140,
        value=persistence["ordinary_point_size"],
        label="Peak point size",
    )
    mo.vstack(
        [
            variant,
            tree_width,
            height,
            persistence_width,
            bins,
            ordinary_size,
        ]
    )
    return (
        bins,
        height,
        ordinary_size,
        persistence_width,
        tree_width,
        variant,
    )


@app.cell
def _(
    FIGURE_ID,
    bins,
    component_data,
    get_spec,
    height,
    mo,
    ordinary_size,
    persistence_width,
    preview_settings,
    render_payload,
    settings,
    tree_width,
    variant,
):
    figure_settings = settings["figures"]["figures"][FIGURE_ID]
    appearance = {
        "histogram_bins": bins.value,
        "tree": {
            **figure_settings["tree"],
            "width": tree_width.value,
            "height": height.value,
        },
        "persistence": {
            **figure_settings["persistence"],
            "width": persistence_width.value,
            "height": height.value,
            "ordinary_point_size": ordinary_size.value,
        },
    }
    preview_config = preview_settings(settings, FIGURE_ID, appearance)
    chart = render_payload(
        get_spec(FIGURE_ID), component_data, preview_config, variant.value
    )
    mo.ui.altair_chart(chart)
    return appearance, chart


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save appearance settings")
    save_button
    return (save_button,)


@app.cell
def _(FIGURE_ID, appearance, mo, save_appearance_settings, save_button, settings):
    mo.stop(not save_button.value)
    saved = save_appearance_settings(FIGURE_ID, settings, appearance)
    mo.md(f"Saved `{saved.name}`. Run `make full-tree-figure` to refresh exports.")


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export local preview SVG and PNG")
    export_button
    return (export_button,)


@app.cell
def _(FIGURE_ID, ROOT, chart, export_button, mo, settings, variant):
    mo.stop(not export_button.value)
    destination = ROOT / "output/previews"
    destination.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_ID if variant.value == "annotated" else f"{FIGURE_ID}_vanilla"
    chart.save(destination / f"{stem}.svg")
    chart.save(destination / f"{stem}.png", ppi=settings["figures"]["png_dpi"])
    mo.md(f"Exported the {variant.value} preview to `{destination.relative_to(ROOT)}`.")


if __name__ == "__main__":
    app.run()
