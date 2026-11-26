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
    from dubanevics2026topological.analysis import (
        load_cached_clamped_comparison,
        load_settings,
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

    FIGURE_ID = "study_tree_persistence_metrics"
    settings = load_settings()
    # The labels are read from the manuscript quantity registry.
    analysis = {
        **load_cached_clamped_comparison(),
        "manuscript_quantities": load_current_manuscript_quantities(ROOT / "output"),
    }
    return (
        FIGURE_ID,
        ROOT,
        analysis,
        get_spec,
        mo,
        preview_settings,
        render_payload,
        save_appearance_settings,
        settings,
    )


@app.cell
def _(mo):
    mo.md("""
    # Published-threshold tree and persistence metrics

    This figure renders the cached Papkou2023 published-threshold landscape at -0.507774.
    The three rows color the split tree and its ordinary H0 persistence points by
    branch vertex count, amino-acid Hamming distance, and Horton–Strahler order.
    Controls below change appearance only; they do not recompute the landscape.
    """)


@app.cell
def _(FIGURE_ID, mo, settings):
    figure = settings["figures"]["figures"][FIGURE_ID]
    tree = figure["tree"]
    tree_width = mo.ui.slider(250, 1200, value=tree["width"], label="Tree width")
    tree_height = mo.ui.slider(180, 900, value=tree["height"], label="Tree height")
    font_size = mo.ui.slider(
        9,
        22,
        value=figure.get("font_size", settings["figures"]["rendering"]["font_size"]),
        label="Figure text size",
    )
    tree_controls_view = mo.vstack([tree_width, tree_height, font_size])
    tree_controls_view
    return font_size, tree_height, tree_width


@app.cell
def _(FIGURE_ID, mo, settings):
    persistence = settings["figures"]["figures"][FIGURE_ID]["persistence"]
    persistence_width = mo.ui.slider(
        180, 1000, value=persistence["width"], label="Persistence panel width"
    )
    persistence_height = mo.ui.slider(
        180, 900, value=persistence["height"], label="Persistence panel height"
    )
    point_size = mo.ui.slider(
        25, 160, value=persistence["point_size"], label="Persistence point size"
    )
    persistence_controls_view = mo.vstack(
        [persistence_width, persistence_height, point_size]
    )
    persistence_controls_view
    return persistence_height, persistence_width, point_size


@app.cell
def _(
    FIGURE_ID,
    analysis,
    font_size,
    get_spec,
    mo,
    persistence_height,
    persistence_width,
    point_size,
    preview_settings,
    render_payload,
    settings,
    tree_height,
    tree_width,
):
    figure_settings = settings["figures"]["figures"][FIGURE_ID]
    appearance = {
        "tree": {
            **figure_settings["tree"],
            "width": tree_width.value,
            "height": tree_height.value,
        },
        "font_size": font_size.value,
        "persistence": {
            **figure_settings["persistence"],
            "width": persistence_width.value,
            "height": persistence_height.value,
            "point_size": point_size.value,
        },
    }
    preview_config = preview_settings(settings, FIGURE_ID, appearance)
    chart = render_payload(get_spec(FIGURE_ID), analysis, preview_config, "vanilla")
    preview = mo.ui.altair_chart(chart)
    preview
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
    mo.md(
        f"Saved `{saved.name}`. Run `make study-metrics-figure` or keep `make watch` running."
    )


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export local preview SVG and PNG")
    export_button
    return (export_button,)


@app.cell
def _(FIGURE_ID, ROOT, chart, export_button, mo, settings):
    mo.stop(not export_button.value)
    destination = ROOT / "output/previews"
    destination.mkdir(parents=True, exist_ok=True)
    chart.save(destination / f"{FIGURE_ID}.svg")
    chart.save(destination / f"{FIGURE_ID}.png", ppi=settings["figures"]["png_dpi"])
    mo.md(f"Exported the preview to `{destination.relative_to(ROOT)}`.")


if __name__ == "__main__":
    app.run()
