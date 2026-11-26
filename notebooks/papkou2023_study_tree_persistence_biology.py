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
    from dubanevics2026topological.analysis.clamped_cache import (
        load_cached_clamped_comparison,
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

    FIGURE_ID = "study_tree_persistence_biology"
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
    # Published-threshold biological annotations

    This figure renders the cached Papkou2023 published-threshold landscape at -0.507774.
    Both rows show the same split tree and persistence diagram. The upper row
    is headed `43 stop-codon containing peaks` and highlights only those
    maxima with dark crimson (`#a50f15`) X marks on the tree and matching solid
    crimson triangles on persistence. Its count is among the 514 peak births, not
    every stop-codon containing sequence or critical point. The lower row is headed
    `111 synonymous leaves` and highlights whole terminal leaf branches
    whose birth amino-acid sequence matches the first adjacent descending
    saddle, with matching green persistence triangles. The headings count the
    highlighted populations dynamically.
    The tree and all unhighlighted persistence points remain gray; the red WT
    star is retained. The persistence comparison uses the adjacent saddle,
    independently of the persistence death partner.
    Both rows use one shared raw-fitness vertical domain, with fitness ticks
    and its title at the left edge. Persistence y-axes are hidden; only the
    lower persistence panel shows the Death fitness x-axis.
    Tree and persistence panels share one height so their fitness scales align.
    Controls below change appearance only; they do not recompute the landscape.
    """)


@app.cell
def _(FIGURE_ID, mo, settings):
    figure = settings["figures"]["figures"][FIGURE_ID]
    tree = figure["tree"]
    tree_width = mo.ui.slider(250, 1200, value=tree["width"], label="Tree width")
    tree_height = mo.ui.slider(180, 900, value=tree["height"], label="Panel height")
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
    point_size = mo.ui.slider(
        25, 160, value=persistence["point_size"], label="Persistence point size"
    )
    persistence_controls_view = mo.vstack([persistence_width, point_size])
    persistence_controls_view
    return persistence_width, point_size


@app.cell
def _(
    FIGURE_ID,
    analysis,
    font_size,
    get_spec,
    mo,
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
            "height": tree_height.value,
            "point_size": point_size.value,
        },
    }
    preview_config = preview_settings(settings, FIGURE_ID, appearance)
    chart = render_payload(get_spec(FIGURE_ID), analysis, preview_config, "annotated")
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
        f"Saved `{saved.name}`. Run `make study-biology-figure` or keep `make watch` running."
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
