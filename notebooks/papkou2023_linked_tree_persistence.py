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
    from dubanevics2026topological.figures import (
        get_spec,
        preview_settings,
        render_payload,
        save_appearance_settings,
    )
    from dubanevics2026topological.figures._quantities import (
        load_current_manuscript_quantities,
    )

    FIGURE_ID = "linked_tree_persistence"
    settings = load_settings()
    analysis = load_cached_analysis()
    # The title's counts are read from the manuscript quantity registry.
    payload = {
        "analysis": analysis,
        "manuscript_quantities": load_current_manuscript_quantities(ROOT / "output"),
    }
    return (
        FIGURE_ID,
        ROOT,
        get_spec,
        mo,
        payload,
        preview_settings,
        render_payload,
        save_appearance_settings,
        settings,
    )


@app.cell
def _(mo):
    mo.md("""
    # Tree and persistence view

    The tree and persistence panel use the same split-tree analysis. Both panels use vertex-count colors, with persistence points and no biological annotations other than the wild-type star.

    This view uses the full unfiltered sequence landscape (261,333 sequences). The published graph's filtered peak analysis reports 514 peaks, so counts from these two scopes are intentionally different.

    The saved analysis loads once. Sliders update this preview without recomputing the science. Saving appearance changes the checked-in TOML settings. Use the explicit preview export button for local SVG and PNG files.
    """)


@app.cell
def _(FIGURE_ID, mo, settings):
    tree = settings["figures"]["figures"][FIGURE_ID]["tree"]
    tree_width = mo.ui.slider(250, 1200, value=tree["width"], label="Tree width")
    tree_height = mo.ui.slider(180, 1000, value=tree["height"], label="Tree height")
    tree_font = mo.ui.slider(
        9,
        22,
        value=tree.get("font_size", settings["figures"]["rendering"]["font_size"]),
        label="Axis text size",
    )
    tree_controls_view = mo.vstack([tree_width, tree_height, tree_font])
    tree_controls_view
    return tree_font, tree_height, tree_width


@app.cell
def _(FIGURE_ID, mo, settings):
    persistence = settings["figures"]["figures"][FIGURE_ID]["persistence"]
    persistence_width = mo.ui.slider(
        180, 1000, value=persistence["width"], label="Persistence panel width"
    )
    persistence_height = mo.ui.slider(
        180, 1000, value=persistence["height"], label="Persistence panel height"
    )
    point_size = mo.ui.slider(
        25, 160, value=persistence["point_size"], label="Point size"
    )
    persistence_controls_view = mo.vstack(
        [persistence_width, persistence_height, point_size]
    )
    persistence_controls_view
    return persistence_height, persistence_width, point_size


@app.cell
def _(
    FIGURE_ID,
    get_spec,
    mo,
    payload,
    persistence_height,
    persistence_width,
    point_size,
    preview_settings,
    render_payload,
    settings,
    tree_font,
    tree_height,
    tree_width,
):
    figure_settings = settings["figures"]["figures"][FIGURE_ID]
    tree_appearance = {
        **figure_settings["tree"],
        "width": tree_width.value,
        "height": tree_height.value,
        "font_size": tree_font.value,
    }
    persistence_appearance = {
        **figure_settings["persistence"],
        "width": persistence_width.value,
        "height": persistence_height.value,
        "point_size": point_size.value,
    }
    appearance = {
        "tree": tree_appearance,
        "persistence": persistence_appearance,
    }
    preview_config = preview_settings(settings, FIGURE_ID, appearance)
    chart = render_payload(get_spec(FIGURE_ID), payload, preview_config, "vanilla")
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
    mo.md(f"Saved `{saved.name}`. Run `make figures` or keep `make watch` running.")


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
