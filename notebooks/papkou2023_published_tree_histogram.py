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
    from dubanevics2026topological.build import load_figure_inputs
    from dubanevics2026topological.figures import (
        compose_payload,
        get_spec,
        preview_settings,
        render_payload,
        save_appearance_settings,
    )

    FIGURE_ID = "published_tree_histogram"
    settings = load_settings()
    spec = get_spec(FIGURE_ID)
    comparison = compose_payload(spec, load_figure_inputs((spec,), settings), settings)
    return (
        FIGURE_ID,
        ROOT,
        comparison,
        get_spec,
        mo,
        preview_settings,
        render_payload,
        save_appearance_settings,
        settings,
    )


@app.cell
def _(comparison, mo):
    diagnostic = comparison["diagnostics"]["study_clamped"]
    mo.md(
        f"""
        # Published tree and raw-fitness population

        The histogram counts every vertex in the selected Published graph once,
        including retained fringe and noncritical vertices. It uses raw fitness
        and 200 bins across the population's observed range. The reduced split tree shows critical structure; its branch colors
        encode branch vertex count. The red star marks wild type.

        Published graph: {diagnostic["selected_component_vertex_count"]:,} vertices.
        """
    )


@app.cell
def _(FIGURE_ID, mo, settings):
    appearance = settings["figures"]["figures"][FIGURE_ID]
    tree_width = mo.ui.slider(
        300, 800, value=appearance["tree"]["width"], label="Tree width"
    )
    histogram_width = mo.ui.slider(
        100, 260, value=appearance["histogram_width"], label="Histogram width"
    )
    height = mo.ui.slider(
        160, 600, value=appearance["tree"]["height"], label="Panel height"
    )
    controls = mo.vstack([tree_width, histogram_width, height])
    controls
    return tree_width, histogram_width, height


@app.cell
def _(
    FIGURE_ID,
    comparison,
    get_spec,
    height,
    histogram_width,
    mo,
    preview_settings,
    render_payload,
    settings,
    tree_width,
):
    original = settings["figures"]["figures"][FIGURE_ID]
    preview_figure_settings = {
        **original,
        "tree": {
            **original["tree"],
            "width": tree_width.value,
            "height": height.value,
        },
        "histogram_width": histogram_width.value,
    }
    preview_config = preview_settings(settings, FIGURE_ID, preview_figure_settings)
    chart = render_payload(get_spec(FIGURE_ID), comparison, preview_config, "annotated")
    mo.ui.altair_chart(chart)
    return chart, preview_figure_settings


@app.cell
def _(FIGURE_ID, mo):
    save_button = mo.ui.run_button(label="Save appearance settings")
    save_button
    return (save_button,)


@app.cell
def _(
    FIGURE_ID,
    mo,
    preview_figure_settings,
    save_appearance_settings,
    save_button,
    settings,
):
    mo.stop(not save_button.value)
    saved = save_appearance_settings(FIGURE_ID, settings, preview_figure_settings)
    mo.md(
        f"Saved `{saved.name}`. Run `make published-histogram-figure` or keep `make watch` running."
    )


if __name__ == "__main__":
    app.run()
