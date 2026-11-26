import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    from dubanevics2026topological.analysis import load_settings
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_graph_tree_persistence_product,
    )
    from dubanevics2026topological.figures.graph_tree_persistence import (
        FIGURE_ID,
        build_graph_tree_persistence_figure,
        build_graph_tree_persistence_preview,
    )

    settings = load_settings()
    construction = load_graph_tree_persistence_product(settings=settings)[
        "construction"
    ]
    return (
        FIGURE_ID,
        ROOT,
        build_graph_tree_persistence_figure,
        build_graph_tree_persistence_preview,
        construction,
        mo,
        settings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Graph · Tree · Persistence

    This four-stage figure shows how a decreasing sweep builds a split tree from a
    sampled graph, and how the same sweep pairs peaks in a persistence diagram.
    The smooth landscape is a conceptual example, not a claim that biological
    sequence space is a smooth surface.

    The manuscript figure below uses the configured fixed camera and four retained
    stages (Graph, Tree, and Persistence rows). The controls change presentation
    and the single-stage preview; they do not rebuild the scientific construction.
    """)


@app.cell
def _(mo, settings):
    control_appearance = settings["figures"]["figures"]["graph_tree_persistence"]
    rendering = settings["figures"]["rendering"]
    checkpoint_names = control_appearance["stage_titles"]
    checkpoint = mo.ui.dropdown(
        options=[*checkpoint_names, "Custom threshold"],
        value=checkpoint_names[0],
        label="Preview checkpoint",
    )
    threshold = mo.ui.slider(
        0.0,
        1.0,
        step=0.001,
        value=0.5,
        label="Custom sweep threshold",
        show_value=True,
    )
    panel_width = mo.ui.slider(
        160, 320, value=int(control_appearance["panel_width"]), label="Panel width"
    )
    panel_height = mo.ui.slider(
        130, 260, value=int(control_appearance["panel_height"]), label="Panel height"
    )
    regular_tree_size = mo.ui.slider(
        40,
        140,
        value=int(control_appearance["tree_regular_node_size"]),
        label="Regular tree-node size",
    )
    overall_font_size = mo.ui.slider(
        10,
        22,
        value=int(rendering["font_size"]),
        label="Overall font size",
    )
    stage_title_size = mo.ui.slider(
        14,
        26,
        value=int(control_appearance["heading_font_size"]),
        label="Stage title size",
    )
    controls = mo.vstack(
        [
            mo.hstack([checkpoint, threshold]),
            mo.md("Choose **Custom threshold** to move the plane with the slider."),
            mo.hstack([panel_width, panel_height, regular_tree_size]),
            mo.hstack([overall_font_size, stage_title_size]),
        ]
    )
    controls
    return (
        overall_font_size,
        panel_height,
        panel_width,
        regular_tree_size,
        stage_title_size,
        checkpoint,
        threshold,
    )


@app.cell
def _(
    FIGURE_ID,
    overall_font_size,
    panel_height,
    panel_width,
    regular_tree_size,
    settings,
    stage_title_size,
):
    base_appearance = settings["figures"]["figures"][FIGURE_ID]
    base_rendering = settings["figures"]["rendering"]
    appearance = {
        **base_appearance,
        "panel_width": panel_width.value,
        "panel_height": panel_height.value,
        "tree_regular_node_size": regular_tree_size.value,
        "heading_font_size": stage_title_size.value,
    }
    preview_settings = {
        **settings,
        "figures": {
            **settings["figures"],
            "rendering": {**base_rendering, "font_size": overall_font_size.value},
            "figures": {
                **settings["figures"]["figures"],
                FIGURE_ID: appearance,
            },
        },
    }
    preview_settings
    return (preview_settings,)


@app.cell
def _(checkpoint, construction, mo, settings, threshold):
    threshold_by_checkpoint = dict(
        zip(
            settings["figures"]["figures"]["graph_tree_persistence"]["stage_titles"],
            construction.thresholds[1:],
            strict=True,
        )
    )
    selected_threshold = (
        float(threshold.value)
        if checkpoint.value == "Custom threshold"
        else float(threshold_by_checkpoint[checkpoint.value])
    )
    mo.md(
        f"Preview sweep threshold: **{selected_threshold:.12g}**. "
        "Checkpoint choices use their exact stored scalar values."
    )
    return (selected_threshold,)


@app.cell
def _(build_graph_tree_persistence_figure, construction, mo, preview_settings):
    manuscript_chart = build_graph_tree_persistence_figure(
        construction, preview_settings
    )
    mo.vstack(
        [
            mo.md("## Four-stage manuscript figure"),
            mo.ui.altair_chart(manuscript_chart, chart_selection=False),
        ]
    )
    return (manuscript_chart,)


@app.cell
def _(
    build_graph_tree_persistence_preview,
    construction,
    mo,
    preview_settings,
    selected_threshold,
):
    preview_chart = build_graph_tree_persistence_preview(
        construction, selected_threshold, preview_settings
    )
    mo.vstack(
        [
            mo.md("## Single-stage preview"),
            mo.ui.altair_chart(preview_chart, chart_selection=False),
        ]
    )
    return (preview_chart,)


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export manuscript SVG and 300-dpi PNG")
    export_button
    return (export_button,)


@app.cell
def _(ROOT, export_button, manuscript_chart, mo, settings):
    mo.stop(not export_button.value)
    destination = ROOT / "output/figures"
    destination.mkdir(parents=True, exist_ok=True)
    svg_path = destination / "graph_tree_persistence.svg"
    png_path = destination / "graph_tree_persistence.png"
    manuscript_chart.save(svg_path)
    manuscript_chart.save(png_path, ppi=settings["figures"]["png_dpi"])
    mo.md(f"Exported the manuscript figure to `{destination.relative_to(ROOT)}`.")


if __name__ == "__main__":
    app.run()
