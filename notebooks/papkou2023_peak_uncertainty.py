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
    from dubanevics2026topological.build.peak_panels import load_peak_panels
    from dubanevics2026topological.build.presentation_io import (
        figure_path,
        save_preview,
    )
    from dubanevics2026topological.figures import (
        compose_payload,
        get_spec,
        preview_settings,
        render_payload,
        save_appearance_settings,
    )
    from dubanevics2026topological.results import RunContext

    FIGURE_ID = "peak_uncertainty"
    context = RunContext(root=ROOT, output_root="output")
    settings = load_settings()
    # The four graphs are read as saved: each variant's pooled_scatter and
    # peak_panel_summary under output/variants/<variant>/ (`just peak-panels-product`);
    # nothing is recomputed.
    panels, _panel_ids = load_peak_panels(context, settings)
    peak_data = compose_payload(
        get_spec(FIGURE_ID),
        {"saved_figure_inputs": {"peak_panels": panels}},
        settings,
    )
    return (
        FIGURE_ID,
        ROOT,
        context,
        figure_path,
        get_spec,
        mo,
        peak_data,
        preview_settings,
        render_payload,
        save_preview,
        save_appearance_settings,
        settings,
    )


@app.cell
def _(mo, peak_data):
    baseline_rows = []
    for _variant, _panel in peak_data["panels"].items():
        baseline_rows.append(
            {
                "Variant": _variant,
                "Fitness threshold": _panel["diagnostics"]["threshold"],
                "Measured sequences shown": _panel["scatter"].height,
                "Raw-fitness local maxima": _panel["diagnostics"]["peak_count"],
            }
        )
    explanation = mo.md(r"""
    # Threshold-landscape peak uncertainty

    The figure compares fitness relative to the wild type with each sequence's
    pooled standard error (ADR 2026-10-22-pooled-standard-error-for-bootstrap), one panel per analysis variant (rows Published graph and Candidate graph, columns Nominal and BY q = 0.05); the wild type is the
    point at fitness 0. Black triangles are raw-fitness local maxima; gray points
    are the other measured sequences. The dashed line marks the landscape's
    functional threshold. A sequence is a local maximum when no neighbour has
    greater raw fitness; equal-fitness neighbours do not disqualify it.

    The peak counts against the adjusted q-value threshold are no longer drawn. They
    are still computed and cached: a maximum is counted when every neighbour has
    q-value less than or equal to the threshold, equivalent to requiring the minimum
    neighbour score, `1 − q`, to be at least the corresponding score threshold (for
    example, q = 0.05 corresponds to 0.95). This transformed score is not a
    posterior probability that a peak is real.

    Controls below change figure appearance only. They reuse the cached
    scientific result and do not recompute fitness comparisons or q-values.
    The canonical PNG is shown initially. After adjusting controls, click
    **Render appearance preview** to apply them to a full-data image.
    **Save appearance settings** updates the publication configuration.
    """)
    summary = mo.ui.table(baseline_rows)
    mo.vstack([explanation, summary])


@app.cell
def _(FIGURE_ID, mo, settings):
    figure = settings["figures"]["figures"][FIGURE_ID]
    panel_size = mo.ui.slider(220, 700, value=figure["panel_size"], label="Panel size")
    font_size = mo.ui.slider(9, 22, value=figure["font_size"], label="Text size")
    sequence_point_size = mo.ui.slider(
        4, 50, value=figure["sequence_point_size"], label="Sequence point size"
    )
    peak_point_size = mo.ui.slider(
        8, 200, value=figure["peak_point_size"], label="Peak point size"
    )
    sequence_opacity = mo.ui.slider(
        0.1, 1.0, step=0.05, value=figure["sequence_opacity"], label="Sequence opacity"
    )
    preview_button = mo.ui.run_button(label="Render appearance preview")
    controls = mo.vstack(
        [
            panel_size,
            font_size,
            sequence_point_size,
            peak_point_size,
            sequence_opacity,
            preview_button,
        ]
    )
    controls
    return (
        font_size,
        panel_size,
        peak_point_size,
        preview_button,
        sequence_opacity,
        sequence_point_size,
    )


@app.cell
def _(
    FIGURE_ID,
    font_size,
    get_spec,
    mo,
    panel_size,
    peak_data,
    peak_point_size,
    preview_settings,
    render_payload,
    sequence_opacity,
    sequence_point_size,
    settings,
):
    figure_settings = settings["figures"]["figures"][FIGURE_ID]
    appearance = {
        **figure_settings,
        "panel_size": panel_size.value,
        "font_size": font_size.value,
        "sequence_point_size": sequence_point_size.value,
        "peak_point_size": peak_point_size.value,
        "sequence_opacity": sequence_opacity.value,
    }
    preview_config = preview_settings(settings, FIGURE_ID, appearance)
    chart = render_payload(get_spec(FIGURE_ID), peak_data, preview_config, "vanilla")
    return appearance, chart


@app.cell
def _(FIGURE_ID, chart, context, figure_path, mo, preview_button, save_preview):
    canonical = figure_path(context, FIGURE_ID, "png")
    initial_view = (
        mo.image(canonical, alt="Study and candidate peak uncertainty", width=740)
        if canonical.is_file()
        else mo.md("Run `make peak-figure` or render an appearance preview below.")
    )
    mo.stop(not preview_button.value, initial_view)
    preview_path = save_preview(
        context, chart, f"{FIGURE_ID}_appearance", "png", ppi=96
    )
    mo.image(preview_path, alt="Current peak uncertainty appearance", width=740)


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save appearance settings")
    save_button
    return (save_button,)


@app.cell
def _(FIGURE_ID, appearance, mo, save_appearance_settings, save_button, settings):
    mo.stop(not save_button.value)
    saved = save_appearance_settings(FIGURE_ID, settings, appearance)
    mo.md(f"Saved `{saved.name}`. Run `make peak-figure` to refresh exports.")


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export local preview SVG and PNG")
    export_button
    return (export_button,)


@app.cell
def _(FIGURE_ID, ROOT, chart, context, export_button, mo, save_preview, settings):
    mo.stop(not export_button.value)
    destination = save_preview(context, chart, FIGURE_ID, "svg").parent
    save_preview(context, chart, FIGURE_ID, "png", ppi=settings["figures"]["png_dpi"])
    mo.md(f"Exported the preview to `{destination.relative_to(ROOT)}`.")


if __name__ == "__main__":
    app.run()
