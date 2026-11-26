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
    from dubanevics2026topological.build import (
        load_figure_inputs,
        require_current_registry,
    )
    from dubanevics2026topological.figures import (
        compose_payload,
        get_spec,
        preview_settings,
        render_payload,
        save_appearance_settings,
    )

    FIGURE_ID = "tree_comparison"
    settings = load_settings()
    require_current_registry(settings)
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
    mo.md("""
    # Papkou tree comparison

    The rows show the full graph with raw fitness, the Published-graph preparation
    selected at -0.507774, and the same selection at the reflected Candidate
    threshold. Each row pairs its tree with a 200-bin histogram of every selected
    graph vertex, using that population's observed fitness range. All three
    histograms share the same vertex-count scale. A vertex is functional when
    raw fitness is at least the threshold;
    original edges are kept when either endpoint is functional. This retains
    functional ties and the below-threshold fringe, but excludes fringe-to-fringe
    edges. The largest weakly connected component is selected. Trees and all
    plotted heights use raw fitness. Branch color represents branch vertex
    count. The red star marks wild type at its raw fitness in each landscape.

    The candidate functional threshold is exploratory and is not a calibrated uncertainty
    estimate. The cached diagnostics include the reference-match checks and
    excluded components. Controls below alter appearance only.
    """)
    rows = []
    display_names = {
        "full": "Full graph",
        "study_clamped": "Published graph",
        "candidate_clamped": "Candidate graph",
    }
    for name, display_name in display_names.items():
        diagnostic = comparison["diagnostics"].get(name, {})
        row = {
            "Landscape": display_name,
            "Vertices in selected graph": diagnostic.get(
                "selected_component_vertex_count"
            ),
            "Edges in selected graph": diagnostic.get("selected_component_edge_count"),
            "Connected components before selection": diagnostic.get("component_count"),
            "Excluded vertices": diagnostic.get("excluded_vertex_count"),
            "Local peaks": diagnostic.get("peak_count"),
        }
        validation = diagnostic.get("reference", {})
        if validation:
            row["Reference identity / edges / peaks"] = " / ".join(
                "match" if validation.get(key) else "mismatch"
                for key in (
                    "vertex_identities_match",
                    "undirected_edges_match",
                    "peak_identities_match",
                )
            )
        rows.append({key: value for key, value in row.items() if value is not None})
    mo.ui.table(rows)
    return (rows,)


@app.cell
def _(FIGURE_ID, mo, settings):
    tree = settings["figures"]["figures"][FIGURE_ID]["tree"]
    width = mo.ui.slider(300, 1000, value=tree["width"], label="Tree width")
    height = mo.ui.slider(160, 600, value=tree["height"], label="Panel height")
    histogram_width = mo.ui.slider(
        100,
        260,
        value=settings["figures"]["figures"][FIGURE_ID]["histogram_width"],
        label="Histogram width",
    )
    font_size = mo.ui.slider(
        9,
        22,
        value=tree.get("font_size", settings["figures"]["rendering"]["font_size"]),
        label="Axis text size",
    )
    controls = mo.vstack([width, height, font_size, histogram_width])
    controls
    return font_size, height, histogram_width, width


@app.cell
def _(
    FIGURE_ID,
    comparison,
    font_size,
    get_spec,
    height,
    histogram_width,
    mo,
    preview_settings,
    render_payload,
    settings,
    width,
):
    figure_settings = settings["figures"]["figures"][FIGURE_ID]
    tree_appearance = {
        **figure_settings["tree"],
        "width": width.value,
        "height": height.value,
        "font_size": font_size.value,
    }
    appearance = {"tree": tree_appearance, "histogram_width": histogram_width.value}
    preview_config = preview_settings(settings, FIGURE_ID, appearance)
    chart = render_payload(get_spec(FIGURE_ID), comparison, preview_config, "annotated")
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
    mo.md(
        f"Saved `{saved.name}`. Run `make comparison-figure` or keep `make watch` running."
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
    chart.save(
        destination / f"{FIGURE_ID}.png",
        ppi=settings["figures"]["png_dpi"],
    )
    mo.md(f"Exported the preview to `{destination.relative_to(ROOT)}`.")


if __name__ == "__main__":
    app.run()
