"""The Figure 6 rows describe the two actual selections."""

from dubanevics2026topological.figures.reflected_fdr_trees import ROWS


def test_row_order_and_selection_names():
    assert [row[0] for row in ROWS] == ["reflected", "reflected_by"]
    assert "point" in ROWS[0][1].lower()
    assert ROWS[1][1].lower().startswith("candidate threshold")
    assert "point" not in ROWS[1][1].lower()


def test_renderer_uses_each_rows_saved_threshold_and_selection(monkeypatch):
    from types import SimpleNamespace

    import altair as alt
    import polars as pl
    from dubanevics2026topological.figures import reflected_fdr_trees as module

    points = pl.DataFrame({"dna_sequence": ["a", "b"], "scalar": [-0.7, 0.3]})
    rows = {
        "reflected": {"layout": object(), "population": points, "threshold": -0.2, "method": "none", "q": None},
        "reflected_by": {"layout": object(), "population": points, "threshold": -0.3, "method": "by", "q": 0.05},
    }
    settings = {"figures": {"figures": {"tree_comparison": {}}, "rendering": {}}}
    chart = alt.Chart(pl.DataFrame({"x": [0], "y": [0]})).mark_point().encode(x="x:Q", y="y:Q")
    seen = []
    monkeypatch.setattr(module, "plot_data_with_metrics", lambda layout: SimpleNamespace(edge_table=pl.DataFrame({"vertex_count": [2]})))
    monkeypatch.setattr(module, "compute_fitness_histogram_table", lambda population, **kwargs: pl.DataFrame({"count": [2]}))
    monkeypatch.setattr(module, "tree_chart", lambda *args, **kwargs: chart)
    monkeypatch.setattr(module, "histogram_chart", lambda *args, **kwargs: chart)
    monkeypatch.setattr(module, "apply_tree_histogram_domain", lambda tree, domain: tree)
    monkeypatch.setattr(module, "overlay_count_colorbar", lambda tree, *args: tree)
    original = module.functional_threshold_layers
    label_texts = []

    def layers(threshold, **kwargs):
        seen.append(threshold)
        label_texts.append(kwargs.get("label_text", "default"))
        return original(threshold, **kwargs)

    monkeypatch.setattr(module, "functional_threshold_layers", layers)
    result = module.build_reflected_fdr_trees_figure({"rows": rows}, settings).to_dict()
    assert seen == [-0.2, -0.2, -0.3, -0.3]
    assert result["vconcat"][0]["hconcat"][0]["title"]["text"] == "Candidate threshold point estimate"
    assert result["vconcat"][1]["hconcat"][0]["title"]["text"] == "Candidate threshold BY, nominal q = 0.05"    # Each row draws only a dashed rule, no text: the threshold is named by an extra
    # three-decimal tick beside the three regular one-decimal ticks.
    assert label_texts == [None] * 4
    for row, threshold in zip(result["vconcat"], (-0.2, -0.3)):
        axis = row["hconcat"][0]["encoding"]["y"]["axis"]
        assert axis["values"] == sorted([-1.2, 0.0, 1.4, threshold])
        assert axis["format"] == ".1f"
        assert f"({threshold!r})" in axis["labelExpr"]
        assert ".3f" in axis["labelExpr"]
