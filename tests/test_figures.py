"""Tests for the figure registry seam: ``get_spec`` and ``render_payload``."""

from types import SimpleNamespace

import polars as pl
import pytest
from _figure_labels import label_quantities

from dubanevics2026topological import figures


def test_render_payload_renders_vanilla_only(monkeypatch):
    import altair as alt

    from dubanevics2026topological.figures import split_tree

    monkeypatch.setattr(
        split_tree,
        "plot_data_with_metrics",
        lambda tree: SimpleNamespace(edge_table=pl.DataFrame({"vertex_count": [1]})),
    )
    monkeypatch.setattr(split_tree, "count_domain", lambda edges: (1.0, 4.0))
    monkeypatch.setattr(
        split_tree,
        "persistence_with_counts",
        lambda *args: pl.DataFrame(
            {
                "birth_label": ["PEAK"],
                "death_label": ["ROOT"],
                "birth_scalar": [1.0],
                "death_scalar": [0.0],
            }
        ),
    )
    calls = []

    def unit_chart(title):
        return (
            alt.Chart(pl.DataFrame({"birth_scalar": [0.8], "death_scalar": [0.2]}))
            .mark_point()
            .encode(x="death_scalar:Q", y="birth_scalar:Q")
            .properties(title=title)
        )

    def fake_tree_chart(*args, annotated, **kwargs):
        calls.append(annotated)
        return unit_chart("tree")

    monkeypatch.setattr(split_tree, "tree_chart", fake_tree_chart)
    monkeypatch.setattr(
        split_tree, "overlay_count_colorbar", lambda tree, *args, **kwargs: tree
    )
    monkeypatch.setattr(
        split_tree, "persistence_chart", lambda *args, **kwargs: unit_chart("p")
    )
    analysis = SimpleNamespace(
        trees={
            "split": SimpleNamespace(
                persistence_table=pl.DataFrame(),
                all_stop_marker_table=pl.DataFrame(schema={"dna_sequence": pl.String}),
                critical_node_table=pl.DataFrame(
                    {"dna_sequence": ["PEAK", "ROOT"], "scalar": [1.0, 0.0]}
                ),
                branch_member_table=pl.DataFrame(
                    {"dna_sequence": ["INTERIOR"], "scalar": [0.5]}
                ),
                event_table=pl.DataFrame({"event": ["maximum"]}),
                wild_type_marker_table=pl.DataFrame(
                    schema={
                        "dna_sequence": pl.String,
                        "aa_sequence": pl.String,
                        "marker_label": pl.String,
                        "x": pl.Float64,
                        "y": pl.Float64,
                    }
                ),
            )
        },
        scalar_domain=(-1.0, 1.0),
    )
    settings = {
        "figures": {
            "figures": {"linked_tree_persistence": {}},
            "rendering": {},
        }
    }

    spec = figures.get_spec("linked_tree_persistence")
    assert spec.variants == {"vanilla"}
    payload = {"analysis": analysis, "manuscript_quantities": label_quantities()}
    figures.render_payload(spec, payload, settings, "vanilla")
    assert calls == [False]

    # The annotated export is gone: asking for it is an error, not a silent vanilla.
    with pytest.raises(ValueError, match="canonical vanilla"):
        figures.render_payload(spec, payload, settings, "annotated")
    with pytest.raises(ValueError, match="variant must be"):
        figures.render_payload(spec, payload, settings, "bogus")
    assert calls == [False]
