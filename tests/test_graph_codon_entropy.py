"""Supplementary S14: Full-graph and Papkou-graph entropy versus fitness."""

from __future__ import annotations

import json

import polars as pl
import pytest

from dubanevics2026topological.analysis import load_settings
from dubanevics2026topological.analysis.codon_entropy import (
    compute_positional_entropy_sweep,
)
from dubanevics2026topological.analysis.graph_codon_entropy import (
    build_graph_codon_entropy,
)
from dubanevics2026topological.figures.graph_codon_entropy import (
    build_graph_codon_entropy_figure,
)

DNA = ["AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG", "AAAAAAAAA", "AAGGAAAAA"]
FITNESS = [1.0, 0.8, 0.4, 0.1, 0.9]


def _graphs():
    full = pl.DataFrame(
        {"dna_sequence": DNA, "raw_fitness": FITNESS},
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64},
    )
    published = full.filter(pl.col("raw_fitness") >= 0.8)
    return full, published


def test_sweep_matches_the_main_text_entropy() -> None:
    """The plotted values are compute_positional_entropy_sweep, not a new definition."""
    full, published = _graphs()
    result = build_graph_codon_entropy(full, published, sweep_points=5)
    assert set(result.sweeps["population"].unique()) == {"full", "published"}
    thresholds = (
        result.sweeps.filter(pl.col("population") == "published")
        .get_column("threshold_scalar")
        .unique()
        .sort()
        .to_list()
    )
    from dubanevics2026topological.analysis.graph_codon_entropy import (
        _population_frame,
    )

    direct = compute_positional_entropy_sweep(
        _population_frame(published), thresholds=thresholds
    )
    got = result.sweeps.filter(pl.col("population") == "published").select(
        "threshold_scalar", "aa_position", "entropy_bits", "active_sequence_count"
    )
    assert got.sort(["aa_position", "threshold_scalar"]).equals(
        direct.sort(["aa_position", "threshold_scalar"])
    )


def test_published_sequences_must_belong_to_the_full_graph() -> None:
    full, _published = _graphs()
    extra = pl.DataFrame(
        {"dna_sequence": ["CCCCCCCCC"], "raw_fitness": [0.2]},
        schema={"dna_sequence": pl.String, "raw_fitness": pl.Float64},
    )
    with pytest.raises(ValueError, match="Full graph"):
        build_graph_codon_entropy(full, extra, sweep_points=3)


@pytest.mark.parametrize("style", ["annotated", "vanilla"])
def test_figure_is_two_graphs_and_no_tree(style) -> None:
    full, published = _graphs()
    chart = build_graph_codon_entropy_figure(
        build_graph_codon_entropy(full, published, sweep_points=5),
        load_settings(),
        style,
        title="Entropy against fitness threshold",
        subtitle="fixture",
    )
    text = json.dumps(chart.to_dict())
    assert "Full graph" in text
    assert "Papkou graph" in text
    assert "KEM" not in text
    assert "TCL" not in text
    assert "tree" not in text.lower()


def test_every_panel_declares_the_one_shared_entropy_bar() -> None:
    full, published = _graphs()
    spec = build_graph_codon_entropy_figure(
        build_graph_codon_entropy(full, published, sweep_points=5),
        load_settings(),
        "vanilla",
        title="Entropy against fitness threshold",
        subtitle="fixture",
    ).to_dict()
    assert spec["resolve"]["scale"]["color"] == "shared"
    legends = []
    for panel in spec["hconcat"]:
        layers = panel.get("layer", [panel])
        for layer in layers:
            color = layer.get("encoding", {}).get("color", {})
            if "legend" in color:
                legends.append(color["legend"])
    assert len(legends) == len(spec["hconcat"])
    assert all(legend is not None for legend in legends)
    assert all(legend["orient"] == "right" for legend in legends)


def test_figure_renders_to_png(tmp_path) -> None:
    from dubanevics2026topological.build import _render_one

    full, published = _graphs()
    chart = build_graph_codon_entropy_figure(
        build_graph_codon_entropy(full, published, sweep_points=5),
        load_settings(),
        "annotated",
        title="Entropy against fitness threshold",
        subtitle="fixture",
    )
    target = tmp_path / "graph_codon_entropy.png"
    _render_one(chart, target, "png", png_dpi=72)
    assert target.stat().st_size > 0
