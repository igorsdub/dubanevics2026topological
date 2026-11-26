"""Leaf tables of a saved band tree and the four band-tree leaf figures."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import vl_convert as vlc
from test_leaf_figures import (
    EDGES,
    MEMBERS,
    NODE_X,
    NODES,
    STOP_MEMBER,
    _landscape,
    _settings,
    _tables,
)

from dubanevics2026topological import figures
from dubanevics2026topological.build import BAND_LEAF_FIGURE_IDS, NON_CANDIDATE_FIGURE_IDS
from dubanevics2026topological.build.band_leaf_inputs import load_band_leaf_inputs
from dubanevics2026topological.figures._leaf_data import saved_tree_leaf_tables

ROOT = Path(__file__).resolve().parents[1]
PERSISTENCE_IDS = ("published_leaf_persistence_band", "candidate_leaf_persistence_band")
SWARM_IDS = ("published_leaf_swarm_band", "candidate_leaf_swarm_band")
BAND_IDS = PERSISTENCE_IDS + SWARM_IDS
# (birth, death) node of each peak, as the nominal fixture pairs them.
PAIRS = [(0, 7, "extended"), (1, 5, "ordinary"), (2, 4, "ordinary"), (3, 6, "ordinary")]


def _saved_inputs():
    """The nominal fixture tree as a saved band tree.

    Edges run lower to upper, as a saved layout's do, and the layout edge table keeps a
    stale branch count of 1 (the saved tree's own counts are on ``branch_table``).
    """
    nodes = pl.DataFrame(
        [
            {
                "tree_node_index": i,
                "event": event,
                "dna_sequence": dna,
                "aa_sequence": aa,
                "scalar": scalar,
                "x": NODE_X[i],
                "has_stop_codon": "*" in aa,
            }
            for i, (dna, aa, scalar, event) in enumerate(NODES)
        ]
    )
    edges = pl.DataFrame(
        [
            {
                "source_tree_node_index": lower,
                "target_tree_node_index": upper,
                "branch_vertex_count": 1,
            }
            for upper, lower in EDGES
        ]
    )
    sizes = {(u, l): 1 for u, l in EDGES}
    for upper, lower, _, _ in MEMBERS:
        sizes[(upper, lower)] += 1
    branch_table = pl.DataFrame(
        [
            {
                "source_tree_node_index": lower,
                "target_tree_node_index": upper,
                "branch_vertex_count": count,
            }
            for (upper, lower), count in sizes.items()
        ]
    )
    rows = [(dna, scalar, aa, dna) for dna, aa, scalar, _ in NODES]
    for upper, _, dna, scalar in MEMBERS:
        rows.append((dna, scalar, "PP", NODES[upper][0]))
    assignments = pl.DataFrame(
        rows,
        schema=["dna_sequence", "raw_fitness", "aa_sequence", "upper_sequence"],
        orient="row",
    ).with_columns((pl.col("dna_sequence") == STOP_MEMBER).alias("has_stop_codon"))
    persistence = pl.DataFrame(
        [
            {
                "birth_label": NODES[b][0],
                "death_label": NODES[d][0],
                "birth_scalar": NODES[b][2],
                "death_scalar": NODES[d][2],
                "persistence": NODES[b][2] - NODES[d][2],
                "pair_kind": kind,
            }
            for b, d, kind in PAIRS
        ]
    )
    view = SimpleNamespace(
        plot_data=SimpleNamespace(node_table=nodes, edge_table=edges),
        branch_table=branch_table,
        assignments=assignments,
        wild_type_marker_table=pl.DataFrame({"dna_sequence": ["GGGGGG"]}),
    )
    return SimpleNamespace(
        tree_analysis=view,
        persistence_table=persistence,
        tree_id="t" * 64,
        landscape_id="l" * 64,
    )


def _payload():
    tables = saved_tree_leaf_tables(_saved_inputs())
    panel = {"leaf_tables": tables, "band_quantile": 0.95, "draws": 10000}
    return {"panels": {"published": panel, "candidate": panel}}


def _band_settings():
    settings = _settings()
    figures_table = settings["figures"]["figures"]
    for figure_id in PERSISTENCE_IDS:
        figures_table[figure_id] = {"panel_size": 300}
    for figure_id in SWARM_IDS:
        figures_table[figure_id] = {"leaves_per_row": 2, "row_width": 200}
    return settings


def test_a_saved_tree_gives_the_nominal_leaf_tables() -> None:
    saved = saved_tree_leaf_tables(_saved_inputs())
    nominal = _tables(_landscape())
    assert saved.leaves.equals(nominal.leaves)
    assert saved.members.equals(nominal.members)


def test_the_hill_size_is_the_saved_branch_count_not_the_layout_edge_count() -> None:
    leaves = saved_tree_leaf_tables(_saved_inputs()).leaves
    sizes = dict(leaves.select("peak_sequence", "member_count").iter_rows())
    assert sizes == {"AAAAAA": 3, "CCCCCC": 1, "GGGGGG": 2, "TCTCTC": 1}


def test_a_peak_without_a_persistence_birth_is_refused() -> None:
    inputs = _saved_inputs()
    inputs.persistence_table = inputs.persistence_table.filter(
        pl.col("birth_label") != "TCTCTC"
    )
    with pytest.raises(ValueError, match="exactly one persistence birth"):
        saved_tree_leaf_tables(inputs)


@pytest.mark.parametrize("figure_id", BAND_IDS)
def test_band_leaf_figures_render_to_svg(figure_id) -> None:
    chart = figures.render_payload(
        figures.get_spec(figure_id), _payload(), _band_settings(), "vanilla"
    )
    assert vlc.vegalite_to_svg(chart.to_dict(), vl_version="v5_21").startswith("<svg")


def test_band_leaf_subtitles_count_the_saved_tree_leaves() -> None:
    persistence = figures.render_payload(
        figures.get_spec("candidate_leaf_persistence_band"),
        _payload(),
        _band_settings(),
        "vanilla",
    ).to_dict()["title"]
    assert persistence["text"].startswith("Candidate graph, BY band-simplified tree")
    assert persistence["subtitle"] == [
        "Papkou et al. (2023) · 4 leaves · bootstrap band quantile 0.95, 10,000 draws",
        "2 singleton hills · largest hill 3 sequences",
    ]
    swarm = figures.render_payload(
        figures.get_spec("published_leaf_swarm_band"),
        _payload(),
        _band_settings(),
        "vanilla",
    ).to_dict()["title"]
    assert swarm["subtitle"][0] == (
        "Papkou et al. (2023) · 4 leaves · 7 sequences in hills · "
        "bootstrap band quantile 0.95, 10,000 draws"
    )


def test_the_band_leaf_figures_are_vanilla_saved_product_figures() -> None:
    assert BAND_LEAF_FIGURE_IDS == frozenset(BAND_IDS)
    tables = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))["figures"]
    for figure_id in BAND_IDS:
        spec = figures.get_spec(figure_id)
        # The Candidate band figures are manuscript figures (S24, S25), so built by default.
        assert spec.default is figure_id.startswith("candidate_")
        assert spec.inputs == frozenset({"band_leaf_tables"})
        assert spec.inputs <= figures.registry.SAVED_PRODUCT_INPUTS
        assert spec.variants == frozenset({"vanilla"})
        assert spec.config_table == figure_id and figure_id in tables
        assert figure_id in NON_CANDIDATE_FIGURE_IDS
        with pytest.raises(ValueError, match="canonical vanilla"):
            figures.render_payload(spec, _payload(), _band_settings(), "annotated")
    for figure_id in PERSISTENCE_IDS:
        assert tables[figure_id]["max_marker_area"] > tables[figure_id]["min_marker_area"]
    for figure_id in SWARM_IDS:
        assert tables[figure_id]["leaves_per_row"] >= 1


def test_the_loader_reads_the_configured_band_trees_through_the_tree_loader(monkeypatch) -> None:
    from dubanevics2026topological.build import story_view_inputs

    settings = tomllib.loads((ROOT / "config/figures.toml").read_text("utf-8"))
    assert settings["story_views"]["band_trees"] == ["published_by_band", "reflected_by_band"]
    labels = {
        "published_by_band": "tree__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_hash=00fae86f",
        "reflected_by_band": "tree__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_hash=1c35ccfe",
    }
    monkeypatch.setattr(
        story_view_inputs.processed,
        "load",
        lambda kind, state, context: SimpleNamespace(label=labels.get(state, f"tree__{state}")),
    )
    band_labels = list(labels.values())
    seen: list[str] = []

    def loader(context, label):
        seen.append(label)
        return _saved_inputs()

    payload = load_band_leaf_inputs(SimpleNamespace(), {"figures": settings}, loader)
    assert seen == band_labels
    assert set(payload["panels"]) == {"published", "candidate"}
    assert payload["panels"]["candidate"]["band_quantile"] == 0.95
    assert payload["panels"]["candidate"]["draws"] == 10000
    assert payload["product_ids"]["published"] == "t" * 64
