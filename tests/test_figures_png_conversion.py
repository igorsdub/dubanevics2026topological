"""Every figure the existing toy fixtures can build converts to PNG through vl-convert.

Vega-Lite 6.4.1's normalizer crashed ("Cannot read properties of undefined (reading
'shape')") on a rule layer with no encoding, which no ``to_dict()`` check sees. The
leaf-figure test renders to SVG with a pinned older Vega-Lite, so it cannot see it
either. This test reuses the fixtures of the per-figure tests and converts each chart
with the default Vega-Lite of ``vl_convert``, which the cluster figures job uses.

Not covered (no toy payload exists, or it needs saved checkpoints): see
``UNCOVERED`` below.
"""

from __future__ import annotations

import pytest
import vl_convert as vlc

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Figure ids with no toy fixture in the tests, so no case here.
UNCOVERED = frozenset(
    {
        "candidate_edge_evidence",  # edge fixtures only, no registry payload
        "tree_comparison",
        "full_tree_histogram",
        "published_tree_histogram",
        "candidate_tree_histogram",
        "linked_tree_persistence",
        "study_tree_persistence_metrics",
        "study_tree_persistence_biology",
        "bootstrap_band_quantile_scan_seeds",
        "bootstrap_band_quantile_scan_grid",
        "bootstrap_four_graphs",
        "reflected_fdr_trees",
        "ridge_entropy",
        "graph_codon_entropy",
        "multiple_testing_comparison",
        "published_fdr_leaf_swarm",
        "candidate_fdr_leaf_swarm",
    }
)


def _codon_entropy(monkeypatch):
    from _figure_labels import SENTINEL  # noqa: F401
    from test_codon_entropy_plot import _codon_payload

    from dubanevics2026topological.figures.registry import get_spec, render_payload

    comparison, settings, *_ = _codon_payload()
    return render_payload(
        get_spec("bootstrap_codon_entropy"), comparison, settings, "annotated"
    )


def _peak_uncertainty(monkeypatch):
    from test_peak_uncertainty_figure import _render

    return _render()


def _band_leaf(figure_id):
    def build(monkeypatch):
        from test_band_leaf_figures import _band_settings, _payload

        from dubanevics2026topological import figures

        return figures.render_payload(
            figures.get_spec(figure_id), _payload(), _band_settings(), "vanilla"
        )

    return build


def _leaf(figure_id):
    def build(monkeypatch):
        from test_leaf_figures import _payload, _settings

        from dubanevics2026topological import figures

        return figures.render_payload(
            figures.get_spec(figure_id), _payload(), _settings(), "vanilla"
        )

    return build


def _full_tree_persistence(monkeypatch):
    from test_full_tree_persistence_figure import _fake_topographer_tree, _render

    _fake_topographer_tree.__wrapped__(monkeypatch)
    return _render("vanilla")


def _full_tree_persistence_vertical(monkeypatch):
    from test_full_tree_persistence_figure import (
        _fake_topographer_tree,
        _render_vertical,
    )

    _fake_topographer_tree.__wrapped__(monkeypatch)
    spec, _ = _render_vertical("annotated")
    return spec


def _graph_tree_persistence(monkeypatch):
    from test_graph_tree_persistence_figure import build_manifold_graph_tree

    from dubanevics2026topological.analysis import load_settings
    from dubanevics2026topological.figures.registry import get_spec, render_payload

    settings = load_settings()
    settings["figures"]["figures"]["graph_tree_persistence"][
        "presentation_surface_resolution"
    ] = 8
    return render_payload(
        get_spec("graph_tree_persistence"),
        build_manifold_graph_tree(),
        settings,
        "vanilla",
    )


def _bootstrap_biology(monkeypatch):
    from test_bootstrap_biology_figure import (
        BIOLOGY_ID,
        _biology_payload,
        _biology_settings,
        _render,
    )

    return _render(_biology_payload(monkeypatch), _biology_settings(), BIOLOGY_ID)


def _bootstrap_tree_histogram(monkeypatch):
    from test_bootstrap_twin_figures import _histogram_spec

    return _histogram_spec(monkeypatch)


def _bootstrap_global_stability(monkeypatch):
    from test_bootstrap_global_stability_figure import _stability_chart

    return _stability_chart(monkeypatch)


def _band_quantile_scan(monkeypatch):
    from test_bootstrap_band_quantile_scan_figure import _spec

    return _spec()


def _peak_rank(figure_id):
    def build(monkeypatch):
        from test_peak_rank import _payload, _settings

        from dubanevics2026topological.figures.peak_rank import build_peak_rank_figure

        return build_peak_rank_figure(_payload(), _settings(), figure_id)

    return build


def _fitness_landmarks(monkeypatch):
    from test_fitness_landmarks_figure import _payload, _settings

    from dubanevics2026topological.figures.fitness_landmarks import (
        build_fitness_landmarks_figure,
    )

    return build_fitness_landmarks_figure(_payload(), _settings())


def _join_tree(monkeypatch):
    from test_join_tree_figure import _payload, _settings

    from dubanevics2026topological.figures.join_tree import build_join_tree_figure

    return build_join_tree_figure(_payload(), _settings(), "vanilla")


def _se_correlation(monkeypatch):
    from test_fitness_se_figures import CORRELATION_ID, _growth, _settings

    from dubanevics2026topological.figures.fitness_se_correlation import (
        build_fitness_se_correlation_figure,
    )

    return build_fitness_se_correlation_figure(
        {"growth_rates": _growth()}, _settings(CORRELATION_ID)
    )


def _vs_pooled_se(monkeypatch):
    from test_fitness_se_figures import PAIR_ID, _growth, _published, _settings

    from dubanevics2026topological.figures.fitness_pooled_se_pair import (
        build_fitness_vs_pooled_se_figure,
    )

    return build_fitness_vs_pooled_se_figure(
        {"growth_rates": _growth(), "published": _published()}, _settings(PAIR_ID)
    )


def _papkou_agreement(monkeypatch):
    from test_papkou_agreement import FIGURE_ID, _fits, _published

    from dubanevics2026topological.figures.papkou_agreement import (
        build_papkou_agreement_figure,
    )

    settings = {
        "figures": {
            "rendering": {"font_family": "Arial, Liberation Sans, Helvetica, sans-serif"},
            "figures": {FIGURE_ID: {"title": "Agreement", "hexbin_min": 2500}},
        }
    }
    return build_papkou_agreement_figure(
        {"growth_rates": _fits(), "published": _published()}, settings
    )


def _functional_counts(figure_id):
    def build(monkeypatch):
        from test_functional_counts_vs_q import _scan, _settings

        from dubanevics2026topological.figures import functional_counts_vs_q as mod

        builder = {
            mod.FIGURE_ID: mod.build_functional_counts_vs_q_figure,
            mod.BH_ID: mod.build_bh_counts_figure,
            mod.BONFERRONI_ID: mod.build_bonferroni_counts_figure,
        }[figure_id]
        return builder({"scan": _scan(), "source": "toy"}, _settings(figure_id))

    return build


def _branch_vertex_counts(monkeypatch):
    from test_branch_vertex_counts_grid import _payload, _settings

    from dubanevics2026topological.figures.branch_vertex_counts_grid import (
        build_branch_vertex_counts_grid,
    )

    return build_branch_vertex_counts_grid(_payload(), _settings())


CASES = {
    "bootstrap_codon_entropy": _codon_entropy,
    "peak_uncertainty": _peak_uncertainty,
    "published_leaf_persistence": _leaf("published_leaf_persistence"),
    "candidate_leaf_persistence": _leaf("candidate_leaf_persistence"),
    "published_leaf_swarm": _leaf("published_leaf_swarm"),
    "candidate_leaf_swarm": _leaf("candidate_leaf_swarm"),
    **{
        figure_id: _band_leaf(figure_id)
        for figure_id in (
            "published_leaf_persistence_band",
            "candidate_leaf_persistence_band",
            "published_leaf_swarm_band",
            "candidate_leaf_swarm_band",
        )
    },
    "full_tree_persistence": _full_tree_persistence,
    "full_tree_persistence_vertical_histogram": _full_tree_persistence_vertical,
    "graph_tree_persistence": _graph_tree_persistence,
    "candidate_tree_bootstrap_biology": _bootstrap_biology,
    "bootstrap_tree_histogram": _bootstrap_tree_histogram,
    "bootstrap_global_stability": _bootstrap_global_stability,
    "bootstrap_band_quantile_scan": _band_quantile_scan,
    "peak_rank": _peak_rank("peak_rank"),
    "peak_rank_annotated": _peak_rank("peak_rank_annotated"),
    "peak_rank_histogram": _peak_rank("peak_rank_histogram"),
    "full_tree_fitness_landmarks": _fitness_landmarks,
    "full_join_tree": _join_tree,
    "fitness_se_correlation": _se_correlation,
    "fitness_vs_pooled_se": _vs_pooled_se,
    "papkou_agreement": _papkou_agreement,
    "functional_counts_vs_q": _functional_counts("functional_counts_vs_q"),
    "functional_counts_vs_q_bh": _functional_counts("functional_counts_vs_q_bh"),
    "functional_counts_vs_q_bonferroni": _functional_counts(
        "functional_counts_vs_q_bonferroni"
    ),
    "branch_vertex_counts_2x2": _branch_vertex_counts,
}


@pytest.mark.parametrize("figure_id", sorted(CASES))
def test_the_figure_converts_to_png_through_vl_convert(figure_id, monkeypatch) -> None:
    built = CASES[figure_id](monkeypatch)
    spec = built if isinstance(built, dict) else built.to_dict()
    assert vlc.vegalite_to_png(spec)[:8] == PNG_MAGIC


def test_covered_and_uncovered_ids_are_registered_figures() -> None:
    from dubanevics2026topological.figures.registry import FIGURE_SPECS

    registered = {spec.figure_id for spec in FIGURE_SPECS}
    assert (set(CASES) | UNCOVERED) - registered == set()
    assert not set(CASES) & UNCOVERED


def test_the_empty_rule_layer_slot_converts_to_png() -> None:
    """The no-rows fallback of the synonymous-edge layers had no encoding."""
    import altair as alt

    from dubanevics2026topological.figures._charts import empty_rule_layer

    base = (
        alt.Chart(alt.Data(values=[{"a": 1, "b": 2}]))
        .mark_point()
        .encode(x="a:Q", y="b:Q")
    )
    chart = alt.layer(base, empty_rule_layer())
    assert vlc.vegalite_to_png(chart.to_dict())[:8] == PNG_MAGIC
