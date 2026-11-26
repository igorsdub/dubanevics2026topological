"""The four-graph composite is an explicit, opt-in saved-product figure."""

from dubanevics2026topological.figures.registry import get_spec


def test_four_graph_composite_has_its_own_product_route():
    spec = get_spec("bootstrap_four_graphs")
    assert spec.inputs == frozenset({"bootstrap_composite_panels"})
    assert spec.variants == frozenset({"annotated"})
    assert not spec.default


def test_published_titles_use_the_recorded_cutoff_rounded_to_three_places():
    from dubanevics2026topological.figures.bootstrap_four_graphs import panel_title

    cutoff = -0.5077739708744209
    assert panel_title("threshold=published_fdr=none", cutoff) == (
        "Papkou graph · T=-0.508 · point estimate"
    )
    assert panel_title("threshold=published_fdr=by_q=0.05", cutoff) == (
        "Papkou + FDR graph · T=-0.508 · BY FDR q=0.05"
    )
    assert "T=-0.432" in panel_title("threshold=published_fdr=none", -0.4321)


def test_candidate_titles_include_rule_and_each_recorded_cutoff():
    from dubanevics2026topological.figures.bootstrap_four_graphs import panel_title

    assert panel_title("threshold=reflected_fdr=none", -0.22224) == (
        "Topographer graph · Topographer threshold, T=-0.222 · point estimate"
    )
    assert panel_title("threshold=reflected_fdr=by_q=0.05", -0.253159) == (
        "Topographer + FDR graph · Topographer threshold, T=-0.253 · BY FDR q=0.05"
    )