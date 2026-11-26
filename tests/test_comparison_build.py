"""Focused cache and variant checks for the three-landscape figure build."""

from __future__ import annotations

from dubanevics2026topological.rawdata.value_digest import value_digest

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import rustworkx as rx
from _analysis_record import write_analysis_record
from _fake_figures import install_fake_figures, write_renderer_sources

from dubanevics2026topological.build import build_figures
from dubanevics2026topological.results import RunContext


def _plotted(payload):
    """The cached comparison a figure got, without the registry its labels come from."""
    if isinstance(payload, dict):
        return {k: v for k, v in payload.items() if k != "manuscript_quantities"}
    return payload


class FakeChart:
    def save(self, path: str | Path, *, format: str, **kwargs: object) -> None:
        Path(path).write_text(format, encoding="utf-8")


@pytest.fixture
def comparison_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import dubanevics2026topological.build as build_module

    root = tmp_path
    source_dir = root / "src/dubanevics2026topological/figures"
    source_dir.mkdir(parents=True)
    write_renderer_sources(source_dir, "# renderer")
    (root / "notebooks").mkdir()
    (root / "notebooks/comparison.py").write_text("# controls", encoding="utf-8")
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "figures.toml").write_text("# appearance", encoding="utf-8")
    provenance = root / "output/analysis/clamped_comparison/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"generation": 1}\n', encoding="utf-8")
    baseline_provenance = write_analysis_record(root / "output", "baseline-1")

    settings = {
        "analysis": {
            "clamped": {
                "published_functional_threshold": -0.507774,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            }
        },
        "figures": {
            "formats": ["svg"],
            "variants": ["vanilla", "annotated"],
            "rendering": {"font_family": "sans-serif"},
            "figures": {
                "tree_comparison": {
                    "notebook": "notebooks/comparison.py",
                    "tree": {"width": 520},
                },
                "candidate_tree_histogram": {
                    "notebook": "notebooks/candidate_histogram.py",
                    "tree": {"width": 420},
                },
                "full_tree_histogram": {
                    "notebook": "notebooks/full_histogram.py",
                    "tree": {"width": 420},
                },
                "published_tree_histogram": {
                    "notebook": "notebooks/article_histogram.py",
                    "tree": {"width": 420},
                },
            },
        },
    }
    builds: list[tuple[str, str, object]] = []
    calls: list[str] = []
    landscape_cache = {"landscapes": {"full": object()}, "thresholds": {"full": None}}
    comparison_module = SimpleNamespace(
        load_cached_clamped_comparison=lambda: calls.append("cached") or landscape_cache
    )
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.clamped_cache",
        comparison_module,
    )
    fake_figures = SimpleNamespace(
        FIGURE_IDS=("tree_comparison",),
        render=lambda payload, figure_id, config, *, variant="annotated": (
            builds.append((figure_id, variant, _plotted(payload))) or FakeChart()
        ),
    )
    install_fake_figures(monkeypatch, fake_figures)
    monkeypatch.setattr(build_module, "ROOT", root)
    monkeypatch.setattr(build_module, "OUTPUT", root / "output")
    monkeypatch.setattr(build_module, "PROVENANCE_PATH", baseline_provenance)
    monkeypatch.setattr(
        build_module, "CHECKPOINT_DIR", root / "output/checkpoints/analysis"
    )
    monkeypatch.setattr(build_module, "FIGURE_DIR", root / "output/figures")
    monkeypatch.setattr(build_module, "RECORD_DIR", root / "output/figures/.records")
    monkeypatch.setattr(build_module, "load_settings", lambda: settings)
    monkeypatch.setattr(
        build_module,
        "load_cached_analysis",
        lambda: (_ for _ in ()).throw(AssertionError("full baseline must not load")),
    )
    return settings, builds, calls, provenance, landscape_cache


def test_comparison_build_is_single_canonical_and_appearance_only_invalidates(
    comparison_build,
) -> None:
    settings, builds, calls, provenance, cached = comparison_build

    outputs = build_figures(("tree_comparison",))
    assert [path.name for path in outputs] == ["tree_comparison.svg"]
    assert builds == [("tree_comparison", "annotated", cached)]

    builds.clear()
    settings["figures"]["figures"]["tree_comparison"]["tree"]["width"] = 700
    build_figures(("tree_comparison",))
    assert builds == [("tree_comparison", "annotated", cached)]
    assert calls == ["cached", "cached"]

    builds.clear()
    provenance.write_text(json.dumps({"generation": 2}), encoding="utf-8")
    build_figures(("tree_comparison",))
    assert builds == [("tree_comparison", "annotated", cached)]


def test_candidate_histogram_build_reuses_comparison_cache_and_is_canonical(
    comparison_build, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, builds, calls, _, cached = comparison_build
    fake_figures = SimpleNamespace(
        FIGURE_IDS=(
            "tree_comparison",
            "candidate_tree_histogram",
        ),
        render=lambda payload, figure_id, config, *, variant="annotated": (
            builds.append((figure_id, variant, _plotted(payload))) or FakeChart()
        ),
    )
    install_fake_figures(monkeypatch, fake_figures)

    outputs = build_figures(("candidate_tree_histogram",))
    assert [path.name for path in outputs] == [
        "candidate_tree_histogram.svg"
    ]
    assert builds == [("candidate_tree_histogram", "annotated", cached)]
    assert calls == ["cached"]

    with pytest.raises(ValueError, match="canonical annotated export"):
        build_figures(("candidate_tree_histogram",), variants=("vanilla",))
    assert calls == ["cached"]


@pytest.mark.parametrize(
    "figure_id",
    ("full_tree_histogram", "published_tree_histogram"),
)
def test_population_histogram_builds_reuse_comparison_cache_and_are_canonical(
    comparison_build, monkeypatch: pytest.MonkeyPatch, figure_id: str
) -> None:
    _, builds, calls, _, cached = comparison_build
    fake_figures = SimpleNamespace(
        FIGURE_IDS=(
            "tree_comparison",
            "candidate_tree_histogram",
            "full_tree_histogram",
            "published_tree_histogram",
        ),
        render=lambda payload, selected_id, config, *, variant="annotated": (
            builds.append((selected_id, variant, _plotted(payload))) or FakeChart()
        ),
    )
    install_fake_figures(monkeypatch, fake_figures)

    outputs = build_figures((figure_id,))
    assert [path.name for path in outputs] == [f"{figure_id}.svg"]
    assert builds == [(figure_id, "annotated", cached)]
    assert calls == ["cached"]

    with pytest.raises(ValueError, match="canonical annotated export"):
        build_figures((figure_id,), variants=("vanilla",))


def test_study_metric_build_uses_comparison_cache_and_canonical_vanilla_name(
    comparison_build, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, builds, calls, _, cached = comparison_build
    settings["figures"]["figures"]["study_tree_persistence_metrics"] = {
        "tree": {"width": 420}
    }
    fake_figures = SimpleNamespace(
        FIGURE_IDS=(
            "tree_comparison",
            "study_tree_persistence_metrics",
        ),
        render=lambda payload, figure_id, config, *, variant="annotated": (
            builds.append((figure_id, variant, _plotted(payload))) or FakeChart()
        ),
    )
    install_fake_figures(monkeypatch, fake_figures)

    outputs = build_figures(
        ("study_tree_persistence_metrics",), variants=("vanilla",)
    )
    assert [path.name for path in outputs] == [
        "study_tree_persistence_metrics.svg"
    ]
    assert builds == [("study_tree_persistence_metrics", "vanilla", cached)]
    assert calls == ["cached"]

    with pytest.raises(ValueError, match="canonical vanilla export"):
        build_figures(
            ("study_tree_persistence_metrics",), variants=("annotated",)
        )


def test_explicit_vanilla_selection_skips_comparison_in_mixed_build(
    comparison_build, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, builds, _, _, _ = comparison_build
    settings["figures"]["figures"]["linked_tree_persistence"] = {
        "tree": {"width": 420}
    }
    fake_figures = SimpleNamespace(
        FIGURE_IDS=("tree_comparison", "linked_tree_persistence"),
        render=lambda payload, figure_id, config, *, variant="annotated": (
            builds.append((figure_id, variant, _plotted(payload))) or FakeChart()
        ),
    )
    import dubanevics2026topological.build as build_module

    # Keep the test independent of the renderer implementation while allowing
    # the ordinary raw-tree input to reuse a sentinel cache.
    install_fake_figures(monkeypatch, fake_figures)
    monkeypatch.setattr(build_module, "load_cached_analysis", lambda **_: object())
    build_figures(variants=("vanilla",))
    assert {variant for _, variant, _ in builds} == {"vanilla"}
    assert all(
        figure_id == "linked_tree_persistence" for figure_id, _, _ in builds
    )


def test_comparison_science_fingerprint_tracks_inputs_but_not_appearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dubanevics2026topological.analysis as analysis_package
    import dubanevics2026topological.analysis.clamped_cache as cache

    root = tmp_path
    (root / "data/raw").mkdir(parents=True)
    graphml = root / "data/raw/reference.graphml"
    peaks = root / "data/raw/reference_peaks.csv"
    provenance = root / "output/analysis/provenance.json"
    for path, value in (
        (graphml, "graph-v1"),
        (peaks, "peaks-v1"),
        (root / "wt.parquet", "wt-v1"),
        (root / "diff.parquet", "diff-v1"),
    ):
        path.write_text(value, encoding="utf-8")
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"fingerprint": "baseline-v1"}', encoding="utf-8")
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "graphml": "data/raw/reference.graphml",
                "peaks": "data/raw/reference_peaks.csv",
                "graphml_value_sha256": "",
                "peaks_value_sha256": "",
                "published_functional_threshold": -0.507774,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        },
        "figures": {"figures": {"comparison": {"tree": {"width": 520}}}},
    }
    monkeypatch.setattr(cache, "ROOT", root)
    # ``source_key`` reads the package ROOT, so keep the two in step.
    monkeypatch.setattr(analysis_package, "ROOT", root)
    monkeypatch.setattr(
        cache, "analysis_provenance_path", lambda context=None: provenance
    )
    monkeypatch.setattr(cache, "scientific_code_paths", list)
    monkeypatch.setattr(
        cache,
        "candidate_selection_fingerprint",
        lambda _, **kwargs: {
            "rounded_threshold": -0.25,
            "source": {"rule": "median_reflected"},
        },
    )
    settings["analysis"]["clamped"]["graphml_value_sha256"] = value_digest(graphml)
    settings["analysis"]["clamped"]["peaks_value_sha256"] = value_digest(peaks)

    original = cache._fingerprint(settings)
    settings["figures"]["figures"]["comparison"]["tree"]["width"] = 800
    assert cache._fingerprint(settings) == original

    settings["analysis"]["clamped"]["candidate_functional_threshold"] = {
        "rule": "median_reflected",
        "decimals": 3,
    }
    assert cache._fingerprint(settings) != original
    settings["analysis"]["clamped"]["candidate_functional_threshold"] = {
        "rule": "median_reflected",
        "decimals": 6,
    }
    assert cache._fingerprint(settings) == original
    (root / "wt.parquet").write_text("wt-v2", encoding="utf-8")
    assert cache._fingerprint(settings) != original
    (root / "wt.parquet").write_text("wt-v1", encoding="utf-8")
    graphml.write_text("graph-v2", encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        cache._fingerprint(settings)


def test_cached_comparison_rejects_old_clamped_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dubanevics2026topological.analysis.clamped_cache as cache

    cache_dir = tmp_path / "comparison"
    cache_dir.mkdir()
    (cache_dir / "provenance.json").write_text(
        json.dumps({"schema": 1, "thresholds": {}}), encoding="utf-8"
    )
    (cache_dir / "diagnostics.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        cache, "cache_dirs", lambda context=None: (cache_dir, tmp_path / "pending")
    )
    with pytest.raises(ValueError, match="incompatible scientific contract"):
        cache.load_cached_clamped_comparison()


def _threshold_record(rounded: float) -> dict[str, object]:
    return {
        "rule": "median_reflected",
        "decimals": 6,
        "median": -0.5,
        "global_minimum": -0.75,
        "median_gap": 0.25,
        "unrounded_threshold": rounded,
        "rounded_threshold": rounded,
        "functional_vertex_count": None,
        "uncertain_vertex_count": None,
        "selected_component_functional_vertex_count": 3,
    }


@pytest.mark.parametrize("scalar", ["raw_fitness", "clamped_fitness"])
def test_cached_comparison_enforces_raw_scalar_after_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scalar: str
) -> None:
    import dubanevics2026topological.analysis.clamped_cache as cache

    cache_dir = tmp_path / "comparison"
    cache_dir.mkdir()
    provenance = {
        "schema": cache.CACHE_SCHEMA,
        "analysis_scalar": cache.RAW_ANALYSIS_SCALAR,
        "edge_selection_rule": "at_least_one_endpoint_functional_class",
        "candidate_edge_selection_rule": "at_least_one_endpoint_functional_class",
        "published_edge_selection_rule": cache.EDGE_SELECTION_RULE,
        "thresholds": {"candidate_clamped": -0.25},
    }
    (cache_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (cache_dir / "diagnostics.json").write_text("{}", encoding="utf-8")
    (cache_dir / "graph_summary.json").write_text(
        json.dumps({"schema": cache.SUMMARY_SCHEMA, "metrics": {}}), encoding="utf-8"
    )
    (cache_dir / cache.CANDIDATE_THRESHOLD_FILE).write_text(
        json.dumps(_threshold_record(-0.25)), encoding="utf-8"
    )
    for name in cache.LANDSCAPE_KEYS:
        metadata = cache_dir / name / "data" / "metadata.toml"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("", encoding="utf-8")
    for name in (
        f"{landscape}_{table}"
        for landscape in cache.LANDSCAPE_KEYS
        for table in (
            "components",
            "excluded_vertices",
            "peaks",
            "branch_table",
            "persistence_table",
        )
    ):
        path = cache_dir / "tables" / f"{name}.parquet"
        path.parent.mkdir(exist_ok=True)
        pl.DataFrame().write_parquet(path)
    analysis = SimpleNamespace(
        scalar_attr=scalar,
        trees={"split": SimpleNamespace(scalar_attr=scalar)},
    )
    monkeypatch.setattr(
        cache, "cache_dirs", lambda context=None: (cache_dir, tmp_path / "pending")
    )
    full_calls: list[object] = []
    monkeypatch.setattr(
        cache,
        "load_cached_analysis",
        lambda **kwargs: full_calls.append(kwargs.get("context")) or "full",
    )
    monkeypatch.setattr(cache, "load_network_analysis", lambda _: analysis)
    if scalar == "raw_fitness":
        result = cache.load_cached_clamped_comparison()
        assert result["landscapes"]["study_clamped"] is analysis
        assert result["landscapes"]["full"] == "full"
        # Inside a cluster job a run context is bound: the Full analysis is skipped
        # by default, but a reader that states a Full quantity asks for it, and the
        # context reaches the Full loader.
        context = SimpleNamespace()
        skipped = cache.load_cached_clamped_comparison(context=context)
        assert skipped["landscapes"]["full"] is None
        full_calls.clear()
        loaded = cache.load_cached_clamped_comparison(
            context=context, include_full=True
        )
        assert loaded["landscapes"]["full"] == "full"
        assert full_calls == [context]
    else:
        with pytest.raises(ValueError, match="raw_fitness is required"):
            cache.load_cached_clamped_comparison()


def _stub_the_candidate_prerequisites(
    monkeypatch: pytest.MonkeyPatch, events: list[str]
) -> None:
    """Replace the growth-rate and classification stages that `all` runs first."""
    import dubanevics2026topological.build as build_module

    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.sequence_growth_rates",
        SimpleNamespace(
            compute_sequence_growth_rates=lambda settings, **kwargs: (
                events.append("sequence-growth-rates") or "rates"
            ),
            load_cached_sequence_growth_rates=lambda **kwargs: SimpleNamespace(
                growth_rates=pl.DataFrame()
            ),
        ),
    )
    functional_classification = SimpleNamespace(
        FUNCTIONAL="functional",
        UNCERTAIN="uncertain",
        compute_functional_classification=lambda settings, **kwargs: events.append(
            "functional-classification"
        ),
        load_cached_functional_classification_with_source=lambda *args, **kwargs: (
            SimpleNamespace(
                summary={"threshold": {"unrounded": -0.5, "rounded": -0.5}},
                classification=pl.DataFrame(
                    {
                        "functional_class": ["functional"],
                        "dna_sequence": ["GCCGATCTC"],
                    }
                ),
            ),
            {"checkpoint": "cached"},
        ),
    )
    # The threshold module is the current implementation; the analysis path is
    # its compatibility alias and may already be imported by the build package.
    for module_name in (
        "dubanevics2026topological.threshold.functional_classification",
        "dubanevics2026topological.threshold.functional_classification",
    ):
        monkeypatch.setitem(sys.modules, module_name, functional_classification)
    monkeypatch.setattr(
        build_module, "build_sequence_growth_rate_tables", lambda *a, **k: []
    )


def test_all_command_prepares_comparison_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dubanevics2026topological.build as build_module

    events: list[str] = []
    comparison = {"diagnostics": {"study_clamped": {}, "candidate_clamped": {}}}
    _stub_the_candidate_prerequisites(monkeypatch, events)
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.candidate_edge_evidence",
        SimpleNamespace(load_or_compute_candidate_edge_evidence=lambda **kwargs: None),
    )
    cache_module = SimpleNamespace(
        load_or_compute_clamped_comparison=lambda *, force=False: (
            events.append("comparison-analysis") or comparison
        )
    )
    monkeypatch.setitem(
        sys.modules, "dubanevics2026topological.analysis.clamped_cache", cache_module
    )
    monkeypatch.setattr(
        build_module, "ensure_analysis_checkpoint", lambda: events.append("analysis")
    )
    monkeypatch.setattr(
        build_module,
        "build_tables",
        lambda *, force=False: events.append("tables") or [],
    )
    monkeypatch.setattr(
        build_module,
        "build_comparison_tables",
        lambda result, *, force=False: events.append("comparison-tables") or [],
    )
    monkeypatch.setattr(
        build_module,
        "build_figures",
        lambda *args, **kwargs: events.append("figures") or [],
    )
    monkeypatch.setattr(
        build_module,
        "build_peak_tables",
        lambda *, force=False: events.append("peak-analysis-and-tables") or [],
    )

    monkeypatch.setattr(
        build_module,
        "build_codon_entropy_tables",
        lambda result, *, force=False: events.append("codon-entropy-tables") or [],
    )

    monkeypatch.setattr(
        build_module,
        "build_leading_saddle_band_tables",
        lambda *, force=False: events.append("leading-saddle-band") or [],
    )
    monkeypatch.setattr(
        build_module,
        "build_biological_tables",
        lambda result, *, force=False: events.append("biological-tables") or [],
    )
    monkeypatch.setattr(
        build_module,
        "build_manuscript_quantities",
        lambda *, force=False: events.append("manuscript-quantities") or [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.pooled_scatter_product.build_pooled_scatter_product",
        lambda *, context=None: events.append("pooled-scatter-product") or [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.threshold_diagnostics_product.build_threshold_diagnostics_product",
        lambda *, context=None: events.append("threshold-diagnostics-product") or [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.bootstrap_band_quantile_scan_product.build_bootstrap_band_quantile_scan_product",
        lambda *a, **k: events.append("bootstrap-band-quantile-scan-product") or [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.bootstrap_codon_entropy_product.build_bootstrap_codon_entropy_product",
        lambda *, context=None: events.append("bootstrap-codon-entropy-product") or [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.graph_tree_persistence_product.build_graph_tree_persistence_product",
        lambda *, context=None: events.append("graph-tree-persistence-product") or [],
    )

    assert build_module.main(["all"]) == 0
    assert events == [
        "sequence-growth-rates",
        "functional-classification",
        "analysis",
        "tables",
        "comparison-analysis",
        "comparison-tables",
        "codon-entropy-tables",
        "peak-analysis-and-tables",
        "pooled-scatter-product",
        "leading-saddle-band",
        "biological-tables",
        "bootstrap-codon-entropy-product",
        "graph-tree-persistence-product",
        "manuscript-quantities",
        "threshold-diagnostics-product",
        "figures",
    ]


def test_tree_support_tables_include_stable_ids_and_figure_count() -> None:
    from dubanevics2026topological.analysis.clamped_cache import _tree_support_tables

    tree = rx.PyDiGraph()
    first = tree.add_node({"dna_sequence": "AAA", "vertex_index": 0})
    second = tree.add_node({"dna_sequence": "AAG", "vertex_index": 1})
    branches = pl.DataFrame(
        {
            "source_tree_node_index": [first],
            "target_tree_node_index": [second],
            "branch_member_count": [2],
        }
    )
    pairs = pl.DataFrame(
        {
            "birth_tree_node_index": [first],
            "death_tree_node_index": [second],
            "pair_kind": ["finite"],
        }
    )
    analysis = SimpleNamespace(
        trees={
            "split": SimpleNamespace(
                tree=tree,
                branch_table=branches,
                branch_member_table=None,
                persistence_table=pairs,
            )
        }
    )

    tables = _tree_support_tables("study_clamped", analysis)
    exported_branches = tables["study_clamped_branch_table"]
    assert exported_branches["source_tree_node_identity"].to_list() == ["AAA"]
    assert exported_branches["target_tree_node_identity"].to_list() == ["AAG"]
    assert exported_branches["branch_vertex_count"].to_list() == [3]
    exported_pairs = tables["study_clamped_persistence_table"]
    assert exported_pairs["birth_tree_node_identity"].to_list() == ["AAA"]
    assert exported_pairs["death_tree_node_identity"].to_list() == ["AAG"]
    assert not any(column.endswith("_index") for column in exported_pairs.columns)


def test_component_csv_uses_separate_sequence_membership_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dubanevics2026topological.build as build_module

    monkeypatch.setattr(build_module, "ROOT", tmp_path)
    monkeypatch.setattr(build_module, "OUTPUT", tmp_path / "output")
    provenance = tmp_path / "output/analysis/clamped_comparison/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"generation": 1}\n', encoding="utf-8")
    components = pl.DataFrame(
        {
            "component_id": [1],
            "vertex_count": [2],
            "selected": [True],
            "minimum_sequence": ["AAA"],
            "sequences": [["AAA", "AAG"]],
        }
    )

    outputs = build_module.build_comparison_tables(
        {"tables": {"study_clamped_components": components}}
    )
    assert {path.name for path in outputs} == {
        "clamped_study_clamped_components.csv",
        "clamped_study_clamped_components.parquet",
        "clamped_study_clamped_components_members.csv",
        "clamped_study_clamped_components_members.parquet",
    }
    csv_components = pl.read_csv(
        tmp_path / "output/tables/clamped_study_clamped_components.csv"
    )
    csv_members = pl.read_csv(
        tmp_path / "output/tables/clamped_study_clamped_components_members.csv"
    )
    assert "sequences" not in csv_components.columns
    assert csv_members["dna_sequence"].to_list() == ["AAA", "AAG"]

    # Output bytes equal the checkpoint's; deleted files are restored, not rebuilt.
    step = tmp_path / "output/checkpoints/clamped_comparison_tables"
    checkpoint = next(step.glob("*/provenance.json")).parent
    for path in outputs:
        assert path.read_bytes() == (checkpoint / path.name).read_bytes()
    created = {
        path: json.loads(
            (tmp_path / "output/tables/.clamped-comparison-provenance.json").read_text()
        )["created_at"]
        for path in outputs[:1]
    }
    outputs[0].unlink()
    build_module.build_comparison_tables(
        {"tables": {"study_clamped_components": components}}
    )
    assert outputs[0].is_file()
    record = json.loads(
        (tmp_path / "output/tables/.clamped-comparison-provenance.json").read_text()
    )
    assert record["created_at"] == created[outputs[0]]

    # Upstream provenance change rebuilds with a new fingerprint.
    provenance.write_text('{"generation": 2}\n', encoding="utf-8")
    build_module.build_comparison_tables(
        {"tables": {"study_clamped_components": components}}
    )
    changed = json.loads(
        (tmp_path / "output/tables/.clamped-comparison-provenance.json").read_text()
    )
    assert changed["fingerprint"] != record["fingerprint"]


def test_analysis_tables_reuse_ignore_provenance_timestamp_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dubanevics2026topological.build as build_module

    output = tmp_path / "output"
    pairs = output / "analysis/component_pairs.parquet"
    pairs.parent.mkdir(parents=True)
    pairs.write_bytes(b"pairs-v1")
    provenance = output / "analysis/provenance.json"
    provenance.write_text(
        json.dumps({"fingerprint": {"config": 1}, "created_at": "a"}),
        encoding="utf-8",
    )
    tables_source = tmp_path / "src/dubanevics2026topological/tables/__init__.py"
    tables_source.parent.mkdir(parents=True)
    tables_source.write_text("# tables", encoding="utf-8")
    calls: list[str] = []

    def export(analysis: object, directory: Path) -> list[Path]:
        calls.append("export")
        paths = [directory / "event_table.csv", directory / "schema.json"]
        for path in paths:
            path.write_text("v1", encoding="utf-8")
        return paths

    def export_pairs(frame: object, directory: Path) -> list[Path]:
        path = directory / "network_component_persistence.csv"
        path.write_text("pairs", encoding="utf-8")
        return [path]

    monkeypatch.setattr(build_module, "ROOT", tmp_path)
    monkeypatch.setattr(build_module, "OUTPUT", output)
    monkeypatch.setattr(build_module, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(build_module, "ensure_analysis_checkpoint", lambda: object())
    monkeypatch.setattr(
        build_module,
        "load_or_compute_component_persistence",
        lambda *, force=False: (object(), None),
    )
    monkeypatch.setattr(build_module, "export_tables", export)
    monkeypatch.setattr(
        build_module, "export_component_persistence_table", export_pairs
    )

    paths = build_module.build_tables()
    assert {path.name for path in paths} == {
        "event_table.csv",
        "schema.json",
        "network_component_persistence.csv",
    }
    step = output / "checkpoints/analysis_tables"
    checkpoint = next(step.glob("*/provenance.json")).parent
    for path in paths:
        assert path.read_bytes() == (checkpoint / path.name).read_bytes()

    # A re-published analysis with only a new timestamp does not invalidate.
    provenance.write_text(
        json.dumps({"fingerprint": {"config": 1}, "created_at": "b"}),
        encoding="utf-8",
    )
    paths[0].unlink()
    build_module.build_tables()
    assert calls == ["export"]
    assert paths[0].is_file()

    # Changed science or component pairs rebuild; force rebuilds too.
    pairs.write_bytes(b"pairs-v2")
    build_module.build_tables()
    assert calls == ["export"] * 2
    provenance.write_text(
        json.dumps({"fingerprint": {"config": 2}, "created_at": "b"}),
        encoding="utf-8",
    )
    build_module.build_tables()
    assert calls == ["export"] * 3
    build_module.build_tables(force=True)
    assert calls == ["export"] * 4


@pytest.mark.parametrize(
    "figure_args", [[], ["--figure", "peak_uncertainty"]]
)
def test_all_command_runs_every_stage_stubbed(
    monkeypatch: pytest.MonkeyPatch, figure_args: list[str]
) -> None:
    import dubanevics2026topological.build as build_module

    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.candidate_edge_evidence",
        SimpleNamespace(load_or_compute_candidate_edge_evidence=lambda **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.clamped_cache",
        SimpleNamespace(load_or_compute_clamped_comparison=lambda **kwargs: {}),
    )
    _stub_the_candidate_prerequisites(monkeypatch, [])
    for name in (
        "ensure_analysis_checkpoint",
        "build_tables",
        "build_comparison_tables",
        "build_codon_entropy_tables",
        "build_peak_tables",
        "build_leading_saddle_band_tables",
        "build_biological_tables",
        "build_manuscript_quantities",
        "build_figures",
    ):
        monkeypatch.setattr(build_module, name, lambda *a, **k: [])
    monkeypatch.setattr(
        "dubanevics2026topological.build.pooled_scatter_product.build_pooled_scatter_product",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.threshold_diagnostics_product.build_threshold_diagnostics_product",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.bootstrap_band_quantile_scan_product.build_bootstrap_band_quantile_scan_product",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.bootstrap_codon_entropy_product.build_bootstrap_codon_entropy_product",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.graph_tree_persistence_product.build_graph_tree_persistence_product",
        lambda *a, **k: [],
    )

    assert build_module.main(["all", *figure_args]) == 0


def test_comparison_checkpoint_bytes_survive_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterisation (ADR 0013, Phase 3): a forced recompute rewrites no byte.

    Phase 3 publishes the comparison checkpoint as a version behind a link, and
    its acceptance claim is that a reader of ``clamped_comparison/`` sees the same
    bytes as before. That is only possible if the bytes are already a function of
    the inputs alone. The real stage runs here on the toy landscape: only the
    workflow that builds the full network, the reference check, the full-analysis
    stage and the fingerprint are replaced. The record has no ``created_at`` and no
    ``git`` key, so nothing is excluded from the comparison.
    """
    from _manifest import tree_manifest
    from _toy_network import WILD_TYPE, FakeWorkflow

    import dubanevics2026topological.analysis.clamped_cache as cache

    output = tmp_path / "output"
    cache_dir = output / "analysis" / "clamped_comparison"
    pl.DataFrame({"dna_from": [WILD_TYPE]}).write_parquet(tmp_path / "wt.parquet")
    pl.DataFrame({"dna_from": [WILD_TYPE]}).write_parquet(tmp_path / "diff.parquet")
    (tmp_path / "graph.graphml").write_text("graph", encoding="utf-8")
    (tmp_path / "peaks.csv").write_text("sequence\npeak\n", encoding="utf-8")
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "graphml": "graph.graphml",
                "peaks": "peaks.csv",
                "graphml_value_sha256": "graph-hash",
                "peaks_value_sha256": "peaks-hash",
                "reference_topographer_commit": "abc123",
                "published_functional_threshold": -1.0,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    # A context-less call resolves a fresh context: root it at the toy tree.
    toy_context = RunContext(root=tmp_path, output_root=output)
    monkeypatch.setattr(cache, "get_run_context", lambda: toy_context)
    monkeypatch.setattr(
        cache,
        "cache_dirs",
        lambda context=None: (
            cache_dir,
            output / "analysis" / ".clamped-comparison-pending",
        ),
    )
    monkeypatch.setattr(
        cache, "_fingerprint", lambda _, **kwargs: "toy-comparison-fingerprint"
    )
    monkeypatch.setattr(cache, "load_or_compute_analysis", lambda **kwargs: "full")
    monkeypatch.setattr(cache, "load_cached_analysis", lambda **_: "full")
    monkeypatch.setattr(cache, "load_papkou2023_article_network", lambda path: object())
    monkeypatch.setattr(
        cache, "validate_clamped_reference", lambda *args: {"matched_peaks": 1}
    )
    monkeypatch.setattr(cache, "Papkou2023Workflow", FakeWorkflow)
    monkeypatch.setattr(FakeWorkflow, "built", [])

    first = cache.load_or_compute_clamped_comparison(settings=settings)
    assert first["provenance"]["fingerprint"] == "toy-comparison-fingerprint"
    assert FakeWorkflow.built == ["full"]
    manifest = tree_manifest(cache_dir)
    assert {"provenance.json", "diagnostics.json"} <= set(manifest)
    assert {
        f"{landscape}/data/metadata.toml" for landscape in cache.LANDSCAPE_KEYS
    } <= set(manifest)
    assert sum(name.startswith("tables/") for name in manifest) == 10
    record = json.loads((cache_dir / "provenance.json").read_text(encoding="utf-8"))
    assert "created_at" not in record and "git" not in record

    # A forced recompute really recomputes, and republishes the same bytes.
    inode = cache_dir.stat().st_ino
    cache.load_or_compute_clamped_comparison(force=True, settings=settings)
    assert FakeWorkflow.built == ["full", "full"]
    assert tree_manifest(cache_dir) == manifest
    assert cache_dir.stat().st_ino != inode  # a new directory, not the old one

    # A reuse and a plain load leave every byte alone as well.
    cache.load_or_compute_clamped_comparison(settings=settings)
    cache.load_cached_clamped_comparison()
    assert FakeWorkflow.built == ["full", "full"]
    assert tree_manifest(cache_dir) == manifest
