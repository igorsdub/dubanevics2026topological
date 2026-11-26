"""Integration checks that both cache publishers preserve stale valid state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from _selection import fixed_mixture_selection, fixed_selection
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.analysis import checkpoint
from dubanevics2026topological.landscape import selection as candidate_selection


def _inject_stage_promotion_failure(
    monkeypatch: pytest.MonkeyPatch, target: Path, stage_prefix: str
) -> None:
    original_replace = os.replace

    def fail_stage(source: str | os.PathLike[str], target: str | os.PathLike[str]):
        source_path, target_path = Path(source), Path(target)
        if source_path.name.startswith(stage_prefix) and target_path == target:
            raise OSError("injected second-rename failure")
        return original_replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", fail_stage)


def _spy_retry_hints(monkeypatch: pytest.MonkeyPatch, module) -> dict[str, str | None]:
    """Record the ``retry_hint`` each checkpoint helper is called with."""
    hints: dict[str, str | None] = {}
    for name in ("recover_checkpoint", "publish_checkpoint"):
        real = getattr(module, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            hints[_name] = kwargs.get("retry_hint")
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return hints


def test_clamped_publisher_rolls_back_stale_checkpoint_but_keeps_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological.analysis import clamped_cache as module

    output = tmp_path / "output"
    cache_dir = output / "analysis" / "clamped_comparison"
    pending = output / "analysis" / ".clamped-comparison-pending"
    cache_dir.mkdir(parents=True)
    (cache_dir / "provenance.json").write_text(
        json.dumps({"fingerprint": {"generation": "stale"}}), encoding="utf-8"
    )
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "graphml": "graph.graphml",
                "peaks": "peaks.csv",
                "graphml_value_sha256": "graph-hash",
                "peaks_value_sha256": "peaks-hash",
                "reference_topographer_commit": "commit",
                "published_functional_threshold": 0.0,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    monkeypatch.setattr(module, "cache_dirs", lambda context=None: (cache_dir, pending))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_settings", lambda **kwargs: settings)
    monkeypatch.setattr(module, "load_or_compute_analysis", lambda **kwargs: object())
    monkeypatch.setattr(module, "_fingerprint", lambda _, **kwargs: "expected")
    hints = _spy_retry_hints(monkeypatch, module)
    monkeypatch.setattr(
        module.pl,
        "read_parquet",
        lambda _: pl.DataFrame({"dna_from": ["WT"]}),
    )
    monkeypatch.setattr(module.pl, "read_csv", lambda _: pl.DataFrame())

    class Workflow:
        def __init__(self, *, landscape, fitness_wt, fitness_diff):
            self.fitness_wt = fitness_wt
            self.config = {"landscape": landscape}

        def build_network(self):
            vertices = pl.DataFrame({"dna_sequence": ["WT"], "fitness": [0.5]})
            edges = pl.DataFrame(
                {
                    "source": pl.Series([], dtype=pl.String),
                    "target": pl.Series([], dtype=pl.String),
                }
            )
            graph = Graph.from_tables(
                vertices,
                edges,
                vertex_id_col="dna_sequence",
                source_id_col="source",
                target_id_col="target",
                vertex_attribute_cols=("fitness",),
            )
            return ScalarNetwork(graph, scalar_attr="fitness")

    monkeypatch.setattr(module, "Papkou2023Workflow", Workflow)
    monkeypatch.setattr(module, "load_papkou2023_article_network", lambda _: object())
    prepared = SimpleNamespace(
        network=Workflow(
            landscape="full", fitness_wt=pl.DataFrame(), fitness_diff=pl.DataFrame()
        ).build_network(),
        diagnostics={
            "edge_selection_rule": "at_least_one_endpoint_raw_fitness_ge_threshold",
            "selected_component_functional_vertex_count": 1,
        },
        components=pl.DataFrame(),
        excluded_vertices=pl.DataFrame(),
        peaks=pl.DataFrame(),
    )
    monkeypatch.setattr(module, "prepare_clamped_landscape", lambda *args: prepared)
    monkeypatch.setattr(
        candidate_selection, "prepare_clamped_landscape", lambda *a, **k: prepared
    )
    monkeypatch.setattr(module, "validate_clamped_reference", lambda *args: {})
    monkeypatch.setattr(
        module, "analyze_prepared_landscape", lambda *args, **kwargs: object()
    )

    def save_analysis(analysis, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metadata.toml").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(module, "save_network_analysis", save_analysis)
    monkeypatch.setattr(
        module,
        "_tree_support_tables",
        lambda name, analysis: {f"{name}_support": pl.DataFrame()},
    )

    def validate(
        directory,
        *,
        allow_pending=False,
        include_full=True,
        pending_path=None,
        context=None,
    ):
        assert pending_path is None or pending_path == pending
        if pending.exists() and not allow_pending:
            raise RuntimeError("comparison refresh is pending")
        provenance_path = directory / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError("provenance missing")
        return {
            "provenance": json.loads(provenance_path.read_text(encoding="utf-8")),
            "landscapes": {"full": None},
        }

    monkeypatch.setattr(module, "_load_clamped_checkpoint", validate)
    _inject_stage_promotion_failure(
        monkeypatch, cache_dir, ".clamped-comparison-stage-"
    )

    with pytest.raises(OSError, match="injected second-rename failure"):
        module.load_or_compute_clamped_comparison(settings=settings)

    # Both checkpoint helpers name the command that repairs this checkpoint.
    assert hints == {
        "recover_checkpoint": "just comparison-analysis",
        "publish_checkpoint": "just comparison-analysis",
    }

    assert json.loads((cache_dir / "provenance.json").read_text())["fingerprint"] == {
        "generation": "stale"
    }
    assert pending.is_file()
    with pytest.raises(RuntimeError, match="refresh is pending"):
        module.load_cached_clamped_comparison()


def test_peak_publisher_rolls_back_stale_checkpoint_but_keeps_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological.analysis import peak_uncertainty as module

    output = tmp_path / "output"
    analysis_dir = output / "analysis"
    cache_dir = analysis_dir / "peak_uncertainty"
    pending = analysis_dir / ".peak-uncertainty-pending"
    cache_dir.mkdir(parents=True)
    (cache_dir / "provenance.json").write_text(
        json.dumps({"fingerprint": {"generation": "stale"}}), encoding="utf-8"
    )
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "published_functional_threshold": 0.0,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    monkeypatch.setattr(
        module, "cache_paths", lambda context=None: (cache_dir, pending)
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        candidate_selection,
        "resolve_candidate_selection",
        fixed_selection(1.0),
    )
    monkeypatch.setattr(module, "load_settings", lambda **kwargs: settings)
    monkeypatch.setattr(module, "_fingerprint", lambda _, **kwargs: "expected")
    hints = _spy_retry_hints(monkeypatch, module)
    monkeypatch.setattr(module.pl, "read_parquet", lambda _: pl.DataFrame())

    class Workflow:
        def __init__(self, *, landscape, fitness_wt, fitness_diff):
            pass

        def build_network(self):
            return SimpleNamespace(vertex_count=1, edge_count=0)

    monkeypatch.setattr(module, "Papkou2023Workflow", Workflow)
    fake_landscape = SimpleNamespace(
        diagnostics={
            "edge_selection_rule": "at_least_one_endpoint_raw_fitness_ge_threshold"
        }
    )
    monkeypatch.setattr(
        module, "prepare_clamped_landscape", lambda *args: fake_landscape
    )
    monkeypatch.setattr(
        candidate_selection, "prepare_clamped_landscape", lambda *a, **k: fake_landscape
    )
    names = [
        f"{key}_{suffix}"
        for key in module.LANDSCAPE_KEYS
        for suffix in module._TABLE_SUFFIXES
    ] + ["pairwise_sign_disagreements"]
    products = {
        "tables": {name: pl.DataFrame() for name in names},
        "diagnostics": {
            "clamped_thresholds": {"study_clamped": 0.0, "candidate_clamped": 1.0},
            "global_bh_family_size": 0,
        },
    }
    monkeypatch.setattr(module, "build_peak_uncertainty", lambda *args: products)

    def validate(directory, *, allow_pending=False, context=None):
        if pending.exists() and not allow_pending:
            raise RuntimeError("peak uncertainty refresh is pending")
        provenance_path = directory / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError("provenance missing")
        return {
            "provenance": json.loads(provenance_path.read_text(encoding="utf-8")),
            "tables": {},
            "diagnostics": {},
        }

    monkeypatch.setattr(module, "_load_peak_checkpoint", validate)
    _inject_stage_promotion_failure(monkeypatch, cache_dir, ".peak-uncertainty-stage-")

    with pytest.raises(OSError, match="injected second-rename failure"):
        module.load_or_compute_peak_uncertainty(settings=settings)

    assert hints == {
        "recover_checkpoint": "just peak-analysis",
        "publish_checkpoint": "just peak-analysis",
    }

    assert json.loads((cache_dir / "provenance.json").read_text())["fingerprint"] == {
        "generation": "stale"
    }
    assert pending.is_file()
    with pytest.raises(RuntimeError, match="refresh is pending"):
        module.load_cached_peak_uncertainty()


def test_clamped_stage_records_and_selects_with_the_rounded_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rounded value (not the unrounded one) selects, is summarised and is recorded.

    g = 0.0 and m = 0.1232, so the unrounded T is 0.2464 and the 3-decimal T is
    0.246: a stage that passed ``.unrounded`` would be caught by the sentinels.
    """
    from dubanevics2026topological.analysis import clamped_cache as module

    output = tmp_path / "output"
    cache_dir = output / "analysis" / "clamped_comparison"
    pending = output / "analysis" / ".clamped-comparison-pending"
    cache_dir.parent.mkdir(parents=True)
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "graphml": "graph.graphml",
                "peaks": "peaks.csv",
                "graphml_value_sha256": "graph-hash",
                "peaks_value_sha256": "peaks-hash",
                "reference_topographer_commit": "commit",
                "published_functional_threshold": 0.0,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 3,
                },
            },
        }
    }
    monkeypatch.setattr(module, "cache_dirs", lambda context=None: (cache_dir, pending))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_or_compute_analysis", lambda **kwargs: object())
    monkeypatch.setattr(module, "_fingerprint", lambda _, **kwargs: "expected")
    monkeypatch.setattr(
        module.pl, "read_parquet", lambda _: pl.DataFrame({"dna_from": ["a"]})
    )
    monkeypatch.setattr(module.pl, "read_csv", lambda _: pl.DataFrame())

    class Workflow:
        def __init__(self, *, landscape, fitness_wt, fitness_diff):
            self.fitness_wt = fitness_wt

        def build_network(self):
            vertices = pl.DataFrame(
                {"dna_sequence": ["a", "b", "c"], "fitness": [0.0, 0.1232, 0.2462]}
            )
            edges = pl.DataFrame(
                {"source": ["a", "b"], "target": ["b", "c"]},
                schema={"source": pl.String, "target": pl.String},
            )
            graph = Graph.from_tables(
                vertices,
                edges,
                vertex_id_col="dna_sequence",
                source_id_col="source",
                target_id_col="target",
                vertex_attribute_cols=("fitness",),
            )
            return ScalarNetwork(graph, scalar_attr="fitness")

    monkeypatch.setattr(module, "Papkou2023Workflow", Workflow)
    monkeypatch.setattr(module, "load_papkou2023_article_network", lambda _: object())
    prepared = SimpleNamespace(
        network=Workflow(
            landscape="full", fitness_wt=pl.DataFrame(), fitness_diff=pl.DataFrame()
        ).build_network(),
        diagnostics={
            "edge_selection_rule": "at_least_one_endpoint_raw_fitness_ge_threshold",
            "selected_component_functional_vertex_count": 2,
        },
        components=pl.DataFrame(),
        excluded_vertices=pl.DataFrame(),
        peaks=pl.DataFrame(),
    )
    prepared_thresholds: list[float] = []

    def prepare(network, threshold, **kwargs):
        prepared_thresholds.append(threshold)
        return prepared

    summary_args: list[tuple] = []

    def summary(*args, **kwargs):
        summary_args.append(args)
        return {"schema": module.SUMMARY_SCHEMA, "metrics": {}}

    monkeypatch.setattr(module, "prepare_clamped_landscape", prepare)
    monkeypatch.setattr(candidate_selection, "prepare_clamped_landscape", prepare)
    from dubanevics2026topological.analysis import graph_summary

    monkeypatch.setattr(graph_summary, "_summary_payload", summary)
    monkeypatch.setattr(module, "validate_clamped_reference", lambda *args: {})
    monkeypatch.setattr(
        module, "analyze_prepared_landscape", lambda *args, **kwargs: object()
    )

    def save_analysis(analysis, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metadata.toml").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(module, "save_network_analysis", save_analysis)
    monkeypatch.setattr(
        module,
        "_tree_support_tables",
        lambda name, analysis: {f"{name}_support": pl.DataFrame()},
    )

    def validate(directory, *, allow_pending=False, include_full=True, context=None):
        provenance = json.loads((directory / "provenance.json").read_text("utf-8"))
        return {"provenance": provenance, "landscapes": {"full": None}}

    monkeypatch.setattr(module, "_load_clamped_checkpoint", validate)

    result = module.load_or_compute_clamped_comparison(settings=settings)

    record = json.loads(
        (cache_dir / module.CANDIDATE_THRESHOLD_FILE).read_text("utf-8")
    )
    assert record == {
        "rule": "median_reflected",
        "decimals": 3,
        "median": 0.1232,
        "global_minimum": 0.0,
        "median_gap": 0.1232,
        "unrounded_threshold": pytest.approx(0.2464),
        "rounded_threshold": 0.246,
        "functional_vertex_count": None,
        "uncertain_vertex_count": None,
        "selected_component_functional_vertex_count": 2,
    }
    assert result["candidate_threshold"]["rounded_threshold"] == 0.246
    # Published first, then the Candidate graph at the rounded threshold.
    assert prepared_thresholds == [0.0, 0.246]
    assert summary_args[0][3:] == (0.0, 0.246)
    provenance = json.loads((cache_dir / "provenance.json").read_text("utf-8"))
    assert provenance["thresholds"]["candidate_clamped"] == 0.246


def test_clamped_stage_counts_the_functional_class_under_the_mixture_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The classes decide: the record and the summary count Functional vertices.

    The raw fitness of a, b, c is 0.0, 0.1232, 0.2462 and T is 0.2, so the raw rule
    would call only c functional; the classes call c functional, b Uncertain and a
    Non-functional, and the Published landscape (0.0) still counts raw fitness. The
    Candidate graph is b - c.
    """
    from dubanevics2026topological.analysis import clamped_cache as module

    output = tmp_path / "output"
    cache_dir = output / "analysis" / "clamped_comparison"
    pending = output / "analysis" / ".clamped-comparison-pending"
    cache_dir.parent.mkdir(parents=True)
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "graphml": "graph.graphml",
                "peaks": "peaks.csv",
                "graphml_value_sha256": "graph-hash",
                "peaks_value_sha256": "peaks-hash",
                "reference_topographer_commit": "commit",
                "published_functional_threshold": 0.0,
                "candidate_functional_threshold": {
                    "rule": "mixture_quantile",
                    "decimals": 6,
                },
            },
        }
    }
    classes = {"a": "non_functional", "b": "uncertain", "c": "functional"}
    monkeypatch.setattr(module, "cache_dirs", lambda context=None: (cache_dir, pending))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_or_compute_analysis", lambda **kwargs: object())
    monkeypatch.setattr(module, "_fingerprint", lambda _, **kwargs: "expected")
    monkeypatch.setattr(
        candidate_selection,
        "resolve_candidate_selection",
        fixed_mixture_selection(classes, threshold=0.2),
    )
    monkeypatch.setattr(
        module.pl, "read_parquet", lambda _: pl.DataFrame({"dna_from": ["a"]})
    )
    monkeypatch.setattr(module.pl, "read_csv", lambda _: pl.DataFrame())

    class Workflow:
        def __init__(self, *, landscape, fitness_wt, fitness_diff):
            self.fitness_wt = fitness_wt

        def build_network(self):
            vertices = pl.DataFrame(
                {"dna_sequence": ["a", "b", "c"], "fitness": [0.0, 0.1232, 0.2462]}
            )
            edges = pl.DataFrame(
                {"source": ["a", "b"], "target": ["b", "c"]},
                schema={"source": pl.String, "target": pl.String},
            )
            graph = Graph.from_tables(
                vertices,
                edges,
                vertex_id_col="dna_sequence",
                source_id_col="source",
                target_id_col="target",
                vertex_attribute_cols=("fitness",),
            )
            return ScalarNetwork(graph, scalar_attr="fitness")

    monkeypatch.setattr(module, "Papkou2023Workflow", Workflow)
    monkeypatch.setattr(module, "load_papkou2023_article_network", lambda _: object())
    monkeypatch.setattr(module, "validate_clamped_reference", lambda *args: {})
    monkeypatch.setattr(
        module, "analyze_prepared_landscape", lambda *args, **kwargs: object()
    )

    def save_analysis(analysis, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metadata.toml").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(module, "save_network_analysis", save_analysis)
    monkeypatch.setattr(
        module,
        "_tree_support_tables",
        lambda name, analysis: {f"{name}_support": pl.DataFrame()},
    )

    def validate(directory, *, allow_pending=False, include_full=True, context=None):
        provenance = json.loads((directory / "provenance.json").read_text("utf-8"))
        return {"provenance": provenance, "landscapes": {"full": None}}

    monkeypatch.setattr(module, "_load_clamped_checkpoint", validate)

    result = module.load_or_compute_clamped_comparison(settings=settings)

    record = json.loads(
        (cache_dir / module.CANDIDATE_THRESHOLD_FILE).read_text("utf-8")
    )
    assert record["rule"] == "mixture_quantile"
    assert module.CANDIDATE_THRESHOLD_KEYS <= record.keys()
    # Over the whole classification, as documented: c, and b Uncertain.
    assert record["functional_vertex_count"] == 1
    assert record["uncertain_vertex_count"] == 1
    # Inside the retained component {b, c}: c only.
    assert record["selected_component_functional_vertex_count"] == 1
    candidate = result["graph_summary"]["metrics"]["Candidate"]
    assert candidate["Number of nodes"] == 2
    assert candidate["Number of functional sequences"] == 1
    assert candidate["Number of non-functional sequences"] == 0
    assert candidate["Number of uncertain sequences"] == 1
    assert candidate["Percentage of functional sequences"] == 50.0
    # Published keeps the raw-fitness count at its threshold (0.0): all three.
    published = result["graph_summary"]["metrics"]["Published"]
    assert published["Number of functional sequences"] == 3
    assert published["Number of uncertain sequences"] is None
    provenance = json.loads((cache_dir / "provenance.json").read_text("utf-8"))
    assert provenance["edge_selection_rule"] == (
        "at_least_one_endpoint_functional_class"
    )
    assert provenance["published_edge_selection_rule"] == module.EDGE_SELECTION_RULE
