"""All-edge coverage, fixed-family correction and exploratory boundaries."""

import polars as pl
import pytest
from _selection import fixed_mixture_selection, fixed_selection
from test_peak_uncertainty import _inputs, _prepared_landscape

from dubanevics2026topological.analysis.candidate_edge_evidence import (
    build_candidate_edge_evidence,
    qualify_edges,
)
from dubanevics2026topological.landscape import selection as candidate_selection


def test_all_edges_keep_raw_orientation_and_full_family():
    _, diff = _inputs()
    result = build_candidate_edge_evidence(diff, _prepared_landscape())
    edges = result["edges"]
    assert edges.height == 4
    assert result["diagnostics"]["global_bh_family_size"] == 5
    row = edges.filter(pl.col("contrast_id") == "peak>valley").row(0, named=True)
    assert row["reported_delta"] == -0.3
    assert row["fitness_delta"] == pytest.approx(-0.3)
    assert row["q_value"] == pytest.approx(0.05)
    assert row["fitness_lower"] == 0.2
    assert row["fitness_upper"] == 0.5
    reduced = build_candidate_edge_evidence(diff.head(4), _prepared_landscape())[
        "edges"
    ]
    assert (
        reduced.filter(pl.col("contrast_id") == "peak>valley")["q_value"][0]
        != row["q_value"]
    )


def test_strict_boundary_and_extremes():
    edges = pl.DataFrame({"q_value": [0.0, 0.05, 1.0]})
    assert qualify_edges(edges, 0.05)["qualifies"].to_list() == [False, False, True]
    assert qualify_edges(edges, 1.0)["qualifies"].sum() == 0
    with pytest.raises(ValueError):
        qualify_edges(edges, float("nan"))


def test_missing_all_ids_and_ambiguous_source():
    _, diff = _inputs()
    with pytest.raises(ValueError, match=r"low\|peak.*low_two\|peak"):
        build_candidate_edge_evidence(
            diff.filter(~pl.col("dna_to").is_in(["low", "low_two"])),
            _prepared_landscape(),
        )
    with pytest.raises(ValueError, match="Duplicate unordered"):
        build_candidate_edge_evidence(
            pl.concat([diff, diff.head(1)]), _prepared_landscape()
        )


def test_zero_delta_off_graph_and_q_extremes_survive():
    _, diff = _inputs()
    source = diff.with_columns(
        pl.when(pl.col("dna_to") == "valley")
        .then(0.0)
        .when(pl.col("dna_to") == "low")
        .then(1.0)
        .otherwise(pl.col("pval"))
        .alias("pval"),
        pl.when(pl.col("dna_to") == "unselected_to")
        .then(0.0)
        .otherwise(pl.col("s"))
        .alias("s"),
    )
    result = build_candidate_edge_evidence(source, _prepared_landscape())
    assert result["diagnostics"]["global_bh_family_size"] == 5
    assert result["edges"]["q_value"].min() == 0
    assert result["edges"]["q_value"].max() == 1


def test_endpoint_uncertainty_and_invalid_pair_identity():
    wt, diff = _inputs()
    result = build_candidate_edge_evidence(diff, _prepared_landscape(), wt)
    assert (
        result["edges"].filter(pl.col("dna_left") == "peak")[
            "fitness_standard_error_left"
        ][0]
        == 0.05
    )
    invalid = diff.with_columns(
        pl.when(pl.col("dna_to") == "low")
        .then(-1.0)
        .otherwise(pl.col("se"))
        .alias("se")
    )
    with pytest.raises(ValueError, match=r"peak\|low"):
        build_candidate_edge_evidence(invalid, _prepared_landscape())


def test_atomic_checkpoint_reuses_and_detects_corruption(tmp_path, monkeypatch):
    import types

    from dubanevics2026topological.analysis import candidate_edge_evidence as module

    wt, diff = _inputs()
    wt.write_parquet(tmp_path / "wt.parquet")
    diff.write_parquet(tmp_path / "diff.parquet")
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        candidate_selection,
        "resolve_candidate_selection",
        fixed_selection(-1.0),
    )
    monkeypatch.setattr(
        module, "cache_dir", lambda context=None: tmp_path / "checkpoint"
    )
    monkeypatch.setattr(
        module, "_fingerprint", lambda settings, **kwargs: {"fixture": 1}
    )
    calls = []

    def workflow(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            build_network=lambda: _prepared_landscape().network
        )

    monkeypatch.setattr(module, "Papkou2023Workflow", workflow)
    built = module.load_or_compute_candidate_edge_evidence(settings=settings)
    assert built["edges"].height == 4
    assert len(calls) == 1
    module.load_or_compute_candidate_edge_evidence(settings=settings)
    assert len(calls) == 1
    # Appearance readers only load the checkpoint; graph rebuilding cannot occur.
    monkeypatch.setattr(
        module, "Papkou2023Workflow", lambda **kwargs: pytest.fail("unexpected science")
    )
    assert module.load_cached_candidate_edge_evidence()["edges"].height == 4
    (module.cache_dir() / "edges.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.load_cached_candidate_edge_evidence()


def test_source_contrast_disagreement_remains_inspectable():
    _, diff = _inputs()
    source = diff.with_columns(
        pl.when(pl.col("dna_to") == "valley")
        .then(0.3)
        .otherwise(pl.col("s"))
        .alias("s")
    )
    edges = build_candidate_edge_evidence(source, _prepared_landscape())["edges"]
    assert edges.filter(pl.col("contrast_id") == "peak>valley")[
        "reported_raw_sign_disagreement"
    ][0]


def test_atomic_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import types

    from dubanevics2026topological.analysis import candidate_edge_evidence as module

    wt, diff = _inputs()
    wt.write_parquet(tmp_path / "wt.parquet")
    diff.write_parquet(tmp_path / "diff.parquet")
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        candidate_selection,
        "resolve_candidate_selection",
        fixed_selection(-1.0),
    )
    monkeypatch.setattr(
        module, "cache_dir", lambda context=None: tmp_path / "checkpoint"
    )
    monkeypatch.setattr(
        module, "_fingerprint", lambda settings, **kwargs: {"fixture": 1}
    )
    monkeypatch.setattr(
        module,
        "Papkou2023Workflow",
        lambda **kwargs: types.SimpleNamespace(
            build_network=lambda: _prepared_landscape().network
        ),
    )
    previous = module.load_or_compute_candidate_edge_evidence(settings=settings)
    replace = module.os.replace

    def fail_stage(source, destination):
        if str(source).split("/")[-1].startswith(".candidate-evidence-") and str(
            source
        ) != str(module.cache_dir().with_name(".candidate-evidence-previous")):
            raise OSError("fixture publication failure")
        replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_stage)
    with pytest.raises(OSError, match="publication failure"):
        module.load_or_compute_candidate_edge_evidence(force=True, settings=settings)
    assert (
        module.load_cached_candidate_edge_evidence()["provenance"]
        == previous["provenance"]
    )


def test_edge_evidence_runs_on_the_functional_class_under_the_mixture_rule(
    tmp_path, monkeypatch
):
    """Only the class selects: one Functional vertex keeps one edge, not four."""
    import types

    from dubanevics2026topological.analysis import candidate_edge_evidence as module

    wt, diff = _inputs()
    wt.write_parquet(tmp_path / "wt.parquet")
    diff.write_parquet(tmp_path / "diff.parquet")
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "mixture_quantile",
                    "decimals": 6,
                },
            },
        }
    }
    classes = {
        "peak": "uncertain",
        "valley": "non_functional",
        "second_peak": "functional",
        "low": "non_functional",
        "low_two": "non_functional",
    }
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        candidate_selection,
        "resolve_candidate_selection",
        fixed_mixture_selection(classes, threshold=-1.0),
    )
    monkeypatch.setattr(
        module, "cache_dir", lambda context=None: tmp_path / "checkpoint"
    )
    monkeypatch.setattr(
        module, "_fingerprint", lambda settings, **kwargs: {"fixture": 1}
    )
    monkeypatch.setattr(
        module,
        "Papkou2023Workflow",
        lambda **kwargs: types.SimpleNamespace(
            build_network=lambda: _prepared_landscape().network
        ),
    )
    built = module.load_or_compute_candidate_edge_evidence(settings=settings)
    edges = built["edges"]
    assert edges.height == 1
    assert edges["contrast_id"].to_list() == ["valley>second_peak"]
