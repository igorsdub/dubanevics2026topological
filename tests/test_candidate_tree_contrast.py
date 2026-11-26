from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological.analysis import candidate_tree_contrast as module
from dubanevics2026topological.analysis.barseq_inputs import BarseqExport
from dubanevics2026topological.analysis.candidate_tree_contrast import (
    build_candidate_tree_contrast,
    compute_candidate_tree_contrast,
    enumerate_tree_edges,
    load_cached_candidate_tree_contrast,
)
from dubanevics2026topological.landscape.selected_landscape import (
    analyze_prepared_landscape,
    prepare_clamped_landscape,
)

CONFIG = {
    "alpha": 0.05,
    "elapsed_time": 1.0,
    "time_unit": "generations",
    "correction": "benjamini_hochberg",
    "family": "candidate_split_tree_edges",
}
# Path a-b-c-d-e-f-g. Leaves f, b, d; saddles e and c (root). Tree edges:
# f|e, d|e, b|c, e|c.
DNA = list("abcdefg")
RAW_FITNESS = [-0.1, 0.5, -0.2, 0.4, -0.15, 0.6, 0.1]
# True growth rates behind the synthetic counts. Leaf d grows exactly like its
# saddle e, so its contrast is not significant although the raw tree says d > e.
TRUE_RATE = {
    "a": -0.1,
    "b": 0.7,
    "c": -0.2,
    "d": -0.15,
    "e": -0.15,
    "f": 0.8,
    "g": 0.1,
}


def _tree(fitness=RAW_FITNESS):
    vertices = pl.DataFrame(
        {
            "dna_sequence": DNA,
            "fitness": fitness,
            "aa_sequence": ["A"] * 7,
            "has_stop_codon": [False] * 7,
        }
    )
    edges = pl.DataFrame({"source": list("abcdef"), "target": list("bcdefg")})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "aa_sequence", "has_stop_codon"),
    )
    prepared = prepare_clamped_landscape(
        ScalarNetwork(graph, scalar_attr="fitness"), -1.0
    )
    analysis = analyze_prepared_landscape(
        prepared, name="synthetic", wild_type_sequence="a"
    )
    return analysis.trees["split"]


def _export(
    rates: dict[str, float] = TRUE_RATE, *, drop: tuple[str, ...] = ()
) -> BarseqExport:
    rng = np.random.default_rng(20261015)
    sample_ids = ["s1", "s2", "s3", "s4"]
    times = {"s1": 0, "s2": 1, "s3": 0, "s4": 1}
    sequence_ids = [f"id_{name}" for name in DNA]
    rows = []
    for name, sequence_id in zip(DNA, sequence_ids, strict=True):
        for sample_id in sample_ids:
            mean = 4000 * np.exp(rates[name] * times[sample_id])
            rows.append((sequence_id, sample_id, int(rng.poisson(mean))))
    counts = pl.DataFrame(
        rows, schema=["sequence_id", "sample_id", "count"], orient="row"
    )
    totals = counts.group_by("sample_id").agg(
        pl.col("count").sum().alias("sample_total")
    )
    samples = pl.DataFrame(
        {
            "sample_id": sample_ids,
            "replicate_id": ["r1", "r1", "r2", "r2"],
            "condition": ["c"] * 4,
            "time": [times[sample_id] for sample_id in sample_ids],
        }
    ).join(totals, on="sample_id")
    sequences = pl.DataFrame({"sequence_id": sequence_ids, "sequence": DNA}).filter(
        ~pl.col("sequence").is_in(drop)
    )
    counts = counts.filter(
        pl.col("sequence_id").is_in(sequences["sequence_id"].to_list())
    )
    return BarseqExport(counts, samples, sequences, Path("."))


def _wt() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dna_from": ["a"] * 6,
            "dna_to": ["b", "c", "d", "e", "z", "g"],
            "s": [0.1] * 6,
            "se": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )


def _fake_fit(monkeypatch, edit) -> None:
    real = module.barseqfit.fit_pair

    def fit(counts, samples, sequences, **kwargs):
        return edit(real(counts, samples, sequences, **kwargs))

    monkeypatch.setattr(module.barseqfit, "fit_pair", fit)


def test_enumeration_columns_orientation_and_terminals():
    edges = enumerate_tree_edges(_tree())

    assert edges.columns == list(module.PAIR_COLUMNS)
    assert edges.height == 4
    assert edges["edge_id"].to_list() == ["b|c", "d|e", "e|c", "f|e"]
    row = edges.filter(pl.col("edge_id") == "f|e").row(0, named=True)
    assert row["upper_sequence"] == "f"
    assert row["lower_sequence"] == "e"
    assert row["raw_fitness_delta"] == pytest.approx(0.75)
    assert row["is_terminal"] is True
    assert row["regular_count"] == 1
    interior = edges.filter(pl.col("edge_id") == "e|c").row(0, named=True)
    assert interior["is_terminal"] is False
    assert interior["lower_in_degree"] == 2
    assert edges.filter(pl.col("is_terminal")).height == 3


def test_enumeration_rejects_forest():
    graph = _tree().tree.copy()
    graph.add_node({"dna_sequence": "x", "scalar": 0.0, "event": "maximum"})
    with pytest.raises(ValueError, match="one root"):
        enumerate_tree_edges(SimpleNamespace(tree=graph))


def test_enumeration_rejects_node_without_identity():
    graph = _tree().tree.copy()
    graph[graph.node_indices()[0]].pop("dna_sequence")
    with pytest.raises(ValueError, match="dna_sequence"):
        enumerate_tree_edges(SimpleNamespace(tree=graph))


def test_fit_orients_lower_to_upper_and_separates_leaves():
    result = build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)
    edges = result.edges.sort("edge_id")

    assert edges.height == 4
    assert edges["from"].to_list() == ["id_c", "id_e", "id_c", "id_e"]
    assert edges["to"].to_list() == ["id_b", "id_d", "id_e", "id_f"]
    by_id = {row["edge_id"]: row for row in edges.iter_rows(named=True)}
    assert by_id["f|e"]["contrast_estimate"] > 0.5
    assert by_id["f|e"]["qualifies"] is False
    assert by_id["b|c"]["qualifies"] is False
    assert by_id["d|e"]["qualifies"] is True
    assert by_id["d|e"]["p_value"] > 0.05
    assert all(status == "converged" for status in edges["status"])
    assert json.loads(by_id["f|e"]["diagnostics"])["converged"] is True
    assert by_id["f|e"]["contrast_gap"] == pytest.approx(
        by_id["f|e"]["contrast_estimate"] - 0.75
    )
    assert by_id["f|e"]["fitness_standard_error_upper"] is None
    assert by_id["f|e"]["fitness_standard_error_lower"] == pytest.approx(0.04)
    assert by_id["d|e"]["fitness_standard_error_upper"] == pytest.approx(0.03)
    diagnostics = result.diagnostics
    assert diagnostics["bh_family"] == "candidate_split_tree_edges"
    assert diagnostics["bh_family_size"] == diagnostics["tree_edge_count"] == 4
    assert diagnostics["qualifying_count"] == int(edges["qualifies"].sum())
    assert diagnostics["terminal_qualifying_count"] == 1
    assert diagnostics["barseqfit"]["version"]
    assert result.pairs.columns[-2:] == ["from", "to"]


def test_bh_family_is_exactly_the_tree_edges(monkeypatch):
    seen = []

    def spy(values):
        seen.append(len(values))
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(module, "compute_bh_qvalues", spy)
    build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)
    assert seen == [4]


def test_qualification_is_strict_at_alpha(monkeypatch):
    alpha = CONFIG["alpha"]
    monkeypatch.setattr(
        module,
        "compute_bh_qvalues",
        lambda values: np.array([alpha, np.nextafter(alpha, 1.0), 0.0, 1.0]),
    )
    result = build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)
    assert result.edges["q_value"].to_list()[:2] == [alpha, np.nextafter(alpha, 1.0)]
    assert result.edges["qualifies"].to_list() == [False, True, False, True]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_alpha_is_validated(alpha):
    with pytest.raises(ValueError, match="alpha"):
        build_candidate_tree_contrast(
            _tree(), _wt(), _export(), {**CONFIG, "alpha": alpha}
        )


def test_wrong_family_is_rejected():
    with pytest.raises(ValueError, match="family"):
        build_candidate_tree_contrast(
            _tree(), _wt(), _export(), {**CONFIG, "family": "all_graph_edges"}
        )


def test_sequence_absent_from_export_names_edges():
    with pytest.raises(ValueError, match=r"absent.*'d\|e'$"):
        build_candidate_tree_contrast(_tree(), _wt(), _export(drop=("d",)), CONFIG)


def _no_adjustment(monkeypatch):
    def forbidden(values):
        raise AssertionError("adjustment must not run after a failed fit")

    monkeypatch.setattr(module, "compute_bh_qvalues", forbidden)


def _edit_row(frame: pl.DataFrame, target: str = "id_f", **changes) -> pl.DataFrame:
    row = pl.col("to") == target
    return frame.with_columns(
        *[
            pl.when(row).then(pl.lit(value)).otherwise(pl.col(name)).alias(name)
            for name, value in changes.items()
        ]
    )


def test_missing_fit_fails_closed(monkeypatch):
    _no_adjustment(monkeypatch)
    _fake_fit(monkeypatch, lambda frame: frame.filter(pl.col("to") != "id_f"))
    with pytest.raises(ValueError, match=r"missing fit: 'f\|e'"):
        build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)


def test_duplicate_fit_fails_closed(monkeypatch):
    _no_adjustment(monkeypatch)
    _fake_fit(
        monkeypatch,
        lambda frame: pl.concat([frame, frame.filter(pl.col("to") == "id_f")]),
    )
    with pytest.raises(ValueError, match=r"duplicate fit: 'f\|e'"):
        build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"se": float("nan")}, "standard error"),
        ({"se": 0.0}, "standard error"),
        ({"m": float("inf")}, "estimate"),
        ({"p": 1.5}, "p-value"),
        ({"status": "failed"}, "status"),
        ({"status": "not_converged"}, "status"),
        ({"status": None}, "status"),
        ({"p": float("nan")}, "p-value"),
        ({"p": None}, "p-value"),
        ({"p": -0.1}, "p-value"),
    ],
)
def test_bad_fit_rows_fail_closed(monkeypatch, changes, message):
    _no_adjustment(monkeypatch)
    _fake_fit(monkeypatch, lambda frame: _edit_row(frame, **changes))
    with pytest.raises(ValueError, match=message + r".*'f\|e'"):
        build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)


def test_extra_unrequested_fit_rows_fail_closed(monkeypatch):
    _no_adjustment(monkeypatch)
    _fake_fit(
        monkeypatch,
        lambda frame: pl.concat(
            [frame, frame.head(1).with_columns(pl.lit("id_zz").alias("to"))]
        ),
    )
    with pytest.raises(ValueError, match="untested pairs"):
        build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)


@pytest.mark.parametrize(
    "column", ["m", "se", "p", "status", "diagnostics", "elapsed_time", "time_unit"]
)
def test_missing_fit_column_fails_closed(monkeypatch, column):
    _no_adjustment(monkeypatch)
    _fake_fit(monkeypatch, lambda frame: frame.drop(column))
    with pytest.raises(ValueError, match=f"lacks columns.*{column}"):
        build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)


def test_sequence_with_several_ids_fails_closed():
    export = _export()
    sequences = pl.concat(
        [export.sequences, pl.DataFrame({"sequence_id": ["id_d2"], "sequence": ["d"]})]
    )
    export = BarseqExport(export.counts, export.samples, sequences, Path("."))
    with pytest.raises(ValueError, match=r"several sequence_id.*'d\|e'"):
        build_candidate_tree_contrast(_tree(), _wt(), export, CONFIG)


def test_sign_disagreement_zero_is_no_evidence():
    m = np.array([0.0, 0.3, -0.3, 0.3, 0.0, -0.3])
    delta = np.array([0.5, 0.0, 0.5, 0.5, 0.0, -0.5])
    assert module._sign_disagreement(m, delta).tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]


def test_zero_raw_delta_is_not_flagged(monkeypatch):
    tree = _tree()
    graph = tree.tree
    for index in graph.node_indices():
        if graph[index]["dna_sequence"] in ("d", "e"):
            graph[index]["scalar"] = 0.4
    _fake_fit(monkeypatch, lambda frame: _edit_row(frame, "id_d", m=-0.3))
    result = build_candidate_tree_contrast(tree, _wt(), _export(), CONFIG)
    row = result.edges.filter(pl.col("edge_id") == "d|e").row(0, named=True)
    assert row["raw_fitness_delta"] == 0.0
    assert row["sign_disagreement"] is False


def test_inverted_edge_fitness_is_rejected():
    graph = _tree().tree.copy()
    for index in graph.node_indices():
        if graph[index]["dna_sequence"] == "f":
            graph[index]["scalar"] = -5.0
    with pytest.raises(ValueError, match=r"upper_fitness below.*'f\|e'"):
        enumerate_tree_edges(SimpleNamespace(tree=graph))


def test_root_error_lists_identities_not_indices():
    graph = _tree().tree.copy()
    graph.add_node({"dna_sequence": "x", "scalar": 0.0, "event": "maximum"})
    with pytest.raises(ValueError, match=r"dna_sequence='x'.*event='maximum'|event="):
        enumerate_tree_edges(SimpleNamespace(tree=graph))


def test_non_finite_diagnostics_are_refused():
    assert json.loads(module._diagnostics_json({"a": 1})) == {"a": 1}
    with pytest.raises(ValueError):
        module._diagnostics_json({"a": float("nan")})
    with pytest.raises(ValueError):
        module._diagnostics_json({"a": float("inf")})


def test_sign_disagreement_is_flagged(monkeypatch):
    # d is raw-higher than e (delta +0.55) but the contrast estimate is negative.
    _fake_fit(monkeypatch, lambda frame: _edit_row(frame, "id_d", m=-0.3))
    result = build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)
    flagged = result.edges.filter(pl.col("sign_disagreement"))
    assert flagged["edge_id"].to_list() == ["d|e"]
    assert flagged["contrast_gap"].to_list() == [pytest.approx(-0.3 - 0.55)]
    assert result.diagnostics["sign_disagreement_count"] == 1
    assert result.diagnostics["sign_disagreement_edge_ids"] == ["d|e"]


def test_checkpoint_publishes_reuses_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_dir = tmp_path / "output/analysis/candidate_tree_contrast"
    pending = tmp_path / "output/analysis/.candidate-tree-contrast.pending"
    monkeypatch.setattr(
        module, "cache_paths", lambda context=None: (cache_dir, pending)
    )
    export = _export()
    fingerprint = {"generation": 1}
    builds = []

    def build(settings, **kwargs):
        builds.append(settings)
        return _tree(), _wt()

    monkeypatch.setattr(module, "load_barseq_export", lambda settings: export)
    monkeypatch.setattr(
        module, "_fingerprint", lambda settings, export, **kwargs: fingerprint
    )
    monkeypatch.setattr(module, "build_candidate_tree", build)
    settings = {"analysis": {"candidate_tree_contrast": CONFIG}}

    first = compute_candidate_tree_contrast(settings)
    assert len(builds) == 1
    assert not pending.exists()
    for name in module.ARTIFACTS + ("provenance.json",):
        assert (cache_dir / name).is_file()
    assert first.provenance["fingerprint"] == fingerprint

    again = compute_candidate_tree_contrast(settings)
    assert len(builds) == 1
    assert again.edges.equals(first.edges)
    assert load_cached_candidate_tree_contrast().edges.equals(first.edges)

    compute_candidate_tree_contrast(settings, force=True)
    assert len(builds) == 2

    fingerprint = {"generation": 2}
    monkeypatch.setattr(
        module, "_fingerprint", lambda settings, export, **kwargs: fingerprint
    )
    compute_candidate_tree_contrast(settings)
    assert len(builds) == 3

    # A corrupted artifact is quarantined and rebuilt rather than trusted.
    (cache_dir / "edges.csv").write_text("tampered\n")
    with pytest.warns(RuntimeWarning):
        compute_candidate_tree_contrast(settings)
    assert len(builds) == 4
    assert not pending.exists()


def test_cached_read_requires_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "cache_paths",
        lambda context=None: (
            tmp_path / "analysis/candidate_tree_contrast",
            tmp_path / "analysis/.candidate-tree-contrast.pending",
        ),
    )
    with pytest.raises(FileNotFoundError, match="candidate-tree-contrast"):
        load_cached_candidate_tree_contrast()


def test_cached_read_refuses_while_pending(tmp_path, monkeypatch):
    cache_dir = tmp_path / "analysis/candidate_tree_contrast"
    pending = tmp_path / "analysis/.candidate-tree-contrast.pending"
    monkeypatch.setattr(
        module, "cache_paths", lambda context=None: (cache_dir, pending)
    )
    cache_dir.mkdir(parents=True)
    (cache_dir / "provenance.json").write_text("{}")
    pending.write_text("pending\n")
    with pytest.raises(RuntimeError, match="refresh is pending"):
        load_cached_candidate_tree_contrast()


def test_tables_are_exported_with_sidecar_provenance(tmp_path, monkeypatch):
    from dubanevics2026topological import build

    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    result = build_candidate_tree_contrast(_tree(), _wt(), _export(), CONFIG)
    result.provenance = {"fingerprint": {"generation": 1}, "artifacts": {}}
    written = build.build_candidate_tree_contrast_tables(result)

    names = sorted(path.name for path in written)
    assert names == [
        "papkou2023_candidate_tree_contrast_edges.csv",
        "papkou2023_candidate_tree_contrast_edges.parquet",
        "papkou2023_candidate_tree_contrast_pairs.csv",
        "papkou2023_candidate_tree_contrast_pairs.parquet",
    ]
    exported = pl.read_parquet(
        tmp_path / "tables/papkou2023_candidate_tree_contrast_edges.parquet"
    )
    assert exported.equals(result.edges)
    sidecar = tmp_path / "tables/.candidate-tree-contrast-provenance.json"
    record = json.loads(sidecar.read_text())
    assert record["stage"] == "candidate-tree-contrast-tables"
    assert set(record["artefacts"]) == {p.name for p in written}

    step = tmp_path / "checkpoints/candidate_tree_contrast_tables"
    checkpoint = next(step.glob("*/provenance.json")).parent
    before = {p: p.read_bytes() for p in written}
    for path in written:
        assert path.read_bytes() == (checkpoint / path.name).read_bytes()
    written[0].unlink()
    again = build.build_candidate_tree_contrast_tables(result)
    assert {p: p.read_bytes() for p in again} == before
    assert json.loads(sidecar.read_text())["created_at"] == record["created_at"]

    result.provenance = {"fingerprint": {"generation": 2}, "artifacts": {}}
    build.build_candidate_tree_contrast_tables(result)
    assert json.loads(sidecar.read_text())["fingerprint"] != record["fingerprint"]
    # force re-produces: a corrupted checkpoint file is rewritten, not reused.
    current = json.loads(sidecar.read_text())["fingerprint"]
    checkpoint = next(
        path.parent
        for path in step.glob("*/provenance.json")
        if json.loads(path.read_text())["fingerprint"] == current
    )
    corrupt = checkpoint / written[1].name
    good = corrupt.read_bytes()
    corrupt.write_bytes(b"corrupt")
    build.build_candidate_tree_contrast_tables(result, force=True)
    assert corrupt.read_bytes() == good
