import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from _selection import fixed_selection

from dubanevics2026topological import build
from dubanevics2026topological.analysis.produce_or_load import Stage, stage_fingerprint


def _write_cached_outputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({"dna_sequence": ["ACGTACGTA"], "fitness": [0.25]})
    for name in build.CODON_ENTROPY_TABLES:
        frame.write_parquet(directory / f"{name}.parquet")
        frame.write_csv(directory / f"{name}.csv")
    (directory / "papkou2023_codon_entropy_metadata.json").write_text(
        json.dumps({"common_saddle": {"dna_sequence": "CTCTGCTTG"}}),
        encoding="utf-8",
    )


SETTINGS = {
    "analysis": {
        "codon_entropy": {"sweep_points": 100},
        "candidate_peak_biology": {"q_value_threshold": 0.05},
    }
}


def test_candidate_leaf_view_resolves_lazy_annotations_import(monkeypatch):
    from dubanevics2026topological.analysis import biological_annotations
    from dubanevics2026topological.simplification import selected_leaf_simplification

    candidate = SimpleNamespace(trees={"split": object()})
    pairs = pl.DataFrame({"birth_sequence": ["A"], "death_sequence": ["B"]})
    calls = []

    def annotations(value, *, landscape):
        calls.append((value, landscape))
        return SimpleNamespace(pairs=pairs)

    monkeypatch.setattr(
        biological_annotations, "_build_biological_annotations", annotations
    )
    monkeypatch.setattr(
        selected_leaf_simplification, "candidate_leaf_ids", lambda *args: {"A"}
    )
    monkeypatch.setattr(
        selected_leaf_simplification,
        "absorb_singleton_leaves",
        lambda tree, ids, **kwargs: (tree, ids, kwargs),
    )
    result = selected_leaf_simplification.build_candidate_leaf_view(
        candidate, pl.DataFrame(), 0.05
    )
    assert calls == [(candidate, "candidate_clamped")]
    assert result == (
        candidate.trees["split"],
        {"A"},
        {"horizontal_layout": "subtree_extremum"},
    )


def test_codon_entropy_producer_resolves_candidate_annotations_import(
    tmp_path, monkeypatch
):
    from dubanevics2026topological.analysis import biological_annotations
    from dubanevics2026topological.simplification import selected_leaf_simplification

    candidate = SimpleNamespace(trees={"split": object()})
    pairs = pl.DataFrame({"birth_sequence": ["A"], "death_sequence": ["B"]})
    calls = []
    monkeypatch.setattr(
        biological_annotations,
        "_build_biological_annotations",
        lambda value, **kwargs: (
            calls.append((value, kwargs)) or SimpleNamespace(pairs=pairs)
        ),
    )
    monkeypatch.setattr(
        selected_leaf_simplification, "candidate_leaf_ids", lambda *args: set()
    )
    monkeypatch.setattr(
        selected_leaf_simplification,
        "absorb_singleton_leaves",
        lambda *args, **kwargs: "view",
    )
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.codon_entropy_populations.extract_codon_entropy_populations",
        lambda **kwargs: _write_cached_outputs(kwargs["output_dir"]),
    )
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.codon_entropy.process_codon_entropy_datasets",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "dubanevics2026topological.landscape.selection.resolve_selection",
        fixed_selection(0.0),
    )
    build._produce_codon_entropy_tables(
        tmp_path,
        comparison={"landscapes": {"candidate_clamped": candidate}},
        settings=SETTINGS,
        peak_data={"tables": {"candidate_clamped_contrasts": pl.DataFrame()}},
    )
    assert calls == [(candidate, {"landscape": "candidate_clamped"})]


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Run the codon-entropy table stage with stubbed science; return the call log."""
    calls = {"extract": 0, "entropy": 0, "settings": {"version": 1}}
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    monkeypatch.setattr(build, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(
        build,
        "_codon_entropy_stage",
        lambda settings=None: Stage("codon-entropy", settings=calls["settings"]),
    )
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.peak_uncertainty.load_or_compute_peak_uncertainty",
        lambda **kwargs: {"tables": {"candidate_clamped_contrasts": pl.DataFrame()}},
    )
    monkeypatch.setattr(
        "dubanevics2026topological.simplification.selected_leaf_simplification.build_candidate_leaf_view",
        lambda *args: "candidate-view",
    )

    def extract(**kwargs):
        calls["extract"] += 1
        calls["output_dir"] = kwargs["output_dir"]
        _write_cached_outputs(Path(kwargs["output_dir"]))

    def entropy(**kwargs):
        calls["entropy"] += 1
        assert kwargs["input_dir"] == kwargs["output_dir"] == calls["output_dir"]

    monkeypatch.setattr(
        "dubanevics2026topological.analysis.codon_entropy_populations.extract_codon_entropy_populations",
        extract,
    )
    monkeypatch.setattr(
        "dubanevics2026topological.landscape.selection.resolve_selection",
        fixed_selection(0.0),
    )
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.codon_entropy.process_codon_entropy_datasets",
        entropy,
    )
    return calls


def _build(**kwargs):
    return build.build_codon_entropy_tables(
        comparison={"landscapes": {"candidate_clamped": object()}}, **kwargs
    )


def test_first_build_exports_checkpoint_bytes_and_record(tmp_path, stage):
    outputs = _build()

    assert len(outputs) == len(build.CODON_ENTROPY_TABLES) * 2 + 1
    assert all(path.parent == tmp_path / "tables" for path in outputs)
    step = tmp_path / "checkpoints" / "codon_entropy"
    checkpoint = next(step.glob("*/provenance.json")).parent
    for path in outputs:
        assert path.read_bytes() == (checkpoint / path.name).read_bytes()
    record = json.loads(
        (tmp_path / "tables/.codon-entropy-provenance.json").read_text()
    )
    assert record["fingerprint"] == stage_fingerprint(build._codon_entropy_stage())
    assert set(record["artefacts"]) == {path.name for path in outputs}
    assert stage["extract"] == stage["entropy"] == 1


def test_current_fingerprint_skips_population_extraction(stage):
    _build()
    _build()

    assert stage["extract"] == 1


def test_missing_export_is_restored_from_checkpoint_without_recompute(tmp_path, stage):
    _build()
    missing = tmp_path / "tables/papkou2023_tcl_aa_entropy.csv"
    expected = missing.read_bytes()
    missing.unlink()

    outputs = _build()

    assert stage["extract"] == 1
    assert missing.read_bytes() == expected
    assert all(path.is_file() for path in outputs)


def test_changed_scientific_declaration_rebuilds(stage):
    _build()
    stage["settings"] = {"version": 2}
    _build()

    assert stage["extract"] == 2


def test_force_rebuilds(stage):
    _build()
    _build(force=True)

    assert stage["extract"] == 2


def test_failed_rebuild_keeps_published_checkpoint_and_no_record(
    tmp_path, stage, monkeypatch
):
    _build()
    stage["settings"] = {"version": 2}

    def boom(**kwargs):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(
        "dubanevics2026topological.analysis.codon_entropy_populations.extract_codon_entropy_populations",
        boom,
    )
    with pytest.raises(RuntimeError, match="extraction failed"):
        _build()

    checkpoint = json.loads(
        next((tmp_path / "checkpoints/codon_entropy").glob("*/provenance.json")).read_text()
    )
    assert checkpoint["declared"]["settings"] == {"version": 1}


def test_peak_is_made_current_before_the_stage_hashes_its_provenance(
    tmp_path, stage, monkeypatch
):
    order = []
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.peak_uncertainty.load_or_compute_peak_uncertainty",
        lambda **kwargs: (
            order.append("peak")
            or {"tables": {"candidate_clamped_contrasts": pl.DataFrame()}}
        ),
    )
    monkeypatch.setattr(
        build,
        "_codon_entropy_stage",
        lambda settings=None: (
            order.append("stage") or Stage("codon-entropy", settings={"v": 1})
        ),
    )

    _build()

    assert order[:2] == ["peak", "stage"]
    assert order.count("peak") == 1
