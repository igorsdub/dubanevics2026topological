from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dubanevics2026topological.analysis import sequence_growth_rates as module
from dubanevics2026topological.analysis.barseq_inputs import BarseqExport
from dubanevics2026topological.analysis.sequence_growth_rates import (
    build_sequence_growth_rates,
    compute_sequence_growth_rates,
    load_cached_sequence_growth_rates,
)

CONFIG = {
    "barseq_export_env": "UNUSED",
    "elapsed_time": 1.0,
    "time_unit": "generations",
    "identity_tolerance": 1e-6,
}
# Seven in-frame codon sequences, labelled a-g in the export ids; "a" is the wild
# type. d, f and g are single stop codons (TAA, TAG, TGA).
LETTERS = "abcdefg"
DNA = ["AAA", "AAC", "ACA", "TAA", "AGA", "TAG", "TGA"]
SEQ = dict(zip(LETTERS, DNA, strict=True))
LABEL = {dna: letter for letter, dna in SEQ.items()}
WILD_TYPE = SEQ["a"]
STOPS = ["TAA", "TAG", "TGA"]
# True growth rates behind the synthetic counts (as in the contrast tests).
TRUE_RATE = {
    "a": -0.1,
    "b": 0.7,
    "c": -0.2,
    "d": -0.15,
    "e": -0.15,
    "f": 0.8,
    "g": 0.1,
}


def _export(*, drop: tuple[str, ...] = ()) -> BarseqExport:
    rng = np.random.default_rng(20261016)
    sample_ids = ["s1", "s2", "s3", "s4"]
    times = {"s1": 0, "s2": 1, "s3": 0, "s4": 1}
    sequence_ids = [f"id_{name}" for name in LETTERS]
    rows = []
    for name, sequence_id in zip(LETTERS, sequence_ids, strict=True):
        for sample_id in sample_ids:
            mean = 4000 * np.exp(TRUE_RATE[name] * times[sample_id])
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
        ~pl.col("sequence").is_in([SEQ.get(name, name) for name in drop])
    )
    counts = counts.filter(
        pl.col("sequence_id").is_in(sequences["sequence_id"].to_list())
    )
    return BarseqExport(counts, samples, sequences, Path("."))


def _published_wt(export: BarseqExport, **shift: float) -> pl.DataFrame:
    """A ``fitness_wt`` table whose ``s`` is barseqfit's own ``fit_wt`` ``m``."""
    fits = module.barseqfit.fit_wt(
        export.counts,
        export.samples,
        export.sequences,
        reference_id="id_a",
        elapsed_time=CONFIG["elapsed_time"],
        time_unit=CONFIG["time_unit"],
        pseudocount="all_zeros",
        dispersion="fixed",
    )
    dna = dict(zip(export.sequences["sequence_id"], export.sequences["sequence"]))
    others = fits.filter(pl.col("sequence_id") != "id_a")
    names = [dna[sequence_id] for sequence_id in others["sequence_id"]]
    return pl.DataFrame(
        {
            "dna_from": [WILD_TYPE] * others.height,
            "dna_to": names,
            "s": [
                m + shift.get(LABEL[name], 0.0)
                for name, m in zip(names, others["m"].to_list(), strict=True)
            ],
            "se": others["se"],
        }
    )


def _build(wt, export, config=CONFIG, stop_codons=STOPS):
    return build_sequence_growth_rates(wt, export, config, stop_codons)


REAL_FIT = module.barseqfit.fit_sequences


def _fake_fit(monkeypatch, edit) -> None:
    def fit(counts, samples, sequences, **kwargs):
        return edit(REAL_FIT(counts, samples, sequences, **kwargs))

    monkeypatch.setattr(module.barseqfit, "fit_sequences", fit)


def _edit_row(frame: pl.DataFrame, target: str = "id_f", **changes) -> pl.DataFrame:
    row = pl.col("sequence_id") == target
    return frame.with_columns(
        *[
            pl.when(row).then(pl.lit(value)).otherwise(pl.col(name)).alias(name)
            for name, value in changes.items()
        ]
    )


def test_identity_passes_on_consistent_data():
    export = _export()
    result = _build(_published_wt(export), export, CONFIG)
    rates = result.growth_rates

    assert rates.columns == list(module.GROWTH_COLUMNS)
    assert rates["dna_sequence"].to_list() == sorted(DNA)
    assert rates["sequence_id"].to_list() == [f"id_{LABEL[dna]}" for dna in sorted(DNA)]
    assert rates["status"].unique().to_list() == ["converged"]
    assert rates["n_replicates"].unique().to_list() == [2]
    assert rates["se_replicate"].is_not_null().all()
    assert (rates["se_pooled"] > 0).all()
    r = dict(zip(rates["dna_sequence"], rates["r"], strict=True))
    wt = _published_wt(export)
    for name, s in zip(wt["dna_to"], wt["s"], strict=True):
        assert (r[name] - r[WILD_TYPE]) - s == pytest.approx(0.0, abs=1e-7)
    diagnostics = result.diagnostics
    assert diagnostics["sequence_count"] == 7
    assert diagnostics["wild_type_sequence"] == WILD_TYPE
    assert diagnostics["max_identity_gap"] < 1e-6
    assert diagnostics["median_se_ratio"] > 0
    assert 0.0 <= diagnostics["share_se_ratio_above_threshold"] <= 1.0
    assert diagnostics["barseqfit"]["version"]
    json.dumps(diagnostics, allow_nan=False)


def test_growth_rates_are_wild_type_inclusive_even_if_not_a_vertex():
    export = _export()
    wt = _published_wt(export)
    assert WILD_TYPE not in wt["dna_to"].to_list()
    result = _build(wt, export)
    assert WILD_TYPE in result.growth_rates["dna_sequence"].to_list()
    assert result.growth_rates.height == 7


def test_perturbed_s_breaks_the_identity():
    export = _export()
    wt = _published_wt(export, d=1e-3)
    with pytest.raises(ValueError, match=r"identity.*1 sequences.*'TAA'"):
        _build(wt, export, CONFIG)


def test_identity_gap_is_judged_against_the_tolerance():
    export = _export()
    wt = _published_wt(export, d=5e-7)
    result = _build(wt, export, CONFIG)
    assert 4e-7 < result.diagnostics["max_identity_gap"] < 1e-6
    with pytest.raises(ValueError, match="identity"):
        _build(wt, export, {**CONFIG, "identity_tolerance": 1e-7})


@pytest.mark.parametrize("tolerance", [-1e-6, float("nan"), float("inf"), True, "x"])
def test_identity_tolerance_is_validated(tolerance):
    export = _export()
    with pytest.raises(ValueError, match="identity_tolerance"):
        _build(
            _published_wt(export),
            export,
            {**CONFIG, "identity_tolerance": tolerance},
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"status": "not_converged"}, "status is not 'converged'"),
        ({"status": "failed"}, "status is not 'converged'"),
        ({"status": None}, "status is not 'converged'"),
        ({"se_replicate": None}, "replicate standard error"),
        ({"se_replicate": float("nan")}, "replicate standard error"),
        ({"se_replicate": -0.1}, "replicate standard error"),
        ({"se_pooled": 0.0}, "pooled standard error"),
        ({"se_pooled": float("nan")}, "pooled standard error"),
        ({"r": float("inf")}, "growth rate not finite"),
        ({"r": None}, "growth rate not finite"),
    ],
)
def test_bad_fit_rows_fail_closed_naming_the_sequence(monkeypatch, changes, message):
    export = _export()
    wt = _published_wt(export)
    _fake_fit(monkeypatch, lambda frame: _edit_row(frame, **changes))
    with pytest.raises(ValueError, match=message + r".*'TAG'"):
        _build(wt, export, CONFIG)


def test_every_offending_sequence_is_listed(monkeypatch):
    export = _export()
    wt = _published_wt(export)
    _fake_fit(
        monkeypatch,
        lambda frame: _edit_row(
            _edit_row(frame, "id_c", status="failed"), "id_f", status="failed"
        ),
    )
    with pytest.raises(ValueError, match=r"'ACA', 'TAG'"):
        _build(wt, export, CONFIG)


def test_missing_duplicate_and_extra_fit_rows_fail_closed(monkeypatch):
    export = _export()
    wt = _published_wt(export)
    _fake_fit(monkeypatch, lambda frame: frame.filter(pl.col("sequence_id") != "id_f"))
    with pytest.raises(ValueError, match=r"missing fit: 'TAG'"):
        _build(wt, export, CONFIG)
    _fake_fit(
        monkeypatch,
        lambda frame: pl.concat([frame, frame.filter(pl.col("sequence_id") == "id_f")]),
    )
    with pytest.raises(ValueError, match=r"duplicate fit: 'TAG'"):
        _build(wt, export, CONFIG)
    _fake_fit(
        monkeypatch,
        lambda frame: pl.concat(
            [frame, frame.head(1).with_columns(pl.lit("id_zz").alias("sequence_id"))]
        ),
    )
    with pytest.raises(ValueError, match="unrequested"):
        _build(wt, export, CONFIG)


@pytest.mark.parametrize("column", list(module.FIT_COLUMNS))
def test_missing_fit_column_fails_closed(monkeypatch, column):
    export = _export()
    wt = _published_wt(export)
    _fake_fit(monkeypatch, lambda frame: frame.drop(column))
    with pytest.raises(ValueError, match=f"lacks columns.*{column}"):
        _build(wt, export, CONFIG)


def test_wild_type_absent_from_export_is_named():
    export = _export(drop=(WILD_TYPE,))
    wt = _published_wt(_export())
    with pytest.raises(ValueError, match=r"absent.*'AAA'"):
        _build(wt, export, CONFIG)


def test_published_sequence_absent_from_export_is_named():
    export = _export(drop=("d",))
    with pytest.raises(ValueError, match=r"absent.*'TAA'"):
        _build(_published_wt(_export()), export, CONFIG)


def test_sequence_with_several_ids_fails_closed():
    export = _export()
    sequences = pl.concat(
        [
            export.sequences,
            pl.DataFrame({"sequence_id": ["id_d2"], "sequence": ["TAA"]}),
        ]
    )
    export = BarseqExport(export.counts, export.samples, sequences, Path("."))
    with pytest.raises(ValueError, match=r"several sequence_id.*'TAA'"):
        _build(_published_wt(_export()), export, CONFIG)


def test_fitness_wt_must_be_one_origin_without_duplicates():
    export = _export()
    wt = _published_wt(export)
    with pytest.raises(ValueError, match="one wild type"):
        _build(
            wt.with_columns(
                pl.when(pl.col("dna_to") == "TGA")
                .then(pl.lit("CCC"))
                .otherwise(pl.col("dna_from"))
                .alias("dna_from")
            ),
            export,
            CONFIG,
        )
    with pytest.raises(ValueError, match=r"Duplicate fitness_wt.*'AAC'"):
        _build(pl.concat([wt, wt.head(1)]), export, CONFIG)
    with pytest.raises(ValueError, match="dna_from, dna_to and s"):
        _build(wt.drop("s"), export, CONFIG)


SETTINGS = {
    "analysis": {
        module.SECTION: CONFIG,
        "codon_entropy": {"stop_codons": STOPS},
    }
}


def _stub_inputs(monkeypatch, wt: pl.DataFrame, builds: list) -> None:
    """Serve ``wt`` as fitness_wt and record every fit, with no network built."""
    real = module.build_sequence_growth_rates

    def build(*args):
        builds.append(args)
        return real(*args)

    monkeypatch.setattr(module, "load_fitness_wt", lambda settings: wt)
    monkeypatch.setattr(module, "build_sequence_growth_rates", build)


def test_checkpoint_publishes_reuses_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_dir = tmp_path / "output/analysis/sequence_growth_rates"
    pending = tmp_path / "output/analysis/.sequence-growth-rates.pending"
    monkeypatch.setattr(module, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(module, "PENDING_PATH", pending)
    export = _export()
    fingerprint = {"generation": 1}
    builds: list = []
    _stub_inputs(monkeypatch, _published_wt(export), builds)
    monkeypatch.setattr(module, "load_barseq_export", lambda settings, section: export)
    monkeypatch.setattr(module, "_fingerprint", lambda settings, export: fingerprint)
    settings = SETTINGS

    first = compute_sequence_growth_rates(settings)
    assert len(builds) == 1
    assert not pending.exists()
    for name in module.ARTIFACTS + ("provenance.json",):
        assert (cache_dir / name).is_file()
    assert first.provenance["fingerprint"] == fingerprint

    again = compute_sequence_growth_rates(settings)
    assert len(builds) == 1
    assert again.growth_rates.equals(first.growth_rates)
    assert load_cached_sequence_growth_rates().growth_rates.equals(first.growth_rates)

    compute_sequence_growth_rates(settings, force=True)
    assert len(builds) == 2

    fingerprint = {"generation": 2}
    monkeypatch.setattr(module, "_fingerprint", lambda settings, export: fingerprint)
    compute_sequence_growth_rates(settings)
    assert len(builds) == 3

    # A corrupted artifact is quarantined and rebuilt rather than trusted.
    (cache_dir / "growth_rates.parquet").write_bytes(b"tampered")
    with pytest.warns(RuntimeWarning):
        compute_sequence_growth_rates(settings)
    assert len(builds) == 4
    assert not pending.exists()


def test_failed_identity_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_dir = tmp_path / "output/analysis/sequence_growth_rates"
    pending = tmp_path / "output/analysis/.sequence-growth-rates.pending"
    monkeypatch.setattr(module, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(module, "PENDING_PATH", pending)
    export = _export()
    _stub_inputs(monkeypatch, _published_wt(export, d=1e-3), [])
    monkeypatch.setattr(module, "load_barseq_export", lambda settings, section: export)
    monkeypatch.setattr(module, "_fingerprint", lambda settings, export: {"g": 1})
    with pytest.raises(ValueError, match="identity"):
        compute_sequence_growth_rates(SETTINGS)
    assert not (cache_dir / "provenance.json").exists()
    assert not pending.exists()


def test_cached_read_requires_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module, "CACHE_DIR", tmp_path / "analysis/sequence_growth_rates"
    )
    with pytest.raises(FileNotFoundError, match="sequence-growth-rates"):
        load_cached_sequence_growth_rates()


def test_cached_read_refuses_while_pending(tmp_path, monkeypatch):
    cache_dir = tmp_path / "analysis/sequence_growth_rates"
    pending = tmp_path / "analysis/.sequence-growth-rates.pending"
    monkeypatch.setattr(module, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(module, "PENDING_PATH", pending)
    cache_dir.mkdir(parents=True)
    (cache_dir / "provenance.json").write_text("{}")
    pending.write_text("pending\n")
    with pytest.raises(RuntimeError, match="refresh is pending"):
        load_cached_sequence_growth_rates()


def test_fingerprint_changes_with_config_stop_codons_and_export(tmp_path, monkeypatch):
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "config/analysis.toml"},
            "codon_entropy": {"stop_codons": STOPS},
            module.SECTION: CONFIG,
        }
    }
    for name in ("counts", "samples", "sequences"):
        (tmp_path / f"{name}.parquet").write_bytes(name.encode())
    export = BarseqExport(*(pl.DataFrame(),) * 3, source_dir=tmp_path)
    base = module._fingerprint(settings, export)
    assert module._fingerprint(settings, export) == base
    # No threshold or Candidate-selection input: the stage runs before them.
    assert not {"candidate_functional_threshold", "candidate_threshold_rounded"} & set(
        base
    )
    names = " ".join(base["files"])
    for retired in ("candidate_threshold", "candidate_tree_contrast", "clamped"):
        assert retired not in names
    thresholded = {
        "analysis": {
            **settings["analysis"],
            "clamped": {"candidate_functional_threshold": {"rule": "other"}},
        }
    }
    assert module._fingerprint(thresholded, export) == base
    changed = {
        "analysis": {
            **settings["analysis"],
            module.SECTION: {**CONFIG, "identity_tolerance": 1e-5},
        }
    }
    assert module._fingerprint(changed, export) != base
    stops = {
        "analysis": {
            **settings["analysis"],
            "codon_entropy": {"stop_codons": ["TAA", "TAG"]},
        }
    }
    assert module._fingerprint(stops, export) != base
    (tmp_path / "counts.parquet").write_bytes(b"other")
    assert module._fingerprint(settings, export) != base


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("AAACCCGGG", 0),
        ("AAATAACCC", 1),
        ("TAATAGTGA", 3),
        ("TAATAATAA", 3),
        # Out of frame: TAA appears at offset 1 only.
        ("ATAAAAAAA", 0),
        ("", 0),
    ],
)
def test_stop_codon_count_counts_in_frame_stops(sequence, expected):
    assert module.stop_codon_count(sequence, STOPS) == expected


def test_stop_codon_count_fails_closed():
    with pytest.raises(ValueError, match="not a multiple of 3"):
        module.stop_codon_count("AAAT", STOPS)
    with pytest.raises(ValueError, match="3-nt strings.*'TA'"):
        module.stop_codon_count("AAA", ["TA", "TAG"])
    with pytest.raises(ValueError, match="non-empty"):
        module.stop_codon_count("AAA", [])


def test_s_is_r_minus_r_wt_from_the_fit_and_zero_for_the_wild_type():
    export = _export()
    rates = _build(_published_wt(export), export).growth_rates
    r = rates["r"].to_numpy()
    s = rates["s"].to_numpy()
    wt_r = rates.filter(pl.col("dna_sequence") == WILD_TYPE)["r"][0]
    assert rates["s"].dtype == pl.Float64
    assert s == pytest.approx(r - wt_r, abs=0)
    assert rates.filter(pl.col("dna_sequence") == WILD_TYPE)["s"][0] == 0.0
    assert rates.columns.index("s") == rates.columns.index("r") + 1


def test_stop_codon_count_column_matches_the_sequences():
    export = _export()
    rates = _build(_published_wt(export), export).growth_rates
    counts = dict(zip(rates["dna_sequence"], rates["stop_codon_count"], strict=True))
    assert rates["stop_codon_count"].dtype == pl.Int64
    assert counts == {
        "AAA": 0,
        "AAC": 0,
        "ACA": 0,
        "AGA": 0,
        "TAA": 1,
        "TAG": 1,
        "TGA": 1,
    }
    assert (
        _build(_published_wt(export), export, CONFIG, ["TAA"])
        .growth_rates["stop_codon_count"]
        .sum()
        == 1
    )


def test_a_sequence_that_is_not_in_frame_fails_closed():
    export = _export()
    wt = _published_wt(export).with_columns(pl.col("dna_to").str.replace("ACA", "ACAT"))
    with pytest.raises(ValueError, match="multiple of 3.*'ACAT'"):
        _build(wt, export)


def test_every_published_sequence_and_the_wild_type_are_fitted_without_a_network():
    export = _export()
    wt = _published_wt(export)
    # Only the published table decides the set: no vertex list is passed.
    fitted = _build(wt, export).growth_rates["dna_sequence"].to_list()
    assert sorted(fitted) == sorted([*wt["dna_to"].to_list(), WILD_TYPE])
    assert not hasattr(module, "build_candidate_network")


def test_tables_are_exported_with_sidecar_provenance(tmp_path, monkeypatch):
    from dubanevics2026topological import build

    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    export = _export()
    result = _build(_published_wt(export), export, CONFIG)
    result.provenance = {"fingerprint": {"generation": 1}, "artifacts": {}}
    written = build.build_sequence_growth_rate_tables(result)

    assert sorted(path.name for path in written) == [
        "papkou2023_sequence_growth_rates.csv",
        "papkou2023_sequence_growth_rates.parquet",
    ]
    exported = pl.read_parquet(
        tmp_path / "tables/papkou2023_sequence_growth_rates.parquet"
    )
    assert exported.equals(result.growth_rates)
    sidecar = tmp_path / "tables/.sequence-growth-rates-provenance.json"
    record = json.loads(sidecar.read_text())
    assert record["stage"] == "sequence-growth-rate-tables"
    assert set(record["artefacts"]) == {p.name for p in written}

    step = tmp_path / "checkpoints/sequence_growth_rate_tables"
    checkpoint = next(step.glob("*/provenance.json")).parent
    before = {p: p.read_bytes() for p in written}
    for path in written:
        assert path.read_bytes() == (checkpoint / path.name).read_bytes()
    written[0].unlink()
    again = build.build_sequence_growth_rate_tables(result)
    assert {p: p.read_bytes() for p in again} == before
    assert json.loads(sidecar.read_text())["created_at"] == record["created_at"]

    result.provenance = {"fingerprint": {"generation": 2}, "artifacts": {}}
    build.build_sequence_growth_rate_tables(result)
    assert json.loads(sidecar.read_text())["fingerprint"] != record["fingerprint"]
    current = json.loads(sidecar.read_text())["fingerprint"]
    checkpoint = next(
        path.parent
        for path in step.glob("*/provenance.json")
        if json.loads(path.read_text())["fingerprint"] == current
    )
    corrupt = checkpoint / written[1].name
    good = corrupt.read_bytes()
    corrupt.write_bytes(b"corrupt")
    build.build_sequence_growth_rate_tables(result, force=True)
    assert corrupt.read_bytes() == good


def test_old_column_checkpoint_reports_a_schema_change(tmp_path, monkeypatch):
    cache_dir = tmp_path / "analysis/sequence_growth_rates"
    monkeypatch.setattr(module, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(module, "PENDING_PATH", tmp_path / "analysis/.pending")
    export = _export()
    result = _build(_published_wt(export), export)
    result.growth_rates = result.growth_rates.drop("s", "stop_codon_count")
    cache_dir.mkdir(parents=True)
    module._write_checkpoint(result, cache_dir, {"generation": 0})
    with pytest.raises(ValueError, match=r"predates the current schema.*just seq"):
        load_cached_sequence_growth_rates()
    with pytest.raises(ValueError, match="predates") as error:
        load_cached_sequence_growth_rates()
    assert "coverage" not in str(error.value)


def test_wild_type_row_against_itself_fails_closed():
    export = _export()
    wt = _published_wt(export)
    own = pl.DataFrame(
        {"dna_from": [WILD_TYPE], "dna_to": [WILD_TYPE], "s": [0.5], "se": [0.1]}
    )
    with pytest.raises(ValueError, match=r"against itself.*'AAA'"):
        _build(pl.concat([wt, own]), export)


def test_null_dna_to_fails_closed_with_a_named_error():
    export = _export()
    wt = _published_wt(export)
    null = pl.DataFrame(
        {"dna_from": [WILD_TYPE], "dna_to": [None], "s": [0.1], "se": [0.1]},
        schema=wt.schema,
    )
    with pytest.raises(ValueError, match="null dna_to"):
        _build(pl.concat([wt, null]), export)


def test_fitness_wt_contents_move_the_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "source_key", lambda path: Path(path).name)
    (tmp_path / "uv.lock").write_text("lock")
    table = tmp_path / "fitness_wt.parquet"
    pl.DataFrame({"dna_to": ["AAC"], "s": [0.1]}).write_parquet(table)
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "fitness_wt.parquet"},
            "codon_entropy": {"stop_codons": STOPS},
            module.SECTION: CONFIG,
        }
    }
    before = module._fingerprint_without_export(settings)
    assert module._fingerprint_without_export(settings) == before
    pl.DataFrame({"dna_to": ["AAC"], "s": [0.2]}).write_parquet(table)
    assert module._fingerprint_without_export(settings) != before


def test_external_raw_directory_has_stable_fingerprint_key(tmp_path):
    from dubanevics2026topological.results import RunContext

    (tmp_path / "fitness_wt.parquet").write_bytes(b"fitness data")
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "data/raw/fitness_wt.parquet"},
            "codon_entropy": {"stop_codons": ["TAA"]},
            module.SECTION: CONFIG,
        }
    }
    context = RunContext(root=module.ROOT, raw_data_dir=tmp_path, env={})
    fingerprint = module._fingerprint_without_export(settings, context=context)
    assert "data/raw/fitness_wt.parquet" in fingerprint["files"]
    assert str(tmp_path) not in str(fingerprint)


def test_export_sequences_outside_fitness_wt_are_not_fitted():
    export = _export()
    wt = _published_wt(export).filter(pl.col("dna_to") != "TGA")
    assert "TGA" in export.sequences["sequence"].to_list()
    fitted = _build(wt, export).growth_rates["dna_sequence"].to_list()
    assert "TGA" not in fitted
    assert sorted(fitted) == sorted([*wt["dna_to"].to_list(), WILD_TYPE])
