"""Ticket 12b (#137) Slice 1 tests: RunContext replaces environment variables for sequence-growth-rates.

Tests for:
1. serializer round trip (serialize_run_context <-> parse_run_context)
2. relocation invariance of paths and scientific identities
3. no whole-context hashing (D19)
4. no_draw polarity (FITNESSTREE_BOOTSTRAP_NO_DRAW semantics)
5. context in job.json outside identity
6. context isolation across multiple executions in the same process
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.results import (
    RunContext,
    parse_run_context,
    serialize_run_context,
)
from dubanevics2026topological.results.context import REPO_ROOT
from dubanevics2026topological.results.hashing import (
    EXCLUDED_FROM_SCIENTIFIC_HASH,
    build_identity_document,
    compute_scientific_hash,
)


def test_serializer_round_trip(tmp_path: Path) -> None:
    """A single function serializes RunContext to CLI flags, and the parser reconstructs it."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "reference").mkdir(parents=True)
    (root / "custom_out").mkdir(parents=True)
    (root / "custom_barseq").mkdir(parents=True)
    (root / "custom_chunks").mkdir(parents=True)
    (root / "custom_fonts").mkdir(parents=True)

    ctx = RunContext(
        output_root=root / "custom_out",
        raw_data_dir=root / "data" / "raw",
        barseq_export_dir=root / "custom_barseq",
        bootstrap_chunk_dir=root / "custom_chunks",
        font_dir=root / "custom_fonts",
        worker_concurrency=8,
        variant_name="fdr_by_q0.05_papkou_original",
        no_draw_allowed=True,
        root=root,
        env={},
    )

    flags = serialize_run_context(ctx)

    # Flags must carry paths relative to project root
    assert "--output-root" in flags
    out_idx = flags.index("--output-root")
    assert flags[out_idx + 1] == "custom_out"

    assert "--processed-root" in flags
    processed_idx = flags.index("--processed-root")
    assert flags[processed_idx + 1] == "data/processed"

    assert "--raw-data-dir" in flags
    raw_idx = flags.index("--raw-data-dir")
    assert flags[raw_idx + 1] == "data/raw"

    assert "--barseq-export-dir" in flags
    barseq_idx = flags.index("--barseq-export-dir")
    assert flags[barseq_idx + 1] == "custom_barseq"

    assert "--bootstrap-chunk-dir" in flags
    chunk_idx = flags.index("--bootstrap-chunk-dir")
    assert flags[chunk_idx + 1] == "custom_chunks"

    assert "--font-dir" in flags
    font_idx = flags.index("--font-dir")
    assert flags[font_idx + 1] == "custom_fonts"

    assert "--worker-concurrency" in flags
    workers_idx = flags.index("--worker-concurrency")
    assert flags[workers_idx + 1] == "8"

    assert "--analysis-variant" in flags
    variant_idx = flags.index("--analysis-variant")
    assert flags[variant_idx + 1] == "fdr_by_q0.05_papkou_original"

    assert "--no-draw" in flags

    # Host/run/pins are retired and must NOT be serialized
    assert "--host" not in flags
    assert "--run-id" not in flags

    # Reconstruct via parser
    reconstructed = parse_run_context(flags, root=root, env={})
    assert reconstructed.output_root == ctx.output_root
    assert reconstructed.processed_root == ctx.processed_root
    assert reconstructed.raw_data_dir == ctx.raw_data_dir
    assert reconstructed.barseq_export_dir == ctx.barseq_export_dir
    assert reconstructed.bootstrap_chunk_dir == ctx.bootstrap_chunk_dir
    assert reconstructed.font_dir == ctx.font_dir
    assert reconstructed.worker_concurrency == 8
    assert reconstructed.variant_name == "fdr_by_q0.05_papkou_original"
    assert reconstructed.no_draw_allowed is True
    assert reconstructed.root == root


def test_serializer_overrides_ambient_no_draw(tmp_path: Path) -> None:
    ctx = RunContext(root=tmp_path, no_draw_allowed=False, env={})
    restored = parse_run_context(
        serialize_run_context(ctx), env={"FITNESSTREE_BOOTSTRAP_NO_DRAW": "1"}
    )
    assert restored.no_draw_allowed is False


def test_serializer_round_trip_defaults(tmp_path: Path) -> None:
    """Round-trip with optional fields unset preserves None/default values."""
    root = tmp_path / "repo_defaults"
    root.mkdir()
    ctx = RunContext(
        output_root=root / "output",
        raw_data_dir=root / "data" / "raw",
        no_draw_allowed=False,
        root=root,
        env={},
    )

    flags = serialize_run_context(ctx)
    assert "--no-draw" not in flags
    assert "--barseq-export-dir" not in flags
    assert "--bootstrap-chunk-dir" not in flags
    assert "--font-dir" not in flags
    assert "--worker-concurrency" not in flags
    assert "--analysis-variant" not in flags

    reconstructed = parse_run_context(flags, root=root, env={})
    assert reconstructed.output_root == ctx.output_root
    assert reconstructed.processed_root == ctx.processed_root
    assert reconstructed.raw_data_dir == ctx.raw_data_dir
    assert reconstructed.barseq_export_dir is None
    assert reconstructed.bootstrap_chunk_dir is None
    assert reconstructed.font_dir is None
    assert reconstructed.worker_concurrency is None
    assert reconstructed.variant_name is None
    assert reconstructed.no_draw_allowed is False
    assert reconstructed.root == root


def test_relocation_invariance(tmp_path: Path) -> None:
    """Paths in context dictionary are relative to project root and invariant under relocation."""
    repo1 = tmp_path / "location_a"
    repo2 = tmp_path / "location_b"
    repo1.mkdir()
    repo2.mkdir()

    ctx1 = RunContext(
        output_root=repo1 / "output",
        raw_data_dir=repo1 / "data" / "raw",
        root=repo1,
        env={},
    )
    ctx2 = RunContext(
        output_root=repo2 / "output",
        raw_data_dir=repo2 / "data" / "raw",
        root=repo2,
        env={},
    )

    d1 = ctx1.to_dict()
    d2 = ctx2.to_dict()
    assert d1 == d2
    assert d1["output_root"] == "output"
    assert d1["processed_root"] == "data/processed"
    assert d1["raw_data_dir"] == "data/raw"
    assert "host" not in d1
    assert "run_id" not in d1


def test_no_whole_context_hashing() -> None:
    """Changing execution/context paths does NOT change the scientific identity (D19)."""
    assert "context" in EXCLUDED_FROM_SCIENTIFIC_HASH

    params = {"thr": "published", "fdr": "none"}
    doc1 = build_identity_document(
        product_type="growth_rates",
        scientific_parameters=params,
    )

    # Passing execution context inside parameters must be filtered out
    doc2 = build_identity_document(
        product_type="growth_rates",
        scientific_parameters={**params, "context": {"output_root": "different"}},
    )
    assert doc1 == doc2

    hash1, short1 = compute_scientific_hash(
        product_type="growth_rates",
        scientific_parameters=params,
    )
    hash2, short2 = compute_scientific_hash(
        product_type="growth_rates",
        scientific_parameters={**params, "context": {"output_root": "different"}},
    )
    assert hash1 == hash2
    assert short1 == short2


def test_no_draw_polarity() -> None:
    """no_draw polarity preserves FITNESSTREE_BOOTSTRAP_NO_DRAW semantics."""
    from dubanevics2026topological.simplification.bootstrap_chunks import (
        no_draw_requested,
    )

    # When no_draw_allowed is False:
    ctx_false = RunContext(no_draw_allowed=False)
    assert ctx_false.no_draw_allowed is False
    assert no_draw_requested(context=ctx_false) is False
    assert ctx_false.env_vars()["FITNESSTREE_BOOTSTRAP_NO_DRAW"] == "0"
    flags_false = serialize_run_context(ctx_false)
    assert "--no-draw" not in flags_false
    reconstructed_false = parse_run_context(flags_false)
    assert reconstructed_false.no_draw_allowed is False

    # When no_draw_allowed is True:
    ctx_true = RunContext(no_draw_allowed=True)
    assert ctx_true.no_draw_allowed is True
    assert no_draw_requested(context=ctx_true) is True
    assert ctx_true.env_vars()["FITNESSTREE_BOOTSTRAP_NO_DRAW"] == "1"
    flags_true = serialize_run_context(ctx_true)
    assert "--no-draw" in flags_true
    reconstructed_true = parse_run_context(flags_true)
    assert reconstructed_true.no_draw_allowed is True



def test_context_digest_is_canonical_and_outside_scientific_identity(
    tmp_path: Path,
) -> None:
    from dubanevics2026topological.results.context import (
        context_record,
        context_record_from_mapping,
    )

    first = context_record(RunContext(root=tmp_path, output_root=tmp_path / "a"))
    second = context_record(RunContext(root=tmp_path, output_root=tmp_path / "b"))
    reordered = dict(reversed(list(first["resolved_context"].items())))
    assert context_record_from_mapping(reordered)["sha256"] == first["sha256"]
    assert first["sha256"] != second["sha256"]
    assert compute_scientific_hash(
        product_type="growth_rates", scientific_parameters={"context": first}
    ) == compute_scientific_hash(
        product_type="growth_rates", scientific_parameters={"context": second}
    )


def test_annotation_comparison_contexts_do_not_cross_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dubanevics2026topological import build
    from dubanevics2026topological.analysis import clamped_cache

    contexts = [
        RunContext(root=tmp_path, output_root=tmp_path / name) for name in ("a", "b")
    ]
    for index, context in enumerate(contexts):
        directory = context.output_root / "analysis" / "clamped_comparison"
        directory.mkdir(parents=True)
        (directory / "provenance.json").write_text(
            json.dumps({"fingerprint": str(index)})
        )

    def read(directory: Path, **kwargs: object) -> dict:
        return json.loads((directory / "provenance.json").read_text())

    monkeypatch.setattr(clamped_cache, "_load_clamped_checkpoint", read)
    fingerprints = []
    for index in (0, 1, 0, 1):
        context = contexts[index]
        assert clamped_cache.load_cached_clamped_comparison(context=context)[
            "fingerprint"
        ] == str(index)
        fingerprints.append(build._comparison_analysis_fingerprint(context))
    assert fingerprints[0] == fingerprints[2]
    assert fingerprints[1] == fingerprints[3]
    assert fingerprints[0] != fingerprints[1]
    (
        contexts[1].output_root / "analysis" / "clamped_comparison" / "provenance.json"
    ).unlink()
    with pytest.raises(FileNotFoundError):
        clamped_cache.load_cached_clamped_comparison(context=contexts[1])
    with pytest.raises(FileNotFoundError):
        build._comparison_analysis_fingerprint(contexts[1])


def test_growth_checkpoint_records_producer_context(tmp_path: Path) -> None:
    from dubanevics2026topological.analysis.sequence_growth_rates import (
        SequenceGrowthRatesResult,
        _write_checkpoint,
    )
    from dubanevics2026topological.results.context import context_record

    context = RunContext(root=tmp_path, output_root=tmp_path / "out")
    result = SequenceGrowthRatesResult(pl.DataFrame({"sequence": ["AAA"]}), {}, {})
    _write_checkpoint(result, tmp_path, {"test": 1}, context=context)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["execution_context"] == context_record(context)


def test_context_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple contexts executed in the same process do not leak output/cache paths."""
    from dubanevics2026topological import build as build_module
    from dubanevics2026topological.analysis import sequence_growth_rates

    root = REPO_ROOT
    out1 = tmp_path / "run_first"
    out2 = tmp_path / "run_second"

    ctx1 = RunContext(output_root=out1, root=root)
    ctx2 = RunContext(output_root=out2, root=root)

    fake_rates1 = SimpleNamespace(
        growth_rates=pl.DataFrame({"dna_sequence": ["AAAAAA"], "growth_rate": [1.0]}),
        provenance={"fingerprint": {"synthetic": 1}, "artifacts": {}},
        diagnostics={"sequence_count": 1, "max_identity_gap": 0.0},
    )
    fake_rates2 = SimpleNamespace(
        growth_rates=pl.DataFrame({"dna_sequence": ["CCCCCC"], "growth_rate": [2.0]}),
        provenance={"fingerprint": {"synthetic": 2}, "artifacts": {}},
        diagnostics={"sequence_count": 1, "max_identity_gap": 0.0},
    )

    # Execute first with ctx1 CLI flags
    monkeypatch.setattr(
        sequence_growth_rates,
        "compute_sequence_growth_rates",
        lambda *args, **kwargs: fake_rates1,
    )
    flags1 = ["sequence-growth-rates", *serialize_run_context(ctx1)]
    assert build_module.main(flags1) == 0

    assert (out1 / "tables" / "papkou2023_sequence_growth_rates.csv").is_file()
    assert (out1 / "tables" / ".sequence-growth-rates-provenance.json").is_file()
    assert not out2.exists()

    # Execute second with ctx2 CLI flags
    monkeypatch.setattr(
        sequence_growth_rates,
        "compute_sequence_growth_rates",
        lambda *args, **kwargs: fake_rates2,
    )
    flags2 = ["sequence-growth-rates", *serialize_run_context(ctx2)]
    assert build_module.main(flags2) == 0

    assert (out2 / "tables" / "papkou2023_sequence_growth_rates.csv").is_file()
    assert (out2 / "tables" / ".sequence-growth-rates-provenance.json").is_file()

    # Verify out1 was NOT overwritten or contaminated by run 2
    prov1 = json.loads(
        (out1 / "tables" / ".sequence-growth-rates-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    prov2 = json.loads(
        (out2 / "tables" / ".sequence-growth-rates-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert prov1["fingerprint"] != prov2["fingerprint"]
    assert prov1["declared"]["settings"]["growth_rate_checkpoint"]["fingerprint"] == {
        "synthetic": 1
    }
    assert prov2["declared"]["settings"]["growth_rate_checkpoint"]["fingerprint"] == {
        "synthetic": 2
    }



def test_processed_root_defaults_to_data_processed(tmp_path):
    ctx = RunContext(root=tmp_path, env={})
    assert ctx.processed_root == (tmp_path / "data" / "processed").resolve()


def test_processed_root_from_env_flag_and_argument(tmp_path):
    from_env = RunContext(
        root=tmp_path, env={"FITNESSTREE_PROCESSED_ROOT": "elsewhere"}
    )
    assert from_env.processed_root == (tmp_path / "elsewhere").resolve()
    explicit = RunContext(root=tmp_path, processed_root="p", env={})
    assert explicit.processed_root == (tmp_path / "p").resolve()
    parsed = parse_run_context(
        ["--processed-root", "flagged"], root=tmp_path, env={}
    )
    assert parsed.processed_root == (tmp_path / "flagged").resolve()
    assert explicit.env_vars()["FITNESSTREE_PROCESSED_ROOT"] == str(
        explicit.processed_root
    )
    assert explicit.to_dict()["processed_root"] == "p"
    assert ["--processed-root", "p"] == serialize_run_context(explicit)[
        serialize_run_context(explicit).index("--processed-root") :
        serialize_run_context(explicit).index("--processed-root") + 2
    ]


def test_retired_fields_are_gone(tmp_path):
    context = RunContext(root=tmp_path, env={})
    for retired in ("reference_root", "pinned_runs", "product_roots"):
        assert not hasattr(context, retired)
