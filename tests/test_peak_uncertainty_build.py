"""Peak builds reuse science across appearance changes and repair missing exports."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from _fake_figures import install_fake_figures

from dubanevics2026topological import build
from dubanevics2026topological.figures import registry
from dubanevics2026topological.figures.peak_uncertainty import PANEL_VARIANTS
from dubanevics2026topological.output_paths import figure_path


@pytest.fixture
def peak_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path
    source = root / "src/dubanevics2026topological/figures/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# figure source\n", encoding="utf-8")
    (source.parent / "peak_uncertainty.py").write_text("# renderer\n", encoding="utf-8")
    composer = root / "src/dubanevics2026topological/analysis/pooled_scatter.py"
    composer.parent.mkdir(parents=True)
    composer.write_text("# composer\n", encoding="utf-8")
    provenance = root / "output/analysis/peak_uncertainty/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"generation": 1}\n', encoding="utf-8")
    settings = {
        "analysis": {
            "sequence_growth_rates": {"identity_tolerance": 1e-6},
            "peak_uncertainty_scatter": {"standard_error_tolerance": 1e-3},
        },
        "figures": {
            "formats": ["svg", "png"],
            "variants": ["vanilla", "annotated"],
            "png_dpi": 300,
            "figures": {build.PEAK_FIGURE_ID: {"width": 300}},
        },
    }
    # The growth-rate checkpoint the pooled scatter is composed from: only its
    # provenance identity is read here, the composition itself is stubbed below.
    growth = SimpleNamespace(
        growth_rates=pl.DataFrame({"dna_sequence": ["AAA"]}),
        provenance={"fingerprint": {"generation": 1}, "artifacts": {"table": "h1"}},
    )
    result = {
        "tables": {
            "study_clamped_peaks": pl.DataFrame(
                {"dna_sequence": ["AAA"], "min_neighbor_score": [0.95]}
            )
        },
        "diagnostics": {"peak_count": 1},
    }
    events = []

    class Chart:
        def save(self, path, *, format, **kwargs):
            events.append(("render", format, kwargs))
            Path(path).write_text(format, encoding="utf-8")

    def assemble(payload, figure_id, settings, *, variant):
        # payload is now the pooled product content (or test stub); accept any for this build cache test
        assert payload is not None
        events.append(("assemble", variant))
        return Chart()

    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "OUTPUT", root / "output")
    monkeypatch.setattr(build, "FIGURE_DIR", root / "output/figures")
    monkeypatch.setattr(build, "RECORD_DIR", root / "output/figures/.records")
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(
        build,
        "load_cached_analysis",
        lambda: pytest.fail("A peak render must not load or rebuild trees"),
    )
    install_fake_figures(
        monkeypatch,
        SimpleNamespace(FIGURE_IDS=(build.PEAK_FIGURE_ID,), render=assemble),
    )
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.sequence_growth_rates",
        SimpleNamespace(load_cached_sequence_growth_rates=lambda: growth),
    )
    # The figure now loads four saved variant panels (no render-time compose): each one a
    # scatter table and the product's diagnostics.
    panels = {
        variant: {
            "scatter": pl.DataFrame(
                {
                    "dna_sequence": ["wt", "p"],
                    "aa_sequence": ["W", "P"],
                    "raw_fitness": [0.0, 0.1],
                    "standard_error": [0.01, 0.02],
                    "is_peak": [False, True],
                }
            ),
            "diagnostics": {"wild_type_sequence": "wt", "variant": variant},
        }
        for variant in PANEL_VARIANTS
    }
    monkeypatch.setitem(
        registry._COMPOSERS,
        build.PEAK_FIGURE_ID,
        lambda inputs, settings: {
            "panels": inputs.get("saved_figure_inputs", {}).get("peak_panels", panels)
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.peak_uncertainty",
        SimpleNamespace(
            load_cached_peak_uncertainty=lambda: result,
            load_or_compute_peak_uncertainty=lambda **kwargs: pytest.fail(
                "A render must not recompute scientific classifications"
            ),
        ),
    )
    saved = {
        "peak_panels": panels,
        "product_ids": {
            "peak_panels": {
                variant: f"deadbeef{index:08x}" for index, variant in enumerate(PANEL_VARIANTS)
            }
        },
    }
    monkeypatch.setattr(
        build,
        "load_figure_inputs",
        lambda specs, settings=None, context=None: {"saved_figure_inputs": saved},
    )
    return settings, result, events, provenance, growth


def test_canonical_build_reuses_cache_and_renders_only_after_changes(peak_build):
    settings, _, events, provenance, _ = peak_build
    paths = build.build_figures((build.PEAK_FIGURE_ID,))
    assert [p.name for p in paths] == [
        "peak_uncertainty.svg",
        "peak_uncertainty.png",
    ]
    assert events == [
        ("assemble", "vanilla"),
        # per format: untitled (canonical/) then its titled twin (canonical/titled/)
        ("render", "svg", {}),
        ("render", "svg", {}),
        ("render", "png", {"ppi": 300}),
        ("render", "png", {"ppi": 300}),
    ]
    for fmt in ("svg", "png"):
        for titled in (False, True):
            assert figure_path(
                build.OUTPUT, "peak_uncertainty", ext=fmt, titled=titled
            ).is_file()
        assert (build.RECORD_DIR / "titled" / f"peak_uncertainty.{fmt}.json").is_file()
    events.clear()
    build.build_figures((build.PEAK_FIGURE_ID,))
    assert events[0] == ("assemble", "vanilla")  # figures are not cached
    settings["figures"]["figures"][build.PEAK_FIGURE_ID]["width"] = 350
    build.build_figures((build.PEAK_FIGURE_ID,))
    assert events[0] == ("assemble", "vanilla")
    events.clear()
    # The figure reads its four variants' products, not the main configuration's peak
    # checkpoint: a new main-config provenance leaves the render current.
    provenance.write_text('{"generation": 2}\n', encoding="utf-8")
    build.build_figures((build.PEAK_FIGURE_ID,))
    assert events[0] == ("assemble", "vanilla")


@pytest.mark.parametrize("variant", PANEL_VARIANTS)
def test_each_variant_panel_identity_keys_the_figure(peak_build, monkeypatch, variant):
    """One variant's summary id changing re-renders; the record holds all four ids."""
    _, _, events, _, _ = peak_build
    original = build.load_figure_inputs
    build.build_figures((build.PEAK_FIGURE_ID,))
    events.clear()
    build.build_figures((build.PEAK_FIGURE_ID,))
    assert events[0] == ("assemble", "vanilla")

    def vary(*args, **kwargs):
        result = original(*args, **kwargs)
        saved = dict(result["saved_figure_inputs"])
        ids = dict(saved["product_ids"]["peak_panels"])
        ids[variant] = "cafe" + ids[variant][4:]
        saved["product_ids"] = {**saved["product_ids"], "peak_panels": ids}
        return {**result, "saved_figure_inputs": saved}

    monkeypatch.setattr(build, "load_figure_inputs", vary)
    build.build_figures((build.PEAK_FIGURE_ID,))
    assert events[0] == ("assemble", "vanilla")
    record = json.loads(
        (build.RECORD_DIR / "peak_uncertainty.svg.json").read_text(encoding="utf-8")
    )
    assert set(record["saved_products"]["peak_panels"]) == set(PANEL_VARIANTS)
    assert record["saved_products"]["peak_panels"][variant].startswith("cafe")


def test_a_timestamp_in_the_growth_rate_provenance_does_not_re_render(peak_build):
    _, _, events, _, growth = peak_build
    build.build_figures((build.PEAK_FIGURE_ID,))
    events.clear()
    growth.provenance["created_at"] = "2026-10-22T12:00:00"
    build.build_figures((build.PEAK_FIGURE_ID,))
    assert events[0] == ("assemble", "vanilla")


def test_annotated_peak_variant_is_rejected(peak_build):
    with pytest.raises(ValueError, match="canonical vanilla"):
        build.build_figures((build.PEAK_FIGURE_ID,), variants=("annotated",))


def test_evidence_export_repairs_deleted_file_and_records_diagnostics(peak_build):
    _, result, _, _, _ = peak_build
    paths = build.build_peak_tables(result)
    assert len(paths) == 3
    csv = next(p for p in paths if p.suffix == ".csv")
    assert pl.read_csv(csv)["dna_sequence"].to_list() == ["AAA"]
    diagnostics = next(p for p in paths if p.suffix == ".json")
    assert json.loads(diagnostics.read_text()) == result["diagnostics"]
    original_mtime = csv.stat().st_mtime_ns
    build.build_peak_tables(result)
    assert csv.stat().st_mtime_ns == original_mtime
    csv.unlink()
    build.build_peak_tables(result)
    assert csv.is_file()


def test_peak_tables_match_checkpoint_bytes_and_record_fingerprint(peak_build):
    _, result, _, provenance, _ = peak_build
    paths = build.build_peak_tables(result)
    step = provenance.parents[3] / "output/checkpoints/peak_uncertainty_tables"
    checkpoint = next(step.glob("*/provenance.json")).parent
    for path in paths:
        assert path.read_bytes() == (checkpoint / path.name).read_bytes()
    record = json.loads(
        (
            provenance.parents[3] / "output/tables/.peak-uncertainty-provenance.json"
        ).read_text()
    )
    assert set(record["artefacts"]) == {path.name for path in paths}
    # The peak checkpoint enters as the digest of its record (without a commit
    # stamp), never as the raw bytes of the file.
    assert record["declared"]["inputs"] == {}
    assert record["declared"]["settings"] == {
        "tables": sorted(result["tables"]),
        "peak_checkpoint": build._peak_analysis_fingerprint(),
    }


def test_peak_tables_rebuild_when_upstream_provenance_changes(peak_build):
    _, result, _, provenance, _ = peak_build
    build.build_peak_tables(result)
    first = json.loads(
        (
            provenance.parents[3] / "output/tables/.peak-uncertainty-provenance.json"
        ).read_text()
    )["fingerprint"]
    build.build_peak_tables(result)
    provenance.write_text('{"generation": 2}\n', encoding="utf-8")
    build.build_peak_tables(result)
    second = json.loads(
        (
            provenance.parents[3] / "output/tables/.peak-uncertainty-provenance.json"
        ).read_text()
    )["fingerprint"]
    assert first != second


def test_peak_checkpoint_bytes_survive_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterisation (ADR 0013, Phase 3): a forced recompute rewrites no byte.

    The peak checkpoint becomes a version behind a link in Phase 3, and a reader of
    ``peak_uncertainty/`` must then see the bytes it saw before. The real stage runs
    on the toy landscape and the pairwise inputs of ``test_peak_uncertainty``; only
    the workflow that builds the full network and the fingerprint are replaced. The
    record has no ``created_at`` and no ``git`` key, so nothing is excluded.
    """
    from _manifest import tree_manifest
    from _toy_network import (
        LOW,
        LOW_TWO,
        PEAK,
        SECOND_PEAK,
        VALLEY,
        FakeWorkflow,
    )
    from test_peak_uncertainty import _inputs

    import dubanevics2026topological.analysis.peak_uncertainty as peak_module

    names = {
        "peak": PEAK,
        "valley": VALLEY,
        "second_peak": SECOND_PEAK,
        "low": LOW,
        "low_two": LOW_TWO,
    }
    wt, diff = (
        frame.with_columns(
            pl.col(column).replace(names)
            for column in ("dna_from", "dna_to")
            if column in frame.columns
        )
        for frame in _inputs()
    )
    wt.write_parquet(tmp_path / "wt.parquet")
    diff.write_parquet(tmp_path / "diff.parquet")
    output = tmp_path / "output"
    cache_dir = output / "analysis" / "peak_uncertainty"
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
            "clamped": {
                "published_functional_threshold": -1.0,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
            },
        }
    }
    monkeypatch.setattr(peak_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        peak_module,
        "cache_paths",
        lambda context=None: (
            cache_dir,
            output / "analysis" / ".peak-uncertainty-pending",
        ),
    )
    monkeypatch.setattr(
        peak_module, "_fingerprint", lambda _, **kwargs: "toy-peak-fingerprint"
    )
    monkeypatch.setattr(peak_module, "Papkou2023Workflow", FakeWorkflow)
    monkeypatch.setattr(FakeWorkflow, "built", [])

    first = peak_module.load_or_compute_peak_uncertainty(settings=settings)
    assert first["provenance"]["fingerprint"] == "toy-peak-fingerprint"
    assert FakeWorkflow.built == ["full"]
    manifest = tree_manifest(cache_dir)
    assert {"provenance.json", "diagnostics.json"} <= set(manifest)
    assert sum(name.startswith("tables/") for name in manifest) == 9
    record = json.loads((cache_dir / "provenance.json").read_text(encoding="utf-8"))
    assert "created_at" not in record and "git" not in record

    # A forced recompute really recomputes, and republishes the same bytes.
    inode = cache_dir.stat().st_ino
    peak_module.load_or_compute_peak_uncertainty(force=True, settings=settings)
    assert FakeWorkflow.built == ["full", "full"]
    assert tree_manifest(cache_dir) == manifest
    assert cache_dir.stat().st_ino != inode  # a new directory, not the old one

    # A reuse and a plain load leave every byte alone as well.
    peak_module.load_or_compute_peak_uncertainty(settings=settings)
    peak_module.load_cached_peak_uncertainty()
    assert FakeWorkflow.built == ["full", "full"]
    assert tree_manifest(cache_dir) == manifest


def test_the_legacy_pooled_scatter_loader_still_reads_the_pooled_write_root(
    monkeypatch, tmp_path
) -> None:
    """A spec that still declares pooled_scatter loads the FDR product from the pooled-write root."""
    from dubanevics2026topological.figures.registry import _compose_peak_uncertainty

    fdr_root = tmp_path / "arbitrary-science-root"
    monkeypatch.setenv("FITNESSTREE_FDR_POOLED_ROOT", str(fdr_root))
    monkeypatch.setenv("FITNESSTREE_ANALYSIS_VARIANT", "unrelated")
    monkeypatch.setenv("FITNESSTREE_OUTPUT_ROOT", str(tmp_path / "unrelated"))
    monkeypatch.setattr(build, "load_inputs", lambda *args, **kwargs: {})
    vanilla = SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256="a" * 64))
    # Equal product identities are legitimate when authenticated checkpoint bytes match.
    fdr = SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256="a" * 64))
    seen: dict[str, object] = {}

    def load_product(context, *, expected=None, expected_checkpoint_fingerprint=None):
        if expected_checkpoint_fingerprint is None:
            return vanilla
        seen["root"] = context.output_root
        seen["variant"] = context.variant_name
        seen["fingerprint"] = expected_checkpoint_fingerprint
        seen["raw_data_dir"] = context.raw_data_dir
        return fdr

    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs.load_pooled_scatter_product",
        load_product,
    )
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.peak_uncertainty._fingerprint",
        lambda settings, *, context: "expected-fingerprint",
    )

    def payload(context, product=None):
        if product is fdr:
            return {"tables": {"row": "fdr"}, "diagnostics": {}}
        return {"tables": {"row": "vanilla"}, "diagnostics": {}}

    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs._load_pooled_scatter_payload",
        payload,
    )
    peak = SimpleNamespace(
        figure_id="peak_uncertainty",
        inputs={"pooled_scatter", "manuscript_quantities"},
    )
    from dubanevics2026topological.results import RunContext

    context = RunContext(output_root=tmp_path / "nominal", raw_data_dir=tmp_path / "raw", env={})
    loaded = build.load_figure_inputs([peak], {}, context=context)
    assert "fdr_pooled_scatter" in loaded["saved_figure_inputs"]
    assert seen["root"] == fdr_root.resolve()
    assert seen["variant"] == "threshold=reflected_fdr=by_q=0.05"
    assert seen["fingerprint"] == "expected-fingerprint"
    assert seen["raw_data_dir"] == (tmp_path / "raw").resolve()
    assert os.environ["FITNESSTREE_ANALYSIS_VARIANT"] == "unrelated"
    assert os.environ["FITNESSTREE_OUTPUT_ROOT"] == str(tmp_path / "unrelated")
    assert loaded["saved_figure_inputs"]["fdr_peak_checkpoint_fingerprint"] == "expected-fingerprint"
    # The shipped spec asks for the four variant panels, not these two products.
    from dubanevics2026topological.results import MissingProductError

    with pytest.raises(MissingProductError, match="just peak-panels-product"):
        _compose_peak_uncertainty(loaded, {})


@pytest.mark.parametrize("root_source", ["explicit", "active"])
def test_fdr_route_rejects_wrong_root_with_build_instruction(monkeypatch, tmp_path, root_source):
    from dubanevics2026topological.results import MissingProductError, RunContext

    wrong = tmp_path / "variants/fdr_by_q0.05_papkou_original"
    if root_source == "explicit":
        monkeypatch.setenv("FITNESSTREE_FDR_POOLED_ROOT", str(wrong))
    else:
        monkeypatch.delenv("FITNESSTREE_FDR_POOLED_ROOT", raising=False)
        monkeypatch.setenv("FITNESSTREE_OUTPUT_ROOT", str(wrong))
    monkeypatch.setattr(build, "load_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "dubanevics2026topological.analysis.peak_uncertainty._fingerprint",
        lambda settings, *, context: "approved",
    )
    nominal = SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256="a" * 64))

    def load_product(context, *, expected=None, expected_checkpoint_fingerprint=None):
        if expected_checkpoint_fingerprint is None:
            return nominal
        assert context.output_root == wrong.resolve()
        assert expected_checkpoint_fingerprint == "approved"
        raise MissingProductError("checkpoint fingerprint mismatch")

    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs.load_pooled_scatter_product",
        load_product,
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs._load_pooled_scatter_payload",
        lambda *args, **kwargs: {},
    )
    peak = SimpleNamespace(figure_id="peak_uncertainty", inputs={"pooled_scatter"})
    with pytest.raises(MissingProductError, match="just fdr-pooled-scatter-product"):
        build.load_figure_inputs([peak], {}, context=RunContext(output_root=tmp_path / "nominal", env={}))
