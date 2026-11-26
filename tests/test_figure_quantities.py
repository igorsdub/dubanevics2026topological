"""Figure labels from the manuscript quantity registry (synthetic fixtures only)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_manuscript_quantities import (
    CONFIG,
    _tree,
    fake_readers,
    write_upstream,
)

from dubanevics2026topological import build
from dubanevics2026topological.build import FiguresFailedError
from dubanevics2026topological.analysis import manuscript_quantities as mq
from dubanevics2026topological.figures import _quantities as fq
from dubanevics2026topological.figures import registry

GAP_KEY = "full_graph.component_count"


def _sources(output: Path) -> mq.Sources:
    return mq.load_sources(output, readers=fake_readers(output, _tree()))


def _export(output: Path) -> dict[str, Any]:
    """Write the registry like the stage does; return the rows by key."""
    sources = _sources(output)
    rows = mq.build_rows(sources, CONFIG, git_commit="abc123")
    destination = output / "tables" / mq.TABLE_SUBDIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    mq.write_tables(destination, rows)
    return {row["key"]: row for row in rows}


def _current(output: Path) -> fq.CurrentFingerprints:
    return lambda: {
        up.stage: up.fingerprint for up in _sources(output).values() if up is not None
    }


@pytest.fixture
def output(tmp_path: Path) -> Path:
    directory = tmp_path / "output"
    write_upstream(directory)
    return directory


def _first_key(rows: dict[str, Any], stage: str) -> str:
    return next(
        key
        for key, row in rows.items()
        if row["source_stage"] == stage and row["display_value"] is not None
    )


def test_display_returns_display_value_and_records_requested_keys(
    output: Path,
) -> None:
    rows = _export(output)
    quantities = fq.load_manuscript_quantities(
        output, current_fingerprints=_current(output)
    )
    key = _first_key(rows, mq.CLAMPED)
    assert quantities.display(key) == rows[key]["display_value"]
    quantities.display(key)
    assert quantities.requested_keys == (key,)
    with pytest.raises(fq.QuantityKeyMissing):
        quantities.display("no.such_key")
    assert quantities.requested_keys == (key, "no.such_key")


def test_missing_registry_names_the_path_and_the_stage(tmp_path: Path) -> None:
    with pytest.raises(fq.QuantityRegistryMissing, match="manuscript-quantities") as e:
        fq.load_manuscript_quantities(tmp_path)
    assert str(fq.registry_path(tmp_path)) in str(e.value)


def test_missing_key_names_key_and_path() -> None:
    quantities = fq.from_rows([], path=Path("reg.parquet"))
    with pytest.raises(fq.QuantityKeyMissing, match=r"a\.b.*reg\.parquet"):
        quantities.display("a.b")


def test_gap_row_raises_a_distinct_error_quoting_the_reason(output: Path) -> None:
    rows = _export(output)
    assert rows[GAP_KEY]["gap_reason"]
    quantities = fq.load_manuscript_quantities(output)
    with pytest.raises(fq.QuantityGapError) as e:
        quantities.display(GAP_KEY)
    assert GAP_KEY in str(e.value) and rows[GAP_KEY]["gap_reason"] in str(e.value)
    assert e.value.reason == rows[GAP_KEY]["gap_reason"]
    assert not isinstance(e.value, (fq.QuantityKeyMissing, fq.QuantityStaleError))


def test_stale_fingerprint_raises_naming_key_and_path(output: Path) -> None:
    rows = _export(output)
    key = _first_key(rows, mq.CLAMPED)
    write_upstream(output, clamped_fingerprint="clamped-fp-2")
    quantities = fq.load_manuscript_quantities(
        output, current_fingerprints=_current(output)
    )
    with pytest.raises(fq.QuantityStaleError, match="stale") as e:
        quantities.display(key)
    assert key in str(e.value) and str(quantities.path) in str(e.value)
    # A row from an upstream that did not change still reads.
    other = next(
        key
        for key, row in rows.items()
        if row["display_value"] is not None
        and row["source_fingerprint"] is not None
        and mq.CLAMPED not in row["source_stage"]
    )
    assert quantities.display(other) == rows[other]["display_value"]


def test_derived_rows_are_checked_against_every_joined_source(output: Path) -> None:
    rows = _export(output)
    derived = next(
        key
        for key, row in rows.items()
        if "+" in row["source_stage"] and row["display_value"] is not None
    )
    write_upstream(output, clamped_fingerprint="clamped-fp-2")
    quantities = fq.load_manuscript_quantities(
        output, current_fingerprints=_current(output)
    )
    if mq.CLAMPED in rows[derived]["source_stage"]:
        with pytest.raises(fq.QuantityStaleError, match=derived):
            quantities.display(derived)
    else:
        assert quantities.display(derived) == rows[derived]["display_value"]


def test_in_memory_rows_skip_staleness_unless_a_map_is_given() -> None:
    row = {
        "key": "a.b",
        "display_value": "3",
        "gap_reason": None,
        "source_stage": mq.PEAK,
        "source_fingerprint": "old",
    }
    assert fq.from_rows([row]).display("a.b") == "3"
    stale = fq.from_rows([row], current_fingerprints=lambda: {mq.PEAK: "new"})
    with pytest.raises(fq.QuantityStaleError, match="a.b"):
        stale.display("a.b")
    fresh = fq.from_rows([row], current_fingerprints=lambda: {mq.PEAK: "old"})
    assert fresh.display("a.b") == "3"


def test_a_combined_row_with_a_fingerprintless_part_matches_what_combine_recorded() -> (
    None
):
    parts = (
        mq.Upstream(mq.PEAK, "peak-fp"),
        mq.Upstream(mq.COMPONENT, None),
        mq.Upstream(mq.CONFIG, None),
    )
    joined = mq.combine(*parts)
    assert joined is not None and joined.fingerprint == "peak-fp+none+none"
    row = {
        "key": "a.b",
        "display_value": "3",
        "gap_reason": None,
        "source_stage": joined.stage,
        "source_fingerprint": joined.fingerprint,
    }
    current = {mq.PEAK: "peak-fp", mq.COMPONENT: None}
    assert (
        fq.from_rows([row], current_fingerprints=lambda: current).display("a.b") == "3"
    )
    changed = {mq.PEAK: "peak-2", mq.COMPONENT: None}
    with pytest.raises(fq.QuantityStaleError, match="peak-2\\+none\\+none"):
        fq.from_rows([row], current_fingerprints=lambda: changed).display("a.b")


def test_manuscript_quantities_is_a_requestable_input() -> None:
    assert "manuscript_quantities" in registry.INPUT_NAMES


def test_dependency_record_includes_the_registry_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A figure that reads no label from the registry (the peak figure now does).
    figure_id = "candidate_edge_evidence"
    spec = registry.get_spec(figure_id)
    assert "manuscript_quantities" not in spec.inputs
    figure_dir = tmp_path / "src/dubanevics2026topological/figures"
    figure_dir.mkdir(parents=True)
    for name in (*registry.renderer_files(spec), *build.SHARED_RENDERER_FILES):
        (figure_dir / name).write_text("x", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", output)
    settings = {"figures": {"figures": {figure_id: {}}}}
    plain = build._figure_dependencies(figure_id, "vanilla", settings)
    assert "manuscript_quantities" not in plain

    monkeypatch.setattr(
        build,
        "FIGURE_SPECS",
        (replace(spec, inputs=spec.inputs | {"manuscript_quantities"}),),
    )
    record = output / "tables" / ".manuscript-quantities-provenance.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"fingerprint": "reg-1"}), encoding="utf-8")
    first = build._figure_dependencies(figure_id, "vanilla", settings)
    assert first["manuscript_quantities"] == "reg-1"
    record.write_text(json.dumps({"fingerprint": "reg-2"}), encoding="utf-8")
    assert build._figure_dependencies(figure_id, "vanilla", settings) != first


def _gap_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    default: bool,
    reason: str = mq.stage_not_run_reason(mq.PEAK),
    inputs: frozenset[str] = frozenset(),
) -> list[str]:
    peak = registry.get_spec("peak_uncertainty")
    quantities = fq.from_rows(
        [{"key": "x.y", "display_value": None, "gap_reason": reason}]
    )

    class Chart:
        def save(self, path: Any, *, format: str, **kwargs: Any) -> None:
            Path(path).write_text(format)

    def render(inputs: Any, settings: Any, variant: str) -> Chart:
        inputs["manuscript_quantities"].display("x.y")
        return Chart()

    spec = replace(
        peak,
        inputs=inputs,
        notebook=None,
        render=render,
        default=default,
        variants=frozenset({"vanilla"}),
    )
    settings = {
        "figures": {
            "formats": ["svg"],
            "variants": ["vanilla"],
            "figures": {spec.figure_id: {}},
        }
    }
    monkeypatch.setattr(build, "FIGURE_SPECS", (spec,))
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(
        build,
        "load_inputs",
        lambda names, settings=None: {"manuscript_quantities": quantities},
    )
    monkeypatch.setattr(build, "_figure_dependencies", lambda *args: {"fake": 1})
    monkeypatch.setattr(build, "_analysis_fingerprint", lambda: "a")
    monkeypatch.setattr(build, "_peak_analysis_fingerprint", lambda context=None: "p")
    monkeypatch.setattr(build, "_pooled_scatter_dependency", lambda *args: "g")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "figures/.records")
    return [spec.figure_id]


def test_opt_in_figure_with_a_gap_is_skipped_not_crashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = _gap_build(tmp_path, monkeypatch, default=False)
    assert build.build_figures(tuple(ids)) == []
    assert (
        "awaiting HPC run: x.y: missing: stage peak-uncertainty not run"
        in capsys.readouterr().out
    )


def test_keep_going_reports_an_opt_in_quantity_gap(tmp_path, monkeypatch):
    from types import SimpleNamespace

    ids = _gap_build(tmp_path, monkeypatch, default=False)
    quantities = fq.from_rows([
        {"key": "x.y", "display_value": None, "gap_reason": mq.stage_not_run_reason(mq.PEAK)}
    ])
    monkeypatch.setattr(
        build, "load_figure_inputs",
        lambda *args, **kwargs: {"manuscript_quantities": quantities},
    )
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    context = SimpleNamespace(output_root=tmp_path, variant_name=None)
    assert not build.build_figures_keep_going(ids, context=context)
    failures = json.loads((tmp_path / "figures/figure-failures.json").read_text())["failures"]
    assert len(failures) == 1
    assert failures[0]["figure_id"] == ids[0]
    assert failures[0]["exception_type"] == "QuantityGapError"
    assert "stage peak-uncertainty not run" in failures[0]["traceback"]


def test_a_skipped_figure_removes_its_record_and_keeps_the_tracked_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _gap_build(tmp_path, monkeypatch, default=False)
    stem = ids[0]
    target = tmp_path / "figures" / f"{stem}.svg"
    record = tmp_path / "figures/.records" / f"{stem}.svg.json"
    record.parent.mkdir(parents=True)
    target.write_text("old numbers")
    record.write_text(json.dumps({"fake": 0}))
    assert build.build_figures(tuple(ids)) == []
    # The figure file may be a git-tracked, HPC-rendered PNG: only the record goes.
    assert target.read_text() == "old numbers" and not record.exists()


def test_an_undefined_gap_is_never_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _gap_build(
        tmp_path, monkeypatch, default=False, reason="undefined: no ordinary pair"
    )
    target = tmp_path / "figures" / f"{ids[0]}.svg"
    target.parent.mkdir(parents=True)
    target.write_text("kept")
    with pytest.raises(FiguresFailedError, match="cannot be skipped.*no ordinary pair"):
        build.build_figures(tuple(ids))
    assert target.read_text() == "kept"


def test_a_stale_gap_of_an_opt_in_figure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _gap_build(
        tmp_path, monkeypatch, default=False, reason="stale: output not current"
    )
    with pytest.raises(FiguresFailedError, match="not current"):
        build.build_figures(tuple(ids))


def test_build_figures_runs_the_full_registry_currency_check_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _gap_build(
        tmp_path,
        monkeypatch,
        default=True,
        reason="x",
        inputs=frozenset({"manuscript_quantities"}),
    )
    calls: list[Any] = []

    def refuse(settings: Any) -> None:
        calls.append(settings)
        raise RuntimeError("The manuscript quantity registry is stale")

    monkeypatch.setattr(build, "require_current_registry", refuse)
    with pytest.raises(RuntimeError, match="registry is stale"):
        build.build_figures(tuple(ids))
    assert len(calls) == 1


def test_build_figures_skips_the_registry_check_when_no_figure_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _gap_build(tmp_path, monkeypatch, default=False)
    monkeypatch.setattr(
        build,
        "require_current_registry",
        lambda settings: pytest.fail("no planned figure declares the registry"),
    )
    assert build.build_figures(tuple(ids)) == []


def test_default_figure_with_a_gap_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _gap_build(tmp_path, monkeypatch, default=True)
    with pytest.raises(FiguresFailedError, match="x.y"):
        build.build_figures(tuple(ids))


def test_a_stale_gap_row_is_a_stale_error_and_a_missing_one_a_gap_error() -> None:
    rows = [
        {"key": "a.stale", "display_value": None, "gap_reason": "stale: not current"},
        {"key": "a.missing", "display_value": None, "gap_reason": "missing: absent"},
        {"key": "a.undefined", "display_value": None, "gap_reason": "undefined: empty"},
    ]
    quantities = fq.from_rows(rows, path=Path("reg.parquet"))
    with pytest.raises(fq.QuantityStaleError, match=r"a\.stale.*reg\.parquet"):
        quantities.display("a.stale")
    for key in ("a.missing", "a.undefined"):
        with pytest.raises(fq.QuantityGapError):
            quantities.display(key)


def test_the_helper_reads_the_registry_record_through_the_analysis_module(
    tmp_path: Path,
) -> None:
    with pytest.raises(fq.QuantityRegistryMissing, match="no export record"):
        fq.registry_fingerprint(tmp_path)
    record = mq.registry_record_path(tmp_path)
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"fingerprint": "fp-1"}), encoding="utf-8")
    assert fq.registry_fingerprint(tmp_path) == mq.read_registry_fingerprint(tmp_path)


@pytest.mark.parametrize(
    "reason",
    [
        "missing: no stage computes the connected components of the Full graph",
        "missing: the raw fitness tables behind the band view: fitness_wt.parquet",
        "missing: the clamped comparison result carries no Full analysis",
    ],
)
def test_a_missing_gap_that_is_not_an_unrun_stage_is_never_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    ids = _gap_build(tmp_path, monkeypatch, default=False, reason=reason)
    with pytest.raises(FiguresFailedError, match="cannot be skipped"):
        build.build_figures(tuple(ids))


def test_the_not_run_reason_has_one_builder_and_one_predicate() -> None:
    reason = mq.stage_not_run_reason("peak-uncertainty")
    assert reason == "missing: stage peak-uncertainty not run"
    assert mq.is_stage_not_run(reason)
    assert mq.is_stage_not_run(f"{reason}: peak provenance.json; rerun just x")
    # A joined derived stage names its upstreams, never the config.
    assert mq.stage_not_run_reason("codon-entropy+config/analysis.toml") == (
        "missing: stage codon-entropy not run"
    )
    for other in (
        "stale: stage peak-uncertainty not run",
        "undefined: stage peak-uncertainty not run",
        "missing: no stage computes it",
        "missing: peak-uncertainty output absent",
        "missing: stage peak-uncertainty not run yet, rerun",
    ):
        assert not mq.is_stage_not_run(other), other


def test_a_row_with_no_display_value_and_no_gap_reason_is_malformed() -> None:
    quantities = fq.from_rows(
        [{"key": "a.empty", "display_value": None, "gap_reason": None}],
        path=Path("reg.parquet"),
    )
    with pytest.raises(fq.QuantityError, match=r"a\.empty.*malformed.*reg\.parquet"):
        quantities.display("a.empty")


def test_the_notebook_loader_refuses_a_stale_config_row(
    output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _export(output)
    sources = _sources(output)
    fixed = sources
    monkeypatch.setattr(mq, "load_sources", lambda out, **kw: fixed)
    settings = {"analysis": CONFIG}
    stage, _ = build._manuscript_quantities_stage(settings, sources)
    record = mq.registry_record_path(output)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps({"fingerprint": build.stage_fingerprint(stage)}), encoding="utf-8"
    )
    key = "thresholds.published_functional.value"
    current = fq.load_current_manuscript_quantities(output, settings)
    assert current.display(key)

    changed = {
        "analysis": {
            **CONFIG,
            "clamped": {"published_functional_threshold": -0.4},
        }
    }
    # The per-key check cannot see a config row (no fingerprint), so the plain loader
    # accepts the stale registry; the whole-registry check refuses it.
    assert fq.load_manuscript_quantities(output).display(key)
    with pytest.raises(RuntimeError, match="registry is stale"):
        fq.load_current_manuscript_quantities(output, changed)
