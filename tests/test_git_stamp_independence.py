"""No identity may depend on the git stamp in a provenance record (ADR 0013 D3).

A provenance record carries a ``git`` key (commit and dirty flag) as a pointer
for people. Every place that digests, compares or keys on a whole record, and
every table stage that keys on an upstream record, must give the same answer
whatever that key holds, including when it is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _analysis_record import write_analysis_record
from _fake_figures import install_fake_figures, write_renderer_sources

from dubanevics2026topological import build
from dubanevics2026topological.analysis.git_stamp import untag
from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint
from dubanevics2026topological.figures import registry
from dubanevics2026topological.provenance import upstream_identity, without_git
from dubanevics2026topological.simplification import (
    bootstrap_smoothed_tree as bootstrap_module,
)

STAMPS = (
    {"commit": "a" * 40, "dirty": False},
    {"commit": "b" * 40, "dirty": True},
    {"commit": None, "dirty": None},
)


def _record(**extra: Any) -> dict[str, Any]:
    return {"fingerprint": {"v": 1}, "artifacts": {"x": "h"}, **extra}


def _variants(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The record without a stamp, then with each stamp."""
    return [record, *({**record, "git": stamp} for stamp in STAMPS)]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- without_git -------------------------------------------------------------


def test_without_git_is_the_project_name_for_the_helpers_untag() -> None:
    """One implementation, in the helper; ``provenance.py`` only renames it."""
    assert without_git is untag


def test_without_git_drops_only_the_stamp_and_copies() -> None:
    record = _record(git=STAMPS[0])
    stripped = without_git(record)
    assert stripped == _record()
    assert "git" in record
    assert stripped is not record
    plain = _record()
    assert without_git(plain) == plain
    assert without_git(plain) is not plain


# --- digests of whole records in the build -----------------------------------


def test_analysis_fingerprint_ignores_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The figures key on the record's fingerprint and artefacts, nothing else.

    The caller's record and the published checkpoint's record each carry the
    helper's stamp, and the two stamps need not agree.
    """
    output = tmp_path / "output"
    monkeypatch.setattr(build, "PROVENANCE_PATH", output / "analysis/provenance.json")
    monkeypatch.setattr(build, "CHECKPOINT_DIR", output / "checkpoints/analysis")
    digests = []
    for record_stamp in (None, *STAMPS):
        for checkpoint_stamp in (None, *STAMPS):
            write_analysis_record(
                output,
                "f" * 64,
                created_at="t",
                checkpoint_extra={}
                if checkpoint_stamp is None
                else {"git": checkpoint_stamp},
                **({} if record_stamp is None else {"git": record_stamp}),
            )
            digests.append(build._analysis_fingerprint())
    assert len(set(digests)) == 1
    # Not vacuous: another fingerprint changes the digest.
    write_analysis_record(output, "e" * 64, git=STAMPS[0])
    assert build._analysis_fingerprint() not in digests


def test_comparison_analysis_fingerprint_ignores_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "provenance.json"
    monkeypatch.setattr(build, "_comparison_provenance_path", lambda context=None: path)
    digests = []
    for record in _variants({"schema": 1, "fingerprint": {"a": 1}}):
        path.write_text(json.dumps(record), encoding="utf-8")
        digests.append(build._comparison_analysis_fingerprint())
    assert len(set(digests)) == 1
    assert digests[0] == _sha(
        json.dumps({"schema": 1, "fingerprint": {"a": 1}}, sort_keys=True)
    )


def test_peak_analysis_fingerprint_ignores_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "provenance.json"
    monkeypatch.setattr(build, "_peak_provenance_path", lambda context=None: path)
    base = {"schema": 2, "fingerprint": {"a": 1}}
    digests = []
    for record in _variants(base):
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        digests.append(build._peak_analysis_fingerprint())
    # One digest whether the record is unstamped or stamped, so the figures and the
    # tables built from it do not rebuild when the peak writer starts to stamp.
    assert len(set(digests)) == 1
    assert digests[0] == build._provenance_digest(base)
    # Key order and layout are not content either.
    path.write_text(
        json.dumps({"fingerprint": {"a": 1}, "schema": 2}), encoding="utf-8"
    )
    assert build._peak_analysis_fingerprint() == digests[0]
    # A real change still changes the digest, stamped or not.
    path.write_text(
        json.dumps({**base, "schema": 3, "git": STAMPS[0]}, indent=2), encoding="utf-8"
    )
    assert build._peak_analysis_fingerprint() != digests[0]


def test_provenance_digest_sees_real_changes_but_not_the_stamp() -> None:
    digests = {build._provenance_digest(record) for record in _variants(_record())}
    assert len(digests) == 1
    changed = _record()
    changed["artifacts"] = {"x": "other"}
    assert build._provenance_digest(changed) not in digests
    assert build._provenance_digest(_record()) == _sha(
        json.dumps(_record(), sort_keys=True, default=str)
    )


def test_bootstrap_digests_ignore_the_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_module, "result_settings", lambda value: value)
    settings = {"analysis": {"bootstrap_smoothed_tree": {}}}
    results = [
        build._bootstrap_digests(SimpleNamespace(provenance=record), settings)
        for record in _variants(_record())
    ]
    assert all(result == results[0] for result in results)
    changed = build._bootstrap_digests(
        SimpleNamespace(provenance=_record(artifacts={"x": "other"})), settings
    )
    assert (
        changed["bootstrap_checkpoint_provenance"]
        != (results[0]["bootstrap_checkpoint_provenance"])
    )


# --- upstream identities that table stages key on ----------------------------


def test_upstream_identity_keeps_two_fields_and_so_drops_the_stamp() -> None:
    identities = [upstream_identity(record) for record in _variants(_record())]
    assert all(identity == identities[0] for identity in identities)
    assert set(identities[0]) == {"fingerprint", "artifacts"}


def _capture_stages(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    stages: list[Any] = []

    def fake(stage: Any, *args: Any, **kwargs: Any) -> list[Any]:
        stages.append(stage)
        return []

    monkeypatch.setattr(build, "_publish_stage_tables", fake)
    return stages


@pytest.mark.parametrize(
    ("builder", "make_result"),
    [
        (
            "build_bootstrap_smoothed_tree_tables",
            lambda p: SimpleNamespace(provenance=p),
        ),
        (
            "build_candidate_tree_contrast_tables",
            lambda p: SimpleNamespace(provenance=p),
        ),
        ("build_sequence_growth_rate_tables", lambda p: SimpleNamespace(provenance=p)),
    ],
)
def test_table_stage_fingerprints_ignore_the_upstream_stamp(
    monkeypatch: pytest.MonkeyPatch, builder: str, make_result: Any
) -> None:
    stages = _capture_stages(monkeypatch)
    for record in _variants(_record()):
        getattr(build, builder)(make_result(record))
    prints = [stage_fingerprint(stage) for stage in stages]
    assert len(prints) == 4 and len(set(prints)) == 1
    # Not vacuous: a changed upstream artefact does change the fingerprint.
    getattr(build, builder)(make_result(_record(artifacts={"x": "other"})))
    assert stage_fingerprint(stages[-1]) != prints[0]


# --- analysis modules --------------------------------------------------------


def test_bootstrap_fingerprint_ignores_the_growth_rate_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scientific = bootstrap_module.load_settings()["analysis"]
    scientific = {
        **scientific,
        "dataset": {"fitness_wt": "pyproject.toml", "fitness_diff": "uv.lock"},
    }
    settings = {"analysis": scientific}
    # Hashing every Topographer source is not what is under test.
    monkeypatch.setattr(bootstrap_module, "package_source_hashes", lambda *_: {})
    # Nor is the classification checkpoint: the selection identity is a fixed value.
    monkeypatch.setattr(
        bootstrap_module,
        "candidate_selection_identity",
        lambda *_: {"rule": "median_reflected"},
    )
    prints = [
        bootstrap_module._fingerprint(settings, record)
        for record in _variants(_record())
    ]
    assert all(print_ == prints[0] for print_ in prints)
    assert json.dumps(prints[0]["growth_rates_checkpoint"], sort_keys=True) == (
        json.dumps(_record(), sort_keys=True)
    )
    changed = bootstrap_module._fingerprint(settings, _record(artifacts={"x": "o"}))
    assert changed != prints[0]


def test_bootstrap_scientific_identity_excludes_code_and_environment() -> None:
    fingerprint = {
        "config": {"seed": 7},
        "candidate_functional_threshold": -0.25,
        "environment": {"python": "3.x", "platform": "cluster-a"},
        "run_fields": {"run_id": "run-a"},
        "created_at": "time-a",
        "files": {
            "data/processed/fitness.parquet": "dataset",
            "src/analysis.py": "code-v1",
            "uv.lock": "lock-v1",
        },
        "growth_rates_checkpoint": {
            "fingerprint": {
                "config": {"elapsed_time": 1.0},
                "files": {
                    "data/raw/export.csv": "barseq-input",
                    "src/growth.py": "growth-code-v1",
                    "uv.lock": "growth-lock-v1",
                    "output/analysis/sequence_growth_rates/growth_rates.parquet": (
                        "numeric-output-v1"
                    ),
                },
            },
            "artifacts": {"growth_rates.parquet": "artifact-v1"},
        },
    }
    baseline = bootstrap_module.checkpoint_identity.scientific_fingerprint(fingerprint)
    changed = json.loads(json.dumps(fingerprint))
    changed["environment"] = {"python": "3.y", "platform": "cluster-b"}
    changed["run_fields"] = {"run_id": "run-b"}
    changed["created_at"] = "time-b"
    changed["files"]["src/analysis.py"] = "code-v2"
    changed["files"]["uv.lock"] = "lock-v2"
    growth_files = changed["growth_rates_checkpoint"]["fingerprint"]["files"]
    growth_files["src/growth.py"] = "growth-code-v2"
    growth_files["uv.lock"] = "growth-lock-v2"
    changed["growth_rates_checkpoint"]["artifacts"]["growth_rates.parquet"] = (
        "artifact-v2"
    )
    assert bootstrap_module.checkpoint_identity.scientific_fingerprint(changed) == baseline

    changed_growth_output = json.loads(json.dumps(changed))
    changed_growth_output["growth_rates_checkpoint"]["fingerprint"]["files"][
        "output/analysis/sequence_growth_rates/growth_rates.parquet"
    ] = "numeric-output-v2"
    assert (
        bootstrap_module.checkpoint_identity.scientific_fingerprint(changed_growth_output)
        != baseline
    )


# --- digests inside build_figures --------------------------------------------


class _Chart:
    def save(self, path: str | Path, *, format: str, **kwargs: object) -> None:
        Path(path).write_bytes(f"{format} artifact".encode())


@pytest.fixture
def figure_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run ``build_figures`` for one figure against inputs the test supplies."""
    figure_ids = ("candidate_edge_evidence",)
    write_renderer_sources(tmp_path / "src/dubanevics2026topological/figures")
    settings = {
        "analysis": {},
        "figures": {
            "formats": ["svg"],
            "variants": ["vanilla"],
            "figures": {figure_id: {"width": 1} for figure_id in figure_ids},
        },
    }
    renders: list[str] = []
    fake = SimpleNamespace(
        FIGURE_IDS=figure_ids,
        render=lambda payload, figure_id, config, *, variant="annotated": (
            renders.append(figure_id) or _Chart()
        ),
    )
    inputs: dict[str, Any] = {}
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(build, "load_inputs", lambda names, settings=None: inputs)
    monkeypatch.setattr(registry, "compose_payload", lambda *args: {})
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "output/figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "output/figures/.records")
    install_fake_figures(monkeypatch, fake)

    def run(figure_id: str) -> list[str]:
        renders.clear()
        build.build_figures((figure_id,))
        return list(renders)

    return inputs, run


def test_edge_evidence_figure_ignores_the_stamp_of_its_provenance(figure_build) -> None:
    inputs, run = figure_build
    figure_id = "candidate_edge_evidence"
    for position, record in enumerate(_variants(_record())):
        inputs["edge_evidence"] = {"provenance": record}
        # Figures are not cached, so a stamp-only change still draws the chart.
        assert run(figure_id) == [figure_id]
    inputs["edge_evidence"] = {"provenance": _record(artifacts={"x": "other"})}
    assert run(figure_id) == [figure_id]
