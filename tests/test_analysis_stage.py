"""The main analysis stage on ``produce_or_load`` (ADR 0013, Phase 2).

These tests stand in a fake Topographer workflow and use a real temporary
``config/analysis.toml`` with the real source hashing, so a fingerprint that
hashed the whole configuration file (or its scheduling keys) would be caught.
"""

from __future__ import annotations

from dubanevics2026topological.rawdata.value_digest import value_digest

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from _analysis_record import write_analysis_record

from dubanevics2026topological import analysis, build
from dubanevics2026topological import stages as project_stages
from dubanevics2026topological.analysis import clamped_cache as cache
from dubanevics2026topological.analysis import peak_uncertainty as peak_analysis
from dubanevics2026topological.analysis import produce_or_load as pol
from dubanevics2026topological.analysis.checkpoint import checkpoint_lock

CONFIG = """\
# Scientific settings consumed by the publication analysis build.
[dataset]
name = "Papkou2023"
landscape = "{landscape}"
fitness_wt = "data/raw/wt.parquet"
fitness_diff = "data/raw/diff.parquet"

[tree]
kind = "split"

[persistence]
kind = "extended_h0"

[clamped]
graphml = "data/raw/graph.graphml"
peaks = "data/raw/peaks.csv"
published_functional_threshold = {published_functional_threshold}
candidate_functional_threshold = {{ rule = "median_reflected", decimals = {candidate_decimals} }}
reference_topographer_commit = "abc123"
graphml_value_sha256 = "{graphml_value_sha256}"
peaks_value_sha256 = "{peaks_value_sha256}"

[codon_entropy]
sweep_points = {sweep_points}

[bootstrap_smoothed_tree]
seed = 1
n_jobs = {n_jobs}
chunk_size = {chunk_size}
"""


class _Workflow:
    """Stands in for Topographer's workflow: publishes a ``data`` generation symlink."""

    runs = 0
    fail = False

    def __init__(self, landscape, fitness_wt=None, fitness_diff=None) -> None:
        self.landscape = landscape

    @property
    def config(self) -> dict[str, Any]:
        """As Topographer's: a threshold landscape also names its threshold."""
        config: dict[str, Any] = {"landscape": self.landscape}
        if self.landscape == "threshold":
            config["threshold"] = 0.25
        return config

    def produce_or_load_analysis(self, stage: Path, *, tree_kind: str, force: bool):
        type(self).runs += 1
        assert tree_kind == "split" and force
        generation = stage / ".data-generations" / "generation-abc"
        generation.mkdir(parents=True)
        (generation / "metadata.toml").write_text("schema = 1\n", encoding="utf-8")
        (generation / "split-node-table.parquet").write_bytes(b"PAR1-nodes")
        if type(self).fail:
            raise RuntimeError("workflow failed")
        (stage / "data").symlink_to(Path(".data-generations") / generation.name)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Repo:
    """A throwaway repository root holding real TOML and input files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.directory = root / "output" / "analysis"
        for name in ("pyproject.toml", "uv.lock"):
            (root / name).write_text(f"{name} v1\n", encoding="utf-8")
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        for name in ("wt.parquet", "diff.parquet", "graph.graphml", "peaks.csv"):
            (raw / name).write_bytes(f"{name}-v1".encode())
        (root / "config").mkdir()
        (root / "config" / "figures.toml").write_text("[figures]\n", encoding="utf-8")
        self.write_config()

    def write_config(
        self,
        *,
        n_jobs: int = 8,
        chunk_size: int = 250,
        sweep_points: int = 100,
        published_functional_threshold: float = -0.5,
        candidate_decimals: int = 6,
        extra: str = "",
        landscape: str = "full",
    ) -> None:
        raw = self.root / "data" / "raw"
        text = CONFIG.format(
            landscape=landscape,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            sweep_points=sweep_points,
            published_functional_threshold=published_functional_threshold,
            candidate_decimals=candidate_decimals,
            graphml_value_sha256=value_digest(raw / "graph.graphml"),
            peaks_value_sha256=value_digest(raw / "peaks.csv"),
        )
        (self.root / "config" / "analysis.toml").write_text(
            text + extra, encoding="utf-8"
        )

    def change_input(self) -> None:
        (self.root / "data" / "raw" / "wt.parquet").write_bytes(b"wt-v2")
        self.write_config()  # keep the reference digests current

    def fingerprint(self) -> str:
        return pol.stage_fingerprint(analysis.analysis_stage())


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Repo:
    repo = _Repo(tmp_path)
    output = tmp_path / "output"
    _Workflow.runs = 0
    _Workflow.fail = False
    source_root = Path(__file__).resolve().parents[1] / "src/dubanevics2026topological"
    package_root = tmp_path / "src/dubanevics2026topological"
    analysis_dir = package_root / "analysis"
    for directory in (
        analysis_dir,
        package_root / "landscape",
        package_root / "threshold",
        package_root / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for source_dir, target_dir, names in (
        (
            source_root / "analysis",
            analysis_dir,
            (
                "__init__.py",
                "component_persistence.py",
                "clamped_cache.py",
                "describe.py",
                "graph_summary.py",
            ),
        ),
        (
            source_root / "landscape",
            package_root / "landscape",
            ("selected_landscape.py", "selection.py", "full_population.py"),
        ),
        (
            source_root / "threshold",
            package_root / "threshold",
            ("functional_threshold.py",),
        ),
        (source_root / "tables", package_root / "tables", ("__init__.py",)),
    ):
        for name in names:
            shutil.copyfile(source_dir / name, target_dir / name)
    monkeypatch.setattr(analysis, "__file__", str(analysis_dir / "__init__.py"))
    monkeypatch.setattr(cache, "__file__", str(analysis_dir / "clamped_cache.py"))
    for module in (analysis, cache, peak_analysis):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for module in (cache, peak_analysis):
        monkeypatch.setattr(
            module,
            "candidate_selection_fingerprint",
            lambda _, **kwargs: {
                "rounded_threshold": -0.25,
                "source": {"rule": "median_reflected"},
            },
        )
    monkeypatch.setattr(analysis, "OUTPUT", output)
    monkeypatch.setattr(analysis, "ANALYSIS_DIR", repo.directory)
    monkeypatch.setattr(analysis, "DATA_DIR", repo.directory / "data")
    monkeypatch.setattr(analysis, "PROVENANCE_PATH", repo.directory / "provenance.json")
    monkeypatch.setattr(analysis, "CHECKPOINT_DIR", output / "checkpoints" / "analysis")
    monkeypatch.setattr(analysis, "Papkou2023Workflow", _Workflow)
    monkeypatch.setattr(analysis.pl, "read_parquet", lambda path: str(path))
    monkeypatch.setattr(
        analysis,
        "load_network_analysis",
        lambda path: sorted(p.name for p in Path(path).iterdir()),
    )
    monkeypatch.setattr(build, "PROVENANCE_PATH", analysis.PROVENANCE_PATH)
    monkeypatch.setattr(build, "CHECKPOINT_DIR", analysis.CHECKPOINT_DIR)
    monkeypatch.setattr(
        cache, "analysis_provenance_path", lambda context=None: analysis.PROVENANCE_PATH
    )
    return repo


def _run(**kwargs: Any):
    return analysis.load_or_compute_analysis(**kwargs)


def _analysis_leaf() -> Path:
    return pol.result_directory(analysis.CHECKPOINT_DIR, analysis.analysis_stage())


def _pending() -> Path:
    leaf = _analysis_leaf()
    return leaf.parent / f".{leaf.name}.pending"


def _checkpoint_record() -> dict[str, Any]:
    return json.loads(
        (_analysis_leaf() / "provenance.json").read_text(encoding="utf-8")
    )


def test_first_run_publishes_checkpoint_and_links_data(stage) -> None:
    result = _run()

    assert result == ["metadata.toml", "split-node-table.parquet"]
    link = analysis.DATA_DIR
    assert link.is_symlink()
    assert (link / "split-node-table.parquet").read_bytes() == b"PAR1-nodes"
    assert link.resolve().is_relative_to(analysis.CHECKPOINT_DIR.resolve())
    assert not (stage.directory / ".data-generations").exists()


def test_record_keeps_caller_keys_and_matches_checkpoint(stage) -> None:
    _run()

    record = json.loads(analysis.PROVENANCE_PATH.read_text(encoding="utf-8"))
    checkpoint = _checkpoint_record()
    assert record["fingerprint"] == checkpoint["fingerprint"]
    assert record["created_at"] == checkpoint["created_at"]
    assert record["tree_kind"] == "split"
    assert record["workflow"] == {"landscape": "full"}
    assert set(record["artefacts"]) == {
        "data/metadata.toml",
        "data/split-node-table.parquet",
    }


def test_record_names_the_workflow_as_topographer_reports_it(stage) -> None:
    """``workflow`` is the workflow's own ``config``, not a copy of one key of it."""
    stage.write_config(landscape="threshold")

    _run()

    record = json.loads(analysis.PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert record["workflow"] == {"landscape": "threshold", "threshold": 0.25}


def test_second_run_reuses_without_recompute_or_rewrite(stage) -> None:
    _run()
    before = analysis.PROVENANCE_PATH.read_bytes()

    _run()

    assert _Workflow.runs == 1
    assert analysis.PROVENANCE_PATH.read_bytes() == before


def test_changed_input_recomputes_and_keeps_data_readable(stage) -> None:
    _run()
    stage.change_input()

    _run()

    assert _Workflow.runs == 2
    assert (analysis.DATA_DIR / "metadata.toml").is_file()


def test_force_recomputes(stage) -> None:
    _run()
    _run(force=True)

    assert _Workflow.runs == 2


def test_failed_workflow_leaves_prior_state_untouched(stage) -> None:
    _run()
    record = analysis.PROVENANCE_PATH.read_bytes()
    stage.change_input()
    _Workflow.fail = True

    with pytest.raises(RuntimeError, match="workflow failed"):
        _run()

    assert analysis.PROVENANCE_PATH.read_bytes() == record
    assert (analysis.DATA_DIR / "metadata.toml").is_file()
    assert not list(analysis.CHECKPOINT_DIR.glob(".*.stage-*"))
    assert not _pending().exists()


def test_legacy_generation_symlink_is_replaced_and_generation_kept(stage) -> None:
    legacy = stage.directory / ".data-generations" / "generation-old"
    legacy.mkdir(parents=True)
    (legacy / "metadata.toml").write_text("old", encoding="utf-8")
    analysis.DATA_DIR.symlink_to(Path(".data-generations") / "generation-old")
    analysis.PROVENANCE_PATH.write_text(
        json.dumps({"fingerprint": {"files": {}}}), encoding="utf-8"
    )
    # The legacy layout has no published checkpoint, so cached readers refuse it
    # until `just analysis` migrates it.
    with pytest.raises(FileNotFoundError):
        analysis.load_cached_analysis()

    _run()

    assert _Workflow.runs == 1
    assert (legacy / "metadata.toml").read_text(encoding="utf-8") == "old"
    assert analysis.DATA_DIR.resolve().is_relative_to(analysis.CHECKPOINT_DIR.resolve())


def test_a_real_data_directory_is_never_replaced(stage) -> None:
    analysis.DATA_DIR.mkdir(parents=True)
    (analysis.DATA_DIR / "metadata.toml").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="real directory"):
        _run()

    assert (analysis.DATA_DIR / "metadata.toml").read_text(encoding="utf-8") == "keep"


def test_missing_sidecar_or_link_is_restored_from_checkpoint(stage) -> None:
    _run()
    analysis.PROVENANCE_PATH.unlink()
    analysis.DATA_DIR.unlink()

    _run()

    assert _Workflow.runs == 1
    assert analysis.PROVENANCE_PATH.is_file()
    assert analysis.DATA_DIR.is_symlink()


def test_load_cached_analysis_requires_a_record(stage) -> None:
    with pytest.raises(FileNotFoundError):
        analysis.load_cached_analysis()
    _run()
    assert analysis.load_cached_analysis() == [
        "metadata.toml",
        "split-node-table.parquet",
    ]


# D1: the fingerprint comes from scientific settings, inputs and code, never from
# scheduling. A real TOML file and the real source hashing mean that a digest of
# the whole of config/analysis.toml would change these fingerprints.


def _analysis(repo: _Repo, settings: dict[str, Any] | None = None) -> str:
    return analysis.scientific_fingerprint(settings)


def _comparison(repo: _Repo, settings: dict[str, Any] | None = None) -> str:
    analysis.PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    analysis.PROVENANCE_PATH.write_text('{"fingerprint": "full"}', encoding="utf-8")
    return cache._fingerprint(settings or analysis.load_settings())


def _peak(repo: _Repo, settings: dict[str, Any] | None = None) -> str:
    return peak_analysis._fingerprint(settings or analysis.load_settings())


FINGERPRINTS = pytest.mark.parametrize(
    "fingerprint, scientific_edit",
    [
        pytest.param(_analysis, {"landscape": "other"}, id="analysis"),
        pytest.param(_comparison, {"candidate_decimals": 3}, id="comparison"),
        pytest.param(_peak, {"published_functional_threshold": -0.4}, id="peak"),
    ],
)


@FINGERPRINTS
def test_scheduling_keys_in_analysis_toml_do_not_change_the_fingerprint(
    stage, fingerprint, scientific_edit
) -> None:
    base = fingerprint(stage)

    stage.write_config(n_jobs=2, chunk_size=50)

    assert fingerprint(stage) == base


@FINGERPRINTS
def test_scientific_config_edit_changes_the_fingerprint(
    stage, fingerprint, scientific_edit
) -> None:
    base = fingerprint(stage)

    stage.write_config(**scientific_edit)

    assert fingerprint(stage) != base


@FINGERPRINTS
def test_config_file_bytes_are_not_hashed(stage, fingerprint, scientific_edit) -> None:
    base = fingerprint(stage)

    stage.write_config(extra="\n# a comment that changes the file digest\n")

    assert fingerprint(stage) == base


@FINGERPRINTS
def test_sections_a_stage_does_not_read_do_not_change_its_fingerprint(
    stage, fingerprint, scientific_edit
) -> None:
    base = fingerprint(stage)

    stage.write_config(sweep_points=101)  # [codon_entropy] belongs to a later stage

    assert fingerprint(stage) == base


@FINGERPRINTS
def test_scheduling_keys_inside_a_declared_section_are_stripped(
    stage, fingerprint, scientific_edit
) -> None:
    """Each stage goes through ``project_stage``, which drops the execution keys."""
    settings = analysis.load_settings()
    base = fingerprint(stage, settings)

    for section in ("dataset", "clamped"):
        settings["analysis"][section]["n_jobs"] = 3
        settings["analysis"][section]["chunk_size"] = 7

    assert fingerprint(stage, settings) == base


def test_analysis_declares_only_the_sections_it_reads(stage) -> None:
    declared = pol._declared(analysis.analysis_stage())

    assert set(declared["settings"]["config"]) == {"dataset", "tree", "persistence"}
    assert declared["settings"]["config"]["dataset"]["landscape"] == "full"
    assert declared["settings"]["tree_kind"] == "split"
    assert not any(
        key.endswith("config/analysis.toml")
        for part in (declared["code"], declared["inputs"])
        for key in part
    )


def test_input_files_are_declared_and_keyed_relative_to_root(stage) -> None:
    declared = pol._declared(analysis.analysis_stage())

    assert set(declared["inputs"]) == {"data/raw/wt.parquet", "data/raw/diff.parquet"}
    base = stage.fingerprint()
    stage.change_input()
    assert stage.fingerprint() != base


def test_code_and_lockfile_edits_change_the_fingerprint(stage) -> None:
    """Code and the lock file are a label. They do not change the key."""
    stage_obj = analysis.analysis_stage()
    base = stage.fingerprint()
    before = pol._label_block(stage_obj, pol._declared(stage_obj))["uv_lock"]

    (stage.root / "uv.lock").write_text("uv.lock v2\n", encoding="utf-8")

    stage_obj = analysis.analysis_stage()
    assert stage.fingerprint() == base
    assert pol._label_block(stage_obj, pol._declared(stage_obj))["uv_lock"] != before
    assert "code" not in pol._key_material(pol._declared(stage_obj))


def test_analysis_stage_is_built_by_the_project_factory(stage) -> None:
    declared = analysis.analysis_stage()

    assert set(declared.execution_keys) == set(project_stages.EXECUTION_KEYS)
    assert tuple(declared.libraries) == tuple(project_stages.STAGE_LIBRARIES)
    assert declared.root == stage.root


def test_unsupported_tree_or_persistence_kind_is_refused(stage) -> None:
    text = (stage.root / "config" / "analysis.toml").read_text(encoding="utf-8")
    (stage.root / "config" / "analysis.toml").write_text(
        text.replace('kind = "split"', 'kind = "other"'), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="tree.kind"):
        analysis.analysis_stage()


def test_missing_input_is_reported_by_name(stage) -> None:
    (stage.root / "data" / "raw" / "diff.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="diff.parquet"):
        analysis.analysis_stage()


def test_scheduling_edit_does_not_recompute_the_analysis(stage) -> None:
    _run()
    before = analysis.PROVENANCE_PATH.read_bytes()

    stage.write_config(n_jobs=1, chunk_size=10)
    _run()

    assert _Workflow.runs == 1
    assert analysis.PROVENANCE_PATH.read_bytes() == before


def test_scientific_edit_recomputes_the_analysis(stage) -> None:
    _run()

    stage.write_config(landscape="other")
    _run()

    assert _Workflow.runs == 2


# The caller's record and the data link are rewritten under the checkpoint lock.
# Cached readers also verify the record against the published checkpoint, because
# the checkpoint's slot is fixed: its bytes change on publication, before the
# record does.


def test_record_and_link_are_written_under_the_checkpoint_lock(
    stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[bool] = []
    real = analysis._point_outputs_at

    def spy(checkpoint, settings) -> None:
        with (
            pytest.raises(RuntimeError, match="already in use"),
            checkpoint_lock(_analysis_leaf(), "probe"),
        ):
            pass
        held.append(checkpoint.reused)
        real(checkpoint, settings)

    monkeypatch.setattr(analysis, "_point_outputs_at", spy)

    _run()  # publishes
    _run()  # reuses

    assert held == [False, True]


def test_interrupted_record_update_is_refused_by_readers_until_repaired(
    stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run()
    stage.change_input()
    real = analysis._point_outputs_at

    def interrupted(checkpoint, settings) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(analysis, "_point_outputs_at", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run()
    monkeypatch.setattr(analysis, "_point_outputs_at", real)

    # New bytes are published but the record still names the old fingerprint.
    with pytest.raises(RuntimeError, match="does not match"):
        analysis.load_cached_analysis()
    with pytest.raises(RuntimeError, match="does not match"):
        build._analysis_fingerprint()

    _run()  # the checkpoint is current, so this only repairs the record

    assert _Workflow.runs == 2
    assert analysis.load_cached_analysis()
    record = json.loads(analysis.PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert record["fingerprint"] == _checkpoint_record()["fingerprint"]
    assert build._analysis_fingerprint()


def test_cached_readers_refuse_while_publication_is_pending(stage) -> None:
    _run()
    _pending().write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="pending"):
        analysis.load_cached_analysis()
    with pytest.raises(RuntimeError, match="pending"):
        build._analysis_fingerprint()


def test_pending_marker_is_the_one_the_helper_uses(
    stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    real = pol.publish_checkpoint

    def spy(staged, cache_dir, pending_path, *args, **kwargs):
        seen.append(pending_path)
        return real(staged, cache_dir, pending_path, *args, **kwargs)

    monkeypatch.setattr(pol, "publish_checkpoint", spy)

    _run()

    assert seen == [_pending()]
    # The readers ask the helper, which names the same marker.
    _pending().write_text("", encoding="utf-8")
    assert pol.publication_state(_analysis_leaf())["pending"]


def test_a_failed_publication_leaves_readers_refused_until_it_is_retried(
    stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run()
    stage.change_input()
    real = os.replace

    def refuse_promotion(source, target):
        # Only the staged payload replacing the checkpoint fails; the rollback
        # of the previous checkpoint still works, but leaves the marker set.
        if Path(target) == _analysis_leaf() and Path(source).name == "payload":
            raise OSError("injected promotion failure")
        return real(source, target)

    monkeypatch.setattr(os, "replace", refuse_promotion)
    with pytest.raises(OSError, match="injected"):
        _run()

    assert _pending().exists()
    with pytest.raises(RuntimeError, match="pending"):
        analysis.load_cached_analysis()


def test_cached_readers_refuse_a_record_that_lags_the_checkpoint(stage) -> None:
    _run()
    record = json.loads(analysis.PROVENANCE_PATH.read_text(encoding="utf-8"))
    record["fingerprint"] = "0" * 64
    analysis.PROVENANCE_PATH.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        analysis.load_cached_analysis()
    with pytest.raises(RuntimeError, match="does not match"):
        build._analysis_fingerprint()


def test_cached_readers_refuse_a_record_whose_artefacts_lag_the_checkpoint(
    stage,
) -> None:
    """The fingerprint agrees, the recorded artefact hashes do not."""
    _run()
    record = json.loads(analysis.PROVENANCE_PATH.read_text(encoding="utf-8"))
    record["artefacts"]["data/metadata.toml"] = "0" * 64
    analysis.PROVENANCE_PATH.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        analysis.load_cached_analysis()
    with pytest.raises(RuntimeError, match="does not match"):
        build._analysis_fingerprint()

    _run()  # the checkpoint is current, so this only repairs the record

    assert analysis.load_cached_analysis()
    assert build._analysis_fingerprint()


def test_cached_readers_refuse_a_record_without_a_published_checkpoint(stage) -> None:
    write_analysis_record(stage.root / "output", "f" * 64, checkpoint=False)

    with pytest.raises(FileNotFoundError, match="just analysis"):
        analysis.load_cached_analysis()
    with pytest.raises(FileNotFoundError, match="just analysis"):
        build._analysis_fingerprint()


def test_build_fingerprint_follows_the_record_identity_not_its_timestamp(
    stage,
) -> None:
    _run()
    first = build._analysis_fingerprint()
    path = analysis.PROVENANCE_PATH
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps({**record, "created_at": "2030-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    assert build._analysis_fingerprint() == first


def test_build_fingerprint_changes_with_the_recorded_artefacts(stage) -> None:
    _run()
    first = build._analysis_fingerprint()
    # A republish that changed an artefact updates the checkpoint and the record
    # together; the readers refuse any other combination.
    for path in (analysis.PROVENANCE_PATH, _analysis_leaf() / "provenance.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["artefacts"]["data/metadata.toml"] = "0" * 64
        path.write_text(json.dumps(record), encoding="utf-8")

    assert build._analysis_fingerprint() != first
