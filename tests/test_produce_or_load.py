"""Contract tests for the shared produce-or-load helper (ADR 0013, D1)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import warnings
from pathlib import Path

import pytest
from _real_stages import build_registry_stages, needs_raw_data

from dubanevics2026topological import stages
from dubanevics2026topological.analysis import checkpoint
from dubanevics2026topological.analysis import produce_or_load as pol
from dubanevics2026topological.analysis.produce_or_load import (
    PROVENANCE_NAME,
    Stage,
    load_checkpoint,
    produce_or_load,
    stage_fingerprint,
)


def _stage(tmp_path: Path, **overrides) -> Stage:
    data = tmp_path / "input.txt"
    code = tmp_path / "code.py"
    data.write_text("data-v1", encoding="utf-8")
    code.write_text("code-v1", encoding="utf-8")
    values = {
        "name": "demo",
        "settings": {"alpha": 0.05, "run": {"n_jobs": 4, "chunk_size": 10, "k": 3}},
        "inputs": (data,),
        "code": (code,),
        "root": tmp_path,
        "execution_keys": ("n_jobs", "chunk_size"),
    }
    values.update(overrides)
    return Stage(**values)


class _Counter:
    def __init__(self) -> None:
        self.calls = 0
        self.directories: list[Path] = []

    def __call__(self, directory: Path) -> None:
        self.calls += 1
        self.directories.append(directory)
        (directory / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (directory / "sub").mkdir()
        (directory / "sub" / "extra.json").write_text("{}", encoding="utf-8")


def _checkpoint_dir(tmp_path: Path) -> Path:
    return tmp_path / "output" / "checkpoints" / "demo"


def test_fingerprint_excludes_execution_settings(tmp_path: Path) -> None:
    base = stage_fingerprint(_stage(tmp_path))
    other = _stage(
        tmp_path,
        settings={"alpha": 0.05, "run": {"n_jobs": 99, "chunk_size": 1, "k": 3}},
    )
    assert stage_fingerprint(other) == base


def test_stage_has_no_project_defaults() -> None:
    stage = Stage("s", {})
    assert tuple(stage.execution_keys) == ()
    assert tuple(stage.libraries) == ()
    assert stage.inputs == () and stage.code == () and stage.root is None


def test_stage_has_no_name_keys_by_default() -> None:
    assert dict(Stage("s", {}).name_keys) == {}


def test_name_keys_is_the_last_field_and_empty_by_default() -> None:
    names = [field.name for field in dataclasses.fields(Stage)]
    assert names[-1] == "name_keys"
    assert dict(Stage("s", {}).name_keys) == {}
    # Each stage gets its own mapping: nothing is shared through the default.
    assert Stage("s", {}).name_keys is not Stage("t", {}).name_keys


@pytest.mark.parametrize("bare", ["landscape", "", b"landscape"])
def test_a_bare_string_is_rejected_for_name_keys(bare) -> None:
    """A ``str`` is not a mapping of alias to path, and would read as characters."""
    with pytest.raises(TypeError, match=r"Stage name_keys must be a mapping"):
        Stage("s", {}, name_keys=bare)


@pytest.mark.parametrize("value", [[("a", "v")], ("a", "v"), {"a"}, 3, None])
def test_name_keys_must_be_a_mapping(value) -> None:
    with pytest.raises(TypeError, match=r"Stage name_keys must be a mapping"):
        Stage("s", {}, name_keys=value)


def test_a_mapping_is_accepted_for_name_keys() -> None:
    name_keys = {"alpha": "run.alpha"}
    assert Stage("s", {}, name_keys=name_keys).name_keys == name_keys


def test_name_keys_do_not_change_fingerprint(tmp_path: Path) -> None:
    """Aliases only label a version, so they are never declared or hashed."""
    bare = _stage(tmp_path)
    labelled = _stage(tmp_path, name_keys={"a": "alpha", "k": "run.k"})
    relabelled = _stage(tmp_path, name_keys={"z": "alpha"})
    assert stage_fingerprint(labelled) == stage_fingerprint(bare)
    assert stage_fingerprint(relabelled) == stage_fingerprint(bare)
    assert pol._declared(labelled) == pol._declared(bare)
    assert "name_keys" not in json.dumps(pol._declared(labelled))
    # The alias paths are not settings: no alias or path appears in the key.
    assert "run.k" not in json.dumps(pol._declared(labelled))


def test_name_keys_never_reach_the_provenance_record(tmp_path: Path) -> None:
    plain = produce_or_load(
        _stage(tmp_path), _Counter(), checkpoint_dir=_checkpoint_dir(tmp_path)
    )
    other = tmp_path / "other"
    labelled = produce_or_load(
        _stage(tmp_path, name_keys={"a": "alpha"}),
        _Counter(),
        checkpoint_dir=other / "checkpoints" / "demo",
    )
    assert labelled.provenance["fingerprint"] == plain.provenance["fingerprint"]
    assert labelled.provenance["declared"] == plain.provenance["declared"]
    assert labelled.provenance["artefacts"] == plain.provenance["artefacts"]
    assert "name_keys" not in (labelled.path / PROVENANCE_NAME).read_text()
    # Nothing is named by the aliases yet: the checkpoint stays where it was asked.
    assert labelled.path == other / "checkpoints" / "demo"


@needs_raw_data
def test_every_registered_stage_has_the_same_fingerprint_without_name_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real stages, built by their factories, with their aliases and without."""
    registered = build_registry_stages(monkeypatch)
    assert set(registered) == set(stages.SAVENAME_KEYS)
    for name, stage in registered.items():
        assert dict(stage.name_keys) == stages.SAVENAME_KEYS[name]
        bare = dataclasses.replace(stage, name_keys={})
        assert stage_fingerprint(bare) == stage_fingerprint(stage), name
        assert pol._declared(bare) == pol._declared(stage), name
        scrambled = dataclasses.replace(
            stage, name_keys={alias: "nothing.here" for alias in stage.name_keys}
        )
        assert stage_fingerprint(scrambled) == stage_fingerprint(stage), name


@pytest.mark.parametrize("field", ["execution_keys", "libraries"])
@pytest.mark.parametrize("bare", ["n_jobs", "numpy", ""])
def test_a_bare_string_is_rejected_for_collection_fields(field: str, bare: str) -> None:
    """A ``str`` would be matched by substring or iterated by character."""
    with pytest.raises(TypeError, match=field):
        Stage("s", {}, **{field: bare})
    with pytest.raises(TypeError, match=field):
        Stage("s", {}, **{field: bare.encode()})


@pytest.mark.parametrize("field", ["execution_keys", "libraries"])
@pytest.mark.parametrize("value", [(), ("a",), ["a", "b"], frozenset({"a"}), {"a": 1}])
def test_real_collections_are_accepted_for_collection_fields(field: str, value) -> None:
    assert getattr(Stage("s", {}, **{field: value}), field) == value


def test_execution_keys_are_caller_supplied_like_drwatson_ignores(
    tmp_path: Path,
) -> None:
    def fingerprint(n_jobs: int, execution_keys) -> str:
        return stage_fingerprint(
            _stage(
                tmp_path,
                settings={"alpha": 0.05, "n_jobs": n_jobs, "workers": 2},
                execution_keys=execution_keys,
            )
        )

    # Nothing is ignored unless the caller says so: n_jobs is a setting.
    assert fingerprint(1, ()) != fingerprint(2, ())
    assert fingerprint(1, ("n_jobs",)) == fingerprint(2, ("n_jobs",))
    # Only the named keys are ignored, at any depth.
    assert fingerprint(1, ("workers",)) != fingerprint(2, ("workers",))
    nested = {"alpha": 0.05, "run": {"workers": 1}}
    other = {"alpha": 0.05, "run": {"workers": 8}}
    assert stage_fingerprint(
        _stage(tmp_path, settings=nested, execution_keys=["workers"])
    ) == stage_fingerprint(
        _stage(tmp_path, settings=other, execution_keys=frozenset({"workers"}))
    )


def test_fingerprint_ignores_dict_order(tmp_path: Path) -> None:
    a = _stage(tmp_path, settings={"x": 1, "y": {"a": 1, "b": 2}})
    b = _stage(tmp_path, settings={"y": {"b": 2, "a": 1}, "x": 1})
    assert stage_fingerprint(a) == stage_fingerprint(b)


def test_fingerprint_tracks_science_and_inputs_not_code(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    base = stage_fingerprint(stage)
    assert stage_fingerprint(_stage(tmp_path, settings={"alpha": 0.01})) != base
    stage.inputs[0].write_text("data-v2", encoding="utf-8")
    changed_input = stage_fingerprint(stage)
    assert changed_input != base
    # A docstring, or any other code edit, is a label. It is not the key.
    stage.code[0].write_text('"""docstring only"""\ncode-v2\n', encoding="utf-8")
    assert stage_fingerprint(stage) == changed_input


def test_fingerprint_is_location_independent_under_root(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    for root in (first, second):
        root.mkdir()
    fingerprints = []
    for root in (first, second):
        (root / "in.txt").write_text("same", encoding="utf-8")
        fingerprints.append(
            stage_fingerprint(
                Stage("s", {"k": 1}, inputs=(root / "in.txt",), code=(), root=root)
            )
        )
    assert fingerprints[0] == fingerprints[1]


def test_missing_input_is_an_error(tmp_path: Path) -> None:
    stage = _stage(tmp_path, inputs=(tmp_path / "absent.txt",))
    with pytest.raises(FileNotFoundError):
        stage_fingerprint(stage)


def test_produces_then_reuses(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    first = produce_or_load(stage, produce, checkpoint_dir=target)
    second = produce_or_load(stage, produce, checkpoint_dir=target)
    assert produce.calls == 1
    assert (first.reused, second.reused) == (False, True)
    assert second.path == target
    assert (target / "table.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert second.provenance == first.provenance


def test_produce_gets_a_fresh_directory_not_the_checkpoint(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    assert produce.directories[0] != target
    assert not produce.directories[0].exists()


def test_provenance_records_fingerprint_hashes_time_and_versions(
    tmp_path: Path,
) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    result = produce_or_load(stage, _Counter(), checkpoint_dir=target)
    record = json.loads((target / PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert record == result.provenance
    assert record["fingerprint"] == stage_fingerprint(stage)
    expected = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert record["artefacts"]["table.csv"] == expected
    assert set(record["artefacts"]) == {"table.csv", "sub/extra.json"}
    assert record["created_at"]
    assert "python" in record["versions"]
    assert "alpha" in json.dumps(record["declared"])
    assert "n_jobs" not in json.dumps(record["declared"])


def test_versions_default_to_python_only_and_follow_the_libraries_argument(
    tmp_path: Path,
) -> None:
    plain = produce_or_load(
        _stage(tmp_path), _Counter(), checkpoint_dir=_checkpoint_dir(tmp_path)
    )
    assert list(plain.provenance["versions"]) == ["python"]

    stage = _stage(tmp_path, libraries=("pytest", "no-such-distribution-xyz"))
    named = produce_or_load(
        stage,
        _Counter(),
        checkpoint_dir=_checkpoint_dir(tmp_path),
        force=True,
    )
    versions = named.provenance["versions"]
    assert versions["pytest"] == importlib.metadata.version("pytest")
    # An absent distribution is skipped, not an error.
    assert set(versions) == {"python", "pytest"}


def test_libraries_do_not_enter_the_fingerprint(tmp_path: Path) -> None:
    assert stage_fingerprint(_stage(tmp_path)) == stage_fingerprint(
        _stage(tmp_path, libraries=("pytest",))
    )


def test_changed_setting_publishes_new_checkpoint(tmp_path: Path) -> None:
    produce, target = _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(_stage(tmp_path), produce, checkpoint_dir=target)
    changed = _stage(tmp_path, settings={"alpha": 0.01})
    result = produce_or_load(changed, produce, checkpoint_dir=target)
    assert produce.calls == 2 and not result.reused
    assert result.provenance["fingerprint"] == stage_fingerprint(changed)
    assert not list(target.parent.glob(".demo.previous"))
    assert not (target.parent / ".demo.pending").exists()


def test_force_reproduces(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    result = produce_or_load(stage, produce, checkpoint_dir=target, force=True)
    assert produce.calls == 2 and not result.reused


def test_failed_produce_keeps_existing_checkpoint_and_leaves_no_debris(
    tmp_path: Path,
) -> None:
    target = _checkpoint_dir(tmp_path)
    produce_or_load(_stage(tmp_path), _Counter(), checkpoint_dir=target)
    before = (target / PROVENANCE_NAME).read_bytes()

    def boom(directory: Path) -> None:
        (directory / "partial.csv").write_text("x", encoding="utf-8")
        raise ValueError("boom")

    changed = _stage(tmp_path, settings={"alpha": 0.01})
    with pytest.raises(ValueError, match="boom"):
        produce_or_load(changed, boom, checkpoint_dir=target, force=True)
    assert (target / PROVENANCE_NAME).read_bytes() == before
    assert sorted(path.name for path in target.parent.iterdir() if path != target) == [
        ".demo.lock"
    ]


def test_tampered_artefact_is_quarantined_and_rebuilt(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    (target / "table.csv").write_text("tampered", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="unusable"):
        result = produce_or_load(stage, produce, checkpoint_dir=target)
    assert produce.calls == 2 and not result.reused
    assert (target / "table.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_missing_artefact_is_rebuilt(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    (target / "sub" / "extra.json").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        produce_or_load(stage, produce, checkpoint_dir=target)
    assert produce.calls == 2


def test_interrupted_publication_restores_matching_backup(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    backup = target.with_name(".demo.previous")
    os.replace(target, backup)  # crash between the two renames
    (target.parent / ".demo.pending").write_text("", encoding="utf-8")
    result = produce_or_load(stage, produce, checkpoint_dir=target)
    assert produce.calls == 1 and result.reused
    assert not backup.exists()
    assert not (target.parent / ".demo.pending").exists()


def test_concurrent_use_is_refused(tmp_path: Path) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    target.parent.mkdir(parents=True)
    with (
        checkpoint.checkpoint_lock(target, "demo"),
        pytest.raises(RuntimeError, match="already in use"),
    ):
        produce_or_load(stage, _Counter(), checkpoint_dir=target)


def test_reserved_provenance_name_is_rejected(tmp_path: Path) -> None:
    def writes_reserved(directory: Path) -> None:
        (directory / PROVENANCE_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=PROVENANCE_NAME):
        produce_or_load(
            _stage(tmp_path), writes_reserved, checkpoint_dir=_checkpoint_dir(tmp_path)
        )


def test_produce_must_write_something(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no artefacts"):
        produce_or_load(
            _stage(tmp_path),
            lambda directory: None,
            checkpoint_dir=_checkpoint_dir(tmp_path),
        )


def test_export_copies_bytes_to_unchanged_paths(tmp_path: Path) -> None:
    target = _checkpoint_dir(tmp_path)
    result = produce_or_load(_stage(tmp_path), _Counter(), checkpoint_dir=target)
    destination = tmp_path / "output" / "tables"
    destination.mkdir(parents=True)
    (destination / "other.csv").write_text("keep", encoding="utf-8")
    written = result.export_to(destination)
    assert sorted(written) == ["sub/extra.json", "table.csv"]
    assert (destination / "table.csv").read_bytes() == (
        target / "table.csv"
    ).read_bytes()
    assert (destination / "other.csv").read_text(encoding="utf-8") == "keep"
    assert PROVENANCE_NAME not in {p.name for p in destination.iterdir()}
    assert not [p for p in destination.rglob(".*.tmp*")]
    # A second export of identical bytes rewrites nothing.
    mtime = (destination / "table.csv").stat().st_mtime_ns
    assert result.export_to(destination) == []
    assert (destination / "table.csv").stat().st_mtime_ns == mtime


def test_export_supports_renaming(tmp_path: Path) -> None:
    result = produce_or_load(
        _stage(tmp_path), _Counter(), checkpoint_dir=_checkpoint_dir(tmp_path)
    )
    destination = tmp_path / "out"
    result.export_to(destination, names={"table.csv": "papkou2023_table.csv"})
    assert (destination / "papkou2023_table.csv").is_file()
    assert not (destination / "table.csv").exists()


def test_out_of_root_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        stage_fingerprint(Stage("s", {"k": 1}, inputs=(outside,), root=root))
    with pytest.raises(ValueError, match="outside"):
        stage_fingerprint(Stage("s", {"k": 1}, code=(outside,), root=root))
    with pytest.raises(ValueError, match="root"):
        stage_fingerprint(Stage("s", {"k": 1}, inputs=(outside,)))


def test_dotdot_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        stage_fingerprint(Stage("s", {}, inputs=(root / ".." / "x.txt",), root=root))


def test_two_declared_paths_with_one_key_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    data = root / "a.txt"
    data.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="same key"):
        stage_fingerprint(Stage("s", {}, inputs=(data, data), root=root))
    spelled_differently = root / "sub" / ".." / "a.txt"
    with pytest.raises(ValueError, match="same key"):
        stage_fingerprint(Stage("s", {}, code=(data, spelled_differently), root=root))


def test_dotdot_after_a_symlink_is_resolved_not_collapsed(tmp_path: Path) -> None:
    """``link/..`` is the link target's parent, not the link's parent."""
    root = tmp_path / "repo"
    (root / "real" / "inner").mkdir(parents=True)
    (root / "real" / "f.txt").write_text("same", encoding="utf-8")
    (root / "link").symlink_to(root / "real" / "inner", target_is_directory=True)
    via_dots = Stage("s", {}, inputs=(root / "link" / ".." / "f.txt",), root=root)
    direct = Stage("s", {}, inputs=(root / "real" / "f.txt",), root=root)
    assert stage_fingerprint(via_dots) == stage_fingerprint(direct)


def test_dotdot_through_a_symlink_cannot_pull_an_outside_file_into_the_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "outside" / "deep").mkdir(parents=True)
    (tmp_path / "outside" / "f.txt").write_text("x", encoding="utf-8")
    (root / "link").symlink_to(tmp_path / "outside" / "deep", target_is_directory=True)
    # Collapsed lexically this would read as ``repo/f.txt``, which is in the root.
    with pytest.raises(ValueError, match="outside"):
        stage_fingerprint(
            Stage("s", {}, inputs=(root / "link" / ".." / "f.txt",), root=root)
        )


def test_fingerprint_is_independent_of_symlinked_checkout(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "in.txt").write_text("same", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    via_real = Stage("s", {"k": 1}, inputs=(real / "in.txt",), root=real)
    via_link = Stage("s", {"k": 1}, inputs=(link / "in.txt",), root=real)
    mixed = Stage("s", {"k": 1}, inputs=(link / "in.txt",), root=link)
    assert (
        stage_fingerprint(via_real)
        == stage_fingerprint(via_link)
        == stage_fingerprint(mixed)
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"p": Path("x")},
        {"s": {1, 2}},
        {"n": __import__("numpy").float64(1.0)},
        {"n": __import__("numpy").int64(1)},
        {"n": float("nan")},
        {"bytes": b"x"},
    ],
)
def test_non_json_settings_are_rejected(tmp_path: Path, bad: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        stage_fingerprint(_stage(tmp_path, settings=bad))


def test_non_string_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        stage_fingerprint(_stage(tmp_path, settings={1: "a"}))
    with pytest.raises(TypeError):
        stage_fingerprint(_stage(tmp_path, settings={"k": {1: "a", "1": "b"}}))


def test_execution_setting_only_change_is_reused(tmp_path: Path) -> None:
    produce, target = _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(_stage(tmp_path), produce, checkpoint_dir=target)
    other = _stage(
        tmp_path,
        settings={"alpha": 0.05, "run": {"n_jobs": 1, "chunk_size": 2, "k": 3}},
    )
    result = produce_or_load(other, produce, checkpoint_dir=target)
    assert produce.calls == 1 and result.reused


def _unusable(target: Path) -> list[Path]:
    return list(target.parent.glob(".demo.unusable-*"))


@pytest.mark.parametrize("damage", ["missing", "invalid", "no_fingerprint", "list"])
def test_bad_provenance_is_quarantined_and_rebuilt(tmp_path: Path, damage: str) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    path = target / PROVENANCE_NAME
    if damage == "missing":
        path.unlink()
    elif damage == "invalid":
        path.write_text("{not json", encoding="utf-8")
    elif damage == "list":
        path.write_text("[]", encoding="utf-8")
    else:
        record = json.loads(path.read_text(encoding="utf-8"))
        del record["fingerprint"]
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="unusable"):
        result = produce_or_load(stage, produce, checkpoint_dir=target)
    assert produce.calls == 2 and not result.reused
    assert len(_unusable(target)) == 1  # kept for inspection, not deleted


def test_first_build_crash_with_pending_marker_recovers(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    target.mkdir(parents=True)
    (target / "table.csv").write_text("half-written", encoding="utf-8")
    (target.parent / ".demo.pending").write_text("", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="unusable"):
        result = produce_or_load(stage, produce, checkpoint_dir=target)
    assert produce.calls == 1 and not result.reused
    assert not (target.parent / ".demo.pending").exists()
    assert len(_unusable(target)) == 1
    assert (_unusable(target)[0] / "table.csv").read_text() == "half-written"


def test_produced_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("s", encoding="utf-8")

    def links(directory: Path) -> None:
        (directory / "real.csv").write_text("a", encoding="utf-8")
        (directory / "link.csv").symlink_to(outside)

    target = _checkpoint_dir(tmp_path)
    with pytest.raises(ValueError, match="(?i)symlink"):
        produce_or_load(_stage(tmp_path), links, checkpoint_dir=target)
    assert not target.exists()


def test_export_rejects_hash_mismatch_without_touching_destination(
    tmp_path: Path,
) -> None:
    target = _checkpoint_dir(tmp_path)
    result = produce_or_load(_stage(tmp_path), _Counter(), checkpoint_dir=target)
    (target / "table.csv").write_text("tampered", encoding="utf-8")
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "table.csv").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        result.export_to(destination)
    assert (destination / "table.csv").read_text(encoding="utf-8") == "old"
    assert not [p for p in destination.rglob("*.tmp")]


@pytest.mark.parametrize(
    "name", ["../escape.csv", "/tmp/absolute.csv", "a/../../b.csv"]
)
def test_export_rejects_names_escaping_destination(tmp_path: Path, name: str) -> None:
    result = produce_or_load(
        _stage(tmp_path), _Counter(), checkpoint_dir=_checkpoint_dir(tmp_path)
    )
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsafe"):
        result.export_to(destination, names={"table.csv": name})
    assert not (tmp_path / "escape.csv").exists()


def test_on_ready_runs_under_the_checkpoint_lock(tmp_path: Path) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    seen: list[tuple[bool, bool]] = []

    def on_ready(ready) -> None:
        try:
            with checkpoint.checkpoint_lock(target, "demo"):
                free = True
        except RuntimeError:
            free = False
        seen.append((ready.reused, free))

    produce_or_load(stage, _Counter(), checkpoint_dir=target, on_ready=on_ready)
    produce_or_load(stage, _Counter(), checkpoint_dir=target, on_ready=on_ready)
    assert seen == [(False, False), (True, False)]
    # The lock is released once produce_or_load returns.
    with checkpoint.checkpoint_lock(target, "demo"):
        pass


def test_failing_on_ready_propagates_and_releases_the_lock(tmp_path: Path) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)

    def on_ready(_ready) -> None:
        raise OSError("export failed")

    with pytest.raises(OSError, match="export failed"):
        produce_or_load(stage, _Counter(), checkpoint_dir=target, on_ready=on_ready)
    with checkpoint.checkpoint_lock(target, "demo"):
        pass


def _snapshot(directory: Path) -> dict[str, tuple[int, int, int]]:
    entries = [directory, *sorted(directory.rglob("*"))]
    return {
        str(path.relative_to(directory)): (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in entries
    }


def test_reuse_writes_nothing_to_the_checkpoint_directory(tmp_path: Path) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    produce_or_load(stage, _Counter(), checkpoint_dir=target)
    before = _snapshot(target)
    again = produce_or_load(stage, _Counter(), checkpoint_dir=target)
    assert again.reused
    assert _snapshot(target) == before


def test_symlink_inside_root_keeps_its_in_repo_key(tmp_path: Path) -> None:
    """A ``data/raw`` symlink to external storage is keyed by its repo path."""
    root = tmp_path / "repo"
    (root / "data" / "raw").mkdir(parents=True)
    link = root / "data" / "raw" / "counts.txt"
    for name in ("bucket_a", "bucket_b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "counts.txt").write_text("same bytes", encoding="utf-8")
    stage = Stage("s", {"k": 1}, inputs=(link,), root=root)

    link.symlink_to(tmp_path / "bucket_a" / "counts.txt")
    first = stage_fingerprint(stage)
    link.unlink()
    link.symlink_to(tmp_path / "bucket_b" / "counts.txt")  # storage moved

    assert stage_fingerprint(stage) == first
    record = produce_or_load(
        stage, _Counter(), checkpoint_dir=root / "output" / "checkpoints" / "s"
    ).provenance
    assert list(record["declared"]["inputs"]) == ["data/raw/counts.txt"]


def test_symlink_inside_root_still_tracks_content(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    store = tmp_path / "store.txt"
    store.write_text("v1", encoding="utf-8")
    (root / "raw.txt").symlink_to(store)
    stage = Stage("s", {}, inputs=(root / "raw.txt",), root=root)
    before = stage_fingerprint(stage)
    store.write_text("v2", encoding="utf-8")
    assert stage_fingerprint(stage) != before


@pytest.mark.parametrize("linked", ["the slot", "the output directory"])
def test_input_behind_a_link_keys_like_a_real_directory(
    tmp_path: Path, linked: str
) -> None:
    """Characterisation (ADR 0013, Phase 3): turning a slot into a link moves no key.

    Phase 3 replaces a slot such as ``output/analysis/clamped_comparison`` by a link
    into ``clamped_comparison.versions/``. The table stages declare a file inside
    that slot as an input, so its key must stay the declared spelling and its content
    must still be hashed through the link, or every such stage would rebuild.
    """
    root = tmp_path / "repo"
    slot = root / "output" / "analysis" / "clamped_comparison"
    slot.mkdir(parents=True)
    record = slot / "provenance.json"
    record.write_text('{"fingerprint": "one"}\n', encoding="utf-8")
    stage = Stage("s", {"k": 1}, inputs=(record,), root=root)
    checkpoint = root / "output" / "checkpoints" / "s"
    key = "output/analysis/clamped_comparison/provenance.json"

    before = stage_fingerprint(stage)
    assert list(pol._declared(stage)["inputs"]) == [key]
    first = produce_or_load(stage, _Counter(), checkpoint_dir=checkpoint)

    if linked == "the slot":
        versions = slot.parent / "clamped_comparison.versions"
        version = versions / "e04f12aa-91d3c0b2"
        versions.mkdir()
        slot.rename(version)
        slot.symlink_to(os.path.relpath(version, slot.parent), target_is_directory=True)
        assert slot.is_symlink() and slot.resolve() == version.resolve()
    else:
        storage = tmp_path / "storage"
        (root / "output").rename(storage)
        (root / "output").symlink_to(storage, target_is_directory=True)
        version = storage / "analysis" / "clamped_comparison"
        assert (root / "output").is_symlink()
    assert record.read_text(encoding="utf-8") == '{"fingerprint": "one"}\n'

    assert list(pol._declared(stage)["inputs"]) == [key]
    assert stage_fingerprint(stage) == before
    counter = _Counter()
    again = produce_or_load(stage, counter, checkpoint_dir=checkpoint)
    assert again.reused and counter.calls == 0  # the same key and the same bytes
    assert again.provenance["fingerprint"] == first.provenance["fingerprint"]

    # The content is still hashed through the link, and a change moves the key.
    (version / "provenance.json").write_text('{"fingerprint": "two"}\n', "utf-8")
    assert stage_fingerprint(stage) != before


def test_nested_provenance_name_is_an_ordinary_artefact(tmp_path: Path) -> None:
    """Only the top-level ``provenance.json`` is reserved; nested ones count."""

    def nested(directory: Path) -> None:
        (directory / "sub").mkdir()
        (directory / "sub" / PROVENANCE_NAME).write_text("{}", encoding="utf-8")
        (directory / "table.csv").write_text("a\n1\n", encoding="utf-8")

    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    first = produce_or_load(stage, nested, checkpoint_dir=target)
    assert f"sub/{PROVENANCE_NAME}" in first.provenance["artefacts"]
    (target / "sub" / PROVENANCE_NAME).write_text('{"x": 1}', encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="unusable"):
        again = produce_or_load(stage, nested, checkpoint_dir=target)
    assert not again.reused


def test_os_metadata_files_do_not_invalidate_a_checkpoint(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    (target / ".DS_Store").write_bytes(b"finder")
    (target / "sub" / ".DS_Store").write_bytes(b"finder")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        again = produce_or_load(stage, produce, checkpoint_dir=target)
    assert again.reused and produce.calls == 1
    assert not _unusable(target)


def test_os_metadata_files_are_not_published_as_artefacts(tmp_path: Path) -> None:
    def writes_finder_file(directory: Path) -> None:
        (directory / "table.csv").write_text("a\n1\n", encoding="utf-8")
        (directory / ".DS_Store").write_bytes(b"finder")

    result = produce_or_load(
        _stage(tmp_path), writes_finder_file, checkpoint_dir=_checkpoint_dir(tmp_path)
    )
    assert list(result.provenance["artefacts"]) == ["table.csv"]


# -- load_checkpoint: the read-only reader -----------------------------------------


def _read_table(ready) -> str:
    return (ready.path / "table.csv").read_text(encoding="utf-8")


def _parent_names(target: Path) -> set[str]:
    return {path.name for path in target.parent.iterdir()}


def test_load_checkpoint_returns_what_read_gets_under_the_lock(tmp_path: Path) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    produced = produce_or_load(stage, _Counter(), checkpoint_dir=target)
    seen: list[tuple[bool, bool, Path]] = []

    def read(ready) -> str:
        try:
            with checkpoint.checkpoint_lock(target, "demo"):
                free = True
        except RuntimeError:
            free = False
        seen.append((ready.reused, free, ready.path))
        return _read_table(ready)

    assert load_checkpoint(stage, read, checkpoint_dir=target) == "a,b\n1,2\n"
    assert seen == [(True, False, target)]
    with checkpoint.checkpoint_lock(target, "demo"):  # released afterwards
        pass
    again = load_checkpoint(
        stage, lambda ready: ready.provenance, checkpoint_dir=target
    )
    assert again["fingerprint"] == produced.provenance["fingerprint"]


def test_load_checkpoint_changes_nothing_on_disk(tmp_path: Path) -> None:
    stage, produce, target = _stage(tmp_path), _Counter(), _checkpoint_dir(tmp_path)
    produce_or_load(stage, produce, checkpoint_dir=target)
    before, names = _snapshot(target), _parent_names(target)
    load_checkpoint(stage, _read_table, checkpoint_dir=target)
    assert _snapshot(target) == before
    assert _parent_names(target) == names
    assert produce.calls == 1


def test_load_checkpoint_without_a_checkpoint_names_the_command_and_creates_nothing(
    tmp_path: Path,
) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="run `just demo`"):
        load_checkpoint(
            stage, _read_table, checkpoint_dir=target, retry_hint="just demo"
        )
    assert not target.parent.exists()
    target.parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="the command that produces"):
        load_checkpoint(stage, _read_table, checkpoint_dir=target)
    assert not target.exists()


def test_load_checkpoint_refuses_a_checkpoint_of_another_fingerprint(
    tmp_path: Path,
) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    produce_or_load(stage, _Counter(), checkpoint_dir=target)
    other = _stage(tmp_path, settings={"alpha": 0.5})
    before, names = _snapshot(target), _parent_names(target)
    with pytest.raises(FileNotFoundError, match="different|current"):
        load_checkpoint(other, _read_table, checkpoint_dir=target)
    assert _snapshot(target) == before and _parent_names(target) == names
    # A code edit is not a different key: the checkpoint still loads.
    (tmp_path / "code.py").write_text("code-v2", encoding="utf-8")
    assert load_checkpoint(stage, _read_table, checkpoint_dir=target) == "a,b\n1,2\n"


@pytest.mark.parametrize("leftover", [".demo.pending", ".demo.previous"])
def test_load_checkpoint_rejects_an_unfinished_publish_without_recovering_it(
    tmp_path: Path, leftover: str
) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    produce_or_load(stage, _Counter(), checkpoint_dir=target)
    path = target.parent / leftover
    if leftover.endswith("pending"):
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir()
        (path / "table.csv").write_text("older", encoding="utf-8")
    before, names = _snapshot(target), _parent_names(target)
    with pytest.raises(RuntimeError, match="mid-publication.*`just demo`"):
        load_checkpoint(
            stage, _read_table, checkpoint_dir=target, retry_hint="just demo"
        )
    # The writer's recovery is not done by a reader: the leftover stays, as does the
    # checkpoint, and nothing is quarantined.
    assert path.exists()
    assert _snapshot(target) == before and _parent_names(target) == names
    assert not _unusable(target)


@pytest.mark.parametrize("damage", ["tampered", "missing", "invalid", "list"])
def test_load_checkpoint_verifies_artefacts_and_never_repairs(
    tmp_path: Path, damage: str
) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    produce_or_load(stage, _Counter(), checkpoint_dir=target)
    if damage == "tampered":
        (target / "table.csv").write_text("changed", encoding="utf-8")
    elif damage == "missing":
        (target / PROVENANCE_NAME).unlink()
    elif damage == "invalid":
        (target / PROVENANCE_NAME).write_text("{not json", encoding="utf-8")
    else:
        (target / PROVENANCE_NAME).write_text("[]", encoding="utf-8")
    before, names = _snapshot(target), _parent_names(target)
    with pytest.raises(RuntimeError, match="unreadable or does not match"):
        load_checkpoint(stage, _read_table, checkpoint_dir=target)
    assert _snapshot(target) == before and _parent_names(target) == names
    assert not _unusable(target)


def test_a_failing_read_propagates_and_releases_the_lock(tmp_path: Path) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    produce_or_load(stage, _Counter(), checkpoint_dir=target)

    def read(_ready) -> None:
        raise OSError("read failed")

    with pytest.raises(OSError, match="read failed"):
        load_checkpoint(stage, read, checkpoint_dir=target)
    with checkpoint.checkpoint_lock(target, "demo"):
        pass


def test_load_checkpoint_is_refused_while_a_writer_holds_the_lock(
    tmp_path: Path,
) -> None:
    stage, target = _stage(tmp_path), _checkpoint_dir(tmp_path)
    produce_or_load(stage, _Counter(), checkpoint_dir=target)
    with (
        checkpoint.checkpoint_lock(target, "demo"),
        pytest.raises(RuntimeError, match="already in use"),
    ):
        load_checkpoint(stage, _read_table, checkpoint_dir=target)


def test_declared_is_computed_once_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inputs are hashed once. The key hashes the declarations except code."""
    calls = []
    real = pol._declared
    monkeypatch.setattr(pol, "_declared", lambda stage: calls.append(1) or real(stage))

    result = produce_or_load(
        _stage(tmp_path), _Counter(), checkpoint_dir=_checkpoint_dir(tmp_path)
    )

    assert len(calls) == 1
    assert result.provenance["fingerprint"] == pol._canonical_digest(
        pol._key_material(result.provenance["declared"])
    )
    assert "code" not in pol._key_material(result.provenance["declared"])


# -- the public view of an unsettled publication ---------------------------------


def _settled() -> dict[str, bool]:
    return {"pending": False, "backup": False}


def test_publication_state_is_settled_for_a_missing_and_a_published_checkpoint(
    tmp_path: Path,
) -> None:
    target = _checkpoint_dir(tmp_path)
    assert pol.publication_state(target) == _settled()
    produce_or_load(_stage(tmp_path), _Counter(), checkpoint_dir=target)
    assert pol.publication_state(target) == _settled()


def test_publication_state_reads_without_creating_or_locking(tmp_path: Path) -> None:
    absent = tmp_path / "nowhere" / "demo"
    assert pol.publication_state(absent) == _settled()
    assert not absent.parent.exists()

    target = _checkpoint_dir(tmp_path)
    target.parent.mkdir(parents=True)
    with checkpoint.checkpoint_lock(target, "demo"):
        assert pol.publication_state(target) == _settled()


def test_publication_state_reports_the_marker_the_helper_leaves_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The names are the ones the helper itself writes, not a copy of them."""
    target = _checkpoint_dir(tmp_path)
    produce_or_load(_stage(tmp_path), _Counter(), checkpoint_dir=target)
    real = os.replace

    def refuse_promotion(source, destination):
        # Only the staged payload replacing the checkpoint fails; the rollback of
        # the previous checkpoint works but leaves the pending marker set.
        if Path(destination) == target and Path(source).name == "payload":
            raise OSError("injected promotion failure")
        return real(source, destination)

    monkeypatch.setattr(os, "replace", refuse_promotion)
    changed = _stage(tmp_path, settings={"alpha": 0.01})
    with pytest.raises(OSError, match="injected"):
        produce_or_load(changed, _Counter(), checkpoint_dir=target)

    assert pol.publication_state(target) == {"pending": True, "backup": False}
    assert (target.parent / ".demo.pending").exists()


def test_publication_state_reports_a_backup_that_could_not_be_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _checkpoint_dir(tmp_path)
    produce_or_load(_stage(tmp_path), _Counter(), checkpoint_dir=target)
    real = checkpoint.shutil.rmtree

    def keep_backup(path, *args, **kwargs):
        if Path(path) == target.with_name(".demo.previous"):
            raise OSError("injected cleanup failure")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint.shutil, "rmtree", keep_backup)
    changed = _stage(tmp_path, settings={"alpha": 0.01})
    with pytest.warns(RuntimeWarning, match="previous"):
        produce_or_load(changed, _Counter(), checkpoint_dir=target)

    assert pol.publication_state(target) == {"pending": False, "backup": True}
    assert target.with_name(".demo.previous").is_dir()


def test_a_docstring_edit_does_not_change_the_key(tmp_path: Path) -> None:
    code = tmp_path / "code.py"
    code.write_text("def answer():\n    return 1\n", encoding="utf-8")
    stage = _stage(tmp_path, code=(code,))
    before = stage_fingerprint(stage)
    code.write_text('"""Edited docstring."""\ndef answer():\n    return 1\n', encoding="utf-8")
    assert stage_fingerprint(stage) == before
    assert pol._declared(stage)["code"]


def test_a_new_parameter_keeps_the_previous_folder(tmp_path: Path) -> None:
    step = tmp_path / "checkpoints" / "demo"
    first_stage = _stage(tmp_path, name_keys={"alpha": "alpha"})
    second_stage = _stage(
        tmp_path, settings={"alpha": 0.01}, name_keys={"alpha": "alpha"}
    )
    first = produce_or_load(
        first_stage,
        _Counter(),
        checkpoint_dir=pol.result_directory(step, first_stage),
    )
    second = produce_or_load(
        second_stage,
        _Counter(),
        checkpoint_dir=pol.result_directory(step, second_stage),
    )
    assert first.path != second.path
    assert first.path.is_dir() and (first.path / "table.csv").is_file()
    assert second.path.is_dir() and (second.path / "table.csv").is_file()
    assert "alpha=0.05" in first.path.name
    assert "alpha=0.01" in second.path.name
    assert "_key=" in first.path.name and "_key=" in second.path.name


def test_forcing_an_identical_result_does_not_move_downstream(tmp_path: Path) -> None:
    """A forced step that writes the same bytes leaves the downstream key alone."""
    upstream_dir = tmp_path / "up"
    downstream_dir = tmp_path / "down"
    stage = _stage(tmp_path)
    calls = {"down": 0}

    def downstream_produce(directory: Path) -> None:
        calls["down"] += 1
        (directory / "out.txt").write_text("down", encoding="utf-8")

    first = produce_or_load(stage, _Counter(), checkpoint_dir=upstream_dir)
    content = pol._canonical_digest(first.provenance["artefacts"])
    downstream = Stage(
        "downstream",
        settings={"upstream": content},
        root=tmp_path,
    )
    produce_or_load(downstream, downstream_produce, checkpoint_dir=downstream_dir)
    assert calls["down"] == 1
    forced = produce_or_load(stage, _Counter(), checkpoint_dir=upstream_dir, force=True)
    assert forced.provenance["artefacts"] == first.provenance["artefacts"]
    assert "replaces" in forced.provenance
    again = Stage(
        "downstream",
        settings={"upstream": pol._canonical_digest(forced.provenance["artefacts"])},
        root=tmp_path,
    )
    produce_or_load(again, downstream_produce, checkpoint_dir=downstream_dir)
    assert calls["down"] == 1
