"""The git stamp: a pointer in provenance records, never part of an identity (ADR 0013 D3)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from dubanevics2026topological.analysis import git_stamp as git_stamp_module
from dubanevics2026topological.analysis import produce_or_load as produce_or_load_module
from dubanevics2026topological.analysis.git_stamp import GIT_KEY, git_stamp, tag, untag
from dubanevics2026topological.analysis.produce_or_load import (
    PROVENANCE_NAME,
    Stage,
    produce_or_load,
    stage_fingerprint,
)

NONE_STAMP = {"commit": None, "dirty": None}


def _git(root: Path, *arguments: str) -> str:
    """Run git on ``root`` with the helper's cleaned environment.

    A test run from a git hook inherits ``GIT_DIR`` and friends, which would
    send these commands to the hook's repository instead of the temporary one.
    """
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.org",
            "-C",
            str(root),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=git_stamp_module._environment(),
    )
    return completed.stdout.strip()


def _make_repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    (root / "tracked.txt").write_text("v1", encoding="utf-8")
    (root / "input.txt").write_text("data", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "first")
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path / "repo")


def _stage(root: Path) -> Stage:
    return Stage(
        "demo", settings={"alpha": 0.05}, inputs=(root / "input.txt",), root=root
    )


def _produce(directory: Path) -> None:
    (directory / "table.csv").write_text("a\n1\n", encoding="utf-8")


def _forbidden(directory: Path) -> None:
    pytest.fail("a git change must not trigger a rebuild")


# --- the stamp ---------------------------------------------------------------


def test_stamp_reports_the_head_commit_and_a_clean_tree(repo: Path) -> None:
    stamp = git_stamp(repo)
    assert stamp == {"commit": _git(repo, "rev-parse", "HEAD"), "dirty": False}
    assert len(str(stamp["commit"])) == 40


def test_dirty_counts_tracked_changes_only(repo: Path) -> None:
    (repo / "untracked.bin").write_text("x", encoding="utf-8")
    assert git_stamp(repo)["dirty"] is False
    (repo / "tracked.txt").write_text("v2", encoding="utf-8")
    assert git_stamp(repo)["dirty"] is True
    _git(repo, "add", "tracked.txt")
    assert git_stamp(repo)["dirty"] is True  # staged but uncommitted
    _git(repo, "commit", "-q", "-m", "second")
    assert git_stamp(repo)["dirty"] is False


def test_a_subdirectory_of_the_repository_is_stamped(repo: Path) -> None:
    (repo / "sub").mkdir()
    assert git_stamp(repo / "sub") == git_stamp(repo)


def test_non_repository_gives_none_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    # Stop discovery at the export's parent so no enclosing repository is found.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert git_stamp(export) == NONE_STAMP


def test_missing_directory_gives_none_values(tmp_path: Path) -> None:
    assert git_stamp(tmp_path / "absent") == NONE_STAMP


def test_repository_without_a_commit_gives_none_values(tmp_path: Path) -> None:
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q")
    assert git_stamp(root) == NONE_STAMP


def test_missing_git_executable_gives_none_values(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert git_stamp(repo) == NONE_STAMP


def test_a_timeout_gives_none_values_and_the_call_is_bounded(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeouts: list[Any] = []

    def slow(*args: Any, **kwargs: Any) -> None:
        timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(git_stamp_module.subprocess, "run", slow)
    assert git_stamp(repo) == NONE_STAMP
    assert timeouts and all(isinstance(value, int | float) for value in timeouts)


def test_a_failing_status_keeps_the_commit_and_nulls_the_flag(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = git_stamp_module._run

    def run(root: Path, *arguments: str) -> str | None:
        return None if "status" in arguments else real(root, *arguments)

    monkeypatch.setattr(git_stamp_module, "_run", run)
    stamp = git_stamp(repo)
    assert stamp["commit"] == _git(repo, "rev-parse", "HEAD")
    assert stamp["dirty"] is None


def test_the_status_query_asks_for_tracked_files_without_optional_locks(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []
    real = git_stamp_module._run

    def run(root: Path, *arguments: str) -> str | None:
        seen.append(arguments)
        return real(root, *arguments)

    monkeypatch.setattr(git_stamp_module, "_run", run)
    git_stamp(repo)
    assert (
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=no",
    ) in seen


@pytest.mark.parametrize(
    "variable", ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"]
)
def test_repository_variables_in_the_environment_do_not_redirect_the_stamp(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    other = _make_repo(tmp_path / "other")
    (other / "other-only.txt").write_text("y", encoding="utf-8")
    _git(other, "add", ".")
    _git(other, "commit", "-q", "-m", "other")
    (other / "tracked.txt").write_text("dirty", encoding="utf-8")
    expected = git_stamp(repo)
    assert expected["dirty"] is False
    assert git_stamp(other)["commit"] != expected["commit"]

    redirect = {
        "GIT_DIR": other / ".git",
        "GIT_WORK_TREE": other,
        "GIT_INDEX_FILE": other / ".git" / "index",
        "GIT_COMMON_DIR": other / ".git",
    }[variable]
    monkeypatch.setenv(variable, str(redirect))
    assert git_stamp(repo) == expected


def test_variables_that_only_limit_discovery_are_kept(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "sub").mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(repo))
    assert git_stamp(repo / "sub") == NONE_STAMP


def test_the_environment_drops_repository_and_configuration_variables_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropped = [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "GIT_CONFIG_KEY_17",
        "GIT_CONFIG_VALUE_17",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
    ]
    kept = ["GIT_CEILING_DIRECTORIES", "GIT_TERMINAL_PROMPT", "PATH"]
    for name in dropped:
        monkeypatch.setenv(name, "x")
    for name in kept:
        monkeypatch.setenv(name, "y")
    environment = git_stamp_module._environment()
    assert [name for name in dropped if name in environment] == []
    assert [name for name in kept if environment.get(name) != "y"] == []


# git dies on reading a boolean it cannot parse, so each variable below visibly
# changes what a git command does when it reaches the command.
BAD_CONFIGURATION = "[core]\n\tfilemode = maybe\n"


@pytest.mark.parametrize(
    "variable",
    [
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
    ],
)
def test_configuration_variables_in_the_environment_do_not_change_the_stamp(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    required_versions = {
        "GIT_CONFIG_COUNT": (2, 31),
        "GIT_CONFIG_GLOBAL": (2, 32),
        "GIT_CONFIG_SYSTEM": (2, 32),
    }
    required = required_versions.get(variable)
    if required is not None:
        version_output = subprocess.run(
            ["git", "--version"], check=True, capture_output=True, text=True
        ).stdout
        match = re.search(r"git version (\d+)\.(\d+)", version_output)
        assert match is not None, f"could not parse {version_output!r}"
        installed = (int(match.group(1)), int(match.group(2)))
        if installed < required:
            pytest.skip(
                f"{variable} requires Git >= {required[0]}.{required[1]} "
                f"(installed {installed[0]}.{installed[1]})"
            )

    # HPC jobs may themselves run with Git configuration injected. Establish
    # the baseline with those ambient overrides removed so the comparison tests
    # only the variable under test, not the job's own git environment.
    for name in tuple(os.environ):
        if name.startswith("GIT_CONFIG_"):
            monkeypatch.delenv(name, raising=False)
    expected = git_stamp(repo)
    assert expected == {"commit": _git(repo, "rev-parse", "HEAD"), "dirty": False}

    bad = tmp_path / "bad.gitconfig"
    bad.write_text(BAD_CONFIGURATION, encoding="utf-8")
    if variable == "GIT_CONFIG_PARAMETERS":
        monkeypatch.setenv(variable, "'core.filemode=maybe'")
    elif variable == "GIT_CONFIG_COUNT":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.filemode")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "maybe")
    else:
        monkeypatch.setenv(variable, str(bad))

    # Not vacuous: left in the environment, the variable does break git.
    plain = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        check=False,
    )
    assert plain.returncode != 0
    assert git_stamp(repo) == expected


def test_a_held_index_lock_does_not_fail_the_stamp(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    lock = repo / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    assert git_stamp(repo) == {"commit": head, "dirty": False}
    (repo / "tracked.txt").write_text("v2", encoding="utf-8")
    assert git_stamp(repo) == {"commit": head, "dirty": True}
    assert lock.exists()  # the stamp neither took nor removed the lock


def test_production_succeeds_while_the_index_is_locked(repo: Path) -> None:
    (repo / ".git" / "index.lock").write_text("", encoding="utf-8")
    checkpoint = produce_or_load(
        _stage(repo), _produce, checkpoint_dir=repo / "output" / "cp"
    )
    assert checkpoint.provenance[GIT_KEY] == {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": False,
    }


# --- the verb ----------------------------------------------------------------


def test_tag_returns_a_stamped_copy_and_leaves_the_record_alone(repo: Path) -> None:
    record = {"fingerprint": "f", "artefacts": {"x": "h"}}
    tagged = tag(record, repo)
    assert tagged == {**record, GIT_KEY: git_stamp(repo)}
    assert GIT_KEY == "git"
    assert GIT_KEY not in record


def test_tag_replaces_an_old_stamp(repo: Path) -> None:
    old = {"fingerprint": "f", GIT_KEY: {"commit": "a" * 40, "dirty": True}}
    assert tag(old, repo)[GIT_KEY] == git_stamp(repo)
    assert old[GIT_KEY]["commit"] == "a" * 40


def test_untag_drops_only_the_stamp_and_copies() -> None:
    record = {"fingerprint": "f", "artefacts": {"x": "h"}, GIT_KEY: NONE_STAMP}
    plain = untag(record)
    assert plain == {"fingerprint": "f", "artefacts": {"x": "h"}}
    assert GIT_KEY in record
    assert plain is not record
    unstamped = {"fingerprint": "f"}
    assert untag(unstamped) == unstamped
    assert untag(unstamped) is not unstamped


def test_untag_is_the_inverse_of_tag(repo: Path) -> None:
    record = {"fingerprint": "f", "artefacts": {"x": "h"}}
    assert untag(tag(record, repo)) == record
    assert untag(tag(tag(record, repo), repo)) == record


def test_the_helper_does_not_re_export_the_stamp_names() -> None:
    """Stamping is the sibling module's surface, not the helper's."""
    for name in ("tag", "untag", "git_stamp", "GIT_KEY"):
        assert not hasattr(produce_or_load_module, name)


# --- the helper stamps what it writes ----------------------------------------


def test_the_helper_stamps_the_record_it_writes(repo: Path) -> None:
    checkpoint = produce_or_load(
        _stage(repo), _produce, checkpoint_dir=repo / "output" / "cp"
    )
    assert checkpoint.provenance[GIT_KEY] == git_stamp(repo)
    on_disk = json.loads((checkpoint.path / PROVENANCE_NAME).read_text("utf-8"))
    assert on_disk[GIT_KEY] == git_stamp(repo)
    assert on_disk[GIT_KEY]["commit"] == _git(repo, "rev-parse", "HEAD")


def test_the_stamp_is_taken_before_the_producer_runs(repo: Path) -> None:
    before = git_stamp(repo)
    assert before == {"commit": _git(repo, "rev-parse", "HEAD"), "dirty": False}

    def produce(directory: Path) -> None:
        # Not a stage input, so the fingerprint is unaffected, but it moves the
        # commit and leaves the tree dirty by the time the producer returns.
        (repo / "tracked.txt").write_text("v2", encoding="utf-8")
        _git(repo, "commit", "-q", "-am", "during production")
        (repo / "tracked.txt").write_text("v3", encoding="utf-8")
        _produce(directory)

    checkpoint = produce_or_load(
        _stage(repo), produce, checkpoint_dir=repo / "output" / "cp"
    )
    assert git_stamp(repo) != before  # not vacuous: the state did change
    assert checkpoint.provenance[GIT_KEY] == before
    on_disk = json.loads((checkpoint.path / PROVENANCE_NAME).read_text("utf-8"))
    assert on_disk[GIT_KEY] == before


def test_a_stage_without_a_root_is_stamped_from_the_checkpoint_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    work = tmp_path / "work"
    work.mkdir()
    checkpoint = produce_or_load(
        Stage("rootless", settings={"a": 1}), _produce, checkpoint_dir=work / "cp"
    )
    assert checkpoint.provenance[GIT_KEY] == NONE_STAMP


def test_production_succeeds_without_git_and_records_none_values(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    checkpoint = produce_or_load(
        _stage(repo), _produce, checkpoint_dir=repo / "output" / "cp"
    )
    assert checkpoint.provenance[GIT_KEY] == NONE_STAMP


def test_commit_change_and_dirty_flip_keep_the_fingerprint_and_reuse(
    repo: Path,
) -> None:
    stage = _stage(repo)
    fingerprint = stage_fingerprint(stage)
    checkpoint_dir = repo / "output" / "cp"
    first = produce_or_load(stage, _produce, checkpoint_dir=checkpoint_dir)
    assert first.provenance[GIT_KEY]["dirty"] is False
    assert not first.reused

    (repo / "tracked.txt").write_text("v2", encoding="utf-8")  # dirty flip
    assert git_stamp(repo)["dirty"] is True
    assert stage_fingerprint(stage) == fingerprint
    assert produce_or_load(stage, _forbidden, checkpoint_dir=checkpoint_dir).reused

    _git(repo, "commit", "-q", "-am", "second")  # commit change
    (repo / "tracked.txt").write_text("v3", encoding="utf-8")  # dirty again
    assert git_stamp(repo)["commit"] != first.provenance[GIT_KEY]["commit"]
    assert stage_fingerprint(stage) == fingerprint

    again = produce_or_load(stage, _forbidden, checkpoint_dir=checkpoint_dir)
    assert again.reused
    assert again.provenance["fingerprint"] == fingerprint
    # The checkpoint stays immutable: it keeps the stamp of the run that made it.
    assert again.provenance[GIT_KEY] == first.provenance[GIT_KEY]


def test_a_forced_rerun_stamps_the_state_it_ran_in(repo: Path) -> None:
    stage = _stage(repo)
    checkpoint_dir = repo / "output" / "cp"
    first = produce_or_load(stage, _produce, checkpoint_dir=checkpoint_dir)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "second")
    forced = produce_or_load(stage, _produce, checkpoint_dir=checkpoint_dir, force=True)
    assert not forced.reused
    assert forced.provenance["fingerprint"] == first.provenance["fingerprint"]
    assert forced.provenance[GIT_KEY]["commit"] == _git(repo, "rev-parse", "HEAD")
    assert forced.provenance[GIT_KEY]["commit"] != first.provenance[GIT_KEY]["commit"]


def test_a_record_written_before_stamping_still_reuses(repo: Path) -> None:
    stage = _stage(repo)
    checkpoint_dir = repo / "output" / "cp"
    produce_or_load(stage, _produce, checkpoint_dir=checkpoint_dir)
    path = checkpoint_dir / PROVENANCE_NAME
    record = json.loads(path.read_text("utf-8"))
    del record[GIT_KEY]
    path.write_text(json.dumps(record), encoding="utf-8")
    again = produce_or_load(stage, _forbidden, checkpoint_dir=checkpoint_dir)
    assert again.reused
    assert GIT_KEY not in again.provenance


def test_a_stamp_edited_on_disk_does_not_break_reuse(repo: Path) -> None:
    stage = _stage(repo)
    checkpoint_dir = repo / "output" / "cp"
    produce_or_load(stage, _produce, checkpoint_dir=checkpoint_dir)
    path = checkpoint_dir / PROVENANCE_NAME
    record = json.loads(path.read_text("utf-8"))
    record[GIT_KEY] = {"commit": "f" * 40, "dirty": True}
    path.write_text(json.dumps(record), encoding="utf-8")
    assert produce_or_load(stage, _forbidden, checkpoint_dir=checkpoint_dir).reused
