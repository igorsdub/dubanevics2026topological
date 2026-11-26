"""Failure recovery for staged, multi-file analysis checkpoints."""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubanevics2026topological.analysis import checkpoint


def _hold_checkpoint_lock(cache_dir: str, ready) -> None:
    with checkpoint.checkpoint_lock(Path(cache_dir), "test"):
        ready.set()
        time.sleep(30)


def _try_checkpoint_lock(cache_dir: str, results) -> None:
    try:
        with checkpoint.checkpoint_lock(Path(cache_dir), "test"):
            results.put("acquired")
    except RuntimeError:
        results.put("busy")


def _save(path: Path, value: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "state.json").write_text(json.dumps({"value": value}), encoding="utf-8")


def _validate(path: Path) -> dict[str, str]:
    value = json.loads((path / "state.json").read_text(encoding="utf-8"))["value"]
    if value.startswith("bad"):
        raise ValueError("incomplete checkpoint")
    return {"value": value}


def test_checkpoint_lock_fails_fast_and_persists_lock_file(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    with (
        checkpoint.checkpoint_lock(cache, "test"),
        pytest.raises(RuntimeError, match="already in use"),
    ):
        checkpoint._lock_file(tmp_path / ".cache.lock", "test")
    assert (tmp_path / ".cache.lock").is_file()
    with checkpoint.checkpoint_lock(cache, "test"):
        pass


def _mock_windows_lock(monkeypatch, *, lock_error: OSError | None = None):
    calls: list[tuple[int, int]] = []

    def locking(fd: int, mode: int, length: int) -> None:
        assert length == 1
        calls.append((mode, os.lseek(fd, 0, os.SEEK_CUR)))
        if mode == 1 and lock_error is not None:
            raise lock_error

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(
        checkpoint,
        "os",
        SimpleNamespace(name="nt", SEEK_END=os.SEEK_END),
    )
    return calls


def test_windows_lock_adapter_locks_and_unlocks_the_first_byte(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _mock_windows_lock(monkeypatch)
    cache = tmp_path / "cache"

    with checkpoint.checkpoint_lock(cache, "test"):
        pass

    assert calls == [(1, 0), (2, 0)]
    assert (tmp_path / ".cache.lock").read_bytes() == b"\0"


@pytest.mark.parametrize(
    ("lock_error", "expected"),
    [
        (OSError(errno.EACCES, "locked"), RuntimeError),
        (OSError(errno.EIO, "disk error"), OSError),
    ],
)
def test_windows_lock_adapter_distinguishes_contention_from_io_errors(
    tmp_path: Path, monkeypatch, lock_error: OSError, expected: type[Exception]
) -> None:
    _mock_windows_lock(monkeypatch, lock_error=lock_error)

    with pytest.raises(expected):
        checkpoint._lock_file(tmp_path / ".cache.lock", "test")


def test_checkpoint_lock_serializes_processes_and_releases_on_exit(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    cache_dir = tmp_path / "cache"
    ready = context.Event()
    results = context.Queue()
    holder = context.Process(
        target=_hold_checkpoint_lock,
        args=(str(cache_dir), ready),
    )
    holder.start()
    try:
        assert ready.wait(timeout=10)
        contender = context.Process(
            target=_try_checkpoint_lock,
            args=(str(cache_dir), results),
        )
        contender.start()
        contender.join(timeout=10)
        assert contender.exitcode == 0
        assert results.get(timeout=2) == "busy"
    finally:
        # Simulate an abrupt writer crash. The OS must release the lock even
        # though the context manager never gets a chance to unlock it.
        holder.terminate()
        holder.join(timeout=10)
    assert holder.exitcode is not None
    assert (tmp_path / ".cache.lock").exists()
    results = context.Queue()
    successor = context.Process(
        target=_try_checkpoint_lock,
        args=(str(cache_dir), results),
    )
    successor.start()
    successor.join(timeout=10)
    assert successor.exitcode == 0
    assert results.get(timeout=2) == "acquired"


def test_recovery_keeps_valid_current_and_clears_postcommit_state(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    backup = tmp_path / ".cache.previous"
    pending = tmp_path / "pending"
    _save(cache, "new")
    _save(backup, "old")
    pending.write_text("pending", encoding="utf-8")

    result = checkpoint.recover_checkpoint(cache, pending, "test", _validate)

    assert result == {"value": "new"}
    assert _validate(cache) == {"value": "new"}
    assert not backup.exists()
    assert not pending.exists()


def test_recovery_restores_valid_backup_when_current_is_missing(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    backup = tmp_path / ".cache.previous"
    pending = tmp_path / "pending"
    _save(backup, "old")
    pending.write_text("pending", encoding="utf-8")

    result = checkpoint.recover_checkpoint(cache, pending, "test", _validate)

    assert result == {"value": "old"}
    assert _validate(cache) == {"value": "old"}
    assert not backup.exists()
    assert not pending.exists()


def test_recovery_prefers_requested_backup_and_keeps_stale_state_quarantined(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    backup = tmp_path / ".cache.previous"
    pending = tmp_path / "pending"
    cache.mkdir()
    (cache / "state.json").write_text(
        json.dumps({"value": "old", "fingerprint": "stale"}), encoding="utf-8"
    )
    backup.mkdir()
    (backup / "state.json").write_text(
        json.dumps({"value": "wanted", "fingerprint": "expected"}),
        encoding="utf-8",
    )
    pending.write_text("pending", encoding="utf-8")

    def validate(path: Path):
        return json.loads((path / "state.json").read_text(encoding="utf-8"))

    with pytest.warns(RuntimeWarning, match="stale"):
        result = checkpoint.recover_checkpoint(
            cache,
            pending,
            "test",
            validate,
            expected_fingerprint="expected",
            fingerprint_of=lambda item: item["fingerprint"],
        )

    assert result["value"] == "wanted"
    assert result["fingerprint"] == "expected"
    assert not pending.exists()
    assert list(tmp_path.glob(".cache.unusable-*"))


def test_stale_recovery_keeps_pending_until_matching_generation_is_published(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    pending = tmp_path / "pending"
    cache.mkdir()
    (cache / "state.json").write_text(
        json.dumps({"value": "old", "fingerprint": "stale"}), encoding="utf-8"
    )
    pending.write_text("pending", encoding="utf-8")

    def validate(path: Path):
        return json.loads((path / "state.json").read_text(encoding="utf-8"))

    result = checkpoint.recover_checkpoint(
        cache,
        pending,
        "test",
        validate,
        expected_fingerprint="expected",
        fingerprint_of=lambda item: item["fingerprint"],
    )

    assert result["fingerprint"] == "stale"
    assert pending.is_file()


def test_recovery_quarantines_invalid_current_before_restoring_backup(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    backup = tmp_path / ".cache.previous"
    pending = tmp_path / "pending"
    _save(cache, "bad-current")
    _save(backup, "old")
    pending.write_text("pending", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="unusable"):
        result = checkpoint.recover_checkpoint(cache, pending, "test", _validate)

    assert result == {"value": "old"}
    assert _validate(cache) == {"value": "old"}
    assert list(tmp_path.glob(".cache.unusable-*"))
    assert not pending.exists()


def test_first_build_with_pending_marker_is_allowed_to_recompute(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    pending = tmp_path / "pending"
    pending.write_text("pending", encoding="utf-8")

    assert checkpoint.recover_checkpoint(cache, pending, "test", _validate) is None
    assert pending.is_file()


def test_failed_stage_promotion_restores_previous_and_clears_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_replace = os.replace

    def fail_stage(source: str | os.PathLike[str], target: str | os.PathLike[str]):
        if Path(source) == stage and Path(target) == cache:
            raise OSError("injected stage promotion failure")
        return original_replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", fail_stage)
    with pytest.raises(OSError, match="injected stage promotion failure"):
        checkpoint.publish_checkpoint(stage, cache, pending, "test", _validate)

    assert _validate(cache) == {"value": "old"}
    assert not cache.with_name(".cache.previous").exists()
    assert not pending.exists()
    assert stage.is_dir()


def test_failed_current_rotation_keeps_current_and_retries_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_replace = os.replace
    backup = cache.with_name(".cache.previous")

    def fail_rotation(source: str | os.PathLike[str], target: str | os.PathLike[str]):
        if Path(source) == cache and Path(target) == backup:
            raise OSError("injected current rotation failure")
        return original_replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", fail_rotation)
    with pytest.raises(OSError, match="injected current rotation failure"):
        checkpoint.publish_checkpoint(stage, cache, pending, "test", _validate)

    assert _validate(cache) == {"value": "old"}
    assert not backup.exists()
    assert pending.is_file()
    assert checkpoint.recover_checkpoint(cache, pending, "test", _validate) == {
        "value": "old"
    }
    assert not pending.exists()


def test_failed_rollback_preserves_backup_and_pending_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    backup = cache.with_name(".cache.previous")
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_replace = os.replace

    def fail_publication_and_rollback(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ):
        if (Path(source), Path(target)) in {
            (stage, cache),
            (backup, cache),
        }:
            raise OSError("injected rename failure")
        return original_replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", fail_publication_and_rollback)
    with pytest.raises(OSError, match="injected rename failure") as exc_info:
        checkpoint.publish_checkpoint(stage, cache, pending, "test", _validate)

    assert "Rollback also failed" in " ".join(exc_info.value.__notes__)
    assert not cache.exists()
    assert _validate(backup) == {"value": "old"}
    assert pending.is_file()


@pytest.mark.parametrize(
    ("retry_hint", "expected"),
    [
        ("just demo-analysis", "retry `just demo-analysis` for recovery."),
        (None, "retry the command that produced this checkpoint for recovery."),
    ],
)
def test_failed_rollback_note_names_only_a_caller_supplied_retry_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_hint: str | None,
    expected: str,
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    backup = cache.with_name(".cache.previous")
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_replace = os.replace

    def fail_publication_and_rollback(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ):
        if (Path(source), Path(target)) in {(stage, cache), (backup, cache)}:
            raise OSError("injected rename failure")
        return original_replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", fail_publication_and_rollback)
    with pytest.raises(OSError, match="injected rename failure") as exc_info:
        checkpoint.publish_checkpoint(
            stage, cache, pending, "comparison", _validate, retry_hint=retry_hint
        )

    notes = " ".join(exc_info.value.__notes__)
    assert expected in notes
    if retry_hint is None:
        assert "just" not in notes


@pytest.mark.parametrize(
    ("retry_hint", "expected"),
    [
        ("just demo-analysis", "retry `just demo-analysis` for recovery."),
        (None, "retry the command that produced this checkpoint for recovery."),
    ],
)
def test_failed_rollback_after_validation_failure_honours_the_retry_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_hint: str | None,
    expected: str,
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    backup = cache.with_name(".cache.previous")
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_replace = os.replace

    def corrupt_then_refuse_restore(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ):
        if (Path(source), Path(target)) == (backup, cache):
            raise OSError("injected restore failure")
        result = original_replace(source, target)
        if (Path(source), Path(target)) == (stage, cache):
            (cache / "state.json").write_text(
                json.dumps({"value": "bad-new"}), encoding="utf-8"
            )
        return result

    monkeypatch.setattr(checkpoint.os, "replace", corrupt_then_refuse_restore)
    with (
        pytest.warns(RuntimeWarning, match="unusable"),
        pytest.raises(RuntimeError, match="failed validation") as exc_info,
    ):
        checkpoint.publish_checkpoint(
            stage, cache, pending, "comparison", _validate, retry_hint=retry_hint
        )

    notes = " ".join(exc_info.value.__notes__)
    assert "Rollback also failed" in notes
    assert expected in notes
    if retry_hint is None:
        assert "just" not in notes
    assert pending.is_file()


def test_publish_validation_failure_rolls_back_old_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_replace = os.replace

    def corrupt_after_move(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ):
        result = original_replace(source, target)
        if Path(source) == stage and Path(target) == cache:
            (cache / "state.json").write_text(
                json.dumps({"value": "bad-new"}), encoding="utf-8"
            )
        return result

    monkeypatch.setattr(checkpoint.os, "replace", corrupt_after_move)
    with pytest.raises(RuntimeError, match="failed validation"):
        checkpoint.publish_checkpoint(stage, cache, pending, "test", _validate)

    assert _validate(cache) == {"value": "old"}
    assert list(tmp_path.glob(".cache.unusable-*"))
    assert not pending.exists()


def test_backup_cleanup_failure_warns_without_losing_published_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_rmtree = shutil.rmtree

    def fail_backup(path: str | os.PathLike[str], *args, **kwargs):
        if Path(path) == cache.with_name(".cache.previous"):
            raise OSError("injected cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint.shutil, "rmtree", fail_backup)
    with pytest.warns(RuntimeWarning, match="new validated checkpoint"):
        result = checkpoint.publish_checkpoint(stage, cache, pending, "test", _validate)

    assert result == {"value": "new"}
    assert _validate(cache) == {"value": "new"}
    assert _validate(cache.with_name(".cache.previous")) == {"value": "old"}
    assert not pending.exists()


def test_pending_marker_clear_failure_leaves_valid_cache_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_pending(path: Path, *args, **kwargs):
        if path == pending:
            raise OSError("injected marker failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending)
    with pytest.raises(RuntimeError, match="pending marker could not be cleared"):
        checkpoint.publish_checkpoint(stage, cache, pending, "test", _validate)

    assert _validate(cache) == {"value": "new"}
    assert pending.is_file()
    assert cache.with_name(".cache.previous").is_dir()


def test_pending_marker_clear_failure_message_is_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_pending(path: Path, *args, **kwargs):
        if path == pending:
            raise OSError("injected marker failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending)
    with pytest.raises(RuntimeError) as exc_info:
        checkpoint.publish_checkpoint(stage, cache, pending, "test", _validate)

    message = str(exc_info.value)
    assert "analysis" not in message
    assert "Retry the command that produced this checkpoint" in message


@pytest.mark.parametrize(
    ("retry_hint", "expected"),
    [
        ("just demo-analysis", "Retry `just demo-analysis` to finish recovery."),
        (None, "Retry the command that produced this checkpoint to finish recovery."),
    ],
)
def test_pending_marker_clear_failure_honours_the_retry_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_hint: str | None,
    expected: str,
) -> None:
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    pending = tmp_path / "pending"
    _save(cache, "old")
    _save(stage, "new")
    pending.write_text("pending", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_pending(path: Path, *args, **kwargs):
        if path == pending:
            raise OSError("injected marker failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending)
    with pytest.raises(RuntimeError) as exc_info:
        checkpoint.publish_checkpoint(
            stage, cache, pending, "test", _validate, retry_hint=retry_hint
        )

    assert expected in str(exc_info.value)
    if retry_hint is None:
        assert "just" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("retry_hint", "expected"),
    [
        ("just demo-analysis", "Retry `just demo-analysis` to finish recovery."),
        (None, "Retry the command that produced this checkpoint to finish recovery."),
    ],
)
def test_recovery_pending_marker_clear_failure_honours_the_retry_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_hint: str | None,
    expected: str,
) -> None:
    cache = tmp_path / "cache"
    pending = tmp_path / "pending"
    _save(cache, "old")
    pending.write_text("pending", encoding="utf-8")
    original_unlink = Path.unlink

    def fail_pending(path: Path, *args, **kwargs):
        if path == pending:
            raise OSError("injected marker failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending)
    with pytest.raises(RuntimeError) as exc_info:
        checkpoint.recover_checkpoint(
            cache, pending, "test", _validate, retry_hint=retry_hint
        )

    assert expected in str(exc_info.value)
    if retry_hint is None:
        assert "just" not in str(exc_info.value)
