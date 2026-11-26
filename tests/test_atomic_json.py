"""Failure-safety checks for the shared atomic JSON writer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dubanevics2026topological.analysis import _write_json_atomic


def test_atomic_json_writes_formatted_utf8_and_creates_parent_directories(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "cache" / "diagnostics.json"
    value = {"zeta": "café", "alpha": [1, 2]}

    _write_json_atomic(destination, value)

    assert (
        destination.read_bytes()
        == b'{\n  "alpha": [\n    1,\n    2\n  ],\n  "zeta": "caf\\u00e9"\n}\n'
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == value
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "diagnostics.json"
    ]


def test_atomic_json_serialization_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "diagnostics.json"
    destination.write_text("previous checkpoint\n", encoding="utf-8")

    with pytest.raises(TypeError):
        _write_json_atomic(destination, {"unsupported": object()})

    assert destination.read_text(encoding="utf-8") == "previous checkpoint\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_json_replacement_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "diagnostics.json"
    destination.write_text("previous checkpoint\n", encoding="utf-8")

    def fail_replace(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("dubanevics2026topological.analysis.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        _write_json_atomic(destination, {"complete": True})

    assert destination.read_text(encoding="utf-8") == "previous checkpoint\n"
    assert list(tmp_path.iterdir()) == [destination]
