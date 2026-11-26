"""Persist figure appearance edits to ``config/figures.toml``."""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from .registry import FIGURE_IDS


def preview_settings(
    settings: dict[str, Any], figure_id: str, appearance: dict[str, Any]
) -> dict[str, Any]:
    """Return a copy of ``settings`` with one figure's appearance overrides merged.

    ``appearance`` is shallow-merged into the existing figure entry (top-level
    keys replace, nothing else is dropped). Raises ``KeyError`` when ``figure_id``
    has no config table. The input mapping is never mutated; only the nested containers on the path to
    ``settings["figures"]["figures"][figure_id]`` are copied.
    """
    figures = settings["figures"]
    return {
        **settings,
        "figures": {
            **figures,
            "figures": {
                **figures["figures"],
                figure_id: {**figures["figures"][figure_id], **appearance},
            },
        },
    }


def save_appearance_settings(
    figure_id: str, settings: dict[str, Any], appearance: dict[str, Any]
) -> Path:
    """Atomically save one figure's appearance while preserving all TOML settings."""
    if figure_id not in FIGURE_IDS:
        raise ValueError(f"Unknown figure ID {figure_id!r}; choose from {FIGURE_IDS}.")
    del settings
    path = Path(__file__).resolve().parents[3] / "config/figures.toml"
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    figures = document["figures"]
    figures[figure_id] = {**figures[figure_id], **appearance}
    lines = ["# Publication figure appearance settings", ""]
    _write_toml_table(lines, "", document)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".figures.", suffix=".toml", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_toml_table(lines: list[str], name: str, values: dict[str, Any]) -> None:
    """Write root values before recursively writing every nested table."""
    scalar_values = {
        key: value for key, value in values.items() if not isinstance(value, dict)
    }
    if name:
        lines.append(f"[{name}]")
    lines.extend(
        f"{key} = {_toml_value(value)}" for key, value in scalar_values.items()
    )
    lines.append("")
    for key, value in values.items():
        if isinstance(value, dict):
            child_name = f"{name}.{key}" if name else key
            _write_toml_table(lines, child_name, value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))
