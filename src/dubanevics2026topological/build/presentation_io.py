"""Paths and writes for presentation outputs, owned by the build layer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ..results.context import RunContext


def figure_path(context: RunContext, stem: str, extension: str) -> Path:
    """Locate a canonical figure without creating it."""
    return context.output_root / "figures" / f"{stem}.{extension}"


def save_preview(
    context: RunContext, chart: Any, stem: str, extension: str, **options: Any
) -> Path:
    """Write a notebook preview and return its location."""
    directory = context.output_root / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.{extension}"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{stem}.tmp-", suffix=f".{extension}", dir=directory
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        chart.save(temporary, **options)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
