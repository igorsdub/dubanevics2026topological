"""The article may import Topographer only through its public modules."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _private_imports(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "topographer" and any(p.startswith("_") for p in parts[1:]):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
    return found


def test_no_import_from_a_topographer_private_module():
    offenders = []
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            offenders.extend(_private_imports(path))
    assert offenders == []
