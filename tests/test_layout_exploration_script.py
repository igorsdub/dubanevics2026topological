"""The layout exploration script still imports: a removed helper must fail in CI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "layout_exploration.py"


def test_the_layout_exploration_script_imports(monkeypatch):
    # ``just layout-exploration`` runs the script directly, and no other test
    # reaches it, so a figure helper it imports can vanish without a failure.
    spec = importlib.util.spec_from_file_location("layout_exploration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its string annotations through ``sys.modules``.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    assert callable(module.build_panels)
    assert callable(module.build_candidate_leaf_view)
