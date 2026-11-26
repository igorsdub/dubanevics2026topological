"""Notebook dependency and explicit-export contracts for the construction figure."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/graph_tree_persistence.py"


def _cells():
    return [
        node
        for node in ast.parse(NOTEBOOK.read_text()).body
        if isinstance(node, ast.FunctionDef)
    ]


def test_sweep_controls_only_rebuild_single_stage_preview():
    dependencies = {}
    for cell in _cells():
        for statement in cell.body:
            if isinstance(statement, ast.Return) and isinstance(
                statement.value, ast.Tuple
            ):
                for name in statement.value.elts:
                    if isinstance(name, ast.Name):
                        dependencies[name.id] = {arg.arg for arg in cell.args.args}

    def ancestors(name, seen=None):
        seen = set() if seen is None else seen
        for dependency in dependencies.get(name, set()) - seen:
            seen.add(dependency)
            ancestors(dependency, seen)
        return seen

    controls = {
        "checkpoint",
        "threshold",
        "panel_width",
        "panel_height",
        "regular_tree_size",
        "overall_font_size",
        "stage_title_size",
    }
    notebook_source = NOTEBOOK.read_text()
    assert "subtitle_size" not in notebook_source
    assert '"legend_font_size": overall_font_size.value' not in notebook_source
    assert not ancestors("construction") & controls
    assert not ancestors("manuscript_chart") & {"checkpoint", "threshold"}
    assert ancestors("preview_chart") & {"checkpoint", "threshold"} == {
        "checkpoint",
        "threshold",
    }


class _StoppedCell(Exception):
    pass


class _ExportRuntime:
    @staticmethod
    def stop(condition):
        if condition:
            raise _StoppedCell

    @staticmethod
    def md(message):
        return message


def _export_cell():
    cell = copy.deepcopy(
        next(
            cell
            for cell in _cells()
            if {arg.arg for arg in cell.args.args}
            >= {"export_button", "manuscript_chart", "ROOT"}
        )
    )
    cell.decorator_list = []
    cell.name = "export_cell"
    namespace = {}
    exec(  # noqa: S102 - execute the trusted local notebook export cell
        compile(ast.Module(body=[cell], type_ignores=[]), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace["export_cell"]


@pytest.mark.parametrize("clicked", [False, True])
def test_export_requires_button_and_uses_canonical_names(tmp_path, clicked):
    calls = []
    chart = SimpleNamespace(save=lambda path, **kwargs: calls.append((path, kwargs)))
    arguments = {
        "ROOT": tmp_path,
        "export_button": SimpleNamespace(value=clicked),
        "manuscript_chart": chart,
        "mo": _ExportRuntime,
        "settings": {"figures": {"png_dpi": 300}},
    }
    destination = tmp_path / "output/figures"
    if not clicked:
        with pytest.raises(_StoppedCell):
            _export_cell()(**arguments)
        assert not destination.exists()
        assert not calls
    else:
        _export_cell()(**arguments)
        assert calls == [
            (destination / "graph_tree_persistence.svg", {}),
            (destination / "graph_tree_persistence.png", {"ppi": 300}),
        ]
