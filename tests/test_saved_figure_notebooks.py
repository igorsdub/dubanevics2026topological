"""Notebook previews use the same routed input composition as the build."""

import ast
from pathlib import Path

import pytest

NOTEBOOKS = (
    "papkou2023_tree_comparison.py",
    "papkou2023_full_tree_histogram.py",
    "papkou2023_published_tree_histogram.py",
    "papkou2023_candidate_tree_histogram.py",
)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_composes_loaded_figure_inputs(notebook):
    path = Path(__file__).resolve().parents[1] / "notebooks" / notebook
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "comparison"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
    assert value.func.id == "compose_payload"
    assert len(value.args) >= 2 and isinstance(value.args[1], ast.Call)
    assert isinstance(value.args[1].func, ast.Name)
    assert value.args[1].func.id == "load_figure_inputs"
    renders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "render_payload"
    ]
    assert renders
    assert all(
        len(call.args) >= 2
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "comparison"
        for call in renders
    )
