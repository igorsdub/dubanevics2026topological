"""Smoke checks that the papkou notebooks use the figure registry seam."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
# Exploratory analysis stages draw from their own cached tables and own no figure
# spec, so they have no registry seam to cross: they are exempt from the check that
# `render_payload` and `get_spec` are used, and from that check only. Every notebook
# still must not call `build_figure`. docs/figure-inventory.md lists them.
EXPLORATORY_NOTEBOOKS = frozenset({"papkou2023_branch_decomposition"})
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("papkou2023_*.py"))
FIGURES_MODULE = "dubanevics2026topological.figures"


def _imported_from_figures(tree: ast.AST) -> set[str]:
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == FIGURES_MODULE
        for alias in node.names
    }


def _names(tree: ast.AST) -> set[str]:
    """Every name used or imported anywhere in the module."""
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    return names


def _called_names(tree: ast.AST) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_notebooks_are_discovered():
    assert NOTEBOOKS


def test_the_exemption_names_only_listed_exploratory_notebooks():
    inventory = (NOTEBOOK_DIR.parent / "docs/figure-inventory.md").read_text()
    for name in EXPLORATORY_NOTEBOOKS:
        assert (NOTEBOOK_DIR / f"{name}.py").is_file()
        assert f"{name}.py" in inventory


def test_the_build_figure_guard_sees_an_import_and_a_call():
    forbidden = ast.parse(
        "from dubanevics2026topological.figures import build_figure\nbuild_figure()"
    )
    assert "build_figure" in _names(forbidden)


@pytest.mark.parametrize("name", sorted(EXPLORATORY_NOTEBOOKS))
def test_an_exempt_notebook_never_uses_build_figure(name):
    """The real notebook is parsed here, so the guard does not depend on the order
    of the statements in ``test_notebook_uses_registry_render_seam``."""
    assert name in {path.stem for path in NOTEBOOKS}
    notebook = NOTEBOOK_DIR / f"{name}.py"
    assert "build_figure" not in _names(ast.parse(notebook.read_text()))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_uses_registry_render_seam(notebook):
    tree = ast.parse(notebook.read_text())
    assert "build_figure" not in _names(tree)

    imported = _imported_from_figures(tree)
    if notebook.stem in EXPLORATORY_NOTEBOOKS:
        return
    assert {"render_payload", "get_spec"} <= imported
    assert "render_payload" in _called_names(tree)

    if "save_appearance_settings" in imported:
        assert "save_appearance_settings" in _called_names(tree)
