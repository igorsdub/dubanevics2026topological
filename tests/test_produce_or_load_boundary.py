"""Extraction boundary of the produce-or-load helper (ADR 0013).

``analysis/produce_or_load.py`` is meant to be liftable into a standalone
package (a Python counterpart of DrWatson.jl). These tests parse the source of
the helper and of its siblings ``analysis/checkpoint.py`` and
``analysis/git_stamp.py`` and fail on any import that would
tie them to this project or to its scientific stack. They read the files as
text and never import them.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

import dubanevics2026topological

PACKAGE = "dubanevics2026topological"
ANALYSIS = f"{PACKAGE}.analysis"
SOURCE = Path(dubanevics2026topological.__file__).parent / "analysis"
# Imports may name only the standard library and, through ALLOWED_PROJECT_IMPORTS,
# the helper's sibling module. Anything else (barseqfit, watchfiles, a new
# dependency) is a violation without having to be listed here.
STDLIB = frozenset(sys.stdlib_module_names)
# Standard-library modules that still tie the helper to this project: it must
# not read configuration files itself.
FORBIDDEN_STDLIB = frozenset({"tomllib"})
DYNAMIC_IMPORTS = frozenset({"__import__", "import_module"})
# module file -> project modules it may import (absolute dotted names).
ALLOWED_PROJECT_IMPORTS = {
    "produce_or_load.py": frozenset(
        {f"{ANALYSIS}.checkpoint", f"{ANALYSIS}.git_stamp"}
    ),
    "checkpoint.py": frozenset(),
    "git_stamp.py": frozenset(),
}


def _imported_modules(source: str, package: str) -> list[str]:
    """Absolute dotted names of every import in ``source``, at any nesting depth.

    ``package`` is the dotted package the module lives in, used to resolve
    relative imports. ``from . import x`` yields ``<package>.x``.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")
            if node.level:
                base = base[: len(base) - (node.level - 1)]
                prefix = ".".join(base)
            else:
                prefix = ""
            if node.module:
                found.append(".".join(part for part in (prefix, node.module) if part))
            else:
                found.extend(f"{prefix}.{alias.name}" for alias in node.names)
    return found


def _dynamic_imports(source: str) -> list[str]:
    """Calls to ``__import__`` or ``importlib.import_module``, however spelled."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in DYNAMIC_IMPORTS:
                found.append(name)
    return found


def _violations(source: str, package: str, allowed: frozenset[str]) -> list[str]:
    problems = []
    for name in _imported_modules(source, package):
        top = name.split(".")[0]
        if top == PACKAGE:
            if name not in allowed:
                problems.append(f"imports {name}")
        elif top not in STDLIB or top in FORBIDDEN_STDLIB:
            problems.append(f"imports {name}")
    problems.extend(f"calls {name}" for name in _dynamic_imports(source))
    return problems


@pytest.mark.parametrize("filename", sorted(ALLOWED_PROJECT_IMPORTS))
def test_helper_modules_stay_inside_the_extraction_boundary(filename: str) -> None:
    source = (SOURCE / filename).read_text(encoding="utf-8")
    assert _violations(source, ANALYSIS, ALLOWED_PROJECT_IMPORTS[filename]) == []


def test_helper_imports_its_two_siblings_and_nothing_else_from_the_project() -> None:
    source = (SOURCE / "produce_or_load.py").read_text(encoding="utf-8")
    project = {
        name
        for name in _imported_modules(source, ANALYSIS)
        if name.split(".")[0] == PACKAGE
    }
    assert project == {f"{ANALYSIS}.checkpoint", f"{ANALYSIS}.git_stamp"}


@pytest.mark.parametrize(
    "line",
    [
        "import polars as pl",
        "import numpy",
        "from numpy import array",
        "import pyarrow.parquet",
        "import topographer",
        "from topographer.api import x",
        "import tomllib",
        "import tomli",
        "from tomllib import loads",
        "import barseqfit",
        "from barseqfit import fit",
        "import watchfiles",
        "from watchfiles import watch",
        "import requests",
        "importlib.import_module('numpy')",
        "__import__('polars')",
        "from importlib import import_module\nimport_module('x')",
        "import importlib as il\nil.import_module('x')",
        "import dubanevics2026topological",
        "from dubanevics2026topological import analysis",
        "from dubanevics2026topological.simplification.bootstrap_chunks import x",
        "from . import analysis",
        "from .describe import x",
        "from .. import build",
        "from ..build import x",
        "from ..tables import x",
    ],
)
def test_the_checker_flags_forbidden_imports(line: str) -> None:
    allowed = ALLOWED_PROJECT_IMPORTS["produce_or_load.py"]
    assert _violations(line, ANALYSIS, allowed)
    late = f"def late():\n{textwrap.indent(line, '    ')}\n"
    assert _violations(late, ANALYSIS, allowed)


@pytest.mark.parametrize(
    "line",
    [
        "import json",
        "from importlib import metadata",
        "from collections.abc import Callable",
        "from __future__ import annotations",
        "import os, errno, uuid, warnings",
        "from datetime import UTC, datetime",
        "from .checkpoint import checkpoint_lock",
        "from . import checkpoint",
        "from dubanevics2026topological.analysis.checkpoint import checkpoint_lock",
        "from .git_stamp import tag",
        "from . import git_stamp",
        "import subprocess",
    ],
)
def test_the_checker_allows_stdlib_and_the_siblings(line: str) -> None:
    allowed = ALLOWED_PROJECT_IMPORTS["produce_or_load.py"]
    assert _violations(line, ANALYSIS, allowed) == []


def test_checkpoint_may_not_import_even_the_helper_or_itself() -> None:
    allowed = ALLOWED_PROJECT_IMPORTS["checkpoint.py"]
    assert _violations("from .produce_or_load import Stage", ANALYSIS, allowed)
    assert _violations("from .checkpoint import x", ANALYSIS, allowed)


def test_git_stamp_may_not_import_the_helper_or_anything_else_from_the_project() -> (
    None
):
    allowed = ALLOWED_PROJECT_IMPORTS["git_stamp.py"]
    assert _violations("from .produce_or_load import Stage", ANALYSIS, allowed)
    assert _violations("from .checkpoint import checkpoint_lock", ANALYSIS, allowed)
    assert _violations("from .git_stamp import tag", ANALYSIS, allowed)
    assert _violations("from ..stages import project_stage", ANALYSIS, allowed)


# Names that tie code to this project: its dependencies, its scheduling keys,
# its configuration file, its bootstrap machinery and its task runner.
PROJECT_WORDS = (
    "polars",
    "numpy",
    "pyarrow",
    "topographer",
    "n_jobs",
    "chunk_size",
    "analysis.toml",
    "bootstrap",
    "mirror",
    "just ",
    "-analysis",
)


def _code_text(source: str) -> list[str]:
    """Identifiers and string literals outside docstrings and comments."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.append(node.value)
        elif isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(node, ast.arg):
            found.append(node.arg)
        elif isinstance(node, ast.alias):
            found.append(node.name)
    return found


@pytest.mark.parametrize("filename", sorted(ALLOWED_PROJECT_IMPORTS))
def test_helper_code_carries_no_project_defaults(filename: str) -> None:
    source = (SOURCE / filename).read_text(encoding="utf-8")
    hits = [
        (word, text)
        for text in _code_text(source)
        for word in PROJECT_WORDS
        if word in text.lower()
    ]
    assert hits == []


def test_the_code_scan_sees_defaults_and_ignores_docstrings() -> None:
    source = (
        '"""Mentions polars and bootstrap chunks in prose."""\n'
        "DEFAULT = ('polars', 'numpy')\n"
        "def f(n_jobs=1):\n"
        '    """Mentions topographer."""\n'
        "    return f'just {n_jobs}'\n"
    )
    text = _code_text(source)
    assert "polars" in text and "n_jobs" in text and "just " in " ".join(text)
    assert "Mentions polars and bootstrap chunks in prose." not in text
    assert "Mentions topographer." not in text


# Words that tie a docstring to this project. A docstring is lifted with its
# module, so it must not name this project's files, packages or scheduling keys.
DOCSTRING_PROJECT_WORDS = (
    "polars",
    "numpy",
    "pyarrow",
    "topographer",
    "n_jobs",
    "chunk_size",
    "fitnesstree",
    "bootstrap",
    "mirror",
    "stages.py",
    "tests/",
    "config/",
    ".toml",
    "justfile",
)


def _docstrings(source: str) -> list[tuple[str, str]]:
    """``(owner, text)`` for the module, every class and every function docstring."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            text = ast.get_docstring(node)
            if text:
                found.append((getattr(node, "name", "<module>"), text))
    return found


def _docstring_hits(source: str) -> list[tuple[str, str]]:
    return [
        (owner, word)
        for owner, text in _docstrings(source)
        for word in DOCSTRING_PROJECT_WORDS
        if word in text.lower()
    ]


@pytest.mark.parametrize("filename", sorted(ALLOWED_PROJECT_IMPORTS))
def test_helper_docstrings_name_no_project_files_or_packages(filename: str) -> None:
    """Docstrings are exempt from the code scan but are lifted with the module."""
    source = (SOURCE / filename).read_text(encoding="utf-8")
    assert _docstring_hits(source) == []


def test_the_docstring_scan_reads_module_class_and_function_docstrings() -> None:
    source = (
        '"""Reads config/analysis.toml."""\n'
        "class C:\n"
        '    """Wraps polars."""\n'
        "    def m(self):\n"
        '        """Calls stages.py."""\n'
        "def f():\n"
        '    """Sets n_jobs."""\n'
        "def g():\n"
        '    """A generic note."""\n'
    )
    assert sorted(_docstring_hits(source)) == sorted(
        [
            ("<module>", "config/"),
            ("<module>", ".toml"),
            ("C", "polars"),
            ("m", "stages.py"),
            ("f", "n_jobs"),
        ]
    )


def _public_names(source: str) -> set[str]:
    """Top-level names the module defines that do not start with an underscore."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def test_module_docstring_lists_the_full_public_surface() -> None:
    source = (SOURCE / "produce_or_load.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    public = _public_names(source)
    assert {
        "Stage",
        "Checkpoint",
        "stage_fingerprint",
        "produce_or_load",
        "publication_state",
        "PROVENANCE_NAME",
        "SCHEMA",
    } <= public
    assert [name for name in sorted(public) if name not in docstring] == []
    assert "export_to" in docstring
    # Defined in ``checkpoint.py`` and re-exported, so it is not a definition here.
    assert "IGNORED_NAMES" in docstring


def test_module_docstring_documents_savename_and_name_keys() -> None:
    source = (SOURCE / "produce_or_load.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    assert "savename" in _public_names(source)
    # ``savename`` is implemented, so it is no longer announced as planned, and the
    # stage field that carries its aliases is documented beside it.
    assert "name_keys" in docstring
    assert "not implemented yet" not in docstring


def test_ignored_names_is_defined_in_checkpoint_and_only_imported_by_the_helper() -> (
    None
):
    helper = (SOURCE / "produce_or_load.py").read_text(encoding="utf-8")
    sibling = (SOURCE / "checkpoint.py").read_text(encoding="utf-8")
    assert "IGNORED_NAMES" in _public_names(sibling)
    assert "IGNORED_NAMES" not in _public_names(helper)
    imported = {
        alias.name
        for node in ast.parse(helper).body
        if isinstance(node, ast.ImportFrom) and node.module == "checkpoint"
        for alias in node.names
    }
    assert "IGNORED_NAMES" in imported


def test_git_stamp_docstring_lists_its_public_surface() -> None:
    source = (SOURCE / "git_stamp.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    public = _public_names(source)
    assert public == {"GIT_KEY", "git_stamp", "tag", "untag"}
    assert [name for name in sorted(public) if name not in docstring] == []


def test_module_docstring_names_the_git_stamp_module_and_its_verbs() -> None:
    source = (SOURCE / "produce_or_load.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    assert [
        name for name in ("git_stamp", "tag", "GIT_KEY") if name not in docstring
    ] == []


def test_module_docstring_says_project_values_come_from_the_stage_factory() -> None:
    source = (SOURCE / "produce_or_load.py").read_text(encoding="utf-8")
    docstring = " ".join((ast.get_docstring(ast.parse(source)) or "").split())
    assert "stage factory" in docstring
    assert "once" in docstring


def test_the_public_name_scan_reads_definitions_not_imports() -> None:
    source = (
        "from x import Imported\n"
        "PUBLIC = 1\n"
        "_private = 2\n"
        "TYPED: int = 3\n"
        "class Shown: ...\n"
        "def shown(): ...\n"
        "def _hidden(): ...\n"
    )
    assert _public_names(source) == {"PUBLIC", "TYPED", "Shown", "shown"}
