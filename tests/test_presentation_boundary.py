"""Guard the presentation path boundary while legacy callers are migrated."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = (
    ROOT / "src/dubanevics2026topological/figures",
    ROOT / "src/dubanevics2026topological/tables",
    ROOT / "notebooks",
)

# Transitional exceptions, removed as PRs 2–4 move callers to results/build.
# Package import bootstraps and config paths are outside this check.
LEGACY_OUTPUT_PATHS = {
    "src/dubanevics2026topological/tables/__init__.py": {"OUTPUT / 'tables'"},
    "notebooks/papkou2023_peak_uncertainty.py": {"ROOT / 'output'"},
    "notebooks/papkou2023_linked_tree_persistence.py": {
        "ROOT / 'output'",
        "ROOT / 'output/previews'",
    },
    "notebooks/graph_tree_persistence.py": {"ROOT / 'output/figures'"},
    "notebooks/papkou2023_study_tree_persistence_metrics.py": {
        "ROOT / 'output'",
        "ROOT / 'output/previews'",
    },
    "notebooks/papkou2023_full_tree_persistence.py": {
        "ROOT / 'output'",
        "ROOT / 'output/previews'",
    },
    "notebooks/papkou2023_study_tree_persistence_biology.py": {
        "ROOT / 'output'",
        "ROOT / 'output/previews'",
    },
    "notebooks/papkou2023_tree_comparison.py": {
        "ROOT / 'output'",
        "ROOT / 'output/previews'",
    },
}

# PR 3: codon populations and threshold diagnostics need saved products.
# PR 4: the legacy annotation views need saved products.
# Active mixture presentation is pending active-config product stages.
# Exceptions are scoped to functions.
LEGACY_SCIENCE_CALLS = {
    ("_legacy_mixture.py", "bootstrap_tree", "build_candidate_tree"),
    ("_legacy_mixture.py", "bootstrap_band_view", "build_band_simplified_view"),
    ("_legacy_mixture.py", "annotations", "describe"),
    ("registry.py", "_compose_bootstrap_codon", "extract_codon_entropy_populations"),
    ("registry.py", "_compose_bootstrap_codon", "resolve_candidate_selection"),
    ("registry.py", "_compose_bootstrap_codon", "compute_entropy_sweeps"),
    ("_leaf_data.py", "leaf_tables", "describe"),
    (
        "study_tree_persistence_biology.py",
        "build_study_tree_persistence_biology_figure",
        "describe",
    ),
}
SCIENTIFIC_CALLS = {entry[2] for entry in LEGACY_SCIENCE_CALLS}
# Pending active-config product stages: these functions reconstruct population
# membership and branch counts only for the explicitly selected mixture path.
LEGACY_MIXTURE_POPULATIONS = {
    ("_legacy_mixture.py", "tree_population"),
    ("_legacy_mixture.py", "selected_view_as_tree"),
}

# PR 3a: the pooled-scatter computation moved to a saved product.
# figures/ must neither import the module nor call pooled_scatter (no fallback).
POOLED_SCATTER_FORBIDDEN = "pooled_scatter"

# PR 3a (B): threshold refits moved to saved product.
# figures/ must not import or call the fit functions.
THRESHOLD_FIT_FORBIDDEN = "compute_candidate_functional_threshold"


def test_no_new_presentation_science_calls() -> None:
    figures = PRESENTATION[0]
    violations = []
    for path in figures.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(function):
                if not isinstance(call, ast.Call) or not isinstance(
                    call.func, ast.Name
                ):
                    continue
                name = call.func.id
                if (
                    name in SCIENTIFIC_CALLS
                    and (path.name, function.name, name) not in LEGACY_SCIENCE_CALLS
                ):
                    violations.append(
                        f"{path.name}:{call.lineno}: {function.name} calls {name}"
                    )
        # Also prohibit any import or attribute of the pooled-scatter computation.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "pooled_scatter" in node.module:
                    violations.append(
                        f"{path.name}:{node.lineno}: imports from pooled_scatter"
                    )
                if node.module and "functional_threshold" in (node.module or ""):
                    for n in node.names or []:
                        if THRESHOLD_FIT_FORBIDDEN in (n.name or ""):
                            violations.append(
                                f"{path.name}:{node.lineno}: imports threshold fit"
                            )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "pooled_scatter" in alias.name:
                        violations.append(
                            f"{path.name}:{node.lineno}: imports pooled_scatter"
                        )
                    if THRESHOLD_FIT_FORBIDDEN in (alias.name or ""):
                        violations.append(
                            f"{path.name}:{node.lineno}: imports threshold fit"
                        )
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == POOLED_SCATTER_FORBIDDEN
            ):
                violations.append(
                    f"{path.name}:{node.lineno}: references pooled_scatter"
                )
            elif (
                isinstance(node, ast.Attribute) and node.attr == THRESHOLD_FIT_FORBIDDEN
            ):
                violations.append(
                    f"{path.name}:{node.lineno}: references compute_candidate_functional_threshold"
                )
    assert not violations, "Unexpected science calls:\n" + "\n".join(violations)


def test_legacy_mixture_population_exceptions_are_explicit() -> None:
    figures = PRESENTATION[0]
    for filename, name in LEGACY_MIXTURE_POPULATIONS:
        tree = ast.parse((figures / filename).read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == name
            for node in tree.body
        )


def test_no_new_scientific_root_paths() -> None:
    violations = []
    for directory in PRESENTATION:
        for path in directory.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                    continue
                right = node.right
                literal = right.value if isinstance(right, ast.Constant) else None
                output_base = (
                    isinstance(node.left, ast.Name) and node.left.id == "OUTPUT"
                )
                scientific_root = isinstance(literal, str) and literal.split("/")[
                    0
                ] in {"output", "data", "results", "tables"}
                if (output_base or scientific_root) and ast.unparse(node) not in (
                    LEGACY_OUTPUT_PATHS.get(relative, set())
                ):
                    violations.append(f"{relative}:{node.lineno}: {ast.unparse(node)}")
    assert not violations, "New scientific root paths:\n" + "\n".join(violations)
