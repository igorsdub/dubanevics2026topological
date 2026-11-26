"""The description API is the only production-facing analysis boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis import DescribedProducts, describe

PRODUCER_MODULES = {"graph_summary", "biological_annotations", "band_view_ridges"}
PRODUCER_NAMES = {
    "summarize_network",
    "_summarize_network",
    "summary_payload",
    "_summary_payload",
    "build_biological_annotations",
    "_build_biological_annotations",
    "export_biological_annotations",
    "_export_biological_annotations",
    "band_view_ridge_counts",
    "_band_view_ridge_counts",
    "load_band_view_ridge_counts",
    "_load_band_view_ridge_counts",
    "in_memory_stop_codon_peak_count",
    "_in_memory_stop_codon_peak_count",
}


def _function_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent.name
        parent = parents.get(parent)
    return None


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _allowed_import(
    filename: str,
    function: str | None,
    node: ast.ImportFrom,
    imported: ast.alias,
) -> bool:
    path = filename.replace("\\", "/")
    module = node.module or ""
    imported_names = {item.name for item in node.names}
    if (
        path.endswith(
            "src/dubanevics2026topological/simplification/selected_leaf_simplification.py"
        )
        and function == "build_candidate_leaf_view"
    ):
        return (
            node.level == 2
            and module == "analysis.biological_annotations"
            and imported.name == "_build_biological_annotations"
            and imported.asname is None
            and imported_names == {"_build_biological_annotations"}
        )
    if (
        path.endswith("scripts/validation/landscape_signatures.py")
        and function == "_description_signature"
    ):
        return (
            (
                module.endswith("analysis.biological_annotations")
                and imported.name == "build_biological_annotations"
                and imported_names == {"build_biological_annotations"}
            )
            or (
                module.endswith("analysis.graph_summary")
                and imported.name == "summarize_network"
                and imported_names == {"summarize_network"}
            )
        ) and imported.asname is None
    return False


def _production_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    module_aliases: dict[str, str] = {}
    name_aliases: dict[str, str] = {}
    violations: list[str] = []
    path = filename.replace("\\", "/")
    inside_analysis = "/analysis/" in path

    def imported_producer_module(
        node: ast.ImportFrom, imported: ast.alias
    ) -> str | None:
        module = node.module or ""
        leaf = module.rsplit(".", 1)[-1]
        selected_leaf_path = filename.replace("\\", "/").endswith(
            "src/dubanevics2026topological/simplification/selected_leaf_simplification.py"
        )
        if (
            selected_leaf_path
            and _function_name(node, parents) == "build_candidate_leaf_view"
            and leaf == "biological_annotations"
        ):
            return leaf
        if leaf in PRODUCER_MODULES and (
            ".analysis." in f".{module}."
            or inside_analysis
            or (node.module is None and inside_analysis)
        ):
            return leaf
        if module.endswith("analysis") and imported.name in PRODUCER_MODULES:
            return imported.name
        if (
            node.module is None
            and inside_analysis
            and imported.name in PRODUCER_MODULES
        ):
            return imported.name
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            function = _function_name(node, parents)
            for imported in node.names:
                local_name = imported.asname or imported.name
                source_module = imported_producer_module(node, imported)
                if source_module is not None:
                    module_aliases[local_name] = source_module
                    if not inside_analysis and not _allowed_import(
                        filename, function, node, imported
                    ):
                        violations.append(
                            f"{node.lineno}: imports module {source_module}"
                        )
                if source_module is not None and imported.name in PRODUCER_NAMES:
                    allowed = _allowed_import(filename, function, node, imported)
                    if not allowed:
                        name_aliases[local_name] = imported.name
                        violations.append(
                            f"{node.lineno}: imports producer {imported.name}"
                        )
        elif isinstance(node, ast.Import):
            for imported in node.names:
                dotted = imported.name
                module = next(
                    (
                        item
                        for item in PRODUCER_MODULES
                        if dotted.endswith(f"analysis.{item}")
                    ),
                    None,
                )
                if module is not None:
                    local_name = imported.asname or dotted.split(".", 1)[0]
                    module_aliases[local_name] = module
                    if not inside_analysis:
                        violations.append(f"{node.lineno}: imports module {module}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = _function_name(node, parents)
        if isinstance(node.func, ast.Name):
            called_name = name_aliases.get(node.func.id, node.func.id)
            if called_name in PRODUCER_NAMES:
                selected_leaf_edge = (
                    path.endswith(
                        "src/dubanevics2026topological/simplification/selected_leaf_simplification.py"
                    )
                    and function == "build_candidate_leaf_view"
                    and called_name == "_build_biological_annotations"
                )
                validation_edge = (
                    path.endswith("scripts/validation/landscape_signatures.py")
                    and function == "_description_signature"
                    and called_name
                    in {"build_biological_annotations", "summarize_network"}
                )
                if not selected_leaf_edge and not validation_edge:
                    violations.append(f"{node.lineno}: calls producer {called_name}")
        else:
            dotted = _dotted_name(node.func)
            if dotted is None:
                continue
            parts = dotted.split(".")
            local_module = module_aliases.get(parts[0])
            if local_module in PRODUCER_MODULES and parts[-1] in PRODUCER_NAMES:
                violations.append(
                    f"{node.lineno}: calls {local_module}.{parts[-1]} via module alias"
                )
            elif any(part in PRODUCER_MODULES for part in parts) and (
                parts[-1] in PRODUCER_NAMES
            ):
                violations.append(f"{node.lineno}: calls producer {dotted}")
    return violations


def test_analysis_exports_describe_and_product_bundle() -> None:
    assert callable(describe)
    assert DescribedProducts.__dataclass_params__.frozen


def test_production_code_uses_only_describe_for_product_description() -> None:
    repository = Path(__file__).parents[1]
    package = repository / "src/dubanevics2026topological"
    roots = [
        package / name
        for name in ("build", "figures", "scripts", "simplification", "tables")
    ]
    roots.extend((repository / "scripts", package / "analysis"))
    implementations = {
        package / "analysis" / name
        for name in (
            "graph_summary.py",
            "biological_annotations.py",
            "band_view_ridges.py",
            "describe.py",
        )
    }
    violations = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path in implementations:
                continue
            violations.extend(
                f"{path}:{item}"
                for item in _production_violations(
                    path.read_text(encoding="utf-8"), path.as_posix()
                )
            )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "from dubanevics2026topological.analysis import graph_summary as g\ng.summary_payload({})",
        "import dubanevics2026topological.analysis.graph_summary as g\ng._summary_payload({})",
        "from dubanevics2026topological.analysis.graph_summary import _summary_payload as make\nmake()",
        "from dubanevics2026topological.analysis.biological_annotations import _build_biological_annotations as make\nmake()",
    ],
)
def test_enforcement_rejects_alias_and_private_name_bypasses(source: str) -> None:
    assert _production_violations(source, "src/example/caller.py")


def test_enforcement_allows_only_the_selected_leaf_annotations_edge() -> None:
    allowed = """\
def build_candidate_leaf_view():
    from ..analysis.biological_annotations import _build_biological_annotations
    return _build_biological_annotations(None)
"""
    assert not _production_violations(
        allowed,
        "src/dubanevics2026topological/simplification/selected_leaf_simplification.py",
    )
    disallowed = allowed.replace(
        "build_biological_annotations", "export_biological_annotations"
    )
    assert _production_violations(
        disallowed,
        "src/dubanevics2026topological/simplification/selected_leaf_simplification.py",
    )
    broken = allowed.replace(
        "..analysis.biological_annotations", ".biological_annotations"
    )
    assert _production_violations(
        broken,
        "src/dubanevics2026topological/simplification/selected_leaf_simplification.py",
    )
    old_edge = broken.replace(
        "_build_biological_annotations", "build_biological_annotations"
    )
    assert _production_violations(
        old_edge,
        "src/dubanevics2026topological/simplification/selected_leaf_simplification.py",
    )


def test_enforcement_allows_only_validation_legacy_edges_in_scoped_function() -> None:
    allowed = """\
def _description_signature():
    from dubanevics2026topological.analysis.biological_annotations import build_biological_annotations
    from dubanevics2026topological.analysis.graph_summary import summarize_network
    build_biological_annotations(None)
    summarize_network(None, None, None)
"""
    assert not _production_violations(
        allowed, "scripts/validation/landscape_signatures.py"
    )
    outside_function = allowed.replace("def _description_signature():\n", "")
    assert _production_violations(
        dedent(outside_function), "scripts/validation/landscape_signatures.py"
    )


def test_requested_product_requires_its_inputs() -> None:
    with pytest.raises(ValueError, match="landscape_summary requires"):
        describe(products={"landscape_summary"})


def test_merge_flow_requires_its_inputs() -> None:
    with pytest.raises(ValueError, match="merge_flow requires"):
        describe(products={"merge_flow"})
    with pytest.raises(ValueError, match="merge_flow_pairs"):
        describe(
            products={"merge_flow"},
            merge_flow_tree=object(),
            merge_flow_state="published",
            merge_flow_vertices={},
            merge_flow_classes={},
            merge_flow_names={},
            merge_flow_settings={},
        )


def test_unknown_product_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown description products"):
        describe(products={"unknown"})


def test_annotations_require_an_existing_landscape() -> None:
    with pytest.raises(ValueError, match="annotations requires"):
        describe(products={"annotations"})


def test_ridge_counts_use_the_cached_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dubanevics2026topological.analysis import band_view_ridges

    expected = {"kem": {"total_count": 3, "functional_count": 2}}
    monkeypatch.setattr(
        band_view_ridges,
        "_load_band_view_ridge_counts",
        lambda comparison, bootstrap: expected,
    )
    result = describe(
        products={"ridge_counts"}, comparison={"cached": True}, bootstrap=object()
    )
    assert result.ridge_counts == expected


def test_ridge_counts_use_the_direct_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dubanevics2026topological.analysis import band_view_ridges

    expected = {"tcl": {"total_count": 4, "functional_count": 1}}
    monkeypatch.setattr(
        band_view_ridges,
        "_band_view_ridge_counts",
        lambda *args, **kwargs: expected,
    )
    result = describe(
        products={"ridge_counts"},
        comparison={},
        hills=pl.DataFrame(),
        tree_analysis=object(),
        ridge_config={},
    )
    assert result.ridge_counts == expected


def test_incomplete_ridge_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="direct ridge_counts requires"):
        describe(products={"ridge_counts"}, comparison={}, hills=pl.DataFrame())


def test_ambiguous_ridge_routes_are_rejected() -> None:
    with pytest.raises(ValueError, match="ridge_counts inputs are ambiguous"):
        describe(
            products={"ridge_counts"},
            comparison={},
            bootstrap=SimpleNamespace(hills=pl.DataFrame()),
            hills=pl.DataFrame(),
            tree_analysis=object(),
            ridge_config={},
        )
