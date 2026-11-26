"""Guard relative module paths in packages moved out of analysis."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src/dubanevics2026topological"


def _module_exists(path: Path) -> bool:
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _package_exports(path: Path) -> set[str]:
    init = path / "__init__.py"
    if not init.is_file():
        return set()
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_moved_packages_relative_import_modules_exist():
    missing = []
    for package in ("simplification", "threshold", "landscape"):
        for source in (PACKAGE / package).rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.level:
                    continue
                base = source.parent
                for _ in range(node.level - 1):
                    base = base.parent
                target = base.joinpath(*(node.module or "").split("."))
                if not _module_exists(target):
                    missing.append(
                        f"{source.relative_to(PACKAGE)}:{node.lineno}: {node.module}"
                    )
                elif node.module is None:
                    exports = _package_exports(target)
                    for alias in node.names:
                        if alias.name not in exports and not _module_exists(
                            target / alias.name
                        ):
                            missing.append(
                                f"{source.relative_to(PACKAGE)}:{node.lineno}: {alias.name}"
                            )
    assert not missing, "Unresolved relative imports:\n" + "\n".join(missing)
