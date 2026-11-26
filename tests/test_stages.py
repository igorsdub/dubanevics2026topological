"""The project stage factory and the guard that every stage goes through it."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from _real_stages import (
    build_bootstrap_stage,
    build_registry_stages,
    config,
    needs_raw_data,
)

import dubanevics2026topological
from dubanevics2026topological import stages
from dubanevics2026topological.analysis import produce_or_load as pol
from dubanevics2026topological.analysis.checkpoint import (
    parse_version_name,
    version_name,
)
from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint

PACKAGE_DIR = Path(dubanevics2026topological.__file__).parent
HELPER = PACKAGE_DIR / "analysis" / "produce_or_load.py"
FACTORY_MODULE = PACKAGE_DIR / "stages.py"
# Project modules that no fingerprint may cover: the stage factory and the rule for
# digesting provenance records.
OUTSIDE_FINGERPRINTS = (FACTORY_MODULE, PACKAGE_DIR / "provenance.py")
# Stage fields whose values only the project factory may supply.
FACTORY_FIELDS = frozenset({"execution_keys", "libraries", "name_keys"})
# The only modules whose whole source a fingerprint may hash.
HASHED_THIRD_PARTY = frozenset({"topographer", "barseqfit"})


def test_the_factory_supplies_the_project_values(tmp_path: Path) -> None:
    stage = stages.project_stage("demo", settings={"a": 1}, root=tmp_path)
    assert stage.name == "demo" and stage.root == tmp_path
    assert set(stage.execution_keys) == {"n_jobs", "chunk_size"}
    assert tuple(stage.libraries) == ("polars", "numpy", "pyarrow", "topographer")
    assert set(stages.EXECUTION_KEYS) == {"n_jobs", "chunk_size"}
    assert tuple(stages.STAGE_LIBRARIES) == stage.libraries


def test_the_factory_gives_a_stage_without_aliases_none(tmp_path: Path) -> None:
    stage = stages.project_stage("demo", settings={"a": 1}, root=tmp_path)
    assert dict(stage.name_keys) == {}


def test_the_factory_keeps_scheduling_keys_out_of_the_fingerprint(
    tmp_path: Path,
) -> None:
    code = tmp_path / "code.py"
    code.write_text("v1", encoding="utf-8")

    def stage(n_jobs: int, chunk_size: int, alpha: float):
        return stages.project_stage(
            "demo",
            settings={
                "alpha": alpha,
                "n_jobs": n_jobs,
                "bootstrap": {"chunk_size": chunk_size, "draws": 10},
            },
            code=(code,),
            root=tmp_path,
        )

    assert stage_fingerprint(stage(1, 5, 0.05)) == stage_fingerprint(stage(8, 50, 0.05))
    assert stage_fingerprint(stage(1, 5, 0.05)) != stage_fingerprint(stage(1, 5, 0.01))


def test_the_factory_passes_inputs_and_code_through(tmp_path: Path) -> None:
    data, code = tmp_path / "in.txt", tmp_path / "code.py"
    data.write_text("d", encoding="utf-8")
    code.write_text("c", encoding="utf-8")
    stage = stages.project_stage(
        "demo", settings={}, inputs=(data,), code=(code,), root=tmp_path
    )
    assert tuple(stage.inputs) == (data,) and tuple(stage.code) == (code,)


def _stage_constructions(source: str) -> list[str]:
    """Name of the enclosing function (or ``<module>``) of each way to get a Stage.

    Flags every use of the class as a value, not only a call: ``Stage(...)``,
    ``module.Stage``, an alias made by ``from ... import Stage as Alias``, an
    assignment such as ``S = Stage``, ``functools.partial(Stage, ...)`` and
    ``getattr(module, "Stage")``. Annotations and imports are not uses. It also
    flags any other call that passes ``execution_keys`` or ``libraries`` (for
    example ``dataclasses.replace(stage, execution_keys=())``), because that
    rebuilds a stage without the project values.
    """
    tree = ast.parse(source)
    names = {"Stage"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "Stage"
            )
    found: list[str] = []

    def is_stage(node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id in names) or (
            isinstance(node, ast.Attribute) and node.attr == "Stage"
        )

    def visit(node: ast.AST, scope: str) -> None:
        if (
            isinstance(node, ast.Call)
            and not is_stage(node.func)
            and any(keyword.arg in FACTORY_FIELDS for keyword in node.keywords)
        ):
            found.append(scope)
        if is_stage(node) or (isinstance(node, ast.Constant) and node.value == "Stage"):
            found.append(scope)
        inner = (
            node.name
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            else scope
        )
        for field, value in ast.iter_fields(node):
            if field in {"annotation", "returns"}:
                continue
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    visit(child, inner)

    visit(tree, "<module>")
    return found


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Stage('s', {})", ["<module>"]),
        ("def f():\n    return Stage('s', {})\n", ["f"]),
        ("def f():\n    return produce_or_load.Stage('s', {})\n", ["f"]),
        ("from x import Stage as S\ndef f():\n    return S('s', {})\n", ["f"]),
        ("class C:\n    def m(self):\n        return Stage('s', {})\n", ["m"]),
        ("S = Stage\nS('s', {})\n", ["<module>"]),
        ("def f():\n    S = Stage\n    return S('s', {})\n", ["f"]),
        ("def f():\n    return functools.partial(Stage, 's')\n", ["f"]),
        ("def f(mod):\n    return getattr(mod, 'Stage')('s', {})\n", ["f"]),
        ("def f(s):\n    return dataclasses.replace(s, execution_keys=())\n", ["f"]),
        ("def f(s):\n    return replace(s, libraries=('numpy',))\n", ["f"]),
        ("def f(s):\n    return replace(s, name_keys={})\n", ["f"]),
        ("def f(s):\n    return dataclasses.replace(s, name_keys=None)\n", ["f"]),
        ("def f(s):\n    return replace(s, settings={})\n", []),
        ("def f(stage: Stage) -> Stage:\n    return build(stage)\n", []),
        ("def f():\n    stage: Stage = build()\n    return stage\n", []),
        ("def f(stage: 'Stage') -> 'Stage':\n    return build(stage)\n", []),
        ("from x import Stage\nimport y.z as Stage_module\n", []),
        ("def f():\n    return project_stage('s', settings={})\n", []),
    ],
)
def test_the_guard_finds_every_way_to_build_a_stage(
    source: str, expected: list[str]
) -> None:
    assert _stage_constructions(source) == expected


def test_only_the_factory_and_the_helper_construct_stage() -> None:
    """Direct ``Stage(...)`` calls would silently drop the project values."""
    offenders = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path == HELPER:
            continue
        scopes = _stage_constructions(path.read_text(encoding="utf-8"))
        if path == FACTORY_MODULE:
            assert scopes == ["project_stage"]
        elif scopes:
            offenders[path.relative_to(PACKAGE_DIR).as_posix()] = scopes
    assert offenders == {}


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _hashes_the_module(source: str, filename: str) -> list[str]:
    """Ways a module could put the file ``filename`` into a fingerprint.

    Finds a string literal outside docstrings whose last path part is
    ``filename``, which covers ``with_name("stages.py")``,
    ``parents[1] / "stages.py"`` and ``"src/dubanevics2026topological/stages.py"``.
    Also finds a ``package_source_hashes`` call over anything but the named
    third-party modules, such as ``package_source_hashes(dubanevics2026topological)``.
    """
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value.replace("\\", "/").rsplit("/", 1)[-1] == filename
        ):
            found.append(f"names {node.value!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            called = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            )
            if called == "package_source_hashes" and not all(
                isinstance(arg, ast.Name) and arg.id in HASHED_THIRD_PARTY
                for arg in node.args
            ):
                found.append("hashes a package other than the third-party ones")
    return found


@pytest.mark.parametrize(
    ("source", "filename", "expected"),
    [
        ("P = 'src/dubanevics2026topological/stages.py'", "stages.py", 1),
        (
            "P = ROOT / 'src' / 'dubanevics2026topological' / 'stages.py'",
            "stages.py",
            1,
        ),
        ("P = Path(__file__).parents[1] / 'stages.py'", "stages.py", 1),
        ("P = Path(__file__).with_name('stages.py')", "stages.py", 1),
        ("P = ('a', 'stages.py')", "stages.py", 1),
        ("P = 'src/dubanevics2026topological/provenance.py'", "provenance.py", 1),
        ("P = Path(__file__).with_name('provenance.py')", "provenance.py", 1),
        ("P = Path(__file__).parents[1] / 'provenance.py'", "provenance.py", 1),
        ("P = ('a', 'provenance.py')", "provenance.py", 1),
        ("package_source_hashes(dubanevics2026topological)", "stages.py", 1),
        ("package_source_hashes(dubanevics2026topological)", "provenance.py", 1),
        ("analysis.package_source_hashes(dubanevics2026topological)", "stages.py", 1),
        (
            "package_source_hashes(topographer, dubanevics2026topological)",
            "stages.py",
            1,
        ),
        ("package_source_hashes(*packages)", "stages.py", 1),
        ("package_source_hashes(topographer)", "stages.py", 0),
        ("package_source_hashes(barseqfit, topographer)", "provenance.py", 0),
        ("P = Path(__file__).with_name('clamped_landscapes.py')", "stages.py", 0),
        ("P = Path(__file__).with_name('clamped_landscapes.py')", "provenance.py", 0),
        ("P = 'tests/test_stages.py'", "stages.py", 0),
        ("P = 'bootstages.py'", "stages.py", 0),
        ("P = 'stages.py'", "provenance.py", 0),
        ("P = 'provenance.py'", "stages.py", 0),
        ("P = 'output/provenance.json'", "provenance.py", 0),
        ('"""Mentions stages.py in prose."""\nX = 1\n', "stages.py", 0),
        ('"""Mentions provenance.py in prose."""\nX = 1\n', "provenance.py", 0),
        ("def f():\n    '''stages.py'''\n    return 1\n", "stages.py", 0),
        ("def f():\n    '''provenance.py'''\n    return 1\n", "provenance.py", 0),
    ],
)
def test_the_hash_guard_finds_ways_to_hash_a_module_outside_the_fingerprints(
    source: str, filename: str, expected: int
) -> None:
    assert len(_hashes_the_module(source, filename)) == expected


@pytest.mark.parametrize("module", OUTSIDE_FINGERPRINTS, ids=lambda path: path.name)
def test_the_module_is_not_part_of_any_stage_fingerprint(module: Path) -> None:
    """Fingerprints hash named files and the analysis, figures and tables trees.

    ``stages.py`` and ``provenance.py`` sit in none of those trees, and no module
    names their paths or hashes the package that holds them, so editing either is
    not a science change.
    """
    assert module.is_file()
    for tree in ("analysis", "figures", "tables"):
        assert not module.is_relative_to(PACKAGE_DIR / tree)
    offenders = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path == module:
            continue
        found = _hashes_the_module(path.read_text(encoding="utf-8"), module.name)
        if found:
            offenders[path.relative_to(PACKAGE_DIR).as_posix()] = found
    assert offenders == {}


# ---- savename aliases (ADR 0013, D2) ----------------------------------------------

ALIAS = re.compile(r"[a-z][a-z0-9]*")
STAGE_FACTORY_CALLS = frozenset({"project_stage", "_stage"})


def _stage_names(source: str) -> set[str]:
    """Names of the stages a module declares through the project factory.

    Finds calls to ``project_stage`` and to a module's own ``_stage`` wrapper whose
    first argument is a string literal, or a module-level name bound to one (the
    bootstrap stage names itself through a ``STAGE`` constant).
    """
    tree = ast.parse(source)
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        func = node.func
        called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if called not in STAGE_FACTORY_CALLS:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
        elif isinstance(first, ast.Name) and first.id in constants:
            found.add(constants[first.id])
    return found


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("project_stage('a', settings={})", {"a"}),
        ("def f():\n    return _stage('b-c', settings={})\n", {"b-c"}),
        ("x.project_stage('d')", {"d"}),
        ("STAGE = 'e'\nproject_stage(STAGE, settings={})\n", {"e"}),
        ("project_stage(name, settings={})", set()),
        ("STAGE = 'e'\nother(STAGE)\n", set()),
        ("project_stage(settings={})", set()),
        ("def project_stage(name):\n    return name\n", set()),
    ],
)
def test_the_stage_name_scan_reads_literals_and_module_constants(
    source: str, expected: set[str]
) -> None:
    assert _stage_names(source) == expected


def _declared_stage_names() -> set[str]:
    names: set[str] = set()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        names |= _stage_names(path.read_text(encoding="utf-8"))
    return names


def test_the_stage_name_scan_finds_the_known_stages() -> None:
    names = _declared_stage_names()
    assert {"analysis", "bootstrap-smoothed-tree", "codon-entropy"} <= names
    assert {"clamped-comparison", "peak-uncertainty"} <= names
    # Seven table stages beside codon-entropy (the contrast stage is gone).
    assert len([name for name in names if name.endswith("tables")]) == 7


def test_the_registry_holds_the_proposed_entries() -> None:
    assert stages.SAVENAME_KEYS == {
        "analysis": {
            "landscape": "config.dataset.landscape",
            "tree": "config.tree.kind",
        },
        "bootstrap-smoothed-tree": {
            "draws": "scientific.config.n_replicates",
            "seed": "scientific.config.seed",
        },
        "codon-entropy": {
            "qthreshold": "candidate_peak_biology.q_value_threshold",
            "sweep": "codon_entropy.sweep_points",
        },
    }


def test_every_registry_entry_names_a_real_stage() -> None:
    assert set(stages.SAVENAME_KEYS) <= _declared_stage_names()


def test_registry_entries_are_well_formed() -> None:
    for name, name_keys in stages.SAVENAME_KEYS.items():
        assert name_keys, f"{name} has an empty entry; leave the stage out instead."
        for alias, path in name_keys.items():
            assert ALIAS.fullmatch(alias), (name, alias)
            assert all(part for part in path.split(".")), (name, path)


def test_no_registry_entry_crosses_an_execution_key() -> None:
    execution = set(stages.EXECUTION_KEYS)
    for name, name_keys in stages.SAVENAME_KEYS.items():
        for alias, path in name_keys.items():
            assert not execution & set(path.split(".")), (name, alias, path)


@needs_raw_data
def test_each_alias_resolves_against_the_settings_the_factory_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dubanevics2026topological.analysis.produce_or_load import savename

    real = build_registry_stages(monkeypatch)
    # An entry without a builder here would go unchecked.
    assert set(real) == set(stages.SAVENAME_KEYS)
    for name, stage in real.items():
        assert dict(stage.name_keys) == stages.SAVENAME_KEYS[name]
        label = savename(
            stage.settings,
            stage.name_keys,
            execution_keys=stage.execution_keys,
            stage=name,
        )
        assert label
        # Every alias is written, and the label is a legal version name.
        for alias in stage.name_keys:
            assert re.search(rf"(^|_){alias}=", label), (name, alias, label)
        named = version_name("0" * 64, "1" * 64, label)
        assert parse_version_name(named).savename == label


@needs_raw_data
def test_the_labels_of_the_real_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    from dubanevics2026topological.analysis.produce_or_load import savename

    scientific = config()
    real = build_registry_stages(monkeypatch)

    def label(name: str) -> str:
        stage = real[name]
        return savename(
            stage.settings, stage.name_keys, execution_keys=stage.execution_keys
        )

    landscape = scientific["dataset"]["landscape"]
    kind = scientific["tree"]["kind"]
    assert label("analysis") == f"landscape={landscape}_tree={kind}"
    run = scientific["bootstrap_smoothed_tree"]
    assert label("bootstrap-smoothed-tree") == (
        f"draws={run['n_replicates']}_seed={run['seed']}"
    )
    assert re.fullmatch(r"qthreshold=[0-9.e+-]+_sweep=[0-9]+", label("codon-entropy"))


def test_the_labels_do_not_move_with_scheduling_settings() -> None:
    from dubanevics2026topological.analysis.produce_or_load import savename

    """``n_jobs`` and ``chunk_size`` are stripped before an alias is read."""
    stage = build_bootstrap_stage()
    inner = stage.settings["scientific"]
    noisy = {
        "scientific": {
            **inner,
            "config": {**inner["config"], "n_jobs": 99, "chunk_size": 1},
        }
    }
    assert savename(
        noisy, stage.name_keys, execution_keys=stage.execution_keys
    ) == savename(stage.settings, stage.name_keys, execution_keys=stage.execution_keys)


def test_the_factory_passes_the_registry_entry_to_the_stage(tmp_path: Path) -> None:
    for name, name_keys in stages.SAVENAME_KEYS.items():
        stage = stages.project_stage(name, settings={}, root=tmp_path)
        assert dict(stage.name_keys) == name_keys
        # A copy: changing one stage's aliases cannot change the registry.
        assert stage.name_keys is not stages.SAVENAME_KEYS[name]
        try:
            stage.name_keys["extra"] = "x"  # type: ignore[index]
        except TypeError:  # a read-only mapping is fine too
            pass
        assert "extra" not in stages.SAVENAME_KEYS[name]
    other = stages.project_stage("not-in-the-registry", settings={}, root=tmp_path)
    assert dict(other.name_keys) == {}


def test_the_factory_keeps_aliases_out_of_the_fingerprint(tmp_path: Path) -> None:
    labelled = stages.project_stage("analysis", settings={"a": 1}, root=tmp_path)
    assert labelled.name_keys
    bare = pol.Stage(
        "analysis",
        settings={"a": 1},
        root=tmp_path,
        execution_keys=stages.EXECUTION_KEYS,
        libraries=stages.STAGE_LIBRARIES,
    )
    assert stage_fingerprint(labelled) == stage_fingerprint(bare)


def test_the_factory_module_imports_only_the_helper_from_the_project() -> None:
    """The registry is plain data: it computes no key and loads no project code."""
    tree = ast.parse(FACTORY_MODULE.read_text(encoding="utf-8"))
    relative = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
    ]
    assert relative == ["analysis.produce_or_load"]
    absolute = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [
        name for name in absolute if name.startswith("dubanevics2026topological")
    ]


def _declared_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys |= _declared_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _declared_keys(item)
    return keys


@needs_raw_data
def test_no_real_stage_declares_the_factory_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the real stages hash, it is never ``stages.py`` (no cascade)."""
    for name, stage in build_registry_stages(monkeypatch).items():
        keys = _declared_keys(pol._declared(stage))
        offenders = [key for key in keys if key.rsplit("/", 1)[-1] == "stages.py"]
        assert offenders == [], name
