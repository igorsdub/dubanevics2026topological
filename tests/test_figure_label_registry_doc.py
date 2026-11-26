"""The figure-label table of the design document, the figures and the registry agree.

``docs/manuscript-quantity-registry-design.md`` (section "Figure labels") lists one row
per printed label: figure identifier, label, registry key. Three things must not drift
apart:

1. the table and the keys each figure asks the registry for (two-way, per figure);
2. the table and the keys the registry stage emits;
3. the figures' label code and the rule that no figure formats a number itself.

The first check needs every figure built on synthetic inputs. Each converted figure
already has one test that builds it with the sentinel quantities of
``_figure_labels.py`` and asserts the exact key set; ``FIGURE_CASES`` names that test
and runs it, reading the keys from the views it created (``_figure_labels.CREATED``).
A converted figure without a case fails ``test_every_converted_figure_has_a_case``, so
a new figure cannot skip the check.
"""

from __future__ import annotations

import ast
import inspect
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import _figure_labels
import pytest

from dubanevics2026topological.analysis import ROOT
from dubanevics2026topological.figures.registry import FIGURE_SPECS

DOC = ROOT / "docs" / "manuscript-quantity-registry-design.md"
FIGURES = ROOT / "src" / "dubanevics2026topological" / "figures"
FIGURE_CONFIG = ROOT / "config" / "figures.toml"

ROW = re.compile(r"^\| `([a-z0-9_]+)` \| (.+) \| `([a-z0-9_.]+)` \|$")


def doc_label_table() -> list[tuple[str, str, str]]:
    """The (figure id, label, key) rows of the document's "Figure labels" section."""
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Figure labels")
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        match = ROW.match(line)
        if match:
            rows.append(match.groups())
    return rows


def doc_keys_by_figure() -> dict[str, set[str]]:
    keys: dict[str, set[str]] = defaultdict(set)
    for figure_id, _, key in doc_label_table():
        keys[figure_id].add(key)
    return dict(keys)


def test_the_doc_table_parses_and_has_no_duplicate_rows() -> None:
    rows = doc_label_table()
    assert len(rows) > 90  # the table is not silently cut short
    assert len({(figure_id, label) for figure_id, label, _ in rows}) == len(rows)
    registered = {spec.figure_id for spec in FIGURE_SPECS}
    assert {figure_id for figure_id, _, _ in rows} <= registered


def test_the_doc_labels_name_quantities_in_words() -> None:
    for figure_id, label, key in doc_label_table():
        assert not re.search(r"[α-ωΑ-Ω]", label + key), (figure_id, label)
        assert "cutoff" not in (label + key).lower(), (figure_id, label)
        # No single-letter word: a label says "radius", not "c".
        assert not re.search(r"(?<![\w-])[a-zA-Z](?![\w-])", label), (figure_id, label)
        words = [word for part in key.split(".") for word in part.split("_")]
        assert not any(len(word) == 1 and word.isalpha() for word in words), key


# -- figures versus the table ------------------------------------------------------

# figure id -> (test module, test function, keyword arguments, fixtures to run first).
# The function is the figure's own test that builds it on synthetic inputs with the
# sentinel quantities; fixtures are autouse fixtures of that module the test needs.
FIGURE_CASES: dict[str, tuple[str, str, dict[str, Any], tuple[str, ...]]] = {
    "full_tree_fitness_landmarks": (
        "test_fitness_landmarks_figure",
        "test_every_label_and_the_subtitle_come_from_the_registry",
        {},
        (),
    ),
    "full_join_tree": (
        "test_join_tree_figure",
        "test_join_tree_labels_come_from_the_registry",
        {},
        (),
    ),
    "tree_comparison": (
        "test_tree_comparison",
        "test_comparison_stacks_independent_wt_only_tree_panels",
        {},
        (),
    ),
    "linked_tree_persistence": (
        "test_split_tree_figure",
        "test_built_figure_variants_encode_one_full_tree_count_domain",
        {},
        (),
    ),
    "full_tree_persistence": (
        "test_full_tree_persistence_figure",
        "test_title_counts_are_the_registrys_and_say_ordinary_h0_pairs_not_peaks",
        {"variant": "vanilla"},
        ("_fake_topographer_tree",),
    ),
    "full_tree_persistence_vertical_histogram": (
        "test_full_tree_persistence_figure",
        "test_vertical_variant_labels_come_from_the_registry",
        {},
        ("_fake_topographer_tree",),
    ),
    "study_tree_persistence_metrics": (
        "test_study_metric_figure",
        "test_metric_figure_composes_three_shared_scale_tree_pair_rows",
        {},
        (),
    ),
    "study_tree_persistence_biology": (
        "test_biological_figure",
        "test_biology_figure_uses_maximum_only_overlays_and_no_count_legend",
        {},
        (),
    ),
    "bootstrap_codon_entropy": (
        "test_codon_entropy_plot",
        "test_build_draws_the_tree_over_three_sweeps",
        {},
        (),
    ),
    "candidate_tree_bootstrap_biology": (
        "test_bootstrap_biology_figure",
        "test_biology_has_two_rows_with_titles_counts_and_a_legend",
        {},
        (),
    ),
    "bootstrap_band_quantile_scan": (
        "test_bootstrap_band_quantile_scan_figure",
        "test_the_subtitle_prints_the_registry_numbers_as_given",
        {},
        (),
    ),
    "bootstrap_band_quantile_scan_seeds": (
        "test_bootstrap_band_quantile_scan_seeds_figure",
        "test_figure_draws_the_single_live_run_with_labels_from_the_registry",
        {},
        (),
    ),
    "bootstrap_global_stability": (
        "test_bootstrap_global_stability_figure",
        "test_stability_asks_the_registry_for_exactly_its_subtitle_numbers",
        {},
        (),
    ),
    **{
        f"{graph}_leaf_{kind}": (
            "test_leaf_figures",
            f"test_leaf_{kind}_subtitle_is_read_from_the_registry",
            {"graph": graph},
            (),
        )
        for graph in ("published", "candidate")
        for kind in ("persistence", "swarm")
    },
}


def converted_figures() -> set[str]:
    """Every registered figure whose spec declares the registry input."""
    return {
        spec.figure_id
        for spec in FIGURE_SPECS
        if "manuscript_quantities" in spec.inputs
    }


def test_every_converted_figure_has_a_case_and_a_doc_table() -> None:
    assert set(FIGURE_CASES) == converted_figures()
    assert set(doc_keys_by_figure()) == converted_figures()


def requested_keys(
    figure_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> set[str]:
    """Run the figure's own label test and return the keys its views were asked for."""
    module_name, function_name, kwargs, fixtures = FIGURE_CASES[figure_id]
    module = __import__(module_name)
    available = {"monkeypatch": monkeypatch, "tmp_path": tmp_path}
    for name in fixtures:
        fixture = getattr(module, name).__wrapped__
        fixture(**{p: available[p] for p in inspect.signature(fixture).parameters})
    function = getattr(module, function_name)
    wanted = inspect.signature(function).parameters
    arguments = {name: available[name] for name in wanted if name in available}
    arguments.update(kwargs)
    first = len(_figure_labels.CREATED)
    function(**arguments)
    views = _figure_labels.CREATED[first:]
    assert views, f"{module_name}.{function_name} built no sentinel quantities"
    return {key for view in views for key in view.requested_keys}


@pytest.mark.parametrize("figure_id", sorted(FIGURE_CASES))
def test_a_figure_asks_for_exactly_the_keys_the_doc_table_lists(
    figure_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asked = requested_keys(figure_id, monkeypatch, tmp_path)
    listed = doc_keys_by_figure()[figure_id]
    assert asked <= listed, (
        f"{figure_id} asks for keys the doc table does not list: "
        f"{sorted(asked - listed)}"
    )
    assert listed <= asked, (
        f"{figure_id}: the doc table lists keys the figure never asks for: "
        f"{sorted(listed - asked)}"
    )


# -- the table versus the registry stage ------------------------------------------


def test_every_doc_key_is_a_key_the_registry_stage_emits(tmp_path: Path) -> None:
    from test_manuscript_quantities import rows_of, write_upstream

    write_upstream(tmp_path, band=True)
    emitted = set(rows_of(tmp_path))
    listed = {key for _, _, key in doc_label_table()}
    assert not sorted(listed - emitted), (
        f"doc table keys the registry stage does not emit: {sorted(listed - emitted)}"
    )


# -- no figure formats a number itself --------------------------------------------

# Modules whose printed numbers are exempt (design document, "Exempt from the
# registry"): the notebook-only exploratory charts and the per-element data labels.
EXEMPT_MODULES = {"branch_decomposition_charts", "_count_labels"}
# (module, enclosing function): the exempt axis ticks, and the frozen reference run of
# the seed overlay, which is not a quantity of the current registry. Tooltips carry
# no format spec in any figure module.
EXEMPT_FUNCTIONS = {
    ("_charts", "horizontal_count_legend"),  # colour-bar count ticks
    ("leaf_swarm", "_overlap_note"),  # layout diagnostic, not a landscape quantity
    ("bootstrap_band_quantile_scan", "top_axis_ticks"),  # top-axis quantile ticks
    # Titles print a saved point value at the fixed three-decimal figure precision (and
    # the nominal q in general format), a visible-label choice the user decides; the
    # registry shows six decimals, so these do not take registry display text.
    ("reflected_fdr_trees", "build_reflected_fdr_trees_figure"),
    ("bootstrap_four_graphs", "panel_title"),
    # The peak-pair figures print statistics of the peak-pair-distances stage tables
    # (Spearman rho, Mantel p, diagonal fraction, mean and median excess), which are not
    # quantities of the manuscript registry, so they have no registry display text.
    ("peak_pair_figures", "_statistic_line"),
    ("peak_pair_figures", "_number"),
    ("peak_pair_figures", "_diagonal_subtitle"),
}
# A format spec that turns a number into text: thousands separators, fixed or general
# precision, percent. A hexadecimal colour (``02X``) is not a printed number.
NUMBER_SPEC = re.compile(r"[,%]|\.\d|[fgeE]$|^g")
TEMPLATE = re.compile(r"\{[A-Za-z_0-9.\[\]'\"]*:([^{}]*)\}")


class _FormatSpecs(ast.NodeVisitor):
    """Collect (line, enclosing function, spec) of every number-formatting spec."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.hits: list[tuple[int, str, str]] = []

    def _scope(self, node: ast.AST) -> None:
        self.stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def _record(self, line: int, spec: str) -> None:
        if NUMBER_SPEC.search(spec):
            self.hits.append((line, self.stack[-1] if self.stack else "", spec))

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        if node.format_spec is not None:
            spec = "".join(
                part.value
                for part in node.format_spec.values
                if isinstance(part, ast.Constant)
            )
            self._record(node.lineno, spec)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            for match in TEMPLATE.finditer(node.value):
                self._record(node.lineno, match.group(1))


def number_format_specs(path: Path) -> list[tuple[int, str, str]]:
    visitor = _FormatSpecs()
    visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
    return visitor.hits


def test_no_figure_module_formats_a_printed_number_with_a_format_spec() -> None:
    """Scope: f-string specs and ``{name:spec}`` template strings in figures/*.py.

    A figure prints the registry's display text as is, so a spec such as ``:,``,
    ``.3f``, ``.3g``, ``.1%`` or ``g`` in a label-building string is a number the
    figure formats itself. Exempt: the modules and functions listed above (ticks, the
    archived reference run, per-element data labels, notebook-only charts). Hexadecimal
    colour specs are not numbers. Format specs applied through other calls (``round``,
    ``str``) are outside this scan; the figure tests' sentinel values catch those.
    """
    offending = []
    for path in sorted(FIGURES.glob("*.py")):
        if path.stem in EXEMPT_MODULES:
            continue
        for line, function, spec in number_format_specs(path):
            if (path.stem, function) not in EXEMPT_FUNCTIONS:
                offending.append(
                    f"{path.name}:{line} in {function or '<module>'}: {spec!r}"
                )
    assert not offending, "figures format numbers themselves:\n" + "\n".join(offending)


def test_the_exempt_functions_exist_and_still_format_a_number() -> None:
    """An allowlist entry that no longer matches code is dead and must be removed."""
    seen = {
        (path.stem, function)
        for path in FIGURES.glob("*.py")
        for _, function, _ in number_format_specs(path)
    }
    assert EXEMPT_FUNCTIONS <= seen, sorted(EXEMPT_FUNCTIONS - seen)


def test_no_label_template_of_the_figure_config_has_a_number_format_spec() -> None:
    """Scope: ``{name:spec}`` inside any string of ``config/figures.toml``.

    Every template fills registry display text as is.
    """
    import tomllib

    config = tomllib.loads(FIGURE_CONFIG.read_text(encoding="utf-8"))
    offending: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str):
            for match in TEMPLATE.finditer(node):
                if NUMBER_SPEC.search(match.group(1)):
                    offending.append(f"{path}: {{...:{match.group(1)}}}")

    walk(config, "")
    assert not offending, "templates format numbers themselves:\n" + "\n".join(
        offending
    )
