"""The justfile's dependency graph: `all` computes, collects and renders; `figures` only renders."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)((?: [^:\n]*)?):(?!=)\s*(.*)$")


def _recipes() -> dict[str, dict[str, object]]:
    """``{name: {"deps": [...], "body": [...]}}`` of every recipe in the justfile."""
    recipes: dict[str, dict[str, object]] = {}
    current = None
    for line in (ROOT / "justfile").read_text(encoding="utf-8").splitlines():
        if line.startswith(("#", "set ")) or not line.strip():
            continue
        if line.startswith((" ", "\t")):
            if current is not None:
                recipes[current]["body"].append(line.strip())
            continue
        match = HEADER.match(line)
        if match is None:
            current = None  # an assignment such as ``python := ...``
            continue
        name, _, dependencies = match.groups()
        deps = [
            token.strip("()").split()[0]
            for token in re.findall(r"\([^)]*\)|\S+", dependencies)
            if token.strip("()")
        ]
        recipes[name] = {"deps": deps, "body": []}
        current = name
    return recipes


def _reaches(recipes: dict, start: str, target: str) -> bool:
    seen: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name == target:
            return True
        if name in seen or name not in recipes:
            continue
        seen.add(name)
        stack.extend(recipes[name]["deps"])
    return False


def _order(recipes: dict, start: str) -> list[str]:
    """The order just runs the recipes of ``start`` in: dependencies first, once each."""
    done: list[str] = []

    def visit(name: str) -> None:
        if name in done or name not in recipes:
            return
        for dependency in recipes[name]["deps"]:
            visit(dependency)
        done.append(name)

    visit(start)
    return done


def test_figures_only_renders_from_processed_data() -> None:
    figures = _recipes()["figures"]
    assert figures["deps"] == ["check-manifest"]
    body = " ".join(figures["body"])
    for forbidden in ("compute", "all", "bootstrap-smoothed-tree", "sequence-growth-rates"):
        assert not re.search(rf"\b{forbidden}\b", body.replace("just_executable()", "")), forbidden
    assert figures["body"][0].endswith("processed-restore")
    assert any("build figures" in line for line in figures["body"])
    assert any(line.endswith("name-outputs") for line in figures["body"])
    assert any(line.endswith("landscape-summary-table") for line in figures["body"])


def test_no_recipe_is_named_hpc() -> None:
    names = _recipes()
    assert not [name for name in names if "hpc" in name]
    for removed in ("bootstrap-restore", "migrate-outputs", "preflight"):
        assert removed not in names


def test_all_is_compute_then_manifest_then_figures() -> None:
    assert _recipes()["all"]["deps"] == ["compute", "processed-manifest", "figures"]


def test_all_reaches_the_papkou_download_before_any_stage() -> None:
    recipes = _recipes()
    assert _reaches(recipes, "all", "fetch-papkou2023")
    order = _order(recipes, "compute")
    for earlier, later in (
        ("fetch-papkou2023", "make-raw"),
        ("make-raw", "barseq-export"),
        ("barseq-export", "sequence-growth-rates"),
        ("sequence-growth-rates", "functional-classification"),
        ("functional-classification", "classification-probability-scans"),
        ("classification-probability-scans", "landscape-products"),
        ("landscape-products", "leaves-product"),
        ("bootstrap-draws", "leaves-product"),
        ("leaves-product", "tree-layout-products"),
        ("tree-layout-products", "tables"),
    ):
        assert order.index(earlier) < order.index(later), (earlier, later)


def test_the_stage_chain_has_every_stage_of_the_design() -> None:
    recipes = _recipes()
    for stage in (
        "comparison-analysis",
        "four-graph-comparisons",
        "landscape-products",
        "bootstrap-draws",
        "bootstrap-band-quantile-scan-product",
        "leaves-product",
        "tree-layout-products",
        "landscape-summary-product",
        "ridges-product",
        "ridge-entropy-product",
        "merge-flow-product",
        "peak-panels-product",
        "pooled-scatter-product",
        "threshold-diagnostics-product",
        "codon-entropy-analysis",
        "bootstrap-codon-entropy-product",
        "graph-tree-persistence-product",
        "candidate-edge-analysis",
        "component-analysis",
        "leading-saddle-band",
        "graph-descriptive-statistics",
        "tables",
    ):
        assert _reaches(recipes, "compute", stage), stage
    assert not _reaches(recipes, "figures", "compute")


def test_the_manifest_is_written_after_the_collect_and_checked_before_rendering() -> None:
    recipes = _recipes()
    assert recipes["processed-manifest"]["deps"] == ["processed-collect"]
    assert any(line.endswith("processed write") for line in recipes["processed-manifest"]["body"])
    assert any(line.endswith("processed check") for line in recipes["check-manifest"]["body"])
    assert any(line.endswith("check-manifest") for line in recipes["fetch-data"]["body"])


def test_both_by_bootstraps_and_scans_are_stages() -> None:
    recipes = _recipes()
    assert recipes["bootstrap-draws"]["deps"] == [
        "bootstrap-smoothed-tree",
        "bootstrap-smoothed-tree-by",
    ]
    assert recipes["bootstrap-band-scans"]["deps"] == [
        "bootstrap-band-quantile-scan",
        "bootstrap-band-quantile-scan-by",
    ]
    assert recipes["bootstrap-band-quantile-scan-product"]["deps"] == ["bootstrap-band-scans"]


def test_barseq_export_runs_export_papkou_in_the_barseqfit_checkout() -> None:
    body = " ".join(_recipes()["barseq-export"]["body"])
    assert "just export-papkou" in body and "barseqfit" in body
    text = (ROOT / "justfile").read_text(encoding="utf-8")
    assert "export PAPKOU2023_BARSEQ_EXPORT" in text


@pytest.mark.parametrize("recipe", ["leaves-product", "landscape-products", "sequence-growth-rates"])
def test_stages_that_later_stages_read_from_processed_collect_first(recipe: str) -> None:
    assert any(line.endswith("processed-collect") for line in _recipes()[recipe]["body"])


def test_dotenv_is_loaded_before_the_exports() -> None:
    lines = (ROOT / "justfile").read_text(encoding="utf-8").splitlines()
    load = lines.index("set dotenv-load")
    export = next(i for i, line in enumerate(lines) if line.startswith("export "))
    assert load < export
    # the barseqfit folder is a fallback only, never a hard default over .env
    assert 'env("PAPKOU2023_BARSEQ_EXPORT", "")' in "\n".join(lines)


def test_env_example_names_both_variables() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "PAPKOU2023_BARSEQ_EXPORT=" in text and "PAPKOU2023RUGGED_DATA=" in text


def test_four_graph_comparisons_need_the_selection() -> None:
    assert "selection" in _recipes()["four-graph-comparisons"]["deps"]


def test_a_variant_bootstrap_has_its_own_growth_rate_checkpoint_first() -> None:
    """Each output root keeps its own growth-rate checkpoint; the variant bootstrap reads its root's."""
    recipes = _recipes()
    assert "sequence-growth-rates-by" in recipes["bootstrap-smoothed-tree-by"]["deps"]
    body = " ".join(recipes["sequence-growth-rates-by"]["body"])
    assert "FITNESSTREE_OUTPUT_ROOT=output/variants/{{variant}}" in body
    assert "FITNESSTREE_ANALYSIS_VARIANT={{variant}}" in body
    assert body.endswith("build sequence-growth-rates")
