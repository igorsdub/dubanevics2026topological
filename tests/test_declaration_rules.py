"""Guard the configured rule projections and committed bootstrap identity."""

import ast
import copy

import pytest

from dubanevics2026topological.analysis import ROOT, load_settings
from dubanevics2026topological.results import RunContext
from dubanevics2026topological.threshold.functional_classification import (
    functional_classification_stage,
)
from dubanevics2026topological.threshold.functional_threshold import (
    FDR_KEYS,
    RULES,
    comparison_declaration,
    configured_declaration,
    declaration_identity,
    validate_functional_threshold_declaration,
)

# Fixed from the variant TOMLs. The classification table's digest is not part of a
# declaration: its integrity is data/processed/MANIFEST.tsv.
GOLDEN_IDENTITIES = {
    "threshold=reflected_fdr=by_q=0.05": {
        "rule": "fdr_fixed_cutoff",
        "decimals": 6,
        "cutoff": "median_reflected",
        "method": "by",
        "q": 0.05,
        "input_dir": "data/processed/classification/reflected",
    },
    "threshold=published_fdr=by_q=0.05": {
        "rule": "fdr_fixed_cutoff",
        "decimals": 16,
        "cutoff": "papkou_original",
        "method": "by",
        "q": 0.05,
        "input_dir": "data/processed/classification/published",
    },
    "mixture_quantile": {"rule": "mixture_quantile", "decimals": 6},
    "threshold=reflected_fdr=none": {"rule": "median_reflected", "decimals": 6},
    "threshold=published_fdr=none": {"rule": "fixed_cutoff", "decimals": 16, "value": -0.5077739708744209},
}


@pytest.mark.parametrize("variant", sorted(GOLDEN_IDENTITIES))
def test_variant_declaration_identity_golden(variant):
    settings = load_settings(context=RunContext(variant_name=variant, env={}))
    assert (
        declaration_identity(configured_declaration(settings))
        == GOLDEN_IDENTITIES[variant]
    )


def test_all_variants_have_golden_identities():
    assert {path.stem for path in (ROOT / "config/variants").glob("*.toml")} == set(
        GOLDEN_IDENTITIES
    )


@pytest.mark.parametrize(
    "variant",
    [None, *(path.stem for path in sorted((ROOT / "config/variants").glob("*.toml")))],
)
def test_shipped_declarations(variant):
    declaration = configured_declaration(
        load_settings(context=RunContext(variant_name=variant, env={}))
    )
    rule, decimals = validate_functional_threshold_declaration(declaration)
    assert comparison_declaration(declaration) == {
        "rule": "median_reflected",
        "decimals": decimals,
    }
    expected_keys = {"rule", "decimals"} | (
        set(FDR_KEYS) if rule == "fdr_fixed_cutoff" else {"value"} if rule == "fixed_cutoff" else set()
    )
    assert declaration_identity(declaration) == {
        key: declaration[key] for key in expected_keys
    }


def test_shipped_and_median_declarations_cover_all_rules():
    variants = (ROOT / "config/variants").glob("*.toml")
    settings = [load_settings(context=RunContext(env={}))]
    settings.extend(
        load_settings(context=RunContext(variant_name=path.stem, env={}))
        for path in variants
    )
    rules = {configured_declaration(item)["rule"] for item in settings}
    rules.add("median_reflected")
    assert rules == set(RULES)
    assert declaration_identity({"rule": "median_reflected", "decimals": 6}) == {
        "rule": "median_reflected",
        "decimals": 6,
    }


@pytest.mark.parametrize("key", FDR_KEYS)
def test_fdr_requires_every_field(key):
    declaration = dict(
        configured_declaration(load_settings(context=RunContext(env={})))
    )
    del declaration[key]
    with pytest.raises(ValueError, match="Missing FDR keys"):
        declaration_identity(declaration)


def test_classification_stage_settings_ignore_active_rule(tmp_path):
    default = load_settings(context=RunContext(env={}))
    mixture = load_settings(context=RunContext(variant_name="mixture_quantile", env={}))
    median = copy.deepcopy(default)
    median["analysis"]["clamped"]["candidate_functional_threshold"] = {
        "rule": "median_reflected",
        "decimals": configured_declaration(default)["decimals"],
    }
    stages = [
        functional_classification_stage(
            settings, growth_rates=tmp_path / "growth.parquet"
        )
        for settings in (default, mixture, median)
    ]
    assert stages[0].settings == stages[1].settings == stages[2].settings


def _declaration_rule_merges(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if not any(
            isinstance(key, ast.Constant) and key.value == "rule" for key in node.keys
        ):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if key is not None:
                continue
            if isinstance(value, ast.Name) and ("decl" in value.id.lower()):
                yield node.lineno
            elif isinstance(value, ast.Subscript):
                index = value.slice
                if (isinstance(index, ast.Name) and index.id == "DECLARATION_KEY") or (
                    isinstance(index, ast.Constant)
                    and index.value
                    in {"candidate_declaration", "candidate_functional_threshold"}
                ):
                    yield node.lineno


def test_declaration_merge_guard_detects_old_patterns():
    for source in (
        '{**declaration, "rule": MEDIAN_REFLECTED}',
        '{**scientific["clamped"][DECLARATION_KEY], "rule": MEDIAN_REFLECTED}',
        '{**settings["candidate_declaration"], "rule": "median_reflected"}',
    ):
        assert list(_declaration_rule_merges(ast.parse(source)))


def test_no_declaration_rule_merges():
    """Declarations must be projected, not relabeled while keeping FDR fields."""
    violations = []
    for root in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in root.rglob("*.py"):
            if any(
                part.startswith("pytest-of-") or part == ".ruff_cache"
                for part in path.parts
            ):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            violations.extend(
                f"{path.relative_to(ROOT)}:{line}"
                for line in _declaration_rule_merges(tree)
            )
    assert not violations
