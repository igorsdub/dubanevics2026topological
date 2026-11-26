"""Fingerprint inputs are keyed by their unresolved path relative to ROOT.

``data/raw/*`` may be symlinks to storage outside the repository. Resolving them
before keying would put them outside ``ROOT``, so every fingerprint must key by the
path as written and use the resolved path only for hashing and de-duplication.
"""

from __future__ import annotations

from dubanevics2026topological.rawdata.value_digest import value_digest

import shutil
import types
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import dubanevics2026topological.analysis.candidate_tree_contrast as contrast_module
import dubanevics2026topological.analysis.clamped_cache as cache
import dubanevics2026topological.analysis.peak_uncertainty as peak_analysis
import dubanevics2026topological.analysis.sequence_growth_rates as growth_module
import dubanevics2026topological.simplification.bootstrap_smoothed_tree as bootstrap_module
from dubanevics2026topological import analysis
from dubanevics2026topological.analysis import produce_or_load as pol
from dubanevics2026topological.analysis import source_key
from dubanevics2026topological.analysis.barseq_inputs import BarseqExport

BOOTSTRAP_CONFIG = {
    "n_replicates": 20,
    "seed": 7,
    "band_quantile": 0.95,
    "n_jobs": 1,
}
CONTRAST_CONFIG = {
    "alpha": 0.05,
    "elapsed_time": 1.0,
    "time_unit": "generations",
    "correction": "benjamini_hochberg",
    "family": "candidate_split_tree_edges",
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A tmp ROOT whose ``data/raw`` files are symlinks to files outside it."""
    root = tmp_path / "root"
    storage = tmp_path / "storage"
    (root / "data/raw").mkdir(parents=True)
    (root / "config").mkdir()
    storage.mkdir()
    names = ("wt.parquet", "diff.parquet", "graph.graphml", "peaks.csv")
    for name in names:
        (storage / name).write_text(f"{name}-v1", encoding="utf-8")
        (root / "data/raw" / name).symlink_to(storage / name)
    for name, text in (
        ("config/analysis.toml", "settings"),
        ("pyproject.toml", "project"),
        ("uv.lock", "lock"),
    ):
        (root / name).write_text(text, encoding="utf-8")
    for module in (analysis, cache, peak_analysis, contrast_module):
        monkeypatch.setattr(module, "ROOT", root)
    for module in (growth_module, bootstrap_module):
        monkeypatch.setattr(module, "ROOT", root)
    for module in (cache, peak_analysis, contrast_module):
        monkeypatch.setattr(
            module,
            "candidate_selection_fingerprint",
            lambda _, **kwargs: {
                "rounded_threshold": -0.25,
                "source": {"rule": "median_reflected"},
            },
        )
    return {"root": root, "storage": storage}


def _copy_module(
    module: Any, repo: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Place the module's source files under the tmp ROOT and point it at them."""
    source = (
        Path(module.__file__).parent
        if module is bootstrap_module
        else Path(contrast_module.__file__).parent
    )
    target = repo["root"] / (
        "src/simplification" if module is bootstrap_module else "src/analysis"
    )
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.py"):
        shutil.copy(path, target / path.name)
    if module is bootstrap_module:
        analysis_source = source.parent / "analysis"
        analysis_target = repo["root"] / "src/analysis"
        analysis_target.mkdir(parents=True, exist_ok=True)
        for name in (
            "candidate_tree_contrast.py",
            "barseq_inputs.py",
        ):
            shutil.copy(analysis_source / name, analysis_target / name)
    threshold_target = repo["root"] / "src/threshold"
    threshold_target.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        source.parent / "threshold" / "functional_threshold.py",
        threshold_target / "functional_threshold.py",
    )
    landscape_source = source.parent / "landscape"
    landscape_target = repo["root"] / "src/landscape"
    landscape_target.mkdir(parents=True, exist_ok=True)
    for name in ("selection.py", "selected_landscape.py", "full_population.py"):
        shutil.copy(landscape_source / name, landscape_target / name)
    monkeypatch.setattr(module, "__file__", str(target / Path(module.__file__).name))


def _settings(**extra: Any) -> dict[str, Any]:
    return {
        "analysis": {
            "dataset": {
                "fitness_wt": "data/raw/wt.parquet",
                "fitness_diff": "data/raw/diff.parquet",
            },
            "clamped": {
                "graphml": "data/raw/graph.graphml",
                "peaks": "data/raw/peaks.csv",
                "published_functional_threshold": -0.5,
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                },
                "reference_topographer_commit": "abc123",
            },
            "codon_entropy": {"stop_codons": ["TAA", "TAG", "TGA"]},
            "tree": {"kind": "split"},
            "persistence": {"kind": "extended_h0"},
            "candidate_tree_contrast": CONTRAST_CONFIG,
            "sequence_growth_rates": CONTRAST_CONFIG,
            "bootstrap_smoothed_tree": BOOTSTRAP_CONFIG,
            **extra,
        },
        "figures": {},
    }


def _assert_no_external_label(files: dict[str, str], name: str) -> None:
    assert f"data/raw/{name}" in files
    assert not [key for key in files if key.startswith("topographer:") and name in key]


def _package_keys(files: dict[str, str], package: str) -> set[str]:
    return {key for key in files if key.startswith(f"{package}:")}


def _local_keys(files: dict[str, str]) -> set[str]:
    """Return the in-repository keys: no BarSeq export and no package source."""
    return {
        key
        for key in files
        if not key.startswith(("barseq_export/", "barseqfit:", "topographer:"))
    }


def test_source_key_is_relative_and_unresolved(repo: dict[str, Path]) -> None:
    link = repo["root"] / "data/raw/wt.parquet"
    assert link.resolve().is_relative_to(repo["storage"].resolve())
    assert source_key(link) == "data/raw/wt.parquet"
    assert source_key(str(link)) == "data/raw/wt.parquet"
    assert source_key(Path("data/raw/wt.parquet")) == "data/raw/wt.parquet"
    assert source_key(repo["root"] / "data/../uv.lock") == "uv.lock"


def test_source_key_rejects_paths_outside_root(repo: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="outside"):
        source_key(repo["storage"] / "wt.parquet")
    with pytest.raises(ValueError, match="outside"):
        source_key(repo["root"] / "../storage/wt.parquet")
    with pytest.raises(ValueError, match="outside"):
        source_key(repo["root"].parent / "root-sibling" / "x")


def test_clamped_cache_fingerprint_keys_symlinked_inputs_by_relative_path(
    repo: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repo["root"]
    monkeypatch.setattr(cache, "scientific_code_paths", list)
    package_root = root / "src/dubanevics2026topological"
    analysis_dir = package_root / "analysis"
    landscape_dir = package_root / "landscape"
    threshold_dir = package_root / "threshold"
    tables_dir = package_root / "tables"
    for directory in (analysis_dir, landscape_dir, threshold_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for directory, names in (
        (
            analysis_dir,
            (
                "clamped_cache.py",
                "describe.py",
                "graph_summary.py",
                "__init__.py",
            ),
        ),
        (
            landscape_dir,
            (
                "selected_landscape.py",
                "selection.py",
                "full_population.py",
            ),
        ),
        (threshold_dir, ("functional_threshold.py",)),
        (tables_dir, ("__init__.py",)),
    ):
        for name in names:
            (directory / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(cache, "__file__", str(analysis_dir / "clamped_cache.py"))
    provenance = root / "output/analysis/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"fingerprint": "baseline"}', encoding="utf-8")
    monkeypatch.setattr(
        cache, "analysis_provenance_path", lambda context=None: provenance
    )
    settings = _settings()
    clamped = settings["analysis"]["clamped"]
    clamped["graphml_value_sha256"] = value_digest(root / "data/raw/graph.graphml")
    clamped["peaks_value_sha256"] = value_digest(root / "data/raw/peaks.csv")

    declared = pol._declared(cache.comparison_stage(settings))  # no outside-root error

    assert set(declared["inputs"]) == {"data/raw/wt.parquet", "data/raw/diff.parquet"}
    _assert_no_external_label(declared["inputs"], "wt.parquet")
    _assert_no_external_label(declared["inputs"], "diff.parquet")
    # The reference files are verified against the configured sha256 values, and a
    # mismatch names the file by its in-repository key, not its storage path.
    (repo["storage"] / "graph.graphml").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="'data/raw/graph.graphml'.*expected"):
        cache._fingerprint(settings)


def test_scientific_fingerprint_does_not_label_symlinked_inputs_external(
    repo: dict[str, Path],
) -> None:
    declared = pol._declared(analysis.analysis_stage(_settings()))
    sources = declared["code"]
    _assert_no_external_label(declared["inputs"], "wt.parquet")
    _assert_no_external_label(declared["inputs"], "diff.parquet")
    assert "config/analysis.toml" not in sources
    assert "config/analysis.toml" not in declared["inputs"]
    # Topographer's install path is not part of the key. Its version is a label.
    assert not any(str(key).startswith("topographer:") for key in sources)


def test_peak_uncertainty_fingerprint_does_not_label_symlinked_inputs_external(
    repo: dict[str, Path],
) -> None:
    declared = pol._declared(peak_analysis.peak_stage(_settings()))
    _assert_no_external_label(declared["inputs"], "wt.parquet")
    _assert_no_external_label(declared["inputs"], "diff.parquet")


def test_dedupe_by_resolved_keeps_first_spelling(repo: dict[str, Path]) -> None:
    link = repo["root"] / "data/raw/wt.parquet"
    real = repo["storage"] / "wt.parquet"
    assert analysis.dedupe_by_resolved([link, real, link]) == [link]
    assert analysis.dedupe_by_resolved([real, link]) == [real]


def test_candidate_contrast_fingerprint_keys_are_relative(
    repo: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_module(contrast_module, repo, monkeypatch)
    files = contrast_module._fingerprint_without_export(_settings())["files"]
    _assert_no_external_label(files, "wt.parquet")
    assert set(files) == {
        "data/raw/wt.parquet",
        "data/raw/diff.parquet",
        "src/analysis/candidate_tree_contrast.py",
        "src/landscape/selection.py",
        "src/landscape/selected_landscape.py",
        "src/landscape/full_population.py",
        "src/threshold/functional_threshold.py",
        "src/analysis/barseq_inputs.py",
        "uv.lock",
    }


def test_sequence_growth_rates_fingerprint_keys_are_relative(
    repo: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_module(growth_module, repo, monkeypatch)
    export = BarseqExport(*(pl.DataFrame(),) * 3, source_dir=repo["storage"])
    for name in ("counts", "samples", "sequences"):
        (repo["storage"] / f"{name}.parquet").write_bytes(name.encode())
    files = growth_module._fingerprint(_settings(), export)["files"]
    assert _local_keys(files) == {
        "data/raw/wt.parquet",
        "src/analysis/sequence_growth_rates.py",
        "src/analysis/barseq_inputs.py",
        "uv.lock",
    }
    # The package the stage computes with is hashed too, under its own label;
    # topographer is not used by this stage.
    assert _package_keys(files, "barseqfit")
    assert not _package_keys(files, "topographer")


def test_bootstrap_fingerprint_keys_are_relative(
    repo: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_module(bootstrap_module, repo, monkeypatch)
    files = bootstrap_module._fingerprint(_settings(), {})["files"]
    assert _local_keys(files) == {
        "data/raw/wt.parquet",
        "data/raw/diff.parquet",
        "src/simplification/bootstrap_smoothed_tree.py",
        "src/simplification/bootstrap_aggregation.py",
        "src/simplification/bootstrap_chunks.py",
        "src/simplification/checkpoint_identity.py",
        "src/analysis/candidate_tree_contrast.py",
        "src/landscape/selection.py",
        "src/landscape/selected_landscape.py",
        "src/landscape/full_population.py",
        "src/simplification/selected_leaf_simplification.py",
        "uv.lock",
    }
    assert _package_keys(files, "topographer")


def test_package_source_hashes_follow_the_effective_package_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit to an (editable) package's source changes its hashes."""
    monkeypatch.setattr(analysis, "ROOT", tmp_path / "project-root")
    package = tmp_path / "editable" / "fakepkg"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (package / "sub" / "core.py").write_text("y = 1\n", encoding="utf-8")
    (package / "data.txt").write_text("not python\n", encoding="utf-8")
    module = types.ModuleType("fakepkg")
    module.__file__ = str(package / "__init__.py")

    before = analysis.package_source_hashes(module)
    assert {Path(key).name for key in before} == {"__init__.py", "core.py"}
    assert all(key.startswith("fakepkg:") for key in before)
    assert analysis.package_source_hashes(module) == before

    (package / "sub" / "core.py").write_text("y = 2\n", encoding="utf-8")
    after = analysis.package_source_hashes(module)
    assert after != before
    changed = {key for key in after if after[key] != before[key]}
    assert {Path(key).name for key in changed} == {"core.py"}


def test_stage_fingerprints_change_with_package_sources(
    repo: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    export = BarseqExport(*(pl.DataFrame(),) * 3, source_dir=repo["storage"])
    for name in ("counts", "samples", "sequences"):
        (repo["storage"] / f"{name}.parquet").write_bytes(name.encode())
    _copy_module(growth_module, repo, monkeypatch)
    _copy_module(bootstrap_module, repo, monkeypatch)
    growth_before = growth_module._fingerprint(_settings(), export)
    bootstrap_before = bootstrap_module._fingerprint(_settings(), {})

    monkeypatch.setattr(
        growth_module,
        "package_source_hashes",
        lambda *packages: {f"{p.__name__}:edited": "new" for p in packages},
    )
    monkeypatch.setattr(
        bootstrap_module,
        "package_source_hashes",
        lambda *packages: {f"{p.__name__}:edited": "new" for p in packages},
    )
    assert growth_module._fingerprint(_settings(), export) != growth_before
    assert bootstrap_module._fingerprint(_settings(), {}) != bootstrap_before
