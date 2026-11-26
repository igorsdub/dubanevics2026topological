"""Shared fixtures."""

from __future__ import annotations

import os

# Test defaults must not depend on paths or variants exported by a batch job.
# Clear them before importing the package, which caches some values at import.
_ENV_OVERRIDES = (
    "FITNESSTREE_OUTPUT_ROOT",
    "FITNESSTREE_ANALYSIS_VARIANT",
    "FITNESSTREE_RAW_DIR",
    "FITNESSTREE_PROCESSED_ROOT",
    "FITNESSTREE_BOOTSTRAP_CHUNK_DIR",
    "FITNESSTREE_FONT_DIR",
    "FITNESSTREE_FDR_POOLED_ROOT",
)
for _name in _ENV_OVERRIDES:
    os.environ.pop(_name, None)

import pytest

from dubanevics2026topological.simplification import bootstrap_chunks


@pytest.fixture(autouse=True)
def _scrub_inherited_environment(monkeypatch):
    """Give each test clean defaults while allowing explicit overrides."""
    for name in _ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def _synthetic_classification_root(tmp_path_factory):
    """A root holding both BY classification folders of ``data/processed`` as tiny tables.

    The shipped configuration names ``data/processed/classification/<variant>``, which a
    clean checkout does not have (``just all`` or ``just fetch-data`` writes it). Tests
    that resolve the shipped selection read this stand-in: the wild type is the one
    functional sequence, so the selection resolves; nothing here claims real counts.
    """
    import json
    import tomllib
    from pathlib import Path

    import polars as pl

    from dubanevics2026topological.threshold.fdr_selection import table_name

    repo = Path(__file__).resolve().parents[1]
    root = tmp_path_factory.mktemp("synthetic-processed")
    with (repo / "config" / "numbers.toml").open("rb") as stream:
        reflected = tomllib.load(stream)["thresholds"]["reflected_functional_threshold"]
    cutoffs = {"median_reflected": reflected, "papkou_original": -0.5077739708744209}
    wild_type = "AAGGAAATG"
    wt_file = repo / "data" / "raw" / "papkou2023_fitness_wt.parquet"
    if wt_file.is_file():
        wild_type = str(pl.read_parquet(wt_file)[0, "dna_from"])
    records = [
        {"name": name, "applied_threshold": value, "source_precision_threshold": value}
        for name, value in cutoffs.items()
    ]
    for variant in ("reflected", "published"):
        directory = root / "data" / "processed" / "classification" / variant
        directory.mkdir(parents=True)
        (directory / "provenance.json").write_text(json.dumps({"cutoffs": records}))
        for name, value in cutoffs.items():
            pl.DataFrame(
                {
                    "dna_sequence": [wild_type, "CTCTGCTTG"],
                    "s": [0.5, -0.229093],
                    "fdr_class": ["functional", "uncertain"],
                }
            ).write_parquet(directory / table_name(value, name, "by", 0.05))
    return root


@pytest.fixture(autouse=True)
def _shipped_classification(
    _synthetic_classification_root, monkeypatch, _scrub_inherited_environment
):
    """Resolve the shipped ``data/processed/...`` ``input_dir`` against the synthetic root.

    Through ``FITNESSTREE_PROCESSED_ROOT``, so a subprocess a test starts sees it too.
    """
    monkeypatch.setenv(
        "FITNESSTREE_PROCESSED_ROOT", str(_synthetic_classification_root / "data" / "processed")
    )


@pytest.fixture(autouse=True)
def _isolated_bootstrap_chunks(tmp_path, monkeypatch, _scrub_inherited_environment):
    """Never let a test read or write the user's saved bootstrap chunks."""
    monkeypatch.setenv(bootstrap_chunks.ENV_CHUNK_DIR, str(tmp_path / "chunks"))


@pytest.fixture(autouse=True)
def _build_source_under_patched_root(monkeypatch):
    """Stage code files must live under ``build.ROOT``; tests often patch ROOT.

    When a test points ``build.ROOT`` at a temporary directory, mirror the real
    build modules there on demand so the stage can hash them.
    """
    import shutil
    from pathlib import Path

    from dubanevics2026topological import build

    real_dir = Path(build.__file__).resolve().parent
    original = build._build_sources

    def sources() -> tuple[Path, ...]:
        paths = original()
        for path in paths:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(real_dir / path.name, path)
        return paths

    monkeypatch.setattr(build, "_build_sources", sources)
