"""Manuscript presentation tables sliced from the landscape summary."""

from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

from dubanevics2026topological.analysis.describe import SUMMARY_COLUMNS
from dubanevics2026topological.tables import presentation_tables as tables

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_ROWS = {
    "Functional threshold": -0.25,
    "Number of nodes": 100,
    "Number of edges": 200,
    "Number of functional sequences": 40,
    "Number of non-functional sequences": 50,
    "Number of uncertain sequences": "not applicable",
    "Peak count": 5,
    "Minimum fitness": -1.0,
    "Maximum fitness": 1.0,
    "Mean fitness": 0.0,
    "Median fitness": 0.0,
    "Extra ignored metric": 99,
}


def _values(index: int) -> dict:
    # Band columns (4, 6) share graph-level metrics with their nominal pair (3, 5).
    rows = dict(BASE_ROWS)
    rows["Number of nodes"] = 100 + (index + 1) // 2
    rows["Peak count"] = 5 + index
    if index >= 3:
        rows["Number of uncertain sequences"] = 10 + ((index + 1) // 2)
    return rows


def _store(tmp_path: Path) -> dict:
    store = {}
    for index, name in enumerate(SUMMARY_COLUMNS):
        values = _values(index)
        directory = tmp_path / f"p{index}"
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps([{"metric": k, "value": v} for k, v in values.items()])
        )
        label = f"landscape_summary__fake{index}"
        store[label] = SimpleNamespace(
            label=label,
            report=SimpleNamespace(status="full", byte_integrity=True),
            manifest=SimpleNamespace(
                scientific_id_sha256=hashlib.sha256(label.encode()).hexdigest(),
                scientific_identity={"column": name},
            ),
            artifacts={"summary.json": directory / "summary.json"},
        )
    return store


def _labels(store: dict) -> dict:
    return dict(zip(SUMMARY_COLUMNS, store, strict=True))


def test_exports_three_presentation_views(tmp_path: Path) -> None:
    store = _store(tmp_path)
    paths = tables.export_presentation_tables(
        None,
        _labels(store),
        "papkou2023",
        tmp_path / "out",
        loader=lambda kind, label: store[label],
    )
    stems = {
        "full_vs_published",
        "candidate_corrections",
        "published_simplification",
    }
    assert {p.stem.removeprefix("papkou2023_") for p in paths if p.suffix == ".csv"} == stems
    assert all(p.is_file() for p in paths)


def test_table2_columns_are_nominal_fdr_bootstrap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tables.export_presentation_tables(
        None,
        _labels(store),
        "papkou2023",
        tmp_path / "out",
        loader=lambda kind, label: store[label],
    )
    with (tmp_path / "out" / "papkou2023_candidate_corrections.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["Metric", "Nominal", "FDR", "Bootstrap"]
        rows = {row["Metric"]: row for row in reader}
    # Median-reflected point-estimate index 2, BY nominal index 5, band index 6.
    # Graph metrics match between FDR (5) and Bootstrap (6); Peak count differs.
    assert rows["Number of nodes"]["Nominal"] == "101"
    assert rows["Number of nodes"]["FDR"] == "103"
    assert rows["Number of nodes"]["Bootstrap"] == ""  # blanked graph-level band cell
    assert rows["Peak count"]["Bootstrap"] == "11"


def test_full_published_candidate_columns(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tables.export_presentation_tables(
        None,
        _labels(store),
        "papkou2023",
        tmp_path / "out",
        loader=lambda kind, label: store[label],
    )
    with (tmp_path / "out" / "papkou2023_full_vs_published.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["Metric", "Full", "Papkou", "Topographer + FDR"]
        rows = {row["Metric"]: row for row in reader}
    assert rows["Number of nodes"]["Topographer + FDR"] == "103"
    sidecar = json.loads(
        (tmp_path / "out" / "papkou2023_full_vs_published.json").read_text()
    )
    assert sidecar["columns"][2]["source"] == "Median-reflected BY (nominal)"
    assert rows["Valley count"] == {
        "Metric": "Valley count", "Full": "", "Papkou": "", "Topographer + FDR": ""
    }
    assert "Papkou" in sidecar["valley_source"]["gaps"]
    assert "Topographer + FDR" in sidecar["valley_source"]["gaps"]


def test_full_valley_source_and_lineage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tables.export_presentation_tables(
        None,
        _labels(store),
        "papkou2023",
        tmp_path / "out",
        loader=lambda kind, label: store[label],
        valley_source=("17", "registry-digest"),
    )
    with (tmp_path / "out" / "papkou2023_full_vs_published.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {row["Metric"]: row for row in csv.DictReader(handle)}
    assert rows["Valley count"]["Full"] == "17"
    assert rows["Valley count"]["Papkou"] == ""
    assert rows["Valley count"]["Topographer + FDR"] == ""
    sidecar = json.loads(
        (tmp_path / "out" / "papkou2023_full_vs_published.json").read_text()
    )
    assert sidecar["valley_source"]["registry_fingerprint"] == "registry-digest"


def test_numbers_toml_has_reported_landmarks() -> None:
    path = REPO_ROOT / "config" / "numbers.toml"
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert data["thresholds"]["published_functional_threshold"] == -0.507774
    assert data["thresholds"]["reflected_functional_threshold"] == -0.253159
    assert data["full_graph_landmarks"]["median"] == -0.7139811714
    assert data["fdr"]["cutoff"] == "median_reflected"
    assert data["fdr"]["q"] == 0.05
    assert data["bootstrap"]["n_replicates"] == 10000


def test_analysis_toml_default_is_fdr_fixed_cutoff() -> None:
    path = REPO_ROOT / "config" / "analysis.toml"
    with path.open("rb") as handle:
        analysis = tomllib.load(handle)
    declaration = analysis["clamped"]["candidate_functional_threshold"]
    assert declaration["rule"] == "fdr_fixed_cutoff"
    assert declaration["cutoff"] == "median_reflected"
    assert declaration["method"] == "by"
    assert declaration["q"] == 0.05


def test_numbers_toml_matches_analysis_fdr_pin() -> None:
    with (REPO_ROOT / "config" / "analysis.toml").open("rb") as handle:
        analysis = tomllib.load(handle)
    with (REPO_ROOT / "config" / "numbers.toml").open("rb") as handle:
        numbers = tomllib.load(handle)
    declaration = analysis["clamped"]["candidate_functional_threshold"]
    assert numbers["thresholds"]["published_functional_threshold"] == (
        analysis["clamped"]["published_functional_threshold"]
    )
    assert numbers["fdr"]["cutoff"] == declaration["cutoff"]
    assert numbers["fdr"]["method"] == declaration["method"]
    assert numbers["fdr"]["q"] == declaration["q"]
    assert numbers["fdr"]["snapshot_dir"] == declaration["input_dir"]
    assert "table_sha256" not in declaration
    assert numbers["bootstrap"]["n_replicates"] == (
        analysis["bootstrap_smoothed_tree"]["n_replicates"]
    )
    assert numbers["bootstrap"]["seed"] == analysis["bootstrap_smoothed_tree"]["seed"]
    assert numbers["bootstrap"]["band_quantile"] == (
        analysis["bootstrap_smoothed_tree"]["band_quantile"]
    )


def test_bootstrap_notes_say_tree_level_only_not_matching_graph_cells(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    tables.export_presentation_tables(
        None,
        _labels(store),
        "papkou2023",
        tmp_path / "out",
        loader=lambda kind, label: store[label],
    )
    for stem in ("candidate_corrections", "published_simplification"):
        text = (tmp_path / "out" / f"papkou2023_{stem}.md").read_text(encoding="utf-8")
        assert "tree-level rows only" in text
        assert "Graph-level rows for Bootstrap match FDR" not in text
