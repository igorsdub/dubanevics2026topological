"""The seven-column landscape summary table: a composite of seven saved products."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubanevics2026topological.results import processed

from dubanevics2026topological.analysis.describe import SUMMARY_COLUMNS
from dubanevics2026topological.build import CONTEXT_AWARE_COMMANDS
from dubanevics2026topological.tables import landscape_summary_table as table

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROWS = {
    "Number of nodes": 10,
    "Functional threshold": 0.5,
    "Full library functional sequences (%)": 12.3456,
}
TREE_ROWS = {"Peak count": 3, "Synonymous leaves (%)": 33.3333}


def _values(index: int, name: str) -> dict:
    graph = {**GRAPH_ROWS, "Number of nodes": 10 + (index + 1) // 2}
    tree = {**TREE_ROWS, "Peak count": 3 + index}
    return {**graph, **tree}


def _store(tmp_path: Path, *, band_nodes: int | None = None) -> dict:
    store = {}
    for index, name in enumerate(SUMMARY_COLUMNS):
        values = _values(index, name)
        if band_nodes is not None and index == 4:
            values["Number of nodes"] = band_nodes
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


def _export(tmp_path, store, **kwargs):
    return table.export_summary_table(
        None,
        _labels(store),
        "toy",
        tmp_path / "out",
        loader=lambda kind, label: store[label],
        **kwargs,
    )


def test_exports_three_tables_and_sidecar(tmp_path):
    paths = _export(tmp_path, _store(tmp_path))
    assert [p.suffix for p in paths] == [".csv", ".html", ".md", ".json"]
    assert all(p.is_file() for p in paths)


def test_nominal_columns_keep_saved_values_and_band_keeps_tree_rows_only(tmp_path):
    paths = _export(tmp_path, _store(tmp_path))
    with paths[0].open(newline="") as handle:
        rows = {row["Metric"]: row for row in csv.DictReader(handle)}
    for index, name in enumerate(SUMMARY_COLUMNS):
        expected = _values(index, name)
        for metric, value in expected.items():
            cell = rows[metric][name]
            if index in (4, 6) and metric not in table.TREE_LEVEL_METRICS:
                assert cell == ""
            else:
                assert cell == str(value)


def test_sidecar_identity_covers_only_the_seven_ids(tmp_path):
    store = _store(tmp_path)
    paths = _export(tmp_path, store)
    sidecar = json.loads(paths[3].read_text())
    ids = [store[label].manifest.scientific_id_sha256 for label in store]
    assert [c["scientific_id_sha256"] for c in sidecar["columns"]] == ids
    assert [c["column"] for c in sidecar["columns"]] == list(SUMMARY_COLUMNS)
    expected = hashlib.sha256(
        json.dumps(list(map(list, zip(SUMMARY_COLUMNS, ids, strict=True)))).encode()
    ).hexdigest()
    assert sidecar["table_identity_sha256"] == expected


def test_percent_display_is_appearance_only(tmp_path):
    store = _store(tmp_path)
    first = _export(tmp_path, store)
    sidecar = first[3].read_text()
    csv_text = first[0].read_text()
    html_text = first[1].read_text()
    second = _export(tmp_path, store, percent_decimals=0)
    assert second[3].read_text() == sidecar
    assert second[0].read_text() == csv_text
    assert second[1].read_text() != html_text
    assert "12%" in second[1].read_text()
    assert "12.35%" in html_text


def test_identity_changes_with_a_product_id(tmp_path):
    store = _store(tmp_path)
    before = _export(tmp_path, store)[3].read_text()
    next(iter(store.values())).manifest.scientific_id_sha256 = "0" * 64
    assert _export(tmp_path, store)[3].read_text() != before


def test_band_graph_metrics_must_match_nominal(tmp_path):
    store = _store(tmp_path, band_nodes=99)
    with pytest.raises(ValueError, match="Graph metric"):
        _export(tmp_path, store)


def test_unverified_product_is_refused(tmp_path):
    store = _store(tmp_path)
    next(iter(store.values())).report.byte_integrity = False
    with pytest.raises(ValueError, match="Unverified"):
        _export(tmp_path, store)


def test_wrong_column_is_refused(tmp_path):
    store = _store(tmp_path)
    store[_labels(store)[SUMMARY_COLUMNS[1]]].manifest.scientific_identity["column"] = (
        SUMMARY_COLUMNS[2]
    )
    with pytest.raises(ValueError, match="not the"):
        _export(tmp_path, store)


def test_labels_must_be_the_seven_ordered_columns(tmp_path):
    store = _store(tmp_path)
    labels = _labels(store)
    del labels[SUMMARY_COLUMNS[0]]
    with pytest.raises(ValueError, match="seven"):
        table.load_summary_columns(None, labels, loader=lambda k, label: store[label])


def test_config_pins_the_seven_columns():
    with (REPO_ROOT / "config/figures.toml").open("rb") as stream:
        section = tomllib.load(stream)["tables"]["landscape_summary"]
    assert tuple(section["labels"]) == SUMMARY_COLUMNS
    assert tuple(section["labels"].values()) == processed.STATES


def test_command_recipe_and_tables_step_exist():
    assert "landscape-summary-table" in CONTEXT_AWARE_COMMANDS
    text = (REPO_ROOT / "justfile").read_text(encoding="utf-8")
    assert re.search(r"^landscape-summary-table:", text, re.MULTILINE)
    tables = re.search(r"^tables:.*\n((?:    .*\n)+)", text, re.MULTILINE)
    assert tables is not None
    assert "build landscape-summary-table" in tables.group(1)
