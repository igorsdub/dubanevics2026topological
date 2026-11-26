"""Tests of the manuscript namer."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from dubanevics2026topological.naming import (
    Entry,
    _validate,
    TableEntry,
    load_manuscript_map,
    load_table_map,
    name_tables,
    name_outputs,
)

ROOT = Path(__file__).resolve().parents[1]


def _png(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def figures(tmp_path: Path) -> Path:
    root = tmp_path / "figures"
    _png(root / "canonical" / "tree.png", b"tree")
    _png(root / "canonical" / "titled" / "tree.png", b"tree titled")
    _png(root / "canonical" / "grid.png", b"grid")
    _png(root / "external" / "workflow.png", b"slide")
    return root


ENTRIES = [
    Entry(section="main", number="1", desc="tree", source="tree"),
    Entry(section="main", number="2", desc="workflow", external=True),
    Entry(section="main", number="3", desc="later", status="planned"),
]


def test_repository_map_is_valid():
    entries = load_manuscript_map(ROOT / "config" / "manuscript.toml")
    main = [entry.number for entry in entries if entry.section == "main"]
    assert main == [str(number) for number in range(1, 10)]


def test_figure_six_maps_the_peak_uncertainty_render(tmp_path: Path):
    entries = load_manuscript_map(ROOT / "config" / "manuscript.toml")
    entry = next(entry for entry in entries if entry.number == "6")
    assert entry.source == "peak_uncertainty" and entry.status != "planned"
    _png(tmp_path / "canonical" / "peak_uncertainty.png", b"figure six")
    name_outputs(tmp_path, [entry])
    assert (
        tmp_path / "main" / "fig6_peak_uncertainty.png"
    ).read_bytes() == b"figure six"


def test_numbered_external_and_exploratory_copies(figures: Path):
    result = name_outputs(figures, ENTRIES)
    assert (figures / "main" / "fig1_tree.png").read_bytes() == b"tree"
    assert (
        figures / "titled" / "main" / "fig1_tree.png"
    ).read_bytes() == b"tree titled"
    assert (figures / "main" / "fig2_workflow.png").read_bytes() == b"slide"
    assert (figures / "exploratory" / "grid.png").read_bytes() == b"grid"
    assert not (figures / "exploratory" / "tree.png").exists()
    assert result.planned == ["fig3_later"]
    assert result.missing == []


def test_manifest_records_hashes(figures: Path):
    name_outputs(figures, ENTRIES)
    with (figures / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    row = next(row for row in rows if row["path"] == "main/fig1_tree.png")
    assert row["sha256"] == hashlib.sha256(b"tree").hexdigest()
    assert row["source"] == "tree" and row["variant"] == "titleless"


def test_renumbering_removes_stale_names(figures: Path):
    name_outputs(figures, ENTRIES)
    renumbered = [Entry(section="main", number="4", desc="tree", source="tree")]
    result = name_outputs(figures, renumbered)
    assert not (figures / "main" / "fig1_tree.png").exists()
    assert (figures / "main" / "fig4_tree.png").exists()
    assert figures / "main" / "fig1_tree.png" in result.removed
    assert (figures / "canonical" / "tree.png").exists()
    assert (figures / "external" / "workflow.png").exists()


def test_missing_render_is_reported_not_fatal(figures: Path):
    (figures / "external" / "workflow.png").unlink()
    result = name_outputs(figures, ENTRIES)
    assert "external/workflow.png" in result.missing
    assert (figures / "main" / "fig1_tree.png").exists()


def test_missing_manuscript_external_figures_in_fresh_output(tmp_path: Path):
    entries = load_manuscript_map(ROOT / "config" / "manuscript.toml")
    result = name_outputs(tmp_path / "figures", entries)
    assert [path for path in result.missing if path.startswith("external/")] == [
        "external/workflow.png",
        "external/merge_tree_simplification_parametric_bootstrap.png",
        "external/topographer_fdr_tree_band_simplification.png",
    ]


@pytest.mark.parametrize(
    "entries",
    [
        [Entry(section="main", desc="tree", source="tree")],
        [Entry(section="main", number="S1", desc="tree", source="tree")],
        [Entry(section="supplementary", number="2", desc="tree", source="tree")],
        [Entry(section="main", number="1", desc="Tree-Plot", source="tree")],
        [
            Entry(section="main", number="1", desc="a", source="a"),
            Entry(section="main", number="1", desc="b", source="b"),
        ],
        [
            Entry(section="main", number="1", desc="a", source="x"),
            Entry(section="main", number="2", desc="b", source="x"),
        ],
        [Entry(section="main", number="1", desc="a")],
        [Entry(section="main", number="1", desc="a", source="a", external=True)],
    ],
)
def test_invalid_maps_are_refused(entries: list[Entry]):
    with pytest.raises(ValueError):
        _validate(entries)


def test_justfile_has_name_outputs_recipe():
    text = (ROOT / "justfile").read_text(encoding="utf-8")
    assert "name-outputs:" in text
    assert "-m dubanevics2026topological.naming" in text


def test_a_titled_twin_without_a_manuscript_entry_goes_to_titled_exploratory(
    figures: Path,
):
    name_outputs(figures, ENTRIES)
    _png(figures / "canonical" / "titled" / "grid.png", b"grid titled")
    name_outputs(figures, ENTRIES)
    assert (figures / "exploratory" / "grid.png").read_bytes() == b"grid"
    assert (
        figures / "titled" / "exploratory" / "grid.png"
    ).read_bytes() == b"grid titled"


def test_repository_tables_are_built_and_valid():
    entries = load_table_map(ROOT / "config" / "manuscript.toml")
    assert [(e.number, e.desc, e.status) for e in entries] == [
        ("1", "graph_descriptive", None),
        ("2", "tree_descriptive", None),
        ("S1", "graph_summary", None),
        ("S2", "topographer_corrections", None),
    ]


def test_tables_are_copied_to_numbered_names(tmp_path: Path):
    canonical = tmp_path / "canonical"
    for extension in ("csv", "html", "md", "json"):
        _png(canonical / f"t.{extension}", extension.encode())
    entries = [
        TableEntry(section="main", number="1", desc="summary", source="t"),
        TableEntry(section="main", number="2", desc="later", status="planned"),
    ]
    result = name_tables(tmp_path, entries)
    assert (tmp_path / "main" / "tab1_summary.csv").read_bytes() == b"csv"
    assert (tmp_path / "main" / "tab1_summary.json").read_bytes() == b"json"
    assert result.planned == ["tab2_later"] and result.missing == []
    with (tmp_path / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["variant"] for row in rows} == {"csv", "html", "md", "json"}
    renumbered = [TableEntry(section="main", number="3", desc="summary", source="t")]
    result = name_tables(tmp_path, renumbered)
    assert not (tmp_path / "main" / "tab1_summary.csv").exists()
    assert (tmp_path / "main" / "tab3_summary.csv").exists()
    assert len(result.removed) == 4


def test_missing_table_export_is_reported_not_fatal(tmp_path: Path):
    entries = [TableEntry(section="main", number="1", desc="summary", source="t")]
    result = name_tables(tmp_path, entries)
    assert "canonical/t.csv" in result.missing


def test_invalid_table_maps_are_refused(tmp_path: Path):
    from dubanevics2026topological.naming import _validate_tables

    with pytest.raises(ValueError):
        _validate_tables(
            [TableEntry(section="main", number="S1", desc="a", source="a")]
        )
    with pytest.raises(ValueError):
        _validate_tables([TableEntry(section="main", number="1", desc="a")])


def _exports(root: Path, stem: str) -> None:
    for extension in ("csv", "html", "md", "json"):
        _png(root / "canonical" / f"{stem}.{extension}", extension.encode())


def test_a_supplementary_table_goes_to_the_supplementary_shelf(tmp_path: Path):
    _exports(tmp_path, "t")
    entry = TableEntry(section="supplementary", number="S3", desc="extra", source="t")
    name_tables(tmp_path, [entry])
    assert (tmp_path / "supplementary" / "tabS3_extra.csv").read_bytes() == b"csv"


def test_stale_removal_leaves_other_files_alone(tmp_path: Path):
    _exports(tmp_path, "t")
    entry = TableEntry(section="main", number="1", desc="summary", source="t")
    name_tables(tmp_path, [entry])
    (tmp_path / "main" / "notes.csv").write_text("keep")
    (tmp_path / "main" / "tab9_other.txt").write_text("keep")
    result = name_tables(tmp_path, [entry])
    assert result.removed == []
    assert (tmp_path / "main" / "notes.csv").exists()
    assert (tmp_path / "main" / "tab9_other.txt").exists()


def test_a_named_copy_goes_when_its_source_disappears(tmp_path: Path):
    _exports(tmp_path, "t")
    entry = TableEntry(section="main", number="1", desc="summary", source="t")
    name_tables(tmp_path, [entry])
    (tmp_path / "canonical" / "t.html").unlink()
    result = name_tables(tmp_path, [entry])
    assert not (tmp_path / "main" / "tab1_summary.html").exists()
    assert (tmp_path / "main" / "tab1_summary.csv").exists()
    assert "canonical/t.html" in result.missing


def test_duplicate_table_numbers_and_sources_are_refused():
    from dubanevics2026topological.naming import _validate_tables

    with pytest.raises(ValueError, match="used twice"):
        _validate_tables(
            [
                TableEntry(section="main", number="1", desc="a", source="a"),
                TableEntry(section="main", number="1", desc="b", source="b"),
            ]
        )
    with pytest.raises(ValueError, match="mapped twice"):
        _validate_tables(
            [
                TableEntry(section="main", number="1", desc="a", source="a"),
                TableEntry(section="main", number="2", desc="b", source="a"),
            ]
        )
