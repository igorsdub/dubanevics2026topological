"""Check shipped selection configuration against its reviewed inputs."""

import hashlib
import json
import tomllib
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis import ROOT, load_settings
from dubanevics2026topological.build import saved_figure_inputs
from dubanevics2026topological.landscape.selection import (
    selection_source,
    threshold_from_summary,
)
from dubanevics2026topological.results import RunContext
from dubanevics2026topological.landscape.selection import wild_type_sequence
from dubanevics2026topological.threshold.fdr_selection import (
    load_fdr_snapshot,
    table_name,
)
from dubanevics2026topological.threshold.functional_threshold import (
    configured_declaration,
    declaration_identity,
)


@pytest.fixture
def shipped_snapshot(tmp_path, monkeypatch):
    """The declared classification folder, written with the reported threshold under a temporary root."""
    if not (ROOT / "data/raw/papkou2023_fitness_wt.parquet").is_file():
        pytest.skip("needs the gitignored raw tables under data/raw/ (absent in a clean public clone)")
    with (ROOT / "config/numbers.toml").open("rb") as stream:
        applied = tomllib.load(stream)["thresholds"]["reflected_functional_threshold"]
    settings = load_settings(context=RunContext(env={}))
    declaration = configured_declaration(settings)
    directory = tmp_path / declaration["input_dir"]
    directory.mkdir(parents=True)
    wild_type = wild_type_sequence(settings)
    table = directory / table_name(
        applied, declaration["cutoff"], declaration["method"], declaration["q"]
    )
    pl.DataFrame(
        {"dna_sequence": [wild_type, "other"], "fdr_class": ["functional", "uncertain"]}
    ).write_parquet(table)
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "cutoffs": [
                    {
                        "name": declaration["cutoff"],
                        "source_precision_threshold": applied,
                        "applied_threshold": applied,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FITNESSTREE_PROCESSED_ROOT", str(tmp_path / "data" / "processed"))
    return directory, table


def test_shipped_declaration_names_the_processed_classification():
    declaration = configured_declaration(load_settings(context=RunContext(env={})))
    assert declaration["input_dir"] == "data/processed/classification/reflected"
    assert "table_sha256" not in declaration


def test_shipped_threshold_matches_reported_value(shipped_snapshot):
    with (ROOT / "config/numbers.toml").open("rb") as stream:
        numbers = tomllib.load(stream)
    threshold = threshold_from_summary(
        load_settings(context=RunContext(env={})), raw_fitness=[-1.0, 0.0, 1.0]
    )
    assert (
        threshold.threshold == numbers["thresholds"]["reflected_functional_threshold"]
    )


def test_shipped_snapshot_reports_the_digest_of_its_own_table(shipped_snapshot):
    _, table = shipped_snapshot
    declaration = configured_declaration(load_settings(context=RunContext(env={})))
    snapshot = load_fdr_snapshot(declaration)
    assert snapshot.source["table"] == table.name
    assert snapshot.source["table_sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()


def test_a_missing_shipped_snapshot_names_the_recipes(tmp_path, monkeypatch):
    monkeypatch.setenv("FITNESSTREE_PROCESSED_ROOT", str(tmp_path / "data" / "processed"))
    declaration = configured_declaration(load_settings(context=RunContext(env={})))
    with pytest.raises(FileNotFoundError, match="just all.*just fetch-data"):
        load_fdr_snapshot(declaration)


@pytest.mark.parametrize("case", ["valid", "missing", "mismatch", "ambiguous"])
def test_shipped_selection_routes_to_processed_product(monkeypatch, case, shipped_snapshot):
    names = []
    context = RunContext(env={})
    default = selection_source(load_settings(context=context), context=context)
    assert "input_dir" not in default
    key = "data/processed/classification/snapshot/" + default["table"]
    provenance = {key: default["table_sha256"]}
    if case == "missing":
        provenance = {}
    elif case == "mismatch":
        provenance[key] = "bad"
    elif case == "ambiguous":
        provenance["another/" + default["table"]] = default["table_sha256"]

    def tree_for_leaf(name, context):
        names.append(name)
        return SimpleNamespace(
            label=name,
            manifest=SimpleNamespace(
                scientific_id_sha256=name, data_provenance=provenance
            ),
        )

    class Products:
        def __init__(self, context):
            pass

        def landscape_population(self, landscape):
            return landscape

    monkeypatch.setattr(saved_figure_inputs, "PresentationProducts", Products)
    monkeypatch.setattr(saved_figure_inputs, "_tree_for_leaf", tree_for_leaf)
    monkeypatch.setattr(
        saved_figure_inputs, "_landscape_for_tree", lambda products, tree: tree
    )
    monkeypatch.setattr(
        saved_figure_inputs, "_layout", lambda products, tree: (tree, "layout")
    )
    if case != "valid":
        message = {
            "missing": "zero matching keys",
            "mismatch": "digest mismatch",
            "ambiguous": "multiple matching keys",
        }[case]
        with pytest.raises(ValueError, match=message):
            saved_figure_inputs.load_saved_figure_inputs(
                context, default, include_band=False
            )
        assert names == ["full", "published", "reflected_by"]
        return
    result = saved_figure_inputs.load_saved_figure_inputs(
        context, default, include_band=False
    )
    assert result["mode"] == "saved"
    assert names == ["full", "published", "reflected_by"]

    names.clear()
    mixture = declaration_identity(
        configured_declaration(
            load_settings(context=RunContext(variant_name="mixture_quantile", env={}))
        )
    )
    result = saved_figure_inputs.load_saved_figure_inputs(
        RunContext(env={}), mixture, include_band=False
    )
    assert result["mode"] == "legacy_mixture"
    assert not names


def test_shipped_reported_saddle_is_scq() -> None:
    """The reported KEM/TCL common saddle is SCQ (AGTTGCCAA), not LCL (CTCTGCTTG)."""
    with (ROOT / "config/analysis.toml").open("rb") as stream:
        entropy = tomllib.load(stream)["codon_entropy"]
    assert entropy["reported_saddle_dna_sequence"] == "AGTTGCCAA"
