"""Registry rows of an ``fdr_fixed_cutoff`` Candidate graph: class-based labels, same numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_manuscript_quantities import CONFIG, _rows_under, _tree, load, write_upstream

from dubanevics2026topological.analysis import manuscript_quantities as mq

SNAPSHOT = "data/processed/classification/reflected"
CASES = {
    "median_reflected": (
        "3c16ddf2f1ab6276ac086b554bfc1b200f4998d3b236cf5ae8109276d7029457",
        -0.253159,
    ),
    "papkou_original": (
        "d8509b632a8d47611ee51b0dd200328df7533d5949e768fef607d3b4131ff229",
        -0.5077739708744209,
    ),
}
TAG = "(fdr_fixed_cutoff, BY, nominal q = 0.05)"


@pytest.fixture(autouse=True)
def classification_snapshot(tmp_path_factory, monkeypatch):
    """Both BY classification folders of data/processed, written as small synthetic tables."""
    import json

    import polars as pl

    from dubanevics2026topological.threshold import fdr_selection

    root = tmp_path_factory.mktemp("processed-root")
    cutoffs = [
        {"name": name, "applied_threshold": applied, "source_precision_threshold": applied}
        for name, (_, applied) in CASES.items()
    ]
    for variant in ("reflected", "published"):
        directory = root / "data/processed/classification" / variant
        directory.mkdir(parents=True)
        (directory / "provenance.json").write_text(json.dumps({"cutoffs": cutoffs}))
        for name, (_, applied) in CASES.items():
            pl.DataFrame(
                {
                    "dna_sequence": ["CTCTGCTTG", "AAAAAAAAA"],
                    "s": [-0.229093, 0.5],
                    "fdr_class": ["uncertain", "functional"],
                }
            ).write_parquet(
                directory / fdr_selection.table_name(applied, name, "by", 0.05)
            )
    monkeypatch.setattr(fdr_selection, "ROOT", root)


def _fdr_rows(tmp_path: Path, cutoff: str) -> dict[str, dict[str, Any]]:
    write_upstream(tmp_path, band=True)
    config = {
        **CONFIG,
        "clamped": {
            **CONFIG["clamped"],
            "candidate_functional_threshold": {
                "rule": "fdr_fixed_cutoff",
                "decimals": 6,
                "cutoff": cutoff,
                "method": "by",
                "q": 0.05,
                "input_dir": SNAPSHOT,
            },
        },
    }
    rows = mq.build_rows(load(tmp_path, _tree()), config, git_commit=None)
    return {row["key"]: row for row in rows}


@pytest.mark.parametrize("cutoff", sorted(CASES))
def test_the_graph_rule_names_the_fdr_rule_with_cutoff_and_value_from_the_declaration(
    tmp_path: Path, cutoff: str
) -> None:
    rows = _fdr_rows(tmp_path, cutoff)
    text = rows["candidate_graph.vertex_count"]["selection_rule"]
    assert text == (
        "fdr_fixed_cutoff: retained Functional class from reviewed full-library BY "
        f"adjustment at nominal q = 0.05, cutoff T = {CASES[cutoff][1]!r} ({cutoff}); "
        "edges with at least one retained Functional endpoint; largest connected component"
    )
    assert "Functional probability" not in text and "confidence level" not in text
    components = rows["candidate_graph.component_count"]["selection_rule"]
    assert components.startswith("fdr_fixed_cutoff: ")
    assert "Functional-class endpoint" in components


@pytest.mark.parametrize("cutoff", sorted(CASES))
def test_the_counts_are_class_based_with_the_fdr_tag_and_an_uncertain_row(
    tmp_path: Path, cutoff: str
) -> None:
    rows = _fdr_rows(tmp_path, cutoff)
    for key in (
        "functional_count",
        "nonfunctional_count",
        "functional_percent",
        "uncertain_count",
    ):
        text = rows[f"candidate_graph.{key}"]["selection_rule"]
        assert TAG in text and "mixture_quantile" not in text, key
    assert (
        "Uncertain class" in rows["candidate_graph.uncertain_count"]["selection_rule"]
    )
    assert "published_graph.uncertain_count" not in rows
    for key in ("ridges.kem.functional_count", "ridges.two_functional_share.fraction"):
        text = rows[key]["selection_rule"]
        assert f"Functional class {TAG}" in text and "raw fitness" not in text, key


@pytest.mark.parametrize("cutoff", sorted(CASES))
def test_the_threshold_rows_name_the_fdr_variant_cutoff_and_not_median_gap(
    tmp_path: Path, cutoff: str
) -> None:
    rows = _fdr_rows(tmp_path, cutoff)
    descriptions = {
        "median_reflected": "the historical applied median-reflected cutoff",
        "papkou_original": "the Papkou et al. original cutoff, reconstructed",
    }
    desc = descriptions[cutoff]
    expected_value = (
        f"Fixed Candidate cutoff T declared by the fdr_fixed_cutoff variant "
        f"({cutoff}: {desc}); not refitted"
    )
    expected_unrounded = f"{expected_value}, before rounding"

    value_row = rows["thresholds.candidate_functional.value"]
    unrounded_row = rows["thresholds.candidate_functional.unrounded_value"]

    assert value_row["selection_rule"] == expected_value
    assert unrounded_row["selection_rule"] == expected_unrounded

    assert "median plus the Median gap" not in value_row["selection_rule"]
    assert "median plus the Median gap" not in unrounded_row["selection_rule"]


@pytest.mark.parametrize("cutoff", sorted(CASES))
def test_uncertain_and_nonfunctional_explanations_have_fdr_semantics(
    tmp_path: Path, cutoff: str
) -> None:
    rows = _fdr_rows(tmp_path, cutoff)
    uncertain_rule = rows["candidate_graph.uncertain_count"]["selection_rule"]
    assert uncertain_rule == (
        f"vertices of the Candidate graph in the Uncertain class {TAG}: "
        "at or above cutoff T but not retained by the full-library BY adjustment at nominal q = 0.05; "
        "excluded from the retained Functional set"
    )
    assert "confidence level" not in uncertain_rule
    assert "neither Functional nor Non-functional" not in uncertain_rule

    nonfunctional_rule = rows["candidate_graph.nonfunctional_count"]["selection_rule"]
    assert nonfunctional_rule == (
        f"vertices of the Candidate graph in the Non-functional class {TAG}: "
        "raw fitness below cutoff T (point-estimate category; not statistically confirmed nonfunctional)"
    )
    assert "confidence level" not in nonfunctional_rule


@pytest.mark.parametrize("cutoff", sorted(CASES))
def test_the_numbers_equal_those_of_the_class_based_mixture_rows(
    tmp_path: Path, cutoff: str
) -> None:
    # Same fixture comparison summary, so the quantities must not change with the rule: only text does.
    fdr = _fdr_rows(tmp_path / "fdr", cutoff)
    mixture = _rows_under(tmp_path / "mixture", "mixture_quantile")
    for key in (
        "candidate_graph.functional_count",
        "candidate_graph.nonfunctional_count",
        "candidate_graph.functional_percent",
        "candidate_graph.uncertain_count",
        "candidate_graph.vertex_count",
        "candidate_graph.component_count",
        "ridges.kem.functional_count",
        "ridges.two_functional_share.fraction",
        "thresholds.candidate_functional.value",
        "thresholds.candidate_functional.unrounded_value",
    ):
        assert fdr[key]["value"] == mixture[key]["value"], key


def test_mixture_rule_texts_remain_byte_identical_to_32b92af(tmp_path: Path) -> None:
    mixture = _rows_under(tmp_path, "mixture_quantile")
    assert (
        mixture["thresholds.candidate_functional.value"]["selection_rule"]
        == "median plus the Median gap of the Full graph raw fitness, rounded to the configured decimals (ADR 2026-10-23)"
    )
    assert (
        mixture["thresholds.candidate_functional.unrounded_value"]["selection_rule"]
        == "median plus the Median gap, before rounding"
    )
    assert (
        mixture["candidate_graph.functional_count"]["selection_rule"]
        == "vertices of the Candidate graph in the Functional class (mixture_quantile), not raw fitness at or above the threshold"
    )
    assert (
        mixture["candidate_graph.nonfunctional_count"]["selection_rule"]
        == "vertices of the Candidate graph in the Non-functional class (mixture_quantile); the Uncertain class is counted separately"
    )
    assert (
        mixture["candidate_graph.functional_percent"]["selection_rule"]
        == "vertices of the Candidate graph in the Functional class (mixture_quantile) as a percentage of all its vertices"
    )
    assert (
        mixture["candidate_graph.uncertain_count"]["selection_rule"]
        == "vertices of the Candidate graph in the Uncertain class (mixture_quantile): neither Functional nor Non-functional at the declared confidence level"
    )
    assert (
        mixture["candidate_graph.vertex_count"]["selection_rule"]
        == "mixture_quantile: Functional class (Functional probability at or above the declared confidence level); edges with at least one Functional endpoint; largest connected component"
    )


def test_the_mixture_classification_rows_stay_named_gaps_in_a_variant_run(
    tmp_path: Path,
) -> None:
    rows = _fdr_rows(tmp_path, "median_reflected")
    row = rows[
        "functional_classification.mixture_quantile_functional_threshold_fitness"
    ]
    assert row["value"] is None and row["gap_reason"]


# ------------------------------------------------------------ common saddle finding


def test_fdr_saddle_differing_reports_finding_and_emits_difference_rows(
    tmp_path: Path,
) -> None:
    import json

    write_upstream(tmp_path, band=True)

    # Set up Candidate split tree where KEM and TCL merge at AGTTGCCAA (s = -0.42699)
    tree = _tree("AGTTGCCAA")
    for i in tree.node_indices():
        if tree[i]["dna_sequence"] == "AGTTGCCAA":
            tree[i]["scalar"] = -0.42699
            tree[i]["aa_sequence"] = "SCQ"

    # Codon entropy metadata consistency check expects common_saddle to match tree's LCA
    path = tmp_path / "tables" / mq.CODON_METADATA
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["common_saddle"]["dna_sequence"] = "AGTTGCCAA"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    config = {
        **CONFIG,
        "clamped": {
            **CONFIG["clamped"],
            "candidate_functional_threshold": {
                "rule": "fdr_fixed_cutoff",
                "decimals": 6,
                "cutoff": "median_reflected",
                "method": "by",
                "q": 0.05,
                "input_dir": SNAPSHOT,
            },
        },
    }

    # Must NOT raise under fdr_fixed_cutoff
    rows = {
        r["key"]: r
        for r in mq.build_rows(load(tmp_path, tree), config, git_commit=None)
    }

    # Computed saddle emitted as primary registry values
    assert rows["common_saddle.dna_sequence"]["value"] == "AGTTGCCAA"
    assert rows["common_saddle.aa_sequence"]["value"] == "SCQ"
    assert rows["common_saddle.raw_fitness"]["value"] == "-0.42699"
    assert rows["common_saddle.raw_fitness"]["display_value"] == "-0.43"

    # Difference rows present with explicit findings
    assert rows["common_saddle.reported_dna_sequence"]["value"] == "CTCTGCTTG"
    assert rows["common_saddle.differs_from_reported"]["value"] == "true"
    assert rows["common_saddle.reported_raw_fitness"]["display_value"] == "-0.23"
    assert rows["common_saddle.merge_height_difference"]["display_value"] == "0.20"
    diff_rule = rows["common_saddle.merge_height_difference"]["selection_rule"]
    assert "reported minus computed" in diff_rule

    mechanism = rows["common_saddle.mechanism"]["value"]
    expected_mechanism = (
        "The reported saddle CTCTGCTTG is classed uncertain under the BY, "
        "nominal q = 0.05 screen at cutoff T; in this Candidate graph (edges need "
        "≥1 retained Functional endpoint) KEM and TCL first merge at AGTTGCCAA "
        "(s = -0.43), 0.20 below the reported saddle (s = -0.23)."
    )
    assert mechanism == expected_mechanism


def test_fdr_saddle_computed_above_reported_says_above(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The computed saddle (CTCTGCTTG) lies above the reported one (AGTTGCCAA)."""
    monkeypatch.setattr(
        mq, "_lookup_fdr_sequence", lambda dna, declaration: (-0.42699, "uncertain")
    )
    write_upstream(tmp_path, band=True)
    tree = _tree("CTCTGCTTG")
    for i in tree.node_indices():
        if tree[i]["dna_sequence"] == "CTCTGCTTG":
            tree[i]["scalar"] = -0.229093
    config = {
        "clamped": {
            **CONFIG["clamped"],
            "candidate_functional_threshold": {
                "rule": "fdr_fixed_cutoff",
                "decimals": 6,
                "cutoff": "median_reflected",
                "method": "by",
                "q": 0.05,
                "input_dir": SNAPSHOT,
            },
        },
        "codon_entropy": {
            **CONFIG["codon_entropy"],
            "reported_saddle_dna_sequence": "AGTTGCCAA",
        },
    }
    rows = {
        r["key"]: r
        for r in mq.build_rows(load(tmp_path, tree), config, git_commit=None)
    }
    assert rows["common_saddle.dna_sequence"]["value"] == "CTCTGCTTG"
    assert rows["common_saddle.reported_raw_fitness"]["display_value"] == "-0.43"
    assert rows["common_saddle.merge_height_difference"]["display_value"] == "-0.20"
    assert rows["common_saddle.mechanism"]["value"] == (
        "The reported saddle AGTTGCCAA is classed uncertain under the BY, "
        "nominal q = 0.05 screen at cutoff T; in this Candidate graph (edges need "
        "≥1 retained Functional endpoint) KEM and TCL first merge at CTCTGCTTG "
        "(s = -0.23), 0.20 above the reported saddle (s = -0.43)."
    )


def test_lookup_fdr_sequence_fails_closed() -> None:
    # 1. Missing sequence raises ValueError
    decl = {
        "rule": "fdr_fixed_cutoff",
        "decimals": 6,
        "cutoff": "median_reflected",
        "method": "by",
        "q": 0.05,
        "input_dir": SNAPSHOT,
    }
    with pytest.raises(ValueError, match="absent from FDR classification table"):
        mq._lookup_fdr_sequence("ZZZZZZZZZ", decl)

    # 2. Missing file raises FileNotFoundError
    missing_decl = {
        **decl,
        "input_dir": "data/processed/classification/nonexistent_dir",
    }
    with pytest.raises((FileNotFoundError, ValueError)):
        mq._lookup_fdr_sequence("CTCTGCTTG", missing_decl)

    # 3. Present sequence resolves exact columns without fallback
    s, fdr_class = mq._lookup_fdr_sequence("CTCTGCTTG", decl)
    assert pytest.approx(s, abs=1e-6) == -0.229093
    assert fdr_class == "uncertain"


def test_fdr_saddle_equal_emits_no_difference_rows(tmp_path: Path) -> None:
    rows = _fdr_rows(tmp_path, "papkou_original")

    # Primary saddle emitted
    assert rows["common_saddle.dna_sequence"]["value"] == "CTCTGCTTG"
    assert rows["common_saddle.aa_sequence"]["value"] == "LCL"

    # No difference rows emitted when computed equals reported
    for key in (
        "common_saddle.reported_dna_sequence",
        "common_saddle.differs_from_reported",
        "common_saddle.reported_raw_fitness",
        "common_saddle.merge_height_difference",
        "common_saddle.mechanism",
    ):
        assert key not in rows, (
            f"Unexpected difference key {key!r} when saddle is equal"
        )


def test_default_mixture_rule_differing_saddle_still_raises(tmp_path: Path) -> None:
    write_upstream(tmp_path, band=True)
    tree = _tree("AGTTGCCAA")
    with pytest.raises(
        ValueError,
        match="The computed common saddle 'AGTTGCCAA' differs from the reported saddle 'CTCTGCTTG'",
    ):
        mq.build_rows(load(tmp_path, tree), CONFIG, git_commit=None)


def test_common_saddle_function_allows_difference_flag() -> None:
    tree = _tree("AGTTGCCAA")
    # Strict by default: raises
    with pytest.raises(
        ValueError,
        match="The computed common saddle 'AGTTGCCAA' differs from the reported saddle 'CTCTGCTTG'",
    ):
        mq.common_saddle(tree, "AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG")

    # Strict when allow_difference=False: raises
    with pytest.raises(ValueError, match="differs from the reported saddle"):
        mq.common_saddle(
            tree, "AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG", allow_difference=False
        )

    # Allowed when allow_difference=True: returns computed saddle
    found = mq.common_saddle(
        tree, "AAGGAAATG", "ACCTGTCTT", "CTCTGCTTG", allow_difference=True
    )
    assert found["dna_sequence"] == "AGTTGCCAA"
    assert found["aa_sequence"] == "LCL"
    assert found["raw_fitness"] == -0.2531585


def test_manuscript_quantities_stage_with_median_reflected_variant(
    tmp_path: Path,
) -> None:
    import json
    import tomllib

    from dubanevics2026topological import build
    from dubanevics2026topological.analysis import ROOT
    from dubanevics2026topological.threshold.functional_threshold import DECLARATION_KEY

    write_upstream(tmp_path, band=True)
    tree = _tree("AGTTGCCAA")
    for i in tree.node_indices():
        if tree[i]["dna_sequence"] == "AGTTGCCAA":
            tree[i]["scalar"] = -0.42699
            tree[i]["aa_sequence"] = "SCQ"

    path = tmp_path / "tables" / mq.CODON_METADATA
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["common_saddle"]["dna_sequence"] = "AGTTGCCAA"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    sources = load(tmp_path, tree)

    with (ROOT / "config/analysis.toml").open("rb") as stream:
        analysis = tomllib.load(stream)
    with (ROOT / "config/variants/threshold=reflected_fdr=by_q=0.05.toml").open(
        "rb"
    ) as stream:
        overlay = tomllib.load(stream)
    for section, values in overlay.items():
        if isinstance(values, dict) and isinstance(analysis.get(section), dict):
            analysis[section] = {**analysis[section], **values}
        else:
            analysis[section] = values
    settings = {"analysis": analysis, "figures": {}}

    # The shipped reported saddle is now the computed one; this test exercises the
    # difference rows, so it reports the other sequence.
    analysis["codon_entropy"] = {
        **analysis["codon_entropy"],
        "reported_saddle_dna_sequence": "CTCTGCTTG",
    }
    stage, stage_config = build._manuscript_quantities_stage(settings, sources)

    # 1. Declaration passes into stage_config["clamped"] and stage fingerprint
    assert DECLARATION_KEY in stage_config["clamped"]
    assert stage_config["clamped"][DECLARATION_KEY]["rule"] == "fdr_fixed_cutoff"
    assert stage_config["clamped"][DECLARATION_KEY]["cutoff"] == "median_reflected"
    assert (
        stage.settings["config"]["clamped"][DECLARATION_KEY]["rule"]
        == "fdr_fixed_cutoff"
    )

    # 2. build_rows receives stage_config: candidate_rule is fdr_fixed_cutoff, does not raise
    rows_list = mq.build_rows(sources, stage_config, git_commit=None)
    rows = {r["key"]: r for r in rows_list}

    # Verify candidate_rule is fdr_fixed_cutoff and FDR texts appear
    assert rows["candidate_graph.vertex_count"]["selection_rule"].startswith(
        "fdr_fixed_cutoff:"
    )
    assert (
        "(fdr_fixed_cutoff, BY, nominal q = 0.05)"
        in rows["candidate_graph.functional_count"]["selection_rule"]
    )
    assert (
        "Fixed Candidate cutoff T declared by the fdr_fixed_cutoff variant "
        "(median_reflected: the historical applied median-reflected cutoff); not refitted"
        == rows["thresholds.candidate_functional.value"]["selection_rule"]
    )

    # Verify saddle difference rows appear
    assert rows["common_saddle.dna_sequence"]["value"] == "AGTTGCCAA"
    assert rows["common_saddle.reported_dna_sequence"]["value"] == "CTCTGCTTG"
    assert rows["common_saddle.differs_from_reported"]["value"] == "true"
    assert rows["common_saddle.reported_raw_fitness"]["display_value"] == "-0.23"
    assert rows["common_saddle.merge_height_difference"]["display_value"] == "0.20"
    assert "common_saddle.mechanism" in rows


def test_manuscript_quantities_stage_when_declaration_absent_stays_byte_identical(
    tmp_path: Path,
) -> None:
    from dubanevics2026topological import build
    from dubanevics2026topological.threshold.functional_threshold import DECLARATION_KEY

    write_upstream(tmp_path, band=True)
    sources = load(tmp_path, _tree())
    settings_absent = {
        "analysis": {
            "clamped": {"published_functional_threshold": -0.507774},
            "codon_entropy": CONFIG["codon_entropy"],
        }
    }
    stage, stage_config = build._manuscript_quantities_stage(settings_absent, sources)
    assert DECLARATION_KEY not in stage_config["clamped"]
    assert stage_config["clamped"] == {"published_functional_threshold": -0.507774}
    assert DECLARATION_KEY not in stage.settings["config"]["clamped"]


def test_lookup_fdr_sequence_fails_on_duplicate_row(tmp_path: Path) -> None:
    import json

    import polars as pl

    from dubanevics2026topological.threshold.fdr_selection import table_name

    applied = -0.253159
    name = table_name(applied, "median_reflected", "by", 0.05)
    df = pl.DataFrame(
        {
            "dna_sequence": ["CTCTGCTTG", "CTCTGCTTG"],
            "s": [-0.229, -0.229],
            "fdr_class": ["uncertain", "uncertain"],
        }
    )
    table_path = tmp_path / name
    df.write_parquet(table_path)
    provenance = {
        "cutoffs": [
            {
                "name": "median_reflected",
                "applied_threshold": applied,
                "source_precision_threshold": applied,
            }
        ]
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    declaration = {
        "rule": "fdr_fixed_cutoff",
        "decimals": 6,
        "cutoff": "median_reflected",
        "method": "by",
        "q": 0.05,
        "input_dir": str(tmp_path),
    }
    with pytest.raises(ValueError, match="matches multiple rows"):
        mq._lookup_fdr_sequence("CTCTGCTTG", declaration)


def test_band_quantile_percent_selection_rule_fdr_vs_default(tmp_path: Path) -> None:
    fdr_rows = _fdr_rows(tmp_path / "fdr", "median_reflected")
    assert fdr_rows["bootstrap.band_quantile_percent"]["selection_rule"] == (
        "the band quantile as a percentage (nominal band level; coverage not independently established)"
    )

    write_upstream(tmp_path / "default", band=True)
    default_rows = {
        r["key"]: r
        for r in mq.build_rows(
            load(tmp_path / "default", _tree()), CONFIG, git_commit=None
        )
    }
    assert default_rows["bootstrap.band_quantile_percent"]["selection_rule"] == (
        "the band quantile as a percentage (the confidence level of the band)"
    )
