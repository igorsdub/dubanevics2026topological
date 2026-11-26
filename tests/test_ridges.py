"""KEM/TCL ridge populations on small, independently inspectable trees."""

from __future__ import annotations

import polars as pl
import pytest

from dubanevics2026topological.analysis import describe
from dubanevics2026topological.analysis.ridges import RidgeTree

SUMMITS = {"kem": "kem", "tcl": "tcl"}


def _tree(nodes, edges, regular=()) -> RidgeTree:
    """``nodes`` are (dna, scalar); ``edges`` (upper, lower); ``regular`` (upper, lower, dna)."""
    return RidgeTree(
        pl.DataFrame(nodes, schema=["tree_node_identity", "scalar"], orient="row"),
        pl.DataFrame(
            edges,
            schema=["source_tree_node_identity", "target_tree_node_identity"],
            orient="row",
        ),
        pl.DataFrame(
            regular,
            schema={
                "source_tree_node_identity": pl.String,
                "target_tree_node_identity": pl.String,
                "dna_sequence": pl.String,
            },
            orient="row",
        ),
    )


# kem and tcl merge at S; a side hill also enters S; the trunk is S -> tr -> root.
BASE = _tree(
    [
        ("kem", 9.0),
        ("tcl", 8.0),
        ("side", 7.0),
        ("S", 4.0),
        ("root", 0.0),
    ],
    [("kem", "S"), ("tcl", "S"), ("side", "S"), ("S", "root")],
    [
        ("kem", "S", "k1"),
        ("tcl", "S", "t1"),
        ("side", "S", "s1"),
        ("S", "root", "tr"),
    ],
)
VERTICES = {
    "kem": 9.0,
    "k1": 8.5,
    "tcl": 8.0,
    "t1": 6.0,
    "side": 7.0,
    "s1": 6.5,
    "S": 4.0,
    "tr": 2.0,
    "root": 0.0,
}
CLASSES = {
    "kem": "functional",
    "k1": "functional",
    "tcl": "functional",
    "t1": "uncertain",
    "side": "functional",
    "s1": "non_functional",
    "S": "non_functional",
    "tr": "non_functional",
    "root": "non_functional",
}


def _ridges(tree=BASE, **kwargs):
    arguments = {
        "ridge_tree": tree,
        "ridge_population": "published_by",
        "ridge_summits": SUMMITS,
        "ridge_vertices": VERTICES,
        "ridge_classes": CLASSES,
    }
    return describe(products={"ridges"}, **{**arguments, **kwargs}).ridges


def test_saddle_and_trunk_belong_to_no_ridge_with_null_name() -> None:
    members = _ridges().members
    names = dict(
        zip(members["member_dna_sequence"], members["ridge_name"], strict=True)
    )
    assert names["kem"] == names["k1"] == "kem"
    assert names["tcl"] == names["t1"] == "tcl"
    for sequence in ("S", "tr", "root", "side", "s1"):
        assert names[sequence] is None
    unassigned = members.filter(pl.col("ridge_name").is_null())
    assert unassigned["summit_dna_sequence"].null_count() == unassigned.height
    assert "unassigned" not in set(members["ridge_name"].drop_nulls())
    assert members.height == len(VERTICES)


def test_summary_counts_and_share() -> None:
    ridges = _ridges()
    rows = {r["ridge_name"]: r for r in ridges.summary.iter_rows(named=True)}
    assert rows["kem"]["summit_dna_sequence"] == "kem"
    assert rows["kem"]["summit_raw_fitness"] == 9.0
    assert rows["kem"]["common_saddle_dna_sequence"] == "S"
    assert rows["kem"]["common_saddle_fitness"] == 4.0
    assert rows["kem"]["member_count"] == 2
    assert rows["kem"]["functional_count"] == 2
    # Four functional sequences in the graph: kem, k1, tcl, side.
    assert rows["kem"]["functional_share"] == pytest.approx(50.0)
    # The uncertain t1 is never counted as functional.
    assert rows["tcl"]["functional_count"] == 1
    assert rows["tcl"]["functional_share"] == pytest.approx(25.0)


def test_common_saddle_is_per_tree_not_fixed() -> None:
    shifted = _tree(
        [
            ("kem", 9.0),
            ("tcl", 8.0),
            ("side", 7.0),
            ("M", 5.0),
            ("S", 4.0),
            ("root", 0.0),
        ],
        [("kem", "M"), ("tcl", "M"), ("M", "S"), ("side", "S"), ("S", "root")],
        [
            ("kem", "M", "k1"),
            ("tcl", "M", "t1"),
            ("side", "S", "s1"),
            ("S", "root", "tr"),
        ],
    )
    vertices = {**VERTICES, "M": 5.0}
    classes = {**CLASSES, "M": "non_functional"}
    ridges = _ridges(shifted, ridge_vertices=vertices, ridge_classes=classes)
    assert set(ridges.summary["common_saddle_dna_sequence"]) == {"M"}
    assert ridges.summary["common_saddle_fitness"].to_list() == [5.0, 5.0]
    # M is the saddle here, so it belongs to no ridge.
    names = dict(
        zip(
            ridges.members["member_dna_sequence"],
            ridges.members["ridge_name"],
            strict=True,
        )
    )
    assert names["M"] is None


def test_ridges_are_distinct_from_leaves() -> None:
    # A ridge holds the whole superlevel component above the saddle: here the kem
    # ridge spans two terminal leaves that merge at m before the common saddle.
    tree = _tree(
        [
            ("kem", 9.0),
            ("kem2", 8.0),
            ("m", 6.0),
            ("tcl", 8.5),
            ("S", 4.0),
            ("root", 0.0),
        ],
        [("kem", "m"), ("kem2", "m"), ("m", "S"), ("tcl", "S"), ("S", "root")],
        [("m", "S", "mk")],
    )
    vertices = {
        "kem": 9.0,
        "kem2": 8.0,
        "m": 6.0,
        "mk": 5.0,
        "tcl": 8.5,
        "S": 4.0,
        "root": 0.0,
    }
    classes = dict.fromkeys(vertices, "functional")
    ridges = _ridges(tree, ridge_vertices=vertices, ridge_classes=classes)
    members = ridges.members.filter(pl.col("ridge_name") == "kem")
    assert set(members["member_dna_sequence"]) == {"kem", "kem2", "m", "mk"}
    assert set(members["summit_dna_sequence"]) == {"kem"}


def test_one_absorbed_summit_keeps_membership_and_has_no_saddle() -> None:
    band = _tree(
        [("kem", 9.0), ("S", 4.0), ("root", 0.0)],
        [("kem", "S"), ("S", "root")],
    )
    ridges = _ridges(band, ridge_tree_state="band")
    assert ridges.members.height == len(VERTICES)
    assert ridges.members["ridge_name"].null_count() == len(VERTICES)
    assert ridges.summary["ridge_name"].to_list() == ["kem"]
    row = ridges.summary.row(0, named=True)
    assert row["common_saddle_dna_sequence"] is None
    assert row["common_saddle_fitness"] is None
    assert row["member_count"] == 0
    assert ridges.sidecar["ridges_present"] is False
    assert set(ridges.sidecar["reason"]) == {"tcl"}


def test_both_summits_absorbed_gives_no_summary_rows() -> None:
    band = _tree([("side", 7.0), ("root", 0.0)], [("side", "root")])
    ridges = _ridges(band, ridge_tree_state="band")
    assert ridges.summary.is_empty()
    assert ridges.members.height == len(VERTICES)
    assert set(ridges.sidecar["reason"]) == {"kem", "tcl"}


def test_band_only_absorption_still_reports_nominal_retention() -> None:
    band = _tree(
        [("kem", 9.0), ("S", 4.0), ("root", 0.0)],
        [("kem", "S"), ("S", "root")],
    )
    ridges = _ridges(band, ridge_tree_state="band", ridge_nominal=BASE)
    assert ridges.members.height == len(VERTICES)
    assert ridges.sidecar["nominal"]["ridges_present"] is True
    assert ridges.sidecar["nominal"]["ridges"]["tcl"]["band_member_count"] == 0
    assert ridges.sidecar["nominal"]["ridges"]["tcl"]["retained_count"] == 0


def test_functional_count_follows_the_class_rule_not_member_count() -> None:
    # t1 and k1 sit in ridges but below the threshold: counts must be lower than
    # member_count.
    classes = {**CLASSES, "k1": "non_functional", "t1": "non_functional"}
    rows = {
        r["ridge_name"]: r
        for r in _ridges(ridge_classes=classes).summary.iter_rows(named=True)
    }
    assert (rows["kem"]["member_count"], rows["kem"]["functional_count"]) == (2, 1)
    assert (rows["tcl"]["member_count"], rows["tcl"]["functional_count"]) == (2, 1)


def test_sidecar_separates_trunk_from_unassigned() -> None:
    composition = _ridges().sidecar["composition"]
    # Ridges kem,k1,tcl,t1; saddle S; trunk tr and root; other side and s1.
    assert composition == {"ridge": 4, "common_saddle": 1, "trunk": 2, "other": 2}


def test_tree_scalar_must_match_graph_raw_fitness() -> None:
    with pytest.raises(ValueError, match="differs from the graph raw fitness"):
        _ridges(ridge_vertices={**VERTICES, "S": 3.0})


def test_band_sidecar_records_retention_and_saddle_shift() -> None:
    band = _tree(
        [("kem", 9.0), ("tcl", 8.0), ("S", 4.0), ("root", 0.0)],
        [("kem", "S"), ("tcl", "S"), ("S", "root")],
        [("kem", "S", "k1")],
    )
    nominal = _tree(
        [("kem", 9.0), ("tcl", 8.0), ("M", 5.0), ("S", 4.0), ("root", 0.0)],
        [("kem", "M"), ("tcl", "M"), ("M", "S"), ("S", "root")],
        [("kem", "M", "k1"), ("tcl", "M", "t1")],
    )
    ridges = _ridges(band, ridge_tree_state="band", ridge_nominal=nominal)
    info = ridges.sidecar["nominal"]
    assert info["saddle_shifted"] is True
    assert info["common_saddle_dna_sequence"] == "M"
    assert ridges.sidecar["common_saddle_dna_sequence"] == "S"
    assert info["ridges"]["kem"] == {
        "nominal_member_count": 2,
        "band_member_count": 2,
        "retained_count": 2,
    }
    assert info["ridges"]["tcl"]["nominal_member_count"] == 2
    assert info["ridges"]["tcl"]["band_member_count"] == 1
    assert info["ridges"]["tcl"]["retained_count"] == 1


def test_schemas_match_the_spec() -> None:
    ridges = _ridges()
    assert ridges.summary.columns == [
        "population",
        "tree_state",
        "ridge_name",
        "summit_dna_sequence",
        "summit_raw_fitness",
        "common_saddle_dna_sequence",
        "common_saddle_fitness",
        "member_count",
        "functional_count",
        "functional_share",
    ]
    assert ridges.members.columns == [
        "population",
        "tree_state",
        "ridge_name",
        "summit_dna_sequence",
        "member_dna_sequence",
        "raw_fitness",
        "functional_class",
    ]


def test_every_vertex_needs_a_class() -> None:
    with pytest.raises(ValueError, match="functional class"):
        _ridges(ridge_classes={k: v for k, v in CLASSES.items() if k != "S"})


def test_ridges_require_their_inputs() -> None:
    with pytest.raises(ValueError, match="ridges requires"):
        describe(products={"ridges"})
