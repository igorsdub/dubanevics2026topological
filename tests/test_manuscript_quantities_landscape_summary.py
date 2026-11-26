"""Registry rows read from the saved landscape summary products (Ticket 15 S4).

Synthetic columns only: the Full and Published point-estimate states are the ones the
manuscript quotes, and every cross-check against another source raises with both numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_manuscript_quantities import CONFIG, _tree, fake_readers, write_upstream

from dubanevics2026topological.analysis import manuscript_quantities as mq

FULL, PUBLISHED = (
    mq.CITED_STATES["full"],
    mq.CITED_STATES["published"],
)


def columns(**published: Any) -> dict[str, dict[str, Any]]:
    """Saved values consistent with ``write_upstream``'s synthetic upstreams."""
    return {
        FULL: {
            "Number of nodes": 1000,
            "Number of components before largest component": 1,
            "Largest component": 1000,
            "Peak count": 6,
        },
        PUBLISHED: {
            "Number of nodes": 100,
            "Peak count": 12,
            "Synonymous leaves": 3,
            "Eligible leaves": 11,
            "Full library functional sequences": 300,
            "Full library functional sequences (%)": 30.0,
            "Full library non-functional sequences": 700,
            "Full library non-functional sequences (%)": 70.0,
            **published,
        },
    }


def products() -> dict[str, dict[str, str]]:
    return {
        name: {"label": f"label-{i}", "scientific_id_sha256": f"{i}" * 64}
        for i, name in enumerate((FULL, PUBLISHED))
    }


def rows(
    tmp_path: Path, summary: dict[str, dict[str, Any]] | None, **kwargs: Any
) -> dict[str, dict[str, Any]]:
    write_upstream(tmp_path, **kwargs)
    readers = fake_readers(tmp_path, _tree())
    if summary is not None:
        readers["landscape_summary"] = lambda: (summary, products())
    sources = mq.load_sources(tmp_path, readers=readers)
    built = mq.build_rows(sources, CONFIG, git_commit="abc123")
    return {row["key"]: row for row in built}


def test_full_components_and_leaf_counts_come_from_the_saved_full_column(
    tmp_path: Path,
) -> None:
    result = rows(tmp_path, columns())
    assert result["full_graph.component_count"]["value"] == "1"
    assert result["full_graph.largest_component_vertex_count"]["value"] == "1000"
    assert result["full_graph.leaf_count"]["value"] == "6"
    assert result["full_graph.component_count"]["source_stage"] == mq.LANDSCAPE_SUMMARY
    # The existing keys keep their sources and values.
    assert result["full_graph.peak_count"]["value"] == "6"
    assert result["full_graph.peak_count"]["source_stage"] == mq.CLAMPED
    assert result["published_graph.peak_count"]["value"] == "12"


def test_full_library_percentages_are_new_keys_beside_the_selected_graph_ones(
    tmp_path: Path,
) -> None:
    result = rows(tmp_path, columns())
    assert result["published_graph.full_library_functional_count"]["value"] == "300"
    assert (
        result["published_graph.full_library_functional_percent"]["display_value"]
        == "30.0"
    )
    assert result["published_graph.full_library_nonfunctional_count"]["value"] == "700"
    assert (
        result["published_graph.full_library_nonfunctional_percent"]["unit"]
        == "percent"
    )
    # The selected-graph percentage is the clamped summary's, unchanged (30 of 100).
    assert result["published_graph.functional_percent"]["value"] == "30.0"
    assert result["published_graph.functional_percent"]["source_stage"] == mq.CLAMPED
    assert result["published_graph.leaf_count"]["value"] == "12"


def test_without_the_saved_products_the_new_rows_are_named_gaps(
    tmp_path: Path,
) -> None:
    result = rows(tmp_path, None)
    for key in (
        "full_graph.component_count",
        "full_graph.largest_component_vertex_count",
        "full_graph.leaf_count",
        "published_graph.leaf_count",
        "published_graph.full_library_functional_percent",
    ):
        assert result[key]["value"] is None, key
        assert mq.is_stage_not_run(result[key]["gap_reason"]), key
    assert result["published_graph.peak_count"]["value"] == "12"


@pytest.mark.parametrize(
    ("state", "metric", "value", "match"),
    [
        ("published", "Peak count", 13, r"leaf count.*13.*peak count.*12"),
        ("full", "Peak count", 7, r"leaf count.*7.*peak count.*6"),
        ("published", "Synonymous leaves", 4, r"Synonymous leaves.*4.*3"),
        ("published", "Eligible leaves", 10, r"Eligible leaves.*10.*11"),
        ("published", "Number of nodes", 99, r"Number of nodes.*99.*100"),
        (
            "full",
            "Number of nodes",
            999,
            r"Largest component.*1000.*Number of nodes.*999",
        ),
    ],
)
def test_an_equality_check_that_fails_reports_both_numbers_and_stops(
    tmp_path: Path, state: str, metric: str, value: int, match: str
) -> None:
    saved = columns()
    saved[mq.CITED_STATES[state]][metric] = value
    with pytest.raises(ValueError, match=match) as error:
        rows(tmp_path, saved)
    assert "neither is chosen" in str(error.value)


def test_the_full_graph_must_be_one_component_with_every_vertex(
    tmp_path: Path,
) -> None:
    saved = columns()
    saved[FULL]["Largest component"] = 900
    with pytest.raises(ValueError, match=r"Largest component.*900.*1000"):
        rows(tmp_path, saved)


def test_the_source_identity_is_the_two_scientific_ids(tmp_path: Path) -> None:
    write_upstream(tmp_path)
    readers = fake_readers(tmp_path, _tree())
    readers["landscape_summary"] = lambda: (columns(), products())
    sources = mq.load_sources(tmp_path, readers=readers)
    identity = mq.source_identities(sources)["landscape_summary"]
    assert identity["stage"] == mq.LANDSCAPE_SUMMARY
    assert identity["identity"] == {
        FULL: "0" * 64,
        PUBLISHED: "1" * 64,
    }


def test_the_new_keys_are_in_the_manuscript_order_and_topics() -> None:
    for key in (
        "full_graph.leaf_count",
        "published_graph.leaf_count",
        "published_graph.full_library_functional_percent",
    ):
        assert key in mq.MANUSCRIPT_ORDER
        assert mq.topic_of(key) != mq.OTHER_TOPIC
