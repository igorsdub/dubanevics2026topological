"""The display labels: one module, repeated verbatim where TOML cannot import it."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from dubanevics2026topological.figures import labels

FIGURES_TOML = Path(__file__).resolve().parents[1] / "config" / "figures.toml"


@pytest.fixture(scope="module")
def figures_config() -> dict:
    with FIGURES_TOML.open("rb") as handle:
        return tomllib.load(handle)


def test_state_labels_cover_every_state() -> None:
    assert labels.STATE_LABELS == {
        "full": "Full",
        "published": "Papkou",
        "published_by": "Papkou + FDR",
        "published_by_band": "Papkou + FDR, bootstrapped",
        "reflected": "Topographer",
        "reflected_by": "Topographer + FDR",
        "reflected_by_band": "Topographer + FDR, bootstrapped",
    }


def test_graph_and_threshold_labels() -> None:
    assert labels.graph_label("published") == "Papkou graph"
    assert labels.graph_label("reflected_by") == "Topographer + FDR graph"
    assert labels.threshold_label("published") == "Papkou threshold"
    assert labels.threshold_label("reflected") == "Topographer threshold"
    with pytest.raises(KeyError):
        labels.state_label("candidate")


def test_landscape_summary_pins_are_identity_names_not_display_labels(
    figures_config: dict,
) -> None:
    """The pinned column names are product identity: the display labels never enter them."""
    from dubanevics2026topological.analysis.graph_summary import SUMMARY_COLUMNS

    table = figures_config["tables"]["landscape_summary"]["labels"]
    assert tuple(table) == SUMMARY_COLUMNS
    assert set(table.values()) == set(labels.STATE_LABELS)
    assert not set(table) & set(labels.STATE_LABELS.values())


def test_landmark_labels_match_module(figures_config: dict) -> None:
    landmarks = figures_config["figures"]["full_tree_fitness_landmarks"]["labels"]
    assert landmarks["published_functional_threshold"] == labels.PAPKOU_THRESHOLD
    assert landmarks["candidate_functional_threshold"] == labels.TOPOGRAPHER_THRESHOLD


def _values_of(node: object, key: str) -> list:
    """Every value stored under ``key`` anywhere in a nested TOML table."""
    found: list = []
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                found.append(value)
            found.extend(_values_of(value, key))
    return found


def test_row_and_branch_labels_match_module(figures_config: dict) -> None:
    assert _values_of(figures_config, "row_labels") == [
        [labels.graph_label("published"), labels.graph_label("reflected")]
    ]
    assert _values_of(figures_config, "ridge_labels") == [
        [labels.KEM_BRANCH, labels.TCL_BRANCH]
    ]


RETIRED_WORDING = (
    "median-reflected",
    "candidate graph",
    "published graph",
    "candidate threshold",
    "published threshold",
    "ridge",
    "band-simplified",
)


def test_no_retired_wording_in_config_display_values(figures_config: dict) -> None:
    """Display values only: the landscape-summary pins are identity keys and are skipped."""

    def strings(node: object, path: tuple[str, ...] = ()):
        if path[:3] == ("tables", "landscape_summary", "labels"):
            return
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield from strings(value, (*path, key))
        elif isinstance(node, list):
            for value in node:
                yield from strings(value, path)

    for text in strings(figures_config):
        lowered = text.lower()
        for word in RETIRED_WORDING:
            assert word not in lowered, (word, text)


def test_summary_display_maps_identity_columns_to_state_labels_in_order() -> None:
    from dubanevics2026topological.analysis.graph_summary import SUMMARY_COLUMNS
    from dubanevics2026topological.tables.graph_summary import SUMMARY_DISPLAY

    order = (
        "full",
        "published",
        "reflected",
        "published_by",
        "published_by_band",
        "reflected_by",
        "reflected_by_band",
    )
    assert tuple(SUMMARY_DISPLAY) == SUMMARY_COLUMNS
    assert tuple(SUMMARY_DISPLAY.values()) == tuple(labels.state_label(s) for s in order)


def test_scan_panel_headings_match_module() -> None:
    from dubanevics2026topological.analysis.band_quantile_scan_panels import PANELS

    assert [panel.row_title for panel in PANELS] == [
        labels.graph_label(panel.state) for panel in PANELS
    ]


def test_other_config_labels_match_module(figures_config: dict) -> None:
    apex = _values_of(figures_config, "threshold_apex_label")
    assert apex
    assert {text.replace("\n", " ") for text in apex} == {labels.TOPOGRAPHER_THRESHOLD}
    assert set(_values_of(figures_config, "candidate_title")) == {
        labels.TOPOGRAPHER_FDR_BOOTSTRAPPED
    }
