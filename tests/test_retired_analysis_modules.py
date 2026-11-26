"""The temporary analysis forwarding modules have been retired."""

import importlib.util

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bootstrap_chunks",
        "checkpoint_identity",
        "bootstrap_slices",
        "bootstrap_smoothed_tree",
        "candidate_selection",
        "candidate_threshold",
        "clamped_landscapes",
        "classification_probability_scans",
        "fdr_selection",
        "full_population",
        "functional_classification",
        "functional_classification_config",
        "selected_leaf_simplification",
    ],
)
def test_retired_forwarding_module_is_absent(name: str) -> None:
    assert (
        importlib.util.find_spec(f"dubanevics2026topological.analysis.{name}") is None
    )


@pytest.mark.parametrize(
    ("module", "names"),
    [
        ("graph_summary", ("summarize_network", "summary_payload")),
        (
            "biological_annotations",
            ("build_biological_annotations", "export_biological_annotations"),
        ),
        (
            "band_view_ridges",
            (
                "band_view_ridge_counts",
                "load_band_view_ridge_counts",
                "in_memory_stop_codon_peak_count",
            ),
        ),
    ],
)
def test_retired_description_aliases_are_absent(
    module: str, names: tuple[str, ...]
) -> None:
    from importlib import import_module

    implementation = import_module(f"dubanevics2026topological.analysis.{module}")
    assert all(not hasattr(implementation, name) for name in names)
