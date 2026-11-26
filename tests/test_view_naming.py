"""Story-view catalog and figure file naming grammar (Spec 2, Sections 2 and 6)."""

from __future__ import annotations

import json

import pytest

from dubanevics2026topological.figures.view_naming import (
    D10_VIEWS,
    comparison_figure_name,
    comparison_hash,
    comparison_sidecar_name,
    figure_relative_path,
    single_result_figure_name,
    stable_figure_path,
    write_comparison_sidecar,
)
from dubanevics2026topological.results.labels import MalformedLabelError

BAND = {
    "thr": "published",
    "fdr": "by",
    "q": 0.05,
    "draws": 10000,
    "seed": 20261019,
    "tree": "band",
    "bandq": 0.95,
    "hash": "a1b2c3d4",
}
FULL = {"graph": "full", "hash": "a1b2c3d4"}


def test_catalog_lists_the_spec_views() -> None:
    assert set(D10_VIEWS) == {
        "tree_persistence",
        "tree_persistence_combined",
        "fitness_histogram",
        "tree_comparison_thresholds",
        "fitness_uncertainty",
        "tree_comparison_fdr",
        "bootstrap_band",
        "bootstrap_band_stops",
        "bootstrap_band_synonymous",
        "bootstrap_band_combined",
        "tree_comparison_band",
        "leaves",
        "band_quantile_scan",
        "ridge_entropy",
        "graph_codon_entropy",
        "edge_evidence",
    }
    assert {n for n, v in D10_VIEWS.items() if v.kind == "comparison"} == {
        "tree_comparison_thresholds",
        "tree_comparison_fdr",
        "tree_comparison_band",
        "graph_codon_entropy",
    }


PENDING = {
    "fitness_histogram": "pending_landmark_product",
}


def test_blocked_views_are_pending_placeholders() -> None:
    for name, view in D10_VIEWS.items():
        assert view.status == PENDING.get(name, "available")


@pytest.mark.parametrize(
    ("view", "params", "style", "expected"),
    [
        (
            "tree_persistence",
            {"graph": "full", "tree": "nominal", "hash": "a1b2c3d4"},
            "annotated",
            (
                "tree_persistence__graph=full_tree=nominal_hash=a1b2c3d4"
                "_style=annotated.png"
            ),
        ),
        (
            "tree_persistence",
            {"thr": "reflected", "fdr": "none", "tree": "nominal", "hash": "a1b2c3d4"},
            "annotated",
            (
                "tree_persistence__thr=reflected_fdr=none_tree=nominal_hash=a1b2c3d4"
                "_style=annotated.png"
            ),
        ),
        (
            "fitness_histogram",
            FULL,
            "vanilla",
            "fitness_histogram__graph=full_hash=a1b2c3d4_style=vanilla.png",
        ),
        (
            "fitness_uncertainty",
            {"thr": "reflected", "fdr": "by", "q": 0.05, "hash": "a1b2c3d4"},
            "vanilla",
            (
                "fitness_uncertainty__thr=reflected_fdr=by_q=0.05_hash=a1b2c3d4"
                "_style=vanilla.png"
            ),
        ),
        (
            "bootstrap_band",
            BAND,
            "annotated",
            (
                "bootstrap_band__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_"
                "tree=band_bandq=0.95_hash=a1b2c3d4_style=annotated.png"
            ),
        ),
        (
            "bootstrap_band_stops",
            BAND,
            "annotated",
            (
                "bootstrap_band_stops__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_"
                "tree=band_bandq=0.95_hash=a1b2c3d4_style=annotated.png"
            ),
        ),
        (
            "bootstrap_band_synonymous",
            BAND,
            "annotated",
            (
                "bootstrap_band_synonymous__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_"
                "tree=band_bandq=0.95_hash=a1b2c3d4_style=annotated.png"
            ),
        ),
        (
            "band_quantile_scan",
            {k: v for k, v in BAND.items() if k not in ("tree", "bandq")},
            "annotated",
            (
                "band_quantile_scan__thr=published_fdr=by_q=0.05_draws=10000_"
                "seed=20261019_hash=a1b2c3d4_style=annotated.png"
            ),
        ),
        (
            "edge_evidence",
            {"thr": "published", "fdr": "by", "q": 0.05, "hash": "a1b2c3d4"},
            "vanilla",
            (
                "edge_evidence__thr=published_fdr=by_q=0.05_hash=a1b2c3d4"
                "_style=vanilla.png"
            ),
        ),
    ],
)
def test_spec_examples(view, params, style, expected) -> None:
    assert single_result_figure_name(view, params, style, "png") == expected


def test_key_order_is_pipeline_order_whatever_the_dict_order() -> None:
    reordered = dict(reversed(list(BAND.items())))
    assert single_result_figure_name(
        "bootstrap_band", reordered, "vanilla", "svg"
    ) == single_result_figure_name("bootstrap_band", BAND, "vanilla", "svg")


@pytest.mark.parametrize(
    ("view", "params"),
    [
        ("tree_persistence", {"graph": "full", "hash": "a1b2c3d4"}),
        (
            "tree_persistence",
            {"thr": "published", "tree": "nominal", "hash": "a1b2c3d4"},
        ),
        ("fitness_histogram", {"graph": "full"}),
        ("bootstrap_band", {k: v for k, v in BAND.items() if k != "draws"}),
        ("fitness_histogram", {**FULL, "host": "local"}),
        ("fitness_histogram", {**FULL, "run": 5}),
        ("band_quantile_scan", BAND),
        ("fitness_uncertainty", {"thr": "published", "q": 0.05, "hash": "a1b2c3d4"}),
        ("bootstrap_band", {**BAND, "tree": "nominal"}),
        (
            "leaves",
            {"thr": "published", "fdr": "none", "tree": "band", "hash": "a1b2c3d4"},
        ),
        ("leaves", {**BAND, "tree": "nominal"}),
    ],
)
def test_missing_or_extra_keys_are_refused(view, params) -> None:
    with pytest.raises(MalformedLabelError):
        single_result_figure_name(view, params, "vanilla", "png")


def test_style_view_and_extension_are_validated() -> None:
    for view, style, ext, params in [
        ("fitness_histogram", "fancy", "png", FULL),
        ("fitness_histogram", "vanilla", "png", {**FULL, "style": "vanilla"}),
        ("no_such_view", "vanilla", "png", FULL),
        ("tree_comparison_fdr", "vanilla", "png", FULL),
        ("fitness_histogram", "vanilla", "p/ng", FULL),
    ]:
        with pytest.raises(MalformedLabelError):
            single_result_figure_name(view, params, style, ext)


def test_leaves_view_may_carry_style_despite_the_leaves_product() -> None:
    assert single_result_figure_name(
        "leaves",
        {"thr": "published", "fdr": "none", "tree": "nominal", "hash": "a1b2c3d4"},
        "vanilla",
        "png",
    ) == ("leaves__thr=published_fdr=none_tree=nominal_hash=a1b2c3d4_style=vanilla.png")


def test_style_never_enters_the_scientific_hash_segment() -> None:
    a = single_result_figure_name("fitness_histogram", FULL, "vanilla", "png")
    b = single_result_figure_name("fitness_histogram", FULL, "annotated", "png")
    assert a.replace("vanilla", "annotated") == b


def test_comparison_names_and_sidecar(tmp_path) -> None:
    digests = ["a" * 64, "b" * 64, "c" * 64]
    spec = {"slots": ["full", "published", "reflected"]}
    h = comparison_hash(digests, spec)
    assert len(h) == 8 and h == comparison_hash(digests, spec)
    assert comparison_hash(digests, {"b": 1, "a": 2}) == comparison_hash(
        digests, {"a": 2, "b": 1}
    )
    assert h != comparison_hash(list(reversed(digests)), spec)
    assert h != comparison_hash(digests, {"slots": ["full"]})
    view = "tree_comparison_thresholds"
    assert (
        comparison_figure_name(view, "main", h, "png") == f"{view}_main__hash={h}.png"
    )
    assert comparison_sidecar_name(view, "main", h) == f"{view}_main__hash={h}.json"
    path = tmp_path / "sidecar.json"
    label = "tree__graph=full_tree=nominal_hash=aaaaaaaa"
    write_comparison_sidecar(
        path,
        view=view,
        comparison="main",
        inputs=[(label, digests[0])],
        commit="deadbeef",
    )
    assert json.loads(path.read_text()) == {
        "view": view,
        "comparison": "main",
        "inputs": [{"label": label, "sha256": digests[0]}],
        "commit": "deadbeef",
    }


def test_comparison_names_refuse_bad_inputs() -> None:
    for args in [
        ("fitness_histogram", "main", "a1b2c3d4"),
        ("tree_comparison_fdr", "ma_in", "a1b2c3d4"),
        ("tree_comparison_fdr", "main", "XYZ"),
    ]:
        with pytest.raises(MalformedLabelError):
            comparison_figure_name(*args, "png")
    with pytest.raises(MalformedLabelError):
        comparison_hash(["short"], {})


def test_directory_placement_is_by_view() -> None:
    name = "fitness_histogram__graph=full_hash=a1b2c3d4_style=vanilla.png"
    assert figure_relative_path("fitness_histogram", name).as_posix() == (
        f"figures/fitness_histogram/{name}"
    )
    with pytest.raises(MalformedLabelError):
        figure_relative_path("fitness_histogram", "tree_persistence__x.png")


def test_directory_placement_validates_the_whole_name() -> None:
    ok = "leaves__thr=published_fdr=none_tree=nominal_hash=a1b2c3d4_style=vanilla.png"
    assert figure_relative_path("leaves", ok).parts[:2] == ("figures", "leaves")
    cmp_name = "tree_comparison_fdr_main__hash=12345678"
    assert figure_relative_path("tree_comparison_fdr", cmp_name + ".png")
    assert figure_relative_path("tree_comparison_fdr", cmp_name + ".json")
    for view, name in [
        ("leaves", "leaves__junk.png"),
        ("leaves", ok.replace("_style=vanilla", "")),
        ("leaves", ok.replace("tree=nominal", "tree=band")),
        ("leaves", "leaves_x__hash=12345678.png"),
        ("tree_comparison_fdr", "tree_comparison_fdr_main__hash=zz.png"),
        ("tree_comparison_fdr", "tree_comparison_fdr_ma_in__hash=12345678.png"),
        ("leaves", "leaves"),
    ]:
        with pytest.raises(MalformedLabelError):
            figure_relative_path(view, name)


def test_single_result_figures_have_no_json_sidecar() -> None:
    name = "leaves__thr=published_fdr=none_tree=nominal_hash=a1b2c3d4_style=vanilla"
    with pytest.raises(MalformedLabelError, match="no sidecar"):
        figure_relative_path("leaves", name + ".json")


def test_stable_paths_are_flat_and_hash_free(tmp_path) -> None:
    single = (
        "leaves__thr=published_fdr=none_tree=nominal_hash=a1b2c3d4_style=vanilla.png"
    )
    assert stable_figure_path(tmp_path, "leaves", single) == (
        tmp_path
        / "figures"
        / "canonical"
        / "leaves__thr=published_fdr=none_tree=nominal_style=vanilla.png"
    )
    comparison = "tree_comparison_fdr_main__hash=12345678.json"
    assert stable_figure_path(tmp_path, "tree_comparison_fdr", comparison) == (
        tmp_path / "figures" / "canonical" / "tree_comparison_fdr_main.json"
    )
    with pytest.raises(MalformedLabelError):
        stable_figure_path(tmp_path, "leaves", "leaves__junk.png")


def test_declared_comparisons_get_distinct_stable_paths(tmp_path) -> None:
    # Declared ids are unique config keys (``[comparisons.<view>_<name>]``), so the
    # hash-free stem ``<view>_<name>`` is unique per declared comparison; the same
    # id re-run on other inputs overwrites by design (the sidecar keeps the digests).
    digests = ("1" * 8, "2" * 8)
    paths = {
        stable_figure_path(
            tmp_path, view, comparison_figure_name(view, name, digest, "png")
        )
        for view in ("tree_comparison_thresholds", "tree_comparison_fdr")
        for name in ("main", "other")
        for digest in digests[:1]
    }
    assert len(paths) == 4
    assert all("hash" not in p.name for p in paths)
