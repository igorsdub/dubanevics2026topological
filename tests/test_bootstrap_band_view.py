"""Band views: surviving/absorbed marking, the 2c loader and identity labels."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.build.d10_band_inputs import (
    load_two_c,
    summary_sha256,
)
from dubanevics2026topological.figures.band_survival import (
    classify_peaks,
    peak_nodes,
)
from dubanevics2026topological.figures.view_naming import single_result_figure_name
from dubanevics2026topological.figures.tree_comparison_view import (
    identity_label_layers,
    place_identity_labels,
)


def _pairs(births, persistence) -> pl.DataFrame:
    return pl.DataFrame({"birth_label": births, "persistence": persistence})


NOMINAL = _pairs(["A", "B", "C", "D"], [3.0, 1.2, 0.9, 0.4])


def test_peaks_above_2c_survive_and_the_rest_are_absorbed() -> None:
    survival = classify_peaks(NOMINAL, _pairs(["A", "B"], [3.0, 1.2]), 1.0)
    assert survival.surviving == {"A", "B"}
    assert survival.absorbed == {"C", "D"}
    assert survival.nominal_peaks == 4
    assert survival.two_c == 1.0


def test_a_peak_exactly_at_2c_is_absorbed() -> None:
    survival = classify_peaks(NOMINAL, _pairs(["A"], [3.0]), 1.2)
    assert survival.absorbed == {"B", "C", "D"}


def test_band_tree_must_agree_with_2c() -> None:
    with pytest.raises(ValueError, match="persistence > 2c"):
        classify_peaks(NOMINAL, _pairs(["A"], [3.0]), 1.0)


def test_band_peaks_must_be_nominal_peaks() -> None:
    with pytest.raises(ValueError, match="nominal tree lacks"):
        classify_peaks(NOMINAL, _pairs(["A", "Z"], [3.0, 1.0]), 1.0)


def test_band_births_must_be_unique() -> None:
    with pytest.raises(ValueError, match="repeat"):
        classify_peaks(NOMINAL, _pairs(["A", "A"], [3.0, 3.0]), 1.0)


def _nodes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.9, 0.5, 0.3, 0.1],
            "dna_sequence": ["AAA", "AAC", "AAG", "AAT"],
            "aa_sequence": ["K", "N", "K", "N"],
            "tree_node_index": [0, 1, 2, 3],
        }
    )


def test_peak_nodes_gives_each_peak_its_sequences_and_position() -> None:
    nodes = peak_nodes(_nodes(), frozenset({"AAC", "AAT"}))
    assert sorted(nodes.columns) == ["aa_sequence", "dna_sequence", "x", "y"]
    assert dict(nodes.select("dna_sequence", "aa_sequence").iter_rows()) == {
        "AAC": "N",
        "AAT": "N",
    }


def test_peak_nodes_refuses_a_peak_absent_from_the_layout() -> None:
    with pytest.raises(ValueError, match="exactly one node"):
        peak_nodes(_nodes(), frozenset({"AAC", "GGG"}))


def test_identity_labels_carry_amino_acid_and_dna() -> None:
    peaks = peak_nodes(_nodes(), frozenset({"AAC"}))
    (layer,) = identity_label_layers(peaks, {})
    assert layer.data["identity"].to_list() == ["N AAC"]


def test_overlapping_identity_labels_are_raised_and_none_are_dropped() -> None:
    # Two peaks closer than a rotated label is thick, at the same height.
    peaks = pl.DataFrame(
        {
            "x": [0.50, 0.51],
            "y": [1.0, 1.0],
            "aa_sequence": ["KEM", "AQR"],
            "dna_sequence": ["AAGGAAATG", "GCAACAATGA"],
        }
    )
    placed = place_identity_labels(
        peaks,
        {"peak_label_font_size": 8},
        x_domain=(0.0, 1.0),
        y_domain=(0.0, 2.0),
        width=420,
        height=250,
    )
    assert set(placed["identity"].to_list()) == {
        "KEM AAGGAAATG",
        "AQR GCAACAATGA",
    }
    assert placed["raised"].to_list().count(True) == 1
    assert placed["y_text"].max() > placed["y"].max()
    layers = identity_label_layers(
        peaks,
        {"peak_label_font_size": 8},
        x_domain=(0.0, 1.0),
        y_domain=(0.0, 2.0),
        width=420,
        height=250,
    )
    kinds = [
        layer.mark.type if hasattr(layer.mark, "type") else layer.mark
        for layer in layers
    ]
    assert kinds == ["rule", "text"]


def _manifest(sha: str) -> SimpleNamespace:
    artifact = {"checkpoint/summary.json": {"sha256": sha}}
    return SimpleNamespace(
        parents=[
            {"product_type": "tree"},
            {"source_lineage": {"source_artifacts": artifact}},
        ]
    )


def _summary(directory, two_c: float) -> str:
    directory.mkdir(parents=True)
    text = json.dumps({"band_simplification": {"two_c": two_c}})
    (directory / "summary.json").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_two_c_is_found_by_the_manifest_hash(tmp_path) -> None:
    _summary(tmp_path / "other", 9.0)
    sha = _summary(tmp_path / "mine", 0.46)
    assert summary_sha256(_manifest(sha)) == sha
    assert load_two_c(_manifest(sha), [tmp_path]) == 0.46


def test_two_c_fails_when_no_summary_matches(tmp_path) -> None:
    _summary(tmp_path / "other", 9.0)
    with pytest.raises(ValueError, match="No checkpoint summary"):
        load_two_c(_manifest("0" * 64), [tmp_path])
    with pytest.raises(ValueError, match="No checkpoint summary"):
        load_two_c(_manifest("0" * 64), [tmp_path / "missing"])


def test_manifest_without_summary_hash_is_refused() -> None:
    with pytest.raises(ValueError, match="records no checkpoint"):
        summary_sha256(SimpleNamespace(parents=[{"product_type": "tree"}]))


def test_bootstrap_band_names_use_the_band_key_set() -> None:
    params = {
        "thr": "published",
        "fdr": "by",
        "q": 0.05,
        "draws": 10000,
        "seed": 20261019,
        "tree": "band",
        "bandq": 0.95,
        "hash": "9b3f0432",
    }
    name = single_result_figure_name("bootstrap_band", params, "annotated", "png")
    # The naming API uses the view name; the canonical shelf strips its band prefix.
    from dubanevics2026topological.output_paths import shelf_stem

    assert name.startswith("bootstrap_band__thr=published_fdr=by_q=0.05_draws=10000")
    assert shelf_stem(name).startswith("bootstrap__thr=published_fdr=by_q=0.05_draws=10000")
    assert name.endswith("_hash=9b3f0432_style=annotated.png")


def test_the_stops_and_synonymous_files_have_their_own_shelf_stems() -> None:
    from dubanevics2026topological.output_paths import shelf_stem

    params = {
        "thr": "reflected",
        "fdr": "by",
        "q": 0.05,
        "draws": 10000,
        "seed": 20261019,
        "tree": "band",
        "bandq": 0.95,
        "hash": "9b3f0432",
    }
    tail = (
        "thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95"
        "_style=annotated"
    )
    for kind in ("stops", "synonymous"):
        name = single_result_figure_name(
            f"bootstrap_band_{kind}", params, "annotated", "png"
        )
        assert name.startswith(f"bootstrap_band_{kind}__thr=reflected")
        assert shelf_stem(name.removesuffix(".png").replace("_hash=9b3f0432", "")) == (
            f"bootstrap_{kind}__{tail}"
        )
