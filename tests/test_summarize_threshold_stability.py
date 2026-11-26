import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_threshold_stability.py"
_SPEC = importlib.util.spec_from_file_location("threshold_stability", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize = _MODULE.summarize


def _inputs(tmp_path):
    scan = tmp_path / "scan"
    classification = tmp_path / "classification"
    scan.mkdir()
    classification.mkdir()
    pl.DataFrame(
        {
            "resample": [0, 0, 1, 1, 2, 2],
            "family": ["gaussian"] * 6,
            "k": [1, 2] * 3,
            "threshold": [-1.0, -1.1, 0.0, 0.0, 0.5, 1.0],
            "converged": [True, True, True, True, True, False],
            "is_resample_bic_chosen": [True, False, False, True, True, False],
            "bic": [1.0, 2.0, 2.0, 1.0, 1.0, float("nan")],
            "selected_components": [[0], [0, 1]] * 3,
            "model_selection_complete": [True, True, True, True, False, False],
        }
    ).write_parquet(scan / "resamples.parquet")
    pl.DataFrame(
        {
            "family": ["gaussian"],
            "k": [1],
            "is_canonical": [True],
            "threshold": [0.0],
            "confidence": [0.95],
        }
    ).write_parquet(scan / "cells.parquet")
    pl.DataFrame(
        {
            "family": ["gaussian"],
            "k": [1],
            "threshold_low": [-0.2],
            "threshold_high": [0.2],
            "width": [0.4],
            "n_converged": [2],
            "n_resamples": [3],
        }
    ).write_parquet(scan / "intervals.parquet")
    (scan / "acceptance.json").write_text(json.dumps({"ok": True}))
    (scan / "provenance.json").write_text(
        json.dumps({"scan_config": {"n_resamples": 3, "k_range": [1, 2]}})
    )
    pl.DataFrame(
        {
            "dna_sequence": ["a", "b", "c"],
            "s": [0.0, 0.5, 1.0],
            "se_pooled": [0.1, 0.1, 0.1],
            "functional_class": ["uncertain", "functional", "functional"],
        }
    ).write_parquet(classification / "classification.parquet")
    (classification / "summary.json").write_text(
        json.dumps(
            {
                "threshold": {"rounded": 0.0, "unrounded": 0.0000001},
                "mixture_k": 1,
                "settings": {"functional_confidence": 0.95},
                "class_counts": {
                    "functional": 2,
                    "non_functional": 0,
                    "uncertain": 1,
                },
            }
        )
    )
    return scan, classification


def test_summary_uses_complete_bic_draws_and_threshold_flips(tmp_path):
    scan, classification = _inputs(tmp_path)
    result = summarize(scan, classification)
    stability = result["bic_reselected_gaussian"]
    assert (stability["attempted"], stability["complete"], stability["incomplete"]) == (
        3,
        2,
        1,
    )
    assert stability["selected_threshold_distribution"]["min"] == -1.0
    assert stability["selected_threshold_distribution"]["max"] == 0.0
    assert stability["selected_k_frequencies"] == {"1": 1, "2": 1}
    assert stability["selected_component_count_frequencies"] == {"1": 1, "2": 1}
    assert [draw["functional_size"] for draw in stability["draws"]] == [3, 2]
    assert [draw["flips"] for draw in stability["draws"]] == [1, 0]
    assert [draw["jaccard"] for draw in stability["draws"]] == [2 / 3, 1.0]
    assert set(result["provenance"]["input_sha256"]) == {
        "resamples.parquet",
        "cells.parquet",
        "intervals.parquet",
        "acceptance.json",
        "provenance.json",
        "classification.parquet",
        "summary.json",
    }


def test_all_incomplete_draws_have_empty_statistics(tmp_path):
    scan, classification = _inputs(tmp_path)
    frame = pl.read_parquet(scan / "resamples.parquet").with_columns(
        pl.lit(False).alias("model_selection_complete"),
        pl.lit(False).alias("converged"),
    )
    frame.write_parquet(scan / "resamples.parquet")
    stability = summarize(scan, classification)["bic_reselected_gaussian"]
    assert stability["complete"] == 0
    assert stability["draws"] == []
    assert stability["selected_threshold_distribution"]["median"] is None
    assert stability["functional_size_range"] is None


def test_rejects_false_complete_flag(tmp_path):
    scan, classification = _inputs(tmp_path)
    frame = pl.read_parquet(scan / "resamples.parquet").with_columns(
        pl.when(pl.col("resample") == 0)
        .then(False)
        .otherwise(pl.col("model_selection_complete"))
        .alias("model_selection_complete")
    )
    frame.write_parquet(scan / "resamples.parquet")
    with pytest.raises(ValueError, match="completeness flag disagrees"):
        summarize(scan, classification)


@pytest.mark.parametrize("failure", ["coverage", "class_counts"])
def test_rejects_incomplete_coverage_or_incorrect_nominal_counts(tmp_path, failure):
    scan, classification = _inputs(tmp_path)
    if failure == "coverage":
        pl.read_parquet(scan / "resamples.parquet").filter(
            ~((pl.col("resample") == 1) & (pl.col("k") == 2))
        ).write_parquet(scan / "resamples.parquet")
    else:
        summary_path = classification / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["class_counts"]["functional"] = 1
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        summarize(scan, classification)
