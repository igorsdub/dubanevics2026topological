"""Summarize threshold-scan stability without fitting models or making figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: None for key in ("min", "median", "q2_5", "q97_5", "max")}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "q2_5": float(np.quantile(array, 0.025)),
        "q97_5": float(np.quantile(array, 0.975)),
        "max": float(array.max()),
    }


def summarize(scan_dir: Path, classification_dir: Path) -> dict[str, Any]:
    paths = {
        name: scan_dir / name
        for name in (
            "resamples.parquet",
            "cells.parquet",
            "intervals.parquet",
            "acceptance.json",
            "provenance.json",
        )
    }
    paths.update(
        {
            "classification.parquet": classification_dir / "classification.parquet",
            "summary.json": classification_dir / "summary.json",
        }
    )
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    resamples = pl.read_parquet(paths["resamples.parquet"])
    cells = pl.read_parquet(paths["cells.parquet"])
    intervals = pl.read_parquet(paths["intervals.parquet"])
    acceptance = _json(paths["acceptance.json"])
    provenance = _json(paths["provenance.json"])
    canonical = pl.read_parquet(paths["classification.parquet"])
    nominal = _json(paths["summary.json"])

    required_class = {"dna_sequence", "s", "se_pooled", "functional_class"}
    if not required_class <= set(canonical.columns):
        raise ValueError(
            f"classification lacks columns {sorted(required_class - set(canonical.columns))}"
        )
    if (
        canonical["dna_sequence"].null_count()
        or canonical["dna_sequence"].n_unique() != canonical.height
    ):
        raise ValueError("classification DNA identities must be non-null and unique")
    if "sequence_id" in canonical.columns and (
        canonical["sequence_id"].null_count()
        or canonical["sequence_id"].n_unique() != canonical.height
    ):
        raise ValueError("classification sequence identities must be non-null and unique")
    se = canonical["se_pooled"].to_numpy()
    if not np.isfinite(se).all() or (se <= 0).any():
        raise ValueError("classification se_pooled must be finite and positive")
    s = canonical["s"].to_numpy()
    if not np.isfinite(s).all():
        raise ValueError("classification s must be finite")

    settings = provenance.get("scan_config", {})
    expected_draws = int(settings.get("n_resamples", -1))
    if expected_draws < 1:
        raise ValueError("scan provenance lacks a positive scan_config.n_resamples")
    draw_ids = resamples["resample"].unique().sort().to_list()
    if len(draw_ids) != expected_draws or draw_ids != list(range(expected_draws)):
        raise ValueError(f"expected resample indices 0..{expected_draws-1}, got {draw_ids}")

    # The production canonical point is explicitly the Gaussian, stop-anchored,
    # configured-quantile/confidence cell marked canonical.
    point = cells.filter(pl.col("is_canonical"))
    if point.height != 1:
        raise ValueError(f"expected one canonical scan cell, found {point.height}")
    point = point.row(0, named=True)
    chosen_k = int(nominal["mixture_k"])
    if point["family"] != "gaussian" or int(point["k"]) != chosen_k:
        raise ValueError("canonical scan cell disagrees with nominal Gaussian chosen k")
    nominal_t = float(nominal["threshold"]["rounded"])
    if float(point["threshold"]) != nominal_t:
        raise ValueError("canonical scan threshold disagrees with classification T")

    confidence = float(nominal["settings"]["functional_confidence"])
    probability = norm.cdf((s - nominal_t) / se)
    expected_labels = np.full(probability.shape, "uncertain", dtype=object)
    expected_labels[probability >= confidence] = "functional"
    expected_labels[probability <= 1.0 - confidence] = "non_functional"
    if not np.array_equal(expected_labels, canonical["functional_class"].to_numpy()):
        raise ValueError("canonical classes disagree with the nominal threshold formula")
    observed_class_counts = {
        "functional": 0,
        "non_functional": 0,
        "uncertain": 0,
    }
    observed_class_counts.update({
        row["functional_class"]: int(row["len"])
        for row in canonical.group_by("functional_class").len().iter_rows(named=True)
    })
    if observed_class_counts != nominal["class_counts"]:
        raise ValueError("classification class counts disagree with nominal summary")

    fixed_interval = intervals.filter(
        (pl.col("family") == "gaussian") & (pl.col("k") == chosen_k)
    )
    if fixed_interval.height != 1:
        raise ValueError("missing unique fixed-k Gaussian interval for nominal chosen k")

    gaussian = resamples.filter(pl.col("family") == "gaussian")
    selected = gaussian.filter(pl.col("is_resample_bic_chosen") == True)
    complete = gaussian.filter(pl.col("model_selection_complete") == True)
    expected_ks = list(range(int(settings["k_range"][0]), int(settings["k_range"][1]) + 1))
    for resample_id in range(expected_draws):
        draw = gaussian.filter(pl.col("resample") == resample_id)
        if draw["k"].sort().to_list() != expected_ks:
            raise ValueError(f"resample {resample_id} lacks complete Gaussian k coverage")
        if draw.filter(pl.col("is_resample_bic_chosen") == True).height != 1:
            raise ValueError(f"resample {resample_id} needs exactly one BIC choice")
        chosen_row = draw.filter(pl.col("is_resample_bic_chosen") == True).row(0, named=True)
        fit_rows = draw.iter_rows(named=True)
        expected_complete = all(
            bool(row["converged"])
            and row["bic"] is not None
            and np.isfinite(row["bic"])
            for row in fit_rows
        )
        complete_flags = draw["model_selection_complete"].unique().to_list()
        if len(complete_flags) != 1:
            raise ValueError(f"resample {resample_id} has inconsistent completeness flags")
        if bool(complete_flags[0]) != expected_complete:
            raise ValueError(f"resample {resample_id} completeness flag disagrees with fits")
        if expected_complete:
            expected_choice = min(
                draw.iter_rows(named=True), key=lambda row: (row["bic"], row["k"])
            )
            if chosen_row["k"] != expected_choice["k"]:
                raise ValueError(f"resample {resample_id} BIC choice is inconsistent")
    if selected["resample"].n_unique() != expected_draws:
        raise ValueError("need exactly one resample-BIC choice recorded for every attempt")
    selected_complete = selected.filter(pl.col("model_selection_complete") == True)

    class_labels = canonical["functional_class"].to_numpy()
    baseline = class_labels == "functional"
    rows: list[dict[str, Any]] = []
    for row in selected_complete.sort("resample").iter_rows(named=True):
        threshold = float(row["threshold"])
        if not np.isfinite(threshold):
            raise ValueError(f"non-finite selected threshold in resample {row['resample']}")
        current = norm.cdf((s - threshold) / se) >= confidence
        union = int(np.logical_or(baseline, current).sum())
        rows.append(
            {
                "resample": int(row["resample"]),
                "threshold": threshold,
                "k": int(row["k"]),
                "component_count": len(row["selected_components"] or []),
                "functional_size": int(current.sum()),
                "flips": int(np.logical_xor(baseline, current).sum()),
                "jaccard": (
                    float(np.logical_and(baseline, current).sum() / union)
                    if union
                    else 1.0
                ),
            }
        )

    thresholds = [row["threshold"] for row in rows]
    def counts(field: str) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in sorted(Counter(row[field] for row in rows).items())
        }

    complete_count = int(complete["resample"].n_unique())

    def value_range(field: str) -> list[int | float] | None:
        values = [row[field] for row in rows]
        return [min(values), max(values)] if values else None

    return {
        "scope": "conditional empirical-population stability",
        "interpretation": (
            "Resamples describe empirical-population stability conditional on the "
            "observed full fitness population; they are not a biological confidence "
            "interval. The separate fixed-graph 10,000-draw tree bootstrap does not "
            "propagate threshold T. With 200 attempts, each 2.5% tail is represented "
            "by about five draws."
        ),
        "nominal": {
            "T": nominal_t, "chosen_k": chosen_k,
            "fixed_k_gaussian_interval": fixed_interval.row(0, named=True),
            "classification_class_counts": observed_class_counts,
            "canonical_scan_threshold": float(point["threshold"]),
        },
        "bic_reselected_gaussian": {
            "attempted": expected_draws,
            "complete": complete_count,
            "incomplete": expected_draws - complete_count,
            "selected_threshold_distribution": _distribution(thresholds),
            "selected_k_frequencies": counts("k"),
            "selected_component_count_frequencies": counts("component_count"),
            "draws": rows,
            "functional_size_range": value_range("functional_size"),
            "flip_count_range": value_range("flips"),
            "jaccard_range": value_range("jaccard"),
        },
        "acceptance": acceptance,
        "provenance": {
            "input_sha256": {name: sha256(path) for name, path in paths.items()},
            "package_versions": {
                name: metadata.version(name)
                for name in ("polars", "numpy", "scipy", "scikit-learn")
            },
            "scan_provenance": provenance,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=Path("output/analysis/functional_threshold_scan"))
    parser.add_argument("--classification-dir", type=Path, default=Path("output/analysis/functional_classification"))
    parser.add_argument("--output", type=Path, default=Path(".context/threshold-stability-summary.json"))
    args = parser.parse_args()
    result = summarize(args.scan_dir, args.classification_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()