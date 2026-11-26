"""Application-owned positional Shannon entropy sweeps for the codon-entropy figure."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from dubanevics2026topological.analysis import OUTPUT


def _entropy(values: Iterable[str]) -> float | None:
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return None
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def positional_entropy(
    df: pl.DataFrame, positions: Iterable[int] = (1, 2, 3)
) -> dict[int, float | None]:
    """Entropy in bits of ``aa<position>`` over one population, null when it is empty."""
    return {
        position: _entropy(df.get_column(f"aa{position}").to_list())
        for position in positions
    }


def compute_positional_entropy_sweep(
    df: pl.DataFrame,
    num_thresholds: int = 100,
    *,
    thresholds: Iterable[float] | None = None,
) -> pl.DataFrame:
    """Compute per-position AA entropy on a threshold-retained raw-fitness set.

    When ``thresholds`` is supplied it is used verbatim, allowing populations to
    share the full study landscape's grid. Values outside this population's
    observed fitness range are emitted with a null entropy and count zero.
    """
    fitness_col = "fitness" if "fitness" in df.columns else "raw_fitness"
    required = {fitness_col, "aa1", "aa2", "aa3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Entropy input is missing columns: {sorted(missing)}")
    if num_thresholds < 1:
        raise ValueError("num_thresholds must be positive.")
    if df.is_empty():
        grid = list(thresholds) if thresholds is not None else []
        if not np.isfinite(np.asarray(grid, dtype=float)).all():
            raise ValueError("Entropy thresholds must be finite.")
        rows = [
            {
                "threshold_scalar": float(threshold),
                "aa_position": pos,
                "entropy_bits": None,
                "active_sequence_count": 0,
            }
            for threshold in grid
            for pos in (1, 2, 3)
        ]
        return pl.DataFrame(
            rows,
            schema={
                "threshold_scalar": pl.Float64,
                "aa_position": pl.Int64,
                "entropy_bits": pl.Float64,
                "active_sequence_count": pl.Int64,
            },
        )

    sorted_df = df.sort(fitness_col, descending=True)
    fitness = np.asarray(sorted_df.get_column(fitness_col).to_list(), dtype=float)
    if not np.isfinite(fitness).all():
        raise ValueError("Entropy input fitness values must be finite.")
    minimum, maximum = float(np.min(fitness)), float(np.max(fitness))
    grid = (
        [float(value) for value in thresholds]
        if thresholds is not None
        else [float(value) for value in np.linspace(minimum, maximum, num_thresholds)]
    )
    if not np.isfinite(np.asarray(grid, dtype=float)).all():
        raise ValueError("Entropy thresholds must be finite.")
    sequence_columns = [
        sorted_df.get_column(f"aa{position}").to_list() for position in (1, 2, 3)
    ]
    rows: list[dict[str, float | int | None]] = []
    for threshold in grid:
        in_range = minimum <= threshold <= maximum
        mask = fitness >= threshold if in_range else np.zeros(len(fitness), dtype=bool)
        active_count = int(mask.sum())
        for position, values in enumerate(sequence_columns, start=1):
            entropy = _entropy(
                value for value, keep in zip(values, mask, strict=True) if keep
            )
            rows.append(
                {
                    "threshold_scalar": threshold,
                    "aa_position": position,
                    "entropy_bits": entropy,
                    "active_sequence_count": active_count,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "threshold_scalar": pl.Float64,
            "aa_position": pl.Int64,
            "entropy_bits": pl.Float64,
            "active_sequence_count": pl.Int64,
        },
    )


def compute_entropy_sweeps(
    frames: dict[str, pl.DataFrame],
    *,
    num_thresholds: int = 100,
    thresholds: Iterable[float] | None = None,
) -> dict[str, pl.DataFrame]:
    """Sweep each population (``core``, ``kem``, ``tcl``) over one common grid.

    The grid is ``num_thresholds`` evenly spaced fitness values across the core
    population (or the given ``thresholds``), plus every population's own
    fitness extremes. Nothing is read or written.
    """
    if thresholds is None:
        core_fitness = frames["core"].get_column("fitness")
        if core_fitness.is_empty():
            shared_grid: list[float] = []
        else:
            shared_grid = [
                float(value)
                for value in np.linspace(
                    float(core_fitness.min()), float(core_fitness.max()), num_thresholds
                )
            ]
    else:
        shared_grid = [float(value) for value in thresholds]
        if not np.isfinite(np.asarray(shared_grid, dtype=float)).all():
            raise ValueError("Entropy thresholds must be finite.")
    endpoints: set[float] = set(shared_grid)
    for frame in frames.values():
        values = frame.get_column("fitness")
        if values.is_empty():
            continue
        endpoints.add(float(values.min()))
        endpoints.add(float(values.max()))
    shared_grid = sorted(endpoints)

    return {
        key: compute_positional_entropy_sweep(frame, thresholds=shared_grid)
        for key, frame in frames.items()
    }


def process_codon_entropy_datasets(
    *,
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    save_files: bool = True,
    num_thresholds: int = 100,
    thresholds: Iterable[float] | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute all population sweeps using one common landscape-wide grid."""
    tables_in = Path(input_dir) if input_dir else OUTPUT / "tables"
    tables_out = Path(output_dir) if output_dir else OUTPUT / "tables"
    datasets = {
        "core": ("papkou2023_candidate_population.parquet", "papkou2023_candidate_aa_entropy"),
        "kem": ("papkou2023_kem_ridge_population.parquet", "papkou2023_kem_aa_entropy"),
        "tcl": ("papkou2023_tcl_ridge_population.parquet", "papkou2023_tcl_aa_entropy"),
    }
    frames: dict[str, pl.DataFrame] = {}
    for key, (filename, _) in datasets.items():
        path = tables_in / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required input population file not found: {path}")
        frames[key] = pl.read_parquet(path)

    results = compute_entropy_sweeps(
        frames, num_thresholds=num_thresholds, thresholds=thresholds
    )
    for key, (_, output_stem) in datasets.items():
        entropy = results[key]
        if save_files:
            tables_out.mkdir(parents=True, exist_ok=True)
            entropy.write_parquet(tables_out / f"{output_stem}.parquet")
            entropy.write_csv(tables_out / f"{output_stem}.csv")
    return results
