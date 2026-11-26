"""``just make-raw``: rebuild the five files of data/raw/ from the Papkou store.

R (scripts/r/rds_to_interim.R) reads the RDS files and writes CSV and GraphML into a
temporary directory; polars does every transform and writes Parquet. A shape check runs
at the end against the counts in config/data.toml (recorded in data/raw/README.md).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import polars as pl

from dubanevics2026topological.rawdata import files as fs
from dubanevics2026topological.rawdata.papkou_fetch import store_path
from dubanevics2026topological.rawdata.value_digest import (
    bytes_sha256,
    graphml_value_digest,
    parquet_value_digest,
    peaks_value_digest,
)

WT_DNA = "GCCGATCTC"
STOP_SYMBOLS = {"TAA": "%", "TAG": "^", "TGA": "&"}
_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: dict[str, str] = {
    a + b + c: _AMINO[16 * i + 4 * j + k]
    for i, a in enumerate(_BASES)
    for j, b in enumerate(_BASES)
    for k, c in enumerate(_BASES)
}
CODON_TABLE.update(STOP_SYMBOLS)

RAW_FILES = (
    "papkou2023_fitness_wt.parquet",
    "papkou2023_fitness_diff.parquet",
    "papkou2023_graph.graphml",
    "papkou2023_peaks.csv",
)
FITNESS_SCHEMA = {
    "dna_from": pl.String,
    "dna_to": pl.String,
    "aa_from": pl.String,
    "aa_to": pl.String,
    "s": pl.Float64,
    "se": pl.Float64,
    "pval": pl.Float64,
}
R_SCRIPT = fs.REPO_ROOT / "scripts" / "r" / "rds_to_interim.R"


def translate_expr(column: str) -> pl.Expr:
    """Standard-code translation of a 9-nt DNA column, codon by codon (stops as % ^ &)."""
    codon = lambda i: pl.col(column).str.slice(3 * i, 3).replace_strict(CODON_TABLE)
    return pl.concat_str([codon(0), codon(1), codon(2)])


def read_interim(path: Path, text_columns: tuple[str, ...]) -> pl.DataFrame:
    """Read an R interim CSV: ``text_columns`` as strings, all others as Float64."""
    raw = pl.read_csv(path, infer_schema=False, null_values=["", "NA"])
    return raw.with_columns(
        [pl.col(c).cast(pl.Float64) for c in raw.columns if c not in text_columns]
    )


def _with_aa(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        aa_from=translate_expr("dna_from"), aa_to=translate_expr("dna_to")
    ).select(list(FITNESS_SCHEMA))


def transform_wt(table: pl.DataFrame) -> pl.DataFrame:
    """fitness_data_wt (SV, m, m_se, m_p.value) -> wt-to-variant rows, WT row dropped."""
    return _with_aa(
        table.filter(pl.col("SV") != WT_DNA).select(
            dna_from=pl.lit(WT_DNA),
            dna_to=pl.col("SV"),
            s=pl.col("m"),
            se=pl.col("m_se"),
            pval=pl.col("m_p.value"),
        )
    )


def transform_diff(table: pl.DataFrame) -> pl.DataFrame:
    """fitness_diff (from, to, m, m_se, m_p.value) -> edges oriented by sign, s = |m|."""
    swap = (pl.col("m") < 0).fill_null(False)
    return _with_aa(
        table.select(
            dna_from=pl.when(swap).then(pl.col("to")).otherwise(pl.col("from")),
            dna_to=pl.when(swap).then(pl.col("from")).otherwise(pl.col("to")),
            s=pl.col("m").abs(),
            se=pl.col("m_se"),
            pval=pl.col("m_p.value"),
        )
    )


def peaks_from_nodes(nodes: pl.DataFrame) -> pl.DataFrame:
    """Single column ``dna``: nodes with out-degree 0, sorted."""
    return (
        nodes.filter(pl.col("out_degree").cast(pl.Int64) == 0)
        .select(dna=pl.col("name"))
        .sort("dna")
    )


def run_r(in_data: Path, interim: Path, rscript: str = "Rscript") -> None:
    if shutil.which(rscript) is None:
        raise SystemExit(f"{rscript} not found: R with the igraph package is required.")
    probe = subprocess.run(
        [rscript, "-e", 'quit(status = !requireNamespace("igraph", quietly = TRUE))'],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "The R package igraph is not installed: install.packages('igraph')."
        )
    subprocess.run([rscript, str(R_SCRIPT), str(in_data), str(interim)], check=True)


def _write_atomic(write, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    write(tmp)
    os.replace(tmp, dest)


def write_outputs(interim: Path, out_dir: Path) -> None:
    """Transform the interim files and write the four raw files into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wt = transform_wt(read_interim(interim / "fitness_wt.csv", ("SV",)))
    _write_atomic(wt.write_parquet, out_dir / "papkou2023_fitness_wt.parquet")
    del wt
    diff = transform_diff(read_interim(interim / "fitness_diff.csv", ("from", "to")))
    _write_atomic(diff.write_parquet, out_dir / "papkou2023_fitness_diff.parquet")
    del diff
    peaks = peaks_from_nodes(
        pl.read_csv(interim / "graph_nodes.csv", schema_overrides={"name": pl.String})
    )
    _write_atomic(peaks.write_csv, out_dir / "papkou2023_peaks.csv")
    _write_atomic(
        lambda p: shutil.copyfile(interim / "graph.graphml", p),
        out_dir / "papkou2023_graph.graphml",
    )


def graph_counts(path: Path) -> tuple[int, int]:
    """(nodes, edges) of a GraphML file, streamed."""
    from xml.etree import ElementTree as ET

    nodes = edges = 0
    for _, element in ET.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "node":
            nodes += 1
            element.clear()
        elif tag == "edge":
            edges += 1
            element.clear()
    return nodes, edges


def check_shapes(raw_dir: Path, expected: dict[str, int]) -> list[str]:
    """Rows, columns, names and dtypes of the raw files; returns the list of problems."""
    problems = []
    for name, key in (
        ("papkou2023_fitness_wt.parquet", "wt_rows"),
        ("papkou2023_fitness_diff.parquet", "diff_rows"),
    ):
        path = raw_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        lazy = pl.scan_parquet(path)
        schema = dict(lazy.collect_schema())
        if schema != FITNESS_SCHEMA:
            problems.append(f"{name}: schema {schema}, expected {FITNESS_SCHEMA}")
        rows = lazy.select(pl.len()).collect().item()
        if rows != expected[key]:
            problems.append(f"{name}: {rows} rows, expected {expected[key]}")
    peaks = raw_dir / "papkou2023_peaks.csv"
    if not peaks.is_file():
        problems.append("papkou2023_peaks.csv: missing")
    else:
        frame = pl.read_csv(peaks, infer_schema=False)
        if frame.columns != ["dna"]:
            problems.append(
                f"papkou2023_peaks.csv: columns {frame.columns}, expected ['dna']"
            )
        elif frame.height != expected["peaks"]:
            problems.append(
                f"papkou2023_peaks.csv: {frame.height} peaks, expected {expected['peaks']}"
            )
    graph = raw_dir / "papkou2023_graph.graphml"
    if not graph.is_file():
        problems.append("papkou2023_graph.graphml: missing")
    else:
        nodes, edges = graph_counts(graph)
        if (nodes, edges) != (expected["graph_nodes"], expected["graph_edges"]):
            problems.append(
                f"papkou2023_graph.graphml: {nodes} nodes / {edges} edges, expected "
                f"{expected['graph_nodes']} / {expected['graph_edges']}"
            )
    return problems


def load_clamped(path: Path | None = None) -> dict[str, Any]:
    """The ``[clamped]`` section of config/analysis.toml (the pinned reference digests)."""
    with open(path or fs.REPO_ROOT / "config" / "analysis.toml", "rb") as handle:
        return tomllib.load(handle)["clamped"]


def report_digests(raw_dir: Path, clamped: dict[str, Any], log=print) -> list[str]:
    """Print the value digests of the rebuilt files; return a warning per difference.

    The GraphML and peak files are compared with the pinned value digests
    (``graphml_value_sha256``, ``peaks_value_sha256``) and with the original byte
    hashes, which stay in config/analysis.toml as reference values. A difference in
    bytes alone is expected of a rebuild; a difference in values means the rebuilt
    file is not the file the analysis pins. Neither fails the rebuild.
    """
    warnings: list[str] = []
    for name, value_of, value_key, bytes_key in (
        (
            "papkou2023_graph.graphml",
            graphml_value_digest,
            "graphml_value_sha256",
            "graphml_bytes_sha256",
        ),
        (
            "papkou2023_peaks.csv",
            peaks_value_digest,
            "peaks_value_sha256",
            "peaks_bytes_sha256",
        ),
    ):
        path = raw_dir / name
        value, raw_bytes = value_of(path), bytes_sha256(path)
        log(f"{name}: value digest {value}, byte sha256 {raw_bytes}")
        if value != clamped[value_key]:
            warnings.append(
                f"{name}: value digest differs from {value_key} in config/analysis.toml "
                f"({value} != {clamped[value_key]}); the analysis will refuse this file."
            )
        if raw_bytes != clamped[bytes_key]:
            equal = "values equal" if value == clamped[value_key] else "values differ"
            warnings.append(
                f"{name}: bytes differ from the reference {bytes_key} ({equal}); "
                "the reference is informational."
            )
    for name in ("papkou2023_fitness_wt.parquet", "papkou2023_fitness_diff.parquet"):
        log(f"{name}: value digest {parquet_value_digest(raw_dir / name)}")
    for warning in warnings:
        log(f"WARNING: {warning}")
    return warnings


def make_raw(config: dict[str, Any], store: Path, out_dir: Path, log=print) -> int:
    in_data = store / config["papkou2023"]["zip_root_in_store"] / "in_data"
    for member in config["papkou2023"]["members"]:
        path = store / config["papkou2023"]["zip_root_in_store"] / member["in_zip"]
        fs.require_md5(path, member["md5"])
    with tempfile.TemporaryDirectory(prefix="make-raw-") as tmp:
        interim = Path(tmp)
        log("Reading the RDS files in R ...")
        run_r(in_data, interim)
        log(f"Writing {out_dir} ...")
        write_outputs(interim, out_dir)
    problems = check_shapes(out_dir, config["raw_shape"])
    if problems:
        log("Shape check FAILED:\n  " + "\n  ".join(problems))
        return 1
    log("Shape check passed (counts as recorded in data/raw/README.md).")
    report_digests(out_dir, load_clamped(), log)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store", help="Papkou store (default $PAPKOU2023RUGGED_DATA)."
    )
    parser.add_argument(
        "--out", default="data/raw", help="Output directory (default data/raw)."
    )
    args = parser.parse_args(argv)
    config = fs.load_config()
    return make_raw(config, store_path(config, args.store), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
