"""Split tree versus bootstrap-band tree for the BY q = 0.05 Topographer graph.

Column 1 is the nominal split tree of variant ``threshold=reflected_fdr=by_q=0.05``.
Column 2 is that graph's band-simplified tree. The bootstrap checkpoint is only
read (``load_cached_bootstrap_smoothed_tree`` never draws). Branch vertex counts
and Horton–Strahler orders come from the branch decomposition. Amino-acid and
nucleotide distances are Topographer's endpoint ``hamming_distance`` on each
branch's two critical endpoints. Ordinary H0 persistence is the pair table's
``persistence`` (the global maximum closed at the global minimum). Branch span
is not used.
"""

from __future__ import annotations

import csv
import html
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from ..analysis.branch_decomposition import BranchDecomposition
from ..sequence_distances import hamming_distance

VARIANT = "threshold=reflected_fdr=by_q=0.05"
TABLE_STEM = "candidate_band_comparison"
SIDECAR_NAME = ".candidate-band-comparison-provenance.json"
COLUMNS = ("statistic", "split_tree", "band_simplified_tree")
COLUMN_TITLES = ("Split tree", "Bootstrapped tree")
HS_ORDERS = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class TreeColumn:
    """One tree, already measured by the existing decomposition and pair table."""

    branches: pl.DataFrame
    persistence: Sequence[float]
    amino_acid: Mapping[str, str]
    nucleotide: Mapping[str, str]


def _has_stop(amino_acid: str) -> bool:
    from topographer.biology import has_stop_codon

    return bool(has_stop_codon(amino_acid))


def _num(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def _join(values: Sequence[float]) -> str:
    return " / ".join(_num(value) for value in values)


def measure_tree(column: TreeColumn) -> dict[str, str]:
    """Aggregate one tree. Distances are endpoint mismatches, one per branch."""
    branches = column.branches
    required = {
        "upper_sequence",
        "lower_sequence",
        "is_terminal",
        "branch_vertex_count",
        "horton_strahler_order",
    }
    missing = required - set(branches.columns)
    if missing:
        raise ValueError(f"Branch table is missing {sorted(missing)}.")
    if branches.height == 0:
        raise ValueError("The split tree has no branches.")
    if not column.persistence:
        raise ValueError("The split tree has no ordinary H0 persistence pairs.")
    amino: list[int] = []
    nucleotide: list[int] = []
    stops = 0
    peaks = 0
    for row in branches.iter_rows(named=True):
        upper = str(row["upper_sequence"])
        lower = str(row["lower_sequence"])
        try:
            amino.append(
                hamming_distance(column.amino_acid[upper], column.amino_acid[lower])
            )
            nucleotide.append(
                hamming_distance(column.nucleotide[upper], column.nucleotide[lower])
            )
        except KeyError as error:
            raise ValueError(
                f"Branch endpoint {error} has no sequence for Hamming distance."
            ) from error
        if row["is_terminal"]:
            peaks += 1
            if _has_stop(column.amino_acid[upper]):
                stops += 1
    saddles = branches.height - peaks
    counts = branches["branch_vertex_count"].to_list()
    if any(int(count) < 1 for count in counts):
        raise ValueError("Branch vertex count must be positive.")
    orders = [int(order) for order in branches["horton_strahler_order"].to_list()]
    extra = sorted(set(orders) - set(HS_ORDERS))
    if extra:
        raise ValueError(f"Horton–Strahler order outside 1–4: {extra}.")
    return {
        "number of peaks; number of saddles": f"{peaks}; {saddles}",
        "branch vertex count: min / median / mean / max": _join(
            (min(counts), median(counts), fmean(counts), max(counts))
        ),
        "amino-acid Hamming distance along a branch: mean / median": _join(
            (fmean(amino), median(amino))
        ),
        "nucleotide Hamming distance along a branch: mean / median": _join(
            (fmean(nucleotide), median(nucleotide))
        ),
        "Horton-Strahler order 1 / 2 / 3 / 4 branch counts": _join(
            tuple(orders.count(order) for order in HS_ORDERS)
        ),
        "branch persistence: median / max": _join(
            (median(column.persistence), max(column.persistence))
        ),
        "stop-codon genotypes on peaks": str(stops),
    }


def comparison_rows(
    split_tree: TreeColumn, band_tree: TreeColumn
) -> list[dict[str, str]]:
    """One manuscript row per statistic line, split tree then band-simplified tree."""
    left = measure_tree(split_tree)
    right = measure_tree(band_tree)
    if tuple(left) != tuple(right):
        raise ValueError("The two trees produced different statistic rows.")
    return [
        {
            "statistic": name,
            "split_tree": left[name],
            "band_simplified_tree": right[name],
        }
        for name in left
    ]


def column_from_decomposition(
    decomposition: BranchDecomposition,
    pairs: pl.DataFrame,
    *,
    amino_acid: Mapping[str, str],
    nucleotide: Mapping[str, str],
) -> TreeColumn:
    """Persistence is the pair table; sequences are keyed by DNA identity."""
    if "persistence" not in pairs.columns:
        raise ValueError("Persistence pairs have no persistence column.")
    return TreeColumn(
        branches=decomposition.branches,
        persistence=[float(value) for value in pairs["persistence"].to_list()],
        amino_acid=dict(amino_acid),
        nucleotide=dict(nucleotide),
    )


def sequences_from_assignments(
    assignments: pl.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """DNA identity to amino-acid and nucleotide strings, from a smoothed-tree assignment."""
    amino: dict[str, str] = {}
    nucleotide: dict[str, str] = {}
    for row in assignments.select("dna_sequence", "aa_sequence").iter_rows(named=True):
        identity = str(row["dna_sequence"])
        aa = row["aa_sequence"]
        if not isinstance(aa, str) or not aa:
            raise ValueError(f"Sequence {identity} has no amino-acid sequence.")
        amino[identity] = aa
        nucleotide[identity] = identity
    return amino, nucleotide


def readme_text(*, n_replicates: int, seed: int, band_quantile: float) -> str:
    """How each row is defined, including the band settings actually used."""
    return f"""# Topographer + FDR split tree and its bootstrap-band tree

Manuscript Table 2. Variant `threshold=reflected_fdr=by_q=0.05`: the Topographer + FDR graph selected by Benjamini–Yekutieli at nominal q = 0.05 with the median-reflected cutoff. Column `split_tree` is that graph's nominal split tree. Column `band_simplified_tree` is the same tree after direct band simplification.

Band settings are those of the restored bootstrap checkpoint (`summary.json` `settings`), read by `load_cached_bootstrap_smoothed_tree`. This table does not draw. The values used are draws (n_replicates) = {n_replicates}, seed = {seed}, band quantile = {band_quantile}. `config/analysis.toml` `[bootstrap_smoothed_tree]` carries the same triple when that file is unchanged. This table is manuscript Table 2.

A branch is one edge of the tree after `absorb_terminal_branches`, as `analysis.branch_decomposition` defines it.

- **Branch vertex count.** `branch_vertex_count` from the branch decomposition: the upper endpoint, the interior vertices on that edge, and the sequences of hills absorbed onto that edge. The lower endpoint belongs to the edge below and is excluded. The minimum is 1. The row reports the minimum, median, mean and maximum of that count over branches.
- **Amino-acid Hamming distance along a branch.** `hamming_distance` from `sequence_distances` between the amino-acid sequences of the branch's two critical endpoints (upper and lower). It counts mismatched positions. Both sequences must have the same length. It is that endpoint mismatch count. It is not a sum of distances along the path of member sequences, and it is not the maximum pairwise distance among the branch's members. The row reports the mean and the median over branches.
- **Nucleotide Hamming distance along a branch.** The same `hamming_distance` between the DNA sequences of those same two critical endpoints. The same endpoint definition applies: mismatch count between the two endpoints, not path length and not maximum pairwise distance among members. The row reports the mean and the median over branches.
- **Horton–Strahler order.** `horton_strahler_order` from Topographer `build_tree_metrics` on the smoothed tree, the column the branch decomposition stores. A terminal branch has order 1. The row reports how many branches have order 1, 2, 3 and 4. An order with no branch is 0.
- **Branch persistence.** Ordinary H0 persistence of each pair on that tree (`persistence` on the pair table produced with the simplified tree): the absolute fitness difference between the peak's birth and its death saddle. The global maximum's pair is closed at the global minimum, so every pair is finite. The row reports the median and the maximum over pairs. This is not branch span (`birth_fitness - death_fitness` of one edge).
- **Peaks and saddles.** Peaks are the terminal branches (split-tree maxima). Saddles are the other branches: each saddle is the upper endpoint of the single edge below it. The root is not a peak and not a saddle. The cell is `peaks; saddles`.
- **Stop-codon genotypes on peaks.** The number of peak genotypes whose amino-acid sequence contains a stop symbol, using Topographer `has_stop_codon` (`topographer.biology.has_stop_codon`). The symbols are `*`, `%`, `^` and `&`, the same set as `STOP_CODON_SYMBOLS` in that module and in `analysis.biological_annotations`. Interior sequences on a branch do not add to this count.
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _delimited(rows: Sequence[Mapping[str, str]], delimiter: str) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=COLUMNS, delimiter=delimiter, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_png(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    """A static rendering of the same rows as the CSV."""
    cell = [[row[name] for name in COLUMNS] for row in rows]
    figure, axis = plt.subplots(figsize=(14, 0.55 * (len(rows) + 2) + 0.8))
    axis.axis("off")
    axis.set_title(
        "Topographer + FDR: split tree and band-simplified tree",
        loc="left",
        fontsize=12,
        pad=12,
    )
    table = axis.table(
        cellText=cell,
        colLabels=("Statistic", *COLUMN_TITLES),
        loc="upper left",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _markdown(rows: Sequence[Mapping[str, str]], *, note: str) -> str:
    headers = ("Statistic", *COLUMN_TITLES)
    lines = [
        "# Topographer + FDR split tree and its bootstrap-band tree",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = (row["statistic"], row["split_tree"], row["band_simplified_tree"])
        lines.append(
            "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
        )
    return "\n".join(lines) + "\n\n" + note


def _html_document(rows: Sequence[Mapping[str, str]]) -> str:
    headers = ("Statistic", *COLUMN_TITLES)
    head = "".join(f"<th>{html.escape(name)}</th>" for name in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(row[name])}</td>"
            for name in ("statistic", "split_tree", "band_simplified_tree")
        )
        + "</tr>"
        for row in rows
    )
    title = "Topographer + FDR: split tree and band-simplified tree"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</body></html>\n"
    )


def write_table(
    directory: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    n_replicates: int,
    seed: int,
    band_quantile: float,
    provenance: Mapping[str, Any],
    sidecar: Path,
) -> list[Path]:
    """Write CSV, TSV, HTML, Markdown, README, PNG and the provenance sidecar."""
    directory.mkdir(parents=True, exist_ok=True)
    note = readme_text(
        n_replicates=n_replicates, seed=seed, band_quantile=band_quantile
    )
    payload = json.dumps(dict(provenance), indent=2) + "\n"
    csv_path = directory / f"{TABLE_STEM}.csv"
    tsv_path = directory / f"{TABLE_STEM}.tsv"
    html_path = directory / f"{TABLE_STEM}.html"
    md_path = directory / f"{TABLE_STEM}.md"
    json_path = directory / f"{TABLE_STEM}.json"
    readme_path = directory / "README.md"
    png_path = directory / f"{TABLE_STEM}.png"
    _write_text(csv_path, _delimited(rows, ","))
    _write_text(tsv_path, _delimited(rows, "\t"))
    _write_text(html_path, _html_document(rows))
    _write_text(md_path, _markdown(rows, note=note))
    _write_text(json_path, payload)
    _write_text(readme_path, note)
    render_png(rows, png_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(payload, encoding="utf-8")
    return [
        csv_path,
        tsv_path,
        html_path,
        md_path,
        json_path,
        readme_path,
        png_path,
        sidecar,
    ]


def copy_flat_tables(written: list[Path], table_dir: Path) -> list[Path]:
    """Copy the bundle's csv/html/md/json beside it, under the run context's output root.

    The namer reads these flat stems from ``<output_root>/tables/canonical``, not the bundle.
    """
    shelf = table_dir.parent
    copied: list[Path] = []
    for path in written:
        if (
            path.parent == table_dir
            and path.name != "README.md"
            and path.suffix in {".csv", ".html", ".md", ".json"}
        ):
            target = shelf / path.name
            target.write_bytes(path.read_bytes())
            copied.append(target)
    return copied


def build_candidate_band_comparison(context: Any | None = None) -> list[Path]:
    """Load the pinned bootstrap, decompose both trees, and write the table.

    Heavy: builds the Candidate split tree. Does not draw bootstrap replicates.
    """
    from ..analysis import load_settings
    from ..analysis.branch_decomposition import (
        ADJACENCY_COLUMNS,
        build_candidate_inputs,
        decomposition_view,
    )
    from ..results import RunContext
    from ..simplification.bootstrap_smoothed_tree import (
        check_hills_match_tree,
        load_cached_bootstrap_smoothed_tree,
    )

    if context is None:
        root = Path("output") / "variants" / VARIANT
        context = RunContext(output_root=root, variant_name=VARIANT)
    if context.variant_name != VARIANT:
        raise ValueError(
            f"This table is the {VARIANT} Topographer + FDR graph, got {context.variant_name!r}."
        )
    settings = load_settings(context)
    bootstrap = load_cached_bootstrap_smoothed_tree(settings, context=context)
    used = bootstrap.summary["settings"]
    tree_analysis, _adjacency = build_candidate_inputs(settings)
    check_hills_match_tree(bootstrap.hills, tree_analysis)
    nominal = decomposition_view("nominal", tree_analysis, bootstrap.hills)
    band = decomposition_view("band", tree_analysis, bootstrap.hills)
    empty = pl.DataFrame(schema=dict.fromkeys(ADJACENCY_COLUMNS, pl.String))
    rows = comparison_rows(
        _column(nominal, empty),
        _column(band, empty),
    )
    provenance = {
        "schema": 1,
        "variant": VARIANT,
        "n_replicates": int(used["n_replicates"]),
        "seed": int(used["seed"]),
        "band_quantile": float(used["band_quantile"]),
        "bootstrap_loader": "load_cached_bootstrap_smoothed_tree",
        "draws_executed": 0,
        "bootstrap_fingerprint": bootstrap.provenance.get("fingerprint"),
    }
    tables = Path(context.output_root, "tables")
    table_dir = Path(tables, "canonical", TABLE_STEM)
    sidecar = Path(tables, SIDECAR_NAME)
    written = write_table(
        table_dir,
        rows,
        n_replicates=provenance["n_replicates"],
        seed=provenance["seed"],
        band_quantile=provenance["band_quantile"],
        provenance=provenance,
        sidecar=sidecar,
    )
    copied = copy_flat_tables(written, table_dir)
    return [*written, *copied]


def _column(view: Any, adjacency: pl.DataFrame) -> TreeColumn:
    from ..analysis.branch_decomposition import decompose_view

    amino, nucleotide = sequences_from_assignments(view.assignments)
    return column_from_decomposition(
        decompose_view(view, adjacency),
        view.pairs,
        amino_acid=amino,
        nucleotide=nucleotide,
    )
