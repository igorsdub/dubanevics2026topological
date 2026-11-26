"""Landscape-summary slices kept as supplementary manuscript tables.

Supplementary S1: Full, Published and median-reflected BY Candidate.
Supplementary S2: Candidate corrections — Nominal | FDR | Bootstrap (median-reflected
point-estimate, median-reflected BY nominal, median-reflected BY band).
Supplementary (unnumbered): Published simplification — Nominal | FDR | Bootstrap on the
Published graph (point-estimate, Published BY nominal, Published BY band).

Main Tables 1 and 2 are the graph and tree descriptive tables, not these slices.

Column sources are the pinned ``landscape_summary`` products already used by
``landscape_summary_table``; this module renames columns for the manuscript and
writes separate CSV/HTML/Markdown exports plus a small identity sidecar.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..analysis.describe import SUMMARY_COLUMNS, TREE_LEVEL_METRICS
from ..figures.labels import FULL, PAPKOU, TOPOGRAPHER_FDR
from ..results import RunContext
from .landscape_summary_table import load_summary_columns, tree_level_only

Loader = Callable[[str, str], Any]

# Manuscript display names → landscape-summary source column. The source strings are the
# saved products' identity names and do not change; only the display names are new.
TABLE1_COLUMNS: tuple[tuple[str, str], ...] = (
    (FULL, "Full (nominal)"),
    (PAPKOU, "Published (point-estimate nominal)"),
    (TOPOGRAPHER_FDR, "Median-reflected BY (nominal)"),
)
TABLE2_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Nominal", "Median-reflected (point-estimate nominal)"),
    ("FDR", "Median-reflected BY (nominal)"),
    ("Bootstrap", "Median-reflected BY (band-simplified)"),
)
PUBLISHED_SIMPLIFICATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Nominal", "Published (point-estimate nominal)"),
    ("FDR", "Published BY (nominal)"),
    ("Bootstrap", "Published BY (band-simplified)"),
)

# Rows kept for the shorter manuscript tables (T11 draft metrics).
MANUSCRIPT_METRICS: tuple[str, ...] = (
    "Functional threshold",
    "Number of nodes",
    "Number of edges",
    "Number of functional sequences",
    "Number of non-functional sequences",
    "Number of uncertain sequences",
    "Peak count",
    "Minimum fitness",
    "Maximum fitness",
    "Mean fitness",
    "Median fitness",
)

SIDECAR_SCHEMA = 1


@dataclass(frozen=True)
class PresentationTable:
    """One manuscript table view over the landscape summary."""

    stem: str
    title: str
    section: str
    columns: tuple[tuple[str, str], ...]
    note: str


PRESENTATION_TABLES: tuple[PresentationTable, ...] = (
    PresentationTable(
        stem="full_vs_published",
        title=f"Full, {PAPKOU} and {TOPOGRAPHER_FDR} landscapes",
        section="supplementary",
        columns=TABLE1_COLUMNS,
        note=(
            "Supplementary table S1. Full, Papkou and "
            "Topographer + FDR graphs. Threshold, counts and fitness "
            "summaries come from the pinned landscape_summary products. Full "
            "valley count comes from the manuscript quantity registry. Papkou "
            "and Topographer + FDR valley counts are unavailable: no authenticated "
            "per-selection valley product exists. A blank Full valley count means "
            "its registry is absent."
        ),
    ),
    PresentationTable(
        stem="candidate_corrections",
        title="Topographer landscape: Nominal, FDR and Bootstrap",
        section="supplementary",
        columns=TABLE2_COLUMNS,
        note=(
            "Supplementary table S2. Topographer graph at three stages: point-estimate "
            "Nominal, Benjamini–Yekutieli FDR (nominal q = 0.05), and the "
            "band-simplified Bootstrap tree on the FDR-selected graph. Bootstrap "
            "reports tree-level rows only (peak count and synonymous-leaf metrics); "
            "its graph-level cells are an em dash because that graph is the FDR "
            "column's (checked equal before blanking)."
        ),
    ),
    PresentationTable(
        stem="published_simplification",
        title="Papkou landscape simplification",
        section="supplementary",
        columns=PUBLISHED_SIMPLIFICATION_COLUMNS,
        note=(
            "Supplementary. Papkou graph at three stages: point-estimate "
            "Nominal, Papkou + FDR (nominal q = 0.05), and the band-simplified "
            "Bootstrap tree. Bootstrap reports tree-level rows only; its graph-level "
            "cells are an em dash because that graph is the FDR column's (checked "
            "equal before blanking)."
        ),
    ),
)


def _require_summary_columns(labels: Mapping[str, str]) -> None:
    if tuple(labels) != SUMMARY_COLUMNS:
        raise ValueError("Pin exactly the seven ordered landscape summary columns.")


def _select_columns(
    columns: Mapping[str, Mapping[str, Any]],
    mapping: Sequence[tuple[str, str]],
    *,
    metrics: Sequence[str] = MANUSCRIPT_METRICS,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for display, source in mapping:
        if source not in columns:
            raise ValueError(f"Missing landscape summary column {source!r}.")
        source_rows = columns[source]
        selected[display] = {}
        for metric in metrics:
            if metric not in source_rows:
                # Uncertain is "not applicable" / absent on point-estimate columns.
                if metric == "Number of uncertain sequences":
                    selected[display][metric] = source_rows.get(
                        metric, "not applicable"
                    )
                    continue
                raise ValueError(f"Metric {metric!r} missing from {source!r}.")
            selected[display][metric] = source_rows[metric]
    return selected


def _export_view(
    columns: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
    stem: str,
    title: str,
    directory: Path,
    note: str,
    percent_decimals: int,
) -> list[Path]:
    names = tuple(columns)
    if not names:
        raise ValueError("Presentation table needs at least one column.")
    metrics = tuple(columns[names[0]])
    if any(tuple(columns[name]) != metrics for name in names):
        raise ValueError("Presentation columns must share the same metrics.")
    rows = [
        {"Metric": metric, **{name: columns[name][metric] for name in names}}
        for metric in metrics
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("Metric", *names), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    def display(value: object, metric: str = "") -> str:
        if value is None:
            return "—"
        if isinstance(value, float) and (
            metric.endswith("(%)") or "percentage" in metric.lower()
        ):
            return f"{value:.{percent_decimals}f}%"
        return str(value)

    headers = ("Metric", *names)
    html_header = "".join(
        f'<th scope="col">{html.escape(name)}</th>' for name in headers
    )
    html_rows = "".join(
        '<tr><th scope="row">'
        + html.escape(row["Metric"])
        + "</th>"
        + "".join(
            f"<td>{html.escape(display(row[name], row['Metric']))}</td>"
            for name in names
        )
        + "</tr>"
        for row in rows
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1><table><thead><tr>"
        f"{html_header}</tr></thead><tbody>{html_rows}</tbody></table>"
        f"<p>{html.escape(note)}</p></body></html>\n"
    )
    md = (
        f"# {title}\n\n| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "".join(
            "| "
            + " | ".join(
                display(row[name], row["Metric"]).replace("|", "\\|")
                for name in headers
            )
            + " |\n"
            for row in rows
        )
        + "\n"
        + note
        + "\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{dataset}_{stem}.{ext}" for ext in ("csv", "html", "md")]
    for path, content in zip(paths, (stream.getvalue(), document, md), strict=True):
        path.write_text(content, encoding="utf-8")
    return paths


def view_identity(
    products: Mapping[str, Mapping[str, str]],
    mapping: Sequence[tuple[str, str]],
) -> str:
    """Digest over the source scientific ids of one presentation view."""
    payload = json.dumps(
        [
            [display, source, products[source]["scientific_id_sha256"]]
            for display, source in mapping
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_presentation_tables(
    context: RunContext,
    labels: Mapping[str, str],
    dataset: str,
    directory: Path,
    *,
    percent_decimals: int = 2,
    loader: Loader | None = None,
    tables: Sequence[PresentationTable] = PRESENTATION_TABLES,
    valley_source: tuple[str, str] | None = None,
) -> list[Path]:
    """Write the manuscript presentation tables from pinned landscape summaries."""
    _require_summary_columns(labels)
    columns, products = load_summary_columns(context, labels, loader=loader)
    columns = tree_level_only(columns)
    written: list[Path] = []
    for table in tables:
        view = _select_columns(columns, table.columns)
        if table.columns == TABLE1_COLUMNS:
            for name, values in view.items():
                values["Valley count"] = (
                    valley_source[0] if name == "Full" and valley_source else None
                )
        paths = _export_view(
            view,
            dataset=dataset,
            stem=table.stem,
            title=table.title,
            directory=directory,
            note=table.note,
            percent_decimals=percent_decimals,
        )
        sidecar = directory / f"{dataset}_{table.stem}.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema": SIDECAR_SCHEMA,
                    "stem": table.stem,
                    "section": table.section,
                    "title": table.title,
                    "table_identity_sha256": (
                        hashlib.sha256(
                            json.dumps(
                                [view_identity(products, table.columns), valley_source],
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        if table.columns == TABLE1_COLUMNS
                        else view_identity(products, table.columns)
                    ),
                    "columns": [
                        {
                            "display": display,
                            "source": source,
                            **products[source],
                        }
                        for display, source in table.columns
                    ],
                    "tree_level_metrics": sorted(TREE_LEVEL_METRICS),
                    "metrics": list(next(iter(view.values()))),
                    **(
                        {
                            "valley_source": {
                                "key": "full_graph.persistence.valley_count",
                                "registry_fingerprint": valley_source[1]
                                if valley_source
                                else None,
                                "gaps": {
                                    **(
                                        {"Full": "Current manuscript registry absent."}
                                        if valley_source is None
                                        else {}
                                    ),
                                    PAPKOU: "No authenticated per-selection valley product.",
                                    TOPOGRAPHER_FDR: "No authenticated per-selection valley product.",
                                },
                            }
                        }
                        if table.columns == TABLE1_COLUMNS
                        else {}
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.extend([*paths, sidecar])
    return written
