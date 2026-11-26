"""Presentation-only export of the seven-column landscape summary table.

The three-column Table 1 that read the manuscript quantity registry is retired; the
seven-column table (``tables/landscape_summary_table.py``) replaces it and this module
only lays out its saved values.
"""

from __future__ import annotations

import csv
import html
import io
from collections.abc import Mapping
from pathlib import Path

from ..analysis.describe import SUMMARY_COLUMNS


def export_landscape_summary(
    columns: Mapping[str, Mapping[str, int | float | None]],
    dataset: str,
    directory: Path,
    *,
    percent_decimals: int = 2,
    note_suffix: str = "",
) -> list[Path]:
    """Export numerical summary columns with their denominators in row labels.

    ``percent_decimals`` and ``note_suffix`` are presentation only; the CSV keeps
    the saved values unformatted.
    """
    names = tuple(columns)
    if names != SUMMARY_COLUMNS:
        raise ValueError("Landscape summary requires exactly the seven ordered states.")
    metrics = tuple(columns[names[0]])
    if any(tuple(columns[name]) != metrics for name in names):
        raise ValueError("Landscape summary columns must have matching metrics.")
    note = (
        "Full-library percentages use the Full library as denominator; selected "
        "graph and stop percentages use the selected-graph vertex count. "
        "Synonymous leaf percentages use eligible leaves and exclude root-only "
        "endpoints. Component and pre-largest-component counts precede largest "
        "component selection. An em dash denotes an undefined quantity." + note_suffix
    )
    rows = [
        {"Metric": metric, **{name: columns[name][metric] for name in names}}
        for metric in metrics
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("Metric", *names), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    def display(value: object, metric: str = "") -> str:
        """Presentation text; the CSV keeps the saved values unformatted."""
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
        f"<title>{html.escape(dataset)} landscape summary</title></head><body>"
        f"<h1>{html.escape(dataset)} landscape summary</h1><table><thead><tr>"
        f"{html_header}</tr></thead><tbody>{html_rows}</tbody></table>"
        f"<p>{html.escape(note)}</p></body></html>\n"
    )
    md = (
        f"# {dataset} landscape summary\n\n| "
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
    paths = [
        directory / f"{dataset}_landscape_summary.{ext}"
        for ext in ("csv", "html", "md")
    ]
    for path, content in zip(paths, (stream.getvalue(), document, md), strict=True):
        path.write_text(content, encoding="utf-8")
    return paths
