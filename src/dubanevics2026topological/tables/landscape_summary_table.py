"""The seven-column landscape summary table, composed from the saved per-state products.

Each column is one saved ``landscape_summary`` product (``build/
landscape_summary_product.py``), pinned by label in ``config/figures.toml``
(``[tables.landscape_summary.labels]``). The table computes nothing: it reads the
saved values, checks that each product is verified and belongs to its column, and
hands them to ``export_landscape_summary``. A band column keeps only the tree-level
rows; its graph-level cells are blank because the graph is the one of the nominal
column beside it (the saved values are checked equal before they are dropped).

The JSON sidecar records the seven products' scientific ids and one digest over
them (``table_identity_sha256``). Formatting, labels and percent display are not
part of it, so an appearance edit changes the exported text and nothing scientific.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..analysis.describe import SUMMARY_COLUMNS, TREE_LEVEL_METRICS
from ..results import RunContext
from ..results.processed import product_loader
from .graph_summary import export_landscape_summary

SUMMARY_FILE = "summary.json"
SIDECAR_SCHEMA = 1
# Band column -> the nominal column that shares its selected graph.
BAND_NOMINAL = {
    SUMMARY_COLUMNS[4]: SUMMARY_COLUMNS[3],
    SUMMARY_COLUMNS[6]: SUMMARY_COLUMNS[5],
}
BAND_NOTE = (
    " Band-simplified columns report tree-level rows only (peak count and "
    "synonymous-leaf rows); their graph-level rows are those of the nominal "
    "column of the same graph and are shown as an em dash."
)

Loader = Callable[[str, str], Any]


def _verified(loader: Loader, label: str) -> Any:
    value = loader("landscape_summary", label)
    if value.report.status != "full" or not value.report.byte_integrity:
        raise ValueError(f"Unverified or incomplete landscape_summary: {label}")
    return value


def load_summary_columns(
    context: RunContext,
    labels: Mapping[str, str],
    *,
    loader: Loader | None = None,
    names: tuple[str, ...] = SUMMARY_COLUMNS,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """The saved values of the columns ``names`` and the products they came from.

    All seven by default; the manuscript quantity registry reads only the states the
    manuscript quotes.
    """
    if tuple(labels) != SUMMARY_COLUMNS:
        raise ValueError("Pin exactly the seven ordered landscape summary columns.")
    if loader is None:

        loader = product_loader(context)

    columns: dict[str, dict[str, Any]] = {}
    products: dict[str, dict[str, str]] = {}
    for name in names:
        product = _verified(loader, labels[name])
        if product.manifest.scientific_identity.get("column") != name:
            raise ValueError(f"{labels[name]} is not the {name!r} column.")
        rows = json.loads(product.artifacts[SUMMARY_FILE].read_text(encoding="utf-8"))
        metrics = [row["metric"] for row in rows]
        if len(set(metrics)) != len(metrics):
            raise ValueError(f"{name}: duplicate metric rows.")
        columns[name] = {row["metric"]: row["value"] for row in rows}
        products[name] = {
            "label": product.label,
            "scientific_id_sha256": product.manifest.scientific_id_sha256,
        }
    return columns, products


def tree_level_only(columns: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Blank the graph-level rows of each band column after checking they are shared."""
    result = {name: dict(values) for name, values in columns.items()}
    for band, nominal in BAND_NOMINAL.items():
        if list(columns[band]) != list(columns[nominal]):
            raise ValueError(f"{band} and {nominal} must have matching metrics.")
        for metric, value in columns[band].items():
            if metric in TREE_LEVEL_METRICS:
                continue
            if value != columns[nominal][metric]:
                raise ValueError(f"Graph metric {metric!r} differs: {band}.")
            result[band][metric] = None
    return result


def table_identity(products: Mapping[str, Mapping[str, str]]) -> str:
    """Digest over the seven scientific ids, in column order; nothing else."""
    payload = json.dumps(
        [[name, products[name]["scientific_id_sha256"]] for name in SUMMARY_COLUMNS]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_summary_table(
    context: RunContext,
    labels: Mapping[str, str],
    dataset: str,
    directory: Path,
    *,
    percent_decimals: int = 2,
    loader: Loader | None = None,
) -> list[Path]:
    """Write the CSV, HTML and Markdown tables and the identity sidecar."""
    columns, products = load_summary_columns(context, labels, loader=loader)
    paths = export_landscape_summary(
        tree_level_only(columns),
        dataset,
        directory,
        percent_decimals=percent_decimals,
        note_suffix=BAND_NOTE,
    )
    sidecar = directory / f"{dataset}_landscape_summary.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": SIDECAR_SCHEMA,
                "table_identity_sha256": table_identity(products),
                "columns": [
                    {"column": name, **products[name]} for name in SUMMARY_COLUMNS
                ],
                "tree_level_only_columns": list(BAND_NOMINAL),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [*paths, sidecar]
