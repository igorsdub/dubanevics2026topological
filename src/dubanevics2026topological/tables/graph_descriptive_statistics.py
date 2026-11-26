"""Descriptive statistics of the four publication graphs.

Columns are built, not relabelled. Full is the whole sequence graph. Papkou is
the Papkou-threshold landscape. Topographer is the median-reflected
point-estimate graph (``threshold=reflected_fdr=none``). Topographer + FDR is built by
loading ``threshold=reflected_fdr=by_q=0.05`` and running ``prepare_selected_landscape``
on that declaration. Counts and degrees are computed from those networks.
Manuscript Table 1 exports nodes, edges, percent functional and the two degree rows.
"""

from __future__ import annotations

import csv
import html
import io
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from ..figures.labels import FULL, PAPKOU, TOPOGRAPHER, TOPOGRAPHER_FDR

TABLE_ID = "graph_descriptive_statistics"
POINT_VARIANT = "threshold=reflected_fdr=none"
BY_VARIANT = "threshold=reflected_fdr=by_q=0.05"
EN_DASH = "\u2013"
SIDECAR_SCHEMA = 1

COLUMNS: tuple[str, ...] = (
    FULL,
    PAPKOU,
    TOPOGRAPHER,
    TOPOGRAPHER_FDR,
)
# Manuscript Table 1. The second item is the key ``summarize_graph`` stores.
MANUSCRIPT_ROWS: tuple[tuple[str, str], ...] = (
    ("nodes", "number of nodes"),
    ("edges", "number of edges"),
    ("% functional", "% functional"),
    ("mean degree (undirected)", "mean degree (undirected)"),
    (
        "mean in/out degree (directed uphill)",
        "mean in/out degree (in = out)",
    ),
)
ROWS: tuple[str, ...] = (
    "number of nodes",
    "number of edges",
    "minimum fitness",
    "maximum fitness",
    "mean fitness",
    "median fitness",
    "functional threshold",
    "functional count",
    "non-functional count",
    "% functional",
    "mean degree (undirected)",
    "mean in/out degree (in = out)",
)

_UNDEFINED_ON_FULL = (
    "functional threshold",
    "functional count",
    "non-functional count",
    "% functional",
)


@dataclass(frozen=True)
class GraphColumn:
    """One graph and the rule that decides which of its vertices are functional."""

    label: str
    network: Any
    threshold: float | None
    classes: Mapping[str, str] | None = None


class _Variant:
    """The only attribute ``load_settings`` reads off a context."""

    def __init__(self, variant_name: str) -> None:
        self.variant_name = variant_name


def _node_records(network: Any) -> list[tuple[str, float]]:
    graph = network.graph
    records: list[tuple[str, float]] = []
    for index in graph.node_indices():
        payload = dict(graph[index])
        identity = payload.get("dna_sequence")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Every vertex must carry a dna_sequence identity.")
        value = payload.get(network.scalar_attr)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Vertex {identity!r} has non-numeric fitness.")
        records.append((identity, float(value)))
    if not records:
        raise ValueError("A descriptive-statistics graph has no nodes.")
    if len({identity for identity, _ in records}) != len(records):
        raise ValueError("Vertex identities are not unique.")
    return records


def directed_uphill(network: Any) -> list[tuple[str, str]]:
    """Orient each undirected edge from lower fitness to higher fitness.

    Equal fitness points from the lexicographically smaller DNA identity to the
    larger one, so each undirected edge becomes exactly one arc. The sum of
    in-degrees then equals the sum of out-degrees, and both equal the edge count.
    """
    graph = network.graph
    fitness = dict(_node_records(network))
    arcs: list[tuple[str, str]] = []
    for left, right in graph.edge_list():
        source = graph[left]["dna_sequence"]
        target = graph[right]["dna_sequence"]
        if source not in fitness or target not in fitness:
            raise ValueError("An edge endpoint is missing from the vertex set.")
        lower_first = fitness[source] < fitness[target] or (
            fitness[source] == fitness[target] and source < target
        )
        arcs.append((source, target) if lower_first else (target, source))
    return arcs


def summarize_graph(column: GraphColumn) -> dict[str, int | float | None]:
    """Row values for one graph. Undefined cells are ``None`` (rendered as an en dash)."""
    records = _node_records(column.network)
    fitness = [value for _, value in records]
    nodes = len(fitness)
    edges = int(column.network.edge_count)
    arcs = directed_uphill(column.network)
    if len(arcs) != edges:
        raise ValueError(
            f"{column.label} directed edge count {len(arcs)} != undirected {edges}."
        )
    identities = [identity for identity, _ in records]
    if column.classes is not None:
        functional = 0
        nonfunctional = 0
        for identity in identities:
            if identity not in column.classes:
                raise ValueError(
                    f"{column.label} sequence {identity!r} has no functional class."
                )
            kind = column.classes[identity]
            if kind == "functional":
                functional += 1
            elif kind == "non_functional":
                nonfunctional += 1
            elif kind != "uncertain":
                raise ValueError(
                    f"{column.label} sequence {identity!r} has class {kind!r}."
                )
        functional_count: int | None = functional
        nonfunctional_count: int | None = nonfunctional
        percent: float | None = 100.0 * functional / nodes
        threshold: float | None = column.threshold
    elif column.threshold is None:
        functional_count = None
        nonfunctional_count = None
        percent = None
        threshold = None
    else:
        functional_count = sum(value >= column.threshold for value in fitness)
        nonfunctional_count = nodes - functional_count
        percent = 100.0 * functional_count / nodes
        threshold = float(column.threshold)
    return {
        "number of nodes": nodes,
        "number of edges": edges,
        "minimum fitness": min(fitness),
        "maximum fitness": max(fitness),
        "mean fitness": mean(fitness),
        "median fitness": median(fitness),
        "functional threshold": threshold,
        "functional count": functional_count,
        "non-functional count": nonfunctional_count,
        "% functional": percent,
        "mean degree (undirected)": (2.0 * edges) / nodes,
        "mean in/out degree (in = out)": edges / nodes,
    }


def statistics_table(
    columns: Sequence[GraphColumn],
) -> dict[str, dict[str, int | float | None]]:
    """Metric name to a mapping of column label to value."""
    labels = tuple(column.label for column in columns)
    if labels != COLUMNS:
        raise ValueError(f"Columns must be {COLUMNS}, got {labels}.")
    by_column = {column.label: summarize_graph(column) for column in columns}
    table = {row: {label: by_column[label][row] for label in labels} for row in ROWS}
    for row in _UNDEFINED_ON_FULL:
        if table[row]["Full"] is not None:
            raise ValueError(f"Full graph {row} must be undefined.")
    return table


def _cell(value: object) -> str:
    if value is None:
        return EN_DASH
    if isinstance(value, bool):
        raise TypeError(f"Unexpected boolean cell {value!r}.")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _manuscript_line(
    table: Mapping[str, Mapping[str, object]], label: str, source: str
) -> tuple[str, ...]:
    return (label, *(_cell(table[source][column]) for column in COLUMNS))


def _delimited(table: Mapping[str, Mapping[str, object]], delimiter: str) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    writer.writerow(("metric", *COLUMNS))
    for label, source in MANUSCRIPT_ROWS:
        writer.writerow(_manuscript_line(table, label, source))
    return stream.getvalue()


def _html_document(table: Mapping[str, Mapping[str, object]]) -> str:
    headers = ("metric", *COLUMNS)
    head = "".join(f"<th>{html.escape(name)}</th>" for name in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(cell)}</td>"
            for cell in _manuscript_line(table, label, source)
        )
        + "</tr>"
        for label, source in MANUSCRIPT_ROWS
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Graph descriptive statistics</title></head><body>"
        "<h1>Graph descriptive statistics</h1>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</body></html>\n"
    )


def _markdown(table: Mapping[str, Mapping[str, object]]) -> str:
    headers = ("metric", *COLUMNS)
    lines = [
        "# Graph descriptive statistics",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for label, source in MANUSCRIPT_ROWS:
        lines.append("| " + " | ".join(_manuscript_line(table, label, source)) + " |")
    lines.extend(["", _readme().split("\n", 1)[1]])
    return "\n".join(lines)


def _readme() -> str:
    return """# Graph descriptive statistics

Manuscript Table 1. Four graphs, one column each. Every cell is computed from
that column's network. An en dash means the quantity is undefined for that graph.
`summarize_graph` still computes fitness, threshold and functional-count metrics;
they are not rows of this table.

| Column | Graph |
| --- | --- |
| Full | Every sequence in the dataset. No functional threshold is applied. |
| Papkou | Papkou threshold, then the largest connected component of the edges that touch a vertex at or above that threshold. A vertex is functional when its raw fitness is at least the threshold. |
| Topographer | The Topographer point-estimate graph (median-reflected threshold) (`threshold=reflected_fdr=none`): threshold `T = m + (m − g)` from the Full-graph raw fitness, vertices with raw fitness at least `T` are functional, same edge and largest-component rule. No FDR classes. |
| Topographer + FDR | Built (Benjamini–Yekutieli, nominal q = 0.05) by loading variant `threshold=reflected_fdr=by_q=0.05` and selecting with `prepare_selected_landscape`. `% functional` follows that variant's saved `functional` class, not a relabeling of the Topographer column. |

## Rows

- **nodes**: vertices in the graph.
- **edges**: undirected edges in the graph.
- **% functional**: `100 * functional count / number of nodes`. Undefined for Full. For Papkou and Topographer, functional is raw fitness at or above the threshold. For Topographer + FDR it is the saved `functional` class. Uncertain BY vertices are not functional.
- **mean degree (undirected)**: `2 * number of edges / number of nodes`.
- **mean in/out degree (directed uphill)**: each undirected edge is oriented once, from the endpoint of lower raw fitness to the endpoint of higher raw fitness (equal fitness points from the lexicographically smaller DNA identity to the larger). Mean in-degree equals mean out-degree because every edge contributes one to each sum, so the row is `number of edges / number of nodes`.
"""


def render_png(
    table: Mapping[str, Mapping[str, object]],
    path: Path,
    *,
    font_family: str,
    font_size: float,
    dpi: float,
) -> None:
    """Draw the table with the shared figure font and PNG dpi."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    family = font_family.split(",")[0].strip() or "sans-serif"
    cell_text = [
        list(_manuscript_line(table, label, source)[1:])
        for label, source in MANUSCRIPT_ROWS
    ]
    figure, axes = plt.subplots(figsize=(14, 2.8))
    axes.axis("off")
    drawn = axes.table(
        cellText=cell_text,
        rowLabels=[label for label, _source in MANUSCRIPT_ROWS],
        colLabels=list(COLUMNS),
        loc="center",
        cellLoc="center",
    )
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(font_size)
    drawn.scale(1, 1.4)
    for cell in drawn.get_celld().values():
        cell.get_text().set_fontfamily(family)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def export_table(
    table: Mapping[str, Mapping[str, object]],
    directory: Path,
    *,
    provenance: Mapping[str, Any],
    font_family: str = "Arial, Liberation Sans, Helvetica, sans-serif",
    font_size: float = 15,
    dpi: float = 300,
) -> list[Path]:
    """Write the manuscript table and the descriptive bundle beside it."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{TABLE_ID}.csv"
    tsv_path = directory / f"{TABLE_ID}.tsv"
    html_path = directory / f"{TABLE_ID}.html"
    md_path = directory / f"{TABLE_ID}.md"
    readme_path = directory / "README.md"
    png_path = directory / f"{TABLE_ID}.png"
    sidecar_path = directory / f"{TABLE_ID}.json"
    csv_path.write_text(_delimited(table, ","), encoding="utf-8")
    tsv_path.write_text(_delimited(table, "\t"), encoding="utf-8")
    html_path.write_text(_html_document(table), encoding="utf-8")
    md_path.write_text(_markdown(table), encoding="utf-8")
    readme_path.write_text(_readme(), encoding="utf-8")
    render_png(
        table,
        png_path,
        font_family=font_family,
        font_size=font_size,
        dpi=dpi,
    )
    sidecar_path.write_text(
        json.dumps(
            {"schema": SIDECAR_SCHEMA, "table_id": TABLE_ID, **provenance}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        csv_path,
        tsv_path,
        html_path,
        md_path,
        readme_path,
        png_path,
        sidecar_path,
    ]


def _load_settings(context: Any) -> dict[str, Any]:
    from ..analysis import load_settings

    return load_settings(context)


def _prepare_clamped(network: Any, threshold: float) -> Any:
    from ..landscape.selected_landscape import prepare_clamped_landscape

    return prepare_clamped_landscape(network, threshold)


def _prepare_selected(network: Any, settings: Mapping[str, Any], context: Any) -> Any:
    from ..landscape.selection import prepare_selected_landscape

    return prepare_selected_landscape(network, settings, context=context)


def _selection_identity(selection: Any) -> dict[str, Any]:
    from ..landscape.selection import selection_identity

    return selection_identity(selection)


def build_full_network(settings: Mapping[str, Any], context: Any | None) -> Any:
    """The Full sequence graph, by the same workflow the comparison stage uses."""
    import polars as pl

    from ..analysis import ROOT, dataset_input_path
    from ..papkou2023 import Papkou2023Workflow

    scientific = settings["analysis"]
    dataset = scientific["dataset"]
    workflow = Papkou2023Workflow(
        landscape="full",
        fitness_wt=pl.read_parquet(
            dataset_input_path(dataset["fitness_wt"], context, root=ROOT)
        ),
        fitness_diff=pl.read_parquet(
            dataset_input_path(dataset["fitness_diff"], context, root=ROOT)
        ),
    )
    return workflow.build_network()


def assemble_columns(
    settings: Mapping[str, Any],
    context: Any | None = None,
    *,
    full_network: Any | None = None,
) -> tuple[tuple[GraphColumn, ...], dict[str, Any]]:
    """Build the four columns. The BY column loads its own variant declaration."""
    network = (
        build_full_network(settings, context) if full_network is None else full_network
    )
    published_threshold = float(
        settings["analysis"]["clamped"]["published_functional_threshold"]
    )
    published = _prepare_clamped(network, published_threshold)
    point_settings = _load_settings(_Variant(POINT_VARIANT))
    by_settings = _load_settings(_Variant(BY_VARIANT))
    point_prepared, point_selection = _prepare_selected(
        network, point_settings, context
    )
    by_prepared, by_selection = _prepare_selected(network, by_settings, context)
    if (
        point_selection.rule != "median_reflected"
        or point_selection.classes is not None
    ):
        raise ValueError(
            "The Topographer column must be the median-reflected point estimate."
        )
    if by_selection.rule != "fdr_fixed_cutoff" or by_selection.classes is None:
        raise ValueError(
            "The BY column must be built by the fdr_fixed_cutoff selection, "
            f"not {by_selection.rule!r}."
        )
    if (
        by_settings["analysis"]["clamped"]["candidate_functional_threshold"].get(
            "cutoff"
        )
        != "median_reflected"
    ):
        raise ValueError("The BY variant cutoff must be median_reflected.")
    columns = (
        GraphColumn("Full", network, None, None),
        GraphColumn(COLUMNS[1], published.network, published_threshold, None),
        GraphColumn(
            COLUMNS[2],
            point_prepared.network,
            float(point_selection.threshold.threshold),
            None,
        ),
        GraphColumn(
            COLUMNS[3],
            by_prepared.network,
            float(by_selection.threshold.threshold),
            by_selection.classes,
        ),
    )
    provenance = {
        "columns": [
            {"display": "Full", "graph": "full", "functional_rule": None},
            {
                "display": COLUMNS[1],
                "graph": "published",
                "threshold_source": "analysis.clamped.published_functional_threshold",
                "functional_rule": "raw_fitness_at_or_above_threshold",
            },
            {
                "display": COLUMNS[2],
                "variant": POINT_VARIANT,
                "functional_rule": "raw_fitness_at_or_above_threshold",
                "selection": _selection_identity(point_selection),
            },
            {
                "display": COLUMNS[3],
                "variant": BY_VARIANT,
                "functional_rule": "saved_fdr_class",
                "selection": _selection_identity(by_selection),
            },
        ]
    }
    return columns, provenance


def output_directory(context: Any) -> Path:
    from ..output_paths import table_path

    return table_path(context.output_root, TABLE_ID).parent / TABLE_ID


def publish(context: Any) -> list[Path]:
    """Compute the table from the current graphs and write it under the canonical shelf."""
    settings = _load_settings(context)
    rendering = settings["figures"]["rendering"]
    columns, provenance = assemble_columns(settings, context)
    table = statistics_table(columns)
    directory = output_directory(context)
    written = export_table(
        table,
        directory,
        provenance=provenance,
        font_family=str(
            rendering.get(
                "font_family", "Arial, Liberation Sans, Helvetica, sans-serif"
            )
        ),
        font_size=float(rendering.get("font_size", 15)),
        dpi=float(settings["figures"].get("png_dpi", 300)),
    )
    # The namer reads flat stems beside canonical/README.md, not the bundle directory.
    shelf = directory.parent
    copied: list[Path] = []
    for path in written:
        if (
            path.suffix in {".csv", ".html", ".md", ".json"}
            and path.name != "README.md"
        ):
            target = shelf / path.name
            shutil.copyfile(path, target)
            copied.append(target)
    return [*written, *copied]
