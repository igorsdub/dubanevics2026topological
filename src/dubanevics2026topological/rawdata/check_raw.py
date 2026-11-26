"""``just check-raw-against <dir>``: one-off comparison of rebuilt raw files with an existing copy.

Compares by value, not by bytes: Parquet frames with ``polars.testing.assert_frame_equal``
(row order included), the peak sets for equality, and the GraphML nodes and edges with
their attributes. Exits 1 and lists every difference; a mismatch means stop and report.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import polars as pl
from polars.testing import assert_frame_equal

REL_TOL = 1e-12

Graph = tuple[dict[str, dict[str, object]], Counter]


def read_graphml(path: Path) -> Graph:
    """Nodes (name -> attributes) and edges ((source name, target name, attributes) -> count)."""
    keys: dict[str, tuple[str, str]] = {}
    ids: dict[str, str] = {}
    nodes: dict[str, dict[str, object]] = {}
    edges: Counter = Counter()

    def convert(key: str, text: str | None) -> object:
        _name, kind = keys[key]
        text = text or ""
        if kind in ("double", "float"):
            return float(text)
        if kind in ("int", "long"):
            return int(text)
        if kind == "boolean":
            return text.strip().lower() in ("true", "1")
        return text

    def attributes(element: ET.Element) -> dict[str, object]:
        return {
            keys[d.get("key")][0]: convert(d.get("key"), d.text)
            for d in element
            if d.tag.rsplit("}", 1)[-1] == "data"
        }

    pending: list[tuple[str, str, dict[str, object]]] = []
    for _, element in ET.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "key":
            keys[element.get("id")] = (
                element.get("attr.name"),
                element.get("attr.type"),
            )
        elif tag == "node":
            attrs = attributes(element)
            name = str(attrs.get("name", element.get("id")))
            ids[element.get("id")] = name
            nodes[name] = attrs
            element.clear()
        elif tag == "edge":
            pending.append(
                (element.get("source"), element.get("target"), attributes(element))
            )
            element.clear()
    for source, target, attrs in pending:
        edges[(ids[source], ids[target], tuple(sorted(attrs.items())))] += 1
    return nodes, edges


def _equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(
            a, b, rel_tol=REL_TOL, abs_tol=0.0
        )
    return a == b


def compare_graphs(rebuilt: Graph, existing: Graph) -> list[str]:
    problems = []
    (n_new, e_new), (n_old, e_old) = rebuilt, existing
    if set(n_new) != set(n_old):
        problems.append(
            f"node sets differ: {len(set(n_new) - set(n_old))} only rebuilt, "
            f"{len(set(n_old) - set(n_new))} only existing"
        )
    bad = [
        name
        for name in set(n_new) & set(n_old)
        if n_new[name].keys() != n_old[name].keys()
        or not all(_equal(n_new[name][k], n_old[name][k]) for k in n_new[name])
    ]
    if bad:
        problems.append(
            f"{len(bad)} nodes differ in attributes, e.g. {sorted(bad)[:3]}"
        )

    def pair_map(edges: Counter) -> dict[tuple[str, str], list[dict[str, object]]]:
        out: dict[tuple[str, str], list[dict[str, object]]] = {}
        for (s, t, attrs), count in edges.items():
            out.setdefault((s, t), []).extend([dict(attrs)] * count)
        return out

    m_new, m_old = pair_map(e_new), pair_map(e_old)
    if set(m_new) != set(m_old):
        problems.append(
            f"edge sets differ: {len(set(m_new) - set(m_old))} only rebuilt, "
            f"{len(set(m_old) - set(m_new))} only existing"
        )
    bad_edges = []
    for pair in set(m_new) & set(m_old):
        a, b = m_new[pair], m_old[pair]
        ok = len(a) == len(b) and all(
            x.keys() == y.keys() and all(_equal(x[k], y[k]) for k in x)
            for x, y in zip(a, b)
        )
        if not ok:
            bad_edges.append(pair)
    if bad_edges:
        problems.append(
            f"{len(bad_edges)} edges differ in attributes, e.g. {sorted(bad_edges)[:3]}"
        )
    return problems


def check_against(rebuilt_dir: Path, existing_dir: Path) -> list[str]:
    """All differences between two raw directories (empty list when they agree)."""
    problems: list[str] = []
    for name in ("papkou2023_fitness_wt.parquet", "papkou2023_fitness_diff.parquet"):
        try:
            assert_frame_equal(
                pl.read_parquet(rebuilt_dir / name),
                pl.read_parquet(existing_dir / name),
                check_row_order=True,
                check_column_order=True,
                check_dtypes=True,
            )
        except AssertionError as error:
            problems.append(f"{name}: {str(error).splitlines()[0]}")
    peaks_new = set(
        pl.read_csv(rebuilt_dir / "papkou2023_peaks.csv", infer_schema=False)["dna"]
    )
    peaks_old = set(
        pl.read_csv(existing_dir / "papkou2023_peaks.csv", infer_schema=False)["dna"]
    )
    if peaks_new != peaks_old:
        problems.append(
            f"papkou2023_peaks.csv: {len(peaks_new - peaks_old)} only rebuilt, "
            f"{len(peaks_old - peaks_new)} only existing"
        )
    problems += [
        f"papkou2023_graph.graphml: {p}"
        for p in compare_graphs(
            read_graphml(rebuilt_dir / "papkou2023_graph.graphml"),
            read_graphml(existing_dir / "papkou2023_graph.graphml"),
        )
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "existing", help="Directory holding the existing copy of the raw files."
    )
    parser.add_argument(
        "--rebuilt", default="data/raw", help="Rebuilt directory (default data/raw)."
    )
    args = parser.parse_args(argv)
    problems = check_against(Path(args.rebuilt), Path(args.existing))
    if problems:
        print(
            "Rebuilt files DIFFER from the existing copy:\n  " + "\n  ".join(problems)
        )
        return 1
    print("Rebuilt files equal the existing copy by value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
