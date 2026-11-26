"""Saved-product adapter for the D10 tree views (Spec 2, Section 6, view 1).

The existing tree renderers (``tree_chart``, ``persistence_chart``) take
Topographer analysis objects. This adapter builds the same inputs from saved
products only, so a figure never rebuilds a landscape or a tree:

- ``tree_analysis``: the verified ``tree_layout`` as ``SavedTreeLayout`` (plot
  data, branch table, wild-type marker), plus the stop-marker table read from the
  layout's own node table;
- ``persistence_table``: the tree product's ``persistence_pairs.parquet`` as the
  Topographer persistence schema (finite ordinary H0 pairs; the global maximum's
  pair is closed at the global minimum by convention, as saved);
- ``scalar_domain``: the extent of the saved landscape population's raw fitness.
  A band tree has no landscape parent: its landscape is its nominal parent's,
  while its layout and pairs are its own.
- ``threshold``: the landscape's functional threshold (``None`` for the full graph).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import polars as pl

from ..results import PresentationProducts, RunContext

PERSISTENCE_COLUMNS = {
    "birth_label": pl.String,
    "death_label": pl.String,
    "birth_scalar": pl.Float64,
    "death_scalar": pl.Float64,
    "persistence": pl.Float64,
    "pair_kind": pl.String,
}
_OPTIONAL_COLUMNS = ("birth_event", "death_event")


@dataclass(frozen=True)
class SavedTreeInputs:
    """Everything the ``tree_persistence`` renderer reads, from saved products."""

    tree_label: str
    tree_id: str
    layout_id: str
    landscape_label: str
    landscape_id: str
    tree_analysis: Any
    persistence_table: pl.DataFrame
    scalar_domain: tuple[float, float]
    vertex_count: int
    threshold: float | None = None
    fitness_values: pl.Series | None = None

    @property
    def analysis(self) -> SimpleNamespace:
        """The slice of an analysis object the tree renderer reads."""
        return SimpleNamespace(scalar_domain=self.scalar_domain)


def persistence_table_from_pairs(pairs: pl.DataFrame) -> pl.DataFrame:
    """Reduce saved persistence pairs to the Topographer persistence schema."""
    missing = set(PERSISTENCE_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"Persistence pairs lack columns {sorted(missing)}.")
    columns = [*PERSISTENCE_COLUMNS, *(c for c in _OPTIONAL_COLUMNS if c in pairs)]
    table = pairs.select(columns).with_columns(
        pl.col(name).cast(dtype) for name, dtype in PERSISTENCE_COLUMNS.items()
    )
    numeric = ("birth_scalar", "death_scalar", "persistence")
    if (
        not table.height
        or any(table[c].null_count() or not table[c].is_finite().all() for c in numeric)
        or table["birth_label"].null_count()
        or table["birth_label"].n_unique() != table.height
    ):
        raise ValueError("Persistence pairs must be finite with unique births.")
    if (table["death_scalar"] > table["birth_scalar"]).any():
        raise ValueError("A persistence pair dies above its birth.")
    return table


def scalar_domain_from_population(population: pl.DataFrame) -> tuple[float, float]:
    """Extent of the saved landscape population's raw fitness."""
    return float(population["scalar"].min()), float(population["scalar"].max())


def functional_threshold(landscape: Any) -> float | None:
    """The saved landscape's functional threshold; ``None`` for the full graph."""
    selection = landscape.manifest.scientific_identity.get("landscape_selection", {})
    threshold = selection.get("threshold")
    return None if threshold is None else float(threshold)


def landscape_source_tree(products: PresentationProducts, tree: Any) -> Any:
    """The tree that names the landscape: itself, or a band tree's nominal parent."""
    if any(ref.get("product_type") == "landscape" for ref in tree.manifest.parents):
        return tree
    refs = [ref for ref in tree.manifest.parents if ref.get("product_type") == "tree"]
    if len(refs) != 1:
        raise ValueError("Band tree must identify exactly one nominal tree parent.")
    nominal = products.load("tree", refs[0]["label"])
    if (
        nominal.report.status != "full"
        or not nominal.report.byte_integrity
        or nominal.manifest.scientific_id_sha256 != refs[0]["scientific_id_sha256"]
    ):
        raise ValueError("Band tree nominal parent is missing or mismatched.")
    return nominal


def stop_marker_table(tree_analysis: Any) -> pl.DataFrame:
    """Stop-codon sequences at their layout positions, from the saved node table.

    Every stop node is kept with its kind (``display_event``); the figure layer keeps
    the kinds in ``[rendering] stop_codon_tree_node_kinds`` (``stop_codon_nodes``).
    """
    return tree_analysis.plot_data.node_table.filter(pl.col("has_stop_codon")).select(
        "x", "y", "dna_sequence", "aa_sequence", "display_event"
    )


def load_saved_tree_inputs(context: RunContext, tree_label: str) -> SavedTreeInputs:
    """Load a verified tree, its layout, landscape population and pairs."""
    from .saved_figure_inputs import _landscape_for_tree, _layout

    products = PresentationProducts(context)
    tree = products.load("tree", tree_label)
    if tree.report.status != "full" or not tree.report.byte_integrity:
        raise ValueError(f"Tree product {tree_label} is not fully verified.")
    landscape = _landscape_for_tree(products, landscape_source_tree(products, tree))
    view, layout_id = _layout(products, tree)
    population = products.landscape_population(landscape)
    view.all_stop_marker_table = stop_marker_table(view)
    pairs = persistence_table_from_pairs(
        pl.read_parquet(tree.artifacts["persistence_pairs.parquet"])
    )
    view.persistence_table = pairs
    return SavedTreeInputs(
        tree_label=tree.label,
        tree_id=tree.manifest.scientific_id_sha256,
        layout_id=layout_id,
        landscape_label=landscape.label,
        landscape_id=landscape.manifest.scientific_id_sha256,
        tree_analysis=view,
        persistence_table=pairs,
        scalar_domain=scalar_domain_from_population(population),
        vertex_count=population.height,
        threshold=functional_threshold(landscape),
        fitness_values=population["scalar"],
    )
