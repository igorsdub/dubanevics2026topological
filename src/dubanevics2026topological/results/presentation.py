"""Context-bound access to identified products for presentation callers."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import polars as pl

from .context import RunContext
from .lookup import LoadedProduct, load


class SavedTreeLayout(SimpleNamespace):
    """Renderer view backed by a verified saved tree layout."""


@dataclass(frozen=True)
class PresentationProducts:
    """Resolve products in one explicit context without exposing storage paths."""

    context: RunContext

    def load(
        self,
        product: str,
        label: str | None = None,
        *,
        expected: dict[str, Any] | None = None,
    ) -> LoadedProduct:
        """Use the normal lookup and compatibility checks for every product."""
        return load(product, label, context=self.context, expected=expected)

    def tree_layout(self, label: str | None, *, tree: LoadedProduct) -> dict[str, Any]:
        """Load layout tables only when their verified parent is the requested tree."""
        if (
            tree.product_type != "tree"
            or tree.report.status != "full"
            or not tree.report.byte_integrity
        ):
            raise ValueError("Tree layout requires a fully verified tree product.")
        layout = self.load(
            "tree_layout",
            label,
            expected=(
                {"tree_scientific_id_sha256": tree.manifest.scientific_id_sha256}
                if label is None
                else None
            ),
        )
        if layout.report.status != "full" or not layout.report.byte_integrity:
            raise ValueError("Tree layout product is not fully verified.")
        tree_id = tree.manifest.scientific_id_sha256
        refs = [
            parent
            for parent in layout.manifest.parents
            if parent.get("product_type") == "tree"
        ]
        if layout.manifest.scientific_identity.get(
            "tree_scientific_id_sha256"
        ) != tree_id or not any(
            ref.get("scientific_id_sha256") == tree_id
            and ref.get("label") == tree.label
            for ref in refs
        ):
            raise ValueError(
                "Tree layout parent identity does not match the requested tree."
            )
        tables = {
            name.removesuffix(".parquet"): pl.read_parquet(path)
            for name, path in layout.artifacts.items()
            if name.endswith(".parquet")
        }
        metadata = json.loads(
            layout.artifacts["plot_metadata.json"].read_text(encoding="utf-8")
        )
        if not isinstance(metadata, dict) or set(metadata) & set(tables):
            raise ValueError("Invalid or conflicting tree layout plot metadata.")
        return {**tables, **metadata}

    def tree_layout_view(
        self, label: str | None, *, tree: LoadedProduct
    ) -> SavedTreeLayout:
        """Restore the renderer-facing plotting object from a verified layout."""
        from topographer.plotting import TreePlotData

        values = self.tree_layout(label, tree=tree)
        names = {field.name for field in fields(TreePlotData)}
        missing = names - values.keys()
        if missing:
            raise ValueError(
                "Tree layout is missing plotting fields "
                f"{sorted(missing)}; run `tree-layout-products` first."
            )
        view = SavedTreeLayout(
            plot_data=TreePlotData(**{name: values[name] for name in names}),
            **{name: value for name, value in values.items() if name not in names},
        )
        if tree.manifest.scientific_identity.get("tree") == "band":
            nodes = dict(
                view.plot_data.node_table.select(
                    "tree_node_index", "dna_sequence"
                ).iter_rows()
            )
            branches = pl.read_parquet(tree.artifacts["branches.parquet"])
            counts = {}
            for upper, lower, count in branches.select(
                "source_tree_node_identity",
                "target_tree_node_identity",
                "branch_vertex_count",
            ).iter_rows():
                key = frozenset((upper, lower))
                if key in counts or upper is None or lower is None or count is None:
                    raise ValueError(
                        "Band tree contains invalid branch identities or counts."
                    )
                counts[key] = count
            rows = []
            for source, target in view.plot_data.edge_table.select(
                "source_tree_node_index", "target_tree_node_index"
            ).iter_rows():
                try:
                    count = counts[frozenset((nodes[source], nodes[target]))]
                except KeyError as exc:
                    raise ValueError(
                        "Band layout edge is absent from the saved tree."
                    ) from exc
                rows.append(
                    {
                        "source_tree_node_index": source,
                        "target_tree_node_index": target,
                        "branch_vertex_count": count,
                    }
                )
            if len(rows) != len(counts):
                raise ValueError("Band layout and saved tree have different branches.")
            view.branch_table = pl.DataFrame(
                rows,
                schema={
                    "source_tree_node_index": pl.Int64,
                    "target_tree_node_index": pl.Int64,
                    "branch_vertex_count": pl.Int64,
                },
            )
        return view

    def landscape_population(self, landscape: LoadedProduct) -> pl.DataFrame:
        """Read the saved selected vertices and their raw fitness values."""
        if (
            landscape.product_type != "landscape"
            or landscape.report.status != "full"
            or not landscape.report.byte_integrity
        ):
            raise ValueError("Population requires a fully verified landscape product.")
        vertices = pl.read_parquet(landscape.artifacts["vertices.parquet"])
        if not {"dna_sequence", "raw_fitness"} <= set(vertices.columns):
            raise ValueError(
                "Landscape product lacks sequence identities or raw fitness."
            )
        population = vertices.select(
            "dna_sequence", pl.col("raw_fitness").alias("scalar")
        )
        if (
            not population.height
            or population["dna_sequence"].null_count()
            or (population["dna_sequence"] == "").any()
            or population["dna_sequence"].n_unique() != population.height
            or population["scalar"].null_count()
            or not population["scalar"].is_finite().all()
        ):
            raise ValueError(
                "Landscape population has invalid identities or fitness values."
            )
        return population

    def tree_annotations(self, *, tree: LoadedProduct) -> Any:
        """Read biological annotation tables for exactly the requested tree."""
        from ..analysis.biological_annotations import BiologicalAnnotations

        if (
            tree.product_type != "tree"
            or tree.report.status != "full"
            or not tree.report.byte_integrity
        ):
            raise ValueError("Annotations require a fully verified tree product.")
        tree_id = tree.manifest.scientific_id_sha256
        product = self.load(
            "tree_annotations", expected={"tree_scientific_id_sha256": tree_id}
        )
        if (
            product.report.status != "full"
            or not product.report.byte_integrity
            or not any(
                ref.get("product_type") == "tree"
                and ref.get("label") == tree.label
                and ref.get("scientific_id_sha256") == tree_id
                for ref in product.manifest.parents
            )
        ):
            raise ValueError("Annotation product has an unverified or mismatched tree.")
        return BiologicalAnnotations(
            branches=pl.read_parquet(product.artifacts["branches.parquet"]),
            pairs=pl.read_parquet(product.artifacts["pairs.parquet"]),
            critical_points=pl.read_parquet(
                product.artifacts["critical_points.parquet"]
            ),
            summary=json.loads(
                product.artifacts["summary.json"].read_text(encoding="utf-8")
            ),
        )
