"""Tree layout publication checks its tree identity before writing."""

from dataclasses import dataclass, fields
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.build.tree_layout import (
    _members,
    build_tree_layout,
    publish_tree_layout,
)
from dubanevics2026topological.landscape.products import _publish
from dubanevics2026topological.results import PresentationProducts, RunContext, load


def test_saved_vertex_identity_is_required_and_unique():
    assert _members(
        pl.DataFrame({"vertex_identity": ["A"], "scalar": [0.5]}), "scalar"
    ) == {"A": 0.5}
    with pytest.raises(ValueError, match="missing or duplicate"):
        _members(pl.DataFrame({"vertex_identity": [None], "scalar": [0.5]}), "scalar")


def test_layout_rejects_label_fields_that_differ_from_parent(tmp_path):
    tree = SimpleNamespace(
        product_type="tree",
        report=SimpleNamespace(status="full", byte_integrity=True),
        manifest=SimpleNamespace(
            scientific_identity={"graph": "full", "tree": "nominal"}
        ),
    )
    with pytest.raises(ValueError, match="label fields differ"):
        publish_tree_layout(
            tree=tree,
            tables={},
            label_fields={"graph": "full", "tree": "band"},
            layout_code_paths=(),
            context=RunContext(root=tmp_path, env={}),
        )


def _fixture(tmp_path):
    context = RunContext(root=tmp_path, env={})
    critical = pl.DataFrame({"dna_sequence": ["A", "B"], "scalar": [1.0, 0.0]})
    members = pl.DataFrame(
        {
            "dna_sequence": ["C"],
            "scalar": [0.5],
            "source_tree_node_index": [0],
            "target_tree_node_index": [1],
        }
    )
    saved_members = members.drop(
        "source_tree_node_index", "target_tree_node_index"
    ).with_columns(
        pl.lit("A").alias("source_tree_node_identity"),
        pl.lit("B").alias("target_tree_node_identity"),
    )
    branches = pl.DataFrame(
        {
            "source_tree_node_identity": ["A"],
            "target_tree_node_identity": ["B"],
            "branch_vertex_count": [2],
        }
    )

    def write(directory):
        critical.write_parquet(directory / "critical_nodes.parquet")
        saved_members.write_parquet(directory / "branch_members.parquet")
        branches.write_parquet(directory / "branches.parquet")

    label_fields = {"graph": "full", "tree": "nominal"}
    path, _ = _publish(
        "tree",
        label_fields,
        label_fields,
        context=context,
        input_data_sha256s={},
        parents=(),
        code_paths=(),
        write_artifacts=write,
    )
    tree = load("tree", path.name, context=context)
    nodes = critical.with_columns(
        pl.Series("tree_node_index", [0, 1]), pl.col("scalar").alias("y")
    )
    edges = pl.DataFrame({"source_tree_node_index": [0], "target_tree_node_index": [1]})
    segments = pl.DataFrame(
        {
            "source_tree_node_index": [0, 0],
            "target_tree_node_index": [1, 1],
            "segment_kind": ["vertical", "horizontal"],
            "y_start": [0.0, 1.0],
            "y_end": [1.0, 1.0],
        }
    )
    plot = SimpleNamespace(
        node_table=nodes, edge_table=edges, edge_segment_table=segments
    )

    class Graph:
        def __init__(self):
            self.nodes = [
                {"dna_sequence": "A", "scalar": 1.0},
                {"dna_sequence": "B", "scalar": 0.0},
            ]

        def node_indices(self):
            return range(len(self.nodes))

        def __getitem__(self, index):
            return self.nodes[index]

    source = SimpleNamespace(
        tree=Graph(),
        plot_data=plot,
        wild_type_marker_table=pl.DataFrame({"dna_sequence": ["A"]}),
        branch_table=edges,
        critical_node_table=critical,
        branch_member_table=members,
    )
    return context, tree, source


def test_layout_rejects_internal_scalar_difference(tmp_path):
    context, tree, source = _fixture(tmp_path)
    source.tree.nodes[0]["scalar"] += 1e-12
    with pytest.raises(
        ValueError,
        match="internal tree scalar mismatch for A; maximum absolute difference",
    ):
        build_tree_layout(source, tree=tree, context=context)


def test_layout_rejects_segment_scalar_difference(tmp_path):
    context, tree, source = _fixture(tmp_path)
    source.plot_data.edge_segment_table = (
        source.plot_data.edge_segment_table.with_columns(
            (pl.col("y_start") + 1e-12).alias("y_start")
        )
    )
    with pytest.raises(ValueError, match="plot scalar mismatch"):
        build_tree_layout(source, tree=tree, context=context)


def test_layout_round_trip_preserves_tables_and_parent(tmp_path):
    context, tree, source = _fixture(tmp_path)

    @dataclass
    class Plot:
        node_table: pl.DataFrame
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame
        root_node_index: int

    source.plot_data = Plot(
        source.plot_data.node_table,
        source.plot_data.edge_table,
        source.plot_data.edge_segment_table,
        1,
    )
    events = []
    path = build_tree_layout(source, tree=tree, context=context, progress=events.append)
    assert events == ["layout builder complete", "reconciliation complete", "published"]
    tables = PresentationProducts(context).tree_layout(path.name, tree=tree)
    assert tables["root_node_index"] == 1
    assert tables["node_table"].equals(source.plot_data.node_table)
    assert tables["branch_member_table"].equals(source.branch_member_table)
    assert tables["wild_type_marker_table"].equals(source.wild_type_marker_table)
    other = SimpleNamespace(
        product_type="tree",
        report=tree.report,
        manifest=SimpleNamespace(scientific_id_sha256="f" * 64),
        label=tree.label,
    )
    with pytest.raises(ValueError, match="parent identity"):
        PresentationProducts(context).tree_layout(path.name, tree=other)


def test_layout_rejects_membership_mismatch(tmp_path):
    context, tree, source = _fixture(tmp_path)
    source.branch_member_table = source.branch_member_table.with_columns(
        pl.lit("D").alias("dna_sequence")
    )
    with pytest.raises(ValueError, match="membership differs"):
        build_tree_layout(source, tree=tree, context=context)


def test_layout_rejects_changed_branch_assignment(tmp_path):
    context, tree, source = _fixture(tmp_path)
    source.branch_member_table = source.branch_member_table.with_columns(
        pl.lit(0).alias("target_tree_node_index")
    )
    with pytest.raises(ValueError, match="branch assignments differ"):
        build_tree_layout(source, tree=tree, context=context)


def test_band_layout_round_trip_uses_both_tree_parents(tmp_path, monkeypatch):
    context, nominal, source = _fixture(tmp_path)
    assignments = pl.DataFrame(
        {
            "dna_sequence": ["A", "B", "C"],
            "raw_fitness": [1.0, 0.0, 0.5],
            "upper_sequence": ["A", "B", "A"],
            "lower_sequence": ["B", "B", "B"],
            "location": ["critical", "root", "absorbed_branch"],
        }
    )
    pairs = pl.DataFrame({"birth_label": ["A"], "persistence": [1.0]})

    def write(directory):
        for name in (
            "critical_nodes.parquet",
            "branch_members.parquet",
            "branches.parquet",
        ):
            frame = pl.read_parquet(nominal.artifacts[name])
            if name == "branch_members.parquet":
                frame = frame.with_columns(pl.lit("absorbed_branch").alias("location"))
            frame.write_parquet(directory / name)
        assignments.filter(pl.col("location") == "root").write_parquet(
            directory / "root_members.parquet"
        )
        pairs.write_parquet(directory / "persistence_pairs.parquet")

    params = {"graph": "full", "draws": 2, "seed": 1, "tree": "band", "bandq": 0.95}
    parent = {
        "product_type": "tree",
        "label": nominal.label,
        "scientific_id_sha256": nominal.manifest.scientific_id_sha256,
    }
    path, _ = _publish(
        "tree",
        params,
        params,
        context=context,
        input_data_sha256s={},
        parents=(parent,),
        code_paths=(),
        write_artifacts=write,
    )
    band = load("tree", path.name, context=context)
    view = SimpleNamespace(
        plot_data=source.plot_data,
        wild_type_marker_table=source.wild_type_marker_table,
        assignments=assignments,
        pairs=pairs,
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.tree_layout.bootstrap_smoothed_tree.build_band_simplified_view",
        lambda *_: view,
    )
    result = build_tree_layout(
        source, tree=band, nominal_tree=nominal, hills=pl.DataFrame(), context=context
    )
    tables = PresentationProducts(context).tree_layout(result.name, tree=band)
    assert tables["assignments"].equals(assignments)
    assert tables["pairs"].equals(pairs)
    product = load("tree_layout", result.name, context=context)
    assert [ref["scientific_id_sha256"] for ref in product.manifest.parents] == [
        nominal.manifest.scientific_id_sha256,
        band.manifest.scientific_id_sha256,
    ]

    @dataclass
    class Plot:
        node_table: pl.DataFrame
        edge_table: pl.DataFrame
        edge_segment_table: pl.DataFrame

    monkeypatch.setattr("topographer.plotting.TreePlotData", Plot)
    restored = PresentationProducts(context).tree_layout_view(result.name, tree=band)
    assert restored.branch_table["branch_vertex_count"].to_list() == [2]


def test_real_topographer_layout_preserves_all_plot_fields(tmp_path):
    from test_landscape_module import _network

    from dubanevics2026topological import landscape
    from dubanevics2026topological.analysis import describe
    from dubanevics2026topological.build.tree_annotations import build_tree_annotations

    context = RunContext(root=tmp_path, env={})
    prepared = landscape.build_selected_landscape(_network(), 0.0)
    analysis = landscape.extract_nominal_merge_tree(
        prepared, name="fixture", wild_type_sequence="AAAAAAAAA"
    )
    _, tree_path = landscape.save_selected_products(
        prepared,
        analysis,
        parameters={"thr": "reflected", "fdr": "by", "q": 0.05},
        context=context,
        input_data_sha256s={"fitness": "a" * 64},
        parents=(),
    )
    tree = load("tree", tree_path.name, context=context)
    source = analysis.trees["split"]
    vertex_identities = {
        index: str(prepared.network.graph[index]["dna_sequence"])
        for index in prepared.network.graph.node_indices()
    }
    layout = build_tree_layout(
        source, tree=tree, context=context, vertex_identities=vertex_identities
    )
    build_tree_annotations(source, tree=tree, layout_label=layout.name, context=context)
    saved_annotations = PresentationProducts(context).tree_annotations(tree=tree)
    expected_annotations = describe(
        products={"annotations"}, landscape=analysis, landscape_name="candidate_clamped"
    ).annotations
    for name in ("branches", "pairs", "critical_points"):
        assert getattr(saved_annotations, name).equals(
            getattr(expected_annotations, name)
        )
    assert saved_annotations.summary == expected_annotations.summary
    restored = PresentationProducts(context).tree_layout(layout.name, tree=tree)
    restored_view = PresentationProducts(context).tree_layout_view(
        layout.name, tree=tree
    )
    for field in fields(source.plot_data):
        value = getattr(source.plot_data, field.name)
        assert field.name in restored, f"layout omitted plot field {field.name}"
        if isinstance(value, pl.DataFrame):
            assert restored[field.name].equals(value)
        else:
            assert type(restored[field.name]) is type(value)
            assert restored[field.name] == value
        restored_value = getattr(restored_view.plot_data, field.name)
        if isinstance(value, pl.DataFrame):
            assert restored_value.equals(value)
        else:
            assert type(restored_value) is type(value)
            assert restored_value == value
