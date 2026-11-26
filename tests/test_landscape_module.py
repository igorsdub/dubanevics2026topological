"""Compatibility, product, and stage-code contracts for the Landscape module."""

from __future__ import annotations

from dubanevics2026topological.rawdata.value_digest import value_digest

import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from topographer.graph import Graph, ScalarNetwork

from dubanevics2026topological import landscape
from dubanevics2026topological.analysis import clamped_cache
from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint
from dubanevics2026topological.landscape import products as product_module
from dubanevics2026topological.results import RunContext, read_manifest
from dubanevics2026topological.results import load as load_product
from dubanevics2026topological.threshold.fdr_selection import load_fdr_snapshot

_INPUTS = {"fitness": "a" * 64}
_PARENT = {
    "product_type": "fitness",
    "label": "fitness__fixture_hash=fixture",
    "scientific_id_sha256": "b" * 64,
}


def _network() -> ScalarNetwork:
    vertices = pl.DataFrame(
        {
            "dna_sequence": ["AAAAAAAAA", "AAAAAAAAC", "AAAAAACAA", "AAAAAACAC"],
            "fitness": [1.0, 0.1, 0.8, -0.2],
            "aa_sequence": ["KKK", "KKN", "KNK", "KNN"],
            "has_stop_codon": [False, False, False, False],
        }
    )
    edges = pl.DataFrame(
        {
            "source": ["AAAAAAAAA", "AAAAAAAAC", "AAAAAACAA"],
            "target": ["AAAAAAAAC", "AAAAAACAA", "AAAAAACAC"],
        }
    )
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "aa_sequence", "has_stop_codon"),
    )
    return ScalarNetwork(graph, scalar_attr="fitness")


def test_selected_products_use_adr_labels_and_manifest_parent(tmp_path: Path) -> None:
    prepared = landscape.build_selected_landscape(_network(), 0.0)
    analysis = landscape.extract_nominal_merge_tree(
        prepared, name="fixture", wild_type_sequence="AAAAAAAAA"
    )
    context = RunContext(output_root=tmp_path / "output", root=tmp_path)
    graph_path, tree_path = landscape.save_selected_products(
        prepared,
        analysis,
        parameters={"thr": "reflected", "fdr": "by", "q": 0.05},
        context=context,
        input_data_sha256s=_INPUTS,
        parents=(_PARENT,),
    )

    assert graph_path.name.startswith("landscape__thr=reflected_fdr=by_q=0.05_hash=")
    assert tree_path.name.startswith(
        "tree__thr=reflected_fdr=by_q=0.05_tree=nominal_hash="
    )
    graph_manifest = read_manifest(graph_path)
    tree_manifest = read_manifest(tree_path)
    assert graph_manifest.product_type == "landscape"
    assert tree_manifest.product_type == "tree"
    assert tree_manifest.parents == [
        {
            "product_type": "landscape",
            "label": graph_path.name,
            "scientific_id_sha256": graph_manifest.scientific_id_sha256,
        }
    ]
    assert graph_manifest.artifacts.keys() >= {
        "vertices.parquet",
        "edges.parquet",
        "excluded_vertices.parquet",
    }
    assert tree_manifest.artifacts.keys() >= {
        "critical_nodes.parquet",
        "persistence_pairs.parquet",
        "branches.parquet",
        "branch_members.parquet",
    }
    assert "stage_code_digest" not in graph_manifest.scientific_identity
    assert graph_manifest.code_provenance["source_sha256s"]

    full_prepared = landscape.build_selected_landscape(_network(), -1.0)
    full_analysis = landscape.extract_nominal_merge_tree(
        full_prepared, name="full-fixture", wild_type_sequence="AAAAAAAAA"
    )
    full_graph_path, _ = landscape.save_selected_products(
        full_prepared,
        full_analysis,
        parameters={"graph": "full"},
        context=context,
        input_data_sha256s=_INPUTS,
        parents=(_PARENT,),
    )
    full_manifest = read_manifest(full_graph_path)
    assert full_manifest.scientific_identity["full_graph_counts"][
        "component_handling"
    ] == ("all_vertices_and_edges_retained_in_one_component")
    assert full_manifest.scientific_identity["full_graph_counts"] == {
        "input_vertices": 4,
        "selected_vertices": 4,
        "input_edges": 3,
        "selected_edges": 3,
        "excluded_vertices": 0,
        "components": 1,
        "component_handling": "all_vertices_and_edges_retained_in_one_component",
    }
    product_readme = (tmp_path / "output/landscape/README.md").read_text()
    assert "graph=full" in product_readme
    assert "4/4 vertices and 3/3 edges" in product_readme


def test_connected_selected_subset_cannot_be_published_as_full(tmp_path: Path) -> None:
    subset = landscape.build_selected_landscape(_network(), 0.9)
    analysis = landscape.extract_nominal_merge_tree(
        subset, name="subset", wild_type_sequence="AAAAAAAAA"
    )
    with pytest.raises(ValueError, match="every input vertex and edge"):
        landscape.save_selected_products(
            subset,
            analysis,
            parameters={"graph": "full"},
            context=RunContext(output_root=tmp_path / "output", root=tmp_path),
            input_data_sha256s=_INPUTS,
            parents=(_PARENT,),
        )


def test_product_writer_allows_input_provenance_without_fake_parents_and_protects_keys(
    tmp_path: Path,
) -> None:
    prepared = landscape.build_selected_landscape(_network(), 0.0)
    tree = landscape.extract_nominal_merge_tree(
        prepared, name="fixture", wild_type_sequence="AAAAAAAAA"
    )
    context = RunContext(output_root=tmp_path / "output", root=tmp_path)
    graph_path, _ = landscape.save_selected_products(
        prepared,
        tree,
        parameters={"thr": "reflected", "fdr": "none"},
        context=context,
        input_data_sha256s=_INPUTS,
        parents=(),
    )
    assert read_manifest(graph_path).parents == []
    with pytest.raises(ValueError, match="reserved identity fields"):
        landscape.save_selected_products(
            prepared,
            tree,
            parameters={"thr": "reflected", "fdr": "none"},
            scientific_parameters={"parameters": {"thr": "published"}},
            context=context,
            input_data_sha256s=_INPUTS,
            parents=(_PARENT,),
        )


@pytest.mark.parametrize(
    "key",
    (
        "thr",
        "fdr",
        "q",
        "tree",
        "host",
        "run",
        "hash",
        "scientific_identity",
        "data_provenance",
        "parents",
        "scientific_id_sha256",
    ),
)
def test_product_writer_rejects_reserved_identity_overrides_before_writing(
    tmp_path: Path, key: str
) -> None:
    prepared = landscape.build_selected_landscape(_network(), 0.0)
    tree = landscape.extract_nominal_merge_tree(
        prepared, name="fixture", wild_type_sequence="AAAAAAAAA"
    )
    context = RunContext(output_root=tmp_path / "output", root=tmp_path)
    with pytest.raises(ValueError, match="reserved identity fields"):
        landscape.save_selected_products(
            prepared,
            tree,
            parameters={"thr": "reflected", "fdr": "by", "q": 0.05},
            scientific_parameters={key: "conflict"},
            context=context,
            input_data_sha256s=_INPUTS,
            parents=(_PARENT,),
        )
    assert not context.output_root.exists()


def test_product_builder_honours_context_root_and_raw_data_dir(tmp_path: Path) -> None:
    from dubanevics2026topological.landscape.build_products import (
        _configured_input_path,
    )

    repository = tmp_path / "custom-repository"
    raw_dir = tmp_path / "custom-raw"
    context = RunContext(root=repository, raw_data_dir=raw_dir, env={})
    assert _configured_input_path(context, "data/raw/fitness.parquet") == (
        raw_dir / "fitness.parquet"
    )

    snapshot = repository / "fixture-fdr"
    snapshot.mkdir(parents=True)
    table_path = snapshot / (
        "fdr_classification_T=-0.5_cutoff=papkou_original_method=by_q=0.05.parquet"
    )
    pl.DataFrame(
        {"dna_sequence": ["WT", "MUT"], "fdr_class": ["functional", "uncertain"]}
    ).write_parquet(table_path)
    (snapshot / "provenance.json").write_text(
        json.dumps(
            {
                "cutoffs": [
                    {
                        "name": "papkou_original",
                        "source_precision_threshold": -0.5,
                        "applied_threshold": -0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    declaration = {
        "cutoff": "papkou_original",
        "method": "by",
        "q": 0.05,
        "input_dir": "fixture-fdr",
    }
    loaded = load_fdr_snapshot(declaration, wild_type="WT", root=context.root)
    assert loaded.summary["threshold"]["rounded"] == -0.5
    assert loaded.source["table"] == table_path.name


def test_product_identity_tracks_science_but_not_run_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    network = _network()
    prepared = landscape.build_selected_landscape(network, 0.0)
    tree = landscape.extract_nominal_merge_tree(
        prepared, name="fixture", wild_type_sequence="AAAAAAAAA"
    )
    code = tmp_path / "writer.py"
    code.write_text("# version one\n", encoding="utf-8")
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("lock version one\n", encoding="utf-8")

    def publish(
        *,
        parameters=None,
        selected=None,
        inputs=None,
        parents=None,
        host="local",
        run_id=None,
        code_file=code,
        output="output",
    ):
        path, _ = landscape.save_selected_products(
            selected or prepared,
            tree,
            parameters=parameters or {"thr": "reflected", "fdr": "by", "q": 0.05},
            scientific_parameters={"fixture": "toy"},
            context=RunContext(
                output_root=tmp_path / output,
                root=tmp_path,
                host=host,
                run_id=run_id,
            ),
            input_data_sha256s=inputs or _INPUTS,
            parents=parents or (_PARENT,),
            code_paths=(code_file,),
        )
        return read_manifest(path).scientific_id_sha256

    baseline = publish()
    assert publish(inputs={"fitness": "c" * 64}) != baseline
    assert publish(parameters={"thr": "reflected", "fdr": "by", "q": 0.01}) != baseline
    assert (
        publish(selected=landscape.build_selected_landscape(network, 0.1)) != baseline
    )
    assert (
        publish(
            selected=landscape.build_selected_landscape(
                network,
                0.0,
                functional_sequences={"AAAAAAAAA"},
                functional_classes={"AAAAAAAAA": "functional"},
            )
        )
        != baseline
    )
    changed_parent = {**_PARENT, "scientific_id_sha256": "c" * 64}
    assert publish(parents=(changed_parent,)) != baseline

    code.write_text("# version two\n", encoding="utf-8")
    lockfile.write_text("lock version two\n", encoding="utf-8")
    monkeypatch.setattr(product_module.platform, "platform", lambda: "other-platform")
    monkeypatch.setattr(product_module.sys, "version", "Python other-version")
    assert (
        publish(host="local", run_id="123", code_file=code, output="other") == baseline
    )


def test_five_variant_driver_writes_real_products(tmp_path: Path) -> None:
    network = _network()
    wild_type = "AAAAAAAAA"
    definitions = (
        ("Full", -1.0, {"graph": "full"}, None, None),
        ("Published", 0.8, {"thr": "published", "fdr": "none"}, None, None),
        ("Median-reflected", 0.2, {"thr": "reflected", "fdr": "none"}, None, None),
        (
            "Published BY",
            0.8,
            {"thr": "published", "fdr": "by", "q": 0.05},
            {"AAAAAAAAA", "AAAAAACAA"},
            {"AAAAAAAAA": "functional", "AAAAAAAAC": "non_functional"},
        ),
        (
            "Median-reflected BY",
            0.2,
            {"thr": "reflected", "fdr": "by", "q": 0.05},
            {"AAAAAAAAA", "AAAAAACAA"},
            {"AAAAAAAAA": "functional", "AAAAAAAAC": "uncertain"},
        ),
    )
    variants = {}
    for name, threshold, parameters, functional, classes in definitions:
        prepared = landscape.build_selected_landscape(
            network,
            threshold,
            functional_sequences=functional,
            functional_classes=classes,
        )
        tree = landscape.extract_nominal_merge_tree(
            prepared, name=name, wild_type_sequence=wild_type
        )
        variants[name] = (
            prepared,
            tree,
            parameters,
            {"fixture": "toy"},
            _INPUTS,
            (_PARENT,),
        )

    context = RunContext(output_root=tmp_path / "output", root=tmp_path)
    saved = landscape.save_landscape_variants(
        variants,
        context=context,
    )

    assert set(saved) == {name for name, *_ in definitions}
    for graph_path, tree_path in saved.values():
        graph_product = load_product("landscape", graph_path.name, context=context)
        tree_product = load_product("tree", tree_path.name, context=context)
        graph_manifest = read_manifest(graph_path)
        tree_manifest = read_manifest(tree_path)
        assert graph_product.label == graph_path.name
        assert tree_product.label == tree_path.name
        assert graph_manifest.artifacts
        assert tree_manifest.artifacts
        landscape_parent = load_product(
            "landscape", tree_manifest.parents[0]["label"], context=context
        )
        assert landscape_parent.manifest.scientific_id_sha256 == (
            graph_product.manifest.scientific_id_sha256
        )


def test_comparison_stage_declares_real_landscape_implementation(
    tmp_path: Path, monkeypatch
) -> None:
    graphml = tmp_path / "graph.graphml"
    peaks = tmp_path / "peaks.csv"
    graphml.write_text("graph", encoding="utf-8")
    peaks.write_text("peaks", encoding="utf-8")
    package_root = tmp_path / "src/dubanevics2026topological"
    analysis_dir = package_root / "analysis"
    for directory in (
        analysis_dir,
        package_root / "landscape",
        package_root / "threshold",
        package_root / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    actual_root = Path(clamped_cache.__file__).resolve().parents[1]
    for source_dir, target_dir, names in (
        (
            actual_root / "analysis",
            analysis_dir,
            (
                "clamped_cache.py",
                "describe.py",
                "graph_summary.py",
                "__init__.py",
            ),
        ),
        (
            actual_root / "landscape",
            package_root / "landscape",
            ("selected_landscape.py", "selection.py", "full_population.py"),
        ),
        (
            actual_root / "threshold",
            package_root / "threshold",
            ("functional_threshold.py",),
        ),
        (actual_root / "tables", package_root / "tables", ("__init__.py",)),
    ):
        for name in names:
            shutil.copyfile(source_dir / name, target_dir / name)
    monkeypatch.setattr(
        clamped_cache, "__file__", str(analysis_dir / "clamped_cache.py")
    )
    settings = {
        "analysis": {
            "dataset": {},
            "clamped": {
                "graphml": str(graphml),
                "peaks": str(peaks),
                "graphml_value_sha256": value_digest(graphml),
                "peaks_value_sha256": value_digest(peaks),
                "candidate_functional_threshold": {"rule": "median_reflected"},
            },
        }
    }
    monkeypatch.setattr(clamped_cache, "ROOT", tmp_path)
    monkeypatch.setattr(
        clamped_cache, "scientific_inputs", lambda _settings, *args, **kwargs: []
    )
    monkeypatch.setattr(clamped_cache, "scientific_code_paths", list)
    monkeypatch.setattr(clamped_cache, "require_files", lambda *_args: None)
    monkeypatch.setattr(
        clamped_cache,
        "candidate_selection_fingerprint",
        lambda _settings, **_kwargs: {},
    )
    monkeypatch.setattr(clamped_cache, "hash_sources", lambda paths: {})
    provenance = tmp_path / "analysis" / "provenance.json"
    provenance.parent.mkdir()
    provenance.write_text('{"fingerprint":"full"}', encoding="utf-8")
    monkeypatch.setattr(clamped_cache, "_output_root", lambda context: tmp_path)
    stage = clamped_cache.comparison_stage(settings)
    implementation = (
        Path(clamped_cache.__file__).resolve().parents[1]
        / "landscape"
        / "selected_landscape.py"
    ).resolve()
    forwarding_shim = Path(clamped_cache.__file__).with_name("clamped_landscapes.py")
    assert implementation in {Path(path).resolve() for path in stage.code}
    assert forwarding_shim.resolve() not in {
        Path(path).resolve() for path in stage.code
    }

    before = stage_fingerprint(stage)
    implementation.write_text(
        implementation.read_text(encoding="utf-8") + "\n# implementation edit\n",
        encoding="utf-8",
    )
    assert stage_fingerprint(stage) == before


class _CountingGraph:
    """Stand-in for a topographer graph: payloads indexed by node id."""

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = payloads

    def node_indices(self) -> range:
        return range(len(self._payloads))

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._payloads[index]


class _CountingNetwork:
    """Fake network whose ``graph`` property, like topographer's, is costly per read."""

    scalar_attr = "s"

    def __init__(self, size: int) -> None:
        self.graph_reads = 0
        self._graph = _CountingGraph(
            [{"s": float(i), "dna_sequence": f"S{i}"} for i in range(size)]
        )

    @property
    def graph(self) -> _CountingGraph:
        self.graph_reads += 1
        return self._graph


def test_node_payloads_reads_the_graph_property_once() -> None:
    from dubanevics2026topological.landscape.build_products import node_payloads

    network = _CountingNetwork(500)
    payloads = node_payloads(network)
    assert [p["dna_sequence"] for p in payloads] == [f"S{i}" for i in range(500)]
    assert network.graph_reads == 1


def test_landscape_summary_column_reads_the_graph_property_once() -> None:
    import pytest

    from dubanevics2026topological.analysis import graph_summary

    network = _CountingNetwork(300)
    item = type("Item", (), {"network": network})()
    # The identity check fails after the node read; only the access count matters.
    with pytest.raises(ValueError):
        graph_summary.landscape_summary_column(item, full_fitness={})
    assert network.graph_reads == 1
