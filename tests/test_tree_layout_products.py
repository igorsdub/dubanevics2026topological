"""The reviewed-layout stage fails closed on unverified parent references."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.build import CONTEXT_AWARE_COMMANDS, tree_layout_products


def test_layout_build_command_is_context_aware():
    assert "tree-layout-products" in CONTEXT_AWARE_COMMANDS


def test_vertex_identity_mapping_reads_only_one_graph_snapshot():
    class Graph:
        def node_indices(self):
            return [7, 2]

        def __getitem__(self, index):
            return {"dna_sequence": {7: "AAA", 2: "AAC"}[index]}

    class Network:
        reads = 0

        @property
        def graph(self):
            self.reads += 1
            return Graph()

    network = Network()
    assert tree_layout_products._vertex_identities(network) == {7: "AAA", 2: "AAC"}
    assert network.reads == 1


def test_progress_includes_timestamp_state_and_event(capsys):
    tree_layout_products._progress("published_by_band", "start")
    line = capsys.readouterr().out.strip()
    timestamp, message = line.split(" ", 1)
    assert datetime.fromisoformat(timestamp).utcoffset() is not None
    assert message == "tree-layout published_by_band: start"


def _leaf_and_tree(tree_id):
    leaf = SimpleNamespace(
        manifest=SimpleNamespace(
            parents=[{"product_type": "tree", "scientific_id_sha256": "expected"}]
        )
    )
    tree = SimpleNamespace(manifest=SimpleNamespace(scientific_id_sha256=tree_id))
    return leaf, tree


def test_leaf_rejects_wrong_tree_id(monkeypatch):
    leaf, tree = _leaf_and_tree("different")
    seen = []

    def fake_load(kind, state, context):
        seen.append((kind, state))
        return leaf if kind == "leaves" else tree

    monkeypatch.setattr(tree_layout_products.processed, "load", fake_load)
    with pytest.raises(ValueError, match="mismatched tree"):
        tree_layout_products._tree_for_leaf("full", object())
    assert ("tree", "full") in seen and ("leaves", "full") in seen


def test_tree_comes_from_processed_and_matches_its_leaves(monkeypatch):
    leaf, tree = _leaf_and_tree("expected")
    monkeypatch.setattr(
        tree_layout_products.processed,
        "load",
        lambda kind, state, context: leaf if kind == "leaves" else tree,
    )
    assert tree_layout_products._tree_for_leaf("published_by_band", object()) is tree


def test_leaves_must_name_exactly_one_tree(monkeypatch):
    leaf = SimpleNamespace(manifest=SimpleNamespace(parents=[]))
    monkeypatch.setattr(
        tree_layout_products.processed, "load", lambda *a, **k: leaf
    )
    with pytest.raises(ValueError, match="exactly one tree"):
        tree_layout_products._tree_for_leaf("full", object())


def test_driver_reconstructs_all_states_without_loading_cached_trees(
    tmp_path, monkeypatch
):
    from dubanevics2026topological import analysis
    from dubanevics2026topological.analysis import clamped_cache
    from dubanevics2026topological.results import RunContext

    def forbidden(*args, **kwargs):
        pytest.fail("layout driver loaded a cached analysis tree")

    monkeypatch.setattr(analysis, "load_or_compute_analysis", forbidden)
    monkeypatch.setattr(analysis, "load_cached_analysis", forbidden)
    monkeypatch.setattr(clamped_cache, "load_cached_clamped_comparison", forbidden)
    monkeypatch.setattr(
        tree_layout_products, "load_or_compute_analysis", forbidden, raising=False
    )
    monkeypatch.setattr(
        tree_layout_products, "load_cached_clamped_comparison", forbidden, raising=False
    )
    context = RunContext(root=tmp_path, env={})
    checkpoint = context.output_root / "analysis/clamped_comparison/provenance.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    variant_dir = tmp_path / "config/variants"
    variant_dir.mkdir(parents=True)
    for variant, _ in tree_layout_products.BY.values():
        (variant_dir / variant).write_text(
            '[clamped.candidate_functional_threshold]\n'
            'rule = "fdr_fixed_cutoff"\ndecimals = 6\ncutoff = "median_reflected"\n'
            'method = "by"\nq = 0.05\ninput_dir = "reference"\n'
        )
    wt = pl.DataFrame({"dna_from": ["A"]})
    monkeypatch.setattr(tree_layout_products.pl, "read_parquet", lambda _: wt)
    monkeypatch.setattr(
        tree_layout_products,
        "load_settings",
        lambda **_: {
            "analysis": {
                "dataset": {"fitness_wt": "wt", "fitness_diff": "diff"},
                "clamped": {"published_functional_threshold": 0.0},
            }
        },
    )

    class Graph:
        def nodes(self):
            return [{"s": 1.0, "dna_sequence": "A"}]

        def node_indices(self):
            return [0]

        def __getitem__(self, index):
            return self.nodes()[index]

    network = SimpleNamespace(graph=Graph(), scalar_attr="s")
    monkeypatch.setattr(
        tree_layout_products,
        "Papkou2023Workflow",
        lambda **_: SimpleNamespace(build_network=lambda: network),
    )
    monkeypatch.setattr(
        tree_layout_products,
        "compute_candidate_functional_threshold",
        lambda *a: SimpleNamespace(threshold=0.0),
    )
    parent = SimpleNamespace(
        manifest=SimpleNamespace(
            scientific_identity={
                "landscape_selection": {"functional_set_sha256": "hash"}
            },
            parents=[
                {"product_type": "direct_band_identity", "scientific_id_sha256": "band"}
            ],
        )
    )
    monkeypatch.setattr(tree_layout_products, "_tree_for_leaf", lambda *a: parent)
    snapshot = SimpleNamespace(
        summary={"threshold": {"rounded": 0.0}},
        table=pl.DataFrame({"dna_sequence": ["A"], "fdr_class": ["functional"]}),
    )
    monkeypatch.setattr(
        tree_layout_products, "load_fdr_snapshot", lambda *a, **k: snapshot
    )
    monkeypatch.setattr(
        tree_layout_products, "require_full_population", lambda *a: None
    )
    prepared = SimpleNamespace(
        network=network, diagnostics={"functional_set_sha256": "hash"}
    )
    monkeypatch.setattr(
        tree_layout_products, "build_selected_landscape", lambda *a, **k: prepared
    )
    names = []

    def extract(item, *, name, wild_type_sequence):
        names.append(name)
        return SimpleNamespace(trees={"split": object()})

    monkeypatch.setattr(tree_layout_products, "extract_nominal_merge_tree", extract)
    monkeypatch.setattr(tree_layout_products, "_require_saved_pairs", lambda *a: None)
    monkeypatch.setattr(
        tree_layout_products, "build_tree_layout", lambda *a, **k: Path("layout")
    )
    monkeypatch.setattr(
        tree_layout_products,
        "build_tree_annotations",
        lambda *a, **k: Path("annotations"),
    )
    summary = tmp_path / "summary.json"
    summary.write_text("{}")
    draws = SimpleNamespace(
        report=SimpleNamespace(status="full", byte_integrity=True),
        manifest=SimpleNamespace(
            scientific_identity={
                "functional_set_sha256": "hash",
                "thr": "published",
                "fdr": "by",
                "q": 0.05,
                "draws": 10000,
                "seed": 20261019,
            }
        ),
        artifacts={
            "checkpoint/summary.json": summary,
            "checkpoint/hills.parquet": tmp_path / "hills",
        },
    )
    def load_draws(kind, state, context):
        assert kind == "bootstrap"
        draws.manifest.scientific_identity["thr"] = state.split("_")[0]
        return draws

    monkeypatch.setattr(tree_layout_products.processed, "load", load_draws)
    monkeypatch.setattr(
        tree_layout_products,
        "direct_band_parent",
        lambda *a: SimpleNamespace(scientific_id_sha256="band"),
    )
    monkeypatch.setattr(tree_layout_products, "check_hills_match_tree", lambda *a: None)
    tree_layout_products.build_tree_layouts(context)
    assert names == ["full", "published", "reflected", "published_by", "reflected_by"]


def test_cached_tree_rejects_different_saved_persistence(tmp_path, monkeypatch):
    path = tmp_path / "pairs.parquet"
    pl.DataFrame({"birth_label": ["A"], "persistence": [1.0]}).write_parquet(path)
    parent = SimpleNamespace(artifacts={"persistence_pairs.parquet": path})
    source = SimpleNamespace(persistence_table=object())
    monkeypatch.setattr(
        tree_layout_products,
        "_public_table",
        lambda *args: pl.DataFrame({"birth_label": ["A"], "persistence": [1.1]}),
    )
    with pytest.raises(ValueError, match="persistence pairs differ"):
        tree_layout_products._require_saved_pairs("full", source, object(), parent)
