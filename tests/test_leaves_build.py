"""The leaves stage publishes seven leaves products and the two band trees from processed inputs."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.build import CONTEXT_AWARE_COMMANDS, leaves_product


def test_the_command_is_registered_and_context_aware():
    assert "leaves-product" in CONTEXT_AWARE_COMMANDS


def test_nominal_states_are_the_five_graphs_and_their_populations():
    assert leaves_product.NOMINAL_STATES == (
        "full",
        "published",
        "reflected",
        "published_by",
        "reflected_by",
    )
    assert leaves_product.POPULATIONS["reflected_by"] == "Median_reflected_BY"


def _draws(**identity):
    base = {
        "thr": "reflected",
        "fdr": "by",
        "q": 0.05,
        "draws": 10000,
        "seed": 20261019,
        "functional_set_sha256": "hash",
    }
    return SimpleNamespace(manifest=SimpleNamespace(scientific_identity={**base, **identity}))


def test_band_draws_must_be_the_states_bootstrap(monkeypatch):
    prepared = SimpleNamespace(diagnostics={"functional_set_sha256": "hash"})
    monkeypatch.setattr(leaves_product.processed, "load", lambda *a, **k: _draws(thr="published"))
    with pytest.raises(ValueError, match="not the"):
        leaves_product._band_draws("reflected_by", prepared, object())


def test_band_draws_must_carry_the_rebuilt_selection(monkeypatch):
    prepared = SimpleNamespace(diagnostics={"functional_set_sha256": "other"})
    monkeypatch.setattr(leaves_product.processed, "load", lambda *a, **k: _draws())
    with pytest.raises(ValueError, match="selected graph differs"):
        leaves_product._band_draws("reflected_by", prepared, object())


def test_assignments_must_equal_the_graph():
    view = SimpleNamespace(assignments=pl.DataFrame({"dna_sequence": ["a", "b"]}))
    leaves_product._check_assignments("full", view, {"a", "b"})
    with pytest.raises(ValueError, match="differ from the graph"):
        leaves_product._check_assignments("full", view, {"a"})
    duplicate = SimpleNamespace(assignments=pl.DataFrame({"dna_sequence": ["a", "a"]}))
    with pytest.raises(ValueError, match="differ from the graph"):
        leaves_product._check_assignments("full", duplicate, {"a"})


def test_driver_publishes_nominal_leaves_for_five_states_and_band_for_two(tmp_path, monkeypatch):
    from dubanevics2026topological.results import RunContext

    context = RunContext(root=tmp_path, env={})
    wt = pl.DataFrame({"dna_from": ["A"]})
    monkeypatch.setattr(leaves_product.pl, "read_parquet", lambda *_: wt)
    monkeypatch.setattr(
        leaves_product,
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
        leaves_product,
        "Papkou2023Workflow",
        lambda **_: SimpleNamespace(build_network=lambda: network),
    )
    monkeypatch.setattr(
        leaves_product,
        "compute_candidate_functional_threshold",
        lambda *a: SimpleNamespace(threshold=0.0),
    )
    variants = tmp_path / "config/variants"
    variants.mkdir(parents=True)
    for variant, _ in leaves_product.BY.values():
        (variants / variant).write_text(
            '[clamped.candidate_functional_threshold]\n'
            'rule = "fdr_fixed_cutoff"\ndecimals = 6\ncutoff = "median_reflected"\n'
            'method = "by"\nq = 0.05\ninput_dir = "reference"\n'
        )
    snapshot = SimpleNamespace(
        summary={"threshold": {"rounded": 0.0}},
        table=pl.DataFrame({"dna_sequence": ["A"], "fdr_class": ["functional"]}),
    )
    monkeypatch.setattr(leaves_product, "load_fdr_snapshot", lambda *a, **k: snapshot)
    monkeypatch.setattr(leaves_product, "require_full_population", lambda *a: None)
    prepared = SimpleNamespace(network=network, diagnostics={"functional_set_sha256": "hash"})
    monkeypatch.setattr(leaves_product, "build_selected_landscape", lambda *a, **k: prepared)
    monkeypatch.setattr(
        leaves_product,
        "extract_nominal_merge_tree",
        lambda item, *, name, wild_type_sequence: SimpleNamespace(trees={"split": object()}),
    )
    view = SimpleNamespace(assignments=pl.DataFrame({"dna_sequence": ["A"]}))
    monkeypatch.setattr(leaves_product, "absorb_terminal_branches", lambda *a: view)
    monkeypatch.setattr(leaves_product, "build_band_simplified_view", lambda *a: view)
    nominal = SimpleNamespace(manifest=SimpleNamespace(parents=[]))
    loaded = []
    saved = []
    monkeypatch.setattr(
        leaves_product,
        "_save_leaves",
        lambda state, *a, **k: saved.append((state, k["tree_state"])) or Path(state),
    )
    summary = tmp_path / "summary.json"
    summary.write_text("{}")
    monkeypatch.setattr(
        leaves_product.processed,
        "load",
        lambda kind, state, context: (
            loaded.append((kind, state))
            or (
                nominal
                if kind == "tree"
                else SimpleNamespace(
                    manifest=_draws(thr=state.split("_")[0]).manifest,
                    artifacts={
                        "checkpoint/summary.json": summary,
                        "checkpoint/hills.parquet": tmp_path / "hills",
                    },
                )
            )
        ),
    )
    monkeypatch.setattr(
        leaves_product.pl, "read_parquet", lambda path: wt if "wt" == path else pl.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(
        leaves_product,
        "direct_band_parent",
        lambda *a: SimpleNamespace(scientific_id_sha256="band"),
    )
    band_tree = SimpleNamespace(
        manifest=SimpleNamespace(
            parents=[{"product_type": "direct_band_identity", "scientific_id_sha256": "band"}]
        )
    )
    monkeypatch.setattr(leaves_product, "save_band_tree", lambda *a, **k: Path("band-tree"))
    monkeypatch.setattr(leaves_product, "load_product", lambda *a, **k: band_tree)
    # The wild type comes from the (stubbed) fitness table; the leaves driver reads it twice.
    monkeypatch.setattr(leaves_product, "_configured_input_path", lambda context, value: value)
    outputs = leaves_product.build_leaves_products(context)
    assert [state for state, _ in saved] == [
        "full",
        "published",
        "reflected",
        "published_by",
        "published_by_band",
        "reflected_by",
        "reflected_by_band",
    ]
    assert dict(saved)["published_by_band"] == "band"
    assert dict(saved)["full"] == "nominal"
    assert ("bootstrap", "reflected_by") in loaded
    assert set(outputs) == {
        "full",
        "published",
        "reflected",
        "published_by",
        "published_by_band_tree",
        "published_by_band",
        "reflected_by",
        "reflected_by_band_tree",
        "reflected_by_band",
    }
