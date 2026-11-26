from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import rustworkx as rx
from topographer.graph import Graph, ScalarNetwork
from dubanevics2026topological.simplification.cooccurrence_leaf import (
    BootstrapResult,
    LeafCooccurrence,
)

from dubanevics2026topological.analysis import git_stamp as git_stamp_module
from dubanevics2026topological.analysis.produce_or_load import (
    PROVENANCE_NAME,
    Checkpoint,
    stage_fingerprint,
)
from dubanevics2026topological.analysis.sequence_growth_rates import (
    SequenceGrowthRatesResult,
)
from dubanevics2026topological.landscape.selected_landscape import (
    analyze_prepared_landscape,
    prepare_clamped_landscape,
)
from dubanevics2026topological.simplification import bootstrap_chunks, checkpoint_identity
from dubanevics2026topological.simplification import bootstrap_smoothed_tree as module
from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
    build_band_simplified_view,
    build_bootstrap_smoothed_tree,
    check_hills_match_tree,
    check_identical_tree,
    classify_hills,
    compute_bootstrap_smoothed_tree,
    decide_hills,
    load_cached_bootstrap_smoothed_tree,
    nominal_hills,
)

CONFIG = {
    "n_replicates": 20,
    "seed": 7,
    "band_quantile": 0.95,
    "n_jobs": 1,
}
ACTIVE_CONFIG = CONFIG


@pytest.fixture(autouse=True)
def _without_n_jobs_override(monkeypatch):
    """The worker-count override of the environment must not reach these tests."""
    monkeypatch.delenv(module.ENV_N_JOBS, raising=False)


# Path a-b-c-d-e-f-g with leaves b (0.5), d (0.4), f (0.6). Saddle e joins d and
# f, the root c joins b and the e arm. Nominal hills: b = {b, a}, d = {d},
# f = {f, g}; e and c are on no hill. Network node index = position in DNA.
DNA = list("abcdefg")
RAW_FITNESS = [-0.1, 0.5, -0.2, 0.4, -0.15, 0.6, 0.1]
WILD_TYPE_R = 0.3
# Nominal persistences of the toy hills: f 0.8 (global, closed at the root c),
# b 0.7 and d 0.55. 2 * BAND_C = 0.72 lets only f clear the band.
BAND_C = 0.36
# The one real-pipeline fact the tests rely on: terminal edges.
TOY_HILLS = {"b": ["b", "a"], "d": ["d"], "f": ["f", "g"]}


def _toy(fitness: list[float] = RAW_FITNESS) -> tuple[ScalarNetwork, SimpleNamespace]:
    """Return the prepared network and the nominal split tree of the toy path."""
    vertices = pl.DataFrame(
        {
            "dna_sequence": DNA,
            "fitness": fitness,
            "aa_sequence": ["A"] * 7,
            "has_stop_codon": [False] * 7,
        }
    )
    edges = pl.DataFrame({"source": list("abcdef"), "target": list("bcdefg")})
    graph = Graph.from_tables(
        vertices,
        edges,
        vertex_id_col="dna_sequence",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("fitness", "aa_sequence", "has_stop_codon"),
    )
    prepared = prepare_clamped_landscape(
        ScalarNetwork(graph, scalar_attr="fitness"), -1.0
    )
    analysis = analyze_prepared_landscape(
        prepared, name="synthetic", wild_type_sequence="a"
    )
    return prepared.network, analysis.trees["split"]


def _growth(
    fitness: list[float] = RAW_FITNESS, se: float | list[float | None] = 0.05
) -> SequenceGrowthRatesResult:
    ses = se if isinstance(se, list) else [se] * len(DNA)
    # ``se_replicate`` is a decoy: the bootstrap must read ``se_pooled`` only.
    decoy = [10.0] * len(DNA)
    table = pl.DataFrame(
        {
            "dna_sequence": DNA,
            "sequence_id": [f"id_{name}" for name in DNA],
            "r": [value + WILD_TYPE_R for value in fitness],
            "s": list(fitness),
            "se_pooled": ses,
            "r_replicate_mean": [value + WILD_TYPE_R for value in fitness],
            "r_replicate_sd": [0.1] * len(DNA),
            "se_replicate": decoy,
            "n_replicates": [2] * len(DNA),
            "status": ["converged"] * len(DNA),
            "stop_codon_count": [0] * len(DNA),
        },
        schema_overrides={"se_pooled": pl.Float64},
    )
    return SequenceGrowthRatesResult(
        growth_rates=table,
        diagnostics={"identity_tolerance": 1e-6, "wild_type_r": WILD_TYPE_R},
        provenance={"fingerprint": {"generation": 1, "files": {}}, "artifacts": {}},
    )


def _checkpoint_fingerprint(generation: int = 1) -> dict:
    """A minimal real-shaped fingerprint for checkpoint writer tests."""
    return {
        "generation": generation,
        "schema": 7,
        "config": {
            "n_replicates": 20,
            "seed": 7,
            "band_quantile": 0.95,
            "noise_model": "pooled",
        },
        "candidate_functional_threshold": -1.0,
        "candidate_selection": {"selected_sequence_set_sha256": "a" * 64},
        "files": {"data/raw/fitness.parquet": "b" * 64},
        "growth_rates_checkpoint": {},
    }


def _network(se: float = 0.05, fitness: list[float] = RAW_FITNESS) -> ScalarNetwork:
    prepared, _ = _toy(fitness)
    ids, edges = module._prepared_tables(prepared)
    return module._make_network(ids, edges, np.array(fitness), np.full(7, se))


def _cooc(
    cores: list[list[str]],
    *,
    support: dict[str, float] | None = None,
    peak_frequency: dict[str, float] | None = None,
    wasserstein_1: list[float] | None = None,
    bottleneck: list[float] | None = None,
    n: int = 20,
    **overrides,
) -> BootstrapResult:
    """Return a hand-built result: each core is pairwise at frequency 1.0.

    The default bottleneck distance gives ``c = BAND_C``: only the global
    maximum f (nominal persistence 0.8) clears ``2c = 0.72``, so the band view
    is nested in the core-rule view whatever the cores are (the elder rule
    always keeps the global maximum).
    """
    members = [name for core in cores for name in core]
    pairs = sorted(
        {
            tuple(sorted((DNA.index(a), DNA.index(b))))
            for core in cores
            for a in core
            for b in core
        }
    )
    pair_frequency = pl.DataFrame(
        {
            "vertex_a": [a for a, _ in pairs],
            "vertex_b": [b for _, b in pairs],
            "frequency": [1.0] * len(pairs),
        },
        schema={
            "vertex_a": pl.Int64,
            "vertex_b": pl.Int64,
            "frequency": pl.Float64,
        },
    )
    peaks = peak_frequency or {}
    terminal = [1.0 if name in members else 0.0 for name in DNA]
    zeros = [0.0] * 7
    vertex_roles = pl.DataFrame(
        {
            "vertex_index": list(range(7)),
            "leaf_frequency": [peaks.get(name, 0.0) for name in DNA],
            "saddle_frequency": zeros,
            "regular_frequency": zeros,
            "root_frequency": zeros,
            "terminal_frequency": terminal,
        }
    )
    w1 = wasserstein_1 if wasserstein_1 is not None else [0.0] * n
    values = {
        "core_id": [i for i, core in enumerate(cores) for _ in core],
        "vertex_index": [DNA.index(name) for core in cores for name in core],
    }
    if "n_replicates" in overrides:
        overrides["n_draws"] = overrides.pop("n_replicates")
    fields = {
        "same_leaf_frequency": pair_frequency,
        "vertex_roles": vertex_roles,
        "leaf_cores": pl.DataFrame(
            values, schema={"core_id": pl.Int64, "vertex_index": pl.Int64}
        ),
        "draw_stats": pl.DataFrame(
            {
                "draw": list(range(n)),
                "leaf_count": [3] * n,
                "wasserstein_1": w1,
                "bottleneck": bottleneck if bottleneck is not None else [BAND_C] * n,
                "reproduced_pair_fraction": [1.0] * n,
            }
        ),
        "nominal_hill_support": pl.DataFrame(
            {
                "hill_id": list(TOY_HILLS),
                "hill_support": [(support or {}).get(h, 1.0) for h in TOY_HILLS],
            }
        ),
        "n_draws": n,
        "seed": CONFIG["seed"],
        "tree_type": "split",
        "core_cutoff": 0.95,
    }
    return BootstrapResult(**{**fields, **overrides})


NOMINAL_CORES = [["b", "a"], ["d"], ["f", "g"]]


def _echo(self, name, network, statistic, *, core_threshold=0.95, label=None):
    """A fake chunk runner that echoes the requested settings, like Topographer does."""
    return _cooc(
        NOMINAL_CORES,
        n=self.store.n_draws,
        n_replicates=self.store.n_draws,
        seed=self.store.seed,
        core_cutoff=core_threshold,
    )


def _patch_seam(monkeypatch, real=None, flat=None):
    """Replace the chunk runner; return the list of (network, call) records."""
    calls = []
    answers = [real or _cooc(NOMINAL_CORES), flat or _cooc(NOMINAL_CORES)]

    def fake(self, name, network, statistic, *, core_threshold=None, label=None):
        calls.append(
            (
                network,
                {
                    "name": name,
                    "statistic": statistic,
                    "core_threshold": core_threshold,
                    "runner": self,
                },
            )
        )
        return answers[len(calls) - 1]

    monkeypatch.setattr(bootstrap_chunks.ChunkedBootstrap, "run", fake)
    return calls


def _build(monkeypatch, real=None, flat=None, *, se=0.05, config=CONFIG, **kwargs):
    calls = _patch_seam(monkeypatch, real, flat)
    prepared, tree = _toy()
    if config is CONFIG:
        config = ACTIVE_CONFIG
    result = build_bootstrap_smoothed_tree(
        prepared, tree, _growth(se=se), config, log=lambda message: None, **kwargs
    )
    return result, calls


def _peaks(view) -> set[str]:
    graph = view.tree
    return {
        graph[i]["dna_sequence"]
        for i in graph.node_indices()
        if graph.in_degree(i) == 0
    }


# -- a hand-built tree for the decision table ---------------------------------
# elder (5.0) and mid (4.0) join at saddle s1; other (4.0, tied with mid) joins
# at lower. Regular vertices: e1 on elder, m1 and m2 on mid; x1 sits on the
# interior branch s1|lower (on no hill); s1, lower and root are critical.
HAND_IDS = ["elder", "mid", "other", "s1", "lower", "root", "e1", "m1", "m2", "x1"]


def _hand_tree(swap_tied_indices: bool = False) -> SimpleNamespace:
    graph = rx.PyDiGraph()
    rows = [
        ("elder", 5.0, "maximum"),
        ("mid", 4.0, "maximum"),
        ("other", 4.0, "maximum"),
        ("s1", 2.0, "split_saddle"),
        ("lower", 1.0, "split_saddle"),
        ("root", 0.0, "minimum"),
    ]
    for i, (dna, scalar, event) in enumerate(rows):
        graph.add_node(
            {
                "vertex_index": i,
                "dna_sequence": dna,
                "scalar": scalar,
                "event": event,
            }
        )
    if swap_tied_indices:  # make vertex_index disagree with name order
        graph[1]["vertex_index"], graph[2]["vertex_index"] = 2, 1
    graph.add_edges_from_no_data([(0, 3), (1, 3), (3, 4), (2, 4), (4, 5)])
    members = pl.DataFrame(
        {
            "source_tree_node_index": [0, 1, 1, 3],
            "target_tree_node_index": [3, 3, 3, 4],
            "dna_sequence": ["e1", "m1", "m2", "x1"],
        }
    )
    return SimpleNamespace(
        tree=graph, branch_member_table=members, wild_type_marker_table=pl.DataFrame()
    )


def _decide(cores, tree=None):
    tree = tree or _hand_tree()
    hills = nominal_hills(tree)
    table = decide_hills(
        tree, hills, {i: frozenset(core) for i, core in enumerate(cores)}
    )
    return {row["peak_sequence"]: row for row in table.iter_rows(named=True)}


# -- nominal hills ------------------------------------------------------------


def test_nominal_hills_are_the_terminal_branches_by_identity():
    hills = nominal_hills(_hand_tree())
    assert [h.peak_sequence for h in hills] == ["elder", "mid", "other"]
    assert {h.peak_sequence: h.members for h in hills} == {
        "elder": ("elder", "e1"),
        "mid": ("mid", "m1", "m2"),
        "other": ("other",),
    }
    assert [h.peak_raw_fitness for h in hills] == [5.0, 4.0, 4.0]
    assert [h.peak_vertex_index for h in hills] == [0, 1, 2]
    # Interior vertices are on no hill.
    on_hill = {d for h in hills for d in h.members}
    assert on_hill.isdisjoint({"s1", "lower", "root", "x1"})
    assert module.hill_vertex_indices(hills, HAND_IDS) == {
        "elder": [0, 6],
        "mid": [1, 7, 8],
        "other": [2],
    }


def test_nominal_hills_of_the_real_toy_tree_match_its_terminal_edges():
    _, tree = _toy()
    hills = nominal_hills(tree)
    assert {h.peak_sequence: list(h.members) for h in hills} == TOY_HILLS
    assert module.hill_vertex_indices(hills, DNA) == {
        "b": [1, 0],
        "d": [3],
        "f": [5, 6],
    }


# -- the decision table -------------------------------------------------------


def test_a_core_extending_onto_off_hill_vertices_is_inside_and_counted():
    tree = _hand_tree()
    hills = nominal_hills(tree)
    cores = {
        0: frozenset({"mid", "x1", "s1"}),  # meets only mid, plus off-hill vertices
        1: frozenset({"elder", "e1"}),  # entirely inside elder
        2: frozenset({"other"}),
    }
    table = decide_hills(tree, hills, cores)
    rows = {r["peak_sequence"]: r for r in table.iter_rows(named=True)}
    assert rows["mid"]["core_relation"] == "inside"
    assert rows["mid"]["outcome"] == "kept"


def test_a_spanning_core_keeps_the_highest_peak_and_nominates_the_rest():
    hills = _decide([["elder", "mid"], ["other"]])
    assert (
        hills["elder"]["core_relation"] == hills["mid"]["core_relation"] == "spanning"
    )
    assert hills["elder"]["outcome"] == "kept"
    assert hills["mid"]["outcome"] == "absorbed"
    assert hills["elder"]["surviving_peak_sequence"] == "elder"
    assert hills["mid"]["surviving_peak_sequence"] == "elder"
    assert hills["other"]["core_relation"] == "inside"
    assert hills["mid"]["core_ids"] == [0]


def test_spanning_ties_go_to_the_smaller_vertex_index():
    # mid (vertex_index 1) and other (2) tie at 4.0: mid wins by index and name.
    hills = _decide([["mid", "other"], ["elder"]])
    assert hills["mid"]["outcome"] == "kept"
    assert hills["other"]["outcome"] == "absorbed"
    assert hills["other"]["surviving_peak_sequence"] == "mid"
    # Swap the payload indices: now other (1) wins although "mid" < "other".
    swapped = _decide([["mid", "other"], ["elder"]], _hand_tree(True))
    assert swapped["other"]["outcome"] == "kept"
    assert swapped["mid"]["outcome"] == "absorbed"
    assert swapped["mid"]["surviving_peak_sequence"] == "other"


def test_a_hill_losing_one_spanning_core_is_nominated_with_the_best_winner():
    # mid wins the m1/other core but loses the elder/mid core (cores are disjoint).
    hills = _decide([["elder", "mid"], ["m1", "other"]])
    assert hills["mid"]["core_ids"] == [0, 1]
    assert hills["mid"]["core_relation"] == "spanning"
    assert hills["mid"]["outcome"] == "absorbed"
    assert hills["mid"]["surviving_peak_sequence"] == "elder"
    assert hills["other"]["outcome"] == "absorbed"
    assert hills["other"]["surviving_peak_sequence"] == "mid"


def test_overlapping_cores_are_refused():
    tree = _hand_tree()
    overlapping = {0: frozenset({"elder", "mid"}), 1: frozenset({"mid", "other"})}
    for decide in (classify_hills, lambda h, c: decide_hills(tree, h, c)):
        with pytest.raises(ValueError, match="disjoint.*'mid'"):
            decide(nominal_hills(tree), overlapping)


def test_two_cores_in_one_hill_make_a_split_candidate_that_is_kept():
    hills = _decide([["elder"], ["e1"], ["mid"], ["other"]])
    assert hills["elder"]["core_relation"] == "split_candidate"
    assert hills["elder"]["split_candidate"] is True
    assert hills["elder"]["core_ids"] == [0, 1]
    assert hills["elder"]["outcome"] == "kept"
    assert hills["mid"]["split_candidate"] is False


def test_a_hill_both_spanning_and_split_records_both_facts():
    hills = _decide([["elder", "mid"], ["e1"], ["other"]])
    elder = hills["elder"]
    assert elder["core_relation"] == "spanning"  # the spanning rule decides
    assert elder["split_candidate"] is True  # and the split is still recorded
    assert elder["core_ids"] == [0, 1]
    assert elder["outcome"] == "kept"
    assert hills["mid"]["split_candidate"] is False
    assert hills["mid"]["outcome"] == "absorbed"


def test_an_orphan_core_acts_on_nothing():
    tree = _hand_tree()
    hills = nominal_hills(tree)
    cores = {
        0: frozenset({"s1", "x1"}),  # on no hill: orphan
        1: frozenset({"elder", "e1"}),
        2: frozenset({"mid"}),
        3: frozenset({"other"}),
    }
    assert (
        decide_hills(tree, hills, cores)["outcome"].to_list() == ["kept"] * 3
    )  # orphan cores do not change any outcome


def test_classify_hills_is_pure_and_reports_nomination_without_the_elder_rule():
    hills = nominal_hills(_hand_tree())
    table = classify_hills(hills, {})
    assert table["nominated"].to_list() == [True] * 3  # elder is not yet protected
    assert table["core_ids"].to_list() == [[], [], []]
    assert table.columns == [
        "peak_sequence",
        "core_ids",
        "core_relation",
        "split_candidate",
        "surviving_peak_sequence",
        "nominated",
    ]


def test_the_elder_rule_protects_a_nominated_elder_arm():
    hills = _decide([])
    # elder is the global maximum, and the elder arm at every saddle on its path.
    assert hills["elder"]["outcome"] == "elder_protected"
    assert hills["mid"]["outcome"] == hills["other"]["outcome"] == "absorbed"


# -- network, identical tree --------------------------------------------------


def test_growth_rows_outside_the_candidate_network_are_ignored():
    """The full-set table holds sequences that are not Candidate vertices."""
    prepared, _ = _toy()
    ids, _edges = module._prepared_tables(prepared)
    raw = np.array(RAW_FITNESS)
    growth = _growth(se=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    extra = growth.growth_rates.head(2).with_columns(
        pl.Series("dna_sequence", ["x", "y"]),
        pl.Series("sequence_id", ["id_x", "id_y"]),
        pl.lit(9.0).alias("se_pooled"),
        pl.lit(123.0).alias("r"),
    )
    bigger = SequenceGrowthRatesResult(
        growth_rates=pl.concat([growth.growth_rates, extra]),
        diagnostics=growth.diagnostics,
        provenance=growth.provenance,
    )
    assert bigger.growth_rates.height == len(DNA) + 2
    expected = module._pooled_se(ids, growth, raw)
    assert np.array_equal(module._pooled_se(ids, bigger, raw), expected)


def test_network_carries_raw_fitness_and_pooled_se():
    prepared, tree = _toy()
    growth = _growth(se=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ids, edges = module._prepared_tables(prepared)
    raw = np.array(RAW_FITNESS)
    se = module._pooled_se(ids, growth, raw)
    network = module._make_network(ids, edges, raw, se)

    assert [str(v) for v in network.vertex_ids] == DNA
    assert network.scalar_attr == "raw_fitness"
    payload = {
        network.graph[i]["dna_sequence"]: network.graph[i]
        for i in network.graph.node_indices()
    }
    assert payload["d"]["raw_fitness"] == 0.4
    assert payload["d"]["se"] == pytest.approx(0.4)  # se_pooled, not the 10.0 decoy
    assert [payload[name]["se"] for name in DNA] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    assert network.edge_count == 6
    check_identical_tree(network, tree)


def test_identical_tree_check_rejects_a_different_landscape():
    _, tree = _toy()
    other = list(RAW_FITNESS)
    other[3], other[5] = 0.7, 0.35  # d now above f: another tree
    with pytest.raises(ValueError, match="does not give the nominal"):
        check_identical_tree(_network(fitness=other), tree)


def test_identical_tree_check_rejects_a_changed_height():
    _, tree = _toy()
    other = list(RAW_FITNESS)
    other[1] = 0.55  # same shape, one height differs
    with pytest.raises(ValueError, match="height or event differ at 'b'"):
        check_identical_tree(_network(fitness=other), tree)


def test_hills_must_match_the_nominal_tree(monkeypatch):
    _, tree = _toy()
    result, _ = _build(monkeypatch)
    check_hills_match_tree(result.hills, tree)
    with pytest.raises(ValueError, match="differ from the nominal"):
        check_hills_match_tree(
            result.hills.filter(pl.col("peak_sequence") != "b"), tree
        )
    with pytest.raises(ValueError, match="member count.*'d'"):
        check_hills_match_tree(
            result.hills.with_columns(
                pl.when(pl.col("peak_sequence") == "d")
                .then(5)
                .otherwise(pl.col("member_count"))
                .alias("member_count")
            ),
            tree,
        )


# -- growth-rate input checks -------------------------------------------------


def test_missing_sequence_in_growth_rates_fails_closed():
    prepared, _ = _toy()
    ids, _ = module._prepared_tables(prepared)
    growth = _growth()
    growth.growth_rates = growth.growth_rates.filter(pl.col("dna_sequence") != "f")
    with pytest.raises(ValueError, match="absent from the growth-rate checkpoint.*'f'"):
        module._pooled_se(ids, growth, np.array(RAW_FITNESS))


@pytest.mark.parametrize("bad", [None, float("nan"), 0.0, -0.1])
def test_bad_pooled_se_fails_closed_naming_the_sequence(bad):
    prepared, _ = _toy()
    ids, _ = module._prepared_tables(prepared)
    growth = _growth(se=[0.1, 0.1, 0.1, bad, 0.1, 0.1, 0.1])
    with pytest.raises(
        ValueError,
        match="Pooled standard errors are missing, non-finite or not positive.*'d'",
    ):
        module._pooled_se(ids, growth, np.array(RAW_FITNESS))


def test_growth_rates_disagreeing_with_the_network_fail_closed():
    prepared, _ = _toy()
    ids, _ = module._prepared_tables(prepared)
    shifted = list(RAW_FITNESS)
    shifted[2] += 0.01
    with pytest.raises(ValueError, match="disagrees with the Candidate network.*'c'"):
        module._pooled_se(ids, _growth(shifted), np.array(RAW_FITNESS))


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"n_replicates": 0}, ValueError),
        ({"n_replicates": 1.5}, TypeError),
        ({"n_jobs": 0}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": True}, TypeError),
        ({"core_threshold": 0}, ValueError),
        ({"core_threshold": 1.1}, ValueError),
        ({"core_threshold": float("nan")}, ValueError),
        ({"core_threshold": True}, ValueError),
        ({"consensus_peak_frequency": 0}, ValueError),
        ({"consensus_peak_frequency": 1.5}, ValueError),
        ({"consensus_peak_frequency": True}, ValueError),
        ({"band_quantile": 0}, ValueError),
        ({"chunk_size": 0}, ValueError),
        ({"chunk_size": 2.5}, TypeError),
        ({"chunk_size": True}, TypeError),
        ({"band_quantile": 1}, ValueError),  # open interval: 1 would make c the max
        ({"band_quantile": 1.5}, ValueError),
        ({"band_quantile": -0.1}, ValueError),
        ({"band_quantile": float("nan")}, ValueError),
        ({"band_quantile": True}, ValueError),
        ({"band_quantile": "0.95"}, ValueError),
    ],
)
def test_config_is_validated(changes, error):
    with pytest.raises(error):
        module._validate_config({**ACTIVE_CONFIG, **changes})


def test_unknown_settings_are_rejected_and_the_known_ones_returned():
    # Revision-one settings are left-overs now: a typo or stale key must not pass.
    with pytest.raises(ValueError, match="Unknown.*comembership_threshold"):
        module._validate_config({**ACTIVE_CONFIG, "comembership_threshold": 0.9})
    with pytest.raises(ValueError, match="monte_carlo_sigmas"):
        module._validate_config({**ACTIVE_CONFIG, "monte_carlo_sigmas": 3.0})
    for retired in ("core_threshold", "consensus_peak_frequency"):
        with pytest.raises(ValueError, match=retired):
            module._validate_config({**ACTIVE_CONFIG, retired: 0.5})
    settings = module._validate_config(ACTIVE_CONFIG)
    assert set(settings) == {
        "n_replicates",
        "seed",
        "band_quantile",
        "n_jobs",
        "chunk_size",
    }
    assert settings["band_quantile"] == 0.95
    assert settings["chunk_size"] == 250  # the default when the key is absent
    assert module.result_settings({**ACTIVE_CONFIG, "chunk_size": 7}) == (
        module.result_settings(ACTIVE_CONFIG)
    )
    with pytest.raises(KeyError):
        module._validate_config(
            {k: v for k, v in ACTIVE_CONFIG.items() if k != "band_quantile"}
        )


# -- the real-versus-flat call ------------------------------------------------


def test_build_runs_the_observed_and_flat_series_with_one_leaf_statistic(monkeypatch):
    pooled = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    result, calls = _build(
        monkeypatch, se=pooled, config={**ACTIVE_CONFIG, "n_jobs": 3, "chunk_size": 5}
    )
    (main, main_call), (flat, flat_call) = calls
    assert (main_call["name"], flat_call["name"]) == ("observed", "flat")
    for call in (main_call, flat_call):
        statistic = call["statistic"]
        assert isinstance(statistic, LeafCooccurrence)
        assert statistic.tree_type == "split"
        assert call["core_threshold"] is None
        store = call["runner"].store
        assert (store.n_draws, store.seed, store.chunk_size) == (20, 7, 5)
        assert call["runner"].n_jobs == 3
    # Both conditions read the REAL nominal tree and the REAL hills.
    main_statistic, flat_statistic = main_call["statistic"], flat_call["statistic"]
    assert main_statistic.reference is flat_statistic.reference
    assert main_statistic.nominal_hills == flat_statistic.nominal_hills
    assert main_statistic.nominal_hills == {"b": [1, 0], "d": [3], "f": [5, 6]}
    reference = main_statistic.reference
    assert {
        str(reference.graph[i]["vertex_id"]) for i in reference.graph.node_indices()
    } == {
        "b",
        "d",
        "f",
        "e",
        "c",
    }
    # The flat network has raw_fitness 0.0 everywhere and the pooled errors.
    raw = [main.graph[i]["raw_fitness"] for i in main.graph.node_indices()]
    assert raw == RAW_FITNESS
    assert {flat.graph[i]["raw_fitness"] for i in flat.graph.node_indices()} == {0.0}
    # Both series carry se_pooled in vertex order (the 10.0 se_replicate is a decoy).
    expected = _growth(se=pooled).growth_rates["se_pooled"].to_list()
    assert [main.graph[i]["se"] for i in main.graph.node_indices()] == expected
    assert [flat.graph[i]["se"] for i in flat.graph.node_indices()] == expected
    assert result.summary["diagnostics"]["flat_landscape"]["constant"] == 0.0


def test_the_flat_copy_must_be_the_same_graph():
    network = _network()
    ids, edges = module._prepared_tables(network)
    flat = module._make_network(ids, edges, np.zeros(7), np.full(7, 0.05))
    module._check_same_graph(network, flat)  # same tables: accepted

    extra = pl.DataFrame({"source": ["a"], "target": ["c"]})
    longer = module._make_network(
        ids, pl.concat([edges, extra]), np.zeros(7), np.full(7, 0.05)
    )
    with pytest.raises(ValueError, match="edges"):
        module._check_same_graph(network, longer)
    star = pl.DataFrame({"source": ["a"] * 6, "target": list("bcdefg")})
    rewired = module._make_network(ids, star, np.zeros(7), np.full(7, 0.05))
    with pytest.raises(ValueError, match="edges"):
        module._check_same_graph(network, rewired)
    reordered = SimpleNamespace(vertex_ids=ids[::-1], graph=flat.graph)
    with pytest.raises(ValueError, match="vertex identities"):
        module._check_same_graph(network, reordered)


def test_build_refuses_a_flat_network_that_is_not_the_same_graph(monkeypatch):
    original = module._make_network
    built = []

    def drifting(ids, edges, raw_fitness, se):
        built.append(1)
        if len(built) == 2:  # the flat copy
            edges = pl.concat([edges, pl.DataFrame({"source": ["a"], "target": ["c"]})])
        return original(ids, edges, raw_fitness, se)

    monkeypatch.setattr(module, "_make_network", drifting)
    calls = _patch_seam(monkeypatch)
    prepared, tree = _toy()
    with pytest.raises(ValueError, match="flat-landscape network.*edges"):
        build_bootstrap_smoothed_tree(
            prepared, tree, _growth(), CONFIG, log=lambda message: None
        )
    assert len(calls) == 1  # the flat bootstrap never ran


def test_distance_conventions_are_recorded(monkeypatch):
    flat = _cooc([["a", "b", "c"]])
    _patch_seam(monkeypatch, flat=flat)
    prepared, tree = _toy()
    messages: list[str] = []
    result = build_bootstrap_smoothed_tree(
        prepared, tree, _growth(), CONFIG, log=messages.append
    )
    assert "memory" not in result.summary
    conventions = result.summary["diagnostics"]["distance_conventions"]
    assert set(conventions) == {
        "wasserstein_1",
        "bottleneck",
        "reproduced_pair_fraction",
    }
    assert "L1 ground metric" in conventions["wasserstein_1"]
    assert "costs its persistence" in conventions["wasserstein_1"]
    assert "L-infinity" in conventions["bottleneck"]
    assert "half its persistence" in conventions["bottleneck"]
    assert (
        "counts the global maximum's (peak, root) pair"
        in (conventions["reproduced_pair_fraction"])
    )
    assert "nominal hills plus 1" not in conventions["reproduced_pair_fraction"]
    assert (
        "equals the number of nominal hills"
        in (conventions["reproduced_pair_fraction"])
    )


def test_the_real_topographer_bootstrap_runs_through_the_stage():
    """No monkeypatch: the stage runs Topographer's chunked bootstrap on the toy path."""
    prepared, tree = _toy()
    result = build_bootstrap_smoothed_tree(
        prepared,
        tree,
        _growth(),
        {**ACTIVE_CONFIG, "n_replicates": 8, "n_jobs": 1},
        log=lambda message: None,
    )
    # Hills were mapped to indices and back: all three nominal peaks are scored.
    assert sorted(result.hills["peak_sequence"]) == ["b", "d", "f"]
    assert result.hills["hill_support"].is_between(0.0, 1.0).all()
    # Exports carry sequence identities, never node indices.
    assert set(result.hills["peak_sequence"]) <= set(DNA)
    assert not hasattr(result, "cores")
    assert result.draw_stats.height == 16
    assert set(result.draw_stats["condition"]) == {"real", "flat"}
    assert json.dumps(result.summary, allow_nan=False)


def test_the_result_must_echo_the_requested_settings(monkeypatch):
    with pytest.raises(ValueError, match="real bootstrap did not run as requested"):
        _build(monkeypatch, real=_cooc(NOMINAL_CORES, seed=99))
    with pytest.raises(ValueError, match="flat bootstrap did not run as requested"):
        _build(monkeypatch, flat=_cooc(NOMINAL_CORES, seed=99))
    with pytest.raises(ValueError, match="draw rows"):
        _build(monkeypatch, real=_cooc(NOMINAL_CORES, n=5, n_replicates=20))


def test_missing_hill_support_fails_closed(monkeypatch):
    with pytest.raises(ValueError, match="returned no hill support"):
        _build(monkeypatch, real=_cooc(NOMINAL_CORES, nominal_hill_support=None))
    partial = _cooc(NOMINAL_CORES).nominal_hill_support.filter(pl.col("hill_id") != "d")
    with pytest.raises(ValueError, match="Hill support is missing.*'d'"):
        _build(monkeypatch, real=_cooc(NOMINAL_CORES, nominal_hill_support=partial))


# -- the stage on the toy tree ------------------------------------------------


def test_direct_band_hills_and_summary(monkeypatch):
    result, _ = _build(monkeypatch)
    summary = result.summary

    assert result.hills.columns == list(module.HILL_COLUMNS)
    assert result.hills["band_outcome"].to_list() == ["absorbed", "absorbed", "kept"]
    assert summary["peaks_before"] == 3 and summary["peaks_after"] == 1
    assert summary["sequences_retained"] == 7
    assert summary["hill_count"] == 3
    assert "core_count" not in summary
    # n_jobs schedules the workers only and may differ between a cached summary
    # and a later call, so it is not a setting of the result; the timing block
    # records what the producing run used.
    assert summary["settings"] == {
        "n_replicates": 20,
        "seed": 7,
        "band_quantile": 0.95,
    }
    assert summary["timing"]["n_jobs"] == 1
    assert result.hills["member_count"].to_list() == [2, 1, 2]
    assert result.hills["hill_support"].to_list() == [1.0, 1.0, 1.0]


def test_core_membership_does_not_change_band_hills(monkeypatch):
    real = _cooc([["d", "f"]], support={"d": 0.6, "b": 0.2})
    result, _ = _build(monkeypatch, real=real)
    rows = {r["peak_sequence"]: r for r in result.hills.iter_rows(named=True)}

    assert {peak: row["band_outcome"] for peak, row in rows.items()} == {
        "b": "absorbed",
        "d": "absorbed",
        "f": "kept",
    }
    assert result.summary["peaks_after"] == 1
    assert result.summary["saddles_removed"] == 1  # e, once d and b are gone
    assert result.summary["sequences_retained"] == 7
    assert result.hills["hill_support"].to_list() == [0.2, 0.6, 1.0]


def test_band_view_keeps_global_peak_above_threshold(monkeypatch):
    result, _ = _build(monkeypatch, real=_cooc([["c", "e"]]))
    assert _peaks(build_band_simplified_view(_toy()[1], result.hills)) == {"f"}


def test_spanning_and_split_candidate_are_counted_together(monkeypatch):
    real = _cooc([["b", "d"], ["a"], ["f"]])
    result, _ = _build(monkeypatch, real=real)
    assert "core_relation" not in result.hills.columns
    assert "split_candidate_count" not in result.summary


def test_core_extending_onto_non_hill_vertices_is_counted_in_the_summary(monkeypatch):
    result, _ = _build(monkeypatch, real=_cooc([["b", "a", "c"], ["d"], ["f", "g"]]))
    assert "cores_extending_outside_hill" not in result.summary
    assert result.hills["band_outcome"].to_list() == ["absorbed", "absorbed", "kept"]


# -- identities ---------------------------------------------------------------


def test_exports_use_sequence_identities_never_node_indices(monkeypatch):
    result, _ = _build(
        monkeypatch,
        real=_cooc([["b", "a"], ["d"], ["f", "g"], ["c"]], peak_frequency={"b": 0.9}),
    )
    assert set(result.hills["peak_sequence"]) == {"b", "d", "f"}
    for frame in (result.hills, result.draw_stats):
        assert not [c for c in frame.columns if "index" in c]
        for column in frame.columns:
            if frame[column].dtype == pl.String:
                assert not frame[column].drop_nulls().str.contains(r"^\d+$").any()


# -- fail fast and the nominal-hill cross-check --------------------------------


def test_a_failing_real_only_check_runs_before_the_flat_bootstrap(monkeypatch):
    # Hill support missing for one hill is a real-result problem: no flat call.
    real = _cooc(
        NOMINAL_CORES,
        nominal_hill_support=pl.DataFrame(
            {"hill_id": ["b", "d"], "hill_support": [1.0, 1.0]}
        ),
    )
    calls = _patch_seam(monkeypatch, real)
    prepared, tree = _toy()
    with pytest.raises(ValueError, match="Hill support is missing.*'f'"):
        build_bootstrap_smoothed_tree(
            prepared, tree, _growth(), CONFIG, log=lambda message: None
        )
    assert len(calls) == 1
    # So is a result that returns no support at all.
    calls = _patch_seam(monkeypatch, _cooc(NOMINAL_CORES, nominal_hill_support=None))
    with pytest.raises(ValueError, match="no hill support"):
        build_bootstrap_smoothed_tree(
            prepared, tree, _growth(), CONFIG, log=lambda message: None
        )
    assert len(calls) == 1


def _replace_cores(result, members):
    """Return ``result`` with the given ``(core_id, sequence)`` core table."""
    import dataclasses

    return dataclasses.replace(
        result,
        leaf_cores=pl.DataFrame(
            {
                "core_id": [core for core, _ in members],
                "vertex_index": [DNA.index(name) for _, name in members],
            },
            schema={"core_id": pl.Int64, "vertex_index": pl.Int64},
        ),
    )


def test_overlapping_retired_cores_do_not_gate_direct_band(monkeypatch):
    real = _cooc(NOMINAL_CORES)
    real = _replace_cores(real, [(0, "b"), (1, "b")])
    calls = _patch_seam(monkeypatch, real)
    prepared, tree = _toy()
    result = build_bootstrap_smoothed_tree(
        prepared, tree, _growth(), CONFIG, log=lambda message: None
    )
    assert len(calls) == 2
    assert result.summary["peaks_after"] == 1


def test_nominal_hills_must_equal_topographers_terminal_branches(monkeypatch):
    prepared, tree = _toy()
    ids, _ = module._prepared_tables(prepared)
    reference = module.ScalarNetworkTopology(_network()).split_tree()
    hills = nominal_hills(tree)
    indices = module.hill_vertex_indices(hills, ids)
    module.check_hills_match_reference(indices, ids, reference)  # the toy agrees

    short = {**indices, "b": [1]}  # a member went missing
    with pytest.raises(ValueError, match="differ.*member sets differ for 'b'"):
        module.check_hills_match_reference(short, ids, reference)
    extra = {**indices, "g": [6]}  # a peak Topographer does not have
    with pytest.raises(ValueError, match="peaks on one side only 'g'"):
        module.check_hills_match_reference(extra, ids, reference)

    # The build refuses before it calls the bootstrap at all.
    calls = _patch_seam(monkeypatch)
    monkeypatch.setattr(module, "terminal_branch_members", lambda ref: {1: [1]})
    with pytest.raises(ValueError, match="terminal branches of the reference tree"):
        build_bootstrap_smoothed_tree(
            prepared, tree, _growth(), CONFIG, log=lambda message: None
        )
    assert calls == []


# -- flat-condition decisions are persisted ------------------------------------


def test_flat_series_does_not_create_core_hill_tables(monkeypatch):
    flat = _cooc([["d", "f"]], support={"b": 0.1, "d": 0.2, "f": 0.3})
    result, _ = _build(monkeypatch, flat=flat)

    assert not hasattr(result, "flat_hills")
    assert not hasattr(result, "flat_cores")
    assert set(result.draw_stats["condition"]) == {"real", "flat"}
    assert result.hills["hill_support"].to_list() == [1.0, 1.0, 1.0]


def test_flat_draws_are_persisted_without_core_false_hill_rate(tmp_path, monkeypatch):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    answers: list = []

    def fake(self, name, network, statistic, *, label=None):
        # The real run finds the nominal cores, the flat run one core on d and f.
        answers.append(1)
        cores = NOMINAL_CORES if len(answers) == 1 else [["d", "f"]]
        return _cooc(
            cores,
            n=self.store.n_draws,
            seed=self.store.seed,
        )

    monkeypatch.setattr(bootstrap_chunks.ChunkedBootstrap, "run", fake)
    compute_bootstrap_smoothed_tree(settings)
    cached = load_cached_bootstrap_smoothed_tree()
    stored = pl.read_parquet(module.cache_dir() / "draw_stats.parquet")
    assert stored.filter(pl.col("condition") == "flat").height == 20
    flat_summary = cached.summary["diagnostics"]["flat_landscape"]
    assert "false_hill_rate" not in flat_summary
    assert cached.draw_stats.equals(stored)


def test_a_checkpoint_without_draw_stats_asks_for_a_rerun(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    result, _ = _build(monkeypatch)
    module._write_checkpoint(result, stage, _checkpoint_fingerprint())
    provenance = json.loads((stage / "provenance.json").read_text())
    del provenance["artifacts"]["draw_stats.parquet"]
    (stage / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="predates.*just bootstrap-smoothed-tree"):
        module._read_checkpoint(stage)


def test_draw_stats_are_part_of_the_checkpoint_hashes(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    result, _ = _build(monkeypatch)
    module._write_checkpoint(result, stage, _checkpoint_fingerprint())
    provenance = json.loads((stage / "provenance.json").read_text())
    assert "draw_stats.parquet" in provenance["artifacts"]
    (stage / "draw_stats.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch: draw_stats.parquet"):
        module._read_checkpoint(stage)


@pytest.mark.parametrize("with_note", [True, False])
def test_schema6_leaf_parent_matches_schema7_producer(tmp_path, monkeypatch, with_note):
    from dubanevics2026topological.analysis import describe
    from dubanevics2026topological.analysis.leaves import (
        direct_band_parent,
        save_band_tree,
    )
    from dubanevics2026topological.landscape.products import _publish
    from dubanevics2026topological.results import (
        RunContext,
        compute_scientific_hash,
        load,
    )

    stage = tmp_path / "checkpoint"
    stage.mkdir()
    result, _ = _build(monkeypatch)
    module._write_checkpoint(result, stage, _checkpoint_fingerprint())
    schema7 = json.loads((stage / "summary.json").read_text())
    expected = schema7.pop("scientific_identities")
    schema7["fingerprint"]["schema"] = 6
    (stage / "summary.json").write_text(json.dumps(schema7))
    artifacts = {
        f"checkpoint/{name}": {
            "sha256": hashlib.sha256((stage / name).read_bytes()).hexdigest()
        }
        for name in ("summary.json", "draw_stats.parquet")
    }
    draw_source = SimpleNamespace(
        product_type="bootstrap",
        report=SimpleNamespace(status="full", byte_integrity=True),
        path=tmp_path,
        legacy_alias="historical",
        label="unused",
        manifest=SimpleNamespace(
            scientific_identity={
                "thr": "published",
                "fdr": "by",
                "q": 0.05,
                "draws": 20,
                "seed": 7,
            },
            scientific_id_sha256="a" * 64,
            data_provenance={},
            artifacts=artifacts,
        ),
    )
    note = (
        {
            "kind": "functional_set_sha256_equivalence",
            "functional_set_sha256": "f" * 64,
            "clean_table_sha256": "c" * 64,
            "dirty_table_sha256": "d" * 64,
        }
        if with_note
        else None
    )
    parent = direct_band_parent(draw_source, schema7, equivalence=note)
    assert (
        parent.lineage["draw_scientific_id_sha256"]
        == expected["draw"]["scientific_id_sha256"]
    )
    assert parent.scientific_id_sha256 == expected["band"]["scientific_id_sha256"]
    assert parent.lineage["historical_checkpoint_method"] == "band_after_core"
    for method in ("core", "band_after_core"):
        historical, _ = compute_scientific_hash(
            "tree",
            {**parent.scientific_identity, "simplification_method": method},
            parent_scientific_ids=(parent.lineage["draw_scientific_id_sha256"],),
        )
        assert parent.scientific_id_sha256 != historical
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    params = {"thr": "published", "fdr": "by", "q": 0.05, "tree": "nominal"}
    nominal_path, _ = _publish(
        "tree",
        params,
        params,
        context=context,
        input_data_sha256s={},
        parents=(),
        code_paths=(),
        write_artifacts=lambda directory: (directory / "tree.json").write_text("{}"),
    )
    nominal = load("tree", nominal_path.name, context=context)
    _, tree = _toy()
    view = module.build_band_simplified_view(tree, result.hills)
    band_path = save_band_tree(view, band=parent, nominal=nominal, context=context)
    band_tree = load("tree", band_path.name, context=context)
    assert band_tree.report.status == "full" and band_tree.report.byte_integrity
    assert {
        "branches.parquet",
        "branch_members.parquet",
        "critical_nodes.parquet",
        "persistence_pairs.parquet",
        "root_members.parquet",
    } <= band_tree.artifacts.keys()
    ref = next(
        p
        for p in band_tree.manifest.parents
        if p["product_type"] == "direct_band_identity"
    )
    assert ref["scientific_id_sha256"] == parent.scientific_id_sha256
    assert (
        ref["source_lineage"]["draw_scientific_id_sha256"]
        == expected["draw"]["scientific_id_sha256"]
    )
    leaves = describe(
        products={"leaves"},
        leaf_tree=tree,
        leaf_population="by_published",
        leaf_tree_state="band",
        leaf_hills=result.hills,
    ).leaves
    assert ref["source_lineage"].get("functional_set_equivalence") == note
    leaf_path = leaves.save(parent=band_tree, context=context)
    loaded = load("leaves", leaf_path.name, context=context)
    leaf_ref = next(p for p in loaded.manifest.parents if p["product_type"] == "tree")
    assert leaf_ref.get("functional_set_equivalence") == note
    for ref in loaded.manifest.parents:
        resolved = load(ref["product_type"], ref["label"], context=context)
        assert resolved.manifest.scientific_id_sha256 == ref["scientific_id_sha256"]


def test_checkpoint_draw_identity_tracks_threshold_and_growth_provenance(
    tmp_path, monkeypatch
):
    result, _ = _build(monkeypatch)
    baseline = _checkpoint_fingerprint()
    baseline["growth_rates_checkpoint"] = {
        "fingerprint": {
            "config": {"elapsed_time": 3.5},
            "files": {"barseq_export/counts.parquet": "c" * 64},
        }
    }
    cases = {
        "threshold": {
            **baseline,
            "candidate_functional_threshold": -0.9,
        },
        "growth_setting": {
            **baseline,
            "growth_rates_checkpoint": {
                "fingerprint": {
                    "config": {"elapsed_time": 3.6},
                    "files": {"barseq_export/counts.parquet": "c" * 64},
                }
            },
        },
        "growth_input": {
            **baseline,
            "growth_rates_checkpoint": {
                "fingerprint": {
                    "config": {"elapsed_time": 3.5},
                    "files": {"barseq_export/counts.parquet": "d" * 64},
                }
            },
        },
    }
    stage = tmp_path / "baseline"
    stage.mkdir()
    module._write_checkpoint(result, stage, baseline)
    baseline_id = json.loads((stage / "summary.json").read_text())[
        "scientific_identities"
    ]["draw"]["scientific_id_sha256"]
    for name, fingerprint in cases.items():
        stage = tmp_path / name
        stage.mkdir()
        module._write_checkpoint(result, stage, fingerprint)
        observed = json.loads((stage / "summary.json").read_text())[
            "scientific_identities"
        ]["draw"]["scientific_id_sha256"]
        assert observed != baseline_id


def test_checkpoint_only_band_change_fails_with_evidence_required_message(
    tmp_path, monkeypatch
):
    result, _ = _build(monkeypatch)
    slot = tmp_path / "slot"
    slot.mkdir()
    monkeypatch.setattr(module, "cache_dir", lambda context=None: slot)
    fingerprint = _checkpoint_fingerprint()
    module._write_checkpoint(result, slot, fingerprint)
    wanted = {
        **result.summary["settings"],
        "band_quantile": result.summary["settings"]["band_quantile"] - 0.05,
    }
    scientific = module.checkpoint_identity.scientific_fingerprint(fingerprint)

    with pytest.raises(
        RuntimeError, match="Evidence required.*refusing to silently draw"
    ):
        module._refuse_checkpoint_only_band_change(wanted, scientific)


# -- global verdict, flat null, draw statistics -------------------------------


def test_draw_stats_carry_both_conditions(monkeypatch):
    real = _cooc(NOMINAL_CORES, wasserstein_1=[0.1 * i for i in range(20)])
    flat = _cooc(NOMINAL_CORES, wasserstein_1=[10.0 + i for i in range(20)])
    result, _ = _build(monkeypatch, real=real, flat=flat)
    draws = result.draw_stats
    assert draws.columns == list(module.DRAW_COLUMNS)
    assert draws.height == 40
    assert draws.filter(pl.col("condition") == "real")["draw"].to_list() == list(
        range(20)
    )
    assert draws.filter(pl.col("condition") == "flat")["wasserstein_1"].min() == 10.0


def _draws_of(real_bottleneck, real_w1=None, flat_bottleneck=None):
    """Return a hand-built ``draw_stats`` table with both conditions."""
    n = len(real_bottleneck)
    flat_bottleneck = flat_bottleneck or [99.0] * n
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "draw": list(range(n)),
                    "condition": [condition] * n,
                    "peak_count": peaks,
                    "wasserstein_1": w1,
                    "bottleneck": bottleneck,
                    "reproduced_pair_fraction": [1.0] * n,
                }
            )
            for condition, peaks, w1, bottleneck in (
                ("real", [3] * n, real_w1 or [1.0] * n, real_bottleneck),
                ("flat", list(range(n)), [1000.0] * n, flat_bottleneck),
            )
        ]
    )


def _diagram_hills(persistence, outcomes):
    """Return the columns of ``hills`` that the verdict reads."""
    return pl.DataFrame(
        {"nominal_persistence": persistence, "outcome": outcomes},
        schema={"nominal_persistence": pl.Float64, "outcome": pl.String},
    )


def test_band_width_is_the_quantile_of_the_real_bottleneck_only():
    real = [float(i) / 10 for i in range(1, 22)]  # 0.1 .. 2.1, linear quantiles
    draws = _draws_of(real, flat_bottleneck=[50.0] * 21)
    # numpy's linear estimator: 0.1 + 0.95 * (21 - 1) * 0.1 = 2.0
    assert module.band_width(draws, 0.95) == pytest.approx(
        float(np.quantile(real, 0.95))
    )
    assert module.band_width(draws, 0.95) == pytest.approx(2.0)
    assert module.band_width(draws, 0.5) == pytest.approx(1.1)
    assert module.QUANTILE_METHOD == "linear"
    with pytest.raises(ValueError, match="no 'real' draws"):
        module.band_width(draws.filter(pl.col("condition") == "flat"), 0.95)
    bad = _draws_of([0.1, float("nan"), 0.3])
    with pytest.raises(ValueError, match="non-finite"):
        module.band_width(bad, 0.95)


def test_global_verdict_counts_points_above_two_c_strictly_and_cross_tabs_outcomes():
    # Real bottleneck 0..20 / 100: the 0.95 quantile is c = 0.19, 2c = 0.38.
    n = 21
    draws = _draws_of([i / 100 for i in range(n)], real_w1=[float(i) for i in range(n)])
    c = module.band_width(draws, 0.95)
    two_c = 2.0 * c
    assert (c, two_c) == (pytest.approx(0.19), pytest.approx(0.38))
    hills = _diagram_hills(
        [two_c, math.nextafter(two_c, 1.0), 2.0, 0.01, 0.05, 1.0],
        ["kept", "kept", "absorbed", "absorbed", "elder_protected", "elder_protected"],
    )
    verdict = module.global_verdict(draws, hills, 0.95)

    assert verdict["band_quantile"] == 0.95
    assert verdict["quantile_method"] == "linear"
    assert verdict["c"] == c and verdict["two_c"] == two_c
    assert verdict["nominal_point_count"] == 6
    # A point exactly at 2c is not significant (strict >); the next float is.
    assert verdict["significant_point_count"] == 3
    assert "significance_by_outcome" not in verdict
    assert json.dumps(verdict, allow_nan=False)


def test_relative_wasserstein_1_divides_by_the_total_nominal_persistence():
    n = 20
    w1 = [2.0] * 10 + [4.0] * 10
    draws = _draws_of([0.1] * n, real_w1=w1)
    hills = _diagram_hills([1.0, 2.0, 5.0], ["kept", "kept", "absorbed"])
    verdict = module.global_verdict(draws, hills, 0.95)
    assert verdict["nominal_total_persistence"] == 8.0
    relative = verdict["relative_wasserstein_1"]
    assert relative["median"] == pytest.approx(3.0 / 8.0)
    assert relative["p2_5"] == pytest.approx(float(np.percentile(w1, 2.5)) / 8.0)
    assert relative["p97_5"] == pytest.approx(float(np.percentile(w1, 97.5)) / 8.0)
    with pytest.raises(ValueError, match="no persistence"):
        module.global_verdict(draws, _diagram_hills([0.0], ["kept"]), 0.95)


def test_global_verdict_keeps_real_only_descriptive_statistics():
    n = 20
    real_w1 = [0.5] * 5 + [5.0] * 15
    draws = _draws_of([0.0] * n, real_w1=real_w1)
    verdict = module.global_verdict(draws, _diagram_hills([1.0], ["kept"]), 0.95)
    # No real-versus-flat comparison of any kind survives in the verdict.
    assert "flat" not in verdict
    assert not [key for key in verdict if "flat" in key or "share_below" in key]
    assert set(verdict["real"]) == {"wasserstein_1", "bottleneck", "peak_count"}
    assert verdict["real"]["wasserstein_1"] == {
        "median": 5.0,
        "p2_5": pytest.approx(float(np.percentile(real_w1, 2.5))),
        "p97_5": 5.0,
    }
    assert verdict["real"]["peak_count"] == {"median": 3.0, "p2_5": 3.0, "p97_5": 3.0}
    assert verdict["real"]["bottleneck"]["median"] == 0.0
    # The flat draws (here far away) never enter any number.
    assert verdict["c"] == 0.0
    assert verdict["real"]["wasserstein_1"]["median"] == 5.0


def test_sanity_check_compares_real_and_flat_bottleneck_and_is_not_the_verdict():
    n = 20
    real = [0.1 + 0.01 * i for i in range(n)]  # 0.10 .. 0.29
    flat = [0.77 + 0.01 * i for i in range(n)]  # 0.77 .. 0.96
    draws = _draws_of(real, flat_bottleneck=flat)
    verdict = module.global_verdict(draws, _diagram_hills([1.0], ["kept"]), 0.95)
    check = verdict["sanity_check"]
    assert check["role"] == "sanity check, not the verdict"
    assert check["statistic"] == "bottleneck"
    assert "real nominal diagram" in check["reference"]
    assert check["real"] == {
        "median": pytest.approx(float(np.median(real))),
        "p2_5": pytest.approx(float(np.percentile(real, 2.5))),
        "p97_5": pytest.approx(float(np.percentile(real, 97.5))),
    }
    assert check["flat"] == {
        "median": pytest.approx(float(np.median(flat))),
        "p2_5": pytest.approx(float(np.percentile(flat, 2.5))),
        "p97_5": pytest.approx(float(np.percentile(flat, 97.5))),
    }
    assert check["flat_bottleneck_p5"] == pytest.approx(float(np.percentile(flat, 5)))
    assert check["real_share_below_flat_bottleneck_p5"] == 1.0
    # The check is the same block as the real descriptive interval, not the band.
    assert check["real"] == verdict["real"]["bottleneck"]
    assert verdict["c"] == pytest.approx(float(np.quantile(real, 0.95)))
    assert verdict["significant_point_count"] == 1
    assert json.dumps(verdict, allow_nan=False)


def test_sanity_check_share_is_strictly_below_the_flat_fifth_percentile():
    n = 21
    flat = [float(i) for i in range(n)]  # p5 = 1.0 exactly
    # Real draws: 0.5 and 0.99 are below 1.0; 1.0 (equal) and 2.0 are not.
    real = [0.5, 0.99, 1.0, 2.0] + [3.0] * (n - 4)
    verdict = module.global_verdict(
        _draws_of(real, flat_bottleneck=flat), _diagram_hills([1.0], ["kept"]), 0.95
    )
    check = verdict["sanity_check"]
    assert check["flat_bottleneck_p5"] == 1.0
    assert check["real_share_below_flat_bottleneck_p5"] == pytest.approx(2 / n)


def test_sanity_check_has_no_wasserstein_comparison_across_conditions():
    n = 20
    draws = _draws_of([0.1] * n, real_w1=[1.0] * n)  # flat W1 is 1000 in the helper
    verdict = module.global_verdict(draws, _diagram_hills([1.0], ["kept"]), 0.95)
    text = json.dumps(verdict)
    assert "1000" not in text  # the flat W1 never enters
    assert not [k for k in verdict["sanity_check"] if "wasserstein" in k.lower()]
    assert "flat_wasserstein_1_p5" not in text
    assert "real_share_below_flat_p5" not in text
    assert set(verdict["relative_wasserstein_1"]) == {"median", "p2_5", "p97_5"}


def test_sanity_check_refuses_missing_or_non_finite_conditions():
    draws = _draws_of([0.1] * 20)
    hills = _diagram_hills([1.0], ["kept"])
    with pytest.raises(ValueError, match="no 'flat' draws"):
        module.global_verdict(draws.filter(pl.col("condition") == "real"), hills, 0.95)
    with pytest.raises(ValueError, match="flat bottleneck has non-finite"):
        module.global_verdict(
            _draws_of([0.1] * 20, flat_bottleneck=[float("nan")] * 20), hills, 0.95
        )


def test_global_verdict_refuses_non_finite_draws():
    draws = _draws_of([0.1] * 20)
    bad = draws.with_columns(
        pl.when(pl.col("condition") == "real")
        .then(float("inf"))
        .otherwise(pl.col("wasserstein_1"))
        .alias("wasserstein_1")
    )
    with pytest.raises(ValueError, match="non-finite"):
        module.global_verdict(bad, _diagram_hills([1.0], ["kept"]), 0.95)


def test_the_build_records_the_band_from_the_real_draws_only(monkeypatch):
    n = 20
    real = _cooc(NOMINAL_CORES, wasserstein_1=[0.1 * i for i in range(n)])
    real = dataclasses.replace(
        real,
        draw_stats=real.draw_stats.with_columns(
            pl.Series("bottleneck", [0.01 * i for i in range(n)])
        ),
    )
    flat = _cooc(NOMINAL_CORES, wasserstein_1=[10.0 + i for i in range(n)])
    result, _ = _build(monkeypatch, real=real, flat=flat)
    verdict = result.summary["diagnostics"]["global_verdict"]
    c = float(np.quantile([0.01 * i for i in range(n)], 0.95))
    assert verdict["c"] == pytest.approx(c) and verdict["two_c"] == pytest.approx(2 * c)
    assert verdict["real"]["wasserstein_1"]["median"] == pytest.approx(0.95)
    assert verdict["nominal_point_count"] == result.hills.height
    assert result.summary["settings"]["band_quantile"] == 0.95
    # hills.parquet carries the per-point columns; clears_band follows 2c strictly.
    hills = result.hills
    assert hills["nominal_persistence"].dtype == pl.Float64
    assert hills["clears_band"].dtype == pl.Boolean
    assert hills["clears_band"].to_list() == (
        (hills["nominal_persistence"] > 2 * c).to_list()
    )
    assert int(hills["clears_band"].sum()) == verdict["significant_point_count"]
    assert hills["nominal_persistence"].to_list() == pytest.approx(
        (hills["nominal_birth"] - hills["nominal_death"]).to_list()
    )
    assert not hasattr(result, "flat_hills")


def test_the_build_logs_the_band(monkeypatch):
    messages: list[str] = []
    _patch_seam(monkeypatch)
    prepared, tree = _toy()
    build_bootstrap_smoothed_tree(
        prepared, tree, _growth(), CONFIG, log=messages.append
    )
    (line,) = [m for m in messages if m.startswith("Confidence band:")]
    assert "0.95 quantile" in line and "of 3 nominal diagram points" in line
    assert not [m for m in messages if "flat 5th" in m or "below flat" in m]


# -- the nominal diagram ------------------------------------------------------


def test_nominal_diagram_of_the_toy_tree_is_ordinary_pairs_plus_the_closed_global_pair():
    # Peaks b 0.5, d 0.4, f 0.6; saddle e -0.15 joins d and f, root c -0.2 joins b
    # and the e arm. d dies at e, b dies at c, the global peak f is closed at c.
    _, tree = _toy()
    diagram = module.nominal_diagram(tree, nominal_hills(tree))
    assert diagram.columns == list(module.DIAGRAM_COLUMNS)
    rows = {r["peak_sequence"]: r for r in diagram.iter_rows(named=True)}
    assert set(rows) == {"b", "d", "f"}  # one point per hill, by sequence identity
    assert (rows["d"]["birth"], rows["d"]["death"]) == (0.4, -0.15)
    assert (rows["b"]["birth"], rows["b"]["death"]) == (0.5, -0.2)
    assert (rows["f"]["birth"], rows["f"]["death"]) == (0.6, -0.2)  # global pair
    for row in rows.values():
        assert row["persistence"] == pytest.approx(row["birth"] - row["death"])
        assert row["persistence"] >= 0
    assert diagram["persistence"].sum() == pytest.approx(0.55 + 0.7 + 0.8)


def test_nominal_diagram_equals_the_diagram_topographer_measures_draws_against():
    from dubanevics2026topological.simplification.cooccurrence_bootstrap import _persistence_diagram

    network, tree = _toy()
    reference = check_identical_tree(network, tree)
    theirs = _persistence_diagram(reference.graph, tree_kind="split", root_to_leaf=True)
    ours = module.nominal_diagram(tree, nominal_hills(tree))
    assert ours.height == theirs.points.shape[0] == len(theirs.pairs)
    assert sorted(zip(ours["birth"], ours["death"], strict=True)) == sorted(
        map(tuple, theirs.points.tolist())
    )


def test_nominal_diagram_count_is_asserted_against_the_hills():
    _, tree = _toy()
    hills = nominal_hills(tree)
    with pytest.raises(ValueError, match="3 points for 2 hills"):
        module.nominal_diagram(tree, hills[:2])
    short = SimpleNamespace(
        tree=tree.tree, persistence_table=tree.persistence_table.head(2)
    )
    with pytest.raises(ValueError, match="2 points for 3 hills"):
        module.nominal_diagram(short, hills)
    no_global = SimpleNamespace(
        tree=tree.tree,
        persistence_table=tree.persistence_table.with_columns(
            pl.lit("ordinary").alias("pair_kind")
        ),
    )
    with pytest.raises(ValueError, match="exactly one global"):
        module.nominal_diagram(no_global, hills)


def test_nominal_diagram_global_pair_must_close_at_the_root():
    _, tree = _toy()
    shifted = tree.persistence_table.with_columns(
        pl.when(pl.col("pair_kind") == "extended")
        .then(pl.col("death_scalar") + 0.01)
        .otherwise(pl.col("death_scalar"))
        .alias("death_scalar")
    )
    broken = SimpleNamespace(tree=tree.tree, persistence_table=shifted)
    with pytest.raises(ValueError, match="closed at the root"):
        module.nominal_diagram(broken, nominal_hills(tree))


def test_reproduced_pair_fraction_must_be_a_multiple_of_the_diagram_size():
    draws = pl.DataFrame({"reproduced_pair_fraction": [0.0, 1 / 3, 2 / 3, 1.0]})
    module.check_reproduced_pair_denominator(draws, 3)
    with pytest.raises(ValueError, match="multiple of 1 / 4"):
        module.check_reproduced_pair_denominator(draws, 4)
    with pytest.raises(ValueError, match="multiple of 1 / 3"):
        module.check_reproduced_pair_denominator(
            pl.DataFrame({"reproduced_pair_fraction": [float("nan")]}), 3
        )


def test_the_real_topographer_run_has_a_denominator_equal_to_the_hill_count():
    prepared, tree = _toy()
    result = build_bootstrap_smoothed_tree(
        prepared,
        tree,
        _growth(),
        {**CONFIG, "n_replicates": 8, "n_jobs": 1},
        log=lambda message: None,
    )
    fractions = result.draw_stats["reproduced_pair_fraction"].to_numpy()
    assert np.allclose(fractions * 3, np.round(fractions * 3))
    assert result.summary["diagnostics"]["global_verdict"]["nominal_point_count"] == 3


def test_flat_condition_is_only_a_distance_sanity_check(monkeypatch):
    flat = _cooc([["b"], ["c", "e"]])
    result, _ = _build(monkeypatch, flat=flat)
    info = result.summary["diagnostics"]["flat_landscape"]
    assert info["n_replicates"] == 20
    assert "false_hill_rate" not in info


def test_flat_core_membership_does_not_create_false_hill_rate(monkeypatch):
    result, _ = _build(monkeypatch)  # flat cores == nominal cores
    assert "false_hill_rate" not in result.summary["diagnostics"]["flat_landscape"]


# -- diagnostics --------------------------------------------------------------


def test_build_takes_hill_support_from_the_result_and_has_no_singleton_check(
    monkeypatch,
):
    result, _ = _build(monkeypatch, real=_cooc(NOMINAL_CORES, support={"d": 0.7}))
    row = result.hills.filter(pl.col("peak_sequence") == "d").row(0, named=True)
    assert row["hill_support"] == 0.7
    assert "singleton_check" not in result.summary["diagnostics"]


def test_exact_ties_count_equal_float64_values_without_tolerance():
    assert module.exact_ties([0.1, 0.2, 0.3]) == {"group_count": 0, "tied_count": 0}
    assert module.exact_ties([0.5, 0.5, 0.5, 0.25, 0.25, 0.1]) == {
        "group_count": 2,
        "tied_count": 5,
    }
    # 0.1 + 0.2 is not exactly 0.3 in float64, so they are not tied.
    assert module.exact_ties([0.1 + 0.2, 0.3]) == {"group_count": 0, "tied_count": 0}


def test_build_records_exact_tie_counts_in_the_summary(monkeypatch):
    result, _ = _build(monkeypatch)
    assert result.summary["diagnostics"]["exact_ties"] == {
        "tree_node_height": {"group_count": 0, "tied_count": 0},
        "network_raw_fitness": {"group_count": 0, "tied_count": 0},
    }

    # b and d tie at 0.5 (both tree nodes); a and g tie at 0.1 (regular vertices).
    heights = [0.1, 0.5, -0.2, 0.5, -0.15, 0.6, 0.1]
    _patch_seam(monkeypatch)
    prepared, tree = _toy(heights)
    tied = build_bootstrap_smoothed_tree(
        prepared, tree, _growth(heights), CONFIG, log=lambda message: None
    )
    assert tied.summary["diagnostics"]["exact_ties"] == {
        "tree_node_height": {"group_count": 1, "tied_count": 2},
        "network_raw_fitness": {"group_count": 2, "tied_count": 4},
    }
    assert json.dumps(tied.summary, allow_nan=False)


def test_summary_is_strict_json_and_complete(monkeypatch):
    result, _ = _build(
        monkeypatch, real=_cooc(NOMINAL_CORES, peak_frequency={"b": 1.0})
    )
    text = json.dumps(result.summary, allow_nan=False)

    def refuse(name):
        raise AssertionError(f"non-strict JSON constant {name}")

    loaded = json.loads(text, parse_constant=refuse)
    assert set(loaded) >= {
        "settings",
        "vertex_count",
        "hill_count",
        "peaks_before",
        "peaks_after",
        "saddles_removed",
        "sequences_retained",
        "diagnostics",
        "timing",
    }
    assert set(loaded["diagnostics"]) == {
        "distance_conventions",
        "global_verdict",
        "flat_landscape",
        "exact_ties",
    }
    assert set(loaded["timing"]) == {
        "n_jobs",
        "real_seconds",
        "real_draws",
        "real_seconds_per_draw",
        "flat_seconds",
        "flat_draws",
        "flat_seconds_per_draw",
        "total_seconds",
    }
    verdict = loaded["diagnostics"]["global_verdict"]
    assert set(verdict) == {
        "band_quantile",
        "quantile_method",
        "c",
        "two_c",
        "significance_rule",
        "nominal_point_count",
        "significant_point_count",
        "nominal_total_persistence",
        "relative_wasserstein_1",
        "real",
        "sanity_check",
    }
    assert set(verdict["sanity_check"]) == {
        "role",
        "statistic",
        "reference",
        "real",
        "flat",
        "flat_bottleneck_p5",
        "real_share_below_flat_bottleneck_p5",
    }
    assert verdict["sanity_check"]["role"] == "sanity check, not the verdict"
    assert not [key for key in text.split('"') if "flat_wasserstein" in key]
    assert "real_share_below_flat_p5" not in text
    assert "false_hill_rate" not in loaded["diagnostics"]["flat_landscape"]
    assert "consensus_peaks" not in loaded["diagnostics"]


# -- step 2: the band-simplified view -------------------------------------------
# Only b has a core (d is absorbed, f is protected), so the core-rule view keeps
# b and f. A bottleneck of 0.3 gives 2c = 0.6: f (0.8) and b (0.7) clear it, d
# (0.55) does not, so the band view keeps the same two peaks.
BAND_CORES = [["b", "a"]]


def _band_build(monkeypatch, cores=None, bottleneck=0.3):
    real = _cooc(BAND_CORES if cores is None else cores, bottleneck=[bottleneck] * 20)
    result, _ = _build(monkeypatch, real=real)
    return result, _toy()[1]


def _pair_persistence(view) -> dict[str, float]:
    return dict(zip(view.pairs["birth_label"], view.pairs["persistence"], strict=True))


def test_band_view_keeps_the_peaks_that_clear_two_c(monkeypatch):
    result, tree = _band_build(monkeypatch)
    view = build_band_simplified_view(tree, result.hills)
    assert _peaks(view) == {"b", "f"}
    assert set(view.pairs["birth_label"]) == {"b", "f"}
    assert set(result.hills.filter(pl.col("clears_band"))["peak_sequence"]) == {
        "b",
        "f",
    }
    assert view.assignments.height == 7  # every sequence keeps its place


def test_band_view_leaves_surviving_persistence_unchanged(monkeypatch):
    result, tree = _band_build(monkeypatch)
    view = build_band_simplified_view(tree, result.hills)
    nominal = dict(
        zip(
            result.hills["peak_sequence"],
            result.hills["nominal_persistence"],
            strict=True,
        )
    )
    persistence = _pair_persistence(view)
    assert persistence == pytest.approx({k: nominal[k] for k in ("b", "f")})
    assert persistence == pytest.approx({"b": 0.7, "f": 0.8})
    assert nominal["d"] == pytest.approx(0.55)


def test_band_outcome_column_follows_clears_band(monkeypatch):
    result, _ = _band_build(monkeypatch)
    for table in (result.hills,):
        assert table.schema["band_outcome"] == pl.String
        assert table.columns[-1] == "band_outcome"
        assert dict(
            zip(table["peak_sequence"], table["band_outcome"], strict=True)
        ) == {
            "b": "kept",
            "d": "absorbed",
            "f": "kept",
        }
        assert (table["band_outcome"] == "kept").to_list() == table[
            "clears_band"
        ].to_list()


def _tampered(result, peak, **values):
    return result.hills.with_columns(
        *[
            pl.when(pl.col("peak_sequence") == peak)
            .then(pl.lit(value))
            .otherwise(pl.col(name))
            .alias(name)
            for name, value in values.items()
        ]
    )


def test_band_view_ignores_core_rule_when_peak_clears_the_band(
    monkeypatch,
):
    result, tree = _band_build(monkeypatch)
    # d is absorbed by the core rule but made to clear the band.
    hills = _tampered(result, "d", clears_band=True, band_outcome="kept")
    assert "d" in set(build_band_simplified_view(tree, hills).pairs["birth_label"])


def test_band_view_refuses_peaks_that_differ_from_the_hills_clearing_the_band(
    monkeypatch,
):
    result, tree = _band_build(monkeypatch)
    original = module.absorb_terminal_branches
    monkeypatch.setattr(
        module, "absorb_terminal_branches", lambda t, ids: original(t, set())
    )
    with pytest.raises(
        ValueError, match="differ from the hills that clear the band.*'d'"
    ):
        build_band_simplified_view(tree, result.hills)


def test_band_view_refuses_changed_persistence(monkeypatch):
    result, tree = _band_build(monkeypatch)
    original = module.absorb_terminal_branches

    def shifted(tree_analysis, peak_ids):
        view = original(tree_analysis, peak_ids)
        return dataclasses.replace(
            view,
            pairs=view.pairs.with_columns(
                pl.when(pl.col("birth_label") == "b")
                .then(pl.col("persistence") + 0.01)
                .otherwise(pl.col("persistence"))
                .alias("persistence")
            ),
        )

    monkeypatch.setattr(module, "absorb_terminal_branches", shifted)
    with pytest.raises(ValueError, match="changed the persistence.*'b'"):
        build_band_simplified_view(tree, result.hills)


def test_band_view_refuses_inconsistent_or_unknown_band_outcomes(monkeypatch):
    result, tree = _band_build(monkeypatch)
    with pytest.raises(ValueError, match="disagrees with clears_band"):
        build_band_simplified_view(tree, _tampered(result, "d", band_outcome="kept"))
    with pytest.raises(ValueError, match="Unknown band outcomes"):
        build_band_simplified_view(tree, _tampered(result, "d", band_outcome="merged"))


def test_the_build_allows_band_peaks_outside_core_view(monkeypatch):
    # Cores keep only f's hill; b (0.7) clears 2c = 0.6 yet the core rule
    # absorbs it, so the band tree is not nested in the core-rule tree.
    result, tree = _band_build(monkeypatch, cores=[["f", "g"]])
    assert "b" in set(
        build_band_simplified_view(tree, result.hills).pairs["birth_label"]
    )
    assert "nested" not in result.summary["band_simplification"]


def test_band_simplification_summary_is_strict_json_and_complete(monkeypatch):
    result, _ = _band_build(monkeypatch, cores=NOMINAL_CORES)
    block = result.summary["band_simplification"]
    loaded = json.loads(json.dumps(result.summary, allow_nan=False))[
        "band_simplification"
    ]
    assert loaded == block
    c = result.summary["diagnostics"]["global_verdict"]["c"]
    assert block["c"] == pytest.approx(0.3) == pytest.approx(c)
    assert block["two_c"] == pytest.approx(0.6)
    assert block["rule"] == "nominal persistence > 2c (strict)"
    assert block["peaks_before"] == result.summary["peaks_before"] == 3
    assert block["peaks_after"] == result.summary["peaks_after"] == 2
    # Absorbing d turns its saddle e into a regular vertex that is contracted.
    assert block["saddles_removed"] == 1
    assert block["saddles_removed"] == result.summary["saddles_removed"]
    assert isinstance(block["band_tree_node_count"], int)
    assert block["band_tree_node_count"] > 0
    assert "lost_after_core_rule" not in block
    assert block["peaks_after"] == int(result.hills["clears_band"].sum())


def test_band_simplification_has_no_core_rule_summary(monkeypatch):
    result, _ = _band_build(monkeypatch)
    block = result.summary["band_simplification"]
    assert block["peaks_after"] == 2
    assert "peaks_after_core_rule" not in block
    assert "lost_after_core_rule" not in block


def test_the_stage_fingerprint_covers_the_module_that_holds_the_band_view():
    import inspect

    source = inspect.getsource(module._fingerprint)
    assert "Path(__file__).resolve()" in source
    assert module.build_band_simplified_view.__module__ == module.__name__


# -- checkpoint lifecycle and exports ----------------------------------------


def _pending(stage_module) -> Path:
    """The helper's marker for a publish in flight, beside the checkpoint folder."""
    directory = stage_module._published_leaf()
    return directory.parent / f".{directory.name}.pending"


def _patched_compute(tmp_path, monkeypatch, builds: list):
    monkeypatch.setattr(
        module,
        "cache_dir",
        lambda context=None: tmp_path / "analysis/bootstrap_smoothed_tree",
    )
    monkeypatch.setattr(
        module,
        "checkpoint_dir",
        lambda context=None: tmp_path / "checkpoints/bootstrap_smoothed_tree",
    )
    real_sha256 = module._sha256

    def sha256(path):  # honest digests for files the stage wrote, fake for inputs
        return real_sha256(path) if Path(path).is_relative_to(tmp_path) else "hash"

    monkeypatch.setattr(module, "_sha256", sha256)
    monkeypatch.setattr(
        module, "load_cached_sequence_growth_rates", lambda **_: _growth()
    )
    monkeypatch.setattr(module, "_check_growth_current", lambda s, g, **_: None)
    monkeypatch.setattr(bootstrap_chunks.ChunkedBootstrap, "run", _echo)

    def nominal(settings, **_):
        builds.append(1)
        return _toy()

    monkeypatch.setattr(module, "build_nominal_tree", nominal)
    settings = {
        "analysis": {
            module.SECTION: {
                key: value
                for key, value in CONFIG.items()
                if key not in {"core_threshold", "consensus_peak_frequency"}
            },
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                }
            },
            "dataset": {"fitness_wt": "wt.parquet", "fitness_diff": "diff.parquet"},
        }
    }
    monkeypatch.setattr(module, "load_settings", lambda: settings)
    return settings


def test_compute_publishes_a_checkpoint_and_reuses_it(tmp_path, monkeypatch):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    first = compute_bootstrap_smoothed_tree(settings)
    cache = module.cache_dir()

    assert sorted(p.name for p in cache.iterdir()) == sorted(
        [*module.ARTIFACTS, "provenance.json"]
    )
    assert sorted(module.ARTIFACTS) == [
        "draw_stats.parquet",
        "hills.parquet",
        "summary.json",
    ]
    assert not _pending(module).exists()
    provenance = json.loads((cache / "provenance.json").read_text())
    assert set(provenance["artifacts"]) == set(module.ARTIFACTS)
    summary = json.loads((cache / "summary.json").read_text())
    assert summary["fingerprint"] == provenance["fingerprint"]
    assert provenance["fingerprint"]["config"] == {
        "n_replicates": 20,
        "seed": 7,
        "band_quantile": 0.95,
        "noise_model": "pooled",
    }
    assert provenance["fingerprint"]["schema"] == module.CACHE_SCHEMA == 7
    assert provenance["fingerprint"]["growth_rates_checkpoint"] == _growth().provenance

    hills = pl.read_parquet(cache / "hills.parquet")
    assert hills.columns == list(module.HILL_COLUMNS)
    assert hills.schema["nominal_persistence"] == pl.Float64
    assert hills.schema["clears_band"] == pl.Boolean
    assert hills.schema["band_outcome"] == pl.String
    assert summary["band_simplification"]["peaks_after"] == int(
        hills["clears_band"].sum()
    )
    assert summary["settings"]["band_quantile"] == 0.95
    draws = pl.read_parquet(cache / "draw_stats.parquet")
    assert draws.columns == [
        "draw",
        "condition",
        "peak_count",
        "wasserstein_1",
        "bottleneck",
        "reproduced_pair_fraction",
    ]
    assert set(draws["condition"]) == {"real", "flat"}
    for frame in (hills, draws):
        assert not [c for c in frame.columns if "index" in c]

    again = compute_bootstrap_smoothed_tree(settings)
    assert builds == [1]
    assert again.summary == first.summary
    assert again.hills.equals(first.hills)
    assert again.draw_stats.equals(first.draw_stats)
    assert load_cached_bootstrap_smoothed_tree().summary == first.summary

    compute_bootstrap_smoothed_tree(settings, force=True)
    assert builds == [1, 1]


def test_n_jobs_does_not_change_the_fingerprint_but_the_seed_does(
    tmp_path, monkeypatch
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    compute_bootstrap_smoothed_tree(settings)
    settings["analysis"][module.SECTION]["n_jobs"] = 2
    compute_bootstrap_smoothed_tree(settings)
    assert builds == [1]
    settings["analysis"][module.SECTION]["seed"] = 8
    compute_bootstrap_smoothed_tree(settings)
    assert builds == [1, 1]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("seed", 8),
        ("band_quantile", 0.9),
    ],
)
def test_result_settings_are_part_of_the_fingerprint(name, value):
    assert module.result_settings({**CONFIG, name: value}) != module.result_settings(
        CONFIG
    )


def test_the_noise_model_is_part_of_the_fingerprint(tmp_path, monkeypatch):
    """A pooled run records ``noise_model`` and so never matches a replicate-SE run."""
    settings = _patched_compute(tmp_path, monkeypatch, [])
    full = module._fingerprint(settings, _growth().provenance)
    assert module.NOISE_MODEL == "pooled"
    assert full["config"]["noise_model"] == "pooled"
    scientific = checkpoint_identity.scientific_fingerprint(full)
    assert scientific["config"]["noise_model"] == "pooled"
    assert "contrast_checkpoint" not in full
    # It is a constant, not a setting of config/analysis.toml.
    section = settings["analysis"][module.SECTION]
    assert "noise_model" not in section
    assert "noise_model" not in module.result_settings(section)

    # The same run as an older, replicate-SE fingerprint (no marker) differs, in the
    # science and so in the mirror directory suffix and the result dataset name.
    older = json.loads(json.dumps(scientific))
    del older["config"]["noise_model"]
    assert checkpoint_identity.differences(older, scientific) == ["config"]
    assert checkpoint_identity.digest8(older) != checkpoint_identity.digest8(scientific)
    # The wanted fingerprint (loader side) carries it too.
    assert (
        module._wanted_scientific(settings, warn=False)["config"]["noise_model"]
        == "pooled"
    )


def test_changed_growth_rates_recompute(tmp_path, monkeypatch):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    compute_bootstrap_smoothed_tree(settings)

    def changed(**_):
        growth = _growth()
        growth.provenance = {"fingerprint": {"generation": 2, "files": {}}}
        return growth

    monkeypatch.setattr(module, "load_cached_sequence_growth_rates", changed)
    compute_bootstrap_smoothed_tree(settings)  # same scientific run: reused
    assert builds == [1]
    compute_bootstrap_smoothed_tree(settings, force=True)
    assert builds == [1, 1]


def test_loading_without_a_checkpoint_or_with_a_pending_marker_fails(
    tmp_path, monkeypatch
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    with pytest.raises(FileNotFoundError, match="just bootstrap-smoothed-tree"):
        load_cached_bootstrap_smoothed_tree()
    compute_bootstrap_smoothed_tree(settings)
    _pending(module).write_text("pending")
    with pytest.raises(RuntimeError, match="pending"):
        load_cached_bootstrap_smoothed_tree()


# -- the stage on the produce-or-load helper (ADR 0013, Phase 2 part B) ---------


def _helper_record() -> dict:
    return json.loads((module._published_leaf() / PROVENANCE_NAME).read_text())


def test_the_checkpoint_is_published_by_the_helper_and_exported_to_the_slot(
    tmp_path, monkeypatch
):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    result = compute_bootstrap_smoothed_tree(settings)
    checkpoint, slot = module._published_leaf(), module.cache_dir()

    record = _helper_record()
    assert record["stage"] == "bootstrap-smoothed-tree"
    assert sorted(record["artefacts"]) == sorted(
        [*module.ARTIFACTS, module.RECORD_NAME]
    )
    assert sorted(p.name for p in checkpoint.iterdir()) == sorted(
        [*module.ARTIFACTS, module.RECORD_NAME, PROVENANCE_NAME]
    )
    # The slot keeps the readers' layout: the artefacts, and the stage's own record
    # as ``provenance.json``, byte for byte what the checkpoint holds.
    assert sorted(p.name for p in slot.iterdir()) == sorted(
        [*module.ARTIFACTS, "provenance.json"]
    )
    for name in module.ARTIFACTS:
        assert (slot / name).read_bytes() == (checkpoint / name).read_bytes()
    assert (slot / "provenance.json").read_bytes() == (
        checkpoint / module.RECORD_NAME
    ).read_bytes()
    assert result.provenance == json.loads((slot / "provenance.json").read_text())

    # The key is the run's scientific identity: no code, no input files declared.
    scientific = checkpoint_identity.scientific_fingerprint(
        result.provenance["fingerprint"]
    )
    assert record["declared"]["settings"] == {"scientific": scientific}
    declared_code = record["declared"]["code"]
    moved_names = {
        "bootstrap_smoothed_tree.py",
        "bootstrap_chunks.py",
        "checkpoint_identity.py",
        "selected_leaf_simplification.py",
    }
    assert moved_names <= {Path(name).name for name in declared_code}
    assert not any("/analysis/bootstrap_" in name for name in declared_code)
    assert record["declared"]["inputs"] == {}
    assert record["fingerprint"] == stage_fingerprint(
        module.bootstrap_stage(scientific)
    )
    assert not list(checkpoint.parent.glob(".*.stage-*"))


def test_the_published_key_ignores_code_and_scheduling_but_not_the_science(
    tmp_path, monkeypatch
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    compute_bootstrap_smoothed_tree(settings)
    key = _helper_record()["fingerprint"]

    # Code and lock-file digests are a label. They do not warn and do not redraw.
    monkeypatch.setattr(
        module, "package_source_hashes", lambda *a: {"topographer:x.py": "edited"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compute_bootstrap_smoothed_tree(settings)
    # Scheduling keys never reach the key.
    settings["analysis"][module.SECTION].update(n_jobs=3, chunk_size=5)
    compute_bootstrap_smoothed_tree(settings)
    assert builds == [1] and _helper_record()["fingerprint"] == key

    # A scientific setting does, and so does the noise model.
    monkeypatch.setattr(module, "NOISE_MODEL", "replicate")
    compute_bootstrap_smoothed_tree(settings)
    assert builds == [1, 1] and _helper_record()["fingerprint"] != key
    monkeypatch.setattr(module, "NOISE_MODEL", "pooled")
    compute_bootstrap_smoothed_tree(settings)
    # The first run's checkpoint was replaced by the second publish and nothing is
    # mirrored any more, so the pooled run is drawn again.
    assert builds == [1, 1, 1] and _helper_record()["fingerprint"] == key
    builds.clear()
    builds.append(1)
    settings["analysis"][module.SECTION]["seed"] = 8
    compute_bootstrap_smoothed_tree(settings)
    assert builds == [1, 1] and _helper_record()["fingerprint"] != key
    scientific = _helper_record()["declared"]["settings"]["scientific"]
    assert scientific["config"]["seed"] == 8
    assert "n_jobs" not in json.dumps(scientific)
    assert "chunk_size" not in json.dumps(scientific)


def test_the_stage_key_strips_scheduling_keys_at_any_depth():
    base = {"config": {"seed": 1}, "growth": {"config": {"elapsed_time": 3.5}}}
    noisy = {
        "config": {"seed": 1, "n_jobs": 8, "chunk_size": 250},
        "growth": {"config": {"elapsed_time": 3.5, "n_jobs": 2}},
    }
    assert stage_fingerprint(module.bootstrap_stage(base)) == stage_fingerprint(
        module.bootstrap_stage(noisy)
    )
    assert stage_fingerprint(module.bootstrap_stage(base)) != stage_fingerprint(
        module.bootstrap_stage({**base, "config": {"seed": 2}})
    )


def test_a_reused_checkpoint_repairs_a_missing_slot_without_a_mirror(
    tmp_path, monkeypatch
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    first = compute_bootstrap_smoothed_tree(settings)
    shutil.rmtree(module.cache_dir())  # the verified checkpoint is the source

    def refuse(*args, **kwargs):
        raise AssertionError("no processed checkpoint may be consulted")

    monkeypatch.setattr(checkpoint_identity, "iter_processed_checkpoints", refuse)
    again = compute_bootstrap_smoothed_tree(settings)
    assert builds == [1]
    assert again.provenance == first.provenance
    assert (module.cache_dir() / "provenance.json").is_file()


@pytest.mark.parametrize("failure", ["export", "record"])
def test_an_interrupted_export_never_leaves_a_record_that_claims_missing_files(
    tmp_path, monkeypatch, failure
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    compute_bootstrap_smoothed_tree(settings)  # run A is in the slot
    settings["analysis"][module.SECTION]["seed"] = 8  # run B replaces it

    real_replace = module.os.replace

    def flaky(source, target):
        if Path(target) == module.cache_dir() / "provenance.json":
            raise OSError("disk full")
        return real_replace(source, target)

    real_export = Checkpoint.export_to

    def broken(self, destination, names=None):
        if Path(destination) != module.cache_dir():  # only the slot export fails
            return real_export(self, destination, names)
        raise OSError("disk full")

    # A nested context undoes only these patches, never the fixtures' isolation.
    with monkeypatch.context() as failing:
        if failure == "export":
            failing.setattr(Checkpoint, "export_to", broken)
        else:
            failing.setattr(module.os, "replace", flaky)
        with pytest.raises(OSError, match="disk full"):
            compute_bootstrap_smoothed_tree(settings)

    # The checkpoint of run B is published; the slot has no record to believe.
    monkeypatch.setattr(module, "load_settings", lambda: settings)
    assert _helper_record()["declared"]["settings"]["scientific"]["config"]["seed"] == 8
    assert not (module.cache_dir() / "provenance.json").exists()
    assert not _pending(module).exists()
    # The loader finds no record to believe and nothing to restore from.
    with pytest.raises(FileNotFoundError, match="just bootstrap-smoothed-tree"):
        load_cached_bootstrap_smoothed_tree()

    # The next run finds the verified checkpoint and only repairs the slot.
    repaired = compute_bootstrap_smoothed_tree(settings)
    assert builds == [1, 1]
    assert repaired.summary["settings"]["seed"] == 8
    assert load_cached_bootstrap_smoothed_tree().summary == repaired.summary
    assert (module.cache_dir() / "provenance.json").read_bytes() == (
        module._published_leaf() / module.RECORD_NAME
    ).read_bytes()


def test_a_failed_run_publishes_nothing_and_a_rerun_completes(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])

    def die(*args, **kwargs):
        raise RuntimeError("a worker died")

    monkeypatch.setattr(bootstrap_chunks.ChunkedBootstrap, "run", die)
    with pytest.raises(RuntimeError, match="a worker died"):
        compute_bootstrap_smoothed_tree(settings)
    assert not (module.checkpoint_dir() / PROVENANCE_NAME).exists()
    assert not any(module.checkpoint_dir().glob("*/" + PROVENANCE_NAME))
    assert not module.cache_dir().exists()
    assert not _pending(module).exists()
    assert not list(module.checkpoint_dir().parent.glob(".*.stage-*"))

    monkeypatch.setattr(bootstrap_chunks.ChunkedBootstrap, "run", _echo)
    compute_bootstrap_smoothed_tree(settings)
    assert (module.cache_dir() / "provenance.json").is_file()
    assert _helper_record()["stage"] == "bootstrap-smoothed-tree"


def test_a_reuse_never_rewrites_the_slot_or_the_checkpoint(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)

    def state():
        return {
            str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for root in (module.checkpoint_dir(), module.cache_dir())
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    before = state()
    compute_bootstrap_smoothed_tree(settings)
    load_cached_bootstrap_smoothed_tree(settings)
    assert state() == before


def test_a_corrupt_slot_is_replaced_from_the_verified_checkpoint(tmp_path, monkeypatch):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    expected = compute_bootstrap_smoothed_tree(settings)
    (module.cache_dir() / "hills.parquet").write_bytes(b"corrupt")

    def refuse(*args, **kwargs):
        raise AssertionError("no processed checkpoint may be consulted")

    monkeypatch.setattr(checkpoint_identity, "iter_processed_checkpoints", refuse)
    with pytest.warns(RuntimeWarning, match="moved it aside") as record:
        restored = compute_bootstrap_smoothed_tree(settings)
    assert builds == [1]
    assert restored.summary == expected.summary
    (kept,) = module.cache_dir().parent.glob(f".{module.cache_dir().name}.replaced-*")
    assert (kept / "hills.parquet").read_bytes() == b"corrupt"
    assert str(kept) in " ".join(str(w.message) for w in record)
    assert (module.cache_dir() / "hills.parquet").read_bytes() == (
        module._published_leaf() / "hills.parquet"
    ).read_bytes()


def test_a_corrupt_slot_with_no_checkpoint_or_mirror_is_recomputed_and_kept_aside(
    tmp_path, monkeypatch
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    compute_bootstrap_smoothed_tree(settings)
    shutil.rmtree(module.checkpoint_dir())
    (module.cache_dir() / "hills.parquet").write_bytes(b"corrupt")
    with pytest.warns(RuntimeWarning) as record:
        compute_bootstrap_smoothed_tree(settings)
    messages = " ".join(str(w.message) for w in record)
    assert "cannot be read" in messages and "moved it aside" in messages
    assert builds == [1, 1]
    (kept,) = module.cache_dir().parent.glob(f".{module.cache_dir().name}.replaced-*")
    assert (kept / "hills.parquet").read_bytes() == b"corrupt"
    load_cached_bootstrap_smoothed_tree(settings)


@pytest.mark.parametrize("change", ["seed", "candidate_functional_threshold"])
def test_a_readable_slot_of_another_run_is_moved_aside_not_overwritten(
    tmp_path, monkeypatch, change
):
    """The settings or the scientific identity differ: the slot is kept, never lost."""
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    first = {
        p.name: p.read_bytes() for p in module.cache_dir().iterdir() if p.is_file()
    }
    if change == "seed":
        settings["analysis"][module.SECTION]["seed"] = 8
    else:
        settings["analysis"]["clamped"]["candidate_functional_threshold"] = {
            "rule": "median_reflected",
            "decimals": 3,
        }
    with pytest.warns(RuntimeWarning, match="another run.*moved it aside") as record:
        compute_bootstrap_smoothed_tree(settings)
    (kept,) = module.cache_dir().parent.glob(f".{module.cache_dir().name}.replaced-*")
    assert {p.name: p.read_bytes() for p in kept.iterdir()} == first
    assert str(kept) in " ".join(str(w.message) for w in record)
    loaded = load_cached_bootstrap_smoothed_tree(settings)
    assert loaded.summary["settings"]["seed"] == (8 if change == "seed" else 7)
    assert (module.cache_dir() / "provenance.json").read_bytes() != first[
        "provenance.json"
    ]


def test_a_forced_rerun_of_the_same_run_overwrites_its_slot_in_place(
    tmp_path, monkeypatch
):
    """Only the full fingerprint differs (a code digest): same run, no aside."""
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    before = (module.cache_dir() / "provenance.json").read_bytes()
    monkeypatch.setattr(
        module, "package_source_hashes", lambda *a: {"topographer:x.py": "edited"}
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compute_bootstrap_smoothed_tree(settings, force=True)
    assert "moved it aside" not in " ".join(str(w.message) for w in record)
    assert (module.cache_dir() / "provenance.json").read_bytes() != before
    assert not list(
        module.cache_dir().parent.glob(f".{module.cache_dir().name}.replaced-*")
    )


def test_a_corrupt_published_checkpoint_is_quarantined_and_rebuilt_from_the_slot(
    tmp_path, monkeypatch
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    expected = compute_bootstrap_smoothed_tree(settings)
    (module._published_leaf() / "hills.parquet").write_bytes(b"corrupt")
    with pytest.warns(RuntimeWarning, match="unusable"):
        again = compute_bootstrap_smoothed_tree(settings)
    assert builds == [1]  # copied from the readable slot, not drawn again
    assert again.summary == expected.summary
    assert _helper_record()["stage"] == "bootstrap-smoothed-tree"
    assert list(module.checkpoint_dir().parent.glob(".*.unusable-*"))


def test_tampered_checkpoint_artifact_is_rejected(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    result, _ = _build(monkeypatch)
    module._write_checkpoint(result, stage, _checkpoint_fingerprint())
    module._read_checkpoint(stage)
    (stage / "hills.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        module._read_checkpoint(stage)


def test_checkpoint_refuses_a_sanity_check_that_disagrees_with_the_draws(
    tmp_path, monkeypatch
):
    result, _ = _build(monkeypatch)
    verdict = result.summary["diagnostics"]["global_verdict"]
    good = dict(verdict["sanity_check"])
    for name, bad in (
        ("missing", None),
        ("moved", {**good, "real_share_below_flat_bottleneck_p5": 0.123}),
    ):
        stage = tmp_path / name
        stage.mkdir()
        if bad is None:
            del verdict["sanity_check"]
        else:
            verdict["sanity_check"] = bad
        module._write_checkpoint(result, stage, _checkpoint_fingerprint())
        with pytest.raises(ValueError, match="coverage mismatch"):
            module._read_checkpoint(stage)
    verdict["sanity_check"] = good
    stage = tmp_path / "good"
    stage.mkdir()
    module._write_checkpoint(result, stage, _checkpoint_fingerprint())
    module._read_checkpoint(stage)


def test_checkpoint_refuses_band_columns_that_disagree_with_each_other(
    tmp_path, monkeypatch
):
    result, _ = _build(monkeypatch)
    hills = result.hills
    summary = result.summary
    assert hills.height, "the toy build must have hills"
    good = (hills, summary["band_simplification"]["peaks_after"])

    def flipped(frame):
        outcome = frame["band_outcome"].to_list()
        outcome[0] = "absorbed" if outcome[0] == "kept" else "kept"
        return frame.with_columns(pl.Series("band_outcome", outcome, dtype=pl.String))

    cases = {
        "hills_band_outcome": lambda: setattr(result, "hills", flipped(hills)),
        "peaks_after": lambda: summary["band_simplification"].update(
            peaks_after=good[1] + 1
        ),
    }
    for name, tamper in cases.items():
        result.hills = good[0]
        summary["band_simplification"]["peaks_after"] = good[1]
        tamper()
        stage = tmp_path / name
        stage.mkdir()
        module._write_checkpoint(result, stage, _checkpoint_fingerprint())
        with pytest.raises(ValueError, match="coverage mismatch"):
            module._read_checkpoint(stage)
    result.hills = good[0]
    summary["band_simplification"]["peaks_after"] = good[1]
    stage = tmp_path / "good"
    stage.mkdir()
    module._write_checkpoint(result, stage, _checkpoint_fingerprint())
    module._read_checkpoint(stage)


def test_stale_growth_checkpoint_is_refused(monkeypatch):
    growth = _growth()
    growth.provenance = {
        "fingerprint": {"files": {"a.py": "x", "barseq_export/counts.parquet": "y"}}
    }
    monkeypatch.setattr(
        module, "_growth_fingerprint_without_export", lambda s: {"files": {"a.py": "x"}}
    )
    module._check_growth_current({}, growth)
    monkeypatch.setattr(
        module, "_growth_fingerprint_without_export", lambda s: {"files": {"a.py": "z"}}
    )
    with pytest.raises(ValueError, match="stale growth-rate checkpoint"):
        module._check_growth_current({}, growth)


def test_table_export_writes_band_tables_and_a_sidecar(tmp_path, monkeypatch):
    from dubanevics2026topological import build

    hills = pl.DataFrame(
        {
            "peak_sequence": ["a"],
            "member_count": [1],
            "hill_support": [0.8],
            "nominal_birth": [1.0],
            "nominal_death": [0.0],
            "nominal_persistence": [1.0],
            "clears_band": [True],
            "band_outcome": ["kept"],
        }
    )
    result = SimpleNamespace(
        hills=hills,
        draw_stats=pl.DataFrame({"draw": list(range(40))}),
        provenance={"fingerprint": {"generation": 1}, "artifacts": {}},
    )
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    written = build.build_bootstrap_smoothed_tree_tables(result)

    assert sorted(p.name for p in written) == [
        "papkou2023_bootstrap_draw_stats.csv",
        "papkou2023_bootstrap_draw_stats.parquet",
        "papkou2023_bootstrap_hills.csv",
        "papkou2023_bootstrap_hills.parquet",
    ]
    tables = tmp_path / "tables"
    hills = pl.read_parquet(tables / "papkou2023_bootstrap_hills.parquet")
    assert hills.equals(result.hills)
    assert (
        pl.read_parquet(tables / "papkou2023_bootstrap_draw_stats.parquet").height == 40
    )
    sidecar = json.loads(
        (tables / ".bootstrap-smoothed-tree-provenance.json").read_text()
    )
    assert sidecar["stage"] == "bootstrap-smoothed-tree-tables"
    assert set(sidecar["artefacts"]) == {p.name for p in written}

    # Output bytes equal the checkpoint's; a deleted table is restored, not rebuilt.
    step = tmp_path / "checkpoints/bootstrap_smoothed_tree_tables"
    matches = [
        path.parent
        for path in step.glob("*/provenance.json")
        if json.loads(path.read_text())["fingerprint"] == sidecar["fingerprint"]
    ]
    assert len(matches) == 1
    checkpoint = matches[0]
    for path in written:
        assert path.read_bytes() == (checkpoint / path.name).read_bytes()
    before = {p: p.read_bytes() for p in written}
    written[0].unlink()
    again = build.build_bootstrap_smoothed_tree_tables(result)
    assert {p: p.read_bytes() for p in again} == before
    recorded = json.loads(
        (tables / ".bootstrap-smoothed-tree-provenance.json").read_text()
    )
    assert recorded["created_at"] == sidecar["created_at"]

    # A new bootstrap checkpoint fingerprint changes the stage fingerprint.
    result.provenance = {"fingerprint": {"generation": 2}, "artifacts": {}}
    build.build_bootstrap_smoothed_tree_tables(result)
    changed = json.loads(
        (tables / ".bootstrap-smoothed-tree-provenance.json").read_text()
    )
    assert changed["fingerprint"] != sidecar["fingerprint"]

    # Same fingerprint but different artefact hashes is a different checkpoint.
    result.provenance = {"fingerprint": {"generation": 2}, "artifacts": {"a": "1"}}
    build.build_bootstrap_smoothed_tree_tables(result)
    rehashed = json.loads(
        (tables / ".bootstrap-smoothed-tree-provenance.json").read_text()
    )
    assert rehashed["fingerprint"] != changed["fingerprint"]


# -- entry point and registration --------------------------------------------


def test_build_entry_point_is_guarded_against_spawned_worker_reimport():
    """``n_jobs > 1`` spawns workers that re-import the entry module as a script."""
    from dubanevics2026topological import build

    source = (Path(build.__file__).parent / "__main__.py").read_text()
    tree = ast.parse(source)
    calls_main = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "main"
    ]
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    ]
    assert len(calls_main) == 1
    assert guards
    assert any(call in list(ast.walk(guard)) for guard in guards for call in calls_main)


def test_stage_is_a_build_command_outside_all_figures_and_tables(monkeypatch, capsys):
    from dubanevics2026topological import build

    justfile = (Path(module.ROOT) / "justfile").read_text()
    assert "bootstrap-smoothed-tree: selection" in justfile
    assert "comembership" not in justfile
    for recipe in ("all", "figures", "tables"):
        header = next(
            line for line in justfile.splitlines() if line.startswith(f"{recipe}:")
        )
        assert "bootstrap-smoothed-tree" not in header

    fake, _ = _build(monkeypatch)
    fake.provenance = {}
    forced = []
    monkeypatch.setattr(
        module,
        "compute_bootstrap_smoothed_tree",
        lambda settings, force, fresh, context=None: (
            forced.append((force, fresh)) or fake
        ),
    )
    table_forces = []
    monkeypatch.setattr(
        build,
        "build_bootstrap_smoothed_tree_tables",
        lambda r, force=False, context=None: table_forces.append(force) or [],
    )
    assert build.main(["bootstrap-smoothed-tree", "--force"]) == 0
    assert build.main(["bootstrap-smoothed-tree", "--fresh"]) == 0
    assert forced == [(True, False), (False, True)]
    assert table_forces == [True, False]
    assert "Bootstrap-smoothed tree ready" in capsys.readouterr().out


def test_committed_settings_are_valid():
    from dubanevics2026topological import build

    settings = build.load_settings()["analysis"][module.SECTION]
    assert module._validate_config(settings) == {
        "n_replicates": 10000,
        "seed": 20261019,
        "band_quantile": 0.95,
        "n_jobs": 8,
        "chunk_size": 250,
    }
    assert set(settings) == {
        "n_replicates",
        "seed",
        "band_quantile",
        "n_jobs",
        "chunk_size",
    }


# -- the git stamp stays out of the run's identity (ADR 0013, D3, Phase 2) -------

GIT_STAMP_A = {"commit": "a" * 40, "dirty": False}


def _stamped_as(monkeypatch, stamp):
    """Make the helper stamp every record it writes with ``stamp``."""
    monkeypatch.setattr(git_stamp_module, "git_stamp", lambda root: dict(stamp))


def _published_state(tmp_path: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for root in (
            module.checkpoint_dir(),
            module.cache_dir(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    "later",
    [
        {"commit": "b" * 40, "dirty": False},
        {"commit": "a" * 40, "dirty": True},
        {"commit": "b" * 40, "dirty": True},
        {"commit": None, "dirty": None},
    ],
    ids=["new-commit", "dirty-flip", "commit-and-dirty", "no-git"],
)
def test_a_git_change_reuses_without_a_redraw_a_warning_or_a_rewrite(
    tmp_path, monkeypatch, later
):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    _stamped_as(monkeypatch, GIT_STAMP_A)
    first = compute_bootstrap_smoothed_tree(settings)
    assert _helper_record()["git"] == GIT_STAMP_A  # the helper stamped the run
    key = _helper_record()["fingerprint"]
    before = _published_state(tmp_path)

    _stamped_as(monkeypatch, later)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a stale-reuse warning would be a failure
        again = compute_bootstrap_smoothed_tree(settings)
        restored = load_cached_bootstrap_smoothed_tree(settings)

    assert builds == [1]  # nothing was drawn, nothing re-classified
    assert again.summary == restored.summary == first.summary
    assert _helper_record()["fingerprint"] == key
    # A reuse never re-stamps: the checkpoint keeps the stamp of the run that made
    # it, and the slot and the checkpoint keep their bytes.
    assert _helper_record()["git"] == GIT_STAMP_A
    assert _published_state(tmp_path) == before


def test_the_stages_own_record_holds_no_stamp(tmp_path, monkeypatch):
    """The helper's record is stamped; what the slot holds is not.

    The slot's ``provenance.json`` is the stage's own record, exported byte for byte;
    stamping it would change a product that is meant to be byte-stable.
    """
    settings = _patched_compute(tmp_path, monkeypatch, [])
    _stamped_as(monkeypatch, GIT_STAMP_A)
    compute_bootstrap_smoothed_tree(settings)

    assert "git" in _helper_record()
    stage_record = module._published_leaf() / module.RECORD_NAME
    for path in (stage_record, module.cache_dir() / "provenance.json"):
        assert "git" not in json.loads(path.read_text(encoding="utf-8"))
    assert (module.cache_dir() / "provenance.json").read_bytes() == stage_record.read_bytes()


def test_a_stamp_on_the_growth_record_changes_no_identity(tmp_path, monkeypatch):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    first = compute_bootstrap_smoothed_tree(settings)
    key = _helper_record()["fingerprint"]
    plain = _growth()
    stamped_growth = dataclasses.replace(
        plain, provenance={**plain.provenance, "git": GIT_STAMP_A}
    )
    full = [
        module._fingerprint(settings, growth.provenance)
        for growth in (plain, stamped_growth)
    ]
    assert full[0] == full[1]
    assert checkpoint_identity.scientific_fingerprint(
        full[0]
    ) == checkpoint_identity.scientific_fingerprint(full[1])

    monkeypatch.setattr(
        module, "load_cached_sequence_growth_rates", lambda: stamped_growth
    )
    before = _published_state(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = compute_bootstrap_smoothed_tree(settings)

    assert builds == [1] and again.summary == first.summary
    assert _helper_record()["fingerprint"] == key
    assert _published_state(tmp_path) == before


# -- FITNESSTREE_N_JOBS ---------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_n_jobs_override_is_off_when_unset_or_blank(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(module.ENV_N_JOBS, value)
    assert module.n_jobs_override() is None
    assert module._validate_config({**CONFIG, "n_jobs": 3})["n_jobs"] == 3


def test_n_jobs_override_reads_the_given_environment_not_the_process_one(monkeypatch):
    monkeypatch.setenv(module.ENV_N_JOBS, "4")
    assert module.n_jobs_override({module.ENV_N_JOBS: "9"}) == 9
    assert module.n_jobs_override({}) is None


@pytest.mark.parametrize(("value", "expected"), [("6", 6), (" 12 ", 12), ("007", 7)])
def test_n_jobs_override_replaces_the_configured_worker_count(
    monkeypatch, value, expected
):
    monkeypatch.setenv(module.ENV_N_JOBS, value)
    assert module.n_jobs_override() == expected
    settings = module._validate_config({**CONFIG, "n_jobs": 2})
    assert settings["n_jobs"] == expected
    # The other validated settings are exactly what the config alone gives.
    monkeypatch.delenv(module.ENV_N_JOBS)
    plain = module._validate_config({**CONFIG, "n_jobs": 2})
    assert plain["n_jobs"] == 2
    assert {k: v for k, v in settings.items() if k != "n_jobs"} == {
        k: v for k, v in plain.items() if k != "n_jobs"
    }


@pytest.mark.parametrize(
    "value", ["0", "-1", "abc", "1.5", "2e1", "1_0", "+3", "true", "\u0663"]
)
def test_n_jobs_override_must_be_a_positive_integer(monkeypatch, value):
    monkeypatch.setenv(module.ENV_N_JOBS, value)
    with pytest.raises(ValueError, match=module.ENV_N_JOBS):
        module.n_jobs_override()
    with pytest.raises(ValueError, match=module.ENV_N_JOBS):
        module._validate_config(CONFIG)


def test_n_jobs_override_leaves_result_settings_and_the_config_check_alone(
    monkeypatch,
):
    expected = module.result_settings({**CONFIG, "n_jobs": 2})
    monkeypatch.setenv(module.ENV_N_JOBS, "9")
    assert module.result_settings({**CONFIG, "n_jobs": 2}) == expected
    assert "n_jobs" not in expected
    # A bad configured value is still refused: the override does not excuse it.
    with pytest.raises(ValueError, match="positive integers"):
        module._validate_config({**CONFIG, "n_jobs": 0})
    # And n_jobs stays a required key.
    with pytest.raises(KeyError):
        module._validate_config({k: v for k, v in CONFIG.items() if k != "n_jobs"})


def test_n_jobs_override_reaches_build_runner_and_summary(monkeypatch):
    monkeypatch.setenv(module.ENV_N_JOBS, "7")
    result, calls = _build(monkeypatch, config={**CONFIG, "n_jobs": 3, "chunk_size": 5})
    assert [call["runner"].n_jobs for _, call in calls] == [7, 7]
    assert result.summary["timing"]["n_jobs"] == 7


def test_n_jobs_override_reaches_the_draws_of_a_computed_stage(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    settings["analysis"][module.SECTION]["n_jobs"] = 3
    runners: list[int] = []

    def record(self, name, network, statistic, *, label=None):
        runners.append(self.n_jobs)
        return _echo(self, name, network, statistic)

    monkeypatch.setattr(bootstrap_chunks.ChunkedBootstrap, "run", record)
    monkeypatch.setenv(module.ENV_N_JOBS, "5")
    result = compute_bootstrap_smoothed_tree(settings)

    assert runners == [5, 5]  # the real series and the flat one
    assert result.summary["timing"]["n_jobs"] == 5
    assert (
        json.loads((module.cache_dir() / "summary.json").read_text())["timing"][
            "n_jobs"
        ]
        == 5
    )


def test_n_jobs_override_does_not_change_the_fingerprint(tmp_path, monkeypatch):
    builds: list = []
    settings = _patched_compute(tmp_path, monkeypatch, builds)
    first = compute_bootstrap_smoothed_tree(settings)
    key = _helper_record()["fingerprint"]
    monkeypatch.setenv(module.ENV_N_JOBS, "5")
    again = compute_bootstrap_smoothed_tree(settings)
    assert builds == [1]  # reused: the same key, nothing was produced again
    assert _helper_record()["fingerprint"] == key
    assert again.provenance == first.provenance


# -- processed bootstrap checkpoints are fingerprint-matched candidates of the loader ------

STATE = "reflected_by"


def _as_processed(tmp_path: Path, monkeypatch) -> Path:
    """Move the working slot of a tiny run into ``processed/bootstrap/<state>/checkpoint``.

    The slot has the layout of a processed checkpoint (the stage record as
    ``provenance.json``); the slot and the published checkpoint are removed so the
    processed folder is the only source left.
    """
    from dubanevics2026topological.results import processed

    root = tmp_path / "processed"
    monkeypatch.setattr(processed, "PROCESSED_ROOT", root)
    target = root / "bootstrap" / STATE / "checkpoint"
    shutil.copytree(module.cache_dir(), target)
    shutil.rmtree(module.cache_dir())
    shutil.rmtree(module.checkpoint_dir())
    return target


def test_the_loader_restores_a_processed_checkpoint_into_the_slot(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    first = compute_bootstrap_smoothed_tree(settings)
    source = _as_processed(tmp_path, monkeypatch)

    restored = load_cached_bootstrap_smoothed_tree(settings)

    assert restored.summary == first.summary
    assert sorted(
        p.name for p in module.cache_dir().iterdir() if not p.name.startswith(".")
    ) == sorted([*module.ARTIFACTS, "provenance.json"])
    assert (module.cache_dir() / "provenance.json").read_bytes() == (
        source / "provenance.json"
    ).read_bytes()
    assert module.restored_source() == STATE


def test_a_slot_reuse_keeps_the_restore_source_stable(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    _as_processed(tmp_path, monkeypatch)
    load_cached_bootstrap_smoothed_tree(settings)
    again = load_cached_bootstrap_smoothed_tree(settings)  # reuses the slot
    assert again is not None
    assert module.restored_source() == STATE


def test_a_computed_slot_has_no_restore_source(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    assert module.restored_source() is None  # computed here, not restored


def test_the_restore_source_is_dropped_when_the_slot_changes(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    _as_processed(tmp_path, monkeypatch)
    load_cached_bootstrap_smoothed_tree(settings)
    record = module.cache_dir() / "provenance.json"
    record.write_text(record.read_text() + " ")
    assert module.restored_source() is None
    (module.cache_dir() / module.RESTORE_MARKER).write_text("not json")
    assert module.restored_source() is None


def test_a_processed_checkpoint_of_another_fingerprint_is_not_used(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    source = _as_processed(tmp_path, monkeypatch)
    record = json.loads((source / "provenance.json").read_text())
    record["fingerprint"]["files"] = {"data/raw/other.parquet": "changed"}
    (source / "provenance.json").write_text(json.dumps(record))
    with (
        pytest.warns(RuntimeWarning, match="scientific fingerprint"),
        pytest.raises(FileNotFoundError, match="just all.*just fetch-data"),
    ):
        load_cached_bootstrap_smoothed_tree(settings)


def test_a_corrupt_processed_checkpoint_is_refused_not_redrawn(tmp_path, monkeypatch):
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    source = _as_processed(tmp_path, monkeypatch)
    (source / "hills.parquet").write_bytes(b"damaged")
    with (
        pytest.warns(RuntimeWarning, match="failed validation"),
        pytest.raises(RuntimeError, match="do not rerun the bootstrap"),
    ):
        load_cached_bootstrap_smoothed_tree(settings)


def _by_context(tmp_path: Path, variant: str):
    from dubanevics2026topological.results import RunContext

    return RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        processed_root=tmp_path / "processed",
        variant_name=variant,
        env={},
    )


def _published_checkpoint(tmp_path, monkeypatch):
    """Run a tiny stage and return its settings and the verified ``Checkpoint`` it published."""
    seen = []
    real = module._publish_processed
    monkeypatch.setattr(
        module,
        "_publish_processed",
        lambda ready, settings, context=None: seen.append(ready) or real(ready, settings, context),
    )
    settings = _patched_compute(tmp_path, monkeypatch, [])
    compute_bootstrap_smoothed_tree(settings)
    return settings, seen[0]


def test_a_by_run_publishes_its_draw_evidence_to_processed(tmp_path, monkeypatch):
    from dubanevics2026topological.results import processed
    from dubanevics2026topological.simplification import checkpoint_identity

    settings, ready = _published_checkpoint(tmp_path, monkeypatch)
    context = _by_context(tmp_path, "threshold=published_fdr=by_q=0.05")
    module._publish_processed(ready, settings, context)

    published = tmp_path / "processed" / "bootstrap" / "published_by"
    assert (published / "checkpoint" / "provenance.json").is_file()
    assert (published / "checkpoint" / "summary.json").is_file()
    loaded = processed.load("bootstrap", "published_by", context)
    assert loaded.report.status == "full" and loaded.report.byte_integrity
    identity = loaded.manifest.scientific_identity
    assert (identity["thr"], identity["fdr"], identity["q"]) == ("published", "by", 0.05)
    assert identity["draws"] == 20 and identity["seed"] == 7
    assert "checkpoint/summary.json" in loaded.manifest.artifacts
    # The processed checkpoint is what a later run restores from.
    summary = json.loads((published / "checkpoint" / "summary.json").read_text())
    found = list(
        checkpoint_identity.iter_processed_checkpoints(
            summary["settings"], None, tmp_path / "processed"
        )
    )
    assert found == [published / "checkpoint"]


def test_a_point_estimate_run_publishes_no_bootstrap(tmp_path, monkeypatch):
    settings, ready = _published_checkpoint(tmp_path, monkeypatch)
    context = _by_context(tmp_path, "threshold=published_fdr=none")
    assert module._publish_processed(ready, settings, context) is None
    assert not (tmp_path / "processed" / "bootstrap").exists()
