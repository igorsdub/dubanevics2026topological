"""Threshold diagnostics product round-trips and matches the old figure computation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from test_full_tree_persistence_figure import _payload as _figure_payload

from dubanevics2026topological.analysis.component_persistence import DIAGNOSTICS_SCHEMA
from dubanevics2026topological.build.threshold_diagnostics_product import (
    build_threshold_diagnostics_product,
)
from dubanevics2026topological.results import MissingProductError, RunContext, load


def _synthetic_full_fitness() -> pl.DataFrame:
    return pl.DataFrame(
        {"dna_sequence": ["a", "b", "c"], "raw_fitness": [-1.0, 0.0, 1.0]},
        schema={"dna_sequence": pl.Utf8, "raw_fitness": pl.Float64},
    )


def _synthetic_pairs() -> pl.DataFrame:
    # Reuse the complete, valid pairs table from the figure test helper (which
    # other tests import helpers from sibling test modules). It has the required
    # columns and at least one ordinary + one relative pair.
    return _figure_payload()["pairs"]


def _synthetic_declaration() -> dict:
    return {"rule": "median_reflected", "decimals": 6}


def _install_synthetic_for_threshold(
    context: RunContext, monkeypatch=None, declaration: dict | None = None
) -> None:
    """Shared setup for component checkpoint and dummy raw files for hashing.

    Writes small placeholder parquets under context.root so the product's
    _file_sha256 (on configured dataset paths) succeeds even when real data/raw
    is absent. If monkeypatch is supplied, installs default load_settings
    (from real, overriding only dataset paths + candidate decl) and patches
    load_full + the component paths (PAIRS, DIAGNOSTICS) on tdp.
    """
    comp_dir = context.output_root / "analysis"
    comp_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = comp_dir / "component_pairs.parquet"
    pairs = _synthetic_pairs()
    pairs.write_parquet(pairs_path)

    # Write minimal valid component diagnostics (schema + pair_count to satisfy
    # load_cached_component_persistence checks) and provenance (with shas of the
    # written files) so every check in load_cached... (141-200) passes.
    diags_content = {"schema": DIAGNOSTICS_SCHEMA, "pair_count": pairs.height}
    diags_path = comp_dir / "component_diagnostics.json"
    diags_path.write_text(json.dumps(diags_content), encoding="utf-8")

    def _sha(p: Path) -> str:
        digest = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    pairs_sha = _sha(pairs_path)
    diags_sha = _sha(diags_path)
    prov_content = {
        "schema": 1,
        "fingerprint": "fp",
        "topographer_revision": "synthetic-test",
        "pair_count": pairs.height,
        "pairs_sha256": pairs_sha,
        "diagnostics_sha256": diags_sha,
    }
    prov_path = comp_dir / "component_provenance.json"
    prov_path.write_text(json.dumps(prov_content), encoding="utf-8")

    # Write placeholders at controlled paths (under context.root) so stage hashing
    # is strict but never sees missing files in test/CI/HPC environments.
    dummy_raw = context.root / "data" / "raw"
    dummy_raw.mkdir(parents=True, exist_ok=True)
    wt_dummy = dummy_raw / "dummy_fitness_wt.parquet"
    diff_dummy = dummy_raw / "dummy_fitness_diff.parquet"
    for p in (wt_dummy, diff_dummy):
        if not p.exists():
            pl.DataFrame({"col": [0.0]}).write_parquet(p)

    if monkeypatch is not None:
        import copy

        import dubanevics2026topological.build.threshold_diagnostics_product as tdp
        from dubanevics2026topological.analysis import (
            load_settings as real_load_settings,
        )

        base = real_load_settings(context=context)
        decl = declaration or {"rule": "median_reflected", "decimals": 6}

        def _default_ls(context=None):
            s = copy.deepcopy(base)
            analysis = s.setdefault("analysis", {})
            dataset = analysis.setdefault("dataset", {})
            dataset["fitness_wt"] = str(wt_dummy)
            dataset["fitness_diff"] = str(diff_dummy)
            # keep all real keys including landscape; override only paths + decl
            clamped = analysis.setdefault("clamped", {})
            clamped["candidate_functional_threshold"] = decl
            return s

        monkeypatch.setattr(tdp, "load_settings", _default_ls)
        monkeypatch.setattr(
            tdp,
            "load_full_network_fitness",
            lambda settings=None: _synthetic_full_fitness(),
        )


def test_threshold_diagnostics_product_round_trip_and_equals_old(tmp_path, monkeypatch):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch)

    path = build_threshold_diagnostics_product(context=context)
    assert path.exists()

    prod = load("threshold_diagnostics", path.name, context=context)
    assert prod.report.status == "full"
    assert prod.report.byte_integrity

    saved = json.loads(
        prod.artifacts["threshold_diagnostics.json"].read_text(encoding="utf-8")
    )
    assert "reflected" in saved  # median_reflected variant saves reflected value

    # equality to old computation: reproduce the exact from_summary + or + compute
    # logic and order that the stage used at extraction time (step 1)
    values = _synthetic_full_fitness()["raw_fitness"].to_list()
    decl = _synthetic_declaration()
    settings_for_from = {
        "analysis": {"clamped": {"candidate_functional_threshold": decl}}
    }
    from dubanevics2026topological.analysis.leading_saddle_band import tree_saddles
    from dubanevics2026topological.landscape.selection import (
        candidate_threshold_from_summary,
    )
    from dubanevics2026topological.threshold.functional_threshold import (
        comparison_declaration,
        compute_candidate_functional_threshold,
    )

    candidate = candidate_threshold_from_summary(settings_for_from, raw_fitness=values)
    computed = candidate or compute_candidate_functional_threshold(values, decl)
    reflected = compute_candidate_functional_threshold(
        values, comparison_declaration(decl)
    )
    pairs = _synthetic_pairs()
    saddles = tree_saddles(pairs)
    expected = {
        "candidate": {
            "rule": computed.rule,
            "decimals": computed.decimals,
            "median": computed.median,
            "global_minimum": computed.global_minimum,
            "median_gap": computed.median_gap,
            "unrounded": computed.unrounded,
            "threshold": computed.threshold,
        },
        "reflected": {
            "rule": reflected.rule,
            "decimals": reflected.decimals,
            "median": reflected.median,
            "global_minimum": reflected.global_minimum,
            "median_gap": reflected.median_gap,
            "unrounded": reflected.unrounded,
            "threshold": reflected.threshold,
        },
        "lowest_split_saddle_death": None
        if saddles.split_trailing is None
        else saddles.split_trailing.fitness,
        "highest_join_saddle_death": None
        if saddles.join_trailing is None
        else saddles.join_trailing.fitness,
    }
    assert saved == expected


def _patch_fdr_table(monkeypatch, sha="f" * 64):
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp

    monkeypatch.setattr(tdp, "_table_sha256", lambda declaration: sha)


@pytest.mark.parametrize("rule", ["mixture_quantile", "fdr_fixed_cutoff"])
def test_producer_and_loader_share_declaration_identity(tmp_path, monkeypatch, rule):
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp
    from dubanevics2026topological.analysis import load_settings
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_threshold_diagnostics_product,
    )
    from dubanevics2026topological.landscape import selection
    from dubanevics2026topological.threshold.functional_threshold import (
        declaration_identity,
    )

    context = RunContext(root=tmp_path, output_root=tmp_path / "output",
                         env={})
    declaration = (load_settings()["analysis"]["clamped"]["candidate_functional_threshold"]
                   if rule == "fdr_fixed_cutoff" else {"rule": rule, "decimals": 6})
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch, declaration=declaration)
    if rule == "fdr_fixed_cutoff":
        def summary(_declaration):
            return {"threshold": {"unrounded": 0.42, "rounded": 0.42}}

        monkeypatch.setattr(tdp, "threshold_summary", summary)
        monkeypatch.setattr(selection, "threshold_summary", summary)
        _patch_fdr_table(monkeypatch)
    else:
        _install_mixture_summary(monkeypatch)

    path = build_threshold_diagnostics_product(context=context)
    product = load("threshold_diagnostics", path.name, context=context)
    assert product.manifest.scientific_identity["candidate_functional_threshold_declaration"] == declaration_identity(declaration)
    _, loaded_id = load_threshold_diagnostics_product(context, settings=tdp.load_settings(context=context))
    assert loaded_id == product.manifest.scientific_id_sha256


def test_threshold_diagnostics_product_missing_raises(tmp_path, monkeypatch):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch)
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp

    with pytest.raises(MissingProductError, match="threshold-diagnostics-product"):
        from dubanevics2026topological.build.saved_figure_inputs import (
            load_threshold_diagnostics_product,
        )

        load_threshold_diagnostics_product(
            context, settings=tdp.load_settings(context=context)
        )


def test_threshold_diagnostics_product_changed_input_new_identity(
    tmp_path, monkeypatch
):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch)

    first = build_threshold_diagnostics_product(context=context)
    first_prod = load("threshold_diagnostics", first.name, context=context)
    first_id = first_prod.manifest.scientific_id_sha256

    # change a byte in a hashed input (a raw fitness dummy parquet)
    wt_path = context.root / "data" / "raw" / "dummy_fitness_wt.parquet"
    pl.DataFrame({"col": [99.0]}).write_parquet(wt_path)

    second = build_threshold_diagnostics_product(context=context)
    second_prod = load("threshold_diagnostics", second.name, context=context)
    assert second_prod.manifest.scientific_id_sha256 != first_id


def test_threshold_diagnostics_product_reads_the_context_component_checkpoint(
    tmp_path, monkeypatch
):
    """The component checkpoint follows the context's output root, so no refusal."""
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "other",
        env={},
    )
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch)
    path = build_threshold_diagnostics_product(context=context)
    assert path.is_file() or path.is_dir()


def _install_mixture_summary(monkeypatch, unrounded: float = 0.42):
    """Patch so candidate_threshold_from_summary for non-median succeeds in tests."""
    import dubanevics2026topological.threshold.functional_classification as fc_mod

    class _Fake:
        def __init__(self) -> None:
            self.summary = {
                "threshold": {"unrounded": unrounded, "rounded": unrounded},
            }

    def _fake_load(settings, tables=False, lock=True, context=None):
        return _Fake(), None

    # Use module + name form.
    monkeypatch.setattr(
        fc_mod, "load_cached_functional_classification_with_source", _fake_load
    )


def test_threshold_diagnostics_product_non_median_round_trip_and_equals_old(
    tmp_path, monkeypatch
):
    """Non-median (e.g. mixture) also round-trips and matches from_summary path."""
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    mixture_decl = {"rule": "mixture_quantile", "decimals": 3}
    _install_synthetic_for_threshold(
        context, monkeypatch=monkeypatch, declaration=mixture_decl
    )
    _install_mixture_summary(monkeypatch)

    path = build_threshold_diagnostics_product(context=context)
    prod = load("threshold_diagnostics", path.name, context=context)
    saved = json.loads(
        prod.artifacts["threshold_diagnostics.json"].read_text(encoding="utf-8")
    )
    assert "reflected" not in saved  # mixture variant saves no reflected value

    values = _synthetic_full_fitness()["raw_fitness"].to_list()
    from dubanevics2026topological.analysis import load_settings as real_load_settings
    from dubanevics2026topological.landscape.selection import (
        candidate_threshold_from_summary,
    )
    from dubanevics2026topological.threshold.functional_threshold import (
        comparison_declaration,
        compute_candidate_functional_threshold,
    )

    base = real_load_settings(context=context)
    settings_for_from = dict(base)
    settings_for_from["analysis"] = dict(settings_for_from.get("analysis", {}))
    settings_for_from["analysis"]["clamped"] = {
        "candidate_functional_threshold": mixture_decl
    }

    candidate = candidate_threshold_from_summary(settings_for_from, raw_fitness=values)
    computed = candidate or compute_candidate_functional_threshold(values, mixture_decl)
    rule_name = mixture_decl["rule"]
    if rule_name == "median_reflected":
        reflected = compute_candidate_functional_threshold(
            values, comparison_declaration(mixture_decl)
        )
    else:
        reflected = None
    pairs = _synthetic_pairs()
    from dubanevics2026topological.analysis.leading_saddle_band import tree_saddles

    saddles = tree_saddles(pairs)
    expected = {
        "candidate": {
            "rule": computed.rule,
            "decimals": computed.decimals,
            "median": computed.median,
            "global_minimum": computed.global_minimum,
            "median_gap": computed.median_gap,
            "unrounded": computed.unrounded,
            "threshold": computed.threshold,
        },
        "lowest_split_saddle_death": None
        if saddles.split_trailing is None
        else saddles.split_trailing.fitness,
        "highest_join_saddle_death": None
        if saddles.join_trailing is None
        else saddles.join_trailing.fitness,
    }
    if reflected is not None:
        expected["reflected"] = {
            "rule": reflected.rule,
            "decimals": reflected.decimals,
            "median": reflected.median,
            "global_minimum": reflected.global_minimum,
            "median_gap": reflected.median_gap,
            "unrounded": reflected.unrounded,
            "threshold": reflected.threshold,
        }
    assert saved == expected


def test_threshold_diagnostics_product_identity_is_config_dependent(
    tmp_path, monkeypatch
):
    """Different thr/fdr in config + declaration + summary produce distinct ids (step 2)."""
    # build twice with different decls (thr labels will differ via selection_source)
    # ids must differ
    context1 = RunContext(
        root=tmp_path / "c1",
        output_root=tmp_path / "c1" / "output",
        env={},
    )
    context2 = RunContext(
        root=tmp_path / "c2",
        output_root=tmp_path / "c2" / "output",
        env={},
    )
    median_decl = {"rule": "median_reflected", "decimals": 6}

    _install_synthetic_for_threshold(
        context1, monkeypatch=monkeypatch, declaration=median_decl
    )
    p1 = build_threshold_diagnostics_product(context=context1)
    id1 = load(
        "threshold_diagnostics", p1.name, context=context1
    ).manifest.scientific_id_sha256

    # second with fdr style
    fdr_decl = {
        "rule": "fdr_fixed_cutoff",
        "decimals": 6,
        "cutoff": "papkou_original",
        "method": "by",
        "q": 0.05,
        "input_dir": "data/processed/classification/published",
    }
    _install_mixture_summary(monkeypatch)
    _patch_fdr_table(monkeypatch)
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp
    import dubanevics2026topological.landscape.selection as selection_mod

    def _summary(_declaration):
        return {"threshold": {"unrounded": 0.42, "rounded": 0.42}}

    monkeypatch.setattr(tdp, "threshold_summary", _summary)
    monkeypatch.setattr(selection_mod, "threshold_summary", _summary)
    _install_synthetic_for_threshold(
        context2, monkeypatch=monkeypatch, declaration=fdr_decl
    )
    p2 = build_threshold_diagnostics_product(context=context2)
    id2 = load(
        "threshold_diagnostics", p2.name, context=context2
    ).manifest.scientific_id_sha256
    assert id1 != id2


def test_threshold_diagnostics_product_missing_field_is_loud(tmp_path, monkeypatch):
    """Missing field in the saved diags json is not defaulted (no .get); loud failure."""
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch)
    path = build_threshold_diagnostics_product(context=context)
    # corrupt the artifact to drop a required field (triggers verified check first)
    prod = load("threshold_diagnostics", path.name, context=context)
    diags = json.loads(
        prod.artifacts["threshold_diagnostics.json"].read_text(encoding="utf-8")
    )
    # remove a field that load and figure expect
    del diags["candidate"]["unrounded"]
    prod.artifacts["threshold_diagnostics.json"].write_text(
        json.dumps(diags) + "\n", encoding="utf-8"
    )
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_threshold_diagnostics_product,
    )

    with pytest.raises(MissingProductError, match="threshold-diagnostics-product"):
        load_threshold_diagnostics_product(
            context, settings=tdp.load_settings(context=context)
        )


def test_threshold_diagnostics_product_fdr_fixed_cutoff_uses_snapshot_not_classification(
    tmp_path, monkeypatch
):
    """FDR_FIXED_CUTOFF hashes threshold_summary(declaration) and never loads
    the functional classification checkpoint (mirrors threshold_from_summary).
    """
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    # Use complete FDR declaration from the shipped BY-variant config overlay.
    import tomllib

    from dubanevics2026topological.analysis import ROOT

    with (ROOT / "config/variants/threshold=published_fdr=by_q=0.05.toml").open(
        "rb"
    ) as stream:
        overlay = tomllib.load(stream)
    fdr_decl = overlay["clamped"]["candidate_functional_threshold"]

    _install_synthetic_for_threshold(
        context, monkeypatch=monkeypatch, declaration=fdr_decl
    )

    import dubanevics2026topological.build.threshold_diagnostics_product as tdp

    def _fake_threshold_summary(declaration):
        return {"threshold": {"unrounded": 0.42, "rounded": 0.42}}

    monkeypatch.setattr(tdp, "threshold_summary", _fake_threshold_summary)

    import dubanevics2026topological.landscape.selection as selection_mod

    monkeypatch.setattr(selection_mod, "threshold_summary", _fake_threshold_summary)
    _patch_fdr_table(monkeypatch, sha="a" * 64)

    # Spy: if the classification loader is ever imported+called for FDR, fail.
    calls = []

    def _spy_loader(*a, **k):
        calls.append((a, k))
        raise AssertionError(
            "FDR_FIXED_CUTOFF must not call functional classification loader"
        )

    # Patch only on the functional_classification module (selection does local import).
    # Use dotted string form for setattr.
    monkeypatch.setattr(
        "dubanevics2026topological.threshold.functional_classification.load_cached_functional_classification_with_source",
        _spy_loader,
    )

    path = build_threshold_diagnostics_product(context=context)
    assert path.exists()

    # Loader was never invoked.
    assert calls == [], "classification loader must not be called for FDR branch"

    # Verify we recorded the snapshot summary hash and the table_sha256.
    prod = load("threshold_diagnostics", path.name, context=context)
    prov = getattr(prod.manifest, "data_provenance", {}) or {}
    # The product records the fdr summary and the declaration's table sha.
    assert "fdr_threshold_summary" in prov or "fdr_table_sha256" in prov
    # The table digest is the one of the processed classification table itself.
    assert prov["fdr_table_sha256"] == "a" * 64
    assert "table_sha256" not in fdr_decl
    # For FDR, no reflected (only for median_reflected rule)
    diags = json.loads(
        prod.artifacts["threshold_diagnostics.json"].read_text(encoding="utf-8")
    )
    assert "reflected" not in diags
    c = diags["candidate"]
    assert c["threshold"] == 0.42
    assert c["rule"] == "fdr_fixed_cutoff"


def test_threshold_stale_on_summary_change(tmp_path, monkeypatch):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch)
    path = build_threshold_diagnostics_product(context=context)
    assert path.exists()
    # change a hashed input (fitness) to make hashes differ -> stale on load
    wt = context.root / "data" / "raw" / "dummy_fitness_wt.parquet"
    pl.DataFrame({"col": [99]}).write_parquet(wt)
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_threshold_diagnostics_product,
    )

    with pytest.raises(MissingProductError, match="stale"):
        load_threshold_diagnostics_product(
            context, settings=tdp.load_settings(context=context)
        )


def test_threshold_load_selects_by_summary_hash(tmp_path, monkeypatch):
    """Refit on changed mixture summary selects by the classification summary hash."""
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    mixture_decl = {"rule": "mixture_quantile", "decimals": 3}
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_threshold_diagnostics_product,
    )

    # A: 0.42
    _install_synthetic_for_threshold(
        context, monkeypatch=monkeypatch, declaration=mixture_decl
    )
    _install_mixture_summary(monkeypatch)
    build_threshold_diagnostics_product(context=context)
    settings = tdp.load_settings(context=context)
    diags, idA = load_threshold_diagnostics_product(context, settings=settings)
    assert diags["candidate"].threshold == 0.42

    # B: 0.43
    _install_mixture_summary(monkeypatch, 0.43)
    _install_synthetic_for_threshold(
        context, monkeypatch=monkeypatch, declaration=mixture_decl
    )
    build_threshold_diagnostics_product(context=context)
    settings = tdp.load_settings(context=context)
    diags, idB = load_threshold_diagnostics_product(context, settings=settings)
    assert idB != idA
    assert diags["candidate"].threshold == 0.43

    # back to A
    _install_mixture_summary(monkeypatch)
    _install_synthetic_for_threshold(
        context, monkeypatch=monkeypatch, declaration=mixture_decl
    )
    settings = tdp.load_settings(context=context)
    diags, idA2 = load_threshold_diagnostics_product(context, settings=settings)
    assert idA2 == idA
    assert diags["candidate"].threshold == 0.42


def test_load_threshold_diagnostics_product_selects_by_data_provenance_on_rebuild(
    tmp_path, monkeypatch
):
    """Build A, change fitness dummy (alters data_provenance), build B; load returns B.
    Restore bytes; load returns A.
    """
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_for_threshold(context, monkeypatch=monkeypatch)
    import dubanevics2026topological.build.threshold_diagnostics_product as tdp
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_threshold_diagnostics_product,
    )

    settings = tdp.load_settings(context=context)
    build_threshold_diagnostics_product(context=context)
    _, id_a = load_threshold_diagnostics_product(context, settings=settings)

    # change input that affects hashes
    wt = context.root / "data" / "raw" / "dummy_fitness_wt.parquet"
    pl.DataFrame({"col": [99]}).write_parquet(wt)

    build_threshold_diagnostics_product(context=context)
    _, id_b = load_threshold_diagnostics_product(context, settings=settings)
    assert id_b != id_a

    # current load selects B
    _, id_loaded = load_threshold_diagnostics_product(context, settings=settings)
    assert id_loaded == id_b

    # restore selects A
    pl.DataFrame({"col": [0.0]}).write_parquet(wt)
    _, id_loaded = load_threshold_diagnostics_product(context, settings=settings)
    assert id_loaded == id_a
