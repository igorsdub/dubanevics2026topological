"""Tests for the pooled_scatter saved product (build + load round trips, identity)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.analysis.pooled_scatter import pooled_scatter
from dubanevics2026topological.build.pooled_scatter_product import (
    build_pooled_scatter_product,
)
from dubanevics2026topological.results import (
    MissingProductError,
    RunContext,
    compute_scientific_hash,
    load,
)

WILD_TYPE = "GCCGATCTC"
PEAK = "AAAAAAAAC"
VALLEY = "AAAAAAAAG"
UNFITTED = "AAAAAAACA"
R_WT = 0.038
SE_WT = 0.001
FITTED = ((PEAK, 0.5, 0.04), (VALLEY, -0.2, 0.08))
UNFITTED_S, UNFITTED_SE_POOLED = -0.3, 0.05


def _published_se(se_pooled: float) -> float:
    return (se_pooled**2 + SE_WT**2) ** 0.5


def _peak_checkpoint() -> dict:
    rows = [
        (PEAK, "KKT", 0.5, _published_se(0.04), True),
        (VALLEY, "KKK", -0.2, _published_se(0.08), False),
        (UNFITTED, "KKT", UNFITTED_S, _published_se(UNFITTED_SE_POOLED), False),
    ]
    scatter = pl.DataFrame(
        rows,
        schema=[
            "dna_sequence",
            "aa_sequence",
            "raw_fitness",
            "standard_error",
            "is_peak",
        ],
        orient="row",
    )
    peaks = pl.DataFrame({"dna_sequence": [PEAK]})
    return {
        "tables": {
            "study_clamped_scatter": scatter,
            "candidate_clamped_scatter": scatter,
            "study_clamped_peaks": peaks,
            "candidate_clamped_peaks": peaks,
            "study_clamped_contrasts": pl.DataFrame({"dna_sequence": []}),
            "candidate_clamped_contrasts": pl.DataFrame({"dna_sequence": []}),
            "study_clamped_decay": pl.DataFrame({"dna_sequence": []}),
            "candidate_clamped_decay": pl.DataFrame({"dna_sequence": []}),
            "pairwise_sign_disagreements": pl.DataFrame({"dna_sequence": []}),
        },
        "diagnostics": {
            "wild_type_sequence": WILD_TYPE,
            "fitness_domain": [-0.2, 0.5],
            "thresholds": {"study_clamped": -0.5, "candidate_clamped": -0.25},
            "wild_type_scatter_excluded": True,
            "wild_type_scatter_excluded_count": 0,
        },
        "provenance": {
            "schema": 2,
            "analysis_scalar": "raw_fitness",
            "edge_selection_rule": "at_least_one_endpoint_raw_fitness_ge_threshold",
            "candidate_edge_selection_rule": "at_least_one_endpoint_raw_fitness_ge_threshold",
            "published_edge_selection_rule": "at_least_one_endpoint_raw_fitness_ge_threshold",
            "bh_family": "all_fitness_diff_rows_including_zero_deltas",
            "fingerprint": "fp-synth",
        },
    }


def _growth_df() -> pl.DataFrame:
    rows = [(WILD_TYPE, R_WT, SE_WT, 0.004)]
    rows += [(dna, R_WT + s, se, 2 * se) for dna, s, se in FITTED]
    return pl.DataFrame(
        rows,
        schema=["dna_sequence", "r", "se_pooled", "se_replicate"],
        orient="row",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_synthetic_growth(context: RunContext) -> None:
    """Install a minimal processed growth-rate table."""
    root = context.processed_root / "growth_rates"
    root.mkdir(parents=True, exist_ok=True)
    _growth_df().write_parquet(root / "growth_rates.parquet")
    (root / "diagnostics.json").write_text(
        json.dumps({"wild_type_sequence": WILD_TYPE}), encoding="utf-8"
    )


def _install_synthetic_peak(context: RunContext, monkeypatch=None) -> Path:
    """Write a synthetic legacy peak checkpoint under context output."""
    peak_dir = context.output_root / "analysis" / "peak_uncertainty"
    tables_dir = peak_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    peak = _peak_checkpoint()
    for name, frame in peak["tables"].items():
        frame.write_parquet(tables_dir / f"{name}.parquet")
    (peak_dir / "diagnostics.json").write_text(
        json.dumps(peak["diagnostics"], sort_keys=True) + "\n", encoding="utf-8"
    )
    (peak_dir / "provenance.json").write_text(
        json.dumps(peak["provenance"], sort_keys=True) + "\n", encoding="utf-8"
    )
    return peak_dir


def test_pooled_scatter_product_round_trip_and_equals_old_compose(
    tmp_path, monkeypatch
):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)

    path = build_pooled_scatter_product(context=context)
    assert path.exists()
    assert (path / "manifest.json").exists()

    # Load via results and compare tables to direct computation on same inputs.
    prod = load("pooled_scatter", path.name, context=context)
    assert prod.report.status == "full"
    assert prod.report.byte_integrity

    # reconstruct payload like figure does
    loaded_tables = {
        n.removesuffix(".parquet"): pl.read_parquet(p)
        for n, p in prod.artifacts.items()
        if n.endswith(".parquet")
    }
    loaded_diags = json.loads(
        prod.artifacts["diagnostics.json"].read_text(encoding="utf-8")
    )

    # old render-time on identical inputs
    peak = _peak_checkpoint()
    gr = _growth_df()
    direct = pooled_scatter(
        peak,
        gr,
        identity_tolerance=1e-6,
        standard_error_tolerance=1e-3,
    )

    # tables equal (same rows, order from concat in pooled_scatter preserved)
    for key in direct["tables"]:
        assert key in loaded_tables
        assert loaded_tables[key].equals(direct["tables"][key])
    assert (
        loaded_diags["wild_type_sequence"]
        == direct["diagnostics"]["wild_type_sequence"]
    )

    # direct load round-trips the product's tables and identity
    again = load("pooled_scatter", path.name, context=context)
    assert again.manifest.scientific_id_sha256 == prod.manifest.scientific_id_sha256


def test_pooled_scatter_product_missing_raises_with_cli_name(tmp_path, monkeypatch):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_peak(context, monkeypatch)
    # no product built, so load inside raises the named Missing
    with pytest.raises(MissingProductError, match="pooled-scatter-product"):
        # the loader inside saved raises; build itself would also fail on growth load
        # but we test the named error path used by figure
        from dubanevics2026topological.build.saved_figure_inputs import (
            _load_pooled_scatter_payload,
        )

        _load_pooled_scatter_payload(context)


def test_build_pooled_scatter_product_needs_the_processed_growth_rates(
    tmp_path, monkeypatch
):
    context = RunContext(root=tmp_path, output_root=tmp_path / "output", env={})
    _install_synthetic_peak(context, monkeypatch)
    with pytest.raises(MissingProductError, match="just all"):
        build_pooled_scatter_product(context=context)


def test_growth_rates_parent_is_the_processed_table_sha(tmp_path, monkeypatch):
    context = RunContext(root=tmp_path, output_root=tmp_path / "output", env={})
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    path = build_pooled_scatter_product(context=context)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    (parent,) = [p for p in manifest["parents"] if p["product_type"] == "growth_rates"]
    table = context.processed_root / "growth_rates" / "growth_rates.parquet"
    assert parent["scientific_id_sha256"] == _sha256(table)


def test_load_pooled_scatter_product_requires_exactly_one_growth_rates_parent(
    tmp_path, monkeypatch
):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    path = build_pooled_scatter_product(context=context)

    import dubanevics2026topological.build.saved_figure_inputs as sfi

    orig_load = sfi.load

    # case (i): no growth_rates parent
    good_manifest_path = path / "manifest.json"
    manifest_dict = json.loads(good_manifest_path.read_text(encoding="utf-8"))
    bad_no = json.loads(json.dumps(manifest_dict))
    bad_no["parents"] = [
        p for p in bad_no.get("parents", []) if p.get("product_type") != "growth_rates"
    ]
    parent_ids = [
        p.get("scientific_id_sha256")
        for p in bad_no.get("parents", [])
        if isinstance(p, dict) and "scientific_id_sha256" in p
    ]
    full, short = compute_scientific_hash(
        bad_no["product_type"],
        bad_no["scientific_identity"],
        parent_scientific_ids=parent_ids,
        input_data_sha256s=bad_no.get("data_provenance", {}),
        stage_code_digest=bad_no.get("code_provenance", {}).get("stage_code_digest"),
    )
    bad_no["scientific_id_sha256"] = full
    good_label = path.name
    if "_hash=" in good_label:
        base = good_label.rsplit("_hash=", 1)[0]
        bad_label = f"{base}_hash={short}"
    else:
        bad_label = good_label
    bad_dir = path.parent / bad_label
    bad_dir.mkdir(parents=True, exist_ok=True)
    for n in list(manifest_dict.get("artifacts", {}).keys()):
        src = path / n
        dst = bad_dir / n
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    (bad_dir / "manifest.json").write_text(
        json.dumps(bad_no, sort_keys=True) + "\n", encoding="utf-8"
    )

    def _force(name, label=None, *, context=None, expected=None):
        if name == "pooled_scatter":
            return load(name, bad_label, context=context)
        return orig_load(name, label=label, context=context, expected=expected)

    monkeypatch.setattr(sfi, "load", _force)
    with pytest.raises(ValueError, match="exactly one growth_rates parent"):
        sfi.load_pooled_scatter_product(context)

    # case (ii): two growth_rates parents
    monkeypatch.setattr(sfi, "load", orig_load)  # restore before next
    bad_two = json.loads(json.dumps(manifest_dict))
    gr = {"product_type": "growth_rates", "scientific_id_sha256": "f" * 64}
    bad_two["parents"] = [
        p for p in bad_two.get("parents", []) if p.get("product_type") != "growth_rates"
    ] + [gr, gr]
    parent_ids = [
        p.get("scientific_id_sha256")
        for p in bad_two.get("parents", [])
        if isinstance(p, dict) and "scientific_id_sha256" in p
    ]
    full, short = compute_scientific_hash(
        bad_two["product_type"],
        bad_two["scientific_identity"],
        parent_scientific_ids=parent_ids,
        input_data_sha256s=bad_two.get("data_provenance", {}),
        stage_code_digest=bad_two.get("code_provenance", {}).get("stage_code_digest"),
    )
    bad_two["scientific_id_sha256"] = full
    if "_hash=" in good_label:
        base = good_label.rsplit("_hash=", 1)[0]
        bad_label = f"{base}_hash={short}"
    else:
        bad_label = good_label
    bad_dir = path.parent / bad_label
    bad_dir.mkdir(parents=True, exist_ok=True)
    for n in list(manifest_dict.get("artifacts", {}).keys()):
        src = path / n
        dst = bad_dir / n
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    (bad_dir / "manifest.json").write_text(
        json.dumps(bad_two, sort_keys=True) + "\n", encoding="utf-8"
    )

    def _force2(name, label=None, *, context=None, expected=None):
        if name == "pooled_scatter":
            return load(name, bad_label, context=context)
        return orig_load(name, label=label, context=context, expected=expected)

    monkeypatch.setattr(sfi, "load", _force2)
    with pytest.raises(ValueError, match="exactly one growth_rates parent"):
        sfi.load_pooled_scatter_product(context)


def test_pooled_scatter_product_changed_input_bytes_new_identity(tmp_path, monkeypatch):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    peak_dir = _install_synthetic_peak(context, monkeypatch)
    first_path = build_pooled_scatter_product(context=context)
    first_prod = load("pooled_scatter", first_path.name, context=context)
    first_id = first_prod.manifest.scientific_id_sha256
    first_input_shas = dict(first_prod.manifest.data_provenance)

    # Deliberate single-byte-affecting change to diagnostics.json (one of the files
    # explicitly hashed into input_data_sha256s at build time).
    diag_path = peak_dir / "diagnostics.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    diag["test_input_change_marker"] = "changed"
    diag_path.write_text(json.dumps(diag, sort_keys=True) + "\n", encoding="utf-8")

    second_path = build_pooled_scatter_product(context=context)
    second_prod = load("pooled_scatter", second_path.name, context=context)

    # (a) input_data_sha256s (recorded as data_provenance) differs
    assert second_prod.manifest.data_provenance != first_input_shas

    # (b) product identity differs and a new directory is published
    assert second_prod.manifest.scientific_id_sha256 != first_id
    assert second_path != first_path

    # (c) old product still loads unchanged
    reloaded_first = load("pooled_scatter", first_path.name, context=context)
    assert reloaded_first.manifest.scientific_id_sha256 == first_id
    assert reloaded_first.manifest.data_provenance == first_input_shas


def test_build_pooled_scatter_product_fails_on_missing_context_checkpoint(tmp_path):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "empty_output",
        env={},
    )
    _install_synthetic_growth(context)
    # no peak checkpoint is installed under the context's output root
    with pytest.raises(FileNotFoundError, match="No peak uncertainty checkpoint"):
        build_pooled_scatter_product(context=context)


def test_load_pooled_scatter_product_does_not_touch_growth_archive(
    tmp_path, monkeypatch
):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    build_pooled_scatter_product(context=context)
    import dubanevics2026topological.build.saved_figure_inputs as sfi

    orig_load = sfi.load
    growth_calls = []

    def recording_load(name, label=None, *, context=None, expected=None):
        if name == "growth_rates":
            growth_calls.append(label)
            return type(
                "P",
                (),
                {"manifest": type("M", (), {"scientific_id_sha256": "0" * 64})()},
            )()
        return orig_load(name, label=label, context=context, expected=expected)

    monkeypatch.setattr(sfi, "load", recording_load)
    _ = sfi.load_pooled_scatter_product(context)
    assert not growth_calls


def test_load_saved_figure_inputs_mixture_does_not_require_pooled(
    tmp_path, monkeypatch
):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    import dubanevics2026topological.build.saved_figure_inputs as sfi

    requested = []

    def spy(name, label=None, **k):
        requested.append(name)
        if name == "pooled_scatter":
            pytest.fail("load_reviewed should not load pooled_scatter")
        from dubanevics2026topological.results import MissingProductError

        raise MissingProductError(f"no {name}")

    monkeypatch.setattr(sfi, "load", spy)
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_saved_figure_inputs,
    )
    from dubanevics2026topological.results import MissingProductError

    # mixture
    sel = {"rule": "mixture_quantile"}
    result = load_saved_figure_inputs(context, sel, include_band=False)
    assert result.get("mode") == "legacy_mixture"
    assert requested == []
    # reviewed non-mixture
    requested.clear()
    monkeypatch.setattr(sfi, "_tree_for_leaf", lambda name, context: spy("tree"))
    sel2 = {"rule": "median_reflected"}
    with pytest.raises(MissingProductError):
        load_saved_figure_inputs(context, sel2, include_band=False)
    assert "pooled_scatter" not in requested


def test_load_pooled_scatter_product_rejects_malformed_non_hex_parent_id(
    tmp_path, monkeypatch
):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    path = build_pooled_scatter_product(context=context)

    import dubanevics2026topological.build.saved_figure_inputs as sfi

    orig = sfi.load

    good_manifest_path = path / "manifest.json"
    manifest_dict = json.loads(good_manifest_path.read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(manifest_dict))
    for p in bad.get("parents", []):
        if p.get("product_type") == "growth_rates":
            p["scientific_id_sha256"] = "z" * 64
            break
    parent_ids = [
        p.get("scientific_id_sha256")
        for p in bad.get("parents", [])
        if isinstance(p, dict) and "scientific_id_sha256" in p
    ]
    full, short = compute_scientific_hash(
        bad["product_type"],
        bad["scientific_identity"],
        parent_scientific_ids=parent_ids,
        input_data_sha256s=bad.get("data_provenance", {}),
        stage_code_digest=bad.get("code_provenance", {}).get("stage_code_digest"),
    )
    bad["scientific_id_sha256"] = full
    good_label = path.name
    if "_hash=" in good_label:
        base = good_label.rsplit("_hash=", 1)[0]
        bad_label = f"{base}_hash={short}"
    else:
        bad_label = good_label
    bad_dir = path.parent / bad_label
    bad_dir.mkdir(parents=True, exist_ok=True)
    for n in list(manifest_dict.get("artifacts", {}).keys()):
        src = path / n
        dst = bad_dir / n
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    (bad_dir / "manifest.json").write_text(
        json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8"
    )

    def _force(name, label=None, *, context=None, expected=None):
        if name == "pooled_scatter":
            return load(name, bad_label, context=context)
        return orig(name, label=label, context=context, expected=expected)

    monkeypatch.setattr(sfi, "load", _force)
    with pytest.raises(ValueError, match="64-hex"):
        sfi.load_pooled_scatter_product(context)


def test_pooled_scatter_product_stale_on_table_byte_change(tmp_path, monkeypatch):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    path = build_pooled_scatter_product(context=context)
    assert path.exists()
    # rewrite one of the 9 tables with different bytes
    table = (
        context.output_root
        / "analysis"
        / "peak_uncertainty"
        / "tables"
        / "study_clamped_scatter.parquet"
    )
    if table.exists():
        pl.DataFrame({"dummy": [1]}).write_parquet(table)
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_pooled_scatter_product,
    )

    with pytest.raises(MissingProductError, match="stale"):
        load_pooled_scatter_product(context)


def test_load_pooled_scatter_product_selects_by_data_provenance_on_rebuild(
    tmp_path, monkeypatch
):
    """Build A, change a peak table (new data_provenance), build B; load returns B.
    Restore original bytes; load returns A. (Uses data_provenance in expected filter.)
    """
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    path_a = build_pooled_scatter_product(context=context)
    prod_a = load("pooled_scatter", path_a.name, context=context)
    dp_a = prod_a.manifest.data_provenance

    # change one table -> different inputs
    # use pairwise_sign_disagreements (one of the 9 in _peak_inputs), which scatter doesn't constrain
    table = (
        context.output_root
        / "analysis"
        / "peak_uncertainty"
        / "tables"
        / "pairwise_sign_disagreements.parquet"
    )
    orig_bytes = table.read_bytes() if table.exists() else b""
    # rewrite with one row added (alters bytes, keeps schema)
    df = pl.read_parquet(table)
    if df.height == 0:
        new_df = pl.DataFrame({"dna_sequence": ["DUMMY"]})
    else:
        new_df = df.slice(0, df.height - 1)  # drop last row
    new_df.write_parquet(table)

    path_b = build_pooled_scatter_product(context=context)
    prod_b = load("pooled_scatter", path_b.name, context=context)
    dp_b = prod_b.manifest.data_provenance
    assert dp_b != dp_a

    # load (unpinned, from settings tols) should now select B because of data_provenance
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_pooled_scatter_product,
    )

    loaded = load_pooled_scatter_product(context)
    assert loaded.manifest.data_provenance == dp_b

    # restore bytes -> current matches A
    if orig_bytes:
        table.write_bytes(orig_bytes)
    else:
        pl.DataFrame({"dummy": [0]}).write_parquet(table)
    loaded = load_pooled_scatter_product(context)
    assert loaded.manifest.data_provenance == dp_a


def test_load_pooled_scatter_product_raises_when_peak_refresh_pending(
    tmp_path, monkeypatch
):
    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    path = build_pooled_scatter_product(context=context)
    assert path.exists()

    import dubanevics2026topological.build.saved_figure_inputs as sfi

    pending = context.output_root / "analysis" / ".peak-uncertainty-pending"
    pending.write_text("pending marker\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refresh is pending"):
        sfi.load_pooled_scatter_product(context)


def test_strict_checkpoint_fingerprint_binds_pooled_product(tmp_path, monkeypatch):
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_pooled_scatter_product,
    )

    context = RunContext(root=tmp_path, output_root=tmp_path / "output", env={})
    _install_synthetic_growth(context)
    _install_synthetic_peak(context, monkeypatch)
    build_pooled_scatter_product(context=context)
    product = load_pooled_scatter_product(
        context, expected_checkpoint_fingerprint="fp-synth"
    )
    assert product.report.byte_integrity
    with pytest.raises(MissingProductError, match="fingerprint"):
        load_pooled_scatter_product(context, expected_checkpoint_fingerprint="wrong")

    table = context.output_root / "analysis/peak_uncertainty/tables/pairwise_sign_disagreements.parquet"
    pl.DataFrame({"dna_sequence": ["changed"]}).write_parquet(table)
    with pytest.raises(MissingProductError, match="stale"):
        load_pooled_scatter_product(context, expected_checkpoint_fingerprint="fp-synth")


@pytest.mark.parametrize("fingerprint", [None, "papkou-fingerprint"])
def test_strict_loader_rejects_missing_or_wrong_fingerprint_even_with_matching_threshold(
    tmp_path, monkeypatch, fingerprint
):
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_pooled_scatter_product,
    )

    context = RunContext(root=tmp_path, output_root=tmp_path / "output", env={})
    _install_synthetic_growth(context)
    checkpoint = _install_synthetic_peak(context, monkeypatch)
    build_pooled_scatter_product(context=context)
    provenance_path = checkpoint / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["fingerprint"] = fingerprint
    provenance["thresholds"] = {"candidate_clamped": -0.253159}
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(MissingProductError, match="fingerprint"):
        load_pooled_scatter_product(context, expected_checkpoint_fingerprint="fp-synth")


def test_strict_loader_requires_checkpoint_and_refuses_pending_refresh(tmp_path, monkeypatch):
    from dubanevics2026topological.build.saved_figure_inputs import (
        load_pooled_scatter_product,
    )

    context = RunContext(root=tmp_path, output_root=tmp_path / "output", env={})
    _install_synthetic_growth(context)
    checkpoint = _install_synthetic_peak(context, monkeypatch)
    build_pooled_scatter_product(context=context)
    pending = checkpoint.parent / ".peak-uncertainty-pending"
    pending.write_text("pending")
    with pytest.raises(RuntimeError, match="pending"):
        load_pooled_scatter_product(context, expected_checkpoint_fingerprint="fp-synth")
    pending.unlink()
    (checkpoint / "provenance.json").unlink()
    with pytest.raises(MissingProductError, match="fdr-pooled-scatter-product"):
        load_pooled_scatter_product(context, expected_checkpoint_fingerprint="fp-synth")
