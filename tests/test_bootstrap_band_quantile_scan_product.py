"""Tests for the bootstrap band quantile scan product (PR 3c).

The bootstrap checkpoint is replaced by a small fake result; the scan directory
and the output root live under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dubanevics2026topological.analysis import (
    bootstrap_band_quantile_scan as scan_module,
)
from dubanevics2026topological.build import (
    bootstrap_band_quantile_scan_product as product_module,
)
from dubanevics2026topological.build import saved_figure_inputs as sfi
from dubanevics2026topological.results import (
    MissingProductError,
    RunContext,
    load,
)

DRAWS = pl.DataFrame(
    {"condition": ["real"] * 5, "bottleneck": [0.1, 0.2, 0.3, 0.4, 0.5]}
)
HILLS = pl.DataFrame({"nominal_persistence": [0.2, 0.5, 0.65, 0.9, 1.5]})


def _settings(step: float = 0.25, band_quantile: float = 0.5) -> dict:
    return {
        "analysis": {
            "bootstrap_band_quantile_scan": {"grid_step": step},
            "bootstrap_smoothed_tree": {"band_quantile": band_quantile},
        }
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    output = tmp_path / "output"
    scan_dir = output / "analysis/bootstrap_band_quantile_scan"
    scan_dir.mkdir(parents=True)
    checkpoint = output / "checkpoints/bootstrap_smoothed_tree"
    slot = output / "analysis/bootstrap_smoothed_tree"
    for directory in (checkpoint, slot):
        directory.mkdir(parents=True)
        (directory / "hills.parquet").write_bytes(b"hills")
    result = SimpleNamespace(
        draw_stats=DRAWS,
        hills=HILLS,
        provenance={"fingerprint": {"id": 1}},
    )
    state = SimpleNamespace(
        output=output,
        scan_dir=scan_dir,
        result=result,
        settings=_settings(),
        context=RunContext(output_root=output),
    )
    monkeypatch.setattr(
        product_module, "load_settings", lambda context=None: state.settings
    )
    monkeypatch.setattr(
        product_module,
        "load_cached_bootstrap_smoothed_tree",
        lambda settings=None, **_: state.result,
    )
    monkeypatch.setattr(scan_module, "ensure_scan_tables", lambda *a, **k: None)
    monkeypatch.setattr(
        product_module.checkpoint_identity,
        "scientific_fingerprint",
        lambda fingerprint: dict(fingerprint),
    )
    write_scan(state)
    return state


def write_scan(state: SimpleNamespace, step: float = 0.25) -> None:
    scan_module.write_scan(
        scan_module.scan_from_result(state.result, step),
        scan_module.scan_paths(state.context)[0],
    )
    survival = scan_module.survival_from_result(state.result)
    survival.write_parquet(scan_module.scan_paths(state.context)[1])
    (state.scan_dir / scan_module.RECORD_NAME).write_text("{}\n", encoding="utf-8")


def test_round_trip(env: SimpleNamespace) -> None:
    path = product_module.build_bootstrap_band_quantile_scan_product(env.context)
    assert path.is_dir()
    prod = load("bootstrap_band_quantile_scan", context=env.context)
    assert prod.report.status == "full" and prod.report.byte_integrity
    for name in ("scan.parquet", "survival.parquet", "metadata.json"):
        assert name in prod.artifacts
    provenance = prod.manifest.data_provenance
    assert "scan/scan.parquet" in provenance
    assert "scan/provenance.json" in provenance
    assert any(key.startswith("bootstrap/") for key in provenance)
    loaded = sfi.load_bootstrap_band_quantile_scan_product(
        env.context, settings=env.settings, bootstrap=env.result
    )
    assert loaded["scan"].equals(
        pl.read_parquet(scan_module.scan_paths(env.context)[0])
    )
    assert (
        product_module.build_bootstrap_band_quantile_scan_product(env.context) == path
    )


def test_parameter_change_is_an_identity_mismatch(env: SimpleNamespace) -> None:
    product_module.build_bootstrap_band_quantile_scan_product(env.context)
    changed = _settings(band_quantile=0.75)
    with pytest.raises(MissingProductError):
        sfi.load_bootstrap_band_quantile_scan_product(
            env.context, settings=changed, bootstrap=env.result
        )


def test_tampered_table_fails_closed(env: SimpleNamespace) -> None:
    product_module.build_bootstrap_band_quantile_scan_product(env.context)
    with scan_module.scan_paths(env.context)[1].open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(MissingProductError, match="Missing or stale"):
        sfi.load_bootstrap_band_quantile_scan_product(
            env.context, settings=env.settings, bootstrap=env.result
        )


def test_changed_bootstrap_file_fails_closed(env: SimpleNamespace) -> None:
    product_module.build_bootstrap_band_quantile_scan_product(env.context)
    (env.output / "analysis/bootstrap_smoothed_tree" / "hills.parquet").write_bytes(
        b"other"
    )
    with pytest.raises(MissingProductError):
        sfi.load_bootstrap_band_quantile_scan_product(
            env.context, settings=env.settings, bootstrap=env.result
        )


def test_stale_tables_are_rejected_at_build(env: SimpleNamespace) -> None:
    env.result = SimpleNamespace(
        draw_stats=DRAWS,
        hills=pl.DataFrame({"nominal_persistence": [0.3, 0.8]}),
        provenance={"fingerprint": {"id": 1}},
    )
    with pytest.raises(RuntimeError, match="stale band quantile tables"):
        product_module.build_bootstrap_band_quantile_scan_product(env.context)


def test_overridden_output_root_reads_its_own_checkpoint(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    """The default-root refusal is gone: another root is read, and has nothing yet."""
    other = RunContext(output_root=tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError, match="Bootstrap checkpoint artifacts"):
        product_module.build_bootstrap_band_quantile_scan_product(other)
