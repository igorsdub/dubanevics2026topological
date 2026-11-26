"""``build figures --with-bootstrap`` and the font directory of a figures job."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dubanevics2026topological.build import (
    BAND_SCAN_FIGURE_IDS,
    BOOTSTRAP_FIGURE_IDS,
    default_selection,
    main,
    register_font_directory,
)
from dubanevics2026topological.figure_exclusions import bootstrap_identity_exclusions
from dubanevics2026topological.figures.registry import FIGURE_SPECS
from dubanevics2026topological.results import RunContext

build_module = sys.modules["dubanevics2026topological.build"]
SPECS = {spec.figure_id: spec for spec in FIGURE_SPECS}


def test_the_default_selection_has_every_manuscript_bootstrap_figure() -> None:
    # The manuscript figures that read the bootstrap are default; the biology figure and the unnumbered scan are not.
    defaults = set(default_selection(SPECS))
    assert BOOTSTRAP_FIGURE_IDS - defaults == {
        "candidate_tree_bootstrap_biology",
        "bootstrap_band_quantile_scan",
    }


def test_with_bootstrap_adds_every_bootstrap_figure_the_band_scans_included() -> None:
    defaults = default_selection(SPECS)
    extended = default_selection(SPECS, with_bootstrap=True)
    added = set(extended) - set(defaults)
    assert added == BOOTSTRAP_FIGURE_IDS - set(defaults)
    assert added == {"candidate_tree_bootstrap_biology", "bootstrap_band_quantile_scan"}
    assert BAND_SCAN_FIGURE_IDS <= set(extended)
    assert build_module.WITH_BOOTSTRAP_FIGURE_IDS == BOOTSTRAP_FIGURE_IDS
    assert [f for f in extended if f in defaults] == list(defaults)  # registry order


def test_the_command_line_passes_with_bootstrap(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        build_module, "build_figures", lambda selected, **kw: calls.append(kw)
    )
    assert main(["figures", "--with-bootstrap", "--variant", "annotated"]) == 0
    assert calls[0]["with_bootstrap"] is True
    assert main(["figures"]) == 0
    assert calls[1]["with_bootstrap"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["figures", "--with-bootstrap", "--figure", "papkou2023_bootstrap_stability"],
        ["tables", "--with-bootstrap"],
        ["figures", "--with-bootstrap", "--label-collision", "rotate"],
    ],
)
def test_with_bootstrap_needs_the_plain_figures_command(argv) -> None:
    with pytest.raises(SystemExit):
        main(argv)


def test_build_figures_refuses_with_bootstrap_and_a_selection() -> None:
    with pytest.raises(ValueError, match="with_bootstrap"):
        build_module.build_figures(["x"], with_bootstrap=True)


def test_the_font_directory_is_registered_when_set(tmp_path, monkeypatch) -> None:
    import vl_convert

    seen: list[str] = []
    monkeypatch.setattr(vl_convert, "register_font_directory", seen.append)
    monkeypatch.delenv("FITNESSTREE_FONT_DIR", raising=False)
    assert register_font_directory() is None and seen == []
    monkeypatch.setenv("FITNESSTREE_FONT_DIR", str(tmp_path))
    assert register_font_directory() == Path(tmp_path).resolve()
    assert seen == [str(Path(tmp_path).resolve())]


def test_a_missing_font_directory_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FITNESSTREE_FONT_DIR", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="FITNESSTREE_FONT_DIR"):
        register_font_directory()


def _slot(directory: Path, *, source: str | None) -> Path:
    """A working slot with a record, and a restore marker naming ``source`` if given."""
    import hashlib
    import json

    directory.mkdir(parents=True)
    record = directory / "provenance.json"
    record.write_text(json.dumps({"fingerprint": {"v": 1}, "artifacts": {"x": "h"}}))
    if source is not None:
        (directory / ".restored_from.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                }
            )
        )
    return directory


def _digests(monkeypatch: pytest.MonkeyPatch, slot: Path) -> dict:
    from types import SimpleNamespace

    from dubanevics2026topological.simplification import (
        bootstrap_smoothed_tree as stage,
    )

    monkeypatch.setattr(stage, "cache_dir", lambda context=None: slot)
    monkeypatch.setattr(stage, "result_settings", lambda value: value)
    return build_module._bootstrap_digests(
        SimpleNamespace(provenance={"fingerprint": {"v": 1}}),
        {"analysis": {"bootstrap_smoothed_tree": {}}},
    )


def test_the_digests_name_the_source_of_a_restored_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot = _slot(tmp_path / "slot", source="reflected_by")
    digests = _digests(monkeypatch, slot)
    assert digests["bootstrap_source"] == "reflected_by"


def test_the_digests_omit_the_source_of_a_locally_computed_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot = _slot(tmp_path / "slot", source=None)
    assert "bootstrap_source" not in _digests(monkeypatch, slot)
    # A slot overwritten after the restore reports none either.
    other = _slot(tmp_path / "other", source="somewhere")
    (other / "provenance.json").write_text("{}")
    assert "bootstrap_source" not in _digests(monkeypatch, other)


# -- the band-quantile scan digests --------------------------------------------


def _scan_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **tables: bytes):
    """Point the scan module at ``tmp_path`` and give the figure a stand-in bootstrap."""
    from types import SimpleNamespace

    from dubanevics2026topological.analysis import bootstrap_band_quantile_scan as scan

    monkeypatch.setattr(
        scan,
        "scan_paths",
        lambda context=None: (tmp_path / "scan.parquet", tmp_path / "survival.parquet"),
    )
    for name, data in tables.items():
        (tmp_path / f"{name}.parquet").write_bytes(data)
    cached = SimpleNamespace(
        summary={"settings": {"n_replicates": 3, "seed": 1}},
        provenance={"fingerprint": {"config": {"n_replicates": 3, "seed": 1}}},
    )
    settings = {
        "analysis": {"bootstrap_band_quantile_scan": {"grid_step": 0.1}},
    }
    return scan, settings, cached


def test_a_band_scan_figure_without_a_scan_fails_naming_the_recipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, settings, cached = _scan_environment(tmp_path, monkeypatch)
    with pytest.raises(
        FileNotFoundError,
        match=r"data/processed/band_quantile_scan.*just bootstrap-band-quantile-scan.*just fetch-data",
    ):
        build_module._band_scan_digests(settings, cached)


def test_the_scan_digests_cover_the_tables_and_their_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    _, settings, cached = _scan_environment(
        tmp_path, monkeypatch, scan=b"scan bytes", survival=b"survival bytes"
    )
    digests = build_module._band_scan_digests(settings, cached)
    assert "band_quantile_scan_dataset" not in digests
    assert (
        digests["band_quantile_scan_table"] == hashlib.sha256(b"scan bytes").hexdigest()
    )
    assert digests["band_quantile_scan_config"] == {"grid_step": 0.1}
    assert "band_quantile_scan_source" in digests


# The three tests below that read the configured table (build_figures,
# build_figures_keep_going, explicit --figure) are coupled to the live
# config/figures.toml: they expect exactly these two ids to be excluded there.
GUARDED = {
    "candidate_tree_bootstrap_biology": "guard reason a",
    "bootstrap_tree_histogram": "guard reason b",
}


def test_without_exclusions_the_selection_is_the_unchanged_ordered_tuple() -> None:
    expected = tuple(
        figure_id
        for figure_id, spec in SPECS.items()
        if spec.default or figure_id in BOOTSTRAP_FIGURE_IDS
    )
    assert default_selection(SPECS, with_bootstrap=True) == expected
    assert default_selection(SPECS, with_bootstrap=True, excluded={}) == expected


def test_exclusions_drop_only_the_named_figures_in_order_and_log_reasons(capsys) -> None:
    full = default_selection(SPECS, with_bootstrap=True)
    selection = default_selection(SPECS, with_bootstrap=True, excluded=GUARDED)
    assert selection == tuple(f for f in full if f not in GUARDED)
    err = capsys.readouterr().err
    assert "Skipping candidate_tree_bootstrap_biology from --with-bootstrap: guard reason a" in err
    assert "Skipping bootstrap_tree_histogram from the default figures: guard reason b" in err
    assert default_selection(SPECS, excluded=GUARDED) == tuple(
        f for f in default_selection(SPECS) if f not in GUARDED
    )


@pytest.mark.parametrize(
    ("excluded", "message"),
    [
        ({"papkou_unknown_figure": "reason"}, "cannot exclude"),
        ({"full_tree_histogram": "reason"}, "cannot exclude"),
        ({"bootstrap_tree_histogram": ""}, "non-empty string"),
        ({"bootstrap_tree_histogram": "  "}, "non-empty string"),
        ({"bootstrap_tree_histogram": 3}, "non-empty string"),
        ([], "must be a table"),
        ("", "must be a table"),
        (0, "must be a table"),
        (False, "must be a table"),
    ],
)
def test_a_malformed_exclusion_table_is_a_loud_error(excluded, message) -> None:
    with pytest.raises(ValueError, match=message):
        default_selection(SPECS, with_bootstrap=True, excluded=excluded)
    with pytest.raises(ValueError, match=message):
        default_selection(SPECS, excluded=excluded)


def test_a_non_table_value_is_refused() -> None:
    with pytest.raises(ValueError, match="must be a table"):
        bootstrap_identity_exclusions(
            {"bootstrap_identity_exclusions": ["x"]}, BOOTSTRAP_FIGURE_IDS
        )


class _Reached(Exception):
    pass


def _inject_exclusion(monkeypatch) -> dict:
    injected = {"bootstrap_tree_histogram": "synthetic exclusion"}
    real = build_module.load_settings

    def settings():
        loaded = real()
        loaded["figures"]["bootstrap_identity_exclusions"] = injected
        return loaded

    monkeypatch.setattr(build_module, "load_settings", settings)
    return injected


def _selection_spy(monkeypatch):
    seen = []

    def spy(specs, **kwargs):
        seen.append(kwargs)
        raise _Reached

    monkeypatch.setattr(build_module, "default_selection", spy)
    return seen


def test_build_figures_passes_the_configured_exclusions(monkeypatch) -> None:
    injected = _inject_exclusion(monkeypatch)
    seen = _selection_spy(monkeypatch)
    with pytest.raises(_Reached):
        build_module.build_figures(with_bootstrap=True, context=RunContext(env={}))
    assert set(seen[0]["excluded"]) == set(injected)


def test_build_figures_keep_going_passes_the_configured_exclusions(monkeypatch) -> None:
    injected = _inject_exclusion(monkeypatch)
    seen = _selection_spy(monkeypatch)
    with pytest.raises(_Reached):
        build_module.build_figures_keep_going(
            with_bootstrap=True, context=RunContext(env={})
        )
    assert set(seen[0]["excluded"]) == set(injected)


def test_an_explicit_figure_still_reaches_input_loading(monkeypatch) -> None:
    _selection_spy(monkeypatch)

    def loader(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(build_module, "load_figure_inputs", loader)
    with pytest.raises(_Reached):
        build_module.build_figures(
            ["candidate_tree_bootstrap_biology"], context=RunContext(env={})
        )


@pytest.mark.parametrize("entry", [[], "", {"unknown_figure": "reason"}])
def test_a_malformed_table_raises_even_with_an_explicit_figure(
    monkeypatch, entry
) -> None:
    real = build_module.load_settings

    def settings(*args, **kwargs):
        loaded = real(*args, **kwargs)
        loaded["figures"]["bootstrap_identity_exclusions"] = entry
        return loaded

    monkeypatch.setattr(build_module, "load_settings", settings)
    context = RunContext(env={})
    with pytest.raises(ValueError, match="bootstrap_identity_exclusions"):
        build_module.build_figures(["candidate_tree_bootstrap_biology"], context=context)
    with pytest.raises(ValueError, match="bootstrap_identity_exclusions"):
        build_module.build_figures_keep_going(
            ["candidate_tree_bootstrap_biology"], context=context
        )


MISSING = {"peak_uncertainty": "product missing"}


def test_a_missing_product_exclusion_drops_a_default_figure_and_logs(capsys) -> None:
    base = default_selection(SPECS)
    assert "peak_uncertainty" in base
    selection = default_selection(SPECS, missing_product=MISSING)
    assert selection == tuple(f for f in base if f != "peak_uncertainty")
    assert "Skipping peak_uncertainty from the default figures: product missing" in (
        capsys.readouterr().err
    )
    # Removing the entry makes the figure renderable again.
    assert default_selection(SPECS, missing_product={}) == base


@pytest.mark.parametrize(
    "bad", [{"full_tree_histogram": "r"}, {"peak_uncertainty": ""}, [], ""]
)
def test_a_malformed_missing_product_table_is_a_loud_error(bad) -> None:
    with pytest.raises(ValueError, match="missing_product_exclusions"):
        default_selection(SPECS, missing_product=bad)


def test_the_live_config_no_longer_excludes_peak_uncertainty(monkeypatch) -> None:
    # Coupled to the live config/figures.toml: the FDR pooled product is built in-chain.
    seen = _selection_spy(monkeypatch)
    with pytest.raises(_Reached):
        build_module.build_figures(context=RunContext(env={}))
    assert dict(seen[0]["missing_product"]) == {}
    assert "peak_uncertainty" in build_module.MISSING_PRODUCT_FIGURE_IDS


def test_an_explicit_peak_uncertainty_is_not_filtered(monkeypatch) -> None:
    _selection_spy(monkeypatch)

    def loader(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(build_module, "load_figure_inputs", loader)
    with pytest.raises(_Reached):
        build_module.build_figures(["peak_uncertainty"], context=RunContext(env={}))
