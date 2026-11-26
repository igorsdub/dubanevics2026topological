"""The leading-saddle-band stage: publication, reuse, fingerprint and cached loading."""

from __future__ import annotations

import json
from types import SimpleNamespace

from _saddle_band import DECLARATION, JOIN_BELOW, _population, isolated_graph

from dubanevics2026topological import build
from dubanevics2026topological.analysis import component_persistence
from dubanevics2026topological.analysis.leading_saddle_band import ARTEFACTS

CODE_FILES = (
    *(f"build/{name}.py" for name in build.BUILD_MODULES),
    "analysis/leading_saddle_band.py",
    "landscape/full_population.py",
    "landscape/selection.py",
    "landscape/selected_landscape.py",
    "threshold/functional_threshold.py",
    "analysis/codon_entropy.py",
    "analysis/codon_entropy_populations.py",
    "analysis/component_persistence.py",
)


def _wire(tmp_path, monkeypatch, published=-0.5):
    monkeypatch.setattr(build, "OUTPUT", tmp_path)
    # The stage hashes its input and code under ROOT, so stub them there.
    monkeypatch.setattr(build, "ROOT", tmp_path)
    (tmp_path / "wt.parquet").write_bytes(b"fitness input v1")
    (tmp_path / "diff.parquet").write_bytes(b"diff input v1")
    for name in CODE_FILES:
        path = tmp_path / "src/dubanevics2026topological" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")
    provenance = tmp_path / "component_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "fingerprint": "f1",
                "pairs_sha256": "p1",
                "diagnostics_sha256": "d1",
                "pair_count": 99,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        component_persistence,
        "component_paths",
        lambda context=None: SimpleNamespace(provenance=provenance),
    )
    monkeypatch.setattr(
        build, "load_cached_component_persistence", lambda: (JOIN_BELOW, {})
    )
    monkeypatch.setattr(build, "load_full_network_fitness", lambda s: _population())
    monkeypatch.setattr(build, "_full_graph", lambda settings, context: isolated_graph())
    settings = {
        "analysis": {
            "dataset": {
                "fitness_wt": "wt.parquet",
                "fitness_diff": "diff.parquet",
                "landscape": "full",
            },
            "clamped": {
                "published_functional_threshold": published,
                "candidate_functional_threshold": DECLARATION,
            },
        }
    }
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    return settings, provenance


def test_stage_publishes_tables_and_reuses_them(tmp_path, monkeypatch, capsys):
    settings, _ = _wire(tmp_path, monkeypatch)
    written = build.build_leading_saddle_band_tables()
    names = {path.name for path in written}
    assert names == (set(ARTEFACTS) - {"summary.json"}) | {build.EXPORTED_SUMMARY}
    assert all(path.is_file() for path in written)
    assert (tmp_path / "tables" / ".leading-saddle-band-provenance.json").is_file()
    step = tmp_path / "checkpoints" / "leading_saddle_band"
    checkpoint = next(step.glob("*/provenance.json")).parent
    assert json.loads((checkpoint / "provenance.json").read_text())["fingerprint"]
    build.build_leading_saddle_band_tables()
    assert "are current" in capsys.readouterr().out
    identity = build._leading_saddle_band_identity()
    assert set(identity) == {"fingerprint", "artefacts"}
    assert set(identity["artefacts"]) == set(ARTEFACTS)
    cached = build.load_cached_leading_saddle_band(settings)
    assert cached.summary["band_count"] == 3
    assert (
        cached.summary
        == json.loads((checkpoint / "summary.json").read_text())
        == json.loads((tmp_path / "tables" / build.EXPORTED_SUMMARY).read_text())
    )


def test_fingerprint_keys_on_inputs(tmp_path, monkeypatch):
    from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint

    settings, provenance = _wire(tmp_path, monkeypatch)
    base = stage_fingerprint(build._leading_saddle_band_stage(settings))
    clamped = settings["analysis"]["clamped"]
    clamped["published_functional_threshold"] = -0.4
    assert stage_fingerprint(build._leading_saddle_band_stage(settings)) != base
    clamped["published_functional_threshold"] = -0.5
    clamped["candidate_functional_threshold"] = {**DECLARATION, "decimals": 3}
    assert stage_fingerprint(build._leading_saddle_band_stage(settings)) != base
    clamped["candidate_functional_threshold"] = DECLARATION
    record = json.loads(provenance.read_text())
    provenance.write_text(json.dumps({**record, "pairs_sha256": "p2"}))
    assert stage_fingerprint(build._leading_saddle_band_stage(settings)) != base
    # A bookkeeping field of the upstream record is not part of the identity.
    provenance.write_text(json.dumps({**record, "pair_count": 1}))
    assert stage_fingerprint(build._leading_saddle_band_stage(settings)) == base


def test_fingerprint_keys_on_the_fitness_input_and_loader(tmp_path, monkeypatch):
    from dubanevics2026topological.analysis.produce_or_load import stage_fingerprint

    settings, _ = _wire(tmp_path, monkeypatch)
    base = stage_fingerprint(build._leading_saddle_band_stage(settings))
    (tmp_path / "wt.parquet").write_bytes(b"fitness input v2")
    changed_input = stage_fingerprint(build._leading_saddle_band_stage(settings))
    assert changed_input != base
    (tmp_path / "wt.parquet").write_bytes(b"fitness input v1")
    (tmp_path / "diff.parquet").write_bytes(b"diff input v2")
    assert stage_fingerprint(build._leading_saddle_band_stage(settings)) != base
    (tmp_path / "diff.parquet").write_bytes(b"diff input v1")
    # The stage hashes its input and code under ROOT, so stub them there.
    monkeypatch.setattr(build, "ROOT", tmp_path)
    (tmp_path / "wt.parquet").write_bytes(b"fitness input v1")
    for name in CODE_FILES:
        path = tmp_path / "src/dubanevics2026topological" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")
    assert stage_fingerprint(build._leading_saddle_band_stage(settings)) == base
    # The loader's code is part of the identity too.
    (
        tmp_path / "src/dubanevics2026topological/landscape/full_population.py"
    ).write_text("# edited\n")
    # The loader's source is a label, not part of the key.
    assert stage_fingerprint(build._leading_saddle_band_stage(settings)) == base


def test_command_reads_the_checkpoint_where_the_stage_writes_it(
    tmp_path, monkeypatch, capsys
):
    """The command's read-back finds the stage folder, not the bare step root."""
    from dubanevics2026topological.analysis.leading_saddle_band import (
        read_leading_saddle_band,
    )

    _wire(tmp_path, monkeypatch)
    build._report_leading_saddle_band(force=False, context=None)
    out = capsys.readouterr().out
    assert "Table:" in out
    step = tmp_path / "checkpoints" / build.LEADING_SADDLE_BAND_CHECKPOINT
    # The artefacts sit one level below the step root, in the keyed folder.
    leaves = [path.parent for path in step.glob("*/provenance.json")]
    assert len(leaves) == 1
    assert not (step / "leading_saddle_band_landmarks.parquet").exists()
    assert read_leading_saddle_band(leaves[0]).summary["band_count"] == 3
