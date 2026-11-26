"""Biological evidence freshness and canonical publication integration."""

import json
from types import SimpleNamespace

import pytest
from _fake_figures import install_fake_figures

from dubanevics2026topological import build


def test_annotation_evidence_reuses_cache_and_repairs_missing_exports(
    tmp_path, monkeypatch
):
    import dubanevics2026topological.analysis.biological_annotations as biology

    source = (
        tmp_path / "src/dubanevics2026topological/analysis/biological_annotations.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("v1")
    (source.parent / "describe.py").write_text("v1")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")
    provenance = tmp_path / "output/analysis/clamped_comparison/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"generation": 1}\n')
    calls = []
    monkeypatch.setattr(
        biology,
        "_build_biological_annotations",
        lambda data, **_: calls.append(data) or object(),
    )

    def export(data, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("branches", "pairs", "critical_points"):
            for extension in ("csv", "parquet"):
                (directory / f"biological_annotation_{name}.{extension}").write_text(
                    "evidence"
                )
        (directory / "biological_annotation_counts.json").write_text("{}")

    monkeypatch.setattr(biology, "_export_biological_annotations", export)
    sentinel = object()
    payload = {"landscapes": {"study_clamped": sentinel, "candidate_clamped": sentinel}}
    outputs = build.build_biological_tables(payload)
    build.build_biological_tables(payload)
    assert calls == [sentinel] * 2
    # A deleted export is restored from the checkpoint without recomputing.
    outputs[0].unlink()
    build.build_biological_tables(payload)
    assert calls == [sentinel] * 2
    assert outputs[0].read_text() == "evidence"
    tables = tmp_path / "output/tables"
    step = tmp_path / "output/checkpoints/biological_annotation_tables"
    leaves = sorted(step.glob("*/provenance.json"))
    checkpoint = leaves[0].parent if leaves else step
    for path in outputs:
        relative = path.relative_to(tables)
        assert path.read_bytes() == (checkpoint / relative).read_bytes()
    record = json.loads(
        (tmp_path / "output/tables/.biological-annotation-provenance.json").read_text()
    )
    assert set(record["artefacts"]) == {
        path.relative_to(tables).as_posix() for path in outputs
    }
    source.write_text("v2")
    build.build_biological_tables(payload)
    # The annotation source is a label, so editing it does not rebuild.
    assert calls == [sentinel] * 2
    provenance.write_text('{"generation": 2}\n')
    build.build_biological_tables(payload)
    assert calls == [sentinel] * 4
    build.build_biological_tables(payload, force=True)
    assert calls == [sentinel] * 6


def test_incomplete_annotation_export_is_not_published(tmp_path, monkeypatch):
    import dubanevics2026topological.analysis.biological_annotations as biology

    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")
    provenance = tmp_path / "output/analysis/clamped_comparison/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text('{"generation": 1}\n')
    source = (
        tmp_path / "src/dubanevics2026topological/analysis/biological_annotations.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("v1")
    (source.parent / "describe.py").write_text("v1")
    monkeypatch.setattr(
        biology, "_build_biological_annotations", lambda data, **_: object()
    )

    def partial_export(data, directory):
        (directory / "biological_annotation_branches.csv").write_text("x")

    monkeypatch.setattr(biology, "_export_biological_annotations", partial_export)
    with pytest.raises(RuntimeError, match="did not produce all evidence files"):
        build.build_biological_tables({"landscapes": {"study_clamped": object()}})
    assert not (tmp_path / "output/checkpoints/biological_annotation_tables").exists()
    assert not list((tmp_path / "output/tables").glob("biological_annotation_*"))


def test_biological_figure_rejects_vanilla_before_loading_analysis(monkeypatch):
    settings = {"figures": {"variants": ["vanilla", "annotated"], "formats": ["svg"]}}
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    install_fake_figures(
        monkeypatch,
        SimpleNamespace(FIGURE_IDS=("study_tree_persistence_biology",)),
    )
    with pytest.raises(ValueError, match="canonical annotated"):
        build.build_figures(
            ("study_tree_persistence_biology",), variants=("vanilla",)
        )


def test_biological_build_uses_cached_study_payload_and_canonical_output(
    tmp_path, monkeypatch
):
    import sys

    figure_id = "study_tree_persistence_biology"
    settings = {
        "figures": {
            "variants": ["vanilla", "annotated"],
            "formats": ["svg"],
            "figures": {figure_id: {"tree": {"width": 420}}},
        }
    }
    payload = {"landscapes": {"study_clamped": object()}}
    calls = []

    class Chart:
        def save(self, path, **kwargs):
            path.write_text("svg")

    figures = SimpleNamespace(
        FIGURE_IDS=(figure_id,),
        render=lambda data, name, config, **kwargs: (
            calls.append(
                (
                    {
                        k: v
                        for k, v in data.items()
                        if k not in ("manuscript_quantities", "annotations")
                    },
                    kwargs["variant"],
                )
            )
            or Chart()
        ),
    )
    monkeypatch.setattr(build, "load_settings", lambda: settings)
    monkeypatch.setattr(build, "load_saved_biological_annotations", dict)
    install_fake_figures(monkeypatch, figures)
    monkeypatch.setitem(
        sys.modules,
        "dubanevics2026topological.analysis.clamped_cache",
        SimpleNamespace(load_cached_clamped_comparison=lambda: payload),
    )
    monkeypatch.setattr(
        build,
        "_figure_dependencies",
        lambda *args: {
            "width": settings["figures"]["figures"][figure_id]["tree"]["width"]
        },
    )
    monkeypatch.setattr(
        build, "_comparison_analysis_fingerprint", lambda: "cached-science"
    )
    monkeypatch.setattr(build, "_digest_file", lambda path: "source")
    monkeypatch.setattr(build, "require_current_registry", lambda settings: None)
    monkeypatch.setattr(
        build,
        "build_biological_tables",
        lambda *args, **kwargs: pytest.fail("a figure build writes no tables"),
    )
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(build, "FIGURE_DIR", tmp_path / "output/figures")
    monkeypatch.setattr(build, "RECORD_DIR", tmp_path / "output/figures/.records")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(
        build, "load_cached_analysis", lambda: pytest.fail("must use comparison cache")
    )
    assert [path.name for path in build.build_figures((figure_id,))] == [
        f"{figure_id}.svg"
    ]
    build.build_figures((figure_id,))
    # Figures are not cached, so the second call draws again.
    assert calls == [(payload, "annotated"), (payload, "annotated")]
    settings["figures"]["figures"][figure_id]["tree"]["width"] = 600
    build.build_figures((figure_id,))
    assert calls == [(payload, "annotated")] * 3
