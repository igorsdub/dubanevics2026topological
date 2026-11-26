"""Focused checks for per-output incremental figure rendering."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import rustworkx as rx
from _analysis_record import write_analysis_record
from _fake_figures import install_fake_figures, write_renderer_sources

from dubanevics2026topological import tables
from dubanevics2026topological.build import FiguresFailedError, build_figures
from dubanevics2026topological.results import RunContext

REAL_ENSURE = sys.modules["dubanevics2026topological.build"].ensure_analysis_checkpoint


class FakeChart:
    def __init__(
        self,
        fail_once: list[bool] | None = None,
        save_calls: list[tuple[str, dict[str, object]]] | None = None,
    ) -> None:
        self.fail_once = fail_once
        self.save_calls = save_calls

    def __deepcopy__(self, memo: dict) -> FakeChart:
        # ``without_headings`` deep-copies the chart for the untitled variant; the
        # copy must share the failure budget, or one simulated failure fires per copy.
        return FakeChart(self.fail_once, self.save_calls)

    def save(self, path: str | Path, *, format: str, **kwargs: object) -> None:
        if self.save_calls is not None:
            self.save_calls.append((format, kwargs))
        if self.fail_once is not None and self.fail_once:
            self.fail_once.pop()
            raise RuntimeError("simulated renderer failure")
        Path(path).write_bytes(f"{format} artifact".encode())


@pytest.fixture
def incremental_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path
    figure_dir = root / "src/dubanevics2026topological/figures"
    figure_dir.mkdir(parents=True)
    write_renderer_sources(figure_dir)
    (root / "config").mkdir()
    (root / "config/figures.toml").write_text("# styles\n", encoding="utf-8")
    (root / "notebooks").mkdir()
    (root / "notebooks/one.py").write_text("# notebook\n", encoding="utf-8")
    provenance = write_analysis_record(root / "output", "generation-1")
    (provenance.parent / "component_pairs.parquet").write_bytes(b"pairs")
    (provenance.parent / "component_diagnostics.json").write_text("{}", "utf-8")
    helper = root / "src/dubanevics2026topological/landscape/full_population.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# population loader\n", encoding="utf-8")
    (helper.parent / "selection.py").write_text("# selection\n", encoding="utf-8")
    (helper.parent / "selected_landscape.py").write_text(
        "# selected landscape implementation\n", encoding="utf-8"
    )
    threshold_source = (
        root / "src/dubanevics2026topological/threshold/functional_threshold.py"
    )
    threshold_source.parent.mkdir(parents=True, exist_ok=True)
    threshold_source.write_text("# functional threshold\n", encoding="utf-8")
    saddle_source = (
        root / "src/dubanevics2026topological/analysis/leading_saddle_band.py"
    )
    saddle_source.parent.mkdir(parents=True, exist_ok=True)
    saddle_source.write_text("# tree saddles\n", encoding="utf-8")
    fitness_path = root / "data/raw/fitness_wt.parquet"
    fitness_path.parent.mkdir(parents=True)
    fitness_path.write_bytes(b"fitness")

    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "data/raw/fitness_wt.parquet"},
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                }
            },
        },
        "figures": {
            "formats": ["svg", "png"],
            "variants": ["vanilla", "annotated"],
            "figures": {
                "full_tree_persistence": {
                    "notebook": "notebooks/one.py",
                    "appearance": {"width": 10},
                },
                "linked_tree_persistence": {"appearance": {"width": 30}},
            },
        },
    }
    builds: list[tuple[str, str]] = []
    fail_once: list[bool] = []
    fake_module = SimpleNamespace(
        FIGURE_IDS=(
            "full_tree_persistence",
            "linked_tree_persistence",
        ),
        render=lambda analysis, figure_id, config, *, variant="annotated": (
            builds.append((figure_id, variant)) or FakeChart(fail_once)
        ),
    )
    build_module = sys.modules["dubanevics2026topological.build"]
    monkeypatch.setattr(build_module, "load_settings", lambda: settings)
    monkeypatch.setattr(build_module, "ensure_analysis_checkpoint", lambda: object())
    monkeypatch.setattr(build_module, "load_cached_analysis", lambda: object())
    monkeypatch.setattr(
        build_module, "load_cached_component_persistence", lambda: (object(), {})
    )
    monkeypatch.setattr(build_module, "load_full_network_fitness", lambda _: object())
    monkeypatch.setattr(build_module, "ROOT", root)
    monkeypatch.setattr(build_module, "OUTPUT", root / "output")
    monkeypatch.setattr(build_module, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(
        build_module, "CHECKPOINT_DIR", root / "output/checkpoints/analysis"
    )
    monkeypatch.setattr(build_module, "FIGURE_DIR", root / "output/figures")
    monkeypatch.setattr(build_module, "RECORD_DIR", root / "output/figures/.records")
    install_fake_figures(monkeypatch, fake_module)

    import dubanevics2026topological.build.saved_figure_inputs as saved_inputs

    def _stub_threshold(ctx, settings=None):
        from dubanevics2026topological.threshold.functional_threshold import (
            CandidateThreshold,
        )

        c = CandidateThreshold(
            rule="median_reflected",
            decimals=6,
            median=0.0,
            global_minimum=-1.0,
            median_gap=1.0,
            unrounded=1.0,
            threshold=1.0,
        )
        return {
            "candidate": c,
            "lowest_split_saddle_death": None,
            "highest_join_saddle_death": None,
        }, "t" * 64

    monkeypatch.setattr(
        saved_inputs, "load_threshold_diagnostics_product", _stub_threshold
    )
    return settings, builds, fail_once, root


def test_unchanged_outputs_are_reused(incremental_build) -> None:
    _, builds, _, _ = incremental_build
    first = build_figures(("full_tree_persistence",))
    assert len(first) == 4
    builds.clear()
    second = build_figures(("full_tree_persistence",))
    assert second == first
    # Figures are not cached: the same chart is drawn again.
    assert builds == [
        ("full_tree_persistence", "vanilla"),
        ("full_tree_persistence", "annotated"),
    ]


def test_appearance_edit_rerenders_only_selected_figure(incremental_build) -> None:
    settings, builds, _, _ = incremental_build
    build_figures()
    builds.clear()
    settings["figures"]["figures"]["full_tree_persistence"][
        "appearance"
    ]["width"] = 11
    rendered = build_figures()
    assert len(rendered) == 6
    # The linked figure has one vanilla export under its plain id; the full-graph
    # figure keeps an annotated and a vanilla export.
    assert {path.name for path in rendered} == {
        f"{stem}.{format_name}"
        for stem in (
            "full_tree_persistence",
            "full_tree_persistence_vanilla",
            "linked_tree_persistence",
        )
        for format_name in ("svg", "png")
    }
    # Figures are not cached, so the untouched linked figure is drawn too.
    assert set(builds) == {
        ("full_tree_persistence", "vanilla"),
        ("full_tree_persistence", "annotated"),
        ("linked_tree_persistence", "vanilla"),
    }


def test_science_fingerprint_change_invalidates_render_record(
    incremental_build,
) -> None:
    _, builds, _, root = incremental_build
    build_figures(("full_tree_persistence",), variants=("vanilla",))
    builds.clear()
    write_analysis_record(root / "output", "generation-2")
    assert (
        len(
            build_figures(
                ("full_tree_persistence",), variants=("vanilla",)
            )
        )
        == 2
    )
    assert builds == [("full_tree_persistence", "vanilla")]


def test_component_figure_tracks_full_fitness_and_passes_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dubanevics2026topological.build as build_module

    root = tmp_path
    (root / "config").mkdir()
    (root / "config/figures.toml").write_text("# styles\n", encoding="utf-8")
    write_renderer_sources(root / "src/dubanevics2026topological/figures")
    helper = root / "src/dubanevics2026topological/landscape/full_population.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# population loader\n", encoding="utf-8")
    (helper.parent / "selection.py").write_text("# selection\n", encoding="utf-8")
    (helper.parent / "selected_landscape.py").write_text(
        "# selected landscape implementation\n", encoding="utf-8"
    )
    threshold_source = (
        root / "src/dubanevics2026topological/threshold/functional_threshold.py"
    )
    threshold_source.parent.mkdir(parents=True, exist_ok=True)
    threshold_source.write_text("# functional threshold\n", encoding="utf-8")
    saddle_source = (
        root / "src/dubanevics2026topological/analysis/leading_saddle_band.py"
    )
    saddle_source.parent.mkdir(parents=True, exist_ok=True)
    saddle_source.write_text("# tree saddles\n", encoding="utf-8")
    notebook = root / "notebooks/component.py"
    notebook.parent.mkdir()
    notebook.write_text("# notebook\n", encoding="utf-8")
    fitness_path = root / "data/raw/fitness_wt.parquet"
    fitness_path.parent.mkdir(parents=True)
    fitness_path.write_bytes(b"fitness-v1")
    analysis_dir = root / "output/analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "component_pairs.parquet").write_bytes(b"pairs")
    (analysis_dir / "component_diagnostics.json").write_text("{}", encoding="utf-8")
    figure_dir = root / "output/figures"
    record_dir = figure_dir / ".records"
    settings = {
        "analysis": {
            "dataset": {"fitness_wt": "data/raw/fitness_wt.parquet"},
            "clamped": {
                "candidate_functional_threshold": {
                    "rule": "median_reflected",
                    "decimals": 6,
                }
            },
        },
        "figures": {
            "formats": ["svg"],
            "variants": ["vanilla"],
            "figures": {
                "full_tree_persistence": {
                    "notebook": "notebooks/component.py"
                }
            },
        },
    }
    full_fitness = pl.DataFrame(
        {"dna_sequence": ["WT", "noncritical"], "raw_fitness": [0.0, 0.25]}
    )
    pair_table = pl.DataFrame({"pair_id": [1]})
    payloads: list[dict[str, object]] = []
    fake_module = SimpleNamespace(
        FIGURE_IDS=("full_tree_persistence",),
        render=lambda payload, *_args, **_kwargs: (
            payloads.append(payload) or FakeChart()
        ),
    )
    monkeypatch.setattr(build_module, "load_settings", lambda: settings)
    monkeypatch.setattr(build_module, "ensure_analysis_checkpoint", lambda: object())
    split_analysis = object()
    monkeypatch.setattr(build_module, "load_cached_analysis", lambda: split_analysis)
    monkeypatch.setattr(
        build_module,
        "load_cached_component_persistence",
        lambda: (pair_table, {"counts": {}}),
    )
    monkeypatch.setattr(
        build_module, "load_full_network_fitness", lambda _: full_fitness
    )
    monkeypatch.setattr(build_module, "ROOT", root)
    monkeypatch.setattr(build_module, "OUTPUT", root / "output")
    provenance = write_analysis_record(root / "output", "generation-1")
    monkeypatch.setattr(build_module, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(
        build_module, "CHECKPOINT_DIR", root / "output/checkpoints/analysis"
    )
    monkeypatch.setattr(build_module, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(build_module, "RECORD_DIR", record_dir)
    install_fake_figures(monkeypatch, fake_module)

    import dubanevics2026topological.build.saved_figure_inputs as saved_inputs

    def _stub_threshold(ctx, settings=None):
        from dubanevics2026topological.threshold.functional_threshold import (
            CandidateThreshold,
        )

        c = CandidateThreshold(
            rule="median_reflected",
            decimals=6,
            median=0.0,
            global_minimum=-1.0,
            median_gap=1.0,
            unrounded=1.0,
            threshold=1.0,
        )
        return {
            "candidate": c,
            "lowest_split_saddle_death": None,
            "highest_join_saddle_death": None,
        }, "t" * 64

    monkeypatch.setattr(
        saved_inputs, "load_threshold_diagnostics_product", _stub_threshold
    )

    build_module.build_figures(
        ("full_tree_persistence",), variants=("vanilla",)
    )
    assert payloads[0]["analysis"] is split_analysis
    assert payloads[0]["pairs"] is pair_table
    assert payloads[0]["full_fitness"] is full_fitness

    payloads.clear()
    fitness_path.write_bytes(b"fitness-v2")
    build_module.build_figures(
        ("full_tree_persistence",), variants=("vanilla",)
    )
    assert len(payloads) == 1

    # The noise band takes its saddles from tree_saddles, so editing that module
    # re-renders the figure too.
    payloads.clear()
    saddle_source.write_text("# tree saddles, edited\n", encoding="utf-8")
    build_module.build_figures(
        ("full_tree_persistence",), variants=("vanilla",)
    )
    assert len(payloads) == 1


def test_failed_render_and_missing_output_are_retried(incremental_build) -> None:
    _, builds, fail_once, root = incremental_build
    fail_once.append(True)
    with pytest.raises(FiguresFailedError, match="full_tree_persistence"):
        build_figures(
            ("full_tree_persistence",), variants=("vanilla",)
        )
    output = root / "output/figures/canonical/full_tree_persistence_vanilla.svg"
    record = (
        root
        / "output/figures/.records/full_tree_persistence_vanilla.svg.json"
    )
    assert not output.exists()
    assert not record.exists()
    twin = root / "output/figures/canonical/titled/full_tree_persistence_vanilla.svg"
    twin_record = (
        root
        / "output/figures/.records/titled/full_tree_persistence_vanilla.svg.json"
    )
    assert not twin.exists()
    assert not twin_record.exists()
    builds.clear()
    assert (
        len(
            build_figures(
                ("full_tree_persistence",), variants=("vanilla",)
            )
        )
        == 2
    )
    assert builds == [("full_tree_persistence", "vanilla")]

    output.unlink()
    builds.clear()
    assert (
        len(
            build_figures(
                ("full_tree_persistence",), variants=("vanilla",)
            )
        )
        == 2
    )
    assert builds == [("full_tree_persistence", "vanilla")]
    assert output.is_file()
    assert twin.is_file()
    assert twin_record.is_file()


def test_one_failing_figure_does_not_stop_the_others(
    incremental_build, capsys
) -> None:
    _, _, fail_once, root = incremental_build
    canonical = root / "output/figures/canonical"
    (canonical / "titled").mkdir(parents=True)
    stale = [
        canonical / "full_tree_persistence_vanilla.png",
        canonical / "titled" / "full_tree_persistence_vanilla.png",
        canonical / "full_tree_persistence_vanilla.svg",
    ]
    for path in stale:
        path.write_bytes(b"stale tracked render")
    records = [
        root / "output/figures/.records" / "full_tree_persistence_vanilla.svg.json",
        root / "output/figures/.records/titled" / "full_tree_persistence_vanilla.svg.json",
    ]
    for record in records:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{}", encoding="utf-8")
    fail_once.append(True)  # the first figure's first save raises
    with pytest.raises(FiguresFailedError, match="full_tree_persistence") as caught:
        build_figures(
            ("full_tree_persistence", "linked_tree_persistence"),
            variants=("vanilla",),
        )
    assert caught.value.figure_ids == ["full_tree_persistence"]
    captured = capsys.readouterr()
    assert "Rendering: full_tree_persistence" in captured.out
    assert "Rendering: linked_tree_persistence" in captured.out
    assert "Figures rendered: 2; failed: 1: full_tree_persistence" in captured.out
    assert (
        "FIGURE FAILED: full_tree_persistence (vanilla): RuntimeError: "
        "simulated renderer failure" in captured.err
    )
    assert "Traceback" in captured.err
    for path in stale:
        assert not path.exists()
    for record in records:
        assert not record.exists()  # no record may outlive its deleted render
    assert list(canonical.glob("linked_tree_persistence*.svg"))
    assert list((canonical / "titled").glob("linked_tree_persistence*.svg"))
    assert not list(canonical.glob(".*"))  # no temporary file left behind


def test_unknown_figure_selection_is_rejected(incremental_build) -> None:
    with pytest.raises(ValueError, match="Unknown figure id"):
        build_figures(("typo",))


def test_png_render_passes_configured_300_dpi_without_changing_svg_options(
    tmp_path: Path,
) -> None:
    from dubanevics2026topological.build import _render_one

    save_calls: list[tuple[str, dict[str, object]]] = []
    chart = FakeChart(save_calls=save_calls)
    _render_one(chart, tmp_path / "figure.png", "png", png_dpi=300)
    _render_one(chart, tmp_path / "figure.svg", "svg", png_dpi=300)

    assert save_calls == [("png", {"ppi": 300}), ("svg", {})]


def test_broken_checkpoint_is_recomputed(incremental_build, monkeypatch) -> None:
    import dubanevics2026topological.build as build_module

    _, _, _, root = incremental_build
    checkpoint_dir = root / "output/analysis"
    data_dir = checkpoint_dir / "data"
    data_dir.mkdir()
    (data_dir / "metadata.toml").write_text("present\n", encoding="utf-8")
    dependencies = [
        root / "data/raw/papkou2023_fitness_wt.parquet",
        root / "data/raw/papkou2023_fitness_diff.parquet",
        root / "config/analysis.toml",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "src/dubanevics2026topological/analysis/__init__.py",
    ]
    for dependency in dependencies:
        dependency.parent.mkdir(parents=True, exist_ok=True)
        dependency.write_text("input\n", encoding="utf-8")
    provenance = checkpoint_dir / "provenance.json"
    provenance.write_text(
        f'{{"created_at": "{datetime.now(UTC).isoformat()}"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_module, "ROOT", root)
    monkeypatch.setattr(build_module, "PROVENANCE_PATH", provenance)
    monkeypatch.setattr(
        build_module,
        "load_settings",
        lambda: {
            "analysis": {
                "dataset": {
                    "landscape": "full",
                    "fitness_wt": "data/raw/papkou2023_fitness_wt.parquet",
                    "fitness_diff": "data/raw/papkou2023_fitness_diff.parquet",
                },
                "tree": {"kind": "split"},
                "persistence": {"kind": "extended_h0"},
            }
        },
    )
    monkeypatch.setattr(
        build_module,
        "load_cached_analysis",
        lambda: (_ for _ in ()).throw(ValueError("damaged checkpoint")),
    )
    recomputes: list[bool] = []
    monkeypatch.setattr(
        build_module,
        "load_or_compute_analysis",
        lambda: recomputes.append(True) or "rebuilt",
    )
    assert REAL_ENSURE() == "rebuilt"
    assert recomputes == [True]


def test_table_endpoint_indexes_become_sequence_identities() -> None:
    tree = rx.PyDiGraph()
    first = tree.add_node({"dna_sequence": "GCCGATCTC", "vertex_index": 0})
    second = tree.add_node({"dna_sequence": "GCCGATCTA", "vertex_index": 1})
    analysis = SimpleNamespace(trees={"split": SimpleNamespace(tree=tree)})
    frame = pl.DataFrame(
        {
            "source_tree_node_index": [first],
            "target_tree_node_index": [second],
            "source_vertex_index": [0],
            "vertex_index": [0],
            "private_edge_index": [42],
            "branch_span": [0.25],
        }
    )
    exported = tables._public_table("branch_table", frame, analysis)
    assert exported.columns == [
        "source_tree_node_identity",
        "target_tree_node_identity",
        "source_vertex_identity",
        "vertex_identity",
        "branch_span",
    ]
    assert exported["source_tree_node_identity"].to_list() == ["GCCGATCTC"]
    assert exported["target_tree_node_identity"].to_list() == ["GCCGATCTA"]


def test_full_graph_dependency_includes_threshold_product_id(
    incremental_build, monkeypatch
):
    _, builds, _, root = incremental_build
    build_figures(("full_tree_persistence",), variants=("vanilla",))
    record = (
        root
        / "output/figures/.records/full_tree_persistence_vanilla.svg.json"
    )
    dep = json.loads(record.read_text(encoding="utf-8"))
    assert dep.get("saved_products", {}).get("threshold_diagnostics") == "t" * 64

    import dubanevics2026topological.build.saved_figure_inputs as saved_inputs

    def _stub_u(ctx, settings=None):
        from dubanevics2026topological.threshold.functional_threshold import (
            CandidateThreshold,
        )

        c = CandidateThreshold(
            rule="median_reflected",
            decimals=6,
            median=0.0,
            global_minimum=-1.0,
            median_gap=1.0,
            unrounded=1.0,
            threshold=1.0,
        )
        return {
            "candidate": c,
            "lowest_split_saddle_death": None,
            "highest_join_saddle_death": None,
        }, "u" * 64

    monkeypatch.setattr(saved_inputs, "load_threshold_diagnostics_product", _stub_u)
    builds.clear()
    build_figures(("full_tree_persistence",), variants=("vanilla",))
    assert len(builds) == 1
    assert builds == [("full_tree_persistence", "vanilla")]


def test_component_load_raises_when_lock_held(tmp_path, monkeypatch):
    from dubanevics2026topological.analysis import component_persistence as cp
    from dubanevics2026topological.analysis.checkpoint import checkpoint_lock

    context = RunContext(
        root=tmp_path,
        output_root=tmp_path / "output",
        env={},
    )
    # setup minimal files so not early fail
    (tmp_path / "output" / "analysis").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output" / "analysis" / "component_pairs.parquet").write_bytes(b"")
    (tmp_path / "output" / "analysis" / "component_diagnostics.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "output" / "analysis" / "component_provenance.json").write_text(
        "{}", encoding="utf-8"
    )
    with (
        checkpoint_lock(
            tmp_path / "output" / "analysis" / "component_persistence",
            "component analysis",
        ),
        pytest.raises(RuntimeError, match="already in use"),
    ):
        cp.load_or_compute_component_persistence(context=context)


def test_titled_twins_write_an_untitled_and_a_titled_render_each(incremental_build) -> None:
    _, builds, _, root = incremental_build
    figures = root / "output" / "figures"
    paths = build_figures(("full_tree_persistence",))
    # two variants x two formats, untitled paths only: the twins sit beside them
    assert len(paths) == 4
    assert all(path.parent == figures / "canonical" for path in paths)
    for stem in ("full_tree_persistence", "full_tree_persistence_vanilla"):
        for ext in ("svg", "png"):
            assert (figures / "canonical" / f"{stem}.{ext}").is_file()
            assert (figures / "canonical" / "titled" / f"{stem}.{ext}").is_file()
            assert (figures / ".records" / f"{stem}.{ext}.json").is_file()
            assert (figures / ".records" / "titled" / f"{stem}.{ext}.json").is_file()
    # the chart is built once per variant, not once per twin
    assert len(builds) == 2
    builds.clear()
    assert build_figures(("full_tree_persistence",)) == paths
    # Figures are not cached, so the second build draws both variants again.
    assert len(builds) == 2


def test_opting_out_of_twins_writes_only_the_untitled_render(incremental_build) -> None:
    _, _, _, root = incremental_build
    figures = root / "output" / "figures"
    build_figures(("full_tree_persistence",), titled_twins=False)
    assert (figures / "canonical" / "full_tree_persistence.png").is_file()
    assert not (figures / "canonical" / "titled").exists()


def test_a_figure_that_cannot_be_stripped_leaves_no_stale_twin(
    incremental_build, monkeypatch
) -> None:
    _, _, _, root = incremental_build
    figures = root / "output" / "figures"
    build_figures(("full_tree_persistence",), formats=("png",))
    twin = figures / "canonical" / "titled" / "full_tree_persistence.png"
    twin_record = figures / ".records" / "titled" / "full_tree_persistence.png.json"
    assert twin.is_file() and twin_record.is_file()
    build_module = sys.modules["dubanevics2026topological.build"]
    monkeypatch.setattr(build_module, "without_headings", lambda chart: None)
    paths = build_figures(("full_tree_persistence",), formats=("png",), force=True)
    assert paths  # nothing raised; canonical keeps today's chart
    assert not twin.exists()
    assert not twin_record.exists()
