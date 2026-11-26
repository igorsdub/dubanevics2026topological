"""The ``just figures`` contract: what is collected into data/processed is what the figures read.

A toy ``output/`` is built with every kind of product the figures read, every location named
by the code's own path functions and constants (so a moved or newly read path fails here), a
symlinked ``analysis/data`` and the hidden stage records. It is collected into a toy processed
folder, the manifest is written and checked, ``output/`` is wiped and restored, and then each
path the figure loaders read must be back with its bytes, the real readers that need nothing
but files must load, and every default figure input must be accounted for.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.analysis import band_quantile_scan_panels as scan_panels
from dubanevics2026topological.analysis import manuscript_quantities as quantities
from dubanevics2026topological.analysis.candidate_edge_evidence import (
    cache_dir as edge_cache_dir,
)
from dubanevics2026topological.analysis.clamped_cache import (
    analysis_provenance_path,
    cache_dirs as comparison_dirs,
)
from dubanevics2026topological.analysis.component_persistence import component_paths
from dubanevics2026topological.analysis.describe import load_saved_annotations
from dubanevics2026topological.analysis.peak_uncertainty import cache_paths as peak_paths
from dubanevics2026topological.build import figures as build_figures
from dubanevics2026topological.build.common import ANNOTATION_TABLE_DIRECTORIES
from dubanevics2026topological.build.functional_counts_vs_q import (
    load_fdr_q_scan,
    scan_path,
)
from dubanevics2026topological.build.peak_panels import (
    PEAK_PANEL_RULES,
    variant_context as peak_variant_context,
)
from dubanevics2026topological.figures._four_graphs import VARIANTS as FOUR_GRAPH_VARIANTS
from dubanevics2026topological.results import processed
from dubanevics2026topological.results.context import RunContext
from dubanevics2026topological.results.hashing import compute_scientific_hash
from dubanevics2026topological.results.manifest import Manifest, write_manifest

REFLECTED_BY = "threshold=reflected_fdr=by_q=0.05"
PUBLISHED_BY = "threshold=published_fdr=by_q=0.05"


def _context(root: Path) -> RunContext:
    return RunContext(
        root=root,
        output_root=root / "output",
        processed_root=root / "data" / "processed",
        raw_data_dir=root / "data" / "raw",
        env={},
    )


def _product(output_root: Path, kind: str, name: str, identity: dict) -> None:
    target = output_root / kind / name
    target.mkdir(parents=True)
    (target / "data.txt").write_text(name)
    digest, _ = compute_scientific_hash(kind, identity)
    write_manifest(
        Manifest(
            product_type=kind,
            scientific_identity=identity,
            scientific_id_sha256=digest,
            parents=[],
            data_provenance={},
            artifacts={
                "data.txt": {
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "size_bytes": len(name),
                }
            },
            run_fields={},
            code_provenance={},
            environment={},
            created_at="2026-01-01T00:00:00Z",
            evidence_availability={
                "checkpoint_present": False,
                "distances_present": False,
                "chunks_present": False,
            },
        ),
        target,
        check_collision=False,
    )


class Toy:
    """A toy output root; ``expect`` records the files that must survive collect and restore."""

    def __init__(self, root: Path) -> None:
        self.context = _context(root)
        self.output = self.context.output_root
        self.expected: dict[Path, bytes] = {}
        self.raw: dict[Path, bytes] = {}

    def write(self, path: Path, data: bytes | str = b"x", *, expect: bool = True) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = data.encode() if isinstance(data, str) else data
        path.write_bytes(content)
        if expect:
            self.expected[path] = content
        return path


def _frame(path: Path, **columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_parquet(path)


def _build_toy_output(toy: Toy) -> None:
    out, context = toy.output, toy.context
    # -- manifest products, by state, by variant root and single
    for state_identity in (
        {"graph": "full", "tree": "nominal"},
        {"thr": "reflected", "fdr": "by", "q": 0.05, "tree": "nominal"},
        {"thr": "reflected", "fdr": "by", "q": 0.05, "tree": "band", "bandq": 0.95},
    ):
        for kind in processed.STATE_KINDS:
            _product(out, kind, json.dumps(state_identity, sort_keys=True), state_identity)
    identity = {"thr": "reflected", "fdr": "by", "q": 0.05}
    for kind in processed.VARIANT_KINDS:
        _product(out, kind, "default-root", {**identity, "root": "default"})
        _product(out / "variants" / REFLECTED_BY, kind, "variant-root", {**identity, "root": "variant"})
    for kind in processed.SINGLE_KINDS:
        _product(out, kind, "single", {"kind": kind})
    # -- plain file sets (hidden files and a symlinked directory included)
    for file_set in processed.FILE_SETS:
        name = file_set.patterns[0].replace("**/", "nested/").replace("*", "x")
        toy.write(out / file_set.output_dir / name, f"{file_set.kind}/{file_set.state}")
    toy.write(out / "analysis/comparison_extra/.hidden", expect=False)
    toy.write(out / "tables" / "unrelated.csv", expect=False)
    # -- the Full analysis: a checkpoint, and output/analysis/data as a link to its data
    leaf = out / "checkpoints" / "analysis" / "key=abcd1234"
    toy.write(leaf / "provenance.json", '{"fingerprint": "f"}')
    toy.write(leaf / "data" / "metadata.toml", "x = 1", expect=False)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "data").symlink_to(Path("..") / "checkpoints" / "analysis" / "key=abcd1234" / "data")
    toy.expected[out / "analysis" / "data" / "metadata.toml"] = b"x = 1"
    toy.write(out / "analysis" / "provenance.json", '{"fingerprint": "f"}')
    # -- the saved comparison, component, peak and edge-evidence checkpoints
    comparison, pending = comparison_dirs(context)
    toy.write(comparison / "provenance.json")
    toy.write(comparison / "tables" / ".nested-hidden")
    toy.write(pending, expect=False)  # pending markers are not deposited
    for path in component_paths(context)[1:4]:
        toy.write(path)
    toy.write(component_paths(context).pending, expect=False)
    toy.write(peak_paths(context)[0] / "tables" / "scatter.parquet")
    toy.write(peak_paths(context)[0] / "diagnostics.json", "{}")
    toy.write(edge_cache_dir(context) / "provenance.json")
    toy.write(edge_cache_dir(context) / "edges.parquet")
    # -- stage checkpoints and the registry's other upstream outputs
    for part in (quantities.BAND_DIR, quantities.FUNCTIONAL_DIR):
        toy.write(out.joinpath(*part) / "key=1234abcd" / "provenance.json")
    toy.write(out.joinpath(*quantities.SCAN_DIR) / "acceptance.json")
    toy.write(out.joinpath(*quantities.BRANCH_DIR) / "provenance.json")
    toy.write(out.joinpath("tables", quantities.RECORD_NAME), '{"fingerprint": "r"}')
    toy.write(out / "tables" / quantities.CODON_SIDECAR)
    toy.write(out / "tables" / "papkou2023_codon_entropy_metadata.json", "{}")
    # -- the biological annotation tables of both landscapes
    for location in ANNOTATION_TABLE_DIRECTORIES.values():
        directory = out / "tables" / location
        for name in ("branches", "pairs", "critical_points"):
            _frame(directory / f"biological_annotation_{name}.parquet", a=[1])
            toy.expected[directory / f"biological_annotation_{name}.parquet"] = (
                directory / f"biological_annotation_{name}.parquet"
            ).read_bytes()
        toy.write(directory / quantities.ANNOTATION_COUNTS, "{}")
        toy.write(directory / quantities.ANNOTATION_SIDECAR)
    # -- the combined FDR scan
    _frame(scan_path(context), cutoff=["a"], fdr_target=[0.05])
    toy.expected[scan_path(context)] = scan_path(context).read_bytes()
    # -- every variant root the four-graph and peak-panel figures read
    for variant in sorted({*FOUR_GRAPH_VARIANTS, *PEAK_PANEL_RULES, REFLECTED_BY, PUBLISHED_BY}):
        scoped = RunContext(
            root=context.root,
            output_root=out / "variants" / variant,
            processed_root=context.processed_root,
            raw_data_dir=context.raw_data_dir,
            env={},
        )
        toy.write(analysis_provenance_path(scoped), '{"fingerprint": "v"}')
        toy.write(comparison_dirs(scoped)[0] / "provenance.json", variant)
        toy.write(peak_paths(scoped)[0] / "tables" / "scatter.parquet", variant)
        toy.write(peak_paths(scoped)[0] / "diagnostics.json", "{}")
    # -- the four raw files (outside output/) and their README
    for name in (*processed.RAW_FILES, processed.RAW_README):
        toy.raw[context.raw_data_dir / name] = toy.write(
            context.raw_data_dir / name, f"raw {name}", expect=False
        ).read_bytes()
    # -- the band-quantile scan tables the stage publishes directly into processed
    for state in ("published", "published_by", "reflected", "reflected_by"):
        directory = context.processed_root / scan_panels.SCAN_DIR_NAME / state
        directory.mkdir(parents=True)


def test_collect_wipe_restore_brings_back_every_figure_input(tmp_path: Path) -> None:
    toy = Toy(tmp_path)
    _build_toy_output(toy)
    context = toy.context
    relative = {path.relative_to(toy.output): data for path, data in toy.expected.items()}
    assert any(r.parts[:2] == ("analysis", "data") for r in relative)  # through the link

    processed.collect(context)
    processed.write_deposit_readme(context.processed_root)
    processed.write_manifest(context.processed_root)
    assert processed.check_manifest(context.processed_root) == []
    manifest = (context.processed_root / processed.MANIFEST_NAME).read_text()
    assert quantities.ANNOTATION_SIDECAR in manifest  # dotfiles are listed
    assert ".clamped-comparison-pending" not in manifest
    assert "unrelated.csv" not in manifest
    # a symlinked directory is copied by its content, never as a link
    assert not any(p.is_symlink() for p in context.processed_root.rglob("*"))
    # the default root and the variant keep separate folders
    states = {p.name for p in (context.processed_root / "pooled_scatter").iterdir()}
    assert states == {"default", "reflected_by"}
    for kind in processed.STATE_KINDS:
        assert processed.load(kind, "full", context).report.status == "full"

    assert (context.processed_root / "README.md").read_text().count("CC-BY-4.0") == 1
    for name in processed.RAW_FILES:
        assert (context.processed_root / "raw" / name).is_file()

    shutil.rmtree(toy.output)
    shutil.rmtree(context.raw_data_dir)
    assert processed.missing_raw(context) == []  # the deposit holds them
    processed.restore(context)
    for path, data in toy.raw.items():
        assert path.read_bytes() == data, f"raw not restored: {path.name}"
    for path, data in relative.items():
        restored = toy.output / path
        assert restored.is_file(), f"not restored: {path}"
        assert restored.read_bytes() == data, f"changed: {path}"
    assert not comparison_dirs(context)[1].exists()
    assert processed.restore(context) == []  # idempotent


def test_restored_output_serves_the_file_readers_of_the_figures(tmp_path: Path) -> None:
    toy = Toy(tmp_path)
    _build_toy_output(toy)
    context = toy.context
    processed.collect(context)
    processed.write_manifest(context.processed_root)
    shutil.rmtree(toy.output)
    processed.restore(context)

    # the combined FDR scan (fdr_q_scan) and the annotation tables (annotations)
    assert load_fdr_q_scan(context)["scan"].height == 1
    for location in ANNOTATION_TABLE_DIRECTORIES.values():
        directory = toy.output / "tables" / location
        loaded = load_saved_annotations(directory)
        assert loaded.pairs.height == 1
    # the checkpoint locations the loaders resolve through the code's own functions
    assert (toy.output / "analysis" / "data" / "metadata.toml").is_file()
    assert comparison_dirs(context)[0].is_dir()
    assert all(path.is_file() for path in component_paths(context)[1:4])
    assert (peak_paths(context)[0] / "diagnostics.json").is_file()
    assert (edge_cache_dir(context) / "edges.parquet").is_file()
    assert (toy.output / "tables" / quantities.RECORD_NAME).is_file()
    for variant in FOUR_GRAPH_VARIANTS:
        scoped = peak_variant_context(context, variant) if variant in PEAK_PANEL_RULES else None
        assert scoped is not None
        assert analysis_provenance_path(scoped).is_file()
        assert comparison_dirs(scoped)[0].is_dir()
        assert (peak_paths(scoped)[0] / "diagnostics.json").is_file()


def test_figures_precondition_names_missing_raw_files(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert processed.missing_raw(context) == list(processed.RAW_FILES)
    (context.processed_root / "raw").mkdir(parents=True)
    (context.processed_root / "raw" / processed.RAW_FILES[0]).write_text("x")
    assert processed.missing_raw(context) == list(processed.RAW_FILES[1:])


# What each default figure input reads, and where this contract covers it. A new default
# input, or a renamed one, fails the test below until it is accounted for here.
INPUT_COVERAGE = {
    "annotations": "tables/ annotation tables (stage_outputs)",
    "band_leaf_tables": "processed leaves/tree products and bootstrap/<state>",
    "band_quantile_scan_grid": "processed band_quantile_scan/<state>",
    "bootstrap_band_quantile_scan": "processed band_quantile_scan/<state> product",
    "bootstrap_codon_entropy": "processed bootstrap_codon_entropy",
    "bootstrap_smoothed": "processed bootstrap/<state>/checkpoint (restored by its loader)",
    "comparison": "analysis/clamped_comparison (comparison/default) and analysis/data",
    "component_persistence": "analysis/component_* (stage_outputs)",
    "edge_evidence": "analysis/candidate_edge_evidence (stage_outputs)",
    "fdr_q_scan": "analysis/classification_probability_scans scan parquet (stage_outputs)",
    "four_graph_comparisons": "variants/<v>/analysis/clamped_comparison (comparison/<state>)",
    "full_fitness": "data/raw",
    "graph_tree_persistence": "processed graph_tree_persistence",
    "graph_codon_entropy": "processed landscape/full and landscape/published",
    "growth_rates": "processed growth_rates",
    "manuscript_quantities": "tables/manuscript_quantities and every registry upstream",
    "peak_panels": "variants/<v>/analysis/peak_uncertainty and peak_panels/<state>",
    "published_fitness": "data/raw",
    "reflected_fdr_trees": "processed tree/leaves/bootstrap products",
    "split_analysis": "analysis/data and checkpoints/analysis (stage_outputs)",
    "threshold_diagnostics": "processed threshold_diagnostics",
    "tree_views": "processed tree/leaves/tree_layout products",
}


def test_every_default_figure_input_is_accounted_for() -> None:
    specs = {spec.figure_id: spec for spec in build_figures.FIGURE_SPECS}
    chosen = build_figures.default_selection(specs)
    needed = set().union(*(specs[figure_id].inputs for figure_id in chosen))
    assert needed <= set(INPUT_COVERAGE), sorted(needed - set(INPUT_COVERAGE))


def test_restore_replaces_analysis_data_with_the_checkpoint_link(tmp_path: Path) -> None:
    toy = Toy(tmp_path)
    _build_toy_output(toy)
    context = toy.context
    processed.collect(context)
    # the data is stored once, with the checkpoints
    stored = context.processed_root / processed.STAGE_OUTPUTS
    assert (stored / "checkpoints/analysis/key=abcd1234/data/metadata.toml").is_file()
    assert not (stored / "analysis" / "data").exists()
    shutil.rmtree(toy.output)
    processed.restore(context)
    link = toy.output / "analysis" / "data"
    assert link.is_symlink()
    target = Path(os.path.relpath(toy.output / "checkpoints/analysis/key=abcd1234/data", link.parent))
    # the analysis stage's own check: a link equal to the relative path to the checkpoint data
    assert Path(os.readlink(link)) == target
    assert (link / "metadata.toml").read_text() == "x = 1"
    assert processed.restore(context) == []


def test_collect_prunes_stale_files(tmp_path: Path) -> None:
    toy = Toy(tmp_path)
    _build_toy_output(toy)
    context = toy.context
    processed.collect(context)
    stale = context.processed_root / processed.STAGE_OUTPUTS / "tables" / "stale.json"
    stale.write_text("x")
    kind_stale = context.processed_root / "pooled_scatter" / "default" / "stale.txt"
    kind_stale.write_text("x")
    processed.collect(context)
    assert not stale.exists()
    assert not kind_stale.exists()
    assert (context.processed_root / "band_quantile_scan" / "reflected_by").is_dir()


def test_restore_does_not_write_through_a_symlinked_directory(tmp_path: Path) -> None:
    toy = Toy(tmp_path)
    _build_toy_output(toy)
    context = toy.context
    processed.collect(context)
    shutil.rmtree(toy.output)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (toy.output / "analysis").mkdir(parents=True)
    (toy.output / "analysis" / "component_extra").symlink_to(outside)
    stored = context.processed_root / processed.STAGE_OUTPUTS / "analysis" / "component_extra"
    stored.mkdir(parents=True)
    (stored / "leak.txt").write_text("x")
    with pytest.warns(UserWarning, match="symlinked"):
        processed._restore_mirror(toy.output, context.processed_root)
    assert list(outside.iterdir()) == []


def _loader_outcomes(context: RunContext, names: list[str]) -> dict[str, str]:
    """What each real figure input loader does on this root: ``ok`` or the error it raises."""
    from dubanevics2026topological.analysis import load_settings

    settings = load_settings(context=context)
    outcome = {}
    for name in names:
        try:
            build_figures.load_inputs([name], settings, context=context)
            outcome[name] = "ok"
        except Exception as error:  # noqa: BLE001 - the outcome is the observation
            outcome[name] = f"{type(error).__name__}: {error}"
    return outcome


def test_round_trip_gives_every_default_figure_loader_the_same_outcome(tmp_path: Path) -> None:
    """collect, wipe, restore: each real default-figure loader behaves as on the original root.

    The toy root holds placeholder bytes for most checkpoints (a real network analysis,
    a real growth-rate table, real raw parquet), so those loaders raise on the toy root
    itself; the contract is that restore gives them the identical outcome (same result or
    same error) as the root it was collected from, and that the loaders the toy data does
    serve (``annotations``, ``fdr_q_scan``) succeed. Inputs in
    ``SAVED_PRODUCT_INPUTS`` are read in place from the processed folder, not through
    ``load_inputs``; the first test above covers them through ``processed.load``.
    """
    toy = Toy(tmp_path)
    _build_toy_output(toy)
    context = toy.context
    specs = {spec.figure_id: spec for spec in build_figures.FIGURE_SPECS}
    chosen = build_figures.default_selection(specs)
    names = sorted(
        set().union(*(specs[i].inputs for i in chosen)) - set(build_figures.SAVED_PRODUCT_INPUTS)
    )
    # pending markers are never deposited: compare against the root without them
    comparison_dirs(context)[1].unlink()
    component_paths(context).pending.unlink()
    before = _loader_outcomes(context, names)
    processed.collect(context)
    processed.write_manifest(context.processed_root)
    shutil.rmtree(toy.output)
    processed.restore(context)
    after = _loader_outcomes(context, names)
    assert after == before
    assert after["annotations"] == "ok" and after["fdr_q_scan"] == "ok"
