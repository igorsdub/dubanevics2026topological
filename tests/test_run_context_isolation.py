"""Spec 1 Section 6: Variant and Execution Order Isolation Test.

Verifies that:
1. Executing Published BY pipeline followed by Median-reflected BY pipeline, and the reverse
   order in a fresh workspace, completes without cross-contamination.
2. Neither execution writes into the other variant's output directory or into the default `output/`.
3. Checksums of the processed files under `data/processed/` remain completely unchanged.
4. Lookup by RunContext retrieves the correct scientific product under both execution orders.
5. All executions use fresh subprocesses per variant.

The checksum snapshot detects file content changes and new files, not identical-content
rewrites or empty directories; directory-name snapshots separately detect new directories.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from dubanevics2026topological.results import (
    RunContext,
    read_manifest,
)
from dubanevics2026topological.results.context import REPO_ROOT

PUBLISHED_VARIANT = "fdr_by_q0.05_papkou_original"
REFLECTED_VARIANT = "fdr_by_q0.05_median_reflected"


def _snapshot_directory_checksums(directory: Path) -> dict[str, str]:
    """Compute sha256 checksums of all regular files in a directory tree."""
    digests: dict[str, str] = {}
    if not directory.exists():
        return digests
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(65536):
                    hasher.update(chunk)
            rel = path.relative_to(directory).as_posix()
            digests[rel] = hasher.hexdigest()
    return digests


def _snapshot_directory_names(directory: Path) -> set[str]:
    """Return relative directory names, including empty directories."""
    if not directory.exists():
        return set()
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_dir()
    }


_SUBPROCESS_WORKER_SCRIPT = """
import hashlib
import os
import sys
from pathlib import Path

from dubanevics2026topological.results import (
    Manifest,
    RunContext,
    format_label,
    write_manifest,
)
from dubanevics2026topological.results.context import (
    ENV_OUTPUT_LEGACY,
    ENV_OUTPUT_ROOT,
)
from dubanevics2026topological.results.hashing import (
    build_identity_document,
    compute_scientific_hash,
)

# 1. Instantiate RunContext from environment variables (Spec 1)
ctx = RunContext()
expected_output = ctx.output_root

# Simulate a process where the package was imported before an explicit CLI output
# override was supplied: legacy module globals must remain bound to default output.
os.environ.pop(ENV_OUTPUT_ROOT, None)
os.environ.pop(ENV_OUTPUT_LEGACY, None)

from dubanevics2026topological.analysis import ROOT, load_settings
from dubanevics2026topological.analysis import component_persistence
from dubanevics2026topological.threshold.fdr_selection import load_fdr_snapshot

assert ctx.output_root == expected_output, f"Output root mismatch: {ctx.output_root} vs {expected_output}"
assert ctx.no_draw_allowed is True, "Expected no_draw_allowed to be True"
assert ctx.variant_name == os.environ["FITNESSTREE_ANALYSIS_VARIANT"]

# 2. Load settings under this explicit context
settings = load_settings(context=ctx)
clamped = settings["analysis"]["clamped"]["candidate_functional_threshold"]
assert clamped["rule"] == "fdr_fixed_cutoff"
assert clamped["method"] == "by"
assert clamped["q"] == 0.05

is_published = ctx.variant_name == "fdr_by_q0.05_papkou_original"
expected_cutoff = "papkou_original" if is_published else "median_reflected"
expected_thr = "published" if is_published else "reflected"
assert clamped["cutoff"] == expected_cutoff

# 3. Read the FDR snapshot from reference data (fail-closed, zero draws)
snapshot = load_fdr_snapshot(clamped)
assert snapshot.summary["cutoff"] == expected_cutoff
assert snapshot.summary["method"] == "by"
assert snapshot.summary["q"] == 0.05

# 4. Exercise a migrated production stage's cache/publication path with tiny fake
# science. The stage still performs its real output-root-directed writes.
import polars as pl

fake_pairs = pl.DataFrame(
    {
        "pair_id": [0],
        "pair_kind": ["extended_positive"],
        "homology_dimension": [0],
        "birth_sequence": ["AAAAAA"],
        "death_sequence": ["CCCCCC"],
        "birth_has_stop_codon": [False],
        "death_has_stop_codon": [False],
        "birth_scalar": [0.0],
        "death_scalar": [1.0],
        "persistence": [1.0],
    }
)
component_persistence.compute_component_persistence_analysis = lambda settings: (
    fake_pairs,
    {
        "schema": component_persistence.DIAGNOSTICS_SCHEMA,
        "pair_count": 1,
        "ordinary_pair_count": 0,
        "ordinary_death_value_frequencies": [],
        "join_tree_trailing_saddle": None,
        "global_minimum": 0.0,
    },
)
component_persistence.scientific_fingerprint = lambda settings: "tiny-fake-stage"
stage_pairs, _ = component_persistence.load_or_compute_component_persistence(
    settings=settings,
    context=ctx,
)
assert stage_pairs.height == 1
for stage_file in (
    "component_pairs.parquet",
    "component_diagnostics.json",
    "component_provenance.json",
):
    assert (ctx.output_root / "analysis" / stage_file).is_file()

# 5. Exercise the CLI dispatch and real table exporter with a tiny synthetic result.
# The package's import-time OUTPUT above remains the default root; only the CLI
# RunContext and exporter can route these files to ctx.output_root.
from types import SimpleNamespace

from dubanevics2026topological.analysis import sequence_growth_rates
from dubanevics2026topological import build as build_module

fake_rates = SimpleNamespace(
    growth_rates=pl.DataFrame({"dna_sequence": ["AAAAAA"], "growth_rate": [1.0]}),
    provenance={"fingerprint": {"synthetic": True}, "artifacts": {}},
    diagnostics={"sequence_count": 1, "max_identity_gap": 0.0},
)
sequence_growth_rates.compute_sequence_growth_rates = lambda *args, **kwargs: fake_rates
build_module.main(["--output-root", str(ctx.output_root), "sequence-growth-rates"])
growth_table = ctx.output_root / "tables" / "papkou2023_sequence_growth_rates.csv"
assert growth_table.is_file()
assert growth_table.with_suffix(".parquet").is_file()
assert (
    ctx.output_root / "tables" / ".sequence-growth-rates-provenance.json"
).is_file()

# 6. Generate scientific identity and manifest for the fitness product
scientific_identity = {
    "fdr": "by",
    "q": 0.05,
    "thr": expected_thr,
}
scientific_id_sha256, short_hash = compute_scientific_hash(
    product_type="fitness",
    scientific_parameters=scientific_identity,
)

label = format_label(
    product_name="fitness",
    params={
        "thr": expected_thr,
        "fdr": "by",
        "q": 0.05,
        "hash": short_hash,
    },
)

# 5. Write product output into context.output_root
product_dir = ctx.product_dir("fitness", label)
product_dir.mkdir(parents=True, exist_ok=True)

# Write summary artifact
summary_file = product_dir / "summary.json"
summary_bytes = f'{{"variant": "{ctx.variant_name}", "cutoff": "{expected_cutoff}", "rows": {snapshot.table.height}}}\\n'.encode("utf-8")
summary_file.write_bytes(summary_bytes)
summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()

# Write checkpoint and tables in context root
(ctx.checkpoint_dir("analysis")).mkdir(parents=True, exist_ok=True)
checkpoint_file = ctx.checkpoint_dir("analysis") / "provenance.json"
checkpoint_file.write_text(f'{{"variant": "{ctx.variant_name}"}}\\n', encoding="utf-8")

(ctx.tables_dir).mkdir(parents=True, exist_ok=True)
table_file = ctx.tables_dir / f"table_{expected_thr}.tsv"
table_file.write_text(f"variant\\t{ctx.variant_name}\\n", encoding="utf-8")

# Write immutable manifest
manifest = Manifest(
    product_type="fitness",
    scientific_identity=scientific_identity,
    scientific_id_sha256=scientific_id_sha256,
    parents=[],
    data_provenance={snapshot.source["table"]: snapshot.source["table_sha256"]},
    artifacts={
        "summary.json": {
            "bytes": len(summary_bytes),
            "sha256": summary_sha256,
        }
    },
    run_fields={"host": ctx.host, "run": ctx.run_id or "test"},
    code_provenance={"commit": "HEAD", "dirty": False, "branch": "test"},
    environment={"platform": "test", "python_version": "3.13", "uv_lock_sha256": "none"},
    created_at="2026-10-27T00:00:00Z",
    evidence_availability={"checkpoint_present": True, "distances_present": False, "chunks_present": False},
)
write_manifest(manifest, product_dir)

print(f"SUCCESS:{ctx.variant_name}:{label}")
"""


def _run_variant_subprocess(
    *,
    context: RunContext,
) -> str:
    """Execute the pipeline worker script in a fresh, isolated Python subprocess."""
    env = {
        **os.environ,
        **context.env_vars(),
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_WORKER_SCRIPT],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Subprocess failed ({result.returncode}).\n"
            f"Output:\n{result.stdout}\nErrors:\n{result.stderr}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("SUCCESS:"):
            return line.strip()
    raise RuntimeError(
        f"Subprocess did not output success line. Output:\n{result.stdout}\nErrors:\n{result.stderr}"
    )


def test_variant_and_execution_order_isolation(tmp_path: Path) -> None:
    """Execute Published BY then Median-reflected BY, and the reverse in a fresh workspace."""
    default_output = REPO_ROOT / "output"
    initial_default_output = _snapshot_directory_checksums(default_output)
    initial_default_directories = _snapshot_directory_names(default_output)
    processed_dir = Path(os.environ["FITNESSTREE_PROCESSED_ROOT"])
    initial_processed_checksums = _snapshot_directory_checksums(processed_dir)
    assert len(initial_processed_checksums) > 0, (
        "Processed data directory must not be empty"
    )

    # =========================================================================
    # Workflow 1: Published BY -> Median-reflected BY
    # =========================================================================
    workspace_1 = tmp_path / "order1"
    root_1_pub = workspace_1 / "output_published"
    root_1_med = workspace_1 / "output_reflected"

    ctx_1_pub = RunContext(
        output_root=root_1_pub,
        variant_name=PUBLISHED_VARIANT,
        no_draw_allowed=True,
    )
    ctx_1_med = RunContext(
        output_root=root_1_med,
        variant_name=REFLECTED_VARIANT,
        no_draw_allowed=True,
    )

    # 1.1 Run Published BY
    out_msg = _run_variant_subprocess(context=ctx_1_pub)
    assert PUBLISHED_VARIANT in out_msg
    assert "thr=published" in out_msg

    # Assert Root A has output and Root B does NOT exist
    assert root_1_pub.is_dir()
    assert not root_1_med.exists()
    assert _snapshot_directory_checksums(default_output) == initial_default_output
    assert _snapshot_directory_names(default_output) == initial_default_directories

    snapshot_1_pub_before = _snapshot_directory_checksums(root_1_pub)
    directories_1_pub_before = _snapshot_directory_names(root_1_pub)
    assert len(snapshot_1_pub_before) > 0

    # 1.2 Run Median-reflected BY
    out_msg_med = _run_variant_subprocess(context=ctx_1_med)
    assert REFLECTED_VARIANT in out_msg_med
    assert "thr=reflected" in out_msg_med

    # Assert Root B exists now, Root A was NOT modified, and default output untouched
    assert root_1_med.is_dir()
    snapshot_1_pub_after = _snapshot_directory_checksums(root_1_pub)
    assert snapshot_1_pub_before == snapshot_1_pub_after, (
        "Median-reflected run mutated Published output root!"
    )
    assert _snapshot_directory_names(root_1_pub) == directories_1_pub_before
    assert _snapshot_directory_checksums(default_output) == initial_default_output
    assert _snapshot_directory_names(default_output) == initial_default_directories

    # 1.3 Lookup by context in Order 1
    # Locate published product
    pub_labels = [p.name for p in (root_1_pub / "fitness").iterdir() if p.is_dir()]
    assert len(pub_labels) == 1
    pub_label = pub_labels[0]
    pub_manifest = read_manifest(ctx_1_pub.product_dir("fitness", pub_label))
    assert pub_manifest.scientific_identity["thr"] == "published"
    assert pub_manifest.scientific_identity["fdr"] == "by"

    # Locate reflected product
    med_labels = [p.name for p in (root_1_med / "fitness").iterdir() if p.is_dir()]
    assert len(med_labels) == 1
    med_label = med_labels[0]
    med_manifest = read_manifest(ctx_1_med.product_dir("fitness", med_label))
    assert med_manifest.scientific_identity["thr"] == "reflected"
    assert med_manifest.scientific_identity["fdr"] == "by"

    # Verify no cross-lookup: ctx_1_pub cannot find med_label and vice versa
    assert not ctx_1_pub.product_dir("fitness", med_label).exists()
    assert not ctx_1_med.product_dir("fitness", pub_label).exists()

    # =========================================================================
    # Workflow 2: Fresh workspace, Median-reflected BY -> Published BY
    # =========================================================================
    workspace_2 = tmp_path / "order2"
    root_2_med = workspace_2 / "output_reflected"
    root_2_pub = workspace_2 / "output_published"

    ctx_2_med = RunContext(
        output_root=root_2_med,
        variant_name=REFLECTED_VARIANT,
        no_draw_allowed=True,
    )
    ctx_2_pub = RunContext(
        output_root=root_2_pub,
        variant_name=PUBLISHED_VARIANT,
        no_draw_allowed=True,
    )

    # 2.1 Run Median-reflected BY first
    out_msg_2_med = _run_variant_subprocess(context=ctx_2_med)
    assert REFLECTED_VARIANT in out_msg_2_med
    assert "thr=reflected" in out_msg_2_med

    # Assert Root C has output and Root D does NOT exist
    assert root_2_med.is_dir()
    assert not root_2_pub.exists()
    assert _snapshot_directory_checksums(default_output) == initial_default_output
    assert _snapshot_directory_names(default_output) == initial_default_directories

    snapshot_2_med_before = _snapshot_directory_checksums(root_2_med)
    directories_2_med_before = _snapshot_directory_names(root_2_med)
    assert len(snapshot_2_med_before) > 0

    # 2.2 Run Published BY second
    out_msg_2_pub = _run_variant_subprocess(context=ctx_2_pub)
    assert PUBLISHED_VARIANT in out_msg_2_pub
    assert "thr=published" in out_msg_2_pub

    # Assert Root D exists now, Root C was NOT modified, and default output untouched
    assert root_2_pub.is_dir()
    snapshot_2_med_after = _snapshot_directory_checksums(root_2_med)
    assert snapshot_2_med_before == snapshot_2_med_after, (
        "Published run mutated Median-reflected output root!"
    )
    assert _snapshot_directory_names(root_2_med) == directories_2_med_before
    assert _snapshot_directory_checksums(default_output) == initial_default_output
    assert _snapshot_directory_names(default_output) == initial_default_directories

    # 2.3 Lookup by context in Order 2
    pub_labels_2 = [p.name for p in (root_2_pub / "fitness").iterdir() if p.is_dir()]
    assert len(pub_labels_2) == 1
    pub_manifest_2 = read_manifest(ctx_2_pub.product_dir("fitness", pub_labels_2[0]))
    assert pub_manifest_2.scientific_identity["thr"] == "published"
    assert pub_manifest_2.scientific_id_sha256 == pub_manifest.scientific_id_sha256

    med_labels_2 = [p.name for p in (root_2_med / "fitness").iterdir() if p.is_dir()]
    assert len(med_labels_2) == 1
    med_manifest_2 = read_manifest(ctx_2_med.product_dir("fitness", med_labels_2[0]))
    assert med_manifest_2.scientific_identity["thr"] == "reflected"
    assert med_manifest_2.scientific_id_sha256 == med_manifest.scientific_id_sha256

    # Verify no cross-lookup in Order 2
    assert not ctx_2_pub.product_dir("fitness", med_labels_2[0]).exists()
    assert not ctx_2_med.product_dir("fitness", pub_labels_2[0]).exists()

    # =========================================================================
    # Global Assertion: data/processed checksums completely unchanged
    # =========================================================================
    final_processed_checksums = _snapshot_directory_checksums(processed_dir)
    assert initial_processed_checksums == final_processed_checksums, (
        "Files under data/processed/ were modified or corrupted during execution!"
    )
