from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from dubanevics2026topological.analysis.produce_or_load import _declared
from dubanevics2026topological.simplification.identities import (
    band_identity,
    draw_identity,
    evidence_availability,
)


def test_band_quantile_changes_band_id_but_not_draw_id():
    args = {
        "selected_graph_sha256": "a" * 64,
        "growth_identity": {"config": {"elapsed_time": 3.5}},
        "fitness_input_sha256s": {"fitness.parquet": "b" * 64},
        "perturbation_method": "normal-pooled-se",
        "seed": 127,
        "draw_ids": range(50),
    }
    _, draw_id = draw_identity(**args)
    band_a, band_id_a = band_identity(
        draw_scientific_id_sha256=draw_id,
        bandq=0.95,
    )
    band_b, band_id_b = band_identity(
        draw_scientific_id_sha256=draw_id,
        bandq=0.90,
    )
    assert band_a["draw_scientific_id_sha256"] == band_b["draw_scientific_id_sha256"]
    assert band_id_a != band_id_b


def test_direct_band_identity_excludes_retired_settings():
    args = {
        "selected_graph_sha256": "a" * 64,
        "growth_identity": {"config": {"elapsed_time": 3.5}},
        "fitness_input_sha256s": {"fitness.parquet": "b" * 64},
        "perturbation_method": "normal-pooled-se",
        "seed": 127,
        "draw_ids": range(50),
    }
    _, draw_id = draw_identity(**args)
    band_args = {
        "draw_scientific_id_sha256": draw_id,
        "bandq": 0.95,
    }
    fields, _ = band_identity(**band_args)
    assert fields == {**band_args, "simplification_method": "direct_band"}
    _, draw_again = draw_identity(**args)
    assert draw_again == draw_id


def test_retired_core_settings_do_not_change_draw_fingerprint():
    from dubanevics2026topological.simplification.checkpoint_identity import (
        scientific_fingerprint,
    )

    direct = {"config": {"n_replicates": 50, "seed": 127, "noise_model": "pooled"}}
    old = {
        "config": {
            **direct["config"],
            "core_threshold": 0.95,
            "consensus_peak_frequency": 0.5,
            "band_quantile": 0.95,
        }
    }
    assert scientific_fingerprint(old) == scientific_fingerprint(direct)


def test_draw_identity_tracks_growth_and_selected_threshold_declaration():
    base = {
        "selected_graph_sha256": "a" * 64,
        "growth_identity": {"config": {"elapsed_time": 3.5}},
        "fitness_input_sha256s": {"fitness.parquet": "b" * 64},
        "perturbation_method": "normal-pooled-se",
        "seed": 127,
        "draw_ids": range(50),
    }
    _, baseline = draw_identity(**base)
    _, changed_growth = draw_identity(
        **{**base, "growth_identity": {"config": {"elapsed_time": 3.6}}}
    )
    # The caller's selected-graph digest covers both the selection and its
    # functional-threshold declaration.
    _, changed_threshold = draw_identity(**{**base, "selected_graph_sha256": "c" * 64})
    assert changed_growth != baseline
    assert changed_threshold != baseline


def test_checkpoint_draw_settings_exclude_band_quantile():
    from dubanevics2026topological.simplification import (
        checkpoint_identity,
        bootstrap_smoothed_tree,
    )

    first = {
        "n_replicates": 50,
        "seed": 127,
        "band_quantile": 0.95,
        "n_jobs": 1,
    }
    second = {**first, "band_quantile": 0.90}
    first_checkpoint = bootstrap_smoothed_tree._config_identity(first)
    second_checkpoint = bootstrap_smoothed_tree._config_identity(second)
    assert first_checkpoint != second_checkpoint
    assert checkpoint_identity.scientific_fingerprint({"config": first_checkpoint}) == (
        checkpoint_identity.scientific_fingerprint({"config": second_checkpoint})
    )
    assert bootstrap_smoothed_tree.result_settings(first) != (
        bootstrap_smoothed_tree.result_settings(second)
    )


def test_draw_identity_changes_with_seed_or_input():
    base = {
        "selected_graph_sha256": "a" * 64,
        "growth_identity": {"config": {"elapsed_time": 3.5}},
        "fitness_input_sha256s": {"fitness.parquet": "b" * 64},
        "perturbation_method": "normal-pooled-se",
        "seed": 127,
        "draw_ids": range(50),
    }
    _, identity = draw_identity(**base)
    _, different_seed = draw_identity(**{**base, "seed": 128})
    _, different_input = draw_identity(
        **{**base, "fitness_input_sha256s": {"fitness.parquet": "c" * 64}}
    )
    assert identity != different_seed
    assert identity != different_input


def test_evidence_availability_tracks_three_artifact_kinds(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "summary.json").touch()
    distances = tmp_path / "distances.parquet"
    distances.touch()
    chunk_manifest = tmp_path / "chunks.json"
    chunk_payload = tmp_path / "chunk.parquet"
    chunk_manifest.touch()
    state = evidence_availability(
        checkpoint=checkpoint,
        distances=[distances],
        chunks=[chunk_payload],
    )
    assert state == {
        "checkpoint_present": True,
        "distances_present": True,
        "chunks_present": False,
    }


def test_moved_modules_import_in_both_fresh_process_orders():
    snippets = (
        (
            "import dubanevics2026topological.simplification.bootstrap_chunks; "
            "import dubanevics2026topological.simplification.bootstrap_smoothed_tree"
        ),
        (
            "import dubanevics2026topological.simplification.bootstrap_smoothed_tree; "
            "import dubanevics2026topological.simplification.bootstrap_chunks"
        ),
    )
    for snippet in snippets:
        subprocess.run([sys.executable, "-c", snippet], check=True)


def test_bootstrap_stage_declares_real_moved_sources(tmp_path: Path):
    from dubanevics2026topological.simplification.bootstrap_smoothed_tree import (
        bootstrap_stage,
    )

    stage = bootstrap_stage({"fixture": True})
    moved_names = {
        "bootstrap_smoothed_tree.py",
        "bootstrap_chunks.py",
        "checkpoint_identity.py",
        "selected_leaf_simplification.py",
    }
    real_sources = [
        path
        for path in stage.code
        if path.name in moved_names
        and "simplification" in path.parts
        and path.is_file()
    ]
    assert {path.name for path in real_sources} == moved_names

    copied = []
    for source in real_sources:
        target = tmp_path / source.name
        shutil.copyfile(source, target)
        copied.append(target)
    isolated = replace(stage, code=tuple(copied), root=tmp_path)
    before = _declared(isolated)
    copied[0].write_text(copied[0].read_text() + "\n# changed\n")
    assert _declared(isolated) != before
