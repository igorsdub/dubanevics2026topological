"""The .gitignore allow-list of manuscript figures must match config/manuscript.toml."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "sync_figure_gitignore", ROOT / "scripts" / "dev" / "sync_figure_gitignore.py"
)
assert _spec and _spec.loader
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def test_gitignore_block_matches_manuscript() -> None:
    current = (ROOT / ".gitignore").read_text(encoding="utf-8")
    wanted = sync.synced_text(current, sync.render_block(sync.referenced_stems()))
    wanted = sync.synced_text(
        wanted,
        sync.render_table_block(sync.referenced_table_stems()),
        sync.TABLE_BEGIN,
        sync.TABLE_END,
    )
    assert current == wanted, (
        "run `python scripts/dev/sync_figure_gitignore.py` to regenerate the block"
    )


def test_stems_skip_external_and_planned() -> None:
    stems = sync.referenced_stems()
    assert stems and "workflow" not in stems


def test_tables_block_tracks_built_tables_only() -> None:
    assert sync.referenced_table_stems() == [
        "papkou2023_candidate_corrections",
        "papkou2023_full_vs_published",
    ]
    block = sync.render_table_block(sync.referenced_table_stems())
    assert "!output/tables/canonical/papkou2023_full_vs_published.csv" in block
    assert "!output/tables/main/" in block
