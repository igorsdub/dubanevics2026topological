# Repository guidance

All computation runs locally through `just`: `just all` downloads and computes everything, collects it into `data/processed/` and renders the figures; `just figures` renders only, from `data/processed/`; `just fetch-data` stays unavailable until a `data/processed/` deposit is published (none exists yet; `just all` produces `data/processed/`). Never borrow another checkout's virtual environment; each checkout uses its own `uv sync --locked`.

This repository supports the manuscript "Topological characterization of combinatorial fitness landscapes using merge trees" through reproducible analysis code, figures, and tables. Keep manuscript prose and captions in their separate source.

## Commands

- `just --list` shows all recipes.
- Tests: `PYTHONPATH=src:. uv run pytest -q -p no:cacheprovider`
- The two packages, barseqfit v0.2.0 and topographer v0.5.0, are installed from their release tags as pinned in `pyproject.toml`; do not change the pins without updating `uv.lock`.

## Work sequence

- Read `GLOSSARY.md`, `docs/figure-inventory.md`, and relevant TOML configuration before changing scientific outputs.
- Keep scientific settings in `config/analysis.toml`; keep presentation settings in `config/figures.toml`. Notebook controls are for appearance.
- Use stable descriptive figure identifiers rather than manuscript numbering.
- Preserve sequence identities in exported results; never expose Rustworkx indices as scientific identifiers.
- Keep branch span, ordinary H0 persistence (the global maximum's pair closed at the global minimum by convention), and extended persistence (not computed) as distinct quantities.
- Treat `data/raw/` as local input storage and `data/processed/` as the finished results (ignored by Git, `MANIFEST.tsv` at its root; never edit a file in it by hand). Local `output/` is disposable once `just all` has collected it into `data/processed/`. Exception: a bootstrap run that has not been published to `data/processed/bootstrap/<state>/` (its `output/checkpoints/bootstrap_smoothed_tree/` and `.replaced-*` copies) must be copied before deleting `output/` or a worktree.

## Package boundaries

Place reusable code in `src/dubanevics2026topological/`, notebooks in `notebooks/`, configuration in `config/`, and publication-facing inventory in `docs/`. Prefer Topographer's public APIs. Record interface proposals in documentation until implementation confirms them.
