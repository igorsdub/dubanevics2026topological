# Figure renderer registry

Status: settled design from an architecture review on 2026-10-15; buckets 1-5 implemented; bucket 5 (independent review) done on 2026-10-15.

Update 2026-10-21: the figures `papkou2023_candidate_grouped_topology`, `papkou2023_candidate_tree_contrast_biology`, `papkou2023_codon_entropy_contrast` and `papkou2023_hill_core_membership` were removed, and `linked_tree_persistence` lost its annotated variant (its vanilla export now has the plain id). The inputs `candidate_grouped_trees`, `candidate_contrast_smoothed`, `codon_entropy_contrast` and `wild_type_sequence` and the read-only `load_cached_candidate_grouped_trees` reader went with them. The text below records the design as settled on 2026-10-15 and still names them.

Update 2026-10-21 (second removal): the figures `papkou2023_codon_entropy`, `papkou2023_candidate_tree_biology` and `papkou2023_bootstrap_core_tree` were removed with the contrast-smoothed tree stage (ADR 0006, superseded by ADR 0008). The `codon_entropy` input, its `load_cached_codon_entropy_tables` reader and the `candidate_leaf_view` composer went with them. `figures/codon_entropy.py` now holds only the builder of `bootstrap_codon_entropy`, and `figures/candidate_tree_biology.py` keeps the row, outcome-key and legend helpers that the bootstrap figures share.

Update 2026-10-22: `peak_uncertainty` now declares the inputs `peak` and `growth_rates` (the new input name, read by `load_cached_sequence_growth_rates`) and a composer, `_compose_peak_uncertainty`, which joins them in memory with `analysis.pooled_scatter.pooled_scatter` (ADR 2026-10-22-pooled-standard-error-for-bootstrap); the figure's dependency record gains `pooled_scatter` (the growth-rate checkpoint identity, the two tolerances and the composer's source, from `_pooled_scatter_dependency`). The peak checkpoint itself is unchanged.

Update 2026-11-25: the grouped-tree code left this repository with the Topographer 0.5.0 repin. The stages `candidate-q-analysis` and `candidate-grouped-trees`, the analysis modules behind them (`candidate_q_hierarchy`, `candidate_grouped_trees`, `candidate_grouped_topology`, `grouped_tree`, `grouped_tree_checkpoint`, `edge_grouping`, `edge_threshold_hierarchy`) and the `[candidate_grouped_topology]` table of `config/analysis.toml` were deleted. Mentions of them below, including the F4 note that `all` prepares the grouped trees, are historical.

## Settled contract

A Figure spec is a frozen dataclass with these fields: `figure_id`; `inputs`, a frozenset of named cached analyses drawn from split_analysis, comparison, peak, edge_evidence, codon_entropy, component_persistence, full_fitness, graph_tree_persistence, and candidate_grouped_trees; `variants`, a frozenset of allowed variants among vanilla and annotated; `config_table`, the `[figures.<id>]` table name in `config/figures.toml`; `notebook`, an optional path; and one `render(inputs, settings, variant)` callable that returns the chart. The spec's interface is the only seam that tests and the build cross.

The registry lives in `figures/registry.py` as an explicit `FIGURE_SPECS` tuple. `FIGURE_IDS` is derived from it. There is no decorator and no import-side-effect registration, so reading the tuple shows every figure the build can render.

Each figure identifier has one renderer module under `figures/`. Identifiers stay stable and descriptive and never encode manuscript numbering. Shared chart vocabulary (tree chart, persistence figure, legends, marker config, colour scales) moves to `figures/_charts.py`, an internal seam used by renderer modules and by no one else. Each renderer module is a deep module: a small spec interface over the chart construction it hides, which keeps change local to one figure.

Build owns input loading. `load_inputs(names)` maps input names to the existing `load_cached_*` readers, and renderers never read `output/`. This keeps the read-only reader rule of ADR-0005 in one place and gives the build leverage over every figure: a new figure declares its inputs and gets loading for free.

Variants are resolved by the build, not by renderers. Build renders the intersection of the requested and allowed variants and raises an error only when that intersection is empty. This replaces the eight single-figure variant guards.

The dependency fingerprint is computed per figure and variant by `_figure_dependencies` in `build/figures.py`. It hashes the figure's renderer file and every file in `SHARED_RENDERER_FILES` (`_charts.py`, `_branch_count_style.py`, `component_persistence.py`, `registry.py`), the Topographer histogram sources for the histogram and comparison figures, and the spec's notebook. It also records the spec's `[figures.<id>]` config table, the shared figure config (every `config/figures.toml` key except `figures`, `formats` and `variants`), and a digest of the build module itself (`render_source`). `build_figures` then adds the analysis dependencies each figure needs: the comparison or full analysis fingerprint, the edge provenance, the peak and codon entropy fingerprints, the component persistence artifacts, the candidate biology settings and sources, or the grouped-topology checkpoint provenance and alpha. An edit to a shared file therefore invalidates every figure, while an edit to one renderer invalidates only that figure's records.

`graph_tree_persistence` and `papkou2023_candidate_grouped_topology` had config tables in `config/figures.toml` but were absent from `FIGURE_IDS`. They were given specs and joined the build (bucket 3).

`build_figure(analysis, figure_id, settings, variant)` remained as a thin wrapper over the registry while the notebooks and figure tests migrated. Bucket 4 removed it.

Tests: chart-helper tests move to a module for `_charts`. Figure tests call `render_payload(get_spec(...))` with fixtures and import no private helpers; `tests/test_figure_registry.py` enforces this with an AST check that rejects underscore-prefixed imports from `dubanevics2026topological.figures.*` other than `_charts`.

## Implementation buckets

1. Scaffold the registry, `FigureSpec`, `load_inputs`, and the generic build loop behind the existing identifiers. All tests stay green.
2. Migrate renderers one per commit, each with its tests.
3. Register the two orphan figures, `graph_tree_persistence` and `candidate_grouped_topology`.
4. Remove the `build_figure` wrapper after the notebooks migrate (done).
5. Independent review against the merge base (done).

## Bucket 2 outcome

Every renderer lives in its own module under `figures/`, mapped in `registry._RENDERER_FILES`:

- `candidate_edge_evidence.py`: `candidate_edge_evidence`
- `peak_uncertainty.py`: `peak_uncertainty`
- `codon_entropy.py`: `papkou2023_codon_entropy`
- `tree_comparison.py`: `tree_comparison`
- `tree_histogram.py`: `full_tree_histogram`, `published_tree_histogram`, `candidate_tree_histogram`
- `study_tree_persistence_metrics.py`: `study_tree_persistence_metrics`
- `study_tree_persistence_biology.py`: `study_tree_persistence_biology`
- `candidate_tree_biology.py`: `papkou2023_candidate_tree_biology`
- `full_tree_persistence.py`: `full_tree_persistence`
- `split_tree.py`: `linked_tree_persistence`

`tree_histogram.py` serves three identifiers that share one body; the figure id selects the behavior.

The dependency fingerprint hashes the figure's renderer file plus `SHARED_RENDERER_FILES` (`_charts.py`, `_branch_count_style.py`, `component_persistence.py`, `registry.py`). An edit to any shared file therefore invalidates every figure. This is conservative; `component_persistence.py` could become a per-figure dependency later.

Chart-helper tests live in `tests/test_charts.py`. Figure tests use `_charts` only as monkeypatch targets.

## Bucket 3 outcome

The two orphan figures are now specs rendered by the generic build loop:

- `graph_tree_persistence` (renderer `graph_tree_persistence.py`): input `graph_tree_persistence`, variant vanilla.
- `papkou2023_candidate_grouped_topology` (renderer `candidate_grouped_topology.py`): inputs `candidate_grouped_trees` and `wild_type_sequence`, variant vanilla; the render builds and returns only the "overlaid" chart (`views=("overlaid",)`).

Three input names were added to `registry.INPUT_NAMES` and `build.load_inputs`:

- `graph_tree_persistence` is built in `load_inputs` by `analysis.manifold_graph_tree.build_manifold_graph_tree(settings["analysis"]["graph_tree_persistence"])`. It is synthetic and needs no `data/raw/`.
- `candidate_grouped_trees` is read by the new `analysis.candidate_grouped_trees.load_cached_candidate_grouped_trees(alpha)`, a read-only reader per ADR-0005. It never computes: a missing checkpoint raises `FileNotFoundError` that tells the reader to rebuild the checkpoint. The reader holds the cache lock while reading and rejects the pending marker.
- `wild_type_sequence` is read in `load_inputs` by `build.load_wild_type_sequence(settings)` from the configured `fitness_wt` file (unique `dna_from`). The renderer requires it as a keyword argument and never reads `data/raw/`; pass `None` only to omit the marker. The fingerprint keeps the `fitness_wt` digest.

Alpha decision: the scientific alpha for the build is `[candidate_grouped_topology] alpha = 0.05` in `config/analysis.toml`, read through settings. The notebook alpha stays exploratory.

The ad hoc `load_settings()` calls are gone from both renderer modules. `settings` is a required parameter of the manifold builders and of `build_candidate_grouped_split_tree_charts` and the two `save_*` functions, and the notebooks, the `candidate-split-preview` command and tests pass it explicitly.

Dependency fingerprints do not call `_analysis_fingerprint()` for these figures. `graph_tree_persistence` records `settings["analysis"]["graph_tree_persistence"]` and a digest of `analysis/manifold_graph_tree.py`. `papkou2023_candidate_grouped_topology` records the alpha, a sha256 of the canonical JSON of the checkpoint provenance already loaded under the lock (it lists the hash of every artifact the reader verifies), the `candidate_tree_histogram` config, a digest of the `fitness_wt` file, and a digest of `analysis/grouped_branch_counts.py`.

## Bucket 4 outcome

The `build_figure` wrapper is removed from `figures/__init__.py`. Callers use the registry directly:

- The 13 `notebooks/papkou2023_*.py` notebooks call `render_payload(get_spec(FIGURE_ID), payload, settings, variant)`.
- The inline nested-dict block the 12 appearance notebooks built for its preview settings is replaced by the public helper `figures.preview_settings(settings, figure_id, appearance)`, which returns a copy and never mutates its input. Notebooks name the local result `preview_config`. The `appearance` dict passed to `save_appearance_settings` is unchanged.
- `render_payload` now rejects any variant other than `vanilla` or `annotated` with `ValueError`, preserving the wrapper's check. Strict per-figure variant rejection is unchanged.
- `tests/_fake_figures.py` exposes the fake as `render` instead of `build_figure`, and the five build tests pass `render=`.
- `tests/test_figures.py` tests the `get_spec` and `render_payload` seam. `tests/test_appearance_notebooks.py` is a static AST smoke test over every `papkou2023_*` notebook (it parses the files and does not execute them): no `build_figure`, `render_payload` and `get_spec` imported from `dubanevics2026topological.figures`, and `save_appearance_settings` still called where imported.

## Superseded behavior

The 212-line if-chain dispatch in `build_figure`, the three hard-coded figure id sets in `build_figures`, the eight per-figure variant guards, and the whole-directory dependency hashing of every `figures/*.py` in `_figure_dependencies` are replaced by the registry. The six ad hoc `if X in selection` input-loading blocks are replaced by `load_inputs`.

## Bucket 5 outcome

The independent review against the merge base found nine issues; these were fixed:

- F1: the codon notebook and its fallbacks now build the payload the way `compose_payload` does, including `candidate_leaf_view`.
- F2: the codon entropy fingerprint includes the q-value cutoff.
- F3: `papkou2023_candidate_grouped_topology` declares `wild_type_sequence` as a spec input; the renderer no longer reads `data/raw/`.
- F4: the CLI `all` command prepares the grouped trees before rendering.
- F5: every figure test reaches its spec through `render_payload(get_spec(...))` with the real builder, so the registry adapters run unfaked. Only the WT-marker locator tests in `tests/test_candidate_grouped_topology_figure.py` still address a private helper, through a module attribute rather than an import.
- F6: a test checks that figure config tables and specs match in both directions.
- F9: the config comment now describes the grouped topology alpha as the canonical build alpha.
