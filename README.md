# Topological characterization of combinatorial fitness landscapes using merge trees

This repository holds the code and data pipeline for that manuscript (Dubanevics et al., in preparation, 2026).

Reproducible full Papkou2023 analysis and publication figures. Manuscript text and captions remain in their separate Google Doc.

## Start

Install uv and just (`brew install just` on macOS with Homebrew; the repository was translated against just 1.58.0). Everything runs locally; there is no cluster workflow.

```sh
uv sync --locked
just all            # download, compute, collect into data/processed/, write MANIFEST.tsv, render the figures
just figures        # render only, from data/processed/ (it checks MANIFEST.tsv first and computes nothing)
just fetch-data     # unavailable until a data/processed/ deposit is published (none exists yet); run `just all` to produce data/processed/
```

`just all` is `compute`, `processed-manifest` and `figures`. `compute` downloads the Papkou et al. 2023 record (`fetch-papkou2023`), rebuilds `data/raw/` (`make-raw`), exports the BarSeq counts with barseqfit (`barseq-export`, which runs `just export-papkou` in the barseqfit checkout `$BARSEQFIT`, default `../barseqfit`), then runs every stage from the growth rates to the quantity registry. Each stage output is collected into `data/processed/` under its state name and `processed-manifest` writes `MANIFEST.tsv` (path, bytes, sha256) last.

### The processed-data folder

`data/processed/` (ignored by Git) is the single folder of finished results; `just all` produces it. No `data/processed/` deposit exists yet, so `just fetch-data` stays unavailable until one is published. The directory name is the identity: `growth_rates/`, `classification/<published|reflected>/`, `bootstrap/<published_by|reflected_by>/`, `band_quantile_scan/<published_by|reflected_by>/`, `landscape/`, `tree/`, `leaves/`, `tree_layout/`, `tree_annotations/`, `landscape_summary/`, `ridges/`, `ridge_entropy_sweeps/` and `merge_flow/` (each `<state>/`, the seven states being `full`, `published`, `reflected`, `published_by`, `published_by_band`, `reflected_by`, `reflected_by_band`), `comparison/`, `peak_panels/` and `pooled_scatter/` (per graph; the default output root is `default`), `raw/` (the four `data/raw` files with their README, credit to Papkou et al. under CC-BY-4.0; `just figures` restores them to `data/raw/`), `stage_outputs/` (verbatim copies of the checkpoints, records and tables the figures read from `output/`, restored by `just figures`), and `threshold_diagnostics/`, `codon_entropy/`, `bootstrap_codon_entropy/`, `graph_tree_persistence/`, `manuscript_quantities/`, `presentation_tables/` and `graph_descriptive_statistics/`. Products keep their own `manifest.json`; `just check-manifest` compares the whole folder with `MANIFEST.tsv` and fails on a missing, extra, resized or modified file.

Only the two BY graphs have a bootstrap, so the band-quantile scan figure shows those two graphs.

Topographer is pinned to tag `v0.5.0` and barseqfit to tag `v0.2.0` (see `[tool.uv.sources]` in `pyproject.toml`). Source data are local and ignored by Git; an exact public acquisition/conversion recipe is not yet established.

### BarSeq export input

The candidate-tree contrast reads count tables exported by the barseqfit repository (https://github.com/oist/barseqfit, tag `v0.2.0`). Run `just export-papkou` there to write `counts.parquet`, `samples.parquet` and `sequences.parquet` into one folder. `just barseq-export` does this for you and sets `PAPKOU2023_BARSEQ_EXPORT` to the folder barseqfit writes (`$BARSEQFIT_SCRATCH`, default `$BARSEQFIT/.scratch/papkou2023`). To use another folder, copy `.env.example` to `.env` and set `PAPKOU2023_BARSEQ_EXPORT` to it (`just` loads `.env`; the barseqfit folder is only the fallback when the variable is unset or empty). Set `PAPKOU2023RUGGED_DATA` there as well for `just fetch-papkou2023` and `just make-raw`. A variable exported in your shell overrides the value in `.env`. The `.env` file is read from the checkout (or worktree) root, so each worktree needs its own.

## Work on figures

```sh
uv run marimo edit notebooks/linked_tree_persistence.py
uv run marimo edit notebooks/full_tree_persistence.py
uv run marimo edit notebooks/tree_comparison.py
uv run marimo edit notebooks/peak_uncertainty.py
just watch
```

Notebook controls change appearance and, where a figure has both, select vanilla or annotated previews. Save appearance writes that figure's panel settings to `config/figures.toml`. The watcher rebuilds affected exports after saved changes. Preview exports go to `output/previews/`. `just figures` writes the identity shelf `output/figures/canonical/`. `just name-outputs` copies those into numbered `main/` and `supplementary/` names and writes `manifest.csv`. See `output/figures/README.md`.

Scientific settings live in `config/analysis.toml`. The publication build supports the full landscape, the split-tree products, and full-network component persistence. Appearance changes reuse saved scientific outputs. Analysis source, raw input, configuration, and effective Topographer source hashes detect scientific changes. Unchanged outputs are reused; failed renders are retried.

The separate component analysis computes ordinary H0 minima, relative H1 maxima (superlevel component connectivity, not independent graph cycles), and extended-positive H0 for the global range. It exports a death-value frequency and threshold diagnostic report. The Topographer threshold is exploratory; a scalar gap is not a calibrated measurement-error estimate. Run `just component-analysis` to refresh this product or `just full-tree-figure` to render the full-graph tree, persistence and histogram figure from the cache.

Run `just peak-figure` for the four-graph comparison (rows Papkou graph and Topographer + FDR graph,
columns Nominal and `BY q = 0.05`; each graph is read from its own analysis variant,
saved by `just peak-panels-product` under `output/variants/<variant>/`, ADR
2026-11-19-peak-analysis-per-variant): the pooled standard error of the growth rate on x and
fitness relative to the wild type (`r - r_wt`) on y (ADR 2026-10-22-pooled-standard-error-for-bootstrap).
A render recomputes nothing. The four plotting areas are equal squares, with
one notebook size control, share their axes and use 5% span padding; only the
left panels show y-axis labels. No grid is drawn, only ticks. Scatter marks are gray for other
measurements and black triangles, twice the original linear size (Altair `size` 112), for each landscape's
raw-fitness local maxima; the wild type is plotted at y = 0 with its own
pooled standard error, as the red wild-type star of the tree figures on the
topmost layer of every panel. The x title is `Pooled fitness SE`. One legend, in the top-right panel,
lists `Peak`, `Not peak` and `Wild type`; every panel draws the dashed functional
threshold with no text label. The peak-count curves against
the q-value threshold that were a second row are no longer drawn; the cached
minimum `1 - q` rule and its counts are unchanged and still exported. The count requires every graph
neighbour comparison to pass the threshold, with inclusive equality. One
Benjamini-Hochberg correction across all full-data pairwise tests supplies
both landscapes. Missing or invalid comparisons stop analysis explicitly.
`just peak-analysis` computes and caches this evidence without constructing
trees. Its CSV/Parquet tables and diagnostics use the `peak_uncertainty_`
prefix under `output/tables/`. Notebook appearance changes reuse this cache.
The notebook initially displays the canonical PNG; use **Render appearance
preview** after adjusting controls. This image contains every measurement and
keeps the browser preview below Marimo's output-size limit.
See the [design](docs/peak-significance-design.md) and
statistical decision.

The published-threshold metric figure reuses the cached Papkou2023 landscape and renders tree–persistence rows for branch vertex count, amino-acid endpoint Hamming distance, and Horton–Strahler order. Its appearance-only controls live in `notebooks/study_tree_persistence_metrics.py`; run `just study-metrics-figure` to refresh the vanilla figure. See the [validation note](docs/study-metric-figure-validation.md) for saved-table counts and rendering checks.

The study biological figure uses the same landscape and panel geometry in two
rows: stop-codon containing maxima above, adjacent-saddle synonymous terminal leaf
branches below. Run
`just study-biology-figure` to export its annotated PNG (SVG via `just figures-svg`); the sequence-keyed
biological evidence tables come from `just biological-annotation-tables`; appearance controls live in
`notebooks/study_tree_persistence_biology.py`. The lower row highlights
whole terminal leaf branches when the peak amino-acid sequence matches its
first adjacent descending saddle. Matching persistence triangles use that same
comparison, which can differ from the persistence death partner. Stop maxima
receive solid crimson (`#a50f15`) × marks on the tree and matching solid crimson
triangles on the persistence diagram. Tree structure remains gray
outside these highlights, and the red wild-type star remains visible. Distinct
stop states remain distinct. Evidence tables retain sequence identities and
both endpoint comparisons. See the
annotation decision and
[validation evidence](docs/study-biological-figure-validation.md).

`just codon-entropy-analysis` refreshes the codon-entropy tables from the cached
comparison: the sequence populations of the full study-selected Papkou graph, including
its retained below-threshold fringe, with site-frequency, site-entropy, claim-summary, and
threshold-sweep tables for the full population and the KEM and TCL branches. Branch membership
follows the branches above their highest shared saddle, excluding that saddle from both
branches. No default figure reads these tables. `bootstrap_codon_entropy` (see below)
runs the same population and sweep functions in memory on the band-simplified tree and
renders the paired trees and entropy panels: 520 × 250 px trees, a shared raw-fitness axis,
branches coloured by vertex count (the focused branch solid, every other branch dimmed, a count label on every branch of the top tree and beside the branches), and one shared 0–log2(23) entropy scale.
See the [codon-entropy validation note](docs/codon-entropy-validation.md) for sequence
identities, branch ownership, and comparison with the reported `>90%` claim.

The separate threshold-selected landscape comparison validates the study selection against locally copied Topographer GraphML and peak references, then prepares the full/raw graph, published functional threshold at -0.507774, and the candidate functional threshold, computed by the `functional-classification` stage as a quantile of the stop-anchored non-functional mixture (the config holds only the rule `mixture_quantile` and digits; run `just functional-classification` first; see the ADR). A vertex is functional when its Functional probability reaches the declared confidence level (the Functional class); the earlier Topographer value `m + (m - g)` (-0.253159, raw fitness at or above it) stays as the `median_reflected` comparison rule. Original edges are retained when at least one endpoint is functional, so functional ties and the below-threshold fringe remain; fringe-to-fringe edges are excluded. Each selection retains its largest weakly connected component. Every tree, persistence result, peak, wild-type position, and plotted fitness domain uses raw fitness. Stable-sequence support tables and provenance record selection and component accounting. The Topographer comparison threshold rests on a symmetry assumption, not a calibrated uncertainty estimate. Run `just comparison-analysis` to refresh the science or `just comparison-figure` to render the single WT-only comparison from cached results. The [validation note](docs/clamped-tree-comparison-validation.md) records the completed full-data raw-fitness rebuild, reference equivalence, observed tree metrics, and figure inspection.

Run `just landscape-summary-table` (a step of `just tables`) to export the seven-column landscape summary, `output/tables/papkou2023_landscape_summary.csv`, `.html`, `.md` and a JSON sidecar, from the seven `landscape_summary` states of `data/processed/` named in `config/figures.toml`; see `docs/figure-inventory.md`. It replaces the three-column Table 1 (`just graph-summary`, retired in Ticket 15 S4). Manuscript Tables 1 and 2 are the graph and tree descriptive tables (`graph_descriptive_statistics`, `candidate_band_comparison`), numbered from the `[[table]]` entries of `config/manuscript.toml`. The landscape-summary slices (`papkou2023_full_vs_published.*`, `papkou2023_candidate_corrections.*`) are supplementary tables S1 and S2, built by `just presentation-tables`. Open the standalone HTML in a browser, select and copy the table, then paste it into Google Docs. This provides a plain table workflow without manuscript numbering; Google Docs paste rendering has not been verified.

## Outputs

- `output/analysis/`: `data` links to the public Topographer checkpoint, which is published under `output/checkpoints/analysis/`; `provenance.json` is the caller's record of it.
- `output/figures/`: `full_tree_persistence` in vanilla and annotated variants, the vanilla `linked_tree_persistence`, plus the canonical `tree_comparison` and `peak_uncertainty` exports, each as a 300 dpi PNG by default; `just figures-svg` renders SVG only, without touching PNGs.
- `output/tables/`: split-tree and component-persistence tables plus `clamped_*` comparison and `peak_uncertainty_*` evidence tables as CSV and Parquet, with stable sequence identities and export provenance.
- Codon-entropy population, site-frequency, site-entropy, sequence-assignment,
  claim-summary, and binned-entropy tables are also exported as CSV/Parquet.
- `output/checkpoints/`: one published checkpoint per table stage and one for the main analysis (artefacts plus `provenance.json`), written by the shared produce-or-load helper (ADR 0013). `output/tables/` holds byte-identical exports of these checkpoints with a `.*provenance.json` sidecar each. A stage is reused only while its fingerprint matches and every artefact hash verifies; invalid state is moved aside as `.*.unusable-*`, never deleted. The first build after the helper landed rebuilds each table stage once.

The full landscape contains 261,333 sequences and 3,518,696 edges; the split tree has 7,815 nodes and 4,055 persistence pairs, including the global maximum's pair, closed at the global minimum by convention so every pair is finite. Fitness is not rescaled to [0,1]. Branch vertex count includes the upper branch endpoint and interior original graph vertices, excludes the lower merger/root endpoint, and has a minimum of one. The global root is counted separately. Existing tree figures use a shared logarithmic count domain; the Papkou tree comparison uses shared configured fitness limits and shared histogram/count-color maxima. Count one is the gray singleton exception explained separately below the color scale. Branch span and persistence are distinct quantities.

The paired tree/persistence figure shares the fitness domain. It is a side-by-side view without a plot title; cross-panel selection is not implemented. Baseline tree synonymous annotations refer to branch endpoints; baseline persistence annotations refer to paired birth/death endpoints. The study biological figure uses the adjacent-saddle definition documented above. The comparison panels are individually labeled by landscape and threshold.

## Commands

`just analysis`, `just component-analysis`, `just comparison-analysis`, `just peak-analysis`, `just codon-entropy-analysis`, `just tables`, `just figures`, `just all`, and `just watch` separate computation and presentation. `just figures` builds both variants of the full-graph figure and the canonical linked-tree, comparison, and peak uncertainty exports; `just full-tree-figure`, `just comparison-figure`, and `just peak-figure` select individual figures. Use `uv run python -m dubanevics2026topological.build figures --figure full_tree_persistence --variant vanilla` to render one variant from cached results, or add `--force` to regenerate outputs. Appearance edits reuse saved scientific analysis checkpoints.

`just candidate-tree-contrast` fits one barseqfit pair contrast per edge of the Candidate split tree and exports `papkou2023_candidate_tree_contrast_pairs` and `papkou2023_candidate_tree_contrast_edges` (CSV and Parquet) under `output/tables/`. It is not part of `just all` or `just figures` because it needs the BarSeq export named in `.env`. The bootstrap stage no longer reads it (singleton check removed, ADR `2026-10-22-pooled-standard-error-for-bootstrap`). See the [design note](docs/candidate-tree-contrast-design.md).

`just sequence-growth-rates` fits one growth rate `r` per sequence of the Candidate network (plus the wild type) with `barseqfit.fit_sequences`, with a pooled and an across-replicate standard error, and exports `papkou2023_sequence_growth_rates` (CSV and Parquet) under `output/tables/`, plus the checkpoint `output/analysis/sequence_growth_rates/` with `diagnostics.json`. The stage fails closed on any non-converged fit or missing replicate SE, and on any sequence where `r - r_wt` differs from the published `s` by more than `identity_tolerance`. Settings are in `[sequence_growth_rates]` in `config/analysis.toml`. It needs the BarSeq export (`just barseq-export`) and is the first stage of `just all`; `just figures` never runs it. See the [bootstrap smoothing design](docs/bootstrap-smoothing-design.md).

`just bootstrap-smoothed-tree` bootstraps the nominal Candidate split tree: it draws `n_replicates` landscapes from `Normal(raw_fitness, se_pooled)` per sequence (the pooled standard error of the growth rate, ADR `2026-10-22-pooled-standard-error-for-bootstrap`; the run's fingerprint records this noise model, so it never matches a run made with the replicate standard error), rebuilds the split tree in each draw and counts, per pair of sequences, how often they sit on the same terminal branch. Sets of sequences that stick together in at least `core_threshold` of the draws are peak cores. A nominal hill that no core meets is absorbed into its saddle; of the hills one core spans, only the one with the highest nominal peak is kept; the elder rule still protects the elder arm at each saddle. A second run on a flat landscape (same seed and draws) gives the chance level for the cores and a false-hill rate. The global verdict is a bootstrap confidence band (Fasy et al. 2014): `c` is the `band_quantile` quantile of the real draws' bottleneck distance to the nominal ordinary H0 persistence diagram (the global maximum paired with the root, so every pair has a finite birth and death), and nominal points with persistence above `2c` are distinguishable from noise. It exports `papkou2023_bootstrap_hills`, `papkou2023_bootstrap_cores`, `papkou2023_bootstrap_vertex_roles` and `papkou2023_bootstrap_draw_stats` (CSV and Parquet) under `output/tables/`, plus the working slot `output/analysis/bootstrap_smoothed_tree/` (not the checkpoint itself, see below) with `pair_frequency.parquet`, `cores.parquet`, `hills.parquet`, `vertex_roles.parquet`, `draw_stats.parquet` and `summary.json` (outcome and core-relation counts, global verdict, flat-landscape false-hill rate, consensus peaks). The recipe depends on `sequence-growth-rates`, so it needs the BarSeq export environment variable. `n_jobs > 1` uses a spawn process pool; the build entry point is guarded for it. Settings are in `[bootstrap_smoothed_tree]` in `config/analysis.toml`; `n_jobs` does not change the result. The checkpoint, the one verified copy, is published to `output/checkpoints/bootstrap_smoothed_tree/` (ADR 0013); the working slot `output/analysis/bootstrap_smoothed_tree/` is a byte-for-byte export of it, the layout the figures and tables read, and a reader refuses a slot that differs from the checkpoint of the same run (`just bootstrap-smoothed-tree` exports it again). When `output/` has no matching checkpoint, the loader restores the one in `data/processed/bootstrap/<state>/checkpoint` that matches the configuration, without drawing anything. The recipe is a step of `just compute`. `just bootstrap-biology-figure` renders `candidate_tree_bootstrap_biology` (the nominal tree and the band-simplified tree, each beside its persistence diagram with the strip of persistence up to `τ = 2c` shaded and peaks marked filled mid-grey (survives) or filled black (absorbed) by whether they survive τ), `just bootstrap-tree-histogram-figure` and `just bootstrap-codon-entropy-figure` render `bootstrap_tree_histogram` and `bootstrap_codon_entropy` (the candidate tree-histogram and codon entropy figures on the band-simplified tree) and `just bootstrap-global-stability-figure` renders `bootstrap_global_stability` (the nominal-landscape bottleneck distances with `c` marked above the flat ones as a sanity check, beside the nominal persistence diagram with the band of persistence up to `2c` shaded); they read the working slot, not the checkpoint. See the [bootstrap smoothing design](docs/bootstrap-smoothing-design.md).

`just branch-decomposition` lists every edge of the band-simplified Candidate split tree (`[branch_decomposition] tree` in `config/analysis.toml`; the core-rule and nominal trees are the other choices) as one branch and exports `branches` and `members` (Parquet and CSV) to `output/analysis/branch_decomposition/`: per branch the sequences of its two endpoints, its elder peak, its terminal or internal kind, Horton–Strahler order computed on the smoothed tree (absorbed hills do not raise it), branch vertex count (after absorption), edge count (single-mutation adjacencies of the Full graph, every sequence of the dataset before the candidate functional threshold prunes it, among its own sequences; a single mutation joining two sequences below the candidate functional threshold counts, though it is not an edge of the Candidate graph), branch span, birth and death fitness and member sequences. It reads the bootstrap checkpoint without drawing (restoring the tracked mirror into `output/` when `output/` has no matching checkpoint, and moving a non-matching one aside; the notebook's reader does the same before it reads the branch checkpoint, which it only reads), needs `data/raw/` for the Candidate tree and the Full graph, and is not part of `just all` or `just figures`. Explore it with `uv run marimo edit notebooks/papkou2023_branch_decomposition.py` (filter by order, per-order histograms with a symmetric-log count axis on the fitness panels by default, optional per-branch facets in one grid, a button that saves PNGs under the gitignored `output/exploratory/branch_decomposition/`); it is an exploratory analysis stage, not a registered figure.

`just leading-saddle-band` reads the cached component pairs and the Full-graph fitness and writes the Full graph's leading and trailing saddles, the landmark, interval and positional-entropy tables and the band and trunk counts to `output/checkpoints/leading_saddle_band/` (copied to `output/tables/`); run `just component-analysis` first. `full_tree_fitness_landmarks` draws it (opt-in: `build figures --figure full_tree_fitness_landmarks`). Details are in `docs/figure-inventory.md`.

For explicit local package development, use `uv run --with-editable ../topographer python -m dubanevics2026topological.build all`, which runs the tables stages (including the leading saddle band, the biological annotation tables and the manuscript quantity registry, in that order) before the figures but never the bootstrap, branch or BarSeq stages. The effective package sources enter analysis provenance. Return to the pinned environment with `uv sync --locked`.

The saved-tree views of the manuscript (D10: `tree_persistence`, `bootstrap_band` and the `tree_comparison_*` views) are ordinary registry figure families built by `build figures`; there is no separate D10 command. The former `legacy/` package no longer exists: the Papkou2023 workflow lives in `papkou2023/` and `network_analysis.py`, the co-occurrence readers in `simplification/cooccurrence_*.py` and the edge-grouping modules in `analysis/`.

Publication PNGs in `output/figures/` can be committed as reviewable figure snapshots. Regenerate them with `just figures` and include updated PNGs alongside changes to their source or appearance settings. Raw data, cached analysis, previews, SVGs, and supporting table exports remain ignored. See [the figure inventory](docs/figure-inventory.md) for the canonical filenames and [the approved figure style](docs/figure-style-design.md) for the appearance contract.

### Candidate fitness histogram and comparison scales

`candidate_tree_histogram` pairs the Candidate split tree with a 200-bin fitness histogram. Every vertex in the selected graph contributes once, including retained fringe and noncritical vertices. Fitness is vertical; touching gray bars show vertex count horizontally. Use `just candidate-histogram-figure` to render from the saved comparison checkpoint, or open `notebooks/candidate_tree_histogram.py` for appearance controls and explicit export.

The Full/Published/Candidate tree comparison uses common fitness limits and one logarithmic branch-count color domain with one shared legend. Compared histogram panels retain independently observed-range bins while sharing displayed fitness and count limits. See the [approved design and implementation buckets](docs/fitness-histogram-design.md).

Full and Published equivalents are `full_tree_histogram` and `published_tree_histogram`, with matching notebooks and just recipes `full-histogram-figure` and `published-histogram-figure`. `tree_comparison` stacks all three tree–histogram rows with shared histogram count and branch-color limits. The tree family uses the configured fitness display domain `[-1.2, 1.4]` without padding; changing it affects visibility only. See [the design contract](docs/fitness-histogram-design.md).

## Software

This is version 1.0.2 of the code (tag `v1.0.2`). Concept DOI (all versions of the code): [10.5281/zenodo.24550751](https://doi.org/10.5281/zenodo.24550751).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24550751.svg)](https://doi.org/10.5281/zenodo.24550751)

The analysis uses two packages:

- barseqfit 0.2.0 (https://github.com/oist/barseqfit): fitness and fitness-difference estimates from barcode sequencing counts. Concept DOI: [10.5281/zenodo.24548492](https://doi.org/10.5281/zenodo.24548492).
- topographer 0.5.0 (https://github.com/oist/topographer): merge trees for graphs carrying a scalar on each vertex. Concept DOI: [10.5281/zenodo.24548494](https://doi.org/10.5281/zenodo.24548494).

The DOIs resolve to the latest archived version. The pinned tags 0.2.0 and 0.5.0 differ from the first archived versions (0.2.1 and 0.5.1) in release metadata only.

If you use them, please cite each package using the `CITATION.cff` in its repository.

## Data

The result files of the analysis run are archived at [doi:10.5281/zenodo.24545428](https://doi.org/10.5281/zenodo.24545428). They are the results the tracked figures and tables were rendered from. They use the folder layout and working names of release 1.0.0, not the `data/processed/` layout; the README of the deposit maps the names. `just fetch-data` cannot read this deposit.
