# Analysis guide

A layer-by-layer walkthrough for the author and reviewers (FT-REQ-01, FT-REQ-06, FT-REQ-08; specification: Specification 2, section 10). It follows the four story layers in order and names, for each product, table and figure, its inputs, its configuration and the command that makes it. Every command below is a `just` recipe of the [justfile](../justfile), a script in the repository, or (where stated) the `dubanevics2026topological.build` entry point that the recipes wrap; nothing here changes a setting or a number. `just all` computes everything, collects it into `data/processed/` and renders the figures; `just fetch-data` downloads `data/processed/` instead, and `just figures` renders from it alone (see section 0).

Scientific settings live in [config/analysis.toml](../config/analysis.toml), presentation settings in [config/figures.toml](../config/figures.toml). Terms are those of the [glossary](../GLOSSARY.md). Figure rows below use the registered figure identifiers of the [figure inventory](figure-inventory.md). The D10 views of the presentation specification (`tree_persistence`, `bootstrap_band` and the `tree_comparison_*` views) are registry figure families built by `build figures` (figure inventory, "Saved-tree view families").

## 0. Before the layers: two ways to get the data

| | Compute (`just all`) | Fetch and render (`just fetch-data`, `just figures`) |
| --- | --- | --- |
| Spec | Specification 1, section 5C | same |
| Inputs | the Papkou et al. 2023 record (`just fetch-papkou2023`), `data/raw/` rebuilt from it (`just make-raw`), and the BarSeq export named by `PAPKOU2023_BARSEQ_EXPORT` (`just barseq-export`); `uv.lock` and the pinned Topographer and barseqfit sources | `data/processed/` (written by `just all`; `just fetch-data` is unavailable until a deposit is published), checked against `MANIFEST.tsv`; the BarSeq count files and the bootstrap chunks are not needed |
| Bootstrap | drawn by `just bootstrap-draws` for the two BY graphs (or restored without drawing from `data/processed/bootstrap/<state>/checkpoint` when a checkpoint there matches the configuration) | never drawn: the loader restores the matching processed checkpoint |
| Growth rates and classification | `just sequence-growth-rates`, `just functional-classification`, `just classification-probability-scans` (they hash the BarSeq export) | read from `data/processed/growth_rates` and `data/processed/classification/<published or reflected>` |
| Identity of the result | the directory name (`<kind>/<state>/`) and each product's `manifest.json` | the same files; `just check-manifest` fails on a missing, extra, resized or modified file |
| Tolerance | point estimates and graphs bit-identical; cross-platform bootstrap quantiles within 1e-6 with identical peak counts and absorption topology | identical artefact checksums and topology; zero draws |

The two BY bootstraps are `data/processed/bootstrap/published_by` (Papkou + FDR) and `data/processed/bootstrap/reflected_by` (Topographer + FDR), both 10,000 draws with seed 20261019 and band quantile 0.95. `just bootstrap-smoothed-tree` runs the default (Topographer + FDR) selection and `just bootstrap-smoothed-tree-by <variant>` a named variant; a run that matches a processed checkpoint draws nothing. The reference hashes of the original run's growth-rate and classification tables are in `config/data.toml` (`[reference_hashes]`): a recompute that differs from them warns and writes a delta report; it never fails.


## Layer 1: nominal baselines (Full and Published point-estimate graphs)

What the layer shows: the uncorrected topology, with the Full graph (every library sequence) and the Papkou graph (point-estimate threshold -0.507774, declared in `config/analysis.toml`) as split trees with ordinary H0 persistence and the raw fitness distribution. Ordinary H0 persistence closes the global maximum's pair at the global minimum; extended persistence is not computed, and branch span is a separate quantity.

| Product, table or figure | Inputs | Configuration | Command |
| --- | --- | --- | --- |
| Growth rates and Functional classification checkpoints | `data/raw/`, BarSeq export (compute) or `data/processed/growth_rates` (fetch) | `config/analysis.toml` | `just sequence-growth-rates`, `just functional-classification` |
| Full-graph analysis and component persistence | the growth-rate checkpoint, `data/raw/` | `config/analysis.toml` | `just analysis`, `just component-analysis` |
| The five landscape and nominal-tree products (`landscape-products` publishes five labelled landscape/tree pairs, ADR 0016; Full, Papkou and Topographer are the nominal states of Layers 1 and 2) | the analysis above | `config/analysis.toml` | `just landscape-products` |
| `leaves` product per state (leaf table and sequence membership) | the saved tree and landscape products | `config/analysis.toml` | `just leaves-product` (collected into `data/processed/leaves/<state>`) |
| `landscape_summary` product per state (seven) | the saved leaves, tree and landscape products | `config/analysis.toml` | `just landscape-summary-product` |
| Manuscript Table 1 (`graph_descriptive_statistics`: Full, Papkou, Topographer no FDR, Topographer-BY q = 0.05; rows nodes, edges, % functional, mean degree, mean in/out degree) | the four graphs | `config/analysis.toml` | `just graph-descriptive-statistics` |
| Manuscript Table 2 (`candidate_band_comparison`: BY q = 0.05 Topographer split tree and its bootstrap-band tree) | the BY bootstrap checkpoint, read and not drawn | `[bootstrap_smoothed_tree]` | `just candidate-band-comparison-table` |
| Supplementary tables S1 and S2 (`papkou2023_full_vs_published.*`, `papkou2023_candidate_corrections.*`, sliced from the seven pinned products by `presentation-tables`; numbered by `[[table]]` entries of `config/manuscript.toml`) | the seven pinned `landscape_summary` products, the registry | `[tables.landscape_summary]` in `config/figures.toml` | `just presentation-tables` (a step of `just tables`) |
| Landscape summary table (`output/tables/papkou2023_landscape_summary.{csv,html,md}` and `.json` sidecar) | the seven pinned `landscape_summary` products | `[tables.landscape_summary]` in `config/figures.toml` | `just landscape-summary-table` (a step of `just tables`) |
| Manuscript quantity registry and the leading saddle band tables | the above and the component analysis | `config/analysis.toml` | `just leading-saddle-band`, `just manuscript-quantities` |
| `full_tree_persistence` (Full tree and persistence) | component analysis, registry | `[figures.full_tree_persistence]` | `just full-tree-figure` |
| `full_tree_histogram`, `published_tree_histogram` | comparison analysis | `config/figures.toml` | `just full-histogram-figure`, `just published-histogram-figure` |
| `full_tree_fitness_landmarks` | `leading-saddle-band` tables | `config/figures.toml` | `just leading-saddle-band`, then `build figures --figure full_tree_fitness_landmarks` (no `just` recipe exists for this figure; the form is `PYTHONPATH=src python -m dubanevics2026topological.build figures --figure <id>`) |

## Layer 2: Topographer threshold

What the layer shows: the structural effect of an objective point-estimate threshold, the Topographer value -0.253159 (the `applied_threshold` in `provenance.json` of the BY classification snapshot, rounded from -0.2531594561094983; ADR 2026-10-23, the Topographer threshold), beside the Published baseline, without any uncertainty adjustment. The Candidate graph of the older figures is the separate `mixture_quantile` selection and is not one of the seven summary states.

| Product, table or figure | Inputs | Configuration | Command |
| --- | --- | --- | --- |
| Topographer landscape and nominal tree | the Layer 1 analysis | `config/analysis.toml` (`median_reflected` rule) | `just landscape-products` |
| Its `leaves`, `landscape_summary` and the "Topographer (nominal)" column | as Layer 1 | as Layer 1 | as Layer 1 |
| Threshold comparison figure `tree_comparison` (Full, Published, Candidate) | comparison analysis, registry | `[figures.tree_comparison]` | `just comparison-figure` |
| Threshold diagnostics product | growth rates, classification, components | `config/analysis.toml` | `just threshold-diagnostics-product` |
| Functional threshold sensitivity scan (supporting) | growth rates, classification | `config/analysis.toml` | `just functional-threshold-scan` |

## Layer 3: Benjamini-Yekutieli uncertainty (nominal q = 0.05)

What the layer shows: how pooled standard errors and the Benjamini-Yekutieli (BY) correction change the Papkou and Topographer selections. Classes are functional (s >= T, q <= 0.05), uncertain (s >= T, q > 0.05) and non-functional (s < T).

| Product, table or figure | Inputs | Configuration | Command |
| --- | --- | --- | --- |
| BY classification (both cutoffs) | the growth rates of `data/processed/growth_rates` | `config/analysis.toml` | `just classification-probability-scans` (writes `data/processed/classification/<published or reflected>/`; the original run's table hashes are the `[reference_hashes]` of `config/data.toml`) |
| BY analysis variants (Papkou + FDR, Topographer + FDR) | the classification above | `config/variants/threshold=published_fdr=by_q=0.05.toml`, `config/variants/threshold=reflected_fdr=by_q=0.05.toml` | `just variant-comparison-analysis "threshold=published_fdr=by_q=0.05"`, `just variant-comparison-analysis "threshold=reflected_fdr=by_q=0.05"` (output under `output/variants/<name>/`) |
| Three-variant graph comparison | the variants and the point-estimate graphs | `config/analysis.toml` | `just fdr-graph-comparison` |
| BY nominal `leaves`, `landscape_summary` and table columns | the variants' saved tree and landscape products | `[tables.landscape_summary.labels]` | as Layer 1 |
| Fitness versus pooled standard error | growth rates, peak analysis | `[figures.peak_uncertainty]` | `just peak-analysis`, `just pooled-scatter-product`, `just fdr-pooled-scatter-product`, `just peak-panels-product`, `just peak-figure` |

## Layer 4: bootstrap stability

What the layer shows: for the bootstrapped graph, the nominal tree against the band-simplified tree, where a peak survives exactly when its ordinary H0 persistence exceeds 2c and c is the band-quantile (0.95) of the real draws' bottleneck distance to the nominal diagram. No retired peak-core rule is involved; the core readers are the read-only archive adapters in `simplification/cooccurrence_*.py`.

| Product, table or figure | Inputs | Configuration | Command |
| --- | --- | --- | --- |
| Bootstrap checkpoints of the BY graphs (10,000 draws, seed 20261019) | growth rates, classification, the selected graph | `[bootstrap_smoothed_tree]` in `config/analysis.toml` | `just bootstrap-draws` (a run that matches `data/processed/bootstrap/<state>/checkpoint` draws nothing) |
| Band quantile scan (`scan.parquet`, `survival.parquet`) | the bootstrap checkpoint | `[bootstrap_band_quantile_scan]` (`grid_step` 0.001) | `just bootstrap-band-quantile-scan`, `just bootstrap-band-quantile-scan-product` (collected into `data/processed/band_quantile_scan/<published_by or reflected_by>`) |
| Band-simplified `leaves`, `landscape_summary`, and the two band columns of the table | the nominal tree product (parent) and the band identity | `config/figures.toml` labels | as Layer 1 |
| `ridges` product (KEM/TCL summary and DNA-keyed membership) | saved leaves, tree and landscape products | `config/analysis.toml` | `just ridges-product` |
| `candidate_tree_bootstrap_biology` (nominal and band-simplified trees) | Candidate-graph bootstrap checkpoint, comparison analysis | `config/figures.toml` | `just bootstrap-biology-figure` |
| `bootstrap_global_stability` | bootstrap checkpoint | `config/figures.toml` | `just bootstrap-global-stability-figure` |
| `bootstrap_tree_histogram` | bootstrap checkpoint | `config/figures.toml` | `just bootstrap-tree-histogram-figure` |
| `bootstrap_codon_entropy` | bootstrap checkpoint, codon-entropy product | `config/figures.toml` | `just bootstrap-codon-entropy-product`, `just bootstrap-codon-entropy-figure` |
| `bootstrap_band_quantile_scan` (and `_seeds`) | scan tables | `config/figures.toml` | `just bootstrap-band-quantile-scan-figure`, `just bootstrap-band-quantile-scan-seeds-figure` |

The bootstrap codon-entropy product records the resolved Candidate selection,
its threshold and source, the sweep setting, and hashes of the comparison and
bootstrap inputs. The build holds the comparison and bootstrap checkpoint locks
together while it reads and hashes them; it rejects a stale comparison or a
bootstrap selection that is missing or differs from the resolved selection. The
figure loader checks the input hashes, selection and sweep setting against the
saved product and refuses a stale product. The selection is checked again during
the build but is not locked throughout that checkpoint window.

The bootstrap figures of this layer render the default Candidate-graph checkpoint (the figure inventory titles one of them "Band-smoothed candidate graph"); the repository does not show them to be drawn per BY graph, and this guide does not claim it.

### What the bootstrap persistence band does and does not say

- It reflects fixed-graph fitness perturbation only: the graph and the threshold are held fixed while measured fitness is resampled.
- It omits threshold-selection uncertainty, graph-selection uncertainty, and the uncertainty and covariance of the standard-error estimates.
- It gives no calibrated coverage (it is not a confidence interval) and no peak-level false discovery rate. The number of retained peaks is a count of peaks above 2c, not an interval for the true number of peaks.

## Including figures in LaTeX

Result file names carry parameter values joined with `=` (for example `thr=published_fdr=by_q=0.05`), which `\includegraphics` cannot parse unless the whole path is wrapped in a second pair of braces:

```latex
\includegraphics{{figures/tree_persistence/tree_persistence__thr=published_fdr=none_tree=nominal_hash=a1b2c3d4_style=annotated.png}}
```

Use the exact file name from the product's manifest or the figure inventory (the name above is the illustrative one of the specification). Figure numbers and captions live only in the manuscript source, so renumbering never renames a file.

## Verification

Check each of these before relying on a layer; counts are the ones recorded in the repository.

- **Inputs:** every file of the roster in Specification 1, section 5A matches its SHA-256; `uv.lock` matches section 5B. The BY classification snapshot has 261,333 rows and 261,333 unique `dna_sequence` per table; counts (functional, uncertain, non-functional): Papkou cutoff 15,851, 2,179, 243,303; Topographer cutoff 12,960, 666, 247,707. Their original SHA-256 are the `[reference_hashes]` of `config/data.toml`.
- **Leaf counts and members (the BY states rebuilt clean have identical counts):** Full nominal 4,055 leaves (261,333 sequences in total); Papkou nominal 514 (135,178); Topographer nominal 158 (102,575); Papkou + FDR nominal 242 and band 51 (118,125); Topographer + FDR nominal 135 and band 29 (98,131). Each state has one root member, and a band tree has the same graph as its nominal tree. The saved tree's leaf count equals the registry's split-tree peak count.
- **Landscape summary table:** seven columns in the order of `[tables.landscape_summary.labels]`; a band column carries tree-level rows only and the build refuses one whose graph-level value differs from its nominal column; the sidecar's `table_identity_sha256` covers the seven scientific ids only, so it is unchanged by an appearance edit.
- **Registry:** `just manuscript-quantities` turns an absent upstream into named gap rows; a `missing:` or `stale:` row means a product above is absent or out of date.
- **Tests:** `PYTHONPATH=src:. uv run pytest -q` runs the full suite; narrow it with pytest node IDs.

## Status and known gaps

- **Resolved (clean BY classification):** the BY classification is written by `just classification-probability-scans` to `data/processed/classification/<variant>`; the BY trees, landscapes, leaves, `landscape_summary`, layouts, ridges and ridge_entropy_sweeps products are rebuilt from it and read from `data/processed/<kind>/<state>` (the state names are in `config/figures.toml`).
- **Manuscript quantity registry identity:** the upstream identity of the registry changed because the Full and Published `landscape_summary` labels were re-pinned in PR3 (`fe9f017c` to `f951fa94`, `4a58cdf8` to `64743e15`). No registry output is committed; it is regenerated at the next publication build.
- **Outstanding for publication:** the earlier variant results (21 PNGs and manifests per variant; the `docs/variants/` snapshot is deleted) were built from a dirty classification table; rebuild or annotate before publication; the figure-by-figure review of the saved-tree views is a blocker before any of them is cited.
- **D10 views:** the views of the presentation specification (`tree_persistence`, `tree_comparison_thresholds`, `tree_comparison_fdr`, `tree_comparison_band`, `bootstrap_band`, `leaves`) are registry figure families (`tree_persistence`, `bootstrap_band`, `tree_comparison_views`) built by `build figures`, part of the default figures and selectable with `--figure`; there is no separate `d10-views` command. The `fitness_uncertainty` view is only a name in the naming catalog (`figures/view_naming.py`), and `merge_flow` is retired.
- **Reproduction evidence:** a full clean-checkout regeneration of the whole story is not recorded yet (FT-REQ-06 is partially implemented).

## What this guide does not claim

- That the Layer 4 bootstrap figures are drawn per BY graph.
- A fresh-checkout command for the `leaves` products.
