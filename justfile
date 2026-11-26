set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments
# Read .env (PAPKOU2023_BARSEQ_EXPORT, PAPKOU2023RUGGED_DATA, ...) before the exports below, so it is not shadowed.
set dotenv-load
python := env("PYTHON", ".venv/bin/python")

# The BarSeq export folder (counts.parquet, samples.parquet, sequences.parquet) that `just barseq-export` writes
# in the barseqfit checkout ($BARSEQFIT, default ../barseqfit). PAPKOU2023_BARSEQ_EXPORT, from the shell or from .env
# (loaded above), wins; the barseqfit folder is only the fallback when it is unset.
barseqfit := env("BARSEQFIT", "../barseqfit")
barseq_export := env("PAPKOU2023_BARSEQ_EXPORT", "")
export PAPKOU2023_BARSEQ_EXPORT := if barseq_export != "" { barseq_export } else { env("BARSEQFIT_SCRATCH", barseqfit + "/.scratch/papkou2023") }

# List the available recipes.
default:
    @just --justfile {{justfile()}} --list

# Download the Papkou et al. 2023 Zenodo record (doi:10.5281/zenodo.13837982) and extract only the files used into
# $PAPKOU2023RUGGED_DATA (store layout barseqfit reads); verifies every md5. Resumable. Extra args: --store, --authors-scripts-from (overrides the Zenodo 8229097 download), --delete-zip.
fetch-papkou2023 *args:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.rawdata.papkou_fetch {{args}}

# Download our own Zenodo record (config/data.toml [ours] zenodo_record): the data/processed folder and the Papkou store files, md5-verified, then check data/processed against its MANIFEST.tsv.
fetch-data *args:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.rawdata.fetch_data {{args}}
    {{just_executable()}} check-manifest

# Rebuild data/raw/ from the Papkou store (R reads the RDS files, polars transforms) and run the shape check. Needs R with igraph.
make-raw out="data/raw" *args:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.rawdata.make_raw --out {{out}} {{args}}

# One-off: compare rebuilt raw files (default data/raw) with an existing copy by value; exits 1 on any difference.
check-raw-against existing rebuilt="data/raw":
    PYTHONPATH=src {{python}} -m dubanevics2026topological.rawdata.check_raw {{existing}} --rebuilt {{rebuilt}}

# Export the BarSeq count tables from the Papkou store with barseqfit (`just export-papkou` in $BARSEQFIT, default ../barseqfit) into PAPKOU2023_BARSEQ_EXPORT.
barseq-export:
    cd "{{barseqfit}}" && just export-papkou

# The inputs of `compute`: the Papkou files, data/raw/ built from them, and the BarSeq export.
data: fetch-papkou2023 make-raw barseq-export

# Copy the finished stage outputs of output/ into data/processed/ under their state names (idempotent).
processed-collect:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.results.processed collect

# Collect everything into data/processed/ and write MANIFEST.tsv (path, bytes, sha256) last. data/processed is the Zenodo deposit.
processed-manifest: processed-collect
    PYTHONPATH=src {{python}} -m dubanevics2026topological.results.processed write

# Verify data/processed against its MANIFEST.tsv: missing, extra, wrong-size and wrong-hash files fail.
check-manifest:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.results.processed check

# Copy the plain-file folders of data/processed/ back to where the figure inputs read them (growth rates, comparisons, tables).
processed-restore:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.results.processed restore

# Everything that is computed, from the Papkou download to the quantity registry and tables. Writes output/ and collects it into data/processed/; no figure is rendered.
compute: data selection comparison-analysis four-graph-comparisons landscape-products bootstrap-draws bootstrap-band-quantile-scan-product leaves-product tree-layout-products landscape-summary-product ridges-product ridge-entropy-product merge-flow-product component-analysis peak-analysis pooled-scatter-product fdr-pooled-scatter-product peak-panels-product threshold-diagnostics-product codon-entropy-analysis bootstrap-codon-entropy-product graph-tree-persistence-product candidate-edge-analysis leading-saddle-band graph-descriptive-statistics tables

# What every selected-graph stage reads: the growth rates, the mixture classification and the BY classification of data/processed/classification/.
selection: sequence-growth-rates functional-classification classification-probability-scans

# Run the 6-nt toy selection, tree, and 50-draw bootstrap into an isolated output root.
# The 9-nt codon-entropy stage is intentionally not run.
toy:
    FITNESSTREE_OUTPUT_ROOT=output/toy PYTHONPATH=src {{python}} scripts/toy_pipeline.py

# Build and verify the base analysis checkpoint.
analysis:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build verify-analysis

# Export the science tables and the quantity registry. The registry runs after its upstream stages, and the seven-column table last, which reads saved products; the leading-saddle-band stage runs before the registry (a declared upstream); the bootstrap and branch outputs are read when present (their quantities are named gap rows otherwise).
tables: selection analysis component-analysis peak-analysis codon-entropy-analysis leading-saddle-band biological-annotation-tables landscape-summary-product
    FITNESSTREE_REQUIRE_PAPER_ROWS=1 PYTHONPATH=src {{python}} -m dubanevics2026topological.build manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build tables
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build landscape-summary-table
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build presentation-tables

# Manuscript Table 2: BY q=0.05 Topographer split tree vs its band-simplified tree. Reads the saved bootstrap (draws none); a step of `just figures`.
candidate-band-comparison-table:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build candidate-band-comparison-table

# The selected landscape checkpoint is the input to this derived analysis.
# Refresh its science explicitly with comparison-analysis; appearance builds
# must not rebuild the upstream graph merely to check these derived tables.
# Under just, figure recipes reach it through the package's content-hash check rather than a file-exists rule.
# Build the threshold-selected landscape comparison analysis.
comparison-analysis: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build comparison-analysis

# Publish all five labelled landscape and nominal-tree products, then collect them into data/processed/{landscape,tree}/<state>.
landscape-products: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build landscape-products
    {{just_executable()}} processed-collect

# Export the biological annotation tables of the cached Published landscape (the registry reads them for
# annotations.published.*; no figure build writes them).
biological-annotation-tables: comparison-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build biological-annotation-tables

# Build the manuscript quantity registry and report from the finished upstream outputs
# (output/tables/manuscript_quantities/); an absent upstream becomes named gap rows.
manuscript-quantities: comparison-analysis component-analysis leading-saddle-band biological-annotation-tables
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build manuscript-quantities

# Render the three-tree comparison figure.
comparison-figure: comparison-analysis manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure tree_comparison --variant annotated

# Render the Candidate tree histogram figure.
candidate-histogram-figure: comparison-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure candidate_tree_histogram --variant annotated

# Render the Full tree histogram figure.
full-histogram-figure: comparison-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure full_tree_histogram --variant annotated

# Render the Article tree histogram figure.
published-histogram-figure: comparison-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure published_tree_histogram --variant annotated

# Render the study tree persistence metrics figure.
study-metrics-figure: comparison-analysis manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure study_tree_persistence_metrics --variant vanilla

# Render the study tree persistence biology figure.
study-biology-figure: comparison-analysis manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure study_tree_persistence_biology --variant annotated

# Compute the peak uncertainty analysis.
peak-analysis: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build peak-analysis

# Publish the pooled scatter tables from the verified growth rates + peak checkpoint.
pooled-scatter-product: sequence-growth-rates peak-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build pooled-scatter-product

# Publish the median-reflected FDR variant's pooled scatter under output/variants/<variant>/, with its own peak checkpoint.
fdr-pooled-scatter-product: sequence-growth-rates
    FITNESSTREE_ANALYSIS_VARIANT=threshold=reflected_fdr=by_q=0.05 FITNESSTREE_OUTPUT_ROOT=output/variants/threshold=reflected_fdr=by_q=0.05 PYTHONPATH=src {{python}} -m dubanevics2026topological.build fdr-pooled-scatter-product

# Publish the four graphs of the peak figure under output/variants/<variant>/: for each of point_papkou_original, point_median_reflected, fdr_by_q0.05_papkou_original and fdr_by_q0.05_median_reflected, its own peak checkpoint, pooled_scatter product and peak_panel_summary (threshold, peak count, sequence counts, largest component). No bootstrap draw; an existing matching checkpoint or product is reused.
peak-panels-product: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build peak-panels-product

# Publish the threshold diagnostics from the verified selection + component data.
threshold-diagnostics-product: selection component-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build threshold-diagnostics-product

# Publish the bootstrap codon entropy populations, sweeps, and precomputed render tables.
# Needs the comparison and bootstrap checkpoints only; no layout product is a parent
# (mixture_quantile has none), so the figure loads this saved product (PR 3b).
bootstrap-codon-entropy-product: selection comparison-analysis bootstrap-smoothed-tree
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build bootstrap-codon-entropy-product

# Publish the conceptual graph/split-tree/persistence construction as a saved product.
graph-tree-persistence-product:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build graph-tree-persistence-product

# Render the peak uncertainty figure (the four graphs). It reads only the saved
# variant panels of `just peak-panels-product`; a render never recomputes them.
peak-figure: selection peak-panels-product
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure peak_uncertainty --variant vanilla

# Render the opt-in peak-rank figures (Reference peak set ranked by fitness; annotated adds stop-codon crosses, histogram adds the all-sequence fitness marginal). They read the Full-graph fitness table and data/raw/papkou2023_peaks.csv. Not part of `figures`.
peak-rank-figures:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure peak_rank --figure peak_rank_annotated --figure peak_rank_histogram

# A step of `compute`: the clamped comparison of each of the four graphs the 2x2 figures draw (Published and Candidate thresholds, nominal and BY q = 0.05), each under output/variants/<variant>/ and collected into data/processed/comparison/<state> (build/four_graph_inputs.py reads them).
four-graph-comparisons: selection
    {{just_executable()}} variant-comparison-analysis "threshold=published_fdr=none"
    {{just_executable()}} variant-comparison-analysis "threshold=reflected_fdr=none"
    {{just_executable()}} variant-comparison-analysis "threshold=published_fdr=by_q=0.05"
    {{just_executable()}} variant-comparison-analysis "threshold=reflected_fdr=by_q=0.05"

# Render the four-graph figures: the two BY leaf swarm panels (opt-in; the nominal panels are published_leaf_swarm and candidate_leaf_swarm) and the branch vertex count 2x2 (a default figure of `figures`). They read only the saved variant comparisons of `just four-graph-comparisons`.
four-graph-figures: four-graph-comparisons
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure published_fdr_leaf_swarm --figure candidate_fdr_leaf_swarm --figure branch_vertex_counts_2x2

# Render the supplementary genotype figures: the Papkou agreement hexbin, fitness against its SE with Spearman rho (a default figure), Papkou published beside our estimate (wild type a red star), and the class proportions against the Benjamini-Hochberg and Bonferroni q (y clipped at 0.1; the BY one is a default figure). They read the growth-rate table, the published fitness table and the classification scan; nothing is refit. The others are opt-in.
supplementary-figures: sequence-growth-rates classification-probability-scans
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure papkou_agreement --figure fitness_se_correlation --figure fitness_vs_pooled_se --figure functional_counts_vs_q --figure functional_counts_vs_q_bh --figure functional_counts_vs_q_bonferroni

# Compute the codon entropy analysis.
codon-entropy-analysis: comparison-analysis peak-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build codon-entropy-analysis

# Render the rotated count-label variant of the band-tree figures into <figure_id>_rotated.png. Opt in; not part of `just figures`.
figures-rotated: bootstrap-codon-entropy-product graph-tree-persistence-product manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --label-collision rotate

# Manuscript Table 1: four-graph descriptive statistics under output/tables/canonical/graph_descriptive_statistics/,
# with flat csv/html/md/json copied beside that directory for the namer. A step of `just figures` (and of `compute`).
graph-descriptive-statistics:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build graph-descriptive-statistics

# Render every manuscript figure and table from data/processed/ alone: check MANIFEST.tsv, restore the plain-file folders, build the figures and the seven-column tables, then name-outputs copies them to main/, supplementary/ and titled/. Computes nothing; `just all` or `just fetch-data` fills data/processed first.
figures: check-manifest
    {{just_executable()}} processed-restore
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures
    {{just_executable()}} landscape-summary-table
    {{just_executable()}} presentation-tables
    {{just_executable()}} graph-descriptive-statistics
    {{just_executable()}} candidate-band-comparison-table
    {{just_executable()}} name-outputs

# Render every registered figure outside the default selection (the ones no manuscript entry needs) into canonical/ with `--keep-going`, then name-outputs copies them to exploratory/. Some of them need extra products and may fail; the failures are listed at the end. Run after `just figures`.
figures-exploratory:
    #!/usr/bin/env bash
    set -uo pipefail
    status=0
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --keep-going --opt-in || status=$?
    {{just_executable()}} name-outputs
    exit "$status"

# Compute the network component persistence analysis.
component-analysis:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build component-analysis

# Render the full-graph tree persistence figure.
full-tree-figure: selection component-analysis analysis manuscript-quantities threshold-diagnostics-product
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure full_tree_persistence

# Render canonical figures as SVG only (PNGs are left untouched); optional figure id, e.g. just figures-svg tree_comparison.
figures-svg figure="": selection analysis component-analysis comparison-analysis peak-analysis pooled-scatter-product fdr-pooled-scatter-product peak-panels-product threshold-diagnostics-product candidate-edge-analysis bootstrap-codon-entropy-product graph-tree-persistence-product manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --format svg {{ if figure == "" { "" } else { "--figure " + figure } }}

# Compute everything, collect and manifest data/processed, then render the figures from it.
all: compute processed-manifest figures

# Rebuild affected exports when sources change.
watch:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build watch

# Compute the Candidate edge evidence analysis.
candidate-edge-analysis: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build candidate-edge-analysis

# Not part of `all` or `figures`: needs the BarSeq export named in .env.
candidate-tree-contrast: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build candidate-tree-contrast

# Fits the growth rates from the BarSeq export (PAPKOU2023_BARSEQ_EXPORT; the stage hashes the export) and collects them into data/processed/growth_rates.
sequence-growth-rates:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build sequence-growth-rates
    {{just_executable()}} processed-collect

# The growth-rate checkpoint of one analysis variant under output/variants/<variant>/. Every output root keeps its own checkpoint and the bootstrap of a variant reads the one of its root, so a variant bootstrap needs this first. The fit does not depend on the variant, so it is the same fit as sequence-growth-rates and needs the BarSeq export too.
sequence-growth-rates-by variant:
    FITNESSTREE_ANALYSIS_VARIANT={{variant}} FITNESSTREE_OUTPUT_ROOT=output/variants/{{variant}} PYTHONPATH=src {{python}} -m dubanevics2026topological.build sequence-growth-rates

# A prerequisite of `tables`, `figures` and `all` (every Candidate stage reads its checkpoint):
# computes the mixture-quantile threshold and the Functional probability classes from the
# growth-rate checkpoint. Like sequence-growth-rates it needs PAPKOU2023_BARSEQ_EXPORT.
functional-classification: sequence-growth-rates
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build functional-classification

# The bootstrap of the default selection (the median-reflected BY graph). A finished run is published to data/processed/bootstrap/reflected_by; an existing matching checkpoint, slot or processed run is reused and nothing is drawn.
bootstrap-smoothed-tree: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build bootstrap-smoothed-tree

# The bootstrap of one analysis variant under output/variants/<variant>/: threshold=published_fdr=by_q=0.05 publishes data/processed/bootstrap/published_by.
bootstrap-smoothed-tree-by variant: selection (sequence-growth-rates-by variant)
    FITNESSTREE_ANALYSIS_VARIANT={{variant}} FITNESSTREE_OUTPUT_ROOT=output/variants/{{variant}} PYTHONPATH=src {{python}} -m dubanevics2026topological.build bootstrap-smoothed-tree

# The two BY bootstraps (published_by, reflected_by) of data/processed/bootstrap/.
bootstrap-draws: bootstrap-smoothed-tree (bootstrap-smoothed-tree-by "threshold=published_fdr=by_q=0.05")

# Scan the band quantile over the default bootstrap checkpoint; the scan is published to data/processed/band_quantile_scan/reflected_by.
bootstrap-band-quantile-scan: bootstrap-smoothed-tree
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build bootstrap-band-quantile-scan

# The scan of one BY variant (published_by for threshold=published_fdr=by_q=0.05).
bootstrap-band-quantile-scan-by variant: (bootstrap-smoothed-tree-by variant)
    FITNESSTREE_ANALYSIS_VARIANT={{variant}} FITNESSTREE_OUTPUT_ROOT=output/variants/{{variant}} PYTHONPATH=src {{python}} -m dubanevics2026topological.build bootstrap-band-quantile-scan

# Both scans of data/processed/band_quantile_scan/ (reflected_by and published_by).
bootstrap-band-scans: bootstrap-band-quantile-scan (bootstrap-band-quantile-scan-by "threshold=published_fdr=by_q=0.05")

# Publish the band quantile scan and survival tables as a saved product (merged into data/processed/band_quantile_scan/reflected_by by `processed-collect`).
bootstrap-band-quantile-scan-product: bootstrap-band-scans
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build bootstrap-band-quantile-scan-product

# Publish the leaves of the seven states and the two band trees from the processed nominal trees and bootstrap evidence; collected into data/processed/{leaves,tree}/<state>.
leaves-product: landscape-products bootstrap-draws
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build leaves-product
    {{just_executable()}} processed-collect

# Publish the layouts (and the BY annotations) of the seven states from the processed trees and leaves.
tree-layout-products: leaves-product
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build tree-layout-products

# Publish one landscape_summary product per state (seven) from the processed leaves, trees and landscapes; it computes no tree, graph or threshold.
landscape-summary-product: leaves-product
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build landscape-summary-product

# Publish one ridges product (KEM/TCL ridge summary, DNA-keyed membership, sidecar) per classifiable state from the processed leaves, trees and landscapes; Full has no functional class and is skipped.
ridges-product: leaves-product
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build ridges-product

# Publish one merge_flow product per state from the processed leaves, trees and landscapes.
merge-flow-product: leaves-product
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build merge-flow-product

# Publish one ridge_entropy_sweeps product (core, KEM and TCL positional codon entropy sweeps and populations) per band tree from the processed leaves, trees, landscapes and ridges.
ridge-entropy-product: ridges-product
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build ridge-entropy-product

# Export the seven-column landscape summary (CSV, HTML, Markdown, JSON sidecar) from the seven landscape_summary states of config/figures.toml; it computes nothing.
landscape-summary-table:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build landscape-summary-table

# Supplementary landscape-summary slices: S1 Full vs Published, S2 Candidate Nominal|FDR|Bootstrap,
# and the unnumbered Published simplification. Main Tables 1 and 2 are the descriptive tables.
# A step of `tables` after landscape-summary-table and of the `figures` chain after manuscript-quantities.
presentation-tables:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build presentation-tables

# Not part of `all` or `figures`: sensitivity scan of the functional threshold (ADR 2026-10-24, Decisions 7 and 8).
# Computes every resample here (about 220-315 s each at n_init = 5, over n_jobs workers) after the point scan.
# Skips a current scan; --force recomputes.
functional-threshold-scan: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build functional-threshold-scan

# Measurement-uncertainty classification and probability scans for the two fixed cutoffs (Papkou original and median-reflected). Reads data/processed/growth_rates and writes data/processed/classification/<published|reflected>/ (the BY table of each cutoff and the scan provenance).
classification-probability-scans: sequence-growth-rates
    PYTHONPATH=src {{python}} -m dubanevics2026topological.threshold.classification_probability_scans

# Opt-in (not a default figure): needs the bootstrap-band-quantile-scan table.
# PAPKOU2023_BARSEQ_EXPORT must be set even on a cache hit (bootstrap-band-quantile-scan depends on bootstrap-smoothed-tree).
bootstrap-band-quantile-scan-figure: bootstrap-band-quantile-scan-product manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure bootstrap_band_quantile_scan --variant annotated

# A default figure of `figures`; this recipe renders it alone from the bootstrap-band-quantile-scan table.
# PAPKOU2023_BARSEQ_EXPORT must be set even on a cache hit (bootstrap-band-quantile-scan depends on bootstrap-smoothed-tree).
bootstrap-band-quantile-scan-seeds-figure: bootstrap-band-quantile-scan-product manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure bootstrap_band_quantile_scan_seeds --variant annotated

# Opt-in (not a default figure): needs the bootstrap-smoothed-tree checkpoint.
# PAPKOU2023_BARSEQ_EXPORT must be set even on a cache hit (bootstrap-smoothed-tree depends on sequence-growth-rates, which hashes the export).
bootstrap-biology-figure: bootstrap-smoothed-tree comparison-analysis manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure candidate_tree_bootstrap_biology --variant annotated

# A default figure of `figures`; this recipe renders it alone from the bootstrap-smoothed-tree checkpoint.
# PAPKOU2023_BARSEQ_EXPORT must be set even on a cache hit (bootstrap-smoothed-tree depends on sequence-growth-rates, which hashes the export).
bootstrap-tree-histogram-figure: bootstrap-smoothed-tree comparison-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure bootstrap_tree_histogram --variant annotated

# A default figure of `figures`; this recipe renders it alone. It consumes the bootstrap-codon-entropy-product (populations, entropy sweeps, precomputed render tables).
# Needs the bootstrap-smoothed-tree checkpoint at build time for the band view used to derive the tables.
# PAPKOU2023_BARSEQ_EXPORT must be set even on a cache hit (bootstrap-smoothed-tree depends on sequence-growth-rates, which hashes the export).
bootstrap-codon-entropy-figure: bootstrap-smoothed-tree comparison-analysis bootstrap-codon-entropy-product manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure bootstrap_codon_entropy --variant annotated

# A default figure of `figures`; this recipe renders it alone from the bootstrap-smoothed-tree checkpoint.
# PAPKOU2023_BARSEQ_EXPORT must be set even on a cache hit (bootstrap-smoothed-tree depends on sequence-growth-rates, which hashes the export).
bootstrap-global-stability-figure: bootstrap-smoothed-tree comparison-analysis manuscript-quantities
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure bootstrap_global_stability --variant annotated

# Not part of `all` or `figures`: draws the horizontal tree layouts side by side (exploration).
# Builds or reuses the comparison and peak analyses, then writes output/figures/layout_exploration/.
layout-exploration: comparison-analysis peak-analysis
    PYTHONPATH=src {{python}} scripts/layout_exploration.py

# Render the Candidate edge evidence figure.
candidate-edge-figure: candidate-edge-analysis
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build figures --figure candidate_edge_evidence --variant vanilla

# Exploratory: not part of `all` or `figures`. Reads the bootstrap checkpoint and never draws, so it does not depend on `bootstrap-smoothed-tree`.
# The loader restores the processed bootstrap/<state>/checkpoint when output/ has no matching checkpoint, and moves a non-matching one aside to .replaced-*; it needs data/raw/ to rebuild the nominal tree.
branch-decomposition: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build branch-decomposition

# Peak-pair distances and Mantel tests on the band-simplified tree of the median-reflected Benjamini-Yekutieli Candidate graph (docs/peak-pair-distance-design.md). Reads the bootstrap checkpoint and never draws; builds the Candidate split tree from data/raw/ (about a minute). Writes output/variants/threshold=reflected_fdr=by_q=0.05/analysis/peak_pair_distances/. Not part of `all` or `figures`.
peak-pair-distances:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build peak-pair-distances

# Reads the cached component pairs and the Full-graph fitness; run `just component-analysis` first. Not part of `all` or `figures`.
leading-saddle-band: selection
    PYTHONPATH=src {{python}} -m dubanevics2026topological.build leading-saddle-band

# Analysis variants: config/variants/<name>.toml merged over config/analysis.toml (fdr_by_q0.05_median_reflected, fdr_by_q0.05_papkou_original). A variant selects its Candidate graph from the processed FDR classification table and runs under its own output root, output/variants/<name>/, so no default output is touched. It needs no mixture functional-classification.
variant-comparison-analysis variant:
    FITNESSTREE_ANALYSIS_VARIANT={{variant}} FITNESSTREE_OUTPUT_ROOT=output/variants/{{variant}} PYTHONPATH=src {{python}} -m dubanevics2026topological.build comparison-analysis

# Three-variant graph comparison (published point estimates, Papkou BY nominal 5%, median-reflected BY nominal 5%) into output/analysis/fdr_graph_comparison/.
fdr-graph-comparison:
    PYTHONPATH=src {{python}} scripts/fdr_graph_comparison.py

# The last step of `figures` and `figures-exploratory`: copy canonical renders to manuscript-numbered names (main/, supplementary/, titled/) and the rest to exploratory/, and write output/figures/manifest.csv. Does not render.
name-outputs:
    PYTHONPATH=src {{python}} -m dubanevics2026topological.naming
