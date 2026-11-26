# output/figures/supplementary

The supplementary figures, titleless, named `figS<N>_<desc>.png`.

## What
Membership and numbers are listed in `config/manuscript.toml`:

| File | Source |
| :--- | :--- |
| `figS1_fitness_and_se_barseqfit_agreement.png` | `papkou2023_fitness_and_se_barseqfit_agreement` |
| `figS2_fdr_scan.png` | `functional_counts_vs_q` |
| `figS3_tree_comparison.png` | `tree_comparison` |
| `figS4_branch_vertex_counts.png` | `branch_vertex_counts_2x2` |
| `figS5_papkou_fdr_tree_combined.png` | `tree_persistence_combined__thr=published_fdr=by_q=0.05_tree=nominal_style=annotated` |
| `figS6_bootstrap_global_stability.png` | `bootstrap_global_stability` |
| `figS7_peaks_surviving_threshold.png` | `bootstrap_band_quantile_scan_grid` |
| `figS8_band_quantile_scan.png` | `bootstrap_band_quantile_scan_seeds` |
| `figS9_papkou_fdr_bootstrapped_combined.png` | `bootstrap_combined__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_style=annotated` |
| `figS10_papkou_leaf_swarm.png` | `published_leaf_swarm` |
| `figS11_papkou_fdr_bootstrapped_leaf_swarm.png` | `published_leaf_swarm_band` |
| `figS12_topographer_fdr_leaf_swarm.png` | `candidate_leaf_swarm` |
| `figS13_topographer_fdr_bootstrapped_leaf_swarm.png` | `candidate_leaf_swarm_band` |
| `figS14_graph_codon_entropy_full_papkou.png` | `graph_codon_entropy_fullpublished` |

Exactly S1 to S14. Every other canonical render is an opt-in or exploratory figure and is written to `../exploratory/`.

## How it is generated
Copies made by the namer from `../canonical/`; see `../README.md`.

## Command
`just name-outputs`

## Inputs
`../canonical/<source>.png`, `config/manuscript.toml`.

## Outputs
`figS<N>_<desc>.png`; titled twins in `../titled/supplementary/`.

## Provenance
`../manifest.csv`.

## Caveats
Do not edit here: the namer overwrites and prunes this directory.
