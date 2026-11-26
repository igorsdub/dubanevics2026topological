# output/figures/supplementary

The supplementary figures, titleless, named `figS<N>_<desc>.png`.

## What
Membership and numbers are listed in `config/manuscript.toml`:

| File | Source |
| :--- | :--- |
| `figS1_full_tree_persistence_top_histogram.png` | `full_tree_persistence` |
| `figS2_tree_comparison.png` | `tree_comparison` |
| `figS3_published_tree_histogram.png` | `published_tree_histogram` |
| `figS4_full_tree_histogram.png` | `full_tree_histogram` |
| `figS5_candidate_tree_histogram.png` | `candidate_tree_histogram` |
| `figS6_branch_vertex_counts_2x2.png` | `branch_vertex_counts_2x2` |
| `figS7_fitness_se_correlation.png` | `fitness_se_correlation` |
| `figS8_functional_counts_vs_q.png` | `functional_counts_vs_q` |
| `figS9_reflected_fdr_trees.png` | `reflected_fdr_trees` |
| `figS10_published_fdr_tree.png` | `tree_persistence__thr=published_fdr=by_q=0.05_tree=nominal_style=annotated` |
| `figS11_bootstrap_tree_histogram.png` | `bootstrap_tree_histogram` |
| `figS12_global_stability.png` | `bootstrap_global_stability` |
| `figS13_band_quantile_scan.png` | `bootstrap_band_quantile_scan_grid` |
| `figS14_band_quantile_scan_seeds.png` | `bootstrap_band_quantile_scan_seeds` |
| `figS15_candidate_bootstrap_simplification_stops.png` | `bootstrap_stops__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_style=annotated` |
| `figS16_candidate_bootstrap_simplification_synonymous.png` | `bootstrap_synonymous__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_style=annotated` |
| `figS17_candidate_bootstrap_simplification_combined.png` | `bootstrap_combined__thr=reflected_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_style=annotated` |
| `figS18_published_bootstrap_simplification_stops.png` | `bootstrap_stops__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_style=annotated` |
| `figS19_published_bootstrap_simplification_synonymous.png` | `bootstrap_synonymous__thr=published_fdr=by_q=0.05_draws=10000_seed=20261019_tree=band_bandq=0.95_style=annotated` |
| `figS20_published_leaf_persistence.png` | `published_leaf_persistence` |
| `figS21_published_leaf_swarm.png` | `published_leaf_swarm` |
| `figS22_candidate_leaf_persistence.png` | `candidate_leaf_persistence` |
| `figS23_candidate_leaf_swarm.png` | `candidate_leaf_swarm` |
| `figS24_candidate_leaf_persistence_band.png` | `candidate_leaf_persistence_band` |
| `figS25_candidate_leaf_swarm_band.png` | `candidate_leaf_swarm_band` |

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
