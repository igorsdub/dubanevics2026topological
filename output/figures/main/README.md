# output/figures/main

The main-text figures of the manuscript, titleless, named by number.

## What
| File | Content | Source |
| :--- | :--- | :--- |
| `fig1_graph_tree_persistence.png` | From graph to merge tree to persistence diagram, on a small example | `graph_tree_persistence` |
| `fig2_workflow.png` | Workflow and tree annotations | external (Google Slides) |
| `fig3_published_tree_metrics.png` | Published tree and persistence: branch vertex count, AA Hamming distance, Horton–Strahler order | `study_tree_persistence_metrics` |
| `fig4_published_tree_biology.png` | Published tree and persistence: stop-codon and synonymous peaks | `study_tree_persistence_biology` |
| `fig5_full_tree_persistence.png` | Full tree with split- and join-tree persistence and the vertical fitness histogram | `full_tree_persistence_vertical_histogram` |
| `fig6_peak_uncertainty.png` | Peak uncertainty: Published and Candidate graphs, with and without BY q = 0.05 | `peak_uncertainty` |
| `fig7_merge_tree_simplification_parametric_bootstrap.png` | Merge-tree simplification with the parametric fitness bootstrap | external (Google Slides) |
| `fig8_candidate_fdr_tree_band_simplification.png` | Candidate FDR tree from nominal to band-simplified | external (Google Slides) |
| `fig9_bootstrap_codon_entropy.png` | Codon entropy of the bootstrapped graph along the two ridges of the simplified tree | `bootstrap_codon_entropy` |

Generated from `config/manuscript.toml`; the table is regenerated with it.

Publication figures carry no figure title, subtitle or panel heading; each panel has a letter (a, b, c) at its top left. The facts the titles carried (counts, thresholds) are in `../manifest.csv` for the captions.

## How it is generated
Copies made by the namer from `../canonical/` and `../external/`; see `../README.md`.

## Command
`just name-outputs`

## Inputs
`../canonical/<source>.png`, `../external/<desc>.png`, `config/manuscript.toml`.

## Outputs
`fig<N>_<desc>.png`; titled twins in `../titled/main/`.

## Provenance
`../manifest.csv` (SHA-256 of each copy and its source stem); the render record of the source.

## Caveats
Do not edit here: the namer overwrites and prunes this directory.
