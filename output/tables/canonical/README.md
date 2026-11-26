# output/tables/canonical

Stage exports and the manuscript table sources, unnumbered. The namer copies a stem into `../main/` or `../supplementary/` from `config/manuscript.toml`.

## What
| Stem | Manuscript |
| --- | --- |
| `graph_descriptive_statistics` | Table 1. Columns Full, Papkou, Topographer, Topographer + FDR (BY, q = 0.05). Rows: nodes, edges, % functional, mean degree (undirected), mean in/out degree (directed uphill). |
| `candidate_band_comparison` | Table 2. Split tree of the BY q = 0.05 Topographer graph, and that tree after bootstrap-band simplification. |
| `papkou2023_full_vs_published` | Supplementary S1, the former main Table 1 (landscape-summary slice). |
| `papkou2023_candidate_corrections` | Supplementary S2, the former main Table 2 (landscape-summary slice). |

The descriptive bundles (README, TSV, PNG) live in a directory of the same stem. The namer reads the flat `csv`, `html`, `md` and `json` in this directory.

Tracked in git: this README and the stems `config/manuscript.toml` names, via the allow-list `scripts/dev/sync_figure_gitignore.py` regenerates.

## How it is generated
`just graph-descriptive-statistics`, `just candidate-band-comparison-table` and `just presentation-tables`.

## Command
Those three recipes, then `just name-outputs`.

## Caveats
Do not edit files here by hand: the namer overwrites the numbered copies.
