# output/tables

Exported tables: every stage export by stable name, and the manuscript tables by number.

## What
| Directory | Content |
| :--- | :--- |
| `canonical/` | Every stage export, unnumbered (today's `output/tables/` content, such as `network_component_persistence.csv`, the manuscript-quantity registry and the biological annotation tables) |
| `main/` | `tab1_graph_descriptive`, `tab2_tree_descriptive` |
| `supplementary/` | `tabS1_graph_summary`, `tabS2_topographer_corrections` |
| `manifest.csv` | Every numbered copy with its SHA-256 |

Tracked in git: Tables 1, 2, S1 and S2 (the four stems in `canonical/`, their numbered copies in `main/` and `supplementary/`), `manifest.csv` and the READMEs; every other stage export is local.

Table 1 is the four-graph descriptive table (`graph_descriptive_statistics`). Columns: Full, Papkou, Topographer, Topographer + FDR (BY, q = 0.05). Rows: nodes, edges, % functional, mean degree (undirected), mean in/out degree (directed uphill). An en dash is an undefined cell (Full has no functional threshold). Table 2 is the tree descriptive table (`candidate_band_comparison`): the nominal split tree of the Topographer + FDR graph (BY, q = 0.05) and that tree after bootstrap-band simplification. Supplementary S1 and S2 are the former main tables, the landscape-summary slices `papkou2023_full_vs_published` and `papkou2023_candidate_corrections`.

## How it is generated
Stage exports write `canonical/`; the namer copies numbered tables from the `[[table]]` entries of `config/manuscript.toml`. Tables 1 and 2 are `just graph-descriptive-statistics` and `just candidate-band-comparison-table`; both are steps of `just figures`, before `name-outputs`, and read saved products only. Supplementary S1 and S2 are `just presentation-tables`. The Papkou and Topographer + FDR valley cells of S1 are an intentional em dash because no per-selection valley product exists. The Full valley cell comes from the manuscript quantity registry, which `manuscript-quantities` refreshes earlier in the same chain.

## Command
`just graph-descriptive-statistics` and `just candidate-band-comparison-table` for Tables 1 and 2, `just presentation-tables` for S1 and S2, then `just name-outputs`.

## Inputs
Stage checkpoints under `output/analysis/`; settings in `config/analysis.toml`.

## Outputs
CSV, HTML, Markdown and JSON for each table (`canonical/<stem>.<ext>`, copied by the namer to `main/` and `supplementary/`).

## Provenance
Each export's stage checkpoint holds the authoritative bytes and its fingerprint; `manifest.csv` holds the copies' hashes.

## Caveats
The tracked figures and tables of 1.0.1 were re-rendered from the archived results of the 1.0.0 analysis run (no recomputation), using the display code of this release ported to the code state that produced those results; a fresh `just all` reproduces the same numbers with new provenance records. The `source` fields and `table_identity_sha256` in the JSON sidecars therefore come from the archived run and differ from a fresh rebuild.

Sequence identities are exported as sequences, never as graph indices.
