# output/tables

Exported tables: every stage export by stable name, and the manuscript tables by number.

## What
| Directory | Content |
| :--- | :--- |
| `canonical/` | Every stage export, unnumbered (today's `output/tables/` content, such as `network_component_persistence.csv`, the manuscript-quantity registry and the biological annotation tables) |
| `main/` | `tab1_graph_summary`, `tab2_candidate_corrections` |
| `supplementary/` | Supplementary tables, `tabS<N>_<desc>`, none yet |
| `manifest.csv` | Every numbered copy with its SHA-256 |

Table 1 summarises the graphs, one column each for Full, Published and Candidate (no FDR, no bootstrap): functional threshold (not applicable for Full), functional and non-functional sequences, vertices, edges, peaks, valleys and mean, median and maximum fitness. Table 2 follows the Candidate tree through its corrections, one column each for no correction, BY 5% FDR and bootstrap simplification: functional and non-functional sequences and peaks. The FDR level and band quantile go in its caption.

## How it is generated
Stage exports write `canonical/`; the namer copies numbered tables. Tables 1 and 2 are built by `presentation-tables` (a step of `just tables` and of the `figures` chain) and named from the `[[table]]` entries of `config/manuscript.toml`; the Published and Candidate valley cells are an intentional em dash because no per-selection valley product exists. The Full valley cell comes from the manuscript quantity registry, which `manuscript-quantities` refreshes earlier in the same chain.

## Command
`just presentation-tables`, then `just name-outputs`.

## Inputs
Stage checkpoints under `output/analysis/`; settings in `config/analysis.toml`.

## Outputs
CSV and Parquet for each table, plus a Markdown or LaTeX rendering for the manuscript tables. No HTML.

## Provenance
Each export's stage checkpoint holds the authoritative bytes and its fingerprint; `manifest.csv` holds the copies' hashes.

## Caveats
Sequence identities are exported as sequences, never as graph indices.
