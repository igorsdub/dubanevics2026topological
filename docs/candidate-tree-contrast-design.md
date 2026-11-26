# Candidate tree contrast design

## Purpose

Each edge of the Candidate split tree joins two critical points, a higher
(upper) and a lower one. The stage asks, for every edge, whether the direct
BarSeqFit contrast between the two endpoints is distinguishable from zero. It
adjusts all edge p-values together as one Benjamini-Hochberg (BH) family over
tree edges only (ADR 0006). The adjustment is nominal: edges are selected from
observed fitness and the tests are dependent. A non-significant contrast does
not establish equal fitness.

## Inputs

- The Candidate split tree, built in memory from the raw fitness tables with
  the candidate cutoff in `config/analysis.toml` (`[clamped] candidate_functional_threshold`).
- The BarSeq export written by `just export-papkou` in the barseqfit
  repository: `counts.parquet`, `samples.parquet`, `sequences.parquet`. The
  folder is named by the environment variable `PAPKOU2023_BARSEQ_EXPORT`
  (set in `.env` or the shell; see the README).
- Settings in `[candidate_tree_contrast]`: `alpha`, `elapsed_time`,
  `time_unit`, `correction = "benjamini_hochberg"`,
  `family = "candidate_split_tree_edges"`.

> Superseded: the Candidate functional threshold is now computed and the Candidate graph is the Functional class of the classification, see ADR 2026-10-24 (the median-reflected rule of ADR 2026-10-23 is the comparison rule).

## Tree construction

The tree is the Candidate split tree of the cutoff-selected landscape, using
the same preparation as the clamped comparison (`prepare_clamped_landscape`,
`analyze_prepared_landscape`). It must have exactly one root. Every node must
carry a DNA sequence identity; Rustworkx indices are never exported as
identifiers.

## Edge enumeration and edge_id

`enumerate_tree_edges` lists every tree edge, upper endpoint first, sorted by
`edge_id`. `edge_id` is `"<upper dna_sequence>|<lower dna_sequence>"`. The
function raises `ValueError` when the tree has no edges, when `edge_id` values
repeat, and when any edge has `upper_fitness < lower_fitness` (the offending
`edge_id` values are listed). An edge is terminal when its upper endpoint is a
maximum with no incoming edge.

Pair columns: `edge_id`, `upper_sequence`, `lower_sequence`, `upper_event`,
`lower_event`, `upper_fitness`, `lower_fitness`, `raw_fitness_delta`
(upper minus lower), `regular_count`, `is_terminal`, `lower_in_degree`.

## Fit call and orientation

DNA sequences are mapped to export `sequence_id` values. Each edge becomes one
pair with `from` = lower endpoint and `to` = upper endpoint, and one call to
`barseqfit.fit_pair` with the configured `elapsed_time` and `time_unit`. The
contrast estimate `m` is therefore the upper minus the lower growth-rate
difference, in the same direction as `raw_fitness_delta`.

## Fail-closed rules

Nothing is adjusted unless every edge has exactly one acceptable fit. The stage
raises `ValueError`, naming the offending `edge_id` values, when:

- a tree sequence is absent from the export, or a DNA sequence maps to several
  `sequence_id` values;
- the fit lacks any of `from`, `to`, `m`, `se`, `p`, `status`, `diagnostics`,
  `elapsed_time`, `time_unit`;
- the fit returns rows for untested pairs, no row for an edge, or several rows
  for an edge;
- `status` is null or not `converged`;
- `m` is not finite, `se` is not finite and positive, or `p` is null, NaN or
  outside [0, 1];
- the per-row diagnostics contain NaN or infinity (the JSON is written with
  `allow_nan=False`);
- `fitness_wt` lacks `dna_to` and `se`, has duplicate endpoint rows, or gives
  an endpoint a null, non-finite or negative standard error.

barseqfit pair-fit `status` vocabulary: `converged`, `not_converged`,
`failed`. Only `converged` is accepted. (`reference` belongs to `fit_wt`.)

## BH family and qualification

The family is `candidate_split_tree_edges`: exactly the tree edges, with size
equal to the edge count. `q_value` comes from `compute_bh_qvalues`. An edge
qualifies when `q_value > alpha` (strict), meaning the contrast is not
significant at `alpha`. `alpha` must be finite and in [0, 1].

## Discrepancy record columns

The edge table adds to the pair columns: `from`, `to`,
`fitness_standard_error_upper`, `fitness_standard_error_lower` (from
`fitness_wt`; null for the reference endpoint), `contrast_estimate`,
`contrast_standard_error`, `p_value`, `q_value`, `qualifies`, `contrast_gap`
(`contrast_estimate - raw_fitness_delta`), `sign_disagreement`, `status`,
`elapsed_time`, `time_unit` and `diagnostics` (per-row JSON string).

## Sign-disagreement convention

`sign_disagreement = sign(m) != sign(raw_fitness_delta)`, evaluated only when
`m != 0` and `raw_fitness_delta != 0`. A zero on either side is no evidence of
disagreement and is not flagged.

## Artifacts and tables

The checkpoint in `output/analysis/candidate_tree_contrast/` holds
`pairs.{parquet,csv}`, `edges.{parquet,csv}`, `diagnostics.json` and
`provenance.json`, published atomically with a pending marker
`output/analysis/.candidate-tree-contrast.pending`. Cached reads refuse while
the marker exists. `just candidate-tree-contrast` also exports
`output/tables/papkou2023_candidate_tree_contrast_pairs.*` and
`papkou2023_candidate_tree_contrast_edges.*` with a provenance sidecar. Diagnostics
record node and edge counts, family size, alpha, qualifying and terminal
qualifying counts, sign-disagreement edge ids, the accepted status, and the
barseqfit version and revision.

## Provenance fingerprint

The fingerprint contains the checkpoint schema version, the
`[candidate_tree_contrast]` settings, the candidate fitness cutoff, and SHA-256
hashes of: the `fitness_wt` and `fitness_diff` inputs, `candidate_tree_contrast.py`,
`clamped_landscapes.py`, `barseq_inputs.py`, `uv.lock`, and the export's
`counts`, `samples` and `sequences` Parquet files. A changed fingerprint
rebuilds the checkpoint; a corrupted artifact is quarantined and rebuilt.

## Real-data run

Run on 2026-10-15 against the barseqfit export in `.scratch/papkou2023/`
(a barseqfit 0.1.0 development revision, before `barseqfit v0.2.0`, `elapsed_time` 3.5166 generations). Every
fit returned status `converged`.

| Quantity | Value |
| :--- | :--- |
| Tree nodes | 329 |
| Tree edges (tests, BH family size) | 328 |
| Terminal edges | 169 |
| Qualifying edges (`q > 0.05`) | 216 |
| Terminal qualifying edges | 70 of 169 |
| Interior qualifying edges | 146 of 159 |
| Sign disagreements | 0 |
| Runtime (`just candidate-tree-contrast`, cold cache) | 32 s |

Observations:

- `contrast_gap` is at most 8e-9 in absolute value on every edge, so the
  direct pair estimate `m` equals the raw wild-type-relative fitness
  difference up to floating point. The discrepancy record therefore carries
  no information in its estimate column; the pair fit contributes only the
  contrast standard error, p-value and q-value. Sign disagreement cannot
  occur under this fit.
- Interior saddle-saddle edges are mostly non-significant (146 of 159), so
  milestone two (interior contraction) will collapse most of the tree's
  interior at `alpha = 0.05`. Milestone one only acts on the 70 qualifying
  terminal edges.

To run it, set `PAPKOU2023_BARSEQ_EXPORT` in `.env` and run:

```
just candidate-tree-contrast
```

## Contrast-smoothed view (retired)

Retired on 2026-10-21 (ADR 0006 is superseded by ADR 0008). The `candidate-contrast-smoothed-tree` stage, its module `analysis/candidate_contrast_smoothed_tree.py`, `just candidate-contrast-smoothed-tree`, its tables and `summary.json` no longer exist; recover them from git history (the parent of the removal commit). The tree-edge contrast stage above stays for now; the bootstrap stage no longer reads it (singleton check removed, ADR `2026-10-22-pooled-standard-error-for-bootstrap`), and its removal is deferred. The rest of this section records the retired design and its real-data run.

The smoothed view applied the checkpoint's q-values to the Candidate split tree (ADR 0006, bucket C). It absorbed terminal branches only.

Rules:

- A terminal edge with strict `q > alpha` (the checkpoint's `qualifies` column) is absorbed, unless it is the elder arm at its saddle (next rule). When absorbed, its peak, and every regular vertex on its branch, join the lower branch of the adjacent saddle.
- Elder rule: at each saddle the incoming arm whose subtree holds the highest maximum (the maximum scalar over every node upstream through that arm; a terminal arm's value is its peak) is never absorbed, so a saddle never loses all its arms to absorption and the global maximum is never absorbed. Exact ties on the subtree maximum go to the smaller `vertex_index` of the node holding it, matching Topographer's persistence pairing tie order (changed from the smaller `dna_sequence` for the bootstrap smoothing design; it alters results only where exact float64 ties exist). The rule is computed on the rebuilt tree graph by `enforce_elder_rule` in `selected_leaf_simplification.py`; `absorb_terminal_branches` itself stays a general mechanism.
- The saddle keeps its measured fitness and the higher sibling keeps the maximum. Raw fitness stays the tree scalar; contrast estimates are never substituted for node heights.
- A saddle left with one incoming arm is contracted; a saddle left with none is absorbed down the trunk, as in `absorb_singleton_leaves`. Persistence is recomputed on the surviving tree.
- Regular vertices of an absorbed branch are assigned `location = "absorbed_branch"`; absorbed peaks keep `absorbed_peak = True`. The cascade into the root belongs to the general `absorb_terminal_branches` mechanism and cannot fire in this stage, because every saddle keeps its elder arm; if it did, `location` becomes `absorbed_root`; the boolean `absorbed_branch_member` stays True for every non-peak row that came from an absorbed branch.
- A terminal edge whose lower endpoint is the root (in-degree at least 2) is absorbed into the root group (`location = "absorbed_root"`).
- When no edge qualifies, the view is built with an empty nomination set and is the unabsorbed tree.

Code:

- `analysis/selected_leaf_simplification.py`: `absorb_terminal_branches(tree_analysis, peak_ids)`; `absorb_singleton_leaves` is a thin wrapper that keeps the singleton check.
- `analysis/candidate_contrast_smoothed_tree.py`: `build_contrast_smoothed_view(tree, edges, alpha)`, `summarise_contrast_smoothed_view(...)` and `compute_candidate_contrast_smoothed_tree(settings=None)`. The build fails closed on: `qualifies` differing from `q_value > alpha` (a stale checkpoint), null or non-finite `q_value` or a `q_value` outside [0, 1], null `is_terminal`, and any edge-identity difference. The identity check is two-way: every rebuilt-tree edge must be in the checkpoint and every checkpoint edge must be in the tree, and `upper_fitness`, `lower_fitness`, `is_terminal` and `regular_count` must agree row by row with `enumerate_tree_edges` of the rebuilt tree. Nominations take `is_terminal` from the rebuilt tree. `compute_candidate_contrast_smoothed_tree` also requires the checkpoint's `diagnostics["alpha"]` to equal the configured alpha and its provenance fingerprint (everything except the `barseq_export/*` hashes) to match a freshly computed one; otherwise it raises "stale checkpoint". The Candidate tree is rebuilt with the public `build_candidate_tree(settings)`.
- `just candidate-contrast-smoothed-tree` runs `dubanevics2026topological.build candidate-contrast-smoothed-tree`.

Outputs:

- Tables `papkou2023_candidate_contrast_smoothed_assignments` and `papkou2023_candidate_contrast_smoothed_persistence` (CSV and Parquet) in `output/tables/`.
- `output/analysis/candidate_contrast_smoothed_tree/summary.json` with `input_node_count`, `input_edge_count`, `test_count`, `alpha`, `absorbed_terminal_branch_count` (qualifying terminal edges minus elder-protected ones), `elder_protected_terminal_edge_count`, `elder_protected_edge_ids` (sorted), `absorbed_regular_vertex_count`, `removed_saddle_count`, `smoothed_node_count`, `smoothed_edge_count`, `surviving_peak_count` and `sequence_count`. `sequence_count` must equal the Candidate population, and `surviving_peak_count` must equal the input peaks minus the absorbed branches (elder-protected edges are not absorbed); otherwise the stage raises. `absorbed_regular_vertex_count` is counted from the assignments' `absorbed_branch_member` column and must equal the sum of `regular_count` over the absorbed edges. The summary also carries a `fingerprint`: `candidate_functional_threshold`, `alpha`, SHA-256 hashes of `fitness_wt`, `fitness_diff`, `selected_leaf_simplification.py` and `candidate_contrast_smoothed_tree.py`, and the contrast checkpoint provenance. `summary.json` is written last, after the tables and sidecar.
- Sidecar `output/tables/.candidate-contrast-smoothed-provenance.json` that is the produce-or-load checkpoint record (stage fingerprint, artefact hashes and the declared `contrast_checkpoint`, `alpha` and `fingerprint`); the authoritative byte-identical copy lives in `output/checkpoints/candidate_contrast_smoothed_tables/`.

Real-data run on 2026-10-15 at `alpha = 0.05`, against the same contrast checkpoint as the tree-edge contrast counts above.

| Quantity | Value |
| :--- | :--- |
| Input nodes | 329 |
| Input edges (tests) | 328 |
| Qualifying edges (`q > 0.05`) | 216 |
| Qualifying terminal edges | 70 |
| Elder-protected terminal edges | 9 |
| Absorbed terminal branches | 61 |
| Absorbed regular vertices | 11 |
| Removed saddles | 53 |
| Smoothed nodes | 215 |
| Smoothed edges | 214 |
| Surviving peaks | 108 of 169 input peaks |
| Sequence count | 104665 (equals the Candidate population) |

The smoothed node count follows from the input node count: 329 - 61 - 53 = 215. The 9 elder-protected `edge_id` values are `ACCGACACG|ACCGATACG`, `ACCGACGTG|ACCGACGTA`, `ACCGATGTC|ACCGATGTA`, `CAAGAAATG|CACGAAATG`, `CGCGAATTC|CGTGAATTC`, `CGTGATAGC|CGTGATACC`, `CTGGAACTA|CTGGAGCTA`, `GTCGACCAA|GTAGACCAA`, `TCCTGCCTT|TCCTGTCTC`.

## Figures (removed)

The two figures that rendered the contrast-smoothed view, `papkou2023_candidate_tree_contrast_biology` and `papkou2023_codon_entropy_contrast`, were removed from the repository on 2026-10-21 together with their `[figures.*]` tables, recipes, tests and PNGs, and the contrast-smoothed stage itself, `build_cached_contrast_view`, its tables and its glossary entry went in a later removal on the same day (see above). `papkou2023_candidate_tree_biology`, which this section compares against, and `papkou2023_codon_entropy` were retired in that later removal too. Recover the code from git history if it is needed again. The results the figures showed are recorded here.

- Candidate split tree with persistence, terminal branches by tree-edge contrast outcome at strict `q > alpha`: absorbed 61, elder-protected 9, kept 99 at `alpha = 0.05`. The 169 terminal branches are the 169 peaks, 15 of them stop-codon peaks.
- Contrast-smoothed tree: 108 surviving peaks, 53 saddles removed and 104,665 sequences retained. Persistence is recomputed on the smoothed tree.
- The identity-selected absorption of `papkou2023_candidate_tree_biology` (`absorb_singleton_leaves` applied to singleton peaks whose q-value is above the `candidate_peak_biology` cutoff) leaves 111 surviving peaks, against 108 here. The two operators remove different peaks, so the counts are not expected to match.
- Codon entropy on the contrast-smoothed view: the KEM and TCL ridge populations (12,526 and 749 functional sequences) and the Candidate population (104,665) contain exactly the same `dna_sequence` sets under both views, and the claim summary is byte-identical: ridge assignment 13,275 / 13,933 = 95.28 %, KEM position-2 E 44.85 % and D 54.94 %, TCL 0 E and 0 D. No ridge assignment changed. The contrast smoothing removes different peaks (61 terminal branches) from the identity rule's singleton set, but both ridges' summits and their common saddle survive both operators.

The view is built from the in-memory tree returned by `build_candidate_tree`, not from the float32 GraphML comparison checkpoint. The checkpoint stores fitness in float32, which does not reproduce the float64 fitness values that the edge identities and the contrast checkpoint were built from, so the edge-identity check fails closed on it (commit 86b33da).
