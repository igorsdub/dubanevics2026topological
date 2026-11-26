# Topographer + FDR split tree and its bootstrap-band tree

| Statistic | Split tree | Bootstrapped tree |
| --- | --- | --- |
| number of peaks; number of saddles | 135; 126 | 29; 28 |
| branch vertex count: min / median / mean / max | 1 / 3 / 375.977011494 / 14025 | 1 / 143 / 1721.57894737 / 14025 |
| amino-acid Hamming distance along a branch: mean / median | 1.17624521073 / 1 | 1.87719298246 / 2 |
| nucleotide Hamming distance along a branch: mean / median | 2.75478927203 / 1 | 4.01754385965 / 5 |
| Horton-Strahler order 1 / 2 / 3 / 4 branch counts | 135 / 39 / 8 / 79 | 29 / 28 / 0 / 0 |
| branch persistence: median / max | 0.0309480958089 / 2.51642274247 | 0.702358415265 / 2.51642274247 |
| stop-codon genotypes on peaks | 11 | 5 |

# Topographer + FDR split tree and its bootstrap-band tree

Manuscript Table 2. Variant `threshold=reflected_fdr=by_q=0.05`: the Topographer + FDR graph selected by Benjamini–Yekutieli at nominal q = 0.05 with the median-reflected cutoff. Column `split_tree` is that graph's nominal split tree. Column `band_simplified_tree` is the same tree after direct band simplification.

Band settings are those of the restored bootstrap checkpoint (`summary.json` `settings`), read by `load_cached_bootstrap_smoothed_tree`. This table does not draw. The values used are draws (n_replicates) = 10000, seed = 20261019, band quantile = 0.95. `config/analysis.toml` `[bootstrap_smoothed_tree]` carries the same triple when that file is unchanged. This table is manuscript Table 2.

A branch is one edge of the tree after `absorb_terminal_branches`, as `analysis.branch_decomposition` defines it.

- **Branch vertex count.** `branch_vertex_count` from the branch decomposition: the upper endpoint, the interior vertices on that edge, and the sequences of hills absorbed onto that edge. The lower endpoint belongs to the edge below and is excluded. The minimum is 1. The row reports the minimum, median, mean and maximum of that count over branches.
- **Amino-acid Hamming distance along a branch.** `hamming_distance` from `sequence_distances` between the amino-acid sequences of the branch's two critical endpoints (upper and lower). It counts mismatched positions. Both sequences must have the same length. It is that endpoint mismatch count. It is not a sum of distances along the path of member sequences, and it is not the maximum pairwise distance among the branch's members. The row reports the mean and the median over branches.
- **Nucleotide Hamming distance along a branch.** The same `hamming_distance` between the DNA sequences of those same two critical endpoints. The same endpoint definition applies: mismatch count between the two endpoints, not path length and not maximum pairwise distance among members. The row reports the mean and the median over branches.
- **Horton–Strahler order.** `horton_strahler_order` from Topographer `build_tree_metrics` on the smoothed tree, the column the branch decomposition stores. A terminal branch has order 1. The row reports how many branches have order 1, 2, 3 and 4. An order with no branch is 0.
- **Branch persistence.** Ordinary H0 persistence of each pair on that tree (`persistence` on the pair table produced with the simplified tree): the absolute fitness difference between the peak's birth and its death saddle. The global maximum's pair is closed at the global minimum, so every pair is finite. The row reports the median and the maximum over pairs. This is not branch span (`birth_fitness - death_fitness` of one edge).
- **Peaks and saddles.** Peaks are the terminal branches (split-tree maxima). Saddles are the other branches: each saddle is the upper endpoint of the single edge below it. The root is not a peak and not a saddle. The cell is `peaks; saddles`.
- **Stop-codon genotypes on peaks.** The number of peak genotypes whose amino-acid sequence contains a stop symbol, using Topographer `has_stop_codon` (`topographer.biology.has_stop_codon`). The symbols are `*`, `%`, `^` and `&`, the same set as `STOP_CODON_SYMBOLS` in that module and in `analysis.biological_annotations`. Interior sequences on a branch do not add to this count.
