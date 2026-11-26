# Peak-pair distance analysis

Status: implemented (stage and figures; see `docs/figure-inventory.md`); glossary terms still proposed. Descriptive analysis; no ADR (reversible, captured here and in `GLOSSARY.md`).

## Question

How far apart are pairs of peaks, in sequence (Hamming distance) and in fitness, measured directly and measured via their common saddle on the Band-simplified tree? Do sequence distance and fitness distance move together across pairs?

## Scope

- The object of inquiry is the Band-simplified tree of the Median-reflected Benjamini–Yekutieli (BY) graph only: the 29 surviving peaks. A 514-peak extension is later work.
- No graph walks. A walked geodesic and a straight-route barrier are out of scope (discussion or future work). "Geodesic distance" stays reserved for walked paths above the saddle, which are not computed.
- Raw fitness throughout (no transforms).

## Quantities per peak pair

Keyed by the peak DNA identities and the common-saddle DNA identity; never by Rustworkx indices. Nucleotide is primary; amino-acid values sit alongside.

- Common saddle: the pair's lowest common ancestor in the Band-simplified tree, which is itself a split tree (see GLOSSARY, Common saddle). The saddle DNA is recorded per node (`critical_nodes.parquet`, `tree_node_identity`); the per-pair table is derived by lowest-common-ancestor lookup.
- Direct Hamming distance, via-saddle Hamming distance, direct fitness difference, via-saddle fitness distance (GLOSSARY, proposed entries).
- Differences: via-saddle minus direct, for Hamming and for fitness.

## Figures

### `peak_pair_matrices`

Three heatmaps side by side; rows and columns follow the leaf order of the Band-simplified tree (the left-to-right order of the peaks in the drawn tree, `subtree_extremum` layout); diagonal 0.

1. Hamming: lower triangle direct, upper triangle via saddle.
2. Fitness: same layout.
3. Difference: lower triangle Hamming (via minus direct), upper triangle fitness (via minus direct).

Each of heatmaps 1 and 2 shares one scale across its two triangles (discrete palette for Hamming, continuous for fitness). Heatmap 3 carries two colour bars with distinct palettes. Fitness colour bars are ticked at whole fitness units only, without decimals. Later: the split tree drawn to the left of the heatmaps.

### `peak_pair_direct_via_saddle_scatter`

One row of three square panels: nucleotide Hamming distance, amino-acid Hamming distance and fitness distance. In each, x is the direct distance and y the distance via the saddle, with a dashed `y = x` baseline. Hamming distances are integers, so each occupied (direct, via) cell is one fixed-size circle shaded in grey on a logarithmic scale of the pair count (legend ticks 1, 10, 100, those up to the largest cell count). Fitness is continuous, so each pair is a filled, semi-transparent point. Axes are padded by 5%, carry identical ticks on x and y, and have no grid.

Each panel is descriptive only. Its subtitle gives "Pairs on the diagonal: n of N (p%)" and "Mean / median excess" (via saddle minus direct). No Spearman ρ or p-value is reported: via ≥ direct by construction, so independence is not a meaningful null; the figure reports how far pairs sit above the diagonal instead. For fitness the excess is 2 × (lower peak − saddle). The fitness excess is strictly positive because every Band-simplified peak sits strictly above its common saddle, so the fitness diagonal fraction is 0.

### `peak_pair_hamming_fitness_scatter`

Two rows by three columns, one filled point per pair. Row 1 is nucleotide Hamming distance (x), row 2 amino-acid Hamming distance (x); y is the matching fitness quantity (Direct fitness difference, Via-saddle fitness distance, or their difference). The columns are direct, via saddle and via saddle minus direct, and each column shares its y range across both rows; the direct and via-saddle columns' y ranges start at 0 and the difference column starts at its data minimum (before padding). Integer Hamming axes carry integer ticks and are jittered horizontally (seeded, display only; the statistics use the exact values); x axes of the Hamming panels are padded by 5% beyond the jitter, other axes by 5%, and there is no grid. Each panel's subtitle reads "Spearman ρ = r (Mantel p = p)", where ρ is Spearman's rank correlation (the Pearson correlation of the ranks) and p the Mantel permutation p-value; p < 0.001 prints as "Mantel p < 0.001" and non-finite values print "Spearman ρ and Mantel p undefined".

These statistics relate two different quantities, sequence distance and fitness distance, which is why this figure carries them and the direct-against-via figure does not. ρ describes the association; the Mantel p tests it only against random relabelling of the peaks; no model is fitted.

### Labels

Label mode `amino_acid` (default), `dna` or `both`. No suffixes; synonymous collisions are shown as they are. Stop codons use the existing per-codon characters `%`, `^`, `&`.

## Mantel test

For the direct, via-saddle and difference matrices, for each of nucleotide and amino-acid Hamming distance (six tests), a Mantel test of Spearman's ρ between the Hamming matrix and the fitness matrix: permute the peak labels of one matrix, recompute ρ, 9999 permutations, seeded; the p-value is two-sided, p = (1 + number of permutations with |ρ_perm| ≥ |ρ_obs|, counted with a tolerance of 1e-12) / (1 + 9999), where ρ_obs is the observed correlation and ρ_perm the correlation after one permutation. Raw fitness. All six Mantel tests share one seed, so they reuse the same permutations and their p-values are dependent; each is valid on its own, but do not combine them (no Fisher or Bonferroni pooling across the six).

## Settings

- `config/analysis.toml`, table `[peak_pair_distances]`: tree = band, Mantel permutations 9999, seed.
- `config/figures.toml`, tables `[figures.peak_pair_matrices]`, `[figures.peak_pair_hamming_fitness_scatter]` and `[figures.peak_pair_direct_via_saddle_scatter]`: label mode, palettes, jitter, marker sizes and count greys.

## Work order

1. Toy fixture with hand-checked matrices.
2. 29-peak pair table.
3. Figures.
4. 514-peak extension, later.

First renders for this work are local, by explicit user request (an exception to the cluster-only rule for renders).

## Literature

- Becker and Karplus 1997, doi:10.1063/1.473299.
- Flamm et al. 2002, doi:10.1524/zpch.2002.216.2.155.
- Kucharík et al. 2014, doi:10.1093/bioinformatics/btu156 (names path length and energy as open).
- Condon et al. 2017, doi:10.1007/s11047-016-9587-9.
- Morgan and Higgs 1998: unverified (full text not read).
