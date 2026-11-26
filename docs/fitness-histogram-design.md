# Fitness histograms and comparison scales

Status: revised contract for issue #42, extending the delivered issue #40 histogram APIs.

## Settled contract

Each Full, Published, or Candidate histogram counts every vertex in its selected graph once, including retained fringe and noncritical vertices. Stable DNA identities and raw fitness are preserved; tree simplification does not change the population. Histogram vertex counts and branch vertex counts are distinct.

Use 200 equal-width bins over each population's observed minimum–maximum, left closed and right open with the maximum included in the final bin. Topographer's reusable API default remains 100; publication callers explicitly request 200. Empty, null/nonfinite, and constant populations remain errors.

The tree-family fitness display domain is configured in figures.toml as [-1.2, 1.4], applied directly without rounding or padding. Editing it intentionally changes the visible region without changing scientific populations or binning. There are no cropping switches or count-limit override controls.

Standalone Full, Published, and Candidate compositions use a tree at left, and a 140 × 250 px histogram at far right. The branch-count legend is a horizontal colour bar inside the tree panel, not a separate column. Fitness increases vertically with one axis titled Fitness (WT = 0); touching gray horizontal bars show Vertex count from zero, without outlines, gridlines, overlays, or histogram legends. Retain the orange WT marker. Each standalone derives histogram and branch-count maxima from its own population; histogram count limits add 5% headroom and use sparse compact tick labels.

The Papkou tree comparison stacks those three tree–histogram rows. Histogram limits run from zero to 105% of the largest bin count across populations, with the horizontal count axis shown only below the bottom histogram. Each population retains its observed-range bins: equal counts across different bin widths do not imply equal density. Branch colors share one logarithmic plasma domain derived from the combined maximum, with the separate gray singleton swatch. Each tree carries its own horizontal colour bar inside the panel; all four compositions use 110 px bars.

## Implementation buckets

1. Extract reusable publication population/composition helpers and configure the fixed display domain. Keep existing Topographer APIs unchanged.
2. Register standalone Full/Published/Candidate thin notebooks and canonical SVG/300 dpi PNG exports, reusing scientific caches and provenance.
3. Rename the comparison identifier to tree_comparison, add histograms with shared count limits, and update callers and documentation. Scientific study_clamped/candidate_clamped cache identifiers remain unchanged.
4. Use light implementation agents and independent integration review; verify counts, bin independence, scales, clipping, exports, and manuscript-size appearance. Complete PR/merge/cleanup and issue closure after validation.

## Superseded behavior

Issue #40 introduced Candidate-only 100-bin publication histograms and corrected tree comparison colors without adding comparison histograms. Issue #42 expands the figure family, uses 200 publication bins, replaces automatic fitness padding with configured display limits for this family, and adds comparison histograms. Other figure families retain their existing scalar conventions. Density views, violin plots, equal-count bins, and stop-codon overlays remain deferred.
