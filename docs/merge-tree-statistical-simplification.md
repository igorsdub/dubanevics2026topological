# Statistical simplification of merge trees under node-height uncertainty

Status: Research note for issue #50 (2026-10-15). Read-only survey of primary
sources; no implementation decisions are taken here.

## Question

When node heights carry measurement uncertainty (here: Papkou2023 BarSeq
fitness = growth-rate difference versus wild type, with a standard error), how
do published methods decide which merge/split/contour-tree branches to
collapse, where the merged node sits afterwards, and whether anyone reconciles
a per-node scalar with a directly fitted pairwise contrast for the same pair?
Also: what simplification API does Topographer already expose?

## Summary

- No published merge-tree or contour-tree paper found here replaces the
  persistence threshold with a per-edge hypothesis test. Topology-visualisation
  simplification (Carr et al. 2004; Morozov et al. 2013; TTK) is deterministic:
  a scalar priority (persistence, hypervolume) and a fixed threshold.
- Statistical persistence (Fasy et al. 2014; Chazal et al. 2013) does supply a
  significance rule, but it is global: one data-driven half-width `c_n` from
  subsampling, concentration, or bootstrap; a diagram point is "noise" when its
  `2c_n` box touches the diagonal. There is no per-pair test and no FDR step.
- Per-split significance exists in clustering: pvclust attaches a multiscale
  bootstrap AU p-value to every dendrogram edge (Suzuki & Shimodaira 2006);
  SigClust/SHC test each dendrogram split with a Gaussian null and sequential
  FWER control (Liu et al. 2008; Kimes et al. 2017). None of them re-height or
  contract the tree; they label and select clusters. Felsenstein (1985) bootstrap
  consensus drops unsupported clades into polytomies (a contraction) without
  changing branch lengths.
- Papkou et al. 2023 (supplement) is the closest primary precedent for issue
  #50: a joint binomial GLM with a `type*time` interaction gives `r_ij` with an
  SE and a Wald p-value for every pair at Hamming distance 1 or 2, BH-adjusted
  over 45.7 M tests. Peaks, however, were called on point estimates only
  ("higher fitness than all its neighbors"); the tests were reported as power
  (Fig. S17, 0.933 at a 2.5% difference), not used to prune peaks.
- Where the merged node sits: Carr et al. prune a leaf by "collapsing the edge
  to its upper vertex" (the saddle survives, the extremum is levelled to the
  saddle height); Morozov-style epsilon-simplification yields the merge tree of
  a nearby function, again keeping the saddle. Topographer's existing
  `smooth_merge_tree` instead copies the payload of the *minimum-scalar* member
  and records `scalar_min`/`scalar_max`; issue #50's "maximum original member
  fitness" rule is a third convention and must be set explicitly.
- Nothing found records discrepancies between a per-node scalar (fitness vs WT)
  and a direct pairwise contrast for the same pair (sign, magnitude, SE). Papkou
  et al. computed both quantities but report no reconciliation. Song & Zhang
  (2021) and Baheti et al. (2025) address noise-inflated peak counts, not
  contrast consistency.
- Topographer (oist/topographer, v0.3.0) already has
  `topographer.topology._simplification.smooth_merge_tree(tree, *, predicate=None,
  edge_attr=None, threshold=None, comparator="le") -> TreeSmoothingResult`, a
  single-pass union-find edge contraction that accepts an arbitrary edge
  predicate, plus `prune_leaf_branches`, `prune_leaf_branches_by_ordinary_persistence`,
  `MergeTree.simplify(persistence_threshold)` and `statistics.compute_bh_qvalues`.
  None of these is exported from `topographer.topology.__init__`.

## 1. Per-edge tests versus a fixed persistence threshold

### Statistical persistence and confidence sets

Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan and Singh (2014) construct a
confidence set `C_n = {P~ : W_inf(P^, P~) <= c_n}` for the persistence diagram
using the bottleneck distance and bottleneck stability (their Theorem 2).
Visualisation rule (Section 4): "The point p is considered indistinguishable
from noise if the corresponding box ... intersects the diagonal", equivalently a
band of width `sqrt(2) c_n` around the diagonal; "points in the band are not
significantly different from noise." Four ways to obtain `c_n` are given:
Method I subsampling (conservative), Method II concentration of measure,
Method III shells, Method IV density estimation. The remark after the rule
concedes that "some points near the diagonal may represent interesting
structure" and that per-point confidence sets of different sizes are possible
but not developed. Chazal, Fasy, Lecci, Rinaldo, Singh and Wasserman (2013)
replace `c_n` by an empirical-bootstrap quantile for diagrams and landscapes;
Chazal et al. (2015) use subsampling for computational relief. All of these
are one global band, not per-edge tests, and the noise model is sampling of a
point cloud, not per-vertex measurement error on a fixed graph.

### Merge-tree stability and distances

Morozov, Beketayev and Weber (2013) prove (Theorem 2) that the interleaving
distance between merge trees satisfies `d_I(T_f, T_g) <= sup_x |f(x) - g(x)|`,
and (Theorem 3) that it is never smaller than the bottleneck distance of the
0-dimensional diagrams. Their framing of simplification is the sup-norm one:
if the input is perturbed by at most epsilon, an epsilon-simplified tree is the
exact merge tree of some nearby function, so simplification "sacrifices no
precision". This justifies a threshold equal to a worst-case noise bound, not a
test; with heteroscedastic SEs it would mean `epsilon = max SE * z`, far more
conservative than a per-edge Wald test. Sridharamurthy, Masood, Kamakshidasan
and Natarajan (2020) define a metric tree edit distance for merge trees for
comparison, not simplification (unverified beyond the abstract whether the
paper discusses stability under noise).

### Noise-aware contour-tree simplification

Carr, Snoeyink and van de Panne (2004) simplify with two operations, "leaf
pruning and vertex reduction", driven by a priority queue of leaf costs
("Always choose the least important leaf to prune"; "Never prune the last up-
or down- leaf at an interior vertex"). The cost is a local geometric measure
(persistence, line length, area, volume, hypervolume). Noise removal is by
threshold on that measure; no statistical model of the scalar field enters.

### Significance of clusters in dendrograms

- pvclust (Suzuki & Shimodaira 2006; CRAN reference): "p-values are computed
  for each of the clusters" by multiscale bootstrap; each edge carries BP, AU
  and (since 2.2-0) SI values; `pvpick(x, alpha=0.95, pv="au", type="geq",
  max.only=TRUE)` returns the clusters whose p-value meets the threshold. The AU
  value "is valid when you know the clusters before looking at the data"; SI
  (Terada & Shimodaira 2017) is for clusters chosen by looking at the tree.
  The dendrogram object is returned unchanged; heights are not altered.
- SigClust (Liu, Hayes, Nobel and Marron 2008, JASA) tests a single two-way
  split against a single-Gaussian null with the 2-means cluster index. SHC
  (Kimes, Liu, Hayes and Marron 2017, Biometrics 73:811-821) applies it "from
  the root node" down every dendrogram split with a sequential stopping rule
  that controls FWER; again a label per node, no re-heighting.
- Felsenstein (1985): majority-rule consensus over bootstrap trees; "If a group
  shows up 95% of the time or more, the evidence for it is taken to be
  statistically significant." Unsupported groups are absent from the consensus,
  i.e. collapsed into a polytomy.

Selection effect worth noting for issue #50: pvclust's own documentation
distinguishes AU (clusters fixed a priori) from SI (clusters read off the
tree). Tree-adjacent pairs in a split tree are chosen by the observed heights,
so the issue's "nominal adjustment" caveat is the right posture.

### Fitness-landscape peak significance

- Papkou et al. 2023 (Science 382:eadh3860; supplement): fitness is the growth
  rate difference `r_variant - r_ref` (selection coefficient scale, WT = 0),
  estimated from a binomial GLM on read counts. For pairs: "Y = (b0 + g*b2) +
  (b1 + g*b3) t + e", where "the coefficient b3 estimates the fitness r_ij of
  variant i relative to variant j", with "a standard error for the b3 term" and
  "a statistical test of the null hypothesis that b3 = 0 (e.g., the Wald
  test)". Fitted with `glm(family=binomial(link=logit))`; BH over 261,331 WT
  tests and separately over 45,730,826 tests for all pairs within two
  substitutions. Peak definition: "a variant (node) which has a higher fitness
  than all its neighbors", no test. Main text: "we had limited power to detect
  fitness differences smaller than 2.5%", which "existed between 16% of all
  mutational neighbors". So issue #50's contrast model is Papkou's own, scoped
  down to tree edges.
- Song & Zhang (2021, Evolution 75:2658-2671): estimation error flips neighbour
  rank order and inflates the local-maximum count `Nmax` most strongly among
  eight ruggedness measures. Their fix is extrapolation of ruggedness against
  replicate-averaged error to zero error, not per-peak testing; they argue
  against significance thresholds because "natural selection can detect a fitness
  differential of 1/Ne", and recommend at least three biological replicates.
- Baheti, Raj and Saini (2024/2025, bioRxiv 10.1101/2024.08.25.609583; eLife
  reviewed preprint 104848): re-analysing Papkou's landscape "using average
  fitness values across replicates together with their standard deviations",
  "Of the 514 peaks reported by Papkou et al., only 127 remain when error is
  considered", all in the high-fitness region. The exact error-aware peak rule
  is not stated in the text retrieved (unverified); eLife reviewers flagged
  statistical rigour.
- de Visser & Krug (2014, Nat Rev Genet 15:480-490) and Szendro et al. (2013,
  J Stat Mech P01005): full texts not retrievable here; no peak-significance
  rule confirmed (unverified). Bank (2022, Annu Rev Ecol Evol Syst 53:457-479)
  abstract does not address measurement error (full text restricted).

## 2. Where the merged node sits after a collapse

| Method | Rule | Why |
| :--- | :--- | :--- |
| Carr et al. 2004 leaf prune | "collapsing the edge to its upper vertex" (lower leaf) or downwards (upper leaf); the saddle survives; "pruning a leaf corresponds to levelling off a maximum or minimum" | Keeps the contour tree a valid tree of a modified field |
| Morozov et al. 2013 (epsilon) | Simplified tree is the exact merge tree of some g with sup-norm distance <= epsilon; the leaf is raised/lowered to its saddle | Stability theorem makes the result interpretable |
| Persistence-pair pruning (Topographer `prune_leaf_branches*`) | Removes the leaf path up to the first degree >= 3 saddle; saddle keeps its height | Elder rule pairs each leaf with a saddle |
| pvclust / SHC / SigClust | No merge; nodes labelled, heights untouched | Inference about clusters, not a new tree |
| Felsenstein consensus | Unsupported clade collapsed into a polytomy; parent keeps its position | Consensus topology only |
| Topographer `smooth_merge_tree` | Union-find contraction; new node copies the payload (including `scalar`) of the minimum-`scalar` member and adds `scalar_min`, `scalar_max`, `member_*`, `event="mixed"` | Deterministic representative; range retained |
| Issue #50 plan | Contracted node takes maximum original member fitness; saddle keeps its height when a non-significant leaf is absorbed | Author's choice; no published precedent found for "maximum" |

Note the tension: issue #50 says both "saddle keeps its height" and "maximum
original member fitness as activation scalar". For a split tree a contracted
(leaf, saddle) pair has the leaf as the maximum, so these agree only if the
*saddle* is the node that survives structurally while the *scalar* reported is
the leaf's. Every visualisation method above instead reports the saddle height
after a prune. Whichever is chosen, record both (`scalar_min`, `scalar_max` as
Topographer already does) so the convention is auditable.

## 3. Recording per-node scalar vs pairwise contrast discrepancies

Nothing found. Papkou et al. compute `r_i` (vs WT) for every variant and
`r_ij` for every pair within two substitutions with the same GLM family, and
note that algebraically `r_ij = (r_i - r_bar) - (r_j - r_bar)`, but report no
comparison of `r_i - r_j` against the fitted `b3`, no sign-disagreement count,
and no SE comparison. Song & Zhang and Baheti et al. work from replicate means
and SDs, not pairwise fits. The merge-tree, persistence and dendrogram
literature has no analogue because node heights there are exact. This item is
genuinely open and would be a small methodological contribution of issue #50
if exported alongside the edge evidence table.

## 4. Topographer simplification API (a development revision of topographer preceding v0.5.0)

File `src/topographer/topology/_simplification.py` (`__all__` lists all four):

- `TreeSmoothingResult` (frozen dataclass): `tree: rx.PyDiGraph`, `stats: dict[str, Any]`.
- `smooth_merge_tree(tree, *, predicate: Callable[[Any], bool] | None = None,
  edge_attr: str | None = None, threshold: float | None = None,
  comparator: str = "le") -> TreeSmoothingResult`. Contracts every edge whose
  predicate is true in one union-find pass (transitive, no refit), exactly the
  "build components once from qualifying edges" step in issue #50. With
  `edge_attr="q_value", threshold=alpha, comparator="gt"` it implements the
  strict `q > alpha` rule once q-values are attached to edge payloads.
  Representative = member with minimum `(scalar, vertex_index, index)`.
- `prune_leaf_branches(tree, *, max_scalar_difference: float,
  scalar_attr: str = "scalar", root_kind: str = "minimum") -> TreeSmoothingResult`.
  Leaf-only; never contracts saddle-saddle edges; a path must end at a node of
  undirected degree >= 3; stats include `saddle_tree_node_index`,
  `scalar_difference`, `removed_node_count`, `pruned_node_count`.
- `prune_leaf_branches_by_ordinary_persistence(tree, *, max_persistence: float,
  tree_kind: str) -> TreeSmoothingResult`. Elder-rule H0 pairs via
  `topology._persistence.build_tree_ordinary_persistence_pair_table`; the elder
  root lineage is never removed.
- `MergeTree.simplify(self, persistence_threshold: float) -> MergeTree`
  (`_merge_tree.py`): cached wrapper over the ordinary-persistence pruner;
  refuses to simplify an already simplified tree.
- `topographer.statistics.compute_bh_qvalues(pvalues) -> np.ndarray` is
  exported and doctested.
- `topographer.statistics.propagate_branch_uncertainty(network, se_attr="se",
  tree_type="split")` and `bootstrap_persistence_uncertainty(network, se_attr,
  tree_type, n_replicates=1000, alpha=0.05, seed=None)` give SE-propagated and
  Gaussian-bootstrap persistence uncertainty per branch; these are the
  Fasy/Chazal-style global alternatives, not per-edge contrasts.

Caveats: the docstrings still reference `topographer.tree.simplification`
(stale after ADR-0014 relocation); the functions are not re-exported from
`topographer.topology`, so the companion layer must import from the private
module or request an export. No leaf-absorption function that keeps the saddle
height while assigning the leaf's maximum exists; that combination is new.

## Implications for issue #50

- The pairwise `type*time` binomial GLM with Wald p and BH is Papkou's own
  published machinery (supplement, "Calculating fitness"), so cite it as the
  model source and present the tree-edge family as a restriction of their
  45.7 M-pair family.
- No topology paper tests edges; the closest statistical precedents are pvclust
  and SHC, both of which label rather than contract. Felsenstein consensus is
  the precedent for contracting unsupported edges. State clearly that the
  smoothed tree is an exploratory quotient, as the issue already does.
- Resolve the "saddle keeps height" vs "maximum member fitness" wording before
  implementation and store `scalar_min`/`scalar_max` plus member identities on
  every contracted node; `smooth_merge_tree` already provides these fields but
  assigns the minimum-scalar payload, so the activation scalar must be
  overridden or a new option added.
- Export a per-edge reconciliation table (`fit_i - fit_j`, `b3`, sign
  agreement, `sqrt(se_i^2 + se_j^2)` vs `se_b3`); no published method does this
  and it is cheap given the edge evidence table.
- Prefer `smooth_merge_tree` with a q-value predicate over the leaf-only
  pruners if saddle-saddle contractions are intended; use `prune_leaf_branches`
  semantics if only leaf absorption is wanted. The issue text allows both
  internal and external edges, which matches `smooth_merge_tree`.
- Treat the BH family as nominal (tree-adjacent pairs are selected by the
  observed heights); pvclust's AU/SI distinction is the citable analogy.

## References

- Fasy BT, Lecci F, Rinaldo A, Wasserman L, Balakrishnan S, Singh A (2014).
  Confidence sets for persistence diagrams. Ann Stat 42(6):2301-2339.
  https://doi.org/10.1214/14-AOS1252 (arXiv:1303.7117)
- Chazal F, Fasy BT, Lecci F, Rinaldo A, Singh A, Wasserman L (2013). On the
  bootstrap for persistence diagrams and landscapes. arXiv:1311.0376.
- Chazal F, Fasy BT, Lecci F, Michel B, Rinaldo A, Wasserman L (2015).
  Subsampling methods for persistent homology. ICML 2015.
  https://proceedings.mlr.press/v37/chazal15.html
- Morozov D, Beketayev K, Weber G (2013). Interleaving distance between merge
  trees. TopoInVis 2013. https://mrzv.org/publications/interleaving-distance-merge-trees/
- Sridharamurthy R, Masood TB, Kamakshidasan A, Natarajan V (2020). Edit
  distance between merge trees. IEEE TVCG 26(3):1518-1531.
  https://doi.org/10.1109/TVCG.2018.2873612
- Carr H, Snoeyink J, van de Panne M (2004). Simplifying flexible isosurfaces
  using local geometric measures. IEEE Visualization 2004.
  https://www.cs.ubc.ca/~van/papers/2004-viz-contourtree.pdf
- Liu Y, Hayes DN, Nobel A, Marron JS (2008). Statistical significance of
  clustering for high-dimension, low-sample-size data. JASA 103(483):1281-1293.
- Kimes PK, Liu Y, Hayes DN, Marron JS (2017). Statistical significance for
  hierarchical clustering. Biometrics 73(3):811-821.
  https://doi.org/10.1111/biom.12647
- Suzuki R, Shimodaira H (2006). pvclust: an R package for assessing the
  uncertainty in hierarchical clustering. Bioinformatics 22(12):1540-1542;
  CRAN manual https://cran.r-project.org/web/packages/pvclust/refman/pvclust.html
- Felsenstein J (1985). Confidence limits on phylogenies: an approach using the
  bootstrap. Evolution 39(4):783-791.
- Papkou A, Garcia-Pastor L, Escudero JA, Wagner A (2023). A rugged yet easily
  navigable fitness landscape. Science 382:eadh3860.
  https://doi.org/10.1126/science.adh3860; preprint and supplement
  https://doi.org/10.1101/2023.02.27.530293
- Song S, Zhang J (2021). Unbiased inference of the fitness landscape
  ruggedness from imprecise fitness estimates. Evolution 75(11):2658-2671.
  https://doi.org/10.1111/evo.14363
- Baheti S, Raj N, Saini S (2024/2025). Fluidity and predictability of
  epistasis on an intragenic fitness landscape. bioRxiv
  https://doi.org/10.1101/2024.08.25.609583; eLife reviewed preprint 104848.
- de Visser JAGM, Krug J (2014). Empirical fitness landscapes and the
  predictability of evolution. Nat Rev Genet 15:480-490.
  https://doi.org/10.1038/nrg3744 (full text not retrieved)
- Szendro IG, Schenk MF, Franke J, Krug J, de Visser JAGM (2013). Quantitative
  analyses of empirical fitness landscapes. J Stat Mech P01005.
  https://doi.org/10.1088/1742-5468/2013/01/P01005 (full text not retrieved)
- Bank C (2022). Epistasis and adaptation on fitness landscapes. Annu Rev Ecol
  Evol Syst 53:457-479 (full text restricted)
- Topographer: https://github.com/oist/topographer,
  `src/topographer/topology/_simplification.py`,
  `src/topographer/topology/_merge_tree.py`,
  `src/topographer/statistics/_multiple_testing.py`,
  `src/topographer/statistics/_uncertainty.py`
