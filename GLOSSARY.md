# Glossary

**Publication analysis repository**: The data preparation, scientific analyses, figures, and tables supporting the manuscript "Topological characterization of combinatorial fitness landscapes using merge trees". Manuscript text and captions are maintained separately.

**Manuscript map**: `config/manuscript.toml`, the only place manuscript figure and table numbers live. Code and fingerprints keep the stable figure id.

**Namer**: The copy step (`just name-outputs`) that places shelf renders from `output/figures/canonical/` into numbered `main/` and `supplementary/` names and writes `manifest.csv`. It does not render.

**Papkou2023 analysis**: The initial biological application, using the full, unfiltered Papkou2023 dataset.

**Full graph**: The graph containing every sequence in the dataset.

**Papkou graph**: The subgraph that the source study constructed and analysed, selected by the Papkou threshold.
_Avoid_: Published graph

**Topographer graph**: The subgraph selected by the Topographer threshold.
_Avoid_: Median-reflected graph, Candidate graph (historical; it named four different selections)

**Papkou + FDR graph**: The Papkou graph with the Benjamini-Yekutieli false discovery rate selection at nominal q = 0.05 with pooled standard errors; the **Topographer + FDR graph** is the same selection applied to the Topographer threshold. In a figure, "Papkou + FDR, bootstrapped" and "Topographer + FDR, bootstrapped" name the same graph with the bootstrapped persistence threshold applied to its tree.
_Avoid_: Published BY graph, Median-reflected BY graph, Candidate graph

**Display names and state names**: Figures and tables show the display names (Full, Papkou, Papkou + FDR, Papkou + FDR, bootstrapped, Topographer, Topographer + FDR, Topographer + FDR, bootstrapped). The unchanged internal state names in code, configuration and file paths are `full`, `published`, `published_by`, `published_by_band`, `reflected`, `reflected_by` and `reflected_by_band`, in that order.

**Selected graph**: A graph named by its functional threshold and its classification: the Papkou graph (`published`, point estimate), the Topographer graph (`reflected`, point estimate), the Mixture graph (mixture quantile, mixture probability; historical/alternative), the Papkou + FDR graph and the Topographer + FDR graph (fixed threshold, BY at nominal q = 0.05 with pooled standard errors). Its label fields are `thr=<published|reflected|mixture>` and `fdr=<none|by>`, with `q=<level>` after `fdr=by` and `p=<level>` for the Mixture graph; the Full graph is `graph=full`.

**Persistence diagram**: A plot of each topological feature as a point at its birth and death values, so that distance from the diagonal shows its persistence.

**Threshold scan**: An ordered progression of threshold values used to inspect which vertices, edges, or connected components are present. Topographer calls the scalar-threshold operation that constructs a merge tree a sweep; this publication workflow prefers scan as the plain-language term (to be aligned with the Scan entry in the scan-rename change).

**Scan**: A run of an analysis over a grid of settings, reported as a sensitivity description and not used to choose a setting. The project calls it a scan, never a sweep (ADR 2026-11-19-hpc-and-scan-naming). Its four named sensitivity scans are `fdr_q_scan`, `functional_classification_scan`, `bootstrap_band_quantile_scan` and `gaussian_mixture_quantile_scan`. The Threshold scan above is a different, lower-level idea: the ordered thresholds of one merge tree. The Sensitivity scan below is a scan in this sense.

**Split tree**: The merge tree for a decreasing superlevel-set scan of a scalar function.

**Persistence**: The lifetime of a topological feature as a threshold changes, recorded by pairing its birth and death events. Ordinary H0 persistence, the quantity this project computes for split trees, is the set of (peak birth, merge-saddle death) pairs: a component is born at a peak and dies when it merges at a saddle, and its persistence is the absolute fitness difference between those events. The global maximum's component never merges, so by convention its pair is closed at the global minimum; all pairs are therefore finite. Figures label this quantity simply "Persistence". Persistence is distinct from branch span, group amplitude, and extended persistence, and is not by itself a statistical confidence or a guarantee of robustness to measurement perturbations.
_Avoid_: Extended H0 for the closed global pair

**Extended persistence**: The pairing of essential classes (those that never die in the ordinary filtration) through an extended, up-then-down filtration using relative homology, following Cohen-Steiner, Edelsbrunner and Harer. This project does not compute it for split trees: the global maximum's pair is ordinary H0 closed at the global minimum by convention, not an extended-persistence pair. The full-network component sweep is a separate computation (see Full-network component persistence). Extended persistence is a different quantity from branch span and from ordinary H0 persistence.

**Branch span**: The absolute scalar difference across a displayed reduced-tree branch between adjacent critical points, such as a maximum and saddle or two adjacent saddles. Branch span is not elder-rule persistence.

**Stable vertex identity**: A scientific sequence identity carried through analysis and exported tables, independent of internal graph-library indices.

**Branch vertex count**: The number of original graph vertices represented by a displayed split-tree branch, including its upper endpoint and interior vertices and excluding its lower merger or root endpoint. Its minimum is one; the global root belongs separately to the tree.
_Avoid_: Interior sequence count, zero-count branch

**Full-network component persistence**: Pairing of connected-component events across both phases of an extended sweep. In the ascending convention, ordinary H0 pairs valleys with join saddles, relative H1 pairs peaks with split saddles, and extended-positive H0 pairs the global minimum with the global maximum. In the descending convention, ordinary H0 pairs peaks with split saddles, relative H1 pairs valleys with join saddles, and the global H0 pair runs from global maximum to global minimum. Relative H1 here does not count independent graph cycles. "Extended sweep" and "extended-positive" are Topographer's names for this two-phase sweep and its global pair; the split-tree persistence diagrams do not use them.

**Peak**: A local fitness maximum in the declared graph. The peak count of a graph equals the number of maxima (leaves) of its split tree. A peak's statistical support is a separate property. Equal-fitness neighbours form one split-tree leaf; ties never create extra peaks.

**Height**: The canonical name of the persistence axis: the scalar value at which a feature is born or dies. In datasets normalised to [0, 1] it equals fitness.

**Valley**: A local fitness minimum in the declared graph.

**Figure marker rules**: Three rules for the markers of every figure, with the Horton–Strahler palette they sit beside. (1) Valley markers only on the full graph: only `full_tree_persistence` and its vertical-histogram variant draw valleys, as a filled grey downward triangle (`▼`) on the tree (at the join-tree saddles that are tree nodes) and on the persistence diagram (join-tree pairs and the global minimum's mirror), with a "valley" legend entry; no other figure draws a valley, the hollow `▽` glyph is never used, and the global minimum on a tree is a plain vertex. A valley count may still be printed as a number. (2) No hollow markers: every marker is filled, triangles included; a figure that needs a second state uses a second colour (for example grey above and black inside the bootstrap band), never an outline. The stop-codon cross is a stroke with no interior and is the only unfilled mark. (3) Combined stop-and-synonymous encoding (the full-graph figures and the `*_combined` files of the saved-tree view families, which are ordinary registry figures built by `build figures`): a synonymous leaf branch is teal, darker the more vertices it holds; a diagram peak `▲` or valley `▼` is teal when synonymous and grey otherwise; a stop codon is the red `×` of the tree drawn on top of the marker, so a point that is both is a teal marker with a red `×`; the key shows `▲ peak`, `▼ valley`, `● synonymous` and `× ≥1 stop codon`. A valley is synonymous when it has the amino acids of the join saddle it dies at. The other figures keep the green leaf and crimson triangle. Horton–Strahler order is always drawn in one purple ramp (`HORTON_STRAHLER_COLORS`, orders 1 to 4, light lavender `#bcbddc` to near-black purple `#2a0a4a`). Presentation only; settings live in `config/figures.toml`.
_Avoid_: Open marker, hollow peak, valley triangle

**Descending component sweep**: A component-persistence sweep that starts above the largest fitness and lowers its threshold. Peaks give birth to ordinary H0 components; valley events belong to the later relative phase. Birth and death fitness remain the measured raw fitness values.

**Mirrored global-extrema representation**: The two orientations of the same global minimum–maximum relationship in a combined peak-and-valley persistence diagram. They represent the opposite sweep directions, not two independent scientific pairs.

**Full-graph fitness population**: The raw fitness values of every sequence in the full graph, with each sequence counted once, including stop-codon containing and noncritical sequences. This population defines the full-network fitness histogram and median.

**Tree-selected fitness population**: The raw fitness values of every vertex in the graph selected for a tree, counted once each, including retained nonfunctional fringe and noncritical vertices. Simplifying the displayed tree does not change this population.

**Fitness-bin vertex count**: The number of vertices in the declared population whose raw fitness falls within a given interval. It is distinct from branch vertex count.

**Functional core**: Vertices whose raw fitness is greater than or equal to a declared functional threshold.

**Nonfunctional fringe**: Below-threshold vertices retained as endpoints of original edges with at least one functional endpoint. Edges between two below-threshold vertices are excluded.

**Threshold-selected graph**: The graph containing original mutation edges with at least one functional-core endpoint and their endpoints, including the nonfunctional fringe, before largest-component selection.

**Selection rule**: The named procedure that turns the Full graph into a Selected graph. The active publication configuration token `"fdr_fixed_cutoff"` (user lock 2026-11-12) maps to `thr=<published|reflected>_fdr=by_q=0.05`, keeping functional sequences rejecting $H_0: s \le T$ under Benjamini–Yekutieli FDR control at nominal $q = 0.05$, retaining nonfunctional fringe endpoints, and taking the largest connected component; the checked-in default uses the Topographer threshold. Point-estimate rules (`"published"`, `"median_reflected"`) keep vertices whose raw fitness $s \ge T$ under the same edge and component rules; the alternative rule `"mixture_quantile"` (available as `config/variants/mixture_quantile.toml`) keeps mixture-probability functional vertices. Every quantity computed on a Selected graph names its selection rule.

**Largest-component landscape**: The connected component containing the greatest number of sequences in a declared prepared landscape. Other retained components are excluded from this analysis population.

**Reference peak set**: The DNA sequence identities in the Papkou2023 reference peak CSV, against which reproduced peak identities are compared.

**Raw fitness**: The measured fitness value for each sequence, before threshold-based selection or any transformation.

**Clamped fitness metadata**: The greater of raw fitness and the functional threshold, retained only for reference and selection diagnostics; it is not the analysis scalar.

**Adjacent saddle**: The first split-tree saddle encountered when descending from a peak. It can differ from the saddle paired with that peak by persistence.

**Synonymous branch endpoints**: Displayed branch endpoints with identical amino-acid sequences, preserving distinct stop states. Endpoint equality does not imply synonymous changes throughout the branch.

**Adjacent-saddle synonymous peak**: A peak whose amino-acid sequence equals its adjacent saddle's sequence. Its persistence point can carry this biological annotation independently of its persistence-paired death endpoint.

**Adjacent-saddle synonymous terminal leaf branch**: A terminal split-tree branch connecting a maximum to its adjacent saddle, where the maximum and saddle have identical amino-acid sequences. A root-only leaf is ineligible and excluded from the synonymous denominator.

**Root-only leaf**: A terminal merge-tree branch whose lower endpoint is the root rather than a saddle. It has no adjacent saddle, so it is neither synonymous nor non-synonymous: it is reported as ineligible and left out of the synonymous-leaf denominator.

**Common saddle**: The highest split-tree saddle connecting two nominated maxima, or the lowest join-tree saddle connecting two nominated minima. Immediately beyond its fitness (above it for maxima, below it for minima), those extrema belong to distinct superlevel components (maxima) or sublevel components (minima). For a peak pair it is their lowest common ancestor in the split tree, the highest node that lies on both peaks' downward paths to the root.

**Direct Hamming distance** _(proposed)_: The nucleotide (or amino-acid) Hamming distance between the sequences of two peaks. It is the baseline for the via-saddle Hamming distance.
_Avoid_: Null (a baseline, not a statistical null), geodesic distance (reserved for walked paths above the saddle, not computed)

**Via-saddle Hamming distance** _(proposed)_: For two peaks, the Hamming distance from peak 1 to their common saddle plus the Hamming distance from the common saddle to peak 2. It is never less than the direct Hamming distance; The excess over the direct distance comes from sites where the saddle differs from a peak: a site where both peaks agree but the saddle differs adds 2, and a site where all three sequences differ adds 1. It is not a walked path length.
_Avoid_: Tree barrier, detour length, geodesic distance (reserved for walked paths above the saddle, not computed)

**Direct fitness difference** _(proposed)_: The absolute difference in fitness between two peaks. It is the baseline for the via-saddle fitness distance.
_Avoid_: Null (a baseline, not a statistical null)

**Via-saddle fitness distance** _(proposed)_: For two peaks, how far peak 1 sits above their common saddle in fitness plus how far peak 2 sits above it. It exceeds the direct fitness difference by twice the lower peak's drop to the common saddle. It is distinct from persistence and branch span.
_Avoid_: Tree barrier, detour length

**Critical-point stop annotation**: A stop-codon flag on a split-tree critical point, or on an actual persistence birth/death endpoint. Stops in represented interior sequences do not establish this flag.

**Raw-fitness local maximum**: A sequence with no neighbour of greater raw fitness in the declared landscape. Equal-fitness neighbours do not disqualify it; this definition does not establish statistical support.
_Avoid_: Statistically significant peak

**Neighbour-supported maximum**: A raw-fitness local maximum whose comparison with every graph neighbour meets the declared adjusted-test threshold. Non-rejection for a neighbour does not establish equal fitness.

**Minimum neighbour score**: One minus the largest adjusted q-value among a maximum's neighbour comparisons. It ranks the limiting comparison's evidence and is not a probability that the maximum is real.

**Neighbour q-value threshold**: The largest allowed adjusted q-value for every neighbour comparison of a local maximum. Lower thresholds require stronger evidence and retain fewer maxima.

**Papkou population**: Every sequence in the Papkou-threshold selected graph used for the Papkou graph and codon-entropy analysis, including below-threshold fringe endpoints. Fitness values remain raw; the functional threshold determines graph selection.
_Avoid_: Published population

**KEM branch / TCL branch**: The two named populations of a Selected graph, each all sequence identities in the superlevel component containing the nominated KEM or TCL summit immediately above their common saddle. The saddle and shared trunk belong to neither. Other summit-side components are unassigned. "Branch" alone keeps its merge-tree meaning (an edge of a tree, as in Branch vertex count, Dimmed branch or Branch decomposition); a KEM or TCL branch is a population of several leaves, and one terminal branch of a merge tree stays a leaf.
_Avoid_: Ridge, KEM/TCL ridge population, a bare "branch" for either population

**Two-functional-ridges share**: The functional sequences in a Selected graph assigned to the KEM or TCL branch over all functional sequences in that Selected graph, compared with the configured minimum `two_functional_ridges_share_minimum`. In the codon-entropy claim summary its boolean is `claim_meets_minimum`, true when the share is at least the minimum (`>=`, so a share exactly equal to it meets it). The per-branch position-2 E/D enrichment rows of the same table use `claim_exceeds_baseline` instead, true only when the branch frequency strictly exceeds the baseline (`>`); each boolean is null on the other rows' kind.

**Leaf population**: The terminal merge-tree branch from a local peak down to its first adjacent descending saddle. Membership includes the upper endpoint (peak) and represented interior sequences and excludes the lower saddle; it does not necessarily extend to the persistence-paired death saddle.
_Avoid_: Ridge, branch population (a Branch is a KEM or TCL population of several leaves)

**Focused branch**: In a cropped tree excerpt, the branch under discussion, drawn at full opacity.

**Dimmed branch**: In a cropped tree excerpt, any branch other than the focused branch, drawn at reduced opacity with its colour unchanged.

**Non-significant singleton leaf**: A terminal branch representing only its peak sequence, whose peak-to-adjacent-saddle adjusted q-value is greater than the declared q-value threshold. Failure to reject this comparison does not establish equal fitness.

**Selected-leaf absorption**: A tree display transformation that removes nominated terminal leaves, reassigns their represented sequences to surviving structure, and removes merger saddles that cease to connect distinct surviving branches. Sequence identities and measured fitness remain unchanged.

**Selected edge**: An original mutation edge in a Selected graph whose adjusted q-value satisfies the declared strict rule `q > alpha`. This is an edge-level screening rule; failure to reject a fitness difference does not establish equal fitness or biological neutrality.

**Figure identifier**: The stable descriptive key of a publication figure, such as tree_comparison, used in config/figures.toml, the build command, and exported file names. It never encodes manuscript numbering.

**Figure spec**: The registry entry for one figure identifier: its required cached analysis inputs, allowed variants, configuration table, optional notebook, and one render callable. The build reaches a figure only through its spec, and figure tests exercise each spec through it.

**Figure variant**: One of the configured export forms of a figure identifier, vanilla or annotated. A spec declares which variants it supports; the build renders the intersection of requested and supported variants.

**Colour axis `none`**: The `color=none` value of a figure's colour setting: trees are drawn uncoloured, with no branch vertex-count colouring and no colour bar. Files are named `<stem>_color=none_style={vanilla,annotated}.png`, where `style` is the Figure variant. Presentation only; the tree and its peaks are unchanged.

**Peak rank figures A/B/C**: Three figures, built by `analysis/peak_rank.py` and `figures/peak_rank.py`, that order the Reference peak set (514 peaks, 43 of them stop-codon, none excluded) by rank. A, B and C label this one family and are not manuscript numbers; stable identifiers are in `config/figures.toml` and `docs/figure-inventory.md`. Peak rank is an ordering of Peaks, not a measure of statistical support.

**Slide tree**: A tree figure rendered for presentation slides by `build figures --slides`, with no title. The tree-only form shows the tree alone; the full form adds the axes, legend and annotations of the Figure variant. Outputs land in `figures/slides/tree_only/` and `figures/slides/full/`; the Colour axis `none` or count colouring applies as configured.

**View**: A named visual presentation or plot kind (such as `tree_persistence`, `tree_comparison`, `bootstrap_band`, `leaves`) rendered from saved products.

**Tree-edge contrast**: A direct pairwise fitness contrast fitted for the two critical points joining one reduced split-tree edge, oriented lower to upper, with its estimate, standard error, and Wald p-value. It is fitted from barcode counts independently of the wild-type-relative fitness used to build the tree, and never replaces node fitness.

**Tree-edge q-value family**: The set of all tree-edge contrasts of one declared split tree, adjusted together as one Benjamini–Hochberg family with its own named alpha. It is distinct from the all-fitness-diff family that supplies graph-edge q-values and from neighbour-peak testing. The adjustment is nominal: the edges are selected from observed fitness and the tests are dependent.

**Contrast discrepancy**: For one tree edge, the difference between the raw wild-type-relative fitness difference of its endpoints and the direct tree-edge contrast estimate, recorded as a signed gap and a sign-disagreement flag. It is a diagnostic annotation and does not alter the tree.

**Path to the root**: The ordered critical points met when descending a split tree from one maximum to the root, each reached through its lower tree edge. Every maximum has exactly one; paths of different maxima share their lower stretch below their common saddle.
_Avoid_: Trunk, root path, lineage

**Hill**: A peak together with every sequence on its terminal branch, down to but excluding its adjacent saddle. A hill is supported when any of its sequences is significantly fitter than that saddle, not only its peak.
_Avoid_: Bump, basin

**Hill support**: The fraction of plausible landscapes, drawn from each sequence's growth-rate uncertainty (its pooled standard error) and rebuilt into split trees, in which a peak lies among the hill's sequences. It qualifies a hill of the nominal tree as a bootstrap value qualifies a clade; it says nothing about the hill's extent, which historical peak cores described.
_Avoid_: Peak significance, hill p-value

**Sibling support**: The fraction of rebuilt trees in which two nominal hills join each other before either joins any other hill. It scores the merge event rather than the saddle sequence, whose identity moves between draws.
_Avoid_: Saddle support

**Bootstrap persistence band**: A diagonal strip of vertical height $2c$ in the persistence diagram, where $c$ is the bottleneck radius at quantile `bandq` of distances between bootstrap draws and the nominal diagram (Fasy et al. 2014); a peak survives if and only if ordinary H0 persistence $> 2c$. Its interpretation is limited to fixed-graph fitness perturbations, omitting threshold, graph-selection, and SE-estimation uncertainty, and does not provide calibrated peak-level coverage or FDR.
_Avoid_: Cone, confidence band (unqualified), significance of a peak, p-value

**Bootstrapped persistence threshold**: The persistence boundary $2c$ that a peak must exceed to survive band simplification, where $c$ is the bottleneck radius at quantile `bandq`. In the persistence diagram it is the diagonal strip of ordinary H0 persistence at most $2c$.
_Avoid_: Cone, radius as the threshold, q (for bandq), significance of a peak

**Band-simplified tree**: The split tree formed by direct absorption of nominal terminal branches whose ordinary H0 persistence does not exceed twice the bottleneck radius ($> 2c$) at quantile `bandq`, with no core prerequisite. Nesting within a core-rule tree applies only to historical core-rule outputs.
_Avoid_: Significant tree, denoised tree, core-simplified tree

**Absorbed branch**: A non-elder terminal branch whose hill is merged into its adjacent saddle by band simplification (or historically by the core rule). Its sequences keep their raw fitness; only the tree no longer shows it as a separate peak. Being absorbed does not establish that the peak is noise.
_Avoid_: Pruned branch, removed branch

**Kept branch**: A terminal branch that a smoothing step leaves in the tree as a separate peak, because its hill is not absorbed. The elder-protected branch is kept by construction, not by evidence.
_Avoid_: Significant branch, surviving peak (reserved for the Bootstrapped persistence threshold)

**Elder-protected branch**: The elder arm at a saddle, the arm whose subtree holds the highest maximum. It is never absorbed, so the global maximum always survives smoothing, whatever its evidence.
_Avoid_: Protected peak

**Branch decomposition**: The listing of every edge of a smoothed split tree as one branch, so that each sequence of the tree except the root is on exactly one branch. A branch is named by the sequence at its upper endpoint, a peak for a terminal branch and a saddle for an internal one, and records the elder peak whose elder path runs through it, the highest peak above it. Its branch vertex count, branch span and Horton–Strahler order (smoothed tree) are those of its edge, and its sequences are its upper endpoint, its interior vertices and the sequences of absorbed hills that smoothing assigned to the edge; the lower endpoint belongs to the branch below, and the root to none. Sequences absorbed straight into the root group are counted on the root edge of the global maximum. Its edge count is the number of single-mutation adjacencies of the Full graph (every sequence of the dataset, before the functional threshold prunes it) that join two of its sequences, which is not its vertex count less one; a single mutation between two sequences below the functional threshold counts, though it is not an edge of the Selected graph. Absorbed hills form no branches of their own. It is a view of the smoothed tree, not a persistence computation, and it is not Topographer's `BranchDecompositionTree`, the elder-rule hierarchy with one branch per persistence pair.
_Avoid_: Elder-rule branch (a peak's whole path to the saddle where it meets an older arm, which spans several edges and is not what this project calls a branch), BranchDecompositionTree (Topographer's name for that elder-rule hierarchy)

**Horton–Strahler order (smoothed tree)**: The Horton–Strahler order of the edges of a smoothed split tree, computed on the smoothed tree itself. A terminal edge has order 1; the edge below a saddle has the larger order of the arms that meet there, plus one when the two largest are equal. An absorbed hill is no longer an arm, so it does not raise the orders below it, and the same landscape can have a lower order in its smoothed tree than in its nominal one. A branch has the order of its edge, so a branch has order 1 exactly when it is terminal. It describes the shape of the tree, not fitness or statistical support.
_Avoid_: Order of the nominal tree, for a smoothed tree

**Topological noise band** (retained for the `median_reflected` comparison rule only): The square of the mirrored persistence diagram between the global minimum fitness `g` and its reflection through the median `m`, the Topographer threshold `m + (m − g)`; the spread of non-functional fitness below the median, reflected above it on the assumption that non-functional variation is symmetric about the median. It is not a calibrated error estimate.
_Avoid_: Bootstrap persistence band (reserved for the bootstrap band), error band

**Functional threshold**: The fitness boundary $T$ separating functional from below-threshold sequences in a Selected graph, defined by an explicit rule such as the Papkou threshold, the Topographer threshold, or a mixture quantile.
_Avoid_: Cutoff

**Candidate functional threshold** (historical): The fitness threshold $T$ at a declared quantile of the non-functional fitness distribution read from a fitted mixture in earlier workflows; kept for historical records.

**Papkou threshold**: The fitness threshold −0.507774 used by Papkou et al. (2023) to define their functional population; it selects the Papkou graph.
_Avoid_: Published functional threshold, Published threshold

**Topographer threshold**: The functional threshold obtained by reflecting the lower half of the fitness distribution about its median, T = m + (m − g) for the median m and the global minimum g; its value is −0.253159. It selects the Topographer graph.
_Avoid_: Median-reflected threshold

**Non-functional component**: A mixture component holding at least a declared share of the double-stop-codon sequences' summed posterior responsibility. The union of these components is the non-functional fitness distribution.

**Stop-anchored component identification**: Identifying the Non-functional components by double-stop-codon membership rather than by lowest mean, so the choice stays defined when the non-functional bell splits across several components.
_Avoid_: Lowest-mean rule (that is the Replication cell's rule)

**Functional probability**: The conditional Normal plug-in exceedance score `Φ((s_i − T)/se_i)` for a sequence with fitness `s_i`, treating the functional threshold `T` and Pooled standard error `se_i` as fixed. It is neither a mixture posterior nor a calibrated probability of biological function.

**Mixture-probability classes**: The three classes of Functional probability using the threshold supplied by the fitted mixture: Functional at or above the declared confidence level, Non-functional at or below its complement, and Uncertain between. The 0.95 level is a classification convention, not an FDR. The historical Mixture graph vertex set is the Functional class; Uncertain is reported, never a selection input.

**Fixed-threshold BY classes**: Sequence classifications under Benjamini–Yekutieli adjustment of gated one-sided p-values against a fixed functional threshold $T$: **non-functional** for below-threshold point estimates ($s < T$, gated $p = 1$), **uncertain** for above-or-equal-threshold sequences ($s \ge T$) without BY rejection, and **functional** for sequences rejecting $H_0: s \le T$ at nominal $q = 0.05$. These are distinct from mixture-probability classes.

**Replication cell**: The computation of Papkou et al.'s exact rule (three components, lowest mean, 0.999 quantile) on our data, always reported alongside the canonical threshold.

**Sensitivity scan**: A reported grid over five axes: mixture family (Gaussian, skew-normal, Student-t), component count `k` (1 to 5), quantile, confidence level and component rule (stop-anchored, lowest-mean). Every `(family, k)` carries a 95% interval on `T` (the 2.5% and 97.5% quantiles over `n_resamples` = 200 nonparametric resamples, sequences drawn with replacement, each seeded by its own index), and a model-free comparator gives the empirical quantiles of the double- and triple-stop sets. It describes how far the classification moves and is never used to pick the rule. Its tables are tables-only (`output/analysis/functional_threshold_scan/`); the Acceptance checks and the canonical interval are manuscript quantities.

**Acceptance checks**: Criteria declared before a run that the canonical classification must meet. For the Sensitivity scan they are (a) at least 99% of the double-stop sequences lie below `T`, (b) the 95% resampled interval on `T` is narrower than 2 × 1.96 × the median pooled standard error, and (c) one scan step from the canonical cell on any axis flips fewer than 5% of the Functional set. A failed check is reported, and the rule is not re-tuned to pass it.

**Median gap** (retained for the `median_reflected` comparison rule only): `h = m − g`, the distance from the full-graph median raw fitness `m` down to the global minimum `g`. Reflecting it above the median gives the Topographer threshold.

**Leading saddle**: In a merge tree, the saddle at which the first ordinary H0 pair closes during the sweep; qualified by tree kind. In the split tree it is the highest split saddle; in the join tree it is the lowest join saddle. It is a reported quantity and selects no graph.

**Trailing saddle**: In a merge tree, the saddle at which the last ordinary H0 pair closes during the sweep, adjacent to the root; qualified by tree kind. In the split tree it is the lowest split saddle; in the join tree it is the highest join saddle, by which point every sublevel component has merged. It is a reported quantity and selects no graph.

**Tree trunk**: In the Full graph's merge trees, the sequences beyond a tree's trailing saddle, toward its root, by raw fitness. The split trunk is every sequence with raw fitness at or below the split-tree trailing saddle (toward the global minimum, the split tree's root); the join trunk is every sequence at or above the join-tree trailing saddle (toward the global maximum, the join tree's root). Both are closed: the trailing saddle belongs to its own trunk. It is a population of sequences defined by a fitness cut, distinct from the Path to the root, which is an ordered list of critical points of one split tree.

**Leading saddle band**: The sequences of the Full graph whose raw fitness lies in the closed interval between the split-tree trailing saddle and the join-tree trailing saddle. When the join trailing saddle is at or below the split one the band is the intersection of the two Tree trunks (`overlap`); when it is above, the band lies between them and in neither apart from its endpoints (`gap`). The band and the trunks are not a partition. Whether the median lies inside the band is to be tested, not assumed.

**Reeb graph**: The quotient of a scalar field on a Selected graph that contracts each connected piece of each level set to a point. On a graph with cycles it is in general not a tree: it keeps a loop wherever two monotone routes join the same pair of critical points. The split tree and join tree are its one-sided views and are always trees.
_Avoid_: Contour tree (reserved for simply connected domains)

**Terminal co-occurrence** (historical): For two sequences, the fraction of bootstrap draws in which both are members of the same terminal branch of the rebuilt split tree in archived analyses. It defined historical peak cores; active pipelines do not produce or use it.
_Avoid_: Co-membership, branch overlap

**Peak core** (historical): A maximal set of sequences in which every linked pair has terminal co-occurrence at or above the declared core threshold in archived analyses. It evaluated hill reproducibility prior to the adoption of direct bootstrap persistence band simplification.
_Avoid_: Consensus peak, stable peak, hill stability

**Bootstrap draw**: One resampled landscape and the split tree rebuilt from it.

**Pooled standard error**: The standard error of a sequence's growth rate `r` from a single model fit to all of that sequence's replicates together, with one fixed dispersion. It is the per-sequence noise scale of the bootstrap draw and of the peak-uncertainty figure. It agrees closely in the marginal comparison with the uncertainty Papkou et al. 2023 report for wild-type-relative fitness `s = r − r_wt`, but independent plug-in perturbations do not reproduce the shared wild-type covariance of those differences. A common recentering leaves persistence unchanged. "Pooled" means over replicates of one sequence, never across sequences. It does not absorb between-replicate overdispersion; the replicate standard error does, and their ratio diagnoses it.

WT-normalization uncertainty and overdispersion are not accounted for by the pooled plug-in error model and are not repaired by BY adjustment (see `threshold/classification_probability_scans.py:2091`).
_Avoid_: Model-based SE, Wald SE

**Replicate standard error**: The standard error from fitting one growth rate per replicate and taking the standard deviation of those estimates divided by the square root of their number. It absorbs between-replicate overdispersion and is typically larger than the pooled standard error. It is reported as a diagnostic of that overdispersion and is not the bootstrap noise scale (ADR `2026-10-22-pooled-standard-error-for-bootstrap` (2026-10-22)).
_Avoid_: Empirical SE, across-replicate SE

**Hamming distance**: The number of positions at which two equal-length sequences differ, over a named alphabet; qualified in new text as nucleotide or amino-acid Hamming distance. Measured from the wild type unless another reference is named, such as a branch's two endpoints or a peak pair.

**Growth rate** (r): A genotype's Malthusian parameter, its exponential growth per generation. BarSeq counts determine it only up to one constant shared by the whole pool, so every genotype including the wild type carries an estimate and a pooled standard error (with a replicate standard error beside it), and differences between genotypes are exact.
_Avoid_: Absolute fitness, growth slope

**Selection coefficient** (s): The difference of two growth rates, variant minus wild type. The source study reports this quantity under the name m; the pool constant cancels in it.
_Avoid_: m, Malthusian coefficient

**Flat stretch**: A contiguous run of critical points on one path to the root in which every pair is not significantly different in fitness. Non-significance is required pairwise because uncertainty need not scale with the fitness difference; it does not establish equal fitness.
_Avoid_: Plateau, uncertainty block

**Monotone flow skeleton** _(proposed)_: The subgraph of the declared landscape formed by monotone fitness paths that connect the split tree's critical points from the root to every maximum. Vertices outside it do not contribute to component births or merges. Exploratory; not used by the band-simplified tree.
_Avoid_: Bedrock graph

## Reproducibility

**Stage**: One named unit of the analysis that turns declared inputs into a published set of outputs; stages depend on each other only through those outputs.

**Fingerprint**: The identity of a stage result, derived from the scientific settings, input data and code that produced it; two results with the same fingerprint are interchangeable. Scheduling settings such as worker count are not part of it. The bootstrap stage is the one deliberate exception on the code side: its key is the run's scientific identity (result settings, functional threshold, the selection identity (the rule and, under `mixture_quantile`, the rounded `T` and the hash of the Functional vertex set), input dataset digests, growth-rate settings and BarSeq digests) and holds no code digest, so a published checkpoint or a tracked mirror stays reusable across code edits (ADR 0013, accepted deviation); its full fingerprint, with code digests, stays in the stage's own record.

**Result label**: A deterministic, readable parameter string identifying a saved result's scientific choices in pipeline data-flow order, with run fields immediately preceding `hash=<8hex>`, the first 8 hex characters of the scientific identity SHA-256.
_Avoid_: Legacy unparsed directory tags

**Scientific identity hash**: The short `hash=` field of a result label: a digest of what a product is scientifically (its type, scientific parameters, parents and input data). Where and when it ran, the environment and where it is stored are not part of it, so two runs of the same science share it.

**Product**: A typed, saved output artifact of the analysis pipeline: `fitness` (threshold $T$ plus sequence-level statistics and classifications), `landscape` (selected graph), `tree` (nominal or band merge tree), `bootstrap` (multi-file bootstrap draws and summary), `leaves` (leaf population membership and characteristics), `landscape_summary` (graph and population metrics), or `ridges` (KEM/TCL ridge memberships and metrics).

**Described products**: The existing summary, biological-annotation, and branch-count outputs returned by the single `analysis.describe(...)` entry point.

**Run fields**: Execution-context identifiers (`host=<hpc|local>_run=<id>`) recording where and when a job ran, positioned immediately before `hash=<8hex>` in file names and excluded from the scientific identity hash.

**Pinned run**: (Historical; `config/runs.toml` was removed.) An explicit entry nominating the canonical run name for a product. The directory name in `data/processed/<kind>/<state>/` is now the identity.

**Checkpoint**: A complete, published stage result, written once and reused by fingerprint; never modified in place.

**Mirror**: A version-controlled, parameter-named copy of an expensive checkpoint, written exactly once so the result survives loss of the working output; restoring from a mirror never changes it.

**Git stamp**: The `git` key (`commit`, `dirty`) that provenance records written by the produce-or-load helper carry as a human-readable pointer to the repository state a result was produced from. It is taken when production starts, not when it ends. `dirty` counts changes to tracked files only, and both values are null outside a repository or without git. It is never part of a fingerprint, a reuse decision or a digest of a whole record, so a dirty tree or a docs-only commit never invalidates a result. A stage that keys on an upstream record names its fields (`fingerprint` and the artefact hashes) or drops the stamp first, and never hashes a provenance file's bytes.

**Variant**: (Proposed, ADR 2026-11-19-parameter-keyed-stage-results.) The parameter string of one run: the pairs `threshold` (`published`, `reflected` or `none`, where `none` names the Full graph), `fdr` (`none` or `by`) and, for `by`, `q` (0.05), joined by underscores exactly as in a result folder name and without the `key=` suffix, for example `threshold=reflected_fdr=by_q=0.05`. It is the argument of `just hpc-run`, and the variant files under `config/variants/` that `hpc-run` runs are named by it. Different variants give results that coexist; `threshold=none_fdr=none` is the Full graph and by default stops before the bootstrap. The nicknames on main (`point_papkou_original` and the like) name the same four variants until that ADR is implemented. It is not a Figure variant (the vanilla or annotated export form of a figure).

**Step key**: What identifies one saved result of a step: the step name, its scientific parameters and the content hash of each upstream result it reads. It holds no code hash, no version number and no git commit, so editing code without changing a result changes no key (ADR 2026-11-19-parameter-keyed-stage-results). Its first eight hex characters are the `key=<8hex>` suffix of the result's folder, `<step>/<key=value pairs joined by underscores>_key=<8hex>/`. Parameters appear in pipeline data-flow order (display only; the key is computed from the names sorted, so order never changes it), with no `_` or `-` inside a name or value. The Full graph is `threshold=none_fdr=none`. The full key and the stored identity (step name, canonical parameters, upstream content hashes) are checked on load. The same parameters and upstream give the same folder; a changed upstream gives a new folder beside the old one. This replaces the code-based Fingerprint above, and the folder grammar and run fields of the Result label, once that ADR is implemented. The ADR gives a part-by-part account of what it supersedes in ADR 0016; figure names carry the parameters but no key.

**Label**: (Proposed, ADR 2026-11-19-parameter-keyed-stage-results.) What a saved result's provenance records for people to read: the git commit and dirty flag, the host and job number, Topographer's version and commit, and the digests of the code files and `uv.lock`. It is never part of a Step key and never decides whether a result is reused; a changed label only shows as "source changed since" in the status command. A forced replace records the previous label, key and content hash in the new result's provenance under `replaces`. It extends the Git stamp. It is not a Result label, which is the parameter string that names a result's folder or file.

**Force**: (Proposed, ADR 2026-11-19-parameter-keyed-stage-results.) Recomputing one named step even though its Step key already has a saved result, with `--force <step>`. The local result is replaced in place (the `data/reference/` mirror only by an explicit flag), the one exception to never overwriting a published result; the previous bytes stay in the Bucket archive of the job that made them, since Bucket job folders are never rewritten. Steps downstream of it rerun only if its output actually changed. After a code fix, force the step that was wrong; to reproduce an old result, check out the old commit and force. The new provenance records the replaced label, key and content hash under `replaces`.

**HPC run**: (Historical; the cluster workflow was removed.) An execution of one stage on the OIST HPC cluster from a pushed, recorded commit of this repository.

**Result dataset**: One stage's checkpoint, identified by its fingerprint and the host that produced it, and carrying provenance (commit, job, hardware, library versions). Two hosts running the same configuration yield two result datasets; the publication cites one by name.

**Draw slice**: A contiguous range of bootstrap draws run as one unit of a sliced bootstrap. Its slices are merged into a single result dataset with the same fingerprint as an unsliced run.

**Reported quantity**: A single number the manuscript states, identified by a stable key independent of manuscript numbering. Each reported quantity records its exact value, the population it is computed over, the selection rule applied to that population, the method and settings that produced it, the source revision, and the rounding used in the manuscript text. Reported quantities are generated from analysis outputs, never typed by hand; the manuscript text references them by key.

**Manuscript quantity registry**: The complete set of reported quantities for one analysis run, held in one source table of reported quantities, each identified by a fully qualified key naming its population and quantity, from which every human-readable rendering (tables, figure labels, the compact list of manuscript numbers) is derived. See Reported quantity.

**Gap row**: A row of the manuscript quantity registry whose quantity the stage cannot source, because its upstream output is absent or no stage computes it. Its value and display value are null and its gap reason says why and starts with a kind: `stale:` (an upstream output exists but is not current), `missing:` (the output is absent or no stage computes the quantity) or `undefined:` (the stage ran and the quantity has no value for its data); it is named rather than dropped, so a missing number is visible in the report and Table 1 shows an em dash. A malformed or stale upstream output raises instead of becoming a gap row. See Manuscript quantity registry.

**Ordinary H0 pair**: One (peak birth, merge-saddle death) pair of ordinary H0 persistence, the global maximum's pair (closed at the global minimum by convention) included. A figure that counts these pairs says "ordinary H0 pairs", not "peaks": a peak is a split-tree maximum, and the two counts are different quantities. See Persistence.
_Avoid_: Peak (for a pair count)

**Band quantile**: The chosen quantile `bandq` (0.95 by default) of the bottleneck distance between bootstrap persistence diagrams and the nominal diagram, setting the radius $c$. Figure labels write "band quantile", never a symbol.
_Avoid_: q (reserved for FDR q-values), confidence level

**Radius**: The bottleneck distance quantile $c$ at `bandq` between bootstrap persistence diagrams and the nominal diagram. Twice the radius ($2c$) defines the threshold for the Bootstrap persistence band. Figure labels write "radius", never a letter.
_Avoid_: c, radius as the threshold

**Bootstrap sanity check**: A descriptive check, not the verdict, that the nominal-landscape draws sit far below the flat-landscape draws (draws from a landscape with no structure): the share of nominal-landscape draws whose bottleneck distance is strictly below the flat-landscape draws' fifth percentile, and the medians of both. Its figure lines say "Sanity check, not the verdict", and its registry keys name what is checked (`bootstrap.global_stability.sanity_check.*`).

**Frozen reference run**: An earlier bootstrap run kept as a tracked copy of its scan and summary (`data/reference/`) for comparison only. It is not a quantity of the current Manuscript quantity registry, so a figure that shows it labels it "frozen reference run" and never as a current-run quantity, and takes its numbers from the copy, not from the registry.
_Avoid_: Archived run (as a label), previous run (it is not the run before the current one, it is a fixed copy)

**Merge-flow component** (retired / historical): A tree arc of a saved merge tree (upper node to lower node) together with the subtree above it. It is born at the scalar of its upper endpoint, dies at the scalar of its lower node where it merges, and is named by its elder summit (highest summit among the arcs entering its upper node, ties broken in the saved product by the smaller `vertex_index`, the landscape vertex payload order, as `enforce_elder_rule` does, with `tie_break` "vertex_index" in `summary.json`; the smaller DNA sequence is only the fallback when no vertex order is supplied, `tie_break` "dna_sequence"; the elder cross-check against `persistence_pairs` raises on any disagreement). It is a different quantity from branch span and from ordinary H0 persistence, and it is the rigorous alternative to the manual KEM/TCL ridge. The saved `merge_flow` product (components, events and levels tables, plus `summary.json`) holds one such table set per state; a band state keeps the nominal counts and names and only marks absent components pruned. Pruning to the display floor `min_functional_count_K` (10, provisional, nominal states only) never changes a count or a name. The `merge_flow` view and its component model were retired from the publication set (PR #214 / PRD 0.4) in favor of branch transparency (Ticket 20); its view builder is deleted and only the product builder remains. See GLOSSARY entries for "branch span" and "ordinary H0 persistence" for the distinct quantities.

**Balanced merge**: At a merge saddle, the second-largest child functional count, that is the smaller of the two largest children. It is stored with the smaller child's total count, the saddle drop (younger summit fitness minus saddle fitness) and the largest child functional count; counts never change with display pruning.

**Functional width**: The width of a merge-flow component at a fitness level: the count of its members with fitness at or above the level whose class is `functional`; uncertain sequences are counted with non-functional ones. The Full state has no class, so its width is the count of all members (`width_basis` "all"). Functional widths do not add up at merges, because non-functional members are excluded.
