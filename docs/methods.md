# Methods

> Naming: this document may still use earlier wording (Published, Candidate, median-reflected, ridge). The current display names (Papkou, Topographer, "+ FDR", bootstrapped, branch) and how they map to the internal state names are in [GLOSSARY.md](../GLOSSARY.md).

Working drafts and verification notes for the manuscript Methods, Supplementary Methods and Discussion. This file is a handoff, not manuscript source. Graph sizes, peak counts, band widths, entropy values and biological findings belong in Results; procedure parameters are filled only from verified provenance. Statements marked `[PENDING FINAL RUN: ...]` depend on a run that has not been accepted. Documentation here does not certify completion of the rerun.

The Candidate rule (Benjamini–Yekutieli at nominal q = 0.05 with the Topographer cutoff) is exploratory: not yet, not citable as a finding until independent statistical review and a user decision (see the Candidate-rule review caveats in this document). Its numbers are exploratory, not findings.

Contents: [Methods](#methods-draft), [Supplementary Methods](#supplementary-methods), [Discussion notes](#discussion-notes), [Verification ledger](#verification-ledger), [Rerun checklist](#rerun-checklist).

## Methods draft

### Fitness estimation and graph construction

We analysed the demultiplexed sequence counts of [Papkou et al. (2023)](https://doi.org/10.1126/science.adh3860), retaining their preprocessing. For each genotype, we pooled before- and after-selection counts from six replicate populations and fitted a grouped binomial generalised linear model with a common intercept and slope:

\[
\tilde y_{ij}\sim\mathrm{Binomial}(\tilde y_{ij}+D_{ij},p_{ij}),\qquad
\operatorname{logit}(p_{ij})=a_i+r_i t_j.
\]

Here y_ij is the raw genotype count, D_ij = n_j − y_ij is its complement in the original sample total, and ỹ_ij = max(y_ij, 1). Complements were calculated before zero-count replacement. Time was measured in generations. The slope r_i describes log-odds read share per generation, rather than absolute growth. Fitness was s_i = r_i − r_WT; the pooled slope's Wald standard error supplied genotype-specific uncertainty.

Graph vertices represented distinct DNA genotypes, joined when they differed at one nucleotide. We compared the full graph, the source-study graph and a Candidate graph. The Candidate graph was defined by selection under the Candidate rule among all 261,333 genotypes (261,332 measured plus wild type), using each genotype's fitness s_i and pooled standard error σ_i.

The Candidate functional threshold T was the Topographer cutoff, T = m + h with m the median raw fitness of the full-graph population (m = −0.713981) and h = m − g, where g is the global minimum raw fitness (g = −1.174803; h = 0.460822). This gave T = −0.253159. Genotypes were selected by the Benjamini–Yekutieli (BY) procedure at nominal q = 0.05 with this cutoff. [PENDING FINAL RUN: number of genotypes selected by BY at q = 0.05 and the class breakdown, if any, under the median-reflected rule, and any interval on T]

An earlier mixture-quantile classification (T = −0.497395) was explored and is not the rule used here. [PENDING FINAL RUN: acceptance checks, sensitivity scan and replication comparison for the median-reflected rule]

For the Candidate graph, we retained original edges with at least one endpoint passing the rule, so that non-selected neighbours were kept, then selected the largest connected component. Under this exploratory rule the nominal Candidate graph had 98,131 vertices and 233,444 edges (the vertex count is not the count of BY-selected genotypes, since the largest component also contains non-selected neighbours) and 135 nominal peaks, against 135,178 vertices and 514 nominal peaks for the source-study graph at the published threshold (−0.507774); [PENDING FINAL RUN: retained-peak count after the bootstrap band at q = 0.95]. Retained genotypes kept their measured fitness; saddle positions were not used to set T. The source-study graph used the published threshold.

### Trees and sequence annotations

Split trees followed connected superlevel components as the fitness threshold decreased; join trees used the corresponding increasing sublevel scan. Under the elder rule, a peak was paired with the saddle where its component joined an older component. Ordinary H0 persistence was the birth–death fitness difference. We closed the global maximum's pair at the global minimum by convention, without computing extended persistence. Branch span instead measured the fitness difference between adjacent displayed critical points.

Branch counts recorded assigned genotypes. Horton–Strahler order was one at terminal branches; below a merger, the largest incoming order increased by one when it occurred at least twice. Orders were recalculated after simplification. DNA and amino-acid Hamming distances compared persistence endpoints. Endpoints were classified as synonymous when DNA distance was positive and amino-acid distance was zero. Peak stop annotations indicated at least one stop codon.

For ridge diversity, we fixed summit-side genotype populations strictly above their common saddle using simplified-tree assignments. At each threshold, we retained members with raw fitness at least that threshold. With equal genotype weights, positional entropy was H = −Σ_kΣ_a p_ka log₂ p_ka, where p_ka is the frequency of state a at position k. Amino-acid encoding distinguished all three stop codons.

### Resampling and simplification

The resampling results in this section depend on which archive backs each figure, and that figure-level provenance remains unconfirmed until each figure's archive is verified before its selection rule is assigned. The retained 10,000-draw archive `27869524` is recorded as the median-reflected (T = −0.253159) BY, nominal q = 0.05 rule on the older snapshot table hash, conditional on the selected graph; that snapshot has identical classes, fitness, standard errors and z-scores but not identical artifact fingerprints relative to the current snapshot. Neither establishes end-to-end selection uncertainty. The older reference datasets recorded the earlier saddle-based threshold (−0.283918). No bootstrap dataset has been regenerated from the current snapshot; any that do not match the Candidate rule are pending rerun and are not results for the present Candidate graph.

On the fixed Candidate graph, we independently sampled s_i^(b) ∼ Normal(s_i, σ_i²), using the pooled slope Wald standard errors σ_i, and rebuilt each split tree and persistence diagram. For bottleneck distances from draw diagrams D_b to the observed diagram D_0, we set

\[
c=Q_{0.95}\{d_B(D_b,D_0)\},\qquad \tau=2c.
\]

Following the confidence-band motivation of [Fasy et al. (2014)](https://doi.org/10.1214/14-AOS1252), nominal peaks cleared this approximate parametric band when persistence was strictly greater than τ. We directly absorbed failing branches into surviving lower structure or the root and contracted redundant saddles, preserving sequence identities and fitness. Sensitivity was assessed through the count N(τ) of peaks exceeding τ and τ(q) = 2Q_q{d_B(D_b,D_0)}. Supplementary flat draws used zero means with the same uncertainties and graph, and were compared with D_0.

We used BarSeqFit for estimation and Topographer for graph topology, trees and persistence; software versions and resampling parameters are specified in the Supplementary Methods.

## Supplementary Methods

### Count model and uncertainty

The grouped response was (ỹ_ij, D_ij) with D_ij = n_j − y_ij calculated before replacing zero genotype counts by one, so its total could exceed the sample total n_j by one. Sample totals included all mapped reads. Each genotype's pooled observations were fitted with the binomial logit model given in the Methods, with intercept and slope shared across replicates and 3.5166197103891137 generations between sampling times. Wild-type-relative fitness was s_i = r_i − r_WT. The genotype-specific uncertainty σ_i was the model-based Wald standard error of the pooled slope (`se_pooled`), including for the wild type. Thus σ_i was SE(r_i); independent perturbations omitted uncertainty and covariance from the shared wild-type subtraction. This uncertainty entered the scalar perturbations below; subtracting a common wild-type estimate translated the nominal landscape without changing persistence.

### Graphs, scans and branch metrics

The full graph comprised measured DNA genotypes and single-nucleotide substitution edges. For the source-study graph, an original edge {u,v} was retained when max(s_u,s_v) was at least the published cutoff. For the Candidate graph, it was retained when at least one endpoint passed the Candidate rule (BY at nominal q = 0.05 with the Topographer cutoff, T = −0.253159). In both graphs we retained the edge endpoints and selected the connected component with the greatest number of vertices. This rule included non-selected neighbours of selected genotypes but excluded edges joining two non-selected genotypes. All retained scalar values remained unchanged.

Decreasing superlevel scans produced split trees; increasing sublevel scans produced join trees. At a split-tree merger, the elder component continued and younger components were paired with the saddle. For birth b and death d, ordinary H0 persistence was b − d. The global maximum was paired with the global minimum by finite-root convention; the opposite scan used the corresponding opposite-extremum closure. These conventions did not compute extended persistence.

Displayed branches connected adjacent critical points, with span equal to their absolute fitness difference. Branch counts included the upper endpoint and assigned interior genotypes, excluding the lower endpoint; the root was treated separately. These counts differed from mutation-edge counts and elder-rule persistence. Horton–Strahler order was one for terminal branches. At a merger with maximum incoming order h, the outgoing order was h + 1 when at least two incoming branches had order h, and h otherwise, including multifurcations. Simplified-tree orders used the simplified topology.

### Sequence annotations and entropy

Persistence endpoint Hamming distances counted nucleotide or amino-acid mismatches between paired birth and death genotypes. Adjacent-saddle annotations instead used the first saddle below a peak. DNA genotypes remained distinct. Translation used 20 amino acids and three separate stop states: TAA = %, TAG = ^ and TGA = &. A peak's binary stop flag indicated at least one stop codon in that genotype.

For each nominated summit, we fixed its ridge population from band-simplified-tree assignments strictly above the common saddle joining the summits. The two summit-side populations were disjoint and excluded the saddle and shared lower structure. At threshold t, members with raw fitness ≥ t contributed equally; connectivity was not recalculated. For state frequencies p_ka(t),

\[
H_k(t)=-\sum_a p_{ka}(t)\log_2p_{ka}(t),\qquad
H(t)=\sum_k H_k(t).
\]

Zero-frequency terms contributed zero. Nucleotide states were A, C, G and T, and amino-acid states were the 23 symbols above. This summed marginal positional diversity was not the joint entropy of complete sequences.

### Bootstrap band and direct simplification

The bootstrap results were produced under earlier thresholds (the Topographer threshold, −0.253159, for the 10,000-draw dataset behind the six bootstrap figures, which the config describes as the point-estimate median-reflected rule, without BY, whereas the Candidate rule adds BY at q = 0.05 [PENDING FINAL RUN: confirm which selection the retained 10,000-draw bootstrap used]; the earlier saddle-based threshold, −0.283918, for the older reference datasets) and are pending rerun under the Candidate rule; the Candidate graph named here is the graph built under the earlier thresholds until the final run. [PENDING FINAL RUN: bootstrap rerun under the Candidate rule]

For each of B = ___ draws, with random seed ___, we independently sampled s_i^(b) ∼ Normal(s_i, σ_i²) on the fixed Candidate graph and rebuilt its split tree and ordinary H0 diagram. We computed δ_b = d_B(D_b,D_0), the bottleneck distance to the observed nominal diagram. Off-diagonal matching used L-infinity cost; a point of persistence p could match to the diagonal at cost p/2. Bottleneck distance minimised the largest matching cost. We defined c = Q_0.95({δ_b}) and τ = 2c, retaining nominal peaks with persistence strictly greater than τ.

This approximate parametric band follows the confidence-set motivation of [Fasy et al. (2014)](https://doi.org/10.1214/14-AOS1252); independent Normal perturbations omit cross-genotype covariance and condition on graph selection, so exact simultaneous coverage is not assumed. Supplementary flat draws used zero means with the same σ_i and fixed graph. Both observed and flat draws were compared with the observed D_0, while only observed-landscape distances determined c.

We simplified the nominal tree directly by absorbing failing branches into surviving lower structure or the root and contracting redundant merger saddles. Genotype identities and measured scalars were preserved, with assigned genotypes transferred to receiving structure. Branch counts and orders were recalculated. Recomputed persistence pairs verified the surviving nominal pairs without refitting fitness.

We calculated N(τ) = Σ_i 1{p_i > τ} for nominal persistence values p_i, and τ(q) = 2Q_q({δ_b}). The empirical q = 0 quantile was the minimum observed distance. This scan measured sensitivity to the band quantile, rather than Monte Carlo convergence.

Count estimation used BarSeqFit 0.2.0; topology, trees and persistence used Topographer 0.5.0, with Python version ___.

## Discussion notes

To compare the effect of the functional cutoff, we retain the Papkou estimation procedure, pseudocount convention and the at-least-one-endpoint graph rule. This holds the upstream analysis fixed while changing how functional genotypes are defined. It does not establish that the original estimation assumptions or either cutoff provide a definitive biological classification of viability.

The pooled binomial model describes uncertainty conditional on its sampling assumptions and does not model additional variation between experimental replicates. Such variation can make pooled standard errors understate uncertainty. A separate analysis of the Papkou export reported a median replicate heterogeneity statistic, Cochran's Q divided by its degrees of freedom, of 1.844 across 261,332 test genotypes. This diagnostic motivates a limitation of the uncertainty model; it is not a replacement standard error. The intended primary analysis retains the pooled binomial slope standard error, with alternative uncertainty models requiring a separately identified sensitivity analysis.

Pseudocount choice can also affect estimated growth. In the same separate analysis, removing pseudocounts from sporadic zeros changed 28,694 of 261,332 test-genotype estimates (11.0%); the median change among affected genotypes was −0.01459. That comparison reported no changes in effect sign, convergence status, or nominal or BH significance calls. These results describe that particular comparison and do not establish that pseudocounts are immaterial to cutoff membership or topology. We retain the published convention for the primary comparison.

Low counts after selection need not identify unreliable measurements alone: depletion can be the biological signal of strongly reduced growth. In the separate count-filter analysis, increasingly strict after-count filters preferentially removed strongly deleterious genotypes. Coverage therefore warrants inspection alongside uncertainty and biological context, rather than automatic exclusion. Multiple-testing adjustments matter for downstream collections of significance tests when those tests are used; their availability in the fitting package does not make them part of the present topological analysis.

The Candidate functional threshold, T = −0.253159, is a statistical rule, not a validated viability threshold. The graph rule keeps a mutation edge if at least one endpoint passes the rule, followed by selection of the largest connected component. This rule can retain non-selected fringe vertices and their edges to selected vertices; it does not induce a graph consisting only of selected sequences. The nominal Candidate graph has 98,131 vertices and 233,444 edges and 135 nominal peaks. [PENDING FINAL RUN: any interval on T] An earlier mixture-quantile classification (T = −0.497395) was explored and is not the rule used here. [PENDING FINAL RUN: sensitivity scan and acceptance checks for the median-reflected rule]

For the source-study graph at the published threshold (−0.507774) there are 514 nominal peaks and 135,178 vertices, against 135 nominal peaks and 98,131 vertices for the Candidate graph. 261,333 is the observed Full-library denominator, and no Peak point was clipped in the figures. [PENDING FINAL RUN: replication of the source study's threshold rule on these data]

The bootstrap-derived results (the uncertainty band, simplified trees, retained-peak counts and their figures) are pending the final run under the Candidate rule. [PENDING FINAL RUN: retained-peak count after the bootstrap band at q = 0.95]

The parametric persistence-diagram uncertainty band is an approximate assessment conditional on the fitted means, standard errors, perturbation model and graph. A stability argument of the kind used for persistence confidence sets motivates the construction; it does not by itself establish calibration for this particular bootstrap. The conventional q = 0.95 result should be accompanied by the retained-feature count as q varies. A plateau near q = 0.99 would describe sensitivity, not demonstrate correct coverage. The final run under the Candidate rule must determine the numerical counts and whether substantive conclusions remain supported.

Merge trees summarize the evolution of connected components across fitness thresholds. They do not encode every topological invariant or establish the existence of all fitness-monotone paths. Biological interpretations should stay within the component relationships and persistence quantities actually measured.

## Verification ledger

The BarSeqFit reports named below live in the sibling `barseqfit` repository (its ADR 0005 and its 2026-10-15 modelling-cleanup-impact report).

| Item | Evidence/status | Consequence for drafting |
| --- | --- | --- |
| Pseudocount and dispersion defaults | BarSeqFit ADR 0005: opt-in extensions preserve published defaults; replicate summaries are diagnostic. | Explain compatibility and limitations separately; do not silently replace pooled SE. |
| Heterogeneity and pseudocount comparison | BarSeqFit impact report, sections 2 and 4, read during drafting. Numbers above are report-derived; the underlying analysis was not rerun here. | Restrict claims to this export and comparison. No inference that topology is unchanged. |
| Low-coverage interpretation | Same report, section 3; after-count filters preferentially remove strongly deleterious genotypes. | Do not equate low coverage with a failed estimate or assert this pattern for every assay. |
| Published cutoff | Reference study: [Papkou et al., Science](https://doi.org/10.1126/science.adh3860). Primary-source cutoff verification belongs with the Methods audit. Working value supplied for comparison: −0.507774, rounded −0.508. | Verify exact source location before manuscript use. Do not describe all stop-containing genotypes as lethal artifacts. |
| Original cutoff derivation | ADR 2026-10-24 cites Papkou et al. SM pp. 14–15: three-component Gaussian mixture, bell-shaped component, 0.999 quantile, validated on double-stop variants. Not re-read from the supplement; the exact-at-two or at-least-two stop definition is unverified. | Cite the derivation as reported in the ADR; verify against the supplement before manuscript use. |
| Historical: mixture-quantile threshold and classes (not the Candidate rule) | Registry keys `functional_classification.mixture_quantile_functional_threshold_fitness`, `.mixture_component_count`, `.functional_class_count`, `.non_functional_class_count`, `.uncertain_class_count`, `.double_stop_fraction_below_threshold`; `output/checkpoints/functional_classification/summary.json` (regenerated, not committed); an earlier cluster run. | Reported as historical only; do not call T a validated viability threshold. |
| Historical: mixture-quantile sensitivity scan (not the Candidate rule) | Registry keys `functional_threshold_scan.acceptance_a_passed` to `_c_passed`, `.canonical_threshold_interval_low` and `_high`, `.max_flip_fraction_one_step`; `output/analysis/functional_threshold_scan/acceptance.json`. Check (c) fails on the quantile axis only. | Report the failed check as a sensitivity; do not re-tune the rule. |
| Historical: Papkou replication cell (mixture-quantile rule) | Registry key `functional_classification.papkou_replication_functional_threshold_fitness` (−0.546201 against −0.507774). | State as open and unexplained; do not offer an explanation. |
| Reported double-stop peaks (historical, mixture-quantile T = −0.497395) | User observation: some double-stop genotypes reportedly lie above the original cutoff and become local peaks. The classification summary (`stop_codon_fractions_below_threshold`) gives 1,639 of 1,647 strictly below T, so 8 at or above; whether those 8 are local peaks, and which sequences they are, remain to be checked. An upper-quantile cutoff can leave an upper tail by construction. | Treat as an observation to verify, not an established result; presence above the cutoff alone does not invalidate it. |
| Persistence confidence construction | Fasy et al., published 2014 (preprint 2013), [DOI 10.1214/14-AOS1252](https://doi.org/10.1214/14-AOS1252). Applicability and calibration assumptions remain to be checked against the implemented parametric procedure. | Use approximate, conditional language; do not claim a proven calibrated confidence level. |
| Bootstrap rerun | The bootstrap has not been rerun under the Candidate rule (median-reflected, BY q = 0.05). Only the earlier 10,000-draw dataset and the band-quantile scan derived from it without new draws (both left the repository with the dataset pin system and are in its history), which together fed the six bootstrap figures, are recorded at the Topographer threshold (−0.253159), which the config describes as the point-estimate median-reflected rule, without BY [PENDING FINAL RUN: confirm which selection the retained 10,000-draw bootstrap used]. The older reference datasets (10,000 draws and 1,000 draws, and a 512-draw one; also in the history) record `candidate_fitness_cutoff` −0.283918357340333, the earlier saddle-based threshold. | No retained count, q = 0.99 count, or claim of unchanged conclusions is accepted yet. |
| Current tracked configuration | Pooled SE is the configured primary method (ADR 2026-10-22); the Candidate rule is the Topographer + FDR q = 0.05 rule, exploratory. Tracked bootstrap provenance uses earlier thresholds (−0.253159 without BY, or −0.283918, see the row above). | Label bootstrap results as old-threshold and pending rerun. |

## Rerun checklist

### Documents

- [x] Author Methods, Supplementary Methods and Discussion drafts (this file).
- [ ] Remove editorial instructions from manuscript-facing drafts; keep verification notes here.
- [ ] Reconcile [GLOSSARY.md](../GLOSSARY.md) with log-odds read-share slopes and the [figure inventory](figure-inventory.md) with refreshed outputs. Distinguish branch span, ordinary H0 persistence and the finite global-pair convention from extended persistence; do not infer methods from the configured `extended_h0` identifier.

### Procedure validation

The agreed primary method uses pooled slope Wald SE (`se_pooled`) and the `mixture_quantile` Candidate functional threshold of the `functional-classification` stage (ADR 2026-10-24): the Candidate graph is the Functional class. The audit found `se_replicate` and cutoff −0.283918357340333 still in tracked configuration; existing 10,000-draw provenance uses the old cutoff. The median-reflection estimate, approximately −0.2531594561, is the `median_reflected` comparison rule and remains to be verified as such.

- [ ] Verify original preprocessing filters, sample/replicate labels and source script; validate complete intended coverage and finite integer raw counts. Do not infer upstream filters from fitted tables.
- [ ] Verify grouped-binomial response, sample totals and complements computed before zero replacement, elapsed generations, solver settings and family. Check WT subtraction and compare matched sequences with source fitness using reported absolute/relative discrepancies and justified tolerance, not an unsupported 1e-15 claim.
- [ ] Trace `se_pooled` through observed and flat bootstrap inputs, including WT handling. Keep means WT-relative; document independent perturbations and omitted covariance. Alternative SE models require separate sensitivity analysis.
- [ ] Recalculate T (the mixture quantile) and the Functional classes from the verified growth rates; recalculate median and minimum from the verified full population for the `median_reflected` comparison. Use saddles only as a consistency check. Verify edges keep at least one Functional endpoint (`at_least_one_endpoint_functional_class`; the comparison rule keeps max(s_u,s_v) ≥ T) before largest-component selection, including the non-Functional fringe and no scalar clamping.
- [ ] Run the functional-threshold sensitivity scan after the classification (`just functional-threshold-scan`; ADR 2026-10-24 Decisions 7 and 8). Read `output/analysis/functional_threshold_scan/acceptance.json`: (a) at least 99% of double-stop sequences below T, (b) the 95% resampled interval on T narrower than 2 × 1.96 × the median pooled standard error, (c) one scan step on any axis flips fewer than 5% of the Functional set. Report each check as it came out, a failed check included; do not re-tune the rule. Confirm the scan is current for the classification checkpoint (the loader refuses a stale one), then rebuild `manuscript-quantities` so the `functional_threshold_scan.*` rows replace their gap rows. The scan reports sensitivity and never picks the rule.
- [ ] Verify largest-component tie rule, equal-fitness event order and equal-birth elder tie rule. Confirm direct absorption, global-maximum/root handling and root-group sequence assignments; do not claim every terminal peak clears the band if an elder arm survives by construction.
- [x] Source audit confirms Horton–Strahler order increments when the maximum child order is repeated, including multifurcations.
- [x] Source audit confirms ridge populations use fixed simplified-tree summit-side assignments strictly above the common saddle, then raw fitness ≥ threshold; connectivity is not recomputed.
- [ ] Verify observed and flat distances both use observed nominal diagram D_0, finite global-pair convention and diagonal matching; primary c uses observed draws alone. Keep strict persistence > 2c, q = 0.95, and direct simplification with sequence identities preserved.
- [ ] Keep experimental co-membership outside the primary analysis; avoid claims that merge trees determine all topological invariants or monotone paths. Mention multiple testing only for downstream tests actually used.

### Rerun and Results

- [ ] Invalidate stale caches and rerun observed and zero-mean flat landscapes using agreed SE, cutoff and graph. The configured B = 10,000 and seed 20261019 do not prove which settings a new run used.
- [ ] Record B, seed, random-generation scheme, empirical quantile interpolation, repository revisions, BarSeqFit/Topographer/Python versions, input digests and cache identifiers for both conditions.
- [ ] Rebuild cutoff displays, sequence/edge counts, nominal and retained peak counts, c and τ, diagrams, simplified trees, branch counts/orders, quantile curves, entropy and biological annotations from consistent inputs. Preserve sequence identities and refresh reference artifacts/provenance.
- [ ] Reassess biological and qualitative conclusions empirically. Do not preserve the earlier q = 0.99 count of about 30 without verification. Report retained-feature sensitivity across q; a plateau is not proof of calibration or Monte Carlo convergence.
- [ ] Transfer verified procedure parameters to Methods and numerical outcomes to Results; update Discussion to match the completed analysis.

### Source validation

- [ ] Check [Papkou et al. (2023)](https://doi.org/10.1126/science.adh3860) and the original supplement/code for the published cutoff (−0.507774, rounded −0.508), derivation and stop-codon reference set. Exactly/at-least-two-stop, Gaussian-mixture and 0.999 details remain unverified recollections (the Replication cell of the classification reports the exact rule on our data).
- [ ] Verify counts and identities behind the user's reported double-stop genotypes above the original cutoff that became local peaks. An upper-quantile rule can leave an upper tail; their presence alone does not invalidate the cutoff.
- [ ] Check applicability/calibration assumptions of [Fasy et al. (2014)](https://doi.org/10.1214/14-AOS1252), [preprint 2013](https://arxiv.org/abs/1303.7117), against this parametric construction. Retain approximate, conditional confidence language.

Implementation pointers: [growth rates](../src/dubanevics2026topological/analysis/sequence_growth_rates.py), [graph selection](clamped-tree-comparison-validation.md), [entropy populations](../src/dubanevics2026topological/analysis/codon_entropy_populations.py), [analysis configuration](../config/analysis.toml). These locate checks; they do not certify the rerun.
