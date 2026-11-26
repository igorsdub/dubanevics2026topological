# Raw-fitness threshold comparison validation

This note records the full-data rebuild for issue #16. The threshold selects the
network; every prepared split tree, persistence result, peak call, wild-type
marker, and plotted vertical axis uses raw fitness. A sequence is functional
when raw fitness is greater than or equal to the threshold. Original mutation
edges are retained when at least one endpoint is functional, which retains
functional ties and the below-threshold fringe while excluding fringe-to-fringe
edges. The largest weakly connected component is analyzed, with deterministic
tie-breaking and excluded sequences exported under `output/tables/`.
Clamped fitness is retained only for comparison with the copied reference.

## Full-data result

| Landscape | Threshold | Selected vertices | Selected edges | Raw fitness range | Components before largest-component selection | Excluded vertices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full graph, raw fitness | — | 261,333 | 3,518,696 | -1.174803 to 1.395791684 | 1 | 0 |
| Published-threshold landscape | -0.507774 | 135,178 | 324,044 | -1.174803 to 1.395791684 | 12 | 126,155 |
| Candidate-threshold landscape | -0.283918357340333 | 104,665 | 248,082 | -1.120631058 to 1.395791684 | 13 | 156,668 |

This comparison corroborates only the Published graph; its Candidate row uses the old saddle-based cutoff (-0.283918), not the current median-reflected rule.

> Superseded: the Candidate functional threshold is now computed and the Candidate graph is the Functional class of the classification, see ADR 2026-10-24; the median-reflected rule of ADR 2026-10-23 is the comparison rule.

For the two prepared landscapes, the retained eligible-edge subgraphs contain
135,486 and 105,001 incident vertices before largest-component selection;
these consist of the selected populations plus 308 and 336 vertices in the
smaller components. The reported excluded-vertex totals are relative to the
full input graph. The diagnostics also record 3,194,355 and 3,270,290 excluded
fringe-to-fringe edges for the study and candidate functional thresholds, respectively.

Both prepared minima are below their selection threshold: the study minimum is
-1.17480288668069, and the candidate minimum is -1.1206310580725365. Thus the
analysis and plotted domains were not clipped at either threshold. The wild-type
sequence `GCCGATCTC` has raw fitness 0 in all three landscapes.

| Landscape | Split-tree nodes / branches | Persistence rows | Branch-member sum + branches + root | Reconciled vertices |
| --- | ---: | ---: | ---: | ---: |
| Full graph, raw fitness | 7,815 / 7,814 | 4,055 | 253,518 + 7,814 + 1 | 261,333 |
| Published-threshold landscape | 1,014 / 1,013 | 514 | 134,164 + 1,013 + 1 | 135,178 |
| Candidate-threshold landscape | 329 / 328 | 169 | 104,336 + 328 + 1 | 104,665 |

Each persisted prepared tree declares `scalar_attr = "raw_fitness"`. The full,
study, and candidate raw ranges are respectively `[-1.17480288668069,
1.395791684395264]`, `[-1.17480288668069, 1.395791684395264]`, and
`[-1.1206310580725365, 1.395791684395264]`. Each persistence table includes
one global pair (the global maximum closed at the global minimum, Topographer's `extended` row; ordinary H0 by this project's convention); the remaining rows are ordinary pairs.

The previous clamped-scalar observation was 1,016 nodes / 1,015 branches for
the study landscape and 330 / 329 for the candidate landscape. The raw-fitness
rebuild has 1,014 / 1,013 and 329 / 328, respectively. Raw fitness changes the
tree structure. Although the corrected edge rule also preserves functional
ties, the observed vertex, edge, and peak counts are unchanged for these
inputs. The full raw tree remains 7,815 / 7,814.

## Reference equivalence and provenance

The published-threshold selection was compared with the local Topographer GraphML and
peak references copied from the Topographer
checkout of 2026-10-03, before `topographer v0.5.0`. All 135,178 DNA identities,
324,044 unordered edges, and 514 peak identities match. Raw fitness and the
separate clamped `Fitness` metadata each have maximum absolute error
`5.10702591327572e-15`, below the declared `1e-12` tolerance. The reference
validation is recorded as successful in the rebuilt diagnostics. The
candidate peak count is an observed 169.

The comparison provenance uses schema 2 and records the reference GraphML
SHA-256 `8a83925aec4aeb08eeeeb446e79ffaffad11efe81f4d912dc9be77d189204dad`
and peak CSV SHA-256
`362726e64ce215d2f0dfffaefe43d7a824224de06c4083e937c07aaac57d01a5`. Full
input and analysis-code hashes are recorded in the run provenance. The
pre-existing full-analysis provenance and metadata hashes match the saved
pre-build values, so refreshing the threshold products left the cached full/raw
baseline unchanged. The refresh also regenerated the stable-identity CSV and
Parquet tables for branches, persistence, components, exclusions, and peaks.

## Historical figure and reproducibility

The original comparison validation used independent raw-fitness and branch-count scales and three legends. That presentation is superseded by issues #40 and #42; it does not describe the current exports. Scientific population and selection evidence above remains applicable.

The current canonical outputs are `output/figures/tree_comparison.svg` and `.png`, built with `just comparison-figure`. See [current histogram/comparison validation](fitness-histogram-validation.md) for configured fitness limits, shared histogram counts and branch colors, one shortened legend, and current checks.
