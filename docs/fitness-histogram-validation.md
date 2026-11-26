# Fitness histogram and comparison validation

## Scientific populations

The saved Full, Published, and Candidate tree tables recover all selected graph vertices by stable DNA identity, including fringe and noncritical vertices. No scientific graph selection or topology is recomputed for these presentation changes.

| Population | Vertices | Bins | Sum of bin counts | Largest bin count | Observed minimum | Observed maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 261,333 | 200 | 261,333 | 20,209 | -1.17480288668069 | 1.395791684395264 |
| Published | 135,178 | 200 | 135,178 | 9,336 | -1.17480288668069 | 1.395791684395264 |
| Candidate | 104,665 | 200 | 104,665 | 7,048 | -1.1206310580725365 | 1.395791684395264 |

The comparison histogram count domain is [0, 21219.45], adding 5% headroom above the largest bin count. Sparse compact axis labels are 0, 10k, and 20k. Standalone maxima remain population-specific with the same 5% headroom and sparse compact labels. All observed values fit the configured [-1.2, 1.4] display domain. Bins retain each population's observed range and are independent of display settings. Topographer remains pinned to the previously validated merged revision; its public histogram API still defaults to 100 bins and publication callers explicitly request 200.

## Rendering and checks

Inspected all four final named SVG/PNG exports at manuscript scale. Trees and histograms share vertical coordinates, gray bars touch without outlines or gridlines, WT markers remain, and every vertical branch-count ramp is 55 px (superseded: the legend is now a horizontal bar inside the tree, see [the design](fitness-histogram-design.md)). Sparse ramp labels (2, 100, 10,000) preserve readability with the separate gray singleton swatch. Only the bottom comparison histogram has the count axis; its labels read 0, 10k, 20k. Candidate and Published standalone count axes read 0, 5k; Full reads 0, 10k, 20k.

PNG dimensions are Full 3066 × 1254, Published and Candidate 3045 × 1254, and comparison 3483 × 4158; all have 300 dpi metadata. A deliberate Candidate display crop to [-0.2, 0.6] clips tree and histogram marks at the plot frame while preserving all 200 bin boundaries and counts.

Focused tests cover all populations, distinct standalone histogram maxima, unequal compared populations sharing a padded maximum, exact configured fitness limits, cropping/bin invariance, sparse count ticks, touching bars, and the 55 px vertical ramp (superseded by the in-tree horizontal bar). Independent integration review found no blocking defects. Repository lint, formatting for all ten changed Python files, all four notebook checks, and sdist/wheel builds pass. The full publication test suite passes 162 tests. Scientific configuration, checkpoints, Topographer pin, and lockfile are unchanged.

The prior issue #40 delivered Candidate-only 100-bin exports and a tree-only comparison; those appearance claims are superseded by [the revised contract](fitness-histogram-design.md).
