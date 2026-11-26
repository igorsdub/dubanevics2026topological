# Peak uncertainty evidence

The author-confirmed four-panel scope for issue #8 was evaluated against both
full local Parquet inputs on 2026-10-04. Scientific inputs and source hashes
are recorded in `output/analysis/peak_uncertainty/provenance.json`; public
acquisition limitations remain in `data/raw/README.md`.

## Observed populations

| Quantity | Published functional threshold | Candidate functional threshold |
| --- | ---: | ---: |
| Raw fitness threshold | -0.507774 | -0.283918357340333 |
| Largest-component vertices | 135,178 | 104,665 |
| Largest-component edges | 324,044 | 248,082 |
| Raw-fitness local maxima / count at score zero | 514 | 169 |
| Peak-neighbour comparisons | 13,691 | 4,550 |
| Retained maxima at score 0.95 (q at most 0.05) | 176 | 90 |
| Maxima failing that threshold | 338 | 79 |
| Retained maxima at score one | 0 | 0 |
| Measured sequences in each scatter | 261,332 | 261,332 |

> Superseded: the Candidate functional threshold is now computed and the Candidate graph is the Functional class of the classification, see ADR 2026-10-24; the median-reflected rule of ADR 2026-10-23 is the comparison rule.

Maximum DNA identities match the existing threshold-landscape peak tables exactly.
WT remains part of graph analysis and neighbour comparisons; it has no measured
self-comparison row in the WT-relative input and is omitted from the scatter.
The BH family contains all 3,518,696 supplied pairwise comparisons, shared by
both landscapes. Every stored mutation edge changes exactly one nucleotide.
The full raw fitness range is -1.17480288668069 to 1.395791684395264.

There are seven sign disagreements between the fitted pairwise estimate and
the difference of WT-relative vertex estimates. Their stable DNA identities,
both effects, p-values, and q-values are exported in the sign-disagreement
table. Graph adjacency determines which comparisons a maximum requires.

The original issue's draft claim of 82 non-significant maxima out of 169 is
not reproduced: the confirmed procedure yields 79 failures and 90 retained
maxima at q at most 0.05. The author deferred the singleton, biological-summary,
and tree/persistence panels; these counts do not establish those claims.

## Independent scientific checks

- SciPy's independent `false_discovery_control(method="bh")` matches the
  exported q-values for every peak-neighbour comparison (relative tolerance
  1e-12, absolute tolerance 1e-15).
- Supplied pairwise p-values match `2 * normal.sf(abs(s / se))`, supporting the
  recovered two-sided Wald test with directly fitted pairwise errors (relative
  tolerance 1e-7, absolute tolerance 1e-15).
- Per-maximum q-values equal the largest exported neighbour q-value, and
  neighbour counts equal exported contrast counts. Every curve row matches
  its inclusive minimum-score selection and is non-increasing.
- Positive q-values smaller than the binary precision around one retain their
  original q in evidence tables. Their display score is the representable
  value immediately below one, preventing an erroneous survivor at threshold one.

Deterministic tests cover pooled BH correction, real graph neighbours,
sign-conflicting effects, weak-neighbour aggregation, ties, q-value endpoints,
threshold equality and the immediate subsequent drop, invalid/missing
comparisons, scientific cache integrity, appearance-only rendering, missing
export repair, shared plot domains, and actual SVG compilation of multiline
axis titles. The repository suite passes 88 tests. Ruff checks, formatting,
Marimo static validation, and `git diff --check` pass.

The accepted scientific and presentation contract is in the
[design](peak-significance-design.md) and
statistical decision. Canonical rendering,
notebook execution, and reuse verification follow the same source contract.

The appearance revision uses four equal 300 by 300 pixel plotting areas,
shared padded row scales, and y-axis labels and titles only on left panels.
The fitness labels are `Fitness SE` and `Fitness (WT = 0)`; counts use
`# peaks` with ticks from 0 to 500 in steps of 100 and bounds -25.7 to 539.7.
The lower axis displays q-value thresholds from 1 to 0 left to right with no
separate stringency label. Rendered SVG checks confirm this
tick order and axis-title multiplicity. Cached score coordinates, count
rows, and step-event ordering remain unchanged. All 11 scientific cache
files preserve their timestamps and sizes through the appearance rebuild.

The publication PNG uses 300 dpi metadata. The SVG
retains all scatter marks; named datasets avoid embedding each measurement
table twice. The notebook's initial full-data raster preview executes without
errors and stays below Marimo's output-size limit; an explicit appearance
preview button rerenders the plot after controls change. SVG/PNG exports and
their supporting records are reused when source, appearance, and science are
unchanged.

Update 2026-10-21: the figure draws only the two scatter panels. The lower row of
peak-count curves against the q-value threshold is not drawn, together with its
presentation keys (`count_tick_step`, `row_gap`, `q_axis_title`,
`count_axis_title`, `decay_color`, `decay_width`); the retained-peak counts are
still computed, cached and exported unchanged, so the checks above on counts,
step events and q-value ordering still describe the analysis. The scatter panels
keep their 300 by 300 pixel squares and shared scales. The peak triangles are
twice the original linear size (`peak_point_size` 112, four times the original
28 in Altair's area unit) and the legend is two rows, `Peak` above `Not peak`.
