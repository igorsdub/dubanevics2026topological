# Study-landscape biological annotation validation

The evidence contract and current presentation decision are recorded in
ADR-0003. The figure reuses
the published-threshold checkpoint at `-0.507774`: 135,178 represented sequences,
1,014 critical points, 1,013 displayed branches, and 514 persistence pairs.
Annotations leave raw fitness, connectivity, persistence pairings, and sequence
identities unchanged.

## Observed biological evidence

| Population | Stop-containing | Adjacent-saddle synonymous | Denominator |
| --- | ---: | ---: | ---: |
| All peaks | 43 | 111 | 514 |
| Singleton peak branches | 17 | 75 | 253 |
| Multi-sequence peak branches | 26 | 36 | 261 |

The upper row highlights 43 stop-containing maxima on the tree and the same 43
birth points in persistence. A stop-containing death endpoint does not receive
a red marker unless its birth maximum contains a stop. Sequences with multiple
stop symbols contribute once. The lower row highlights 111 whole terminal leaf
branches whose birth peak matches its first adjacent descending saddle, and the
same 111 persistence birth points. Fourteen pairs have different adjacent-
saddle and persistence-partner equality flags, so these comparisons remain
separate. A root without an adjacent descending saddle receives no green
highlight.

Amino-acid sequences come from the cached input annotations. The preserved stop
alphabet is `*`, `%`, `^`, and `&`; different stop states are unequal. Counts
and alphabet provenance are saved in
`output/tables/biological_annotation_counts.json`. Branch, pair, and critical
point evidence is saved as CSV and Parquet under `output/tables/` using the
`biological_annotation_` prefix. No internal graph indices are exported.

## Verification and reproduction

Run `just biological-annotation-tables` to export the evidence (the
manuscript quantity registry reads it; no figure build writes it), and
`just study-biology-figure` to verify or reuse the scientific checkpoint and
render the canonical annotated SVG and 300 dpi PNG.
An unchanged second build reuses figure outputs. Missing evidence exports are
repaired; annotation-source edits refresh evidence and figures without
recomputing the landscape. Appearance edits do not invalidate annotation tables.

Saved tables reconcile all 514 birth peaks and adjacent saddles, both equality
flags, and represented vertex accounting. The appearance change does not alter
these scientific products.

The revised appearance uses gray tree structure and persistence points, with
the selected terminal leaf branches and matching persistence births highlighted
above the gray structure. It retains the red wild-type star. Geometry follows
the preceding study metric figure: 420 × 250 px tree panels, 250 × 250 px
persistence panels. Each row uses one raw-fitness domain shared by its tree and
persistence panel, with aligned fitness ticks and the title `Fitness / Birth fitness` at the left.
The persistence-diagram y-axes are omitted, and only the lower persistence
panel displays the `Death fitness` x-axis. The two panel types use equal heights
so their common fitness domains align. Bounds remain outward-rounded and padded
using the publication convention. The counts of 43 stop-containing
maxima/births and 111 adjacent-saddle synonymous terminal branches/births
describe the highlighted populations, not all stop-containing critical points
or all persistence endpoints.

The row headings report the highlighted counts dynamically: **Stop-containing
maxima (n = 43)** and **Synonymous leaves (n = 111)**. The 43 stop-containing
maxima are counted among the 514 peak births; this is not a count of every
stop-containing sequence or critical point. The 111 synonymous leaves are the
terminal peak branches whose birth amino-acid sequence matches the first
adjacent descending saddle.

The revised PNG was inspected at manuscript size for visible red crosses,
green terminal branches, matching persistence triangles, readable axes, and
clipping. Check that each row has the same fitness ticks on the tree and
persistence scales, the tree axes align vertically, neither persistence panel
shows a y-axis, and the `Death fitness` axis appears only below the lower
persistence panel. The chart specification contains 43 stop maxima and matching red
points, and 111 green terminal branches and matching green points, with no
color encodings or count legend. Biological highlights are above gray
structure; the WT star remains above those highlights. Tree stop marker size
is 21, half the prior 42, with unchanged stroke width. Zero-length orthogonal
segments are omitted from the green overlay to avoid PNG stroke warnings.

All 151 non-provenance analysis files and 113 non-provenance table files
retained identical hashes. The normal Make target refreshed two analysis and
two table provenance records because their existing fingerprints were stale;
scientific values and biological populations were unchanged. The repository's
104 tests under `tests/`, Ruff checks, and Marimo checks pass. Executing the notebook via Marimo
HTML export also verifies the cached-data preview and its imports. The
committed 300 dpi PNG is the reviewable snapshot; SVG and evidence remain
disposable generated output.
