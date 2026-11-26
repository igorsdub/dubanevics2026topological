# Visual identity refresh validation

## Scientific boundary

The publication component diagram now uses a descending view over the existing
ascending checkpoint. Finite split pairs are ordinary H0 peak births; finite
join pairs are relative H1 valley births. Raw scalar values, finite endpoint
identities, persistence lengths, and source pair IDs are preserved. The global
pair reverses its two endpoints, including their stop flags. Its raw coordinates
are death = -1.17480288668069 and birth = 1.395791684395264.

The global peak is `AAGGAAATG` and has no stop codon. The global valley is
`TAGGGGTGT` and contains a stop codon; this does not make the global peak's
marker red. The annotated diagram tests a stop flag only at the represented
birth extremum, and draws red triangles after the gray base marks.

All 4,054 finite descending peak pairs match the cached full split-tree
persistence pairs by both sequence endpoints and persistence. The diagram has
8,195 finite valley pairs and one global pair. Its subtitle counts all extrema:
4,055 peaks and 8,196 valleys, including the global minimum at the death of the
global pair. The full graph has 261,333 sequences.

Small-network tests compare the view with the public Topographer component API
on negated fitness, converting coordinates back to raw fitness. Both distinct
values and tied finite values are covered. On globally tied values, this view
retains the original checkpoint's chosen extrema and reverses them; it does not
claim to choose the same identities as a fresh negated-fitness computation. The
full-data global extrema are unique, so this distinction does not affect the
publication diagram.

The ascending component analysis module, checkpoint, exported tables, and
threshold diagnostics remain unchanged. The adapter resides in the figure
layer and is included in the figure renderer fingerprint, without invalidating
base scientific analysis.

## Cache evidence

Before editing, SHA-256 hashes were recorded for 284 files under
`output/analysis/` and `output/tables/`. Existing codon-entropy provenance was
already stale against the current comparison checkpoint and build source.
Refreshing that bounded derived analysis reproduced every CSV/Parquet table
byte-for-byte. Its metadata source fingerprint and provenance record were refreshed. The
biological annotation provenance record also refreshed without changing its
scientific tables. Thus 281 of 284 recorded files remain byte-identical; the
three differences are metadata/provenance records. No raw-input, graph, split-tree, component, threshold, or entropy-value
changes were introduced.

## Presentation checks

All 14 publication variants were regenerated as SVG and PNG (28 exports),
then inspected at approximately 180 mm manuscript reproduction width. Headings,
measured counts, threshold text, orange WT stars with left labels, and paired-row
axes were checked across the complete set. The metrics legends sit in the
persistence panels' empty lower-right triangle, avoiding the tree and data marks.
The global component peak is gray at top-left in both variants; valley markers
point down and peak markers point up. Red overlays use represented birth extrema,
including the ordinary split persistence overlays.

Validation completed:

- Full test suite: **110 passed**.
- Ruff checks over source, tests, and notebooks: passed.
- Strict Marimo checks over all notebooks: passed.
- `git diff --check`: passed.
- Complete publication export build: passed; subsequent freshness check confirmed
  all 28 exports current.

Changes remain local. No push or merge was performed. Unrelated pre-existing
untracked scripts were preserved.
