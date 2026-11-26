# Full-network component persistence: implementation buckets

> Update 2026-11-25: the dataset pin system, the cluster (HPC) workflow and their recipes (`hpc-*`, `bootstrap-restore`, `config/runs.toml`, `data/reference/`, the Bucket) left this repository. `data/processed/` is now the one folder of results (`just all`, `just fetch-data`, `just figures`). Mentions of the removed parts below are historical.

Status: implementation buckets A–C complete; author approved the final visual
revision and authorized commit, push, merge, branch deletion, and fetch.

## Agreed figure contract

Scope was `network_component_persistence`, annotated and vanilla variants.
That figure id is retired: its diagram is now the middle panel of
`full_tree_persistence`, which keeps this contract and adds the
split tree and vertex-count histogram on a shared fitness axis.

- Plot measured raw fitness at death on x and fitness at birth on y. The middle
  y axis is hidden because the fitness axis is shared with the tree. Show the
  birth = death diagonal.
- Retain peak and valley finite pairs. Show the global extrema relationship in
  both directions: maximum birth/minimum death and minimum birth/maximum death.
  These are two display representations of one scientific global pair; neither
  the cached pair count nor the extrema counts may be inflated.
- Place a compact gray count histogram above the scatter panel, sharing its
  fitness x-domain and plot width. Include every full-graph sequence once,
  including stop-containing sequences. Start with counts; density is deferred
  pending visual review.
- Calculate the median over that same full population. Draw a vertical dotted
  guide behind the marks in both panels and label it `Median fitness` above the
  histogram. Add a tick at the exact median position on the death-fitness axis,
  formatted to one decimal place (−0.7 for the full dataset). Show the fitness
  x labels/title only below the scatter panel. Use 100 histogram bins (200 since
  2026-10-22, matching the tree histograms; `histogram_bins` in
  `config/figures.toml`) and 14 pt median and color-key labels, as agreed in
  author review.
- Red marks identify a represented birth maximum/minimum whose sequence contains
  at least one stop codon. A stop at only the paired death endpoint is insufficient.
  Annotated exports have a bottom-right key: red means stop-containing sequence;
  gray means no stop codon. Vanilla exports have no biological key or red marks.
- Keep study, extrema counts, and total sequence count in the subtitle; remove
  its biological color explanation.

## Bucket A: full-population data

Deliver a validated stable-identity table with `dna_sequence` and `raw_fitness`,
loaded from the declared full dataset without recomputing persistence. Wire it
into CLI and notebook component figure inputs as `full_fitness`. Verify uniqueness,
finite values, full population coverage, and inclusion of noncritical sequences.
Do not change scientific pairing or cutoff diagnostics.

## Bucket B: figure composition

Accept `full_fitness` as a required keyword input to the component renderer.
Add the mirrored global display point, histogram, median guides, diagonal,
ticks, revised title, and annotated key. Keep appearance choices in the existing
figure-specific TOML section. Update focused figure tests, including stop status
of the mirrored minimum, both variants, median population, and shared domains.
Own the renderer and figure tests; Bucket A owns its callers.

## Bucket C: integration and review

The orchestrator reviews agent changes, integrates the renderer call, updates
inventory/glossary, runs focused checks, and renders both full-data variants.
An independent agent then reviews population/pair semantics and chart output.
Record validation evidence and deliver inspectable previews for author review.
Author approval was received before commit, push, or merge. Ask the author if an unresolved scientific
decision or material scope change is required.

## Validation evidence (2026-10-05)

- Luna agents completed the population input and figure composition buckets;
  a separate Luna reviewer found no scientific or visual blockers.
- Full population: 261,333 unique sequence identities, including WT. Median raw
  fitness is -0.7139811713950942; observed bounds are -1.17480288668069 to
  1.395791684395264. The public Papkou vertex-table builder is the same source
  used to construct the full graph; no topology recomputation is needed.
- Both histogram totals equal 261,333. Each variant contains 12,251 display
  points, including one mirrored global point, while the scientific table
  remains at 12,250 pairs. Display headings report 4,055 peaks and 8,196 valleys.
- The cached pair Parquet checksum is unchanged:
  `2a0eee218204a3370f40db30e7f42dee8e55bb02dba067e4bf5a6b719ebc3c8f`.
- Full test suite: 121 passed. Ruff lint/format, Marimo notebook check, TOML
  parsing, and `git diff --check` passed.
- Both full-data PNG and SVG variants were regenerated through the figure CLI.
  Visual inspection confirmed aligned fitness domains, histogram columns from
  zero, visible ticks/diagonal, median label, and a key clear of the mirrored
  minimum. The initial floating-bar and key-overlap issues were corrected.

Review artifacts are the former `network_component_persistence.png` and
`network_component_persistence_vanilla.png` figure outputs, with SVG
counterparts. Only these figure outputs were regenerated. Author review was completed and Git publication authorized.

## Author visual revision (2026-10-05)

Added the exact median as a death-axis tick labelled −0.7, increased the
median and color-key text to 14 pt, and narrowed histogram bins by using
100 bins. Shared x-scale remains explicit. Both PNG/SVG variants regenerated
and the annotated export visually inspected. Figure tests (23) and Ruff
lint/format passed. Author approved these revisions and authorized Git publication.
