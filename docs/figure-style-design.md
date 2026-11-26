# Publication figure style

Status: approved shared convention for existing publication figures.

## Agreed decisions

- Publication appearance belongs in `config/figures.toml`. Scientific settings remain in `config/analysis.toml`.
- Design for 180 mm combined manuscript reproduction. The baseline is a 420 × 250 px tree panel beside a 250 × 250 px persistence panel, with a 10 px gap. Use 15 px base type and render PNG exports at 300 dpi.
- Use one `Branch vertex count` legend title, above a horizontal bar placed inside each tree panel, for branch vertex count. Counts use one colour, black, whose opacity rises on a logarithmic scale from count 1 (0.08) to the largest count (1.0; `count_color`, `count_alpha_min`, `count_alpha_max`), with supported labels 2, 10, 100, 1,000, and 10,000 and no separate swatch for count 1 (PR #199; this replaced the plasma palette and gray count-1 rectangle). The linked figure has its count bar inside the tree panel.
- Count the upper branch endpoint and interior original graph vertices, excluding the lower endpoint. The global root is separately owned.
- Derive a distinct branch vertex count; retain the existing interior-member count and mapping semantics.
- Branch vertex counts start at one. Counts use the one-colour logarithmic opacity ramp (previously plasma, with count one in gray); count one is the faintest step, with no separate swatch. Remove the separate zero-count legend. Preserve established metric-specific palettes: amino-acid Hamming distance is a discrete `bluepurple` bar with width-1 bins over the observed distance range (minimum to maximum), styled like the Horton–Strahler bar; Horton–Strahler order is ordinal with four discrete steps of one purple ramp (orders 1–4, lightest to darkest: `#bcbddc`, `#807dba`, `#54278f`, `#2a0a4a`), shared as `HORTON_STRAHLER_COLORS`.
- (Removed on 2026-10-21 with `papkou2023_candidate_grouped_topology`, kept for the record; the glossary entry went with it.) The exploratory grouped split-tree display used a separate grouped branch sequence-count ownership convention. It retains zero-count structural edges, colors them neutral gray outside the positive logarithmic scale, and shows a zero swatch only when one occurs. The ordinary branch-count minimum-one convention above is unchanged.
- (Removed with the same figure.) Grouped split-tree views used the established Altair tree orientation: measured fitness is vertical and increases upward; the global-minimum root is on the right, with visible y ticks but no axis line or grid. Use the configured 420 × 250 px Candidate tree panel and shared type/color settings from `config/figures.toml`. The count-colored view overlays the Candidate tree histogram's horizontal bar inside the tree panel, with the same ticks; the gray-and-gold view has no count legend. Draw no chart-level title; the notebook supplies Markdown headings. Preserve the full displayed grouped fitness extent rather than cropping to the histogram family's fixed domain.
- Export vanilla and annotated versions of the full-graph tree, persistence and histogram figure and a vanilla version of the linked tree–persistence figure, whose annotated variant was removed on 2026-10-21 (the standalone tree and standalone ordinary H0 persistence figures are retired). Vanilla retains the geometry, vertex-count colors, and WT marker on every tree; it omits stop-codon and synonymous annotations.
- Use row headings and subtitles for comparison figures; omit a separate combined title. Use `Full graph`, `Papkou graph`, and `Topographer graph` for the comparison landscapes. The amino-acid Hamming row has the single-line legend title `AA Hamming distance`, a horizontal in-tree colour bar like the count bar. Axis labels remain.
- Annotated figures use a red five-point star for wild type, solid crimson (`#a50f15`) X markers for stop-codon sequences (ADR 0012), and (until the annotated linked tree was removed on 2026-10-21) green outlined squares for synonymous endpoints. Stop-codon markers use size 42 on tree panels and 110 on persistence panels, with a 1.1 px stroke. Omit structural-event markers.
- Show split-tree persistence birth peaks, including the surviving global peak, as triangle-up markers in both variants.
- Color each persistence point by the vertex count of the immediate displayed branch below its birth maximum, rather than its aggregated dying lineage. The global pair, closed at the global minimum, uses the branch below the surviving global maximum, not the entire component.
- Use the full tree's branch-count domain consistently across standalone and linked figures. Do not add a combined title to the linked figure.
- Retain an independent scalar data range for each landscape. Within a landscape, tree and persistence fitness axes share that scalar domain; metric color domains remain independent. Round the observed minimum and maximum outward to 0.1, place ticks at the rounded minimum, zero, and rounded maximum when in range, and extend the displayed domain by 5% of the rounded span on each side. Apply this consistently to tree fitness and persistence axes; do not force separate landscapes onto one range.

## Implementation and verification

- Match births to branches through scientific identities; verify endpoint direction rather than assuming table source/target conventions.
- Check tiny trees: a direct maximum-to-merger branch counts one; three interior vertices plus its upper endpoint count four. Branch counts plus the separately owned root recover the original vertex total for a connected tree.
- Check ordinary and global-pair birth-to-branch assignments and identical color domains. Preserve persistence pair coordinates and scientific analysis settings.
- Read existing cached analysis for rendering changes. Do not rerun expensive scientific analysis solely to restyle figures.
- Inspect SVG and 300 dpi PNG exports at final manuscript width: readable labels, untruncated multiline legend titles and the `1 = gray` note, clear right-side or inset legends, wide tree and square persistence panels, and consistent vanilla versus annotated views.

## Accepted edge case

A one-vertex tree has no branch; its persistence point receives count one and therefore appears gray, as explained by the separate `1 = gray` note. This exception is accepted as part of the specification.

## Approved visual identity refresh

The approved refresh supersedes earlier presentation notes that omitted headings
or described the full-network component diagram's ascending convention. Marker
colours follow ADR 0009 as amended by
ADR 0012: wild type is a
red star and a stop codon a dark crimson (`#a50f15`) cross. Use comparison row headings with larger configurable
heading and subtitle type (18 px and 14 px defaults, controlled by
`rendering.heading_font_size` and `rendering.subtitle_font_size`). Put each
landscape's reference, plotted population counts, and applicable functional
threshold in its subtitle, written `Functional threshold = <value>` with the value
to three decimals. Papkou graph counts include its retained nonfunctional
fringe in the total; functional counts use raw fitness greater than or equal to
the threshold. In the Published and Candidate rows the dashed threshold rule
crosses the tree and its attached histogram at the same fitness (a shared y
scale); only the tree's rule carries the `Functional threshold` label.

For codon entropy, use the full band-simplified Candidate tree and its entropy in
the top row. The bottom row contains cropped KEM ridge/entropy and TCL
ridge/entropy pairs. Include the complete Candidate selected population,
including fringe, and retain raw fitness. Ridge membership follows the
simplified-tree summit and shared-saddle definition in the codon-entropy
contract. Keep full-layout coordinates when cropping; colour every branch by its
branch vertex count on the top tree's plasma scale. Use one identical crop for both
ridges and show both ridge traces in each bottom tree: the focused ridge at full
opacity and every other branch, the opposite ridge and the nearby context, at the
dimmed branch opacity (0.25) with its colour unchanged. Label every branch of the
top tree with its count (`branch_count_label_mode` `"all"`: terminal branches above
their tips, the others under their steps; `"tips"` labels only terminal branches of
at most `branch_count_label_max`, 9, vertices), and each ridge beside its branch in
the excerpts, in peak-grey text two points below the body font. A label that
collides is nudged vertically (`branch_count_label_collision` `"nudge"`, the
default) and one moved off its branch carries a thin peak-grey leader line to it,
or every label is rotated a quarter turn (`"rotate"`); a ridge label moves along its
ridge or over its top when the middle is taken. Include the WT marker in
both at its full-layout coordinates, and give
each ridge panel the same fitness ticks and labels without a y-axis spine. Use
`Band-smoothed candidate graph` as the exact top heading with data-derived Candidate
counts inline; ridge headings include data-derived functional counts without
subtitles. Keep the top tree at 520 × 250 px, bottom trees at 260 × 250 px, and
all entropy panels at 200 × 250 px; align the top pair to the left above the
two bottom pairs. Share entropy thresholds, fitness range, and color scale
across populations. Place its `Entropy` / `(bits)` legend beside the top
heatmap, oriented right, and use `#f5f5f5` for regions outside the scan.

All split-tree wild-type markers use a red star with its WT label to the
left, including every row in metric and vanilla trees. Standalone Fitness trees
show the scalar convention `Fitness (WT = 0)`. Preserve the original ascending
component-persistence diagnostic and threshold evidence; the approved full
network presentation uses the descending convention described in the visual
identity refresh specification. Appearance changes do not redefine cutoff
evidence or scientific populations.

Paired tree–persistence rows have one visible y-axis labeled `Fitness (at birth, WT = 0)` and persistence x-axis `Fitness at death`. Metric legends sit in the empty lower-right triangle of the persistence panel, with a compact bar and a light backing. Dark crimson (`#a50f15`) identifies stop codons, green synonymous annotations, and blue non-significant singleton leaves. See [refresh validation](visual-identity-refresh-validation.md) for the descending endpoint and cache evidence.

The full-graph persistence diagram labels its marker shape in a key (`▲ peak`, `▼ valley`: a filled valley triangle, drawn on the full-graph figures only, GLOSSARY "Figure marker rules") above the stop-codon key, in both variants. Line styles follow one rule: a functional threshold is a dashed rule and a median a dotted one. `full_tree_persistence` draws both that way (`threshold_dash` `[5, 4]`, `median_dash` `[1, 3]`, same colour and width), puts the threshold label above the rule at the diagram's right edge (the left edge runs it through the median rule), keeps the marker key below the rule whatever the data put it at, and bins its marginal histogram in 200 bins like the tree histograms.

In every split-tree persistence diagram, crimson stop overlays test only the birth peak identity. A stop-containing paired saddle does not make the represented peak crimson. Tree stop marks continue to identify the sequences at their own displayed positions.

For the full-graph persistence diagram, the default crimson stop peak is filled at opacity 1; ordinary peaks follow count opacity. Valleys are filled down triangles on the full-graph figures only (restored 2026-11-24; the hollow stop valley and the `STOP_VALLEY_SOLID` switch stay gone): every marker is filled. The tree triangles follow the black logarithmic count-opacity ramp and the tree has one gradient count bar. See ADR 0012.

In every tree panel the stop-codon crosses are drawn over the branches, every branch overlay and the count labels, with the wild-type star last of all: the order is fixed in `compose_tree_layers` (`figures/_charts.py`, see the "Tree layer order" paragraph of the [figure inventory](figure-inventory.md)), not chosen per figure.

## Shared comparison scales and fitness histograms

For the Full/Published/Candidate three-tree comparison, the [histogram and comparison contract](fitness-histogram-design.md) supersedes landscape-specific limits: use the configured [-1.2, 1.4] fitness domain without padding, one common logarithmic branch-count color domain, shared histogram count limits, and one horizontal branch-count colour bar inside each tree panel. The gray singleton exception remains. Histogram panels use touching gray bars without outlines or gridlines; fitness is vertical and vertex count is horizontal from zero. Per-population binning remains independent of shared displayed domains.
