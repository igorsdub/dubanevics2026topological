# Candidate codon-entropy figure contract

The codon-entropy figure is `bootstrap_codon_entropy`. It was first drawn
as `papkou2023_codon_entropy` on the peak-significance simplified tree; that figure
was retired on 2026-10-21 and the contract below now applies to the band-simplified
tree of the bootstrap checkpoint (ADR 0008), with the populations and sweeps computed
in memory by the functions of `just codon-entropy-analysis`.

The figure has two rows. The top row pairs the
full band-simplified Candidate tree with positional amino-acid entropy for the full
Candidate population. The bottom row pairs a cropped KEM ridge with KEM entropy,
then a cropped TCL ridge with TCL entropy.

The full population contains every sequence in the Candidate selected graph,
including retained fringe sequences. Fitness remains raw. KEM and TCL ridge
membership is assigned on the band-simplified Candidate tree using the configured
summit identities and their first shared saddle. Assigned sequences strictly
above the saddle belong to the corresponding ridge; the saddle is excluded.
Ridges are disjoint.

The full-tree layout is computed once. Each bottom tree is a crop of that same
layout, using one shared crop, equal padding on both sides, and the wild-type
marker in view. Both bottom trees show both ridge traces at their original
coordinates. Every branch is coloured by its post-absorption branch vertex count
on the plasma scale of the top tree (the colour bar sits inside the top tree); the
focused ridge, the one the panel is named for, is drawn at opacity 1.0 and every
other branch, the other ridge and the nearby context, at `dimmed_branch_opacity`
(0.25) with its colour unchanged. Every branch of the top tree carries its count
(`branch_count_label_mode` `"all"`; the `"tips"` mode labels terminal branches of at
most `branch_count_label_max`, 9, vertices), and each ridge branch of both ridges
carries its count beside it in the excerpts, whatever its count and the mode, in
peak-grey text; the excerpts take no tree labels, so no branch is labelled twice in a
panel. A top-tree label that collides is nudged vertically and carries a thin leader
line to its branch (`branch_count_label_collision` `"nudge"`; `"rotate"` turns every
label a quarter turn instead); a ridge label goes level with the ridge's middle or,
when that is taken, further along the ridge or over its top. The excerpts' scales are not niced, so a label's pixels
match the branches Vega draws. Both ridge panels retain
the same wild-type marker coordinates as the full tree and use identical
fitness ticks and labels without a y-axis spine. The top tree is 520 × 250 px, bottom trees are
260 × 250 px, and all entropy panels are 200 × 250 px. The top pair is
left-aligned over the two bottom ridge/entropy pairs. The top heading is
`Band-smoothed candidate graph`, with data-derived Candidate counts inline. Bottom
headings identify `KEM ridge` and `TCL ridge` and include data-derived ridge
functional counts. There are no subtitles. The entropy panels show the three amino-acid
positions and positional Shannon entropy in bits; stop codons remain distinct
symbols as configured in the amino-acid alphabet. All three entropy panels
share the same raw-fitness threshold grid, fitness-axis range, and entropy
color scale. Empty selections remain distinct from zero entropy. The legend
title is `Entropy` on the first line and `(bits)` on the second. Place the
legend beside the top entropy heatmap, oriented to the right and close to the
heatmap. Mark fitness regions outside the scan in `#f5f5f5`.

Scientific settings are in `config/analysis.toml` under `[codon_entropy]`;
appearance settings are in `config/figures.toml` under
`[figures.bootstrap_codon_entropy]`. Candidate functional threshold selection comes from the
Candidate landscape configuration. The sequence-keyed
analysis tables (`just codon-entropy-analysis`, built on the peak-significance
simplified tree) and the rendered SVG/300 dpi PNG must be regenerated and validated
against this contract when the implementation is complete. Previously recorded
Published-population counts and entropy measurements describe a superseded
analysis and are not evidence for this Candidate figure.
