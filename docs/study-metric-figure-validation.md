# Published-threshold metric figure validation

> Naming: this document keeps its original wording. The current display names (Papkou, Topographer, "+ FDR", bootstrapped, branch) and how they map to the internal state names are in [GLOSSARY.md](../GLOSSARY.md).

The manuscript metric figure uses the cached Papkou2023 published-threshold landscape
at raw fitness `-0.507774`. Its saved stable-identity support products contain
1,013 branch records and 514 persistence pairs. The selected graph has 135,178
vertices and 324,044 edges; the source peak reference contains 514 peaks. Every
shown branch and pair has the required branch vertex count, amino-acid Hamming
distance, and Horton–Strahler order.

The figure stacks three rows: branch vertex count, amino-acid endpoint Hamming
distance, and Horton–Strahler order. Each row places the split tree beside its
ordinary H0 persistence diagram (global pair closed at the global minimum). Persistence branch counts are owned by the
immediate branch below each pair's birth. No stop-codon or synonymous-endpoint
overlays are included.

## Presentation contract

Each row's colour bar sits inside its tree panel in the same horizontal style
and position (`colorbar_x_fraction` 0.9), titled `Branch vertex count`,
`AA Hamming distance` and `Horton–Strahler order` above the bar; the
persistence panels carry no legends. Amino-acid endpoint
Hamming distance uses a continuous linear cividis palette over 0–3, with
integer ticks 0–3 (superseded: the bar is now discrete, with width-1 bins over
the observed distance range, and since 2026-10-21 coloured with the Altair/Vega
scheme `bluepurple` instead of cividis). Horton–Strahler order uses the shared
`HORTON_STRAHLER_COLORS` swatches for orders 1–4, light lavender to near-black
purple (`#bcbddc`, `#807dba`, `#54278f`, `#2a0a4a`).

The study landscape uses one scalar domain shared by its tree and persistence
axes across all three metric rows. More generally, retain one scalar domain per
landscape, round its observed minimum and maximum outward to the nearest 0.1,
tick at the rounded minimum, zero, and rounded maximum when in range, and
extend the domain by 5% of the rounded span on each side. Metric color domains
remain independent.

The branch-count legend starts at 2 on a logarithmic plasma scale and shows
supported labels 2, 10, 100, 1,000, and 10,000. A same-width gray rectangle
directly below the scale is labeled 1 for singleton branches/pairs.

The appearance notebook changes only the tree and persistence panel sizes,
shared figure text size, and persistence point size. It saves these values in
`config/figures.toml`; the rendered figure reuses cached science. Both the study
metric figure and the existing threshold comparison were rendered as PNG and SVG;
subsequent builds confirmed both cached figures were current. The 3,837 × 3,970
study PNG and 3,058 × 3,537 comparison PNG were each reduced proportionally to
680 px wide (180 mm at 96 dpi) for manuscript-scale inspection. Labels and
legends remained readable without clipping. The analysis configuration retains
the candidate value `-0.283918357340333`; only its visible comparison label is
rounded to `-0.283918`.

> Superseded: the Candidate functional threshold is now computed and the Candidate graph is the Functional class of the classification, see ADR 2026-10-24; the median-reflected rule of ADR 2026-10-23 is the comparison rule.

## Verification

- `ruff format --check .` reported 35 files already formatted; `ruff check .`
  passed; `pytest -q` passed all 56 tests.
- `marimo check notebooks/study_tree_persistence_metrics.py` passed.
- `uv build` produced the source distribution and wheel. The default uv cache was
  inaccessible under the workspace sandbox, so the successful build used
  `UV_CACHE_DIR=/tmp/uv-cache-topographer-figure6`.
- Repeating `just study-metrics-figure comparison-figure` reported both PNG and
  SVG outputs as current, confirming figure cache reuse.

## Presentation refinement verification

Superseded: this paragraph describes the previous layout, with a vertical
branch-count legend and a magma Hamming palette, and is pending a re-render with
the in-tree horizontal count bar and cividis palette. It was not reviewed
against the new layout.

The earlier refined PNG and SVG were regenerated from the cached analysis. The
PNG is 3,799 × 3,812 pixels; proportional reduction to 680 px wide (about
180 mm at 96 dpi) yielded a height of about 682 px. Visual review at that size
confirmed that the count labels `2` and `1` were distinct and readable, the
singleton gray swatch matched the vertical count bar width and sat directly
below it, and the discrete Hamming magma and Horton–Strahler gray-to-black keys
matched the contract of that layout. The figure had no overall title or
biological overlays, and no labels or legends were clipped.

Tree fitness and persistence birth fitness show identical ticks at `-1.2`,
`0.0`, and `1.4`; both axes have the requested 5% padded domains. Persistence
death fitness uses that same padded domain. These observed ticks correspond to
the outward-rounded data bounds for the current cached landscape.

The full project gates passed after the refinement: all 60 pytest tests,
`ruff check .`, `ruff format --check .` (35 files already formatted), and
`marimo check notebooks/study_tree_persistence_metrics.py`. `uv build`
successfully produced the source distribution and wheel using
`UV_CACHE_DIR=/tmp/uv-cache-topographer-figure6`.

## Shared-style propagation

The approved shared convention was propagated to all ten existing figure
variants, producing 20 SVG and PNG files from cached analyses. All 185
scientific cache and table files retained identical hashes. The regenerated
SVGs have matching tree-fitness and persistence-birth tick labels and absolute
positions in the linked annotated and vanilla figures; the three study-metric
rows also have aligned tree and birth axes. Manuscript-scale contact sheets of
all ten PNG variants showed the requested titles, palettes, count swatches, and
legends without clipping or overlap.

Repository checks passed: all 61 pytest tests, Ruff lint and format checks (35
files already formatted), the metric figure Marimo check, and `uv build` with
`UV_CACHE_DIR=/tmp/uv-cache-topographer-figure6`. A final repeated figure build reported all 20 SVG/PNG files current.
