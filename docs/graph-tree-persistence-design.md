# Graph, tree, and persistence construction figure

## Purpose and scientific status

The figure explains how a split tree records connected components during a
decreasing superlevel-set sweep. A smooth two-peak landscape is the illustrative
ground truth, a regular sampled graph approximates that landscape, the split
tree summarizes the graph's component events, and a persistence panel records
each peak's birth and death. The smooth landscape itself is not drawn in the
composed figure. This is a conceptual
algorithm illustration, not sequence-fitness data or a scientific result about
sequence space. In particular, the smooth two-dimensional manifold is not a
claim that biological sequence space is a smooth surface.

The stable repository figure ID is `graph_tree_persistence`; it is independent of
manuscript numbering. The existing source is the Topographer Marimo notebook
[`notebooks/manifold_graph_tree.py`](https://github.com/oist/topographer/blob/v0.5.0/notebooks/manifold_graph_tree.py),
as it stood on 2026-10-03, before the `topographer v0.5.0` release; the notebook
in that release has since been revised. The publication repository's initial
copied notebook is byte-identical to the 2026-10-03 source (SHA-256
`7d9ed71f69387a61a43f9e3e2c5988aee0e35bb525650e075e1623bd3d624a57`). Treat it
as provenance and a reference for the construction while the reusable
implementation is developed here.

## Layout and visual encoding

The manuscript figure is a static grid of three rows and four columns. The rows
are labelled **Graph**, **Tree**, and **Persistence**; each column is one stage
of the descending sweep and has a numbered heading. There are no per-stage
subtitles or shared legend. Every column uses the same sampled graph,
projection, and tree layout; only the descending threshold and the resulting
active state change.

All geometric context is grayscale. The three muted lineage accents are:

| Color | Hex | Meaning |
| --- | --- | --- |
| Slate | `#6F7F8F` | First peak's component |
| Plum | `#8B6F8E` | Second peak's component |
| Sand | `#AA9470` | Shared lower region after the components merge |

These colors track the construction's component lineage. They do not encode
wild type, stop codons, or other biological annotations. The two peak regions
keep their individual accents after the saddle joins them; the shared lower
region and tree trunk use sand. Inactive samples and graph context remain
grayscale.

**Graph row.** The projected sampled graph retains all 36 samples at every
stage, drawn over a dimmed surface (`graph_surface_color`,
`graph_surface_opacity`). A critical sample appears with its event marker only
when the descending sweep reaches its scalar value: an upward triangle for a
peak, a diamond for a saddle, and a downward triangle for a valley. Marker
activation uses the scientific scalar value directly. Future critical samples
remain ordinary circles until their event.

**Tree row.** The tree grows continuously to the current threshold, including
partial branches whose parent event has not yet occurred; its shared trunk
stays vertical. Across columns, the tree uses the same y scale and panel
bounds; the tree arms occupy 65% of the former rendered span, centered at 0.5.
A dashed threshold rule marks the current height.

**Persistence row.** Axes are fitness at death (x, titled `Fitness at death`) and
birth (y, titled `Fitness at birth`), on the Tree's y domain, so the dashed
threshold rule sits at the same height in both rows. The coordinates are the
normalised fitness scalar already used as the filtration (`scalar` /
`birth_scalar` / `death_scalar`); the titles name that quantity. No axis has ticks or tick
numbers; the Graph and Tree rows title their axis `Fitness`. A
diagonal spans the domain. Pairs come from `MergeTree.persistence()`: the
elder-rule ordinary H0 pairs plus the global pair (Topographer's `extended` row; ordinary H0 by this project's convention). At threshold `t` a pair is
visible if its birth is at least `t`, and dead if its death is at least `t`
(inclusive); otherwise the peak is alive.

Every marker is filled and outlined in a darker shade of its own fill colour.
The one exception is the peak appearing at a stage, which is hollow in its own
lineage colour: ADR 0012 reserves a hollow marker for a second state, and this
figure's second state is "appears now". A peak appears at the stage where it is
visible at the stage's threshold but was not at the previous stage's.

- Alive peaks are filled markers in their lineage color at death = threshold,
  sliding left as the sweep descends. The first two columns each draw the peak
  they introduce hollow instead.
- A merged lower peak becomes a filled grey marker (`persistence_dead_color`) at
  (saddle, birth).
- In the last column the surviving peak stays a filled grey marker, paired with
  the global minimum (the global pair closed at the global minimum).
- The merge and final columns introduce no peak, so they have no hollow marker.

Branch span, ordinary H0 persistence (global pair closed at the global
minimum by convention), and extended persistence (not computed) stay
distinct quantities; this panel shows birth-death pairs, labelled simply
"Persistence".

## Four displayed stages

Thresholds descend from the first peak to the complete graph. The source
convention reserves threshold `1` as the initial state even though the
landscape's scalar values lie in `[0, 1]`; at that position no sample is
active and the figure no longer composes this opening stage. Event thresholds
include the sample at equality.

| Stage | Heading | Component count |
| --- | --- | ---: |
| 1 | First peak appears | 1 |
| 2 | Second peak appears | 2 |
| 3 | Peaks merge | 1 |
| 4 | Scan complete | 1 |

The first and second peak checkpoints retain their source thresholds (`0.99`
and `0.70`); the merge checkpoint is exactly the sampled saddle scalar; the
final checkpoint is `0`.

## Rendering, configuration, and deliverables

Compose the figure with native Altair charts, using Altair concatenation for
the panels in three rows and four columns. Render the sampled graph and its
surface from a
fixed 3D-to-2D projection matching the source view (elevation 29 degrees,
azimuth -57 degrees). Surface polygons, graph edges, sweep plane, and markers
are vector chart marks; do not embed raster panel images. Reuse Topographer's
public tree-layout API. Keep projection, polygon ordering, and chart extents
deterministic across stages.

Put scientific construction parameters in `config/analysis.toml`. Put palette,
projection, panel dimensions, typography, row and stage copy in
`config/figures.toml`. Reusable construction and rendering code belongs in
`src/dubanevics2026topological/`; the Marimo notebook in `notebooks/` is a thin
interactive entry point. Export the canonical light-background SVG and 300 dpi
PNG to `output/figures/` using the stable figure ID. Opening the notebook must
not write exports. It provides an interactive single-stage threshold preview
and appearance controls using the same renderer as the static manuscript
grid. Animation file export and movable camera controls are outside this
design.

## Original implementation history

The original implementation was a five-stage, three-panel (Manifold, Graph,
Tree) figure delivered in six reviewed buckets. This list records that
history and is superseded by the current 3 x 4 contract specified above.

1. **Record the contract** — this design note and the byte-identical copied
   source notebook. Gate: verify the design against the agreed choices and
   confirm the copied notebook matches its source. Commit:
   `docs: record manifold graph tree figure contract`.
2. **Extract scientific construction** — move deterministic landscape,
   sampled graph, split tree, and sweep state into reusable analysis code;
   use public Topographer graph, topology, and layout APIs. Preserve sample
   coordinates, scalar values, connectivity, and critical events. Confirm the
   five component counts `0, 1, 2, 1, 1`. Commit:
   `refactor: extract conceptual split tree construction`.
3. **Build native projected panels** — render surface, graph, sweep plane,
   markers, and tree as native Altair vector graphics with the fixed
   projection. Gate: inspect intersections, depth ordering, marker timing,
   and ensure SVG has no embedded panel images. Commit:
   `feat: render construction panels with native Altair graphics`.
4. **Compose the manuscript figure** — concatenate panels and stages; add
   stage headings, explanations, column headings, and one shared key. Gate:
   verify alignment and consistent projection, extents, colors, and tree
   positions. Commit: `feat: compose five stage construction figure`.
5. **Integrate the Marimo notebook** — make it call reusable modules; provide
   threshold and appearance controls, explicit SVG/PNG export, and the
   single-stage preview. Gate: opening the notebook causes no exports and
   controls do not rebuild the scientific construction. Commit:
   `feat: integrate Altair construction notebook and exports`.
6. **Verify and document** — review the assembled behavior, update the figure
   inventory, and record validation evidence. Gate: focused event-boundary and
   component-count checks, Ruff, Marimo checks, SVG/PNG inspection, and
   readability at 180 mm manuscript width. Commit:
   `test: validate construction timing and publication exports`.

At every critical event, verify marker state immediately before, at equality,
and immediately after. Verify lineage colors remain distinct after merger,
and all samples appear at the final threshold. If native vector projection
cannot meet the visual gate, return to discussion before changing this design.
