# Graph, tree, and persistence figure validation

The stable figure ID is `graph_tree_persistence`. The canonical exports are
`output/figures/graph_tree_persistence.svg` and `.png`; the PNG uses 300 dpi.
The PNG is retained as a repository snapshot, while code and TOML configuration
remain the reproducible source.

## Scientific provenance

The unchanged baseline notebook came from the Topographer repository as of
2026-10-03 (before `topographer v0.5.0`), with SHA256
`7d9ed71f69387a61a43f9e3e2c5988aee0e35bb525650e075e1623bd3d624a57`.
The source location and initial contract are recorded in
[the design document](graph-tree-persistence-design.md).
An independent evaluation of the source notebook matched all 36 sample
coordinates and scalar values to the extracted construction within `1e-14`.
The graph retains its 60 lattice edges. Construction uses public graph,
topology, and tree-layout APIs; graph indices remain internal.

| Stage | Threshold | Components | Entered samples |
| --- | --- | --- | --- |
| (Opening sentinel, no longer composed) | 1 | 0 | 0 |
| 1 First peak appears | 0.99 | 1 | 1 |
| 2 Second peak appears | 0.70 | 2 | 6 |
| 3 Peaks merge | 0.3525253669629021 | 1 | 19 |
| 4 Scan complete | 0 | 1 | 36 |

The unequal peak heights place the diagonal basin separator at the saddle's
`x + y`, approximately 0.9613160094. Rendering derives that separator from the
construction instead of the source's rounded cut at 1, which colored a narrow
part of the higher peak incorrectly. The landscape and sampled graph are
unchanged. Existing peak regions retain their accents after merger; only shared
lower structure receives sand.

## Current contract checks

The figure is a 3 x 4 grid (rows Graph, Tree, Persistence; columns "1 First
peak appears", "2. Second peak appears", "3. Peaks merge", "4. Scan complete").
Automated tests check:

- Persistence events from `ManifoldGraphTreeConstruction.persistence_events`,
  derived from `MergeTree.persistence()` pairs: pair kind (`ordinary` for the
  lower peak, `extended` for the global peak, Topographer's label for the global maximum's pair closed at the global minimum; ordinary H0 by this project's convention), birth and death sample
  identities, and inclusive death at the saddle and at `0`.
- Per-column filled and hollow marker counts and names (ADR 0012): the peak
  appearing at a stage is the only hollow one, drawn in its own lineage colour;
  closed markers in `persistence_dead_color`, alive markers in lineage colors,
  and the filled survivor paired with the global minimum in the last column.
- Every marker outlined in a darker shade of its own fill colour, the threshold
  drawn as a dashed rule on the Tree and Persistence rows, and no tick marks or
  tick numbers on any axis (titles `Fitness`, `Fitness at death`, `Fitness at birth`).
- Tree and Persistence threshold rules and y scales aligned, shared domains,
  and the diagonal spanning the domain.
- The Graph surface honouring configured `graph_surface_color` and
  `graph_surface_opacity`, and the single-stage preview showing three named
  panels.
- Critical markers entering inclusively at their scalar value while the graph
  retains all 36 samples.

No new pixel, viewport, or manuscript-width measurements are recorded for the
3 x 4 layout; regenerate and inspect the export when those are needed.

## Historical evidence (five-stage, three-panel figure)

The sections below describe the superseded five-row Manifold, Graph, and Tree
figure. They are retained as history only and do not establish acceptance of
the current layout.

## Initial implementation evidence

The following evidence was recorded for the initial five-stage rendering and
is retained as historical context; it does not establish visual acceptance of
the current presentation update.

Focused regression tests check marker activation immediately before, at, and
after critical events, the opening sentinel, component counts, complete final
inclusion, retained colors, vertical trunk, partial branches, and the exact
marker/plane alignment without a display-height offset. They also check the
saddle-derived basin separator, composition, native vector marks, notebook
control dependencies, and explicit export guard.

Initial headless Marimo execution completed without errors and left canonical
export hashes and modification times unchanged. The construction cell had no
control dependencies; appearance and preview controls reused it. Regression
tests cover that the export button's inactive path performs no saves or
directory creation.

The initial SVG contained no embedded image elements. Its five rows had a
consistent projection and tree positions, shared headings, and one
lineage/shape key. The 692 × 1388 SVG viewport produced a 2883 × 5783 PNG at
300 dpi. The initial figure was inspected at 180 mm width, including headings,
the merge stage, opening/final stages, plane intersection, and enlarged
regular tree circles. Those visual findings apply to the initial presentation
and are superseded.

## Previous regenerated export

The canonical SVG and 300 dpi PNG were then regenerated from the default scientific
and presentation settings after the presentation update. The SVG has no
embedded image elements. Its viewport is 692 × 1195; the PNG is 2883 × 4979
pixels with 300 dpi metadata. A 680 × 1174 review thumbnail is available at
`/private/tmp/graph_tree_persistence_visual_update_180mm.png`. Review at 180 mm
width confirmed all five headings are readable and unclipped, the manifold
omits regular and future markers, the narrower tree retains clear enlarged
circles, and subtitles and the bottom key are absent. `git diff --check` passed
after the documentation updates.

## Reproduction

```bash
uv run marimo edit notebooks/graph_tree_persistence.py
uv run pytest -q
uv run ruff check src tests notebooks/graph_tree_persistence.py
uv run marimo check --strict notebooks/graph_tree_persistence.py
git diff --check
```

Use the notebook's explicit export button to regenerate canonical outputs.
For headless validation without export:

```bash
uv run marimo export html notebooks/graph_tree_persistence.py \
  --no-include-code -o /private/tmp/graph_tree_persistence.html
```

Scientific settings live in `config/analysis.toml`; projection, palette,
typography, dimensions, and stage text live in `config/figures.toml`.
