# Publication visual identity refresh

Status: implemented and validated. See [validation evidence](visual-identity-refresh-validation.md).

Requested execution: 2026-10-05 at 02:40 Asia/Tokyo (2026-10-04 17:40 UTC).
Scope: all current publication figures. The user confirmed the final convention
and authorized autonomous implementation using small buckets and lighter agents.

## Settled presentation decisions

- Use the entropy figure's heading/subtitle layout across the publication set.
  Comparison headings are Full graph, Papkou graph, and Topographer graph.
  Subtitles carry the reference, peak counts, functional and total sequence
  counts, and applicable threshold. Counts come from the plotted population.
- Entropy uses Papkou graph in place of the former study tree label. Preserve branch context.
- Biology headings state the computed count and category, for example
  “43 stop-codon containing peaks” or “111 synonymous leaves”.
- Use peaks in place of maxima in reader-facing labels. Preserve scientific
  category definitions and pairing coordinates.
- Tree–persistence rows have one visible shared y-axis labeled
  “Fitness (at birth, WT = 0)”; persistence x reads “Fitness at death”.
  Keep landscape-specific domains and existing scientific scalar values.
- All trees, including metric and vanilla trees, show an orange wild-type star
  with its label on the left. Red is reserved for stop codons, green for
  synonymous annotations, and blue for non-significant singleton leaves.
- Red component-persistence points indicate a stop codon in the represented
  critical extremum: the peak for descending ordinary H0 or the valley for
  descending relative H1.
  A stop codon only in its paired endpoint does not qualify.
- The full-network component persistence figure is the requested gray redesign,
  with downward triangles for valleys and upward triangles for peaks. Export
  vanilla and stop-annotated variants; draw red annotated points above gray.
  Omit grids and axis lines. Headings/subtitles identify the Full graph,
  Papkou et al. (2023), sequence counts, peak counts, and valley counts.
- Remove the component diagram's bottom legend. Treat the surviving global peak
  as a normal upward peak triangle with the same gray/red rule. The requested
  descending sweep places its pair at x = global minimum, y = global maximum.
- Compact metric legends and place them inside a panel, choosing a clear tree
  interior or persistence bottom-right position after visual inspection.
- Enlarge heading/subtitle type as needed for readability at manuscript size;
  appearance remains configurable in config/figures.toml.

## Approved descending convention

- The current component API uses an ascending convention: ordinary H0 births
  at minima, relative H1 births at maxima, global pair minimum to maximum.
  The user requests a descending convention: ordinary H0 peak births,
  relative valley births, global maximum to minimum. Implement a
  scientifically explicit descending result/view rather than relabeling the
  existing ascending table. Preserve the original ascending checkpoint and
  threshold diagnostics; do not redefine threshold evidence through styling.
- Global peak uses the ordinary peak appearance and birth-extremum stop rule.
  No separate global-range legend or color is required.

## Small implementation buckets

The primary agent orchestrates, reviews each diff, and owns final verification.
Delegate bounded tasks to gpt-6-luna with low reasoning effort and limited
context. At most three workers run concurrently. Assign explicit file/function
ownership; serialize edits to the shared figures module and figures TOML.
Do not ask the user to manage handoffs or routine implementation decisions.

1. Baseline audit: inspect current guidance, dirty files, cached products,
   figure exports, and available render commands. Record scientific hashes and
   existing output coverage. Preserve the untracked helper scripts.
2. Shared primitives: configurable headings/subtitles and axis labels, orange
   WT stars and left-aligned placement relative to the star, grid/axis-line
   styling. Review helper behavior before migrating consumers.
3. Comparison and entropy headings: graph labels, reference, actual population
   counts, threshold subtitles, readable type; retain ridge information.
4. Descending component scientific adapter: explicitly derive and validate the
   descending convention from public APIs or verified cached join/split pairs.
   Ordinary peak births and relative valley births must have correct scientific
   categories. Reorient the global pair to maximum -> minimum. Retain original
   ascending products and diagnostics. Verify tie handling, endpoint identities,
   raw scalar coordinates, dimensions, and absolute persistence on a small
   asymmetric network plus full-data evidence. Never merely rename categories
   without validating the descending interpretation.
5. Component appearance: gray upward/downward triangles, gray global peak,
   birth-extremum red overlays above gray, no bottom legend, heading and counts.
6. Metric layout: shared y-axis, WT on every tree, compact inset legends.
   Compare both allowed placements and choose by overlap/readability at 180 mm.
7. Biology and remaining consumers: counted headings, preserved biological
   endpoint definitions, consistent WT, axes, and terminology across all exports.
8. Integration and delivery: inspect every SVG/PNG at manuscript width, run
   focused scientific and plotting checks plus required lint/notebook gates,
   update inventory/style guidance and validation evidence, and summarize the
   local diff and any limitations. Check scientific hashes; only the approved
   descending view may introduce new scientific artifacts. No remote push or
   merge is authorized.

Buckets 3 and 4 can run independently once ownership is disjoint. Buckets 5–7
depend on the corresponding reviewed primitives/scientific adapter. Split a
bucket further when its diff or context becomes too large for a lighter agent.
Workers report exact files changed, validation, and unresolved concerns; the
primary agent independently checks scientific semantics and assembled outputs.

## Execution and verification constraints

Reuse scientific caches and preserve analysis settings, stable identities,
populations, and scientific result values except for the explicitly confirmed
descending component-persistence convention. Update the existing style
guidance and inventory to reflect the confirmed presentation contract. Inspect
all regenerated SVG/PNG exports at manuscript size, checking subtitle wrapping,
legend overlaps, shared axes, marker visibility, and annotation semantics.
Use lighter subagents with low reasoning effort if delegating, followed by
independent review. Preserve concurrent/unrelated files. Git push, merge, and
remote publication have not been requested.
