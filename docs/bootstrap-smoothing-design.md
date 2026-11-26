# Bootstrap peak-core smoothing: implementation design

Status: design for ADR 0008 (accepted), revision 2 (2026-10-17). Noise model revised
by ADR `2026-10-22-pooled-standard-error-for-bootstrap` (2026-10-22): the draws use
`se_pooled`, not `se_replicate`, and the singleton check described below was
removed; passages that still say otherwise are history. Implements the
decision in ADR 0008 (bootstrap branch co-membership smoothing). Terms are
defined in `GLOSSARY.md` (growth rate, selection coefficient, hill, hill
support, peak core, terminal co-occurrence, path to the root, Reeb graph).
This document is the brief for an orchestrated implementation: three
repositories, in order, each piece reviewable on its own. Facts were verified
against the checkouts on 2026-10-16/17; line numbers are indicative.

**Revision 2 replaces revision 1's branch-by-branch co-membership table.**
What changed and why, for a session that already started on revision 1:

- Revision 1 scored each nominal branch as a set and averaged a pair fraction
  per branch pair. That averages the wrong thing: it cannot tell "the top of
  the hill stays together while its flank fluctuates" from "everything
  half-fluctuates", and under pure noise no single partner branch reaches the
  cutoff, so nothing would be absorbed.
- Revision 2 counts, per pair of sequences, how often they sit on the same
  terminal branch of the rebuilt tree, then reads **peak cores** (sets that
  stick together) off that table. The nominal tree is used only afterwards, to
  interpret cores and to draw. A global step 0 (ordinary H0 persistence-diagram
  distances) is added as the overall verdict. It first compared real with flat
  distances; it is now a bootstrap confidence band around the nominal diagram
  (see Global verdict below). A bottleneck comparison with the flat null is kept
  only as a sanity check, not as the verdict.
- Step A (barseqfit `fit_sequences`) is unchanged. Step B's function is
  renamed and its outputs replaced. Step C's stage keeps its name; its outputs
  change.

Measured on the nominal Candidate tree (2026-10-17, from
`output/tables/clamped_candidate_clamped_branch_table.parquet`): 169 hills
hold 574 sequences of the 104,664 in the network; median hill size 1, largest
23, 90 singletons. The brute-force pair table over terminal branches costs
about 5,000 pair updates per draw, so no approximation is needed.

## Purpose

Replace the per-edge q-value rule of the contrast-smoothed tree (ADR 0006) with
a parametric bootstrap on the nominal Candidate split tree: draw plausible
landscapes from each sequence's growth-rate uncertainty, rebuild the split tree
in each draw, record which sequences share a terminal branch, and extract the
sets that stick together (peak cores). A nominal hill with no core is absorbed
into its adjacent saddle with the existing `absorb_terminal_branches`; two
nominal hills sharing one core are one hill. Nominal tree heights never change.

## Order of work and repositories

| Step | Repository | Deliverable | Depends on |
| :--- | :--- | :--- | :--- |
| A | barseqfit (`~/GitHub/barseqfit`) | `fit_sequences`: per-sequence growth rate `r` with pooled and across-replicate SE | nothing |
| B | Topographer (`~/GitHub/topographer`) | `bootstrap_terminal_cooccurrence`: draws, rebuilds, pair table, cores, roles, global distances | nothing (toy networks suffice) |
| C | this repository | stages `sequence_growth_rates` and `bootstrap_smoothed_tree`, config, checks, figures, docs | A and B, pinned via `uv` |

A and B are independent and may run in parallel in separate sessions. C needs
both merged (or at least pushed) and the git pins in `pyproject.toml`/`uv.lock`
bumped. Each step is one feature branch, one PR, with tests passing before the
next step consumes it.

## Statistical definitions (binding for all three steps)

### Measurements

- **Growth rate `r`** of sequence i: the slope per `time_unit` of
  `logit(count / sample_total)` against `time`, fitted from sequence i's own
  rows only, binomial GLM with logit link (`cbind(Nreads, Diff) ~ 1 + time`).
  This is the per-genotype half of the existing `type*time` model: because that
  design is block-separable, `fit_wt`'s `m` equals `r_i - r_wt` exactly at
  convergence. That identity is a required test.
- **Pooled fit**: all samples of the sequence (Papkou: 6 replicates x 2 time
  points = 12 rows, 2 parameters). Gives `r` and the Wald `se_pooled`.
- **Replicate fit**: one fit per `replicate_id` (2 rows, 2 parameters, exact:
  `r_rep = (logit(p_after) - logit(p_before)) / (t_after - t_before)`). Gives
  `r_replicate_mean`, `r_replicate_sd`, and `se_replicate = sd / sqrt(n)`.
- **Which numbers drive what**: tree heights stay the published
  wild-type-relative `s = r_i - r_wt` (unchanged tree); the bootstrap draws use
  `se_pooled` (revised by ADR `2026-10-22-pooled-standard-error-for-bootstrap`;
  it was `se_replicate`). The wild type is drawn around 0 with its own `se_pooled`.
  Because `s` is `r` shifted by one constant, drawing `Normal(s_i, se_i)` is the
  same landscape as `Normal(r_i, se_i)` shifted; no rebuild of the network from
  `r` is needed.
- **Draw**: one landscape, every vertex v of the network replaced by
  `Normal(scalar_v, se_v)`, independently, from `numpy.random.default_rng(seed)`.

### Per draw

- **Terminal branch of a rebuilt tree**: a reduced-tree edge whose upper node
  is a maximum with in-degree 0. Its **members** are the maximum and the
  regular vertices on the edge (`branch_member_table` rows for that edge).
  **The lower endpoint (adjacent saddle or root) is not a member.** Vertices on
  interior (saddle–saddle) branches, saddles, and the root have no terminal
  branch in this draw.
- **Terminal co-occurrence**: for every unordered pair {u, v} of members of the
  same terminal branch, add 1. For every member v, add 1 to
  `terminal_count[v]`. Also add 1 to `pair_count[v, v]` so the diagonal equals
  `terminal_count`.
- **Vertex role**: each vertex is one of `peak` (upper node of a terminal
  branch), `saddle`, `regular`, `root` in the rebuilt tree.
- **Global distance to the nominal tree**: from the rebuilt tree's ordinary H0
  persistence pairs (birth = peak scalar, death = pairing-saddle scalar), with
  the global maximum's pair closed at the global minimum (the root scalar) so
  every pair has a finite birth and death, compute against the nominal ordinary
  H0 diagram:
  `wasserstein_1` (optimal matching, unmatched points go to the diagonal at
  cost = persistence), `bottleneck` (largest matched move), and `peak_count`.
  Also `reproduced_pair_fraction`: share of nominal (peak, death vertex) pairs
  present unchanged in the draw; its denominator counts the global maximum's
  (peak, root) pair together with the ordinary pairs, so it equals the number of
  nominal hills (one pair per peak): 168 ordinary pairs plus the global pair
  is 169 on Papkou.

### Across draws (`n_replicates` draws)

- **Terminal co-occurrence frequency** `f(u, v) = pair_count[u, v] /
  n_replicates`; `f(v, v)` is how often v is on any terminal branch at all.
- **Peak core**: a connected component of the graph on vertices with
  `f(v, v) >= cutoff` and edges `{u, v}` with `f(u, v) >= cutoff`. Cores are
  the sets of sequences that stick together at the top of a hill. A core may
  be one sequence. Cores are disjoint by construction.
- **Hill support** of nominal hill H (its nominal member set): fraction of
  draws in which at least one member of H is a `peak`.
- **Anchored membership** `f(v, p)` for nominal peak p: a row of the pair
  table (how often v stays with p's hill). It was drawn by the colour-fading
  figure `papkou2023_hill_core_membership`, removed on 2026-10-21; no code computes
  it now, and the pair table that holds it is unchanged.
- **Flat null**: the same procedure with every vertex scalar set to one
  constant and the real SEs (same seed, same `n_replicates`). Its cores are the
  chance level for the cores and the false-hill rate. Its distances are kept in
  `draw_stats` and are not used for the verdict: a sum over diagram points
  (Wasserstein-1) is not like for like when the flat draws carry about 44,000
  noise peaks against the real 186, so Wasserstein-1 is never compared across
  conditions. Only the bottleneck distance is compared, as a sanity check (see
  below).
- **Global verdict (bootstrap confidence band)**, after Fasy, Lecci, Rinaldo,
  Wasserman, Balakrishnan and Singh (2014, *The Annals of Statistics*, 42(6),
  2301-2339, https://doi.org/10.1214/14-AOS1252). The nominal diagram `D` is the
  ordinary H0 diagram of the nominal tree with the global maximum's pair closed
  at the root scalar (finite births and deaths; birth = peak scalar, death =
  pairing-saddle scalar), the very diagram Topographer measures every draw
  against; it has one point per nominal hill. Let `c` be the `band_quantile`
  quantile (0.95, numpy's default linear estimator) of the REAL draws'
  `bottleneck` distance to `D`. A point of `D` with persistence (birth - death)
  strictly greater than `2c` is *significant* at that level; a point at or
  below `2c` is not distinguishable from noise. The factor 2 follows from the
  L-infinity ground metric, in which a point matched to the diagonal costs half
  its persistence: a point within bottleneck distance `c` of the diagonal has
  persistence at most `2c`. Reported per hill as `nominal_persistence` and
  `clears_band`, and cross-tabulated by hill outcome (kept, absorbed,
  elder-protected). The verdict also reports `relative_wasserstein_1`, each real
  draw's `wasserstein_1` divided by the nominal diagram's total persistence (L1
  mass, global pair included); it is descriptive and not a test. This is a
  parametric bootstrap around the nominal diagram (draws from each sequence's
  estimated standard error), an approximation to Fasy et al.'s setting, which
  bootstraps the sampling distribution of the diagram.
- **Sanity check (not the verdict)**: the real and the flat draws' `bottleneck`
  distances, both measured against the REAL nominal diagram, side by side. The
  real draws should sit far below the flat ones, and
  `real_share_below_flat_bottleneck_p5` is the share of real draws whose
  bottleneck is strictly below the flat 5th percentile (`flat_bottleneck_p5`,
  linear estimator). It confirms that the real draws are not just noise; it
  decides nothing. The verdict stays the confidence band above (`c`, `2c`,
  the points above `2c`, the cross-tabulation by outcome). The bottleneck is
  the one statistic compared across conditions because it is a maximum over
  matched points, not a sum that grows with the number of peaks; the comparison
  is still only a plausibility check.

### Decisions on the nominal tree

For each nominal hill H, intersect its member set with the cores:

| cores meeting H | outcome |
| :--- | :--- |
| none | **nominated** for absorption (H is not reproducible) |
| one core, lying inside H | **kept**; the core is H's peak as a set |
| one core spanning H and other nominal hills G… | the hill with the highest nominal peak among them is **kept**, the others **nominated** (ties by `vertex_index`, smaller wins, Topographer's elder convention) |
| two or more cores inside H | **kept**; recorded as `split_candidate` (sub-structure the nominal tree merged; no action in milestone one) |

Nominated hills pass through `enforce_elder_rule` (tie-break changed to
`vertex_index`, see C3); the survivors are absorbed with
`absorb_terminal_branches` on the nominal tree, as the contrast stage does.
Outcomes per hill: `kept`, `absorbed`, `elder_protected`. Cores that meet no
nominal hill are **orphan cores**: structure the nominal tree missed; listed,
not acted on.

#### Step 2: the band-simplified tree

The pipeline is a nested sequence of three trees: step 0 is the nominal tree,
step 1 the core rule above, step 2 the **band-simplified tree**. Step 2 keeps
only the nominal hills whose ordinary H0 persistence exceeds `2c` (`clears_band`
true, `band_outcome` `kept`) and absorbs every other hill with one call,
`absorb_terminal_branches(tree, drop)`, `drop` being all the hills with
`clears_band` false. The core rule judges each hill by its own cores; the band
is one simultaneous threshold on the whole diagram, so the two views are
conservative in different ways and are shown side by side, not merged. The
surviving pairs are recomputed from the surviving tree, and the stage fails
closed unless (a) the surviving peaks are exactly the `clears_band` hills, (b)
every surviving pair has its nominal persistence, and (c) the band tree's peaks
are a subset of the core-rule tree's peaks (nested). A hill the band absorbs
that the core rule kept, elder-protected ones included, is listed in
`summary.json` with its step-1 outcome.

- **Deferred, not to be implemented**: effect-size floor, sibling/merge-order
  support, join tree, basins, acting on split candidates or orphan cores,
  consensus trees, interior saddle contraction, flank absorption (nominal
  members of a kept hill with low `f(v, p)` stay on the branch, coloured).

## Step A: barseqfit `fit_sequences`

Package root `src/barseqfit/`.

### A1. Public API

```python
def fit_sequences(
    counts: pl.DataFrame,
    samples: pl.DataFrame,
    sequences: pl.DataFrame,
    *,
    elapsed_time: float,
    time_unit: str,
    engine: str = "vectorised",
    chunk_size: int = 2000,
    pseudocount: str = "all_zeros",
    min_before_reads: int | None = None,
    min_after_reads: int | None = None,
) -> pl.DataFrame
```

Same validation as `fit_wt` (`_validate`, `_as_frame`, required columns
`counts{sequence_id,sample_id,count}`, `samples{sample_id,replicate_id,
condition,time,sample_total}`, `sequences{sequence_id,sequence}`, exactly two
time values, `elapsed_time == max(time) - min(time)`). No `reference_id`, no
`dispersion` argument (the replicate SE replaces the quasi option; `se_pooled`
stays the fixed-dispersion Wald SE so it is comparable with `fit_wt`).

Output, one row per `sequence_id` in `sequences` (all of them, as `fit_wt` does):

| column | dtype | meaning |
| :--- | :--- | :--- |
| `sequence_id` | String | |
| `r` | Float64 | pooled slope per `time_unit` |
| `se_pooled` | Float64 | Wald SE of `r`, fixed dispersion |
| `p_pooled` | Float64 | two-sided z test of `r = 0` (kept for parity; not used downstream) |
| `n_replicates` | Int64 | replicates with both time points present |
| `n_replicates_converged` | Int64 | |
| `r_replicate_mean` | Float64 | mean of per-replicate slopes |
| `r_replicate_sd` | Float64 | sample SD (ddof=1) of per-replicate slopes; null if `n_replicates_converged < 2` |
| `se_replicate` | Float64 | `r_replicate_sd / sqrt(n_replicates_converged)`; null when sd is null |
| `status` | String | `converged`, `not_converged`, `failed` (same semantics as `fit_wt`) |
| `dispersion` | Float64 | pooled-fit Pearson phi (diagnostic) |
| `df_residual` | Int64 | pooled fit |
| `flags` | String | as `fit_wt` |
| `diagnostics` | Struct | as `fit_wt` |
| `elapsed_time` | Float64 | echoed |
| `time_unit` | String | echoed |

Per-replicate slopes are not returned by this function; add
`fit_sequence_replicates(...)` returning `sequence_id, replicate_id, r, se,
status, flags, diagnostics, elapsed_time, time_unit` (mirror of
`fit_replicates`) and have `fit_sequences` aggregate from it, so the
per-replicate values are inspectable.

### A2. Implementation notes

- New design in `_comparison.py` (or a new `_single.py`): columns
  `[ones, time]`; reuse `_logit_fit` / `_logit_fit_batch` from `_fit.py` for
  IRLS. The vectorised engine stacks sequences in chunks exactly as
  `_fit_comparisons_vectorised` does.
- Pseudocount scopes apply as in `_validate` (`_PSEUDOCOUNTS`). A replicate fit
  with a zero count at either time is still fitted after pseudocount; record a
  flag `zero_count_replicate`.
- Status rules: `failed` when the pooled IRLS fails or any required sample is
  missing; `not_converged` when IRLS hit `maxit`; replicate failures do not
  fail the row, they reduce `n_replicates_converged`.
- Export both functions from `__init__.py`; add ADR 0006 (per-sequence growth rates)
  (highest existing is 0005) recording: why a per-genotype fit, the
  block-separability identity with `fit_wt`, and that `se_replicate` is the
  honest SE while `se_pooled` is kept for parity.

### A3. Tests (`tests/test_fit_sequences.py`)

1. **Identity with `fit_wt`**: on `tests/data` (loaded via `tests/_papkou.py`,
   `REFERENCE_ID = "GCCGATCTC"`), for every converged sequence,
   `abs((r_i - r_wt) - m_fit_wt) < 1e-7`, and
   `abs(sqrt(se_pooled_i**2 + se_pooled_wt**2) - se_fit_wt) < 1e-6`.
2. **Exact replicate slope**: for a synthetic sequence with known counts,
   per-replicate `r` equals the closed form above to 1e-12.
3. **Aggregation**: `r_replicate_mean`, `r_replicate_sd`, `se_replicate`
   against `numpy` on the same per-replicate table; null handling when fewer
   than 2 replicates converge.
4. **Reference row**: the wild type gets a real fit (`status == "converged"`,
   `se_pooled > 0`), never a synthetic zero row.
5. **Validation errors** mirror `fit_wt` (missing columns, three time points,
   wrong `elapsed_time`).
6. R harness: not required; note in the ADR that `r/` has no per-genotype fit.

## Step B: Topographer `bootstrap_terminal_cooccurrence`

Module
`src/topographer/statistics/_bootstrap_cooccurrence.py`, exported from
`statistics/__init__.py`. Sits beside `bootstrap_persistence_uncertainty`
(`statistics/_uncertainty.py:649`) and reuses its pieces: `validate_se_attr`
(`:35`), the draw `rng.normal(loc=scalars, scale=se)` (`:575`), and the rebuild
`ScalarNetworkTopology(rep).split_tree()`. If a session already built
revision 1's `bootstrap_branch_comembership`, keep its draw/rebuild loop and
replace the scoring; delete the branch-pair matrix.

### B1. Public API

```python
@dataclass(frozen=True)
class TerminalCooccurrenceResult:
    pair_frequency: pl.DataFrame   # vertex_a, vertex_b (Int64, a <= b), frequency (Float64); diagonal rows included
    vertex_roles: pl.DataFrame     # vertex_index, peak_frequency, saddle_frequency, regular_frequency, root_frequency, terminal_frequency
    cores: pl.DataFrame            # core_id (Int64), vertex_index (Int64); cores at `core_threshold`
    draw_stats: pl.DataFrame       # draw (Int64), peak_count, wasserstein_1, bottleneck, reproduced_pair_fraction
    nominal_hill_support: pl.DataFrame | None  # hill_id, hill_support (only when `nominal_hills` given)
    n_replicates: int
    seed: int | None
    tree_type: str
    core_threshold: float

def bootstrap_terminal_cooccurrence(
    network: ScalarNetwork,
    *,
    se_attr: str = "se",
    tree_type: Literal["split", "join"] = "split",
    n_replicates: int = 1000,
    seed: int | None = None,
    core_threshold: float = 0.95,
    reference: MergeTree | None = None,
    nominal_hills: Mapping[Hashable, Sequence[int]] | None = None,
    n_jobs: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TerminalCooccurrenceResult

def extract_cores(pair_frequency: pl.DataFrame, *, cutoff: float) -> pl.DataFrame
def terminal_branch_members(tree: MergeTree) -> dict[int, list[int]]  # upper vertex_index -> member vertex_indices (lower endpoint excluded)
```

`reference` defaults to the tree of `network`; it supplies the nominal
diagram for `draw_stats` and `reproduced_pair_fraction`. `nominal_hills` maps
a hill id to its nominal member vertex indices; when given, hill support is
computed. Vertex identities are rustworkx node indices of `network.graph`;
the caller maps them to stable ids. `extract_cores` is exported so callers can
re-cut at another cutoff without re-running draws.

### B2. Algorithm

1. Validate SE with `validate_se_attr` (finite, non-negative; zero allowed).
2. Build the reference tree once; take its ordinary H0 persistence pairs, with
   the global maximum's pair closed at the root scalar so every pair has a
   finite birth and death
   (`build_tree_extended_persistence_pair_table` in `topology/_persistence.py`,
   a Topographer function name; its closed global pair is ordinary H0 here;
   Topographer computes the equivalent diagram directly), as the nominal diagram
   and the nominal (peak, death vertex) pairs.
3. Per draw: copy the reference rustworkx graph once outside the loop; per
   draw overwrite each node's `scalar_attr` payload with the drawn value;
   `build_split_tree` + `reduce_merge_tree` + `build_tree_location_tables`
   (what `construct_oriented_merge_tree` does; reuse it if it accepts a raw
   graph, else call the pieces). Then:
   - `terminal_branch_members` on the rebuilt tree; for each branch, members
     sorted, all pairs `(a, b)` with `a <= b` (including `(v, v)`) appended to
     an accumulator keyed by the pair. Use a `dict[tuple[int, int], int]` or a
     `scipy.sparse.coo` accumulation per draw merged into a running sparse
     matrix; both are fine at this size.
   - roles from the reduced tree's node `event`; `terminal_frequency` is the
     diagonal.
   - ordinary H0 persistence pairs of the rebuilt tree (global pair closed at the
     root) → `wasserstein_1`,
     `bottleneck` against the nominal diagram. Implement W1 with
     `scipy.optimize.linear_sum_assignment` on the diagonal-augmented cost
     matrix (n + m points) and bottleneck by bisection over candidate costs
     with a bipartite matching; diagrams have ≤ a few hundred points, so O(n³)
     is fine. `scipy` is already a runtime dependency; `gudhi` is dev-only and
     may be used in tests as the oracle, not at runtime.
   - `reproduced_pair_fraction`: nominal ordinary H0 (peak, death vertex) pairs
     that appear identically; the denominator counts the global maximum's
     (peak, root) pair.
   - hill support increments when `nominal_hills` is given.
4. After all draws divide counts by `n_replicates`; `extract_cores` at
   `core_threshold` (vertices with diagonal ≥ cutoff, edges with frequency ≥
   cutoff, connected components via union-find; `core_id` ordered by the
   highest reference scalar in the core, then smallest vertex index).
5. Seeding: `numpy.random.SeedSequence(seed).spawn(n_replicates)` so draws are
   reproducible independent of `n_jobs`. `n_jobs > 1` uses
   `concurrent.futures.ProcessPoolExecutor` over draw indices; each worker
   returns its sparse pair counts and per-draw stats; the parent merges.
6. Determinism: tree construction already breaks ties by node index
   (`_tree_construction.py:55`); do not add tie noise.

### B3. Tests (`tests/statistics/test_bootstrap_cooccurrence.py`)

Use `tests/conftest.py` fixtures (`bifurcating_two_peaks`,
`single_saddle_diamond`, `linear_path_network`) and `Graph.from_tables(...,
vertex_attribute_cols=("val", "se"))` as in `test_uncertainty.py:217-250`.

1. **Zero SE**: every draw equals the reference; `pair_frequency` is 1 on
   every pair within a nominal terminal branch and absent elsewhere; roles are
   one-hot; cores equal the nominal terminal member sets; `wasserstein_1 ==
   bottleneck == 0`; `reproduced_pair_fraction == 1`.
2. **Saddle exclusion**: on `single_saddle_diamond`, the saddle vertex never
   appears in `pair_frequency` with any peak at zero SE.
3. **Two peaks, one clearly lower**: `bifurcating_two_peaks` with a large SE on
   the lower peak only; the lower peak's diagonal drops below 1 and its pair
   frequency with the higher peak's members rises; the higher peak's core is
   unchanged; `peak_count` takes values 1 and 2 across draws.
4. **Swapping top**: a path network with two adjacent top vertices of equal
   scalar and equal SE; after draws, their pair frequency is ≈ 1 while each
   one's `peak_frequency` is ≈ 0.5; `extract_cores` returns both in one core.
5. **Distances**: `wasserstein_1` and `bottleneck` against `gudhi` (dev extra)
   on random small diagrams, including unequal point counts.
6. **Seed reproducibility** across `n_jobs=1` and `n_jobs=2`.
7. **SE validation** errors (NaN, negative) via `validate_se_attr`.
8. Lint per `justfile` (`ruff format --check`, `ruff check`, line length 99).
9. ADR 0007 (bootstrap terminal co-occurrence) (highest existing is
   0006): why pair co-occurrence on terminal branches with the saddle excluded,
   why cores are read off the pair graph rather than matched to a reference,
   and why distances use `scipy` rather than a new runtime dependency.

## Step C: this repository

Prerequisite: bump the `barseqfit`
and `topographer` git pins in `pyproject.toml` and `uv.lock` to the merged
commits of A and B (now the tags `v0.2.0` and `v0.5.0`).

### C1. Stage `sequence_growth_rates`

Module `src/dubanevics2026topological/analysis/sequence_growth_rates.py`, modelled
on `candidate_tree_contrast.py` (checkpoint lock, pending path, fingerprint,
`provenance.json`). Config section:

```toml
[sequence_growth_rates]
barseq_export_env = "PAPKOU2023_BARSEQ_EXPORT"
elapsed_time = 3.5166197103891137
time_unit = "generations"
identity_tolerance = 1e-6
```

Steps:
1. `build_candidate_tree(settings)` for the nominal tree and `wt`; collect the
   set of all network vertex `dna_sequence` values (the clamped Candidate
   network, not only the tree's critical points) plus the wild type.
2. `load_barseq_export(settings)`; map sequences with the same logic as
   `_map_sequence_ids` (generalise it to a vertex table; identities stay
   `dna_sequence`).
3. `barseqfit.fit_sequences(...)` on those sequences; fail closed if any
   `status != "converged"` or any `se_replicate` is null (list the sequences in
   the error).
4. Identity check against `papkou2023_fitness_wt.parquet`: for every sequence,
   `abs((r_i - r_wt) - s_i) <= identity_tolerance`; record the max gap in
   `diagnostics.json`, raise if exceeded.
5. Write `output/analysis/sequence_growth_rates/growth_rates.parquet` with
   columns `dna_sequence, sequence_id, r, se_pooled, r_replicate_mean,
   r_replicate_sd, se_replicate, n_replicates, status`, plus `diagnostics.json`
   (max identity gap, median `se_replicate / se_pooled`, share of sequences with
   that ratio above 2) and `provenance.json`. Export `growth_rates` to
   `output/tables/` with the usual provenance sidecar.

### C2. Stage `bootstrap_smoothed_tree`

Module `src/dubanevics2026topological/analysis/bootstrap_smoothed_tree.py`. Config:

```toml
[bootstrap_smoothed_tree]
n_replicates = 10000
seed = 20261019
core_threshold = 0.95
n_jobs = 8
```

Steps:
1. Load growth rates (C1) and the nominal tree via `build_candidate_tree`.
2. Build a `ScalarNetwork` whose vertices carry `raw_fitness` (unchanged) and
   `se = se_pooled` (wild type included, its `raw_fitness` is 0). Use
   `Graph.from_tables(vertices, edges, vertex_attribute_cols=("raw_fitness",
   "se"))` from the prepared clamped network's vertex and edge tables.
3. Assert that `ScalarNetworkTopology(network).split_tree()` has the
   same critical vertices and edges as `tree_analysis.tree` (same
   `dna_sequence` at every critical node, same edge set). This is the
   "identical tree" check; it guards the table-rebuild step.
4. Build `nominal_hills` from the nominal tree: for every terminal edge, the
   peak plus its `branch_member_table` rows, as network node indices; keyed by
   the peak `dna_sequence`.
5. `bootstrap_terminal_cooccurrence(network, se_attr="se", n_replicates, seed,
   core_threshold, nominal_hills, n_jobs)`. Then the flat null: the same call on
   a copy of the network with every `raw_fitness` set to 0.0 (same seed).
6. Map every `vertex_index` in the results to `dna_sequence`; never export
   Rustworkx indices as identifiers.
7. Apply the decision table from the definitions section:
   `nominated` → `protected = enforce_elder_rule(tree, nominated)` →
   `absorbed = nominated - protected` →
   `view = absorb_terminal_branches(tree, absorbed)`.
8. Write under `output/analysis/bootstrap_smoothed_tree/`:
   - `pair_frequency.parquet`: `sequence_a, sequence_b, frequency` (sparse,
     `a <= b` by vertex order; diagonal included).
   - `cores.parquet`: `core_id, dna_sequence, raw_fitness, nominal_hill
     (peak sequence or null), is_orphan`.
   - `hills.parquet`: `peak_sequence, member_count, hill_support, core_ids
     (list), core_relation (none|inside|spanning|split_candidate),
     surviving_peak_sequence (for spanning), outcome
     (kept|absorbed|elder_protected), nominal_birth, nominal_death,
     nominal_persistence (Float64, the hill's point of the nominal diagram),
     clears_band (Boolean, `nominal_persistence > 2c`), band_outcome
     (kept|absorbed, `kept` exactly when `clears_band`)`. `flat_hills.parquet`
     has the same columns, with the REAL band.
   - `vertex_roles.parquet`: `dna_sequence, peak_frequency, saddle_frequency,
     regular_frequency, root_frequency, terminal_frequency`.
   - `draw_stats.parquet`: `draw, condition (real|flat), peak_count,
     wasserstein_1, bottleneck, reproduced_pair_fraction`.
   - `summary.json`: counts per outcome and per `core_relation`, orphan core
     count and sizes, peaks before/after, saddles removed, sequences retained,
     settings, fingerprint (hash of C1 provenance + this section + source
     files), and the diagnostics below.
   - `provenance.json`.
9. Diagnostics in `summary.json` (report, do not assert):
   - **Global verdict** (`diagnostics.global_verdict`): `band_quantile`,
     `quantile_method` (`linear`), `c`, `two_c`, `nominal_point_count`,
     `significant_point_count` (persistence strictly above `2c`),
     `significance_by_outcome` (significant and not significant counts per hill
     outcome), `nominal_total_persistence`, `relative_wasserstein_1` (median,
     2.5% and 97.5%) and, for the real draws only, the median and 95% interval
     of `wasserstein_1`, `bottleneck`, `peak_count` under `real`. The block
     `sanity_check` is labelled `role: "sanity check, not the verdict"`: the
     median and 2.5/97.5% interval of `bottleneck` for `real` and for `flat`
     (both against the real nominal diagram), `flat_bottleneck_p5` and
     `real_share_below_flat_bottleneck_p5` (share of real draws strictly below
     it). There is no real-versus-flat Wasserstein-1 comparison. The stage asserts that the nominal diagram has
     one point per hill and that every `reproduced_pair_fraction` is a multiple
     of 1 / `nominal_point_count`, which is Topographer's denominator.
   - **Band simplification** (`band_simplification`, top level of
     `summary.json`): `c`, `two_c`, `rule`, `peaks_before` (nominal),
     `peaks_after_core_rule`, `peaks_after`, `saddles_removed` (from the
     nominal tree), `saddles_removed_core_rule`,
     `saddles_removed_beyond_core_rule`, `nested` (always `true`) and
     `lost_after_core_rule`, the peaks step 1 keeps and step 2 absorbs, each
     with `peak_sequence`, its step-1 `outcome` and `nominal_persistence`.
   - ~~Singleton check~~ (removed 2026-10-22, ADR
     `2026-10-22-pooled-standard-error-for-bootstrap`; the full bootstrap is the
     only verdict).
   - **Flat-null cores**: number and sizes of cores under `flat`; the share of
     nominal hills `kept` under `flat` (false-hill rate).
   - **Consensus peak list**: sequences with `peak_frequency >= 0.5`; how many
     are nominal peaks of kept hills, of absorbed hills, or on no nominal hill.
   - **Float64 ties**: count of exact ties among nominal critical scalars.
10. Table exports to `output/tables/`: `bootstrap_hills`, `bootstrap_cores`,
    `bootstrap_vertex_roles`, `bootstrap_draw_stats`.

### C3. Changes to existing code

- `selected_leaf_simplification.enforce_elder_rule`: tie-break on
  `vertex_index` (the node payload field) instead of `dna_sequence`, matching
  `topographer.topology._persistence._elder_key`. Update its docstring and the
  existing test that pins the tie behaviour. Note in the ADR 0008 consequences
  that this changes nothing on Papkou unless exact float64 ties exist (check
  and record the count).
- `_map_sequence_ids`: generalise to accept any column of sequences, or add a
  sibling helper; keep the duplicate/absent-sequence errors.
- `build/__init__.py`: register the two stages and the figures, following the
  contrast stage wiring (`:293-300`, `:349-356`, `:762-767`).
- `justfile`: recipes `sequence-growth-rates`, `bootstrap-smoothed-tree`
  (depends on the former), `bootstrap-biology-figure` and
  `bootstrap-global-stability-figure` (depend on the latter and
  `comparison-analysis`); all "Not part of `all` or `figures`", like the
  contrast recipes at lines 101-119. (A `hill-core-membership-figure` recipe
  was part of this plan and was removed on 2026-10-21 with its figure.)

### C4. Figures

- `candidate_tree_bootstrap_biology` (the main figure, the band tree is
  the primary result): two rows through the shared row assembler (the
  persistence x-axis title is on the last row only), each a tree beside its
  ordinary H0 persistence diagram on a shared fitness axis. The threshold is
  named τ = 2c (written so once, then τ), the band the 95% simultaneous
  confidence band (parametric bootstrap). Row 1 is the nominal tree: each peak
  is coloured survives (persistence > τ) or absorbed (persistence ≤ τ) on its
  terminal branch and its diagram point, with a counted legend; the diagram
  shades the strip of persistence up to τ; a stop-codon containing peak gets
  a grey × on the tree and a grey ring on the diagram. Row 2 is the
  band-simplified tree (step 2), same encoding, every point clearing τ. The
  core rule is no longer a row of this figure. Titles and subtitles (dataset,
  draws, seed, τ at q = 0.95, peak counts) are read from `summary.json` and
  `hills.parquet` and are templates in `figures.toml`. Config table
  `[figures.candidate_tree_bootstrap_biology]`, `default=False`,
  annotated variant only.
- `papkou2023_bootstrap_core_tree` (removed on 2026-10-21; recover it from git
  history): the core-rule smoothed tree (step 1) with its ordinary H0
  persistence diagram, one row on the same axis pairing, peaks coloured kept or
  elder-protected (the tree repeated the orange on the terminal branch of each
  elder-protected hill), the τ strip shaded, and a subtitle with the core cutoff
  and the nominal, core-rule, absorbed and elder-protected counts. It shared the
  row assembler and the branch and marker helpers of the main figure;
  `just bootstrap-core-tree-figure`, `default=False`, annotated variant only.
- `papkou2023_hill_core_membership` (removed on 2026-10-21; recover it from git
  history): the nominal tree with every sequence on a terminal branch coloured
  by `f(v, p)` for its own nominal peak p (anchored membership, 0 to 1,
  sequential colormap); core members outlined; orphan cores marked.
  Presentation keys in `figures.toml` (size, colormap, outline width, label
  limits). `default=False`.
- `bootstrap_global_stability`: two panels side by side. (a) Two stacked
  histograms of the `bottleneck` distance, each on its own x scale: the real
  draws on top with `c` marked by a labelled rule, the flat draws below, titled
  as a sanity check (not the verdict). The scales are independent because the
  real distances (about 0.1 to 0.3) would fill only a few bins on an axis shared
  with the flat ones (about 0.8 to 1); the condition colours are
  `real_color` and `flat_color` in `config/figures.toml`.
  (b) The nominal ordinary H0 persistence diagram (birth against death in
  fitness units, the global maximum's pair closed at the root so every birth
  and death is finite) with the diagonal and the strip of persistence up to
  `2c` shaded; points take the three outcome colours of
  `candidate_tree_bootstrap_biology`, filled when persistence exceeds
  `2c`, open otherwise. The subtitle states `c`, `2c`, the significant count of
  the nominal points, the median relative Wasserstein-1 and, labelled as a
  sanity check, the share of real draws below the flat bottleneck 5th
  percentile. Read from
  `summary.json`, `hills.parquet` and `draw_stats.parquet`. `default=False`.
- Add all to `docs/figure-inventory.md` (row format at line 16) and the figure
  registry (`figures/registry.py`, `_spec(...)` as at `:444-451`), with
  `_STRICT_PAYLOAD_VARIANTS` entries if the renderer requires them.

### C5. Tests

- `tests/test_sequence_growth_rates.py`: synthetic export as in
  `test_candidate_tree_contrast.py` (`_export`, `TRUE_RATE`), monkeypatched
  `barseqfit.fit_sequences`; identity check passes on consistent data and
  raises on a perturbed `s`; fail-closed on `status != "converged"`.
- `tests/test_bootstrap_smoothed_tree.py`: toy tree via `SimpleNamespace(tree,
  branch_member_table, wild_type_marker_table)` as in
  `test_selected_leaf_simplification.py:12-45`; monkeypatch
  `bootstrap_terminal_cooccurrence` to return hand-built
  `TerminalCooccurrenceResult`s covering each row of the decision table (none,
  inside, spanning with elder choice and `vertex_index` tie, split candidate,
  orphan core); assert outcomes, elder protection, identity mapping to
  `dna_sequence`, and `summary.json` with `allow_nan=False`.
- Figure tests through the registry (`test_figure_registry.py`,
  `_fake_figures.py` pattern).
- Run with `uv run pytest` and `uv run ruff check` / `ruff format --check`
  (line length 88, py312).

### C6. Documentation

- Fill a `## Real-data run` section in this document after the Papkou run:
  hills kept/absorbed/elder-protected, split candidates, orphan cores, peaks
  before/after, global verdict numbers, flat-null
  false-hill rate, consensus-peak agreement, wall time per draw and total.
- Add the figures to `docs/figure-inventory.md`. ADR 0008 is `accepted` and
  ADR 0006 stays `superseded`.

## Real-data run

Run of 2026-10-19 (10,000 draws, seed 20261019) on the Papkou Candidate clamped
network. Numbers come from `output/analysis/bootstrap_smoothed_tree/`
(`summary.json`, `hills.parquet`, `flat_hills.parquet`, `cores.parquet`,
`draw_stats.parquet`, `pair_frequency.parquet`) and
`output/analysis/sequence_growth_rates/diagnostics.json`. `output/` is
disposable generated state, so these values are recorded here to survive a
clean. The earlier 1,000-draw run (seed 20261016, 4 workers) is archived under
`output/analysis/bootstrap_smoothed_tree_1000draws_seed20261016/`; its values
that differ are named "the earlier 1,000-draw run" below, and
a peak-by-peak comparison of the two runs was made when the 10,000-draw run was adopted.

### Setup

| Item | Value |
| :--- | :--- |
| Network | Papkou Candidate clamped network, 104,665 sequences |
| Topographer | tag `v0.5.0` (0.5.0), pinned in `pyproject.toml` `[tool.uv.sources]` (the 10,000-draw run itself used a development merge commit that precedes the release) |
| barseqfit | tag `v0.2.0` (0.2.0), same place (the run itself used a development merge commit that precedes the release) |
| Bootstrap | `n_replicates` 10000, `seed` 20261019, `core_threshold` 0.95, `band_quantile` 0.95, `n_jobs` 8 |

The 2026-10-17 1,000-draw run used the development pull-request heads of
Topographer and of barseqfit. The 10,000-draw run used the development merge
commits on main noted in the table. Their library code is identical to the
pull-request heads: the Topographer tree is identical, and barseqfit adds
only `_env.py` (a validation-script helper the package does not import) and
drops `python-dotenv` from its dev extra. The `summary.json` fingerprints of both
runs (`fingerprint.files`) hold the same `uv.lock` digest and the same
Topographer file digests, which confirms the equivalence for the installed
packages.

Growth-rate stage: the maximum identity gap is 2.2e-8 against a tolerance of
1e-6. The median `se_replicate` / `se_pooled` ratio is 1.57, and 25.1% of
sequences have a ratio above 2. Five sequences have a partial replicate set.
The identical-tree check passed. There are 0 exact ties among tree node
heights, so the `vertex_index` elder tie-break is a no-op on this dataset; the
network has 28 vertices in 14 tied `raw_fitness` groups.

### Outcomes on the real landscape

| Quantity | Value |
| :--- | :--- |
| Nominal hills | 169 |
| Kept / absorbed / elder-protected | 89 / 71 / 9 |
| Peaks before / after | 169 / 98 |
| Saddles removed | 63 |
| Regular vertices reassigned by absorption | 25 |
| Sequences retained | 104,665 (all) |
| `core_relation` inside / none / spanning / split candidate | 89 / 80 / 0 / 0 |
| Spanning and split candidate together | 0 |
| Orphan cores | 0 |
| Cores extending outside their hill | 0 |

The 89 peak cores have sizes 1 (78 cores), 2 (5), 3 (4), 4 (1) and 5 (1).
Every core lies inside one nominal hill, and the 80 hills met by no core are
the `none` hills; none of them is kept. The earlier 1,000-draw run had 90 cores
(79 of size 1) and 79 `none` hills; the difference is TACGACCAA, whose
self-frequency fell from 0.955 to 0.941 against the 0.95 cutoff.

Outcome by hill size (`member_count`):

| Hill size | Kept | Absorbed | Elder-protected |
| :--- | ---: | ---: | ---: |
| 1 (singleton) | 28 | 58 | 4 |
| 2 to 5 | 35 | 12 | 5 |
| above 5 | 26 | 1 | 0 |

`hill_support` by outcome:

| Outcome | Minimum | Median | Maximum |
| :--- | ---: | ---: | ---: |
| Kept | 0.747 | 1.0 | 1.0 |
| Absorbed | 0.234 | 0.550 | 0.999 |
| Elder-protected | 0.488 | 0.712 | 0.979 |

Three absorbed hills have `hill_support` of at least 0.98. Each has a peak
somewhere among its members in almost every draw, yet none is kept:

| Peak `dna_sequence` | `member_count` | `hill_support` | Largest `terminal_frequency` among members (10,000-draw run) |
| :--- | ---: | ---: | ---: |
| CTGGAAATG | 4 | 0.987 | 0.814 (CTGGAGATG) |
| GTCGATACC | 8 | 0.999 | 0.916 (GTAGATACC) |
| TACGATGTT | 4 | 0.982 | 0.852 (TACGATGTT) |

The three `hill_support` values are from the 10,000-draw run (0.991, 1.0 and
0.984 in the earlier run). The largest member of each hill is the same sequence
in both runs; its `terminal_frequency` was 0.809, 0.919 and 0.840 in the earlier
run.

Member sets are the nominal hills as the stage builds them (`build_nominal_tree`
then `nominal_hills` on the rebuilt Candidate tree, joined to
`vertex_roles.parquet`). No member of any of the three reaches the 0.95 cutoff,
so no core forms: the top moves between members from draw to draw, and no set
of sequences stays together on a terminal branch in 95% of draws. The rule
asks for a stable set of sequences, not just a reproduced top, so it absorbs
these hills. This is a known milestone-one limitation, recorded and not acted
on; the decision rule is unchanged.

### Global verdict

The earlier real-versus-flat Wasserstein-1 comparison is withdrawn: the flat
draws carry about 44,000 noise peaks, so summing over diagram points gave 5.4
for real and 2,348 for flat by construction. The verdict is now the bootstrap
confidence band of Fasy et al. (2014); see Statistical definitions. The numbers
below come from the 10,000-draw Papkou run with `band_quantile = 0.95`.

| Quantity | Value |
| :--- | :--- |
| `c` (0.95 quantile of the real `bottleneck`, linear) | 0.2474 |
| `2c` | 0.4948 |
| Nominal diagram points | 169 (one per hill) |
| Significant points (persistence above `2c`) | 42 of 169 |
| Significant / not significant, kept | 42 / 47 |
| Significant / not significant, absorbed | 0 / 71 |
| Significant / not significant, elder-protected | 0 / 9 |
| Nominal total persistence | 50.29 |
| Relative Wasserstein-1, median [2.5%, 97.5%] | 0.108 [0.094, 0.123] |
| Sanity check (not the verdict): bottleneck, real vs flat median [2.5%, 97.5%], flat 5th percentile, share of real draws below it | real 0.158 [0.102, 0.270]; flat 0.907 [0.774, 0.976]; flat 5th percentile 0.803; 100% of real draws below it |

Every point that clears the band is a kept hill, and no absorbed or
elder-protected hill clears it. The 47 kept hills inside the band have
`member_count` 1 (21 hills), 2 to 5 (20) and above 5 (6). Observation, not a
rule change: the 42 kept hills above `2c` are distinguishable from noise at the
95% level by the diagram band, while the other 47 kept hills reproduce as cores
but their persistence lies within `2c`. In the earlier 1,000-draw run `c` was
0.2520, `2c` 0.5039 and 41 points cleared the band (41 kept, 49 kept inside it,
70 absorbed); GTTTTGTAT (persistence 0.50256) lay 0.00138 below `2c` then and
lies 0.00776 above it now. The sanity check is measured against the
real nominal diagram for both conditions and is not the verdict.

Descriptive real-draw statistics (median [2.5%, 97.5%] over 10,000 draws) from the
same run:

| Statistic | Real |
| :--- | :--- |
| Wasserstein-1 | 5.41 [4.74, 6.17] |
| Bottleneck | 0.158 [0.102, 0.270] |
| `peak_count` | 186 [174, 199] |

The nominal tree has 169 peaks.

#### Step 2: the band-simplified tree

The band-simplified tree (`band_simplification` in `summary.json`) keeps the
hills whose ordinary H0 persistence exceeds `2c = 0.4948`: 42 peaks, from 169 in
the nominal tree and 98 after the core rule. It removes 118 saddles from the
nominal tree, 55 more than the core rule's 63, and the stage's checks pass: the
surviving peaks equal the 42 `clears_band` hills, every surviving pair keeps its
nominal persistence, and the band tree is nested in the core-rule tree (`nested`
is `true`). Its node count, `band_tree_node_count` in the same block, is
computed from the band view (82 in the prototype; 84 in the Papkou checkpoint).
The 56 core-rule peaks it loses are 47 kept hills and 9 elder-protected hills;
their nominal persistence runs from 0.012 to 0.483, just under `2c`.
Observation: the core rule and the band agree on the 42 top peaks and differ on
the 56 whose persistence lies within `2c`, so the band is the more conservative
of the two on this landscape. The earlier 1,000-draw run kept 41 peaks, from 99
after the core rule, and lost 58 core-rule peaks (49 kept, 9 elder-protected,
persistence up to 0.503).

Distance conventions: the diagrams are ordinary H0 persistence diagrams
(birth = peak scalar, death = pairing-saddle scalar), with the global maximum's
pair closed at the global minimum (the root scalar), so every pair has a finite
birth and death. Wasserstein-1 uses the L1 ground metric, so a point matched to
the diagonal costs its persistence. Bottleneck uses the L-infinity ground
metric, so a point matched to the diagonal costs half its persistence.
`reproduced_pair_fraction` counts the global maximum's (peak, root) pair
together with the ordinary pairs in its denominator. That denominator is 169,
the number of nominal hills: the nominal diagram has 168 ordinary pairs (one per
non-global peak) plus the one global pair, and every real `reproduced_pair_fraction`
times 169 is a whole number. An earlier note gave the denominator as hills plus 1
(170); that was a misreading and is corrected here. The median real
`reproduced_pair_fraction` is 0.467 (range 0.367 to 0.562).

### Flat null

The flat landscape yields 0 cores. All 169 real hills have `core_relation`
`none`, giving 152 absorbed, 17 elder-protected and 0 kept. The false-hill
rate is 0.0.

### Singleton check

Superseded: result of the `se_replicate` run, kept for the record. The check itself was removed on 2026-10-22 (ADR `2026-10-22-pooled-standard-error-for-bootstrap`).

All 90 singleton hills have an adjacent saddle and were compared. Seven have
`hill_support` above `z_support`; the largest excess is 3.0e-9, which is 0.0055
Monte Carlo SE. The median absolute difference between `hill_support` and
`z_support` is 0.067. (The earlier 1,000-draw run had twelve above, the largest
excess 0.013 or 1.41 Monte Carlo SE, and a median gap of 0.056.) The result is consistent with `hill_support` <=
`z_support` up to Monte Carlo error. The `fit_pair` q-value from the contrast
stage is attached to each row (that stage tested 328 edges, 216 with q above
alpha = 0.05).

### Consensus peaks

132 sequences have `peak_frequency` of at least 0.5. Of these, 89 are peaks of
kept hills, 37 are peaks of absorbed hills, 6 are peaks of elder-protected
hills, 0 are non-peak members of a hill, and 0 lie on no hill. Every kept
hill's peak is therefore a consensus peak. The 37 absorbed peaks are peaks in
at least half the draws but fall short of the 0.95 terminal co-occurrence
needed for a core. (The earlier 1,000-draw run had 128: 90, 33 and 5.)

### Memory and timing

| Item | Real | Flat |
| :--- | ---: | ---: |
| `pair_frequency` rows | 30,816 | 981,708 |
| Largest core | 5 | 0 |
| Wall time | 1798 s (0.18 s/draw) | 8749 s (0.87 s/draw) |

The largest nominal hill has 23 sequences. The stage total was 10,550 s (2.93 h)
with 8 workers on an Apple-silicon laptop (this run). Seconds per draw are wall
time divided by draws, so they mix worker counts and machine load and are not a
clean speed comparison. The archived 1,000-draw run (seed 20261016, 4 workers)
took 283 s (0.28 s/draw) for real and 617 s (0.62 s/draw) for flat, 903 s in
total; earlier 1,000-draw runs took 260 s and 647 s (911 s in total), 214 s and
646 s (863 s), and, under heavier machine load, 0.67 s/draw (real) and 0.84
s/draw (flat), 1521 s.

### Figures

Built with `just`: `candidate_tree_bootstrap_biology` (two rows: nominal
169 peaks of which 42 survive τ and 127 are absorbed, then the band-simplified
tree with 42 peaks), `papkou2023_bootstrap_core_tree` (supplementary: core rule,
98 peaks; removed on 2026-10-21), `bootstrap_global_stability`.
(`papkou2023_hill_core_membership` was also built here and was removed on
2026-10-21.)

## Acceptance criteria

1. Steps A and B merged with their tests and ADRs; pins bumped in C.
2. `just bootstrap-smoothed-tree` runs on Papkou with
   `PAPKOU2023_BARSEQ_EXPORT` set, fails closed on any non-converged sequence,
   and writes every artefact listed in C2, for both `real` and `flat`.
3. The identity check (C1.4) and the identical-tree check (C2.3) pass on Papkou.
4. All three figures build via `just`, and the inventory lists them.
5. `uv run pytest` and ruff pass in all three repositories.
6. This document's Real-data run section is filled and ADR 0008 is `accepted`.

## Open points to decide during implementation (not design changes)

- Wall time of one rebuild on 104k vertices. If above ~3 s, set `n_jobs` to
  the machine's cores and consider a smaller `n_replicates` for development
  only; the committed config is 10,000 draws (1,000 in the earlier run). The flat
  null doubles the run.
- Whether `construct_oriented_merge_tree` accepts a raw rustworkx graph for the
  per-draw rebuild (scout note: `_resolve_scalar_graph` does; untested).
- Float64 exact ties in Papkou `s` values: count them and record in
  `summary.json`; if zero, the C3 tie-break change is a no-op on this dataset.
