# Cutoff-landscape peak uncertainty figure

> Naming: this document keeps its original wording. The current display names (Papkou, Topographer, "+ FDR", bootstrapped, branch) and how they map to the internal state names are in [GLOSSARY.md](../GLOSSARY.md).

Status: Accepted after the issue #8 design interview and explicit author
confirmation on 2026-10-04. The implementation follows the narrowed scope below.

## Appearance revision interview

On 2026-10-04 the author accepted four equal 300 by 300 pixel plotting areas,
controlled by one panel-size control in Marimo. Each row shares both axis
scales; only the left panel shows y-axis labels and its title, while both
panels retain x-axis labels. The requested titles are `Fitness SE`,
`Fitness (WT = 0)`, and `# peaks`.
Panel placement uses the plotting areas as its layout bounds so both rows
align by column; a 30 pixel horizontal gap separates adjacent tick labels.

Peak-count ticks are 0, 100, 200, 300, 400, and 500 for the observed baseline
of 514 study maxima. Count bounds retain 5% padding around the actual 0-514
span, giving -25.7 to 539.7. This replaces the irregular quartile ticks.

The author confirmed that the present decreasing curve and scientific
calculation are correct and must be retained. Increasing the minimum
neighbour score removes weakly supported maxima first. The historical
complement instead removes maxima with small q-values first.

The author confirmed a direct q-value display decreasing from 1 to 0 left
to right, titled `q-value cutoff`; the figure carries no separate stringency label.
Ticks are 1, 0.75, 0.5, 0.25, and 0. The existing score coordinates and event
ordering remain unchanged; only their displayed labels and tooltip are
reparameterized to the corresponding q-value cutoff. This preserves the
curve shape and neighbour-support rule without recomputing q-values.
The appearance revision is accepted for implementation.

## Confirmed figure scope

The requested publication figure is a two-by-two composition. The upper row
contains study- and candidate-cutoff scatter panels, horizontally concatenated.
The lower row contains the corresponding decreasing peak-count curves,
horizontally concatenated. Vertically concatenate the two rows. Add a Marimo
appearance notebook and the existing cached-analysis, table, and SVG/PNG build
integration. Tree/persistence, singleton, and biological-summary panels from
the broader issue are outside this implementation scope.

Both scatter panels show all measured sequences except WT. Their x-axis is the
reported WT-relative standard error; their y-axis is raw WT-relative fitness.
Black marks identify raw-fitness local maxima of each selected landscape, and
gray marks identify the remaining measured sequences. Show each landscape's
fitness cutoff as a horizontal line. The two panels share fitness and SE
domains. Fitness uses the existing full-landscape span, outward rounding, and
5% padding convention; preserve 5% span padding on the other shared axes too.

Reuse the approved edge selection, nonfunctional fringe, largest-component,
and raw-fitness contract in ADR-0001. Preserve the existing local-maximum rule:
no neighbour has greater raw fitness, so equality does not disqualify a maximum.
The configured study cutoff is -0.507774; the candidate cutoff is
-0.283918357340333. Current cached observations are 514 and 169 maxima,
respectively. They are observations, not fixed acceptance constants.

> Superseded: the Candidate functional threshold is now computed and the Candidate graph is the Functional class of the classification, see ADR 2026-10-24; the median-reflected rule of ADR 2026-10-23 is the comparison rule.

The lower panels (not drawn since 2026-10-21; the counts are still computed and
exported) show retained maxima as a minimum `1 - q` cutoff increases
from zero to one. For a maximum with complete neighbour comparisons, define
`peak_q = max(neighbour q)` and `minimum_neighbour_score = 1 - peak_q`.
Retain it when `minimum_neighbour_score >= cutoff`, equivalently when every
neighbour q is at most `1 - cutoff`. Thus cutoff zero retains every maximum,
and raising the cutoff produces a non-increasing count. This transformation
does not turn q-values into probabilities that a maximum is real. The count
axis is shared between landscapes. Its unpadded range is zero to the larger
observed baseline peak count, with 5% of that span added at each end. The score
cutoff has unpadded range zero to one and padded range -0.05 to 1.05.

## Recovered source evidence

The historical Topographer notebook is
`notebooks/0-20_papkou2023_peak-significance.py` of the Topographer checkout (`<topographer checkout>`).
It reads supplied pairwise fitness-difference p-values, applies one
Benjamini-Hochberg adjustment across nonzero pairwise comparisons before
landscape selection, and aggregates the largest incoming-neighbour q-value.
Its older decreasing curve removes maxima whose q is below the increasing
cutoff. That curve has a different meaning from the author's confirmed
`1 - q` selection and will not be copied unchanged.

The original pairwise model code is
`lib/fit_model_from_to.v2.R` of the authors' ampseq-nextflow repository (`<ampseq-nextflow checkout>`),
lines 34-52. It fits a binomial GLM with a type-by-time interaction and exports
the interaction estimate, its fitted standard error, and a two-sided Wald
p-value for zero interaction. These pairwise errors are fitted directly, not
computed by combining the two WT-relative standard errors. Local pairwise
p-values numerically match the two-sided normal-tail expression using their
reported estimate and standard error. Input acquisition limitations remain
documented in `data/raw/README.md`.

The historical notebook orients neighbours using the sign of the fitted
pairwise estimate. The current landscapes use WT-relative raw vertex fitness;
these signs disagree for seven local pairwise records. Therefore orientation
by pairwise sign must not silently determine neighbourhood membership.

## Confirmed scientific rules

Use supplied pairwise p-values and one full-data BH family, attaching
comparisons by unordered stable sequence pair to neighbours in the approved
selected graph. Include zero fitted deltas in that family when present; they
do not remove graph neighbours or resolve raw-fitness ties.

Cutoff equality is inclusive and uses full precision. A q-value of one remains
at cutoff zero only; a q-value of zero survives cutoff one. Export each event
and its next representable larger cutoff to distinguish inclusive equality
from the immediate subsequent drop in the displayed step curve. Require a
valid comparison for every graph neighbour; missing or invalid comparisons
stop analysis with an explicit error. For positive q-values so small that
binary subtraction rounds `1 - q` to one, store the preceding representable
score instead. This preserves the inclusive ordering for representable cutoffs
and ensures only an exact q-value of zero survives cutoff one; retain the
original full-precision q-value in the exported evidence. Do not retain untested maxima as
statistically supported. No posterior-probability interpretation is assigned
to the transformed score.

Scientific settings belong in `config/analysis.toml`; presentation settings
belong in `config/figures.toml`. All exported observations preserve DNA
identities, and appearance-only changes reuse cached scientific products.
