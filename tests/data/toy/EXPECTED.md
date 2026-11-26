# Toy landscape expected results

This is the minimal five-sequence, six-nucleotide fixture. It is a star of
single-nucleotide mutations centered on `TAAAAA`; wild type is `TAAACA`, whose
input-loader fitness is synthesized as $s=0$. Fitness values are shifted by
$-0.8$ from the convenient derivation scale. Every pooled standard error is
$0.03$. BY uses the fixed fixture-scale threshold $T=-1.3$ (equivalent to
$-0.5$ before shifting) and target $q=0.05$ over all five rows.

For candidate rows, $p_i=1-\Phi((s_i-T)/se_i)$; below-threshold values are
gated to 1. The BY factor is $c(5)=\sum_{j=1}^{5}1/j=2.283333333333333$. The
functional criterion is $s\ge T$ and adjusted $q_i\le0.05$; other candidates
are uncertain, while below-threshold rows are non-functional.

| DNA | Amino acids | Unshifted $s$ | Shifted $s$ | Stops | Expected class |
|---|---|---:|---:|---:|---|
| TAATAA | %% | -0.600 | -1.400 | 2 | non-functional |
| CAAAAA | QK | -0.470 | -1.270 | 0 | uncertain |
| TAAAAA | %K | 0.050 | -0.750 | 1 | functional |
| TAAAAG | %K | 0.080 | -0.720 | 1 | functional |
| TAAACA | %T | 0.800 | 0 (wild type) | 1 | functional |

The corresponding gated p and BY-adjusted q values are:

| DNA | Gated p | BY q |
|---|---:|---:|
| TAATAA | 1 | 1 |
| CAAAAA | 0.1586552539 | 0.4528285373 |
| TAAAAA | $2.242774\times10^{-75}$ | $8.535002\times10^{-75}$ |
| TAAAAG | $1.408108\times10^{-83}$ | $8.037949\times10^{-83}$ |
| TAAACA | underflow to 0 | underflow to 0 |

All four mutation edges connect a leaf to the central saddle `TAAAAA`:
`TAATAA`, `CAAAAA`, `TAAAAG`, and `TAAACA`. These are the exact induced
Hamming-1 edges among the five fixture sequences; in particular, `TAAAGA`
would also be one mutation from `TAAACA`, while `CAAAAA` is not. The non-functional and uncertain
rows each remain as a fringe vertex because each is adjacent to a functional
vertex. Thus selection preserves all five vertices and four edges. All
vertices have unique fitness values, and the sole saddle is binary.

The split-tree ordinary pairs are `TAAAAG→TAAAAA` (persistence
$0.08-0.05=0.03$) and the global pair `TAAACA→TAATAA`, closed at the global
minimum by convention (persistence $0.8-(-0.6)=1.4$). The global peak's
adjacent saddle is `TAAAAA`, but its ordinary H0 death is the global minimum.
The local peak and saddle both translate to `%K`, so that leaf is synonymous.
The wild-type peak `%T` and adjacent saddle `%K` are non-synonymous. TAA, TAG,
and TGA retain the repository's distinct translations `%`, `^`, and `&`.

## Median-reflected selection and band

On the shifted fixture values the median is $-0.75$ and the minimum is
$-1.4$, giving reflected threshold $2(-0.75)-(-1.4)=-0.1$. The point-estimate
core is the wild type `TAAACA`; fringe retention adds its adjacent saddle
`TAAAAA`. The resulting two-vertex tree has one root-only leaf, with null
adjacent saddle, so it is ineligible for leaf contrast counts.

For the seeded `just toy` run (50 draws, seed 127, band quantile 0.95), the
measured radius is $c=0.07178182986543097$ and $2c=0.14356365973086194$. The
local persistence $0.03$ is absorbed under the strict $p>2c$ rule; the global
pair survives. The output reports one peak after simplification and conserves
all five sequences. Codon entropy is not run because these sequences are only
6 nt.

Stop composition is one zero-stop row, three one-stop rows, one two-stop row,
and zero three-stop rows; the companion check `TAATAGTGA` confirms that three
stop codons are counted. Both nominal hills contain one member, so ranking by
member count and then ascending peak DNA yields `TAAAAG`, `TAAACA`.

The five rows are minimal for this set of properties: two peak identities are
needed for synonymous and non-synonymous leaves; they require a binary saddle;
the fourth row supplies a distinct global minimum; and the separate
uncertain fringe row supplies the requested uncertainty class.