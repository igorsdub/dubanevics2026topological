# Monte Carlo error of the 1000-draw parametric bootstrap

Source run: the earlier 1,000-draw run (seed 20261016, `core_threshold` 0.95, `band_quantile` 0.95), now archived at `output/analysis/bootstrap_smoothed_tree_1000draws_seed20261016/`; the figures below are that run's and stay as documented. The current run has 10,000 draws; `docs/bootstrap-10k-comparison.md` reports what changed (c 0.24740, SE of c 0.00152, 42 surviving peaks, 89 cores). Peaks are identified by sequence only.

| Quantity | Value |
| :--- | :--- |
| Real draws (n) | 1000 |
| c (0.95 quantile of real bottleneck distance) | 0.25197 |
| Monte Carlo SE of c | 0.00410 |
| 95% interval for c | [0.24247, 0.25842] |
| 2c | 0.50394 |
| 95% interval for 2c | [0.48495, 0.51685] |
| Nominal peaks with persistence in the 2c interval | 3 of 169 |
| Gap from 2c to nearest persistence above / below | +0.00457 / -0.00138 |
| Borderline pairs (frequency in [0.943, 0.957]) | 17 (11 between two sequences, 6 self-frequencies) |
| Cores affected at threshold 0.943 | 0 existing cores change; 1 new single-sequence core |
| Cores affected at threshold 0.957 | 5 of 90 existing cores change (3 shrink, 2 vanish) |

## Method

The 1000 real-condition bottleneck distances were read from `draw_stats.parquet` and c was recomputed with the definition in `band_width` (numpy linear 0.95 quantile); it matches the stored value 0.2519695556579359 exactly. The Monte Carlo error of c is a nonparametric bootstrap of that quantile (2000 resamples with replacement, seed 20261019; reproduced exactly by the snippet below): the SE is the standard deviation of the resampled quantiles and the interval is their 2.5th and 97.5th percentiles, so it is an approximation, not an exact interval. Cores were re-cut from `pair_frequency.parquet` at 0.943 and 0.957 with Topographer's public `extract_cores` (connected components of sequences with self-frequency at or above the threshold, linked by pairs at or above it); the 0.95 cut reproduces `cores.parquet` exactly (90 cores). The nominal persistence used is ordinary H0 persistence from `hills.parquet`, where a peak survives the band when its persistence exceeds 2c and is absorbed otherwise. (`kept` is only the stored value of the `band_outcome` column that records this survival; the tables below quote it verbatim.)

The Monte Carlo error snippet, with `r` the 1000 real bottleneck distances from `draw_stats.parquet`:

```python
rng = np.random.default_rng(20261019)
qs = [np.quantile(rng.choice(r, 1000, replace=True), 0.95) for _ in range(2000)]
se = np.std(qs, ddof=0)
low, high = np.percentile(qs, [2.5, 97.5])
```

`np.quantile` uses the default linear method, as `band_width` does.

The resample size 1000 in the snippet is the number of real draws of the documented run. For a run with another number of draws n, replace it with `len(r)`, which is identical here: with the 1000 left in place, the 10,000-draw run returns 0.00441, which reproduces the 1,000-draw noise level, while the SE of c with `len(r)` is 0.00152 (`docs/bootstrap-10k-comparison.md`).

## Borderline peaks (persistence in [2lo, 2hi] = [0.48495, 0.51685])

| Peak sequence | Persistence | Core-rule outcome | Band outcome (at the 1,000-draw run's 2c = 0.50394) |
| :--- | :--- | :--- | :--- |
| GTTTTGTAT | 0.50256 | kept | absorbed |
| GTAAGGCGG | 0.50851 | kept | kept |
| TACCGGGTA | 0.51428 | kept | kept |

GTTTTGTAT lies 0.00138 below 2c and GTAAGGCGG 0.00457 above it. GTTTTGTAT would clear the band if c fell below 0.25128, a shift of about 0.17 SE.

## Borderline cores (threshold 0.943 to 0.957)

Cores are keyed by their peak sequence.

- Threshold 0.957 shrinks the cores of CCATCGTTC (5 to 4 sequences), CCTACGCTG (2 to 1) and GAAAGCAGA (4 to 3). Each still meets its own hill, so no outcome changes.
- Threshold 0.957 removes the single-sequence cores of ACTGATACC and TACGACCAA (self-frequencies 0.955). Their hills would have no core, so they would be nominated instead of surviving. Their persistence is 0.04884 and 0.03107, so they would be absorbed unless elder-protected.
- Threshold 0.943 adds a single-sequence core for ACCGATCAA (self-frequency 0.944). That hill, now absorbed with no core, would survive.
- No core splits or merges at either threshold.

## Verdict

Yes, 10,000 draws could change the sets: three peaks sit within the 2c interval (one only 0.0014 from 2c), and up to three hills (ACTGATACC, TACGACCAA, ACCGATCAA) have their survives-or-absorbed outcome decided by a self-frequency within about one binomial SE (0.007) of the 0.95 threshold; no splits are at risk.
