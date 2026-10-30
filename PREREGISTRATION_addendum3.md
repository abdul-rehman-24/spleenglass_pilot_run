# Addendum 3 to PREREGISTRATION.md (after commits e092150, 8081e7b, 32a79d3 and 582437b)

Dated 2026-10-30. This addendum pre-registers experiment E4, a confirmatory test of the hypothesis that E2 generated. It changes no run that has already been done.

**Disclosure:** all results of E1, E2 and the extension (K = 19 subsets), including the rank of E2, the single-run comparisons and the position of E2 outside the range of the random subsets, had been seen when this addendum was written. E4 was designed knowing them. E2 produced the hypothesis, so it is excluded from the confirmatory analysis below. No E4 result exists yet.

## C1. Question

E2 (the five volumes with the highest size-controlled boundary measure) ranked first of 20 among random subsets, but it has a large mean volume (342 mL) and a boundary measure (mean residual log SA/V = 0.083) that lies outside the range of the random subsets (-0.047 to 0.050). E4 separates the two factors:
- E4a: random subsets with the same large volume as E2 but an ordinary boundary measure (does volume alone help?).
- E4b: subsets with a high boundary measure but moderate volume (does the boundary measure help at ordinary volume?).

## C2. Fixed design (12 runs)

Attention U-Net, N = 5, one run per subset, training seed = selection seed, same commit `485d1dc`, protocol `epoch_baseline_v2`, Tesla T4, same hyper-parameters and validation volumes. Boundary measure of a volume: residual of log(SA/V) after regressing on log(volume), fitted on all 32 pool volumes (as in E2). Subset values are means over its five volumes.

Rules (deterministic, shuffle rule as `gen_random`):
- E4a: scan selection seeds 2001, 2002, ... and accept a subset if (i) its mean volume is within 10% of E2's (307.7 to 376.1 mL), (ii) its mean boundary measure lies inside the range of the 19 random subsets (-0.047 to 0.050), (iii) it shares at most 2 volumes with E2, (iv) it is not identical to any subset used before. Keep the first 6.
- E4b: scan selection seeds 3001, 3002, ...; draw 5 volumes with `random.Random(seed).sample` from the 10 volumes with the highest boundary measure (E2 is the top 5 of these); accept if (i) it shares at most 2 volumes with E2, (ii) its mean boundary measure is above the maximum of the 19 random subsets (0.050), (iii) it is not identical to any subset used before. Keep the first 6.

E4a (volume-matched, ordinary boundary measure):

| selection seed | volumes (spleen_ID) | mean volume (mL) | mean boundary measure | shared with E2 |
|---|---|---|---|---|
| 2019 | 12, 45, 31, 10, 59 | 340 | +0.008 | 0 |
| 2023 | 14, 18, 59, 12, 19 | 373 | +0.005 | 1 |
| 2038 | 32, 12, 49, 52, 14 | 347 | +0.021 | 2 |
| 2040 | 12, 47, 10, 46, 19 | 325 | -0.015 | 0 |
| 2047 | 22, 60, 33, 19, 2 | 320 | -0.000 | 1 |
| 2058 | 33, 61, 59, 52, 31 | 308 | +0.003 | 0 |

E4b (high boundary measure, moderate volume):

| selection seed | volumes (spleen_ID) | mean volume (mL) | mean boundary measure | shared with E2 |
|---|---|---|---|---|
| 3002 | 8, 61, 60, 10, 56 | 259 | +0.067 | 2 |
| 3003 | 49, 10, 8, 56, 53 | 252 | +0.058 | 2 |
| 3004 | 10, 24, 8, 56, 60 | 238 | +0.062 | 2 |
| 3007 | 56, 61, 10, 8, 32 | 267 | +0.066 | 2 |
| 3013 | 10, 60, 49, 32, 61 | 271 | +0.074 | 2 |
| 3017 | 61, 60, 49, 53, 56 | 193 | +0.069 | 2 |

Stopping rule: no further runs are added after these twelve, whatever the result. Every run is reported; a crashed run is repeated with the same ID and seeds. The generating script (`analysis/e4_design.py`) will be committed with the results.

## C3. Confirmatory analysis

Outcome per subset: best-epoch DSC of its single run (primary); mean of the last 5 epochs (sensitivity). Reference ("cloud"): the 19 diagonal runs of the random subsets (training seed = selection seed).
- Comparisons of group means with a 95% percentile bootstrap interval (resampling subsets within each group, 5000 resamples) and a two-sided permutation p-value (10000 shuffles): C-1 = E4b minus cloud; C-2 = E4a minus cloud; C-3 = E4b minus E4a.
- Joint regression on the 31 subsets (19 cloud, 6 E4a, 6 E4b): DSC = b0 + b_vol * z(mean volume) + b_res * z(mean boundary measure), ordinary least squares with HC3 standard errors. E2 is excluded; a sensitivity version adds E2 (mean of its three runs) as a 32nd point.

Decision rules (applied to the primary metric; "robust" means the same holds for the sensitivity metric):
- D1, boundary measure supported: C-1 > 0 with the interval excluding 0 AND b_res > 0 with the interval excluding 0. Reported as "consistent with a boundary-complexity effect that is not explained by volume (spleen, Attention U-Net, N = 5)". Robust only if both conditions also hold for the last-5 metric, otherwise fragile.
- D2, volume contributes: C-2 > 0 with the interval excluding 0. It "accounts for the advantage of E2" only if C-2 is at least half of (mean of the three E2 runs minus mean of the cloud).
- D3: anything else is reported as inconclusive. A result whose interval includes 0 is not evidence of absence.
- No result is described as proof or as statistically significant beyond the stated intervals.

## C4. Power (expected, for planning)

The SD of best-epoch DSC across the 19 diagonal cloud runs is 3.39 DSC points. Comparing 6 subsets with 19 gives SE of about 1.6. With a two-sided 5% level and a normal approximation, power is about 0.47 for a true difference of 3 DSC points, 0.81 for 4.5 points and 0.97 for 6 points. A null result with this design therefore cannot exclude effects of 3 points or less.

## C5. Unchanged

Everything else in PREREGISTRATION.md and addenda 1 and 2 is unchanged, including the metric rule (best-epoch primary, mean-last-5 sensitivity).
