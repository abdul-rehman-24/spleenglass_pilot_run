# Addendum 2 to PREREGISTRATION.md (after commits e092150, 8081e7b and 32a79d3)

Dated 2026-10-29. This addendum fixes how the pre-approved extension of Rule 2 is carried out and how its results will be read. It changes no run that has already been done.

**Disclosure:** all E1/E2 results in `analysis/e1_results.txt` (commit 32a79d3) had been seen when this addendum was written, including the rank of E2. The decisions below were taken knowing them.

## B1. The extension is run, and why

Rule 2 allows one extension (6 more random subsets, selection seeds 1011-1016) "if a deterministic strategy sits in the extreme 2 ranks of 14". It did not say whether E2, which is treated in Rule 3, counts as a deterministic strategy. For CT-FM and SA/V alone the condition was not met (SA/V was extreme in the primary version but not in the single-run version of addendum 1, A2). E2 was rank 1 of 14 in both versions and under both metrics. Because the wording was ambiguous, E2 is counted and the extension is run. This decision was made after seeing the E1/E2 results. The reason is that more random subsets can only make the test of E2 stricter, and the extension is run whichever way the results point.

## B2. Fixed design (6 runs)

Attention U-Net, N = 5, subsets generated with `gen_random(seed)` as before, training seed = selection seed, same commit `485d1dc`, protocol `epoch_baseline_v2`, Tesla T4, same hyper-parameters and validation volumes.

| selection seed | volumes (spleen_ID) | mean volume (mL) |
|---|---|---|
| 1011 | 12, 18, 8, 59, 38 | 343 |
| 1012 | 46, 21, 29, 49, 8 | 261 |
| 1013 | 27, 12, 10, 14, 28 | 295 |
| 1014 | 22, 46, 52, 59, 14 | 303 |
| 1015 | 12, 21, 22, 53, 8 | 298 |
| 1016 | 60, 10, 52, 49, 26 | 249 |

Stopping rule: no further runs are added after these six, whatever the result. Every run is reported; a crashed run is repeated with the same ID and seeds.

## B3. Analysis after the extension

- All analyses of `analysis/e1_analysis.py` are repeated with K = 19 subsets (the 13 of E1 plus these 6; 25 runs). Rule 1 is unchanged (REML primary, ANOVA cross-check, subset-level bootstrap, both metrics).
- The bins of Rule 3 are rescaled to 20 values (19 cloud subsets plus E2): ranks 1-3 = consistent with the corrected hypothesis (single-subset evidence); ranks 4-10 = inconclusive; ranks 11-20 = the size-free boundary measure is not informative. The same bins apply to the single-run version (cloud = the 19 diagonal runs, median of the three ranks). If the two versions fall in different bins the result is inconclusive.
- Rule 2 positions of CT-FM and SA/V are reported again with K = 19.
- The K = 13 results of commit 32a79d3 stay in the paper as the preliminary analysis. If K = 13 and K = 19 disagree, both are reported.

## B4. Multiplicity

Three deterministic subsets (CT-FM, SA/V, E2) are compared with the random cloud. Percentile values p* are reported without adjustment. A Bonferroni threshold (0.05 / 3, about 0.017) cannot be reached with K = 19 (the minimum p* is 0.05). No claim of statistical significance is therefore made for any of them. Statements stay position-based ("consistent with", "not informative").

## B5. Size

E2 has a mean volume of 342 mL, above all 13 random subsets of E1 (144-312 mL). The extension adds subsets up to 343 mL. The exploratory Spearman analysis of Rule 4 is repeated with the 19 subsets. No claim about size is made beyond that analysis.

## B6. Unchanged

Everything else in PREREGISTRATION.md and addendum 1 is unchanged, including the metric rule (best-epoch primary, mean-last-5 sensitivity).
