# Addendum 5 to PREREGISTRATION.md (after commits e092150, 8081e7b, 32a79d3, 582437b, 8531f97 and addendum 4)

Dated 2026-11-03. This addendum pre-registers experiment E6: more training seeds on six random subsets, for both models, to measure the training noise and its structure. It changes no run that has already been done.

**Disclosure:** all earlier results had been seen when this addendum was written (Attention U-Net: E1, E2, extension, E4, K = 19 analysis; SAMed: the 60 earlier runs and E5), including the variance split with its wide intervals. E6 was designed because that split could not be identified with one run per subset. The three repeated subsets (selection seeds 42, 123, 7) were fixed by the earlier design. The three additional subsets are chosen by a seeded draw that uses no result: `random.Random(20261102).sample(range(1001, 1017), 3)` gives selection seeds 1003, 1007 and 1011. No E6 result exists yet.

## F1. Questions

- Q1: is the training noise within a subset different from subset to subset (subset-specific noise)?
- Q2: with a better estimate of the training noise, is a subset effect detectable, and how large is the share of variance of a single run and of a 3-run mean (intraclass correlation)?
- Q3: what are the K = 19 variance split and the minimum detectable difference when the noise comes from replicated subsets?

## F2. Fixed design (36 runs)

Attention U-Net and SAMed, N = 5, same commit `485d1dc`, protocol `epoch_baseline_v2`, Tesla T4, same hyper-parameters and validation volumes as all earlier runs (Attention U-Net: batch size 8, 8,717,685 parameters; SAMed: batch size 1, 4,212,016 trainable of 90,884,656). Six subsets, five runs each, per model:

| selection seed | volumes (spleen_ID) | origin | existing runs (training seeds) | new runs (training seeds) |
|---|---|---|---|---|
| 42 | 56, 19, 24, 3, 53 | repeated earlier | 42, 123, 7 | 99, 777 |
| 123 | 56, 28, 18, 27, 20 | repeated earlier | 42, 123, 7 | 99, 777 |
| 7 | 59, 20, 33, 21, 60 | repeated earlier | 42, 123, 7 | 99, 777 |
| 1003 | 14, 46, 38, 59, 26 | seeded draw | 1003 | 42, 123, 7, 99 |
| 1007 | 12, 49, 19, 63, 21 | seeded draw | 1007 | 42, 123, 7, 99 |
| 1011 | 12, 18, 8, 59, 38 | seeded draw | 1011 | 42, 123, 7, 99 |

The existing runs are reused (for subsets 1003, 1007 and 1011 the existing run has training seed = selection seed). New runs per model: 6 + 12 = 18; in total 36. Earlier pilot logs with seeds 99 and 777 (before the seed fix) are not used. Training seeds are nested within subsets. Stopping rule: no further runs are added after these 36, whatever the result; every run is reported and a crashed run is repeated with the same ID and seeds. HD95 is not computed for E6 (secondary, heavy-tailed).

## F3. Analyses

For each model and metric (best-epoch DSC primary, mean of the last 5 epochs as sensitivity; a conclusion is called robust only if it holds for both metrics):
- A1, subset effect: balanced one-way random-effects ANOVA on the six subsets with five runs (F = MSB/MSW, 5 and 24 degrees of freedom, one-sided p-value). Intraclass correlation ICC(1) = (MSB - MSW) / (MSB + 4 MSW) with the exact 95% interval from the F distribution; reliability of a 3-run mean by the Spearman-Brown formula. "Subset effect detected" if p < 0.05; otherwise "not detected", which is not evidence of absence.
- A2, subset-specific noise: standard deviation of the five runs of each subset, and the Brown-Forsythe test of equal variances across the six subsets. "Subset-specific noise supported" if p < 0.05.
- A3, updated variance split: REML random-intercept model with homogeneous training noise on all 19 subsets (the 6 above with five runs, the 13 others with one run), subset-level bootstrap with 5000 resamples, and the branches of Rule 1. This estimate is the one used in the paper; the 25-run estimate of earlier addenda stays as preliminary.
- A4: minimum detectable difference recomputed with the pooled within-subset SD (24 degrees of freedom) and the subset SD of A3.
- A5: descriptive comparison of the five-run means of the six subsets between the two models (n = 6, no test).

Analyses of existing logs, without new runs (descriptive, no decision rules): (i) checkpoint-rule sensitivity of subset scores and strategy ranks (best epoch, mean of the last 1, 3, 5 and 10 epochs) from the per-epoch curves; (ii) subset-level bootstrap intervals for the pair fractions of the single-run comparisons, because the pairs share subsets and are not independent observations.

## F4. Power (for planning)

For the F test with 6 subsets and 5 runs the power at the 5% level is about 0.18, 0.36, 0.54, 0.82 and 0.96 for a true intraclass correlation of 0.1, 0.2, 0.3, 0.5 and 0.7. The six subsets limit the precision of the subset variance; the design mainly improves the training-noise estimate (24 instead of 6 degrees of freedom).

## F5. Unchanged

Everything else in PREREGISTRATION.md and addenda 1 to 4 is unchanged, including the metric rule (best-epoch primary, mean-last-5 sensitivity).
