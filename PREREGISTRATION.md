# SpleenGLASS: pre-registration of experiments E1 and E2

Frozen on 2026-10-27, before any E1/E2 run is started. This file is committed to the repository before the first run; the commit hash is the proof of order.

## 0. What was already seen before writing this

Disclosure: the 60 v2 runs (Attention U-Net and SAMed; N = 5, 10, 20 and shared; seeds 42, 123, 7), Table 1, and the following no-training analyses had already been looked at: size audit of SA/V, rough variance split from the 3-seed SDs, checkpoint-rule sensitivity (best / mean-last-5 / last), bootstrap over validation volumes, MDE arithmetic, and HD95 of all 60 checkpoints. Nothing from E1 or E2 has been seen. All rules below concern E1/E2 only.

## 1. Fixed protocol (nothing may change)

- Code: repository commit `485d1dc` (`src/` untouched), protocol `epoch_baseline_v2`, Tesla T4, model Attention U-Net, N = 5, 2D slice training, same 9 validation volumes, same hyper-parameters (batch 8, AdamW lr 1e-4, wd 1e-4, max 100 epochs, patience 15, min_delta 0.001).
- Primary metric: best-epoch mean 3D DSC on the 9 validation volumes. Sensitivity metric: mean of the last 5 epochs. Secondary: HD95 (mm), voxel spacing 1.5 x 1.5 x 2.0, definition = max of the two directed 95th percentiles; the pooled version is saved as a sensitivity column.

## 2. Fixed design (19 runs)

E1, random-subset cloud. Subsets are drawn with `gen_random(seed)` (sorted pool, `random.Random(seed).shuffle`, first 5); training_seed = selection_seed for new subsets.

| selection seed | volumes (spleen_ID) | mean volume (mL) |
|---|---|---|
| 1001 | 29, 47, 33, 19, 52 | 302 |
| 1002 | 13, 31, 49, 22, 27 | 274 |
| 1003 | 14, 46, 38, 59, 26 | 274 |
| 1004 | 13, 60, 28, 61, 19 | 242 |
| 1005 | 3, 49, 19, 33, 18 | 280 |
| 1006 | 10, 61, 13, 60, 19 | 261 |
| 1007 | 12, 49, 19, 63, 21 | 312 |
| 1008 | 24, 14, 27, 28, 52 | 201 |
| 1009 | 63, 56, 10, 24, 27 | 179 |
| 1010 | 47, 27, 45, 60, 21 | 243 |

Existing random subsets (selection seeds 42, 123, 7) already have one run each (training seed = selection seed). Six reseeds are added: subset 42 with training seeds 123 and 7; subset 123 with training seeds 42 and 7; subset 7 with training seeds 42 and 123. This gives K = 13 subsets in the cloud (E1 = 10 new + 6 reseed runs = 16 runs).

Note: the seed-42 subset comes from the earlier `random_selection.json`, not from `gen_random`; this is reported in the paper.

E2, size-controlled boundary measure (3 runs). Rank pool volumes by the residual of log(SA/V) after regressing on log(volume) (fit on all 32 pool volumes; ties broken by volume ID). Top 5: spleen_60, spleen_32, spleen_53, spleen_8, spleen_14 (mean volume 342 mL). Training seeds 42, 123, 7.

## 3. Estimation

- Training noise sigma_xi^2: within-subset variance of the three fully reseeded random subsets (6 df) and, separately, of the deterministic strategies (CT-FM, SA/V; 3 seeds each); both are reported, and a pooled value.
- Subset variance: sigma_S^2 = Var(subset means) - sigma_xi^2 / mean number of seeds per subset (floored at 0).
- Subset share = sigma_S^2 / (sigma_S^2 + sigma_xi^2). 95% interval: subset-level bootstrap (resample the 13 subsets with their runs, 5000 resamples, percentile interval).

## 4. Decision rules

1. Headline.
   - Interval excludes 0 and point estimate >= 50%: headline = "at N = 5, subset choice matters as much as or more than the training seed".
   - Interval excludes 0 and point estimate < 50%: headline = "subset choice contributes a measurable but smaller share"; the number is reported as estimated.
   - Interval includes 0 and upper bound < 50%: headline = "most variation at N = 5 is training noise".
   - Interval includes 0 and upper bound >= 50%: inconclusive. No X% claim is made; the contribution is the protocol and the MDE.
2. Position of deterministic subsets. For CT-FM and SA/V, report p* = (1 + #{cloud means >= strategy mean}) / (K + 1) as a rank/position, not as a significance test (with K = 13 the minimum is 0.071). One extension is pre-approved: 6 more random subsets, selection seeds 1011-1016 (K = 19), run only if a deterministic strategy sits in the extreme 2 ranks of 14. No other additional runs.
3. E2. Rank E2's 3-seed mean among the 13 cloud means plus E2 (14 values). Rank 1-2: "consistent with the corrected hypothesis (single-subset evidence)". Rank 3-7: inconclusive. Rank 8-14: "the size-free boundary measure is not informative here". The word "proves" is never used. E2 is compared with the SA/V subset as well.
4. Size-proxy. The audit numbers (log-log slope, Spearman, overlap with the N smallest volumes) are fixed as computed. The 13-subset Spearman between subset mean volume and subset DSC is labelled exploratory.
5. Metric robustness. A claim is "robust" only if its direction agrees under best-epoch DSC and mean-last-5 DSC; otherwise it is reported as fragile. HD95 is secondary; no headline depends on it. HD95 is reported as median and as the number of volumes above 50 mm, not only as a mean.
6. Conduct. Every run is reported. A crashed run is repeated with the same ID and seeds. No hyper-parameter, code or validation-set change. No run is dropped or added except under rule 2.
