# Addendum 1 to PREREGISTRATION.md (commit e092150)

Dated 2026-10-28. This addendum changes analysis details only. No run, subset, seed, code or hyper-parameter changes.

**Disclosure:**
- [x] When this addendum was written, no E1/E2 DSC, HD95 or variance values had been looked at. Only the completeness and integrity checks of the logs had been run.

## A1. Variance-component estimator (replaces the plug-in formula in section 3)

The E1 design is unbalanced: three subsets have 3 training seeds each, ten subsets have 1. The plug-in formula "Var(subset means) minus sigma_xi^2 divided by the average number of seeds" is therefore not used. Instead:

- Primary estimate: one-way random-effects model (subset = random effect, training seed = residual), fitted by REML, on the Attention U-Net N = 5 random-subset runs (13 subsets, 19 runs).
- Cross-check: the unbalanced ANOVA (method-of-moments) estimator, with n0 = (N - sum(n_k^2) / N) / (K - 1).
- Subset share = sigma_S^2 / (sigma_S^2 + sigma_xi^2). The 95% interval is the subset-level bootstrap of section 3 (resample the 13 subsets together with all their runs, refit, 5000 resamples, percentile interval).
- Rule 1 is applied to the REML estimate and its interval. If the ANOVA estimate would put the result in a different branch of Rule 1, both are reported and the result is called sensitive to the estimator.
- The within-subset and deterministic-strategy estimates of sigma_xi^2 described in section 3 are still reported separately.

## A2. Single-run (diagonal) sensitivity version of Rules 2 and 3

In the cloud, the three existing subsets are summarised by 3 seeds and the ten new subsets by 1 seed, while a deterministic strategy (or E2) is summarised by the mean of 3 seeds. Means of 3 seeds are less noisy, which pulls their rank towards the middle. Rules 2 and 3 stay exactly as written (primary). In addition they are repeated in a single-run version: the 13 diagonal runs (training seed = selection seed) form the cloud, and each seed of CT-FM, SA/V and E2 is ranked against it separately; the median of the three ranks is reported. If the primary and the single-run versions fall in different bins of Rule 3 (top 2, ranks 3-7, ranks 8-14), or if only one of them satisfies the "extreme 2 ranks" condition of Rule 2, the result is reported as not extreme / inconclusive.

## A3. Unchanged

Everything else in PREREGISTRATION.md is unchanged, including the six rules, the metric rule (best-epoch primary, mean-last-5 sensitivity) and the one pre-approved extension (seeds 1011-1016).
