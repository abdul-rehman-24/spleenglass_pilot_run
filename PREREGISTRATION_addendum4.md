# Addendum 4 to PREREGISTRATION.md (after commits e092150, 8081e7b, 32a79d3, 582437b and 8531f97)

Dated 2026-10-31. This addendum pre-registers experiment E5, the replication of the random-subset design with a second model (SAMed). It changes no run that has already been done.

**Disclosure:** all Attention U-Net results (E1, E2, the extension, E4, the K = 19 analysis, the single-run comparisons) and all SAMed results of the 60 earlier runs (Table 1: 3 seeds per setting, including the three random N = 5 subsets and the CT-FM and SA/V subsets) had been seen when this addendum was written. E5 was designed knowing them. One observation motivated question Q3: among the three random subsets for which both models had been run (selection seeds 7, 42, 123, one run each), the order of the subsets differs between the models (subset 123 is the best for Attention U-Net and the worst for SAMed). With single runs this is anecdotal. No E5 result exists yet.

## E1. Questions

- Q1 (generality): does the Attention U-Net pattern (subset share of variance not identifiable with best-epoch DSC but large with the last-5-epoch mean; single-run comparisons unreliable; MDE of about 5 DSC points) also hold for SAMed?
- Q2: where does the E2 subset rank among SAMed's random subsets?
- Q3 (transfer): is the quality of a subset shared between the two models?

## E2. Fixed design (25 SAMed runs)

SAMed (LoRA rank 4, SAM ViT-B), N = 5, batch size 1, same commit `485d1dc`, protocol `epoch_baseline_v2`, Tesla T4, same hyper-parameters and validation volumes as the 60 earlier runs (4,212,016 trainable parameters out of 90,884,656). Subsets are exactly the ones already fixed in PREREGISTRATION.md (selection seeds 1001-1010), addendum 2 (1011-1016), the earlier random subsets (selection seeds 42, 123, 7) and E2 (spleen_60, 32, 53, 8, 14). No new subset is designed.

- 16 new runs, one per random subset with selection seeds 1001-1016 (training seed = selection seed).
- 6 reseeds: subset 42 with training seeds 123 and 7; subset 123 with 42 and 7; subset 7 with 42 and 123.
- 3 runs of E2 (training seeds 42, 123, 7).

The runs of SAMed already done are reused: the diagonal runs of subsets 42, 123 and 7 (v2 logs) and the CT-FM and SA/V runs (3 seeds each). Together the SAMed cloud has 19 subsets and 25 runs, as for Attention U-Net.

Stopping rule: no further runs are added after these 25, whatever the result. SAMed is not run on E4. Every run is reported; a crashed run is repeated with the same ID and seeds.

## E3. Analyses

A. Variance split (Q1). The estimator, intervals and branches of Rule 1 (with addenda 1 and 2) are applied to the SAMed cloud, for the best-epoch DSC (primary) and the last-5 mean (sensitivity). The subset share of the two models is described side by side, not tested, because the intervals are wide.

B. Single-run comparisons (Q1). As for Attention U-Net: pairs of single runs of the random subsets, and of CT-FM, SA/V and E2 against single runs of the random subsets (percentage of pairs in which the first is better, differences above 2 and 5 DSC points, range).

C. E2 on SAMed (Q2). Rank of the three-run mean of E2 among the 19 random-subset means (primary) and single-run version (median rank of the three runs among the 19 diagonal runs), with the bins 1-3 / 4-10 / 11-20 of addendum 2. This is descriptive. E4 showed that the advantage of E2 did not replicate for Attention U-Net, so no claim about a boundary criterion is made from this result.

D. Transfer (Q3). Unit: the 19 random subsets. Score of a subset for a model = DSC of its diagonal run (training seed = selection seed). The primary metric for the transfer question is the last-5 mean, because the best-epoch DSC of Attention U-Net has an estimated subset share close to zero (0% by REML, 27% by ANOVA), which would make any correlation of single runs uninformative; this choice was made after seeing the Attention U-Net variance split. The best-epoch version is reported as secondary.
- Spearman correlation between the two models over the 19 subsets, with a 95% percentile bootstrap interval (resampling subsets, 5000 resamples). Pearson correlation as a check.
- Reliability note: the reliability of a single run as a measure of subset quality is R = sigma_S^2 / (sigma_S^2 + sigma_xi^2) from the REML fit of each model and metric (section A). The largest correlation that can be observed even if subset quality were perfectly shared is about sqrt(R_AU x R_SAMed). The correlation divided by this ceiling is reported as a descriptive attenuation-corrected value, with its bootstrap interval, and is not used in a decision.
- T1, subset quality is partly shared: the interval of the last-5 Spearman correlation excludes 0 with a positive lower bound. Robust only if the best-epoch correlation also has a positive lower bound.
- T2, weakly shared or model-specific: only if the ceiling for the last-5 metric is at least 0.6 AND the upper end of the interval is below 0.3.
- T3, anything else: inconclusive. An interval that includes 0 cannot separate "not shared" from "measured too noisily".
- This is a single primary test (T1 on the last-5 metric); everything else is descriptive.

## E4. Power (expected, for planning)

With 19 subsets a two-sided 5% test of a Spearman correlation needs about 0.46 to be significant, and the 95% interval has a half-width of about 0.35-0.4. Only strong sharing can therefore be detected, and the attenuation described above makes the observed correlation smaller than the true one.

## E5. Unchanged

Everything else in PREREGISTRATION.md and addenda 1, 2 and 3 is unchanged, including the metric rule (best-epoch primary, mean-last-5 sensitivity) for all analyses other than the transfer question.
