# Experiment Observations

## Dataset audit
Expected large overlap of patients under image-level split based on 33,126 images
of 2,056 patients. Observed 1,704 of 2,056 patients (82.9%) appearing in both
splits. Both splitting methods produce 117 validation positives at the same rate,
isolating the leakage problem.

## Timing probe (E0, 3 epochs)
~130 s/epoch after entering steady state on RTX 4000 Ada. ROC-AUC still growing
to 3rd epoch (0.846 -> 0.881 -> 0.895), so 8 epochs set as the constant for all.

## E0 baseline
ROC-AUC 0.895 and recall 0.000 at threshold 0.5 due to just 1 false positive
-- majority class collapsing. Ranking is fine; the probability mass is below the
decision threshold.

## E1 vs E0
ROC-AUC 0.895 -> 0.903. Recall at 0.5 increased from 0.000 to 0.427, but at a
matched 10% FPR only from 0.615 to 0.735. Much of the gain from balancing
comes from moving the decision threshold down, not better discrimination.

## E3 determinism check
E3 reproduced E1 exactly at every epoch, confirming that the pipeline is
deterministic and ablation effects are not artifacts of different code paths.

## E7 leakage comparison
Expected the leaky image-level split to do better. It didn't (0.8975 vs 0.9034).
Patients have similar skin tones and lighting but very different lesion
morphologies, and 428 patients contain 584 melanomas, so most positive patients
have just 1 melanoma -- being the same patient is poor evidence for being
malignant.

## Ablations
Incoherent as a causal story: removal of age (0.9142) and removal of sex
(0.9140) both beat the full metadata model (0.9050). Motivated the seed study.

## Seed variance
E1 0.8973 +/- 0.0071, E2 0.9006 +/- 0.0046 over seeds 42/0/1. Variance
between seeds exceeds every variance between configurations measured, including
the leakage comparison.

## Failure and subgroup analysis
At matched 10% FPR: male recall 0.732 vs female 0.652 at near-identical FPR.
Age 40-55 recall 0.576 vs age over-70 0.857, with FPR also increasing with
age. Head/neck under-represented among failures (5.7% of misses against 13.8%
of all melanomas); torso over-represented (51.4% against 46.6%). Positive
counts per subgroup are small, so every rate is reported next to its count.
