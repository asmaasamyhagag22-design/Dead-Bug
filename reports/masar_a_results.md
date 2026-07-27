# Track A -- Rehab-Pile model ladder

## KERAAL_clf_mc_CTK

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.301 ± 0.144 | 0.301 ± 0.144 | 0.444 ± 0.079 | 0.538 ± 0.273 | 0.222 0.089 0.176 0.417 0.462 0.440 |
| rf_flatten | 0.493 ± 0.249 | 0.493 ± 0.249 | 0.593 ± 0.179 | 0.659 ± 0.260 | 0.593 0.196 0.176 0.844 0.708 0.440 |
| rf_summary | 0.440 ± 0.225 | 0.440 ± 0.225 | 0.541 ± 0.174 | 0.636 ± 0.244 | 0.541 0.196 0.176 0.844 0.440 0.440 |

Pooled per-class F1 for `rf_flatten`: C=0.772, E1=0.582, E2=0.000, E3=0.000

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.
