# Track A -- Rehab-Pile benchmark

## KIMORE_clf_bn_LA

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.607 ± 0.323 | 0.607 ± 0.323 | 0.700 ± 0.245 | 0.719 ± 0.242 | 0.300 1.000 0.400 0.333 1.000 |
| rf_flatten | 0.607 ± 0.323 | 0.607 ± 0.323 | 0.700 ± 0.245 | 0.719 ± 0.242 | 0.300 1.000 0.400 0.333 1.000 |
| rf_summary | 0.607 ± 0.323 | 0.607 ± 0.323 | 0.700 ± 0.245 | 0.719 ± 0.242 | 0.300 1.000 0.400 0.333 1.000 |
| minirocket | 0.610 ± 0.222 | 0.701 ± 0.232 | 0.717 ± 0.194 | 0.743 ± 0.167 | 0.708 0.455 0.400 0.486 1.000 |

Pooled per-class F1 for `minirocket`: 0=0.429, 1=0.833

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.
