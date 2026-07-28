# Track A -- Rehab-Pile benchmark

## Headline: mean rank over 14 datasets

| model | mean rank | wins | mean macro_f1_mean | median | datasets |
|---|---|---|---|---|---|
| minirocket | **1.79** | 8 | 0.610 | 0.614 | 14 |
| rf_summary | **2.14** | 4 | 0.542 | 0.587 | 14 |
| rf_flatten | **2.68** | 2 | 0.477 | 0.510 | 14 |
| majority | **3.39** | 1 | 0.368 | 0.341 | 14 |

Mean rank, not mean score. Test folds here hold 6-14 samples, so a per-fold score can swing on sampling alone, and averaging raw scores would let a dataset where everything scores 0.9 outweigh one where everything scores 0.4. Ranking is the standard presentation in the time-series-classification literature for exactly this reason. `wins` counts datasets where a model tied or took the top score.

`class_weight='balanced'` is set on RandomForest and MiniRocket. aeon does not expose it for LITETimeClassifier, so LITEMV runs unweighted -- relevant wherever the classes are imbalanced.

## Per collection

| family | n | majority | minirocket | rf_flatten | rf_summary |
|---|---|---|---|---|---|
| IRDS | 1 | 0.436 | 0.615 | 0.236 | 0.330 |
| KERAAL | 6 | 0.431 | 0.554 | 0.446 | 0.566 |
| KIMORE | 5 | 0.392 | 0.614 | 0.505 | 0.530 |
| UCDHE | 2 | 0.084 | 0.765 | 0.621 | 0.610 |

## Per dataset

## KIMORE_clf_bn_LA

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.607 ± 0.323 | 0.607 ± 0.323 | 0.700 ± 0.245 | 0.719 ± 0.242 | 0.300 1.000 0.400 0.333 1.000 |
| rf_flatten | 0.607 ± 0.323 | 0.607 ± 0.323 | 0.700 ± 0.245 | 0.719 ± 0.242 | 0.300 1.000 0.400 0.333 1.000 |
| rf_summary | 0.607 ± 0.323 | 0.607 ± 0.323 | 0.700 ± 0.245 | 0.719 ± 0.242 | 0.300 1.000 0.400 0.333 1.000 |
| minirocket | 0.610 ± 0.222 | 0.701 ± 0.232 | 0.717 ± 0.194 | 0.743 ± 0.167 | 0.708 0.455 0.400 0.486 1.000 |

Pooled per-class F1 for `minirocket`: 0=0.429, 1=0.833

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KIMORE_clf_bn_LT

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.289 ± 0.074 | 0.289 ± 0.074 | 0.500 ± 0.000 | 0.419 ± 0.129 | 0.300 0.333 0.333 0.143 0.333 |
| rf_flatten | 0.352 ± 0.225 | 0.352 ± 0.225 | 0.467 ± 0.194 | 0.390 ± 0.230 | 0.222 0.625 0.143 0.143 0.625 |
| rf_summary | 0.457 ± 0.158 | 0.457 ± 0.158 | 0.573 ± 0.139 | 0.519 ± 0.132 | 0.300 0.625 0.250 0.486 0.625 |
| minirocket | 0.420 ± 0.227 | 0.420 ± 0.227 | 0.435 ± 0.214 | 0.476 ± 0.205 | 0.708 0.667 0.143 0.333 0.250 |

Pooled per-class F1 for `rf_summary`: 0=0.348, 1=0.615

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KIMORE_clf_bn_PR

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.237 ± 0.090 | 0.237 ± 0.090 | 0.500 ± 0.000 | 0.329 ± 0.155 | 0.125 0.333 0.143 0.250 0.333 |
| rf_flatten | 0.632 ± 0.271 | 0.632 ± 0.271 | 0.690 ± 0.269 | 0.652 ± 0.247 | 0.417 0.829 0.250 0.667 1.000 |
| rf_summary | 0.632 ± 0.271 | 0.632 ± 0.271 | 0.690 ± 0.269 | 0.652 ± 0.247 | 0.417 0.829 0.250 0.667 1.000 |
| minirocket | 0.678 ± 0.109 | 0.678 ± 0.109 | 0.755 ± 0.102 | 0.714 ± 0.103 | 0.533 0.829 0.778 0.625 0.625 |

Pooled per-class F1 for `minirocket`: 0=0.780, 1=0.571

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KIMORE_clf_bn_Sq

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.362 ± 0.080 | 0.362 ± 0.080 | 0.500 ± 0.000 | 0.590 ± 0.185 | 0.222 0.455 0.400 0.400 0.333 |
| rf_flatten | 0.407 ± 0.134 | 0.407 ± 0.134 | 0.508 ± 0.093 | 0.590 ± 0.185 | 0.222 0.455 0.333 0.400 0.625 |
| rf_summary | 0.369 ± 0.171 | 0.369 ± 0.171 | 0.458 ± 0.179 | 0.524 ± 0.253 | 0.222 0.455 0.143 0.400 0.625 |
| minirocket | 0.703 ± 0.068 | 0.703 ± 0.068 | 0.748 ± 0.098 | 0.743 ± 0.076 | 0.708 0.778 0.778 0.625 0.625 |

Pooled per-class F1 for `minirocket`: 0=0.636, 1=0.800

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KIMORE_clf_bn_TR

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.465 ± 0.288 | 0.465 ± 0.288 | 0.600 ± 0.200 | 0.629 ± 0.275 | 0.125 0.400 0.400 0.400 1.000 |
| rf_flatten | 0.527 ± 0.317 | 0.527 ± 0.317 | 0.625 ± 0.224 | 0.629 ± 0.295 | 0.125 0.333 0.400 0.778 1.000 |
| rf_summary | 0.583 ± 0.279 | 0.583 ± 0.279 | 0.692 ± 0.207 | 0.690 ± 0.237 | 0.286 0.829 0.400 0.400 1.000 |
| minirocket | 0.661 ± 0.346 | 0.741 ± 0.322 | 0.783 ± 0.194 | 0.729 ± 0.318 | 0.125 1.000 0.778 1.000 0.400 |

Pooled per-class F1 for `minirocket`: 0=0.526, 1=0.791

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KERAAL_clf_mc_CTK

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.301 ± 0.144 | 0.301 ± 0.144 | 0.444 ± 0.079 | 0.538 ± 0.273 | 0.222 0.089 0.176 0.417 0.462 0.440 |
| rf_flatten | 0.492 ± 0.236 | 0.492 ± 0.236 | 0.593 ± 0.168 | 0.659 ± 0.264 | 0.644 0.208 0.176 0.775 0.708 0.440 |
| rf_summary | 0.429 ± 0.194 | 0.429 ± 0.194 | 0.520 ± 0.133 | 0.624 ± 0.234 | 0.541 0.222 0.176 0.754 0.440 0.440 |
| minirocket | 0.428 ± 0.191 | 0.462 ± 0.189 | 0.523 ± 0.169 | 0.636 ± 0.274 | 0.462 0.343 0.107 0.754 0.462 0.440 |

Pooled per-class F1 for `rf_flatten`: C=0.765, E1=0.593, E2=0.000, E3=0.000

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KERAAL_clf_mc_ELK

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.591 ± 0.309 | 0.591 ± 0.309 | 0.667 ± 0.236 | 0.798 ± 0.284 | 1.000 0.467 0.444 0.481 1.000 0.154 |
| rf_flatten | 0.535 ± 0.209 | 0.609 ± 0.238 | 0.636 ± 0.207 | 0.770 ± 0.175 | 1.000 0.429 0.412 0.481 0.440 0.450 |
| rf_summary | 0.645 ± 0.253 | 0.645 ± 0.253 | 0.687 ± 0.240 | 0.830 ± 0.162 | 1.000 0.467 0.412 0.462 1.000 0.530 |
| minirocket | 0.479 ± 0.262 | 0.571 ± 0.304 | 0.617 ± 0.254 | 0.724 ± 0.289 | 0.481 0.467 0.333 0.462 1.000 0.133 |

Pooled per-class F1 for `rf_summary`: C=0.421, E2=0.911

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KERAAL_clf_mc_RTK

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.488 ± 0.386 | 0.488 ± 0.386 | 0.556 ± 0.356 | 0.600 ± 0.363 | 0.300 1.000 0.000 1.000 0.185 0.440 |
| rf_flatten | 0.262 ± 0.132 | 0.434 ± 0.270 | 0.430 ± 0.237 | 0.459 ± 0.235 | 0.211 0.467 0.100 0.316 0.118 0.364 |
| rf_summary | 0.599 ± 0.218 | 0.742 ± 0.203 | 0.720 ± 0.183 | 0.809 ± 0.115 | 0.775 0.385 0.471 1.000 0.525 0.440 |
| minirocket | 0.799 ± 0.305 | 0.799 ± 0.305 | 0.812 ± 0.271 | 0.849 ± 0.248 | 0.928 1.000 1.000 1.000 0.157 0.708 |

Pooled per-class F1 for `minirocket`: C=0.875, E1=0.000, E2=0.756

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## UCDHE_clf_mc_MP

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.100 ± 0.001 | 0.100 ± 0.001 | 0.250 ± 0.000 | 0.250 ± 0.004 | 0.101 0.102 0.099 0.099 0.099 |
| rf_flatten | 0.585 ± 0.061 | 0.585 ± 0.061 | 0.592 ± 0.060 | 0.594 ± 0.061 | 0.565 0.542 0.623 0.513 0.682 |
| rf_summary | 0.496 ± 0.040 | 0.496 ± 0.040 | 0.509 ± 0.033 | 0.510 ± 0.035 | 0.452 0.472 0.564 0.477 0.515 |
| minirocket | 0.798 ± 0.018 | 0.798 ± 0.018 | 0.797 ± 0.016 | 0.798 ± 0.016 | 0.813 0.778 0.805 0.774 0.819 |

Pooled per-class F1 for `minirocket`: a=0.882, arch=0.752, n=0.728, r=0.841

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## UCDHE_clf_mc_Rowing

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.067 ± 0.002 | 0.067 ± 0.002 | 0.200 ± 0.000 | 0.202 ± 0.008 | 0.068 0.067 0.063 0.068 0.070 |
| rf_flatten | 0.657 ± 0.052 | 0.657 ± 0.052 | 0.660 ± 0.052 | 0.657 ± 0.051 | 0.693 0.652 0.693 0.689 0.559 |
| rf_summary | 0.724 ± 0.042 | 0.724 ± 0.042 | 0.726 ± 0.046 | 0.721 ± 0.046 | 0.760 0.773 0.724 0.710 0.653 |
| minirocket | 0.732 ± 0.035 | 0.732 ± 0.035 | 0.733 ± 0.037 | 0.727 ± 0.037 | 0.763 0.786 0.707 0.707 0.699 |

Pooled per-class F1 for `minirocket`: a=0.719, ext=0.966, n=0.528, r=0.684, rb=0.766

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KERAAL_clf_bn_CTK

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.293 ± 0.127 | 0.293 ± 0.127 | 0.500 ± 0.000 | 0.462 ± 0.273 | 0.333 0.458 0.440 0.222 0.125 0.176 |
| rf_flatten | 0.559 ± 0.238 | 0.559 ± 0.238 | 0.647 ± 0.170 | 0.712 ± 0.229 | 0.857 0.458 0.176 0.844 0.576 0.440 |
| rf_summary | 0.400 ± 0.165 | 0.400 ± 0.165 | 0.477 ± 0.117 | 0.578 ± 0.209 | 0.300 0.350 0.176 0.714 0.417 0.440 |
| minirocket | 0.525 ± 0.190 | 0.525 ± 0.190 | 0.596 ± 0.122 | 0.681 ± 0.215 | 0.854 0.226 0.576 0.591 0.462 0.440 |

Pooled per-class F1 for `rf_flatten`: C=0.755, E=0.647

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KERAAL_clf_bn_ELK

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.591 ± 0.309 | 0.591 ± 0.309 | 0.667 ± 0.236 | 0.798 ± 0.284 | 1.000 0.467 0.444 0.481 1.000 0.154 |
| rf_flatten | 0.486 ± 0.254 | 0.563 ± 0.301 | 0.616 ± 0.229 | 0.729 ± 0.272 | 1.000 0.467 0.375 0.462 0.462 0.154 |
| rf_summary | 0.591 ± 0.309 | 0.591 ± 0.309 | 0.667 ± 0.236 | 0.798 ± 0.284 | 1.000 0.467 0.444 0.481 1.000 0.154 |
| minirocket | 0.479 ± 0.258 | 0.553 ± 0.296 | 0.600 ± 0.226 | 0.712 ± 0.285 | 0.440 0.467 0.333 0.481 1.000 0.154 |

Pooled per-class F1 for `majority`: C=0.000, E=0.899

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## KERAAL_clf_bn_RTK

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.320 ± 0.340 | 0.320 ± 0.340 | 0.417 ± 0.344 | 0.400 ± 0.363 | 0.364 0.000 1.000 0.000 0.381 0.176 |
| rf_flatten | 0.339 ± 0.143 | 0.464 ± 0.141 | 0.435 ± 0.150 | 0.414 ± 0.135 | 0.626 0.333 0.182 0.235 0.278 0.378 |
| rf_summary | 0.732 ± 0.221 | 0.796 ± 0.157 | 0.771 ± 0.170 | 0.803 ± 0.147 | 0.754 0.385 1.000 1.000 0.675 0.576 |
| minirocket | 0.612 ± 0.232 | 0.770 ± 0.245 | 0.775 ± 0.208 | 0.768 ± 0.225 | 0.785 0.467 1.000 0.480 0.291 0.650 |

Pooled per-class F1 for `rf_summary`: C=0.825, E=0.774

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.

## IRDS_clf_bn_EFL

| model | macro-F1 (union) | macro-F1 (present) | balanced acc | accuracy | per fold (macro-F1 union) |
|---|---|---|---|---|---|
| majority | 0.436 ± 0.324 | 0.436 ± 0.324 | 0.500 ± 0.316 | 0.599 ± 0.348 | 0.447 0.309 0.000 1.000 0.424 |
| rf_flatten | 0.236 ± 0.198 | 0.236 ± 0.198 | 0.290 ± 0.237 | 0.351 ± 0.301 | 0.446 0.309 0.000 0.000 0.424 |
| rf_summary | 0.330 ± 0.174 | 0.424 ± 0.304 | 0.478 ± 0.282 | 0.577 ± 0.325 | 0.447 0.309 0.000 0.471 0.424 |
| minirocket | 0.615 ± 0.289 | 0.763 ± 0.236 | 0.726 ± 0.247 | 0.788 ± 0.230 | 0.914 1.000 0.267 0.471 0.424 |

Pooled per-class F1 for `minirocket`: 1=0.889, 2=0.800

The reported number is the **mean over folds with its standard deviation**. The pooled confusion matrix below is for display only; the macro-F1 computed from it is a different quantity. A large std measures how much the score depends on which subjects were held out.
