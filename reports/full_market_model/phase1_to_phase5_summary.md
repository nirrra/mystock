# Phase 1-5 Full-Market Model Summary

Date: 2026-05-07

This document summarizes the current standalone reproduction phases. Each phase is treated as an independent reference-method reproduction. Phase 6 is intentionally not started.

## Phase 1: Tail-Decline Risk Classification

Reference target: rolling tail-decline risk classification using common machine-learning classifiers.

Best current risk models:

| model | windows | avg PR-AUC | baseline PR-AUC | avg ROC-AUC | filter pass |
|---|---:|---:|---:|---:|---|
| logistic_regression | 6 | 0.1509 | 0.0572 | 0.7015 | true |
| linear_discriminant_analysis | 6 | 0.1505 | 0.0572 | 0.7037 | true |

Highest-risk 20% filter impact:

| model | risk label delta | 5d return delta | 5d max drawdown delta |
|---|---:|---:|---:|
| logistic_regression | -0.0180 | +0.000238 | +0.002202 |
| linear_discriminant_analysis | -0.0179 | +0.000252 | +0.002586 |

Interpretation: Phase 1 is a valid broad risk filter. It has strong ROC-AUC and stable risk-decile ordering, but the trading-impact uplift is smaller than Phase 2 and old Phase 3.

## Phase 2: Triple-Barrier / Mlfin-Style Risk Label

Reference target: triple-barrier risk labels using CUSUM event sampling and LightGBM classification.

Best current configuration:

| config | model | windows | avg PR-AUC | baseline PR-AUC | avg ROC-AUC | filter pass |
|---|---|---:|---:|---:|---:|---|
| mlfin_h5_pt1_sl1_minret0.005 | lightgbm_classifier | 6 | 0.4929 | 0.4114 | 0.5829 | true |

Highest-risk 20% filter impact:

| 5d return delta | 5d max drawdown delta |
|---:|---:|
| +0.002578 | +0.003975 |

Interpretation: Phase 2 is currently the strongest practical risk exclusion model. It gives much larger return and drawdown improvement than Phase 1. Although its ROC-AUC is lower than Phase 1, the target is more trading-like and its filter impact is better.

## Phase 3: Alpha158 Risk Adaptation Experiment

Status: This is no longer the strict mainline reproduction. It was an Alpha158 + LightGBM risk adaptation using Phase 2 labels. It remains useful as an internal risk-feature experiment.

| model | windows | avg PR-AUC | baseline PR-AUC | avg ROC-AUC | filter pass |
|---|---:|---:|---:|---:|---|
| lightgbm_classifier | 6 | 0.4946 | 0.4115 | 0.5839 | true |

Highest-risk 20% filter impact:

| 5d return delta | 5d max drawdown delta |
|---:|---:|
| +0.002679 | +0.003879 |

Interpretation: old Phase 3 is slightly better than Phase 2 by PR-AUC and return uplift, but slightly worse by drawdown improvement. Because it is not a strict reference-method phase and depends on Phase 2 labels, it should not replace Phase 2 as the main risk model without additional independent validation.

## Phase 4: Qlib Alpha158 + LightGBM Return Regression

Reference target: Qlib Alpha158 features, Qlib `LABEL0 = Ref($close, -2) / Ref($close, -1) - 1`, and Qlib benchmark LightGBM regression parameters.

Signal metrics:

| split | rows | days | mean IC | ICIR | positive IC rate | mean RankIC | RankICIR | positive RankIC rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 1,385,904 | 484 | 0.0674 | 0.9710 | 0.8347 | 0.0366 | 0.5246 | 0.7355 |
| test | 1,660,633 | 562 | 0.0587 | 0.7604 | 0.7794 | 0.0414 | 0.4813 | 0.7153 |

TopK local approximation:

| split | avg net return | annualized net return | information ratio | max drawdown | hit rate |
|---|---:|---:|---:|---:|---:|
| valid | +0.002374 | 0.7757 | 2.7646 | -0.2130 | 0.5930 |
| test | +0.002381 | 0.7470 | 2.0863 | -0.2886 | 0.5996 |

Interpretation: Phase 4 is the current standalone return model. It is not a risk exclusion model. Its IC and RankIC are positive out of sample, and the high-score bucket outperforms the low-score bucket.

## Phase 5: MCD Crash-Risk Label Generation

Reference target: Karasan, Alp, and Weber (2025) crash-risk label generation only.

Current full-sample distribution:

| measure | value |
|---|---:|
| firm-years | 27,879 |
| NEGOUTLIER rate | 0.1488 |
| CRASH rate | 0.0834 |
| NCSKEW mean | -0.7717 |
| DUVOL mean | -0.5655 |

Important correlations:

| pair | correlation |
|---|---:|
| NEGOUTLIER vs CRASH | 0.2918 |
| NEGOUTLIER vs MINRET | -0.5967 |
| NCSKEW vs DUVOL | 0.9068 |

Interpretation: Phase 5 is a long-horizon crash-risk label generator, not a daily trading model. Its `NEGOUTLIER` coverage is higher than the paper's reported mean, mainly due to local A-share data, local equal-weight market return, and the available feature set. It should be used later as a risk-profile slice or auxiliary risk field only after calibration.

## Risk Exclusion Model Comparison

For the specific task "exclude the riskiest names", compare Phase 1, Phase 2, and old Phase 3 by filter impact:

| phase | model | filter | 5d return delta | 5d max drawdown delta | notes |
|---|---|---:|---:|---:|---|
| Phase 1 | logistic_regression | highest-risk 20% | +0.000238 | +0.002202 | broad tail-risk baseline |
| Phase 1 | linear_discriminant_analysis | highest-risk 20% | +0.000252 | +0.002586 | best Phase 1 drawdown uplift |
| Phase 2 | barrier LightGBM | highest-risk 20% | +0.002578 | +0.003975 | strongest mainline risk filter |
| old Phase 3 | Alpha158 risk LightGBM | highest-risk 20% | +0.002679 | +0.003879 | slightly better return delta, slightly weaker drawdown delta |

Current recommendation:

Use **Phase 2 `mlfin_h5_pt1_sl1_minret0.005` LightGBM** as the primary risk exclusion model for the next combined backtest layer.

Reason:

- It is an independent, strict risk-method phase.
- Its drawdown improvement is the best among the mainline risk candidates.
- It has nearly the same PR-AUC as old Phase 3.
- It avoids using the old Phase 3 adaptation as a dependency, which is consistent with the rule that phases should remain independent.

Secondary option:

Use old Phase 3 only as a robustness comparator in the later unified backtest layer. Do not promote it as the main risk filter until it is rerun as a standalone, strictly defined reference method.
