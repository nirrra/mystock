# V4 Alpha Ranker Design

Date: 2026-05-03

## 1. Background

V3 with `weighted-all` and `bad_risk_sample_weight=0.7` has become the current main risk-control version.

On the out-of-sample test Top 20, V3 0.7 achieved:

- 20-day average return: 2.44%.
- 20-day take-profit rate: 15.98%.
- 20-day stop-loss rate: 13.91%.
- 20-day average maximum drawdown: 4.47%.
- Bad-risk rate: 21.05%.

This shows that the risk gate is useful. The remaining bottleneck is upside ranking inside the low-risk candidate pool. V4 therefore keeps the V3 0.7 risk filter fixed and adds a dedicated alpha ranking layer.

## 2. Goal

Improve return capture without giving back the V3 risk-control gain.

Primary V4 test targets:

- Keep daily Top 20 20-day stop-loss rate at or below 15%.
- Increase daily Top 20 20-day take-profit rate from 15.98% toward 20% or higher.
- Increase daily Top 20 average 20-day return from 2.44% toward 3% or higher.
- Keep daily Top 20 average maximum drawdown at or below 5%.
- Keep daily Top 20 bad-risk rate at or below 23%.

V4 should be judged by daily Top N trade quality, not by global regression error.

## 3. Scope

Included:

- A-share mainboard daily data.
- Existing V3 0.7 hard risk gate as the admission layer.
- Existing Stage 1 horizon model outputs for 5d, 10d, 20d, and 60d.
- Existing technical, moving-average, volume, amount, and block-return features.
- A new risk-adjusted alpha grade target.
- A new daily cross-sectional ranking model.
- Validation/test reports comparing V3 0.7 versus V4.

Excluded:

- Pattern-recognition outputs.
- Existing repository probability models.
- Existing watchlist or trend scores as labels.
- Intraday execution.
- Portfolio allocation and live trading rules.

## 4. Architecture

V4 is an additional ranking layer after V3 risk admission:

```text
daily features
  -> Stage 1 horizon models
      -> up/down/neutral probabilities for 5d, 10d, 20d, 60d
  -> V3 0.7 Risk Filter
      -> hard low-risk candidate pool
  -> alpha label builder
      -> alpha_grade from future 20d return, take-profit, drawdown, and bad_risk
  -> Alpha Ranker
      -> alpha_rank_score
  -> final V4 ranking
      -> daily Top N report
```

The V3 gate remains a hard gate. V4 does not allow a high-alpha stock into Top N if it fails the V3 risk admission filter.

## 5. Alpha Grade Target

The model should not directly regress 20-day return. Raw future return is too noisy and does not express path quality. V4 will train on a 0-4 ordinal grade:

| Grade | Meaning |
|---|---|
| 4 | Strong 20-day upside, near or above the 15% take-profit target, with controlled drawdown |
| 3 | Clear positive 20-day return with acceptable drawdown |
| 2 | Small positive return or flat result |
| 1 | Weak or small negative return without severe risk |
| 0 | Stop-loss, large drawdown, or bad-risk outcome |

Initial grade rules:

```text
grade 4:
  period_return_20d >= 12%
  and max_drawdown_20d <= 8%
  and bad_risk == 0

grade 3:
  period_return_20d >= 6%
  and max_drawdown_20d <= 10%
  and bad_risk == 0

grade 2:
  period_return_20d >= 0%
  and max_drawdown_20d <= 12%

grade 1:
  period_return_20d > -6%
  and max_drawdown_20d <= 15%

grade 0:
  otherwise
```

The initial thresholds are intentionally conservative. They use the same 20-day trade horizon as the current main evaluation target, while still letting the model learn from near-take-profit winners instead of only exact +15% hits.

## 6. Ranking Model

The preferred implementation is a daily cross-sectional ranking model.

Training unit:

- Each trade date is one query group.
- Each stock on that date is one row.
- The label is `alpha_grade`.
- The model learns which stocks deserve higher rank on the same date.

Preferred engine:

- If available locally: LightGBM ranker with a ranking objective such as `lambdarank` or `rank_xendcg`.
- Fallback: `HistGradientBoostingRegressor` trained on `alpha_grade`, evaluated strictly as a ranking model.

The fallback keeps implementation unblocked if LightGBM is not installed, but the report must state which engine was used.

## 7. Features

V4 uses the same safe feature universe available at prediction time:

- Raw daily technical features.
- Stage 1 horizon probabilities and expected values.
- Stage 2 derived features such as 20d/60d edge, short-term down pressure, and weighted expected value.
- V3 `risk_score`, `clean_win_score`, and risk-gate fields as model features or final blend inputs.

No future fields or realized labels may enter the feature matrix.

Feature additions can be considered after the first V4 run:

- Market regime features.
- Industry relative strength.
- Market breadth.
- Recent excess return versus market or industry.

These are later additions, not required for the first V4 implementation.

## 8. Final Score

First implementation:

```text
candidate_pool = rows passing V3 0.7 hard risk gate
final_score_v4 = alpha_rank_score
```

Secondary blend for comparison:

```text
final_score_v4_blend =
    alpha_rank_score
  + 0.20 * clean_win_score
  - 0.10 * risk_score
```

Reports should include both pure alpha and blended ranking if implementation cost is low. The selected default should be whichever performs better on validation Top 20 under the risk constraints.

## 9. Validation

V4 parameter and model selection must use only the validation split. Test split is used once for final reporting.

Reports:

- Split metrics for all rows, V3 candidate pool, V4 Top 20, and V4 Top 50.
- Daily Top N metrics.
- V3 0.7 versus V4 comparison table.
- Alpha grade distribution by split.
- Prediction file for the latest requested trade date.

Key metrics:

- 20-day win rate.
- 20-day average return.
- 20-day take-profit rate.
- 20-day stop-loss rate.
- 20-day average take-profit.
- 20-day average stop-loss.
- 20-day average maximum drawdown.
- Bad-risk rate.
- Candidate count.

## 10. Acceptance Criteria

V4 is accepted as the new main version only if it improves expected return while preserving V3 risk behavior.

Minimum acceptance:

- Test Top 20 20-day stop-loss rate <= 15%.
- Test Top 20 20-day average maximum drawdown <= 5%.
- Test Top 20 bad-risk rate <= 23%.
- Test Top 20 20-day average return > V3 0.7.

Preferred acceptance:

- Test Top 20 20-day take-profit rate >= 20%.
- Test Top 20 20-day average return >= 3%.

If V4 improves return but violates the risk constraints, it should remain an experiment and V3 0.7 stays the main model.
