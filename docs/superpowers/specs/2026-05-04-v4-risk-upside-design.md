# V4 Risk-Upside Design

Date: 2026-05-04

## 1. Goal

Build a new V4 model family that separates risk prediction from upside prediction.

The current V3/V3.1 structure shares the same 5/10/20/60-day Stage 1 outputs, then uses Stage 2 for risk gating, clean-win ranking, and buy-trigger scoring. V4 will split these responsibilities:

- V4 Risk Model predicts only downside risk.
- V4 Long-Upside Model predicts only medium/long-term return quality.
- The final decision first applies a hard risk gate, then ranks only the surviving stocks by long-upside score.

This design treats short horizons as useful for risk detection, while treating 20/60-day horizons as more suitable for return prediction.

## 2. Naming

This is not the existing `v4_alpha_ranker` experiment. Use a separate model family:

- Model directory: `data/ml/v4_risk_upside/`
- Report directory: `reports/v4_risk_upside/`
- Model kind: `v4_risk_upside`

The existing `predict_model` daily interface can later point to this model after validation, but this design does not require changing the daily interface first.

## 3. Architecture

```text
daily feature frame
  |
  +-- V4 Risk Model
  |     Stage 1: M5 / M10 / M20 / M60
  |     Stage 2: risk_score only
  |     Output: risk_action = pass / block
  |
  +-- V4 Long-Upside Model
        Stage 1: M20 / M60 only
        Stage 2: long_upside_score only
        Output: long_upside_rank

final decision:
  technical hard exclusions
  -> V4 risk gate
  -> rank by long_upside_score
```

Risk and upside are trained separately. A high upside score cannot override the risk gate.

## 4. V4 Risk Model

The risk model keeps the existing four-horizon Stage 1 structure because 5/10-day signals are useful for detecting near-term downside.

Risk Stage 1:

| Model | Horizon | Label |
|---|---:|---|
| M5 | 5 trading days | up / down / neutral |
| M10 | 10 trading days | up / down / neutral |
| M20 | 20 trading days | up / down / neutral |
| M60 | 60 trading days | up / down / neutral |

The current stop/take-profit definitions remain:

| Horizon | Take Profit | Stop Loss |
|---|---:|---:|
| 5d | +8% | -5% |
| 10d | +12% | -6% |
| 20d | +15% | -8% |
| 60d | +30% | -15% |

Risk Stage 2 input:

- Stage 1 probabilities for 5/10/20/60.
- Weighted downside probability.
- Existing technical, moving average, volume, amount, ATR, MACD, RSI, KDJ, and block-return features.

Risk target:

```text
bad_risk = 1 if any condition is true:
  outcome_5d == down
  outcome_10d == down
  outcome_20d == down
  outcome_60d == down
  max_drawdown_20d > 10%
  max_drawdown_60d > 18%
```

Risk output:

- `risk_score`
- `down_prob_5d`
- `down_prob_10d`
- `down_prob_20d`
- `down_prob_60d`
- `stage2_weighted_down_prob`
- `top_risk_horizon`
- `risk_gate_reason`
- `risk_action`

Initial risk gate selection should optimize validation Top 20 stop-loss rate, bad-risk rate, and average maximum drawdown. It should prefer lower risk even if the selected count is smaller.

## 5. V4 Long-Upside Model

The long-upside model ignores 5/10-day training targets. It only learns 20/60-day return quality.

Long-Upside Stage 1:

| Model | Horizon | Label |
|---|---:|---|
| M20 | 20 trading days | up / down / neutral or clean value |
| M60 | 60 trading days | up / down / neutral or clean value |

This model should not consume M5/M10 Stage 1 outputs. Raw short-term technical features may remain only as current-state timing context, but the labels and Stage 1 model outputs are strictly 20/60-only.

Long-upside target:

```text
long_upside_value =
  0.40 * clean_value_20d
  + 0.60 * clean_value_60d
```

The higher 60-day weight reflects the desired bias toward stable medium/long-term continuation.

Clean value should reward:

- Hitting take-profit before stop-loss.
- Higher final return.
- Higher maximum upside.
- Faster take-profit.

Clean value should penalize:

- Stop-loss hit.
- Large maximum drawdown.
- Unstable path with high upside but also high drawdown.
- Severe overheat near the entry point.

Long-upside output:

- `long_upside_score`
- `long_upside_rank_pct`
- `up_prob_20d`
- `up_prob_60d`
- `top_upside_horizon`

## 6. Final Decision

The final V4 decision is a two-step process:

```text
candidate pool:
  full market or pattern-matched stocks

hard exclusions:
  MACD dead cross
  MACD top divergence
  bearish volume-price divergence
  existing technical risk flags

risk gate:
  risk_action == pass
  risk_score <= selected threshold
  down_prob_20d <= selected threshold
  down_prob_60d <= selected threshold
  stage2_weighted_down_prob <= selected threshold

ranking:
  sort by long_upside_score descending
```

The final score should not mix risk and upside into one blended number. Risk is an admission gate. Upside is the ranking signal after admission.

## 7. Training And Evaluation

Training flow:

1. Build the existing stacked dataset from local daily bars.
2. Train Risk Stage 1 models with 5/10/20/60 labels.
3. Train Risk Stage 2 model on out-of-fold Stage 1 predictions.
4. Train Long-Upside Stage 1 models with 20/60 labels only.
5. Train Long-Upside Stage 2 model on out-of-fold 20/60 Stage 1 predictions.
6. Select risk gate thresholds on validation data.
7. Evaluate Top N results on train, validation, and test splits.

Evaluation should report:

- Selected count.
- Win rate.
- Take-profit rate.
- Stop-loss rate.
- Average return.
- Median return.
- Average maximum upside.
- Average maximum drawdown.
- Bad-risk rate.
- Average `risk_score`.
- Average `long_upside_score`.

Primary comparison:

- Existing V3.1 / current `predict_model`.
- New V4 Risk-Upside.

The key success criterion is not raw probability accuracy. The key criterion is whether low-risk Top 20/Top 50 selections improve win rate, stop-loss rate, and return/drawdown quality.

## 8. Prediction Contract

V4 prediction output should include:

- `rank`
- `trade_date`
- `symbol`
- `name`
- `model_version`
- `risk_action`
- `risk_gate_reason`
- `risk_score`
- `long_upside_score`
- `long_upside_rank_pct`
- `final_action`
- `down_prob_5d`
- `down_prob_10d`
- `down_prob_20d`
- `down_prob_60d`
- `up_prob_20d`
- `up_prob_60d`
- `top_risk_horizon`
- `top_upside_horizon`

`predict_model` can later map this output into `reports/predict_model/predictions_<date>.csv` once V4 is validated.

## 9. Scope

In scope:

- New V4 risk-upside train and predict commands.
- New model/report directories.
- Separate risk and long-upside artifacts.
- Validation/test comparison against the current main model.
- Reusing existing dataset, Stage 1 helpers, label builders, and evaluation helpers where possible.

Out of scope:

- Removing V3/V3.1.
- Replacing daily `predict_model` before V4 comparison is reviewed.
- Changing pattern-recognition rules.
- Changing intraday screening.
- Building strict walk-forward retraining automation.

## 10. Risks

The long-upside model may favor slow trend stocks and reduce short-term entry sharpness. The buy-trigger layer can be reintroduced later as a separate timing filter if V4 improves long-term quality but weakens entry timing.

The first implementation is still a historical split validation, not a live retrained walk-forward system. Its results should be interpreted as model research until the daily replacement decision is made.
