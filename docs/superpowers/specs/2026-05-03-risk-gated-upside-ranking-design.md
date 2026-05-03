# Risk-Gated Upside Ranking Model Design

Date: 2026-05-03

## 1. Background

The first stacked multi-horizon trade-value model learned useful downside avoidance but did not improve upside selection enough. On the out-of-sample test split, Top 20 reduced the 20-day stop-loss rate and maximum drawdown, but it also lowered the 20-day take-profit rate versus the full test population.

That result suggests the current combined `trade_value` objective is too conservative. It mixes downside avoidance and upside selection into one target, so the model tends to prefer lower-risk stocks even when they have limited upside.

V2 separates the decision into two explicit questions:

- Is this stock too risky to consider?
- If risk is acceptable, which stock has the best upside?

This design remains independent from existing repository probability models, pattern recognition outputs, trend scores, watchlists, and TradingView aggregate scores.

## 2. Goal

Build a second-stage architecture that uses the existing four Stage 1 horizon models as shared inputs, then trains two separate fusion models:

- `risk_filter_model`: predicts unacceptable downside risk.
- `upside_ranker_model`: ranks upside potential only inside risk-acceptable samples.

The final ranking first applies a risk gate, then ranks the survivors by upside-adjusted score.

The main success criterion is not global accuracy. The model is useful only if validation and test Top 20 / Top 50 improve the balance of:

- 20-day take-profit rate.
- 20-day stop-loss rate.
- Average 20-day return.
- Average maximum drawdown.

## 3. Scope

Included in V2:

- A-share mainboard stocks only.
- Daily OHLCV and amount data.
- Existing V1 raw feature groups for M5, M10, M20, and M60.
- Existing Stage 1 horizon definitions:
  - M5: 5d +8% / -5%.
  - M10: 10d +12% / -6%.
  - M20: 20d +15% / -8%.
  - M60: 60d +30% / -15%.
- A risk-filter fusion model.
- An upside-ranking fusion model.
- Validation-set search over risk thresholds and risk penalties.
- Test-set report using the selected validation parameters.

Excluded in V2:

- Existing probability model outputs.
- Existing pattern recognition outputs.
- Existing trend or watchlist scores.
- TradingView aggregate scores or labels.
- Intraday data.
- Portfolio sizing, order execution, or automated trading.

## 4. Architecture

The V2 architecture is:

```text
Raw daily features
  -> Stage 1 horizon models: M5 / M10 / M20 / M60
      -> horizon probabilities and expected values
  -> Stage 2A: risk_filter_model
      -> risk_score, risk_tier, top_risk_horizon
  -> Stage 2B: upside_ranker_model
      -> upside_score, top_upside_horizon
  -> Decision layer
      -> risk gate, final_score, action
```

Stage 1 is shared. Stage 2A and Stage 2B are separate models with separate labels and separate evaluation.

## 5. Stage 1 Inputs

Stage 1 remains the four-horizon structure from V1.

Each horizon model outputs:

- `up_prob_Nd`
- `down_prob_Nd`
- `neutral_prob_Nd`
- `expected_value_Nd`
- `risk_adjusted_value_Nd`

Stage 2 models use only these out-of-fold Stage 1 outputs and simple cross-horizon derived features. They must not use Stage 1 in-sample predictions.

Recommended shared Stage 2 input features:

- All horizon `up_prob`, `down_prob`, `neutral_prob`.
- All horizon `expected_value` and `risk_adjusted_value`.
- `up_prob_20d - down_prob_20d`.
- `up_prob_60d - down_prob_60d`.
- `down_prob_20d + down_prob_60d`.
- `down_prob_5d + down_prob_10d`.
- `up_prob_20d * (1 - down_prob_60d)`.
- `up_prob_5d * up_prob_10d`.

## 6. Risk Filter Model

The risk-filter model answers one question: should this stock be excluded or heavily penalized because the downside risk is unacceptable?

### 6.1 Label

The V2 risk label is:

```text
bad_risk = 1 if:
  20d outcome is down
  or 60d outcome is down
  or 20d max_drawdown > 10%
  or 60d max_drawdown > 18%
else 0
```

The label intentionally uses both path outcomes and realized drawdown, because a stock can fail as a practical trade even if it does not exactly touch the fixed path stop.

### 6.2 Output

The model outputs:

- `risk_score`: probability of `bad_risk = 1`.
- `risk_tier`: low / medium / high.
- `top_risk_horizon`: the horizon contributing the most risk, based on Stage 1 down probabilities.

Initial risk tier mapping:

```text
low_risk    = risk_score <= selected_low_threshold
medium_risk = selected_low_threshold < risk_score <= selected_high_threshold
high_risk   = risk_score > selected_high_threshold
```

The thresholds are selected on the validation split, not hand-picked from the test split.

## 7. Upside Ranker Model

The upside model answers a separate question: among stocks that are not too risky, which ones have the best upside?

### 7.1 Training Population

The upside model trains only on risk-acceptable samples.

For V2, risk-acceptable samples are defined inside training folds using the realized `bad_risk` label:

```text
bad_risk = 0
```

This makes the upside model learn from historical cases where the risk profile was tolerable. During prediction, it receives model-predicted risk-gated candidates.

### 7.2 Label

The upside model uses a continuous `upside_value` label. Unlike V1 `trade_value`, this label does not include large downside penalties, because downside is handled by the risk model.

Initial definition:

```text
upside_value =
  0.20 * upside_5d
+ 0.25 * upside_10d
+ 0.35 * upside_20d
+ 0.20 * upside_60d
```

Where each `upside_Nd` rewards:

- hitting the upside target,
- stronger period return,
- faster upside trigger,
- higher maximum upside.

It may apply only a light drawdown discount to avoid rewarding highly unstable paths, but it must not duplicate the full downside penalty from the risk model.

Initial single-horizon upside value:

```text
upside_Nd =
  up_hit_bonus
+ final_return_bonus
+ max_upside_bonus
+ speed_bonus
- light_drawdown_discount
```

Suggested coefficients:

```text
up_hit_bonus           = 1.00 if outcome_Nd is up else 0.00
final_return_bonus     = 0.30 * clip(period_return / upside_target, -1, 1)
max_upside_bonus       = 0.20 * clip(max_upside / upside_target, 0, 1.5)
speed_bonus            = 0.20 * (1 - hit_day / horizon_days) if outcome_Nd is up else 0
light_drawdown_discount = 0.10 * min(max_drawdown / downside_threshold, 1.5)
```

### 7.3 Output

The model outputs:

- `upside_score`: predicted upside value.
- `top_upside_horizon`: the horizon with the largest positive contribution.

## 8. Decision Layer

The final score is not a raw model output. It is a deterministic decision-layer score.

Candidate handling:

```text
if risk_score > risk_threshold:
    action = avoid
else:
    action = candidate
```

Candidate ranking:

```text
final_score = upside_score - lambda * risk_score
```

The model should still output avoided names for diagnostics, but daily Top N ranking should be based on `candidate` rows first. Avoided names can be shown separately.

## 9. Validation Search

V2 should not fix the risk threshold or `lambda` upfront. It should choose them on the validation split.

Search space:

```text
risk_score thresholds: 0.20, 0.25, 0.30, 0.35, 0.40
risk percentile gates: lowest-risk 20%, 30%, 40%, 50%
lambda: 0.25, 0.50, 0.75, 1.00
```

Each candidate setting is evaluated by daily Top 20 and Top 50 performance.

Primary validation objective:

```text
maximize Top20 average 20d return
subject to:
  Top20 stop_loss_rate_20d <= full validation stop_loss_rate_20d
  Top20 avg_max_drawdown_20d <= full validation avg_max_drawdown_20d
```

Tie-breakers:

1. Higher Top20 take-profit rate.
2. Lower Top20 stop-loss rate.
3. Better Top50 average 20d return.
4. More stable daily candidate count.

After selecting parameters on validation, the test split is evaluated once.

## 10. Leakage Control

The same anti-leakage rule from V1 applies.

Training flow:

1. Split data chronologically.
2. Train Stage 1 models on earlier folds.
3. Generate Stage 1 out-of-fold predictions for unseen folds.
4. Train `risk_filter_model` only on Stage 1 out-of-fold predictions.
5. Train `upside_ranker_model` only on risk-acceptable rows from Stage 1 out-of-fold predictions.
6. Select risk gate and `lambda` on validation.
7. Evaluate final selected settings on test.
8. For live prediction, train deployment Stage 1 models on all eligible history and apply the frozen Stage 2 models and selected decision parameters.

Random train/test splitting is not allowed.

## 11. Reports

Training should save:

- `risk_filter_metrics.csv`
- `upside_ranker_metrics.csv`
- `risk_gate_grid.csv`
- `v2_split_metrics.csv`
- `v2_topn_metrics.csv`
- `v2_predictions_YYYY-MM-DD.csv`

Prediction output should include:

- `rank`
- `symbol`
- `name`
- `trade_date`
- `risk_score`
- `risk_tier`
- `upside_score`
- `final_score`
- `action`
- `top_risk_horizon`
- `top_upside_horizon`
- all Stage 1 horizon probabilities
- key explanation fields

## 12. Evaluation

Required evaluation:

- Full split metrics for train OOF, validation, and test.
- Top 20 and Top 50 metrics after the selected risk gate.
- Metrics for excluded high-risk rows to verify risk filtering.
- Validation grid report showing how each threshold/lambda pair performed.

Core reported metrics:

- Direction accuracy for `bad_risk`.
- Bad-risk rate by risk bucket.
- 20d take-profit rate.
- 20d stop-loss rate.
- 20d neutral rate.
- Average 20d return.
- Average 20d maximum drawdown.
- Average take-profit return.
- Average stop-loss return.
- Candidate count per day.

## 13. Testing Strategy

Unit tests:

- `bad_risk` label from path down and drawdown conditions.
- `upside_value` calculation.
- Risk-tier assignment.
- Final score calculation.
- Validation grid selection respects constraints.

Integration tests:

- Small universe trains Stage 1 OOF predictions.
- Risk model trains from OOF predictions.
- Upside model trains only from risk-acceptable OOF rows.
- Prediction report contains risk, upside, final score, action, and horizon fields.

Regression tests:

- Stage 2 refuses in-sample Stage 1 predictions.
- Test split is not used for threshold selection.
- Missing Stage 1 artifacts fail clearly.

## 14. V2 Completion Criteria

V2 is complete when:

- Risk and upside labels are generated.
- Risk filter and upside ranker train from Stage 1 OOF outputs.
- Validation grid selects risk gate and `lambda`.
- Test metrics are saved using the selected validation setting.
- Daily prediction produces candidate and avoid lists.
- Final report clearly states whether V2 improves over V1 on Top20/Top50 stop-loss, take-profit, average return, and drawdown.
