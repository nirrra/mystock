# Hard Risk Gate Clean-Win Ranking Model Design

Date: 2026-05-03

## 1. Background

V2 proved that the risk model has usable signal, but the final Top 20 ranking still allows too many risky names back into the list.

On the full out-of-sample test split:

- The full test population had a 20-day stop-loss rate of 29.72%.
- The risk-filtered candidate pool reduced the 20-day stop-loss rate to 17.06%.
- The final daily Top 20 increased the 20-day stop-loss rate to 32.34%.

This means the main bottleneck is no longer whether the model can identify risk. The bottleneck is the final decision layer: upside ranking can override the risk signal too easily.

V3 changes the decision order from soft risk penalty to hard risk admission:

```text
raw universe
  -> hard risk gate
  -> clean-win ranking
  -> risk-constrained validation selection
  -> daily Top N
```

The design remains independent from existing repository probability models, pattern-recognition outputs, trend scores, watchlists, and TradingView aggregate scores.

## 2. Goal

Build a V3 model that treats low risk as a hard entry condition, then ranks only the survivors by clean upside quality.

Primary target:

- Keep daily Top 20 20-day stop-loss rate below 25% on validation, and aim for the same behavior on test.
- Keep daily Top 20 average maximum drawdown below 6% when possible.
- Preserve daily Top 20 average 20-day return above 2.5%.
- Preserve daily Top 20 20-day take-profit rate near or above 25%.

The model should be judged by risk-adjusted trade usefulness, not by global classification accuracy.

## 3. Scope

Included in V3:

- A-share mainboard daily data.
- Existing V1/V2 Stage 1 horizon models: M5, M10, M20, M60.
- Existing technical and multi-block features used by V2.
- A hard risk gate selected only on the validation split.
- A new clean-win ranking target.
- A risk-constrained validation objective.
- Train, validation, test, Top N, and prediction reports.

Excluded in V3:

- Existing repository probability model outputs.
- Existing pattern-recognition outputs.
- Existing trend/watchlist scores.
- TradingView aggregate recommendation scores or labels.
- Intraday data.
- Position sizing, portfolio construction, and execution logic.

## 4. Architecture

V3 keeps the useful parts of V2 and changes the decision layer and ranking target.

```text
daily features
  -> Stage 1 horizon models
      -> up/down/neutral probabilities for 5d, 10d, 20d, 60d
  -> risk_filter_model
      -> risk_score and horizon risk fields
  -> hard risk gate
      -> candidate / avoid
  -> clean_win_ranker_model
      -> clean_win_score
  -> risk-constrained parameter selector
      -> selected gate and ranking parameters
  -> daily Top N report
```

The hard gate is evaluated before ranking. A stock that fails the gate is not eligible for Top N, even if its upside score is high.

## 5. Hard Risk Gate

The gate combines model-level and horizon-level risk constraints.

Initial candidate conditions:

```text
risk_score <= daily risk percentile threshold
and down_prob_20d <= selected down20 threshold
and down_prob_60d <= selected down60 threshold
and not (down_prob_20d and down_prob_60d are both high)
```

V3 initial implementation will not add a separate drawdown-prediction model. Drawdown control is enforced through the validation objective and through Stage 1 downside probabilities. A separate drawdown model can be added in a later version only if V3 still fails the drawdown target.

Initial search grid:

- Daily risk percentile: 25%, 30%, 35%, 40%, 50%.
- `down_prob_20d` maximum: 45%, 50%, 55%, 60%.
- `down_prob_60d` maximum: 25%, 30%, 35%, 40%.
- `risk_score` absolute maximum: no cap, 55%, 60%, 65%.

The final grid can be narrowed after the first smoke run, but test data must not influence selected thresholds.

If a trading day has too few candidates after hard filtering, the selector may use a fallback gate. The fallback must still be risk-first:

```text
relax daily risk percentile by one grid step
then re-apply down_prob_20d and down_prob_60d caps
```

It must not bypass all risk constraints.

## 6. Clean-Win Ranking Target

The V2 upside target rewards upside too directly. V3 uses `clean_win_value`, which rewards upside that arrives without unacceptable early downside.

Initial target:

```text
clean_win_value =
  0.15 * clean_5d
+ 0.25 * clean_10d
+ 0.40 * clean_20d
+ 0.20 * clean_60d
```

The 20-day horizon remains the largest component because the intended use case is swing-trading selection.

Single-horizon component:

```text
clean_Nd =
  up_hit_bonus
+ final_return_bonus
+ max_upside_bonus
+ speed_bonus
- downside_hit_penalty
- max_drawdown_penalty
- early_drawdown_penalty
- unstable_path_penalty
```

Initial coefficient intent:

- Reward hitting the upside target before the downside threshold.
- Reward stronger final return and higher maximum upside.
- Reward faster upside hits.
- Penalize hitting the downside threshold.
- Penalize maximum drawdown, especially if it occurs before meaningful upside.
- Penalize paths that require deep drawdown before recovering.

This target is different from pure upside ranking. It should prefer a lower-volatility 20-day gain over a stock that has high upside but frequently falls hard first.

## 7. Validation Objective

V3 selects parameters on validation by constraints first, return second.

For each gate/ranker parameter combination, compute daily Top 20 and Top 50 metrics. A combination is eligible only if validation Top 20 satisfies:

```text
stop_loss_rate_20d <= min(full_valid_stop_loss_rate_20d, 0.25)
bad_risk_rate <= full_valid_bad_risk_rate
avg_max_drawdown_20d <= full_valid_avg_max_drawdown_20d
selected_count >= 10 average daily names
```

Among eligible combinations, rank by:

```text
objective =
  avg_return_20d
+ 0.50 * take_profit_rate_20d
+ 0.25 * win_rate
- 0.75 * stop_loss_rate_20d
- 0.50 * avg_max_drawdown_20d
```

If no combination satisfies all constraints, choose the least-risk-violating combination first, then maximize the same objective. The report must flag this as `constraints_satisfied = false`.

The test split is evaluated once using the selected validation parameters.

## 8. Reporting

V3 reports must make the risk tradeoff visible.

Required report files:

- `v3_split_metrics.csv`
- `v3_topn_metrics.csv`
- `v3_gate_grid.csv`
- `v3_predictions_<date>.csv`
- `v3_model_metadata.json`

Required metrics:

- Rows and trading days.
- Win rate.
- 20-day take-profit rate.
- 20-day stop-loss rate.
- Neutral rate.
- Average 20-day return.
- Average take-profit return.
- Average stop-loss return.
- Average maximum drawdown.
- Bad-risk rate.
- Average risk score.
- Average clean-win score.
- Candidate count per day.
- Constraint pass/fail fields.

Prediction output should include:

- `symbol`, `name`, `rank`.
- `action`: `candidate` or `avoid`.
- `risk_score`.
- `clean_win_score`.
- `final_score`.
- `risk_gate_reason` for avoided stocks.
- `up_prob_20d`, `down_prob_20d`, `up_prob_60d`, `down_prob_60d`.
- `top_risk_horizon`, `top_upside_horizon`.

## 9. Testing

Unit and workflow tests should cover:

- Clean-win label generation.
- Hard gate behavior.
- Fallback gate behavior when too few names pass.
- Validation parameter selection without test leakage.
- Prediction artifact load and daily ranking output.
- CLI smoke tests for training and prediction commands.

Full verification should run:

```powershell
pytest tests\test_stacked_trade_value.py tests\test_cli.py -q
python -m stocks_analyzer --project-root . train-hard-risk-clean-win --max-iter 20 --top-n 20,50 --predict-date 2026-04-30
python -m stocks_analyzer --project-root . predict-hard-risk-clean-win --date 2026-04-30 --top-n 20
```

## 10. Acceptance Criteria

V3 is considered an improvement over V2 if the test Top 20 result meets at least three of these four conditions:

- 20-day stop-loss rate is lower than V2 Top 20.
- 20-day average maximum drawdown is lower than V2 Top 20.
- 20-day average return is at least 2.5%.
- 20-day take-profit rate is at least 25%.

V3 is considered not production-ready if Top 20 stop-loss remains above 30% or if the hard gate reduces candidate count so much that daily selection becomes unstable.
