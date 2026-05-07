# Phase 7 Trade-Day Gate Design

Date: 2026-05-07

## 1. Objective

Phase 7 builds an independent daily market gate for technical trading.

The model answers one question:

```text
After observing trade_date close, is the next trading day suitable for technical buy entries?
```

It does not rank individual stocks. It produces a date-level buy-day risk signal that can later be combined with:

- Phase 2 stock-level risk exclusion.
- Phase 4 stock-level return ranking.
- Pattern recognition as an evaluation slice or candidate source.

Phase 7 v1 must remain independent from Phase 2 and Phase 4 during training and validation. Combination testing is a later step.

## 2. Scope

In scope:

- Build one row per trading day.
- Build market-level technical features from local daily stock data.
- Train and validate a buy-day risk classifier.
- Save a deployable model artifact.
- Predict the trade permission for a requested date.

Out of scope for v1:

- No integration into `daily-screening`.
- No pattern dependency.
- No Phase 2 or Phase 4 features as model inputs.
- No intraday, minute, tick, or order-book data.
- No final portfolio backtest with transaction constraints.

## 3. Architecture

Phase 7 has four layers:

```text
local stock daily bars
  -> synthetic market and breadth features
  -> trade-day label builder
  -> walk-forward validation / final artifact training / daily prediction
```

Implementation should use a new module:

```text
src/stocks_analyzer/full_market_trade_day.py
```

The module should not import or call the old V4/V4.2 opportunity ranker logic. The old opportunity ranker is useful prior art, but Phase 7 is a new standalone phase.

## 4. Dataset

Sample grain:

```text
one sample = one trade_date
```

The dataset should include only dates with enough stock coverage to represent the market. Default minimum coverage:

```text
min_stock_count = 500
```

The primary market proxy should be generated locally instead of using external index APIs:

- `synthetic_equal_weight_index`
- `synthetic_amount_weight_index`
- market breadth fields from the same local full-market panel

The existing `build-synthetic-market` command and `reports/full_market_model/synthetic_market.csv` should be reused when available. If it is missing or does not cover the requested date range, Phase 7 may rebuild it from local `data/daily`.

## 5. Feature Set

Feature values for `trade_date` may only use information available at or before that date's close.

### 5.1 Market Proxy Trend

Use both equal-weight and amount-weight market proxies when present:

```text
index_return_1d
index_return_5d
index_return_20d
index_return_60d
index_above_ma20
index_above_ma60
index_above_ma120
index_ma20_slope_5d
index_ma60_slope_10d
index_drawdown_20d
index_drawdown_60d
index_volatility_20d
index_volatility_60d
```

Prefix amount-weight versions with `amount_weight_` where needed.

### 5.2 Market Breadth

Use cross-sectional stock states on each trade date:

```text
breadth_above_ma20
breadth_above_ma60
breadth_positive_return_5d
breadth_positive_return_20d
advancing_ratio_1d
declining_ratio_1d
new_high_20d_ratio
new_low_20d_ratio
new_high_60d_ratio
new_low_60d_ratio
```

### 5.3 Market Stress and Liquidity

Use daily cross-sectional distribution and liquidity features:

```text
limit_up_ratio
limit_down_ratio
amount_ratio_5d_20d
amount_ratio_20d_60d
cross_section_return_mean_1d
cross_section_return_median_1d
cross_section_return_dispersion_1d
cross_section_volatility_median_20d
cross_section_max_drawdown_median_20d
```

Limit-up and limit-down detection should use local A-share rules where possible. If exact limit detection is ambiguous, use a conservative proxy and record the deviation in config.

### 5.4 Feature Audit

Validation and training should write a feature audit:

```text
reports/full_market_model/trade_day_gate_feature_audit.csv
```

Audit fields:

```text
feature
missing_rate
mean
std
min
max
```

## 6. Labels

Phase 7 is a buy-day risk gate, so the primary label is a bad-buy-day label.

Default v1 label:

```text
bad_buy_day_5d = 1
if future_market_max_drawdown_5d <= -0.03
or future_market_return_5d <= -0.02
else 0
```

Where:

- `future_market_return_5d = market_proxy[t+5] / market_proxy[t] - 1`
- `future_market_max_drawdown_5d` is the minimum future return from `t+1` to `t+5` relative to `t`
- default market proxy is `synthetic_equal_weight_index`

Label grid for validation:

```text
horizon_days: 5, 10
drawdown_threshold: -0.02, -0.03, -0.05
return_threshold: -0.01, -0.02, -0.03
```

The first promoted model should use the best validation configuration, not a manually selected in-sample setting.

## 7. Models

Baseline models:

```text
always_allow
rule_market_gate
logistic_regression
linear_discriminant_analysis
naive_bayes
```

Primary candidate:

```text
lightgbm_classifier
```

`rule_market_gate` should be simple and auditable. Initial rule:

```text
allow if:
  synthetic_equal_weight_index > MA20
  and MA20 slope over 5 days > 0
  and breadth_above_ma20 >= 0.45
  and limit_down_ratio <= 0.03
else no_trade
```

The rule baseline must remain in all reports. The ML model is useful only if it improves on this baseline.

## 8. Walk-Forward Validation

Use trading-day walk-forward splits:

```text
train_days = 1000
valid_days = 250
step_days = 250
embargo_days = horizon_days
```

Validation command:

```powershell
python -m stocks_analyzer --project-root . validate-trade-day-gate --start-date 2015-01-01 --end-date 2026-05-07
```

Reports:

```text
reports/full_market_model/trade_day_gate_dataset.csv
reports/full_market_model/trade_day_gate_feature_audit.csv
reports/full_market_model/trade_day_gate_metrics.csv
reports/full_market_model/trade_day_gate_decile_report.csv
reports/full_market_model/trade_day_gate_filter_impact.csv
reports/full_market_model/trade_day_gate_config.json
```

Classification metrics:

```text
bad_day_rate
accuracy
bad_day_precision
bad_day_recall
bad_day_f1
roc_auc
pr_auc
pr_auc_baseline
brier
confusion matrix at threshold 0.5
```

Trading-impact metrics:

```text
allowed_day_coverage
blocked_day_coverage
bad_buy_day_rate_allowed
bad_buy_day_rate_blocked
future_return_mean_allowed
future_return_mean_blocked
future_max_drawdown_mean_allowed
future_max_drawdown_mean_blocked
max_consecutive_no_trade_days
```

Risk decile report:

```text
buy_day_risk_decile
rows
bad_buy_day_rate
avg_future_market_return
avg_future_market_max_drawdown
```

## 9. Promotion Criteria

Phase 7 passes only if all conditions hold out of sample:

- PR-AUC beats the bad-day-rate baseline in at least 70% of walk-forward windows.
- Highest predicted-risk decile has higher bad-buy-day rate than the lowest decile in at least 70% of windows.
- Blocking the highest-risk 20% or 30% of days improves future max drawdown.
- Blocking high-risk days does not materially worsen average future return.
- Allowed-day coverage remains at least 50%.
- `max_consecutive_no_trade_days` is reported and does not create an impractical gate unless explicitly accepted.

## 10. Training Artifact

Training command:

```powershell
python -m stocks_analyzer --project-root . train-trade-day-gate-model --start-date 2015-01-01 --end-date 2026-05-07
```

Artifact paths:

```text
data/ml/full_market_trade_day_gate/trade_day_gate_model.pkl
data/ml/full_market_trade_day_gate/trade_day_gate_model_metadata.json
```

Metadata must include:

```text
model_version
created_at
model_name
feature_columns
label_config
train_config
train_rows
train_start
train_end
bad_day_rate
selected_threshold
allowed_day_coverage_on_training
```

## 11. Prediction

Prediction command:

```powershell
python -m stocks_analyzer --project-root . predict-trade-day-gate --date 2026-05-07
```

Prediction output:

```text
reports/full_market_model/trade_day_gate_prediction_2026-05-07.csv
```

Columns:

```text
trade_date
buy_day_risk_score
selected_threshold
trade_permission
suggested_action
reason
model_name
model_version
feature columns used by the model
```

Decision mapping:

```text
trade_permission = allow
if buy_day_risk_score < selected_threshold

trade_permission = no_trade
if buy_day_risk_score >= selected_threshold
```

`suggested_action` should use:

```text
allow -> candidate_allowed
no_trade -> observation_only
```

This keeps Phase 7 semantically separate from stock-level `avoid`, which belongs to risk exclusion models such as Phase 2.

## 12. Later Combination Layer

Phase 7 v1 should not train on Phase 2 or Phase 4 scores.

After Phase 7 passes standalone validation, evaluate combinations:

```text
Phase 7 only
Phase 2 only
Phase 4 only
Phase 7 -> Phase 2
Phase 7 -> Phase 4
Phase 7 -> Phase 2 -> Phase 4
pattern only
pattern + Phase 7
pattern + Phase 7 + Phase 2 + Phase 4
```

The expected production decision order is:

```text
1. Phase 7 decides whether the next trading day is suitable for buy entries.
2. Phase 2 excludes high-risk stocks.
3. Phase 4 ranks remaining stocks by return score.
4. Pattern recognition can be used as an additional candidate slice.
```

## 13. Implementation Notes

- Reuse `reports/full_market_model/synthetic_market.csv` if present and sufficiently covered.
- Rebuild synthetic market from local `data/daily` if needed.
- Keep all Phase 7 reports under `reports/full_market_model`.
- Keep all Phase 7 artifacts under `data/ml/full_market_trade_day_gate`.
- Add progress logs for full-market feature construction.
- Do not change `daily-screening` in the first implementation pass.
- Do not delete or modify old V4/V4.2 opportunity ranker code.

## 14. Initial Implementation Order

1. Add trade-day dataset builder.
2. Add label builder and feature audit.
3. Add walk-forward validation and reports.
4. Add final training artifact command.
5. Add prediction command.
6. Run smoke tests with `--limit` or short date ranges.
7. Run full validation after smoke tests pass.

