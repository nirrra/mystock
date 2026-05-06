# Event Risk Ranker Design

Date: 2026-05-06

## 1. Context

The current daily screening workflow already produces market updates, technical indicators, TradingView-style scores, model predictions, MACD/ATR reports, and pattern-recognition results. The next model direction intentionally ignores the existing TradingView score and all existing prediction-model designs.

The retained component is pattern recognition. A pattern match is treated as a trade event candidate, not as a final recommendation. The new module answers one practical trading question:

> Given a pattern event that can be acted on at the next open, is its expected risk/reward high enough to trade, and how should it rank against other same-day events?

The primary objective is risk/reward quality, measured in realized R multiples. Win rate is a supporting metric, not the optimization target.

## 2. Evidence Base

The design follows methods with broad paper or practitioner support:

- Event labeling with triple-barrier and meta-labeling, popularized by Marcos Lopez de Prado, for deciding whether an existing signal should be traded.
- Cross-sectional learning-to-rank for stock selection, including listwise ranking work on China A-share portfolios.
- Momentum, liquidity, and volatility features, which repeatedly appear as important predictors in empirical asset-pricing machine-learning research.
- Market-regime and volatility-aware exposure control, consistent with time-series momentum and volatility-managed portfolio evidence.

References:

- Gu, Kelly, and Xiu, "Empirical Asset Pricing via Machine Learning", Review of Financial Studies, 2020.
- Moskowitz, Ooi, and Pedersen, "Time Series Momentum", Journal of Financial Economics, 2012.
- Moreira and Muir, "Volatility-Managed Portfolios", Journal of Finance, 2017.
- Frazzini and Pedersen, "Betting Against Beta", Journal of Financial Economics, 2014.
- Zhang, Wu, and Chen, "ListFold: Learn-to-Rank for Stock Portfolios", arXiv, 2021.
- Lopez de Prado, "Advances in Financial Machine Learning", 2018.

## 3. Goals

Build a new `event_risk_ranker` model family that:

- Uses pattern matches as event candidates.
- Builds triple-barrier labels from next-open executable entries.
- Predicts downside event risk before ranking.
- Ranks only risk-passed events by expected R quality.
- Supports no-trade days when candidate quality or market regime is poor.
- Produces a separate `watchlist_event` output before any daily-screening replacement.

Primary metrics:

- Top10 and Top20 average realized R.
- Top10 and Top20 median realized R.
- Profit factor in R units.
- Stop-first rate.
- Take-profit-first rate.
- Average max drawdown in R units.
- Coverage days.
- Per-pattern stability.
- Walk-forward test-window stability.

Initial promotion thresholds versus `baseline_pattern_all`:

- Top20 average realized R improves by at least 25%.
- Top20 median realized R is above 0.
- Top20 stop-first rate is not higher than baseline.
- Top20 profit factor is above 1.2.
- Coverage days are at least 25%.
- Most walk-forward test windows have positive Top20 average realized R.

The full model must also not regress versus `baseline_rules` on Top20 average realized R, and its Top20 stop-first rate must not exceed `baseline_rules` by more than 2 percentage points.

## 4. Non-Goals

The first version does not:

- Use TradingView scores.
- Use any existing `predict-model`, V4, V5, or V5.1 model output.
- Predict absolute future return as the main target.
- Use minute-level data.
- Use deep learning or reinforcement learning.
- Replace the existing daily screening output immediately.
- Change pattern-recognition rules.

## 5. Event Definition

Each pattern match creates one event:

```text
event_id = signal_date + symbol + pattern_id
entry_date = first trading day after signal_date
entry_price = next_open
```

The next-open entry matches the current pattern-backtest execution assumption and avoids pretending that a signal found after daily screening can be executed at the same close.

Required event fields:

- `event_id`
- `symbol`
- `name`
- `pattern_id`
- `signal_date`
- `entry_date`
- `entry_price`
- `trigger_reason`
- `pattern_family`

Different patterns on the same stock and date remain separate training events. Prediction output later deduplicates by stock after scoring.

## 6. Triple-Barrier Labels

The label builder uses ATR-adjusted barriers:

```text
initial_risk = entry_price - stop_loss_price
ATR14 = value computed after signal_date close, never from entry_date
stop_loss_price = entry_price - stop_atr_mult * ATR14
take_profit_price = entry_price + take_atr_mult * ATR14
max_holding_days = 20 by default
realized_R = (exit_price - entry_price) / initial_risk
```

If ATR14 is missing, non-positive, or produces `initial_risk <= 0`, the event is skipped with `skip_reason = invalid_atr_or_risk`.

Initial parameter grid:

| Parameter | Candidates |
|---|---|
| `stop_atr_mult` | `1.0`, `1.2`, `1.5` |
| `take_atr_mult` | `2.0`, `2.5`, `3.0` |
| `max_holding_days` | `10`, `20`, `40` |

Barrier outcomes:

- `take_profit_first`
- `stop_loss_first`
- `timeout`

If a daily bar touches take-profit and stop-loss on the same day, label it as `stop_loss_first`. Daily data cannot know intraday order, so the conservative assumption avoids inflated results.

### A-Share Fillability And Limit Handling

Entry uses `next_open` only if the next-open trade is realistically fillable. If the entry day is a one-price locked limit-up bar, or otherwise detected as unfillable at the open, the event is marked:

```text
barrier_outcome = entry_unfillable
skip_reason = entry_unfillable_limit_up
```

`entry_unfillable` events are excluded from training and evaluation and written to `skipped_events.csv`. This prevents impossible next-open entries from inflating returns.

Normal stop-loss and take-profit exits use the barrier price when the barrier is executable. If the stop-loss day is locked limit-down and selling at the stop price is not realistically executable, the exit is recorded conservatively at the close or detected locked-limit price, whichever is worse for the trade. This can make `realized_R < -1`. Take-profit exits on limit-up bars may use the take-profit barrier price because selling into a limit-up market is generally executable.

Timeout exits use the close on the final holding day. Their R value is:

```text
timeout_R = (timeout_close - entry_price) / initial_risk
```

Path-risk quantities are computed only from bars in the actual holding window, from `entry_date` through `exit_date` inclusive:

```text
max_drawdown_R = max(0, entry_price - min(holding_window.low)) / initial_risk
max_upside_R = max(0, max(holding_window.high) - entry_price) / initial_risk
holding_days_penalty = min(1.0, holding_days / max_holding_days)
```

`max_drawdown_R` is computed even when no stop-loss is triggered, so late or unresolved drawdowns still penalize the rank label.

The default training labels are:

```text
risk_label = 1 if barrier_outcome == stop_loss_first or realized_R <= -0.8
expected_R_label = clipped(realized_R, -1.2, 3.0)
rank_value = realized_R - 0.4 * max_drawdown_R - 0.2 * holding_days_penalty
```

The `-0.8R` risk threshold is a defensive buffer before the nominal `-1R` stop. It catches timeout exits and locked-limit exits that did not formally hit the stop barrier but still produced materially bad risk/reward. For normal executable stop-loss events, `realized_R` is `-1.0`; locked limit-down exits may be worse than `-1.0`.

`rank_value` is converted within each `signal_date` into ordinal ranking grades:

Percentiles are computed within each `signal_date` from `rank_value` in ascending order, with larger `rank_value` considered better:

| Grade | Percentile range |
|---:|---|
| 4 | `[0.90, 1.00]` |
| 3 | `[0.70, 0.90)` |
| 2 | `[0.35, 0.70)` |
| 1 | `[0.15, 0.35)` |
| 0 | `[0.00, 0.15)` |

This makes the ranker optimize same-day candidate selection rather than all-market return regression.

If a `signal_date` has fewer than 5 eligible candidate events, that date is excluded from same-day relative ranker training. Those events can still train the risk classifier and appear in validation metrics.

## 7. Duplicate Signal Policy

For the same `symbol` and `pattern_id`, repeat signals inside the active holding window are not treated as independent trades.

Default:

```text
cooldown_trading_days = max_holding_days
```

The unit is trading days. This differs from the current `pattern_backtest.py` default of 5 trading days; `event_risk_ranker` ties cooldown to the active event horizon because overlapping events would share most of the same forward path.

The first event is kept. Later same-symbol, same-pattern events inside the cooldown window are labeled as `cooldown_skipped` and excluded from model training metrics.

When several pattern IDs hit the same stock on the same date, keep all event rows for training. Prediction output deduplicates by symbol and keeps the event with the highest final score while preserving merged `pattern_ids`.

For training, same-date multi-pattern rows for the same symbol are not treated as leakage because the pattern identity is part of the event. To avoid overweighting one stock's identical forward path, each row receives:

```text
sample_weight = 1 / count_events_for(signal_date, symbol)
```

The total same-day weight of one stock is therefore capped at 1 even if multiple patterns fire.

## 8. Feature Design

All features must satisfy:

```text
feature_date <= signal_date
```

No future-window labels, future returns, or post-signal path statistics may enter features.

### Pattern Event Features

- `pattern_id`
- `pattern_family`
- `pattern_recent_frequency_20d`
- `same_symbol_pattern_count_20d`
- `days_since_last_same_pattern`
- `multi_pattern_hit_count`

### Trend And Momentum Features

- `return_5d`
- `return_10d`
- `return_20d`
- `return_60d`
- `distance_to_ma20`
- `distance_to_ma60`
- `ma20_slope`
- `ma60_slope`
- `new_high_20d`
- `new_high_60d`

### Volatility And Risk Features

- `atr_pct`
- `realized_vol_20d`
- `realized_vol_60d`
- `max_drawdown_20d`
- `gap_pct`
- `intraday_range_pct`
- `down_day_count_10d`
- `limit_down_recent_count`

### Volume And Liquidity Features

- `amount_ma20`
- `avg_amount_20d`
- `amount_ratio_5d_20d`
- `volume_ratio_1d_20d`
- `price_volume_corr_20d`
- `up_volume_share_20d`
- `accumulation_days_20d`
- `distribution_days_20d`
- `turnover_proxy`
- `limit_up_recent_count`

### Market-Regime Features

Use only data available as of the signal date:

- `index_return_5d`
- `index_return_20d`
- `index_above_ma20`
- `index_above_ma60`
- `market_breadth_ma20`
- `market_breadth_ma60`
- `market_realized_vol_20d`
- `candidate_count_today`

Default market proxy:

- Primary index: CSI All Share if local data are available.
- Fallback index: CSI 300.
- Breadth universe: the project's current tradable stock universe after basic liquidity and data-availability filters, not only pattern candidates.

If index or breadth data are unavailable, keep the event but mark missing market features and use model fallback handling. Missing market-regime data should not silently become bullish.

## 9. Model Architecture

The module has three narrow layers.

### 9.1 Rule Baseline

The rule baseline is a non-ML benchmark:

```text
rule_score =
  trend_score
+ momentum_score
+ volume_confirmation_score
+ liquidity_score
+ market_regime_score
- volatility_penalty
- overheat_penalty
- gap_penalty
```

Initial formulas use simple bounded components so the baseline is reproducible:

```text
trend_score =
  0.5 * 1{close > ma20}
+ 0.5 * 1{ma20 > ma60}
+ 0.5 * 1{ma20_slope > 0}

momentum_score =
  clip(return_20d / 0.15, -1, 1)
+ 0.5 * clip(return_60d / 0.30, -1, 1)

volume_confirmation_score =
  0.5 * 1{amount_ratio_5d_20d >= 1.1}
+ 0.5 * 1{up_volume_share_20d >= 0.55}

liquidity_score =
  1{avg_amount_20d >= configured_min_avg_amount_20d}

market_regime_score =
  0.5 * 1{index_above_ma20}
+ 0.5 * 1{market_breadth_ma20 >= 0.45}

volatility_penalty =
  0.5 * 1{atr_pct > p80_by_date}
+ 0.5 * 1{realized_vol_20d > p80_by_date}

overheat_penalty =
  0.5 * 1{return_5d > 0.18}
+ 0.5 * 1{distance_to_ma20 > 0.18}

gap_penalty =
  1{abs(gap_pct) > 0.06}
```

These are starting formulas, not tuned research conclusions. Any later tuning must happen only on validation windows.

Outputs:

- `rule_risk_pass`
- `rule_score`
- `rule_reason`

If the ML model cannot beat this baseline out of sample, it is not promoted.

### 9.2 Meta-Label Risk Classifier

The risk model filters events likely to hit stop-loss first.

Primary model:

```text
HistGradientBoostingClassifier
```

Fallback:

```text
LogisticRegression
```

Input:

- Event, trend, volatility, liquidity, volume, and market-regime features.

Label:

```text
risk_label = 1 for bad-risk events
```

Outputs:

- `p_stop_first`
- `risk_score`
- `risk_pass`
- `risk_tier`
- `risk_reason`

The risk threshold is selected on validation data, not hard-coded. The first threshold objective is:

```text
score(t) =
  avg_realized_R(t)
- 0.5 * stop_first_rate(t)
- 1.0 * max(0, min_coverage - coverage_days_rate(t))

min_coverage = 0.25
```

Ties prefer the lower stop-first rate, then the higher median realized R.

### 9.3 Expected-R Ranker

The ranker scores only events with `risk_pass = true`.

Primary model:

```text
LightGBM LambdaRank
```

Fallback:

```text
HistGradientBoostingRegressor on daily rank percentile
```

For the fallback model, the ordinal rank grade maps to a regression target:

```text
rank_target = rank_grade / 4.0
```

Fallback training uses the same eligible dates as LambdaRank, including the minimum 5-event same-day requirement and the same sample weights.

Outputs:

- `expected_R_score`
- `rank_score`
- `rank_pct`

Initial fusion:

```text
final_score =
  0.65 * rank_pct
+ 0.25 * expected_R_score_pct
+ 0.10 * rule_score_pct
```

The validation process may select a different blend, including a pure ranker score, if it improves out-of-sample R metrics without degrading stop-first risk.

## 10. Market Opportunity Gate

The system must be able to output no-trade days.

The first version uses a simple gate derived from:

- Market trend features.
- Market breadth.
- Market volatility.
- Risk-passed candidate count.
- Median and upper-tail `final_score`.

The gate output is:

- `trade_permission = allow`
- `trade_permission = no_trade`

When `no_trade`, prediction reports still include scored events for diagnostics, but `watchlist_event` marks them as observation-only.

## 11. Training And Validation

Use walk-forward validation only. Do not use random splits.

Default window setup:

```text
train_days = 280
valid_days = 60
test_days = 60
windows = 8
embargo_days = max(max_holding_days_grid)
```

With the initial grid, `embargo_days = 40`. The embargo is fixed at the maximum grid horizon so all barrier parameter candidates share the same leakage-safe split. The embargo prevents samples whose forward label window overlaps a validation or test segment from leaking into training.

Each window trains:

1. Rule baseline parameters.
2. Risk classifier.
3. Risk threshold.
4. Ranker.
5. Optional score-fusion weights.
6. Market opportunity gate threshold.

Evaluation is always reported on the test segment for each window.

## 12. Baselines

Every validation report includes:

- `baseline_pattern_all`: all pattern events, no risk filter, sorted by pattern priority.
- `baseline_rules`: rule baseline filter plus rule-score ranking.
- `baseline_risk_only`: meta-label risk filter with simple expected-R or rule-score ranking.
- `event_risk_ranker`: full risk filter, ranker, and market gate.

The full model must beat both `baseline_pattern_all` and `baseline_rules` before integration promotion.

## 13. Metrics

TopN metrics are computed for Top10 and Top20:

- `avg_realized_R`
- `median_realized_R`
- `win_rate_R_positive`
- `stop_first_rate`
- `take_profit_first_rate`
- `timeout_rate`
- `avg_max_drawdown_R`
- `profit_factor_R`
- `coverage_days`
- `selected_events`
- `selected_symbols`
- `avg_holding_days`

`profit_factor_R` is:

```text
sum(positive_R) / abs(sum(negative_R))
```

If the denominator is 0, report `profit_factor_R = inf` and always show `selected_events` and `selected_symbols` beside it to avoid overreading tiny samples.

TopN metrics are reported in two scopes:

- `event`: raw event ranking.
- `symbol_dedup`: per `signal_date`, keep the highest-scored event per symbol before selecting TopN.

The `symbol_dedup` scope is the primary promotion metric because it better matches one-position-per-stock execution.

Reports also include per-pattern metrics:

- `pattern_id`
- event count
- average R
- median R
- stop-first rate
- take-profit-first rate
- selected share

## 14. Reports And Artifacts

Reports:

```text
reports/event_risk_ranker/
  event_labels_YYYY-MM-DD_YYYY-MM-DD.csv
  event_features_YYYY-MM-DD_YYYY-MM-DD.csv
  skipped_symbols.csv
  skipped_events.csv
  walkforward_windows.csv
  walkforward_topn_metrics.csv
  walkforward_summary.csv
  threshold_grid.csv
  feature_importance.csv
  predictions_YYYY-MM-DD.csv
```

`event_labels_*.csv` must include at least:

- `event_id`
- `symbol`
- `name`
- `pattern_id`
- `signal_date`
- `entry_date`
- `entry_price`
- `stop_loss_price`
- `take_profit_price`
- `max_holding_days`
- `barrier_outcome`
- `exit_date`
- `exit_price`
- `realized_R`
- `max_drawdown_R`
- `max_upside_R`
- `holding_days`
- `holding_days_penalty`
- `cooldown_skipped`
- `skip_reason`

Artifacts:

```text
data/ml/event_risk_ranker/
  event_risk_ranker.pkl
  event_risk_ranker_metadata.json
```

Daily watchlist output:

```text
reports/watchlists/watchlist_event_YYYY-MM-DD.json
```

Minimum `watchlist_event` schema:

```json
{
  "trade_date": "YYYY-MM-DD",
  "model_version": "event_risk_ranker_v1",
  "trade_permission": "allow",
  "candidate_count": 0,
  "items": [
    {
      "symbol": "000001",
      "name": "example",
      "pattern_id": "1",
      "pattern_ids": "1,3",
      "final_score": 0.0,
      "expected_R_score": 0.0,
      "p_stop_first": 0.0,
      "suggested_action": "candidate",
      "entry_price_ref": 0.0,
      "stop_loss_price_ref": 0.0,
      "take_profit_price_ref": 0.0,
      "max_holding_days": 20,
      "risk_reason": "passed"
    }
  ],
  "warnings": []
}
```

## 15. Prediction Contract

Daily predictions write:

```text
reports/event_risk_ranker/predictions_YYYY-MM-DD.csv
```

Required columns:

- `trade_date`
- `symbol`
- `name`
- `pattern_id`
- `pattern_ids`
- `event_id`
- `rule_score`
- `risk_pass`
- `risk_tier`
- `p_stop_first`
- `expected_R_score`
- `rank_score`
- `final_score`
- `trade_permission`
- `suggested_action`
- `risk_reason`
- `entry_price_ref`
- `stop_loss_price_ref`
- `take_profit_price_ref`
- `max_holding_days`
- `model_version`

The `suggested_action` values are:

- `candidate`
- `observe`
- `avoid`

`candidate` requires `risk_pass = true` and `trade_permission = allow`.
`observe` is used when `risk_pass = true` but `trade_permission = no_trade`.
`avoid` is used when `risk_pass = false` or the event has insufficient history.

## 16. CLI Design

Add commands:

```bash
python -m stocks_analyzer --project-root . build-event-labels --start-date 2022-01-01 --end-date 2026-04-30
python -m stocks_analyzer --project-root . train-event-risk-ranker --windows 8 --top-n 10,20 --stop-atr-grid 1.0,1.2,1.5 --take-atr-grid 2.0,2.5,3.0 --holding-days-grid 10,20,40
python -m stocks_analyzer --project-root . predict-event-risk-ranker --date 2026-05-06 --top-n 20
python -m stocks_analyzer --project-root . validate-event-risk-ranker --windows 8 --top-n 10,20 --holding-days-grid 10,20,40
```

The example range is chosen to support the default 8-window setup. If the available local trading dates are insufficient, validation generates fewer valid windows and records that in `walkforward_windows.csv` and `walkforward_summary.csv`.

The train command may build labels and features internally if cached files are missing.

## 17. Daily Screening Integration

Phase 1 runs in parallel and does not replace existing outputs:

```text
update
pattern
event-risk-ranker prediction
watchlist_event
```

The new output is:

```text
reports/watchlists/watchlist_event_YYYY-MM-DD.json
```

The existing `watchlist` and `watchlist_pattern` files remain unchanged.

Phase 2, only after promotion criteria pass, changes daily screening to:

```text
update
macd
atr
pattern
event-risk-ranker
watchlist_event
```

At that point daily screening can remove TradingView and old `predict-model` dependencies.

## 18. Error Handling

If the model artifact is missing:

- `predict-event-risk-ranker` raises a clear error telling the user to train first.
- `daily-screening` phase 1 skips `watchlist_event`, records a warning, and does not fail the whole daily run.

If a symbol lacks enough historical data:

- Minimum required history before `signal_date` is 120 trading days.
- Training skips the event and records the symbol-level issue in `skipped_symbols.csv` and event-level issue in `skipped_events.csv`.
- Prediction keeps the raw pattern row for diagnostics, marks `risk_reason = insufficient_history`, and sets `suggested_action = avoid`.

If label construction lacks enough future bars:

- Training skips the event with `skip_reason = insufficient_forward_bars`.
- Prediction is unaffected because future bars are never required at prediction time.

If market-regime features are missing:

- Prediction continues with missing-value handling.
- The report includes `market_features_available = false`.
- The opportunity gate must not treat missing market data as bullish.

## 19. Testing

Focused tests:

- Triple-barrier labeling.
- Conservative handling when take-profit and stop-loss are both touched in one daily bar.
- Entry-unfillable limit-up handling.
- Locked limit-down stop-exit handling.
- Timeout R calculation.
- `max_drawdown_R` and `holding_days_penalty` calculation.
- Cooldown deduplication.
- Same-symbol multi-pattern sample weights.
- Feature date checks preventing future leakage.
- Walk-forward window generation with embargo.
- TopN R metrics, symbol-dedup scope, and profit factor denominator-zero handling.
- Rule-baseline output shape.
- Risk-threshold selection.
- Ranker fallback path when LightGBM is unavailable.
- CLI parser for the four new commands.
- Prediction output contract.
- `watchlist_event` construction.
- Daily-screening phase-1 skip behavior when the model artifact is missing.

## 20. Scope For First Implementation Plan

The first implementation should be split into isolated modules:

- `event_labels.py`: event construction, triple-barrier labels, cooldown handling.
- `event_features.py`: leakage-safe feature construction.
- `event_risk_ranker.py`: rule baseline, risk model, ranker, validation, prediction contract.
- `event_watchlist.py`: `watchlist_event` construction.
- CLI wiring and focused tests.

Keep old model code untouched except for optional daily-screening integration hooks in phase 1.

## 21. Approval

Approved design direction:

- Optimize risk/reward, not raw return prediction.
- Use pattern events as the only retained signal source.
- Use triple-barrier labels and meta-label risk screening.
- Use same-day cross-sectional ranking for risk-passed events.
- Run in parallel first and promote only after walk-forward validation beats the rule and pure-pattern baselines.
