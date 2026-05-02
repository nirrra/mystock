# Stacked Multi-Horizon Trade Value Model Design

Date: 2026-05-03

## 1. Background

This design defines a new stock decision-support model for A-share mainboard stocks.

The model is intentionally independent from the existing probability model, pattern recognition logic, rule-based scores, and prior strategy outputs in this repository. Existing model results, pattern labels, trend scores, and TradingView aggregate scores must not be used as features, labels, filters, or priors.

The central idea is to separate four horizons:

- 5 trading days: short-term entry timing.
- 10 trading days: short-term momentum continuation.
- 20 trading days: primary swing-trading value.
- 60 trading days: medium-term trend space and large-risk control.

The model uses a two-stage stacking architecture. Stage 1 trains one model per horizon. Stage 2 learns a continuous trade-value score from the out-of-fold outputs of those Stage 1 models.

## 2. Goals

The system should support both selection and decision assistance:

- Rank all eligible A-share mainboard stocks each day.
- Identify whether a top-ranked stock has enough trade value to consider buying.
- Separate upside opportunity from downside risk.
- Show whether the final score is mainly supported by 5-day timing, 10-day continuation, 20-day swing value, or 60-day trend quality.

The main output is `trade_value_pred`, a continuous score where higher means better expected trading value after accounting for upside, downside, drawdown, and speed.

## 3. Scope

Included in V1:

- A-share mainboard stock universe only.
- Daily OHLCV and amount data.
- Moving-average, return, volume, amount, volatility, drawdown, and low-level technical-indicator features.
- Four Stage 1 horizon models: 5d, 10d, 20d, 60d.
- One Stage 2 fusion model trained on Stage 1 out-of-fold predictions.
- Daily Top N ranking and explanation fields.
- Strict time-series validation.

Excluded in V1:

- Existing repository probability model outputs.
- Existing pattern recognition outputs.
- Existing trend score or watchlist score outputs.
- TradingView aggregate scores, labels, or recommendations.
- Intraday data.
- Automatic trading, order placement, or position sizing.
- Non-mainboard stocks.

## 4. Label Definition

Signals are generated after the close of day `t`. Entry price is the next trading day open, `t+1 open`.

Each horizon uses path-based labels. The path window starts from the entry day.

| Horizon | Upside target | Downside threshold | Purpose |
|---|---:|---:|---|
| 5d | +8% | -5% | Entry timing |
| 10d | +12% | -6% | Momentum continuation |
| 20d | +15% | -8% | Primary swing value |
| 60d | +30% | -15% | Trend space and large-risk control |

For each horizon:

- `up`: price hits the upside target before the downside threshold.
- `down`: price hits the downside threshold before the upside target.
- `neutral`: neither side is hit within the horizon.
- `conflict`: the same daily bar hits both upside and downside. V1 excludes these rows because daily data cannot determine which happened first.

## 5. Single-Horizon Value

Each horizon also produces a continuous realized value, `value_Nd`, rather than only a class label.

The exact coefficients may be tuned during implementation, but V1 starts from this fixed structure:

```text
value_Nd =
  outcome_score
+ final_return_score
- drawdown_penalty
+ speed_adjustment
```

Initial outcome scores:

```text
up      = +1.00
down    = -1.20
neutral =  0.00
```

Initial modifiers:

```text
final_return_score = 0.30 * clip(period_return / upside_target, -1, 1)
drawdown_penalty   = 0.20 * min(max_drawdown / downside_threshold, 2)
up_speed_bonus     = 0.20 * (1 - hit_day / horizon_days)
down_speed_penalty = -0.30 * (1 - hit_day / horizon_days)
```

This makes early upside better than late upside, early downside worse than late downside, and high-drawdown paths worse even when they eventually recover.

## 6. Stage 2 Target

The Stage 2 fusion model trains on a continuous `trade_value` label:

```text
trade_value =
  0.10 * value_5d
+ 0.20 * value_10d
+ 0.40 * value_20d
+ 0.30 * value_60d
```

The weight order is intentional:

1. 20d is largest because the main use case is swing-trading value.
2. 60d is second because medium-term trend quality and large-risk control should strongly affect whether a trade is worth taking.
3. 10d confirms momentum continuation.
4. 5d confirms entry timing and should not dominate because short-term noise is high.

## 7. Stage 1 Models

Stage 1 contains four independent models.

### 7.1 M5

Purpose: predict 5-day path outcome and short-term entry timing.

Target: 5d +8% / -5%.

Main feature groups:

- 1d, 2d, 3d, and 5d returns.
- Daily candle body, range, upper shadow, lower shadow, and gap.
- Price distance to MA5.
- MA5 slope and acceleration.
- 1d, 3d, and 5d volume and amount changes.
- 5d volatility and drawdown.
- Short-term overheat variables such as 5d return and consecutive up days.
- Low-level technical indicators at the daily/short horizon.

### 7.2 M10

Purpose: predict 10-day momentum continuation.

Target: 10d +12% / -6%.

Main feature groups:

- 5d and 10d returns.
- Price distance to MA5 and MA10.
- MA5 and MA10 slopes.
- MA5-MA10 distance and distance change.
- 10d range position.
- 10d volume, amount, volume ratio, and amount ratio.
- 10d volatility and drawdown.
- Low-level technical indicators at the short-to-medium horizon.

### 7.3 M20

Purpose: predict primary swing-trading value.

Target: 20d +15% / -8%.

Main feature groups:

- 10d and 20d returns.
- Price distance to MA20.
- MA20 slope and stability.
- 20d range position and distance to 20d high/low.
- 20d volume and amount structure.
- 20d volatility and drawdown.
- Multiple 20d block features:
  - Recent 20d return, volume, amount, drawdown, and volatility.
  - Prior 20d return, volume, amount, drawdown, and volatility.
  - Third-prior 20d return, volume, amount, drawdown, and volatility.
  - Ratios between adjacent 20d blocks.
- Continuous variables for over-extension and pullback repair.

### 7.4 M60

Purpose: predict medium-term trend space and large downside risk.

Target: 60d +30% / -15%.

Main feature groups:

- 20d, 40d, and 60d returns.
- Price distance to MA20 and MA60.
- MA60 slope and MA60 slope changes over 5d, 10d, and 20d.
- MA20-MA60 relative position and distance change.
- 60d and 120d range position.
- Distance to 60d and 120d high/low.
- 60d volatility and drawdown.
- Three 60d block features:
  - `block60_0`: latest 0-60 trading days.
  - `block60_1`: prior 60-120 trading days.
  - `block60_2`: prior 120-180 trading days.
  - For each block: return, volume change, amount change, drawdown, and volatility.
  - Ratios between adjacent blocks for volume and amount.

The M60 model should learn whether the current structure is early launch, trend continuation, high-level exhaustion, or repair after a prior decline.

## 8. Technical Indicator Features

The model may use low-level technical indicators that are commonly included in TradingView-style calculations, including:

- MACD, signal line, histogram, and histogram slope.
- RSI with multiple windows.
- KDJ or stochastic K/D/J values.
- CCI.
- ADX and directional indicators.
- Williams %R.
- Bollinger Band width and price position.
- ATR and ATR percentage.

These indicators must be used only as raw numeric features.

The model must not use:

- TradingView aggregate score.
- TradingView MA score.
- TradingView oscillator score.
- TradingView buy/sell labels.
- Any existing repository rule score or pattern output.

## 9. Stage 2 Fusion Model

Stage 2 input should be limited to Stage 1 model outputs and simple derived cross-horizon relationships.

Required Stage 1 outputs per horizon:

- `up_prob_Nd`
- `down_prob_Nd`
- `neutral_prob_Nd`
- `expected_value_Nd`
- `risk_adjusted_value_Nd`

Recommended derived Stage 2 features:

- `up_prob_20d - down_prob_20d`
- `up_prob_60d - down_prob_60d`
- `down_prob_5d + down_prob_10d`
- `up_prob_20d * (1 - down_prob_60d)`
- `up_prob_5d * up_prob_10d`
- `down_prob_60d` as a large-risk suppressor.

Stage 2 output:

- `trade_value_pred`
- `buy_score`
- `down_risk_score`
- `trend_quality_score`
- `timing_score`

The derived user-facing scores are deterministic transformations of model outputs. The primary trained output remains `trade_value_pred`.

## 10. Leakage Control

Stage 2 must never train on Stage 1 predictions generated from the same samples that Stage 1 saw during training.

V1 must use walk-forward or time-series out-of-fold prediction:

1. Split history into chronological folds.
2. For each fold, train Stage 1 models only on earlier data.
3. Predict Stage 1 outputs for the next unseen fold.
4. Concatenate all unseen-fold Stage 1 outputs.
5. Train Stage 2 only on these out-of-fold Stage 1 outputs.
6. For live prediction, train Stage 1 on all eligible historical data, predict current-day Stage 1 outputs, then apply the frozen Stage 2 model.

Random train/test splitting is not allowed.

## 11. Daily Output

For each prediction date, the system should output a ranked table with:

- `rank`
- `symbol`
- `name`
- `trade_date`
- `trade_value_pred`
- `buy_score`
- `down_risk_score`
- `trend_quality_score`
- `timing_score`
- `up_prob_5d`, `down_prob_5d`, `neutral_prob_5d`
- `up_prob_10d`, `down_prob_10d`, `neutral_prob_10d`
- `up_prob_20d`, `down_prob_20d`, `neutral_prob_20d`
- `up_prob_60d`, `down_prob_60d`, `neutral_prob_60d`
- Top contributing horizon.
- Highest-risk horizon.
- Key raw explanation fields, such as MA20 distance, MA60 distance, latest 20d return, latest 60d return, latest 60d block volume/amount ratios, MACD histogram, RSI, ATR percentage.

## 12. Evaluation

Evaluation should focus on whether the ranking has trading value, not only classification metrics.

Required evaluation:

- Top 20 and Top 50 average future return.
- Top 20 and Top 50 downside-trigger rate.
- Top 20 and Top 50 maximum drawdown.
- Score-bucket monotonicity.
- Performance stability by year, quarter, and market regime.
- Compare Top N against random mainboard selection.
- Compare high-score bucket against low-score bucket.
- Evaluate whether high `down_risk_score` names actually have higher downside-trigger rates.

Useful diagnostics:

- Stage 1 classification metrics by horizon.
- Stage 2 regression error on `trade_value`.
- Correlation between predicted `trade_value_pred` and realized `trade_value`.
- Feature importance by Stage 1 model and Stage 2 fusion model.

## 13. Error Handling

The pipeline should explicitly handle:

- Missing daily bars.
- Insufficient history for 180-day block features.
- Missing volume or amount fields.
- Zero or invalid prices.
- Suspended stocks or non-trading days.
- Insufficient future window for labels.
- Same-day up/down label conflicts.
- Model artifact and feature-schema mismatch.

Batch training or prediction should skip invalid symbols and record skip reasons instead of failing the whole run.

## 14. Testing Strategy

Unit tests:

- Path label outcomes for up, down, neutral, and same-day conflict.
- `t+1 open` entry logic.
- `value_Nd` calculation.
- `trade_value` weighted calculation.
- 20d and 60d block feature alignment.
- Technical indicator features use only data available at or before `t`.
- Exclusion of existing model outputs, pattern outputs, and aggregate TradingView scores.

Integration tests:

- Small mainboard-like universe builds Stage 1 datasets.
- Walk-forward out-of-fold Stage 1 predictions are generated without leakage.
- Stage 2 trains only on out-of-fold Stage 1 outputs.
- Daily prediction produces ranking and explanation fields.

Regression tests:

- Changing horizon thresholds changes model metadata.
- Loading a model with mismatched feature schema fails clearly.
- Prediction refuses to run if required Stage 1 artifacts are missing.

## 15. V1 Completion Criteria

V1 is complete when:

- The four Stage 1 datasets and labels can be built.
- The four Stage 1 models can be trained independently.
- Walk-forward Stage 1 out-of-fold predictions can be generated.
- The Stage 2 `trade_value` model can be trained from those predictions.
- Daily mainboard ranking can be generated.
- The report includes trade value, risk, trend, timing, and horizon-level probabilities.
- Backtest reports show Top N return, downside rate, drawdown, and score-bucket monotonicity.

## 16. Future Extensions

Possible later work:

- Add intraday data to resolve same-day conflicts.
- Add index and industry relative-strength features.
- Add market-regime-specific calibration.
- Add probability calibration for each Stage 1 model.
- Add portfolio-level simulation with transaction costs and liquidity constraints.
