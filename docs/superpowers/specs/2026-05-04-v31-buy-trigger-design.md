# V3.1 Buy Trigger Design

Date: 2026-05-04

## 1. Background

The current main version is V3 0.7:

```text
clean_win_train_scope = weighted-all
bad_risk_sample_weight = 0.7
```

V3 0.7 is valuable mainly as a risk-avoidance layer. On the out-of-sample test Top 20 it achieved:

- 20-day average return: 2.44%.
- 20-day take-profit rate: 15.98%.
- 20-day stop-loss rate: 13.91%.
- 20-day average maximum drawdown: 4.47%.
- Bad-risk rate: 21.05%.

V4 Alpha Ranker tested a learned alpha ranking layer, but it did not generalize on the test split. It lowered risk further while reducing return and take-profit rate. Therefore V4 should remain an experiment, and V3 0.7 stays the main risk model.

The next problem is narrower:

```text
After V3 0.7 removes high-risk stocks, which remaining stocks are actually worth buying?
```

V3.1 answers this as a buy-trigger problem rather than another return-prediction model.

## 2. Goal

Build a deterministic buy-trigger layer on top of the V3 0.7 low-risk candidate pool.

The layer should improve trade selection by requiring evidence of:

- Acceptable market regime.
- Individual trend confirmation.
- Short-term momentum improvement.
- Volume/amount confirmation.
- No obvious short-term overheat.

Primary target:

- Keep 20-day stop-loss rate <= 15%.
- Keep 20-day average maximum drawdown <= 5%.
- Increase 20-day average return above V3 0.7's 2.44%.
- Increase 20-day take-profit rate toward 20%.

V3.1 is accepted only if it improves return without materially weakening the V3 risk profile.

## 3. Scope

Included:

- V3 0.7 candidate pool as hard prerequisite.
- Rule-based buy trigger score.
- Grid search over trigger thresholds using validation only.
- Test reporting against V3 0.7.
- Optional prediction report for the latest trade date.

Excluded:

- New learned return model.
- Pattern-recognition outputs.
- Existing probability model outputs.
- Intraday execution.
- Portfolio allocation.
- Live trading automation.

## 4. Architecture

```text
daily features
  -> Stage 1 horizon models
  -> V3 0.7 hard risk gate
      -> candidate / avoid
  -> V3.1 buy-trigger features
      -> trigger_score
      -> trigger_flags
  -> validation-selected trigger threshold
  -> final buy candidates
  -> daily Top N report
```

V3.1 never rescues a stock that V3 0.7 rejects. A stock must first pass the V3 risk gate, then pass the buy-trigger layer.

## 5. Trigger Components

The first implementation uses transparent rule factors. Each component contributes to `trigger_score`.

### 5.1 Trend Confirmation

Positive signals:

- `distance_to_ma20 >= -1%`.
- `ma20_slope_5d > 0`.
- `distance_to_ma60 >= -5%`.
- `ma20_to_ma60` improving or not deeply negative.

Negative signals:

- Price far below MA20.
- MA20 slope still falling.
- MA20 far below MA60.

### 5.2 Momentum Improvement

Positive signals:

- `macd_hist` improving.
- `macd_hist_slope > 0`.
- RSI not extremely weak.
- Recent 5-day return is positive but not excessive.

Negative signals:

- MACD histogram still deteriorating.
- RSI too weak.
- Recent 5-day or 10-day return already too high.

### 5.3 Volume And Amount Confirmation

Positive signals:

- `volume_ratio_5` or `amount_ratio_5` moderately above 1.
- `volume_ratio_20` not extremely low.
- Recent amount improves together with price.

Negative signals:

- Volume collapse.
- Extreme one-day volume spike after short-term overheat.

### 5.4 Overheat Filter

Avoid candidates when:

- `return_5d` is above a selected maximum.
- `return_10d` is above a selected maximum.
- Price is too far above MA20.
- `position_in_range_20d` is too close to the top after a sharp short-term rally.

The goal is not to avoid strength entirely. The goal is to avoid buying after a low-risk stock has already moved too far too quickly.

## 6. Initial Score

Initial score formula:

```text
trigger_score =
    trend_score
  + momentum_score
  + volume_score
  - overheat_penalty
```

Scores should be normalized by daily cross-section where practical. This avoids comparing raw indicator scales across dates.

The final daily ranking remains:

```text
V3 candidate pool
  -> pass trigger threshold
  -> sort by trigger_score
```

## 7. Validation Search

Validation search should test small, interpretable grids:

- Minimum trigger score.
- Maximum 5-day return.
- Maximum 10-day return.
- Minimum MA20 slope condition.
- Minimum volume/amount confirmation.
- Optional top N: 10, 20, 50.

Selection objective:

```text
objective =
    avg_return_20d
  + 0.55 * take_profit_rate_20d
  + 0.25 * win_rate
  - 0.80 * stop_loss_rate_20d
  - 0.55 * avg_max_drawdown_20d
  - 0.20 * bad_risk_rate
```

Constraints:

- `stop_loss_rate_20d <= 15%`.
- `avg_max_drawdown_20d <= 5%`.
- `bad_risk_rate <= 23%`.
- Average daily selected count must be high enough for the requested Top N.

If no grid point satisfies all constraints, V3.1 should remain experimental and V3 0.7 stays the main version.

## 8. Reports

Training/report command should output:

- V3.1 split metrics.
- V3.1 Top N metrics.
- V3 0.7 versus V3.1 comparison.
- Trigger grid results.
- Trigger component diagnostics.
- Latest prediction file when requested.

Core metrics:

- 20-day win rate.
- 20-day average return.
- 20-day take-profit rate.
- 20-day stop-loss rate.
- 20-day average take-profit.
- 20-day average stop-loss.
- 20-day average maximum drawdown.
- Bad-risk rate.
- Average candidate count.
- Average selected count.

## 9. Acceptance Criteria

V3.1 becomes the new execution candidate layer only if test Top 20 satisfies:

- 20-day average return > V3 0.7.
- 20-day take-profit rate > V3 0.7.
- 20-day stop-loss rate <= 15%.
- 20-day average maximum drawdown <= 5%.
- Bad-risk rate <= 23%.

If it only lowers risk while reducing return, it is not accepted. That was the V4 failure mode, and V3.1 should not repeat it.

## 10. Implementation Notes

The implementation should reuse V3 scoring outputs:

- `risk_score`.
- `clean_win_score`.
- `down_prob_20d`.
- `down_prob_60d`.
- `action`.
- `risk_gate_reason`.

It should add new fields:

- `trigger_score`.
- `trend_score`.
- `momentum_score`.
- `volume_score`.
- `overheat_penalty`.
- `trigger_action`.
- `trigger_reason`.
- `buy_score_v31`.

Prediction output should keep V3 risk fields visible so the user can inspect why a stock passed both layers.
