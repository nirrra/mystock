# Practical Trading Model Targets

Date: 2026-05-02

This file records the minimum practical targets for turning the probability model into a trading system. Future feature engineering, model changes, loss-function changes, and backtests should be judged against these targets rather than only against AUC, PR-AUC, or validation loss.

## Trading Objective

The system is intended to select mainboard A-share candidates that can rise steadily after entry.

Primary trade definition:

- Holding window: 20 trading days.
- Target profit: at least 10%.
- Maximum tolerated drawdown: no more than 8%.
- Preferred behavior: stable upward movement, not a high-volatility rebound that requires bearing large interim losses.

## Hard Minimum Gate

A model is not considered ready for practical use unless it passes all of the following gates on strict out-of-sample evaluation.

Top50 gate:

- Take-profit rate: at least 50%.
- Stop-loss rate: no more than 28%.
- Average maximum drawdown: no more than 6.5%.
- Average 20-day net return after realistic costs: at least 2%.
- Highest-score bucket must clearly outperform the lowest-score bucket on risk-adjusted return.
- Bucket performance should be directionally monotonic: higher score should generally mean higher trading value, not only higher predicted probability.
- Rolling out-of-sample performance must stay stable for at least 12 months.
- A frozen-model paper-trading period of at least 3 months must not show obvious decay versus backtest performance.

Stricter Top20 target:

- Take-profit rate: at least 55%.
- Stop-loss rate: no more than 25%.
- Average maximum drawdown: no more than 6%.
- Average 20-day net return after realistic costs: at least 3%.

## Backtest Requirements

Backtests must approximate real trading conditions before results can be used for model decisions.

Required constraints:

- Use walk-forward training and testing rather than a single static split.
- Cover different market regimes, including rising, sideways, and falling phases.
- Deduct realistic transaction costs, stamp duty, slippage, and execution friction.
- Respect limit-up, limit-down, suspension, liquidity, and turnover constraints.
- Use only data that would have been available at the decision time.
- Avoid survivorship bias, future constituent bias, and future-adjusted data leakage.
- Compare against random mainboard selection, broad market benchmarks, and industry-neutral baselines.

Minimum portfolio-level expectations:

- Out-of-sample annualized return should exceed the benchmark by at least 8% to 10%.
- Maximum drawdown should be no more than 60% to 70% of the benchmark drawdown.
- Sharpe ratio should be greater than 1.0.
- Return-to-drawdown ratio should be greater than 1.5.
- Monthly win rate should be greater than 55%.
- Consecutive losing months should generally not exceed 3 to 4 months.

## Paper-Trading Gate

Before using real capital, the model must pass a frozen-rule paper-trading stage.

Minimum requirements:

- Paper-trading duration: 3 to 6 months.
- Minimum number of real generated signals: 80 to 150.
- Model, parameters, ranking formula, entry rule, exit rule, and position sizing must be frozen before the paper-trading period starts.
- Failed trades must not be removed after the fact.
- Signal generation time, candidate list, entry price, exit price, take-profit event, stop-loss event, timeout event, slippage, and unfilled orders must be recorded.
- Paper-trading TopN performance should remain close to walk-forward backtest performance.

## Current Gap

The latest horizon-conditioned model is not yet practical.

Latest observed test result:

- Top50 take-profit rate: 36.37%.
- Top50 stop-loss rate: 36.38%.
- Test-set baseline take-profit rate: 34.91%.
- Top50 lift: 1.04.
- Top50 average return: 1.34%.
- Top50 average maximum drawdown: 6.68%.

Interpretation:

- The model has learned some weak signal.
- The model is better at reducing some downside risk than at identifying high take-profit probability.
- The current ranking is not strong enough for practical trading.
- Future work should prioritize trading-value ranking, drawdown-aware objectives, and walk-forward validation rather than only improving classification metrics.

## Optimization Direction

Future model work should be evaluated by whether it moves the system toward the hard gate.

Priority directions:

- Replace pure classification loss with a trading utility or ranking objective.
- Penalize high-score stop-loss cases more heavily than ordinary classification errors.
- Separate risk filtering from upside ranking if a single model cannot optimize both.
- Add relative strength features versus market, industry, and comparable stocks.
- Add market-regime and industry-regime features.
- Evaluate models by TopN return, stop-loss rate, drawdown, and bucket monotonicity.
- Treat AUC and PR-AUC as diagnostic metrics, not final success criteria.
