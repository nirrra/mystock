# Daily Screening Predict Model Design

## Goal

Replace TradingView's role in daily main watchlist selection with a stable `predict_model` step. TradingView still runs and keeps producing its reports, but the main watchlist no longer uses TradingView label or 5-day score for admission or ranking.

The first implementation of `predict_model` uses the existing V3.1 buy-trigger model. Future model versions should be able to replace that implementation without changing the daily screening stage name, prediction output path, or watchlist integration contract.

## Daily Flow

`daily-screening` runs these stages in order:

1. `update`
2. `tradingview`
3. `predict_model`
4. `macd`
5. `atr`
6. `trend-universe`
7. `trend`
8. `pattern`

The `predict_model` stage writes `reports/predict_model/predictions_<date>.csv`.

## Prediction Contract

The prediction file is keyed by `symbol` and `trade_date`. The watchlist integration requires these fields:

- `symbol`
- `trade_date`
- `trigger_action`
- `trigger_reason`
- `action`
- `risk_tier`
- `risk_gate_reason`
- `risk_score`
- `clean_win_score`
- `trigger_score`
- `final_score_v31`
- `buy_score_v31`

The file may keep version-specific score column names while the path remains generic. Later versions can add new columns and map their final rank score into the same integration surface.

## Watchlist Selection

The pattern stage remains the source of the main candidate pool. It still scans the six existing technical pattern types and appends MACD, ATR, trend-universe, and trend summaries.

When generating `watchlist_<date>.json` and `watchlist_pattern_<date>.json`, the pattern candidates are joined to `reports/predict_model/predictions_<date>.csv` by normalized `symbol`.

Main admission rules:

1. The stock must have matched at least one pattern.
2. The stock must have a prediction row for the same date.
3. `trigger_action` must be `buy`.
4. Existing hard technical risk exclusions remain active: MACD dead cross, MACD top divergence, bearish volume-price divergence, recent MACD top divergence flag, and bearish volume-price divergence flag.

TradingView label and TradingView 5-day average are not admission rules in this path.

Ranking rules:

1. `final_score_v31` descending.
2. `buy_score_v31` descending.
3. Existing pattern priority as a deterministic tie-breaker.

The watchlist candidate output includes model explanation fields alongside the existing pattern and risk context. TradingView fields may remain in pattern reports for reference, but they are not used to decide membership or order.

## Error Handling

If `predict_model` cannot produce predictions, `daily-screening` should fail instead of silently falling back to TradingView. A missing model prediction file during watchlist generation is also a hard error. This keeps the daily result honest after the selection source changes.

If a specific pattern candidate has no matching model row, that candidate is skipped and the reason is visible from the absence of model fields in debug/report data. The final watchlist contains only candidates that satisfy the model contract.

## Scope

In scope:

- Add a generic `predict_model` CLI command or stage wrapper.
- Write predictions to `reports/predict_model/predictions_<date>.csv`.
- Run `predict_model` after `tradingview` inside `daily-screening`.
- Use the prediction file to replace TradingView filtering and ranking in pattern watchlist generation.
- Update tests for stage order, prediction output path, and watchlist selection behavior.

Out of scope:

- Re-training V3.1.
- Changing trend watchlist selection.
- Removing TradingView reports.
- Changing intraday screening display fields.
