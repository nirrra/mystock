# ATR Assist Design

## Goal

Add ATR as an independent daily report and a final watchlist assist signal without changing any existing trend filtering or scoring behavior.

## Scope

- Add `atr` as an independent CLI command and `daily-screening` stage.
- Generate `reports/atr/atr_<date>.csv` with Chinese headers for manual review.
- Merge ATR-derived risk fields into the final pattern watchlist payload.
- Keep ATR out of trend-universe filtering, trend scoring, and watchlist ranking.

## Data Flow

1. `daily-screening` runs `macd`.
2. `daily-screening` runs `atr`.
3. `atr` scans the market, computes ATR14 snapshots, and saves an ATR report.
4. `pattern` merges the ATR snapshot by `symbol` before writing the final watchlist.

## ATR Fields

Internal fields:

- `atr_14`
- `atr_pct_14`
- `atr_stop_loss_1x`
- `atr_stop_loss_2x`
- `atr_take_profit_2x`
- `atr_take_profit_3x`
- `atr_volatility_regime`

Chinese export fields:

- `代码`
- `名称`
- `交易日期`
- `收盘价`
- `ATR14`
- `ATR%`
- `1ATR止损参考`
- `2ATR止损参考`
- `2ATR止盈参考`
- `3ATR止盈参考`
- `波动分层`

## Rules

- ATR period is fixed at 14.
- ATR uses Wilder smoothing.
- `atr_pct_14` is stored internally as a ratio and shown externally as a percent number.
- Volatility regimes:
  - `< 3%`: `低波动`
  - `3% ~ < 6%`: `中等波动`
  - `>= 6%`: `高波动`

## Compatibility

- Existing MACD, trend, and watchlist filtering rules stay unchanged.
- If ATR data is missing, final watchlist generation still succeeds.
- Existing watchlist files are not migrated.

## Testing

- Indicator test for ATR14 and ATR%.
- CLI/parser test for the new `atr` command.
- ATR merge test for final pattern enrichment.
- Daily screening test for the new stage and report path.
