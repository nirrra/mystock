# Daily Screening Component Backtest Commands

Date: 2026-05-07

This document lists the commands needed to evaluate the current `daily-screening` pipeline components and their combinations.

Important status:

- Existing CLI already supports Phase-level validation and pure pattern backtests.
- Existing CLI now includes `backtest-daily-screening-components`, which replays a small historical sample across Phase1/Phase2/Phase4/Phase5/Phase7 and patterns.
- The older per-strategy examples later in this file are retained as design notes. Prefer the runnable combined commands in this section.

Runnable smoke command:

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date 2026-01-05 --end-date 2026-02-06 --horizons 5,10,20,60 --top-n 5 --max-signal-days 2 --symbol-limit 80 --output-dir reports\daily_screening_smoke_backtest_test --progress
```

Runnable larger small-sample command:

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date 2025-01-01 --end-date 2026-02-06 --horizons 5,10,20,60 --top-n 20 --max-signal-days 60 --symbol-limit 500 --output-dir reports\daily_screening_smoke_backtest --progress
```

## 1. Recommended Backtest Window

Use the 8-10 year local A-share daily dataset and leave the last 20 trading days out of labels.

```powershell
$env:PYTHONPATH = "src"
$START = "2016-01-01"
$END = "2026-04-30"
$OUT = "reports\daily_screening_component_backtests"
```

If runtime is too long, first run a smoke version:

```powershell
$env:PYTHONPATH = "src"
$START = "2024-01-01"
$END = "2026-04-30"
$OUT = "reports\daily_screening_component_backtests_smoke"
```

## 2. Existing Commands

These commands are directly runnable with the current CLI and should be run before the combination backtest.

### 2.1 Data Audit

```powershell
python -m stocks_analyzer --project-root . audit-full-market-data --min-exact-history-days 900 --tail-lookback-days 100 --max-horizon-days 20
```

### 2.2 Phase1 Tail-Risk Validation

```powershell
python -m stocks_analyzer --project-root . validate-tail-risk-walkforward --start-date $START --end-date $END --windows 6
```

### 2.3 Phase2 Triple-Barrier Risk Validation

```powershell
python -m stocks_analyzer --project-root . validate-barrier-risk-grid --start-date $START --end-date $END
```

### 2.4 Phase4 Alpha158/Qlib Return Validation

```powershell
python -m stocks_analyzer --project-root . validate-alpha158-qlib-return --start-date $START --end-date $END
```

### 2.5 Phase5 MCD Crash-Risk Validation

```powershell
python -m stocks_analyzer --project-root . validate-mcd-crash-risk --start-date $START --end-date $END
```

### 2.6 Phase7 Trade-Day Gate Validation

```powershell
python -m stocks_analyzer --project-root . validate-trade-day-gate --start-date $START --end-date $END
```

### 2.7 Pure Pattern Backtests

All patterns:

```powershell
python -m stocks_analyzer --project-root . backtest-patterns --start-date $START --date $END --save-forward-prices --forward-days 40 --output "$OUT\pattern_all_detail.csv"
```

Single patterns:

```powershell
python -m stocks_analyzer --project-root . backtest-patterns --start-date $START --date $END --1 --save-forward-prices --forward-days 40 --output "$OUT\pattern1_detail.csv"
python -m stocks_analyzer --project-root . backtest-patterns --start-date $START --date $END --2 --save-forward-prices --forward-days 40 --output "$OUT\pattern2_detail.csv"
python -m stocks_analyzer --project-root . backtest-patterns --start-date $START --date $END --3 --save-forward-prices --forward-days 40 --output "$OUT\pattern3_detail.csv"
python -m stocks_analyzer --project-root . backtest-patterns --start-date $START --date $END --4 --save-forward-prices --forward-days 40 --output "$OUT\pattern4_detail.csv"
python -m stocks_analyzer --project-root . backtest-patterns --start-date $START --date $END --5 --save-forward-prices --forward-days 40 --output "$OUT\pattern5_detail.csv"
python -m stocks_analyzer --project-root . backtest-patterns --start-date $START --date $END --6 --save-forward-prices --forward-days 40 --output "$OUT\pattern6_detail.csv"
```

## 3. Combination Backtest Command Contract

Current CLI command:

```text
backtest-daily-screening-components
```

Important arguments:

```text
--start-date YYYY-MM-DD
--end-date YYYY-MM-DD
--strategies STRATEGY_A,STRATEGY_B
--horizons 5,10,20,60
--top-n 20
--phase4-top-n 20
--phase1-filter-rate 0.2
--phase2-filter-rate 0.2
--stop-loss-pct 0.08
--take-profit-pct 0.15
--max-signal-days 30
--symbol-limit 500
--output-dir PATH
--progress
```

Output files:

```text
reports/daily_screening_smoke_backtest/
  trades.csv
  daily_portfolio.csv
  summary.csv
  comparison.csv
```

Metrics required in every summary:

- average candidates per day
- no-candidate day rate
- average future return for 5/10/20 trading days
- median future return for 5/10/20 trading days
- win rate for 5/10/20 trading days
- average future max drawdown for 5/10/20 trading days
- 5% quantile return
- loss worse than -5% rate
- loss worse than -8% rate
- stop-like drawdown rate
- daily equal-weight portfolio return
- annual return
- max drawdown
- Sharpe
- Calmar
- turnover
- sample count

## 4. Single-Component Combination Commands

These commands test whether each component has standalone value.

### 4.1 Full-Market Random Baseline

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy full_market_random --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
```

### 4.2 Phase1 Standalone Risk Filter

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase1_filter_only --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --output-dir $OUT --progress
```

### 4.3 Phase2 Standalone Risk Filter

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase2_filter_only --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 4.4 Phase4 Standalone Return Ranking

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase4_top10_only --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
```

### 4.5 Phase5 Standalone Extreme-Risk Grouping

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase5_group_only --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
```

### 4.6 Phase7 Standalone Trade-Day Gate

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase7_gate_only --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase7-block-rate 0.2 --output-dir $OUT --progress
```

### 4.7 Patterns Standalone

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy patterns_only --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
```

## 5. Core Daily-Screening Combination Commands

These commands test the current candidate-construction logic and its main ablations.

### 5.1 Phase1 + Phase4

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase1_filter_phase4_top10 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.2 Phase2 + Phase4

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase2_filter_phase4_top10 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.3 Phase1 + Phase2 + Phase4

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy phase1_phase2_filter_phase4_top10 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.4 Patterns + Phase1

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy patterns_phase1_filter --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.5 Patterns + Phase2

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy patterns_phase2_filter --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.6 Patterns + Phase1 + Phase2

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy patterns_phase1_phase2_filter --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.7 Patterns + Phase4 Sort

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy patterns_phase4_sort --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
```

### 5.8 Patterns + Phase1 + Phase2 + Phase4 Sort

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy patterns_phase1_phase2_filter_phase4_sort --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.9 Current Watchlist Without Phase7

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy current_watchlist_without_phase7 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 5.10 Current Watchlist With Phase7

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy current_watchlist_with_phase7 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --phase7-block-rate 0.2 --output-dir $OUT --progress
```

## 6. Pattern-by-Pattern Commands

These commands split pattern effectiveness by pattern ID. They answer whether one or two patterns carry the whole signal.

### 6.1 Pattern-Only by ID

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern1_only --pattern-ids 1 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern2_only --pattern-ids 2 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern3_only --pattern-ids 3 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern4_only --pattern-ids 4 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern5_only --pattern-ids 5 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern6_only --pattern-ids 6 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --output-dir $OUT --progress
```

### 6.2 Pattern + Phase1/Phase2 Filter by ID

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern1_phase1_phase2_filter --pattern-ids 1 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern2_phase1_phase2_filter --pattern-ids 2 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern3_phase1_phase2_filter --pattern-ids 3 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern4_phase1_phase2_filter --pattern-ids 4 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern5_phase1_phase2_filter --pattern-ids 5 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern6_phase1_phase2_filter --pattern-ids 6 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

### 6.3 Pattern + Phase1/Phase2 Filter + Phase4 Sort by ID

```powershell
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern1_phase1_phase2_filter_phase4_sort --pattern-ids 1 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern2_phase1_phase2_filter_phase4_sort --pattern-ids 2 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern3_phase1_phase2_filter_phase4_sort --pattern-ids 3 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern4_phase1_phase2_filter_phase4_sort --pattern-ids 4 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern5_phase1_phase2_filter_phase4_sort --pattern-ids 5 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
python -m stocks_analyzer --project-root . backtest-daily-screening-components --start-date $START --end-date $END --strategy pattern6_phase1_phase2_filter_phase4_sort --pattern-ids 6 --entry next_open --horizons 5,10,20 --top-n 10,20,50 --phase1-filter-rate 0.2 --phase2-filter-rate 0.2 --output-dir $OUT --progress
```

## 7. Batch Runner After Command Is Implemented

After `backtest-daily-screening-components` exists, this PowerShell loop should run the main strategy matrix.

```powershell
$env:PYTHONPATH = "src"
$START = "2016-01-01"
$END = "2026-04-30"
$OUT = "reports\daily_screening_component_backtests"

$strategies = @(
  "full_market_random",
  "phase1_filter_only",
  "phase2_filter_only",
  "phase4_top10_only",
  "phase5_group_only",
  "phase7_gate_only",
  "patterns_only",
  "phase1_filter_phase4_top10",
  "phase2_filter_phase4_top10",
  "phase1_phase2_filter_phase4_top10",
  "patterns_phase1_filter",
  "patterns_phase2_filter",
  "patterns_phase1_phase2_filter",
  "patterns_phase4_sort",
  "patterns_phase1_phase2_filter_phase4_sort",
  "current_watchlist_without_phase7",
  "current_watchlist_with_phase7"
)

foreach ($s in $strategies) {
  python -m stocks_analyzer --project-root . backtest-daily-screening-components `
    --start-date $START `
    --end-date $END `
    --strategy $s `
    --entry next_open `
    --horizons 5,10,20 `
    --top-n 10,20,50 `
    --phase1-filter-rate 0.2 `
    --phase2-filter-rate 0.2 `
    --phase7-block-rate 0.2 `
    --output-dir $OUT `
    --progress
}
```

## 8. Result Comparison Priority

Read results in this order:

1. `patterns_only` vs `full_market_random`
2. `phase4_top10_only` vs `full_market_random`
3. `phase1_filter_phase4_top10` vs `phase4_top10_only`
4. `phase2_filter_phase4_top10` vs `phase4_top10_only`
5. `phase1_phase2_filter_phase4_top10` vs both single-risk-filter versions
6. `patterns_phase1_phase2_filter` vs `patterns_only`
7. `patterns_phase1_phase2_filter_phase4_sort` vs `patterns_phase1_phase2_filter`
8. `current_watchlist_with_phase7` vs `current_watchlist_without_phase7`

Decision rules:

- Keep Phase1 only if it reduces max drawdown and tail-loss rate without materially reducing average return.
- Keep Phase2 only if it adds risk reduction beyond Phase1 or performs better on pattern subsets.
- Keep Phase4 only if TopN beats random and bottom-ranked names across multiple years.
- Keep Phase7 only if no-trade days are consistently worse than allow days and applying the gate improves portfolio drawdown.
- Treat Phase5 as warning or downgrade unless the backtest shows a hard filter improves both downside and return.
- Keep individual patterns only if their standalone or filtered performance beats random with enough sample count.
