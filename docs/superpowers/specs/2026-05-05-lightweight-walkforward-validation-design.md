# Lightweight Walk-Forward Model Validation Design

## Goal

Validate whether the current mainline stock model generalizes across different market periods. The target model is the daily `predict-model` logic, currently `v42_gate_v4_rank`: V4.2 opportunity gate plus V4 `long_upside_score` ranking.

This validation does not promote a new model and does not change daily screening behavior. It only adds a repeatable multi-window evaluation command.

## Scope

The first version evaluates only the mainline V4.2 hybrid model. V5 and V5.1 stay out of scope so the report answers one question clearly: whether the current production candidate filter is stable across future time windows.

The command should be lightweight enough to run locally. It may use fewer boosting iterations by default than the full overnight training path.

## Command

Add a command:

```bash
python -m stocks_analyzer --project-root . validate-model-walkforward --model v42 --windows 8 --max-iter 40 --top-n 20,50
```

Supported options:

- `--model`: initially only `v42`.
- `--windows`: number of rolling windows.
- `--train-days`: approximate number of trading dates in each training window.
- `--valid-days`: approximate number of trading dates in each validation window.
- `--test-days`: approximate number of trading dates in each test window.
- `--max-iter`: training iteration budget passed into V4.2 training.
- `--top-n`: TopN list for evaluation.
- `--min-train-days`: minimum training dates required to keep a window.

## Window Logic

Use trading dates available in the model dataset, not calendar dates. Each window is chronological:

```text
train_start -> train_end -> valid_start -> valid_end -> test_start -> test_end
```

No row from the validation or test segment may enter training. Windows roll forward by a stride derived from the available date count and requested window count. If there is not enough history for the requested number of windows, generate as many valid windows as possible.

The first implementation uses fixed-size rolling windows rather than random splits. This preserves time order and matches the actual trading use case.

## Metrics

Each window should write TopN metrics for the test split and keep the existing V4.2 evaluation definitions where possible:

- evaluable trading days
- allowed trading days
- coverage
- average 20-day return
- median 20-day return
- win rate
- 20-day take-profit rate
- 20-day stop-loss rate
- bad-risk rate
- average positive return
- average negative return

The summary report should aggregate every metric by model/top_n across windows:

- mean
- standard deviation
- minimum
- maximum
- pass rate when a threshold applies

Initial stability thresholds for Top20:

- mean win rate >= 0.70
- worst-window win rate >= 0.55
- mean average 20-day return > 0.05
- mean stop-loss rate <= 0.06
- mean bad-risk rate <= 0.15
- mean coverage >= 0.20

These thresholds are diagnostic, not automatic promotion rules.

## Outputs

Write reports to:

```text
reports/model_walkforward/
  v42_walkforward_windows.csv
  v42_walkforward_topn_metrics.csv
  v42_walkforward_summary.csv
  v42_walkforward_config.json
```

The terminal summary should show per-window Top20 test metrics and the aggregate Top20 pass/fail summary.

## Testing

Add focused tests for:

- chronological window generation
- insufficient-history behavior
- summary aggregation and threshold flags
- CLI parsing for `validate-model-walkforward`

Avoid a full model training test in unit tests. The implementation should keep pure window and summary functions testable without expensive training.
