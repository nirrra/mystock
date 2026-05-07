from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from stocks_analyzer.full_market_labels import build_tail_risk_frame
from stocks_analyzer.full_market_risk import (
    build_risk_filter_impact,
    build_tail_risk_walkforward_windows,
    build_risk_decile_report,
    predict_tail_risk,
    reproduce_tail_risk,
    summarize_risk_filter_impact,
    train_tail_risk_model,
    validate_tail_risk_walkforward,
)
from stocks_analyzer.models import StorageConfig
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


def test_build_tail_risk_frame_uses_past_quantile_and_future_label() -> None:
    bars = _bars([10, 10.1, 10.2, 10.3, 10.4, 9.0, 9.1, 9.2])

    frame = build_tail_risk_frame(bars, symbol="600000", lookback_days=3, quantile=0.2, horizon_days=1)

    event_row = frame[frame["trade_date"].eq(pd.Timestamp("2024-01-08"))].iloc[0]
    previous_row = frame[frame["trade_date"].eq(pd.Timestamp("2024-01-05"))].iloc[0]
    assert event_row["tail_event_today"] == 1.0
    assert previous_row["risk_label"] == 1.0


def test_build_risk_decile_report_orders_by_score() -> None:
    scored = pd.DataFrame(
        {
            "model_name": ["m"] * 20,
            "risk_score": [index / 20 for index in range(20)],
            "risk_label": [0] * 10 + [1] * 10,
            "forward_log_return": [0.01] * 10 + [-0.02] * 10,
            "future_return_5d": [0.02] * 10 + [-0.04] * 10,
            "future_max_drawdown_5d": [-0.01] * 10 + [-0.08] * 10,
        }
    )

    report = build_risk_decile_report(scored)

    low = report[report["risk_decile"].eq(0)].iloc[0]
    high = report[report["risk_decile"].eq(report["risk_decile"].max())].iloc[0]
    assert low["risk_label_rate"] == 0.0
    assert high["risk_label_rate"] == 1.0


def test_build_risk_filter_impact_removes_highest_risk_bucket() -> None:
    scored = pd.DataFrame(
        {
            "window_id": ["wf_01"] * 10,
            "scope": ["panel_walkforward"] * 10,
            "split": ["valid"] * 10,
            "model_name": ["m"] * 10,
            "symbol": [f"60000{index}" for index in range(10)],
            "risk_score": [index / 10 for index in range(10)],
            "risk_label": [0] * 8 + [1, 1],
            "future_return_5d": [0.01] * 8 + [0.0, 0.0],
            "future_max_drawdown_5d": [-0.02] * 8 + [-0.10, -0.12],
        }
    )

    impact = build_risk_filter_impact(scored, filter_rates=(0.2,), return_tolerance=0.02)
    summary = summarize_risk_filter_impact(impact)

    row = impact.iloc[0]
    assert row["removed_rows"] == 2
    assert row["kept_rows"] == 8
    assert row["future_max_drawdown_5d_delta"] > 0
    assert bool(row["filter_pass"]) is True
    assert bool(summary.iloc[0]["phase1_filter_pass"]) is True


def test_reproduce_tail_risk_writes_metrics_on_short_sample() -> None:
    root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "tail_risk_repro"
    root.mkdir(parents=True, exist_ok=True)
    storage = _storage(root)
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "甲"}, {"symbol": "600001", "name": "乙"}]))
    storage.save_daily_bars("600000", _bars([10 + i * 0.1 for i in range(160)]))
    storage.save_daily_bars("600001", _bars([12 + i * 0.05 + (0.8 if i % 17 == 0 else 0) for i in range(160)]))

    result = reproduce_tail_risk(
        storage=storage,
        project_root=root,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 8, 9),
        train_end=date(2024, 5, 31),
        valid_end=date(2024, 8, 9),
        lookback_days=20,
        min_training_rows=20,
        allow_short_sample=True,
    )

    assert result.metrics_path.exists()
    assert result.deciles_path.exists()
    assert result.index_reproduction_path.exists()
    assert result.index_dataset_path.exists()
    assert set(result.metrics["model_name"]) >= {"dummy_prior", "logistic_regression"}
    assert set(result.index_reproduction["model_name"]) >= {
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "linear_discriminant_analysis",
        "naive_bayes",
        "quadratic_discriminant_analysis",
        "adaboost",
        "gradient_boosting",
    }


def test_build_tail_risk_walkforward_windows_uses_embargo() -> None:
    dataset = pd.DataFrame({"trade_date": pd.bdate_range("2024-01-01", periods=80)})

    windows = build_tail_risk_walkforward_windows(
        dataset,
        train_days=20,
        valid_days=10,
        step_days=10,
        embargo_days=2,
        max_windows=2,
    )

    assert windows["window_id"].tolist() == ["wf_01", "wf_02"]
    assert windows.iloc[0]["train_end"] == "2024-01-26"
    assert windows.iloc[0]["valid_start"] == "2024-01-31"


def test_validate_tail_risk_walkforward_writes_reports_on_short_sample() -> None:
    root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "tail_risk_walkforward"
    root.mkdir(parents=True, exist_ok=True)
    storage = _storage(root)
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "甲"}, {"symbol": "600001", "name": "乙"}]))
    storage.save_daily_bars("600000", _bars([10 + i * 0.08 for i in range(180)]))
    storage.save_daily_bars("600001", _bars([12 + i * 0.04 + (0.6 if i % 19 == 0 else 0) for i in range(180)]))

    result = validate_tail_risk_walkforward(
        storage=storage,
        project_root=root,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 9, 6),
        train_days=50,
        valid_days=20,
        step_days=20,
        embargo_days=1,
        max_windows=2,
        lookback_days=10,
        min_training_rows=20,
        allow_short_sample=True,
        panel_model_names=("dummy_prior", "logistic_regression"),
    )

    assert result.windows_path.exists()
    assert result.metrics_path.exists()
    assert result.deciles_path.exists()
    assert result.filter_impact_path.exists()
    assert result.filter_summary_path.exists()
    assert result.summary_path.exists()
    assert len(result.windows) == 2
    assert set(result.summary["model_name"]) == {"dummy_prior", "logistic_regression"}
    assert set(result.filter_summary["model_name"]) == {"dummy_prior", "logistic_regression"}


def test_train_and_predict_tail_risk_model_roundtrip_on_short_sample() -> None:
    root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "tail_risk_artifact"
    root.mkdir(parents=True, exist_ok=True)
    storage = _storage(root)
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "甲"}, {"symbol": "600001", "name": "乙"}]))
    storage.save_daily_bars("600000", _bars([10 + i * 0.06 for i in range(180)]))
    storage.save_daily_bars("600001", _bars([12 + i * 0.03 + (-0.9 if i % 23 == 0 else 0) for i in range(180)]))

    train_result = train_tail_risk_model(
        storage=storage,
        project_root=root,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 8, 30),
        lookback_days=10,
        min_training_rows=20,
    )
    predict_result = predict_tail_risk(
        storage=storage,
        project_root=root,
        trade_date=date(2024, 8, 30),
    )

    assert train_result.model_path.exists()
    assert train_result.metadata_path.exists()
    assert train_result.model_name == "logistic_regression"
    assert predict_result.output_path.exists()
    assert not predict_result.predictions.empty
    assert set(predict_result.predictions["symbol"]) == {"600000", "600001"}
    assert predict_result.predictions["risk_score"].between(0.0, 1.0).all()


def _storage(root: Path) -> Storage:
    paths = ProjectPaths(
        root,
        StorageConfig(base_dir=Path("data"), universe_file="universe.parquet", signals_dir="signals", reports_dir="reports", daily_dir="daily"),
    )
    return Storage(paths)


def _bars(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [100000] * len(closes),
            "amount": [1000000] * len(closes),
        }
    )
