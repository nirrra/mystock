from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from stocks_analyzer.full_market_labels import build_tail_risk_frame
from stocks_analyzer.full_market_risk import build_risk_decile_report, reproduce_tail_risk
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
    assert set(result.metrics["model_name"]) >= {"dummy_prior", "logistic_regression"}


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
