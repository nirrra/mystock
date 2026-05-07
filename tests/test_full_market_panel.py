from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.full_market_panel import audit_full_market_data
from stocks_analyzer.models import StorageConfig
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


def test_audit_full_market_data_reports_history_eligibility() -> None:
    root = _make_workspace_tmp_dir("full_market_audit")
    storage = _storage(root)
    storage.save_universe(
        pd.DataFrame(
            [
                {"symbol": "600000", "name": "长样本"},
                {"symbol": "600001", "name": "短样本"},
                {"symbol": "600002", "name": "缺数据"},
            ]
        )
    )
    storage.save_daily_bars("600000", _bars(130))
    storage.save_daily_bars("600001", _bars(80))

    result = audit_full_market_data(
        storage=storage,
        project_root=root,
        min_exact_history_days=900,
        tail_lookback_days=100,
        max_horizon_days=20,
    )

    assert result.detail_path.exists()
    assert result.summary_path.exists()
    assert result.summary["symbols_total"] == 3
    assert result.summary["symbols_readable"] == 2
    assert result.summary["symbols_unreadable"] == 1
    assert result.summary["eligible_tail_risk_symbols"] == 1
    assert result.summary["eligible_barrier_symbols"] == 1
    assert result.summary["eligible_exact_history_symbols"] == 0
    assert result.summary["strict_reproduction_ready"] is False

    row = result.detail[result.detail["symbol"].eq("600000")].iloc[0]
    assert row["trading_days"] == 130
    assert row["eligible_tail_risk"] == True
    assert row["eligible_barrier"] == True


def _make_workspace_tmp_dir(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _storage(root: Path) -> Storage:
    paths = ProjectPaths(
        root,
        StorageConfig(base_dir=Path("data"), universe_file="universe.parquet", signals_dir="signals", reports_dir="reports", daily_dir="daily"),
    )
    return Storage(paths)


def _bars(count: int) -> pd.DataFrame:
    rows = []
    for index, trade_date in enumerate(pd.bdate_range("2024-01-01", periods=count)):
        close = 10.0 + index * 0.01
        rows.append(
            {
                "trade_date": trade_date,
                "open": close - 0.02,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": 100000 + index,
                "amount": (100000 + index) * close,
            }
        )
    return pd.DataFrame(rows)
