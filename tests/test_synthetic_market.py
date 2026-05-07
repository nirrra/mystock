from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.models import StorageConfig
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage
from stocks_analyzer.synthetic_market import aggregate_synthetic_market, build_synthetic_market_index


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _storage(root: Path) -> Storage:
    return Storage(
        ProjectPaths(
            root,
            StorageConfig(
                base_dir=Path("data"),
                universe_file="universe.parquet",
                signals_dir="signals",
                reports_dir="reports",
                daily_dir="daily",
            ),
        )
    )


def _bars(symbol: str, closes: list[float], amounts: list[float] | None = None) -> pd.DataFrame:
    if amounts is None:
        amounts = [1000.0] * len(closes)
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "open": closes,
            "high": [price * 1.01 for price in closes],
            "low": [price * 0.99 for price in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
            "amount": amounts,
        }
    )


def test_aggregate_synthetic_market_builds_equal_and_amount_weighted_returns() -> None:
    panel = pd.DataFrame(
        [
            {
                "trade_date": "2026-01-02",
                "symbol": "600000",
                "daily_return": 0.10,
                "amount": 100.0,
                "above_ma20": True,
                "above_ma60": False,
                "limit_up_like": False,
                "limit_down_like": False,
            },
            {
                "trade_date": "2026-01-02",
                "symbol": "600001",
                "daily_return": -0.10,
                "amount": 300.0,
                "above_ma20": False,
                "above_ma60": False,
                "limit_up_like": False,
                "limit_down_like": True,
            },
            {
                "trade_date": "2026-01-02",
                "symbol": "600002",
                "daily_return": float("inf"),
                "amount": 900.0,
                "above_ma20": True,
                "above_ma60": True,
                "limit_up_like": False,
                "limit_down_like": False,
            },
        ]
    )

    result = aggregate_synthetic_market(panel)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["stock_count"] == 2
    assert row["equal_weight_return"] == 0.0
    assert row["amount_weight_return"] == -0.05
    assert row["breadth_up_ratio"] == 0.5
    assert row["above_ma20_ratio"] == 0.5
    assert row["limit_down_count"] == 1


def test_build_synthetic_market_index_uses_local_daily_bars_and_writes_csv() -> None:
    root = _make_workspace_tmp_dir("synthetic_market")
    storage = _storage(root)
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "甲"}, {"symbol": "600001", "name": "乙"}]))
    storage.save_daily_bars("600000", _bars("600000", [10.0, 11.0, 12.1], [100.0, 100.0, 100.0]))
    storage.save_daily_bars("600001", _bars("600001", [20.0, 18.0, 19.8], [100.0, 300.0, 300.0]))

    result = build_synthetic_market_index(
        storage=storage,
        project_root=root,
        start_date="2026-01-02",
        end_date="2026-01-03",
        min_stock_count=1,
    )

    assert result.output_path == root / "reports" / "full_market_model" / "synthetic_market.csv"
    assert result.output_path.exists()
    assert result.skipped.empty
    assert result.frame["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-03"]
    assert result.frame["stock_count"].tolist() == [2, 2]
    assert result.frame["equal_weight_return"].round(6).tolist() == [0.0, 0.1]
    assert result.frame["amount_weight_return"].round(6).tolist() == [0.0, 0.1]
    assert "synthetic_equal_weight_index" in result.frame.columns


def test_build_synthetic_market_index_filters_low_coverage_dates() -> None:
    root = _make_workspace_tmp_dir("synthetic_market_coverage")
    storage = _storage(root)
    storage.save_universe(pd.DataFrame([{"symbol": "600000", "name": "甲"}, {"symbol": "600001", "name": "乙"}]))
    storage.save_daily_bars("600000", _bars("600000", [10.0, 11.0, 12.1], [100.0, 100.0, 100.0]))
    storage.save_daily_bars("600001", _bars("600001", [20.0, 18.0], [100.0, 300.0]))

    result = build_synthetic_market_index(
        storage=storage,
        project_root=root,
        start_date="2026-01-02",
        end_date="2026-01-03",
        min_stock_count=2,
    )

    assert result.frame["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02"]
