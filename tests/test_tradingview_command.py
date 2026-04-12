from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from stocks_analyzer.cli import _run_tradingview
from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_daily_bars(symbol: str, seed: int, start: str = "2025-01-01", length: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trade_date = pd.date_range(start, periods=length, freq="B")
    trend = np.linspace(0, 12, length)
    seasonal = np.sin(np.linspace(0, 10 * np.pi, length)) * 2
    noise = rng.normal(0, 0.8, length)
    close = np.maximum(20 + trend + seasonal + noise, 5)
    open_ = close + rng.normal(0, 0.4, length)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.9, length)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.9, length)
    volume = rng.integers(700_000, 1_600_000, size=length)
    amount = close * volume * 100
    return pd.DataFrame(
        {
            "trade_date": trade_date,
            "symbol": [symbol] * length,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": amount,
            "pct_change": pd.Series(close).pct_change().fillna(0.0),
            "change": pd.Series(close).diff().fillna(0.0),
            "amplitude": (high - low) / close,
            "turnover": [1.0] * length,
        }
    )


def test_run_tradingview_generates_csv() -> None:
    tmp_path = _make_workspace_tmp_dir("tradingview_command")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    universe = pd.DataFrame(
        [
            {"symbol": "600000", "name": "测试一"},
            {"symbol": "600001", "name": "测试二"},
            {"symbol": "600002", "name": "旧数据"},
        ]
    )
    storage.save_universe(universe)
    storage.save_daily_bars("600000", _make_daily_bars("600000", seed=1))
    storage.save_daily_bars("600001", _make_daily_bars("600001", seed=2))
    storage.save_daily_bars("600002", _make_daily_bars("600002", seed=3, start="2024-01-01", length=90))

    output_path = tmp_path / "reports" / "tradingview" / "ratings.csv"
    _run_tradingview(
        storage=storage,
        config=config,
        trade_date=date(2025, 10, 31),
        top_n=10,
        output=str(output_path),
    )

    assert output_path.exists()
    result = pd.read_csv(output_path)
    assert not result.empty
    assert {"ma_rating", "osc_rating", "all_rating", "avg_all_rating_5d", "all_rating_label"}.issubset(result.columns)
    rating_date_columns = [column for column in result.columns if column.startswith("all_rating_2025")]
    assert len(rating_date_columns) == 5
    expected_prefix = ["symbol", "name", *sorted(rating_date_columns), "avg_all_rating_5d"]
    assert result.columns[: len(expected_prefix)].tolist() == expected_prefix
    daily_files = sorted(output_path.parent.glob("tradingview_*.csv"))
    assert len(daily_files) == 5
    assert all("2025" in item.name for item in daily_files)
