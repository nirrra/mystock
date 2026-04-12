from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from stocks_analyzer.cli import _append_recent_tradingview_scores
from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_daily_bars(symbol: str, seed: int, length: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trade_date = pd.date_range("2025-01-01", periods=length, freq="B")
    trend = np.linspace(0, 10, length)
    seasonal = np.sin(np.linspace(0, 8 * np.pi, length)) * 2
    noise = rng.normal(0, 0.6, length)
    close = np.maximum(30 + trend + seasonal + noise, 5)
    open_ = close + rng.normal(0, 0.3, length)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.8, length)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.8, length)
    volume = rng.integers(700_000, 1_500_000, size=length)
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


def test_append_recent_tradingview_scores_adds_five_columns_when_history_is_sufficient() -> None:
    tmp_path = _make_workspace_tmp_dir("pattern_tradingview")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)
    universe = pd.DataFrame([{"symbol": "600000", "name": "测试一"}])
    storage.save_universe(universe)
    storage.save_daily_bars("600000", _make_daily_bars("600000", seed=1))

    exported = pd.DataFrame(
        [
            {
                "trade_date": "2025-10-31",
                "symbol": '="600000"',
                "name": "测试一",
                "pattern_id": "1",
                "close": 10.0,
                "reason": "demo",
            }
        ]
    )

    result = _append_recent_tradingview_scores(storage, exported, as_of=date(2025, 10, 31), lookback_days=5)

    expected_columns = {
        "tradingview_all_rating_2025-10-27",
        "tradingview_all_rating_2025-10-28",
        "tradingview_all_rating_2025-10-29",
        "tradingview_all_rating_2025-10-30",
        "tradingview_all_rating_2025-10-31",
        "tradingview_avg_all_rating_5d",
        "tradingview_all_rating_label",
    }
    assert expected_columns.issubset(result.columns)
    assert result.loc[0, "tradingview_all_rating_2025-10-31"] is not None
    expected_average = float(
        result.loc[
            0,
            [
                "tradingview_all_rating_2025-10-27",
                "tradingview_all_rating_2025-10-28",
                "tradingview_all_rating_2025-10-29",
                "tradingview_all_rating_2025-10-30",
                "tradingview_all_rating_2025-10-31",
            ],
        ].mean()
    )
    assert np.isclose(result.loc[0, "tradingview_avg_all_rating_5d"], expected_average)
    assert result.loc[0, "tradingview_all_rating_label"] in {
        "strong_buy",
        "buy",
        "neutral",
        "sell",
        "strong_sell",
    }
