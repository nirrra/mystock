from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage
from stocks_analyzer.tradingview_factor_research import (
    run_tradingview_factor_research,
    save_tradingview_factor_research_reports,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_daily_bars(symbol: str, seed: int, length: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trade_date = pd.date_range("2025-01-01", periods=length, freq="B")
    trend = np.linspace(0, seed * 0.8, length)
    seasonal = np.sin(np.linspace(0, 8 * np.pi, length)) * 0.8
    noise = rng.normal(0, 0.15, length)
    close = np.maximum(10 + trend + seasonal + noise, 2)
    open_ = close + rng.normal(0, 0.08, length)
    high = np.maximum(open_, close) + rng.uniform(0.04, 0.18, length)
    low = np.minimum(open_, close) - rng.uniform(0.04, 0.18, length)
    volume = rng.integers(600_000, 1_400_000, size=length)
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


def test_tradingview_factor_research_builds_samples_and_topn_reports() -> None:
    tmp_path = _make_workspace_tmp_dir("tradingview_factor_research")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    instruments = [{"symbol": f"60000{index}", "name": f"测试{index}"} for index in range(8)]
    storage.save_universe(pd.DataFrame(instruments))
    for index, instrument in enumerate(instruments, start=1):
        storage.save_daily_bars(instrument["symbol"], _make_daily_bars(instrument["symbol"], seed=index))

    result = run_tradingview_factor_research(
        storage,
        start_date=date(2025, 11, 3),
        end_date=date(2025, 11, 14),
        horizons=(1, 5),
        factor_fields=("all_rating", "avg_all_rating_5d"),
        rank_fields=("all_rating", "avg_all_rating_5d"),
        top_n=3,
        quantiles=3,
    )

    assert not result.samples.empty
    assert {"forward_return_1d", "forward_return_5d", "max_drawdown_5d", "avg_all_rating_5d"}.issubset(result.samples.columns)
    assert (result.samples["entry_date"] > result.samples["trade_date"]).all()
    assert set(result.topn_detail["rank_field"].unique()) == {"all_rating", "avg_all_rating_5d"}
    assert result.topn_detail["rank"].max() == 3
    assert not result.topn_summary.empty
    assert {"rank_field", "top_count", "horizon_days", "avg_daily_equal_weight_return"}.issubset(result.topn_summary.columns)

    report_paths = save_tradingview_factor_research_reports(
        paths,
        result=result,
        start_date=date(2025, 11, 3),
        end_date=date(2025, 11, 14),
        top_n=3,
    )

    assert report_paths["samples_path"].exists()
    assert report_paths["topn_detail_path"].exists()
    assert report_paths["topn_summary_path"].exists()
