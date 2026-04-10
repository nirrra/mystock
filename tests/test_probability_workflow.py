from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from stocks_analyzer.cli import _run_predict_prob, _run_train_prob
from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_daily_bars(symbol: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    length = 320
    trade_date = pd.date_range("2024-01-01", periods=length, freq="B")
    trend = np.linspace(0, 20, length)
    seasonal = np.sin(np.linspace(0, 12 * np.pi, length)) * 4
    noise = rng.normal(0, 1.2, length)
    close = 50 + trend + seasonal + noise
    close = np.maximum(close, 5)
    open_ = close + rng.normal(0, 0.6, length)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.5, length)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.5, length)
    volume = rng.integers(800_000, 2_000_000, size=length)
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


def test_probability_training_and_prediction_workflow() -> None:
    tmp_path = _make_workspace_tmp_dir("probability_workflow")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    universe = pd.DataFrame(
        [
            {"symbol": "600000", "name": "测试一"},
            {"symbol": "600001", "name": "测试二"},
            {"symbol": "600002", "name": "测试三"},
        ]
    )
    storage.save_universe(universe)
    for idx, symbol in enumerate(universe["symbol"], start=1):
        storage.save_daily_bars(symbol, _make_daily_bars(symbol, seed=idx))

    _run_train_prob(
        storage=storage,
        config=config,
        start_date=None,
        end_date=None,
        train_end=None,
        valid_end=None,
        test_end=None,
        limit=None,
    )
    assert (paths.ml_models_dir / "xgboost.pkl").exists()

    prediction_date = date(2025, 2, 14)
    output_path = tmp_path / "reports" / "probability" / "out.csv"
    _run_predict_prob(
        storage=storage,
        config=config,
        trade_date=prediction_date,
        top_n=5,
        output=str(output_path),
    )
    assert output_path.exists()

    result = pd.read_csv(output_path)
    assert not result.empty
    assert "success_prob" in result.columns
