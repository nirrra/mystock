from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.cli import _refresh_or_load_universe, _update_daily_cache_for_symbol
from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


class EmptyProvider:
    def get_instruments(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "name", "trade_status"])


class RecordingDailyProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, str]] = []

    def get_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        self.calls.append(
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "adjust": adjust,
            }
        )
        return self.frame.copy()


def _make_workspace_tmp_dir(name: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_refresh_or_load_universe_falls_back_to_cached_universe() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = _make_workspace_tmp_dir("update_fallback")
    config = load_config(project_root / "config" / "default.yaml")
    paths = ProjectPaths(root, config.storage)
    storage = Storage(paths)
    cached = pd.DataFrame(
        [
            {"symbol": "600000", "name": "浦发银行", "exchange": "sh", "board": "main_board", "is_st": False, "is_suspended": False}
        ]
    )
    storage.save_universe(cached)

    result = _refresh_or_load_universe(storage, EmptyProvider(), exclude_st=True)

    assert result["symbol"].tolist() == ["600000"]


def _make_daily_frame(symbol: str, dates: list[str]) -> pd.DataFrame:
    trade_dates = pd.to_datetime(dates)
    close = pd.Series(range(10, 10 + len(dates)), dtype=float)
    return pd.DataFrame(
        {
            "trade_date": trade_dates,
            "symbol": [symbol] * len(dates),
            "open": close,
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "volume": [1000] * len(dates),
            "amount": [100000] * len(dates),
            "pct_change": [0.0] * len(dates),
            "change": [0.0] * len(dates),
            "amplitude": [0.0] * len(dates),
            "turnover": [1.0] * len(dates),
        }
    )


def test_update_daily_cache_initializes_from_requested_start_date() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = _make_workspace_tmp_dir("update_init")
    config = load_config(project_root / "config" / "default.yaml")
    paths = ProjectPaths(root, config.storage)
    storage = Storage(paths)
    provider = RecordingDailyProvider(_make_daily_frame("600000", ["2026-04-10", "2026-04-11"]))

    _update_daily_cache_for_symbol(
        storage=storage,
        provider=provider,
        symbol="600000",
        start_date="20260401",
        end_date="20260411",
        adjust="qfq",
    )

    assert provider.calls == [
        {
            "symbol": "600000",
            "start_date": "20260401",
            "end_date": "20260411",
            "adjust": "qfq",
        }
    ]
    assert storage.load_daily_bars("600000")["trade_date"].dt.date.max().isoformat() == "2026-04-11"


def test_update_daily_cache_appends_from_next_day_even_when_user_start_is_later() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = _make_workspace_tmp_dir("update_incremental")
    config = load_config(project_root / "config" / "default.yaml")
    paths = ProjectPaths(root, config.storage)
    storage = Storage(paths)
    storage.save_daily_bars("600000", _make_daily_frame("600000", ["2026-04-09", "2026-04-10"]))
    provider = RecordingDailyProvider(_make_daily_frame("600000", ["2026-04-11", "2026-04-12"]))

    _update_daily_cache_for_symbol(
        storage=storage,
        provider=provider,
        symbol="600000",
        start_date="20260415",
        end_date="20260420",
        adjust="qfq",
    )

    assert provider.calls[0]["start_date"] == "20260411"
    merged = storage.load_daily_bars("600000")
    assert merged["trade_date"].dt.date.max().isoformat() == "2026-04-12"


def test_update_daily_cache_skips_fetch_when_cached_data_already_covers_end_date() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = _make_workspace_tmp_dir("update_skip")
    config = load_config(project_root / "config" / "default.yaml")
    paths = ProjectPaths(root, config.storage)
    storage = Storage(paths)
    storage.save_daily_bars("600000", _make_daily_frame("600000", ["2026-04-09", "2026-04-10"]))
    provider = RecordingDailyProvider(_make_daily_frame("600000", ["2026-04-11"]))

    _update_daily_cache_for_symbol(
        storage=storage,
        provider=provider,
        symbol="600000",
        start_date="20260401",
        end_date="20260410",
        adjust="qfq",
    )

    assert provider.calls == []
    assert storage.load_daily_bars("600000")["trade_date"].dt.date.astype(str).tolist() == [
        "2026-04-09",
        "2026-04-10",
    ]


def test_update_daily_cache_merges_new_rows_and_deduplicates_by_trade_date() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = _make_workspace_tmp_dir("update_merge")
    config = load_config(project_root / "config" / "default.yaml")
    paths = ProjectPaths(root, config.storage)
    storage = Storage(paths)
    storage.save_daily_bars("600000", _make_daily_frame("600000", ["2026-04-09", "2026-04-10"]))
    provider = RecordingDailyProvider(
        _make_daily_frame("600000", ["2026-04-11", "2026-04-12", "2026-04-12"])
    )

    _update_daily_cache_for_symbol(
        storage=storage,
        provider=provider,
        symbol="600000",
        start_date="20260401",
        end_date="20260420",
        adjust="qfq",
    )

    merged = storage.load_daily_bars("600000")
    assert merged["trade_date"].dt.date.astype(str).tolist() == [
        "2026-04-09",
        "2026-04-10",
        "2026-04-11",
        "2026-04-12",
    ]
