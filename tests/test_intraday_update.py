from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.intraday_update import load_latest_watchlist_symbols, run_intraday_update
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_storage(tmp_path: Path) -> Storage:
    config = load_config(ROOT / "config" / "default.yaml")
    return Storage(ProjectPaths(tmp_path, config.storage))


def _write_watchlist(project_root: Path, trade_date: str, symbols: list[str]) -> Path:
    target = project_root / "reports" / "watchlists" / f"watchlist_{trade_date}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"candidates": [{"symbol": symbol} for symbol in symbols]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def _quotes(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "2026-05-08",
                "symbol": symbol,
                "name": f"测试{symbol}",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "pre_close": 10.1,
                "volume": 100000,
                "amount": 1020000,
                "pct_change": 0.99,
                "quote_date": "2026-05-08",
                "quote_time": "13:20:00",
                "quote_datetime": "2026-05-08 13:20:00",
            }
            for symbol in symbols
        ]
    )


def test_load_latest_watchlist_symbols_uses_latest_main_watchlist() -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_latest_watchlist")
    _write_watchlist(tmp_path, "2026-05-06", ["600000"])
    latest = _write_watchlist(tmp_path, "2026-05-07", ["600001", "600002"])
    pattern = tmp_path / "reports" / "watchlists" / "watchlist_pattern_2026-05-08.json"
    pattern.write_text(json.dumps({"candidates": [{"symbol": "600999"}]}), encoding="utf-8")

    path, symbols = load_latest_watchlist_symbols(tmp_path)

    assert path == latest
    assert symbols == ["600001", "600002"]


def test_run_intraday_update_replaces_existing_intraday_files(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_replace")
    storage = _make_storage(tmp_path)
    _write_watchlist(tmp_path, "2026-05-07", ["600001", "600002"])
    storage.save_intraday_bars("600001", pd.DataFrame([{"symbol": "600001"}, {"symbol": "600001"}]))

    def fake_fetch(symbols: list[str], *, source: str, timeout_seconds: float, chunk_size: int) -> pd.DataFrame:
        assert source == "eastmoney_direct"
        assert symbols == ["600001", "600002"]
        return _quotes(symbols)

    monkeypatch.setattr("stocks_analyzer.intraday_update.fetch_intraday_quotes", fake_fetch)

    result = run_intraday_update(storage=storage, project_root=tmp_path, watchlist_only=True)

    assert result.updated_symbols == ["600001", "600002"]
    assert result.failed_symbols == []
    assert (tmp_path / "data" / "intraday" / "600001.parquet").exists()
    assert not (tmp_path / "data" / "daily" / "600001.parquet").exists()

    replaced = storage.load_intraday_bars("600001")
    assert len(replaced) == 1
    assert replaced.loc[0, "source"] == "eastmoney_direct"
    assert bool(replaced.loc[0, "provisional"]) is True


def test_run_intraday_update_defaults_to_universe_symbols(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_universe")
    storage = _make_storage(tmp_path)
    storage.save_universe(pd.DataFrame([{"symbol": "600010", "name": "测试A"}, {"symbol": "600011", "name": "测试B"}]))

    def fake_fetch(symbols: list[str], *, source: str, timeout_seconds: float, chunk_size: int) -> pd.DataFrame:
        assert symbols == ["600010", "600011"]
        return _quotes(symbols)

    monkeypatch.setattr("stocks_analyzer.intraday_update.fetch_intraday_quotes", fake_fetch)

    result = run_intraday_update(storage=storage, project_root=tmp_path)

    assert result.source_watchlist_path is None
    assert result.updated_symbols == ["600010", "600011"]


def test_run_intraday_update_accepts_explicit_symbols_and_sina_source(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_explicit")
    storage = _make_storage(tmp_path)

    def fake_fetch(symbols: list[str], *, source: str, timeout_seconds: float, chunk_size: int) -> pd.DataFrame:
        assert source == "sina_raw"
        assert symbols == ["600003"]
        return _quotes(symbols)

    monkeypatch.setattr("stocks_analyzer.intraday_update.fetch_intraday_quotes", fake_fetch)

    result = run_intraday_update(
        storage=storage,
        project_root=tmp_path,
        source="sina_raw",
        symbols=["600003"],
    )

    assert result.source_watchlist_path is None
    assert result.updated_symbols == ["600003"]
    saved = storage.load_intraday_bars("600003")
    assert saved.loc[0, "source"] == "sina_raw"
