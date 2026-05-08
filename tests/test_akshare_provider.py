from __future__ import annotations

import pandas as pd
from requests import RequestException

import stocks_analyzer.data_sources.akshare_provider as akshare_provider


def test_get_daily_bars_defaults_to_sina(monkeypatch) -> None:
    def fake_daily(**kwargs):
        return pd.DataFrame(
            [
                {
                    "date": "2026-04-14",
                    "open": 10.0,
                    "high": 10.3,
                    "low": 9.9,
                    "close": 10.2,
                    "volume": 1000000,
                    "amount": 10200000,
                    "turnover": 0.02,
                }
            ]
        )

    monkeypatch.setattr(akshare_provider.ak, "stock_zh_a_daily", fake_daily)

    provider = akshare_provider.AKShareDataProvider()
    frame = provider.get_daily_bars(symbol="600004", start_date="20260414", end_date="20260414", adjust="qfq")

    assert provider._daily_backend == "sina"
    assert frame.loc[0, "symbol"] == "600004"
    assert float(frame.loc[0, "amount"]) == 10200000.0
    assert "pct_change" in frame.columns
    assert "change" in frame.columns
    assert "amplitude" in frame.columns


def test_get_daily_bars_falls_back_to_eastmoney_when_sina_fails(monkeypatch) -> None:
    def fake_daily(**kwargs):
        raise RequestException("sina blocked")

    def fake_hist(**kwargs):
        return pd.DataFrame(
            [
                {
                    "日期": "2026-04-14",
                    "股票代码": "600004",
                    "开盘": 10.0,
                    "收盘": 10.2,
                    "最高": 10.3,
                    "最低": 9.9,
                    "成交量": 1000000,
                    "成交额": 10200000,
                    "振幅": 3.92,
                    "涨跌幅": 2.0,
                    "涨跌额": 0.2,
                    "换手率": 0.02,
                }
            ]
        )

    monkeypatch.setattr(akshare_provider.ak, "stock_zh_a_daily", fake_daily)
    monkeypatch.setattr(akshare_provider.ak, "stock_zh_a_hist", fake_hist)

    provider = akshare_provider.AKShareDataProvider()
    frame = provider.get_daily_bars(symbol="600004", start_date="20260414", end_date="20260414", adjust="qfq")

    assert provider._daily_backend == "eastmoney"
    assert frame.loc[0, "symbol"] == "600004"
    assert float(frame.loc[0, "pct_change"]) == 2.0
