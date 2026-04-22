from __future__ import annotations

from datetime import date

import pandas as pd

import stocks_analyzer.trading_calendar as trading_calendar


def test_is_trading_day_uses_akshare_for_non_akshare_provider(monkeypatch) -> None:
    calls: list[str] = []

    class FakeAkshare:
        @staticmethod
        def tool_trade_date_hist_sina() -> pd.DataFrame:
            calls.append("tool_trade_date_hist_sina")
            return pd.DataFrame({"trade_date": ["2026-04-22", "2026-04-23"]})

    monkeypatch.setattr(trading_calendar, "ak", FakeAkshare())

    assert trading_calendar.is_trading_day("baostock", date(2026, 4, 22)) is True
    assert calls == ["tool_trade_date_hist_sina"]


def test_is_trading_day_falls_back_to_weekday_when_akshare_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(trading_calendar, "ak", None)

    assert trading_calendar.is_trading_day("baostock", date(2026, 4, 22)) is True
    assert trading_calendar.is_trading_day("baostock", date(2026, 4, 25)) is False
