from __future__ import annotations

from datetime import date

import pandas as pd

import stocks_analyzer.data_sources.tushare_provider as tushare_provider


class FakePro:
    def __init__(self) -> None:
        self.rt_calls: list[dict[str, str]] = []
        self.stk_calls: list[dict[str, str]] = []

    def rt_min(self, *, ts_code: str, freq: str) -> pd.DataFrame:
        self.rt_calls.append({"ts_code": ts_code, "freq": freq})
        return pd.DataFrame(
            [
                {
                    "time": "2026-04-14 09:35:00",
                    "open": 10.0,
                    "close": 10.1,
                    "high": 10.2,
                    "low": 9.9,
                    "vol": 1000,
                    "amount": 10100.0,
                }
            ]
        )

    def stk_mins(self, *, ts_code: str, freq: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.stk_calls.append(
            {
                "ts_code": ts_code,
                "freq": freq,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return pd.DataFrame(
            [
                {
                    "trade_time": "2026-04-11 09:35:00",
                    "open": 10.0,
                    "close": 10.1,
                    "high": 10.2,
                    "low": 9.9,
                    "vol": 1000,
                    "amount": 10100.0,
                }
            ]
        )


class FakeTs:
    def __init__(self, pro: FakePro) -> None:
        self._pro = pro
        self.tokens: list[str] = []

    def pro_api(self, token: str) -> FakePro:
        self.tokens.append(token)
        return self._pro


def test_tushare_provider_uses_rt_min_for_today(monkeypatch) -> None:
    pro = FakePro()
    fake_ts = FakeTs(pro)
    monkeypatch.setattr(tushare_provider, "ts", fake_ts)
    monkeypatch.setattr(tushare_provider, "_IMPORT_ERROR", None)
    monkeypatch.setenv("TUSHARE_TOKEN", "demo-token")
    monkeypatch.setattr(tushare_provider, "date", type("FakeDate", (), {"today": staticmethod(lambda: date(2026, 4, 14))}))

    provider = tushare_provider.TushareDataProvider()
    frame = provider.get_intraday_bars(
        symbol="002850",
        start_datetime="2026-04-14",
        end_datetime="2026-04-14",
        period="5",
        adjust="qfq",
    )

    assert fake_ts.tokens == ["demo-token"]
    assert pro.rt_calls == [{"ts_code": "002850.SZ", "freq": "5MIN"}]
    assert pro.stk_calls == []
    assert frame.loc[0, "symbol"] == "002850"
    assert str(frame.loc[0, "timestamp"]) == "2026-04-14 09:35:00"


def test_tushare_provider_uses_stk_mins_for_history(monkeypatch) -> None:
    pro = FakePro()
    fake_ts = FakeTs(pro)
    monkeypatch.setattr(tushare_provider, "ts", fake_ts)
    monkeypatch.setattr(tushare_provider, "_IMPORT_ERROR", None)
    monkeypatch.setenv("TUSHARE_TOKEN", "demo-token")
    monkeypatch.setattr(tushare_provider, "date", type("FakeDate", (), {"today": staticmethod(lambda: date(2026, 4, 14))}))

    provider = tushare_provider.TushareDataProvider()
    frame = provider.get_intraday_bars(
        symbol="600000",
        start_datetime="2026-04-11",
        end_datetime="2026-04-11",
        period="5",
        adjust="qfq",
    )

    assert pro.rt_calls == []
    assert pro.stk_calls == [
        {
            "ts_code": "600000.SH",
            "freq": "5min",
            "start_date": "2026-04-11 09:00:00",
            "end_date": "2026-04-11 15:30:00",
        }
    ]
    assert frame.loc[0, "symbol"] == "600000"


def test_tushare_provider_requires_token(monkeypatch) -> None:
    pro = FakePro()
    fake_ts = FakeTs(pro)
    monkeypatch.setattr(tushare_provider, "ts", fake_ts)
    monkeypatch.setattr(tushare_provider, "_IMPORT_ERROR", None)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("tushare_token", raising=False)

    try:
        tushare_provider.TushareDataProvider()
    except RuntimeError as exc:
        assert "TUSHARE_TOKEN" in str(exc)
    else:
        raise AssertionError("expected missing token error")
