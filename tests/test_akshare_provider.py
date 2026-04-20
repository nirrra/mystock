from __future__ import annotations

import os

import pandas as pd
from requests import RequestException

import stocks_analyzer.data_sources.akshare_provider as akshare_provider


def test_get_intraday_bars_temporarily_disables_proxy_environment(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": {
                    "klines": [
                        "2026-04-14 09:35:00,10.0,10.1,10.2,9.9,1000,10100.0,1.0,0.5,0.1,0.2",
                    ]
                }
            }

    def fake_get(self, url: str, *, timeout: int, params: dict[str, str], headers: dict[str, str]):
        seen["trust_env"] = self.trust_env
        seen["HTTP_PROXY"] = os.environ.get("HTTP_PROXY")
        seen["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY")
        seen["ALL_PROXY"] = os.environ.get("ALL_PROXY")
        seen["url"] = url
        seen["params"] = params
        seen["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(akshare_provider.requests.Session, "get", fake_get)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:7897")

    provider = akshare_provider.AKShareDataProvider()
    frame = provider.get_intraday_bars(
        symbol="002850",
        start_datetime="2026-04-14",
        end_datetime="2026-04-14",
        period="5",
        adjust="qfq",
    )

    assert seen["trust_env"] is False
    assert seen["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert seen["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert seen["ALL_PROXY"] == "http://127.0.0.1:7897"
    assert str(seen["url"]).endswith("/api/qt/stock/kline/get")
    assert seen["params"]["secid"] == "0.002850"
    assert "Mozilla/5.0" in str(seen["headers"]["User-Agent"])
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["ALL_PROXY"] == "http://127.0.0.1:7897"
    assert frame.loc[0, "symbol"] == "002850"
    assert str(frame.loc[0, "timestamp"]) == "2026-04-14 09:35:00"


def test_get_intraday_bars_falls_back_to_trends_and_aggregates(monkeypatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(self, url: str, *, timeout: int, params: dict[str, str], headers: dict[str, str]):
        calls.append(url)
        if url.endswith("/api/qt/stock/kline/get"):
            raise RequestException("kline blocked")
        return FakeResponse(
            {
                "data": {
                    "trends": [
                        "2026-04-14 09:31:00,10.0,10.1,10.2,9.9,100,1010,10.05",
                        "2026-04-14 09:32:00,10.1,10.2,10.3,10.0,200,2040,10.15",
                        "2026-04-14 09:33:00,10.2,10.3,10.4,10.1,300,3090,10.25",
                        "2026-04-14 09:34:00,10.3,10.4,10.5,10.2,400,4160,10.35",
                        "2026-04-14 09:35:00,10.4,10.5,10.6,10.3,500,5250,10.45",
                    ]
                }
            }
        )

    monkeypatch.setattr(akshare_provider.requests.Session, "get", fake_get)

    provider = akshare_provider.AKShareDataProvider()
    frame = provider.get_intraday_bars(
        symbol="002850",
        start_datetime="2026-04-14",
        end_datetime="2026-04-14",
        period="5",
        adjust="qfq",
    )

    assert len(frame) == 1
    assert calls[0].endswith("/api/qt/stock/kline/get")
    assert calls[1].endswith("/api/qt/stock/trends2/get")
    assert str(frame.loc[0, "timestamp"]) == "2026-04-14 09:35:00"
    assert float(frame.loc[0, "open"]) == 10.0
    assert float(frame.loc[0, "close"]) == 10.5
    assert float(frame.loc[0, "high"]) == 10.6
    assert float(frame.loc[0, "low"]) == 9.9
    assert float(frame.loc[0, "volume"]) == 1500.0
