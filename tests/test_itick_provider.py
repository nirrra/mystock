from __future__ import annotations

import pandas as pd

import stocks_analyzer.data_sources.itick_provider as itick_provider


def _to_epoch_ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="Asia/Shanghai").tz_convert("UTC").timestamp() * 1000)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_itick_provider_requires_token(monkeypatch) -> None:
    monkeypatch.delenv("ITICK_TOKEN", raising=False)
    monkeypatch.delenv("itick_token", raising=False)

    try:
        itick_provider.ITickDataProvider()
    except RuntimeError as exc:
        assert "ITICK_TOKEN" in str(exc)
    else:
        raise AssertionError("expected missing token error")


def test_itick_provider_maps_stock_kline_response(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_get(self, url: str, *, params: dict[str, object], timeout: int):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(
            {
                    "code": 0,
                    "msg": "success",
                    "data": [
                        {
                            "t": _to_epoch_ms("2026-04-13 09:40:00"),
                            "o": "10.0",
                            "c": "10.2",
                            "h": "10.3",
                        "l": "9.9",
                        "v": "1000",
                        "tu": "10200",
                    }
                ],
            }
        )

    monkeypatch.setenv("ITICK_TOKEN", "demo-token")
    monkeypatch.setattr(itick_provider.requests.Session, "get", fake_get)

    provider = itick_provider.ITickDataProvider()
    frame = provider.get_intraday_bars(
        symbol="002850",
        start_datetime="2026-04-13",
        end_datetime="2026-04-13",
        period="5",
        adjust="qfq",
    )
    provider.close()

    assert str(seen["url"]) == itick_provider.ITICK_API_URL
    assert seen["params"]["region"] == "SZ"
    assert seen["params"]["code"] == "002850"
    assert seen["params"]["kType"] == 2
    assert frame.loc[0, "symbol"] == "002850"
    assert str(frame.loc[0, "timestamp"]) == "2026-04-13 09:40:00"
    assert float(frame.loc[0, "amount"]) == 10200.0


def test_itick_provider_filters_rows_outside_requested_window(monkeypatch) -> None:
    def fake_get(self, url: str, *, params: dict[str, object], timeout: int):
        return FakeResponse(
            {
                "code": 0,
                "msg": "success",
                "data": [
                    {"t": _to_epoch_ms("2026-04-13 09:30:00"), "o": "9.8", "c": "9.9", "h": "10.0", "l": "9.7", "v": "500", "tu": "4950"},
                    {"t": _to_epoch_ms("2026-04-13 09:40:00"), "o": "10.0", "c": "10.2", "h": "10.3", "l": "9.9", "v": "1000", "tu": "10200"},
                ],
            }
        )

    monkeypatch.setenv("ITICK_TOKEN", "demo-token")
    monkeypatch.setattr(itick_provider.requests.Session, "get", fake_get)

    provider = itick_provider.ITickDataProvider()
    frame = provider.get_intraday_bars(
        symbol="600000",
        start_datetime="2026-04-13 09:35:00",
        end_datetime="2026-04-13 09:45:00",
        period="5",
        adjust="qfq",
    )
    provider.close()

    assert frame["symbol"].tolist() == ["600000"]
    assert str(frame.loc[0, "timestamp"]) == "2026-04-13 09:40:00"


def test_itick_provider_batches_symbols_by_region(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_get(self, url: str, *, params: dict[str, object], timeout: int):
        calls.append({"url": url, "params": params})
        codes = str(params["codes"]).split(",")
        return FakeResponse(
            {
                "code": 0,
                "msg": "success",
                "data": {
                    code: [
                        {
                            "t": _to_epoch_ms("2026-04-14 09:35:00"),
                            "o": "10.0",
                            "c": "10.2",
                            "h": "10.3",
                            "l": "9.9",
                            "v": "1000",
                            "tu": "10200",
                        }
                    ]
                    for code in codes
                },
            }
        )

    monkeypatch.setenv("ITICK_TOKEN", "demo-token")
    monkeypatch.setattr(itick_provider.requests.Session, "get", fake_get)

    provider = itick_provider.ITickDataProvider()
    frames, failures = provider.get_intraday_bars_batch(
        symbols=["600000", "600001", "002850"],
        start_datetime="2026-04-14",
        end_datetime="2026-04-14",
        period="5",
        adjust="qfq",
    )
    provider.close()

    assert failures == []
    assert len(calls) == 2
    assert calls[0]["url"] == itick_provider.ITICK_BATCH_API_URL
    assert calls[0]["params"]["region"] == "SH"
    assert calls[0]["params"]["codes"] == "600000,600001"
    assert calls[1]["params"]["region"] == "SZ"
    assert calls[1]["params"]["codes"] == "002850"
    assert sorted(frames.keys()) == ["002850", "600000", "600001"]
    assert str(frames["002850"].loc[0, "timestamp"]) == "2026-04-14 09:35:00"
