from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests import RequestException

from .base import DataProvider


ITICK_API_URL = "https://api.itick.org/stock/kline"
ITICK_BATCH_API_URL = "https://api.itick.org/stock/klines"
ITICK_BATCH_SIZE = 10
KTYPE_MAP = {
    "1": 1,
    "5": 2,
    "15": 3,
    "30": 4,
    "60": 5,
    "D": 8,
    "W": 9,
    "M": 10,
}


class ITickDataProvider(DataProvider):
    def __init__(self) -> None:
        token = os.environ.get("ITICK_TOKEN") or os.environ.get("itick_token")
        if not token:
            raise RuntimeError("ITICK_TOKEN is not set. Please export your iTick API token first.")
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {
                "accept": "application/json",
                "token": token,
            }
        )

    def get_instruments(self) -> pd.DataFrame:
        raise RuntimeError("ITickDataProvider does not implement instrument universe loading in this project.")

    def get_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        raise RuntimeError("ITickDataProvider daily bars are not enabled in this project. Keep `provider` on baostock.")

    def get_intraday_bars(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        del adjust
        k_type = KTYPE_MAP.get(str(period))
        if k_type is None:
            raise ValueError(f"Unsupported iTick intraday period: {period}")

        start_dt = _normalize_start(start_datetime)
        end_dt = _normalize_end(end_datetime)
        response = self.session.get(
            ITICK_API_URL,
            params={
                "region": _to_region(symbol),
                "code": str(symbol).zfill(6),
                "kType": k_type,
                "et": int(end_dt.timestamp() * 1000),
                "limit": _estimate_limit(start_dt=start_dt, end_dt=end_dt, period=str(period)),
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("code", -1)) != 0:
            raise RuntimeError(f"iTick API error: {payload.get('msg') or payload.get('code')}")

        data = payload.get("data") or []
        if not isinstance(data, list) or not data:
            return pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume", "amount", "symbol"])

        frame = pd.DataFrame(data)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume", "amount", "symbol"])

        renamed = frame.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "c": "close",
                "h": "high",
                "l": "low",
                "v": "volume",
                "tu": "amount",
            }
        )
        normalized = renamed.loc[:, ["timestamp", "open", "close", "high", "low", "volume", "amount"]].copy()
        normalized["timestamp"] = (
            pd.to_datetime(normalized["timestamp"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
        normalized = normalized[(normalized["timestamp"] >= start_dt) & (normalized["timestamp"] <= end_dt)]
        normalized["symbol"] = str(symbol).zfill(6)
        for column in ("open", "close", "high", "low", "volume", "amount"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized.sort_values("timestamp").reset_index(drop=True)

    def get_intraday_bars_batch(
        self,
        *,
        symbols: list[str],
        start_datetime: str,
        end_datetime: str,
        period: str,
        adjust: str,
    ) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
        del adjust
        k_type = KTYPE_MAP.get(str(period))
        if k_type is None:
            raise ValueError(f"Unsupported iTick intraday period: {period}")

        start_dt = _normalize_start(start_datetime)
        end_dt = _normalize_end(end_datetime)
        grouped_symbols = _group_symbols_by_region(symbols)
        result: dict[str, pd.DataFrame] = {}
        failures: list[dict[str, str]] = []

        for region, region_symbols in grouped_symbols.items():
            for batch in _chunked(region_symbols, ITICK_BATCH_SIZE):
                try:
                    response = self.session.get(
                        ITICK_BATCH_API_URL,
                        params={
                            "region": region,
                            "codes": ",".join(batch),
                            "kType": k_type,
                            "et": int(end_dt.timestamp() * 1000),
                            "limit": _estimate_limit(start_dt=start_dt, end_dt=end_dt, period=str(period)),
                        },
                        timeout=15,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if int(payload.get("code", -1)) != 0:
                        raise RuntimeError(f"iTick API error: {payload.get('msg') or payload.get('code')}")
                    batch_data = payload.get("data") or {}
                    if not isinstance(batch_data, dict):
                        raise RuntimeError("iTick batch API returned unexpected data structure")
                    for symbol in batch:
                        result[str(symbol).zfill(6)] = _normalize_itick_rows(
                            rows=batch_data.get(str(symbol).zfill(6)) or batch_data.get(str(int(symbol))) or [],
                            symbol=symbol,
                            start_dt=start_dt,
                            end_dt=end_dt,
                        )
                except Exception as exc:
                    for symbol in batch:
                        failures.append(
                            {
                                "symbol": str(symbol).zfill(6),
                                "name": "",
                                "error": str(exc),
                            }
                        )
        return result, failures

    def close(self) -> None:
        self.session.close()


def _to_region(symbol: str) -> str:
    normalized = str(symbol).strip().zfill(6)
    return "SH" if normalized.startswith("6") else "SZ"


def _normalize_start(value: str) -> datetime:
    text = str(value).strip()
    if " " in text:
        return datetime.fromisoformat(text)
    return datetime.fromisoformat(f"{text} 09:30:00")


def _normalize_end(value: str) -> datetime:
    text = str(value).strip()
    if " " in text:
        return datetime.fromisoformat(text)
    return datetime.fromisoformat(f"{text} 15:00:00")


def _estimate_limit(*, start_dt: datetime, end_dt: datetime, period: str) -> int:
    period_minutes = int(period)
    delta = max(end_dt - start_dt, timedelta(minutes=period_minutes))
    bars = int(delta.total_seconds() // (period_minutes * 60)) + 5
    return max(10, min(bars, 500))


def _normalize_itick_rows(
    *,
    rows: list[dict[str, object]],
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume", "amount", "symbol"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume", "amount", "symbol"])
    renamed = frame.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "c": "close",
            "h": "high",
            "l": "low",
            "v": "volume",
            "tu": "amount",
        }
    )
    normalized = renamed.loc[:, ["timestamp", "open", "close", "high", "low", "volume", "amount"]].copy()
    normalized["timestamp"] = (
        pd.to_datetime(normalized["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    )
    normalized = normalized[(normalized["timestamp"] >= start_dt) & (normalized["timestamp"] <= end_dt)]
    normalized["symbol"] = str(symbol).zfill(6)
    for column in ("open", "close", "high", "low", "volume", "amount"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.sort_values("timestamp").reset_index(drop=True)


def _group_symbols_by_region(symbols: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for symbol in symbols:
        normalized = str(symbol).zfill(6)
        grouped.setdefault(_to_region(normalized), []).append(normalized)
    return grouped


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
