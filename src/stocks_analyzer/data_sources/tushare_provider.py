from __future__ import annotations

import os
from datetime import date, datetime

import pandas as pd

from .base import DataProvider

try:
    import tushare as ts
except ImportError as exc:  # pragma: no cover
    ts = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class TushareDataProvider(DataProvider):
    def __init__(self) -> None:
        if ts is None:
            raise RuntimeError("tushare is not installed. Run `pip install -e .` first.") from _IMPORT_ERROR
        token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("tushare_token")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not set. Please export your Tushare Pro token first.")
        self.pro = ts.pro_api(token)

    def get_instruments(self) -> pd.DataFrame:
        dataframe = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name",
        )
        renamed = dataframe.rename(columns={"symbol": "symbol", "name": "name"})
        normalized = renamed.loc[:, ["symbol", "name"]].copy()
        normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
        normalized["latest_price"] = pd.NA
        normalized["volume"] = pd.NA
        normalized["amount"] = pd.NA
        normalized["turnover_rate"] = pd.NA
        return normalized

    def get_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        raise RuntimeError("TushareDataProvider daily bars are not enabled in this project. Keep `provider` on baostock.")

    def get_intraday_bars(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        ts_code = _to_ts_code(symbol)
        target_date = _parse_date_only(start_datetime)
        if _is_same_day_request(start_datetime, end_datetime) and target_date == date.today():
            dataframe = self.pro.rt_min(ts_code=ts_code, freq=f"{int(period)}MIN")
            renamed = dataframe.rename(
                columns={
                    "time": "timestamp",
                    "vol": "volume",
                }
            )
        else:
            dataframe = self.pro.stk_mins(
                ts_code=ts_code,
                freq=f"{int(period)}min",
                start_date=_normalize_intraday_start(start_datetime),
                end_date=_normalize_intraday_end(end_datetime),
            )
            renamed = dataframe.rename(
                columns={
                    "trade_time": "timestamp",
                    "vol": "volume",
                }
            )
        normalized = renamed.loc[:, ["timestamp", "open", "close", "high", "low", "volume", "amount"]].copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
        normalized["symbol"] = str(symbol).zfill(6)
        return normalized.sort_values("timestamp").reset_index(drop=True)


def _to_ts_code(symbol: str) -> str:
    normalized = str(symbol).strip().zfill(6)
    suffix = ".SH" if normalized.startswith("6") else ".SZ"
    return f"{normalized}{suffix}"


def _parse_date_only(value: str) -> date:
    text = str(value).strip()
    if " " in text:
        return datetime.fromisoformat(text).date()
    return datetime.fromisoformat(text).date()


def _is_same_day_request(start_datetime: str, end_datetime: str) -> bool:
    return _parse_date_only(start_datetime) == _parse_date_only(end_datetime)


def _normalize_intraday_start(value: str) -> str:
    text = str(value).strip()
    if " " in text:
        return text
    return f"{text} 09:00:00"


def _normalize_intraday_end(value: str) -> str:
    text = str(value).strip()
    if " " in text:
        return text
    return f"{text} 15:30:00"
