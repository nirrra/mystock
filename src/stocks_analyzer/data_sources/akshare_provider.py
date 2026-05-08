from __future__ import annotations

import time

import pandas as pd
from requests import RequestException

from .base import DataProvider

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover
    ak = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class AKShareDataProvider(DataProvider):
    def __init__(self, *, daily_backend: str = "sina") -> None:
        if ak is None:
            raise RuntimeError("akshare is not installed. Run `pip install -e .` first.") from _IMPORT_ERROR
        normalized_backend = str(daily_backend).strip().lower()
        if normalized_backend not in {"sina", "eastmoney"}:
            raise ValueError(f"Unsupported AKShare daily backend: {daily_backend}")
        self._daily_backend = normalized_backend

    def get_instruments(self) -> pd.DataFrame:
        dataframe = ak.stock_info_a_code_name()
        renamed = dataframe.rename(columns={"code": "symbol", "name": "name"})
        normalized = renamed.loc[:, ["symbol", "name"]].copy()
        normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
        normalized["latest_price"] = pd.NA
        normalized["volume"] = pd.NA
        normalized["amount"] = pd.NA
        normalized["turnover_rate"] = pd.NA
        return normalized

    def get_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        dataframe = self._retry_fetch_daily_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        return _normalize_daily_bars(dataframe, symbol=symbol)

    def get_index_daily_bars(self, index_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        normalized_index_symbol = _normalize_index_symbol(index_symbol)
        dataframe = ak.index_zh_a_hist(
            symbol=normalized_index_symbol[2:],
            period="daily",
            start_date=start_date,
            end_date=end_date,
        )
        return _normalize_daily_bars(dataframe, symbol=normalized_index_symbol)

    def _retry_fetch_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return self._fetch_daily_bars(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)
            except RequestException as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(attempt)

        if self._daily_backend == "sina":
            try:
                dataframe = self._fetch_eastmoney_daily_bars(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)
                self._daily_backend = "eastmoney"
                return dataframe
            except RequestException:
                pass

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to fetch daily bars for {symbol}")

    def _fetch_daily_bars(self, *, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        if self._daily_backend == "sina":
            return ak.stock_zh_a_daily(
                symbol=_with_exchange_prefix(symbol),
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        return self._fetch_eastmoney_daily_bars(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)

    def _fetch_eastmoney_daily_bars(self, *, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )


def _normalize_daily_bars(dataframe: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    rename_map = {
        "日期": "trade_date",
        "date": "trade_date",
        "股票代码": "symbol",
        "开盘": "open",
        "open": "open",
        "收盘": "close",
        "close": "close",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "成交量": "volume",
        "volume": "volume",
        "成交额": "amount",
        "amount": "amount",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "振幅": "amplitude",
        "换手率": "turnover",
        "turnover": "turnover",
    }
    renamed = dataframe.rename(columns={key: value for key, value in rename_map.items() if key in dataframe.columns}).copy()
    renamed["trade_date"] = pd.to_datetime(renamed["trade_date"])
    renamed["symbol"] = str(symbol).zfill(6)

    for column in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
        else:
            renamed[column] = pd.NA

    if "change" in renamed.columns:
        renamed["change"] = pd.to_numeric(renamed["change"], errors="coerce")
    else:
        renamed["change"] = renamed["close"].diff()

    if "pct_change" in renamed.columns:
        renamed["pct_change"] = pd.to_numeric(renamed["pct_change"], errors="coerce")
    else:
        renamed["pct_change"] = renamed["close"].pct_change().mul(100)

    if "amplitude" in renamed.columns:
        renamed["amplitude"] = pd.to_numeric(renamed["amplitude"], errors="coerce")
    else:
        renamed["amplitude"] = (
            (renamed["high"] - renamed["low"]).div(renamed["close"].replace(0, pd.NA)).mul(100)
        )

    return renamed.loc[
        :,
        [
            "trade_date",
            "symbol",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "pct_change",
            "change",
            "amplitude",
            "turnover",
        ],
    ].sort_values("trade_date").reset_index(drop=True)


def _with_exchange_prefix(symbol: str) -> str:
    code = str(symbol).zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh{code}"
    return f"sz{code}"


def _normalize_index_symbol(index_symbol: str) -> str:
    text = str(index_symbol).strip().lower().replace(".", "")
    if text.startswith(("sh", "sz")):
        return f"{text[:2]}{text[2:].zfill(6)}"
    code = text.zfill(6)
    prefix = "sz" if code.startswith("399") else "sh"
    return f"{prefix}{code}"
