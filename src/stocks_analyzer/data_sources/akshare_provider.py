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
    def __init__(self) -> None:
        if ak is None:
            raise RuntimeError("akshare is not installed. Run `pip install -e .` first.") from _IMPORT_ERROR

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
        renamed = dataframe.rename(
            columns={
                "日期": "trade_date",
                "股票代码": "symbol",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "振幅": "amplitude",
                "换手率": "turnover",
            }
        )
        normalized = renamed.loc[
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
        ].copy()
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"])
        normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
        return normalized.sort_values("trade_date").reset_index(drop=True)

    def get_intraday_bars(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        dataframe = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            start_date=start_datetime,
            end_date=end_datetime,
            period=period,
            adjust=adjust,
        )
        rename_map = {
            "时间": "timestamp",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "均价": "average_price",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "振幅": "amplitude",
            "换手率": "turnover",
        }
        renamed = dataframe.rename(columns={key: value for key, value in rename_map.items() if key in dataframe.columns})
        renamed["symbol"] = symbol
        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"])
        return renamed.sort_values("timestamp").reset_index(drop=True)

    def _retry_fetch_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            except RequestException as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to fetch daily bars for {symbol}")
