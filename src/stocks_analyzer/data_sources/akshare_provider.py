from __future__ import annotations

import time

import pandas as pd
import requests
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
        dataframe = self._retry_fetch_intraday_bars(
            symbol=symbol,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
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

    def _retry_fetch_intraday_bars(
        self,
        *,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return _fetch_eastmoney_intraday_bars(
                    symbol=symbol,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    period=period,
                    adjust=adjust,
                )
            except (RequestException, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to fetch intraday bars for {symbol}")


def _fetch_eastmoney_intraday_bars(
    *,
    symbol: str,
    start_datetime: str,
    end_datetime: str,
    period: str,
    adjust: str,
) -> pd.DataFrame:
    market_code = 1 if str(symbol).startswith("6") else 0
    adjust_map = {"": "0", "qfq": "1", "hfq": "2"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }

    session = requests.Session()
    session.trust_env = False
    try:
        if period == "1":
            temp_df = _fetch_eastmoney_trends(
                session=session,
                headers=headers,
                market_code=market_code,
                symbol=symbol,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
        else:
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "klt": period,
                "fqt": adjust_map[adjust],
                "secid": f"{market_code}.{symbol}",
                "beg": "0",
                "end": "20500000",
            }
            try:
                response = session.get(url, timeout=15, params=params, headers=headers)
                response.raise_for_status()
                data_json = response.json()
                temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
                temp_df.columns = ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
            except (RequestException, KeyError, TypeError, ValueError):
                minute_frame = _fetch_eastmoney_trends(
                    session=session,
                    headers=headers,
                    market_code=market_code,
                    symbol=symbol,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                )
                temp_df = _aggregate_trends_to_period(minute_frame, period=period)
        temp_df.index = pd.to_datetime(temp_df["时间"])
        temp_df = temp_df[start_datetime:end_datetime]
        temp_df.reset_index(drop=True, inplace=True)
        for column in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
            temp_df[column] = pd.to_numeric(temp_df[column], errors="coerce")
        if "均价" in temp_df.columns:
            temp_df["均价"] = pd.to_numeric(temp_df["均价"], errors="coerce")
        if "振幅" in temp_df.columns:
            temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
            temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
            temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
            temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
            temp_df = temp_df[
                ["时间", "开盘", "收盘", "最高", "最低", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", "换手率"]
            ]
        temp_df["时间"] = pd.to_datetime(temp_df["时间"]).astype(str)
        return temp_df
    finally:
        session.close()


def _fetch_eastmoney_trends(
    *,
    session: requests.Session,
    headers: dict[str, str],
    market_code: int,
    symbol: str,
    start_datetime: str,
    end_datetime: str,
) -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "ndays": "5",
        "iscr": "0",
        "secid": f"{market_code}.{symbol}",
    }
    response = session.get(url, timeout=15, params=params, headers=headers)
    response.raise_for_status()
    data_json = response.json()
    temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["trends"]])
    temp_df.columns = ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "均价"]
    temp_df.index = pd.to_datetime(temp_df["时间"])
    temp_df = temp_df[start_datetime:end_datetime]
    temp_df.reset_index(drop=True, inplace=True)
    for column in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "均价"]:
        temp_df[column] = pd.to_numeric(temp_df[column], errors="coerce")
    temp_df["时间"] = pd.to_datetime(temp_df["时间"]).astype(str)
    return temp_df


def _aggregate_trends_to_period(dataframe: pd.DataFrame, *, period: str) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    minutes = int(period)
    frame = dataframe.copy()
    frame["时间"] = pd.to_datetime(frame["时间"])
    frame = frame.set_index("时间")
    aggregated = frame.resample(f"{minutes}min", label="right", closed="right").agg(
        {
            "开盘": "first",
            "收盘": "last",
            "最高": "max",
            "最低": "min",
            "成交量": "sum",
            "成交额": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["开盘", "收盘", "最高", "最低"]).reset_index()
    aggregated["涨跌幅"] = pd.NA
    aggregated["涨跌额"] = pd.NA
    aggregated["振幅"] = pd.NA
    aggregated["换手率"] = pd.NA
    aggregated["时间"] = aggregated["时间"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return aggregated[
        ["时间", "开盘", "收盘", "最高", "最低", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", "换手率"]
    ]
