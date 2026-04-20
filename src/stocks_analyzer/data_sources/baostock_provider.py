from __future__ import annotations

import io
import logging
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from time import sleep

import pandas as pd

from .base import DataProvider

try:
    import baostock as bs
except ImportError as exc:  # pragma: no cover
    bs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


ADJUSTMENT_MAP = {
    "hfq": "1",
    "qfq": "2",
    "": "3",
    "bfq": "3",
    None: "3",
}

DEFAULT_LOGIN_ATTEMPTS = 3
DEFAULT_LOGIN_RETRY_DELAY_SECONDS = 1.0


def login_baostock(
    *,
    attempts: int = DEFAULT_LOGIN_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_LOGIN_RETRY_DELAY_SECONDS,
    logger: logging.Logger | None = None,
    silence_output: bool = True,
):
    if bs is None:
        raise RuntimeError("baostock is not installed. Run `pip install baostock` first.") from _IMPORT_ERROR

    resolved_logger = logger or logging.getLogger(__name__)
    attempt_count = max(1, int(attempts))
    last_message = ""

    for attempt in range(1, attempt_count + 1):
        buffer = io.StringIO() if silence_output else None
        context = (
            redirect_stdout(buffer),
            redirect_stderr(buffer),
        ) if buffer is not None else (nullcontext(), nullcontext())
        try:
            with context[0], context[1]:
                login_result = bs.login()
        except Exception as exc:
            last_message = str(exc)
            if attempt >= attempt_count:
                raise RuntimeError(f"baostock login failed after {attempt_count} attempt(s): {last_message}") from exc
            resolved_logger.warning(
                "BaoStock login attempt %s/%s raised %s; retrying in %.1fs",
                attempt,
                attempt_count,
                last_message,
                retry_delay_seconds,
            )
            sleep(retry_delay_seconds)
            continue

        if login_result.error_code == "0":
            return login_result

        last_message = f"{login_result.error_code} {login_result.error_msg}".strip()
        if attempt >= attempt_count:
            break

        resolved_logger.warning(
            "BaoStock login attempt %s/%s failed: %s; retrying in %.1fs",
            attempt,
            attempt_count,
            last_message,
            retry_delay_seconds,
        )
        sleep(retry_delay_seconds)

    raise RuntimeError(f"baostock login failed after {attempt_count} attempt(s): {last_message}")


class BaoStockDataProvider(DataProvider):
    def __init__(self) -> None:
        login_baostock()

    def get_instruments(self) -> pd.DataFrame:
        result = bs.query_all_stock()
        dataframe = _result_to_dataframe(result)
        renamed = dataframe.rename(columns={"code": "raw_symbol", "code_name": "name", "tradeStatus": "trade_status"})
        normalized = renamed.loc[:, ["raw_symbol", "name", "trade_status"]].copy()
        normalized["symbol"] = normalized["raw_symbol"].map(_strip_exchange_prefix)
        normalized["latest_price"] = pd.NA
        normalized["volume"] = pd.NA
        normalized["amount"] = pd.NA
        normalized["turnover_rate"] = pd.NA
        return normalized.loc[:, ["symbol", "name", "latest_price", "volume", "amount", "turnover_rate", "trade_status"]]

    def get_daily_bars(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        result = bs.query_history_k_data_plus(
            _with_exchange_prefix(symbol),
            "date,code,open,high,low,close,volume,amount,turn,tradestatus,pctChg",
            start_date=_normalize_date(start_date),
            end_date=_normalize_date(end_date),
            frequency="d",
            adjustflag=_map_adjustment(adjust),
        )
        dataframe = _result_to_dataframe(result)
        renamed = dataframe.rename(
            columns={
                "date": "trade_date",
                "code": "raw_symbol",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "amount": "amount",
                "turn": "turnover",
                "tradestatus": "trade_status",
                "pctChg": "pct_change",
            }
        )
        normalized = renamed.copy()
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"])
        normalized["symbol"] = normalized["raw_symbol"].map(_strip_exchange_prefix)

        for column in ["open", "close", "high", "low", "volume", "amount", "turnover", "pct_change"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized["change"] = normalized["close"].diff()
        normalized["amplitude"] = (
            (normalized["high"] - normalized["low"]).div(normalized["close"].replace(0, pd.NA)).mul(100)
        )
        return normalized.loc[
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
                "trade_status",
            ],
        ].sort_values("trade_date").reset_index(drop=True)

    def get_intraday_bars(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        result = bs.query_history_k_data_plus(
            _with_exchange_prefix(symbol),
            "date,time,code,open,high,low,close,volume,amount",
            start_date=_normalize_date(start_datetime),
            end_date=_normalize_date(end_datetime),
            frequency=period,
            adjustflag=_map_adjustment(adjust),
        )
        dataframe = _result_to_dataframe(result)
        renamed = dataframe.rename(columns={"code": "raw_symbol"})
        renamed["symbol"] = renamed["raw_symbol"].map(_strip_exchange_prefix)
        renamed["timestamp"] = renamed["time"].map(_parse_baostock_time)

        for column in ["open", "high", "low", "close", "volume", "amount"]:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")

        return renamed.loc[:, ["timestamp", "symbol", "open", "high", "low", "close", "volume", "amount"]].sort_values(
            "timestamp"
        ).reset_index(drop=True)

    def close(self) -> None:
        bs.logout()


def _result_to_dataframe(result) -> pd.DataFrame:
    if result.error_code != "0":
        raise RuntimeError(f"baostock query failed: {result.error_code} {result.error_msg}")

    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def _strip_exchange_prefix(symbol: str) -> str:
    return symbol.split(".", maxsplit=1)[-1].zfill(6)


def _with_exchange_prefix(symbol: str) -> str:
    code = str(symbol).zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh.{code}"
    return f"sz.{code}"


def _normalize_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value[:10]


def _map_adjustment(adjust: str | None) -> str:
    return ADJUSTMENT_MAP.get(adjust, "2")


def _parse_baostock_time(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value[:14], "%Y%m%d%H%M%S"))
