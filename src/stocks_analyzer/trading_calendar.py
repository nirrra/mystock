from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from datetime import date

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None

try:
    import baostock as bs
except ImportError:  # pragma: no cover
    bs = None


def is_trading_day(provider_name: str, trade_date: date) -> bool:
    provider = str(provider_name or "").strip().lower()
    if provider == "akshare":
        resolved = _is_trading_day_akshare(trade_date)
        if resolved is not None:
            return resolved

    resolved = _is_trading_day_baostock(trade_date)
    if resolved is not None:
        return resolved

    return trade_date.weekday() < 5


def _is_trading_day_baostock(trade_date: date) -> bool | None:
    if bs is None:
        return None

    silent = io.StringIO()
    with redirect_stdout(silent), redirect_stderr(silent):
        login_result = bs.login()
    if login_result.error_code != "0":
        return None

    try:
        with redirect_stdout(silent), redirect_stderr(silent):
            result = bs.query_trade_dates(
                start_date=trade_date.isoformat(),
                end_date=trade_date.isoformat(),
            )
        if result.error_code != "0":
            return None
        while result.error_code == "0" and result.next():
            row = dict(zip(result.fields, result.get_row_data()))
            flag = str(row.get("is_trading_day", "")).strip()
            if flag:
                return flag == "1"
    finally:
        with redirect_stdout(silent), redirect_stderr(silent):
            bs.logout()

    return None


def _is_trading_day_akshare(trade_date: date) -> bool | None:
    if ak is None:
        return None

    try:
        frame = ak.tool_trade_date_hist_sina()
    except Exception:
        return None

    if "trade_date" not in frame.columns:
        return None
    dates = set(frame["trade_date"].astype(str))
    return trade_date.isoformat() in dates
