from __future__ import annotations

from datetime import date

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None

def is_trading_day(provider_name: str, trade_date: date) -> bool:
    _ = provider_name
    resolved = _is_trading_day_akshare(trade_date)
    if resolved is not None:
        return resolved

    return trade_date.weekday() < 5

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
