from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


RECOMMENDED_POSITION_PERCENT_FIELD = "建议总仓位%"
POSITION_RISK_FRACTION = 0.02
POSITION_STOP_ATR_MULT = 2.0
POSITION_STAGED_EFFECTIVE_RISK_MULT = 0.85
POSITION_MAX_SYMBOL_FRACTION = 0.40


def recommended_position_percent(
    *,
    atr_pct: object = None,
    atr: object = None,
    price: object = None,
) -> float | None:
    atr_ratio = _normalize_atr_ratio(atr_pct)
    if atr_ratio is None:
        atr_value = _safe_float(atr)
        price_value = _safe_float(price)
        if atr_value is not None and price_value is not None and price_value > 0:
            atr_ratio = atr_value / price_value
    if atr_ratio is None or atr_ratio <= 0:
        return None

    effective_stop_ratio = POSITION_STAGED_EFFECTIVE_RISK_MULT * POSITION_STOP_ATR_MULT * atr_ratio
    position_fraction = POSITION_RISK_FRACTION / effective_stop_ratio
    position_fraction = min(POSITION_MAX_SYMBOL_FRACTION, max(0.0, position_fraction))
    return round(position_fraction * 100.0, 2)


def recommended_position_percent_from_mapping(row: Mapping[str, Any] | pd.Series) -> float | None:
    atr_pct = _first_present(row, ("atr_pct_14", "ATR%", "atr_pct", "ATR_pct"))
    atr = _first_present(row, ("atr_14", "ATR14", "atr"))
    price = _first_present(row, ("close", "atr_close", "收盘价", "price"))
    return recommended_position_percent(atr_pct=atr_pct, atr=atr, price=price)


def add_recommended_position_percent(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[RECOMMENDED_POSITION_PERCENT_FIELD] = [
        recommended_position_percent_from_mapping(row) for _, row in result.iterrows()
    ]
    return result


def _first_present(row: Mapping[str, Any] | pd.Series, columns: tuple[str, ...]) -> object:
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        if not _is_missing(value):
            return value
    return None


def _normalize_atr_ratio(value: object) -> float | None:
    numeric = _safe_float(value)
    if numeric is None or numeric <= 0:
        return None
    if numeric > 1.5:
        return numeric / 100.0
    return numeric


def _safe_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(numeric):
        return None
    return numeric


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
