from __future__ import annotations

from datetime import date

import pandas as pd

from .indicators import add_indicators


ATR_COLUMN_LABELS = {
    "symbol": "代码",
    "name": "名称",
    "trade_date": "交易日期",
    "close": "收盘价",
    "atr_14": "ATR14",
    "atr_pct_14": "ATR%",
    "atr_stop_loss_1x": "1ATR止损参考",
    "atr_stop_loss_2x": "2ATR止损参考",
    "atr_take_profit_2x": "2ATR止盈参考",
    "atr_take_profit_3x": "3ATR止盈参考",
    "atr_volatility_regime": "波动分层",
}
ATR_LABEL_TO_COLUMN = {label: column for column, label in ATR_COLUMN_LABELS.items()}
ATR_INTERNAL_COLUMNS = list(ATR_COLUMN_LABELS.keys())
ATR_EXPORT_COLUMNS = list(ATR_COLUMN_LABELS.values())
ATR_WATCHLIST_FIELD_MAP = (
    ("trade_date", "交易日期"),
    ("close", "收盘价"),
    ("atr_14", "ATR14"),
    ("atr_pct_14", "ATR%"),
    ("atr_stop_loss_1x", "1ATR止损参考"),
    ("atr_stop_loss_2x", "2ATR止损参考"),
    ("atr_take_profit_2x", "2ATR止盈参考"),
    ("atr_take_profit_3x", "3ATR止盈参考"),
    ("atr_volatility_regime", "波动分层"),
)


def build_atr_snapshot_row(
    dataframe: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    trade_date: date,
) -> dict[str, object] | None:
    _ = trade_date
    frame = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    required = {
        "trade_date",
        "close",
        "atr_14",
        "atr_pct_14",
        "atr_stop_loss_1x",
        "atr_stop_loss_2x",
        "atr_take_profit_2x",
        "atr_take_profit_3x",
        "atr_volatility_regime",
    }
    if not required.issubset(frame.columns):
        frame = add_indicators(frame).sort_values("trade_date").reset_index(drop=True)
    if frame.empty:
        return None

    latest = frame.iloc[-1]
    atr_value = _safe_float_or_none(latest.get("atr_14"))
    atr_pct = _safe_float_or_none(latest.get("atr_pct_14"))
    if atr_value is None or atr_pct is None:
        return None

    trade_date_value = pd.Timestamp(latest["trade_date"]).date().isoformat()
    return {
        "symbol": symbol,
        "name": name,
        "trade_date": trade_date_value,
        "close": _safe_float_or_none(latest.get("close")),
        "atr_14": atr_value,
        "atr_pct_14": atr_pct,
        "atr_stop_loss_1x": _safe_float_or_none(latest.get("atr_stop_loss_1x")),
        "atr_stop_loss_2x": _safe_float_or_none(latest.get("atr_stop_loss_2x")),
        "atr_take_profit_2x": _safe_float_or_none(latest.get("atr_take_profit_2x")),
        "atr_take_profit_3x": _safe_float_or_none(latest.get("atr_take_profit_3x")),
        "atr_volatility_regime": str(latest.get("atr_volatility_regime") or ""),
    }


def normalize_atr_summary_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    used_export_labels = any(label in frame.columns for label in ATR_LABEL_TO_COLUMN)
    rename_map = {label: column for label, column in ATR_LABEL_TO_COLUMN.items() if label in frame.columns}
    if rename_map:
        frame = frame.rename(columns=rename_map)
    if used_export_labels and "atr_pct_14" in frame.columns:
        frame["atr_pct_14"] = pd.to_numeric(frame["atr_pct_14"], errors="coerce") / 100.0
    return frame


def build_atr_export_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_atr_summary_frame(dataframe)
    export = pd.DataFrame()
    for column in ATR_INTERNAL_COLUMNS:
        export[column] = frame[column] if column in frame.columns else pd.NA

    if "trade_date" in export.columns:
        export["trade_date"] = pd.to_datetime(export["trade_date"], errors="coerce").dt.date.astype("string")
    if "atr_pct_14" in export.columns:
        export["atr_pct_14"] = pd.to_numeric(export["atr_pct_14"], errors="coerce") * 100.0

    for column in (
        "close",
        "atr_14",
        "atr_pct_14",
        "atr_stop_loss_1x",
        "atr_stop_loss_2x",
        "atr_take_profit_2x",
        "atr_take_profit_3x",
    ):
        if column in export.columns:
            export[column] = pd.to_numeric(export[column], errors="coerce").round(4)

    export = export.rename(columns=ATR_COLUMN_LABELS)
    return export.loc[:, ATR_EXPORT_COLUMNS]


def _safe_float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)
