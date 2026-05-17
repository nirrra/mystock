from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import math

import pandas as pd


INTERNAL_COLUMNS = [
    "trade_date",
    "symbol",
    "name",
    "daily_close",
    "daily_prev_close",
    "daily_return_pct",
    "daily_return_1d",
    "daily_volume",
    "daily_amount",
]

CSV_RENAME_MAP = {
    "trade_date": "交易日期",
    "symbol": "编号",
    "name": "名称",
    "daily_close": "收盘价",
    "daily_prev_close": "前收盘价",
    "daily_return_pct": "当日涨幅%",
    "daily_return_1d": "当日涨幅小数",
    "daily_volume": "成交量",
    "daily_amount": "成交额",
}

CSV_TO_INTERNAL_MAP = {value: key for key, value in CSV_RENAME_MAP.items()}


def full_market_daily_returns_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "daily_screening" / f"full_market_daily_returns_{trade_date.isoformat()}.csv"


def write_full_market_daily_returns(
    *,
    project_root: Path,
    trade_date: date,
    symbols: Iterable[str] | None = None,
) -> Path:
    frame = build_full_market_daily_returns(project_root=project_root, trade_date=trade_date, symbols=symbols)
    target = full_market_daily_returns_path(project_root, trade_date)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = frame.loc[:, INTERNAL_COLUMNS].rename(columns=CSV_RENAME_MAP)
    output.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def read_full_market_daily_returns(*, project_root: Path, trade_date: date) -> pd.DataFrame:
    target = full_market_daily_returns_path(project_root, trade_date)
    if not target.exists():
        return pd.DataFrame(columns=INTERNAL_COLUMNS)
    frame = pd.read_csv(target, dtype={"编号": str, "symbol": str})
    data = frame.rename(columns={key: value for key, value in CSV_TO_INTERNAL_MAP.items() if key in frame.columns}).copy()
    if "symbol" not in data.columns:
        return pd.DataFrame(columns=INTERNAL_COLUMNS)
    data["symbol"] = data["symbol"].map(normalize_symbol)
    for column in INTERNAL_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data.loc[:, INTERNAL_COLUMNS].dropna(subset=["symbol"]).drop_duplicates("symbol", keep="first")


def build_full_market_daily_returns(
    *,
    project_root: Path,
    trade_date: date,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    daily_root = project_root / "data" / "daily"
    selected = {normalize_symbol(symbol) for symbol in symbols} if symbols is not None else None
    if selected is not None:
        selected.discard("")
    name_map = _load_name_map(project_root)

    rows: list[dict[str, object]] = []
    paths = sorted(daily_root.glob("*.parquet")) if daily_root.exists() else []
    for path in paths:
        symbol = normalize_symbol(path.stem)
        if not symbol or (selected is not None and symbol not in selected):
            continue
        item = _read_daily_return_row(path=path, symbol=symbol, trade_date=trade_date)
        if not item:
            continue
        item["name"] = name_map.get(symbol, "")
        rows.append(item)

    if not rows:
        return pd.DataFrame(columns=INTERNAL_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in INTERNAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, INTERNAL_COLUMNS].sort_values("symbol").reset_index(drop=True)


def normalize_symbol(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    if text.startswith("="):
        text = text.lstrip("=").strip().strip('"')
    if text.endswith(".0"):
        text = text[:-2]
    lower = text.lower()
    if lower.startswith(("sh", "sz", "bj")):
        text = text[2:]
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def _read_daily_return_row(*, path: Path, symbol: str, trade_date: date) -> dict[str, object] | None:
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    required = {"trade_date", "close"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.date
    data = data.dropna(subset=["trade_date"]).sort_values("trade_date")
    upto = data[data["trade_date"].le(trade_date)].copy()
    if upto.empty or upto.iloc[-1]["trade_date"] != trade_date:
        return None

    latest = upto.iloc[-1]
    previous = upto.iloc[-2] if len(upto) >= 2 else None
    close = _to_float(latest.get("close"))
    prev_close = _to_float(previous.get("close")) if previous is not None else math.nan
    pct_change = _to_float(latest.get("pct_change")) if "pct_change" in latest.index else math.nan
    if math.isnan(pct_change) and not math.isnan(close) and not math.isnan(prev_close) and prev_close != 0:
        pct_change = (close / prev_close - 1.0) * 100.0

    daily_return_1d = pct_change / 100.0 if not math.isnan(pct_change) else math.nan
    return {
        "trade_date": trade_date.isoformat(),
        "symbol": symbol,
        "daily_close": _clean_float(close),
        "daily_prev_close": _clean_float(prev_close),
        "daily_return_pct": _clean_float(pct_change),
        "daily_return_1d": _clean_float(daily_return_1d),
        "daily_volume": _clean_float(_to_float(latest.get("volume"))),
        "daily_amount": _clean_float(_to_float(latest.get("amount"))),
    }


def _load_name_map(project_root: Path) -> dict[str, str]:
    paths = [
        project_root / "data" / "universe" / "main_board.parquet",
        project_root / "data" / "sector_membership" / "stock_sector_membership.csv",
    ]
    names: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, dtype={"symbol": str})
        except Exception:
            continue
        if "symbol" not in frame.columns or "name" not in frame.columns:
            continue
        for _, row in frame.loc[:, ["symbol", "name"]].dropna(subset=["symbol"]).iterrows():
            symbol = normalize_symbol(row.get("symbol"))
            name = str(row.get("name") or "").strip()
            if symbol and name and symbol not in names:
                names[symbol] = name
    return names


def _to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _clean_float(value: float) -> float | str:
    if math.isnan(value) or math.isinf(value):
        return ""
    return round(float(value), 6)
