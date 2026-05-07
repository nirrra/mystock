from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import DailyBarsReadError, Storage


MIN_VALID_DAILY_RETURN = -0.5
MAX_VALID_DAILY_RETURN = 1.0
SYNTHETIC_MARKET_COLUMNS = [
    "trade_date",
    "stock_count",
    "equal_weight_return",
    "median_return",
    "amount_weight_return",
    "breadth_up_ratio",
    "above_ma20_ratio",
    "above_ma60_ratio",
    "limit_up_count",
    "limit_down_count",
    "synthetic_equal_weight_index",
    "synthetic_amount_weight_index",
]


@dataclass(slots=True)
class SyntheticMarketResult:
    frame: pd.DataFrame
    output_path: Path
    skipped: pd.DataFrame


def synthetic_market_path(project_root: Path) -> Path:
    return project_root / "reports" / "full_market_model" / "synthetic_market.csv"


def build_synthetic_market_index(
    *,
    storage: Storage,
    project_root: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    min_stock_count: int = 500,
    output: Path | None = None,
) -> SyntheticMarketResult:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()

    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    for instrument in universe.to_dict("records"):
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        frame = _symbol_market_frame(bars, symbol=symbol)
        if start_date:
            frame = frame[frame["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            frame = frame[frame["trade_date"] <= pd.Timestamp(end_date)]
        if frame.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_market_rows"})
            continue
        rows.append(frame)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    market = aggregate_synthetic_market(panel, min_stock_count=min_stock_count)
    output_path = output if output is not None else synthetic_market_path(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(output_path, index=False, encoding="utf-8-sig")
    return SyntheticMarketResult(frame=market, output_path=output_path, skipped=pd.DataFrame(skipped))


def aggregate_synthetic_market(panel: pd.DataFrame, *, min_stock_count: int = 1) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=SYNTHETIC_MARKET_COLUMNS)
    grouped = panel.groupby("trade_date", sort=True)
    rows = []
    for trade_date, day in grouped:
        amount = pd.to_numeric(day["amount"], errors="coerce").fillna(0.0)
        daily_return = pd.to_numeric(day["daily_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = daily_return.between(MIN_VALID_DAILY_RETURN, MAX_VALID_DAILY_RETURN)
        day = day.loc[valid].copy()
        daily_return = daily_return.loc[valid]
        amount = amount.loc[valid]
        if day.empty:
            continue
        stock_count = int(day["symbol"].nunique())
        if stock_count < min_stock_count:
            continue
        amount_weight_return = _weighted_return(daily_return=daily_return, amount=amount)
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "stock_count": stock_count,
                "equal_weight_return": float(daily_return.mean()),
                "median_return": float(daily_return.median()),
                "amount_weight_return": amount_weight_return,
                "breadth_up_ratio": float(daily_return.gt(0).mean()),
                "above_ma20_ratio": float(day["above_ma20"].mean()),
                "above_ma60_ratio": float(day["above_ma60"].mean()),
                "limit_up_count": int(day["limit_up_like"].sum()),
                "limit_down_count": int(day["limit_down_like"].sum()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=SYNTHETIC_MARKET_COLUMNS)
    result = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    result["synthetic_equal_weight_index"] = (1.0 + result["equal_weight_return"].fillna(0.0)).cumprod()
    result["synthetic_amount_weight_index"] = (1.0 + result["amount_weight_return"].fillna(0.0)).cumprod()
    return result


def _weighted_return(*, daily_return: pd.Series, amount: pd.Series) -> float:
    amount_sum = float(amount.sum())
    if amount_sum > 0:
        return float((daily_return * amount).sum() / amount_sum)
    return float(daily_return.mean())


def _symbol_market_frame(bars: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    for column in ("close", "high", "low", "amount"):
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        frame[column] = pd.to_numeric(values, errors="coerce")
    close = frame["close"]
    prev_close = close.shift(1)
    daily_return = close.div(prev_close).sub(1.0).replace([np.inf, -np.inf], np.nan)
    valid_return = (
        close.gt(0)
        & prev_close.gt(0)
        & daily_return.between(MIN_VALID_DAILY_RETURN, MAX_VALID_DAILY_RETURN)
    )
    daily_return = daily_return.where(valid_return)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    locked = frame["high"].sub(frame["low"]).abs().le((close.abs() * 0.0005).fillna(0.0))
    return pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "symbol": str(symbol).zfill(6),
            "daily_return": daily_return,
            "amount": frame["amount"].shift(1),
            "above_ma20": close.gt(ma20),
            "above_ma60": close.gt(ma60),
            "limit_up_like": locked & daily_return.ge(0.095),
            "limit_down_like": locked & daily_return.le(-0.095),
        }
    ).dropna(subset=["daily_return"])
