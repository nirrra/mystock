from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import pandas as pd

from .storage import DailyBarsReadError, Storage


FULL_MARKET_REPORT_DIRNAME = "full_market_model"
REQUIRED_DAILY_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(slots=True)
class FullMarketDataAuditResult:
    detail: pd.DataFrame
    summary: dict[str, Any]
    report_dir: Path
    detail_path: Path
    summary_path: Path


def full_market_report_dir(project_root: Path) -> Path:
    return project_root / "reports" / FULL_MARKET_REPORT_DIRNAME


def audit_full_market_data(
    *,
    storage: Storage,
    project_root: Path,
    limit: int | None = None,
    min_exact_history_days: int = 900,
    tail_lookback_days: int = 100,
    max_horizon_days: int = 20,
    output_dir: Path | None = None,
) -> FullMarketDataAuditResult:
    symbols = _load_audit_symbols(storage)
    if limit is not None:
        symbols = symbols[: max(int(limit), 0)]

    rows: list[dict[str, Any]] = []
    total_symbols = len(symbols)
    for index, symbol_info in enumerate(symbols, start=1):
        if index == 1 or index % 500 == 0 or index == total_symbols:
            print(f"INFO full-market-audit progress: {index}/{total_symbols}")
        rows.append(
            _audit_symbol(
                storage,
                symbol_info,
                min_exact_history_days=min_exact_history_days,
                tail_lookback_days=tail_lookback_days,
                max_horizon_days=max_horizon_days,
            )
        )

    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail = detail.sort_values(["readable", "trading_days", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    summary = _build_audit_summary(
        detail,
        min_exact_history_days=min_exact_history_days,
        tail_lookback_days=tail_lookback_days,
        max_horizon_days=max_horizon_days,
    )

    report_dir = output_dir if output_dir is not None else full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    detail_path = report_dir / "data_audit.csv"
    summary_path = report_dir / "data_audit_summary.json"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return FullMarketDataAuditResult(
        detail=detail,
        summary=summary,
        report_dir=report_dir,
        detail_path=detail_path,
        summary_path=summary_path,
    )


def format_full_market_audit_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Full-market data audit complete.",
        f"Symbols: {summary.get('symbols_total', 0)}",
        f"Readable: {summary.get('symbols_readable', 0)}",
        f"Unreadable/missing: {summary.get('symbols_unreadable', 0)}",
        f"Global date range: {summary.get('global_first_trade_date', '')} -> {summary.get('global_last_trade_date', '')}",
        f"Trading days median: {summary.get('trading_days_median', 0)}",
        f"Trading days min/max: {summary.get('trading_days_min', 0)} / {summary.get('trading_days_max', 0)}",
        f"Eligible for tail-risk labels: {summary.get('eligible_tail_risk_symbols', 0)}",
        f"Eligible for barrier labels: {summary.get('eligible_barrier_symbols', 0)}",
        f"Eligible for strict 900-day reproduction: {summary.get('eligible_exact_history_symbols', 0)}",
    ]
    if not summary.get("strict_reproduction_ready", False):
        lines.append(f"Strict reproduction status: blocked - {summary.get('strict_reproduction_blocker', '')}")
    else:
        lines.append("Strict reproduction status: ready")
    return "\n".join(lines)


def _load_audit_symbols(storage: Storage) -> list[dict[str, str]]:
    universe_symbols: list[dict[str, str]] = []
    try:
        universe = storage.load_universe()
    except FileNotFoundError:
        universe = pd.DataFrame()
    if not universe.empty and "symbol" in universe.columns:
        frame = universe.copy()
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        for row in frame.sort_values("symbol").to_dict("records"):
            universe_symbols.append({"symbol": str(row.get("symbol", "")).zfill(6), "name": str(row.get("name", ""))})
    if universe_symbols:
        return universe_symbols

    symbols = []
    for path in sorted(storage.paths.daily_dir.glob("*.parquet")):
        symbols.append({"symbol": path.stem.zfill(6) if path.stem.isdigit() else path.stem, "name": ""})
    return symbols


def _audit_symbol(
    storage: Storage,
    symbol_info: dict[str, str],
    *,
    min_exact_history_days: int,
    tail_lookback_days: int,
    max_horizon_days: int,
) -> dict[str, Any]:
    symbol = str(symbol_info.get("symbol", "")).zfill(6)
    base: dict[str, Any] = {
        "symbol": symbol,
        "name": symbol_info.get("name", ""),
        "readable": False,
        "error": "",
        "first_trade_date": "",
        "last_trade_date": "",
        "trading_days": 0,
        "duplicate_trade_dates": 0,
        "zero_volume_days": 0,
        "limit_up_detectable_count": 0,
        "limit_down_detectable_count": 0,
        "eligible_tail_risk": False,
        "eligible_barrier": False,
        "eligible_exact_history": False,
    }
    for column in REQUIRED_DAILY_COLUMNS:
        base[f"missing_{column}_count"] = 0
        base[f"has_{column}"] = False

    try:
        bars = storage.load_daily_bars(symbol)
    except (FileNotFoundError, DailyBarsReadError) as exc:
        base["error"] = str(exc)
        return base

    if bars.empty:
        base["error"] = "empty_daily_bars"
        return base

    frame = bars.copy()
    if "trade_date" not in frame.columns:
        base["error"] = "missing_trade_date_column"
        return base
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame[frame["trade_date"].notna()].sort_values("trade_date")
    if frame.empty:
        base["error"] = "no_valid_trade_date"
        return base

    trading_days = int(frame["trade_date"].nunique())
    base.update(
        {
            "readable": True,
            "first_trade_date": frame["trade_date"].min().date().isoformat(),
            "last_trade_date": frame["trade_date"].max().date().isoformat(),
            "trading_days": trading_days,
            "duplicate_trade_dates": int(len(frame) - trading_days),
            "eligible_tail_risk": trading_days >= tail_lookback_days + 2,
            "eligible_barrier": trading_days >= tail_lookback_days + max_horizon_days + 2,
            "eligible_exact_history": trading_days >= min_exact_history_days,
        }
    )

    for column in REQUIRED_DAILY_COLUMNS:
        base[f"has_{column}"] = column in frame.columns
        base[f"missing_{column}_count"] = int(frame[column].isna().sum()) if column in frame.columns else trading_days

    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(dtype=float)
    base["zero_volume_days"] = int(volume.fillna(0).eq(0).sum()) if not volume.empty else trading_days
    base["limit_up_detectable_count"], base["limit_down_detectable_count"] = _count_locked_limits(frame)
    return base


def _count_locked_limits(frame: pd.DataFrame) -> tuple[int, int]:
    required = {"high", "low", "close"}
    if not required.issubset(frame.columns):
        return 0, 0
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    prev_close = close.shift(1)
    daily_ret = close / prev_close - 1.0
    locked = high.sub(low).abs().le((close.abs() * 0.0005).fillna(0.0))
    limit_up = locked & daily_ret.ge(0.095)
    limit_down = locked & daily_ret.le(-0.095)
    return int(limit_up.sum()), int(limit_down.sum())


def _build_audit_summary(
    detail: pd.DataFrame,
    *,
    min_exact_history_days: int,
    tail_lookback_days: int,
    max_horizon_days: int,
) -> dict[str, Any]:
    if detail.empty:
        return {
            "symbols_total": 0,
            "symbols_readable": 0,
            "symbols_unreadable": 0,
            "strict_reproduction_ready": False,
            "strict_reproduction_blocker": "no_symbols_found",
            "min_exact_history_days": int(min_exact_history_days),
            "tail_lookback_days": int(tail_lookback_days),
            "max_horizon_days": int(max_horizon_days),
        }
    readable = detail[detail["readable"].astype(bool)].copy()
    trading_days = pd.to_numeric(readable.get("trading_days", pd.Series(dtype=float)), errors="coerce")
    first_dates = pd.to_datetime(readable.get("first_trade_date", pd.Series(dtype=str)), errors="coerce")
    last_dates = pd.to_datetime(readable.get("last_trade_date", pd.Series(dtype=str)), errors="coerce")
    eligible_exact = int(readable.get("eligible_exact_history", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    strict_ready = eligible_exact > 0
    blocker = "" if strict_ready else f"no readable symbol has at least {min_exact_history_days} trading days"
    return {
        "symbols_total": int(len(detail)),
        "symbols_readable": int(len(readable)),
        "symbols_unreadable": int(len(detail) - len(readable)),
        "total_daily_rows": int(pd.to_numeric(readable.get("trading_days", pd.Series(dtype=float)), errors="coerce").sum()),
        "global_first_trade_date": _date_or_empty(first_dates.min()),
        "global_last_trade_date": _date_or_empty(last_dates.max()),
        "trading_days_min": _finite_int(trading_days.min()),
        "trading_days_median": _finite_int(trading_days.median()),
        "trading_days_max": _finite_int(trading_days.max()),
        "eligible_tail_risk_symbols": int(readable.get("eligible_tail_risk", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "eligible_barrier_symbols": int(readable.get("eligible_barrier", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "eligible_exact_history_symbols": eligible_exact,
        "strict_reproduction_ready": strict_ready,
        "strict_reproduction_blocker": blocker,
        "min_exact_history_days": int(min_exact_history_days),
        "tail_lookback_days": int(tail_lookback_days),
        "max_horizon_days": int(max_horizon_days),
        "symbols_with_missing_ohlcv": int(_symbols_with_missing_required_columns(readable)),
        "duplicate_trade_dates_total": int(pd.to_numeric(readable.get("duplicate_trade_dates", pd.Series(dtype=float)), errors="coerce").sum()),
        "zero_volume_days_total": int(pd.to_numeric(readable.get("zero_volume_days", pd.Series(dtype=float)), errors="coerce").sum()),
        "limit_up_detectable_total": int(pd.to_numeric(readable.get("limit_up_detectable_count", pd.Series(dtype=float)), errors="coerce").sum()),
        "limit_down_detectable_total": int(pd.to_numeric(readable.get("limit_down_detectable_count", pd.Series(dtype=float)), errors="coerce").sum()),
    }


def _symbols_with_missing_required_columns(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    masks = []
    for column in REQUIRED_DAILY_COLUMNS:
        missing_column = f"missing_{column}_count"
        has_column = f"has_{column}"
        masks.append(~frame.get(has_column, pd.Series(False, index=frame.index)).fillna(False).astype(bool))
        masks.append(pd.to_numeric(frame.get(missing_column, pd.Series(0, index=frame.index)), errors="coerce").fillna(0).gt(0))
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask
    return int(combined.sum())


def _date_or_empty(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _finite_int(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(numeric):
        return 0
    return int(round(numeric))
