from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .exit_rule_optimization import _load_selected_forward_paths, _normalize_paths
from .indicators import add_indicators
from .storage import DailyBarsReadError, Storage


DEFAULT_STAGED_STRATEGIES = ("centered_risk_top20",)
DEFAULT_BATCH_WEIGHTS = (0.30, 0.30, 0.20, 0.20)


@dataclass(slots=True)
class StagedPositionBacktestResult:
    trades: pd.DataFrame
    orders: pd.DataFrame
    summary: pd.DataFrame
    by_year: pd.DataFrame
    by_split: pd.DataFrame
    output_dir: Path
    trades_path: Path
    orders_path: Path
    summary_path: Path
    by_year_path: Path
    by_split_path: Path


def backtest_staged_position_strategy(
    *,
    storage: Storage,
    strict_dir: Path,
    strategies: tuple[str, ...] = DEFAULT_STAGED_STRATEGIES,
    horizons: tuple[int, ...] = (60,),
    output_dir: Path | None = None,
    account_risk_pct: float = 0.02,
    max_position_pct: float = 0.40,
    batch_weights: tuple[float, float, float, float] = DEFAULT_BATCH_WEIGHTS,
    atr_mult: float = 2.0,
    trailing_atr_mult: float = 2.5,
    tune_end_date: date = date(2023, 12, 29),
    test_start_date: date = date(2024, 1, 1),
    progress: bool = False,
) -> StagedPositionBacktestResult:
    strict_dir = strict_dir.resolve()
    output_root = (output_dir or strict_dir / "staged_position_backtest").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    paths = _normalize_paths(_load_selected_forward_paths(strict_dir))
    if strategies:
        wanted = {str(item) for item in strategies}
        paths = paths[paths["strategy"].astype(str).isin(wanted)].copy()
    if paths.empty:
        raise RuntimeError("No selected forward paths remain after strategy filtering.")

    max_horizon = max(int(item) for item in horizons)
    paths = paths[pd.to_numeric(paths["day_offset"], errors="coerce").le(max_horizon)].copy()
    group_columns = ["strategy", "signal_date", "symbol", "selected_rank"]
    groups = list(paths.sort_values(group_columns + ["day_offset"]).groupby(group_columns, sort=False, dropna=False))
    indicator_cache: dict[str, pd.DataFrame] = {}

    trade_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    total_jobs = len(groups) * len(horizons)
    job_index = 0
    for group_key, group in groups:
        strategy, signal_date_value, symbol, selected_rank = group_key
        symbol = str(symbol).zfill(6)
        indicator_frame = _indicator_frame_for_symbol(storage, indicator_cache, symbol)
        signal_atr_pct = _signal_atr_pct(indicator_frame, str(signal_date_value))
        enriched_path = _attach_path_indicators(group, indicator_frame)
        for horizon in horizons:
            job_index += 1
            if progress and (job_index == 1 or job_index == total_jobs or job_index % 5000 == 0):
                elapsed = perf_counter() - started
                eta = elapsed / max(job_index, 1) * max(total_jobs - job_index, 0)
                print(
                    f"Staged-position progress: {job_index}/{total_jobs} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                    flush=True,
                )
            outcome, orders = simulate_staged_position_path(
                enriched_path,
                signal_atr_pct=signal_atr_pct,
                horizon=int(horizon),
                account_risk_pct=account_risk_pct,
                max_position_pct=max_position_pct,
                batch_weights=batch_weights,
                atr_mult=atr_mult,
                trailing_atr_mult=trailing_atr_mult,
            )
            if outcome is None:
                continue
            signal_date_text = str(signal_date_value)
            signal_dt = pd.to_datetime(signal_date_text, errors="coerce")
            year = int(signal_dt.year) if pd.notna(signal_dt) else -1
            split = _split_name(signal_dt.date(), tune_end_date=tune_end_date, test_start_date=test_start_date) if pd.notna(signal_dt) else "unknown"
            base = {
                "strategy": str(strategy),
                "signal_date": signal_date_text,
                "symbol": symbol,
                "selected_rank": int(selected_rank),
                "horizon": int(horizon),
                "year": year,
                "split": split,
            }
            trade_rows.append({**base, **outcome})
            for order in orders:
                order_rows.append({**base, **order})

    trades = pd.DataFrame(trade_rows)
    orders = pd.DataFrame(order_rows)
    summary = _summarize_trades(trades, group_columns=("strategy", "horizon"))
    by_year = _summarize_trades(trades, group_columns=("strategy", "horizon", "year"))
    by_split = _summarize_trades(trades, group_columns=("strategy", "horizon", "split"))

    trades_path = output_root / "staged_position_trades.csv"
    orders_path = output_root / "staged_position_orders.csv"
    summary_path = output_root / "staged_position_summary.csv"
    by_year_path = output_root / "staged_position_by_year.csv"
    by_split_path = output_root / "staged_position_by_split.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    orders.to_csv(orders_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")
    by_split.to_csv(by_split_path, index=False, encoding="utf-8-sig")

    return StagedPositionBacktestResult(
        trades=trades,
        orders=orders,
        summary=summary,
        by_year=by_year,
        by_split=by_split,
        output_dir=output_root,
        trades_path=trades_path,
        orders_path=orders_path,
        summary_path=summary_path,
        by_year_path=by_year_path,
        by_split_path=by_split_path,
    )


def simulate_staged_position_path(
    path: pd.DataFrame,
    *,
    signal_atr_pct: float | None,
    horizon: int,
    account_risk_pct: float = 0.02,
    max_position_pct: float = 0.40,
    batch_weights: tuple[float, float, float, float] = DEFAULT_BATCH_WEIGHTS,
    atr_mult: float = 2.0,
    trailing_atr_mult: float = 2.5,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    frame = path.sort_values("day_offset").head(int(horizon)).copy()
    if len(frame) < int(horizon) or frame.empty:
        return None, []
    entry = _safe_float(frame.iloc[0].get("entry_open"))
    if entry is None:
        entry = _safe_float(frame.iloc[0].get("open"))
    if entry is None or entry <= 0:
        return None, []
    if signal_atr_pct is None or not math.isfinite(float(signal_atr_pct)) or signal_atr_pct <= 0:
        return None, []
    risk_denominator = float(max_position_pct) * (float(batch_weights[0]) + float(batch_weights[1]) / 2.0)
    if risk_denominator <= 0:
        return None, []
    r_cap_pct = float(account_risk_pct) / risk_denominator
    r_pct = min(float(atr_mult) * float(signal_atr_pct), r_cap_pct)
    if r_pct <= 0 or not math.isfinite(r_pct):
        return None, []

    stop_initial = entry * (1.0 - r_pct)
    planned_qty = float(max_position_pct) / entry
    batch_qty = [planned_qty * float(weight) for weight in batch_weights]
    add_prices = [
        entry,
        entry * (1.0 - r_pct / 2.0),
        entry * (1.0 + r_pct),
        entry * (1.0 + 2.0 * r_pct),
    ]
    take1_price = entry * (1.0 + 1.5 * r_pct)
    take2_price = entry * (1.0 + 2.5 * r_pct)
    stop_plus_half_r = entry * (1.0 + 0.5 * r_pct)

    cash_flow = 0.0
    position_qty = 0.0
    cost_basis = 0.0
    active_stop = stop_initial
    highest_close = entry
    highest_high = entry
    batches_filled = [False, False, False, False]
    take1_done = False
    take2_done = False
    trailing_active = False
    exit_reason = "timeout"
    exit_date = str(frame.iloc[-1].get("trade_date"))
    holding_days = int(horizon)
    orders: list[dict[str, Any]] = []
    max_equity_profit = 0.0
    max_equity_drawdown = 0.0
    max_position_profit = 0.0
    max_position_drawdown = 0.0

    def buy(batch_number: int, trade_date: str, price: float, reason: str) -> None:
        nonlocal cash_flow, position_qty, cost_basis
        qty = batch_qty[batch_number - 1]
        if qty <= 0:
            return
        cash_flow -= qty * price
        cost_basis += qty * price
        position_qty += qty
        batches_filled[batch_number - 1] = True
        orders.append(
            {
                "trade_date": trade_date,
                "action": "buy",
                "batch": batch_number,
                "price": price,
                "qty": qty,
                "position_qty_after": position_qty,
                "reason": reason,
            }
        )

    def sell(qty: float, trade_date: str, price: float, reason: str) -> None:
        nonlocal cash_flow, position_qty, cost_basis
        qty = min(qty, position_qty)
        if qty <= 0:
            return
        avg_cost = cost_basis / position_qty if position_qty > 0 else 0.0
        cash_flow += qty * price
        cost_basis -= avg_cost * qty
        position_qty -= qty
        if position_qty <= 1e-12:
            position_qty = 0.0
            cost_basis = 0.0
        orders.append(
            {
                "trade_date": trade_date,
                "action": "sell",
                "batch": None,
                "price": price,
                "qty": qty,
                "position_qty_after": position_qty,
                "reason": reason,
            }
        )

    def avg_cost() -> float:
        return cost_basis / position_qty if position_qty > 0 else math.nan

    def mark_profit(price: float) -> float:
        return cash_flow + position_qty * price

    first = frame.iloc[0]
    first_date = str(first.get("trade_date"))
    first_open = _safe_float(first.get("open")) or entry
    buy(1, first_date, first_open, "initial_entry")

    for offset, row in enumerate(frame.itertuples(index=False), start=1):
        row_date = str(getattr(row, "trade_date"))
        row_open = float(getattr(row, "open"))
        row_high = float(getattr(row, "high"))
        row_low = float(getattr(row, "low"))
        row_close = float(getattr(row, "close"))
        row_atr = _safe_float(getattr(row, "atr_14", math.nan))
        row_ma20 = _safe_float(getattr(row, "ma_20", math.nan))

        if offset > 1 and position_qty > 0 and row_open <= active_stop:
            sell(position_qty, row_date, row_open, "stop_open_gap")
            exit_reason = "stop_open_gap"
            exit_date = row_date
            holding_days = offset
            break

        if position_qty > 0 and not batches_filled[1]:
            second_price = add_prices[1]
            if row_open <= active_stop:
                sell(position_qty, row_date, row_open, "stop_open_gap")
                exit_reason = "stop_open_gap"
                exit_date = row_date
                holding_days = offset
                break
            if row_open <= second_price and row_open > active_stop:
                buy(2, row_date, row_open, "pullback_R_half_open")
            elif row_low <= second_price:
                buy(2, row_date, second_price, "pullback_R_half")

        if position_qty > 0 and row_low <= active_stop:
            sell(position_qty, row_date, active_stop, "stop_loss")
            exit_reason = "stop_loss"
            exit_date = row_date
            holding_days = offset
            break

        if position_qty > 0 and row_high >= add_prices[2]:
            if not batches_filled[2]:
                buy(3, row_date, add_prices[2], "profit_add_1R")
            active_stop = max(active_stop, avg_cost())

        if position_qty > 0 and row_low <= active_stop:
            sell(position_qty, row_date, active_stop, "moved_stop")
            exit_reason = "moved_stop"
            exit_date = row_date
            holding_days = offset
            break

        if position_qty > 0 and row_high >= take1_price and not take1_done:
            sell(planned_qty * 0.30, row_date, take1_price, "take_profit_1_5R")
            take1_done = True

        if position_qty > 0 and row_high >= add_prices[3]:
            if not batches_filled[3]:
                buy(4, row_date, add_prices[3], "profit_add_2R")
            active_stop = max(active_stop, stop_plus_half_r)

        if position_qty > 0 and row_low <= active_stop:
            sell(position_qty, row_date, active_stop, "moved_stop")
            exit_reason = "moved_stop"
            exit_date = row_date
            holding_days = offset
            break

        if position_qty > 0 and row_high >= take2_price and not take2_done:
            sell(planned_qty * 0.30, row_date, take2_price, "take_profit_2_5R")
            take2_done = True
            trailing_active = True

        highest_high = max(highest_high, row_high)
        highest_close = max(highest_close, row_close)
        if trailing_active and position_qty > 0:
            trailing_candidates = []
            if row_atr is not None and math.isfinite(row_atr):
                trailing_candidates.append(highest_close - float(trailing_atr_mult) * row_atr)
            if row_ma20 is not None and math.isfinite(row_ma20):
                trailing_candidates.append(row_ma20)
            if trailing_candidates:
                active_stop = max(active_stop, max(trailing_candidates))
            if row_low <= active_stop:
                sell(position_qty, row_date, active_stop, "trailing_stop")
                exit_reason = "trailing_stop"
                exit_date = row_date
                holding_days = offset
                break

        high_profit = mark_profit(row_high)
        low_profit = mark_profit(row_low)
        max_equity_profit = max(max_equity_profit, high_profit)
        max_equity_drawdown = min(max_equity_drawdown, low_profit)
        max_position_profit = max(max_position_profit, high_profit / max_position_pct)
        max_position_drawdown = min(max_position_drawdown, low_profit / max_position_pct)

    if position_qty > 0:
        last = frame.iloc[min(holding_days, len(frame)) - 1]
        final_close = _safe_float(last.get("close"))
        if final_close is None:
            return None, []
        sell(position_qty, str(last.get("trade_date")), final_close, "timeout")

    realized_account_return = cash_flow
    realized_position_return = cash_flow / max_position_pct if max_position_pct > 0 else math.nan
    invested_capital = sum(float(order["price"]) * float(order["qty"]) for order in orders if order["action"] == "buy")
    invested_return = cash_flow / invested_capital if invested_capital > 0 else math.nan
    filled_count = sum(1 for item in batches_filled if item)
    outcome = {
        "entry_open": entry,
        "signal_atr_pct": signal_atr_pct,
        "r_pct": r_pct,
        "r_cap_pct": r_cap_pct,
        "initial_stop_price": stop_initial,
        "second_add_price": add_prices[1],
        "third_add_price": add_prices[2],
        "fourth_add_price": add_prices[3],
        "final_cash_flow": cash_flow,
        "account_return": realized_account_return,
        "position_return": realized_position_return,
        "invested_return": invested_return,
        "return_R": realized_account_return / float(account_risk_pct) if account_risk_pct > 0 else math.nan,
        "win": realized_account_return > 0,
        "max_account_profit": max_equity_profit,
        "max_account_drawdown": max_equity_drawdown,
        "max_position_profit": max_position_profit,
        "max_position_drawdown": max_position_drawdown,
        "holding_days": holding_days,
        "exit_date": exit_date,
        "exit_reason": exit_reason,
        "batches_filled": filled_count,
        "second_batch_filled": batches_filled[1],
        "third_batch_filled": batches_filled[2],
        "fourth_batch_filled": batches_filled[3],
        "take1_done": take1_done,
        "take2_done": take2_done,
        "planned_qty": planned_qty,
        "invested_capital": invested_capital,
    }
    return outcome, orders


def _indicator_frame_for_symbol(storage: Storage, cache: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
    if symbol in cache:
        return cache[symbol]
    try:
        frame = add_indicators(storage.load_daily_bars(symbol))
    except (FileNotFoundError, DailyBarsReadError):
        frame = pd.DataFrame()
    if not frame.empty:
        frame = frame.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date.astype(str)
    cache[symbol] = frame
    return frame


def _signal_atr_pct(indicator_frame: pd.DataFrame, signal_date: str) -> float | None:
    if indicator_frame.empty or "atr_pct_14" not in indicator_frame.columns:
        return None
    frame = indicator_frame[indicator_frame["trade_date"].le(signal_date)]
    if frame.empty:
        return None
    return _safe_float(frame.iloc[-1].get("atr_pct_14"))


def _attach_path_indicators(path: pd.DataFrame, indicator_frame: pd.DataFrame) -> pd.DataFrame:
    result = path.copy()
    if "trade_date" in result.columns:
        result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.date.astype(str)
    if indicator_frame.empty:
        result["atr_14"] = np.nan
        result["ma_20"] = np.nan
        return result
    keep = [column for column in ("trade_date", "atr_14", "ma_20") if column in indicator_frame.columns]
    if len(keep) <= 1:
        result["atr_14"] = np.nan
        result["ma_20"] = np.nan
        return result
    return result.merge(indicator_frame.loc[:, keep], on="trade_date", how="left")


def _summarize_trades(trades: pd.DataFrame, *, group_columns: tuple[str, ...]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in trades.groupby(list(group_columns), sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key, strict=False)}
        returns = pd.to_numeric(group["position_return"], errors="coerce")
        account_returns = pd.to_numeric(group["account_return"], errors="coerce")
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        avg_gain = float(gains.mean()) if len(gains) else math.nan
        avg_loss = float(losses.mean()) if len(losses) else math.nan
        row.update(
            {
                "trade_count": int(len(group)),
                "avg_position_return": float(returns.mean()),
                "median_position_return": float(returns.median()),
                "avg_account_return": float(account_returns.mean()),
                "median_account_return": float(account_returns.median()),
                "avg_daily_position_return": float((returns / pd.to_numeric(group["holding_days"], errors="coerce").clip(lower=1)).mean()),
                "win_rate": float(pd.to_numeric(group["win"], errors="coerce").mean()),
                "avg_R": float(pd.to_numeric(group["return_R"], errors="coerce").mean()),
                "avg_gain": avg_gain,
                "avg_loss": avg_loss,
                "payoff_ratio": avg_gain / abs(avg_loss) if math.isfinite(avg_gain) and math.isfinite(avg_loss) and avg_loss != 0 else math.nan,
                "avg_max_position_profit": float(pd.to_numeric(group["max_position_profit"], errors="coerce").mean()),
                "median_max_position_profit": float(pd.to_numeric(group["max_position_profit"], errors="coerce").median()),
                "avg_max_position_drawdown": float(pd.to_numeric(group["max_position_drawdown"], errors="coerce").mean()),
                "median_max_position_drawdown": float(pd.to_numeric(group["max_position_drawdown"], errors="coerce").median()),
                "avg_holding_days": float(pd.to_numeric(group["holding_days"], errors="coerce").mean()),
                "median_holding_days": float(pd.to_numeric(group["holding_days"], errors="coerce").median()),
                "avg_batches_filled": float(pd.to_numeric(group["batches_filled"], errors="coerce").mean()),
                "second_batch_rate": float(pd.to_numeric(group["second_batch_filled"], errors="coerce").mean()),
                "third_batch_rate": float(pd.to_numeric(group["third_batch_filled"], errors="coerce").mean()),
                "fourth_batch_rate": float(pd.to_numeric(group["fourth_batch_filled"], errors="coerce").mean()),
                "take1_rate": float(pd.to_numeric(group["take1_done"], errors="coerce").mean()),
                "take2_rate": float(pd.to_numeric(group["take2_done"], errors="coerce").mean()),
                "moved_stop_rate": float(group["exit_reason"].astype(str).isin({"moved_stop", "trailing_stop"}).mean()),
                "stop_loss_rate": float(group["exit_reason"].astype(str).isin({"stop_loss", "stop_open_gap"}).mean()),
                "timeout_rate": float(group["exit_reason"].astype(str).eq("timeout").mean()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(list(group_columns)).reset_index(drop=True)


def _split_name(signal_date: date, *, tune_end_date: date, test_start_date: date) -> str:
    if signal_date <= tune_end_date:
        return "tune"
    if signal_date >= test_start_date:
        return "test"
    return "gap"


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"
