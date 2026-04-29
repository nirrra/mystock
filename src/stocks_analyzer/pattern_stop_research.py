from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_HOLDING_DAYS = (5, 10, 20, 40)
DEFAULT_TAKE_PROFITS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
DEFAULT_STOP_LOSSES = (0.03, 0.05, 0.07, 0.10, 0.12)


def research_pattern_stop_grid(
    forward_prices: pd.DataFrame,
    *,
    holding_days: Sequence[int] = DEFAULT_HOLDING_DAYS,
    take_profits: Sequence[float] = DEFAULT_TAKE_PROFITS,
    stop_losses: Sequence[float] = DEFAULT_STOP_LOSSES,
    same_day_policy: str = "stop_first",
    ma20_stop: bool = False,
    ma20_stop_tolerance: float = 0.0,
) -> dict[str, pd.DataFrame]:
    if forward_prices.empty:
        return {"trades": pd.DataFrame(), "summary": pd.DataFrame(), "best": pd.DataFrame()}

    if same_day_policy not in {"stop_first", "take_profit_first"}:
        raise ValueError(f"Unsupported same_day_policy: {same_day_policy}")
    if ma20_stop_tolerance < 0:
        raise ValueError("ma20_stop_tolerance must be non-negative")

    prepared = _prepare_forward_prices(forward_prices, require_ma20=ma20_stop)
    samples = _build_sample_trajectories(prepared)
    trade_rows: list[dict[str, object]] = []
    for holding_day in sorted({int(item) for item in holding_days}):
        if holding_day <= 0:
            raise ValueError("holding_days must be positive")
        for take_profit in sorted({float(item) for item in take_profits}):
            if take_profit <= 0:
                raise ValueError("take_profits must be positive")
            for stop_loss in sorted({float(item) for item in stop_losses}):
                if stop_loss <= 0:
                    raise ValueError("stop_losses must be positive")

                for sample in samples:
                    exit_row = _simulate_stop_trade(
                        sample,
                        holding_days=holding_day,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        same_day_policy=same_day_policy,
                        ma20_stop=ma20_stop,
                        ma20_stop_tolerance=ma20_stop_tolerance,
                    )
                    if exit_row is not None:
                        trade_rows.append(exit_row)

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        return {"trades": pd.DataFrame(), "summary": pd.DataFrame(), "best": pd.DataFrame()}

    summary = summarize_pattern_stop_trades(trades)
    best = select_best_pattern_stop_grid(summary)
    return {"trades": trades, "summary": summary, "best": best}


def summarize_pattern_stop_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = trades.groupby(["pattern_id", "strategy_name", "holding_days", "take_profit_pct", "stop_loss_pct"], dropna=False)
    for (pattern_id, strategy_name, holding_days, take_profit_pct, stop_loss_pct), subset in grouped:
        exit_counts = subset["exit_reason"].value_counts(normalize=True)
        rows.append(
            {
                "pattern_id": str(pattern_id),
                "strategy_name": strategy_name,
                "holding_days": int(holding_days),
                "take_profit_pct": round(float(take_profit_pct), 4),
                "stop_loss_pct": round(float(stop_loss_pct), 4),
                "sample_count": int(len(subset)),
                "win_rate": round(float((subset["realized_return_pct"] > 0).mean()), 4),
                "avg_return_pct": round(float(subset["realized_return_pct"].mean()), 4),
                "median_return_pct": round(float(subset["realized_return_pct"].median()), 4),
                "return_std_pct": round(float(subset["realized_return_pct"].std(ddof=0)), 4),
                "avg_exit_day": round(float(subset["exit_forward_day"].mean()), 2),
                "take_profit_rate": round(float(exit_counts.get("take_profit", 0.0)), 4),
                "stop_loss_rate": round(float(exit_counts.get("stop_loss", 0.0)), 4),
                "ma20_stop_rate": round(float(exit_counts.get("ma20_stop", 0.0)), 4),
                "time_exit_rate": round(float(exit_counts.get("time_exit", 0.0)), 4),
                "avg_max_upside_before_exit": round(float(subset["max_upside_before_exit"].mean()), 4),
                "avg_max_drawdown_before_exit": round(float(subset["max_drawdown_before_exit"].mean()), 4),
                "same_day_policy": subset["same_day_policy"].iloc[0],
                "ma20_stop": bool(subset["ma20_stop"].iloc[0]),
                "ma20_stop_tolerance": round(float(subset["ma20_stop_tolerance"].iloc[0]), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(["pattern_id", "holding_days", "take_profit_pct", "stop_loss_pct"]).reset_index(drop=True)


def select_best_pattern_stop_grid(summary: pd.DataFrame, *, min_samples: int = 30) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    candidates = summary[summary["sample_count"] >= int(min_samples)].copy()
    if candidates.empty:
        candidates = summary.copy()

    return (
        candidates.sort_values(
            ["pattern_id", "holding_days", "win_rate", "avg_return_pct", "avg_max_drawdown_before_exit"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["pattern_id", "holding_days"], as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def _prepare_forward_prices(forward_prices: pd.DataFrame, *, require_ma20: bool = False) -> pd.DataFrame:
    required_columns = {
        "sample_id",
        "pattern_id",
        "strategy_name",
        "forward_day",
        "trade_date",
        "entry_date",
        "entry_price",
        "high_return_pct",
        "low_return_pct",
        "close_return_pct",
    }
    if require_ma20:
        required_columns.add("ma_20")
        required_columns.add("close")
    missing = sorted(required_columns.difference(forward_prices.columns))
    if missing:
        raise ValueError(f"forward price data missing required columns: {', '.join(missing)}")

    frame = forward_prices.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["pattern_id"] = frame["pattern_id"].astype(str)
    frame["forward_day"] = frame["forward_day"].astype(int)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["entry_price"] = frame["entry_price"].astype(float)
    frame["high_return_pct"] = frame["high_return_pct"].astype(float)
    frame["low_return_pct"] = frame["low_return_pct"].astype(float)
    frame["close_return_pct"] = frame["close_return_pct"].astype(float)
    if "ma_20" in frame.columns:
        frame["ma_20"] = pd.to_numeric(frame["ma_20"], errors="coerce")
    return frame.sort_values(["sample_id", "forward_day"]).reset_index(drop=True)


def _build_sample_trajectories(forward_prices: pd.DataFrame) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for _, sample in forward_prices.groupby("sample_id", sort=False):
        ordered = sample.sort_values("forward_day").reset_index(drop=True)
        first = ordered.iloc[0]
        samples.append(
            {
                "sample_id": str(first["sample_id"]),
                "symbol": str(first.get("symbol", "")).zfill(6) if "symbol" in first else "",
                "name": first.get("name", ""),
                "pattern_id": str(first["pattern_id"]),
                "strategy_name": first["strategy_name"],
                "signal_date": pd.Timestamp(first["signal_date"]) if "signal_date" in first else pd.NaT,
                "entry_date": pd.Timestamp(first["entry_date"]),
                "entry_price": round(float(first["entry_price"]), 4),
                "forward_days": ordered["forward_day"].astype(int).tolist(),
                "trade_dates": pd.to_datetime(ordered["trade_date"]).tolist(),
                "high_returns": ordered["high_return_pct"].astype(float).tolist(),
                "low_returns": ordered["low_return_pct"].astype(float).tolist(),
                "close_returns": ordered["close_return_pct"].astype(float).tolist(),
                "close_prices": ordered["close"].astype(float).tolist() if "close" in ordered.columns else [],
                "ma20_values": ordered["ma_20"].astype(float).tolist() if "ma_20" in ordered.columns else [],
            }
        )
    return samples


def _simulate_stop_trade(
    sample: dict[str, object],
    *,
    holding_days: int,
    take_profit: float,
    stop_loss: float,
    same_day_policy: str,
    ma20_stop: bool,
    ma20_stop_tolerance: float,
) -> dict[str, object] | None:
    forward_days = sample["forward_days"]
    if not isinstance(forward_days, list) or len(forward_days) < holding_days:
        return None

    exit_reason = "time_exit"
    exit_offset = holding_days - 1
    high_returns = sample["high_returns"]
    low_returns = sample["low_returns"]
    close_returns = sample["close_returns"]
    trade_dates = sample["trade_dates"]
    close_prices = sample["close_prices"]
    ma20_values = sample["ma20_values"]
    if not isinstance(high_returns, list) or not isinstance(low_returns, list) or not isinstance(close_returns, list) or not isinstance(trade_dates, list):
        return None
    if ma20_stop and (not isinstance(close_prices, list) or not isinstance(ma20_values, list) or len(ma20_values) < holding_days):
        return None
    realized_return = float(close_returns[exit_offset])

    for offset in range(holding_days):
        hit_take_profit = float(high_returns[offset]) >= take_profit
        hit_stop_loss = float(low_returns[offset]) <= -stop_loss
        hit_ma20_stop = False
        if ma20_stop:
            ma20 = float(ma20_values[offset])
            close_price = float(close_prices[offset])
            hit_ma20_stop = pd.notna(ma20) and close_price < ma20 * (1.0 - ma20_stop_tolerance)
        if not hit_take_profit and not hit_stop_loss and not hit_ma20_stop:
            continue

        exit_offset = offset
        if hit_take_profit and hit_stop_loss:
            if same_day_policy == "take_profit_first":
                exit_reason = "take_profit"
                realized_return = take_profit
            else:
                exit_reason = "stop_loss"
                realized_return = -stop_loss
        elif hit_take_profit:
            exit_reason = "take_profit"
            realized_return = take_profit
        elif hit_stop_loss:
            exit_reason = "stop_loss"
            realized_return = -stop_loss
        else:
            exit_reason = "ma20_stop"
            realized_return = float(close_returns[offset])
        break

    max_upside = max(float(item) for item in high_returns[: exit_offset + 1])
    max_drawdown = max(0.0, -min(float(item) for item in low_returns[: exit_offset + 1]))
    return {
        "sample_id": str(sample["sample_id"]),
        "symbol": str(sample["symbol"]),
        "name": sample["name"],
        "pattern_id": str(sample["pattern_id"]),
        "strategy_name": sample["strategy_name"],
        "signal_date": pd.Timestamp(sample["signal_date"]),
        "entry_date": pd.Timestamp(sample["entry_date"]),
        "entry_price": round(float(sample["entry_price"]), 4),
        "holding_days": int(holding_days),
        "take_profit_pct": round(float(take_profit), 4),
        "stop_loss_pct": round(float(stop_loss), 4),
        "exit_date": pd.Timestamp(trade_dates[exit_offset]),
        "exit_forward_day": int(forward_days[exit_offset]),
        "exit_reason": exit_reason,
        "realized_return_pct": round(float(realized_return), 4),
        "max_upside_before_exit": round(max_upside, 4),
        "max_drawdown_before_exit": round(max_drawdown, 4),
        "same_day_policy": same_day_policy,
        "ma20_stop": bool(ma20_stop),
        "ma20_stop_tolerance": round(float(ma20_stop_tolerance), 4),
    }
