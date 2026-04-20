from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .models import TrendBacktestConfig


ENTRY_NOTES = {
    "same_close": "Research entry assumption: signal confirmed at close and entered at the same close.",
    "next_open": "Research entry assumption: score calculated at close and entered at the next open.",
}


def backtest_signal_returns(
    signals: pd.DataFrame,
    daily_history_by_symbol: dict[str, pd.DataFrame],
    config: TrendBacktestConfig,
    *,
    entry_timing: str | None = None,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    resolved_entry_timing = entry_timing or config.entry_timing
    prepared_history = _prepare_daily_history_map(daily_history_by_symbol)
    rows: list[dict[str, object]] = []
    for signal in signals.to_dict("records"):
        symbol = str(signal["symbol"]).zfill(6)
        history = prepared_history.get(symbol)
        if history is None:
            continue

        signal_trade_date = pd.Timestamp(signal["trade_date"]).date()
        signal_index = history["index_by_date"].get(signal_trade_date)
        if signal_index is None:
            continue

        for holding_days in config.holding_days:
            trade_row = _build_trade_row(signal, history["frame"], signal_index, holding_days, resolved_entry_timing)
            if trade_row is None:
                continue
            rows.append(trade_row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(["entry_date", "signal_type", "holding_days", "symbol"]).reset_index(drop=True)
    result["entry_note"] = ENTRY_NOTES.get(resolved_entry_timing, resolved_entry_timing)
    return result


def summarize_signal_backtest(backtest_results: pd.DataFrame) -> pd.DataFrame:
    if backtest_results.empty:
        return pd.DataFrame()

    threshold_trend = float(backtest_results["trend_score"].quantile(0.8))
    threshold_entry = float(backtest_results["entry_score"].quantile(0.8))
    groups = {
        "all": backtest_results,
        "high_trend_score": backtest_results[backtest_results["trend_score"] >= threshold_trend],
        "high_entry_score": backtest_results[backtest_results["entry_score"] >= threshold_entry],
    }

    rows: list[dict[str, object]] = []
    for sample_group, subset in groups.items():
        if subset.empty:
            continue
        grouped = subset.groupby(["signal_type", "holding_days"], dropna=False)
        for (signal_type, holding_days), frame in grouped:
            rows.append(
                {
                    "sample_group": sample_group,
                    "signal_type": signal_type,
                    "holding_days": int(holding_days),
                    "signal_count": int(len(frame)),
                    "win_rate": round(float((frame["return_pct"] > 0).mean()), 4),
                    "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
                    "median_return_pct": round(float(frame["return_pct"].median()), 4),
                    "return_std_pct": round(float(frame["return_pct"].std(ddof=0)), 4),
                    "avg_max_drawdown_pct": round(float(frame["max_drawdown_pct"].mean()), 4),
                    "avg_max_upside_pct": round(float(frame["max_upside_pct"].mean()), 4),
                    "avg_entry_score": round(float(frame["entry_score"].mean()), 4),
                    "avg_trend_score": round(float(frame["trend_score"].mean()), 4),
                    "avg_buy_score": round(float(frame["buy_score"].mean()), 4) if "buy_score" in frame.columns else None,
                    "entry_note": frame["entry_note"].iloc[0] if "entry_note" in frame.columns else None,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["sample_group", "signal_type", "holding_days"]).reset_index(drop=True)


def backtest_portfolios(
    signals: pd.DataFrame,
    daily_history_by_symbol: dict[str, pd.DataFrame],
    config: TrendBacktestConfig,
    *,
    entry_timing: str | None = None,
    rank_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    if signals.empty:
        return {
            "positions": pd.DataFrame(),
            "equity": pd.DataFrame(),
            "summary": pd.DataFrame(),
        }

    resolved_entry_timing = entry_timing or config.entry_timing
    prepared_history = _prepare_daily_history_map(daily_history_by_symbol)
    scored_signals = signals.copy()
    scored_signals["trade_date"] = pd.to_datetime(scored_signals["trade_date"])
    resolved_rank_column = rank_column or _resolve_rank_column(scored_signals)
    if resolved_rank_column == "portfolio_rank_score":
        scored_signals["portfolio_rank_score"] = (
            scored_signals["entry_score"] * config.entry_score_weight + scored_signals["trend_score"] * config.trend_score_weight
        ).round(4)
    else:
        scored_signals["portfolio_rank_score"] = scored_signals[resolved_rank_column].astype(float).round(4)

    position_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for top_n in config.portfolio_top_n:
        selected = (
            scored_signals.sort_values(["trade_date", "portfolio_rank_score", "entry_score", "symbol"], ascending=[True, False, False, True])
            .groupby("trade_date", group_keys=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        for holding_days in config.holding_days:
            position_rows: list[dict[str, object]] = []
            for signal in selected.to_dict("records"):
                symbol = str(signal["symbol"]).zfill(6)
                history = prepared_history.get(symbol)
                if history is None:
                    continue
                signal_trade_date = pd.Timestamp(signal["trade_date"]).date()
                signal_index = history["index_by_date"].get(signal_trade_date)
                if signal_index is None:
                    continue
                trade_row = _build_trade_row(signal, history["frame"], signal_index, holding_days, resolved_entry_timing)
                if trade_row is None:
                    continue
                trade_row["portfolio_top_n"] = int(top_n)
                trade_row["portfolio_rank_score"] = float(signal["portfolio_rank_score"])
                position_rows.append(trade_row)

            positions = pd.DataFrame(position_rows)
            if positions.empty:
                continue

            equity = _build_portfolio_equity_curve(positions, prepared_history, entry_timing=resolved_entry_timing)
            equity["holding_days"] = int(holding_days)
            equity["portfolio_top_n"] = int(top_n)
            equity["entry_note"] = ENTRY_NOTES.get(resolved_entry_timing, resolved_entry_timing)

            positions["holding_days"] = int(holding_days)
            positions["portfolio_top_n"] = int(top_n)
            positions["entry_note"] = ENTRY_NOTES.get(resolved_entry_timing, resolved_entry_timing)

            position_frames.append(positions)
            equity_frames.append(equity)
            summary_rows.append(_summarize_portfolio_run(positions, equity, top_n=top_n, holding_days=holding_days))

    positions_df = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    equity_df = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["portfolio_top_n", "holding_days"]).reset_index(drop=True) if summary_rows else pd.DataFrame()
    return {
        "positions": positions_df,
        "equity": equity_df,
        "summary": summary_df,
    }


def _build_trade_row(
    signal: dict[str, object],
    history: pd.DataFrame,
    signal_index: int,
    holding_days: int,
    entry_timing: str,
) -> dict[str, object] | None:
    if entry_timing == "next_open":
        entry_index = signal_index + 1
        exit_index = entry_index + int(holding_days) - 1
        if entry_index >= len(history) or exit_index >= len(history):
            return None
        entry_row = history.iloc[entry_index]
        future_slice = history.iloc[entry_index : exit_index + 1]
        exit_row = history.iloc[exit_index]
        entry_price = float(entry_row["open"])
    else:
        entry_index = signal_index
        exit_index = entry_index + int(holding_days)
        if exit_index >= len(history):
            return None
        entry_row = history.iloc[entry_index]
        future_slice = history.iloc[entry_index + 1 : exit_index + 1]
        if future_slice.empty:
            return None
        exit_row = history.iloc[exit_index]
        entry_price = float(entry_row["close"])
    if entry_price <= 0:
        return None

    max_upside_pct = float(future_slice["high"].max()) / entry_price - 1
    max_drawdown_pct = 1 - float(future_slice["low"].min()) / entry_price
    min_return_pct = float((future_slice["close"].astype(float) / entry_price - 1).min())
    return_pct = float(exit_row["close"]) / entry_price - 1

    return {
        "trade_date": pd.Timestamp(signal["trade_date"]),
        "signal_date": pd.Timestamp(signal["trade_date"]),
        "entry_date": pd.Timestamp(entry_row["trade_date"]),
        "entry_price": round(entry_price, 4),
        "exit_date": pd.Timestamp(exit_row["trade_date"]),
        "exit_price": round(float(exit_row["close"]), 4),
        "holding_days": int(holding_days),
        "symbol": str(signal["symbol"]).zfill(6),
        "name": signal.get("name", ""),
        "signal_type": signal.get("signal_type"),
        "trend_score": round(float(signal.get("trend_score", 0.0)), 4),
        "entry_score": round(float(signal.get("entry_score", 0.0)), 4),
        "buy_score": round(float(signal.get("buy_score", signal.get("entry_score", 0.0))), 4),
        "return_pct": round(return_pct, 4),
        "max_upside_pct": round(max_upside_pct, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "min_return_pct": round(min_return_pct, 4),
        "trigger_reason": signal.get("trigger_reason", ""),
        "entry_timing": entry_timing,
    }


def _build_portfolio_equity_curve(
    positions: pd.DataFrame,
    prepared_history: dict[str, dict[str, object]],
    *,
    entry_timing: str,
) -> pd.DataFrame:
    daily_returns: dict[pd.Timestamp, list[float]] = defaultdict(list)
    for position in positions.to_dict("records"):
        symbol = str(position["symbol"]).zfill(6)
        history = prepared_history.get(symbol)
        if history is None:
            continue

        entry_trade_date = pd.Timestamp(position["entry_date"]).date()
        exit_trade_date = pd.Timestamp(position["exit_date"]).date()
        entry_index = history["index_by_date"].get(entry_trade_date)
        exit_index = history["index_by_date"].get(exit_trade_date)
        if entry_index is None or exit_index is None or exit_index <= entry_index:
            continue

        slice_frame = history["frame"].iloc[entry_index : exit_index + 1].reset_index(drop=True)
        if entry_timing == "next_open":
            entry_row = slice_frame.iloc[0]
            entry_open = float(entry_row["open"])
            entry_close = float(entry_row["close"])
            if entry_open > 0:
                daily_returns[pd.Timestamp(entry_row["trade_date"])].append(entry_close / entry_open - 1)
            previous_close = entry_close
            start_offset = 1
        else:
            previous_close = float(slice_frame.iloc[0]["close"])
            start_offset = 1

        for index in range(start_offset, len(slice_frame)):
            row = slice_frame.iloc[index]
            current_close = float(row["close"])
            if previous_close <= 0:
                previous_close = current_close
                continue
            current_trade_date = pd.Timestamp(row["trade_date"])
            daily_returns[current_trade_date].append(current_close / previous_close - 1)
            previous_close = current_close

    if not daily_returns:
        return pd.DataFrame()

    net_value = 1.0
    peak_value = 1.0
    rows: list[dict[str, object]] = []
    for trade_date in sorted(daily_returns):
        returns = daily_returns[trade_date]
        daily_return = sum(returns) / len(returns) if returns else 0.0
        net_value *= 1 + daily_return
        peak_value = max(peak_value, net_value)
        rows.append(
            {
                "trade_date": trade_date,
                "daily_return": round(daily_return, 6),
                "net_value": round(net_value, 6),
                "drawdown": round(0.0 if peak_value <= 0 else 1 - net_value / peak_value, 6),
                "active_positions": int(len(returns)),
            }
        )

    return pd.DataFrame(rows)


def _summarize_portfolio_run(
    positions: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    top_n: int,
    holding_days: int,
) -> dict[str, object]:
    final_net_value = float(equity["net_value"].iloc[-1]) if not equity.empty else 1.0
    periods = max(len(equity), 1)
    annualized_return = final_net_value ** (252 / periods) - 1 if periods > 1 else final_net_value - 1
    signal_type_share = positions["signal_type"].value_counts(normalize=True)
    return {
        "portfolio_top_n": int(top_n),
        "holding_days": int(holding_days),
        "position_count": int(len(positions)),
        "win_rate": round(float((positions["return_pct"] > 0).mean()), 4),
        "avg_return_pct": round(float(positions["return_pct"].mean()), 4),
        "avg_entry_score": round(float(positions["entry_score"].mean()), 4),
        "avg_trend_score": round(float(positions["trend_score"].mean()), 4),
        "final_net_value": round(final_net_value, 6),
        "annualized_return": round(float(annualized_return), 4),
        "max_drawdown": round(float(equity["drawdown"].max()), 4) if not equity.empty else 0.0,
        "avg_active_positions": round(float(equity["active_positions"].mean()), 4) if not equity.empty else 0.0,
        "breakout_share": round(float(signal_type_share.get("breakout", 0.0)), 4),
        "pullback_share": round(float(signal_type_share.get("pullback", 0.0)), 4),
        "avg_buy_score": round(float(positions["buy_score"].mean()), 4) if "buy_score" in positions.columns else None,
        "entry_note": positions["entry_note"].iloc[0] if "entry_note" in positions.columns else None,
    }


def _prepare_daily_history_map(daily_history_by_symbol: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    prepared: dict[str, dict[str, object]] = {}
    for symbol, frame in daily_history_by_symbol.items():
        normalized_symbol = str(symbol).zfill(6)
        history = frame.copy().sort_values("trade_date").reset_index(drop=True)
        history["trade_date"] = pd.to_datetime(history["trade_date"])
        history["trade_day"] = history["trade_date"].dt.date
        prepared[normalized_symbol] = {
            "frame": history,
            "index_by_date": {trade_day: index for index, trade_day in enumerate(history["trade_day"].tolist())},
        }
    return prepared


def _resolve_rank_column(signals: pd.DataFrame) -> str:
    if "buy_score" in signals.columns:
        return "buy_score"
    if "portfolio_rank_score" in signals.columns:
        return "portfolio_rank_score"
    return "entry_score"
