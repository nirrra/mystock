from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EXIT_STRATEGIES = ("centered_risk_top20", "mixed_010_top20", "mixed_top20", "all90", "phase4_top20")
DEFAULT_STOP_GRID = (0.06, 0.08, 0.10)
DEFAULT_TAKE_GRID = (0.08, 0.10, 0.12, 0.15, 0.20)
DEFAULT_TRAILING_GRID = (0.06, 0.08, 0.10)
DEFAULT_BREAKEVEN_TRIGGER_GRID = (0.08, 0.10)
DEFAULT_TIME_STOP_DAYS_GRID = (5, 10)
DEFAULT_TIME_STOP_MIN_RETURN_GRID = (0.02, 0.03)


@dataclass(slots=True)
class ExitRuleOptimizationResult:
    summary: pd.DataFrame
    by_year: pd.DataFrame
    by_split: pd.DataFrame
    selection_report: pd.DataFrame
    output_dir: Path
    summary_path: Path
    by_year_path: Path
    by_split_path: Path
    selection_report_path: Path
    recommendations_path: Path
    trades_path: Path | None = None


def optimize_exit_rules(
    *,
    strict_dir: Path,
    strategies: tuple[str, ...] = DEFAULT_EXIT_STRATEGIES,
    horizons: tuple[int, ...] = (5, 10, 20, 60),
    output_dir: Path | None = None,
    stop_grid: tuple[float, ...] = DEFAULT_STOP_GRID,
    take_grid: tuple[float, ...] = DEFAULT_TAKE_GRID,
    trailing_grid: tuple[float, ...] = DEFAULT_TRAILING_GRID,
    breakeven_trigger_grid: tuple[float, ...] = DEFAULT_BREAKEVEN_TRIGGER_GRID,
    time_stop_days_grid: tuple[int, ...] = DEFAULT_TIME_STOP_DAYS_GRID,
    time_stop_min_return_grid: tuple[float, ...] = DEFAULT_TIME_STOP_MIN_RETURN_GRID,
    tune_end_date: date = date(2023, 12, 29),
    test_start_date: date = date(2024, 1, 1),
    save_trades: bool = False,
    progress: bool = False,
) -> ExitRuleOptimizationResult:
    strict_dir = strict_dir.resolve()
    output_root = (output_dir or strict_dir / "exit_rule_optimization").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    paths = _load_selected_forward_paths(strict_dir)
    if paths.empty:
        raise RuntimeError(f"No selected forward paths found in {strict_dir}")
    paths = _normalize_paths(paths)
    if strategies:
        wanted = {str(item) for item in strategies}
        paths = paths[paths["strategy"].astype(str).isin(wanted)].copy()
    if paths.empty:
        raise RuntimeError("No selected forward paths remain after strategy filtering.")
    rules = _build_rule_specs(
        stop_grid=stop_grid,
        take_grid=take_grid,
        trailing_grid=trailing_grid,
        breakeven_trigger_grid=breakeven_trigger_grid,
        time_stop_days_grid=time_stop_days_grid,
        time_stop_min_return_grid=time_stop_min_return_grid,
    )
    if not rules:
        raise RuntimeError("No exit rules were generated.")
    if progress:
        print(
            f"Exit-rule optimization started: groups={paths[['strategy','signal_date','symbol','selected_rank']].drop_duplicates().shape[0]} "
            f"rules={len(rules)} elapsed=0s",
            flush=True,
        )
    if save_trades:
        print("--save-trades is ignored by the vectorized optimizer to avoid writing very large intermediate files.", flush=True)
    matrix = _build_path_matrix(paths, tune_end_date=tune_end_date, test_start_date=test_start_date)
    summary_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    total_jobs = sum(1 for horizon in horizons for rule in rules if _rule_applies_to_horizon(rule, int(horizon)))
    job_index = 0
    for horizon in horizons:
        horizon = int(horizon)
        for rule in rules:
            if not _rule_applies_to_horizon(rule, horizon):
                continue
            job_index += 1
            if progress and (job_index == 1 or job_index == total_jobs or job_index % 25 == 0):
                elapsed = perf_counter() - started_at
                avg = elapsed / max(job_index, 1)
                eta = avg * max(total_jobs - job_index, 0)
                print(
                    f"Exit-rule vector progress: {job_index}/{total_jobs} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                    flush=True,
                )
            outcome = simulate_exit_rule_matrix(matrix, rule=rule, horizon=horizon)
            base = {"horizon": horizon, **rule}
            summary_rows.extend(_aggregate_outcome_by_category(outcome, matrix["strategy"], base=base, category_name=None))
            year_rows.extend(_aggregate_outcome_by_category(outcome, matrix["year"], base=base, category_name="year", strategy=matrix["strategy"]))
            split_rows.extend(_aggregate_outcome_by_category(outcome, matrix["split"], base=base, category_name="split", strategy=matrix["strategy"]))
    summary = _finalize_stats_frame(pd.DataFrame(summary_rows))
    by_year = _finalize_stats_frame(pd.DataFrame(year_rows))
    by_split = _finalize_stats_frame(pd.DataFrame(split_rows))
    selection_report = _build_selection_report(by_split)
    summary_path = output_root / "exit_rule_summary.csv"
    by_year_path = output_root / "exit_rule_by_year.csv"
    by_split_path = output_root / "exit_rule_by_split.csv"
    selection_report_path = output_root / "exit_rule_selection_report.csv"
    recommendations_path = output_root / "recommended_exit_rules.md"
    trades_path: Path | None = None
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")
    by_split.to_csv(by_split_path, index=False, encoding="utf-8-sig")
    selection_report.to_csv(selection_report_path, index=False, encoding="utf-8-sig")
    _write_recommendations(recommendations_path, selection_report=selection_report, summary=summary)
    return ExitRuleOptimizationResult(
        summary=summary,
        by_year=by_year,
        by_split=by_split,
        selection_report=selection_report,
        output_dir=output_root,
        summary_path=summary_path,
        by_year_path=by_year_path,
        by_split_path=by_split_path,
        selection_report_path=selection_report_path,
        recommendations_path=recommendations_path,
        trades_path=trades_path,
    )


def simulate_exit_rule_path(path: pd.DataFrame, *, rule: dict[str, Any], horizon: int) -> dict[str, Any] | None:
    frame = path.sort_values("day_offset").head(int(horizon)).copy()
    if len(frame) < int(horizon):
        return None
    entry = _safe_float(frame.iloc[0].get("entry_open"))
    if entry is None:
        entry = _safe_float(frame.iloc[0].get("open"))
    if entry is None or entry <= 0:
        return None
    initial_stop_pct = float(rule.get("initial_stop_pct", 0.08))
    stop_enabled = not bool(rule.get("disable_stop_loss"))
    stop_price = entry * (1.0 - initial_stop_pct) if stop_enabled else None
    take_pct = _optional_float(rule.get("take_profit_pct"))
    take_price = entry * (1.0 + take_pct) if take_pct is not None else None
    breakeven_trigger = _optional_float(rule.get("breakeven_trigger_pct"))
    trailing_start = _optional_float(rule.get("trailing_start_pct"))
    trailing_pct = _optional_float(rule.get("trailing_pct"))
    time_stop_days = _optional_int(rule.get("time_stop_days"))
    time_stop_min_return = _optional_float(rule.get("time_stop_min_return"))
    highest_high = entry
    breakeven_active = False
    trailing_active = False
    exit_reason = "timeout"
    exit_price = _safe_float(frame.iloc[-1].get("close"))
    exit_date = str(frame.iloc[-1].get("trade_date"))
    holding_days = int(horizon)
    exit_index = len(frame) - 1
    for offset, row in enumerate(frame.itertuples(index=False), start=1):
        row_open = float(getattr(row, "open"))
        row_high = float(getattr(row, "high"))
        row_low = float(getattr(row, "low"))
        row_close = float(getattr(row, "close"))
        row_date = str(getattr(row, "trade_date"))
        active_stop = stop_price
        stop_reason = "stop_loss"
        if active_stop is not None and breakeven_active and entry > active_stop:
            active_stop = entry
            stop_reason = "breakeven_stop"
        if active_stop is not None and trailing_active and trailing_pct is not None:
            trailing_stop = highest_high * (1.0 - trailing_pct)
            if trailing_stop > active_stop:
                active_stop = trailing_stop
                stop_reason = "trailing_stop"
        if active_stop is not None and row_open <= active_stop:
            exit_reason = f"{stop_reason}_open_gap"
            exit_price = row_open
            exit_date = row_date
            holding_days = offset
            exit_index = offset - 1
            break
        if take_price is not None and row_open >= take_price:
            exit_reason = "take_profit_open_gap"
            exit_price = row_open
            exit_date = row_date
            holding_days = offset
            exit_index = offset - 1
            break
        if active_stop is not None and take_price is not None and row_low <= active_stop and row_high >= take_price:
            exit_reason = f"{stop_reason}_first"
            exit_price = active_stop
            exit_date = row_date
            holding_days = offset
            exit_index = offset - 1
            break
        if active_stop is not None and row_low <= active_stop:
            exit_reason = stop_reason
            exit_price = active_stop
            exit_date = row_date
            holding_days = offset
            exit_index = offset - 1
            break
        if take_price is not None and row_high >= take_price:
            exit_reason = "take_profit"
            exit_price = take_price
            exit_date = row_date
            holding_days = offset
            exit_index = offset - 1
            break
        if time_stop_days is not None and offset == time_stop_days and time_stop_min_return is not None:
            if row_close / entry - 1.0 < time_stop_min_return:
                exit_reason = "time_stop"
                exit_price = row_close
                exit_date = row_date
                holding_days = offset
                exit_index = offset - 1
                break
        if row_high > highest_high:
            highest_high = row_high
        if breakeven_trigger is not None and row_high / entry - 1.0 >= breakeven_trigger:
            breakeven_active = True
        if trailing_start is not None and row_high / entry - 1.0 >= trailing_start:
            trailing_active = True
    if exit_price is None or not math.isfinite(float(exit_price)):
        return None
    observed = frame.iloc[: exit_index + 1]
    max_profit = pd.to_numeric(observed["high"], errors="coerce").max() / entry - 1.0
    max_drawdown = pd.to_numeric(observed["low"], errors="coerce").min() / entry - 1.0
    rule_return = float(exit_price) / entry - 1.0
    return {
        "entry_open": entry,
        "exit_date": exit_date,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "holding_days": holding_days,
        "rule_return": rule_return,
        "rule_R": rule_return / initial_stop_pct if initial_stop_pct > 0 else math.nan,
        "max_profit": float(max_profit),
        "max_drawdown": float(max_drawdown),
        "win": rule_return > 0,
        "stop_loss_hit": exit_reason.startswith("stop_loss"),
        "take_profit_hit": exit_reason.startswith("take_profit"),
        "trailing_stop_hit": exit_reason.startswith("trailing_stop"),
        "breakeven_stop_hit": exit_reason.startswith("breakeven_stop"),
        "time_stop_hit": exit_reason == "time_stop",
    }


def simulate_exit_rule_matrix(matrix: dict[str, Any], *, rule: dict[str, Any], horizon: int) -> dict[str, np.ndarray]:
    open_values = matrix["open"][:, :horizon]
    high_values = matrix["high"][:, :horizon]
    low_values = matrix["low"][:, :horizon]
    close_values = matrix["close"][:, :horizon]
    entry = matrix["entry_open"]
    n = len(entry)
    valid = np.isfinite(entry) & (entry > 0) & np.isfinite(close_values[:, -1])
    initial_stop_pct = float(rule.get("initial_stop_pct", 0.08))
    stop_enabled = not bool(rule.get("disable_stop_loss"))
    stop_price = entry * (1.0 - initial_stop_pct) if stop_enabled else np.full(n, -np.inf, dtype=np.float64)
    take_pct = _optional_float(rule.get("take_profit_pct"))
    take_price = entry * (1.0 + take_pct) if take_pct is not None else np.full(n, np.nan, dtype=np.float32)
    breakeven_trigger = _optional_float(rule.get("breakeven_trigger_pct"))
    trailing_start = _optional_float(rule.get("trailing_start_pct"))
    trailing_pct = _optional_float(rule.get("trailing_pct"))
    time_stop_days = _optional_int(rule.get("time_stop_days"))
    time_stop_min_return = _optional_float(rule.get("time_stop_min_return"))

    active = valid.copy()
    exit_price = close_values[:, -1].astype("float64", copy=True)
    holding_days = np.full(n, int(horizon), dtype=np.float32)
    max_high_seen = np.full(n, np.nan, dtype=np.float64)
    min_low_seen = np.full(n, np.nan, dtype=np.float64)
    highest_high = entry.astype("float64", copy=True)
    breakeven_active = np.zeros(n, dtype=bool)
    trailing_active = np.zeros(n, dtype=bool)
    stop_loss_hit = np.zeros(n, dtype=bool)
    take_profit_hit = np.zeros(n, dtype=bool)
    trailing_stop_hit = np.zeros(n, dtype=bool)
    breakeven_stop_hit = np.zeros(n, dtype=bool)
    time_stop_hit = np.zeros(n, dtype=bool)

    for offset in range(horizon):
        current = active & np.isfinite(open_values[:, offset]) & np.isfinite(high_values[:, offset]) & np.isfinite(low_values[:, offset])
        if not current.any():
            continue
        max_high_seen[current] = np.fmax(np.nan_to_num(max_high_seen[current], nan=-np.inf), high_values[current, offset])
        min_low_seen[current] = np.fmin(np.nan_to_num(min_low_seen[current], nan=np.inf), low_values[current, offset])
        active_stop = stop_price.astype("float64", copy=True)
        stop_kind = np.zeros(n, dtype=np.int8)  # 0 stop, 1 breakeven, 2 trailing
        be_mask = breakeven_active & (entry > active_stop)
        active_stop[be_mask] = entry[be_mask]
        stop_kind[be_mask] = 1
        if trailing_pct is not None:
            trailing_stop = highest_high * (1.0 - trailing_pct)
            trail_mask = trailing_active & (trailing_stop > active_stop)
            active_stop[trail_mask] = trailing_stop[trail_mask]
            stop_kind[trail_mask] = 2

        open_stop = current & (open_values[:, offset] <= active_stop)
        _apply_matrix_exit(
            open_stop,
            offset=offset,
            price=open_values[:, offset],
            active=active,
            exit_price=exit_price,
            holding_days=holding_days,
            stop_loss_hit=stop_loss_hit,
            trailing_stop_hit=trailing_stop_hit,
            breakeven_stop_hit=breakeven_stop_hit,
            stop_kind=stop_kind,
        )
        if take_pct is not None:
            open_take = active & current & (open_values[:, offset] >= take_price)
            _apply_matrix_take_exit(open_take, offset=offset, price=open_values[:, offset], active=active, exit_price=exit_price, holding_days=holding_days, take_profit_hit=take_profit_hit)
            both = active & current & (low_values[:, offset] <= active_stop) & (high_values[:, offset] >= take_price)
        else:
            both = np.zeros(n, dtype=bool)
        _apply_matrix_exit(
            both,
            offset=offset,
            price=active_stop,
            active=active,
            exit_price=exit_price,
            holding_days=holding_days,
            stop_loss_hit=stop_loss_hit,
            trailing_stop_hit=trailing_stop_hit,
            breakeven_stop_hit=breakeven_stop_hit,
            stop_kind=stop_kind,
        )
        stop = active & current & (low_values[:, offset] <= active_stop)
        _apply_matrix_exit(
            stop,
            offset=offset,
            price=active_stop,
            active=active,
            exit_price=exit_price,
            holding_days=holding_days,
            stop_loss_hit=stop_loss_hit,
            trailing_stop_hit=trailing_stop_hit,
            breakeven_stop_hit=breakeven_stop_hit,
            stop_kind=stop_kind,
        )
        if take_pct is not None:
            take = active & current & (high_values[:, offset] >= take_price)
            _apply_matrix_take_exit(take, offset=offset, price=take_price, active=active, exit_price=exit_price, holding_days=holding_days, take_profit_hit=take_profit_hit)
        if time_stop_days is not None and offset + 1 == time_stop_days and time_stop_min_return is not None:
            time_stop = active & current & np.isfinite(close_values[:, offset]) & ((close_values[:, offset] / entry - 1.0) < time_stop_min_return)
            exit_price[time_stop] = close_values[time_stop, offset]
            holding_days[time_stop] = float(offset + 1)
            time_stop_hit[time_stop] = True
            active[time_stop] = False
        still_active = active & current
        highest_high[still_active] = np.fmax(highest_high[still_active], high_values[still_active, offset])
        if breakeven_trigger is not None:
            breakeven_active[still_active & ((high_values[:, offset] / entry - 1.0) >= breakeven_trigger)] = True
        if trailing_start is not None:
            trailing_active[still_active & ((high_values[:, offset] / entry - 1.0) >= trailing_start)] = True

    timeout = valid & active
    max_high_seen[timeout] = np.nanmax(high_values[timeout, :horizon], axis=1)
    min_low_seen[timeout] = np.nanmin(low_values[timeout, :horizon], axis=1)
    rule_return = exit_price / entry - 1.0
    return {
        "valid": valid & np.isfinite(rule_return),
        "rule_return": rule_return,
        "rule_R": rule_return / initial_stop_pct if initial_stop_pct > 0 else np.full(n, np.nan),
        "max_profit": max_high_seen / entry - 1.0,
        "max_drawdown": min_low_seen / entry - 1.0,
        "holding_days": holding_days,
        "stop_loss_hit": stop_loss_hit,
        "take_profit_hit": take_profit_hit,
        "trailing_stop_hit": trailing_stop_hit,
        "breakeven_stop_hit": breakeven_stop_hit,
        "time_stop_hit": time_stop_hit,
    }


def _apply_matrix_exit(
    mask: np.ndarray,
    *,
    offset: int,
    price: np.ndarray,
    active: np.ndarray,
    exit_price: np.ndarray,
    holding_days: np.ndarray,
    stop_loss_hit: np.ndarray,
    trailing_stop_hit: np.ndarray,
    breakeven_stop_hit: np.ndarray,
    stop_kind: np.ndarray,
) -> None:
    if not mask.any():
        return
    exit_price[mask] = price[mask]
    holding_days[mask] = float(offset + 1)
    trailing = mask & (stop_kind == 2)
    breakeven = mask & (stop_kind == 1)
    regular = mask & (stop_kind == 0)
    trailing_stop_hit[trailing] = True
    breakeven_stop_hit[breakeven] = True
    stop_loss_hit[regular] = True
    active[mask] = False


def _apply_matrix_take_exit(
    mask: np.ndarray,
    *,
    offset: int,
    price: np.ndarray,
    active: np.ndarray,
    exit_price: np.ndarray,
    holding_days: np.ndarray,
    take_profit_hit: np.ndarray,
) -> None:
    if not mask.any():
        return
    exit_price[mask] = price[mask]
    holding_days[mask] = float(offset + 1)
    take_profit_hit[mask] = True
    active[mask] = False


def _build_path_matrix(paths: pd.DataFrame, *, tune_end_date: date, test_start_date: date) -> dict[str, Any]:
    group_columns = ["strategy", "signal_date", "symbol", "selected_rank"]
    ordered = paths.sort_values(group_columns + ["day_offset"]).reset_index(drop=True)
    trade_id = ordered.groupby(group_columns, sort=False).ngroup().to_numpy(dtype=np.int64)
    meta = ordered.loc[~ordered.duplicated(group_columns), group_columns].reset_index(drop=True)
    max_horizon = int(pd.to_numeric(ordered["day_offset"], errors="coerce").max())
    n = int(trade_id.max() + 1) if len(trade_id) else 0
    arrays = {
        "open": np.full((n, max_horizon), np.nan, dtype=np.float32),
        "high": np.full((n, max_horizon), np.nan, dtype=np.float32),
        "low": np.full((n, max_horizon), np.nan, dtype=np.float32),
        "close": np.full((n, max_horizon), np.nan, dtype=np.float32),
    }
    offsets = pd.to_numeric(ordered["day_offset"], errors="coerce").to_numpy(dtype=np.int64) - 1
    valid_offsets = (offsets >= 0) & (offsets < max_horizon)
    for column, target in arrays.items():
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=np.float32)
        target[trade_id[valid_offsets], offsets[valid_offsets]] = values[valid_offsets]
    if "entry_open" in ordered.columns:
        entry_by_row = pd.to_numeric(ordered["entry_open"], errors="coerce").to_numpy(dtype=np.float32)
        entry_open = np.full(n, np.nan, dtype=np.float32)
        first_rows = ~ordered.duplicated(group_columns).to_numpy()
        entry_open[trade_id[first_rows]] = entry_by_row[first_rows]
    else:
        entry_open = arrays["open"][:, 0].copy()
    missing_entry = ~np.isfinite(entry_open)
    entry_open[missing_entry] = arrays["open"][missing_entry, 0]
    signal_dates = pd.to_datetime(meta["signal_date"], errors="coerce").dt.date
    years = signal_dates.map(lambda item: int(item.year) if pd.notna(item) else -1).to_numpy()
    splits = signal_dates.map(lambda item: _split_name(item, tune_end_date=tune_end_date, test_start_date=test_start_date) if pd.notna(item) else "unknown").astype(str).to_numpy()
    return {
        **arrays,
        "entry_open": entry_open,
        "strategy": meta["strategy"].astype(str).to_numpy(),
        "year": years,
        "split": splits,
    }


def _aggregate_outcome_by_category(
    outcome: dict[str, np.ndarray],
    category: np.ndarray,
    *,
    base: dict[str, Any],
    category_name: str | None,
    strategy: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    valid = outcome["valid"]
    rows: list[dict[str, Any]] = []
    if strategy is None:
        combos = [(item,) for item in pd.unique(pd.Series(category[valid]))]
    else:
        combo_frame = pd.DataFrame({"strategy": strategy[valid], "category": category[valid]}).drop_duplicates()
        combos = [tuple(row) for row in combo_frame.loc[:, ["strategy", "category"]].to_numpy()]
    for combo in combos:
        if strategy is None:
            strategy_value = combo[0]
            mask = valid & (category == strategy_value)
            row_base = {**base, "strategy": strategy_value}
        else:
            strategy_value, category_value = combo
            mask = valid & (strategy == strategy_value) & (category == category_value)
            row_base = {**base, "strategy": strategy_value, str(category_name): category_value}
        row = _outcome_stats_row(outcome, mask, row_base)
        if row:
            rows.append(row)
    return rows


def _outcome_stats_row(outcome: dict[str, np.ndarray], mask: np.ndarray, base: dict[str, Any]) -> dict[str, Any] | None:
    count = int(mask.sum())
    if count <= 0:
        return None
    returns = outcome["rule_return"][mask]
    holding_days = np.maximum(outcome["holding_days"][mask], 1.0)
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    avg_gain = float(np.nanmean(gains)) if len(gains) else math.nan
    avg_loss = float(np.nanmean(losses)) if len(losses) else math.nan
    payoff = avg_gain / abs(avg_loss) if math.isfinite(avg_gain) and math.isfinite(avg_loss) and avg_loss != 0 else math.nan
    row = {
        **base,
        "trade_count": count,
        "avg_return": float(np.nanmean(returns)),
        "avg_daily_return": float(np.nanmean(returns / holding_days)),
        "win_rate": float(np.nanmean(returns > 0)),
        "avg_R": float(np.nanmean(outcome["rule_R"][mask])),
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "avg_max_profit": float(np.nanmean(outcome["max_profit"][mask])),
        "avg_max_drawdown": float(np.nanmean(outcome["max_drawdown"][mask])),
        "avg_holding_days": float(np.nanmean(holding_days)),
        "stop_loss_rate": float(np.nanmean(outcome["stop_loss_hit"][mask])),
        "take_profit_rate": float(np.nanmean(outcome["take_profit_hit"][mask])),
        "trailing_stop_rate": float(np.nanmean(outcome["trailing_stop_hit"][mask])),
        "breakeven_stop_rate": float(np.nanmean(outcome["breakeven_stop_hit"][mask])),
        "time_stop_rate": float(np.nanmean(outcome["time_stop_hit"][mask])),
    }
    clipped_payoff = min(float(payoff), 5.0) if math.isfinite(payoff) else 0.0
    horizon = max(float(row.get("horizon", 1)), 1.0)
    row["balanced_score"] = (
        float(row["avg_R"])
        + 0.3 * float(row["win_rate"])
        + 0.2 * clipped_payoff
        - 0.5 * abs(float(row["avg_max_drawdown"]))
        - 0.3 * float(row["stop_loss_rate"])
        - 0.05 * (float(row["avg_holding_days"]) / horizon)
    )
    return row


def _finalize_stats_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [column for column in ("strategy", "horizon", "balanced_score") if column in frame.columns]
    ascending = [True, True, False][: len(sort_columns)]
    return frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def _load_selected_forward_paths(strict_dir: Path) -> pd.DataFrame:
    parquet_path = strict_dir / "selected_forward_paths.parquet"
    csv_path = strict_dir / "selected_forward_paths.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"selected_forward_paths.parquet/csv not found in {strict_dir}")


def _normalize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    result = paths.copy()
    required = {"strategy", "signal_date", "symbol", "selected_rank", "day_offset", "open", "high", "low", "close"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"selected_forward_paths is missing required columns: {missing}")
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce").dt.date.astype(str)
    for column in ("selected_rank", "day_offset"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    for column in ("open", "high", "low", "close", "entry_open"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["signal_date", "day_offset", "open", "high", "low", "close"])
    return result


def _build_rule_specs(
    *,
    stop_grid: tuple[float, ...],
    take_grid: tuple[float, ...],
    trailing_grid: tuple[float, ...],
    breakeven_trigger_grid: tuple[float, ...],
    time_stop_days_grid: tuple[int, ...],
    time_stop_min_return_grid: tuple[float, ...],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    reference_stop = float(stop_grid[0]) if stop_grid else 0.08
    rules.append(
        {
            "rule_id": "fixed_horizon",
            "rule_family": "fixed_horizon",
            "initial_stop_pct": reference_stop,
            "disable_stop_loss": True,
        }
    )
    for stop in stop_grid:
        for take in take_grid:
            rules.append(
                {
                    "rule_id": f"fixed_sl{_pct_id(stop)}_tp{_pct_id(take)}",
                    "rule_family": "fixed",
                    "initial_stop_pct": float(stop),
                    "take_profit_pct": float(take),
                }
            )
    for stop in stop_grid:
        for trigger in breakeven_trigger_grid:
            for take in take_grid:
                rules.append(
                    {
                        "rule_id": f"breakeven_sl{_pct_id(stop)}_be{_pct_id(trigger)}_tp{_pct_id(take)}",
                        "rule_family": "breakeven",
                        "initial_stop_pct": float(stop),
                        "take_profit_pct": float(take),
                        "breakeven_trigger_pct": float(trigger),
                    }
                )
    for stop in stop_grid:
        for trigger in breakeven_trigger_grid:
            for trail in trailing_grid:
                rules.append(
                    {
                        "rule_id": f"trailing_sl{_pct_id(stop)}_start{_pct_id(trigger)}_trail{_pct_id(trail)}",
                        "rule_family": "trailing",
                        "initial_stop_pct": float(stop),
                        "trailing_start_pct": float(trigger),
                        "trailing_pct": float(trail),
                    }
                )
    for stop in stop_grid:
        for take in take_grid:
            for day in time_stop_days_grid:
                for min_return in time_stop_min_return_grid:
                    rules.append(
                        {
                            "rule_id": f"time_sl{_pct_id(stop)}_tp{_pct_id(take)}_d{int(day)}_min{_pct_id(min_return)}",
                            "rule_family": "time_stop",
                            "initial_stop_pct": float(stop),
                            "take_profit_pct": float(take),
                            "time_stop_days": int(day),
                            "time_stop_min_return": float(min_return),
                        }
                    )
    return rules


def _rule_applies_to_horizon(rule: dict[str, Any], horizon: int) -> bool:
    time_stop_days = _optional_int(rule.get("time_stop_days"))
    return time_stop_days is None or time_stop_days <= int(horizon)


def _update_stats(target: dict[tuple[Any, ...], dict[str, Any]], key: tuple[Any, ...], base: dict[str, Any], outcome: dict[str, Any]) -> None:
    stats = target.setdefault(
        key,
        {
            **base,
            "trade_count": 0,
            "sum_return": 0.0,
            "sum_R": 0.0,
            "win_count": 0,
            "gain_sum": 0.0,
            "gain_count": 0,
            "loss_sum": 0.0,
            "loss_count": 0,
            "max_profit_sum": 0.0,
            "max_drawdown_sum": 0.0,
            "holding_days_sum": 0.0,
            "daily_return_sum": 0.0,
            "stop_loss_count": 0,
            "take_profit_count": 0,
            "trailing_stop_count": 0,
            "breakeven_stop_count": 0,
            "time_stop_count": 0,
        },
    )
    value = float(outcome["rule_return"])
    stats["trade_count"] += 1
    stats["sum_return"] += value
    holding_days = max(float(outcome.get("holding_days", math.nan)), 1.0)
    stats["daily_return_sum"] += value / holding_days
    stats["sum_R"] += float(outcome.get("rule_R", math.nan))
    stats["max_profit_sum"] += float(outcome.get("max_profit", math.nan))
    stats["max_drawdown_sum"] += float(outcome.get("max_drawdown", math.nan))
    stats["holding_days_sum"] += holding_days
    if value > 0:
        stats["win_count"] += 1
        stats["gain_sum"] += value
        stats["gain_count"] += 1
    elif value < 0:
        stats["loss_sum"] += value
        stats["loss_count"] += 1
    stats["stop_loss_count"] += int(bool(outcome.get("stop_loss_hit")))
    stats["take_profit_count"] += int(bool(outcome.get("take_profit_hit")))
    stats["trailing_stop_count"] += int(bool(outcome.get("trailing_stop_hit")))
    stats["breakeven_stop_count"] += int(bool(outcome.get("breakeven_stop_hit")))
    stats["time_stop_count"] += int(bool(outcome.get("time_stop_hit")))


def _stats_to_frame(stats: dict[tuple[Any, ...], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in stats.values():
        count = int(item["trade_count"])
        if count <= 0:
            continue
        avg_gain = item["gain_sum"] / item["gain_count"] if item["gain_count"] else math.nan
        avg_loss = item["loss_sum"] / item["loss_count"] if item["loss_count"] else math.nan
        payoff = avg_gain / abs(avg_loss) if math.isfinite(avg_gain) and math.isfinite(avg_loss) and avg_loss != 0 else math.nan
        row = {
            key: value
            for key, value in item.items()
            if not key.endswith("_sum") and (not key.endswith("_count") or key == "trade_count") and key not in {"win_count"}
        }
        row.update(
            {
                "avg_return": item["sum_return"] / count,
                "avg_daily_return": item["daily_return_sum"] / count,
                "win_rate": item["win_count"] / count,
                "avg_R": item["sum_R"] / count,
                "avg_gain": avg_gain,
                "avg_loss": avg_loss,
                "payoff_ratio": payoff,
                "avg_max_profit": item["max_profit_sum"] / count,
                "avg_max_drawdown": item["max_drawdown_sum"] / count,
                "avg_holding_days": item["holding_days_sum"] / count,
                "stop_loss_rate": item["stop_loss_count"] / count,
                "take_profit_rate": item["take_profit_count"] / count,
                "trailing_stop_rate": item["trailing_stop_count"] / count,
                "breakeven_stop_rate": item["breakeven_stop_count"] / count,
                "time_stop_rate": item["time_stop_count"] / count,
            }
        )
        clipped_payoff = min(float(payoff), 5.0) if math.isfinite(payoff) else 0.0
        horizon = max(float(row.get("horizon", 1)), 1.0)
        row["balanced_score"] = (
            float(row["avg_R"])
            + 0.3 * float(row["win_rate"])
            + 0.2 * clipped_payoff
            - 0.5 * abs(float(row["avg_max_drawdown"]))
            - 0.3 * float(row["stop_loss_rate"])
            - 0.05 * (float(row["avg_holding_days"]) / horizon)
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["strategy", "horizon", "balanced_score"], ascending=[True, True, False]).reset_index(drop=True)


def _build_selection_report(by_split: pd.DataFrame) -> pd.DataFrame:
    if by_split.empty or "split" not in by_split.columns:
        return pd.DataFrame()
    tune = by_split[by_split["split"].eq("tune")].copy()
    test = by_split[by_split["split"].eq("test")].copy()
    if tune.empty or test.empty:
        return pd.DataFrame()
    idx = tune.groupby(["strategy", "horizon"], dropna=False)["balanced_score"].idxmax()
    selected = tune.loc[idx].copy()
    join_columns = ["strategy", "horizon", "rule_id"]
    join_columns = [column for column in join_columns if column in selected.columns and column in test.columns]
    tune_keep = selected.loc[:, join_columns + ["balanced_score", "avg_return", "avg_daily_return", "win_rate", "payoff_ratio", "avg_R", "avg_max_drawdown", "trade_count"]].rename(
        columns={
            "balanced_score": "tune_balanced_score",
            "avg_return": "tune_avg_return",
            "avg_daily_return": "tune_avg_daily_return",
            "win_rate": "tune_win_rate",
            "payoff_ratio": "tune_payoff_ratio",
            "avg_R": "tune_avg_R",
            "avg_max_drawdown": "tune_avg_max_drawdown",
            "trade_count": "tune_trade_count",
        }
    )
    test_keep = test.loc[
        :,
        join_columns
        + [
            "balanced_score",
            "avg_return",
            "avg_daily_return",
            "win_rate",
            "payoff_ratio",
            "avg_R",
            "avg_max_drawdown",
            "avg_holding_days",
            "stop_loss_rate",
            "take_profit_rate",
            "trailing_stop_rate",
            "breakeven_stop_rate",
            "time_stop_rate",
            "trade_count",
        ],
    ].rename(
        columns={
            "balanced_score": "test_balanced_score",
            "avg_return": "test_avg_return",
            "avg_daily_return": "test_avg_daily_return",
            "win_rate": "test_win_rate",
            "payoff_ratio": "test_payoff_ratio",
            "avg_R": "test_avg_R",
            "avg_max_drawdown": "test_avg_max_drawdown",
            "avg_holding_days": "test_avg_holding_days",
            "stop_loss_rate": "test_stop_loss_rate",
            "take_profit_rate": "test_take_profit_rate",
            "trailing_stop_rate": "test_trailing_stop_rate",
            "breakeven_stop_rate": "test_breakeven_stop_rate",
            "time_stop_rate": "test_time_stop_rate",
            "trade_count": "test_trade_count",
        }
    )
    result = tune_keep.merge(test_keep, on=join_columns, how="left")
    return result.sort_values(["strategy", "horizon"]).reset_index(drop=True)


def _write_recommendations(path: Path, *, selection_report: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = ["# Recommended Exit Rules", ""]
    source = selection_report if not selection_report.empty else summary.groupby(["strategy", "horizon"], as_index=False).head(1)
    if source.empty:
        lines.append("No recommendation rows were produced.")
    else:
        lines.append("| strategy | horizon | rule | test/overall return | daily return | win rate | payoff | avg R | max drawdown |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in source.to_dict("records"):
            avg_return = row.get("test_avg_return", row.get("avg_return"))
            daily_return = row.get("test_avg_daily_return", row.get("avg_daily_return"))
            win_rate = row.get("test_win_rate", row.get("win_rate"))
            payoff = row.get("test_payoff_ratio", row.get("payoff_ratio"))
            avg_r = row.get("test_avg_R", row.get("avg_R"))
            drawdown = row.get("test_avg_max_drawdown", row.get("avg_max_drawdown"))
            lines.append(
                "| "
                f"{row.get('strategy')} | {int(row.get('horizon'))} | `{row.get('rule_id')}` | "
                f"{_format_pct(avg_return)} | {_format_pct(daily_return)} | {_format_pct(win_rate)} | {_format_number(payoff)} | "
                f"{_format_number(avg_r)} | {_format_pct(drawdown)} |"
            )
    lines.extend(
        [
            "",
            "Selection uses tune-period balanced_score when a tune/test split is available.",
            "Use this report as an exit-rule shortlist; the final trading rule should prefer stable rows across horizons and years.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _stats_key(base: dict[str, Any]) -> tuple[Any, ...]:
    return (
        base.get("strategy"),
        base.get("horizon"),
        base.get("rule_id"),
        base.get("year"),
        base.get("split"),
    )


def _split_name(signal_date: date, *, tune_end_date: date, test_start_date: date) -> str:
    if signal_date <= tune_end_date:
        return "tune"
    if signal_date >= test_start_date:
        return "test"
    return "gap"


def _pct_id(value: float) -> str:
    return str(int(round(float(value) * 100))).zfill(2)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_float(value: Any) -> float | None:
    number = _safe_float(value)
    return number


def _optional_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.2%}"


def _format_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
