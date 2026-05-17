from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import logging
import math
import pandas as pd

from .phase_display import normalize_symbol
from .sector_membership import load_sector_membership, sector_performance_dir


SECTOR_PULLBACK_METRIC_COLUMNS = [
    "trade_date",
    "sector_type",
    "sector_name",
    "sector_label",
    "member_count",
    "latest_valid_count",
    "avg_valid_count",
    "valid_days",
    "latest_index",
    "buy_score",
    "buy_level",
    "pullback_depth_score",
    "pullback_timing_score",
    "stabilization_score",
    "rebound_confirmation_score",
    "risk_control_score",
    "buy_flags",
    "buy_reason",
    "return_1d",
    "return_3d",
    "return_5d",
    "excess_return_3d",
    "no_new_20d_low_in_5d",
    "no_new_60d_low_in_5d",
    "no_new_20d_low_in_10d",
    "close_above_ma5",
    "close_above_ma10",
    "close_above_ma20",
    "volatility_ratio_5d_20d",
    "distance_to_recent_low_pct",
    "recent_down_speed_5d",
    "down_amount_ratio_10d",
    "max_consecutive_up_days_1y",
    "max_consecutive_outperform_days_1y",
    "outperform_day_ratio_1y",
    "previous_peak_date",
    "previous_peak_index",
    "drawdown_from_peak_pct",
    "days_since_peak",
    "peak_confirmed",
    "recent_high",
    "peak_method",
    "ma5_slope_pct",
    "ma10_slope_pct",
    "ma20_slope_pct",
    "ma60_slope_pct",
]


@dataclass(frozen=True)
class SectorPullbackMetricsResult:
    trade_date: date | None
    output_path: Path
    row_count: int
    symbol_count: int
    sector_count: int


def sector_pullback_metrics_path(project_root: Path, trade_date: date) -> Path:
    return sector_performance_dir(project_root) / f"sector_pullback_metrics_{trade_date.isoformat()}.csv"


def analyze_sector_pullback_metrics(
    *,
    project_root: Path,
    trade_date: date | None = None,
    daily_dir: Path | None = None,
    history_days: int = 420,
    strength_lookback_days: int = 252,
    local_peak_window: int = 60,
    slope_lag_days: int = 5,
    min_members: int = 5,
    output: Path | None = None,
    progress: bool = False,
) -> SectorPullbackMetricsResult:
    if history_days <= 0:
        raise ValueError("history_days must be positive")
    if strength_lookback_days <= 0:
        raise ValueError("strength_lookback_days must be positive")
    if local_peak_window <= 0:
        raise ValueError("local_peak_window must be positive")
    if slope_lag_days <= 0:
        raise ValueError("slope_lag_days must be positive")
    if min_members <= 0:
        raise ValueError("min_members must be positive")

    daily_root = daily_dir if daily_dir is not None else project_root / "data" / "daily"
    members = _prepare_membership(project_root=project_root, min_members=min_members)
    if members.empty:
        resolved_output = output or sector_performance_dir(project_root) / "sector_pullback_metrics_empty.csv"
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=SECTOR_PULLBACK_METRIC_COLUMNS).to_csv(resolved_output, index=False, encoding="utf-8-sig")
        return SectorPullbackMetricsResult(
            trade_date=trade_date,
            output_path=resolved_output,
            row_count=0,
            symbol_count=0,
            sector_count=0,
        )

    sector_symbols = sorted(members["symbol"].unique())
    symbols = sorted(set(sector_symbols) | set(_available_daily_symbols(daily_root)))
    stock_returns = _load_stock_return_history(
        daily_root=daily_root,
        symbols=symbols,
        trade_date=trade_date,
        history_days=history_days,
        progress=progress,
    )
    if stock_returns.empty:
        resolved_output = output or sector_performance_dir(project_root) / "sector_pullback_metrics_empty.csv"
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=SECTOR_PULLBACK_METRIC_COLUMNS).to_csv(resolved_output, index=False, encoding="utf-8-sig")
        return SectorPullbackMetricsResult(
            trade_date=trade_date,
            output_path=resolved_output,
            row_count=0,
            symbol_count=0,
            sector_count=members["sector_key"].nunique(),
        )

    resolved_trade_date = stock_returns["trade_date"].max().date()
    all_dates = pd.Index(sorted(stock_returns["trade_date"].unique()))
    if len(all_dates) > history_days:
        all_dates = all_dates[-history_days:]
        stock_returns = stock_returns[stock_returns["trade_date"].isin(all_dates)].copy()

    benchmark_returns = stock_returns.groupby("trade_date", sort=True)["return_pct"].mean().reindex(all_dates).fillna(0.0)
    benchmark_index = (1.0 + benchmark_returns / 100.0).cumprod() * 100.0
    sector_returns = _build_sector_return_frame(stock_returns=stock_returns, members=members)
    sector_info = _build_sector_info(members)

    rows: list[dict[str, object]] = []
    grouped = sector_returns.groupby("sector_key", sort=True)
    for index, (sector_key, group) in enumerate(grouped, start=1):
        if progress and (index == 1 or index % 100 == 0 or index == len(grouped)):
            logging.info("Sector pullback metrics progress: %s/%s", index, len(grouped))
        info = sector_info.loc[sector_key]
        series = group.set_index("trade_date").sort_index()
        sector_ret = series["sector_return_pct"].reindex(all_dates).fillna(0.0)
        sector_amount = series["total_amount"].reindex(all_dates).fillna(0.0)
        valid_count = series["valid_count"].reindex(all_dates).fillna(0.0)
        sector_index = (1.0 + sector_ret / 100.0).cumprod() * 100.0
        if sector_index.empty:
            continue

        strength_dates = all_dates[-min(strength_lookback_days, len(all_dates)) :]
        strength_sector_ret = sector_ret.reindex(strength_dates).fillna(0.0)
        strength_benchmark_ret = benchmark_returns.reindex(strength_dates).fillna(0.0)
        outperform = strength_sector_ret > strength_benchmark_ret
        peak = _find_previous_peak(sector_index=sector_index, local_peak_window=local_peak_window)
        latest_index = float(sector_index.iloc[-1])
        previous_peak_index = float(sector_index.loc[peak.peak_date])
        drawdown = (latest_index / previous_peak_index - 1.0) * 100.0 if previous_peak_index else math.nan
        latest_position = len(sector_index) - 1
        peak_position = int(sector_index.index.get_loc(peak.peak_date))
        buy_metrics = _build_buy_metrics(
            sector_type=str(info["sector_type"]),
            sector_index=sector_index,
            sector_ret=sector_ret,
            sector_amount=sector_amount,
            benchmark_index=benchmark_index,
            drawdown_from_peak_pct=drawdown,
            days_since_peak=latest_position - peak_position,
        )

        rows.append(
            {
                "trade_date": resolved_trade_date.isoformat(),
                "sector_type": info["sector_type"],
                "sector_name": info["sector_name"],
                "sector_label": info["sector_label"],
                "member_count": int(info["member_count"]),
                "latest_valid_count": int(valid_count.iloc[-1]) if not valid_count.empty else 0,
                "avg_valid_count": _round_or_na(float(valid_count.mean()), 2),
                "valid_days": int((valid_count > 0).sum()),
                "latest_index": _round_or_na(latest_index, 4),
                **buy_metrics,
                "max_consecutive_up_days_1y": _max_consecutive_true(strength_sector_ret > 0),
                "max_consecutive_outperform_days_1y": _max_consecutive_true(outperform),
                "outperform_day_ratio_1y": _round_or_na(float(outperform.mean()), 6) if len(outperform) else pd.NA,
                "previous_peak_date": peak.peak_date.date().isoformat(),
                "previous_peak_index": _round_or_na(previous_peak_index, 4),
                "drawdown_from_peak_pct": _round_or_na(drawdown, 4),
                "days_since_peak": latest_position - peak_position,
                "peak_confirmed": peak.confirmed,
                "recent_high": peak.recent_high,
                "peak_method": peak.method,
                "ma5_slope_pct": _moving_average_slope(sector_index, window=5, lag=slope_lag_days),
                "ma10_slope_pct": _moving_average_slope(sector_index, window=10, lag=slope_lag_days),
                "ma20_slope_pct": _moving_average_slope(sector_index, window=20, lag=slope_lag_days),
                "ma60_slope_pct": _moving_average_slope(sector_index, window=60, lag=slope_lag_days),
            }
        )

    result = pd.DataFrame(rows, columns=SECTOR_PULLBACK_METRIC_COLUMNS)
    if not result.empty:
        result = result.sort_values(["sector_type", "sector_name"], kind="stable").reset_index(drop=True)
    resolved_output = output or sector_pullback_metrics_path(project_root, resolved_trade_date)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(resolved_output, index=False, encoding="utf-8-sig")
    return SectorPullbackMetricsResult(
        trade_date=resolved_trade_date,
        output_path=resolved_output,
        row_count=len(result),
        symbol_count=stock_returns["symbol"].nunique(),
        sector_count=members["sector_key"].nunique(),
    )


def _prepare_membership(*, project_root: Path, min_members: int) -> pd.DataFrame:
    members = load_sector_membership(project_root=project_root)
    if members.empty:
        return pd.DataFrame(columns=["symbol", "sector_key", "sector_type", "sector_name", "sector_label"])
    members = members.copy()
    members["symbol"] = members["symbol"].map(normalize_symbol)
    members = members[members["symbol"].astype(str).str.len().eq(6)].copy()
    members["sector_type"] = members["sector_type"].astype(str).str.strip()
    members["sector_name"] = members["sector_name"].astype(str).str.strip()
    members["sector_label"] = members["sector_label"].astype(str).str.strip()
    members = members[
        members["sector_type"].isin(["industry", "concept"])
        & members["sector_name"].ne("")
        & members["sector_label"].ne("")
    ].copy()
    if members.empty:
        return pd.DataFrame(columns=["symbol", "sector_key", "sector_type", "sector_name", "sector_label"])
    members["sector_key"] = members["sector_type"] + "\x1f" + members["sector_label"]
    members = members.drop_duplicates(["symbol", "sector_key"], keep="first")
    counts = members.groupby("sector_key")["symbol"].nunique()
    keep_keys = counts[counts >= min_members].index
    members = members[members["sector_key"].isin(keep_keys)].copy()
    return members.loc[:, ["symbol", "sector_key", "sector_type", "sector_name", "sector_label"]]


def _load_stock_return_history(
    *,
    daily_root: Path,
    symbols: list[str],
    trade_date: date | None,
    history_days: int,
    progress: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    read_days = history_days + 5
    for index, symbol in enumerate(symbols, start=1):
        if progress and (index == 1 or index % 500 == 0 or index == len(symbols)):
            logging.info("Sector daily history load progress: %s/%s", index, len(symbols))
        frame = _read_symbol_daily_returns(daily_root=daily_root, symbol=symbol, trade_date=trade_date, read_days=read_days)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["trade_date", "return_pct"])
    result = result.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return result


def _available_daily_symbols(daily_root: Path) -> list[str]:
    if not daily_root.exists():
        return []
    symbols: list[str] = []
    for path in daily_root.glob("*.parquet"):
        symbol = normalize_symbol(path.stem)
        if symbol and len(symbol) == 6:
            symbols.append(symbol)
    return symbols


def _read_symbol_daily_returns(*, daily_root: Path, symbol: str, trade_date: date | None, read_days: int) -> pd.DataFrame:
    target = daily_root / f"{normalize_symbol(symbol)}.parquet"
    if not target.exists():
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    try:
        frame = pd.read_parquet(target, columns=["trade_date", "close", "amount", "pct_change"])
    except Exception:
        try:
            frame = pd.read_parquet(target)
        except Exception as exc:
            logging.warning("Failed to read daily bars for sector metrics %s: %s", symbol, exc)
            return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    if frame.empty or "trade_date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    if "amount" in data.columns:
        data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        data["amount"] = 0.0
    data = data.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    if trade_date is not None:
        data = data[data["trade_date"].dt.date <= trade_date].copy()
    if data.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "return_pct", "close", "amount"])
    if len(data) > read_days:
        data = data.tail(read_days).copy()
    if "pct_change" in data.columns:
        return_pct = pd.to_numeric(data["pct_change"], errors="coerce")
    else:
        return_pct = pd.Series(pd.NA, index=data.index, dtype="Float64")
    computed_pct = data["close"].pct_change() * 100.0
    data["return_pct"] = return_pct.fillna(computed_pct)
    data["symbol"] = normalize_symbol(symbol)
    data = data.dropna(subset=["return_pct"])
    return data.loc[:, ["symbol", "trade_date", "return_pct", "close", "amount"]]


def _build_sector_return_frame(*, stock_returns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    sector_symbols = members.loc[:, ["symbol", "sector_key"]].copy()
    sector_symbols["sector_key"] = sector_symbols["sector_key"].astype("category")
    merged = stock_returns.merge(sector_symbols, on="symbol", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["sector_key", "trade_date", "sector_return_pct", "valid_count", "total_amount"])
    grouped = (
        merged.groupby(["sector_key", "trade_date"], observed=True, sort=True)
        .agg(sector_return_pct=("return_pct", "mean"), valid_count=("return_pct", "count"), total_amount=("amount", "sum"))
        .reset_index()
    )
    grouped["sector_key"] = grouped["sector_key"].astype(str)
    return grouped


def _build_sector_info(members: pd.DataFrame) -> pd.DataFrame:
    info = (
        members.groupby("sector_key", sort=True)
        .agg(
            sector_type=("sector_type", "first"),
            sector_name=("sector_name", "first"),
            sector_label=("sector_label", "first"),
            member_count=("symbol", "nunique"),
        )
        .sort_values(["sector_type", "sector_name"])
    )
    return info


@dataclass(frozen=True)
class _PeakInfo:
    peak_date: pd.Timestamp
    confirmed: bool
    recent_high: bool
    method: str


def _find_previous_peak(*, sector_index: pd.Series, local_peak_window: int) -> _PeakInfo:
    clean = sector_index.dropna()
    if clean.empty:
        raise ValueError("sector_index is empty")
    latest_date = clean.index[-1]
    latest_value = float(clean.iloc[-1])
    confirmed_peaks = pd.Series(dtype="float64")
    window = local_peak_window * 2 + 1
    if len(clean) >= window:
        centered_max = clean.rolling(window=window, center=True, min_periods=window).max()
        peak_mask = clean.eq(centered_max) & centered_max.notna()
        confirmed_peaks = clean.loc[peak_mask]

    if not confirmed_peaks.empty:
        usable = confirmed_peaks[confirmed_peaks >= latest_value]
        if not usable.empty:
            peak_date = usable.index[-1]
            days_since_peak = int(clean.index.get_loc(latest_date)) - int(clean.index.get_loc(peak_date))
            return _PeakInfo(
                peak_date=peak_date,
                confirmed=True,
                recent_high=days_since_peak < local_peak_window,
                method="confirmed_local_high",
            )

    peak_date = clean.idxmax()
    days_since_peak = int(clean.index.get_loc(latest_date)) - int(clean.index.get_loc(peak_date))
    return _PeakInfo(
        peak_date=peak_date,
        confirmed=False,
        recent_high=days_since_peak < local_peak_window,
        method="fallback_window_high",
    )


def _build_buy_metrics(
    *,
    sector_type: str,
    sector_index: pd.Series,
    sector_ret: pd.Series,
    sector_amount: pd.Series,
    benchmark_index: pd.Series,
    drawdown_from_peak_pct: float,
    days_since_peak: int,
) -> dict[str, object]:
    latest_index = float(sector_index.iloc[-1])
    return_1d = _window_return(sector_index, 1)
    return_3d = _window_return(sector_index, 3)
    return_5d = _window_return(sector_index, 5)
    benchmark_return_3d = _window_return(benchmark_index, 3)
    excess_return_3d = (
        float(return_3d) - float(benchmark_return_3d)
        if pd.notna(return_3d) and pd.notna(benchmark_return_3d)
        else pd.NA
    )
    no_new_20d_low_in_5d = _no_new_stage_low_in_recent_days(sector_index, recent_days=5, low_window_days=20)
    no_new_60d_low_in_5d = _no_new_stage_low_in_recent_days(sector_index, recent_days=5, low_window_days=60)
    no_new_20d_low_in_10d = _no_new_stage_low_in_recent_days(sector_index, recent_days=10, low_window_days=20)
    close_above_ma5 = _close_above_ma(sector_index, 5)
    close_above_ma10 = _close_above_ma(sector_index, 10)
    close_above_ma20 = _close_above_ma(sector_index, 20)
    volatility_ratio = _volatility_ratio(sector_ret, recent_days=5, base_days=20)
    recent_low_20 = sector_index.tail(20).min()
    distance_to_recent_low = (
        (latest_index / float(recent_low_20) - 1.0) * 100.0
        if pd.notna(recent_low_20) and float(recent_low_20) > 0
        else pd.NA
    )
    down_amount_ratio = _down_amount_ratio(sector_ret=sector_ret, sector_amount=sector_amount, days=10)

    pullback_depth_score = _score_pullback_depth(sector_type=sector_type, drawdown_from_peak_pct=drawdown_from_peak_pct)
    pullback_timing_score = _score_pullback_timing(days_since_peak)
    stabilization_score = _score_stabilization(
        no_new_20d_low_in_5d=no_new_20d_low_in_5d,
        no_new_60d_low_in_5d=no_new_60d_low_in_5d,
        no_new_20d_low_in_10d=no_new_20d_low_in_10d,
        ma5_slope=_moving_average_slope_raw(sector_index, window=5, lag=5),
        ma10_slope=_moving_average_slope_raw(sector_index, window=10, lag=5),
        ma20_slope=_moving_average_slope_raw(sector_index, window=20, lag=5),
        ma60_slope=_moving_average_slope_raw(sector_index, window=60, lag=5),
        close_above_ma5=close_above_ma5,
        close_above_ma10=close_above_ma10,
        close_above_ma20=close_above_ma20,
        volatility_ratio=volatility_ratio,
    )
    rebound_confirmation_score = _score_rebound_confirmation(
        return_3d=return_3d,
        return_5d=return_5d,
        excess_return_3d=excess_return_3d,
        up_ratio_3d=_recent_up_ratio(sector_ret, 3),
    )
    risk_control_score = _score_risk_control(
        distance_to_recent_low=distance_to_recent_low,
        return_5d=return_5d,
        down_amount_ratio=down_amount_ratio,
        ma20_slope=_moving_average_slope_raw(sector_index, window=20, lag=5),
    )
    raw_buy_score = (
        0.15 * pullback_depth_score
        + 0.10 * pullback_timing_score
        + 0.40 * stabilization_score
        + 0.20 * rebound_confirmation_score
        + 0.15 * risk_control_score
    )
    capped_buy_score, flags = _apply_buy_score_caps(
        score=raw_buy_score,
        no_new_20d_low_in_5d=no_new_20d_low_in_5d,
        no_new_60d_low_in_5d=no_new_60d_low_in_5d,
        drawdown_from_peak_pct=drawdown_from_peak_pct,
        return_3d=return_3d,
        return_5d=return_5d,
        volatility_ratio=volatility_ratio,
        down_amount_ratio=down_amount_ratio,
        ma5_slope=_moving_average_slope_raw(sector_index, window=5, lag=5),
        ma10_slope=_moving_average_slope_raw(sector_index, window=10, lag=5),
        ma20_slope=_moving_average_slope_raw(sector_index, window=20, lag=5),
    )
    buy_level = _buy_level(capped_buy_score)
    return {
        "buy_score": _round_or_na(capped_buy_score, 2),
        "buy_level": buy_level,
        "pullback_depth_score": _round_or_na(pullback_depth_score, 2),
        "pullback_timing_score": _round_or_na(pullback_timing_score, 2),
        "stabilization_score": _round_or_na(stabilization_score, 2),
        "rebound_confirmation_score": _round_or_na(rebound_confirmation_score, 2),
        "risk_control_score": _round_or_na(risk_control_score, 2),
        "buy_flags": "/".join(flags),
        "buy_reason": _buy_reason(buy_level=buy_level, flags=flags),
        "return_1d": _round_or_na(return_1d, 4),
        "return_3d": _round_or_na(return_3d, 4),
        "return_5d": _round_or_na(return_5d, 4),
        "excess_return_3d": _round_or_na(excess_return_3d, 4),
        "no_new_20d_low_in_5d": no_new_20d_low_in_5d,
        "no_new_60d_low_in_5d": no_new_60d_low_in_5d,
        "no_new_20d_low_in_10d": no_new_20d_low_in_10d,
        "close_above_ma5": close_above_ma5,
        "close_above_ma10": close_above_ma10,
        "close_above_ma20": close_above_ma20,
        "volatility_ratio_5d_20d": _round_or_na(volatility_ratio, 4),
        "distance_to_recent_low_pct": _round_or_na(distance_to_recent_low, 4),
        "recent_down_speed_5d": _round_or_na(return_5d, 4),
        "down_amount_ratio_10d": _round_or_na(down_amount_ratio, 4),
    }


def _window_return(series: pd.Series, days: int) -> object:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= days:
        return pd.NA
    previous = clean.iloc[-1 - days]
    latest = clean.iloc[-1]
    if pd.isna(previous) or pd.isna(latest) or float(previous) == 0:
        return pd.NA
    return (float(latest) / float(previous) - 1.0) * 100.0


def _no_new_stage_low_in_recent_days(series: pd.Series, *, recent_days: int, low_window_days: int) -> object:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < max(recent_days, low_window_days):
        return pd.NA
    rolling_low = clean.rolling(window=low_window_days, min_periods=low_window_days).min()
    recent = clean.tail(recent_days)
    recent_rolling_low = rolling_low.reindex(recent.index)
    new_low = recent <= recent_rolling_low * 1.002
    return not bool(new_low.fillna(False).any())


def _close_above_ma(series: pd.Series, window: int) -> object:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < window:
        return pd.NA
    moving_average = clean.rolling(window=window, min_periods=window).mean()
    latest_ma = moving_average.iloc[-1]
    if pd.isna(latest_ma):
        return pd.NA
    return bool(float(clean.iloc[-1]) > float(latest_ma))


def _volatility_ratio(series: pd.Series, *, recent_days: int, base_days: int) -> object:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < base_days:
        return pd.NA
    recent = clean.tail(recent_days).std()
    base = clean.tail(base_days).std()
    if pd.isna(recent) or pd.isna(base) or float(base) == 0:
        return pd.NA
    return float(recent / base)


def _down_amount_ratio(*, sector_ret: pd.Series, sector_amount: pd.Series, days: int) -> object:
    ret = pd.to_numeric(sector_ret.tail(days), errors="coerce")
    amount = pd.to_numeric(sector_amount.tail(days), errors="coerce")
    if ret.empty or amount.dropna().empty:
        return pd.NA
    base = amount.mean()
    if pd.isna(base) or float(base) == 0:
        return pd.NA
    down_amount = amount[ret < 0].mean()
    if pd.isna(down_amount):
        return 0.0
    return float(down_amount / base)


def _recent_up_ratio(series: pd.Series, days: int) -> object:
    clean = pd.to_numeric(series.tail(days), errors="coerce").dropna()
    if clean.empty:
        return pd.NA
    return float((clean > 0).mean())


def _score_pullback_depth(*, sector_type: str, drawdown_from_peak_pct: float) -> float:
    if pd.isna(drawdown_from_peak_pct):
        return 50.0
    depth = max(0.0, -float(drawdown_from_peak_pct))
    if sector_type == "industry":
        return _ideal_range_score(depth, low=5.0, high=15.0, deep=30.0)
    return _ideal_range_score(depth, low=8.0, high=25.0, deep=45.0)


def _ideal_range_score(value: float, *, low: float, high: float, deep: float) -> float:
    if value <= 0:
        return 20.0
    if value < low:
        return _linear_score(value, 0.0, low, 30.0, 85.0)
    if value <= high:
        return 100.0
    if value <= deep:
        return _linear_score(value, high, deep, 70.0, 35.0)
    return 25.0


def _score_pullback_timing(days_since_peak: int) -> float:
    days = max(0, int(days_since_peak))
    if days <= 3:
        return _linear_score(days, 0.0, 3.0, 20.0, 50.0)
    if days <= 45:
        return 100.0
    if days <= 90:
        return _linear_score(days, 46.0, 90.0, 75.0, 45.0)
    return 35.0


def _score_stabilization(
    *,
    no_new_20d_low_in_5d: object,
    no_new_60d_low_in_5d: object,
    no_new_20d_low_in_10d: object,
    ma5_slope: object,
    ma10_slope: object,
    ma20_slope: object,
    ma60_slope: object,
    close_above_ma5: object,
    close_above_ma10: object,
    close_above_ma20: object,
    volatility_ratio: object,
) -> float:
    no_new_low_score = 0.0
    no_new_low_score += 45.0 if _is_true(no_new_20d_low_in_5d) else 0.0
    no_new_low_score += 35.0 if _is_true(no_new_60d_low_in_5d) else 0.0
    no_new_low_score += 20.0 if _is_true(no_new_20d_low_in_10d) else 0.0
    slope_score = _weighted_average(
        [
            (_score_slope_stability(ma5_slope), 0.30),
            (_score_slope_stability(ma10_slope), 0.30),
            (_score_slope_stability(ma20_slope), 0.25),
            (_score_slope_stability(ma60_slope), 0.15),
        ]
    )
    reclaim_score = 0.0
    reclaim_score += 40.0 if _is_true(close_above_ma5) else 0.0
    reclaim_score += 30.0 if _is_true(close_above_ma10) else 0.0
    reclaim_score += 30.0 if _is_true(close_above_ma20) else 0.0
    volatility_score = _score_volatility_contraction(volatility_ratio)
    return _weighted_average(
        [
            (no_new_low_score, 0.25),
            (slope_score, 0.35),
            (reclaim_score, 0.20),
            (volatility_score, 0.20),
        ]
    )


def _score_rebound_confirmation(*, return_3d: object, return_5d: object, excess_return_3d: object, up_ratio_3d: object) -> float:
    return _weighted_average(
        [
            (_score_moderate_rebound(return_3d), 0.30),
            (_score_moderate_rebound(return_5d), 0.25),
            (_score_excess_rebound(excess_return_3d), 0.25),
            (_score_ratio(up_ratio_3d), 0.20),
        ]
    )


def _score_risk_control(*, distance_to_recent_low: object, return_5d: object, down_amount_ratio: object, ma20_slope: object) -> float:
    return _weighted_average(
        [
            (_score_distance_to_low(distance_to_recent_low), 0.35),
            (_score_recent_down_speed(return_5d), 0.25),
            (_score_down_amount(down_amount_ratio), 0.20),
            (_score_ma20_not_accelerating_down(ma20_slope), 0.20),
        ]
    )


def _is_true(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _apply_buy_score_caps(
    *,
    score: float,
    no_new_20d_low_in_5d: object,
    no_new_60d_low_in_5d: object,
    drawdown_from_peak_pct: object,
    return_3d: object,
    return_5d: object,
    volatility_ratio: object,
    down_amount_ratio: object,
    ma5_slope: object,
    ma10_slope: object,
    ma20_slope: object,
) -> tuple[float, list[str]]:
    flags: list[str] = []
    capped = float(score)
    if pd.notna(no_new_20d_low_in_5d) and not bool(no_new_20d_low_in_5d):
        capped = min(capped, 55.0)
        flags.append("近5日创20日新低")
    if pd.notna(no_new_60d_low_in_5d) and not bool(no_new_60d_low_in_5d):
        capped = min(capped, 50.0)
        flags.append("近5日创60日新低")
    slopes = [ma5_slope, ma10_slope, ma20_slope]
    if all(pd.notna(value) and float(value) < -0.5 for value in slopes):
        capped = min(capped, 50.0)
        flags.append("短中均线明显下行")
    if pd.notna(return_3d) and float(return_3d) > 10.0:
        capped = min(capped, 70.0)
        flags.append("反抽偏高")
    if pd.notna(drawdown_from_peak_pct) and float(drawdown_from_peak_pct) > -5.0 and pd.notna(return_5d) and float(return_5d) > 5.0:
        capped = min(capped, 60.0)
        flags.append("回调过浅且短涨")
    if (
        pd.notna(volatility_ratio)
        and float(volatility_ratio) > 1.3
        and pd.notna(down_amount_ratio)
        and float(down_amount_ratio) > 1.1
    ):
        capped = min(capped, 55.0)
        flags.append("放量下跌波动放大")
    return capped, flags


def _buy_level(score: float) -> str:
    if score >= 80:
        return "重点观察/布局"
    if score >= 65:
        return "可以试仓"
    if score >= 50:
        return "观察等确认"
    if score >= 35:
        return "有回调未企稳"
    return "暂不适合买"


def _buy_reason(*, buy_level: str, flags: list[str]) -> str:
    if flags:
        return f"{buy_level}；风险：{'/'.join(flags)}"
    return buy_level


def _score_slope_stability(value: object) -> float:
    if pd.isna(value):
        return 50.0
    slope = float(value)
    if slope >= 0:
        return 100.0
    if slope >= -0.5:
        return _linear_score(slope, -0.5, 0.0, 80.0, 100.0)
    if slope >= -1.0:
        return _linear_score(slope, -1.0, -0.5, 60.0, 80.0)
    if slope >= -2.0:
        return _linear_score(slope, -2.0, -1.0, 30.0, 60.0)
    return 20.0


def _score_volatility_contraction(value: object) -> float:
    if pd.isna(value):
        return 50.0
    ratio = float(value)
    if ratio <= 0.7:
        return 100.0
    if ratio <= 1.0:
        return _linear_score(ratio, 0.7, 1.0, 100.0, 70.0)
    if ratio <= 1.3:
        return _linear_score(ratio, 1.0, 1.3, 70.0, 40.0)
    return 25.0


def _score_moderate_rebound(value: object) -> float:
    if pd.isna(value):
        return 50.0
    ret = float(value)
    if ret < -3.0:
        return 20.0
    if ret < 0.0:
        return _linear_score(ret, -3.0, 0.0, 40.0, 70.0)
    if ret <= 5.0:
        return _linear_score(ret, 0.0, 5.0, 80.0, 100.0)
    if ret <= 10.0:
        return _linear_score(ret, 5.0, 10.0, 80.0, 60.0)
    return 45.0


def _score_excess_rebound(value: object) -> float:
    if pd.isna(value):
        return 50.0
    ret = float(value)
    if ret < -3.0:
        return 25.0
    if ret < 0.0:
        return _linear_score(ret, -3.0, 0.0, 45.0, 70.0)
    if ret <= 5.0:
        return _linear_score(ret, 0.0, 5.0, 80.0, 100.0)
    return 70.0


def _score_ratio(value: object) -> float:
    if pd.isna(value):
        return 50.0
    return max(0.0, min(100.0, float(value) * 100.0))


def _score_distance_to_low(value: object) -> float:
    if pd.isna(value):
        return 50.0
    distance = max(0.0, float(value))
    if distance < 3.0:
        return _linear_score(distance, 0.0, 3.0, 45.0, 90.0)
    if distance <= 10.0:
        return 100.0
    if distance <= 20.0:
        return _linear_score(distance, 10.0, 20.0, 80.0, 50.0)
    return 35.0


def _score_recent_down_speed(value: object) -> float:
    if pd.isna(value):
        return 50.0
    ret = float(value)
    if ret >= -1.0:
        return 100.0
    if ret >= -5.0:
        return _linear_score(ret, -5.0, -1.0, 40.0, 80.0)
    if ret >= -8.0:
        return _linear_score(ret, -8.0, -5.0, 20.0, 40.0)
    return 15.0


def _score_down_amount(value: object) -> float:
    if pd.isna(value):
        return 50.0
    ratio = float(value)
    if ratio <= 0.8:
        return 100.0
    if ratio <= 1.2:
        return _linear_score(ratio, 0.8, 1.2, 80.0, 50.0)
    if ratio <= 1.5:
        return _linear_score(ratio, 1.2, 1.5, 50.0, 25.0)
    return 20.0


def _score_ma20_not_accelerating_down(value: object) -> float:
    return _score_slope_stability(value)


def _weighted_average(items: list[tuple[float, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if pd.isna(value):
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator == 0:
        return 50.0
    return numerator / denominator


def _linear_score(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y1
    ratio = (float(value) - x0) / (x1 - x0)
    ratio = max(0.0, min(1.0, ratio))
    return y0 + ratio * (y1 - y0)


def _moving_average_slope_raw(series: pd.Series, *, window: int, lag: int) -> object:
    if len(series) < window + lag:
        return pd.NA
    moving_average = series.rolling(window=window, min_periods=window).mean()
    latest = moving_average.iloc[-1]
    previous = moving_average.iloc[-1 - lag]
    if pd.isna(latest) or pd.isna(previous) or float(previous) == 0:
        return pd.NA
    return (float(latest) / float(previous) - 1.0) * 100.0


def _moving_average_slope(series: pd.Series, *, window: int, lag: int) -> object:
    if len(series) < window + lag:
        return pd.NA
    moving_average = series.rolling(window=window, min_periods=window).mean()
    latest = moving_average.iloc[-1]
    previous = moving_average.iloc[-1 - lag]
    if pd.isna(latest) or pd.isna(previous) or float(previous) == 0:
        return pd.NA
    return _round_or_na((float(latest) / float(previous) - 1.0) * 100.0, 4)


def _max_consecutive_true(values: pd.Series) -> int:
    current = 0
    best = 0
    for value in values.fillna(False).astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _round_or_na(value: float, digits: int) -> object:
    if pd.isna(value):
        return pd.NA
    if not math.isfinite(float(value)):
        return pd.NA
    return round(float(value), digits)
