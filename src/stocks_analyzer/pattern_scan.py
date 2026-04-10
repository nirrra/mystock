from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .indicators import add_indicators


@dataclass(slots=True)
class PatternScanConfig:
    min_old_high_gap_days: int = 50
    max_old_high_gap_days: int = 150
    min_drawdown_pct: float = 0.30
    near_high_threshold_pct: float = 0.05
    breakout_lookback_days: int = 15
    breakout_pullback_min_distance_pct: float = -0.15
    breakout_pullback_max_distance_pct: float = 0.30
    peak_window_days: int = 10
    volume_window_min_days: int = 50
    volume_window_max_days: int = 150
    volume_median_multiplier: float = 1.5


def scan_directory(data_dir: str | Path, config: PatternScanConfig | None = None) -> pd.DataFrame:
    scan_config = config or PatternScanConfig()
    base_dir = Path(data_dir)
    results: list[dict[str, object]] = []

    for path in sorted(base_dir.glob("*.parquet")):
        try:
            dataframe = pd.read_parquet(path)
            symbol = path.stem
            matches = analyze_symbol(dataframe, symbol, scan_config)
            for match in matches:
                match["data_path"] = str(path)
            results.extend(matches)
        except Exception as exc:
            results.append(
                {
                    "symbol": path.stem,
                    "pattern_type": "error",
                    "notes": str(exc),
                    "data_path": str(path),
                }
            )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def find_near_old_high_match(dataframe: pd.DataFrame, symbol: str, config: PatternScanConfig | None = None) -> dict[str, object] | None:
    scan_config = config or PatternScanConfig()
    df = _prepare_frame(dataframe)
    if len(df) < max(60, scan_config.min_old_high_gap_days + 1):
        return None

    latest = df.iloc[-1]
    if pd.isna(latest["ma_20"]) or pd.isna(latest["ma_60"]):
        return None

    future_min_low = _future_minimum_low(df["low"])
    candidates = _build_old_high_candidates(df, future_min_low, scan_config)
    if not candidates:
        return None

    return _find_near_high_match(df, symbol, candidates, scan_config)


def analyze_symbol(dataframe: pd.DataFrame, symbol: str, config: PatternScanConfig | None = None) -> list[dict[str, object]]:
    scan_config = config or PatternScanConfig()
    df = _prepare_frame(dataframe)
    if len(df) < max(60, scan_config.min_old_high_gap_days + 1):
        return []

    latest_idx = len(df) - 1
    latest = df.iloc[latest_idx]
    if pd.isna(latest["ma_20"]) or pd.isna(latest["ma_60"]):
        return []

    future_min_low = _future_minimum_low(df["low"])
    candidates = _build_old_high_candidates(df, future_min_low, scan_config)
    if not candidates:
        return []

    matches: list[dict[str, object]] = []

    near_match = _find_near_high_match(df, symbol, candidates, scan_config)
    if near_match is not None:
        matches.append(near_match)

    breakout_match = _find_breakout_pullback_match(df, symbol, candidates, scan_config)
    if breakout_match is not None:
        matches.append(breakout_match)

    return matches


def _prepare_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"ma_5", "ma_20", "ma_60"}
    if required.issubset(dataframe.columns):
        return dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    return add_indicators(dataframe).sort_values("trade_date").reset_index(drop=True)


def _future_minimum_low(lows: pd.Series) -> list[float]:
    values = lows.astype(float).tolist()
    result = [float("inf")] * len(values)
    running_min = float("inf")
    for index in range(len(values) - 1, -1, -1):
        result[index] = running_min
        running_min = min(running_min, values[index])
    return result


def _build_old_high_candidates(
    df: pd.DataFrame,
    future_min_low: list[float],
    config: PatternScanConfig,
) -> list[dict[str, object]]:
    latest_idx = len(df) - 1
    candidates: list[dict[str, object]] = []

    for index in range(0, latest_idx + 1):
        days_since_old_high = latest_idx - index
        if days_since_old_high < config.min_old_high_gap_days or days_since_old_high > config.max_old_high_gap_days:
            continue

        if not _is_local_peak(df, index, config.peak_window_days):
            continue

        if not _has_volume_surge(df, index, config):
            continue

        old_high = float(df.iloc[index]["high"])
        if old_high <= 0:
            continue

        subsequent_min_low = future_min_low[index]
        if subsequent_min_low == float("inf"):
            continue

        drawdown = (old_high - subsequent_min_low) / old_high
        if drawdown < config.min_drawdown_pct:
            continue

        candidates.append(
            {
                "index": index,
                "old_high_date": pd.Timestamp(df.iloc[index]["trade_date"]).date().isoformat(),
                "old_high_price": old_high,
                "days_since_old_high": days_since_old_high,
                "max_drawdown_since_old_high": drawdown,
            }
        )

    return candidates


def _is_local_peak(df: pd.DataFrame, index: int, peak_window_days: int) -> bool:
    left = index - peak_window_days
    right = index + peak_window_days
    if left < 0 or right >= len(df):
        return False

    window_highs = df.iloc[left : right + 1]["high"].astype(float)
    current_high = float(df.iloc[index]["high"])
    return current_high >= float(window_highs.max())


def _has_volume_surge(df: pd.DataFrame, index: int, config: PatternScanConfig) -> bool:
    start = index - config.volume_window_max_days
    end = index - config.volume_window_min_days
    if start < 0 or end < 0 or start > end:
        return False

    baseline_window = df.iloc[start : end + 1]["volume"].astype(float)
    baseline_median = float(baseline_window.median())
    if pd.isna(baseline_median) or baseline_median <= 0:
        return False

    current_volume = float(df.iloc[index]["volume"])
    return current_volume >= baseline_median * config.volume_median_multiplier


def _is_unbroken_until(df: pd.DataFrame, start_index: int, end_index: int) -> bool:
    if end_index <= start_index:
        return True

    candidate_high = float(df.iloc[start_index]["high"])
    later_highs = df.iloc[start_index + 1 : end_index + 1]["high"].astype(float)
    if later_highs.empty:
        return True
    return float(later_highs.max()) <= candidate_high


def _find_near_high_match(
    df: pd.DataFrame,
    symbol: str,
    candidates: list[dict[str, object]],
    config: PatternScanConfig,
) -> dict[str, object] | None:
    latest = df.iloc[-1]
    if not _is_ma_bullish(latest):
        return None

    close_price = float(latest["close"])
    best_match: dict[str, object] | None = None
    best_old_high = float("-inf")

    for candidate in candidates:
        old_high_price = float(candidate["old_high_price"])
        candidate_index = int(candidate["index"])

        if not _is_unbroken_until(df, candidate_index, len(df) - 1):
            continue

        if close_price > old_high_price:
            continue

        distance_pct = (old_high_price - close_price) / old_high_price
        if distance_pct > config.near_high_threshold_pct:
            continue

        if old_high_price > best_old_high:
            best_old_high = old_high_price
            best_match = _build_result_row(
                symbol=symbol,
                pattern_type="near_old_high",
                current_row=latest,
                candidate=candidate,
                distance_to_old_high_pct=distance_pct,
                breakout_date=None,
            )

    return best_match


def _find_breakout_pullback_match(
    df: pd.DataFrame,
    symbol: str,
    candidates: list[dict[str, object]],
    config: PatternScanConfig,
) -> dict[str, object] | None:
    latest = df.iloc[-1]
    if pd.isna(latest["ma_20"]) or float(latest["close"]) < float(latest["ma_20"]):
        return None

    start_idx = max(0, len(df) - config.breakout_lookback_days)
    breakout_window = df.iloc[start_idx:].reset_index(drop=False)

    best_match: dict[str, object] | None = None
    best_old_high = float("-inf")
    best_breakout_index = -1

    for breakout_row in breakout_window.itertuples(index=False):
        if not _is_ma_bullish(breakout_row):
            continue

        matching_candidates = [
            candidate
            for candidate in candidates
            if int(candidate["index"]) < breakout_row.index
            and float(candidate["old_high_price"]) < breakout_row.high
            and _is_unbroken_until(df, int(candidate["index"]), int(breakout_row.index) - 1)
        ]
        if not matching_candidates:
            continue

        candidate = max(matching_candidates, key=lambda item: float(item["old_high_price"]))
        old_high_price = float(candidate["old_high_price"])
        distance_pct = (float(latest["close"]) - old_high_price) / old_high_price
        if distance_pct < config.breakout_pullback_min_distance_pct:
            continue
        if distance_pct > config.breakout_pullback_max_distance_pct:
            continue

        if old_high_price > best_old_high or (old_high_price == best_old_high and breakout_row.index > best_breakout_index):
            best_old_high = old_high_price
            best_breakout_index = int(breakout_row.index)
            best_match = _build_result_row(
                symbol=symbol,
                pattern_type="breakout_pullback_watch",
                current_row=latest,
                candidate=candidate,
                distance_to_old_high_pct=distance_pct,
                breakout_date=pd.Timestamp(breakout_row.trade_date).date().isoformat(),
            )

    return best_match


def _build_result_row(
    symbol: str,
    pattern_type: str,
    current_row,
    candidate: dict[str, object],
    distance_to_old_high_pct: float,
    breakout_date: str | None,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "pattern_type": pattern_type,
        "current_date": pd.Timestamp(current_row["trade_date"]).date().isoformat()
        if isinstance(current_row, pd.Series)
        else pd.Timestamp(current_row.trade_date).date().isoformat(),
        "old_high_date": candidate["old_high_date"],
        "old_high_price": round(float(candidate["old_high_price"]), 4),
        "days_since_old_high": int(candidate["days_since_old_high"]),
        "max_drawdown_since_old_high": round(float(candidate["max_drawdown_since_old_high"]), 4),
        "current_close": round(float(current_row["close"]) if isinstance(current_row, pd.Series) else float(current_row.close), 4),
        "current_ma5": round(float(current_row["ma_5"]) if isinstance(current_row, pd.Series) else float(current_row.ma_5), 4),
        "current_ma20": round(float(current_row["ma_20"]) if isinstance(current_row, pd.Series) else float(current_row.ma_20), 4),
        "current_ma60": round(float(current_row["ma_60"]) if isinstance(current_row, pd.Series) else float(current_row.ma_60), 4),
        "distance_to_old_high_pct": round(distance_to_old_high_pct, 4),
        "breakout_date": breakout_date,
        "notes": "",
    }


def _is_ma_bullish(row) -> bool:
    ma5 = float(row["ma_5"]) if isinstance(row, pd.Series) else float(row.ma_5)
    ma20 = float(row["ma_20"]) if isinstance(row, pd.Series) else float(row.ma_20)
    ma60 = float(row["ma_60"]) if isinstance(row, pd.Series) else float(row.ma_60)
    if pd.isna(ma5) or pd.isna(ma20) or pd.isna(ma60):
        return False
    return ma5 > ma20 > ma60
