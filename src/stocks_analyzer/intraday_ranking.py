from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from .data_sources import create_data_provider
from .watchlist import PATTERN_PRIORITY, _stable_score


NEUTRAL_INTRADAY_SCORE = 50.0
INTRADAY_PERIOD = "5"
DEFAULT_EVENT_RESULT = {
    "intraday_5m_score": NEUTRAL_INTRADAY_SCORE,
    "intraday_volume_divergence_hit": False,
    "intraday_volume_divergence_type": "none",
    "intraday_volume_score": 0.0,
    "intraday_macd_divergence_hit": False,
    "intraday_macd_divergence_type": "none",
    "intraday_macd_divergence_score": 0.0,
    "intraday_macd_cross_hit": False,
    "intraday_macd_cross_type": "none",
    "intraday_macd_cross_score": 0.0,
    "intraday_ma_event_hit": False,
    "intraday_ma_event_type": "none",
    "intraday_ma_score": 0.0,
}
CSV_HEADER_MAP = {
    "rank": "排名",
    "symbol": "代码",
    "name": "名称",
    "pattern_ids": "形态",
    "tradingview_avg_all_rating_5d": "TradingView五日均分",
    "tradingview_all_rating": "TradingView当日分数",
    "tradingview_all_rating_label": "TradingView评级",
    "daily_macd_top_divergence_15d": "日线顶背离",
    "daily_macd_bottom_divergence_15d": "日线底背离",
    "daily_score": "日线分数",
    "intraday_5m_score": "5分钟分数",
    "intraday_volume_divergence_hit": "量价背离命中",
    "intraday_volume_divergence_type": "量价背离类型",
    "intraday_volume_score": "量价背离分",
    "intraday_macd_divergence_hit": "5分钟MACD背离命中",
    "intraday_macd_divergence_type": "5分钟MACD背离类型",
    "intraday_macd_divergence_score": "5分钟MACD背离分",
    "intraday_macd_cross_hit": "5分钟金叉死叉命中",
    "intraday_macd_cross_type": "5分钟金叉死叉类型",
    "intraday_macd_cross_score": "5分钟金叉死叉分",
    "intraday_ma_event_hit": "均线事件命中",
    "intraday_ma_event_type": "均线事件类型",
    "intraday_ma_score": "均线事件分",
}


def save_intraday_rankings(
    *,
    trade_date: date,
    intraday_provider: str,
    adjust: str,
    watchlist_payload: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    result = build_intraday_rankings(
        trade_date=trade_date,
        intraday_provider=intraday_provider,
        adjust=adjust,
        watchlist_payload=watchlist_payload,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_frame = localize_ranking_columns(result["ranking"])
    export_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "output_path": output_path,
        "ranking": result["ranking"],
        "processed_count": len(result["ranking"]),
        "failed_symbols": result["failed_symbols"],
        "failed_count": len(result["failed_symbols"]),
    }


def build_intraday_rankings(
    *,
    trade_date: date,
    intraday_provider: str,
    adjust: str,
    watchlist_payload: dict[str, object],
) -> dict[str, object]:
    summary_frame = build_watchlist_summary_frame(watchlist_payload)
    symbols = summary_frame["symbol"].astype(str).tolist() if not summary_frame.empty else []

    provider = create_data_provider(intraday_provider)
    try:
        batch_fetcher = getattr(provider, "get_intraday_bars_batch", None)
        if callable(batch_fetcher):
            batch_result, failed_symbols = batch_fetcher(
                symbols=symbols,
                start_datetime=trade_date.isoformat(),
                end_datetime=trade_date.isoformat(),
                period=INTRADAY_PERIOD,
                adjust=adjust,
            )
            available_symbols = [symbol for symbol in symbols if str(symbol).zfill(6) in batch_result]
            ranking, additional_failures = build_intraday_ranking_frame(
                symbols=available_symbols,
                daily_frame=summary_frame,
                intraday_fetcher=lambda symbol: batch_result[str(symbol).zfill(6)],
            )
            failed_symbols = failed_symbols + additional_failures
        else:
            ranking, failed_symbols = build_intraday_ranking_frame(
                symbols=symbols,
                daily_frame=summary_frame,
                intraday_fetcher=lambda symbol: provider.get_intraday_bars(
                    symbol=symbol,
                    start_datetime=trade_date.isoformat(),
                    end_datetime=trade_date.isoformat(),
                    period=INTRADAY_PERIOD,
                    adjust=adjust,
                ),
            )
    finally:
        provider.close()
    return {
        "ranking": ranking,
        "failed_symbols": failed_symbols,
    }


def build_intraday_ranking_frame(
    *,
    symbols: list[str],
    daily_frame: pd.DataFrame,
    intraday_fetcher: Callable[[str], pd.DataFrame],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if daily_frame.empty and not symbols:
        return pd.DataFrame(), []

    indexed_daily = daily_frame.set_index("symbol", drop=False) if not daily_frame.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    failed_symbols: list[dict[str, str]] = []
    for symbol in symbols:
        row = indexed_daily.loc[symbol].to_dict() if symbol in getattr(indexed_daily, "index", []) else _default_daily_row(symbol)
        try:
            intraday_frame = intraday_fetcher(symbol)
        except Exception as exc:
            logging.warning("Failed to fetch intraday bars for %s: %s", symbol, exc)
            failed_symbols.append(
                {
                    "symbol": str(symbol).zfill(6),
                    "name": str(row.get("name") or ""),
                    "error": str(exc),
                }
            )
            continue
        event_summary = summarize_intraday_events(intraday_frame)
        row.update(event_summary)
        rows.append(row)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking, failed_symbols

    ranking["intraday_5m_score"] = pd.to_numeric(ranking["intraday_5m_score"], errors="coerce").fillna(
        NEUTRAL_INTRADAY_SCORE
    )
    ranking = ranking.sort_values(
        ["intraday_5m_score", "symbol"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking, failed_symbols


def summarize_intraday_events(dataframe: pd.DataFrame) -> dict[str, object]:
    if dataframe.empty:
        return DEFAULT_EVENT_RESULT.copy()

    required = {"timestamp", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(dataframe.columns):
        return DEFAULT_EVENT_RESULT.copy()

    frame = _prepare_intraday_frame(dataframe)
    if frame.empty:
        return DEFAULT_EVENT_RESULT.copy()

    volume_type, volume_score = _detect_volume_divergence(frame)
    macd_divergence_type, macd_divergence_score = _detect_macd_divergence(frame)
    macd_cross_type, macd_cross_score = _detect_macd_cross(frame)
    ma_event_type, ma_score = _detect_ma_event(frame)

    total = _clip_score(
        NEUTRAL_INTRADAY_SCORE + volume_score + macd_divergence_score + macd_cross_score + ma_score
    )
    return {
        "intraday_5m_score": total,
        "intraday_volume_divergence_hit": volume_type != "none",
        "intraday_volume_divergence_type": volume_type,
        "intraday_volume_score": volume_score,
        "intraday_macd_divergence_hit": macd_divergence_type != "none",
        "intraday_macd_divergence_type": macd_divergence_type,
        "intraday_macd_divergence_score": macd_divergence_score,
        "intraday_macd_cross_hit": macd_cross_type != "none",
        "intraday_macd_cross_type": macd_cross_type,
        "intraday_macd_cross_score": macd_cross_score,
        "intraday_ma_event_hit": ma_event_type != "none",
        "intraday_ma_event_type": ma_event_type,
        "intraday_ma_score": ma_score,
    }


def _build_daily_summary_frame(
    *,
    symbols: list[str],
    tradingview_path: Path,
    divergence_path: Path,
    pattern_path: Path,
) -> pd.DataFrame:
    tradingview = _load_tradingview_summary(tradingview_path)
    divergence = _load_divergence_summary(divergence_path)
    patterns = _load_pattern_summary(pattern_path)

    merged = pd.DataFrame({"symbol": [str(symbol).zfill(6) for symbol in symbols]})
    if not tradingview.empty:
        merged = merged.merge(tradingview, how="left", on="symbol")
    if not divergence.empty:
        merged = merged.merge(divergence, how="left", on="symbol")
    if not patterns.empty:
        merged = merged.merge(patterns, how="left", on="symbol")

    name_columns = [column for column in merged.columns if column.startswith("name")]
    if name_columns:
        resolved_name = pd.Series([None] * len(merged), index=merged.index, dtype=object)
        for column in name_columns:
            resolved_name = resolved_name.fillna(merged[column])
        merged = merged.drop(columns=name_columns, errors="ignore")
        merged["name"] = resolved_name

    merged["pattern_ids"] = _column_or_default(merged, "pattern_ids", "")
    merged["pattern_id"] = _column_or_default(merged, "pattern_id", "")
    merged["tradingview_avg_all_rating_5d"] = pd.to_numeric(
        _column_or_default(merged, "tradingview_avg_all_rating_5d", 0.0), errors="coerce"
    ).fillna(0.0)
    merged["tradingview_all_rating"] = pd.to_numeric(
        _column_or_default(merged, "tradingview_all_rating", 0.0), errors="coerce"
    ).fillna(0.0)
    merged["tradingview_all_rating_label"] = _column_or_default(merged, "tradingview_all_rating_label", None)
    merged["daily_macd_top_divergence_15d"] = _column_or_default(
        merged, "daily_macd_top_divergence_15d", False
    ).fillna(False).astype(bool)
    merged["daily_macd_bottom_divergence_15d"] = _column_or_default(
        merged, "daily_macd_bottom_divergence_15d", False
    ).fillna(False).astype(bool)

    daily_columns = sorted(column for column in merged.columns if column.startswith("tradingview_all_rating_20"))
    merged["daily_score"] = merged.apply(lambda row: round(float(_stable_score(row, daily_columns)), 4), axis=1)

    ordered_columns = [
        "symbol",
        "name",
        "pattern_ids",
        "pattern_id",
        "tradingview_avg_all_rating_5d",
        "tradingview_all_rating",
        "tradingview_all_rating_label",
        "daily_macd_top_divergence_15d",
        "daily_macd_bottom_divergence_15d",
        "daily_score",
    ]
    remaining = [column for column in merged.columns if column not in ordered_columns]
    return merged.loc[:, ordered_columns + remaining]


def build_watchlist_summary_frame(payload: dict[str, object]) -> pd.DataFrame:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "pattern_ids",
                "tradingview_avg_all_rating_5d",
                "tradingview_all_rating_label",
                "daily_macd_top_divergence_15d",
                "daily_macd_bottom_divergence_15d",
            ]
        )

    rows: list[dict[str, object]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = _normalize_symbol(item.get("symbol", ""))
        if symbol == "000000":
            continue
        pattern_id = str(item.get("pattern_id", "")).strip()
        rows.append(
            {
                "symbol": symbol,
                "name": item.get("name"),
                "pattern_ids": pattern_id,
                "tradingview_avg_all_rating_5d": pd.to_numeric(item.get("tradingview_avg_5d"), errors="coerce"),
                "tradingview_all_rating_label": item.get("tradingview_label"),
                "daily_macd_top_divergence_15d": bool(item.get("macd_top_divergence_15d", False)),
                "daily_macd_bottom_divergence_15d": bool(item.get("macd_bottom_divergence_15d", False)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "pattern_ids",
                "tradingview_avg_all_rating_5d",
                "tradingview_all_rating_label",
                "daily_macd_top_divergence_15d",
                "daily_macd_bottom_divergence_15d",
            ]
        )

    frame = pd.DataFrame(rows)
    merged_rows: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        pattern_ids = sorted(
            {
                pattern_id.strip()
                for pattern_id in group["pattern_ids"].fillna("").astype(str)
                if pattern_id.strip()
            },
            key=lambda item: (-PATTERN_PRIORITY.get(item, 0.0), item),
        )
        names = group["name"].dropna()
        labels = group["tradingview_all_rating_label"].dropna()
        tradingview_scores = pd.to_numeric(group["tradingview_avg_all_rating_5d"], errors="coerce")
        merged_rows.append(
            {
                "symbol": symbol,
                "name": names.iloc[0] if not names.empty else None,
                "pattern_ids": ",".join(pattern_ids),
                "tradingview_avg_all_rating_5d": (
                    float(tradingview_scores.max()) if tradingview_scores.notna().any() else pd.NA
                ),
                "tradingview_all_rating_label": labels.iloc[0] if not labels.empty else None,
                "daily_macd_top_divergence_15d": bool(group["daily_macd_top_divergence_15d"].fillna(False).any()),
                "daily_macd_bottom_divergence_15d": bool(
                    group["daily_macd_bottom_divergence_15d"].fillna(False).any()
                ),
            }
        )
    return pd.DataFrame(merged_rows)


def _load_tradingview_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["symbol"])
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    rename_map = {
        "avg_all_rating_5d": "tradingview_avg_all_rating_5d",
        "all_rating": "tradingview_all_rating",
        "all_rating_label": "tradingview_all_rating_label",
    }
    for column in list(frame.columns):
        if column.startswith("all_rating_20"):
            rename_map[column] = f"tradingview_{column}"
    frame = frame.rename(columns=rename_map)
    columns = [
        "symbol",
        "name",
        "tradingview_avg_all_rating_5d",
        "tradingview_all_rating",
        "tradingview_all_rating_label",
        *[column for column in frame.columns if column.startswith("tradingview_all_rating_20")],
    ]
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


def _load_divergence_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["symbol"])
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    rename_map = {
        "macd_top_divergence_15d": "daily_macd_top_divergence_15d",
        "macd_bottom_divergence_15d": "daily_macd_bottom_divergence_15d",
    }
    frame = frame.rename(columns=rename_map)
    columns = [
        "symbol",
        "name",
        "daily_macd_top_divergence_15d",
        "daily_macd_bottom_divergence_15d",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


def _load_pattern_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["symbol"])
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame["pattern_id"] = frame["pattern_id"].astype(str)
    rows: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        pattern_ids = sorted(set(group["pattern_id"]), key=lambda item: (-PATTERN_PRIORITY.get(item, 0.0), item))
        primary_pattern_id = max(pattern_ids, key=lambda item: PATTERN_PRIORITY.get(item, 0.0), default="")
        rows.append(
            {
                "symbol": symbol,
                "name": group["name"].iloc[0],
                "pattern_ids": ",".join(pattern_ids),
                "pattern_id": primary_pattern_id,
            }
        )
    return pd.DataFrame(rows)


def _prepare_intraday_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "high", "low", "close", "volume", "amount"]).reset_index(drop=True)
    if frame.empty:
        return frame
    frame["ema_12"] = frame["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    frame["ema_26"] = frame["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    frame["macd_dif"] = frame["ema_12"] - frame["ema_26"]
    frame["macd_dea"] = frame["macd_dif"].ewm(span=9, adjust=False, min_periods=9).mean()
    frame["macd_hist"] = (frame["macd_dif"] - frame["macd_dea"]) * 2
    frame["ma_5"] = frame["close"].rolling(5).mean()
    frame["ma_10"] = frame["close"].rolling(10).mean()
    frame["pivot_volume"] = frame["volume"].rolling(3, center=True, min_periods=1).mean()
    return frame


def _detect_volume_divergence(frame: pd.DataFrame) -> tuple[str, float]:
    high_pivots = _collect_pivots(frame, price_column="high", value_column="pivot_volume")
    low_pivots = _collect_pivots(frame, price_column="low", value_column="pivot_volume")
    bearish = _find_volume_divergence(high_pivots, mode="top")
    bullish = _find_volume_divergence(low_pivots, mode="bottom")
    return _pick_latest_signal(bullish, bearish, bullish_score=12.0, bearish_score=-15.0)


def _detect_macd_divergence(frame: pd.DataFrame) -> tuple[str, float]:
    high_pivots = _collect_pivots(frame, price_column="high", value_column="macd_dif")
    low_pivots = _collect_pivots(frame, price_column="low", value_column="macd_dif")
    bearish = _find_divergence(high_pivots, mode="top", compare_column="value")
    bullish = _find_divergence(low_pivots, mode="bottom", compare_column="value")
    return _pick_latest_signal(bullish, bearish, bullish_score=20.0, bearish_score=-20.0)


def _detect_macd_cross(frame: pd.DataFrame) -> tuple[str, float]:
    if len(frame) < 4:
        return "none", 0.0

    delta = frame["macd_dif"] - frame["macd_dea"]
    valid = delta.notna() & delta.shift(1).notna()
    if not valid.any():
        return "none", 0.0

    recent_start = max(1, len(frame) - 3)
    latest_index: int | None = None
    latest_type = "none"
    for index in range(recent_start, len(frame)):
        if not bool(valid.iloc[index]):
            continue
        previous = float(delta.iloc[index - 1])
        current = float(delta.iloc[index])
        if previous <= 0 < current:
            latest_index = index
            latest_type = "golden_cross"
        elif previous >= 0 > current:
            latest_index = index
            latest_type = "death_cross"

    if latest_index is None:
        return "none", 0.0

    hist = frame["macd_hist"]
    if latest_type == "golden_cross":
        if len(hist) >= 3 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-2]) and hist.iloc[-1] > hist.iloc[-2] > 0:
            return "golden_cross_continuation", 18.0
        return "golden_cross", 12.0
    if len(hist) >= 3 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-2]) and hist.iloc[-1] < hist.iloc[-2] < 0:
        return "death_cross_continuation", -18.0
    return "death_cross", -12.0


def _detect_ma_event(frame: pd.DataFrame) -> tuple[str, float]:
    if len(frame) < 10:
        return "none", 0.0

    latest = frame.iloc[-1]
    prev = frame.iloc[-2]
    prev2 = frame.iloc[-3]
    latest_above = _is_above_short_mas(latest)
    prev_above = _is_above_short_mas(prev)
    prev2_above = _is_above_short_mas(prev2)

    if latest_above and (not prev_above or not prev2_above):
        return "reclaim_ma", 12.0

    pullback_window = frame.iloc[-4:]
    if latest_above and latest["close"] > prev["close"]:
        touched_ma = (
            (pullback_window["low"] <= pullback_window["ma_5"] * 1.002)
            | (pullback_window["low"] <= pullback_window["ma_10"] * 1.002)
        ).fillna(False)
        if bool(touched_ma.any()) and prev_above:
            return "pullback_hold_ma", 15.0

    latest_below = _is_below_short_mas(latest)
    if latest_below and prev_above:
        return "break_ma", -12.0
    return "none", 0.0


def _collect_pivots(
    frame: pd.DataFrame,
    *,
    price_column: str,
    value_column: str,
    left: int = 2,
    right: int = 2,
) -> list[dict[str, object]]:
    pivots: list[dict[str, object]] = []
    for index in range(left, len(frame) - right):
        price = frame.iloc[index][price_column]
        value = frame.iloc[index][value_column]
        if pd.isna(price) or pd.isna(value):
            continue

        window = frame.iloc[index - left : index + right + 1]
        center_price = float(price)
        if price_column == "high":
            is_pivot = center_price == float(window[price_column].max())
            strict = all(center_price > float(item) for item in window[price_column].tolist() if float(item) != center_price)
        else:
            is_pivot = center_price == float(window[price_column].min())
            strict = all(center_price < float(item) for item in window[price_column].tolist() if float(item) != center_price)
        if not is_pivot or not strict:
            continue

        pivots.append(
            {
                "index": index,
                "timestamp": frame.iloc[index]["timestamp"],
                "price": center_price,
                "value": float(value),
            }
        )
    return pivots


def _find_divergence(
    pivots: list[dict[str, object]],
    *,
    mode: str,
    compare_column: str,
) -> dict[str, object] | None:
    latest: dict[str, object] | None = None
    for previous, current in zip(pivots, pivots[1:]):
        previous_price = float(previous["price"])
        current_price = float(current["price"])
        previous_value = float(previous[compare_column])
        current_value = float(current[compare_column])
        if mode == "top":
            matched = current_price > previous_price and current_value < previous_value
        else:
            matched = current_price < previous_price and current_value > previous_value
        if matched:
            latest = current
    return latest


def _find_volume_divergence(
    pivots: list[dict[str, object]],
    *,
    mode: str,
) -> dict[str, object] | None:
    latest: dict[str, object] | None = None
    for previous, current in zip(pivots, pivots[1:]):
        previous_price = float(previous["price"])
        current_price = float(current["price"])
        previous_value = float(previous["value"])
        current_value = float(current["value"])
        if mode == "top":
            matched = current_price > previous_price and current_value < previous_value
        else:
            matched = current_price < previous_price and current_value < previous_value
        if matched:
            latest = current
    return latest


def _pick_latest_signal(
    bullish: dict[str, object] | None,
    bearish: dict[str, object] | None,
    *,
    bullish_score: float,
    bearish_score: float,
) -> tuple[str, float]:
    if bullish and bearish:
        if pd.Timestamp(bullish["timestamp"]) >= pd.Timestamp(bearish["timestamp"]):
            return "bullish", bullish_score
        return "bearish", bearish_score
    if bullish:
        return "bullish", bullish_score
    if bearish:
        return "bearish", bearish_score
    return "none", 0.0


def _normalize_symbol(value: object) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    return text.zfill(6)


def _default_daily_row(symbol: str) -> dict[str, object]:
    return {
        "symbol": str(symbol).zfill(6),
        "name": None,
        "pattern_ids": "",
        "tradingview_avg_all_rating_5d": 0.0,
        "tradingview_all_rating_label": None,
        "daily_macd_top_divergence_15d": False,
        "daily_macd_bottom_divergence_15d": False,
    }


def localize_ranking_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.rename(columns=CSV_HEADER_MAP)
    return dataframe.rename(columns=CSV_HEADER_MAP)


def _column_or_default(dataframe: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in dataframe.columns:
        return dataframe[column]
    return pd.Series([default] * len(dataframe), index=dataframe.index)


def _is_above_short_mas(row: pd.Series) -> bool:
    return bool(
        pd.notna(row.get("ma_5"))
        and pd.notna(row.get("ma_10"))
        and float(row["close"]) > float(row["ma_5"])
        and float(row["close"]) > float(row["ma_10"])
    )


def _is_below_short_mas(row: pd.Series) -> bool:
    return bool(
        pd.notna(row.get("ma_5"))
        and pd.notna(row.get("ma_10"))
        and float(row["close"]) < float(row["ma_5"])
        and float(row["close"]) < float(row["ma_10"])
    )


def _clip_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 4)
