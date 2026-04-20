from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd

from .macd_divergence import summarize_recent_macd_divergence
from .models import AppConfig, TrendEntryRulesConfig, TrendSignalEntryRulesConfig
from .storage import Storage
from .trend_signals import generate_trend_signals_from_frame
from .trend_universe import build_symbol_trend_frame


SCORED_ENTRY_COLUMNS = [
    "trade_date",
    "planned_entry_date",
    "planned_entry_open",
    "symbol",
    "name",
    "setup_type",
    "signal_type",
    "trend_score",
    "entry_score",
    "trend_base_score",
    "price_action_score",
    "macd_score",
    "volume_score",
    "volume_price_divergence_score",
    "boll_score",
    "rsi_score",
    "kdj_score",
    "atr_score",
    "trigger_score",
    "buy_score",
    "positive_indicator_count",
    "macd_cross_state",
    "macd_divergence_state",
    "volume_price_divergence_state",
    "macd_top_divergence_flag",
    "macd_bottom_divergence_flag",
    "bullish_volume_price_divergence_flag",
    "bearish_volume_price_divergence_flag",
    "distance_to_breakout_pct",
    "distance_to_ma20",
    "distance_to_ma60",
    "drawdown_from_recent_high",
    "volume_ratio_20",
    "volume_contraction_ratio",
    "trigger_reason",
    "buy_reason",
    "entry_timing",
]


def score_symbol_trend_entries(
    daily_bars: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    trend_frame = build_symbol_trend_frame(
        daily_bars,
        symbol=symbol,
        name=name,
        config=config.trend_universe,
    )
    trend_frame = _add_scoring_indicator_fields(trend_frame)
    setups = generate_trend_signals_from_frame(
        trend_frame,
        config=config,
        start_date=start_date,
        end_date=end_date,
    )
    if setups.empty:
        return pd.DataFrame()

    index_by_date = {pd.Timestamp(row["trade_date"]).date(): index for index, row in trend_frame.iterrows()}
    rows: list[dict[str, object]] = []
    for setup in setups.to_dict("records"):
        trade_date = pd.Timestamp(setup["trade_date"]).date()
        frame_index = index_by_date.get(trade_date)
        if frame_index is None:
            continue
        row = trend_frame.iloc[frame_index]
        next_row = trend_frame.iloc[frame_index + 1] if frame_index + 1 < len(trend_frame) else None
        scored = _score_setup(
            trend_frame=trend_frame.iloc[: frame_index + 1].reset_index(drop=True),
            row=row,
            setup=setup,
            next_row=next_row,
            config=config,
        )
        rows.append(scored)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reindex(columns=SCORED_ENTRY_COLUMNS)


def scan_indicator_scored_entries(
    storage: Storage,
    config: AppConfig,
    *,
    trade_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    symbols: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date

    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    if symbols:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        universe = universe[universe["symbol"].isin(symbol_set)].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total_instruments = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        symbol = str(instrument["symbol"]).zfill(6)
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            if progress_callback is not None:
                progress_callback(index, total_instruments)
            continue
        scored = score_symbol_trend_entries(
            bars,
            symbol=symbol,
            name=str(instrument.get("name", "")),
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        if scored.empty:
            if progress_callback is not None:
                progress_callback(index, total_instruments)
            continue
        rows.extend(scored.to_dict("records"))
        if progress_callback is not None:
            progress_callback(index, total_instruments)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).reindex(columns=SCORED_ENTRY_COLUMNS)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result.sort_values(["trade_date", "buy_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def build_next_open_entries(scored_entries: pd.DataFrame) -> pd.DataFrame:
    if scored_entries.empty:
        return pd.DataFrame()
    frame = scored_entries.copy()
    frame = frame[frame["planned_entry_date"].notna()].reset_index(drop=True)
    if frame.empty:
        return frame
    frame["entry_timing"] = "next_open"
    return frame


def select_tradable_entries(scored_entries: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    entries = build_next_open_entries(scored_entries)
    if entries.empty:
        return entries

    filtered_rows: list[dict[str, object]] = []
    for row in entries.to_dict("records"):
        rules = _resolve_entry_rules(str(row.get("signal_type", "")).lower(), config.trend_entry_rules)
        if _passes_entry_rules(row, rules):
            filtered_rows.append(row)

    if not filtered_rows:
        return entries.iloc[0:0].copy()
    return pd.DataFrame(filtered_rows).reindex(columns=entries.columns).reset_index(drop=True)


def _resolve_entry_rules(signal_type: str, rules: TrendEntryRulesConfig) -> TrendSignalEntryRulesConfig:
    override = rules.breakout if signal_type == "breakout" else rules.pullback if signal_type == "pullback" else None
    return TrendSignalEntryRulesConfig(
        buy_score_min=_resolve_optional_threshold(getattr(override, "buy_score_min", None), rules.buy_score_min),
        trend_base_score_min=_resolve_optional_threshold(getattr(override, "trend_base_score_min", None), rules.trend_base_score_min),
        price_action_score_min=_resolve_optional_threshold(getattr(override, "price_action_score_min", None), rules.price_action_score_min),
        macd_score_min=_resolve_optional_threshold(getattr(override, "macd_score_min", None), rules.macd_score_min),
        positive_indicator_count_min=_resolve_optional_threshold(
            getattr(override, "positive_indicator_count_min", None), rules.positive_indicator_count_min
        ),
    )


def _resolve_optional_threshold(value: float | int | None, fallback: float | int) -> float | int:
    return fallback if value is None else value


def _passes_entry_rules(row: dict[str, object], rules: TrendSignalEntryRulesConfig) -> bool:
    return (
        float(row.get("buy_score", float("-inf"))) >= float(rules.buy_score_min)
        and float(row.get("trend_base_score", float("-inf"))) >= float(rules.trend_base_score_min)
        and float(row.get("price_action_score", float("-inf"))) >= float(rules.price_action_score_min)
        and float(row.get("macd_score", float("-inf"))) >= float(rules.macd_score_min)
        and int(row.get("positive_indicator_count", -1)) >= int(rules.positive_indicator_count_min)
    )


def _score_setup(
    *,
    trend_frame: pd.DataFrame,
    row: pd.Series,
    setup: dict[str, object],
    next_row: pd.Series | None,
    config: AppConfig,
) -> dict[str, object]:
    trend_base_score = _score_trend_base(row)
    price_action_score = _score_price_action(setup, row, config)
    macd_summary = summarize_recent_macd_divergence(trend_frame)
    macd_score = _score_macd(trend_frame, row, macd_summary)
    volume_score = _score_volume(setup, row, config)
    volume_price_divergence_score, bullish_volume_price_divergence, bearish_volume_price_divergence = _score_volume_price_divergence(
        setup,
        row,
        config,
    )
    boll_score = _score_boll(setup, row)
    rsi_score = _score_rsi(trend_frame, row)
    kdj_score = _score_kdj(trend_frame, row)
    atr_score = _score_atr(row)

    weights = config.trend_indicator_weights
    trigger_score = (
        weights.price_action_weight * price_action_score
        + weights.macd_weight * macd_score
        + weights.volume_weight * volume_score
        + weights.volume_price_divergence_weight * volume_price_divergence_score
        + weights.boll_weight * boll_score
        + weights.rsi_weight * rsi_score
        + weights.kdj_weight * kdj_score
        + weights.atr_weight * atr_score
    )
    buy_score = weights.trend_base_weight * trend_base_score + (1 - weights.trend_base_weight) * trigger_score
    component_scores = {
        "price_action": price_action_score,
        "macd": macd_score,
        "volume": volume_score,
        "volume_price_divergence": volume_price_divergence_score,
        "boll": boll_score,
        "rsi": rsi_score,
        "kdj": kdj_score,
        "atr": atr_score,
    }
    positive_indicator_count = sum(1 for value in component_scores.values() if value >= 60.0)
    buy_reason = _compose_buy_reason(component_scores, setup)
    macd_cross_state = _describe_macd_cross_state(trend_frame, row)
    macd_divergence_state = _describe_macd_divergence_state(macd_summary)
    volume_price_divergence_state = _describe_volume_price_divergence_state(
        bullish_volume_price_divergence,
        bearish_volume_price_divergence,
    )

    return {
        "trade_date": pd.Timestamp(setup["trade_date"]),
        "planned_entry_date": pd.Timestamp(next_row["trade_date"]) if next_row is not None else None,
        "planned_entry_open": round(float(next_row["open"]), 4) if next_row is not None and pd.notna(next_row.get("open")) else None,
        "symbol": str(setup["symbol"]).zfill(6),
        "name": setup.get("name", ""),
        "setup_type": setup.get("signal_type"),
        "signal_type": setup.get("signal_type"),
        "trend_score": round(float(setup.get("trend_score", 0.0)), 4),
        "entry_score": round(float(setup.get("entry_score", 0.0)), 4),
        "trend_base_score": round(trend_base_score, 4),
        "price_action_score": round(price_action_score, 4),
        "macd_score": round(macd_score, 4),
        "volume_score": round(volume_score, 4),
        "volume_price_divergence_score": round(volume_price_divergence_score, 4),
        "boll_score": round(boll_score, 4),
        "rsi_score": round(rsi_score, 4),
        "kdj_score": round(kdj_score, 4),
        "atr_score": round(atr_score, 4),
        "trigger_score": round(trigger_score, 4),
        "buy_score": round(buy_score, 4),
        "positive_indicator_count": int(positive_indicator_count),
        "macd_cross_state": macd_cross_state,
        "macd_divergence_state": macd_divergence_state,
        "volume_price_divergence_state": volume_price_divergence_state,
        "macd_top_divergence_flag": bool(macd_summary.get("macd_top_divergence_15d", False)),
        "macd_bottom_divergence_flag": bool(macd_summary.get("macd_bottom_divergence_15d", False)),
        "bullish_volume_price_divergence_flag": bool(bullish_volume_price_divergence),
        "bearish_volume_price_divergence_flag": bool(bearish_volume_price_divergence),
        "distance_to_breakout_pct": setup.get("distance_to_breakout_pct"),
        "distance_to_ma20": setup.get("distance_to_ma20"),
        "distance_to_ma60": setup.get("distance_to_ma60"),
        "drawdown_from_recent_high": setup.get("drawdown_from_recent_high"),
        "volume_ratio_20": setup.get("volume_ratio_20"),
        "volume_contraction_ratio": setup.get("volume_contraction_ratio"),
        "trigger_reason": setup.get("trigger_reason", ""),
        "buy_reason": buy_reason,
        "entry_timing": "next_open",
    }


def _add_scoring_indicator_fields(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy().sort_values("trade_date").reset_index(drop=True)
    close = enriched["close"].astype(float)
    high = enriched["high"].astype(float)
    low = enriched["low"].astype(float)

    enriched["boll_mid_20"] = close.rolling(20).mean()
    boll_std = close.rolling(20).std(ddof=0)
    enriched["boll_upper_20"] = enriched["boll_mid_20"] + 2 * boll_std
    enriched["boll_lower_20"] = enriched["boll_mid_20"] - 2 * boll_std
    enriched["boll_band_width_20"] = (
        enriched["boll_upper_20"] - enriched["boll_lower_20"]
    ).div(enriched["boll_mid_20"].replace(0.0, pd.NA))

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    enriched["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    enriched["atr_ratio_14"] = enriched["atr_14"].div(close.replace(0.0, pd.NA))
    enriched["atr_ratio_median_20"] = enriched["atr_ratio_14"].rolling(20).median()
    enriched["kdj_j"] = 3 * enriched["stoch_k"] - 2 * enriched["stoch_d"]
    return enriched


def _score_trend_base(row: pd.Series) -> float:
    ma_structure_score = 0.0
    if bool(row.get("trend_ma_alignment", False)):
        ma_structure_score += 50.0
    elif pd.notna(row.get("ma_20")) and pd.notna(row.get("ma_60")) and float(row["ma_20"]) > float(row["ma_60"]):
        ma_structure_score += 35.0

    if bool(row.get("trend_ma_short_rising", False)):
        ma_structure_score += 20.0
    if bool(row.get("trend_ma_medium_rising", False)):
        ma_structure_score += 15.0
    if bool(row.get("trend_price_above_short", False)):
        ma_structure_score += 10.0
    if bool(row.get("trend_price_above_medium", False)):
        ma_structure_score += 5.0
    ma_structure_score = _clip(ma_structure_score)

    adx = _safe_float_or_none(row.get("adx_14"))
    plus_di = _safe_float_or_none(row.get("plus_di_14"))
    minus_di = _safe_float_or_none(row.get("minus_di_14"))
    if adx is None or plus_di is None or minus_di is None:
        adx_score = 50.0
    else:
        if adx >= 35:
            adx_score = 90.0
        elif adx >= 25:
            adx_score = 78.0
        elif adx >= 20:
            adx_score = 60.0
        else:
            adx_score = 35.0
        adx_score += 10.0 if plus_di > minus_di else -20.0
        adx_score = _clip(adx_score)
    return round(0.7 * ma_structure_score + 0.3 * adx_score, 4)


def _score_price_action(setup: dict[str, object], row: pd.Series, config: AppConfig) -> float:
    setup_quality = _clip(float(setup.get("entry_score", 0.0)))
    if setup.get("signal_type") == "breakout":
        distance = abs(_safe_float(setup.get("distance_to_breakout_pct"), 0.0))
        key_level_score = _clip(100.0 * (1 - distance / max(config.trend_signals.breakout.breakout_max_distance_pct, 1e-6)))
        candle_confirmation_score = _clip(
            45.0
            + max(_safe_float(row.get("return_1d"), 0.0), 0.0) * 500.0
            + _safe_float(row.get("body_pct"), 0.0) * 30.0
            + (1 - _safe_float(row.get("upper_shadow_pct"), 1.0)) * 25.0
        )
    else:
        ma20_distance = abs(_safe_float(setup.get("distance_to_ma20"), 0.0))
        ma60_distance = abs(_safe_float(setup.get("distance_to_ma60"), 0.0))
        distance_ratio = min(
            ma20_distance / max(config.trend_signals.pullback.proximity_to_ma20, 1e-6),
            ma60_distance / max(config.trend_signals.pullback.proximity_to_ma60, 1e-6),
        )
        key_level_score = _clip(100.0 * (1 - distance_ratio / 1.5))
        candle_confirmation_score = _clip(
            35.0
            + max(_safe_float(row.get("return_1d"), 0.0), 0.0) * 450.0
            + _safe_float(row.get("lower_shadow_pct"), 0.0) * 35.0
            + _safe_float(row.get("body_pct"), 0.0) * 20.0
        )
    return round(0.5 * setup_quality + 0.25 * key_level_score + 0.25 * candle_confirmation_score, 4)


def _score_macd(trend_frame: pd.DataFrame, row: pd.Series, macd_summary: dict[str, object]) -> float:
    macd = _safe_float_or_none(row.get("macd"))
    signal_line = _safe_float_or_none(row.get("macd_signal_line"))
    hist = _safe_float_or_none(row.get("macd_hist"))
    previous = trend_frame.iloc[-2] if len(trend_frame) >= 2 else None
    previous2 = trend_frame.iloc[-3] if len(trend_frame) >= 3 else None
    if macd is None or signal_line is None or hist is None:
        return 50.0

    if macd > signal_line and macd > 0 and signal_line > 0:
        trend_state = 95.0
    elif macd > signal_line:
        trend_state = 75.0
    elif abs(macd - signal_line) <= 0.03:
        trend_state = 55.0
    else:
        trend_state = 20.0

    recent_cross_up = False
    recent_cross_down = False
    for offset in range(max(0, len(trend_frame) - 3), len(trend_frame)):
        if offset == 0:
            continue
        prev_row = trend_frame.iloc[offset - 1]
        current_row = trend_frame.iloc[offset]
        if pd.isna(prev_row.get("macd")) or pd.isna(prev_row.get("macd_signal_line")):
            continue
        if float(prev_row["macd"]) <= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) > float(current_row["macd_signal_line"]):
            recent_cross_up = True
        if float(prev_row["macd"]) >= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) < float(current_row["macd_signal_line"]):
            recent_cross_down = True
    if recent_cross_up and macd > 0:
        cross_state = 95.0
    elif recent_cross_up:
        cross_state = 75.0
    elif recent_cross_down:
        cross_state = 15.0
    elif macd > signal_line:
        cross_state = 60.0
    else:
        cross_state = 40.0

    previous_hist = float(previous["macd_hist"]) if previous is not None and pd.notna(previous.get("macd_hist")) else None
    previous2_hist = float(previous2["macd_hist"]) if previous2 is not None and pd.notna(previous2.get("macd_hist")) else None
    if previous_hist is None:
        hist_momentum = 50.0
    elif hist > 0 and previous_hist > 0 and (previous2_hist is None or hist > previous_hist >= previous2_hist):
        hist_momentum = 90.0
    elif hist > 0 and hist >= previous_hist:
        hist_momentum = 80.0
    elif hist < 0 and hist > previous_hist:
        hist_momentum = 60.0
    else:
        hist_momentum = 20.0

    if bool(macd_summary.get("macd_bottom_divergence_15d", False)):
        divergence_state = 95.0
    elif bool(macd_summary.get("macd_top_divergence_15d", False)):
        divergence_state = 10.0
    else:
        divergence_state = 50.0

    return round(0.35 * trend_state + 0.20 * cross_state + 0.15 * hist_momentum + 0.30 * divergence_state, 4)


def _describe_macd_cross_state(trend_frame: pd.DataFrame, row: pd.Series) -> str:
    macd = _safe_float_or_none(row.get("macd"))
    signal_line = _safe_float_or_none(row.get("macd_signal_line"))
    if macd is None or signal_line is None:
        return "unknown"

    recent_cross_up = False
    recent_cross_down = False
    for offset in range(max(0, len(trend_frame) - 3), len(trend_frame)):
        if offset == 0:
            continue
        prev_row = trend_frame.iloc[offset - 1]
        current_row = trend_frame.iloc[offset]
        if pd.isna(prev_row.get("macd")) or pd.isna(prev_row.get("macd_signal_line")):
            continue
        if float(prev_row["macd"]) <= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) > float(current_row["macd_signal_line"]):
            recent_cross_up = True
        if float(prev_row["macd"]) >= float(prev_row["macd_signal_line"]) and float(current_row["macd"]) < float(current_row["macd_signal_line"]):
            recent_cross_down = True

    if recent_cross_up:
        return "golden_cross"
    if recent_cross_down:
        return "dead_cross"
    if macd > signal_line:
        return "above_signal"
    return "below_signal"


def _describe_macd_divergence_state(macd_summary: dict[str, object]) -> str:
    if bool(macd_summary.get("macd_bottom_divergence_15d", False)):
        return "bottom_divergence"
    if bool(macd_summary.get("macd_top_divergence_15d", False)):
        return "top_divergence"
    return "none"


def _describe_volume_price_divergence_state(bullish: bool, bearish: bool) -> str:
    if bullish and not bearish:
        return "bullish"
    if bearish and not bullish:
        return "bearish"
    return "none"


def _score_volume(setup: dict[str, object], row: pd.Series, config: AppConfig) -> float:
    volume_ratio_20 = _safe_float(setup.get("volume_ratio_20"), 1.0)
    if setup.get("signal_type") == "breakout":
        return round(_clip((volume_ratio_20 - 0.8) / 1.0 * 100.0), 4)

    contraction_ratio = _safe_float(setup.get("volume_contraction_ratio"), 1.0)
    score = _clip((1.2 - contraction_ratio) / 0.7 * 100.0)
    if _safe_float(row.get("return_1d"), 0.0) > 0:
        score = _clip(score + 10.0)
    return round(score, 4)


def _score_volume_price_divergence(setup: dict[str, object], row: pd.Series, config: AppConfig) -> tuple[float, bool, bool]:
    volume_ratio_20 = _safe_float(setup.get("volume_ratio_20"), 1.0)
    bullish = False
    bearish = False
    if setup.get("signal_type") == "breakout":
        bullish = volume_ratio_20 >= config.trend_signals.breakout.breakout_volume_ratio_min
        bearish = bool(setup.get("distance_to_new_high_pct") is not None) and volume_ratio_20 < 1.0
    else:
        contraction_ratio = _safe_float(setup.get("volume_contraction_ratio"), 1.0)
        bullish = contraction_ratio <= config.trend_signals.pullback.volume_contraction_max and _safe_float(row.get("return_1d"), 0.0) > 0
        bearish = _safe_float(row.get("return_1d"), 0.0) < 0 and volume_ratio_20 > 1.1

    if bullish and not bearish:
        return 85.0, True, False
    if bearish and not bullish:
        return 15.0, False, True
    return 55.0, False, False


def _score_boll(setup: dict[str, object], row: pd.Series) -> float:
    if pd.isna(row.get("boll_mid_20")) or pd.isna(row.get("boll_upper_20")) or pd.isna(row.get("boll_lower_20")):
        return 50.0
    close = float(row["close"])
    mid = float(row["boll_mid_20"])
    upper = float(row["boll_upper_20"])
    lower = float(row["boll_lower_20"])
    width = _safe_float(row.get("boll_band_width_20"), 0.0)
    previous_width = _safe_float(row.get("boll_band_width_20"), 0.0)
    if setup.get("signal_type") == "breakout":
        if close >= mid and close <= upper and width >= previous_width:
            return 85.0
        if close > upper:
            return 70.0
        if close < mid:
            return 25.0
        return 60.0
    if close >= mid and abs(close / mid - 1) <= 0.03:
        return 90.0
    if close >= lower:
        return 65.0
    return 20.0


def _score_rsi(trend_frame: pd.DataFrame, row: pd.Series) -> float:
    rsi = _safe_float_or_none(row.get("rsi_14"))
    previous = trend_frame.iloc[-2] if len(trend_frame) >= 2 else None
    previous_rsi = float(previous["rsi_14"]) if previous is not None and pd.notna(previous.get("rsi_14")) else None
    if rsi is None:
        return 50.0
    if 50 <= rsi <= 70 and previous_rsi is not None and rsi >= previous_rsi:
        return 88.0
    if 40 <= rsi < 50 and previous_rsi is not None and rsi > previous_rsi:
        return 76.0
    if rsi > 75 and previous_rsi is not None and rsi >= previous_rsi:
        return 62.0
    if rsi > 75 and previous_rsi is not None and rsi < previous_rsi:
        return 30.0
    if rsi < 40 and previous_rsi is not None and rsi > previous_rsi:
        return 52.0
    return 25.0


def _score_kdj(trend_frame: pd.DataFrame, row: pd.Series) -> float:
    k = _safe_float_or_none(row.get("stoch_k"))
    d = _safe_float_or_none(row.get("stoch_d"))
    if k is None or d is None:
        return 50.0
    previous = trend_frame.iloc[-2] if len(trend_frame) >= 2 else None
    previous_k = float(previous["stoch_k"]) if previous is not None and pd.notna(previous.get("stoch_k")) else None
    previous_d = float(previous["stoch_d"]) if previous is not None and pd.notna(previous.get("stoch_d")) else None
    cross_up = previous_k is not None and previous_d is not None and previous_k <= previous_d and k > d
    if k < 30 and cross_up:
        return 90.0
    if 30 <= k <= 60 and cross_up:
        return 78.0
    if k > 80 and k < d:
        return 20.0
    if k > d:
        return 60.0
    return 40.0


def _score_atr(row: pd.Series) -> float:
    atr_ratio = _safe_float_or_none(row.get("atr_ratio_14"))
    atr_median = _safe_float_or_none(row.get("atr_ratio_median_20"))
    if atr_ratio is None or atr_median is None or atr_median <= 0:
        return 50.0
    relative = atr_ratio / atr_median
    deviation = abs(relative - 1.0)
    score = 90.0 - min(deviation * 80.0, 70.0)
    if relative > 1.8:
        score = min(score, 35.0)
    elif relative < 0.6:
        score = min(score, 55.0)
    return round(_clip(score), 4)


def _compose_buy_reason(component_scores: dict[str, float], setup: dict[str, object]) -> str:
    strongest = sorted(component_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    top_text = ", ".join(f"{name}={score:.1f}" for name, score in strongest)
    return f"{setup.get('signal_type')} with {top_text}"


def _clip(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _safe_float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
