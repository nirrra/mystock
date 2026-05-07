from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .atr import ATR_WATCHLIST_FIELD_MAP
from .models import PickTrendWatchlistConfig, WatchlistTrendFilterConfig


WATCHLIST_FILENAME_RE = re.compile(r"watchlist_(\d{4}-\d{2}-\d{2})\.json$")
WATCHLIST_STREAK_FIELD = "连续上榜天数"
PATTERN_PRIORITY = {
    "5": 6.0,
    "1": 5.0,
    "6": 4.0,
    "3": 3.0,
    "2": 2.0,
    "4": 1.0,
}
LABEL_BONUS = {
    "strong_buy": 1.0,
    "buy": 0.6,
    "neutral": 0.0,
    "sell": -0.6,
    "strong_sell": -1.0,
}
V42_PREDICT_MODEL_REQUIRED_FIELDS = (
    "symbol",
    "trade_date",
    "action",
    "trade_permission",
    "risk_tier",
    "risk_gate_reason",
    "risk_score",
)
PREDICT_MODEL_CANDIDATE_FIELDS = (
    "model_version",
    "action",
    "risk_candidate_action",
    "risk_action",
    "final_action",
    "trade_permission",
    "opportunity_score",
    "opportunity_threshold",
    "opportunity_quality",
    "risk_tier",
    "risk_gate_reason",
    "risk_score",
    "long_upside_score",
    "opportunity_rank_score",
    "opportunity_rank_score_pct",
    "final_score_v42",
    "buy_score_v42",
    "rank_source_v42",
    "top_risk_horizon",
    "top_upside_horizon",
    "up_prob_5d",
    "down_prob_5d",
    "neutral_prob_5d",
    "up_prob_10d",
    "down_prob_10d",
    "neutral_prob_10d",
    "up_prob_20d",
    "down_prob_20d",
    "neutral_prob_20d",
    "up_prob_60d",
    "down_prob_60d",
    "neutral_prob_60d",
)
INDEX_NAME_MARKERS = {
    "指数",
    "上证综指",
    "深证成指",
    "创业板指",
    "沪深300",
    "中证500",
    "中证1000",
    "科创50",
}
TREND_CANDIDATE_FIELDS = (
    "signal_type",
    "trend_score",
    "entry_score",
    "trend_base_score",
    "price_action_score",
    "macd_score",
    "buy_score",
    "positive_indicator_count",
    "macd_cross_state",
    "macd_divergence_state",
    "volume_price_divergence_state",
    "macd_top_divergence_flag",
    "macd_bottom_divergence_flag",
    "bullish_volume_price_divergence_flag",
    "bearish_volume_price_divergence_flag",
    "trigger_reason",
    "buy_reason",
)
TREND_UNIVERSE_CANDIDATE_FIELDS = (
    "in_trend_universe",
    "trend_universe_score",
    "trend_direction_score",
    "trend_strength_score",
    "trend_quality_score",
    "trend_liquidity_score",
)
PATTERN_CANDIDATE_FIELDS = (
    "old_high_date",
    "old_high_price",
    "days_since_old_high",
    "max_drawdown_since_old_high",
    "distance_to_old_high_pct",
    "breakout_date",
    "breakout_volume_ratio",
    "breakout_close_position",
    "breakout_upper_shadow_pct",
    "breakout_body_pct",
    "breakout_turnover",
    "breakout_turnover_state",
    "extension_above_old_high_pct",
    "days_after_breakout",
    "post_breakout_max_high_extension_pct",
    "duck_peak_date",
    "duck_peak_price",
    "days_since_duck_peak",
    "neck_start_date",
    "neck_return_pct",
    "neck_low_to_peak_return_pct",
    "pullback_low_date",
    "pullback_low_price",
    "peak_to_pullback_drawdown_pct",
    "nostril_cross_date",
    "days_since_nostril_cross",
    "cross_after_pullback_low_days",
    "nostril_cross_ma5_ma10_gap_pct",
    "latest_ma5_ma10_gap_pct",
    "nostril_volume_ma20_ratio",
    "distance_to_duck_peak_pct",
    "pullback_volume_peak_tail_ratio",
    "pullback_back_half_volume_ratio",
    "pullback_max_single_day_peak_tail_ratio",
    "large_bearish_count",
    "max_bearish_body_pct",
    "max_bearish_volume_ratio",
    "platform_window_days",
    "platform_range_pct",
    "platform_volume_contraction_ratio",
    "platform_range_contraction_ratio",
    "platform_low_lift_pct",
    "platform_max_bearish_body_pct",
    "platform_max_bearish_volume_ratio",
    "distance_to_platform_high_pct",
    "distance_to_ma20",
    "drawdown_15d",
    "consolidation_days",
    "consolidation_range_pct",
    "consolidation_volume_ratio",
    "volume_ratio_20",
    "main_rise_start_date",
    "main_rise_end_date",
    "main_rise_return_pct",
    "transition_days",
    "platform_start_date",
    "platform_end_date",
    "platform_high",
    "recent_high_date",
    "recent_high_price",
    "days_since_recent_high",
    "distance_from_recent_high_pct",
    "ma20_slope_short_pct",
    "ma20_slope_long_pct",
    "ma60_slope_short_pct",
    "ma60_slope_long_pct",
    "pullback_volume_contraction_ratio",
    "ma20_touch_date",
    "ma20_touch_distance",
    "pattern6_branch",
    "anchor_date",
    "anchor_close",
    "support_price",
    "anchor_volume_ratio_prev",
    "anchor_volume_ratio_ma20",
    "launch_confirm_high_date",
    "launch_confirm_high_price",
    "launch_confirm_return_pct",
    "peak_date",
    "peak_price",
    "anchor_to_peak_return_pct",
    "limit_up_like_count",
    "pullback_low_date",
    "pullback_low_price",
    "peak_to_pullback_drawdown_pct",
    "pullback_volume_ratio_to_anchor",
    "pullback_front_half_avg_volume",
    "pullback_back_half_avg_volume",
    "pullback_back_half_volume_ratio",
    "rise_tail_avg_volume",
    "pullback_max_volume",
    "pullback_max_rise_tail_volume_ratio",
    "support_touch_date",
    "breakdown_date",
    "breakdown_volume_ratio_to_anchor",
    "reclaim_date",
    "days_to_reclaim",
    "post_reclaim_days",
)


def watchlists_dir(project_root: Path) -> Path:
    return project_root / "reports" / "watchlists"


def watchlist_path(project_root: Path, trade_date: date) -> Path:
    return watchlists_dir(project_root) / f"watchlist_{trade_date.isoformat()}.json"


def watchlist_pattern_path(project_root: Path, trade_date: date) -> Path:
    return watchlists_dir(project_root) / f"watchlist_pattern_{trade_date.isoformat()}.json"


def watchlist_trend_path(project_root: Path, trade_date: date) -> Path:
    return watchlists_dir(project_root) / f"watchlist_trend_{trade_date.isoformat()}.json"


def build_watchlist_candidates_from_patterns(
    pattern_frame: pd.DataFrame,
    *,
    source_file: str,
    limit: int | None = 30,
    model_predictions: pd.DataFrame | None = None,
) -> dict[str, object]:
    frame = pattern_frame.copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame = frame[~frame["name"].map(_is_true_index_name)].copy()
    if model_predictions is not None:
        return _build_watchlist_candidates_from_patterns_with_model(
            frame,
            model_predictions=model_predictions,
            source_file=source_file,
            limit=limit,
        )

    daily_columns = _daily_rating_columns(frame)
    required = {"symbol", "name", "pattern_id", "tradingview_avg_all_rating_5d", "tradingview_all_rating_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Latest patterns frame is missing required TradingView columns: {missing}")

    frame["stable_score"] = frame.apply(lambda row: _stable_score(row, daily_columns), axis=1)
    frame["base_tier"] = frame.apply(_base_tier, axis=1)
    frame = frame.dropna(subset=["base_tier"]).copy()
    frame = frame[~frame.apply(_is_row_risk_excluded, axis=1)].copy()
    frame = frame.sort_values(["base_tier", "stable_score", "tradingview_avg_all_rating_5d"], ascending=[True, False, False])

    candidates: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        candidate = {
            "tier": row["base_tier"],
            "symbol": row["symbol"],
            "name": row["name"],
            "pattern_id": str(row["pattern_id"]),
            "macd_top_divergence_15d": bool(row.get("macd_top_divergence_15d", False)),
            "macd_bottom_divergence_15d": bool(row.get("macd_bottom_divergence_15d", False)),
            "tradingview_label": str(row["tradingview_all_rating_label"]).strip().lower(),
            "tradingview_avg_5d": round(float(row["tradingview_avg_all_rating_5d"]), 4),
            "five_day_scores": [round(float(row[column]), 4) for column in daily_columns if pd.notna(row.get(column))],
            "stable_score": round(float(row["stable_score"]), 4),
            "reason": row.get("reason", ""),
        }
        _append_supported_fields(candidate, row)
        candidates.append(candidate)

    return {
        "source_file": source_file,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _build_watchlist_candidates_from_patterns_with_model(
    pattern_frame: pd.DataFrame,
    *,
    model_predictions: pd.DataFrame,
    source_file: str,
    limit: int | None,
) -> dict[str, object]:
    prediction_frame = _prepare_model_predictions_for_join(model_predictions)
    model_columns = [column for column in prediction_frame.columns if column != "symbol"]
    frame = pattern_frame.drop(columns=[column for column in model_columns if column in pattern_frame.columns], errors="ignore")
    frame = frame.merge(prediction_frame, on="symbol", how="left")
    frame["risk_sort_score"] = pd.to_numeric(frame.get("risk_score", pd.Series(math.inf, index=frame.index)), errors="coerce")
    frame = frame[frame.apply(_is_model_low_risk_row, axis=1)].copy()
    if frame.empty:
        return {
            "source_file": source_file,
            **_trade_permission_metadata(model_predictions),
            "candidate_count": 0,
            "candidates": [],
        }

    frame = frame[~frame.apply(_is_row_risk_excluded, axis=1)].copy()
    if frame.empty:
        return {
            "source_file": source_file,
            **_trade_permission_metadata(model_predictions),
            "candidate_count": 0,
            "candidates": [],
        }

    frame["pattern_priority"] = frame["pattern_id"].astype(str).map(PATTERN_PRIORITY).fillna(0.0)
    frame["base_tier"] = frame.apply(_model_base_tier, axis=1)
    frame["watchlist_sort_score"] = frame.apply(_pattern_watchlist_sort_score, axis=1)
    frame = frame.sort_values(
        ["risk_sort_score", "pattern_priority", "watchlist_sort_score", "symbol"],
        ascending=[True, False, False, True],
    )

    daily_columns = _daily_rating_columns(frame)
    candidates: list[dict[str, object]] = []
    selected = frame if limit is None else frame.head(limit)
    for _, row in selected.iterrows():
        candidate = {
            "tier": row["base_tier"],
            "symbol": row["symbol"],
            "name": row["name"],
            "source": "pattern",
            "source_tags": ["pattern"],
            "pattern_match": True,
            "pattern_id": str(row["pattern_id"]),
            "macd_top_divergence_15d": bool(row.get("macd_top_divergence_15d", False)),
            "macd_bottom_divergence_15d": bool(row.get("macd_bottom_divergence_15d", False)),
            "watchlist_sort_score": round(float(row["watchlist_sort_score"]), 4),
            "reason": row.get("reason", ""),
        }
        if "tradingview_all_rating_label" in row.index and pd.notna(row.get("tradingview_all_rating_label")):
            candidate["tradingview_label"] = str(row["tradingview_all_rating_label"]).strip().lower()
        if "tradingview_avg_all_rating_5d" in row.index and pd.notna(row.get("tradingview_avg_all_rating_5d")):
            candidate["tradingview_avg_5d"] = round(float(row["tradingview_avg_all_rating_5d"]), 4)
        if daily_columns:
            candidate["five_day_scores"] = [round(float(row[column]), 4) for column in daily_columns if pd.notna(row.get(column))]
        _append_supported_fields(candidate, row)
        candidates.append(candidate)

    return {
        "source_file": source_file,
        **_trade_permission_metadata(model_predictions),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_daily_watchlist_candidates(
    pattern_frame: pd.DataFrame,
    *,
    model_predictions: pd.DataFrame,
    pattern_source_file: str,
    model_source_file: str,
    model_top_n: int = 20,
) -> dict[str, object]:
    _ = model_top_n  # kept for backward-compatible callers; model TopN is no longer used for daily selection.
    pattern_payload = build_watchlist_candidates_from_patterns(
        pattern_frame,
        source_file=pattern_source_file,
        limit=None,
        model_predictions=model_predictions,
    )
    candidates = pattern_payload.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    return {
        "source_file": pattern_source_file,
        "model_source_file": model_source_file,
        **_trade_permission_metadata(model_predictions),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_watchlist_candidates_from_trend(
    trend_frame: pd.DataFrame,
    *,
    source_file: str,
    thresholds: PickTrendWatchlistConfig,
    limit: int = 30,
) -> dict[str, object]:
    frame = trend_frame.copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    required = {"symbol", "name", "buy_score", "price_action_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Trend frame is missing required columns: {missing}")

    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame = frame[
        pd.to_numeric(frame["buy_score"], errors="coerce").ge(thresholds.buy_score_min)
        & pd.to_numeric(frame["price_action_score"], errors="coerce").ge(thresholds.price_action_score_min)
    ].copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    frame = frame[~frame.apply(_is_row_risk_excluded, axis=1)].copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    sort_columns = [
        column
        for column in ("buy_score", "price_action_score", "trend_score", "trend_base_score", "entry_score")
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    candidates: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        candidate = {
            "source": "trend",
            "symbol": row["symbol"],
            "name": row.get("name", ""),
            "signal_type": row.get("signal_type", ""),
            "buy_score": round(float(row["buy_score"]), 4),
            "price_action_score": round(float(row["price_action_score"]), 4),
        }
        _append_supported_fields(candidate, row)
        candidates.append(candidate)

    return {
        "source_file": source_file,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_watchlist_candidates(project_root: Path, limit: int = 30) -> dict[str, object]:
    patterns_file = _latest_patterns_file(project_root)
    frame = pd.read_csv(patterns_file)
    return build_watchlist_candidates_from_patterns(frame, source_file=str(patterns_file), limit=limit)


def apply_trend_filter_to_watchlist_payload(
    payload: dict[str, object],
    *,
    trend_frame: pd.DataFrame,
    trend_filter: WatchlistTrendFilterConfig,
) -> dict[str, object]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return {
            "source_file": payload.get("source_file"),
            "candidate_count": 0,
            "candidates": [],
        }

    frame = trend_frame.copy()
    required = {"symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Trend report is missing required columns: {missing}")
    if frame.empty:
        return {
            "source_file": payload.get("source_file"),
            "candidate_count": 0,
            "candidates": [],
        }
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)

    if "in_trend_universe" in frame.columns:
        frame = frame[frame["in_trend_universe"].fillna(False).astype(bool)].copy()
    sort_columns = [
        column
        for column in ("trend_universe_score", "trend_score", "buy_score", "price_action_score")
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    frame = frame.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    trend_by_symbol = {str(row["symbol"]).zfill(6): row for row in frame.to_dict("records")}

    filtered: list[dict[str, object]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = _normalize_symbol(item.get("symbol", ""))
        trend_row = trend_by_symbol.get(symbol)
        if trend_row is None:
            continue
        if _is_row_risk_excluded(trend_row):
            continue

        enriched = deepcopy(item)
        enriched["symbol"] = symbol
        _append_supported_fields(enriched, trend_row)
        filtered.append(enriched)

    return {
        "source_file": payload.get("source_file"),
        "candidate_count": len(filtered),
        "candidates": filtered,
    }


def write_watchlist(
    *,
    project_root: Path,
    trade_date: date,
    picker_payload: dict[str, object],
    kind: str | None = None,
) -> Path:
    target = _resolve_watchlist_target(project_root, trade_date, kind=kind)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_watchlist_payload(trade_date=trade_date, picker_payload=picker_payload)
    if kind in {None, "pattern"}:
        payload = _attach_main_watchlist_streaks(project_root=project_root, trade_date=trade_date, payload=payload)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return target


def build_watchlist_payload(*, trade_date: date, picker_payload: dict[str, object]) -> dict[str, object]:
    payload = deepcopy(picker_payload)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    passthrough_fields = {
        "model_source_file",
        "trade_permission",
        "next_open_trade_permission",
        "next_open_trade_warning",
        "trade_permission_note",
    }
    passthrough = {key: payload[key] for key in passthrough_fields if key in payload}
    return {
        "trade_date": trade_date.isoformat(),
        "source_file": payload.get("source_file"),
        **passthrough,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def load_watchlist(*, project_root: Path, trade_date: date, kind: str | None = None) -> dict[str, object]:
    target = _resolve_watchlist_target(project_root, trade_date, kind=kind)
    if not target.exists():
        raise FileNotFoundError(f"Watchlist not found for {trade_date.isoformat()}: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def find_latest_watchlist_before(*, project_root: Path, trade_date: date) -> tuple[date, Path]:
    candidates: list[tuple[date, Path]] = []
    for path in watchlists_dir(project_root).glob("watchlist_*.json"):
        parsed = _parse_watchlist_date(path)
        if parsed is None or parsed >= trade_date:
            continue
        candidates.append((parsed, path))

    if not candidates:
        raise FileNotFoundError(f"No watchlist found before {trade_date.isoformat()} in {watchlists_dir(project_root)}")

    return max(candidates, key=lambda item: item[0])


def extract_watchlist_symbols(payload: dict[str, object]) -> list[str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []

    symbols: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().zfill(6)
        if symbol == "000000" or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _parse_watchlist_date(path: Path) -> date | None:
    match = WATCHLIST_FILENAME_RE.fullmatch(path.name)
    if not match:
        return None
    return datetime.fromisoformat(match.group(1)).date()


def _latest_patterns_file(project_root: Path) -> Path:
    files = sorted(
        (project_root / "reports" / "patterns").glob("patterns_all_*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No reports/patterns/patterns_all_*.csv file found. Run mystock pattern first.")
    return files[0]


def _normalize_symbol(value: object) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    return text.zfill(6)


def _daily_rating_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("tradingview_all_rating_20"))


def _stable_score(row: pd.Series, daily_columns: list[str]) -> float:
    avg_score = float(row.get("tradingview_avg_all_rating_5d", 0.0))
    latest_score = float(row.get(daily_columns[-1], avg_score)) if daily_columns else avg_score
    pattern_id = str(row.get("pattern_id", ""))
    label = str(row.get("tradingview_all_rating_label", "")).strip().lower()
    return round(
        avg_score * 4.0
        + latest_score * 1.5
        + PATTERN_PRIORITY.get(pattern_id, 0.0)
        + LABEL_BONUS.get(label, -1.0),
        4,
    )


def _base_tier(row: pd.Series) -> str | None:
    avg_score = float(row.get("tradingview_avg_all_rating_5d", 0.0))
    pattern_id = str(row.get("pattern_id", ""))
    label = str(row.get("tradingview_all_rating_label", "")).strip().lower()

    if label not in {"buy", "strong_buy"}:
        return None
    if avg_score < 0.22:
        return None

    if pattern_id in {"1", "2", "3"} and avg_score >= 0.35:
        return "第一梯队"
    if pattern_id in {"4", "5", "6"} and avg_score >= 0.36:
        return "第一梯队"
    if pattern_id in {"1", "2", "3"} and avg_score >= 0.28:
        return "第二梯队"
    if pattern_id in {"4", "5", "6"} and avg_score >= 0.32:
        return "第二梯队"

    return "第三梯队"


def _prepare_model_predictions_for_join(model_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = model_predictions.copy()
    required = V42_PREDICT_MODEL_REQUIRED_FIELDS
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Predict model frame is missing required columns: {missing}")
    if frame.empty:
        return frame

    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    if "risk_score" in frame.columns:
        frame["_risk_sort_score"] = pd.to_numeric(frame["risk_score"], errors="coerce")
        frame = frame.sort_values("_risk_sort_score", ascending=True)
    frame = frame.drop_duplicates(subset=["symbol"], keep="first")
    frame = frame.drop(columns=["_risk_sort_score"], errors="ignore")
    if "trade_date" in frame.columns:
        frame = frame.rename(columns={"trade_date": "model_trade_date"})
    columns = []
    for column in ("symbol", "name", "model_trade_date", *PREDICT_MODEL_CANDIDATE_FIELDS):
        if column in frame.columns and column not in columns:
            columns.append(column)
    return frame.loc[:, columns].copy()


def _trade_permission_metadata(model_predictions: pd.DataFrame) -> dict[str, object]:
    if model_predictions.empty or "trade_permission" not in model_predictions.columns:
        return {
            "trade_permission": "unknown",
            "next_open_trade_permission": "unknown",
            "next_open_trade_warning": True,
            "trade_permission_note": "未找到模型交易许可字段，候选仅供观察。",
        }
    permission_values = (
        model_predictions["trade_permission"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if permission_values.empty:
        permission = "unknown"
    elif permission_values.eq("allow").all():
        permission = "allow"
    elif permission_values.eq("no_trade").any():
        permission = "no_trade"
    else:
        permission = permission_values.iloc[0]
    warning = permission != "allow"
    note = (
        "模型判断次日开盘环境允许选股。"
        if permission == "allow"
        else "模型判断次日开盘环境不适合买入，仅保留低风险候选供观察。"
    )
    return {
        "trade_permission": permission,
        "next_open_trade_permission": permission,
        "next_open_trade_warning": warning,
        "trade_permission_note": note,
    }


def _is_model_low_risk_row(row: pd.Series | dict[str, object]) -> bool:
    final_action = str(_row_value(row, "final_action", "")).strip().lower()
    action = str(_row_value(row, "action", "")).strip().lower()
    risk_candidate_action = str(_row_value(row, "risk_candidate_action", "")).strip().lower()
    risk_action = str(_row_value(row, "risk_action", "")).strip().lower()
    risk_tier = str(_row_value(row, "risk_tier", "")).strip().lower()
    if final_action == "avoid" or action == "avoid" or risk_tier == "high":
        return False
    if risk_candidate_action in {"candidate", "pass", "low_risk"}:
        return True
    if risk_action in {"pass", "candidate", "low_risk"}:
        return True
    if risk_tier in {"low", "medium", "中", "低"}:
        return True
    return action == "candidate"


def _pattern_watchlist_sort_score(row: pd.Series | dict[str, object]) -> float:
    score = PATTERN_PRIORITY.get(str(_row_value(row, "pattern_id", "")), 0.0)
    risk_score = _float_or_default(_row_value(row, "risk_score", math.inf), math.inf)
    if math.isfinite(risk_score):
        score += max(0.0, 1.0 - risk_score)
    if _is_truthy_flag(_row_value(row, "macd_bottom_divergence_15d", False)):
        score += 0.6
    if str(_row_value(row, "macd_cross_state", "")).strip().lower() == "golden_cross":
        score += 0.4
    if str(_row_value(row, "macd_divergence_state", "")).strip().lower() == "bottom_divergence":
        score += 0.4
    if str(_row_value(row, "volume_price_divergence_state", "")).strip().lower() == "bullish":
        score += 0.4
    if _is_truthy_flag(_row_value(row, "bullish_volume_price_divergence_flag", False)):
        score += 0.4
    atr_pct = _float_or_default(_row_value(row, "ATR%", _row_value(row, "atr_pct_14", math.nan)), math.nan)
    if math.isfinite(atr_pct):
        if atr_pct > 1:
            atr_pct = atr_pct / 100.0
        score -= min(max(atr_pct, 0.0), 0.2)
    return round(score, 4)


def _model_base_tier(row: pd.Series) -> str:
    pattern_id = str(row.get("pattern_id", ""))
    if pattern_id in {"5", "1"}:
        return "第一梯队"
    if pattern_id in {"6", "3"}:
        return "第二梯队"
    return "第三梯队"


def _is_true_index_name(name: object) -> bool:
    normalized = str(name).strip().replace(" ", "")
    return any(marker in normalized for marker in INDEX_NAME_MARKERS)


def _resolve_watchlist_target(project_root: Path, trade_date: date, kind: str | None) -> Path:
    if kind == "pattern":
        return watchlist_pattern_path(project_root, trade_date)
    if kind == "trend":
        return watchlist_trend_path(project_root, trade_date)
    return watchlist_path(project_root, trade_date)


def _append_supported_fields(candidate: dict[str, object], row: pd.Series | dict[str, object]) -> None:
    for field in PATTERN_CANDIDATE_FIELDS:
        value = _row_value(row, field, pd.NA)
        if pd.isna(value):
            continue
        candidate[field] = _normalize_candidate_value(value)

    for field in PREDICT_MODEL_CANDIDATE_FIELDS:
        value = _row_value(row, field, pd.NA)
        if pd.isna(value):
            continue
        candidate[field] = _normalize_candidate_value(value)

    for field in TREND_UNIVERSE_CANDIDATE_FIELDS + TREND_CANDIDATE_FIELDS:
        value = _row_value(row, field, pd.NA)
        if pd.isna(value):
            continue
        candidate[field] = _normalize_candidate_value(value)

    for source_field, target_field in ATR_WATCHLIST_FIELD_MAP:
        value = _row_value(row, source_field, pd.NA)
        if pd.isna(value):
            continue
        if source_field == "atr_pct_14":
            candidate[target_field] = round(float(value) * 100.0, 4)
        else:
            candidate[target_field] = _normalize_candidate_value(value)


def _attach_main_watchlist_streaks(
    *,
    project_root: Path,
    trade_date: date,
    payload: dict[str, object],
) -> dict[str, object]:
    enriched = deepcopy(payload)
    candidates = enriched.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return enriched

    previous_streaks = _load_previous_main_watchlist_streaks(project_root=project_root, trade_date=trade_date)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = _normalize_symbol(item.get("symbol", ""))
        if symbol == "000000":
            continue
        item[WATCHLIST_STREAK_FIELD] = int(previous_streaks.get(symbol, 0)) + 1
    return enriched


def _load_previous_main_watchlist_streaks(*, project_root: Path, trade_date: date) -> dict[str, int]:
    try:
        previous_date, _ = find_latest_watchlist_before(project_root=project_root, trade_date=trade_date)
    except FileNotFoundError:
        return {}

    previous_payload = load_watchlist(project_root=project_root, trade_date=previous_date)
    candidates = previous_payload.get("candidates")
    if not isinstance(candidates, list):
        return {}

    streaks: dict[str, int] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = _normalize_symbol(item.get("symbol", ""))
        if symbol == "000000":
            continue
        raw_value = item.get(WATCHLIST_STREAK_FIELD, 1)
        try:
            streaks[symbol] = int(raw_value)
        except (TypeError, ValueError):
            streaks[symbol] = 1
    return streaks


def _is_row_risk_excluded(row: pd.Series | dict[str, object]) -> bool:
    macd_cross_state = str(_row_value(row, "macd_cross_state", "")).strip().lower()
    macd_divergence_state = str(_row_value(row, "macd_divergence_state", "")).strip().lower()
    volume_price_divergence_state = str(_row_value(row, "volume_price_divergence_state", "")).strip().lower()

    if macd_cross_state == "dead_cross":
        return True
    if macd_divergence_state == "top_divergence":
        return True
    if volume_price_divergence_state == "bearish":
        return True
    if _is_truthy_flag(_row_value(row, "macd_top_divergence_15d", False)):
        return True
    if _is_truthy_flag(_row_value(row, "bearish_volume_price_divergence_flag", False)):
        return True
    return False


def _row_value(row: pd.Series | dict[str, object], key: str, default: object) -> object:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _is_truthy_flag(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _float_or_default(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _normalize_candidate_value(value: object) -> object:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, pd.Timestamp):
        return _serialize_temporal_value(value)
    if isinstance(value, datetime):
        return _serialize_temporal_value(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 4)
    return value


def _json_default(value: object) -> object:
    normalized = _normalize_candidate_value(value)
    if normalized is value:
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
    return normalized


def _serialize_temporal_value(value: datetime) -> str:
    if (
        value.hour == 0
        and value.minute == 0
        and value.second == 0
        and value.microsecond == 0
    ):
        return value.date().isoformat()
    return value.isoformat()
