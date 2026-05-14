from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .atr import ATR_WATCHLIST_FIELD_MAP, normalize_atr_summary_frame
from .phase4_rolling import PHASE4_ROLLING_COLUMNS, PHASE4_ROLLING_RANK_COLUMNS, merge_phase4_rolling_frame
from .phase_display import PHASE_TABLE_DROP_COLUMNS, add_phase5_score_100, phase7_score_100, score_series_100
from .position_sizing import RECOMMENDED_POSITION_PERCENT_FIELD, recommended_position_percent_from_mapping


WATCHLIST_FILENAME_RE = re.compile(r"watchlist_(\d{4}-\d{2}-\d{2})\.json$")
WATCHLIST_STREAK_FIELD = "连续上榜天数"
LIMIT_UP_DAILY_RETURN_THRESHOLD = 0.099
CENTERED_RISK_PHASE_TARGET = 80.0
CENTERED_RISK_PHASE_WIDTH_MULT = 2.0
CENTERED_RISK_PHASE1_WEIGHT = 0.08
CENTERED_RISK_PHASE2_WEIGHT = 0.12
CENTERED_RISK_TOP_MIN_PHASE1_SCORE = 40.0
CENTERED_RISK_TOP_MIN_PHASE2_SCORE = 50.0
CENTERED_RISK_TOP_MIN_PHASE4_SCORE = 70.0
PATTERN_MIN_PHASE4_SCORE = 70.0
PHASE8_TOP_N = 5
INTRADAY_POOL_SIZE = 100
INTRADAY_POOL_PATTERN_LIMIT = 30
INTRADAY_POOL_P124_TOP_N = 50
PATTERN_PRIORITY = {
    "5": 6.0,
    "1": 5.0,
    "6": 4.0,
    "3": 3.0,
    "2": 2.0,
    "4": 1.0,
}
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
TECHNICAL_CANDIDATE_FIELDS = (
    "macd",
    "macd_signal_line",
    "macd_hist",
    "macd_cross_state",
    "macd_divergence_state",
    "volume_price_divergence_state",
    "macd_top_divergence_15d",
    "macd_bottom_divergence_15d",
    "macd_top_divergence_signal_date",
    "macd_bottom_divergence_signal_date",
    "bullish_volume_price_divergence_flag",
    "bearish_volume_price_divergence_flag",
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


def intraday_pool_path(project_root: Path, trade_date: date) -> Path:
    return watchlists_dir(project_root) / f"intraday_pool_{trade_date.isoformat()}.json"


def build_watchlist_candidates_from_patterns(
    pattern_frame: pd.DataFrame,
    *,
    source_file: str,
    limit: int | None = 30,
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

    required = {"symbol", "name", "pattern_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Latest patterns frame is missing required columns: {missing}")

    frame["stable_score"] = frame.apply(_pattern_watchlist_sort_score, axis=1)
    frame["base_tier"] = frame.apply(_model_base_tier, axis=1)
    frame = frame.dropna(subset=["base_tier"]).copy()
    frame = frame[~frame.apply(_is_row_risk_excluded, axis=1)].copy()
    frame["pattern_priority"] = frame["pattern_id"].astype(str).map(PATTERN_PRIORITY).fillna(0.0)
    frame = frame.sort_values(["base_tier", "stable_score", "pattern_priority", "symbol"], ascending=[True, False, False, True])

    candidates: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        candidate = {
            "tier": row["base_tier"],
            "symbol": row["symbol"],
            "name": row["name"],
            "pattern_id": str(row["pattern_id"]),
            "macd_top_divergence_15d": bool(row.get("macd_top_divergence_15d", False)),
            "macd_bottom_divergence_15d": bool(row.get("macd_bottom_divergence_15d", False)),
            "stable_score": round(float(row["stable_score"]), 4),
            "reason": row.get("reason", ""),
        }
        for field in (
            "phase1_score_100",
            "phase2_score_100",
            "phase2_is_cusum_event",
            "phase4_score_100",
            *PHASE4_ROLLING_COLUMNS,
            *PHASE4_ROLLING_RANK_COLUMNS,
            "phase8_score_100",
            "phase8_rank",
            "today_limit_up_excluded",
            "phase5_score_100",
            "phase7_score_100",
            "phase7_trade_permission",
            "daily_return_1d",
            "涨幅%",
        ):
            _copy_candidate_field(candidate, row, field)
        _append_supported_fields(candidate, row)
        candidates.append(candidate)

    return {
        "source_file": source_file,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_phase_daily_watchlist_candidates(
    *,
    trade_date: date,
    pattern_frame: pd.DataFrame,
    phase1_predictions: pd.DataFrame,
    phase2_predictions: pd.DataFrame,
    phase4_predictions: pd.DataFrame,
    phase7_prediction: pd.DataFrame,
    phase8_predictions: pd.DataFrame | None = None,
    phase5_measures: pd.DataFrame | None = None,
    macd_frame: pd.DataFrame | None = None,
    atr_frame: pd.DataFrame | None = None,
    source_files: dict[str, str] | None = None,
    phase_filter_rate: float = 0.2,
    phase4_top_n: int = 20,
) -> dict[str, object]:
    phase1 = _prepare_phase_risk_predictions(
        phase1_predictions,
        score_column="risk_score",
        output_score_column="phase1_risk_score",
        prefix="phase1",
        filter_rate=phase_filter_rate,
        model_prefix="phase1",
        extra_columns=("log_return_1d",),
    )
    phase2 = _prepare_phase_risk_predictions(
        phase2_predictions,
        score_column="barrier_risk_score",
        output_score_column="phase2_barrier_risk_score",
        prefix="phase2",
        filter_rate=phase_filter_rate,
        model_prefix="phase2",
        extra_columns=("is_cusum_event", "mlfin_daily_vol", "mlfin_cusum_threshold"),
    )
    phase4 = _prepare_phase4_predictions(phase4_predictions)
    phase4 = merge_phase4_rolling_frame(
        phase4,
        project_root=_project_root_from_source_files(source_files),
        trade_date=trade_date,
    )
    phase8 = _prepare_phase8_predictions(phase8_predictions)
    phase5 = _prepare_phase5_measures(phase5_measures, trade_date=trade_date)
    macd = _prepare_macd_frame(macd_frame)
    atr = _prepare_atr_frame(atr_frame)
    pattern_groups = _prepare_pattern_groups(pattern_frame)
    phase7 = _phase7_metadata(phase7_prediction)

    if phase1.empty:
        raise RuntimeError("Phase 1 predictions are empty; cannot build phase watchlist.")
    if phase2.empty:
        raise RuntimeError("Phase 2 predictions are empty; cannot build phase watchlist.")
    if phase4.empty:
        raise RuntimeError("Phase 4 predictions are empty; cannot build phase watchlist.")

    pool = phase1.merge(phase2, on="symbol", how="inner")
    pool = pool.merge(phase4, on="symbol", how="left")
    if not phase8.empty:
        pool = pool.merge(phase8, on="symbol", how="left")
    if not phase5.empty:
        pool = pool.merge(phase5, on="symbol", how="left")
    if not macd.empty:
        pool = pool.merge(macd, on="symbol", how="left")
    if not atr.empty:
        pool = pool.merge(atr, on="symbol", how="left")

    if "name" not in pool.columns:
        pool["name"] = ""
    if "phase4_name" in pool.columns:
        pool["name"] = pool["name"].where(pool["name"].astype(str).str.strip().ne(""), pool["phase4_name"])
    if pattern_groups:
        pattern_names = {symbol: str(group.iloc[0].get("name", "")) for symbol, group in pattern_groups.items()}
        pool["name"] = pool.apply(
            lambda row: row["name"] if str(row.get("name", "")).strip() else pattern_names.get(str(row["symbol"]), ""),
            axis=1,
        )

    evaluated = pool.copy()
    evaluated["daily_return_1d"] = _derive_daily_return_1d(evaluated)
    daily_return_values = pd.to_numeric(evaluated["daily_return_1d"], errors="coerce")
    evaluated["涨幅%"] = (daily_return_values * 100.0).round(4)
    evaluated["limit_up_excluded_by_daily_return"] = evaluated["daily_return_1d"].gt(LIMIT_UP_DAILY_RETURN_THRESHOLD).fillna(False)
    for score_column in ("phase1_score_100", "phase2_score_100", "phase4_score_100"):
        if score_column in evaluated.columns:
            evaluated[score_column] = pd.to_numeric(evaluated[score_column], errors="coerce")
        else:
            evaluated[score_column] = pd.NA
    evaluated = add_centered_risk_scores(evaluated)
    evaluated["phase4_composite_score"] = evaluated["centered_risk_score"]
    evaluated["phase4_top_score_filter_pass"] = False
    pattern_symbol_set = set(pattern_groups)
    evaluated["pattern_score_filter_pass_before_limit_up"] = (
        evaluated["symbol"].astype(str).str.zfill(6).isin(pattern_symbol_set)
        & evaluated["phase4_score_100"].gt(PATTERN_MIN_PHASE4_SCORE)
    )
    evaluated["pattern_score_filter_pass"] = (
        evaluated["pattern_score_filter_pass_before_limit_up"].fillna(False).astype(bool)
        & ~evaluated["limit_up_excluded_by_daily_return"].fillna(False).astype(bool)
    )

    hard_filter_mask = (
        ~evaluated["phase1_excluded_by_top20_risk"].fillna(True).astype(bool)
        & ~evaluated["phase2_excluded_by_top20_risk"].fillna(True).astype(bool)
    )
    hard_filter_pass_count_before_limit_up = int(hard_filter_mask.sum())
    limit_up_candidate_mask = hard_filter_mask | evaluated["pattern_score_filter_pass_before_limit_up"].fillna(False).astype(bool)
    limit_up_excluded = int((limit_up_candidate_mask & evaluated["limit_up_excluded_by_daily_return"].fillna(False).astype(bool)).sum())
    passed = evaluated[hard_filter_mask & ~evaluated["limit_up_excluded_by_daily_return"].fillna(False).astype(bool)].copy()
    for score_column in ("phase1_score_100", "phase2_score_100", "phase4_score_100"):
        if score_column in passed.columns:
            passed[score_column] = pd.to_numeric(passed[score_column], errors="coerce")
        else:
            passed[score_column] = pd.NA
    passed["phase4_top_score_filter_pass"] = (
        passed["phase1_score_100"].ge(CENTERED_RISK_TOP_MIN_PHASE1_SCORE)
        & passed["phase2_score_100"].ge(CENTERED_RISK_TOP_MIN_PHASE2_SCORE)
        & passed["phase4_score_100"].ge(CENTERED_RISK_TOP_MIN_PHASE4_SCORE)
        & passed["centered_risk_score"].notna()
    )
    phase4_top_pool = passed[passed["phase4_top_score_filter_pass"].fillna(False)].copy()
    phase4_top_pool = phase4_top_pool.sort_values(
        ["centered_risk_score", "phase4_score_100", "phase1_center_score", "phase2_center_score", "symbol"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    phase4_top_pool["phase4_composite_rank"] = range(1, len(phase4_top_pool) + 1)
    phase4_composite_rank_by_symbol = phase4_top_pool.set_index("symbol")["phase4_composite_rank"].to_dict()
    passed["phase4_composite_rank"] = passed["symbol"].map(phase4_composite_rank_by_symbol)
    evaluated["phase4_composite_rank"] = evaluated["symbol"].map(phase4_composite_rank_by_symbol)

    pattern_pool = evaluated[evaluated["pattern_score_filter_pass"].fillna(False)].copy()
    row_pool = pd.concat([pattern_pool, passed], ignore_index=True).drop_duplicates("symbol", keep="first")
    row_by_symbol = {str(row["symbol"]).zfill(6): row for row in row_pool.to_dict("records")}
    pattern_symbols = pattern_pool["symbol"].astype(str).str.zfill(6).tolist()
    phase4_top_symbols = phase4_top_pool.head(max(int(phase4_top_n), 0))["symbol"].astype(str).str.zfill(6).tolist()
    p8_top_pool = evaluated.copy()
    if "phase8_score_100" in p8_top_pool.columns:
        p8_top_pool["phase8_score_100"] = pd.to_numeric(p8_top_pool["phase8_score_100"], errors="coerce")
        p8_top_pool = p8_top_pool[
            p8_top_pool["phase8_score_100"].notna()
            & ~p8_top_pool.get("today_limit_up_excluded", pd.Series(False, index=p8_top_pool.index)).fillna(False).astype(bool)
            & ~p8_top_pool["limit_up_excluded_by_daily_return"].fillna(False).astype(bool)
        ].copy()
        p8_top_pool = p8_top_pool.sort_values(["phase8_score_100", "phase4_score_100", "symbol"], ascending=[False, False, True], na_position="last")
    else:
        p8_top_pool = evaluated.head(0).copy()
    p8_top_symbols = p8_top_pool.head(PHASE8_TOP_N)["symbol"].astype(str).str.zfill(6).tolist()
    p8_top_rank_by_symbol = {symbol: index + 1 for index, symbol in enumerate(p8_top_symbols)}

    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for symbol in pattern_symbols:
        if symbol in seen:
            continue
        row = row_by_symbol.get(symbol)
        if row is None:
            continue
        candidates.append(
            _phase_watchlist_candidate(
                row,
                source="pattern",
                source_tags=["pattern"],
                pattern_group=pattern_groups.get(symbol),
                phase7=phase7,
            )
        )
        seen.add(symbol)

    phase4_added = 0
    for symbol in phase4_top_symbols:
        if symbol in seen:
            continue
        row = row_by_symbol.get(symbol)
        if row is None:
            continue
        candidates.append(
            _phase_watchlist_candidate(
                row,
                source="phase4_top",
                source_tags=["phase4_top"],
                pattern_group=pattern_groups.get(symbol),
                phase7=phase7,
            )
        )
        seen.add(symbol)
        phase4_added += 1

    p8_added = 0
    for symbol in p8_top_symbols:
        row = row_by_symbol.get(symbol)
        if row is None:
            row_matches = evaluated[evaluated["symbol"].astype(str).str.zfill(6).eq(symbol)]
            if row_matches.empty:
                continue
            row = row_matches.iloc[0].to_dict()
        if symbol in seen:
            _append_candidate_source_tag(candidates, symbol=symbol, tag="p8_top5")
            continue
        candidate = _phase_watchlist_candidate(
            row,
            source="p8_top5",
            source_tags=["p8_top5"],
            pattern_group=pattern_groups.get(symbol),
            phase7=phase7,
        )
        candidate["phase8_top5_rank"] = p8_top_rank_by_symbol.get(symbol)
        candidates.append(candidate)
        seen.add(symbol)
        p8_added += 1

    candidates = sorted(
        candidates,
        key=lambda item: (
            0 if item.get("pattern_match") else 1,
            0 if "p8_top5" in item.get("source_tags", []) else 1,
            -_float_or_default(
                item.get("phase4_composite_score"),
                _float_or_default(item.get("phase4_score_100"), -math.inf),
            ),
            str(item.get("symbol", "")),
        ),
    )

    phase1_excluded = int(phase1["phase1_excluded_by_top20_risk"].fillna(False).astype(bool).sum())
    phase2_excluded = int(phase2["phase2_excluded_by_top20_risk"].fillna(False).astype(bool).sum())
    return {
        "source_file": (source_files or {}).get("pattern"),
        "model_source_files": source_files or {},
        "selection_policy": {
            "phase1_filter": "exclude highest risk 20%",
            "phase2_filter": "exclude highest risk 20%",
            "phase_filter_rate": float(phase_filter_rate),
            "phase4_top_n": int(phase4_top_n),
            "centered_risk_min_phase1_score": float(CENTERED_RISK_TOP_MIN_PHASE1_SCORE),
            "centered_risk_min_phase2_score": float(CENTERED_RISK_TOP_MIN_PHASE2_SCORE),
            "centered_risk_min_phase4_score": float(CENTERED_RISK_TOP_MIN_PHASE4_SCORE),
            "pattern_min_phase4_score": float(PATTERN_MIN_PHASE4_SCORE),
            "pattern_filter": "pattern hits ignore phase1/phase2 floors; require phase4_score_100 > 70 and same-day return <= 9.9%",
            "centered_risk_sort_formula": "phase4_score_100 + 0.08 * max(0, 100 - 2 * abs(phase1_score_100 - 80)) + 0.12 * max(0, 100 - 2 * abs(phase2_score_100 - 80))",
            "phase4_top_sort_formula": "centered_risk_score",
            "phase8_policy": "display-only; append Phase8 Top5 as extra candidates/source tags when prediction file exists",
            "phase7_no_trade_policy": "block highest-risk 20% trade days based on trained threshold",
            "limit_up_filter": "exclude both pattern and phase4_top candidates with same-day return > 9.9% before watchlist selection",
            "limit_up_filter_threshold": float(LIMIT_UP_DAILY_RETURN_THRESHOLD),
            "position_size_formula": "D = 2 * ATR14; effective average stop distance before third add = 0.85D; total planned position = min(40%, 2% / (0.85 * 2 * ATR14 / close))",
        },
        "trade_permission": phase7.get("phase7_trade_permission", "unknown"),
        "next_open_trade_permission": phase7.get("phase7_trade_permission", "unknown"),
        "next_open_trade_warning": phase7.get("phase7_trade_permission") != "allow",
        "trade_permission_note": _phase7_trade_permission_note(phase7),
        **phase7,
        "filter_summary": {
            "phase1_rows": int(len(phase1)),
            "phase1_excluded_top20": phase1_excluded,
            "phase2_rows": int(len(phase2)),
            "phase2_excluded_top20": phase2_excluded,
            "phase1_phase2_intersection": int(len(pool)),
            "hard_filter_pass_count_before_limit_up_filter": hard_filter_pass_count_before_limit_up,
            "limit_up_excluded_gt_9_9pct": limit_up_excluded,
            "hard_filter_pass_count": int(len(passed)),
            "pattern_symbols_after_filter": int(len(set(pattern_symbols))),
            "pattern_symbols_phase4_gt_70": int(len(set(pattern_symbols))),
            "phase4_top_candidates_after_score_floor": int(len(phase4_top_pool)),
            "phase4_top_n_added_count": int(phase4_added),
            "phase8_rows": int(len(phase8)),
            "phase8_top5_symbols": p8_top_symbols,
            "phase8_top5_added_count": int(p8_added),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_intraday_pool_candidates(
    *,
    trade_date: date,
    pattern_frame: pd.DataFrame,
    phase1_predictions: pd.DataFrame,
    phase2_predictions: pd.DataFrame,
    phase4_predictions: pd.DataFrame,
    phase7_prediction: pd.DataFrame,
    phase8_predictions: pd.DataFrame | None = None,
    phase5_measures: pd.DataFrame | None = None,
    macd_frame: pd.DataFrame | None = None,
    atr_frame: pd.DataFrame | None = None,
    source_files: dict[str, str] | None = None,
    phase_filter_rate: float = 0.2,
    pattern_limit: int = INTRADAY_POOL_PATTERN_LIMIT,
    p124_top_n: int = INTRADAY_POOL_P124_TOP_N,
    pool_size: int = INTRADAY_POOL_SIZE,
) -> dict[str, object]:
    phase1 = _prepare_phase_risk_predictions(
        phase1_predictions,
        score_column="risk_score",
        output_score_column="phase1_risk_score",
        prefix="phase1",
        filter_rate=phase_filter_rate,
        model_prefix="phase1",
        extra_columns=("log_return_1d",),
    )
    phase2 = _prepare_phase_risk_predictions(
        phase2_predictions,
        score_column="barrier_risk_score",
        output_score_column="phase2_barrier_risk_score",
        prefix="phase2",
        filter_rate=phase_filter_rate,
        model_prefix="phase2",
        extra_columns=("is_cusum_event", "mlfin_daily_vol", "mlfin_cusum_threshold"),
    )
    phase4 = _prepare_phase4_predictions(phase4_predictions)
    phase4 = merge_phase4_rolling_frame(
        phase4,
        project_root=_project_root_from_source_files(source_files),
        trade_date=trade_date,
    )
    phase8 = _prepare_phase8_predictions(phase8_predictions)
    phase5 = _prepare_phase5_measures(phase5_measures, trade_date=trade_date)
    macd = _prepare_macd_frame(macd_frame)
    atr = _prepare_atr_frame(atr_frame)
    pattern_groups = _prepare_pattern_groups(pattern_frame)
    phase7 = _phase7_metadata(phase7_prediction)

    if phase1.empty:
        raise RuntimeError("Phase 1 predictions are empty; cannot build intraday pool.")
    if phase2.empty:
        raise RuntimeError("Phase 2 predictions are empty; cannot build intraday pool.")
    if phase4.empty:
        raise RuntimeError("Phase 4 predictions are empty; cannot build intraday pool.")

    pool = phase1.merge(phase2, on="symbol", how="inner")
    pool = pool.merge(phase4, on="symbol", how="left")
    if not phase8.empty:
        pool = pool.merge(phase8, on="symbol", how="left")
    if not phase5.empty:
        pool = pool.merge(phase5, on="symbol", how="left")
    if not macd.empty:
        pool = pool.merge(macd, on="symbol", how="left")
    if not atr.empty:
        pool = pool.merge(atr, on="symbol", how="left")

    if "name" not in pool.columns:
        pool["name"] = ""
    if "phase4_name" in pool.columns:
        pool["name"] = pool["name"].where(pool["name"].astype(str).str.strip().ne(""), pool["phase4_name"])
    if pattern_groups:
        pattern_names = {symbol: str(group.iloc[0].get("name", "")) for symbol, group in pattern_groups.items()}
        pool["name"] = pool.apply(
            lambda row: row["name"] if str(row.get("name", "")).strip() else pattern_names.get(str(row["symbol"]), ""),
            axis=1,
        )

    evaluated = pool.copy()
    evaluated["daily_return_1d"] = _derive_daily_return_1d(evaluated)
    daily_return_values = pd.to_numeric(evaluated["daily_return_1d"], errors="coerce")
    evaluated["涨幅%"] = (daily_return_values * 100.0).round(4)
    evaluated["limit_up_excluded_by_daily_return"] = evaluated["daily_return_1d"].gt(LIMIT_UP_DAILY_RETURN_THRESHOLD).fillna(False)
    for score_column in ("phase1_score_100", "phase2_score_100", "phase4_score_100"):
        if score_column in evaluated.columns:
            evaluated[score_column] = pd.to_numeric(evaluated[score_column], errors="coerce")
        else:
            evaluated[score_column] = pd.NA
    evaluated = add_centered_risk_scores(evaluated)
    evaluated["phase4_composite_score"] = evaluated["centered_risk_score"]
    evaluated["phase4_top_score_filter_pass"] = False

    p124_pool = evaluated[
        evaluated["centered_risk_score"].notna()
        & ~evaluated["limit_up_excluded_by_daily_return"].fillna(False).astype(bool)
    ].copy()
    p124_pool = p124_pool.sort_values(
        ["centered_risk_score", "phase4_score_100", "phase1_center_score", "phase2_center_score", "symbol"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    p124_pool["phase4_composite_rank"] = range(1, len(p124_pool) + 1)
    rank_by_symbol = p124_pool.set_index("symbol")["phase4_composite_rank"].to_dict()
    evaluated["phase4_composite_rank"] = evaluated["symbol"].map(rank_by_symbol)
    p124_pool["phase4_composite_rank"] = p124_pool["symbol"].map(rank_by_symbol)

    pattern_rows: list[dict[str, object]] = []
    evaluated_by_symbol = {str(row["symbol"]).zfill(6): row for row in evaluated.to_dict("records")}
    for symbol, group in pattern_groups.items():
        row = dict(evaluated_by_symbol.get(symbol, {}))
        if not row:
            row = group.iloc[0].to_dict()
            row["symbol"] = symbol
            row["name"] = str(row.get("name", "") or "")
        row["_pattern_priority"] = max(PATTERN_PRIORITY.get(str(value), 0.0) for value in group["pattern_id"].dropna().astype(str).tolist()) if "pattern_id" in group.columns else 0.0
        pattern_rows.append(row)
    pattern_pool = pd.DataFrame(pattern_rows)
    if not pattern_pool.empty:
        if "phase4_score_100" not in pattern_pool.columns:
            pattern_pool["phase4_score_100"] = pd.NA
        pattern_pool["phase4_score_100"] = pd.to_numeric(pattern_pool["phase4_score_100"], errors="coerce")
        pattern_pool = pattern_pool.sort_values(
            ["phase4_score_100", "_pattern_priority", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        )

    p8_pool = evaluated.copy()
    if "phase8_score_100" in p8_pool.columns:
        p8_pool["phase8_score_100"] = pd.to_numeric(p8_pool["phase8_score_100"], errors="coerce")
        if "today_limit_up_excluded" in p8_pool.columns:
            today_limit_up = p8_pool["today_limit_up_excluded"].map(_truthy_flag)
        else:
            today_limit_up = pd.Series(False, index=p8_pool.index)
        p8_pool = p8_pool[
            p8_pool["phase8_score_100"].notna()
            & ~today_limit_up.fillna(False).astype(bool)
            & ~p8_pool["limit_up_excluded_by_daily_return"].fillna(False).astype(bool)
        ].copy()
        p8_pool = p8_pool.sort_values(["phase8_score_100", "phase4_score_100", "symbol"], ascending=[False, False, True], na_position="last")
    else:
        p8_pool = evaluated.head(0).copy()

    candidates: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_candidate(row: dict[str, object], *, source: str, tag: str) -> bool:
        symbol = _normalize_symbol(row.get("symbol", ""))
        if not symbol:
            return False
        if symbol in seen:
            _append_candidate_source_tag(candidates, symbol=symbol, tag=tag)
            return False
        candidate = _phase_watchlist_candidate(
            row,
            source=source,
            source_tags=[tag],
            pattern_group=pattern_groups.get(symbol),
            phase7=phase7,
        )
        candidates.append(candidate)
        seen.add(symbol)
        return True

    for row in pattern_pool.head(max(int(pattern_limit), 0)).to_dict("records") if not pattern_pool.empty else []:
        add_candidate(row, source="pattern_pool", tag="pattern_pool")

    for row in p124_pool.head(max(int(p124_top_n), 0)).to_dict("records"):
        add_candidate(row, source="p124_top50", tag="p124_top50")

    for row in p8_pool.to_dict("records"):
        if len(candidates) >= max(int(pool_size), 0):
            break
        added = add_candidate(row, source="p8_fill", tag="p8_fill")
        if added:
            candidates[-1]["phase8_pool_rank"] = int(pd.to_numeric(row.get("phase8_rank"), errors="coerce")) if pd.notna(row.get("phase8_rank", pd.NA)) else pd.NA

    for index, candidate in enumerate(candidates, start=1):
        candidate["intraday_pool_rank"] = index

    return {
        "source_file": (source_files or {}).get("pattern"),
        "model_source_files": source_files or {},
        "selection_policy": {
            "intraday_pool_size": int(pool_size),
            "pattern_pool_limit": int(pattern_limit),
            "pattern_pool_policy": "include all patterns_all hits; if over limit, keep highest phase4_score_100",
            "p124_top_n": int(p124_top_n),
            "p124_sort_formula": "phase4_score_100 + 0.08 * max(0, 100 - 2 * abs(phase1_score_100 - 80)) + 0.12 * max(0, 100 - 2 * abs(phase2_score_100 - 80))",
            "phase8_fill_policy": "fill remaining slots by phase8_score_100 after pattern_pool and p124_top50, excluding same-day limit-up rows",
            "limit_up_filter_threshold": float(LIMIT_UP_DAILY_RETURN_THRESHOLD),
        },
        "trade_permission": phase7.get("phase7_trade_permission", "unknown"),
        "next_open_trade_permission": phase7.get("phase7_trade_permission", "unknown"),
        "next_open_trade_warning": phase7.get("phase7_trade_permission") != "allow",
        "trade_permission_note": _phase7_trade_permission_note(phase7),
        **phase7,
        "filter_summary": {
            "phase1_rows": int(len(phase1)),
            "phase2_rows": int(len(phase2)),
            "phase4_rows": int(len(phase4)),
            "phase8_rows": int(len(phase8)),
            "pattern_symbols_total": int(len(pattern_groups)),
            "pattern_pool_count": int(min(len(pattern_pool), max(int(pattern_limit), 0))) if not pattern_pool.empty else 0,
            "p124_top50_count": int(min(len(p124_pool), max(int(p124_top_n), 0))),
            "phase8_fill_available": int(len(p8_pool)),
            "intraday_pool_count": int(len(candidates)),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _prepare_phase_risk_predictions(
    predictions: pd.DataFrame,
    *,
    score_column: str,
    output_score_column: str,
    prefix: str,
    filter_rate: float,
    model_prefix: str,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    if "symbol" not in frame.columns or score_column not in frame.columns:
        return pd.DataFrame()
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame[output_score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame = frame.dropna(subset=[output_score_column]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_values([output_score_column, "symbol"], ascending=[False, True]).drop_duplicates("symbol")
    removed_rows = max(1, int(math.ceil(len(frame) * float(filter_rate)))) if len(frame) else 0
    frame[f"{prefix}_risk_rank"] = range(1, len(frame) + 1)
    frame[f"{prefix}_risk_percentile"] = frame[output_score_column].rank(pct=True, method="max")
    frame[f"{prefix}_score_100"] = score_series_100(frame[output_score_column], higher_is_better=False)
    frame[f"{prefix}_excluded_by_top20_risk"] = False
    if removed_rows:
        frame.loc[frame.index[:removed_rows], f"{prefix}_excluded_by_top20_risk"] = True
    keep = [
        "symbol",
        f"{prefix}_score_100",
        output_score_column,
        f"{prefix}_risk_rank",
        f"{prefix}_risk_percentile",
        f"{prefix}_excluded_by_top20_risk",
    ]
    if "name" in frame.columns and prefix == "phase1":
        keep.insert(1, "name")
    if "feature_trade_date" in frame.columns:
        target = f"{prefix}_feature_trade_date"
        frame[target] = frame["feature_trade_date"]
        keep.append(target)
    for column in extra_columns:
        if column in frame.columns:
            target = f"{prefix}_{column}"
            frame[target] = frame[column]
            keep.append(target)
    for column in ("model_name", "model_version"):
        if column in frame.columns:
            target = f"{model_prefix}_{column}"
            frame[target] = frame[column]
            keep.append(target)
    return frame.loc[:, keep].copy()


def _derive_daily_return_1d(frame: pd.DataFrame) -> pd.Series:
    if "phase1_log_return_1d" in frame.columns:
        values = pd.to_numeric(frame["phase1_log_return_1d"], errors="coerce")
        return values.map(lambda value: math.expm1(float(value)) if pd.notna(value) and math.isfinite(float(value)) else pd.NA)
    for column in ("phase1_return_1d", "return_1d", "daily_return_1d", "pct_change"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if column == "pct_change" and values.abs().dropna().gt(1.5).any():
            return values / 100.0
        return values
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def add_centered_risk_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for score_column in ("phase1_score_100", "phase2_score_100", "phase4_score_100"):
        if score_column in result.columns:
            result[score_column] = pd.to_numeric(result[score_column], errors="coerce")
        else:
            result[score_column] = pd.NA
    result["phase1_center_score"] = _centered_phase_score(result["phase1_score_100"])
    result["phase2_center_score"] = _centered_phase_score(result["phase2_score_100"])
    result["centered_risk_score"] = (
        result["phase4_score_100"]
        + CENTERED_RISK_PHASE1_WEIGHT * result["phase1_center_score"]
        + CENTERED_RISK_PHASE2_WEIGHT * result["phase2_center_score"]
    ).round(4)
    return result


def _centered_phase_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    centered = 100.0 - CENTERED_RISK_PHASE_WIDTH_MULT * (numeric - CENTERED_RISK_PHASE_TARGET).abs()
    return centered.clip(lower=0.0, upper=100.0)


def _prepare_phase4_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions is None or predictions.empty or "symbol" not in predictions.columns or "return_score" not in predictions.columns:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame["phase4_return_score"] = pd.to_numeric(frame["return_score"], errors="coerce")
    frame = frame.dropna(subset=["phase4_return_score"]).sort_values(["phase4_return_score", "symbol"], ascending=[False, True])
    frame = frame.drop_duplicates("symbol").reset_index(drop=True)
    frame["phase4_rank"] = frame.index + 1
    frame["phase4_score_percentile"] = frame["phase4_return_score"].rank(pct=True, method="max")
    frame["phase4_score_100"] = score_series_100(frame["phase4_return_score"], higher_is_better=True)
    keep = ["symbol", "phase4_score_100", "phase4_return_score", "phase4_rank", "phase4_score_percentile"]
    if "name" in frame.columns:
        frame["phase4_name"] = frame["name"]
        keep.append("phase4_name")
    if "feature_trade_date" in frame.columns:
        frame["phase4_feature_trade_date"] = frame["feature_trade_date"]
        keep.append("phase4_feature_trade_date")
    for column in ("model_name", "model_version"):
        if column in frame.columns:
            target = f"phase4_{column}"
            frame[target] = frame[column]
            keep.append(target)
    return frame.loc[:, keep].copy()


def _prepare_phase8_predictions(predictions: pd.DataFrame | None) -> pd.DataFrame:
    if predictions is None or predictions.empty or "symbol" not in predictions.columns:
        return pd.DataFrame(columns=["symbol"])
    frame = predictions.copy()
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    if "phase8_score_100" in frame.columns:
        frame["phase8_score_100"] = pd.to_numeric(frame["phase8_score_100"], errors="coerce")
    elif "phase8_raw_score" in frame.columns:
        frame["phase8_raw_score"] = pd.to_numeric(frame["phase8_raw_score"], errors="coerce")
        frame["phase8_score_100"] = score_series_100(frame["phase8_raw_score"], higher_is_better=True)
    else:
        return pd.DataFrame(columns=["symbol"])
    if "phase8_raw_score" in frame.columns:
        frame["phase8_raw_score"] = pd.to_numeric(frame["phase8_raw_score"], errors="coerce")
    if "phase8_rank" in frame.columns:
        frame["phase8_rank"] = pd.to_numeric(frame["phase8_rank"], errors="coerce")
    else:
        frame = frame.sort_values(["phase8_score_100", "symbol"], ascending=[False, True], na_position="last")
        frame["phase8_rank"] = range(1, len(frame) + 1)
    if "today_limit_up_excluded" in frame.columns:
        frame["today_limit_up_excluded"] = frame["today_limit_up_excluded"].map(_truthy_flag)
    else:
        frame["today_limit_up_excluded"] = False
    rename = {
        "feature_trade_date": "phase8_feature_trade_date",
        "model_name": "phase8_model_name",
        "model_version": "phase8_model_version",
    }
    frame = frame.rename(columns={source: target for source, target in rename.items() if source in frame.columns})
    keep = [
        "symbol",
        "phase8_score_100",
        "phase8_raw_score",
        "phase8_rank",
        "today_limit_up_excluded",
        "today_high_return_vs_prev_close",
        "today_close_return_vs_prev_close",
        "phase8_feature_trade_date",
        "phase8_model_name",
        "phase8_model_version",
    ]
    return frame.loc[:, [column for column in keep if column in frame.columns]].drop_duplicates("symbol", keep="first").copy()


def _prepare_phase5_measures(measures: pd.DataFrame | None, *, trade_date: date) -> pd.DataFrame:
    if measures is None or measures.empty or "symbol" not in measures.columns:
        return pd.DataFrame()
    frame = measures.copy()
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame["year"] = pd.to_numeric(frame.get("year"), errors="coerce")
    frame = frame.dropna(subset=["year"]).copy()
    if frame.empty:
        return pd.DataFrame()
    eligible = frame[frame["year"].astype(int).le(trade_date.year)].copy()
    if eligible.empty:
        eligible = frame.copy()
    eligible = eligible.sort_values(["symbol", "year"]).drop_duplicates("symbol", keep="last")
    keep = ["symbol"]
    for column in ("year", "weeks", "NEGOUTLIER", "CRASH", "CRASH_count", "NCSKEW", "DUVOL", "RET", "SIGMA", "MINRET"):
        if column in eligible.columns:
            target = f"phase5_{column}"
            eligible[target] = eligible[column]
            keep.append(target)
    eligible = add_phase5_score_100(eligible)
    keep.insert(1, "phase5_score_100")
    return eligible.loc[:, keep].copy()


def _prepare_macd_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result["symbol"] = result["symbol"].map(_normalize_symbol)
    keep = [
        "symbol",
        "macd",
        "macd_signal_line",
        "macd_hist",
        "macd_cross_state",
        "macd_divergence_state",
        "volume_price_divergence_state",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "macd_top_divergence_signal_date",
        "macd_bottom_divergence_signal_date",
        "bullish_volume_price_divergence_flag",
        "bearish_volume_price_divergence_flag",
    ]
    return result.loc[:, [column for column in keep if column in result.columns]].drop_duplicates("symbol", keep="first")


def _prepare_atr_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = normalize_atr_summary_frame(frame)
    if "symbol" not in result.columns:
        return pd.DataFrame()
    result["symbol"] = result["symbol"].map(_normalize_symbol)
    keep = ["symbol", *(source for source, _ in ATR_WATCHLIST_FIELD_MAP)]
    return result.loc[:, [column for column in keep if column in result.columns]].drop_duplicates("symbol", keep="first")


def _prepare_pattern_groups(pattern_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if pattern_frame is None or pattern_frame.empty or "symbol" not in pattern_frame.columns:
        return {}
    frame = pattern_frame.copy()
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame["_pattern_priority"] = frame.get("pattern_id", pd.Series("", index=frame.index)).astype(str).map(PATTERN_PRIORITY).fillna(0.0)
    frame = frame.sort_values(["symbol", "_pattern_priority"], ascending=[True, False])
    return {symbol: group.drop(columns=["_pattern_priority"], errors="ignore").copy() for symbol, group in frame.groupby("symbol", sort=False)}


def _phase7_metadata(prediction: pd.DataFrame) -> dict[str, object]:
    if prediction is None or prediction.empty:
        return {
            "phase7_trade_permission": "unknown",
            "phase7_reason": "missing_phase7_prediction",
        }
    row = prediction.iloc[-1]
    mapping = {
        "feature_trade_date": "phase7_feature_trade_date",
        "buy_day_risk_score": "phase7_buy_day_risk_score",
        "selected_threshold": "phase7_selected_threshold",
        "trade_permission": "phase7_trade_permission",
        "suggested_action": "phase7_suggested_action",
        "reason": "phase7_reason",
        "model_name": "phase7_model_name",
        "model_version": "phase7_model_version",
    }
    result: dict[str, object] = {}
    for source, target in mapping.items():
        value = row.get(source, pd.NA)
        if _is_missing_value(value):
            continue
        result[target] = _normalize_candidate_value(value)
    result["phase7_score_100"] = phase7_score_100(result.get("phase7_trade_permission"))
    return result


def _phase7_trade_permission_note(phase7: dict[str, object]) -> str:
    permission = str(phase7.get("phase7_trade_permission", "unknown")).strip().lower()
    if permission == "allow":
        return "Phase7 判断当日买点环境允许交易。"
    if permission == "no_trade":
        return "Phase7 判断当日属于最高风险 20% 交易日，仅观察不新开仓。"
    return "Phase7 交易日闸门状态未知，仅供观察。"


def _phase_watchlist_candidate(
    row: dict[str, object],
    *,
    source: str,
    source_tags: list[str],
    pattern_group: pd.DataFrame | None,
    phase7: dict[str, object],
) -> dict[str, object]:
    symbol = _normalize_symbol(row.get("symbol", ""))
    candidate: dict[str, object] = {
        "source": source,
        "source_tags": source_tags,
        "symbol": symbol,
        "name": str(row.get("name", "") or ""),
        "pattern_match": pattern_group is not None and not pattern_group.empty,
    }
    candidate.update(phase7)
    for field in (
        "phase1_score_100",
        "phase1_risk_score",
        "phase1_log_return_1d",
        "daily_return_1d",
        "涨幅%",
        "limit_up_excluded_by_daily_return",
        "phase1_feature_trade_date",
        "phase1_risk_rank",
        "phase1_risk_percentile",
        "phase1_excluded_by_top20_risk",
        "phase1_model_name",
        "phase1_model_version",
        "phase2_score_100",
        "phase2_barrier_risk_score",
        "phase2_feature_trade_date",
        "phase2_risk_rank",
        "phase2_risk_percentile",
        "phase2_excluded_by_top20_risk",
        "phase2_is_cusum_event",
        "phase2_mlfin_daily_vol",
        "phase2_mlfin_cusum_threshold",
        "phase2_model_name",
        "phase2_model_version",
        "phase1_center_score",
        "phase2_center_score",
        "centered_risk_score",
        "phase4_score_100",
        *PHASE4_ROLLING_COLUMNS,
        *PHASE4_ROLLING_RANK_COLUMNS,
        "phase4_composite_score",
        "phase4_composite_rank",
        "phase4_top_score_filter_pass",
        "phase4_return_score",
        "phase4_feature_trade_date",
        "phase4_rank",
        "phase4_score_percentile",
        "phase4_model_name",
        "phase4_model_version",
        "phase8_score_100",
        "phase8_raw_score",
        "phase8_rank",
        "phase8_top5_rank",
        "today_limit_up_excluded",
        "today_high_return_vs_prev_close",
        "today_close_return_vs_prev_close",
        "phase8_feature_trade_date",
        "phase8_model_name",
        "phase8_model_version",
        "phase5_score_100",
        "phase5_year",
        "phase5_weeks",
        "phase5_NEGOUTLIER",
        "phase5_CRASH",
        "phase5_CRASH_count",
        "phase5_NCSKEW",
        "phase5_DUVOL",
        "phase5_RET",
        "phase5_SIGMA",
        "phase5_MINRET",
        "macd",
        "macd_signal_line",
        "macd_hist",
        "macd_cross_state",
        "macd_divergence_state",
        "volume_price_divergence_state",
        "macd_top_divergence_15d",
        "macd_bottom_divergence_15d",
        "macd_top_divergence_signal_date",
        "macd_bottom_divergence_signal_date",
        "bullish_volume_price_divergence_flag",
        "bearish_volume_price_divergence_flag",
    ):
        _copy_candidate_field(candidate, row, field)
    _append_supported_fields(candidate, row)

    if pattern_group is not None and not pattern_group.empty:
        first_pattern = pattern_group.iloc[0]
        candidate["pattern_ids"] = [str(value) for value in pattern_group["pattern_id"].dropna().astype(str).unique().tolist()]
        candidate["pattern_id"] = str(first_pattern.get("pattern_id", ""))
        candidate["patterns"] = [_compact_pattern_record(pattern_row) for _, pattern_row in pattern_group.iterrows()]
        reasons = [str(value).strip() for value in pattern_group.get("reason", pd.Series(dtype=str)).dropna().tolist() if str(value).strip()]
        if reasons:
            candidate["reason"] = " | ".join(dict.fromkeys(reasons))
        _append_supported_fields(candidate, first_pattern)
    return candidate


def _compact_pattern_record(row: pd.Series) -> dict[str, object]:
    record: dict[str, object] = {}
    for field in ("pattern_id", "reason", "close", *PATTERN_CANDIDATE_FIELDS):
        value = row.get(field, pd.NA)
        if _is_missing_value(value):
            continue
        record[field] = _normalize_candidate_value(value)
    return record


def _copy_candidate_field(candidate: dict[str, object], row: dict[str, object], field: str) -> None:
    value = row.get(field, pd.NA)
    if _is_missing_value(value):
        return
    candidate[field] = _normalize_candidate_value(value)


def _append_candidate_source_tag(candidates: list[dict[str, object]], *, symbol: str, tag: str) -> None:
    normalized = _normalize_symbol(symbol)
    for candidate in candidates:
        if _normalize_symbol(candidate.get("symbol", "")) != normalized:
            continue
        tags = candidate.get("source_tags")
        if isinstance(tags, list):
            source_tags = [str(item) for item in tags]
        elif tags:
            source_tags = [str(tags)]
        else:
            source_tags = []
        if tag not in source_tags:
            source_tags.append(tag)
        candidate["source_tags"] = source_tags
        source = str(candidate.get("source", "") or "")
        if source and tag not in source.split("+"):
            candidate["source"] = f"{source}+{tag}"
        elif not source:
            candidate["source"] = tag
        return


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


def _truthy_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def build_watchlist_candidates(project_root: Path, limit: int = 30) -> dict[str, object]:
    patterns_file = _latest_patterns_file(project_root)
    frame = pd.read_csv(patterns_file)
    return build_watchlist_candidates_from_patterns(frame, source_file=str(patterns_file), limit=limit)


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
    _write_watchlist_csv(target.with_suffix(".csv"), payload)
    return target


def write_intraday_pool(*, project_root: Path, trade_date: date, picker_payload: dict[str, object]) -> Path:
    target = intraday_pool_path(project_root, trade_date)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_watchlist_payload(trade_date=trade_date, picker_payload=picker_payload)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_watchlist_csv(target.with_suffix(".csv"), payload, kind="intraday_pool")
    return target


def _write_watchlist_csv(target: Path, payload: dict[str, object], *, kind: str = "watchlist") -> None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    top_level_fields = [
        "trade_date",
        "trade_permission",
        "next_open_trade_permission",
        "next_open_trade_warning",
        "trade_permission_note",
        "phase7_score_100",
        "phase7_buy_day_risk_score",
        "phase7_selected_threshold",
        "phase7_trade_permission",
        "phase7_suggested_action",
        "phase7_reason",
    ]
    rows: list[dict[str, object]] = []
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        row = {
            "trade_date": payload.get("trade_date"),
            "candidate_index": index,
        }
        for field in top_level_fields:
            if field in payload and field not in row:
                row[field] = payload.get(field)
        row.update(item)
        for field in PHASE_TABLE_DROP_COLUMNS:
            row.pop(field, None)
        rows.append({key: _csv_cell_value(value) for key, value in row.items()})
    if rows:
        frame = pd.DataFrame(rows)
        if "symbol" in frame.columns:
            frame["symbol"] = frame["symbol"].map(_format_symbol_for_excel)
        technical_columns = [
            "macd_cross_state",
            "macd_divergence_state",
            "volume_price_divergence_state",
            "macd_top_divergence_15d",
            "macd_bottom_divergence_15d",
            "macd_top_divergence_signal_date",
            "macd_bottom_divergence_signal_date",
            "bullish_volume_price_divergence_flag",
            "bearish_volume_price_divergence_flag",
            "macd",
            "macd_signal_line",
            "macd_hist",
            "ATR14",
            "1ATR止损参考",
            "2ATR止损参考",
            "2ATR止盈参考",
            "3ATR止盈参考",
            "波动分层",
        ]
        phase_detail_columns = [
            "phase2_is_cusum_event",
            "phase1_center_score",
            "phase2_center_score",
            "centered_risk_score",
            "phase4_composite_score",
            "phase4_composite_rank",
            "phase7_score_100",
            "phase7_trade_permission",
            "phase1_feature_trade_date",
            "phase1_model_name",
            "phase1_model_version",
            "phase2_feature_trade_date",
            "phase2_model_name",
            "phase2_model_version",
            "phase4_feature_trade_date",
            "phase4_model_name",
            "phase4_model_version",
            "phase8_rank",
            "phase8_top5_rank",
            "phase8_raw_score",
            "phase8_feature_trade_date",
            "phase8_model_name",
            "phase8_model_version",
            "today_limit_up_excluded",
            "today_high_return_vs_prev_close",
            "today_close_return_vs_prev_close",
            "phase7_feature_trade_date",
            "phase7_model_name",
            "phase7_model_version",
            "trade_permission",
            "next_open_trade_permission",
            "next_open_trade_warning",
            "trade_permission_note",
            "phase7_reason",
        ]
        pattern_detail_columns = [
            "reason",
            "patterns",
            *PATTERN_CANDIDATE_FIELDS,
        ]
        if kind == "intraday_pool":
            phase_detail_columns = [
                "candidate_index",
                "source",
                "phase5_score_100",
                "phase4_5d_std",
                *phase_detail_columns,
            ]
            preferred = [
                "trade_date",
                "symbol",
                "name",
                "涨幅%",
                "phase1_score_100",
                "phase2_score_100",
                "phase4_score_100",
                "phase8_score_100",
                "phase4_5d_mean",
                "pattern_match",
                "pattern_ids",
                "pattern_id",
                "ATR%",
                RECOMMENDED_POSITION_PERCENT_FIELD,
                *technical_columns,
                *phase_detail_columns,
                *pattern_detail_columns,
            ]
        else:
            preferred = [
                "trade_date",
                "candidate_index",
                "symbol",
                "name",
                "涨幅%",
                "source",
                "pattern_match",
                "pattern_ids",
                "pattern_id",
                "phase1_score_100",
                "phase2_score_100",
                "phase4_score_100",
                *PHASE4_ROLLING_COLUMNS,
                "phase8_score_100",
                "phase5_score_100",
                "ATR%",
                RECOMMENDED_POSITION_PERCENT_FIELD,
                *technical_columns,
                *phase_detail_columns,
                *pattern_detail_columns,
            ]
        seen_columns: set[str] = set()
        ordered = []
        for column in preferred:
            if column in frame.columns and column not in seen_columns:
                ordered.append(column)
                seen_columns.add(column)
        frame = frame.loc[:, ordered + [column for column in frame.columns if column not in ordered]]
    else:
        if kind == "intraday_pool":
            frame = pd.DataFrame(
                columns=[
                    "trade_date",
                    "symbol",
                    "name",
                    "涨幅%",
                    "phase1_score_100",
                    "phase2_score_100",
                    "phase4_score_100",
                    "phase8_score_100",
                    "phase4_5d_mean",
                    "pattern_match",
                    "ATR%",
                    RECOMMENDED_POSITION_PERCENT_FIELD,
                ]
            )
        else:
            frame = pd.DataFrame(
                columns=[
                    "trade_date",
                    "candidate_index",
                    "symbol",
                    "name",
                    "涨幅%",
                    "source",
                    "pattern_match",
                    "phase1_score_100",
                    "phase2_score_100",
                    "phase4_score_100",
                    *PHASE4_ROLLING_COLUMNS,
                    "phase8_score_100",
                    "phase5_score_100",
                    "phase7_score_100",
                ]
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")


def _format_symbol_for_excel(value: object) -> object:
    symbol = _normalize_symbol(value)
    if not symbol:
        return value
    return f'="{symbol}"'


def _csv_cell_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    return value


def build_watchlist_payload(*, trade_date: date, picker_payload: dict[str, object]) -> dict[str, object]:
    payload = deepcopy(picker_payload)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    passthrough_fields = {
        "model_source_file",
        "model_source_files",
        "selection_policy",
        "filter_summary",
        "trade_permission",
        "next_open_trade_permission",
        "next_open_trade_warning",
        "trade_permission_note",
        "phase7_buy_day_risk_score",
        "phase7_score_100",
        "phase7_feature_trade_date",
        "phase7_selected_threshold",
        "phase7_trade_permission",
        "phase7_suggested_action",
        "phase7_reason",
        "phase7_model_name",
        "phase7_model_version",
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


def load_intraday_pool(*, project_root: Path, trade_date: date) -> dict[str, object]:
    target = intraday_pool_path(project_root, trade_date)
    if not target.exists():
        raise FileNotFoundError(f"Intraday pool not found for {trade_date.isoformat()}: {target}")
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


def find_latest_intraday_pool_before(*, project_root: Path, trade_date: date) -> tuple[date, Path]:
    candidates: list[tuple[date, Path]] = []
    for path in watchlists_dir(project_root).glob("intraday_pool_*.json"):
        parsed = _parse_intraday_pool_date(path)
        if parsed is None or parsed >= trade_date:
            continue
        candidates.append((parsed, path))

    if not candidates:
        raise FileNotFoundError(f"No intraday pool found before {trade_date.isoformat()} in {watchlists_dir(project_root)}")

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


def _parse_intraday_pool_date(path: Path) -> date | None:
    match = re.fullmatch(r"intraday_pool_(\d{4}-\d{2}-\d{2})\.json", path.name)
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
    return watchlist_path(project_root, trade_date)


def _project_root_from_source_files(source_files: dict[str, str] | None) -> Path:
    if source_files and source_files.get("phase4"):
        path = Path(str(source_files["phase4"])).resolve()
        if len(path.parents) >= 3:
            return path.parents[2]
    return Path(".").resolve()


def _append_supported_fields(candidate: dict[str, object], row: pd.Series | dict[str, object]) -> None:
    for field in PATTERN_CANDIDATE_FIELDS:
        value = _row_value(row, field, pd.NA)
        if pd.isna(value):
            continue
        candidate[field] = _normalize_candidate_value(value)

    for field in TECHNICAL_CANDIDATE_FIELDS:
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
    position_pct = recommended_position_percent_from_mapping(row)
    if position_pct is not None:
        candidate[RECOMMENDED_POSITION_PERCENT_FIELD] = position_pct


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
