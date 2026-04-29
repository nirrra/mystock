from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.watchlist import (
    apply_trend_filter_to_watchlist_payload,
    build_watchlist_candidates,
    build_watchlist_candidates_from_trend,
    extract_watchlist_symbols,
    find_latest_watchlist_before,
    load_watchlist,
    write_watchlist,
    watchlist_pattern_path,
    watchlist_trend_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_write_and_load_watchlist_round_trip() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_round_trip")
    payload = {
        "source_file": str(tmp_path / "reports" / "patterns" / "patterns_all_2026-04-10.csv"),
        "candidates": [
            {"symbol": "2579", "name": "中京电子", "tier": "第一梯队"},
            {"symbol": "603803", "name": "瑞斯康达", "tier": "第二梯队"},
        ],
        "analysis": {"summary": "should not be persisted"},
    }

    target = write_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), picker_payload=payload)
    loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10))

    assert target.exists()
    assert loaded["trade_date"] == "2026-04-10"
    assert loaded["candidate_count"] == 2
    assert "analysis" not in loaded
    assert extract_watchlist_symbols(loaded) == ["002579", "603803"]
    assert loaded["candidates"][0]["连续上榜天数"] == 1
    assert loaded["candidates"][1]["连续上榜天数"] == 1


def test_write_and_load_kind_specific_watchlists_round_trip() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_kind_round_trip")
    payload = {
        "source_file": "demo.csv",
        "candidates": [{"symbol": "600000", "name": "测试股份"}],
    }

    pattern_target = write_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), picker_payload=payload, kind="pattern")
    trend_target = write_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), picker_payload=payload, kind="trend")

    pattern_loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), kind="pattern")
    trend_loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), kind="trend")

    assert pattern_target == watchlist_pattern_path(tmp_path, date(2026, 4, 10))
    assert trend_target == watchlist_trend_path(tmp_path, date(2026, 4, 10))
    assert pattern_loaded["candidate_count"] == 1
    assert trend_loaded["candidate_count"] == 1
    assert pattern_loaded["candidates"][0]["连续上榜天数"] == 1
    assert "连续上榜天数" not in trend_loaded["candidates"][0]


def test_write_watchlist_serializes_temporal_candidate_fields() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_temporal_fields")
    payload = {
        "source_file": "demo.csv",
        "candidates": [
            {
                "symbol": "600000",
                "name": "测试股份",
                "trade_date": pd.Timestamp("2026-04-21"),
                "planned_entry_date": pd.Timestamp("2026-04-22 09:30:00"),
            }
        ],
    }

    write_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 21), picker_payload=payload, kind="trend")
    loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 21), kind="trend")

    assert loaded["candidates"][0]["trade_date"] == "2026-04-21"
    assert loaded["candidates"][0]["planned_entry_date"] == "2026-04-22T09:30:00"


def test_write_watchlist_increments_main_watchlist_streaks_from_previous_day() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_streaks")

    write_watchlist(
        project_root=tmp_path,
        trade_date=date(2026, 4, 10),
        picker_payload={
            "source_file": "day1.csv",
            "candidates": [
                {"symbol": "600000", "name": "测试甲"},
                {"symbol": "600001", "name": "测试乙"},
            ],
        },
    )

    write_watchlist(
        project_root=tmp_path,
        trade_date=date(2026, 4, 11),
        picker_payload={
            "source_file": "day2.csv",
            "candidates": [
                {"symbol": "600000", "name": "测试甲"},
                {"symbol": "600002", "name": "测试丙"},
            ],
        },
    )

    loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 11))
    streaks = {item["symbol"]: item["连续上榜天数"] for item in loaded["candidates"]}

    assert streaks["600000"] == 2
    assert streaks["600002"] == 1


def test_find_latest_watchlist_before_uses_latest_prior_file() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_latest")
    for item_date in (date(2026, 4, 9), date(2026, 4, 10), date(2026, 4, 12)):
        write_watchlist(
            project_root=tmp_path,
            trade_date=item_date,
            picker_payload={"source_file": "demo.csv", "candidates": [{"symbol": "600000"}]},
        )

    resolved_date, resolved_path = find_latest_watchlist_before(project_root=tmp_path, trade_date=date(2026, 4, 11))

    assert resolved_date == date(2026, 4, 10)
    assert resolved_path.name == "watchlist_2026-04-10.json"


def test_build_watchlist_candidates_preserves_technical_rules() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_build")
    patterns_dir = tmp_path / "reports" / "patterns"
    patterns_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "symbol": '="002579"',
                "name": "中京电子",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.44,
                "tradingview_all_rating_label": "buy",
                "tradingview_all_rating_2026-04-08": 0.46,
                "tradingview_all_rating_2026-04-09": 0.60,
                "tradingview_all_rating_2026-04-10": 0.46,
                "tradingview_all_rating_2026-04-11": 0.26,
                "tradingview_all_rating_2026-04-12": 0.40,
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": False,
                "old_high_date": "2026-02-18",
                "old_high_price": 13.2,
                "days_since_old_high": 36,
                "distance_to_old_high_pct": 0.051,
                "reason": "demo-1",
            },
            {
                "symbol": "603803",
                "name": "瑞斯康达",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.44,
                "tradingview_all_rating_label": "buy",
                "tradingview_all_rating_2026-04-08": 0.46,
                "tradingview_all_rating_2026-04-09": 0.60,
                "tradingview_all_rating_2026-04-10": 0.46,
                "tradingview_all_rating_2026-04-11": 0.26,
                "tradingview_all_rating_2026-04-12": 0.40,
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": False,
                "breakout_date": "2026-04-12",
                "breakout_volume_ratio": 3.2,
                "breakout_close_position": 0.72,
                "breakout_upper_shadow_pct": 0.18,
                "breakout_body_pct": 0.45,
                "breakout_turnover": 8.5,
                "breakout_turnover_state": "normal",
                "days_after_breakout": 2,
                "post_breakout_max_high_extension_pct": 0.072,
                "ma20_slope_short_pct": 0.012,
                "ma20_slope_long_pct": 0.08,
                "ma60_slope_short_pct": 0.004,
                "ma60_slope_long_pct": 0.03,
                "pullback_volume_contraction_ratio": 0.82,
                "pullback_max_rise_tail_volume_ratio": 0.82,
                "platform_volume_contraction_ratio": 0.72,
                "platform_range_contraction_ratio": 0.64,
                "platform_low_lift_pct": 0.018,
                "platform_max_bearish_body_pct": 0.025,
                "platform_max_bearish_volume_ratio": 1.2,
                "reason": "demo-2",
            },
            {
                "symbol": "600111",
                "name": "风险票",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.46,
                "tradingview_all_rating_label": "strong_buy",
                "tradingview_all_rating_2026-04-08": 0.48,
                "tradingview_all_rating_2026-04-09": 0.52,
                "tradingview_all_rating_2026-04-10": 0.46,
                "tradingview_all_rating_2026-04-11": 0.42,
                "tradingview_all_rating_2026-04-12": 0.42,
                "macd_cross_state": "dead_cross",
                "reason": "risk",
            },
            {
                "symbol": "000001",
                "name": "上证综指",
                "pattern_id": "1",
                "tradingview_avg_all_rating_5d": 0.52,
                "tradingview_all_rating_label": "strong_buy",
                "tradingview_all_rating_2026-04-08": 0.52,
                "tradingview_all_rating_2026-04-09": 0.51,
                "tradingview_all_rating_2026-04-10": 0.52,
                "tradingview_all_rating_2026-04-11": 0.53,
                "tradingview_all_rating_2026-04-12": 0.52,
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": False,
                "reason": "index",
            },
            {
                "symbol": "600000",
                "name": "弱票",
                "pattern_id": "4",
                "tradingview_avg_all_rating_5d": 0.15,
                "tradingview_all_rating_label": "buy",
                "tradingview_all_rating_2026-04-08": 0.10,
                "tradingview_all_rating_2026-04-09": 0.15,
                "tradingview_all_rating_2026-04-10": 0.18,
                "tradingview_all_rating_2026-04-11": 0.16,
                "tradingview_all_rating_2026-04-12": 0.16,
            },
        ]
    ).to_csv(patterns_dir / "patterns_all_2026-04-12.csv", index=False, encoding="utf-8-sig")

    payload = build_watchlist_candidates(tmp_path, limit=10)

    assert Path(payload["source_file"]).parent.name == "patterns"
    assert payload["candidate_count"] == 2
    symbols = [item["symbol"] for item in payload["candidates"]]
    assert symbols == ["002579", "603803"]

    stable_scores = {item["symbol"]: item["stable_score"] for item in payload["candidates"]}
    assert stable_scores["002579"] == stable_scores["603803"]
    assert payload["candidates"][0]["tier"] == "第一梯队"
    assert payload["candidates"][0]["old_high_date"] == "2026-02-18"
    assert payload["candidates"][0]["old_high_price"] == 13.2
    assert payload["candidates"][0]["distance_to_old_high_pct"] == 0.051
    assert payload["candidates"][1]["breakout_date"] == "2026-04-12"
    assert payload["candidates"][1]["breakout_volume_ratio"] == 3.2
    assert payload["candidates"][1]["breakout_turnover"] == 8.5
    assert payload["candidates"][1]["breakout_turnover_state"] == "normal"
    assert payload["candidates"][1]["days_after_breakout"] == 2
    assert payload["candidates"][1]["post_breakout_max_high_extension_pct"] == 0.072
    assert payload["candidates"][1]["ma20_slope_short_pct"] == 0.012
    assert payload["candidates"][1]["ma60_slope_long_pct"] == 0.03
    assert payload["candidates"][1]["pullback_volume_contraction_ratio"] == 0.82
    assert payload["candidates"][1]["pullback_max_rise_tail_volume_ratio"] == 0.82
    assert payload["candidates"][1]["platform_volume_contraction_ratio"] == 0.72
    assert payload["candidates"][1]["platform_range_contraction_ratio"] == 0.64
    assert payload["candidates"][1]["platform_low_lift_pct"] == 0.018
    assert payload["candidates"][1]["platform_max_bearish_volume_ratio"] == 1.2


def test_build_watchlist_candidates_from_trend_applies_thresholds_and_risk_filter() -> None:
    trend_frame = pd.DataFrame(
        [
            {
                "symbol": '="600000"',
                "name": "趋势甲",
                "signal_type": "breakout",
                "buy_score": 78.0,
                "price_action_score": 61.0,
                "trend_score": 82.0,
                "macd_cross_state": "golden_cross",
                "macd_divergence_state": "none",
                "volume_price_divergence_state": "none",
                "trade_date": "2026-04-12",
                "close": 10.0,
                "atr_14": 0.8,
                "atr_pct_14": 0.08,
                "atr_stop_loss_1x": 9.2,
                "atr_stop_loss_2x": 8.4,
                "atr_take_profit_2x": 11.6,
                "atr_take_profit_3x": 12.4,
                "atr_volatility_regime": "高波动",
            },
            {
                "symbol": "600001",
                "name": "趋势乙",
                "signal_type": "pullback",
                "buy_score": 76.0,
                "price_action_score": 62.0,
                "trend_score": 79.0,
                "macd_cross_state": "dead_cross",
                "macd_divergence_state": "none",
                "volume_price_divergence_state": "none",
            },
            {
                "symbol": "600002",
                "name": "趋势丙",
                "signal_type": "breakout",
                "buy_score": 65.0,
                "price_action_score": 51.0,
                "trend_score": 81.0,
                "macd_cross_state": "above_signal",
                "macd_divergence_state": "none",
                "volume_price_divergence_state": "none",
            },
        ]
    )
    thresholds = type("PickTrendWatchlist", (), {"buy_score_min": 70.0, "price_action_score_min": 55.0})()

    payload = build_watchlist_candidates_from_trend(
        trend_frame,
        source_file="reports/trend/trend_2026-04-12.csv",
        thresholds=thresholds,
        limit=10,
    )

    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["symbol"] == "600000"
    assert payload["candidates"][0]["source"] == "trend"
    assert payload["candidates"][0]["buy_score"] == 78.0
    assert payload["candidates"][0]["ATR14"] == 0.8
    assert payload["candidates"][0]["ATR%"] == 8.0
    assert payload["candidates"][0]["波动分层"] == "高波动"


def test_apply_trend_filter_to_watchlist_payload_requires_strict_intersection_with_trend_universe() -> None:
    payload = {
        "source_file": "reports/patterns/patterns_all_2026-04-12.csv",
        "candidate_count": 3,
        "candidates": [
            {"symbol": "002579", "name": "中京电子", "tier": "第一梯队"},
            {"symbol": "603803", "name": "瑞斯康达", "tier": "第二梯队"},
            {"symbol": "600000", "name": "浦发银行", "tier": "第三梯队"},
        ],
    }
    trend_frame = pd.DataFrame(
        [
            {
                "symbol": '="002579"',
                "in_trend_universe": True,
                "trend_universe_score": 88.0,
                "signal_type": "breakout",
                "macd_cross_state": "golden_cross",
                "macd_divergence_state": "none",
                "volume_price_divergence_state": "bullish",
            },
            {
                "symbol": "603803",
                "in_trend_universe": False,
                "trend_universe_score": 81.0,
                "signal_type": "pullback",
            },
        ]
    )
    trend_filter = type(
        "TrendFilter",
        (),
        {"buy_score_min": 70.0, "price_action_score_min": 55.0},
    )()

    filtered = apply_trend_filter_to_watchlist_payload(
        payload,
        trend_frame=trend_frame,
        trend_filter=trend_filter,
    )

    assert filtered["candidate_count"] == 1
    assert filtered["candidates"][0]["symbol"] == "002579"
    assert filtered["candidates"][0]["trend_universe_score"] == 88.0
    assert filtered["candidates"][0]["macd_cross_state"] == "golden_cross"
