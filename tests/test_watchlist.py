from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.watchlist import (
    build_intraday_pool_candidates,
    build_phase_daily_watchlist_candidates,
    build_watchlist_candidates,
    build_watchlist_candidates_from_patterns,
    extract_watchlist_symbols,
    find_latest_watchlist_before,
    load_watchlist,
    watchlist_pattern_path,
    write_watchlist,
    write_intraday_pool,
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
            {"symbol": "2579", "name": "中京电子", "涨幅%": 2.34, "tier": "第一梯队"},
            {"symbol": "603803", "name": "瑞斯康达", "tier": "第二梯队"},
        ],
        "analysis": {"summary": "should not be persisted"},
    }

    target = write_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), picker_payload=payload)
    loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10))

    assert target.exists()
    assert target.with_suffix(".csv").exists()
    assert loaded["trade_date"] == "2026-04-10"
    assert loaded["candidate_count"] == 2
    assert "analysis" not in loaded
    assert extract_watchlist_symbols(loaded) == ["002579", "603803"]
    assert loaded["candidates"][0]["连续上榜天数"] == 1
    assert loaded["candidates"][1]["连续上榜天数"] == 1
    csv_frame = pd.read_csv(target.with_suffix(".csv"))
    assert list(csv_frame.columns[:5]) == ["trade_date", "candidate_index", "symbol", "name", "涨幅%"]


def test_write_intraday_pool_csv_prioritizes_intraday_review_columns() -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_pool_column_order")
    payload = {
        "source_file": "demo.csv",
        "candidates": [
            {
                "symbol": "600000",
                "name": "测试股份",
                "source": "pattern_pool",
                "涨幅%": 1.23,
                "phase1_score_100": 71.0,
                "phase2_score_100": 82.0,
                "phase4_score_100": 93.0,
                "phase8_score_100": 64.0,
                "phase4_5d_mean": 88.0,
                "pattern_match": "是",
                "pattern_ids": "5",
                "ATR%": 3.21,
                "建议总仓位%": 36.6,
                "macd_cross_state": "above_signal",
                "reason": "pattern detail",
            }
        ],
    }

    target = write_intraday_pool(project_root=tmp_path, trade_date=date(2026, 4, 10), picker_payload=payload)
    csv_frame = pd.read_csv(target.with_suffix(".csv"))

    assert list(csv_frame.columns[:15]) == [
        "trade_date",
        "symbol",
        "name",
        "source",
        "涨幅%",
        "phase1_score_100",
        "phase2_score_100",
        "phase4_score_100",
        "phase8_score_100",
        "phase4_5d_mean",
        "pattern_match",
        "pattern_ids",
        "ATR%",
        "建议总仓位%",
        "macd_cross_state",
    ]


def test_write_and_load_pattern_watchlist_round_trip() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_pattern_round_trip")
    payload = {
        "source_file": "demo.csv",
        "candidates": [{"symbol": "600000", "name": "测试股份"}],
    }

    target = write_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), picker_payload=payload, kind="pattern")
    loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 10), kind="pattern")

    assert target == watchlist_pattern_path(tmp_path, date(2026, 4, 10))
    assert loaded["candidate_count"] == 1
    assert loaded["candidates"][0]["连续上榜天数"] == 1


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

    write_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 21), picker_payload=payload)
    loaded = load_watchlist(project_root=tmp_path, trade_date=date(2026, 4, 21))

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


def test_build_watchlist_candidates_preserves_pattern_and_technical_fields() -> None:
    tmp_path = _make_workspace_tmp_dir("watchlist_build")
    patterns_dir = tmp_path / "reports" / "patterns"
    patterns_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "symbol": '="002579"',
                "name": "中京电子",
                "pattern_id": "1",
                "old_high_date": "2026-02-18",
                "old_high_price": 13.2,
                "days_since_old_high": 36,
                "distance_to_old_high_pct": 0.051,
                "macd_cross_state": "golden_cross",
                "macd_bottom_divergence_15d": True,
                "reason": "demo-1",
            },
            {
                "symbol": "603803",
                "name": "瑞斯康达",
                "pattern_id": "5",
                "recent_high_date": "2026-04-01",
                "distance_from_recent_high_pct": -0.03,
                "pullback_volume_contraction_ratio": 0.82,
                "atr_14": 0.8,
                "atr_pct_14": 0.08,
                "reason": "demo-2",
            },
            {
                "symbol": "600111",
                "name": "风险票",
                "pattern_id": "1",
                "macd_cross_state": "dead_cross",
                "reason": "risk",
            },
            {
                "symbol": "000001",
                "name": "上证综指",
                "pattern_id": "1",
                "reason": "index",
            },
        ]
    ).to_csv(patterns_dir / "patterns_all_2026-04-12.csv", index=False, encoding="utf-8-sig")

    payload = build_watchlist_candidates(tmp_path, limit=10)

    assert Path(payload["source_file"]).parent.name == "patterns"
    assert payload["candidate_count"] == 2
    symbols = [item["symbol"] for item in payload["candidates"]]
    assert symbols == ["002579", "603803"]
    assert payload["candidates"][0]["tier"] == "第一梯队"
    assert payload["candidates"][0]["old_high_price"] == 13.2
    assert payload["candidates"][0]["macd_cross_state"] == "golden_cross"
    assert payload["candidates"][1]["recent_high_date"] == "2026-04-01"
    assert payload["candidates"][1]["ATR14"] == 0.8
    assert payload["candidates"][1]["ATR%"] == 8.0
    assert payload["candidates"][1]["建议总仓位%"] == 14.71


def test_build_phase_daily_watchlist_filters_phase1_phase2_and_adds_phase4_top() -> None:
    pattern_frame = pd.DataFrame(
        [
            {"symbol": "600000", "name": "模式保留", "pattern_id": "1", "reason": "pattern ok"},
            {"symbol": "600002", "name": "模式但风险高", "pattern_id": "5", "reason": "filtered"},
        ]
    )
    phase1 = pd.DataFrame(
        [
            {"symbol": "600000", "name": "模式保留", "risk_score": 0.10, "model_name": "logistic"},
            {"symbol": "600001", "name": "一阶段高风险", "risk_score": 0.90, "model_name": "logistic"},
            {"symbol": "600002", "name": "二阶段高风险", "risk_score": 0.20, "model_name": "logistic"},
            {"symbol": "600003", "name": "收益候选甲", "risk_score": 0.30, "model_name": "logistic"},
            {"symbol": "600004", "name": "收益候选乙", "risk_score": 0.40, "model_name": "logistic"},
        ]
    )
    phase2 = pd.DataFrame(
        [
            {"symbol": "600000", "barrier_risk_score": 0.10, "is_cusum_event": False, "model_name": "barrier"},
            {"symbol": "600001", "barrier_risk_score": 0.20, "is_cusum_event": True, "model_name": "barrier"},
            {"symbol": "600002", "barrier_risk_score": 0.95, "is_cusum_event": True, "model_name": "barrier"},
            {"symbol": "600003", "barrier_risk_score": 0.30, "is_cusum_event": False, "model_name": "barrier"},
            {"symbol": "600004", "barrier_risk_score": 0.40, "is_cusum_event": False, "model_name": "barrier"},
        ]
    )
    phase4 = pd.DataFrame(
        [
            {"symbol": "600000", "return_score": 0.01, "model_name": "lightgbm"},
            {"symbol": "600001", "return_score": 0.03, "model_name": "lightgbm"},
            {"symbol": "600002", "return_score": 0.10, "model_name": "lightgbm"},
            {"symbol": "600003", "return_score": 0.09, "model_name": "lightgbm"},
            {"symbol": "600004", "return_score": 0.02, "model_name": "lightgbm"},
        ]
    )
    phase7 = pd.DataFrame(
        [
            {
                "feature_trade_date": "2026-05-07",
                "buy_day_risk_score": 0.12,
                "selected_threshold": 0.20,
                "trade_permission": "allow",
                "suggested_action": "allow",
                "reason": "below threshold",
            }
        ]
    )
    phase5 = pd.DataFrame(
        [
            {"symbol": "600000", "year": 2026, "NEGOUTLIER": 0, "CRASH": 0, "NCSKEW": 0.1, "DUVOL": 0.1},
            {"symbol": "600004", "year": 2026, "NEGOUTLIER": 0, "CRASH": 0, "NCSKEW": 0.2, "DUVOL": 0.2},
        ]
    )
    atr = pd.DataFrame(
        [
            {
                "symbol": "600003",
                "trade_date": "2026-05-07",
                "close": 10.0,
                "atr_14": 0.5,
                "atr_pct_14": 0.05,
                "atr_stop_loss_2x": 9.0,
                "atr_take_profit_2x": 11.0,
            }
        ]
    )

    payload = build_phase_daily_watchlist_candidates(
        trade_date=date(2026, 5, 7),
        pattern_frame=pattern_frame,
        phase1_predictions=phase1,
        phase2_predictions=phase2,
        phase4_predictions=phase4,
        phase7_prediction=phase7,
        phase5_measures=phase5,
        atr_frame=atr,
        phase_filter_rate=0.2,
        phase4_top_n=2,
    )

    assert payload["trade_permission"] == "allow"
    assert payload["filter_summary"]["phase1_excluded_top20"] == 1
    assert payload["filter_summary"]["phase2_excluded_top20"] == 1
    assert [item["symbol"] for item in payload["candidates"]] == ["600002", "600003"]
    assert payload["candidates"][0]["source"] == "pattern"
    assert payload["candidates"][0]["pattern_ids"] == ["5"]
    assert payload["candidates"][0]["phase2_excluded_by_top20_risk"] is True
    assert payload["candidates"][0]["phase4_score_100"] == 100.0
    assert payload["candidates"][1]["source"] == "phase4_top"
    assert payload["candidates"][1]["phase4_score_100"] == 75.0
    assert payload["candidates"][1]["phase1_center_score"] == 40.0
    assert payload["candidates"][1]["phase2_center_score"] == 40.0
    assert payload["candidates"][1]["centered_risk_score"] == 83.0
    assert payload["candidates"][1]["phase4_composite_score"] == 83.0
    assert payload["candidates"][1]["phase4_composite_rank"] == 1
    assert payload["candidates"][1]["建议总仓位%"] == 23.53
    assert payload["candidates"][1]["phase4_top_score_filter_pass"] is True
    assert "600004" not in [item["symbol"] for item in payload["candidates"]]
    assert payload["selection_policy"]["centered_risk_min_phase1_score"] == 40.0
    assert payload["selection_policy"]["centered_risk_min_phase2_score"] == 50.0
    assert payload["selection_policy"]["centered_risk_min_phase4_score"] == 70.0
    assert payload["selection_policy"]["pattern_min_phase4_score"] == 70.0
    assert payload["filter_summary"]["pattern_symbols_after_filter"] == 1
    assert payload["filter_summary"]["phase4_top_candidates_after_score_floor"] == 1


def test_build_intraday_pool_uses_pattern_then_p124_then_phase8_fill() -> None:
    pattern_frame = pd.DataFrame(
        [
            {"symbol": "600000", "name": "低P4模式", "pattern_id": "1", "reason": "pattern low p4"},
            {"symbol": "600001", "name": "高P4模式", "pattern_id": "5", "reason": "pattern high p4"},
            {"symbol": "600002", "name": "中P4模式", "pattern_id": "6", "reason": "pattern mid p4"},
        ]
    )
    phase1 = pd.DataFrame(
        [
            {"symbol": "600000", "name": "低P4模式", "risk_score": 0.90, "model_name": "tail"},
            {"symbol": "600001", "name": "高P4模式", "risk_score": 0.80, "model_name": "tail"},
            {"symbol": "600002", "name": "中P4模式", "risk_score": 0.70, "model_name": "tail"},
            {"symbol": "600003", "name": "混合候选", "risk_score": 0.10, "model_name": "tail"},
            {"symbol": "600004", "name": "P8候选甲", "risk_score": 0.20, "model_name": "tail"},
            {"symbol": "600005", "name": "P8候选乙", "risk_score": 0.30, "model_name": "tail"},
        ]
    )
    phase2 = pd.DataFrame(
        [
            {"symbol": "600000", "barrier_risk_score": 0.90, "model_name": "barrier"},
            {"symbol": "600001", "barrier_risk_score": 0.80, "model_name": "barrier"},
            {"symbol": "600002", "barrier_risk_score": 0.70, "model_name": "barrier"},
            {"symbol": "600003", "barrier_risk_score": 0.10, "model_name": "barrier"},
            {"symbol": "600004", "barrier_risk_score": 0.20, "model_name": "barrier"},
            {"symbol": "600005", "barrier_risk_score": 0.30, "model_name": "barrier"},
        ]
    )
    phase4 = pd.DataFrame(
        [
            {"symbol": "600000", "return_score": 0.01, "model_name": "return"},
            {"symbol": "600001", "return_score": 0.60, "model_name": "return"},
            {"symbol": "600002", "return_score": 0.50, "model_name": "return"},
            {"symbol": "600003", "return_score": 0.40, "model_name": "return"},
            {"symbol": "600004", "return_score": 0.30, "model_name": "return"},
            {"symbol": "600005", "return_score": 0.20, "model_name": "return"},
        ]
    )
    phase8 = pd.DataFrame(
        [
            {"symbol": "600005", "phase8_score_100": 99.0},
            {"symbol": "600004", "phase8_score_100": 98.0},
            {"symbol": "600003", "phase8_score_100": 97.0},
        ]
    )
    phase7 = pd.DataFrame([{"trade_permission": "allow", "buy_day_risk_score": 0.1}])

    payload = build_intraday_pool_candidates(
        trade_date=date(2026, 5, 7),
        pattern_frame=pattern_frame,
        phase1_predictions=phase1,
        phase2_predictions=phase2,
        phase4_predictions=phase4,
        phase8_predictions=phase8,
        phase7_prediction=phase7,
        pattern_limit=2,
        p124_top_n=2,
        pool_size=5,
    )

    symbols = [item["symbol"] for item in payload["candidates"]]
    assert symbols[:2] == ["600001", "600002"]
    assert "600000" not in symbols
    assert len(symbols) == 5
    assert "pattern_pool" in payload["candidates"][0]["source_tags"]
    assert any("p8_fill" in item.get("source_tags", []) for item in payload["candidates"])
    assert payload["filter_summary"]["pattern_symbols_total"] == 3
    assert payload["filter_summary"]["pattern_pool_count"] == 2
    assert payload["selection_policy"]["intraday_pool_size"] == 5


def test_build_phase_daily_watchlist_excludes_same_day_limit_up_candidates() -> None:
    pattern_frame = pd.DataFrame(
        [
            {"symbol": "600000", "name": "涨停模式", "pattern_id": "1", "reason": "limit up pattern"},
            {"symbol": "600003", "name": "普通模式", "pattern_id": "5", "reason": "normal pattern"},
        ]
    )
    phase1 = pd.DataFrame(
        [
            {"symbol": "600000", "name": "涨停模式", "risk_score": 0.10, "log_return_1d": math.log(1.10), "model_name": "logistic"},
            {"symbol": "600001", "name": "一阶段高风险", "risk_score": 0.90, "log_return_1d": 0.00, "model_name": "logistic"},
            {"symbol": "600002", "name": "普通候选甲", "risk_score": 0.20, "log_return_1d": 0.01, "model_name": "logistic"},
            {"symbol": "600003", "name": "普通模式", "risk_score": 0.30, "log_return_1d": 0.02, "model_name": "logistic"},
            {"symbol": "600004", "name": "普通候选乙", "risk_score": 0.40, "log_return_1d": 0.03, "model_name": "logistic"},
        ]
    )
    phase2 = pd.DataFrame(
        [
            {"symbol": "600000", "barrier_risk_score": 0.10, "model_name": "barrier"},
            {"symbol": "600001", "barrier_risk_score": 0.95, "model_name": "barrier"},
            {"symbol": "600002", "barrier_risk_score": 0.20, "model_name": "barrier"},
            {"symbol": "600003", "barrier_risk_score": 0.30, "model_name": "barrier"},
            {"symbol": "600004", "barrier_risk_score": 0.40, "model_name": "barrier"},
        ]
    )
    phase4 = pd.DataFrame(
        [
            {"symbol": "600000", "return_score": 0.20, "model_name": "lightgbm"},
            {"symbol": "600001", "return_score": 0.09, "model_name": "lightgbm"},
            {"symbol": "600002", "return_score": 0.18, "model_name": "lightgbm"},
            {"symbol": "600003", "return_score": 0.19, "model_name": "lightgbm"},
            {"symbol": "600004", "return_score": 0.07, "model_name": "lightgbm"},
        ]
    )
    phase7 = pd.DataFrame(
        [
            {
                "feature_trade_date": "2026-05-07",
                "buy_day_risk_score": 0.12,
                "selected_threshold": 0.20,
                "trade_permission": "allow",
            }
        ]
    )

    payload = build_phase_daily_watchlist_candidates(
        trade_date=date(2026, 5, 7),
        pattern_frame=pattern_frame,
        phase1_predictions=phase1,
        phase2_predictions=phase2,
        phase4_predictions=phase4,
        phase7_prediction=phase7,
        phase_filter_rate=0.2,
        phase4_top_n=2,
    )

    assert payload["selection_policy"]["limit_up_filter_threshold"] == 0.099
    assert payload["filter_summary"]["limit_up_excluded_gt_9_9pct"] == 1
    assert payload["filter_summary"]["hard_filter_pass_count_before_limit_up_filter"] == 4
    assert payload["filter_summary"]["hard_filter_pass_count"] == 3
    assert [item["symbol"] for item in payload["candidates"]] == ["600003"]
    assert all(item["symbol"] != "600000" for item in payload["candidates"])
    assert payload["candidates"][0]["source"] == "pattern"
    assert payload["candidates"][0]["涨幅%"] == 2.0201
