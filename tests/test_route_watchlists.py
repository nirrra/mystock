from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.route_watchlists import build_route_watchlists


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_a_routes_keep_broad_candidates_for_review_selection() -> None:
    project_root = _make_workspace_tmp_dir("route_watchlists_broad_candidates")
    trade_date = date(2026, 5, 15)
    sector_payload = {
        "sectors": [
            {
                "sector_type": "concept",
                "sector_label": "a1",
                "sector_name": "近期主线",
                "selected_as_long_mainline": True,
                "long_mainline_score_100": 90,
                "short_mainline_score_100": 95,
                "phase9_score_100": 30,
            },
            {
                "sector_type": "concept",
                "sector_label": "a2",
                "sector_name": "轮转主线",
                "selected_as_long_mainline": True,
                "long_mainline_score_100": 88,
                "short_mainline_score_100": 20,
                "phase9_score_100": 98,
            },
        ]
    }
    concern_members = pd.DataFrame(
        [
            {"编号": "000001", "名称": "A1宽候选", "板块类型": "concept", "板块名称": "近期主线", "板块代码": "a1", "龙头指数": 61},
            {"编号": "000002", "名称": "A1风险低", "板块类型": "concept", "板块名称": "近期主线", "板块代码": "a1", "龙头指数": 95},
            {"编号": "000003", "名称": "A2宽候选", "板块类型": "concept", "板块名称": "轮转主线", "板块代码": "a2", "龙头指数": 61},
            {"编号": "000004", "名称": "A2风险低", "板块类型": "concept", "板块名称": "轮转主线", "板块代码": "a2", "龙头指数": 95},
        ]
    )
    stock_scores = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A1宽候选",
                "phase1_score_100": 25,
                "phase2_score_100": 25,
                "phase4_score_100": 10,
                "centered_risk_score": 50,
                "daily_return_pct": 1.0,
                "limit_up_excluded_by_daily_return": False,
            },
            {
                "symbol": "000002",
                "name": "A1风险低",
                "phase1_score_100": 19,
                "phase2_score_100": 90,
                "phase4_score_100": 99,
                "centered_risk_score": 99,
                "daily_return_pct": 1.0,
                "limit_up_excluded_by_daily_return": False,
            },
            {
                "symbol": "000003",
                "name": "A2宽候选",
                "phase1_score_100": 45,
                "phase2_score_100": 45,
                "phase4_score_100": 10,
                "centered_risk_score": 60,
                "daily_return_pct": 1.0,
                "limit_up_excluded_by_daily_return": False,
            },
            {
                "symbol": "000004",
                "name": "A2风险低",
                "phase1_score_100": 39,
                "phase2_score_100": 90,
                "phase4_score_100": 99,
                "centered_risk_score": 99,
                "daily_return_pct": 1.0,
                "limit_up_excluded_by_daily_return": False,
            },
        ]
    )

    payloads = build_route_watchlists(
        trade_date=trade_date,
        project_root=project_root,
        stock_scores=stock_scores,
        sector_payload=sector_payload,
        concern_members=concern_members,
    )

    a1_symbols = {item["symbol"] for item in payloads["a1"]["candidates"]}
    assert "000001" in a1_symbols
    assert "000002" not in a1_symbols
    assert [item["symbol"] for item in payloads["a2"]["candidates"]] == ["000003"]
