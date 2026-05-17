from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.sector_watchlist import build_sector_tracking_payload, build_sector_watchlist


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_sector_watchlist_uses_topn_mainline_pool_and_marks_short_phase9_inside_pool() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_watchlist")
    trade_date = date(2026, 5, 15)
    performance = pd.DataFrame(
        [
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": "concept",
                "sector_label": "A",
                "sector_name": "长期短期共振",
                "member_count": 12,
                "valid_count": 12,
                "avg_pct_change": 2.0,
                "amount_weighted_pct_change": 5.0,
                "up_ratio": 0.8,
            },
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": "concept",
                "sector_label": "B",
                "sector_name": "长期未启动",
                "member_count": 12,
                "valid_count": 12,
                "avg_pct_change": -1.0,
                "amount_weighted_pct_change": -2.0,
                "up_ratio": 0.35,
            },
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": "concept",
                "sector_label": "C",
                "sector_name": "短期强但非长期",
                "member_count": 12,
                "valid_count": 12,
                "avg_pct_change": 4.0,
                "amount_weighted_pct_change": 5.0,
                "up_ratio": 0.9,
            },
        ]
    )
    mainline = pd.DataFrame(
        [
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": "concept",
                "sector_label": "A",
                "sector_name": "长期短期共振",
                "long_mainline_score": 91.0,
                "return_5d": 8.0,
                "return_20d": 18.0,
                "ma5_slope_pct": 1.2,
            },
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": "concept",
                "sector_label": "B",
                "sector_name": "长期未启动",
                "long_mainline_score": 92.0,
                "return_5d": -3.0,
                "return_20d": -8.0,
                "ma5_slope_pct": -0.8,
            },
            {
                "trade_date": trade_date.isoformat(),
                "sector_type": "concept",
                "sector_label": "C",
                "sector_name": "短期强但非长期",
                "long_mainline_score": 89.0,
                "return_5d": 8.0,
                "return_20d": 18.0,
                "ma5_slope_pct": 1.2,
            },
        ]
    )
    phase9 = pd.DataFrame(
        [
            {"sector_type": "concept", "sector_label": "A", "phase9_score_100": 75.0, "phase9_rank": 1},
            {"sector_type": "concept", "sector_label": "B", "phase9_score_100": 50.0, "phase9_rank": 2},
            {"sector_type": "concept", "sector_label": "C", "phase9_score_100": 99.0, "phase9_rank": 3},
        ]
    )

    payload = build_sector_watchlist(
        project_root=tmp_path,
        trade_date=trade_date,
        sector_performance=performance,
        sector_leaders=pd.DataFrame(),
        phase9_predictions=phase9,
        mainline_scores=mainline,
    )

    assert payload["selection_policy"]["long_mainline_top_n"] == 100
    assert payload["selection_policy"]["short_mainline_top_n"] == 10
    assert payload["selection_policy"]["phase9_buy_top_n"] == 10
    assert [item["sector_label"] for item in payload["sectors"]] == ["B", "A", "C"]
    assert [item["long_mainline_rank"] for item in payload["sectors"]] == [1, 2, 3]
    assert all(item["selected_as_long_mainline"] is True for item in payload["sectors"])
    assert all(item["selected_as_short_mainline"] is True for item in payload["sectors"])
    assert all(item["selected_as_phase9_buy"] is True for item in payload["sectors"])
    assert payload["mainline_strength_summary"]["selected_count"] == 3

    tracking_payload = build_sector_tracking_payload(
        project_root=tmp_path,
        trade_date=trade_date,
        sector_performance=performance,
        sector_leaders=pd.DataFrame(),
        phase9_predictions=phase9,
        mainline_scores=mainline,
    )
    assert [item["sector_label"] for item in tracking_payload["sectors"]] == ["B", "A", "C"]
    assert tracking_payload["selection_policy"]["scope"] == "all_sectors"
