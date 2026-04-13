from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.watchlist import (
    build_watchlist_candidates,
    extract_watchlist_symbols,
    find_latest_watchlist_before,
    load_watchlist,
    write_watchlist,
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
                "macd_top_divergence_15d": True,
                "macd_bottom_divergence_15d": False,
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
                "reason": "demo-2",
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
