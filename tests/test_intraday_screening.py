from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.intraday_screening import run_intraday_screening
from stocks_analyzer.route_watchlists import watchlist_sector_leader_pool_path


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_intraday_screening_uses_sector_leader_pool_without_track_stock(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_sector_pool")
    pool_path = watchlist_sector_leader_pool_path(tmp_path, date(2026, 5, 7))
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_payload = {
        "trade_date": "2026-05-07",
        "selection_policy": {"source_scope": "daily_sector_leader_pool"},
        "sectors": [
            {
                "sector_name": "机器人",
                "sector_type": "concept",
                "sector_label": "robot",
                "long_mainline_score_100": 90,
                "short_mainline_score_100": 88,
                "phase9_score_100": 80,
                "pool_reason": "短期强度Top20",
                "leaders": [{"symbol": "002579", "name": "中京电子"}],
            }
        ],
        "candidates": [
            {
                "symbol": "002579",
                "name": "中京电子",
                "source": "sector_leader_pool",
                "pool_route": "短期强度Top20",
                "matched_mainline_sector": "机器人",
                "source_sectors": "机器人",
                "leader_score": 86,
                "phase1_score_100": 70,
                "phase2_score_100": 72,
                "phase4_score_100": 75,
            }
        ],
    }
    pool_path.write_text(json.dumps(pool_payload, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("stocks_analyzer.intraday_screening.run_intraday_update", lambda **kwargs: type("R", (), {"updated_symbols": [], "failed_symbols": []})())
    monkeypatch.setattr("stocks_analyzer.intraday_screening._run_candidate_analysis", lambda **kwargs: _fake_analysis(tmp_path))
    monkeypatch.setattr("stocks_analyzer.intraday_screening._load_intraday_snapshot_frame", lambda storage, symbols: pd.DataFrame({"symbol": symbols, "pct_change": [1.2] * len(symbols)}))
    monkeypatch.setattr("stocks_analyzer.intraday_screening._load_latest_sector_watchlist", lambda **kwargs: {"trade_date": "2026-05-07"})
    monkeypatch.setattr("stocks_analyzer.intraday_screening._ensure_intraday_sector_phase9_predictions", lambda **kwargs: None)
    monkeypatch.setattr("stocks_analyzer.intraday_screening.build_sector_tracking_payload_from_files", lambda **kwargs: {"sectors": []})
    monkeypatch.setattr("stocks_analyzer.intraday_screening.write_sector_intraday_tracking_workbook", lambda **kwargs: tmp_path / "sector.xlsx")

    storage = type("Storage", (), {"paths": type("Paths", (), {"intraday_dir": tmp_path / "data" / "intraday"})()})()
    result = run_intraday_screening(storage=storage, project_root=tmp_path, trade_date=date(2026, 5, 8), skip_intraday_update=True)

    assert result.source_pool_path == pool_path
    assert result.output_path.name == "intraday_watchlist_a_2026-05-08.csv"
    assert not (tmp_path / "track_stock.xlsx").exists()
    output = pd.read_csv(result.output_path)
    assert "关切板块" in output.columns
    assert "契合主线" in output.columns


def _fake_analysis(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "symbol": "002579",
                "name": "中京电子",
                "prev_source": "sector_leader_pool",
                "pool_route": "短期强度Top20",
                "matched_mainline_sector": "机器人",
                "source_sectors": "机器人",
                "leader_score": 86,
                "intraday_trade_date": "2026-05-08",
                "intraday_pct_change": 1.2,
                "phase1_score_100": 70,
                "phase2_score_100": 72,
                "phase4_score_100": 75,
                "phase4_5d_mean": 74,
                "phase4_5d_std": 5,
                "atr_pct_14": 3.2,
                "centered_risk_score": 76,
            }
        ]
    )
    path = tmp_path / "dummy.csv"
    return type(
        "Analysis",
        (),
        {
            "frame": frame,
            "phase1": pd.DataFrame(),
            "phase2": pd.DataFrame(),
            "phase4": pd.DataFrame(),
            "macd": pd.DataFrame(),
            "atr": pd.DataFrame(),
            "phase1_path": path,
            "phase2_path": path,
            "phase4_path": path,
        },
    )()
