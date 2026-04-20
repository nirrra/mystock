from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from stocks_analyzer.cli import _run_intraday_screening
from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage
from stocks_analyzer.watchlist import write_watchlist


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_intraday_screening_uses_latest_prior_watchlist(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_screening")
    config = load_config(ROOT / "config" / "default.yaml")
    assert config.intraday_provider == "itick"
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    write_watchlist(
        project_root=tmp_path,
        trade_date=date(2026, 4, 10),
        picker_payload={
            "source_file": "demo.csv",
            "candidates": [
                {"symbol": "002579", "name": "中京电子", "tier": "第一梯队", "pattern_id": "1", "tradingview_label": "buy", "tradingview_avg_5d": 0.42},
                {"symbol": "603803", "name": "瑞斯康达", "tier": "第二梯队", "pattern_id": "2", "tradingview_label": "strong_buy", "tradingview_avg_5d": 0.50},
            ],
        },
    )

    def should_not_run(*args, **kwargs) -> None:
        raise AssertionError("legacy daily/incremental intraday path should not be called")

    def fake_save_intraday_rankings(
        *,
        trade_date,
        intraday_provider,
        adjust,
        watchlist_payload,
        output_path,
    ) -> dict[str, object]:
        assert intraday_provider == "itick"
        received_symbols = [item["symbol"] for item in watchlist_payload["candidates"]]
        assert received_symbols == ["002579", "603803"]
        Path(output_path).write_text("代码,5分钟分数\n002579,55.0\n", encoding="utf-8")
        return {
            "output_path": Path(output_path),
            "ranking": None,
            "processed_count": 1,
            "failed_symbols": [{"symbol": "603803", "name": "瑞斯康达", "error": "network down"}],
            "failed_count": 1,
        }

    monkeypatch.setattr("stocks_analyzer.cli._run_update", should_not_run)
    monkeypatch.setattr("stocks_analyzer.cli._run_tradingview", should_not_run)
    monkeypatch.setattr("stocks_analyzer.cli._run_macd", should_not_run)
    monkeypatch.setattr("stocks_analyzer.cli._run_pattern", should_not_run)
    monkeypatch.setattr("stocks_analyzer.cli.save_intraday_rankings", fake_save_intraday_rankings)

    result = _run_intraday_screening(
        storage=storage,
        config=config,
        project_root=tmp_path,
        trade_date=date(2026, 4, 11),
        watchlist_date=None,
        start_date="20240101",
        top_n=20,
    )

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["watchlist_date"] == "2026-04-10"
    assert report["symbol_count"] == 2
    assert report["successful_symbol_count"] == 1
    assert report["failed_symbol_count"] == 1
    assert report["failed_symbols"] == [{"symbol": "603803", "name": "瑞斯康达", "error": "network down"}]
    assert Path(report["intraday_rank_path"]).name == "intraday_rank_2026-04-11.csv"
    assert "tradingview_path" not in report
    assert "macd_path" not in report
    assert "pattern_path" not in report
