from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

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


class _FakeProvider:
    def close(self) -> None:
        return None


def test_run_intraday_screening_uses_latest_prior_watchlist(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_screening")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    storage.save_universe(
        pd.DataFrame(
            [
                {"symbol": "002579", "name": "中京电子"},
                {"symbol": "603803", "name": "瑞斯康达"},
            ]
        )
    )

    write_watchlist(
        project_root=tmp_path,
        trade_date=date(2026, 4, 10),
        picker_payload={
            "source_file": "demo.csv",
            "candidates": [
                {"symbol": "002579", "name": "中京电子", "tier": "第一梯队"},
                {"symbol": "603803", "name": "瑞斯康达", "tier": "第二梯队"},
            ],
        },
    )

    updated_symbols: list[str] = []
    received_symbols: dict[str, list[str]] = {}

    monkeypatch.setattr("stocks_analyzer.cli.create_data_provider", lambda provider_name: _FakeProvider())

    def fake_run_update(storage, provider, exclude_st, adjust, symbol, start_date, end_date, limit, skip_existing) -> None:
        updated_symbols.append(symbol)

    def fake_run_tradingview(*, storage, config, trade_date, top_n, output, symbols=None) -> None:
        received_symbols["tradingview"] = list(symbols or [])
        Path(output).write_text("symbol\n002579\n", encoding="utf-8")

    def fake_run_divergence(*, storage, config, trade_date, top_n, output, symbols=None) -> None:
        received_symbols["divergence"] = list(symbols or [])
        Path(output).write_text("symbol\n002579\n", encoding="utf-8")

    def fake_run_pattern(*, storage, provider_name, config, as_of, selected_patterns, limit, output, plot_all, symbols=None) -> None:
        received_symbols["pattern"] = list(symbols or [])
        Path(output).write_text("symbol\n002579\n", encoding="utf-8")

    monkeypatch.setattr("stocks_analyzer.cli._run_update", fake_run_update)
    monkeypatch.setattr("stocks_analyzer.cli._run_tradingview", fake_run_tradingview)
    monkeypatch.setattr("stocks_analyzer.cli._run_divergence", fake_run_divergence)
    monkeypatch.setattr("stocks_analyzer.cli._run_pattern", fake_run_pattern)

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
    assert updated_symbols == ["002579", "603803"]
    assert received_symbols["tradingview"] == ["002579", "603803"]
    assert received_symbols["divergence"] == ["002579", "603803"]
    assert received_symbols["pattern"] == ["002579", "603803"]
    assert report["watchlist_date"] == "2026-04-10"
    assert report["symbol_count"] == 2
