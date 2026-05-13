from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
from openpyxl import Workbook

from stocks_analyzer.config import load_config
from stocks_analyzer.intraday_screening import _select_top20_focus, run_intraday_screening
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakePredictionResult:
    predictions: pd.DataFrame


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_storage(tmp_path: Path) -> Storage:
    config = load_config(ROOT / "config" / "default.yaml")
    return Storage(ProjectPaths(tmp_path, config.storage))


def _daily_bars(symbol: str) -> pd.DataFrame:
    dates = pd.date_range("2026-03-01", periods=70, freq="D")
    rows = []
    for index, trade_date in enumerate(dates):
        close = 10.0 + index * 0.03
        rows.append(
            {
                "trade_date": trade_date,
                "symbol": symbol,
                "open": close - 0.05,
                "high": close + 0.12,
                "low": close - 0.16,
                "close": close,
                "volume": 100000 + index * 100,
                "amount": close * (100000 + index * 100),
                "pct_change": 0.2,
                "change": 0.03,
                "amplitude": 1.0,
                "turnover": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _write_watchlist(project_root: Path) -> Path:
    target = project_root / "reports" / "watchlists" / "watchlist_2026-05-07.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "symbol": "600001",
                        "name": "测试股份",
                        "source": "pattern",
                        "pattern_match": True,
                        "pattern_id": "5",
                        "pattern_ids": ["5"],
                        "reason": "previous pattern reason",
                        "phase1_score_100": 40.0,
                        "phase2_score_100": 50.0,
                        "phase4_score_100": 60.0,
                        "连续上榜天数": 3,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return target


def _save_intraday(storage: Storage) -> None:
    storage.save_intraday_bars(
        "600001",
        pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "symbol": "600001",
                    "name": "测试股份",
                    "open": 12.0,
                    "high": 12.6,
                    "low": 11.8,
                    "close": 12.4,
                    "pre_close": 12.1,
                    "volume": 200000,
                    "amount": 2480000,
                    "pct_change": None,
                    "quote_datetime": "2026-05-08 13:30:00",
                    "quote_time": "13:30:00",
                    "source": "sina_raw",
                    "fetched_at": "2026-05-08T13:31:00",
                    "provisional": True,
                }
            ]
        ),
    )


def test_intraday_screening_combines_intraday_bar_and_previous_watchlist(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_screening")
    storage = _make_storage(tmp_path)
    source_watchlist = _write_watchlist(tmp_path)
    storage.save_universe(pd.DataFrame([{"symbol": "600001", "name": "测试股份"}]))
    storage.save_daily_bars("600001", _daily_bars("600001"))
    _save_intraday(storage)
    report_dir = tmp_path / "reports" / "intraday_screening"
    report_dir.mkdir(parents=True, exist_ok=True)
    old_intermediate = report_dir / "intraday_all_tail_risk_predictions_2026-05-01.csv"
    old_final = report_dir / "intraday_screening_2026-05-01.csv"
    old_focus = report_dir / "intraday_screening_focus_2026-05-01.csv"
    old_top10 = report_dir / "intraday_top10_2026-05-01.csv"
    unrelated = report_dir / "manual_note.csv"
    for path in (old_intermediate, old_final, old_focus, old_top10, unrelated):
        path.write_text("x\n", encoding="utf-8")

    def fake_tail(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        bars = storage.load_daily_bars("600001")
        assert pd.Timestamp(bars.iloc[-1]["trade_date"]).date().isoformat() == "2026-05-08"
        assert float(bars.iloc[-1]["close"]) == 12.4
        frame = pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "feature_trade_date": "2026-05-08",
                    "symbol": "600001",
                    "name": "测试股份",
                    "risk_score": 0.25,
                    "model_name": "tail",
                    "model_version": "v1",
                }
            ]
        )
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    def fake_barrier(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        frame = pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "feature_trade_date": "2026-05-08",
                    "symbol": "600001",
                    "name": "测试股份",
                    "barrier_risk_score": 0.35,
                    "is_cusum_event": 1,
                    "model_name": "barrier",
                    "model_version": "v1",
                }
            ]
        )
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    def fake_return(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        frame = pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "feature_trade_date": "2026-05-08",
                    "symbol": "600001",
                    "name": "测试股份",
                    "return_score": 0.45,
                    "model_name": "return",
                    "model_version": "v1",
                }
            ]
        )
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_tail_risk", fake_tail)
    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_barrier_risk", fake_barrier)
    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_alpha158_qlib_return", fake_return)

    result = run_intraday_screening(
        storage=storage,
        project_root=tmp_path,
        trade_date=pd.Timestamp("2026-05-08").date(),
        skip_intraday_update=True,
        report_keep_dates=1,
    )

    assert result.source_watchlist_path == source_watchlist
    assert result.candidate_count == 1
    assert result.top20_path.exists()
    assert result.cleaned_report_files == 4
    assert result.phase1_path.parent == tmp_path / "data" / "intraday" / "screening_cache"
    assert not old_intermediate.exists()
    assert not old_final.exists()
    assert not old_focus.exists()
    assert not old_top10.exists()
    assert unrelated.exists()
    output = pd.read_csv(result.output_path)
    assert output.loc[0, "symbol"] == 600001
    assert output.loc[0, "name"] == "测试股份"
    assert output.loc[0, "intraday_source"] == "sina_raw"
    assert round(float(output.loc[0, "intraday_pct_change"]), 2) == 2.48
    assert output.loc[0, "phase1_score_100"] == 100.0
    assert output.loc[0, "phase2_score_100"] == 100.0
    assert output.loc[0, "phase4_score_100"] == 100.0
    assert output.loc[0, "intraday_focus_score"] == 112.0
    assert 0 < output.loc[0, "建议总仓位%"] <= 40.0
    assert output.loc[0, "phase1_rank"] == 1
    assert output.loc[0, "phase2_rank"] == 1
    assert output.loc[0, "phase4_rank"] == 1
    assert output.loc[0, "prev_pattern_id"] == 5
    assert output.loc[0, "prev_reason"] == "previous pattern reason"
    assert "phase5_score_100" in output.columns
    assert "phase7_score_100" not in output.columns
    assert "intraday_open" not in output.columns
    assert "intraday_high" not in output.columns
    assert "intraday_low" not in output.columns
    assert "intraday_close" not in output.columns
    assert "intraday_volume" not in output.columns
    assert "phase1_risk_score" not in output.columns
    assert "phase2_barrier_risk_score" not in output.columns
    assert "phase4_return_score" not in output.columns
    assert "name_x" not in output.columns
    assert "name_y" not in output.columns
    top20 = pd.read_csv(result.top20_path)
    assert top20.loc[0, "symbol"] == 600001
    assert 0 < top20.loc[0, "建议总仓位%"] <= 40.0
    focus_payload = json.loads((tmp_path / "data" / "intraday" / "focus_top20.json").read_text(encoding="utf-8"))
    assert focus_payload["symbols"] == ["600001"]


def test_intraday_screening_prioritizes_previous_focus_before_remaining(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_focus_first")
    storage = _make_storage(tmp_path)
    _write_watchlist(tmp_path)
    storage.save_universe(
        pd.DataFrame(
            [
                {"symbol": "600001", "name": "测试股份"},
                {"symbol": "600002", "name": "焦点股份"},
            ]
        )
    )
    for symbol in ("600001", "600002"):
        storage.save_daily_bars(symbol, _daily_bars(symbol))
        storage.save_intraday_bars(
            symbol,
            pd.DataFrame(
                [
                    {
                        "trade_date": "2026-05-08",
                        "symbol": symbol,
                        "name": f"测试{symbol}",
                        "open": 12.0,
                        "high": 12.6,
                        "low": 11.8,
                        "close": 12.4,
                        "pre_close": 12.1,
                        "volume": 200000,
                        "amount": 2480000,
                        "pct_change": 2.48,
                        "quote_datetime": "2026-05-08 13:30:00",
                        "quote_time": "13:30:00",
                        "source": "sina_raw",
                        "fetched_at": "2026-05-08T13:31:00",
                        "provisional": True,
                    }
                ]
            ),
        )
    focus_path = tmp_path / "data" / "intraday" / "focus_top10.json"
    focus_path.write_text(json.dumps({"symbols": ["600002"]}), encoding="utf-8")
    calls: list[list[str]] = []

    def fake_frame(symbols: list[str], score_column: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "feature_trade_date": "2026-05-08",
                    "symbol": symbol,
                    "name": f"测试{symbol}",
                        score_column: 0.1 if score_column != "return_score" and symbol == "600001" else 0.9,
                    "is_cusum_event": 0,
                    "model_name": score_column,
                    "model_version": "v1",
                }
                for symbol in symbols
            ]
        )

    def fake_tail(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        symbols = storage.load_universe()["symbol"].astype(str).tolist()
        calls.append(symbols)
        frame = fake_frame(symbols, "risk_score")
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    def fake_barrier(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        symbols = storage.load_universe()["symbol"].astype(str).tolist()
        frame = fake_frame(symbols, "barrier_risk_score")
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    def fake_return(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        symbols = storage.load_universe()["symbol"].astype(str).tolist()
        frame = fake_frame(symbols, "return_score")
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_tail_risk", fake_tail)
    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_barrier_risk", fake_barrier)
    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_alpha158_qlib_return", fake_return)

    result = run_intraday_screening(
        storage=storage,
        project_root=tmp_path,
        trade_date=pd.Timestamp("2026-05-08").date(),
        skip_intraday_update=True,
    )

    assert calls[0] == ["600002"]
    assert calls[1] == ["600001"]
    assert result.focus_output_path is not None
    assert result.focus_output_path.exists()
    assert result.focus_output_path.name == "intraday_top20_previous_2026-05-08.csv"
    focus_output = pd.read_csv(result.focus_output_path)
    assert focus_output["symbol"].astype(str).str.zfill(6).tolist() == ["600002"]
    output = pd.read_csv(result.output_path)
    assert output["symbol"].astype(str).str.zfill(6).tolist() == ["600001", "600002"]
    top20 = pd.read_csv(result.top20_path)
    assert top20["symbol"].astype(str).str.zfill(6).tolist() == ["600001"]
    focus_payload = json.loads((tmp_path / "data" / "intraday" / "focus_top20.json").read_text(encoding="utf-8"))
    assert focus_payload["symbols"] == ["600001"]


def test_select_top20_focus_excludes_weak_scores_and_intraday_gain_above_8_percent() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "phase1_score_100": 80.0,
                "phase2_score_100": 70.0,
                "phase4_score_100": 60.0,
                "intraday_pct_change": 2.0,
            },
            {
                "symbol": "600002",
                "phase1_score_100": 85.0,
                "phase2_score_100": 75.0,
                "phase4_score_100": 99.0,
                "intraday_pct_change": 9.2,
            },
            {
                "symbol": "600003",
                "phase1_score_100": 90.0,
                "phase2_score_100": 80.0,
                "phase4_score_100": 80.0,
                "intraday_pct_change": 8.0,
            },
            {
                "symbol": "600004",
                "phase1_score_100": 40.0,
                "phase2_score_100": 99.0,
                "phase4_score_100": 100.0,
                "intraday_pct_change": 1.0,
            },
            {
                "symbol": "600005",
                "phase1_score_100": 41.0,
                "phase2_score_100": 41.0,
                "phase4_score_100": 90.0,
                "intraday_pct_change": 1.5,
            },
            {
                "symbol": "600006",
                "phase1_score_100": 95.0,
                "phase2_score_100": 95.0,
                "phase4_score_100": 80.0,
                "intraday_pct_change": 1.0,
            },
            {
                "symbol": "600007",
                "phase1_score_100": 5.0,
                "phase2_score_100": 5.0,
                "phase4_score_100": 85.0,
                "intraday_pct_change": 1.0,
                "prev_pattern_match": True,
            },
        ]
    )

    top20 = _select_top20_focus(frame)

    assert top20["symbol"].tolist() == ["600004", "600003", "600006", "600007"]
    scores = dict(zip(top20["symbol"], top20["intraday_focus_score"]))
    assert scores["600004"] == 109.04
    assert scores["600003"] == 98.4
    assert scores["600006"] == 94.0
    assert scores["600007"] == 85.0


def test_intraday_screening_appends_track_stock_to_top20(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_track_stock")
    storage = _make_storage(tmp_path)
    _write_watchlist(tmp_path)
    storage.save_universe(
        pd.DataFrame(
            [
                {"symbol": "600001", "name": "强势股份"},
                {"symbol": "600002", "name": "跟踪股份"},
            ]
        )
    )
    for symbol in ("600001", "600002"):
        storage.save_daily_bars(symbol, _daily_bars(symbol))
        storage.save_intraday_bars(
            symbol,
            pd.DataFrame(
                [
                    {
                        "trade_date": "2026-05-08",
                        "symbol": symbol,
                        "name": f"测试{symbol}",
                        "open": 12.0,
                        "high": 12.6,
                        "low": 11.8,
                        "close": 12.4,
                        "pre_close": 12.1,
                        "volume": 200000,
                        "amount": 2480000,
                        "pct_change": 2.48,
                        "quote_datetime": "2026-05-08 13:30:00",
                        "quote_time": "13:30:00",
                        "source": "sina_raw",
                        "fetched_at": "2026-05-08T13:31:00",
                        "provisional": True,
                    }
                ]
            ),
        )
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet["A1"] = "股票代码"
    sheet["A2"] = "600002"
    workbook.save(tmp_path / "track_stock.xlsx")

    def fake_frame(symbols: list[str], score_column: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "feature_trade_date": "2026-05-08",
                    "symbol": symbol,
                    "name": f"测试{symbol}",
                    score_column: 0.9 if (score_column == "return_score" or symbol == "600002") else 0.1,
                    "is_cusum_event": 0,
                    "model_name": score_column,
                    "model_version": "v1",
                }
                for symbol in symbols
            ]
        )

    def fake_tail(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        symbols = storage.load_universe()["symbol"].astype(str).tolist()
        frame = fake_frame(symbols, "risk_score")
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    def fake_barrier(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        symbols = storage.load_universe()["symbol"].astype(str).tolist()
        frame = fake_frame(symbols, "barrier_risk_score")
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    def fake_return(*, storage, project_root, trade_date, output, limit=None, **kwargs):
        symbols = storage.load_universe()["symbol"].astype(str).tolist()
        frame = fake_frame(symbols, "return_score")
        frame.to_csv(output, index=False)
        return FakePredictionResult(frame)

    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_tail_risk", fake_tail)
    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_barrier_risk", fake_barrier)
    monkeypatch.setattr("stocks_analyzer.intraday_screening.predict_alpha158_qlib_return", fake_return)

    result = run_intraday_screening(
        storage=storage,
        project_root=tmp_path,
        trade_date=pd.Timestamp("2026-05-08").date(),
        skip_intraday_update=True,
    )

    top20 = pd.read_csv(result.top20_path, dtype={"symbol": str})
    assert top20["symbol"].astype(str).str.zfill(6).tolist() == ["600001", "600002"]
    assert top20["建议总仓位%"].notna().all()
    assert top20.loc[top20["symbol"].eq("600002"), "intraday_selection_source"].iloc[0] == "track_stock"
    assert bool(top20.loc[top20["symbol"].eq("600002"), "track_stock"].iloc[0]) is True
    assert result.track_stock_path is not None
    track = pd.read_csv(result.track_stock_path, dtype={"symbol": str})
    assert track["symbol"].astype(str).str.zfill(6).tolist() == ["600002"]
    assert track["建议总仓位%"].notna().all()
