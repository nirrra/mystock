from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
from openpyxl import Workbook

from stocks_analyzer.config import load_config
from stocks_analyzer.intraday_screening import run_intraday_screening
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage
from stocks_analyzer.watchlist import intraday_pool_path


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


def _write_intraday_pool(project_root: Path) -> Path:
    target = intraday_pool_path(project_root, pd.Timestamp("2026-05-07").date())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "symbol": "600001",
                        "name": "测试股份",
                        "source": "pattern_pool",
                        "source_tags": ["pattern_pool"],
                        "pattern_match": True,
                        "pattern_id": "5",
                        "pattern_ids": ["5"],
                        "reason": "previous pattern reason",
                        "phase1_score_100": 40.0,
                        "phase2_score_100": 50.0,
                        "phase4_score_100": 60.0,
                        "连续上榜天数": 3,
                        "intraday_pool_rank": 1,
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


def test_intraday_screening_combines_intraday_bar_and_previous_pool(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_screening")
    storage = _make_storage(tmp_path)
    source_pool = _write_intraday_pool(tmp_path)
    storage.save_universe(pd.DataFrame([{"symbol": "600001", "name": "测试股份"}]))
    storage.save_daily_bars("600001", _daily_bars("600001"))
    _save_intraday(storage)
    report_dir = tmp_path / "reports" / "intraday_screening"
    report_dir.mkdir(parents=True, exist_ok=True)
    old_intermediate = report_dir / "intraday_all_tail_risk_predictions_2026-05-01.csv"
    old_final = report_dir / "intraday_pool_screening_2026-05-01.csv"
    old_track = report_dir / "intraday_track_stock_2026-05-01.csv"
    unrelated = report_dir / "manual_note.csv"
    for path in (old_intermediate, old_final, old_track, unrelated):
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

    assert result.source_pool_path == source_pool
    assert result.candidate_count == 1
    assert result.pool_candidate_count == 1
    assert result.output_path.name == "intraday_pool_screening_2026-05-08.csv"
    assert result.cleaned_report_files == 3
    assert result.phase1_path.parent == tmp_path / "data" / "intraday" / "screening_cache"
    assert not old_intermediate.exists()
    assert not old_final.exists()
    assert not old_track.exists()
    assert unrelated.exists()
    output = pd.read_csv(result.output_path)
    assert list(output.columns[:5]) == ["intraday_trade_date", "symbol", "name", "intraday_selection_source", "intraday_pct_change"]
    assert output.loc[0, "symbol"] == 600001
    assert output.loc[0, "name"] == "测试股份"
    assert output.loc[0, "intraday_source"] == "sina_raw"
    assert round(float(output.loc[0, "intraday_pct_change"]), 2) == 2.48
    assert output.loc[0, "phase1_score_100"] == 100.0
    assert output.loc[0, "phase2_score_100"] == 100.0
    assert output.loc[0, "phase4_score_100"] == 100.0
    assert output.loc[0, "intraday_pool_score"] == 112.0
    assert output.loc[0, "intraday_selection_source"] == "pattern_pool"
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


def test_intraday_screening_uses_intraday_pool_without_full_market_remaining(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_pool_only")
    storage = _make_storage(tmp_path)
    pool_path = tmp_path / "reports" / "watchlists" / "intraday_pool_2026-05-07.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "symbol": "600002",
                        "name": "池内股份",
                        "source": "p124_top50",
                        "source_tags": ["p124_top50"],
                        "intraday_pool_rank": 1,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    storage.save_universe(
        pd.DataFrame(
            [
                {"symbol": "600001", "name": "池外股份"},
                {"symbol": "600002", "name": "池内股份"},
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
    assert len(calls) == 1
    assert result.source_pool_path == pool_path
    output = pd.read_csv(result.output_path)
    assert output["symbol"].astype(str).str.zfill(6).tolist() == ["600002"]
    assert output.loc[0, "intraday_selection_source"] == "p124_top50"


def test_intraday_screening_prefers_same_day_full_market_pool(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("same_day_full_market_pool")
    storage = _make_storage(tmp_path)
    previous_pool = tmp_path / "reports" / "watchlists" / "intraday_pool_2026-05-07.json"
    today_pool = tmp_path / "reports" / "watchlists" / "intraday_pool_2026-05-08.json"
    today_pool.parent.mkdir(parents=True, exist_ok=True)
    previous_pool.write_text(
        json.dumps({"candidates": [{"symbol": "600001", "name": "旧池", "source": "p124_top50"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    today_pool.write_text(
        json.dumps(
            {
                "selection_policy": {"source_scope": "intraday_full_market"},
                "candidates": [{"symbol": "600002", "name": "全市场新池", "source": "p8_fill"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    storage.save_universe(
        pd.DataFrame(
            [
                {"symbol": "600001", "name": "旧池"},
                {"symbol": "600002", "name": "全市场新池"},
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
    calls: list[list[str]] = []

    def fake_frame(symbols: list[str], score_column: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "feature_trade_date": "2026-05-08",
                    "symbol": symbol,
                    "name": f"测试{symbol}",
                    score_column: 0.9,
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

    assert result.source_pool_path == today_pool
    assert calls[0] == ["600002"]
    output = pd.read_csv(result.output_path)
    assert output["symbol"].astype(str).str.zfill(6).tolist() == ["600002"]
    assert output.loc[0, "intraday_selection_source"] == "p8_fill"


def test_intraday_screening_refreshes_full_market_pool(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("refresh_full_market_pool")
    storage = _make_storage(tmp_path)
    storage.save_universe(
        pd.DataFrame(
            [
                {"symbol": "600001", "name": "全市场甲"},
                {"symbol": "600002", "name": "全市场乙"},
            ]
        )
    )
    for symbol in ("600001", "600002"):
        storage.save_daily_bars(symbol, _daily_bars(symbol))

    def fake_frame(symbols: list[str], score_column: str) -> pd.DataFrame:
        values = {"600001": 0.1, "600002": 0.9}
        return pd.DataFrame(
            [
                {
                    "trade_date": "2026-05-08",
                    "feature_trade_date": "2026-05-08",
                    "symbol": symbol,
                    "name": f"测试{symbol}",
                    score_column: values[symbol],
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
        refresh_full_market_pool=True,
    )

    assert result.full_market_pool_refreshed is True
    assert result.full_market_scanned_count == 2
    assert result.source_pool_path == intraday_pool_path(tmp_path, pd.Timestamp("2026-05-08").date())
    payload = json.loads(result.source_pool_path.read_text(encoding="utf-8"))
    assert payload["selection_policy"]["source_scope"] == "intraday_full_market"
    assert payload["filter_summary"]["full_market_scan_symbols"] == 2
    output = pd.read_csv(result.output_path, dtype={"symbol": str})
    assert set(output["symbol"].str.zfill(6)) == {"600001", "600002"}


def test_intraday_screening_appends_track_stock_to_pool(monkeypatch) -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_track_stock")
    storage = _make_storage(tmp_path)
    _write_intraday_pool(tmp_path)
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

    pool = pd.read_csv(result.output_path, dtype={"symbol": str})
    assert pool["symbol"].astype(str).str.zfill(6).tolist() == ["600001", "600002"]
    assert pool["建议总仓位%"].notna().all()
    assert pool.loc[pool["symbol"].eq("600002"), "intraday_selection_source"].iloc[0] == "track_stock"
    assert bool(pool.loc[pool["symbol"].eq("600002"), "track_stock"].iloc[0]) is True
    assert result.track_stock_path is not None
    track = pd.read_csv(result.track_stock_path, dtype={"symbol": str})
    assert track["symbol"].astype(str).str.zfill(6).tolist() == ["600002"]
    assert track["建议总仓位%"].notna().all()
