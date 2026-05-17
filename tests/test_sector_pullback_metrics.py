from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.sector_membership import sector_membership_path
from stocks_analyzer.sector_pullback_metrics import analyze_sector_pullback_metrics


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_daily(root: Path, symbol: str, dates: pd.DatetimeIndex, returns: list[float]) -> None:
    close = 10.0
    rows = []
    for trade_date, return_pct in zip(dates, returns):
        close = close * (1.0 + return_pct / 100.0)
        rows.append(
            {
                "trade_date": trade_date.date(),
                "symbol": symbol,
                "open": close,
                "close": close,
                "high": close,
                "low": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
                "pct_change": return_pct,
            }
        )
    pd.DataFrame(rows).to_parquet(root / f"{symbol}.parquet", index=False)


def test_analyze_sector_pullback_metrics_builds_strength_pullback_and_stability_groups() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_pullback_metrics")
    daily_dir = tmp_path / "data" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    membership_file = sector_membership_path(tmp_path)
    membership_file.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2025-01-01", periods=90)
    strong_returns = [1.0] * 20 + [-1.0] * 20 + [0.2] * 50
    flat_returns = [0.0] * len(dates)
    _write_daily(daily_dir, "000001", dates, strong_returns)
    _write_daily(daily_dir, "000002", dates, strong_returns)
    _write_daily(daily_dir, "000003", dates, flat_returns)

    pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "强势一",
                "sector_type": "concept",
                "sector_name": "强势概念",
                "sector_label": "C001",
                "source": "test",
                "updated_at": "2026-05-16T00:00:00",
            },
            {
                "symbol": "000002",
                "name": "强势二",
                "sector_type": "concept",
                "sector_name": "强势概念",
                "sector_label": "C001",
                "source": "test",
                "updated_at": "2026-05-16T00:00:00",
            },
            {
                "symbol": "000003",
                "name": "平盘",
                "sector_type": "concept",
                "sector_name": "弱势概念",
                "sector_label": "C002",
                "source": "test",
                "updated_at": "2026-05-16T00:00:00",
            },
        ]
    ).to_csv(membership_file, index=False, encoding="utf-8-sig")

    result = analyze_sector_pullback_metrics(
        project_root=tmp_path,
        trade_date=date(2025, 5, 6),
        daily_dir=daily_dir,
        history_days=90,
        strength_lookback_days=60,
        local_peak_window=5,
        slope_lag_days=5,
        min_members=1,
    )

    assert result.trade_date == date(2025, 5, 6)
    assert result.row_count == 2
    assert result.output_path.exists()

    metrics = pd.read_csv(result.output_path)
    strong = metrics.loc[metrics["sector_name"].eq("强势概念")].iloc[0]
    assert "buy_score" in metrics.columns
    assert "stabilization_score" in metrics.columns
    assert "rebound_confirmation_score" in metrics.columns
    assert "no_new_20d_low_in_5d" in metrics.columns
    assert "no_new_60d_low_in_5d" in metrics.columns
    assert "no_new_low_5d" not in metrics.columns
    assert pd.notna(strong["buy_score"])
    assert 0 <= strong["buy_score"] <= 100
    assert strong["max_consecutive_up_days_1y"] >= 49
    assert strong["max_consecutive_outperform_days_1y"] >= 49
    assert strong["outperform_day_ratio_1y"] > 0.7
    assert strong["drawdown_from_peak_pct"] < 0
    assert strong["days_since_peak"] > 0
    assert bool(strong["peak_confirmed"]) is True
    assert strong["ma5_slope_pct"] > 0
