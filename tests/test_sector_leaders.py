from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.sector_leaders import analyze_sector_leaders, detect_sector_swings
from stocks_analyzer.sector_membership import sector_membership_path


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_daily(root: Path, symbol: str, dates: pd.DatetimeIndex, returns: list[float], *, amount: float = 1000.0) -> None:
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
                "volume": amount / max(close, 0.01),
                "amount": amount,
                "pct_change": return_pct,
            }
        )
    pd.DataFrame(rows).to_parquet(root / f"{symbol}.parquet", index=False)


def test_detect_sector_swings_finds_rising_windows_and_removes_overlaps() -> None:
    dates = pd.bdate_range("2025-01-01", periods=80)
    returns = [0.0] * 10 + [1.2] * 12 + [-0.3] * 20 + [1.0] * 10 + [0.0] * 28
    index = (1.0 + pd.Series(returns, index=dates) / 100.0).cumprod() * 100.0

    swings = detect_sector_swings(index, min_length=5, max_length=30, low_to_high_threshold=0.08)

    assert swings
    assert all(swing.end_pos > swing.start_pos for swing in swings)
    assert max(swing.return_pct for swing in swings) >= 8.0


def test_analyze_sector_leaders_outputs_long_term_and_swing_lists() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_leaders")
    daily_dir = tmp_path / "data" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    membership_file = sector_membership_path(tmp_path)
    membership_file.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2025-01-01", periods=100)
    steady = [0.35] * len(dates)
    swing = [0.0] * 35 + [4.0] * 8 + [0.0] * (len(dates) - 43)
    flat = [0.0] * len(dates)
    _write_daily(daily_dir, "000001", dates, steady, amount=5000.0)
    _write_daily(daily_dir, "000002", dates, swing, amount=3000.0)
    _write_daily(daily_dir, "000003", dates, flat, amount=1000.0)
    _write_daily(daily_dir, "000004", dates, steady, amount=1000.0)

    pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "稳健龙头",
                "sector_type": "concept",
                "sector_name": "测试概念",
                "sector_label": "C001",
                "source": "test",
                "updated_at": "2026-05-17T00:00:00",
            },
            {
                "symbol": "000002",
                "name": "波段先锋",
                "sector_type": "concept",
                "sector_name": "测试概念",
                "sector_label": "C001",
                "source": "test",
                "updated_at": "2026-05-17T00:00:00",
            },
            {
                "symbol": "000003",
                "name": "普通成员",
                "sector_type": "concept",
                "sector_name": "测试概念",
                "sector_label": "C001",
                "source": "test",
                "updated_at": "2026-05-17T00:00:00",
            },
            {
                "symbol": "000004",
                "name": "小板块成员",
                "sector_type": "concept",
                "sector_name": "过小概念",
                "sector_label": "C002",
                "source": "test",
                "updated_at": "2026-05-17T00:00:00",
            },
        ]
    ).to_csv(membership_file, index=False, encoding="utf-8-sig")

    result = analyze_sector_leaders(
        project_root=tmp_path,
        trade_date=date(2025, 5, 20),
        daily_dir=daily_dir,
        lookback_days=100,
        min_history_days=40,
        min_valid_members=2,
        top_n=2,
        progress=False,
    )

    assert result.trade_date == date(2025, 5, 20)
    assert result.row_count == 4
    assert result.all_score_row_count == 3
    assert result.summary_row_count == 1
    assert result.skipped_count == 1
    assert result.output_path.exists()
    assert result.all_scores_path.exists()
    assert result.summary_path.exists()
    assert result.skipped_path.exists()

    leaders = pd.read_csv(result.output_path)
    assert set(leaders["leader_type"]) == {"long_term", "swing"}
    assert leaders["leader_score"].between(0, 100).all()
    assert leaders["swing_count"].max() >= 1
    assert "long_term_leader_score" in leaders.columns
    assert "swing_leader_score" in leaders.columns
    assert leaders.loc[leaders["leader_type"].eq("swing"), "swing_leader_score"].max() > 0

    all_scores = pd.read_csv(result.all_scores_path)
    assert len(all_scores) == 3
    assert set(all_scores["name"]) == {"稳健龙头", "波段先锋", "普通成员"}
    assert all_scores["combined_leader_score"].between(0, 100).all()
    assert {"long_term_rank", "swing_rank", "combined_rank"}.issubset(all_scores.columns)

    summary = pd.read_csv(result.summary_path)
    assert summary.loc[0, "sector_name"] == "测试概念"
    assert "波段先锋" in summary.loc[0, "swing_top5"]
