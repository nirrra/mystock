from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.sector_membership import sector_membership_path
from stocks_analyzer.sector_phase9 import (
    add_sector_rule_buy_score_columns,
    build_sector_phase9_panel,
    build_sector_phase9_walkforward_windows,
    filter_long_mainline_rows,
    sector_phase9_feature_columns,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_daily_from_close(root: Path, symbol: str, dates: pd.DatetimeIndex, closes: list[float]) -> None:
    rows = []
    previous = None
    for trade_date, close in zip(dates, closes):
        pct_change = 0.0 if previous is None else (float(close) / float(previous) - 1.0) * 100.0
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
                "pct_change": pct_change,
            }
        )
        previous = close
    pd.DataFrame(rows).to_parquet(root / f"{symbol}.parquet", index=False)


def test_sector_phase9_label_uses_20th_future_close_not_intraperiod_max() -> None:
    tmp_path = _make_workspace_tmp_dir("sector_phase9_label")
    daily_dir = tmp_path / "data" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    membership_file = sector_membership_path(tmp_path)
    membership_file.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2025-01-01", periods=32)

    spike_then_fade = [100.0] + [101.0 + i for i in range(10)] + [109.0, 108.0, 107.0, 106.0, 105.5, 105.0, 104.8, 104.5, 104.2, 104.0]
    spike_then_fade.extend([104.0] * (len(dates) - len(spike_then_fade)))
    steady_advance = [100.0 + 0.3 * i for i in range(len(dates))]
    _write_daily_from_close(daily_dir, "000001", dates, spike_then_fade)
    _write_daily_from_close(daily_dir, "000002", dates, steady_advance)

    pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "冲高回落",
                "sector_type": "concept",
                "sector_name": "冲高概念",
                "sector_label": "C001",
                "source": "test",
                "updated_at": "2026-05-16T00:00:00",
            },
            {
                "symbol": "000002",
                "name": "稳步上涨",
                "sector_type": "concept",
                "sector_name": "稳步概念",
                "sector_label": "C002",
                "source": "test",
                "updated_at": "2026-05-16T00:00:00",
            },
        ]
    ).to_csv(membership_file, index=False, encoding="utf-8-sig")

    panel = build_sector_phase9_panel(
        project_root=tmp_path,
        daily_dir=daily_dir,
        start_date=date(2025, 1, 1),
        end_date=dates[-1].date(),
        history_days=40,
        min_members=1,
        horizon_days=20,
        return_threshold=0.05,
        min_feature_history_days=0,
    )

    assert not panel.dataset.empty
    faded = panel.dataset[
        panel.dataset["sector_name"].eq("冲高概念")
        & panel.dataset["trade_date"].dt.date.eq(dates[0].date())
    ].iloc[0]
    assert faded["future_max_return_20d"] > 0.05
    assert faded["future_return_20d_close"] < 0.05
    assert faded["phase9_label"] == 0

    advanced = panel.dataset[
        panel.dataset["sector_name"].eq("稳步概念")
        & panel.dataset["trade_date"].dt.date.eq(dates[0].date())
    ].iloc[0]
    assert advanced["future_return_20d_close"] >= 0.05
    assert advanced["phase9_label"] == 1
    assert "return_1d" in sector_phase9_feature_columns(panel.dataset)

    scored = add_sector_rule_buy_score_columns(panel.dataset)
    assert "rule_buy_score" in scored.columns
    assert scored["rule_buy_score"].between(0, 100).all()

    assert "long_mainline_score" in panel.dataset.columns
    assert panel.dataset["long_mainline_score"].between(0, 100).all()
    mainline = filter_long_mainline_rows(panel.dataset, score_threshold=90, top_pct=0.5)
    assert not mainline.empty
    assert mainline["trade_date"].nunique() == panel.dataset["trade_date"].nunique()


def test_sector_phase9_walkforward_windows_respects_embargo() -> None:
    dates = pd.bdate_range("2025-01-01", periods=120)
    dataset = pd.DataFrame(
        {
            "trade_date": dates,
            "phase9_label": [0, 1] * 60,
        }
    )

    windows = build_sector_phase9_walkforward_windows(
        dataset,
        test_start_date=dates[80].date(),
        test_window_days=10,
        step_days=10,
        embargo_days=20,
        min_train_days=40,
    )

    assert not windows.empty
    first = windows.iloc[0]
    assert first["test_start"] == dates[80].date()
    assert first["train_end"] == dates[59].date()
