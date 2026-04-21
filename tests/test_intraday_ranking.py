from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

import stocks_analyzer.intraday_ranking as intraday_ranking
from stocks_analyzer.intraday_ranking import (
    _build_daily_summary_frame,
    _detect_ma_event,
    _detect_macd_cross,
    _detect_macd_divergence,
    _detect_volume_divergence,
    build_watchlist_summary_frame,
    build_intraday_ranking_frame,
    localize_ranking_columns,
)
from stocks_analyzer.watchlist import _stable_score


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_intraday_indicator_frame(
    *,
    low: list[float],
    high: list[float],
    close: list[float],
    volume: list[float] | None = None,
    macd_dif: list[float] | None = None,
    macd_dea: list[float] | None = None,
    macd_hist: list[float] | None = None,
    ma_5: list[float] | None = None,
    ma_10: list[float] | None = None,
    pivot_volume: list[float] | None = None,
) -> pd.DataFrame:
    size = len(close)
    volume_values = volume or [100.0] * size
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-14 09:35:00", periods=size, freq="5min"),
            "low": low,
            "high": high,
            "close": close,
            "volume": volume_values,
            "amount": [close[index] * volume_values[index] for index in range(size)],
            "macd_dif": macd_dif or [0.0] * size,
            "macd_dea": macd_dea or [0.0] * size,
            "macd_hist": macd_hist or [0.0] * size,
            "ma_5": ma_5 or [pd.NA] * size,
            "ma_10": ma_10 or [pd.NA] * size,
            "pivot_volume": pivot_volume or volume_values,
        }
    )


def test_build_daily_summary_frame_reuses_watchlist_stable_score() -> None:
    tmp_path = _make_workspace_tmp_dir("intraday_daily_score")
    tradingview_path = tmp_path / "tradingview.csv"
    divergence_path = tmp_path / "divergence.csv"
    pattern_path = tmp_path / "pattern.csv"

    pd.DataFrame(
        [
            {
                "symbol": '="600000"',
                "name": "测试一",
                "all_rating_2026-04-08": 0.20,
                "all_rating_2026-04-09": 0.35,
                "all_rating_2026-04-10": 0.50,
                "avg_all_rating_5d": 0.40,
                "all_rating": 0.50,
                "all_rating_label": "buy",
            }
        ]
    ).to_csv(tradingview_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "symbol": '="600000"',
                "name": "测试一",
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": True,
            }
        ]
    ).to_csv(divergence_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"symbol": '="600000"', "name": "测试一", "pattern_id": "3"},
            {"symbol": '="600000"', "name": "测试一", "pattern_id": "1"},
        ]
    ).to_csv(pattern_path, index=False, encoding="utf-8-sig")

    result = _build_daily_summary_frame(
        symbols=["600000"],
        tradingview_path=tradingview_path,
        divergence_path=divergence_path,
        pattern_path=pattern_path,
    )

    assert result.loc[0, "pattern_ids"] == "3,1"
    assert result.loc[0, "pattern_id"] == "3"
    expected = round(
        float(
            _stable_score(
                result.loc[0],
                ["tradingview_all_rating_2026-04-08", "tradingview_all_rating_2026-04-09", "tradingview_all_rating_2026-04-10"],
            )
        ),
        4,
    )
    assert result.loc[0, "daily_score"] == expected
    assert bool(result.loc[0, "daily_macd_bottom_divergence_15d"]) is True


def test_detect_volume_divergence_bullish() -> None:
    frame = _make_intraday_indicator_frame(
        low=[14, 12, 10, 12, 14, 13, 9, 13, 14],
        high=[15, 13, 11, 13, 15, 14, 10, 14, 15],
        close=[14.5, 12.5, 10.5, 12.2, 14.2, 13.4, 9.4, 13.1, 14.1],
        volume=[120, 110, 100, 105, 115, 95, 80, 90, 100],
        pivot_volume=[120, 110, 100, 105, 115, 95, 80, 90, 100],
    )

    event_type, score = _detect_volume_divergence(frame)

    assert event_type == "bullish"
    assert score == 12.0


def test_detect_macd_divergence_bullish() -> None:
    frame = _make_intraday_indicator_frame(
        low=[14, 12, 10, 12, 14, 13, 9, 13, 14],
        high=[15, 13, 11, 13, 15, 14, 10, 14, 15],
        close=[14.5, 12.5, 10.5, 12.2, 14.2, 13.4, 9.4, 13.1, 14.1],
        macd_dif=[-0.2, -0.3, -1.8, -0.6, -0.4, -0.3, -1.0, -0.2, 0.1],
    )

    event_type, score = _detect_macd_divergence(frame)

    assert event_type == "bullish"
    assert score == 20.0


def test_detect_macd_cross_continuation() -> None:
    frame = _make_intraday_indicator_frame(
        low=[10, 10, 10, 10, 10],
        high=[11, 11, 11, 11, 11],
        close=[10.0, 10.1, 10.2, 10.4, 10.6],
        macd_dif=[-0.5, -0.4, -0.1, 0.2, 0.4],
        macd_dea=[-0.3, -0.25, -0.15, -0.05, 0.1],
        macd_hist=[-0.2, -0.1, 0.05, 0.2, 0.4],
    )

    event_type, score = _detect_macd_cross(frame)

    assert event_type == "golden_cross_continuation"
    assert score == 18.0


def test_detect_ma_pullback_hold() -> None:
    frame = _make_intraday_indicator_frame(
        low=[9.8, 9.9, 10.0, 10.0, 10.0, 10.1, 10.2, 9.99, 10.1, 10.4],
        high=[10.3, 10.4, 10.5, 10.4, 10.5, 10.6, 10.7, 10.6, 10.9, 11.2],
        close=[10.1, 10.2, 10.3, 10.2, 10.3, 10.4, 10.5, 10.4, 10.6, 11.0],
        ma_5=[10.0] * 10,
        ma_10=[10.0] * 10,
    )

    event_type, score = _detect_ma_event(frame)

    assert event_type == "pullback_hold_ma"
    assert score == 15.0


def test_build_watchlist_summary_frame_extracts_daily_fields_from_watchlist_payload() -> None:
    payload = {
        "candidates": [
            {
                "symbol": "2579",
                "name": "中京电子",
                "pattern_id": "1",
                "tradingview_label": "buy",
                "tradingview_avg_5d": 0.42,
                "macd_top_divergence_15d": True,
                "macd_bottom_divergence_15d": False,
            }
        ]
    }

    result = build_watchlist_summary_frame(payload)

    assert result.loc[0, "symbol"] == "002579"
    assert result.loc[0, "pattern_ids"] == "1"
    assert result.loc[0, "tradingview_all_rating_label"] == "buy"
    assert float(result.loc[0, "tradingview_avg_all_rating_5d"]) == 0.42
    assert bool(result.loc[0, "daily_macd_top_divergence_15d"]) is True


def test_build_watchlist_summary_frame_merges_duplicate_symbols() -> None:
    payload = {
        "candidates": [
            {
                "symbol": "603259",
                "name": "药明康德",
                "pattern_id": "2",
                "tradingview_label": "buy",
                "tradingview_avg_5d": 0.42,
                "macd_top_divergence_15d": False,
                "macd_bottom_divergence_15d": False,
            },
            {
                "symbol": "603259",
                "name": "药明康德",
                "pattern_id": "1",
                "tradingview_label": "buy",
                "tradingview_avg_5d": 0.42,
                "macd_top_divergence_15d": True,
                "macd_bottom_divergence_15d": False,
            },
        ]
    }

    result = build_watchlist_summary_frame(payload)

    assert len(result) == 1
    assert result.loc[0, "symbol"] == "603259"
    assert result.loc[0, "pattern_ids"] == "2,1"
    assert bool(result.loc[0, "daily_macd_top_divergence_15d"]) is True
    assert float(result.loc[0, "tradingview_avg_all_rating_5d"]) == 0.42


def test_build_intraday_ranking_frame_sorts_by_intraday_score_then_symbol() -> None:
    daily_frame = pd.DataFrame(
        [
            {"symbol": "600000", "name": "甲", "pattern_ids": "1"},
            {"symbol": "600001", "name": "乙", "pattern_ids": "2"},
        ]
    )

    def intraday_fetcher(symbol: str) -> pd.DataFrame:
        frame = pd.DataFrame({"placeholder": [1]})
        frame.attrs["symbol"] = symbol
        return frame

    def fake_summarize(frame: pd.DataFrame) -> dict[str, object]:
        symbol = frame.attrs["symbol"]
        score = {"600000": 70.0, "600001": 70.0}[symbol]
        return {
            "intraday_5m_score": score,
            "intraday_volume_divergence_hit": False,
            "intraday_volume_divergence_type": "none",
            "intraday_volume_score": 0.0,
            "intraday_macd_divergence_hit": False,
            "intraday_macd_divergence_type": "none",
            "intraday_macd_divergence_score": 0.0,
            "intraday_macd_cross_hit": False,
            "intraday_macd_cross_type": "none",
            "intraday_macd_cross_score": 0.0,
            "intraday_ma_event_hit": False,
            "intraday_ma_event_type": "none",
            "intraday_ma_score": 0.0,
        }

    original = intraday_ranking.summarize_intraday_events
    intraday_ranking.summarize_intraday_events = fake_summarize
    try:
        result, failed_symbols = build_intraday_ranking_frame(
            symbols=["600000", "600001"],
            daily_frame=daily_frame,
            intraday_fetcher=intraday_fetcher,
        )
    finally:
        intraday_ranking.summarize_intraday_events = original

    assert result["symbol"].tolist() == ["600000", "600001"]
    assert result["rank"].tolist() == [1, 2]
    assert failed_symbols == []


def test_build_intraday_ranking_frame_skips_failed_symbol_and_records_error() -> None:
    daily_frame = pd.DataFrame(
        [
            {"symbol": "600000", "name": "甲", "pattern_ids": "1"},
            {"symbol": "600001", "name": "乙", "pattern_ids": "2"},
        ]
    )

    def intraday_fetcher(symbol: str) -> pd.DataFrame:
        if symbol == "600000":
            raise RuntimeError(f"{symbol} fetch failed")
        frame = pd.DataFrame({"placeholder": [1]})
        frame.attrs["symbol"] = symbol
        return frame

    def fake_summarize(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "intraday_5m_score": 70.0,
            "intraday_volume_divergence_hit": False,
            "intraday_volume_divergence_type": "none",
            "intraday_volume_score": 0.0,
            "intraday_macd_divergence_hit": False,
            "intraday_macd_divergence_type": "none",
            "intraday_macd_divergence_score": 0.0,
            "intraday_macd_cross_hit": False,
            "intraday_macd_cross_type": "none",
            "intraday_macd_cross_score": 0.0,
            "intraday_ma_event_hit": False,
            "intraday_ma_event_type": "none",
            "intraday_ma_score": 0.0,
        }

    original = intraday_ranking.summarize_intraday_events
    intraday_ranking.summarize_intraday_events = fake_summarize
    try:
        result, failed_symbols = build_intraday_ranking_frame(
            symbols=["600000", "600001"],
            daily_frame=daily_frame,
            intraday_fetcher=intraday_fetcher,
        )
    finally:
        intraday_ranking.summarize_intraday_events = original

    assert result["symbol"].tolist() == ["600001"]
    assert failed_symbols == [{"symbol": "600000", "name": "甲", "error": "600000 fetch failed"}]


def test_localize_ranking_columns_renames_export_headers_to_chinese() -> None:
    frame = pd.DataFrame(
        [
            {
                "rank": 1,
                "symbol": "600000",
                "name": "测试一",
                "pattern_ids": "1",
                "intraday_5m_score": 66.0,
                "intraday_macd_cross_type": "golden_cross",
            }
        ]
    )

    result = localize_ranking_columns(frame)

    assert result.columns.tolist() == ["排名", "代码", "名称", "形态", "5分钟分数", "5分钟金叉死叉类型"]
