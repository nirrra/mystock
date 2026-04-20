from __future__ import annotations

import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.trend_threshold_research import (
    build_default_threshold_candidates,
    build_combo_threshold_candidates,
    derive_threshold_candidates,
    evaluate_combo_thresholds,
    evaluate_threshold_candidates,
    sample_threshold_research_entries,
    summarize_indicator_distributions,
)


def _load_test_config():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return load_config(root / "config" / "default.yaml")


def _make_research_dataset() -> pd.DataFrame:
    rows = []
    base_returns = [-0.08, -0.03, 0.01, 0.04, 0.09]
    signal_types = ["breakout", "pullback", "breakout", "pullback", "breakout"]
    for holding_days in [5, 10]:
        for index, return_pct in enumerate(base_returns):
            rows.append(
                {
                    "dataset_split": "all_period",
                    "trade_date": pd.Timestamp("2025-01-31"),
                    "planned_entry_date": pd.Timestamp("2025-02-03"),
                    "entry_date": pd.Timestamp("2025-02-03"),
                    "exit_date": pd.Timestamp("2025-02-10"),
                    "holding_days": holding_days,
                    "symbol": f"6000{index:02d}",
                    "name": f"样本{index}",
                    "signal_type": signal_types[index],
                    "setup_type": signal_types[index],
                    "entry_score": 60 + index * 5,
                    "trend_score": 55 + index * 5,
                    "trend_base_score": 58 + index * 4,
                    "price_action_score": 52 + index * 8,
                    "macd_score": 30 + index * 10,
                    "volume_score": 45 + index * 6,
                    "volume_price_divergence_score": 48 + index * 5,
                    "boll_score": 50 + index * 6,
                    "rsi_score": 53 + index * 5,
                    "kdj_score": 47 + index * 7,
                    "atr_score": 42 + index * 4,
                    "buy_score": 62 + index * 7,
                    "positive_indicator_count": 1 + index,
                    "return_pct": return_pct if holding_days == 5 else return_pct + 0.01,
                    "max_drawdown_pct": 0.02 + index * 0.01,
                    "max_upside_pct": 0.03 + index * 0.02,
                    "min_return_pct": min(return_pct, 0.0),
                    "entry_note": "next_open",
                    "entry_timing": "next_open",
                }
            )
    return pd.DataFrame(rows)


def test_sample_threshold_research_entries_keeps_last_date_per_month() -> None:
    entries = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2025-01-03"), "buy_score": 60.0, "symbol": "600000"},
            {"trade_date": pd.Timestamp("2025-01-31"), "buy_score": 65.0, "symbol": "600001"},
            {"trade_date": pd.Timestamp("2025-02-07"), "buy_score": 70.0, "symbol": "600002"},
            {"trade_date": pd.Timestamp("2025-02-28"), "buy_score": 75.0, "symbol": "600003"},
        ]
    )

    sampled = sample_threshold_research_entries(entries, sample_mode="monthly")

    assert sampled["trade_date"].tolist() == [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")]


def test_derive_threshold_candidates_uses_distribution_quantiles() -> None:
    distributions = summarize_indicator_distributions(_make_research_dataset())

    candidates = derive_threshold_candidates(distributions)

    assert not candidates.empty
    buy_score_balanced = candidates[
        (candidates["dataset_split"] == "all_period")
        & (candidates["holding_days"] == 5)
        & (candidates["signal_scope"] == "all")
        & (candidates["metric"] == "buy_score")
        & (candidates["candidate_type"] == "balanced")
    ]
    assert len(buy_score_balanced) == 1
    assert float(buy_score_balanced.iloc[0]["threshold"]) >= 69.0


def test_evaluate_threshold_candidates_filters_selected_rows() -> None:
    dataset = _make_research_dataset()
    distributions = summarize_indicator_distributions(dataset)
    candidates = derive_threshold_candidates(distributions)

    evaluation = evaluate_threshold_candidates(dataset, candidates)

    assert not evaluation.empty
    row = evaluation[
        (evaluation["dataset_split"] == "all_period")
        & (evaluation["holding_days"] == 5)
        & (evaluation["signal_scope"] == "all")
        & (evaluation["metric"] == "buy_score")
        & (evaluation["candidate_type"] == "strict")
    ].iloc[0]
    assert int(row["selected_count"]) > 0
    assert float(row["coverage"]) <= 1.0


def test_combo_threshold_candidates_include_current_defaults() -> None:
    config = _load_test_config()
    distributions = summarize_indicator_distributions(_make_research_dataset())
    candidates = derive_threshold_candidates(distributions)

    combo_candidates = build_combo_threshold_candidates(candidates, config)
    combo_evaluation = evaluate_combo_thresholds(_make_research_dataset(), combo_candidates)

    assert "current_default_rules" in combo_candidates["combo_name"].tolist()
    assert not combo_evaluation.empty


def test_build_default_threshold_candidates_returns_signal_specific_rows() -> None:
    config = _load_test_config()
    dataset = _make_research_dataset()
    distributions = summarize_indicator_distributions(dataset)
    candidates = derive_threshold_candidates(distributions)
    combo_candidates = build_combo_threshold_candidates(candidates, config)
    combo_evaluation = evaluate_combo_thresholds(dataset, combo_candidates)

    default_candidates = build_default_threshold_candidates(candidates, combo_candidates, combo_evaluation)

    assert not default_candidates.empty
    assert set(default_candidates["signal_scope"].tolist()) == {"breakout", "pullback"}
    assert "recommended_combo_name" in default_candidates.columns
    assert default_candidates["buy_score_min"].notna().any()
