from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.stacked_trade_value import (
    HORIZONS,
    HorizonSpec,
    add_v41_long_quality_labels,
    add_v4_risk_upside_labels,
    add_all_horizon_labels,
    add_path_label,
    apply_v41_risk_gate_decision,
    apply_v42_opportunity_decision,
    build_v42_opportunity_frame,
    build_stacked_feature_frame,
    compare_v42_topn_metrics,
    long_quality_ranker_feature_columns,
    opportunity_gate_feature_columns,
    long_upside_feature_columns,
    predict_opportunity_ranker,
    realized_horizon_value,
    score_v42_v4_rank_frame,
    train_opportunity_ranker_model,
)
from stocks_analyzer.storage import Storage


ROOT = Path(__file__).resolve().parents[1]


def _make_workspace_tmp_dir(name: str) -> Path:
    path = ROOT / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _flat_bars(length: int = 90) -> pd.DataFrame:
    trade_date = pd.date_range("2024-01-01", periods=length, freq="B")
    close = np.full(length, 100.0)
    return pd.DataFrame(
        {
            "trade_date": trade_date,
            "symbol": ["600000"] * length,
            "open": close,
            "close": close,
            "high": close,
            "low": close,
            "volume": np.full(length, 5_000_000.0),
            "amount": np.full(length, 500_000_000.0),
            "pct_change": np.zeros(length),
            "change": np.zeros(length),
            "amplitude": np.zeros(length),
            "turnover": np.ones(length),
        }
    )


def _make_daily_bars(symbol: str, seed: int, length: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trade_date = pd.date_range("2024-01-01", periods=length, freq="B")
    trend = np.linspace(0, 35 + seed * 2, length)
    seasonal = np.sin(np.linspace(0, 10 * np.pi, length)) * (4 + seed)
    noise = rng.normal(0, 1.8, length)
    close = np.maximum(40 + trend + seasonal + noise, 5)
    open_ = close + rng.normal(0, 0.8, length)
    high = np.maximum(open_, close) + rng.uniform(0.2, 2.5, length)
    low = np.minimum(open_, close) - rng.uniform(0.2, 2.5, length)
    for spike_index in range(35, length, 47):
        high[spike_index] = max(high[spike_index], open_[spike_index] * 1.18)
    for dip_index in range(55, length, 53):
        low[dip_index] = min(low[dip_index], open_[dip_index] * 0.88)
    volume = rng.integers(3_000_000, 8_000_000, size=length).astype(float)
    amount = close * volume * 100
    return pd.DataFrame(
        {
            "trade_date": trade_date,
            "symbol": [symbol] * length,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": amount,
            "pct_change": pd.Series(close).pct_change().fillna(0.0),
            "change": pd.Series(close).diff().fillna(0.0),
            "amplitude": (high - low) / close,
            "turnover": np.ones(length),
        }
    )


def test_path_label_uses_next_open_and_detects_up_first() -> None:
    bars = _flat_bars()
    bars.loc[1, "open"] = 100.0
    bars.loc[2, "high"] = 109.0
    frame = add_path_label(bars, HorizonSpec(5, 0.08, 0.05, 0.10, "5d"))

    assert frame.loc[0, "entry_open_5d"] == 100.0
    assert frame.loc[0, "outcome_5d"] == "up"
    assert frame.loc[0, "outcome_class_5d"] == 0
    assert frame.loc[0, "hit_day_5d"] == 2
    assert frame.loc[0, "value_5d"] > 0


def test_trade_value_weight_order_matches_design() -> None:
    frame = _flat_bars(length=260)
    labeled = add_all_horizon_labels(frame)
    row = labeled.dropna(subset=["trade_value"]).iloc[0]
    expected = sum(spec.trade_value_weight * row[f"value_{spec.name}"] for spec in HORIZONS)
    assert row["trade_value"] == expected
    assert [spec.trade_value_weight for spec in HORIZONS] == [0.10, 0.20, 0.40, 0.30]


def test_block60_features_include_three_prior_blocks() -> None:
    features = build_stacked_feature_frame(_make_daily_bars("600000", seed=1, length=260))
    latest = features.iloc[-1]
    for column in ("block60_0_return", "block60_1_return", "block60_2_return"):
        assert pd.notna(latest[column])


def test_realized_horizon_value_penalizes_early_down_more_than_late_down() -> None:
    spec = HorizonSpec(20, 0.15, 0.08, 0.40, "20d")
    early = realized_horizon_value(outcome="down", period_return=-0.08, max_drawdown=0.08, hit_day=1, spec=spec)
    late = realized_horizon_value(outcome="down", period_return=-0.08, max_drawdown=0.08, hit_day=20, spec=spec)
    assert early < late


def test_v4_risk_upside_labels_use_short_risk_but_long_target() -> None:
    frame = _flat_bars(length=260)
    labeled = add_all_horizon_labels(frame)
    short_risk_index = labeled.index[0]
    long_good_index = labeled.index[1]

    labeled.loc[short_risk_index, "outcome_5d"] = "down"
    labeled.loc[short_risk_index, "outcome_20d"] = "up"
    labeled.loc[short_risk_index, "outcome_60d"] = "up"
    labeled.loc[short_risk_index, "period_return_20d"] = 0.16
    labeled.loc[short_risk_index, "period_return_60d"] = 0.32
    labeled.loc[short_risk_index, "max_upside_20d"] = 0.17
    labeled.loc[short_risk_index, "max_upside_60d"] = 0.34
    labeled.loc[short_risk_index, "max_drawdown_20d"] = 0.02
    labeled.loc[short_risk_index, "max_drawdown_60d"] = 0.04
    labeled.loc[short_risk_index, "hit_day_20d"] = 5
    labeled.loc[short_risk_index, "hit_day_60d"] = 20

    labeled.loc[long_good_index, "outcome_5d"] = "neutral"
    labeled.loc[long_good_index, "outcome_10d"] = "neutral"
    labeled.loc[long_good_index, "outcome_20d"] = "up"
    labeled.loc[long_good_index, "outcome_60d"] = "up"
    labeled.loc[long_good_index, "period_return_20d"] = 0.16
    labeled.loc[long_good_index, "period_return_60d"] = 0.32
    labeled.loc[long_good_index, "max_upside_20d"] = 0.17
    labeled.loc[long_good_index, "max_upside_60d"] = 0.34
    labeled.loc[long_good_index, "max_drawdown_20d"] = 0.02
    labeled.loc[long_good_index, "max_drawdown_60d"] = 0.04
    labeled.loc[long_good_index, "hit_day_20d"] = 5
    labeled.loc[long_good_index, "hit_day_60d"] = 20

    result = add_v4_risk_upside_labels(labeled)

    assert result.loc[short_risk_index, "bad_risk"] == 1.0
    assert result.loc[long_good_index, "bad_risk"] == 0.0
    assert result.loc[short_risk_index, "long_upside_value"] == result.loc[long_good_index, "long_upside_value"]


def test_v41_long_quality_labels_rank_only_candidates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02"] * 4),
            "symbol": ["600000", "600001", "600002", "600003"],
            "action": ["candidate", "candidate", "candidate", "avoid"],
            "period_return_20d": [0.02, 0.08, 0.18, 0.30],
            "period_return_60d": [0.03, 0.16, 0.36, 0.50],
            "max_upside_20d": [0.04, 0.10, 0.20, 0.34],
            "max_upside_60d": [0.06, 0.20, 0.40, 0.60],
            "max_drawdown_20d": [0.03, 0.02, 0.01, 0.01],
            "max_drawdown_60d": [0.06, 0.04, 0.02, 0.02],
        }
    )

    result = add_v41_long_quality_labels(frame)

    assert result.loc[0, "long_quality_grade"] < result.loc[1, "long_quality_grade"]
    assert result.loc[1, "long_quality_grade"] < result.loc[2, "long_quality_grade"]
    assert pd.isna(result.loc[3, "long_quality_grade"])
    assert pd.isna(result.loc[3, "long_quality_rank_pct"])


def test_long_upside_feature_columns_exclude_short_stage1_outputs() -> None:
    frame = pd.DataFrame(
        {
            "long_up_prob_20d": [0.6],
            "long_down_prob_20d": [0.2],
            "long_neutral_prob_20d": [0.2],
            "long_expected_value_20d": [0.3],
            "long_risk_adjusted_value_20d": [0.36],
            "long_up_prob_60d": [0.5],
            "long_down_prob_60d": [0.2],
            "long_neutral_prob_60d": [0.3],
            "long_expected_value_60d": [0.26],
            "long_risk_adjusted_value_60d": [0.26],
            "long_stage2_edge_20d": [0.4],
            "long_stage2_edge_60d": [0.3],
            "long_stage2_weighted_expected_value": [0.276],
            "up_prob_5d": [0.9],
            "down_prob_10d": [0.8],
            "return_20d": [0.1],
            "return_60d": [0.2],
            "macd_hist": [0.01],
        }
    )

    columns = long_upside_feature_columns(frame)

    assert "long_up_prob_20d" in columns
    assert "long_up_prob_60d" in columns
    assert "return_20d" in columns
    assert "up_prob_5d" not in columns
    assert "down_prob_10d" not in columns


def test_long_quality_ranker_features_include_rank_context_without_label_leakage() -> None:
    frame = pd.DataFrame(
        {
            "long_quality": [0.2],
            "long_quality_rank_pct": [1.0],
            "long_quality_grade": [4],
            "risk_score": [0.12],
            "down_prob_20d": [0.20],
            "long_up_prob_20d": [0.6],
            "cs_rank_risk_score": [0.1],
            "cs_rank_long_up_prob_20d": [0.8],
            "return_20d": [0.1],
        }
    )

    columns = long_quality_ranker_feature_columns(frame)

    assert "risk_score" in columns
    assert "cs_rank_risk_score" in columns
    assert "cs_rank_long_up_prob_20d" in columns
    assert "long_quality" not in columns
    assert "long_quality_rank_pct" not in columns
    assert "long_quality_grade" not in columns


def test_v41_risk_gate_filters_before_long_quality_ranking() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02"] * 4),
            "risk_score": [0.1, 0.2, 0.3, 0.4],
            "long_quality_score": [0.1, 0.4, 1.0, 2.0],
            "down_prob_20d": [0.3, 0.3, 0.7, 0.3],
            "down_prob_60d": [0.2, 0.2, 0.2, 0.5],
            "stage2_weighted_down_prob": [0.25, 0.25, 0.25, 0.25],
        }
    )
    result = apply_v41_risk_gate_decision(
        frame,
        {
            "risk_percentile": 1.0,
            "risk_score_max": None,
            "down20_max": 0.5,
            "down60_max": 0.4,
            "weighted_down_max": 0.34,
            "min_candidates_per_day": 1,
        },
    )

    assert result["action"].tolist() == ["candidate", "candidate", "avoid", "avoid"]
    assert result.loc[2, "risk_gate_reason"] == "down20_cap"
    assert result.loc[3, "risk_gate_reason"] == "down60_cap"
    assert result.loc[1, "risk_action"] == "pass"


def test_v42_opportunity_frame_labels_good_and_bad_days() -> None:
    rows: list[dict[str, object]] = []
    for day, strong in [(pd.Timestamp("2026-01-02"), True), (pd.Timestamp("2026-01-05"), False)]:
        for idx in range(60):
            is_top = idx < 20
            if strong:
                period_return = 0.05 if is_top else 0.01
                outcome = "up" if idx < 8 else "neutral"
                drawdown = 0.025
            else:
                period_return = -0.03 if is_top else -0.01
                outcome = "down" if idx < 8 else "neutral"
                drawdown = 0.09
            rows.append(
                {
                    "trade_date": day,
                    "symbol": f"60{idx:04d}",
                    "dataset_split": "train_oof",
                    "action": "candidate",
                    "risk_score": 0.2 + idx / 1000,
                    "down_prob_20d": 0.10,
                    "down_prob_60d": 0.08,
                    "stage2_weighted_down_prob": 0.09,
                    "long_upside_score": 1.0 - idx / 100,
                    "period_return_20d": period_return,
                    "max_drawdown_20d": drawdown,
                    "outcome_20d": outcome,
                    "trade_value": period_return,
                    "return_20d": 0.02,
                    "return_60d": 0.04,
                    "distance_to_ma20": 0.03,
                    "distance_to_ma60": 0.04,
                    "macd_hist": 0.01,
                }
            )

    opportunity = build_v42_opportunity_frame(pd.DataFrame(rows))
    features = opportunity_gate_feature_columns(opportunity)

    assert opportunity["good_opportunity_day"].tolist() == [1, 0]
    assert opportunity.loc[0, "candidate_count"] == 60
    assert opportunity.loc[0, "top20_avg_return_20d"] > 0
    assert "candidate_count" in features
    assert "top20_avg_return_20d" not in features
    assert "good_opportunity_day" not in features


def test_v42_opportunity_decision_can_block_whole_day() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02"] * 3 + ["2026-01-05"] * 3),
            "symbol": ["600000", "600001", "600002", "600003", "600004", "600005"],
            "action": ["candidate", "candidate", "avoid", "candidate", "candidate", "avoid"],
            "risk_action": ["pass", "pass", "block", "pass", "pass", "block"],
            "final_score_v42": [0.3, 0.2, 1.0, 0.5, 0.4, 2.0],
        }
    )
    opportunity = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "opportunity_score": [0.7, 0.2],
            "good_opportunity_day": [1, 0],
            "opportunity_quality": [0.4, -0.2],
        }
    )

    result = apply_v42_opportunity_decision(frame, opportunity, {"opportunity_threshold": 0.5})

    assert result.loc[:2, "trade_permission"].eq("allow").all()
    assert result.loc[3:, "trade_permission"].eq("no_trade").all()
    assert result["action"].tolist() == ["candidate", "candidate", "avoid", "no_trade", "no_trade", "avoid"]
    assert result.loc[0, "action_rank_v42"] == 0
    assert result.loc[3, "action_rank_v42"] == 1


def test_v42_v4_rank_frame_reuses_long_upside_score() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02"] * 3),
            "symbol": ["600000", "600001", "600002"],
            "action": ["candidate", "candidate", "avoid"],
            "long_upside_score": [0.2, 0.5, 0.9],
        }
    )

    result = score_v42_v4_rank_frame(frame)

    assert result["rank_source_v42"].eq("v4_long_upside_score").all()
    assert result["final_score_v42"].tolist() == [0.2, 0.5, 0.9]
    assert result["opportunity_rank_score"].tolist() == [0.2, 0.5, 0.9]
    assert result.loc[1, "buy_score_v42"] > result.loc[0, "buy_score_v42"]


def test_v42_comparison_includes_gate_v4_rank_variant() -> None:
    rows: list[dict[str, object]] = []
    for day in pd.to_datetime(["2026-01-02", "2026-01-05"]):
        for idx in range(3):
            rows.append(
                {
                    "trade_date": day,
                    "dataset_split": "test",
                    "symbol": f"60000{idx}",
                    "action": "candidate",
                    "risk_action": "pass",
                    "long_upside_score": 1.0 - idx * 0.1,
                    "final_score_v42": 0.5 - idx * 0.1,
                    "opportunity_rank_score": 0.5 - idx * 0.1,
                    "period_return_20d": 0.05 - idx * 0.01,
                    "max_drawdown_20d": 0.02,
                    "outcome_20d": "up" if idx == 0 else "neutral",
                    "bad_risk": 0.0,
                    "risk_score": 0.1,
                    "trade_value": 0.05 - idx * 0.01,
                }
                | {
                    f"period_return_{suffix}": 0.05 - idx * 0.01
                    for suffix in ("5d", "10d", "60d")
                }
                | {
                    f"max_drawdown_{suffix}": 0.02
                    for suffix in ("5d", "10d", "60d")
                }
                | {
                    f"outcome_{suffix}": "up" if idx == 0 else "neutral"
                    for suffix in ("5d", "10d", "60d")
                }
            )
    baseline = pd.DataFrame(rows)
    hybrid = score_v42_v4_rank_frame(baseline)
    hybrid = apply_v42_opportunity_decision(
        hybrid,
        pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                "opportunity_score": [0.8, 0.2],
                "good_opportunity_day": [1, 0],
                "opportunity_quality": [0.1, -0.1],
            }
        ),
        {"opportunity_threshold": 0.5},
    )

    comparison = compare_v42_topn_metrics(
        baseline_scored=baseline,
        v42_scored=baseline,
        hybrid_v4_scored=hybrid,
        top_n_list=(2,),
    )

    assert "v42_gate_v4_rank" in set(comparison["model_version"])
    hybrid_row = comparison[comparison["model_version"].eq("v42_gate_v4_rank")].iloc[0]
    assert hybrid_row["days"] == 1


def test_opportunity_ranker_training_and_prediction_workflow() -> None:
    tmp_path = _make_workspace_tmp_dir("v42_opportunity_ranker")
    config = load_config(ROOT / "config" / "default.yaml")
    paths = ProjectPaths(tmp_path, config.storage)
    storage = Storage(paths)

    universe = pd.DataFrame(
        [
            {"symbol": "600000", "name": "测试一"},
            {"symbol": "600001", "name": "测试二"},
            {"symbol": "600002", "name": "测试三"},
        ]
    )
    storage.save_universe(universe)
    for idx, symbol in enumerate(universe["symbol"], start=1):
        storage.save_daily_bars(symbol, _make_daily_bars(symbol, seed=idx))

    result = train_opportunity_ranker_model(
        storage=storage,
        config=config,
        project_root=tmp_path,
        max_iter=2,
        start_date=None,
        end_date=None,
        train_end=None,
        valid_end=None,
        test_end=None,
        limit=None,
        top_n_list=(2,),
        prediction_date=date(2025, 5, 16),
    )

    assert result.model_path.exists()
    assert result.model_path.name == "v42_opportunity_ranker.pkl"
    assert result.metadata_path.exists()
    assert not result.opportunity_metrics.empty
    assert result.selected_hybrid_opportunity_params
    assert result.prediction_path is not None and result.prediction_path.exists()
    predictions = predict_opportunity_ranker(
        storage=storage,
        config=config,
        project_root=tmp_path,
        trade_date=date(2025, 5, 16),
    )
    assert not predictions.empty
    assert {
        "trade_permission",
        "opportunity_score",
        "opportunity_threshold",
        "opportunity_rank_score",
        "final_score_v42",
        "buy_score_v42",
        "rank_source_v42",
    }.issubset(predictions.columns)
    hybrid_predictions = predict_opportunity_ranker(
        storage=storage,
        config=config,
        project_root=tmp_path,
        trade_date=date(2025, 5, 16),
        rank_source="v4",
    )
    assert not hybrid_predictions.empty
    assert hybrid_predictions["rank_source_v42"].eq("v4_long_upside_score").all()
