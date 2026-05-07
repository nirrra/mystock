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
    add_v5_volume_price_extreme_risk_flag,
    add_v5_volume_price_features,
    add_v5_volume_price_labels,
    add_v51_candidate_rank_labels,
    add_v51_cross_sectional_rank_features,
    add_all_horizon_labels,
    add_path_label,
    apply_v41_risk_gate_decision,
    apply_v51_blend_score,
    apply_v5_decision,
    apply_v42_opportunity_decision,
    build_v42_opportunity_frame,
    build_stacked_feature_frame,
    compare_v42_topn_metrics,
    fit_v5_fusion_model,
    fit_v51_candidate_ranker_model,
    fit_volume_price_quality_model,
    fit_volume_price_risk_model,
    generate_walkforward_windows,
    long_quality_ranker_feature_columns,
    opportunity_gate_feature_columns,
    long_upside_feature_columns,
    predict_opportunity_ranker,
    realized_horizon_value,
    score_v51_candidate_ranker_frame,
    score_v5_fusion_frame,
    score_volume_price_submodels,
    score_v42_v4_rank_frame,
    select_v51_blend_params,
    summarize_walkforward_topn_metrics,
    train_opportunity_ranker_model,
    v51_candidate_ranker_feature_columns,
    v5_fusion_feature_columns,
    volume_price_feature_columns,
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


def test_v5_volume_price_features_cover_1d_5d_20d_windows() -> None:
    features = build_stacked_feature_frame(_make_daily_bars("600000", seed=2, length=120))
    latest = features.iloc[-1]

    for column in (
        "vp_close_position_1d",
        "vp_volume_ratio_1d_to_20d",
        "vp_5d_price_volume_confirm",
        "vp_20d_accumulation_score",
        "vp_5d_vs_20d_volume_accel",
    ):
        assert column in features.columns
        assert pd.notna(latest[column])

    columns = volume_price_feature_columns(features)
    assert "vp_close_position_1d" in columns
    assert "vp_20d_distribution_score" in columns


def test_v5_extreme_volume_price_risk_flags_high_volume_weak_candle() -> None:
    frame = build_stacked_feature_frame(_flat_bars(length=80))
    idx = frame.index[-1]
    frame.loc[idx, "vp_20d_range_position"] = 0.90
    frame.loc[idx, "vp_volume_ratio_1d_to_20d"] = 2.2
    frame.loc[idx, "vp_amount_ratio_1d_to_20d"] = 2.1
    frame.loc[idx, "vp_upper_shadow_1d"] = 0.55
    frame.loc[idx, "vp_close_position_1d"] = 0.25

    result = add_v5_volume_price_extreme_risk_flag(frame)

    assert bool(result.loc[idx, "volume_price_extreme_risk_flag"]) is True
    assert result.loc[idx, "volume_price_extreme_risk_reason"] == "high_volume_upper_shadow"


def test_v5_volume_price_quality_rewards_healthier_forward_path() -> None:
    frame = build_stacked_feature_frame(_flat_bars(length=260))
    labeled = add_all_horizon_labels(frame)
    good_index = labeled.index[0]
    bad_index = labeled.index[1]

    labeled.loc[good_index, "period_return_20d"] = 0.14
    labeled.loc[good_index, "period_return_60d"] = 0.24
    labeled.loc[good_index, "max_upside_20d"] = 0.18
    labeled.loc[good_index, "max_upside_60d"] = 0.30
    labeled.loc[good_index, "max_drawdown_20d"] = 0.025
    labeled.loc[good_index, "max_drawdown_60d"] = 0.05
    labeled.loc[good_index, "outcome_20d"] = "up"
    labeled.loc[good_index, "outcome_60d"] = "up"

    labeled.loc[bad_index, "period_return_20d"] = -0.04
    labeled.loc[bad_index, "period_return_60d"] = -0.08
    labeled.loc[bad_index, "max_upside_20d"] = 0.03
    labeled.loc[bad_index, "max_upside_60d"] = 0.05
    labeled.loc[bad_index, "max_drawdown_20d"] = 0.11
    labeled.loc[bad_index, "max_drawdown_60d"] = 0.18
    labeled.loc[bad_index, "outcome_20d"] = "down"
    labeled.loc[bad_index, "outcome_60d"] = "down"

    result = add_v5_volume_price_labels(add_v4_risk_upside_labels(labeled))

    assert result.loc[good_index, "volume_price_quality_value"] > result.loc[bad_index, "volume_price_quality_value"]
    assert result.loc[bad_index, "volume_price_risk_label"] == 1.0


def test_v5_submodels_and_fusion_score_generate_rankable_fields() -> None:
    rows: list[dict[str, object]] = []
    for day in pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]):
        for idx in range(6):
            good = idx >= 3
            rows.append(
                {
                    "trade_date": day,
                    "symbol": f"6000{idx:02d}",
                    "action": "candidate",
                    "risk_score": 0.15 + idx * 0.02,
                    "long_upside_score": 0.2 + idx * 0.1,
                    "opportunity_score": 0.8,
                    "down_prob_20d": 0.18,
                    "down_prob_60d": 0.12,
                    "stage2_weighted_down_prob": 0.16,
                    "long_stage2_weighted_up_prob": 0.40 + idx * 0.02,
                    "long_stage2_weighted_down_prob": 0.10,
                    "long_stage2_weighted_expected_value": 0.20 + idx * 0.03,
                    "long_upside_value": 0.2 + idx * 0.1,
                    "bad_risk": 0.0 if good else 1.0,
                    "period_return_20d": 0.08 if good else -0.03,
                    "period_return_60d": 0.18 if good else -0.05,
                    "max_upside_20d": 0.12 if good else 0.03,
                    "max_upside_60d": 0.24 if good else 0.05,
                    "max_drawdown_20d": 0.03 if good else 0.12,
                    "max_drawdown_60d": 0.06 if good else 0.18,
                    "outcome_20d": "up" if good else "down",
                    "outcome_60d": "up" if good else "down",
                    "vp_close_position_1d": 0.75 if good else 0.25,
                    "vp_upper_shadow_1d": 0.10 if good else 0.55,
                    "vp_signed_body_1d": 0.30 if good else -0.30,
                    "vp_volume_ratio_1d_to_20d": 1.1 if good else 2.0,
                    "vp_amount_ratio_1d_to_20d": 1.1 if good else 2.0,
                    "vp_5d_price_volume_confirm": 0.10 if good else -0.02,
                    "vp_5d_volume_without_price": 0.00 if good else 0.20,
                    "vp_5d_shrink_pullback_score": 0.04 if good else 0.00,
                    "vp_5d_high_volume_weak_days": 0.0 if good else 2.0,
                    "vp_20d_accumulation_score": 0.8 if good else 0.2,
                    "vp_20d_distribution_score": 0.1 if good else 0.7,
                    "vp_20d_up_down_volume_ratio": 1.5 if good else 0.6,
                    "vp_5d_vs_20d_return_accel": 0.03 if good else -0.02,
                    "vp_5d_vs_20d_volume_accel": 0.05 if good else 0.20,
                    "vp_volume_accel_without_price": 0.00 if good else 0.10,
                    "vp_short_shrink_after_strength": 0.03 if good else 0.00,
                    "vp_pullback_depth_in_20d": 0.20 if good else 1.20,
                }
            )
    frame = add_v5_volume_price_labels(pd.DataFrame(rows))
    vp_features = volume_price_feature_columns(frame)
    risk_model = fit_volume_price_risk_model(frame, feature_columns=vp_features, max_iter=2)
    quality_model = fit_volume_price_quality_model(frame, feature_columns=vp_features, max_iter=2)
    scored = score_volume_price_submodels(
        frame,
        risk_model=risk_model,
        quality_model=quality_model,
        feature_columns=vp_features,
    )
    fusion_ready = add_v5_volume_price_labels(scored)
    fusion_ready["v5_fusion_value"] = (
        fusion_ready["long_upside_value"] + fusion_ready["volume_price_quality_value"] - fusion_ready["bad_risk"]
    )
    fusion_features = v5_fusion_feature_columns(fusion_ready)
    fusion_model = fit_v5_fusion_model(fusion_ready, feature_columns=fusion_features, max_iter=2)
    result = apply_v5_decision(score_v5_fusion_frame(fusion_ready, fusion_model=fusion_model, fusion_features=fusion_features))

    assert {"volume_price_risk_score", "volume_price_quality_score", "final_score_v5", "buy_score_v5"}.issubset(result.columns)
    assert result.loc[result["symbol"].eq("600005"), "buy_score_v5"].iloc[0] > result.loc[
        result["symbol"].eq("600000"), "buy_score_v5"
    ].iloc[0]


def test_v51_candidate_rank_labels_are_daily_candidate_only() -> None:
    rows: list[dict[str, object]] = []
    for day in pd.to_datetime(["2026-01-02", "2026-01-05"]):
        for idx in range(10):
            rows.append(
                {
                    "trade_date": day,
                    "symbol": f"600{idx:03d}",
                    "action": "candidate" if idx < 9 else "avoid",
                    "trade_permission": "allow",
                    "volume_price_extreme_risk_flag": False,
                    "period_return_20d": -0.04 + idx * 0.02,
                    "period_return_60d": -0.06 + idx * 0.03,
                    "max_drawdown_20d": 0.12 - idx * 0.01,
                    "bad_risk": 0.0 if idx >= 5 else 1.0,
                    "outcome_20d": "up" if idx >= 7 else ("down" if idx <= 1 else "neutral"),
                }
            )
    frame = pd.DataFrame(rows)

    result = add_v51_candidate_rank_labels(frame)

    day = result[result["trade_date"].eq(pd.Timestamp("2026-01-02"))]
    assert day.loc[day["symbol"].eq("600000"), "v51_rank_grade"].iloc[0] == 0
    assert day.loc[day["symbol"].eq("600008"), "v51_rank_grade"].iloc[0] == 4
    assert pd.isna(day.loc[day["symbol"].eq("600009"), "v51_rank_value"].iloc[0])
    assert bool(day.loc[day["symbol"].eq("600009"), "v51_candidate_eligible"].iloc[0]) is False


def test_v51_candidate_ranker_scores_and_blends_candidates() -> None:
    rows: list[dict[str, object]] = []
    for day in pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]):
        for idx in range(8):
            good = idx >= 5
            rows.append(
                {
                    "trade_date": day,
                    "symbol": f"600{idx:03d}",
                    "action": "candidate",
                    "trade_permission": "allow",
                    "volume_price_extreme_risk_flag": False,
                    "long_upside_score": 0.10 + idx * 0.08,
                    "risk_score": 0.35 - idx * 0.02,
                    "opportunity_score": 0.9,
                    "final_score_v42": 0.10 + idx * 0.07,
                    "buy_score_v42": 50 + idx * 5,
                    "opportunity_rank_score_pct": 0.2 + idx * 0.08,
                    "down_prob_20d": 0.25 - idx * 0.01,
                    "down_prob_60d": 0.20 - idx * 0.01,
                    "stage2_weighted_down_prob": 0.22 - idx * 0.01,
                    "volume_price_risk_score": 0.40 - idx * 0.02,
                    "volume_price_quality_score": 0.10 + idx * 0.06,
                    "volume_price_risk_score_pct": 0.80 - idx * 0.05,
                    "volume_price_quality_score_pct": 0.20 + idx * 0.08,
                    "final_score_v5": 0.08 + idx * 0.08,
                    "buy_score_v5": 45 + idx * 6,
                    "vp_close_position_1d": 0.70 if good else 0.30,
                    "vp_signed_body_1d": 0.25 if good else -0.20,
                    "vp_upper_shadow_1d": 0.10 if good else 0.45,
                    "vp_volume_ratio_1d_to_20d": 1.1 if good else 1.8,
                    "vp_5d_price_volume_confirm": 0.08 if good else -0.02,
                    "vp_20d_accumulation_score": 0.75 if good else 0.25,
                    "vp_20d_distribution_score": 0.10 if good else 0.70,
                    "period_return_5d": 0.03 if good else -0.01,
                    "period_return_10d": 0.05 if good else -0.02,
                    "period_return_20d": 0.08 if good else -0.03,
                    "period_return_60d": 0.18 if good else -0.05,
                    "max_drawdown_5d": 0.01 if good else 0.04,
                    "max_drawdown_10d": 0.02 if good else 0.06,
                    "max_drawdown_20d": 0.03 if good else 0.11,
                    "max_drawdown_60d": 0.06 if good else 0.18,
                    "bad_risk": 0.0 if good else 1.0,
                    "outcome_5d": "up" if good else "down",
                    "outcome_10d": "up" if good else "down",
                    "outcome_20d": "up" if good else "down",
                    "outcome_60d": "up" if good else "down",
                    "trade_value": 0.6 if good else -0.4,
                }
            )
    frame = add_v51_cross_sectional_rank_features(add_v51_candidate_rank_labels(pd.DataFrame(rows)))
    features = v51_candidate_ranker_feature_columns(frame)
    ranker_model, engine = fit_v51_candidate_ranker_model(frame, feature_columns=features, max_iter=2)
    scored = score_v51_candidate_ranker_frame(frame, ranker_model=ranker_model, feature_columns=features)
    selected, grid = select_v51_blend_params(scored, top_n_list=(3,))
    result = apply_v51_blend_score(scored, selected)

    assert engine in {"constant", "hist_gradient_boosting_regressor", "lightgbm_lambdarank"}
    assert not grid.empty
    assert "blend_weight" in selected
    assert {"candidate_rank_score_v51", "final_score_v51", "buy_score_v51"}.issubset(result.columns)
    assert result["candidate_rank_score_v51"].notna().all()


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


def test_generate_walkforward_windows_are_chronological() -> None:
    trade_dates = pd.date_range("2025-01-01", periods=120, freq="B")

    windows = generate_walkforward_windows(
        trade_dates,
        windows=3,
        train_days=40,
        valid_days=10,
        test_days=10,
        min_train_days=30,
    )

    assert len(windows) == 3
    assert windows["train_start"].is_monotonic_increasing
    for _, row in windows.iterrows():
        assert row["train_start"] < row["train_end"] < row["valid_start"] < row["valid_end"]
        assert row["valid_end"] < row["test_start"] < row["test_end"]
        assert row["train_days"] == 40
        assert row["valid_days"] == 10
        assert row["test_days"] == 10


def test_generate_walkforward_windows_returns_empty_when_history_is_short() -> None:
    trade_dates = pd.date_range("2025-01-01", periods=35, freq="B")

    windows = generate_walkforward_windows(
        trade_dates,
        windows=4,
        train_days=30,
        valid_days=10,
        test_days=10,
        min_train_days=25,
    )

    assert windows.empty


def test_summarize_walkforward_topn_metrics_adds_threshold_flags() -> None:
    metrics = pd.DataFrame(
        [
            {
                "window_id": 1,
                "model_version": "v42_gate_v4_rank",
                "top_n": 20,
                "test_days": 20,
                "allowed_days": 10,
                "coverage": 0.50,
                "rows": 200,
                "win_rate": 0.80,
                "avg_return_20d": 0.08,
                "median_return_20d": 0.04,
                "take_profit_rate_20d": 0.40,
                "stop_loss_rate_20d": 0.03,
                "bad_risk_rate": 0.10,
                "avg_take_profit_20d": 0.18,
                "avg_stop_loss_20d": -0.07,
                "avg_positive_return_20d": 0.12,
                "avg_negative_return_20d": -0.04,
            },
            {
                "window_id": 2,
                "model_version": "v42_gate_v4_rank",
                "top_n": 20,
                "test_days": 20,
                "allowed_days": 8,
                "coverage": 0.40,
                "rows": 160,
                "win_rate": 0.70,
                "avg_return_20d": 0.06,
                "median_return_20d": 0.03,
                "take_profit_rate_20d": 0.35,
                "stop_loss_rate_20d": 0.04,
                "bad_risk_rate": 0.12,
                "avg_take_profit_20d": 0.17,
                "avg_stop_loss_20d": -0.08,
                "avg_positive_return_20d": 0.11,
                "avg_negative_return_20d": -0.05,
            },
        ]
    )

    summary = summarize_walkforward_topn_metrics(metrics)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["windows"] == 2
    assert row["win_rate_mean"] == 0.75
    assert row["win_rate_min"] == 0.70
    assert row["pass_all_top20_thresholds"] == 1
    assert row["window_pass_rate"] == 1.0


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
