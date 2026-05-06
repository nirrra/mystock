from __future__ import annotations

import pandas as pd

from stocks_analyzer.event_labels import EventLabelConfig, add_rank_labels, build_event_labels
from stocks_analyzer.event_risk_ranker import evaluate_event_ranker_topn, score_event_risk_ranker_frame
from stocks_analyzer.event_watchlist import build_watchlist_event_payload


def test_build_event_labels_uses_stop_when_both_barriers_touched() -> None:
    bars = _bars(
        130,
        close=10.0,
        overrides={
            121: {"open": 10.0, "high": 13.0, "low": 8.0, "close": 10.0},
        },
    )
    signals = _signals(signal_index=120)

    labels, skipped = build_event_labels(
        signals,
        {"600000": bars},
        config=EventLabelConfig(stop_atr_mult=1.0, take_atr_mult=2.0, max_holding_days=2, min_history_days=60),
    )

    assert skipped.empty
    assert labels.loc[0, "barrier_outcome"] == "stop_loss_first"
    assert labels.loc[0, "realized_R"] == -1.0
    assert labels.loc[0, "max_drawdown_R"] >= 1.0
    assert labels.loc[0, "holding_days_penalty"] == 0.5


def test_build_event_labels_skips_unfillable_limit_up_entry() -> None:
    bars = _bars(
        130,
        close=10.0,
        overrides={
            120: {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
            121: {"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0},
        },
    )
    signals = _signals(signal_index=120)

    labels, skipped = build_event_labels(
        signals,
        {"600000": bars},
        config=EventLabelConfig(stop_atr_mult=1.0, take_atr_mult=2.0, max_holding_days=2, min_history_days=60),
    )

    assert labels.empty
    assert skipped.loc[0, "skip_reason"] == "entry_unfillable_limit_up"


def test_add_rank_labels_requires_minimum_same_day_events() -> None:
    labels = pd.DataFrame(
        [
            {"signal_date": pd.Timestamp("2026-01-01"), "rank_value": 0.1},
            {"signal_date": pd.Timestamp("2026-01-01"), "rank_value": 0.2},
            {"signal_date": pd.Timestamp("2026-01-01"), "rank_value": 0.3},
            {"signal_date": pd.Timestamp("2026-01-01"), "rank_value": 0.4},
            {"signal_date": pd.Timestamp("2026-01-01"), "rank_value": 0.5},
            {"signal_date": pd.Timestamp("2026-01-02"), "rank_value": 1.0},
        ]
    )

    ranked = add_rank_labels(labels)

    assert ranked.loc[:4, "rank_train_eligible"].tolist() == [True] * 5
    assert ranked.loc[4, "rank_grade"] == 4
    assert ranked.loc[5, "rank_train_eligible"] == False


def test_evaluate_event_ranker_topn_reports_symbol_dedup_scope() -> None:
    frame = pd.DataFrame(
        [
            _prediction("600000", 0.9, 1.0),
            _prediction("600000", 0.8, -1.0),
            _prediction("600001", 0.7, 0.5),
        ]
    )

    metrics = evaluate_event_ranker_topn(frame, top_n_list=(2,), model_name="m", dataset_split="test")

    scopes = set(metrics["scope"].tolist())
    assert scopes == {"event", "symbol_dedup"}
    symbol_metric = metrics[metrics["scope"].eq("symbol_dedup")].iloc[0]
    assert symbol_metric["selected_symbols"] == 2
    assert symbol_metric["avg_realized_R"] == 0.75


def test_build_watchlist_event_payload_marks_candidate_count() -> None:
    predictions = pd.DataFrame(
        [
            {
                "symbol": "600000",
                "name": "甲",
                "pattern_id": "1",
                "pattern_ids": "1",
                "final_score": 0.9,
                "expected_R_score": 0.4,
                "p_stop_first": 0.2,
                "suggested_action": "candidate",
                "trade_permission": "allow",
                "entry_price_ref": 10,
                "stop_loss_price_ref": 9,
                "take_profit_price_ref": 12,
                "max_holding_days": 20,
                "risk_reason": "passed",
                "model_version": "event_risk_ranker_v1",
            }
        ]
    )

    payload = build_watchlist_event_payload(predictions, trade_date=pd.Timestamp("2026-05-06").date())

    assert payload["candidate_count"] == 1
    assert payload["items"][0]["symbol"] == "600000"


def test_score_event_ranker_orders_candidates_before_avoids() -> None:
    frame = pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp("2026-05-06"),
                "symbol": "600000",
                "pattern_id": "1",
                "risk_feature": 0.9,
                "rank_feature": 1.0,
                "rule_score": 2.0,
                "entry_price": 10.0,
                "stop_loss_price": 9.0,
                "take_profit_price": 12.0,
            },
            {
                "signal_date": pd.Timestamp("2026-05-06"),
                "symbol": "600001",
                "pattern_id": "1",
                "risk_feature": 0.1,
                "rank_feature": 0.2,
                "rule_score": 1.0,
                "entry_price": 10.0,
                "stop_loss_price": 9.0,
                "take_profit_price": 12.0,
            },
        ]
    )
    artifact = {
        "feature_columns": ["risk_feature", "rank_feature"],
        "risk_model": _FeatureRiskModel(),
        "rank_model": _FeatureRankModel(),
        "risk_threshold": 0.5,
        "opportunity_threshold": 0.0,
    }

    scored = score_event_risk_ranker_frame(frame, artifact=artifact)

    assert scored.loc[0, "symbol"] == "600001"
    assert scored.loc[0, "suggested_action"] == "candidate"
    assert scored.loc[1, "suggested_action"] == "avoid"


def _signals(*, signal_index: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-01-01") + pd.offsets.BDay(signal_index),
                "symbol": "600000",
                "name": "测试",
                "pattern_id": "1",
                "strategy_name": "volume_top_pre_breakout",
                "reason": "test",
            }
        ]
    )


def _bars(count: int, *, close: float, overrides: dict[int, dict[str, float]] | None = None) -> pd.DataFrame:
    rows = []
    for index, trade_date in enumerate(pd.bdate_range("2026-01-01", periods=count)):
        value = close
        row = {
            "trade_date": trade_date,
            "open": value,
            "high": value + 0.5,
            "low": value - 0.5,
            "close": value,
            "volume": 100000 + index,
            "amount": (100000 + index) * value,
        }
        if overrides and index in overrides:
            row.update(overrides[index])
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction(symbol: str, score: float, realized_r: float) -> dict[str, object]:
    return {
        "signal_date": pd.Timestamp("2026-01-01"),
        "symbol": symbol,
        "final_score": score,
        "realized_R": realized_r,
        "suggested_action": "candidate",
        "trade_permission": "allow",
        "barrier_outcome": "take_profit_first" if realized_r > 0 else "stop_loss_first",
        "max_drawdown_R": 0.1,
        "holding_days": 2,
    }


class _FeatureRiskModel:
    def predict_proba(self, X: pd.DataFrame) -> object:
        p = pd.to_numeric(X["risk_feature"], errors="coerce").fillna(1.0)
        return pd.concat([1.0 - p, p], axis=1).to_numpy()


class _FeatureRankModel:
    def predict(self, X: pd.DataFrame) -> object:
        return pd.to_numeric(X["rank_feature"], errors="coerce").fillna(0.0).to_numpy()
