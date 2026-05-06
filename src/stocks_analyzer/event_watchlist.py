from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


def watchlist_event_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "watchlists" / f"watchlist_event_{trade_date.isoformat()}.json"


def build_watchlist_event_payload(
    predictions: pd.DataFrame,
    *,
    trade_date: date,
    limit: int | None = 30,
) -> dict[str, object]:
    if predictions.empty:
        return {
            "trade_date": trade_date.isoformat(),
            "model_version": "event_risk_ranker_v1",
            "trade_permission": "no_trade",
            "candidate_count": 0,
            "items": [],
            "warnings": ["no_predictions"],
        }

    frame = predictions.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["final_score_sort"] = pd.to_numeric(frame.get("final_score", 0.0), errors="coerce").fillna(-1e9)
    frame = frame.sort_values(["symbol", "final_score_sort"], ascending=[True, False]).drop_duplicates("symbol", keep="first")
    action_rank = {"candidate": 0, "observe": 1, "avoid": 2}
    frame["action_rank"] = frame.get("suggested_action", "avoid").map(action_rank).fillna(3)
    frame = frame.sort_values(["action_rank", "final_score_sort"], ascending=[True, False])
    if limit is not None:
        frame = frame.head(limit)

    trade_permission = str(predictions.get("trade_permission", pd.Series(["no_trade"])).iloc[0])
    items = [_watchlist_item(row) for row in frame.to_dict("records")]
    return {
        "trade_date": trade_date.isoformat(),
        "model_version": str(predictions.get("model_version", pd.Series(["event_risk_ranker_v1"])).iloc[0]),
        "trade_permission": trade_permission,
        "candidate_count": int(sum(1 for item in items if item.get("suggested_action") == "candidate")),
        "items": items,
        "warnings": [],
    }


def write_watchlist_event(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _watchlist_item(row: dict[str, object]) -> dict[str, object]:
    return {
        "symbol": str(row.get("symbol", "")).zfill(6),
        "name": row.get("name", ""),
        "pattern_id": str(row.get("pattern_id", "")),
        "pattern_ids": row.get("pattern_ids", row.get("pattern_id", "")),
        "final_score": _rounded(row.get("final_score")),
        "expected_R_score": _rounded(row.get("expected_R_score")),
        "p_stop_first": _rounded(row.get("p_stop_first")),
        "suggested_action": row.get("suggested_action", "avoid"),
        "entry_price_ref": _rounded(row.get("entry_price_ref"), digits=4),
        "stop_loss_price_ref": _rounded(row.get("stop_loss_price_ref"), digits=4),
        "take_profit_price_ref": _rounded(row.get("take_profit_price_ref"), digits=4),
        "max_holding_days": int(float(row.get("max_holding_days", 0) or 0)),
        "risk_reason": row.get("risk_reason", ""),
    }


def _rounded(value: object, *, digits: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)
