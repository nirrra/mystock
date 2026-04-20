from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .models import PickTrendWatchlistConfig, WatchlistTrendFilterConfig


WATCHLIST_FILENAME_RE = re.compile(r"watchlist_(\d{4}-\d{2}-\d{2})\.json$")
PATTERN_PRIORITY = {
    "1": 3.0,
    "3": 2.8,
    "2": 2.5,
    "4": 1.8,
}
LABEL_BONUS = {
    "strong_buy": 1.0,
    "buy": 0.6,
    "neutral": 0.0,
    "sell": -0.6,
    "strong_sell": -1.0,
}
INDEX_NAME_MARKERS = {
    "指数",
    "上证综指",
    "深证成指",
    "创业板指",
    "沪深300",
    "中证500",
    "中证1000",
    "科创50",
}
TREND_CANDIDATE_FIELDS = (
    "signal_type",
    "trend_score",
    "entry_score",
    "trend_base_score",
    "price_action_score",
    "macd_score",
    "buy_score",
    "positive_indicator_count",
    "macd_cross_state",
    "macd_divergence_state",
    "volume_price_divergence_state",
    "macd_top_divergence_flag",
    "macd_bottom_divergence_flag",
    "bullish_volume_price_divergence_flag",
    "bearish_volume_price_divergence_flag",
    "trigger_reason",
    "buy_reason",
)
TREND_UNIVERSE_CANDIDATE_FIELDS = (
    "in_trend_universe",
    "trend_universe_score",
    "trend_direction_score",
    "trend_strength_score",
    "trend_quality_score",
    "trend_liquidity_score",
)


def watchlists_dir(project_root: Path) -> Path:
    return project_root / "reports" / "watchlists"


def watchlist_path(project_root: Path, trade_date: date) -> Path:
    return watchlists_dir(project_root) / f"watchlist_{trade_date.isoformat()}.json"


def watchlist_pattern_path(project_root: Path, trade_date: date) -> Path:
    return watchlists_dir(project_root) / f"watchlist_pattern_{trade_date.isoformat()}.json"


def watchlist_trend_path(project_root: Path, trade_date: date) -> Path:
    return watchlists_dir(project_root) / f"watchlist_trend_{trade_date.isoformat()}.json"


def build_watchlist_candidates_from_patterns(
    pattern_frame: pd.DataFrame,
    *,
    source_file: str,
    limit: int = 30,
) -> dict[str, object]:
    frame = pattern_frame.copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame = frame[~frame["name"].map(_is_true_index_name)].copy()

    daily_columns = _daily_rating_columns(frame)
    required = {"symbol", "name", "pattern_id", "tradingview_avg_all_rating_5d", "tradingview_all_rating_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Latest patterns frame is missing required TradingView columns: {missing}")

    frame["stable_score"] = frame.apply(lambda row: _stable_score(row, daily_columns), axis=1)
    frame["base_tier"] = frame.apply(_base_tier, axis=1)
    frame = frame.dropna(subset=["base_tier"]).copy()
    frame = frame[~frame.apply(_is_row_risk_excluded, axis=1)].copy()
    frame = frame.sort_values(["base_tier", "stable_score", "tradingview_avg_all_rating_5d"], ascending=[True, False, False])

    candidates: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        candidate = {
            "tier": row["base_tier"],
            "symbol": row["symbol"],
            "name": row["name"],
            "pattern_id": str(row["pattern_id"]),
            "macd_top_divergence_15d": bool(row.get("macd_top_divergence_15d", False)),
            "macd_bottom_divergence_15d": bool(row.get("macd_bottom_divergence_15d", False)),
            "tradingview_label": str(row["tradingview_all_rating_label"]).strip().lower(),
            "tradingview_avg_5d": round(float(row["tradingview_avg_all_rating_5d"]), 4),
            "five_day_scores": [round(float(row[column]), 4) for column in daily_columns if pd.notna(row.get(column))],
            "stable_score": round(float(row["stable_score"]), 4),
            "reason": row.get("reason", ""),
        }
        for field in TREND_UNIVERSE_CANDIDATE_FIELDS + TREND_CANDIDATE_FIELDS:
            if field not in row or pd.isna(row.get(field)):
                continue
            value = row.get(field)
            if isinstance(value, float):
                candidate[field] = round(float(value), 4)
            else:
                candidate[field] = value
        candidates.append(candidate)

    return {
        "source_file": source_file,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_watchlist_candidates_from_trend(
    trend_frame: pd.DataFrame,
    *,
    source_file: str,
    thresholds: PickTrendWatchlistConfig,
    limit: int = 30,
) -> dict[str, object]:
    frame = trend_frame.copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    required = {"symbol", "name", "buy_score", "price_action_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Trend frame is missing required columns: {missing}")

    frame["symbol"] = frame["symbol"].map(_normalize_symbol)
    frame = frame[
        pd.to_numeric(frame["buy_score"], errors="coerce").ge(thresholds.buy_score_min)
        & pd.to_numeric(frame["price_action_score"], errors="coerce").ge(thresholds.price_action_score_min)
    ].copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    frame = frame[~frame.apply(_is_row_risk_excluded, axis=1)].copy()
    if frame.empty:
        return {
            "source_file": source_file,
            "candidate_count": 0,
            "candidates": [],
        }

    sort_columns = [
        column
        for column in ("buy_score", "price_action_score", "trend_score", "trend_base_score", "entry_score")
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    candidates: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        candidate = {
            "source": "trend",
            "symbol": row["symbol"],
            "name": row.get("name", ""),
            "signal_type": row.get("signal_type", ""),
            "buy_score": round(float(row["buy_score"]), 4),
            "price_action_score": round(float(row["price_action_score"]), 4),
        }
        for field in TREND_UNIVERSE_CANDIDATE_FIELDS + TREND_CANDIDATE_FIELDS:
            if field not in row or pd.isna(row.get(field)):
                continue
            value = row.get(field)
            if isinstance(value, float):
                candidate[field] = round(float(value), 4)
            else:
                candidate[field] = value
        candidates.append(candidate)

    return {
        "source_file": source_file,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_watchlist_candidates(project_root: Path, limit: int = 30) -> dict[str, object]:
    patterns_file = _latest_patterns_file(project_root)
    frame = pd.read_csv(patterns_file)
    return build_watchlist_candidates_from_patterns(frame, source_file=str(patterns_file), limit=limit)


def apply_trend_filter_to_watchlist_payload(
    payload: dict[str, object],
    *,
    trend_frame: pd.DataFrame,
    trend_filter: WatchlistTrendFilterConfig,
) -> dict[str, object]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return {
            "source_file": payload.get("source_file"),
            "candidate_count": 0,
            "candidates": [],
        }

    frame = trend_frame.copy()
    required = {"symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Trend report is missing required columns: {missing}")
    if frame.empty:
        return {
            "source_file": payload.get("source_file"),
            "candidate_count": 0,
            "candidates": [],
        }
    frame["symbol"] = frame["symbol"].map(_normalize_symbol)

    if "in_trend_universe" in frame.columns:
        frame = frame[frame["in_trend_universe"].fillna(False).astype(bool)].copy()
    sort_columns = [
        column
        for column in ("trend_universe_score", "trend_score", "buy_score", "price_action_score")
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    frame = frame.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    trend_by_symbol = {str(row["symbol"]).zfill(6): row for row in frame.to_dict("records")}

    filtered: list[dict[str, object]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = _normalize_symbol(item.get("symbol", ""))
        trend_row = trend_by_symbol.get(symbol)
        if trend_row is None:
            continue
        if _is_row_risk_excluded(trend_row):
            continue

        enriched = deepcopy(item)
        enriched["symbol"] = symbol
        for field in TREND_UNIVERSE_CANDIDATE_FIELDS + TREND_CANDIDATE_FIELDS:
            if field not in trend_row:
                continue
            value = trend_row.get(field)
            if pd.isna(value):
                continue
            if isinstance(value, float):
                enriched[field] = round(float(value), 4)
            else:
                enriched[field] = value
        filtered.append(enriched)

    return {
        "source_file": payload.get("source_file"),
        "candidate_count": len(filtered),
        "candidates": filtered,
    }


def write_watchlist(
    *,
    project_root: Path,
    trade_date: date,
    picker_payload: dict[str, object],
    kind: str | None = None,
) -> Path:
    target = _resolve_watchlist_target(project_root, trade_date, kind=kind)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_watchlist_payload(trade_date=trade_date, picker_payload=picker_payload)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def build_watchlist_payload(*, trade_date: date, picker_payload: dict[str, object]) -> dict[str, object]:
    payload = deepcopy(picker_payload)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    return {
        "trade_date": trade_date.isoformat(),
        "source_file": payload.get("source_file"),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def load_watchlist(*, project_root: Path, trade_date: date, kind: str | None = None) -> dict[str, object]:
    target = _resolve_watchlist_target(project_root, trade_date, kind=kind)
    if not target.exists():
        raise FileNotFoundError(f"Watchlist not found for {trade_date.isoformat()}: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def find_latest_watchlist_before(*, project_root: Path, trade_date: date) -> tuple[date, Path]:
    candidates: list[tuple[date, Path]] = []
    for path in watchlists_dir(project_root).glob("watchlist_*.json"):
        parsed = _parse_watchlist_date(path)
        if parsed is None or parsed >= trade_date:
            continue
        candidates.append((parsed, path))

    if not candidates:
        raise FileNotFoundError(f"No watchlist found before {trade_date.isoformat()} in {watchlists_dir(project_root)}")

    return max(candidates, key=lambda item: item[0])


def extract_watchlist_symbols(payload: dict[str, object]) -> list[str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []

    symbols: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().zfill(6)
        if symbol == "000000" or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _parse_watchlist_date(path: Path) -> date | None:
    match = WATCHLIST_FILENAME_RE.fullmatch(path.name)
    if not match:
        return None
    return datetime.fromisoformat(match.group(1)).date()


def _latest_patterns_file(project_root: Path) -> Path:
    files = sorted(
        (project_root / "reports" / "patterns").glob("patterns_all_*.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No reports/patterns/patterns_all_*.csv file found. Run mystock pattern first.")
    return files[0]


def _normalize_symbol(value: object) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    return text.zfill(6)


def _daily_rating_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("tradingview_all_rating_20"))


def _stable_score(row: pd.Series, daily_columns: list[str]) -> float:
    avg_score = float(row.get("tradingview_avg_all_rating_5d", 0.0))
    latest_score = float(row.get(daily_columns[-1], avg_score)) if daily_columns else avg_score
    pattern_id = str(row.get("pattern_id", ""))
    label = str(row.get("tradingview_all_rating_label", "")).strip().lower()
    return round(
        avg_score * 4.0
        + latest_score * 1.5
        + PATTERN_PRIORITY.get(pattern_id, 0.0)
        + LABEL_BONUS.get(label, -1.0),
        4,
    )


def _base_tier(row: pd.Series) -> str | None:
    avg_score = float(row.get("tradingview_avg_all_rating_5d", 0.0))
    pattern_id = str(row.get("pattern_id", ""))
    label = str(row.get("tradingview_all_rating_label", "")).strip().lower()

    if label not in {"buy", "strong_buy"}:
        return None
    if avg_score < 0.22:
        return None

    if pattern_id in {"1", "3"} and avg_score >= 0.36:
        return "第一梯队"
    if pattern_id == "2" and avg_score >= 0.35:
        return "第一梯队"
    if pattern_id == "4" and avg_score >= 0.42 and label == "strong_buy":
        return "第一梯队"

    if pattern_id in {"1", "2", "3"} and avg_score >= 0.28:
        return "第二梯队"
    if pattern_id == "4" and avg_score >= 0.32:
        return "第二梯队"

    return "第三梯队"


def _is_true_index_name(name: object) -> bool:
    normalized = str(name).strip().replace(" ", "")
    return any(marker in normalized for marker in INDEX_NAME_MARKERS)


def _resolve_watchlist_target(project_root: Path, trade_date: date, kind: str | None) -> Path:
    if kind == "pattern":
        return watchlist_pattern_path(project_root, trade_date)
    if kind == "trend":
        return watchlist_trend_path(project_root, trade_date)
    return watchlist_path(project_root, trade_date)


def _is_row_risk_excluded(row: pd.Series | dict[str, object]) -> bool:
    macd_cross_state = str(_row_value(row, "macd_cross_state", "")).strip().lower()
    macd_divergence_state = str(_row_value(row, "macd_divergence_state", "")).strip().lower()
    volume_price_divergence_state = str(_row_value(row, "volume_price_divergence_state", "")).strip().lower()

    if macd_cross_state == "dead_cross":
        return True
    if macd_divergence_state == "top_divergence":
        return True
    if volume_price_divergence_state == "bearish":
        return True
    if _is_truthy_flag(_row_value(row, "macd_top_divergence_15d", False)):
        return True
    if _is_truthy_flag(_row_value(row, "bearish_volume_price_divergence_flag", False)):
        return True
    return False


def _row_value(row: pd.Series | dict[str, object], key: str, default: object) -> object:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _is_truthy_flag(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)
