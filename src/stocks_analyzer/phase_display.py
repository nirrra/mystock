from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from .phase4_rolling import PHASE4_ROLLING_COLUMNS, merge_phase4_rolling_frame


PHASE_DISPLAY_COLUMNS = [
    "phase1_score_100",
    "phase2_score_100",
    "phase2_is_cusum_event",
    "phase4_score_100",
    *PHASE4_ROLLING_COLUMNS,
    "phase8_score_100",
    "phase8_rank",
    "phase5_score_100",
    "phase7_score_100",
    "phase7_trade_permission",
]

PHASE_TABLE_DROP_COLUMNS = [
    "phase1_risk_score",
    "phase1_log_return_1d",
    "daily_return_1d",
    "limit_up_excluded_by_daily_return",
    "phase1_risk_rank",
    "phase1_risk_percentile",
    "phase1_excluded_top20",
    "phase1_excluded_by_top20_risk",
    "phase2_barrier_risk_score",
    "phase2_risk_rank",
    "phase2_risk_percentile",
    "phase2_excluded_top20",
    "phase2_excluded_by_top20_risk",
    "phase2_mlfin_daily_vol",
    "phase2_mlfin_cusum_threshold",
    "phase4_return_score",
    "phase8_raw_score",
    "phase4_top_score_filter_pass",
    "phase4_rank",
    "phase4_score_percentile",
    "phase5_NEGOUTLIER",
    "phase5_CRASH",
    "phase5_CRASH_count",
    "phase5_NCSKEW",
    "phase5_DUVOL",
    "phase5_RET",
    "phase5_SIGMA",
    "phase5_MINRET",
    "phase5_year",
    "phase5_weeks",
    "phase7_buy_day_risk_score",
    "phase7_selected_threshold",
    "phase7_suggested_action",
    "phase7_reason",
]

PHASE5_SCORE_COMPONENTS = (
    ("phase5_NEGOUTLIER", False),
    ("phase5_CRASH", False),
    ("phase5_CRASH_count", False),
    ("phase5_NCSKEW", False),
    ("phase5_DUVOL", False),
    ("phase5_SIGMA", False),
    ("phase5_MINRET", True),
)


def append_daily_phase_display_columns(frame: pd.DataFrame, *, project_root: Path, trade_date: date) -> pd.DataFrame:
    cleaned = frame.drop(columns=[column for column in PHASE_TABLE_DROP_COLUMNS if column in frame.columns], errors="ignore")
    if frame.empty or "symbol" not in frame.columns:
        return _ensure_phase_columns(cleaned.copy())
    display = build_daily_phase_display_frame(project_root=project_root, trade_date=trade_date)
    if display.empty:
        return _ensure_phase_columns(cleaned.copy())
    result = cleaned.copy()
    result["_phase_symbol"] = result["symbol"].map(normalize_symbol)
    phase = display.rename(columns={"symbol": "_phase_symbol"})
    result = result.drop(columns=[column for column in PHASE_DISPLAY_COLUMNS if column in result.columns], errors="ignore")
    result = result.merge(phase, on="_phase_symbol", how="left")
    return result.drop(columns=["_phase_symbol"], errors="ignore")


def build_daily_phase_display_frame(*, project_root: Path, trade_date: date) -> pd.DataFrame:
    phase1 = _prepare_phase1(_read_csv_columns(_phase1_path(project_root, trade_date), {"symbol", "risk_score"}))
    phase2 = _prepare_phase2(
        _read_csv_columns(
            _phase2_path(project_root, trade_date),
            {"symbol", "barrier_risk_score", "is_cusum_event"},
        )
    )
    phase4 = _prepare_phase4(_read_csv_columns(_phase4_path(project_root, trade_date), {"symbol", "return_score"}))
    phase4 = merge_phase4_rolling_frame(phase4, project_root=project_root, trade_date=trade_date)
    phase8 = _prepare_phase8(
        _read_csv_columns(
            _phase8_path(project_root, trade_date),
            {"symbol", "phase8_score_100", "phase8_raw_score", "phase8_rank", "today_limit_up_excluded"},
        )
    )
    phase5 = _prepare_phase5(
        _read_csv_columns(
            _phase5_path(project_root),
            {"symbol", "year", "NEGOUTLIER", "CRASH", "CRASH_count", "NCSKEW", "DUVOL", "RET", "SIGMA", "MINRET"},
        ),
        trade_date=trade_date,
    )
    frames = [item for item in (phase1, phase2, phase4, phase8, phase5) if not item.empty]
    if not frames:
        result = pd.DataFrame(columns=["symbol"])
    else:
        result = frames[0]
        for item in frames[1:]:
            result = result.merge(item, on="symbol", how="outer")
    phase7 = _prepare_phase7(_read_csv_columns(
        _phase7_path(project_root, trade_date),
        {"trade_permission", "buy_day_risk_score", "selected_threshold", "suggested_action", "reason"},
    ))
    for key, value in phase7.items():
        result[key] = value
    result = _ensure_phase_columns(result)
    return result.loc[:, ["symbol", *PHASE_DISPLAY_COLUMNS]].copy()


def score_series_100(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Float64")
    valid = numeric.dropna()
    if valid.empty:
        return result
    if len(valid) == 1:
        result.loc[valid.index] = 100.0
        return result
    rank = valid.rank(method="max", ascending=higher_is_better)
    result.loc[valid.index] = ((rank - 1.0) / (len(valid) - 1.0) * 100.0).round(2)
    return result


def add_phase5_score_100(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    component_scores: list[pd.Series] = []
    for column, higher_is_better in PHASE5_SCORE_COMPONENTS:
        if column not in result.columns:
            continue
        component_scores.append(score_series_100(result[column], higher_is_better=higher_is_better))
    if component_scores:
        result["phase5_score_100"] = pd.concat(component_scores, axis=1).mean(axis=1).round(2)
    else:
        result["phase5_score_100"] = pd.NA
    return result


def phase7_score_100(trade_permission: object) -> float | None:
    permission = str(trade_permission or "").strip().lower()
    if permission == "allow":
        return 100.0
    if permission == "no_trade":
        return 0.0
    return None


def normalize_symbol(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    if text.startswith("="):
        text = text.lstrip("=").strip().strip('"')
    text = text.replace(".0", "") if text.endswith(".0") else text
    lower = text.lower()
    if lower.startswith(("sh", "sz", "bj")):
        text = text[2:]
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def _prepare_phase1(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "risk_score" not in frame.columns:
        return pd.DataFrame(columns=["symbol"])
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["phase1_risk_score"] = pd.to_numeric(result["risk_score"], errors="coerce")
    result = result.dropna(subset=["symbol", "phase1_risk_score"]).sort_values(["phase1_risk_score", "symbol"], ascending=[False, True])
    result = result.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    result["phase1_risk_rank"] = result.index + 1
    result["phase1_excluded_top20"] = False
    if not result.empty:
        result.loc[: max(0, math.ceil(len(result) * 0.2) - 1), "phase1_excluded_top20"] = True
    result["phase1_score_100"] = score_series_100(result["phase1_risk_score"], higher_is_better=False)
    return result.loc[:, ["symbol", "phase1_score_100", "phase1_risk_score", "phase1_risk_rank", "phase1_excluded_top20"]]


def _prepare_phase2(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "barrier_risk_score" not in frame.columns:
        return pd.DataFrame(columns=["symbol"])
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["phase2_barrier_risk_score"] = pd.to_numeric(result["barrier_risk_score"], errors="coerce")
    result = result.dropna(subset=["symbol", "phase2_barrier_risk_score"]).sort_values(
        ["phase2_barrier_risk_score", "symbol"],
        ascending=[False, True],
    )
    result = result.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    result["phase2_risk_rank"] = result.index + 1
    result["phase2_excluded_top20"] = False
    if not result.empty:
        result.loc[: max(0, math.ceil(len(result) * 0.2) - 1), "phase2_excluded_top20"] = True
    result["phase2_score_100"] = score_series_100(result["phase2_barrier_risk_score"], higher_is_better=False)
    if "is_cusum_event" in result.columns:
        result["phase2_is_cusum_event"] = result["is_cusum_event"]
    keep = ["symbol", "phase2_score_100", "phase2_barrier_risk_score", "phase2_risk_rank", "phase2_excluded_top20", "phase2_is_cusum_event"]
    return result.loc[:, [column for column in keep if column in result.columns]]


def _prepare_phase4(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "return_score" not in frame.columns:
        return pd.DataFrame(columns=["symbol"])
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["phase4_return_score"] = pd.to_numeric(result["return_score"], errors="coerce")
    result = result.dropna(subset=["symbol", "phase4_return_score"]).sort_values(["phase4_return_score", "symbol"], ascending=[False, True])
    result = result.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    result["phase4_rank"] = result.index + 1
    result["phase4_score_100"] = score_series_100(result["phase4_return_score"], higher_is_better=True)
    return result.loc[:, ["symbol", "phase4_score_100", "phase4_return_score", "phase4_rank"]]


def _prepare_phase8(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame(columns=["symbol"])
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    if "phase8_score_100" in result.columns:
        result["phase8_score_100"] = pd.to_numeric(result["phase8_score_100"], errors="coerce")
    elif "phase8_raw_score" in result.columns:
        result["phase8_raw_score"] = pd.to_numeric(result["phase8_raw_score"], errors="coerce")
        result["phase8_score_100"] = score_series_100(result["phase8_raw_score"], higher_is_better=True)
    else:
        return pd.DataFrame(columns=["symbol"])
    if "phase8_rank" in result.columns:
        result["phase8_rank"] = pd.to_numeric(result["phase8_rank"], errors="coerce")
    else:
        result = result.sort_values(["phase8_score_100", "symbol"], ascending=[False, True], na_position="last").reset_index(drop=True)
        result["phase8_rank"] = result.index + 1
    keep = ["symbol", "phase8_score_100", "phase8_rank"]
    if "phase8_raw_score" in result.columns:
        result["phase8_raw_score"] = pd.to_numeric(result["phase8_raw_score"], errors="coerce")
        keep.append("phase8_raw_score")
    if "today_limit_up_excluded" in result.columns:
        keep.append("today_limit_up_excluded")
    return result.loc[:, [column for column in keep if column in result.columns]].drop_duplicates("symbol", keep="first")


def _prepare_phase5(frame: pd.DataFrame, *, trade_date: date) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "year" not in frame.columns:
        return pd.DataFrame(columns=["symbol"])
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result = result.dropna(subset=["symbol", "year"])
    if result.empty:
        return pd.DataFrame(columns=["symbol"])
    eligible = result[result["year"].astype(int).le(trade_date.year)].copy()
    if eligible.empty:
        eligible = result.copy()
    eligible = eligible.sort_values(["symbol", "year"]).drop_duplicates("symbol", keep="last")
    rename_map = {
        "year": "phase5_year",
        "NEGOUTLIER": "phase5_NEGOUTLIER",
        "CRASH": "phase5_CRASH",
        "CRASH_count": "phase5_CRASH_count",
        "NCSKEW": "phase5_NCSKEW",
        "DUVOL": "phase5_DUVOL",
        "RET": "phase5_RET",
        "SIGMA": "phase5_SIGMA",
        "MINRET": "phase5_MINRET",
    }
    eligible = eligible.rename(columns={source: target for source, target in rename_map.items() if source in eligible.columns})
    eligible = add_phase5_score_100(eligible)
    keep = ["symbol", "phase5_score_100", "phase5_NEGOUTLIER", "phase5_CRASH", "phase5_CRASH_count", "phase5_NCSKEW", "phase5_DUVOL", "phase5_RET", "phase5_SIGMA", "phase5_MINRET", "phase5_year"]
    return eligible.loc[:, [column for column in keep if column in eligible.columns]]


def _prepare_phase7(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    permission = row.get("trade_permission", pd.NA)
    score = phase7_score_100(permission)
    result: dict[str, object] = {
        "phase7_trade_permission": permission,
        "phase7_score_100": score,
    }
    mapping = {
        "buy_day_risk_score": "phase7_buy_day_risk_score",
        "selected_threshold": "phase7_selected_threshold",
        "suggested_action": "phase7_suggested_action",
        "reason": "phase7_reason",
    }
    for source, target in mapping.items():
        if source in frame.columns:
            result[target] = row.get(source, pd.NA)
    return result


def _ensure_phase_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in PHASE_DISPLAY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _read_csv_columns(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=lambda column: column in columns)
    except ValueError:
        return pd.read_csv(path)


def _phase1_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "full_market_model" / f"tail_risk_predictions_{trade_date.isoformat()}.csv"


def _phase2_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "full_market_model" / f"barrier_risk_predictions_{trade_date.isoformat()}.csv"


def _phase4_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "full_market_model" / f"alpha158_qlib_return_predictions_{trade_date.isoformat()}.csv"


def _phase8_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "full_market_model" / f"limit_up_3d_opportunity_predictions_{trade_date.isoformat()}.csv"


def _phase5_path(project_root: Path) -> Path:
    return project_root / "reports" / "full_market_model" / "mcd_crash_annual_measures.csv"


def _phase7_path(project_root: Path, trade_date: date) -> Path:
    return project_root / "reports" / "full_market_model" / f"trade_day_gate_prediction_{trade_date.isoformat()}.csv"
