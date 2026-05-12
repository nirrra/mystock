from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd


PHASE4_ROLLING_COLUMNS = [
    "phase4_5d_mean",
    "phase4_5d_std",
]
PHASE4_ROLLING_EXTRA_COLUMNS = [
    "phase4_5d_count",
]
PHASE4_ROLLING_RANK_COLUMNS = [
    "phase4_5d_mean_rank",
    "phase4_5d_mean_top5",
]

_PHASE4_REPORT_RE = re.compile(r"^alpha158_qlib_return_predictions_(\d{4}-\d{2}-\d{2})\.csv$")


def build_phase4_rolling_frame(
    *,
    project_root: Path,
    trade_date: date,
    window: int = 5,
) -> pd.DataFrame:
    report_dir = project_root / "reports" / "full_market_model"
    files = _recent_phase4_report_files(report_dir=report_dir, trade_date=trade_date, window=window)
    return build_phase4_rolling_frame_from_files(files)


def build_phase4_rolling_frame_from_files(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=["symbol", *PHASE4_ROLLING_COLUMNS, *PHASE4_ROLLING_EXTRA_COLUMNS, *PHASE4_ROLLING_RANK_COLUMNS])
    parts: list[pd.DataFrame] = []
    for path in files:
        frame = _read_phase4_score_file(path)
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=["symbol", *PHASE4_ROLLING_COLUMNS, *PHASE4_ROLLING_EXTRA_COLUMNS, *PHASE4_ROLLING_RANK_COLUMNS])
    combined = pd.concat(parts, ignore_index=True)
    grouped = combined.groupby("symbol", sort=False)["phase4_daily_score_100"]
    result = grouped.agg(
        phase4_5d_mean="mean",
        phase4_5d_std=lambda values: pd.to_numeric(values, errors="coerce").std(ddof=0),
        phase4_5d_count="count",
    ).reset_index()
    result["phase4_5d_mean"] = pd.to_numeric(result["phase4_5d_mean"], errors="coerce").round(2)
    result["phase4_5d_std"] = pd.to_numeric(result["phase4_5d_std"], errors="coerce").round(2)
    result["phase4_5d_count"] = pd.to_numeric(result["phase4_5d_count"], errors="coerce").astype("Int64")
    result = result.sort_values(["phase4_5d_mean", "symbol"], ascending=[False, True]).reset_index(drop=True)
    result["phase4_5d_mean_rank"] = result.index + 1
    result["phase4_5d_mean_top5"] = result["phase4_5d_mean_rank"].le(5)
    return result.loc[:, ["symbol", *PHASE4_ROLLING_COLUMNS, *PHASE4_ROLLING_EXTRA_COLUMNS, *PHASE4_ROLLING_RANK_COLUMNS]]


def merge_phase4_rolling_frame(
    frame: pd.DataFrame,
    *,
    project_root: Path,
    trade_date: date,
    window: int = 5,
) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return _ensure_rolling_columns(frame.copy())
    rolling = build_phase4_rolling_frame(project_root=project_root, trade_date=trade_date, window=window)
    if rolling.empty:
        return _ensure_rolling_columns(frame.copy())
    result = frame.copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result = result.drop(
        columns=[
            column
            for column in (*PHASE4_ROLLING_COLUMNS, *PHASE4_ROLLING_EXTRA_COLUMNS, *PHASE4_ROLLING_RANK_COLUMNS)
            if column in result.columns
        ],
        errors="ignore",
    )
    result = result.merge(rolling, on="symbol", how="left")
    return _ensure_rolling_columns(result)


def _recent_phase4_report_files(*, report_dir: Path, trade_date: date, window: int) -> list[Path]:
    if not report_dir.exists():
        return []
    candidates: list[tuple[date, Path]] = []
    for path in report_dir.glob("alpha158_qlib_return_predictions_*.csv"):
        match = _PHASE4_REPORT_RE.match(path.name)
        if not match:
            continue
        try:
            parsed = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if parsed <= trade_date:
            candidates.append((parsed, path))
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    return [path for _parsed, path in candidates[: max(int(window), 0)]]


def _read_phase4_score_file(path: Path) -> pd.DataFrame:
    try:
        header = pd.read_csv(path, nrows=0)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["symbol", "phase4_daily_score_100"])
    if "symbol" not in header.columns or "return_score" not in header.columns:
        return pd.DataFrame(columns=["symbol", "phase4_daily_score_100"])
    frame = pd.read_csv(path, usecols=["symbol", "return_score"])
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "phase4_daily_score_100"])
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame["return_score"] = pd.to_numeric(frame["return_score"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "return_score"])
    frame = frame.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    frame["phase4_daily_score_100"] = score_series_100(frame["return_score"], higher_is_better=True)
    return frame.loc[:, ["symbol", "phase4_daily_score_100"]].dropna(subset=["phase4_daily_score_100"])


def _ensure_rolling_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (*PHASE4_ROLLING_COLUMNS, *PHASE4_ROLLING_EXTRA_COLUMNS, *PHASE4_ROLLING_RANK_COLUMNS):
        if column not in result.columns:
            result[column] = pd.NA
    return result


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
