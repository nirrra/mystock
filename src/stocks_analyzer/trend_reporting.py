from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .atr import build_atr_export_frame
from .paths import ProjectPaths


def save_macd_report(
    paths: ProjectPaths,
    *,
    trade_date: date,
    dataframe: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "macd"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"macd_{trade_date.isoformat()}.csv"
    summary_path = target_dir / f"macd_{trade_date.isoformat()}.json"
    dataframe.to_csv(detail_path, index=False, encoding="utf-8-sig")
    _write_summary(summary_path, dataframe, trade_date=trade_date, kind="macd")
    return {"detail_path": detail_path, "summary_path": summary_path}


def save_atr_report(
    paths: ProjectPaths,
    *,
    trade_date: date,
    dataframe: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "atr"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"atr_{trade_date.isoformat()}.csv"
    summary_path = target_dir / f"atr_{trade_date.isoformat()}.json"
    build_atr_export_frame(dataframe).to_csv(detail_path, index=False, encoding="utf-8-sig")
    _write_summary(summary_path, dataframe, trade_date=trade_date, kind="atr")
    return {"detail_path": detail_path, "summary_path": summary_path}


def _write_summary(path: Path, dataframe: pd.DataFrame, *, trade_date: date, kind: str) -> None:
    summary = {
        "trade_date": trade_date.isoformat(),
        "kind": kind,
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
    }
    pd.Series(summary, dtype="object").to_json(path, force_ascii=False, indent=2)
