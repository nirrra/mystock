from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .paths import ProjectPaths


def save_trend_universe_report(
    paths: ProjectPaths,
    *,
    trade_date: date,
    dataframe: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "trend_universe"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"trend_universe_{trade_date.isoformat()}.csv"
    dataframe.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = target_dir / f"trend_universe_{trade_date.isoformat()}.json"
    payload = {
        "trade_date": trade_date.isoformat(),
        "candidate_count": int(len(dataframe)),
        "symbols": dataframe["symbol"].astype(str).tolist() if "symbol" in dataframe.columns else [],
        "report_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


def save_trend_signals_report(
    paths: ProjectPaths,
    *,
    trade_date: date,
    dataframe: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "trend_signals"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"trend_signals_{trade_date.isoformat()}.csv"
    dataframe.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = target_dir / f"trend_signals_{trade_date.isoformat()}.json"
    payload = {
        "trade_date": trade_date.isoformat(),
        "signal_count": int(len(dataframe)),
        "signal_type_counts": (
            {str(key): int(value) for key, value in dataframe["signal_type"].value_counts().to_dict().items()}
            if "signal_type" in dataframe.columns
            else {}
        ),
        "report_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


def save_trend_entries_report(
    paths: ProjectPaths,
    *,
    trade_date: date,
    dataframe: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "trend_entries"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"trend_entries_{trade_date.isoformat()}.csv"
    dataframe.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = target_dir / f"trend_entries_{trade_date.isoformat()}.json"
    payload = {
        "trade_date": trade_date.isoformat(),
        "entry_count": int(len(dataframe)),
        "symbols": dataframe["symbol"].astype(str).tolist() if "symbol" in dataframe.columns else [],
        "report_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


def save_trend_scores_report(
    paths: ProjectPaths,
    *,
    trade_date: date,
    dataframe: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "trend_scores"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"trend_scores_{trade_date.isoformat()}.csv"
    dataframe.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = target_dir / f"trend_scores_{trade_date.isoformat()}.json"
    payload = {
        "trade_date": trade_date.isoformat(),
        "row_count": int(len(dataframe)),
        "symbols": dataframe["symbol"].astype(str).tolist() if "symbol" in dataframe.columns else [],
        "report_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


def save_trend_report(
    paths: ProjectPaths,
    *,
    trade_date: date,
    dataframe: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "trend"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"trend_{trade_date.isoformat()}.csv"
    dataframe.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = target_dir / f"trend_{trade_date.isoformat()}.json"
    payload = {
        "trade_date": trade_date.isoformat(),
        "row_count": int(len(dataframe)),
        "symbols": dataframe["symbol"].astype(str).tolist() if "symbol" in dataframe.columns else [],
        "report_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


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
    dataframe.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_path = target_dir / f"macd_{trade_date.isoformat()}.json"
    payload = {
        "trade_date": trade_date.isoformat(),
        "row_count": int(len(dataframe)),
        "symbols": dataframe["symbol"].astype(str).tolist() if "symbol" in dataframe.columns else [],
        "report_path": str(detail_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


def save_signal_backtest_reports(
    paths: ProjectPaths,
    *,
    report_date: date,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "backtests" / "signals"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"signal_backtest_details_{report_date.isoformat()}.csv"
    summary_path = target_dir / f"signal_backtest_summary_{report_date.isoformat()}.csv"
    json_path = target_dir / f"signal_backtest_summary_{report_date.isoformat()}.json"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    payload = {
        "report_date": report_date.isoformat(),
        "detail_rows": int(len(detail)),
        "summary_rows": int(len(summary)),
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "entry_note": detail["entry_note"].iloc[0] if not detail.empty and "entry_note" in detail.columns else None,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
        "json_path": json_path,
    }


def save_entry_backtest_reports(
    paths: ProjectPaths,
    *,
    report_date: date,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "backtests" / "entries"
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = Path(output) if output else target_dir / f"entry_backtest_details_{report_date.isoformat()}.csv"
    summary_path = target_dir / f"entry_backtest_summary_{report_date.isoformat()}.csv"
    json_path = target_dir / f"entry_backtest_summary_{report_date.isoformat()}.json"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    payload = {
        "report_date": report_date.isoformat(),
        "detail_rows": int(len(detail)),
        "summary_rows": int(len(summary)),
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "entry_note": detail["entry_note"].iloc[0] if not detail.empty and "entry_note" in detail.columns else None,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
        "json_path": json_path,
    }


def save_portfolio_backtest_reports(
    paths: ProjectPaths,
    *,
    report_date: date,
    positions: pd.DataFrame,
    equity: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "backtests" / "portfolio"
    target_dir.mkdir(parents=True, exist_ok=True)
    positions_path = target_dir / f"portfolio_positions_{report_date.isoformat()}.csv"
    equity_path = target_dir / f"portfolio_equity_{report_date.isoformat()}.csv"
    summary_path = target_dir / f"portfolio_summary_{report_date.isoformat()}.csv"
    json_path = target_dir / f"portfolio_summary_{report_date.isoformat()}.json"

    positions.to_csv(positions_path, index=False, encoding="utf-8-sig")
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    payload = {
        "report_date": report_date.isoformat(),
        "position_rows": int(len(positions)),
        "equity_rows": int(len(equity)),
        "summary_rows": int(len(summary)),
        "positions_path": str(positions_path),
        "equity_path": str(equity_path),
        "summary_path": str(summary_path),
        "entry_note": positions["entry_note"].iloc[0] if not positions.empty and "entry_note" in positions.columns else None,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "positions_path": positions_path,
        "equity_path": equity_path,
        "summary_path": summary_path,
        "json_path": json_path,
    }


def save_entry_portfolio_backtest_reports(
    paths: ProjectPaths,
    *,
    report_date: date,
    positions: pd.DataFrame,
    equity: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "backtests" / "entries_portfolio"
    target_dir.mkdir(parents=True, exist_ok=True)
    positions_path = target_dir / f"entry_portfolio_positions_{report_date.isoformat()}.csv"
    equity_path = target_dir / f"entry_portfolio_equity_{report_date.isoformat()}.csv"
    summary_path = target_dir / f"entry_portfolio_summary_{report_date.isoformat()}.csv"
    json_path = target_dir / f"entry_portfolio_summary_{report_date.isoformat()}.json"

    positions.to_csv(positions_path, index=False, encoding="utf-8-sig")
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    payload = {
        "report_date": report_date.isoformat(),
        "position_rows": int(len(positions)),
        "equity_rows": int(len(equity)),
        "summary_rows": int(len(summary)),
        "positions_path": str(positions_path),
        "equity_path": str(equity_path),
        "summary_path": str(summary_path),
        "entry_note": positions["entry_note"].iloc[0] if not positions.empty and "entry_note" in positions.columns else None,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "positions_path": positions_path,
        "equity_path": equity_path,
        "summary_path": summary_path,
        "json_path": json_path,
    }


def save_threshold_research_reports(
    paths: ProjectPaths,
    *,
    report_date: date,
    samples: pd.DataFrame,
    distributions: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_evaluation: pd.DataFrame,
    combo_candidates: pd.DataFrame,
    combo_evaluation: pd.DataFrame,
    default_candidates: pd.DataFrame,
    output: str | None = None,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "threshold_research"
    target_dir.mkdir(parents=True, exist_ok=True)

    samples_path = Path(output) if output else target_dir / f"threshold_samples_{report_date.isoformat()}.csv"
    distributions_path = target_dir / f"threshold_distributions_{report_date.isoformat()}.csv"
    candidates_path = target_dir / f"threshold_candidates_{report_date.isoformat()}.csv"
    candidate_eval_path = target_dir / f"threshold_candidate_eval_{report_date.isoformat()}.csv"
    combo_candidates_path = target_dir / f"threshold_combo_candidates_{report_date.isoformat()}.csv"
    combo_eval_path = target_dir / f"threshold_combo_eval_{report_date.isoformat()}.csv"
    default_candidates_path = target_dir / f"threshold_default_candidates_{report_date.isoformat()}.csv"
    json_path = target_dir / f"threshold_research_{report_date.isoformat()}.json"

    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    distributions.to_csv(distributions_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    candidate_evaluation.to_csv(candidate_eval_path, index=False, encoding="utf-8-sig")
    combo_candidates.to_csv(combo_candidates_path, index=False, encoding="utf-8-sig")
    combo_evaluation.to_csv(combo_eval_path, index=False, encoding="utf-8-sig")
    default_candidates.to_csv(default_candidates_path, index=False, encoding="utf-8-sig")

    payload = {
        "report_date": report_date.isoformat(),
        "samples_rows": int(len(samples)),
        "distributions_rows": int(len(distributions)),
        "candidates_rows": int(len(candidates)),
        "candidate_evaluation_rows": int(len(candidate_evaluation)),
        "combo_candidates_rows": int(len(combo_candidates)),
        "combo_evaluation_rows": int(len(combo_evaluation)),
        "default_candidates_rows": int(len(default_candidates)),
        "samples_path": str(samples_path),
        "distributions_path": str(distributions_path),
        "candidates_path": str(candidates_path),
        "candidate_evaluation_path": str(candidate_eval_path),
        "combo_candidates_path": str(combo_candidates_path),
        "combo_evaluation_path": str(combo_eval_path),
        "default_candidates_path": str(default_candidates_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "samples_path": samples_path,
        "distributions_path": distributions_path,
        "candidates_path": candidates_path,
        "candidate_evaluation_path": candidate_eval_path,
        "combo_candidates_path": combo_candidates_path,
        "combo_evaluation_path": combo_eval_path,
        "default_candidates_path": default_candidates_path,
        "json_path": json_path,
    }
