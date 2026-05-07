from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LinearRegression

from .full_market_panel import full_market_report_dir
from .storage import DailyBarsReadError, Storage


PROGRESS_LOG_INTERVAL = 500
DEFAULT_MCD_SUPPORT_FRACTION = 0.75
DEFAULT_MCD_CONTAMINATION = 0.04


@dataclass(slots=True)
class MCDCrashValidationResult:
    weekly_returns: pd.DataFrame
    annual_measures: pd.DataFrame
    distribution: pd.DataFrame
    correlation: pd.DataFrame
    report_dir: Path
    weekly_returns_path: Path
    annual_measures_path: Path
    distribution_path: Path
    correlation_path: Path
    config_path: Path


def validate_mcd_crash_risk(
    *,
    storage: Storage,
    project_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    min_weeks_per_year: int = 26,
    mcd_support_fraction: float = DEFAULT_MCD_SUPPORT_FRACTION,
    mcd_contamination: float = DEFAULT_MCD_CONTAMINATION,
) -> MCDCrashValidationResult:
    logging.info("MCD crash-risk weekly panel build started")
    weekly_returns = build_firm_specific_weekly_returns(
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if weekly_returns.empty:
        raise RuntimeError("MCD crash-risk reproduction has no weekly returns.")
    annual_measures = build_crash_measures(
        weekly_returns,
        min_weeks_per_year=min_weeks_per_year,
        mcd_support_fraction=mcd_support_fraction,
        mcd_contamination=mcd_contamination,
    )
    if annual_measures.empty:
        raise RuntimeError("MCD crash-risk reproduction has no annual crash measures.")
    distribution = summarize_crash_measure_distribution(annual_measures)
    correlation = build_crash_measure_correlation(annual_measures)

    report_dir = full_market_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    weekly_returns_path = report_dir / "mcd_crash_weekly_returns.csv"
    annual_measures_path = report_dir / "mcd_crash_annual_measures.csv"
    distribution_path = report_dir / "mcd_crash_label_distribution.csv"
    correlation_path = report_dir / "mcd_crash_measure_correlation.csv"
    config_path = report_dir / "mcd_crash_config.json"
    weekly_returns.to_csv(weekly_returns_path, index=False, encoding="utf-8-sig")
    annual_measures.to_csv(annual_measures_path, index=False, encoding="utf-8-sig")
    distribution.to_csv(distribution_path, index=False, encoding="utf-8-sig")
    correlation.to_csv(correlation_path, index=False, encoding="utf-8-sig")
    config_path.write_text(
        json.dumps(
            {
                "reference": "Karasan, Alp and Weber (2025), Machine learning approach to stock price crash risk",
                "scope": "label generation only: NEGOUTLIER, CRASH, NCSKEW, DUVOL",
                "frequency": "weekly",
                "market_return": "local equal-weight weekly return built from available stock daily bars",
                "start_date": start_date.isoformat() if start_date else "",
                "end_date": end_date.isoformat() if end_date else "",
                "limit": limit,
                "min_weeks_per_year": int(min_weeks_per_year),
                "mcd_support_fraction": float(mcd_support_fraction),
                "mcd_contamination": float(mcd_contamination),
                "local_deviations": [
                    "CRSP/Compustat accounting variables are not available, so pooled logistic regression is not reproduced",
                    "official market index is replaced by local equal-weight weekly market return",
                    "industry and year fixed-effect regressions are outside this label-generation phase",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return MCDCrashValidationResult(
        weekly_returns=weekly_returns,
        annual_measures=annual_measures,
        distribution=distribution,
        correlation=correlation,
        report_dir=report_dir,
        weekly_returns_path=weekly_returns_path,
        annual_measures_path=annual_measures_path,
        distribution_path=distribution_path,
        correlation_path=correlation_path,
        config_path=config_path,
    )


def build_firm_specific_weekly_returns(
    *,
    storage: Storage,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    universe = storage.load_universe().copy()
    if limit is not None:
        universe = universe.head(max(int(limit), 0)).copy()
    weekly_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    instruments = universe.to_dict("records")
    total_symbols = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        _log_progress("MCD weekly returns", index, total_symbols)
        symbol = str(instrument.get("symbol", "")).zfill(6)
        name = str(instrument.get("name", ""))
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError) as exc:
            skipped.append({"symbol": symbol, "name": name, "reason": type(exc).__name__})
            continue
        weekly = build_symbol_weekly_return_frame(bars, symbol=symbol, name=name)
        if start_date is not None:
            weekly = weekly[weekly["week_end"].dt.date >= start_date]
        if end_date is not None:
            weekly = weekly[weekly["week_end"].dt.date <= end_date]
        if weekly.empty:
            skipped.append({"symbol": symbol, "name": name, "reason": "no_weekly_returns"})
            continue
        weekly_parts.append(weekly)
    panel = pd.concat(weekly_parts, ignore_index=True) if weekly_parts else pd.DataFrame()
    if panel.empty:
        return pd.DataFrame()
    market = panel.groupby("week_end", sort=True)["weekly_return"].mean().rename("market_weekly_return")
    panel = panel.merge(market, on="week_end", how="left")
    panel = panel.sort_values(["symbol", "week_end"]).reset_index(drop=True)
    panel["firm_specific_weekly_return"] = np.nan
    for symbol, group_index in panel.groupby("symbol").groups.items():
        group = panel.loc[group_index].copy()
        valid = group[["weekly_return", "market_weekly_return"]].dropna()
        if len(valid) < 20 or valid["market_weekly_return"].nunique() < 2:
            panel.loc[group_index, "firm_specific_weekly_return"] = group["weekly_return"].values
            continue
        model = LinearRegression()
        model.fit(valid[["market_weekly_return"]], valid["weekly_return"])
        pred = model.predict(group[["market_weekly_return"]].fillna(0.0))
        panel.loc[group_index, "firm_specific_weekly_return"] = group["weekly_return"].values - pred
    return panel


def build_symbol_weekly_return_frame(bars: pd.DataFrame, *, symbol: str, name: str = "") -> pd.DataFrame:
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").set_index("trade_date")
    close = pd.to_numeric(frame["close"], errors="coerce").where(lambda values: values.gt(0))
    weekly_close = close.resample("W-FRI").last().dropna()
    weekly_return = weekly_close.div(weekly_close.shift(1)).sub(1.0).replace([np.inf, -np.inf], np.nan)
    result = pd.DataFrame(
        {
            "week_end": weekly_return.index,
            "symbol": str(symbol).zfill(6),
            "name": name,
            "weekly_return": weekly_return.values,
        }
    ).dropna(subset=["weekly_return"])
    result["year"] = result["week_end"].dt.year
    return result.reset_index(drop=True)


def build_crash_measures(
    weekly_returns: pd.DataFrame,
    *,
    min_weeks_per_year: int = 26,
    mcd_support_fraction: float = DEFAULT_MCD_SUPPORT_FRACTION,
    mcd_contamination: float = DEFAULT_MCD_CONTAMINATION,
) -> pd.DataFrame:
    rows = []
    for (symbol, year), group in weekly_returns.groupby(["symbol", "year"], sort=True):
        firm_return = pd.to_numeric(group["firm_specific_weekly_return"], errors="coerce").dropna()
        if len(firm_return) < min_weeks_per_year:
            continue
        crash_threshold = firm_return.mean() - 3.2 * firm_return.std(ddof=1)
        crash_flags = firm_return.lt(crash_threshold)
        down = firm_return[firm_return.lt(firm_return.mean())]
        up = firm_return[firm_return.ge(firm_return.mean())]
        rows.append(
            {
                "symbol": str(symbol).zfill(6),
                "name": str(group["name"].iloc[0]),
                "year": int(year),
                "weeks": int(len(firm_return)),
                "NEGOUTLIER": 0,
                "CRASH": int(crash_flags.any()),
                "CRASH_count": int(crash_flags.sum()),
                "NCSKEW": _ncskew(firm_return),
                "DUVOL": _duvol(up=up, down=down),
                "RET": float(firm_return.mean()),
                "SIGMA": float(firm_return.std(ddof=1)),
                "MINRET": float(firm_return.min()),
            }
        )
    measures = pd.DataFrame(rows)
    if measures.empty:
        return measures
    measures["NEGOUTLIER"] = _annual_mcd_negoutlier_flags(
        measures,
        support_fraction=mcd_support_fraction,
        contamination=mcd_contamination,
    ).astype(int)
    return measures


def summarize_crash_measure_distribution(annual_measures: pd.DataFrame) -> pd.DataFrame:
    if annual_measures.empty:
        return pd.DataFrame()
    rows = []
    for year, group in annual_measures.groupby("year", sort=True):
        rows.append(_distribution_row(group, year=year))
    rows.append(_distribution_row(annual_measures, year="all"))
    return pd.DataFrame(rows)


def build_crash_measure_correlation(annual_measures: pd.DataFrame) -> pd.DataFrame:
    columns = ["NEGOUTLIER", "CRASH", "NCSKEW", "DUVOL", "RET", "SIGMA", "MINRET"]
    available = [column for column in columns if column in annual_measures.columns]
    if len(available) < 2:
        return pd.DataFrame()
    corr = annual_measures.loc[:, available].corr()
    rows = []
    for left in available:
        for right in available:
            rows.append({"measure_left": left, "measure_right": right, "correlation": float(corr.loc[left, right])})
    return pd.DataFrame(rows)


def _annual_mcd_negoutlier_flags(
    measures: pd.DataFrame,
    *,
    support_fraction: float,
    contamination: float,
) -> pd.Series:
    flags = pd.Series(False, index=measures.index)
    candidate_columns = ["NCSKEW", "DUVOL", "MINRET"]
    for _, group in measures.groupby("year", sort=True):
        raw_features = group.loc[:, candidate_columns].replace([np.inf, -np.inf], np.nan)
        feature_columns = [column for column in candidate_columns if raw_features[column].notna().sum() >= max(10, len(group) // 2)]
        features = raw_features.loc[:, feature_columns].dropna()
        if len(features) < max(10, len(feature_columns) * 3) or not feature_columns:
            continue
        threshold = chi2.ppf(max(0.0, min(1.0, 1.0 - contamination)), df=len(feature_columns))
        try:
            mcd = MinCovDet(support_fraction=support_fraction, random_state=42).fit(features)
            distances = pd.Series(mcd.mahalanobis(features), index=features.index)
        except Exception:
            centered = features.sub(features.median())
            scale = centered.abs().median().replace(0, np.nan).fillna(features.std(ddof=1)).replace(0, np.nan)
            distances = centered.div(scale).pow(2).sum(axis=1)
        negative_side = pd.Series(False, index=features.index)
        if "MINRET" in features:
            negative_side = negative_side | features["MINRET"].lt(features["MINRET"].median())
        if "NCSKEW" in features:
            negative_side = negative_side | features["NCSKEW"].gt(features["NCSKEW"].median())
        flags.loc[features.index] = distances.gt(threshold) & negative_side
    return flags


def _ncskew(values: pd.Series) -> float:
    n = len(values)
    if n < 3:
        return np.nan
    numerator = n * (n - 1) ** 1.5 * (values**3).sum()
    denominator = (n - 1) * (n - 2) * ((values**2).sum() ** 1.5)
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return float(-numerator / denominator)


def _duvol(*, up: pd.Series, down: pd.Series) -> float:
    if len(up) < 2 or len(down) < 2:
        return np.nan
    up_var = (up**2).sum() / (len(up) - 1)
    down_var = (down**2).sum() / (len(down) - 1)
    if up_var <= 0 or down_var <= 0:
        return np.nan
    return float(np.log(down_var / up_var))


def _distribution_row(group: pd.DataFrame, *, year: int | str) -> dict[str, object]:
    return {
        "year": year,
        "firm_years": int(len(group)),
        "NEGOUTLIER_rate": float(group["NEGOUTLIER"].mean()),
        "CRASH_rate": float(group["CRASH"].mean()),
        "CRASH_count_mean": float(group["CRASH_count"].mean()),
        "NCSKEW_mean": float(group["NCSKEW"].mean()),
        "DUVOL_mean": float(group["DUVOL"].mean()),
        "RET_mean": float(group["RET"].mean()),
        "SIGMA_mean": float(group["SIGMA"].mean()),
        "MINRET_mean": float(group["MINRET"].mean()),
    }


def _log_progress(stage_name: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current % PROGRESS_LOG_INTERVAL == 0 or current == total:
        logging.info("%s progress: %s/%s", stage_name, current, total)
