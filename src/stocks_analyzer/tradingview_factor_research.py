from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from .paths import ProjectPaths
from .storage import DailyBarsReadError, Storage
from .technical_ratings import add_technical_ratings


DEFAULT_FACTOR_FIELDS = ("all_rating", "avg_all_rating_5d", "ma_rating", "osc_rating")
DEFAULT_RANK_FIELDS = ("all_rating", "avg_all_rating_5d")
DEFAULT_HORIZONS = (1, 5, 10, 20)
DEFAULT_TOP_N = 10
DEFAULT_QUANTILES = 5


@dataclass(slots=True)
class TradingViewFactorResearchResult:
    samples: pd.DataFrame
    daily_ic: pd.DataFrame
    ic_summary: pd.DataFrame
    quantile_returns: pd.DataFrame
    label_returns: pd.DataFrame
    topn_detail: pd.DataFrame
    topn_daily: pd.DataFrame
    topn_summary: pd.DataFrame


def run_tradingview_factor_research(
    storage: Storage,
    *,
    start_date: date,
    end_date: date,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    factor_fields: tuple[str, ...] = DEFAULT_FACTOR_FIELDS,
    rank_fields: tuple[str, ...] = DEFAULT_RANK_FIELDS,
    top_n: int = DEFAULT_TOP_N,
    quantiles: int = DEFAULT_QUANTILES,
    symbols: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TradingViewFactorResearchResult:
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date")
    if not horizons or any(int(item) <= 0 for item in horizons):
        raise ValueError("horizons must contain positive integers")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")

    normalized_horizons = tuple(sorted({int(item) for item in horizons}))
    normalized_factor_fields = tuple(dict.fromkeys(factor_fields))
    normalized_rank_fields = tuple(dict.fromkeys(rank_fields))

    samples = build_tradingview_factor_samples(
        storage,
        start_date=start_date,
        end_date=end_date,
        horizons=normalized_horizons,
        symbols=symbols,
        progress_callback=progress_callback,
    )
    daily_ic = summarize_daily_ic(samples, factor_fields=normalized_factor_fields, horizons=normalized_horizons)
    ic_summary = summarize_ic(daily_ic)
    quantile_returns = summarize_quantile_returns(
        samples,
        factor_fields=normalized_factor_fields,
        horizons=normalized_horizons,
        quantiles=quantiles,
    )
    label_returns = summarize_label_returns(samples, horizons=normalized_horizons)
    topn_detail = build_topn_forward_returns(
        samples,
        rank_fields=normalized_rank_fields,
        horizons=normalized_horizons,
        top_n=top_n,
    )
    topn_daily = summarize_topn_daily_returns(topn_detail, horizons=normalized_horizons, top_n=top_n)
    topn_summary = summarize_topn_returns(topn_detail, topn_daily, horizons=normalized_horizons, top_n=top_n)
    return TradingViewFactorResearchResult(
        samples=samples,
        daily_ic=daily_ic,
        ic_summary=ic_summary,
        quantile_returns=quantile_returns,
        label_returns=label_returns,
        topn_detail=topn_detail,
        topn_daily=topn_daily,
        topn_summary=topn_summary,
    )


def build_tradingview_factor_samples(
    storage: Storage,
    *,
    start_date: date,
    end_date: date,
    horizons: tuple[int, ...],
    symbols: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    universe = storage.load_universe().copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    if symbols:
        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        universe = universe[universe["symbol"].isin(symbol_set)].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    instruments = universe.to_dict("records")
    total = len(instruments)
    for index, instrument in enumerate(instruments, start=1):
        symbol = str(instrument["symbol"]).zfill(6)
        try:
            bars = storage.load_daily_bars(symbol)
        except (FileNotFoundError, DailyBarsReadError):
            if progress_callback is not None:
                progress_callback(index, total)
            continue

        rows.extend(
            _build_symbol_factor_rows(
                bars,
                symbol=symbol,
                name=str(instrument.get("name", "")),
                start_date=start_date,
                end_date=end_date,
                horizons=horizons,
            )
        )
        if progress_callback is not None:
            progress_callback(index, total)

    if not rows:
        return pd.DataFrame(columns=_sample_columns(horizons))

    samples = pd.DataFrame(rows)
    samples["trade_date"] = pd.to_datetime(samples["trade_date"])
    samples["entry_date"] = pd.to_datetime(samples["entry_date"])
    samples = samples.sort_values(["trade_date", "symbol"], ascending=[True, True]).reset_index(drop=True)
    return samples.reindex(columns=_sample_columns(horizons))


def _build_symbol_factor_rows(
    daily_bars: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    start_date: date,
    end_date: date,
    horizons: tuple[int, ...],
) -> list[dict[str, object]]:
    if daily_bars.empty:
        return []

    frame = add_technical_ratings(daily_bars).copy().sort_values("trade_date").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["avg_all_rating_5d"] = frame["all_rating"].rolling(5, min_periods=5).mean()
    frame["avg_ma_rating_5d"] = frame["ma_rating"].rolling(5, min_periods=5).mean()
    frame["avg_osc_rating_5d"] = frame["osc_rating"].rolling(5, min_periods=5).mean()

    rows: list[dict[str, object]] = []
    for row_index in range(len(frame)):
        trade_date = pd.Timestamp(frame.iloc[row_index]["trade_date"]).date()
        if trade_date < start_date or trade_date > end_date:
            continue

        entry_index = row_index + 1
        if entry_index >= len(frame):
            continue
        entry = frame.iloc[entry_index]
        entry_open = _safe_float_or_none(entry.get("open"))
        if entry_open is None or entry_open <= 0:
            continue

        row = frame.iloc[row_index]
        sample: dict[str, object] = {
            "trade_date": pd.Timestamp(row["trade_date"]),
            "symbol": symbol,
            "name": name,
            "close": _safe_float_or_none(row.get("close")),
            "ma_rating": _safe_float_or_none(row.get("ma_rating")),
            "osc_rating": _safe_float_or_none(row.get("osc_rating")),
            "all_rating": _safe_float_or_none(row.get("all_rating")),
            "avg_ma_rating_5d": _safe_float_or_none(row.get("avg_ma_rating_5d")),
            "avg_osc_rating_5d": _safe_float_or_none(row.get("avg_osc_rating_5d")),
            "avg_all_rating_5d": _safe_float_or_none(row.get("avg_all_rating_5d")),
            "ma_rating_label": row.get("ma_rating_label"),
            "osc_rating_label": row.get("osc_rating_label"),
            "all_rating_label": row.get("all_rating_label"),
            "entry_date": pd.Timestamp(entry["trade_date"]),
            "entry_open": round(entry_open, 4),
        }

        for horizon in horizons:
            exit_index = entry_index + horizon - 1
            if exit_index >= len(frame):
                _attach_empty_forward_fields(sample, horizon)
                continue
            window = frame.iloc[entry_index : exit_index + 1]
            exit_row = frame.iloc[exit_index]
            exit_close = _safe_float_or_none(exit_row.get("close"))
            max_high = _safe_float_or_none(pd.to_numeric(window["high"], errors="coerce").max())
            min_low = _safe_float_or_none(pd.to_numeric(window["low"], errors="coerce").min())
            sample[f"exit_date_{horizon}d"] = pd.Timestamp(exit_row["trade_date"])
            sample[f"exit_close_{horizon}d"] = round(exit_close, 4) if exit_close is not None else None
            sample[f"forward_return_{horizon}d"] = round(exit_close / entry_open - 1.0, 6) if exit_close is not None else None
            sample[f"max_upside_{horizon}d"] = round(max_high / entry_open - 1.0, 6) if max_high is not None else None
            sample[f"max_drawdown_{horizon}d"] = round(min_low / entry_open - 1.0, 6) if min_low is not None else None

        # Keep only rows that can evaluate at least the shortest requested horizon.
        if sample.get(f"forward_return_{min(horizons)}d") is not None:
            rows.append(sample)

    return rows


def summarize_daily_ic(
    samples: pd.DataFrame,
    *,
    factor_fields: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if samples.empty:
        return pd.DataFrame(columns=["trade_date", "factor", "horizon_days", "sample_count", "rank_ic"])

    for factor in factor_fields:
        if factor not in samples.columns:
            continue
        for horizon in horizons:
            return_column = f"forward_return_{horizon}d"
            if return_column not in samples.columns:
                continue
            for trade_date, group in samples.groupby("trade_date", sort=True):
                valid = group[[factor, return_column]].dropna()
                if len(valid) < 5:
                    rank_ic = None
                else:
                    rank_ic = valid[factor].corr(valid[return_column], method="spearman")
                rows.append(
                    {
                        "trade_date": pd.Timestamp(trade_date),
                        "factor": factor,
                        "horizon_days": horizon,
                        "sample_count": int(len(valid)),
                        "rank_ic": round(float(rank_ic), 6) if rank_ic is not None and pd.notna(rank_ic) else None,
                    }
                )
    return pd.DataFrame(rows)


def summarize_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    if daily_ic.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "horizon_days",
                "ic_days",
                "mean_rank_ic",
                "median_rank_ic",
                "std_rank_ic",
                "positive_ic_rate",
                "avg_sample_count",
            ]
        )

    rows: list[dict[str, object]] = []
    for (factor, horizon), group in daily_ic.groupby(["factor", "horizon_days"], sort=True):
        ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        rows.append(
            {
                "factor": factor,
                "horizon_days": int(horizon),
                "ic_days": int(len(ic)),
                "mean_rank_ic": round(float(ic.mean()), 6) if not ic.empty else None,
                "median_rank_ic": round(float(ic.median()), 6) if not ic.empty else None,
                "std_rank_ic": round(float(ic.std(ddof=0)), 6) if len(ic) > 1 else 0.0 if len(ic) == 1 else None,
                "positive_ic_rate": round(float((ic > 0).mean()), 6) if not ic.empty else None,
                "avg_sample_count": round(float(pd.to_numeric(group["sample_count"], errors="coerce").mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def summarize_quantile_returns(
    samples: pd.DataFrame,
    *,
    factor_fields: tuple[str, ...],
    horizons: tuple[int, ...],
    quantiles: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if samples.empty:
        return pd.DataFrame(columns=_group_summary_columns(group_columns=["factor", "horizon_days", "quantile"]))

    for factor in factor_fields:
        if factor not in samples.columns:
            continue
        frame = samples.copy()
        frame["quantile"] = frame.groupby("trade_date", group_keys=False)[factor].apply(
            lambda series: _daily_quantiles(series, quantiles)
        )
        frame = frame.dropna(subset=["quantile"])
        if frame.empty:
            continue
        frame["quantile"] = frame["quantile"].astype(int)
        for horizon in horizons:
            rows.extend(
                _summarize_return_groups(
                    frame,
                    group_columns=["quantile"],
                    horizon=horizon,
                    metadata={"factor": factor},
                )
            )
    if not rows:
        return pd.DataFrame(columns=_group_summary_columns(group_columns=["factor", "horizon_days", "quantile"]))
    return pd.DataFrame(rows).sort_values(["factor", "horizon_days", "quantile"]).reset_index(drop=True)


def summarize_label_returns(samples: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    if samples.empty or "all_rating_label" not in samples.columns:
        return pd.DataFrame(columns=_group_summary_columns(group_columns=["horizon_days", "all_rating_label"]))

    rows: list[dict[str, object]] = []
    frame = samples.dropna(subset=["all_rating_label"]).copy()
    for horizon in horizons:
        rows.extend(_summarize_return_groups(frame, group_columns=["all_rating_label"], horizon=horizon, metadata={}))
    if not rows:
        return pd.DataFrame(columns=_group_summary_columns(group_columns=["horizon_days", "all_rating_label"]))
    return pd.DataFrame(rows).sort_values(["horizon_days", "all_rating_label"]).reset_index(drop=True)


def build_topn_forward_returns(
    samples: pd.DataFrame,
    *,
    rank_fields: tuple[str, ...],
    horizons: tuple[int, ...],
    top_n: int,
) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=_topn_columns(horizons))

    rows: list[pd.DataFrame] = []
    for rank_field in rank_fields:
        if rank_field not in samples.columns:
            continue
        frame = samples.dropna(subset=[rank_field]).copy()
        if frame.empty:
            continue
        frame["rank_field"] = rank_field
        frame["rank"] = frame.groupby("trade_date")[rank_field].rank(method="first", ascending=False)
        frame = frame[frame["rank"] <= top_n].copy()
        if frame.empty:
            continue
        frame["rank"] = frame["rank"].astype(int)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=_topn_columns(horizons))
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["rank_field", "trade_date", "rank"]).reset_index(drop=True).reindex(columns=_topn_columns(horizons))


def summarize_topn_daily_returns(topn_detail: pd.DataFrame, *, horizons: tuple[int, ...], top_n: int) -> pd.DataFrame:
    if topn_detail.empty:
        return pd.DataFrame(columns=["rank_field", "top_count", "trade_date", *[f"portfolio_return_{horizon}d" for horizon in horizons]])

    rows: list[dict[str, object]] = []
    top_counts = _top_counts(top_n)
    for rank_field, field_group in topn_detail.groupby("rank_field", sort=True):
        for top_count in top_counts:
            top_frame = field_group[field_group["rank"] <= top_count]
            for trade_date, date_group in top_frame.groupby("trade_date", sort=True):
                row: dict[str, object] = {
                    "rank_field": rank_field,
                    "top_count": top_count,
                    "trade_date": pd.Timestamp(trade_date),
                    "stock_count": int(len(date_group)),
                }
                for horizon in horizons:
                    values = pd.to_numeric(date_group[f"forward_return_{horizon}d"], errors="coerce").dropna()
                    row[f"portfolio_return_{horizon}d"] = round(float(values.mean()), 6) if not values.empty else None
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["rank_field", "top_count", "trade_date"]).reset_index(drop=True)


def summarize_topn_returns(
    topn_detail: pd.DataFrame,
    topn_daily: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    top_n: int,
) -> pd.DataFrame:
    if topn_detail.empty:
        return pd.DataFrame(columns=_topn_summary_columns())

    rows: list[dict[str, object]] = []
    for rank_field, field_group in topn_detail.groupby("rank_field", sort=True):
        for top_count in _top_counts(top_n):
            stocks = field_group[field_group["rank"] <= top_count].copy()
            daily = topn_daily[(topn_daily["rank_field"] == rank_field) & (topn_daily["top_count"] == top_count)].copy()
            for horizon in horizons:
                stock_returns = pd.to_numeric(stocks[f"forward_return_{horizon}d"], errors="coerce").dropna()
                daily_returns = pd.to_numeric(daily[f"portfolio_return_{horizon}d"], errors="coerce").dropna()
                max_upside = pd.to_numeric(stocks[f"max_upside_{horizon}d"], errors="coerce").dropna()
                max_drawdown = pd.to_numeric(stocks[f"max_drawdown_{horizon}d"], errors="coerce").dropna()
                rows.append(
                    {
                        "rank_field": rank_field,
                        "top_count": top_count,
                        "horizon_days": horizon,
                        "trade_days": int(len(daily_returns)),
                        "stock_samples": int(len(stock_returns)),
                        "avg_stock_return": _rounded_mean(stock_returns),
                        "median_stock_return": _rounded_median(stock_returns),
                        "stock_win_rate": _rounded_rate(stock_returns > 0),
                        "avg_daily_equal_weight_return": _rounded_mean(daily_returns),
                        "median_daily_equal_weight_return": _rounded_median(daily_returns),
                        "daily_win_rate": _rounded_rate(daily_returns > 0),
                        "worst_daily_equal_weight_return": _rounded_min(daily_returns),
                        "avg_max_upside": _rounded_mean(max_upside),
                        "avg_max_drawdown": _rounded_mean(max_drawdown),
                    }
                )
    return pd.DataFrame(rows).sort_values(["rank_field", "top_count", "horizon_days"]).reset_index(drop=True)


def save_tradingview_factor_research_reports(
    paths: ProjectPaths,
    *,
    result: TradingViewFactorResearchResult,
    start_date: date,
    end_date: date,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Path]:
    target_dir = paths.reports_dir / "tradingview_factor"
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start_date.isoformat()}_{end_date.isoformat()}"
    report_paths = {
        "samples_path": target_dir / f"tradingview_factor_samples_{suffix}.csv",
        "daily_ic_path": target_dir / f"tradingview_factor_daily_ic_{suffix}.csv",
        "ic_summary_path": target_dir / f"tradingview_factor_ic_summary_{suffix}.csv",
        "quantile_returns_path": target_dir / f"tradingview_factor_quantile_returns_{suffix}.csv",
        "label_returns_path": target_dir / f"tradingview_factor_label_returns_{suffix}.csv",
        "topn_detail_path": target_dir / f"tradingview_top{top_n}_forward_returns_{suffix}.csv",
        "topn_daily_path": target_dir / f"tradingview_top{top_n}_daily_portfolio_{suffix}.csv",
        "topn_summary_path": target_dir / f"tradingview_top{top_n}_summary_{suffix}.csv",
        "json_path": target_dir / f"tradingview_factor_research_{suffix}.json",
    }
    result.samples.to_csv(report_paths["samples_path"], index=False, encoding="utf-8-sig")
    result.daily_ic.to_csv(report_paths["daily_ic_path"], index=False, encoding="utf-8-sig")
    result.ic_summary.to_csv(report_paths["ic_summary_path"], index=False, encoding="utf-8-sig")
    result.quantile_returns.to_csv(report_paths["quantile_returns_path"], index=False, encoding="utf-8-sig")
    result.label_returns.to_csv(report_paths["label_returns_path"], index=False, encoding="utf-8-sig")
    result.topn_detail.to_csv(report_paths["topn_detail_path"], index=False, encoding="utf-8-sig")
    result.topn_daily.to_csv(report_paths["topn_daily_path"], index=False, encoding="utf-8-sig")
    result.topn_summary.to_csv(report_paths["topn_summary_path"], index=False, encoding="utf-8-sig")

    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "sample_rows": int(len(result.samples)),
        "daily_ic_rows": int(len(result.daily_ic)),
        "topn_detail_rows": int(len(result.topn_detail)),
        **{name: str(path) for name, path in report_paths.items() if name != "json_path"},
    }
    report_paths["json_path"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_paths


def _summarize_return_groups(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    horizon: int,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    return_column = f"forward_return_{horizon}d"
    drawdown_column = f"max_drawdown_{horizon}d"
    upside_column = f"max_upside_{horizon}d"
    if return_column not in frame.columns:
        return []

    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        drawdowns = pd.to_numeric(group[drawdown_column], errors="coerce").dropna()
        upsides = pd.to_numeric(group[upside_column], errors="coerce").dropna()
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(metadata)
        row.update(
            {
                "horizon_days": horizon,
                "sample_count": int(len(returns)),
                "avg_return": _rounded_mean(returns),
                "median_return": _rounded_median(returns),
                "win_rate": _rounded_rate(returns > 0),
                "avg_max_upside": _rounded_mean(upsides),
                "avg_max_drawdown": _rounded_mean(drawdowns),
            }
        )
        rows.append(row)
    return rows


def _daily_quantiles(series: pd.Series, quantiles: int) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    mask = valid.notna()
    if int(mask.sum()) < quantiles:
        return result
    ranks = valid.loc[mask].rank(method="first", pct=True)
    result.loc[mask] = (ranks.mul(quantiles).apply(lambda value: min(quantiles, max(1, int(value + 0.999999))))).astype(int)
    return result


def _attach_empty_forward_fields(sample: dict[str, object], horizon: int) -> None:
    sample[f"exit_date_{horizon}d"] = None
    sample[f"exit_close_{horizon}d"] = None
    sample[f"forward_return_{horizon}d"] = None
    sample[f"max_upside_{horizon}d"] = None
    sample[f"max_drawdown_{horizon}d"] = None


def _sample_columns(horizons: tuple[int, ...]) -> list[str]:
    columns = [
        "trade_date",
        "symbol",
        "name",
        "close",
        "ma_rating",
        "osc_rating",
        "all_rating",
        "avg_ma_rating_5d",
        "avg_osc_rating_5d",
        "avg_all_rating_5d",
        "ma_rating_label",
        "osc_rating_label",
        "all_rating_label",
        "entry_date",
        "entry_open",
    ]
    for horizon in horizons:
        columns.extend(
            [
                f"exit_date_{horizon}d",
                f"exit_close_{horizon}d",
                f"forward_return_{horizon}d",
                f"max_upside_{horizon}d",
                f"max_drawdown_{horizon}d",
            ]
        )
    return columns


def _topn_columns(horizons: tuple[int, ...]) -> list[str]:
    return ["rank_field", "rank", *_sample_columns(horizons)]


def _group_summary_columns(*, group_columns: list[str]) -> list[str]:
    return [
        *group_columns,
        "sample_count",
        "avg_return",
        "median_return",
        "win_rate",
        "avg_max_upside",
        "avg_max_drawdown",
    ]


def _topn_summary_columns() -> list[str]:
    return [
        "rank_field",
        "top_count",
        "horizon_days",
        "trade_days",
        "stock_samples",
        "avg_stock_return",
        "median_stock_return",
        "stock_win_rate",
        "avg_daily_equal_weight_return",
        "median_daily_equal_weight_return",
        "daily_win_rate",
        "worst_daily_equal_weight_return",
        "avg_max_upside",
        "avg_max_drawdown",
    ]


def _top_counts(top_n: int) -> list[int]:
    defaults = [1, 3, 5, top_n]
    return sorted({item for item in defaults if item <= top_n})


def _safe_float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _rounded_mean(values: pd.Series) -> float | None:
    return round(float(values.mean()), 6) if not values.empty else None


def _rounded_median(values: pd.Series) -> float | None:
    return round(float(values.median()), 6) if not values.empty else None


def _rounded_min(values: pd.Series) -> float | None:
    return round(float(values.min()), 6) if not values.empty else None


def _rounded_rate(mask: pd.Series) -> float | None:
    return round(float(mask.mean()), 6) if len(mask) > 0 else None
