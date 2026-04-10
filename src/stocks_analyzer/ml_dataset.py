from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .features import build_feature_frame, numeric_feature_columns
from .labels import add_forward_labels
from .models import AppConfig
from .storage import Storage


@dataclass(slots=True)
class DatasetSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    label_column: str


def build_probability_dataset(
    storage: Storage,
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    universe = storage.load_universe()
    records: list[pd.DataFrame] = []

    symbols = universe["symbol"].tolist()
    if limit is not None:
        symbols = symbols[:limit]

    for symbol in symbols:
        try:
            bars = storage.load_daily_bars(symbol)
        except FileNotFoundError:
            continue

        frame = build_feature_frame(bars)
        frame = add_forward_labels(
            frame,
            horizon_days=config.probability.horizon_days,
            min_future_return=config.probability.min_future_return,
            max_future_drawdown=config.probability.max_future_drawdown,
        )

        instrument = universe[universe["symbol"] == symbol].iloc[0]
        frame["symbol"] = symbol
        frame["name"] = instrument["name"]
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])

        if start_date is not None:
            frame = frame[frame["trade_date"].dt.date >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"].dt.date <= end_date]

        frame = frame[
            frame["amount_ma_20"].notna() & (frame["amount_ma_20"] >= config.universe.min_avg_amount_20d)
        ].copy()
        if len(frame) < config.probability.min_history_days:
            continue

        records.append(frame)

    if not records:
        return pd.DataFrame()

    dataset = pd.concat(records, ignore_index=True)
    dataset = dataset.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return dataset


def split_probability_dataset(
    dataset: pd.DataFrame,
    train_end: date,
    valid_end: date,
    test_end: date,
    label_column: str = "label_stable_up",
) -> DatasetSplit:
    if dataset.empty:
        raise ValueError("Probability dataset is empty")

    frame = dataset.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame[frame[label_column].notna()].copy()

    train_mask = frame["trade_date"].dt.date <= train_end
    valid_mask = (frame["trade_date"].dt.date > train_end) & (frame["trade_date"].dt.date <= valid_end)
    test_mask = (frame["trade_date"].dt.date > valid_end) & (frame["trade_date"].dt.date <= test_end)

    train = frame.loc[train_mask].reset_index(drop=True)
    valid = frame.loc[valid_mask].reset_index(drop=True)
    test = frame.loc[test_mask].reset_index(drop=True)

    if train.empty or valid.empty or test.empty:
        raise ValueError("Time split produced an empty train, valid, or test partition")

    features = numeric_feature_columns(frame)
    if label_column in features:
        features.remove(label_column)
    features = [column for column in features if train[column].notna().any()]

    return DatasetSplit(
        train=train,
        valid=valid,
        test=test,
        feature_columns=features,
        label_column=label_column,
    )


def infer_split_dates(dataset: pd.DataFrame) -> tuple[date, date, date]:
    if dataset.empty:
        raise ValueError("Probability dataset is empty")

    dates = sorted(pd.to_datetime(dataset["trade_date"]).dt.date.unique())
    if len(dates) < 10:
        raise ValueError("Need at least 10 distinct trade dates to infer dataset splits")

    train_index = max(int(len(dates) * 0.6) - 1, 0)
    valid_index = max(int(len(dates) * 0.8) - 1, train_index + 1)
    test_index = len(dates) - 1
    return dates[train_index], dates[valid_index], dates[test_index]
