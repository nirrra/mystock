from __future__ import annotations

import pandas as pd


def add_forward_labels(
    dataframe: pd.DataFrame,
    horizon_days: int,
    min_future_return: float,
    max_future_drawdown: float,
) -> pd.DataFrame:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)
    close = df["close"].astype(float)
    low = df["low"].astype(float)
    high = df["high"].astype(float)

    future_close = close.shift(-horizon_days)
    df[f"future_{horizon_days}d_return"] = future_close.div(close) - 1

    future_min_low = pd.concat([low.shift(-step) for step in range(1, horizon_days + 1)], axis=1).min(axis=1)
    future_max_high = pd.concat([high.shift(-step) for step in range(1, horizon_days + 1)], axis=1).max(axis=1)
    future_min_close = pd.concat([close.shift(-step) for step in range(1, horizon_days + 1)], axis=1).min(axis=1)

    df[f"future_{horizon_days}d_max_drawdown"] = 1 - future_min_low.div(close)
    df[f"future_{horizon_days}d_max_upside"] = future_max_high.div(close) - 1
    df[f"future_{horizon_days}d_min_return"] = future_min_close.div(close) - 1

    label_column = f"label_stable_up_{horizon_days}d"
    df[label_column] = (
        (df[f"future_{horizon_days}d_return"] > min_future_return)
        & (df[f"future_{horizon_days}d_max_drawdown"] <= max_future_drawdown)
    ).astype("float")

    horizon_ready = close.shift(-horizon_days).notna()
    df.loc[~horizon_ready, label_column] = pd.NA

    rename_map = {
        f"future_{horizon_days}d_return": "future_20d_return" if horizon_days == 20 else f"future_{horizon_days}d_return",
        f"future_{horizon_days}d_max_drawdown": (
            "future_20d_max_drawdown" if horizon_days == 20 else f"future_{horizon_days}d_max_drawdown"
        ),
        f"future_{horizon_days}d_max_upside": "future_20d_max_upside" if horizon_days == 20 else f"future_{horizon_days}d_max_upside",
        f"future_{horizon_days}d_min_return": "future_20d_min_return" if horizon_days == 20 else f"future_{horizon_days}d_min_return",
        label_column: "label_stable_up" if horizon_days == 20 else label_column,
    }
    return df.rename(columns=rename_map)
