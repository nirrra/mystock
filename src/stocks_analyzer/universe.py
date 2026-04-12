from __future__ import annotations

import pandas as pd


MAIN_BOARD_PREFIXES = {
    "sh": ("600", "601", "603", "605"),
    "sz": ("000", "001", "002", "003"),
}


def _classify_market(symbol: str) -> tuple[str | None, str | None]:
    for exchange, prefixes in MAIN_BOARD_PREFIXES.items():
        if symbol.startswith(prefixes):
            return exchange, "main_board"
    if symbol.startswith(("300", "301")):
        return "sz", "gem"
    if symbol.startswith(("688", "689")):
        return "sh", "star"
    if symbol.startswith(("8", "4", "92")):
        return "bj", "beijing"
    return None, None


def build_main_board_universe(instruments: pd.DataFrame, exclude_st: bool) -> pd.DataFrame:
    dataframe = instruments.copy()
    if dataframe.empty:
        dataframe = dataframe.reindex(columns=list(dataframe.columns) + ["exchange", "board", "is_st", "is_suspended"])
        return dataframe.reset_index(drop=True)

    dataframe["symbol"] = dataframe["symbol"].astype(str).str.zfill(6)
    market_info = pd.DataFrame(
        dataframe["symbol"].map(_classify_market).tolist(),
        columns=["exchange", "board"],
        index=dataframe.index,
    )
    dataframe[["exchange", "board"]] = market_info
    dataframe["is_st"] = dataframe["name"].astype(str).str.upper().str.contains("ST", regex=False)
    trade_status = dataframe["trade_status"] if "trade_status" in dataframe.columns else None
    if trade_status is not None:
        dataframe["is_suspended"] = trade_status.astype(str) != "1"
    else:
        dataframe["is_suspended"] = False
    dataframe = dataframe[dataframe["board"] == "main_board"].copy()
    if exclude_st:
        dataframe = dataframe[~dataframe["is_st"]].copy()
    return dataframe.sort_values(["exchange", "symbol"]).reset_index(drop=True)
