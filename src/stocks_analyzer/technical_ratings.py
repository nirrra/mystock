from __future__ import annotations

import math

import numpy as np
import pandas as pd


RATING_STRONG_BOUND = 0.5
RATING_WEAK_BOUND = 0.1

MOVING_AVERAGE_COLUMNS = (
    "sma_10",
    "sma_20",
    "sma_30",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_10",
    "ema_20",
    "ema_30",
    "ema_50",
    "ema_100",
    "ema_200",
    "hma_9",
    "vwma_20",
)

MA_SIGNAL_COLUMNS = tuple(f"{name}_signal" for name in MOVING_AVERAGE_COLUMNS) + ("ichimoku_signal",)
OSCILLATOR_SIGNAL_COLUMNS = (
    "rsi_signal",
    "stoch_signal",
    "cci_signal",
    "adx_signal",
    "ao_signal",
    "momentum_signal",
    "macd_signal",
    "stoch_rsi_signal",
    "williams_r_signal",
    "bull_bear_power_signal",
    "uo_signal",
)


def add_technical_ratings(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy().sort_values("trade_date").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    typical_price = (high + low + close) / 3
    median_price = (high + low) / 2

    for period in (10, 20, 30, 50, 100, 200):
        df[f"sma_{period}"] = close.rolling(period).mean()
        df[f"ema_{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()

    df["hma_9"] = _hma(close, 9)
    df["vwma_20"] = _vwma(close, volume, 20)

    df["ichimoku_conversion"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df["ichimoku_base"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["ichimoku_span_a"] = (df["ichimoku_conversion"] + df["ichimoku_base"]) / 2
    df["ichimoku_span_b"] = (high.rolling(52).max() + low.rolling(52).min()) / 2

    df["rsi_14"] = _rsi(close, 14)
    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, 14, 3, 3)
    df["cci_20"] = _cci(high, low, close, 20)
    df["plus_di_14"], df["minus_di_14"], df["adx_14"] = _adx(high, low, close, 14, 14)
    df["ao"] = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    df["momentum_10"] = close - close.shift(10)
    df["macd"], df["macd_signal_line"], df["macd_hist"] = _macd(close, 12, 26, 9)
    df["stoch_rsi_k"], df["stoch_rsi_d"] = _stochastic_rsi(close, 14, 14, 3, 3)
    df["williams_r_14"] = _williams_r(high, low, close, 14)
    # tradingview.md does not spell out the uptrend/downtrend filter for Stoch RSI
    # and Bull Bear Power, so we use EMA50 as a stable explicit trend proxy.
    df["trend_ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    df["bull_power_50"] = high - df["trend_ema_50"]
    df["bear_power_50"] = low - df["trend_ema_50"]
    df["uo_7_14_28"] = _ultimate_oscillator(high, low, close, 7, 14, 28)

    for column in MOVING_AVERAGE_COLUMNS:
        df[f"{column}_signal"] = _price_vs_indicator_signal(close, df[column])

    df["ichimoku_signal"] = _ichimoku_signal(df)
    df["rsi_signal"] = _rsi_signal(df["rsi_14"])
    df["stoch_signal"] = _stochastic_signal(df["stoch_k"], df["stoch_d"])
    df["cci_signal"] = _cci_signal(df["cci_20"])
    df["adx_signal"] = _adx_signal(df["plus_di_14"], df["minus_di_14"], df["adx_14"])
    df["ao_signal"] = _ao_signal(df["ao"])
    df["momentum_signal"] = _momentum_signal(df["momentum_10"])
    df["macd_signal"] = _macd_rating_signal(df["macd"], df["macd_signal_line"])
    df["stoch_rsi_signal"] = _stochastic_rsi_signal(df["trend_ema_50"], close, df["stoch_rsi_k"], df["stoch_rsi_d"])
    df["williams_r_signal"] = _williams_signal(df["williams_r_14"])
    df["bull_bear_power_signal"] = _bull_bear_power_signal(
        df["trend_ema_50"],
        close,
        df["bull_power_50"],
        df["bear_power_50"],
    )
    df["uo_signal"] = _uo_signal(df["uo_7_14_28"])

    df["ma_rating"] = df.loc[:, MA_SIGNAL_COLUMNS].mean(axis=1, skipna=True)
    df["osc_rating"] = df.loc[:, OSCILLATOR_SIGNAL_COLUMNS].mean(axis=1, skipna=True)
    df["all_rating"] = df.loc[:, ["ma_rating", "osc_rating"]].mean(axis=1, skipna=True)

    df["ma_rating_label"] = df["ma_rating"].map(rating_status)
    df["osc_rating_label"] = df["osc_rating"].map(rating_status)
    df["all_rating_label"] = df["all_rating"].map(rating_status)

    return df


def rating_status(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    if value < -RATING_STRONG_BOUND:
        return "strong_sell"
    if value < -RATING_WEAK_BOUND:
        return "sell"
    if value <= RATING_WEAK_BOUND:
        return "neutral"
    if value <= RATING_STRONG_BOUND:
        return "buy"
    return "strong_buy"


def _price_vs_indicator_signal(price: pd.Series, indicator: pd.Series) -> pd.Series:
    signal = pd.Series(np.nan, index=price.index, dtype=float)
    mask = indicator.notna()
    signal.loc[mask & (price > indicator)] = 1.0
    signal.loc[mask & (price < indicator)] = -1.0
    signal.loc[mask & (price == indicator)] = 0.0
    return signal


def _ichimoku_signal(df: pd.DataFrame) -> pd.Series:
    signal = pd.Series(np.nan, index=df.index, dtype=float)
    required = df[["ichimoku_span_a", "ichimoku_span_b", "ichimoku_base", "ichimoku_conversion", "close"]].notna().all(axis=1)

    buy = (
        required
        & (df["ichimoku_span_a"] > df["ichimoku_span_b"])
        & (df["ichimoku_base"] > df["ichimoku_span_a"])
        & (df["ichimoku_conversion"] > df["ichimoku_base"])
        & (df["close"] > df["ichimoku_conversion"])
    )
    sell = (
        required
        & (df["ichimoku_span_a"] < df["ichimoku_span_b"])
        & (df["ichimoku_base"] < df["ichimoku_span_a"])
        & (df["ichimoku_conversion"] < df["ichimoku_base"])
        & (df["close"] < df["ichimoku_span_b"])
    )
    signal.loc[required] = 0.0
    signal.loc[buy] = 1.0
    signal.loc[sell] = -1.0
    return signal


def _rsi_signal(rsi: pd.Series) -> pd.Series:
    prev = rsi.shift(1)
    signal = pd.Series(np.nan, index=rsi.index, dtype=float)
    valid = rsi.notna() & prev.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (rsi < 30) & (rsi > prev)] = 1.0
    signal.loc[valid & (rsi > 70) & (rsi < prev)] = -1.0
    return signal


def _stochastic_signal(k: pd.Series, d: pd.Series) -> pd.Series:
    signal = pd.Series(np.nan, index=k.index, dtype=float)
    valid = k.notna() & d.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (k < 20) & (d < 20) & (k > d)] = 1.0
    signal.loc[valid & (k > 80) & (d > 80) & (k < d)] = -1.0
    return signal


def _cci_signal(cci: pd.Series) -> pd.Series:
    prev = cci.shift(1)
    signal = pd.Series(np.nan, index=cci.index, dtype=float)
    valid = cci.notna() & prev.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (cci < -100) & (cci > prev)] = 1.0
    signal.loc[valid & (cci > 100) & (cci < prev)] = -1.0
    return signal


def _adx_signal(plus_di: pd.Series, minus_di: pd.Series, adx: pd.Series) -> pd.Series:
    prev = adx.shift(1)
    signal = pd.Series(np.nan, index=adx.index, dtype=float)
    valid = plus_di.notna() & minus_di.notna() & adx.notna() & prev.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (plus_di > minus_di) & (adx > 20) & (adx > prev)] = 1.0
    signal.loc[valid & (plus_di < minus_di) & (adx > 20) & (adx < prev)] = -1.0
    return signal


def _ao_signal(ao: pd.Series) -> pd.Series:
    prev = ao.shift(1)
    prev2 = ao.shift(2)
    signal = pd.Series(np.nan, index=ao.index, dtype=float)
    valid = ao.notna() & prev.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (ao > 0) & (prev <= 0)] = 1.0
    signal.loc[valid & (ao < 0) & (prev >= 0)] = -1.0

    valid_turn = valid & prev2.notna()
    signal.loc[valid_turn & (ao > 0) & (prev > 0) & (ao > prev) & (prev < prev2)] = 1.0
    signal.loc[valid_turn & (ao < 0) & (prev < 0) & (ao < prev) & (prev > prev2)] = -1.0
    return signal


def _momentum_signal(momentum: pd.Series) -> pd.Series:
    prev = momentum.shift(1)
    signal = pd.Series(np.nan, index=momentum.index, dtype=float)
    valid = momentum.notna() & prev.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (momentum > prev)] = 1.0
    signal.loc[valid & (momentum < prev)] = -1.0
    return signal


def _macd_rating_signal(macd: pd.Series, signal_line: pd.Series) -> pd.Series:
    signal = pd.Series(np.nan, index=macd.index, dtype=float)
    valid = macd.notna() & signal_line.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (macd > signal_line)] = 1.0
    signal.loc[valid & (macd < signal_line)] = -1.0
    return signal


def _stochastic_rsi_signal(trend_ema: pd.Series, close: pd.Series, k: pd.Series, d: pd.Series) -> pd.Series:
    signal = pd.Series(np.nan, index=close.index, dtype=float)
    valid = trend_ema.notna() & close.notna() & k.notna() & d.notna()
    signal.loc[valid] = 0.0
    uptrend = close > trend_ema
    downtrend = close < trend_ema
    signal.loc[valid & downtrend & (k < 20) & (d < 20) & (k > d)] = 1.0
    signal.loc[valid & uptrend & (k > 80) & (d > 80) & (k < d)] = -1.0
    return signal


def _williams_signal(williams_r: pd.Series) -> pd.Series:
    prev = williams_r.shift(1)
    signal = pd.Series(np.nan, index=williams_r.index, dtype=float)
    valid = williams_r.notna() & prev.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (williams_r < -80) & (williams_r > prev)] = 1.0
    signal.loc[valid & (williams_r > -20) & (williams_r < prev)] = -1.0
    return signal


def _bull_bear_power_signal(
    trend_ema: pd.Series,
    close: pd.Series,
    bull_power: pd.Series,
    bear_power: pd.Series,
) -> pd.Series:
    signal = pd.Series(np.nan, index=close.index, dtype=float)
    valid = trend_ema.notna() & close.notna() & bull_power.notna() & bear_power.notna()
    signal.loc[valid] = 0.0
    uptrend = close > trend_ema
    downtrend = close < trend_ema
    signal.loc[valid & uptrend & (bear_power < 0) & (bear_power > bear_power.shift(1))] = 1.0
    signal.loc[valid & downtrend & (bull_power > 0) & (bull_power < bull_power.shift(1))] = -1.0
    return signal


def _uo_signal(uo: pd.Series) -> pd.Series:
    signal = pd.Series(np.nan, index=uo.index, dtype=float)
    valid = uo.notna()
    signal.loc[valid] = 0.0
    signal.loc[valid & (uo > 70)] = 1.0
    signal.loc[valid & (uo < 30)] = -1.0
    return signal


def _vwma(price: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    numerator = (price * volume).rolling(period).sum()
    denominator = volume.rolling(period).sum()
    return numerator.div(denominator.replace(0.0, np.nan))


def _wma(values: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return values.rolling(period).apply(lambda array: float(np.dot(array, weights) / weights.sum()), raw=True)


def _hma(values: pd.Series, period: int) -> pd.Series:
    half_period = max(period // 2, 1)
    sqrt_period = max(int(math.sqrt(period)), 1)
    wma_half = _wma(values, half_period)
    wma_full = _wma(values, period)
    hull_input = 2 * wma_half - wma_full
    return _wma(hull_input, sqrt_period)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain.div(avg_loss.replace(0.0, np.nan))
    return 100 - (100 / (1 + rs))


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int,
    smooth_period: int,
    d_period: int,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    span = (highest_high - lowest_low).replace(0.0, np.nan)
    raw_k = 100 * (close - lowest_low).div(span)
    k = raw_k.rolling(smooth_period).mean()
    d = k.rolling(d_period).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    typical_price = (high + low + close) / 3
    moving_average = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda array: float(np.mean(np.abs(array - np.mean(array)))), raw=True)
    return (typical_price - moving_average).div(0.015 * mad.replace(0.0, np.nan))


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    di_period: int,
    adx_period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / di_period, adjust=False, min_periods=di_period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(alpha=1 / di_period, adjust=False, min_periods=di_period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(alpha=1 / di_period, adjust=False, min_periods=di_period).mean()

    plus_di = 100 * plus_dm_smooth.div(atr.replace(0.0, np.nan))
    minus_di = 100 * minus_dm_smooth.div(atr.replace(0.0, np.nan))
    dx = 100 * (plus_di - minus_di).abs().div((plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()
    return plus_di, minus_di, adx


def _macd(close: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = macd - signal
    return macd, signal, hist


def _stochastic_rsi(
    close: pd.Series,
    rsi_period: int,
    stoch_period: int,
    smooth_period: int,
    d_period: int,
) -> tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, rsi_period)
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    raw = 100 * (rsi - rsi_low).div((rsi_high - rsi_low).replace(0.0, np.nan))
    k = raw.rolling(smooth_period).mean()
    d = k.rolling(d_period).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close).div((highest_high - lowest_low).replace(0.0, np.nan))


def _ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, fast: int, middle: int, slow: int) -> pd.Series:
    prev_close = close.shift(1)
    buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    true_range = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    avg_fast = buying_pressure.rolling(fast).sum().div(true_range.rolling(fast).sum().replace(0.0, np.nan))
    avg_middle = buying_pressure.rolling(middle).sum().div(true_range.rolling(middle).sum().replace(0.0, np.nan))
    avg_slow = buying_pressure.rolling(slow).sum().div(true_range.rolling(slow).sum().replace(0.0, np.nan))
    return 100 * ((4 * avg_fast) + (2 * avg_middle) + avg_slow) / 7
