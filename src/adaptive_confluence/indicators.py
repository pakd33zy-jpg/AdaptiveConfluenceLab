from __future__ import annotations

import numpy as np
import pandas as pd


def _series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    return pd.Series(x, dtype=float)


def sma(s: pd.Series, length: int) -> pd.Series:
    return _series(s).rolling(length, min_periods=length).mean()


def ema(s: pd.Series, length: int) -> pd.Series:
    return _series(s).ewm(span=length, adjust=False, min_periods=length).mean()


def rma(s: pd.Series, length: int) -> pd.Series:
    return _series(s).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def wma(s: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return _series(s).rolling(length, min_periods=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(s: pd.Series, length: int) -> pd.Series:
    half = max(1, length // 2)
    root = max(1, int(np.sqrt(length)))
    return wma(2 * wma(s, half) - wma(s, length), root)


def vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    pv = _series(close) * _series(volume)
    denom = _series(volume).rolling(length, min_periods=length).sum()
    return pv.rolling(length, min_periods=length).sum() / denom.replace(0, np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    return pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return rma(true_range(df), length)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = _series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, length) / rma(loss, length).replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ma = ema(close, fast)
    slow_ma = ema(close, slow)
    line = fast_ma - slow_ma
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(df: pd.DataFrame, length: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    low_n = df["low"].rolling(length, min_periods=length).min()
    high_n = df["high"].rolling(length, min_periods=length).max()
    raw = 100 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    k = sma(raw, smooth_k)
    d = sma(k, smooth_d)
    return k, d


def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mean = sma(tp, length)
    md = tp.rolling(length, min_periods=length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - mean) / (0.015 * md.replace(0, np.nan))


def dmi_adx(df: pd.DataFrame, length: int = 14, smooth: int = 14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df)
    atr_like = rma(tr, length)
    plus_di = 100 * rma(plus_dm, length) / atr_like.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, length) / atr_like.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx, smooth)
    return plus_di, minus_di, adx


def bollinger(close: pd.Series, length: int = 20, std_mult: float = 2.0):
    basis = sma(close, length)
    dev = _series(close).rolling(length, min_periods=length).std(ddof=0) * std_mult
    upper = basis + dev
    lower = basis - dev
    width_pct = 100 * (upper - lower) / basis.replace(0, np.nan)
    pct_b = (_series(close) - lower) / (upper - lower).replace(0, np.nan)
    return basis, upper, lower, width_pct, pct_b


def session_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    if isinstance(df.index, pd.DatetimeIndex):
        key = pd.Series(df.index.tz_convert("America/New_York").date if df.index.tz is not None else df.index.date, index=df.index)
        cum_pv = pv.groupby(key).cumsum()
        cum_v = df["volume"].groupby(key).cumsum()
        return cum_pv / cum_v.replace(0, np.nan)
    return pv.cumsum() / df["volume"].cumsum().replace(0, np.nan)


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw = tp * df["volume"]
    pos = raw.where(tp > tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    neg = raw.where(tp < tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    ratio = pos / neg.replace(0, np.nan)
    out = 100 - 100 / (1 + ratio)
    out = out.where(neg != 0, 100.0)
    return out


def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / spread
    mfv = mfm.fillna(0) * df["volume"]
    return mfv.rolling(length, min_periods=length).sum() / df["volume"].rolling(length, min_periods=length).sum().replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def choppiness(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr_sum = true_range(df).rolling(length, min_periods=length).sum()
    high_n = df["high"].rolling(length, min_periods=length).max()
    low_n = df["low"].rolling(length, min_periods=length).min()
    denom = (high_n - low_n).replace(0, np.nan)
    return 100 * np.log10(tr_sum / denom) / np.log10(length)


def supertrend(df: pd.DataFrame, length: int = 10, factor: float = 3.0):
    a = atr(df, length)
    hl2 = (df["high"] + df["low"]) / 2
    basic_up = hl2 + factor * a
    basic_dn = hl2 - factor * a
    final_up = basic_up.copy()
    final_dn = basic_dn.copy()
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        prev = i - 1
        if pd.isna(a.iloc[i]):
            continue
        final_up.iloc[i] = basic_up.iloc[i] if (basic_up.iloc[i] < final_up.iloc[prev] or df["close"].iloc[prev] > final_up.iloc[prev]) else final_up.iloc[prev]
        final_dn.iloc[i] = basic_dn.iloc[i] if (basic_dn.iloc[i] > final_dn.iloc[prev] or df["close"].iloc[prev] < final_dn.iloc[prev]) else final_dn.iloc[prev]
        if pd.isna(st.iloc[prev]):
            st.iloc[i] = final_up.iloc[i]
            direction.iloc[i] = -1
        elif st.iloc[prev] == final_up.iloc[prev]:
            if df["close"].iloc[i] > final_up.iloc[i]:
                st.iloc[i] = final_dn.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i] = final_up.iloc[i]
                direction.iloc[i] = -1
        else:
            if df["close"].iloc[i] < final_dn.iloc[i]:
                st.iloc[i] = final_up.iloc[i]
                direction.iloc[i] = -1
            else:
                st.iloc[i] = final_dn.iloc[i]
                direction.iloc[i] = 1
    return st, direction


def williams_r(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_n = df["high"].rolling(length, min_periods=length).max()
    low_n = df["low"].rolling(length, min_periods=length).min()
    return -100 * (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan)


def ultimate_oscillator(df: pd.DataFrame, a: int = 7, b: int = 14, c: int = 28) -> pd.Series:
    pc = df["close"].shift(1)
    bp = df["close"] - pd.concat([df["low"], pc], axis=1).min(axis=1)
    tr = pd.concat([df["high"], pc], axis=1).max(axis=1) - pd.concat([df["low"], pc], axis=1).min(axis=1)

    def avg(n):
        return bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum().replace(0, np.nan)

    return 100 * (4 * avg(a) + 2 * avg(b) + avg(c)) / 7


def roc(close: pd.Series, length: int = 10) -> pd.Series:
    return 100 * (_series(close) / _series(close).shift(length) - 1)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = df.copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=required)
    return out
