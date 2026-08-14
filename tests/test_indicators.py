from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_confluence.indicators import atr, rsi, session_vwap
from adaptive_confluence.strategy import StrategyConfig, compute_features
from adaptive_confluence.backtest import run_backtest


def sample(n=400):
    idx = pd.date_range("2026-01-02 14:30", periods=n, freq="5min", tz="UTC")
    base = 100 + np.linspace(0, 8, n) + np.sin(np.arange(n) / 8) * 0.8
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = pd.Series(1_000_000 + (np.arange(n) % 20) * 25_000, index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_rsi_range():
    df = sample()
    x = rsi(df.close, 14).dropna()
    assert ((x >= 0) & (x <= 100)).all()


def test_atr_positive():
    df = sample()
    x = atr(df, 14).dropna()
    assert (x > 0).all()


def test_vwap_finite():
    df = sample()
    x = session_vwap(df).dropna()
    assert np.isfinite(x).all()


def test_feature_smoke():
    df = sample(600)
    feat = compute_features(df, StrategyConfig())
    for col in ["direction_score", "adx", "chop", "rel_vol", "signal", "setup"]:
        assert col in feat.columns
    assert feat.signal.isin([-1, 0, 1]).all()


def test_backtest_smoke():
    df = sample(800)
    result = run_backtest(df)
    assert "stats" in result
    assert result["stats"]["ending_equity"] > 0
