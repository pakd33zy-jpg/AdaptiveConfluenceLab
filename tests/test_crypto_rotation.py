from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_crypto_rotation.py"
spec = importlib.util.spec_from_file_location("crypto_rotation", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def synthetic_frames():
    dates = pd.date_range("2023-01-01", periods=520, freq="D", tz="UTC")
    frames = {}
    for symbol, drift in {
        "BTC/USD": 0.0010,
        "ETH/USD": 0.0015,
        "SOL/USD": 0.0020,
        "DOGE/USD": -0.0005,
    }.items():
        close = 100.0 * np.cumprod(np.full(len(dates), 1.0 + drift))
        open_ = np.r_[close[0], close[:-1]]
        volume = np.full(len(dates), 500000.0)
        frames[symbol] = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "close": close,
                "volume": volume,
            }
        )
    return frames


def test_target_weights_ranks_positive_trend_assets():
    matrices = mod.build_matrices(synthetic_frames())
    config = mod.CryptoConfig(
        momentum_days=21,
        sma_days=100,
        rebalance_days=5,
        top_k=2,
        weighting="equal",
        min_median_dollar_volume=1.0,
    )
    target = mod.target_weights(matrices, 300, config)
    assert set(target) == {"SOL/USD", "ETH/USD"}
    assert abs(sum(target.values()) - 1.0) < 1e-9


def test_rank_weighting_favors_first_rank():
    matrices = mod.build_matrices(synthetic_frames())
    config = mod.CryptoConfig(
        momentum_days=21,
        sma_days=100,
        rebalance_days=5,
        top_k=3,
        weighting="rank",
        min_median_dollar_volume=1.0,
    )
    target = mod.target_weights(matrices, 300, config)
    assert target["SOL/USD"] > target["ETH/USD"] > target["BTC/USD"]
    assert abs(sum(target.values()) - 1.0) < 1e-9


def test_negative_assets_can_leave_portfolio_in_cash():
    frames = synthetic_frames()
    for symbol, frame in frames.items():
        close = 100.0 * np.cumprod(np.full(len(frame), 0.998))
        frame["close"] = close
        frame["open"] = np.r_[close[0], close[:-1]]
    matrices = mod.build_matrices(frames)
    config = mod.CryptoConfig(
        momentum_days=21,
        sma_days=100,
        rebalance_days=5,
        top_k=2,
        weighting="equal",
        min_median_dollar_volume=1.0,
    )
    assert mod.target_weights(matrices, 300, config) == {}


def test_backtest_executes_after_prior_signal_and_stays_finite():
    matrices = mod.build_matrices(synthetic_frames())
    config = mod.CryptoConfig(
        momentum_days=21,
        sma_days=100,
        rebalance_days=5,
        top_k=2,
        weighting="equal",
        cost_bps=35.0,
        min_median_dollar_volume=1.0,
    )
    result = mod.run_backtest(matrices, config)
    assert len(result.equity) == len(matrices.dates)
    assert np.isfinite(result.equity["equity"]).all()
    assert (result.equity["equity"] > 0).all()
    assert result.rebalance_count > 20
    assert not result.cycles.empty


def test_chronological_splits_do_not_overlap():
    dates = pd.date_range("2020-01-01", periods=1000, freq="D", tz="UTC")
    train, val, hold = mod.chronological_splits(dates)
    assert train[1] < val[0]
    assert val[1] < hold[0]
    assert train[0] == dates[0]
    assert hold[1] == dates[-1]


def test_v2_excludes_usdg_and_uses_relative_liquidity():
    assert "USDG" in mod.EXCLUDED_BASES
    assert mod.DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME == 0.0
