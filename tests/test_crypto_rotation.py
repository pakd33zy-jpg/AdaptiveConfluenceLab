from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_crypto_rotation.py"
spec = importlib.util.spec_from_file_location('crypto_v3', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def synthetic_frames(periods=700):
    dates = pd.date_range('2023-01-01', periods=periods, freq='D', tz='UTC')
    frames = {}
    params = {
        'BTC/USD': 0.0010,
        'ETH/USD': 0.0014,
        'SOL/USD': 0.0018,
        'AVAX/USD': 0.0012,
        'DOGE/USD': -0.0002,
    }
    for k, (symbol, drift) in enumerate(params.items()):
        wiggle = 0.004 * np.sin(np.arange(periods) / (9.0 + k))
        rets = drift + wiggle
        close = 100.0 * np.cumprod(1.0 + rets)
        open_ = np.r_[close[0], close[:-1]]
        volume = np.full(periods, 500000.0 + k * 10000)
        frames[symbol] = pd.DataFrame({'date': dates, 'open': open_, 'close': close, 'volume': volume})
    return frames


def config(**kw):
    data = dict(
        momentum_days=42,
        sma_days=100,
        rebalance_days=5,
        top_k=3,
        weighting='equal',
        regime_profile='balanced',
        vol_target_pct=65.0,
    )
    data.update(kw)
    return mod.CryptoConfig(**data)


def test_v3_excludes_cash_like_bases_and_has_no_fixed_liquidity_floor():
    assert 'USDT' in mod.EXCLUDED_BASES
    assert 'USDG' in mod.EXCLUDED_BASES
    assert mod.DEFAULT_MIN_MEDIAN_DOLLAR_VOLUME == 0.0


def test_grid_size_is_288():
    size = len(mod.MOMENTUM_GRID)*len(mod.SMA_GRID)*len(mod.REBALANCE_GRID)*len(mod.TOP_K_GRID)*len(mod.WEIGHTING_GRID)*len(mod.REGIME_PROFILE_GRID)*len(mod.VOL_TARGET_GRID)
    assert size == 288


def test_no_pre_evaluation_trading_and_exact_initial_anchor():
    matrices = mod.build_matrices(synthetic_frames())
    start = mod.WARMUP_DAYS
    result = mod.run_backtest(matrices, config(), trade_start_index=start)
    assert result.initial_equity == 100000.0
    assert result.evaluation_start == matrices.dates[start]
    assert pd.Timestamp(result.equity['date'].iloc[0]) == matrices.dates[start]
    assert len(result.equity) == len(matrices.dates) - start
    assert result.rebalance_count > 20


def test_full_segment_return_uses_100k_not_first_recorded_close():
    matrices = mod.build_matrices(synthetic_frames())
    start = mod.WARMUP_DAYS
    result = mod.run_backtest(matrices, config(), trade_start_index=start)
    metrics = mod.segment_metrics(result, matrices.dates[start], matrices.dates[-1])
    ending = float(result.equity['equity'].iloc[-1])
    expected = (ending / 100000.0 - 1.0) * 100.0
    assert abs(metrics['return_pct'] - expected) < 1e-9


def test_target_weights_respect_exposure_and_diversification():
    matrices = mod.build_matrices(synthetic_frames())
    weights, regime, vol, exposure = mod.target_weights(matrices, 400, config(weighting='equal'))
    assert np.isfinite(weights).all()
    assert np.sum(weights > 0) <= 3
    assert abs(weights.sum() - exposure) < 1e-8 or weights.sum() == 0
    assert 0.0 <= exposure <= 1.0
    assert regime in {'up','down','sideways'}


def test_defensive_profile_never_allocates_in_down_regime():
    matrices = mod.build_matrices(synthetic_frames())
    # Directly verify profile contract; the regime classifier itself is data-dependent.
    assert mod.REGIME_PROFILES['defensive']['down'] == 0.0
    assert mod.REGIME_PROFILES['defensive']['sideways'] < mod.REGIME_PROFILES['defensive']['up']


def test_volatility_scaling_never_increases_regime_exposure():
    matrices = mod.build_matrices(synthetic_frames())
    c = config(regime_profile='balanced', vol_target_pct=45.0)
    exposure, regime, vol = mod.exposure_for_signal(matrices, 500, c)
    assert exposure <= mod.REGIME_PROFILES['balanced'][regime] + 1e-12
    assert 0 <= exposure <= 1


def test_chronological_splits_do_not_overlap():
    dates = pd.date_range('2020-01-01', periods=1000, freq='D', tz='UTC')
    train, val, diag = mod.chronological_splits(dates)
    assert train[1] < val[0]
    assert val[1] < diag[0]
    assert train[0] == dates[0]
    assert diag[1] == dates[-1]


def test_backtest_equity_stays_finite_positive():
    matrices = mod.build_matrices(synthetic_frames())
    result = mod.run_backtest(matrices, config(), trade_start_index=mod.WARMUP_DAYS)
    assert np.isfinite(result.equity['equity']).all()
    assert (result.equity['equity'] > 0).all()
    assert not result.cycles.empty
