from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_equity_rotation_v27.py"
spec = importlib.util.spec_from_file_location("equity_v27", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def synthetic_frames(periods=1800):
    dates = pd.bdate_range("2018-01-02", periods=periods, tz="UTC")
    symbols = [
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
        "SMH", "IBB", "KRE", "XHB", "VNQ", "EFA", "EEM",
        "TLT", "IEF", "LQD", "GLD", "USO",
    ]
    frames = {}
    for k, symbol in enumerate(symbols):
        drift = 0.00035 + (k % 6) * 0.00005
        if symbol in {"TLT", "IEF", "LQD", "GLD"}:
            drift = 0.00020
        wiggle = 0.003 * np.sin(np.arange(periods) / (17.0 + k))
        ret = drift + wiggle
        close = 100.0 * np.cumprod(1.0 + ret)
        open_ = np.r_[close[0], close[:-1]]
        volume = np.full(periods, 2_000_000.0 + k * 10000)
        frames[symbol] = pd.DataFrame(
            {"date": dates, "open": open_, "close": close, "volume": volume}
        )
    return frames


def cfg(**kw):
    data = dict(
        momentum_days=63, sma_days=150, rebalance_days=10,
        top_k=4, weighting="equal", risk_profile="strict",
        vol_target_pct=16.0,
    )
    data.update(kw)
    return mod.EquityConfig(**data)


def test_universe_is_expanded_and_diversified():
    assert len(mod.UNIVERSE) == 35
    assert "SPY" in mod.UNIVERSE and "GLD" in mod.UNIVERSE and "TLT" in mod.UNIVERSE
    assert len(set(mod.GROUPS.values())) >= 10


def test_grid_size_is_288():
    size = (
        len(mod.MOMENTUM_GRID) * len(mod.SMA_GRID) * len(mod.REBALANCE_GRID)
        * len(mod.TOP_K_GRID) * len(mod.WEIGHTING_GRID)
        * len(mod.RISK_PROFILE_GRID) * len(mod.VOL_TARGET_GRID)
    )
    assert size == 288


def test_exact_100k_evaluation_anchor():
    m = mod.build_matrices(synthetic_frames())
    start = mod.WARMUP_DAYS + 5
    r = mod.run_backtest(m, cfg(), trade_start_index=start)
    assert r.initial_equity == 100000.0
    assert r.evaluation_start == m.dates[start]
    assert pd.Timestamp(r.equity["date"].iloc[0]) == m.dates[start]


def test_signal_is_prior_bar_and_regime_flip_executes_next_open():
    m = mod.build_matrices(synthetic_frames())
    m.spy_regime = m.spy_regime.copy()
    m.spy_realized_vol_pct = m.spy_realized_vol_pct.copy()
    start = mod.WARMUP_DAYS + 20
    m.spy_regime[:] = "up"
    m.spy_realized_vol_pct[:] = 10.0
    # Force raw risk-off on signal date start+17. It must act on next bar start+18.
    m.spy_regime[start + 17] = "down"
    r = mod.run_backtest(m, cfg(rebalance_days=20), trade_start_index=start)
    target_date = m.dates[start + 18]
    row = r.equity[r.equity["date"] == target_date].iloc[0]
    assert bool(row["rebalanced"])
    assert row["rebalance_reason"] == "regime_flip"
    assert row["mode"] == "cash"


def test_strict_profile_is_cash_outside_confirmed_up():
    m = mod.build_matrices(synthetic_frames())
    m.spy_regime = m.spy_regime.copy()
    m.spy_realized_vol_pct = m.spy_realized_vol_pct.copy()
    m.spy_regime[:] = "neutral"
    m.spy_realized_vol_pct[:] = 10.0
    exposure, regime, mode, vol = mod.exposure_and_mode(m, 500, cfg(risk_profile="strict"))
    assert exposure == 0.0
    assert mode == "cash"


def test_defensive_profile_uses_only_defensive_symbols():
    m = mod.build_matrices(synthetic_frames())
    m.spy_regime = m.spy_regime.copy()
    m.spy_realized_vol_pct = m.spy_realized_vol_pct.copy()
    m.spy_regime[:] = "neutral"
    m.spy_realized_vol_pct[:] = 10.0
    w, regime, mode, vol, exposure = mod.target_weights(
        m, 700, cfg(risk_profile="defensive")
    )
    chosen = {m.symbols[j] for j in np.flatnonzero(w > 0)}
    assert chosen.issubset(mod.DEFENSIVE_SYMBOLS)
    assert 0 <= exposure <= 0.60 + 1e-12


def test_category_cap_prevents_duplicate_group_selection():
    m = mod.build_matrices(synthetic_frames())
    mom = np.arange(len(m.symbols), dtype=float)
    candidates = list(range(len(m.symbols)))
    chosen = mod._group_diversified_selection(m.symbols, candidates, mom, 10)
    groups = [mod.GROUPS.get(m.symbols[j], m.symbols[j]) for j in chosen]
    assert len(groups) == len(set(groups))


def test_vol_scaling_never_increases_base_exposure():
    m = mod.build_matrices(synthetic_frames())
    m.spy_regime = m.spy_regime.copy()
    m.spy_realized_vol_pct = m.spy_realized_vol_pct.copy()
    m.spy_regime[:] = "up"
    m.spy_realized_vol_pct[:] = 40.0
    exposure, regime, mode, vol = mod.exposure_and_mode(m, 700, cfg(vol_target_pct=12.0))
    assert 0.0 <= exposure <= 1.0
    assert exposure <= 12.0 / 40.0 + 1e-12


def test_segment_return_uses_initial_anchor():
    m = mod.build_matrices(synthetic_frames())
    start = mod.WARMUP_DAYS + 5
    r = mod.run_backtest(m, cfg(), trade_start_index=start)
    met = mod.segment_metrics(r, m.dates[start], m.dates[-1])
    ending = float(r.equity["equity"].iloc[-1])
    expected = (ending / 100000.0 - 1.0) * 100.0
    assert abs(met["return_pct"] - expected) < 1e-9


def test_development_split_is_four_folds_plus_final_diagnostic():
    dates = pd.bdate_range("2019-01-01", periods=1500, tz="UTC")
    folds, diagnostic = mod.development_splits(dates)
    assert len(folds) == 4
    assert folds[0][0] == dates[0]
    assert folds[-1][1] < diagnostic[0]
    assert diagnostic[1] == dates[-1]


def test_equity_curve_is_finite_and_positive():
    m = mod.build_matrices(synthetic_frames())
    r = mod.run_backtest(m, cfg(), trade_start_index=mod.WARMUP_DAYS + 5)
    assert np.isfinite(r.equity["equity"]).all()
    assert (r.equity["equity"] > 0).all()
    assert r.rebalance_count > 20


def test_research_has_no_order_submission_function():
    assert not hasattr(mod.AlpacaEquityData, "submit_order")
