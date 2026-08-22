from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / "scripts" / "paper_forward_v32.py"
spec = importlib.util.spec_from_file_location("paper_forward_v32", P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def frames(periods=700):
    dates = pd.bdate_range("2024-01-02", periods=periods, tz="UTC")
    syms = [
        "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","SMH","IBB","KRE","XHB",
        "VNQ","RSP","MTUM","QUAL","USMV","TLT","IEF","LQD","GLD","EFA","EEM","DBC"
    ]
    out = {}
    t = np.arange(periods)
    for k, s in enumerate(syms):
        phase = k * .19
        drift = np.where(t < 300, .00045, np.where(t < 430, -.00025, .00040))
        rr = drift + .0015*np.sin(t/27 + phase) + .0008*np.sin(t/8 + phase)
        close = 100*np.exp(np.cumsum(rr))
        open_ = close*np.exp(.0005*np.sin(t/6 + phase))
        vol = np.full(periods, 2_000_000 + k*20_000, dtype=float)
        out[s] = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close)*1.002,
            "low": np.minimum(open_, close)*.998,
            "close": close,
            "volume": vol,
        })
    return out


def test_no_order_endpoints_or_submission():
    x = P.read_text().lower()
    assert "submit_order" not in x
    assert "/v2/orders" not in x
    assert "paper-api.alpaca.markets" not in x


def test_frozen_v26_contract():
    assert m.V26_UNIVERSE == (
        "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","GLD","TLT"
    )
    assert m.V26_MOMENTUM_DAYS == 63
    assert m.V26_SMA_DAYS == 150
    assert m.V26_REBALANCE_EVERY == 5
    assert m.V26_TOP_WEIGHTS == (0.70, 0.30)


def test_frozen_v32_contract():
    c = m.V32_CONFIG
    assert c.momentum_days == 126
    assert c.sma_days == 200
    assert c.rebalance_days == 20
    assert c.top_k == 4
    assert c.weighting == "equal"
    assert c.risk_profile == "drawdown_guard"
    assert c.vol_target_pct == 14.0
    assert c.cost_bps == 10.0


def test_v26_shadow_runs_and_is_long_only():
    mat = m.base.build_matrices(frames())
    start_i = 360
    r = m.run_v26_shadow(mat, start_i, cost_bps=10.0)
    assert len(r.equity) > 0
    assert np.isfinite(r.equity["equity"]).all()
    assert (r.equity["equity"] > 0).all()
    assert (r.equity["positions"] >= 0).all()
    assert (r.equity["positions"] <= 2).all()


def test_v26_targets_bounded():
    mat = m.base.build_matrices(frames())
    for i in (200, 350, 500, 650):
        w, _ = m._v26_target_weights(mat, i)
        assert np.all(w >= -1e-12)
        assert w.sum() <= 1.000001
        assert np.count_nonzero(w > 1e-12) <= 2


def test_v32_shadow_runs():
    mat = m.base.build_matrices(frames())
    r = m.v32.run_backtest_v32(mat, m.V32_CONFIG, trade_start_index=360)
    assert len(r.equity) > 0
    assert np.isfinite(r.equity["equity"]).all()
    assert (r.equity["equity"] > 0).all()
