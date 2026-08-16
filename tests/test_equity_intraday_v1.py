from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_equity_intraday_v1.py"
spec = importlib.util.spec_from_file_location("intraday_v1", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

def synthetic_history():
    times = pd.date_range("2026-01-05 14:30:00+00:00", periods=80, freq="5min")
    session = ["2026-01-05"] * len(times)
    close = [100 + i * 0.08 for i in range(len(times))]
    return {
        "SPY": pd.DataFrame({
            "time": times,
            "open": [x - 0.02 for x in close],
            "high": [x + 0.10 for x in close],
            "low": [x - 0.10 for x in close],
            "close": close,
            "volume": [1000 + i * 5 for i in range(len(times))],
            "session": session,
        })
    }

def test_grid_is_modest_and_nonempty():
    grid = mod.config_grid()
    assert 1 <= len(grid) <= 256

def test_research_file_contains_no_order_endpoint():
    text = SCRIPT.read_text(encoding="utf-8").lower()
    assert "/v2/orders" not in text
    assert "submit_order" not in text
    assert "placeorder" not in text

def test_compound_sizing_uses_current_equity():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "sizing_equity = equity_now if compound else start_capital" in text

def test_no_leverage_position_cap():
    cfg = mod.Config(8, 24, 3, 0.05, 1.0, 1.0, 1.25, 6)
    assert 0 < cfg.max_position_fraction <= 1.0

def test_next_bar_signal_construction():
    cfg = mod.Config(8, 24, 3, 0.05, 1.0, 1.0, 1.25, 6)
    events = mod.prepare_events(synthetic_history(), cfg)
    assert events
    # Every event is built from row i but its entry signal comes from row i-1.
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"entry_signal": bool(prev["signal_long"])' in text

def test_summary_turns_100_into_110_for_10pct_gain():
    eq = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        "equity": [100.0, 110.0],
    })
    trades = pd.DataFrame({"pnl": [10.0]})
    m = mod.summarize(100.0, 110.0, eq, trades)
    assert m["ending_capital"] == 110.0
    assert abs(m["return_pct"] - 10.0) < 1e-9

def test_costs_are_nonzero():
    cfg = mod.config_grid()[0]
    assert cfg.fee_bps_per_side > 0
    assert cfg.slippage_bps_per_side > 0
