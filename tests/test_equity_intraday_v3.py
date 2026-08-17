import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_equity_intraday_v3.py"
spec = importlib.util.spec_from_file_location("intraday_v3", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

def sample_history():
    times = pd.date_range("2026-01-05 14:30:00+00:00", periods=40, freq="5min")
    hist = {}
    for i, symbol in enumerate(["SPY", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        close = np.linspace(20 + i, 22 + i, len(times))
        volume = np.linspace(500000 + i*10000, 900000 + i*10000, len(times))
        hist[symbol] = pd.DataFrame({
            "time": times,
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": volume,
            "session": ["2026-01-05"] * len(times),
        })
    return hist

def test_no_order_submission():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "submit_order" not in source
    assert "/v2/orders" not in source

def test_starting_capital_is_100():
    assert mod.STARTING_CAPITAL == 100.0

def test_dynamic_leader_count():
    assert mod.DYNAMIC_LEADER_COUNT == 5

def test_dynamic_leader_map_builds():
    leaders = mod.build_dynamic_leader_map(sample_history())
    assert leaders
    assert all(len(v) <= 5 for v in leaders.values())

def test_v3_grid_more_opportunities():
    grid = mod.config_grid()
    assert len(grid) > 64
    assert min(c.min_momentum_pct for c in grid) == 0.10
    assert min(c.min_volume_ratio for c in grid) == 1.10

def test_under_100_filter_remains():
    assert all(c.max_entry_price == 100.0 for c in mod.config_grid())

def test_compounding_logic_remains():
    import inspect
    src = inspect.getsource(mod.backtest)
    assert "sizing_equity = equity_now if compound else start_capital" in src

def test_json_default_handles_numpy():
    assert mod.json_default(np.int64(3)) == 3
    assert mod.json_default(np.float64(1.5)) == 1.5

def test_cost_stress_parameter_exists():
    import inspect
    assert "cost_mult" in inspect.signature(mod.backtest).parameters

def test_dynamic_market_leader_filter_is_used():
    import inspect
    src = inspect.getsource(mod.prepare_events)
    assert "dynamic_leaders" in src
    assert "symbol in dynamic_leaders" in src
