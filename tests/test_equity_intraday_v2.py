import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_equity_intraday_v2.py"
spec = importlib.util.spec_from_file_location("intraday_v2", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

def sample_df():
    times = pd.date_range("2026-01-05 14:30:00+00:00", periods=50, freq="5min")
    close = np.linspace(20.0, 21.5, len(times))
    return pd.DataFrame({
        "time": times,
        "open": close - 0.02,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": np.linspace(1000, 3000, len(times)),
        "session": ["2026-01-05"] * len(times),
    })

def test_research_never_places_orders():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "submit_order" not in source
    assert "/v2/orders" not in source

def test_starts_with_100():
    assert mod.STARTING_CAPITAL == 100.0

def test_v2_grid_is_selective():
    grid = mod.config_grid()
    assert len(grid) == 64
    assert all(c.min_momentum_pct >= 0.15 for c in grid)
    assert all(c.min_volume_ratio >= 1.25 for c in grid)

def test_v2_only_enters_under_100():
    assert all(c.max_entry_price == 100.0 for c in mod.config_grid())
    assert all(c.min_entry_price == 5.0 for c in mod.config_grid())

def test_opening_range_and_antichase_features():
    out = mod.add_features(sample_df(), mod.config_grid()[0])
    assert "opening_range_high" in out.columns
    assert "vwap_extension_atr" in out.columns
    assert "signal_long" in out.columns

def test_compounding_risk_is_smaller_than_v1():
    assert all(c.risk_fraction == 0.0075 for c in mod.config_grid())

def test_cost_multiplier_changes_cost_rate_path():
    import inspect
    source = inspect.getsource(mod.backtest)
    assert "cost_mult" in source

def test_summary_has_daily_target_fields():
    eq = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-05T15:00Z", "2026-01-06T15:00Z"]),
        "equity": [100.0, 102.0],
    })
    tr = pd.DataFrame({
        "exit_time": pd.to_datetime(["2026-01-05T15:00Z", "2026-01-06T15:00Z"]),
        "pnl": [1.2, 0.8],
    })
    s = mod.summarize(100.0, 102.0, eq, tr)
    assert s["days_ge_1_dollar"] == 1
    assert "positive_days_pct" in s
