import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_equity_intraday_v4.py"
spec = importlib.util.spec_from_file_location("intraday_v4", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

def sample_df():
    times = pd.date_range("2026-01-05 14:30:00+00:00", periods=50, freq="5min")
    close = np.linspace(20.0, 22.0, len(times))
    return pd.DataFrame({
        "time": times,
        "open": close - 0.02,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": np.linspace(100000, 400000, len(times)),
        "session": ["2026-01-05"] * len(times),
    })

def test_no_order_submission():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "submit_order" not in source
    assert "/v2/orders" not in source

def test_starting_capital_100():
    assert mod.STARTING_CAPITAL == 100.0

def test_score_threshold_is_in_config():
    c = mod.config_grid()[0]
    assert hasattr(c, "min_setup_score")
    assert 50 <= c.min_setup_score <= 90

def test_grid_uses_multiple_score_thresholds():
    vals = {c.min_setup_score for c in mod.config_grid()}
    assert vals == {62.0, 68.0, 74.0}

def test_feature_score_exists():
    out = mod.add_features(sample_df(), mod.config_grid()[0])
    assert "setup_score" in out.columns
    assert "score_trend" in out.columns
    assert "score_volume" in out.columns
    assert out["setup_score"].dropna().between(0, 100).all()

def test_soft_scoring_replaces_hard_market_gate():
    import inspect
    src = inspect.getsource(mod.prepare_events)
    assert "leader_bonus" in src
    assert "market_bonus" in src
    assert "total_score" in src

def test_entry_uses_score_cutoff():
    import inspect
    src = inspect.getsource(mod.backtest)
    assert 'setup_score' in src
    assert 'min_setup_score' in src

def test_under_100_filter_remains():
    assert all(c.max_entry_price == 100.0 for c in mod.config_grid())

def test_compounding_remains():
    import inspect
    src = inspect.getsource(mod.backtest)
    assert "sizing_equity = equity_now if compound else start_capital" in src

def test_dynamic_leaders_remain():
    assert hasattr(mod, "build_dynamic_leader_map")

def test_cost_stress_remains():
    import inspect
    assert "cost_mult" in inspect.signature(mod.backtest).parameters
