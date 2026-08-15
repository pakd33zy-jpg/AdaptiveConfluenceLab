from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "paper_v26.py"
spec = importlib.util.spec_from_file_location("paper_v26", MODULE_PATH)
v26 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v26
spec.loader.exec_module(v26)


def _bars(symbol: str, drift: float, n: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=n)
    close = pd.Series([100.0 * ((1.0 + drift) ** i) for i in range(n)])
    return pd.DataFrame(
        {
            "session": dates.date.astype(str),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_compute_targets_picks_top_two_eligible():
    drifts = {
        "SPY": 0.0010,
        "QQQ": 0.0020,
        "IWM": 0.0007,
        "DIA": 0.0006,
        "XLK": 0.0030,
        "XLF": 0.0005,
        "XLE": -0.0010,
        "GLD": 0.0004,
        "TLT": -0.0005,
    }
    bars = {s: _bars(s, d) for s, d in drifts.items()}
    session, targets, rows = v26.compute_targets(bars)
    assert session
    assert targets == {"XLK": 0.70, "QQQ": 0.30}
    assert rows[0].symbol == "XLK"


def test_single_eligible_keeps_30_percent_cash():
    bars = {s: _bars(s, -0.0010) for s in v26.UNIVERSE}
    bars["SPY"] = _bars("SPY", 0.0020)
    _, targets, _ = v26.compute_targets(bars)
    assert targets == {"SPY": 0.70}
    assert sum(targets.values()) == 0.70


def test_rebalance_due_after_five_completed_sessions():
    sessions = [f"2026-08-{d:02d}" for d in range(3, 13)]
    due, elapsed = v26.rebalance_due(sessions, "2026-08-05", every=5)
    assert due is True
    assert elapsed == 7

    due2, elapsed2 = v26.rebalance_due(sessions[:6], "2026-08-05", every=5)
    assert due2 is False
    assert elapsed2 == 3


def test_build_orders_sells_before_buys():
    positions = [
        {
            "symbol": "SPY",
            "market_value": "50000",
            "qty": "100",
            "current_price": "500",
        }
    ]
    orders = v26.build_orders(
        session="2026-08-14",
        targets={"QQQ": 0.70, "XLK": 0.30},
        positions=positions,
        account_equity=100000.0,
    )
    assert orders[0]["side"] == "sell"
    assert orders[0]["symbol"] == "SPY"
    assert {o["symbol"] for o in orders[1:]} == {"QQQ", "XLK"}
    assert all(o["time_in_force"] == "day" for o in orders)
