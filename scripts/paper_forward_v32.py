#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import research_equity_rotation_v32 as v32

base = v32.base

FORWARD_VERSION = "V32_PAPER_FORWARD"
FORWARD_START = pd.Timestamp("2026-08-24", tz="UTC")
OUT_DIR = Path("V32_paper_forward_snapshot")
STARTING_EQUITY = float(getattr(base, "INITIAL_EQUITY", 100000.0))

# Frozen V26 rules copied from scripts/paper_v26.py at blob
# bc703bad87bdf5a13ab3c66c0ec1a8c68f3927e7.
V26_SOURCE_BLOB = "bc703bad87bdf5a13ab3c66c0ec1a8c68f3927e7"
V26_UNIVERSE = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "GLD", "TLT")
V26_MOMENTUM_DAYS = 63
V26_SMA_DAYS = 150
V26_REBALANCE_EVERY = 5
V26_TOP_WEIGHTS = (0.70, 0.30)

# Frozen V32 winner from run 32563109224 / commit 1ba06af.
V32_CONFIG = base.EquityConfig(
    momentum_days=126,
    sma_days=200,
    rebalance_days=20,
    top_k=4,
    weighting="equal",
    risk_profile="drawdown_guard",
    vol_target_pct=14.0,
    cost_bps=10.0,
    max_liquid_assets=24,
)


def _metrics_from_result(result, start, end):
    return base.segment_metrics(result, start, end)


def _find_forward_start_index(m):
    dates = pd.DatetimeIndex(m.dates)
    idx = np.flatnonzero(dates >= FORWARD_START)
    return int(idx[0]) if idx.size else None


def _v26_target_weights(m, signal_i):
    weights = np.zeros(len(m.symbols), dtype=float)
    if signal_i < V26_SMA_DAYS:
        return weights, []

    ranked = []
    for symbol in V26_UNIVERSE:
        if symbol not in m.symbols:
            continue
        j = m.symbols.index(symbol)
        close = float(m.closes[signal_i, j])
        past_i = signal_i - V26_MOMENTUM_DAYS
        if past_i < 0:
            continue
        past = float(m.closes[past_i, j])
        sma_window = np.asarray(
            m.closes[signal_i - V26_SMA_DAYS + 1:signal_i + 1, j],
            dtype=float,
        )
        valid = sma_window[np.isfinite(sma_window) & (sma_window > 0)]
        if (
            not np.isfinite(close)
            or close <= 0
            or not np.isfinite(past)
            or past <= 0
            or len(valid) < V26_SMA_DAYS
        ):
            continue
        momentum = close / past - 1.0
        sma150 = float(valid.mean())
        if momentum > 0.0 and close > sma150:
            ranked.append((momentum, j, symbol, close, sma150))

    ranked.sort(reverse=True, key=lambda x: x[0])
    chosen = ranked[:2]
    if chosen:
        weights[chosen[0][1]] = V26_TOP_WEIGHTS[0]
    if len(chosen) >= 2:
        weights[chosen[1][1]] = V26_TOP_WEIGHTS[1]

    rows = [
        {
            "symbol": symbol,
            "weight": float(weights[j]),
            "momentum_63": float(momentum),
            "close": float(close),
            "sma150": float(sma150),
        }
        for momentum, j, symbol, close, sma150 in chosen
    ]
    return weights, rows


def run_v26_shadow(m, trade_start_index, cost_bps=10.0):
    dates = m.dates
    n, k = m.closes.shape
    cash = float(STARTING_EQUITY)
    qty = np.zeros(k, dtype=float)
    rows = []
    cycles = []
    last_rebalance_i = None
    cycle_start_equity = None
    cycle_start_date = None
    cycle_active = False
    turnover = 0.0
    rebalances = 0
    cost_rate = float(cost_bps) / 10000.0

    for i in range(trade_start_index, n):
        date = dates[i]
        op = np.where(
            np.isfinite(m.valuation_opens[i]) & (m.valuation_opens[i] > 0),
            m.valuation_opens[i],
            0.0,
        )
        cp = np.where(
            np.isfinite(m.valuation_closes[i]) & (m.valuation_closes[i] > 0),
            m.valuation_closes[i],
            op,
        )
        pre_equity = cash + float(np.dot(qty, op))
        due = last_rebalance_i is None or (i - last_rebalance_i) >= V26_REBALANCE_EVERY
        traded_notional = 0.0
        target_exposure = float(np.dot((qty > 1e-12).astype(float), np.zeros(k)))

        if due:
            signal_i = i - 1
            weights, _ = _v26_target_weights(m, signal_i)

            if cycle_start_equity is not None and cycle_start_equity > 0:
                cycles.append(
                    {
                        "start_date": cycle_start_date,
                        "end_date": date,
                        "return": pre_equity / cycle_start_equity - 1.0,
                        "pnl": pre_equity - cycle_start_equity,
                        "active": bool(cycle_active),
                    }
                )

            desired = pre_equity * weights
            current = qty * op
            delta = desired - current
            tradable = np.isfinite(m.opens[i]) & (m.opens[i] > 0)

            for j in np.flatnonzero((delta < -1e-9) & tradable & (qty > 0)):
                price = float(m.opens[i, j])
                value = min(float(current[j]), float(-delta[j]))
                sell_qty = min(float(qty[j]), value / price)
                actual = sell_qty * price
                fee = actual * cost_rate
                qty[j] = max(0.0, qty[j] - sell_qty)
                cash += actual - fee
                traded_notional += actual

            current = qty * op
            delta = desired - current
            buys = np.flatnonzero((delta > 1e-9) & tradable)
            if buys.size:
                buys = buys[np.argsort(desired[buys])[::-1]]

            for j in buys:
                price = float(m.opens[i, j])
                max_buy = cash / (1.0 + cost_rate)
                value = max(0.0, min(float(delta[j]), max_buy))
                if value <= 0:
                    continue
                fee = value * cost_rate
                qty[j] += value / price
                cash -= value + fee
                traded_notional += value

            post_equity = cash + float(np.dot(qty, op))
            cycle_start_equity = post_equity
            cycle_start_date = date
            cycle_active = bool(np.any(qty > 1e-12))
            last_rebalance_i = i
            rebalances += 1
            target_exposure = float(weights.sum())

        close_equity = cash + float(np.dot(qty, cp))
        rows.append(
            {
                "date": date,
                "equity": close_equity,
                "cash": cash,
                "positions": int(np.sum(qty > 1e-12)),
                "rebalanced": bool(due),
                "target_exposure": target_exposure,
                "turnover_notional": traded_notional,
            }
        )
        turnover += traded_notional

    if cycle_start_equity is not None and cycle_start_equity > 0 and rows:
        final_equity = float(rows[-1]["equity"])
        cycles.append(
            {
                "start_date": cycle_start_date,
                "end_date": pd.Timestamp(rows[-1]["date"]),
                "return": final_equity / cycle_start_equity - 1.0,
                "pnl": final_equity - cycle_start_equity,
                "active": bool(cycle_active),
            }
        )

    return base.BacktestResult(
        equity=pd.DataFrame(rows),
        cycles=pd.DataFrame(cycles),
        turnover=float(turnover),
        rebalance_count=int(rebalances),
        initial_equity=float(STARTING_EQUITY),
        evaluation_start=pd.Timestamp(dates[trade_start_index]),
    )


def latest_v32_targets(m):
    i = len(m.dates) - 1
    w, regime, mode, vol, exposure = v32.target_weights_v32(m, i, V32_CONFIG)
    rows = [
        {"symbol": m.symbols[j], "weight": float(x)}
        for j, x in enumerate(w)
        if float(x) > 1e-12
    ]
    rows.sort(key=lambda r: r["weight"], reverse=True)
    return {
        "signal_session": pd.Timestamp(m.dates[i]).isoformat(),
        "regime": str(regime),
        "mode": str(mode),
        "spy_realized_vol_pct": None if not np.isfinite(vol) else float(vol),
        "target_exposure": float(exposure),
        "targets": rows,
    }


def latest_v26_targets(m):
    i = len(m.dates) - 1
    w, rows = _v26_target_weights(m, i)
    return {
        "signal_session": pd.Timestamp(m.dates[i]).isoformat(),
        "target_exposure": float(w.sum()),
        "targets": rows,
    }


def _safe_metric_summary(result, start, end):
    if result.equity.empty:
        return {
            "return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "cycles": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
        }
    return _metrics_from_result(result, start, end)


def main():
    print("V32 PAPER-FORWARD SHADOW VALIDATION — NO ORDERS")

    api = base.AlpacaEquityData()
    active = api.validate_assets(base.UNIVERSE)
    frames = api.daily_bars(active, start=base.DEFAULT_START)
    usable = {s: f for s, f in frames.items() if len(f) >= 300}

    missing_v26 = [s for s in V26_UNIVERSE if s not in usable]
    if "SPY" not in usable or missing_v26:
        raise SystemExit(f"Missing required forward symbols: {missing_v26}")

    m = base.build_matrices(usable)
    start_i = _find_forward_start_index(m)

    OUT_DIR.mkdir(exist_ok=True)

    base_summary = {
        "validator": FORWARD_VERSION,
        "orders_placed": False,
        "forward_start_session": FORWARD_START.date().isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v26_frozen_source_blob": V26_SOURCE_BLOB,
        "v32_frozen_config": asdict(V32_CONFIG),
        "latest_v26_signal": latest_v26_targets(m),
        "latest_v32_signal": latest_v32_targets(m),
    }

    if start_i is None:
        summary = {
            **base_summary,
            "status": "WAITING_FOR_FIRST_FORWARD_SESSION",
            "sessions_observed": 0,
            "message":
                "Forward validation begins with the first completed session on or after 2026-08-24.",
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary, indent=2, default=str))
        print("NO ORDERS WERE PLACED.")
        return 0

    v32_result = v32.run_backtest_v32(m, V32_CONFIG, trade_start_index=start_i)
    v26_result = run_v26_shadow(m, start_i, cost_bps=10.0)

    start = pd.Timestamp(m.dates[start_i])
    end = pd.Timestamp(m.dates[-1])

    v32_metrics = _safe_metric_summary(v32_result, start, end)
    v26_metrics = _safe_metric_summary(v26_result, start, end)
    spy_metrics = base.benchmark_spy(m, start_i)

    v32_result.equity.to_csv(OUT_DIR / "v32_equity_curve.csv", index=False)
    v26_result.equity.to_csv(OUT_DIR / "v26_equity_curve.csv", index=False)
    v32_result.cycles.to_csv(OUT_DIR / "v32_cycles.csv", index=False)
    v26_result.cycles.to_csv(OUT_DIR / "v26_cycles.csv", index=False)

    sessions_observed = int(len(v32_result.equity))
    summary = {
        **base_summary,
        "status":
            "COLLECTING_FORWARD_EVIDENCE"
            if sessions_observed < 60
            else "FORWARD_REVIEW_DUE",
        "sessions_observed": sessions_observed,
        "evaluation_start": start.isoformat(),
        "evaluation_end": end.isoformat(),
        "v32": v32_metrics,
        "v26": v26_metrics,
        "spy": spy_metrics,
        "comparison": {
            "v32_minus_v26_return_pct":
                float(v32_metrics["return_pct"] - v26_metrics["return_pct"]),
            "v32_minus_spy_return_pct":
                float(v32_metrics["return_pct"] - spy_metrics["return_pct"]),
            "v32_minus_v26_sharpe":
                float(v32_metrics["sharpe"] - v26_metrics["sharpe"]),
            "v32_minus_spy_sharpe":
                float(v32_metrics["sharpe"] - spy_metrics["sharpe"]),
        },
        "review_policy": {
            "minimum_sessions_before_primary_review": 60,
            "rules_frozen_during_forward_window": True,
            "live_trading_authorized": False,
        },
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print("NO ORDERS WERE PLACED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
